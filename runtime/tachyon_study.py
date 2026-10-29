#!/usr/bin/env python3
"""Durable serial controller for the three-model Tachyon CPU screening.

The controller deliberately owns only processes that have a durable launch
receipt.  Remote process launches are performed by Python ``subprocess`` with
an argv array; SSH is only the transport.  A failed provenance or ownership
check is terminal and is recorded in the study status before this program
returns nonzero.
"""

from __future__ import annotations

import argparse
import base64
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STUDY = ROOT / "results/tachyon-study-20261023-v1"
ADAPTER = ROOT / "runtime/tachyon_benchmark.py"
FROZEN_RUNNER = (
    ROOT / "results/public-benchmark-pilot-20261023-v1/artifacts/public_benchmark_pilot.py"
)
INITIAL_ADAPTER_ARCHIVE = (
    DEFAULT_STUDY / "artifacts/tachyon_benchmark-initial-99222448.py"
)
PYTHON = ROOT / ".venv-public-bench/bin/python"
VENDOR = ROOT / "vendor/public-benchmarks"
EVALPLUS = VENDOR / "evalplus-v0.3.1"
IFEVAL = (
    VENDOR
    / "google-research-e6890f85757dd84e27ca6df2dd30651dafad28e0"
    / "instruction_following_eval"
)
REMOTE_ROOT = "/external/device-workspace"
REMOTE_RUNTIME = REMOTE_ROOT + "/runtime/llama-cpp-cpu"
REMOTE_RESULTS = REMOTE_ROOT + "/results/tachyon-study-20261023-v1"
ENDPOINT = "http://127.0.0.1:18185/v1"
GRADER_IMAGE = "sha256:85a1ac1c9042d3f2e6066f1a58bdda1ec9951832a6f3d1e4fa6a87fcd8c6efd7"
MANIFEST_SHA256 = "da6a70653695d317ac131d99eb09fe772cb8191c41d467345e9e3f4c11e90caf"
FROZEN_RUNNER_SHA256 = "a661ac74f414769b80a2484e072924de75bc2935ee470347d7244797dbc945dc"
INITIAL_RUNNER_PID = 2_839_731
INITIAL_SERVER_PID = 3_107_372
INITIAL_ADAPTER_SHA256 = "99222448b63687d6f242b714ae1930b1f583cd245fee8bd8bebed870cd0112d6"
TERMINAL_QUALITY = frozenset(("completed", "truncated"))
SSH_OPTIONS = (
    "-o", "BatchMode=yes",
    "-o", "StrictHostKeyChecking=yes",
    "-o", "ClearAllForwardings=yes",
    "-o", "ForwardAgent=no",
    "-o", "RequestTTY=no",
)
SMOKE_PROMPT = (
    'Return only a JSON object with keys "answer" and "explanation". '
    "A sensor records 18, 22, 20, 24 degrees. What is the average? "
    "Keep the explanation to one sentence."
)
MODELS: tuple[dict[str, Any], ...] = (
    {
        "key": "bonsai2-2b",
        "file": "Ternary-Bonsai-2-2B-Q2_0.gguf",
        "size": 546_689_504,
        "sha256": "c2893fa6f951daee8b7263d5f49b455e57989ef56417a22125b03566dc560650",
        "profile": "bonsai-nothink",
    },
    {
        "key": "bonsai2-4b",
        "file": "Ternary-Bonsai-2-4B-Q2_0.gguf",
        "size": 1_212_103_520,
        "sha256": "64a5183b44921bdb5ac4a6f5f92007821d6ebe5c66997d3c147fd214ab9433d4",
        "profile": "bonsai-nothink",
    },
    {
        "key": "lfm25-26b",
        "file": "LFM2.5-2.6B-QAD-Q4_0.gguf",
        "size": 1_593_894_944,
        "sha256": "a247afd6414918eac8e520a9e6137dc271235461ecbe1180462221d5b8d40b03",
        "profile": "lfm26",
    },
)


class StudyError(RuntimeError):
    """A durable controller invariant failed."""


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise StudyError(f"invalid or unreadable JSON: {path}") from exc


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("xb") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def process_identity(pid: int) -> dict[str, Any]:
    """Return a local Linux process identity; zombies count as exited."""
    try:
        raw_stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        tail = raw_stat[raw_stat.rfind(")") + 2 :].split()
        state = tail[0]
        cmdline = [
            part.decode("utf-8", "surrogateescape")
            for part in Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
            if part
        ]
        return {
            "pid": pid,
            "alive": state != "Z",
            "state": state,
            "start_ticks": int(tail[19]),
            "argv": cmdline,
        }
    except (FileNotFoundError, ProcessLookupError):
        return {"pid": pid, "alive": False, "state": "exited", "argv": []}
    except (OSError, ValueError, IndexError) as exc:
        raise StudyError(f"cannot inspect local process {pid}: {exc}") from exc


def verify_adopted_runner(receipt: Mapping[str, Any], expected_argv: Sequence[str]) -> dict[str, Any]:
    """Fail closed if a purported local runner is not the recorded process."""
    pid = receipt.get("pid")
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        raise StudyError("quality runner receipt has an invalid PID")
    recorded = receipt.get("argv")
    if recorded != list(expected_argv):
        raise StudyError("adopted quality runner receipt argv differs from the frozen command")
    expected_source = sha256(ADAPTER)
    allowed_sources = {expected_source}
    if pid == INITIAL_RUNNER_PID:
        # This one process and receipt predate the controller and were
        # explicitly authorized. A later summary-only adapter edit cannot
        # change the source already loaded in that live interpreter.
        if (
            not INITIAL_ADAPTER_ARCHIVE.is_file()
            or sha256(INITIAL_ADAPTER_ARCHIVE) != INITIAL_ADAPTER_SHA256
        ):
            raise StudyError("archived initial adapter provenance is missing or changed")
        allowed_sources.add(INITIAL_ADAPTER_SHA256)
    if receipt.get("source_sha256") not in allowed_sources:
        raise StudyError("adopted quality runner source SHA-256 differs")
    identity = process_identity(pid)
    if identity["alive"]:
        if identity["argv"] != list(expected_argv):
            raise StudyError("adopted quality runner /proc argv mismatch")
        start_ticks = receipt.get("start_ticks")
        if start_ticks is not None and start_ticks != identity["start_ticks"]:
            raise StudyError("adopted quality runner start-tick identity mismatch")
    return identity


def validate_server_argv(argv: Sequence[str], model: Mapping[str, Any]) -> None:
    expected_exe = REMOTE_RUNTIME + "/bin/llama-server"
    expected_model = REMOTE_ROOT + f"/models/{model['key']}/{model['file']}"
    command = list(argv)
    if command[:3] == ["taskset", "-c", "4-7"]:
        command = command[3:]
    if not command or command[0] != expected_exe:
        raise StudyError("server executable is outside the pinned CPU runtime")
    required_pairs = {
        "-m": expected_model,
        "--alias": f"tachyon-{model['key']}",
        "--host": "127.0.0.1",
        "--port": "18085",
        "-c": "16384",
        "-np": "1",
        "-t": "4",
        "-tb": "4",
        "-b": "512",
        "-ub": "128",
        "-ngl": "0",
        "--reasoning-effort": "none",
        "--reasoning-format": "deepseek",
        "--threads-http": "2",
    }
    for flag, expected in required_pairs.items():
        try:
            actual = command[command.index(flag) + 1]
        except (ValueError, IndexError) as exc:
            raise StudyError(f"server argv lacks {flag} {expected}") from exc
        if actual != expected:
            raise StudyError(f"server argv has unexpected {flag}: {actual}")
    if "--jinja" not in command:
        raise StudyError("server argv lacks --jinja")


def verify_owned_process(
    receipt: Mapping[str, Any], identity: Mapping[str, Any], *, kind: str
) -> bool:
    """Verify PID, birth identity, and exact argv before treating it as owned."""
    if not identity.get("alive"):
        return False
    if receipt.get("pid") != identity.get("pid"):
        raise StudyError(f"{kind} PID differs from its ownership receipt")
    if receipt.get("start_ticks") != identity.get("start_ticks"):
        raise StudyError(f"{kind} start-tick identity mismatch")
    if receipt.get("argv") != identity.get("argv"):
        raise StudyError(f"{kind} cmdline differs from its ownership receipt")
    return True


def verify_model_receipt(model: Mapping[str, Any], observation: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the already-hashed artifact receipt and current file size."""
    artifact = observation.get("artifact")
    if not isinstance(artifact, dict):
        raise StudyError(f"{model['key']} artifact.json is not an object")
    expected_path = REMOTE_ROOT + f"/models/{model['key']}/{model['file']}"
    expected = {
        "key": model["key"],
        "file": model["file"],
        "size": model["size"],
        "sha256": model["sha256"],
        "actual_sha256": model["sha256"],
        "path": expected_path,
    }
    for field, value in expected.items():
        if artifact.get(field) != value:
            raise StudyError(
                f"{model['key']} artifact receipt {field} differs: {artifact.get(field)!r}"
            )
    if observation.get("path") != expected_path or observation.get("size") != model["size"]:
        raise StudyError(f"{model['key']} on-disk path or byte size differs")
    return {
        "verified_at": now(),
        "artifact": artifact,
        "observed_path": expected_path,
        "observed_size": observation["size"],
        "verification": "verified download receipt plus current file size; no redundant full rehash",
    }


def quality_command(
    python: str,
    study: Path,
    model: Mapping[str, Any],
    *,
    rerun_failed: bool = False,
) -> list[str]:
    command = [
        python,
        str(ADAPTER),
        "run",
        "--manifest",
        str(study / "manifest.json"),
        "--endpoint",
        ENDPOINT,
        "--model",
        f"tachyon-{model['key']}",
        "--model-key",
        str(model["key"]),
        "--profile",
        str(model["profile"]),
        "--output-dir",
        str(study / str(model["key"])),
        "--max-tokens",
        "8192",
        "--request-timeout",
        "3600",
        "--seed",
        "20261023",
    ]
    if rerun_failed:
        command.append("--rerun-failed")
    return command


def server_command(model: Mapping[str, Any]) -> list[str]:
    return [
        "taskset", "-c", "4-7",
        REMOTE_RUNTIME + "/bin/llama-server",
        "-m", REMOTE_ROOT + f"/models/{model['key']}/{model['file']}",
        "--alias", f"tachyon-{model['key']}",
        "--host", "127.0.0.1",
        "--port", "18085",
        "-c", "16384",
        "-np", "1",
        "-t", "4",
        "-tb", "4",
        "-b", "512",
        "-ub", "128",
        "-ngl", "0",
        "--jinja",
        "--reasoning-effort", "none",
        "--reasoning-format", "deepseek",
        "--threads-http", "2",
    ]


def bench_command(model: Mapping[str, Any]) -> list[str]:
    return [
        "taskset", "-c", "4-7",
        REMOTE_RUNTIME + "/bin/llama-bench",
        "-m", REMOTE_ROOT + f"/models/{model['key']}/{model['file']}",
        "-p", "128", "-n", "64", "-t", "4", "-r", "3", "-ngl", "0",
        "-o", "json",
    ]


def validate_bench_receipt(
    receipt: Mapping[str, Any], model: Mapping[str, Any]
) -> None:
    if receipt.get("kind") != "tachyon_llama_bench":
        raise StudyError("remote benchmark receipt kind differs")
    if receipt.get("launch_argv") != bench_command(model):
        raise StudyError("remote benchmark launch argv differs from the pinned command")
    if receipt.get("status") == "launching":
        supervisor = receipt.get("supervisor_pid")
        if isinstance(supervisor, bool) or not isinstance(supervisor, int) or supervisor <= 0:
            raise StudyError("remote benchmark launching receipt lacks supervisor PID")
        return
    argv = receipt.get("argv")
    expected_exe = REMOTE_RUNTIME + "/bin/llama-bench"
    expected_model = REMOTE_ROOT + f"/models/{model['key']}/{model['file']}"
    if not isinstance(argv, list) or not argv or argv[0] != expected_exe:
        raise StudyError("remote benchmark executable is outside the pinned CPU runtime")
    required = {"-m": expected_model, "-p": "128", "-n": "64", "-t": "4", "-r": "3", "-ngl": "0", "-o": "json"}
    for flag, expected in required.items():
        try:
            actual = argv[argv.index(flag) + 1]
        except (ValueError, IndexError) as exc:
            raise StudyError(f"remote benchmark argv lacks {flag} {expected}") from exc
        if actual != expected:
            raise StudyError(f"remote benchmark argv has unexpected {flag}: {actual}")


def valid_bench_output(path: Path) -> bool:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False
    return isinstance(value, (dict, list)) and bool(value)


def grading_command(study: Path, output: Path) -> list[str]:
    """Build the immutable, networkless official-grader command."""
    manifest = study / "manifest.json"
    return [
        "docker", "run", "--rm",
        "--name", f"tachyon-grade-{output.name}-{os.getpid()}",
        "--network", "none",
        "--read-only",
        "--cap-drop", "ALL",
        "--security-opt", "no-new-privileges",
        "--pids-limit", "256",
        "--cpus", "4",
        "--memory", "4g",
        "--user", f"{os.getuid()}:{os.getgid()}",
        "--tmpfs", "/tmp:rw,nosuid,nodev,size=2g",
        "--env", "HOME=/tmp",
        "--env", "XDG_CACHE_HOME=/tmp/cache",
        "--env", "PYTHONDONTWRITEBYTECODE=1",
        "--env", "OMP_NUM_THREADS=1",
        "--env", "OPENBLAS_NUM_THREADS=1",
        "--env", "TACHYON_GRADER_SANDBOX=1",
        "--mount", f"type=bind,src={ADAPTER},dst={ADAPTER},readonly",
        "--mount", f"type=bind,src={FROZEN_RUNNER},dst={FROZEN_RUNNER},readonly",
        "--mount", f"type=bind,src={manifest},dst={manifest},readonly",
        "--mount", f"type=bind,src={VENDOR},dst={VENDOR},readonly",
        "--mount", f"type=bind,src={output},dst={output}",
        GRADER_IMAGE,
        "python", str(ADAPTER), "grade",
        "--manifest", str(manifest),
        "--output-dir", str(output),
        "--evalplus-root", str(EVALPLUS),
        "--ifeval-root", str(IFEVAL),
    ]


def audit_quality(
    output: Path, model: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    status_path = output / "status.json"
    if not status_path.is_file():
        return {"present": False, "terminal": False, "terminal_count": 0}
    status = read_json(status_path)
    cases = status.get("cases") if isinstance(status, dict) else None
    counts = status.get("counts") if isinstance(status, dict) else None
    if not isinstance(cases, list) or len(cases) != 16 or not isinstance(counts, dict):
        raise StudyError(f"quality status has wrong case cardinality: {status_path}")
    if any(not isinstance(row, dict) for row in cases):
        raise StudyError(f"quality status has malformed case rows: {status_path}")
    terminal_count = sum(row.get("status") in TERMINAL_QUALITY for row in cases)
    terminal = terminal_count == 16
    if terminal and counts.get("completed", 0) + counts.get("truncated", 0) != 16:
        raise StudyError("quality terminal rows disagree with status counts")
    if model is not None:
        run = read_json(output / "run.json")
        settings = run.get("settings") if isinstance(run, dict) else None
        expected = {
            "model": f"tachyon-{model['key']}",
            "model_key": model["key"],
            "original_profile": model["profile"],
            "max_tokens": 8192,
            "endpoint": ENDPOINT,
            "seed": 20_261_023,
            "frozen_runner_sha256": FROZEN_RUNNER_SHA256,
        }
        if not isinstance(settings, dict):
            raise StudyError(f"quality run binding is missing: {output / 'run.json'}")
        for field, value in expected.items():
            if settings.get(field) != value:
                raise StudyError(
                    f"quality run binding differs for {model['key']} field {field}"
                )
        manifest = run.get("manifest")
        if not isinstance(manifest, dict) or manifest.get("sha256") != MANIFEST_SHA256:
            raise StudyError(f"quality run manifest binding differs for {model['key']}")
    return {
        "present": True,
        "terminal": terminal,
        "terminal_count": terminal_count,
        "counts": counts,
        "active": status.get("active"),
    }


def valid_grading(output: Path, model: Mapping[str, Any] | None = None) -> bool:
    path = output / "grading/summary.json"
    if not path.is_file():
        return False
    value = read_json(path)
    coverage = value.get("coverage") if isinstance(value, dict) else None
    run_path = output / "run.json"
    run = read_json(run_path) if run_path.is_file() else None
    settings = run.get("settings") if isinstance(run, dict) else None
    evalplus = value.get("evalplus") if isinstance(value, dict) else None
    ifeval = value.get("ifeval") if isinstance(value, dict) else None
    return bool(
        isinstance(coverage, dict)
        and isinstance(run, dict)
        and isinstance(settings, dict)
        and value.get("status") == "completed"
        and value.get("manifest_sha256") == MANIFEST_SHA256
        and value.get("settings_sha256") == run.get("settings_sha256")
        and coverage.get("total") == 16
        and coverage.get("completed", 0) + coverage.get("truncated", 0) == 16
        and isinstance(evalplus, dict)
        and evalplus.get("status") == "completed"
        and evalplus.get("task_count") == 8
        and isinstance(ifeval, dict)
        and ifeval.get("status") == "completed"
        and (model is None or settings.get("model_key") == model["key"])
    )


def _remote_process_script() -> str:
    return r'''import json, pathlib, sys
p=json.loads(sys.argv[1]); pid=int(p["pid"]); base=pathlib.Path("/proc")/str(pid)
try:
 s=(base/"stat").read_text(); tail=s[s.rfind(")")+2:].split(); state=tail[0]
 argv=[x.decode("utf-8","surrogateescape") for x in (base/"cmdline").read_bytes().split(b"\0") if x]
 print(json.dumps({"pid":pid,"alive":state!="Z","state":state,"start_ticks":int(tail[19]),"argv":argv}))
except FileNotFoundError: print(json.dumps({"pid":pid,"alive":False,"state":"exited","argv":[]}))'''


class Controller:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.study = args.study.resolve()
        self.python = str(Path(args.python).resolve())
        self.state_path = self.study / "study-status.json"
        self.events_path = self.study / "events.jsonl"
        self.lock_stream: io.TextIOWrapper | None = None
        self.local_children: dict[int, subprocess.Popen[Any]] = {}
        self.interrupted: int | None = None
        self.preexisting_complete = False
        self.state: dict[str, Any] = {}

    def acquire_lock(self) -> None:
        self.study.mkdir(parents=True, exist_ok=True)
        self.lock_stream = (self.study / ".controller.lock").open("a+")
        try:
            fcntl.flock(self.lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise StudyError("another Tachyon study controller holds the lock") from exc
        self.lock_stream.seek(0)
        self.lock_stream.truncate()
        self.lock_stream.write(f"{os.getpid()}\n")
        self.lock_stream.flush()

    def event(self, name: str, **fields: Any) -> None:
        row = {"at": now(), "event": name, **fields}
        append_jsonl(self.events_path, row)
        print(json.dumps(row, sort_keys=True), flush=True)

    def save(self, **fields: Any) -> None:
        self.state.update(fields)
        self.state["controller_pid"] = os.getpid()
        self.state["updated_at"] = now()
        atomic_json(self.state_path, self.state)

    def check_interrupt(self) -> None:
        if self.interrupted is not None:
            raise StudyError(f"controller interrupted by signal {self.interrupted}")

    def _handle_signal(self, signum: int, _frame: Any) -> None:
        self.interrupted = signum

    def source_pins(self) -> dict[str, str]:
        paths = {
            Path(__file__).resolve(),
            ADAPTER,
            FROZEN_RUNNER,
            INITIAL_ADAPTER_ARCHIVE,
            self.study / "manifest.json",
            VENDOR / "datasets/HumanEvalPlus-v0.1.10.jsonl",
        }
        # The immutable container supplies dependencies, while these mounted
        # Python trees supply the official grader implementations.
        paths.update((EVALPLUS / "evalplus").rglob("*.py"))
        paths.update(IFEVAL.rglob("*.py"))
        return {str(path): sha256(path) for path in sorted(paths)}

    def initialize(self) -> None:
        manifest_path = self.study / "manifest.json"
        if sha256(manifest_path) != MANIFEST_SHA256:
            raise StudyError("frozen Tachyon manifest SHA-256 mismatch")
        manifest = read_json(manifest_path)
        cases = manifest.get("cases") if isinstance(manifest, dict) else None
        if (
            not isinstance(cases, list)
            or len(cases) != 16
            or manifest.get("selected_count_per_suite") != 8
            or sum(row.get("suite") == "humanevalplus" for row in cases) != 8
            or sum(row.get("suite") == "ifeval" for row in cases) != 8
        ):
            raise StudyError("controller requires the frozen 8+8 Tachyon manifest")
        if sha256(FROZEN_RUNNER) != FROZEN_RUNNER_SHA256:
            raise StudyError("frozen public runner SHA-256 mismatch")
        pins = self.source_pins()
        pin_path = self.study / "source-pins.json"
        if pin_path.exists():
            recorded = read_json(pin_path)
            if not isinstance(recorded, dict) or recorded.get("sha256") != pins:
                raise StudyError("source pins changed since the controller was initialized")
        else:
            atomic_json(pin_path, {"schema_version": 1, "created_at": now(), "sha256": pins})
        if self.state_path.exists():
            state = read_json(self.state_path)
            if not isinstance(state, dict):
                raise StudyError("study status is not an object")
            if state.get("manifest_sha256") != MANIFEST_SHA256:
                raise StudyError("study status manifest binding differs")
            if state.get("model_order") != [row["key"] for row in MODELS]:
                raise StudyError("study status model order differs")
            self.state = state
        else:
            self.state = {
                "schema_version": 1,
                "kind": "tachyon_cpu_public_screening",
                "created_at": now(),
                "status": "starting",
                "study": str(self.study),
                "manifest_sha256": MANIFEST_SHA256,
                "model_order": [row["key"] for row in MODELS],
                "models": {},
                "controls": {
                    "serial": True,
                    "context_tokens": 16_384,
                    "max_output_tokens": 8_192,
                    "quality_cases_per_model": 16,
                    "remote_server_port": 18_085,
                    "local_tunnel_port": 18_185,
                    "server_threads": 4,
                    "benchmark_cpu_set": "4-7",
                    "benchmark_prompt_tokens": 128,
                    "benchmark_generation_tokens": 64,
                    "benchmark_repetitions": 3,
                },
            }
        self.preexisting_complete = self.state.get("status") == "complete"
        self.state["source_pins"] = pins
        self.save(status="complete" if self.preexisting_complete else "running", error=None)

    def remote_json(self, script: str, payload: Mapping[str, Any], *, timeout: float = 90) -> Any:
        encoded = json.dumps(payload, separators=(",", ":"))
        remote_command = shlex.join(["python3", "-c", script, encoded])
        completed = subprocess.run(
            ["ssh", *SSH_OPTIONS, "-T", self.args.remote, remote_command],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
            check=False,
        )
        if completed.returncode:
            detail = completed.stderr.strip()[-1000:]
            raise StudyError(f"remote command failed ({completed.returncode}): {detail}")
        try:
            return json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise StudyError("remote command returned invalid JSON") from exc

    def remote_process(self, pid: int) -> dict[str, Any]:
        value = self.remote_json(_remote_process_script(), {"pid": pid})
        if not isinstance(value, dict):
            raise StudyError("remote process observation is not an object")
        return value

    def observe_model(self, model: Mapping[str, Any]) -> dict[str, Any]:
        script = r'''import json, pathlib, sys
p=json.loads(sys.argv[1]); receipt=pathlib.Path(p["receipt"]); model=pathlib.Path(p["model"])
try: artifact=json.loads(receipt.read_text()); st=model.stat()
except Exception as e: print(json.dumps({"error":str(e)})); raise SystemExit(0)
print(json.dumps({"artifact":artifact,"path":str(model),"size":st.st_size,"regular":model.is_file()}))'''
        path = REMOTE_ROOT + f"/models/{model['key']}/{model['file']}"
        observed = self.remote_json(
            script,
            {"receipt": REMOTE_ROOT + f"/models/{model['key']}/artifact.json", "model": path},
        )
        if not isinstance(observed, dict) or observed.get("error") or not observed.get("regular"):
            raise StudyError(f"remote model observation failed for {model['key']}: {observed}")
        return verify_model_receipt(model, observed)

    def read_remote_json_file(self, path: str) -> dict[str, Any] | None:
        script = r'''import json, pathlib, sys
p=pathlib.Path(json.loads(sys.argv[1])["path"])
if not p.is_file(): print("null")
else:
 try: print(json.dumps(json.loads(p.read_text())))
 except Exception as e: print(json.dumps({"_read_error":str(e)}))'''
        value = self.remote_json(script, {"path": path})
        if value is None:
            return None
        if not isinstance(value, dict) or value.get("_read_error"):
            raise StudyError(f"invalid remote receipt: {path}")
        return value

    def remote_file(self, path: str) -> bytes:
        script = r'''import base64, json, pathlib, sys
p=pathlib.Path(json.loads(sys.argv[1])["path"])
print(json.dumps({"data":base64.b64encode(p.read_bytes()).decode("ascii")}))'''
        value = self.remote_json(script, {"path": path}, timeout=180)
        try:
            return base64.b64decode(value["data"], validate=True)
        except (KeyError, ValueError, TypeError) as exc:
            raise StudyError(f"failed to retrieve remote file: {path}") from exc

    def append_launch(self, name: str, receipt: Mapping[str, Any]) -> None:
        append_jsonl(self.study / f"{name}-launches.jsonl", dict(receipt))

    def launch_server(self, model: Mapping[str, Any]) -> dict[str, Any]:
        argv = server_command(model)
        validate_server_argv(argv, model)
        remote_receipt = REMOTE_RESULTS + f"/{model['key']}-server-launch.json"
        script = r'''import json, os, pathlib, subprocess, sys, time
p=json.loads(sys.argv[1]); log=pathlib.Path(p["log"]); rec=pathlib.Path(p["receipt"])
log.parent.mkdir(parents=True,exist_ok=True); f=log.open("ab",buffering=0)
child=subprocess.Popen(p["argv"],stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
base=pathlib.Path("/proc")/str(child.pid)
deadline=time.monotonic()+2; argv=[]
while time.monotonic()<deadline:
 if not base.exists(): raise SystemExit("server exited during launch")
 argv=[x.decode("utf-8","surrogateescape") for x in (base/"cmdline").read_bytes().split(b"\0") if x]
 if argv and argv[0].endswith("/llama-server"): break
 time.sleep(.01)
if not argv or not argv[0].endswith("/llama-server"): raise SystemExit("server wrapper did not exec llama-server")
s=(base/"stat").read_text(); tail=s[s.rfind(")")+2:].split()
value={"schema_version":1,"kind":"tachyon_server","pid":child.pid,"start_ticks":int(tail[19]),"argv":argv,"launch_argv":p["argv"],"model_key":p["model_key"],"alias":p["alias"],"launched_at":p["at"],"log":str(log)}
tmp=rec.with_name(rec.name+".tmp-"+str(os.getpid())); tmp.write_text(json.dumps(value,sort_keys=True)+"\n"); os.replace(tmp,rec)
print(json.dumps(value))'''
        receipt = self.remote_json(
            script,
            {
                "argv": argv,
                "model_key": model["key"],
                "alias": f"tachyon-{model['key']}",
                "at": now(),
                "log": REMOTE_RESULTS + f"/{model['key']}-server.log",
                "receipt": remote_receipt,
            },
        )
        if not isinstance(receipt, dict):
            raise StudyError("remote server launch receipt is malformed")
        validate_server_argv(receipt.get("argv", []), model)
        identity = self.remote_process(receipt.get("pid", -1))
        verify_owned_process(receipt, identity, kind="remote server")
        receipt["remote_receipt"] = remote_receipt
        atomic_json(self.study / f"{model['key']}-server-launch.json", receipt)
        self.append_launch(f"{model['key']}-server", receipt)
        self.event("server_launched", model=model["key"], pid=receipt["pid"])
        return receipt

    def adopt_initial_server(self, model: Mapping[str, Any]) -> dict[str, Any] | None:
        if model["key"] != "bonsai2-2b":
            return None
        identity = self.remote_process(INITIAL_SERVER_PID)
        if not identity.get("alive"):
            return None
        validate_server_argv(identity.get("argv", []), model)
        receipt = {
            "schema_version": 1,
            "kind": "tachyon_server",
            "pid": identity["pid"],
            "start_ticks": identity["start_ticks"],
            "argv": identity["argv"],
            "model_key": model["key"],
            "alias": f"tachyon-{model['key']}",
            "adopted": True,
            "adopted_at": now(),
            "authorization": "pre-existing Tachyon 2B CPU server explicitly supplied to controller",
        }
        atomic_json(self.study / f"{model['key']}-server-launch.json", receipt)
        self.append_launch(f"{model['key']}-server", receipt)
        self.event("server_adopted", model=model["key"], pid=identity["pid"])
        return receipt

    def current_server(self, model: Mapping[str, Any]) -> dict[str, Any] | None:
        path = self.study / f"{model['key']}-server-launch.json"
        if not path.is_file():
            remote_path = REMOTE_RESULTS + f"/{model['key']}-server-launch.json"
            receipt = self.read_remote_json_file(remote_path)
            if receipt is None:
                return None
            recovered = True
        else:
            receipt = read_json(path)
            recovered = False
        if not isinstance(receipt, dict):
            raise StudyError("server ownership receipt is malformed")
        if (
            receipt.get("kind") != "tachyon_server"
            or receipt.get("model_key") != model["key"]
            or receipt.get("alias") != f"tachyon-{model['key']}"
        ):
            raise StudyError("server ownership receipt model identity differs")
        validate_server_argv(receipt.get("argv", []), model)
        identity = self.remote_process(receipt.get("pid", -1))
        if verify_owned_process(receipt, identity, kind="remote server"):
            if recovered:
                receipt["remote_receipt"] = remote_path
                atomic_json(path, receipt)
                self.append_launch(f"{model['key']}-server", receipt)
                self.event(
                    "server_receipt_recovered",
                    model=model["key"],
                    pid=receipt["pid"],
                )
            return receipt
        return None

    def ensure_server(self, model: Mapping[str, Any]) -> dict[str, Any]:
        receipt = self.current_server(model)
        if receipt is None:
            path = self.study / f"{model['key']}-server-launch.json"
            if not path.exists():
                receipt = self.adopt_initial_server(model)
            if receipt is None:
                receipt = self.launch_server(model)
        self.wait_endpoint_alias(f"tachyon-{model['key']}", receipt)
        return receipt

    def endpoint_models(self) -> list[str]:
        try:
            with urlopen(ENDPOINT + "/models", timeout=10) as response:
                payload = json.load(response)
        except (OSError, HTTPError, URLError, json.JSONDecodeError) as exc:
            raise StudyError(f"cannot query local Tachyon tunnel endpoint: {exc}") from exc
        rows = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(rows, list):
            raise StudyError("endpoint /models response lacks data")
        return [row["id"] for row in rows if isinstance(row, dict) and isinstance(row.get("id"), str)]

    def wait_endpoint_alias(self, alias: str, receipt: Mapping[str, Any]) -> None:
        deadline = time.monotonic() + 600
        latest = "not queried"
        while time.monotonic() < deadline:
            self.check_interrupt()
            identity = self.remote_process(receipt["pid"])
            verify_owned_process(receipt, identity, kind="remote server")
            try:
                models = self.endpoint_models()
                if models == [alias] or (alias in models and len(set(models)) == 1):
                    return
                latest = f"advertised models {models!r}"
            except StudyError as exc:
                latest = str(exc)
            time.sleep(2)
        raise StudyError(f"endpoint did not become ready with exact alias {alias}: {latest}")

    def smoke_paths(self, model: Mapping[str, Any]) -> tuple[Path, Path, Path]:
        prefix = self.study / f"{model['key']}-smoke"
        return (
            prefix.with_name(prefix.name + "-request.json"),
            prefix.with_name(prefix.name + "-response.json"),
            prefix.with_name(prefix.name + "-evaluation.json"),
        )

    def run_smoke(self, model: Mapping[str, Any]) -> None:
        request_path, response_path, evaluation_path = self.smoke_paths(model)
        present = [path.exists() for path in (request_path, response_path, evaluation_path)]
        if all(present):
            self.event("smoke_preserved", model=model["key"])
            return
        if any(present):
            raise StudyError(f"partial smoke evidence exists for {model['key']}; refusing overwrite")
        # Import only after the adapter's source pin is established.
        from runtime import tachyon_benchmark

        fields = tachyon_benchmark.native_recipe(model["profile"], model["key"])["request_fields"]
        payload = {
            "model": f"tachyon-{model['key']}",
            "messages": [{"role": "user", "content": SMOKE_PROMPT}],
            "max_tokens": 192,
            "seed": 20_261_023,
            **fields,
            "stream": True,
            "stream_options": {"include_usage": True},
            "timings_per_token": True,
        }
        atomic_json(request_path, payload)
        request = Request(
            ENDPOINT + "/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        chunks: list[dict[str, Any]] = []
        content: list[str] = []
        started = time.monotonic()
        ttft: float | None = None
        try:
            with urlopen(request, timeout=900) as response:
                for raw in response:
                    line = raw.decode("utf-8").strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        continue
                    row = json.loads(data)
                    chunks.append(row)
                    for choice in row.get("choices", []):
                        delta = choice.get("delta", {}) if isinstance(choice, dict) else {}
                        text = delta.get("content") if isinstance(delta, dict) else None
                        if isinstance(text, str):
                            if text and ttft is None:
                                ttft = time.monotonic() - started
                            content.append(text)
        except (OSError, HTTPError, URLError, json.JSONDecodeError) as exc:
            raise StudyError(f"smoke probe failed for {model['key']}: {exc}") from exc
        wall = time.monotonic() - started
        last = chunks[-1] if chunks else None
        response_value = {
            "wall_seconds": wall,
            "ttft_seconds": ttft,
            "content": "".join(content),
            "last": last,
            "chunks": chunks,
        }
        atomic_json(response_path, response_value)
        parsed: Any = None
        json_valid = False
        try:
            parsed = json.loads(response_value["content"])
            json_valid = isinstance(parsed, dict) and set(parsed) == {"answer", "explanation"}
        except json.JSONDecodeError:
            pass
        timings = last.get("timings", {}) if isinstance(last, dict) else {}
        usage = last.get("usage", {}) if isinstance(last, dict) else {}
        observed = parsed.get("answer") if json_valid else None
        atomic_json(
            evaluation_path,
            {
                "expected_answer": 21,
                "observed_content": response_value["content"],
                "completion_tokens": usage.get("completion_tokens"),
                "native_generation_tps": timings.get("predicted_per_second"),
                "wall_seconds": wall,
                "ttft_seconds": ttft,
                "json_valid": json_valid,
                "answer_correct": observed == 21,
            },
        )
        self.event("smoke_complete", model=model["key"], json_valid=json_valid, answer_correct=observed == 21)

    def quality_receipt(self, model: Mapping[str, Any]) -> Path:
        return self.study / f"{model['key']}-runner-launch.json"

    def launch_quality(
        self, model: Mapping[str, Any], *, rerun_failed: bool
    ) -> dict[str, Any]:
        argv = quality_command(
            self.python, self.study, model, rerun_failed=rerun_failed
        )
        log_path = self.study / f"{model['key']}-runner.log"
        log = log_path.open("ab", buffering=0)
        child = subprocess.Popen(
            argv,
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        self.local_children[child.pid] = child
        identity = process_identity(child.pid)
        if not identity["alive"]:
            raise StudyError(f"quality runner exited during launch for {model['key']}")
        receipt = {
            "at": now(),
            "pid": child.pid,
            "start_ticks": identity["start_ticks"],
            "argv": argv,
            "source_sha256": sha256(ADAPTER),
        }
        atomic_json(self.quality_receipt(model), receipt)
        self.append_launch(f"{model['key']}-runner", receipt)
        self.event("quality_launched", model=model["key"], pid=child.pid)
        return receipt

    def wait_quality(self, model: Mapping[str, Any]) -> dict[str, Any]:
        output = self.study / str(model["key"])
        audit = audit_quality(output, model)
        if audit["terminal"]:
            self.event("quality_preserved", model=model["key"], terminal_count=16)
            return audit
        path = self.quality_receipt(model)
        receipt: dict[str, Any] | None = None
        if path.is_file():
            value = read_json(path)
            if not isinstance(value, dict):
                raise StudyError("quality runner receipt is malformed")
            receipt = value
            allowed = (
                quality_command(self.python, self.study, model),
                quality_command(self.python, self.study, model, rerun_failed=True),
            )
            if receipt.get("argv") not in allowed:
                raise StudyError("quality runner receipt is not an allowed frozen command")
            identity = verify_adopted_runner(receipt, receipt["argv"])
            if not identity["alive"]:
                receipt = None
        if receipt is None:
            # Re-check after observing an exited process. Its final status write
            # can race the ownership observation, and terminal answers must
            # never be duplicated.
            audit = audit_quality(output, model)
            if audit["terminal"]:
                self.event("quality_preserved", model=model["key"], terminal_count=16)
                return audit
            receipt = self.launch_quality(model, rerun_failed=audit["present"])
        while True:
            self.check_interrupt()
            child = self.local_children.get(receipt["pid"])
            if child is not None:
                child.poll()  # reap an owned zombie before /proc inspection
            identity = verify_adopted_runner(receipt, receipt["argv"])
            if not identity["alive"]:
                break
            time.sleep(5)
        # Status replacement follows runner exit; allow its final atomic write to settle.
        time.sleep(0.2)
        audit = audit_quality(output, model)
        if not audit["terminal"]:
            raise StudyError(
                f"quality runner for {model['key']} exited with only {audit['terminal_count']}/16 terminal answers"
            )
        self.event("quality_complete", model=model["key"], counts=audit["counts"])
        return audit

    def grade(self, model: Mapping[str, Any]) -> None:
        output = self.study / str(model["key"])
        if valid_grading(output, model):
            self.event("grading_preserved", model=model["key"])
            return
        command = grading_command(self.study, output)
        receipt_path = self.study / f"{model['key']}-grading-command.json"
        atomic_json(
            receipt_path,
            {
                "at": now(),
                "argv": command,
                "image": GRADER_IMAGE,
                "isolation": "network none; read-only root and inputs; only current model output writable",
            },
        )
        with (self.study / f"{model['key']}-grading.log").open("ab", buffering=0) as log:
            completed = subprocess.run(
                command,
                cwd=ROOT,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=21_600,
                check=False,
            )
        if completed.returncode or not valid_grading(output, model):
            raise StudyError(f"official sandbox grading failed for {model['key']}: {completed.returncode}")
        self.event("grading_complete", model=model["key"])

    def snapshot(self, model: Mapping[str, Any], receipt: Mapping[str, Any]) -> None:
        path = self.study / f"{model['key']}-resource-snapshot.json"
        if path.exists():
            self.event("resource_snapshot_preserved", model=model["key"])
            return
        script = r'''import glob, json, pathlib, sys
p=json.loads(sys.argv[1]); pid=int(p["pid"]); base=pathlib.Path("/proc")/str(pid)
s=(base/"stat").read_text(); tail=s[s.rfind(")")+2:].split(); state=tail[0]
argv=[x.decode("utf-8","surrogateescape") for x in (base/"cmdline").read_bytes().split(b"\0") if x]
status={}
for line in (base/"status").read_text().splitlines():
 if ":" in line:
  k,v=line.split(":",1)
  if k in ("VmRSS","VmHWM","VmSwap"): status[k]=v.strip()
mem={}
for line in pathlib.Path("/proc/meminfo").read_text().splitlines():
 if ":" in line:
  k,v=line.split(":",1)
  if k in ("MemTotal","MemAvailable","SwapTotal","SwapFree"): mem[k]=v.strip()
thermal=[]
for temp in sorted(glob.glob("/sys/class/thermal/thermal_zone*/temp")):
 d=pathlib.Path(temp).parent
 try: thermal.append({"zone":d.name,"type":(d/"type").read_text().strip(),"millidegrees_c":int((d/"temp").read_text().strip())})
 except (OSError,ValueError): pass
cpufreq=[]
for policy in sorted(glob.glob("/sys/devices/system/cpu/cpufreq/policy*")):
 d=pathlib.Path(policy); row={"policy":d.name}
 for name in ("scaling_governor","scaling_cur_freq","scaling_max_freq","cpuinfo_max_freq"):
  try: row[name]=(d/name).read_text().strip()
  except OSError: row[name]=None
 cpufreq.append(row)
print(json.dumps({"pid":pid,"alive":state!="Z","state":state,"start_ticks":int(tail[19]),"argv":argv,"process_status":status,"meminfo":mem,"thermal":thermal,"cpufreq":cpufreq}))'''
        value = self.remote_json(script, {"pid": receipt["pid"]})
        if not isinstance(value, dict):
            raise StudyError("resource snapshot is malformed")
        verify_owned_process(receipt, value, kind="remote server snapshot")
        value.update({"captured_at": now(), "model_key": model["key"]})
        atomic_json(path, value)
        self.event("resource_snapshot_complete", model=model["key"])

    def stop_server(self, model: Mapping[str, Any], receipt: Mapping[str, Any]) -> None:
        identity = self.remote_process(receipt["pid"])
        if not verify_owned_process(receipt, identity, kind="remote server"):
            self.event("server_already_exited", model=model["key"], pid=receipt["pid"])
            return
        script = r'''import json, os, pathlib, signal, sys, time
p=json.loads(sys.argv[1]); pid=int(p["pid"]); base=pathlib.Path("/proc")/str(pid)
def ident():
 try:
  s=(base/"stat").read_text(); tail=s[s.rfind(")")+2:].split(); state=tail[0]
  argv=[x.decode("utf-8","surrogateescape") for x in (base/"cmdline").read_bytes().split(b"\0") if x]
  return {"alive":state!="Z","state":state,"start_ticks":int(tail[19]),"argv":argv}
 except FileNotFoundError: return {"alive":False,"state":"exited","argv":[]}
i=ident()
if not i["alive"]: print(json.dumps({"stopped":True,"already_exited":True})); raise SystemExit(0)
if i["start_ticks"]!=p["start_ticks"] or i["argv"]!=p["argv"]: print(json.dumps({"error":"ownership mismatch","identity":i})); raise SystemExit(0)
os.kill(pid,signal.SIGTERM); deadline=time.monotonic()+60
while time.monotonic()<deadline:
 i=ident()
 if not i["alive"]: print(json.dumps({"stopped":True,"final_state":i["state"]})); raise SystemExit(0)
 time.sleep(.2)
print(json.dumps({"error":"owned server did not exit after SIGTERM","identity":ident()}))'''
        result = self.remote_json(script, dict(receipt), timeout=75)
        if not isinstance(result, dict) or result.get("stopped") is not True:
            raise StudyError(f"owned server stop failed for {model['key']}: {result}")
        stop_value = {"at": now(), "ownership": dict(receipt), "result": result}
        atomic_json(self.study / f"{model['key']}-server-stop.json", stop_value)
        self.event("server_stopped", model=model["key"], pid=receipt["pid"])

    def assert_no_live_server(self, receipt: Mapping[str, Any]) -> None:
        identity = self.remote_process(receipt["pid"])
        if identity.get("alive"):
            verify_owned_process(receipt, identity, kind="remote server")
            raise StudyError("refusing benchmark while the model server is live")

    def launch_benchmark(self, model: Mapping[str, Any]) -> dict[str, Any]:
        argv = bench_command(model)
        remote_receipt = REMOTE_RESULTS + f"/{model['key']}-bench-launch.json"
        remote_json = REMOTE_RESULTS + f"/{model['key']}-llama-bench.json"
        remote_log = REMOTE_RESULTS + f"/{model['key']}-llama-bench.log"
        worker = r'''import json, os, pathlib, subprocess, sys
p=json.loads(sys.argv[1]); rec=pathlib.Path(p["receipt"]); out=pathlib.Path(p["stdout"]); err=pathlib.Path(p["stderr"])
def write(v):
 t=rec.with_name(rec.name+".tmp-"+str(os.getpid())); t.write_text(json.dumps(v,sort_keys=True)+"\n"); os.replace(t,rec)
with out.open("wb") as so, err.open("wb") as se:
 c=subprocess.Popen(p["argv"],stdin=subprocess.DEVNULL,stdout=so,stderr=se,start_new_session=True)
 deadline=__import__("time").monotonic()+2; base=pathlib.Path("/proc")/str(c.pid); argv=[]
 while __import__("time").monotonic()<deadline:
  if not base.exists(): raise SystemExit("llama-bench exited during launch")
  argv=[x.decode("utf-8","surrogateescape") for x in (base/"cmdline").read_bytes().split(b"\0") if x]
  if argv and argv[0].endswith("/llama-bench"): break
  __import__("time").sleep(.01)
 if not argv or not argv[0].endswith("/llama-bench"): raise SystemExit("taskset did not exec llama-bench")
 s=(pathlib.Path("/proc")/str(c.pid)/"stat").read_text(); tail=s[s.rfind(")")+2:].split()
 v={"schema_version":1,"kind":"tachyon_llama_bench","status":"running","launch_id":p["launch_id"],"pid":c.pid,"start_ticks":int(tail[19]),"argv":argv,"launch_argv":p["argv"],"started_at":p["at"],"stdout":str(out),"stderr":str(err)}; write(v)
 rc=c.wait(); v.update(status="complete" if rc==0 else "failed",returncode=rc,finished_at=__import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat()); write(v)'''
        launcher = r'''import json, os, pathlib, subprocess, sys
p=json.loads(sys.argv[1]); rec=pathlib.Path(p["receipt"]); rec.parent.mkdir(parents=True,exist_ok=True)
c=subprocess.Popen(["python3","-c",p["worker"],json.dumps(p["job"],separators=(",",":"))],stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)
v={"schema_version":1,"kind":"tachyon_llama_bench","status":"launching","launch_id":p["job"]["launch_id"],"supervisor_pid":c.pid,"launch_argv":p["job"]["argv"],"started_at":p["job"]["at"],"stdout":p["job"]["stdout"],"stderr":p["job"]["stderr"]}
t=rec.with_name(rec.name+".tmp-"+str(os.getpid())); t.write_text(json.dumps(v,sort_keys=True)+"\n"); os.replace(t,rec)
print(json.dumps(v))'''
        launch_id = f"{os.getpid()}-{time.time_ns()}"
        job = {
            "launch_id": launch_id,
            "argv": argv,
            "receipt": remote_receipt,
            "stdout": remote_json,
            "stderr": remote_log,
            "at": now(),
        }
        self.remote_json(launcher, {"receipt": remote_receipt, "worker": worker, "job": job})
        deadline = time.monotonic() + 20
        receipt = None
        while time.monotonic() < deadline:
            receipt = self.read_remote_json_file(remote_receipt)
            if receipt is not None and receipt.get("launch_id") == launch_id:
                break
            time.sleep(0.2)
        if receipt is None:
            raise StudyError("remote benchmark did not publish an ownership receipt")
        validate_bench_receipt(receipt, model)
        atomic_json(self.study / f"{model['key']}-bench-launch.json", receipt)
        self.append_launch(f"{model['key']}-bench", receipt)
        self.event("benchmark_launched", model=model["key"], pid=receipt.get("pid"))
        return receipt

    def benchmark(self, model: Mapping[str, Any], server_receipt: Mapping[str, Any]) -> None:
        local_raw = self.study / f"{model['key']}-llama-bench.json"
        local_log = self.study / f"{model['key']}-llama-bench.log"
        local_receipt_path = self.study / f"{model['key']}-bench-launch.json"
        if local_raw.is_file() and local_log.is_file() and local_receipt_path.is_file():
            receipt = read_json(local_receipt_path)
            if isinstance(receipt, dict):
                validate_bench_receipt(receipt, model)
            if (
                isinstance(receipt, dict)
                and receipt.get("status") == "complete"
                and receipt.get("returncode") == 0
                and valid_bench_output(local_raw)
            ):
                self.event("benchmark_preserved", model=model["key"])
                return
        self.assert_no_live_server(server_receipt)
        remote_receipt_path = REMOTE_RESULTS + f"/{model['key']}-bench-launch.json"
        receipt = self.read_remote_json_file(remote_receipt_path)
        if receipt is not None:
            validate_bench_receipt(receipt, model)
        if receipt is None or receipt.get("status") not in {"launching", "running", "complete"}:
            receipt = self.launch_benchmark(model)
        deadline = time.monotonic() + 7_200
        dead_since: float | None = None
        launching_since = time.monotonic()
        while receipt.get("status") in {"launching", "running"}:
            self.check_interrupt()
            if time.monotonic() >= deadline:
                raise StudyError(f"llama-bench timed out for {model['key']}")
            if receipt.get("status") == "launching":
                if time.monotonic() - launching_since > 30:
                    raise StudyError(
                        f"remote benchmark did not publish its PID for {model['key']}"
                    )
            else:
                identity = self.remote_process(receipt["pid"])
                if identity.get("alive"):
                    verify_owned_process(receipt, identity, kind="remote benchmark")
                    dead_since = None
                else:
                    # The worker updates the receipt just after reaping llama-bench.
                    dead_since = dead_since or time.monotonic()
                    if time.monotonic() - dead_since > 10:
                        raise StudyError(
                            f"remote benchmark exited without a terminal receipt for {model['key']}"
                        )
            time.sleep(2)
            updated = self.read_remote_json_file(remote_receipt_path)
            if updated is None:
                raise StudyError("remote benchmark receipt disappeared")
            validate_bench_receipt(updated, model)
            receipt = updated
        atomic_json(local_receipt_path, receipt)
        if receipt.get("status") != "complete" or receipt.get("returncode") != 0:
            raise StudyError(f"llama-bench failed for {model['key']}: {receipt.get('returncode')}")
        atomic_bytes(local_raw, self.remote_file(receipt["stdout"]))
        atomic_bytes(local_log, self.remote_file(receipt["stderr"]))
        if not valid_bench_output(local_raw):
            raise StudyError(f"llama-bench returned invalid JSON for {model['key']}")
        self.event("benchmark_complete", model=model["key"])

    def summarize_one(self, model: Mapping[str, Any]) -> None:
        output = self.study / str(model["key"])
        for suffix in ("json", "csv"):
            destination = self.study / f"{model['key']}-summary.{suffix}"
            command = [
                self.python, str(ADAPTER), "summarize",
                "--output-dir", str(output), "--output", str(destination),
            ]
            completed = subprocess.run(command, cwd=ROOT, check=False)
            if completed.returncode:
                raise StudyError(f"adapter summarize failed for {model['key']} ({suffix})")

    def _native_rates(self, output: Path) -> dict[str, float | None]:
        """Aggregate the server-provided rates over selected terminal results.

        llama.cpp's reported generation rate uses generated transitions
        (normally ``predicted_n - 1``), while the adapter's count/time metric
        uses ``predicted_n``. Both definitions are retained explicitly.
        """
        status = read_json(output / "status.json")
        records = status.get("cases") if isinstance(status, dict) else None
        if not isinstance(records, list):
            raise StudyError(f"quality status lacks cases: {output}")
        prompt_weighted = generation_weighted = 0.0
        prompt_ms_total = generation_ms_total = 0.0
        prompt_n_total = 0.0
        for record in records:
            if not isinstance(record, dict):
                continue
            suite, storage_id = record.get("suite"), record.get("storage_id")
            if not isinstance(suite, str) or not isinstance(storage_id, str):
                continue
            selected: dict[str, Any] | None = None
            for path in sorted((output / "cases" / suite / storage_id).glob("attempt-*/result.json")):
                value = read_json(path)
                if isinstance(value, dict) and value.get("status") in TERMINAL_QUALITY:
                    selected = value
                    break
            timing = selected.get("native_timings") if selected is not None else None
            if not isinstance(timing, dict):
                continue
            prompt_ms = timing.get("prompt_ms")
            prompt_rate = timing.get("prompt_per_second")
            prompt_n = timing.get("prompt_n")
            if (
                isinstance(prompt_ms, (int, float))
                and not isinstance(prompt_ms, bool)
                and prompt_ms > 0
                and isinstance(prompt_rate, (int, float))
                and not isinstance(prompt_rate, bool)
            ):
                prompt_ms_total += float(prompt_ms)
                prompt_weighted += float(prompt_rate) * float(prompt_ms)
                if isinstance(prompt_n, (int, float)) and not isinstance(prompt_n, bool):
                    prompt_n_total += float(prompt_n)
            generation_ms = timing.get("predicted_ms")
            generation_rate = timing.get("predicted_per_second")
            if (
                isinstance(generation_ms, (int, float))
                and not isinstance(generation_ms, bool)
                and generation_ms > 0
                and isinstance(generation_rate, (int, float))
                and not isinstance(generation_rate, bool)
            ):
                generation_ms_total += float(generation_ms)
                generation_weighted += float(generation_rate) * float(generation_ms)
        return {
            "prompt_reported": prompt_weighted / prompt_ms_total if prompt_ms_total else None,
            "prompt_count_time": prompt_n_total * 1000 / prompt_ms_total if prompt_ms_total else None,
            "generation_reported": generation_weighted / generation_ms_total if generation_ms_total else None,
        }

    def consolidate(self) -> None:
        rows: list[dict[str, Any]] = []
        for model in MODELS:
            summary = read_json(self.study / f"{model['key']}-summary.json")
            metrics = next((row for row in summary.get("rows", []) if row.get("scope") == "all"), {})
            grading = read_json(self.study / str(model["key"]) / "grading/summary.json")
            evalplus = grading.get("evalplus", {}) if isinstance(grading, dict) else {}
            ifeval = grading.get("ifeval", {}) if isinstance(grading, dict) else {}
            strict = ifeval.get("strict", {}) if isinstance(ifeval, dict) else {}
            loose = ifeval.get("loose", {}) if isinstance(ifeval, dict) else {}
            native = self._native_rates(self.study / str(model["key"]))
            server_reported = metrics.get(
                "server_reported_generation_tokens_per_second"
            )
            if (
                isinstance(server_reported, (int, float))
                and isinstance(native["generation_reported"], (int, float))
                and abs(float(server_reported) - float(native["generation_reported"]))
                > 1e-9 * max(1.0, abs(float(server_reported)))
            ):
                raise StudyError(
                    f"adapter and controller native rate aggregation differ for {model['key']}"
                )
            row = {
                "model_key": model["key"],
                "model_alias": f"tachyon-{model['key']}",
                "model_size_bytes": model["size"],
                "native_prompt_tokens_per_second": native["prompt_reported"],
                "native_generation_tokens_per_second": server_reported,
                "native_generation_reported_case_count": metrics.get(
                    "server_reported_generation_timing_case_count"
                ),
                "native_prompt_count_time_tokens_per_second": native["prompt_count_time"],
                "native_generation_count_time_tokens_per_second": metrics.get(
                    "count_time_derived_generation_tokens_per_second"
                ),
                "wall_seconds": metrics.get("wall_seconds_sum"),
                "ttft_seconds_mean": metrics.get("ttft_seconds_mean"),
                "prompt_tokens": metrics.get("prompt_tokens"),
                "completion_tokens": metrics.get("completion_tokens"),
                "total_tokens": metrics.get("total_tokens"),
                "completed": metrics.get("count_completed"),
                "truncated": metrics.get("count_truncated"),
                "api_errors": metrics.get("count_api_failure"),
                "evalplus_base_pass_count": evalplus.get("base_pass_count"),
                "evalplus_base_failure_count": evalplus.get("base_test_failure_count"),
                "evalplus_plus_pass_count": evalplus.get("plus_pass_count"),
                "evalplus_plus_failure_count": evalplus.get("plus_test_failure_count"),
                "ifeval_strict_prompt_accuracy": strict.get("prompt_accuracy") if isinstance(strict, dict) else None,
                "ifeval_strict_instruction_accuracy": strict.get("instruction_accuracy") if isinstance(strict, dict) else None,
                "ifeval_loose_prompt_accuracy": loose.get("prompt_accuracy") if isinstance(loose, dict) else None,
                "ifeval_loose_instruction_accuracy": loose.get("instruction_accuracy") if isinstance(loose, dict) else None,
                "llama_bench_raw": str(self.study / f"{model['key']}-llama-bench.json"),
            }
            rows.append(row)
        document = {
            "schema_version": 1,
            "kind": "tachyon_cpu_screening_consolidated",
            "created_at": now(),
            "manifest_sha256": MANIFEST_SHA256,
            "full_leaderboard_claim": False,
            "rows": rows,
        }
        atomic_json(self.study / "tachyon-study-summary.json", document)
        csv_path = self.study / "tachyon-study-summary.csv"
        buffer = io.StringIO(newline="")
        writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        csv_path.write_text(buffer.getvalue(), encoding="utf-8")
        self.write_chart(rows)

    def write_chart(self, rows: Sequence[Mapping[str, Any]]) -> None:
        # Small dependency-free chart. Missing measurements remain labelled N/A.
        width, height = 920, 330
        values = [row.get("native_generation_tokens_per_second") for row in rows]
        numeric = [float(value) for value in values if isinstance(value, (int, float))]
        maximum = max(numeric, default=1.0)
        svg = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            '<rect width="100%" height="100%" fill="#111827"/>',
            '<text x="40" y="42" fill="#f9fafb" font-family="sans-serif" font-size="22">Tachyon CPU native generation throughput</text>',
            '<text x="40" y="68" fill="#9ca3af" font-family="sans-serif" font-size="13">16-case frozen screening; publisher-native recipes</text>',
        ]
        colors = ("#60a5fa", "#34d399", "#fbbf24")
        for index, (row, value) in enumerate(zip(rows, values)):
            x = 70 + index * 285
            bar_height = 0 if not isinstance(value, (int, float)) else 180 * float(value) / maximum
            y = 270 - bar_height
            svg.append(f'<rect x="{x}" y="{y:.1f}" width="170" height="{bar_height:.1f}" rx="5" fill="{colors[index]}"/>')
            label = "N/A" if not isinstance(value, (int, float)) else f"{float(value):.2f} tok/s"
            svg.append(f'<text x="{x + 85}" y="{max(100, y - 10):.1f}" text-anchor="middle" fill="#f9fafb" font-family="sans-serif" font-size="16">{label}</text>')
            svg.append(f'<text x="{x + 85}" y="300" text-anchor="middle" fill="#d1d5db" font-family="sans-serif" font-size="14">{row["model_key"]}</text>')
        svg.append("</svg>\n")
        (self.study / "tachyon-study-chart.svg").write_text("".join(svg), encoding="utf-8")

    def run_model(self, model: Mapping[str, Any], *, last: bool) -> None:
        key = str(model["key"])
        model_state = self.state.setdefault("models", {}).setdefault(key, {})
        model_state.update(status="verifying", model_size_bytes=model["size"], started_at=model_state.get("started_at", now()))
        self.save(current_model=key)
        verified = self.observe_model(model)
        atomic_json(self.study / f"{key}-model-verification.json", verified)
        model_state["verification"] = str(self.study / f"{key}-model-verification.json")
        audit = audit_quality(self.study / key, model)
        needs_server = not audit["terminal"] or not all(path.exists() for path in self.smoke_paths(model)) or not (self.study / f"{key}-resource-snapshot.json").exists()
        receipt: dict[str, Any] | None = self.current_server(model)
        if needs_server:
            receipt = self.ensure_server(model)
            model_state["status"] = "probing"
            self.save()
            self.run_smoke(model)
            model_state["status"] = "quality"
            self.save()
            audit = self.wait_quality(model)
        elif not audit["terminal"]:
            raise StudyError(f"quality evidence incomplete for {key}")
        model_state.update(status="grading", quality_counts=audit.get("counts"))
        self.save()
        self.grade(model)
        if not (self.study / f"{key}-resource-snapshot.json").exists():
            receipt = self.ensure_server(model)
            self.snapshot(model, receipt)
        if receipt is None:
            receipt = self.current_server(model)
        if receipt is not None:
            self.stop_server(model, receipt)
        else:
            stop_path = self.study / f"{key}-server-stop.json"
            if not stop_path.is_file():
                raise StudyError(f"no owned server receipt or stop evidence for {key}")
            receipt = read_json(stop_path)["ownership"]
        model_state["status"] = "benchmarking"
        self.save()
        self.benchmark(model, receipt)
        self.summarize_one(model)
        if last:
            final_receipt = self.ensure_server(model)
            model_state["final_server"] = {
                "pid": final_receipt["pid"],
                "start_ticks": final_receipt["start_ticks"],
                "alias": final_receipt["alias"],
                "left_running": True,
            }
        model_state.update(status="complete", finished_at=now())
        self.save()
        self.event("model_complete", model=key)

    def run(self) -> int:
        self.acquire_lock()
        for signum in (signal.SIGINT, signal.SIGTERM):
            signal.signal(signum, self._handle_signal)
        try:
            self.initialize()
            if self.preexisting_complete:
                self.event("study_already_complete")
                return 0
            self.event("study_started", manifest_sha256=MANIFEST_SHA256)
            for index, model in enumerate(MODELS):
                self.run_model(model, last=index == len(MODELS) - 1)
            self.consolidate()
            self.save(status="complete", current_model=None, finished_at=now())
            self.event("study_complete")
            return 0
        except BaseException as exc:
            status = "interrupted" if self.interrupted is not None else "failed"
            if not self.state:
                if self.state_path.exists():
                    try:
                        existing = read_json(self.state_path)
                        if isinstance(existing, dict):
                            self.state = existing
                    except StudyError:
                        atomic_json(
                            self.study / "controller-failure.json",
                            {"at": now(), "status": status, "error": str(exc)},
                        )
                else:
                    self.state = {
                        "schema_version": 1,
                        "kind": "tachyon_cpu_public_screening",
                        "study": str(self.study),
                        "created_at": now(),
                        "manifest_sha256": MANIFEST_SHA256,
                        "model_order": [row["key"] for row in MODELS],
                        "models": {},
                    }
            if self.state:
                self.save(status=status, error=str(exc), finished_at=now())
                try:
                    self.event("study_failed", error=str(exc), status=status)
                except Exception:
                    pass
            print(f"tachyon_study: {exc}", file=sys.stderr, flush=True)
            return 1


def parse_args(arguments: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=DEFAULT_STUDY)
    parser.add_argument("--python", default=str(PYTHON))
    parser.add_argument("--remote", default="akovanda@tachyon")
    return parser.parse_args(arguments)


def main(arguments: Iterable[str] = sys.argv[1:]) -> int:
    return Controller(parse_args(arguments)).run()


if __name__ == "__main__":
    raise SystemExit(main())
