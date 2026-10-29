#!/usr/bin/env python3
"""Run a durable CPU8 and partial-OpenCL trial after the Tachyon CPU study.

The controller waits for the CPU study to be complete, its recorded current
controller to exit, and its flock to be released before touching Tachyon.  It
only stops processes whose PID, Linux start ticks, and exact argv match a
durable receipt.  Regardless of trial success it restores the proven CPU4 LFM
server; GPU checks are smoke checks and never become a quality claim.
"""

from __future__ import annotations

import argparse
import base64
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import time
from types import ModuleType
from typing import Any, Iterable, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STUDY = ROOT / "results/tachyon-study-20261023-v1"
DEFAULT_TRIAL = DEFAULT_STUDY / "gpu-trial"
ADAPTER = ROOT / "runtime/tachyon_benchmark.py"
CPU_CONTROLLER = ROOT / "runtime/tachyon_study.py"
OPENCL_RECEIPT = DEFAULT_STUDY / "opencl-runtime-build.json"
REMOTE_ROOT = "/external/device-workspace"
REMOTE_CPU_RUNTIME = REMOTE_ROOT + "/runtime/llama-cpp-cpu"
REMOTE_OPENCL_RUNTIME = REMOTE_ROOT + "/runtime/llama-cpp-opencl"
REMOTE_ICD = REMOTE_ROOT + "/runtime/opencl/vendors"
REMOTE_RESULTS = REMOTE_ROOT + "/results/tachyon-gpu-trial-20261023-v1"
REMOTE_MODEL = REMOTE_ROOT + "/models/lfm25-26b/LFM2.5-2.6B-QAD-Q4_0.gguf"
MODEL_SIZE = 1_593_894_944
MODEL_SHA256 = "a247afd6414918eac8e520a9e6137dc271235461ecbe1180462221d5b8d40b03"
CONTROLLER_SHA256 = "376c09ff4693ceaafb412fda26601394a46f563ec1a7928308cdda4ad8ce491b"
ADAPTER_SHA256 = "3da0e9a074adde12755da6846ef4e480e383665ac225586e4bfd7181a6285459"
MANIFEST_SHA256 = "da6a70653695d317ac131d99eb09fe772cb8191c41d467345e9e3f4c11e90caf"
OPENCL_RECEIPT_SHA256 = "4a35945df5ac4c68aa3cca8fdd3dfedaaad8ae4c140c2b095252cf7a6d255da9"
CPU_ALIAS = "tachyon-lfm25-26b"
GPU_ALIAS = "tachyon-lfm25-26b-gpu"
ENDPOINT = "http://127.0.0.1:18185/v1"
SENSOR_PROMPT = (
    'Return only a JSON object with keys "answer" and "explanation". '
    "A sensor records 18, 22, 20, 24 degrees. What is the average? "
    "Keep the explanation to one sentence."
)
EXACT_PROMPT = "Reply with exactly this text and nothing else: tachyon-opencl-ok"
SSH_OPTIONS = (
    "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes",
    "-o", "ClearAllForwardings=yes", "-o", "ForwardAgent=no",
    "-o", "RequestTTY=no",
)
VARIANTS: tuple[dict[str, Any], ...] = (
    {"key": "cpu8", "backend": "cpu", "cpuset": "0-7", "threads": 8, "ngl": 0},
    {"key": "opencl-ngl4", "backend": "opencl", "cpuset": "4-7", "threads": 4, "ngl": 4},
    {"key": "opencl-ngl8", "backend": "opencl", "cpuset": "4-7", "threads": 4, "ngl": 8},
)


class TrialError(RuntimeError):
    """A provenance, ownership, or durability invariant failed."""


class VariantFailure(RuntimeError):
    """A bounded runtime/hardware variant could not be exercised."""


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise TrialError(f"cannot hash required artifact: {path}") from exc
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TrialError(f"invalid or unreadable JSON: {path}") from exc


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
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        tail = stat[stat.rfind(")") + 2 :].split()
        state = tail[0]
        argv = [
            item.decode("utf-8", "surrogateescape")
            for item in Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0") if item
        ]
        return {"pid": pid, "alive": state != "Z", "state": state,
                "start_ticks": int(tail[19]), "argv": argv}
    except (FileNotFoundError, ProcessLookupError):
        return {"pid": pid, "alive": False, "state": "exited", "argv": []}
    except (OSError, ValueError, IndexError) as exc:
        raise TrialError(f"cannot inspect local process {pid}: {exc}") from exc


def lock_is_free(path: Path) -> bool:
    """Probe the CPU study lock without changing the lock-file bytes."""
    if not path.is_file():
        return False
    with path.open("r") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
    return True


def verify_owned_process(receipt: Mapping[str, Any], identity: Mapping[str, Any], *, kind: str) -> bool:
    if not identity.get("alive"):
        return False
    if receipt.get("pid") != identity.get("pid"):
        raise TrialError(f"{kind} PID differs from its ownership receipt")
    if receipt.get("start_ticks") != identity.get("start_ticks"):
        raise TrialError(f"{kind} start-tick identity mismatch")
    if receipt.get("argv") != identity.get("argv"):
        raise TrialError(f"{kind} cmdline differs from its ownership receipt")
    return True


def flag_value(argv: Sequence[str], flag: str) -> str:
    try:
        return argv[argv.index(flag) + 1]
    except (ValueError, IndexError) as exc:
        raise TrialError(f"command lacks required {flag}") from exc


def validate_cpu_server_argv(argv: Sequence[str]) -> None:
    command = list(argv)
    if command[:3] == ["taskset", "-c", "4-7"]:
        command = command[3:]
    if not command or command[0] != REMOTE_CPU_RUNTIME + "/bin/llama-server":
        raise TrialError("CPU server executable differs from the private pinned runtime")
    expected = {
        "-m": REMOTE_MODEL, "--alias": CPU_ALIAS, "--host": "127.0.0.1",
        "--port": "18085", "-c": "16384", "-np": "1", "-t": "4",
        "-tb": "4", "-b": "512", "-ub": "128", "-ngl": "0",
        "--reasoning-effort": "none", "--reasoning-format": "deepseek",
        "--threads-http": "2",
    }
    for flag, wanted in expected.items():
        if flag_value(command, flag) != wanted:
            raise TrialError(f"CPU server has unexpected {flag}")
    if "--jinja" not in command:
        raise TrialError("CPU server lacks --jinja")


def actual_exec_argv(launch_argv: Sequence[str]) -> list[str]:
    command = list(launch_argv)
    if command[:1] == ["env"]:
        command = command[2:]
    if command[:1] == ["taskset"]:
        command = command[3:]
    return command


def bench_command(variant: Mapping[str, Any], *, device_supported: bool) -> list[str]:
    runtime = REMOTE_OPENCL_RUNTIME if variant["backend"] == "opencl" else REMOTE_CPU_RUNTIME
    command: list[str] = []
    if variant["backend"] == "opencl":
        command += ["env", f"OCL_ICD_VENDORS={REMOTE_ICD}"]
    command += [
        "taskset", "-c", str(variant["cpuset"]), runtime + "/bin/llama-bench",
        "-m", REMOTE_MODEL, "-p", "128", "-n", "64",
        "-t", str(variant["threads"]), "-r", "3", "-ngl", str(variant["ngl"]),
    ]
    if variant["backend"] == "opencl" and device_supported:
        command += ["--device", "GPUOpenCL"]
    if variant["backend"] == "opencl":
        # llama-bench installs a null llama.cpp log callback unless verbose is
        # requested.  The loader's offloaded-layer count is required evidence,
        # and remains on stderr while ``-o json`` stays clean on stdout.
        command.append("-v")
    return command + ["-o", "json"]


def server_command(*, backend: str, ngl: int, device_supported: bool, fit_supported: bool) -> list[str]:
    runtime = REMOTE_OPENCL_RUNTIME if backend == "opencl" else REMOTE_CPU_RUNTIME
    alias = GPU_ALIAS if backend == "opencl" else CPU_ALIAS
    command: list[str] = []
    if backend == "opencl":
        command += ["env", f"OCL_ICD_VENDORS={REMOTE_ICD}"]
    command += [
        "taskset", "-c", "4-7", runtime + "/bin/llama-server",
        "-m", REMOTE_MODEL, "--alias", alias, "--host", "127.0.0.1",
        "--port", "18085", "-c", "16384", "-np", "1", "-t", "4",
        "-tb", "4", "-b", "512", "-ub", "128", "-ngl", str(ngl),
    ]
    if backend == "opencl" and device_supported:
        command += ["--device", "GPUOpenCL"]
    if backend == "opencl" and fit_supported:
        command += ["--fit", "off"]
    return command + ["--jinja", "--reasoning-effort", "none",
                      "--reasoning-format", "deepseek", "--threads-http", "2"]


def validate_bench_receipt(receipt: Mapping[str, Any], expected: Sequence[str]) -> None:
    if receipt.get("kind") != "tachyon_gpu_trial_bench":
        raise TrialError("benchmark receipt kind differs")
    if receipt.get("launch_argv") != list(expected):
        raise TrialError("benchmark receipt launch argv differs")
    if receipt.get("status") == "launching":
        if not isinstance(receipt.get("supervisor_pid"), int):
            raise TrialError("launching benchmark lacks supervisor PID")
    elif receipt.get("status") == "failed" and "pid" not in receipt:
        # Popen/exec can fail before a child identity exists (for example, an
        # unsupported binary).  The exact launch argv is still bound above;
        # there is no child we could safely claim or signal.
        if not isinstance(receipt.get("error"), str):
            raise TrialError("pre-exec failed benchmark lacks an error receipt")
    elif receipt.get("argv") != actual_exec_argv(expected):
        raise TrialError("benchmark receipt executable argv differs")


def parse_offload_evidence(log: str, requested: int) -> dict[str, Any]:
    lines = [line for line in log.splitlines() if re.search(r"offload|OpenCL|GPUOpenCL", line, re.I)]
    counts: list[int] = []
    for line in lines:
        for pattern in (r"offload(?:ing|ed)?\s+(\d+)\s+(?:repeating\s+)?layers?",
                        r"offloaded\s+(\d+)\s*/\s*\d+\s+layers?"):
            match = re.search(pattern, line, re.I)
            if match:
                counts.append(int(match.group(1)))
    actual = max(counts) if counts else None
    proven = bool(requested and actual and actual > 0)
    return {
        "requested_gpu_layers": requested, "reported_offloaded_layers": actual,
        "opencl_log_evidence": lines, "gpu_compute_proven": proven,
        "classification": "cpu_only" if requested == 0 else
                          "partial_gpu_offload" if proven else "gpu_offload_not_proven",
    }


def benchmark_rates(value: Any) -> dict[str, float | None]:
    rows = value if isinstance(value, list) else [value]
    rates: dict[str, list[float]] = {"prompt": [], "generation": []}
    for row in rows:
        if not isinstance(row, dict) or isinstance(row.get("avg_ts"), bool) or not isinstance(row.get("avg_ts"), (int, float)):
            continue
        test = str(row.get("test", ""))
        if test.startswith("pp"):
            rates["prompt"].append(float(row["avg_ts"]))
        elif test.startswith("tg"):
            rates["generation"].append(float(row["avg_ts"]))
    return {"prompt_tokens_per_second": rates["prompt"][-1] if rates["prompt"] else None,
            "generation_tokens_per_second": rates["generation"][-1] if rates["generation"] else None}


def load_adapter() -> ModuleType:
    if sha256(ADAPTER) != ADAPTER_SHA256:
        raise TrialError("Tachyon adapter SHA-256 changed")
    spec = importlib.util.spec_from_file_location("_tachyon_gpu_trial_adapter", ADAPTER)
    if spec is None or spec.loader is None:
        raise TrialError("cannot import pinned Tachyon adapter")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def smoke_payload(prompt: str, max_tokens: int, model_alias: str) -> dict[str, Any]:
    recipe = load_adapter().native_recipe("lfm26", "lfm25-26b")
    return {
        "model": model_alias, "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens, "seed": 20_261_023, **recipe["request_fields"],
        # The GGUF chat template unconditionally opens <think>.  This server
        # request key forces the native reasoning-budget sampler to close it
        # immediately so short decision probes measure final-answer behavior.
        "reasoning_budget_tokens": 0,
        "stream": True, "stream_options": {"include_usage": True},
        "timings_per_token": True,
    }


def remote_process_script() -> str:
    return r'''import json,pathlib,sys
p=json.loads(sys.argv[1]); pid=int(p["pid"]); base=pathlib.Path("/proc")/str(pid)
try:
 s=(base/"stat").read_text(); t=s[s.rfind(")")+2:].split(); state=t[0]
 a=[x.decode("utf-8","surrogateescape") for x in (base/"cmdline").read_bytes().split(b"\0") if x]
 print(json.dumps({"pid":pid,"alive":state!="Z","state":state,"start_ticks":int(t[19]),"argv":a}))
except FileNotFoundError: print(json.dumps({"pid":pid,"alive":False,"state":"exited","argv":[]}))'''


class Controller:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.study = args.study.resolve()
        self.trial = args.trial_dir.resolve()
        self.status_path = self.trial / "trial-status.json"
        self.events_path = self.trial / "events.jsonl"
        self.state: dict[str, Any] = {}
        self.lock_stream: io.TextIOWrapper | None = None
        self.initial_cpu_receipt: dict[str, Any] | None = None
        self.cpu_was_stopped = False
        self.restore_needs_relaunch = False
        self.interrupted: int | None = None
        self.capabilities: dict[str, bool] = {}

    def event(self, name: str, **fields: Any) -> None:
        row = {"at": now(), "event": name, **fields}
        append_jsonl(self.events_path, row)
        print(json.dumps(row, sort_keys=True), flush=True)

    def save(self, **fields: Any) -> None:
        self.state.update(fields)
        self.state.update(controller_pid=os.getpid(), updated_at=now())
        atomic_json(self.status_path, self.state)

    def acquire_lock(self) -> None:
        self.trial.mkdir(parents=True, exist_ok=True)
        self.lock_stream = (self.trial / ".controller.lock").open("a+")
        try:
            fcntl.flock(self.lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise TrialError("another Tachyon GPU trial controller holds the lock") from exc
        self.lock_stream.seek(0)
        self.lock_stream.truncate()
        self.lock_stream.write(f"{os.getpid()}\n")
        self.lock_stream.flush()

    def check_interrupt(self) -> None:
        if self.interrupted is not None:
            raise TrialError(f"controller interrupted by signal {self.interrupted}")

    def _handle_signal(self, signum: int, _frame: Any) -> None:
        self.interrupted = signum

    def wait_for_cpu_study(self) -> dict[str, Any]:
        deadline = time.monotonic() + self.args.wait_timeout
        status_path = self.study / "study-status.json"
        launch_path = self.study / "controller-launch.json"
        lock_path = self.study / ".controller.lock"
        latest = "study status absent"
        while time.monotonic() < deadline:
            self.check_interrupt()
            if status_path.is_file() and launch_path.is_file():
                state, launch = read_json(status_path), read_json(launch_path)
                if not isinstance(state, dict) or not isinstance(launch, dict):
                    raise TrialError("CPU controller status/launch receipt is malformed")
                pid = state.get("controller_pid")
                if pid != launch.get("pid"):
                    raise TrialError("CPU status PID differs from latest controller-launch receipt")
                if launch.get("source_sha256") != CONTROLLER_SHA256:
                    raise TrialError("latest CPU controller launch source SHA-256 differs")
                argv = launch.get("argv")
                direct = [str(ROOT / ".venv-public-bench/bin/python"), str(CPU_CONTROLLER),
                          "--study", str(self.study)]
                module = [str(ROOT / ".venv-public-bench/bin/python"), "-m", "runtime.tachyon_study",
                          "--study", str(self.study)]
                if argv not in (direct, module):
                    raise TrialError("latest CPU controller launch argv is not the pinned direct/module command")
                identity = process_identity(pid) if isinstance(pid, int) and not isinstance(pid, bool) else {"alive": False}
                free = lock_is_free(lock_path)
                if state.get("status") == "complete" and not identity.get("alive") and free:
                    return state
                latest = (f"status={state.get('status')!r}, controller_alive="
                          f"{bool(identity.get('alive'))}, lock_free={free}")
            time.sleep(self.args.poll)
        raise TrialError(f"timed out waiting for completed/released CPU study: {latest}")

    def verify_provenance(self, cpu_state: Mapping[str, Any]) -> None:
        expected = {
            CPU_CONTROLLER: CONTROLLER_SHA256, ADAPTER: ADAPTER_SHA256,
            self.study / "manifest.json": MANIFEST_SHA256,
            OPENCL_RECEIPT: OPENCL_RECEIPT_SHA256,
        }
        for path, digest in expected.items():
            if sha256(path) != digest:
                raise TrialError(f"required source SHA-256 differs: {path}")
        if cpu_state.get("manifest_sha256") != MANIFEST_SHA256:
            raise TrialError("CPU study manifest binding differs")
        pinned = cpu_state.get("source_pins")
        if not isinstance(pinned, dict):
            raise TrialError("CPU study lacks source pins")
        for path, digest in ((CPU_CONTROLLER, CONTROLLER_SHA256), (ADAPTER, ADAPTER_SHA256)):
            if pinned.get(str(path)) != digest:
                raise TrialError(f"CPU study source pin differs: {path}")
        lfm = cpu_state.get("models", {}).get("lfm25-26b")
        if not isinstance(lfm, dict) or lfm.get("status") != "complete":
            raise TrialError("CPU study lacks a complete LFM2.5 phase")
        required = [
            self.study / "source-pins.json", self.study / "study-status.json",
            self.study / "study-plan.json", self.study / "controller-launch.json",
            self.study / "lfm25-26b-model-verification.json",
            self.study / "lfm25-26b-server-launch.json",
            self.study / "lfm25-26b-bench-launch.json",
            self.study / "lfm25-26b-llama-bench.json",
            self.study / "lfm25-26b-smoke-request.json",
            self.study / "lfm25-26b-smoke-response.json",
            self.study / "lfm25-26b-smoke-evaluation.json",
        ]
        pins = {str(path): sha256(path) for path in required}
        pins.update({str(path): digest for path, digest in expected.items()})
        pins[str(Path(__file__).resolve())] = sha256(Path(__file__).resolve())
        path = self.trial / "source-pins.json"
        document = {"schema_version": 1, "created_at": now(), "sha256": pins}
        if path.exists():
            recorded = read_json(path)
            if not isinstance(recorded, dict) or recorded.get("sha256") != pins:
                raise TrialError("GPU trial pins changed since initialization")
        else:
            atomic_json(path, document)

    def remote_json(self, script: str, payload: Mapping[str, Any], *, timeout: float = 90) -> Any:
        remote_command = shlex.join(["python3", "-c", script,
                                     json.dumps(payload, separators=(",", ":"))])
        try:
            completed = subprocess.run(
                ["ssh", *SSH_OPTIONS, "-T", self.args.remote, remote_command],
                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, timeout=timeout, check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise TrialError("SSH control command timed out") from exc
        if completed.returncode:
            raise TrialError(f"remote command failed ({completed.returncode}): {completed.stderr.strip()[-1000:]}")
        try:
            return json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise TrialError("remote command returned invalid JSON") from exc

    def remote_process(self, pid: int) -> dict[str, Any]:
        result = self.remote_json(remote_process_script(), {"pid": pid})
        if not isinstance(result, dict):
            raise TrialError("remote process observation is malformed")
        return result

    def remote_json_file(self, path: str) -> dict[str, Any] | None:
        script = r'''import json,pathlib,sys
p=pathlib.Path(json.loads(sys.argv[1])["path"])
if not p.is_file(): print("null")
else:
 try: print(json.dumps(json.loads(p.read_text())))
 except Exception as e: print(json.dumps({"_error":str(e)}))'''
        value = self.remote_json(script, {"path": path})
        if value is None:
            return None
        if not isinstance(value, dict) or value.get("_error"):
            raise TrialError(f"invalid remote JSON receipt: {path}")
        return value

    def remote_file(self, path: str) -> bytes:
        script = r'''import base64,json,pathlib,sys
p=pathlib.Path(json.loads(sys.argv[1])["path"])
print(json.dumps({"data":base64.b64encode(p.read_bytes()).decode("ascii")}))'''
        value = self.remote_json(script, {"path": path}, timeout=180)
        try:
            return base64.b64decode(value["data"], validate=True)
        except (KeyError, ValueError, TypeError) as exc:
            raise TrialError(f"cannot retrieve remote file: {path}") from exc

    def initialize_remote(self) -> None:
        script = r'''import json,pathlib,sys
p=json.loads(sys.argv[1]); root=pathlib.Path(p["results"]); model=pathlib.Path(p["model"]); root.mkdir(parents=True,exist_ok=True)
try: st=model.stat(); artifact=json.loads((model.parent/"artifact.json").read_text())
except Exception as e: print(json.dumps({"error":str(e)})); raise SystemExit(0)
print(json.dumps({"path":str(model),"size":st.st_size,"regular":model.is_file(),"artifact":artifact}))'''
        value = self.remote_json(script, {"results": REMOTE_RESULTS, "model": REMOTE_MODEL})
        if not isinstance(value, dict) or value.get("error") or not value.get("regular"):
            raise TrialError(f"remote model observation failed: {value}")
        artifact = value.get("artifact")
        if not isinstance(artifact, dict):
            raise TrialError("remote model artifact receipt is malformed")
        expected = {"path": REMOTE_MODEL, "size": MODEL_SIZE,
                    "sha256": MODEL_SHA256, "actual_sha256": MODEL_SHA256}
        for field, wanted in expected.items():
            observed = value.get(field) if field in {"path", "size"} else artifact.get(field)
            if observed != wanted:
                raise TrialError(f"remote model {field} differs")
        atomic_json(self.trial / "model-verification.json", {"verified_at": now(), **value})

    def original_cpu_receipt(self, cpu_state: Mapping[str, Any]) -> dict[str, Any]:
        final = cpu_state.get("models", {}).get("lfm25-26b", {}).get("final_server")
        launch = read_json(self.study / "lfm25-26b-server-launch.json")
        if not isinstance(final, dict) or not isinstance(launch, dict):
            raise TrialError("final CPU server receipts are malformed")
        for field in ("pid", "start_ticks", "alias"):
            if final.get(field) != launch.get(field):
                raise TrialError(f"final CPU server {field} receipts disagree")
        if final.get("left_running") is not True or launch.get("alias") != CPU_ALIAS:
            raise TrialError("CPU study did not leave the expected LFM server running")
        validate_cpu_server_argv(launch.get("argv", []))
        return dict(launch)

    def stop_owned(self, receipt: Mapping[str, Any], *, label: str) -> dict[str, Any]:
        identity = self.remote_process(int(receipt.get("pid", -1)))
        if not verify_owned_process(receipt, identity, kind=label):
            return {"stopped": True, "already_exited": True, "identity": identity}
        script = r'''import json,os,pathlib,signal,sys,time
p=json.loads(sys.argv[1]); pid=int(p["pid"]); base=pathlib.Path("/proc")/str(pid)
def ident():
 try:
  s=(base/"stat").read_text(); t=s[s.rfind(")")+2:].split(); state=t[0]
  a=[x.decode("utf-8","surrogateescape") for x in (base/"cmdline").read_bytes().split(b"\0") if x]
  return {"pid":pid,"alive":state!="Z","state":state,"start_ticks":int(t[19]),"argv":a}
 except FileNotFoundError: return {"pid":pid,"alive":False,"state":"exited","argv":[]}
i=ident()
if not i["alive"]: print(json.dumps({"stopped":True,"already_exited":True,"identity":i})); raise SystemExit(0)
if i["start_ticks"]!=p["start_ticks"] or i["argv"]!=p["argv"]: print(json.dumps({"error":"ownership mismatch","identity":i})); raise SystemExit(0)
os.kill(pid,signal.SIGTERM); end=time.monotonic()+60
while time.monotonic()<end:
 i=ident()
 if not i["alive"]: print(json.dumps({"stopped":True,"final_identity":i})); raise SystemExit(0)
 time.sleep(.2)
print(json.dumps({"error":"owned process did not exit after SIGTERM","identity":ident()}))'''
        result = self.remote_json(script, dict(receipt), timeout=75)
        if not isinstance(result, dict) or result.get("stopped") is not True:
            raise TrialError(f"failed to stop {label}: {result}")
        return result

    def stop_initial_cpu(self, cpu_state: Mapping[str, Any]) -> None:
        self.initial_cpu_receipt = self.original_cpu_receipt(cpu_state)
        stop_path = self.trial / "cpu-server-stop.json"
        if stop_path.exists():
            prior = read_json(stop_path)
            if not isinstance(prior, dict) or prior.get("ownership") != self.initial_cpu_receipt:
                raise TrialError("prior CPU stop receipt differs from CPU study ownership")
            identity = self.remote_process(self.initial_cpu_receipt["pid"])
            if verify_owned_process(self.initial_cpu_receipt, identity, kind="initial CPU server"):
                raise TrialError("CPU stop receipt exists but original server is still live")
            self.cpu_was_stopped = True
            return
        identity = self.remote_process(self.initial_cpu_receipt["pid"])
        if not verify_owned_process(self.initial_cpu_receipt, identity, kind="initial CPU server"):
            raise TrialError("CPU study's final owned server exited before GPU handoff")
        atomic_json(self.trial / "cpu-server-before-stop.json",
                    {"captured_at": now(), "ownership": self.initial_cpu_receipt, "identity": identity})
        result = self.stop_owned(self.initial_cpu_receipt, label="initial CPU server")
        atomic_json(stop_path, {"stopped_at": now(), "ownership": self.initial_cpu_receipt, "result": result})
        self.cpu_was_stopped = True
        self.event("initial_cpu_server_stopped", pid=self.initial_cpu_receipt["pid"])

    def stop_prior_restore_for_resume(self) -> None:
        """Take back a CPU server restored by our own earlier failed run."""
        path = self.trial / "cpu-restore-server-launch.json"
        if not path.is_file():
            return
        receipt = read_json(path)
        if not isinstance(receipt, dict):
            raise TrialError("prior CPU restore receipt is malformed")
        validate_cpu_server_argv(receipt.get("argv", []))
        stop_path = self.trial / "cpu-restore-resume-stops.jsonl"
        identity = self.remote_process(receipt["pid"])
        if verify_owned_process(receipt, identity, kind="prior restored CPU server"):
            result = self.stop_owned(receipt, label="prior restored CPU server")
            append_jsonl(stop_path, {"stopped_at": now(), "ownership": receipt, "result": result})
            self.event("prior_restored_cpu_stopped_for_resume", pid=receipt["pid"])
        elif not stop_path.is_file():
            raise TrialError("prior restored CPU server exited without our resume-stop evidence")
        self.restore_needs_relaunch = True

    def query_capabilities(self) -> None:
        script = r'''import json,os,subprocess,sys
p=json.loads(sys.argv[1]); env=dict(os.environ); env["OCL_ICD_VENDORS"]=p["icd"]; result={}
for name,path in p["bins"].items():
 c=subprocess.run([path,"--help"],stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,env=env,timeout=30); result[name]={"returncode":c.returncode,"help":c.stdout}
print(json.dumps(result))'''
        value = self.remote_json(script, {"icd": REMOTE_ICD, "bins": {
            "bench": REMOTE_OPENCL_RUNTIME + "/bin/llama-bench",
            "server": REMOTE_OPENCL_RUNTIME + "/bin/llama-server"}}, timeout=75)
        if not isinstance(value, dict):
            raise TrialError("OpenCL runtime capability response is malformed")
        bench_help = value.get("bench", {}).get("help", "")
        server_help = value.get("server", {}).get("help", "")
        self.capabilities = {
            "bench_device": "--device" in bench_help,
            "server_device": "--device" in server_help,
            "server_fit_off": "--fit" in server_help and "[on|off]" in server_help,
            "bench_fit_default_off": "fit model to device memory" in bench_help and "default: off" in bench_help,
        }
        if not self.capabilities["bench_fit_default_off"]:
            raise TrialError("llama-bench help does not prove automatic fitting is off")
        atomic_json(self.trial / "runtime-capabilities.json",
                    {"captured_at": now(), "capabilities": self.capabilities, "raw": value})

    def launch_benchmark(self, variant: Mapping[str, Any], command: Sequence[str]) -> dict[str, Any]:
        key = str(variant["key"])
        receipt_path = REMOTE_RESULTS + f"/{key}-bench-launch.json"
        stdout_path = REMOTE_RESULTS + f"/{key}-llama-bench.json"
        stderr_path = REMOTE_RESULTS + f"/{key}-llama-bench.log"
        worker = r'''import datetime,json,os,pathlib,subprocess,sys,time
p=json.loads(sys.argv[1]); rec=pathlib.Path(p["receipt"]); out=pathlib.Path(p["stdout"]); err=pathlib.Path(p["stderr"])
def write(v):
 t=rec.with_name(rec.name+".tmp-"+str(os.getpid())); t.write_text(json.dumps(v,sort_keys=True)+"\n"); os.replace(t,rec)
try:
 with out.open("wb") as so,err.open("wb") as se:
  c=subprocess.Popen(p["argv"],stdin=subprocess.DEVNULL,stdout=so,stderr=se,start_new_session=True)
  base=pathlib.Path("/proc")/str(c.pid); end=time.monotonic()+2; argv=[]
  while time.monotonic()<end:
   if not base.exists(): break
   argv=[x.decode("utf-8","surrogateescape") for x in (base/"cmdline").read_bytes().split(b"\0") if x]
   if argv and argv[0].endswith("/llama-bench"): break
   time.sleep(.01)
  if not argv or not argv[0].endswith("/llama-bench"):
   rc=c.wait(); write({"schema_version":1,"kind":"tachyon_gpu_trial_bench","status":"failed","launch_id":p["launch_id"],"launch_argv":p["argv"],"returncode":rc,"error":"wrapper did not expose llama-bench","stdout":str(out),"stderr":str(err)}); raise SystemExit(0)
  s=(base/"stat").read_text(); t=s[s.rfind(")")+2:].split()
  v={"schema_version":1,"kind":"tachyon_gpu_trial_bench","status":"running","launch_id":p["launch_id"],"pid":c.pid,"start_ticks":int(t[19]),"argv":argv,"launch_argv":p["argv"],"started_at":p["at"],"stdout":str(out),"stderr":str(err)}; write(v)
  rc=c.wait(); v.update(status="complete" if rc==0 else "failed",returncode=rc,finished_at=datetime.datetime.now(datetime.timezone.utc).isoformat()); write(v)
except Exception as e:
 write({"schema_version":1,"kind":"tachyon_gpu_trial_bench","status":"failed","launch_id":p["launch_id"],"launch_argv":p["argv"],"error":str(e),"stdout":str(out),"stderr":str(err)})'''
        launcher = r'''import json,os,pathlib,subprocess,sys
p=json.loads(sys.argv[1]); rec=pathlib.Path(p["receipt"]); rec.parent.mkdir(parents=True,exist_ok=True)
c=subprocess.Popen(["python3","-c",p["worker"],json.dumps(p["job"],separators=(",",":"))],stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)
v={"schema_version":1,"kind":"tachyon_gpu_trial_bench","status":"launching","launch_id":p["job"]["launch_id"],"supervisor_pid":c.pid,"launch_argv":p["job"]["argv"],"started_at":p["job"]["at"],"stdout":p["job"]["stdout"],"stderr":p["job"]["stderr"]}
t=rec.with_name(rec.name+".tmp-"+str(os.getpid())); t.write_text(json.dumps(v,sort_keys=True)+"\n"); os.replace(t,rec); print(json.dumps(v))'''
        launch_id = f"{os.getpid()}-{time.time_ns()}"
        job = {"launch_id": launch_id, "argv": list(command), "receipt": receipt_path,
               "stdout": stdout_path, "stderr": stderr_path, "at": now()}
        self.remote_json(launcher, {"receipt": receipt_path, "worker": worker, "job": job})
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            receipt = self.remote_json_file(receipt_path)
            if receipt is not None and receipt.get("launch_id") == launch_id:
                validate_bench_receipt(receipt, command)
                return receipt
            time.sleep(0.2)
        raise TrialError(f"{key} benchmark did not publish an ownership receipt")

    def run_benchmark(self, variant: Mapping[str, Any]) -> dict[str, Any]:
        key = str(variant["key"])
        result_path = self.trial / f"{key}-result.json"
        if result_path.is_file():
            result = read_json(result_path)
            if isinstance(result, dict) and result.get("status") in {"complete", "failed", "timed_out"}:
                self.event("benchmark_preserved", variant=key, status=result["status"])
                return result
            raise TrialError(f"nonterminal local result exists for {key}")
        command = bench_command(variant, device_supported=self.capabilities.get("bench_device", False))
        remote_path = REMOTE_RESULTS + f"/{key}-bench-launch.json"
        receipt = self.remote_json_file(remote_path)
        if receipt is not None:
            validate_bench_receipt(receipt, command)
        if receipt is None or receipt.get("status") not in {"launching", "running", "complete", "failed"}:
            receipt = self.launch_benchmark(variant, command)
        started, dead_since = time.monotonic(), None
        while receipt.get("status") in {"launching", "running"}:
            self.check_interrupt()
            if time.monotonic() - started >= self.args.variant_timeout:
                if receipt.get("status") != "running":
                    raise TrialError(f"{key} timed out before publishing a child PID")
                stop = self.stop_owned(receipt, label=f"{key} timed-out benchmark")
                atomic_json(self.trial / f"{key}-timeout-stop.json",
                            {"stopped_at": now(), "ownership": receipt, "result": stop})
                receipt = dict(receipt, status="timed_out", timeout_seconds=self.args.variant_timeout)
                break
            if receipt.get("status") == "running":
                identity = self.remote_process(receipt["pid"])
                if identity.get("alive"):
                    verify_owned_process(receipt, identity, kind=f"{key} benchmark")
                    dead_since = None
                else:
                    dead_since = dead_since or time.monotonic()
                    if time.monotonic() - dead_since > 10:
                        raise TrialError(f"{key} exited without a terminal receipt")
            time.sleep(min(2, self.args.poll))
            updated = self.remote_json_file(remote_path)
            if updated is None:
                raise TrialError(f"{key} remote receipt disappeared")
            validate_bench_receipt(updated, command)
            receipt = updated
        atomic_json(self.trial / f"{key}-bench-launch.json", receipt)
        raw = log = b""
        for remote, local, is_raw in (
            (receipt.get("stdout"), self.trial / f"{key}-llama-bench.json", True),
            (receipt.get("stderr"), self.trial / f"{key}-llama-bench.log", False),
        ):
            if isinstance(remote, str):
                try:
                    data = self.remote_file(remote)
                    atomic_bytes(local, data)
                    if is_raw:
                        raw = data
                    else:
                        log = data
                except TrialError:
                    if receipt.get("status") == "complete":
                        raise
        status = "complete" if receipt.get("status") == "complete" and receipt.get("returncode") == 0 else str(receipt.get("status", "failed"))
        rates = {"prompt_tokens_per_second": None, "generation_tokens_per_second": None}
        if raw:
            try:
                rates = benchmark_rates(json.loads(raw))
            except (UnicodeDecodeError, json.JSONDecodeError):
                status = "failed"
        evidence = parse_offload_evidence(log.decode("utf-8", "replace"), int(variant["ngl"]))
        result = {
            "schema_version": 1, "variant": key, "backend": variant["backend"],
            "status": status, "returncode": receipt.get("returncode"), "command": command,
            "benchmark": {"prompt_tokens": 128, "generation_tokens": 64,
                          "repetitions": 3, "threads": variant["threads"],
                          "cpuset": variant["cpuset"]},
            **rates, "offload": evidence, "error": receipt.get("error"), "finished_at": now(),
        }
        atomic_json(result_path, result)
        self.event("benchmark_terminal", variant=key, status=status)
        return result

    def launch_server(self, *, backend: str, ngl: int, key: str) -> dict[str, Any]:
        command = server_command(
            backend=backend, ngl=ngl,
            device_supported=self.capabilities.get("server_device", False),
            fit_supported=self.capabilities.get("server_fit_off", False),
        )
        receipt_path = REMOTE_RESULTS + f"/{key}-server-launch.json"
        log_path = REMOTE_RESULTS + f"/{key}-server.log"
        script = r'''import json,os,pathlib,subprocess,sys,time
p=json.loads(sys.argv[1]); log=pathlib.Path(p["log"]); rec=pathlib.Path(p["receipt"]); log.parent.mkdir(parents=True,exist_ok=True); f=log.open("ab",buffering=0)
try:
 c=subprocess.Popen(p["argv"],stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True); base=pathlib.Path("/proc")/str(c.pid); end=time.monotonic()+2; argv=[]
 while time.monotonic()<end:
  if not base.exists(): break
  argv=[x.decode("utf-8","surrogateescape") for x in (base/"cmdline").read_bytes().split(b"\0") if x]
  if argv and argv[0].endswith("/llama-server"): break
  time.sleep(.01)
 if not argv or not argv[0].endswith("/llama-server"): print(json.dumps({"error":"server exited or wrapper did not exec"})); raise SystemExit(0)
 s=(base/"stat").read_text(); t=s[s.rfind(")")+2:].split(); v={"schema_version":1,"kind":"tachyon_gpu_trial_server","pid":c.pid,"start_ticks":int(t[19]),"argv":argv,"launch_argv":p["argv"],"backend":p["backend"],"ngl":p["ngl"],"alias":p["alias"],"launched_at":p["at"],"log":str(log)}
 tmp=rec.with_name(rec.name+".tmp-"+str(os.getpid())); tmp.write_text(json.dumps(v,sort_keys=True)+"\n"); os.replace(tmp,rec); print(json.dumps(v))
except Exception as e: print(json.dumps({"error":str(e)}))'''
        receipt = self.remote_json(script, {"argv": command, "receipt": receipt_path,
            "log": log_path, "backend": backend, "ngl": ngl,
            "alias": GPU_ALIAS if backend == "opencl" else CPU_ALIAS, "at": now()})
        if not isinstance(receipt, dict) or receipt.get("error"):
            raise VariantFailure(f"{key} server launch failed: {receipt}")
        if receipt.get("argv") != actual_exec_argv(command) or receipt.get("launch_argv") != command:
            raise TrialError(f"{key} server launch receipt argv differs")
        identity = self.remote_process(receipt["pid"])
        verify_owned_process(receipt, identity, kind=f"{key} server")
        receipt["remote_receipt"] = receipt_path
        atomic_json(self.trial / f"{key}-server-launch.json", receipt)
        append_jsonl(self.trial / f"{key}-server-launches.jsonl", receipt)
        return receipt

    def wait_server(self, receipt: Mapping[str, Any], alias: str) -> None:
        script = r'''import json,urllib.request
try:
 with urllib.request.urlopen("http://127.0.0.1:18085/v1/models",timeout=5) as r: print(json.dumps({"value":json.load(r)}))
except Exception as e: print(json.dumps({"error":str(e)}))'''
        deadline, latest = time.monotonic() + self.args.server_timeout, "not queried"
        while time.monotonic() < deadline:
            identity = self.remote_process(receipt["pid"])
            if not verify_owned_process(receipt, identity, kind=f"{alias} server"):
                raise VariantFailure(f"server {alias} exited during load")
            value = self.remote_json(script, {}, timeout=15)
            if isinstance(value, dict) and isinstance(value.get("value"), dict):
                rows = value["value"].get("data")
                models = [row.get("id") for row in rows or [] if isinstance(row, dict)]
                if models == [alias] or (alias in models and len(set(models)) == 1):
                    return
                latest = f"models={models!r}"
            else:
                latest = str(value.get("error") if isinstance(value, dict) else value)
            time.sleep(2)
        raise VariantFailure(f"server {alias} did not become ready: {latest}")

    def request_smoke(
        self,
        key: str,
        name: str,
        prompt: str,
        max_tokens: int,
        model_alias: str,
    ) -> dict[str, Any]:
        prefix = self.trial / f"{key}-{name}"
        paths = {
            "request": prefix.with_name(prefix.name + "-request.json"),
            "response": prefix.with_name(prefix.name + "-response.json"),
            "evaluation": prefix.with_name(prefix.name + "-evaluation.json"),
            "raw": prefix.with_name(prefix.name + "-raw-sse.txt"),
        }
        if all(path.is_file() for path in paths.values()):
            value = read_json(paths["evaluation"])
            if not isinstance(value, dict):
                raise TrialError(f"preserved {key}/{name} evaluation is malformed")
            return value
        if any(path.exists() for path in paths.values()):
            raise TrialError(f"partial {key}/{name} smoke evidence exists")
        payload = smoke_payload(prompt, max_tokens, model_alias)
        atomic_json(paths["request"], payload)
        request = Request(
            self.args.endpoint.rstrip("/") + "/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}, method="POST",
        )
        raw_lines: list[bytes] = []
        chunks: list[dict[str, Any]] = []
        content: list[str] = []
        reasoning: list[str] = []
        started, ttft = time.monotonic(), None
        try:
            with urlopen(request, timeout=self.args.request_timeout) as response:
                for raw in response:
                    raw_lines.append(raw)
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
                        for field, target in (("content", content), ("reasoning_content", reasoning)):
                            text = delta.get(field) if isinstance(delta, dict) else None
                            if isinstance(text, str):
                                if text and ttft is None:
                                    ttft = time.monotonic() - started
                                target.append(text)
        except (OSError, HTTPError, URLError, json.JSONDecodeError) as exc:
            atomic_bytes(paths["raw"], b"".join(raw_lines))
            raise VariantFailure(f"{key}/{name} request failed: {exc}") from exc
        wall = time.monotonic() - started
        atomic_bytes(paths["raw"], b"".join(raw_lines))
        last = chunks[-1] if chunks else {}
        usage: Mapping[str, Any] = {}
        timings: Mapping[str, Any] = {}
        finish_reason: str | None = None
        for chunk in reversed(chunks):
            if not usage and isinstance(chunk.get("usage"), dict):
                usage = chunk["usage"]
            if not timings and isinstance(chunk.get("timings"), dict):
                timings = chunk["timings"]
            if finish_reason is None:
                for choice in chunk.get("choices", []):
                    candidate = choice.get("finish_reason") if isinstance(choice, dict) else None
                    if isinstance(candidate, str):
                        finish_reason = candidate
                        break
        response_value = {
            "content": "".join(content), "reasoning_content": "".join(reasoning),
            "usage": usage, "native_timings": timings, "wall_seconds": wall,
            "ttft_seconds": ttft, "chunk_count": len(chunks), "last": last,
            "finish_reason": finish_reason,
            "final_character_count": len("".join(content)),
            "reasoning_character_count": len("".join(reasoning)),
        }
        atomic_json(paths["response"], response_value)
        if name == "sensor":
            try:
                parsed = json.loads(response_value["content"])
            except json.JSONDecodeError:
                parsed = None
            passed = (isinstance(parsed, dict) and set(parsed) == {"answer", "explanation"}
                      and parsed.get("answer") == 21 and isinstance(parsed.get("explanation"), str))
            expected: Any = {"answer": 21, "keys": ["answer", "explanation"]}
        else:
            passed = response_value["content"].strip() == "tachyon-opencl-ok"
            expected = "tachyon-opencl-ok"
        evaluation = {
            "check": name, "passed": passed, "expected": expected,
            "observed_content": response_value["content"],
            "requested_reasoning_budget_tokens": payload["reasoning_budget_tokens"],
            "actual_reasoning_character_count": response_value["reasoning_character_count"],
            "actual_final_character_count": response_value["final_character_count"],
            "finish_reason": finish_reason,
            "usage": usage,
            "completion_tokens": usage.get("completion_tokens"),
            "native_generation_tokens": timings.get("predicted_n"),
            "native_generation_ms": timings.get("predicted_ms"),
            "native_generation_tokens_per_second": timings.get("predicted_per_second"),
            "wall_seconds": wall, "ttft_seconds": ttft, "quality_claim": False,
        }
        atomic_json(paths["evaluation"], evaluation)
        return evaluation

    def run_fast_probe_pair(self, key: str, model_alias: str) -> dict[str, Any]:
        """Run the same bounded fast-mode checks on CPU or partial OpenCL."""
        path = self.trial / f"{key}-probe-result.json"
        if path.is_file():
            value = read_json(path)
            if isinstance(value, dict) and value.get("status") in {"complete", "failed"}:
                return value
            raise TrialError(f"nonterminal probe result exists for {key}")
        try:
            sensor = self.request_smoke(
                key, "sensor", SENSOR_PROMPT, 512, model_alias
            )
            exact = self.request_smoke(
                key, "exact-prose", EXACT_PROMPT, 128, model_alias
            )
            value = {
                "variant": key,
                "mode": "measured_fast_reasoning_budget_zero",
                "requested_reasoning_budget_tokens": 0,
                "status": "complete" if sensor["passed"] and exact["passed"] else "failed",
                "sensor_passed": sensor["passed"],
                "deterministic_prose_passed": exact["passed"],
                "sensor_evaluation": sensor,
                "exact_prose_evaluation": exact,
                "quality_claim": False,
                "interpretation": "two fast-mode smoke checks only; not a public-benchmark quality result",
            }
        except VariantFailure as exc:
            value = {
                "variant": key,
                "mode": "measured_fast_reasoning_budget_zero",
                "requested_reasoning_budget_tokens": 0,
                "status": "failed",
                "error": str(exc),
                "quality_claim": False,
            }
        atomic_json(path, value)
        self.event("fast_probe_terminal", variant=key, status=value["status"])
        return value

    def probe_cpu_baseline(self, cpu_state: Mapping[str, Any]) -> dict[str, Any]:
        receipt = self.original_cpu_receipt(cpu_state)
        identity = self.remote_process(receipt["pid"])
        if not verify_owned_process(receipt, identity, kind="CPU fast-probe baseline server"):
            raise TrialError("CPU fast-probe baseline server exited before handoff")
        return self.run_fast_probe_pair("cpu4-fast-baseline", CPU_ALIAS)

    def probe_gpu_variant(self, result: Mapping[str, Any], variant: Mapping[str, Any]) -> dict[str, Any]:
        key = str(variant["key"])
        path = self.trial / f"{key}-probe-result.json"
        if path.is_file():
            value = read_json(path)
            if isinstance(value, dict) and value.get("status") in {"complete", "failed"}:
                return value
            raise TrialError(f"nonterminal probe result exists for {key}")
        if result.get("status") != "complete" or result.get("offload", {}).get("gpu_compute_proven") is not True:
            value = {"variant": key, "status": "failed",
                     "error": "benchmark did not prove OpenCL layer offload", "quality_claim": False}
            atomic_json(path, value)
            return value
        receipt: dict[str, Any] | None = None
        try:
            receipt = self.launch_server(backend="opencl", ngl=int(variant["ngl"]), key=key)
            self.wait_server(receipt, GPU_ALIAS)
            value = self.run_fast_probe_pair(key, GPU_ALIAS)
        except VariantFailure as exc:
            value = {
                "variant": key, "mode": "measured_fast_reasoning_budget_zero",
                "requested_reasoning_budget_tokens": 0,
                "status": "failed", "error": str(exc), "quality_claim": False,
            }
        finally:
            if receipt is not None:
                stop = self.stop_owned(receipt, label=f"{key} GPU server")
                atomic_json(self.trial / f"{key}-server-stop.json",
                            {"stopped_at": now(), "ownership": receipt, "result": stop})
                atomic_bytes(self.trial / f"{key}-server.log", self.remote_file(receipt["log"]))
        # run_fast_probe_pair writes this for request/evaluation outcomes.  A
        # launch/load failure reaches here before that helper and needs its own
        # terminal receipt.
        if not path.is_file():
            atomic_json(path, value)
        self.event("gpu_probe_terminal", variant=key, status=value["status"])
        return value

    def restore_cpu(self) -> dict[str, Any]:
        path = self.trial / "cpu-restore-server-launch.json"
        if path.is_file():
            receipt = read_json(path)
            if not isinstance(receipt, dict):
                raise TrialError("CPU restore receipt is malformed")
            identity = self.remote_process(receipt["pid"])
            if verify_owned_process(receipt, identity, kind="restored CPU server"):
                self.wait_server(receipt, CPU_ALIAS)
                return receipt
            if not self.restore_needs_relaunch:
                raise TrialError("recorded restored CPU server exited without owned resume-stop evidence")
        receipt = self.launch_server(backend="cpu", ngl=0, key="cpu-restore")
        validate_cpu_server_argv(receipt.get("argv", []))
        self.wait_server(receipt, CPU_ALIAS)
        # launch_server already wrote this path; rewrite only to make the intended
        # durable name explicit and identical.
        atomic_json(path, receipt)
        atomic_json(self.trial / "current-launch.json", {
            "selected_at": now(), "default_restored": True,
            "reason": "CPU4 endpoint has proven public-screen quality; GPU checks are smoke-only",
            "server": receipt,
        })
        self.event("cpu_server_restored", pid=receipt["pid"])
        return receipt

    def consolidate(
        self,
        results: Sequence[Mapping[str, Any]],
        probes: Mapping[str, Mapping[str, Any]],
        cpu_fast_baseline: Mapping[str, Any],
    ) -> None:
        rows: list[dict[str, Any]] = []
        for result in results:
            probe = probes.get(str(result["variant"]), {})
            offload = result.get("offload", {})
            rows.append({
                "variant": result["variant"], "backend": result["backend"],
                "status": result["status"],
                "prompt_tokens_per_second": result.get("prompt_tokens_per_second"),
                "generation_tokens_per_second": result.get("generation_tokens_per_second"),
                "requested_gpu_layers": offload.get("requested_gpu_layers"),
                "reported_offloaded_layers": offload.get("reported_offloaded_layers"),
                "offload_classification": offload.get("classification"),
                "sensor_passed": probe.get("sensor_passed"),
                "deterministic_prose_passed": probe.get("deterministic_prose_passed"),
                "quality_claim": False, "error": result.get("error") or probe.get("error"),
            })
        atomic_json(self.trial / "summary.json", {
            "schema_version": 1, "kind": "tachyon_cpu8_and_partial_opencl_trial",
            "created_at": now(),
            "model": {"key": "lfm25-26b", "path": REMOTE_MODEL, "sha256": MODEL_SHA256},
            "baseline": {"source": str(self.study / "lfm25-26b-llama-bench.json"),
                         "configuration": "CPU4 taskset 4-7; preserved, not rerun"},
            "fast_probe_baseline": dict(cpu_fast_baseline),
            "rows": rows, "quality_claim": False,
            "interpretation": "CPU8 and partial-OpenCL performance trial; CPU/GPU fast-mode correctness is matched smoke-only evidence and is not a public-benchmark quality result",
            "default_endpoint": "restored CPU4 LFM2.5 server",
        })
        output = io.StringIO(newline="")
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        (self.trial / "summary.csv").write_text(output.getvalue(), encoding="utf-8")

    def run(self) -> int:
        self.acquire_lock()
        for signum in (signal.SIGINT, signal.SIGTERM):
            signal.signal(signum, self._handle_signal)
        if self.status_path.is_file():
            existing = read_json(self.status_path)
            if isinstance(existing, dict) and existing.get("status") == "complete":
                self.event("trial_already_complete")
                return 0
        self.state = {
            "schema_version": 1, "kind": "tachyon_cpu8_and_partial_opencl_trial",
            "status": "waiting_for_cpu_study", "created_at": now(),
            "study": str(self.study), "trial": str(self.trial),
            "variants": [row["key"] for row in VARIANTS],
        }
        self.save()
        failure: BaseException | None = None
        try:
            cpu_state = self.wait_for_cpu_study()
            self.verify_provenance(cpu_state)
            self.save(status="taking_ownership")
            self.initialize_remote()
            self.save(status="cpu_fast_baseline")
            cpu_fast_baseline = self.probe_cpu_baseline(cpu_state)
            self.stop_initial_cpu(cpu_state)
            self.stop_prior_restore_for_resume()
            self.query_capabilities()
            results: list[dict[str, Any]] = []
            probes: dict[str, dict[str, Any]] = {}
            for variant in VARIANTS:
                self.save(status="benchmarking", current_variant=variant["key"])
                result = self.run_benchmark(variant)
                results.append(result)
                if variant["backend"] == "opencl":
                    self.save(status="probing", current_variant=variant["key"])
                    probes[str(variant["key"])] = self.probe_gpu_variant(result, variant)
            self.consolidate(results, probes, cpu_fast_baseline)
        except BaseException as exc:
            failure = exc
        finally:
            if self.cpu_was_stopped:
                try:
                    self.save(status="restoring_cpu", current_variant=None)
                    self.restore_cpu()
                except BaseException as restore_exc:
                    failure = restore_exc if failure is None else TrialError(
                        f"{failure}; CPU restore also failed: {restore_exc}")
        if failure is not None:
            self.save(status="failed", error=str(failure), finished_at=now())
            self.event("trial_failed", error=str(failure))
            print(f"tachyon_gpu_trial: {failure}", file=sys.stderr)
            return 1
        self.save(status="complete", error=None, current_variant=None, finished_at=now())
        self.event("trial_complete")
        return 0


def parse_args(arguments: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=DEFAULT_STUDY)
    parser.add_argument("--trial-dir", type=Path, default=DEFAULT_TRIAL)
    parser.add_argument("--remote", default="akovanda@tachyon")
    parser.add_argument("--endpoint", default=ENDPOINT)
    parser.add_argument("--wait-timeout", type=float, default=86_400)
    parser.add_argument("--poll", type=float, default=10)
    parser.add_argument("--variant-timeout", type=float, default=600)
    parser.add_argument("--server-timeout", type=float, default=600)
    parser.add_argument("--request-timeout", type=float, default=900)
    args = parser.parse_args(arguments)
    for name in ("wait_timeout", "poll", "variant_timeout", "server_timeout", "request_timeout"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    return args


def main(arguments: Iterable[str] = sys.argv[1:]) -> int:
    return Controller(parse_args(arguments)).run()


if __name__ == "__main__":
    raise SystemExit(main())
