#!/usr/bin/env python3
"""Durable serial supervisor for the frozen public-model expansion.

Every local profile gets an owned GPU window, an alias/transport preflight, the
same frozen 64 scored cases, clean restoration, and isolated official grading.
Windows may expire at gpu_window's attested 24-hour limit; attempts remain
durable and the same pinned profile resumes in a fresh window.
"""

from __future__ import annotations

import argparse
import contextlib
from dataclasses import asdict
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import stat
import subprocess
import sys
import time
from types import SimpleNamespace
from typing import Any, Iterable, Mapping, Sequence
import uuid

if __package__:
    from . import gpu_window, llama_trial, specialist_study_trial as ownership
    from .signal_exec import unblocked_exec_command
else:
    import gpu_window
    import llama_trial
    import specialist_study_trial as ownership
    from signal_exec import unblocked_exec_command


ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "runtime"
CONFIG_ROOT = RUNTIME / "public-model-expansion-20261024-v1"
CATALOG = CONFIG_ROOT / "profile-catalog.json"
SMOKE_PLAN = CONFIG_ROOT / "transport-smoke-plan.json"
DEFAULT_STUDY = ROOT / "results/public-model-expansion-20261024-v1"
FROZEN_STUDY = ROOT / "results/public-benchmark-pilot-20261023-v1"
MANIFEST = FROZEN_STUDY / "manifest.json"
MANIFEST_SHA256 = "758607e95961b95ca98bdb2bdf11342b961239973dfcff73ff424c3a69603039"
EXTENDED_RUNNER = RUNTIME / "public_benchmark_extended.py"
BASE_RUNNER = RUNTIME / "public_benchmark_pilot.py"
GPU_WINDOW = RUNTIME / "gpu_window.py"
MODEL_SESSION = RUNTIME / "model_session.py"
VENDOR = ROOT / "vendor/public-benchmarks"
OSS_HANDOFF = Path("oss-repair/handoff.json")
TERMINAL = {"completed", "truncated"}
RETRYABLE = {"api_failure", "cancelled", "incomplete", "not_started"}
PROFILE_KEYS = {
    "oss20b_repaired",
    "flash_next_iq3_s",
    "flash_next_coder_iq1_m",
    "kat_coder_v25_iq3_xxs",
    "bonsai2_27b_pq2_0",
    "bonsai2_27b_ptq1_0",
    "qwen36_35b_a3b_ud_q4_k_xl",
    "qwen35_122b_a10b_ud_q4_k_xl_native",
    "qwen35_122b_a10b_ud_q4_k_xl_strata",
}
PINNED_SOURCES = (
    EXTENDED_RUNNER,
    BASE_RUNNER,
    GPU_WINDOW,
    MODEL_SESSION,
    RUNTIME / "llama_trial.py",
    RUNTIME / "strata_trial.py",
    RUNTIME / "specialist_study_trial.py",
    RUNTIME / "signal_exec.py",
    Path(__file__).resolve(),
)


class ExpansionError(RuntimeError):
    pass


class SafetyStop(ExpansionError):
    """A cleanup, ownership, or source-integrity failure forbids queue advance."""


class ExpansionInterrupted(BaseException):
    pass


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            value.update(block)
    return value.hexdigest()


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ExpansionError(f"unreadable or invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ExpansionError(f"JSON must be an object: {path}")
    return value


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    data = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
    try:
        with temporary.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()


def append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def verify_manifest(path: Path = MANIFEST) -> dict[str, Any]:
    if digest(path) != MANIFEST_SHA256:
        raise ExpansionError("frozen benchmark manifest SHA256 mismatch")
    manifest = read_json(path)
    cases = manifest.get("cases")
    counts = {"humanevalplus": 0, "ifeval": 0}
    if not isinstance(cases, list):
        raise ExpansionError("frozen manifest cases must be a list")
    for row in cases:
        if isinstance(row, dict) and row.get("suite") in counts:
            counts[str(row["suite"])] += 1
    if (
        manifest.get("selection_seed") != 20261023
        or manifest.get("selected_count_per_suite") != 32
        or len(cases) != 64
        or counts != {"humanevalplus": 32, "ifeval": 32}
    ):
        raise ExpansionError("expansion requires the frozen seed-20261023 32+32 manifest")
    return manifest


def _context_from_config(path: Path, backend: str) -> tuple[int, str]:
    raw = read_json(path)
    alias = raw.get("model_name")
    if not isinstance(alias, str) or not alias:
        raise ExpansionError(f"config lacks model_name: {path}")
    if backend == "llama":
        context = raw.get("context")
    else:
        args = raw.get("args")
        if not isinstance(args, list):
            raise ExpansionError(f"Strata config lacks args: {path}")
        values = [args[index + 1] for index, value in enumerate(args[:-1]) if value == "--max-context"]
        if len(values) != 1:
            raise ExpansionError(f"Strata config needs one --max-context: {path}")
        try:
            context = int(values[0])
        except (TypeError, ValueError) as exc:
            raise ExpansionError(f"invalid Strata --max-context: {path}") from exc
    if context != 73728:
        raise ExpansionError(f"profile config is not exact context 73728: {path}")
    return int(context), alias


def load_catalog(path: Path = CATALOG) -> dict[str, Any]:
    catalog = read_json(path)
    if catalog.get("schema_version") != 1 or catalog.get("review_status") != "approved":
        raise ExpansionError("profile catalog must be schema 1 and root-approved")
    if catalog.get("manifest_sha256") != MANIFEST_SHA256:
        raise ExpansionError("profile catalog is bound to a different manifest")
    profiles = catalog.get("profiles")
    if not isinstance(profiles, list) or len(profiles) != 9:
        raise ExpansionError("profile catalog must contain exactly nine new profiles")
    keys = [row.get("key") for row in profiles if isinstance(row, dict)]
    if set(keys) != PROFILE_KEYS or len(set(keys)) != len(keys):
        raise ExpansionError("profile catalog keys do not match the approved nine-profile roster")
    for row in profiles:
        if not isinstance(row, dict):
            raise ExpansionError("catalog profile must be an object")
        controls = row.get("controls")
        if not isinstance(controls, dict) or (
            controls.get("context_tokens") != 73728
            or controls.get("max_output_tokens") != 65536
            or controls.get("request_timeout_seconds") != 21600
            or controls.get("seed") != 20261023
        ):
            raise ExpansionError(f"profile {row.get('key')} does not pin 72K/64K/21600/seed controls")
        if row.get("managed_externally") is True:
            continue
        backend = row.get("runtime_backend")
        if backend not in {"llama", "strata"}:
            raise ExpansionError(f"invalid runtime backend for {row.get('key')}")
        candidates = row.get("config_candidates")
        if not isinstance(candidates, list) or not candidates:
            raise ExpansionError(f"profile {row.get('key')} has no placement config")
        for candidate in candidates:
            if not isinstance(candidate, dict) or not isinstance(candidate.get("path"), str):
                raise ExpansionError(f"invalid placement candidate for {row.get('key')}")
            config_path = Path(candidate["path"]).expanduser().resolve()
            context, alias = _context_from_config(config_path, backend)
            if context != 73728 or alias != row.get("model_name"):
                raise ExpansionError(f"profile/config identity mismatch for {row.get('key')}")
    return catalog


def endpoint_for_backend(backend: str) -> str:
    if backend == "llama":
        return "http://127.0.0.1:18081/v1"
    if backend == "strata":
        return "http://127.0.0.1:18080/v1"
    raise ExpansionError(f"unsupported backend: {backend}")


def profile_request_args(profile: Mapping[str, Any]) -> list[str]:
    controls = profile["controls"]
    options = profile.get("request_options")
    if not isinstance(options, dict):
        raise ExpansionError(f"profile {profile.get('key')} request_options must be an object")
    allowed = {
        "reasoning_effort": "--reasoning-effort",
        "temperature": "--temperature",
        "top_p": "--top-p",
        "top_k": "--top-k",
        "min_p": "--min-p",
        "presence_penalty": "--presence-penalty",
        "repeat_penalty": "--repeat-penalty",
        "chat_template_thinking": "--chat-template-thinking",
        "reasoning_budget_tokens": "--reasoning-budget-tokens",
        "thinking_budget_tokens": "--thinking-budget-tokens",
    }
    unknown = sorted(set(options) - set(allowed))
    if unknown:
        raise ExpansionError("non-allowlisted profile request option(s): " + ", ".join(unknown))
    argv = [
        "--max-tokens", str(controls["max_output_tokens"]),
        "--request-timeout", str(controls["request_timeout_seconds"]),
        "--seed", str(controls["seed"]),
    ]
    for key, flag in allowed.items():
        if key in options:
            value = options[key]
            if isinstance(value, bool) or isinstance(value, (dict, list)) or value is None:
                raise ExpansionError(f"invalid scalar request option {key}")
            argv.extend((flag, str(value).lower() if isinstance(value, bool) else str(value)))
    return argv


def process_identity(pid: int) -> dict[str, Any]:
    try:
        raw_stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        close = raw_stat.rfind(")")
        fields = raw_stat[close + 2 :].split()
        start_ticks = int(fields[19])
        argv = [part.decode("utf-8", "surrogateescape") for part in Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0") if part]
        pgid = os.getpgid(pid)
    except (OSError, ValueError, IndexError) as exc:
        raise ExpansionError(f"cannot capture process identity for PID {pid}") from exc
    return {"pid": pid, "pgid": pgid, "start_ticks": start_ticks, "argv": argv}


def wait_process_identity(pid: int, expected_fragment: str, timeout: float = 5.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    last: dict[str, Any] | None = None
    while time.monotonic() < deadline:
        with contextlib.suppress(ExpansionError):
            last = process_identity(pid)
            # signal_exec is an exec trampoline.  Capturing it would retain the
            # same PID/start ticks but the argv would change before a later
            # signal check, correctly (but needlessly) failing ownership.
            trampoline = str(RUNTIME / "signal_exec.py")
            if expected_fragment in last["argv"] and trampoline not in last["argv"]:
                return last
        time.sleep(0.05)
    raise ExpansionError(f"PID {pid} never exposed expected argv fragment {expected_fragment!r}; last={last}")


def validate_process_identity(identity: Mapping[str, Any]) -> bool:
    try:
        current = process_identity(int(identity["pid"]))
    except ExpansionError:
        return False
    return (
        current["start_ticks"] == identity.get("start_ticks")
        and current["argv"] == identity.get("argv")
        and current["pgid"] == identity.get("pgid") == identity.get("pid")
    )


def terminate_owned_process(process: subprocess.Popen[Any], identity: Mapping[str, Any]) -> bool:
    if process.poll() is not None:
        return True
    if not validate_process_identity(identity):
        raise SafetyStop("refusing to signal process after PID/start-ticks/argv validation failed")
    os.killpg(int(identity["pgid"]), signal.SIGTERM)
    try:
        process.wait(timeout=45)
    except subprocess.TimeoutExpired as exc:
        raise SafetyStop("owned process did not stop after SIGTERM; SIGKILL was not sent") from exc
    return True


def validate_grade_coverage(summary: Mapping[str, Any]) -> None:
    coverage = summary.get("coverage")
    evalplus = summary.get("evalplus")
    ifeval = summary.get("ifeval")
    if (
        summary.get("status") != "completed"
        or not isinstance(coverage, dict)
        or coverage.get("total") != 64
        or coverage.get("completed", 0) + coverage.get("truncated", 0) != 64
        or not isinstance(evalplus, dict)
        or evalplus.get("status") != "completed"
        or evalplus.get("task_count") != 32
        or not isinstance(ifeval, dict)
        or ifeval.get("status") != "completed"
        or ifeval.get("strict", {}).get("prompt_count") != 32
        or ifeval.get("loose", {}).get("prompt_count") != 32
    ):
        raise ExpansionError("official grading summary does not cover the frozen 32+32 cases")


def _read_events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        with contextlib.suppress(json.JSONDecodeError):
            value = json.loads(line)
            if isinstance(value, dict):
                rows.append(value)
    return rows


def transport_smoke_evidence(events_path: Path) -> dict[str, Any]:
    """Require a terminal API stream while ignoring answer semantics."""
    steps = [row for row in _read_events(events_path) if row.get("event") == "step_complete"]
    if len(steps) != 1 or not isinstance(steps[0].get("result"), str):
        raise ExpansionError("transport smoke lifecycle is missing its single result")
    result_path = Path(steps[0]["result"])
    try:
        rows = [json.loads(line) for line in result_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ExpansionError("transport smoke result is unreadable or invalid") from exc
    if len(rows) != 1 or not isinstance(rows[0], dict):
        raise ExpansionError("transport smoke must contain exactly one result row")
    result = rows[0]
    response = result.get("response")
    request = result.get("request")
    if (
        result.get("error") is not None
        or not isinstance(response, dict)
        or response.get("stream_complete") is not True
        or not isinstance(response.get("done_reason"), str)
        or not response.get("done_reason")
        or not isinstance(request, dict)
        or request.get("output_token_limit") != 32
    ):
        raise ExpansionError("transport smoke did not produce a terminal 32-token API stream")
    evaluation = result.get("evaluation")
    return {
        "result": str(result_path),
        "done_reason": response["done_reason"],
        "stream_complete": True,
        "error": None,
        "semantic_passed": evaluation.get("passed") if isinstance(evaluation, dict) else None,
        "semantic_detail": evaluation.get("detail") if isinstance(evaluation, dict) else None,
    }


def case_coverage(output: Path) -> dict[str, Any] | None:
    path = output / "status.json"
    return read_json(path) if path.exists() else None


def full_coverage(output: Path) -> bool:
    status = case_coverage(output)
    if status is None or not isinstance(status.get("counts"), dict):
        return False
    counts = status["counts"]
    return counts.get("total") == 64 and counts.get("completed", 0) + counts.get("truncated", 0) == 64


def external_gate(study: Path) -> tuple[bool, dict[str, Any] | None, str]:
    path = study / OSS_HANDOFF
    if not path.exists():
        return False, None, "OSS repair handoff has not been published"
    value = read_json(path)
    gpu = value.get("gpu")
    if not isinstance(gpu, dict) or gpu.get("state") != "clear":
        return False, value, "OSS repair still owns or has not cleared the GPU"
    if value.get("adoptable") is not True or value.get("full64_official_grading_complete") is not True:
        return False, value, "OSS repaired full64 run and official grading are not yet adoptable"
    return True, value, "ready"


def grade_command(image: str, study: Path, output: Path, model_key: str, manifest: Mapping[str, Any], runner: Path) -> list[str]:
    return [
        "docker", "run", "--rm", "--name", f"moe-public-expand-grade-{model_key}-{os.getpid()}",
        "--network", "none", "--read-only", "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
        "--pids-limit", "256", "--cpus", "4", "--memory", "4g", "--shm-size", "256m",
        "--user", f"{os.getuid()}:{os.getgid()}", "--tmpfs", "/tmp:rw,nosuid,size=2g",
        "--env", "HOME=/tmp", "--env", "XDG_CACHE_HOME=/tmp/cache", "--env", "PYTHONDONTWRITEBYTECODE=1",
        "--env", "OMP_NUM_THREADS=1", "--env", "OPENBLAS_NUM_THREADS=1",
        "--mount", f"type=bind,src={runner},dst=/app/public_benchmark_pilot.py,readonly",
        "--mount", f"type=bind,src={VENDOR},dst={VENDOR},readonly",
        "--mount", f"type=bind,src={FROZEN_STUDY},dst={FROZEN_STUDY},readonly",
        "--mount", f"type=bind,src={study},dst={study},readonly",
        "--mount", f"type=bind,src={output},dst={output}", image,
        "python", "/app/public_benchmark_pilot.py", "grade", "--manifest", str(MANIFEST),
        "--output-dir", str(output), "--evalplus-root", str(manifest["sources"]["evalplus"]["root"]),
        "--ifeval-root", str(Path(manifest["sources"]["ifeval"]["root"]) / "instruction_following_eval"),
    ]


def profile_binding(profile: Mapping[str, Any], config: Path, runner: Path) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "created_at": now(),
        "manifest_sha256": MANIFEST_SHA256,
        "profile_key": profile["key"],
        "profile_sha256": canonical_hash(profile),
        "runtime_backend": profile["runtime_backend"],
        "model_name": profile["model_name"],
        "config": {"path": str(config), "sha256": digest(config)},
        "runner": {"path": str(runner), "sha256": digest(runner)},
        "controls": profile["controls"],
        "request_options": profile["request_options"],
    }


def ensure_profile_binding(path: Path, expected: Mapping[str, Any]) -> dict[str, Any]:
    if path.exists():
        existing = read_json(path)
        comparable = ("manifest_sha256", "profile_key", "profile_sha256", "runtime_backend", "model_name", "config", "runner", "controls", "request_options")
        if any(existing.get(key) != expected.get(key) for key in comparable):
            raise ExpansionError(f"profile output is pinned to a different executed profile: {path}")
        return existing
    atomic_json(path, expected)
    return dict(expected)


class Supervisor:
    def __init__(self, args: argparse.Namespace, catalog: Mapping[str, Any], manifest: Mapping[str, Any]) -> None:
        self.args = args
        self.study = args.study_dir.resolve()
        self.catalog = dict(catalog)
        self.manifest = dict(manifest)
        self.catalog_sha256 = digest(args.catalog)
        self.received_signal: int | None = None
        self.artifacts = self.study / "artifacts"
        self.runner = self.artifacts / "public_benchmark_extended.py"
        self.grade_runner = self.artifacts / "public_benchmark_pilot.py"
        self.smoke_plan = self.artifacts / "transport-smoke-plan.json"
        status_path = self.study / "supervisor-status.json"
        if status_path.exists():
            self.state = read_json(status_path)
            if self.state.get("catalog_sha256") != self.catalog_sha256 or self.state.get("manifest_sha256") != MANIFEST_SHA256:
                raise ExpansionError("existing supervisor state has different catalog or manifest pins")
        else:
            self.state = {
                "schema_version": 1,
                "status": "starting",
                "started_at": now(),
                "supervisor_pid": os.getpid(),
                "manifest_sha256": MANIFEST_SHA256,
                "catalog_sha256": self.catalog_sha256,
                "current_model": None,
                "model_order": [row["key"] for row in self.catalog["profiles"]],
                "models": {},
            }
            for profile in self.catalog["profiles"]:
                output = self.study / ("oss-repair" if profile.get("managed_externally") else f"profiles/{profile['key']}")
                self.state["models"][profile["key"]] = {
                    "key": profile["key"], "display_name": profile["display_name"],
                    "model_name": profile["model_name"], "runtime_backend": profile["runtime_backend"],
                    "state": "waiting", "output_dir": str(output), "profile_binding": None,
                    "preflight": None, "run_status": None, "grading_summary": None, "error": None,
                }

    def save(self, **fields: Any) -> None:
        self.state.update(fields)
        self.state["updated_at"] = now()
        atomic_json(self.study / "supervisor-status.json", self.state)

    def event(self, event: str, **fields: Any) -> None:
        row = {"at": now(), "event": event, **fields}
        append_jsonl(self.study / "supervisor-events.jsonl", row)
        print(json.dumps(row, separators=(",", ":")), flush=True)

    def signal_handler(self, signum: int, _frame: Any) -> None:
        self.received_signal = signum

    def check_signal(self) -> None:
        if self.received_signal is not None:
            raise ExpansionInterrupted(f"signal {self.received_signal}")

    def prepare_artifacts(self) -> None:
        self.artifacts.mkdir(parents=True, exist_ok=True)
        sources = {
            "public_benchmark_extended.py": EXTENDED_RUNNER,
            "public_benchmark_pilot.py": BASE_RUNNER,
            "public_model_expansion.py": Path(__file__).resolve(),
            "transport-smoke-plan.json": SMOKE_PLAN,
        }
        config_dir = self.artifacts / "configs"
        config_dir.mkdir(exist_ok=True)
        for profile in self.catalog["profiles"]:
            for candidate in profile.get("config_candidates", []):
                source = Path(candidate["path"]).resolve()
                sources[f"configs/{source.name}"] = source
        receipt: dict[str, Any] = {"schema_version": 1, "captured_at": now(), "files": {}}
        for name, source in sources.items():
            target = self.artifacts / name
            if target.exists():
                if digest(target) != digest(source):
                    raise ExpansionError(f"artifact snapshot differs from current pinned source: {target}")
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
                with target.open("rb") as handle:
                    os.fsync(handle.fileno())
            receipt["files"][name] = {"source": str(source), "snapshot": str(target), "sha256": digest(target), "size": target.stat().st_size}
        pins = set(PINNED_SOURCES) | {self.args.catalog.resolve(), SMOKE_PLAN.resolve()}
        receipt["runtime_pins"] = {str(path): digest(path) for path in sorted(pins)}
        receipt_path = self.artifacts / "source-pins.json"
        if receipt_path.exists():
            existing = read_json(receipt_path)
            if existing.get("files") != receipt["files"] or existing.get("runtime_pins") != receipt["runtime_pins"]:
                raise ExpansionError("source pin receipt changed across resume")
        else:
            atomic_json(receipt_path, receipt)

    def verify_pins(self) -> None:
        receipt = read_json(self.artifacts / "source-pins.json")
        for filename, expected in receipt["runtime_pins"].items():
            if digest(Path(filename)) != expected:
                raise SafetyStop(f"pinned runtime changed before model boundary: {filename}")

    def record_controls(self) -> None:
        controls = self.study / "controls"
        controls.mkdir(exist_ok=True)
        references = {
            "lfm25_original_control": FROZEN_STUDY / "lfm25/grading/summary.json",
            "oss20b_original_medium_baseline": FROZEN_STUDY / "oss20b/grading/summary.json",
        }
        for key, source in references.items():
            receipt = {"schema_version": 1, "role": key, "source": str(source), "exists": source.is_file()}
            if source.is_file():
                receipt.update({"sha256": digest(source), "size": source.stat().st_size})
            path = controls / f"{key}.json"
            if not path.exists():
                atomic_json(path, receipt)

    def trigger_report_update(self) -> None:
        """Start a best-effort report refresh without gating inference."""
        script = RUNTIME / "public_model_expansion_report.py"
        command = [
            self.args.python, str(script), "--study-dir", str(self.study),
            "--output-dir", str(self.study / "report"), "--catalog", str(self.args.catalog),
            "--manifest", str(MANIFEST),
        ]
        self.state["report_hook"] = {
            "argv": command,
            "policy": "best_effort_detached; never gates inference or grading",
        }
        if not script.is_file():
            self.state["report_hook"]["last_status"] = "reporter_not_yet_available"
            self.save()
            return
        try:
            log = (self.study / "report-hook.log").open("ab", buffering=0)
            try:
                process = subprocess.Popen(
                    unblocked_exec_command(command), cwd=ROOT, stdin=subprocess.DEVNULL,
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True, close_fds=True,
                )
            finally:
                log.close()
            report_process: dict[str, Any] = {"pid": process.pid}
            with contextlib.suppress(ExpansionError):
                report_process = process_identity(process.pid)
            self.state["report_hook"].update(
                last_status="started", last_started_at=now(), process=report_process
            )
        except OSError as exc:
            self.state["report_hook"].update(
                last_status="launch_failed", last_started_at=now(), error=str(exc)
            )
        self.save()

    def wait_for_oss(self, profile: Mapping[str, Any]) -> None:
        row = self.state["models"][profile["key"]]
        while True:
            self.check_signal()
            ready, handoff, reason = external_gate(self.study)
            output = Path(row["output_dir"])
            live_paths = {
                "handoff": self.study / OSS_HANDOFF,
                "run": output / "run.json",
                "status": output / "status.json",
                "grading_summary": output / "grading/summary.json",
            }
            fingerprint = {
                key: digest(path) for key, path in live_paths.items() if path.is_file()
            }
            changed = fingerprint != row.get("external_artifact_fingerprint")
            if live_paths["run"].is_file():
                external_run = read_json(live_paths["run"])
                row["external_run"] = external_run
                row["external_run_settings"] = external_run.get("settings")
                row["profile_binding"] = str(live_paths["run"])
            if live_paths["status"].is_file():
                row["run_status"] = read_json(live_paths["status"])
            else:
                row["run_status"] = handoff
            if live_paths["grading_summary"].is_file():
                row["grading_summary"] = read_json(live_paths["grading_summary"])
            row["external_artifacts"] = {
                key: str(path) for key, path in live_paths.items() if path.is_file()
            }
            row["external_artifact_fingerprint"] = fingerprint
            if ready:
                row.update(
                    state="complete",
                    preflight=handoff.get("diagnostic_outcomes"),
                    grading_summary=row.get("grading_summary") or handoff.get("grading_summary"),
                    error=None,
                )
                self.save(status="running", current_model=None, error=None, finished_at=None)
                self.event("external_oss_adopted", handoff=str(self.study / OSS_HANDOFF))
                self.trigger_report_update()
                return
            external_status = handoff.get("status") if isinstance(handoff, dict) else None
            external_gpu = handoff.get("gpu") if isinstance(handoff, dict) else None
            if (
                external_status in {"full64_failed", "grading_failed", "failed", "cancelled"}
                and isinstance(external_gpu, dict)
                and external_gpu.get("state") == "clear"
            ):
                row.update(
                    state="failed", error=f"external OSS terminal failure: {external_status}",
                    failed_at=now(),
                )
                self.save(
                    status="running_with_profile_failures", current_model=None,
                    error=None, finished_at=None,
                )
                self.event(
                    "external_oss_failed_after_clear",
                    status=external_status, error=handoff.get("error"),
                )
                self.trigger_report_update()
                raise ExpansionError(f"external OSS terminal failure after GPU clear: {external_status}")
            row.update(state="waiting_external_handoff", error=reason)
            self.save(
                status="waiting_for_oss_handoff", current_model=profile["key"],
                error=None, finished_at=None,
            )
            if changed:
                self.event(
                    "external_oss_artifacts_updated",
                    artifacts=row["external_artifacts"], fingerprint=fingerprint,
                )
                self.trigger_report_update()
            time.sleep(self.args.external_poll_seconds)

    def _window_number(self, output: Path) -> int:
        windows = output / "windows"
        if not windows.exists():
            return 1
        return len([path for path in windows.iterdir() if path.is_dir() and path.name.startswith("window-")]) + 1

    def _candidate_snapshot(self, candidate: Mapping[str, Any]) -> Path:
        return self.artifacts / "configs" / Path(str(candidate["path"])).name

    def _start_window(self, profile: Mapping[str, Any], config: Path, output: Path, number: int) -> tuple[subprocess.Popen[Any], dict[str, Any], Path, Path, Path]:
        backend = str(profile["runtime_backend"])
        tag = f"pmx-{profile['key'][:46]}-w{number:03d}"
        control = ROOT / "results" / f"{tag}-control.json"
        events = ROOT / "results" / f"{tag}-runtime-events.jsonl"
        window_dir = output / "windows" / f"window-{number:04d}"
        window_dir.mkdir(parents=True, exist_ok=False)
        command = [
            self.args.python, str(GPU_WINDOW), "--wait-seconds", "900", "--max-seconds", "86400", "--long-session", "--",
            str(MODEL_SESSION), "--runtime-backend", backend, "--config", str(config), "--plan", str(self.smoke_plan),
            "--tag", tag, "--startup-timeout", "480", "--hold-after-plan", "--control-file", str(control),
        ]
        environment = dict(os.environ)
        runtime_environment = profile.get("runtime_environment", {})
        if not isinstance(runtime_environment, dict) or any(not isinstance(k, str) or not isinstance(v, str) for k, v in runtime_environment.items()):
            raise ExpansionError(f"invalid runtime_environment for {profile['key']}")
        environment.update(runtime_environment)
        log_handle = (window_dir / "gpu-window.log").open("xb")
        try:
            process = subprocess.Popen(
                unblocked_exec_command(command), cwd=ROOT, env=environment,
                stdin=subprocess.DEVNULL, stdout=log_handle, stderr=subprocess.STDOUT,
                start_new_session=True, close_fds=True,
            )
        finally:
            log_handle.close()
        identity = wait_process_identity(process.pid, str(GPU_WINDOW))
        atomic_json(window_dir / "launch.json", {"at": now(), "argv": command, "process": identity, "config_sha256": digest(config), "runtime_environment": runtime_environment})
        return process, identity, control, events, window_dir

    def _wait_holding(self, process: subprocess.Popen[Any], profile: Mapping[str, Any], config: Path, control: Path, events: Path, number: int) -> ownership.HoldingSession:
        holder = SimpleNamespace(
            tag=f"pmx-{profile['key'][:46]}-w{number:03d}", config=config,
            plan=self.smoke_plan, runtime_backend=profile["runtime_backend"],
        )
        return ownership.wait_for_holding(
            process, events, args=holder, control=control,
            deadline=time.monotonic() + 1800, signals=ownership.SignalLatch(),
        )

    def _preflight_document(self, profile: Mapping[str, Any], config: Path, identity: ownership.HoldingSession, events: Path, attempts: list[Mapping[str, Any]]) -> dict[str, Any]:
        rows = _read_events(events)
        steps = [row for row in rows if row.get("event") == "step_complete"]
        step = steps[-1] if steps else None
        transport = transport_smoke_evidence(events)
        return {
            "schema_version": 1, "accepted_at": now(), "profile_key": profile["key"],
            "runtime_backend": profile["runtime_backend"], "model_name": profile["model_name"],
            "context_tokens": 73728, "selected_config": str(config), "selected_config_sha256": digest(config),
            "session": asdict(identity), "health_alias_gate_passed": True,
            "transport_smoke_gate_passed": True, "transport_smoke": transport,
            "semantic_answer_used_as_gate": False,
            "smoke_semantic_passed": transport["semantic_passed"],
            "smoke_returncode": step.get("returncode") if isinstance(step, dict) else None,
            "placement_attempts": list(attempts),
        }

    def _runner_command(self, profile: Mapping[str, Any], output: Path) -> list[str]:
        return [
            self.args.python, str(self.runner), "run", "--manifest", str(MANIFEST),
            "--endpoint", endpoint_for_backend(str(profile["runtime_backend"])),
            "--backend-adapter", str(profile["runtime_backend"]), "--model", str(profile["model_name"]),
            "--model-key", str(profile["key"]), "--output-dir", str(output),
            *profile_request_args(profile), "--rerun-failed",
        ]

    def _run_scored(self, profile: Mapping[str, Any], output: Path, window_dir: Path, outer: subprocess.Popen[Any]) -> tuple[bool, str]:
        command = self._runner_command(profile, output)
        for retry in range(1, self.args.runner_retries_per_window + 1):
            log_path = window_dir / f"runner-{retry:02d}.log"
            with log_path.open("xb") as log_handle:
                child = subprocess.Popen(
                    unblocked_exec_command(command), cwd=ROOT, stdin=subprocess.DEVNULL,
                    stdout=log_handle, stderr=subprocess.STDOUT, start_new_session=True, close_fds=True,
                )
                child_identity = wait_process_identity(child.pid, str(self.runner))
                atomic_json(window_dir / f"runner-{retry:02d}-command.json", {"argv": command, "process": child_identity})
                try:
                    while child.poll() is None:
                        self.check_signal()
                        if outer.poll() is not None:
                            terminate_owned_process(child, child_identity)
                            return False, "gpu_window_expired_or_exited"
                        self.state["models"][profile["key"]]["run_status"] = case_coverage(output)
                        self.save()
                        time.sleep(2)
                except BaseException:
                    if child.poll() is None:
                        terminate_owned_process(child, child_identity)
                    raise
            if child.returncode == 0 and full_coverage(output):
                return True, "complete"
            if outer.poll() is not None:
                return False, "gpu_window_expired_or_exited"
        return False, "runner_retry_limit"

    def _stop_window(self, process: subprocess.Popen[Any], identity: Mapping[str, Any], control: Path, backend: str) -> dict[str, Any]:
        stop_requested = ownership.request_session_stop(control) if control.exists() else False
        code, _ = ownership.wait_for_outer_exit(process, timeout=180, control=control)
        if code is None:
            if not validate_process_identity(identity):
                raise SafetyStop("refusing outer termination after identity validation failed")
            ownership.request_outer_termination(process)
            code, _ = ownership.wait_for_outer_exit(process, timeout=180, control=control)
        restoration = ownership.restoration_evidence(backend)
        if code is None or restoration.get("original_paused") or not restoration.get("original_running") or not restoration.get("owned_loopback_port_clear"):
            raise SafetyStop(f"owned window cleanup/restoration not clean: returncode={code}, restoration={restoration}")
        return {"returncode": code, "stop_requested": stop_requested, "restoration": restoration}

    def _grade(self, profile: Mapping[str, Any], output: Path) -> dict[str, Any]:
        if not full_coverage(output):
            raise ExpansionError("official grading requires 64 terminal scored cases")
        existing = output / "grading/summary.json"
        if existing.exists():
            summary = read_json(existing)
            validate_grade_coverage(summary)
            return summary
        command = grade_command(self.args.grade_image, self.study, output, str(profile["key"]), self.manifest, self.grade_runner)
        atomic_json(output / "grading-command.json", {"argv": command, "isolation": "network none; read-only root and public inputs; only profile output writable"})
        with (output / "grading.log").open("xb") as log:
            completed = subprocess.run(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, timeout=21600, check=False)
        if completed.returncode != 0 or not existing.exists():
            raise ExpansionError(f"official grading exited {completed.returncode}")
        summary = read_json(existing)
        validate_grade_coverage(summary)
        return summary

    def run_profile(self, profile: Mapping[str, Any]) -> None:
        key = str(profile["key"])
        row = self.state["models"][key]
        output = Path(row["output_dir"])
        output.mkdir(parents=True, exist_ok=True)
        if row.get("state") == "complete" and row.get("grading_summary"):
            return
        summary_path = output / "grading/summary.json"
        if summary_path.exists():
            summary = read_json(summary_path)
            validate_grade_coverage(summary)
            row.update(state="complete", grading_summary=summary, error=None)
            self.save()
            return
        candidates = profile["config_candidates"]
        binding_path = output / "profile-binding.json"
        selected: Path | None = None
        if binding_path.exists():
            binding = read_json(binding_path)
            selected = Path(binding["config"]["path"])
            if digest(selected) != binding["config"]["sha256"]:
                raise ExpansionError(f"selected profile config changed: {selected}")
        placement_attempts: list[dict[str, Any]] = []
        consecutive_windows = 0
        while not full_coverage(output):
            self.check_signal()
            self.verify_pins()
            if consecutive_windows >= self.args.max_windows_per_profile:
                raise ExpansionError("profile exhausted its maximum resumable GPU windows")
            number = self._window_number(output)
            candidate_list = [selected] if selected is not None else [self._candidate_snapshot(candidate) for candidate in candidates]
            held: tuple[subprocess.Popen[Any], dict[str, Any], Path, Path, Path, ownership.HoldingSession, Path] | None = None
            for config in candidate_list:
                process: subprocess.Popen[Any] | None = None
                identity: dict[str, Any] | None = None
                control: Path | None = None
                try:
                    row.update(state="preflighting", error=None)
                    self.save(
                        status="running", current_model=key,
                        error=None, finished_at=None,
                    )
                    process, identity, control, events, window_dir = self._start_window(profile, config, output, number)
                    session = self._wait_holding(process, profile, config, control, events, number)
                    placement_attempts.append({"config": str(config), "config_sha256": digest(config), "status": "accepted", "window": number})
                    held = (process, identity, control, events, window_dir, session, config)
                    break
                except BaseException as exc:
                    placement_attempts.append({"config": str(config), "config_sha256": digest(config), "status": "failed", "window": number, "error": str(exc)})
                    if process is not None and identity is not None and control is not None:
                        self._stop_window(process, identity, control, str(profile["runtime_backend"]))
                    if selected is not None or isinstance(exc, (ExpansionInterrupted, SafetyStop)):
                        raise
                    number += 1
            if held is None:
                atomic_json(output / "preflight.json", {"schema_version": 1, "status": "failed", "placement_attempts": placement_attempts})
                raise ExpansionError("all exact-72K placement candidates failed preflight")
            process, process_id, control, events, window_dir, session, config = held
            success = False
            reason = "preflight_or_binding_failed"
            try:
                if selected is None:
                    preflight = self._preflight_document(profile, config, session, events, placement_attempts)
                    selected = config
                    binding = ensure_profile_binding(binding_path, profile_binding(profile, selected, self.runner))
                    atomic_json(output / "preflight.json", preflight)
                    row.update(profile_binding=binding, preflight=preflight)
                else:
                    ensure_profile_binding(binding_path, profile_binding(profile, selected, self.runner))
                row.update(state="generating", error=None)
                self.save()
                self.event("model_window_scoring_started", model=key, window=number, session=asdict(session))
                success, reason = self._run_scored(profile, output, window_dir, process)
            finally:
                cleanup = self._stop_window(process, process_id, control, str(profile["runtime_backend"]))
                atomic_json(window_dir / "completion.json", {"at": now(), "scoring_complete": success, "reason": reason, **cleanup})
            consecutive_windows += 1
            row["run_status"] = case_coverage(output)
            self.save()
            if success:
                break
            if reason == "runner_retry_limit":
                self.event("model_window_retry", model=key, window=number, reason=reason)

        row.update(state="grading", run_status=case_coverage(output))
        self.save()
        self.event("model_grading_started", model=key)
        summary = self._grade(profile, output)
        row.update(state="complete", grading_summary=summary, error=None)
        self.save(current_model=None)
        self.event("model_complete", model=key)
        self.trigger_report_update()

    def run(self) -> int:
        for signum in (signal.SIGINT, signal.SIGTERM):
            signal.signal(signum, self.signal_handler)
        self.prepare_artifacts()
        self.record_controls()
        self.state["grade_image"] = self.args.grade_image
        self.state["supervisor_process"] = process_identity(os.getpid())
        self.save(status="running", error=None, finished_at=None)
        self.event("supervisor_started", manifest_sha256=MANIFEST_SHA256, catalog_sha256=self.catalog_sha256)
        try:
            failed_profiles: list[str] = []
            for profile in self.catalog["profiles"]:
                self.check_signal()
                key = str(profile["key"])
                try:
                    if profile.get("managed_externally"):
                        self.wait_for_oss(profile)
                    else:
                        self.run_profile(profile)
                except SafetyStop:
                    raise
                except ExpansionInterrupted:
                    raise
                except Exception as exc:
                    failed_profiles.append(key)
                    self.state["models"][key].update(
                        state="failed", error=str(exc), failed_at=now(),
                        run_status=case_coverage(Path(self.state["models"][key]["output_dir"])),
                    )
                    self.save(
                        status="running_with_profile_failures", current_model=None,
                        failed_profiles=list(failed_profiles), failure_count=len(failed_profiles),
                        error=None, finished_at=None,
                    )
                    self.event(
                        "model_failed_continuing_queue", model=key,
                        error_type=type(exc).__name__, error=str(exc),
                    )
                    self.trigger_report_update()
            final_status = "complete_with_profile_failures" if failed_profiles else "complete"
            self.save(
                status=final_status, current_model=None, finished_at=now(),
                failed_profiles=failed_profiles, failure_count=len(failed_profiles),
                error=None,
            )
            self.event("supervisor_complete", status=final_status, failed_profiles=failed_profiles)
            self.trigger_report_update()
            return 1 if failed_profiles else 0
        except ExpansionInterrupted as exc:
            self.save(status="interrupted", error=str(exc), finished_at=now())
            self.event("supervisor_interrupted", error=str(exc))
            return 130
        except BaseException as exc:
            current = self.state.get("current_model")
            if isinstance(current, str) and current in self.state["models"]:
                self.state["models"][current].update(state="failed", error=str(exc))
            self.save(status="failed", error=str(exc), finished_at=now())
            self.event("supervisor_failed", error_type=type(exc).__name__, error=str(exc))
            return 1


def _supervisor_lock(path: Path) -> Iterable[Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+b")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        handle.close()
        raise ExpansionError("another expansion supervisor holds the study lock") from exc
    try:
        yield handle
    finally:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


_supervisor_lock = contextlib.contextmanager(_supervisor_lock)


def launch(args: argparse.Namespace) -> int:
    study = args.study_dir.resolve()
    study.mkdir(parents=True, exist_ok=True)
    verify_manifest(args.manifest)
    load_catalog(args.catalog)
    launch_path = study / "launch.json"
    if launch_path.exists():
        existing = read_json(launch_path)
        identity = existing.get("process")
        if isinstance(identity, dict) and validate_process_identity(identity):
            print(json.dumps({"status": "already_running", "process": identity}, indent=2))
            return 0
    command = [
        args.python, str(Path(__file__).resolve()), "supervise", "--study-dir", str(study),
        "--manifest", str(args.manifest.resolve()), "--catalog", str(args.catalog.resolve()),
        "--python", args.python, "--grade-image", args.grade_image,
        "--external-poll-seconds", str(args.external_poll_seconds),
        "--runner-retries-per-window", str(args.runner_retries_per_window),
        "--max-windows-per-profile", str(args.max_windows_per_profile),
    ]
    log = (study / "supervisor.log").open("ab", buffering=0)
    try:
        process = subprocess.Popen(
            unblocked_exec_command(command), cwd=ROOT, stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True, close_fds=True,
        )
    finally:
        log.close()
    identity = wait_process_identity(process.pid, str(Path(__file__).resolve()))
    value = {"schema_version": 1, "launched_at": now(), "argv": command, "process": identity, "manifest_sha256": MANIFEST_SHA256, "catalog_sha256": digest(args.catalog)}
    atomic_json(launch_path, value)
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("launch", "supervise", "dry-run"):
        child = subparsers.add_parser(name)
        child.add_argument("--study-dir", type=Path, default=DEFAULT_STUDY)
        child.add_argument("--manifest", type=Path, default=MANIFEST)
        child.add_argument("--catalog", type=Path, default=CATALOG)
        child.add_argument("--python", default=str(ROOT / ".venv-public-bench/bin/python"))
        child.add_argument("--grade-image", default="moe-public-benchmark-grader:20261023-v1")
        child.add_argument("--external-poll-seconds", type=float, default=30.0)
        child.add_argument("--runner-retries-per-window", type=int, default=3)
        child.add_argument("--max-windows-per-profile", type=int, default=32)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.study_dir = args.study_dir.resolve()
        args.manifest = args.manifest.resolve()
        args.catalog = args.catalog.resolve()
        manifest = verify_manifest(args.manifest)
        catalog = load_catalog(args.catalog)
        if args.command == "dry-run":
            print(json.dumps({"manifest_sha256": MANIFEST_SHA256, "catalog_sha256": digest(args.catalog), "profiles": catalog["profiles"]}, indent=2))
            return 0
        if args.command == "launch":
            return launch(args)
        with _supervisor_lock(args.study_dir / "supervisor.lock"):
            return Supervisor(args, catalog, manifest).run()
    except ExpansionError as exc:
        print(f"public model expansion: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
