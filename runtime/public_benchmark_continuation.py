#!/usr/bin/env python3
"""Continue the frozen public pilot after its attested unscored warmup failure.

The supervisor never replaces the original controller status.  It waits for
the original controller lock, snapshots the completed evidence, admits only a
precisely described warmup failure, and then owns any required resume windows,
grading, and the LFM run serially.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import stat
import subprocess
import sys
import time
from types import SimpleNamespace
from typing import Any, Callable, Iterable, Mapping, Sequence
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
VENDOR = ROOT / "vendor/public-benchmarks"
GPU_WINDOW = ROOT / "runtime/gpu_window.py"
MODEL_SESSION = ROOT / "runtime/model_session.py"
PLAN_V1 = ROOT / "runtime/public-benchmark-pilot-ready-v1-plan.json"
PLAN_V2 = ROOT / "runtime/public-benchmark-pilot-ready-v2-plan.json"
DEFAULT_STUDY = ROOT / "results/public-benchmark-pilot-20261023-v1"
MODELS = {
    "oss20b": ROOT / "runtime/gpt20b-llama-specialists-72k-wide64k.json",
    "lfm25": ROOT / "runtime/public-benchmark-pilot-lfm72k-auto-v1.json",
}
TERMINAL_STATUSES = {"completed", "truncated"}
RESUMABLE_STATUSES = {"not_started", "api_failure", "cancelled", "incomplete"}
ALL_STATUSES = TERMINAL_STATUSES | RESUMABLE_STATUSES


class ContinuationError(RuntimeError):
    pass


class ContinuationInterrupted(BaseException):
    pass


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            value.update(block)
    return value.hexdigest()


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ContinuationError(f"unreadable or invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ContinuationError(f"JSON must be an object: {path}")
    return value


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    payload = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()


def append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def verify_manifest(study: Path) -> dict[str, Any]:
    checksum_path = study / "manifest.sha256"
    try:
        expected = checksum_path.read_text(encoding="utf-8").split()[0]
    except (OSError, IndexError) as exc:
        raise ContinuationError("frozen manifest checksum is unavailable") from exc
    manifest_path = study / "manifest.json"
    if digest(manifest_path) != expected:
        raise ContinuationError("frozen benchmark manifest SHA256 mismatch")
    manifest = read_json(manifest_path)
    cases = manifest.get("cases")
    if manifest.get("selected_count_per_suite") != 32 or not isinstance(cases, list) or len(cases) != 64:
        raise ContinuationError("continuation requires the frozen 32+32 manifest")
    return manifest


def acquire_controller_lock(
    path: Path,
    *,
    poll_seconds: float,
    on_wait: Callable[[], None],
    check_interrupt: Callable[[], None] = lambda: None,
) -> Any:
    """Wait for the old controller without inspecting or signalling its process."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+b")
    while True:
        check_interrupt()
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return handle
        except BlockingIOError:
            on_wait()
            time.sleep(poll_seconds)


def copy_exclusive(source: Path, destination: Path) -> dict[str, Any]:
    before = source.lstat()
    if not stat.S_ISREG(before.st_mode) or source.is_symlink():
        raise ContinuationError(f"snapshot source is not a regular file: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(destination, flags, 0o600)
    try:
        with source.open("rb") as incoming, os.fdopen(descriptor, "wb", closefd=False) as outgoing:
            shutil.copyfileobj(incoming, outgoing, 1024 * 1024)
            outgoing.flush()
            os.fsync(outgoing.fileno())
    finally:
        os.close(descriptor)
    after = source.lstat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise ContinuationError(f"snapshot source changed while copying: {source}")
    return {
        "source": str(source),
        "snapshot": str(destination),
        "size": before.st_size,
        "sha256": digest(destination),
    }


def snapshot_original_evidence(
    study: Path, controller: Mapping[str, Any]
) -> dict[str, Any]:
    model = controller.get("models", {}).get("oss20b", {})
    tag = model.get("session_tag")
    sources: dict[str, Path] = {
        "controller_status": study / "controller-status.json",
        "controller_log": study / "controller.log",
        "controller_events": study / "controller-events.jsonl",
        "launch": study / "launch.json",
        "oss_status": study / "oss20b/status.json",
        "oss_run": study / "oss20b/run.json",
        "oss_runner_log": study / "oss20b/runner.log",
        "oss_gpu_window_log": study / "oss20b/gpu-window.log",
        "controller_source": study / "artifacts/run_public_benchmark_pilot.final.py",
        "runner_source": study / "artifacts/public_benchmark_pilot.py",
        "source_pins": study / "artifacts/source-pins.json",
    }
    if isinstance(tag, str) and tag:
        sources.update(
            {
                "session_report": ROOT / "results" / f"{tag}-session-report.json",
                "runtime_events": Path(str(model.get("runtime_events", ""))),
                "warmup_result": ROOT / "results" / f"{tag}-01-ready.jsonl",
                "control": Path(str(model.get("control_file", ""))),
            }
        )
    missing = [name for name, path in sources.items() if not path.is_file()]
    if missing:
        raise ContinuationError("original evidence is incomplete: " + ", ".join(missing))
    target = study / "continuation-evidence/original"
    if target.exists():
        raise ContinuationError(f"original evidence snapshot already exists: {target}")
    target.mkdir(parents=True)
    rows = {
        name: copy_exclusive(path, target / f"{name}{path.suffix or '.bin'}")
        for name, path in sources.items()
    }
    receipt = {"schema_version": 1, "captured_at": now(), "files": rows}
    atomic_json(target / "receipt.json", receipt)
    return receipt


def _one_jsonl(path: Path) -> dict[str, Any]:
    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ContinuationError(f"invalid JSONL evidence: {path}") from exc
    if len(rows) != 1 or not isinstance(rows[0], dict):
        raise ContinuationError(f"expected exactly one evidence row: {path}")
    return rows[0]


def attest_warmup_only_failure(
    report: Mapping[str, Any], warmup: Mapping[str, Any], *, expected_plan_sha256: str
) -> dict[str, Any]:
    if report.get("status") != "completed_with_step_failures" or report.get("problem") is not None:
        raise ContinuationError("original session was not a clean completed_with_step_failures warmup")
    plan = report.get("plan")
    if not isinstance(plan, dict) or plan.get("sha256") != expected_plan_sha256 or plan.get("step_count") != 1:
        raise ContinuationError("original session plan does not match the frozen readiness plan")
    steps = report.get("steps")
    if not isinstance(steps, list) or len(steps) != 1 or not isinstance(steps[0], dict):
        raise ContinuationError("original session must have exactly one warmup step")
    step = steps[0]
    controls = step.get("controls")
    expected_controls = {
        "suite": "quick",
        "cases": ["quick_generation"],
        "thinking": "medium",
        "temperature": 0.0,
        "seed": 20261023,
        "output_tokens": 32,
        "request_timeout": 120.0,
    }
    if not isinstance(controls, dict) or any(controls.get(key) != value for key, value in expected_controls.items()):
        raise ContinuationError("original nonzero step is not the expected capped warmup")
    if step.get("id") != "ready" or step.get("returncode") != 1 or step.get("passed") is not False or step.get("error") is not None:
        raise ContinuationError("original readiness step failure has an unexpected shape")
    server = report.get("server")
    cleanup = server.get("cleanup") if isinstance(server, dict) else None
    if (
        not isinstance(server, dict)
        or server.get("runtime_backend") != "llama"
        or server.get("context") != 73728
        or not isinstance(cleanup, dict)
        or cleanup.get("stopped") is not True
        or cleanup.get("kill_sent") is not False
    ):
        raise ContinuationError("original owned server cleanup is not clean")
    evaluation = warmup.get("evaluation")
    response = warmup.get("response")
    request = warmup.get("request")
    if (
        warmup.get("case_id") != "quick_generation"
        or warmup.get("error") is not None
        or not isinstance(evaluation, dict)
        or evaluation.get("passed") is not False
        or evaluation.get("truncated") is not True
        or evaluation.get("detail") != "response ended because of the length limit"
        or not isinstance(response, dict)
        or response.get("done_reason") != "length"
        or response.get("truncated") is not True
        or response.get("stream_complete") is not True
        or not isinstance(request, dict)
        or request.get("output_token_limit") != 32
    ):
        raise ContinuationError("warmup result is not the expected semantic length-cap failure")
    return {
        "accepted": True,
        "scope": "unscored_server_readiness_only",
        "code": "quick_generation_length_cap_semantic_failure",
        "step_id": "ready",
        "step_returncode": 1,
        "output_token_limit": 32,
        "finish_reason": "length",
        "stream_complete": True,
        "inference_score_effect": "none",
    }


def require_clean_restoration(value: Mapping[str, Any], label: str) -> None:
    if (
        value.get("original_running") is not True
        or value.get("original_paused") is not False
        or value.get("owned_loopback_port_clear") is not True
    ):
        raise ContinuationError(f"{label} does not prove clean restoration")


def attest_owned_24h_timeout(
    report: Mapping[str, Any],
    warmup: Mapping[str, Any],
    gpu_window_log: Path,
    audit: Mapping[str, Any],
    *,
    expected_plan_sha256: str,
) -> dict[str, Any]:
    """Admit only gpu_window's exact owned 86400-second boundary."""
    # Reuse every readiness/cleanup check while separately retaining the real
    # interrupted status and problem below.
    semantic_report = dict(report)
    semantic_report["status"] = "completed_with_step_failures"
    semantic_report["problem"] = None
    warmup_evidence = attest_warmup_only_failure(
        semantic_report, warmup, expected_plan_sha256=expected_plan_sha256
    )
    problem = report.get("problem")
    if (
        report.get("status") != "interrupted"
        or not isinstance(problem, dict)
        or problem.get("error_type") != "KeyboardInterrupt"
        or problem.get("error") != "signal 15"
    ):
        raise ContinuationError("original interruption is not the owned SIGTERM timeout path")
    rows = _event_rows(gpu_window_log)
    timeouts = [row for row in rows if row.get("event") == "trial_timeout"]
    starts = [row for row in rows if row.get("event") == "trial_started"]
    finishes = [row for row in rows if row.get("event") == "trial_finished"]
    completes = [row for row in rows if row.get("event") == "window_complete"]
    forbidden = {
        "trial_cleanup_kill",
        "trial_cleanup_failed",
        "restore_blocked_orphan_guard",
        "restore_failed",
    }
    if (
        len(timeouts) != 1
        or len(starts) != 1
        or starts[0].get("max_seconds") != 86400.0
        or timeouts[0].get("max_seconds") != 86400.0
        or timeouts[0].get("child_pgid") != starts[0].get("child_pid")
        or not finishes
        or finishes[-1].get("returncode") != 124
        or not completes
        or completes[-1].get("returncode") != 124
        or any(row.get("event") in forbidden for row in rows)
    ):
        raise ContinuationError("GPU-window log does not prove the exact clean 24h timeout")
    candidates = audit.get("resume_candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ContinuationError("24h timeout has no saved nonterminal tasks to resume")
    return {
        "accepted": True,
        "scope": "owned_gpu_window_boundary",
        "code": "gpu_window_exact_86400_second_timeout",
        "gpu_window_returncode": 124,
        "signal": 15,
        "remaining_nonterminal_count": len(candidates),
        "warmup": warmup_evidence,
    }


def _task_storage_id(task_id: str) -> str:
    return hashlib.sha256(task_id.encode("utf-8")).hexdigest()[:16]


def audit_model_output(
    manifest: Mapping[str, Any],
    manifest_sha256: str,
    output: Path,
    expected_settings: Mapping[str, Any],
    *,
    allow_absent: bool = False,
) -> dict[str, Any]:
    run_path = output / "run.json"
    if not run_path.exists():
        if allow_absent:
            return {
                "present": False,
                "total": 64,
                "terminal_count": 0,
                "resume_candidates": [
                    {"index": index, "suite": case["suite"], "task_id": case["task_id"], "status": "not_started", "attempts": []}
                    for index, case in enumerate(manifest["cases"])
                ],
                "terminal_preserved": [],
                "cases": [],
            }
        raise ContinuationError(f"model run binding is missing: {run_path}")
    binding = read_json(run_path)
    if binding.get("manifest", {}).get("sha256") != manifest_sha256:
        raise ContinuationError(f"model output manifest binding mismatch: {output}")
    settings = binding.get("settings")
    if not isinstance(settings, dict) or any(settings.get(key) != value for key, value in expected_settings.items()):
        raise ContinuationError(f"model output settings mismatch: {output}")
    if binding.get("settings_sha256") != canonical_hash(settings):
        raise ContinuationError(f"model output settings hash mismatch: {output}")
    records: list[dict[str, Any]] = []
    counts = {status: 0 for status in sorted(ALL_STATUSES)}
    terminal: list[dict[str, Any]] = []
    resume: list[dict[str, Any]] = []
    for index, case in enumerate(manifest["cases"]):
        suite, task_id = case["suite"], case["task_id"]
        case_dir = output / "cases" / suite / _task_storage_id(task_id)
        attempt_dirs = sorted(case_dir.glob("attempt-*")) if case_dir.is_dir() else []
        attempts: list[dict[str, Any]] = []
        selected: dict[str, Any] | None = None
        latest_status = "not_started"
        for attempt_dir in attempt_dirs:
            result_path = attempt_dir / "result.json"
            if not result_path.exists():
                item = {"attempt": attempt_dir.name, "status": "incomplete", "result": None}
                latest_status = "incomplete"
            else:
                result = read_json(result_path)
                status_value = result.get("status")
                if status_value not in ALL_STATUSES - {"not_started"}:
                    raise ContinuationError(f"invalid attempt status in {result_path}")
                if result.get("manifest_sha256") != manifest_sha256 or result.get("settings_sha256") != binding.get("settings_sha256"):
                    raise ContinuationError(f"attempt binding mismatch: {result_path}")
                item = {
                    "attempt": attempt_dir.name,
                    "status": status_value,
                    "result": str(result_path),
                    "result_sha256": digest(result_path),
                    "truncated": result.get("truncated"),
                    "finish_reason": result.get("finish_reason"),
                }
                latest_status = str(status_value)
                if status_value in TERMINAL_STATUSES:
                    if selected is not None:
                        raise ContinuationError(f"multiple terminal samples exist for {suite} {task_id}")
                    selected = item
            attempts.append(item)
        status_value = str(selected["status"]) if selected else latest_status
        if status_value not in ALL_STATUSES:
            raise ContinuationError(f"unrecognized case status for {suite} {task_id}")
        record = {
            "index": index,
            "suite": suite,
            "task_id": task_id,
            "status": status_value,
            "latest_status": latest_status,
            "attempt_count": len(attempts),
            "attempts": attempts,
            "scoring_attempt": selected["attempt"] if selected else None,
        }
        counts[status_value] += 1
        records.append(record)
        (terminal if selected else resume).append(record)
    return {
        "present": True,
        "run_sha256": digest(run_path),
        "settings_sha256": binding["settings_sha256"],
        "settings": settings,
        "total": len(records),
        "terminal_count": len(terminal),
        "counts": counts,
        "terminal_preserved": terminal,
        "resume_candidates": resume,
        "cases": records,
    }


def verify_resume_transition(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
    before_cases = {(row["suite"], row["task_id"]): row for row in before["cases"]}
    after_cases = {(row["suite"], row["task_id"]): row for row in after["cases"]}
    if before_cases.keys() != after_cases.keys():
        raise ContinuationError("case set changed across resume")
    resumed: list[dict[str, Any]] = []
    preserved: list[dict[str, Any]] = []
    for identity, old in before_cases.items():
        new = after_cases[identity]
        if old["status"] in TERMINAL_STATUSES:
            if (
                new["status"] != old["status"]
                or new["attempt_count"] != old["attempt_count"]
                or new["scoring_attempt"] != old["scoring_attempt"]
            ):
                raise ContinuationError(f"terminal first-passage sample was changed: {identity}")
            preserved.append(
                {"suite": identity[0], "task_id": identity[1], "status": old["status"], "scoring_attempt": old["scoring_attempt"]}
            )
        else:
            if new["attempt_count"] <= old["attempt_count"]:
                raise ContinuationError(f"resume did not create an attempt for {identity}")
            resumed.append(
                {
                    "suite": identity[0],
                    "task_id": identity[1],
                    "before_status": old["status"],
                    "before_attempt_count": old["attempt_count"],
                    "after_status": new["status"],
                    "after_attempt_count": new["attempt_count"],
                    "scoring_attempt": new["scoring_attempt"],
                    "not_best_of": old["scoring_attempt"] is None,
                }
            )
    return {"terminal_untouched": preserved, "resumed_nonterminal": resumed}


def expected_settings(alias: str, key: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "endpoint": "http://127.0.0.1:18081/v1",
        "model": alias,
        "model_key": key,
        "max_tokens": 65536,
        "request_timeout_seconds": 21600.0,
        "reasoning_effort": "medium",
        "temperature": 0.0,
        "seed": 20261023,
        "api_key_env": None,
        "prompt_policy_version": "public-pilot-20261023-v1",
        "humaneval_system_instruction": "Return the complete Python function, including its signature. Output only Python code. Do not include tests.",
        "ifeval_prompt_policy": "verbatim_user_message_only",
        "stream_options_include_usage": True,
    }


def runner_command(
    python: str,
    runner: Path,
    study: Path,
    output: Path,
    settings: Mapping[str, Any],
    *,
    rerun_failed: bool,
) -> list[str]:
    command = [
        python,
        str(runner),
        "run",
        "--manifest",
        str(study / "manifest.json"),
        "--endpoint",
        str(settings["endpoint"]),
        "--model",
        str(settings["model"]),
        "--model-key",
        str(settings["model_key"]),
        "--output-dir",
        str(output),
        "--max-tokens",
        str(settings["max_tokens"]),
        "--request-timeout",
        format(float(settings["request_timeout_seconds"]), "g"),
        "--reasoning-effort",
        str(settings["reasoning_effort"]),
        "--temperature",
        format(float(settings["temperature"]), "g"),
        "--seed",
        str(settings["seed"]),
    ]
    if rerun_failed:
        command.append("--rerun-failed")
    return command


def grade_command(
    image: str,
    runner: Path,
    study: Path,
    output: Path,
    model_key: str,
    manifest: Mapping[str, Any],
    continuation_id: str,
) -> list[str]:
    return [
        "docker",
        "run",
        "--rm",
        "--name",
        f"moe-public-grade-cont-{model_key}-{continuation_id}",
        "--network",
        "none",
        "--read-only",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--pids-limit",
        "256",
        "--cpus",
        "4",
        "--memory",
        "4g",
        "--shm-size",
        "256m",
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "--tmpfs",
        "/tmp:rw,nosuid,size=2g",
        "--env",
        "HOME=/tmp",
        "--env",
        "XDG_CACHE_HOME=/tmp/cache",
        "--env",
        "PYTHONDONTWRITEBYTECODE=1",
        "--env",
        "OMP_NUM_THREADS=1",
        "--env",
        "OPENBLAS_NUM_THREADS=1",
        "--mount",
        f"type=bind,src={runner},dst=/app/public_benchmark_pilot.py,readonly",
        "--mount",
        f"type=bind,src={VENDOR},dst={VENDOR},readonly",
        "--mount",
        f"type=bind,src={study},dst={study},readonly",
        "--mount",
        f"type=bind,src={output},dst={output}",
        image,
        "python",
        "/app/public_benchmark_pilot.py",
        "grade",
        "--manifest",
        str(study / "manifest.json"),
        "--output-dir",
        str(output),
        "--evalplus-root",
        str(manifest["sources"]["evalplus"]["root"]),
        "--ifeval-root",
        str(Path(manifest["sources"]["ifeval"]["root"]) / "instruction_following_eval"),
    ]


def _event_rows(path: Path) -> list[dict[str, Any]]:
    try:
        return [
            value
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and isinstance((value := json.loads(line)), dict)
        ]
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ContinuationError(f"runtime events are invalid: {path}") from exc


def attest_v2_readiness(events_path: Path) -> dict[str, Any]:
    rows = [row for row in _event_rows(events_path) if row.get("event") == "step_complete"]
    if len(rows) != 1:
        raise ContinuationError("continuation window did not emit exactly one readiness result")
    row = rows[0]
    controls = row.get("controls")
    if (
        row.get("id") != "ready-json"
        or row.get("returncode") != 0
        or row.get("passed") is not True
        or row.get("error") is not None
        or not isinstance(controls, dict)
        or controls.get("cases") != ["quick_json_extract"]
        or controls.get("output_tokens") != 8192
        or controls.get("request_timeout") != 900.0
        or controls.get("thinking") != "medium"
        or controls.get("temperature") != 0.0
        or controls.get("seed") != 20261023
    ):
        raise ContinuationError("continuation readiness did not pass the exact v2 warmup")
    return {
        "status": "passed",
        "step_id": "ready-json",
        "case": "quick_json_extract",
        "output_tokens": 8192,
        "returncode": 0,
    }


class Supervisor:
    def __init__(self, args: argparse.Namespace, manifest: dict[str, Any]) -> None:
        self.args = args
        self.study = args.study_dir.resolve()
        self.manifest = manifest
        self.manifest_sha256 = digest(self.study / "manifest.json")
        self.continuation_id = f"{int(time.time())}-{uuid.uuid4().hex[:8]}"
        self.received_signal: int | None = None
        self.lock_handle: Any = None
        self.continuation_source_pins = {
            str(Path(__file__).resolve()): digest(Path(__file__).resolve()),
            str(PLAN_V2): digest(PLAN_V2),
        }
        self.state: dict[str, Any] = {
            "schema_version": 1,
            "kind": "public_benchmark_continuation",
            "continuation_id": self.continuation_id,
            "supervisor_pid": os.getpid(),
            "started_at": now(),
            "status": "starting",
            "manifest_sha256": self.manifest_sha256,
            "models": {},
            "windows": [],
            "continuation_source_pins": self.continuation_source_pins,
        }

    @property
    def status_path(self) -> Path:
        return self.study / "continuation-status.json"

    def save(self, **fields: Any) -> None:
        self.state.update(fields)
        self.state["updated_at"] = now()
        atomic_json(self.status_path, self.state)

    def event(self, event: str, **fields: Any) -> None:
        row = {"at": now(), "event": event, "continuation_id": self.continuation_id, **fields}
        append_jsonl(self.study / "continuation-events.jsonl", row)
        print(json.dumps(row, separators=(",", ":")), flush=True)

    def handle_signal(self, signum: int, _frame: Any) -> None:
        self.received_signal = signum

    def check_interrupt(self) -> None:
        if self.received_signal is not None:
            raise ContinuationInterrupted(f"signal {self.received_signal}")

    def waiting_update(self) -> None:
        self.save(
            status="waiting_for_original_controller",
            waiting_for=str(self.study / "controller.lock"),
        )

    def verify_source_pins(self, controller: Mapping[str, Any], launch: Mapping[str, Any]) -> None:
        pins = controller.get("runtime_file_sha256")
        if not isinstance(pins, dict):
            raise ContinuationError("original controller has no source pins")
        for filename, expected in pins.items():
            path = Path(filename)
            if not path.is_file() or digest(path) != expected:
                raise ContinuationError(f"original source pin mismatch: {path}")
        controller_hash = launch.get("controller_sha256")
        artifact_controller = self.study / "artifacts/run_public_benchmark_pilot.final.py"
        if (
            not isinstance(controller_hash, str)
            or digest(ROOT / "runtime/run_public_benchmark_pilot.py") != controller_hash
            or digest(artifact_controller) != controller_hash
        ):
            raise ContinuationError("original controller hash does not match launch and snapshot")
        if digest(self.args.runner) != controller.get("runner_sha256"):
            raise ContinuationError("immutable benchmark runner hash mismatch")
        if launch.get("grader_image") != self.args.grade_image:
            raise ContinuationError("immutable grader image ID mismatch")
        self.state["validated_source_pins"] = dict(pins)
        self.state["original_controller_sha256"] = controller_hash
        self.state["runner_sha256"] = digest(self.args.runner)

    def attest_original(self) -> dict[str, Any]:
        controller = read_json(self.study / "controller-status.json")
        launch = read_json(self.study / "launch.json")
        self.verify_source_pins(controller, launch)
        snapshot = snapshot_original_evidence(self.study, controller)
        model = controller.get("models", {}).get("oss20b")
        if not isinstance(model, dict):
            raise ContinuationError("original controller has no OSS model record")
        tag = model.get("session_tag")
        if not isinstance(tag, str):
            raise ContinuationError("original OSS session tag is missing")
        report_path = ROOT / "results" / f"{tag}-session-report.json"
        warmup_path = ROOT / "results" / f"{tag}-01-ready.jsonl"
        report = read_json(report_path)
        warmup = _one_jsonl(warmup_path)
        if controller.get("status") != "failed" or controller.get("current_model") != "oss20b":
            raise ContinuationError("original controller did not finish at the expected OSS boundary")
        if model.get("status") != "failed":
            raise ContinuationError("original OSS model did not finish in the expected failed state")
        if model.get("runner_cleanup_error") is not None:
            raise ContinuationError("original runner cleanup was not clean")
        restoration = model.get("restoration")
        final_restoration = controller.get("final_restoration")
        if not isinstance(restoration, dict) or not isinstance(final_restoration, dict):
            raise ContinuationError("original restoration evidence is missing")
        require_clean_restoration(restoration, "original model restoration")
        require_clean_restoration(final_restoration, "original final restoration")
        if "lfm25" in controller.get("models", {}):
            raise ContinuationError("original controller unexpectedly started LFM")
        oss_cfg = llama_trial.load_trial_config(MODELS["oss20b"])
        audit = audit_model_output(
            self.manifest,
            self.manifest_sha256,
            self.study / "oss20b",
            expected_settings(oss_cfg.model_name, "oss20b"),
        )
        gpu_code = model.get("gpu_window_returncode")
        if gpu_code == 1:
            if controller.get("error") != "owned model cleanup/restoration not clean: 1":
                raise ContinuationError("original controller failure is not the exact warmup-returncode bug")
            accepted = attest_warmup_only_failure(
                report, warmup, expected_plan_sha256=digest(PLAN_V1)
            )
        elif gpu_code == 124:
            if controller.get("error") != "owned model cleanup/restoration not clean: 124":
                raise ContinuationError("original controller failure is not the exact 24h window boundary")
            accepted = attest_owned_24h_timeout(
                report,
                warmup,
                self.study / "oss20b/gpu-window.log",
                audit,
                expected_plan_sha256=digest(PLAN_V1),
            )
        else:
            raise ContinuationError("original GPU-window returncode is unexplained")
        inference_code = model.get("inference_returncode")
        if inference_code == 0 and audit["terminal_count"] != 64:
            raise ContinuationError("original runner claimed success without 64 terminal cases")
        if inference_code == 1 and not audit["resume_candidates"]:
            raise ContinuationError("original runner nonzero has no resumable case evidence")
        if gpu_code == 124 and inference_code is None and audit["resume_candidates"]:
            pass
        elif inference_code not in {0, 1}:
            raise ContinuationError("original inference returncode is unexplained")
        evidence = {
            "accepted_nonzero": accepted,
            "snapshot": snapshot,
            "original_controller_status": controller["status"],
            "original_inference_returncode": inference_code,
            "original_gpu_window_returncode": gpu_code,
            "oss_terminal_count": audit["terminal_count"],
            "oss_resume_candidates": audit["resume_candidates"],
        }
        self.state["original_attestation"] = evidence
        self.state["original_container_state"] = controller.get("original_container_state")
        self.state["original_resident_models"] = controller.get("original_resident_models", [])
        self.save(status="original_attested")
        self.event("original_controller_attested", accepted_nonzero=accepted, terminal_count=audit["terminal_count"])
        return audit

    def _window_paths(self, model_key: str, sequence: int) -> tuple[str, Path, Path, Path]:
        tag = f"public-pilot-cont-{model_key}-{self.continuation_id[-8:]}-{sequence:02d}"
        control = ROOT / "results" / f"{tag}-control.json"
        events = ROOT / "results" / f"{tag}-runtime-events.jsonl"
        directory = self.study / "continuation-windows" / tag
        if control.exists() or events.exists() or directory.exists():
            raise ContinuationError(f"continuation window tag collision: {tag}")
        directory.mkdir(parents=True)
        return tag, control, events, directory

    def _stop_owned_runner(self, child: subprocess.Popen[Any]) -> bool:
        if child.poll() is not None:
            return True
        try:
            os.killpg(child.pid, signal.SIGTERM)
        except ProcessLookupError:
            return True
        deadline = time.monotonic() + 60
        while child.poll() is None and time.monotonic() < deadline:
            time.sleep(0.2)
        return child.poll() is not None

    def run_window(
        self,
        model_key: str,
        config_path: Path,
        output: Path,
        settings: Mapping[str, Any],
        before: Mapping[str, Any],
        sequence: int,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        self.check_interrupt()
        for filename, expected in self.state["validated_source_pins"].items():
            if digest(Path(filename)) != expected:
                raise ContinuationError(f"source pin changed before continuation boundary: {filename}")
        for filename, expected in self.continuation_source_pins.items():
            if digest(Path(filename)) != expected:
                raise ContinuationError(f"continuation source changed before model boundary: {filename}")
        tag, control, events, directory = self._window_paths(model_key, sequence)
        window_command = [
            self.args.python,
            str(GPU_WINDOW),
            "--wait-seconds",
            format(self.args.wait_seconds, "g"),
            "--max-seconds",
            format(self.args.window_seconds, "g"),
            "--long-session",
            "--",
            str(MODEL_SESSION),
            "--runtime-backend",
            "llama",
            "--config",
            str(config_path),
            "--plan",
            str(PLAN_V2),
            "--tag",
            tag,
            "--startup-timeout",
            "480",
            "--hold-after-plan",
            "--control-file",
            str(control),
        ]
        run_command = runner_command(
            self.args.python,
            self.args.runner,
            self.study,
            output,
            settings,
            rerun_failed=bool(before["present"]),
        )
        window = {
            "model": model_key,
            "sequence": sequence,
            "tag": tag,
            "status": "loading",
            "started_at": now(),
            "directory": str(directory),
            "control": str(control),
            "events": str(events),
            "terminal_before": before["terminal_count"],
            "resume_candidates_before": before["resume_candidates"],
            "window_argv": window_command,
            "runner_argv": run_command,
        }
        self.state["windows"].append(window)
        self.save(status="running_window", current_model=model_key)
        outer: subprocess.Popen[Any] | None = None
        child: subprocess.Popen[Any] | None = None
        outer_ended_during_runner = False
        with (directory / "gpu-window.log").open("xb") as window_log, (
            directory / "runner.log"
        ).open("xb") as runner_log:
            try:
                outer = subprocess.Popen(
                    unblocked_exec_command(window_command),
                    cwd=ROOT,
                    stdin=subprocess.DEVNULL,
                    stdout=window_log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                window["gpu_window_pid"] = outer.pid
                self.save()
                holder_args = SimpleNamespace(
                    tag=tag,
                    config=config_path,
                    plan=PLAN_V2,
                    runtime_backend="llama",
                )
                identity = ownership.wait_for_holding(
                    outer,
                    events,
                    args=holder_args,
                    control=control,
                    deadline=time.monotonic() + 1800,
                    signals=self.args.signal_latch,
                )
                self.check_interrupt()
                readiness = attest_v2_readiness(events)
                window.update(
                    status="generating",
                    session={
                        "run_id": identity.run_id,
                        "model_name": identity.model_name,
                        "context": identity.context,
                    },
                    readiness=readiness,
                )
                self.save()
                child = subprocess.Popen(
                    unblocked_exec_command(run_command),
                    cwd=ROOT,
                    stdin=subprocess.DEVNULL,
                    stdout=runner_log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                window["runner_pid"] = child.pid
                self.save()
                while child.poll() is None:
                    self.check_interrupt()
                    outer_code = outer.poll()
                    if outer_code is not None:
                        outer_ended_during_runner = True
                        window["outer_early_returncode"] = outer_code
                        if not self._stop_owned_runner(child):
                            raise ContinuationError("owned runner did not stop after its window ended")
                        break
                    time.sleep(2)
                window["runner_returncode"] = child.poll()
            except BaseException:
                if child is not None and child.poll() is None and not self._stop_owned_runner(child):
                    window["runner_cleanup_error"] = "owned runner did not exit after SIGTERM"
                raise
            finally:
                if outer is not None and outer.poll() is None:
                    window["stop_requested"] = ownership.request_session_stop(control)
                    outer_code, _ = ownership.wait_for_outer_exit(outer, timeout=180, control=control)
                    if outer_code is None:
                        window["outer_term_requested"] = ownership.request_outer_termination(outer)
                        outer_code, _ = ownership.wait_for_outer_exit(outer, timeout=180, control=control)
                else:
                    outer_code = outer.poll() if outer is not None else None
                window["gpu_window_returncode"] = outer_code
                restoration = ownership.restoration_evidence("llama")
                window["restoration"] = restoration
                window["finished_at"] = now()
                self.save()
                require_clean_restoration(restoration, f"{tag} restoration")
        after = audit_model_output(
            self.manifest, self.manifest_sha256, output, settings
        )
        transition = verify_resume_transition(before, after) if before["present"] else {
            "terminal_untouched": [],
            "resumed_nonterminal": [],
        }
        window["terminal_after"] = after["terminal_count"]
        window["attempt_transition"] = transition
        progress = after["terminal_count"] - before["terminal_count"]
        window["new_terminal_count"] = progress
        if progress <= 0:
            window["status"] = "failed_no_progress"
            self.save()
            raise ContinuationError(
                f"{model_key} continuation window made no terminal progress; retry refused"
            )
        if outer_ended_during_runner and window.get("gpu_window_returncode") not in {124}:
            raise ContinuationError(
                f"owned window exited unexpectedly during {model_key}: {window.get('gpu_window_returncode')}"
            )
        if not outer_ended_during_runner and window.get("gpu_window_returncode") != 0:
            raise ContinuationError(
                f"v2 readiness window exited nonzero after runner: {window.get('gpu_window_returncode')}"
            )
        window["status"] = "complete" if after["terminal_count"] == 64 else "needs_resume"
        self.save()
        return after, window

    def complete_model(self, model_key: str, initial: Mapping[str, Any]) -> dict[str, Any]:
        config_path = MODELS[model_key]
        config = llama_trial.load_trial_config(config_path)
        if config.context != 73728:
            raise ContinuationError(f"{model_key} config no longer has matched 72K context")
        settings = expected_settings(config.model_name, model_key)
        output = self.study / model_key
        output.mkdir(exist_ok=True)
        audit = dict(initial)
        sequence = 0
        model_state = {
            "alias": config.model_name,
            "config": str(config_path),
            "config_sha256": digest(config_path),
            "output_dir": str(output),
            "status": "recovering" if audit["present"] else "pending",
            "terminal_initial": audit["terminal_count"],
        }
        self.state["models"][model_key] = model_state
        self.save(current_model=model_key)
        while audit["terminal_count"] < 64:
            sequence += 1
            prior_failure_signature = [
                (row["suite"], row["task_id"], row["status"])
                for row in audit["resume_candidates"]
            ]
            audit, window = self.run_window(
                model_key, config_path, output, settings, audit, sequence
            )
            current_signature = [
                (row["suite"], row["task_id"], row["status"])
                for row in audit["resume_candidates"]
            ]
            if current_signature == prior_failure_signature:
                raise ContinuationError(
                    f"{model_key} repeated the same nonterminal failures; retry refused"
                )
            model_state.update(
                status="generated" if audit["terminal_count"] == 64 else "recovering",
                terminal_count=audit["terminal_count"],
                latest_window=window["tag"],
            )
            self.save()
        if audit["counts"].get("truncated", 0):
            model_state["truncated_terminal_count"] = audit["counts"]["truncated"]
        model_state["status"] = "grading"
        self.save(status="grading", current_model=model_key)
        command = grade_command(
            self.args.grade_image,
            self.args.runner,
            self.study,
            output,
            model_key,
            self.manifest,
            self.continuation_id,
        )
        grading_dir = self.study / "continuation-grading"
        grading_dir.mkdir(exist_ok=True)
        command_path = grading_dir / f"{model_key}-command.json"
        if command_path.exists():
            raise ContinuationError(f"continuation grading command already exists: {command_path}")
        atomic_json(
            command_path,
            {
                "argv": command,
                "isolation": "network disabled; read-only root/public inputs; only model output writable",
            },
        )
        log_path = grading_dir / f"{model_key}.log"
        with log_path.open("xb") as handle:
            completed = subprocess.run(
                command,
                cwd=ROOT,
                stdin=subprocess.DEVNULL,
                stdout=handle,
                stderr=subprocess.STDOUT,
                timeout=21600,
                check=False,
            )
            handle.flush()
            os.fsync(handle.fileno())
        summary_path = output / "grading/summary.json"
        if completed.returncode != 0 or not summary_path.is_file():
            raise ContinuationError(f"official grading failed for {model_key}: {completed.returncode}")
        summary = read_json(summary_path)
        if summary.get("status") != "completed" or summary.get("coverage", {}).get("total") != 64:
            raise ContinuationError(f"official grading summary is incomplete for {model_key}")
        model_state.update(
            status="complete",
            terminal_count=64,
            grading_returncode=0,
            grading_summary=str(summary_path),
            finished_at=now(),
        )
        self.save()
        self.event("model_complete", model=model_key, truncated=audit["counts"].get("truncated", 0))
        return audit

    def restore_original_residency(self) -> dict[str, Any]:
        evidence = ownership.restoration_evidence("llama")
        require_clean_restoration(evidence, "final restoration")
        expected = self.state.get("original_resident_models", [])
        if not isinstance(expected, list):
            raise ContinuationError("original resident model snapshot is invalid")
        if len(expected) == 1:
            row = expected[0]
            payload = {
                "model": gpu_window.ORIGINAL_MODEL,
                "keep_alive": -1,
                "stream": False,
                "options": {"num_ctx": row.get("context_length", 32768)},
            }
            gpu_window._api_request("POST", "/api/generate", 180, payload)
        elif len(expected) > 1:
            raise ContinuationError("original resident model snapshot is ambiguous")
        actual = gpu_window.loaded_models(time.monotonic() + 10)
        if len(expected) != len(actual):
            raise ContinuationError("original resident model count was not restored")
        if expected and (
            actual[0].get("name") != expected[0].get("name")
            or actual[0].get("context_length") != expected[0].get("context_length")
        ):
            raise ContinuationError("original resident model identity/context was not restored")
        return {"lifecycle": evidence, "expected": expected, "actual": actual}

    def run(self) -> int:
        for signum in (signal.SIGTERM, signal.SIGINT):
            signal.signal(signum, self.handle_signal)
        self.args.signal_latch = SimpleNamespace(check=self.check_interrupt)
        self.save(status="waiting_for_original_controller", waiting_for=str(self.study / "controller.lock"))
        self.event("continuation_started")
        try:
            self.lock_handle = acquire_controller_lock(
                self.study / "controller.lock",
                poll_seconds=self.args.poll_seconds,
                on_wait=self.waiting_update,
                check_interrupt=self.check_interrupt,
            )
            self.save(status="original_controller_released", original_lock_acquired_at=now())
            oss_audit = self.attest_original()
            oss_final = self.complete_model("oss20b", oss_audit)
            lfm_cfg = llama_trial.load_trial_config(MODELS["lfm25"])
            lfm_initial = audit_model_output(
                self.manifest,
                self.manifest_sha256,
                self.study / "lfm25",
                expected_settings(lfm_cfg.model_name, "lfm25"),
                allow_absent=True,
            )
            self.complete_model("lfm25", lfm_initial)
            restoration = self.restore_original_residency()
            self.save(
                status="complete",
                current_model=None,
                finished_at=now(),
                final_restoration=restoration,
                oss_terminal_count=oss_final["terminal_count"],
            )
            self.event("continuation_complete")
            return 0
        except ContinuationInterrupted as exc:
            self.save(status="interrupted", error=str(exc), finished_at=now())
            self.event("continuation_interrupted", error=str(exc))
            return 130
        except BaseException as exc:
            with contextlib.suppress(Exception):
                restoration = ownership.restoration_evidence("llama")
                self.state["failure_restoration"] = restoration
            self.save(status="failed", error=f"{type(exc).__name__}: {exc}", finished_at=now())
            self.event("continuation_failed", error=f"{type(exc).__name__}: {exc}")
            return 1
        finally:
            if self.lock_handle is not None:
                fcntl.flock(self.lock_handle.fileno(), fcntl.LOCK_UN)
                self.lock_handle.close()


def resolve_launch_defaults(args: argparse.Namespace) -> None:
    launch = read_json(args.study_dir / "launch.json")
    controller = read_json(args.study_dir / "controller-status.json")
    if args.runner is None:
        argv = launch.get("argv")
        if not isinstance(argv, list) or "--runner" not in argv:
            raise ContinuationError("launch does not identify the immutable runner")
        args.runner = Path(argv[argv.index("--runner") + 1]).resolve()
    else:
        args.runner = args.runner.resolve()
    if args.grade_image is None:
        args.grade_image = launch.get("grader_image")
    if args.grade_image != controller.get("grade_image"):
        raise ContinuationError("launch/controller grader image identity mismatch")


def positive_seconds(value: str, maximum: float) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or not 0 < parsed <= maximum:
        raise argparse.ArgumentTypeError(f"must be finite and from >0 through {maximum:g}")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--study-dir", type=Path, default=DEFAULT_STUDY)
    run_parser.add_argument("--python", default=str(ROOT / ".venv-public-bench/bin/python"))
    run_parser.add_argument("--runner", type=Path)
    run_parser.add_argument("--grade-image")
    run_parser.add_argument("--wait-seconds", type=lambda value: positive_seconds(value, 3600), default=900.0)
    run_parser.add_argument("--window-seconds", type=lambda value: positive_seconds(value, 86400), default=86400.0)
    run_parser.add_argument("--poll-seconds", type=lambda value: positive_seconds(value, 60), default=5.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.study_dir = args.study_dir.resolve()
    try:
        manifest = verify_manifest(args.study_dir)
        resolve_launch_defaults(args)
        if not args.runner.is_file():
            raise ContinuationError(f"immutable runner is missing: {args.runner}")
        if not PLAN_V2.is_file():
            raise ContinuationError("continuation readiness plan is missing")
        with (args.study_dir / "continuation.lock").open("a+b") as continuation_lock:
            try:
                fcntl.flock(
                    continuation_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB
                )
            except BlockingIOError as exc:
                raise ContinuationError("another continuation supervisor is active") from exc
            if (args.study_dir / "continuation-status.json").exists():
                raise ContinuationError("continuation status already exists; refusing to overwrite")
            return Supervisor(args, manifest).run()
    except ContinuationError as exc:
        print(f"public_benchmark_continuation: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
