#!/usr/bin/env python3
"""Drive the held OSS20B endpoint through full64, grading, and GPU release."""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

if __package__:
    from . import gpu_window
    from . import public_model_expansion as expansion
    from .signal_exec import unblocked_exec_command
else:
    import gpu_window
    import public_model_expansion as expansion
    from signal_exec import unblocked_exec_command


ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "runtime"
STUDY = ROOT / "results/public-model-expansion-20261024-v1"
OUTPUT = STUDY / "oss-repair"
HANDOFF = OUTPUT / "handoff.json"
MANIFEST = ROOT / "results/public-benchmark-pilot-20261023-v1/manifest.json"
MANIFEST_SHA256 = expansion.MANIFEST_SHA256
CONFIG = RUNTIME / "gpt20b-llama-specialists-72k-wide64k.json"
EXTENDED_RUNNER = RUNTIME / "public_benchmark_extended.py"
BASE_RUNNER = RUNTIME / "public_benchmark_pilot.py"
REPORTER = RUNTIME / "public_model_expansion_report.py"
PROBE = RUNTIME / "oss20b_public_retry_probe.py"
CONTROL = ROOT / "results/oss20b-public-retry-20261024-v1-held-control.json"
ENDPOINT = "http://127.0.0.1:18081/v1"
MODEL = "gpt-oss-20b-mxfp4"
GRADE_IMAGE = "sha256:85a1ac1c9042d3f2e6066f1a58bdda1ec9951832a6f3d1e4fa6a87fcd8c6efd7"
MAX_RUNNER_PASSES = 16
REPORT_DEBOUNCE_SECONDS = 30.0

SELECTED_PROFILE: dict[str, Any] = {
    "label": "low-greedy-rerun",
    "reasoning_effort": "low",
    "temperature": 0.0,
    "top_p": 1.0,
    "top_k": 0,
    "min_p": 0.0,
    "repeat_penalty": 1.0,
}


class FullRunError(RuntimeError):
    """The selected full64 run could not safely advance."""


def read_handoff() -> dict[str, Any]:
    value = expansion.read_json(HANDOFF)
    if value.get("manifest_sha256") != MANIFEST_SHA256:
        raise FullRunError("handoff manifest binding mismatch")
    return value


def validate_selected_profile(handoff: Mapping[str, Any]) -> dict[str, Any]:
    selected = handoff.get("profile_selected")
    if not isinstance(selected, dict):
        raise FullRunError("handoff has not selected the frozen full64 profile")
    actual = {key: selected.get(key) for key in SELECTED_PROFILE}
    if actual != SELECTED_PROFILE:
        raise FullRunError("handoff selected profile does not match low-greedy-rerun")
    return dict(SELECTED_PROFILE)


def runner_command(python: str, profile: Mapping[str, Any]) -> list[str]:
    return [
        python,
        str(EXTENDED_RUNNER),
        "run",
        "--manifest",
        str(MANIFEST),
        "--endpoint",
        ENDPOINT,
        "--backend-adapter",
        "llama",
        "--model",
        MODEL,
        "--model-key",
        "oss20b_repaired",
        "--output-dir",
        str(OUTPUT),
        "--max-tokens",
        "65536",
        "--request-timeout",
        "21600",
        "--seed",
        "20261023",
        "--reasoning-effort",
        str(profile["reasoning_effort"]),
        "--temperature",
        str(profile["temperature"]),
        "--top-p",
        str(profile["top_p"]),
        "--top-k",
        str(profile["top_k"]),
        "--min-p",
        str(profile["min_p"]),
        "--repeat-penalty",
        str(profile["repeat_penalty"]),
        "--rerun-failed",
    ]


def full_coverage() -> bool:
    return expansion.full_coverage(OUTPUT)


def terminal_count() -> int:
    status = expansion.case_coverage(OUTPUT)
    counts = status.get("counts") if isinstance(status, dict) else None
    if not isinstance(counts, dict):
        return 0
    return int(counts.get("completed", 0)) + int(counts.get("truncated", 0))


def validate_held_model(handoff: Mapping[str, Any]) -> dict[str, Any]:
    gpu = handoff.get("gpu")
    if not isinstance(gpu, dict):
        raise FullRunError("handoff lacks held GPU identity")
    server_pid = gpu.get("server_pid")
    server_start_ticks = gpu.get("server_start_ticks")
    server_argv = gpu.get("argv")
    if not isinstance(server_pid, int) or not isinstance(server_start_ticks, int) or not isinstance(server_argv, list):
        raise FullRunError("handoff held server identity is incomplete")
    current = expansion.process_identity(server_pid)
    if current["start_ticks"] != server_start_ticks or current["argv"] != server_argv:
        raise FullRunError("held server PID/start-ticks/argv identity changed")
    config = expansion.read_json(CONFIG)
    if config.get("context") != 73728 or config.get("model_name") != MODEL:
        raise FullRunError("runtime config is not the selected 73,728-context OSS20B config")
    if expansion.digest(CONFIG) != handoff.get("config_sha256"):
        raise FullRunError("runtime config SHA-256 differs from handoff")
    return current


def snapshot_sources() -> dict[str, Any]:
    destination = OUTPUT / "full64-artifacts"
    destination.mkdir(parents=True, exist_ok=True)
    sources = (EXTENDED_RUNNER, BASE_RUNNER, REPORTER, PROBE, Path(__file__).resolve())
    receipt: dict[str, Any] = {}
    for source in sources:
        target = destination / source.name
        if target.exists():
            if expansion.digest(target) != expansion.digest(source):
                raise FullRunError(f"existing source snapshot differs: {target}")
        else:
            shutil.copy2(source, target)
        receipt[source.name] = {"path": str(target), "sha256": expansion.digest(target)}
    expansion.atomic_json(destination / "source-pins.json", receipt)
    return receipt


def update_report(python: str, events: Path, reason: str) -> None:
    command = [python, str(REPORTER), "--study-dir", str(STUDY)]
    log_path = OUTPUT / "full64-report-updates.log"
    try:
        with log_path.open("ab", buffering=0) as log:
            completed = subprocess.run(
                unblocked_exec_command(command),
                cwd=ROOT,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=60,
                check=False,
            )
        expansion.append_jsonl(
            events,
            {"at": expansion.now(), "event": "report_update", "reason": reason, "returncode": completed.returncode},
        )
    except (OSError, subprocess.SubprocessError) as exc:
        expansion.append_jsonl(
            events,
            {"at": expansion.now(), "event": "report_update_failed", "reason": reason, "error": str(exc)},
        )


def _wait_runner_with_reports(process: subprocess.Popen[Any], python: str, events: Path) -> int:
    last_count = terminal_count()
    last_report = 0.0
    while process.poll() is None:
        count = terminal_count()
        now = time.monotonic()
        if count != last_count and now - last_report >= REPORT_DEBOUNCE_SECONDS:
            update_report(python, events, f"terminal_count_{count}")
            last_report = time.monotonic()
        last_count = count
        time.sleep(2)
    count = terminal_count()
    if count != last_count or time.monotonic() - last_report >= REPORT_DEBOUNCE_SECONDS:
        update_report(python, events, f"runner_exit_terminal_count_{count}")
    return int(process.returncode or 0)


def run_scored(python: str, profile: Mapping[str, Any], events: Path) -> None:
    command = runner_command(python, profile)
    passes_dir = OUTPUT / "full64-runner-passes"
    passes_dir.mkdir(parents=True, exist_ok=True)
    for number in range(1, MAX_RUNNER_PASSES + 1):
        if full_coverage():
            return
        command_path = passes_dir / f"pass-{number:02d}-command.json"
        log_path = passes_dir / f"pass-{number:02d}.log"
        expansion.atomic_json(command_path, {"argv": command, "started_at": expansion.now()})
        expansion.append_jsonl(events, {"at": expansion.now(), "event": "runner_pass_started", "pass": number})
        with log_path.open("ab", buffering=0) as log:
            process = subprocess.Popen(
                unblocked_exec_command(command),
                cwd=ROOT,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                close_fds=True,
            )
            identity = expansion.wait_process_identity(process.pid, str(EXTENDED_RUNNER))
            expansion.atomic_json(command_path, {"argv": command, "started_at": expansion.now(), "process": identity})
            returncode = _wait_runner_with_reports(process, python, events)
        expansion.append_jsonl(
            events,
            {
                "at": expansion.now(),
                "event": "runner_pass_finished",
                "pass": number,
                "returncode": returncode,
                "coverage": expansion.case_coverage(OUTPUT),
            },
        )
        if full_coverage():
            return
        time.sleep(5)
    raise FullRunError(f"full64 did not reach terminal coverage in {MAX_RUNNER_PASSES} runner passes")


def run_official_grade(manifest: Mapping[str, Any], events: Path) -> dict[str, Any]:
    grade_runner = OUTPUT / "full64-artifacts/public_benchmark_pilot.py"
    command = expansion.grade_command(
        GRADE_IMAGE, STUDY, OUTPUT, "oss20b_repaired", manifest, grade_runner
    )
    grade_dir = OUTPUT / "full64-grade-process"
    grade_dir.mkdir(parents=True, exist_ok=True)
    expansion.atomic_json(grade_dir / "command.json", {"argv": command, "started_at": expansion.now()})
    log_path = grade_dir / "container.log"
    expansion.append_jsonl(events, {"at": expansion.now(), "event": "official_grading_started"})
    with log_path.open("ab", buffering=0) as log:
        completed = subprocess.run(
            unblocked_exec_command(command),
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode != 0:
        raise FullRunError(f"isolated official grader exited {completed.returncode}")
    summary = expansion.read_json(OUTPUT / "grading/summary.json")
    expansion.validate_grade_coverage(summary)
    expansion.append_jsonl(events, {"at": expansion.now(), "event": "official_grading_complete"})
    update_report(str(ROOT / ".venv-public-bench/bin/python"), events, "official_grading_complete")
    return summary


def _process_matches(pid: int, start_ticks: int | None) -> bool:
    try:
        current = expansion.process_identity(pid)
    except expansion.ExpansionError:
        return False
    return start_ticks is None or current["start_ticks"] == start_ticks


def release_gpu_window(handoff: Mapping[str, Any], events: Path) -> dict[str, Any]:
    gpu = handoff["gpu"]
    session_pid = gpu.get("session_pid")
    session_start_ticks = gpu.get("session_start_ticks")
    server_pid = gpu.get("server_pid")
    server_start_ticks = gpu.get("server_start_ticks")
    if not isinstance(session_pid, int) or not isinstance(session_start_ticks, int):
        raise FullRunError("held model-session PID/start-ticks identity is incomplete")
    if not isinstance(server_pid, int) or not isinstance(server_start_ticks, int):
        raise FullRunError("held server PID/start-ticks identity is incomplete")
    if not _process_matches(session_pid, session_start_ticks):
        raise FullRunError("held model session identity changed before clean release")
    if not _process_matches(server_pid, server_start_ticks):
        raise FullRunError("held server identity changed before clean release")
    expansion.atomic_json(CONTROL, {"stop": True})
    expansion.append_jsonl(events, {"at": expansion.now(), "event": "gpu_release_requested", "control": str(CONTROL)})
    deadline = time.monotonic() + 180.0
    while time.monotonic() < deadline:
        session_alive = _process_matches(session_pid, session_start_ticks)
        server_alive = _process_matches(server_pid, server_start_ticks)
        try:
            state = gpu_window.original_state()
            busy = gpu_window.busy_gpu_processes(10.0)
        except gpu_window.WindowError:
            state = {}
            busy = [{"error": "inspection_failed"}]
        if not session_alive and not server_alive and not busy and state.get("Running") is True and state.get("Paused") is not True:
            receipt = {
                "released_at": expansion.now(),
                "server_process_gone": True,
                "session_process_gone": True,
                "busy_gpu_processes": [],
                "original_container_running": True,
                "original_container_paused": False,
            }
            expansion.append_jsonl(events, {"at": expansion.now(), "event": "gpu_release_complete", **receipt})
            return receipt
        time.sleep(1)
    raise FullRunError("GPU window did not cleanly release within 180 seconds")


def final_handoff(
    prior: Mapping[str, Any],
    profile: Mapping[str, Any],
    summary: Mapping[str, Any],
    release: Mapping[str, Any],
) -> dict[str, Any]:
    value = dict(prior)
    value.update(
        {
            "status": "complete",
            "profile_selected": dict(profile),
            "output_dir": str(OUTPUT),
            "full64_official_grading_complete": True,
            "adoptable": True,
            "grading_summary": dict(summary),
        }
    )
    gpu = dict(value.get("gpu", {}))
    gpu.update({"state": "clear", "release": dict(release)})
    value["gpu"] = gpu
    return value


@contextlib.contextmanager
def supervisor_lock(path: Path) -> Iterable[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+b")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        handle.close()
        raise FullRunError("another OSS20B full64 supervisor owns the lock") from exc
    try:
        yield
    finally:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def supervise(args: argparse.Namespace) -> int:
    events = OUTPUT / "full64-events.jsonl"
    with supervisor_lock(OUTPUT / ".full64-supervisor.lock"):
        manifest = expansion.verify_manifest(MANIFEST)
        handoff = read_handoff()
        profile = validate_selected_profile(handoff)
        server_identity = validate_held_model(handoff)
        sources = snapshot_sources()
        working = dict(handoff)
        working.update(
            {
                "status": "full64_running",
                "adoptable": False,
                "full64_official_grading_complete": False,
                "profile_selected": profile,
                "full64_supervisor": expansion.process_identity(os.getpid()),
                "source_pins": sources,
            }
        )
        working_gpu = dict(working["gpu"])
        working_gpu.update({"state": "owned_scoring", "validated_server_process": server_identity})
        working["gpu"] = working_gpu
        expansion.atomic_json(HANDOFF, working)
        expansion.append_jsonl(events, {"at": expansion.now(), "event": "full64_started", "profile": profile})
        try:
            run_scored(args.python, profile, events)
            summary = run_official_grade(manifest, events)
            release = release_gpu_window(working, events)
            expansion.atomic_json(HANDOFF, final_handoff(working, profile, summary, release))
            update_report(args.python, events, "final_handoff")
            return 0
        except BaseException as exc:
            failed = read_handoff()
            failed.update(
                {
                    "status": "full64_failed",
                    "adoptable": False,
                    "full64_official_grading_complete": False,
                    "error": {"type": type(exc).__name__, "message": str(exc)},
                }
            )
            expansion.atomic_json(HANDOFF, failed)
            expansion.append_jsonl(events, {"at": expansion.now(), "event": "full64_failed", "error_type": type(exc).__name__, "error": str(exc)})
            raise


def launch(args: argparse.Namespace) -> int:
    if (OUTPUT / "full64-launch.json").exists():
        raise FullRunError("full64 launch receipt already exists")
    handoff = read_handoff()
    validate_selected_profile(handoff)
    validate_held_model(handoff)
    command = [
        args.python,
        str(Path(__file__).resolve()),
        "supervise",
        "--python",
        args.python,
    ]
    log_path = OUTPUT / "full64-supervisor.log"
    with log_path.open("ab", buffering=0) as log:
        process = subprocess.Popen(
            unblocked_exec_command(command),
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )
    identity = expansion.wait_process_identity(process.pid, str(Path(__file__).resolve()))
    receipt = {
        "schema_version": 1,
        "launched_at": expansion.now(),
        "profile_selected": dict(SELECTED_PROFILE),
        "argv": command,
        "process": identity,
        "source_sha256": expansion.digest(Path(__file__).resolve()),
        "manifest_sha256": MANIFEST_SHA256,
    }
    expansion.atomic_json(OUTPUT / "full64-launch.json", receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("launch", "supervise"):
        child = subparsers.add_parser(name)
        child.add_argument("--python", default=str(ROOT / ".venv-public-bench/bin/python"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return launch(args) if args.command == "launch" else supervise(args)
    except (FullRunError, expansion.ExpansionError, gpu_window.WindowError, OSError, subprocess.SubprocessError) as exc:
        print(f"oss20b full64 supervisor: {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
