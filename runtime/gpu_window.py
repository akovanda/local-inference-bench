#!/usr/bin/env python3
"""Borrow an idle GPU window while preserving the original Ollama container."""

from __future__ import annotations

import argparse
import contextlib
import http.client
import json
import math
import os
import signal
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

if __package__:
    from .signal_exec import unblocked_exec_command
else:
    from signal_exec import unblocked_exec_command


ROOT = Path(__file__).resolve().parents[1]
EVENTS_PATH = ROOT / "results" / "gpu-window-events.jsonl"
ORIGINAL_CONTAINER = "local-codex-ollama-cpu"
ORIGINAL_HOST = "127.0.0.1"
ORIGINAL_PORT = 11436
ORIGINAL_MODEL = "gpt-oss-20b:64k"
GPU_LIMIT_MIB = 256
TRIAL_GRACE_SECONDS = 35.0
DEFAULT_MAX_SECONDS = 600.0
LONG_SESSION_MAX_SECONDS = 86_400.0
MODEL_SESSION_TRIAL = (ROOT / "runtime" / "model_session.py").resolve()
ALLOWED_TRIALS = {
    (ROOT / "runtime" / "native_trial.py").resolve(),
    (ROOT / "runtime" / "strata_trial.py").resolve(),
    (ROOT / "runtime" / "llama_trial.py").resolve(),
    (ROOT / "runtime" / "expert_cache_trial.py").resolve(),
    MODEL_SESSION_TRIAL,
}


class WindowError(RuntimeError):
    pass


class WindowInterrupted(BaseException):
    def __init__(self, signum: int) -> None:
        super().__init__(f"signal {signum}")
        self.signum = signum


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


class EventLog:
    def __init__(self, window_id: str) -> None:
        self.window_id = window_id
        EVENTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    def write(self, event: str, **fields: Any) -> None:
        row = {"at": utc_now(), "event": event, "window_id": self.window_id, **fields}
        with EVENTS_PATH.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        print(json.dumps(row, separators=(",", ":")), flush=True)


def cleanup_event(events: EventLog, event: str, **fields: Any) -> None:
    """Do not let a logging failure prevent process or container cleanup."""
    try:
        events.write(event, **fields)
    except OSError:
        pass


def _docker(*args: str, timeout: float = 10.0) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            unblocked_exec_command(["docker", *args]),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise WindowError(f"docker {' '.join(args[:1])} could not complete") from exc


def original_state() -> dict[str, Any]:
    completed = _docker("inspect", "--format", "{{json .State}}", ORIGINAL_CONTAINER)
    if completed.returncode != 0:
        raise WindowError("original Ollama container could not be inspected")
    try:
        state = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise WindowError("original Ollama container returned invalid state") from exc
    if not isinstance(state, dict):
        raise WindowError("original Ollama container returned invalid state")
    return state


def require_original_available() -> None:
    state = original_state()
    if state.get("Paused") is True:
        raise WindowError("original Ollama container is already paused; refusing ownership")
    if state.get("Running") is not True:
        raise WindowError("original Ollama container is not running")


def pause_original() -> None:
    completed = _docker("pause", ORIGINAL_CONTAINER, timeout=15)
    if completed.returncode != 0:
        raise WindowError("failed to pause original Ollama container")


def restore_original(events: EventLog, attempts: int = 5) -> None:
    last_problem = "unknown"
    for attempt in range(1, attempts + 1):
        cleanup_event(events, "original_unpause_attempt", attempt=attempt)
        try:
            completed = _docker("unpause", ORIGINAL_CONTAINER, timeout=10)
        except WindowError as exc:
            completed = None
            last_problem = str(exc)
        try:
            state = original_state()
        except WindowError as exc:
            last_problem = str(exc)
        else:
            if state.get("Running") is True and state.get("Paused") is not True:
                cleanup_event(events, "original_restored", attempt=attempt)
                return
            last_problem = "container did not report running and unpaused"
        if completed is not None and completed.returncode != 0 and last_problem == "unknown":
            last_problem = "docker unpause failed"
        if attempt < attempts:
            time.sleep(1)
    raise WindowError(f"could not restore original Ollama container: {last_problem}")


def _api_request(method: str, path: str, timeout: float, payload: dict[str, Any] | None = None) -> Any:
    connection = http.client.HTTPConnection(ORIGINAL_HOST, ORIGINAL_PORT, timeout=max(0.1, timeout))
    body = None if payload is None else json.dumps(payload, separators=(",", ":")).encode("utf-8")
    headers = {"Accept": "application/json"}
    if body is not None:
        headers["Content-Type"] = "application/json"
    try:
        connection.request(method, path, body=body, headers=headers)
        response = connection.getresponse()
        data = response.read(2 * 1024 * 1024)
        if response.status < 200 or response.status >= 300:
            raise WindowError(f"original Ollama API returned HTTP {response.status}")
        return json.loads(data) if data else {}
    except (OSError, TimeoutError, http.client.HTTPException, json.JSONDecodeError) as exc:
        raise WindowError("original Ollama API request failed") from exc
    finally:
        connection.close()


def request_model_unload(deadline: float) -> None:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise WindowError("timed out before model unload completed")
    _api_request(
        "POST",
        "/api/generate",
        remaining,
        {"model": ORIGINAL_MODEL, "keep_alive": 0, "stream": False},
    )


def loaded_models(deadline: float) -> list[dict[str, Any]]:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise WindowError("timed out while checking original model residency")
    result = _api_request("GET", "/api/ps", min(3.0, remaining))
    if not isinstance(result, dict) or not isinstance(result.get("models"), list):
        raise WindowError("original Ollama /api/ps returned invalid data")
    return [model for model in result["models"] if isinstance(model, dict)]


def gpu_compute_processes(timeout: float = 10.0) -> list[dict[str, Any]]:
    try:
        completed = subprocess.run(
            unblocked_exec_command([
                "nvidia-smi",
                "--query-compute-apps=pid,process_name,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ]),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=max(0.1, timeout),
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise WindowError("could not inspect GPU compute processes") from exc
    if completed.returncode != 0:
        raise WindowError("nvidia-smi could not inspect GPU compute processes")
    processes: list[dict[str, Any]] = []
    for row in completed.stdout.splitlines():
        if not row.strip():
            continue
        fields = [field.strip() for field in row.rsplit(",", 2)]
        if len(fields) != 3:
            raise WindowError("nvidia-smi returned an invalid compute-process row")
        try:
            processes.append({"pid": int(fields[0]), "name": fields[1], "used_mib": int(fields[2])})
        except ValueError as exc:
            raise WindowError("nvidia-smi returned non-numeric compute-process data") from exc
    return processes


def busy_gpu_processes(timeout: float = 10.0) -> list[dict[str, Any]]:
    return [process for process in gpu_compute_processes(timeout) if process["used_mib"] > GPU_LIMIT_MIB]


def wait_for_idle(wait_seconds: float, events: EventLog) -> None:
    deadline = time.monotonic() + wait_seconds
    events.write("model_unload_requested", model=ORIGINAL_MODEL)
    request_model_unload(deadline)
    last_unload = time.monotonic()
    while True:
        models = loaded_models(deadline)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise WindowError("timed out waiting for original model and GPU to become idle")
        busy = busy_gpu_processes(min(10.0, remaining))
        if not models and not busy:
            events.write("resources_idle")
            return
        # A request may have arrived between the prior unload and residency poll.
        if models and time.monotonic() - last_unload >= 2.0:
            request_model_unload(deadline)
            last_unload = time.monotonic()
        time.sleep(min(1.0, max(0.0, deadline - time.monotonic())))


def validate_trial(command: Sequence[str]) -> list[str]:
    if not command:
        raise WindowError("a trial command is required after --")
    candidate = Path(command[0])
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise WindowError("trial script does not exist") from exc
    if resolved not in ALLOWED_TRIALS:
        raise WindowError("trial must be one of the configured runtime trial scripts")
    return [sys.executable, str(resolved), *command[1:]]


@dataclass
class SignalState:
    child: subprocess.Popen[Any] | None = None
    received: int | None = None
    forced_kill: bool = False

    def handle(self, signum: int, _frame: Any) -> None:
        self.received = signum
        if self.child is None:
            raise WindowInterrupted(signum)
        if self.child.poll() is None:
            try:
                os.killpg(self.child.pid, signum)
            except ProcessLookupError:
                pass


@contextlib.contextmanager
def defer_termination_signals():
    if not hasattr(signal, "pthread_sigmask"):
        yield
        return
    watched = {signal.SIGINT, signal.SIGTERM}
    previous = signal.pthread_sigmask(signal.SIG_BLOCK, watched)
    try:
        yield
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, previous)


def monitor_trial(
    process: subprocess.Popen[Any],
    max_seconds: float,
    signals: SignalState,
    events: EventLog,
) -> int:
    deadline = time.monotonic() + max_seconds
    grace_deadline: float | None = None
    timed_out = False
    forwarded_logged = False
    while True:
        returncode = process.poll()
        if returncode is not None:
            if timed_out:
                return 124
            if returncode < 0:
                return 128 - returncode
            return returncode
        now = time.monotonic()
        if signals.received is not None and grace_deadline is None:
            grace_deadline = now + TRIAL_GRACE_SECONDS
        if signals.received is not None and not forwarded_logged:
            events.write("signal_forwarded", signal=signals.received, child_pgid=process.pid)
            forwarded_logged = True
        if now >= deadline and grace_deadline is None:
            timed_out = True
            grace_deadline = now + TRIAL_GRACE_SECONDS
            events.write("trial_timeout", max_seconds=max_seconds, child_pgid=process.pid)
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        if grace_deadline is not None and now >= grace_deadline:
            signals.forced_kill = True
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                pass
            return 124 if timed_out else 128 + int(signals.received or signal.SIGTERM)
        time.sleep(0.2)


def terminate_owned_trial(process: subprocess.Popen[Any], events: EventLog) -> bool:
    """Bound cleanup for an owned trial that escaped the normal monitor path."""
    if process.poll() is not None:
        return False
    cleanup_event(events, "trial_cleanup_started", child_pgid=process.pid)
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return False
    deadline = time.monotonic() + TRIAL_GRACE_SECONDS
    while process.poll() is None and time.monotonic() < deadline:
        time.sleep(0.2)
    if process.poll() is None:
        cleanup_event(events, "trial_cleanup_kill", child_pgid=process.pid)
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired as exc:
            raise WindowError("owned trial process group did not exit") from exc
        return True
    return False


def run_window(
    trial_command: Sequence[str],
    wait_seconds: float,
    max_seconds: float,
    events: EventLog,
    signals: SignalState,
) -> int:
    paused_by_us = False
    process: subprocess.Popen[Any] | None = None
    outcome = 1
    try:
        require_original_available()
        events.write("original_verified", container=ORIGINAL_CONTAINER)
        wait_for_idle(wait_seconds, events)
        # Signals cannot land between Docker's successful pause and ownership registration.
        with defer_termination_signals():
            require_original_available()
            if loaded_models(time.monotonic() + 3.0):
                raise WindowError("original model reloaded before pause; refusing to pause")
            if busy_gpu_processes(3.0):
                raise WindowError("GPU became busy before pause; refusing to pause")
            pause_original()
            paused_by_us = True

        raced = busy_gpu_processes(3.0)
        if raced:
            raise WindowError("GPU memory raced back after pause; aborting window")
        events.write("original_paused", container=ORIGINAL_CONTAINER)

        # Child creation and ownership registration are atomic with respect to
        # the termination signals this wrapper handles.
        with defer_termination_signals():
            process = subprocess.Popen(
                unblocked_exec_command(trial_command),
                stdin=subprocess.DEVNULL,
                start_new_session=True,
                close_fds=True,
            )
            signals.child = process
        events.write("trial_started", child_pid=process.pid, max_seconds=max_seconds)
        outcome = monitor_trial(process, max_seconds, signals, events)
        events.write("trial_finished", returncode=outcome)
    except WindowInterrupted as exc:
        outcome = 128 + exc.signum
        events.write("window_interrupted", signal=exc.signum)
    except (WindowError, OSError, subprocess.SubprocessError) as exc:
        outcome = 1
        events.write("window_failed", error_type=type(exc).__name__, error=str(exc))
    finally:
        cleanup_problem: BaseException | None = None
        restore_problem: BaseException | None = None
        orphan_guard_problem: str | None = None
        try:
            # Keep restoration indivisible with respect to wrapper termination.
            # A pending signal is delivered only after our child is gone and the
            # original container is unpaused.
            with defer_termination_signals():
                if process is not None:
                    try:
                        signals.forced_kill = (
                            terminate_owned_trial(process, events) or signals.forced_kill
                        )
                    except (WindowError, OSError, subprocess.SubprocessError) as exc:
                        cleanup_problem = exc
                signals.child = None
                if paused_by_us:
                    if process is not None and cleanup_problem is not None:
                        orphan_guard_problem = "owned trial process group cleanup was not confirmed"
                    elif process is not None and signals.forced_kill:
                        orphan_guard_problem = (
                            "owned trial required SIGKILL; nested server cleanup is not confirmed"
                        )
                    elif process is not None:
                        try:
                            busy = busy_gpu_processes(10.0)
                        except WindowError as exc:
                            orphan_guard_problem = f"GPU cleanup could not be verified: {exc}"
                        else:
                            if busy:
                                summary = ", ".join(
                                    f"pid {item['pid']} ({item['used_mib']} MiB)" for item in busy
                                )
                                orphan_guard_problem = (
                                    "GPU compute process remained after trial cleanup: " + summary
                                )
                    if orphan_guard_problem is None:
                        try:
                            restore_original(events)
                        except (WindowError, OSError, subprocess.SubprocessError) as exc:
                            restore_problem = exc
                    else:
                        cleanup_event(
                            events,
                            "restore_blocked_orphan_guard",
                            reason=orphan_guard_problem,
                            original_left_paused=True,
                        )
        except WindowInterrupted as exc:
            outcome = 128 + exc.signum
            events.write("window_interrupted", signal=exc.signum)
        if cleanup_problem is not None:
            outcome = 1
            events.write(
                "trial_cleanup_failed",
                error_type=type(cleanup_problem).__name__,
                error=str(cleanup_problem),
            )
        if orphan_guard_problem is not None:
            outcome = 1 if outcome == 0 else outcome
        if restore_problem is not None:
            outcome = 1
            events.write(
                "restore_failed",
                error_type=type(restore_problem).__name__,
                error=str(restore_problem),
            )
    events.write("window_complete", returncode=outcome)
    return outcome


def positive_seconds(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be a finite value greater than zero")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait-seconds", type=positive_seconds, default=300.0)
    parser.add_argument("--max-seconds", type=positive_seconds, default=240.0)
    parser.add_argument(
        "--long-session",
        action="store_true",
        help="explicitly allow up to 86400 seconds for runtime/model_session.py only",
    )
    parser.add_argument("trial", nargs=argparse.REMAINDER, help="-- runtime/native_trial.py ARGS...")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    raw_trial = list(args.trial)
    if raw_trial and raw_trial[0] == "--":
        raw_trial.pop(0)
    try:
        trial_command = validate_trial(raw_trial)
    except WindowError as exc:
        parser.error(str(exc))
    is_model_session = Path(trial_command[1]).resolve() == MODEL_SESSION_TRIAL
    if args.long_session and not is_model_session:
        parser.error("--long-session is only permitted for runtime/model_session.py")
    max_allowed = LONG_SESSION_MAX_SECONDS if args.long_session else DEFAULT_MAX_SECONDS
    if args.max_seconds > max_allowed:
        if is_model_session and not args.long_session:
            parser.error(
                "--max-seconds above 600 for runtime/model_session.py requires --long-session"
            )
        parser.error(f"--max-seconds cannot exceed {int(max_allowed)}")

    window_id = str(uuid.uuid4())
    events = EventLog(window_id)
    state = SignalState()
    signal.signal(signal.SIGINT, state.handle)
    signal.signal(signal.SIGTERM, state.handle)
    events.write(
        "window_started",
        trial=Path(trial_command[1]).name,
        wait_seconds=args.wait_seconds,
        max_seconds=args.max_seconds,
        long_session=args.long_session,
    )
    return run_window(trial_command, args.wait_seconds, args.max_seconds, events, state)


if __name__ == "__main__":
    raise SystemExit(main())
