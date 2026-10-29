#!/usr/bin/env python3
"""Supervise the two pinned Flash-Next range downloads until verified publication.

Only failures that ``range_fetch.py`` identifies as transient are restarted.  Its
manifest and per-chunk hashes remain the source of resume state and the final
whole-file SHA-256 remains the publication gate.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runtime import prepare_flash_next


RANGE_FETCH = ROOT / "runtime" / "range_fetch.py"
STATE_PATH = ROOT / "results" / "flash-next-download-supervisor.json"
LOCK_PATH = ROOT / "results" / "flash-next-download-supervisor.lock"
LOG_DIR = ROOT / "results"

TRANSIENT_MESSAGES = (
    "request reached its 60 second deadline",
    "request exhausted its retry budget",
    "ranged response byte count was incomplete",
    "request failed with ",
    "transient HTTP status ",
)


@dataclass(frozen=True)
class JobSpec:
    label: str
    file: prepare_flash_next.FileSpec
    workers: int

    @property
    def log_path(self) -> Path:
        return LOG_DIR / f"flash-next-download-{self.label}.log"


JOBS = (
    JobSpec("shard1", prepare_flash_next.FILES[0], 8),
    JobSpec("shard2", prepare_flash_next.FILES[1], 4),
)


def atomic_write_json(path: Path, value: Any) -> None:
    prepare_flash_next.atomic_write_json(path, value)


def classify_failure(output: str) -> str:
    """Return ``transient`` only for errors explicitly transient in range_fetch."""
    return "transient" if any(message in output for message in TRANSIENT_MESSAGES) else "permanent"


def download_command(job: JobSpec) -> list[str]:
    return [
        sys.executable,
        str(RANGE_FETCH),
        "--url",
        job.file.url,
        "--output",
        str(job.file.path),
        "--size",
        str(job.file.size),
        "--sha256",
        job.file.sha256,
        "--workers",
        str(job.workers),
        "--chunk-mib",
        "32",
        "--retries",
        "5",
    ]


def manifest_progress(job: JobSpec) -> dict[str, Any]:
    if job.file.path.is_file() and job.file.path.stat().st_size == job.file.size:
        return {"completed_bytes": job.file.size, "expected_bytes": job.file.size, "percent": 100.0}
    path = Path(str(job.file.path) + ".range-part.manifest.json")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        identity = (
            manifest.get("url"),
            manifest.get("expected_size"),
            manifest.get("expected_sha256"),
            manifest.get("chunk_size"),
        )
        expected = (job.file.url, job.file.size, job.file.sha256, 32 * 1024 * 1024)
        if identity != expected or not isinstance(manifest.get("chunks"), dict):
            raise ValueError("manifest identity differs from the pinned request")
        completed = sum(int(entry["bytes"]) for entry in manifest["chunks"].values())
        if completed < 0 or completed > job.file.size:
            raise ValueError("manifest byte count is outside the pinned size")
        return {
            "completed_bytes": completed,
            "expected_bytes": job.file.size,
            "percent": round(100.0 * completed / job.file.size, 3),
            "manifest_updated_at": manifest.get("updated_at"),
        }
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        return {
            "completed_bytes": 0,
            "expected_bytes": job.file.size,
            "percent": 0.0,
            "progress_error": str(exc),
        }


def log_tail(path: Path, offset: int, limit: int = 1024 * 1024) -> str:
    try:
        with path.open("rb") as handle:
            end = handle.seek(0, os.SEEK_END)
            handle.seek(max(offset, end - limit))
            return handle.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


class Supervisor:
    def __init__(self, poll_seconds: float, backoff_base: float, backoff_max: float) -> None:
        self.poll_seconds = poll_seconds
        self.backoff_base = backoff_base
        self.backoff_max = backoff_max
        self.stop = False
        self.started_at = prepare_flash_next.utc_now()
        self.runtime: dict[str, dict[str, Any]] = {
            job.label: {
                "process": None,
                "log_handle": None,
                "log_offset": 0,
                "attempts": 0,
                "next_start": 0.0,
                "status": "pending",
                "last_exit_code": None,
                "last_failure_class": None,
                "last_failure": None,
            }
            for job in JOBS
        }

    def request_stop(self, _signum: int, _frame: object) -> None:
        self.stop = True

    def spawn(self, job: JobSpec) -> None:
        state = self.runtime[job.label]
        job.log_path.parent.mkdir(parents=True, exist_ok=True)
        log_handle = job.log_path.open("ab", buffering=0)
        offset = log_handle.tell()
        banner = (
            f"\n[{prepare_flash_next.utc_now()}] supervisor attempt {state['attempts'] + 1}; "
            f"workers={job.workers}\n"
        ).encode("utf-8")
        log_handle.write(banner)
        process = subprocess.Popen(
            download_command(job),
            cwd=ROOT,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
        state.update(
            process=process,
            log_handle=log_handle,
            log_offset=offset,
            attempts=state["attempts"] + 1,
            status="running",
            child_pid=process.pid,
            last_started_at=prepare_flash_next.utc_now(),
            next_start=0.0,
        )

    def observe_exit(self, job: JobSpec) -> None:
        state = self.runtime[job.label]
        process: subprocess.Popen[bytes] = state["process"]
        code = process.poll()
        if code is None:
            return
        log_handle = state.pop("log_handle")
        if log_handle is not None:
            log_handle.close()
        state["process"] = None
        state.pop("child_pid", None)
        state["last_exit_code"] = code
        state["last_exited_at"] = prepare_flash_next.utc_now()
        if code == 0:
            state.update(status="complete", last_failure_class=None, last_failure=None)
            return
        tail = log_tail(job.log_path, int(state["log_offset"]))
        failure_class = classify_failure(tail)
        last_line = next((line for line in reversed(tail.splitlines()) if line.strip()), "no error text")
        state["last_failure_class"] = failure_class
        state["last_failure"] = last_line[-500:]
        if failure_class == "transient" and not self.stop:
            exponent = min(max(int(state["attempts"]) - 1, 0), 16)
            delay = min(self.backoff_max, self.backoff_base * (2**exponent))
            state["status"] = "backoff"
            state["next_start"] = time.monotonic() + delay
            state["retry_delay_seconds"] = delay
        else:
            state["status"] = "failed"

    def serializable_state(self) -> dict[str, Any]:
        jobs: dict[str, Any] = {}
        for job in JOBS:
            runtime = self.runtime[job.label]
            record = {
                key: value
                for key, value in runtime.items()
                if key not in {"process", "log_handle", "next_start", "log_offset"}
            }
            if runtime.get("status") == "backoff":
                record["retry_in_seconds"] = max(0.0, round(runtime["next_start"] - time.monotonic(), 1))
            record.update(
                output=str(job.file.path),
                expected_size=job.file.size,
                expected_sha256=job.file.sha256,
                workers=job.workers,
                log=str(job.log_path),
                progress=manifest_progress(job),
            )
            jobs[job.label] = record
        statuses = {record["status"] for record in jobs.values()}
        overall = "failed" if "failed" in statuses else "complete" if statuses == {"complete"} else "running"
        return {
            "schema_version": 1,
            "supervisor_pid": os.getpid(),
            "started_at": self.started_at,
            "updated_at": prepare_flash_next.utc_now(),
            "status": overall,
            "repo": prepare_flash_next.REPO,
            "revision": prepare_flash_next.REVISION,
            "jobs": jobs,
        }

    def write_state(self) -> None:
        atomic_write_json(STATE_PATH, self.serializable_state())

    def terminate_children(self) -> None:
        for job in JOBS:
            process = self.runtime[job.label].get("process")
            if process is not None and process.poll() is None:
                process.send_signal(signal.SIGINT)
        deadline = time.monotonic() + 70.0
        for job in JOBS:
            state = self.runtime[job.label]
            process = state.get("process")
            if process is None:
                continue
            remaining = max(0.0, deadline - time.monotonic())
            try:
                process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                process.terminate()
                process.wait(timeout=10.0)
            log_handle = state.get("log_handle")
            if log_handle is not None:
                log_handle.close()
            state.update(process=None, log_handle=None, status="stopped")
            state.pop("child_pid", None)

    def run(self) -> int:
        signal.signal(signal.SIGINT, self.request_stop)
        signal.signal(signal.SIGTERM, self.request_stop)
        while not self.stop:
            now = time.monotonic()
            for job in JOBS:
                state = self.runtime[job.label]
                if state["status"] == "running":
                    self.observe_exit(job)
                if state["status"] in {"pending", "backoff"} and now >= state["next_start"]:
                    self.spawn(job)
            self.write_state()
            statuses = {state["status"] for state in self.runtime.values()}
            if statuses <= {"complete", "failed"}:
                return 1 if "failed" in statuses else 0
            time.sleep(self.poll_seconds)
        self.terminate_children()
        self.write_state()
        return 130


def positive_seconds(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poll-seconds", type=positive_seconds, default=5.0)
    parser.add_argument("--backoff-base", type=positive_seconds, default=5.0)
    parser.add_argument("--backoff-max", type=positive_seconds, default=120.0)
    args = parser.parse_args()
    if args.backoff_base > args.backoff_max:
        parser.error("--backoff-base cannot exceed --backoff-max")

    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOCK_PATH.open("a+b") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(f"another Flash-Next download supervisor holds {LOCK_PATH}", file=sys.stderr)
            return 2
        return Supervisor(args.poll_seconds, args.backoff_base, args.backoff_max).run()


if __name__ == "__main__":
    raise SystemExit(main())
