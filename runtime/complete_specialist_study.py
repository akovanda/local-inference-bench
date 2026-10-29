#!/usr/bin/env python3
"""Safely continue the pinned specialist study after its first wave exits."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import signal
import socket
import stat
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

try:
    from . import gpu_window
except ImportError:  # pragma: no cover - used when invoked as a script
    import gpu_window


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
DEFAULT_COMMANDS_PLAN = RESULTS / "specialists-final12k16k-remaining-commands.json"
CANONICAL_FIRST_WAVE = (
    "lfm25-qad-q4_0",
    "kat-v25-iq3_xxs",
    "gemma4-style12-q5_k_m",
)
EXPECTED_FOLLOWUPS = (
    ("bonsai2-pq2_0", "llama", 18081),
    ("bonsai2-ptq1_0", "llama", 18081),
    ("flash-next-coder-iq1_m", "strata", 18080),
)
COMPLETE_STATUSES = ("complete", "completed_with_evaluation_failures")
RUN_PREFIX_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
MAX_JSON_BYTES = 32 * 1024 * 1024
STATIC_PATH_OPTIONS = {
    "--config",
    "--plan",
    "--role-plan",
    "--sampler-plan",
    "--standard-plan",
}


class ContinuationError(RuntimeError):
    pass


class Cancelled(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return sha256_bytes(payload)


def _read_regular(path: Path, description: str) -> bytes:
    try:
        details = path.lstat()
        if path.is_symlink() or not stat.S_ISREG(details.st_mode):
            raise ContinuationError(f"{description} must be a regular non-symlink file")
        if details.st_size > MAX_JSON_BYTES:
            raise ContinuationError(f"{description} exceeds the {MAX_JSON_BYTES}-byte limit")
        return path.read_bytes()
    except FileNotFoundError as exc:
        raise ContinuationError(f"{description} is missing: {path}") from exc
    except OSError as exc:
        raise ContinuationError(f"{description} is unreadable: {path}") from exc


def read_json(path: Path, description: str) -> tuple[dict[str, Any], str]:
    payload = _read_regular(path, description)
    try:
        value = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ContinuationError(f"{description} contains malformed JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ContinuationError(f"{description} must be a JSON object")
    return value, sha256_bytes(payload)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class Audit:
    """One-owner audit whose snapshots are atomically replaced and fsynced."""

    def __init__(self, path: Path, document: dict[str, Any]) -> None:
        self.path = path
        self.document = document
        self.audit_id = str(document["audit_id"])
        path.parent.mkdir(parents=True, exist_ok=True)
        self._write(exclusive=True)

    def _payload(self) -> bytes:
        return (json.dumps(self.document, indent=2, sort_keys=True) + "\n").encode(
            "utf-8"
        )

    def _write(self, *, exclusive: bool) -> None:
        payload = self._payload()
        if exclusive:
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            try:
                descriptor = os.open(self.path, flags, 0o600)
            except FileExistsError as exc:
                raise ContinuationError(
                    f"continuation audit already exists; refusing duplicate launch: {self.path}"
                ) from exc
            try:
                view = memoryview(payload)
                offset = 0
                while offset < len(view):
                    written = os.write(descriptor, view[offset:])
                    if written <= 0:
                        raise OSError("short audit write")
                    offset += written
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            _fsync_directory(self.path.parent)
            return

        existing, _ = read_json(self.path, "continuation audit")
        if existing.get("audit_id") != self.audit_id:
            raise ContinuationError("continuation audit ownership changed during execution")
        temporary = self.path.with_name(
            self.path.name + f".tmp.{os.getpid()}.{uuid.uuid4().hex}"
        )
        descriptor = -1
        try:
            descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            view = memoryview(payload)
            offset = 0
            while offset < len(view):
                written = os.write(descriptor, view[offset:])
                if written <= 0:
                    raise OSError("short audit write")
                offset += written
            os.fsync(descriptor)
            os.close(descriptor)
            descriptor = -1
            os.replace(temporary, self.path)
            _fsync_directory(self.path.parent)
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            temporary.unlink(missing_ok=True)

    def event(self, name: str, **fields: Any) -> None:
        row = {"at": utc_now(), "event": name, **fields}
        self.document["events"].append(row)
        self.document["progress"] = name
        self.document["updated_at"] = row["at"]
        self._write(exclusive=False)
        print(json.dumps(row, separators=(",", ":")), flush=True)

    def finish(self, status: str, exit_code: int, **fields: Any) -> None:
        finished_at = utc_now()
        self.document.update(
            status=status,
            exit_code=exit_code,
            progress="finished",
            finished_at=finished_at,
            updated_at=finished_at,
            **fields,
        )
        self.document["events"].append(
            {"at": finished_at, "event": "finished", "status": status, "exit_code": exit_code}
        )
        self._write(exclusive=False)


def _option(argv: Sequence[str], name: str) -> str:
    positions = [index for index, value in enumerate(argv) if value == name]
    if len(positions) != 1 or positions[0] + 1 >= len(argv):
        raise ContinuationError(f"saved command must contain exactly one {name}")
    return argv[positions[0] + 1]


def _command_entry(value: Any, index: int, kind: str) -> dict[str, Any]:
    if not isinstance(value, dict) or value.get("kind") != kind:
        raise ContinuationError(f"commands[{index}] must be the {kind} entry")
    argv = value.get("argv")
    if (
        not isinstance(argv, list)
        or len(argv) < 2
        or any(not isinstance(item, str) or not item for item in argv)
    ):
        raise ContinuationError(f"commands[{index}].argv must be a non-empty string list")
    return value


def _static_hashes(argv: Sequence[str]) -> dict[str, str]:
    candidates: list[Path] = []
    script = Path(argv[1])
    if script.is_absolute():
        candidates.append(script)
    for index, value in enumerate(argv[:-1]):
        if value in STATIC_PATH_OPTIONS:
            candidates.append(Path(argv[index + 1]))
    hashes: dict[str, str] = {}
    for path in candidates:
        payload = _read_regular(path, f"saved command input {path}")
        hashes[str(path)] = sha256_bytes(payload)
    return hashes


@dataclass(frozen=True)
class LoadedPlan:
    path: Path
    sha256: str
    first_wave_receipt: Path
    followup_receipt: Path
    matching_receipt: Path
    expected_first_wave: tuple[str, ...]
    commands: tuple[dict[str, Any], dict[str, Any]]


def load_commands_plan(
    path: Path, expected_sha256: str, run_prefix: str, results: Path
) -> LoadedPlan:
    document, actual_sha = read_json(path, "remaining commands plan")
    if actual_sha != expected_sha256:
        raise ContinuationError(
            f"remaining commands plan SHA-256 mismatch: expected {expected_sha256}, got {actual_sha}"
        )
    if document.get("schema_version") != 1:
        raise ContinuationError("remaining commands plan schema_version must be 1")
    commands = document.get("commands")
    if not isinstance(commands, list) or len(commands) != 3:
        raise ContinuationError("remaining commands plan must contain exactly three commands")
    first_kind = commands[0].get("kind") if isinstance(commands[0], dict) else None
    if first_kind not in ("selected_kat_style", "selected_first_wave"):
        raise ContinuationError(
            "commands[0] must be the selected_kat_style or selected_first_wave entry"
        )
    first = _command_entry(commands[0], 0, str(first_kind))
    followup = _command_entry(commands[1], 1, "bonsai_and_flash")
    matching = _command_entry(commands[2], 2, "matching_20b_reference_LAST")
    first_argv = first["argv"]
    followup_argv = followup["argv"]
    matching_argv = matching["argv"]
    if _option(first_argv, "--run-prefix") != run_prefix:
        raise ContinuationError("commands[0] run prefix differs from the requested prefix")
    if _option(followup_argv, "--run-prefix") != run_prefix:
        raise ContinuationError("commands[1] run prefix differs from the requested prefix")
    expected_first_wave = tuple(_option(first_argv, "--models").split(","))
    if (
        not expected_first_wave
        or any(not item for item in expected_first_wave)
        or len(set(expected_first_wave)) != len(expected_first_wave)
        or expected_first_wave
        != tuple(key for key in CANONICAL_FIRST_WAVE if key in expected_first_wave)
    ):
        raise ContinuationError(
            "commands[0] --models must be a nonempty unique subset in canonical order"
        )
    sampler_path = document.get("sampler_plan")
    sampler_sha = document.get("sampler_plan_sha256")
    if not isinstance(sampler_path, str) or not SHA256_RE.fullmatch(str(sampler_sha)):
        raise ContinuationError("remaining commands plan has no pinned sampler plan")
    if _option(followup_argv, "--sampler-plan") != sampler_path:
        raise ContinuationError("commands[1] sampler plan differs from the plan pin")
    if sha256_bytes(_read_regular(Path(sampler_path), "pinned sampler plan")) != sampler_sha:
        raise ContinuationError("pinned sampler plan SHA-256 does not match")
    first_wave_receipt = Path(_option(followup_argv, "--first-wave-receipt"))
    tag = _option(matching_argv, "--tag")
    return LoadedPlan(
        path=path,
        sha256=actual_sha,
        first_wave_receipt=first_wave_receipt,
        followup_receipt=results / f"{run_prefix}-followups-cohort.json",
        matching_receipt=results / f"{tag}-study-trial.json",
        expected_first_wave=expected_first_wave,
        commands=(followup, matching),
    )


def process_start_ticks(pid: int) -> int | None:
    try:
        payload = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise ContinuationError(f"could not inspect expected parent PID {pid}") from exc
    closing = payload.rfind(")")
    if closing < 0:
        raise ContinuationError(f"malformed /proc stat for expected parent PID {pid}")
    fields = payload[closing + 1 :].split()
    try:
        return int(fields[19])
    except (IndexError, ValueError) as exc:
        raise ContinuationError(f"malformed /proc stat for expected parent PID {pid}") from exc


def port_clear(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.3):
            return False
    except ConnectionRefusedError:
        return True
    except OSError:
        return False


def launch_boundary() -> dict[str, Any]:
    try:
        state = gpu_window.original_state()
    except Exception as exc:
        raise ContinuationError(f"original Docker state probe failed: {exc}") from exc
    evidence = {
        "checked_at": utc_now(),
        "original_container": gpu_window.ORIGINAL_CONTAINER,
        "original_running": state.get("Running") is True,
        "original_paused": state.get("Paused") is True,
        "ports_clear": {str(port): port_clear(port) for port in (18080, 18081)},
    }
    if (
        evidence["original_running"] is not True
        or evidence["original_paused"] is not False
        or not all(evidence["ports_clear"].values())
    ):
        raise ContinuationError("original Docker is not restored or owned ports are not clear")
    return evidence


def validate_study_receipt(
    path: Path,
    expected_sha256: str | None,
    *,
    expected_profile: str,
    expected_backend: str,
    expected_port: int,
) -> dict[str, Any]:
    receipt, actual_sha = read_json(path, f"study receipt for {expected_profile}")
    if expected_sha256 is not None and actual_sha != expected_sha256:
        raise ContinuationError(f"study receipt SHA-256 differs for {expected_profile}")
    status_value = receipt.get("status")
    cleanup = receipt.get("cleanup")
    restoration = receipt.get("restoration")
    controls = receipt.get("controls")
    session_cleanup = cleanup.get("model_session_cleanup") if isinstance(cleanup, dict) else None
    expected_outer_returncode = 1 if status_value == "completed_with_evaluation_failures" else 0
    safe = (
        receipt.get("schema_version") == 1
        and status_value in COMPLETE_STATUSES
        and receipt.get("runtime_backend") == expected_backend
        and isinstance(controls, dict)
        and controls.get("profile") == expected_profile
        and isinstance(cleanup, dict)
        and cleanup.get("outer_exited") is True
        and cleanup.get("outer_returncode") == expected_outer_returncode
        and cleanup.get("orchestrator_sigkill_sent") is False
        and isinstance(session_cleanup, dict)
        and session_cleanup.get("stopped") is True
        and session_cleanup.get("kill_sent") is False
        and isinstance(restoration, dict)
        and restoration.get("original_running") is True
        and restoration.get("original_paused") is False
        and restoration.get("owned_loopback_port_clear") is True
        and restoration.get("owned_loopback_port") == expected_port
        and receipt.get("problem") is None
    )
    if not safe:
        raise ContinuationError(
            f"study receipt for {expected_profile} does not prove evaluation completion and safe restoration"
        )
    return {
        "path": str(path),
        "sha256": actual_sha,
        "status": status_value,
        "outer_returncode": cleanup["outer_returncode"],
        "cleanup_verified": True,
    }


def _selected_study(model: Mapping[str, Any]) -> tuple[Path, str]:
    selected = model.get("selected_profile")
    if selected == "strict_all_layers_fit_off":
        path = model.get("strict_receipt")
        digest = model.get("strict_receipt_sha256")
    elif selected == "auto_fallback_after_strict_cuda_oom":
        path = model.get("fallback_receipt")
        digest = model.get("fallback_receipt_sha256")
    else:
        raise ContinuationError("first-wave model has no recognized selected study")
    if not isinstance(path, str) or not SHA256_RE.fullmatch(str(digest)):
        raise ContinuationError("first-wave selected study identity is incomplete")
    return Path(path), str(digest)


def validate_first_wave(path: Path, expected_models: Sequence[str]) -> dict[str, Any]:
    receipt, digest = read_json(path, "first-wave cohort receipt")
    if receipt.get("schema_version") != 1 or receipt.get("status") not in COMPLETE_STATUSES:
        raise ContinuationError("first-wave cohort is not evaluation-complete")
    if receipt.get("problem") is not None:
        raise ContinuationError("first-wave cohort reports an infrastructure problem")
    controls = receipt.get("controls")
    models = receipt.get("models")
    if (
        not isinstance(controls, dict)
        or controls.get("selected_models") != list(expected_models)
        or not isinstance(models, list)
        or [row.get("key") if isinstance(row, dict) else None for row in models]
        != list(expected_models)
    ):
        raise ContinuationError("first-wave receipt does not exactly match commands[0] --models")
    studies = []
    for key, model in zip(expected_models, models, strict=True):
        assert isinstance(model, dict)
        study_path, study_sha = _selected_study(model)
        study = validate_study_receipt(
            study_path,
            study_sha,
            expected_profile=key,
            expected_backend="llama",
            expected_port=18081,
        )
        if model.get("status") != study["status"]:
            raise ContinuationError(f"first-wave cohort status differs from {key} study")
        studies.append(study)
    expected_status = (
        "completed_with_evaluation_failures"
        if any(row["status"] == "completed_with_evaluation_failures" for row in studies)
        else "complete"
    )
    if receipt["status"] != expected_status:
        raise ContinuationError("first-wave cohort aggregate status disagrees with its studies")
    return {
        "path": str(path),
        "sha256": digest,
        "status": receipt["status"],
        "selected_models": list(expected_models),
        "studies": studies,
    }


def validate_followups(path: Path) -> dict[str, Any]:
    receipt, digest = read_json(path, "follow-up cohort receipt")
    if receipt.get("schema_version") != 1 or receipt.get("status") not in COMPLETE_STATUSES:
        raise ContinuationError("follow-up cohort is not evaluation-complete")
    if receipt.get("problem") is not None:
        raise ContinuationError("follow-up cohort reports an infrastructure problem")
    models = receipt.get("models")
    if (
        not isinstance(models, list)
        or [row.get("key") if isinstance(row, dict) else None for row in models]
        != [row[0] for row in EXPECTED_FOLLOWUPS]
    ):
        raise ContinuationError("follow-up receipt has an incomplete or reordered model set")
    studies = []
    for (key, backend, port), model in zip(EXPECTED_FOLLOWUPS, models, strict=True):
        assert isinstance(model, dict)
        identity = model.get("study")
        if (
            not isinstance(identity, dict)
            or not isinstance(identity.get("path"), str)
            or not SHA256_RE.fullmatch(str(identity.get("sha256")))
        ):
            raise ContinuationError(f"follow-up receipt has no study identity for {key}")
        study = validate_study_receipt(
            Path(identity["path"]),
            str(identity["sha256"]),
            expected_profile=key,
            expected_backend=backend,
            expected_port=port,
        )
        if model.get("status") != study["status"]:
            raise ContinuationError(f"follow-up cohort status differs from {key} study")
        studies.append(study)
    expected_status = (
        "completed_with_evaluation_failures"
        if any(row["status"] == "completed_with_evaluation_failures" for row in studies)
        else "complete"
    )
    if receipt["status"] != expected_status:
        raise ContinuationError("follow-up cohort aggregate status disagrees with its studies")
    return {
        "path": str(path),
        "sha256": digest,
        "status": receipt["status"],
        "models": [row[0] for row in EXPECTED_FOLLOWUPS],
        "studies": studies,
    }


@dataclass(frozen=True)
class Dependencies:
    popen: Callable[..., Any] = subprocess.Popen
    pid_probe: Callable[[int], int | None] = process_start_ticks
    boundary_probe: Callable[[], dict[str, Any]] = launch_boundary
    monotonic: Callable[[], float] = time.monotonic
    sleep: Callable[[float], None] = time.sleep
    cancelled: Callable[[], bool] = lambda: False


def _wait_boundary(
    deps: Dependencies, timeout: float, poll_seconds: float
) -> dict[str, Any]:
    deadline = deps.monotonic() + timeout
    last_error = "boundary was not probed"
    while deps.monotonic() < deadline:
        try:
            return deps.boundary_probe()
        except ContinuationError as exc:
            last_error = str(exc)
        if deps.cancelled():
            raise Cancelled("continuation cancelled while waiting for a safe launch boundary")
        deps.sleep(poll_seconds)
    raise ContinuationError(f"safe launch boundary did not appear: {last_error}")


def wait_initial_gate(
    plan: LoadedPlan,
    args: argparse.Namespace,
    audit: Audit,
    deps: Dependencies,
) -> tuple[dict[str, Any], dict[str, Any]]:
    deadline = deps.monotonic() + args.initial_wait_seconds
    first_wave: dict[str, Any] | None = None
    receipt_recorded = False
    while deps.monotonic() < deadline:
        if deps.cancelled():
            raise Cancelled("continuation cancelled during the initial wait")
        if first_wave is None and os.path.lexists(plan.first_wave_receipt):
            first_wave = validate_first_wave(
                plan.first_wave_receipt, plan.expected_first_wave
            )
        if first_wave is not None and not receipt_recorded:
            audit.document["receipts"]["first_wave"] = first_wave
            audit.event("first_wave_evaluation_complete", status=first_wave["status"])
            receipt_recorded = True
        observed_ticks = deps.pid_probe(args.expected_parent_pid)
        parent_live = observed_ticks == args.expected_parent_start_ticks
        if first_wave is not None and not parent_live:
            boundary = _wait_boundary(
                deps,
                max(0.001, deadline - deps.monotonic()),
                args.poll_seconds,
            )
            audit.document["owned_parent"].update(
                observed_start_ticks=observed_ticks,
                exited=True,
                exited_at=utc_now(),
            )
            audit.document["boundaries"].append(boundary)
            audit.event("initial_gate_passed", boundary=boundary)
            return first_wave, boundary
        deps.sleep(args.poll_seconds)
    missing = []
    if first_wave is None:
        missing.append("evaluation-complete first-wave receipt")
    if deps.pid_probe(args.expected_parent_pid) == args.expected_parent_start_ticks:
        missing.append("exact owned parent exit")
    raise ContinuationError("initial wait expired before " + " and ".join(missing or ["safe boundary"]))


def _verify_static_hashes(expected: Mapping[str, str]) -> None:
    for raw_path, expected_sha in expected.items():
        actual = sha256_bytes(_read_regular(Path(raw_path), f"saved command input {raw_path}"))
        if actual != expected_sha:
            raise ContinuationError(f"saved command input changed after audit creation: {raw_path}")


def run_stage(
    stage_index: int,
    command: Mapping[str, Any],
    args: argparse.Namespace,
    audit: Audit,
    deps: Dependencies,
) -> int:
    argv = list(command["argv"])
    static_hashes = audit.document["commands"][stage_index]["static_input_sha256"]
    _verify_static_hashes(static_hashes)
    stage = audit.document["stages"][stage_index]
    stage.update(status="launching", launch_started_at=utc_now())
    audit.event("stage_launching", stage=stage_index + 1, kind=command["kind"])
    try:
        process = deps.popen(
            argv,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
            close_fds=True,
            cwd=str(ROOT),
        )
    except OSError as exc:
        raise ContinuationError(f"could not launch {command['kind']}") from exc
    stage.update(
        status="running",
        pid=process.pid,
        pid_start_ticks=deps.pid_probe(process.pid),
        started_at=utc_now(),
    )
    audit.event("stage_started", stage=stage_index + 1, kind=command["kind"], pid=process.pid)
    deadline = deps.monotonic() + args.stage_timeout_seconds
    interrupt_sent = False
    interrupt_reason: str | None = None
    interrupt_deadline: float | None = None
    while True:
        returncode = process.poll()
        if returncode is not None:
            stage.update(status="exited", returncode=returncode, finished_at=utc_now())
            audit.event(
                "stage_exited",
                stage=stage_index + 1,
                kind=command["kind"],
                pid=process.pid,
                returncode=returncode,
            )
            if interrupt_sent:
                if interrupt_reason == "cancelled":
                    raise Cancelled(f"cancelled while {command['kind']} was running")
                raise ContinuationError(f"{command['kind']} exceeded its stage timeout")
            return int(returncode)
        now = deps.monotonic()
        if (deps.cancelled() or now >= deadline) and not interrupt_sent:
            try:
                process.send_signal(signal.SIGINT)
            except ProcessLookupError:
                pass
            interrupt_sent = True
            interrupt_reason = "cancelled" if deps.cancelled() else "timeout"
            interrupt_deadline = now + args.cancel_grace_seconds
            stage["sigint_sent_at"] = utc_now()
            audit.event(
                "stage_sigint_sent",
                stage=stage_index + 1,
                kind=command["kind"],
                reason=interrupt_reason,
            )
        if interrupt_sent and interrupt_deadline is not None and now >= interrupt_deadline:
            raise ContinuationError(
                f"{command['kind']} did not finish cleanup after SIGINT; no SIGKILL was sent"
            )
        deps.sleep(args.poll_seconds)


def _expected_returncode(status_value: str) -> int:
    return 1 if status_value == "completed_with_evaluation_failures" else 0


def execute(args: argparse.Namespace, *, deps: Dependencies = Dependencies()) -> tuple[dict[str, Any], int]:
    plan = load_commands_plan(
        args.commands_plan, args.commands_sha256, args.run_prefix, args.results_dir
    )
    audit_path = args.results_dir / f"{args.run_prefix}-complete-specialist-study-audit.json"
    command_rows = []
    for index, command in enumerate(plan.commands, start=1):
        argv = command["argv"]
        command_rows.append(
            {
                "plan_index": index,
                "kind": command["kind"],
                "argv": argv,
                "argv_sha256": canonical_sha256(argv),
                "entry_sha256": canonical_sha256(command),
                "static_input_sha256": _static_hashes(argv),
            }
        )
    created_at = utc_now()
    document: dict[str, Any] = {
        "schema_version": 1,
        "audit_id": str(uuid.uuid4()),
        "status": "running",
        "progress": "created",
        "created_at": created_at,
        "updated_at": created_at,
        "run_prefix": args.run_prefix,
        "commands_plan": {"path": str(plan.path), "sha256": plan.sha256},
        "commands": command_rows,
        "owned_parent": {
            "pid": args.expected_parent_pid,
            "expected_start_ticks": args.expected_parent_start_ticks,
            "observed_start_ticks": deps.pid_probe(args.expected_parent_pid),
            "exited": False,
        },
        "receipts": {},
        "boundaries": [],
        "stages": [
            {"plan_index": 1, "kind": plan.commands[0]["kind"], "status": "pending"},
            {"plan_index": 2, "kind": plan.commands[1]["kind"], "status": "pending"},
        ],
        "events": [],
    }
    audit = Audit(audit_path, document)
    semantic_failure = False
    try:
        audit.event(
            "waiting_for_first_wave_and_parent_exit",
            receipt=str(plan.first_wave_receipt),
            pid=args.expected_parent_pid,
            start_ticks=args.expected_parent_start_ticks,
        )
        first_wave, _ = wait_initial_gate(plan, args, audit, deps)
        semantic_failure = first_wave["status"] == "completed_with_evaluation_failures"

        followup_code = run_stage(0, plan.commands[0], args, audit, deps)
        followups = validate_followups(plan.followup_receipt)
        if followup_code != _expected_returncode(followups["status"]):
            raise ContinuationError("follow-up command return code disagrees with its cohort receipt")
        audit.document["receipts"]["followups"] = followups
        audit.document["stages"][0]["receipt_status"] = followups["status"]
        semantic_failure |= followups["status"] == "completed_with_evaluation_failures"
        audit.event("followups_validated", status=followups["status"])

        boundary = _wait_boundary(deps, args.boundary_wait_seconds, args.poll_seconds)
        audit.document["boundaries"].append(boundary)
        audit.event("matching_launch_gate_passed", boundary=boundary)

        matching_code = run_stage(1, plan.commands[1], args, audit, deps)
        expected_profile = _option(plan.commands[1]["argv"], "--profile")
        matching = validate_study_receipt(
            plan.matching_receipt,
            None,
            expected_profile=expected_profile,
            expected_backend="llama",
            expected_port=18081,
        )
        if matching_code != _expected_returncode(matching["status"]):
            raise ContinuationError("matching reference return code disagrees with its study receipt")
        audit.document["receipts"]["matching_reference"] = matching
        audit.document["stages"][1]["receipt_status"] = matching["status"]
        semantic_failure |= matching["status"] == "completed_with_evaluation_failures"
        audit.event("matching_reference_validated", status=matching["status"])

        boundary = _wait_boundary(deps, args.boundary_wait_seconds, args.poll_seconds)
        audit.document["boundaries"].append(boundary)
        audit.event("final_restoration_verified", boundary=boundary)
        status_value = "completed_with_evaluation_failures" if semantic_failure else "complete"
        code = 1 if semantic_failure else 0
        audit.finish(status_value, code)
        return audit.document, code
    except Cancelled as exc:
        problem = {"error_type": type(exc).__name__, "error": str(exc)}
        audit.finish("cancelled", 130, problem=problem)
        return audit.document, 130
    except (ContinuationError, OSError, subprocess.SubprocessError) as exc:
        problem = {"error_type": type(exc).__name__, "error": str(exc)}
        audit.finish("infrastructure_failed", 2, problem=problem)
        return audit.document, 2


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def bounded_seconds(maximum: float):
    def parse(value: str) -> float:
        parsed = float(value)
        if not math.isfinite(parsed) or not 0 < parsed <= maximum:
            raise argparse.ArgumentTypeError(f"must be finite and from >0 through {maximum:g}")
        return parsed

    return parse


def run_prefix(value: str) -> str:
    if RUN_PREFIX_RE.fullmatch(value) is None:
        raise argparse.ArgumentTypeError("must be a safe 1-64 character run prefix")
    return value


def sha256_value(value: str) -> str:
    lowered = value.lower()
    if SHA256_RE.fullmatch(lowered) is None:
        raise argparse.ArgumentTypeError("must be a 64-character hexadecimal SHA-256")
    return lowered


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commands-plan", type=Path, default=DEFAULT_COMMANDS_PLAN)
    parser.add_argument("--commands-sha256", type=sha256_value, required=True)
    parser.add_argument("--run-prefix", type=run_prefix, default="specialists-final12k16k")
    parser.add_argument("--results-dir", type=Path, default=RESULTS)
    parser.add_argument("--expected-parent-pid", type=positive_int, required=True)
    parser.add_argument("--expected-parent-start-ticks", type=positive_int, required=True)
    parser.add_argument(
        "--initial-wait-seconds", type=bounded_seconds(604_800.0), default=46_800.0
    )
    parser.add_argument(
        "--stage-timeout-seconds", type=bounded_seconds(604_800.0), default=129_600.0
    )
    parser.add_argument(
        "--boundary-wait-seconds", type=bounded_seconds(1_800.0), default=300.0
    )
    parser.add_argument(
        "--cancel-grace-seconds", type=bounded_seconds(1_800.0), default=900.0
    )
    parser.add_argument("--poll-seconds", type=bounded_seconds(60.0), default=1.0)
    return parser


class SignalFlag:
    def __init__(self) -> None:
        self.received: int | None = None

    def __call__(self) -> bool:
        return self.received is not None

    def handler(self, signum: int, _frame: Any) -> None:
        self.received = signum


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.commands_plan = args.commands_plan.expanduser().resolve()
    args.results_dir = args.results_dir.expanduser().resolve()
    if args.expected_parent_pid == os.getpid():
        print("continuation runner: expected parent PID cannot be this runner", file=sys.stderr)
        return 2
    flag = SignalFlag()
    previous = {
        signum: signal.signal(signum, flag.handler) for signum in (signal.SIGINT, signal.SIGTERM)
    }
    try:
        _, code = execute(args, deps=Dependencies(cancelled=flag))
        return code
    except ContinuationError as exc:
        print(f"continuation runner: {exc}", file=sys.stderr)
        return 2
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


if __name__ == "__main__":
    raise SystemExit(main())
