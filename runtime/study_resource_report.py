#!/usr/bin/env python3
"""Build an offline, interval-scoped resource report for owned model servers."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import stat
import statistics
from typing import Any, Iterable


SCHEMA_VERSION = 1
TERMINAL_EVENTS = {"session_complete", "session_failed", "session_interrupted"}
GPU_METRICS = {
    "utilization_percent": ("utilization.gpu", "%"),
    "board_power_watts": ("power.draw", "W"),
    "sm_clock_mhz": ("clocks.sm", "MHz"),
    "memory_clock_mhz": ("clocks.mem", "MHz"),
    "temperature_celsius": ("temperature.gpu", "C"),
}
BRAKE_FIELD = "clocks_event_reasons.hw_power_brake_slowdown"
OWNERSHIP_KIND = "strata_owned_gpu_process_observation"
OWNERSHIP_NAME = re.compile(r"^(?P<tag>.+)-gpu-ownership-v(?P<version>[1-9][0-9]*)\.json$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")


class ReportInputError(ValueError):
    """Raised when the report cannot safely interpret its inputs."""


@dataclass(frozen=True)
class SourceRow:
    source: str
    line: int
    value: dict[str, Any]
    raw_sha256: str
    canonical_sha256: str


@dataclass(frozen=True)
class TelemetrySample:
    at: datetime
    source: str
    line: int
    gpu: tuple[dict[str, Any], ...]
    processes: tuple[dict[str, Any], ...]


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _parse_time(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(timezone.utc)


def _read_jsonl(path: Path, kind: str) -> tuple[list[SourceRow], dict[str, Any]]:
    rows: list[SourceRow] = []
    issues: list[dict[str, Any]] = []
    snapshot = path.read_bytes()
    for line_number, raw_bytes in enumerate(snapshot.splitlines(), 1):
        if not raw_bytes.strip():
            continue
        raw = raw_bytes.decode("utf-8", errors="replace")
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            issues.append({"line": line_number, "kind": "malformed_json", "detail": exc.msg})
            continue
        if not isinstance(value, dict):
            issues.append({"line": line_number, "kind": "non_object_json"})
            continue
        canonical = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
        rows.append(SourceRow(
            str(path.resolve()), line_number, value,
            hashlib.sha256(raw_bytes).hexdigest(), hashlib.sha256(canonical).hexdigest(),
        ))
    return rows, {
        "path": str(path.resolve()),
        "kind": kind,
        "sha256": hashlib.sha256(snapshot).hexdigest(),
        "snapshot_byte_count": len(snapshot),
        "valid_record_count": len(rows),
        "invalid_record_count": len(issues),
        "issues": issues,
    }


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _integer(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        result = int(str(value).strip())
    except (TypeError, ValueError):
        return None
    return result if result > 0 else None


def _positive_decimal(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    text = str(value)
    if not text or not text.isdecimal():
        return None
    result = int(text)
    return result if result > 0 else None


def _strict_json(snapshot: bytes) -> Any:
    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"duplicate JSON key: {key}")
            value[key] = item
        return value

    return json.loads(snapshot.decode("utf-8"), object_pairs_hook=object_pairs)


def _pinned_json(
    binding: Any, label: str, issues: list[str],
) -> tuple[Path | None, dict[str, Any] | None]:
    if not isinstance(binding, dict) or set(binding) != {"path", "sha256", "size"}:
        issues.append(f"{label} binding must contain exactly path, sha256, and size")
        return None, None
    raw_path = binding.get("path")
    expected_sha = binding.get("sha256")
    expected_size = binding.get("size")
    if not isinstance(raw_path, str) or not raw_path or not Path(raw_path).is_absolute():
        issues.append(f"{label} path must be a nonempty absolute path")
        return None, None
    path = Path(raw_path)
    if not isinstance(expected_sha, str) or SHA256.fullmatch(expected_sha) is None:
        issues.append(f"{label} sha256 is invalid")
    if isinstance(expected_size, bool) or not isinstance(expected_size, int) or expected_size < 0:
        issues.append(f"{label} size is invalid")
    try:
        file_stat = path.lstat()
    except OSError as exc:
        issues.append(f"{label} file is unavailable: {exc.strerror or exc}")
        return path, None
    if path.is_symlink() or not stat.S_ISREG(file_stat.st_mode):
        issues.append(f"{label} file is not a regular nonsymlink file")
        return path, None
    try:
        snapshot = path.read_bytes()
    except OSError as exc:
        issues.append(f"{label} file could not be read: {exc.strerror or exc}")
        return path, None
    if isinstance(expected_size, int) and not isinstance(expected_size, bool) and len(snapshot) != expected_size:
        issues.append(f"{label} size does not match the pinned file")
    actual_sha = hashlib.sha256(snapshot).hexdigest()
    if isinstance(expected_sha, str) and actual_sha != expected_sha:
        issues.append(f"{label} sha256 does not match the pinned file")
    try:
        value = _strict_json(snapshot)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        issues.append(f"{label} is not strict JSON: {exc}")
        return path, None
    if not isinstance(value, dict):
        issues.append(f"{label} JSON is not an object")
        return path, None
    return path, value


def _validate_ownership_receipt(
    path: Path,
    lifecycle_rows: dict[str, list[SourceRow]],
    candidates_by_identity: dict[tuple[str, str], list[dict[str, Any]]],
    generated_at: datetime,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    display_path = str(path.absolute())
    match = OWNERSHIP_NAME.fullmatch(path.name)
    metadata: dict[str, Any] = {
        "path": display_path,
        "kind": "gpu_ownership",
        "sha256": None,
        "snapshot_byte_count": None,
        "accepted": False,
        "receipt_file_version": int(match.group("version")) if match else None,
        "identity": None,
        "issues": [],
    }
    issues: list[str] = metadata["issues"]
    if match is None:
        issues.append("filename does not match the declared GPU ownership receipt pattern")
        return metadata, None
    try:
        file_stat = path.lstat()
    except OSError as exc:
        issues.append(f"receipt is unavailable: {exc.strerror or exc}")
        return metadata, None
    if path.is_symlink() or not stat.S_ISREG(file_stat.st_mode):
        issues.append("receipt is not a regular nonsymlink file")
        return metadata, None
    try:
        snapshot = path.read_bytes()
    except OSError as exc:
        issues.append(f"receipt could not be read: {exc.strerror or exc}")
        return metadata, None
    metadata["sha256"] = hashlib.sha256(snapshot).hexdigest()
    metadata["snapshot_byte_count"] = len(snapshot)
    try:
        receipt = _strict_json(snapshot)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        issues.append(f"receipt is not strict JSON: {exc}")
        return metadata, None
    if not isinstance(receipt, dict):
        issues.append("receipt JSON is not an object")
        return metadata, None

    expected_keys = {
        "api_process", "direct_child_parent_relation_verified", "engine_environment",
        "expected_engine_exe", "gpu_observation", "gpu_process", "kernel_clock", "kind",
        "lifecycle_events", "lifecycle_source", "live_kernel_start_identities_verified",
        "model_session_run_id", "observed_at", "observed_during_live_matching_model_session",
        "private_config", "prose_bodies_read", "runtime_backend", "schema_version", "scope",
        "source_config", "tag",
    }
    if set(receipt) != expected_keys:
        issues.append("receipt fields do not exactly match schema version 1")
    tag = receipt.get("tag")
    run_id = receipt.get("model_session_run_id")
    if isinstance(tag, str) and tag and isinstance(run_id, str) and run_id:
        metadata["identity"] = {"tag": tag, "run_id": run_id}
    else:
        issues.append("receipt tag and model_session_run_id must be nonempty strings")
    if tag != match.group("tag"):
        issues.append("receipt tag does not match its filename")
    if type(receipt.get("schema_version")) is not int or receipt.get("schema_version") != 1:
        issues.append("receipt schema_version must be integer 1")
    if type(receipt.get("schema_version")) is int and int(match.group("version")) != receipt["schema_version"]:
        issues.append("receipt filename version does not match schema_version")
    if receipt.get("kind") != OWNERSHIP_KIND:
        issues.append(f"receipt kind must be {OWNERSHIP_KIND}")
    if receipt.get("runtime_backend") != "strata":
        issues.append("receipt runtime_backend must be strata")
    for field in (
        "direct_child_parent_relation_verified", "live_kernel_start_identities_verified",
        "observed_during_live_matching_model_session",
    ):
        if receipt.get(field) is not True:
            issues.append(f"{field} must be true")
    if receipt.get("prose_bodies_read") is not False:
        issues.append("prose_bodies_read must be false")
    if not isinstance(receipt.get("scope"), str) or not receipt.get("scope"):
        issues.append("scope must be a nonempty string")

    identity = (tag, run_id) if isinstance(tag, str) and isinstance(run_id, str) else None
    identity_candidates = candidates_by_identity.get(identity, []) if identity else []
    candidate = identity_candidates[0] if len(identity_candidates) == 1 else None
    if len(identity_candidates) != 1:
        issues.append(
            f"expected exactly one matching lifecycle identity; found {len(identity_candidates)}"
        )
    lifecycle_source = receipt.get("lifecycle_source")
    if not isinstance(lifecycle_source, str) or not lifecycle_source or not Path(lifecycle_source).is_absolute():
        issues.append("lifecycle_source must be a nonempty absolute path")
        lifecycle_source = None
    elif candidate is not None and lifecycle_source != candidate["lifecycle_source"]:
        issues.append("lifecycle_source does not match the lifecycle candidate")
    if lifecycle_source is not None:
        try:
            lifecycle_stat = Path(lifecycle_source).lstat()
        except OSError as exc:
            issues.append(f"lifecycle_source is unavailable: {exc.strerror or exc}")
        else:
            if Path(lifecycle_source).is_symlink() or not stat.S_ISREG(lifecycle_stat.st_mode):
                issues.append("lifecycle_source is not a regular nonsymlink file")

    bound_rows: dict[str, SourceRow] = {}
    event_bindings = receipt.get("lifecycle_events")
    if not isinstance(event_bindings, list) or len(event_bindings) != 2:
        issues.append("lifecycle_events must contain exactly two bindings")
    elif lifecycle_source is not None:
        rows_by_line = {row.line: row for row in lifecycle_rows.get(lifecycle_source, [])}
        for binding in event_bindings:
            if not isinstance(binding, dict) or set(binding) != {
                "event", "source_line", "source_raw_line_sha256", "source_row_canonical_sha256",
            }:
                issues.append("each lifecycle event binding must exactly match schema version 1")
                continue
            event_name = binding.get("event")
            line = binding.get("source_line")
            raw_sha = binding.get("source_raw_line_sha256")
            canonical_sha = binding.get("source_row_canonical_sha256")
            if event_name not in {"session_started", "server_started"}:
                issues.append("lifecycle event binding has an unexpected event")
                continue
            if event_name in bound_rows:
                issues.append(f"lifecycle event binding duplicates {event_name}")
                continue
            if isinstance(line, bool) or not isinstance(line, int) or line <= 0:
                issues.append(f"{event_name} source_line is invalid")
                continue
            row = rows_by_line.get(line)
            if row is None:
                issues.append(f"{event_name} bound lifecycle row is absent")
                continue
            if not isinstance(raw_sha, str) or SHA256.fullmatch(raw_sha) is None or raw_sha != row.raw_sha256:
                issues.append(f"{event_name} raw row hash does not match")
            if (not isinstance(canonical_sha, str) or SHA256.fullmatch(canonical_sha) is None
                    or canonical_sha != row.canonical_sha256):
                issues.append(f"{event_name} canonical row hash does not match")
            if row.value.get("event") != event_name:
                issues.append(f"{event_name} binding points to a different event")
            if row.value.get("tag") != tag or row.value.get("run_id") != run_id:
                issues.append(f"{event_name} binding has a different tag/run identity")
            bound_rows[event_name] = row
        if set(bound_rows) != {"session_started", "server_started"}:
            issues.append("both session_started and server_started rows must be bound")

    api = receipt.get("api_process")
    gpu_process = receipt.get("gpu_process")
    gpu_observation = receipt.get("gpu_observation")
    if not isinstance(api, dict) or set(api) != {"exe", "parent_pid", "pid", "start_ticks", "state"}:
        issues.append("api_process does not exactly match schema version 1")
        api = {}
    if not isinstance(gpu_process, dict) or set(gpu_process) != {
        "exe", "parent_pid", "pid", "start_ticks", "state",
    }:
        issues.append("gpu_process does not exactly match schema version 1")
        gpu_process = {}
    if not isinstance(gpu_observation, dict) or set(gpu_observation) != {"name", "pid", "used_mib"}:
        issues.append("gpu_observation does not exactly match schema version 1")
        gpu_observation = {}
    api_pid = _positive_decimal(api.get("pid"))
    api_parent_pid = _positive_decimal(api.get("parent_pid"))
    api_start_ticks = _positive_decimal(api.get("start_ticks"))
    gpu_pid = _positive_decimal(gpu_process.get("pid"))
    gpu_parent_pid = _positive_decimal(gpu_process.get("parent_pid"))
    gpu_start_ticks = _positive_decimal(gpu_process.get("start_ticks"))
    observed_gpu_pid = _positive_decimal(gpu_observation.get("pid"))
    for label, value in (
        ("api_process.pid", api_pid), ("api_process.parent_pid", api_parent_pid),
        ("api_process.start_ticks", api_start_ticks), ("gpu_process.pid", gpu_pid),
        ("gpu_process.parent_pid", gpu_parent_pid),
        ("gpu_process.start_ticks", gpu_start_ticks),
        ("gpu_observation.pid", observed_gpu_pid),
    ):
        if value is None:
            issues.append(f"{label} must be a nonempty positive decimal identity")
    if candidate is not None and api_pid != candidate["server_pid"]:
        issues.append("API PID does not match the lifecycle server_pid")
    server_row = bound_rows.get("server_started")
    if server_row is not None and api_pid != _positive_decimal(server_row.value.get("pid")):
        issues.append("API PID does not match the bound server_started row")
    if gpu_pid is not None and api_pid is not None and gpu_pid == api_pid:
        issues.append("GPU process PID must differ from the API PID")
    if gpu_parent_pid is not None and api_pid is not None and gpu_parent_pid != api_pid:
        issues.append("GPU process is not a direct child of the API PID")
    if observed_gpu_pid is not None and gpu_pid is not None and observed_gpu_pid != gpu_pid:
        issues.append("GPU observation PID does not match the GPU process PID")
    if (api_start_ticks is not None and gpu_start_ticks is not None
            and gpu_start_ticks <= api_start_ticks):
        issues.append("GPU process start identity does not follow the API process start identity")
    for label, process in (("api_process", api), ("gpu_process", gpu_process)):
        if not isinstance(process.get("exe"), str) or not process.get("exe"):
            issues.append(f"{label}.exe must be a nonempty string")
        if not isinstance(process.get("state"), str) or not process.get("state"):
            issues.append(f"{label}.state must be a nonempty string")
    used_mib = _number(gpu_observation.get("used_mib"))
    if used_mib is None or used_mib < 0:
        issues.append("gpu_observation.used_mib must be a nonnegative number")

    expected_exe = receipt.get("expected_engine_exe")
    if not isinstance(expected_exe, str) or not expected_exe or not Path(expected_exe).is_absolute():
        issues.append("expected_engine_exe must be a nonempty absolute path")
    else:
        if gpu_process.get("exe") != expected_exe:
            issues.append("GPU process executable does not match expected_engine_exe")
        if gpu_observation.get("name") != expected_exe:
            issues.append("GPU observation name does not match expected_engine_exe")

    environment = receipt.get("engine_environment")
    if (not isinstance(environment, dict) or not environment
            or not all(isinstance(key, str) and key and isinstance(value, str)
                       for key, value in environment.items())):
        issues.append("engine_environment must be a nonempty string mapping")
        environment = None
    session_row = bound_rows.get("session_started")
    if session_row is not None:
        if session_row.value.get("runtime_backend") != "strata":
            issues.append("bound session_started runtime_backend is not strata")
        if session_row.value.get("engine") != expected_exe:
            issues.append("bound session_started engine does not match expected_engine_exe")
        if environment is not None and session_row.value.get("engine_environment") != environment:
            issues.append("bound session_started engine_environment does not match")

    private_path, private_config = _pinned_json(receipt.get("private_config"), "private_config", issues)
    source_path, source_config = _pinned_json(receipt.get("source_config"), "source_config", issues)
    if session_row is not None:
        if private_path is not None and session_row.value.get("private_config") != str(private_path):
            issues.append("private_config path does not match the bound session_started row")
        if source_path is not None and session_row.value.get("source_config") != str(source_path):
            issues.append("source_config path does not match the bound session_started row")
    if server_row is not None and private_path is not None:
        argv = server_row.value.get("argv")
        config_bound = isinstance(argv, list) and any(
            argv[index] == "--config" and argv[index + 1] == str(private_path)
            for index in range(len(argv) - 1)
        )
        if not config_bound:
            issues.append("bound server_started argv does not select private_config")
    for label, config in (("private_config", private_config), ("source_config", source_config)):
        if config is not None and config.get("exe") != expected_exe:
            issues.append(f"{label} executable does not match expected_engine_exe")
        reference_environment = config.get("reference_environment") if config is not None else None
        if environment is not None and (
            not isinstance(reference_environment, dict)
            or any(reference_environment.get(key) != value for key, value in environment.items())
        ):
            issues.append(f"{label} does not bind the observed engine_environment")

    kernel = receipt.get("kernel_clock")
    if not isinstance(kernel, dict) or set(kernel) != {
        "boot_id", "boot_time_resolution_seconds", "boot_time_unix_seconds",
        "clock_ticks_per_second", "gpu_process_start_unix_seconds",
    }:
        issues.append("kernel_clock does not exactly match schema version 1")
        kernel = {}
    boot_id = kernel.get("boot_id")
    if not isinstance(boot_id, str) or not boot_id:
        issues.append("kernel_clock.boot_id must be a nonempty string")
    boot_time = _number(kernel.get("boot_time_unix_seconds"))
    resolution = _number(kernel.get("boot_time_resolution_seconds"))
    ticks_per_second = _positive_decimal(kernel.get("clock_ticks_per_second"))
    supplied_gpu_birth = _number(kernel.get("gpu_process_start_unix_seconds"))
    if boot_time is None or boot_time <= 0:
        issues.append("kernel boot time must be positive")
    if resolution is None or resolution <= 0:
        issues.append("kernel boot-time resolution must be positive")
    if ticks_per_second is None:
        issues.append("kernel clock tick frequency must be a positive decimal")
    gpu_birth: datetime | None = None
    api_birth: datetime | None = None
    if (boot_time is not None and ticks_per_second is not None
            and gpu_start_ticks is not None and api_start_ticks is not None):
        derived_gpu_birth = boot_time + gpu_start_ticks / ticks_per_second
        derived_api_birth = boot_time + api_start_ticks / ticks_per_second
        if supplied_gpu_birth is None or not math.isclose(
            supplied_gpu_birth, derived_gpu_birth, rel_tol=0, abs_tol=1e-6,
        ):
            issues.append("supplied GPU process birth time does not match kernel identities")
        try:
            gpu_birth = datetime.fromtimestamp(derived_gpu_birth, timezone.utc)
            api_birth = datetime.fromtimestamp(derived_api_birth, timezone.utc)
        except (OverflowError, OSError, ValueError):
            issues.append("derived process birth time is outside the supported range")
    observed_at = _parse_time(receipt.get("observed_at"))
    if observed_at is None:
        issues.append("observed_at must be a timezone-aware timestamp")
    elif observed_at > generated_at:
        issues.append("observed_at is later than report generated_at")
    if candidate is not None and resolution is not None and api_birth is not None:
        server_started = candidate["_start"]
        if server_started is not None and not (api_birth <= server_started <= api_birth + timedelta(seconds=resolution)):
            issues.append("API kernel birth window does not contain lifecycle server_started")
    if candidate is not None and gpu_birth is not None and resolution is not None:
        server_started = candidate["_start"]
        if server_started is not None and gpu_birth + timedelta(seconds=resolution) < server_started:
            issues.append("GPU engine birth window precedes lifecycle server_started")
        if observed_at is not None and gpu_birth + timedelta(seconds=resolution) > observed_at:
            issues.append("GPU engine birth window is later than observed_at")
        if candidate["_stop"] is not None and observed_at is not None and observed_at > candidate["_stop"]:
            issues.append("ownership observation is later than server_stopped")

    if issues or candidate is None or gpu_pid is None or gpu_birth is None or resolution is None:
        return metadata, None
    metadata["accepted"] = True
    ownership = {
        "identity": identity,
        "candidate": candidate,
        "gpu_pid": gpu_pid,
        "api_start_ticks": api_start_ticks,
        "gpu_start_ticks": gpu_start_ticks,
        "gpu_birth": gpu_birth,
        "join_start": gpu_birth + timedelta(seconds=resolution),
        "resolution": resolution,
        "observed_at": observed_at,
        "receipt_path": display_path,
        "receipt_sha256": metadata["sha256"],
        "lifecycle_bindings": event_bindings,
        "source_metadata": metadata,
    }
    return metadata, ownership


def _identity(row: SourceRow) -> tuple[str, str] | None:
    tag = row.value.get("tag")
    run_id = row.value.get("run_id")
    if not isinstance(tag, str) or not tag or not isinstance(run_id, str) or not run_id:
        return None
    return tag, run_id


def _event_time(row: SourceRow) -> datetime | None:
    return _parse_time(row.value.get("at"))


def _lifecycle_candidates(rows: list[SourceRow], source_path: str) -> list[dict[str, Any]]:
    relevant = {
        "session_started", "server_started", "server_ready", "server_stopped",
        "server_cleanup_failed", *TERMINAL_EVENTS,
    }
    grouped: dict[tuple[str, str], list[SourceRow]] = {}
    unidentified: list[SourceRow] = []
    for row in rows:
        if row.value.get("event") not in relevant:
            continue
        identity = _identity(row)
        if identity is None:
            unidentified.append(row)
        else:
            grouped.setdefault(identity, []).append(row)

    candidates: list[dict[str, Any]] = []
    for (tag, run_id), group in sorted(grouped.items()):
        by_event: dict[str, list[SourceRow]] = {}
        for row in group:
            by_event.setdefault(str(row.value.get("event")), []).append(row)
        starts = by_event.get("session_started", [])
        servers = by_event.get("server_started", [])
        ready = by_event.get("server_ready", [])
        stopped = by_event.get("server_stopped", [])
        terminal = [row for name in TERMINAL_EVENTS for row in by_event.get(name, [])]
        cleanup_failed = by_event.get("server_cleanup_failed", [])
        notes: list[str] = []
        if not starts:
            notes.append("session_started event is absent; context/backend/model metadata are unavailable")
        elif len(starts) > 1:
            notes.append(f"multiple session_started events ({len(starts)}); session metadata are ambiguous")
        if len(servers) != 1:
            notes.append(f"expected exactly one server_started event; found {len(servers)}")
        if len(ready) > 1:
            notes.append(f"multiple server_ready events ({len(ready)})")
        if len(stopped) > 1:
            notes.append(f"multiple server_stopped events ({len(stopped)})")
        if len(terminal) > 1:
            notes.append(f"multiple terminal session events ({len(terminal)})")
        if cleanup_failed:
            notes.append("server cleanup failure was recorded")

        session_row = starts[0] if len(starts) == 1 else None
        server_row = servers[0] if len(servers) == 1 else None
        ready_row = ready[0] if len(ready) == 1 else None
        stop_row = stopped[0] if len(stopped) == 1 else None
        terminal_row = terminal[0] if len(terminal) == 1 else None
        start_at = _event_time(server_row) if server_row else None
        stop_at = _event_time(stop_row) if stop_row else None
        terminal_at = _event_time(terminal_row) if terminal_row else None
        session_at = _event_time(session_row) if session_row else None
        pid = _integer(server_row.value.get("pid")) if server_row else None

        for label, row, parsed in (
            ("session_started", session_row, session_at),
            ("server_started", server_row, start_at),
            ("server_stopped", stop_row, stop_at),
            ("terminal session", terminal_row, terminal_at),
        ):
            if row is not None and parsed is None:
                notes.append(f"{label} has a missing or invalid timezone-aware timestamp")
        if server_row and pid is None:
            notes.append("server_started has a missing or invalid PID")
        if start_at and session_at and start_at < session_at:
            notes.append("server_started precedes session_started")
        if stop_at and start_at and stop_at < start_at:
            notes.append("server_stopped precedes server_started")
        if terminal_at and start_at and terminal_at < start_at:
            notes.append("terminal session event precedes server_started")

        cleanup_rows = ([stop_row] if stop_row else []) + cleanup_failed
        cleanup_events = [{
            "event": row.value.get("event"),
            "at": _iso(_event_time(row)) if _event_time(row) else None,
            "booleans": {key: value for key, value in row.value.items() if isinstance(value, bool)},
        } for row in cleanup_rows]
        cleanup = cleanup_events[0]["booleans"] if stop_row else (
            cleanup_events[0]["booleans"] if cleanup_events else {}
        )
        initial_load_seconds = None
        if ready_row:
            initial_load_seconds = _number(ready_row.value.get("initial_load_seconds"))
        backend = session_row.value.get("runtime_backend") if session_row else None
        context = session_row.value.get("context") if session_row else None
        model = session_row.value.get("model_name") if session_row else None
        if session_row and not isinstance(backend, str):
            notes.append("session_started lacks runtime_backend")
        if session_row and not isinstance(context, int):
            notes.append("session_started lacks an integer context")

        structurally_valid = not any(
            note.startswith(prefix)
            for note in notes
            for prefix in (
                "expected exactly one server_started", "multiple server_stopped",
                "server_started has", "server_started precedes",
                "server_stopped precedes", "terminal session event precedes",
            )
        ) and start_at is not None
        candidates.append({
            "identity": {"tag": tag, "run_id": run_id},
            "lifecycle_source": source_path,
            "session_started_at": _iso(session_at) if session_at else None,
            "server_pid": pid,
            "server_started_at": _iso(start_at) if start_at else None,
            "server_ready_at": _iso(_event_time(ready_row)) if ready_row and _event_time(ready_row) else None,
            "server_stopped_at": _iso(stop_at) if stop_at else None,
            "terminal_event": terminal_row.value.get("event") if terminal_row else None,
            "terminal_status": terminal_row.value.get("status") if terminal_row else None,
            "runtime_backend": backend if isinstance(backend, str) else None,
            "context_tokens": context if isinstance(context, int) else None,
            "model_name": model if isinstance(model, str) else None,
            "initial_load_seconds": initial_load_seconds,
            "cleanup": cleanup,
            "cleanup_events": cleanup_events,
            "_start": start_at,
            "_stop": stop_at,
            "_terminal": terminal_at,
            "_terminal_count": len(terminal),
            "_valid": structurally_valid,
            "notes": notes,
        })
    if unidentified:
        candidates.append({
            "identity": {"tag": None, "run_id": None},
            "lifecycle_source": source_path,
            "session_started_at": None,
            "server_pid": None,
            "server_started_at": None,
            "server_ready_at": None,
            "server_stopped_at": None,
            "terminal_event": None,
            "terminal_status": None,
            "runtime_backend": None,
            "context_tokens": None,
            "model_name": None,
            "initial_load_seconds": None,
            "cleanup": {},
            "cleanup_events": [],
            "_start": None,
            "_stop": None,
            "_terminal": None,
            "_terminal_count": 0,
            "_valid": False,
            "notes": [f"{len(unidentified)} lifecycle events lack an exact tag/run_id identity"],
        })
    return candidates


def _normalize_telemetry(rows: Iterable[SourceRow]) -> tuple[list[TelemetrySample], list[dict[str, Any]]]:
    samples: list[TelemetrySample] = []
    issues: list[dict[str, Any]] = []
    for row in rows:
        at = _event_time(row)
        raw_gpu = row.value.get("gpu")
        if isinstance(raw_gpu, dict):
            gpu = (raw_gpu,)
        elif isinstance(raw_gpu, list) and all(isinstance(item, dict) for item in raw_gpu):
            gpu = tuple(raw_gpu)
        else:
            gpu = ()
            issues.append({"source": row.source, "line": row.line, "kind": "invalid_gpu_shape"})
        raw_processes = row.value.get("gpu_processes")
        if isinstance(raw_processes, list) and all(isinstance(item, dict) for item in raw_processes):
            processes = tuple(raw_processes)
        else:
            processes = ()
            issues.append({"source": row.source, "line": row.line, "kind": "invalid_process_shape"})
        if at is None:
            issues.append({"source": row.source, "line": row.line, "kind": "invalid_timestamp"})
            continue
        samples.append(TelemetrySample(at, row.source, row.line, gpu, processes))
    return samples, issues


def _sample_fingerprint(sample: TelemetrySample) -> tuple[Any, ...]:
    return (
        sample.at,
        json.dumps(sample.gpu, sort_keys=True, separators=(",", ":")),
        json.dumps(sample.processes, sort_keys=True, separators=(",", ":")),
    )


def _sample_processes(sample: TelemetrySample) -> list[tuple[int | None, float | None]]:
    result = []
    for process in sample.processes:
        vram = process.get("used_mib")
        if vram is None:
            vram = process.get("used_gpu_memory")
        result.append((_integer(process.get("pid")), _number(vram)))
    return result


def _stats(values: list[float], unit: str) -> dict[str, Any] | None:
    if not values:
        return None
    return {
        "unit": unit,
        "sample_count": len(values),
        "min": min(values),
        "median": statistics.median(values),
        "sampled_max": max(values),
    }


def _intervals_overlap(left: tuple[datetime, datetime], right: tuple[datetime, datetime]) -> bool:
    return max(left[0], right[0]) <= min(left[1], right[1])


def _finalize_server(candidate: dict[str, Any], samples: list[TelemetrySample], generated_at: datetime) -> dict[str, Any]:
    notes = list(candidate["notes"])
    start = candidate.pop("_start")
    stop = candidate.pop("_stop")
    terminal = candidate.pop("_terminal")
    terminal_count = candidate.pop("_terminal_count")
    valid = candidate.pop("_valid")
    pid = candidate.pop("_gpu_join_pid")
    join_start = candidate.pop("_gpu_join_start")
    candidate.pop("_gpu_process_start_ticks")

    if stop is not None:
        end = stop
        status = "ended"
        if candidate["cleanup"].get("stopped") is False:
            status = "ended_cleanup_failed"
    elif terminal_count:
        end = None
        status = "missing_stop_terminal_session"
        valid = False
        notes.append("terminal session event has no server_stopped event; interval is not joinable")
    elif start is not None and start <= generated_at:
        end = generated_at
        status = "ongoing_bounded_by_generated_at"
        notes.append("ongoing server interval is bounded by report generated_at")
    else:
        end = None
        status = "invalid_interval"
        valid = False
        if start is not None:
            notes.append("server start is later than report generated_at")

    candidate["interval_status"] = status
    candidate["interval_end_at"] = _iso(end) if end else None
    candidate["join_status"] = (
        "eligible" if valid and end is not None and pid and join_start is not None else "not_joined"
    )
    candidate["coverage"] = {
        "interval_sample_count": 0,
        "owned_pid_observed_sample_count": 0,
        "isolated_sample_count": 0,
        "excluded_sample_count": 0,
        "excluded_reasons": {},
        "first_sample_at": None,
        "last_sample_at": None,
    }
    candidate["process_vram"] = None
    candidate["gpu_metrics"] = {name: None for name in GPU_METRICS}
    candidate["hardware_power_brake"] = {"sample_count": 0, "values": [], "observations": []}
    candidate["telemetry_sources"] = []

    if candidate["join_status"] != "eligible":
        candidate["notes"] = sorted(set(notes))
        return candidate

    interval_samples = [sample for sample in samples if join_start <= sample.at <= end]
    candidate["coverage"]["interval_sample_count"] = len(interval_samples)
    if interval_samples:
        candidate["coverage"]["first_sample_at"] = _iso(interval_samples[0].at)
        candidate["coverage"]["last_sample_at"] = _iso(interval_samples[-1].at)
    isolated: list[tuple[TelemetrySample, float, dict[str, Any]]] = []
    reason_counts: dict[str, int] = {}
    for sample in interval_samples:
        processes = _sample_processes(sample)
        owned_seen = any(process_pid == pid for process_pid, _ in processes)
        if owned_seen:
            candidate["coverage"]["owned_pid_observed_sample_count"] += 1
        reasons: list[str] = []
        if len(sample.gpu) != 1:
            reasons.append("gpu_record_count_not_one")
        if len(processes) != 1:
            reasons.append("gpu_process_count_not_one")
        elif processes[0][0] != pid:
            reasons.append("sole_process_pid_mismatch")
        elif processes[0][1] is None:
            reasons.append("invalid_process_vram")
        if reasons:
            for reason in reasons:
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
            continue
        isolated.append((sample, processes[0][1], sample.gpu[0]))

    candidate["coverage"]["isolated_sample_count"] = len(isolated)
    candidate["coverage"]["excluded_sample_count"] = len(interval_samples) - len(isolated)
    candidate["coverage"]["excluded_reasons"] = reason_counts
    candidate["telemetry_sources"] = sorted({sample.source for sample, _, _ in isolated})
    if not isolated:
        notes.append("no uncontaminated telemetry samples matched the owned GPU PID and interval")
        candidate["join_status"] = "joined_without_usable_samples"
        candidate["notes"] = sorted(set(notes))
        return candidate

    candidate["join_status"] = "joined"
    vram_values = [vram for _, vram, _ in isolated]
    process_vram = _stats(vram_values, "MiB")
    assert process_vram is not None
    process_vram["first_sample"] = {"at": _iso(isolated[0][0].at), "value": isolated[0][1],
                                    "source": isolated[0][0].source, "line": isolated[0][0].line}
    process_vram["last_sample"] = {"at": _iso(isolated[-1][0].at), "value": isolated[-1][1],
                                   "source": isolated[-1][0].source, "line": isolated[-1][0].line}
    candidate["process_vram"] = process_vram
    for name, (field, unit) in GPU_METRICS.items():
        values = [value for _, _, gpu in isolated if (value := _number(gpu.get(field))) is not None]
        candidate["gpu_metrics"][name] = _stats(values, unit)
    brake_values = [gpu[BRAKE_FIELD] for _, _, gpu in isolated if gpu.get(BRAKE_FIELD) is not None]
    distinct_brakes: dict[str, dict[str, Any]] = {}
    for value in brake_values:
        key = json.dumps(value, sort_keys=True, separators=(",", ":"))
        distinct_brakes.setdefault(key, {"value": value, "sample_count": 0})["sample_count"] += 1
    observations = [distinct_brakes[key] for key in sorted(distinct_brakes)]
    candidate["hardware_power_brake"] = {
        "sample_count": len(brake_values),
        "values": [item["value"] for item in observations],
        "observations": observations,
    }
    if reason_counts:
        notes.append("contaminated or structurally ambiguous telemetry samples were excluded")
    candidate["notes"] = sorted(set(notes))
    return candidate


def build_report(results_dir: Path, *, generated_at: datetime | None = None) -> dict[str, Any]:
    """Build a report without writing it or mutating any runtime resources."""
    results_dir = results_dir.resolve()
    if not results_dir.is_dir():
        raise ReportInputError(f"results directory does not exist: {results_dir}")
    generated_at = generated_at or datetime.now(timezone.utc)
    if generated_at.tzinfo is None:
        raise ReportInputError("generated_at must be timezone-aware")
    generated_at = generated_at.astimezone(timezone.utc)
    lifecycle_paths = sorted(results_dir.glob("*-runtime-events.jsonl"))
    telemetry_paths = sorted(results_dir.glob("specialists-machine-telemetry*.jsonl"))
    ownership_paths = sorted(
        path for path in results_dir.iterdir() if OWNERSHIP_NAME.fullmatch(path.name)
    )

    source_metadata: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    lifecycle_rows: dict[str, list[SourceRow]] = {}
    all_telemetry_rows: list[SourceRow] = []
    for path in lifecycle_paths:
        rows, metadata = _read_jsonl(path, "lifecycle")
        source_metadata.append(metadata)
        lifecycle_rows[str(path.resolve())] = rows
        candidates.extend(_lifecycle_candidates(rows, str(path.resolve())))
    telemetry_ranges: dict[str, tuple[datetime, datetime]] = {}
    for path in telemetry_paths:
        rows, metadata = _read_jsonl(path, "telemetry")
        source_metadata.append(metadata)
        all_telemetry_rows.extend(rows)
        times = [time for row in rows if (time := _event_time(row)) is not None]
        if times:
            telemetry_ranges[str(path.resolve())] = (min(times), max(times))

    samples, telemetry_issues = _normalize_telemetry(all_telemetry_rows)
    samples.sort(key=lambda sample: (sample.at, sample.source, sample.line))
    deduplicated: list[TelemetrySample] = []
    fingerprints: set[tuple[Any, ...]] = set()
    duplicate_sample_count = 0
    for sample in samples:
        fingerprint = _sample_fingerprint(sample)
        if fingerprint in fingerprints:
            duplicate_sample_count += 1
            continue
        fingerprints.add(fingerprint)
        deduplicated.append(sample)
    samples = deduplicated

    global_notes: list[str] = []
    range_items = sorted(telemetry_ranges.items())
    for index, (left_path, left_range) in enumerate(range_items):
        for right_path, right_range in range_items[index + 1:]:
            if _intervals_overlap(left_range, right_range):
                global_notes.append(
                    f"telemetry source time ranges overlap: {left_path} and {right_path}; samples were not summed"
                )

    identities: dict[tuple[Any, Any], list[dict[str, Any]]] = {}
    for candidate in candidates:
        identity = candidate["identity"]
        if identity["tag"] is not None:
            identities.setdefault((identity["tag"], identity["run_id"]), []).append(candidate)
    for identity_candidates in identities.values():
        if len(identity_candidates) > 1:
            for candidate in identity_candidates:
                candidate["_valid"] = False
                candidate["notes"].append(
                    f"exact tag/run_id identity appears in {len(identity_candidates)} lifecycle groups"
                )

    for candidate in candidates:
        candidate["gpu_process_pid"] = None
        candidate["gpu_process_started_at"] = None
        candidate["gpu_process_start_time_resolution_seconds"] = None
        candidate["gpu_ownership"] = None
        candidate["_gpu_join_pid"] = candidate["server_pid"]
        candidate["_gpu_join_start"] = candidate["_start"]
        candidate["_gpu_process_start_ticks"] = None

    receipt_metadata: list[dict[str, Any]] = []
    ownerships_by_candidate: dict[int, list[dict[str, Any]]] = {}
    declared_by_candidate: dict[int, list[dict[str, Any]]] = {}
    candidates_by_identity = {
        (str(identity[0]), str(identity[1])): identity_candidates
        for identity, identity_candidates in identities.items()
    }
    for path in ownership_paths:
        metadata, ownership = _validate_ownership_receipt(
            path, lifecycle_rows, candidates_by_identity, generated_at,
        )
        receipt_metadata.append(metadata)
        identity = metadata.get("identity")
        matching_candidates = candidates_by_identity.get(
            (identity.get("tag"), identity.get("run_id")), []
        ) if isinstance(identity, dict) else []
        for candidate in matching_candidates:
            declared_by_candidate.setdefault(id(candidate), []).append(metadata)
        if ownership is not None:
            ownerships_by_candidate.setdefault(id(ownership["candidate"]), []).append(ownership)
        if metadata["issues"]:
            global_notes.append(
                f"GPU ownership receipt rejected: {metadata['path']}: "
                + "; ".join(metadata["issues"])
            )

    selected_ownerships: list[dict[str, Any]] = []
    for candidate in candidates:
        declared = declared_by_candidate.get(id(candidate), [])
        valid_ownerships = ownerships_by_candidate.get(id(candidate), [])
        if len(declared) > 1:
            note = f"multiple GPU ownership receipts ({len(declared)}) match this lifecycle identity"
            candidate["notes"].append(note)
            global_notes.append(
                f"GPU ownership is ambiguous for {candidate['identity']['tag']}/"
                f"{candidate['identity']['run_id']}: {note}"
            )
            for metadata in declared:
                metadata["accepted"] = False
                metadata["issues"].append(note)
            continue
        if not valid_ownerships:
            if candidate["runtime_backend"] == "strata":
                note = (
                    "no unique valid GPU ownership receipt; telemetry retains the API server PID join key"
                )
                candidate["notes"].append(note)
                global_notes.append(
                    f"{candidate['identity']['tag']}/{candidate['identity']['run_id']}: {note}"
                )
            continue
        ownership = valid_ownerships[0]
        selected_ownerships.append(ownership)
        candidate["gpu_process_pid"] = ownership["gpu_pid"]
        candidate["gpu_process_started_at"] = _iso(ownership["gpu_birth"])
        candidate["gpu_process_start_time_resolution_seconds"] = ownership["resolution"]
        candidate["gpu_ownership"] = {
            "receipt_path": ownership["receipt_path"],
            "receipt_sha256": ownership["receipt_sha256"],
            "observed_at": _iso(ownership["observed_at"]),
            "api_process_start_ticks": str(ownership["api_start_ticks"]),
            "gpu_process_start_ticks": str(ownership["gpu_start_ticks"]),
            "lifecycle_event_bindings": ownership["lifecycle_bindings"],
        }
        candidate["_gpu_join_pid"] = ownership["gpu_pid"]
        candidate["_gpu_join_start"] = max(candidate["_start"], ownership["join_start"])
        candidate["_gpu_process_start_ticks"] = ownership["gpu_start_ticks"]
        candidate["notes"].append(
            "GPU telemetry ownership is bound to a validated direct engine child receipt"
        )

    for index, left_ownership in enumerate(selected_ownerships):
        left = left_ownership["candidate"]
        if left["gpu_process_pid"] is None:
            continue
        for right_ownership in selected_ownerships[index + 1:]:
            right = right_ownership["candidate"]
            if right["gpu_process_pid"] is None:
                continue
            same_process_identity = (
                left["gpu_process_pid"] == right["gpu_process_pid"]
                and left["_gpu_process_start_ticks"] == right["_gpu_process_start_ticks"]
            )
            if not same_process_identity:
                continue
            note = (
                f"GPU process identity {left['gpu_process_pid']}/"
                f"{left['_gpu_process_start_ticks']} is claimed by multiple lifecycle identities"
            )
            for candidate, ownership in ((left, left_ownership), (right, right_ownership)):
                candidate["_valid"] = False
                candidate["notes"].append(note)
                ownership["source_metadata"]["accepted"] = False
                ownership["source_metadata"]["issues"].append(note)
            global_notes.append(note)

    provisional: list[tuple[dict[str, Any], datetime, datetime, int]] = []
    for candidate in candidates:
        start = candidate["_gpu_join_start"]
        if candidate["_stop"] is not None:
            end = candidate["_stop"]
        elif candidate["_terminal_count"] == 0:
            end = generated_at
        else:
            end = None
        if candidate["_valid"] and start and end and candidate["_gpu_join_pid"]:
            provisional.append((candidate, start, end, candidate["_gpu_join_pid"]))
    for index, (left, left_start, left_end, left_pid) in enumerate(provisional):
        for right, right_start, right_end, right_pid in provisional[index + 1:]:
            if left_pid == right_pid and _intervals_overlap((left_start, left_end), (right_start, right_end)):
                left["_valid"] = False
                right["_valid"] = False
                note = (
                    f"GPU telemetry PID {left_pid} has overlapping lifecycle intervals; "
                    "telemetry ownership is ambiguous"
                )
                left["notes"].append(note)
                right["notes"].append(note)
                global_notes.append(note)
                for candidate in (left, right):
                    for ownership in selected_ownerships:
                        if ownership["candidate"] is candidate:
                            ownership["source_metadata"]["accepted"] = False
                            ownership["source_metadata"]["issues"].append(note)
                            break

    servers = [_finalize_server(candidate, samples, generated_at) for candidate in candidates]
    servers.sort(key=lambda item: (
        item["server_started_at"] or "", item["identity"]["tag"] or "", item["lifecycle_source"]
    ))
    malformed_count = sum(source["invalid_record_count"] for source in source_metadata)
    report = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": _iso(generated_at),
        "disclaimers": [
            "sampled_max is the maximum observed sample, not a true peak",
            "GPU board power is neither model energy nor whole-machine power",
            "telemetry is sampled at approximately 30-second intervals; overlapping source parts are not summed",
        ],
        "sources": {
            "lifecycle": [item for item in source_metadata if item["kind"] == "lifecycle"],
            "telemetry": [item for item in source_metadata if item["kind"] == "telemetry"],
            "gpu_ownership": receipt_metadata,
        },
        "telemetry_issues": telemetry_issues,
        "coverage_notes": global_notes,
        "summary": {
            "lifecycle_file_count": len(lifecycle_paths),
            "telemetry_file_count": len(telemetry_paths),
            "gpu_ownership_receipt_count": len(receipt_metadata),
            "accepted_gpu_ownership_receipt_count": sum(
                source["accepted"] for source in receipt_metadata
            ),
            "server_count": len(servers),
            "joined_server_count": sum(server["join_status"] == "joined" for server in servers),
            "joined_without_usable_samples_count": sum(
                server["join_status"] == "joined_without_usable_samples" for server in servers
            ),
            "not_joined_server_count": sum(server["join_status"] == "not_joined" for server in servers),
            "ongoing_server_count": sum(
                server["interval_status"] == "ongoing_bounded_by_generated_at" for server in servers
            ),
            "malformed_or_non_object_line_count": malformed_count,
            "telemetry_normalization_issue_count": len(telemetry_issues),
            "deduplicated_telemetry_sample_count": duplicate_sample_count,
            "usable_telemetry_sample_count": len(samples),
        },
        "servers": servers,
    }
    return report


def _csv_row(server: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "tag": server["identity"]["tag"],
        "run_id": server["identity"]["run_id"],
        "lifecycle_source": server["lifecycle_source"],
        "runtime_backend": server["runtime_backend"],
        "model_name": server["model_name"],
        "context_tokens": server["context_tokens"],
        "server_pid": server["server_pid"],
        "gpu_process_pid": server["gpu_process_pid"],
        "gpu_process_started_at": server["gpu_process_started_at"],
        "gpu_process_start_time_resolution_seconds": (
            server["gpu_process_start_time_resolution_seconds"]
        ),
        "gpu_ownership_receipt_path": (
            server["gpu_ownership"]["receipt_path"] if server["gpu_ownership"] else None
        ),
        "gpu_ownership_receipt_sha256": (
            server["gpu_ownership"]["receipt_sha256"] if server["gpu_ownership"] else None
        ),
        "server_started_at": server["server_started_at"],
        "server_ready_at": server["server_ready_at"],
        "server_stopped_at": server["server_stopped_at"],
        "interval_end_at": server["interval_end_at"],
        "interval_status": server["interval_status"],
        "join_status": server["join_status"],
        "initial_load_seconds": server["initial_load_seconds"],
        "interval_sample_count": server["coverage"]["interval_sample_count"],
        "isolated_sample_count": server["coverage"]["isolated_sample_count"],
        "excluded_sample_count": server["coverage"]["excluded_sample_count"],
        "cleanup_json": json.dumps(server["cleanup"], sort_keys=True, separators=(",", ":")),
        "cleanup_events_json": json.dumps(
            server["cleanup_events"], sort_keys=True, separators=(",", ":")
        ),
        "hardware_power_brake_values_json": json.dumps(
            server["hardware_power_brake"]["values"], separators=(",", ":")
        ),
        "notes": " | ".join(server["notes"]),
    }
    vram = server["process_vram"]
    for field in ("sample_count", "min", "median", "sampled_max"):
        row[f"process_vram_{field}_mib"] = vram.get(field) if vram else None
    row["process_vram_first_at"] = vram["first_sample"]["at"] if vram else None
    row["process_vram_last_at"] = vram["last_sample"]["at"] if vram else None
    for metric_name, metric in server["gpu_metrics"].items():
        for field in ("sample_count", "min", "median", "sampled_max"):
            row[f"{metric_name}_{field}"] = metric.get(field) if metric else None
    return row


def write_report(report: dict[str, Any], output_prefix: Path) -> tuple[Path, Path]:
    json_path = Path(str(output_prefix) + ".json")
    csv_path = Path(str(output_prefix) + ".csv")
    existing = [path for path in (json_path, csv_path) if path.exists()]
    if existing:
        raise ReportInputError("refusing to overwrite existing output: " + ", ".join(map(str, existing)))
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    rows = [_csv_row(server) for server in report["servers"]]
    fieldnames = list(_csv_row({
        "identity": {"tag": None, "run_id": None}, "lifecycle_source": None,
        "runtime_backend": None, "model_name": None, "context_tokens": None,
        "server_pid": None, "gpu_process_pid": None, "gpu_process_started_at": None,
        "gpu_process_start_time_resolution_seconds": None, "gpu_ownership": None,
        "server_started_at": None, "server_ready_at": None,
        "server_stopped_at": None, "interval_end_at": None, "interval_status": None,
        "join_status": None, "initial_load_seconds": None,
        "coverage": {"interval_sample_count": 0, "isolated_sample_count": 0, "excluded_sample_count": 0},
        "cleanup": {}, "hardware_power_brake": {"values": []}, "notes": [],
        "cleanup_events": [],
        "process_vram": None, "gpu_metrics": {name: None for name in GPU_METRICS},
    }).keys())
    with csv_path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return json_path, csv_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--results-dir", type=Path, default=root / "results")
    parser.add_argument(
        "--output-prefix", type=Path, default=root / "results" / "specialist-resource-report",
        help="output path without extension; .json and .csv are created exclusively",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        report = build_report(args.results_dir)
        json_path, csv_path = write_report(report, args.output_prefix)
    except (OSError, ReportInputError) as exc:
        parser.error(str(exc))
    print(json.dumps({
        "json_output": str(json_path.resolve()),
        "csv_output": str(csv_path.resolve()),
        **report["summary"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
