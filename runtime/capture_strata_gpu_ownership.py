#!/usr/bin/env python3
"""Capture a metadata-only Strata API/direct-GPU-child ownership receipt."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import tempfile
from typing import Any, Callable, Mapping, Sequence


MAX_CONFIG_BYTES = 4 * 1024 * 1024
MAX_LIFECYCLE_BYTES = 128 * 1024 * 1024
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
TAG_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
TAIL_ENVIRONMENT = {"STRATA_ROUTE_TAIL_SKIP": "0"}
TERMINAL_EVENTS = {
    "server_stopped",
    "server_cleanup_failed",
    "session_complete",
    "session_failed",
    "session_interrupted",
}
STOPPED_PROCESS_STATES = {"T", "t", "Z", "X", "x"}
SCOPE = (
    "API parent PID and direct owned GPU engine child PID are distinct. "
    "This observation binds ownership and does not claim a memory peak or speed result."
)


class CaptureError(RuntimeError):
    """Raised when live metadata cannot support an ownership receipt."""


@dataclass(frozen=True)
class FileBinding:
    path: Path
    sha256: str
    size: int

    def as_dict(self) -> dict[str, Any]:
        return {"path": str(self.path), "sha256": self.sha256, "size": self.size}


@dataclass(frozen=True)
class ProcessIdentity:
    pid: int
    parent_pid: int
    start_ticks: str
    state: str
    exe: str

    def stable_key(self) -> tuple[int, int, str, str]:
        return self.pid, self.parent_pid, self.start_ticks, self.exe


@dataclass(frozen=True)
class KernelClock:
    boot_id: str
    boot_time_unix_seconds: int
    clock_ticks_per_second: int
    boot_time_resolution_seconds: int = 1


@dataclass(frozen=True)
class LifecycleRow:
    line: int
    value: Mapping[str, Any]
    raw_sha256: str
    canonical_sha256: str


@dataclass(frozen=True)
class LifecycleSnapshot:
    start: LifecycleRow
    server: LifecycleRow
    ready: LifecycleRow
    run_id: str
    api_pid: int
    model_name: str
    context: int
    engine: str


@dataclass(frozen=True)
class Dependencies:
    process_identity: Callable[[int], ProcessIdentity]
    process_environment: Callable[[int], Mapping[str, str]]
    gpu_processes: Callable[[], Sequence[Mapping[str, Any]]]
    kernel_clock: Callable[[], KernelClock]
    now: Callable[[], datetime]


def _strict_json(payload: bytes, description: str) -> Mapping[str, Any]:
    def pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(payload.decode("utf-8"), object_pairs_hook=pairs_hook)
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise CaptureError(f"{description} is not strict JSON") from exc
    if not isinstance(value, dict):
        raise CaptureError(f"{description} must be a JSON object")
    return value


def _read_regular(path: Path, description: str, maximum: int) -> tuple[bytes, FileBinding]:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise CaptureError(f"{description} is missing, inaccessible, or a symlink: {path}") from exc
    try:
        details = os.fstat(descriptor)
        if not stat.S_ISREG(details.st_mode):
            raise CaptureError(f"{description} must be a regular non-symlink file: {path}")
        if details.st_size > maximum:
            raise CaptureError(f"{description} exceeds its metadata size limit")
        blocks: list[bytes] = []
        total = 0
        while True:
            block = os.read(descriptor, min(1024 * 1024, maximum + 1 - total))
            if not block:
                break
            blocks.append(block)
            total += len(block)
            if total > maximum:
                raise CaptureError(f"{description} exceeds its metadata size limit")
    finally:
        os.close(descriptor)
    payload = b"".join(blocks)
    resolved = path.resolve()
    return payload, FileBinding(resolved, hashlib.sha256(payload).hexdigest(), len(payload))


def _load_config(path: Path, description: str) -> tuple[Mapping[str, Any], bytes, FileBinding]:
    payload, binding = _read_regular(path, description, MAX_CONFIG_BYTES)
    return _strict_json(payload, description), payload, binding


def _parse_time(value: Any, description: str) -> datetime:
    if not isinstance(value, str):
        raise CaptureError(f"{description} timestamp is absent")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise CaptureError(f"{description} timestamp is invalid") from exc
    if parsed.tzinfo is None:
        raise CaptureError(f"{description} timestamp must include a timezone")
    return parsed.astimezone(timezone.utc)


def _read_lifecycle_rows(path: Path) -> list[LifecycleRow]:
    payload, _ = _read_regular(path, "runtime lifecycle", MAX_LIFECYCLE_BYTES)
    rows: list[LifecycleRow] = []
    for line_number, raw in enumerate(payload.splitlines(), 1):
        if not raw.strip():
            raise CaptureError("runtime lifecycle contains a blank row")
        value = _strict_json(raw, f"runtime lifecycle line {line_number}")
        canonical = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
        rows.append(LifecycleRow(
            line_number,
            value,
            hashlib.sha256(raw).hexdigest(),
            hashlib.sha256(canonical).hexdigest(),
        ))
    if not rows:
        raise CaptureError("runtime lifecycle is empty")
    return rows


def _option_value(args: Any, option: str) -> str:
    if not isinstance(args, list) or not all(isinstance(item, str) for item in args):
        raise CaptureError("source config args must be a string list")
    values: list[str] = []
    for index, item in enumerate(args):
        if item == option:
            if index + 1 >= len(args):
                raise CaptureError(f"source config {option} lacks a value")
            values.append(args[index + 1])
        elif item.startswith(option + "="):
            values.append(item.split("=", 1)[1])
    if len(values) != 1:
        raise CaptureError(f"source config must contain exactly one {option}")
    return values[0]


def _validate_config_pair(
    source: Mapping[str, Any],
    private: Mapping[str, Any],
    *,
    expected_log: Path,
    allow_model_session_reasoning_default_none: bool,
) -> tuple[str, str, int]:
    engine = source.get("exe")
    model_name = source.get("model_name")
    if not isinstance(engine, str) or not Path(engine).is_absolute() or Path(engine).name != "strata":
        raise CaptureError("source config is not an absolute Strata engine configuration")
    if not isinstance(model_name, str) or not model_name:
        raise CaptureError("source config model_name is absent")
    try:
        context = int(_option_value(source.get("args"), "--max-context"))
    except ValueError as exc:
        raise CaptureError("source config --max-context is not an integer") from exc
    if context <= 0:
        raise CaptureError("source config --max-context must be positive")
    source_environment = source.get("reference_environment")
    private_environment = private.get("reference_environment")
    if (
        not isinstance(source_environment, dict)
        or source_environment.get("STRATA_ROUTE_TAIL_SKIP") != "0"
        or not isinstance(private_environment, dict)
        or private_environment.get("STRATA_ROUTE_TAIL_SKIP") != "0"
    ):
        raise CaptureError("source/private config does not bind STRATA_ROUTE_TAIL_SKIP=0")
    strict_private = dict(source)
    strict_private["log"] = str(expected_log)
    allowed_private = strict_private
    if allow_model_session_reasoning_default_none:
        if source.get("reasoning_effort") is not None:
            raise CaptureError(
                "reasoning-default opt-in requires source reasoning_effort to be absent or null"
            )
        allowed_private = dict(strict_private)
        allowed_private["reasoning_effort"] = "none"
    if private != allowed_private or source.get("log") == private.get("log"):
        if allow_model_session_reasoning_default_none:
            raise CaptureError(
                "private config differs beyond the per-tag log and allowed reasoning_effort=none"
            )
        raise CaptureError("private config must differ from source config only at the per-tag log path")
    return engine, model_name, context


def _active_reasoning_step(rows: Sequence[LifecycleRow]) -> LifecycleRow:
    active: dict[tuple[int, str], LifecycleRow] = {}
    for row in rows:
        event = row.value.get("event")
        if event not in {"step_started", "step_complete"}:
            continue
        index = row.value.get("index")
        step_id = row.value.get("step_id") if event == "step_started" else row.value.get("id")
        if isinstance(index, bool) or not isinstance(index, int) or index <= 0:
            raise CaptureError(f"{event} has an invalid step index")
        if not isinstance(step_id, str) or not step_id:
            raise CaptureError(f"{event} has an invalid step identity")
        identity = index, step_id
        if event == "step_started":
            if identity in active:
                raise CaptureError("lifecycle has a duplicated active step_started identity")
            active[identity] = row
        elif identity not in active:
            raise CaptureError("step_complete lacks its unique step_started identity")
        else:
            del active[identity]
    if len(active) != 1:
        raise CaptureError("reasoning-default opt-in requires exactly one active step_started row")
    return next(iter(active.values()))


def _validate_per_request_reasoning(rows: Sequence[LifecycleRow]) -> None:
    step = _active_reasoning_step(rows)
    controls = step.value.get("controls")
    argv = step.value.get("argv")
    if not isinstance(controls, dict) or controls.get("per_request_reasoning_effort") is not True:
        raise CaptureError(
            "reasoning-default opt-in requires active per_request_reasoning_effort=true"
        )
    thinking = controls.get("thinking")
    if not isinstance(thinking, str) or not thinking:
        raise CaptureError("reasoning-default opt-in active controls.thinking is absent")
    if not isinstance(argv, list) or not all(isinstance(item, str) for item in argv):
        raise CaptureError("reasoning-default opt-in active benchmark argv is absent")
    values = [
        argv[index + 1]
        for index, item in enumerate(argv[:-1])
        if item == "--thinking"
    ]
    values.extend(item.split("=", 1)[1] for item in argv if item.startswith("--thinking="))
    if values != [thinking]:
        raise CaptureError(
            "reasoning-default opt-in requires one --thinking value matching active controls"
        )


def _validate_lifecycle(
    rows: Sequence[LifecycleRow],
    *,
    tag: str,
    source_path: Path,
    private_path: Path,
    engine: str,
    model_name: str,
    context: int,
    require_per_request_reasoning_proof: bool,
) -> LifecycleSnapshot:
    starts = [row for row in rows if row.value.get("event") == "session_started"]
    servers = [row for row in rows if row.value.get("event") == "server_started"]
    ready = [row for row in rows if row.value.get("event") == "server_ready"]
    if len(starts) != 1 or len(servers) != 1 or len(ready) != 1:
        raise CaptureError("lifecycle must contain exactly one session_started, server_started, and server_ready")
    start, server, ready_row = starts[0], servers[0], ready[0]
    run_id = start.value.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise CaptureError("lifecycle run_id is absent")
    if any(row.value.get("tag") != tag or row.value.get("run_id") != run_id for row in rows):
        raise CaptureError("lifecycle contains ambiguous tag/run identities")
    terminal = [str(row.value.get("event")) for row in rows if row.value.get("event") in TERMINAL_EVENTS]
    if terminal:
        raise CaptureError("lifecycle is terminal or stopped: " + ", ".join(terminal))
    for row, label in ((start, "session_started"), (server, "server_started"), (ready_row, "server_ready")):
        _parse_time(row.value.get("at"), label)
        if row.value.get("runtime_backend") != "strata":
            raise CaptureError(f"{label} does not identify the Strata backend")
    if (
        start.value.get("source_config") != str(source_path)
        or start.value.get("private_config") != str(private_path)
        or start.value.get("engine") != engine
        or start.value.get("model_name") != model_name
        or start.value.get("context") != context
        or start.value.get("engine_environment") != TAIL_ENVIRONMENT
    ):
        raise CaptureError("session_started source/config/engine/model/context/environment identity differs")
    api_pid = server.value.get("pid")
    if isinstance(api_pid, bool) or not isinstance(api_pid, int) or api_pid <= 0:
        raise CaptureError("server_started API PID is invalid")
    argv = server.value.get("argv")
    if not isinstance(argv, list) or not all(isinstance(item, str) for item in argv):
        raise CaptureError("server_started argv is absent")
    config_selected = any(
        argv[index] == "--config" and argv[index + 1] == str(private_path)
        for index in range(len(argv) - 1)
    )
    if not config_selected:
        raise CaptureError("server_started argv does not select the per-tag private config")
    if ready_row.value.get("model") != model_name:
        raise CaptureError("server_ready model differs from the source config")
    if require_per_request_reasoning_proof:
        _validate_per_request_reasoning(rows)
    return LifecycleSnapshot(start, server, ready_row, run_id, api_pid, model_name, context, engine)


def _proc_identity(pid: int) -> ProcessIdentity:
    try:
        raw = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        close = raw.rfind(")")
        fields = raw[close + 2:].split()
        executable = os.readlink(f"/proc/{pid}/exe")
    except OSError as exc:
        raise CaptureError(f"process identity is unavailable for PID {pid}") from exc
    if close < 1 or len(fields) < 20:
        raise CaptureError(f"process stat is malformed for PID {pid}")
    try:
        parent_pid = int(fields[1])
        start_ticks = fields[19]
    except ValueError as exc:
        raise CaptureError(f"process stat contains invalid identity fields for PID {pid}") from exc
    identity = ProcessIdentity(pid, parent_pid, start_ticks, fields[0], executable)
    _validate_process_identity(identity, f"PID {pid}")
    return identity


def _validate_process_identity(identity: ProcessIdentity, description: str) -> None:
    if (
        isinstance(identity.pid, bool)
        or not isinstance(identity.pid, int)
        or identity.pid <= 0
        or isinstance(identity.parent_pid, bool)
        or not isinstance(identity.parent_pid, int)
        or identity.parent_pid <= 0
        or not isinstance(identity.start_ticks, str)
        or not identity.start_ticks.isdecimal()
        or int(identity.start_ticks) <= 0
        or not isinstance(identity.exe, str)
        or not identity.exe
        or not isinstance(identity.state, str)
        or len(identity.state) != 1
    ):
        raise CaptureError(f"{description} has an invalid process identity")
    if identity.state in STOPPED_PROCESS_STATES:
        raise CaptureError(f"{description} is terminal or stopped")


def _process_environment(pid: int) -> Mapping[str, str]:
    try:
        raw = Path(f"/proc/{pid}/environ").read_bytes()
    except OSError as exc:
        raise CaptureError(f"process environment is unavailable for PID {pid}") from exc
    environment: dict[str, str] = {}
    for item in raw.split(b"\0"):
        if b"=" not in item:
            continue
        key, value = item.split(b"=", 1)
        environment[key.decode("utf-8", "surrogateescape")] = value.decode(
            "utf-8", "surrogateescape"
        )
    return environment


def _gpu_processes() -> Sequence[Mapping[str, Any]]:
    command = [
        "nvidia-smi",
        "--query-compute-apps=pid,process_name,used_gpu_memory",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise CaptureError("could not inspect GPU compute processes") from exc
    if completed.returncode != 0:
        raise CaptureError("nvidia-smi could not inspect GPU compute processes")
    observations: list[dict[str, Any]] = []
    for line in completed.stdout.splitlines():
        if not line.strip():
            continue
        fields = [field.strip() for field in line.rsplit(",", 2)]
        if len(fields) != 3:
            raise CaptureError("nvidia-smi returned an invalid compute-process row")
        try:
            pid = int(fields[0])
            used_mib = int(fields[2])
        except ValueError as exc:
            raise CaptureError("nvidia-smi returned non-numeric compute-process data") from exc
        observations.append({"pid": pid, "name": fields[1], "used_mib": used_mib})
    return observations


def _kernel_clock() -> KernelClock:
    try:
        boot_id = Path("/proc/sys/kernel/random/boot_id").read_text(encoding="ascii").strip()
        boot_times = [
            int(line.split()[1])
            for line in Path("/proc/stat").read_text(encoding="ascii").splitlines()
            if line.startswith("btime ")
        ]
        ticks = int(os.sysconf("SC_CLK_TCK"))
    except (OSError, ValueError) as exc:
        raise CaptureError("kernel clock identity is unavailable") from exc
    if not boot_id or len(boot_times) != 1 or boot_times[0] <= 0 or ticks <= 0:
        raise CaptureError("kernel clock identity is invalid")
    return KernelClock(boot_id, boot_times[0], ticks)


def _now() -> datetime:
    return datetime.now(timezone.utc)


SYSTEM_DEPENDENCIES = Dependencies(
    process_identity=_proc_identity,
    process_environment=_process_environment,
    gpu_processes=_gpu_processes,
    kernel_clock=_kernel_clock,
    now=_now,
)


def _validated_gpu_observation(
    observations: Sequence[Mapping[str, Any]],
    *,
    api: ProcessIdentity,
    engine: str,
    process_identity: Callable[[int], ProcessIdentity],
) -> tuple[Mapping[str, Any], ProcessIdentity]:
    seen: set[int] = set()
    candidates: list[tuple[Mapping[str, Any], ProcessIdentity]] = []
    for observation in observations:
        pid = observation.get("pid")
        name = observation.get("name")
        used_mib = observation.get("used_mib")
        if (
            isinstance(pid, bool)
            or not isinstance(pid, int)
            or pid <= 0
            or not isinstance(name, str)
            or not name
            or isinstance(used_mib, bool)
            or not isinstance(used_mib, (int, float))
            or not math.isfinite(float(used_mib))
            or float(used_mib) < 0
        ):
            raise CaptureError("GPU process observation contains invalid metadata")
        if pid in seen:
            raise CaptureError("GPU process observation duplicates a PID")
        seen.add(pid)
        identity = process_identity(pid)
        _validate_process_identity(identity, f"GPU-observed PID {pid}")
        if identity.pid != pid:
            raise CaptureError("GPU process identity reader returned a different PID")
        if identity.parent_pid == api.pid and identity.exe == engine and name == engine:
            candidates.append((observation, identity))
    if len(candidates) != 1:
        raise CaptureError("expected exactly one direct Strata GPU-engine child of the API")
    observation, gpu = candidates[0]
    if gpu.pid == api.pid or int(gpu.start_ticks) <= int(api.start_ticks):
        raise CaptureError("GPU child identity does not follow the distinct API identity")
    return observation, gpu


def _assert_stable(before: ProcessIdentity, after: ProcessIdentity, description: str) -> None:
    _validate_process_identity(after, description)
    if before.stable_key() != after.stable_key():
        raise CaptureError(f"{description} PID/PPID/start/executable identity changed")


def _format_time(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _write_exclusive_atomic(path: Path, value: Mapping[str, Any]) -> FileBinding:
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    parent = path.parent
    try:
        details = parent.lstat()
    except OSError as exc:
        raise CaptureError(f"output directory is unavailable: {parent}") from exc
    if parent.is_symlink() or not stat.S_ISDIR(details.st_mode):
        raise CaptureError(f"output directory must be a real directory: {parent}")
    descriptor = -1
    temporary = ""
    linked = False
    try:
        descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=parent)
        os.fchmod(descriptor, 0o600)
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise OSError("short receipt write")
            offset += written
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError as exc:
            raise CaptureError(f"refusing to overwrite existing output: {path}") from exc
        linked = True
        directory = os.open(parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        if linked:
            try:
                if os.path.samefile(temporary, path):
                    path.unlink()
            except OSError:
                pass
        raise
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
    return FileBinding(path.resolve(), hashlib.sha256(payload).hexdigest(), len(payload))


def capture(
    *,
    tag: str,
    source_config: Path,
    results_dir: Path,
    output: Path | None = None,
    expected_source_sha256: str | None = None,
    allow_model_session_reasoning_default_none: bool = False,
    dependencies: Dependencies = SYSTEM_DEPENDENCIES,
) -> tuple[Path, Mapping[str, Any]]:
    """Capture and exclusively write one ownership receipt."""
    if TAG_RE.fullmatch(tag) is None:
        raise CaptureError("tag must contain only letters, digits, dot, underscore, and hyphen")
    if expected_source_sha256 is not None and SHA256_RE.fullmatch(expected_source_sha256) is None:
        raise CaptureError("expected source SHA-256 must be 64 lowercase hexadecimal characters")
    results_dir = results_dir.absolute()
    try:
        result_details = results_dir.lstat()
    except OSError as exc:
        raise CaptureError(f"results directory is unavailable: {results_dir}") from exc
    if results_dir.is_symlink() or not stat.S_ISDIR(result_details.st_mode):
        raise CaptureError("results directory must be a real directory")
    source_path = source_config.absolute()
    lifecycle_path = results_dir / f"{tag}-runtime-events.jsonl"
    private_path = results_dir / f"{tag}-config.json"
    expected_log = results_dir / f"{tag}-engine.log"
    expected_name = f"{tag}-gpu-ownership-v1.json"
    output_path = (results_dir / expected_name) if output is None else output.absolute()
    if output_path.name != expected_name:
        raise CaptureError(f"output filename must be {expected_name} for schema version 1")
    if os.path.lexists(output_path):
        raise CaptureError(f"refusing to overwrite existing output: {output_path}")

    source, source_payload, source_binding = _load_config(source_path, "source config")
    if expected_source_sha256 is not None and source_binding.sha256 != expected_source_sha256:
        raise CaptureError("source config SHA-256 differs from the expected pin")
    private, private_payload, private_binding = _load_config(private_path, "private config")
    engine, model_name, context = _validate_config_pair(
        source,
        private,
        expected_log=expected_log,
        allow_model_session_reasoning_default_none=allow_model_session_reasoning_default_none,
    )
    rows = _read_lifecycle_rows(lifecycle_path)
    lifecycle = _validate_lifecycle(
        rows,
        tag=tag,
        source_path=source_binding.path,
        private_path=private_binding.path,
        engine=engine,
        model_name=model_name,
        context=context,
        require_per_request_reasoning_proof=allow_model_session_reasoning_default_none,
    )

    api_before = dependencies.process_identity(lifecycle.api_pid)
    _validate_process_identity(api_before, "API process")
    if api_before.pid != lifecycle.api_pid:
        raise CaptureError("API process identity reader returned a different PID")
    api_parent_before = dependencies.process_identity(api_before.parent_pid)
    _validate_process_identity(api_parent_before, "API parent process")
    if api_parent_before.pid != api_before.parent_pid:
        raise CaptureError("API parent identity reader returned a different PID")
    observations = dependencies.gpu_processes()
    observation, gpu_before = _validated_gpu_observation(
        observations,
        api=api_before,
        engine=engine,
        process_identity=dependencies.process_identity,
    )
    if dependencies.process_environment(gpu_before.pid).get("STRATA_ROUTE_TAIL_SKIP") != "0":
        raise CaptureError("GPU engine environment does not have STRATA_ROUTE_TAIL_SKIP=0")
    clock = dependencies.kernel_clock()
    if (
        not isinstance(clock.boot_id, str)
        or not clock.boot_id
        or clock.boot_time_unix_seconds <= 0
        or clock.clock_ticks_per_second <= 0
        or clock.boot_time_resolution_seconds <= 0
    ):
        raise CaptureError("kernel clock identity is invalid")

    final_source, final_source_binding = _read_regular(source_path, "source config", MAX_CONFIG_BYTES)
    final_private, final_private_binding = _read_regular(private_path, "private config", MAX_CONFIG_BYTES)
    if (
        final_source != source_payload
        or final_source_binding != source_binding
        or final_private != private_payload
        or final_private_binding != private_binding
    ):
        raise CaptureError("source or private config bytes changed during capture")
    final_rows = _read_lifecycle_rows(lifecycle_path)
    final_lifecycle = _validate_lifecycle(
        final_rows,
        tag=tag,
        source_path=source_binding.path,
        private_path=private_binding.path,
        engine=engine,
        model_name=model_name,
        context=context,
        require_per_request_reasoning_proof=allow_model_session_reasoning_default_none,
    )
    if (
        final_lifecycle.run_id != lifecycle.run_id
        or final_lifecycle.api_pid != lifecycle.api_pid
        or final_lifecycle.start != lifecycle.start
        or final_lifecycle.server != lifecycle.server
    ):
        raise CaptureError("bound lifecycle identity changed during capture")
    api_after = dependencies.process_identity(api_before.pid)
    gpu_after = dependencies.process_identity(gpu_before.pid)
    api_parent_after = dependencies.process_identity(api_parent_before.pid)
    _assert_stable(api_before, api_after, "API process")
    _assert_stable(gpu_before, gpu_after, "GPU process")
    _assert_stable(api_parent_before, api_parent_after, "API parent process")
    if gpu_after.parent_pid != api_after.pid or gpu_after.exe != engine:
        raise CaptureError("GPU process is no longer the expected direct API child")
    if dependencies.process_environment(gpu_after.pid).get("STRATA_ROUTE_TAIL_SKIP") != "0":
        raise CaptureError("GPU engine tail environment changed during capture")

    observed_at = dependencies.now()
    if observed_at.tzinfo is None:
        raise CaptureError("observation clock must be timezone-aware")
    observed_at = observed_at.astimezone(timezone.utc)
    api_birth = clock.boot_time_unix_seconds + int(api_after.start_ticks) / clock.clock_ticks_per_second
    gpu_birth = clock.boot_time_unix_seconds + int(gpu_after.start_ticks) / clock.clock_ticks_per_second
    server_at = _parse_time(lifecycle.server.value.get("at"), "server_started")
    ready_at = _parse_time(lifecycle.ready.value.get("at"), "server_ready")
    api_birth_time = datetime.fromtimestamp(api_birth, timezone.utc)
    gpu_birth_time = datetime.fromtimestamp(gpu_birth, timezone.utc)
    resolution = timedelta(seconds=clock.boot_time_resolution_seconds)
    if not (api_birth_time <= server_at <= api_birth_time + resolution):
        raise CaptureError("API kernel birth window does not contain server_started")
    if gpu_birth_time + resolution < server_at or gpu_birth_time + resolution > observed_at:
        raise CaptureError("GPU kernel birth window is inconsistent with lifecycle/observation time")
    if ready_at > observed_at:
        raise CaptureError("server_ready is later than the ownership observation")

    receipt: Mapping[str, Any] = {
        "api_process": {
            "exe": api_after.exe,
            "parent_pid": api_after.parent_pid,
            "pid": api_after.pid,
            "start_ticks": api_after.start_ticks,
            "state": api_after.state,
        },
        "direct_child_parent_relation_verified": True,
        "engine_environment": TAIL_ENVIRONMENT,
        "expected_engine_exe": engine,
        "gpu_observation": {
            "name": observation["name"],
            "pid": gpu_after.pid,
            "used_mib": observation["used_mib"],
        },
        "gpu_process": {
            "exe": gpu_after.exe,
            "parent_pid": gpu_after.parent_pid,
            "pid": gpu_after.pid,
            "start_ticks": gpu_after.start_ticks,
            "state": gpu_after.state,
        },
        "kernel_clock": {
            "boot_id": clock.boot_id,
            "boot_time_resolution_seconds": clock.boot_time_resolution_seconds,
            "boot_time_unix_seconds": clock.boot_time_unix_seconds,
            "clock_ticks_per_second": clock.clock_ticks_per_second,
            "gpu_process_start_unix_seconds": gpu_birth,
        },
        "kind": "strata_owned_gpu_process_observation",
        "lifecycle_events": [
            {
                "event": "session_started",
                "source_line": lifecycle.start.line,
                "source_raw_line_sha256": lifecycle.start.raw_sha256,
                "source_row_canonical_sha256": lifecycle.start.canonical_sha256,
            },
            {
                "event": "server_started",
                "source_line": lifecycle.server.line,
                "source_raw_line_sha256": lifecycle.server.raw_sha256,
                "source_row_canonical_sha256": lifecycle.server.canonical_sha256,
            },
        ],
        "lifecycle_source": str(lifecycle_path),
        "live_kernel_start_identities_verified": True,
        "model_session_run_id": lifecycle.run_id,
        "observed_at": _format_time(observed_at),
        "observed_during_live_matching_model_session": True,
        "private_config": private_binding.as_dict(),
        "prose_bodies_read": False,
        "runtime_backend": "strata",
        "schema_version": 1,
        "scope": SCOPE,
        "source_config": source_binding.as_dict(),
        "tag": tag,
    }
    _write_exclusive_atomic(output_path, receipt)
    return output_path, receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Capture a metadata-only Strata API/direct-GPU-child ownership receipt."
    )
    parser.add_argument("--tag", required=True)
    parser.add_argument("--source-config", required=True, type=Path)
    parser.add_argument("--expected-source-sha256")
    parser.add_argument(
        "--allow-model-session-reasoning-default-none",
        action="store_true",
        help=(
            "allow model_session's private reasoning_effort=none default only when the source "
            "omits/nulls it and one active step proves matching per-request --thinking"
        ),
    )
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None, *, dependencies: Dependencies = SYSTEM_DEPENDENCIES) -> int:
    args = _parser().parse_args(argv)
    try:
        output, _ = capture(
            tag=args.tag,
            source_config=args.source_config,
            expected_source_sha256=args.expected_source_sha256,
            allow_model_session_reasoning_default_none=(
                args.allow_model_session_reasoning_default_none
            ),
            results_dir=args.results_dir,
            output=args.output,
            dependencies=dependencies,
        )
    except (CaptureError, OSError) as exc:
        print(f"capture_strata_gpu_ownership: {exc}", file=sys.stderr, flush=True)
        return 2
    print(output, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
