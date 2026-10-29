#!/usr/bin/env python3
"""Publish and wait on an opt-in, receipt-bound per-model handoff gate."""

from __future__ import annotations

import hashlib
import json
import os
import re
import socket
import stat
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

try:
    from . import gpu_window
except ImportError:  # pragma: no cover - script-style imports
    import gpu_window


MAX_JSON_BYTES = 1024 * 1024
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class BoundaryGateError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}.{uuid.uuid4().hex}")
    descriptor = -1
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        view = memoryview(payload)
        offset = 0
        while offset < len(view):
            written = os.write(descriptor, view[offset:])
            if written <= 0:
                raise OSError("short boundary ready write")
            offset += written
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise BoundaryGateError(f"boundary ready file already exists: {path}") from exc
        _fsync_directory(path.parent)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def _read_json(path: Path, description: str) -> tuple[dict[str, Any], str]:
    try:
        details = path.lstat()
        if path.is_symlink() or not stat.S_ISREG(details.st_mode):
            raise BoundaryGateError(f"{description} must be a regular non-symlink file")
        if details.st_size > MAX_JSON_BYTES:
            raise BoundaryGateError(f"{description} exceeds its size limit")
        payload = path.read_bytes()
        value = json.loads(payload)
    except (OSError, json.JSONDecodeError) as exc:
        raise BoundaryGateError(f"{description} is unreadable or malformed: {path}") from exc
    if not isinstance(value, dict):
        raise BoundaryGateError(f"{description} must be a JSON object")
    return value, hashlib.sha256(payload).hexdigest()


def _verify_study(path: Path, expected_sha256: str) -> None:
    try:
        details = path.lstat()
        if path.is_symlink() or not stat.S_ISREG(details.st_mode):
            raise BoundaryGateError("model boundary study receipt must be a regular non-symlink file")
        actual_sha256 = sha256_file(path)
    except OSError as exc:
        raise BoundaryGateError(f"model boundary study receipt is unreadable: {path}") from exc
    if actual_sha256 != expected_sha256:
        raise BoundaryGateError("model boundary study receipt SHA-256 differs from its identity")


def _port_clear(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.3):
            return False
    except ConnectionRefusedError:
        return True
    except OSError:
        return False


def safe_boundary() -> dict[str, Any]:
    try:
        state = gpu_window.original_state()
    except Exception as exc:
        raise BoundaryGateError(f"original Docker state probe failed: {exc}") from exc
    evidence = {
        "checked_at": utc_now(),
        "original_container": gpu_window.ORIGINAL_CONTAINER,
        "original_running": state.get("Running") is True,
        "original_paused": state.get("Paused") is True,
        "ports_clear": {str(port): _port_clear(port) for port in (18080, 18081)},
    }
    if (
        evidence["original_running"] is not True
        or evidence["original_paused"] is not False
        or not all(evidence["ports_clear"].values())
    ):
        raise BoundaryGateError("original Docker is not restored or owned ports are not clear")
    return evidence


def _boundary_is_safe(evidence: Mapping[str, Any]) -> bool:
    ports = evidence.get("ports_clear")
    return bool(
        evidence.get("original_running") is True
        and evidence.get("original_paused") is False
        and isinstance(ports, dict)
        and ports.get("18080") is True
        and ports.get("18081") is True
    )


def publish_and_wait(
    directory: Path,
    run_prefix: str,
    model_key: str,
    study: Mapping[str, Any],
    timeout: float,
    *,
    boundary_probe: Callable[[], dict[str, Any]] = safe_boundary,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    poll_seconds: float = 1.0,
) -> dict[str, Any]:
    """Publish an immutable ready receipt and accept only its matching safe release."""
    if (
        not isinstance(study.get("path"), str)
        or not isinstance(study.get("sha256"), str)
        or SHA256_RE.fullmatch(str(study["sha256"])) is None
    ):
        raise BoundaryGateError("model boundary study identity is incomplete")
    _verify_study(Path(str(study["path"])), str(study["sha256"]))
    stem = f"{run_prefix}-{model_key}"
    ready_path = directory / f"{stem}-ready.json"
    release_path = directory / f"{stem}-release.json"
    if os.path.lexists(release_path):
        raise BoundaryGateError(f"boundary release predates ready publication: {release_path}")
    ready = {
        "schema_version": 1,
        "kind": "model_boundary_ready",
        "run_prefix": run_prefix,
        "model_key": model_key,
        "published_at": utc_now(),
        "study": dict(study),
    }
    _write_exclusive(ready_path, ready)
    ready_sha = sha256_file(ready_path)
    deadline = monotonic() + timeout
    last_boundary_error = "release file has not appeared"
    while monotonic() < deadline:
        if os.path.lexists(release_path):
            release, release_sha = _read_json(release_path, "model boundary release")
            if (
                release.get("schema_version") != 1
                or release.get("kind") != "model_boundary_release"
                or release.get("run_prefix") != run_prefix
                or release.get("model_key") != model_key
                or release.get("ready_sha256") != ready_sha
            ):
                raise BoundaryGateError("model boundary release does not match the published ready receipt")
            try:
                boundary = boundary_probe()
            except BoundaryGateError as exc:
                last_boundary_error = str(exc)
            else:
                if not _boundary_is_safe(boundary):
                    last_boundary_error = (
                        "original Docker is not restored or owned ports are not clear"
                    )
                    sleep(min(poll_seconds, max(0.0, deadline - monotonic())))
                    continue
                return {
                    "ready": {"path": str(ready_path), "sha256": ready_sha},
                    "release": {"path": str(release_path), "sha256": release_sha},
                    "boundary": boundary,
                }
        sleep(min(poll_seconds, max(0.0, deadline - monotonic())))
    raise BoundaryGateError(f"model boundary release was not safely accepted before deadline: {last_boundary_error}")
