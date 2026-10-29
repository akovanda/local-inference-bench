#!/usr/bin/env python3
"""Run the pinned Bonsai and Flash-Coder specialist follow-up cohort safely."""

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

if __package__:
    from . import gpu_window, model_boundary_gate
else:
    import gpu_window
    import model_boundary_gate


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIRST_WAVE_RECEIPT = RESULTS / "specialists-first-wave-cohort.json"
COHORT_RECEIPT = RESULTS / "specialists-followups-cohort.json"
COHORT_EVENTS = RESULTS / "specialists-followups-cohort-events.jsonl"
BONSAI_HELPER = ROOT / "runtime" / "prepare_bonsai2.py"
FLASH_HELPER = ROOT / "runtime" / "prepare_flash_coder.py"
STUDY_HELPER = ROOT / "runtime" / "specialist_study_trial.py"
STANDARD_PLAN = ROOT / "runtime" / "specialists-standard-8k-plan.json"
ROLE_PLAN = ROOT / "runtime" / "specialists-role-screen-plan.json"
BONSAI_RECEIPT = RESULTS / "bonsai2-runtime-artifact.json"
FLASH_RECEIPT = RESULTS / "flash-next-coder-artifact.json"
FLASH_CONFIG = ROOT / "runtime" / "flash-next-coder-reference-8k.json"
MAX_WAIT_SECONDS = 7_200.0
MAX_WINDOW_SECONDS = 86_400.0
MAX_OUTPUT_CAP = 65_536
MAX_REQUEST_TIMEOUT = 21_600.0
MAX_CONTEXT = 131_072
RUN_PREFIX_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
PROCESS_TERM_SECONDS = 600.0
EVENT_MAX_BYTES = 64 * 1024 * 1024
BASE_CONTEXT = 8_192
FIT_CTX_THRESHOLD = 32_768
BONSAI_TEMPLATE_SHA256 = "c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041"
DEFERRED_GEMMA_PROFILE = "gemma4-style12-q5_k_m"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class FollowupError(RuntimeError):
    pass


@dataclass(frozen=True)
class Profile:
    key: str
    label: str
    backend: str
    model_path: Path
    model_size: int
    model_sha256: str
    config: Path
    plan: Path
    tag: str
    quantization: str
    prose_thinking: str = "none"
    environment: tuple[tuple[str, str], ...] = ()
    base_config: Path | None = None

    @property
    def port(self) -> int:
        return 18080 if self.backend == "strata" else 18081


BONSAI_DIR = Path("/external/storage/moe-bench/models/bonsai2-27b")
FLASH_DIR = Path("/external/storage/moe-bench/models/flash-next-coder-IQ1_M")
PROFILES = (
    Profile(
        "bonsai2-pq2_0",
        "Ternary-Bonsai-2-27B PQ2_0",
        "llama",
        BONSAI_DIR / "Ternary-Bonsai-2-27B-PQ2_0.gguf",
        7_206_168_928,
        "3907dc1658db1f78a9826bf8d5bcb8dc65db0d466388937af57f2294fae62ec1",
        ROOT / "runtime" / "bonsai2-pq2_0-fitoff-8k.json",
        STANDARD_PLAN,
        "specialists-bonsai2-pq2_0-strict-8k",
        "PQ2_0",
    ),
    Profile(
        "bonsai2-ptq1_0",
        "Ternary-Bonsai-2-27B PTQ1_0",
        "llama",
        BONSAI_DIR / "Ternary-Bonsai-2-27B-PTQ1_0.gguf",
        5_946_648_928,
        "53107f530aa52eb00912263ab1ee29bd199261c87cd7b4ad4ca1318c1fe33ee3",
        ROOT / "runtime" / "bonsai2-ptq1_0-fitoff-8k.json",
        ROLE_PLAN,
        "specialists-bonsai2-ptq1_0-strict-8k",
        "PTQ1_0",
    ),
    Profile(
        "flash-next-coder-iq1_m",
        "Qwen3.8 Flash-Next Coder IQ1_M",
        "strata",
        FLASH_DIR / "Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf",
        29_608_446_496,
        "bfa3644db4adfff072fc946c20994f2909d2cf89879d83f80a4c811d4016c211",
        FLASH_CONFIG,
        ROLE_PLAN,
        "specialists-flash-next-coder-iq1_m-reference-8k",
        "IQ1_M",
        environment=(("STRATA_ROUTE_TAIL_SKIP", "0"),),
    ),
)


@dataclass(frozen=True)
class Paths:
    results: Path = RESULTS
    first_wave_receipt: Path = FIRST_WAVE_RECEIPT
    cohort_receipt: Path = COHORT_RECEIPT
    cohort_events: Path = COHORT_EVENTS
    bonsai_helper: Path = BONSAI_HELPER
    flash_helper: Path = FLASH_HELPER
    study_helper: Path = STUDY_HELPER
    bonsai_receipt: Path = BONSAI_RECEIPT
    flash_receipt: Path = FLASH_RECEIPT


DEFAULT_PATHS = Paths()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path, description: str) -> dict[str, Any]:
    try:
        details = path.lstat()
        if not stat.S_ISREG(details.st_mode) or path.is_symlink():
            raise FollowupError(f"{description} must be a regular non-symlink file")
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FollowupError(f"{description} is unreadable or invalid: {path}") from exc
    if not isinstance(value, dict):
        raise FollowupError(f"{description} must be a JSON object")
    return value


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
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
                raise OSError("short immutable JSON write")
            offset += written
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise FollowupError(f"immutable output already exists: {path}") from exc
        _fsync_directory(path.parent)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def publish_json(path: Path, value: Mapping[str, Any], description: str) -> None:
    if os.path.lexists(path):
        if _read_json(path, description) != value:
            raise FollowupError(f"refusing to overwrite different {description}: {path}")
        return
    write_json_exclusive(path, value)


class EventLog:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        self.path = path
        self._descriptor = os.open(path, flags, 0o600)

    def write(self, event: str, **fields: Any) -> None:
        row = {"at": utc_now(), "event": event, **fields}
        payload = (json.dumps(row, separators=(",", ":")) + "\n").encode("utf-8")
        view = memoryview(payload)
        offset = 0
        while offset < len(view):
            written = os.write(self._descriptor, view[offset:])
            if written <= 0:
                raise OSError("short cohort event write")
            offset += written
        os.fsync(self._descriptor)
        print(json.dumps(row, separators=(",", ":")), flush=True)

    def close(self) -> None:
        if self._descriptor >= 0:
            os.close(self._descriptor)
            self._descriptor = -1


def _safe_study(receipt: Mapping[str, Any], expected_port: int) -> bool:
    cleanup = receipt.get("cleanup")
    restoration = receipt.get("restoration")
    session_cleanup = cleanup.get("model_session_cleanup") if isinstance(cleanup, dict) else None
    return bool(
        receipt.get("status") in ("complete", "completed_with_evaluation_failures")
        and isinstance(cleanup, dict)
        and cleanup.get("outer_exited") is True
        and cleanup.get("orchestrator_sigkill_sent") is not True
        and isinstance(session_cleanup, dict)
        and session_cleanup.get("stopped") is True
        and session_cleanup.get("kill_sent") is False
        and isinstance(restoration, dict)
        and restoration.get("original_running") is True
        and restoration.get("original_paused") is False
        and restoration.get("owned_loopback_port_clear") is True
        and restoration.get("owned_loopback_port") == expected_port
    )


def port_clear(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.2):
            return False
    except ConnectionRefusedError:
        return True
    except OSError:
        return False


def process_start_ticks(pid: int) -> int | None:
    try:
        payload = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise FollowupError(f"could not inspect deferred first-wave parent PID {pid}") from exc
    closing = payload.rfind(")")
    if closing < 0:
        raise FollowupError(f"malformed /proc stat for deferred first-wave parent PID {pid}")
    fields = payload[closing + 1 :].split()
    try:
        return int(fields[19])
    except (IndexError, ValueError) as exc:
        raise FollowupError(
            f"malformed /proc stat for deferred first-wave parent PID {pid}"
        ) from exc


def validate_user_deferral(
    path: Path,
    expected_sha256: str,
    first_wave_path: Path,
    *,
    port_probe: Callable[[int], bool] = port_clear,
    docker_state_probe: Callable[[], Mapping[str, Any]] = gpu_window.original_state,
    pid_probe: Callable[[int], int | None] = process_start_ticks,
) -> dict[str, Any]:
    if SHA256_RE.fullmatch(expected_sha256) is None:
        raise FollowupError("user-deferral SHA-256 must be 64 lowercase hexadecimal characters")
    deferral = _read_json(path, "user-requested model deferral receipt")
    actual_deferral_sha = sha256_file(path)
    if actual_deferral_sha != expected_sha256:
        raise FollowupError("user-requested model deferral receipt SHA-256 differs")
    if (
        deferral.get("schema_version") != 1
        or deferral.get("kind") != "user_requested_model_deferral"
        or deferral.get("deferred_models") != [DEFERRED_GEMMA_PROFILE]
        or not isinstance(deferral.get("user_instruction"), str)
        or not deferral["user_instruction"].strip()
    ):
        raise FollowupError("user-requested model deferral receipt has invalid identity")

    first_wave = _read_json(first_wave_path, "deferred first-wave cohort receipt")
    first_wave_sha = sha256_file(first_wave_path)
    cohort_identity = deferral.get("cohort_receipt")
    if not isinstance(cohort_identity, dict) or cohort_identity != {
        "path": str(first_wave_path),
        "sha256": first_wave_sha,
    }:
        raise FollowupError("user deferral does not bind the selected first-wave cohort receipt")
    models = first_wave.get("models")
    if (
        first_wave.get("schema_version") != 1
        or first_wave.get("status") != "infrastructure_failed"
        or not isinstance(models, list)
        or len(models) != 1
        or not isinstance(models[0], dict)
        or models[0].get("key") != DEFERRED_GEMMA_PROFILE
        or models[0].get("strict_status") != "interrupted"
    ):
        raise FollowupError("deferred first-wave cohort is not the interrupted Gemma cohort")
    model = models[0]
    study_path_value = model.get("strict_receipt")
    study_sha_value = model.get("strict_receipt_sha256")
    study_identity = deferral.get("study_receipt")
    if (
        not isinstance(study_path_value, str)
        or not isinstance(study_sha_value, str)
        or SHA256_RE.fullmatch(study_sha_value) is None
        or not isinstance(study_identity, dict)
        or study_identity != {"path": study_path_value, "sha256": study_sha_value}
    ):
        raise FollowupError("user deferral does not bind the cohort's interrupted Gemma study")
    study_path = Path(study_path_value)
    study = _read_json(study_path, "interrupted Gemma study receipt")
    study_sha = sha256_file(study_path)
    cleanup = study.get("cleanup")
    session_cleanup = cleanup.get("model_session_cleanup") if isinstance(cleanup, dict) else None
    restoration = study.get("restoration")
    controls = study.get("controls")
    problem = study.get("problem")
    if (
        study_sha != study_sha_value
        or study.get("schema_version") != 1
        or study.get("runtime_backend") != "llama"
        or not isinstance(controls, dict)
        or controls.get("profile") != DEFERRED_GEMMA_PROFILE
        or study.get("status") != "interrupted"
        or not isinstance(problem, dict)
        or problem.get("error_type") != "StudyInterrupted"
        or problem.get("signal") != signal.SIGTERM
        or not isinstance(cleanup, dict)
        or cleanup.get("outer_exited") is not True
        or cleanup.get("orchestrator_sigkill_sent") is not False
        or not isinstance(session_cleanup, dict)
        or session_cleanup.get("stopped") is not True
        or session_cleanup.get("kill_sent") is not False
        or not isinstance(restoration, dict)
        or restoration.get("original_running") is not True
        or restoration.get("original_paused") is not False
        or restoration.get("owned_loopback_port") != 18081
        or restoration.get("owned_loopback_port_clear") is not True
    ):
        raise FollowupError("interrupted Gemma study does not prove safe signal-15 cleanup")

    owned_parent = deferral.get("owned_parent")
    if not isinstance(owned_parent, dict):
        raise FollowupError("user deferral has no owned first-wave parent identity")
    parent_pid = owned_parent.get("pid")
    parent_start_ticks = owned_parent.get("start_ticks")
    if (
        isinstance(parent_pid, bool)
        or not isinstance(parent_pid, int)
        or parent_pid <= 0
        or isinstance(parent_start_ticks, bool)
        or not isinstance(parent_start_ticks, int)
        or parent_start_ticks <= 0
    ):
        raise FollowupError("user deferral has an invalid owned first-wave parent identity")
    observed_start_ticks = pid_probe(parent_pid)
    if observed_start_ticks == parent_start_ticks:
        raise FollowupError("deferred first-wave parent identity is still live")

    try:
        docker_state = docker_state_probe()
    except Exception as exc:
        raise FollowupError(f"original Docker state probe failed: {exc}") from exc
    if not isinstance(docker_state, Mapping):
        raise FollowupError("original Docker state probe returned invalid state")
    clear = {str(port): port_probe(port) for port in (18080, 18081)}
    if (
        docker_state.get("Running") is not True
        or docker_state.get("Paused") is not False
        or not all(clear.values())
    ):
        raise FollowupError("original Docker is not restored or owned ports are not clear")

    return {
        "gate_kind": "user_requested_model_deferral",
        "gemma_primary_complete": False,
        "deferred_models": [DEFERRED_GEMMA_PROFILE],
        "cohort_receipt": str(first_wave_path),
        "cohort_receipt_sha256": first_wave_sha,
        "source_cohort_status": first_wave.get("status"),
        "selected_deferral_receipt": str(path),
        "selected_deferral_receipt_sha256": actual_deferral_sha,
        "interrupted_study_receipt": str(study_path),
        "interrupted_study_receipt_sha256": study_sha,
        "interrupted_study_proof": {
            "status": study.get("status"),
            "error_type": problem.get("error_type"),
            "signal": problem.get("signal"),
            "cleanup_safe": True,
            "recorded_restoration_safe": True,
        },
        "owned_parent": {
            "pid": parent_pid,
            "start_ticks": parent_start_ticks,
            "observed_start_ticks": observed_start_ticks,
            "old_identity_exited": True,
        },
        "current_restoration": {
            "original_container": gpu_window.ORIGINAL_CONTAINER,
            "original_running": True,
            "original_paused": False,
            "ports_clear": clear,
        },
    }


def _selected_last_study(first_wave: Mapping[str, Any]) -> tuple[Path, str]:
    models = first_wave.get("models")
    if not isinstance(models, list) or not models or not isinstance(models[-1], dict):
        raise FollowupError("first-wave receipt has no last study")
    last = models[-1]
    selected = last.get("selected_profile")
    if selected == "strict_all_layers_fit_off":
        path, digest = last.get("strict_receipt"), last.get("strict_receipt_sha256")
    elif selected == "auto_fallback_after_strict_cuda_oom":
        path, digest = last.get("fallback_receipt"), last.get("fallback_receipt_sha256")
    else:
        raise FollowupError("first-wave last study has no recognized selected profile")
    if not isinstance(path, str) or not isinstance(digest, str) or len(digest) != 64:
        raise FollowupError("first-wave last study receipt identity is incomplete")
    return Path(path), digest


def validate_first_wave(
    receipt: Mapping[str, Any],
    *,
    port_probe: Callable[[int], bool] = port_clear,
) -> dict[str, Any]:
    if receipt.get("status") not in ("complete", "completed_with_evaluation_failures"):
        raise FollowupError("first-wave cohort is not evaluation-complete")
    study_path, expected_sha = _selected_last_study(receipt)
    study = _read_json(study_path, "first-wave last study receipt")
    actual_sha = sha256_file(study_path)
    if actual_sha != expected_sha or not _safe_study(study, 18081):
        raise FollowupError("first-wave last study does not prove safe completion and restoration")
    clear = {str(port): port_probe(port) for port in (18080, 18081)}
    if not all(clear.values()):
        raise FollowupError("one or both owned loopback ports are not clear")
    return {
        "last_study_receipt": str(study_path),
        "last_study_sha256": actual_sha,
        "last_study_status": study.get("status"),
        "ports_clear": clear,
    }


def wait_for_first_wave(
    path: Path,
    timeout: float,
    *,
    port_probe: Callable[[int], bool] = port_clear,
) -> tuple[dict[str, Any], dict[str, Any]]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if os.path.lexists(path):
            receipt = _read_json(path, "first-wave cohort receipt")
            try:
                return receipt, validate_first_wave(receipt, port_probe=port_probe)
            except FollowupError as exc:
                if "ports are not clear" not in str(exc):
                    raise
        time.sleep(1.0)
    raise FollowupError("completed first-wave cohort and clear owned ports did not appear before deadline")


def wait_for_atomic_file(path: Path, size: int, timeout: float) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if os.path.lexists(path):
            try:
                details = path.lstat()
            except OSError as exc:
                raise FollowupError(f"published artifact became inaccessible: {path}") from exc
            if not stat.S_ISREG(details.st_mode) or path.is_symlink():
                raise FollowupError(f"published artifact must be a regular non-symlink file: {path}")
            if details.st_size != size:
                raise FollowupError(
                    f"published artifact has size {details.st_size}; expected {size}: {path}"
                )
            return {
                "path": str(path),
                "size": size,
                "device": details.st_dev,
                "inode": details.st_ino,
                "mtime_ns": details.st_mtime_ns,
                "ctime_ns": details.st_ctime_ns,
            }
        time.sleep(1.0)
    raise FollowupError(f"final artifact publication did not appear before deadline: {path}")


def _terminate_owned(process: subprocess.Popen[Any]) -> bool:
    if process.poll() is not None:
        return True
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return True
    deadline = time.monotonic() + PROCESS_TERM_SECONDS
    while process.poll() is None and time.monotonic() < deadline:
        time.sleep(0.2)
    return process.poll() is not None


def run_command(
    argv: Sequence[str],
    timeout: float,
    kind: str,
    *,
    environment: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    overrides = dict(environment or {})
    started_at = utc_now()
    started = time.monotonic()
    try:
        process = subprocess.Popen(
            list(argv),
            stdin=subprocess.DEVNULL,
            start_new_session=True,
            close_fds=True,
            env={**os.environ, **overrides},
        )
    except OSError as exc:
        raise FollowupError(f"{kind} could not be launched") from exc
    try:
        while process.poll() is None:
            if time.monotonic() - started >= timeout:
                if not _terminate_owned(process):
                    raise FollowupError(
                        f"{kind} exceeded its timeout and ignored SIGTERM; SIGKILL was not sent"
                    )
                returncode = 124
                break
            time.sleep(0.25)
        else:
            returncode = process.returncode
    except KeyboardInterrupt:
        if not _terminate_owned(process):
            raise FollowupError(f"interrupted {kind} ignored SIGTERM; SIGKILL was not sent")
        raise
    return {
        "kind": kind,
        "argv": list(argv),
        "environment": overrides,
        "started_at": started_at,
        "finished_at": utc_now(),
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "returncode": returncode,
    }


def _verified_shard(receipt: Mapping[str, Any], profile: Profile) -> Mapping[str, Any]:
    shards = receipt.get("shards")
    if not isinstance(shards, list):
        raise FollowupError("preparation receipt has no shard list")
    matches = [
        row for row in shards
        if isinstance(row, dict)
        and row.get("path") == str(profile.model_path)
        and row.get("size") == profile.model_size
        and row.get("sha256") == profile.model_sha256
        and row.get("verified") is True
    ]
    if len(matches) != 1:
        raise FollowupError(f"preparation receipt does not verify {profile.key}")
    return matches[0]


def _validate_bonsai_template(document: Mapping[str, Any], profile: Profile, receipt: Path) -> None:
    expected_args = [
        "--gpu-layers", "all", "--fit", "off", "--parallel", "1",
        "--jinja", "--no-webui", "--no-slots",
    ]
    manifest = document.get("artifact_manifest")
    resolved_manifest = (
        (profile.config.parent / manifest).resolve() if isinstance(manifest, str) else None
    )
    if (
        document.get("schema_version") != 1
        or document.get("model") != str(profile.model_path)
        or document.get("context") != BASE_CONTEXT
        or document.get("host") != "127.0.0.1"
        or document.get("port") != 18081
        or document.get("args") != expected_args
        or resolved_manifest != receipt.resolve()
    ):
        raise FollowupError(f"Bonsai template for {profile.key} violates the strict 8K contract")


def validate_bonsai_receipt(
    receipt: Mapping[str, Any], profiles: Sequence[Profile], receipt_path: Path
) -> dict[str, Mapping[str, Any]]:
    verdict = receipt.get("verdict")
    runtime = receipt.get("runtime")
    cuda = runtime.get("cuda_backend") if isinstance(runtime, dict) else None
    fatbinary = cuda.get("fatbinary") if isinstance(cuda, dict) else None
    prebuilt = runtime.get("prebuilt_candidate") if isinstance(runtime, dict) else None
    prebuilt_fatbinary = prebuilt.get("fatbinary") if isinstance(prebuilt, dict) else None
    architectures = fatbinary.get("architectures") if isinstance(fatbinary, dict) else None
    if (
        not isinstance(verdict, dict)
        or verdict.get("artifact_complete") is not True
        or verdict.get("static_tesla_t4_compatibility") is not True
        or not isinstance(runtime, dict)
        or runtime.get("selected_artifact") != "source_build_cuda13_sm75"
        or not isinstance(fatbinary, dict)
        or fatbinary.get("tesla_t4_sm75_present") is not True
        or not isinstance(architectures, list)
        or 75 not in architectures
        or not isinstance(prebuilt_fatbinary, dict)
        or prebuilt_fatbinary.get("tesla_t4_sm75_present") is not False
    ):
        raise FollowupError("Bonsai receipt does not prove the selected SM75 source build")
    templates = receipt.get("trial_config_templates")
    capabilities = receipt.get("template_capabilities")
    if not isinstance(templates, dict):
        raise FollowupError("Bonsai receipt has no trial config templates")
    if (
        not isinstance(capabilities, dict)
        or capabilities.get("chat_template_sha256") != BONSAI_TEMPLATE_SHA256
        or capabilities.get("enable_thinking_false_supported") is not True
        or capabilities.get("enable_thinking_false_behavior") != "empty_think_block"
    ):
        raise FollowupError(
            "Bonsai receipt does not attest the pinned template's thinking-off behavior"
        )
    result: dict[str, Mapping[str, Any]] = {}
    for profile in profiles:
        _verified_shard(receipt, profile)
        document = templates.get(profile.quantization)
        if not isinstance(document, dict):
            raise FollowupError(f"Bonsai receipt has no {profile.quantization} config template")
        _validate_bonsai_template(document, profile, receipt_path)
        result[profile.key] = document
    return result


def ensure_bonsai_prepared(
    profiles: Sequence[Profile], paths: Paths, timeout: float, context: int = BASE_CONTEXT
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    commands: list[dict[str, Any]] = []
    if not os.path.lexists(paths.bonsai_receipt):
        command = [
            sys.executable, str(paths.bonsai_helper), "--quantization", "PQ2_0",
            "--context", str(BASE_CONTEXT), "--require-model",
        ]
        record = run_command(command, timeout, "prepare_bonsai2")
        commands.append(record)
        if record["returncode"] != 0:
            raise FollowupError("Bonsai preparation failed")
    receipt = _read_json(paths.bonsai_receipt, "Bonsai runtime artifact receipt")
    templates = validate_bonsai_receipt(receipt, profiles, paths.bonsai_receipt)
    configs: dict[str, Any] = {}
    for profile in profiles:
        document = json.loads(json.dumps(templates[profile.key]))
        document["context"] = context
        args = document.get("args")
        if not isinstance(args, list) or any(not isinstance(item, str) for item in args):
            raise FollowupError(f"Bonsai template args for {profile.key} are invalid")
        if context > FIT_CTX_THRESHOLD:
            positions = [index for index, value in enumerate(args) if value == "--fit-ctx"]
            inline = [value for value in args if value.startswith("--fit-ctx=")]
            if inline or len(positions) > 1 or (positions and positions[0] + 1 >= len(args)):
                raise FollowupError(f"Bonsai template for {profile.key} has ambiguous --fit-ctx")
            if positions:
                args[positions[0] + 1] = str(context)
            else:
                args.extend(("--fit-ctx", str(context)))
        publish_json(profile.config, document, f"{profile.key} strict config")
        configs[profile.key] = {
            "path": str(profile.config),
            "sha256": sha256_file(profile.config),
            "source_template_document_sha256": hashlib.sha256(
                (json.dumps(templates[profile.key], sort_keys=True, separators=(",", ":")) + "\n").encode()
            ).hexdigest(),
            "context": context,
        }
    return commands, {
        "receipt": str(paths.bonsai_receipt),
        "receipt_sha256": sha256_file(paths.bonsai_receipt),
        "configs": configs,
        "runtime_gate": "source_build_cuda13_sm75",
        "template_capabilities": {
            "chat_template_sha256": BONSAI_TEMPLATE_SHA256,
            "enable_thinking_false_supported": True,
            "enable_thinking_false_behavior": "empty_think_block",
        },
    }


def validate_flash_receipt(
    receipt: Mapping[str, Any], profile: Profile, paths: Paths
) -> dict[str, Any]:
    verdict = receipt.get("verdict")
    mtp = receipt.get("mtp")
    runtime = receipt.get("runtime")
    expert = receipt.get("expert_profile")
    if (
        not isinstance(verdict, dict)
        or verdict.get("artifact_complete") is not True
        or verdict.get("full_sha256_verified") is not True
        or verdict.get("shared_ple_identity_bound") is not True
        or verdict.get("pack_complete") is not True
        or not isinstance(mtp, dict)
        or mtp.get("enabled") is not False
        or mtp.get("companion_available") is not False
        or not isinstance(expert, dict)
        or expert.get("verified") is not True
        or expert.get("sha256") != "a514661124d5305effe9c1bcecff7dc069af398b2b19ff9d11628a2cad0a8915"
        or not isinstance(runtime, dict)
        or runtime.get("config") != str(profile.base_config or profile.config)
    ):
        raise FollowupError("Flash-Coder receipt does not prove the pinned no-MTP/profile/pack contract")
    _verified_shard(receipt, profile)
    base_config = profile.base_config or profile.config
    config = _read_json(base_config, "Flash-Coder runtime config")
    args = config.get("args")
    reference = config.get("reference_environment")
    if (
        config.get("model_name") != "qwen3.8-flash-next-coder-iq1_m"
        or config.get("host") != "127.0.0.1"
        or config.get("port") != 18080
        or not isinstance(args, list)
        or not any(
            args[index:index + 2] == ["--suffix-draft", "0"]
            for index in range(max(0, len(args) - 1))
        )
        or not isinstance(reference, dict)
        or reference.get("STRATA_ROUTE_TAIL_SKIP") != "0"
        or reference.get("caller_must_set_and_verify_in_runtime_events") is not True
        or runtime.get("config_sha256") != sha256_file(base_config)
    ):
        raise FollowupError("Flash-Coder config violates the reference/no-MTP contract")
    return {
        "receipt": str(paths.flash_receipt),
        "receipt_sha256": sha256_file(paths.flash_receipt),
        "base_config": str(base_config),
        "base_config_sha256": sha256_file(base_config),
        "expert_profile_sha256": expert["sha256"],
        "mtp_enabled": False,
        "reference_environment": dict(profile.environment),
    }


def ensure_flash_prepared(
    profile: Profile, paths: Paths, timeout: float, context: int = BASE_CONTEXT
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    commands: list[dict[str, Any]] = []
    if not os.path.lexists(paths.flash_receipt):
        record = run_command(
            [sys.executable, str(paths.flash_helper), "--prepare"],
            timeout,
            "prepare_flash_coder",
        )
        commands.append(record)
        if record["returncode"] != 0:
            raise FollowupError("Flash-Coder preparation failed")
    receipt = _read_json(paths.flash_receipt, "Flash-Coder artifact receipt")
    evidence = validate_flash_receipt(receipt, profile, paths)
    base_path = profile.base_config or profile.config
    base = _read_json(base_path, "Flash-Coder base runtime config")
    document = (
        base
        if base_path.resolve() == profile.config.resolve() and context == BASE_CONTEXT
        else flash_context_config_document(base, context, profile.tag)
    )
    publish_json(profile.config, document, "derived Flash-Coder runtime config")
    evidence.update({
        "config": str(profile.config),
        "config_sha256": sha256_file(profile.config),
        "context": context,
    })
    return commands, evidence


def flash_context_config_document(
    source: Mapping[str, Any], context: int, tag: str
) -> dict[str, Any]:
    document = json.loads(json.dumps(source))
    args = document.get("args")
    if not isinstance(args, list) or any(not isinstance(item, str) for item in args):
        raise FollowupError("Flash-Coder base config args must be a string list")
    positions = [index for index, value in enumerate(args) if value == "--max-context"]
    if len(positions) != 1 or positions[0] + 1 >= len(args):
        raise FollowupError("Flash-Coder base config must have exactly one --max-context")
    try:
        source_context = int(args[positions[0] + 1])
    except ValueError as exc:
        raise FollowupError("Flash-Coder base --max-context must be an integer") from exc
    if source_context != BASE_CONTEXT:
        raise FollowupError("Flash-Coder prepared base config is not the pinned 8K config")
    args[positions[0] + 1] = str(context)
    document["log"] = str(RESULTS / f"{tag}-engine.log")
    return document


def study_command(profile: Profile, helper: Path, args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        str(helper),
        "--runtime-backend", profile.backend,
        "--config", str(profile.config),
        "--plan", str(profile.plan),
        "--tag", profile.tag,
        "--profile", profile.key,
        "--code-thinking", "medium",
        "--prose-thinking", profile.prose_thinking,
        "--wait-for-idle", format(args.wait_for_idle, ".15g"),
        "--max-seconds", format(args.max_window_seconds, ".15g"),
        "--code-output-cap", str(args.code_output_cap),
        "--prose-output-cap", str(args.prose_output_cap),
        "--code-timeout", format(args.code_timeout, ".15g"),
        "--prose-timeout", format(args.prose_timeout, ".15g"),
    ]
    sampler_plan = getattr(args, "sampler_plan", None)
    if sampler_plan is not None:
        command.extend(("--sampler-plan", str(sampler_plan)))
    if getattr(args, "standard_only", False):
        command.append("--standard-only")
    return command


def _study_summary(receipt: Mapping[str, Any]) -> dict[str, Any]:
    manifest = receipt.get("specialist_manifest")
    return {
        "tag": receipt.get("tag"),
        "status": receipt.get("status"),
        "wall_seconds": receipt.get("wall_seconds"),
        "source_config": receipt.get("source_config"),
        "plan": receipt.get("plan"),
        "specialist_manifest_sha256": (
            manifest.get("sha256") if isinstance(manifest, dict) else None
        ),
        "session": receipt.get("session"),
        "cleanup": receipt.get("cleanup"),
        "restoration": receipt.get("restoration"),
    }


def validate_study_identity(
    receipt: Mapping[str, Any], profile: Profile, args: argparse.Namespace
) -> None:
    source = receipt.get("source_config")
    plan = receipt.get("plan")
    controls = receipt.get("controls")
    session = receipt.get("session")
    sampler_plan = getattr(args, "sampler_plan", None)
    sampler_receipt = receipt.get("sampler_plan")
    sampler_identity_ok = sampler_plan is None or (
        isinstance(sampler_receipt, dict)
        and sampler_receipt.get("path") == str(sampler_plan)
        and sampler_receipt.get("sha256") == sha256_file(sampler_plan)
    )
    if (
        receipt.get("tag") != profile.tag
        or receipt.get("runtime_backend") != profile.backend
        or not isinstance(source, dict)
        or source.get("path") != str(profile.config)
        or source.get("sha256") != sha256_file(profile.config)
        or not isinstance(plan, dict)
        or plan.get("path") != str(profile.plan)
        or plan.get("sha256") != sha256_file(profile.plan)
        or not isinstance(controls, dict)
        or controls.get("code_thinking") != "medium"
        or controls.get("prose_thinking") != profile.prose_thinking
        or not isinstance(controls.get("code"), dict)
        or controls["code"].get("output_cap") != args.code_output_cap
        or controls["code"].get("timeout") != args.code_timeout
        or not isinstance(controls.get("prose"), dict)
        or controls["prose"].get("output_cap") != args.prose_output_cap
        or controls["prose"].get("timeout") != args.prose_timeout
        or bool(controls.get("standard_only", False))
        != bool(getattr(args, "standard_only", False))
        or not isinstance(session, dict)
        or session.get("context") != args.context
        or not sampler_identity_ok
    ):
        raise FollowupError(f"study receipt identity or request controls differ for {profile.key}")


def validate_flash_environment(tag: str, results: Path) -> dict[str, Any]:
    path = results / f"{tag}-runtime-events.jsonl"
    try:
        details = path.lstat()
        if (
            not stat.S_ISREG(details.st_mode)
            or path.is_symlink()
            or details.st_size > EVENT_MAX_BYTES
        ):
            raise FollowupError("Flash runtime event log is unsafe or over its size bound")
        payload = path.read_bytes()
    except OSError as exc:
        raise FollowupError("Flash runtime event log is unreadable") from exc
    started: list[dict[str, Any]] = []
    try:
        for raw in payload.splitlines():
            row = json.loads(raw)
            if isinstance(row, dict) and row.get("event") == "session_started":
                started.append(row)
    except json.JSONDecodeError as exc:
        raise FollowupError("Flash runtime event log contains invalid JSON") from exc
    if len(started) != 1:
        raise FollowupError("Flash runtime event log has ambiguous session_started evidence")
    environment = started[0].get("engine_environment")
    if (
        started[0].get("runtime_backend") != "strata"
        or not isinstance(environment, dict)
        or environment.get("STRATA_ROUTE_TAIL_SKIP") != "0"
    ):
        raise FollowupError("Flash server did not record STRATA_ROUTE_TAIL_SKIP=0")
    return {
        "path": str(path),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "engine_environment": {"STRATA_ROUTE_TAIL_SKIP": "0"},
    }


def run_study(
    profile: Profile,
    args: argparse.Namespace,
    paths: Paths,
    *,
    port_probe: Callable[[int], bool] = port_clear,
) -> tuple[dict[str, Any], dict[str, Any]]:
    record = run_command(
        study_command(profile, paths.study_helper, args),
        args.wait_for_idle + args.max_window_seconds + PROCESS_TERM_SECONDS,
        f"study_{profile.key}",
        environment=dict(profile.environment),
    )
    receipt_path = paths.results / f"{profile.tag}-study-trial.json"
    receipt = _read_json(receipt_path, f"study receipt for {profile.tag}")
    validate_study_identity(receipt, profile, args)
    expected_returncode = 1 if receipt.get("status") == "completed_with_evaluation_failures" else 0
    if record["returncode"] != expected_returncode or not _safe_study(receipt, profile.port):
        raise FollowupError(f"study for {profile.key} did not complete with safe restoration")
    both_clear = {str(port): port_probe(port) for port in (18080, 18081)}
    if not all(both_clear.values()):
        raise FollowupError(f"owned loopback ports are not both clear after {profile.key}")
    environment_evidence = (
        validate_flash_environment(profile.tag, paths.results)
        if profile.backend == "strata"
        else None
    )
    return record, {
        "path": str(receipt_path),
        "sha256": sha256_file(receipt_path),
        "summary": _study_summary(receipt),
        "ports_clear_after": both_clear,
        "environment_evidence": environment_evidence,
    }


def wait_at_model_boundary(
    args: argparse.Namespace, profile: Profile, study: Mapping[str, Any]
) -> dict[str, Any] | None:
    directory = getattr(args, "model_boundary_gates", None)
    if directory is None:
        return None
    try:
        return model_boundary_gate.publish_and_wait(
            directory,
            getattr(args, "run_prefix", "specialists"),
            profile.key,
            {"path": study.get("path"), "sha256": study.get("sha256")},
            args.model_boundary_wait_seconds,
        )
    except model_boundary_gate.BoundaryGateError as exc:
        raise FollowupError(f"model boundary gate failed for {profile.key}: {exc}") from exc


def _validate_static_inputs(profiles: Sequence[Profile], paths: Paths) -> dict[str, str]:
    required = [paths.bonsai_helper, paths.flash_helper, paths.study_helper]
    required.extend({profile.plan for profile in profiles})
    hashes: dict[str, str] = {}
    for path in required:
        if not path.is_file() or path.is_symlink():
            raise FollowupError(f"required helper or plan is missing: {path}")
        hashes[str(path)] = sha256_file(path)
    return hashes


def execute_cohort(
    args: argparse.Namespace,
    *,
    profiles: Sequence[Profile] = PROFILES,
    paths: Paths = DEFAULT_PATHS,
    port_probe: Callable[[int], bool] = port_clear,
    docker_state_probe: Callable[[], Mapping[str, Any]] = gpu_window.original_state,
    pid_probe: Callable[[int], int | None] = process_start_ticks,
) -> tuple[dict[str, Any], int]:
    if len(profiles) != 3 or [profile.backend for profile in profiles] != ["llama", "llama", "strata"]:
        raise FollowupError("follow-up profile sequence must be Bonsai PQ, Bonsai PTQ, Flash-Coder")
    if os.path.lexists(paths.cohort_receipt):
        raise FollowupError("immutable follow-up cohort receipt already exists")
    events = EventLog(paths.cohort_events)
    started = time.monotonic()
    started_at = utc_now()
    records: list[dict[str, Any]] = []
    commands: list[dict[str, Any]] = []
    publications: dict[str, dict[str, Any]] = {}
    problem: dict[str, str] | None = None
    first_wave_gate: dict[str, Any] | None = None
    source_hashes: dict[str, str] = {}
    code = 1
    try:
        source_hashes = _validate_static_inputs(profiles, paths)
        sampler_plan = getattr(args, "sampler_plan", None)
        if sampler_plan is not None:
            if not sampler_plan.is_file() or sampler_plan.is_symlink():
                raise FollowupError(f"sampler plan is missing or unsafe: {sampler_plan}")
            source_hashes[str(sampler_plan)] = sha256_file(sampler_plan)
        if getattr(args, "model_boundary_gates", None) is not None:
            gate_source = Path(model_boundary_gate.__file__).resolve()
            if not gate_source.is_file() or gate_source.is_symlink():
                raise FollowupError(
                    f"model boundary gate helper is missing or unsafe: {gate_source}"
                )
            source_hashes[str(gate_source)] = sha256_file(gate_source)
        user_deferral_receipt = getattr(args, "user_deferral_receipt", None)
        user_deferral_sha256 = getattr(args, "user_deferral_sha256", None)
        if (user_deferral_receipt is None) != (user_deferral_sha256 is None):
            raise FollowupError(
                "user-deferral receipt and SHA-256 must be provided together"
            )
        if user_deferral_receipt is None:
            events.write("first_wave_wait_started", path=str(paths.first_wave_receipt))
            _, first_wave_gate = wait_for_first_wave(
                paths.first_wave_receipt, args.first_wave_wait_seconds, port_probe=port_probe
            )
            first_wave_gate["cohort_receipt"] = str(paths.first_wave_receipt)
            first_wave_gate["cohort_receipt_sha256"] = sha256_file(paths.first_wave_receipt)
        else:
            events.write(
                "user_deferral_gate_started",
                path=str(user_deferral_receipt),
                cohort_receipt=str(paths.first_wave_receipt),
            )
            first_wave_gate = validate_user_deferral(
                user_deferral_receipt,
                user_deferral_sha256,
                paths.first_wave_receipt,
                port_probe=port_probe,
                docker_state_probe=docker_state_probe,
                pid_probe=pid_probe,
            )
        events.write("first_wave_gate_passed", **first_wave_gate)

        bonsai_profiles = profiles[:2]
        for profile in bonsai_profiles:
            events.write("artifact_wait_started", model=profile.key, path=str(profile.model_path))
            publication = wait_for_atomic_file(
                profile.model_path, profile.model_size, args.download_wait_seconds
            )
            publications[profile.key] = publication
            events.write("artifact_published", model=profile.key, **publication)
        prep_commands, bonsai_preparation = ensure_bonsai_prepared(
            bonsai_profiles, paths, args.prepare_timeout_seconds, args.context
        )
        commands.extend(prep_commands)
        for command in prep_commands:
            events.write("stage_complete", **command)
        events.write("bonsai_prepared", **bonsai_preparation)

        for index, profile in enumerate(profiles):
            if index == 2:
                events.write("artifact_wait_started", model=profile.key, path=str(profile.model_path))
                publication = wait_for_atomic_file(
                    profile.model_path, profile.model_size, args.download_wait_seconds
                )
                publications[profile.key] = publication
                events.write("artifact_published", model=profile.key, **publication)
                prep_commands, flash_preparation = ensure_flash_prepared(
                    profile, paths, args.prepare_timeout_seconds, args.context
                )
                commands.extend(prep_commands)
                for command in prep_commands:
                    events.write("stage_complete", **command)
                events.write("flash_coder_prepared", **flash_preparation)
                preparation = flash_preparation
            else:
                preparation = {
                    "receipt": bonsai_preparation["receipt"],
                    "receipt_sha256": bonsai_preparation["receipt_sha256"],
                    "config": bonsai_preparation["configs"][profile.key],
                    "runtime_gate": bonsai_preparation["runtime_gate"],
                }
            record = {
                "key": profile.key,
                "label": profile.label,
                "backend": profile.backend,
                "tag": profile.tag,
                "plan": str(profile.plan),
                "plan_sha256": source_hashes[str(profile.plan)],
                "controls": {
                    "context": args.context,
                    "code_thinking": "medium",
                    "prose_thinking": profile.prose_thinking,
                    "code_output_cap": args.code_output_cap,
                    "prose_output_cap": args.prose_output_cap,
                    "code_timeout": args.code_timeout,
                    "prose_timeout": args.prose_timeout,
                    "environment": dict(profile.environment),
                    **({"standard_only": True} if getattr(args, "standard_only", False) else {}),
                },
                "publication": publications[profile.key],
                "preparation": preparation,
                "status": "pending",
            }
            records.append(record)
            events.write(
                "study_started",
                model=profile.key,
                tag=profile.tag,
                backend=profile.backend,
                environment=dict(profile.environment),
            )
            process, study = run_study(profile, args, paths, port_probe=port_probe)
            commands.append(process)
            record["command"] = process
            record["study"] = study
            record["status"] = study["summary"]["status"]
            events.write(
                "study_complete",
                model=profile.key,
                status=record["status"],
                returncode=process["returncode"],
                elapsed_seconds=process["elapsed_seconds"],
                ports_clear=study["ports_clear_after"],
            )
            boundary = wait_at_model_boundary(args, profile, study)
            if boundary is not None:
                record["boundary_gate"] = boundary
                events.write("model_boundary_released", model=profile.key, **boundary)
        code = 1 if any(
            row["status"] == "completed_with_evaluation_failures" for row in records
        ) else 0
    except KeyboardInterrupt as exc:
        problem = {"error_type": type(exc).__name__, "error": "interrupted"}
        code = 130
        events.write("cohort_interrupted")
    except (FollowupError, OSError, subprocess.SubprocessError) as exc:
        problem = {"error_type": type(exc).__name__, "error": str(exc)}
        code = 2
        events.write("cohort_failed", **problem)
    finally:
        events.close()

    complete = len(records) == len(profiles) and all(
        row.get("status") in ("complete", "completed_with_evaluation_failures") for row in records
    )
    status = (
        "completed_with_evaluation_failures"
        if complete and any(row["status"] == "completed_with_evaluation_failures" for row in records)
        else "complete" if complete else "infrastructure_failed"
    )
    receipt = {
        "schema_version": 1,
        "immutable": True,
        "status": status,
        "started_at": started_at,
        "finished_at": utc_now(),
        "wall_seconds": round(time.monotonic() - started, 3),
        "first_wave_gate": first_wave_gate,
        "source_hashes": source_hashes,
        "controls": {
            "sequence": [profile.key for profile in profiles],
            "download_wait_seconds_per_artifact": args.download_wait_seconds,
            "max_window_seconds": args.max_window_seconds,
            "context": args.context,
            "code_output_cap": args.code_output_cap,
            "prose_output_cap": args.prose_output_cap,
            "code_timeout": args.code_timeout,
            "prose_timeout": args.prose_timeout,
            "bonsai_gpu_layers": "all",
            "bonsai_fit": "off",
            "flash_route_tail_skip": "0",
            "flash_mtp": False,
            **({"standard_only": True} if getattr(args, "standard_only", False) else {}),
        },
        "commands": commands,
        "models": records,
        "events": str(paths.cohort_events),
        "problem": problem,
    }
    return receipt, code


def bounded_seconds(maximum: float):
    def parse(value: str) -> float:
        parsed = float(value)
        if not math.isfinite(parsed) or not 0 < parsed <= maximum:
            raise argparse.ArgumentTypeError(f"must be finite and from >0 through {maximum:g}")
        return parsed
    return parse


def bounded_int(maximum: int):
    def parse(value: str) -> int:
        parsed = int(value)
        if not 1 <= parsed <= maximum:
            raise argparse.ArgumentTypeError(f"must be from 1 through {maximum}")
        return parsed
    return parse


def run_prefix(value: str) -> str:
    if RUN_PREFIX_RE.fullmatch(value) is None:
        raise argparse.ArgumentTypeError("must be a safe 1-64 character run prefix")
    return value


def sha256_argument(value: str) -> str:
    normalized = value.lower()
    if SHA256_RE.fullmatch(normalized) is None:
        raise argparse.ArgumentTypeError("must be exactly 64 hexadecimal characters")
    return normalized


class FollowupArgumentParser(argparse.ArgumentParser):
    def parse_args(
        self, args: Sequence[str] | None = None, namespace: argparse.Namespace | None = None
    ) -> argparse.Namespace:
        parsed = super().parse_args(args, namespace)
        if (parsed.user_deferral_receipt is None) != (parsed.user_deferral_sha256 is None):
            self.error(
                "--user-deferral-receipt and --user-deferral-sha256 must be provided together"
            )
        return parsed


def _context_label(context: int) -> str:
    return f"{context // 1024}k" if context % 1024 == 0 else f"ctx{context}"


def profiles_for_args(args: argparse.Namespace) -> tuple[Profile, ...]:
    label = _context_label(args.context)
    plans = {
        "bonsai2-pq2_0": args.standard_plan,
        "bonsai2-ptq1_0": args.role_plan,
        "flash-next-coder-iq1_m": args.role_plan,
    }
    profiles: list[Profile] = []
    for original in PROFILES:
        default_identity = args.run_prefix == "specialists" and args.context == BASE_CONTEXT
        config = (
            original.config
            if default_identity
            else ROOT / "runtime" / f"{args.run_prefix}-{original.key}-{label}.json"
        )
        profiles.append(
            Profile(
                key=original.key,
                label=original.label,
                backend=original.backend,
                model_path=original.model_path,
                model_size=original.model_size,
                model_sha256=original.model_sha256,
                config=config,
                plan=plans[original.key],
                tag=(
                    original.tag
                    if default_identity
                    else f"{args.run_prefix}-{original.key}-strict-{label}"
                ),
                quantization=original.quantization,
                prose_thinking=original.prose_thinking,
                environment=original.environment,
                base_config=(FLASH_CONFIG if original.backend == "strata" else None),
            )
        )
    return tuple(profiles)


def build_parser() -> argparse.ArgumentParser:
    parser = FollowupArgumentParser(description=__doc__)
    parser.add_argument("--run-prefix", type=run_prefix, default="specialists")
    parser.add_argument("--first-wave-receipt", type=Path, default=FIRST_WAVE_RECEIPT)
    parser.add_argument("--user-deferral-receipt", type=Path)
    parser.add_argument("--user-deferral-sha256", type=sha256_argument)
    parser.add_argument("--context", type=bounded_int(MAX_CONTEXT), default=BASE_CONTEXT)
    parser.add_argument("--standard-plan", type=Path, default=STANDARD_PLAN)
    parser.add_argument("--role-plan", type=Path, default=ROLE_PLAN)
    parser.add_argument("--sampler-plan", type=Path)
    parser.add_argument("--standard-only", action="store_true")
    parser.add_argument("--first-wave-wait-seconds", type=bounded_seconds(MAX_WAIT_SECONDS), default=7200.0)
    parser.add_argument("--download-wait-seconds", type=bounded_seconds(MAX_WAIT_SECONDS), default=7200.0)
    parser.add_argument("--prepare-timeout-seconds", type=bounded_seconds(MAX_WAIT_SECONDS), default=7200.0)
    parser.add_argument("--max-window-seconds", type=bounded_seconds(MAX_WINDOW_SECONDS), default=7200.0)
    parser.add_argument("--code-output-cap", type=bounded_int(MAX_OUTPUT_CAP), default=4096)
    parser.add_argument("--prose-output-cap", type=bounded_int(MAX_OUTPUT_CAP), default=2048)
    parser.add_argument("--code-timeout", type=bounded_seconds(MAX_REQUEST_TIMEOUT), default=900.0)
    parser.add_argument("--prose-timeout", type=bounded_seconds(MAX_REQUEST_TIMEOUT), default=900.0)
    parser.add_argument("--wait-for-idle", type=bounded_seconds(900.0), default=900.0)
    parser.add_argument("--model-boundary-gates", type=Path)
    parser.add_argument(
        "--model-boundary-wait-seconds",
        type=bounded_seconds(MAX_WINDOW_SECONDS),
        default=MAX_WINDOW_SECONDS,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.first_wave_receipt = args.first_wave_receipt.expanduser().resolve()
    if args.user_deferral_receipt is not None:
        args.user_deferral_receipt = args.user_deferral_receipt.expanduser().resolve()
    args.standard_plan = args.standard_plan.expanduser().resolve()
    args.role_plan = args.role_plan.expanduser().resolve()
    if args.sampler_plan is not None:
        args.sampler_plan = args.sampler_plan.expanduser().resolve()
    if args.model_boundary_gates is not None:
        args.model_boundary_gates = args.model_boundary_gates.expanduser().resolve()
    profiles = profiles_for_args(args)
    cohort_receipt = RESULTS / f"{args.run_prefix}-followups-cohort.json"
    cohort_events = RESULTS / f"{args.run_prefix}-followups-cohort-events.jsonl"
    paths = Paths(
        first_wave_receipt=args.first_wave_receipt,
        cohort_receipt=cohort_receipt,
        cohort_events=cohort_events,
    )
    if os.path.lexists(cohort_receipt) or os.path.lexists(cohort_events):
        print("follow-up cohort outputs already exist; refusing overwrite", file=sys.stderr)
        return 2
    try:
        receipt, code = execute_cohort(args, profiles=profiles, paths=paths)
        write_json_exclusive(cohort_receipt, receipt)
    except (FollowupError, OSError) as exc:
        print(f"specialist follow-ups: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    return code


if __name__ == "__main__":
    raise SystemExit(main())
