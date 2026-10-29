#!/usr/bin/env python3
"""Wait for the accepted Flash main boundary, attest it, and launch one controller.

``--check-only`` performs immutable CPU/file validation only.  It never reads
process state, opens sockets, queries Docker/GPU/API state, writes stage or
acceptance files, or launches a subprocess.  ``--execute`` is the sole path
which may wait for the already-running main owner and, after genuine boundary
evidence, launch the separately reviewed sampling controller exactly once.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
import time
from typing import Any, Callable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runtime import flash_modes_sampling_window as controller


RESULTS = ROOT / "results"
RUNTIME = ROOT / "runtime"
STAGE_SOURCE = Path(__file__).resolve()
STAGE_TESTS = ROOT / "tests/test_flash_modes_sampling_parent_stage.py"
STAGE_PREFLIGHT = RESULTS / "flash-modes-sampling-parent-stage-preflight-v1.json"
STAGE_PREFIX = "flash-modes-sampling-parent-stage-v1"
ACCEPTANCE_PATH = RESULTS / f"{controller.TAG}-parent-acceptance-v1.json"
CODE_REVIEW = RESULTS / "flash-modes-sampling-controller-parent-code-review-v2.json"

MAIN_RUN_ID = "0484c7f6-eb4b-4dad-8ed1-bf6b103bf226"
MAIN_OWNER_PID = 2_369_924
MAIN_OWNER_START_TICKS = "83736456"
MAIN_API_PID = 1_777_545
MAIN_API_START_TICKS = "86871431"
MAIN_GPU_PID = 1_777_616
MAIN_GPU_START_TICKS = "86871553"

MAX_MAIN_WAIT_SECONDS = 86_400.0
WAIT_POLL_SECONDS = 2.0
RESTORATION_ATTEMPTS = 60
RESTORATION_POLL_SECONDS = 2.0
INTERRUPTED_CHILD_WAIT_SECONDS = 240.0
MAX_JSON_BYTES = 32 * 1024 * 1024

PINNED_INPUTS: Mapping[Path, tuple[str, int]] = {
    Path(controller.__file__).resolve(): (
        "a5f0f77beb8f1163178840b154261f6ec0832ea2787b80c5ca4276feab7a3e8b", 103_685,
    ),
    controller.TEST_PATH: (
        "a4b92cc5f2aa75e3d7dccbbb317ac915f9f4553cfa9ea8b133fa05b2aa57b2a5", 39_347,
    ),
    controller.PREFLIGHT_PATH: (
        "ec9194e3adb86948c8166bf5f9fa95741d9aaffc636b617c1f5fbc20158c215f", 13_241,
    ),
    CODE_REVIEW: (
        "eb2ec6a56773c4b7a01940f51a07e9e9242c1bd58bc8ea18cc30716c771ff535", 6_672,
    ),
    controller.PLAN_PATH: (controller.PLAN_SHA256, controller.PLAN_SIZE),
    controller.SOURCE_CONFIG: (controller.SOURCE_CONFIG_SHA256, controller.SOURCE_CONFIG_SIZE),
}


class StageError(RuntimeError):
    """The stager could not prove a required immutable or live invariant."""


class ProcessGone(StageError):
    """A specifically addressed process identity no longer exists."""


class StageInterrupted(BaseException):
    def __init__(self, signum: int) -> None:
        super().__init__(f"signal {signum}")
        self.signum = signum


@dataclass(frozen=True)
class ProcessIdentity:
    pid: int
    parent_pid: int
    start_ticks: str
    state: str
    exe: str

    def scalar_dict(self) -> dict[str, Any]:
        return {
            "pid": self.pid, "parent_pid": self.parent_pid,
            "start_ticks": self.start_ticks, "state": self.state, "exe": self.exe,
        }


@dataclass(frozen=True)
class MainEvidence:
    ready: controller.Binding
    study: controller.Binding
    ready_document: Mapping[str, Any]
    study_document: Mapping[str, Any]
    ready_published_at: datetime
    artifacts: Mapping[str, Mapping[str, Any]]
    rows: Mapping[str, tuple[controller.JsonlRow, ...]]


@dataclass
class SignalLatch:
    received: int | None = None

    def handle(self, signum: int, _frame: Any) -> None:
        self.received = signum

    def check(self) -> None:
        if self.received is not None:
            raise StageInterrupted(self.received)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _regular(path: Path, description: str, maximum: int | None = None) -> os.stat_result:
    try:
        details = path.lstat()
    except OSError as exc:
        raise StageError(f"{description} is missing or inaccessible: {path}") from exc
    if path.is_symlink() or not stat.S_ISREG(details.st_mode):
        raise StageError(f"{description} must be a regular non-symlink file: {path}")
    if maximum is not None and details.st_size > maximum:
        raise StageError(f"{description} exceeds its size bound: {path}")
    return details


def file_binding(path: Path, description: str) -> controller.Binding:
    details = _regular(path, description, MAX_JSON_BYTES)
    return controller.Binding(path.resolve(), sha256_file(path), details.st_size)


def validate_pinned_file(path: Path, expected_sha256: str, expected_size: int) -> controller.Binding:
    binding = file_binding(path, "pinned parent-stage input")
    if binding.sha256 != expected_sha256 or binding.size != expected_size:
        raise StageError(f"pinned parent-stage input changed: {path}")
    return binding


def _expected_review_source_pins() -> dict[str, dict[str, Any]]:
    keys = {
        "controller": Path(controller.__file__).resolve(),
        "tests": controller.TEST_PATH,
        "preflight": controller.PREFLIGHT_PATH,
        "plan": controller.PLAN_PATH,
    }
    return {
        name: validate_pinned_file(path, *PINNED_INPUTS[path]).as_dict()
        for name, path in keys.items()
    }


def validate_code_review_document(
    review: Mapping[str, Any], expected_sources: Mapping[str, Mapping[str, Any]],
) -> None:
    if review.get("schema_version") != 2 or review.get("kind") != "flash_modes_sampling_parent_code_review":
        raise StageError("controller parent code review schema or kind differs")
    if review.get("code_review_accepted") is not True:
        raise StageError("controller parent code review is not affirmative")
    if review.get("actual_new_window_executed") is not False or review.get("execution_acceptance_created") is not False:
        raise StageError("controller code review claims premature execution or acceptance")
    sources = review.get("source_pins")
    if not isinstance(sources, Mapping):
        raise StageError("controller code review source pins are absent")
    for name, expected in expected_sources.items():
        if sources.get(name) != expected:
            raise StageError(f"controller code review {name} binding differs")
    pending = review.get("pending_actual_execution_evidence")
    if not (
        isinstance(pending, Mapping)
        and pending.get("execution_acceptance_exists") is False
        and pending.get("execution_acceptance_path") == str(ACCEPTANCE_PATH)
        and pending.get("main_ready_exists") is False
        and pending.get("main_ready_path") == str(controller.MAIN_READY)
        and pending.get("main_release_exists") is False
        and pending.get("main_release_path") == str(controller.MAIN_RELEASE)
    ):
        raise StageError("controller code review pending-evidence contract differs")


def validate_stage_preflight(document: Mapping[str, Any]) -> None:
    if document.get("schema_version") != 1 or document.get("kind") != "flash_modes_sampling_parent_stage_preflight":
        raise StageError("parent-stage preflight schema or kind differs")
    if document.get("status") != "CPU_validated_not_executed":
        raise StageError("parent-stage preflight status differs")
    pins = document.get("source_pins")
    if not isinstance(pins, Mapping):
        raise StageError("parent-stage preflight source pins are absent")
    expected = {
        str(path): {"sha256": digest, "size": size}
        for path, (digest, size) in PINNED_INPUTS.items()
    }
    for path in (STAGE_SOURCE, STAGE_TESTS):
        binding = file_binding(path, "parent-stage reviewed source")
        expected[str(path)] = {"sha256": binding.sha256, "size": binding.size}
    if pins != expected:
        raise StageError("parent-stage preflight source pins differ from current bytes")


def validate_prerequisites() -> dict[str, Any]:
    bindings = {
        str(path): validate_pinned_file(path, digest, size).as_dict()
        for path, (digest, size) in PINNED_INPUTS.items()
    }
    controller.validate_static_sources()
    review, observed_review = controller.load_json(CODE_REVIEW, "controller parent code review")
    if observed_review.sha256 != PINNED_INPUTS[CODE_REVIEW][0] or observed_review.size != PINNED_INPUTS[CODE_REVIEW][1]:
        raise StageError("controller parent code review changed during validation")
    validate_code_review_document(review, _expected_review_source_pins())
    preflight, _ = controller.load_json(STAGE_PREFLIGHT, "parent-stage preflight")
    validate_stage_preflight(preflight)
    return {
        "validated_at": utc_now(),
        "bindings": bindings,
        "code_review_accepted": True,
        "controller_static_sources_validated": True,
    }


def stage_paths(results: Path = RESULTS) -> dict[str, Path]:
    return {
        "claim": results / f"{STAGE_PREFIX}-claim.json",
        "events": results / f"{STAGE_PREFIX}-events.jsonl",
        "receipt": results / f"{STAGE_PREFIX}-receipt.json",
    }


def candidate_collisions(
    *, acceptance_path: Path = ACCEPTANCE_PATH, allow_acceptance: bool = False,
) -> list[Path]:
    collisions = {
        path.resolve()
        for path in controller.controller_paths().values()
        if os.path.lexists(path)
    }
    if os.path.lexists(acceptance_path) and not allow_acceptance:
        collisions.add(acceptance_path.resolve())
    if RESULTS.is_dir():
        collisions.update(
            path.resolve() for path in RESULTS.glob(f"{controller.TAG}*")
            if not (allow_acceptance and path.resolve() == acceptance_path.resolve())
        )
    return sorted(collisions)


def assert_single_use_unused(paths: Mapping[str, Path], *, acceptance_path: Path = ACCEPTANCE_PATH) -> None:
    used_stage = sorted(path.resolve() for path in paths.values() if os.path.lexists(path))
    if used_stage:
        raise StageError("parent stage is single-use and already has evidence: " + ", ".join(map(str, used_stage)))
    collisions = candidate_collisions(acceptance_path=acceptance_path)
    if collisions:
        raise StageError("candidate acceptance/tag is already used: " + ", ".join(map(str, collisions)))


def _write_exclusive_json(path: Path, value: Mapping[str, Any]) -> controller.Binding:
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise OSError("short immutable parent-stage JSON write")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return controller.Binding(path.resolve(), hashlib.sha256(payload).hexdigest(), len(payload))


class EventWriter:
    def __init__(self, path: Path) -> None:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(path, flags, 0o600)
        os.close(descriptor)
        self.path = path

    def write(self, event: str, **fields: Any) -> None:
        row = {"at": utc_now(), "event": event, **fields}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())


def read_process_identity(pid: int) -> ProcessIdentity:
    try:
        raw = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        close = raw.rfind(")")
        fields = raw[close + 2:].split()
        exe = os.readlink(f"/proc/{pid}/exe")
    except FileNotFoundError as exc:
        raise ProcessGone(f"process identity is gone for PID {pid}") from exc
    except OSError as exc:
        raise StageError(f"process identity is unavailable for PID {pid}") from exc
    if close < 1 or len(fields) < 20:
        raise StageError(f"process stat is malformed for PID {pid}")
    return ProcessIdentity(pid, int(fields[1]), fields[19], fields[0], exe)


def stable_identity_matches(observed: ProcessIdentity, expected: ProcessIdentity) -> bool:
    return bool(
        observed.state not in {"Z", "X", "x"}
        and observed.pid == expected.pid
        and observed.parent_pid == expected.parent_pid
        and observed.start_ticks == expected.start_ticks
        and observed.exe == expected.exe
    )


def validate_main_owner(identity: ProcessIdentity) -> None:
    if (
        identity.pid != MAIN_OWNER_PID
        or identity.start_ticks != MAIN_OWNER_START_TICKS
        or identity.state in {"Z", "X", "x"}
        or not identity.exe
    ):
        raise StageError("current main owner is gone, reused, zombie, or has the wrong start identity")


def wait_for_boundary_ready(
    *, signals: SignalLatch, max_wait_seconds: float = MAX_MAIN_WAIT_SECONDS,
    owner_reader: Callable[[int], ProcessIdentity] = read_process_identity,
    exists: Callable[[Path], bool] = os.path.lexists,
    collision_probe: Callable[[], Sequence[Path]] = candidate_collisions,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> Mapping[str, Any]:
    if not (0 < max_wait_seconds <= MAX_MAIN_WAIT_SECONDS):
        raise StageError("main wait bound must be positive and at most 86400 seconds")
    started = monotonic()
    observations = 0
    while True:
        signals.check()
        if exists(controller.MAIN_RELEASE):
            raise StageError("main boundary release appeared before sampling acceptance")
        collisions = list(collision_probe())
        if collisions:
            raise StageError("candidate outputs appeared before the parent stage launched them")
        owner = owner_reader(MAIN_OWNER_PID)
        validate_main_owner(owner)
        observations += 1
        if exists(controller.MAIN_READY):
            return {
                "ready_observed_at": utc_now(), "owner": owner.scalar_dict(),
                "owner_observation_count": observations,
                "wait_elapsed_seconds": round(monotonic() - started, 3),
            }
        elapsed = monotonic() - started
        if elapsed >= max_wait_seconds:
            raise StageError("bounded wait expired before genuine main boundary READY appeared")
        sleep(min(WAIT_POLL_SECONDS, max_wait_seconds - elapsed))


def _row_pin(row: controller.JsonlRow, expected_case: str) -> dict[str, Any]:
    if row.value.get("case_id") != expected_case:
        raise StageError("main artifact case order differs while constructing acceptance")
    if not controller._terminal(row.value):
        raise StageError(f"main case is not terminal: {expected_case}")
    return {
        "case_id": expected_case,
        "source_line": row.line,
        "source_raw_line_sha256": row.raw_sha256,
        "source_row_canonical_sha256": row.canonical_sha256,
    }


def collect_main_evidence() -> MainEvidence:
    if os.path.lexists(controller.MAIN_RELEASE):
        raise StageError("main release is present before acceptance construction")
    ready_document, ready = controller.load_json(controller.MAIN_READY, "main boundary READY")
    study_document, study = controller.load_json(controller.MAIN_STUDY, "main terminal study")
    artifacts: dict[str, Mapping[str, Any]] = {}
    artifact_bindings: dict[str, controller.Binding] = {}
    artifact_rows: dict[str, tuple[controller.JsonlRow, ...]] = {}
    for name, cases in controller.MAIN_CASES.items():
        rows, binding = controller.read_jsonl(controller.MAIN_ARTIFACT_PATHS[name], f"main {name}")
        pins = [_row_pin(row, case_id) for row, case_id in zip(rows, cases)]
        value = {"source": binding.as_dict(), "rows": pins}
        checked_binding, checked_rows = controller._validate_main_artifact(name, value)
        artifacts[name] = value
        artifact_bindings[name] = checked_binding
        artifact_rows[name] = checked_rows
    controller._validate_study(
        study_document, run_id=MAIN_RUN_ID, binding=study,
        artifact_bindings=artifact_bindings,
    )
    ready_at = controller._validate_ready(ready_document, ready, study=study)
    if study_document.get("session", {}).get("run_id") != MAIN_RUN_ID:
        raise StageError("terminal main study has the wrong model-session run ID")
    return MainEvidence(
        ready, study, ready_document, study_document, ready_at,
        artifacts, artifact_rows,
    )


def stable_start_identity_gone(
    pid: int, start_ticks: str,
    *, reader: Callable[[int], ProcessIdentity] = read_process_identity,
) -> bool:
    try:
        observed = reader(pid)
    except ProcessGone:
        return True
    return observed.start_ticks != start_ticks


def observe_restoration(
    *, live_probe: Callable[[], Mapping[str, Any]] = controller.live_preflight,
    identity_reader: Callable[[int], ProcessIdentity] = read_process_identity,
    attempts: int = RESTORATION_ATTEMPTS,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    errors: list[str] = []
    last: Mapping[str, Any] | None = None
    for attempt in range(1, attempts + 1):
        try:
            live = dict(live_probe())
            api_gone = stable_start_identity_gone(
                MAIN_API_PID, MAIN_API_START_TICKS, reader=identity_reader,
            )
            gpu_gone = stable_start_identity_gone(
                MAIN_GPU_PID, MAIN_GPU_START_TICKS, reader=identity_reader,
            )
            last = {
                "original_running": live.get("original_running") is True,
                "original_paused": live.get("original_paused") is True,
                "original_api_healthy": live.get("original_api_healthy") is True,
                "ports_clear": live.get("ports_clear"),
                "foreign_gpu_process_count": live.get("gpu_compute_process_count"),
                "main_api_identity_gone": api_gone,
                "main_gpu_identity_gone": gpu_gone,
            }
            healthy = bool(
                last["original_running"]
                and not last["original_paused"]
                and last["original_api_healthy"]
                and last["ports_clear"] == [18_080, 18_081]
                and last["foreign_gpu_process_count"] == 0
                and api_gone and gpu_gone
                and not os.path.lexists(controller.MAIN_RELEASE)
            )
            if healthy:
                observed_at = utc_now()
                attestation = {
                    "kind": "flash_modes_sampling_parent_restoration_attestation",
                    "observed_at": observed_at,
                    "original_running": True,
                    "original_paused": False,
                    "original_api_healthy": True,
                    "ports_clear": {"18080": True, "18081": True},
                    "no_foreign_gpu_process": True,
                    "release_absent": True,
                }
                controller._validate_restoration_attestation(attestation)
                return attestation, {
                    "attempt": attempt, "observed_at": observed_at,
                    "main_api_identity_gone": True, "main_gpu_identity_gone": True,
                    "observation_errors": errors,
                }
            errors.append(f"attempt {attempt} unhealthy: {json.dumps(last, sort_keys=True)}")
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
        if attempt < attempts:
            sleep(RESTORATION_POLL_SECONDS)
    reason = errors[-1] if errors else json.dumps(last, sort_keys=True)
    raise StageError(f"post-READY restoration did not become safe: {reason}")


def _accepted_review_bindings() -> dict[str, Mapping[str, Any]]:
    return {
        "plan": validate_pinned_file(controller.PLAN_PATH, *PINNED_INPUTS[controller.PLAN_PATH]).as_dict(),
        "controller": validate_pinned_file(Path(controller.__file__).resolve(), *PINNED_INPUTS[Path(controller.__file__).resolve()]).as_dict(),
        "tests": validate_pinned_file(controller.TEST_PATH, *PINNED_INPUTS[controller.TEST_PATH]).as_dict(),
        "preflight": validate_pinned_file(controller.PREFLIGHT_PATH, *PINNED_INPUTS[controller.PREFLIGHT_PATH]).as_dict(),
    }


def build_acceptance(
    evidence: MainEvidence, restoration_attestation: Mapping[str, Any],
    *, created_at: str | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "kind": "flash_modes_sampling_parent_execution_acceptance",
        "accepted": True,
        "created_at": created_at or utc_now(),
        "review": _accepted_review_bindings(),
        "main": {
            "tag": controller.MAIN_TAG,
            "model": controller.MODEL,
            "model_session_run_id": MAIN_RUN_ID,
            "source_config": validate_pinned_file(
                controller.SOURCE_CONFIG, *PINNED_INPUTS[controller.SOURCE_CONFIG],
            ).as_dict(),
            "ready": evidence.ready.as_dict(),
            "study": evidence.study.as_dict(),
            "artifacts": dict(evidence.artifacts),
            "release_path": str(controller.MAIN_RELEASE),
            "restoration_attestation": dict(restoration_attestation),
        },
    }


def validate_acceptance_shape(document: Mapping[str, Any], evidence: MainEvidence) -> None:
    if set(document) != {"schema_version", "kind", "accepted", "created_at", "review", "main"}:
        raise StageError("assembled acceptance top-level schema differs")
    if document.get("schema_version") != 1 or document.get("kind") != "flash_modes_sampling_parent_execution_acceptance" or document.get("accepted") is not True:
        raise StageError("assembled acceptance is not affirmative schema version 1")
    if document.get("review") != _accepted_review_bindings():
        raise StageError("assembled acceptance review bindings differ")
    main = document.get("main")
    expected_main_keys = {
        "tag", "model", "model_session_run_id", "source_config", "ready", "study",
        "artifacts", "release_path", "restoration_attestation",
    }
    if not isinstance(main, Mapping) or set(main) != expected_main_keys:
        raise StageError("assembled acceptance main schema differs")
    if (
        main.get("tag") != controller.MAIN_TAG
        or main.get("model") != controller.MODEL
        or main.get("model_session_run_id") != MAIN_RUN_ID
        or main.get("source_config") != validate_pinned_file(
            controller.SOURCE_CONFIG, *PINNED_INPUTS[controller.SOURCE_CONFIG],
        ).as_dict()
        or main.get("ready") != evidence.ready.as_dict()
        or main.get("study") != evidence.study.as_dict()
        or main.get("artifacts") != evidence.artifacts
        or main.get("release_path") != str(controller.MAIN_RELEASE)
    ):
        raise StageError("assembled acceptance main evidence identity differs")
    controller._validate_restoration_attestation(main.get("restoration_attestation"))
    accepted_at = controller._parse_time(document.get("created_at"), "assembled acceptance created_at")
    restoration_at = controller._parse_time(
        main["restoration_attestation"].get("observed_at"), "assembled restoration observed_at",
    )
    if restoration_at < evidence.ready_published_at or accepted_at < restoration_at:
        raise StageError("assembled READY/restoration/acceptance chronology differs")
    if os.path.lexists(controller.MAIN_RELEASE):
        raise StageError("main release appeared during acceptance construction")


def prevalidate_acceptance(document: Mapping[str, Any], evidence: MainEvidence) -> None:
    validate_acceptance_shape(document, evidence)
    ready_document, ready_binding = controller.load_json(controller.MAIN_READY, "main READY revalidation")
    study_document, study_binding = controller.load_json(controller.MAIN_STUDY, "main study revalidation")
    if (
        ready_binding != evidence.ready or ready_document != evidence.ready_document
        or study_binding != evidence.study or study_document != evidence.study_document
    ):
        raise StageError("main READY or study changed during acceptance assembly")
    artifact_bindings: dict[str, controller.Binding] = {}
    for name in controller.MAIN_CASES:
        binding, rows = controller._validate_main_artifact(name, document["main"]["artifacts"][name])
        artifact_bindings[name] = binding
        if rows != evidence.rows[name]:
            raise StageError(f"main {name} rows changed during acceptance assembly")
    controller._validate_study(
        evidence.study_document, run_id=MAIN_RUN_ID, binding=evidence.study,
        artifact_bindings=artifact_bindings,
    )
    observed_ready_at = controller._validate_ready(
        evidence.ready_document, evidence.ready, study=evidence.study,
    )
    if observed_ready_at != evidence.ready_published_at:
        raise StageError("main READY timestamp changed during acceptance assembly")


def _controller_receipt_summary(
    returncode: int, receipt: Mapping[str, Any], binding: controller.Binding,
    acceptance: controller.Binding,
) -> dict[str, Any]:
    if (
        receipt.get("schema_version") != 1
        or receipt.get("kind") != "flash_modes_sampling_controller_execution"
        or receipt.get("tag") != controller.TAG
        or receipt.get("acceptance") != acceptance.as_dict()
    ):
        raise StageError("controller receipt identity or acceptance binding differs")
    claims = receipt.get("claims")
    if not isinstance(claims, Mapping) or claims.get("main_release_published") is not False:
        raise StageError("controller receipt does not preserve the main release boundary")
    status = receipt.get("status")
    allowed = {
        "complete", "selected_work_pending_continuation", "infrastructure_failed",
        "cleanup_failed", "interrupted",
    }
    if status not in allowed:
        raise StageError("controller receipt has an unknown terminal status")
    if status in {"complete", "selected_work_pending_continuation"} and returncode != 0:
        raise StageError("successful/pending controller receipt has a nonzero return code")
    if status in {"infrastructure_failed", "cleanup_failed"} and returncode == 0:
        raise StageError("failed controller receipt has a zero return code")
    return {
        "stage_status": status,
        "returncode": returncode,
        "receipt": binding.as_dict(),
        "controller_status": status,
        "pending_selected_work": receipt.get("pending_selected_work", []),
        "primary_problem": receipt.get("primary_problem"),
        "cleanup_problem": receipt.get("cleanup_problem"),
    }


def cooperative_interrupt_child(
    process: subprocess.Popen[Any], expected: ProcessIdentity,
    *, identity_reader: Callable[[int], ProcessIdentity] = read_process_identity,
) -> Mapping[str, Any]:
    if process.poll() is not None:
        return {"signal_sent": False, "reason": "child_already_terminal", "still_live": False}
    try:
        observed = identity_reader(expected.pid)
    except StageError as exc:
        return {"signal_sent": False, "reason": str(exc), "still_live": process.poll() is None}
    if not stable_identity_matches(observed, expected):
        return {
            "signal_sent": False,
            "reason": "refused because controller child PID/PPID/start/executable identity changed",
            "still_live": process.poll() is None,
        }
    process.send_signal(signal.SIGTERM)
    return {"signal_sent": True, "reason": "cooperative SIGTERM to exact owned controller child", "still_live": True}


def launch_controller_once(
    acceptance: controller.Binding, *, signals: SignalLatch,
    events: EventWriter,
    popen_factory: Callable[..., subprocess.Popen[Any]] = subprocess.Popen,
    identity_reader: Callable[[int], ProcessIdentity] = read_process_identity,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> Mapping[str, Any]:
    argv = [
        "python3", "-m", "runtime.flash_modes_sampling_window",
        "--acceptance", str(acceptance.path), "--execute",
    ]
    environment = dict(os.environ)
    environment["STRATA_ROUTE_TAIL_SKIP"] = "0"
    process = popen_factory(
        argv, cwd=str(ROOT), env=environment, stdin=subprocess.DEVNULL,
        start_new_session=True, close_fds=True,
    )
    identity: ProcessIdentity | None = None
    try:
        identity = identity_reader(process.pid)
        if identity.parent_pid != os.getpid() or identity.state in {"Z", "X", "x"}:
            raise StageError("launched controller child identity is not an exact live direct child")
        events.write(
            "controller_launched_once", argv=argv,
            environment_overrides={"STRATA_ROUTE_TAIL_SKIP": "0"},
            controller_source_sha256=PINNED_INPUTS[Path(controller.__file__).resolve()][0],
            process=identity.scalar_dict(),
        )
    except (StageError, OSError) as exc:
        cleanup: Mapping[str, Any]
        if identity is None:
            cleanup = {
                "signal_sent": False,
                "reason": "controller identity could not be proven; no signal sent",
                "still_live": process.poll() is None,
            }
        else:
            cleanup = cooperative_interrupt_child(
                process, identity, identity_reader=identity_reader,
            )
        # The Popen object proves one launch occurred.  Observe any cleanup
        # for a bounded interval and report a still-live child explicitly.
        until = monotonic() + INTERRUPTED_CHILD_WAIT_SECONDS
        while process.poll() is None and monotonic() < until:
            sleep(0.5)
        cleanup = {
            **cleanup,
            "still_live": process.poll() is None,
            "bounded_wait_seconds": INTERRUPTED_CHILD_WAIT_SECONDS,
        }
        return {
            "argv": argv,
            "process": (
                {"pid": process.pid, "identity_captured": False}
                if identity is None else identity.scalar_dict()
            ),
            "returncode": process.poll(),
            "stage_interrupted_signal": None,
            "interrupt_cleanup": cleanup,
            "stage_status": "controller_launch_identity_failed",
            "controller_status": None,
            "receipt": None,
            "pending_selected_work": [],
            "primary_problem": {"error_type": type(exc).__name__, "error": str(exc)},
            "cleanup_problem": None,
        }
    interrupted: StageInterrupted | None = None
    interrupt_cleanup: Mapping[str, Any] | None = None
    while process.poll() is None:
        try:
            signals.check()
        except StageInterrupted as exc:
            interrupted = exc
            interrupt_cleanup = cooperative_interrupt_child(
                process, identity, identity_reader=identity_reader,
            )
            break
        sleep(0.5)
    if interrupted is not None and process.poll() is None:
        until = monotonic() + INTERRUPTED_CHILD_WAIT_SECONDS
        while process.poll() is None and monotonic() < until:
            sleep(0.5)
        if interrupt_cleanup is not None:
            interrupt_cleanup = {
                **interrupt_cleanup,
                "still_live": process.poll() is None,
                "bounded_wait_seconds": INTERRUPTED_CHILD_WAIT_SECONDS,
            }
    returncode = process.poll()
    receipt_path = controller.controller_paths()["receipt"]
    result: dict[str, Any] = {
        "argv": argv,
        "process": identity.scalar_dict(),
        "returncode": returncode,
        "stage_interrupted_signal": None if interrupted is None else interrupted.signum,
        "interrupt_cleanup": interrupt_cleanup,
    }
    if returncode is None:
        result.update({
            "stage_status": "interrupted_controller_still_live",
            "controller_status": None,
            "receipt": None,
            "pending_selected_work": [],
            "primary_problem": {"error": "owned controller remained live after bounded cooperative cleanup wait"},
            "cleanup_problem": None,
        })
        return result
    if not receipt_path.is_file() or receipt_path.is_symlink():
        result.update({
            "stage_status": "controller_failed_without_receipt",
            "controller_status": None,
            "receipt": None,
            "pending_selected_work": [],
            "primary_problem": {"error": "controller exited without its immutable receipt"},
            "cleanup_problem": None,
        })
        return result
    receipt, binding = controller.load_json(receipt_path, "controller execution receipt")
    result.update(_controller_receipt_summary(returncode, receipt, binding, acceptance))
    if interrupted is not None:
        result["stage_status"] = "interrupted"
    return result


def _main_evidence_summary(evidence: MainEvidence) -> dict[str, Any]:
    return {
        "ready": evidence.ready.as_dict(),
        "study": evidence.study.as_dict(),
        "model_session_run_id": MAIN_RUN_ID,
        "terminal_case_counts": {name: len(rows) for name, rows in evidence.rows.items()},
        "artifacts": {
            name: {
                "source": value["source"],
                "rows": value["rows"],
            }
            for name, value in evidence.artifacts.items()
        },
    }


def execute_stage(
    *, signals: SignalLatch,
    paths: Mapping[str, Path] | None = None,
    acceptance_path: Path = ACCEPTANCE_PATH,
    max_wait_seconds: float = MAX_MAIN_WAIT_SECONDS,
    prerequisite_fn: Callable[[], Mapping[str, Any]] | None = None,
    unused_fn: Callable[[Mapping[str, Path]], None] | None = None,
    wait_fn: Callable[..., Mapping[str, Any]] | None = None,
    evidence_fn: Callable[[], MainEvidence] | None = None,
    restoration_fn: Callable[[], tuple[Mapping[str, Any], Mapping[str, Any]]] | None = None,
    prevalidate_fn: Callable[[Mapping[str, Any], MainEvidence], None] | None = None,
    acceptance_validate_fn: Callable[[Path], Any] | None = None,
    launch_fn: Callable[..., Mapping[str, Any]] | None = None,
    stage_identity_reader: Callable[[int], ProcessIdentity] = read_process_identity,
    monotonic: Callable[[], float] = time.monotonic,
    collision_fn: Callable[[Path, bool], Sequence[Path]] | None = None,
) -> tuple[Mapping[str, Any], int]:
    stage_started = monotonic()
    selected_paths = stage_paths() if paths is None else dict(paths)
    prerequisite_fn = validate_prerequisites if prerequisite_fn is None else prerequisite_fn
    unused_fn = (
        (lambda value: assert_single_use_unused(value, acceptance_path=acceptance_path))
        if unused_fn is None else unused_fn
    )
    wait_fn = wait_for_boundary_ready if wait_fn is None else wait_fn
    evidence_fn = collect_main_evidence if evidence_fn is None else evidence_fn
    restoration_fn = observe_restoration if restoration_fn is None else restoration_fn
    prevalidate_fn = prevalidate_acceptance if prevalidate_fn is None else prevalidate_fn
    acceptance_validate_fn = controller.validate_acceptance if acceptance_validate_fn is None else acceptance_validate_fn
    launch_fn = launch_controller_once if launch_fn is None else launch_fn
    collision_fn = (
        (lambda acceptance, allowed: candidate_collisions(
            acceptance_path=acceptance, allow_acceptance=allowed,
        ))
        if collision_fn is None else collision_fn
    )

    prerequisites = prerequisite_fn()
    unused_fn(selected_paths)
    stage_identity = stage_identity_reader(os.getpid())
    if stage_identity.pid != os.getpid() or stage_identity.state in {"Z", "X", "x"}:
        raise StageError("parent-stage process identity is not an exact live self identity")
    claim = {
        "schema_version": 1,
        "kind": "flash_modes_sampling_parent_stage_claim",
        "claimed_at": utc_now(),
        "single_use": True,
        "main_tag": controller.MAIN_TAG,
        "main_run_id": MAIN_RUN_ID,
        "candidate_tag": controller.TAG,
        "acceptance_path": str(acceptance_path),
        "stage_process": stage_identity.scalar_dict(),
        "controller_source": validate_pinned_file(
            Path(controller.__file__).resolve(), *PINNED_INPUTS[Path(controller.__file__).resolve()],
        ).as_dict(),
        "stage_preflight": file_binding(STAGE_PREFLIGHT, "parent-stage preflight").as_dict(),
    }
    claim_binding = _write_exclusive_json(selected_paths["claim"], claim)
    events = EventWriter(selected_paths["events"])
    events.write("parent_stage_started", claim=claim_binding.as_dict())

    wait_observation: Mapping[str, Any] | None = None
    evidence: MainEvidence | None = None
    restoration_observation: Mapping[str, Any] | None = None
    acceptance_binding: controller.Binding | None = None
    controller_outcome: Mapping[str, Any] | None = None
    primary_problem: Mapping[str, Any] | None = None
    status = "stage_failed"
    code = 1
    try:
        signals.check()
        remaining_wait = max_wait_seconds - (monotonic() - stage_started)
        if remaining_wait <= 0:
            raise StageError("parent-stage preflight consumed the bounded main wait allowance")
        wait_observation = wait_fn(signals=signals, max_wait_seconds=remaining_wait)
        signals.check()
        events.write("genuine_main_ready_observed", **dict(wait_observation))
        evidence = evidence_fn()
        signals.check()
        events.write(
            "main_evidence_validated", ready=evidence.ready.as_dict(),
            study=evidence.study.as_dict(), terminal_case_count=sum(len(rows) for rows in evidence.rows.values()),
        )
        restoration_attestation, restoration_observation = restoration_fn()
        signals.check()
        events.write("post_ready_restoration_validated", **dict(restoration_observation))
        acceptance_document = build_acceptance(evidence, restoration_attestation)
        prevalidate_fn(acceptance_document, evidence)
        signals.check()
        if os.path.lexists(controller.MAIN_RELEASE):
            raise StageError("main release appeared immediately before acceptance write")
        if collision_fn(acceptance_path, False):
            raise StageError("candidate outputs appeared before exclusive acceptance write")
        if os.path.lexists(acceptance_path):
            raise StageError("candidate acceptance appeared before exclusive write")
        acceptance_binding = _write_exclusive_json(acceptance_path, acceptance_document)
        accepted = acceptance_validate_fn(acceptance_path)
        accepted_source = getattr(accepted, "source", None)
        if accepted_source is not None and accepted_source != acceptance_binding:
            raise StageError("full controller acceptance validation returned a different binding")
        events.write("execution_acceptance_created_and_validated", acceptance=acceptance_binding.as_dict())
        signals.check()
        if collision_fn(acceptance_path, True):
            raise StageError("candidate runtime outputs appeared before reviewed controller launch")
        controller_outcome = launch_fn(acceptance_binding, signals=signals, events=events)
        status = str(controller_outcome.get("stage_status"))
        interrupted_signal = controller_outcome.get("stage_interrupted_signal")
        if isinstance(interrupted_signal, int) and not isinstance(interrupted_signal, bool):
            code = 128 + interrupted_signal
        else:
            code = 0 if status in {"complete", "selected_work_pending_continuation"} else 1
    except StageInterrupted as exc:
        primary_problem = {"error_type": type(exc).__name__, "error": str(exc), "signal": exc.signum}
        status = "interrupted_before_controller_launch"
        code = 128 + exc.signum
    except (StageError, controller.ControllerError, OSError, subprocess.SubprocessError) as exc:
        primary_problem = {"error_type": type(exc).__name__, "error": str(exc)}
        status = "stage_failed"
        code = 1
    if os.path.lexists(controller.MAIN_RELEASE):
        primary_problem = primary_problem or {
            "error_type": "StageError", "error": "main release appeared during parent stage",
        }
        status, code = "stage_failed", 1
    receipt = {
        "schema_version": 1,
        "kind": "flash_modes_sampling_parent_stage_execution",
        "status": status,
        "created_at": utc_now(),
        "claim": claim_binding.as_dict(),
        "prerequisites": prerequisites,
        "wait_observation": wait_observation,
        "main_evidence": None if evidence is None else _main_evidence_summary(evidence),
        "restoration_observation": restoration_observation,
        "acceptance": None if acceptance_binding is None else acceptance_binding.as_dict(),
        "controller": controller_outcome,
        "primary_problem": primary_problem,
        "claims": {
            "controller_launched_at_most_once": True,
            "candidate_lifecycle_reimplemented_by_stager": False,
            "main_release_created": False,
            "main_owner_or_foreign_process_signaled": False,
            "docker_or_hardware_mutation": False,
            "baseline_20b_executed": False,
            "generated_answer_prose_or_private_reasoning_in_receipt": False,
        },
    }
    _write_exclusive_json(selected_paths["receipt"], receipt)
    return receipt, code


def offline_summary() -> Mapping[str, Any]:
    prerequisites = validate_prerequisites()
    ready_present = os.path.lexists(controller.MAIN_READY)
    acceptance_present = os.path.lexists(ACCEPTANCE_PATH)
    release_present = os.path.lexists(controller.MAIN_RELEASE)
    return {
        "accepted_controller_prerequisites": True,
        "mode": "offline CPU/static validation only; no process, socket, Docker, GPU, API, model, or lifecycle action",
        "status": (
            "waiting_for_actual_main_evidence"
            if not ready_present else "actual_evidence_present_but_requires_execute_mode_live_validation"
        ),
        "main_tag": controller.MAIN_TAG,
        "main_run_id": MAIN_RUN_ID,
        "main_ready_file_present": ready_present,
        "main_release_file_present": release_present,
        "execution_acceptance_present": acceptance_present,
        "main_evidence_required_before_acceptance": True,
        "controller_launch_performed": False,
        "prerequisites": prerequisites,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--execute", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.check_only:
        try:
            print(json.dumps(offline_summary(), sort_keys=True))
            return 0
        except (StageError, controller.ControllerError, OSError) as exc:
            print(f"flash_modes_sampling_parent_stage: {exc}", file=sys.stderr, flush=True)
            return 2
    signals = SignalLatch()
    previous = {signum: signal.getsignal(signum) for signum in (signal.SIGINT, signal.SIGTERM)}
    try:
        for signum in previous:
            signal.signal(signum, signals.handle)
        _, code = execute_stage(signals=signals)
        return code
    except (StageError, controller.ControllerError, OSError, subprocess.SubprocessError) as exc:
        print(f"flash_modes_sampling_parent_stage: {exc}", file=sys.stderr, flush=True)
        return 2
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


if __name__ == "__main__":
    raise SystemExit(main())
