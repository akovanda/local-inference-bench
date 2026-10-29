#!/usr/bin/env python3
"""Fail-closed continuation for the adjudicated Bonsai2 DSpark pilot.

Run from the repository root with ``python3 -m runtime.bonsai_dspark_continuation``.
Check-only mode creates no files or processes.  Execution requires a separate
parent acceptance receipt which binds this source and its CPU preflight.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from runtime import bonsai_dspark_pilot as p


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
TAG = "specialists-bonsai-ptq-dspark-cont-v1"
ORIGINAL_TAG = "specialists-bonsai-ptq-dspark-v1"
ORIGINAL_RUN_ID = "9498a5b3-4149-4664-ad35-785b998dde68"
ORIGINAL_DEADLINE_TEXT = "2026-10-21T05:49:11.799450+00:00"
ORIGINAL_DEADLINE = datetime.fromisoformat(ORIGINAL_DEADLINE_TEXT)
ALREADY_SPENT = 1
CONTINUATION_MAX = 3
TOTAL_MAX = 4
RESTORATION_RESERVE_SECONDS = 1_800.0
SETUP_STEP_ID = "matched_quick_medium_65536"
QUICK_IDS = ("quick_generation", "quick_json_extract")

ELIGIBILITY = RESULTS / "bonsai2-dspark-pilot-eligibility-v1.json"
ELIGIBILITY_SHA256 = "7772b6fd96989898b7073b8dc78e91f1dcd767401bdb0a7d3b11755b445f783a"
ELIGIBILITY_SIZE = 9046
ADJUDICATION = RESULTS / "bonsai2-dspark-first-smoke-parent-adjudication-v1.json"
ADJUDICATION_SHA256 = "9dc68b437fdfdd582c48d3559bf5f9977f9abfdd27d53dd5b4e8d947819f55c4"
ADJUDICATION_SIZE = 6126
PLAN = RESULTS / "bonsai2-dspark-continuation-plan-v1.json"
PLAN_SHA256 = "59c29891c0ed63b27bbfa42327c09cba5e331d05d0c716bedecb006e2abea6f7"
PLAN_SIZE = 4179
ORIGINAL_CONFIG = RESULTS / f"{ORIGINAL_TAG}-candidate-config.json"
ORIGINAL_CONFIG_SHA256 = "4069730099a528a1f522a4fcada2431ba30bd5e74af1c747128e088e54813bf2"
ORIGINAL_CONFIG_SIZE = 1485
PILOT_SOURCE = ROOT / "runtime/bonsai_dspark_pilot.py"
PILOT_SOURCE_SHA256 = "998a6f0b4a68f7ab1ce7a90659a3b7a76fc55db198623d85694da2cf33323417"
PILOT_SOURCE_SIZE = 88435
PREFLIGHT = RESULTS / "bonsai2-dspark-continuation-preflight-v2.json"
ACCEPTANCE = RESULTS / "bonsai2-dspark-continuation-parent-acceptance-v1.json"

EXPECTED_SETUP_PLAN: dict[str, Any] = {
    "schema_version": 1,
    "name": "Bonsai PTQ DSpark remaining matched quick2, no duplicate smoke",
    "steps": [{
        "id": SETUP_STEP_ID,
        "suite": "quick",
        "cases": list(QUICK_IDS),
        "thinking": "medium",
        "temperature": 0,
        "seed": 0,
        "output_tokens": 65_536,
        "request_timeout": 21_600,
        "repeat": 1,
    }],
}


class ContinuationError(p.PilotError):
    """A continuation authorization, identity, or lifecycle invariant failed."""


@dataclass(frozen=True)
class ContinuationInputs:
    acceptance: p.Binding
    adjudication: p.Binding
    plan: p.Binding
    eligibility: p.Eligibility
    original_config: p.Binding
    deadline: datetime


@dataclass(frozen=True)
class ContinuationPaths:
    config: Path
    setup_plan: Path
    control: Path
    events: Path
    server_log: Path
    session_report: Path
    quick: Path
    quality: Path
    quality_texts: Path
    receipt: Path

    def pilot_paths(self) -> p.PilotPaths:
        return p.PilotPaths(
            config=self.config,
            smoke_plan=self.setup_plan,
            control=self.control,
            events=self.events,
            server_log=self.server_log,
            session_report=self.session_report,
            smoke=self.quick,
            quick=self.quick,
            quality=self.quality,
            quality_texts=self.quality_texts,
            receipt=self.receipt,
        )


def continuation_paths(*, results: Path = RESULTS) -> ContinuationPaths:
    prefix = results / TAG
    return ContinuationPaths(
        config=Path(f"{prefix}-candidate-config.json"),
        setup_plan=Path(f"{prefix}-setup-plan.json"),
        control=Path(f"{prefix}-control.json"),
        events=Path(f"{prefix}-runtime-events.jsonl"),
        server_log=Path(f"{prefix}-server.log"),
        session_report=Path(f"{prefix}-session-report.json"),
        quick=Path(f"{prefix}-01-{SETUP_STEP_ID}.jsonl"),
        quality=Path(f"{prefix}-quality-pilot.jsonl"),
        quality_texts=Path(f"{prefix}-quality-pilot.jsonl.texts"),
        receipt=Path(f"{prefix}-continuation-controller.json"),
    )


def _same_path(value: Any, expected: Path) -> bool:
    try:
        return Path(str(value)).expanduser().resolve() == expected.resolve()
    except (OSError, RuntimeError, ValueError):
        return False


def _binding_dict(binding: p.Binding) -> dict[str, Any]:
    return {"path": str(binding.path), "sha256": binding.sha256, "size": binding.size}


def _load_pinned(path: Path, digest: str, size: int, description: str) -> tuple[dict[str, Any], p.Binding]:
    document, binding = p._json(path, description)
    if binding.sha256 != digest or binding.size != size:
        raise ContinuationError(f"{description} differs from its immutable pin")
    return document, binding


def _require_binding(value: Any, path: Path, digest: str, size: int, description: str) -> p.Binding:
    binding = p.validate_binding(value, description)
    if binding.path != path.resolve() or binding.sha256 != digest or binding.size != size:
        raise ContinuationError(f"{description} does not match its exact expected binding")
    return binding


def _canonical_sha(row: Mapping[str, Any]) -> str:
    raw = json.dumps(row, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _validate_telemetry_row(value: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(str(value.get("path"))).expanduser().resolve()
    line_number = value.get("source_line")
    if not (
        path == (RESULTS / "specialists-machine-telemetry-part3b.jsonl").resolve()
        and line_number == 264
        and value.get("source_raw_line_sha256") == "ac97397549f88ddebc08245e2acfe4c1e38660cf7e9af30a4c718ddc57f0ba99"
        and value.get("source_row_canonical_sha256") == "6356b3f69b9fc3cd45d73191ba78263e480053e88763fa5ba4362dc7f0f09b04"
    ):
        raise ContinuationError("first-smoke telemetry row binding changed")
    raw: bytes | None = None
    try:
        with path.open("rb") as handle:
            for index, line in enumerate(handle, 1):
                if index == line_number:
                    raw = line.rstrip(b"\r\n")
                    break
    except OSError as exc:
        raise ContinuationError("first-smoke telemetry source is unreadable") from exc
    if raw is None or hashlib.sha256(raw).hexdigest() != value["source_raw_line_sha256"]:
        raise ContinuationError("first-smoke telemetry source row changed")
    try:
        row = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ContinuationError("first-smoke telemetry source row is malformed") from exc
    if not isinstance(row, dict) or _canonical_sha(row) != value["source_row_canonical_sha256"]:
        raise ContinuationError("first-smoke telemetry canonical row changed")
    processes = row.get("gpu_processes") if isinstance(row.get("gpu_processes"), list) else []
    gpu = row.get("gpu") if isinstance(row.get("gpu"), list) else []
    matches = [x for x in processes if isinstance(x, Mapping) and x.get("pid") == "1519592"]
    if not (
        row.get("at") == "2026-10-20T21:49:37.683966+00:00"
        and len(matches) == 1
        and matches[0].get("used_gpu_memory") == "13638"
        and len(gpu) >= 1
        and isinstance(gpu[0], Mapping)
        and gpu[0].get("utilization.gpu") == "100"
        and value.get("owned_native_pid") == 1519592
        and value.get("used_mib") == 13638
        and value.get("gpu_utilization_percent") == 100
    ):
        raise ContinuationError("first-smoke telemetry identity or measurements changed")
    return {
        "path": str(path), "source_line": line_number,
        "source_raw_line_sha256": value["source_raw_line_sha256"],
        "source_row_canonical_sha256": value["source_row_canonical_sha256"],
        "sample_at": row["at"], "owned_native_pid": 1519592,
        "used_mib": 13638, "gpu_utilization_percent": 100,
    }


def validate_adjudication() -> tuple[dict[str, Any], p.Binding]:
    document, binding = _load_pinned(
        ADJUDICATION, ADJUDICATION_SHA256, ADJUDICATION_SIZE,
        "first-smoke parent adjudication",
    )
    if not (
        document.get("schema_version") == 1
        and document.get("kind") == "bonsai2_dspark_first_smoke_parent_adjudication"
        and document.get("accepted_for_continuation") is True
        and document.get("decision") == "clean_smoke_pass_with_real_speculation_and_owned_gpu_residency"
        and document.get("original_tag") == ORIGINAL_TAG
        and document.get("model_session_run_id") == ORIGINAL_RUN_ID
        and document.get("first_pilot_created_at") == "2026-10-20T21:49:11.799450+00:00"
        and document.get("original_operational_deadline") == ORIGINAL_DEADLINE_TEXT
        and document.get("spent_generation_requests") == 1
        and document.get("remaining_generation_requests_max") == 3
        and document.get("full_pilot_generation_requests_max") == 4
        and document.get("prose_bodies_read") is False
    ):
        raise ContinuationError("first-smoke adjudication does not authorize this exact continuation")
    _require_binding(document.get("candidate_config"), ORIGINAL_CONFIG, ORIGINAL_CONFIG_SHA256,
                     ORIGINAL_CONFIG_SIZE, "original candidate config")
    _require_binding(document.get("original_controller"), PILOT_SOURCE, PILOT_SOURCE_SHA256,
                     PILOT_SOURCE_SIZE, "original controller")
    for key in ("original_execution_receipt", "original_launch", "runtime_events", "server_log",
                "session_report", "smoke_plan", "smoke_raw"):
        p.validate_binding(document.get(key), f"adjudicated {key}")
    smoke = document.get("smoke_result") if isinstance(document.get("smoke_result"), Mapping) else {}
    if not (
        smoke.get("call_id") == "61ccc0a6-60cf-4ece-9f8f-782e998ee9aa"
        and smoke.get("case_id") == "quick_json_extract"
        and smoke.get("passed") is True and smoke.get("eos") is True
        and smoke.get("nonempty") is True and smoke.get("error_free") is True
        and smoke.get("truncated") is False and smoke.get("draft_proposals") == 48
        and smoke.get("draft_accepted") == 41 and smoke.get("output_tokens") == 51
        and smoke.get("native_predicted_n") == 51
        and math.isclose(float(smoke.get("native_predicted_seconds")), 9.35972)
        and math.isclose(float(smoke.get("wall_seconds")), 18.064972648047842)
    ):
        raise ContinuationError("adjudicated smoke summary changed")
    cleanup = document.get("independent_cleanup_verification")
    cleanup = cleanup if isinstance(cleanup, Mapping) else {}
    restoration = cleanup.get("restoration") if isinstance(cleanup.get("restoration"), Mapping) else {}
    server_cleanup = cleanup.get("server_cleanup") if isinstance(cleanup.get("server_cleanup"), Mapping) else {}
    ports = restoration.get("ports_clear") if isinstance(restoration.get("ports_clear"), Mapping) else {}
    if not (
        cleanup.get("session_status") == "passed"
        and cleanup.get("semantic_step_failures") is False
        and server_cleanup.get("stopped") is True and server_cleanup.get("kill_sent") is False
        and restoration.get("original_running") is True
        and restoration.get("original_paused") is False
        and restoration.get("original_api_healthy") is True
        and ports.get("18080") is True and ports.get("18081") is True
    ):
        raise ContinuationError("adjudicated independent cleanup is not clean")
    telemetry = document.get("gpu_residency_evidence")
    if not isinstance(telemetry, Mapping):
        raise ContinuationError("adjudication lacks first-smoke GPU residency evidence")
    _validate_telemetry_row(telemetry)
    quiet = document.get("quiet_log_evidence") if isinstance(document.get("quiet_log_evidence"), Mapping) else {}
    if not (
        quiet.get("exact_draft_filename_logged") is True
        and quiet.get("dflash_implementation_logged") is True
        and quiet.get("draft_dspark_enabled_logged") is True
        and quiet.get("explicit_cuda_buffer_log_present") is False
        and quiet.get("no_inferred_layer_count") is True
    ):
        raise ContinuationError("adjudicated quiet-log evidence changed")
    return document, binding


def validate_plan() -> tuple[dict[str, Any], p.Binding]:
    document, binding = _load_pinned(PLAN, PLAN_SHA256, PLAN_SIZE, "continuation plan")
    budget = document.get("generation_budget") if isinstance(document.get("generation_budget"), Mapping) else {}
    gates = document.get("quick_gates") if isinstance(document.get("quick_gates"), Mapping) else {}
    quality = document.get("quality") if isinstance(document.get("quality"), Mapping) else {}
    time_budget = document.get("time_budget") if isinstance(document.get("time_budget"), Mapping) else {}
    if not (
        document.get("schema_version") == 1
        and document.get("kind") == "bonsai2_dspark_post_smoke_continuation_plan"
        and document.get("status") == "parent_approved_within_original_four_request_budget"
        and document.get("new_tag") == TAG and document.get("original_pilot") == ORIGINAL_TAG
        and document.get("setup_plan") == EXPECTED_SETUP_PLAN
        and budget == {"already_spent": 1, "duplicate_json_smoke_allowed": False,
                       "quick_setup_step_counts_as_measured_requests": True,
                       "remaining_maximum": 3, "total_maximum": 4}
        and gates.get("baseline_native_tps") == 4.601375520869196
        and gates.get("baseline_wall_seconds") == 532.4893405900802
        and gates.get("minimum_native_tps_ratio") == 1.05
        and gates.get("maximum_combined_wall_ratio") == 0.95
        and quality.get("kind") == "code"
        and quality.get("case_id") == "code_fresh_batch_module"
        and quality.get("maximum_requests") == 1
        and quality.get("thinking") == "medium"
        and quality.get("chat_template_thinking") == "on"
        and quality.get("temperature") == 0 and quality.get("seed") == 0
        and quality.get("output_token_limit") == 65_536
        and quality.get("timeout_seconds") == 21_600
        and time_budget.get("original_operational_deadline") == ORIGINAL_DEADLINE_TEXT
        and time_budget.get("no_fresh_eight_hour_allowance") is True
        and time_budget.get("restoration_reserve_seconds") == 1_800
        and document.get("source_and_output_artifacts_immutable") is True
        and document.get("prose_bodies_read") is False
    ):
        raise ContinuationError("continuation plan controls or budget changed")
    _require_binding(document.get("candidate_config"), ORIGINAL_CONFIG, ORIGINAL_CONFIG_SHA256,
                     ORIGINAL_CONFIG_SIZE, "plan original candidate config")
    _require_binding(document.get("eligibility"), ELIGIBILITY, ELIGIBILITY_SHA256,
                     ELIGIBILITY_SIZE, "plan eligibility")
    _require_binding(document.get("first_smoke_adjudication"), ADJUDICATION,
                     ADJUDICATION_SHA256, ADJUDICATION_SIZE, "plan adjudication")
    _require_binding(document.get("original_controller"), PILOT_SOURCE, PILOT_SOURCE_SHA256,
                     PILOT_SOURCE_SIZE, "plan original controller")
    return document, binding


def validate_acceptance(path: Path) -> p.Binding:
    if path.expanduser().resolve() != ACCEPTANCE.resolve():
        raise ContinuationError("parent acceptance must use the fixed continuation acceptance path")
    document, binding = p._json(path, "continuation parent acceptance")
    budget = document.get("generation_budget") if isinstance(document.get("generation_budget"), Mapping) else {}
    if not (
        document.get("schema_version") == 1
        and document.get("kind") == "bonsai2_dspark_continuation_parent_acceptance"
        and document.get("decision") == "execute"
        and document.get("accepted_by_parent") is True
        and document.get("tag") == TAG and document.get("original_tag") == ORIGINAL_TAG
        and document.get("original_model_session_run_id") == ORIGINAL_RUN_ID
        and document.get("original_operational_deadline") == ORIGINAL_DEADLINE_TEXT
        and budget == {"already_spent": 1, "continuation_maximum": 3,
                       "total_maximum": 4, "duplicate_smoke": False}
    ):
        raise ContinuationError("parent acceptance does not authorize this exact continuation")
    _require_binding(document.get("controller"), Path(__file__), p.sha256_file(Path(__file__)),
                     Path(__file__).stat().st_size, "accepted continuation controller")
    preflight_doc, preflight_binding = p._json(PREFLIGHT, "continuation CPU preflight")
    if not (
        preflight_doc.get("schema_version") == 1
        and preflight_doc.get("kind") == "bonsai2_dspark_continuation_preflight"
        and preflight_doc.get("status") == "passed"
        and preflight_doc.get("gpu_or_model_execution_performed") is False
        and preflight_doc.get("controller_sha256") == p.sha256_file(Path(__file__))
    ):
        raise ContinuationError("accepted continuation CPU preflight is invalid")
    _require_binding(document.get("preflight"), PREFLIGHT, preflight_binding.sha256,
                     preflight_binding.size, "accepted continuation preflight")
    _require_binding(document.get("continuation_plan"), PLAN, PLAN_SHA256, PLAN_SIZE,
                     "accepted continuation plan")
    _require_binding(document.get("first_smoke_adjudication"), ADJUDICATION,
                     ADJUDICATION_SHA256, ADJUDICATION_SIZE, "accepted first-smoke adjudication")
    _require_binding(document.get("eligibility"), ELIGIBILITY, ELIGIBILITY_SHA256,
                     ELIGIBILITY_SIZE, "accepted eligibility")
    _require_binding(document.get("original_candidate_config"), ORIGINAL_CONFIG,
                     ORIGINAL_CONFIG_SHA256, ORIGINAL_CONFIG_SIZE, "accepted original config")
    return binding


def validate_inputs(acceptance_path: Path, *, now: datetime | None = None) -> ContinuationInputs:
    p.validate_static_sources()
    if p.sha256_file(PILOT_SOURCE) != PILOT_SOURCE_SHA256 or PILOT_SOURCE.stat().st_size != PILOT_SOURCE_SIZE:
        raise ContinuationError("original pilot controller changed")
    adjudication_document, adjudication = validate_adjudication()
    plan_document, plan = validate_plan()
    acceptance = validate_acceptance(acceptance_path)
    eligibility = p.validate_eligibility(ELIGIBILITY, now=now)
    if eligibility.source.sha256 != ELIGIBILITY_SHA256 or eligibility.source.size != ELIGIBILITY_SIZE:
        raise ContinuationError("eligibility differs from the continuation plan pin")
    if not (
        eligibility.baseline_native_tps == 4.601375520869196
        and eligibility.baseline_wall_seconds == 532.4893405900802
        and eligibility.quality_kind == "code"
        and eligibility.quality_case_id == "code_fresh_batch_module"
    ):
        raise ContinuationError("eligible quick baseline or selected quality case changed")
    original_config = _require_binding(
        plan_document.get("candidate_config"), ORIGINAL_CONFIG, ORIGINAL_CONFIG_SHA256,
        ORIGINAL_CONFIG_SIZE, "continuation source config",
    )
    if adjudication_document.get("original_operational_deadline") != ORIGINAL_DEADLINE_TEXT:
        raise ContinuationError("adjudication and continuation deadline disagree")
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        raise ContinuationError("continuation time must be timezone-aware")
    if (ORIGINAL_DEADLINE - current.astimezone(timezone.utc)).total_seconds() <= RESTORATION_RESERVE_SECONDS:
        raise ContinuationError("original pilot deadline lacks the restoration reserve")
    return ContinuationInputs(acceptance, adjudication, plan, eligibility, original_config,
                              ORIGINAL_DEADLINE)


def _assert_unused(paths: ContinuationPaths) -> None:
    collisions = [str(path) for path in paths.__dict__.values() if os.path.lexists(path)]
    if collisions:
        raise ContinuationError("continuation would overwrite an existing artifact: " + ", ".join(collisions))


def _exclusive_copy(source: Path, destination: Path) -> p.Binding:
    raw = source.read_bytes()
    if len(raw) != ORIGINAL_CONFIG_SIZE or hashlib.sha256(raw).hexdigest() != ORIGINAL_CONFIG_SHA256:
        raise ContinuationError("original candidate config changed before materialization")
    destination.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(destination, flags, 0o600)
    try:
        view = memoryview(raw)
        offset = 0
        while offset < len(view):
            written = os.write(descriptor, view[offset:])
            if written <= 0:
                raise OSError("short exclusive config copy")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return p.Binding(destination.resolve(), ORIGINAL_CONFIG_SHA256, len(raw))


def prepare_runtime(paths: ContinuationPaths, plan_document: Mapping[str, Any]) -> dict[str, p.Binding]:
    _assert_unused(paths)
    setup = plan_document.get("setup_plan")
    if setup != EXPECTED_SETUP_PLAN:
        raise ContinuationError("only the exact no-smoke quick2 setup plan may be materialized")
    config = _exclusive_copy(ORIGINAL_CONFIG, paths.config)
    setup_binding = p._exclusive_json(paths.setup_plan, setup)
    return {"config": config, "setup_plan": setup_binding}


def remaining_time(now: datetime | None = None) -> dict[str, Any]:
    current = now or datetime.now(timezone.utc)
    total = (ORIGINAL_DEADLINE - current.astimezone(timezone.utc)).total_seconds()
    usable = total - RESTORATION_RESERVE_SECONDS
    if usable <= p.STARTUP_TIMEOUT:
        raise ContinuationError("original pilot deadline has insufficient time for bounded startup and restoration")
    return {
        "original_deadline": ORIGINAL_DEADLINE_TEXT,
        "remaining_total_seconds": total,
        "restoration_reserve_seconds": RESTORATION_RESERVE_SECONDS,
        "usable_request_seconds": usable,
        "window_max_seconds": max(1, math.floor(total)),
    }


def outer_command(paths: ContinuationPaths, window_seconds: int) -> list[str]:
    return [
        sys.executable, str(ROOT / "runtime/gpu_window.py"),
        "--long-session", "--wait-seconds", str(int(p.IDLE_WAIT_SECONDS)),
        "--max-seconds", str(window_seconds), "--",
        str(ROOT / "runtime/model_session.py"), "--runtime-backend", "llama",
        "--config", str(paths.config), "--plan", str(paths.setup_plan),
        "--tag", TAG, "--startup-timeout", str(int(p.STARTUP_TIMEOUT)),
        "--hold-after-plan", "--control-file", str(paths.control),
    ]


def _quick_lifecycle(paths: ContinuationPaths, session: Any) -> None:
    from runtime import specialist_study_trial as study
    rows = study._read_event_rows(paths.events)
    completions = [row for row in rows if row.get("event") == "step_complete"]
    if not (
        len(completions) == 1
        and completions[0].get("id") == SETUP_STEP_ID
        and completions[0].get("result") == str(paths.quick)
        and completions[0].get("returncode") in {0, 1}
        and isinstance(completions[0].get("passed"), bool)
        and all(row.get("run_id") == session.run_id for row in rows)
    ):
        raise ContinuationError("quick2 setup is not bound to the active held session")


def _draft_counters(path: Path, description: str) -> list[dict[str, Any]]:
    rows = p._metadata_rows(path, description)
    result: list[dict[str, Any]] = []
    for row in rows:
        native = row.get("response", {}).get("native")
        native = native if isinstance(native, Mapping) else {}
        proposed, accepted = native.get("draft_n"), native.get("draft_n_accepted")
        if not (
            isinstance(proposed, (int, float)) and not isinstance(proposed, bool)
            and math.isfinite(float(proposed)) and float(proposed) > 0
            and isinstance(accepted, (int, float)) and not isinstance(accepted, bool)
            and math.isfinite(float(accepted)) and 0 <= float(accepted) <= float(proposed)
        ):
            raise ContinuationError(f"{description} lacks genuine draft proposal/acceptance counters")
        result.append({"case_id": row.get("case_id"), "draft_proposals": proposed,
                       "draft_accepted": accepted})
    return result


def _runtime_engagement(paths: ContinuationPaths, session: Any,
                        gpu_processes: Callable[[], list[dict[str, Any]]]) -> dict[str, Any]:
    from runtime import specialist_study_trial as study
    events = study._read_event_rows(paths.events)
    starts = [row for row in events if row.get("event") == "server_started"]
    if len(starts) != 1 or starts[0].get("run_id") != session.run_id:
        raise ContinuationError("continuation lacks one owned server process")
    pid = starts[0].get("pid")
    resident = [row for row in gpu_processes() if row.get("pid") == pid
                and isinstance(row.get("used_mib"), (int, float)) and row.get("used_mib") > 0]
    if len(resident) != 1:
        raise ContinuationError("owned continuation server lacks unique live GPU residency")
    p._regular(paths.server_log, "continuation server log")
    draft_name = p.DS_PARK_MODEL.name.lower()
    exact_draft = dflash = speculative = False
    with paths.server_log.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            lowered = line.lower()
            exact_draft |= draft_name in lowered
            dflash |= "dflash" in lowered
            speculative |= "speculative decoding enabled" in lowered and "draft-dspark" in lowered
    if not (exact_draft and dflash and speculative):
        raise ContinuationError("continuation server log lacks exact draft, dflash, or speculative evidence")
    return {
        "owned_gpu_process": {"pid": pid, "used_mib": resident[0]["used_mib"]},
        "exact_draft_filename_logged": True,
        "dflash_implementation_logged": True,
        "draft_dspark_enabled_logged": True,
        "cuda_buffer_log_required": False,
        "requested_draft_ngl": "all",
        "inferred_loaded_layer_count": None,
    }


def validate_quick(paths: ContinuationPaths, eligibility: p.Eligibility, session: Any,
                   gpu_processes: Callable[[], list[dict[str, Any]]]) -> dict[str, Any]:
    _quick_lifecycle(paths, session)
    caught: BaseException | None = None
    result: dict[str, Any]
    try:
        result = p.validate_quick_candidate(paths.quick, eligibility)
    except (p.PilotGateStopped, p.PilotStageError) as exc:
        caught = exc
        result = dict(exc.evidence)
    counters = _draft_counters(paths.quick, "continuation quick result")
    engagement = _runtime_engagement(paths, session, gpu_processes)
    result["draft_counters"] = counters
    result["runtime_engagement"] = engagement
    if caught is not None:
        caught.evidence = result  # type: ignore[attr-defined]
        raise caught
    return result


def _observation(path: Path) -> dict[str, Any]:
    return p._result_row_counts(path)


def request_accounting(paths: ContinuationPaths, *, quick_launched: bool,
                       quick_completed: bool, quality_launched: bool,
                       quality_completed: bool) -> dict[str, Any]:
    stages: dict[str, Any] = {}
    lower = ALREADY_SPENT
    upper = ALREADY_SPENT
    exact_values: list[int | None] = [ALREADY_SPENT]
    violation = False
    for name, reserved, path, launched, completed in (
        ("matched_quick_setup", 2, paths.quick, quick_launched, quick_completed),
        ("quality_pilot", 1, paths.quality, quality_launched, quality_completed),
    ):
        observed = _observation(path)
        rows = observed["observed_result_rows"]
        terminal = observed["observed_terminal_or_error_rows"]
        stage_lower = rows if isinstance(rows, int) else 0
        if stage_lower > reserved:
            violation = True
        if not launched and rows == 0:
            exact: int | None = 0
        elif completed and isinstance(rows, int) and rows > 0 and terminal == rows:
            exact = rows
        else:
            exact = None
        stage_upper = exact if exact is not None else reserved if launched else stage_lower
        lower += stage_lower
        upper += stage_upper
        exact_values.append(exact)
        stages[name] = {
            "allocated_request_slots": reserved, "launch_attempted": launched,
            "stage_completed": completed, **observed,
            "attempts_lower_bound": stage_lower, "attempts_upper_bound": stage_upper,
            "exact_attempts": exact, "has_unobserved_attempt_uncertainty": exact is None,
        }
    exact_total = sum(x for x in exact_values if x is not None) if all(
        x is not None for x in exact_values) else None
    violation |= lower > TOTAL_MAX or upper > TOTAL_MAX
    return {
        "first_smoke_already_spent": ALREADY_SPENT,
        "continuation_stages": stages,
        "observed_total_attempts_lower_bound": lower,
        "possible_total_attempts_upper_bound": upper,
        "exact_total_attempts": exact_total,
        "total_maximum": TOTAL_MAX,
        "continuation_maximum": CONTINUATION_MAX,
        "budget_violation": violation,
        "duplicate_smoke_requests": 0,
        "reserved_slots_are_not_reported_as_actual_attempts": True,
    }


def retry_cleanup_evidence(paths: p.PilotPaths, session: Any, cleanup: Mapping[str, Any], *,
                           attempts: int = 61, delay: float = 2.0,
                           validator: Callable[[p.PilotPaths, Any, Mapping[str, Any]], dict[str, Any]] = p._final_cleanup_evidence,
                           sleeper: Callable[[float], None] = time.sleep) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    errors: list[dict[str, str]] = []
    for index in range(1, attempts + 1):
        try:
            evidence = validator(paths, session, cleanup)
            return evidence, {"attempts": index, "bounded_attempt_limit": attempts,
                              "delayed_restoration_observed": index > 1, "errors": errors}
        except Exception as exc:
            errors.append({"error_type": type(exc).__name__, "error": str(exc)})
            if index < attempts:
                sleeper(delay)
    return None, {"attempts": attempts, "bounded_attempt_limit": attempts,
                  "delayed_restoration_observed": attempts > 1, "errors": errors}


def execute(inputs: ContinuationInputs, *, paths: ContinuationPaths | None = None,
            now: Callable[[], datetime] = lambda: datetime.now(timezone.utc)) -> tuple[dict[str, Any], int]:
    from runtime import gpu_window

    selected = paths or continuation_paths()
    time_before = remaining_time(now())
    boundary_before = p.live_boundary_preflight()
    _assert_unused(selected)
    p.verify_dspark_model(full_hash=True)
    # Recheck immutable authorization, READY/release, deadline, and live restore
    # after the long model hash and immediately before any artifact/process.
    refreshed = validate_inputs(inputs.acceptance.path, now=now())
    time_launch = remaining_time(now())
    boundary_before_launch = p.live_boundary_preflight()
    plan_document, _ = _load_pinned(PLAN, PLAN_SHA256, PLAN_SIZE, "continuation plan")
    prepared = prepare_runtime(selected, plan_document)
    pilot_paths = selected.pilot_paths()
    outer: subprocess.Popen[Any] | None = None
    session: Any | None = None
    signals = p.SignalLatch()
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    started = time.monotonic()
    helper_deadline = started + time_launch["usable_request_seconds"]
    commands: list[dict[str, Any]] = []
    quick: dict[str, Any] | None = None
    quality: dict[str, Any] | None = None
    gate_outcome: dict[str, Any] | None = None
    primary_problem: dict[str, Any] | None = None
    cleanup_problem: dict[str, Any] | None = None
    cleanup: dict[str, Any] = {}
    cleanup_evidence: dict[str, Any] | None = None
    cleanup_retry: dict[str, Any] | None = None
    quick_launched = quick_completed = quality_launched = quality_completed = False
    status, code = "infrastructure_failed", 1
    try:
        for sig in previous:
            signal.signal(sig, signals.handle)
        argv = outer_command(selected, int(time_launch["window_max_seconds"]))
        commands.append({"kind": "gpu_window_model_session_quick2_setup", "argv": argv})
        quick_launched = True
        outer = subprocess.Popen(argv, stdin=subprocess.DEVNULL, start_new_session=True, close_fds=True)
        session = p._wait_holding(outer, pilot_paths, TAG, helper_deadline, signals)
        quick_completed = True
        quick = validate_quick(selected, refreshed.eligibility, session,
                               gpu_window.gpu_compute_processes)
        quality_argv = p.quality_command(TAG, refreshed.eligibility, session, pilot_paths)
        quality_launched = True
        run = p._run_owned(quality_argv, outer, helper_deadline, signals)
        commands.append({"kind": "single_quality_pilot", **run})
        quality_completed = True
        quality = p.validate_quality_candidate(selected.quality, refreshed.eligibility)
        quality["draft_counters"] = _draft_counters(selected.quality, "continuation quality result")
        if run["returncode"] != 0:
            raise ContinuationError("quality helper returned nonzero despite a passing result")
        status, code = "complete", 0
    except p.PilotInterrupted as exc:
        primary_problem = {"error_type": type(exc).__name__, "error": str(exc), "signal": exc.signum}
        status, code = "interrupted", 128 + exc.signum
    except p.PilotGateStopped as exc:
        if exc.stage == "quick": quick = dict(exc.evidence)
        elif exc.stage == "quality": quality = dict(exc.evidence)
        gate_outcome = {"stage": exc.stage, "reason": str(exc),
                        "classification": "expected_gate_stop"}
        status, code = "stopped_at_gate", 0
    except p.PilotStageError as exc:
        if exc.stage == "quick": quick = dict(exc.evidence)
        elif exc.stage == "quality": quality = dict(exc.evidence)
        primary_problem = {"error_type": type(exc).__name__, "error": str(exc), "stage": exc.stage}
    except Exception as exc:
        primary_problem = {"error_type": type(exc).__name__, "error": str(exc)}
        helper_cleanup = getattr(exc, "owned_helper_cleanup", None)
        if helper_cleanup is not None:
            primary_problem["owned_helper_cleanup"] = helper_cleanup
    finally:
        try:
            cleanup = p._session_cleanup(outer, pilot_paths, session)
        except Exception as exc:
            cleanup_problem = {"error_type": type(exc).__name__, "error": str(exc)}
        if outer is not None and session is not None and cleanup.get("outer_exited") is True:
            cleanup_evidence, cleanup_retry = retry_cleanup_evidence(pilot_paths, session, cleanup)
            if cleanup_evidence is None:
                last = cleanup_retry["errors"][-1] if cleanup_retry and cleanup_retry["errors"] else {}
                cleanup_problem = {"error_type": last.get("error_type", "ContinuationError"),
                                   "error": last.get("error", "clean restoration was not observed")}
        if cleanup_problem is not None:
            status, code = "cleanup_failed", 1
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    accounting = request_accounting(
        selected, quick_launched=quick_launched, quick_completed=quick_completed,
        quality_launched=quality_launched, quality_completed=quality_completed,
    )
    if accounting["budget_violation"] is True:
        primary_problem = primary_problem or {
            "error_type": "ContinuationError", "error": "four-request pilot budget exceeded"}
        status, code = "infrastructure_failed", 1
    receipt = {
        "schema_version": 1, "kind": "bonsai2_dspark_continuation_execution",
        "status": status, "tag": TAG, "created_at": p.utc_now(),
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "authorization": {
            "acceptance": _binding_dict(refreshed.acceptance),
            "adjudication": _binding_dict(refreshed.adjudication),
            "continuation_plan": _binding_dict(refreshed.plan),
            "eligibility": _binding_dict(refreshed.eligibility.source),
            "original_config": _binding_dict(refreshed.original_config),
        },
        "candidate": {key: _binding_dict(value) for key, value in prepared.items()},
        "draft": {"path": str(p.DS_PARK_MODEL), "size": p.DS_PARK_SIZE,
                  "sha256": p.DS_PARK_SHA256, "full_sha256_verified_for_execution": True,
                  "architecture": "dflash", "legacy_markov_cuda_option_used": False},
        "time_budget_before_validation": time_before,
        "time_budget_before_launch": time_launch,
        "live_boundary_before": boundary_before,
        "live_boundary_before_launch": boundary_before_launch,
        "session": ({"run_id": session.run_id, "model_name": session.model_name,
                     "context": session.context, "initial_load_seconds": session.initial_load_seconds}
                    if session is not None else None),
        "commands": commands, "first_smoke_rerun": False,
        "quick": quick, "quality": quality, "gate_outcome": gate_outcome,
        "generation_budget": accounting,
        "cleanup": cleanup, "cleanup_retry": cleanup_retry,
        "cleanup_evidence": cleanup_evidence,
        "primary_problem": primary_problem, "cleanup_problem": cleanup_problem,
        "artifacts": {
            name: {"path": str(path.resolve()),
                   "exists": path.is_file() and not path.is_symlink(),
                   "size": path.stat().st_size if path.is_file() and not path.is_symlink() else None,
                   "sha256": p.sha256_file(path) if path.is_file() and not path.is_symlink() else None}
            for name, path in {"matched_quick": selected.quick, "single_quality": selected.quality,
                              "runtime_events": selected.events, "server_log": selected.server_log}.items()
        },
        "claims": {"full_code3_or_hard4_candidate_score": False,
                   "official_or_trained_bonsai2_paired_head": False,
                   "private_reasoning_retained": False,
                   "inferred_draft_layer_count": False},
    }
    p._exclusive_json(selected.receipt, receipt)
    return receipt, code


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acceptance", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--execute", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        inputs = validate_inputs(args.acceptance)
        if args.check_only:
            timing = remaining_time()
            print(json.dumps({
                "authorized": True,
                "mode": "CPU-only validation; no runtime artifacts or processes created",
                "acceptance_sha256": inputs.acceptance.sha256,
                "deadline": ORIGINAL_DEADLINE_TEXT,
                "remaining_total_seconds": round(timing["remaining_total_seconds"], 3),
                "requests_already_spent": ALREADY_SPENT,
                "continuation_requests_maximum": CONTINUATION_MAX,
                "total_requests_maximum": TOTAL_MAX,
                "setup_cases": list(QUICK_IDS),
                "duplicate_smoke": False,
            }, sort_keys=True))
            return 0
        _, code = execute(inputs)
        return code
    except p.PilotError as exc:
        print(f"bonsai_dspark_continuation: {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
