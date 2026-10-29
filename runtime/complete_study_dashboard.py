#!/usr/bin/env python3
"""Build an offline metadata dashboard from normalized benchmark reports."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import html
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = 1
OUTCOME_BUCKETS = ("passed", "completed_wrong", "truncated", "timeout", "request_failure", "other")
ATTEMPT_CONTRACT_IDENTITY_KEYS = (
    "case_id", "case_sha256", "case_version", "case_source_sha256", "prompt_sha256",
    "call_id", "visible_text_sha256",
)
CSV_FIELDS = (
    "dashboard_record_id", "call_id", "run_id", "started_at", "finished_at", "profile_id", "profile_key", "group_id", "model", "runtime",
    "backend", "weight_quant", "suite", "role", "case_id", "parent_case_id", "case_version",
    "case_sha256", "grader_contract_note_ids_json", "supplemental_evaluator_diagnostics_json",
    "repeat_index", "outcome", "outcome_bucket", "completion_success", "quality_passed",
    "content_passed", "comparison_content_passed", "raw_json_object", "mechanical_constraints_passed",
    "review_status", "review_score", "review_max_points", "input_tokens", "output_tokens",
    "reasoning_tokens_reported", "native_prompt_seconds", "native_decode_n", "native_decode_seconds", "native_decode_tps",
    "native_load_seconds", "wall_seconds", "ttft_reasoning_seconds", "ttft_visible_seconds", "speed_eligible", "latency_eligible",
    "objective_count", "objectives_passed", "assertion_count", "assertions_passed",
    "hidden_test_count", "hidden_tests_passed", "requested_context_tokens", "actual_context_tokens",
    "output_token_limit", "request_timeout_seconds", "thinking", "chat_template_thinking",
    "reasoning_budget_tokens_requested", "reasoning_budget_tokens_config_default",
    "native_thinking_budget_tokens_requested", "temperature", "sampling_seed", "sampler_recipe_label",
    "sampler_recipe_sha256", "source_file", "source_line", "benchmark_source_file", "benchmark_source_line",
    "specialist_source_file", "source_config", "source_config_sha256",
    "case_source_sha256", "prompt_sha256", "visible_text_sha256", "model_session_run_id",
    "lifecycle_tag", "resource_join_status", "resource_scope", "resource_interval_status",
    "resource_lifecycle_context_tokens", "resource_initial_load_seconds",
    "resource_process_vram_sampled_max_mib", "resource_gpu_utilization_sampled_max_percent",
    "resource_board_power_sampled_max_watts", "resource_temperature_sampled_max_celsius",
    "resource_identity_note", "resource_hardware_power_brake_json", "resource_notes_json", "controls_json", "sampling_config_json",
    "sampler_overrides_json", "runtime_options_json", "requested_native_runtime_controls_json",
    "hidden_failure_labels_json", "source_reports_json",
)


class DashboardError(ValueError):
    """A normalized input or output contract is unsafe or malformed."""


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: datetime) -> str:
    if value.tzinfo is None:
        raise DashboardError("generated_at must be timezone-aware")
    return value.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _number(value: Any) -> int | float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _boolean(value: Any) -> bool | None:
    return value if isinstance(value, bool) else None


def _text(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _mapping(value: Any) -> dict[str, Any] | None:
    return dict(value) if isinstance(value, Mapping) else None


def _first(*values: Any) -> Any:
    return next((value for value in values if value is not None), None)


def _snapshot_json(path: Path, *, required: bool, label: str) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        if required:
            raise DashboardError(f"{label} does not exist: {resolved}")
        return None, {"kind": label, "path": str(resolved), "present": False}
    raw = resolved.read_bytes()
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DashboardError(f"{label} is not valid JSON: {resolved}: {exc}") from exc
    if not isinstance(value, dict):
        raise DashboardError(f"{label} must contain a JSON object: {resolved}")
    return value, {
        "kind": label,
        "path": str(resolved),
        "present": True,
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "generated_at": value.get("generated_at"),
        "schema_version": value.get("schema_version"),
    }


def _rows(document: Mapping[str, Any], key: str, label: str) -> list[dict[str, Any]]:
    value = document.get(key)
    if not isinstance(value, list) or any(not isinstance(row, dict) for row in value):
        raise DashboardError(f"{label}.{key} must be an array of objects")
    return [dict(row) for row in value]


def _stable_id(value: Any, length: int = 16) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:length]


def _supplemental_grade(value: Any, label: str) -> dict[str, Any]:
    keys = (
        "outcome", "passed", "objective_count", "objectives_passed",
        "assertion_count", "assertions_passed",
    )
    if not isinstance(value, Mapping):
        raise DashboardError(f"specialist_report supplemental diagnostic {label} must be an object")
    grade = {key: value.get(key) for key in keys}
    if (
        not _text(grade["outcome"])
        or not isinstance(grade["passed"], bool)
        or any(
            isinstance(grade[key], bool) or not isinstance(grade[key], int)
            for key in (
                "objective_count", "objectives_passed", "assertion_count", "assertions_passed",
            )
        )
    ):
        raise DashboardError(
            f"specialist_report supplemental diagnostic {label} has an invalid grade schema"
        )
    return grade


def _grader_contract_notes(document: Mapping[str, Any]) -> list[dict[str, Any]]:
    if "grader_contract_notes" not in document:
        return []
    notes = _rows(document, "grader_contract_notes", "specialist_report")
    required_identity = ("case_id", "case_sha256", "case_version", "prompt_sha256")
    normalized = []
    for note in notes:
        identity = note.get("case_identity")
        evidence = note.get("evidence")
        if (
            not isinstance(identity, Mapping)
            or any(not _text(identity.get(key)) for key in required_identity)
            or not _text(note.get("note_id"))
            or not _text(note.get("summary"))
            or note.get("score_adjustment") is not False
            or not isinstance(evidence, Mapping)
            or not _text(evidence.get("path"))
            or not _text(evidence.get("sha256"))
        ):
            raise DashboardError(
                "specialist_report.grader_contract_notes requires exact identity, "
                "supplemental wording, evidence, and score_adjustment=false"
            )
        normalized_note = {
            "note_id": note["note_id"],
            "kind": _text(note.get("kind")),
            "scope": _text(note.get("scope")),
            "case_identity": {key: identity[key] for key in required_identity},
            "summary": note["summary"],
            "score_adjustment": False,
            "other_case_failures_explained": note.get("other_case_failures_explained") is True,
            "evidence": {"path": evidence["path"], "sha256": evidence["sha256"]},
        }
        source_evidence = note.get("source_evidence")
        if source_evidence is not None:
            if (
                not isinstance(source_evidence, Mapping)
                or not _text(source_evidence.get("path"))
                or not _text(source_evidence.get("sha256"))
            ):
                raise DashboardError(
                    "specialist_report grader-contract source_evidence requires path and sha256"
                )
            normalized_note["source_evidence"] = {
                "path": source_evidence["path"],
                "sha256": source_evidence["sha256"],
            }
        attempt_identity = note.get("attempt_identity")
        supplemental = note.get("supplemental_diagnostic")
        if attempt_identity is not None or normalized_note["kind"] == "evaluator_environment_limitation":
            original = supplemental.get("original_strict") if isinstance(supplemental, Mapping) else None
            diagnostic = (
                supplemental.get("supplemental_environment_replay")
                if isinstance(supplemental, Mapping) else None
            )
            if (
                not isinstance(attempt_identity, Mapping)
                or any(not _text(attempt_identity.get(key)) for key in ATTEMPT_CONTRACT_IDENTITY_KEYS)
                or any(attempt_identity.get(key) != identity.get(key) for key in required_identity)
                or not isinstance(supplemental, Mapping)
                or not _text(supplemental.get("note_id"))
                or supplemental.get("note_id") != note.get("note_id")
                or supplemental.get("status") != "posthoc_unchanged_source_environment_diagnostic"
                or supplemental.get("source_unchanged") is not True
                or supplemental.get("primary_grade_preserved") is not True
                or supplemental.get("new_inference_requests") != 0
                or not _text(supplemental.get("environment_delta"))
            ):
                raise DashboardError(
                    "specialist_report evaluator-environment note requires exact attempt "
                    "identity and an unchanged-source, score-preserving diagnostic"
                )
            original_grade = _supplemental_grade(original, "original_strict")
            diagnostic_grade = _supplemental_grade(
                diagnostic, "supplemental_environment_replay"
            )
            normalized_note["attempt_identity"] = {
                key: attempt_identity[key] for key in ATTEMPT_CONTRACT_IDENTITY_KEYS
            }
            normalized_note["supplemental_diagnostic"] = {
                "note_id": supplemental["note_id"],
                "status": supplemental["status"],
                "source_unchanged": True,
                "primary_grade_preserved": True,
                "new_inference_requests": 0,
                "environment_delta": supplemental["environment_delta"],
                "original_strict": original_grade,
                "supplemental_environment_replay": diagnostic_grade,
            }
        normalized.append(normalized_note)
    return normalized


def _unambiguous_contract_identity(
    benchmark: Mapping[str, Any], specialist: Mapping[str, Any]
) -> dict[str, str] | None:
    identity: dict[str, str] = {}
    for key in ("case_id", "case_sha256", "case_version", "prompt_sha256"):
        values = {
            value for row in (benchmark, specialist)
            if (value := _text(row.get(key))) is not None
        }
        if len(values) != 1:
            return None
        identity[key] = next(iter(values))
    return identity


def _unambiguous_contract_attempt_identity(
    benchmark: Mapping[str, Any], specialist: Mapping[str, Any]
) -> dict[str, str] | None:
    identity: dict[str, str] = {}
    for key in ATTEMPT_CONTRACT_IDENTITY_KEYS:
        values = {
            value for row in (benchmark, specialist)
            if (value := _text(row.get(key))) is not None
        }
        if len(values) != 1:
            return None
        identity[key] = next(iter(values))
    return identity


def _attach_grader_contract_notes(
    attempts: Sequence[dict[str, Any]], notes: Sequence[Mapping[str, Any]]
) -> dict[str, int]:
    match_counts = {str(note["note_id"]): 0 for note in notes}
    for attempt in attempts:
        identity = attempt.pop("_grader_contract_identity", None)
        attempt_identity = attempt.pop("_grader_contract_attempt_identity", None)
        references = []
        diagnostics = []
        for note in notes:
            expected_attempt = note.get("attempt_identity")
            matched = (
                attempt_identity == expected_attempt
                if isinstance(expected_attempt, Mapping)
                else identity == note.get("case_identity")
            )
            if not matched:
                continue
            note_id = str(note["note_id"])
            references.append(note_id)
            match_counts[note_id] += 1
            supplemental = note.get("supplemental_diagnostic")
            if isinstance(supplemental, Mapping):
                original = supplemental.get("original_strict")
                observed = {
                    "outcome": attempt.get("outcome"),
                    "passed": attempt.get("quality_passed"),
                    "objective_count": attempt.get("objective_count"),
                    "objectives_passed": attempt.get("objectives_passed"),
                    "assertion_count": attempt.get("assertion_count"),
                    "assertions_passed": attempt.get("assertions_passed"),
                }
                if not isinstance(original, Mapping) or observed != dict(original):
                    raise DashboardError(
                        f"grader-contract audit original grade does not match dashboard "
                        f"attempt {attempt.get('call_id')}: {note_id}"
                    )
                diagnostics.append(dict(supplemental))
        if references:
            attempt["grader_contract_note_ids"] = references
        if diagnostics:
            attempt["supplemental_evaluator_diagnostics"] = diagnostics
    return match_counts


def _outcome_bucket(row: Mapping[str, Any]) -> str:
    outcome = str(row.get("outcome") or row.get("objective_outcome") or "").lower()
    error_type = str(row.get("request_error_type") or "").lower()
    if "timeout" in outcome or "timeout" in error_type:
        return "timeout"
    if outcome in {"request_failure", "http_error", "error"} or error_type:
        return "request_failure"
    if row.get("truncated") is True or outcome in {"truncated", "incomplete_stream"}:
        return "truncated"
    if outcome in {"passed", "pass", "success", "complete"}:
        return "passed"
    if outcome in {"wrong_answer", "failed", "failure", "completed_wrong"}:
        return "completed_wrong"
    return "other"


def _resource_identity(*rows: Mapping[str, Any]) -> tuple[dict[str, str] | None, str | None]:
    pairs: set[tuple[str, str]] = set()
    for row in rows:
        for key in ("resource_identity", "lifecycle_identity"):
            identity = row.get(key)
            if isinstance(identity, Mapping):
                tag, run_id = _text(identity.get("tag")), _text(identity.get("run_id"))
                if tag and run_id:
                    pairs.add((tag, run_id))
        run_id = _first(_text(row.get("model_session_run_id")), _text(row.get("session_run_id")))
        if run_id:
            for key in ("lifecycle_tag", "session_tag", "study_tag", "tag"):
                tag = _text(row.get(key))
                if tag:
                    pairs.add((tag, run_id))
    if len(pairs) == 1:
        tag, run_id = next(iter(pairs))
        return {"tag": tag, "run_id": run_id}, None
    if len(pairs) > 1:
        return None, "multiple lifecycle tag/run_id pairs are present"
    return None, "exact lifecycle tag and model-session run_id are unavailable"


def _review_metadata(specialist: Mapping[str, Any] | None, review_label: str | None) -> dict[str, Any]:
    review = specialist.get("review") if specialist and isinstance(specialist.get("review"), Mapping) else None
    score = _number(review.get("total_score")) if review else None
    maximum = _number(review.get("max_points")) if review else None
    eligible = _boolean(specialist.get("blinded_prose_review_eligible")) if specialist else None
    completion = _boolean(specialist.get("completion_success")) if specialist else None
    if score is not None and maximum is not None:
        status = "reviewed"
    elif eligible is True and completion is True:
        status = "pending"
    elif specialist and specialist.get("case_group") == "prose":
        status = "not_eligible"
    else:
        status = "not_applicable"
    return {"status": status, "score": score, "max_points": maximum,
            "reviewer_label": review_label if status == "reviewed" else None}


def _normalize_attempt(
    benchmark: Mapping[str, Any] | None,
    specialist: Mapping[str, Any] | None,
    *,
    ordinal: int,
    review_label: str | None,
) -> dict[str, Any]:
    benchmark = benchmark or {}
    specialist = specialist or {}
    value = lambda specialist_key, benchmark_key=None: _first(
        specialist.get(specialist_key), benchmark.get(benchmark_key or specialist_key)
    )
    call_id = _text(value("call_id"))
    suite = _text(value("suite"))
    role = _first(_text(specialist.get("case_group")), _text(benchmark.get("case_group")))
    if suite is None and role:
        suite = f"specialist_{role}"
    role = role or suite or "unclassified"
    profile_id = _text(benchmark.get("profile_id")) or _text(specialist.get("profile_id"))
    outcome = _text(value("outcome")) or _text(value("objective_outcome"))
    raw_for_outcome = dict(benchmark) | dict(specialist)
    raw_for_outcome["outcome"] = outcome
    bucket = _outcome_bucket(raw_for_outcome)
    completion = _boolean(specialist.get("completion_success"))
    if completion is None:
        completion = bucket in {"passed", "completed_wrong"}
    review_input = dict(specialist)
    review_input["completion_success"] = completion
    review = _review_metadata(review_input or None, review_label)
    content_passed = _boolean(benchmark.get("content_passed"))
    comparison_passed = _boolean(benchmark.get("comparison_content_passed"))
    if role == "prose":
        quality_passed = None
    else:
        quality_passed = _first(
            _boolean(specialist.get("success")), comparison_passed, content_passed
        )
        if bucket not in {"passed", "completed_wrong"}:
            quality_passed = False
    identity, identity_note = _resource_identity(benchmark, specialist)
    requested_context = _number(_first(
        specialist.get("requested_context_max_tokens"), benchmark.get("context_max_tokens")
    ))
    actual_context = _number(_first(
        specialist.get("actual_context_max_tokens"), benchmark.get("actual_context_max_tokens")
    ))
    controls = {
        "model": _text(value("model")),
        "runtime": _text(value("runtime")),
        "backend": _text(value("backend")),
        "weight_quant": _text(value("weight_quant")),
        "draft_weight_quant": _text(value("draft_weight_quant")),
        "kv_cache": value("kv_cache"),
        "source_config": _text(value("source_config")),
        "specialist_profile": _text(value("specialist_profile")),
        "requested_context_tokens": requested_context,
        "actual_context_tokens": actual_context,
        "output_token_limit": _number(value("output_token_limit")),
        "request_timeout_seconds": _number(value("request_timeout_seconds")),
        "thinking": _text(value("thinking")),
        "chat_template_thinking": _text(value("chat_template_thinking")),
        "reasoning_budget_tokens_requested": _number(value("reasoning_budget_tokens_requested")),
        "reasoning_budget_tokens_config_default": _number(value("reasoning_budget_tokens_config_default")),
        "native_thinking_budget_tokens_requested": _number(value("native_thinking_budget_tokens_requested")),
        "temperature": _number(value("temperature")),
        "strata_pcie_frac": _number(value("strata_pcie_frac")),
        "strata_aux_cpus": _number(value("strata_aux_cpus")),
        "sampling_config": _mapping(value("sampling_config")) or {},
        "sampler_overrides": _mapping(value("sampler_overrides")) or {},
        "sampler_recipe_label": _text(value("sampler_recipe_label")),
        "sampler_recipe_sha256": _text(value("sampler_recipe_sha256")),
        "runtime_options": _mapping(value("runtime_options")) or {},
        "diagnostic_variant": _text(value("diagnostic_variant")),
    }
    requested_native_runtime_controls = _mapping(
        value("requested_native_runtime_controls")
    )
    if requested_native_runtime_controls:
        # Requested config argv is part of the comparison control surface, but
        # does not establish effective cache or device residency.
        controls["requested_native_runtime_controls"] = requested_native_runtime_controls
    source_reports = [name for name, row in (("benchmark", benchmark), ("specialist", specialist)) if row]
    attempt = {
        "dashboard_record_id": f"request-{ordinal:06d}",
        "call_id": call_id,
        "run_id": _text(value("run_id")),
        "started_at": _text(value("started_at")),
        "finished_at": _text(value("finished_at")),
        "profile_id": profile_id,
        "profile_id_source": "normalized_report" if profile_id else "missing",
        "model": controls["model"],
        "runtime": controls["runtime"],
        "backend": controls["backend"],
        "weight_quant": controls["weight_quant"],
        "suite": suite or "unclassified",
        "role": role,
        "case_id": _text(value("case_id")),
        "parent_case_id": _text(value("parent_case_id")),
        "case_version": value("case_version"),
        "case_sha256": _text(value("case_sha256")),
        "repeat_index": _number(value("repeat_index")),
        "outcome": outcome,
        "outcome_bucket": bucket,
        "completion_success": completion,
        "quality_passed": quality_passed,
        "content_passed": content_passed,
        "comparison_content_passed": comparison_passed,
        "raw_json_object": _boolean(benchmark.get("raw_json_object")),
        "mechanical_constraints_passed": _boolean(specialist.get("mechanical_constraints_passed")),
        "review_status": review["status"],
        "review_score": review["score"],
        "review_max_points": review["max_points"],
        "reviewer_label": review["reviewer_label"],
        "input_tokens": _number(_first(specialist.get("input_tokens"), benchmark.get("api_input_tokens"))),
        "output_tokens": _number(_first(specialist.get("output_tokens"), benchmark.get("completion_tokens"))),
        "reasoning_tokens_reported": _number(_first(
            specialist.get("reasoning_tokens_reported"), benchmark.get("reasoning_tokens")
        )),
        "native_prompt_seconds": _number(value("native_prompt_seconds")),
        "native_decode_n": _number(_first(specialist.get("native_decode_tokens"), benchmark.get("native_decode_n"))),
        "native_decode_seconds": _number(value("native_decode_seconds")),
        "native_decode_tps": _number(value("native_decode_tps")),
        "native_load_seconds": _number(value("native_load_seconds")),
        "wall_seconds": _number(value("wall_seconds")),
        "ttft_reasoning_seconds": _number(value("ttft_reasoning_seconds")),
        "ttft_visible_seconds": _number(_first(
            specialist.get("ttft_visible_seconds"), benchmark.get("ttft_final_seconds")
        )),
        "speed_eligible": _boolean(value("speed_eligible")),
        "latency_eligible": _first(
            _boolean(benchmark.get("latency_eligible")), _boolean(specialist.get("speed_eligible"))
        ),
        "objective_count": _number(value("objective_count")),
        "objectives_passed": _number(value("objectives_passed")),
        "assertion_count": _number(value("assertion_count")),
        "assertions_passed": _number(value("assertions_passed")),
        "hidden_test_count": _number(benchmark.get("hidden_test_count")),
        "hidden_tests_passed": _number(benchmark.get("hidden_tests_passed")),
        "hidden_failure_labels": (
            list(benchmark["hidden_failure_labels"])
            if isinstance(benchmark.get("hidden_failure_labels"), list) else []
        ),
        "requested_context_tokens": requested_context,
        "actual_context_tokens": actual_context,
        "output_token_limit": controls["output_token_limit"],
        "request_timeout_seconds": controls["request_timeout_seconds"],
        "thinking": controls["thinking"],
        "chat_template_thinking": controls["chat_template_thinking"],
        "reasoning_budget_tokens_requested": controls["reasoning_budget_tokens_requested"],
        "reasoning_budget_tokens_config_default": controls["reasoning_budget_tokens_config_default"],
        "native_thinking_budget_tokens_requested": controls["native_thinking_budget_tokens_requested"],
        "temperature": controls["temperature"],
        "sampling_seed": _number(value("sampling_seed")),
        "sampling_config": controls["sampling_config"],
        "sampler_overrides": controls["sampler_overrides"],
        "sampler_recipe_label": controls["sampler_recipe_label"],
        "sampler_recipe_sha256": controls["sampler_recipe_sha256"],
        "runtime_options": controls["runtime_options"],
        "controls": controls,
        "source_file": _text(value("source_file")),
        "source_line": _number(value("source_line")),
        "benchmark_source_file": _text(benchmark.get("source_file")),
        "benchmark_source_line": _number(benchmark.get("source_line")),
        "specialist_source_file": _text(specialist.get("source_file")),
        "source_config": controls["source_config"],
        "source_config_sha256": _text(value("source_config_sha256")),
        "case_source_sha256": _text(value("case_source_sha256")),
        "prompt_sha256": _text(value("prompt_sha256")),
        "visible_text_sha256": _text(specialist.get("visible_text_sha256")),
        "model_session_run_id": identity["run_id"] if identity else _text(value("model_session_run_id")),
        "lifecycle_tag": identity["tag"] if identity else None,
        "resource_identity": identity,
        "resource_identity_note": identity_note,
        "source_reports": source_reports,
        "_grader_contract_identity": _unambiguous_contract_identity(benchmark, specialist),
        "_grader_contract_attempt_identity": _unambiguous_contract_attempt_identity(
            benchmark, specialist
        ),
    }
    if requested_native_runtime_controls:
        attempt["requested_native_runtime_controls"] = requested_native_runtime_controls
    return attempt


def _merge_attempts(
    benchmark_rows: list[dict[str, Any]],
    specialist_rows: list[dict[str, Any]],
    review_label: str | None,
    gaps: list[str],
) -> list[dict[str, Any]]:
    benchmark_by_call: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in benchmark_rows:
        call_id = _text(row.get("call_id"))
        if call_id:
            benchmark_by_call[call_id].append(row)
    specialist_by_call: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in specialist_rows:
        call_id = _text(row.get("call_id"))
        if call_id:
            specialist_by_call[call_id].append(row)
    for call_id, rows in benchmark_by_call.items():
        if len(rows) > 1:
            gaps.append(f"benchmark report has {len(rows)} rows for call_id {call_id}; specialist overlay was not guessed")
    for call_id, rows in specialist_by_call.items():
        if len(rows) > 1:
            gaps.append(f"specialist report has {len(rows)} rows for call_id {call_id}; review overlay is ambiguous")

    merged: list[dict[str, Any]] = []
    overlaid: set[str] = set()
    ordinal = 1
    for row in benchmark_rows:
        call_id = _text(row.get("call_id"))
        specialist = None
        if call_id and len(benchmark_by_call[call_id]) == 1 and len(specialist_by_call.get(call_id, [])) == 1:
            specialist = specialist_by_call[call_id][0]
            overlaid.add(call_id)
        merged.append(_normalize_attempt(row, specialist, ordinal=ordinal, review_label=review_label))
        ordinal += 1
    for row in specialist_rows:
        call_id = _text(row.get("call_id"))
        if call_id and call_id in overlaid:
            continue
        if call_id and call_id in benchmark_by_call:
            continue
        merged.append(_normalize_attempt(None, row, ordinal=ordinal, review_label=review_label))
        ordinal += 1
        gaps.append(f"specialist attempt {call_id or ordinal - 1} lacks a unique benchmark row and profile overlay")
    return merged


def _resource_value(resource: Mapping[str, Any], metric: str, field: str = "sampled_max") -> Any:
    metrics = resource.get("gpu_metrics")
    value = metrics.get(metric) if isinstance(metrics, Mapping) else None
    return _number(value.get(field)) if isinstance(value, Mapping) else None


def _normalize_resource(row: Mapping[str, Any], ordinal: int) -> dict[str, Any]:
    identity = row.get("identity") if isinstance(row.get("identity"), Mapping) else {}
    process_vram = row.get("process_vram") if isinstance(row.get("process_vram"), Mapping) else {}
    brake = row.get("hardware_power_brake") if isinstance(row.get("hardware_power_brake"), Mapping) else {}
    return {
        "resource_record_id": f"resource-{ordinal:05d}",
        "identity": {"tag": _text(identity.get("tag")), "run_id": _text(identity.get("run_id"))},
        "model_name": _text(row.get("model_name")),
        "runtime_backend": _text(row.get("runtime_backend")),
        "lifecycle_context_tokens": _number(row.get("context_tokens")),
        "initial_load_seconds": _number(row.get("initial_load_seconds")),
        "server_pid": _number(row.get("server_pid")),
        "server_started_at": _text(row.get("server_started_at")),
        "server_stopped_at": _text(row.get("server_stopped_at")),
        "interval_end_at": _text(row.get("interval_end_at")),
        "interval_status": _text(row.get("interval_status")),
        "join_status": _text(row.get("join_status")),
        "process_vram_sample_count": _number(process_vram.get("sample_count")),
        "process_vram_sampled_max_mib": _number(process_vram.get("sampled_max")),
        "process_vram_median_mib": _number(process_vram.get("median")),
        "gpu_utilization_sampled_max_percent": _resource_value(row, "utilization_percent"),
        "board_power_sampled_max_watts": _resource_value(row, "board_power_watts"),
        "temperature_sampled_max_celsius": _resource_value(row, "temperature_celsius"),
        "sm_clock_sampled_max_mhz": _resource_value(row, "sm_clock_mhz"),
        "memory_clock_sampled_max_mhz": _resource_value(row, "memory_clock_mhz"),
        "hardware_power_brake": {
            "sample_count": _number(brake.get("sample_count")),
            "values": list(brake.get("values") or []),
            "observations": list(brake.get("observations") or []),
        },
        "cleanup": _mapping(row.get("cleanup")) or {},
        "notes": [str(value) for value in row.get("notes", []) if isinstance(value, str)],
        "lifecycle_source": _text(row.get("lifecycle_source")),
        "telemetry_sources": [str(value) for value in row.get("telemetry_sources", []) if isinstance(value, str)],
        "statistics_scope": "per_server_run_sampled_statistics_not_per_request_peak",
        "matched_attempt_count": 0,
    }


def _timestamp(value: Any) -> datetime | None:
    text = _text(value)
    if text is None:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed.astimezone(timezone.utc) if parsed.tzinfo is not None else None


def _source_tag_matches(
    attempt: Mapping[str, Any], resources: Sequence[Mapping[str, Any]]
) -> list[Mapping[str, Any]]:
    """Resolve an absent run id only from an exact, unique lifecycle-tagged artifact name.

    Study artifacts append a phase suffix to the owning model-session tag.  A match is
    accepted only when the longest tag is a delimiter-bounded filename prefix, model
    and configured context agree when present, and a completed server interval contains
    the request.  An ongoing snapshot has no known upper bound after ``interval_end_at``.
    """
    sources = {
        Path(value).name.removesuffix(".jsonl")
        for key in ("source_file", "benchmark_source_file", "specialist_source_file")
        if (value := _text(attempt.get(key)))
    }
    if not sources:
        return []
    tagged: list[Mapping[str, Any]] = []
    for resource in resources:
        tag = _text(resource.get("identity", {}).get("tag"))
        if tag and any(stem == tag or stem.startswith(tag + "-") or stem.startswith(tag + "_")
                       for stem in sources):
            tagged.append(resource)
    if not tagged:
        return []
    longest = max(len(str(resource["identity"]["tag"])) for resource in tagged)
    tagged = [resource for resource in tagged if len(str(resource["identity"]["tag"])) == longest]

    model = _text(attempt.get("model"))
    context = _number(_first(attempt.get("actual_context_tokens"), attempt.get("requested_context_tokens")))
    started = _timestamp(attempt.get("started_at"))
    finished = _timestamp(attempt.get("finished_at"))
    if started is None or finished is None or finished < started:
        return []
    matches: list[Mapping[str, Any]] = []
    for resource in tagged:
        resource_model = _text(resource.get("model_name"))
        resource_context = _number(resource.get("lifecycle_context_tokens"))
        if model and resource_model and model != resource_model:
            continue
        if context is not None and resource_context is not None and context != resource_context:
            continue
        interval_start = _timestamp(resource.get("server_started_at"))
        interval_stop = _timestamp(resource.get("server_stopped_at"))
        if interval_start is None or started < interval_start:
            continue
        if resource.get("server_stopped_at") and interval_stop is None:
            continue
        if interval_stop and (started > interval_stop or finished > interval_stop):
            continue
        matches.append(resource)
    return matches


def _attach_resources(
    attempts: list[dict[str, Any]], resources: list[dict[str, Any]] | None, gaps: list[str]
) -> list[dict[str, Any]]:
    normalized = [_normalize_resource(row, index) for index, row in enumerate(resources or [], 1)]
    by_identity: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for resource in normalized:
        identity = resource["identity"]
        if identity["tag"] and identity["run_id"]:
            by_identity[(identity["tag"], identity["run_id"])].append(resource)
        else:
            gaps.append(f"resource record {resource['resource_record_id']} lacks an exact tag/run_id identity")
    if resources is None:
        gaps.append("resource report is absent; GPU and startup resource fields remain null")
    for identity, rows in by_identity.items():
        if len(rows) > 1:
            gaps.append(f"resource identity {identity[0]}/{identity[1]} appears {len(rows)} times and is ambiguous")
    for attempt in attempts:
        identity = attempt.get("resource_identity")
        resource = None
        if resources is None:
            join_status = "resource_report_absent"
        elif identity is None:
            matches = _source_tag_matches(attempt, normalized)
            if len(matches) == 1:
                resource = matches[0]
                resource["matched_attempt_count"] += 1
                join_status = "exact_source_tag_match"
                resolved = dict(resource["identity"])
                attempt["resource_identity"] = resolved
                attempt["lifecycle_tag"] = resolved["tag"]
                attempt["model_session_run_id"] = resolved["run_id"]
                attempt["resource_identity_note"] = (
                    "exact lifecycle identity resolved from a delimiter-bounded normalized "
                    "source filename tag and one compatible resource run; model, context, "
                    "and any completed lifecycle interval were checked"
                )
            elif len(matches) > 1:
                join_status = "ambiguous_source_tag_match"
            else:
                join_status = "identity_unavailable"
        else:
            matches = by_identity.get((identity["tag"], identity["run_id"]), [])
            if len(matches) == 1:
                resource = matches[0]
                resource["matched_attempt_count"] += 1
                join_status = "exact_identity_match"
            elif len(matches) > 1:
                join_status = "ambiguous_resource_identity"
            else:
                join_status = "no_exact_resource_match"
        attempt["resource_join_status"] = join_status
        attempt["resource_scope"] = (
            "per_server_run_sampled_statistics_not_per_request_peak" if resource else None
        )
        attempt["resource_interval_status"] = resource["interval_status"] if resource else None
        attempt["resource_record_id"] = resource["resource_record_id"] if resource else None
        attempt["resource_lifecycle_context_tokens"] = (
            resource["lifecycle_context_tokens"] if resource else None
        )
        attempt["resource_initial_load_seconds"] = resource["initial_load_seconds"] if resource else None
        attempt["resource_process_vram_sampled_max_mib"] = (
            resource["process_vram_sampled_max_mib"] if resource else None
        )
        attempt["resource_gpu_utilization_sampled_max_percent"] = (
            resource["gpu_utilization_sampled_max_percent"] if resource else None
        )
        attempt["resource_board_power_sampled_max_watts"] = (
            resource["board_power_sampled_max_watts"] if resource else None
        )
        attempt["resource_temperature_sampled_max_celsius"] = (
            resource["temperature_sampled_max_celsius"] if resource else None
        )
        attempt["resource_hardware_power_brake"] = resource["hardware_power_brake"] if resource else None
        attempt["resource_notes"] = list(resource["notes"]) if resource else []
        if (resource and attempt["actual_context_tokens"] is not None
                and resource["lifecycle_context_tokens"] is not None):
            if attempt["actual_context_tokens"] != resource["lifecycle_context_tokens"]:
                attempt["resource_notes"].append(
                    "normalized attempt actual context differs from exact-identity lifecycle/configured context"
                )
    return normalized


def _underlying_identity(row: Mapping[str, Any]) -> tuple[Any, ...]:
    task = _text(row.get("parent_case_id")) or _text(row.get("case_id"))
    if task:
        return (row.get("suite"), task)
    return (row.get("suite"), row.get("case_version"), row.get("prompt_sha256"), row.get("call_id"))


def _variant_identity(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (row.get("case_id"), row.get("case_version"), row.get("prompt_sha256"),
            (row.get("controls") or {}).get("diagnostic_variant"))


def _observed(items: Sequence[Mapping[str, Any]], key: str, unit: str) -> dict[str, Any]:
    values = [float(value) for item in items if (value := _number(item.get(key))) is not None]
    return {"unit": unit, "sum": sum(values) if values else None, "sample_count": len(values),
            "missing_count": len(items) - len(values)}


def _groups(attempts: list[dict[str, Any]], gaps: list[str]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    missing_profiles: set[str] = set()
    for item in attempts:
        signature = _stable_id(item["controls"])
        profile_id = item["profile_id"]
        if profile_id is None:
            profile_key = "missing-profile-" + signature
            item["profile_id_source"] = "missing_separated_by_controls"
            missing_profiles.add(profile_key)
        else:
            profile_key = profile_id
        item["profile_key"] = profile_key
        grouped[(profile_key, item["suite"], item["role"], signature)].append(item)
    for profile in sorted(missing_profiles):
        gaps.append(f"{profile} has no normalized profile_id; it was isolated by exact control metadata")
    profile_shapes: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    for profile, suite, role, signature in grouped:
        profile_shapes[(profile, suite, role)].add(signature)
    for (profile, suite, role), shapes in profile_shapes.items():
        if len(shapes) > 1 and not profile.startswith("missing-profile-"):
            gaps.append(
                f"profile_id {profile} has {len(shapes)} distinct control sets in {suite}/{role}; groups were split"
            )

    summaries: list[dict[str, Any]] = []
    for (profile_key, suite, role, signature), items in sorted(grouped.items()):
        outcome_counts = Counter(str(item["outcome_bucket"]) for item in items)
        underlying: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
        for item in items:
            underlying[_underlying_identity(item)].append(item)
        quality_applicable = role != "prose"
        quality_passes = sum(item["quality_passed"] is True for item in items) if quality_applicable else None
        tasks_with_pass = (
            sum(any(item["quality_passed"] is True for item in rows) for rows in underlying.values())
            if quality_applicable else None
        )
        input_tokens = _observed(items, "input_tokens", "tokens")
        output_tokens = _observed(items, "output_tokens", "tokens")
        wall = _observed(items, "wall_seconds", "seconds")
        latency_values = [
            float(value) for item in items
            if item.get("latency_eligible") is True
            and (value := _number(item.get("wall_seconds"))) is not None
        ]
        decode_eligible = [item for item in items if item.get("speed_eligible") is True]
        decode_pairs = [
            (float(tokens), float(seconds)) for item in decode_eligible
            if (tokens := _number(item.get("native_decode_n"))) is not None
            and (seconds := _number(item.get("native_decode_seconds"))) is not None
            and float(seconds) > 0
        ]
        decode_tokens = sum(tokens for tokens, _ in decode_pairs)
        decode_seconds = sum(seconds for _, seconds in decode_pairs)
        reviews = [item for item in items if item["review_status"] == "reviewed"]
        assertion_counts = [float(item["assertion_count"]) for item in items
                            if item["assertion_count"] is not None]
        assertion_passes = [float(item["assertions_passed"]) for item in items
                            if item["assertions_passed"] is not None]
        resources = sorted({item["resource_record_id"] for item in items if item.get("resource_record_id")})
        resource_observations = []
        for resource_id in resources:
            item = next(value for value in items if value.get("resource_record_id") == resource_id)
            resource_observations.append({
                "resource_record_id": resource_id,
                "scope": item.get("resource_scope"),
                "interval_status": item.get("resource_interval_status"),
                "lifecycle_context_tokens": item.get("resource_lifecycle_context_tokens"),
                "initial_load_seconds": item.get("resource_initial_load_seconds"),
                "process_vram_sampled_max_mib": item.get("resource_process_vram_sampled_max_mib"),
                "gpu_utilization_sampled_max_percent": item.get(
                    "resource_gpu_utilization_sampled_max_percent"
                ),
                "board_power_sampled_max_watts": item.get(
                    "resource_board_power_sampled_max_watts"
                ),
                "hardware_power_brake": item.get("resource_hardware_power_brake"),
            })
        group_id = "group-" + _stable_id([profile_key, suite, role, signature])
        for item in items:
            item["group_id"] = group_id
        summaries.append({
            "group_id": group_id,
            "profile_id": None if profile_key.startswith("missing-profile-") else profile_key,
            "profile_key": profile_key,
            "control_signature": signature,
            "suite": suite,
            "role": role,
            "models": sorted({str(item.get("model") or "unknown") for item in items}),
            "runtimes": sorted({str(item.get("runtime") or "unknown") for item in items}),
            "controls": items[0]["controls"],
            "sampling_seeds": sorted({item["sampling_seed"] for item in items if item["sampling_seed"] is not None}),
            "attempt_count": len(items),
            "variant_case_count": len({_variant_identity(item) for item in items}),
            "independent_underlying_task_count": len(underlying),
            "outcome_counts": {name: outcome_counts.get(name, 0) for name in OUTCOME_BUCKETS},
            "completed_attempt_count": sum(item["completion_success"] is True for item in items),
            "quality_pass_count": quality_passes,
            "quality_attempt_denominator": len(items) if quality_applicable else None,
            "underlying_tasks_with_any_quality_pass": tasks_with_pass,
            "quality_underlying_task_denominator": len(underlying) if quality_applicable else None,
            "content_pass_counts": {
                "true": sum(item["content_passed"] is True for item in items),
                "false": sum(item["content_passed"] is False for item in items),
                "missing": sum(item["content_passed"] is None for item in items),
            },
            "raw_json_format_counts": {
                "true": sum(item["raw_json_object"] is True for item in items),
                "false": sum(item["raw_json_object"] is False for item in items),
                "missing": sum(item["raw_json_object"] is None for item in items),
            },
            "mechanical_constraint_counts": {
                "passed": sum(item["mechanical_constraints_passed"] is True for item in items),
                "failed": sum(item["mechanical_constraints_passed"] is False for item in items),
                "missing": sum(item["mechanical_constraints_passed"] is None for item in items),
            },
            "reviewed_attempt_count": len(reviews),
            "pending_review_attempt_count": sum(item["review_status"] == "pending" for item in items),
            "review_not_eligible_attempt_count": sum(item["review_status"] == "not_eligible" for item in items),
            "review_score_sum": sum(float(item["review_score"]) for item in reviews) if reviews else None,
            "review_max_points_sum": sum(float(item["review_max_points"]) for item in reviews) if reviews else None,
            "input_tokens": input_tokens,
            "output_tokens_including_reasoning": output_tokens,
            "observed_wall_seconds": wall,
            "median_eligible_request_latency_seconds": statistics.median(latency_values) if latency_values else None,
            "latency_eligible_sample_count": len(latency_values),
            "native_decode_tps": {
                "value": decode_tokens / decode_seconds if decode_seconds > 0 else None,
                "unit": "tokens/second",
                "aggregation": "sum(native_decode_n) / sum(native_decode_seconds)",
                "eligible_attempt_count": len(decode_eligible),
                "paired_sample_count": len(decode_pairs),
                "missing_pair_count": len(decode_eligible) - len(decode_pairs),
                "native_decode_n_sum": decode_tokens if decode_pairs else None,
                "native_decode_seconds_sum": decode_seconds if decode_pairs else None,
            },
            "code_assertion_diagnostics": {
                "assertion_count_sum_across_attempts": sum(assertion_counts) if assertion_counts else None,
                "assertion_count_sample_count": len(assertion_counts),
                "assertions_passed_sum_across_attempts": sum(assertion_passes) if assertion_passes else None,
                "assertions_passed_sample_count": len(assertion_passes),
                "note": "Assertions are secondary diagnostics within tasks, not independent problems.",
            },
            "resource_record_ids": resources,
            "resource_observations": resource_observations,
        })
    return summaries


def build_dashboard(
    benchmark_report: Path,
    specialist_report: Path,
    resource_report: Path,
    *,
    generated_at: datetime | None = None,
) -> dict[str, Any]:
    benchmark, benchmark_source = _snapshot_json(
        benchmark_report, required=True, label="benchmark_report"
    )
    specialist, specialist_source = _snapshot_json(
        specialist_report, required=True, label="specialist_report"
    )
    resource, resource_source = _snapshot_json(
        resource_report, required=False, label="resource_report"
    )
    assert benchmark is not None and specialist is not None
    benchmark_rows = _rows(benchmark, "request_rows", "benchmark_report")
    coverage_notes = (
        _rows(benchmark, "not_run", "benchmark_report") if "not_run" in benchmark else []
    )
    specialist_rows = _rows(specialist, "attempts", "specialist_report")
    grader_contract_notes = _grader_contract_notes(specialist)
    resource_rows = _rows(resource, "servers", "resource_report") if resource is not None else None
    review_document = specialist.get("review") if isinstance(specialist.get("review"), Mapping) else {}
    review_label = _text(review_document.get("reviewer_label"))
    gaps: list[str] = []
    attempts = _merge_attempts(benchmark_rows, specialist_rows, review_label, gaps)
    note_match_counts = _attach_grader_contract_notes(attempts, grader_contract_notes)
    for note in grader_contract_notes:
        if isinstance(note.get("attempt_identity"), Mapping):
            note_id = str(note["note_id"])
            if note_match_counts.get(note_id) != 1:
                raise DashboardError(
                    f"grader-contract audit exact attempt identity matched "
                    f"{note_match_counts.get(note_id, 0)} dashboard attempts: {note_id}"
                )
    resources = _attach_resources(attempts, resource_rows, gaps)
    groups = _groups(attempts, gaps)
    exact_join_statuses = {"exact_identity_match", "exact_source_tag_match"}
    missing_resource_attempts = sum(
        item["resource_join_status"] not in exact_join_statuses for item in attempts
    )
    upstream = {
        "benchmark": list(benchmark.get("source_provenance") or []),
        "specialist": list(specialist.get("source_provenance") or []),
        "resource": (resource.get("sources") if resource else None),
    }
    generated_at = generated_at or _now()
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": _iso(generated_at),
        "scope": "Metadata-only dashboard of normalized model request attempts; no raw answers or reasoning are included.",
        "policy_notes": [
            "Different budgets, contexts, samplers, runtimes, and quantization controls are separate profiles and are not a causal model-only comparison.",
            "Requested native runtime controls reproduce explicit config arguments, not measured cache or device residency. gpu_layers=all does not establish entirely GPU-resident execution; n_cpu_moe requests CPU expert blocks.",
            "These are small, generally one-shot suites; repeats and prompt variants do not create new underlying benchmark tasks.",
            "Subjective assistant prose review is blinded to model labels but is not an independent human evaluation.",
            "No pooled global quality rank is calculated across profiles, suites, or roles.",
            "Prose mechanical checks report format constraints only and are not prose-quality passes; missing eligible review is pending, not zero.",
            "Completion-token counts can include reasoning. No hidden-versus-visible token split is inferred.",
            "Native decode TPS is aggregated only as sum(native_decode_n) / sum(native_decode_seconds) for eligible rows with both values; wall-derived TPS is not substituted.",
            "Request wall time is observed answer latency. Model startup/load time is reported separately.",
            "Resource lifecycle context is configured session metadata, not an independent /props observation; normalized actual context remains unknown unless explicitly reported.",
            "Resource statistics cover an owned server run, may come from an ongoing snapshot, and are not exact per-request measurements. Process VRAM sampled_max is an observed sample, not a true peak. GPU board power is neither model energy nor whole-machine power.",
        ],
        "input_sources": [benchmark_source, specialist_source, resource_source],
        "upstream_source_provenance": upstream,
        "grader_contract_notes": grader_contract_notes,
        "gaps": sorted(set(gaps)),
        "inventory": {
            "attempt_count": len(attempts),
            "group_count": len(groups),
            "independent_underlying_task_count": len({_underlying_identity(item) for item in attempts}),
            "independent_task_denominator_sum_across_groups": sum(
                group["independent_underlying_task_count"] for group in groups
            ),
            "resource_record_count": len(resources),
            "resource_exact_match_attempt_count": len(attempts) - missing_resource_attempts,
            "resource_unmatched_attempt_count": missing_resource_attempts,
            "resource_unreferenced_record_count": sum(resource["matched_attempt_count"] == 0 for resource in resources),
        },
        "group_summaries": groups,
        "attempts": attempts,
        "resources": resources,
        "coverage_notes": [
            {
                **{key: note.get(key) for key in (
                    "model", "weight_quant", "runtime", "status", "reason", "source_file"
                )},
                "coverage": {
                    key: _number(note["coverage"].get(key)) for key in (
                        "planned_primary_attempt_slots", "terminal_completed_attempts",
                        "active_request_interrupted_by_user", "remaining_unexecuted_primary_slots",
                    )
                } if isinstance(note.get("coverage"), Mapping) else None,
            }
            for note in coverage_notes
        ],
    }


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return value


def _csv_row(attempt: Mapping[str, Any]) -> dict[str, Any]:
    row = {key: attempt.get(key) for key in CSV_FIELDS}
    row.update({
        "resource_hardware_power_brake_json": attempt.get("resource_hardware_power_brake"),
        "resource_notes_json": attempt.get("resource_notes"),
        "controls_json": attempt.get("controls"),
        "sampling_config_json": attempt.get("sampling_config"),
        "sampler_overrides_json": attempt.get("sampler_overrides"),
        "runtime_options_json": attempt.get("runtime_options"),
        "requested_native_runtime_controls_json": attempt.get(
            "requested_native_runtime_controls"
        ),
        "hidden_failure_labels_json": attempt.get("hidden_failure_labels"),
        "source_reports_json": attempt.get("source_reports"),
        "grader_contract_note_ids_json": attempt.get("grader_contract_note_ids"),
        "supplemental_evaluator_diagnostics_json": attempt.get(
            "supplemental_evaluator_diagnostics"
        ),
    })
    return {key: _csv_value(row.get(key)) for key in CSV_FIELDS}


def _fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _html_json(value: Any) -> str:
    serialized = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return (serialized.replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")
            .replace("\u2028", "\\u2028").replace("\u2029", "\\u2029"))


def _grader_contract_notes_html(notes: Sequence[Mapping[str, Any]]) -> str:
    if not notes:
        return ""
    items = []
    for note in notes:
        evidence = note.get("evidence")
        source_evidence = note.get("source_evidence")
        path = str(evidence.get("path")) if isinstance(evidence, Mapping) else ""
        digest = str(evidence.get("sha256")) if isinstance(evidence, Mapping) else ""
        href = Path(path).name
        if note.get("kind") == "evaluator_environment_limitation":
            heading = "Evaluator environment limitation (posthoc diagnostic; no score adjustment)."
        else:
            heading = "Potential specification ambiguity (supplemental; no score adjustment)."
        source_link = ""
        if isinstance(source_evidence, Mapping):
            source_path = str(source_evidence.get("path") or "")
            source_digest = str(source_evidence.get("sha256") or "")
            source_link = (
                f' <a href="{html.escape(Path(source_path).name, quote=True)}">'
                f"Pinned raw attempt source</a> (SHA-256 {html.escape(source_digest)})."
            )
        items.append(
            f"<li><b>{html.escape(heading)}</b> "
            f"{html.escape(str(note.get('summary') or ''))} "
            f'<a href="{html.escape(href, quote=True)}">Audit evidence</a> '
            f"(SHA-256 {html.escape(digest)}).{source_link}</li>"
        )
    return (
        '<details open id="grader-contract-notes"><summary>Grader-contract qualifications</summary><ul>'
        + "".join(items)
        + "</ul></details>"
    )


def render_html(report: Mapping[str, Any]) -> str:
    inventory = report["inventory"]
    policies = "".join(f"<li>{html.escape(str(note))}</li>" for note in report["policy_notes"])
    gaps = "".join(f"<li>{html.escape(str(note))}</li>" for note in report["gaps"]) or "<li>None recorded.</li>"
    contract_notes = _grader_contract_notes_html(report.get("grader_contract_notes", []))
    resource_rows = []
    for item in report["resources"]:
        identity = item["identity"]
        resource_rows.append("<tr>" + "".join(
            f"<td>{html.escape(_fmt(value))}</td>" for value in (
                identity.get("tag"), identity.get("run_id"), item.get("model_name"),
                item.get("lifecycle_context_tokens"), item.get("initial_load_seconds"),
                item.get("process_vram_sampled_max_mib"),
                item.get("gpu_utilization_sampled_max_percent"),
                item.get("board_power_sampled_max_watts"),
                ", ".join(str(x) for x in item["hardware_power_brake"].get("values", [])),
                item.get("interval_status"),
                item.get("matched_attempt_count"),
            )
        ) + "</tr>")
    resource_body = "".join(resource_rows) or '<tr><td colspan="11">No resource report records.</td></tr>'
    coverage_rows = []
    for note in report.get("coverage_notes", []):
        coverage = note.get("coverage")
        counts = json.dumps(coverage, sort_keys=True) if coverage is not None else None
        coverage_rows.append("<tr>" + "".join(
            f"<td>{html.escape(_fmt(value))}</td>" for value in (
                note.get("model"), note.get("status"), counts,
                note.get("reason"), note.get("source_file"),
            )
        ) + "</tr>")
    coverage_body = "".join(coverage_rows) or '<tr><td colspan="5">None recorded.</td></tr>'
    data = {"attempts": report["attempts"], "groups": report["group_summaries"]}
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Complete model study dashboard</title>
<style>
:root{{--bg:#f5f7fb;--paper:#fff;--ink:#17202a;--muted:#607080;--line:#dbe2ea;--accent:#2457a7;--pass:#2f855a;--wrong:#c05621;--bad:#c53030;--other:#718096}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,-apple-system,sans-serif}}main{{max-width:1600px;margin:auto;padding:24px}}h1{{margin-bottom:4px}}h2{{margin-top:30px}}.lede,.note{{color:var(--muted)}}#grader-contract-notes{{overflow-wrap:anywhere}}.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin:20px 0}}.card{{background:var(--paper);border:1px solid var(--line);border-radius:8px;padding:14px}}.card b{{font-size:24px;display:block}}.filters{{display:flex;flex-wrap:wrap;gap:10px;background:var(--paper);padding:12px;border:1px solid var(--line);position:sticky;top:0;z-index:2}}select,input{{padding:7px;border:1px solid #aab6c3;border-radius:5px;min-width:150px}}.table-wrap{{overflow:auto;max-height:620px;border:1px solid var(--line);background:var(--paper)}}table{{border-collapse:collapse;width:100%;font-size:12px}}th,td{{border-bottom:1px solid var(--line);padding:7px;text-align:left;vertical-align:top;white-space:nowrap}}th{{position:sticky;top:0;background:#edf2f7;z-index:1}}tr:hover{{background:#f7fafc}}canvas{{width:100%;height:360px;background:var(--paper);border:1px solid var(--line)}}code{{font-size:11px}}details{{background:var(--paper);padding:10px;border:1px solid var(--line);margin:8px 0}}.pill{{padding:2px 6px;border-radius:10px;background:#edf2f7}}
</style></head><body><main>
<h1>Complete model study dashboard</h1><p class="lede">Generated {html.escape(str(report['generated_at']))}. Every request attempt stays visible. Groups keep profile, suite, role, budget, sampler, runtime, and quantization controls separate.</p>
<div class="cards"><div class="card"><b>{inventory['attempt_count']}</b>request attempts</div><div class="card"><b>{inventory['group_count']}</b>profile/suite/role groups</div><div class="card"><b>{inventory['resource_exact_match_attempt_count']}</b>exact resource joins</div><div class="card"><b>{inventory['resource_unmatched_attempt_count']}</b>resource gaps</div></div>
<details open><summary>Interpretation policy</summary><ul>{policies}</ul></details><details><summary>Recorded gaps</summary><ul>{gaps}</ul></details>
{contract_notes}
<details><summary>Deferred and unrun work</summary><p class="note">These are coverage notes, not additional request attempts. Completed results remain in the tables. User-interrupted and unexecuted work receives no inferred correctness grade.</p><div class="table-wrap"><table><thead><tr><th>Model</th><th>Status</th><th>Recorded primary coverage</th><th>Reason</th><th>Evidence source</th></tr></thead><tbody>{coverage_body}</tbody></table></div></details>
<div class="filters"><select id="model"><option value="">All models</option></select><select id="suite"><option value="">All suites</option></select><select id="profile"><option value="">All profiles</option></select><select id="role"><option value="">All roles</option></select><select id="control"><option value="">All budgets / modes</option></select><input id="search" placeholder="Filter task, call ID, controls"><span id="shown" class="pill"></span></div>
<h2>Profile, suite, and role summaries</h2><p class="note">Quality denominators retain truncations and request failures. Prose uses completion and blinded rubric coverage; mechanical checks remain separate. Native TPS coverage requires both native decode tokens and seconds. Requested native controls are configured arguments, not measurements of actual placement.</p><div class="table-wrap"><table id="groups"><thead><tr><th>Model</th><th>Suite / role</th><th>Profile</th><th>Attempts</th><th>Underlying tasks</th><th>Outcomes</th><th>Quality pass / attempts</th><th>Prose review</th><th>Input / output tokens</th><th>Wall total / eligible median (s)</th><th>Native TPS / coverage</th><th>Controls</th><th>Requested native runtime controls</th><th>Resources</th></tr></thead><tbody></tbody></table></div>
<h2>Output tokens versus request wall time</h2><p class="note">One dot per filtered request with both values. Output tokens may include reasoning. Colors: <span style="color:#2f855a">passed</span>, <span style="color:#c05621">completed wrong</span>, <span style="color:#805ad5">truncated</span>, <span style="color:#c53030">timeout</span>, <span style="color:#9b2c2c">request failure</span>. This plot does not pool a quality rank.</p><canvas id="scatter" width="1400" height="360"></canvas>
<h2>All request attempts</h2><p class="note">Basic content pass and strict raw-JSON format are separate. Assertion counts are nested diagnostics, not task counts. Startup/resource load is separate from request wall latency. Joined resource values summarize the whole owned server run and are not exact per-request peaks. Contract notes are supplemental and do not adjust strict results.</p><div class="table-wrap"><table id="attempts"><thead><tr><th>Model / runtime / quant</th><th>Suite / role</th><th>Profile</th><th>Task</th><th>Call ID</th><th>Outcome</th><th>Quality</th><th>Contract note</th><th>Posthoc evaluator diagnostic</th><th>Content / raw JSON</th><th>Review</th><th>Input / output tokens</th><th>Wall / visible TTFT s</th><th>Native n / s / TPS</th><th>Context req / actual</th><th>Cap / thinking / T / seed / sampler</th><th>Requested native runtime controls</th><th>Checks</th><th>Server-run VRAM sampled max MiB</th><th>Server load s</th><th>Resource join</th><th>Source</th></tr></thead><tbody></tbody></table></div>
<h2>Owned-server resource observations</h2><p class="note">Every row covers one owned server run; ongoing rows are bounded snapshots. Values are not exact per-request measurements. Lifecycle context comes from session configuration and is not an independent /props measurement. sampled_max is not a true peak. Board power is neither model energy nor whole-machine power. No layer placement is inferred.</p><div class="table-wrap"><table><thead><tr><th>Lifecycle tag</th><th>Run ID</th><th>Model</th><th>Lifecycle/configured context</th><th>Load s</th><th>Server-run process VRAM sampled max MiB</th><th>Server-run GPU util sampled max %</th><th>Server-run board power sampled max W</th><th>HW power brake values</th><th>Interval status</th><th>Matched attempts</th></tr></thead><tbody>{resource_body}</tbody></table></div>
<script id="dashboard-data" type="application/json">{_html_json(data)}</script>
<script>
const DATA=JSON.parse(document.getElementById('dashboard-data').textContent);const $=id=>document.getElementById(id);const filters=['model','suite','profile','role','control'];
const txt=v=>v===null||v===undefined?'—':String(v);const cell=(tr,v)=>{{const td=document.createElement('td');td.textContent=txt(v);tr.appendChild(td)}};
function options(id,values){{for(const value of [...new Set(values.filter(Boolean))].sort()){{const o=document.createElement('option');o.value=value;o.textContent=value;$(id).appendChild(o)}}}}
const controlKey=x=>'ctx '+txt(x.requested_context_tokens)+' | cap '+txt(x.output_token_limit)+' | '+txt(x.thinking)+' | native '+JSON.stringify(x.requested_native_runtime_controls||{{}});options('model',DATA.attempts.map(x=>x.model));options('suite',DATA.attempts.map(x=>x.suite));options('profile',DATA.attempts.map(x=>x.profile_key));options('role',DATA.attempts.map(x=>x.role));options('control',DATA.attempts.map(controlKey));
function matches(x){{if($('model').value&&x.model!==$('model').value)return false;if($('suite').value&&x.suite!==$('suite').value)return false;if($('role').value&&x.role!==$('role').value)return false;if($('profile').value&&x.profile_key!==$('profile').value)return false;if($('control').value&&controlKey(x)!==$('control').value)return false;const q=$('search').value.toLowerCase();return !q||JSON.stringify(x).toLowerCase().includes(q)}}
function renderAttempts(rows){{const body=$('attempts').tBodies[0];body.replaceChildren();for(const x of rows){{const tr=document.createElement('tr');cell(tr,txt(x.model)+' / '+txt(x.runtime)+' / '+txt(x.weight_quant));cell(tr,x.suite+' / '+x.role);cell(tr,x.profile_key);cell(tr,x.case_id+(x.parent_case_id?' ← '+x.parent_case_id:''));cell(tr,x.call_id);cell(tr,x.outcome_bucket);cell(tr,x.quality_passed);cell(tr,(x.grader_contract_note_ids||[]).join(', '));cell(tr,JSON.stringify(x.supplemental_evaluator_diagnostics||[]));cell(tr,txt(x.content_passed)+' / '+txt(x.raw_json_object));cell(tr,x.review_status+(x.review_score!==null?' '+x.review_score+'/'+x.review_max_points:''));cell(tr,txt(x.input_tokens)+' / '+txt(x.output_tokens));cell(tr,txt(x.wall_seconds)+' / '+txt(x.ttft_visible_seconds));cell(tr,txt(x.native_decode_n)+' / '+txt(x.native_decode_seconds)+' / '+txt(x.native_decode_tps));cell(tr,txt(x.requested_context_tokens)+' / '+txt(x.actual_context_tokens));cell(tr,txt(x.output_token_limit)+' / '+txt(x.thinking)+' / '+txt(x.temperature)+' / '+txt(x.sampling_seed)+' / '+txt(x.sampler_recipe_label));cell(tr,JSON.stringify(x.requested_native_runtime_controls||{{}}));cell(tr,txt(x.assertions_passed)+'/'+txt(x.assertion_count)+'; hidden '+txt(x.hidden_tests_passed)+'/'+txt(x.hidden_test_count)+(x.hidden_failure_labels.length?' ['+x.hidden_failure_labels.join(', ')+']':''));cell(tr,x.resource_process_vram_sampled_max_mib);cell(tr,x.resource_initial_load_seconds);cell(tr,x.resource_join_status);cell(tr,txt(x.source_file)+':'+txt(x.source_line));body.appendChild(tr)}}}}
function renderGroups(rows){{const ids=new Set(rows.map(x=>x.group_id));const body=$('groups').tBodies[0];body.replaceChildren();for(const g of DATA.groups.filter(x=>ids.has(x.group_id))){{const tr=document.createElement('tr');const o=Object.entries(g.outcome_counts).filter(x=>x[1]).map(x=>x[0]+':'+x[1]).join(', ');const review=g.role==='prose'?'reviewed '+g.reviewed_attempt_count+', pending '+g.pending_review_attempt_count+(g.review_score_sum!==null?', score '+g.review_score_sum+'/'+g.review_max_points_sum:''):'n/a';cell(tr,g.models.join(', ')+' / '+g.runtimes.join(', '));cell(tr,g.suite+' / '+g.role);cell(tr,g.profile_id||g.profile_key);cell(tr,g.attempt_count);cell(tr,g.independent_underlying_task_count+' ('+g.variant_case_count+' variants)');cell(tr,o);cell(tr,g.quality_pass_count===null?'n/a':g.quality_pass_count+' / '+g.quality_attempt_denominator);cell(tr,review);cell(tr,txt(g.input_tokens.sum)+' / '+txt(g.output_tokens_including_reasoning.sum)+' ('+g.output_tokens_including_reasoning.sample_count+'/'+g.attempt_count+')');cell(tr,txt(g.observed_wall_seconds.sum)+' / '+txt(g.median_eligible_request_latency_seconds));cell(tr,txt(g.native_decode_tps.value)+' ('+g.native_decode_tps.paired_sample_count+'/'+g.native_decode_tps.eligible_attempt_count+')');const c=g.controls;const sampler=c.sampler_recipe_label||JSON.stringify(c.sampling_config);cell(tr,'ctx '+txt(c.requested_context_tokens)+'/'+txt(c.actual_context_tokens)+'; cap '+txt(c.output_token_limit)+'; '+txt(c.thinking)+'; T '+txt(c.temperature)+'; quant '+txt(c.weight_quant)+'; sampler '+txt(sampler));cell(tr,JSON.stringify(c.requested_native_runtime_controls||{{}}));cell(tr,g.resource_observations.map(r=>r.resource_record_id+': server-run ctx '+txt(r.lifecycle_context_tokens)+', sampled max VRAM '+txt(r.process_vram_sampled_max_mib)+' MiB, load '+txt(r.initial_load_seconds)+' s, '+txt(r.interval_status)).join('; '));body.appendChild(tr)}}}}
let scatterRows=[];
function drawWrappedText(ctx,text,x,y,maxWidth,lineHeight){{
 const words=text.split(' ');let line='';
 for(const word of words){{
  const candidate=line?line+' '+word:word;
  if(line&&ctx.measureText(candidate).width>maxWidth){{ctx.fillText(line,x,y);line=word;y+=lineHeight}}
  else line=candidate
 }}
 if(line)ctx.fillText(line,x,y)
}}
function drawScatter(){{
 const c=$('scatter'),ctx=c.getContext('2d');
 const width=c.clientWidth,height=c.clientHeight,dpr=Math.max(1,window.devicePixelRatio||1);
 const backingWidth=Math.max(1,Math.round(width*dpr)),backingHeight=Math.max(1,Math.round(height*dpr));
 if(c.width!==backingWidth)c.width=backingWidth;if(c.height!==backingHeight)c.height=backingHeight;
 ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,width,height);
 const left=80,right=25,top=15,bottom=40,plotWidth=width-left-right,plotHeight=height-top-bottom;
 const rows=scatterRows;
 const pts=rows.filter(x=>Number.isFinite(x.output_tokens)&&Number.isFinite(x.wall_seconds));
 ctx.fillStyle='#607080';ctx.font='13px system-ui';
 ctx.save();ctx.textAlign='right';ctx.fillText('output tokens →',width-right,height-10);ctx.restore();
 ctx.save();ctx.translate(14,height/2);ctx.rotate(-Math.PI/2);ctx.textAlign='center';
 ctx.fillText('request wall seconds →',0,0);ctx.restore();
 if(!pts.length){{drawWrappedText(ctx,'No filtered rows have both token and wall-time observations.',left,top+35,plotWidth,18);return}}
 const maxX=Math.max(...pts.map(x=>x.output_tokens),1),maxY=Math.max(...pts.map(x=>x.wall_seconds),1);
 const colors={{passed:'#2f855a',completed_wrong:'#c05621',truncated:'#805ad5',timeout:'#c53030',request_failure:'#9b2c2c',other:'#718096'}};
 ctx.strokeStyle='#cbd5e0';ctx.strokeRect(left,top,plotWidth,plotHeight);
 for(const p of pts){{
  const x=left+(p.output_tokens/maxX)*plotWidth,y=height-bottom-(p.wall_seconds/maxY)*plotHeight;
  ctx.beginPath();ctx.arc(x,y,4,0,Math.PI*2);ctx.fillStyle=colors[p.outcome_bucket]||colors.other;ctx.fill()
 }}
 ctx.fillStyle='#607080';ctx.fillText('0',left-15,height-bottom+5);
 ctx.save();ctx.textAlign='right';ctx.fillText(String(maxX),width-right,height-22);ctx.restore();
 ctx.save();ctx.textAlign='right';ctx.fillText(String(Math.round(maxY*100)/100),left-8,top+7);ctx.restore()
}}
function scatter(rows){{scatterRows=rows;drawScatter()}}
let scatterResizeFrame=0;function resizeScatter(){{if(scatterResizeFrame)return;scatterResizeFrame=requestAnimationFrame(()=>{{scatterResizeFrame=0;drawScatter()}})}}
if('ResizeObserver'in window)new ResizeObserver(resizeScatter).observe($('scatter'));window.addEventListener('resize',resizeScatter);
function update(){{const rows=DATA.attempts.filter(matches);$('shown').textContent=rows.length+' attempts shown';renderAttempts(rows);renderGroups(rows);scatter(rows)}}for(const id of filters)$(id).addEventListener('change',update);$('search').addEventListener('input',update);update();
</script></main></body></html>"""


def write_dashboard(report: Mapping[str, Any], output_prefix: Path) -> tuple[Path, Path, Path]:
    prefix = output_prefix.expanduser().resolve()
    paths = (Path(str(prefix) + ".json"), Path(str(prefix) + ".csv"), Path(str(prefix) + ".html"))
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise DashboardError("refusing to overwrite dashboard output: " + ", ".join(existing))
    prefix.parent.mkdir(parents=True, exist_ok=True)
    with paths[0].open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    with paths[1].open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(_csv_row(attempt) for attempt in report["attempts"])
    with paths[2].open("x", encoding="utf-8") as handle:
        handle.write(render_html(report))
    return paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark-report", type=Path, default=ROOT / "results" / "model-benchmark-report.json"
    )
    parser.add_argument(
        "--specialist-report", type=Path, default=ROOT / "results" / "specialist-study-report.json"
    )
    parser.add_argument(
        "--resource-report", type=Path, default=ROOT / "results" / "specialist-resource-report.json",
        help="optional normalized resource report; a missing file is recorded as a data gap",
    )
    parser.add_argument(
        "--output-prefix", type=Path, default=ROOT / "results" / "complete-study-dashboard",
        help="output prefix; .json, .csv, and .html are created exclusively",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = build_dashboard(args.benchmark_report, args.specialist_report, args.resource_report)
        json_path, csv_path, html_path = write_dashboard(report, args.output_prefix)
    except (DashboardError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({
        "attempt_count": report["inventory"]["attempt_count"],
        "group_count": report["inventory"]["group_count"],
        "resource_exact_match_attempt_count": report["inventory"]["resource_exact_match_attempt_count"],
        "json_output": str(json_path), "csv_output": str(csv_path), "html_output": str(html_path),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
