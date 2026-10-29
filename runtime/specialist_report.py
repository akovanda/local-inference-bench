#!/usr/bin/env python3
"""Build role-specific specialist reports and blinded prose review packages.

Raw benchmark JSONL and study receipts remain authoritative.  Report generation
does not run candidate code, infer missing token counts, or score prose.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import io
import json
import math
import secrets
import sys
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runtime import final_report, specialist_cases  # noqa: E402


SCHEMA_VERSION = 1
REVIEW_METHOD = "assistant_rubric_assessment_with_model_labels_hidden"
REVIEW_LIMITATION = (
    "Subjective assistant rubric assessment with model labels hidden; small sample, "
    "not an independent human evaluation and not guaranteed to be free of bias."
)
LEDGER_CONTRACT_AUDIT_FILENAME = (
    "specialist-ledger-extra-key-exception-contract-audit-v1.json"
)
LEDGER_CONTRACT_AUDIT_SHA256 = (
    "ce51705356430a678c64917f8ad925381bdb9076502709a811bfdee065897dc3"
)
LEDGER_CONTRACT_NOTE_ID = "ledger-surplus-transfer-key-exception-class-v1"
LEDGER_CONTRACT_NOTE_SUMMARY = (
    "The strict check expects TypeError for surplus transfer keys; this specific "
    "exception class is not explicitly named for that condition in the prompt. "
    "Original strict grades are retained."
)
FROZENSET_DIAGNOSTIC_AUDIT_FILENAME = (
    "flash-T1-pivot-frozenset-environment-diagnostic-v1.json"
)
FROZENSET_DIAGNOSTIC_AUDIT_SHA256 = (
    "898b6ecd2380bd14ab3c2ca4dab705867dc5bc4d49c41ba3c5a351b49b6d99a9"
)
FROZENSET_DIAGNOSTIC_RAW_FILENAME = (
    "specialists-flash-modes-sampling-72k64k-v2-publisher-thinking-code.jsonl"
)
FROZENSET_DIAGNOSTIC_RAW_SHA256 = (
    "2472f4b07771eb76224dd519188b37502d2126793e4f22f8adee29e95e72540e"
)
FROZENSET_DIAGNOSTIC_NOTE_ID = "pivot-frozenset-builtin-whitelist-v1"
FROZENSET_DIAGNOSTIC_SUMMARY = (
    "For this exact saved attempt, the frozen strict evaluator reports 1/9 assertions. "
    "Replaying the unchanged visible Python source in an isolated runner with only the "
    "standard frozenset builtin added produces 9/9. This is a supplemental posthoc "
    "evaluator-environment diagnostic; the original strict grade and all primary metrics "
    "remain unchanged."
)
FROZENSET_ENVIRONMENT_DELTA = (
    "Add Python standard immutable-container builtin frozenset to restricted safe_builtins "
    "dictionary. All source validation, assertions and resource/subprocess restrictions "
    "preserved. No source code changed."
)
ATTEMPT_CONTRACT_IDENTITY_KEYS = (
    "case_id", "case_sha256", "case_version", "case_source_sha256", "prompt_sha256",
    "call_id", "visible_text_sha256",
)
CSV_COLUMNS = (
    "row_type", "source_config", "source_config_sha256", "specialist_profile", "case_group",
    "model", "runtime", "case_id", "case_version", "case_sha256", "case_source_sha256",
    "prompt_sha256", "grader_contract_note_ids", "supplemental_evaluator_diagnostics",
    "call_id", "outcome", "success", "quality_status",
    "thinking", "temperature", "sampling_seed", "sampling_seeds",
    "sampling_config", "sampler_recipe_label", "sampler_recipe_sha256",
    "native_thinking_budget_tokens_requested",
    "ledger_json_schema_path", "ledger_json_schema_sha256", "ledger_json_schema_requested",
    "requested_context_max_tokens", "actual_context_max_tokens", "output_token_limit",
    "request_timeout_seconds",
    "completion_success", "mechanical_constraints_passed", "input_tokens", "output_tokens",
    "reasoning_tokens_reported", "wall_seconds", "ttft_visible_seconds",
    "native_decode_seconds", "native_decode_tps", "attempt_count", "unique_case_count",
    "independent_task_count", "eligible_attempt_count", "completed_attempt_count",
    "failed_completion_attempt_count", "successful_attempt_count",
    "input_tokens_sum", "output_tokens_sum", "wall_seconds_sum",
    "native_reported_decode_tps_weighted_by_native_decode_seconds", "reviewed_attempt_count",
    "review_score_sum", "review_max_points_sum", "study_tag", "study_status",
    "model_session_run_id", "startup_seconds",
    "session_started_at", "server_ready_at", "holding_started_at", "session_hold_seconds",
    "cache_state_note", "background_io_note", "visible_text_sha256",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _number(value: Any) -> int | float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _raw_specialist_rows(results_dir: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for path in sorted(results_dir.glob("*.jsonl")):
        parsed, _ = final_report._jsonl(path)
        for _, value in parsed:
            call_id = value.get("call_id")
            if (
                isinstance(call_id, str)
                and str(value.get("suite") or "").startswith("specialist_")
                and call_id not in rows
            ):
                rows[call_id] = value
    return rows


def _case_map() -> dict[str, specialist_cases.SpecialistCase]:
    return {case.case_id: case for case in specialist_cases.CASES}


def _load_ledger_contract_note(results_dir: Path) -> list[dict[str, Any]]:
    path = results_dir / LEDGER_CONTRACT_AUDIT_FILENAME
    if not path.is_file():
        return []
    digest = final_report._sha256(path)
    if digest != LEDGER_CONTRACT_AUDIT_SHA256:
        raise ValueError(
            f"grader-contract audit hash mismatch for {path}: {digest}"
        )
    audit = _json(path)
    if not isinstance(audit, dict):
        raise ValueError(f"grader-contract audit is not a JSON object: {path}")
    case_identity = audit.get("case_identity")
    identity = {
        "case_id": case_identity.get("case_id") if isinstance(case_identity, dict) else None,
        "case_sha256": (
            case_identity.get("case_sha256") if isinstance(case_identity, dict) else None
        ),
        "case_version": (
            case_identity.get("case_version") if isinstance(case_identity, dict) else None
        ),
        "prompt_sha256": audit.get("prompt_sha256"),
    }
    if any(not isinstance(value, str) or not value for value in identity.values()):
        raise ValueError(f"grader-contract audit lacks an exact case/prompt identity: {path}")
    if (
        audit.get("kind") != "frozen_case_contract_interpretation_caveat"
        or audit.get("status") != "original_strict_grades_preserved"
    ):
        raise ValueError(f"grader-contract audit has an unexpected status: {path}")
    return [{
        "note_id": LEDGER_CONTRACT_NOTE_ID,
        "kind": "potential_specification_ambiguity",
        "scope": "exact_frozen_case_and_prompt_across_all_model_profiles",
        "case_identity": identity,
        "summary": LEDGER_CONTRACT_NOTE_SUMMARY,
        "score_adjustment": False,
        "other_case_failures_explained": False,
        "evidence": {
            "path": final_report._relative(path),
            "sha256": digest,
        },
    }]


def _diagnostic_grade(value: Any, label: str, path: Path) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"grader-contract audit {label} is not an object: {path}")
    grade = {
        key: value.get(key) for key in (
            "outcome", "passed", "validation", "independent_task_count",
            "independent_tasks_passed", "objective_count", "objectives_passed",
            "assertion_count", "assertions_passed",
        )
    }
    if (
        not isinstance(grade["outcome"], str)
        or not isinstance(grade["passed"], bool)
        or not isinstance(grade["validation"], str)
        or any(
            isinstance(grade[key], bool) or not isinstance(grade[key], int)
            for key in (
                "independent_task_count", "independent_tasks_passed", "objective_count",
                "objectives_passed", "assertion_count", "assertions_passed",
            )
        )
    ):
        raise ValueError(f"grader-contract audit {label} has an invalid grade schema: {path}")
    return grade


def _load_frozenset_diagnostic_note(results_dir: Path) -> list[dict[str, Any]]:
    path = results_dir / FROZENSET_DIAGNOSTIC_AUDIT_FILENAME
    if not path.is_file():
        return []
    digest = final_report._sha256(path)
    if digest != FROZENSET_DIAGNOSTIC_AUDIT_SHA256:
        raise ValueError(
            f"grader-contract audit hash mismatch for {path}: {digest}"
        )
    raw_path = results_dir / FROZENSET_DIAGNOSTIC_RAW_FILENAME
    if not raw_path.is_file():
        raise ValueError(f"grader-contract audit pinned raw source is missing: {raw_path}")
    raw_digest = final_report._sha256(raw_path)
    if raw_digest != FROZENSET_DIAGNOSTIC_RAW_SHA256:
        raise ValueError(
            f"grader-contract audit raw source hash mismatch for {raw_path}: {raw_digest}"
        )
    audit = _json(path)
    if not isinstance(audit, dict):
        raise ValueError(f"grader-contract audit is not a JSON object: {path}")
    visible = audit.get("visible_text")
    identity = {
        "case_id": audit.get("case_id"),
        "case_sha256": audit.get("case_sha256"),
        "case_version": audit.get("case_version"),
        "case_source_sha256": audit.get("case_source_sha256"),
        "prompt_sha256": audit.get("prompt_sha256"),
        "call_id": audit.get("call_id"),
        "visible_text_sha256": visible.get("sha256") if isinstance(visible, Mapping) else None,
    }
    if any(not isinstance(value, str) or not value for value in identity.values()):
        raise ValueError(f"grader-contract audit lacks an exact attempt identity: {path}")
    original = _diagnostic_grade(audit.get("original_frozen_grade"), "original_frozen_grade", path)
    replay = _diagnostic_grade(audit.get("original_strict_replay"), "original_strict_replay", path)
    diagnostic = _diagnostic_grade(audit.get("one_builtin_diagnostic"), "one_builtin_diagnostic", path)
    expected_original = {
        "outcome": "wrong_answer", "passed": False, "validation": "accepted",
        "independent_task_count": 1, "independent_tasks_passed": 0,
        "objective_count": 3, "objectives_passed": 0,
        "assertion_count": 9, "assertions_passed": 1,
    }
    expected_diagnostic = {
        "outcome": "passed", "passed": True, "validation": "accepted",
        "independent_task_count": 1, "independent_tasks_passed": 1,
        "objective_count": 3, "objectives_passed": 3,
        "assertion_count": 9, "assertions_passed": 9,
    }
    pins = audit.get("frozen_source_pins")
    runner_hashes = (audit.get("original_runner_sha256"), audit.get("diagnostic_runner_sha256"))
    valid_hashes = all(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
        for value in runner_hashes
    )
    if (
        audit.get("schema_version") != 1
        or audit.get("kind") != "frozen_python_builtin_environment_diagnostic"
        or audit.get("new_inference_requests") != 0
        or audit.get("primary_grades_and_model_outputs_modified") is not False
        or audit.get("only_execution_environment_delta") != FROZENSET_ENVIRONMENT_DELTA
        or not isinstance(visible, Mapping)
        or visible.get("reasoning_excluded") is not True
        or not isinstance(pins, Mapping)
        or pins.get("runtime/specialist_cases.py") != identity["case_source_sha256"]
        or not isinstance(pins.get("bench_cases.py"), str)
        or not valid_hashes
        or runner_hashes[0] == runner_hashes[1]
        or original != replay
        or original != expected_original
        or diagnostic != expected_diagnostic
    ):
        raise ValueError(
            f"grader-contract audit has an unexpected schema, environment delta, or "
            f"unchanged-grade status: {path}"
        )
    compact_original = {
        key: original[key] for key in (
            "outcome", "passed", "objective_count", "objectives_passed",
            "assertion_count", "assertions_passed",
        )
    }
    compact_diagnostic = {
        key: diagnostic[key] for key in (
            "outcome", "passed", "objective_count", "objectives_passed",
            "assertion_count", "assertions_passed",
        )
    }
    return [{
        "note_id": FROZENSET_DIAGNOSTIC_NOTE_ID,
        "kind": "evaluator_environment_limitation",
        "scope": "one_exact_saved_attempt",
        "case_identity": {
            key: identity[key] for key in (
                "case_id", "case_sha256", "case_version", "prompt_sha256",
            )
        },
        "attempt_identity": identity,
        "summary": FROZENSET_DIAGNOSTIC_SUMMARY,
        "score_adjustment": False,
        "other_case_failures_explained": False,
        "supplemental_diagnostic": {
            "note_id": FROZENSET_DIAGNOSTIC_NOTE_ID,
            "status": "posthoc_unchanged_source_environment_diagnostic",
            "source_unchanged": True,
            "primary_grade_preserved": True,
            "new_inference_requests": 0,
            "environment_delta": "safe_builtins_plus_frozenset_only",
            "original_strict": compact_original,
            "supplemental_environment_replay": compact_diagnostic,
        },
        "evidence": {
            "path": final_report._relative(path),
            "sha256": digest,
        },
        "source_evidence": {
            "path": final_report._relative(raw_path),
            "sha256": raw_digest,
        },
    }]


def _load_grader_contract_notes(results_dir: Path) -> list[dict[str, Any]]:
    return (
        _load_ledger_contract_note(results_dir)
        + _load_frozenset_diagnostic_note(results_dir)
    )


def _attach_grader_contract_notes(
    attempts: Sequence[dict[str, Any]], notes: Sequence[Mapping[str, Any]]
) -> dict[str, int]:
    keys = ("case_id", "case_sha256", "case_version", "prompt_sha256")
    match_counts = {str(note["note_id"]): 0 for note in notes}
    for attempt in attempts:
        references = []
        diagnostics = []
        for note in notes:
            exact_attempt = note.get("attempt_identity")
            identity = exact_attempt if isinstance(exact_attempt, Mapping) else note.get("case_identity")
            if not isinstance(identity, Mapping):
                continue
            identity_keys = ATTEMPT_CONTRACT_IDENTITY_KEYS if isinstance(exact_attempt, Mapping) else keys
            if all(
                isinstance(attempt.get(key), str)
                and attempt.get(key) == identity.get(key)
                for key in identity_keys
            ):
                note_id = str(note["note_id"])
                if isinstance(exact_attempt, Mapping):
                    diagnostic = note.get("supplemental_diagnostic")
                    original = (
                        diagnostic.get("original_strict")
                        if isinstance(diagnostic, Mapping) else None
                    )
                    observed = {
                        "outcome": attempt.get("outcome"),
                        "passed": attempt.get("success"),
                        "objective_count": attempt.get("objective_count"),
                        "objectives_passed": attempt.get("objectives_passed"),
                        "assertion_count": attempt.get("assertion_count"),
                        "assertions_passed": attempt.get("assertions_passed"),
                    }
                    if not isinstance(original, Mapping) or observed != dict(original):
                        raise ValueError(
                            f"grader-contract audit original grade does not match report attempt "
                            f"{attempt.get('call_id')}: {note_id}"
                        )
                    diagnostics.append(dict(diagnostic))
                references.append(note_id)
                match_counts[note_id] += 1
        if references:
            attempt["grader_contract_note_ids"] = references
        if diagnostics:
            attempt["supplemental_evaluator_diagnostics"] = diagnostics
    return match_counts


def _load_review(
    path: Path | None,
    attempts: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any] | None]:
    if path is None:
        return {}, None
    document = _json(path)
    if not isinstance(document, dict):
        raise ValueError(f"review artifact is not a JSON object: {path}")
    reviewer_label = document.get("reviewer_label")
    reviews = document.get("reviews")
    if not isinstance(reviewer_label, str) or not reviewer_label.strip():
        raise ValueError("review artifact requires a non-empty reviewer_label")
    if not isinstance(reviews, dict):
        raise ValueError("review artifact requires a reviews object keyed by call_id")
    cases = _case_map()
    accepted: dict[str, dict[str, Any]] = {}
    for call_id, review in reviews.items():
        attempt = attempts.get(call_id)
        if not isinstance(call_id, str) or not isinstance(review, dict) or attempt is None:
            raise ValueError(f"review refers to an unknown or malformed call_id: {call_id!r}")
        if attempt.get("case_group") != "prose":
            raise ValueError(f"review call_id is not a prose attempt: {call_id}")
        if attempt.get("blinded_prose_review_eligible") is not True:
            raise ValueError(f"review call_id is not an eligible completed prose attempt: {call_id}")
        case_id = str(attempt.get("case_id"))
        if review.get("case_id") not in (None, case_id):
            raise ValueError(f"review case_id does not match attempt {call_id}")
        expected_hash = attempt.get("visible_text_sha256")
        if review.get("visible_text_sha256") not in (None, expected_hash):
            raise ValueError(f"review text hash does not match attempt {call_id}")
        case = cases.get(case_id)
        if case is None:
            raise ValueError(f"review uses unknown frozen case: {case_id}")
        expected = {item.criterion_id: item.max_points for item in case.rubric}
        scores = review.get("rubric_scores")
        if not isinstance(scores, list) or len(scores) != len(expected):
            raise ValueError(f"review rubric is incomplete for {call_id}")
        observed: set[str] = set()
        score_sum = 0.0
        for score in scores:
            if not isinstance(score, dict) or not isinstance(score.get("criterion_id"), str):
                raise ValueError(f"review rubric row is malformed for {call_id}")
            criterion_id = score["criterion_id"]
            points = _number(score.get("score"))
            if criterion_id not in expected or criterion_id in observed or points is None:
                raise ValueError(f"review criterion is invalid for {call_id}: {criterion_id!r}")
            if float(points) < 0 or float(points) > expected[criterion_id]:
                raise ValueError(f"review score is outside the frozen rubric for {call_id}")
            if score.get("max_points") not in (None, expected[criterion_id]):
                raise ValueError(f"review max_points changed the frozen rubric for {call_id}")
            observed.add(criterion_id)
            score_sum += float(points)
        total = _number(review.get("total_score"))
        if total is None or not math.isclose(float(total), score_sum, abs_tol=1e-9):
            raise ValueError(f"review total_score does not equal rubric score sum for {call_id}")
        accepted[call_id] = dict(review) | {
            "case_id": case_id,
            "visible_text_sha256": expected_hash,
            "total_score": total,
            "max_points": sum(expected.values()),
        }
    metadata = {
        "source_file": final_report._relative(path),
        "source_sha256": final_report._sha256(path),
        "reviewer_label": reviewer_label,
        "review_method": REVIEW_METHOD,
        "limitations": REVIEW_LIMITATION,
        "source_review_method": document.get("review_method"),
        "source_limitations": document.get("limitations"),
        "reviewed_attempt_count": len(accepted),
    }
    return accepted, metadata


def _attempt_row(
    row: Mapping[str, Any],
    review: Mapping[str, Any] | None,
    raw: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    raw = raw or {}
    raw_request = raw.get("request") if isinstance(raw.get("request"), Mapping) else {}
    raw_identity = {
        "case_id": raw.get("case_id"),
        "case_version": raw.get("case_version"),
        "prompt_sha256": raw_request.get("prompt_sha256"),
    }
    raw_identity_matches = all(
        isinstance(value, str)
        and (not isinstance(row.get(key), str) or row.get(key) == value)
        for key, value in raw_identity.items()
    )
    group = row.get("case_group")
    if group == "code":
        success = row.get("content_passed") if isinstance(row.get("content_passed"), bool) else None
        quality_status = "passed" if success is True else ("failed" if success is False else "unverified")
    else:
        success = None
        quality_status = (
            "failed_completion_not_review_eligible"
            if row.get("blinded_prose_review_eligible") is not True
            else ("reviewed" if review is not None else "unscored_pending_blinded_review")
        )
    result = {
        "row_type": "attempt",
        "source_config": row.get("source_config"),
        "source_config_sha256": row.get("source_config_sha256"),
        "specialist_profile": row.get("specialist_profile"),
        "case_group": group,
        "model": row.get("model"),
        "runtime": row.get("runtime"),
        "case_id": row.get("case_id"),
        "case_version": row.get("case_version"),
        "case_sha256": row.get("case_sha256") or (
            raw.get("case_sha256") if raw_identity_matches else None
        ),
        "case_source_sha256": row.get("case_source_sha256"),
        "prompt_sha256": row.get("prompt_sha256") or (
            raw_identity["prompt_sha256"] if raw_identity_matches else None
        ),
        "call_id": row.get("call_id"),
        "outcome": row.get("outcome"),
        "success": success,
        "completion_success": row.get("speed_eligible") is True,
        "quality_status": quality_status,
        "mechanical_constraints_passed": row.get("mechanical_constraints_passed"),
        "objective_count": row.get("objective_count"),
        "objectives_passed": row.get("objectives_passed"),
        "assertion_count": row.get("assertion_count"),
        "assertions_passed": row.get("assertions_passed"),
        "input_tokens": row.get("api_input_tokens"),
        "output_tokens": row.get("completion_tokens"),
        "reasoning_tokens_reported": row.get("reasoning_tokens"),
        "reasoning_chars_observed": row.get("reasoning_chars"),
        "wall_seconds": row.get("wall_seconds"),
        "ttft_visible_seconds": row.get("ttft_final_seconds"),
        "ttft_reasoning_seconds": row.get("ttft_reasoning_seconds"),
        "native_decode_tokens": row.get("native_decode_n"),
        "native_decode_seconds": row.get("native_decode_seconds"),
        "native_decode_tps": row.get("native_decode_tps"),
        "speed_eligible": row.get("speed_eligible"),
        "coding_objective_eligible": row.get("coding_objective_eligible"),
        "prose_constraint_eligible": row.get("prose_constraint_eligible"),
        "blinded_prose_review_eligible": row.get("blinded_prose_review_eligible"),
        "reasoning_metric_eligible": row.get("reasoning_metric_eligible"),
        "thinking": row.get("thinking"),
        "temperature": row.get("temperature"),
        "sampling_seed": row.get("sampling_seed"),
        "sampling_config": row.get("sampling_config"),
        "sampler_recipe_label": row.get("sampler_recipe_label"),
        "sampler_recipe_sha256": row.get("sampler_recipe_sha256"),
        "sampler_recipe_sources": row.get("sampler_recipe_sources"),
        "sampler_recipe_claim": row.get("sampler_recipe_claim"),
        "sampler_application_evidence": row.get("sampler_application_evidence"),
        "requested_context_max_tokens": row.get("context_max_tokens"),
        "reasoning_controls_requested": row.get("reasoning_controls_requested"),
        "reasoning_observed": row.get("reasoning_observed"),
        "capability_note": row.get("capability_note"),
        "actual_context_max_tokens": row.get("actual_context_max_tokens"),
        "output_token_limit": row.get("output_token_limit"),
        "request_timeout_seconds": row.get("request_timeout_seconds"),
        "study_tag": row.get("study_tag"),
        "study_status": row.get("study_status"),
        "study_receipt": row.get("study_receipt"),
        "model_session_run_id": row.get("model_session_run_id"),
        "session_started_at": row.get("session_started_at"),
        "server_ready_at": row.get("server_ready_at"),
        "holding_started_at": row.get("holding_started_at"),
        "visible_text_path": row.get("visible_text_path"),
        "visible_text_sha256": row.get("visible_text_sha256"),
        "visible_text_chars": row.get("visible_text_chars"),
        "visible_text_bytes": row.get("visible_text_bytes"),
        "source_file": row.get("source_file"),
        "review": dict(review) if review is not None else None,
    }
    if "native_thinking_budget_tokens_requested" in row:
        result["native_thinking_budget_tokens_requested"] = row.get(
            "native_thinking_budget_tokens_requested"
        )
    if "ledger_json_schema_sha256" in row:
        result.update({
            "ledger_json_schema_path": row.get("ledger_json_schema_path"),
            "ledger_json_schema_sha256": row.get("ledger_json_schema_sha256"),
            "ledger_json_schema_requested": row.get("ledger_json_schema_requested"),
        })
    return result


def _observed(items: Sequence[Mapping[str, Any]], key: str) -> dict[str, Any]:
    values = [float(value) for item in items if (value := _number(item.get(key))) is not None]
    return {
        "sum": sum(values) if values else None,
        "sample_count": len(values),
        "missing_count": len(items) - len(values),
    }


def _role_summaries(attempts: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in attempts:
        sampling_config = row.get("sampling_config") if isinstance(row.get("sampling_config"), dict) else {}
        grouped[(
            row.get("source_config"), row.get("specialist_profile"), row.get("case_group"),
            row.get("requested_context_max_tokens"), row.get("actual_context_max_tokens"),
            row.get("output_token_limit"), row.get("request_timeout_seconds"),
            row.get("thinking"), row.get("temperature"),
            row.get("native_thinking_budget_tokens_requested"),
            row.get("ledger_json_schema_sha256"),
            json.dumps(sampling_config, sort_keys=True, separators=(",", ":")),
        )].append(row)
    summaries: list[dict[str, Any]] = []
    for controls, items in sorted(grouped.items(), key=lambda x: tuple(str(v or "") for v in x[0])):
        (
            source_config, profile, group, requested_context, actual_context,
            output_limit, request_timeout, thinking, temperature,
            native_thinking_budget, ledger_json_schema_sha256, sampling_config_json,
        ) = controls
        input_tokens = _observed(items, "input_tokens")
        output_tokens = _observed(items, "output_tokens")
        reasoning_tokens = _observed(items, "reasoning_tokens_reported")
        wall = _observed(items, "wall_seconds")
        ttft = _observed(items, "ttft_visible_seconds")
        weighted = [
            (float(rate), float(seconds))
            for item in items
            if item.get("speed_eligible") is True
            and (rate := _number(item.get("native_decode_tps"))) is not None
            and (seconds := _number(item.get("native_decode_seconds"))) is not None
            and float(seconds) > 0
        ]
        weighted_seconds = sum(seconds for _, seconds in weighted)
        weighted_rate = (
            sum(rate * seconds for rate, seconds in weighted) / weighted_seconds
            if weighted_seconds > 0 else None
        )
        identities = {
            (item.get("case_id"), item.get("case_version"), item.get("case_source_sha256"))
            for item in items
        }
        eligible_identities = {
            (item.get("case_id"), item.get("case_version"), item.get("case_source_sha256"))
            for item in items
            if (group == "code" and item.get("coding_objective_eligible") is True)
        }
        reviews = [item["review"] for item in items if isinstance(item.get("review"), dict)]
        ordered_items = sorted(items, key=lambda x: (str(x.get("case_id")), str(x.get("call_id"))))
        summary = {
            "row_type": "role_summary",
            "source_config": source_config,
            "source_config_sha256": next((x.get("source_config_sha256") for x in items if x.get("source_config_sha256")), None),
            "specialist_profile": profile,
            "case_group": group,
            "requested_context_max_tokens": requested_context,
            "actual_context_max_tokens": actual_context,
            "output_token_limit": output_limit,
            "request_timeout_seconds": request_timeout,
            "thinking": thinking,
            "temperature": temperature,
            "sampling_config": json.loads(sampling_config_json),
            "sampler_recipe_label": next(
                (x.get("sampler_recipe_label") for x in items if x.get("sampler_recipe_label")), None,
            ),
            "sampler_recipe_sha256": next(
                (x.get("sampler_recipe_sha256") for x in items if x.get("sampler_recipe_sha256")), None,
            ),
            "sampling_seeds": sorted({
                value for item in items
                if (value := _number(item.get("sampling_seed"))) is not None
            }),
            "models": sorted({str(x.get("model")) for x in items}),
            "runtimes": sorted({str(x.get("runtime")) for x in items}),
            "attempt_count": len(items),
            "unique_case_count": len(identities),
            "independent_task_count": len(identities) if group == "code" else 0,
            "eligible_independent_task_count": len(eligible_identities),
            "eligible_attempt_count": sum(
                item.get("coding_objective_eligible") is True if group == "code"
                else item.get("blinded_prose_review_eligible") is True
                for item in items
            ),
            "completed_attempt_count": sum(item.get("completion_success") is True for item in items),
            "failed_completion_attempt_count": sum(item.get("completion_success") is not True for item in items),
            "successful_attempt_count": sum(item.get("success") is True for item in items),
            "outcomes": dict(sorted(Counter(str(item.get("outcome")) for item in items).items())),
            "input_tokens_sum": input_tokens["sum"],
            "input_tokens_sample_count": input_tokens["sample_count"],
            "input_tokens_missing_count": input_tokens["missing_count"],
            "output_tokens_sum": output_tokens["sum"],
            "output_tokens_sample_count": output_tokens["sample_count"],
            "output_tokens_missing_count": output_tokens["missing_count"],
            "reasoning_tokens_reported_sum": reasoning_tokens["sum"],
            "reasoning_tokens_reported_sample_count": reasoning_tokens["sample_count"],
            "reasoning_tokens_reported_missing_count": reasoning_tokens["missing_count"],
            "wall_seconds_sum": wall["sum"],
            "wall_seconds_sample_count": wall["sample_count"],
            "wall_seconds_missing_count": wall["missing_count"],
            "ttft_visible_seconds_sum": ttft["sum"],
            "ttft_visible_seconds_sample_count": ttft["sample_count"],
            "ttft_visible_seconds_missing_count": ttft["missing_count"],
            "native_reported_decode_tps_weighted_by_native_decode_seconds": weighted_rate,
            "native_decode_weight_seconds": weighted_seconds if weighted else None,
            "native_decode_weighted_sample_count": len(weighted),
            "reviewed_attempt_count": len(reviews),
            "review_score_sum": sum(float(x["total_score"]) for x in reviews) if reviews else None,
            "review_max_points_sum": sum(float(x["max_points"]) for x in reviews) if reviews else None,
            "per_case": [
                {
                    key: item.get(key) for key in (
                        "case_id", "call_id", "outcome", "completion_success", "success", "quality_status",
                        "mechanical_constraints_passed", "input_tokens", "output_tokens",
                        "requested_context_max_tokens", "actual_context_max_tokens",
                        "output_token_limit", "request_timeout_seconds", "thinking",
                        "temperature", "sampling_seed", "sampling_config",
                        "sampler_recipe_label", "sampler_recipe_sha256",
                        "reasoning_tokens_reported", "wall_seconds", "ttft_visible_seconds",
                        "native_decode_seconds", "native_decode_tps", "speed_eligible", "review",
                    )
                }
                for item in ordered_items
            ],
        }
        for detail, item in zip(summary["per_case"], ordered_items):
            if "native_thinking_budget_tokens_requested" in item:
                detail["native_thinking_budget_tokens_requested"] = item.get(
                    "native_thinking_budget_tokens_requested"
                )
            if "ledger_json_schema_sha256" in item:
                detail["ledger_json_schema_sha256"] = item.get("ledger_json_schema_sha256")
        if any("native_thinking_budget_tokens_requested" in item for item in items):
            summary["native_thinking_budget_tokens_requested"] = native_thinking_budget
        if any("ledger_json_schema_sha256" in item for item in items):
            summary.update({
                "ledger_json_schema_path": next(
                    (item.get("ledger_json_schema_path") for item in items), None,
                ),
                "ledger_json_schema_sha256": ledger_json_schema_sha256,
                "ledger_json_schema_requested": next(
                    (item.get("ledger_json_schema_requested") for item in items), None,
                ),
            })
        summaries.append(summary)
    return summaries


def _startup_observations(results_dir: Path, attempts: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_receipt: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for attempt in attempts:
        if isinstance(attempt.get("study_receipt"), str):
            by_receipt[str(attempt["study_receipt"])].append(attempt)
    observations: list[dict[str, Any]] = []
    for receipt_name, items in sorted(by_receipt.items()):
        receipt_path = Path(receipt_name)
        if not receipt_path.is_absolute():
            receipt_path = ROOT / receipt_path
        receipt = _json(receipt_path)
        if not isinstance(receipt, dict):
            receipt = {}
        startup = receipt.get("startup") if isinstance(receipt.get("startup"), dict) else {}
        session = receipt.get("session") if isinstance(receipt.get("session"), dict) else {}
        cache_note = startup.get("cache_state_note") or receipt.get("cache_state_note")
        background_note = startup.get("background_io_note") or receipt.get("background_io_note")
        first = items[0]
        observations.append({
            "row_type": "startup",
            "study_tag": first.get("study_tag"),
            "study_status": first.get("study_status"),
            "study_receipt": receipt_name,
            "source_config": first.get("source_config"),
            "source_config_sha256": first.get("source_config_sha256"),
            "specialist_profile": first.get("specialist_profile"),
            "model": first.get("model"),
            "runtime": first.get("runtime"),
            "model_session_run_id": first.get("model_session_run_id"),
            "actual_context_max_tokens": first.get("actual_context_max_tokens"),
            "startup_seconds": _number(session.get("initial_load_seconds")) if session else None,
            "session_hold_seconds": _number(session.get("hold_seconds")) if session else None,
            "session_started_at": session.get("session_started_at"),
            "server_ready_at": session.get("server_ready_at"),
            "holding_started_at": session.get("holding_started_at"),
            "cache_state_note": cache_note or (
                "Cache state was not classified in the study receipt; observed start-to-ready is not a cold-load claim."
            ),
            "background_io_note": background_note or (
                "Background I/O was not quantified in the study receipt."
            ),
            "request_wall_excludes_startup": True,
        })
    return observations


def build_report(results_dir: Path, review_path: Path | None = None) -> dict[str, Any]:
    specialist_cases.assert_frozen_suite()
    results_dir = results_dir.resolve()
    combined = final_report.build_report(results_dir)
    normalized = {
        str(row["call_id"]): row
        for row in combined.get("request_rows", [])
        if str(row.get("suite") or "").startswith("specialist_")
    }
    reviews, review_metadata = _load_review(review_path, normalized)
    raw_specialist_rows = _raw_specialist_rows(results_dir)
    attempts = [
        _attempt_row(row, reviews.get(call_id), raw_specialist_rows.get(call_id))
        for call_id, row in sorted(normalized.items(), key=lambda x: (str(x[1].get("started_at") or ""), x[0]))
    ]
    grader_contract_notes = _load_grader_contract_notes(results_dir)
    note_match_counts = _attach_grader_contract_notes(attempts, grader_contract_notes)
    for note in grader_contract_notes:
        if isinstance(note.get("attempt_identity"), Mapping):
            note_id = str(note["note_id"])
            if note_match_counts.get(note_id) != 1:
                raise ValueError(
                    f"grader-contract audit exact attempt identity matched "
                    f"{note_match_counts.get(note_id, 0)} report attempts: {note_id}"
                )
    startup = _startup_observations(results_dir, attempts)
    artifact_overview = final_report._specialist_artifact_overview(results_dir)
    wanted_sources = {
        str(value)
        for row in attempts
        for value in (row.get("source_file"), row.get("study_receipt"))
        if isinstance(value, str)
    }
    wanted_sources.update(
        str(row["source_file"])
        for row in artifact_overview
        if isinstance(row.get("source_file"), str)
    )
    for row in artifact_overview:
        preview = row.get("header_preview")
        if isinstance(preview, dict) and isinstance(preview.get("source_file"), str):
            wanted_sources.add(preview["source_file"])
    provenance = [
        item for item in combined.get("source_provenance", [])
        if str(item.get("path")) in wanted_sources
    ]
    known_paths = {str(item.get("path")) for item in provenance}
    for value in sorted({str(row["source_config"]) for row in attempts if isinstance(row.get("source_config"), str)}):
        path = Path(value)
        if not path.is_absolute():
            path = ROOT / path
        relative = final_report._relative(path)
        if path.is_file() and relative not in known_paths:
            provenance.append({"path": relative, "bytes": path.stat().st_size, "sha256": final_report._sha256(path)})
            known_paths.add(relative)
    if review_path is not None:
        provenance.append({
            "path": final_report._relative(review_path),
            "bytes": review_path.stat().st_size,
            "sha256": final_report._sha256(review_path),
        })
    for note in grader_contract_notes:
        evidence = note.get("evidence")
        if isinstance(evidence, Mapping) and evidence.get("path") not in known_paths:
            audit_path = ROOT / str(evidence["path"])
            provenance.append({
                "path": evidence["path"],
                "bytes": audit_path.stat().st_size,
                "sha256": evidence["sha256"],
            })
            known_paths.add(str(evidence["path"]))
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": _now(),
        "scope": "Frozen specialist code and prose prompts; every attempt is retained and there is no global ranking.",
        "suite": {
            "case_version": specialist_cases.CASE_VERSION,
            "suite_sha256": specialist_cases.FROZEN_SUITE_SHA256,
            "case_count": len(specialist_cases.CASES),
        },
        "policy": {
            "no_global_ranking": True,
            "independent_task_unit": "one unique code prompt; objectives and assertions are nested checks, not extra samples",
            "prose_constraints": "mechanical format/substring flags remain separate from subjective prose quality",
            "review_attribution": REVIEW_LIMITATION,
            "throughput_aggregation": "native reported decode TPS weighted by native decode seconds; API totals are not divided by wall time",
            "startup": "session start-to-ready is reported separately from request wall time; no cold-load claim is inferred",
            "reasoning": "requested controls and observed counters are separate; reasoning tokens appear only when the server reported them",
            "budget_comparison": "role summaries separate requested context, observed session context, output cap, request timeout, template thinking, native thinking budget, response-schema hash, and temperature; seeds remain listed within each group",
        },
        "inventory": {
            "attempt_count": len(attempts),
            "code_attempt_count": sum(x.get("case_group") == "code" for x in attempts),
            "prose_attempt_count": sum(x.get("case_group") == "prose" for x in attempts),
            "reviewed_prose_attempt_count": len(reviews),
            "startup_observation_count": len(startup),
        },
        "review": review_metadata,
        "grader_contract_notes": grader_contract_notes,
        "model_artifact_overview": artifact_overview,
        "attempts": attempts,
        "role_summaries": _role_summaries(attempts),
        "startup_observations": startup,
        "source_provenance": provenance,
    }


def _visible_text(raw: Mapping[str, Any]) -> tuple[str, Path, str]:
    visible = raw.get("visible_text") if isinstance(raw.get("visible_text"), dict) else {}
    path_value = visible.get("path")
    expected_hash = visible.get("sha256")
    if not isinstance(path_value, str) or not isinstance(expected_hash, str):
        raise ValueError(f"prose attempt {raw.get('call_id')} lacks visible text provenance")
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    text = path.read_text(encoding="utf-8")
    actual_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if actual_hash != expected_hash:
        raise ValueError(f"visible text hash mismatch for prose attempt {raw.get('call_id')}")
    return text, path.resolve(), actual_hash


def build_blind_documents(results_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    specialist_cases.assert_frozen_suite()
    results_dir = results_dir.resolve()
    raw_rows = _raw_specialist_rows(results_dir)
    combined = final_report.build_report(results_dir)
    normalized = {
        str(row["call_id"]): row
        for row in combined.get("request_rows", [])
        if row.get("case_group") == "prose" and (_number(row.get("visible_text_chars")) or 0) > 0
    }
    by_case: dict[str, list[tuple[str, Mapping[str, Any], Mapping[str, Any]]]] = defaultdict(list)
    for call_id, row in normalized.items():
        raw = raw_rows.get(call_id)
        if raw is None:
            raise ValueError(f"normalized prose attempt lacks raw JSONL row: {call_id}")
        by_case[str(row.get("case_id"))].append((call_id, row, raw))
    randomizer = secrets.SystemRandom()
    used_labels: set[str] = set()
    mapping_rows: list[dict[str, Any]] = []
    package_cases: list[dict[str, Any]] = []
    cases = _case_map()
    for case_id in sorted(by_case):
        case = cases.get(case_id)
        if case is None or case.group != "prose":
            raise ValueError(f"unknown prose case in benchmark rows: {case_id}")
        candidates = list(by_case[case_id])
        randomizer.shuffle(candidates)
        answers: list[dict[str, Any]] = []
        for call_id, row, raw in candidates:
            label = "answer-" + secrets.token_hex(6)
            while label in used_labels:
                label = "answer-" + secrets.token_hex(6)
            used_labels.add(label)
            text, path, digest = _visible_text(raw)
            review_eligible = row.get("blinded_prose_review_eligible") is True
            answers.append({
                "label": label,
                "text": text,
                "sha256": digest,
                "review_eligible": review_eligible,
            })
            mapping_rows.append({
                "label": label,
                "call_id": call_id,
                "case_id": case_id,
                "case_version": row.get("case_version"),
                "case_source_sha256": row.get("case_source_sha256"),
                "suite_sha256": raw.get("suite_sha256"),
                "model": row.get("model"),
                "runtime": row.get("runtime"),
                "specialist_profile": row.get("specialist_profile"),
                "source_config": row.get("source_config"),
                "source_config_sha256": row.get("source_config_sha256"),
                "study_tag": row.get("study_tag"),
                "study_receipt": row.get("study_receipt"),
                "model_session_run_id": row.get("model_session_run_id"),
                "session_started_at": row.get("session_started_at"),
                "server_ready_at": row.get("server_ready_at"),
                "holding_started_at": row.get("holding_started_at"),
                "actual_context_max_tokens": row.get("actual_context_max_tokens"),
                "startup_seconds": row.get("startup_seconds"),
                "reasoning_controls_requested": row.get("reasoning_controls_requested"),
                "capability_note": row.get("capability_note"),
                "sampling_config": row.get("sampling_config"),
                "sampler_recipe_label": row.get("sampler_recipe_label"),
                "sampler_recipe_sha256": row.get("sampler_recipe_sha256"),
                "sampler_recipe_sources": row.get("sampler_recipe_sources"),
                "sampler_recipe_claim": row.get("sampler_recipe_claim"),
                "sampler_application_evidence": row.get("sampler_application_evidence"),
                "objective_outcome": row.get("outcome"),
                "blinded_prose_review_eligible": review_eligible,
                "visible_text_path": final_report._relative(path),
                "visible_text_sha256": digest,
                "source_file": row.get("source_file"),
            })
        package_cases.append({
            "case_id": case_id,
            "prompt": case.prompt,
            "rubric": [asdict(item) for item in case.rubric],
            "rubric_max_points": sum(item.max_points for item in case.rubric),
            "answers": answers,
        })
    package = {
        "schema_version": SCHEMA_VERSION,
        "review_method": REVIEW_METHOD,
        "limitations": REVIEW_LIMITATION,
        "instructions": (
            "Score each review_eligible opaque answer only against its frozen rubric. Nonempty partial answers "
            "remain visible but are marked ineligible and receive no subjective score. Record scores under the "
            "opaque label; model identities remain in a separate mapping until scoring is complete."
        ),
        "cases": package_cases,
        "label_scores_template": {
            answer["label"]: ({
                "review_status": "score_with_frozen_rubric",
                "rubric_scores": [
                    {
                        "criterion_id": criterion["criterion_id"],
                        "score": None,
                        "max_points": criterion["max_points"],
                        "comment": "",
                    }
                    for criterion in case["rubric"]
                ],
                "total_score": None,
                "comment": "",
            } if answer["review_eligible"] else {
                "review_status": "not_scored_incomplete",
                "rubric_scores": [],
                "total_score": None,
                "comment": "Incomplete response; no subjective prose score.",
            })
            for case in package_cases
            for answer in case["answers"]
        },
    }
    mapping = {
        "schema_version": SCHEMA_VERSION,
        "warning": "Keep sealed until the label-keyed review artifact is written; this file reveals model identities.",
        "suite_sha256": specialist_cases.FROZEN_SUITE_SHA256,
        "entries": sorted(mapping_rows, key=lambda x: x["label"]),
    }
    return package, mapping


def write_blind_package(results_dir: Path, package_path: Path, mapping_path: Path) -> tuple[Path, Path]:
    if package_path.resolve() == mapping_path.resolve():
        raise ValueError("blind package and identity mapping must use different paths")
    package, mapping = build_blind_documents(results_dir)
    package_text = json.dumps(package, indent=2, ensure_ascii=False) + "\n"
    mapping["blind_package_sha256"] = hashlib.sha256(package_text.encode("utf-8")).hexdigest()
    final_report._atomic_write(package_path, package_text)
    final_report._atomic_write(mapping_path, json.dumps(mapping, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    return package_path, mapping_path


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return "" if value is None else value


def render_csv(report: Mapping[str, Any]) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=CSV_COLUMNS, extrasaction="ignore")
    writer.writeheader()
    for row in list(report.get("role_summaries", [])) + list(report.get("attempts", [])) + list(report.get("startup_observations", [])):
        writer.writerow({key: _csv_value(row.get(key)) for key in CSV_COLUMNS})
    return buffer.getvalue()


def _display(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.3f}".rstrip("0").rstrip(".")
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return str(value)


def _table(rows: Iterable[Mapping[str, Any]], columns: Sequence[tuple[str, str]]) -> str:
    heads = "".join(f"<th>{html.escape(label)}</th>" for _, label in columns)
    bodies = []
    for row in rows:
        cells = "".join(f"<td>{html.escape(_display(row.get(key)))}</td>" for key, _ in columns)
        bodies.append(f"<tr>{cells}</tr>")
    return f'<div class="scroll"><table><thead><tr>{heads}</tr></thead><tbody>{"".join(bodies)}</tbody></table></div>'


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
        '<details open class="contract-notes"><summary>Grader-contract qualifications</summary><ul>'
        + "".join(items)
        + "</ul></details>"
    )


def render_html(report: Mapping[str, Any]) -> str:
    artifact_columns = (
        ("model", "Model"), ("packing", "Packing"), ("status", "Status"),
        ("expected_weight_bytes", "Expected bytes"), ("recorded_weight_bytes", "Verified bytes"),
        ("expected_weight_sha256", "Expected SHA-256"),
        ("recorded_weight_sha256", "Verified SHA-256"),
        ("verified", "Payload verified"), ("verification_scope", "Verification scope"),
        ("source_repo", "Source repo"), ("source_revision", "Source revision"),
        ("runtime", "Runtime"), ("runtime_source_revision", "Runtime revision"),
        ("runtime_server_sha256", "Runtime server SHA-256"),
        ("runtime_load_validated", "Runtime load validated"),
        ("all_gpu_fit_claimed", "All-GPU fit claimed"),
        ("header_preview", "Header preview evidence"),
        ("quantization_caveat", "Quantization note"), ("model_card", "Model card"),
        ("source_file", "Receipt"),
    )
    summary_columns = (
        ("source_config", "Source config"), ("specialist_profile", "Profile"), ("case_group", "Role"),
        ("requested_context_max_tokens", "Requested context"),
        ("actual_context_max_tokens", "Actual context"),
        ("output_token_limit", "Output cap"), ("request_timeout_seconds", "Request timeout s"),
        ("thinking", "Thinking"), ("temperature", "Temperature"),
        ("native_thinking_budget_tokens_requested", "Native thinking budget requested"),
        ("ledger_json_schema_sha256", "Ledger schema SHA-256"),
        ("sampler_recipe_label", "Sampler recipe"), ("sampling_config", "Sampling config"),
        ("sampling_seeds", "Seeds"),
        ("attempt_count", "Attempts"), ("unique_case_count", "Unique prompts"),
        ("independent_task_count", "Independent code tasks"), ("eligible_attempt_count", "Eligible"),
        ("successful_attempt_count", "Code successes"), ("outcomes", "Outcomes"),
        ("input_tokens_sum", "Input tokens sum"), ("output_tokens_sum", "Output tokens sum"),
        ("wall_seconds_sum", "Wall s sum"),
        ("native_reported_decode_tps_weighted_by_native_decode_seconds", "Native decode tok/s (decode-second weighted)"),
        ("reviewed_attempt_count", "Reviewed prose"), ("review_score_sum", "Review score sum"),
        ("review_max_points_sum", "Review max sum"),
    )
    attempt_columns = (
        ("model", "Model"), ("specialist_profile", "Profile"), ("case_group", "Role"),
        ("case_id", "Case"), ("outcome", "Objective outcome"), ("success", "Code success"),
        ("grader_contract_note_ids", "Supplemental contract note"),
        ("supplemental_evaluator_diagnostics", "Posthoc evaluator diagnostic"),
        ("requested_context_max_tokens", "Requested context"),
        ("actual_context_max_tokens", "Actual context"),
        ("output_token_limit", "Output cap"), ("request_timeout_seconds", "Request timeout s"),
        ("thinking", "Thinking"), ("temperature", "Temperature"), ("sampling_seed", "Seed"),
        ("native_thinking_budget_tokens_requested", "Native thinking budget requested"),
        ("ledger_json_schema_sha256", "Ledger schema SHA-256"),
        ("ledger_json_schema_requested", "Ledger schema request evidence"),
        ("sampler_recipe_label", "Sampler recipe"), ("sampling_config", "Sampling config"),
        ("completion_success", "Completed response"),
        ("quality_status", "Quality status"), ("mechanical_constraints_passed", "Mechanical prose constraints"),
        ("input_tokens", "Input tokens"), ("output_tokens", "Output tokens"),
        ("reasoning_tokens_reported", "Reported reasoning tokens"), ("wall_seconds", "Wall s"),
        ("ttft_visible_seconds", "Visible TTFT s"), ("native_decode_seconds", "Native decode s"),
        ("native_decode_tps", "Native decode tok/s"), ("review", "Blinded review"),
        ("call_id", "Call ID"),
    )
    startup_columns = (
        ("model", "Model"), ("specialist_profile", "Profile"), ("study_tag", "Study"),
        ("actual_context_max_tokens", "Actual context"), ("startup_seconds", "Observed start-to-ready s"),
        ("session_hold_seconds", "Session hold s"), ("cache_state_note", "Cache note"),
        ("background_io_note", "Background I/O note"),
    )
    review_note = report.get("policy", {}).get("review_attribution")
    contract_notes = _grader_contract_notes_html(report.get("grader_contract_notes", []))
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Specialist study report</title><style>
body{{font:14px system-ui,sans-serif;margin:0;background:#0b1020;color:#e5e7eb}}main{{max-width:1600px;margin:auto;padding:24px}}h1,h2{{color:#f8fafc}}p{{color:#bac7db;max-width:1100px}}.contract-notes{{overflow-wrap:anywhere}}.scroll{{overflow:auto;border:1px solid #334155;border-radius:6px;margin:10px 0 26px}}table{{border-collapse:collapse;width:100%;background:#111827}}th{{position:sticky;top:0;background:#1e293b;white-space:nowrap}}th,td{{padding:7px 9px;border-bottom:1px solid #293548;text-align:left;vertical-align:top;max-width:460px}}tr:hover{{background:#172554}}
</style></head><body><main><h1>Specialist study evidence</h1><p>Generated {html.escape(str(report.get("generated_at")))}. Every row is one attempt; the report makes no global ranking. {html.escape(str(review_note))}</p>
{contract_notes}
<h2>Model and packing overview</h2><p>Payload hashes, header previews, runtime builds, and runtime load observations are separate evidence. A blank all-GPU field means the artifact receipt makes no such claim.</p>{_table(report.get("model_artifact_overview", []), artifact_columns)}
<h2>Role summaries</h2><p>Requested context, actual session context, output cap, request timeout, template thinking, native thinking budget, response-schema hash, temperature, and explicit sampling config define separate comparison groups. Seeds are listed within each group, so repeats do not add independent tasks. Native reported decode rates are weighted by native decode seconds. Startup is excluded and reported separately.</p>{_table(report.get("role_summaries", []), summary_columns)}
<h2>Per-case time, tokens, and outcome</h2><p>Mechanical prose constraints remain separate from blinded subjective review. Missing reported reasoning-token counts remain blank.</p>{_table(report.get("attempts", []), attempt_columns)}
<h2>Startup observations</h2><p>These are observed session start-to-ready values, with receipt-level cache/background qualifications; they are not cold-load claims.</p>{_table(report.get("startup_observations", []), startup_columns)}
</main></body></html>'''


def write_report(report: Mapping[str, Any], prefix: Path) -> list[Path]:
    outputs = [prefix.with_suffix(".json"), prefix.with_suffix(".csv"), prefix.with_suffix(".html")]
    final_report._atomic_write(outputs[0], json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    final_report._atomic_write(outputs[1], render_csv(report))
    final_report._atomic_write(outputs[2], render_html(report))
    return outputs


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--output-prefix", type=Path, default=ROOT / "results" / "specialist-study-report")
    parser.add_argument("--review", type=Path, help="optional call_id-keyed blinded prose review JSON")
    parser.add_argument("--blind-package", type=Path, help="write model-free prose review package JSON")
    parser.add_argument("--blind-mapping", type=Path, help="write separate opaque-label identity mapping JSON")
    args = parser.parse_args(argv)
    if (args.blind_package is None) != (args.blind_mapping is None):
        parser.error("--blind-package and --blind-mapping must be supplied together")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(args.results_dir, args.review)
    outputs = write_report(report, args.output_prefix)
    if args.blind_package is not None:
        outputs.extend(write_blind_package(args.results_dir, args.blind_package, args.blind_mapping))
    print(f"wrote {len(report['attempts'])} specialist attempts to " + ", ".join(str(x) for x in outputs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
