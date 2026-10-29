#!/usr/bin/env python3
"""Append new hard-code diagnostics to a preserved legacy hard report.

The legacy aggregates and comparisons remain baseline-only.  New current-case
Python attempts are exposed through one supplemental metadata group so
``runtime.final_report`` can consume per-call diagnostics without raw answers.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
from datetime import datetime, timezone
import glob
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Callable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import bench_cases  # noqa: E402
from runtime import hard_report  # noqa: E402


PINNED_BASELINE_SHA256 = "22afc4167d7ba0351a6e20098457d69e35b13f575cf010aef264b76fed652256"
SUPPLEMENTAL_GROUP_NAME = "supplemental_new_hard_python_diagnostics"
ELIGIBLE_CASE_IDS = ("hard_weighted_intervals", "hard_temporal_kv")
_CASES = {case.case_id: case for case in bench_cases.HARD_CASES if case.case_id in ELIGIBLE_CASE_IDS}


class ExtensionError(ValueError):
    """An input cannot safely produce the requested preserved-baseline report."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_sha256(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return _sha256_bytes(raw)


_CURRENT_CASE_SOURCE_SHA256 = _sha256_bytes(Path(bench_cases.__file__).read_bytes())


def _sha256_or_none(value: Any) -> str | None:
    if not isinstance(value, str) or len(value) != 64:
        return None
    return value if all(char in "0123456789abcdef" for char in value) else None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.path.expanduser(str(path))))


def _reject_symlink(path: Path, label: str) -> None:
    if path.is_symlink():
        raise ExtensionError(f"{label} must not be a symlink: {path}")


def _reject_symlink_components(path: Path) -> None:
    candidate = _absolute(path)
    for component in (candidate, *candidate.parents):
        if os.path.lexists(component) and component.is_symlink():
            raise ExtensionError(f"output path must not traverse a symlink: {component}")


def resolve_inputs(patterns: Sequence[str], output: Path) -> list[Path]:
    """Resolve explicit result globs while retaining a no-symlink contract."""
    found: list[Path] = []
    missing: list[str] = []
    for pattern in patterns:
        matches = [Path(value) for value in glob.glob(pattern, recursive=True)]
        literal = Path(pattern).expanduser()
        if not matches and os.path.lexists(literal):
            matches = [literal]
        if not matches:
            missing.append(pattern)
            continue
        for match in matches:
            _reject_symlink(match, "input")
            if match.is_file():
                found.append(_absolute(match))
    if missing:
        raise ExtensionError("input pattern matched no files: " + ", ".join(missing))
    unique = sorted(set(found), key=str)
    if not unique:
        raise ExtensionError("no regular input files were found")
    output_absolute = _absolute(output)
    if output_absolute in unique:
        raise ExtensionError("output path must not be an input path")
    return unique


def _load_baseline(path: Path, expected_sha256: str) -> tuple[dict[str, Any], dict[str, Any]]:
    path = _absolute(path)
    _reject_symlink(path, "baseline")
    if not path.is_file():
        raise ExtensionError(f"baseline report does not exist: {path}")
    if len(expected_sha256) != 64 or any(char not in "0123456789abcdef" for char in expected_sha256):
        raise ExtensionError("expected baseline SHA-256 must be 64 lowercase hexadecimal characters")
    raw = path.read_bytes()
    actual = _sha256_bytes(raw)
    if actual != expected_sha256:
        raise ExtensionError(
            f"baseline SHA-256 mismatch: expected {expected_sha256}, observed {actual}"
        )
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ExtensionError(f"baseline report is not valid JSON: {exc}") from exc
    if not isinstance(value, dict) or not isinstance(value.get("groups"), list):
        raise ExtensionError("baseline report must be an object with a groups array")
    if "supplemental_enrichment" in value or any(
        isinstance(group, Mapping) and group.get("group_name") == SUPPLEMENTAL_GROUP_NAME
        for group in value["groups"]
    ):
        raise ExtensionError("baseline already contains this supplemental enrichment")
    return value, {
        "path": str(path), "bytes": len(raw), "sha256": actual,
        "expected_sha256": expected_sha256,
    }


def _baseline_call_ids(baseline: Mapping[str, Any]) -> set[str]:
    call_ids: set[str] = set()
    for group in baseline.get("groups", []):
        if not isinstance(group, Mapping):
            continue
        tasks = group.get("tasks")
        if not isinstance(tasks, list):
            continue
        for task in tasks:
            call_id = task.get("call_id") if isinstance(task, Mapping) else None
            if isinstance(call_id, str) and call_id:
                call_ids.add(call_id)
    return call_ids


def _source_inventory(paths: Sequence[Path]) -> list[dict[str, Any]]:
    inventory = []
    for path in paths:
        _reject_symlink(path, "input")
        raw = path.read_bytes()
        inventory.append({
            "path": str(path), "bytes": len(raw), "sha256": _sha256_bytes(raw),
            "accepted_call_ids": [],
        })
    return inventory


def _prompt_sha256(case: bench_cases.BenchmarkCase) -> str:
    return hashlib.sha256(case.prompt.encode("utf-8")).hexdigest()


def _rescore_fields(row: Mapping[str, Any], diagnostics: Mapping[str, Any]) -> tuple[str, str]:
    all_passed = diagnostics.get("hidden_tests_passed") == diagnostics.get("hidden_test_count")
    current_name_error = any(
        "raised NameError" in str(failure.get("detail", ""))
        for failure in diagnostics.get("failed_tests", [])
        if isinstance(failure, Mapping)
    )
    evaluation = row.get("evaluation") if isinstance(row.get("evaluation"), Mapping) else {}
    original_name_error = "NameError" in str(evaluation.get("detail", ""))
    original_passed = evaluation.get("passed") is True
    outcome = "passed" if all_passed else "wrong_answer"
    if original_name_error and all_passed:
        cause = "recorded_evaluator_builtin_limitation"
    elif all_passed and original_passed:
        cause = "recorded_pass_confirmed_by_hidden_tests"
    elif all_passed:
        cause = "recorded_failure_not_reproduced"
    elif current_name_error:
        cause = "unresolved_name_requires_evaluator_review"
    else:
        cause = "semantic_or_contract_failure"
    return outcome, cause


def _safe_task(
    loaded: Mapping[str, Any], diagnostics: Mapping[str, Any], row_sha256: str
) -> dict[str, Any]:
    row = loaded["value"]
    request = row["request"]
    response = row["response"]
    outcome, cause = _rescore_fields(row, diagnostics)
    return {
        "call_id": row["call_id"],
        "suite": row["suite"],
        "case_id": row["case_id"],
        "case_version": row["case_version"],
        "prompt_sha256": request["prompt_sha256"],
        "case_source_sha256": _sha256_or_none(row.get("case_source_sha256")),
        "source_file": str(loaded["source_file"]),
        "source_line": loaded.get("source_line"),
        "source_row_sha256": row_sha256,
        "response_text_sha256": _sha256_bytes(response["response_text"].encode("utf-8")),
        "original_outcome": hard_report.classify(row),
        "original_evaluation_passed": (
            row["evaluation"].get("passed")
            if isinstance(row["evaluation"].get("passed"), bool) else None
        ),
        "hidden_test_diagnostics": copy.deepcopy(dict(diagnostics)),
        "rescored_outcome": outcome,
        "rescored_cause": cause,
    }


def _scan(
    records: Sequence[Mapping[str, Any]],
    baseline_call_ids: set[str],
    evaluator: Callable[[Mapping[str, Any], str], dict[str, Any] | None],
    *,
    invalid_json_lines: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    skips: Counter[str] = Counter()
    skipped_ids: dict[str, list[str]] = defaultdict(list)
    if invalid_json_lines:
        skips["invalid_json_line"] = invalid_json_lines

    # Detect conflicting call identities before running any restricted evaluation.
    unique: list[tuple[Mapping[str, Any], str]] = []
    seen: dict[str, str] = {}
    for loaded in records:
        row = loaded.get("value")
        if not hard_report._is_benchmark_row(row):
            skips["non_benchmark_object"] += 1
            continue
        call_id = row.get("call_id")
        if not isinstance(call_id, str) or not call_id:
            skips["missing_call_id"] += 1
            continue
        fingerprint = _canonical_sha256(row)
        previous = seen.get(call_id)
        if previous is not None:
            if previous != fingerprint:
                raise ExtensionError(f"conflicting duplicate rows for call_id {call_id}")
            skips["identical_duplicate_call_id"] += 1
            skipped_ids["identical_duplicate_call_id"].append(call_id)
            continue
        seen[call_id] = fingerprint
        unique.append((loaded, fingerprint))

    tasks: list[dict[str, Any]] = []

    def skip(reason: str, call_id: str) -> None:
        skips[reason] += 1
        skipped_ids[reason].append(call_id)

    for loaded, fingerprint in unique:
        row = loaded["value"]
        call_id = row["call_id"]
        if call_id in baseline_call_ids:
            skip("baseline_call_id", call_id)
            continue
        if row.get("suite") != "hard" or row.get("case_id") not in ELIGIBLE_CASE_IDS:
            skip("ineligible_suite_or_case", call_id)
            continue
        case = _CASES[str(row["case_id"])]
        request = row["request"]
        if row.get("case_version") != bench_cases.CASE_VERSION:
            skip("case_version_mismatch", call_id)
            continue
        if request.get("prompt_sha256") != _prompt_sha256(case):
            skip("prompt_sha256_mismatch", call_id)
            continue
        if isinstance(row.get("error"), Mapping):
            skip("request_error", call_id)
            continue
        response = row.get("response")
        if not isinstance(response, Mapping):
            skip("missing_response", call_id)
            continue
        evaluation = row["evaluation"]
        if (response.get("truncated") is True or evaluation.get("truncated") is True
                or response.get("stream_complete") is False):
            skip("truncated_or_incomplete", call_id)
            continue
        response_text = response.get("response_text")
        if not isinstance(response_text, str) or not response_text.strip():
            skip("empty_response_text", call_id)
            continue
        try:
            diagnostics = evaluator(row, response_text)
        except Exception as exc:
            raise ExtensionError(
                f"restricted evaluator failed for call_id {call_id}: {type(exc).__name__}"
            ) from exc
        if not isinstance(diagnostics, Mapping):
            skip("diagnostics_unavailable", call_id)
            continue
        if diagnostics.get("available") is not True:
            skip("evaluator_rejected_source", call_id)
            continue
        if (diagnostics.get("rescored_with_case_version") != bench_cases.CASE_VERSION
                or diagnostics.get("rescored_with_case_source_sha256")
                != _CURRENT_CASE_SOURCE_SHA256):
            raise ExtensionError(f"stale hidden-test diagnostics for call_id {call_id}")
        count = diagnostics.get("hidden_test_count")
        passed = diagnostics.get("hidden_tests_passed")
        if (not isinstance(count, int) or isinstance(count, bool) or count <= 0
                or not isinstance(passed, int) or isinstance(passed, bool)
                or passed < 0 or passed > count):
            raise ExtensionError(f"malformed hidden-test diagnostics for call_id {call_id}")
        tasks.append(_safe_task(loaded, diagnostics, fingerprint))

    tasks.sort(key=lambda task: task["call_id"])
    return tasks, {
        "skip_counts": dict(sorted(skips.items())),
        "skipped_call_ids_by_reason": {
            reason: sorted(set(call_ids)) for reason, call_ids in sorted(skipped_ids.items())
        },
        "unique_benchmark_call_id_count": len(unique),
    }


def build_enriched_report(
    baseline_path: Path,
    input_paths: Sequence[Path],
    *,
    expected_baseline_sha256: str = PINNED_BASELINE_SHA256,
    evaluator: Callable[[Mapping[str, Any], str], dict[str, Any] | None] = hard_report.hidden_test_diagnostics,
) -> dict[str, Any]:
    baseline, baseline_source = _load_baseline(baseline_path, expected_baseline_sha256)
    sources = _source_inventory(input_paths)
    try:
        records, read_stats = hard_report.load_inputs(input_paths)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ExtensionError(f"result input is not readable JSON text: {type(exc).__name__}") from exc
    after = _source_inventory(input_paths)
    if [(item["path"], item["sha256"]) for item in sources] != [
        (item["path"], item["sha256"]) for item in after
    ]:
        raise ExtensionError("an input file changed while it was being scanned")

    tasks, scan = _scan(
        records, _baseline_call_ids(baseline), evaluator,
        invalid_json_lines=int(read_stats.get("invalid_json_lines", 0)),
    )
    by_path: dict[str, list[str]] = defaultdict(list)
    for task in tasks:
        by_path[task["source_file"]].append(task["call_id"])
    for source in sources:
        source["accepted_call_ids"] = sorted(by_path.get(source["path"], []))

    check_count = sum(task["hidden_test_diagnostics"]["hidden_test_count"] for task in tasks)
    checks_passed = sum(task["hidden_test_diagnostics"]["hidden_tests_passed"] for task in tasks)
    outcomes = Counter(task["rescored_outcome"] for task in tasks)
    group = {
        "group_name": SUPPLEMENTAL_GROUP_NAME,
        "group_kind": "supplemental_per_call_diagnostics_only",
        "suite": "hard",
        "eligible_case_ids": list(ELIGIBLE_CASE_IDS),
        "attempt_count": len(tasks),
        "independent_task_count": len({
            (task["case_id"], task["case_version"], task["prompt_sha256"]) for task in tasks
        }),
        "hidden_test_check_count": check_count,
        "hidden_tests_passed": checks_passed,
        "rescored_outcome_counts": {
            "passed": outcomes.get("passed", 0), "wrong_answer": outcomes.get("wrong_answer", 0)
        },
        "new_call_ids": [task["call_id"] for task in tasks],
        "tasks": tasks,
        "aggregation_note": (
            "Per-call diagnostics only; this group is excluded from preserved legacy root "
            "aggregates and comparisons and does not define a cross-profile ranking."
        ),
    }
    helper_path = Path(hard_report.__file__).resolve()
    case_path = Path(bench_cases.__file__).resolve()
    producer_path = Path(__file__).resolve()
    enrichment = {
        "schema_version": 1,
        "generated_at": _utc_now(),
        "baseline_source": baseline_source,
        "baseline_unchanged_invariant": {
            "preserved": True,
            "baseline_root_fields": list(baseline.keys()),
            "baseline_document_canonical_sha256": _canonical_sha256(baseline),
            "baseline_group_count": len(baseline["groups"]),
            "baseline_groups_canonical_sha256": _canonical_sha256(baseline["groups"]),
            "note": (
                "All legacy root aggregates, comparisons, groups, tasks, and values remain "
                "unchanged. Legacy aggregates and comparisons cover baseline groups only."
            ),
        },
        "source_inputs": sources,
        "source_file_count": len(sources),
        "objects_read": read_stats.get("objects_read"),
        "newly_rescored_attempt_count": len(tasks),
        "newly_rescored_hidden_test_check_count": check_count,
        "newly_rescored_hidden_tests_passed": checks_passed,
        "new_call_ids": [task["call_id"] for task in tasks],
        **scan,
        "producer_source": {
            "path": str(producer_path), "sha256": _sha256_bytes(producer_path.read_bytes())
        },
        "helper_source": {"path": str(helper_path), "sha256": _sha256_bytes(helper_path.read_bytes())},
        "current_case_source": {
            "path": str(case_path),
            "sha256": _CURRENT_CASE_SOURCE_SHA256,
            "case_version": bench_cases.CASE_VERSION,
            "prompt_sha256_by_case": {
                case_id: _prompt_sha256(_CASES[case_id]) for case_id in ELIGIBLE_CASE_IDS
            },
        },
    }

    report = copy.deepcopy(baseline)
    report["groups"].append(group)
    report["supplemental_enrichment"] = enrichment
    return report


def write_exclusive_json(path: Path, report: Mapping[str, Any]) -> None:
    path = _absolute(path)
    _reject_symlink_components(path)
    if os.path.lexists(path):
        raise ExtensionError(f"refusing to overwrite existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    _reject_symlink_components(path.parent)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as handle:
            temporary = handle.name
            json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path, follow_symlinks=False)
        os.unlink(temporary)
        temporary = None
        if os.name == "posix":
            descriptor = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    except FileExistsError as exc:
        raise ExtensionError(f"refusing to overwrite existing output: {path}") from exc
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline", type=Path, default=ROOT / "results" / "legacy-hard-round-report.json",
        help="preserved legacy hard-round report",
    )
    parser.add_argument(
        "--expected-baseline-sha256", default=PINNED_BASELINE_SHA256,
        help="required SHA-256 pin for the baseline report",
    )
    parser.add_argument(
        "--inputs", nargs="+", required=True,
        help="frozen result JSON/JSONL paths or quoted glob patterns",
    )
    parser.add_argument("--output", type=Path, required=True, help="new exclusive JSON output path")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        paths = resolve_inputs(args.inputs, args.output)
        report = build_enriched_report(
            args.baseline, paths, expected_baseline_sha256=args.expected_baseline_sha256
        )
        write_exclusive_json(args.output, report)
    except (ExtensionError, OSError) as exc:
        print(f"extend_hard_diagnostics: {exc}", file=sys.stderr)
        return 2
    audit = report["supplemental_enrichment"]
    print(json.dumps({
        "output": str(_absolute(args.output)),
        "newly_rescored_attempt_count": audit["newly_rescored_attempt_count"],
        "newly_rescored_hidden_test_check_count": audit["newly_rescored_hidden_test_check_count"],
        "skipped_count": sum(audit["skip_counts"].values()),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
