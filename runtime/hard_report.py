#!/usr/bin/env python3
"""Build an offline JSON report from benchmark result JSONL files.

Model response code is never copied into the report.  Failed, current hard-suite
code answers may be checked one hidden test at a time, using bench_cases' existing
restricted subprocess evaluator.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import re
import sys
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import urlsplit, urlunsplit


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import bench_cases  # noqa: E402


SCHEMA_VERSION = 1
OUTCOMES = ("passed", "truncated", "timeout", "request_failure", "wrong_answer")
_FENCE_RE = re.compile(
    r"^\s*```(?:json)?\s*\n?(.*?)\n?```\s*$", re.DOTALL | re.IGNORECASE
)
_HARD_BY_ID = {case.case_id: case for case in bench_cases.HARD_CASES}
_CURRENT_CASE_SOURCE_SHA256 = hashlib.sha256(Path(bench_cases.__file__).read_bytes()).hexdigest()


class ReportError(ValueError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _finite_number(value: Any) -> int | float | None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _clean_endpoint(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    parsed = urlsplit(value)
    if not parsed.scheme or not parsed.hostname:
        return value
    host = parsed.hostname
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    netloc = host + (f":{parsed.port}" if parsed.port is not None else "")
    return urlunsplit((parsed.scheme, netloc, parsed.path, "", ""))


def _engine_environment(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    names = ("STRATA_SM75_PERF", "STRATA_ROUTE_TAIL_SKIP", "STRATA_ADAPT_OVERLAP")
    return {
        name: value[name]
        for name in names
        if isinstance(value.get(name), str)
    }


def resolve_inputs(patterns: Sequence[str], output: Path) -> list[Path]:
    files: list[Path] = []
    unmatched: list[str] = []
    for pattern in patterns:
        matches = [Path(value) for value in glob.glob(pattern, recursive=True)]
        if not matches and Path(pattern).is_file():
            matches = [Path(pattern)]
        if not matches:
            unmatched.append(pattern)
            continue
        files.extend(path for path in matches if path.is_file())
    if unmatched:
        raise ReportError("input pattern matched no files: " + ", ".join(unmatched))
    unique = sorted({path.expanduser().resolve() for path in files}, key=str)
    if not unique:
        raise ReportError("no input files were found")
    output_resolved = output.expanduser().resolve()
    if output_resolved in unique:
        raise ReportError("output path must not match an input path")
    return unique


def _read_objects(path: Path) -> tuple[list[tuple[int, Any]], int]:
    """Read one JSON document or line-delimited JSON objects.

    Pretty-printed configuration JSON is treated as one non-result object instead
    of producing one parse warning per line.
    """
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        return [], 0
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        objects: list[tuple[int, Any]] = []
        invalid = 0
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                objects.append((line_number, json.loads(line)))
            except json.JSONDecodeError:
                invalid += 1
        return objects, invalid
    return [(1, value)], 0


def load_inputs(paths: Sequence[Path]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    loaded: list[dict[str, Any]] = []
    invalid_lines = 0
    for path in paths:
        objects, invalid = _read_objects(path)
        invalid_lines += invalid
        for line_number, value in objects:
            loaded.append({"source_file": str(path), "source_line": line_number, "value": value})
    return loaded, {"invalid_json_lines": invalid_lines, "objects_read": len(loaded)}


def _result_path(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return str(path.resolve())


LifecycleMetadata = tuple[
    dict[str, dict[str, Any]],
    tuple[tuple[str, dict[str, Any]], ...],
]


def lifecycle_metadata(records: Sequence[Mapping[str, Any]]) -> LifecycleMetadata:
    starts: dict[str, dict[str, Any]] = {}
    result_runs: dict[str, str] = {}
    allowed = (
        "tag",
        "runtime",
        "source_config",
        "private_config",
        "engine",
        "model",
        "model_name",
        "cache",
        "context",
        "pool_workers",
        "cpus",
        "engine_environment",
    )
    for loaded in records:
        value = loaded.get("value")
        if not isinstance(value, dict):
            continue
        run_id = value.get("run_id") or value.get("session_id")
        if not isinstance(run_id, str):
            continue
        if value.get("event") in ("trial_started", "trial_start", "session_started"):
            starts[run_id] = {key: value[key] for key in allowed if key in value}
        if value.get("event") in ("suite_started", "step_started", "step_complete"):
            path = _result_path(value.get("result") or value.get("output"))
            if path is not None:
                result_runs[path] = run_id
    exact = {
        path: starts[run_id]
        for path, run_id in result_runs.items()
        if run_id in starts
    }
    by_tag: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for start in starts.values():
        tag = start.get("tag")
        if isinstance(tag, str) and tag:
            by_tag[tag].append(start)
    prefixes = tuple(sorted(
        ((tag, values[0]) for tag, values in by_tag.items() if len(values) == 1),
        key=lambda item: (-len(item[0]), item[0]),
    ))
    return exact, prefixes


def _trial_metadata(source_file: str, lifecycle: LifecycleMetadata) -> dict[str, Any]:
    exact, prefixes = lifecycle
    source_path = Path(source_file)
    matched = exact.get(str(source_path.expanduser().resolve()))
    if matched is not None:
        return matched
    stem = source_path.stem
    for tag, trial in prefixes:
        if stem == tag or stem.startswith(tag + "-"):
            return trial
    return {}


def _is_benchmark_row(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and isinstance(value.get("case_id"), str)
        and isinstance(value.get("suite"), str)
        and isinstance(value.get("request"), dict)
        and isinstance(value.get("evaluation"), dict)
    )


def classify(row: Mapping[str, Any]) -> str:
    error = row.get("error")
    if isinstance(error, dict):
        error_type = str(error.get("type", ""))
        message = str(error.get("message", ""))
        if "timeout" in (error_type + " " + message).lower():
            return "timeout"
        return "request_failure"
    response = row.get("response")
    evaluation = row.get("evaluation") if isinstance(row.get("evaluation"), dict) else {}
    if isinstance(response, dict) and response.get("truncated") is True:
        return "truncated"
    if evaluation.get("truncated") is True:
        return "truncated"
    if evaluation.get("passed") is True:
        return "passed"
    return "wrong_answer"


def _elapsed_from_timestamps(row: Mapping[str, Any]) -> float | None:
    try:
        start = datetime.fromisoformat(str(row["started_at"]).replace("Z", "+00:00"))
        finish = datetime.fromisoformat(str(row["finished_at"]).replace("Z", "+00:00"))
    except (KeyError, TypeError, ValueError):
        return None
    seconds = (finish - start).total_seconds()
    return seconds if seconds >= 0 else None


def _case_matches_current(case: bench_cases.BenchmarkCase, row: Mapping[str, Any]) -> bool:
    request = row.get("request") if isinstance(row.get("request"), dict) else {}
    prompt_hash = hashlib.sha256(case.prompt.encode("utf-8")).hexdigest()
    return (
        row.get("case_version") == bench_cases.CASE_VERSION
        and request.get("prompt_sha256") == prompt_hash
    )


def hidden_test_diagnostics(
    row: Mapping[str, Any], response_text: str
) -> dict[str, Any] | None:
    case = _HARD_BY_ID.get(str(row.get("case_id")))
    if case is None or case.evaluator != "python_function" or not case.function_name:
        return None
    if not _case_matches_current(case, row):
        return {
            "available": False,
            "reason": "row case version or prompt hash does not match the current case definition",
        }
    source = bench_cases.extract_python_source(response_text)
    valid, errors = bench_cases.validate_python_source(source, case.function_name)
    if not valid:
        return {
            "available": False,
            "reason": "candidate source was rejected by the restricted evaluator",
            "validation_errors": errors,
        }

    passed = 0
    failures: list[dict[str, Any]] = []
    for index, test in enumerate(case.tests):
        result = bench_cases.run_python_checks(
            source, case.function_name, [test], timeout=3.0
        )
        if result.get("passed") is True:
            passed += 1
        else:
            failures.append(
                {
                    "index": index,
                    "label": test.get("label", f"test_{index}"),
                    "category": test.get("category", "unspecified"),
                    "detail": result.get("detail", "failed"),
                }
            )
    return {
        "available": True,
        "rescored_with_case_version": bench_cases.CASE_VERSION,
        "rescored_with_case_source_sha256": _CURRENT_CASE_SOURCE_SHA256,
        "hidden_test_count": len(case.tests),
        "hidden_tests_passed": passed,
        "failed_tests": failures,
    }


def _strict_equal(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(_strict_equal(left[key], right[key]) for key in left)
    if isinstance(left, list):
        return len(left) == len(right) and all(_strict_equal(a, b) for a, b in zip(left, right))
    return left == right


def _reasonable_json_value(value: Any, limit: int = 8192) -> Any | None:
    try:
        encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError):
        return None
    return value if len(encoded) <= limit else None


def json_diagnostics(row: Mapping[str, Any], response_text: str) -> dict[str, Any] | None:
    case = _HARD_BY_ID.get(str(row.get("case_id")))
    if case is None or case.evaluator != "json_exact":
        return None
    if not _case_matches_current(case, row):
        return {
            "available": False,
            "reason": "row case version or prompt hash does not match the current case definition",
        }
    match = _FENCE_RE.fullmatch(response_text)
    candidate = match.group(1).strip() if match else response_text.strip()
    try:
        actual = json.loads(candidate)
    except json.JSONDecodeError as exc:
        return {"available": True, "parse_error": f"{exc.msg} at line {exc.lineno} column {exc.colno}"}

    detail: dict[str, Any] = {"available": True}
    if isinstance(case.expected, dict) and isinstance(actual, dict):
        expected_keys = set(case.expected)
        actual_keys = set(actual)
        detail["top_level_keys"] = {
            "missing": sorted(expected_keys - actual_keys),
            "unexpected": sorted(actual_keys - expected_keys),
            "different_values": sorted(
                key
                for key in expected_keys & actual_keys
                if not _strict_equal(case.expected[key], actual[key])
            ),
        }
    else:
        detail["expected_top_level_type"] = type(case.expected).__name__
        detail["actual_top_level_type"] = type(actual).__name__
    expected_value = _reasonable_json_value(case.expected)
    actual_value = _reasonable_json_value(actual)
    if expected_value is not None and actual_value is not None:
        detail["expected_value"] = expected_value
        detail["actual_value"] = actual_value
    else:
        detail["values_omitted"] = "expected or actual JSON exceeded the report size limit"
    return detail


def _profile(
    row: Mapping[str, Any], source_file: str, lifecycle: LifecycleMetadata
) -> dict[str, Any]:
    request = row["request"]
    trial = _trial_metadata(source_file, lifecycle)
    engine_environment = _engine_environment(
        request.get("engine_environment", trial.get("engine_environment"))
    )
    return {
        "model": request.get("model"),
        "thinking": request.get("thinking"),
        "temperature": request.get("temperature"),
        "output_token_limit": request.get("output_token_limit"),
        "endpoint": _clean_endpoint(request.get("endpoint")),
        "backend": request.get("backend"),
        "context_max_tokens": request.get("context_max_tokens"),
        "context_mode": request.get("context_mode"),
        "runtime": request.get("runtime") or trial.get("runtime") or trial.get("engine"),
        "source_config": request.get("source_config") or trial.get("source_config"),
        "runtime_cache": request.get("runtime_cache", trial.get("cache")),
        "pool_workers": request.get("pool_workers", trial.get("pool_workers")),
        "cpus": request.get("cpus", trial.get("cpus")),
        "engine_environment": engine_environment,
        "reasoning_budget_tokens_requested": request.get("reasoning_budget_tokens"),
        "strata_pcie_frac": request.get("strata_pcie_frac"),
        "strata_aux_cpus": request.get("strata_aux_cpus"),
        "chat_template_thinking": request.get("chat_template_thinking"),
    }


def summarize_task(
    loaded: Mapping[str, Any], lifecycle: LifecycleMetadata
) -> tuple[dict[str, Any], dict[str, Any]]:
    row = loaded["value"]
    source_file = str(loaded["source_file"])
    request = row["request"]
    trial = _trial_metadata(source_file, lifecycle)
    engine_environment = _engine_environment(
        request.get("engine_environment", trial.get("engine_environment"))
    )
    response = row.get("response") if isinstance(row.get("response"), dict) else {}
    timing = response.get("timing") if isinstance(response.get("timing"), dict) else {}
    throughput = response.get("throughput") if isinstance(response.get("throughput"), dict) else {}
    outcome = classify(row)
    wall = _finite_number(timing.get("wall_seconds"))
    if wall is None:
        wall = _elapsed_from_timestamps(row)
    task: dict[str, Any] = {
        "run_id": row.get("run_id"),
        "call_id": row.get("call_id"),
        "repeat_index": row.get("repeat_index"),
        "suite": row.get("suite"),
        "case_id": row.get("case_id"),
        "outcome": outcome,
        "passed": outcome == "passed",
        "wall_seconds": wall,
        "ttft_final_seconds": _finite_number(timing.get("ttft_final_content_seconds")),
        "native_completion_tokens_per_second": _finite_number(
            throughput.get("completion_tokens_per_second_native")
        ),
        "prompt_tokens": response.get("actual_prompt_tokens"),
        "completion_tokens": response.get("actual_completion_tokens"),
        "reasoning_chars": response.get("reasoning_chars"),
        "finish_reason": response.get("done_reason"),
        "sampling_seed": request.get("seed"),
        "temperature": request.get("temperature"),
        "private_config": request.get("private_config") or trial.get("private_config"),
        "trial_tag": request.get("trial_tag", trial.get("tag")),
        "engine_environment": engine_environment,
        "reasoning_budget_tokens_requested": request.get("reasoning_budget_tokens"),
        "strata_pcie_frac": request.get("strata_pcie_frac"),
        "strata_aux_cpus": request.get("strata_aux_cpus"),
        "chat_template_thinking": request.get("chat_template_thinking"),
        "source_file": source_file,
        "source_line": loaded.get("source_line"),
        "prompt_sha256": request.get("prompt_sha256"),
        "case_source_sha256": row.get("case_source_sha256"),
        "case_version": row.get("case_version"),
        "original_evaluation": {
            key: row["evaluation"][key]
            for key in ("passed", "detail", "tests_passed", "validation", "truncated", "source_chars")
            if key in row["evaluation"]
        },
    }
    if isinstance(row.get("error"), dict):
        task["request_error_type"] = row["error"].get("type")

    response_text = response.get("response_text")
    if outcome == "wrong_answer" and isinstance(response_text, str):
        coding = hidden_test_diagnostics(row, response_text)
        if coding is not None:
            task["hidden_test_diagnostics"] = coding
            if coding.get("available") is True:
                all_passed = coding.get("hidden_tests_passed") == coding.get("hidden_test_count")
                current_name_error = any(
                    "raised NameError" in str(failure.get("detail", ""))
                    for failure in coding.get("failed_tests", [])
                )
                original_name_error = "NameError" in str(row["evaluation"].get("detail", ""))
                task["rescored_outcome"] = "passed" if all_passed else "wrong_answer"
                if original_name_error and all_passed:
                    task["rescored_cause"] = "recorded_evaluator_builtin_limitation"
                elif current_name_error:
                    task["rescored_cause"] = "unresolved_name_requires_evaluator_review"
                else:
                    task["rescored_cause"] = "semantic_or_contract_failure"
        json_detail = json_diagnostics(row, response_text)
        if json_detail is not None:
            task["json_diagnostics"] = json_detail
    return task, _profile(row, source_file, lifecycle)


def _mean(values: Iterable[Any]) -> float | None:
    numbers = [float(value) for value in values if _finite_number(value) is not None]
    return sum(numbers) / len(numbers) if numbers else None


def _outcome_counts(tasks: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts = Counter(str(task["outcome"]) for task in tasks)
    return {outcome: counts.get(outcome, 0) for outcome in OUTCOMES}


def _profile_id(profile: Mapping[str, Any]) -> str:
    encoded = json.dumps(profile, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _task_identity(task: Mapping[str, Any]) -> tuple[str, str, str, str] | None:
    keys = ("suite", "case_id", "prompt_sha256", "case_version")
    if not all(isinstance(task.get(key), str) and bool(task.get(key)) for key in keys):
        return None
    return tuple(task[key] for key in keys)  # type: ignore[return-value]


def build_report(paths: Sequence[Path]) -> dict[str, Any]:
    loaded, read_stats = load_inputs(paths)
    lifecycle = lifecycle_metadata(loaded)
    benchmark = [item for item in loaded if _is_benchmark_row(item.get("value"))]

    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    summarized: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for item in benchmark:
        task, profile = summarize_task(item, lifecycle)
        summarized.append((task, profile))
        profile_id = _profile_id(profile)
        key = (profile_id, str(task["suite"]))
        group = grouped.setdefault(
            key,
            {
                "profile_id": profile_id,
                "profile": profile,
                "suite": task["suite"],
                "tasks": [],
            },
        )
        group["tasks"].append(task)

    groups: list[dict[str, Any]] = []
    for key in sorted(grouped):
        group = grouped[key]
        tasks = sorted(
            group["tasks"],
            key=lambda task: (
                str(task.get("case_id")),
                task.get("repeat_index") if isinstance(task.get("repeat_index"), int) else -1,
                str(task.get("source_file")),
                int(task.get("source_line") or 0),
            ),
        )
        repeats = Counter(str(task["case_id"]) for task in tasks)
        identities = {_task_identity(task) for task in tasks} - {None}
        group["attempt_count"] = len(tasks)
        group["independent_task_count"] = len(identities)
        group["task_count"] = len(tasks)
        group["task_count_definition"] = "compatibility alias of attempt_count (result rows)"
        group["outcome_counts"] = _outcome_counts(tasks)
        group["rescored_outcome_counts"] = _outcome_counts(
            [dict(task, outcome=task.get("rescored_outcome", task["outcome"])) for task in tasks]
        )
        group["coverage"] = {
            "case_ids": sorted(repeats),
            "unique_case_count": len(repeats),
            "unique_case_prompt_count": len(identities),
            "result_rows_by_case": dict(sorted(repeats.items())),
        }
        group["private_config_paths"] = sorted(
            {
                str(task["private_config"])
                for task in tasks
                if isinstance(task.get("private_config"), str) and task["private_config"]
            }
        )
        group["mean_wall_seconds"] = _mean(task.get("wall_seconds") for task in tasks)
        group["mean_native_completion_tokens_per_second"] = _mean(
            task.get("native_completion_tokens_per_second") for task in tasks
        )
        group["tasks"] = tasks
        groups.append(group)

    comparison_buckets: dict[tuple[str, str, str, str], list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for task, profile in summarized:
        identity = _task_identity(task)
        if identity is None:
            continue
        comparison_buckets[identity].append((task, profile))

    comparisons: list[dict[str, Any]] = []
    for identity in sorted(comparison_buckets):
        entries = comparison_buckets[identity]
        by_profile: dict[str, dict[str, Any]] = {}
        for task, profile in entries:
            profile_id = _profile_id(profile)
            bucket = by_profile.setdefault(profile_id, {"profile": profile, "tasks": []})
            bucket["tasks"].append(task)
        if len(by_profile) < 2:
            continue
        profiles = []
        for profile_id in sorted(by_profile):
            bucket = by_profile[profile_id]
            tasks = bucket["tasks"]
            profiles.append(
                {
                    "profile_id": profile_id,
                    "profile": bucket["profile"],
                    "attempt_count": len(tasks),
                    "independent_task_count": 1,
                    "task_count": len(tasks),
                    "task_count_definition": "compatibility alias of attempt_count (result rows)",
                    "outcome_counts": _outcome_counts(tasks),
                    "rescored_outcome_counts": _outcome_counts(
                        [dict(task, outcome=task.get("rescored_outcome", task["outcome"])) for task in tasks]
                    ),
                    "mean_wall_seconds": _mean(task.get("wall_seconds") for task in tasks),
                    "mean_native_completion_tokens_per_second": _mean(
                        task.get("native_completion_tokens_per_second") for task in tasks
                    ),
                }
            )
        comparisons.append(
            {
                "suite": identity[0],
                "case_id": identity[1],
                "prompt_sha256": identity[2],
                "case_version": identity[3],
                "profiles": profiles,
            }
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": utc_now(),
        "inputs": {
            "files": [str(path) for path in paths],
            "file_count": len(paths),
            "benchmark_result_rows": len(benchmark),
            "skipped_non_result_objects": len(loaded) - len(benchmark),
            **read_stats,
        },
        "attempt_count": len(benchmark),
        "independent_task_count": len(
            {_task_identity(task) for task, _ in summarized} - {None}
        ),
        "groups": groups,
        "comparisons": comparisons,
        "comparison_policy": "same suite, case_id, prompt_sha256, and case_version only; no overall model rank",
        "report_evaluator": {
            "case_version": bench_cases.CASE_VERSION,
            "case_source_sha256": _CURRENT_CASE_SOURCE_SHA256,
            "coding_rescore_policy": "failed valid current hard-case code is run once per hidden test in the restricted evaluator",
        },
    }


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
        temporary_name = None
        if os.name == "posix":
            directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
    finally:
        if temporary_name is not None:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate an offline hard-benchmark report from result JSONL files."
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="input file paths or glob patterns (quote shell globs for internal expansion)",
    )
    parser.add_argument("--output", required=True, help="atomic JSON report destination")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output = Path(args.output)
    try:
        inputs = resolve_inputs(args.inputs, output)
        report = build_report(inputs)
        atomic_write_json(output, report)
    except (OSError, ReportError) as exc:
        print(f"hard_report: {exc}", file=sys.stderr)
        return 2
    print(
        f"wrote {output} with {report['attempt_count']} attempts across "
        f"{report['independent_task_count']} independent task definitions",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
