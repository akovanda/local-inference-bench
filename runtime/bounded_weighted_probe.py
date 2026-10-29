#!/usr/bin/env python3
"""Run one bounded-domain weighted-interval diagnostic on a held server.

This probe changes the original task's stated input bound and explicitly allows
exhaustive search.  Its result is therefore a diagnostic for one underlying
problem, not a comparable score for the frozen benchmark case.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import bench  # noqa: E402
import bench_cases  # noqa: E402
from bench_cases import BenchmarkCase  # noqa: E402
from runtime import budget_prompt_probe, sampler_recipe_probe  # noqa: E402


SCHEMA_VERSION = 1
CONTEXT_TOKENS = 16_384
DEFAULT_OUTPUT_CAP = 8_192
MAX_OUTPUT_CAP = 12_288
DEFAULT_TIMEOUT_SECONDS = 1_200.0
MAX_TIMEOUT_SECONDS = 1_800.0
CHAT_TEMPLATE_TOKEN_RESERVE = 256
THINKING = "medium"
TEMPERATURE = 0.0
SEED = 0
PARENT_CASE_ID = "hard_weighted_intervals"
VARIANT_CASE_ID = "hard_weighted_intervals_bounded12_exhaustive_allowed"
VARIANT_CASE_VERSION = f"{bench_cases.CASE_VERSION}.bounded12-exhaustive-allowed.1"
DIAGNOSTIC_VARIANT = "bounded12_exhaustive_allowed"
MAX_JOB_ENTRIES = 12
INSTRUCTION_SUFFIX = (
    "\n\nFor this bounded-size diagnostic, jobs contains at most 12 entries. "
    "An exact exhaustive search over all subsets is acceptable; no optimization "
    "beyond correctness is required."
)
FORBIDDEN_RESPONSE_FIELDS = {
    "analysis",
    "analysis_text",
    "chain_of_thought",
    "cot",
    "raw_reasoning",
    "reasoning",
    "reasoning_content",
    "reasoning_text",
}


class ProbeError(RuntimeError):
    """A diagnostic precondition or artifact contract was not satisfied."""


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def original_case() -> BenchmarkCase:
    cases = bench_cases.get_cases("hard", [PARENT_CASE_ID])
    if len(cases) != 1:
        raise ProbeError("the frozen weighted-interval case did not resolve uniquely")
    return cases[0]


def _is_valid_jobs_input(value: Any) -> bool:
    if not isinstance(value, list):
        return False
    seen: set[str] = set()
    for job in value:
        if not isinstance(job, list) or len(job) != 4:
            return False
        job_id, start, end, weight = job
        if not isinstance(job_id, str) or not job_id or job_id in seen:
            return False
        if (
            isinstance(start, bool)
            or not isinstance(start, int)
            or isinstance(end, bool)
            or not isinstance(end, int)
            or end <= start
            or isinstance(weight, bool)
            or not isinstance(weight, int)
            or weight < 0
        ):
            return False
        seen.add(job_id)
    return True


def validate_parent_test_bound(case: BenchmarkCase) -> dict[str, int]:
    """Prove that every valid jobs input in the retained grader is in-domain."""
    if case.evaluator != "python_function" or case.function_name != "select_weighted_jobs":
        raise ProbeError("the parent case no longer uses the expected frozen evaluator")
    valid_count = 0
    largest = 0
    for index, test in enumerate(case.tests):
        if not isinstance(test, Mapping):
            raise ProbeError(f"parent grader test {index} is not a mapping")
        args = test.get("args")
        if not isinstance(args, (list, tuple)) or len(args) != 1:
            raise ProbeError(f"parent grader test {index} does not have one jobs argument")
        jobs = args[0]
        if _is_valid_jobs_input(jobs):
            valid_count += 1
            largest = max(largest, len(jobs))
            if len(jobs) > MAX_JOB_ENTRIES:
                raise ProbeError(
                    f"valid parent grader test {index} has {len(jobs)} jobs, "
                    f"exceeding the diagnostic bound of {MAX_JOB_ENTRIES}"
                )
    if not valid_count:
        raise ProbeError("the parent grader has no valid jobs inputs")
    return {
        "total_test_count": len(case.tests),
        "valid_input_test_count": valid_count,
        "largest_valid_input_entries": largest,
        "validated_max_entries": MAX_JOB_ENTRIES,
    }


def build_variant() -> BenchmarkCase:
    parent = original_case()
    parent_prompt_hash = sha256_text(parent.prompt)
    full_prompt = parent.prompt + INSTRUCTION_SUFFIX
    metadata = dict(parent.metadata)
    metadata.update(
        {
            "diagnostic_kind": "bounded_weighted_interval_domain_probe",
            "diagnostic_variant": DIAGNOSTIC_VARIANT,
            "diagnostic_domain_changed": True,
            "diagnostic_algorithm_guidance": True,
            "parent_case_id": PARENT_CASE_ID,
            "parent_case_version": bench_cases.CASE_VERSION,
            "parent_prompt_sha256": parent_prompt_hash,
            "instruction_suffix_sha256": sha256_text(INSTRUCTION_SUFFIX),
            "full_variant_prompt_sha256": sha256_text(full_prompt),
            "changed_input_bound_max_entries": MAX_JOB_ENTRIES,
            "exact_exhaustive_search_allowed": True,
            "benchmark_comparability": "changed-domain diagnostic; not an original benchmark score",
        }
    )
    variant = replace(
        parent,
        case_id=VARIANT_CASE_ID,
        prompt=full_prompt,
        metadata=metadata,
    )
    if variant.tests is not parent.tests or variant.evaluator != parent.evaluator:
        raise ProbeError("the bounded variant changed the frozen tests or evaluator")
    validate_parent_test_bound(parent)
    return variant


def artifact_paths(prefix: Path) -> dict[str, Path]:
    return {
        "requests": Path(f"{prefix}-bounded-weighted-requests.jsonl"),
        "manifest": Path(f"{prefix}-bounded-weighted-manifest.json"),
        "receipt": Path(f"{prefix}-bounded-weighted-receipt.json"),
    }


def _validate_artifact_paths(paths: Mapping[str, Path], tag: str) -> None:
    suffix = "-bounded-weighted-requests.jsonl"
    prefix_name = paths["requests"].name.removesuffix(suffix)
    if prefix_name != tag and not prefix_name.startswith(tag + "-"):
        raise ProbeError(
            "--output-prefix basename must equal --tag or start with '<tag>-' "
            "for lifecycle joins"
        )
    existing = [str(path) for path in paths.values() if path.exists()]
    if existing:
        raise ProbeError("refusing to overwrite existing diagnostic artifacts: " + ", ".join(existing))


def _exclusive_json(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _safe_public_value(value: Any) -> Any:
    """Copy public response data without ever fetching raw-reasoning values."""
    if isinstance(value, Mapping):
        result: dict[Any, Any] = {}
        for key in value:
            if isinstance(key, str) and key in FORBIDDEN_RESPONSE_FIELDS:
                continue
            result[key] = _safe_public_value(value[key])
        return result
    if isinstance(value, list):
        return [_safe_public_value(item) for item in value]
    if isinstance(value, tuple):
        return [_safe_public_value(item) for item in value]
    return value


def _error(exc: BaseException) -> dict[str, str]:
    return {"type": type(exc).__name__, "message": str(exc)[:500]}


def _prompt_feasibility(endpoint: str, case: BenchmarkCase, output_cap: int) -> dict[str, Any]:
    _, origin = budget_prompt_probe._endpoint_parts(endpoint)
    try:
        raw_tokens = budget_prompt_probe._tokenize_count(origin, case.prompt)
    except budget_prompt_probe.ProbeHTTPError as exc:
        if exc.status in {404, 405, 501}:
            raise ProbeError(
                "resident endpoint does not support /tokenize; exact prompt-plus-cap preflight is required"
            ) from exc
        raise
    required = raw_tokens + CHAT_TEMPLATE_TOKEN_RESERVE + output_cap
    if required > CONTEXT_TOKENS:
        raise ProbeError("bounded diagnostic prompt plus output cap does not fit verified context")
    return {
        "method": "llama.cpp /tokenize raw prompt token IDs",
        "raw_prompt_tokens": raw_tokens,
        "chat_template_token_reserve": CHAT_TEMPLATE_TOKEN_RESERVE,
        "output_token_limit": output_cap,
        "required_context_bound": required,
        "context_tokens": CONTEXT_TOKENS,
        "fits": True,
        "uncertainty": (
            "The raw /tokenize endpoint excludes the chat template; the bound includes a "
            f"fixed {CHAT_TEMPLATE_TOKEN_RESERVE}-token single-message wrapper reserve."
        ),
    }


def _request_metadata(
    args: argparse.Namespace,
    case: BenchmarkCase,
    *,
    parent: BenchmarkCase,
    caller_source_hash: str,
    source_case_hash: str,
) -> dict[str, Any]:
    return {
        "endpoint": args.endpoint,
        "backend": "openai",
        "model": args.model,
        "context_max_tokens": CONTEXT_TOKENS,
        "thinking": THINKING,
        "temperature": TEMPERATURE,
        "seed": SEED,
        "output_token_limit": args.output_cap,
        "timeout_seconds": args.timeout,
        "api_key_env": None,
        "context_mode": "full",
        "strata_pcie_frac": None,
        "strata_aux_cpus": None,
        "reasoning_budget_tokens": None,
        "chat_template_thinking": "on",
        "prompt_chars": len(case.prompt),
        "prompt_sha256": sha256_text(case.prompt),
        "parent_case_id": PARENT_CASE_ID,
        "parent_case_version": bench_cases.CASE_VERSION,
        "parent_prompt_sha256": sha256_text(parent.prompt),
        "instruction_suffix_sha256": sha256_text(INSTRUCTION_SUFFIX),
        "full_variant_prompt_sha256": sha256_text(case.prompt),
        "source_case_sha256": source_case_hash,
        "caller_source_sha256": caller_source_hash,
        "diagnostic_variant": DIAGNOSTIC_VARIANT,
        "diagnostic_domain_changed": True,
        "diagnostic_algorithm_guidance": True,
        "changed_input_bound_max_entries": MAX_JOB_ENTRIES,
        "exact_exhaustive_search_allowed": True,
        "model_session_run_id": args.model_session_run_id,
    }


def _result_row(
    *,
    args: argparse.Namespace,
    parent: BenchmarkCase,
    case: BenchmarkCase,
    probe_run_id: str,
    call_id: str,
    started_at: str,
    caller_source_hash: str,
    source_case_hash: str,
    call: Mapping[str, Any] | None,
    request_error: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if call is None:
        response = None
        evaluation = {"passed": False, "detail": "request failed"}
    else:
        response = _safe_public_value(call)
        evaluation = bench_cases.evaluate_case(
            case,
            str(call.get("response_text", "")),
            truncated=bool(call.get("truncated")),
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": probe_run_id,
        "call_id": call_id,
        "started_at": started_at,
        "finished_at": bench.utc_now(),
        "repeat_index": 0,
        "tag": args.tag,
        "model": args.model,
        "suite": case.suite,
        "case_id": case.case_id,
        "case_version": VARIANT_CASE_VERSION,
        "case_source_sha256": source_case_hash,
        "case_metadata": dict(case.metadata),
        "request": _request_metadata(
            args,
            case,
            parent=parent,
            caller_source_hash=caller_source_hash,
            source_case_hash=source_case_hash,
        ),
        "telemetry_before": None,
        "telemetry_after": None,
        "response": response,
        "evaluation": evaluation,
        "error": dict(request_error) if request_error is not None else None,
    }


def _attempt_summary(row: Mapping[str, Any]) -> dict[str, Any]:
    response = row.get("response") if isinstance(row.get("response"), Mapping) else {}
    evaluation = row.get("evaluation") if isinstance(row.get("evaluation"), Mapping) else {}
    timing = response.get("timing") if isinstance(response.get("timing"), Mapping) else {}
    return {
        "call_id": row.get("call_id"),
        "case_id": row.get("case_id"),
        "case_version": row.get("case_version"),
        "started_at": row.get("started_at"),
        "finished_at": row.get("finished_at"),
        "wall_seconds": timing.get("wall_seconds"),
        "prompt_tokens": response.get("actual_prompt_tokens"),
        "completion_tokens": response.get("actual_completion_tokens"),
        "reasoning_chars": response.get("reasoning_chars"),
        "stream_complete": response.get("stream_complete"),
        "truncated": response.get("truncated"),
        "evaluator_passed": (
            evaluation.get("passed") if isinstance(evaluation.get("passed"), bool) else None
        ),
        "request_failed": row.get("error") is not None,
    }


def run(args: argparse.Namespace) -> int:
    if not args.tag or not args.model_session_run_id or not args.model:
        raise ProbeError("tag, model-session-run-id, and model must be non-empty")
    if (
        isinstance(args.output_cap, bool)
        or not isinstance(args.output_cap, int)
        or not 1 <= args.output_cap <= MAX_OUTPUT_CAP
    ):
        raise ProbeError(f"output cap must be an integer from 1 through {MAX_OUTPUT_CAP}")
    if (
        isinstance(args.timeout, bool)
        or not isinstance(args.timeout, (int, float))
        or not math.isfinite(float(args.timeout))
        or not 0 < float(args.timeout) <= MAX_TIMEOUT_SECONDS
    ):
        raise ProbeError(f"timeout must be finite, positive, and at most {MAX_TIMEOUT_SECONDS:g}")

    prefix = Path(args.output_prefix).expanduser().resolve()
    paths = artifact_paths(prefix)
    _validate_artifact_paths(paths, args.tag)
    parent = original_case()
    original_fingerprint = _sha256_json(
        {
            "case_id": parent.case_id,
            "prompt": parent.prompt,
            "max_tokens": parent.max_tokens,
            "evaluator": parent.evaluator,
            "function_name": parent.function_name,
            "tests": parent.tests,
            "metadata": parent.metadata,
        }
    )
    case = build_variant()
    test_bound = validate_parent_test_bound(parent)
    if _sha256_json(parent.tests) != _sha256_json(case.tests):
        raise ProbeError("bounded variant does not retain the exact parent test tuple")

    lifecycle_path = Path(args.runtime_events).expanduser().resolve()
    lifecycle = sampler_recipe_probe.wait_for_active_hold(
        lifecycle_path,
        run_id=args.model_session_run_id,
        tag=args.tag,
        model=args.model,
        timeout=0.0,
    )
    resident = sampler_recipe_probe.verify_resident(args.endpoint, args.model)
    feasibility = _prompt_feasibility(args.endpoint, case, args.output_cap)
    sampler_recipe_probe.assert_active_hold(
        lifecycle_path,
        run_id=args.model_session_run_id,
        tag=args.tag,
        model=args.model,
    )

    caller_source_hash = sha256_file(Path(__file__))
    source_case_hash = sha256_file(Path(bench_cases.__file__))
    bench_source_hash = sha256_file(Path(bench.__file__))
    tests_hash = _sha256_json(parent.tests)
    probe_run_id = str(uuid.uuid4())
    controls = {
        "context_max_tokens": CONTEXT_TOKENS,
        "output_token_limit": args.output_cap,
        "timeout_seconds": float(args.timeout),
        "thinking": THINKING,
        "temperature": TEMPERATURE,
        "seed": SEED,
        "chat_template_thinking": "on",
        "reasoning_budget_tokens": None,
        "json_schema": None,
        "sampler_tuning": None,
        "inherited_server_defaults_are_canonical": True,
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at": bench.utc_now(),
        "diagnostic_kind": "bounded_weighted_interval_domain_probe",
        "run_id": probe_run_id,
        "tag": args.tag,
        "model_session_run_id": args.model_session_run_id,
        "underlying_problem_count": 1,
        "planned_new_attempts": 1,
        "comparability": "changed-domain/algorithm-clarification diagnostic; not an original benchmark score",
        "parent": {
            "case_id": parent.case_id,
            "case_version": bench_cases.CASE_VERSION,
            "prompt": parent.prompt,
            "prompt_sha256": sha256_text(parent.prompt),
            "source_case_sha256": source_case_hash,
            "evaluator": parent.evaluator,
            "function_name": parent.function_name,
            "tests_sha256": tests_hash,
            "test_count": len(parent.tests),
        },
        "variant": {
            "case_id": case.case_id,
            "case_version": VARIANT_CASE_VERSION,
            "diagnostic_variant": DIAGNOSTIC_VARIANT,
            "diagnostic_domain_changed": True,
            "diagnostic_algorithm_guidance": True,
            "prompt": case.prompt,
            "prompt_sha256": sha256_text(case.prompt),
            "parent_case_id": PARENT_CASE_ID,
            "parent_prompt_sha256": sha256_text(parent.prompt),
            "instruction_suffix": INSTRUCTION_SUFFIX,
            "instruction_suffix_sha256": sha256_text(INSTRUCTION_SUFFIX),
            "changed_input_bound_max_entries": MAX_JOB_ENTRIES,
            "exact_exhaustive_search_allowed": True,
            "evaluator_unchanged": True,
            "tests_tuple_retained": case.tests is parent.tests,
            "tests_sha256": tests_hash,
        },
        "parent_test_bound_validation": test_bound,
        "controls": controls,
        "prompt_feasibility": feasibility,
        "sources": {
            "caller_source_path": str(Path(__file__).resolve()),
            "caller_source_sha256": caller_source_hash,
            "source_case_path": str(Path(bench_cases.__file__).resolve()),
            "source_case_sha256": source_case_hash,
            "bench_source_path": str(Path(bench.__file__).resolve()),
            "bench_source_sha256": bench_source_hash,
        },
    }

    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    _exclusive_json(paths["manifest"], manifest)
    receipt: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "started_at": bench.utc_now(),
        "status": "running",
        "diagnostic_kind": "bounded_weighted_interval_domain_probe",
        "run_id": probe_run_id,
        "tag": args.tag,
        "model": args.model,
        "model_session_run_id": args.model_session_run_id,
        "underlying_problem_count": 1,
        "planned_request_count": 1,
        "attempted_request_count": 0,
        "recorded_request_count": 0,
        "request_failure_count": 0,
        "truncated_count": 0,
        "evaluator_pass_count": 0,
        "lifecycle": lifecycle,
        "resident": resident,
        "controls": controls,
        "prompt_feasibility": feasibility,
        "parent_test_bound_validation": test_bound,
        "provenance": {
            "parent_case_id": PARENT_CASE_ID,
            "parent_case_version": bench_cases.CASE_VERSION,
            "variant_case_id": case.case_id,
            "variant_case_version": VARIANT_CASE_VERSION,
            "parent_prompt_sha256": sha256_text(parent.prompt),
            "instruction_suffix_sha256": sha256_text(INSTRUCTION_SUFFIX),
            "full_variant_prompt_sha256": sha256_text(case.prompt),
            "source_case_sha256": source_case_hash,
            "caller_source_sha256": caller_source_hash,
            "tests_sha256": tests_hash,
            "tests_tuple_retained": case.tests is parent.tests,
            "diagnostic_variant": DIAGNOSTIC_VARIANT,
            "diagnostic_domain_changed": True,
            "diagnostic_algorithm_guidance": True,
            "changed_input_bound_max_entries": MAX_JOB_ENTRIES,
            "exact_exhaustive_search_allowed": True,
        },
        "artifacts": {key: str(value) for key, value in paths.items()},
        "attempts": [],
    }
    run_started = time.monotonic()
    interrupted: BaseException | None = None
    try:
        with paths["requests"].open("x", encoding="utf-8", buffering=1) as handle:
            # Re-check both authorities immediately before the one GPU request.
            sampler_recipe_probe.assert_active_hold(
                lifecycle_path,
                run_id=args.model_session_run_id,
                tag=args.tag,
                model=args.model,
            )
            sampler_recipe_probe.verify_resident(args.endpoint, args.model)
            config = bench.RunConfig(
                endpoint=args.endpoint,
                backend="openai",
                model=args.model,
                context=CONTEXT_TOKENS,
                thinking=THINKING,
                temperature=TEMPERATURE,
                seed=SEED,
                num_predict=args.output_cap,
                timeout=float(args.timeout),
                chat_template_thinking="on",
            )
            call_id = str(uuid.uuid4())
            started_at = bench.utc_now()
            receipt["attempted_request_count"] = 1
            receipt["in_flight"] = {"call_id": call_id, "started_at": started_at}
            call: Mapping[str, Any] | None = None
            request_error: dict[str, str] | None = None
            try:
                call = bench.run_request(config, case)
            except (bench.BenchmarkHTTPError, TimeoutError, OSError, ValueError) as exc:
                request_error = _error(exc)
            row = _result_row(
                args=args,
                parent=parent,
                case=case,
                probe_run_id=probe_run_id,
                call_id=call_id,
                started_at=started_at,
                caller_source_hash=caller_source_hash,
                source_case_hash=source_case_hash,
                call=call,
                request_error=request_error,
            )
            bench._append_jsonl(handle, row)
            summary = _attempt_summary(row)
            receipt["attempts"].append(summary)
            receipt["recorded_request_count"] = 1
            receipt["request_failure_count"] = int(request_error is not None)
            receipt["truncated_count"] = int(summary["truncated"] is True)
            receipt["evaluator_pass_count"] = int(summary["evaluator_passed"] is True)
            receipt.pop("in_flight", None)
            print(
                "RECORDED "
                f"{case.case_id} call_id={call_id} "
                f"prompt_tokens={summary['prompt_tokens']} "
                f"completion_tokens={summary['completion_tokens']} "
                f"reasoning_chars={summary['reasoning_chars']} "
                f"evaluator_passed={summary['evaluator_passed']}",
                file=sys.stderr,
                flush=True,
            )
        sampler_recipe_probe.assert_active_hold(
            lifecycle_path,
            run_id=args.model_session_run_id,
            tag=args.tag,
            model=args.model,
        )
    except BaseException as exc:
        receipt["status"] = "interrupted"
        receipt["interruption"] = _error(exc)
        interrupted = exc
    else:
        receipt["status"] = (
            "complete"
            if receipt["request_failure_count"] == 0
            else "completed_with_request_failures"
        )
    finally:
        if original_fingerprint != _sha256_json(
            {
                "case_id": parent.case_id,
                "prompt": parent.prompt,
                "max_tokens": parent.max_tokens,
                "evaluator": parent.evaluator,
                "function_name": parent.function_name,
                "tests": parent.tests,
                "metadata": parent.metadata,
            }
        ):
            receipt["status"] = "interrupted"
            receipt["interruption"] = {
                "type": "ProbeError",
                "message": "frozen parent case mutated during the diagnostic",
            }
            interrupted = ProbeError("frozen parent case mutated during the diagnostic")
        receipt["finished_at"] = bench.utc_now()
        receipt["elapsed_seconds"] = time.monotonic() - run_started
        receipt["artifacts"].update(
            {
                "requests_sha256": (
                    sha256_file(paths["requests"]) if paths["requests"].exists() else None
                ),
                "manifest_sha256": sha256_file(paths["manifest"]),
            }
        )
        _exclusive_json(paths["receipt"], receipt)
    if interrupted is not None:
        raise interrupted
    return 1 if receipt["request_failure_count"] else 0


def _bounded_output_cap(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if not 1 <= parsed <= MAX_OUTPUT_CAP:
        raise argparse.ArgumentTypeError(f"must be from 1 through {MAX_OUTPUT_CAP}")
    return parsed


def _bounded_timeout(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if not math.isfinite(parsed) or not 0 < parsed <= MAX_TIMEOUT_SECONDS:
        raise argparse.ArgumentTypeError(
            f"must be finite, positive, and at most {MAX_TIMEOUT_SECONDS:g}"
        )
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-events", required=True, help="exact active model_session event ledger")
    parser.add_argument("--tag", required=True, help="exact active model_session tag")
    parser.add_argument("--model-session-run-id", required=True, help="exact active model_session run ID")
    parser.add_argument("--endpoint", required=True, help="loopback native llama endpoint")
    parser.add_argument("--model", required=True, help="exact resident model alias")
    parser.add_argument("--output-prefix", required=True, help="unique artifact prefix beginning with --tag")
    parser.add_argument("--output-cap", type=_bounded_output_cap, default=DEFAULT_OUTPUT_CAP)
    parser.add_argument("--timeout", type=_bounded_timeout, default=DEFAULT_TIMEOUT_SECONDS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run(args)
    except KeyboardInterrupt:
        print("bounded_weighted_probe: interrupted", file=sys.stderr)
        return 130
    except (
        ProbeError,
        budget_prompt_probe.ProbeError,
        sampler_recipe_probe.ProbeError,
        ValueError,
    ) as exc:
        print(f"bounded_weighted_probe: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
