#!/usr/bin/env python3
"""Run a sampler recipe on frozen hard cases in an already-held native session.

This helper never starts, stops, or reconfigures a model. It binds every
serialized request to one exact active ``model_session`` hold and records
benchmark-compatible metadata without retaining raw reasoning content.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import bench  # noqa: E402
import bench_cases  # noqa: E402
from runtime import budget_prompt_probe, sampler_recipe_probe  # noqa: E402


SCHEMA_VERSION = 1
SEED = 0
DEFAULT_OUTPUT_CAP = 32_768
MAX_OUTPUT_CAP = 65_536
DEFAULT_TIMEOUT_SECONDS = 14_400.0
MAX_TIMEOUT_SECONDS = 21_600.0
DEFAULT_CONTEXT = 49_152
MAX_CONTEXT = 131_072
SAFE_RESPONSE_FIELDS = {
    "response_text",
    "reasoning_chars",
    "timing",
    "native_ollama",
    "openai_usage",
    "openai_native_timings",
    "actual_prompt_tokens",
    "actual_completion_tokens",
    "done_reason",
    "stream_complete",
    "truncated",
    "throughput",
}


class HardProbeError(RuntimeError):
    """A frozen-case, resident-session, or artifact contract failed."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _csv_cases(value: str) -> tuple[str, ...]:
    values = tuple(item.strip() for item in value.split(","))
    if not values or any(not item for item in values):
        raise HardProbeError("--cases must contain one or more comma-separated hard case IDs")
    if len(set(values)) != len(values):
        raise HardProbeError("--cases must not contain duplicates")
    available = {case.case_id for case in bench_cases.HARD_CASES}
    unknown = [case_id for case_id in values if case_id not in available]
    if unknown:
        raise HardProbeError(
            "unknown hard case(s): " + ", ".join(unknown)
            + "; available: " + ", ".join(case.case_id for case in bench_cases.HARD_CASES)
        )
    return values


def selected_cases(value: str) -> list[Any]:
    requested = _csv_cases(value)
    try:
        cases = bench_cases.get_cases("hard", requested)
    except ValueError as exc:  # defensive join to the frozen suite contract
        raise HardProbeError(str(exc)) from exc
    if [case.case_id for case in cases] != list(requested):
        raise HardProbeError("frozen hard case resolution changed the requested order")
    return cases


def _bounded_int(maximum: int):
    def parse(value: str) -> int:
        parsed = int(value)
        if not 1 <= parsed <= maximum:
            raise argparse.ArgumentTypeError(f"must be from 1 through {maximum}")
        return parsed

    return parse


def _bounded_seconds(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or not 0 < parsed <= MAX_TIMEOUT_SECONDS:
        raise argparse.ArgumentTypeError(
            f"must be finite, positive, and at most {MAX_TIMEOUT_SECONDS:g}"
        )
    return parsed


def _artifact_paths(prefix: Path) -> dict[str, Path]:
    return {
        "requests": Path(f"{prefix}-native-hard-requests.jsonl"),
        "manifest": Path(f"{prefix}-native-hard-manifest.json"),
        "receipt": Path(f"{prefix}-native-hard-receipt.json"),
    }


def _validate_artifacts(paths: Mapping[str, Path], tag: str) -> None:
    stem = paths["requests"].name.removesuffix("-native-hard-requests.jsonl")
    if stem != tag and not stem.startswith(tag + "-"):
        raise HardProbeError(
            "--output-prefix basename must equal --tag or start with '<tag>-'"
        )
    existing = [str(path) for path in paths.values() if os.path.lexists(path)]
    if existing:
        raise HardProbeError("refusing to overwrite probe artifacts: " + ", ".join(existing))


def _exclusive_json(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _safe_response(call: Mapping[str, Any]) -> dict[str, Any]:
    """Copy only bench.run_request's public result contract, never raw reasoning."""
    return {key: call[key] for key in SAFE_RESPONSE_FIELDS if key in call}


def _error(exc: BaseException) -> dict[str, str]:
    return {"type": type(exc).__name__, "message": str(exc)[:500]}


def _case_manifest(case: Any) -> dict[str, Any]:
    return {
        "case_id": case.case_id,
        "suite": "hard",
        "case_version": bench_cases.CASE_VERSION,
        "prompt_sha256": hashlib.sha256(case.prompt.encode("utf-8")).hexdigest(),
        "evaluator": case.evaluator,
        "metadata": dict(case.metadata),
    }


def _request_metadata(
    args: argparse.Namespace,
    case: Any,
    config: bench.RunConfig,
    recipe: Mapping[str, Any],
    recipe_path: Path,
    recipe_sha256: str,
) -> dict[str, Any]:
    request = bench._request_config(config, case)
    thinking_budget_tokens = getattr(args, "thinking_budget_tokens", None)
    reasoning_controls = {
        "reasoning_effort": args.thinking,
        "chat_template_thinking": "off" if args.thinking == "none" else "on",
    }
    request.update(
        {
            "temperature": recipe["sampler_overrides"].get("temperature"),
            "expected_context_tokens": args.expected_context,
            "actual_context_tokens": args.expected_context,
            "sampler_recipe_path": str(recipe_path),
            "sampler_recipe_sha256": recipe_sha256,
            "sampler_recipe_label": recipe["label"],
            "sampler_recipe_claim": recipe["claim"],
            "sampler_recipe_sources": list(recipe["sources"]),
            "sampler_overrides": dict(recipe["sampler_overrides"]),
            "sampler_unspecified_fields": list(recipe["unspecified_fields"]),
            "sampler_application_evidence": recipe["application_evidence"],
            "reasoning_controls_requested": reasoning_controls,
            "application_limits": (
                "The outgoing payload was inspected in-process. Endpoint acceptance does not "
                "independently prove that sampler or reasoning controls were applied or enforced."
            ),
            "resident_identity_verified_before_and_after": True,
        }
    )
    if thinking_budget_tokens is not None:
        reasoning_controls["thinking_budget_tokens"] = thinking_budget_tokens
        request.update(
            {
                "native_thinking_budget_tokens_requested": thinking_budget_tokens,
                "native_thinking_budget_application_evidence": (
                    sampler_recipe_probe.NATIVE_THINKING_BUDGET_APPLICATION_EVIDENCE
                ),
            }
        )
    return request


def _outcome(
    response: Mapping[str, Any] | None,
    evaluation: Mapping[str, Any],
    error: Mapping[str, Any] | None,
) -> str:
    if error is not None:
        return "timeout" if error.get("type") == "TimeoutError" else "request_failure"
    if response is not None and response.get("truncated") is True:
        return "truncated"
    return "passed" if evaluation.get("passed") is True else "wrong_answer"


def _row(
    *,
    args: argparse.Namespace,
    case: Any,
    config: bench.RunConfig,
    recipe: Mapping[str, Any],
    recipe_path: Path,
    recipe_sha256: str,
    run_id: str,
    call_id: str,
    started_at: str,
    call: Mapping[str, Any] | None,
    error: Mapping[str, Any] | None,
) -> dict[str, Any]:
    response = _safe_response(call) if call is not None else None
    evaluation = (
        bench_cases.evaluate_case(
            case,
            str(call.get("response_text", "")),
            truncated=bool(call.get("truncated")),
        )
        if call is not None
        else {"passed": False, "detail": "request failed"}
    )
    outcome = _outcome(response, evaluation, error)
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "call_id": call_id,
        "started_at": started_at,
        "finished_at": bench.utc_now(),
        "repeat_index": 0,
        "tag": args.tag,
        "model": args.model,
        "profile": args.profile or recipe["label"],
        "suite": "hard",
        "case_id": case.case_id,
        "case_version": bench_cases.CASE_VERSION,
        "case_source_sha256": sha256_file(Path(bench_cases.__file__)),
        "case_metadata": dict(case.metadata)
        | {
            "diagnostic_kind": "native_hard_recipe_probe",
            "model_session_run_id": args.model_session_run_id,
        },
        "request": _request_metadata(
            args, case, config, recipe, recipe_path, recipe_sha256
        ),
        "telemetry_before": None,
        "telemetry_after": None,
        "response": response,
        "evaluation": evaluation,
        "objective_outcome": outcome,
        "error": error,
    }


def _summary(row: Mapping[str, Any]) -> dict[str, Any]:
    response = row.get("response") if isinstance(row.get("response"), Mapping) else {}
    timing = response.get("timing") if isinstance(response.get("timing"), Mapping) else {}
    native = (
        response.get("openai_native_timings")
        if isinstance(response.get("openai_native_timings"), Mapping)
        else {}
    )
    usage = (
        response.get("openai_usage")
        if isinstance(response.get("openai_usage"), Mapping)
        else {}
    )
    evaluation = row.get("evaluation") if isinstance(row.get("evaluation"), Mapping) else {}
    error = row.get("error") if isinstance(row.get("error"), Mapping) else None
    result = {
        "call_id": row.get("call_id"),
        "case_id": row.get("case_id"),
        "outcome": row.get("objective_outcome"),
        "passed": evaluation.get("passed") if isinstance(evaluation.get("passed"), bool) else None,
        "stream_complete": response.get("stream_complete"),
        "truncated": response.get("truncated"),
        "input_tokens": response.get("actual_prompt_tokens", usage.get("prompt_tokens")),
        "output_tokens": response.get(
            "actual_completion_tokens", usage.get("completion_tokens")
        ),
        "wall_seconds": timing.get("wall_seconds"),
        "native_decode_seconds": (
            float(native["predicted_ms"]) / 1000
            if isinstance(native.get("predicted_ms"), (int, float))
            and not isinstance(native.get("predicted_ms"), bool)
            else None
        ),
        "error_type": error.get("type") if error is not None else None,
    }
    request = row.get("request") if isinstance(row.get("request"), Mapping) else {}
    if "native_thinking_budget_tokens_requested" in request:
        result.update(
            {
                "native_thinking_budget_tokens_requested": request.get(
                    "native_thinking_budget_tokens_requested"
                ),
                "native_thinking_budget_application_evidence": request.get(
                    "native_thinking_budget_application_evidence"
                ),
            }
        )
    return result


def _validate_thinking_budget(args: argparse.Namespace) -> int | None:
    try:
        budget = sampler_recipe_probe._validate_thinking_budget(
            getattr(args, "thinking_budget_tokens", None), args.output_cap
        )
    except sampler_recipe_probe.ProbeError as exc:
        raise HardProbeError(str(exc)) from exc
    if budget is not None and args.thinking != "medium":
        raise HardProbeError("--thinking-budget-tokens is valid only with --thinking medium")
    return budget


def run(args: argparse.Namespace) -> int:
    if not 1 <= args.expected_context <= MAX_CONTEXT:
        raise HardProbeError(f"--expected-context must be from 1 through {MAX_CONTEXT}")
    thinking_budget_tokens = _validate_thinking_budget(args)
    cases = selected_cases(args.cases)
    recipe_path = Path(args.recipe).expanduser().resolve()
    try:
        recipe, recipe_sha256 = sampler_recipe_probe.load_recipe(recipe_path)
    except sampler_recipe_probe.ProbeError as exc:
        raise HardProbeError(str(exc)) from exc
    prefix = Path(args.output_prefix).expanduser().resolve()
    paths = _artifact_paths(prefix)
    _validate_artifacts(paths, args.tag)
    lifecycle_path = Path(args.runtime_events).expanduser().resolve()

    lifecycle = sampler_recipe_probe.wait_for_active_hold(
        lifecycle_path,
        run_id=args.model_session_run_id,
        tag=args.tag,
        model=args.model,
        timeout=args.holding_timeout,
        expected_context=args.expected_context,
    )
    resident = sampler_recipe_probe.verify_resident(
        args.endpoint, args.model, args.expected_context
    )
    feasibility = sampler_recipe_probe.prompt_feasibility(
        args.endpoint,
        [sampler_recipe_probe.SelectedCase("hard", case) for case in cases],
        args.output_cap,
        args.expected_context,
    )
    sampler_recipe_probe.assert_active_hold(
        lifecycle_path,
        run_id=args.model_session_run_id,
        tag=args.tag,
        model=args.model,
        expected_context=args.expected_context,
    )

    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    run_id = str(uuid.uuid4())
    controls = {
        "thinking": args.thinking,
        "chat_template_thinking": "off" if args.thinking == "none" else "on",
        "seed": SEED,
        "expected_context_tokens": args.expected_context,
        "actual_context_tokens": resident["actual_context_tokens"],
        "output_token_limit": args.output_cap,
        "timeout_seconds": args.timeout,
        "sampler_overrides": dict(recipe["sampler_overrides"]),
        "sampler_application_evidence": recipe["application_evidence"],
        "application_limits": (
            "Request payload presence is recorded; server-side application and effectiveness "
            "are not independently verified."
        ),
    }
    if thinking_budget_tokens is not None:
        controls.update(
            {
                "native_thinking_budget_tokens_requested": thinking_budget_tokens,
                "native_thinking_budget_application_evidence": (
                    sampler_recipe_probe.NATIVE_THINKING_BUDGET_APPLICATION_EVIDENCE
                ),
            }
        )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at": bench.utc_now(),
        "diagnostic_kind": "native_hard_recipe_probe",
        "run_id": run_id,
        "tag": args.tag,
        "model": args.model,
        "model_session_run_id": args.model_session_run_id,
        "recipe": recipe | {"path": str(recipe_path), "sha256": recipe_sha256},
        "controls": controls,
        "sources": {
            "native_hard_recipe_probe.py": sha256_file(Path(__file__)),
            "sampler_recipe_probe.py": sha256_file(Path(sampler_recipe_probe.__file__)),
            "bench.py": sha256_file(Path(bench.__file__)),
            "bench_cases.py": sha256_file(Path(bench_cases.__file__)),
        },
        "case_count": len(cases),
        "cases": [_case_manifest(case) for case in cases],
        "prompt_feasibility": feasibility,
    }
    _exclusive_json(paths["manifest"], manifest)

    receipt: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "started_at": bench.utc_now(),
        "status": "running",
        "diagnostic_kind": "native_hard_recipe_probe",
        "run_id": run_id,
        "tag": args.tag,
        "model": args.model,
        "model_session_run_id": args.model_session_run_id,
        "runtime_backend": "llama",
        "lifecycle": lifecycle,
        "resident": resident,
        "recipe": recipe | {"path": str(recipe_path), "sha256": recipe_sha256},
        "controls": controls,
        "case_count": len(cases),
        "attempted_count": 0,
        "completed_count": 0,
        "passed_count": 0,
        "wrong_answer_count": 0,
        "truncated_count": 0,
        "request_failure_count": 0,
        "artifacts": {key: str(path) for key, path in paths.items()},
        "attempts": [],
    }
    interrupted: BaseException | None = None
    try:
        with paths["requests"].open("x", encoding="utf-8", buffering=1) as handle:
            with sampler_recipe_probe.sampler_payload_overrides(
                recipe["sampler_overrides"],
                thinking_budget_tokens=thinking_budget_tokens,
            ):
                for case in cases:
                    sampler_recipe_probe.assert_active_hold(
                        lifecycle_path,
                        run_id=args.model_session_run_id,
                        tag=args.tag,
                        model=args.model,
                        expected_context=args.expected_context,
                    )
                    sampler_recipe_probe.verify_resident(
                        args.endpoint, args.model, args.expected_context
                    )
                    config = bench.RunConfig(
                        endpoint=args.endpoint,
                        backend="openai",
                        model=args.model,
                        context=args.expected_context,
                        thinking=args.thinking,
                        temperature=float(recipe["sampler_overrides"].get("temperature", 0.0)),
                        seed=SEED,
                        num_predict=args.output_cap,
                        timeout=args.timeout,
                        chat_template_thinking="off" if args.thinking == "none" else "on",
                    )
                    call_id = str(uuid.uuid4())
                    started_at = bench.utc_now()
                    call: dict[str, Any] | None = None
                    request_error: dict[str, Any] | None = None
                    receipt["attempted_count"] += 1
                    receipt["in_flight_case"] = {
                        "case_id": case.case_id,
                        "call_id": call_id,
                        "started_at": started_at,
                    }
                    try:
                        call = bench.run_request(config, case)
                    except (bench.BenchmarkHTTPError, TimeoutError, OSError, ValueError) as exc:
                        request_error = _error(exc)
                    sampler_recipe_probe.assert_active_hold(
                        lifecycle_path,
                        run_id=args.model_session_run_id,
                        tag=args.tag,
                        model=args.model,
                        expected_context=args.expected_context,
                    )
                    sampler_recipe_probe.verify_resident(
                        args.endpoint, args.model, args.expected_context
                    )
                    row = _row(
                        args=args,
                        case=case,
                        config=config,
                        recipe=recipe,
                        recipe_path=recipe_path,
                        recipe_sha256=recipe_sha256,
                        run_id=run_id,
                        call_id=call_id,
                        started_at=started_at,
                        call=call,
                        error=request_error,
                    )
                    bench._append_jsonl(handle, row)
                    summary = _summary(row)
                    receipt["attempts"].append(summary)
                    receipt.pop("in_flight_case", None)
                    if summary["stream_complete"] is True and summary["truncated"] is not True:
                        receipt["completed_count"] += 1
                    if summary["outcome"] == "passed":
                        receipt["passed_count"] += 1
                    elif summary["outcome"] == "wrong_answer":
                        receipt["wrong_answer_count"] += 1
                    elif summary["outcome"] == "truncated":
                        receipt["truncated_count"] += 1
                    if request_error is not None:
                        receipt["request_failure_count"] += 1
                    print(
                        f"RECORDED {case.case_id} call_id={call_id}",
                        file=sys.stderr,
                        flush=True,
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
        receipt["finished_at"] = bench.utc_now()
        _exclusive_json(paths["receipt"], receipt)
    if interrupted is not None:
        raise interrupted
    return 1 if receipt["request_failure_count"] else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", required=True)
    parser.add_argument("--runtime-events", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--model-session-run-id", required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--expected-context", type=_bounded_int(MAX_CONTEXT), default=DEFAULT_CONTEXT
    )
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--profile")
    parser.add_argument("--cases", required=True)
    parser.add_argument("--thinking", choices=("none", "medium"), required=True)
    parser.add_argument("--output-cap", type=_bounded_int(MAX_OUTPUT_CAP), default=DEFAULT_OUTPUT_CAP)
    parser.add_argument(
        "--thinking-budget-tokens",
        type=_bounded_int(MAX_OUTPUT_CAP),
        help=(
            "optional native llama thinking ceiling; valid only with --thinking medium and "
            "must not exceed --output-cap"
        ),
    )
    parser.add_argument("--timeout", type=_bounded_seconds, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--holding-timeout", type=bench.positive_float, default=60.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run(args)
    except (HardProbeError, sampler_recipe_probe.ProbeError, budget_prompt_probe.ProbeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
