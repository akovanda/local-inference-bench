#!/usr/bin/env python3
"""Run the frozen specialist suite against a loopback OpenAI-compatible server."""

from __future__ import annotations

import argparse
import hashlib
import ipaddress
import json
import math
import os
import sys
import uuid
from pathlib import Path
from typing import Any, Iterable, Mapping
from urllib.parse import urlsplit


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import bench  # noqa: E402
from runtime import specialist_cases  # noqa: E402


SCHEMA_VERSION = 1
MAX_OUTPUT_CAP = 65_536
MAX_TIMEOUT_SECONDS = 21_600.0
MAX_CONTEXT_TOKENS = 131_072


def _csv_values(value: str | None) -> tuple[str, ...] | None:
    if value is None:
        return None
    values = tuple(dict.fromkeys(item.strip() for item in value.split(",") if item.strip()))
    if not values:
        raise ValueError("comma-separated selection must contain at least one value")
    return values


def validate_loopback_endpoint(endpoint: str) -> None:
    parsed = urlsplit(endpoint)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("--endpoint must be an absolute http(s) URL")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("--endpoint must not contain credentials")
    hostname = parsed.hostname
    if hostname == "localhost":
        return
    try:
        is_loopback = ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        is_loopback = False
    if not is_loopback:
        raise ValueError("--endpoint must address a loopback host")


def _bounded_positive_int(value: str) -> int:
    parsed = int(value)
    if not 1 <= parsed <= MAX_OUTPUT_CAP:
        raise argparse.ArgumentTypeError(f"must be from 1 through {MAX_OUTPUT_CAP}")
    return parsed


def _bounded_timeout(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or not 0 < parsed <= MAX_TIMEOUT_SECONDS:
        raise argparse.ArgumentTypeError(
            f"must be a finite number greater than 0 and no more than {MAX_TIMEOUT_SECONDS:g}"
        )
    return parsed


def _bounded_context(value: str) -> int:
    parsed = int(value)
    if not 1 <= parsed <= MAX_CONTEXT_TOKENS:
        raise argparse.ArgumentTypeError(f"must be from 1 through {MAX_CONTEXT_TOKENS}")
    return parsed


def _temperature(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or not 0 <= parsed <= 2:
        raise argparse.ArgumentTypeError("must be a finite number from 0 through 2")
    return parsed


def manifest() -> dict[str, Any]:
    return {
        "case_version": specialist_cases.CASE_VERSION,
        "suite_sha256": specialist_cases.FROZEN_SUITE_SHA256,
        "case_count": len(specialist_cases.CASES),
        "cases": [
            {
                "case_id": case.case_id,
                "group": case.group,
                "category": case.category,
                "case_sha256": specialist_cases.case_sha256(case),
                "default_output_cap": case.max_tokens,
                "independent_task_count": 1 if case.group == "code" else 0,
                "objective_count": len(case.code_objectives),
                "assertion_count": sum(
                    len(objective.assertions) for objective in case.code_objectives
                ),
                "deterministic_prose_constraint_count": len(case.prose_constraints),
                "blinded_rubric_max_points": sum(
                    criterion.max_points for criterion in case.rubric
                ),
            }
            for case in specialist_cases.CASES
        ],
    }


def _reasoning_tokens(usage: Any) -> int | None:
    if not isinstance(usage, Mapping):
        return None
    containers = (
        usage.get("completion_tokens_details"),
        usage.get("output_tokens_details"),
    )
    for details in containers:
        if not isinstance(details, Mapping):
            continue
        value = details.get("reasoning_tokens")
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return value
    return None


def _milliseconds_to_seconds(value: Any) -> float | None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    if not math.isfinite(float(value)) or value < 0:
        return None
    return float(value) / 1000.0


def _nanoseconds_to_seconds(value: Any) -> float | None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    if not math.isfinite(float(value)) or value < 0:
        return None
    return float(value) / 1_000_000_000.0


def response_metrics(call: Mapping[str, Any]) -> dict[str, Any]:
    timing = call.get("timing") if isinstance(call.get("timing"), Mapping) else {}
    throughput = (
        call.get("throughput") if isinstance(call.get("throughput"), Mapping) else {}
    )
    native = (
        call.get("openai_native_timings")
        if isinstance(call.get("openai_native_timings"), Mapping)
        else {}
    )
    ollama_native = (
        call.get("native_ollama")
        if isinstance(call.get("native_ollama"), Mapping)
        else {}
    )
    load_seconds = None
    load_source = None
    if "load_seconds" in native and isinstance(native["load_seconds"], (int, float)):
        load_seconds = float(native["load_seconds"])
        load_source = "openai_native_timings.load_seconds"
    elif "load_ms" in native:
        load_seconds = _milliseconds_to_seconds(native["load_ms"])
        load_source = "openai_native_timings.load_ms" if load_seconds is not None else None
    elif "load_duration" in ollama_native:
        load_seconds = _nanoseconds_to_seconds(ollama_native["load_duration"])
        load_source = "native_ollama.load_duration" if load_seconds is not None else None

    reasoning_chars_raw = call.get("reasoning_chars")
    reasoning_chars = (
        reasoning_chars_raw
        if isinstance(reasoning_chars_raw, int)
        and not isinstance(reasoning_chars_raw, bool)
        and reasoning_chars_raw > 0
        else None
    )
    reasoning_tokens = _reasoning_tokens(call.get("openai_usage"))
    return {
        "input_tokens_reported": call.get("actual_prompt_tokens"),
        "output_tokens_reported": call.get("actual_completion_tokens"),
        "reasoning_tokens_reported": reasoning_tokens,
        "reasoning_chars_observed": reasoning_chars,
        "reasoning_stream_observed": reasoning_chars is not None,
        "finish_reason": call.get("done_reason"),
        "stream_complete": call.get("stream_complete"),
        "truncated": call.get("truncated"),
        "wall_seconds": timing.get("wall_seconds"),
        "ttft_visible_seconds": timing.get("ttft_final_content_seconds"),
        "ttft_reasoning_seconds": timing.get("ttft_reasoning_seconds"),
        "native_prompt_tokens": native.get("prompt_n"),
        "native_cached_prompt_tokens": native.get("cache_n"),
        "native_decode_tokens": native.get("predicted_n"),
        "native_decode_seconds": _milliseconds_to_seconds(native.get("predicted_ms")),
        "native_prompt_rate_tokens_per_second": throughput.get(
            "prompt_tokens_per_second_native"
        ),
        "native_decode_rate_tokens_per_second": throughput.get(
            "completion_tokens_per_second_native"
        ),
        "wall_decode_rate_tokens_per_second": throughput.get(
            "completion_tokens_per_second_wall"
        ),
        "load_seconds_reported": load_seconds,
        "load_metric_source": load_source,
    }


def _write_visible_text(directory: Path, call_id: str, text: str) -> dict[str, Any]:
    directory.mkdir(parents=True, exist_ok=True)
    path = (directory / f"{call_id}.txt").resolve()
    with path.open("x", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    encoded = text.encode("utf-8")
    return {
        "path": str(path),
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "chars": len(text),
        "bytes": len(encoded),
        "reasoning_excluded": True,
    }


def _error_record(exc: BaseException) -> dict[str, str]:
    return {"type": type(exc).__name__, "message": str(exc)}


def _request_outcome(error: Mapping[str, Any]) -> str:
    return "timeout" if error.get("type") == "TimeoutError" else "request_failure"


def _eligibility(
    case: specialist_cases.SpecialistCase,
    call: Mapping[str, Any] | None,
    evaluation: Mapping[str, Any],
    error: Mapping[str, Any] | None,
) -> dict[str, bool]:
    complete = bool(
        error is None
        and call is not None
        and call.get("stream_complete") is True
        and call.get("truncated") is not True
    )
    evaluation_accepted = evaluation.get("outcome") != "evaluator_rejected"
    return {
        "speed_eligible": complete,
        "coding_objective_eligible": bool(
            complete and case.group == "code" and evaluation_accepted
        ),
        "prose_constraint_eligible": bool(complete and case.group == "prose"),
        "blinded_prose_review_eligible": bool(complete and case.group == "prose"),
        "reasoning_metric_eligible": bool(
            complete
            and call is not None
            and (
                response_metrics(call)["reasoning_stream_observed"]
                or response_metrics(call)["reasoning_tokens_reported"] is not None
            )
        ),
    }


def _case_source_sha256() -> str:
    return hashlib.sha256(Path(specialist_cases.__file__).read_bytes()).hexdigest()


def _selected_cases(args: argparse.Namespace) -> list[specialist_cases.SpecialistCase]:
    return specialist_cases.get_cases(
        groups=_csv_values(args.groups),
        case_ids=_csv_values(args.cases),
    )


def _validate_run_args(args: argparse.Namespace) -> None:
    missing = [
        flag
        for flag, value in (
            ("--endpoint", args.endpoint),
            ("--model", args.model),
            ("--tag", args.tag),
            ("--profile", args.profile),
            ("--jsonl", args.jsonl),
        )
        if not value
    ]
    if missing:
        raise ValueError("required for a benchmark run: " + ", ".join(missing))
    validate_loopback_endpoint(args.endpoint)
    if args.reasoning_budget_tokens is not None and args.reasoning_budget_tokens < 1:
        raise ValueError("--reasoning-budget-tokens must be positive")


def run(args: argparse.Namespace) -> int:
    specialist_cases.assert_frozen_suite()
    _validate_run_args(args)
    cases = _selected_cases(args)
    output = Path(args.jsonl).expanduser().resolve()
    visible_dir = (
        Path(args.visible_dir).expanduser().resolve()
        if args.visible_dir
        else output.with_name(output.name + ".texts")
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    run_id = str(uuid.uuid4())
    case_source_hash = _case_source_sha256()
    had_request_error = False

    with output.open("a", encoding="utf-8", buffering=1) as handle:
        for repeat_index in range(args.repeat):
            for case in cases:
                output_cap = args.output_cap if args.output_cap is not None else case.max_tokens
                config = bench.RunConfig(
                    endpoint=args.endpoint,
                    backend="openai",
                    model=args.model,
                    context=args.context,
                    thinking=args.thinking,
                    temperature=args.temperature,
                    seed=args.seed,
                    num_predict=output_cap,
                    timeout=args.timeout,
                    reasoning_budget_tokens=args.reasoning_budget_tokens,
                    chat_template_thinking=args.chat_template_thinking,
                )
                bench.validate_request_controls(config)
                call_id = str(uuid.uuid4())
                started_at = bench.utc_now()
                call: dict[str, Any] | None = None
                error: dict[str, str] | None = None
                try:
                    call = bench.run_request(config, case)
                except (bench.BenchmarkHTTPError, TimeoutError, OSError, ValueError) as exc:
                    error = _error_record(exc)
                    had_request_error = True

                visible_text = str(call.get("response_text", "")) if call is not None else ""
                visible_record = _write_visible_text(visible_dir, call_id, visible_text)
                if call is None:
                    evaluation: dict[str, Any] = {
                        "outcome": _request_outcome(error or {}),
                        "passed": False,
                        "detail": "request failed before a complete response was available",
                    }
                    metrics = None
                else:
                    evaluation = specialist_cases.evaluate_case(
                        case,
                        visible_text,
                        truncated=bool(call.get("truncated")),
                    )
                    metrics = response_metrics(call)

                objective_outcome = evaluation.get("outcome", "evaluator_internal_error")
                response = (
                    {key: value for key, value in call.items() if key != "response_text"}
                    if call is not None
                    else None
                )
                record = {
                    "schema_version": SCHEMA_VERSION,
                    "run_id": run_id,
                    "call_id": call_id,
                    "started_at": started_at,
                    "finished_at": bench.utc_now(),
                    "repeat_index": repeat_index,
                    "tag": args.tag,
                    "model": args.model,
                    "profile": args.profile,
                    "suite": f"specialist_{case.group}",
                    "case_id": case.case_id,
                    "case_group": case.group,
                    "case_category": case.category,
                    "case_version": specialist_cases.CASE_VERSION,
                    "case_sha256": specialist_cases.case_sha256(case),
                    "suite_sha256": specialist_cases.FROZEN_SUITE_SHA256,
                    "case_source_sha256": case_source_hash,
                    "request": {
                        "endpoint": args.endpoint,
                        "backend": "openai",
                        "model": args.model,
                        "profile": args.profile,
                        "thinking": args.thinking,
                        "chat_template_thinking": args.chat_template_thinking,
                        "reasoning_budget_tokens": args.reasoning_budget_tokens,
                        "temperature": args.temperature,
                        "seed": args.seed,
                        "context_max_tokens": args.context,
                        "output_token_limit": output_cap,
                        "timeout_seconds": args.timeout,
                        "prompt_chars": len(case.prompt),
                        "prompt_sha256": hashlib.sha256(
                            case.prompt.encode("utf-8")
                        ).hexdigest(),
                        "reasoning_controls_requested": {
                            "reasoning_effort": args.thinking,
                            "chat_template_thinking": args.chat_template_thinking,
                            "reasoning_budget_tokens": args.reasoning_budget_tokens,
                        },
                        "capability_note": args.capability_note,
                    },
                    "response": response,
                    "response_metrics": metrics,
                    "visible_text": visible_record,
                    "objective_outcome": objective_outcome,
                    "objective_error": error,
                    "error": error,
                    "evaluation": evaluation,
                    "eligibility": _eligibility(case, call, evaluation, error),
                }
                bench._append_jsonl(handle, record)
                status = "PASS" if evaluation.get("passed") is True else objective_outcome.upper()
                # Visible response content is intentionally absent from console output.
                print(
                    f"{status} {case.case_id} repeat={repeat_index + 1}/{args.repeat} "
                    f"call_id={call_id}",
                    file=sys.stderr,
                    flush=True,
                )
    return 1 if had_request_error else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--print-manifest",
        action="store_true",
        help="print the frozen case manifest as JSON and exit without a request",
    )
    parser.add_argument("--endpoint", help="loopback llama-server base URL")
    parser.add_argument("--model", help="model identifier sent to the server")
    parser.add_argument("--tag", help="unique run/model artifact tag")
    parser.add_argument("--profile", help="comparison profile label, such as coding-medium")
    parser.add_argument("--jsonl", help="append-only result JSONL path")
    parser.add_argument(
        "--visible-dir",
        help="visible-answer directory (default: <jsonl>.texts)",
    )
    parser.add_argument(
        "--groups",
        default="code,prose",
        help="comma-separated code/prose groups; default runs all six cases",
    )
    parser.add_argument("--cases", help="comma-separated case IDs after group filtering")
    parser.add_argument(
        "--thinking",
        choices=("none", "low", "medium", "high"),
        default="none",
        help="requested reasoning_effort; capability note records unsupported controls",
    )
    parser.add_argument("--chat-template-thinking", choices=("on", "off"))
    parser.add_argument("--reasoning-budget-tokens", type=bench.positive_int)
    parser.add_argument(
        "--capability-note",
        help="document model/template control limits without claiming they were enforced",
    )
    parser.add_argument("--output-cap", type=_bounded_positive_int)
    parser.add_argument("--timeout", type=_bounded_timeout, default=600.0)
    parser.add_argument("--context", type=_bounded_context, default=32768)
    parser.add_argument("--temperature", type=_temperature, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repeat", type=bench.positive_int, default=1)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        specialist_cases.assert_frozen_suite()
        if args.print_manifest:
            print(json.dumps(manifest(), ensure_ascii=False, indent=2, sort_keys=True))
            return 0
        return run(args)
    except ValueError as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

