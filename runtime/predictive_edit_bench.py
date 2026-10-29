#!/usr/bin/env python3
"""Standalone copy/edit benchmark for weight-free speculative decoding."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import uuid
from pathlib import Path
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import bench  # noqa: E402
import bench_cases  # noqa: E402
from bench_cases import BenchmarkCase  # noqa: E402


CASE_ID = "quality_copy_edit_module"
CASE_VERSION = "2026-10-19.copy-edit.1"
OUTPUT_TOKENS = 2048
_EDIT_FROM = "TAX_BASIS_POINTS = 1200"
_EDIT_TO = "TAX_BASIS_POINTS = 1500"
_OUTER_PYTHON_FENCE = re.compile(
    r"^\s*```(?:python|py)?[ \t]*\r?\n(.*?)\r?\n```[ \t]*\s*$",
    re.DOTALL | re.IGNORECASE,
)


ORIGINAL_SOURCE = '''def tax_basis_points():
    TAX_BASIS_POINTS = 1200
    return TAX_BASIS_POINTS


def _nonnegative_int(value, label):
    if type(value) is not int:
        raise TypeError(label + " must be int")
    if value < 0:
        raise ValueError(label + " must be nonnegative")
    return value


def _line_total(line):
    if type(line) is not dict:
        raise TypeError("line must be dict")
    quantity = _nonnegative_int(line.get("quantity"), "quantity")
    unit_cents = _nonnegative_int(line.get("unit_cents"), "unit_cents")
    discount = _nonnegative_int(line.get("discount_basis_points", 0), "discount")
    if quantity == 0:
        raise ValueError("quantity must be positive")
    if discount > 10000:
        raise ValueError("discount too large")
    gross = quantity * unit_cents
    return gross - (gross * discount + 5000) // 10000


def summarize_invoice(invoice):
    if type(invoice) is not dict:
        raise TypeError("invoice must be dict")
    lines = invoice.get("lines", [])
    if type(lines) is not list:
        raise TypeError("lines must be a list")
    shipping_cents = _nonnegative_int(invoice.get("shipping_cents", 0), "shipping")
    shipping_taxable = invoice.get("shipping_taxable", False)
    if type(shipping_taxable) is not bool:
        raise TypeError("shipping_taxable must be bool")

    taxable_cents = shipping_cents if shipping_taxable else 0
    nontaxable_cents = 0 if shipping_taxable else shipping_cents
    for line in lines:
        subtotal_cents = _line_total(line)
        taxable = line.get("taxable", True)
        if type(taxable) is not bool:
            raise TypeError("taxable must be bool")
        if taxable:
            taxable_cents += subtotal_cents
        else:
            nontaxable_cents += subtotal_cents

    tax_cents = (taxable_cents * tax_basis_points() + 5000) // 10000
    subtotal_cents = taxable_cents + nontaxable_cents
    return {
        "line_count": len(lines),
        "taxable_cents": taxable_cents,
        "nontaxable_cents": nontaxable_cents,
        "subtotal_cents": subtotal_cents,
        "tax_cents": tax_cents,
        "total_cents": subtotal_cents + tax_cents,
    }
'''

if ORIGINAL_SOURCE.count(_EDIT_FROM) != 1:
    raise RuntimeError("copy/edit source must contain exactly one requested edit target")
EDITED_SOURCE = ORIGINAL_SOURCE.replace(_EDIT_FROM, _EDIT_TO)


BEHAVIOR_TESTS: tuple[Mapping[str, Any], ...] = (
    {
        "args": [{"lines": []}],
        "expected": {
            "line_count": 0, "taxable_cents": 0, "nontaxable_cents": 0,
            "subtotal_cents": 0, "tax_cents": 0, "total_cents": 0,
        },
        "assert_inputs_unchanged": True,
    },
    {
        "args": [{
            "lines": [{"quantity": 2, "unit_cents": 1000, "taxable": True}],
            "shipping_cents": 500,
        }],
        "expected": {
            "line_count": 1, "taxable_cents": 2000, "nontaxable_cents": 500,
            "subtotal_cents": 2500, "tax_cents": 300, "total_cents": 2800,
        },
        "assert_inputs_unchanged": True,
    },
    {
        "args": [{
            "lines": [{
                "quantity": 3, "unit_cents": 999, "discount_basis_points": 2500,
                "taxable": False,
            }],
            "shipping_cents": 100, "shipping_taxable": True,
        }],
        "expected": {
            "line_count": 1, "taxable_cents": 100, "nontaxable_cents": 2248,
            "subtotal_cents": 2348, "tax_cents": 15, "total_cents": 2363,
        },
        "assert_inputs_unchanged": True,
    },
    {
        "args": [{
            "lines": [{"quantity": 1, "unit_cents": 333}],
            "shipping_cents": 67, "shipping_taxable": True,
        }],
        "expected": {
            "line_count": 1, "taxable_cents": 400, "nontaxable_cents": 0,
            "subtotal_cents": 400, "tax_cents": 60, "total_cents": 460,
        },
        "assert_inputs_unchanged": True,
    },
    {
        "args": [{
            "lines": [
                {"quantity": 1, "unit_cents": 100, "discount_basis_points": 3333},
                {"quantity": 2, "unit_cents": 75, "taxable": False},
            ],
            "shipping_cents": 1,
        }],
        "expected": {
            "line_count": 2, "taxable_cents": 67, "nontaxable_cents": 151,
            "subtotal_cents": 218, "tax_cents": 10, "total_cents": 228,
        },
        "assert_inputs_unchanged": True,
    },
    {"args": [{"lines": [{"quantity": 0, "unit_cents": 100}]}], "raises": "ValueError"},
    {"args": [{"lines": [{"quantity": 1, "unit_cents": 100, "discount_basis_points": 10001}]}], "raises": "ValueError"},
    {"args": [{"lines": [{"quantity": True, "unit_cents": 100}]}], "raises": "TypeError"},
)


PROMPT = (
    "Apply one mechanical edit to the Python module below. Change the unique assignment "
    "TAX_BASIS_POINTS = 1200 to TAX_BASIS_POINTS = 1500. Return the entire module verbatim "
    "with that one numeric change and no other edits or commentary. You may return raw Python "
    "or place the whole module inside one outer Python code fence. Preserve names, whitespace, "
    "strings, line order, and blank lines.\n\nBEGIN MODULE\n"
    + ORIGINAL_SOURCE
    + "END MODULE"
)


def make_case() -> BenchmarkCase:
    return BenchmarkCase(
        case_id=CASE_ID,
        suite="quality",
        prompt=PROMPT,
        max_tokens=OUTPUT_TOKENS,
        evaluator="copy_edit_module",
        function_name="summarize_invoice",
        tests=BEHAVIOR_TESTS,
        metadata={
            "difficulty": "matched_copy_edit",
            "category": "predictive_copy_edit",
            "hidden_test_count": len(BEHAVIOR_TESTS),
            "independent_task_count": 1,
            "requested_edit": {"from": 1200, "to": 1500},
            "scoring_dimensions": ("semantic_behavior", "source_edit_fidelity"),
        },
    )


def _candidate_source(text: str) -> tuple[str | None, str]:
    match = _OUTER_PYTHON_FENCE.fullmatch(text)
    if match:
        return match.group(1), "outer_python_fence"
    if "```" in text:
        return None, "invalid_code_fence"
    return text, "raw_python"


def _normalize_final_newline(source: str) -> str:
    return source.rstrip("\r\n") + "\n"


def evaluate_response(text: str, *, truncated: bool = False) -> dict[str, Any]:
    if truncated:
        return {
            "passed": False,
            "detail": "response was incomplete or length-truncated; behavior and fidelity were not scored",
            "semantic_passed": None,
            "fidelity_passed": False,
            "truncated": True,
        }
    source, response_format = _candidate_source(text)
    if source is None:
        return {
            "passed": False,
            "detail": "response did not contain raw Python or exactly one outer Python code fence",
            "semantic_passed": False,
            "fidelity_passed": False,
            "response_format": response_format,
        }

    behavior = bench_cases.run_python_checks(source, "summarize_invoice", BEHAVIOR_TESTS)
    semantic_passed = behavior.get("passed") is True
    fidelity_passed = _normalize_final_newline(source) == _normalize_final_newline(EDITED_SOURCE)
    if semantic_passed and fidelity_passed:
        detail = "behavior checks passed and the only source edit was TAX_BASIS_POINTS 1200 to 1500"
    elif semantic_passed:
        detail = "behavior checks passed but source edit fidelity failed"
    elif fidelity_passed:
        detail = "source edit fidelity passed but behavior checks failed"
    else:
        detail = "behavior checks and source edit fidelity failed"
    return {
        "passed": semantic_passed and fidelity_passed,
        "detail": detail,
        "semantic_passed": semantic_passed,
        "fidelity_passed": fidelity_passed,
        "behavior_detail": behavior.get("detail"),
        "behavior_validation": behavior.get("validation"),
        "hidden_tests_passed": behavior.get("tests_passed", 0),
        "hidden_test_count": len(BEHAVIOR_TESTS),
        "response_format": response_format,
        "source_chars": len(source),
    }


def run(args: argparse.Namespace) -> int:
    case = make_case()
    config = bench.RunConfig(
        endpoint=args.endpoint,
        backend=args.backend,
        model=args.model,
        context=args.context,
        thinking=args.thinking,
        temperature=args.temperature,
        seed=args.seed,
        num_predict=args.num_predict,
        timeout=args.timeout,
        api_key_env=args.api_key_env,
        chat_template_thinking=args.chat_template_thinking,
    )
    bench.validate_request_controls(config)
    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    run_id = str(uuid.uuid4())
    source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    failed = False

    with output.open("a", encoding="utf-8", buffering=1) as handle:
        for repeat_index in range(args.repeat):
            call_id = str(uuid.uuid4())
            started_at = bench.utc_now()
            before = None if args.skip_telemetry else bench.collect_telemetry(args.endpoint)
            call: dict[str, Any] | None = None
            error: dict[str, str] | None = None
            try:
                call = bench.run_request(config, case)
            except (bench.BenchmarkHTTPError, TimeoutError, OSError, ValueError) as exc:
                error = {"type": type(exc).__name__, "message": str(exc)}
            after = None if args.skip_telemetry else bench.collect_telemetry(args.endpoint)
            evaluation = (
                evaluate_response(call["response_text"], truncated=bool(call["truncated"]))
                if call is not None
                else {
                    "passed": False, "detail": "request failed",
                    "semantic_passed": None, "fidelity_passed": None,
                }
            )
            failed = failed or evaluation.get("passed") is not True
            record = {
                "schema_version": 1,
                "run_id": run_id,
                "call_id": call_id,
                "started_at": started_at,
                "finished_at": bench.utc_now(),
                "repeat_index": repeat_index,
                "suite": case.suite,
                "case_id": case.case_id,
                "case_version": CASE_VERSION,
                "case_source_sha256": source_hash,
                "case_metadata": dict(case.metadata),
                "request": bench._request_config(config, case),
                "telemetry_before": before,
                "telemetry_after": after,
                "response": call,
                "evaluation": evaluation,
                "error": error,
            }
            bench._append_jsonl(handle, record)
            status = "PASS" if evaluation.get("passed") else "FAIL"
            print(
                f"{status} {case.case_id} repeat={repeat_index + 1}/{args.repeat} call_id={call_id}",
                file=sys.stderr,
                flush=True,
            )
    return 1 if failed else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--backend", choices=("openai",), default="openai")
    parser.add_argument("--model", required=True)
    parser.add_argument("--context", type=bench.positive_int, required=True)
    parser.add_argument("--thinking", choices=("none", "low", "medium", "high"), default="none")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-predict", type=bench.positive_int, default=OUTPUT_TOKENS)
    parser.add_argument("--timeout", type=bench.positive_float, default=300.0)
    parser.add_argument("--repeat", type=bench.positive_int, default=1)
    parser.add_argument("--output", required=True)
    parser.add_argument("--chat-template-thinking", choices=("on", "off"))
    parser.add_argument("--api-key-env")
    parser.add_argument("--skip-telemetry", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.temperature < 0:
        parser.error("--temperature must be non-negative")
    try:
        return run(args)
    except ValueError as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
