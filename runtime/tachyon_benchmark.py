#!/usr/bin/env python3
"""Tachyon screening adapter for the frozen public benchmark pilot.

This module owns no model process and never imports model code.  It validates
and reuses the frozen endpoint runner for durable requests, locks, resume, SSE,
usage accounting, and official grader integration.  Official grading must be
invoked from the parent-owned grader container with
``TACHYON_GRADER_SANDBOX=1``.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import statistics
import sys
from types import ModuleType
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
FROZEN_RUNNER = (
    ROOT
    / "results/public-benchmark-pilot-20261023-v1/artifacts/public_benchmark_pilot.py"
)
FROZEN_RUNNER_SHA256 = (
    "a661ac74f414769b80a2484e072924de75bc2935ee470347d7244797dbc945dc"
)
EVALPLUS_ROOT = ROOT / "vendor/public-benchmarks/evalplus-v0.3.1"
IFEVAL_ROOT = (
    ROOT
    / "vendor/public-benchmarks/google-research-e6890f85757dd84e27ca6df2dd30651dafad28e0"
)
MAX_TACHYON_TOKENS = 8_192
TACHYON_CONTEXT_TOKENS = 16_384
SCREENING_LABEL = "tachyon-16k-context-8k-output-public-screening"
MODEL_SOURCES: dict[str, dict[str, str]] = {
    "bonsai2-2b": {
        "url": "https://huggingface.co/prism-ml/Ternary-Bonsai-2-2B-gguf",
        "revision": "683f7be2629342a50987627b40881e54761a2430",
    },
    "bonsai2-4b": {
        "url": "https://huggingface.co/prism-ml/Ternary-Bonsai-2-4B-gguf",
        "revision": "8d6557168168f91ea6697a4a49ed2ef622b53adb",
    },
    "lfm25-26b": {
        "url": "https://huggingface.co/LiquidAI/LFM2.5-2.6B-GGUF",
        "revision": "e7caca5d835a3901a8e0d63e94009429bafafdfc",
    },
}
PROFILE_MODEL_KEYS = {
    "bonsai-nothink": frozenset(("bonsai2-2b", "bonsai2-4b")),
    "lfm26": frozenset(("lfm25-26b",)),
}


class TachyonBenchmarkError(RuntimeError):
    """The adapter contract or a frozen-runner precondition failed."""


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def load_frozen_runner(
    path: Path = FROZEN_RUNNER,
    expected_sha256: str = FROZEN_RUNNER_SHA256,
) -> ModuleType:
    """Hash-check the runner before allowing Python to parse or execute it."""
    source = path.resolve()
    try:
        raw = source.read_bytes()
    except OSError as exc:
        raise TachyonBenchmarkError(f"frozen runner is unreadable: {source}") from exc
    actual = hashlib.sha256(raw).hexdigest()
    if actual != expected_sha256:
        raise TachyonBenchmarkError(
            f"frozen runner SHA-256 mismatch: expected {expected_sha256}, got {actual}"
        )
    module_name = "_tachyon_frozen_public_benchmark_pilot"
    spec = importlib.util.spec_from_file_location(module_name, source)
    if spec is None or spec.loader is None:
        raise TachyonBenchmarkError(f"cannot import frozen runner: {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


def native_recipe(profile: str, model_key: str) -> dict[str, Any]:
    allowed = PROFILE_MODEL_KEYS.get(profile)
    if allowed is None:
        raise TachyonBenchmarkError(f"unsupported profile: {profile}")
    if model_key not in allowed:
        choices = ", ".join(sorted(allowed))
        raise TachyonBenchmarkError(
            f"profile {profile} requires model-key {choices}; got {model_key}"
        )
    if profile == "bonsai-nothink":
        request_fields: dict[str, Any] = {
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,
            "min_p": 0.0,
            "presence_penalty": 1.5,
            "repeat_penalty": 1.0,
            "repeat_last_n": 64,
            "reasoning_effort": "none",
            "chat_template_kwargs": {"enable_thinking": False},
        }
        thinking_mode = "publisher_nonthinking"
    else:
        request_fields = {
            "temperature": 0.1,
            "top_p": 1.0,
            "top_k": 50,
            "min_p": 0.0,
            "repeat_penalty": 1.1,
            "reasoning_effort": "none",
        }
        thinking_mode = "native_nonthinking_default"
    return {
        "schema_version": 1,
        "profile": profile,
        "model_key": model_key,
        "model_card": dict(MODEL_SOURCES[model_key]),
        "thinking_mode": thinking_mode,
        "request_fields": request_fields,
    }


def _validate_native_recipe(settings: Mapping[str, Any]) -> dict[str, Any]:
    profile = settings.get("original_profile")
    model_key = settings.get("model_key")
    if not isinstance(profile, str) or not isinstance(model_key, str):
        raise TachyonBenchmarkError("run settings lack profile/model recipe binding")
    expected = native_recipe(profile, model_key)
    actual = settings.get("native_recipe")
    if actual != expected:
        raise TachyonBenchmarkError("native recipe differs from the bound publisher recipe")
    expected_sha = _sha256_json(expected)
    if settings.get("native_recipe_sha256") != expected_sha:
        raise TachyonBenchmarkError("native recipe SHA-256 is invalid")
    return expected


def adapter_run_settings(
    runner: ModuleType, args: argparse.Namespace
) -> dict[str, Any]:
    if isinstance(args.max_tokens, bool) or not isinstance(args.max_tokens, int):
        raise TachyonBenchmarkError("max-tokens must be an integer")
    if not 1 <= args.max_tokens <= MAX_TACHYON_TOKENS:
        raise TachyonBenchmarkError(
            f"max-tokens must be from 1 through {MAX_TACHYON_TOKENS} for Tachyon"
        )
    recipe = native_recipe(args.profile, args.model_key)
    compatible = copy.copy(args)
    # The frozen runner predates the llama.cpp "none" spelling.  Validate all
    # of its other inputs through its exported function, then bind our exact
    # native request recipe below.
    compatible.reasoning_effort = "low"
    compatible.temperature = recipe["request_fields"]["temperature"]
    settings = runner.run_settings(compatible)
    settings.update(
        {
            "reasoning_effort": "none",
            "temperature": recipe["request_fields"]["temperature"],
            "original_profile": args.profile,
            "native_recipe": recipe,
            "native_recipe_sha256": _sha256_json(recipe),
            "frozen_runner_sha256": FROZEN_RUNNER_SHA256,
            "screening": {
                "label": SCREENING_LABEL,
                "context_window_tokens": TACHYON_CONTEXT_TOKENS,
                "max_output_tokens": args.max_tokens,
                "full_leaderboard_claim": False,
                "per_phase_publisher_recipes": True,
                "comparison_policy": (
                    "published per-model recipes; not a strict same-sampler comparison"
                ),
            },
        }
    )
    _validate_native_recipe(settings)
    return settings


def adapter_request_payload(
    runner: ModuleType, case: Any, settings: Mapping[str, Any]
) -> dict[str, Any]:
    recipe = _validate_native_recipe(settings)
    payload = runner.build_request_payload(case, settings)
    payload.update(copy.deepcopy(recipe["request_fields"]))
    return payload


def install_runner_wrappers(runner: ModuleType) -> None:
    """Wrap only the two documented frozen-runner extension seams."""
    original_settings = runner.run_settings
    original_payload = runner.build_request_payload

    def wrapped_settings(args: argparse.Namespace) -> dict[str, Any]:
        original = runner.run_settings
        try:
            runner.run_settings = original_settings
            return adapter_run_settings(runner, args)
        finally:
            runner.run_settings = original

    def wrapped_payload(case: Any, settings: Mapping[str, Any]) -> dict[str, Any]:
        original = runner.build_request_payload
        try:
            runner.build_request_payload = original_payload
            return adapter_request_payload(runner, case, settings)
        finally:
            runner.build_request_payload = original

    runner.run_settings = wrapped_settings
    runner.build_request_payload = wrapped_payload


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TachyonBenchmarkError(f"unreadable or invalid JSON: {path}") from exc


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    parsed = float(value)
    return parsed if math.isfinite(parsed) else None


def _integer(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _selected_results(output_dir: Path, run: Mapping[str, Any]) -> list[dict[str, Any]]:
    status = _read_json(output_dir / "status.json")
    if not isinstance(status, dict) or not isinstance(status.get("cases"), list):
        raise TachyonBenchmarkError("status.json lacks case records")
    manifest_sha = run.get("manifest", {}).get("sha256")
    settings_sha = run.get("settings_sha256")
    selected: list[dict[str, Any]] = []
    for record in status["cases"]:
        if not isinstance(record, dict):
            continue
        suite = record.get("suite")
        storage_id = record.get("storage_id")
        if not isinstance(suite, str) or not isinstance(storage_id, str):
            continue
        case_dir = output_dir / "cases" / suite / storage_id
        for result_path in sorted(case_dir.glob("attempt-*/result.json")):
            result = _read_json(result_path)
            if not isinstance(result, dict):
                continue
            if result.get("manifest_sha256") != manifest_sha:
                raise TachyonBenchmarkError(f"manifest binding mismatch: {result_path}")
            if result.get("settings_sha256") != settings_sha:
                raise TachyonBenchmarkError(f"settings binding mismatch: {result_path}")
            if result.get("status") in {"completed", "truncated"}:
                selected.append({"suite": suite, "result": result})
                break
    return selected


def _metric_row(
    scope: str,
    statuses: list[str],
    selected: list[Mapping[str, Any]],
) -> dict[str, Any]:
    status_names = (
        "not_started",
        "completed",
        "truncated",
        "api_failure",
        "incomplete",
        "cancelled",
        "invalid",
    )
    row: dict[str, Any] = {"scope": scope, "case_count": len(statuses)}
    for name in status_names:
        row[f"count_{name}"] = sum(value == name for value in statuses)
    prompt_tokens: list[int] = []
    completion_tokens: list[int] = []
    total_tokens: list[int] = []
    reasoning_tokens: list[int] = []
    walls: list[float] = []
    ttfts: list[float] = []
    final_ttfts: list[float] = []
    native_predicted_n = 0.0
    native_predicted_ms = 0.0
    native_cases = 0
    server_reported_rate_ms = 0.0
    server_reported_ms = 0.0
    server_reported_cases = 0
    authoritative_cases = 0
    for result in selected:
        counts = result.get("counts")
        if isinstance(counts, dict):
            if counts.get("token_counts_authoritative") is True:
                authoritative_cases += 1
            for collection, key in (
                (prompt_tokens, "prompt_tokens"),
                (completion_tokens, "completion_tokens"),
                (total_tokens, "total_tokens"),
                (reasoning_tokens, "reasoning_tokens"),
            ):
                value = _integer(counts.get(key))
                if value is not None:
                    collection.append(value)
        timing = result.get("timing")
        if isinstance(timing, dict):
            for collection, key in (
                (walls, "wall_seconds"),
                (ttfts, "ttft_seconds"),
                (final_ttfts, "ttft_final_seconds"),
            ):
                value = _number(timing.get(key))
                if value is not None:
                    collection.append(value)
        native = result.get("native_timings")
        if isinstance(native, dict):
            predicted_n = _number(native.get("predicted_n"))
            predicted_ms = _number(native.get("predicted_ms"))
            if predicted_n is not None and predicted_ms is not None and predicted_ms > 0:
                native_predicted_n += predicted_n
                native_predicted_ms += predicted_ms
                native_cases += 1
            predicted_per_second = _number(native.get("predicted_per_second"))
            if (
                predicted_per_second is not None
                and predicted_ms is not None
                and predicted_ms > 0
            ):
                server_reported_rate_ms += predicted_per_second * predicted_ms
                server_reported_ms += predicted_ms
                server_reported_cases += 1
    count_time_rate = (
        native_predicted_n * 1000.0 / native_predicted_ms
        if native_predicted_ms > 0
        else None
    )
    row.update(
        {
            "selected_result_count": len(selected),
            "authoritative_usage_case_count": authoritative_cases,
            "prompt_tokens": sum(prompt_tokens),
            "completion_tokens": sum(completion_tokens),
            "total_tokens": sum(total_tokens),
            "reasoning_tokens_reported": sum(reasoning_tokens),
            "reasoning_token_case_count": len(reasoning_tokens),
            "native_timing_case_count": native_cases,
            "native_completion_tokens_per_second": count_time_rate,
            "count_time_derived_generation_tokens_per_second": count_time_rate,
            "server_reported_generation_timing_case_count": server_reported_cases,
            "server_reported_generation_tokens_per_second": (
                server_reported_rate_ms / server_reported_ms
                if server_reported_ms > 0
                else None
            ),
            "wall_seconds_sum": sum(walls),
            "wall_seconds_mean": _mean(walls),
            "ttft_seconds_mean": _mean(ttfts),
            "ttft_seconds_median": statistics.median(ttfts) if ttfts else None,
            "ttft_final_seconds_mean": _mean(final_ttfts),
        }
    )
    return row


def build_summary(output_dir: Path) -> dict[str, Any]:
    output_dir = output_dir.expanduser().resolve()
    run = _read_json(output_dir / "run.json")
    status = _read_json(output_dir / "status.json")
    if not isinstance(run, dict) or not isinstance(run.get("settings"), dict):
        raise TachyonBenchmarkError("run.json lacks settings")
    if not isinstance(status, dict) or not isinstance(status.get("cases"), list):
        raise TachyonBenchmarkError("status.json lacks case records")
    settings = run["settings"]
    _validate_native_recipe(settings)
    if settings.get("frozen_runner_sha256") != FROZEN_RUNNER_SHA256:
        raise TachyonBenchmarkError("run was not bound to the expected frozen runner")
    if run.get("settings_sha256") != _sha256_json(settings):
        raise TachyonBenchmarkError("run settings SHA-256 is invalid")
    selected_records = _selected_results(output_dir, run)
    records = [record for record in status["cases"] if isinstance(record, dict)]
    rows: list[dict[str, Any]] = []
    for scope in ("all", "humanevalplus", "ifeval"):
        scoped_records = (
            records if scope == "all" else [r for r in records if r.get("suite") == scope]
        )
        scoped_results = [
            item["result"]
            for item in selected_records
            if scope == "all" or item["suite"] == scope
        ]
        rows.append(
            _metric_row(
                scope,
                [str(record.get("status", "invalid")) for record in scoped_records],
                scoped_results,
            )
        )
    grading_path = output_dir / "grading" / "summary.json"
    grading: dict[str, Any] | None = None
    if grading_path.is_file():
        source = _read_json(grading_path)
        if isinstance(source, dict):
            evalplus = source.get("evalplus") if isinstance(source.get("evalplus"), dict) else {}
            ifeval = source.get("ifeval") if isinstance(source.get("ifeval"), dict) else {}
            strict = ifeval.get("strict") if isinstance(ifeval.get("strict"), dict) else {}
            loose = ifeval.get("loose") if isinstance(ifeval.get("loose"), dict) else {}
            grading = {
                "status": source.get("status"),
                "evalplus_base_pass_at_1": evalplus.get("base_pass_at_1"),
                "evalplus_plus_pass_at_1": evalplus.get("plus_pass_at_1"),
                "ifeval_strict_prompt_accuracy": strict.get("prompt_accuracy"),
                "ifeval_strict_instruction_accuracy": strict.get("instruction_accuracy"),
                "ifeval_loose_prompt_accuracy": loose.get("prompt_accuracy"),
                "ifeval_loose_instruction_accuracy": loose.get("instruction_accuracy"),
            }
    return {
        "schema_version": 1,
        "kind": "tachyon_public_screening_summary",
        "screening_label": SCREENING_LABEL,
        "full_leaderboard_claim": False,
        "output_dir": str(output_dir),
        "manifest_sha256": run.get("manifest", {}).get("sha256"),
        "settings_sha256": run.get("settings_sha256"),
        "model": settings.get("model"),
        "model_key": settings.get("model_key"),
        "profile": settings.get("original_profile"),
        "native_recipe_sha256": settings.get("native_recipe_sha256"),
        "rows": rows,
        "grading": grading,
    }


def _write_summary(runner: ModuleType, path: Path, summary: Mapping[str, Any]) -> None:
    destination = path.expanduser().resolve()
    suffix = destination.suffix.lower()
    if suffix == ".json":
        runner.atomic_write_json(destination, summary)
        return
    if suffix != ".csv":
        raise TachyonBenchmarkError("summary output must end in .json or .csv")
    grading = summary.get("grading") or {}
    rows: list[dict[str, Any]] = []
    for metric in summary["rows"]:
        row = {
            "model": summary.get("model"),
            "model_key": summary.get("model_key"),
            "profile": summary.get("profile"),
            "screening_label": summary.get("screening_label"),
            **metric,
            **grading,
        }
        rows.append(row)
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    runner.atomic_write_bytes(destination, buffer.getvalue().encode("utf-8"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="run or resume the Tachyon screening")
    run.add_argument("--manifest", required=True)
    run.add_argument("--endpoint", default="http://127.0.0.1:18185/v1")
    run.add_argument("--model", required=True, help="OpenAI API model alias")
    run.add_argument("--model-key", required=True, choices=sorted(MODEL_SOURCES))
    run.add_argument("--profile", required=True, choices=sorted(PROFILE_MODEL_KEYS))
    run.add_argument("--output-dir", required=True)
    run.add_argument("--max-tokens", type=int, default=MAX_TACHYON_TOKENS)
    run.add_argument("--request-timeout", type=float, default=3_600.0)
    run.add_argument("--seed", type=int, default=20_261_023)
    run.add_argument("--api-key-env", default=None)
    run.add_argument("--rerun-failed", action="store_true")

    grade = subparsers.add_parser(
        "grade", help="delegate official grading (inside the grader sandbox only)"
    )
    grade.add_argument("--manifest", required=True)
    grade.add_argument("--output-dir", required=True)
    grade.add_argument("--evalplus-root", default=str(EVALPLUS_ROOT))
    grade.add_argument("--ifeval-root", default=str(IFEVAL_ROOT))
    grade.add_argument("--evalplus-workers", type=int, default=4)
    grade.add_argument("--grader-timeout", type=float, default=21_600.0)

    summarize = subparsers.add_parser("summarize", help="write JSON or CSV metrics")
    summarize.add_argument("--output-dir", required=True)
    summarize.add_argument("--output", required=True)
    return parser


def dispatch(args: argparse.Namespace) -> int:
    runner = load_frozen_runner()
    if args.command == "run":
        install_runner_wrappers(runner)
        return runner.run_command(args)
    if args.command == "grade":
        if os.environ.get("TACHYON_GRADER_SANDBOX") != "1":
            raise TachyonBenchmarkError(
                "grade is restricted to the parent-owned grader sandbox; "
                "set TACHYON_GRADER_SANDBOX=1 inside that container"
            )
        return runner.grade_command(args)
    if args.command == "summarize":
        summary = build_summary(Path(args.output_dir))
        _write_summary(runner, Path(args.output), summary)
        return 0
    raise TachyonBenchmarkError(f"unknown command: {args.command}")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return dispatch(args)
    except TachyonBenchmarkError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        # Preserve the frozen runner's concise CLI treatment without binding
        # this adapter to a class imported before its hash is verified.
        if exc.__class__.__name__ == "PilotError":
            print(f"error: {exc}", file=sys.stderr)
            return 2
        raise


if __name__ == "__main__":
    raise SystemExit(main())
