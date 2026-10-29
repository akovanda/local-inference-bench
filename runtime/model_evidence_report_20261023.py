#!/usr/bin/env python3
"""Build the 2026-10-23 cross-study evidence report without regrading."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import hashlib
import html
import io
import json
import math
import os
from pathlib import Path
import shutil
from typing import Any, Iterable, Mapping, Sequence

try:
    from runtime import public_evidence
except ModuleNotFoundError:  # Direct execution from runtime/.
    import public_evidence  # type: ignore[no-redef]


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = RESULTS / "model-evidence-20261023-v2"
LEGACY = RESULTS / "complete-study-dashboard.csv"
PUBLIC_T4 = RESULTS / "public-benchmark-pilot-20261023-v1"
PUBLIC_BOARD = RESULTS / "tachyon-study-20261023-v1"
WORKER = RESULTS / "tachyon-worker-study-20261023-v1"
LATENCY = RESULTS / "tachyon-worker-latency-20261023-v1"
PAIRED_COMPARISON = PUBLIC_T4 / "final-paired-comparison.json"
GRADER_CAVEATS = PUBLIC_T4 / "official-grader-caveats.json"
QNN_ASSESSMENT = RESULTS / "tachyon-qnn-investigation-20261024-v1/assessment.json"
_ACTIVE_EVIDENCE: public_evidence.VerifiedEvidence | None = None

EVIDENCE_FIELDS = [
    "record_id", "model", "profile", "study_family", "hardware", "suite",
    "case_id", "status", "success_definition", "pass", "input_tokens",
    "output_tokens", "reasoning_tokens", "native_generation_tps",
    "native_predicted_tokens", "native_predicted_ms", "wall_seconds",
    "ttft_first_seconds", "ttft_visible_seconds", "context_cap_tokens",
    "output_cap_tokens", "request_timeout_seconds", "capped", "caps_json",
    "source_path", "grade_source_path", "notes",
]

GROUP_FIELDS = [
    "model", "profile", "suite", "study_family", "hardware",
    "success_definition", "n", "pass", "unknown", "caps", "output_tokens",
    "weighted_native_tps", "median_wall_seconds", "p95_wall_seconds",
    "total_wall_seconds",
]

ACCELERATOR_FIELDS = [
    "variant", "backend", "hardware", "status", "generation_tps",
    "prompt_tps", "requested_gpu_layers", "reported_offloaded_layers",
    "quality_claim", "interpretation", "source_path",
]

EXCLUSION_FIELDS = [
    "evidence", "status", "included_in_817_stable_rows", "interpretation",
    "paths",
]


class ReportError(RuntimeError):
    pass


def _published(path: Path) -> Path:
    if _ACTIVE_EVIDENCE is None:
        return path
    try:
        return _ACTIVE_EVIDENCE.ensure_published_file(path)
    except public_evidence.EvidenceError as exc:
        raise ReportError(str(exc)) from exc


def resolve_source(value: Any, *bases: Path, required: bool = False) -> Path | None:
    """Resolve a nested source path without cwd fallback in public mode."""
    if not isinstance(value, str) or not value:
        if required:
            raise ReportError("required evidence reference is empty")
        return None
    if _ACTIVE_EVIDENCE is None:
        candidate = Path(value)
        if candidate.is_absolute():
            return candidate
        if candidate.parts and candidate.parts[0] in {"results", "runtime", "deployment", "tests"}:
            return ROOT / candidate
        for base in bases:
            joined = base / candidate
            if joined.exists():
                return joined
        return (bases[0] / candidate) if bases else candidate
    try:
        return _ACTIVE_EVIDENCE.resolve(value, bases=bases, required=required)
    except public_evidence.EvidenceError as exc:
        raise ReportError(str(exc)) from exc


@contextmanager
def _public_root(evidence: public_evidence.VerifiedEvidence | None):
    global ROOT, RESULTS, OUT, LEGACY, PUBLIC_T4, PUBLIC_BOARD, WORKER, LATENCY
    global PAIRED_COMPARISON, GRADER_CAVEATS, QNN_ASSESSMENT, _ACTIVE_EVIDENCE
    if evidence is None:
        yield
        return
    previous = (
        ROOT, RESULTS, OUT, LEGACY, PUBLIC_T4, PUBLIC_BOARD, WORKER, LATENCY,
        PAIRED_COMPARISON, GRADER_CAVEATS, QNN_ASSESSMENT, _ACTIVE_EVIDENCE,
    )
    ROOT = evidence.root
    RESULTS = ROOT / "results"
    OUT = RESULTS / "model-evidence-20261023-v2"
    LEGACY = RESULTS / "complete-study-dashboard.csv"
    PUBLIC_T4 = RESULTS / "public-benchmark-pilot-20261023-v1"
    PUBLIC_BOARD = RESULTS / "tachyon-study-20261023-v1"
    WORKER = RESULTS / "tachyon-worker-study-20261023-v1"
    LATENCY = RESULTS / "tachyon-worker-latency-20261023-v1"
    PAIRED_COMPARISON = PUBLIC_T4 / "final-paired-comparison.json"
    GRADER_CAVEATS = PUBLIC_T4 / "official-grader-caveats.json"
    QNN_ASSESSMENT = RESULTS / "tachyon-qnn-investigation-20261024-v1/assessment.json"
    _ACTIVE_EVIDENCE = evidence
    try:
        yield
    finally:
        (
            ROOT, RESULTS, OUT, LEGACY, PUBLIC_T4, PUBLIC_BOARD, WORKER, LATENCY,
            PAIRED_COMPARISON, GRADER_CAVEATS, QNN_ASSESSMENT, _ACTIVE_EVIDENCE,
        ) = previous


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_json(path: Path) -> Any:
    path = _published(path)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ReportError(f"invalid JSON: {path}") from exc


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    path = _published(path)
    rows: list[dict[str, Any]] = []
    try:
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ReportError(f"non-object JSONL row {path}:{number}")
                rows.append(value)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ReportError(f"invalid JSONL: {path}") from exc
    return rows


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative(path: Path | str) -> str:
    value = Path(path)
    try:
        return str(value.resolve().relative_to(ROOT))
    except ValueError:
        return str(value)


def number(value: Any) -> int | float | None:
    if value is None or value == "" or isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return int(parsed) if parsed.is_integer() else parsed


def boolean(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        if value.lower() == "true":
            return True
        if value.lower() == "false":
            return False
    return None


def percentile(values: Sequence[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(probability * len(ordered)) - 1)]


def median(values: Sequence[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2


def json_cell(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def empty_row() -> dict[str, Any]:
    return {field: None for field in EVIDENCE_FIELDS}


def write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="raise")
    writer.writeheader()
    for source in rows:
        row = {}
        for field in fieldnames:
            value = source.get(field)
            if isinstance(value, bool):
                value = "true" if value else "false"
            row[field] = "" if value is None else value
        writer.writerow(row)
    path.write_text(output.getvalue(), encoding="utf-8")


def legacy_rows() -> list[dict[str, Any]]:
    with _published(LEGACY).open(newline="", encoding="utf-8") as stream:
        source_rows = list(csv.DictReader(stream))
    if len(source_rows) != 480:
        raise ReportError(f"legacy dashboard row count differs: {len(source_rows)}")
    rows: list[dict[str, Any]] = []
    for source in source_rows:
        row = empty_row()
        decode_seconds = number(source.get("native_decode_seconds"))
        outcome = source.get("outcome_bucket") or source.get("outcome") or "unknown"
        caps = {
            "context_tokens": number(source.get("requested_context_tokens")),
            "output_tokens": number(source.get("output_token_limit")),
            "request_timeout_seconds": number(source.get("request_timeout_seconds")),
            "thinking": source.get("thinking") or None,
            "chat_template_thinking": source.get("chat_template_thinking") or None,
            "reasoning_budget_tokens_requested": number(
                source.get("reasoning_budget_tokens_requested")
            ),
            "sampler_recipe_label": source.get("sampler_recipe_label") or None,
        }
        ttft_reasoning = number(source.get("ttft_reasoning_seconds"))
        ttft_visible = number(source.get("ttft_visible_seconds"))
        row.update({
            "record_id": f"legacy:{source['dashboard_record_id']}",
            "model": source.get("model"),
            "profile": source.get("profile_id") or source.get("profile_key"),
            "study_family": "legacy_complete_study_dashboard",
            "hardware": "legacy T4 study host; see copied original resource fields",
            "suite": source.get("suite"),
            "case_id": source.get("case_id"),
            "status": outcome,
            "success_definition": "legacy_dashboard_quality_passed",
            "pass": boolean(source.get("quality_passed")),
            "input_tokens": number(source.get("input_tokens")),
            "output_tokens": number(source.get("output_tokens")),
            "reasoning_tokens": number(source.get("reasoning_tokens_reported")),
            "native_generation_tps": number(source.get("native_decode_tps")),
            "native_predicted_tokens": number(source.get("native_decode_n")),
            "native_predicted_ms": decode_seconds * 1000 if decode_seconds is not None else None,
            "wall_seconds": number(source.get("wall_seconds")),
            "ttft_first_seconds": ttft_reasoning if ttft_reasoning is not None else ttft_visible,
            "ttft_visible_seconds": ttft_visible,
            "context_cap_tokens": caps["context_tokens"],
            "output_cap_tokens": caps["output_tokens"],
            "request_timeout_seconds": caps["request_timeout_seconds"],
            "capped": outcome == "truncated",
            "caps_json": json_cell(caps),
            "source_path": source.get("source_file"),
            "grade_source_path": (
                "results/complete-study-dashboard.csv#"
                f"dashboard_record_id={source['dashboard_record_id']}"
            ),
            "notes": json_cell({
                "runtime": source.get("runtime"), "backend": source.get("backend"),
                "weight_quant": source.get("weight_quant"),
                "original_fields_preserved_in": (
                    "results/model-evidence-20261023-v2/"
                    "legacy-complete-study-dashboard.csv"
                ),
            }),
        })
        rows.append(row)
    return rows


def terminal_results(model_dir: Path) -> dict[str, tuple[Path, dict[str, Any]]]:
    results: dict[str, tuple[Path, dict[str, Any]]] = {}
    for path in sorted(model_dir.glob("cases/*/*/attempt-*/result.json")):
        value = read_json(path)
        task_id = str(value.get("case", {}).get("task_id", ""))
        if not task_id:
            raise ReportError(f"result lacks task ID: {path}")
        if task_id in results:
            raise ReportError(f"multiple terminal result files for {model_dir.name}:{task_id}")
        results[task_id] = (path, value)
    return results


def public_rows(study: Path, model_key: str, *, study_family: str,
                hardware: str, context_tokens: int, output_tokens: int,
                subset_note: str) -> list[dict[str, Any]]:
    model_dir = study / model_key
    manifest_path = study / "manifest.json"
    manifest = read_json(manifest_path)
    cases = manifest.get("cases", [])
    run = read_json(model_dir / "run.json")
    settings = run.get("settings", {})
    grading_path = model_dir / "grading/summary.json"
    if _ACTIVE_EVIDENCE is not None:
        _published(grading_path)
    grading = read_json(grading_path) if grading_path.is_file() else {}
    plus = {
        str(item["task_id"]): item.get("plus_status")
        for item in grading.get("evalplus", {}).get("per_task", [])
    }
    strict_path_value = grading.get("ifeval", {}).get("strict_results")
    strict_path = resolve_source(
        strict_path_value, grading_path.parent, model_dir, study,
        required=_ACTIVE_EVIDENCE is not None and isinstance(strict_path_value, str),
    )
    strict_rows = read_jsonl(strict_path) if strict_path and strict_path.is_file() else []
    strict: dict[str, bool] = {}
    for item in strict_rows:
        prompt = item.get("prompt")
        if not isinstance(prompt, str) or prompt in strict:
            raise ReportError(f"IFEval strict prompt is missing/duplicate: {strict_path}")
        strict[prompt] = item.get("follow_all_instructions") is True
    results = terminal_results(model_dir)
    profile_parts = [str(settings.get("original_profile") or settings.get("reasoning_effort") or "published")]
    native_recipe = settings.get("native_recipe")
    if isinstance(native_recipe, dict) and native_recipe.get("thinking_mode"):
        profile_parts.append(str(native_recipe["thinking_mode"]))
    profile = "/".join(profile_parts)
    rows: list[dict[str, Any]] = []
    for case in cases:
        task_id = str(case["task_id"])
        result_pair = results.get(task_id)
        path, result = result_pair if result_pair else (model_dir, {})
        suite = case["suite"]
        if suite == "humanevalplus":
            grade_value = plus.get(task_id)
            passed = True if grade_value == "pass" else False if grade_value == "fail" else None
            success_definition = "evalplus_enhanced_plus_per_task_pass"
            grade_source = grading_path
        elif suite == "ifeval":
            passed = strict.get(case["prompt"])
            success_definition = "ifeval_strict_prompt_pass"
            grade_source = strict_path or grading_path
        else:
            raise ReportError(f"unknown public suite: {suite}")
        timing = result.get("timing", {})
        native = result.get("native_timings", {})
        usage = result.get("usage", {})
        counts = result.get("counts", {})
        completion_details = usage.get("completion_tokens_details", {}) if isinstance(usage, dict) else {}
        reasoning_tokens = counts.get("reasoning_tokens")
        if reasoning_tokens is None and isinstance(completion_details, dict):
            reasoning_tokens = completion_details.get("reasoning_tokens")
        status = result.get("status", "not_started")
        capped = result.get("truncated") is True or status == "truncated"
        cap_values = {
            "context_tokens": context_tokens,
            "output_tokens": output_tokens,
            "request_timeout_seconds": settings.get("request_timeout_seconds"),
            "reasoning_effort": settings.get("reasoning_effort"),
            "temperature": settings.get("temperature"),
        }
        row = empty_row()
        row.update({
            "record_id": f"{study_family}:{model_key}:{suite}:{task_id}",
            "model": settings.get("model") or model_key,
            "profile": profile,
            "study_family": study_family,
            "hardware": hardware,
            "suite": suite,
            "case_id": task_id,
            "status": status,
            "success_definition": success_definition,
            "pass": passed,
            "input_tokens": number(usage.get("prompt_tokens")),
            "output_tokens": number(usage.get("completion_tokens")),
            "reasoning_tokens": number(reasoning_tokens),
            "native_generation_tps": number(native.get("predicted_per_second")),
            "native_predicted_tokens": number(native.get("predicted_n")),
            "native_predicted_ms": number(native.get("predicted_ms")),
            "wall_seconds": number(timing.get("wall_seconds")),
            "ttft_first_seconds": number(timing.get("ttft_seconds")),
            "ttft_visible_seconds": number(timing.get("ttft_final_seconds")),
            "context_cap_tokens": context_tokens,
            "output_cap_tokens": output_tokens,
            "request_timeout_seconds": number(settings.get("request_timeout_seconds")),
            "capped": capped,
            "caps_json": json_cell(cap_values),
            "source_path": relative(path),
            "grade_source_path": relative(grade_source),
            "notes": subset_note,
        })
        rows.append(row)
    if len(rows) != len(cases):
        raise ReportError(f"public case count differs for {model_key}")
    return rows


def worker_rows() -> list[dict[str, Any]]:
    manifest = read_json(WORKER / "manifest.json")
    expected_ids = {str(case["id"]) for case in manifest["cases"]}
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for path in sorted(WORKER.glob("models/*/cases/*/result.json")):
        result = read_json(path)
        model_key = path.parts[-4]
        case_id = str(result["case_id"])
        identity = (model_key, case_id)
        if identity in seen:
            raise ReportError(f"duplicate worker result: {identity}")
        seen.add(identity)
        run = read_json(WORKER / "models" / model_key / "run.json")
        settings = run["settings"]
        usage = result.get("usage", {})
        timing = result.get("native_timings", {})
        evaluation = result.get("evaluation", {})
        suite = result["suite"]
        success_definition = (
            "worker_automatic_closed_facts_keys_and_word_limit"
            if suite == "summary" else "worker_exact_closed_contract"
        )
        caps = {
            "context_tokens": settings.get("context_tokens"),
            "output_tokens": settings.get("max_tokens"),
            "request_timeout_seconds": settings.get("request_timeout_seconds"),
            "mode": settings.get("mode"),
            "recipe": settings.get("recipe"),
            "summary_prose_manual_review_required": suite == "summary",
        }
        row = empty_row()
        row.update({
            "record_id": f"worker:{model_key}:{case_id}",
            "model": model_key,
            "profile": str(settings.get("mode_label") or settings.get("mode")),
            "study_family": "tachyon_worker_production_schema_24",
            "hardware": "Tachyon QCM6490 CPU4 (taskset 4-7)",
            "suite": suite,
            "case_id": case_id,
            "status": result.get("status"),
            "success_definition": success_definition,
            "pass": evaluation.get("semantic_pass") if isinstance(evaluation, dict) else None,
            "input_tokens": number(usage.get("prompt_tokens")),
            "output_tokens": number(usage.get("completion_tokens")),
            "reasoning_tokens": number(
                usage.get("completion_tokens_details", {}).get("reasoning_tokens")
                if isinstance(usage.get("completion_tokens_details"), dict) else None
            ),
            "native_generation_tps": number(timing.get("predicted_per_second")),
            "native_predicted_tokens": number(timing.get("predicted_n")),
            "native_predicted_ms": number(timing.get("predicted_ms")),
            "wall_seconds": number(result.get("wall_seconds")),
            "ttft_first_seconds": number(result.get("ttft_first_seconds")),
            "ttft_visible_seconds": number(result.get("ttft_final_seconds")),
            "context_cap_tokens": number(settings.get("context_tokens")),
            "output_cap_tokens": number(settings.get("max_tokens")),
            "request_timeout_seconds": number(settings.get("request_timeout_seconds")),
            "capped": result.get("finish_reason") in {"length", "max_tokens"},
            "caps_json": json_cell(caps),
            "source_path": relative(path),
            "grade_source_path": relative(path) + "#evaluation",
            "notes": (
                "summary prose remains separately manually reviewed"
                if suite == "summary" else "typed production public-schema contract"
            ),
        })
        rows.append(row)
    models = {path.parent.name for path in (WORKER / "models").glob("*/run.json")}
    if len(rows) != len(models) * len(expected_ids) or len(rows) != 120:
        raise ReportError(f"worker row count differs: {len(rows)}")
    return rows


def latency_result_row(path: Path, *, family: str, profile: str) -> dict[str, Any]:
    result = read_json(path)
    usage = result.get("usage", {})
    timing = result.get("native_timings", {})
    evaluation = result.get("evaluation", {})
    request_path = path.with_name("request.json")
    request = read_json(request_path) if request_path.is_file() else {}
    caps = {
        "context_tokens": 16384, "output_tokens": request.get("max_tokens", 512),
        "request_timeout_seconds": None,
        "temperature": request.get("temperature"),
        "reasoning_effort": request.get("reasoning_effort"),
        "reasoning_budget_tokens": request.get("reasoning_budget_tokens"),
    }
    row = empty_row()
    row.update({
        "record_id": f"latency:{family}:{profile}:{result['case_id']}",
        "model": "lfm25-26b",
        "profile": profile,
        "study_family": family,
        "hardware": "Tachyon QCM6490 CPU4 (taskset 4-7)",
        "suite": result.get("suite", "routing"),
        "case_id": result["case_id"],
        "status": result.get("status"),
        "success_definition": "latency_diagnostic_exact_routing_contract",
        "pass": evaluation.get("semantic_pass") if isinstance(evaluation, dict) else None,
        "input_tokens": number(usage.get("prompt_tokens")),
        "output_tokens": number(usage.get("completion_tokens")),
        "reasoning_tokens": number(
            usage.get("completion_tokens_details", {}).get("reasoning_tokens")
            if isinstance(usage.get("completion_tokens_details"), dict) else None
        ),
        "native_generation_tps": number(timing.get("predicted_per_second")),
        "native_predicted_tokens": number(timing.get("predicted_n")),
        "native_predicted_ms": number(timing.get("predicted_ms")),
        "wall_seconds": number(result.get("wall_seconds")),
        "ttft_first_seconds": number(result.get("ttft_first_seconds")),
        "ttft_visible_seconds": number(result.get("ttft_final_seconds")),
        "context_cap_tokens": 16384,
        "output_cap_tokens": number(request.get("max_tokens", 512)),
        "request_timeout_seconds": None,
        "capped": result.get("finish_reason") in {"length", "max_tokens"},
        "caps_json": json_cell(caps),
        "source_path": relative(path),
        "grade_source_path": relative(path) + "#evaluation",
        "notes": "repeated routing latency diagnostic; not an independent quality set",
    })
    return row


def latency_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted((LATENCY / "epochs").glob("*/*/result.json")):
        value = read_json(path)
        epoch = value.get("epoch", {})
        key = str(epoch.get("key") or path.parent.parent.name)
        ubatch = epoch.get("ubatch")
        rows.append(latency_result_row(
            path, family="tachyon_worker_latency_ab",
            profile=f"lfm26-fast-budget0-{key}-ub{ubatch}",
        ))
    for path in sorted((LATENCY / "greedy-diagnostic/epochs").glob("*/*/*/result.json")):
        key = path.parents[2].name
        rows.append(latency_result_row(
            path, family="tachyon_worker_latency_greedy_diagnostic",
            profile=f"lfm26-fast-budget0-{key}",
        ))
    ordinary = LATENCY / "greedy-diagnostic/ordinary-default-probe/result.json"
    rows.append(latency_result_row(
        ordinary, family="tachyon_worker_latency_ordinary_default_probe",
        profile="lfm26-fast-server-default-ordinary-client",
    ))
    identities = {row["record_id"] for row in rows}
    if len(rows) != 41 or len(identities) != len(rows):
        raise ReportError(f"latency row count/identity differs: {len(rows)}/{len(identities)}")
    return rows


def group_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        key = tuple(row[field] for field in (
            "model", "profile", "suite", "study_family", "hardware",
            "success_definition",
        ))
        groups[key].append(row)
    output: list[dict[str, Any]] = []
    for key, scoped in sorted(groups.items(), key=lambda item: tuple(str(x) for x in item[0])):
        walls = [float(row["wall_seconds"]) for row in scoped
                 if isinstance(row.get("wall_seconds"), (int, float))]
        weighted = [
            (float(row["native_generation_tps"]), float(row["native_predicted_ms"]))
            for row in scoped
            if isinstance(row.get("native_generation_tps"), (int, float))
            and isinstance(row.get("native_predicted_ms"), (int, float))
            and float(row["native_predicted_ms"]) > 0
        ]
        output.append({
            "model": key[0], "profile": key[1], "suite": key[2],
            "study_family": key[3], "hardware": key[4],
            "success_definition": key[5],
            "n": len(scoped),
            "pass": sum(row.get("pass") is True for row in scoped),
            "unknown": sum(row.get("pass") is None for row in scoped),
            "caps": sum(row.get("capped") is True for row in scoped),
            "output_tokens": sum(
                int(row["output_tokens"]) for row in scoped
                if isinstance(row.get("output_tokens"), int)
            ),
            "weighted_native_tps": (
                sum(rate * milliseconds for rate, milliseconds in weighted)
                / sum(milliseconds for _, milliseconds in weighted)
                if weighted else None
            ),
            "median_wall_seconds": median(walls),
            "p95_wall_seconds": percentile(walls, .95),
            "total_wall_seconds": sum(walls) if walls else None,
        })
    return output


def accelerator_rows() -> list[dict[str, Any]]:
    model = "lfm25-26b"
    baseline_path = PUBLIC_BOARD / f"{model}-llama-bench.json"
    cpu8_path = PUBLIC_BOARD / "gpu-trial/cpu8-llama-bench.json"
    baseline = read_json(baseline_path)
    cpu8 = read_json(cpu8_path)
    baseline_prompt = next((row.get("avg_ts") for row in baseline if row.get("n_prompt")), None)
    baseline_generation = next((row.get("avg_ts") for row in baseline if row.get("n_gen")), None)
    cpu8_prompt = next((row.get("avg_ts") for row in cpu8 if row.get("n_prompt")), None)
    cpu8_generation = next((row.get("avg_ts") for row in cpu8 if row.get("n_gen")), None)
    rows = [
        {"variant": "cpu4-baseline", "backend": "cpu", "hardware": "Tachyon QCM6490",
         "status": "complete", "generation_tps": baseline_generation,
         "prompt_tps": baseline_prompt, "requested_gpu_layers": 0,
         "reported_offloaded_layers": None, "quality_claim": False,
         "interpretation": "llama-bench CPU4 reference", "source_path": relative(baseline_path)},
        {"variant": "cpu8", "backend": "cpu", "hardware": "Tachyon QCM6490",
         "status": "complete", "generation_tps": cpu8_generation,
         "prompt_tps": cpu8_prompt, "requested_gpu_layers": 0,
         "reported_offloaded_layers": None, "quality_claim": False,
         "interpretation": "CPU8 generation was slower than CPU4 in this probe",
         "source_path": relative(cpu8_path)},
    ]
    old_summary_path = PUBLIC_BOARD / "gpu-trial/summary.json"
    old_summary = read_json(old_summary_path)
    for item in old_summary.get("rows", []):
        if item.get("backend") != "opencl":
            continue
        rows.append({
            "variant": item.get("variant"), "backend": "opencl",
            "hardware": "Tachyon Adreno OpenCL", "status": item.get("status"),
            "generation_tps": None, "prompt_tps": None,
            "requested_gpu_layers": item.get("requested_gpu_layers"),
            "reported_offloaded_layers": item.get("reported_offloaded_layers"),
            "quality_claim": False,
            "interpretation": item.get("error") or old_summary.get("interpretation"),
            "source_path": relative(old_summary_path),
        })
    generic_path = LATENCY / "opencl-generic-ngl4-result.json"
    generic = read_json(generic_path)
    rows.append({
        "variant": generic.get("variant"), "backend": "opencl-generic",
        "hardware": "Tachyon Adreno OpenCL", "status": generic.get("status"),
        "generation_tps": None, "prompt_tps": None,
        "requested_gpu_layers": generic.get("requested_gpu_layers"),
        "reported_offloaded_layers": generic.get("reported_offloaded_layers"),
        "quality_claim": False,
        "interpretation": "compiler failure; no throughput measurement",
        "source_path": relative(generic_path),
    })
    qnn_path = PUBLIC_BOARD / "qnn-llm-compatibility.json"
    qnn = read_json(qnn_path)
    rows.append({
        "variant": "qnn-calculator", "backend": "QNN HTP",
        "hardware": "Tachyon Hexagon v68", "status": "calculator_only_not_llm_inference",
        "generation_tps": None, "prompt_tps": None,
        "requested_gpu_layers": None, "reported_offloaded_layers": None,
        "quality_claim": False, "interpretation": qnn.get("bottom_line"),
        "source_path": relative(qnn_path),
    })
    return rows


def manual_review_rows() -> tuple[list[dict[str, Any]], Path]:
    source = WORKER / "analysis/manual-summary-review.json"
    value = read_json(source)
    rows: list[dict[str, Any]] = []
    for item in value.get("rows", []):
        rows.append({
            "model": item.get("model_key"), "case_id": item.get("case_id"),
            "prose_grounding_review": item.get("prose_grounding_review"),
            "automatic_fact_and_word_limit_pass": item.get(
                "automatic_fact_and_word_limit_pass"
            ),
            "prose_omissions": json_cell(item.get("prose_omissions")),
            "review_note": item.get("review_note"),
            "result_path": item.get("result_path"),
            "result_sha256": item.get("result_sha256"),
        })
    if len(rows) != 40:
        raise ReportError(f"manual summary review count differs: {len(rows)}")
    return rows, source


def evidence_exclusions() -> list[dict[str, Any]]:
    """Index diagnostic/preparatory evidence without treating it as benchmark rows."""
    return [
        {
            "evidence": "Tachyon role study v1",
            "status": read_json(RESULTS / "tachyon-role-study-20261023-v1/study-status.json").get("status"),
            "included_in_817_stable_rows": False,
            "interpretation": "Interrupted diagnostic mode; exposed free-output/parser behavior, not a completed five-model comparison.",
            "paths": [
                "results/tachyon-role-study-20261023-v1/study-status.json",
                "results/tachyon-role-study-20261023-v1/source-pins.json",
                "results/tachyon-role-study-20261023-v1/lfm-fast-profile-interrupt.json",
            ],
        },
        {
            "evidence": "Tachyon role study v2",
            "status": read_json(RESULTS / "tachyon-role-study-20261023-v2/study-status.json").get("status"),
            "included_in_817_stable_rows": False,
            "interpretation": "Interrupted diagnostic mode superseded by the completed production-schema worker study.",
            "paths": [
                "results/tachyon-role-study-20261023-v2/study-status.json",
                "results/tachyon-role-study-20261023-v2/source-pins.json",
                "results/tachyon-role-study-20261023-v2/production-priority-interrupt.json",
            ],
        },
        {
            "evidence": "LFM2.6 fast parser probes",
            "status": "diagnostic_complete",
            "included_in_817_stable_rows": False,
            "interpretation": "Preparatory free, JSON-object, closed-schema, and zero-budget parser probes; not ranked or normalized.",
            "paths": [
                "results/tachyon-study-20261023-v1/fast-parser-probes/",
            ],
        },
        {
            "evidence": "QNN readiness and compatibility",
            "status": "readiness_only_not_llm_inference",
            "included_in_817_stable_rows": False,
            "interpretation": "Platform/readiness evidence only; the calculator does not establish LLM inference.",
            "paths": [
                "results/tachyon-study-20261023-v1/qnn-readiness.json",
                "results/tachyon-study-20261023-v1/qnn-llm-compatibility.json",
            ],
        },
        {
            "evidence": "Tachyon model download and artifact receipts",
            "status": "provenance_only",
            "included_in_817_stable_rows": False,
            "interpretation": "Model size/hash/download provenance; not request-quality observations.",
            "paths": [
                "results/tachyon-study-20261023-v1/remote-evidence/",
                "results/tachyon-role-study-20261023-v1/remote-evidence/",
                "results/tachyon-role-study-20261023-v1/small-download-launch.json",
                "results/tachyon-role-study-20261023-v1/small-download-plan.json",
            ],
        },
        {
            "evidence": "Tachyon runtime build receipts",
            "status": "provenance_only",
            "included_in_817_stable_rows": False,
            "interpretation": "CPU/OpenCL runtime construction evidence; failed accelerator trials remain in the separate accelerator table.",
            "paths": [
                "results/tachyon-study-20261023-v1/opencl-runtime-build.json",
                "results/tachyon-study-20261023-v1/opencl-generic-runtime-build.json",
                "results/tachyon-worker-latency-20261023-v1/remote-artifact-verification.json",
            ],
        },
        {
            "evidence": "Gemma follow-up",
            "status": "user_deferred",
            "included_in_817_stable_rows": False,
            "interpretation": "No result is included because this follow-up was deferred by the user.",
            "paths": [],
        },
    ]


def runtime_status(counts: Mapping[str, int]) -> dict[str, Any]:
    paired = read_json(PAIRED_COMPARISON)
    caveats = read_json(GRADER_CAVEATS)
    if len(paired.get("per_case", [])) != 64:
        raise ReportError("matched T4 comparison must contain all 64 frozen cases")
    return {
        "generated_at": now(),
        "status": "complete",
        "source_status": {
            "legacy_dashboard": "complete_snapshot",
            "t4_public": read_json(PUBLIC_T4 / "continuation-status.json").get("status"),
            "tachyon_public": read_json(PUBLIC_BOARD / "study-status.json").get("status"),
            "tachyon_worker": read_json(WORKER / "study-status.json").get("status"),
            "tachyon_latency": read_json(LATENCY / "trial-status.json").get("status"),
        },
        "detail_counts": dict(counts),
        "matched_t4_comparison": {
            "source_path": relative(PAIRED_COMPARISON),
            "scope": paired.get("scope"),
            "suites": paired.get("suites"),
        },
        "official_grader_caveats": {
            "source_path": relative(GRADER_CAVEATS),
            "official_grades_unchanged": caveats.get("official_grades_unchanged"),
            "scope": caveats.get("scope"),
            "notes": caveats.get("notes"),
        },
        "limitations": [
            "No quality score or rank is pooled across study families, suites, profiles, contexts, samplers, or token budgets.",
            "The legacy 480-row dashboard retains its original quality semantics and full columns in the included byte-identical CSV.",
            "T4 public tests use 73,728 context / 65,536 output caps; Tachyon public uses 16,384 / 8,192; worker and latency use 16,384 / 512.",
            "Public benchmark rows use EvalPlus enhanced per-task pass or IFEval strict prompt pass; those definitions are not interchangeable.",
            "Worker summary automatic pass covers closed facts, keys, and word limit only; free-prose grounding is in the separate 40-record manual review.",
            "LFM2.6 Tachyon public screening is native thinking; worker/latency profiles use fast zero-reasoning-budget controls.",
            "Latency rows repeat routing cases and cache states for diagnostics; they are not additional independent quality cases.",
            "Weighted native TPS uses server-reported predicted_per_second weighted by predicted_ms; missing timings remain null.",
            "Failed OpenCL compiler/offload rows have blank throughput, never zero. The QNN calculator is not LLM inference.",
            "Matched T4 latency is task-dependent: the paired-case artifact must be used instead of inferring universal speed from total wall time.",
            "The official LFM IFEval score remains 31/32; its sole miss is documented as a checker-specific Markdown emphasis criterion, with no semantic regrade substituted.",
        ],
    }


def html_table(rows: Sequence[Mapping[str, Any]], fields: Sequence[str], table_id: str) -> str:
    head = "".join(f"<th>{html.escape(field)}</th>" for field in fields)
    body = []
    for row in rows:
        cells = []
        for field in fields:
            value = row.get(field)
            if isinstance(value, float):
                rendered = f"{value:.6g}"
            elif value is None:
                rendered = "—"
            else:
                rendered = str(value)
            cells.append(f"<td>{html.escape(rendered)}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return f'<table id="{table_id}"><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table>'


def local_href(target: Path, output_dir: Path) -> str:
    """Build a local link whose target remains inside the public checkout."""
    try:
        resolved = target.resolve(strict=False)
        resolved.relative_to(ROOT.resolve())
    except (OSError, ValueError) as exc:
        raise ReportError(f"HTML link target is outside the public root: {target}") from exc
    return Path(os.path.relpath(resolved, start=output_dir.resolve())).as_posix()


def build_html(groups: Sequence[Mapping[str, Any]], accelerator: Sequence[Mapping[str, Any]],
               status: Mapping[str, Any], manual: Sequence[Mapping[str, Any]],
               exclusions: Sequence[Mapping[str, Any]], output_dir: Path) -> str:
    statuses = status["source_status"]
    status_cards = "".join(
        f'<div class="card"><b>{html.escape(name)}</b><br>{html.escape(str(value))}</div>'
        for name, value in statuses.items()
    )
    limitations = "".join(f"<li>{html.escape(item)}</li>" for item in status["limitations"])
    reviews = Counter(str(row["prose_grounding_review"]) for row in manual)
    review_text = ", ".join(f"{key}: {value}" for key, value in sorted(reviews.items()))
    paired = status["matched_t4_comparison"]
    paired_rows = paired["suites"]
    paired_fields = [
        "suite", "case_count", "both_pass", "lfm_only_pass", "oss_only_pass",
        "both_fail", "lfm_faster_case_count",
        "paired_latency_ratio_lfm_over_oss_median", "oss_median_wall_seconds",
        "lfm_median_wall_seconds", "oss_cap_hits", "lfm_cap_hits",
    ]
    caveats = status["official_grader_caveats"]
    caveat_items = "".join(
        "<li>" + html.escape(
            f"{item.get('model')} {item.get('suite')} {item.get('task_id')}: "
            f"{item.get('interpretation')}"
        ) + "</li>"
        for item in caveats["notes"]
    )
    exclusion_links = "".join(
        "<li><b>" + html.escape(str(item["evidence"])) + ":</b> "
        + (", ".join(
            '<a href="' + html.escape(local_href(ROOT / str(path), output_dir), quote=True)
            + '">' + html.escape(str(path)) + "</a>"
            for path in item["paths"]
        ) or "no artifact (deferred)") + "</li>"
        for item in exclusions
    )
    paired_href = html.escape(local_href(PAIRED_COMPARISON, output_dir), quote=True)
    caveats_href = html.escape(local_href(GRADER_CAVEATS, output_dir), quote=True)
    qnn_assessment_href = html.escape(local_href(QNN_ASSESSMENT, output_dir), quote=True)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Model evidence report — 2026-10-23</title>
<style>
body{{font:14px system-ui,sans-serif;margin:24px;color:#17202a;background:#f7f8fa}} h1,h2{{margin:.6em 0}}
.warning{{background:#fff4ce;border-left:5px solid #d79b00;padding:12px;max-width:1100px}} .cards{{display:flex;flex-wrap:wrap;gap:8px;margin:12px 0}}
.card{{background:white;border:1px solid #ccd3da;border-radius:6px;padding:9px 12px}} input{{width:min(720px,95%);padding:8px;margin:8px 0}}
.tablewrap{{overflow:auto;max-height:62vh;border:1px solid #ccd3da;background:white}} table{{border-collapse:collapse;width:100%;font-size:12px}}
th{{position:sticky;top:0;background:#243447;color:white}} th,td{{padding:6px;border-bottom:1px solid #e1e5e8;text-align:left;white-space:nowrap}}
tr:nth-child(even){{background:#f2f5f7}} code{{background:#e9edf0;padding:2px 4px}} a{{color:#075ea8}}
</style></head><body>
<h1>Model evidence report — 2026-10-23</h1>
<div class="warning"><b>Do not read this as one leaderboard.</b> Quality definitions, hardware, context/output caps, samplers, and thinking modes differ. Group rows are intentionally separated by study family + profile + suite.</div>
<div class="cards">{status_cards}</div>
<p>Detail: <a href="evidence.csv">evidence.csv</a> · grouped: <a href="group-summary.csv">group-summary.csv</a> · accelerator: <a href="accelerator-evidence.csv">accelerator-evidence.csv</a> · manual review: <a href="manual-summary-review.csv">manual-summary-review.csv</a> · exclusions: <a href="evidence-exclusions-index.json">evidence-exclusions-index.json</a> · full legacy columns: <a href="legacy-complete-study-dashboard.csv">legacy-complete-study-dashboard.csv</a> · provenance: <a href="provenance.json">provenance.json</a></p>
<h2>Runtime status and limitations</h2><ul>{limitations}</ul>
<p>Manual summary reviews (40, separate from automatic pass): {html.escape(review_text)}</p>
<h2>Evidence and exclusions index</h2>
<p>The 817 rows are the consolidated stable comparison. Preparatory and interrupted evidence remains linked here, but is not normalized or ranked with it.</p>
<div class="tablewrap">{html_table(exclusions, EXCLUSION_FIELDS, 'exclusions')}</div>
<ul>{exclusion_links}</ul>
<h2>Matched T4 public comparison</h2>
<p>This is a same-case comparison only; it is not pooled with the other families. Source: <a href="{paired_href}">final-paired-comparison.json</a>.</p>
<div class="tablewrap">{html_table(paired_rows, paired_fields, 'paired')}</div>
<h2>Official-grader caveat</h2>
<p>Official grades are unchanged. Source: <a href="{caveats_href}">official-grader-caveats.json</a>.</p><ul>{caveat_items}</ul>
<h2>QNN evidence chronology</h2>
<p>The QNN readiness statements in this report describe the historical 2026-10-23 snapshot. The later 2026-10-24 investigation supersedes those conclusions; see <a href="{qnn_assessment_href}">the later QNN assessment</a>. Numeric benchmark CSVs are unchanged.</p>
<h2>Grouped evidence</h2><input id="filter" placeholder="Filter model, profile, family, suite, hardware…" oninput="filterRows()">
<div class="tablewrap">{html_table(groups, GROUP_FIELDS, 'groups')}</div>
<h2>Accelerator evidence (not request-quality rows)</h2><div class="tablewrap">{html_table(accelerator, ACCELERATOR_FIELDS, 'accelerator')}</div>
<script>function filterRows(){{const q=document.getElementById('filter').value.toLowerCase();for(const r of document.querySelectorAll('#groups tbody tr'))r.style.display=r.innerText.toLowerCase().includes(q)?'':'none';}}</script>
</body></html>"""


def _generate(output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    rows = legacy_rows()
    family_counts = {"legacy_complete_study_dashboard": len(rows)}
    public_specs = [
        (PUBLIC_T4, "oss20b", "public_benchmark_t4_64case", "NVIDIA T4",
         73728, 65536, "seeded 32 HumanEval+ + 32 IFEval public subset; native thinking medium"),
        (PUBLIC_T4, "lfm25", "public_benchmark_t4_64case", "NVIDIA T4",
         73728, 65536, "seeded 32 HumanEval+ + 32 IFEval public subset; native thinking medium"),
        (PUBLIC_BOARD, "bonsai2-2b", "public_screen_tachyon_16case",
         "Tachyon QCM6490 CPU4 (taskset 4-7)", 16384, 8192,
         "seeded 8 HumanEval+ + 8 IFEval public screening subset"),
        (PUBLIC_BOARD, "bonsai2-4b", "public_screen_tachyon_16case",
         "Tachyon QCM6490 CPU4 (taskset 4-7)", 16384, 8192,
         "seeded 8 HumanEval+ + 8 IFEval public screening subset"),
        (PUBLIC_BOARD, "lfm25-26b", "public_screen_tachyon_16case",
         "Tachyon QCM6490 CPU4 (taskset 4-7)", 16384, 8192,
         "seeded 8 HumanEval+ + 8 IFEval public screening subset; native thinking"),
    ]
    for spec in public_specs:
        added = public_rows(spec[0], spec[1], study_family=spec[2], hardware=spec[3],
                            context_tokens=spec[4], output_tokens=spec[5], subset_note=spec[6])
        rows.extend(added); family_counts[spec[2]] = family_counts.get(spec[2], 0) + len(added)
    workers = worker_rows(); rows.extend(workers)
    family_counts["tachyon_worker_production_schema_24"] = len(workers)
    latency = latency_rows(); rows.extend(latency)
    for item in latency:
        family_counts[item["study_family"]] = family_counts.get(item["study_family"], 0) + 1
    if len(rows) != 817 or len({row["record_id"] for row in rows}) != len(rows):
        raise ReportError(f"detail row count/identity differs: {len(rows)}")
    groups = group_rows(rows)
    accelerator = accelerator_rows()
    manual, manual_source = manual_review_rows()
    exclusions = evidence_exclusions()
    counts = {**family_counts, "detail_total": len(rows),
              "manual_summary_reviews_separate": len(manual),
              "accelerator_rows_separate": len(accelerator)}
    status = runtime_status(counts)
    write_csv(output / "evidence.csv", EVIDENCE_FIELDS, rows)
    write_csv(output / "group-summary.csv", GROUP_FIELDS, groups)
    write_csv(output / "accelerator-evidence.csv", ACCELERATOR_FIELDS, accelerator)
    manual_fields = list(manual[0])
    write_csv(output / "manual-summary-review.csv", manual_fields, manual)
    (output / "evidence-exclusions-index.json").write_text(
        json.dumps(exclusions, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    shutil.copyfile(_published(LEGACY), output / "legacy-complete-study-dashboard.csv")
    source_paths = [
        LEGACY, PUBLIC_T4 / "manifest.json", PUBLIC_T4 / "continuation-status.json",
        PUBLIC_T4 / "oss20b/grading/summary.json", PUBLIC_T4 / "lfm25/grading/summary.json",
        PUBLIC_BOARD / "manifest.json", PUBLIC_BOARD / "study-status.json",
        *(PUBLIC_BOARD / key / "grading/summary.json" for key in
          ("bonsai2-2b", "bonsai2-4b", "lfm25-26b")),
        WORKER / "manifest.json", WORKER / "study-status.json", WORKER / "summary.json",
        WORKER / "analysis/role-comparison.csv", manual_source,
        LATENCY / "trial-status.json", LATENCY / "summary.json",
        LATENCY / "greedy-diagnostic/summary.json",
        PUBLIC_BOARD / "gpu-trial/summary.json",
        LATENCY / "opencl-generic-ngl4-result.json",
        PUBLIC_BOARD / "qnn-llm-compatibility.json",
        PAIRED_COMPARISON, GRADER_CAVEATS, QNN_ASSESSMENT,
        RESULTS / "tachyon-role-study-20261023-v1/study-status.json",
        RESULTS / "tachyon-role-study-20261023-v1/source-pins.json",
        RESULTS / "tachyon-role-study-20261023-v1/lfm-fast-profile-interrupt.json",
        RESULTS / "tachyon-role-study-20261023-v2/study-status.json",
        RESULTS / "tachyon-role-study-20261023-v2/source-pins.json",
        RESULTS / "tachyon-role-study-20261023-v2/production-priority-interrupt.json",
        PUBLIC_BOARD / "qnn-readiness.json",
        PUBLIC_BOARD / "opencl-runtime-build.json",
        PUBLIC_BOARD / "opencl-generic-runtime-build.json",
        LATENCY / "remote-artifact-verification.json",
    ]
    provenance = {
        "schema_version": 1, "generated_at": now(),
        "generator": relative(Path(__file__)), "generator_sha256": sha256(Path(__file__)),
        "sources": [
            {"path": relative(path), "sha256": sha256(_published(path))}
            for path in source_paths
        ],
        "outputs": {"detail_rows": len(rows), "group_rows": len(groups),
                    "manual_review_rows": len(manual), "accelerator_rows": len(accelerator)},
        "legacy_copy_sha256": sha256(output / "legacy-complete-study-dashboard.csv"),
    }
    (output / "runtime-status.json").write_text(
        json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "report.html").write_text(
        build_html(groups, accelerator, status, manual, exclusions, output), encoding="utf-8"
    )
    return {"rows": rows, "groups": groups, "accelerator": accelerator,
            "manual": manual, "exclusions": exclusions, "status": status,
            "provenance": provenance}


def generate(output: Path | None = None, *, evidence_root: Path | None = None,
             evidence_manifest: Path | None = None) -> dict[str, Any]:
    """Build the report, optionally from a verified relocated public archive."""
    evidence = None
    if evidence_root is not None:
        try:
            evidence = public_evidence.verify_manifest(
                evidence_root, manifest_path=evidence_manifest,
                required_paths=(
                    "results/complete-study-dashboard.csv",
                    "results/public-benchmark-pilot-20261023-v1/manifest.json",
                    "results/tachyon-study-20261023-v1/manifest.json",
                    "results/tachyon-worker-study-20261023-v1/manifest.json",
                ),
            )
        except (OSError, public_evidence.EvidenceError) as exc:
            raise ReportError(str(exc)) from exc
    with _public_root(evidence):
        chosen_output = Path(output).resolve() if output is not None else OUT
        result = _generate(chosen_output)
        if evidence is not None:
            result["provenance"]["evidence_manifest"] = {
                "path": relative(evidence.manifest_path),
                "sha256": evidence.manifest_sha256,
            }
            (chosen_output / "provenance.json").write_text(
                json.dumps(result["provenance"], indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", type=Path)
    parser.add_argument("--evidence-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if (args.evidence_root is None) != (args.output_dir is None):
        parser.error("--evidence-root and --output-dir must be supplied together for a public rebuild")
    result = generate(args.output_dir, evidence_root=args.evidence_root,
                      evidence_manifest=args.evidence_manifest)
    output = args.output_dir.resolve() if args.output_dir is not None else OUT
    print(json.dumps({"output": str(output), "detail_rows": len(result["rows"]),
                      "group_rows": len(result["groups"]),
                      "manual_rows": len(result["manual"]),
                      "accelerator_rows": len(result["accelerator"])}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
