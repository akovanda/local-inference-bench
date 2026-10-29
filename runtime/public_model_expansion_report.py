#!/usr/bin/env python3
"""Render the frozen-64 public model expansion as a durable partial/final report.

This module is deliberately read-only with respect to benchmark inputs.  It does
not start servers, run inference, or invoke graders.  Every invocation rebuilds
the report from the supervisor/catalog plus the runner's immutable attempts.
"""

from __future__ import annotations

import argparse
import contextlib
from contextlib import contextmanager
import csv
import datetime as dt
import hashlib
import html
import io
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence
import uuid

try:
    from runtime import public_evidence
except ModuleNotFoundError:  # Direct execution from runtime/.
    import public_evidence  # type: ignore[no-redef]


ROOT = Path(__file__).resolve().parents[1]
FROZEN_MANIFEST = ROOT / "results/public-benchmark-pilot-20261023-v1/manifest.json"
FROZEN_MANIFEST_SHA256 = "758607e95961b95ca98bdb2bdf11342b961239973dfcff73ff424c3a69603039"
DEFAULT_STUDY = ROOT / "results/public-model-expansion-20261024-v1"
DEFAULT_CATALOG = ROOT / "runtime/public-model-expansion-20261024-v1/profile-catalog.json"
ORIGINAL_STUDY = ROOT / "results/public-benchmark-pilot-20261023-v1"
_ACTIVE_EVIDENCE: public_evidence.VerifiedEvidence | None = None
SUITE_LABELS = {
    "humanevalplus": "HumanEval+ enhanced (32)",
    "ifeval": "IFEval strict prompt (32)",
}
TERMINAL_ATTEMPT_STATUSES = {"completed", "truncated"}
ERROR_STATUSES = {"api_failure", "cancelled", "incomplete", "invalid"}
FINAL_SUPERVISOR_STATES = {
    "complete", "completed", "completed_with_errors", "failed", "cancelled", "stopped"
}

DETAIL_FIELDS = [
    "profile_key", "display_name", "model_name", "profile_kind", "profile_state",
    "runtime_backend", "comparison_mode", "suite", "suite_label", "case_index", "case_id", "storage_id",
    "attempt_number", "attempt_status", "latest_case_status", "selected_for_benchmark",
    "selection_policy", "finish_reason", "capped", "grade_known", "primary_pass",
    "humaneval_base_pass", "humaneval_enhanced_pass", "ifeval_strict_prompt_pass",
    "ifeval_loose_prompt_pass", "ifeval_strict_instruction_pass_count",
    "ifeval_strict_instruction_count", "ifeval_loose_instruction_pass_count",
    "ifeval_loose_instruction_count", "prompt_tokens", "completion_tokens", "total_tokens",
    "reasoning_tokens", "cached_prompt_tokens", "server_generation_tps",
    "server_predicted_tokens", "server_predicted_ms", "server_prompt_tps",
    "server_prompt_tokens", "server_prompt_ms", "wall_seconds", "ttft_seconds",
    "ttft_reasoning_seconds", "ttft_final_seconds", "context_tokens", "output_cap_tokens",
    "request_timeout_seconds", "seed", "request_options_json", "placement_json",
    "profile_tag", "profile_sha256", "catalog_sha256", "model_artifact_sha256",
    "model_artifact_path", "config_path", "config_sha256", "runner_sha256",
    "profile_parameters_json", "source_path", "grade_source_path",
    "run_path", "preflight_path", "profile_binding_path",
]

SUMMARY_FIELDS = [
    "profile_key", "display_name", "model_name", "profile_kind", "profile_state",
    "runtime_backend", "comparison_mode", "suite", "suite_label", "expected_cases", "terminal_cases",
    "completed_cases", "truncated_cases", "pending_cases", "error_cases", "cap_cases",
    "graded_cases", "graded_unknown_cases", "quality_passes", "quality_denominator",
    "quality_rate", "quality_metric", "humaneval_base_passes",
    "humaneval_base_graded", "humaneval_enhanced_passes", "humaneval_enhanced_graded",
    "ifeval_strict_prompt_passes", "ifeval_strict_prompt_graded",
    "ifeval_loose_prompt_passes", "ifeval_loose_prompt_graded",
    "ifeval_strict_instruction_passes", "ifeval_strict_instruction_total",
    "ifeval_loose_instruction_passes", "ifeval_loose_instruction_total",
    "selected_prompt_tokens", "selected_completion_tokens", "selected_reasoning_tokens",
    "all_request_prompt_tokens", "all_request_completion_tokens",
    "all_request_reasoning_tokens", "weighted_server_generation_tps",
    "weighted_server_prompt_tps", "all_request_wall_seconds", "median_wall_seconds",
    "p95_wall_seconds", "median_ttft_seconds", "p95_ttft_seconds",
    "median_final_ttft_seconds", "p95_final_ttft_seconds", "cold_load_seconds",
    "cold_load_source", "attempt_count", "retry_count", "context_tokens",
    "output_cap_tokens", "request_timeout_seconds", "seed", "request_options_json",
    "placement_json", "profile_tag", "profile_sha256", "catalog_sha256",
    "model_artifact_sha256", "model_artifact_path", "config_path", "config_sha256",
    "runner_sha256", "profile_parameters_json",
    "output_dir", "grading_summary_path",
]

DIAGNOSTIC_FIELDS = [
    "name", "status", "scored", "caps_json", "path", "note",
]


class ReportError(RuntimeError):
    """A source violates the frozen comparison/reporting contract."""


def _published(path: Path) -> Path:
    if _ACTIVE_EVIDENCE is None:
        return path
    try:
        return _ACTIVE_EVIDENCE.ensure_published_file(path)
    except public_evidence.EvidenceError as exc:
        raise ReportError(str(exc)) from exc


@contextmanager
def _public_root(evidence: public_evidence.VerifiedEvidence | None):
    global ROOT, FROZEN_MANIFEST, DEFAULT_STUDY, DEFAULT_CATALOG, ORIGINAL_STUDY
    global _ACTIVE_EVIDENCE
    if evidence is None:
        yield
        return
    previous = (
        ROOT, FROZEN_MANIFEST, DEFAULT_STUDY, DEFAULT_CATALOG, ORIGINAL_STUDY,
        _ACTIVE_EVIDENCE,
    )
    ROOT = evidence.root
    FROZEN_MANIFEST = ROOT / "results/public-benchmark-pilot-20261023-v1/manifest.json"
    DEFAULT_STUDY = ROOT / "results/public-model-expansion-20261024-v1"
    DEFAULT_CATALOG = ROOT / "runtime/public-model-expansion-20261024-v1/profile-catalog.json"
    ORIGINAL_STUDY = ROOT / "results/public-benchmark-pilot-20261023-v1"
    _ACTIVE_EVIDENCE = evidence
    try:
        yield
    finally:
        (
            ROOT, FROZEN_MANIFEST, DEFAULT_STUDY, DEFAULT_CATALOG, ORIGINAL_STUDY,
            _ACTIVE_EVIDENCE,
        ) = previous


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    path = _published(path)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ReportError(f"invalid JSON: {path}") from exc


def read_json_if_present(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        return {}
    value = read_json(path)
    if not isinstance(value, dict):
        raise ReportError(f"expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[Mapping[str, Any]]:
    path = _published(path)
    rows: list[Mapping[str, Any]] = []
    try:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ReportError(f"non-object JSONL row: {path}:{line_number}")
            rows.append(value)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ReportError(f"invalid JSONL: {path}") from exc
    return rows


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    try:
        with temporary.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()


def atomic_json(path: Path, value: Any) -> None:
    atomic_write(path, (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode())


def finite_number(value: Any) -> int | float | None:
    if value is None or value == "" or isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return int(parsed) if parsed.is_integer() else parsed


def integer(value: Any) -> int | None:
    parsed = finite_number(value)
    return int(parsed) if parsed is not None and float(parsed).is_integer() else None


def json_cell(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def rel(path: Path | str | None) -> str | None:
    if path is None:
        return None
    candidate = Path(path)
    try:
        return str(candidate.resolve().relative_to(ROOT))
    except (OSError, ValueError):
        return str(candidate)


def resolve_path(value: Any, *bases: Path) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    if _ACTIVE_EVIDENCE is not None:
        try:
            return _ACTIVE_EVIDENCE.resolve(value, bases=bases)
        except public_evidence.EvidenceError as exc:
            raise ReportError(str(exc)) from exc
    candidate = Path(value).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    if candidate.parts and candidate.parts[0] in {"results", "runtime", "vendor", "tests"}:
        return (ROOT / candidate).resolve()
    for base in bases:
        joined = (base / candidate).resolve()
        if joined.exists():
            return joined
    return (bases[0] / candidate).resolve() if bases else candidate.resolve()


def storage_id(task_id: str) -> str:
    return hashlib.sha256(task_id.encode()).hexdigest()[:16]


def load_manifest(path: Path, expected_sha256: str | None = None) -> tuple[Mapping[str, Any], list[dict[str, Any]], str]:
    actual_sha = sha256_file(path)
    if expected_sha256 is not None and actual_sha != expected_sha256:
        raise ReportError(f"manifest SHA differs: expected {expected_sha256}, found {actual_sha}")
    value = read_json(path)
    if not isinstance(value, dict) or not isinstance(value.get("cases"), list):
        raise ReportError("manifest must be an object with cases")
    seen: set[tuple[str, str]] = set()
    cases: list[dict[str, Any]] = []
    for index, source in enumerate(value["cases"]):
        if not isinstance(source, dict):
            raise ReportError(f"manifest case {index} is not an object")
        suite, task_id, prompt = source.get("suite"), source.get("task_id"), source.get("prompt")
        if suite not in SUITE_LABELS or not isinstance(task_id, str) or not isinstance(prompt, str):
            raise ReportError(f"invalid manifest case {index}")
        identity = (suite, task_id)
        if identity in seen:
            raise ReportError(f"duplicate manifest case {suite}:{task_id}")
        seen.add(identity)
        cases.append({"index": index, "suite": suite, "task_id": task_id,
                      "prompt": prompt, "storage_id": storage_id(task_id)})
    if expected_sha256 == FROZEN_MANIFEST_SHA256:
        counts = {suite: sum(case["suite"] == suite for case in cases) for suite in SUITE_LABELS}
        if counts != {"humanevalplus": 32, "ifeval": 32}:
            raise ReportError(f"frozen manifest must contain 32 cases per suite, found {counts}")
    return value, cases, actual_sha


def _catalog_profiles(catalog: Mapping[str, Any]) -> list[tuple[str, Mapping[str, Any]]]:
    source = catalog.get("profiles")
    if isinstance(source, dict):
        return [(str(key), value) for key, value in source.items() if isinstance(value, dict)]
    if isinstance(source, list):
        rows = []
        for item in source:
            if not isinstance(item, dict):
                continue
            key = item.get("key") or item.get("profile_key")
            if not isinstance(key, str) or not key:
                raise ReportError("catalog profile lacks key")
            rows.append((key, item))
        return rows
    raise ReportError("profile catalog must have a profiles object or list")


def _status_models(status: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    source = status.get("models")
    if not isinstance(source, dict):
        return {}
    return {str(key): value for key, value in source.items() if isinstance(value, dict)}


def profile_specs(study_dir: Path, catalog: Mapping[str, Any], supervisor: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    models = _status_models(supervisor)
    specs: list[dict[str, Any]] = []
    catalog_keys: set[str] = set()
    for key, entry in _catalog_profiles(catalog):
        catalog_keys.add(key)
        state = models.get(key, {})
        output = resolve_path(state.get("output_dir"), study_dir) or study_dir / "profiles" / key
        specs.append({"key": key, "catalog": entry, "status": state, "output_dir": output,
                      "kind": "expansion", "quality_eligible": True})
    diagnostics = []
    for key, state in models.items():
        if key in catalog_keys:
            continue
        output = resolve_path(state.get("output_dir"), study_dir)
        diagnostics.append({
            "name": state.get("display_name") or key,
            "status": state.get("state") or "unknown",
            "scored": False,
            "caps_json": json_cell(state.get("caps") or {}),
            "path": rel(output) if output else None,
            "note": "Supervisor entry is outside the frozen profile catalog; excluded from benchmark denominators.",
        })
    controls = [
        ("control-oss20b-original", "Original OSS 20B medium", "oss20b"),
        ("control-lfm24-original", "Original LFM2.5 24B medium", "lfm25"),
    ]
    for key, display, directory in controls:
        specs.append({
            "key": key,
            "catalog": {
                "display_name": display,
                "runtime_backend": "llama.cpp control",
                "parameters": {"context_tokens": 73728, "max_output_tokens": 65536,
                               "request_timeout_seconds": 21600, "reasoning_effort": "medium",
                               "temperature": 0.0, "seed": 20261023},
                "profile_tag": "public-pilot-20261023-v1/original-medium",
            },
            "status": {"state": "complete", "output_dir": str(ORIGINAL_STUDY / directory)},
            "output_dir": ORIGINAL_STUDY / directory,
            "kind": "original_control",
            "quality_eligible": True,
        })
    return specs, diagnostics


def _nested_mapping(value: Mapping[str, Any], *keys: str) -> Mapping[str, Any]:
    for key in keys:
        candidate = value.get(key)
        if isinstance(candidate, dict):
            return candidate
    return {}


def _first(*values: Any) -> Any:
    return next((value for value in values if value is not None and value != ""), None)


def _recursive_find_number(value: Any, keys: set[str]) -> tuple[int | float | None, str | None]:
    if isinstance(value, dict):
        for key, item in value.items():
            if key in keys and finite_number(item) is not None:
                return finite_number(item), key
        for item in value.values():
            found, key = _recursive_find_number(item, keys)
            if found is not None:
                return found, key
    elif isinstance(value, list):
        for item in value:
            found, key = _recursive_find_number(item, keys)
            if found is not None:
                return found, key
    return None, None


def profile_metadata(spec: Mapping[str, Any], catalog_sha: str, study_dir: Path) -> tuple[dict[str, Any], set[Path]]:
    output = Path(spec["output_dir"])
    run_path = output / "run.json"
    status_path = output / "status.json"
    binding_path = output / "profile-binding.json"
    preflight_path = output / "preflight.json"
    run = read_json_if_present(run_path)
    run_status = read_json_if_present(status_path)
    binding = read_json_if_present(binding_path)
    preflight = read_json_if_present(preflight_path)
    status = spec["status"]
    entry = spec["catalog"]
    handoff_path = resolve_path(entry.get("external_handoff"), ROOT, study_dir)
    handoff = read_json_if_present(handoff_path) if handoff_path and handoff_path.is_file() else {}
    settings = _nested_mapping(run, "settings")
    parameters = dict(_nested_mapping(entry, "parameters", "params"))
    controls = _nested_mapping(entry, "controls")
    for key, value in controls.items():
        parameters.setdefault(key, value)
    for key in ("context_tokens", "max_output_tokens", "request_timeout_seconds", "seed",
                "request_options", "placement", "config_candidates"):
        if key in entry:
            parameters.setdefault(key, entry[key])
    selected_external = handoff.get("profile_selected")
    if isinstance(selected_external, dict):
        parameters["request_options"] = {
            key: value for key, value in selected_external.items()
            if key in {"reasoning_effort", "temperature", "top_p", "top_k", "min_p",
                       "presence_penalty", "repeat_penalty", "chat_template_thinking",
                       "reasoning_budget_tokens", "thinking_budget_tokens"}
        }
    request_options = _nested_mapping(parameters, "request_options")
    if not request_options:
        request_options = {key: _first(parameters.get(key), settings.get(key)) for key in (
            "reasoning_effort", "temperature", "top_p", "top_k", "min_p",
            "presence_penalty", "repeat_penalty", "chat_template_thinking",
            "reasoning_budget_tokens", "thinking_budget_tokens")}
        request_options = {key: value for key, value in request_options.items() if value is not None}
    placement = _first(entry.get("placement"), parameters.get("placement"),
                       preflight.get("placement"), status.get("placement"), {})
    if not isinstance(placement, (dict, list)):
        placement = {"value": placement}
    elif isinstance(placement, dict):
        placement = dict(placement)
    else:
        placement = {"resolved": placement}
    if entry.get("config_candidates") is not None:
        placement.setdefault("config_candidates", entry["config_candidates"])
    if entry.get("runtime_environment") is not None:
        placement.setdefault("runtime_environment", entry["runtime_environment"])
    model_provenance = _first(binding.get("model"), binding.get("model_provenance"),
                              preflight.get("model"), handoff.get("actual_model"),
                              entry.get("model"), {})
    if not isinstance(model_provenance, dict):
        model_provenance = {"value": model_provenance}
    cold_load, cold_key = _recursive_find_number(
        preflight, {"cold_load_seconds", "model_load_seconds", "load_seconds"}
    )
    grade_value = _first(status.get("grading_summary"), handoff.get("grading_summary"))
    grade_path = resolve_path(grade_value, study_dir, output) or output / "grading/summary.json"
    profile_hash = _first(binding.get("profile_sha256"), binding.get("sha256"),
                          status.get("profile_sha256"), entry.get("profile_sha256"),
                          run.get("settings_sha256"))
    profile_tag = _first(binding.get("profile_tag"), binding.get("tag"),
                         entry.get("profile_tag"), entry.get("tag"),
                         settings.get("original_profile"), spec["key"])
    artifact_sha = _first(model_provenance.get("sha256"), binding.get("model_sha256"),
                          preflight.get("model_sha256"), entry.get("model_sha256"))
    artifact_path = _first(model_provenance.get("path"), binding.get("model_path"),
                           preflight.get("model_path"), entry.get("model_path"))
    config = binding.get("config") if isinstance(binding.get("config"), dict) else {}
    runner = binding.get("runner") if isinstance(binding.get("runner"), dict) else {}
    config_path = _first(config.get("path"), preflight.get("selected_config"),
                         handoff.get("config_path"))
    config_sha = _first(config.get("sha256"), preflight.get("selected_config_sha256"),
                        handoff.get("config_sha256"))
    key_lower = str(spec["key"]).lower()
    if spec["kind"] == "original_control":
        comparison_mode = "original medium/T0 control"
    elif "oss" in key_lower and ("repair" in key_lower or "rerun" in key_lower):
        comparison_mode = "repaired OSS diagnostic-selected low/full-sampling (different decoding)"
    else:
        comparison_mode = "matched T0/medium baseline"
    meta = {
        "profile_key": spec["key"],
        "display_name": _first(status.get("display_name"), entry.get("display_name"), spec["key"]),
        "model_name": _first(status.get("model_name"), entry.get("model_name"), settings.get("model"), spec["key"]),
        "profile_kind": spec["kind"],
        "profile_state": _first(status.get("state"), run_status.get("state"), run_status.get("status"), "pending"),
        "runtime_backend": _first(status.get("runtime_backend"), entry.get("runtime_backend"),
                                  preflight.get("runtime_backend"), "unknown"),
        "comparison_mode": _first(entry.get("comparison_label"), entry.get("comparison_mode"), comparison_mode),
        "context_tokens": integer(_first(parameters.get("context_tokens"), settings.get("context_tokens"))),
        "output_cap_tokens": integer(_first(parameters.get("max_output_tokens"), parameters.get("output_tokens"),
                                                    settings.get("max_tokens"))),
        "request_timeout_seconds": finite_number(_first(parameters.get("request_timeout_seconds"),
                                                          settings.get("request_timeout_seconds"))),
        "seed": integer(_first(parameters.get("seed"), settings.get("seed"))),
        "request_options_json": json_cell(request_options),
        "placement_json": json_cell(placement),
        "profile_tag": profile_tag,
        "profile_sha256": profile_hash,
        "catalog_sha256": catalog_sha,
        "model_artifact_sha256": artifact_sha,
        "model_artifact_path": artifact_path,
        "config_path": config_path,
        "config_sha256": config_sha,
        "runner_sha256": runner.get("sha256"),
        "profile_parameters_json": json_cell(parameters),
        "cold_load_seconds": cold_load,
        "cold_load_source": f"{rel(preflight_path)}#{cold_key}" if cold_load is not None else None,
        "output_dir": rel(output),
        "grading_summary_path": rel(grade_path),
        "run_path": rel(run_path) if run_path.is_file() else None,
        "preflight_path": rel(preflight_path) if preflight_path.is_file() else None,
        "profile_binding_path": rel(binding_path) if binding_path.is_file() else rel(handoff_path) if handoff_path else None,
        "settings_sha256": run.get("settings_sha256"),
        "manifest_sha256": _nested_mapping(run, "manifest").get("sha256"),
        "grade_path": grade_path,
    }
    sources = {path for path in (run_path, status_path, binding_path, preflight_path, handoff_path)
               if path is not None and path.is_file()}
    return meta, sources


def _grade_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value == "pass":
        return True
    if value == "fail":
        return False
    return None


def _official_path(value: Any, summary_path: Path, output: Path, study_dir: Path) -> Path | None:
    if _ACTIVE_EVIDENCE is not None and isinstance(value, str) and value:
        try:
            path = _ACTIVE_EVIDENCE.resolve(
                value, bases=(study_dir, output, summary_path.parent),
                required=not (value == "/external" or value.startswith("/external/")),
            )
        except public_evidence.EvidenceError as exc:
            raise ReportError(str(exc)) from exc
    else:
        path = resolve_path(value, study_dir, output, summary_path.parent)
    if path is not None and path.is_file():
        return path
    return None


def load_grades(summary_path: Path, cases: Sequence[Mapping[str, Any]], output: Path,
                study_dir: Path) -> tuple[dict[tuple[str, str], dict[str, Any]], set[Path]]:
    grades: dict[tuple[str, str], dict[str, Any]] = {}
    sources: set[Path] = set()
    if not summary_path.is_file():
        return grades, sources
    summary = read_json_if_present(summary_path)
    sources.add(summary_path)
    evalplus = _nested_mapping(summary, "evalplus")
    per_task = evalplus.get("per_task")
    if isinstance(per_task, list):
        allowed = {str(case["task_id"]) for case in cases if case["suite"] == "humanevalplus"}
        seen: set[str] = set()
        for item in per_task:
            if not isinstance(item, dict) or not isinstance(item.get("task_id"), str):
                continue
            task_id = item["task_id"]
            if task_id not in allowed:
                raise ReportError(f"EvalPlus grade task is outside frozen manifest: {task_id}")
            if task_id in seen:
                raise ReportError(f"duplicate EvalPlus grade task: {task_id}")
            seen.add(task_id)
            grades[("humanevalplus", task_id)] = {
                "humaneval_base_pass": _grade_bool(item.get("base_status")),
                "humaneval_enhanced_pass": _grade_bool(item.get("plus_status")),
                "grade_source_path": rel(summary_path),
            }
    ifeval = _nested_mapping(summary, "ifeval")
    allowed_prompts = {
        str(case["prompt"]): str(case["task_id"]) for case in cases if case["suite"] == "ifeval"
    }
    for mode, key in (("strict", "strict_results"), ("loose", "loose_results")):
        result_path = _official_path(ifeval.get(key), summary_path, output, study_dir)
        if result_path is None:
            continue
        sources.add(result_path)
        seen_prompts: set[str] = set()
        for item in read_jsonl(result_path):
            prompt = item.get("prompt")
            if not isinstance(prompt, str) or prompt not in allowed_prompts:
                raise ReportError(f"IFEval {mode} result prompt does not exactly match frozen manifest: {result_path}")
            if prompt in seen_prompts:
                raise ReportError(f"duplicate IFEval {mode} prompt: {result_path}")
            seen_prompts.add(prompt)
            task_id = allowed_prompts[prompt]
            instruction = item.get("follow_instruction_list")
            if not isinstance(instruction, list) or not all(isinstance(value, bool) for value in instruction):
                instruction = []
            target = grades.setdefault(("ifeval", task_id), {})
            target[f"ifeval_{mode}_prompt_pass"] = (
                item.get("follow_all_instructions") if isinstance(item.get("follow_all_instructions"), bool) else None
            )
            target[f"ifeval_{mode}_instruction_pass_count"] = sum(instruction) if instruction else None
            target[f"ifeval_{mode}_instruction_count"] = len(instruction) if instruction else None
            target["grade_source_path"] = rel(result_path)
    return grades, sources


def _attempt_number(path: Path) -> int:
    match = re.fullmatch(r"attempt-(\d+)", path.name)
    return int(match.group(1)) if match else 0


def _attempt_paths(output: Path, case: Mapping[str, Any]) -> list[Path]:
    case_dir = output / "cases" / str(case["suite"]) / str(case["storage_id"])
    if not case_dir.is_dir():
        return []
    return sorted((path for path in case_dir.iterdir() if path.is_dir() and re.fullmatch(r"attempt-\d+", path.name)),
                  key=_attempt_number)


def _native_duration_ms(native: Mapping[str, Any], prefix: str) -> int | float | None:
    direct = finite_number(native.get(f"{prefix}_ms"))
    if direct is not None and direct > 0:
        return direct
    tokens = finite_number(native.get(f"{prefix}_n"))
    rate = finite_number(native.get(f"{prefix}_per_second"))
    if tokens is not None and rate is not None and rate > 0:
        return float(tokens) / float(rate) * 1000
    return None


def detail_rows_for_profile(spec: Mapping[str, Any], meta: Mapping[str, Any],
                            cases: Sequence[Mapping[str, Any]], grades: Mapping[tuple[str, str], Mapping[str, Any]]) -> tuple[list[dict[str, Any]], set[Path]]:
    output = Path(spec["output_dir"])
    rows: list[dict[str, Any]] = []
    sources: set[Path] = set()
    for case in cases:
        attempts: list[tuple[Path, Mapping[str, Any] | None]] = []
        for attempt_path in _attempt_paths(output, case):
            result_path = attempt_path / "result.json"
            result = read_json_if_present(result_path) if result_path.is_file() else None
            if result is not None:
                sources.add(result_path)
                manifest_sha = result.get("manifest_sha256")
                if meta.get("manifest_sha256") and manifest_sha != meta["manifest_sha256"]:
                    raise ReportError(f"attempt manifest binding mismatch: {result_path}")
                settings_sha = result.get("settings_sha256")
                if meta.get("settings_sha256") and settings_sha != meta["settings_sha256"]:
                    raise ReportError(f"attempt settings binding mismatch: {result_path}")
            attempts.append((attempt_path, result))
        selected_index = next((index for index, (_, result) in enumerate(attempts)
                               if result is not None and result.get("status") in TERMINAL_ATTEMPT_STATUSES), None)
        latest_status = (
            str(attempts[-1][1].get("status")) if attempts and attempts[-1][1] is not None
            else "attempt_in_progress" if attempts else "not_started"
        )
        iterable = attempts or [(output, None)]
        for index, (attempt_path, result) in enumerate(iterable):
            selected = selected_index is not None and index == selected_index
            result = result or {}
            counts = result.get("counts") if isinstance(result.get("counts"), dict) else {}
            usage = result.get("usage") if isinstance(result.get("usage"), dict) else {}
            native = result.get("native_timings") if isinstance(result.get("native_timings"), dict) else {}
            timing = result.get("timing") if isinstance(result.get("timing"), dict) else {}
            grade = grades.get((str(case["suite"]), str(case["task_id"])), {}) if selected else {}
            primary_pass = (grade.get("humaneval_enhanced_pass") if case["suite"] == "humanevalplus"
                            else grade.get("ifeval_strict_prompt_pass"))
            completion_details = usage.get("completion_tokens_details") if isinstance(usage.get("completion_tokens_details"), dict) else {}
            row = {field: None for field in DETAIL_FIELDS}
            row.update({key: meta.get(key) for key in (
                "profile_key", "display_name", "model_name", "profile_kind", "profile_state",
                "runtime_backend", "comparison_mode", "context_tokens", "output_cap_tokens", "request_timeout_seconds",
                "seed", "request_options_json", "placement_json", "profile_tag", "profile_sha256",
                "catalog_sha256", "model_artifact_sha256", "model_artifact_path",
                "config_path", "config_sha256", "runner_sha256", "profile_parameters_json",
                "run_path", "preflight_path", "profile_binding_path")})
            attempt_number = _attempt_number(attempt_path) if attempts else None
            status = str(result.get("status")) if result else latest_status
            row.update({
                "suite": case["suite"], "suite_label": SUITE_LABELS[str(case["suite"])],
                "case_index": case["index"], "case_id": case["task_id"], "storage_id": case["storage_id"],
                "attempt_number": attempt_number, "attempt_status": status,
                "latest_case_status": latest_status, "selected_for_benchmark": selected,
                "selection_policy": "first_completed_or_truncated",
                "finish_reason": result.get("finish_reason"),
                "capped": selected and (result.get("truncated") is True or status == "truncated"),
                "grade_known": selected and primary_pass is not None, "primary_pass": primary_pass,
                "prompt_tokens": integer(_first(counts.get("prompt_tokens"), usage.get("prompt_tokens"))),
                "completion_tokens": integer(_first(counts.get("completion_tokens"), usage.get("completion_tokens"))),
                "total_tokens": integer(_first(counts.get("total_tokens"), usage.get("total_tokens"))),
                "reasoning_tokens": integer(_first(counts.get("reasoning_tokens"), completion_details.get("reasoning_tokens"))),
                "cached_prompt_tokens": integer(counts.get("cached_prompt_tokens")),
                "server_generation_tps": finite_number(native.get("predicted_per_second")),
                "server_predicted_tokens": finite_number(native.get("predicted_n")),
                "server_predicted_ms": _native_duration_ms(native, "predicted"),
                "server_prompt_tps": finite_number(native.get("prompt_per_second")),
                "server_prompt_tokens": finite_number(native.get("prompt_n")),
                "server_prompt_ms": _native_duration_ms(native, "prompt"),
                "wall_seconds": finite_number(timing.get("wall_seconds")),
                "ttft_seconds": finite_number(timing.get("ttft_seconds")),
                "ttft_reasoning_seconds": finite_number(timing.get("ttft_reasoning_seconds")),
                "ttft_final_seconds": finite_number(timing.get("ttft_final_seconds")),
                "source_path": rel(attempt_path / "result.json") if result else rel(attempt_path) if attempts else None,
                "grade_source_path": grade.get("grade_source_path"),
            })
            for key in DETAIL_FIELDS:
                if key in grade:
                    row[key] = grade[key]
            rows.append(row)
    return rows, sources


def median(values: Sequence[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    middle = len(ordered) // 2
    return ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2


def percentile(values: Sequence[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(probability * len(ordered)) - 1)]


def _sum_int(rows: Iterable[Mapping[str, Any]], field: str) -> int:
    return sum(int(row[field]) for row in rows if isinstance(row.get(field), int) and not isinstance(row.get(field), bool))


def _weighted(rows: Iterable[Mapping[str, Any]], rate_field: str, duration_field: str) -> float | None:
    pairs = [(float(row[rate_field]), float(row[duration_field])) for row in rows
             if isinstance(row.get(rate_field), (int, float)) and not isinstance(row.get(rate_field), bool)
             and isinstance(row.get(duration_field), (int, float)) and not isinstance(row.get(duration_field), bool)
             and float(row[duration_field]) > 0]
    return sum(rate * duration for rate, duration in pairs) / sum(duration for _, duration in pairs) if pairs else None


def summarize_profile(meta: Mapping[str, Any], rows: Sequence[Mapping[str, Any]],
                      cases: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for suite in SUITE_LABELS:
        suite_rows = [row for row in rows if row["suite"] == suite]
        selected = [row for row in suite_rows if row["selected_for_benchmark"] is True]
        attempts = [row for row in suite_rows if row["attempt_number"] is not None]
        by_case: dict[str, list[Mapping[str, Any]]] = {}
        for row in suite_rows:
            by_case.setdefault(str(row["case_id"]), []).append(row)
        expected = sum(case["suite"] == suite for case in cases)
        pending = errors = 0
        for scoped in by_case.values():
            if any(row["selected_for_benchmark"] is True for row in scoped):
                continue
            latest = str(scoped[-1]["latest_case_status"])
            if latest in ERROR_STATUSES:
                errors += 1
            else:
                pending += 1
        walls = [float(row["wall_seconds"]) for row in selected if isinstance(row.get("wall_seconds"), (int, float))]
        first_ttft = [float(row["ttft_seconds"]) for row in selected if isinstance(row.get("ttft_seconds"), (int, float))]
        final_ttft = [float(row["ttft_final_seconds"]) for row in selected if isinstance(row.get("ttft_final_seconds"), (int, float))]
        known = [row for row in selected if row.get("primary_pass") is not None]
        base = [row for row in selected if row.get("humaneval_base_pass") is not None]
        enhanced = [row for row in selected if row.get("humaneval_enhanced_pass") is not None]
        strict = [row for row in selected if row.get("ifeval_strict_prompt_pass") is not None]
        loose = [row for row in selected if row.get("ifeval_loose_prompt_pass") is not None]
        strict_instruction = [row for row in selected if row.get("ifeval_strict_instruction_count") is not None]
        loose_instruction = [row for row in selected if row.get("ifeval_loose_instruction_count") is not None]
        summary = {field: None for field in SUMMARY_FIELDS}
        summary.update({key: meta.get(key) for key in (
            "profile_key", "display_name", "model_name", "profile_kind", "profile_state",
            "runtime_backend", "comparison_mode", "context_tokens", "output_cap_tokens", "request_timeout_seconds",
            "seed", "request_options_json", "placement_json", "profile_tag", "profile_sha256",
            "catalog_sha256", "model_artifact_sha256", "model_artifact_path",
            "config_path", "config_sha256", "runner_sha256", "profile_parameters_json",
            "output_dir", "grading_summary_path", "cold_load_seconds",
            "cold_load_source")})
        summary.update({
            "suite": suite, "suite_label": SUITE_LABELS[suite], "expected_cases": expected,
            "terminal_cases": len(selected),
            "completed_cases": sum(row["attempt_status"] == "completed" for row in selected),
            "truncated_cases": sum(row["attempt_status"] == "truncated" for row in selected),
            "pending_cases": pending, "error_cases": errors,
            "cap_cases": sum(row["capped"] is True for row in selected),
            "graded_cases": len(known), "graded_unknown_cases": expected - len(known),
            "quality_passes": sum(row["primary_pass"] is True for row in known),
            "quality_denominator": len(known),
            "quality_rate": sum(row["primary_pass"] is True for row in known) / len(known) if known else None,
            "quality_metric": "EvalPlus enhanced pass@1" if suite == "humanevalplus" else "IFEval strict prompt accuracy",
            "humaneval_base_passes": sum(row["humaneval_base_pass"] is True for row in base),
            "humaneval_base_graded": len(base),
            "humaneval_enhanced_passes": sum(row["humaneval_enhanced_pass"] is True for row in enhanced),
            "humaneval_enhanced_graded": len(enhanced),
            "ifeval_strict_prompt_passes": sum(row["ifeval_strict_prompt_pass"] is True for row in strict),
            "ifeval_strict_prompt_graded": len(strict),
            "ifeval_loose_prompt_passes": sum(row["ifeval_loose_prompt_pass"] is True for row in loose),
            "ifeval_loose_prompt_graded": len(loose),
            "ifeval_strict_instruction_passes": _sum_int(strict_instruction, "ifeval_strict_instruction_pass_count"),
            "ifeval_strict_instruction_total": _sum_int(strict_instruction, "ifeval_strict_instruction_count"),
            "ifeval_loose_instruction_passes": _sum_int(loose_instruction, "ifeval_loose_instruction_pass_count"),
            "ifeval_loose_instruction_total": _sum_int(loose_instruction, "ifeval_loose_instruction_count"),
            "selected_prompt_tokens": _sum_int(selected, "prompt_tokens"),
            "selected_completion_tokens": _sum_int(selected, "completion_tokens"),
            "selected_reasoning_tokens": _sum_int(selected, "reasoning_tokens"),
            "all_request_prompt_tokens": _sum_int(attempts, "prompt_tokens"),
            "all_request_completion_tokens": _sum_int(attempts, "completion_tokens"),
            "all_request_reasoning_tokens": _sum_int(attempts, "reasoning_tokens"),
            "weighted_server_generation_tps": _weighted(selected, "server_generation_tps", "server_predicted_ms"),
            "weighted_server_prompt_tps": _weighted(selected, "server_prompt_tps", "server_prompt_ms"),
            "all_request_wall_seconds": sum(float(row["wall_seconds"]) for row in attempts
                                             if isinstance(row.get("wall_seconds"), (int, float))),
            "median_wall_seconds": median(walls), "p95_wall_seconds": percentile(walls, .95),
            "median_ttft_seconds": median(first_ttft), "p95_ttft_seconds": percentile(first_ttft, .95),
            "median_final_ttft_seconds": median(final_ttft), "p95_final_ttft_seconds": percentile(final_ttft, .95),
            "attempt_count": len(attempts), "retry_count": max(0, len(attempts) - expected),
        })
        output.append(summary)
    return output


def diagnostic_rows(study_dir: Path, initial: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], set[Path]]:
    rows = [dict(row) for row in initial]
    sources: set[Path] = set()
    repair = study_dir / "oss-repair"
    if repair.is_dir():
        for path in sorted(repair.rglob("*.json")):
            lowered = "/".join(part.lower() for part in path.parts)
            value = read_json(path)
            mapping = value if isinstance(value, dict) else {}
            kind = str(mapping.get("kind") or "").lower()
            is_diagnostic = (
                "diagnostic" in lowered or "probe" in lowered or "diagnostic" in kind
                or path.name in {"handoff.json", "declaration.json"}
            )
            if not is_diagnostic:
                continue
            sources.add(path)
            diagnostic = mapping.get("diagnostic") if isinstance(mapping.get("diagnostic"), dict) else {}
            caps = _first(mapping.get("caps"), mapping.get("parameters"), diagnostic, {})
            if isinstance(caps, dict):
                caps = dict(caps)
                for key in ("max_tokens", "request_timeout_seconds", "seed", "case_indices"):
                    if key in mapping and key not in caps:
                        caps[key] = mapping[key]
                if isinstance(mapping.get("recipe"), dict):
                    caps["recipe"] = mapping["recipe"]
            rows.append({
                "name": mapping.get("name") or mapping.get("kind") or path.stem,
                "status": mapping.get("status") or mapping.get("state") or "recorded",
                "scored": False,
                "caps_json": json_cell(caps if isinstance(caps, dict) else {"value": caps}),
                "path": rel(path),
                "note": "Repair diagnostic only; excluded from the frozen-64 benchmark denominator.",
            })
    if not rows:
        rows.append({"name": "OSS repair diagnostics", "status": "pending", "scored": False,
                     "caps_json": "{}", "path": rel(repair),
                     "note": "Diagnostics, if produced, remain unscored and separate from the full-64 repaired rerun."})
    return rows, sources


def csv_bytes(fields: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, extrasaction="raise")
    writer.writeheader()
    for source in rows:
        row = {}
        for field in fields:
            value = source.get(field)
            if isinstance(value, bool):
                value = "true" if value else "false"
            row[field] = "" if value is None else value
        writer.writerow(row)
    return buffer.getvalue().encode()


def _display(value: Any, *, digits: int = 3) -> str:
    if value is None or value == "":
        return "N/A"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:,.{digits}f}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def _table(rows: Sequence[Mapping[str, Any]], fields: Sequence[str], table_id: str) -> str:
    headers = "".join(f"<th>{html.escape(field)}</th>" for field in fields)
    body = []
    for row in rows:
        cells = "".join(f"<td>{html.escape(_display(row.get(field)))}</td>" for field in fields)
        body.append(f"<tr>{cells}</tr>")
    return f'<table id="{html.escape(table_id, quote=True)}"><thead><tr>{headers}</tr></thead><tbody>{"".join(body)}</tbody></table>'


def _chart(summary: Sequence[Mapping[str, Any]]) -> str:
    maxima = {}
    for key in ("weighted_server_generation_tps", "selected_completion_tokens", "median_wall_seconds"):
        maxima[key] = max((float(row[key]) for row in summary if isinstance(row.get(key), (int, float))), default=0)
    blocks = []
    for row in summary:
        quality = float(row["quality_rate"]) if isinstance(row.get("quality_rate"), (int, float)) else None
        metrics = [
            ("official success", quality, 1, f'{_display(row.get("quality_passes"), digits=0)}/{_display(row.get("quality_denominator"), digits=0)}'),
            ("server generation TPS", row.get("weighted_server_generation_tps"), maxima["weighted_server_generation_tps"], _display(row.get("weighted_server_generation_tps"))),
            ("selected output tokens", row.get("selected_completion_tokens"), maxima["selected_completion_tokens"], _display(row.get("selected_completion_tokens"), digits=0)),
            ("median request wall", row.get("median_wall_seconds"), maxima["median_wall_seconds"], _display(row.get("median_wall_seconds"))),
        ]
        bars = []
        for label, value, maximum, rendered in metrics:
            width = (float(value) / float(maximum) * 100) if isinstance(value, (int, float)) and maximum else 0
            bars.append(f'<div class="metric"><span>{html.escape(label)}</span><div class="track"><i style="width:{max(0, min(width, 100)):.3f}%"></i></div><b>{html.escape(rendered)}</b></div>')
        title = f'{row["display_name"]} · {row["suite_label"]}'
        progress = f'{row["terminal_cases"]}/{row["expected_cases"]} terminal · {row["cap_cases"]} caps · {row["graded_unknown_cases"]} grade unknown'
        blocks.append(f'<article class="chartcard"><h3>{html.escape(title)}</h3><p>{html.escape(progress)}</p>{"".join(bars)}</article>')
    return "".join(blocks)


def _model_overview(summary: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Roll up progress/cost only; never synthesize cross-suite quality."""
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in summary:
        grouped.setdefault(str(row["profile_key"]), []).append(row)
    output = []
    for rows in grouped.values():
        first = rows[0]
        terminal = sum(int(row["terminal_cases"]) for row in rows)
        expected = sum(int(row["expected_cases"]) for row in rows)
        output.append({
            "display_name": first["display_name"],
            "profile_kind": first["profile_kind"],
            "profile_state": first["profile_state"],
            "comparison_mode": first["comparison_mode"],
            "terminal_of_64": f"{terminal}/{expected}",
            "completed": sum(int(row["completed_cases"]) for row in rows),
            "truncated": sum(int(row["truncated_cases"]) for row in rows),
            "pending": sum(int(row["pending_cases"]) for row in rows),
            "errors": sum(int(row["error_cases"]) for row in rows),
            "caps": sum(int(row["cap_cases"]) for row in rows),
            "grade_unknown": sum(int(row["graded_unknown_cases"]) for row in rows),
            "selected_completion_tokens": sum(int(row["selected_completion_tokens"]) for row in rows),
            "all_request_wall_seconds": sum(float(row["all_request_wall_seconds"]) for row in rows),
            "context_tokens": first["context_tokens"],
            "output_cap_tokens": first["output_cap_tokens"],
            "request_timeout_seconds": first["request_timeout_seconds"],
        })
    return output


def local_href(target: Path, output_dir: Path) -> str:
    """Build a local link whose target remains inside the public checkout."""
    try:
        resolved = target.resolve(strict=False)
        resolved.relative_to(ROOT.resolve())
    except (OSError, ValueError) as exc:
        raise ReportError(f"HTML link target is outside the public root: {target}") from exc
    return Path(os.path.relpath(resolved, start=output_dir.resolve())).as_posix()


def build_html(summary: Sequence[Mapping[str, Any]], details: Sequence[Mapping[str, Any]],
               diagnostics: Sequence[Mapping[str, Any]], report_status: Mapping[str, Any],
               output_dir: Path) -> str:
    overview = _model_overview(summary)
    overview_fields = [
        "display_name", "profile_kind", "profile_state", "comparison_mode", "terminal_of_64",
        "completed", "truncated", "pending", "errors", "caps", "grade_unknown",
        "selected_completion_tokens", "all_request_wall_seconds", "context_tokens",
        "output_cap_tokens", "request_timeout_seconds",
    ]
    summary_fields = [
        "display_name", "profile_kind", "profile_state", "comparison_mode", "suite_label", "terminal_cases",
        "expected_cases", "pending_cases", "error_cases", "cap_cases", "quality_passes",
        "quality_denominator", "graded_unknown_cases", "weighted_server_generation_tps",
        "weighted_server_prompt_tps", "selected_completion_tokens", "all_request_wall_seconds",
        "median_wall_seconds", "p95_wall_seconds", "median_ttft_seconds", "p95_ttft_seconds",
        "median_final_ttft_seconds", "p95_final_ttft_seconds", "cold_load_seconds",
        "context_tokens", "output_cap_tokens", "request_timeout_seconds", "profile_tag",
        "profile_sha256", "request_options_json", "placement_json",
    ]
    detail_fields = [
        "display_name", "suite", "case_id", "attempt_number", "attempt_status",
        "selected_for_benchmark", "capped", "primary_pass", "grade_known", "prompt_tokens",
        "completion_tokens", "reasoning_tokens", "server_generation_tps", "server_prompt_tps",
        "wall_seconds", "ttft_seconds", "ttft_final_seconds", "source_path", "grade_source_path",
    ]
    cards = [
        ("report", report_status["phase"]),
        ("queue", report_status["supervisor_status"]),
        ("new profiles terminal", f'{report_status["queue_profiles_terminal"]}/{report_status["queue_profiles_expected"]}'),
        ("new cases terminal", f'{report_status["queue_cases_terminal"]}/{report_status["queue_cases_expected"]}'),
        ("pending / errors / caps", f'{report_status["queue_cases_pending"]} / {report_status["queue_cases_error"]} / {report_status["queue_cap_cases"]}'),
        ("generated", report_status["generated_at"]),
    ]
    cards_html = "".join(f'<div class="card"><b>{html.escape(label)}</b><span>{html.escape(str(value))}</span></div>' for label, value in cards)
    limitations = [
        "HumanEval+ enhanced pass@1 and IFEval strict prompt accuracy are separate official metrics; they are never pooled into one quality score.",
        "Original controls, repaired OSS, and every new profile remain distinct. The repaired rerun is the full frozen 64 and is never best-of or selectively combined with the original OSS run.",
        "Non-OSS candidate profiles use the matched T0/medium baseline (seed 20261023, no extra sampler kwargs). This is a controlled baseline, not a claim about inherent or peak capability or publisher-optimal settings.",
        "Repaired OSS uses a separate diagnostic-selected low/full-sampling treatment and is flagged as different decoding from the original medium/T0 OSS control.",
        "The selected benchmark response is the first completed or truncated attempt. Failed/cancelled retries remain in detail and all-request cost totals.",
        "Caps count only selected terminal truncations. Missing official grades stay unknown; request completion never implies a pass.",
        "Generation TPS is server-reported predicted_per_second weighted by predicted_ms. Cold load is shown only when a source records it.",
        "Repair probes/diagnostics are an unscored appendix and never enter benchmark denominators.",
        "This comparison is not aggregated with the older custom 480-row study or the tiny NPU assessment.",
    ]
    limitations_html = "".join(f"<li>{html.escape(item)}</li>" for item in limitations)
    cross_report_href = html.escape(local_href(
        ROOT / "results/model-evidence-20261023-v2/report.html", output_dir
    ), quote=True)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Public larger-model expansion — frozen 64</title>
<style>
:root{{--ink:#17202a;--muted:#5c6873;--line:#d6dde3;--paper:#fff;--blue:#1769aa;--gold:#bd7b00}}
*{{box-sizing:border-box}}body{{margin:0;background:#f4f6f8;color:var(--ink);font:14px system-ui,sans-serif}}main{{max-width:1800px;margin:auto;padding:24px}}
h1{{margin:.1em 0}}h2{{margin-top:28px}}.lede{{color:var(--muted);max-width:1050px}}.warning{{background:#fff4ce;border-left:5px solid var(--gold);padding:12px;max-width:1200px}}
.cards,.charts{{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:10px;margin:16px 0}}.card,.chartcard{{background:var(--paper);border:1px solid var(--line);border-radius:8px;padding:12px}}.card span{{display:block;font-size:18px;margin-top:5px}}.chartcard{{min-width:300px}}.chartcard h3{{font-size:14px;margin:0}}.chartcard p{{color:var(--muted);font-size:12px}}
.metric{{display:grid;grid-template-columns:145px 1fr 78px;gap:8px;align-items:center;margin:8px 0;font-size:12px}}.metric b{{text-align:right}}.track{{height:10px;background:#e9eef2;border-radius:8px;overflow:hidden}}.track i{{display:block;height:100%;background:linear-gradient(90deg,#1769aa,#38a3a5)}}
.tablewrap{{overflow:auto;max-height:70vh;background:white;border:1px solid var(--line)}}table{{border-collapse:collapse;width:100%;font-size:12px}}th{{position:sticky;top:0;background:#243447;color:#fff;z-index:1}}th,td{{padding:6px 8px;border-bottom:1px solid #e4e8eb;text-align:left;white-space:nowrap}}tr:nth-child(even){{background:#f7f9fa}}input{{width:min(760px,100%);padding:9px;margin:8px 0;border:1px solid #aeb8c2;border-radius:5px}}a{{color:#075ea8}}
</style></head><body><main>
<h1>Public larger-model expansion</h1><p class="lede">Exact same frozen public set: 32 HumanEval+ and 32 IFEval prompts. This file is self-contained and remains valid while the queue is running.</p>
<div class="warning"><b>Read each suite and profile separately.</b> Different quality definitions are exposed side by side, never collapsed into a synthetic score.</div>
<div class="cards">{cards_html}</div>
<p><a href="detailed-cases.csv">detailed cases CSV</a> · <a href="model-suite-summary.csv">model/suite summary CSV</a> · <a href="diagnostics.csv">unscored diagnostics CSV</a> · <a href="report-status.json">report status</a> · <a href="provenance.json">provenance</a> · <a href="{cross_report_href}">older cross-study evidence (separate)</a></p>
<h2>Interpretation guardrails</h2><ul>{limitations_html}</ul>
<h2>Profile progress (terminal / 64)</h2><div class="tablewrap">{_table(overview, overview_fields, 'overview')}</div>
<h2>Deep comparison chart</h2><div class="charts">{_chart(summary)}</div>
<h2>Model × suite summary</h2><input id="summaryFilter" placeholder="Filter model, profile, suite, backend…" oninput="filterTable('summary',this.value)"><div class="tablewrap">{_table(summary, summary_fields, 'summary')}</div>
<h2>Attempt-level detail</h2><p>All attempts are retained. Only rows marked selected_for_benchmark feed quality and selected performance; all attempts feed all-request cost.</p><input id="detailFilter" placeholder="Filter case, status, model, suite…" oninput="filterTable('details',this.value)"><div class="tablewrap">{_table(details, detail_fields, 'details')}</div>
<h2>Unscored repair diagnostics</h2><div class="tablewrap">{_table(diagnostics, DIAGNOSTIC_FIELDS, 'diagnostics')}</div>
<script>function filterTable(id,q){{q=q.toLowerCase();for(const r of document.querySelectorAll('#'+id+' tbody tr'))r.style.display=r.innerText.toLowerCase().includes(q)?'':'none'}}</script>
</main></body></html>"""


def _generate(*, study_dir: Path, output_dir: Path | None,
              catalog_path: Path, manifest_path: Path,
              expected_manifest_sha256: str | None,
              include_original_controls: bool) -> dict[str, Any]:
    study_dir = study_dir.resolve()
    output_dir = (output_dir or study_dir / "report").resolve()
    catalog_path = catalog_path.resolve()
    manifest_path = manifest_path.resolve()
    manifest, cases, manifest_sha = load_manifest(manifest_path, expected_manifest_sha256)
    selection_identity = None
    if _ACTIVE_EVIDENCE is not None:
        try:
            selection_identity = public_evidence.selection_identity_sha256(manifest)
        except (KeyError, public_evidence.EvidenceError) as exc:
            raise ReportError(f"invalid published selection identity: {exc}") from exc
        expected_selection = _ACTIVE_EVIDENCE.selection_identity_sha256
        if expected_selection is None:
            raise ReportError("evidence-manifest.json lacks selection_identity_sha256")
        if selection_identity != expected_selection:
            raise ReportError(
                "published benchmark selection identity differs: "
                f"expected {expected_selection}, found {selection_identity}"
            )
    catalog_value = read_json(catalog_path)
    if not isinstance(catalog_value, dict):
        raise ReportError("profile catalog must be a JSON object")
    catalog_sha = sha256_file(catalog_path)
    supervisor_path = study_dir / "supervisor-status.json"
    if _ACTIVE_EVIDENCE is not None:
        _published(supervisor_path)
    supervisor = read_json_if_present(supervisor_path)
    if (_ACTIVE_EVIDENCE is None
            and supervisor.get("manifest_sha256") not in (None, manifest_sha)):
        raise ReportError("supervisor status is bound to a different manifest")
    if (_ACTIVE_EVIDENCE is None
            and supervisor.get("catalog_sha256") not in (None, catalog_sha)):
        raise ReportError("supervisor status is bound to a different profile catalog")
    specs, supervisor_diagnostics = profile_specs(study_dir, catalog_value, supervisor)
    if not include_original_controls:
        specs = [spec for spec in specs if spec["kind"] != "original_control"]
    details: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    sources: set[Path] = {catalog_path, manifest_path}
    if supervisor_path.is_file():
        sources.add(supervisor_path)
    queue_profiles_terminal = 0
    for spec in specs:
        meta, meta_sources = profile_metadata(spec, catalog_sha, study_dir)
        sources.update(meta_sources)
        if (_ACTIVE_EVIDENCE is None
                and meta.get("manifest_sha256") not in (None, manifest_sha)):
            raise ReportError(f"{spec['key']} run is bound to a different manifest")
        grades, grade_sources = load_grades(Path(meta["grade_path"]), cases, Path(spec["output_dir"]), study_dir)
        sources.update(grade_sources)
        profile_details, attempt_sources = detail_rows_for_profile(spec, meta, cases, grades)
        sources.update(attempt_sources)
        details.extend(profile_details)
        summaries.extend(summarize_profile(meta, profile_details, cases))
        if spec["kind"] == "expansion" and str(meta["profile_state"]).lower() in FINAL_SUPERVISOR_STATES:
            queue_profiles_terminal += 1
    diagnostics, diagnostic_sources = diagnostic_rows(study_dir, supervisor_diagnostics)
    sources.update(diagnostic_sources)
    expansion_summary = [row for row in summaries if row["profile_kind"] == "expansion"]
    queue_expected = len(_catalog_profiles(catalog_value))
    supervisor_state = str(supervisor.get("status") or "not_started")
    phase = "final" if supervisor_state.lower() in FINAL_SUPERVISOR_STATES else "partial"
    report_status = {
        "schema_version": 1,
        "generated_at": utc_now(),
        "phase": phase,
        "supervisor_status": supervisor_state,
        "manifest_path": rel(manifest_path),
        "manifest_sha256": manifest_sha,
        "catalog_path": rel(catalog_path),
        "catalog_sha256": catalog_sha,
        "generator_path": rel(Path(__file__)),
        "generator_sha256": sha256_file(Path(__file__)),
        "queue_profiles_expected": queue_expected,
        "queue_profiles_terminal": queue_profiles_terminal,
        "queue_cases_expected": sum(int(row["expected_cases"]) for row in expansion_summary),
        "queue_cases_terminal": sum(int(row["terminal_cases"]) for row in expansion_summary),
        "queue_cases_pending": sum(int(row["pending_cases"]) for row in expansion_summary),
        "queue_cases_error": sum(int(row["error_cases"]) for row in expansion_summary),
        "queue_cap_cases": sum(int(row["cap_cases"]) for row in expansion_summary),
        "attempt_selection": "first_completed_or_truncated",
        "quality_policy": "official_grades_only; unknown remains unknown; no cross-suite pooling",
    }
    if _ACTIVE_EVIDENCE is not None:
        report_status.update({
            "evidence_manifest_path": rel(_ACTIVE_EVIDENCE.manifest_path),
            "evidence_manifest_sha256": _ACTIVE_EVIDENCE.manifest_sha256,
            "selection_identity_sha256": selection_identity,
        })
    output_dir.mkdir(parents=True, exist_ok=True)
    detail_path = output_dir / "detailed-cases.csv"
    summary_path = output_dir / "model-suite-summary.csv"
    diagnostic_path = output_dir / "diagnostics.csv"
    html_path = output_dir / "report.html"
    report_status_path = output_dir / "report-status.json"
    atomic_write(detail_path, csv_bytes(DETAIL_FIELDS, details))
    atomic_write(summary_path, csv_bytes(SUMMARY_FIELDS, summaries))
    atomic_write(diagnostic_path, csv_bytes(DIAGNOSTIC_FIELDS, diagnostics))
    atomic_json(report_status_path, report_status)
    atomic_write(
        html_path,
        build_html(summaries, details, diagnostics, report_status, output_dir).encode(),
    )
    source_records = [{"path": rel(path), "sha256": sha256_file(path)} for path in sorted(sources) if path.is_file()]
    provenance = {
        "schema_version": 1, "generated_at": report_status["generated_at"],
        "generator": {"path": rel(Path(__file__)), "sha256": report_status["generator_sha256"]},
        "frozen_manifest": {"path": rel(manifest_path), "sha256": manifest_sha,
                            "case_count": len(cases), "selection_seed": manifest.get("selection_seed"),
                            "selection_identity_sha256": selection_identity},
        "catalog": {"path": rel(catalog_path), "sha256": catalog_sha},
        "sources": source_records,
        "outputs": [
            {"path": rel(path), "sha256": sha256_file(path)}
            for path in (detail_path, summary_path, diagnostic_path, report_status_path, html_path)
        ],
        "row_counts": {"detail": len(details), "model_suite_summary": len(summaries),
                       "unscored_diagnostics": len(diagnostics)},
    }
    atomic_json(output_dir / "provenance.json", provenance)
    return {"details": details, "summaries": summaries, "diagnostics": diagnostics,
            "status": report_status, "provenance": provenance, "output_dir": output_dir}


def generate(*, study_dir: Path | None = None, output_dir: Path | None = None,
             catalog_path: Path | None = None, manifest_path: Path | None = None,
             expected_manifest_sha256: str | None = FROZEN_MANIFEST_SHA256,
             include_original_controls: bool = True,
             evidence_root: Path | None = None,
             evidence_manifest: Path | None = None) -> dict[str, Any]:
    """Build the report, optionally from a verified relocated public archive."""
    evidence = None
    if evidence_root is not None:
        try:
            evidence = public_evidence.verify_manifest(
                evidence_root, manifest_path=evidence_manifest,
                required_paths=(
                    "results/public-benchmark-pilot-20261023-v1/manifest.json",
                    "runtime/public-model-expansion-20261024-v1/profile-catalog.json",
                    "results/public-model-expansion-20261024-v1/supervisor-status.json",
                ),
            )
        except (OSError, public_evidence.EvidenceError) as exc:
            raise ReportError(str(exc)) from exc
    with _public_root(evidence):
        if evidence is not None:
            try:
                selected_study = (
                    evidence.resolve(str(study_dir), required=True)
                    if study_dir is not None else DEFAULT_STUDY
                )
                selected_catalog = (
                    evidence.resolve(str(catalog_path), required=True)
                    if catalog_path is not None else DEFAULT_CATALOG
                )
                selected_manifest = (
                    evidence.resolve(str(manifest_path), required=True)
                    if manifest_path is not None else FROZEN_MANIFEST
                )
            except public_evidence.EvidenceError as exc:
                raise ReportError(str(exc)) from exc
            assert selected_study is not None
            assert selected_catalog is not None
            assert selected_manifest is not None
        else:
            selected_study = Path(study_dir).resolve() if study_dir is not None else DEFAULT_STUDY
            selected_catalog = Path(catalog_path).resolve() if catalog_path is not None else DEFAULT_CATALOG
            selected_manifest = Path(manifest_path).resolve() if manifest_path is not None else FROZEN_MANIFEST
        return _generate(
            study_dir=selected_study,
            output_dir=output_dir,
            catalog_path=selected_catalog,
            manifest_path=selected_manifest,
            expected_manifest_sha256=(None if evidence is not None else expected_manifest_sha256),
            include_original_controls=include_original_controls,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--catalog", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--evidence-root", type=Path)
    parser.add_argument("--evidence-manifest", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if (args.evidence_root is None) != (args.output_dir is None):
        parser.error("--evidence-root and --output-dir must be supplied together for a public rebuild")
    result = generate(study_dir=args.study_dir, output_dir=args.output_dir,
                      catalog_path=args.catalog, manifest_path=args.manifest,
                      expected_manifest_sha256=FROZEN_MANIFEST_SHA256,
                      evidence_root=args.evidence_root,
                      evidence_manifest=args.evidence_manifest)
    print(json.dumps({"output_dir": rel(result["output_dir"]),
                      "phase": result["status"]["phase"],
                      "detail_rows": len(result["details"]),
                      "summary_rows": len(result["summaries"])}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
