#!/usr/bin/env python3
"""Generate the offline, reproducible final model benchmark report.

The raw JSONL request rows remain authoritative.  This module deliberately does
not copy model response text into its outputs and never runs a model or candidate
code.  Lifecycle/configuration JSON is used only to add provenance.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import io
import json
import math
import os
import re
import statistics
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping
from urllib.parse import urlsplit, urlunsplit


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = 1
AMBIGUOUS_CONSTRAINT_VERSION = "2026-10-18.1"
JSON_FORMAT_CASE_IDS = frozenset({"hard_transaction_ledger", "hard_dependency_constraints"})
_OUTER_CODE_FENCE_RE = re.compile(
    r"^\s*```(?:json|python|py)?\s*\n?(.*?)\n?```\s*$",
    re.DOTALL | re.IGNORECASE,
)
CSV_COLUMNS = (
    "row_type", "row_id", "call_id", "run_id", "source_file", "source_line", "started_at", "finished_at",
    "model", "weight_quant", "draft_weight_quant", "kv_cache", "runtime", "backend", "endpoint", "suite",
    "case_id", "case_version", "case_source_sha256", "prompt_sha256", "parent_case_id",
    "diagnostic_variant", "diagnostic_domain_changed", "diagnostic_algorithm_guidance",
    "repeat_index", "thinking",
    "reasoning_budget_tokens", "reasoning_budget_tokens_requested",
    "reasoning_budget_tokens_config_default", "native_thinking_budget_tokens_requested",
    "native_thinking_budget_application_evidence", "strata_pcie_frac", "strata_aux_cpus",
    "ledger_json_schema_path", "ledger_json_schema_sha256", "ledger_json_schema_requested",
    "chat_template_thinking", "temperature", "sampling_seed",
    "context_max_tokens", "context_mode", "output_token_limit", "api_input_tokens",
    "ollama_prompt_eval_count", "openai_prompt_tokens", "native_prompt_n",
    "native_cache_n", "api_cached_prompt_tokens", "completion_tokens",
    "ollama_eval_count", "native_decode_n", "native_draft_n", "native_draft_n_accepted",
    "native_draft_acceptance_rate",
    "reasoning_chars", "reasoning_tokens", "visible_answer_tokens",
    "native_prompt_seconds", "native_decode_seconds", "native_load_seconds",
    "native_prompt_tps", "native_decode_tps", "wall_decode_tps", "wall_seconds",
    "ttft_reasoning_seconds", "ttft_final_seconds", "finish_reason", "stream_complete", "truncated",
    "outcome", "content_passed", "semantic_behavior_passed", "source_edit_fidelity_passed",
    "comparison_content_passed", "rescored_outcome", "rescored_cause", "evaluation_detail", "hidden_tests_passed", "hidden_test_count",
    "hidden_failure_labels", "hidden_failure_categories", "raw_json_object", "outer_code_fence", "empty_answer",
    "quality_eligible", "speed_eligible", "latency_eligible", "startup_seconds", "response_bytes",
    "exclusion_reasons", "request_error_type",
    "request_error_message", "source_config", "source_config_sha256", "private_config", "sampling_config",
    "runtime_options", "requested_native_runtime_controls", "ignored_runtime_options", "engine_environment", "profile_id", "match_id",
    "specialist_profile", "case_group", "objective_outcome", "mechanical_constraints_passed",
    "objective_count", "objectives_passed", "assertion_count", "assertions_passed",
    "actual_context_max_tokens", "study_tag", "study_status", "study_receipt",
    "model_session_run_id", "session_started_at", "server_ready_at", "holding_started_at",
    "session_hold_seconds", "capability_note", "reasoning_controls_requested",
    "reasoning_observed",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _number(value: Any) -> int | float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_id(value: Any, length: int = 16) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:length]


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def _absolute(value: Any) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return str(path.resolve())


def _safe_endpoint(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    parsed = urlsplit(value)
    if not parsed.scheme or not parsed.hostname:
        return value
    host = parsed.hostname
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    port = f":{parsed.port}" if parsed.port is not None else ""
    return urlunsplit((parsed.scheme, host + port, parsed.path, "", ""))


def _safe_message(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)[:2000]
    def clean(match: re.Match[str]) -> str:
        return _safe_endpoint(match.group(0)) or "[URL]"
    return re.sub(r"https?://[^\s]+", clean, text)


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _jsonl(path: Path) -> tuple[list[tuple[int, dict[str, Any]]], int]:
    rows: list[tuple[int, dict[str, Any]]] = []
    bad = 0
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return rows, 1
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            bad += 1
            continue
        if isinstance(value, dict):
            rows.append((number, value))
    return rows, bad


def _request_row(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and isinstance(value.get("call_id"), str)
        and isinstance(value.get("case_id"), str)
        and isinstance(value.get("request"), dict)
        and isinstance(value.get("evaluation"), dict)
    )


def _specialist_receipt_index(
    results_dir: Path,
) -> tuple[dict[str, dict[str, Any]], list[Path]]:
    """Index specialist result artifacts by the exact paths recorded by receipts."""
    result: dict[str, dict[str, Any]] = {}
    sources: list[Path] = []
    for receipt_path in sorted(results_dir.glob("*-study-trial.json")):
        receipt = _load_json(receipt_path)
        if not isinstance(receipt, dict):
            continue
        artifacts = receipt.get("artifacts") if isinstance(receipt.get("artifacts"), dict) else {}
        session = receipt.get("session") if isinstance(receipt.get("session"), dict) else {}
        source_config = receipt.get("source_config") if isinstance(receipt.get("source_config"), dict) else {}
        controls = receipt.get("controls") if isinstance(receipt.get("controls"), dict) else {}
        private_artifact = artifacts.get("private_config") if isinstance(artifacts.get("private_config"), dict) else {}
        trial = {
            "runtime_backend": receipt.get("runtime_backend"),
            "model_name": session.get("model_name"),
            "source_config": source_config.get("path"),
            "private_config": private_artifact.get("path") if private_artifact.get("exists") is True else None,
            "study_tag": receipt.get("tag"),
            "study_status": receipt.get("status"),
            "study_receipt": str(receipt_path.resolve()),
            "model_session_run_id": session.get("run_id"),
            "actual_context_max_tokens": _number(session.get("context")),
            "initial_load_seconds": _number(session.get("initial_load_seconds")),
            "session_hold_seconds": _number(session.get("hold_seconds")),
            "session_started_at": session.get("session_started_at"),
            "server_ready_at": session.get("server_ready_at"),
            "holding_started_at": session.get("holding_started_at"),
            "specialist_controls": controls,
            "source_config_sha256": source_config.get("sha256"),
        }
        found = False
        for name in ("code", "prose"):
            artifact = artifacts.get(name) if isinstance(artifacts.get(name), dict) else {}
            value = artifact.get("path")
            if not isinstance(value, str) or not value:
                continue
            path = Path(value).expanduser()
            if not path.is_absolute():
                path = receipt_path.parent / path
            result[str(path.resolve())] = dict(trial)
            found = True
        if found:
            sources.append(receipt_path)
    return result, sources


def _event_index(results_dir: Path) -> tuple[dict[str, dict[str, Any]], list[tuple[str, dict[str, Any]]], list[dict[str, Any]], list[Path]]:
    trials: dict[str, dict[str, Any]] = {}
    result_to_run: dict[str, str] = {}
    events: list[dict[str, Any]] = []
    sources: list[Path] = []
    for path in sorted(results_dir.glob("*.jsonl")):
        found_event = False
        for line, value in _jsonl(path)[0]:
            if not isinstance(value.get("event"), str):
                continue
            found_event = True
            event = dict(value)
            event["_source_file"] = _relative(path)
            event["_source_line"] = line
            events.append(event)
            run_id = value.get("run_id") or value.get("session_id")
            if not isinstance(run_id, str):
                continue
            if value.get("event") in ("trial_started", "trial_start", "session_started"):
                trials[run_id] = dict(value)
            if value.get("event") == "server_ready":
                trials.setdefault(run_id, {}).setdefault("server_ready", dict(value))
            if value.get("event") in ("suite_started", "step_started", "step_complete"):
                result = _absolute(value.get("result") or value.get("output"))
                if result:
                    result_to_run[result] = run_id
        if found_event:
            sources.append(path)
    metadata: dict[str, dict[str, Any]] = {}
    for result, run_id in result_to_run.items():
        if run_id in trials:
            metadata[result] = trials[run_id]
    by_tag: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for trial in trials.values():
        tag = trial.get("tag")
        if isinstance(tag, str) and tag:
            by_tag[tag].append(trial)
    # A reused tag is not safe to infer.  Longest-first makes nested session
    # names deterministic (for example flash-next vs flash-next-bulk-32k).
    prefixes = sorted(
        [(tag, values[0]) for tag, values in by_tag.items() if len(values) == 1],
        key=lambda item: (-len(item[0]), item[0]),
    )
    return metadata, prefixes, events, sources


def _cli_option(args: list[Any], flag: str) -> Any:
    if flag not in args:
        return None
    index = args.index(flag)
    if index + 1 >= len(args) or str(args[index + 1]).startswith("--"):
        return True
    return args[index + 1]


def _cli_number(value: Any) -> Any:
    """Keep symbolic CLI values such as ``auto`` while typing numeric controls."""
    if not isinstance(value, str):
        return value
    try:
        number = float(value)
    except ValueError:
        return value
    return int(number) if number.is_integer() else number


def _requested_native_runtime_controls(args: list[Any]) -> dict[str, Any]:
    """Return only native engine controls explicitly present in config argv.

    These are requested configuration values, not observations of device
    placement, cache residency, or the engine's effective runtime settings.
    """
    controls: dict[str, Any] = {}
    numeric = {
        "--fit-ctx", "--batch-size", "--ubatch-size", "--n-cpu-moe", "--gpu-layers",
        "--threads", "--threads-batch", "--poll", "--poll-batch",
    }
    for flag in (
        "--cache-type-k", "--cache-type-v", "--flash-attn", "--fit",
        "--fit-ctx", "--batch-size", "--ubatch-size", "--n-cpu-moe", "--gpu-layers",
        "--threads", "--threads-batch", "--poll", "--poll-batch",
    ):
        if flag not in args:
            continue
        value = _cli_option(args, flag)
        controls[flag.lstrip("-").replace("-", "_")] = (
            _cli_number(value) if flag in numeric else value
        )
    return controls


def _q36_ignored_options(config: Mapping[str, Any], config_path: Path) -> set[str]:
    """Read the q36 startup's explicit option-rejection receipts, when present."""
    log_value = config.get("log")
    if not isinstance(log_value, str) or not log_value:
        return set()
    log_path = Path(log_value)
    if not log_path.is_absolute():
        log_path = config_path.parent / log_path
    try:
        # Validation warnings are emitted before model loading.  Bound the read
        # so a long inference log cannot make report generation expensive.
        with log_path.open("r", encoding="utf-8", errors="replace") as handle:
            prefix = handle.read(256 * 1024)
    except OSError:
        return set()
    return set(re.findall(
        r"^strata-q36: ignoring\s+(--[a-z0-9-]+)(?:\s|$).*not an option of this engine",
        prefix,
        flags=re.MULTILINE | re.IGNORECASE,
    ))


def _config_metadata(trial: Mapping[str, Any]) -> dict[str, Any]:
    config_path = trial.get("private_config") or trial.get("source_config")
    resolved_config = Path(config_path) if isinstance(config_path, str) else None
    value = _load_json(resolved_config) if resolved_config is not None else None
    if not isinstance(value, dict):
        return {
            "reasoning_budget_tokens": None, "sampling": {},
            "runtime_options": {}, "requested_native_runtime_controls": {},
            "ignored_runtime_options": {},
        }
    args = value.get("args") if isinstance(value.get("args"), list) else []
    runtime_options: dict[str, Any] = {}
    for name in ("pool_workers", "cpus", "arch"):
        if name in trial:
            runtime_options[name] = trial[name]
        elif name in value:
            runtime_options[name] = value[name]
    for flag in (
        "--mtp", "--prefill", "--cpu-moe", "--gpu-layers", "--spec", "--suffix-draft",
        "--vram-reserve-mib", "--expert-cache", "--pcie-frac", "--spec-type",
        "--spec-ngram-simple-size-n", "--spec-ngram-simple-size-m",
        "--spec-ngram-simple-min-hits", "--spec-draft-model", "--spec-draft-device",
        "--spec-draft-ngl", "--spec-draft-n-max", "--spec-draft-n-min", "--spec-draft-p-min",
        "--spec-draft-n-ubatch", "--spec-draft-type-k", "--spec-draft-type-v",
        "--chat-template-file",
    ):
        if flag in args:
            runtime_options[flag.lstrip("-").replace("-", "_")] = _cli_option(args, flag)

    # q36 native-mode controls affect memory placement and kernel selection.
    # Missing configuration leaves these unknown; a readable q36 config makes
    # the boolean absence explicit so MMQ-on and --no-mmq are not conflated.
    arch = trial.get("arch") or value.get("arch")
    is_q36_config = arch == "qwen35moe" and "--native" in args
    if is_q36_config:
        runtime_options["native_prompt"] = True
        runtime_options["no_mmq"] = "--no-mmq" in args
        prefill = _cli_option(args, "--prefill")
        if prefill is not None:
            runtime_options["prefill"] = _cli_number(prefill)
        batch = _cli_option(args, "--batch")
        if batch is not None:
            runtime_options["batch"] = _cli_number(batch)
        expert_vram = _cli_option(args, "--expert-vram")
        if expert_vram is not None:
            runtime_options["expert_vram_gib"] = _cli_number(expert_vram)

    ignored_options: dict[str, Any] = {}
    ignored_flags = _q36_ignored_options(value, resolved_config) if is_q36_config and resolved_config else set()
    for flag in sorted(ignored_flags):
        name = flag.lstrip("-").replace("-", "_")
        if name in runtime_options:
            ignored_options[name] = runtime_options.pop(name)
    if runtime_options.get("spec_type") == "draft-eagle3":
        # llama.cpp intentionally omits this flag when the draft should follow
        # the target device.  Preserve that distinction as an explicit null.
        runtime_options.setdefault("spec_draft_device", None)
    return {
        "reasoning_budget_tokens": _number(value.get("reasoning_budget_tokens")),
        "sampling": value.get("sampling") if isinstance(value.get("sampling"), dict) else {},
        "runtime_options": runtime_options,
        "requested_native_runtime_controls": _requested_native_runtime_controls(args),
        "ignored_runtime_options": ignored_options,
    }


def _prism_llama_runtime_name(trial: Mapping[str, Any]) -> str | None:
    """Use the selected custom build manifest instead of the bundle default."""
    config_path = trial.get("source_config")
    if not isinstance(config_path, str):
        return None
    config = _load_json(Path(config_path))
    if not isinstance(config, dict) or not isinstance(config.get("artifact_manifest"), str):
        return None
    manifest_path = Path(config["artifact_manifest"])
    if not manifest_path.is_absolute():
        manifest_path = Path(config_path).parent / manifest_path
    manifest = _load_json(manifest_path)
    if not isinstance(manifest, dict):
        return None
    source = manifest.get("source")
    if not isinstance(source, dict) or source.get("llama_repo") != "https://github.com/PrismML-Eng/llama.cpp.git":
        return None
    unknown = "llama.cpp server (Prism custom build; identity unverified)"
    runtime = manifest.get("runtime")
    if not isinstance(runtime, dict):
        return unknown
    server = runtime.get("server")
    release = runtime.get("release")
    revision = source.get("llama_revision")
    if (not isinstance(server, dict) or not isinstance(release, str)
            or not re.fullmatch(r"prism-[a-zA-Z0-9._-]+", release)
            or not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision)
            or not isinstance(server.get("sha256"), str)
            or not re.fullmatch(r"[0-9a-f]{64}", server["sha256"])):
        return unknown
    expected = server.get("path")
    configured = config.get("server")
    recorded = trial.get("server") or trial.get("engine") or configured
    if not all(isinstance(value, str) and value for value in (expected, configured, recorded)):
        return unknown
    if len({str(Path(value).resolve()) for value in (expected, configured, recorded)}) != 1:
        return unknown
    return f"llama.cpp server ({release})"


def _runtime_name(trial: Mapping[str, Any], model: str | None) -> str | None:
    runtime = trial.get("runtime")
    if isinstance(runtime, str):
        match = re.search(r"ollama-(\d+(?:\.\d+)+)", runtime)
        if match:
            return f"Ollama {match.group(1)}"
        ready = trial.get("server_ready") if isinstance(trial.get("server_ready"), dict) else {}
        version = ready.get("version")
        if isinstance(version, dict):
            version = version.get("version")
        if isinstance(version, str):
            return f"Ollama {version}"
        return f"Ollama ({Path(runtime).parent.name})"
    source = str(trial.get("source_config", ""))
    server = str(trial.get("server", ""))
    engine = str(trial.get("engine", ""))
    if trial.get("runtime_backend") == "llama" or "llama" in source or "llama-server" in server:
        return _prism_llama_runtime_name(trial) or "llama.cpp server (Ollama 0.40.6 bundle)"
    if "qwen122" in source or "qwen122" in engine or (model and "122" in model):
        return "Strata experimental patched 0.1.45"
    if source or engine:
        return "Strata 0.1.45 pinned"
    return None


def _legacy_trial(source: Path) -> dict[str, Any]:
    """Source-backed metadata for pre-runner original-service artifacts.

    baseline-summary.json and comparison-summary.json record these filename
    families as the original Ollama 0.20.0 / f16 baseline.  No other missing
    fields are inferred.
    """
    name = source.name
    if name.startswith(("old-f16-", "shared-old-f16-", "context-indexed-low-", "current-models-restored-", "initial-probe")):
        return {
            "runtime": "/recorded/ollama-0.20.0/runtime.json",
            "cache": "f16",
            "model_name": "gpt-oss-20b:64k",
            "metadata_source": "results/baseline-summary.json and results/comparison-summary.json",
        }
    return {}


_QUANT_LABEL_RE = re.compile(
    r"(?<![A-Z0-9])((?:UD-)?(?:IQ|Q|TQ|PQ|PTQ)\d+(?:_[A-Z0-9]+)+|MXFP\d+|BF16|F16|F32)(?![A-Z0-9])",
    re.IGNORECASE,
)
_UNKNOWN_UD_QUANT_RE = re.compile(
    r"(?<![A-Z0-9])(UD-[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+)(?![A-Z0-9])",
    re.IGNORECASE,
)


def _explicit_quant(text: Any) -> str | None:
    if not isinstance(text, str):
        return None
    match = _QUANT_LABEL_RE.search(text) or _UNKNOWN_UD_QUANT_RE.search(text)
    return match.group(1).upper() if match else None


def _config_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from _config_strings(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _config_strings(item)


def _weight_quant(model: str | None, row: Mapping[str, Any], trial: Mapping[str, Any]) -> str | None:
    response = row.get("response") if isinstance(row.get("response"), dict) else {}
    native = response.get("native_ollama") if isinstance(response.get("native_ollama"), dict) else {}
    for container in (native, row.get("telemetry_before"), row.get("telemetry_after")):
        if isinstance(container, dict):
            text = json.dumps(container, sort_keys=True)
            match = re.search(r'"quantization_level"\s*:\s*"([^"]+)"', text)
            if match:
                return match.group(1)
    source_config = trial.get("source_config")
    explicit_sources = (
        model,
        trial.get("model_name"),
        trial.get("model"),
        trial.get("model_path"),
        trial.get("weight_path"),
        trial.get("weights"),
        trial.get("gguf"),
        source_config,
    )
    for value in explicit_sources:
        quant = _explicit_quant(value)
        if quant:
            return quant
    config = _load_json(Path(source_config)) if isinstance(source_config, str) else None
    if isinstance(config, dict):
        for value in _config_strings(config):
            quant = _explicit_quant(value)
            if quant:
                return quant

    # Historical rows sometimes record only the architecture.  These fallbacks
    # apply only after every available model/weight/config string lacked a quant label.
    haystack = " ".join(str(x) for x in (model, trial.get("model"), source_config)).lower()
    if "gpt-oss" in haystack:
        return "MXFP4"
    if "qwen3.6" in haystack or "qwen36" in haystack or "qwen122" in haystack or "122b" in haystack:
        return "UD-Q4_K_XL"
    if "flash" in haystack:
        return "IQ3_S"
    return None


def _kv_cache(trial: Mapping[str, Any]) -> str | None:
    cache = trial.get("cache") or trial.get("kv_cache")
    if cache is not None:
        return str(cache)
    model = str(trial.get("model_name") or trial.get("model") or "").lower()
    source = str(trial.get("source_config") or "").lower()
    if trial.get("runtime_backend") == "llama" and ("gpt-oss-20b" in model or "gpt20b-llama" in source):
        return "f16"
    return None


def _classify(row: Mapping[str, Any]) -> str:
    error = row.get("error")
    if error:
        text = json.dumps(error).lower()
        return "timeout" if "timeout" in text else "request_failure"
    response = row.get("response") if isinstance(row.get("response"), dict) else {}
    evaluation = row.get("evaluation") if isinstance(row.get("evaluation"), dict) else {}
    if response.get("truncated") is True or evaluation.get("truncated") is True:
        return "truncated"
    if response.get("stream_complete") is False:
        return "incomplete_stream"
    objective_outcome = row.get("objective_outcome")
    if (
        str(row.get("suite") or "").startswith("specialist_")
        and isinstance(objective_outcome, str)
        and objective_outcome
    ):
        return objective_outcome
    if (
        evaluation.get("validation") == "rejected"
        and evaluation.get("detail") == "private or dunder attribute access is not allowed"
    ):
        return "evaluator_rejected"
    return "passed" if evaluation.get("passed") is True else "wrong_answer"


def _json_format_audit(case_id: str, response_text: Any) -> dict[str, bool | None]:
    """Audit saved final-answer formatting without evaluating its meaning."""
    if case_id not in JSON_FORMAT_CASE_IDS or not isinstance(response_text, str):
        return {"raw_json_object": None, "outer_code_fence": None, "empty_answer": None}
    candidate = response_text.strip()
    empty = not candidate
    raw_object = False
    if not empty:
        try:
            raw_object = isinstance(json.loads(candidate), dict)
        except (json.JSONDecodeError, TypeError):
            pass
    return {
        "raw_json_object": raw_object,
        "outer_code_fence": bool(_OUTER_CODE_FENCE_RE.fullmatch(response_text)) if not empty else False,
        "empty_answer": empty,
    }


def _diagnostics(results_dir: Path) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    hard_tasks: dict[str, dict[str, Any]] = {}
    report = _load_json(results_dir / "hard-round-report.json")
    if isinstance(report, dict):
        for group in report.get("groups", []):
            if not isinstance(group, dict):
                continue
            for task in group.get("tasks", []):
                if isinstance(task, dict) and isinstance(task.get("call_id"), str):
                    hard_tasks[task["call_id"]] = {
                        key: task.get(key) for key in (
                            "hidden_test_diagnostics", "rescored_outcome", "rescored_cause"
                        )
                    }
    formats: dict[str, dict[str, Any]] = {}
    audit = _load_json(results_dir / "hard-json-format-audit.json")
    if isinstance(audit, dict):
        for item in audit.get("rows", []):
            if isinstance(item, dict) and isinstance(item.get("call_id"), str):
                formats[item["call_id"]] = item
    return hard_tasks, formats


def _timing_exclusions(results_dir: Path) -> tuple[dict[tuple[str, str | None], str], list[Path]]:
    result: dict[tuple[str, str | None], str] = {}
    sources: list[Path] = []
    for path in sorted(results_dir.glob("*-annotation.json")):
        annotation = _load_json(path)
        if not isinstance(annotation, dict) or annotation.get("timing_valid_for_speed_comparison") is not False:
            continue
        target = annotation.get("result_expected")
        if not isinstance(target, str):
            continue
        resolved = _absolute(target)
        if resolved is None:
            continue
        sources.append(path)
        reason = str(annotation.get("reason") or "timing annotation excludes speed comparison")
        if "call_ids" not in annotation:
            result[(resolved, None)] = reason
            continue
        call_ids = annotation.get("call_ids")
        if not isinstance(call_ids, list) or not call_ids or any(
            not isinstance(call_id, str) or not call_id.strip() for call_id in call_ids
        ):
            continue
        for call_id in call_ids:
            result[(resolved, call_id)] = reason
    return result, sources


def _diagnostic_variant_identity(row: Mapping[str, Any]) -> dict[str, Any]:
    variant = row.get("diagnostic_variant")
    domain_changed = row.get("diagnostic_domain_changed") is True
    algorithm_guidance = row.get("diagnostic_algorithm_guidance") is True
    if (not isinstance(variant, str) or variant == "original") and not (
        domain_changed or algorithm_guidance
    ):
        return {}
    return {
        key: row.get(key)
        for key in (
            "parent_case_id", "diagnostic_variant", "diagnostic_domain_changed",
            "diagnostic_algorithm_guidance",
        )
        if key in row
    }


def _underlying_task_identity(row: Mapping[str, Any]) -> tuple[Any, ...]:
    parent_case_id = row.get("parent_case_id")
    if isinstance(parent_case_id, str) and parent_case_id:
        return ("parent_case_id", parent_case_id)
    return (
        "case", row.get("case_id"), row.get("case_version"), row.get("prompt_sha256"),
    )


def _profile_parts(row: Mapping[str, Any]) -> dict[str, Any]:
    result = {
        key: row.get(key) for key in (
            "model", "weight_quant", "draft_weight_quant", "kv_cache", "runtime", "backend", "context_max_tokens",
            "thinking", "reasoning_budget_tokens_requested", "reasoning_budget_tokens_config_default",
            "strata_pcie_frac", "strata_aux_cpus", "chat_template_thinking", "temperature",
            "output_token_limit",
        )
    } | {
        "engine_environment": row.get("engine_environment") or {},
        "runtime_options": row.get("runtime_options") or {},
        "sampling_config": row.get("sampling_config") or {},
        "source_config": row.get("source_config"),
    }
    # Specialist profile names are explicit experimental controls.  Add the
    # field only for those rows so legacy profile IDs remain byte-for-byte
    # stable when specialist support is enabled.
    if row.get("specialist_profile") is not None:
        result["specialist_profile"] = row.get("specialist_profile")
    if "request_timeout_seconds" in row:
        result["request_timeout_seconds"] = row.get("request_timeout_seconds")
    if "native_thinking_budget_tokens_requested" in row:
        result["native_thinking_budget_tokens_requested"] = row.get(
            "native_thinking_budget_tokens_requested"
        )
    if "ledger_json_schema_sha256" in row:
        result["ledger_json_schema_sha256"] = row.get("ledger_json_schema_sha256")
    result.update(_diagnostic_variant_identity(row))
    return result


def _match_parts(row: Mapping[str, Any]) -> dict[str, Any]:
    result = {
        key: row.get(key) for key in (
            "suite", "case_id", "case_version", "prompt_sha256", "context_max_tokens",
            "output_token_limit", "temperature",
        )
    }
    result.update(_diagnostic_variant_identity(row))
    return result


def normalize_request(
    raw: Mapping[str, Any], source: Path, line: int, trial: Mapping[str, Any],
    hidden: Mapping[str, Mapping[str, Any]], formats: Mapping[str, Mapping[str, Any]],
    timing_exclusions: Mapping[tuple[str, str | None], str],
) -> dict[str, Any]:
    request = raw.get("request") if isinstance(raw.get("request"), dict) else {}
    response = raw.get("response") if isinstance(raw.get("response"), dict) else {}
    evaluation = raw.get("evaluation") if isinstance(raw.get("evaluation"), dict) else {}
    response_metrics = raw.get("response_metrics") if isinstance(raw.get("response_metrics"), dict) else {}
    visible_text = raw.get("visible_text") if isinstance(raw.get("visible_text"), dict) else {}
    eligibility = raw.get("eligibility") if isinstance(raw.get("eligibility"), dict) else {}
    timing = response.get("timing") if isinstance(response.get("timing"), dict) else {}
    throughput = response.get("throughput") if isinstance(response.get("throughput"), dict) else {}
    native = response.get("native_ollama") if isinstance(response.get("native_ollama"), dict) else {}
    usage = response.get("openai_usage") if isinstance(response.get("openai_usage"), dict) else {}
    usage_detail = usage.get("prompt_tokens_details") if isinstance(usage.get("prompt_tokens_details"), dict) else {}
    native_timing = response.get("openai_native_timings") if isinstance(response.get("openai_native_timings"), dict) else {}
    config = _config_metadata(trial)
    draft_weight_quant = _explicit_quant(config["runtime_options"].get("spec_draft_model"))
    model = request.get("model") if isinstance(request.get("model"), str) else trial.get("model_name")
    call_id = str(raw.get("call_id"))
    hard_diag = hidden.get(call_id, {})
    diag = hard_diag.get("hidden_test_diagnostics") if isinstance(hard_diag.get("hidden_test_diagnostics"), dict) else {}
    outcome = _classify(raw)
    suite = str(raw.get("suite") or "")
    specialist = suite.startswith("specialist_")
    case_group = raw.get("case_group") if isinstance(raw.get("case_group"), str) else (
        suite.removeprefix("specialist_") if specialist else None
    )
    error = raw.get("error")
    error_type = error.get("type") if isinstance(error, dict) else (type(error).__name__ if error else None)
    error_message = error.get("message") if isinstance(error, dict) else error
    source_abs = str(source.resolve())
    exclusions: list[str] = []
    quality = outcome in ("passed", "wrong_answer")
    speed = outcome not in ("timeout", "request_failure", "incomplete_stream")
    latency = speed
    if specialist:
        if isinstance(eligibility.get("speed_eligible"), bool):
            speed = eligibility["speed_eligible"]
            latency = speed
        quality = bool(
            case_group == "code"
            and eligibility.get("coding_objective_eligible") is True
            and outcome in ("passed", "wrong_answer")
        )
    case_id = str(raw.get("case_id"))
    audit = (
        formats[call_id]
        if call_id in formats
        else _json_format_audit(case_id, response.get("response_text"))
    )
    case_version = raw.get("case_version")
    output_limit = _number(request.get("output_token_limit"))
    if case_id == "quick_generation" and output_limit == 224:
        quality = False
        latency = False
        exclusions.append("224-token quick_generation is a throughput probe, not a quality or answer-latency task")
    if case_id == "quality_constraints" and case_version == AMBIGUOUS_CONSTRAINT_VERSION:
        quality = False
        exclusions.append("ambiguous constraints V1 prompt excluded from quality comparison")
    timing_reason = timing_exclusions.get((source_abs, call_id))
    if timing_reason is None:
        timing_reason = timing_exclusions.get((source_abs, None))
    if timing_reason:
        speed = False
        latency = False
        exclusions.append(timing_reason)
    endpoint = _safe_endpoint(request.get("endpoint"))
    if isinstance(endpoint, str) and ":11436" in endpoint:
        latency = False
        exclusions.append("shared original-service wall time may include queueing or concurrent setup; native server rates remain separately reported")
    if outcome in ("truncated", "incomplete_stream", "timeout", "request_failure"):
        quality = False
        exclusions.append(f"{outcome} is not a completed quality result")
    if outcome == "evaluator_rejected":
        exclusions.append("restricted evaluator did not execute private/dunder attribute access; semantic correctness is unverified")
    if specialist and case_group == "prose":
        quality = False
        exclusions.append("mechanical prose constraints are not an overall prose-quality or factual-correctness score")
    recorded_pass = (
        evaluation.get("passed")
        if isinstance(evaluation.get("passed"), bool) and not (specialist and case_group == "prose")
        else None
    )
    mechanical_constraints_passed = (
        evaluation.get("passed")
        if specialist and case_group == "prose" and isinstance(evaluation.get("passed"), bool)
        else None
    )
    rescored_outcome = hard_diag.get("rescored_outcome")
    comparison_pass = (rescored_outcome == "passed") if isinstance(rescored_outcome, str) else recorded_pass
    if outcome == "evaluator_rejected" or (specialist and case_group == "prose"):
        comparison_pass = None
    hidden_count = _number(diag.get("hidden_test_count"))
    if hidden_count is None:
        metadata = raw.get("case_metadata") if isinstance(raw.get("case_metadata"), dict) else {}
        hidden_count = _number(metadata.get("hidden_test_count"))
    hidden_passed = _number(diag.get("hidden_tests_passed"))
    if hidden_passed is None:
        hidden_passed = _number(evaluation.get("hidden_tests_passed"))
    if hidden_passed is None:
        hidden_passed = _number(evaluation.get("tests_passed"))
    if hidden_passed is None and recorded_pass is True and hidden_count:
        hidden_passed = hidden_count
    requested_budget = _number(request.get("reasoning_budget_tokens"))
    native_thinking_budget = _number(request.get("native_thinking_budget_tokens_requested"))
    config_budget = config["reasoning_budget_tokens"]
    native_draft_n = _number(native_timing.get("draft_n"))
    native_draft_accepted = _number(native_timing.get("draft_n_accepted"))
    input_tokens = _number(response.get("actual_prompt_tokens"))
    if input_tokens is None:
        input_tokens = _number(response_metrics.get("input_tokens_reported"))
    completion_tokens = _number(response.get("actual_completion_tokens"))
    if completion_tokens is None:
        completion_tokens = _number(response_metrics.get("output_tokens_reported"))
    reasoning_chars = _number(response.get("reasoning_chars"))
    if reasoning_chars is None:
        reasoning_chars = _number(response_metrics.get("reasoning_chars_observed"))
    reasoning_tokens = _number(response_metrics.get("reasoning_tokens_reported"))
    native_prompt_n = _number(native_timing.get("prompt_n"))
    if native_prompt_n is None:
        native_prompt_n = _number(response_metrics.get("native_prompt_tokens"))
    native_cache_n = _number(native_timing.get("cache_n"))
    if native_cache_n is None:
        native_cache_n = _number(response_metrics.get("native_cached_prompt_tokens"))
    native_decode_n = _number(native_timing.get("predicted_n"))
    if native_decode_n is None:
        native_decode_n = _number(response_metrics.get("native_decode_tokens"))
    native_decode_seconds = (
        float(native["eval_duration"]) / 1_000_000_000
        if _number(native.get("eval_duration")) is not None
        else (float(native_timing["predicted_ms"]) / 1000 if _number(native_timing.get("predicted_ms")) is not None else None)
    )
    if native_decode_seconds is None:
        native_decode_seconds = _number(response_metrics.get("native_decode_seconds"))
    native_load_seconds = (
        float(native["load_duration"]) / 1_000_000_000
        if _number(native.get("load_duration")) is not None
        else None
    )
    if native_load_seconds is None:
        native_load_seconds = _number(response_metrics.get("load_seconds_reported"))
    wall_seconds = _number(timing.get("wall_seconds"))
    if wall_seconds is None:
        wall_seconds = _number(response_metrics.get("wall_seconds"))
    ttft_reasoning = _number(timing.get("ttft_reasoning_seconds"))
    if ttft_reasoning is None:
        ttft_reasoning = _number(response_metrics.get("ttft_reasoning_seconds"))
    ttft_final = _number(timing.get("ttft_final_content_seconds"))
    if ttft_final is None:
        ttft_final = _number(response_metrics.get("ttft_visible_seconds"))
    native_prompt_tps = _number(throughput.get("prompt_tokens_per_second_native"))
    if native_prompt_tps is None:
        native_prompt_tps = _number(response_metrics.get("native_prompt_rate_tokens_per_second"))
    native_decode_tps = _number(throughput.get("completion_tokens_per_second_native"))
    if native_decode_tps is None:
        native_decode_tps = _number(response_metrics.get("native_decode_rate_tokens_per_second"))
    wall_decode_tps = _number(throughput.get("completion_tokens_per_second_wall"))
    if wall_decode_tps is None:
        wall_decode_tps = _number(response_metrics.get("wall_decode_rate_tokens_per_second"))
    requested_reasoning_controls = (
        request.get("reasoning_controls_requested")
        if isinstance(request.get("reasoning_controls_requested"), dict)
        else None
    )
    diagnostic_metadata: dict[str, Any] = {}
    for key in ("parent_case_id", "diagnostic_variant"):
        value = request.get(key)
        if isinstance(value, str) and value:
            diagnostic_metadata[key] = value
    for key in ("diagnostic_domain_changed", "diagnostic_algorithm_guidance"):
        value = request.get(key)
        if isinstance(value, bool):
            diagnostic_metadata[key] = value
    sampler_overrides = (
        request.get("sampler_overrides")
        if isinstance(request.get("sampler_overrides"), dict)
        else None
    )
    sampling_config = config["sampling"]
    if sampler_overrides is not None:
        # Request-scoped recipe probes must never collapse into a server-config
        # profile that happened to share temperature/context controls.
        sampling_config = dict(sampling_config)
        sampling_config["request_overrides"] = dict(sampler_overrides)
        sampling_config["recipe_label"] = request.get("sampler_recipe_label")
        sampling_config["recipe_sha256"] = request.get("sampler_recipe_sha256")
    row = {
        "row_type": "request",
        "row_id": call_id,
        "call_id": call_id,
        "run_id": raw.get("run_id"),
        "source_file": _relative(source),
        "source_line": line,
        "started_at": raw.get("started_at"),
        "finished_at": raw.get("finished_at"),
        "model": model,
        "weight_quant": _weight_quant(str(model) if model else None, raw, trial),
        "draft_weight_quant": draft_weight_quant,
        "kv_cache": _kv_cache(trial),
        "runtime": _runtime_name(trial, str(model) if model else None),
        "backend": request.get("backend"),
        "endpoint": endpoint,
        "suite": raw.get("suite"),
        "case_group": case_group,
        "case_id": case_id,
        "case_version": case_version,
        "case_source_sha256": raw.get("case_source_sha256"),
        "prompt_sha256": request.get("prompt_sha256"),
        "repeat_index": _number(raw.get("repeat_index")),
        "thinking": request.get("thinking"),
        # Keep the compatibility field as the effective request-or-config value,
        # while retaining which layer supplied it for reproducible comparisons.
        "reasoning_budget_tokens": requested_budget if requested_budget is not None else config_budget,
        "reasoning_budget_tokens_requested": requested_budget,
        "reasoning_budget_tokens_config_default": config_budget,
        "strata_pcie_frac": _number(request.get("strata_pcie_frac")),
        "strata_aux_cpus": _number(request.get("strata_aux_cpus")),
        "chat_template_thinking": (
            request.get("chat_template_thinking")
            if request.get("chat_template_thinking") in ("on", "off")
            else None
        ),
        "temperature": _number(request.get("temperature")),
        "sampling_seed": _number(request.get("seed")),
        "context_max_tokens": _number(request.get("context_max_tokens")),
        "context_mode": request.get("context_mode"),
        "output_token_limit": output_limit,
        "api_input_tokens": input_tokens,
        "ollama_prompt_eval_count": _number(native.get("prompt_eval_count")),
        "openai_prompt_tokens": _number(usage.get("prompt_tokens")),
        "native_prompt_n": native_prompt_n,
        "native_cache_n": native_cache_n,
        "api_cached_prompt_tokens": _number(usage_detail.get("cached_tokens")),
        "completion_tokens": completion_tokens,
        "ollama_eval_count": _number(native.get("eval_count")),
        "native_decode_n": native_decode_n,
        "native_draft_n": native_draft_n,
        "native_draft_n_accepted": native_draft_accepted,
        "native_draft_acceptance_rate": (
            float(native_draft_accepted) / float(native_draft_n)
            if native_draft_n is not None and native_draft_n > 0 and native_draft_accepted is not None
            else None
        ),
        "reasoning_chars": reasoning_chars,
        "reasoning_tokens": reasoning_tokens,
        "visible_answer_tokens": None,
        "native_prompt_seconds": (
            float(native["prompt_eval_duration"]) / 1_000_000_000
            if _number(native.get("prompt_eval_duration")) is not None
            else (float(native_timing["prompt_ms"]) / 1000 if _number(native_timing.get("prompt_ms")) is not None else None)
        ),
        "native_decode_seconds": native_decode_seconds,
        "native_load_seconds": native_load_seconds,
        "native_prompt_tps": native_prompt_tps,
        "native_decode_tps": native_decode_tps,
        "wall_decode_tps": wall_decode_tps,
        "wall_seconds": wall_seconds,
        "ttft_reasoning_seconds": ttft_reasoning,
        "ttft_final_seconds": ttft_final,
        "finish_reason": response.get("done_reason"),
        "stream_complete": response.get("stream_complete"),
        "truncated": response.get("truncated"),
        "outcome": outcome,
        "objective_outcome": raw.get("objective_outcome"),
        "content_passed": recorded_pass,
        "mechanical_constraints_passed": mechanical_constraints_passed,
        "objective_count": _number(evaluation.get("objective_count")),
        "objectives_passed": _number(evaluation.get("objectives_passed")),
        "assertion_count": _number(evaluation.get("assertion_count")),
        "assertions_passed": _number(evaluation.get("assertions_passed")),
        "semantic_behavior_passed": (
            evaluation.get("semantic_passed")
            if isinstance(evaluation.get("semantic_passed"), bool)
            else None
        ),
        "source_edit_fidelity_passed": (
            evaluation.get("fidelity_passed")
            if isinstance(evaluation.get("fidelity_passed"), bool)
            else None
        ),
        "comparison_content_passed": comparison_pass,
        "rescored_outcome": rescored_outcome,
        "rescored_cause": hard_diag.get("rescored_cause"),
        "evaluation_detail": _safe_message(evaluation.get("detail")),
        "request_error_type": error_type,
        "request_error_message": _safe_message(error_message),
        "hidden_tests_passed": hidden_passed,
        "hidden_test_count": hidden_count,
        "hidden_failure_labels": [x.get("label") for x in diag.get("failed_tests", []) if isinstance(x, dict)],
        "hidden_failure_categories": [x.get("category") for x in diag.get("failed_tests", []) if isinstance(x, dict)],
        "raw_json_object": audit.get("raw_json_object") if isinstance(audit.get("raw_json_object"), bool) else None,
        "outer_code_fence": audit.get("outer_code_fence") if isinstance(audit.get("outer_code_fence"), bool) else None,
        "empty_answer": audit.get("empty_answer") if isinstance(audit.get("empty_answer"), bool) else None,
        "quality_eligible": quality,
        "speed_eligible": speed,
        "latency_eligible": latency,
        "coding_objective_eligible": eligibility.get("coding_objective_eligible") if specialist else None,
        "prose_constraint_eligible": eligibility.get("prose_constraint_eligible") if specialist else None,
        "blinded_prose_review_eligible": eligibility.get("blinded_prose_review_eligible") if specialist else None,
        "reasoning_metric_eligible": eligibility.get("reasoning_metric_eligible") if specialist else None,
        "exclusion_reasons": exclusions,
        "engine_environment": trial.get("engine_environment") if isinstance(trial.get("engine_environment"), dict) else {},
        "runtime_options": config["runtime_options"],
        "ignored_runtime_options": config["ignored_runtime_options"],
        "sampling_config": sampling_config,
        "source_config": _relative(Path(trial["source_config"])) if isinstance(trial.get("source_config"), str) else None,
        "private_config": _relative(Path(trial["private_config"])) if isinstance(trial.get("private_config"), str) else None,
        "specialist_profile": raw.get("profile") or request.get("profile"),
        "actual_context_max_tokens": (
            _number(trial.get("actual_context_max_tokens"))
            if sampler_overrides is None
            else (
                _number(trial.get("actual_context_max_tokens"))
                or _number(trial.get("context"))
            )
        ),
        "study_tag": trial.get("study_tag"),
        "study_status": trial.get("study_status"),
        "study_receipt": _relative(Path(trial["study_receipt"])) if isinstance(trial.get("study_receipt"), str) else None,
        "model_session_run_id": trial.get("model_session_run_id"),
        "session_started_at": trial.get("session_started_at"),
        "server_ready_at": trial.get("server_ready_at"),
        "holding_started_at": trial.get("holding_started_at"),
        "startup_seconds": _number(trial.get("initial_load_seconds")),
        "session_hold_seconds": _number(trial.get("session_hold_seconds")),
        "source_config_sha256": trial.get("source_config_sha256"),
        "capability_note": request.get("capability_note"),
        "reasoning_controls_requested": requested_reasoning_controls,
        "reasoning_observed": {
            "reasoning_tokens_reported": reasoning_tokens,
            "reasoning_chars_observed": reasoning_chars,
            "reasoning_stream_observed": response_metrics.get("reasoning_stream_observed"),
        } if specialist else None,
        "visible_text_path": _relative(Path(visible_text["path"])) if isinstance(visible_text.get("path"), str) else None,
        "visible_text_sha256": visible_text.get("sha256"),
        "visible_text_chars": _number(visible_text.get("chars")),
        "visible_text_bytes": _number(visible_text.get("bytes")),
        "wrapper_wait_or_startup_included": False,
    }
    # Specialist rows record the actual per-request deadline. Keep this field
    # absent from legacy rows so their normalized schema and profile IDs remain
    # byte-for-byte stable.
    request_timeout = _number(request.get("timeout_seconds"))
    if (specialist or sampler_overrides is not None) and request_timeout is not None:
        row["request_timeout_seconds"] = request_timeout
    if sampler_overrides is not None:
        # Keep these optional so every pre-probe normalized row and profile ID
        # remains unchanged.
        row.update({
            "sampler_overrides": dict(sampler_overrides),
            "sampler_recipe_label": request.get("sampler_recipe_label"),
            "sampler_recipe_sha256": request.get("sampler_recipe_sha256"),
            "sampler_recipe_sources": (
                list(request["sampler_recipe_sources"])
                if isinstance(request.get("sampler_recipe_sources"), list)
                else None
            ),
            "sampler_recipe_claim": request.get("sampler_recipe_claim"),
            "sampler_application_evidence": request.get("sampler_application_evidence"),
        })
    if "native_thinking_budget_tokens_requested" in request:
        # This is llama.cpp's per-chat payload control. Keep it separate from
        # Strata's reasoning_budget_tokens and from template reasoning effort,
        # and omit it entirely from legacy rows to preserve their profile IDs.
        row.update({
            "native_thinking_budget_tokens_requested": native_thinking_budget,
            "native_thinking_budget_application_evidence": request.get(
                "native_thinking_budget_application_evidence"
            ),
        })
    if "ledger_json_schema_sha256" in request:
        # The schema content is bound by this hash. Path and request evidence
        # remain visible, while profile identity depends on content rather than
        # where the same schema happened to be stored.
        row.update({
            "ledger_json_schema_path": request.get("ledger_json_schema_path"),
            "ledger_json_schema_sha256": request.get("ledger_json_schema_sha256"),
            "ledger_json_schema_requested": (
                dict(request["ledger_json_schema_requested"])
                if isinstance(request.get("ledger_json_schema_requested"), dict)
                else None
            ),
        })
    if config["requested_native_runtime_controls"]:
        # Additive evidence only: profile/match identity remains defined by the
        # legacy control fields above. Values describe requested native argv,
        # never measured placement or cache residency.
        row["requested_native_runtime_controls"] = dict(
            config["requested_native_runtime_controls"]
        )
    row.update(diagnostic_metadata)
    row["profile_id"] = _stable_id(_profile_parts(row))
    row["match_id"] = _stable_id(_match_parts(row))
    return row


def _health_rows(results_dir: Path, events: Iterable[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[Path]]:
    event_by_result: dict[str, Mapping[str, Any]] = {}
    trial_by_run: dict[str, Mapping[str, Any]] = {}
    for event in events:
        run_id = event.get("run_id") or event.get("session_id")
        if event.get("event") in ("trial_started", "trial_start", "session_started") and isinstance(run_id, str):
            trial_by_run[run_id] = event
        if event.get("event") == "health_complete":
            path = _absolute(event.get("result"))
            if path:
                event_by_result[path] = event
    rows: list[dict[str, Any]] = []
    sources: list[Path] = []
    candidates = set(results_dir.glob("*-health.json"))
    if (results_dir / "initial-probe.json").is_file():
        candidates.add(results_dir / "initial-probe.json")
    for path in sorted(candidates):
        value = _load_json(path)
        if not isinstance(value, dict):
            continue
        sources.append(path)
        event = event_by_result.get(str(path.resolve()), {})
        identity = event.get("run_id") or event.get("session_id")
        trial = trial_by_run.get(str(identity), {}) or _legacy_trial(path)
        response = value.get("response")
        recorded_outcome = value.get("outcome")
        if not isinstance(recorded_outcome, str):
            recorded_outcome = "passed" if value.get("passed") is True else "setup_error"
        note_text = "; ".join(str(item) for item in value.get("notes", [])) if isinstance(value.get("notes"), list) else None
        health_detail = value.get("server_error") or value.get("error") or note_text
        rows.append({
            "row_type": "health",
            "row_id": "health-" + _stable_id([_relative(path), value.get("started_at")]),
            "source_file": _relative(path), "source_line": 1,
            "started_at": value.get("started_at"), "model": value.get("model") or trial.get("model_name"),
            "weight_quant": _weight_quant(str(trial.get("model_name", "")), {}, trial),
            "kv_cache": _kv_cache(trial), "runtime": _runtime_name(trial, str(trial.get("model_name", ""))),
            "backend": "health", "suite": "health", "case_id": "health_probe",
            "context_max_tokens": _number(value.get("context")), "thinking": value.get("thinking"),
            "completion_tokens": _number(value.get("eval_count")), "api_input_tokens": _number(value.get("prompt_eval_count")),
            "native_decode_tps": _number(value.get("decode_tokens_per_second")),
            "wall_seconds": _number(value.get("wall_seconds")), "ttft_final_seconds": _number(value.get("first_content_seconds")),
            "outcome": recorded_outcome,
            "content_passed": value.get("passed") if isinstance(value.get("passed"), bool) else None,
            "evaluation_detail": _safe_message(health_detail),
            "response_bytes": _number(value.get("response_bytes")),
            "response_present": bool(response), "quality_eligible": False, "speed_eligible": False,
            "latency_eligible": False, "exclusion_reasons": ["health/setup probe is not a benchmark task"],
        })
    return rows, sources


def _parse_time(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _setup_rows(events: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Summarize each owned runtime trial without treating task failure as setup failure."""
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for event in events:
        identity = event.get("run_id") or event.get("session_id")
        if isinstance(identity, str):
            grouped[identity].append(event)
    rows: list[dict[str, Any]] = []
    for run_id, items in sorted(grouped.items()):
        start = next((x for x in items if x.get("event") in ("trial_started", "trial_start", "session_started")), None)
        if start is None:
            continue
        ready = next((x for x in items if x.get("event") == "server_ready"), None)
        terminal = next((x for x in reversed(items) if x.get("event") in ("trial_complete", "trial_failed", "server_failed", "startup_failed", "session_complete", "session_failed")), None)
        if ready is not None:
            status = "ready"
        elif terminal is not None:
            status = "setup_error"
        else:
            status = "in_progress"
        started = _parse_time(start.get("at"))
        readied = _parse_time(ready.get("at")) if ready else None
        startup_seconds = (readied - started).total_seconds() if started and readied else None
        error_detail = None
        if status == "setup_error" and terminal:
            error_detail = terminal.get("error") or terminal.get("message") or terminal.get("reason")
        model = start.get("model_name") or start.get("model")
        trial = dict(start)
        if ready is not None:
            trial["server_ready"] = ready
        config = _config_metadata(trial)
        row = {
            "row_type": "runtime_trial", "row_id": "trial-" + run_id,
            "source_file": start.get("_source_file"), "source_line": start.get("_source_line"),
            "started_at": start.get("at"), "model": model,
            "weight_quant": _weight_quant(str(model) if model else None, {}, trial),
            "kv_cache": _kv_cache(trial), "runtime": _runtime_name(trial, str(model) if model else None),
            "backend": "runtime_trial", "suite": "setup", "case_id": "server_startup",
            "context_max_tokens": _number(start.get("context")), "thinking": start.get("thinking"),
            "startup_seconds": startup_seconds, "wall_seconds": None, "outcome": status, "content_passed": None,
            "evaluation_detail": error_detail, "quality_eligible": False, "speed_eligible": False,
            "latency_eligible": False,
            "exclusion_reasons": ["runtime startup lifecycle is supplemental and not a scored model task"],
            "engine_environment": start.get("engine_environment") if isinstance(start.get("engine_environment"), dict) else {},
            "source_config": _relative(Path(start["source_config"])) if isinstance(start.get("source_config"), str) else None,
            "private_config": _relative(Path(start["private_config"])) if isinstance(start.get("private_config"), str) else None,
            "runtime_options": config["runtime_options"],
            "ignored_runtime_options": config["ignored_runtime_options"],
            "sampling_config": config["sampling"],
        }
        if config["requested_native_runtime_controls"]:
            row["requested_native_runtime_controls"] = dict(
                config["requested_native_runtime_controls"]
            )
        rows.append(row)
    return rows


def _cache_rows(results_dir: Path) -> tuple[list[dict[str, Any]], list[Path]]:
    path = results_dir / "expert-cache-summary.json"
    value = _load_json(path)
    if not isinstance(value, dict):
        return [], []
    method = value.get("method") if isinstance(value.get("method"), dict) else {}
    rows: list[dict[str, Any]] = []
    for item in value.get("rows", []):
        if not isinstance(item, dict):
            continue
        native = item.get("native_timing") if isinstance(item.get("native_timing"), dict) else {}
        rows.append({
            "row_type": "expert_cache", "row_id": "cache-" + _stable_id(item.get("tag")),
            "source_file": _relative(path), "source_line": 1, "model": method.get("model"),
            "weight_quant": "UD-Q4_K_XL", "kv_cache": None, "runtime": method.get("runtime"),
            "backend": "strata_native_cli", "suite": "expert_cache", "case_id": method.get("case"),
            "thinking": method.get("thinking"), "context_max_tokens": _number(item.get("context")),
            "api_input_tokens": _number(method.get("prompt_tokens")), "completion_tokens": _number(method.get("fixed_generation_tokens")),
            "native_prompt_tps": _number(native.get("prompt_tokens_per_second")),
            "native_decode_tps": _number(native.get("generated_tokens_per_second")),
            "outcome": item.get("status"), "content_passed": None,
            "quality_eligible": False, "speed_eligible": False, "latency_eligible": False,
            "cache_experiment_timing_valid": item.get("use_timing_comparison") is True,
            "exclusion_reasons": ["fixed-N expert-cache probe is not answer quality or answer latency"] + ([str(item.get("contention_note"))] if item.get("use_timing_comparison") is False else []),
            "expert_cache": {key: item.get(key) for key in ("slots_per_layer", "experts_per_layer", "expert_vram_gib", "experts_host_gib", "gpu_entry_fraction", "cpu_entry_fraction", "slot_replacements", "cpu_pool_accumulated_ms")},
        })
    return rows, [path]


def _diagnostic_rows(results_dir: Path) -> tuple[list[dict[str, Any]], list[Path]]:
    rows: list[dict[str, Any]] = []
    sources: list[Path] = []
    for path in sorted(results_dir.glob("*-diagnosis.json")):
        value = _load_json(path)
        if not isinstance(value, dict):
            continue
        sources.append(path)
        is_new_20b = path.name.startswith("native-new-cpu-")
        for operation, detail in sorted(value.items()):
            if not isinstance(detail, dict):
                continue
            status = _number(detail.get("status"))
            error = detail.get("error")
            if isinstance(error, str) and "timed out" in error.lower():
                outcome = "timeout"
            elif status == 200 and not error:
                outcome = "passed"
            else:
                outcome = "setup_error"
            rows.append({
                "row_type": "setup_diagnostic", "row_id": "diagnostic-" + _stable_id([_relative(path), operation]),
                "source_file": _relative(path), "source_line": 1,
                "model": "gpt-oss-20b:64k" if is_new_20b else None,
                "weight_quant": "MXFP4" if is_new_20b else None, "kv_cache": None,
                "runtime": "Ollama 0.40.6" if is_new_20b else None,
                "backend": "ollama_diagnostic", "suite": "setup", "case_id": f"api_{operation}",
                "outcome": outcome, "content_passed": None, "evaluation_detail": _safe_message(error),
                "quality_eligible": False, "speed_eligible": False, "latency_eligible": False,
                "exclusion_reasons": ["diagnostic API operation is supplemental and not a scored benchmark task"],
                "http_status": status,
            })
    return rows, sources


def _preferred_mtp_receipt(results_dir: Path) -> tuple[Path | None, dict[str, Any] | None]:
    for name in ("flash-next-mtp-artifact.json", "flash-next-mtp-candidate.json"):
        path = results_dir / name
        value = _load_json(path)
        if isinstance(value, dict):
            return path, value
    return None, None


def _not_run(results_dir: Path, observed_rows: Iterable[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[Path]]:
    notes: list[dict[str, Any]] = []
    sources: list[Path] = []
    observed = list(observed_rows)
    model_text = " ".join(str(row.get("model")).lower() for row in observed if row.get("model"))
    flash = _load_json(results_dir / "flash-next-artifact.json")
    if isinstance(flash, dict):
        path = results_dir / "flash-next-artifact.json"; sources.append(path)
        if "flash" not in model_text:
            notes.append({"model": flash.get("model"), "weight_quant": flash.get("quantization"), "status": flash.get("state"), "reason": "artifact preparation only; no request rows yet", "source_file": _relative(path)})
    gpt20b_path = results_dir / "gpt20b-llama-artifact.json"
    gpt20b = _load_json(gpt20b_path)
    if isinstance(gpt20b, dict):
        sources.append(gpt20b_path)
        native_gpt20b_seen = any(
            "gpt-oss-20b" in str(row.get("model", "")).lower()
            and "llama.cpp" in str(row.get("runtime", "")).lower()
            for row in observed
        )
        if not native_gpt20b_seen:
            verdict = gpt20b.get("verdict") if isinstance(gpt20b.get("verdict"), dict) else {}
            notes.append({
                "model": "GPT-OSS-20B", "weight_quant": "MXFP4",
                "runtime": "llama.cpp server (Ollama 0.40.6 bundle)",
                "status": gpt20b.get("state"),
                "reason": (
                    "Verified source weights and pinned Harmony template are recorded; "
                    f"runtime load validated={verdict.get('runtime_load_validated')!r}, "
                    f"reasoning-budget enforcement validated={verdict.get('reasoning_budget_runtime_validated')!r}; "
                    "no native llama request rows exist yet."
                ),
                "source_file": _relative(gpt20b_path),
            })
    eagle3_path = results_dir / "gpt20b-eagle3-derived-artifact.json"
    if not eagle3_path.is_file():
        eagle3_path = results_dir / "gpt20b-eagle3-artifact.json"
    eagle3 = _load_json(eagle3_path)
    if isinstance(eagle3, dict):
        sources.append(eagle3_path)
        eagle3_measured = any(
            (row.get("runtime_options") or {}).get("spec_type") == "draft-eagle3"
            and bool((row.get("runtime_options") or {}).get("spec_draft_model"))
            for row in observed if isinstance(row.get("runtime_options"), dict)
        )
        if not eagle3_measured:
            verdict = eagle3.get("verdict") if isinstance(eagle3.get("verdict"), dict) else {}
            draft = eagle3.get("gguf_artifact") if isinstance(eagle3.get("gguf_artifact"), dict) else {}
            notes.append({
                "model": "GPT-OSS-20B with EAGLE3 draft",
                "weight_quant": "MXFP4", "draft_weight_quant": draft.get("quantization"),
                "runtime": "llama.cpp server (Ollama 0.40.6 bundle)",
                "status": eagle3.get("state"),
                "reason": (
                    f"EAGLE3 artifact complete={verdict.get('artifact_complete')!r}, "
                    f"runtime load validated={verdict.get('runtime_load_validated')!r}, "
                    f"inference validated={verdict.get('inference_validated')!r}; "
                    "no actual draft-eagle3 request rows exist yet."
                ),
                "source_file": _relative(eagle3_path),
            })
    qwen122_path = results_dir / "qwen122-artifact.json"
    qwen122 = _load_json(qwen122_path)
    qwen122_layout_path = results_dir / "qwen122-layout.json"
    qwen122_layout = _load_json(qwen122_layout_path)
    if isinstance(qwen122, dict):
        sources.append(qwen122_path)
        if isinstance(qwen122_layout, dict):
            sources.append(qwen122_layout_path)
        if "122" not in model_text:
            model = qwen122.get("model") if isinstance(qwen122.get("model"), dict) else {}
            verdict = qwen122.get("verdict") if isinstance(qwen122.get("verdict"), dict) else {}
            layout_verified = isinstance(qwen122_layout, dict) and all(
                qwen122_layout.get(key) is True for key in (
                    "all_offsets_aligned", "all_tensor_directory_entries_validated",
                    "all_tensor_names_unique", "all_tensor_payload_bounds_validated",
                    "no_payload_overlaps",
                )
            )
            ready = verdict.get("artifact_complete") is True and layout_verified
            notes.append({
                "model": "Qwen3.5-122B-A10B", "weight_quant": model.get("quant"),
                "status": "artifact_complete_layout_verified_no_inference" if ready else "artifact_receipt_no_inference",
                "reason": (
                    "Recorded shards and tensor layout are verified; full-weight load and inference have not run."
                    if ready else "Artifact/layout receipts exist, but do not establish a full-weight load or inference result."
                ),
                "source_file": _relative(qwen122_path),
                "layout_source_file": _relative(qwen122_layout_path) if isinstance(qwen122_layout, dict) else None,
            })
    candidate = _load_json(results_dir / "qwen122-strata-candidate.json")
    if isinstance(candidate, dict):
        path = results_dir / "qwen122-strata-candidate.json"; sources.append(path)
        if "122" not in model_text:
            notes.append({"model": "Qwen3.5-122B-A10B", "weight_quant": ((candidate.get("artifact_evidence") or {}).get("quant")), "runtime": "Strata experimental patched 0.1.45", "status": candidate.get("status"), "reason": "isolated patched candidate; no inference request rows yet", "source_file": _relative(path), "patch_sha256": ((candidate.get("source") or {}).get("patch_sha256")), "binary_sha256": ((((candidate.get("validation") or {}).get("cuda_build") or {}).get("binary") or {}).get("sha256"))})
    kernel = _load_json(results_dir / "qwen36-t4-kernel-comparison.json")
    if isinstance(kernel, dict) and not kernel.get("flag_rows"):
        path = results_dir / "qwen36-t4-kernel-comparison.json"; sources.append(path)
        notes.append({"model": kernel.get("model"), "status": kernel.get("status"), "reason": "STRATA_SM75_PERF=1 A/B blocked before start; no flag result or speedup claim", "source_file": _relative(path)})
    mtp_path, mtp_receipt = _preferred_mtp_receipt(results_dir)
    if mtp_path is not None and mtp_receipt is not None:
        sources.append(mtp_path)
        mtp_measured = any(
            "flash" in str(row.get("model", "")).lower()
            and str((row.get("runtime_options") or {}).get("mtp", "0")) not in ("0", "None", "")
            for row in observed
        )
        if not mtp_measured:
            download = mtp_receipt.get("download") if isinstance(mtp_receipt.get("download"), dict) else {}
            rejection = mtp_receipt.get("automatic_approval_rejection") if isinstance(mtp_receipt.get("automatic_approval_rejection"), dict) else {}
            notes.append({
                "model": "Qwen3.8-Flash-Next optional MTP", "status": mtp_receipt.get("state"),
                "reason": rejection.get("reason") or download.get("reason") or "MTP preparation/fetch did not complete; no MTP inference measurement exists.",
                "source_file": _relative(mtp_path),
            })
    accuracy_path = results_dir / "flash-next-accuracy-reference.json"
    accuracy = _load_json(accuracy_path)
    if isinstance(accuracy, dict):
        sources.append(accuracy_path)
        if _number(accuracy.get("inference_attempts")) in (None, 0):
            notes.append({
                "model": accuracy.get("model") or "Qwen3.8-Flash-Next",
                "status": accuracy.get("state"),
                "reason": accuracy.get("rejection") or "accuracy reference was blocked before launch; no inference measurement exists",
                "source_file": _relative(accuracy_path),
                "requested_engine_environment": accuracy.get("requested_engine_environment"),
            })
    overlap_path = results_dir / "flash-next-adaptive-overlap-blocked.json"
    overlap = _load_json(overlap_path)
    if isinstance(overlap, dict):
        sources.append(overlap_path)
        overlap_measured = any(
            str((row.get("engine_environment") or {}).get("STRATA_ADAPT_OVERLAP")) == "1"
            for row in observed if isinstance(row.get("engine_environment"), dict)
        )
        if not overlap_measured and _number(overlap.get("inference_requests")) in (None, 0):
            notes.append({
                "model": "Qwen3.8-Flash-Next", "runtime": "Strata 0.1.45 pinned",
                "status": overlap.get("status"), "reason": overlap.get("reason"),
                "requested_engine_environment": overlap.get("requested_environment"),
                "retry_attempts": _number(overlap.get("retry_attempts")),
                "source_file": _relative(overlap_path),
            })
    q3_path = results_dir / "qwen36-q3-artifact.json"
    q3 = _load_json(q3_path)
    if isinstance(q3, dict):
        sources.append(q3_path)
        q3_seen = any(
            "q3_k" in str(row.get("weight_quant", "")).lower()
            or "q3_k" in str(row.get("model", "")).lower()
            for row in observed
        )
        if not q3_seen:
            model = q3.get("model") if isinstance(q3.get("model"), dict) else {}
            state = q3.get("state")
            notes.append({
                "model": "Qwen3.6-35B-A3B", "weight_quant": model.get("quant"), "status": state,
                "reason": f"Q3 artifact state is {state or 'unknown'}; no Q3 inference request rows exist yet.",
                "source_file": _relative(q3_path),
            })
    quant_group_path = results_dir / "qwen35-quant-placement-group-receipt.json"
    quant_group = _load_json(quant_group_path)
    if isinstance(quant_group, dict):
        sources.append(quant_group_path)
        blocker = quant_group.get("blocker") or {}
        for profile in quant_group.get("profiles", []):
            if not isinstance(profile, dict) or profile.get("status") != "not_run":
                continue
            notes.append({
                "model": "Qwen3.6-35B-A3B",
                "runtime": "llama.cpp server (Ollama 0.40.6 bundle)",
                "status": profile["status"],
                "profile_tag": profile.get("tag"),
                "benchmark_requests": profile.get("benchmark_requests"),
                "reason": profile.get("reason") or blocker.get("reason"),
                "blocked_action": blocker.get("action"),
                "source_file": _relative(quant_group_path),
            })
    deferral_path = results_dir / "specialists-gemma-user-deferral-v1.json"
    deferral = _load_json(deferral_path)
    if (
        isinstance(deferral, dict)
        and deferral.get("schema_version") == 1
        and deferral.get("kind") == "user_requested_model_deferral"
        and deferral.get("deferred_models") == ["gemma4-style12-q5_k_m"]
    ):
        sources.append(deferral_path)
        coverage = deferral.get("coverage")
        coverage = coverage if isinstance(coverage, dict) else {}
        notes.append({
            "model": "Gemma-4-12B-StyleTune", "weight_quant": "Q5_K_M",
            "status": "user_deferred_partial_results_preserved",
            "reason": (
                "The user deferred the remaining Gemma tests and accelerator probes. "
                "Completed request results remain in the report; the active hard request "
                "was interrupted at the user's request. Unexecuted tests have no grades. "
                "Revisit after other candidate results; no automatic resume."
            ),
            "coverage": {
                key: _number(coverage.get(key)) for key in (
                    "planned_primary_attempt_slots", "terminal_completed_attempts",
                    "active_request_interrupted_by_user", "remaining_unexecuted_primary_slots",
                )
            },
            "source_file": _relative(deferral_path),
        })
    return notes, sources


FLASH_CODER_MODEL_CARD = "https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF"
FLASH_CODER_QUANTIZATION_CAVEAT = (
    "IQ1_M is the file/package tag for effective compression after expert pruning, not a claim that "
    "the retained weights are 1-bit. The model card describes 256 of 512 experts retained with "
    "top-10 routing, mixed retained transformer weights at about 3.5 bpw, and 1.89 effective bits "
    "amortized over the original 125B transformer."
)


def _specialist_artifact_overview(results_dir: Path) -> list[dict[str, Any]]:
    """Normalize small specialist receipts without touching model payloads."""
    rows: list[dict[str, Any]] = []
    first_wave = {
        "lfm25-qad-q4_0-artifact.json": "LFM2.5-24B-A2B",
        "kat-v25-iq3_xxs-artifact.json": "KAT-Coder-V2.5-Dev",
        "gemma4-style12-q5_k_m-artifact.json": "Gemma-4-12B-StyleTune",
    }
    for filename, display_name in first_wave.items():
        path = results_dir / filename
        receipt = _load_json(path)
        if not isinstance(receipt, dict):
            continue
        model = receipt.get("model") if isinstance(receipt.get("model"), dict) else {}
        verdict = receipt.get("verdict") if isinstance(receipt.get("verdict"), dict) else {}
        runtime = receipt.get("runtime") if isinstance(receipt.get("runtime"), dict) else {}
        shards = receipt.get("shards") if isinstance(receipt.get("shards"), list) else []
        hashes = {
            str(item.get("name")): item.get("sha256")
            for item in shards if isinstance(item, dict) and item.get("name") and item.get("sha256")
        }
        size = sum(
            int(item["size"]) for item in shards
            if isinstance(item, dict) and isinstance(item.get("size"), int)
        )
        verified = bool(shards) and verdict.get("artifact_complete") is True \
            and verdict.get("full_sha256_verified") is True \
            and all(
                isinstance(item, dict)
                and item.get("verified") is True
                and item.get("full_sha256_verified") is True
                for item in shards
            )
        rows.append({
            "artifact_family": "specialist_first_wave",
            "model": model.get("alias") or display_name,
            "packing": model.get("quantization"),
            "weight_quant": model.get("quantization"),
            "status": receipt.get("state"),
            "source_repo": model.get("repo"),
            "source_revision": model.get("revision"),
            "recorded_weight_bytes": size if verified else None,
            "recorded_weight_sha256": hashes if verified else None,
            "verified": verified,
            "verification_scope": "full model-file SHA-256" if verified else "artifact receipt is not fully verified",
            "runtime": "llama.cpp server",
            "runtime_source_revision": runtime.get("llama_cpp_revision"),
            "runtime_load_validated": verdict.get("runtime_load_validated"),
            "all_gpu_fit_claimed": runtime.get("all_gpu_fit_claimed"),
            "artifact_role": "pinned model-file provenance; study receipts and request rows establish runtime placement and inference",
            "source_file": _relative(path),
            "source_receipt_sha256": _sha256(path),
        })

    bonsai_path = results_dir / "bonsai2-runtime-artifact.json"
    bonsai = _load_json(bonsai_path)
    if isinstance(bonsai, dict):
        verdict = bonsai.get("verdict") if isinstance(bonsai.get("verdict"), dict) else {}
        source = bonsai.get("source") if isinstance(bonsai.get("source"), dict) else {}
        runtime = bonsai.get("runtime") if isinstance(bonsai.get("runtime"), dict) else {}
        selected = bonsai.get("model") if isinstance(bonsai.get("model"), dict) else {}
        server = runtime.get("server") if isinstance(runtime.get("server"), dict) else {}
        cuda = runtime.get("cuda_backend") if isinstance(runtime.get("cuda_backend"), dict) else {}
        for shard in bonsai.get("shards", []):
            if not isinstance(shard, dict):
                continue
            verified = shard.get("verified") is True and verdict.get("model_artifact_complete") is True
            rows.append({
                "artifact_family": "specialist_bonsai2",
                "model": "Ternary-Bonsai-2-27B",
                "packing": next((
                    item.get("quantization") for item in bonsai.get("available_model_packings", [])
                    if isinstance(item, dict) and item.get("name") == shard.get("name")
                ), None),
                "weight_quant": next((
                    item.get("quantization") for item in bonsai.get("available_model_packings", [])
                    if isinstance(item, dict) and item.get("name") == shard.get("name")
                ), None),
                "selected_for_trial": shard.get("name") == selected.get("name"),
                "status": bonsai.get("state"),
                "source_repo": source.get("model_repo"),
                "source_revision": source.get("model_revision"),
                "recorded_weight_bytes": shard.get("size") if verified else None,
                "recorded_weight_sha256": shard.get("sha256") if verified else None,
                "verified": verified,
                "verification_scope": (
                    "full model-file SHA-256 plus static runtime artifact checks"
                    if verified and verdict.get("runtime_artifact_complete") is True
                    else "full model-file SHA-256" if verified
                    else "artifact receipt is not fully verified"
                ),
                "runtime": "PrismML llama.cpp fork, source-built CUDA sm_75",
                "runtime_source_revision": runtime.get("source_revision") or source.get("llama_revision"),
                "runtime_server_sha256": server.get("sha256"),
                "runtime_cuda_backend_sha256": cuda.get("sha256"),
                "runtime_artifact_complete": verdict.get("runtime_artifact_complete"),
                "runtime_load_validated": verdict.get("gpu_load_proven"),
                "static_tesla_t4_compatibility": verdict.get("static_tesla_t4_compatibility"),
                "all_gpu_fit_claimed": None,
                "artifact_role": "verified model and static runtime provenance; no GPU load, placement, inference, or performance claim",
                "source_file": _relative(bonsai_path),
                "source_receipt_sha256": _sha256(bonsai_path),
            })

    flash_path = results_dir / "flash-next-coder-artifact.json"
    preview_path = results_dir / "flash-next-coder-header-preview.json"
    flash = _load_json(flash_path)
    preview = _load_json(preview_path)
    preview = preview if isinstance(preview, dict) else {}
    if isinstance(flash, dict) or preview:
        receipt = flash if isinstance(flash, dict) else {}
        verdict = receipt.get("verdict") if isinstance(receipt.get("verdict"), dict) else {}
        model = receipt.get("model") if isinstance(receipt.get("model"), dict) else {}
        runtime = receipt.get("runtime") if isinstance(receipt.get("runtime"), dict) else {}
        engine = runtime.get("engine") if isinstance(runtime.get("engine"), dict) else {}
        shards = receipt.get("shards") if isinstance(receipt.get("shards"), list) else []
        hashes = {
            str(item.get("name")): item.get("sha256")
            for item in shards if isinstance(item, dict) and item.get("name") and item.get("sha256")
        }
        size = sum(
            int(item["size"]) for item in shards
            if isinstance(item, dict) and isinstance(item.get("size"), int)
        )
        verified = bool(shards) and verdict.get("artifact_complete") is True \
            and verdict.get("full_sha256_verified") is True \
            and all(
                isinstance(item, dict)
                and item.get("verified") is True
                and item.get("full_sha256_verified") is True
                for item in shards
            )
        transformer_elements = _number(preview.get("transformer_tensor_elements"))
        ple_elements = _number(preview.get("shared_ple_tensor_elements"))
        expected_flash_shards = {
            "Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf": {
                "size": 29_608_446_496,
                "sha256": "bfa3644db4adfff072fc946c20994f2909d2cf89879d83f80a4c811d4016c211",
            },
            "Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00002-of-00002.gguf": {
                "size": 28_800_138_432,
                "sha256": "316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113",
            },
        }
        received_flash_shards = {
            str(item.get("name")): {"size": item.get("size"), "sha256": item.get("sha256")}
            for item in shards if isinstance(item, dict) and item.get("name")
        }
        verified = verified \
            and received_flash_shards == expected_flash_shards \
            and verdict.get("shared_ple_identity_bound") is True \
            and verdict.get("pack_complete") is True
        rows.append({
            "artifact_family": "specialist_flash_next_coder",
            "model": model.get("model_name") or preview.get("model_name") or "Qwen3.8-Flash-Next GSQ-RCO Coder",
            "packing": model.get("quantization") or "IQ1_M",
            "weight_quant": model.get("quantization") or "IQ1_M",
            "quantization_caveat": FLASH_CODER_QUANTIZATION_CAVEAT,
            "status": receipt.get("state") or "header_preview_only_payload_unverified",
            "source_repo": model.get("repo") or "ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF",
            "source_revision": model.get("revision") or "bba88856f8c3c0c9e40e45f82d05e228b814ac43",
            "model_card": FLASH_CODER_MODEL_CARD,
            "expected_weight_bytes": sum(item["size"] for item in expected_flash_shards.values()),
            "expected_weight_sha256": {name: item["sha256"] for name, item in expected_flash_shards.items()},
            "recorded_weight_bytes": size if verified else None,
            "recorded_weight_sha256": hashes if verified else None,
            "verified": verified,
            "verification_scope": (
                "full two-shard SHA-256, shared-PLE identity binding, and prepared pack"
                if verified else "GGUF header preview only; payload full SHA-256 is not verified"
            ),
            "runtime": "Strata",
            "runtime_source_revision": runtime.get("strata_revision"),
            "runtime_server_sha256": engine.get("sha256"),
            "runtime_load_validated": verdict.get("runtime_load_validated"),
            "gpu_placement_validated": verdict.get("gpu_placement_validated"),
            "all_gpu_fit_claimed": None,
            "header_preview": {
                "source_file": _relative(preview_path) if preview else None,
                "payload_fully_verified": preview.get("payload_fully_verified") if preview else None,
                "chunk_zero_manifest_hash_matches": preview.get("chunk_zero_manifest_hash_matches") if preview else None,
                "blocks": preview.get("blocks"),
                "experts_retained": preview.get("experts"),
                "experts_used": preview.get("experts_used"),
                "transformer_tensor_elements": transformer_elements,
                "shared_ple_tensor_elements": ple_elements,
                "total_tensor_elements": (
                    int(transformer_elements) + int(ple_elements)
                    if transformer_elements is not None and ple_elements is not None else None
                ),
                "individual_iq1_m_tensor_observed_in_parsed_mixed_type_table": False if preview else None,
            },
            "compression_interpretation": {
                "experts_original": 512,
                "experts_retained": 256,
                "experts_active": 10,
                "retained_transformer_weight_bpw_approx": 3.5,
                "effective_bits_over_original_125b_transformer": 1.89,
            },
            "artifact_role": (
                "verified model/package provenance; study receipts and request rows establish runtime placement and inference"
                if verified else
                "preview/provenance only until full receipt verification; study receipts and request rows establish runtime placement and inference"
            ),
            "source_file": _relative(flash_path) if isinstance(flash, dict) else _relative(preview_path),
            "source_receipt_sha256": _sha256(flash_path) if isinstance(flash, dict) else _sha256(preview_path),
        })
    return rows


def _artifact_provenance(results_dir: Path) -> list[dict[str, Any]]:
    """Small recorded manifests only; model weight files are never read or hashed."""
    provenance: list[dict[str, Any]] = _specialist_artifact_overview(results_dir)
    qwen36 = _load_json(results_dir / "qwen36-artifact.json")
    if isinstance(qwen36, dict):
        provenance.append({
            "model": "Qwen3.6-35B-A3B", "weight_quant": "UD-Q4_K_XL",
            "source_revision": qwen36.get("revision"), "recorded_weight_sha256": qwen36.get("sha256"),
            "verified": qwen36.get("verified"), "source_file": "results/qwen36-artifact.json",
        })
    gpt20b_path = results_dir / "gpt20b-llama-artifact.json"
    gpt20b = _load_json(gpt20b_path)
    if isinstance(gpt20b, dict):
        model = gpt20b.get("model") if isinstance(gpt20b.get("model"), dict) else {}
        harmony = model.get("harmony") if isinstance(model.get("harmony"), dict) else {}
        verdict = gpt20b.get("verdict") if isinstance(gpt20b.get("verdict"), dict) else {}
        budget = gpt20b.get("reasoning_budget") if isinstance(gpt20b.get("reasoning_budget"), dict) else {}
        server_template = budget.get("direct_server_template") if isinstance(budget.get("direct_server_template"), dict) else {}
        llama_cpp = gpt20b.get("llama_cpp") if isinstance(gpt20b.get("llama_cpp"), dict) else {}
        shards = gpt20b.get("shards") if isinstance(gpt20b.get("shards"), list) else []
        shard_hashes = {
            str(item.get("name")): item.get("sha256")
            for item in shards if isinstance(item, dict) and item.get("name") and item.get("sha256")
        }
        weights_verified = (
            verdict.get("artifact_complete") is True
            and model.get("fresh_full_sha256_verified") is True
            and bool(shards)
            and len(shard_hashes) == len(shards)
            and all(item.get("verified") is True for item in shards if isinstance(item, dict))
        )
        provenance.append({
            "model": "GPT-OSS-20B", "weight_quant": "MXFP4",
            "runtime": "llama.cpp server (Ollama 0.40.6 bundle)",
            "status": gpt20b.get("state"), "recorded_weight_bytes": model.get("size"),
            "recorded_weight_sha256": shard_hashes, "verified": weights_verified,
            "harmony_template_sha256": harmony.get("template_sha256"),
            "llama_chat_template_sha256": server_template.get("sha256"),
            "runtime_source_revision": llama_cpp.get("upstream_revision"),
            "runtime_server_sha256": llama_cpp.get("server_sha256"),
            "runtime_load_validated": verdict.get("runtime_load_validated"),
            "reasoning_budget_runtime_validated": verdict.get("reasoning_budget_runtime_validated"),
            "artifact_role": "verified source weights and template/runtime compatibility provenance; benchmark request rows establish load and inference measurements",
            "source_file": _relative(gpt20b_path),
            "source_receipt_sha256": _sha256(gpt20b_path),
        })
    for eagle3_name in (
        "gpt20b-eagle3-artifact.json", "gpt20b-eagle3-derived-artifact.json",
        "gpt20b-eagle3-q4-artifact.json",
    ):
        eagle3_path = results_dir / eagle3_name
        eagle3 = _load_json(eagle3_path)
        if not isinstance(eagle3, dict):
            continue
        draft = eagle3.get("gguf_artifact") if isinstance(eagle3.get("gguf_artifact"), dict) else {}
        target = eagle3.get("target") if isinstance(eagle3.get("target"), dict) else {}
        verdict = eagle3.get("verdict") if isinstance(eagle3.get("verdict"), dict) else {}
        verified = verdict.get("artifact_complete") is True and verdict.get("full_sha256_verified") is True
        provenance.append({
            "model": "GPT-OSS-20B EAGLE3 draft", "weight_quant": draft.get("quantization"),
            "target_model": "GPT-OSS-20B", "target_weight_quant": "MXFP4",
            "target_kv_cache": "f16", "draft_kv_cache": "f16",
            "runtime": "llama.cpp server (Ollama 0.40.6 bundle)",
            "status": eagle3.get("state"),
            "source_repo": draft.get("repository"), "source_revision": draft.get("revision"),
            "expected_weight_bytes": draft.get("size"),
            "expected_weight_sha256": draft.get("sha256"),
            "recorded_weight_bytes": draft.get("size") if verified else None,
            "recorded_weight_sha256": draft.get("sha256") if verified else None,
            "verified": verified, "target_weight_sha256": target.get("sha256"),
            "profiles": eagle3.get("profiles"),
            "runtime_load_validated": verdict.get("runtime_load_validated"),
            "inference_validated": verdict.get("inference_validated"),
            "compatibility": eagle3.get("compatibility"),
            "artifact_role": "EAGLE3 draft provenance for an MXFP4 target; benchmark request rows establish draft loading, acceptance, quality, and speed",
            "source_file": _relative(eagle3_path),
            "source_receipt_sha256": _sha256(eagle3_path),
        })
    qwen122_path = results_dir / "qwen122-artifact.json"
    qwen122 = _load_json(qwen122_path)
    qwen122_layout_path = results_dir / "qwen122-layout.json"
    qwen122_layout = _load_json(qwen122_layout_path)
    if isinstance(qwen122, dict):
        model = qwen122.get("model") if isinstance(qwen122.get("model"), dict) else {}
        verdict = qwen122.get("verdict") if isinstance(qwen122.get("verdict"), dict) else {}
        shards = qwen122.get("shards") if isinstance(qwen122.get("shards"), list) else []
        shard_hashes = {
            str(item.get("name")): item.get("sha256")
            for item in shards if isinstance(item, dict) and item.get("name") and item.get("sha256")
        }
        recorded_bytes = sum(
            int(item["size"]) for item in shards
            if isinstance(item, dict) and isinstance(item.get("size"), int)
        )
        weights_verified = (
            verdict.get("artifact_complete") is True
            and bool(shards)
            and len(shard_hashes) == len(shards)
            and all(item.get("present") is True and item.get("verified") is True for item in shards if isinstance(item, dict))
            and recorded_bytes == model.get("expected_total_size")
        )
        layout_verified = isinstance(qwen122_layout, dict) and all(
            qwen122_layout.get(key) is True for key in (
                "all_offsets_aligned", "all_tensor_directory_entries_validated",
                "all_tensor_names_unique", "all_tensor_payload_bounds_validated",
                "no_payload_overlaps",
            )
        )
        provenance.append({
            "model": "Qwen3.5-122B-A10B", "weight_quant": model.get("quant"),
            "status": "artifact_complete_layout_verified" if weights_verified and layout_verified else "artifact_receipt_recorded",
            "source_repo": model.get("repo"), "source_revision": model.get("revision"),
            "expected_weight_bytes": model.get("expected_total_size"),
            "recorded_weight_bytes": recorded_bytes, "recorded_weight_sha256": shard_hashes,
            "shard_count": len(shards), "verified": weights_verified,
            "layout_verified": layout_verified,
            "layout_proof": ({
                "split_count": qwen122_layout.get("split_count"),
                "tensor_count": qwen122_layout.get("tensor_count"),
                "shard_tensor_counts": [
                    item.get("tensor_count") for item in qwen122_layout.get("shards", [])
                    if isinstance(item, dict)
                ],
                "tensor_type_counts": qwen122_layout.get("tensor_type_counts"),
                "artifact_manifest_sha256": qwen122_layout.get("artifact_manifest_sha256"),
            } if isinstance(qwen122_layout, dict) else None),
            "artifact_role": "weights and layout provenance only; benchmark request rows establish load and inference measurements",
            "source_file": _relative(qwen122_path),
            "source_receipt_sha256": _sha256(qwen122_path),
            "layout_source_file": _relative(qwen122_layout_path) if isinstance(qwen122_layout, dict) else None,
            "layout_receipt_sha256": _sha256(qwen122_layout_path) if isinstance(qwen122_layout, dict) else None,
        })
    mtp_path, mtp_receipt = _preferred_mtp_receipt(results_dir)
    if mtp_path is not None and mtp_receipt is not None:
        source = mtp_receipt.get("source") if isinstance(mtp_receipt.get("source"), dict) else {}
        payload = mtp_receipt.get("payload") if isinstance(mtp_receipt.get("payload"), dict) else {}
        provenance.append({
            "model": "Qwen3.8-Flash-Next optional MTP", "status": mtp_receipt.get("state"),
            "source_revision": source.get("revision"),
            "raw_tensor_bytes": source.get("raw_tensor_bytes", source.get("total_bytes")),
            "payload_bytes_present": payload.get("bytes_present"),
            "download": mtp_receipt.get("download"),
            "automatic_approval_rejection": mtp_receipt.get("automatic_approval_rejection"),
            "artifact_role": "optional companion receipt; benchmark request rows are authoritative for measurements",
            "source_file": _relative(mtp_path),
        })
    q3_path = results_dir / "qwen36-q3-artifact.json"
    q3 = _load_json(q3_path)
    if isinstance(q3, dict):
        model = q3.get("model") if isinstance(q3.get("model"), dict) else {}
        shards = q3.get("shards") if isinstance(q3.get("shards"), list) else []
        shard_hashes = {
            str(item.get("name")): item.get("sha256")
            for item in shards if isinstance(item, dict) and item.get("name") and item.get("sha256")
        }
        provenance.append({
            "model": "Qwen3.6-35B-A3B", "weight_quant": model.get("quant"),
            "status": q3.get("state"), "source_revision": model.get("revision"),
            "source_repo": model.get("repo"), "recorded_weight_sha256": shard_hashes,
            "recorded_weight_bytes": sum(int(item["size"]) for item in shards if isinstance(item, dict) and isinstance(item.get("size"), int)),
            "verified": bool(shards) and all(item.get("verified") is True for item in shards if isinstance(item, dict)),
            "llama_cpp": q3.get("llama_cpp"), "strata_compatibility": q3.get("strata_compatibility"),
            "source_file": _relative(q3_path),
        })
    flash = _load_json(results_dir / "flash-next-artifact.json")
    if isinstance(flash, dict):
        source = flash.get("source") if isinstance(flash.get("source"), dict) else {}
        runtime = flash.get("runtime") if isinstance(flash.get("runtime"), dict) else {}
        files = flash.get("files") if isinstance(flash.get("files"), list) else []
        local_bytes = sum(int(item["size"]) for item in files if isinstance(item, dict) and isinstance(item.get("size"), int))
        provenance.append({
            "model": flash.get("model"), "weight_quant": flash.get("quantization"),
            "artifact_state": flash.get("state"), "source_revision": source.get("revision"),
            "runtime_config": runtime.get("config"), "runtime_revision": runtime.get("strata_revision"),
            "mtp": runtime.get("mtp"), "source_file": "results/flash-next-artifact.json",
            "parameter_accounting": {
                "main_parameters_label_billions": 125,
                "activated_main_parameters_billions": 6,
                "ngram_embedding_parameters_billions": 51,
                "optional_mtp_parameters_billions": 4,
                "label_note": "The 125B main-label count excludes the 51B n-gram embedding table; optional MTP is separate and was disabled in the baseline.",
                "model_card": "https://huggingface.co/Qwen/Qwen3.8-Flash-Next",
            },
            "local_quantized_artifact_bytes": local_bytes or None,
            "local_quantized_artifact_gib": round(local_bytes / (1024 ** 3), 3) if local_bytes else None,
            "sampling_provenance": {
                "controlled_baseline": runtime.get("sampling"),
                "controlled_baseline_note": "T=0 / presence=0 is a reproducible baseline, not the model-card optimum.",
                "model_card_thinking": {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "presence_penalty": 0},
                "model_card_nonthinking": {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "presence_penalty": 1.5},
                "mode_note": "none, low, and medium are distinct request modes and remain separate report profiles.",
            },
        })
    candidate = _load_json(results_dir / "qwen122-strata-candidate.json")
    if isinstance(candidate, dict):
        source = candidate.get("source") if isinstance(candidate.get("source"), dict) else {}
        validation = candidate.get("validation") if isinstance(candidate.get("validation"), dict) else {}
        binary = (validation.get("cuda_build") or {}).get("binary") or {}
        offline = candidate.get("offline_runtime_preparation") if isinstance(candidate.get("offline_runtime_preparation"), dict) else {}
        trial = offline.get("trial_config") if isinstance(offline.get("trial_config"), dict) else {}
        tokenizer = offline.get("tokenizer") if isinstance(offline.get("tokenizer"), dict) else {}
        tokenizer_hashes = {
            name: item.get("sha256") for name, item in (tokenizer.get("files") or {}).items()
            if isinstance(item, dict) and item.get("sha256")
        }
        provenance.append({
            "model": "Qwen3.5-122B-A10B", "weight_quant": ((candidate.get("artifact_evidence") or {}).get("quant")),
            "runtime": "Strata experimental patched 0.1.45", "status": candidate.get("status"),
            "base_revision": source.get("base_revision"), "patch_sha256": source.get("patch_sha256"),
            "binary_sha256": binary.get("sha256"), "runtime_config_sha256": trial.get("sha256"),
            "tokenizer_file_sha256": tokenizer_hashes,
            "experimental_changes": candidate.get("candidate_changes"),
            "source_file": "results/qwen122-strata-candidate.json",
        })
    return provenance


def _platform_provenance(results_dir: Path) -> dict[str, Any]:
    comparison = _load_json(results_dir / "comparison-summary.json")
    brake = _load_json(results_dir / "gpu-brake-readonly.json")
    memory = _load_json(results_dir / "flash-next-bulk-memory-snapshot.json")
    hardware = comparison.get("hardware") if isinstance(comparison, dict) and isinstance(comparison.get("hardware"), dict) else {}
    conclusion = brake.get("conclusion") if isinstance(brake, dict) and isinstance(brake.get("conclusion"), dict) else {}
    memory_provenance: dict[str, Any] | None = None
    if isinstance(memory, dict):
        system = memory.get("system_kib") if isinstance(memory.get("system_kib"), dict) else {}
        engine = memory.get("engine_kib") if isinstance(memory.get("engine_kib"), dict) else {}
        memory_provenance = {
            "captured_at": memory.get("captured_at"), "trial_tag": memory.get("trial_tag"),
            "phase": memory.get("phase"), "source": memory.get("source"),
            "system_kib": system, "engine_kib": engine,
            "engine_rss_gib": round(float(engine["VmRSS"]) / (1024 ** 2), 3) if _number(engine.get("VmRSS")) is not None else None,
            "system_mem_available_gib": round(float(system["MemAvailable"]) / (1024 ** 2), 3) if _number(system.get("MemAvailable")) is not None else None,
            "engine_vm_swap_kib": _number(engine.get("VmSwap")),
            "memory_pressure": memory.get("memory_pressure"),
            "preceding_live_vmstat_interval": memory.get("preceding_live_vmstat_interval"),
            "interpretation": memory.get("interpretation"),
        }
    return {
        "hardware": hardware,
        "gpu_power_brake": {
            "hardware_power_brake_evidenced": conclusion.get("hardware_power_brake_is_evidenced"),
            "active_platform_power_cap_evidenced": conclusion.get("active_platform_power_cap_is_evidenced"),
            "platform_cap_status": conclusion.get("platform_cap_status"),
            "source_localized": conclusion.get("source_localized"),
            "summary": conclusion.get("summary"),
        },
        "flash_memory_snapshot": memory_provenance,
        "flash_followups_memory_interval": _load_json(results_dir / "flash-next-followups-memory-interval.json"),
        "sources": ["results/comparison-summary.json", "results/gpu-brake-readonly.json", "results/flash-next-bulk-memory-snapshot.json"] + (
            ["results/flash-next-followups-memory-interval.json"]
            if (results_dir / "flash-next-followups-memory-interval.json").is_file() else []
        ),
    }


def _profile_summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def observed(items: list[dict[str, Any]], key: str) -> dict[str, Any]:
        values = [value for item in items if (value := _number(item.get(key))) is not None]
        return {
            "sum": sum(values) if values else None,
            "sample_count": len(values),
            "missing_count": len(items) - len(values),
        }

    def clean_distribution(
        items: list[dict[str, Any]], eligibility_key: str, value_key: str
    ) -> dict[str, Any]:
        eligible = [item for item in items if item.get(eligibility_key) is True]
        values = [float(value) for item in eligible if (value := _number(item.get(value_key))) is not None]
        return {
            "median": statistics.median(values) if values else None,
            "min": min(values) if values else None,
            "max": max(values) if values else None,
            "sample_count": len(values),
            "missing_count": len(eligible) - len(values),
        }

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["profile_id"]].append(row)
    summaries: list[dict[str, Any]] = []
    for profile_id, items in sorted(groups.items()):
        variant_tasks = {
            (x.get("case_id"), x.get("case_version"), x.get("prompt_sha256")) for x in items
        }
        underlying_tasks = {_underlying_task_identity(x) for x in items}
        input_tokens = observed(items, "api_input_tokens")
        completion_tokens = observed(items, "completion_tokens")
        wall_total = observed(items, "wall_seconds")
        decode = clean_distribution(items, "speed_eligible", "native_decode_tps")
        clean_wall = clean_distribution(items, "latency_eligible", "wall_seconds")
        clean_ttft = clean_distribution(items, "latency_eligible", "ttft_final_seconds")
        summary = {
            "profile_id": profile_id, "profile": _profile_parts(items[0]),
            "attempt_count": len(items), "independent_task_count": len(underlying_tasks),
            "case_ids": sorted({str(x.get("case_id")) for x in items}),
            "outcomes": dict(sorted(Counter(str(x.get("outcome")) for x in items).items())),
            "quality_eligible_attempts": sum(x.get("quality_eligible") is True for x in items),
            "comparison_passed_attempts": sum(x.get("quality_eligible") is True and x.get("comparison_content_passed") is True for x in items),
            "comparison_failed_attempts": sum(x.get("quality_eligible") is True and x.get("comparison_content_passed") is False for x in items),
            "speed_eligible_attempts": sum(x.get("speed_eligible") is True for x in items),
            "speed_ineligible_attempts": sum(x.get("speed_eligible") is not True for x in items),
            "latency_eligible_attempts": sum(x.get("latency_eligible") is True for x in items),
            "latency_ineligible_attempts": sum(x.get("latency_eligible") is not True for x in items),
            "observed_api_input_tokens_sum": input_tokens["sum"],
            "observed_api_input_tokens_sample_count": input_tokens["sample_count"],
            "observed_api_input_tokens_missing_count": input_tokens["missing_count"],
            "observed_completion_tokens_sum": completion_tokens["sum"],
            "observed_completion_tokens_sample_count": completion_tokens["sample_count"],
            "observed_completion_tokens_missing_count": completion_tokens["missing_count"],
            "observed_request_wall_seconds_sum": wall_total["sum"],
            "observed_request_wall_seconds_sample_count": wall_total["sample_count"],
            "observed_request_wall_seconds_missing_count": wall_total["missing_count"],
            "clean_native_decode_tps_median": decode["median"],
            "clean_native_decode_tps_min": decode["min"],
            "clean_native_decode_tps_max": decode["max"],
            "clean_native_decode_tps_sample_count": decode["sample_count"],
            "clean_native_decode_tps_missing_count": decode["missing_count"],
            "clean_request_wall_seconds_median": clean_wall["median"],
            "clean_request_wall_seconds_min": clean_wall["min"],
            "clean_request_wall_seconds_max": clean_wall["max"],
            "clean_request_wall_seconds_sample_count": clean_wall["sample_count"],
            "clean_request_wall_seconds_missing_count": clean_wall["missing_count"],
            "clean_ttft_final_seconds_median": clean_ttft["median"],
            "clean_ttft_final_seconds_min": clean_ttft["min"],
            "clean_ttft_final_seconds_max": clean_ttft["max"],
            "clean_ttft_final_seconds_sample_count": clean_ttft["sample_count"],
            "clean_ttft_final_seconds_missing_count": clean_ttft["missing_count"],
        }
        if any("parent_case_id" in item for item in items):
            summary.update({
                "variant_task_count": len(variant_tasks),
                "parent_case_ids": sorted({
                    str(item["parent_case_id"])
                    for item in items
                    if isinstance(item.get("parent_case_id"), str)
                }),
                "diagnostic_variants": sorted({
                    str(item["diagnostic_variant"])
                    for item in items
                    if isinstance(item.get("diagnostic_variant"), str)
                }),
            })
        summaries.append(summary)
    return summaries


def _matched(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("quality_eligible") is True:
            groups[row["match_id"]].append(row)
    result = []
    for match_id, items in sorted(groups.items()):
        profiles = {x["profile_id"] for x in items}
        if len(profiles) < 2:
            continue
        result.append({
            "match_id": match_id, "settings": _match_parts(items[0]),
            "attempt_count": len(items), "independent_task_count": 1,
            "attempts": [{key: x.get(key) for key in ("row_id", "profile_id", "model", "runtime", "thinking", "sampling_seed", "outcome", "content_passed", "comparison_content_passed", "rescored_cause", "native_decode_tps", "wall_seconds")} for x in items],
        })
    return result


def build_report(results_dir: Path) -> dict[str, Any]:
    results_dir = results_dir.resolve()
    lifecycle, lifecycle_prefixes, events, event_sources = _event_index(results_dir)
    specialist_lifecycle, specialist_receipt_sources = _specialist_receipt_index(results_dir)
    hidden, formats = _diagnostics(results_dir)
    timing_exclusions, timing_annotation_sources = _timing_exclusions(results_dir)
    raw: list[tuple[Path, int, dict[str, Any]]] = []
    input_sources: list[Path] = []
    invalid_lines = 0
    for path in sorted(results_dir.glob("*.jsonl")):
        parsed, bad = _jsonl(path); invalid_lines += bad
        selected = [(line, value) for line, value in parsed if _request_row(value)]
        if selected:
            input_sources.append(path)
            raw.extend((path, line, value) for line, value in selected)
    dedup: dict[str, tuple[Path, int, dict[str, Any]]] = {}
    duplicate_locations: dict[str, list[str]] = defaultdict(list)
    duplicate_conflicts: list[str] = []
    for path, line, value in raw:
        call_id = value["call_id"]
        location = f"{_relative(path)}:{line}"
        if call_id in dedup:
            duplicate_locations[call_id].append(location)
            if value != dedup[call_id][2]:
                duplicate_conflicts.append(call_id)
            continue
        dedup[call_id] = (path, line, value)
        duplicate_locations[call_id].append(location)
    def trial_for(path: Path) -> dict[str, Any]:
        specialist_trial = specialist_lifecycle.get(str(path.resolve()))
        if specialist_trial:
            return specialist_trial
        exact = lifecycle.get(str(path.resolve()))
        if exact:
            return exact
        stem = path.stem
        for tag, trial in lifecycle_prefixes:
            if stem == tag or stem.startswith(tag + "-"):
                return trial
        return _legacy_trial(path)

    rows = [normalize_request(value, path, line, trial_for(path), hidden, formats, timing_exclusions) for path, line, value in dedup.values()]
    rows.sort(key=lambda x: (str(x.get("started_at") or ""), x["row_id"]))
    health, health_sources = _health_rows(results_dir, events)
    setup = _setup_rows(events)
    cache, cache_sources = _cache_rows(results_dir)
    diagnostics, diagnostic_sources = _diagnostic_rows(results_dir)
    not_run, note_sources = _not_run(results_dir, rows)
    reference_names = (
        "baseline-summary.json", "comparison-summary.json", "current-model-hard-comparison.json",
        "current-model-hard-comparison.csv",
        "hard-round-report.json", "newer-20b-hard-quality-report.json",
        "newer-20b-quality-control-annotation.json", "hard-json-format-audit.json",
        "qwen36-long-context-comparison.json", "qwen36-artifact.json",
        "qwen36-native-template-parity.json", "qwen36-native-generation-defaults.json",
        "gpu-brake-readonly.json",
        "flash-next-mtp-artifact.json", "flash-next-mtp-candidate.json",
        "flash-next-accuracy-reference.json", "flash-next-adaptive-overlap-blocked.json",
        "qwen36-q3-artifact.json", "gpt20b-llama-artifact.json", "gpt20b-eagle3-artifact.json",
        "gpt20b-eagle3-derived-artifact.json", "gpt20b-eagle3-tokenizer-comparison.json",
        "gpt20b-eagle3-q4-artifact.json", "gpt20b-eagle3-comparison.json",
        "gpt20b-eagle3-summary.csv", "gpt20b-eagle3-final-cleanup.json",
        "qwen122-artifact.json", "qwen122-layout.json",
        "flash-next-bulk-memory-snapshot.json", "flash-next-followups-memory-interval.json",
        "lfm25-qad-q4_0-artifact.json", "kat-v25-iq3_xxs-artifact.json",
        "gemma4-style12-q5_k_m-artifact.json", "bonsai2-runtime-artifact.json",
        "flash-next-coder-artifact.json", "flash-next-coder-header-preview.json",
        "benchmark-harness-verification.json",
    )
    reference_sources = [results_dir / name for name in reference_names if (results_dir / name).is_file()]
    source_paths = sorted(set(input_sources + event_sources + specialist_receipt_sources + health_sources + cache_sources + diagnostic_sources + note_sources + timing_annotation_sources + reference_sources), key=str)
    source_provenance = [{"path": _relative(path), "bytes": path.stat().st_size, "sha256": _sha256(path)} for path in source_paths if path.is_file()]
    return {
        "schema_version": SCHEMA_VERSION, "generated_at": _now(),
        "scope": "Real model request, health/setup, and expert-cache experiment attempts; harness unit tests are excluded.",
        "policy": {
            "no_global_ranking": True,
            "quality_unit": "independent task; hidden assertions diagnose a task and are not additional tasks",
            "repeat_policy": "Every attempt is retained; repeat seeds do not increase independent-task coverage.",
            "evaluator_rejections": "Private/dunder-only policy rejections preserve the recorded workflow failure but leave semantic correctness unverified and excluded from quality comparison.",
            "json_format": "Semantic JSON evaluator accepts one outer code fence; raw_json_object is a separate strict-format audit.",
            "rescoring": "Original content_passed is preserved. comparison_content_passed uses the recorded restricted-evaluator rescore when hard-round-report provides one.",
            "timing": "Native decode/prompt rates are distinct from request wall latency. Request wall excludes wrapper wait and model-server startup.",
            "token_counts": "API/native counts are preserved under their original semantics. Missing full-prefix, reasoning-token, and visible-answer-token counts remain null.",
            "reasoning_budgets": "Budget columns record requested/configured settings, not measured total reasoning tokens. Native thinking_budget_tokens is kept separate from Strata reasoning_budget_tokens and template effort. Runtime enforcement requires separate evidence; the checked native sampler can re-arm a budget for each reasoning block.",
            "time_to_first_visible_output": "Final TTFT is the first visible content after any separate reasoning. It can be a preamble and does not indicate that a complete or correct answer is available.",
            "specialist_prose": "Deterministic prose constraints are mechanical format/substring checks. They are retained separately and never treated as an overall prose-quality pass; subjective quality requires the optional blinded review artifact.",
            "diagnostic_variants": "Rows with a non-original diagnostic variant, a changed task domain, or algorithm guidance are isolated from original prompt profiles and matched comparisons. Variants sharing parent_case_id count as one underlying benchmark problem; this does not claim equal difficulty or a new independent benchmark problem.",
        },
        "data_gaps": [
            "Visible answer tokens are not separately recorded. Reasoning tokens remain null unless the server explicitly reported them; no estimate is substituted.",
            "Private/dunder-only evaluator rejections did not execute the generated code; these are marked evaluator_rejected rather than inferred to be semantically wrong.",
            "Explicit reasoning budgets are recorded when present in the request or saved engine config; otherwise they remain null.",
            "Initial result files without lifecycle linkage may have unknown runtime or KV-cache metadata.",
            "Ollama prompt_eval_count, OpenAI usage.prompt_tokens, Strata prompt_n, cache_n, and cached_tokens retain server-specific meanings.",
        ],
        "inventory": {"request_rows_before_dedup": len(raw), "request_rows": len(rows), "duplicate_call_ids": {k: v for k, v in duplicate_locations.items() if len(v) > 1}, "conflicting_duplicate_call_ids": sorted(set(duplicate_conflicts)), "invalid_json_lines": invalid_lines, "health_rows": len(health), "runtime_trial_rows": len(setup), "setup_diagnostic_rows": len(diagnostics), "expert_cache_rows": len(cache)},
        "request_rows": rows,
        "supplemental_rows": setup + health + diagnostics + cache,
        "profile_summaries": _profile_summaries(rows),
        "matched_comparisons": _matched(rows),
        "not_run": not_run,
        "artifact_provenance": _artifact_provenance(results_dir),
        "platform_provenance": _platform_provenance(results_dir),
        "ignored_status_sources": [
            "results/test-status.json (stale first-round state is not used for current completion status)",
            "results/hard-round-state.json (stale checkpoint; raw request and lifecycle evidence determine coverage)",
        ],
        "source_provenance": source_provenance,
    }


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    if value is None:
        return ""
    return value


def render_csv(report: Mapping[str, Any]) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=CSV_COLUMNS, extrasaction="ignore")
    writer.writeheader()
    for row in list(report.get("request_rows", [])) + list(report.get("supplemental_rows", [])):
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


def _table(rows: Iterable[Mapping[str, Any]], columns: list[tuple[str, str]], table_id: str) -> str:
    heads = "".join(f'<th data-key="{html.escape(key)}">{html.escape(label)}</th>' for key, label in columns)
    bodies = []
    for row in rows:
        cells = "".join(f"<td>{html.escape(_display(row.get(key)))}</td>" for key, _ in columns)
        search = html.escape(" ".join(_display(row.get(key)) for key, _ in columns).lower(), quote=True)
        title = html.escape(_display(row.get("_row_title"))) if row.get("_row_title") is not None else ""
        bodies.append(f'<tr data-search="{search}" title="{title}">{cells}</tr>')
    return f'<div class="scroll"><table id="{html.escape(table_id)}"><thead><tr>{heads}</tr></thead><tbody>{"".join(bodies)}</tbody></table></div>'


def _scatter(rows: list[Mapping[str, Any]]) -> str:
    points = [r for r in rows if _number(r.get("wall_seconds")) is not None and _number(r.get("native_decode_tps")) is not None and r.get("speed_eligible") is True]
    if not points:
        return "<p>No eligible request speed points yet.</p>"
    models = sorted({str(row.get("model") or "unknown") for row in points})
    palette = ("#38bdf8", "#fb7185", "#4ade80", "#c084fc", "#fbbf24", "#2dd4bf")
    colors = {model: palette[index % len(palette)] for index, model in enumerate(models)}
    width, height, pad = 900, 390, 52
    plot_top = 54 + 17 * len(models)
    max_x = max(float(r["wall_seconds"]) for r in points) or 1
    max_y = max(float(r["native_decode_tps"]) for r in points) or 1
    marks = []
    for row in points:
        x = pad + float(row["wall_seconds"]) / max_x * (width - 2 * pad)
        y = height - pad - float(row["native_decode_tps"]) / max_y * (height - pad - plot_top)
        model = str(row.get("model") or "unknown")
        color = colors[model]
        title = html.escape(
            f"{model} | {row.get('runtime')} | thinking={row.get('thinking')} | "
            f"{row.get('case_id')} | cap={row.get('output_token_limit')} | "
            f"outcome={row.get('outcome')} | comparison_pass={row.get('comparison_content_passed')} | "
            f"wall={float(row['wall_seconds']):.2f}s | native={float(row['native_decode_tps']):.2f} tok/s"
        )
        status = str(row.get("outcome"))
        if status == "passed" and row.get("comparison_content_passed") is True:
            mark = f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{color}" stroke="#e2e8f0"><title>{title}</title></circle>'
        elif status == "wrong_answer":
            mark = f'<path d="M{x:.1f},{y-6:.1f} L{x+6:.1f},{y:.1f} L{x:.1f},{y+6:.1f} L{x-6:.1f},{y:.1f} Z" fill="none" stroke="{color}" stroke-width="2"><title>{title}</title></path>'
        elif status in ("truncated", "incomplete_stream"):
            mark = f'<rect x="{x-5:.1f}" y="{y-5:.1f}" width="10" height="10" fill="none" stroke="{color}" stroke-width="2" stroke-dasharray="3 2"><title>{title}</title></rect>'
        else:
            mark = f'<path d="M{x-5:.1f},{y-5:.1f} L{x+5:.1f},{y+5:.1f} M{x+5:.1f},{y-5:.1f} L{x-5:.1f},{y+5:.1f}" stroke="{color}" stroke-width="2"><title>{title}</title></path>'
        marks.append(mark)
    model_legend = []
    for index, model in enumerate(models):
        y = 20 + index * 17
        model_legend.append(f'<circle cx="{pad}" cy="{y}" r="4" fill="{colors[model]}"/><text x="{pad+10}" y="{y+4}" fill="#dbeafe">{html.escape(model)}</text>')
    status_y = 20
    status_legend = (
        f'<text x="520" y="{status_y+4}" fill="#dbeafe">● pass</text>'
        f'<text x="600" y="{status_y+4}" fill="#dbeafe">◇ wrong answer</text>'
        f'<text x="735" y="{status_y+4}" fill="#dbeafe">□ truncated</text>'
        f'<text x="830" y="{status_y+4}" fill="#dbeafe">× other</text>'
    )
    return f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Individual eligible request wall latency versus native decode rate, colored by model and shaped by outcome"><line x1="{pad}" y1="{height-pad}" x2="{width-pad}" y2="{height-pad}" stroke="#64748b"/><line x1="{pad}" y1="{plot_top}" x2="{pad}" y2="{height-pad}" stroke="#64748b"/><text x="{width/2}" y="{height-8}" text-anchor="middle" fill="#dbeafe">Request wall seconds</text><text x="16" y="{(plot_top+height-pad)/2}" transform="rotate(-90 16 {(plot_top+height-pad)/2})" text-anchor="middle" fill="#dbeafe">Native decode tokens/s</text>{"".join(model_legend)}{status_legend}{"".join(marks)}</svg>'


def render_html(report: Mapping[str, Any]) -> str:
    requests = list(report.get("request_rows", [])); supplemental = list(report.get("supplemental_rows", []))
    compact_rows = []
    for row in requests:
        item = dict(row)
        passed = row.get("hidden_tests_passed")
        count = row.get("hidden_test_count")
        item["hidden_checks"] = f"{passed if passed is not None else '?'}/{count}" if count else None
        item["_row_title"] = "Profile controls: " + json.dumps(_profile_parts(row), sort_keys=True, separators=(",", ":"))
        compact_rows.append(item)
    compact_cols = [("model", "Model"), ("runtime", "Runtime"), ("thinking", "Thinking"), ("case_id", "Task"), ("output_token_limit", "Cap"), ("api_input_tokens", "API-reported input tokens"), ("completion_tokens", "Completion tokens"), ("native_decode_tps", "Native decode tok/s"), ("wall_seconds", "Request wall s"), ("outcome", "Outcome"), ("comparison_content_passed", "Content pass"), ("hidden_checks", "Hidden checks"), ("profile_id", "Profile")]
    req_cols = [("started_at", "UTC"), ("model", "Model"), ("weight_quant", "Weight quant"), ("draft_weight_quant", "Draft weight quant"), ("kv_cache", "KV cache"), ("runtime", "Runtime"), ("source_config", "Source config"), ("source_config_sha256", "Config SHA-256"), ("runtime_options", "Runtime options"), ("requested_native_runtime_controls", "Requested native runtime controls"), ("ignored_runtime_options", "Configured but rejected options"), ("sampling_config", "Sampler config"), ("suite", "Suite"), ("case_id", "Task"), ("parent_case_id", "Parent task"), ("diagnostic_variant", "Diagnostic variant"), ("diagnostic_domain_changed", "Domain changed"), ("diagnostic_algorithm_guidance", "Algorithm guidance"), ("thinking", "Thinking"), ("chat_template_thinking", "Template thinking"), ("reasoning_budget_tokens_requested", "Strata reason budget requested"), ("reasoning_budget_tokens_config_default", "Strata reason budget config default"), ("native_thinking_budget_tokens_requested", "Native thinking budget requested"), ("native_thinking_budget_application_evidence", "Native thinking budget evidence"), ("ledger_json_schema_sha256", "Ledger schema SHA-256"), ("ledger_json_schema_requested", "Ledger schema request evidence"), ("strata_pcie_frac", "Strata PCIe frac"), ("strata_aux_cpus", "Strata aux CPUs"), ("temperature", "Temp"), ("sampling_seed", "Seed"), ("context_max_tokens", "Context"), ("output_token_limit", "Output cap"), ("api_input_tokens", "API input"), ("ollama_prompt_eval_count", "Ollama evaluated"), ("native_prompt_n", "Native prompt n"), ("native_cache_n", "Native cached"), ("api_cached_prompt_tokens", "API cached"), ("completion_tokens", "Completion"), ("ollama_eval_count", "Ollama eval n"), ("native_decode_n", "Native decode n"), ("native_draft_n", "Draft proposed"), ("native_draft_n_accepted", "Draft accepted"), ("native_draft_acceptance_rate", "Draft acceptance rate"), ("reasoning_chars", "Reason chars"), ("native_prompt_seconds", "Native prompt s"), ("native_decode_seconds", "Native decode s"), ("native_load_seconds", "Native load s"), ("native_prompt_tps", "Prompt tok/s"), ("native_decode_tps", "Decode tok/s"), ("wall_seconds", "Wall s"), ("ttft_final_seconds", "TTFT final s"), ("outcome", "Outcome"), ("content_passed", "Original pass"), ("semantic_behavior_passed", "Semantic behavior pass"), ("source_edit_fidelity_passed", "Source edit fidelity pass"), ("comparison_content_passed", "Comparison pass"), ("rescored_cause", "Rescore note"), ("raw_json_object", "Raw JSON object"), ("outer_code_fence", "Outer code fence"), ("empty_answer", "Empty answer"), ("quality_eligible", "Quality compare"), ("speed_eligible", "Speed compare"), ("request_error_message", "Request error"), ("exclusion_reasons", "Notes"), ("source_file", "Source")]
    sup_cols = [("row_type", "Type"), ("model", "Model"), ("runtime", "Runtime"), ("source_config", "Source config"), ("runtime_options", "Runtime options"), ("requested_native_runtime_controls", "Requested native runtime controls"), ("ignored_runtime_options", "Configured but rejected options"), ("case_id", "Probe"), ("context_max_tokens", "Context"), ("startup_seconds", "Startup s"), ("wall_seconds", "Wall s"), ("response_bytes", "Response bytes"), ("native_decode_tps", "Decode tok/s"), ("outcome", "Outcome"), ("evaluation_detail", "Detail"), ("exclusion_reasons", "Notes"), ("source_file", "Source")]
    profile_cols = [
        ("profile_id", "Profile"), ("profile", "Controls"), ("attempt_count", "Attempts"),
        ("independent_task_count", "Independent underlying tasks"),
        ("variant_task_count", "Concrete task variants"), ("case_ids", "Coverage"),
        ("diagnostic_variants", "Diagnostic variants"), ("outcomes", "Outcomes"),
        ("quality_eligible_attempts", "Quality attempts"), ("comparison_passed_attempts", "Passed"),
        ("comparison_failed_attempts", "Failed"),
        ("observed_api_input_tokens_sum", "Observed API input tokens sum"),
        ("observed_api_input_tokens_sample_count", "Input n"),
        ("observed_api_input_tokens_missing_count", "Input missing"),
        ("observed_completion_tokens_sum", "Observed completion tokens sum"),
        ("observed_completion_tokens_sample_count", "Completion n"),
        ("observed_completion_tokens_missing_count", "Completion missing"),
        ("observed_request_wall_seconds_sum", "Observed request wall s sum"),
        ("observed_request_wall_seconds_sample_count", "Wall total n"),
        ("observed_request_wall_seconds_missing_count", "Wall total missing"),
        ("speed_eligible_attempts", "Clean speed eligible"),
        ("speed_ineligible_attempts", "Speed excluded"),
        ("clean_native_decode_tps_sample_count", "Decode TPS n"),
        ("clean_native_decode_tps_missing_count", "Decode TPS missing"),
        ("clean_native_decode_tps_median", "Decode TPS median"),
        ("clean_native_decode_tps_min", "Decode TPS min"),
        ("clean_native_decode_tps_max", "Decode TPS max"),
        ("latency_eligible_attempts", "Clean latency eligible"),
        ("latency_ineligible_attempts", "Latency excluded"),
        ("clean_request_wall_seconds_sample_count", "Clean wall n"),
        ("clean_request_wall_seconds_missing_count", "Clean wall missing"),
        ("clean_request_wall_seconds_median", "Clean wall median s"),
        ("clean_request_wall_seconds_min", "Clean wall min s"),
        ("clean_request_wall_seconds_max", "Clean wall max s"),
        ("clean_ttft_final_seconds_sample_count", "Final TTFT n"),
        ("clean_ttft_final_seconds_missing_count", "Final TTFT missing"),
        ("clean_ttft_final_seconds_median", "Final TTFT median s"),
        ("clean_ttft_final_seconds_min", "Final TTFT min s"),
        ("clean_ttft_final_seconds_max", "Final TTFT max s"),
    ]
    artifact_cols = [("artifact_family", "Artifact family"), ("model", "Model"), ("packing", "Packing"), ("weight_quant", "Weight quant"), ("quantization_caveat", "Quantization note"), ("target_weight_quant", "Target weight quant"), ("runtime", "Runtime"), ("runtime_source_revision", "Runtime revision"), ("runtime_server_sha256", "Runtime server SHA-256"), ("runtime_load_validated", "Runtime load validated"), ("all_gpu_fit_claimed", "All-GPU fit claimed"), ("status", "Status"), ("artifact_role", "Evidence role"), ("source_repo", "Source repo"), ("source_revision", "Source revision"), ("expected_weight_bytes", "Expected weight bytes"), ("expected_weight_sha256", "Expected weight SHA-256"), ("recorded_weight_bytes", "Recorded weight bytes"), ("recorded_weight_sha256", "Recorded weight SHA-256"), ("verified", "Weights verified"), ("verification_scope", "Verification scope"), ("header_preview", "Header preview evidence"), ("model_card", "Model card"), ("layout_verified", "Layout verified"), ("layout_source_file", "Layout source"), ("base_revision", "Runtime base"), ("patch_sha256", "Patch SHA-256"), ("binary_sha256", "Binary SHA-256"), ("runtime_config_sha256", "Config SHA-256"), ("tokenizer_file_sha256", "Tokenizer hashes"), ("source_file", "Source")]
    notes = "".join(f"<li>{html.escape(str(x))}</li>" for x in report.get("data_gaps", []))
    not_run = "".join(f"<li><strong>{html.escape(_display(x.get('model')))}</strong>: {html.escape(_display(x.get('status')))} — {html.escape(_display(x.get('reason')))}</li>" for x in report.get("not_run", []))
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Model benchmark report</title><style>
body{{font:14px system-ui,sans-serif;margin:0;background:#0b1020;color:#e5e7eb}}main{{max-width:1500px;margin:auto;padding:24px}}h1,h2{{color:#f8fafc}}.lede,.note{{color:#bac7db;max-width:1000px}}.cards{{display:flex;gap:12px;flex-wrap:wrap}}.card{{background:#172033;border:1px solid #334155;border-radius:8px;padding:12px;min-width:145px}}.card b{{font-size:24px;display:block}}input{{width:min(600px,90%);padding:9px;background:#111827;color:#fff;border:1px solid #475569;border-radius:5px}}.scroll{{overflow:auto;border:1px solid #334155;border-radius:6px;margin:10px 0 26px}}table{{border-collapse:collapse;width:100%;background:#111827}}th{{position:sticky;top:0;background:#1e293b;cursor:pointer;white-space:nowrap}}th,td{{padding:7px 9px;border-bottom:1px solid #293548;text-align:left;vertical-align:top;max-width:440px}}tr:hover{{background:#172554}}svg{{width:100%;max-height:360px;background:#111827;border:1px solid #334155;border-radius:6px}}a{{color:#7dd3fc}}code{{color:#fda4af}}
</style></head><body><main><h1>Model benchmark evidence</h1><p class="lede">Generated {html.escape(str(report.get('generated_at')))} from raw request JSONL and recorded lifecycle artifacts. Each dot and request row is one attempt. There is no pooled global model ranking.</p><div class="cards"><div class="card"><b>{len(requests)}</b>request attempts</div><div class="card"><b>{len(report.get('profile_summaries', []))}</b>profiles</div><div class="card"><b>{len(report.get('matched_comparisons', []))}</b>matched sets</div><div class="card"><b>{len(supplemental)}</b>health/cache rows</div></div>
<h2>How to read it</h2><p class="note">Weight quantization and KV-cache type are separate. Requested native runtime controls reproduce explicit config arguments; they are not measurements of cache or device residency. In particular, gpu_layers=all does not establish entirely GPU-resident execution, and n_cpu_moe requests CPU expert blocks. API input/evaluated/cached token fields retain server-specific semantics. Completion tokens can include private reasoning; visible-answer tokens were not recorded separately. Final TTFT measures the first visible content, which may be a preamble, rather than a complete or correct answer. Request wall time excludes wrapper wait and server startup.</p><ul>{notes}</ul>
<h2>Eligible request speed points</h2>{_scatter(requests)}
<h2>Compact attempt results</h2><input id="filter" placeholder="Filter compact and full request tables"><p class="note">Every row is one attempt. Hover a row for its exact profile controls; hidden checks diagnose one task and are never counted as extra tasks.</p>{_table(compact_rows, compact_cols, 'compact')}
<h2>Profile coverage and observed work</h2><p class="note">Attempt count includes repeats. Independent task count uses parent_case_id for diagnostic variants when available and otherwise uses unique task/version/prompt coverage; hidden assertions never increase it. Concrete variant counts remain separate. Observed totals retain every available row, including timing-contaminated attempts, and show missing counts. Clean medians and ranges use only the corresponding speed- or latency-eligible attempts. API input-token totals retain each backend's reported semantics.</p>{_table(report.get('profile_summaries', []), profile_cols, 'profiles')}
<h2>All model request attempts and controls</h2><p class="note">Click a column heading to sort. Exclusions stay visible instead of being dropped.</p>{_table(requests, req_cols, 'requests')}
<h2>Health, setup, and expert-cache experiments</h2>{_table(supplemental, sup_cols, 'supplemental')}
<h2>Artifact and runtime provenance</h2>{_table(report.get('artifact_provenance', []), artifact_cols, 'artifacts')}
<h2>Measured platform</h2><pre>{html.escape(json.dumps(report.get('platform_provenance'), indent=2, sort_keys=True))}</pre>
<h2>Not run / unresolved</h2><ul>{not_run}</ul>
<h2>Comparison policy</h2><pre>{html.escape(json.dumps(report.get('policy'), indent=2, sort_keys=True))}</pre>
</main><script>
const q=document.getElementById('filter'),filtered=[document.getElementById('compact'),document.getElementById('requests')];q.addEventListener('input',()=>{{const v=q.value.toLowerCase();for(const t of filtered)for(const r of t.tBodies[0].rows)r.hidden=!r.dataset.search.includes(v)}});for(const table of document.querySelectorAll('table'))for(const [i,h] of [...table.tHead.rows[0].cells].entries())h.onclick=()=>{{const b=table.tBodies[0],rows=[...b.rows],asc=h.dataset.asc!=='1';rows.sort((a,c)=>a.cells[i].innerText.localeCompare(c.cells[i].innerText,undefined,{{numeric:true}})*(asc?1:-1));for(const r of rows)b.appendChild(r);h.dataset.asc=asc?'1':'0'}};
</script></body></html>'''


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False)
    try:
        with handle:
            handle.write(content)
            handle.flush(); os.fsync(handle.fileno())
        os.replace(handle.name, path)
    except BaseException:
        try: os.unlink(handle.name)
        except OSError: pass
        raise


def write_report(report: Mapping[str, Any], prefix: Path) -> list[Path]:
    outputs = [prefix.with_suffix(".json"), prefix.with_suffix(".csv"), prefix.with_suffix(".html")]
    _atomic_write(outputs[0], json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    _atomic_write(outputs[1], render_csv(report))
    _atomic_write(outputs[2], render_html(report))
    return outputs


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--output-prefix", type=Path, default=ROOT / "results" / "model-benchmark-report")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(args.results_dir)
    outputs = write_report(report, args.output_prefix)
    print(f"wrote {len(report['request_rows'])} request attempts to " + ", ".join(str(x) for x in outputs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
