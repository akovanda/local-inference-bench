#!/usr/bin/env python3
"""Run an explicit sampler recipe against an already-held native llama server.

The probe never starts, stops, or reconfigures a server. It binds to one exact
``model_session`` holding lifecycle, preflights every selected frozen prompt,
then runs the selected cases serially. Sampler fields in the artifacts describe
requested request overrides; they are not proof that a server applied them.
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
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import bench  # noqa: E402
import bench_cases  # noqa: E402
from runtime import budget_prompt_probe, specialist_bench, specialist_cases  # noqa: E402


SCHEMA_VERSION = 1
CONTEXT_TOKENS = 16_384
DEFAULT_OUTPUT_CAP = 8_192
MAX_OUTPUT_CAP = 65_536
DEFAULT_TIMEOUT_SECONDS = 2_400.0
MAX_TIMEOUT_SECONDS = 21_600.0
MAX_CONTEXT_TOKENS = 131_072
CHAT_TEMPLATE_TOKEN_RESERVE = 256
SEED = 0
LEDGER_CASE_ID = "hard_transaction_ledger"
GROUPS = ("ledger", "code", "prose")
NATIVE_THINKING_BUDGET_APPLICATION_EVIDENCE = (
    "requested_in_outgoing_payload_not_independently_verified_applied_or_enforced"
)
LEDGER_JSON_SCHEMA_APPLICATION_EVIDENCE = (
    "strict_json_schema_requested_in_outgoing_payload_enforcement_not_independently_verified"
)
MAX_LEDGER_SCHEMA_BYTES = 64 * 1024
MAX_LEDGER_SCHEMA_DEPTH = 8
MAX_LEDGER_SCHEMA_NODES = 128
MAX_LEDGER_OBJECT_PROPERTIES = 64
SAMPLER_FIELDS = (
    "temperature", "top_p", "top_k", "min_p", "presence_penalty", "repeat_penalty",
)
TERMINAL_EVENTS = {
    "holding_stopped", "session_interrupted", "session_failed", "server_stopped",
    "server_cleanup_failed", "session_complete",
}
RECIPE_TOP_LEVEL_FIELDS = {
    "schema_version", "label", "claim", "sources", "sampler_overrides", "notes",
}


class ProbeError(RuntimeError):
    """The held-server or recipe preconditions were not satisfied."""


@dataclass(frozen=True)
class SelectedCase:
    group: str
    case: Any


@dataclass(frozen=True)
class LedgerJsonSchema:
    path: Path
    sha256: str
    schema: dict[str, Any]


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def _sha256_file(path: Path) -> str:
    return budget_prompt_probe.sha256_file(path)


def _json_file(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ProbeError(f"recipe does not exist: {path}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise ProbeError(f"recipe is not readable JSON: {path}") from exc


def _validate_ledger_schema_node(
    value: Any, *, location: str, depth: int, node_count: list[int],
) -> None:
    if depth > MAX_LEDGER_SCHEMA_DEPTH:
        raise ProbeError(f"ledger JSON Schema exceeds maximum depth at {location}")
    if not isinstance(value, dict):
        raise ProbeError(f"ledger JSON Schema node at {location} must be an object")
    node_count[0] += 1
    if node_count[0] > MAX_LEDGER_SCHEMA_NODES:
        raise ProbeError("ledger JSON Schema has too many nodes")
    schema_type = value.get("type")
    if schema_type == "object":
        allowed = {"type", "properties", "required", "additionalProperties"}
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise ProbeError(
                f"unsupported ledger JSON Schema fields at {location}: {', '.join(unknown)}"
            )
        properties = value.get("properties")
        required = value.get("required")
        if not isinstance(properties, dict) or not properties:
            raise ProbeError(f"ledger JSON Schema object at {location} requires properties")
        if len(properties) > MAX_LEDGER_OBJECT_PROPERTIES:
            raise ProbeError(f"ledger JSON Schema object at {location} has too many properties")
        if any(
            not isinstance(name, str) or not name or len(name) > 128 or any(ord(ch) < 32 for ch in name)
            for name in properties
        ):
            raise ProbeError(f"ledger JSON Schema property names at {location} are invalid")
        if (
            not isinstance(required, list)
            or any(not isinstance(name, str) for name in required)
            or len(set(required)) != len(required)
            or set(required) != set(properties)
        ):
            raise ProbeError(
                f"ledger JSON Schema object at {location} must require every declared property exactly once"
            )
        if value.get("additionalProperties") is not False:
            raise ProbeError(
                f"ledger JSON Schema object at {location} must set additionalProperties to false"
            )
        for name, child in properties.items():
            _validate_ledger_schema_node(
                child, location=f"{location}.properties[{name!r}]", depth=depth + 1,
                node_count=node_count,
            )
        return
    if schema_type == "array":
        unknown = sorted(set(value) - {"type", "items"})
        if unknown:
            raise ProbeError(
                f"unsupported ledger JSON Schema fields at {location}: {', '.join(unknown)}"
            )
        items = value.get("items")
        if not isinstance(items, dict) or set(items) != {"type"} or items.get("type") not in {
            "string", "integer",
        }:
            raise ProbeError(
                f"ledger JSON Schema array at {location} must have string or integer items"
            )
        _validate_ledger_schema_node(
            items, location=f"{location}.items", depth=depth + 1, node_count=node_count,
        )
        return
    if schema_type in {"string", "integer"}:
        unknown = sorted(set(value) - {"type"})
        if unknown:
            raise ProbeError(
                f"unsupported ledger JSON Schema fields at {location}: {', '.join(unknown)}"
            )
        return
    raise ProbeError(
        f"ledger JSON Schema type at {location} must be object, array, string, or integer"
    )


def load_ledger_json_schema(path: Path) -> LedgerJsonSchema:
    try:
        raw = path.read_bytes()
    except FileNotFoundError as exc:
        raise ProbeError(f"ledger JSON Schema does not exist: {path}") from exc
    except OSError as exc:
        raise ProbeError(f"ledger JSON Schema is not readable: {path}") from exc
    if len(raw) > MAX_LEDGER_SCHEMA_BYTES:
        raise ProbeError(
            f"ledger JSON Schema exceeds {MAX_LEDGER_SCHEMA_BYTES} bytes: {path}"
        )
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ProbeError(f"ledger JSON Schema is not valid UTF-8: {path}") from exc

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ProbeError(f"ledger JSON Schema contains duplicate key: {key}")
            result[key] = value
        return result

    try:
        schema = json.loads(text, object_pairs_hook=reject_duplicate_keys)
    except json.JSONDecodeError as exc:
        raise ProbeError(f"ledger JSON Schema is not readable JSON: {path}") from exc
    _validate_ledger_schema_node(schema, location="$", depth=0, node_count=[0])
    if schema.get("type") != "object":
        raise ProbeError("ledger JSON Schema root must be an object schema")
    return LedgerJsonSchema(path=path, sha256=_sha256_bytes(raw), schema=schema)


def _ledger_schema_controls(value: LedgerJsonSchema) -> dict[str, Any]:
    return {
        "ledger_json_schema_path": str(value.path),
        "ledger_json_schema_sha256": value.sha256,
        "ledger_json_schema_requested": {
            "response_format_type": "json_schema",
            "name": "ledger_output",
            "strict": True,
            "application_evidence": LEDGER_JSON_SCHEMA_APPLICATION_EVIDENCE,
        },
    }


def _finite_number(value: Any, name: str) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProbeError(f"sampler override {name} must be a number")
    if not math.isfinite(float(value)):
        raise ProbeError(f"sampler override {name} must be finite")
    return value


def validate_recipe(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ProbeError("recipe must be a JSON object")
    unknown = sorted(set(value) - RECIPE_TOP_LEVEL_FIELDS)
    if unknown:
        raise ProbeError(f"unknown recipe fields: {', '.join(unknown)}")
    if value.get("schema_version", SCHEMA_VERSION) != SCHEMA_VERSION:
        raise ProbeError(f"recipe schema_version must be {SCHEMA_VERSION}")
    label = value.get("label")
    if not isinstance(label, str) or not label.strip() or len(label) > 160 or "\n" in label:
        raise ProbeError("recipe label must be a non-empty single line of at most 160 characters")
    claim = value.get("claim", "custom_experiment")
    if claim not in {"custom_experiment", "publisher_inspired", "full_allowlisted_recipe"}:
        raise ProbeError("recipe claim must be custom_experiment, publisher_inspired, or full_allowlisted_recipe")
    sources = value.get("sources", [])
    if not isinstance(sources, list) or any(
        not isinstance(item, str) or not item.startswith("https://") for item in sources
    ):
        raise ProbeError("recipe sources must be an array of HTTPS URLs")
    if (claim != "custom_experiment" or "publisher" in label.lower()) and not sources:
        raise ProbeError("publisher recipe labels or claims require at least one HTTPS source")
    overrides = value.get("sampler_overrides")
    if not isinstance(overrides, dict) or not overrides:
        raise ProbeError("recipe sampler_overrides must be a non-empty object")
    unknown_overrides = sorted(set(overrides) - set(SAMPLER_FIELDS))
    if unknown_overrides:
        raise ProbeError(f"unknown sampler overrides: {', '.join(unknown_overrides)}")
    normalized: dict[str, int | float] = {}
    for name, raw in overrides.items():
        number = _finite_number(raw, name)
        number_f = float(number)
        if name == "temperature" and not 0 <= number_f <= 2:
            raise ProbeError("sampler override temperature must be from 0 through 2")
        if name == "top_p" and not 0 < number_f <= 1:
            raise ProbeError("sampler override top_p must be greater than 0 and at most 1")
        if name == "top_k":
            if not isinstance(number, int) or not 0 <= number <= 10_000:
                raise ProbeError("sampler override top_k must be an integer from 0 through 10000")
        if name == "min_p" and not 0 <= number_f <= 1:
            raise ProbeError("sampler override min_p must be from 0 through 1")
        if name == "presence_penalty" and not -2 <= number_f <= 2:
            raise ProbeError("sampler override presence_penalty must be from -2 through 2")
        if name == "repeat_penalty" and not 0 < number_f <= 10:
            raise ProbeError("sampler override repeat_penalty must be greater than 0 and at most 10")
        normalized[name] = number
    complete = set(normalized) == set(SAMPLER_FIELDS)
    implied_full = all(word in label.lower() for word in ("full", "publisher", "recipe"))
    if (claim == "full_allowlisted_recipe" or implied_full) and not complete:
        missing = ", ".join(sorted(set(SAMPLER_FIELDS) - set(normalized)))
        raise ProbeError(f"a full publisher recipe claim requires all allowlisted fields; missing: {missing}")
    notes = value.get("notes")
    if notes is not None and not isinstance(notes, str):
        raise ProbeError("recipe notes must be a string when present")
    return {
        "schema_version": SCHEMA_VERSION,
        "label": label.strip(),
        "claim": claim,
        "sources": list(sources),
        "sampler_overrides": normalized,
        "allowlisted_recipe_complete": complete,
        "unspecified_fields": [item for item in SAMPLER_FIELDS if item not in normalized],
        "unspecified_behavior": "inherit_active_server_default",
        "application_evidence": "requested_in_outgoing_payload_not_independently_verified_effective",
        **({"notes": notes} if notes is not None else {}),
    }


def load_recipe(path: Path) -> tuple[dict[str, Any], str]:
    return validate_recipe(_json_file(path)), _sha256_file(path)


def _csv(value: str | None, *, label: str) -> tuple[str, ...] | None:
    if value is None:
        return None
    values = tuple(dict.fromkeys(item.strip() for item in value.split(",") if item.strip()))
    if not values:
        raise ProbeError(f"{label} must contain at least one value")
    return values


def selected_cases(groups_value: str, cases_value: str | None) -> list[SelectedCase]:
    specialist_cases.assert_frozen_suite()
    groups = _csv(groups_value, label="--groups") or ()
    invalid_groups = sorted(set(groups) - set(GROUPS))
    if invalid_groups:
        raise ProbeError(f"unknown groups: {', '.join(invalid_groups)}")
    requested = _csv(cases_value, label="--cases")
    choices: list[SelectedCase] = []
    if "ledger" in groups:
        ledger = bench_cases.get_cases("hard", [LEDGER_CASE_ID])
        if len(ledger) != 1:
            raise ProbeError("frozen ledger case did not resolve uniquely")
        choices.append(SelectedCase("ledger", ledger[0]))
    for group in ("code", "prose"):
        if group in groups:
            choices.extend(SelectedCase(group, case) for case in specialist_cases.get_cases(groups=(group,)))
    if requested is not None:
        by_id = {item.case.case_id: item for item in choices}
        missing = [case_id for case_id in requested if case_id not in by_id]
        if missing:
            raise ProbeError(f"selected case IDs are unknown or outside --groups: {', '.join(missing)}")
        choices = [by_id[case_id] for case_id in requested]
    if not choices:
        raise ProbeError("no cases selected")
    return choices


def _thinking_for_group(group: str, args: argparse.Namespace) -> str:
    if group == "ledger":
        return "medium"
    if group == "code":
        return getattr(args, "code_thinking", "medium")
    return args.prose_thinking


def _artifact_paths(prefix: Path) -> dict[str, Path]:
    return {
        "requests": Path(f"{prefix}-sampler-requests.jsonl"),
        "manifest": Path(f"{prefix}-sampler-manifest.json"),
        "receipt": Path(f"{prefix}-sampler-receipt.json"),
        "visible_dir": Path(f"{prefix}-sampler-visible-texts"),
    }


def _validate_artifact_paths(paths: Mapping[str, Path], tag: str) -> None:
    prefix_name = paths["requests"].name.removesuffix("-sampler-requests.jsonl")
    if prefix_name != tag and not prefix_name.startswith(tag + "-"):
        raise ProbeError("--output-prefix basename must equal --tag or start with '<tag>-' for lifecycle joins")
    existing = [str(path) for path in paths.values() if path.exists()]
    if existing:
        raise ProbeError("refusing to overwrite existing probe artifacts: " + ", ".join(existing))


def _validate_lifecycle_rows(
    rows: Sequence[Mapping[str, Any]], *, run_id: str, tag: str, model: str,
    expected_context: int = CONTEXT_TOKENS,
) -> dict[str, Any] | None:
    if rows and any(row.get("run_id") != run_id for row in rows):
        raise ProbeError("runtime event ledger does not belong exclusively to the supplied run ID")
    if rows and any(row.get("tag") != tag for row in rows):
        raise ProbeError("runtime event ledger tag does not match --tag")
    starts = [row for row in rows if row.get("event") == "session_started"]
    holds = [row for row in rows if row.get("event") == "holding_started"]
    if len(starts) > 1 or len(holds) > 1:
        raise ProbeError("runtime event ledger has an ambiguous session lifecycle")
    if any(row.get("event") in TERMINAL_EVENTS for row in rows):
        raise ProbeError("resident session is no longer in its active holding lifecycle")
    if starts:
        start = starts[0]
        if start.get("runtime_backend") != "llama":
            raise ProbeError("sampler recipe probe supports only a native llama model_session")
        if start.get("model_name") != model or start.get("context") != expected_context:
            raise ProbeError("runtime session model alias or context does not match this probe")
    if holds:
        if len(starts) != 1:
            raise ProbeError("holding_started has no unique session_started event")
        return {
            "session_started_at": starts[0].get("at"),
            "holding_started_at": holds[0].get("at"),
            "control_file": holds[0].get("control_file"),
            "source_config": starts[0].get("source_config"),
            "runtime_backend": starts[0].get("runtime_backend"),
            "context_tokens": starts[0].get("context"),
        }
    return None


def wait_for_active_hold(
    path: Path, *, run_id: str, tag: str, model: str, timeout: float,
    expected_context: int = CONTEXT_TOKENS, poll_seconds: float = 1.0,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while True:
        lifecycle = _validate_lifecycle_rows(
            budget_prompt_probe._read_event_rows(path), run_id=run_id, tag=tag, model=model,
            expected_context=expected_context,
        )
        if lifecycle is not None:
            return lifecycle
        if time.monotonic() >= deadline:
            raise ProbeError("timed out waiting for the exact active holding_started event")
        time.sleep(min(poll_seconds, max(0.0, deadline - time.monotonic())))


def assert_active_hold(
    path: Path, *, run_id: str, tag: str, model: str,
    expected_context: int = CONTEXT_TOKENS,
) -> None:
    if _validate_lifecycle_rows(
        budget_prompt_probe._read_event_rows(path), run_id=run_id, tag=tag, model=model,
        expected_context=expected_context,
    ) is None:
        raise ProbeError("resident session is not in its active holding lifecycle")


def verify_resident(
    endpoint: str, model: str, expected_context: int = CONTEXT_TOKENS
) -> dict[str, Any]:
    _, origin = budget_prompt_probe._endpoint_parts(endpoint)
    models_base = endpoint.rstrip("/")
    if models_base.endswith("/chat/completions"):
        models_base = models_base[: -len("/chat/completions")]
    models = budget_prompt_probe._request_json("GET", models_base + "/models", label="models")
    model_ids = [
        item.get("id") for item in models.get("data", [])
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    ] if isinstance(models, dict) and isinstance(models.get("data"), list) else []
    if model_ids != [model]:
        raise ProbeError("models response does not contain exactly the supplied resident alias")
    props = budget_prompt_probe._request_json("GET", origin + "/props", label="props")
    settings = props.get("default_generation_settings") if isinstance(props, dict) else None
    context = settings.get("n_ctx") if isinstance(settings, dict) else None
    if context != expected_context or props.get("model_alias") != model:
        raise ProbeError("resident /props alias or context does not match this probe")
    params = settings.get("params") if isinstance(settings, dict) else None
    params = params if isinstance(params, dict) else {}
    defaults = {
        name: {"present": name in params, "value": params.get(name)} for name in SAMPLER_FIELDS
    }
    result = {
        "model_ids": model_ids,
        "model_alias": model,
        "context_tokens": context,
        "expected_context_tokens": expected_context,
        "actual_context_tokens": context,
        "sampler_defaults_before_override": defaults,
    }
    native_budget_defaults = {
        name: params[name]
        for name in ("reasoning_budget_tokens", "thinking_budget_tokens")
        if name in params
    }
    if native_budget_defaults:
        result["native_thinking_budget_props_evidence"] = {
            "reported_default_parameters": native_budget_defaults,
            "evidence_scope": (
                "/props reports server defaults only; it does not prove per-request application "
                "or enforcement"
            ),
        }
    return result


def prompt_feasibility(
    endpoint: str,
    cases: Sequence[SelectedCase],
    output_cap: int,
    expected_context: int = CONTEXT_TOKENS,
) -> dict[str, Any]:
    _, origin = budget_prompt_probe._endpoint_parts(endpoint)
    checks: list[dict[str, Any]] = []
    for selected in cases:
        try:
            raw_tokens = budget_prompt_probe._tokenize_count(origin, selected.case.prompt)
        except budget_prompt_probe.ProbeHTTPError as exc:
            if exc.status in {404, 405, 501}:
                raise ProbeError(
                    "resident endpoint does not support /tokenize; exact prompt-plus-cap preflight is required"
                ) from exc
            raise
        required = raw_tokens + CHAT_TEMPLATE_TOKEN_RESERVE + output_cap
        check = {
            "case_id": selected.case.case_id,
            "group": selected.group,
            "raw_prompt_tokens": raw_tokens,
            "chat_template_token_reserve": CHAT_TEMPLATE_TOKEN_RESERVE,
            "output_token_limit": output_cap,
            "required_context_bound": required,
            "fits": required <= expected_context,
        }
        checks.append(check)
    if not all(item["fits"] for item in checks):
        failed = ", ".join(str(item["case_id"]) for item in checks if not item["fits"])
        raise ProbeError(f"selected prompt plus output cap does not fit verified context: {failed}")
    return {
        "method": "llama.cpp /tokenize raw prompt token IDs",
        "expected_context_tokens": expected_context,
        "actual_context_tokens": expected_context,
        "uncertainty": (
            "The raw /tokenize endpoint excludes the chat template; the bound includes a "
            f"fixed {CHAT_TEMPLATE_TOKEN_RESERVE}-token single-message wrapper reserve."
        ),
        "checks": checks,
    }


@contextmanager
def sampler_payload_overrides(
    overrides: Mapping[str, int | float], *, thinking_budget_tokens: int | None = None,
    ledger_json_schema: Mapping[str, Any] | None = None,
) -> Iterator[None]:
    """Temporarily add only the explicit allowlisted overrides to OpenAI payloads."""
    original = bench.build_openai_payload

    def builder(config: bench.RunConfig, case: Any) -> dict[str, Any]:
        payload = original(config, case)
        # RunConfig requires a temperature, but omission here means inherit the
        # active server default rather than silently creating an experiment knob.
        if "temperature" not in overrides:
            payload.pop("temperature", None)
        for name, value in overrides.items():
            payload[name] = value
        if thinking_budget_tokens is not None:
            payload["thinking_budget_tokens"] = thinking_budget_tokens
        if ledger_json_schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "ledger_output",
                    "strict": True,
                    "schema": ledger_json_schema,
                },
            }
        return payload

    bench.build_openai_payload = builder
    try:
        yield
    finally:
        bench.build_openai_payload = original


def _exclusive_json(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _request_metadata(
    *, args: argparse.Namespace, recipe: Mapping[str, Any], recipe_sha: str,
    case: Any, thinking: str, ledger_schema: LedgerJsonSchema | None,
) -> dict[str, Any]:
    expected_context = getattr(args, "expected_context", CONTEXT_TOKENS)
    overrides = dict(recipe["sampler_overrides"])
    thinking_budget_tokens = getattr(args, "thinking_budget_tokens", None)
    reasoning_controls = {
        "reasoning_effort": thinking,
        "chat_template_thinking": "off" if thinking == "none" else "on",
        "reasoning_budget_tokens": None,
    }
    result = {
        "endpoint": args.endpoint,
        "backend": "openai",
        "model": args.model,
        "profile": args.profile or recipe["label"],
        "thinking": thinking,
        "chat_template_thinking": "off" if thinking == "none" else "on",
        "temperature": overrides.get("temperature"),
        "seed": SEED,
        "context_max_tokens": expected_context,
        "expected_context_tokens": expected_context,
        "actual_context_tokens": expected_context,
        "output_token_limit": args.output_cap,
        "timeout_seconds": args.timeout,
        "prompt_chars": len(case.prompt),
        "prompt_sha256": _sha256_text(case.prompt),
        "sampler_overrides": overrides,
        "sampler_recipe_label": recipe["label"],
        "sampler_recipe_sha256": recipe_sha,
        "sampler_recipe_sources": list(recipe["sources"]),
        "sampler_recipe_claim": recipe["claim"],
        "sampler_unspecified_fields": list(recipe["unspecified_fields"]),
        "sampler_application_evidence": recipe["application_evidence"],
        "reasoning_controls_requested": reasoning_controls,
        "capability_note": (
            "Native llama request controls were requested in the payload; sampler effectiveness "
            "and model-specific reasoning behavior were not independently verified."
        ),
    }
    if thinking_budget_tokens is not None:
        reasoning_controls["thinking_budget_tokens"] = thinking_budget_tokens
        result.update({
            "native_thinking_budget_tokens_requested": thinking_budget_tokens,
            "native_thinking_budget_application_evidence": (
                NATIVE_THINKING_BUDGET_APPLICATION_EVIDENCE
            ),
            "capability_note": (
                "Native llama request controls, including thinking_budget_tokens, were requested "
                "in the payload; endpoint acceptance does not establish budget enforcement."
            ),
        })
    if ledger_schema is not None:
        result.update(_ledger_schema_controls(ledger_schema))
        result["capability_note"] += (
            " A strict JSON Schema response format was also requested; endpoint acceptance does "
            "not establish schema enforcement or equivalent reasoning behavior."
        )
    return result


def _error(exc: BaseException) -> dict[str, str]:
    return {"type": type(exc).__name__, "message": str(exc)[:500]}


def _safe_response(call: Mapping[str, Any], *, include_visible_text: bool) -> dict[str, Any]:
    forbidden = {"reasoning_text", "reasoning_content", "analysis_text", "raw_reasoning"}
    return {
        key: value for key, value in call.items()
        if key not in forbidden and (include_visible_text or key != "response_text")
    }


def _case_manifest(selected: SelectedCase) -> dict[str, Any]:
    case = selected.case
    if selected.group == "ledger":
        return {
            "case_id": case.case_id, "group": "ledger", "case_version": bench_cases.CASE_VERSION,
            "prompt_sha256": _sha256_text(case.prompt), "evaluator": case.evaluator,
            "independent_task_count": 1,
        }
    return {
        "case_id": case.case_id, "group": selected.group,
        "case_version": specialist_cases.CASE_VERSION,
        "prompt_sha256": _sha256_text(case.prompt),
        "case_sha256": specialist_cases.case_sha256(case),
        "evaluator": "specialist_cases.evaluate_case",
        "independent_task_count": 1 if selected.group == "code" else 0,
        "objective_count": len(case.code_objectives),
        "assertion_count": sum(len(item.assertions) for item in case.code_objectives),
        "rubric_max_points": sum(item.max_points for item in case.rubric),
    }


def _base_row(
    selected: SelectedCase, *, args: argparse.Namespace, recipe: Mapping[str, Any],
    recipe_sha: str, probe_run_id: str, model_session_run_id: str, call_id: str,
    started_at: str, call: Mapping[str, Any] | None, error: Mapping[str, Any] | None,
    visible_dir: Path, ledger_schema: LedgerJsonSchema | None,
) -> dict[str, Any]:
    case = selected.case
    thinking = _thinking_for_group(selected.group, args)
    request = _request_metadata(
        args=args, recipe=recipe, recipe_sha=recipe_sha, case=case, thinking=thinking,
        ledger_schema=ledger_schema,
    )
    if selected.group == "ledger":
        evaluation = (
            bench_cases.evaluate_case(case, str(call.get("response_text", "")), truncated=bool(call.get("truncated")))
            if call is not None else {"passed": False, "detail": "request failed"}
        )
        return {
            "schema_version": SCHEMA_VERSION, "run_id": probe_run_id, "call_id": call_id,
            "started_at": started_at, "finished_at": bench.utc_now(), "repeat_index": 0,
            "tag": args.tag, "suite": case.suite, "case_id": case.case_id,
            "case_version": bench_cases.CASE_VERSION,
            "case_source_sha256": _sha256_file(Path(bench_cases.__file__)),
            "case_metadata": dict(case.metadata) | {
                "diagnostic_kind": "resident_sampler_recipe_probe",
                "model_session_run_id": model_session_run_id,
            },
            "request": request, "telemetry_before": None, "telemetry_after": None,
            "response": _safe_response(call, include_visible_text=True) if call is not None else None,
            "evaluation": evaluation, "error": error,
        }

    text = str(call.get("response_text", "")) if call is not None else ""
    visible = specialist_bench._write_visible_text(visible_dir, call_id, text)
    if call is None:
        evaluation = {
            "outcome": "timeout" if error and error.get("type") == "TimeoutError" else "request_failure",
            "passed": False, "detail": "request failed before a complete response was available",
        }
        metrics = None
    else:
        evaluation = specialist_cases.evaluate_case(case, text, truncated=bool(call.get("truncated")))
        metrics = specialist_bench.response_metrics(call)
    response = _safe_response(call, include_visible_text=False) if call else None
    return {
        "schema_version": SCHEMA_VERSION, "run_id": probe_run_id, "call_id": call_id,
        "started_at": started_at, "finished_at": bench.utc_now(), "repeat_index": 0,
        "tag": args.tag, "model": args.model, "profile": args.profile or recipe["label"],
        "suite": f"specialist_{selected.group}", "case_id": case.case_id,
        "case_group": selected.group, "case_category": case.category,
        "case_version": specialist_cases.CASE_VERSION,
        "case_sha256": specialist_cases.case_sha256(case),
        "suite_sha256": specialist_cases.FROZEN_SUITE_SHA256,
        "case_source_sha256": _sha256_file(Path(specialist_cases.__file__)),
        "request": request, "response": response, "response_metrics": metrics,
        "visible_text": visible, "objective_outcome": evaluation.get("outcome"),
        "objective_error": error, "error": error, "evaluation": evaluation,
        "eligibility": specialist_bench._eligibility(case, call, evaluation, error),
    }


def _summary(row: Mapping[str, Any]) -> dict[str, Any]:
    request = row.get("request") if isinstance(row.get("request"), Mapping) else {}
    response = row.get("response") if isinstance(row.get("response"), Mapping) else {}
    metrics = row.get("response_metrics") if isinstance(row.get("response_metrics"), Mapping) else {}
    timing = response.get("timing") if isinstance(response.get("timing"), Mapping) else {}
    throughput = response.get("throughput") if isinstance(response.get("throughput"), Mapping) else {}
    native = response.get("openai_native_timings") if isinstance(response.get("openai_native_timings"), Mapping) else {}
    evaluation = row.get("evaluation") if isinstance(row.get("evaluation"), Mapping) else {}
    error = row.get("error") if isinstance(row.get("error"), Mapping) else None
    outcome = row.get("objective_outcome")
    if not isinstance(outcome, str):
        if error is not None:
            outcome = "timeout" if error.get("type") == "TimeoutError" else "request_failure"
        elif response.get("truncated") is True:
            outcome = "truncated"
        else:
            outcome = "passed" if evaluation.get("passed") is True else "wrong_answer"
    result = {
        "call_id": row.get("call_id"), "case_id": row.get("case_id"),
        "case_group": row.get("case_group") or "ledger", "outcome": outcome,
        "passed": evaluation.get("passed") if isinstance(evaluation.get("passed"), bool) else None,
        "thinking": request.get("thinking"), "output_token_limit": request.get("output_token_limit"),
        "timeout_seconds": request.get("timeout_seconds"),
        "stream_complete": response.get("stream_complete"), "truncated": response.get("truncated"),
        "input_tokens": metrics.get("input_tokens_reported", response.get("actual_prompt_tokens")),
        "output_tokens": metrics.get("output_tokens_reported", response.get("actual_completion_tokens")),
        "wall_seconds": metrics.get("wall_seconds", timing.get("wall_seconds")),
        "native_decode_seconds": metrics.get(
            "native_decode_seconds",
            float(native["predicted_ms"]) / 1000
            if isinstance(native.get("predicted_ms"), (int, float)) and not isinstance(native.get("predicted_ms"), bool)
            else None,
        ),
        "native_decode_tps": metrics.get(
            "native_decode_rate_tokens_per_second", throughput.get("completion_tokens_per_second_native"),
        ),
        "error_type": error.get("type") if error is not None else None,
    }
    if "native_thinking_budget_tokens_requested" in request:
        result.update({
            "native_thinking_budget_tokens_requested": request.get(
                "native_thinking_budget_tokens_requested"
            ),
            "native_thinking_budget_application_evidence": request.get(
                "native_thinking_budget_application_evidence"
            ),
        })
    if "ledger_json_schema_sha256" in request:
        result.update({
            "ledger_json_schema_path": request.get("ledger_json_schema_path"),
            "ledger_json_schema_sha256": request.get("ledger_json_schema_sha256"),
            "ledger_json_schema_requested": request.get("ledger_json_schema_requested"),
        })
    return result


def _bounded_cap(value: str) -> int:
    parsed = int(value)
    if not 1 <= parsed <= MAX_OUTPUT_CAP:
        raise argparse.ArgumentTypeError(f"must be from 1 through {MAX_OUTPUT_CAP}")
    return parsed


def _bounded_timeout(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or not 0 < parsed <= MAX_TIMEOUT_SECONDS:
        raise argparse.ArgumentTypeError(f"must be finite, positive, and at most {MAX_TIMEOUT_SECONDS:g}")
    return parsed


def _bounded_context(value: str) -> int:
    parsed = int(value)
    if not 1 <= parsed <= MAX_CONTEXT_TOKENS:
        raise argparse.ArgumentTypeError(f"must be from 1 through {MAX_CONTEXT_TOKENS}")
    return parsed


def _positive_thinking_budget(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _validate_thinking_budget(thinking_budget_tokens: Any, output_cap: int) -> int | None:
    if thinking_budget_tokens is None:
        return None
    if isinstance(thinking_budget_tokens, bool) or not isinstance(thinking_budget_tokens, int):
        raise ProbeError("--thinking-budget-tokens must be a positive integer")
    if thinking_budget_tokens <= 0:
        raise ProbeError("--thinking-budget-tokens must be a positive integer")
    if thinking_budget_tokens > output_cap:
        raise ProbeError("--thinking-budget-tokens must not exceed --output-cap")
    return thinking_budget_tokens


def run(args: argparse.Namespace) -> int:
    expected_context = getattr(args, "expected_context", CONTEXT_TOKENS)
    if (
        isinstance(expected_context, bool)
        or not isinstance(expected_context, int)
        or not 1 <= expected_context <= MAX_CONTEXT_TOKENS
    ):
        raise ProbeError(
            f"--expected-context must be from 1 through {MAX_CONTEXT_TOKENS}"
        )
    code_thinking = getattr(args, "code_thinking", "medium")
    if code_thinking not in ("none", "medium"):
        raise ProbeError("--code-thinking must be none or medium")
    thinking_budget_tokens = _validate_thinking_budget(
        getattr(args, "thinking_budget_tokens", None), args.output_cap,
    )
    ledger_schema_value = getattr(args, "ledger_json_schema", None)
    ledger_schema = (
        load_ledger_json_schema(Path(ledger_schema_value).expanduser().resolve())
        if ledger_schema_value is not None
        else None
    )
    recipe_path = Path(args.recipe).expanduser().resolve()
    recipe, recipe_sha = load_recipe(recipe_path)
    cases = selected_cases(args.groups, args.cases)
    if ledger_schema is not None and (
        len(cases) != 1 or cases[0].group != "ledger" or cases[0].case.case_id != LEDGER_CASE_ID
    ):
        raise ProbeError("--ledger-json-schema requires a ledger-only selection")
    if any(item.group == "prose" for item in cases) and args.prose_thinking is None:
        raise ProbeError("--prose-thinking is required when prose cases are selected")
    prefix = Path(args.output_prefix).expanduser().resolve()
    paths = _artifact_paths(prefix)
    _validate_artifact_paths(paths, args.tag)
    lifecycle_path = Path(args.runtime_events).expanduser().resolve()
    lifecycle = wait_for_active_hold(
        lifecycle_path, run_id=args.model_session_run_id, tag=args.tag, model=args.model,
        timeout=args.holding_timeout, expected_context=expected_context,
    )
    resident = verify_resident(args.endpoint, args.model, expected_context)
    feasibility = prompt_feasibility(
        args.endpoint, cases, args.output_cap, expected_context
    )
    assert_active_hold(
        lifecycle_path, run_id=args.model_session_run_id, tag=args.tag, model=args.model,
        expected_context=expected_context,
    )

    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    probe_run_id = str(uuid.uuid4())
    controls = {
        "context_max_tokens": expected_context,
        "expected_context_tokens": expected_context,
        "actual_context_tokens": resident["actual_context_tokens"],
        "output_token_limit": args.output_cap,
        "timeout_seconds": args.timeout, "seed": SEED,
        "prose_thinking": args.prose_thinking,
    }
    if code_thinking == "medium":
        controls["ledger_and_code_thinking"] = "medium"
    else:
        controls.update({"ledger_thinking": "medium", "code_thinking": "none"})
    manifest = {
        "schema_version": SCHEMA_VERSION, "created_at": bench.utc_now(),
        "diagnostic_kind": "resident_sampler_recipe_probe", "run_id": probe_run_id,
        "tag": args.tag, "model_session_run_id": args.model_session_run_id,
        "recipe": recipe | {"path": str(recipe_path), "sha256": recipe_sha},
        "sources": {
            "sampler_recipe_probe.py": _sha256_file(Path(__file__)),
            "bench.py": _sha256_file(Path(bench.__file__)),
            "bench_cases.py": _sha256_file(Path(bench_cases.__file__)),
            "specialist_bench.py": _sha256_file(Path(specialist_bench.__file__)),
            "specialist_cases.py": _sha256_file(Path(specialist_cases.__file__)),
        },
        "controls": controls,
        "case_count": len(cases),
        "independent_task_count": sum(item.group in {"ledger", "code"} for item in cases),
        "prose_prompt_count": sum(item.group == "prose" for item in cases),
        "cases": [_case_manifest(item) for item in cases],
        "prompt_feasibility": feasibility,
    }
    if thinking_budget_tokens is not None:
        manifest["controls"].update({
            "native_thinking_budget_tokens_requested": thinking_budget_tokens,
            "native_thinking_budget_application_evidence": (
                NATIVE_THINKING_BUDGET_APPLICATION_EVIDENCE
            ),
        })
    if ledger_schema is not None:
        manifest["controls"].update(_ledger_schema_controls(ledger_schema))
    _exclusive_json(paths["manifest"], manifest)

    receipt: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION, "started_at": bench.utc_now(), "status": "running",
        "diagnostic_kind": "resident_sampler_recipe_probe", "run_id": probe_run_id,
        "tag": args.tag, "model": args.model, "model_session_run_id": args.model_session_run_id,
        "runtime_backend": "llama", "lifecycle": lifecycle, "resident": resident,
        "recipe": recipe | {"path": str(recipe_path), "sha256": recipe_sha},
        "controls": manifest["controls"], "case_count": len(cases),
        "independent_task_count": manifest["independent_task_count"],
        "prose_prompt_count": manifest["prose_prompt_count"], "attempted_count": 0,
        "completed_count": 0, "truncated_count": 0, "request_failure_count": 0,
        "artifacts": {key: str(value) for key, value in paths.items()}, "attempts": [],
    }
    interrupted: BaseException | None = None
    try:
        with paths["requests"].open("x", encoding="utf-8", buffering=1) as handle:
            with sampler_payload_overrides(
                recipe["sampler_overrides"], thinking_budget_tokens=thinking_budget_tokens,
                ledger_json_schema=ledger_schema.schema if ledger_schema is not None else None,
            ):
                for selected in cases:
                    assert_active_hold(
                        lifecycle_path, run_id=args.model_session_run_id, tag=args.tag,
                        model=args.model, expected_context=expected_context,
                    )
                    verify_resident(args.endpoint, args.model, expected_context)
                    thinking = _thinking_for_group(selected.group, args)
                    config = bench.RunConfig(
                        endpoint=args.endpoint, backend="openai", model=args.model,
                        context=expected_context, thinking=thinking,
                        temperature=float(recipe["sampler_overrides"].get("temperature", 0.0)),
                        seed=SEED, num_predict=args.output_cap, timeout=args.timeout,
                        chat_template_thinking="off" if thinking == "none" else "on",
                    )
                    call_id = str(uuid.uuid4())
                    started_at = bench.utc_now()
                    call: dict[str, Any] | None = None
                    request_error: dict[str, Any] | None = None
                    receipt["attempted_count"] += 1
                    receipt["in_flight_case"] = {
                        "case_id": selected.case.case_id, "case_group": selected.group,
                        "call_id": call_id, "started_at": started_at,
                    }
                    try:
                        call = bench.run_request(config, selected.case)
                    except (bench.BenchmarkHTTPError, TimeoutError, OSError, ValueError) as exc:
                        request_error = _error(exc)
                    row = _base_row(
                        selected, args=args, recipe=recipe, recipe_sha=recipe_sha,
                        probe_run_id=probe_run_id, model_session_run_id=args.model_session_run_id,
                        call_id=call_id, started_at=started_at, call=call, error=request_error,
                        visible_dir=paths["visible_dir"], ledger_schema=ledger_schema,
                    )
                    bench._append_jsonl(handle, row)
                    summary = _summary(row)
                    receipt["attempts"].append(summary)
                    receipt.pop("in_flight_case", None)
                    if summary["stream_complete"] is True and summary["truncated"] is not True:
                        receipt["completed_count"] += 1
                    if summary["truncated"] is True:
                        receipt["truncated_count"] += 1
                    if request_error is not None:
                        receipt["request_failure_count"] += 1
                    print(
                        f"RECORDED {selected.case.case_id} call_id={call_id}", file=sys.stderr, flush=True,
                    )
            assert_active_hold(
                lifecycle_path, run_id=args.model_session_run_id, tag=args.tag, model=args.model,
                expected_context=expected_context,
            )
    except BaseException as exc:
        receipt["status"] = "interrupted"
        receipt["interruption"] = _error(exc)
        interrupted = exc
    else:
        receipt["status"] = (
            "complete" if receipt["request_failure_count"] == 0 else "completed_with_request_failures"
        )
    finally:
        receipt["finished_at"] = bench.utc_now()
        _exclusive_json(paths["receipt"], receipt)
    if interrupted is not None:
        raise interrupted
    return 1 if receipt["request_failure_count"] else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", required=True, help="JSON sampler recipe")
    parser.add_argument("--runtime-events", required=True, help="exact active model_session event ledger")
    parser.add_argument("--tag", required=True, help="exact active model_session tag")
    parser.add_argument("--model-session-run-id", required=True, help="exact active model_session run ID")
    parser.add_argument("--endpoint", required=True, help="loopback native llama endpoint")
    parser.add_argument("--model", required=True, help="exact resident model alias")
    parser.add_argument(
        "--expected-context",
        type=_bounded_context,
        default=CONTEXT_TOKENS,
        help="exact context expected in model_session lifecycle and llama /props",
    )
    parser.add_argument("--output-prefix", required=True, help="unique artifact prefix beginning with --tag")
    parser.add_argument("--profile", help="optional study profile; defaults to recipe label")
    parser.add_argument("--groups", default="ledger,code,prose", help="comma-separated ledger/code/prose")
    parser.add_argument("--cases", help="optional comma-separated case IDs within selected groups")
    parser.add_argument(
        "--prose-thinking", choices=("none", "low", "medium"),
        help="required when prose is selected; ledger always uses medium",
    )
    parser.add_argument(
        "--code-thinking", choices=("none", "medium"), default="medium",
        help="native code reasoning/template mode (default: medium)",
    )
    parser.add_argument("--output-cap", type=_bounded_cap, default=DEFAULT_OUTPUT_CAP)
    parser.add_argument(
        "--thinking-budget-tokens", type=_positive_thinking_budget,
        help=(
            "optional positive native llama per-request thinking budget; must not exceed "
            "--output-cap"
        ),
    )
    parser.add_argument(
        "--ledger-json-schema",
        help="optional strict output-shape JSON Schema; allowed only with a ledger-only selection",
    )
    parser.add_argument("--timeout", type=_bounded_timeout, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--holding-timeout", type=bench.positive_float, default=60.0)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        return run(args)
    except (ProbeError, budget_prompt_probe.ProbeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
