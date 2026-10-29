#!/usr/bin/env python3
"""Opt-in resident-server probe for ledger output budget and prompt sensitivity.

This runner never manages a server.  It binds itself to an already-held
``model_session`` run, executes three required serial requests and an optional
fourth sampling request, and emits ordinary benchmark JSONL plus a prompt
manifest and a compact diagnostic receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import math
import os
import sys
import time
import uuid
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import SplitResult, urlsplit

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import bench
import bench_cases
from bench_cases import BenchmarkCase


CONTEXT_TOKENS = 16_384
TIMEOUT_SECONDS = 2_400.0
THINKING = "medium"
TEMPERATURE = 0.0
SEED = 0
PARENT_CASE_ID = "hard_transaction_ledger"
TIGHTER_CASE_ID = "hard_transaction_ledger_budget_prompt_tighter"
TIGHTER_CASE_VERSION = f"{bench_cases.CASE_VERSION}.budget-prompt-probe.1"
INSTRUCTION_SUFFIX = (
    "\n\nComplete only the specified ledger operations. Do not explore unrelated "
    "alternatives or repeat analysis. Keep the final answer concise and return "
    "the requested JSON object only."
)
CHAT_TEMPLATE_TOKEN_RESERVE = 256
HTTP_MAX_BYTES = 2 * 1024 * 1024
MODEL_CARD_URL = "https://huggingface.co/LiquidAI/LFM2.5-24B-A2B-GGUF"
RECOMMENDED_TEMPERATURE = 0.1
RECOMMENDED_TOP_K = 50
RECOMMENDED_REPEAT_PENALTY = 1.05

CELLS = (
    ("original_cap12288", "original", 12_288, TEMPERATURE, "shared_temperature_0"),
    ("tighter_cap8192", "tighter", 8_192, TEMPERATURE, "shared_temperature_0"),
    ("tighter_cap12288", "tighter", 12_288, TEMPERATURE, "shared_temperature_0"),
)
RECOMMENDED_TEMPERATURE_CELL = (
    "original_cap8192_recommended_temperature",
    "original",
    8_192,
    RECOMMENDED_TEMPERATURE,
    "liquidai_recommended_temperature_with_verified_server_defaults",
)


class ProbeError(RuntimeError):
    """A fail-closed precondition or request failure."""


class ProbeHTTPError(ProbeError):
    def __init__(self, status: int, label: str) -> None:
        super().__init__(f"{label} returned HTTP {status}")
        self.status = status


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def original_case() -> BenchmarkCase:
    cases = bench_cases.get_cases("hard", [PARENT_CASE_ID])
    if len(cases) != 1:
        raise ProbeError("the frozen ledger case did not resolve uniquely")
    return cases[0]


def build_variants() -> tuple[BenchmarkCase, BenchmarkCase]:
    original = original_case()
    tighter = replace(
        original,
        case_id=TIGHTER_CASE_ID,
        prompt=original.prompt + INSTRUCTION_SUFFIX,
    )
    # The variant may change only instructions and identity.  It deliberately
    # shares the exact frozen expected value and evaluator with its parent.
    if tighter.expected is not original.expected or tighter.evaluator != original.evaluator:
        raise ProbeError("the tighter variant changed the frozen evaluator or expected value")
    return original, tighter


def artifact_paths(prefix: Path) -> dict[str, Path]:
    return {
        "requests": Path(f"{prefix}-requests.jsonl"),
        "prompt_manifest": Path(f"{prefix}-prompt-manifest.json"),
        "receipt": Path(f"{prefix}-receipt.json"),
    }


def _endpoint_parts(endpoint: str) -> tuple[SplitResult, str]:
    parsed = urlsplit(endpoint)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
        raise ProbeError("--endpoint must be an absolute HTTP(S) URL without user information")
    hostname = parsed.hostname.lower()
    loopback = hostname == "localhost"
    if not loopback:
        try:
            import ipaddress

            loopback = ipaddress.ip_address(hostname).is_loopback
        except ValueError:
            loopback = False
    if not loopback:
        raise ProbeError("--endpoint must use a literal loopback address or localhost")
    if parsed.query or parsed.fragment:
        raise ProbeError("--endpoint must not contain a query or fragment")
    if parsed.path.rstrip("/") not in {"", "/v1", "/v1/chat/completions"}:
        raise ProbeError("--endpoint path must be the server root, /v1, or /v1/chat/completions")
    port = f":{parsed.port}" if parsed.port is not None else ""
    host = f"[{parsed.hostname}]" if ":" in parsed.hostname else parsed.hostname
    return parsed, f"{parsed.scheme}://{host}{port}"


def _connection(parsed: SplitResult, timeout: float) -> http.client.HTTPConnection:
    if parsed.scheme == "http":
        return http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=timeout)
    return http.client.HTTPSConnection(parsed.hostname, parsed.port, timeout=timeout)


def _request_json(
    method: str,
    url: str,
    *,
    payload: Mapping[str, Any] | None = None,
    timeout: float = 10.0,
    label: str,
) -> Any:
    parsed = urlsplit(url)
    body = None if payload is None else json.dumps(payload, separators=(",", ":")).encode("utf-8")
    headers = {"Accept": "application/json"}
    if body is not None:
        headers["Content-Type"] = "application/json"
    connection = _connection(parsed, timeout)
    try:
        connection.request(method, parsed.path or "/", body=body, headers=headers)
        response = connection.getresponse()
        if response.status < 200 or response.status >= 300:
            response.read(4096)
            raise ProbeHTTPError(response.status, label)
        raw = response.read(HTTP_MAX_BYTES + 1)
        if len(raw) > HTTP_MAX_BYTES:
            raise ProbeError(f"{label} response exceeded {HTTP_MAX_BYTES} bytes")
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ProbeError(f"{label} returned invalid JSON") from exc
    except (OSError, TimeoutError) as exc:
        raise ProbeError(f"{label} request failed: {type(exc).__name__}") from exc
    finally:
        connection.close()
    return value


def verify_server_identity(
    endpoint: str,
    model: str,
    *,
    require_recommended_defaults: bool = False,
) -> dict[str, Any]:
    _, origin = _endpoint_parts(endpoint)
    models_base = endpoint.rstrip("/")
    if models_base.endswith("/chat/completions"):
        models_base = models_base[: -len("/chat/completions")]
    models = _request_json("GET", models_base + "/models", label="models")
    if not isinstance(models, dict) or not isinstance(models.get("data"), list):
        raise ProbeError("models response has no data array")
    model_ids = [
        item.get("id")
        for item in models["data"]
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    ]
    if model_ids != [model]:
        raise ProbeError("models response does not contain exactly the supplied resident alias")

    props = _request_json("GET", origin + "/props", label="props")
    if not isinstance(props, dict):
        raise ProbeError("props response is not an object")
    settings = props.get("default_generation_settings")
    context = settings.get("n_ctx") if isinstance(settings, dict) else None
    if context != CONTEXT_TOKENS:
        raise ProbeError(f"resident context is {context!r}, expected {CONTEXT_TOKENS}")
    if props.get("model_alias") != model:
        raise ProbeError("props model_alias does not match the supplied resident alias")
    params = settings.get("params") if isinstance(settings, dict) else None
    top_k = params.get("top_k") if isinstance(params, dict) else None
    repeat_penalty = params.get("repeat_penalty") if isinstance(params, dict) else None
    defaults_match = (
        isinstance(top_k, (int, float))
        and not isinstance(top_k, bool)
        and float(top_k) == RECOMMENDED_TOP_K
        and isinstance(repeat_penalty, (int, float))
        and not isinstance(repeat_penalty, bool)
        and math.isfinite(float(repeat_penalty))
        and math.isclose(float(repeat_penalty), RECOMMENDED_REPEAT_PENALTY, rel_tol=0.0, abs_tol=1e-6)
    )
    if require_recommended_defaults and not defaults_match:
        raise ProbeError(
            "optional recommended-temperature cell requires active /props defaults "
            "top_k=50 and repeat_penalty=1.05"
        )
    return {
        "model_ids": model_ids,
        "model_alias": model,
        "context_tokens": context,
        "server_sampling_defaults": {"top_k": top_k, "repeat_penalty": repeat_penalty},
        "recommended_sampling_defaults_match": defaults_match,
    }


def _read_event_rows(path: Path) -> list[dict[str, Any]]:
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return []
    lines = raw.splitlines()
    # The model-session process appends concurrently.  Ignore only an
    # unterminated final fragment; a completed malformed line remains fatal.
    if raw and not raw.endswith("\n"):
        lines = lines[:-1]
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ProbeError(f"runtime event line {line_number} is invalid JSON") from exc
        if not isinstance(row, dict):
            raise ProbeError(f"runtime event line {line_number} is not an object")
        rows.append(row)
    return rows


def wait_for_holding_session(
    path: Path,
    *,
    expected_run_id: str,
    expected_tag: str,
    expected_model: str,
    timeout: float,
    poll_seconds: float = 1.0,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while True:
        rows = _read_event_rows(path)
        if rows:
            if any(row.get("run_id") != expected_run_id for row in rows):
                raise ProbeError("runtime event ledger does not belong exclusively to the supplied run ID")
            if any(row.get("tag") != expected_tag for row in rows):
                raise ProbeError("runtime event ledger tag does not match --tag")
            starts = [row for row in rows if row.get("event") == "session_started"]
            if len(starts) > 1:
                raise ProbeError("runtime event ledger contains multiple session_started events")
            if starts:
                start = starts[0]
                if start.get("model_name") != expected_model or start.get("context") != CONTEXT_TOKENS:
                    raise ProbeError("runtime session model alias or context does not match this probe")
            holdings = [row for row in rows if row.get("event") == "holding_started"]
            if holdings:
                if len(starts) != 1 or len(holdings) != 1:
                    raise ProbeError("runtime event ledger has an ambiguous holding lifecycle")
                return {
                    "session_started_at": starts[0].get("at"),
                    "holding_started_at": holdings[0].get("at"),
                    "control_file": holdings[0].get("control_file"),
                }
            if any(
                row.get("event") in {"session_failed", "session_interrupted", "server_stopped", "session_complete"}
                for row in rows
            ):
                raise ProbeError("owned session ended before holding_started")
        if time.monotonic() >= deadline:
            raise ProbeError("timed out waiting for the exact owned holding_started event")
        time.sleep(min(poll_seconds, max(0.0, deadline - time.monotonic())))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ProbeError(f"cannot read comparator JSONL: {type(exc).__name__}") from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ProbeError(f"comparator line {line_number} is invalid JSON") from exc
        if not isinstance(value, dict):
            raise ProbeError(f"comparator line {line_number} is not an object")
        rows.append(value)
    return rows


def load_comparator(path: Path, call_id: str, model: str, parent: BenchmarkCase) -> dict[str, Any]:
    matches = [row for row in _load_jsonl(path) if row.get("call_id") == call_id]
    if len(matches) != 1:
        raise ProbeError("comparator call ID did not resolve to exactly one row")
    row = matches[0]
    request = row.get("request") if isinstance(row.get("request"), dict) else {}
    expected_controls = {
        "backend": "openai",
        "model": model,
        "context_max_tokens": CONTEXT_TOKENS,
        "thinking": THINKING,
        "temperature": TEMPERATURE,
        "seed": SEED,
        "output_token_limit": 8_192,
        "prompt_sha256": sha256_text(parent.prompt),
    }
    if row.get("case_id") != PARENT_CASE_ID or row.get("case_version") != bench_cases.CASE_VERSION:
        raise ProbeError("comparator is not the frozen original ledger case/version")
    for key, expected in expected_controls.items():
        if request.get(key) != expected:
            raise ProbeError(f"comparator request field {key} does not match the diagnostic controls")
    if request.get("chat_template_thinking") not in (None, "on"):
        raise ProbeError("comparator disabled the thinking chat template")
    return row


def _tokenize_count(origin: str, prompt: str) -> int:
    payload = _request_json(
        "POST",
        origin + "/tokenize",
        payload={"content": prompt, "add_special": True, "with_pieces": False},
        label="tokenize",
    )
    tokens = payload.get("tokens") if isinstance(payload, dict) else None
    if not isinstance(tokens, list) or any(isinstance(token, bool) or not isinstance(token, int) for token in tokens):
        raise ProbeError("tokenize response has no integer tokens array")
    return len(tokens)


def prompt_feasibility(
    endpoint: str,
    original: BenchmarkCase,
    tighter: BenchmarkCase,
    cells: Sequence[tuple[str, str, int, float, str]],
) -> dict[str, Any]:
    _, origin = _endpoint_parts(endpoint)
    uncertainty = (
        "The raw /tokenize endpoint does not apply the chat template; a 256-token reserve "
        "covers the single-message wrapper."
    )
    try:
        original_tokens = _tokenize_count(origin, original.prompt)
        tighter_tokens = _tokenize_count(origin, tighter.prompt)
        method = "llama.cpp /tokenize raw prompt token IDs"
        original_bound = original_tokens
        tighter_bound = tighter_tokens
        suffix_estimate = tighter_tokens - original_tokens
    except ProbeHTTPError as exc:
        if exc.status not in {404, 405, 501}:
            raise
        if not original.prompt.isascii() or not tighter.prompt.isascii():
            raise ProbeError("tokenize is unavailable and prompts are not ASCII for byte upper bounds")
        original_tokens = None
        tighter_tokens = None
        original_bound = len(original.prompt.encode("ascii"))
        tighter_bound = len(tighter.prompt.encode("ascii"))
        suffix_estimate = len(INSTRUCTION_SUFFIX.encode("ascii"))
        method = "conservative ASCII-byte token upper bound"
        uncertainty = (
            "The endpoint does not expose /tokenize. One token per ASCII byte is used as a "
            "conservative raw-prompt upper bound, plus a 256-token chat-template reserve."
        )

    checks = []
    for cell_id, variant, cap, temperature, sampling_variant in cells:
        prompt_bound = original_bound if variant == "original" else tighter_bound
        required = prompt_bound + CHAT_TEMPLATE_TOKEN_RESERVE + cap
        fits = required <= CONTEXT_TOKENS
        checks.append(
            {
                "cell_id": cell_id,
                "output_token_limit": cap,
                "temperature": temperature,
                "sampling_variant": sampling_variant,
                "prompt_token_bound": prompt_bound,
                "chat_template_token_reserve": CHAT_TEMPLATE_TOKEN_RESERVE,
                "required_context_bound": required,
                "fits": fits,
            }
        )
    if not all(check["fits"] for check in checks):
        raise ProbeError("a diagnostic cell does not fit the verified 16384-token context")
    return {
        "method": method,
        "original_raw_prompt_tokens": original_tokens,
        "tighter_raw_prompt_tokens": tighter_tokens,
        "tighter_suffix_token_estimate": suffix_estimate,
        "original_prompt_token_bound": original_bound,
        "tighter_prompt_token_bound": tighter_bound,
        "uncertainty": uncertainty,
        "checks": checks,
    }


def _exclusive_json(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _append_jsonl(handle: Any, value: Mapping[str, Any]) -> None:
    handle.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def _request_record(
    config: bench.RunConfig,
    case: BenchmarkCase,
    *,
    case_version: str,
    cell_id: str,
    variant: str,
    sampling_variant: str,
    probe_run_id: str,
    model_session_run_id: str,
    comparator_path: Path,
    comparator_call_id: str,
    source_hash: str,
    call: Mapping[str, Any] | None,
    error: BaseException | None,
) -> dict[str, Any]:
    parent_hash = sha256_text(original_case().prompt)
    metadata = dict(case.metadata)
    metadata.update(
        {
            "diagnostic_kind": "resident_ledger_budget_prompt_probe",
            "diagnostic_cell_id": cell_id,
            "cell_variant": cell_id,
            "diagnostic_variant": variant,
            "prompt_variant": variant,
            "sampling_variant": sampling_variant,
            "underlying_problem_id": PARENT_CASE_ID,
            "parent_case_id": PARENT_CASE_ID,
            "parent_case_version": bench_cases.CASE_VERSION,
            "parent_prompt_sha256": parent_hash,
            "instruction_suffix_sha256": sha256_text(INSTRUCTION_SUFFIX) if variant == "tighter" else None,
            "model_session_run_id": model_session_run_id,
            "historical_comparator_call_id": comparator_call_id,
        }
    )
    request = {
        "endpoint": config.endpoint,
        "backend": config.backend,
        "model": config.model,
        "context_max_tokens": config.context,
        "thinking": config.thinking,
        "temperature": config.temperature,
        "seed": config.seed,
        "output_token_limit": bench.effective_output_tokens(config, case),
        "timeout_seconds": config.timeout,
        "api_key_env": config.api_key_env,
        "context_mode": config.context_mode,
        "strata_pcie_frac": config.strata_pcie_frac,
        "strata_aux_cpus": config.strata_aux_cpus,
        "reasoning_budget_tokens": config.reasoning_budget_tokens,
        "chat_template_thinking": config.chat_template_thinking,
        "prompt_chars": len(case.prompt),
        "prompt_sha256": sha256_text(case.prompt),
        "diagnostic_cell_id": cell_id,
        "cell_variant": cell_id,
        "diagnostic_variant": variant,
        "prompt_variant": variant,
        "sampling_variant": sampling_variant,
        "parent_case_id": PARENT_CASE_ID,
        "parent_case_version": bench_cases.CASE_VERSION,
        "parent_prompt_sha256": parent_hash,
        "instruction_suffix_sha256": sha256_text(INSTRUCTION_SUFFIX) if variant == "tighter" else None,
        "model_session_run_id": model_session_run_id,
        "historical_comparator_path": str(comparator_path),
        "historical_comparator_call_id": comparator_call_id,
    }
    if sampling_variant == "liquidai_recommended_temperature_with_verified_server_defaults":
        recommended_sampling = {
            "source": MODEL_CARD_URL,
            "temperature": {
                "value": RECOMMENDED_TEMPERATURE,
                "source": "request override",
            },
            "top_k": {
                "value": RECOMMENDED_TOP_K,
                "source": "verified active server /props default; not overridden by request",
            },
            "repeat_penalty": {
                "value": RECOMMENDED_REPEAT_PENALTY,
                "source": "verified active server /props default; not overridden by request",
            },
        }
        metadata["declared_sampling"] = recommended_sampling
        request["declared_sampling"] = recommended_sampling
    if call is not None:
        evaluation = bench_cases.evaluate_case(
            case,
            str(call.get("response_text", "")),
            truncated=bool(call.get("truncated")),
        )
    else:
        evaluation = {"passed": False, "detail": "request failed"}
    serialized_error = None
    if error is not None:
        serialized_error = {"type": type(error).__name__, "message": str(error)[:500]}
    return {
        "schema_version": 1,
        "run_id": probe_run_id,
        "call_id": str(uuid.uuid4()),
        "started_at": utc_now(),
        "finished_at": utc_now(),
        "repeat_index": 0,
        "suite": case.suite,
        "case_id": case.case_id,
        "case_version": case_version,
        "case_source_sha256": source_hash,
        "case_metadata": metadata,
        "request": request,
        "telemetry_before": None,
        "telemetry_after": None,
        "response": dict(call) if call is not None else None,
        "evaluation": evaluation,
        "error": serialized_error,
    }


def _summary(row: Mapping[str, Any], *, historical: bool = False) -> dict[str, Any]:
    request = row.get("request") if isinstance(row.get("request"), dict) else {}
    response = row.get("response") if isinstance(row.get("response"), dict) else {}
    evaluation = row.get("evaluation") if isinstance(row.get("evaluation"), dict) else {}
    timing = response.get("timing") if isinstance(response.get("timing"), dict) else {}
    throughput = response.get("throughput") if isinstance(response.get("throughput"), dict) else {}
    truncated = response.get("truncated") is True
    if row.get("error") is not None:
        outcome = "request_failure"
    elif truncated:
        outcome = "budget_limited"
    elif evaluation.get("passed") is True:
        outcome = "passed"
    else:
        outcome = "wrong_answer"
    return {
        "historical": historical,
        "call_id": row.get("call_id"),
        "case_id": row.get("case_id"),
        "case_version": row.get("case_version"),
        "prompt_sha256": request.get("prompt_sha256"),
        "output_token_limit": request.get("output_token_limit"),
        "temperature": request.get("temperature"),
        "sampling_variant": request.get("sampling_variant"),
        "objective_passed": evaluation.get("passed") if isinstance(evaluation.get("passed"), bool) else None,
        "outcome": outcome,
        "truncated": response.get("truncated"),
        "completion_tokens": response.get("actual_completion_tokens"),
        "reasoning_chars": response.get("reasoning_chars"),
        "wall_seconds": timing.get("wall_seconds"),
        "completion_tokens_per_second_native": throughput.get("completion_tokens_per_second_native"),
        "completion_tokens_per_second_wall": throughput.get("completion_tokens_per_second_wall"),
    }


def _manifest(
    original: BenchmarkCase,
    tighter: BenchmarkCase,
    *,
    include_recommended_temperature: bool,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "created_at": utc_now(),
        "diagnostic_kind": "resident_ledger_budget_prompt_probe",
        "underlying_problem_count": 1,
        "include_recommended_temperature": include_recommended_temperature,
        "parent": {
            "case_id": original.case_id,
            "case_version": bench_cases.CASE_VERSION,
            "prompt": original.prompt,
            "prompt_sha256": sha256_text(original.prompt),
            "evaluator": original.evaluator,
        },
        "tighter_variant": {
            "case_id": tighter.case_id,
            "case_version": TIGHTER_CASE_VERSION,
            "prompt": tighter.prompt,
            "prompt_sha256": sha256_text(tighter.prompt),
            "parent_case_id": original.case_id,
            "parent_case_version": bench_cases.CASE_VERSION,
            "instruction_suffix": INSTRUCTION_SUFFIX,
            "instruction_suffix_sha256": sha256_text(INSTRUCTION_SUFFIX),
            "scope": (
                "Instruction-sensitivity variant discouraging unrelated alternatives and repeated "
                "analysis; original facts, operations, output schema, evaluator, and expected value are unchanged."
            ),
        },
        "optional_recommended_temperature_cell": {
            "enabled": include_recommended_temperature,
            "case_id": original.case_id,
            "case_version": bench_cases.CASE_VERSION,
            "prompt_sha256": sha256_text(original.prompt),
            "output_token_limit": 8_192,
            "temperature": RECOMMENDED_TEMPERATURE,
            "server_defaults_required": {
                "top_k": RECOMMENDED_TOP_K,
                "repeat_penalty": RECOMMENDED_REPEAT_PENALTY,
            },
            "recommendation_source": MODEL_CARD_URL,
        },
    }


def run_probe(args: argparse.Namespace) -> int:
    prefix = Path(args.output_prefix).expanduser()
    paths = artifact_paths(prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    collisions = [str(path) for path in paths.values() if path.exists()]
    if collisions:
        raise ProbeError("diagnostic artifact already exists: " + ", ".join(collisions))
    if not args.model or not args.model_session_run_id or not args.tag or not args.comparator_call_id:
        raise ProbeError("model, model-session-run-id, tag, and comparator-call-id must be non-empty")
    if args.holding_timeout <= 0 or not math.isfinite(args.holding_timeout):
        raise ProbeError("holding timeout must be a finite positive number")

    original, tighter = build_variants()
    include_recommended = bool(args.include_recommended_temperature)
    cells = CELLS + ((RECOMMENDED_TEMPERATURE_CELL,) if include_recommended else ())
    original_fingerprint = (
        original.case_id,
        original.prompt,
        original.evaluator,
        original.expected,
        original.metadata,
    )
    identity = verify_server_identity(
        args.endpoint,
        args.model,
        require_recommended_defaults=include_recommended,
    )
    session = wait_for_holding_session(
        Path(args.runtime_events).expanduser(),
        expected_run_id=args.model_session_run_id,
        expected_tag=args.tag,
        expected_model=args.model,
        timeout=args.holding_timeout,
    )
    # Rebind after the wait so a replaced or reconfigured server cannot inherit
    # the authority established by the earlier endpoint check.
    verify_server_identity(
        args.endpoint,
        args.model,
        require_recommended_defaults=include_recommended,
    )
    comparator_path = Path(args.comparator).expanduser().resolve()
    comparator = load_comparator(comparator_path, args.comparator_call_id, args.model, original)
    feasibility = prompt_feasibility(args.endpoint, original, tighter, cells)

    _exclusive_json(
        paths["prompt_manifest"],
        _manifest(
            original,
            tighter,
            include_recommended_temperature=include_recommended,
        ),
    )
    source_hash = hashlib.sha256(Path(bench_cases.__file__).read_bytes()).hexdigest()
    probe_run_id = str(uuid.uuid4())
    completed: list[dict[str, Any]] = []
    status = "complete"
    problem: dict[str, str] | None = None
    interrupted: BaseException | None = None

    try:
        with paths["requests"].open("x", encoding="utf-8", buffering=1) as handle:
            variants = {"original": original, "tighter": tighter}
            for cell_id, variant_name, cap, temperature, sampling_variant in cells:
                verify_server_identity(
                    args.endpoint,
                    args.model,
                    require_recommended_defaults=include_recommended,
                )
                case = variants[variant_name]
                config = bench.RunConfig(
                    endpoint=args.endpoint,
                    backend="openai",
                    model=args.model,
                    context=CONTEXT_TOKENS,
                    thinking=THINKING,
                    temperature=temperature,
                    seed=SEED,
                    num_predict=cap,
                    timeout=TIMEOUT_SECONDS,
                    context_mode="full",
                    chat_template_thinking="on",
                )
                call: Mapping[str, Any] | None = None
                request_error: BaseException | None = None
                started_at = utc_now()
                try:
                    call = bench.run_request(config, case)
                except (bench.BenchmarkHTTPError, TimeoutError, OSError, ValueError) as exc:
                    request_error = exc
                row = _request_record(
                    config,
                    case,
                    case_version=bench_cases.CASE_VERSION if variant_name == "original" else TIGHTER_CASE_VERSION,
                    cell_id=cell_id,
                    variant=variant_name,
                    sampling_variant=sampling_variant,
                    probe_run_id=probe_run_id,
                    model_session_run_id=args.model_session_run_id,
                    comparator_path=comparator_path,
                    comparator_call_id=args.comparator_call_id,
                    source_hash=source_hash,
                    call=call,
                    error=request_error,
                )
                row["started_at"] = started_at
                row["finished_at"] = utc_now()
                _append_jsonl(handle, row)
                completed.append(row)
                result = "PASS" if row["evaluation"].get("passed") else "FAIL"
                print(f"{result} {cell_id} call_id={row['call_id']}", file=sys.stderr, flush=True)
                if request_error is not None:
                    status = "failed"
                    problem = {"type": type(request_error).__name__, "message": str(request_error)[:500]}
                    break
    except BaseException as exc:
        status = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        problem = {"type": type(exc).__name__, "message": str(exc)[:500]}
        interrupted = exc
    finally:
        if original_fingerprint != (
            original.case_id,
            original.prompt,
            original.evaluator,
            original.expected,
            original.metadata,
        ):
            status = "failed"
            problem = {"type": "ProbeError", "message": "frozen original case mutated during the probe"}
        receipt = {
            "schema_version": 1,
            "created_at": utc_now(),
            "status": status,
            "diagnostic_kind": "resident_ledger_budget_prompt_probe",
            "probe_run_id": probe_run_id,
            "model_session_run_id": args.model_session_run_id,
            "tag": args.tag,
            "underlying_problem_count": 1,
            "planned_new_attempts": len(cells),
            "completed_new_attempts": len(completed),
            "historical_comparator_count": 1,
            "attempt_relationship": (
                "One historical comparator and three paired prompt/budget attempts of the same "
                "ledger problem; they are not independent benchmark problems."
            ),
            "controls": {
                "context_tokens": CONTEXT_TOKENS,
                "thinking": THINKING,
                "default_cell_temperature": TEMPERATURE,
                "seed": SEED,
                "timeout_seconds": TIMEOUT_SECONDS,
                "serial_execution": True,
                "include_recommended_temperature": include_recommended,
                "optional_recommended_temperature": (
                    RECOMMENDED_TEMPERATURE if include_recommended else None
                ),
            },
            "resident_identity": identity,
            "holding_session": session,
            "prompt_feasibility": feasibility,
            "historical_comparator": {
                "source": str(comparator_path),
                "source_sha256": sha256_file(comparator_path),
                "summary": _summary(comparator, historical=True),
            },
            "new_attempts": [_summary(row) for row in completed],
            "artifacts": {
                "requests": str(paths["requests"]),
                "requests_sha256": sha256_file(paths["requests"]) if paths["requests"].exists() else None,
                "prompt_manifest": str(paths["prompt_manifest"]),
                "prompt_manifest_sha256": sha256_file(paths["prompt_manifest"]),
            },
            "problem": problem,
        }
        _exclusive_json(paths["receipt"], receipt)

    if interrupted is not None:
        raise interrupted
    return 1 if status == "failed" else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True, help="already-loaded loopback OpenAI-compatible /v1 endpoint")
    parser.add_argument("--model", required=True, help="exact resident model alias")
    parser.add_argument("--model-session-run-id", required=True, help="exact owned model_session run ID")
    parser.add_argument("--tag", required=True, help="exact tag in the runtime event ledger")
    parser.add_argument("--runtime-events", required=True, help="model_session runtime event JSONL")
    parser.add_argument("--comparator", required=True, help="historical original-prompt 8192-cap JSONL")
    parser.add_argument("--comparator-call-id", required=True, help="exact call ID in the comparator JSONL")
    parser.add_argument("--output-prefix", required=True, help="unique prefix for requests, manifest, and receipt")
    parser.add_argument("--holding-timeout", type=float, default=3600.0, help="seconds to wait for holding_started")
    parser.add_argument(
        "--include-recommended-temperature",
        action="store_true",
        help=(
            "append an original-prompt, 8192-cap cell at temperature 0.1 after verifying "
            "active top_k=50 and repeat_penalty=1.05 defaults"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run_probe(args)
    except KeyboardInterrupt:
        print("budget_prompt_probe: interrupted", file=sys.stderr)
        return 130
    except ProbeError as exc:
        print(f"budget_prompt_probe: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
