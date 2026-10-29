#!/usr/bin/env python3
"""Reproducible JSONL benchmark client for Ollama and OpenAI-compatible servers."""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import math
import os
import socket
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping
from urllib.parse import SplitResult, urlsplit

import bench_cases
from bench_cases import BenchmarkCase


NATIVE_OLLAMA_FIELDS = (
    "prompt_eval_count",
    "prompt_eval_duration",
    "eval_count",
    "eval_duration",
    "load_duration",
    "done_reason",
)

OPENAI_NATIVE_TIMING_FIELDS = (
    "prompt_ms",
    "prompt_n",
    "cache_n",
    "prompt_per_token_ms",
    "prompt_per_second",
    "predicted_ms",
    "predicted_n",
    "predicted_per_token_ms",
    "predicted_per_second",
    "draft_n",
    "draft_n_accepted",
)

MAX_REASONING_BUDGET_TOKENS = 1_000_000


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


@dataclass(frozen=True)
class RunConfig:
    endpoint: str
    backend: str
    model: str
    context: int
    thinking: str
    temperature: float
    seed: int
    num_predict: int | None
    timeout: float
    api_key_env: str | None = None
    context_mode: str = "full"
    strata_pcie_frac: float | None = None
    strata_aux_cpus: int | None = None
    reasoning_budget_tokens: int | None = None
    chat_template_thinking: str | None = None


def effective_output_tokens(config: RunConfig, case: BenchmarkCase) -> int:
    return config.num_predict if config.num_predict is not None else case.max_tokens


def ollama_think_value(model: str, thinking: str) -> str | bool:
    normalized = model.lower().replace("_", "-")
    if thinking == "none":
        return False
    if "qwen" in normalized:
        # Current Ollama Qwen thinking controls are boolean rather than effort strings.
        return True
    if "gpt-oss" in normalized or "gptoss" in normalized:
        return thinking
    return thinking


def validate_request_controls(config: RunConfig) -> None:
    controls = (
        config.strata_pcie_frac,
        config.strata_aux_cpus,
        config.reasoning_budget_tokens,
        config.chat_template_thinking,
    )
    if config.backend != "openai" and any(value is not None for value in controls):
        raise ValueError(
            "--strata-pcie-frac, --strata-aux-cpus, --reasoning-budget-tokens, and "
            "--chat-template-thinking require --backend openai"
        )
    pcie = config.strata_pcie_frac
    if pcie is not None and (
        isinstance(pcie, bool)
        or not isinstance(pcie, (int, float))
        or not math.isfinite(float(pcie))
        or not 0.0 <= float(pcie) <= 1.0
    ):
        raise ValueError("strata_pcie_frac must be a finite number from 0 through 1")
    aux = config.strata_aux_cpus
    if aux is not None and (isinstance(aux, bool) or not isinstance(aux, int) or aux not in (0, 1)):
        raise ValueError("strata_aux_cpus must be 0 or 1")
    budget = config.reasoning_budget_tokens
    if budget is not None and (
        isinstance(budget, bool)
        or not isinstance(budget, int)
        or not 1 <= budget <= MAX_REASONING_BUDGET_TOKENS
    ):
        raise ValueError(
            f"reasoning_budget_tokens must be an integer from 1 through {MAX_REASONING_BUDGET_TOKENS}"
        )
    if config.chat_template_thinking not in (None, "on", "off"):
        raise ValueError("chat_template_thinking must be 'on' or 'off'")


def build_ollama_payload(config: RunConfig, case: BenchmarkCase) -> dict[str, Any]:
    validate_request_controls(config)
    return {
        "model": config.model,
        "messages": [{"role": "user", "content": case.prompt}],
        "stream": True,
        "think": ollama_think_value(config.model, config.thinking),
        "options": {
            "num_ctx": config.context,
            "num_predict": effective_output_tokens(config, case),
            "temperature": config.temperature,
            "seed": config.seed,
        },
    }


def build_openai_payload(config: RunConfig, case: BenchmarkCase) -> dict[str, Any]:
    validate_request_controls(config)
    payload: dict[str, Any] = {
        "model": config.model,
        "messages": [{"role": "user", "content": case.prompt}],
        "stream": True,
        "stream_options": {"include_usage": True},
        "max_tokens": effective_output_tokens(config, case),
        "temperature": config.temperature,
        "seed": config.seed,
    }
    # Strata defaults omitted reasoning controls to high, so "none" is explicit.
    payload["reasoning_effort"] = config.thinking
    tune: dict[str, int | float] = {}
    if config.strata_pcie_frac is not None:
        tune["pcie_frac"] = float(config.strata_pcie_frac)
    if config.strata_aux_cpus is not None:
        tune["aux_cpus"] = config.strata_aux_cpus
    if tune:
        payload["strata_tune"] = tune
    if config.reasoning_budget_tokens is not None:
        payload["reasoning_budget_tokens"] = config.reasoning_budget_tokens
    if config.chat_template_thinking is not None:
        payload["chat_template_kwargs"] = {
            "enable_thinking": config.chat_template_thinking == "on",
        }
    return payload


def api_url(endpoint: str, backend: str) -> str:
    base = endpoint.rstrip("/")
    if backend == "ollama":
        return base if base.endswith("/api/chat") else base + "/api/chat"
    if base.endswith("/v1/chat/completions"):
        return base
    if base.endswith("/v1"):
        return base + "/chat/completions"
    return base + "/v1/chat/completions"


def ollama_ps_url(endpoint: str) -> str:
    parsed = urlsplit(endpoint)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError("endpoint must be an absolute http(s) URL")
    return f"{parsed.scheme}://{parsed.netloc}/api/ps"


def _connection(parsed: SplitResult, timeout: float) -> http.client.HTTPConnection:
    if parsed.scheme == "http":
        return http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=timeout)
    if parsed.scheme == "https":
        return http.client.HTTPSConnection(parsed.hostname, parsed.port, timeout=timeout)
    raise ValueError("endpoint URL scheme must be http or https")


def _path_with_query(parsed: SplitResult) -> str:
    path = parsed.path or "/"
    return path + (("?" + parsed.query) if parsed.query else "")


class BenchmarkHTTPError(RuntimeError):
    pass


def stream_json_events(
    url: str,
    payload: Mapping[str, Any],
    headers: Mapping[str, str],
    timeout: float,
    backend: str,
) -> Iterator[dict[str, Any]]:
    """Yield NDJSON or SSE events with a total wall-clock deadline."""
    parsed = urlsplit(url)
    connection = _connection(parsed, timeout)
    deadline = time.monotonic() + timeout
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    request_headers = {"Content-Type": "application/json", "Accept": "application/json"}
    request_headers.update(headers)
    try:
        connection.request("POST", _path_with_query(parsed), body=body, headers=request_headers)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"request exceeded {timeout:g}s total timeout")
        if connection.sock is not None:
            connection.sock.settimeout(max(0.001, remaining))
        response = connection.getresponse()
        if response.status < 200 or response.status >= 300:
            error_body = response.read(4096).decode("utf-8", "replace")
            try:
                parsed_error = json.loads(error_body)
                message = parsed_error.get("error", {}).get("message") if isinstance(parsed_error.get("error"), dict) else parsed_error.get("error")
            except (json.JSONDecodeError, AttributeError):
                message = None
            raise BenchmarkHTTPError(f"HTTP {response.status}: {str(message or response.reason)[:500]}")

        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"request exceeded {timeout:g}s total timeout")
            if connection.sock is not None:
                connection.sock.settimeout(max(0.001, remaining))
            line = response.readline(16 * 1024 * 1024)
            if not line:
                break
            decoded = line.decode("utf-8", "replace").strip()
            if not decoded:
                continue
            if backend == "openai":
                if not decoded.startswith("data:"):
                    continue
                decoded = decoded[5:].strip()
                if decoded == "[DONE]":
                    yield {"_stream_done": True}
                    break
            try:
                event = json.loads(decoded)
            except json.JSONDecodeError as exc:
                raise BenchmarkHTTPError(f"invalid streamed JSON: {exc.msg}") from exc
            if isinstance(event, dict):
                yield event
    except socket.timeout as exc:
        raise TimeoutError(f"request exceeded {timeout:g}s total timeout") from exc
    finally:
        connection.close()


def _text_from_content(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        pieces: list[str] = []
        for item in value:
            if isinstance(item, str):
                pieces.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                pieces.append(item["text"])
        return "".join(pieces)
    return ""


def _reasoning_char_count(value: Any) -> int:
    if isinstance(value, str):
        return len(value)
    if value is None:
        return 0
    if isinstance(value, (list, dict)):
        return len(json.dumps(value, ensure_ascii=False, separators=(",", ":")))
    return len(str(value))


class OllamaAccumulator:
    def __init__(self) -> None:
        self._content: list[str] = []
        self.reasoning_chars = 0
        self.first_reasoning_seconds: float | None = None
        self.first_content_seconds: float | None = None
        self.native: dict[str, Any] = {}
        self.done = False

    def feed(self, event: Mapping[str, Any], elapsed: float) -> None:
        message = event.get("message") if isinstance(event.get("message"), dict) else {}
        reasoning = message.get("thinking")
        if reasoning is None:
            reasoning = message.get("reasoning_content", message.get("reasoning"))
        reasoning_chars = _reasoning_char_count(reasoning)
        if reasoning_chars:
            if self.first_reasoning_seconds is None:
                self.first_reasoning_seconds = elapsed
            self.reasoning_chars += reasoning_chars
        content = _text_from_content(message.get("content"))
        if content:
            if self.first_content_seconds is None:
                self.first_content_seconds = elapsed
            self._content.append(content)
        if event.get("done") is True:
            self.done = True
        for key in NATIVE_OLLAMA_FIELDS:
            if key in event:
                self.native[key] = event[key]

    @property
    def content(self) -> str:
        return "".join(self._content)


class OpenAIAccumulator:
    def __init__(self) -> None:
        self._content: list[str] = []
        self.reasoning_chars = 0
        self.first_reasoning_seconds: float | None = None
        self.first_content_seconds: float | None = None
        self.usage: dict[str, Any] | None = None
        self.native_timings: dict[str, int | float] | None = None
        self.finish_reason: str | None = None
        self.done_marker = False

    def feed(self, event: Mapping[str, Any], elapsed: float) -> None:
        if event.get("_stream_done"):
            self.done_marker = True
            return
        if event.get("error"):
            error = event["error"]
            code = error.get("code") if isinstance(error, dict) else None
            raise BenchmarkHTTPError(f"server stream error{f' ({code})' if code else ''}")
        if isinstance(event.get("usage"), dict):
            self.usage = dict(event["usage"])
        if isinstance(event.get("timings"), dict):
            timings = {
                key: value
                for key in OPENAI_NATIVE_TIMING_FIELDS
                if isinstance((value := event["timings"].get(key)), (int, float))
                and not isinstance(value, bool)
                and (isinstance(value, int) or math.isfinite(value))
            }
            # Replace rather than merge so a later timing object cannot retain omitted fields.
            self.native_timings = timings or None
        choices = event.get("choices")
        if not isinstance(choices, list):
            return
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            if choice.get("finish_reason") is not None:
                self.finish_reason = str(choice["finish_reason"])
            delta = choice.get("delta") if isinstance(choice.get("delta"), dict) else {}
            reasoning = delta.get("reasoning_content")
            if reasoning is None:
                reasoning = delta.get("reasoning", delta.get("reasoning_details"))
            reasoning_chars = _reasoning_char_count(reasoning)
            if reasoning_chars:
                if self.first_reasoning_seconds is None:
                    self.first_reasoning_seconds = elapsed
                self.reasoning_chars += reasoning_chars
            content = _text_from_content(delta.get("content"))
            if content:
                if self.first_content_seconds is None:
                    self.first_content_seconds = elapsed
                self._content.append(content)

    @property
    def content(self) -> str:
        return "".join(self._content)


def _simple_get_json(url: str, timeout: float = 2.0) -> Any:
    parsed = urlsplit(url)
    connection = _connection(parsed, timeout)
    try:
        connection.request("GET", _path_with_query(parsed), headers={"Accept": "application/json"})
        response = connection.getresponse()
        if response.status < 200 or response.status >= 300:
            raise BenchmarkHTTPError(f"HTTP {response.status}")
        raw = response.read(2 * 1024 * 1024)
        return json.loads(raw)
    finally:
        connection.close()


def collect_ollama_residency(endpoint: str) -> dict[str, Any]:
    try:
        payload = _simple_get_json(ollama_ps_url(endpoint))
        models = payload.get("models", []) if isinstance(payload, dict) else []
        return {"available": True, "models": models}
    except Exception as exc:  # best-effort telemetry must not fail a benchmark call
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}


def collect_nvidia() -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=3,
            check=False,
        )
        if completed.returncode != 0:
            return {"available": False, "error": f"exit status {completed.returncode}"}
        gpus = []
        for line in completed.stdout.splitlines():
            columns = [column.strip() for column in line.split(",")]
            if len(columns) != 5:
                continue
            try:
                gpus.append(
                    {
                        "index": int(columns[0]),
                        "name": columns[1],
                        "memory_used_mib": int(columns[2]),
                        "memory_total_mib": int(columns[3]),
                        "utilization_percent": int(columns[4]),
                    }
                )
            except ValueError:
                continue
        return {"available": True, "gpus": gpus}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}


def collect_telemetry(endpoint: str) -> dict[str, Any]:
    return {
        "captured_at": utc_now(),
        "ollama_ps": collect_ollama_residency(endpoint),
        "nvidia_smi": collect_nvidia(),
    }


def _rate(count: Any, duration_seconds: float | None) -> float | None:
    if not isinstance(count, (int, float)) or isinstance(count, bool):
        return None
    if not duration_seconds or duration_seconds <= 0:
        return None
    return float(count) / duration_seconds


def _nanoseconds_to_seconds(value: Any) -> float | None:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
        return None
    return float(value) / 1_000_000_000.0


def _usage_token(usage: Mapping[str, Any] | None, *names: str) -> int | None:
    if not usage:
        return None
    for name in names:
        value = usage.get(name)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return None


def run_request(config: RunConfig, case: BenchmarkCase) -> dict[str, Any]:
    headers: dict[str, str] = {}
    if config.api_key_env:
        api_key = os.environ.get(config.api_key_env)
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

    if config.backend == "ollama":
        payload = build_ollama_payload(config, case)
        accumulator: OllamaAccumulator | OpenAIAccumulator = OllamaAccumulator()
    else:
        payload = build_openai_payload(config, case)
        accumulator = OpenAIAccumulator()

    started = time.monotonic()
    for event in stream_json_events(
        api_url(config.endpoint, config.backend), payload, headers, config.timeout, config.backend
    ):
        accumulator.feed(event, time.monotonic() - started)
    wall_seconds = time.monotonic() - started

    if isinstance(accumulator, OllamaAccumulator):
        native = accumulator.native
        prompt_tokens = native.get("prompt_eval_count")
        completion_tokens = native.get("eval_count")
        done_reason = native.get("done_reason")
        complete = accumulator.done
        usage = None
        metrics = {
            "prompt_tokens_per_second_native": _rate(
                prompt_tokens, _nanoseconds_to_seconds(native.get("prompt_eval_duration"))
            ),
            "completion_tokens_per_second_native": _rate(
                completion_tokens, _nanoseconds_to_seconds(native.get("eval_duration"))
            ),
            "completion_tokens_per_second_wall": _rate(completion_tokens, wall_seconds),
        }
    else:
        native = None
        usage = accumulator.usage
        openai_native_timings = accumulator.native_timings
        prompt_tokens = _usage_token(usage, "prompt_tokens", "input_tokens")
        completion_tokens = _usage_token(usage, "completion_tokens", "output_tokens")
        done_reason = accumulator.finish_reason
        complete = accumulator.done_marker or accumulator.finish_reason is not None
        metrics = {
            "completion_tokens_per_second_wall": _rate(completion_tokens, wall_seconds),
        }
        if openai_native_timings is not None:
            if "prompt_per_second" in openai_native_timings:
                metrics["prompt_tokens_per_second_native"] = openai_native_timings[
                    "prompt_per_second"
                ]
            if "predicted_per_second" in openai_native_timings:
                metrics["completion_tokens_per_second_native"] = openai_native_timings[
                    "predicted_per_second"
                ]

    truncated = done_reason in {"length", "max_tokens", "max_output_tokens"} or not complete
    result = {
        "response_text": accumulator.content,
        "reasoning_chars": accumulator.reasoning_chars,
        "timing": {
            "wall_seconds": wall_seconds,
            "ttft_reasoning_seconds": accumulator.first_reasoning_seconds,
            "ttft_final_content_seconds": accumulator.first_content_seconds,
        },
        "native_ollama": native,
        "openai_usage": usage,
        "actual_prompt_tokens": prompt_tokens,
        "actual_completion_tokens": completion_tokens,
        "done_reason": done_reason,
        "stream_complete": complete,
        "truncated": truncated,
        "throughput": metrics,
    }
    if isinstance(accumulator, OpenAIAccumulator) and accumulator.native_timings is not None:
        result["openai_native_timings"] = accumulator.native_timings
    return result


def _source_hash() -> str:
    return hashlib.sha256(Path(bench_cases.__file__).read_bytes()).hexdigest()


def _request_config(config: RunConfig, case: BenchmarkCase) -> dict[str, Any]:
    return {
        "endpoint": config.endpoint,
        "backend": config.backend,
        "model": config.model,
        "context_max_tokens": config.context,
        "thinking": config.thinking,
        "temperature": config.temperature,
        "seed": config.seed,
        "output_token_limit": effective_output_tokens(config, case),
        "timeout_seconds": config.timeout,
        "api_key_env": config.api_key_env,
        "context_mode": config.context_mode,
        "strata_pcie_frac": config.strata_pcie_frac,
        "strata_aux_cpus": config.strata_aux_cpus,
        "reasoning_budget_tokens": config.reasoning_budget_tokens,
        "chat_template_thinking": config.chat_template_thinking,
        "prompt_chars": len(case.prompt),
        "prompt_sha256": hashlib.sha256(case.prompt.encode("utf-8")).hexdigest(),
    }


def _append_jsonl(handle: Any, record: Mapping[str, Any]) -> None:
    handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def run_benchmark(args: argparse.Namespace) -> int:
    selected = [item.strip() for item in args.cases.split(",") if item.strip()] if args.cases else None
    base_config = RunConfig(
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
        context_mode=args.context_mode,
        strata_pcie_frac=getattr(args, "strata_pcie_frac", None),
        strata_aux_cpus=getattr(args, "strata_aux_cpus", None),
        reasoning_budget_tokens=getattr(args, "reasoning_budget_tokens", None),
        chat_template_thinking=getattr(args, "chat_template_thinking", None),
    )
    validate_request_controls(base_config)
    cases = bench_cases.get_cases(
        args.suite,
        selected,
        context_target=args.context_target,
        context_max=args.context,
        output_tokens=args.num_predict,
        seed=args.seed,
        context_mode=args.context_mode,
    )
    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    run_id = str(uuid.uuid4())
    source_hash = _source_hash()
    had_request_error = False
    had_case_failure = False

    with output.open("a", encoding="utf-8", buffering=1) as handle:
        for repeat_index in range(args.repeat):
            config = replace(
                base_config,
                seed=args.seed + repeat_index * args.seed_step,
            )
            for case in cases:
                call_id = str(uuid.uuid4())
                started_at = utc_now()
                before = None if args.skip_telemetry else collect_telemetry(args.endpoint)
                call: dict[str, Any] | None = None
                error: dict[str, str] | None = None
                try:
                    call = run_request(config, case)
                except (BenchmarkHTTPError, TimeoutError, OSError, ValueError) as exc:
                    had_request_error = True
                    error = {"type": type(exc).__name__, "message": str(exc)}
                after = None if args.skip_telemetry else collect_telemetry(args.endpoint)

                if call is not None:
                    evaluation = bench_cases.evaluate_case(
                        case, call["response_text"], truncated=bool(call["truncated"])
                    )
                else:
                    evaluation = {"passed": False, "detail": "request failed"}
                if not evaluation.get("passed"):
                    had_case_failure = True
                record: dict[str, Any] = {
                    "schema_version": 1,
                    "run_id": run_id,
                    "call_id": call_id,
                    "started_at": started_at,
                    "finished_at": utc_now(),
                    "repeat_index": repeat_index,
                    "suite": case.suite,
                    "case_id": case.case_id,
                    "case_version": bench_cases.CASE_VERSION,
                    "case_source_sha256": source_hash,
                    "case_metadata": dict(case.metadata),
                    "request": _request_config(config, case),
                    "telemetry_before": before,
                    "telemetry_after": after,
                    "response": call,
                    "evaluation": evaluation,
                    "error": error,
                }
                _append_jsonl(handle, record)
                status = "PASS" if evaluation.get("passed") else "FAIL"
                print(
                    f"{status} {case.case_id} repeat={repeat_index + 1}/{args.repeat} "
                    f"call_id={call_id}",
                    file=sys.stderr,
                    flush=True,
                )
    return 1 if had_request_error or (getattr(args, "fail_on_case", False) and had_case_failure) else 0


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than 0")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be at least 0")
    return parsed


def unit_interval_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("must be a finite number from 0 through 1")
    return parsed


def reasoning_budget_int(value: str) -> int:
    parsed = int(value)
    if not 1 <= parsed <= MAX_REASONING_BUDGET_TOKENS:
        raise argparse.ArgumentTypeError(
            f"must be from 1 through {MAX_REASONING_BUDGET_TOKENS}"
        )
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", help="server base URL")
    parser.add_argument("--backend", choices=("ollama", "openai"))
    parser.add_argument("--model", help="model name sent to the server")
    parser.add_argument("--context", type=positive_int, help="selected maximum context tokens")
    parser.add_argument("--thinking", choices=("low", "medium", "high", "none"), default="low")
    parser.add_argument("--suite", choices=("quick", "quality", "context", "hard"), default="quick")
    parser.add_argument("--cases", help="comma-separated case IDs within the selected suite")
    parser.add_argument("--context-target", type=positive_int, help="approximate prompt tokens for context suite")
    parser.add_argument(
        "--context-mode",
        choices=("full", "indexed"),
        default="full",
        help="full registry or exact-ID indexed evidence for the context suite",
    )
    parser.add_argument("--repeat", type=positive_int, default=1)
    parser.add_argument("--output", help="append-only JSONL output path")
    parser.add_argument("--timeout", type=positive_float, default=180.0, help="total per-call timeout seconds")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--seed-step",
        type=nonnegative_int,
        default=0,
        help="add this amount to the sampling seed after each repeat; case prompts keep the base seed",
    )
    parser.add_argument("--num-predict", type=positive_int, help="override each case's output token limit")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--strata-pcie-frac",
        type=unit_interval_float,
        help="Strata per-request PCIe scheduling fraction (OpenAI backend only)",
    )
    parser.add_argument(
        "--strata-aux-cpus",
        type=int,
        choices=(0, 1),
        help="Strata per-request auxiliary CPU placement toggle (OpenAI backend only)",
    )
    parser.add_argument(
        "--reasoning-budget-tokens",
        type=reasoning_budget_int,
        help="Strata per-request private-reasoning budget (OpenAI backend only)",
    )
    parser.add_argument(
        "--chat-template-thinking",
        choices=("on", "off"),
        help="llama.cpp per-request enable_thinking template switch (OpenAI backend only)",
    )
    parser.add_argument("--api-key-env", help="environment variable containing a bearer token; its value is never recorded")
    parser.add_argument("--skip-telemetry", action="store_true", help="disable best-effort /api/ps and nvidia-smi snapshots")
    parser.add_argument(
        "--fail-on-case",
        action="store_true",
        help="exit nonzero when any case evaluation fails or is incomplete",
    )
    parser.add_argument("--list-cases", action="store_true", help="list stable case IDs and exit")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.list_cases:
        print(json.dumps(bench_cases.all_case_ids(), indent=2))
        return 0
    missing = [name for name in ("endpoint", "backend", "model", "context", "output") if getattr(args, name) is None]
    if missing:
        parser.error("the following arguments are required unless --list-cases is used: " + ", ".join("--" + name for name in missing))
    if args.temperature < 0:
        parser.error("--temperature must be non-negative")
    try:
        return run_benchmark(args)
    except ValueError as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
