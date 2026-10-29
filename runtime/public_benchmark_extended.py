#!/usr/bin/env python3
"""Extended request adapter for the frozen public benchmark.

The durable storage, SSE parsing, resume semantics, and official graders live
in :mod:`public_benchmark_pilot`. This module changes only the scored request
settings surface. It accepts a small allowlist instead of an arbitrary JSON
body so a profile cannot replace prompts or serialize grader data.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence
from urllib.parse import urlsplit

if __package__:
    from . import public_benchmark_pilot as base
else:
    import public_benchmark_pilot as base


ADAPTER_VERSION = "public-expansion-openai-adapter-20261024-v1"
BACKEND_ADAPTERS = ("llama", "strata")
REASONING_EFFORTS = ("none", "low", "medium", "high")


def _optional_float(value: Any, label: str, minimum: float, maximum: float) -> float | None:
    if value is None:
        return None
    return base._finite_float(value, label, minimum, maximum)


def _optional_int(value: Any, label: str, minimum: int, maximum: int) -> int | None:
    if value is None:
        return None
    return base._plain_int(value, label, minimum, maximum)


def run_settings(args: argparse.Namespace) -> dict[str, Any]:
    endpoint = args.endpoint.rstrip("/")
    parsed = urlsplit(endpoint)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise base.PilotError("endpoint must be an absolute http(s) URL")
    if parsed.query or parsed.fragment:
        raise base.PilotError("endpoint must not contain a query or fragment")
    if not isinstance(args.model, str) or not args.model:
        raise base.PilotError("model must be a non-empty API alias")
    if not isinstance(args.model_key, str) or not args.model_key:
        raise base.PilotError("model-key must be a non-empty reporting key")
    if args.backend_adapter not in BACKEND_ADAPTERS:
        raise base.PilotError("backend-adapter must be one of " + ", ".join(BACKEND_ADAPTERS))
    if args.reasoning_effort not in REASONING_EFFORTS:
        raise base.PilotError("reasoning-effort must be one of " + ", ".join(REASONING_EFFORTS))

    max_tokens = base._plain_int(args.max_tokens, "max-tokens", 1, base.MAX_OUTPUT_TOKENS)
    request_options: dict[str, Any] = {
        "temperature": base._finite_float(args.temperature, "temperature", 0.0, 2.0),
        "reasoning_effort": args.reasoning_effort,
    }
    optional_values = {
        "top_p": _optional_float(args.top_p, "top-p", 0.0, 1.0),
        "top_k": _optional_int(args.top_k, "top-k", 0, 1_000_000),
        "min_p": _optional_float(args.min_p, "min-p", 0.0, 1.0),
        "presence_penalty": _optional_float(args.presence_penalty, "presence-penalty", -2.0, 2.0),
        "repeat_penalty": _optional_float(args.repeat_penalty, "repeat-penalty", 0.0, 10.0),
        "reasoning_budget_tokens": _optional_int(
            args.reasoning_budget_tokens, "reasoning-budget-tokens", 0, base.MAX_OUTPUT_TOKENS
        ),
        "thinking_budget_tokens": _optional_int(
            args.thinking_budget_tokens, "thinking-budget-tokens", 0, base.MAX_OUTPUT_TOKENS
        ),
    }
    for key, value in optional_values.items():
        if value is not None:
            request_options[key] = value
    for key in ("reasoning_budget_tokens", "thinking_budget_tokens"):
        if request_options.get(key, 0) > max_tokens:
            raise base.PilotError(f"{key.replace('_', '-')} cannot exceed max-tokens")
    if args.chat_template_thinking != "inherit":
        request_options["chat_template_kwargs"] = {
            "enable_thinking": args.chat_template_thinking == "on"
        }

    return {
        "schema_version": base.SCHEMA_VERSION,
        "adapter_version": ADAPTER_VERSION,
        "backend_adapter": args.backend_adapter,
        "endpoint": endpoint,
        "model": args.model,
        "model_key": args.model_key,
        "max_tokens": max_tokens,
        "request_timeout_seconds": base._finite_float(
            args.request_timeout, "request-timeout", 1.0, base.MAX_REQUEST_TIMEOUT
        ),
        "seed": base._plain_int(args.seed, "seed", -(2**63), 2**63 - 1),
        "api_key_env": args.api_key_env,
        "request_options": request_options,
        "prompt_policy_version": base.PROMPT_POLICY_VERSION,
        "humaneval_system_instruction": base.HUMANEVAL_SYSTEM_INSTRUCTION,
        "ifeval_prompt_policy": "verbatim_user_message_only",
        "stream_options_include_usage": True,
    }


def build_request_payload(case: base.Case, settings: Mapping[str, Any]) -> dict[str, Any]:
    """Build a request from public prompt fields plus allowlisted controls only."""
    options = settings.get("request_options")
    if not isinstance(options, Mapping):
        raise base.PilotError("settings request_options must be an object")
    allowed = {
        "temperature", "reasoning_effort", "top_p", "top_k", "min_p",
        "presence_penalty", "repeat_penalty", "reasoning_budget_tokens",
        "thinking_budget_tokens", "chat_template_kwargs",
    }
    unknown = sorted(set(options) - allowed)
    if unknown:
        raise base.PilotError(
            "settings contain non-allowlisted request option(s): " + ", ".join(unknown)
        )
    payload: dict[str, Any] = {
        "model": settings["model"],
        "messages": base.build_messages(case),
        "max_tokens": settings["max_tokens"],
        "seed": settings["seed"],
        "n": 1,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    payload.update(options)
    return payload


def execute_case(
    output_dir: Path,
    manifest: base.Manifest,
    binding: Mapping[str, Any],
    settings: Mapping[str, Any],
    case: base.Case,
    api_key: str | None,
) -> Mapping[str, Any]:
    """Execute one case using the base runner's durable attempt protocol."""
    attempt_number, attempt_dir = base._new_attempt(output_dir, case)
    attempt_started = base.time.monotonic()
    attempt_started_at = base.utc_now()
    payload = build_request_payload(case, settings)
    case_record = {"index": case.index, "suite": case.suite, "task_id": case.task_id}
    request_record = {
        "schema_version": base.SCHEMA_VERSION,
        "created_at": attempt_started_at,
        "manifest_sha256": manifest.sha256,
        "settings_sha256": binding["settings_sha256"],
        "case": case_record,
        "attempt": attempt_number,
        "url": settings["endpoint"].rstrip("/") + "/chat/completions",
        "payload": payload,
    }
    active = {**case_record, "case_index": case.index, "attempt": attempt_number, "started_at": attempt_started_at}
    active.pop("index")
    response: Mapping[str, Any] | None = None
    try:
        base.atomic_write_json(attempt_dir / "request.json", request_record)
        base.write_status(output_dir, manifest, binding, active=active)
        try:
            response = base.request_chat_completion(
                settings["endpoint"], payload, settings["request_timeout_seconds"],
                attempt_dir / "raw.sse", api_key,
            )
        except InterruptedError as exc:
            response = json.loads(str(exc))
            raise KeyboardInterrupt
        result = {
            "schema_version": base.SCHEMA_VERSION,
            "manifest_sha256": manifest.sha256,
            "settings_sha256": binding["settings_sha256"],
            "case": case_record,
            "attempt": attempt_number,
            **response,
        }
        base.atomic_write_json(attempt_dir / "result.json", result)
        base.write_status(output_dir, manifest, binding)
        return result
    except KeyboardInterrupt:
        if response is None:
            response = base._response_result(
                accumulator=base.ChatAccumulator(), started=attempt_started,
                started_at=attempt_started_at, http_status=None, content_type=None,
                error={"type": "KeyboardInterrupt", "message": "request cancelled"},
                cancelled=True,
            )
        raw_path = attempt_dir / "raw.sse"
        if not raw_path.exists():
            base.atomic_write_bytes(raw_path, b"")
        if not (attempt_dir / "request.json").exists():
            base.atomic_write_json(attempt_dir / "request.json", request_record)
        result = {
            "schema_version": base.SCHEMA_VERSION,
            "manifest_sha256": manifest.sha256,
            "settings_sha256": binding["settings_sha256"],
            "case": case_record,
            "attempt": attempt_number,
            **response,
        }
        base.atomic_write_json(attempt_dir / "result.json", result)
        base.write_status(output_dir, manifest, binding)
        raise


def run_command(args: argparse.Namespace) -> int:
    manifest = base.load_manifest(Path(args.manifest))
    settings = run_settings(args)
    output_dir = Path(args.output_dir).expanduser().resolve()
    api_key = os.environ.get(args.api_key_env) if args.api_key_env else None
    with base.output_lock(output_dir):
        binding = base.ensure_binding(output_dir, manifest, settings)
        base.write_status(output_dir, manifest, binding)
        with base.cancel_on_sigterm():
            try:
                for case in manifest.cases:
                    state = base.case_state(output_dir, case, binding)
                    if state["selected_result"] is not None:
                        continue
                    if state["attempt_count"] and not args.rerun_failed:
                        continue
                    execute_case(output_dir, manifest, binding, settings, case, api_key)
            except KeyboardInterrupt:
                base.write_status(output_dir, manifest, binding)
                return 130
        final_status = base.write_status(output_dir, manifest, binding)
    counts = final_status["counts"]
    return 0 if counts["completed"] + counts["truncated"] == counts["total"] else 1


def build_run_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the frozen public benchmark with a pinned backend adapter."
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--backend-adapter", choices=BACKEND_ADAPTERS, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-tokens", type=int, default=65_536)
    parser.add_argument("--request-timeout", type=float, default=21_600.0)
    parser.add_argument("--reasoning-effort", choices=REASONING_EFFORTS, default="medium")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--min-p", type=float)
    parser.add_argument("--presence-penalty", type=float)
    parser.add_argument("--repeat-penalty", type=float)
    parser.add_argument("--chat-template-thinking", choices=("inherit", "on", "off"), default="inherit")
    parser.add_argument("--reasoning-budget-tokens", type=int)
    parser.add_argument("--thinking-budget-tokens", type=int)
    parser.add_argument("--seed", type=int, default=20_261_023)
    parser.add_argument("--api-key-env")
    parser.add_argument("--rerun-failed", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] in {"grade", "_evalplus-worker"}:
        return base.main(arguments)
    if arguments and arguments[0] == "run":
        arguments.pop(0)
    parser = build_run_parser()
    args = parser.parse_args(arguments)
    try:
        return run_command(args)
    except base.PilotError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
