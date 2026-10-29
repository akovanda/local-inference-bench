#!/usr/bin/env python3
"""Durable endpoint-only runner for a frozen HumanEval+/IFEval pilot.

This module deliberately owns neither a model server nor a GPU.  ``run`` only
sends frozen requests to an already-running OpenAI-compatible endpoint.  The
``grade`` command invokes the pinned official graders in subprocesses.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import datetime as dt
import fcntl
import hashlib
import http.client
import importlib
import json
import math
import os
import pickle
import signal
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import urlsplit


SCHEMA_VERSION = 1
SUITES = ("humanevalplus", "ifeval")
MAX_OUTPUT_TOKENS = 65_536
MAX_REQUEST_TIMEOUT = 21_600.0
LENGTH_FINISH_REASONS = {"length", "max_tokens", "max_output_tokens"}
HUMANEVAL_SYSTEM_INSTRUCTION = (
    "Return the complete Python function, including its signature. "
    "Output only Python code. Do not include tests."
)
PROMPT_POLICY_VERSION = "public-pilot-20261023-v1"


class PilotError(RuntimeError):
    """A benchmark contract or execution precondition failed."""


class RequestFailure(PilotError):
    """An endpoint request failed without a valid terminal completion."""


@dataclass(frozen=True)
class Case:
    index: int
    suite: str
    task_id: str
    prompt: str
    original: Mapping[str, Any]

    @property
    def storage_id(self) -> str:
        digest = hashlib.sha256(self.task_id.encode("utf-8")).hexdigest()[:16]
        return digest


@dataclass(frozen=True)
class Manifest:
    path: Path
    sha256: str
    raw: Mapping[str, Any]
    cases: tuple[Case, ...]


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def _json_bytes(value: Any, *, pretty: bool = True) -> bytes:
    if pretty:
        text = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    else:
        text = json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
    return text.encode("utf-8")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(_json_bytes(value, pretty=False)).hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    try:
        with temporary.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_bytes(path, _json_bytes(value))


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PilotError(f"unreadable or invalid JSON: {path}") from exc


def _plain_int(value: Any, label: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise PilotError(f"{label} must be an integer")
    if not minimum <= value <= maximum:
        raise PilotError(f"{label} must be from {minimum} through {maximum}")
    return value


def _finite_float(value: Any, label: str, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PilotError(f"{label} must be a number")
    parsed = float(value)
    if not math.isfinite(parsed) or not minimum <= parsed <= maximum:
        raise PilotError(f"{label} must be finite and from {minimum:g} through {maximum:g}")
    return parsed


def load_manifest(path: Path) -> Manifest:
    source = path.expanduser().resolve()
    try:
        raw_bytes = source.read_bytes()
        raw = json.loads(raw_bytes.decode("utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PilotError(f"manifest is unreadable or invalid: {source}") from exc
    if not isinstance(raw, dict):
        raise PilotError("manifest must be a JSON object")
    if raw.get("schema_version") != 1:
        raise PilotError("manifest schema_version must be 1")
    _plain_int(raw.get("selection_seed"), "manifest selection_seed", -(2**63), 2**63 - 1)
    selected_count = _plain_int(
        raw.get("selected_count_per_suite"),
        "manifest selected_count_per_suite",
        1,
        10_000,
    )
    rows = raw.get("cases")
    if not isinstance(rows, list):
        raise PilotError("manifest cases must be a list")
    cases: list[Case] = []
    seen: set[tuple[str, str]] = set()
    suite_counts = {suite: 0 for suite in SUITES}
    for index, row in enumerate(rows):
        prefix = f"manifest cases[{index}]"
        if not isinstance(row, dict):
            raise PilotError(f"{prefix} must be an object")
        suite = row.get("suite")
        task_id = row.get("task_id")
        prompt = row.get("prompt")
        original = row.get("original")
        if suite not in SUITES:
            raise PilotError(f"{prefix}.suite must be one of {', '.join(SUITES)}")
        if not isinstance(task_id, str) or not task_id:
            raise PilotError(f"{prefix}.task_id must be a non-empty string")
        if not isinstance(prompt, str) or not prompt:
            raise PilotError(f"{prefix}.prompt must be a non-empty string")
        if not isinstance(original, dict):
            raise PilotError(f"{prefix}.original must be an object")
        if "prompt" in original and original["prompt"] != prompt:
            raise PilotError(f"{prefix}.prompt differs from original.prompt")
        identity = (suite, task_id)
        if identity in seen:
            raise PilotError(f"duplicate manifest case: {suite} {task_id}")
        seen.add(identity)
        suite_counts[suite] += 1
        cases.append(Case(index, suite, task_id, prompt, original))
    expected_total = selected_count * len(SUITES)
    if len(cases) != expected_total or any(
        suite_counts[suite] != selected_count for suite in SUITES
    ):
        raise PilotError(
            "manifest must contain exactly selected_count_per_suite cases for each suite"
        )
    return Manifest(
        path=source,
        sha256=hashlib.sha256(raw_bytes).hexdigest(),
        raw=raw,
        cases=tuple(cases),
    )


def build_messages(case: Case) -> list[dict[str, str]]:
    if case.suite == "humanevalplus":
        return [
            {"role": "system", "content": HUMANEVAL_SYSTEM_INSTRUCTION},
            {"role": "user", "content": case.prompt},
        ]
    if case.suite == "ifeval":
        return [{"role": "user", "content": case.prompt}]
    raise PilotError(f"unsupported suite: {case.suite}")


def build_request_payload(case: Case, settings: Mapping[str, Any]) -> dict[str, Any]:
    # Deliberately use only the public prompt. ``case.original`` can contain
    # grader data and is never passed to this function's result.
    return {
        "model": settings["model"],
        "messages": build_messages(case),
        "max_tokens": settings["max_tokens"],
        "temperature": settings["temperature"],
        "seed": settings["seed"],
        "reasoning_effort": settings["reasoning_effort"],
        "n": 1,
        "stream": True,
        "stream_options": {"include_usage": True},
    }


def run_settings(args: argparse.Namespace) -> dict[str, Any]:
    endpoint = args.endpoint.rstrip("/")
    parsed = urlsplit(endpoint)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise PilotError("endpoint must be an absolute http(s) URL")
    if parsed.query or parsed.fragment:
        raise PilotError("endpoint must not contain a query or fragment")
    if not isinstance(args.model, str) or not args.model:
        raise PilotError("model must be a non-empty API alias")
    if not isinstance(args.model_key, str) or not args.model_key:
        raise PilotError("model-key must be a non-empty reporting key")
    if args.reasoning_effort not in {"low", "medium", "high"}:
        raise PilotError("reasoning-effort must be low, medium, or high")
    return {
        "schema_version": SCHEMA_VERSION,
        "endpoint": endpoint,
        "model": args.model,
        "model_key": args.model_key,
        "max_tokens": _plain_int(args.max_tokens, "max-tokens", 1, MAX_OUTPUT_TOKENS),
        "request_timeout_seconds": _finite_float(
            args.request_timeout, "request-timeout", 1.0, MAX_REQUEST_TIMEOUT
        ),
        "reasoning_effort": args.reasoning_effort,
        "temperature": _finite_float(args.temperature, "temperature", 0.0, 2.0),
        "seed": _plain_int(args.seed, "seed", -(2**63), 2**63 - 1),
        "api_key_env": args.api_key_env,
        "prompt_policy_version": PROMPT_POLICY_VERSION,
        "humaneval_system_instruction": HUMANEVAL_SYSTEM_INSTRUCTION,
        "ifeval_prompt_policy": "verbatim_user_message_only",
        "stream_options_include_usage": True,
    }


def _binding(manifest: Manifest, settings: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_now(),
        "manifest": {
            "path_at_creation": str(manifest.path),
            "sha256": manifest.sha256,
            "case_count": len(manifest.cases),
        },
        "settings": dict(settings),
        "settings_sha256": sha256_json(settings),
    }


def ensure_binding(output_dir: Path, manifest: Manifest, settings: Mapping[str, Any]) -> dict[str, Any]:
    path = output_dir / "run.json"
    expected_settings_hash = sha256_json(settings)
    if path.exists():
        existing = read_json(path)
        if not isinstance(existing, dict):
            raise PilotError("existing run.json must be an object")
        if existing.get("manifest", {}).get("sha256") != manifest.sha256:
            raise PilotError("output directory is bound to a different manifest")
        if existing.get("settings_sha256") != expected_settings_hash:
            raise PilotError("output directory is bound to different run settings")
        return existing
    value = _binding(manifest, settings)
    atomic_write_json(path, value)
    return value


@contextlib.contextmanager
def output_lock(output_dir: Path) -> Iterable[None]:
    output_dir.mkdir(parents=True, exist_ok=True)
    lock_path = output_dir / ".lock"
    with lock_path.open("a+b") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise PilotError(f"another worker holds the output lock: {output_dir}") from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@contextlib.contextmanager
def cancel_on_sigterm() -> Iterable[None]:
    """Convert controller termination into the durable cancellation path."""
    try:
        previous = signal.getsignal(signal.SIGTERM)

        def handle_sigterm(_signum: int, _frame: Any) -> None:
            raise KeyboardInterrupt

        signal.signal(signal.SIGTERM, handle_sigterm)
    except ValueError:
        # Only the main Python thread may install signal handlers.  CLI use is
        # main-threaded; embedders in a worker thread retain their own policy.
        yield
        return
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)


class DurableRawWriter:
    """Append exact response bytes and periodically make them durable."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = path.open("xb")
        self._unsynced = 0
        self._last_sync = time.monotonic()

    def write(self, data: bytes) -> None:
        self._handle.write(data)
        self._unsynced += len(data)
        now = time.monotonic()
        if self._unsynced >= 64 * 1024 or now - self._last_sync >= 1.0:
            self.sync()

    def sync(self) -> None:
        self._handle.flush()
        os.fsync(self._handle.fileno())
        self._unsynced = 0
        self._last_sync = time.monotonic()

    def close(self) -> None:
        self.sync()
        self._handle.close()

    def __enter__(self) -> "DurableRawWriter":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


def _text_content(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        pieces: list[str] = []
        for item in value:
            if isinstance(item, str):
                pieces.append(item)
            elif isinstance(item, dict):
                for key in ("text", "content", "summary"):
                    if isinstance(item.get(key), str):
                        pieces.append(item[key])
                        break
        return "".join(pieces)
    if isinstance(value, dict):
        for key in ("text", "content", "summary"):
            if isinstance(value.get(key), str):
                return value[key]
    return ""


class ChatAccumulator:
    def __init__(self) -> None:
        self.final_parts: list[str] = []
        self.reasoning_parts: list[str] = []
        self.usage: dict[str, Any] | None = None
        self.native_timings: dict[str, Any] | None = None
        self.finish_reason: str | None = None
        self.done_marker = False
        self.first_token_seconds: float | None = None
        self.first_reasoning_seconds: float | None = None
        self.first_final_seconds: float | None = None

    def feed(self, event: Mapping[str, Any], elapsed: float) -> None:
        if event.get("_done_marker") is True:
            self.done_marker = True
            return
        if event.get("error") is not None:
            error = event["error"]
            code = error.get("code") if isinstance(error, dict) else None
            suffix = f" ({code})" if code is not None else ""
            raise RequestFailure(f"server stream error{suffix}")
        if isinstance(event.get("usage"), dict):
            self.usage = dict(event["usage"])
        if isinstance(event.get("timings"), dict):
            # Replace rather than merge: the final native timing object is the
            # endpoint's authoritative snapshot for this request.
            self.native_timings = dict(event["timings"])
        choices = event.get("choices")
        if not isinstance(choices, list):
            return
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            if choice.get("finish_reason") is not None:
                self.finish_reason = str(choice["finish_reason"])
            delta = choice.get("delta")
            if not isinstance(delta, dict):
                delta = choice.get("message") if isinstance(choice.get("message"), dict) else {}
            reasoning_value = delta.get("reasoning_content")
            if reasoning_value is None:
                reasoning_value = delta.get("reasoning", delta.get("reasoning_details"))
            reasoning = _text_content(reasoning_value)
            if reasoning:
                if self.first_token_seconds is None:
                    self.first_token_seconds = elapsed
                if self.first_reasoning_seconds is None:
                    self.first_reasoning_seconds = elapsed
                self.reasoning_parts.append(reasoning)
            content = _text_content(delta.get("content"))
            if content:
                if self.first_token_seconds is None:
                    self.first_token_seconds = elapsed
                if self.first_final_seconds is None:
                    self.first_final_seconds = elapsed
                self.final_parts.append(content)

    @property
    def final_text(self) -> str:
        return "".join(self.final_parts)

    @property
    def reasoning_text(self) -> str:
        return "".join(self.reasoning_parts)


def _feed_sse_body(
    response: http.client.HTTPResponse,
    writer: DurableRawWriter,
    accumulator: ChatAccumulator,
    started: float,
    timeout: float,
    connection: http.client.HTTPConnection,
) -> None:
    data_lines: list[bytes] = []

    def feed_event() -> None:
        if not data_lines:
            return
        payload = b"\n".join(data_lines).decode("utf-8")
        data_lines.clear()
        if payload.strip() == "[DONE]":
            accumulator.feed({"_done_marker": True}, time.monotonic() - started)
            return
        try:
            event = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise RequestFailure(f"invalid streamed JSON: {exc.msg}") from exc
        if not isinstance(event, dict):
            raise RequestFailure("streamed JSON event must be an object")
        accumulator.feed(event, time.monotonic() - started)

    while True:
        remaining = timeout - (time.monotonic() - started)
        if remaining <= 0:
            raise TimeoutError(f"request exceeded {timeout:g}s total timeout")
        if connection.sock is not None:
            connection.sock.settimeout(remaining)
        line = response.readline()
        if not line:
            feed_event()
            break
        writer.write(line)
        stripped = line.rstrip(b"\r\n")
        if stripped == b"":
            feed_event()
        elif stripped.startswith(b"data:"):
            value = stripped[5:]
            if value.startswith(b" "):
                value = value[1:]
            data_lines.append(value)


def _feed_json_body(
    response: http.client.HTTPResponse,
    writer: DurableRawWriter,
    accumulator: ChatAccumulator,
    started: float,
    timeout: float,
    connection: http.client.HTTPConnection,
) -> None:
    chunks: list[bytes] = []
    while True:
        remaining = timeout - (time.monotonic() - started)
        if remaining <= 0:
            raise TimeoutError(f"request exceeded {timeout:g}s total timeout")
        if connection.sock is not None:
            connection.sock.settimeout(remaining)
        chunk = response.read(64 * 1024)
        if not chunk:
            break
        writer.write(chunk)
        chunks.append(chunk)
    try:
        value = json.loads(b"".join(chunks).decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise RequestFailure("response body is not valid JSON") from exc
    if not isinstance(value, dict):
        raise RequestFailure("response JSON must be an object")
    accumulator.feed(value, time.monotonic() - started)


def _usage_int(usage: Mapping[str, Any] | None, key: str) -> int | None:
    if not usage:
        return None
    value = usage.get(key)
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return None


def _nested_usage_int(usage: Mapping[str, Any] | None, parent: str, key: str) -> int | None:
    if not usage or not isinstance(usage.get(parent), dict):
        return None
    value = usage[parent].get(key)
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return None


def _response_result(
    *,
    accumulator: ChatAccumulator,
    started: float,
    started_at: str,
    http_status: int | None,
    content_type: str | None,
    error: Mapping[str, Any] | None,
    cancelled: bool = False,
) -> dict[str, Any]:
    wall = time.monotonic() - started
    stream_complete = accumulator.done_marker or accumulator.finish_reason is not None
    truncated = accumulator.finish_reason in LENGTH_FINISH_REASONS
    if cancelled:
        status = "cancelled"
    elif error is not None:
        status = "api_failure"
    elif truncated:
        status = "truncated"
    elif not stream_complete:
        status = "incomplete"
    else:
        status = "completed"
    usage = accumulator.usage
    return {
        "status": status,
        "response_terminal": stream_complete,
        "final_text": accumulator.final_text,
        "reasoning_text": accumulator.reasoning_text,
        "finish_reason": accumulator.finish_reason,
        "stream_complete": stream_complete,
        "truncated": truncated,
        "usage": usage,
        "native_timings": accumulator.native_timings,
        "counts": {
            "prompt_tokens": _usage_int(usage, "prompt_tokens"),
            "completion_tokens": _usage_int(usage, "completion_tokens"),
            "total_tokens": _usage_int(usage, "total_tokens"),
            "reasoning_tokens": _nested_usage_int(
                usage, "completion_tokens_details", "reasoning_tokens"
            ),
            "cached_prompt_tokens": _nested_usage_int(
                usage, "prompt_tokens_details", "cached_tokens"
            ),
            "final_characters": len(accumulator.final_text),
            "reasoning_characters": len(accumulator.reasoning_text),
            "token_counts_authoritative": usage is not None,
        },
        "timing": {
            "clock": "time.monotonic",
            "wall_seconds": wall,
            "ttft_seconds": accumulator.first_token_seconds,
            "ttft_reasoning_seconds": accumulator.first_reasoning_seconds,
            "ttft_final_seconds": accumulator.first_final_seconds,
        },
        "http": {"status": http_status, "content_type": content_type},
        "started_at": started_at,
        "completed_at": utc_now(),
        "error": dict(error) if error is not None else None,
    }


def request_chat_completion(
    endpoint: str,
    payload: Mapping[str, Any],
    timeout: float,
    raw_path: Path,
    api_key: str | None,
) -> dict[str, Any]:
    target = endpoint.rstrip("/") + "/chat/completions"
    parsed = urlsplit(target)
    connection_cls = (
        http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
    )
    port = parsed.port
    connection = connection_cls(parsed.hostname, port=port, timeout=timeout)
    path = parsed.path or "/"
    if parsed.query:
        path += "?" + parsed.query
    headers = {"Accept": "text/event-stream, application/json", "Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    body = _json_bytes(payload, pretty=False)
    accumulator = ChatAccumulator()
    started = time.monotonic()
    started_at = utc_now()
    http_status: int | None = None
    content_type: str | None = None
    error: dict[str, Any] | None = None
    cancelled = False
    try:
        with DurableRawWriter(raw_path) as writer:
            try:
                connection.request("POST", path, body=body, headers=headers)
                response = connection.getresponse()
                http_status = response.status
                content_type = response.getheader("Content-Type")
                if not 200 <= response.status < 300:
                    while chunk := response.read(64 * 1024):
                        writer.write(chunk)
                    raise RequestFailure(f"HTTP {response.status} {response.reason}")
                if content_type and "text/event-stream" in content_type.lower():
                    _feed_sse_body(
                        response, writer, accumulator, started, timeout, connection
                    )
                else:
                    _feed_json_body(
                        response, writer, accumulator, started, timeout, connection
                    )
            except KeyboardInterrupt:
                cancelled = True
                raise
            except Exception as exc:
                error = {"type": type(exc).__name__, "message": str(exc)}
    except KeyboardInterrupt:
        result = _response_result(
            accumulator=accumulator,
            started=started,
            started_at=started_at,
            http_status=http_status,
            content_type=content_type,
            error={"type": "KeyboardInterrupt", "message": "request cancelled"},
            cancelled=True,
        )
        raise InterruptedError(json.dumps(result, ensure_ascii=False))
    finally:
        connection.close()
    return _response_result(
        accumulator=accumulator,
        started=started,
        started_at=started_at,
        http_status=http_status,
        content_type=content_type,
        error=error,
        cancelled=cancelled,
    )


def _case_directory(output_dir: Path, case: Case) -> Path:
    return output_dir / "cases" / case.suite / case.storage_id


def _attempt_directories(output_dir: Path, case: Case) -> list[Path]:
    case_dir = _case_directory(output_dir, case)
    if not case_dir.exists():
        return []
    return sorted(
        (path for path in case_dir.iterdir() if path.is_dir() and path.name.startswith("attempt-")),
        key=lambda item: item.name,
    )


def _attempt_result(path: Path) -> Mapping[str, Any] | None:
    result_path = path / "result.json"
    if not result_path.exists():
        return None
    value = read_json(result_path)
    if not isinstance(value, dict):
        raise PilotError(f"attempt result must be an object: {result_path}")
    return value


def case_state(output_dir: Path, case: Case, binding: Mapping[str, Any]) -> dict[str, Any]:
    attempts = _attempt_directories(output_dir, case)
    latest_status = "not_started"
    selected: Mapping[str, Any] | None = None
    latest_path: Path | None = None
    for attempt_path in attempts:
        latest_path = attempt_path
        result = _attempt_result(attempt_path)
        if result is None:
            latest_status = "incomplete"
            continue
        if result.get("manifest_sha256") != binding["manifest"]["sha256"]:
            raise PilotError(f"attempt manifest binding mismatch: {attempt_path}")
        if result.get("settings_sha256") != binding["settings_sha256"]:
            raise PilotError(f"attempt settings binding mismatch: {attempt_path}")
        latest_status = str(result.get("status", "invalid"))
        if selected is None and latest_status in {"completed", "truncated"}:
            selected = result
    return {
        "attempt_count": len(attempts),
        "latest_status": latest_status,
        "latest_attempt": latest_path.name if latest_path else None,
        "selected_result": selected,
    }


def _status_document(
    output_dir: Path,
    manifest: Manifest,
    binding: Mapping[str, Any],
    *,
    active: Mapping[str, Any] | None = None,
    grading: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    counts = {
        "total": len(manifest.cases),
        "not_started": 0,
        "completed": 0,
        "truncated": 0,
        "api_failure": 0,
        "incomplete": 0,
        "cancelled": 0,
        "invalid": 0,
    }
    suite_counts = {
        suite: {key: 0 for key in counts if key != "total"} | {"total": 0}
        for suite in SUITES
    }
    attempt_total = 0
    for case in manifest.cases:
        state = case_state(output_dir, case, binding)
        status = state["latest_status"]
        if state["selected_result"] is not None:
            status = str(state["selected_result"]["status"])
        if status not in counts:
            status = "invalid"
        counts[status] += 1
        suite_counts[case.suite]["total"] += 1
        suite_counts[case.suite][status] += 1
        attempt_total += state["attempt_count"]
        records.append(
            {
                "index": case.index,
                "suite": case.suite,
                "task_id": case.task_id,
                "storage_id": case.storage_id,
                "status": status,
                "latest_status": state["latest_status"],
                "attempt_count": state["attempt_count"],
                "latest_attempt": state["latest_attempt"],
            }
        )
    document: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "updated_at": utc_now(),
        "manifest_sha256": manifest.sha256,
        "settings_sha256": binding["settings_sha256"],
        "counts": counts,
        "suite_counts": suite_counts,
        "attempt_count": attempt_total,
        "active": dict(active) if active else None,
        "cases": records,
    }
    existing = output_dir / "status.json"
    if grading is not None:
        document["grading"] = dict(grading)
    elif existing.exists():
        prior = read_json(existing)
        if isinstance(prior, dict) and isinstance(prior.get("grading"), dict):
            document["grading"] = prior["grading"]
    return document


def write_status(
    output_dir: Path,
    manifest: Manifest,
    binding: Mapping[str, Any],
    *,
    active: Mapping[str, Any] | None = None,
    grading: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    value = _status_document(
        output_dir, manifest, binding, active=active, grading=grading
    )
    atomic_write_json(output_dir / "status.json", value)
    return value


def _new_attempt(output_dir: Path, case: Case) -> tuple[int, Path]:
    case_dir = _case_directory(output_dir, case)
    case_dir.mkdir(parents=True, exist_ok=True)
    existing = _attempt_directories(output_dir, case)
    number = len(existing) + 1
    path = case_dir / f"attempt-{number:04d}"
    path.mkdir()
    _fsync_directory(case_dir)
    return number, path


def execute_case(
    output_dir: Path,
    manifest: Manifest,
    binding: Mapping[str, Any],
    settings: Mapping[str, Any],
    case: Case,
    api_key: str | None,
) -> Mapping[str, Any]:
    attempt_number, attempt_dir = _new_attempt(output_dir, case)
    attempt_started = time.monotonic()
    attempt_started_at = utc_now()
    payload = build_request_payload(case, settings)
    request_record = {
        "schema_version": SCHEMA_VERSION,
        "created_at": attempt_started_at,
        "manifest_sha256": manifest.sha256,
        "settings_sha256": binding["settings_sha256"],
        "case": {"index": case.index, "suite": case.suite, "task_id": case.task_id},
        "attempt": attempt_number,
        "url": settings["endpoint"].rstrip("/") + "/chat/completions",
        "payload": payload,
    }
    active = {
        "case_index": case.index,
        "suite": case.suite,
        "task_id": case.task_id,
        "attempt": attempt_number,
        "started_at": attempt_started_at,
    }
    response: Mapping[str, Any] | None = None
    try:
        atomic_write_json(attempt_dir / "request.json", request_record)
        write_status(output_dir, manifest, binding, active=active)
        try:
            response = request_chat_completion(
                settings["endpoint"],
                payload,
                settings["request_timeout_seconds"],
                attempt_dir / "raw.sse",
                api_key,
            )
        except InterruptedError as exc:
            response = json.loads(str(exc))
            raise KeyboardInterrupt
        result = {
            "schema_version": SCHEMA_VERSION,
            "manifest_sha256": manifest.sha256,
            "settings_sha256": binding["settings_sha256"],
            "case": {"index": case.index, "suite": case.suite, "task_id": case.task_id},
            "attempt": attempt_number,
            **response,
        }
        atomic_write_json(attempt_dir / "result.json", result)
        write_status(output_dir, manifest, binding)
        return result
    except KeyboardInterrupt:
        if response is None:
            response = _response_result(
                accumulator=ChatAccumulator(),
                started=attempt_started,
                started_at=attempt_started_at,
                http_status=None,
                content_type=None,
                error={"type": "KeyboardInterrupt", "message": "request cancelled"},
                cancelled=True,
            )
        raw_path = attempt_dir / "raw.sse"
        if not raw_path.exists():
            atomic_write_bytes(raw_path, b"")
        if not (attempt_dir / "request.json").exists():
            atomic_write_json(attempt_dir / "request.json", request_record)
        cancelled_result = {
            "schema_version": SCHEMA_VERSION,
            "manifest_sha256": manifest.sha256,
            "settings_sha256": binding["settings_sha256"],
            "case": {"index": case.index, "suite": case.suite, "task_id": case.task_id},
            "attempt": attempt_number,
            **response,
        }
        atomic_write_json(attempt_dir / "result.json", cancelled_result)
        write_status(output_dir, manifest, binding)
        raise


def run_command(args: argparse.Namespace) -> int:
    manifest = load_manifest(Path(args.manifest))
    settings = run_settings(args)
    output_dir = Path(args.output_dir).expanduser().resolve()
    api_key = os.environ.get(args.api_key_env) if args.api_key_env else None
    with output_lock(output_dir):
        binding = ensure_binding(output_dir, manifest, settings)
        write_status(output_dir, manifest, binding)
        with cancel_on_sigterm():
            try:
                for case in manifest.cases:
                    state = case_state(output_dir, case, binding)
                    if state["selected_result"] is not None:
                        continue
                    if state["attempt_count"] and not args.rerun_failed:
                        continue
                    execute_case(
                        output_dir, manifest, binding, settings, case, api_key
                    )
            except KeyboardInterrupt:
                write_status(output_dir, manifest, binding)
                return 130
        final_status = write_status(output_dir, manifest, binding)
    counts = final_status["counts"]
    return 0 if counts["completed"] + counts["truncated"] == counts["total"] else 1


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    data = b"".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":")).encode("utf-8") + b"\n"
        for row in rows
    )
    atomic_write_bytes(path, data)


def _selected_response(output_dir: Path, case: Case, binding: Mapping[str, Any]) -> tuple[str, str]:
    state = case_state(output_dir, case, binding)
    result = state["selected_result"]
    if result is None:
        return "", state["latest_status"]
    value = result.get("final_text")
    return (value if isinstance(value, str) else ""), str(result.get("status"))


def _run_subprocess(
    command: Sequence[str],
    *,
    cwd: Path,
    env: Mapping[str, str],
    stdout_path: Path,
    stderr_path: Path,
    timeout: float,
) -> dict[str, Any]:
    started = time.monotonic()
    started_at = utc_now()
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    timed_out = False
    returncode: int | None = None
    error: str | None = None
    with stdout_path.open("wb") as stdout_handle, stderr_path.open("wb") as stderr_handle:
        try:
            completed = subprocess.run(
                list(command),
                cwd=cwd,
                env=dict(env),
                stdin=subprocess.DEVNULL,
                stdout=stdout_handle,
                stderr=stderr_handle,
                timeout=timeout,
                check=False,
            )
            returncode = completed.returncode
        except subprocess.TimeoutExpired:
            timed_out = True
            error = f"grader exceeded {timeout:g}s timeout"
        except OSError as exc:
            error = f"{type(exc).__name__}: {exc}"
        finally:
            for handle in (stdout_handle, stderr_handle):
                handle.flush()
                os.fsync(handle.fileno())
    return {
        "command": list(command),
        "cwd": str(cwd),
        "started_at": started_at,
        "completed_at": utc_now(),
        "wall_seconds": time.monotonic() - started,
        "returncode": returncode,
        "timed_out": timed_out,
        "error": error,
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
    }


def _grade_evalplus(
    manifest: Manifest,
    output_dir: Path,
    binding: Mapping[str, Any],
    evalplus_root: Path,
    grading_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    cases = [case for case in manifest.cases if case.suite == "humanevalplus"]
    raw_rows = []
    statuses: dict[str, str] = {}
    for case in cases:
        response, status = _selected_response(output_dir, case, binding)
        raw_rows.append(
            {"task_id": case.task_id, "prompt": case.prompt, "raw_solution": response}
        )
        statuses[case.task_id] = status
    content_hash = sha256_json(raw_rows)[:20]
    suite_dir = grading_dir / "evalplus" / content_hash
    suite_dir.mkdir(parents=True, exist_ok=True)
    raw_path = suite_dir / "raw_samples.jsonl"
    selected_path = suite_dir / "selected_ids.json"
    samples_path = suite_dir / "samples.jsonl"
    _write_jsonl(raw_path, raw_rows)
    atomic_write_json(selected_path, [case.task_id for case in cases])
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "_evalplus-worker",
        "--evalplus-root",
        str(evalplus_root),
        "--raw-samples",
        str(raw_path),
        "--selected-ids",
        str(selected_path),
        "--samples-out",
        str(samples_path),
        "--workers",
        str(args.evalplus_workers),
    ]
    env = dict(os.environ)
    staged_dataset = evalplus_root.parent / "datasets" / "HumanEvalPlus-v0.1.10.jsonl"
    if "HUMANEVAL_OVERRIDE_PATH" not in env and staged_dataset.is_file():
        env["HUMANEVAL_OVERRIDE_PATH"] = str(staged_dataset)
    process = _run_subprocess(
        command,
        cwd=evalplus_root,
        env=env,
        stdout_path=suite_dir / "stdout.log",
        stderr_path=suite_dir / "stderr.log",
        timeout=args.grader_timeout,
    )
    result_path = samples_path.with_name(samples_path.stem + "_eval_results.json")
    value: dict[str, Any] = {
        "status": "grader_error",
        "input_status_counts": {
            status: sum(1 for current in statuses.values() if current == status)
            for status in sorted(set(statuses.values()))
        },
        "raw_samples": str(raw_path),
        "sanitized_samples": str(samples_path),
        "official_results": str(result_path),
        "process": process,
        "dataset_path": env.get("HUMANEVAL_OVERRIDE_PATH"),
    }
    if process["returncode"] != 0 or not result_path.exists():
        value["error"] = process["error"] or f"official EvalPlus exited {process['returncode']}"
        return value
    results = read_json(result_path)
    evaluations = results.get("eval") if isinstance(results, dict) else None
    if not isinstance(evaluations, dict) or set(evaluations) != {case.task_id for case in cases}:
        value["error"] = "official EvalPlus result task set does not match selection"
        return value
    base_pass = 0
    plus_pass = 0
    per_task: list[dict[str, Any]] = []
    for case in cases:
        rows = evaluations.get(case.task_id)
        if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], dict):
            value["error"] = f"official EvalPlus result malformed for {case.task_id}"
            return value
        row = rows[0]
        base_ok = row.get("base_status") == "pass"
        plus_ok = base_ok and row.get("plus_status") == "pass"
        base_pass += int(base_ok)
        plus_pass += int(plus_ok)
        per_task.append(
            {
                "task_id": case.task_id,
                "input_status": statuses[case.task_id],
                "base_status": row.get("base_status"),
                "plus_status": row.get("plus_status"),
            }
        )
    total = len(cases)
    value.update(
        {
            "status": "completed",
            "task_count": total,
            "base_pass_count": base_pass,
            "base_test_failure_count": total - base_pass,
            "plus_pass_count": plus_pass,
            "plus_test_failure_count": total - plus_pass,
            "base_pass_at_1": base_pass / total,
            "plus_pass_at_1": plus_pass / total,
            "per_task": per_task,
            "error": None,
        }
    )
    return value


def _aggregate_ifeval(path: Path) -> dict[str, Any]:
    rows: list[Mapping[str, Any]] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError("row is not an object")
                rows.append(value)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise PilotError(f"invalid official IFEval output: {path}") from exc
    prompt_passes = [row.get("follow_all_instructions") for row in rows]
    if not all(isinstance(value, bool) for value in prompt_passes):
        raise PilotError(f"IFEval output lacks prompt pass booleans: {path}")
    instruction_values: list[bool] = []
    for row in rows:
        values = row.get("follow_instruction_list")
        if not isinstance(values, list) or not all(isinstance(value, bool) for value in values):
            raise PilotError(f"IFEval output lacks instruction pass booleans: {path}")
        instruction_values.extend(values)
    if not rows or not instruction_values:
        raise PilotError(f"IFEval output is empty: {path}")
    return {
        "prompt_count": len(rows),
        "prompt_pass_count": sum(prompt_passes),
        "prompt_failure_count": len(rows) - sum(prompt_passes),
        "prompt_accuracy": sum(prompt_passes) / len(rows),
        "instruction_count": len(instruction_values),
        "instruction_pass_count": sum(instruction_values),
        "instruction_failure_count": len(instruction_values) - sum(instruction_values),
        "instruction_accuracy": sum(instruction_values) / len(instruction_values),
    }


def _grade_ifeval(
    manifest: Manifest,
    output_dir: Path,
    binding: Mapping[str, Any],
    ifeval_root: Path,
    grading_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    cases = [case for case in manifest.cases if case.suite == "ifeval"]
    statuses: dict[str, str] = {}
    responses: list[dict[str, str]] = []
    originals: list[Mapping[str, Any]] = []
    for case in cases:
        response, status = _selected_response(output_dir, case, binding)
        statuses[case.task_id] = status
        responses.append({"prompt": case.prompt, "response": response})
        originals.append(case.original)
    content_hash = sha256_json(responses)[:20]
    suite_dir = grading_dir / "ifeval" / content_hash
    official_output = suite_dir / "official-output"
    official_output.mkdir(parents=True, exist_ok=True)
    input_path = suite_dir / "selected_input_data.jsonl"
    response_path = suite_dir / "input_response_data.jsonl"
    _write_jsonl(input_path, originals)
    _write_jsonl(response_path, responses)
    package_parent = ifeval_root.parent
    command = [
        sys.executable,
        "-m",
        "instruction_following_eval.evaluation_main",
        f"--input_data={input_path}",
        f"--input_response_data={response_path}",
        f"--output_dir={official_output}",
    ]
    env = dict(os.environ)
    existing_path = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(package_parent) + (os.pathsep + existing_path if existing_path else "")
    process = _run_subprocess(
        command,
        cwd=package_parent,
        env=env,
        stdout_path=suite_dir / "stdout.log",
        stderr_path=suite_dir / "stderr.log",
        timeout=args.grader_timeout,
    )
    strict_path = official_output / "eval_results_strict.jsonl"
    loose_path = official_output / "eval_results_loose.jsonl"
    value: dict[str, Any] = {
        "status": "grader_error",
        "input_status_counts": {
            status: sum(1 for current in statuses.values() if current == status)
            for status in sorted(set(statuses.values()))
        },
        "selected_input_data": str(input_path),
        "input_response_data": str(response_path),
        "strict_results": str(strict_path),
        "loose_results": str(loose_path),
        "process": process,
    }
    if process["returncode"] != 0 or not strict_path.exists() or not loose_path.exists():
        value["error"] = process["error"] or f"official IFEval exited {process['returncode']}"
        return value
    try:
        strict = _aggregate_ifeval(strict_path)
        loose = _aggregate_ifeval(loose_path)
    except PilotError as exc:
        value["error"] = str(exc)
        return value
    if strict["prompt_count"] != len(cases) or loose["prompt_count"] != len(cases):
        value["error"] = "official IFEval output count does not match selection"
        return value
    value.update({"status": "completed", "strict": strict, "loose": loose, "error": None})
    return value


def grade_command(args: argparse.Namespace) -> int:
    manifest = load_manifest(Path(args.manifest))
    output_dir = Path(args.output_dir).expanduser().resolve()
    evalplus_root = Path(args.evalplus_root).expanduser().resolve()
    ifeval_root = Path(args.ifeval_root).expanduser().resolve()
    if not evalplus_root.is_dir():
        raise PilotError(f"EvalPlus root is not a directory: {evalplus_root}")
    if not (ifeval_root / "evaluation_main.py").is_file():
        raise PilotError(f"IFEval root lacks evaluation_main.py: {ifeval_root}")
    args.evalplus_workers = _plain_int(
        args.evalplus_workers, "evalplus-workers", 1, 64
    )
    args.grader_timeout = _finite_float(
        args.grader_timeout, "grader-timeout", 1.0, MAX_REQUEST_TIMEOUT
    )
    with output_lock(output_dir):
        binding_path = output_dir / "run.json"
        if not binding_path.exists():
            raise PilotError("output directory has no run.json")
        binding = read_json(binding_path)
        if not isinstance(binding, dict) or binding.get("manifest", {}).get("sha256") != manifest.sha256:
            raise PilotError("output directory is bound to a different manifest")
        grading_dir = output_dir / "grading"
        grading_dir.mkdir(parents=True, exist_ok=True)
        write_status(
            output_dir,
            manifest,
            binding,
            grading={"status": "running", "started_at": utc_now()},
        )
        evalplus = _grade_evalplus(
            manifest, output_dir, binding, evalplus_root, grading_dir, args
        )
        ifeval = _grade_ifeval(
            manifest, output_dir, binding, ifeval_root, grading_dir, args
        )
        status = (
            "completed"
            if evalplus["status"] == "completed" and ifeval["status"] == "completed"
            else "grader_error"
        )
        summary = {
            "schema_version": SCHEMA_VERSION,
            "status": status,
            "created_at": utc_now(),
            "manifest_sha256": manifest.sha256,
            "settings_sha256": binding.get("settings_sha256"),
            "coverage": _status_document(output_dir, manifest, binding)["counts"],
            "evalplus": evalplus,
            "ifeval": ifeval,
        }
        atomic_write_json(grading_dir / "summary.json", summary)
        write_status(
            output_dir,
            manifest,
            binding,
            grading={
                "status": status,
                "completed_at": summary["created_at"],
                "summary": str(grading_dir / "summary.json"),
                "evalplus": {
                    key: evalplus.get(key)
                    for key in (
                        "status",
                        "task_count",
                        "base_pass_count",
                        "base_test_failure_count",
                        "plus_pass_count",
                        "plus_test_failure_count",
                        "error",
                    )
                },
                "ifeval": {
                    "status": ifeval.get("status"),
                    "strict": ifeval.get("strict"),
                    "loose": ifeval.get("loose"),
                    "error": ifeval.get("error"),
                },
            },
        )
    return 0 if status == "completed" else 1


def _evalplus_check_one(arguments: tuple[Any, ...]) -> dict[str, Any]:
    """Apply the official EvalPlus checker primitives to one selected sample."""
    (
        task_id,
        problem,
        solution,
        oracle,
    ) = arguments
    evaluator = importlib.import_module("evalplus.eval")
    base_status, base_details = evaluator.untrusted_check(
        "humaneval",
        solution,
        problem["base_input"],
        problem["entry_point"],
        expected=oracle["base"],
        atol=problem["atol"],
        ref_time=oracle["base_time"],
        fast_check=True,
    )
    plus_status, plus_details = evaluator.untrusted_check(
        "humaneval",
        solution,
        problem["plus_input"],
        problem["entry_point"],
        expected=oracle["plus"],
        atol=problem["atol"],
        ref_time=oracle["plus_time"],
        fast_check=True,
    )

    def failed_input(status: str, details: Any, inputs: Sequence[Any]) -> list[Any]:
        details_list = list(details)
        if status == evaluator.PASS or not details_list:
            return []
        return [inputs[len(details_list) - 1]]

    return {
        "task_id": task_id,
        "solution": solution,
        "base_status": base_status,
        "plus_status": plus_status,
        "base_fail_tests": failed_input(
            base_status, base_details, problem["base_input"]
        ),
        "plus_fail_tests": failed_input(
            plus_status, plus_details, problem["plus_input"]
        ),
    }


def evalplus_worker(args: argparse.Namespace) -> int:
    """Run the pinned official evaluator over an exact selected subset."""
    evalplus_root = Path(args.evalplus_root).resolve()
    sys.path.insert(0, str(evalplus_root))
    data_module = importlib.import_module("evalplus.data")
    sanitize_module = importlib.import_module("evalplus.sanitize")
    evaluator = importlib.import_module("evalplus.eval")
    trusted_module = importlib.import_module("evalplus.gen.util")
    selected_ids = read_json(Path(args.selected_ids))
    if not isinstance(selected_ids, list) or not all(isinstance(item, str) for item in selected_ids):
        raise PilotError("selected EvalPlus IDs must be a string list")
    all_problems = data_module.get_human_eval_plus()
    missing = [task_id for task_id in selected_ids if task_id not in all_problems]
    if missing:
        raise PilotError(f"selected IDs absent from official EvalPlus: {', '.join(missing)}")
    selected = {task_id: all_problems[task_id] for task_id in selected_ids}
    raw_rows: list[Mapping[str, Any]] = []
    for line in Path(args.raw_samples).read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if not isinstance(row, dict):
                raise PilotError("raw EvalPlus sample must be an object")
            raw_rows.append(row)
    if [row.get("task_id") for row in raw_rows] != selected_ids:
        raise PilotError("raw EvalPlus samples do not match selected ID order")
    samples = []
    for row in raw_rows:
        task_id = row["task_id"]
        if row.get("prompt") != selected[task_id]["prompt"]:
            raise PilotError(f"manifest prompt differs from official EvalPlus for {task_id}")
        raw_solution = row.get("raw_solution")
        if not isinstance(raw_solution, str):
            raise PilotError(f"raw solution must be a string for {task_id}")
        solution = sanitize_module.sanitize(
            raw_solution, entrypoint=selected[task_id]["entry_point"]
        )
        samples.append({"task_id": task_id, "solution": solution})
    _write_jsonl(Path(args.samples_out), samples)
    official_hash = data_module.get_human_eval_plus_hash()
    subset_hash = hashlib.sha256(
        (str(official_hash) + "\0" + "\0".join(selected_ids)).encode("utf-8")
    ).hexdigest()
    oracle_path = Path(args.samples_out).with_name(f"oracle-{subset_hash}.pkl")
    if oracle_path.exists():
        with oracle_path.open("rb") as handle:
            oracles = pickle.load(handle)
    else:
        oracles = {}
        for task_id, problem in selected.items():
            canonical_code = problem["prompt"] + problem["canonical_solution"]
            base_expected, base_time = trusted_module.trusted_exec(
                canonical_code,
                problem["base_input"],
                problem["entry_point"],
                record_time=True,
                output_not_none=False,
            )
            plus_expected, plus_time = trusted_module.trusted_exec(
                canonical_code,
                problem["plus_input"],
                problem["entry_point"],
                record_time=True,
                output_not_none=False,
            )
            oracles[task_id] = {
                "base": base_expected,
                "base_time": base_time,
                "plus": plus_expected,
                "plus_time": plus_time,
            }
        temporary = oracle_path.with_name(f".{oracle_path.name}.tmp-{os.getpid()}")
        with temporary.open("wb") as handle:
            pickle.dump(oracles, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, oracle_path)
        _fsync_directory(oracle_path.parent)

    work = [
        (sample["task_id"], selected[sample["task_id"]], sample["solution"], oracles[sample["task_id"]])
        for sample in samples
    ]
    if args.workers == 1:
        checked = [_evalplus_check_one(item) for item in work]
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
            checked = list(executor.map(_evalplus_check_one, work))
    base_correct = sum(row["base_status"] == evaluator.PASS for row in checked)
    plus_correct = sum(
        row["base_status"] == row["plus_status"] == evaluator.PASS for row in checked
    )
    total = len(checked)
    result = {
        "date": utc_now(),
        "hash": subset_hash,
        "official_dataset_hash": official_hash,
        "selection": selected_ids,
        "eval": {row["task_id"]: [row] for row in checked},
        "pass_at_k": {
            "base": {"pass@1": base_correct / total},
            "plus": {"pass@1": plus_correct / total},
        },
        "grader": {
            "implementation": "official EvalPlus v0.3.1 primitives",
            "sanitize": "evalplus.sanitize.sanitize",
            "oracle": "evalplus.gen.util.trusted_exec",
            "check": "evalplus.eval.untrusted_check",
            "pass_constant": evaluator.PASS,
        },
    }
    result_path = Path(args.samples_out).with_name(
        Path(args.samples_out).stem + "_eval_results.json"
    )
    atomic_write_json(result_path, result)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run and grade a frozen endpoint-only public benchmark pilot."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="query one existing model endpoint")
    run_parser.add_argument("--manifest", required=True)
    run_parser.add_argument("--endpoint", required=True)
    run_parser.add_argument("--model", required=True, help="OpenAI API model alias")
    run_parser.add_argument("--model-key", required=True, help="stable reporting model key")
    run_parser.add_argument("--output-dir", required=True)
    run_parser.add_argument("--max-tokens", type=int, default=65_536)
    run_parser.add_argument("--request-timeout", type=float, default=21_600.0)
    run_parser.add_argument("--reasoning-effort", default="medium")
    run_parser.add_argument("--temperature", type=float, default=0.0)
    run_parser.add_argument("--seed", type=int, default=20_261_023)
    run_parser.add_argument(
        "--api-key-env",
        default=None,
        help="optional environment variable containing the API key",
    )
    run_parser.add_argument(
        "--rerun-failed",
        action="store_true",
        help="explicitly create a new attempt for API-failed, cancelled, or incomplete cases",
    )

    grade_parser = subparsers.add_parser("grade", help="invoke pinned official graders")
    grade_parser.add_argument("--manifest", required=True)
    grade_parser.add_argument("--output-dir", required=True)
    grade_parser.add_argument("--evalplus-root", required=True)
    grade_parser.add_argument("--ifeval-root", required=True)
    grade_parser.add_argument("--evalplus-workers", type=int, default=4)
    grade_parser.add_argument("--grader-timeout", type=float, default=21_600.0)

    worker_parser = subparsers.add_parser("_evalplus-worker", help=argparse.SUPPRESS)
    worker_parser.add_argument("--evalplus-root", required=True)
    worker_parser.add_argument("--raw-samples", required=True)
    worker_parser.add_argument("--selected-ids", required=True)
    worker_parser.add_argument("--samples-out", required=True)
    worker_parser.add_argument("--workers", type=int, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            return run_command(args)
        if args.command == "grade":
            return grade_command(args)
        if args.command == "_evalplus-worker":
            return evalplus_worker(args)
        raise PilotError(f"unknown command: {args.command}")
    except PilotError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
