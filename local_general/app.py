from __future__ import annotations

from contextlib import asynccontextmanager
from copy import deepcopy
from datetime import datetime, timezone
import asyncio
import json
import math
import secrets
import time
from typing import Any, AsyncIterator, Awaitable, Callable, Mapping
from urllib.parse import urlsplit, urlunsplit
import uuid

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse

from .config import Settings


POLICY_HEADER = "X-Local-General-Reasoning-Policy"
POLICY_VALUE = "reasoning_effort=metadata-only; ollama-think=display-only"
KEEP_ALIVE_HEADER = "X-Local-General-Keep-Alive-Policy"
KEEP_ALIVE_VALUE = "server-managed; no per-request unloading"
LEGACY_ALIAS = "gpt-oss-20b:64k"


class GatewayError(Exception):
    def __init__(
        self,
        status_code: int,
        message: str,
        *,
        error_type: str = "invalid_request_error",
        code: str | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.message = message
        self.error_type = error_type
        self.code = code
        self.headers = dict(headers or {})


class CapacityLease:
    def __init__(self, limiter: "CapacityLimiter") -> None:
        self._limiter = limiter
        self._released = False

    async def release(self) -> None:
        if self._released:
            return
        self._released = True
        await self._limiter.release()


class CapacityLimiter:
    """A small FIFO-ish active/pending limiter with cancellation-safe accounting."""

    def __init__(self, active_limit: int, pending_limit: int, timeout: float) -> None:
        self.active_limit = active_limit
        self.pending_limit = pending_limit
        self.timeout = timeout
        self.active = 0
        self.pending = 0
        self._condition = asyncio.Condition()

    async def acquire(self) -> CapacityLease:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.timeout
        async with self._condition:
            if self.active < self.active_limit and self.pending == 0:
                self.active += 1
                return CapacityLease(self)
            if self.pending >= self.pending_limit:
                raise GatewayError(
                    429,
                    "request queue is full",
                    error_type="rate_limit_error",
                    code="queue_full",
                    headers={"Retry-After": "1"},
                )
            self.pending += 1
            try:
                while self.active >= self.active_limit:
                    remaining = deadline - loop.time()
                    if remaining <= 0:
                        raise asyncio.TimeoutError
                    await asyncio.wait_for(self._condition.wait(), timeout=remaining)
                self.pending -= 1
                self.active += 1
                return CapacityLease(self)
            except asyncio.TimeoutError as exc:
                self.pending -= 1
                self._condition.notify_all()
                raise GatewayError(
                    429,
                    "timed out waiting for model capacity",
                    error_type="rate_limit_error",
                    code="queue_timeout",
                    headers={"Retry-After": "1"},
                ) from exc
            except BaseException:
                self.pending -= 1
                self._condition.notify_all()
                raise

    async def release(self) -> None:
        async with self._condition:
            if self.active <= 0:
                raise RuntimeError("capacity lease released without an active request")
            self.active -= 1
            self._condition.notify_all()


class RequestIdMiddleware:
    """Attach an opaque gateway-generated request ID without buffering streams."""

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        request_id = uuid.uuid4().hex
        scope.setdefault("state", {})["request_id"] = request_id

        async def send_with_request_id(message: dict[str, Any]) -> None:
            if message.get("type") == "http.response.start":
                headers = list(message.get("headers", []))
                headers.append((b"x-request-id", request_id.encode("ascii")))
                message["headers"] = headers
            await send(message)

        await self.app(scope, receive, send_with_request_id)


class GatewayService:
    def __init__(
        self,
        settings: Settings,
        upstream_client: httpx.AsyncClient,
        health_client: httpx.AsyncClient,
    ) -> None:
        self.settings = settings
        self.upstream_client = upstream_client
        self.health_client = health_client
        self.capacity = CapacityLimiter(
            settings.max_active, settings.max_pending, settings.queue_timeout_seconds
        )

    @property
    def response_headers(self) -> dict[str, str]:
        return {
            POLICY_HEADER: POLICY_VALUE,
            KEEP_ALIVE_HEADER: KEEP_ALIVE_VALUE,
            "X-Local-General-Context-Window": str(self.settings.context_window),
            "X-Local-General-Max-Output-Tokens": str(self.settings.max_output_tokens),
        }

    def upstream_headers(self, request_id: str | None = None) -> dict[str, str]:
        headers = {"Accept": "application/json"}
        if self.settings.upstream_api_token:
            headers["Authorization"] = f"Bearer {self.settings.upstream_api_token}"
        if request_id:
            headers["X-Request-ID"] = request_id
        return headers

    def upstream_url(self, path: str) -> str:
        return f"{self.settings.upstream_base_url.rstrip('/')}/{path.lstrip('/')}"

    def health_url(self) -> str:
        split = urlsplit(self.settings.upstream_base_url)
        path = split.path.rstrip("/")
        if path.endswith("/v1"):
            path = path[:-3]
        return urlunsplit((split.scheme, split.netloc, f"{path}/health", "", ""))

    async def authorize(self, request: Request) -> None:
        authorization = request.headers.get("authorization", "")
        scheme, separator, supplied = authorization.partition(" ")
        if (
            not separator
            or scheme.lower() != "bearer"
            or not secrets.compare_digest(supplied, self.settings.api_token)
        ):
            raise GatewayError(
                401,
                "invalid or missing bearer token",
                error_type="authentication_error",
                code="invalid_api_key",
                headers={"WWW-Authenticate": "Bearer"},
            )

    def validate_model(self, value: Any) -> None:
        if not isinstance(value, str) or not value:
            raise GatewayError(400, "model must be a non-empty string", code="invalid_model")
        accepted = {self.settings.canonical_model, f"{self.settings.canonical_model}:latest"}
        if self.settings.enable_legacy_gpt_oss_alias:
            accepted.update({LEGACY_ALIAS, f"{LEGACY_ALIAS}:latest"})
        if value not in accepted:
            raise GatewayError(
                404,
                f"model {value!r} is not available",
                error_type="not_found_error",
                code="model_not_found",
            )

    def validate_output_budget(self, payload: dict[str, Any]) -> None:
        values: list[tuple[str, Any]] = []
        for name in ("max_tokens", "max_completion_tokens"):
            if name in payload:
                values.append((name, payload[name]))
        for name, value in values:
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise GatewayError(400, f"{name} must be a positive integer", code=name)
            if value > self.settings.max_output_tokens:
                raise GatewayError(
                    400,
                    f"{name} exceeds the server output cap of {self.settings.max_output_tokens}",
                    code="output_budget_exceeded",
                )
        if len(values) == 2 and values[0][1] != values[1][1]:
            raise GatewayError(
                400,
                "max_tokens and max_completion_tokens must match when both are supplied",
                code="conflicting_output_budgets",
            )

    def prepare_openai_chat(self, body: dict[str, Any]) -> dict[str, Any]:
        self.validate_model(body.get("model"))
        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            raise GatewayError(400, "messages must be a non-empty array", code="invalid_messages")
        for index, message in enumerate(messages):
            if not isinstance(message, dict) or not isinstance(message.get("role"), str):
                raise GatewayError(
                    400, f"messages[{index}] must contain a string role", code="invalid_messages"
                )
        payload = deepcopy(body)
        payload["model"] = self.settings.upstream_model
        if "stream" in payload and not isinstance(payload["stream"], bool):
            raise GatewayError(400, "stream must be a boolean", code="invalid_stream")
        if "n" in payload and (
            isinstance(payload["n"], bool)
            or not isinstance(payload["n"], int)
            or payload["n"] != 1
        ):
            raise GatewayError(
                400,
                "local-general supports exactly one completion per request (n=1)",
                code="unsupported_completion_count",
            )
        for context_field in ("num_ctx", "n_ctx", "context_length"):
            if context_field not in payload:
                continue
            if payload[context_field] != self.settings.context_window:
                raise GatewayError(
                    400,
                    f"{context_field} cannot override the server context of "
                    f"{self.settings.context_window}",
                    code="unsupported_context_override",
                )
            payload.pop(context_field)
        effort = payload.pop("reasoning_effort", None)
        if effort is not None and effort not in {"low", "medium", "high"}:
            raise GatewayError(
                400,
                "reasoning_effort is inert metadata here and must be low, medium, or high",
                code="invalid_reasoning_effort",
            )
        self.validate_output_budget(payload)
        if "max_tokens" not in payload and "max_completion_tokens" not in payload:
            payload["max_tokens"] = self.settings.default_max_tokens
        return payload

    def prepare_ollama(self, body: dict[str, Any], *, generate: bool) -> tuple[dict[str, Any], bool]:
        self.validate_model(body.get("model"))
        common = {"model", "stream", "format", "options", "think", "keep_alive"}
        allowed = common | ({"prompt", "system"} if generate else {"messages", "tools"})
        unsupported = sorted(set(body) - allowed)
        if unsupported:
            raise GatewayError(
                400,
                f"unsupported Ollama request field(s): {', '.join(unsupported)}",
                code="unsupported_field",
            )
        stream = body.get("stream", True)
        if not isinstance(stream, bool):
            raise GatewayError(400, "stream must be a boolean", code="invalid_stream")
        think = body.get("think", True)
        if not isinstance(think, bool):
            raise GatewayError(
                400, "think must be a boolean display preference", code="invalid_think"
            )
        if "keep_alive" in body:
            keep_alive = body["keep_alive"]
            if (
                isinstance(keep_alive, bool)
                or not isinstance(keep_alive, (str, int, float))
                or isinstance(keep_alive, str)
                and not keep_alive.strip()
                or isinstance(keep_alive, float)
                and not math.isfinite(keep_alive)
            ):
                raise GatewayError(
                    400,
                    "keep_alive must be a non-empty duration string or finite number",
                    code="invalid_keep_alive",
                )
        if generate:
            prompt = body.get("prompt")
            if not isinstance(prompt, str):
                raise GatewayError(400, "prompt must be a string", code="invalid_prompt")
            messages: list[dict[str, Any]] = []
            if "system" in body:
                if not isinstance(body["system"], str):
                    raise GatewayError(400, "system must be a string", code="invalid_system")
                messages.append({"role": "system", "content": body["system"]})
            messages.append({"role": "user", "content": prompt})
        else:
            messages = body.get("messages")
            if not isinstance(messages, list) or not messages:
                raise GatewayError(
                    400, "messages must be a non-empty array", code="invalid_messages"
                )
            for index, message in enumerate(messages):
                if not isinstance(message, dict) or not isinstance(message.get("role"), str):
                    raise GatewayError(
                        400,
                        f"messages[{index}] must contain a string role",
                        code="invalid_messages",
                    )
        normalized_messages = (
            messages if generate else _normalize_ollama_messages(messages)
        )
        payload: dict[str, Any] = {
            "model": self.settings.upstream_model,
            "messages": deepcopy(normalized_messages),
            "stream": stream,
        }
        if not generate and "tools" in body:
            if not isinstance(body["tools"], list):
                raise GatewayError(400, "tools must be an array", code="invalid_tools")
            payload["tools"] = deepcopy(body["tools"])
        response_format = _ollama_response_format(body.get("format"))
        if response_format is not None:
            payload["response_format"] = response_format
        options = body.get("options", {})
        if not isinstance(options, dict):
            raise GatewayError(400, "options must be an object", code="invalid_options")
        option_names = {
            "num_predict",
            "temperature",
            "top_p",
            "top_k",
            "min_p",
            "repeat_penalty",
            "seed",
            "num_ctx",
        }
        unsupported_options = sorted(set(options) - option_names)
        if unsupported_options:
            raise GatewayError(
                400,
                f"unsupported Ollama option(s): {', '.join(unsupported_options)}",
                code="unsupported_option",
            )
        if "num_ctx" in options:
            num_ctx = options["num_ctx"]
            if (
                isinstance(num_ctx, bool)
                or not isinstance(num_ctx, int)
                or num_ctx != self.settings.context_window
            ):
                raise GatewayError(
                    400,
                    f"num_ctx cannot override the server context of {self.settings.context_window}",
                    code="unsupported_context_override",
                )
        mapping = {
            "num_predict": "max_tokens",
            "temperature": "temperature",
            "top_p": "top_p",
            "top_k": "top_k",
            "min_p": "min_p",
            "repeat_penalty": "repeat_penalty",
            "seed": "seed",
        }
        for source, target in mapping.items():
            if source in options:
                payload[target] = options[source]
        for name in ("temperature", "top_p", "min_p", "repeat_penalty"):
            if name in payload and (
                isinstance(payload[name], bool) or not isinstance(payload[name], (int, float))
            ):
                raise GatewayError(400, f"{name} must be numeric", code=f"invalid_{name}")
        if "top_k" in payload and (
            isinstance(payload["top_k"], bool)
            or not isinstance(payload["top_k"], int)
            or payload["top_k"] < 0
        ):
            raise GatewayError(400, "top_k must be a non-negative integer", code="invalid_top_k")
        if "seed" in payload and (
            isinstance(payload["seed"], bool) or not isinstance(payload["seed"], int)
        ):
            raise GatewayError(400, "seed must be an integer", code="invalid_seed")
        self.validate_output_budget(payload)
        if "max_tokens" not in payload:
            payload["max_tokens"] = self.settings.default_max_tokens
        return payload, think

    async def open_stream(
        self, payload: dict[str, Any], request_id: str
    ) -> tuple[httpx.Response, CapacityLease]:
        lease = await self.capacity.acquire()
        try:
            request = self.upstream_client.build_request(
                "POST",
                self.upstream_url("chat/completions"),
                json=payload,
                headers=self.upstream_headers(request_id),
            )
            response = await self.upstream_client.send(request, stream=True)
            if response.status_code >= 400:
                data = await _read_capped(response, self.settings.max_request_bytes)
                await response.aclose()
                raise _upstream_error(response.status_code, data)
            content_type = response.headers.get("content-type", "").split(";", 1)[0].strip().lower()
            if content_type != "text/event-stream":
                await _read_capped(response, self.settings.max_request_bytes)
                await response.aclose()
                raise GatewayError(
                    502,
                    "model upstream did not return an SSE stream",
                    error_type="upstream_error",
                    code="invalid_upstream_response",
                )
            return response, lease
        except BaseException:
            await lease.release()
            raise

    async def request_json(self, payload: dict[str, Any], request_id: str) -> dict[str, Any]:
        lease = await self.capacity.acquire()
        try:
            try:
                response = await self.upstream_client.post(
                    self.upstream_url("chat/completions"),
                    json=payload,
                    headers=self.upstream_headers(request_id),
                )
            except httpx.TimeoutException as exc:
                raise GatewayError(
                    504,
                    "model upstream timed out",
                    error_type="upstream_timeout",
                    code="upstream_timeout",
                ) from exc
            except httpx.HTTPError as exc:
                raise GatewayError(
                    502,
                    "model upstream is unavailable",
                    error_type="upstream_error",
                    code="upstream_unavailable",
                ) from exc
            if response.status_code >= 400:
                raise _upstream_error(response.status_code, response.content)
            try:
                result = response.json()
            except (ValueError, UnicodeDecodeError) as exc:
                raise GatewayError(
                    502,
                    "model upstream returned invalid JSON",
                    error_type="upstream_error",
                    code="invalid_upstream_response",
                ) from exc
            if not isinstance(result, dict):
                raise GatewayError(
                    502,
                    "model upstream returned an invalid response object",
                    error_type="upstream_error",
                    code="invalid_upstream_response",
                )
            return result
        finally:
            await lease.release()


_NO_RESULT = object()


async def _wait_for_disconnect(request: Request) -> bool:
    """Wait for ASGI's disconnect message after the request body has been consumed."""

    while True:
        message = await request.receive()
        if message["type"] == "http.disconnect":
            return True


async def _cancel_task(task: asyncio.Task[Any]) -> Any:
    if not task.done():
        task.cancel()
    try:
        return await task
    except asyncio.CancelledError:
        return _NO_RESULT
    except Exception:
        return _NO_RESULT


async def _cancel_disconnect_monitor(task: asyncio.Task[bool]) -> bool:
    if not task.done():
        task.cancel()
    try:
        return await task
    except asyncio.CancelledError:
        return False


async def _await_with_disconnect(
    request: Request,
    operation: Awaitable[Any],
    *,
    discard_result: Callable[[Any], Awaitable[None]] | None = None,
) -> Any:
    """Cancel queued/upstream work promptly if ASGI reports a vanished client."""

    operation_task = asyncio.create_task(operation)
    disconnect_task = asyncio.create_task(_wait_for_disconnect(request))
    result: Any = _NO_RESULT
    result_discarded = False
    try:
        done, _ = await asyncio.wait(
            {operation_task, disconnect_task}, return_when=asyncio.FIRST_COMPLETED
        )
        disconnected = False
        if disconnect_task in done:
            try:
                disconnected = disconnect_task.result()
            except Exception:
                # A non-conforming embedding transport may stop exposing its
                # receive channel. The in-flight operation remains usable.
                disconnected = False
        if disconnected:
            result = await _cancel_task(operation_task)
            if result is not _NO_RESULT and discard_result is not None:
                await discard_result(result)
                result_discarded = True
            raise asyncio.CancelledError

        result = await operation_task
        # Closing the monitor is also the hand-off to StreamingResponse. If a
        # disconnect won that race, dispose of any opened upstream stream here.
        if await _cancel_disconnect_monitor(disconnect_task):
            if discard_result is not None:
                await discard_result(result)
                result_discarded = True
            raise asyncio.CancelledError
        return result
    except asyncio.CancelledError:
        if result is _NO_RESULT:
            result = await _cancel_task(operation_task)
        if (
            result is not _NO_RESULT
            and discard_result is not None
            and not result_discarded
        ):
            await discard_result(result)
        raise
    finally:
        if not disconnect_task.done():
            disconnect_task.cancel()
        try:
            await disconnect_task
        except asyncio.CancelledError:
            pass
        except Exception:
            pass


async def _discard_open_stream(opened: Any) -> None:
    response, lease = opened
    try:
        await response.aclose()
    finally:
        await lease.release()


def _normalize_ollama_messages(messages: list[Any]) -> list[dict[str, Any]]:
    """Translate Ollama-native tool turns into OpenAI-compatible messages."""

    normalized: list[dict[str, Any]] = []
    pending_calls: list[tuple[str, str]] = []
    for message_index, original in enumerate(messages):
        if not isinstance(original, dict) or not isinstance(original.get("role"), str):
            raise GatewayError(
                400,
                f"messages[{message_index}] must contain a string role",
                code="invalid_messages",
            )
        message = deepcopy(original)
        if message["role"] == "assistant" and "tool_calls" in message:
            tool_calls = message["tool_calls"]
            if not isinstance(tool_calls, list):
                raise GatewayError(
                    400,
                    f"messages[{message_index}].tool_calls must be an array",
                    code="invalid_tool_calls",
                )
            for call_index, call in enumerate(tool_calls):
                if not isinstance(call, dict) or not isinstance(call.get("function"), dict):
                    raise GatewayError(
                        400,
                        f"messages[{message_index}].tool_calls[{call_index}] is invalid",
                        code="invalid_tool_calls",
                    )
                function = call["function"]
                name = function.get("name")
                if not isinstance(name, str) or not name:
                    raise GatewayError(
                        400,
                        f"messages[{message_index}].tool_calls[{call_index}] has no function name",
                        code="invalid_tool_calls",
                    )
                call_id = call.get("id")
                if not isinstance(call_id, str) or not call_id:
                    call_id = f"ollama_call_{message_index}_{call_index}"
                    call["id"] = call_id
                call.setdefault("type", "function")
                arguments = function.get("arguments")
                if isinstance(arguments, dict):
                    function["arguments"] = json.dumps(arguments, separators=(",", ":"))
                elif arguments is not None and not isinstance(arguments, str):
                    raise GatewayError(
                        400,
                        f"messages[{message_index}].tool_calls[{call_index}] arguments "
                        "must be an object or JSON string",
                        code="invalid_tool_arguments",
                    )
                pending_calls.append((name, call_id))
        elif message["role"] == "tool":
            call_id = message.get("tool_call_id")
            tool_name = message.pop("tool_name", None)
            if isinstance(call_id, str) and call_id:
                pending_calls = [item for item in pending_calls if item[1] != call_id]
            elif isinstance(tool_name, str) and tool_name:
                match = next(
                    (index for index, item in enumerate(pending_calls) if item[0] == tool_name),
                    None,
                )
                if match is None:
                    raise GatewayError(
                        400,
                        f"messages[{message_index}] tool_name has no matching tool call",
                        code="unmatched_tool_result",
                    )
                _, call_id = pending_calls.pop(match)
                message["tool_call_id"] = call_id
            else:
                raise GatewayError(
                    400,
                    f"messages[{message_index}] tool result needs tool_call_id or tool_name",
                    code="invalid_tool_result",
                )
        normalized.append(message)
    return normalized


def _ollama_response_format(value: Any) -> dict[str, Any] | None:
    if value is None or value == "":
        return None
    if value == "json":
        return {"type": "json_object"}
    if isinstance(value, dict):
        return {
            "type": "json_schema",
            "json_schema": {"name": "ollama_response", "strict": True, "schema": deepcopy(value)},
        }
    raise GatewayError(
        400, "format must be 'json' or a JSON Schema object", code="invalid_format"
    )


async def _read_json_object(request: Request, limit: int) -> dict[str, Any]:
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            if int(content_length) > limit:
                raise GatewayError(413, "request body is too large", code="request_too_large")
        except ValueError as exc:
            raise GatewayError(400, "invalid Content-Length", code="invalid_request") from exc
    body_buffer = bytearray()
    async for chunk in request.stream():
        if len(body_buffer) + len(chunk) > limit:
            raise GatewayError(413, "request body is too large", code="request_too_large")
        body_buffer.extend(chunk)
    body = bytes(body_buffer)
    try:
        value = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise GatewayError(400, "request body must be valid JSON", code="invalid_json") from exc
    if not isinstance(value, dict):
        raise GatewayError(400, "request body must be a JSON object", code="invalid_json")
    return value


def _upstream_error(status_code: int, body: bytes) -> GatewayError:
    message = "model upstream rejected the request"
    try:
        parsed = json.loads(body[: 64 * 1024])
        if isinstance(parsed, dict):
            error = parsed.get("error", parsed.get("detail"))
            if error is None:
                error = parsed.get("message")
            if isinstance(error, dict):
                error = error.get("message")
            if isinstance(error, str) and error.strip():
                message = error.strip()[:500]
    except (ValueError, UnicodeDecodeError):
        pass
    if status_code in {401, 403} or status_code >= 500:
        public_status = 502
    elif status_code == 408:
        public_status = 504
    else:
        public_status = status_code
    return GatewayError(
        public_status,
        message,
        error_type="upstream_error",
        code="upstream_rejected_request",
    )


async def _read_capped(response: httpx.Response, limit: int) -> bytes:
    data = bytearray()
    async for chunk in response.aiter_bytes(chunk_size=64 * 1024):
        remaining = limit - len(data)
        if remaining <= 0:
            break
        data.extend(chunk[:remaining])
    return bytes(data)


async def _iter_sse_data(response: httpx.Response, max_event_bytes: int) -> AsyncIterator[bytes]:
    line_buffer = bytearray()
    data_lines: list[bytes] = []
    event_size = 0
    # Do not request a fixed decoded chunk size here. httpx would wait until
    # that many bytes accumulate, delaying the first token on sparse streams.
    async for chunk in response.aiter_bytes():
        position = 0
        while position < len(chunk):
            newline = chunk.find(b"\n", position)
            if newline < 0:
                remainder = chunk[position:]
                if len(line_buffer) + len(remainder) > max_event_bytes:
                    raise GatewayError(
                        502, "upstream SSE line is too large", code="invalid_upstream_sse"
                    )
                line_buffer.extend(remainder)
                break
            fragment = chunk[position:newline]
            if len(line_buffer) + len(fragment) > max_event_bytes:
                raise GatewayError(
                    502, "upstream SSE line is too large", code="invalid_upstream_sse"
                )
            line_buffer.extend(fragment)
            line = bytes(line_buffer).rstrip(b"\r")
            line_buffer.clear()
            position = newline + 1
            event_size += len(line) + 1
            if event_size > max_event_bytes:
                raise GatewayError(502, "upstream SSE event is too large", code="invalid_upstream_sse")
            if not line:
                if data_lines:
                    yield b"\n".join(data_lines)
                data_lines = []
                event_size = 0
            elif line.startswith(b"data:"):
                value = line[5:]
                if value.startswith(b" "):
                    value = value[1:]
                data_lines.append(value)
    if line_buffer:
        line = bytes(line_buffer).rstrip(b"\r")
        event_size += len(line)
        if event_size > max_event_bytes:
            raise GatewayError(502, "upstream SSE event is too large", code="invalid_upstream_sse")
        if line.startswith(b"data:"):
            value = line[5:]
            data_lines.append(value[1:] if value.startswith(b" ") else value)
    if data_lines:
        yield b"\n".join(data_lines)


def _canonicalize_openai(value: dict[str, Any], model: str) -> dict[str, Any]:
    result = deepcopy(value)
    if "model" in result:
        result["model"] = model
    return result


def _openai_error_payload(error: GatewayError) -> dict[str, Any]:
    return {
        "error": {
            "message": error.message,
            "type": error.error_type,
            "param": None,
            "code": error.code,
        }
    }


def _created_at() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _usage_counts(usage: Any) -> dict[str, int]:
    if not isinstance(usage, dict):
        return {}
    result: dict[str, int] = {}
    prompt = usage.get("prompt_tokens", usage.get("input_tokens"))
    completion = usage.get("completion_tokens", usage.get("output_tokens"))
    if isinstance(prompt, int) and not isinstance(prompt, bool) and prompt >= 0:
        result["prompt_eval_count"] = prompt
    if isinstance(completion, int) and not isinstance(completion, bool) and completion >= 0:
        result["eval_count"] = completion
    return result


def _ollama_tool_calls(value: Any, *, complete: bool) -> Any:
    if not isinstance(value, list):
        return value
    result = deepcopy(value)
    if complete:
        for call in result:
            if not isinstance(call, dict):
                continue
            function = call.get("function")
            if not isinstance(function, dict):
                continue
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                try:
                    parsed = json.loads(arguments)
                except json.JSONDecodeError:
                    continue
                if isinstance(parsed, dict):
                    function["arguments"] = parsed
    return result


def _ollama_nonstream(
    upstream: dict[str, Any], *, generate: bool, expose_thinking: bool, elapsed_ns: int, model: str
) -> dict[str, Any]:
    choices = upstream.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise GatewayError(502, "upstream response has no completion choice", code="invalid_upstream_response")
    choice = choices[0]
    message = choice.get("message")
    if not isinstance(message, dict):
        raise GatewayError(502, "upstream response has no assistant message", code="invalid_upstream_response")
    content = message.get("content")
    if content is None:
        content = ""
    if not isinstance(content, str):
        raise GatewayError(502, "upstream assistant content is invalid", code="invalid_upstream_response")
    response: dict[str, Any] = {
        "model": model,
        "created_at": _created_at(),
        "done": True,
        "total_duration": elapsed_ns,
    }
    finish_reason = choice.get("finish_reason")
    if isinstance(finish_reason, str):
        response["done_reason"] = finish_reason
    reasoning = message.get("reasoning_content")
    if generate:
        response["response"] = content
        if expose_thinking and isinstance(reasoning, str) and reasoning:
            response["thinking"] = reasoning
    else:
        ollama_message: dict[str, Any] = {"role": "assistant", "content": content}
        if expose_thinking and isinstance(reasoning, str) and reasoning:
            ollama_message["thinking"] = reasoning
        if "tool_calls" in message:
            ollama_message["tool_calls"] = _ollama_tool_calls(
                message["tool_calls"], complete=True
            )
        response["message"] = ollama_message
    response.update(_usage_counts(upstream.get("usage")))
    return response


async def _openai_stream(
    response: httpx.Response,
    lease: CapacityLease,
    settings: Settings,
) -> AsyncIterator[bytes]:
    done = False
    try:
        try:
            async for data in _iter_sse_data(response, settings.max_sse_event_bytes):
                if data.strip() == b"[DONE]":
                    done = True
                    yield b"data: [DONE]\n\n"
                    break
                try:
                    value = json.loads(data)
                    if not isinstance(value, dict):
                        raise ValueError
                except (ValueError, UnicodeDecodeError):
                    raise GatewayError(502, "upstream sent invalid SSE JSON", code="invalid_upstream_sse")
                value = _canonicalize_openai(value, settings.canonical_model)
                yield b"data: " + json.dumps(value, separators=(",", ":")).encode() + b"\n\n"
        except (httpx.HTTPError, GatewayError) as exc:
            error = (
                exc
                if isinstance(exc, GatewayError)
                else GatewayError(502, "model upstream stream failed", code="upstream_stream_failed")
            )
            yield b"data: " + json.dumps(_openai_error_payload(error), separators=(",", ":")).encode() + b"\n\n"
        if not done:
            yield b"data: [DONE]\n\n"
    finally:
        try:
            await response.aclose()
        finally:
            await lease.release()


async def _ollama_stream(
    response: httpx.Response,
    lease: CapacityLease,
    settings: Settings,
    *,
    generate: bool,
    expose_thinking: bool,
    started_ns: int,
) -> AsyncIterator[bytes]:
    finish_reason: str | None = None
    usage: dict[str, int] = {}
    try:
        try:
            async for data in _iter_sse_data(response, settings.max_sse_event_bytes):
                if data.strip() == b"[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                    if not isinstance(chunk, dict):
                        raise ValueError
                except (ValueError, UnicodeDecodeError):
                    raise GatewayError(502, "upstream sent invalid SSE JSON", code="invalid_upstream_sse")
                usage.update(_usage_counts(chunk.get("usage")))
                choices = chunk.get("choices")
                if not isinstance(choices, list) or not choices:
                    continue
                choice = choices[0]
                if not isinstance(choice, dict):
                    continue
                delta = choice.get("delta")
                if not isinstance(delta, dict):
                    delta = {}
                event: dict[str, Any] = {
                    "model": settings.canonical_model,
                    "created_at": _created_at(),
                    "done": False,
                }
                content = delta.get("content")
                reasoning = delta.get("reasoning_content")
                if generate:
                    event["response"] = content if isinstance(content, str) else ""
                    if expose_thinking and isinstance(reasoning, str) and reasoning:
                        event["thinking"] = reasoning
                else:
                    message: dict[str, Any] = {
                        "role": delta.get("role", "assistant"),
                        "content": content if isinstance(content, str) else "",
                    }
                    if expose_thinking and isinstance(reasoning, str) and reasoning:
                        message["thinking"] = reasoning
                    if "tool_calls" in delta:
                        message["tool_calls"] = _ollama_tool_calls(
                            delta["tool_calls"], complete=False
                        )
                    event["message"] = message
                choice_finish = choice.get("finish_reason")
                if isinstance(choice_finish, str):
                    finish_reason = choice_finish
                has_delta = any(
                    key in delta for key in ("content", "reasoning_content", "tool_calls")
                )
                if has_delta:
                    yield json.dumps(event, separators=(",", ":")).encode() + b"\n"
            event = {
                "model": settings.canonical_model,
                "created_at": _created_at(),
                "done": True,
                "total_duration": time.monotonic_ns() - started_ns,
                **usage,
            }
            if finish_reason is not None:
                event["done_reason"] = finish_reason
            yield json.dumps(event, separators=(",", ":")).encode() + b"\n"
        except (httpx.HTTPError, GatewayError) as exc:
            error = (
                exc
                if isinstance(exc, GatewayError)
                else GatewayError(502, "model upstream stream failed", code="upstream_stream_failed")
            )
            yield json.dumps({"error": error.message, "done": True}, separators=(",", ":")).encode() + b"\n"
    finally:
        try:
            await response.aclose()
        finally:
            await lease.release()


def _fallback_docs() -> str:
    return (
        "<!doctype html><html><head><meta charset='utf-8'><title>Local General</title></head>"
        "<body><h1>Local General</h1><p>The local model gateway is running.</p></body></html>"
    )


def create_app(
    settings: Settings,
    *,
    upstream_client: httpx.AsyncClient | None = None,
    health_client: httpx.AsyncClient | None = None,
) -> FastAPI:
    supplied_client = upstream_client
    supplied_health_client = health_client

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if supplied_client is not None:
            client = supplied_client
            owns_client = False
        else:
            timeout = httpx.Timeout(
                connect=settings.upstream_connect_timeout_seconds,
                read=settings.upstream_read_timeout_seconds,
                write=30.0,
                pool=settings.upstream_connect_timeout_seconds,
            )
            client = httpx.AsyncClient(
                timeout=timeout,
                limits=httpx.Limits(
                    max_connections=settings.max_active,
                    max_keepalive_connections=settings.max_active,
                ),
                follow_redirects=False,
            )
            owns_client = True
        if supplied_health_client is not None:
            probe_client = supplied_health_client
            owns_probe_client = False
        elif supplied_client is not None:
            # Test/embedding callers may intentionally share an injected
            # transport. Production always receives the independent client.
            probe_client = client
            owns_probe_client = False
        else:
            probe_timeout = min(settings.upstream_connect_timeout_seconds, 5.0)
            probe_client = httpx.AsyncClient(
                timeout=probe_timeout,
                limits=httpx.Limits(max_connections=1, max_keepalive_connections=1),
                follow_redirects=False,
            )
            owns_probe_client = True
        app.state.gateway = GatewayService(settings, client, probe_client)
        try:
            yield
        finally:
            if owns_client:
                await client.aclose()
            if owns_probe_client:
                await probe_client.aclose()

    app = FastAPI(
        title="Local General Gateway",
        version="1.0.0",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
    )
    app.add_middleware(RequestIdMiddleware)
    if supplied_client is not None:
        app.state.gateway = GatewayService(
            settings, supplied_client, supplied_health_client or supplied_client
        )

    def service(request: Request) -> GatewayService:
        return request.app.state.gateway

    @app.exception_handler(GatewayError)
    async def gateway_error_handler(request: Request, error: GatewayError) -> JSONResponse:
        gateway = service(request)
        headers = {**gateway.response_headers, **error.headers}
        if request.url.path.startswith("/api/"):
            return JSONResponse({"error": error.message}, status_code=error.status_code, headers=headers)
        return JSONResponse(
            _openai_error_payload(error), status_code=error.status_code, headers=headers
        )

    @app.get("/", include_in_schema=False)
    @app.get("/docs", include_in_schema=False)
    async def docs() -> HTMLResponse:
        content = (
            settings.docs_html.read_text(encoding="utf-8")
            if settings.docs_html is not None
            else _fallback_docs()
        )
        return HTMLResponse(
            content,
            headers={
                "Cache-Control": "no-store",
                "Content-Security-Policy": "default-src 'self'; style-src 'self' 'unsafe-inline'",
                "X-Content-Type-Options": "nosniff",
            },
        )

    @app.get("/healthz", include_in_schema=False)
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz", include_in_schema=False)
    async def readyz(request: Request) -> JSONResponse:
        gateway = service(request)
        request_id = request.state.request_id
        probe_timeout = min(settings.upstream_connect_timeout_seconds, 5.0)
        try:
            response = await gateway.health_client.get(
                gateway.health_url(),
                headers=gateway.upstream_headers(request_id),
                timeout=probe_timeout,
            )
            if not response.is_success:
                response = await gateway.health_client.get(
                    gateway.upstream_url("models"),
                    headers=gateway.upstream_headers(request_id),
                    timeout=probe_timeout,
                )
            ready = response.is_success
        except httpx.HTTPError:
            ready = False
        return JSONResponse(
            {"status": "ready" if ready else "not_ready"},
            status_code=200 if ready else 503,
        )

    async def authorized_body(request: Request) -> tuple[GatewayService, dict[str, Any]]:
        gateway = service(request)
        await gateway.authorize(request)
        body = await _read_json_object(request, settings.max_request_bytes)
        return gateway, body

    @app.get("/v1/models")
    async def models(request: Request) -> JSONResponse:
        gateway = service(request)
        await gateway.authorize(request)
        aliases = [settings.canonical_model]
        if settings.enable_legacy_gpt_oss_alias:
            aliases.append(LEGACY_ALIAS)
        return JSONResponse(
            {
                "object": "list",
                "data": [
                    {"id": alias, "object": "model", "created": 0, "owned_by": "local"}
                    for alias in aliases
                ],
            },
            headers=gateway.response_headers,
        )

    @app.post("/v1/chat/completions")
    async def openai_chat(request: Request) -> Response:
        gateway, body = await authorized_body(request)
        payload = gateway.prepare_openai_chat(body)
        if payload.get("stream") is True:
            try:
                upstream, lease = await _await_with_disconnect(
                    request,
                    gateway.open_stream(payload, request.state.request_id),
                    discard_result=_discard_open_stream,
                )
            except httpx.TimeoutException as exc:
                raise GatewayError(504, "model upstream timed out", code="upstream_timeout") from exc
            except httpx.HTTPError as exc:
                raise GatewayError(502, "model upstream is unavailable", code="upstream_unavailable") from exc
            return StreamingResponse(
                _openai_stream(upstream, lease, settings),
                media_type="text/event-stream",
                headers={
                    **gateway.response_headers,
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                },
            )
        result = await _await_with_disconnect(
            request, gateway.request_json(payload, request.state.request_id)
        )
        return JSONResponse(
            _canonicalize_openai(result, settings.canonical_model),
            headers=gateway.response_headers,
        )

    async def unsupported_openai(request: Request) -> None:
        gateway = service(request)
        await gateway.authorize(request)
        raise GatewayError(
            404,
            "this endpoint is not supported by local-general",
            error_type="not_found_error",
            code="unsupported_endpoint",
        )

    app.add_api_route("/v1/responses", unsupported_openai, methods=["POST"])
    app.add_api_route("/v1/embeddings", unsupported_openai, methods=["POST"])

    @app.get("/api/version")
    async def ollama_version(request: Request) -> JSONResponse:
        gateway = service(request)
        await gateway.authorize(request)
        return JSONResponse({"version": "0.0.0-local-general"}, headers=gateway.response_headers)

    @app.get("/api/tags")
    async def ollama_tags(request: Request) -> JSONResponse:
        gateway = service(request)
        await gateway.authorize(request)
        names = [settings.canonical_model]
        if settings.enable_legacy_gpt_oss_alias:
            names.append(LEGACY_ALIAS)
        return JSONResponse(
            {"models": [{"name": name, "model": name} for name in names]},
            headers=gateway.response_headers,
        )

    @app.get("/api/ps")
    async def ollama_ps(request: Request) -> JSONResponse:
        gateway = service(request)
        await gateway.authorize(request)
        return JSONResponse(
            {"models": [{"name": settings.canonical_model, "model": settings.canonical_model}]},
            headers=gateway.response_headers,
        )

    @app.post("/api/show")
    async def ollama_show(request: Request) -> JSONResponse:
        gateway, body = await authorized_body(request)
        unsupported = sorted(set(body) - {"model", "verbose"})
        if unsupported:
            raise GatewayError(
                400,
                f"unsupported Ollama show field(s): {', '.join(unsupported)}",
                code="unsupported_field",
            )
        if "verbose" in body and not isinstance(body["verbose"], bool):
            raise GatewayError(
                400,
                "verbose must be a boolean",
                code="invalid_verbose",
            )
        gateway.validate_model(body.get("model"))
        return JSONResponse(
            {
                "details": {
                    "format": "gguf",
                    "family": "lfm2.5",
                    "families": ["lfm2.5"],
                    "parameter_size": "24B-A2B",
                    "quantization_level": "Q4_0",
                },
                "model_info": {
                    "general.architecture": "lfm2.5",
                    "local_general.context_length": settings.context_window,
                    "local_general.max_output_tokens": settings.max_output_tokens,
                },
            },
            headers=gateway.response_headers,
        )

    async def ollama_completion(request: Request, *, generate: bool) -> Response:
        gateway, body = await authorized_body(request)
        payload, expose_thinking = gateway.prepare_ollama(body, generate=generate)
        started_ns = time.monotonic_ns()
        if payload["stream"]:
            try:
                upstream, lease = await _await_with_disconnect(
                    request,
                    gateway.open_stream(payload, request.state.request_id),
                    discard_result=_discard_open_stream,
                )
            except httpx.TimeoutException as exc:
                raise GatewayError(504, "model upstream timed out", code="upstream_timeout") from exc
            except httpx.HTTPError as exc:
                raise GatewayError(502, "model upstream is unavailable", code="upstream_unavailable") from exc
            return StreamingResponse(
                _ollama_stream(
                    upstream,
                    lease,
                    settings,
                    generate=generate,
                    expose_thinking=expose_thinking,
                    started_ns=started_ns,
                ),
                media_type="application/x-ndjson",
                headers={
                    **gateway.response_headers,
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                },
            )
        result = await _await_with_disconnect(
            request, gateway.request_json(payload, request.state.request_id)
        )
        transformed = _ollama_nonstream(
            result,
            generate=generate,
            expose_thinking=expose_thinking,
            elapsed_ns=time.monotonic_ns() - started_ns,
            model=settings.canonical_model,
        )
        return JSONResponse(transformed, headers=gateway.response_headers)

    @app.post("/api/chat")
    async def ollama_chat(request: Request) -> Response:
        return await ollama_completion(request, generate=False)

    @app.post("/api/generate")
    async def ollama_generate(request: Request) -> Response:
        return await ollama_completion(request, generate=True)

    async def unsupported_ollama(request: Request) -> None:
        gateway = service(request)
        await gateway.authorize(request)
        raise GatewayError(404, "embeddings are not supported", code="unsupported_endpoint")

    app.add_api_route("/api/embed", unsupported_ollama, methods=["POST"])
    app.add_api_route("/api/embeddings", unsupported_ollama, methods=["POST"])

    return app
