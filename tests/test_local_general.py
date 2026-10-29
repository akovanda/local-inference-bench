from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import httpx

from local_general import Settings, create_app
from local_general.app import _iter_sse_data


AUTH = {"Authorization": "Bearer gateway-secret"}


class ChunkStream(httpx.AsyncByteStream):
    def __init__(self, chunks: list[bytes], *, closed: asyncio.Event | None = None) -> None:
        self.chunks = chunks
        self.closed = closed

    async def __aiter__(self):
        for chunk in self.chunks:
            await asyncio.sleep(0)
            yield chunk

    async def aclose(self) -> None:
        if self.closed is not None:
            self.closed.set()


class BlockingStream(httpx.AsyncByteStream):
    def __init__(self, started: asyncio.Event, closed: asyncio.Event) -> None:
        self.started = started
        self.closed = closed
        self.block = asyncio.Event()

    async def __aiter__(self):
        self.started.set()
        yield b'data: {"choices":[{"delta":{"content":"first"},"finish_reason":null}]}\n\n'
        await self.block.wait()

    async def aclose(self) -> None:
        self.closed.set()


def settings(**overrides) -> Settings:
    values = {
        "api_token": "gateway-secret",
        "upstream_api_token": "upstream-secret",
        "queue_timeout_seconds": 1.0,
    }
    values.update(overrides)
    return Settings(**values)


def gateway_client(
    handler,
    *,
    gateway_settings: Settings | None = None,
) -> tuple[httpx.AsyncClient, httpx.AsyncClient, object]:
    upstream = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    app = create_app(gateway_settings or settings(), upstream_client=upstream)
    client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://gateway.test"
    )
    return client, upstream, app


def completion_scope(request_body: bytes) -> dict:
    return {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/v1/chat/completions",
        "raw_path": b"/v1/chat/completions",
        "query_string": b"",
        "root_path": "",
        "headers": [
            (b"host", b"gateway.test"),
            (b"authorization", b"Bearer gateway-secret"),
            (b"content-type", b"application/json"),
            (b"content-length", str(len(request_body)).encode()),
        ],
        "client": ("127.0.0.1", 12345),
        "server": ("127.0.0.1", 11441),
    }


class LocalGeneralGatewayTests(unittest.IsolatedAsyncioTestCase):
    async def test_sse_parser_yields_first_event_before_upstream_closes(self) -> None:
        started = asyncio.Event()
        closed = asyncio.Event()
        response = httpx.Response(200, stream=BlockingStream(started, closed))
        events = _iter_sse_data(response, 1024 * 1024)
        first = await asyncio.wait_for(anext(events), timeout=0.2)
        self.assertIn(b'"content":"first"', first)
        self.assertTrue(started.is_set())
        self.assertFalse(closed.is_set())
        await events.aclose()
        await response.aclose()
        self.assertTrue(closed.is_set())

    async def test_openai_nonstream_preserves_tools_json_and_reasoning(self) -> None:
        seen: list[dict] = []
        upstream_request_ids: list[str] = []

        async def handler(request: httpx.Request) -> httpx.Response:
            self.assertEqual(request.url.path, "/v1/chat/completions")
            self.assertEqual(request.headers["authorization"], "Bearer upstream-secret")
            body = json.loads(request.content)
            seen.append(body)
            upstream_request_ids.append(request.headers["x-request-id"])
            return httpx.Response(
                200,
                json={
                    "id": "chatcmpl-1",
                    "object": "chat.completion",
                    "model": "private-model-name",
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "stop",
                            "message": {
                                "role": "assistant",
                                "reasoning_content": "private chain",
                                "content": "final answer",
                            },
                        }
                    ],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 2},
                },
            )

        client, upstream, _ = gateway_client(handler)
        try:
            unauthenticated = await client.post(
                "/v1/chat/completions",
                json={"model": "local-general", "messages": [{"role": "user", "content": "x"}]},
            )
            self.assertEqual(unauthenticated.status_code, 401)
            self.assertEqual(seen, [])

            messages = [
                {"role": "user", "content": "weather?"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {"name": "weather", "arguments": '{"city":"Oslo"}'},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call-1", "content": "cold"},
            ]
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "weather",
                        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
                    },
                }
            ]
            response_format = {"type": "json_object"}
            response = await client.post(
                "/v1/chat/completions",
                headers=AUTH,
                json={
                    "model": "local-general",
                    "messages": messages,
                    "tools": tools,
                    "tool_choice": {"type": "function", "function": {"name": "weather"}},
                    "response_format": response_format,
                    "temperature": 0.2,
                    "reasoning_effort": "high",
                },
            )
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["model"], "local-general")
            message = response.json()["choices"][0]["message"]
            self.assertEqual(message["content"], "final answer")
            self.assertEqual(message["reasoning_content"], "private chain")
            self.assertIn("metadata-only", response.headers["x-local-general-reasoning-policy"])
            self.assertEqual(response.headers["x-request-id"], upstream_request_ids[0])
            self.assertEqual(seen[0]["model"], "lfm2.5-24b-a2b-qad-q4_0")
            self.assertEqual(seen[0]["messages"], messages)
            self.assertEqual(seen[0]["tools"], tools)
            self.assertEqual(seen[0]["response_format"], response_format)
            self.assertNotIn("reasoning_effort", seen[0])
            self.assertEqual(seen[0]["max_tokens"], 8192)
        finally:
            await client.aclose()
            await upstream.aclose()

    async def test_models_docs_health_and_readiness_are_intentionally_scoped(self) -> None:
        calls: list[str] = []

        async def handler(request: httpx.Request) -> httpx.Response:
            calls.append(request.url.path)
            if request.url.path == "/health":
                return httpx.Response(404)
            if request.url.path == "/v1/models":
                return httpx.Response(200, json={"data": []})
            raise AssertionError(request.url)

        with tempfile.TemporaryDirectory() as directory:
            docs_path = Path(directory) / "gateway.html"
            docs_path.write_text("<html><body>operator supplied</body></html>", encoding="utf-8")
            client, upstream, _ = gateway_client(
                handler, gateway_settings=settings(docs_html=docs_path)
            )
            try:
                self.assertEqual((await client.get("/healthz")).status_code, 200)
                self.assertIn("operator supplied", (await client.get("/docs")).text)
                self.assertEqual((await client.get("/readyz")).json(), {"status": "ready"})
                self.assertEqual(calls, ["/health", "/v1/models"])
                self.assertEqual((await client.get("/v1/models")).status_code, 401)
                models = await client.get("/v1/models", headers=AUTH)
                self.assertEqual([item["id"] for item in models.json()["data"]], ["local-general"])
                tags = await client.get("/api/tags", headers=AUTH)
                self.assertEqual(tags.json(), {"models": [{"name": "local-general", "model": "local-general"}]})
            finally:
                await client.aclose()
                await upstream.aclose()

    async def test_ollama_nonstream_schema_options_tool_calls_and_think_display(self) -> None:
        requests: list[dict] = []

        async def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            requests.append(body)
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "finish_reason": "tool_calls",
                            "message": {
                                "role": "assistant",
                                "content": "",
                                "reasoning_content": "hidden because think=false",
                                "tool_calls": [
                                    {
                                        "id": "c1",
                                        "type": "function",
                                        "function": {"name": "lookup", "arguments": '{"id":7}'},
                                    }
                                ],
                            },
                        }
                    ],
                    "usage": {"prompt_tokens": 9, "completion_tokens": 4},
                },
            )

        client, upstream, _ = gateway_client(handler)
        schema = {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
        }
        messages = [
            {"role": "user", "content": "lookup"},
            {"role": "tool", "tool_call_id": "previous", "content": "result"},
        ]
        tools = [{"type": "function", "function": {"name": "lookup", "parameters": schema}}]
        try:
            response = await client.post(
                "/api/chat",
                headers=AUTH,
                json={
                    "model": "local-general",
                    "messages": messages,
                    "tools": tools,
                    "format": schema,
                    "options": {
                        "num_predict": 123,
                        "temperature": 0.4,
                        "top_p": 0.8,
                        "top_k": 20,
                        "min_p": 0.05,
                        "repeat_penalty": 1.1,
                        "seed": 5,
                        "num_ctx": 49152,
                    },
                    "think": False,
                    "stream": False,
                },
            )
            self.assertEqual(response.status_code, 200)
            result = response.json()
            self.assertEqual(result["model"], "local-general")
            self.assertNotIn("thinking", result["message"])
            self.assertEqual(
                result["message"]["tool_calls"][0]["function"]["arguments"], {"id": 7}
            )
            self.assertEqual(result["prompt_eval_count"], 9)
            self.assertEqual(result["eval_count"], 4)
            self.assertGreaterEqual(result["total_duration"], 0)
            sent = requests[0]
            self.assertEqual(sent["messages"], messages)
            self.assertEqual(sent["tools"], tools)
            self.assertEqual(sent["max_tokens"], 123)
            self.assertEqual(sent["top_k"], 20)
            self.assertEqual(
                sent["response_format"],
                {
                    "type": "json_schema",
                    "json_schema": {"name": "ollama_response", "strict": True, "schema": schema},
                },
            )
        finally:
            await client.aclose()
            await upstream.aclose()

    async def test_ollama_native_tool_roundtrip_and_keep_alive_metadata(self) -> None:
        captured: list[dict] = []

        async def handler(request: httpx.Request) -> httpx.Response:
            captured.append(json.loads(request.content))
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {"finish_reason": "stop", "message": {"role": "assistant", "content": "done"}}
                    ]
                },
            )

        client, upstream, _ = gateway_client(handler)
        try:
            response = await client.post(
                "/api/chat",
                headers=AUTH,
                json={
                    "model": "local-general",
                    "stream": False,
                    # ollama.Client serializes keep_alive=-1 as the JSON float -1.0.
                    "keep_alive": -1.0,
                    "messages": [
                        {"role": "user", "content": "use tools"},
                        {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "given-id",
                                    "function": {"name": "first", "arguments": {"value": 1}},
                                },
                                {
                                    "function": {"name": "second", "arguments": {"value": 2}},
                                },
                            ],
                        },
                        {"role": "tool", "tool_name": "second", "content": "second-result"},
                        {"role": "tool", "tool_call_id": "given-id", "content": "first-result"},
                    ],
                },
            )
            self.assertEqual(response.status_code, 200)
            self.assertIn(
                "server-managed", response.headers["x-local-general-keep-alive-policy"]
            )
            sent = captured[0]
            self.assertNotIn("keep_alive", sent)
            calls = sent["messages"][1]["tool_calls"]
            self.assertEqual(calls[0]["id"], "given-id")
            self.assertEqual(calls[0]["type"], "function")
            self.assertEqual(calls[0]["function"]["arguments"], '{"value":1}')
            self.assertEqual(calls[1]["id"], "ollama_call_1_1")
            self.assertEqual(calls[1]["function"]["arguments"], '{"value":2}')
            self.assertEqual(sent["messages"][2]["tool_call_id"], "ollama_call_1_1")
            self.assertNotIn("tool_name", sent["messages"][2])
            self.assertEqual(sent["messages"][3]["tool_call_id"], "given-id")
        finally:
            await client.aclose()
            await upstream.aclose()

    async def test_ollama_show_accepts_boolean_verbose_metadata(self) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            raise AssertionError("synthetic show metadata must not call upstream")

        client, upstream, _ = gateway_client(handler)
        try:
            verbose = await client.post(
                "/api/show",
                headers=AUTH,
                json={"model": "local-general", "verbose": True},
            )
            self.assertEqual(verbose.status_code, 200)
            self.assertEqual(verbose.json()["model_info"]["local_general.context_length"], 49152)
            invalid = await client.post(
                "/api/show",
                headers=AUTH,
                json={"model": "local-general", "verbose": "true"},
            )
            self.assertEqual(invalid.status_code, 400)
            self.assertEqual(invalid.json()["error"], "verbose must be a boolean")
        finally:
            await client.aclose()
            await upstream.aclose()

    async def test_ollama_sse_is_incrementally_mapped_to_ndjson(self) -> None:
        captured: list[dict] = []

        async def handler(request: httpx.Request) -> httpx.Response:
            captured.append(json.loads(request.content))
            chunks = [
                b'data: {"model":"private","choices":[{"delta":{"role":"assistant","reasoning_content":"checking"},"finish_reason":null}]}\n',
                b'\ndata: {"choices":[{"delta":{"content":"answer"},"finish_reason":null}]}\n\n',
                b'data: {"choices":[{"delta":{},"finish_reason":"length"}],"usage":{"prompt_tokens":6,"completion_tokens":3}}\n\n',
                b'data: [DONE]\n\n',
            ]
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                stream=ChunkStream(chunks),
            )

        client, upstream, _ = gateway_client(handler)
        try:
            response = await client.post(
                "/api/generate",
                headers=AUTH,
                json={"model": "local-general", "prompt": "question", "think": True},
            )
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.headers["content-type"].split(";")[0], "application/x-ndjson")
            records = [json.loads(line) for line in response.text.splitlines()]
            self.assertEqual(records[0]["thinking"], "checking")
            self.assertEqual(records[1]["response"], "answer")
            self.assertTrue(records[2]["done"])
            self.assertEqual(records[2]["done_reason"], "length")
            self.assertEqual(records[2]["prompt_eval_count"], 6)
            self.assertEqual(records[2]["eval_count"], 3)
            self.assertEqual(captured[0]["messages"], [{"role": "user", "content": "question"}])
            self.assertTrue(captured[0]["stream"])
        finally:
            await client.aclose()
            await upstream.aclose()

    async def test_openai_sse_is_preserved_and_model_is_canonicalized(self) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            self.assertTrue(json.loads(request.content)["stream"])
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream; charset=utf-8"},
                stream=ChunkStream(
                    [
                        b'data: {"model":"private-name","choices":[{"delta":{"reasoning_content":"why"},"finish_reason":null}]}\n\n',
                        b'data: {"model":"private-name","choices":[{"delta":{"content":"answer"},"finish_reason":"stop"}]}\n\n',
                        b'data: [DONE]\n\n',
                    ]
                ),
            )

        client, upstream, _ = gateway_client(handler)
        try:
            response = await client.post(
                "/v1/chat/completions",
                headers=AUTH,
                json={
                    "model": "local-general",
                    "messages": [{"role": "user", "content": "x"}],
                    "stream": True,
                },
            )
            self.assertEqual(response.status_code, 200)
            data_lines = [
                line.removeprefix("data: ")
                for line in response.text.splitlines()
                if line.startswith("data: ")
            ]
            self.assertEqual(data_lines[-1], "[DONE]")
            first = json.loads(data_lines[0])
            second = json.loads(data_lines[1])
            self.assertEqual(first["model"], "local-general")
            self.assertEqual(first["choices"][0]["delta"]["reasoning_content"], "why")
            self.assertEqual(second["choices"][0]["delta"]["content"], "answer")
        finally:
            await client.aclose()
            await upstream.aclose()

    async def test_queue_full_is_429_and_capacity_recovers(self) -> None:
        first_started = asyncio.Event()
        release_first = asyncio.Event()
        call_count = 0

        async def handler(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                first_started.set()
                await release_first.wait()
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {"finish_reason": "stop", "message": {"role": "assistant", "content": "ok"}}
                    ]
                },
            )

        configured = settings(max_active=1, max_pending=1)
        client, upstream, app = gateway_client(handler, gateway_settings=configured)
        request = {
            "model": "local-general",
            "messages": [{"role": "user", "content": "x"}],
        }
        try:
            first = asyncio.create_task(client.post("/v1/chat/completions", headers=AUTH, json=request))
            await asyncio.wait_for(first_started.wait(), timeout=1)
            second = asyncio.create_task(client.post("/v1/chat/completions", headers=AUTH, json=request))
            for _ in range(100):
                if app.state.gateway.capacity.pending == 1:
                    break
                await asyncio.sleep(0.005)
            self.assertEqual(app.state.gateway.capacity.pending, 1)
            overloaded = await client.post("/v1/chat/completions", headers=AUTH, json=request)
            self.assertEqual(overloaded.status_code, 429)
            self.assertEqual(overloaded.headers["retry-after"], "1")
            self.assertEqual(overloaded.json()["error"]["code"], "queue_full")
            release_first.set()
            self.assertEqual((await first).status_code, 200)
            self.assertEqual((await second).status_code, 200)
            self.assertEqual(app.state.gateway.capacity.active, 0)
            self.assertEqual(app.state.gateway.capacity.pending, 0)
        finally:
            release_first.set()
            await client.aclose()
            await upstream.aclose()

    async def test_asgi_disconnect_cancels_nonstream_upstream_and_releases_capacity(self) -> None:
        upstream_started = asyncio.Event()
        upstream_cancelled = asyncio.Event()
        never = asyncio.Event()

        async def handler(request: httpx.Request) -> httpx.Response:
            upstream_started.set()
            try:
                await never.wait()
            except asyncio.CancelledError:
                upstream_cancelled.set()
                raise
            raise AssertionError("unreachable")

        upstream = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        app = create_app(settings(), upstream_client=upstream)
        request_body = json.dumps(
            {
                "model": "local-general",
                "messages": [{"role": "user", "content": "wait"}],
            }
        ).encode()
        incoming: asyncio.Queue[dict] = asyncio.Queue()
        await incoming.put({"type": "http.request", "body": request_body, "more_body": False})
        sent: list[dict] = []

        async def receive() -> dict:
            return await incoming.get()

        async def send(message: dict) -> None:
            sent.append(message)

        request_task = asyncio.create_task(app(completion_scope(request_body), receive, send))
        try:
            await asyncio.wait_for(upstream_started.wait(), timeout=1)
            self.assertEqual(app.state.gateway.capacity.active, 1)
            await incoming.put({"type": "http.disconnect"})
            with self.assertRaises(asyncio.CancelledError):
                await asyncio.wait_for(request_task, timeout=1)
            await asyncio.wait_for(upstream_cancelled.wait(), timeout=1)
            self.assertEqual(app.state.gateway.capacity.active, 0)
            self.assertEqual(app.state.gateway.capacity.pending, 0)
            self.assertEqual(sent, [])
        finally:
            if not request_task.done():
                request_task.cancel()
            await upstream.aclose()

    async def test_asgi_disconnect_removes_queued_request_without_taking_capacity(self) -> None:
        first_started = asyncio.Event()
        release_first = asyncio.Event()
        calls = 0

        async def handler(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            if calls == 1:
                first_started.set()
                await release_first.wait()
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {"finish_reason": "stop", "message": {"role": "assistant", "content": "ok"}}
                    ]
                },
            )

        client, upstream, app = gateway_client(
            handler, gateway_settings=settings(max_active=1, max_pending=1)
        )
        body_object = {
            "model": "local-general",
            "messages": [{"role": "user", "content": "queued"}],
        }
        first = asyncio.create_task(
            client.post("/v1/chat/completions", headers=AUTH, json=body_object)
        )
        queued: asyncio.Task | None = None
        try:
            await asyncio.wait_for(first_started.wait(), timeout=1)
            request_body = json.dumps(body_object).encode()
            incoming: asyncio.Queue[dict] = asyncio.Queue()
            await incoming.put(
                {"type": "http.request", "body": request_body, "more_body": False}
            )

            async def receive() -> dict:
                return await incoming.get()

            async def send(message: dict) -> None:
                raise AssertionError(
                    f"disconnected queued request sent a response: {message}"
                )

            queued = asyncio.create_task(
                app(completion_scope(request_body), receive, send)
            )
            for _ in range(100):
                if app.state.gateway.capacity.pending == 1:
                    break
                await asyncio.sleep(0.005)
            self.assertEqual(app.state.gateway.capacity.active, 1)
            self.assertEqual(app.state.gateway.capacity.pending, 1)
            await incoming.put({"type": "http.disconnect"})
            with self.assertRaises(asyncio.CancelledError):
                await asyncio.wait_for(queued, timeout=1)
            self.assertEqual(app.state.gateway.capacity.active, 1)
            self.assertEqual(app.state.gateway.capacity.pending, 0)
            self.assertEqual(calls, 1)
            release_first.set()
            self.assertEqual((await first).status_code, 200)
        finally:
            release_first.set()
            if not first.done():
                first.cancel()
            if queued is not None and not queued.done():
                queued.cancel()
            await client.aclose()
            await upstream.aclose()

    async def test_readiness_uses_reserved_client_during_active_generation(self) -> None:
        model_started = asyncio.Event()
        release_model = asyncio.Event()
        health_calls: list[str] = []

        async def model_handler(request: httpx.Request) -> httpx.Response:
            model_started.set()
            await release_model.wait()
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {"finish_reason": "stop", "message": {"role": "assistant", "content": "ok"}}
                    ]
                },
            )

        async def health_handler(request: httpx.Request) -> httpx.Response:
            health_calls.append(request.url.path)
            return httpx.Response(200, json={"status": "ok"})

        model_client = httpx.AsyncClient(transport=httpx.MockTransport(model_handler))
        health_client = httpx.AsyncClient(transport=httpx.MockTransport(health_handler))
        app = create_app(
            settings(max_active=1),
            upstream_client=model_client,
            health_client=health_client,
        )
        client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://gateway.test"
        )
        request = {
            "model": "local-general",
            "messages": [{"role": "user", "content": "busy"}],
        }
        generation = asyncio.create_task(
            client.post("/v1/chat/completions", headers=AUTH, json=request)
        )
        try:
            await asyncio.wait_for(model_started.wait(), timeout=1)
            self.assertEqual(app.state.gateway.capacity.active, 1)
            ready = await asyncio.wait_for(client.get("/readyz"), timeout=1)
            self.assertEqual(ready.status_code, 200)
            self.assertEqual(health_calls, ["/health"])
            self.assertIs(app.state.gateway.health_client, health_client)
            release_model.set()
            self.assertEqual((await generation).status_code, 200)
        finally:
            release_model.set()
            if not generation.done():
                generation.cancel()
            await client.aclose()
            await model_client.aclose()
            await health_client.aclose()

    async def test_cancelled_stream_closes_upstream_and_releases_capacity(self) -> None:
        stream_started = asyncio.Event()
        stream_closed = asyncio.Event()
        calls = 0

        async def handler(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            if calls == 1:
                return httpx.Response(
                    200,
                    headers={"content-type": "text/event-stream"},
                    stream=BlockingStream(stream_started, stream_closed),
                )
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {"finish_reason": "stop", "message": {"role": "assistant", "content": "ok"}}
                    ]
                },
            )

        client, upstream, app = gateway_client(handler)
        stream_request = {
            "model": "local-general",
            "messages": [{"role": "user", "content": "x"}],
            "stream": True,
        }
        try:
            task = asyncio.create_task(
                client.post("/v1/chat/completions", headers=AUTH, json=stream_request)
            )
            await asyncio.wait_for(stream_started.wait(), timeout=1)
            self.assertEqual(app.state.gateway.capacity.active, 1)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            await asyncio.wait_for(stream_closed.wait(), timeout=1)
            self.assertEqual(app.state.gateway.capacity.active, 0)
            followup = await client.post(
                "/v1/chat/completions",
                headers=AUTH,
                json={
                    "model": "local-general",
                    "messages": [{"role": "user", "content": "again"}],
                },
            )
            self.assertEqual(followup.status_code, 200)
        finally:
            await client.aclose()
            await upstream.aclose()

    async def test_backend_unavailable_and_malformed_requests_have_safe_errors(self) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("backend address must not leak", request=request)

        client, upstream, _ = gateway_client(
            handler, gateway_settings=settings(default_max_tokens=100, max_output_tokens=100)
        )
        try:
            malformed = await client.post(
                "/v1/chat/completions",
                headers={**AUTH, "Content-Type": "application/json"},
                content=b"{broken",
            )
            self.assertEqual(malformed.status_code, 400)
            self.assertEqual(malformed.json()["error"]["code"], "invalid_json")
            too_many = await client.post(
                "/v1/chat/completions",
                headers=AUTH,
                json={
                    "model": "local-general",
                    "messages": [{"role": "user", "content": "x"}],
                    "max_tokens": 101,
                },
            )
            self.assertEqual(too_many.status_code, 400)
            multiple = await client.post(
                "/v1/chat/completions",
                headers=AUTH,
                json={
                    "model": "local-general",
                    "messages": [{"role": "user", "content": "x"}],
                    "n": 2,
                },
            )
            self.assertEqual(multiple.status_code, 400)
            self.assertEqual(
                multiple.json()["error"]["code"], "unsupported_completion_count"
            )
            bad_option = await client.post(
                "/api/generate",
                headers=AUTH,
                json={
                    "model": "local-general",
                    "prompt": "x",
                    "stream": False,
                    "options": {"mirostat": 2},
                },
            )
            self.assertEqual(bad_option.status_code, 400)
            self.assertIn("mirostat", bad_option.json()["error"])
            unavailable = await client.post(
                "/v1/chat/completions",
                headers=AUTH,
                json={"model": "local-general", "messages": [{"role": "user", "content": "x"}]},
            )
            self.assertEqual(unavailable.status_code, 502)
            self.assertEqual(unavailable.json()["error"]["code"], "upstream_unavailable")
            unsupported = await client.post("/v1/responses", headers=AUTH, json={})
            self.assertEqual(unsupported.status_code, 404)
        finally:
            await client.aclose()
            await upstream.aclose()


class LocalGeneralSettingsTests(unittest.TestCase):
    def test_real_ollama_sdk_serializes_integer_keep_alive_as_finite_float(self) -> None:
        try:
            import ollama
        except ImportError:
            self.skipTest("ollama SDK is not installed in this test environment")

        captured: list[dict] = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured.append(json.loads(request.content))
            return httpx.Response(
                200,
                json={
                    "model": "local-general",
                    "created_at": "2026-01-01T00:00:00Z",
                    "message": {"role": "assistant", "content": "ok"},
                    "done": True,
                },
            )

        client = ollama.Client(
            host="http://gateway.test",
            transport=httpx.MockTransport(handler),
        )
        try:
            result = client.chat(
                model="local-general",
                messages=[{"role": "user", "content": "hello"}],
                keep_alive=-1,
            )
        finally:
            client.close()
        self.assertEqual(result.message.content, "ok")
        self.assertEqual(captured[0]["keep_alive"], -1.0)
        self.assertIsInstance(captured[0]["keep_alive"], float)

    def test_environment_fails_closed_and_supports_token_file(self) -> None:
        names = {
            "LOCAL_GENERAL_API_TOKEN": "",
            "LOCAL_GENERAL_API_TOKEN_FILE": "",
            "LOCAL_GENERAL_UPSTREAM_API_TOKEN": "",
            "LOCAL_GENERAL_UPSTREAM_API_TOKEN_FILE": "",
        }
        with patch.dict(os.environ, names, clear=False):
            with self.assertRaisesRegex(ValueError, "LOCAL_GENERAL_API_TOKEN"):
                Settings.from_env()
        with tempfile.TemporaryDirectory() as directory:
            token_path = Path(directory) / "token"
            token_path.write_text("file-secret\n", encoding="utf-8")
            with patch.dict(
                os.environ,
                {**names, "LOCAL_GENERAL_API_TOKEN_FILE": str(token_path)},
                clear=False,
            ):
                loaded = Settings.from_env()
            self.assertEqual(loaded.api_token, "file-secret")
            self.assertEqual(loaded.bind_host, "127.0.0.1")
            self.assertEqual(loaded.bind_port, 11441)

    def test_legacy_alias_is_opt_in(self) -> None:
        disabled = settings()
        enabled = settings(enable_legacy_gpt_oss_alias=True)
        self.assertFalse(disabled.enable_legacy_gpt_oss_alias)
        self.assertTrue(enabled.enable_legacy_gpt_oss_alias)


if __name__ == "__main__":
    unittest.main()
