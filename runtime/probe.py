#!/usr/bin/env python3
"""Bounded, local-only health probe; never saves model reasoning text."""
from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path
import time
import urllib.error
import urllib.request
from typing import Sequence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://127.0.0.1:11440")
    parser.add_argument("--context", type=int, default=32768)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--thinking", default="low")
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.endpoint.startswith(("http://127.0.0.1:", "http://localhost:")):
        parser.error("this health probe is limited to loopback")
    payload = {
        "model": "gpt-oss-20b:64k",
        "messages": [{"role": "user", "content": "Reply with exactly LOCAL_TEST_OK and nothing else."}],
        "think": args.thinking,
        "stream": True,
        "options": {
            "num_ctx": args.context, "num_predict": args.tokens,
            "temperature": 0, "seed": 42,
        },
    }
    result = {
        "started_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "endpoint": args.endpoint,
        "context": args.context,
        "thinking": args.thinking,
        "num_predict": args.tokens,
        "response": "",
        "thinking_chars": 0,
    }
    start = time.monotonic()
    request = urllib.request.Request(
        args.endpoint + "/api/chat",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            for line in response:
                data = json.loads(line)
                if "error" in data:
                    raise RuntimeError(data["error"])
                message = data.get("message", {})
                thinking = message.get("thinking") or ""
                content = message.get("content") or ""
                if thinking or content:
                    result.setdefault("ttft_seconds", time.monotonic() - start)
                if content:
                    result.setdefault("first_content_seconds", time.monotonic() - start)
                result["thinking_chars"] += len(thinking)
                result["response"] += content
                if data.get("done"):
                    for key in (
                        "done_reason", "total_duration", "load_duration",
                        "prompt_eval_count", "prompt_eval_duration", "eval_count",
                        "eval_duration",
                    ):
                        result[key] = data.get(key)
        result["passed"] = (
            result["response"].strip() == "LOCAL_TEST_OK"
            and result.get("done_reason") != "length"
        )
    except urllib.error.HTTPError as exc:
        result["http_status"] = exc.code
        result["error"] = str(exc)
        try:
            error_body = json.loads(exc.read(4096))
            if isinstance(error_body, dict) and isinstance(error_body.get("error"), str):
                result["server_error"] = error_body["error"][:2000]
        except (ValueError, OSError):
            pass
        result["passed"] = False
    except Exception as exc:  # Probe failures are recorded in the requested result file.
        result["error"] = str(exc)
        result["passed"] = False
    result["wall_seconds"] = time.monotonic() - start
    if result.get("eval_duration"):
        result["decode_tokens_per_second"] = (
            result["eval_count"] / (result["eval_duration"] / 1e9)
        )
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
