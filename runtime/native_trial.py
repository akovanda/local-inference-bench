#!/usr/bin/env python3
"""Run isolated native Ollama benchmark trials from an extracted runtime descriptor."""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import re
import signal
import socket
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.error import URLError
from urllib.request import Request, urlopen

if __package__:
    from .signal_exec import unblocked_exec_command
else:
    from signal_exec import unblocked_exec_command


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
ENDPOINT = "http://127.0.0.1:11440"
HOST = "127.0.0.1"
PORT = 11440
MODEL = "gpt-oss-20b:64k"
STARTUP_TIMEOUT = 90.0
REQUEST_TIMEOUT = 3.0
TERM_TIMEOUT = 20.0
KILL_TIMEOUT = 10.0
GPU_PROCESS_LIMIT_MIB = 256
QUALITY_CASES = "quality_merge_intervals,quality_stable_toposort,quality_constraints"
TAG_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")
CASE_ID_RE = re.compile(r"^[A-Za-z0-9_]{1,128}$")
MAX_HARD_CASES = 64
DEFAULT_HARD_OUTPUT_TOKENS = 2048
DEFAULT_HARD_REQUEST_TIMEOUT = 240.0


class TrialError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


class EventLog:
    def __init__(self, path: Path, run_id: str, tag: str) -> None:
        self.path = path
        self.run_id = run_id
        self.tag = tag
        path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, event: str, **fields: Any) -> None:
        row = {
            "at": utc_now(),
            "event": event,
            "run_id": self.run_id,
            "tag": self.tag,
            **fields,
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        print(json.dumps(row, separators=(",", ":")), flush=True)


def descriptor_path(runtime: Path) -> Path:
    return runtime / "runtime.json" if runtime.is_dir() else runtime


def load_descriptor(runtime: Path) -> dict[str, Any]:
    path = descriptor_path(runtime)
    try:
        descriptor = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TrialError("runtime descriptor is unreadable or invalid") from exc
    if not isinstance(descriptor, dict) or descriptor.get("schema_version") != 1:
        raise TrialError("runtime descriptor schema_version must be 1")
    launch = descriptor.get("launch")
    if not isinstance(launch, dict):
        raise TrialError("runtime descriptor is missing launch configuration")
    argv = launch.get("argv")
    env = launch.get("env")
    if (
        not isinstance(argv, list)
        or not argv
        or any(not isinstance(value, str) or not value or "\x00" in value for value in argv)
    ):
        raise TrialError("runtime launch.argv must be a non-empty string list")
    if not isinstance(env, dict) or any(
        not isinstance(key, str)
        or not isinstance(value, str)
        or "\x00" in key
        or "\x00" in value
        for key, value in env.items()
    ):
        raise TrialError("runtime launch.env must contain only strings")
    if not env.get("OLLAMA_MODELS"):
        raise TrialError("runtime descriptor must pin OLLAMA_MODELS")
    binary = Path(argv[0])
    if not binary.is_absolute() or not binary.is_file() or not os.access(binary, os.X_OK):
        raise TrialError("runtime launch binary must be an executable absolute path")
    return descriptor


def build_server_env(
    descriptor: Mapping[str, Any],
    cache: str,
    context: int,
    base_env: Mapping[str, str] | None = None,
) -> dict[str, str]:
    source = dict(os.environ if base_env is None else base_env)
    launch_env = descriptor["launch"]["env"]
    model_store = launch_env["OLLAMA_MODELS"]
    preserved = {key: source.get(key) for key in ("HOME", "CODEX_HOME")}
    source.update(launch_env)
    for key, value in preserved.items():
        if value is None:
            source.pop(key, None)
        else:
            source[key] = value
    source.update(
        {
            "OLLAMA_FLASH_ATTENTION": "1",
            "OLLAMA_KV_CACHE_TYPE": cache,
            "OLLAMA_NUM_PARALLEL": "1",
            "OLLAMA_CONTEXT_LENGTH": str(context),
            "OLLAMA_KEEP_ALIVE": "10m",
            "OLLAMA_NOPRUNE": "1",
            "OLLAMA_NO_CLOUD": "1",
            "OLLAMA_HOST": f"{HOST}:{PORT}",
        }
    )
    if source.get("OLLAMA_MODELS") != model_store:
        raise TrialError("runtime model store changed unexpectedly")
    return source


def port_is_occupied(host: str = HOST, port: int = PORT, timeout: float = 0.3) -> bool:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.settimeout(timeout)
        return sock.connect_ex((host, port)) == 0
    finally:
        sock.close()


def gpu_compute_processes() -> list[dict[str, Any]]:
    command = [
        "nvidia-smi",
        "--query-compute-apps=pid,process_name,used_gpu_memory",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise TrialError("unable to verify GPU compute-process occupancy") from exc
    if completed.returncode != 0:
        raise TrialError("nvidia-smi could not verify GPU compute-process occupancy")
    processes: list[dict[str, Any]] = []
    for line in completed.stdout.splitlines():
        if not line.strip():
            continue
        columns = [column.strip() for column in line.rsplit(",", 2)]
        if len(columns) != 3:
            raise TrialError("nvidia-smi returned an unexpected compute-process row")
        try:
            pid = int(columns[0])
            used_mib = int(columns[2])
        except ValueError as exc:
            raise TrialError("nvidia-smi returned non-numeric compute-process data") from exc
        processes.append({"pid": pid, "name": columns[1], "used_mib": used_mib})
    return processes


def assert_preflight_clear() -> list[dict[str, Any]]:
    if port_is_occupied():
        raise TrialError(f"loopback port {PORT} is occupied; refusing to alter its process")
    processes = gpu_compute_processes()
    busy = [process for process in processes if process["used_mib"] > GPU_PROCESS_LIMIT_MIB]
    if busy:
        summary = ", ".join(f"pid {item['pid']} ({item['used_mib']} MiB)" for item in busy)
        raise TrialError(f"GPU has another compute process over {GPU_PROCESS_LIMIT_MIB} MiB: {summary}")
    return processes


@dataclass
class ServerHandle:
    process: subprocess.Popen[Any]
    pgid: int
    log_handle: Any


def start_server(
    descriptor: Mapping[str, Any],
    env: Mapping[str, str],
    log_path: Path,
) -> ServerHandle:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = log_path.open("ab", buffering=0)
    log_handle.write((f"\n=== native trial server start {utc_now()} ===\n").encode("utf-8"))
    try:
        process = subprocess.Popen(
            unblocked_exec_command(descriptor["launch"]["argv"]),
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            env=dict(env),
            start_new_session=True,
            close_fds=True,
        )
    except BaseException:
        log_handle.close()
        raise
    return ServerHandle(process=process, pgid=process.pid, log_handle=log_handle)


def wait_for_server(handle: ServerHandle, expected_version: str | None) -> dict[str, Any]:
    deadline = time.monotonic() + STARTUP_TIMEOUT
    last_problem = "endpoint not ready"
    while time.monotonic() < deadline:
        returncode = handle.process.poll()
        if returncode is not None:
            raise TrialError(f"native server exited during startup with status {returncode}")
        try:
            request = Request(ENDPOINT + "/api/version", headers={"Accept": "application/json"})
            with urlopen(request, timeout=min(REQUEST_TIMEOUT, max(0.1, deadline - time.monotonic()))) as response:
                payload = json.loads(response.read(64 * 1024))
            if not isinstance(payload, dict) or not isinstance(payload.get("version"), str):
                last_problem = "version endpoint returned invalid JSON"
            elif expected_version and payload["version"] != expected_version:
                raise TrialError(
                    f"native server version {payload['version']!r} does not match descriptor {expected_version!r}"
                )
            else:
                return payload
        except TrialError:
            raise
        except (OSError, URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_problem = type(exc).__name__
        time.sleep(1)
    raise TrialError(f"native server did not become ready within {int(STARTUP_TIMEOUT)}s ({last_problem})")


def _group_exists(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False


def _wait_group_gone(pgid: int, process: subprocess.Popen[Any], timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        process.poll()
        if not _group_exists(pgid):
            return True
        time.sleep(0.1)
    return not _group_exists(pgid)


def stop_server(handle: ServerHandle) -> dict[str, Any]:
    result = {"pgid": handle.pgid, "term_sent": False, "kill_sent": False, "stopped": False}
    try:
        if _group_exists(handle.pgid):
            os.killpg(handle.pgid, signal.SIGTERM)
            result["term_sent"] = True
        if not _wait_group_gone(handle.pgid, handle.process, TERM_TIMEOUT):
            os.killpg(handle.pgid, signal.SIGKILL)
            result["kill_sent"] = True
            result["stopped"] = _wait_group_gone(handle.pgid, handle.process, KILL_TIMEOUT)
        else:
            result["stopped"] = True
        try:
            handle.process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            pass
        if not result["stopped"]:
            raise TrialError("created native server process group did not stop after SIGKILL")
        return result
    finally:
        handle.log_handle.close()


def context_stem(tag: str, cache: str, context: int) -> str:
    context_label = f"{context // 1024}k" if context % 1024 == 0 else str(context)
    return f"{tag}-{cache}-{context_label}"


def probe_command(context: int, output: Path, *, thinking: str = "low") -> list[str]:
    return [
        sys.executable,
        str(ROOT / "runtime" / "probe.py"),
        "--endpoint",
        ENDPOINT,
        "--context",
        str(context),
        "--timeout",
        "180",
        "--thinking",
        thinking,
        "--tokens",
        "128",
        "--output",
        str(output),
    ]


def bench_command(
    context: int,
    suite: str,
    output: Path,
    *,
    context_target: int | None = None,
    quick_repeat: int = 2,
    thinking: str = "low",
    hard_cases: str | None = None,
    output_tokens: int = DEFAULT_HARD_OUTPUT_TOKENS,
    request_timeout: float = DEFAULT_HARD_REQUEST_TIMEOUT,
) -> list[str]:
    command = [
        sys.executable,
        str(ROOT / "bench.py"),
        "--endpoint",
        ENDPOINT,
        "--backend",
        "ollama",
        "--model",
        MODEL,
        "--context",
        str(context),
        "--thinking",
        thinking,
        "--suite",
        suite,
        "--timeout",
        format_number(request_timeout) if suite == "hard" else "540" if suite == "context" else "300",
        "--fail-on-case",
        "--output",
        str(output),
    ]
    if suite == "quick":
        command.extend(["--repeat", str(quick_repeat), "--num-predict", "512"])
    elif suite == "quality":
        command.extend(["--cases", QUALITY_CASES])
    elif suite == "context":
        if context_target is None:
            raise TrialError("context suite requires a target")
        command.extend(["--context-target", str(context_target), "--num-predict", "768"])
    elif suite == "hard":
        if hard_cases is None:
            raise TrialError("hard suite requires case IDs")
        command.extend(["--cases", hard_cases, "--num-predict", str(output_tokens)])
    else:
        raise TrialError(f"unsupported benchmark suite {suite!r}")
    return command


def inspect_probe_result(path: Path) -> dict[str, Any]:
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TrialError("health probe did not produce valid JSON") from exc
    if not isinstance(result, dict) or not isinstance(result.get("passed"), bool):
        raise TrialError("health probe result did not contain a boolean passed field")
    return result


def run_probe(
    context: int,
    health_path: Path,
    run_id: str,
    *,
    thinking: str = "low",
) -> dict[str, Any]:
    temporary = health_path.with_name(health_path.name + f".tmp.{run_id}")
    completed = subprocess.run(
        probe_command(context, temporary, thinking=thinking),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=210,
        check=False,
    )
    if completed.returncode != 0:
        raise TrialError(f"health probe process exited with status {completed.returncode}")
    result = inspect_probe_result(temporary)
    os.replace(temporary, health_path)
    return result


def suite_process_timeout(
    suite: str,
    *,
    hard_case_count: int = 0,
    request_timeout: float = DEFAULT_HARD_REQUEST_TIMEOUT,
) -> float:
    if suite == "quick":
        return 4 * 300 + 90
    if suite == "quality":
        return 3 * 300 + 90
    if suite == "hard":
        if hard_case_count < 1:
            raise TrialError("hard suite requires at least one case")
        return hard_case_count * request_timeout + 90
    # Long prompts can spend several minutes in prefill before emitting a token.
    # The outer GPU window independently bounds service downtime.
    return 540 + 90


def run_suite(
    command: Sequence[str],
    suite: str,
    *,
    hard_case_count: int = 0,
    request_timeout: float = DEFAULT_HARD_REQUEST_TIMEOUT,
) -> int:
    try:
        completed = subprocess.run(
            list(command),
            stdin=subprocess.DEVNULL,
            timeout=suite_process_timeout(
                suite,
                hard_case_count=hard_case_count,
                request_timeout=request_timeout,
            ),
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise TrialError(f"{suite} benchmark process exceeded its total timeout") from exc
    return completed.returncode


def parse_contexts(value: str) -> list[int]:
    try:
        contexts = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("contexts must be comma-separated integers") from exc
    if not contexts or any(context < 2048 or context > 262144 for context in contexts):
        raise argparse.ArgumentTypeError("contexts must be between 2048 and 262144")
    if len(set(contexts)) != len(contexts):
        raise argparse.ArgumentTypeError("contexts must not contain duplicates")
    return contexts


def parse_case_ids(value: str) -> str:
    items = value.split(",")
    if (
        not items
        or len(items) > MAX_HARD_CASES
        or any(not item or CASE_ID_RE.fullmatch(item) is None for item in items)
    ):
        raise argparse.ArgumentTypeError(
            f"must be 1-{MAX_HARD_CASES} comma-separated alphanumeric or underscore case IDs"
        )
    if len(set(items)) != len(items):
        raise argparse.ArgumentTypeError("hard case IDs must not contain duplicates")
    return ",".join(items)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be a finite number greater than 0")
    return parsed


def format_number(value: float) -> str:
    return format(value, ".15g")


def install_signal_handlers() -> None:
    def interrupt(signum: int, _frame: Any) -> None:
        raise KeyboardInterrupt(f"signal {signum}")

    signal.signal(signal.SIGINT, interrupt)
    signal.signal(signal.SIGTERM, interrupt)


@contextlib.contextmanager
def defer_termination_signals():
    """Close launch/cleanup signal races so an owned process group cannot escape."""
    if not hasattr(signal, "pthread_sigmask"):
        yield
        return
    watched = {signal.SIGINT, signal.SIGTERM}
    previous = signal.pthread_sigmask(signal.SIG_BLOCK, watched)
    try:
        yield
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, previous)


def run_context(
    descriptor: Mapping[str, Any],
    tag: str,
    cache: str,
    context: int,
    quality: bool,
    context_target: int | None,
    run_id: str,
    events: EventLog,
    quick_repeat: int = 2,
    skip_quick: bool = False,
    thinking: str = "low",
    hard_cases: str | None = None,
    output_tokens: int = DEFAULT_HARD_OUTPUT_TOKENS,
    request_timeout: float = DEFAULT_HARD_REQUEST_TIMEOUT,
) -> bool:
    stem = context_stem(tag, cache, context)
    handle: ServerHandle | None = None
    success = True
    try:
        gpu_processes = assert_preflight_clear()
        events.write(
            "preflight_passed",
            cache=cache,
            context=context,
            observed_gpu_compute_processes=gpu_processes,
        )
        env = build_server_env(descriptor, cache, context)
        with defer_termination_signals():
            handle = start_server(descriptor, env, RESULTS / f"{stem}-server.log")
        events.write("server_started", cache=cache, context=context, pid=handle.process.pid, pgid=handle.pgid)
        version = wait_for_server(handle, descriptor.get("ollama_version"))
        events.write("server_ready", cache=cache, context=context, version=version)

        health_path = RESULTS / f"{stem}-health.json"
        health = run_probe(context, health_path, run_id, thinking=thinking)
        events.write(
            "health_complete",
            cache=cache,
            context=context,
            passed=health["passed"],
            result=str(health_path.relative_to(ROOT)),
        )
        if not health["passed"]:
            success = False
        else:
            suites: list[tuple[str, int | None]] = []
            if not skip_quick:
                suites.append(("quick", None))
            if quality:
                suites.append(("quality", None))
            if context_target is not None:
                suites.append(("context", context_target))
            if hard_cases is not None:
                suites.append(("hard", None))
            for suite, target in suites:
                output = RESULTS / f"{stem}-{suite}.jsonl"
                command = bench_command(
                    context,
                    suite,
                    output,
                    context_target=target,
                    quick_repeat=quick_repeat,
                    thinking=thinking,
                    hard_cases=hard_cases,
                    output_tokens=output_tokens,
                    request_timeout=request_timeout,
                )
                events.write(
                    "suite_started",
                    cache=cache,
                    context=context,
                    suite=suite,
                    result=str(output.relative_to(ROOT)),
                )
                returncode = run_suite(
                    command,
                    suite,
                    hard_case_count=len(hard_cases.split(",")) if suite == "hard" and hard_cases else 0,
                    request_timeout=request_timeout,
                )
                passed = returncode == 0
                events.write(
                    "suite_complete",
                    cache=cache,
                    context=context,
                    suite=suite,
                    returncode=returncode,
                    passed=passed,
                )
                success = success and passed
    except (TrialError, OSError, subprocess.SubprocessError) as exc:
        events.write(
            "context_failed",
            cache=cache,
            context=context,
            error_type=type(exc).__name__,
            error=str(exc),
        )
        success = False
    finally:
        if handle is not None:
            try:
                with defer_termination_signals():
                    cleanup = stop_server(handle)
                events.write("server_stopped", cache=cache, context=context, **cleanup)
            except (TrialError, OSError) as exc:
                events.write(
                    "server_cleanup_failed",
                    cache=cache,
                    context=context,
                    error_type=type(exc).__name__,
                    error=str(exc),
                )
                success = False
    return success


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", required=True, type=Path, help="runtime directory or runtime.json")
    parser.add_argument("--tag", required=True, help="safe result filename prefix")
    parser.add_argument("--cache", required=True, choices=("f16", "q8_0"))
    parser.add_argument("--contexts", required=True, type=parse_contexts, help="comma-separated token limits")
    parser.add_argument("--thinking", choices=("low", "medium"), default="low")
    parser.add_argument("--quality", action="store_true")
    parser.add_argument("--context-target", type=int)
    parser.add_argument("--quick-repeat", type=int, choices=(1, 2), default=2)
    parser.add_argument("--skip-quick", action="store_true")
    parser.add_argument("--hard-cases", type=parse_case_ids)
    parser.add_argument("--output-tokens", type=positive_int, default=DEFAULT_HARD_OUTPUT_TOKENS)
    parser.add_argument(
        "--request-timeout",
        type=positive_float,
        default=DEFAULT_HARD_REQUEST_TIMEOUT,
        help="per-request timeout in seconds for the hard suite",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if TAG_RE.fullmatch(args.tag) is None:
        parser.error("--tag must contain only letters, digits, dot, underscore, or hyphen")
    if args.context_target is not None and args.context_target < 512:
        parser.error("--context-target must be at least 512")
    if args.skip_quick and not (args.quality or args.context_target is not None or args.hard_cases):
        parser.error("--skip-quick requires --quality, --context-target, or --hard-cases")

    try:
        descriptor = load_descriptor(args.runtime)
    except TrialError as exc:
        parser.error(str(exc))
    RESULTS.mkdir(parents=True, exist_ok=True)
    run_id = str(uuid.uuid4())
    events = EventLog(RESULTS / f"{args.tag}-runtime-events.jsonl", run_id, args.tag)
    install_signal_handlers()
    events.write(
        "trial_started",
        runtime=str(descriptor_path(args.runtime)),
        cache=args.cache,
        contexts=args.contexts,
        thinking=args.thinking,
        quality=args.quality,
        context_target=args.context_target,
        quick_repeat=args.quick_repeat,
        skip_quick=args.skip_quick,
        hard_cases=args.hard_cases,
        output_tokens=args.output_tokens,
        request_timeout=args.request_timeout,
    )
    all_passed = True
    try:
        for context in args.contexts:
            passed = run_context(
                descriptor,
                args.tag,
                args.cache,
                context,
                args.quality,
                args.context_target,
                run_id,
                events,
                quick_repeat=args.quick_repeat,
                skip_quick=args.skip_quick,
                thinking=args.thinking,
                hard_cases=args.hard_cases,
                output_tokens=args.output_tokens,
                request_timeout=args.request_timeout,
            )
            all_passed = all_passed and passed
    except KeyboardInterrupt:
        events.write("trial_interrupted")
        return 130
    events.write("trial_complete", passed=all_passed)
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
