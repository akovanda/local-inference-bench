#!/usr/bin/env python3
"""Run one isolated Strata benchmark trial on loopback port 18080."""

from __future__ import annotations

import argparse
import contextlib
import copy
import json
import math
import os
import re
import shutil
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
STRATA_ROOT = ROOT / "vendor" / "strata"
STRATA_PYTHON = STRATA_ROOT / ".venv" / "bin" / "python"
DEFAULT_CONFIG = ROOT / "runtime" / "qwen36-32k.json"
HOST = "127.0.0.1"
PORT = 18080
ENDPOINT = f"http://{HOST}:{PORT}/v1"
STARTUP_TIMEOUT = 180.0
MIN_STARTUP_TIMEOUT = 30.0
MAX_STARTUP_TIMEOUT = 480.0
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
DEFAULT_CONTEXT_OUTPUT_TOKENS = 768
DEFAULT_CONTEXT_TIMEOUT = 300.0
CPU_ITEM_RE = re.compile(r"^(\d+)(?:-(\d+)(?::(\d+))?)?$")


class TrialError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def engine_environment_metadata(environment: Mapping[str, str] | None = None) -> dict[str, str]:
    """Record only named engine controls, never unrelated environment values."""
    source = os.environ if environment is None else environment
    names = ("STRATA_SM75_PERF", "STRATA_ROUTE_TAIL_SKIP", "STRATA_ADAPT_OVERLAP")
    return {name: source[name] for name in names if name in source}


class EventLog:
    def __init__(self, path: Path, run_id: str, tag: str) -> None:
        self.path = path
        self.run_id = run_id
        self.tag = tag

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


def _config_cwd(config: Mapping[str, Any], server_root: Path) -> Path:
    raw = config.get("cwd")
    if raw is None:
        return server_root
    if not isinstance(raw, str) or not raw or "\x00" in raw:
        raise TrialError("config cwd must be a non-empty string")
    path = Path(raw)
    return path.resolve() if path.is_absolute() else (server_root / path).resolve()


def _resolve_from(path_value: str, base: Path, field: str) -> Path:
    if not path_value or "\x00" in path_value:
        raise TrialError(f"config {field} must be a non-empty path")
    path = Path(path_value)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _option_value(args: Sequence[str], flag: str) -> str:
    values: list[str] = []
    index = 0
    while index < len(args):
        value = args[index]
        if value == flag:
            if index + 1 >= len(args):
                raise TrialError(f"config {flag} is missing its value")
            values.append(args[index + 1])
            index += 2
            continue
        if value.startswith(flag + "="):
            values.append(value.split("=", 1)[1])
        index += 1
    if len(values) != 1 or not values[0]:
        raise TrialError(f"config must contain exactly one {flag} value")
    return values[0]


def _replace_option(args: Sequence[str], flag: str, replacement: str) -> list[str]:
    result: list[str] = []
    index = 0
    while index < len(args):
        value = args[index]
        if value == flag:
            if index + 1 >= len(args):
                raise TrialError(f"config {flag} is missing its value")
            index += 2
            continue
        if value.startswith(flag + "="):
            index += 1
            continue
        result.append(value)
        index += 1
    result.extend((flag, replacement))
    return result


@dataclass(frozen=True)
class TrialConfig:
    source_path: Path
    private_path: Path
    engine_log_path: Path
    config: dict[str, Any]
    engine_path: Path
    model_path: Path
    tokenizer_path: Path
    model_name: str
    context: int


def load_trial_config(
    source_path: Path,
    tag: str,
    thinking: str,
    pool_workers: int | None,
    *,
    results: Path = RESULTS,
    server_root: Path = STRATA_ROOT,
) -> TrialConfig:
    source_path = source_path.expanduser().resolve()
    try:
        source = json.loads(source_path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TrialError(f"source config is unreadable or invalid: {source_path}") from exc
    if not isinstance(source, dict):
        raise TrialError("source config must be a JSON object")
    args = source.get("args")
    if not isinstance(args, list) or any(
        not isinstance(value, str) or not value or "\x00" in value for value in args
    ):
        raise TrialError("config args must be a non-empty string list")
    if not args:
        raise TrialError("config args must not be empty")

    cwd = _config_cwd(source, server_root)
    if not cwd.is_dir():
        raise TrialError(f"config cwd is missing: {cwd}")
    exe = source.get("exe")
    if not isinstance(exe, str):
        raise TrialError("config is missing engine exe")
    engine_path = _resolve_from(exe, cwd, "exe")
    if not engine_path.is_file() or not os.access(engine_path, os.X_OK):
        raise TrialError(f"engine executable is missing or not executable: {engine_path}")

    model_value = _option_value(args, "--native")
    model_path = _resolve_from(model_value, cwd, "--native")
    if not model_path.is_file() or not os.access(model_path, os.R_OK):
        raise TrialError(f"model is missing or unreadable: {model_path}")

    tokenizer = source.get("tokenizer")
    if not isinstance(tokenizer, str):
        raise TrialError("config is missing tokenizer")
    # Strata resolves the tokenizer in the server process, whose cwd is STRATA_ROOT.
    tokenizer_path = _resolve_from(tokenizer, server_root, "tokenizer")
    if not tokenizer_path.is_dir() or not (tokenizer_path / "vocab.json").is_file():
        raise TrialError(f"tokenizer is missing vocab.json: {tokenizer_path}")

    model_name = source.get("model_name")
    if not isinstance(model_name, str) or not model_name.strip() or "\x00" in model_name:
        raise TrialError("config is missing model_name")
    try:
        context = int(_option_value(args, "--max-context"))
    except ValueError as exc:
        raise TrialError("config --max-context must be an integer") from exc
    if context < 1:
        raise TrialError("config --max-context must be positive")

    private = copy.deepcopy(source)
    private["host"] = HOST
    private["port"] = PORT
    private["open_browser"] = False
    private["reasoning_effort"] = thinking
    engine_log_path = (results / f"{tag}-engine.log").resolve()
    private["log"] = str(engine_log_path)
    if pool_workers is not None:
        private["args"] = _replace_option(args, "--pool-workers", str(pool_workers))

    return TrialConfig(
        source_path=source_path,
        private_path=(results / f"{tag}-config.json").resolve(),
        engine_log_path=engine_log_path,
        config=private,
        engine_path=engine_path,
        model_path=model_path,
        tokenizer_path=tokenizer_path,
        model_name=model_name.strip(),
        context=context,
    )


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_private_config(config: TrialConfig) -> None:
    config.private_path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(config.config, indent=2, sort_keys=True) + "\n").encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = -1
    try:
        descriptor = os.open(config.private_path, flags, 0o600)
        view = memoryview(payload)
        written = 0
        while written < len(view):
            count = os.write(descriptor, view[written:])
            if count <= 0:
                raise OSError("short private-config write")
            written += count
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        _fsync_directory(config.private_path.parent)
    except FileExistsError as exc:
        raise TrialError(f"private trial config already exists: {config.private_path}") from exc
    except BaseException:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            config.private_path.unlink()
        except FileNotFoundError:
            pass
        raise


def validate_cpu_list(value: str) -> str:
    for item in value.split(","):
        match = CPU_ITEM_RE.fullmatch(item)
        if match is None:
            raise argparse.ArgumentTypeError("must be a Linux taskset list such as 0-7,16-23")
        start = int(match.group(1))
        end = int(match.group(2)) if match.group(2) is not None else start
        stride = int(match.group(3)) if match.group(3) is not None else 1
        if end < start or stride < 1:
            raise argparse.ArgumentTypeError("CPU ranges must ascend and use a positive stride")
    return value


def server_command(
    private_config: Path,
    cpus: str | None,
    *,
    python: Path = STRATA_PYTHON,
    taskset_path: str | None = None,
) -> list[str]:
    if not python.is_file() or not os.access(python, os.X_OK):
        raise TrialError(f"Strata virtual-environment Python is missing: {python}")
    command = [
        str(python),
        "-m",
        "serve.server",
        "--engine",
        "strata",
        "--config",
        str(private_config),
        "--host",
        HOST,
        "--port",
        str(PORT),
    ]
    if cpus is None:
        return command
    if not sys.platform.startswith("linux"):
        raise TrialError("--cpus requires Linux taskset")
    taskset = taskset_path or shutil.which("taskset")
    if not taskset or not Path(taskset).is_file() or not os.access(taskset, os.X_OK):
        raise TrialError("--cpus requires an executable taskset")
    return [taskset, "--cpu-list", cpus, *command]


def server_environment(base_env: Mapping[str, str] | None = None) -> dict[str, str]:
    """Preserve the caller environment; Strata applies config lib_dirs to its engine."""
    return dict(os.environ if base_env is None else base_env)


def port_is_occupied(timeout: float = 0.3) -> bool:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.settimeout(timeout)
        return sock.connect_ex((HOST, PORT)) == 0
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


def start_server(command: Sequence[str], env: Mapping[str, str], log_path: Path) -> ServerHandle:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        log_handle = log_path.open("xb", buffering=0)
    except FileExistsError as exc:
        raise TrialError(f"server log already exists: {log_path}") from exc
    log_handle.write((f"=== Strata trial server start {utc_now()} ===\n").encode("utf-8"))
    try:
        process = subprocess.Popen(
            unblocked_exec_command(command),
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            cwd=str(STRATA_ROOT),
            env=dict(env),
            start_new_session=True,
            close_fds=True,
        )
    except BaseException:
        log_handle.close()
        raise
    return ServerHandle(process=process, pgid=process.pid, log_handle=log_handle)


def _ready_model(payload: Any, expected_model: str) -> bool:
    if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
        return False
    return any(
        isinstance(model, dict) and model.get("id") == expected_model
        for model in payload["data"]
    )


def wait_for_server(
    handle: ServerHandle,
    expected_model: str,
    *,
    timeout: float = STARTUP_TIMEOUT,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    last_problem = "models endpoint not ready"
    while time.monotonic() < deadline:
        returncode = handle.process.poll()
        if returncode is not None:
            raise TrialError(f"Strata server exited during startup with status {returncode}")
        try:
            request = Request(ENDPOINT + "/models", headers={"Accept": "application/json"})
            with urlopen(request, timeout=min(REQUEST_TIMEOUT, max(0.1, deadline - time.monotonic()))) as response:
                payload = json.loads(response.read(2 * 1024 * 1024))
            if _ready_model(payload, expected_model):
                if handle.process.poll() is not None:
                    raise TrialError("Strata server exited as its models endpoint became ready")
                return payload
            last_problem = "models endpoint did not list the configured model"
        except TrialError:
            raise
        except (OSError, URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_problem = type(exc).__name__
        time.sleep(1)
    raise TrialError(f"Strata server did not become ready within {int(timeout)}s ({last_problem})")


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
            raise TrialError("created Strata process group did not stop after SIGKILL")
        return result
    finally:
        handle.log_handle.close()


def bench_command(
    config: TrialConfig,
    thinking: str,
    suite: str,
    output: Path,
    *,
    context_target: int | None = None,
    quick_repeat: int = 2,
    hard_cases: str | None = None,
    output_tokens: int = DEFAULT_HARD_OUTPUT_TOKENS,
    request_timeout: float = DEFAULT_HARD_REQUEST_TIMEOUT,
    context_output_tokens: int = DEFAULT_CONTEXT_OUTPUT_TOKENS,
    context_timeout: float = DEFAULT_CONTEXT_TIMEOUT,
    temperature: float = 0.0,
    seed: int = 0,
    hard_repeat: int = 1,
) -> list[str]:
    suite_timeout = request_timeout if suite == "hard" else context_timeout if suite == "context" else 300
    command = [
        sys.executable,
        str(ROOT / "bench.py"),
        "--endpoint",
        ENDPOINT,
        "--backend",
        "openai",
        "--model",
        config.model_name,
        "--context",
        str(config.context),
        "--thinking",
        thinking,
        "--temperature",
        format_number(temperature),
        "--seed",
        str(seed),
        "--suite",
        suite,
        "--timeout",
        format_number(suite_timeout),
        "--fail-on-case",
        "--output",
        str(output),
    ]
    if suite == "quick":
        command.extend(("--repeat", str(quick_repeat), "--num-predict", "512"))
    elif suite == "quality":
        command.extend(("--cases", QUALITY_CASES))
    elif suite == "context":
        if context_target is None:
            raise TrialError("context suite requires a target")
        command.extend(
            ("--context-target", str(context_target), "--num-predict", str(context_output_tokens))
        )
    elif suite == "hard":
        if hard_cases is None:
            raise TrialError("hard suite requires case IDs")
        command.extend(("--cases", hard_cases, "--num-predict", str(output_tokens)))
        if hard_repeat > 1:
            command.extend(("--repeat", str(hard_repeat), "--seed-step", "1"))
    else:
        raise TrialError(f"unsupported benchmark suite {suite!r}")
    return command


def suite_process_timeout(
    suite: str,
    *,
    hard_case_count: int = 0,
    request_timeout: float = DEFAULT_HARD_REQUEST_TIMEOUT,
    context_timeout: float = DEFAULT_CONTEXT_TIMEOUT,
    hard_repeat: int = 1,
) -> float:
    if suite == "quick":
        return 4 * 300 + 90
    if suite == "quality":
        return 3 * 300 + 90
    if suite == "hard":
        if hard_case_count < 1:
            raise TrialError("hard suite requires at least one case")
        return hard_case_count * hard_repeat * request_timeout + 90
    if suite == "context":
        return context_timeout + 90
    return 300 + 90


def run_suite(
    command: Sequence[str],
    suite: str,
    *,
    hard_case_count: int = 0,
    request_timeout: float = DEFAULT_HARD_REQUEST_TIMEOUT,
    context_timeout: float = DEFAULT_CONTEXT_TIMEOUT,
    hard_repeat: int = 1,
) -> int:
    try:
        completed = subprocess.run(
            list(command),
            stdin=subprocess.DEVNULL,
            timeout=suite_process_timeout(
                suite,
                hard_case_count=hard_case_count,
                request_timeout=request_timeout,
                context_timeout=context_timeout,
                hard_repeat=hard_repeat,
            ),
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise TrialError(f"{suite} benchmark process exceeded its total timeout") from exc
    return completed.returncode


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


def result_paths(
    tag: str,
    quality: bool,
    context_target: int | None,
    skip_quick: bool = False,
    hard_cases: str | None = None,
) -> list[Path]:
    paths = [
        RESULTS / f"{tag}-config.json",
        RESULTS / f"{tag}-config.shared-settings.json",
        RESULTS / f"{tag}-engine.log",
        RESULTS / f"{tag}-server.log",
        RESULTS / f"{tag}-runtime-events.jsonl",
    ]
    if not skip_quick:
        paths.append(RESULTS / f"{tag}-quick.jsonl")
    if quality:
        paths.append(RESULTS / f"{tag}-quality.jsonl")
    if context_target is not None:
        paths.append(RESULTS / f"{tag}-context.jsonl")
    if hard_cases is not None:
        paths.append(RESULTS / f"{tag}-hard.jsonl")
    return paths


def assert_results_unused(
    tag: str,
    quality: bool,
    context_target: int | None,
    skip_quick: bool = False,
    hard_cases: str | None = None,
) -> None:
    collisions = [
        path
        for path in result_paths(tag, quality, context_target, skip_quick, hard_cases)
        if os.path.lexists(path)
    ]
    if collisions:
        raise TrialError("trial tag would overwrite existing result: " + ", ".join(str(path) for path in collisions))


def run_trial(config: TrialConfig, args: argparse.Namespace, events: EventLog) -> bool:
    command = server_command(config.private_path, args.cpus)
    handle: ServerHandle | None = None
    success = True
    try:
        with defer_termination_signals():
            handle = start_server(command, server_environment(), RESULTS / f"{args.tag}-server.log")
        events.write("server_started", pid=handle.process.pid, pgid=handle.pgid, argv=command)
        models = wait_for_server(
            handle,
            config.model_name,
            timeout=getattr(args, "startup_timeout", STARTUP_TIMEOUT),
        )
        events.write("server_ready", model=config.model_name, models=models)

        suites: list[tuple[str, int | None]] = []
        if not getattr(args, "skip_quick", False):
            suites.append(("quick", None))
        if args.quality:
            suites.append(("quality", None))
        if args.context_target is not None:
            suites.append(("context", args.context_target))
        hard_cases = getattr(args, "hard_cases", None)
        if hard_cases is not None:
            suites.append(("hard", None))
        for suite, target in suites:
            output = RESULTS / f"{args.tag}-{suite}.jsonl"
            command = bench_command(
                config,
                args.thinking,
                suite,
                output,
                context_target=target,
                quick_repeat=getattr(args, "quick_repeat", 2),
                hard_cases=hard_cases,
                output_tokens=getattr(args, "output_tokens", DEFAULT_HARD_OUTPUT_TOKENS),
                request_timeout=getattr(args, "request_timeout", DEFAULT_HARD_REQUEST_TIMEOUT),
                context_output_tokens=getattr(
                    args, "context_output_tokens", DEFAULT_CONTEXT_OUTPUT_TOKENS
                ),
                context_timeout=getattr(args, "context_timeout", DEFAULT_CONTEXT_TIMEOUT),
                temperature=getattr(args, "temperature", 0.0),
                seed=getattr(args, "seed", 0),
                hard_repeat=getattr(args, "hard_repeat", 1),
            )
            events.write("suite_started", suite=suite, result=str(output.relative_to(ROOT)))
            returncode = run_suite(
                command,
                suite,
                hard_case_count=len(hard_cases.split(",")) if suite == "hard" and hard_cases else 0,
                request_timeout=getattr(args, "request_timeout", DEFAULT_HARD_REQUEST_TIMEOUT),
                context_timeout=getattr(args, "context_timeout", DEFAULT_CONTEXT_TIMEOUT),
                hard_repeat=getattr(args, "hard_repeat", 1),
            )
            passed = returncode == 0
            events.write("suite_complete", suite=suite, returncode=returncode, passed=passed)
            success = success and passed
    except (TrialError, OSError, subprocess.SubprocessError) as exc:
        events.write("trial_failed", error_type=type(exc).__name__, error=str(exc))
        success = False
    finally:
        if handle is not None:
            try:
                with defer_termination_signals():
                    cleanup = stop_server(handle)
                events.write("server_stopped", **cleanup)
            except (TrialError, OSError) as exc:
                events.write(
                    "server_cleanup_failed",
                    error_type=type(exc).__name__,
                    error=str(exc),
                )
                success = False
    return success


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


def startup_timeout_value(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or not MIN_STARTUP_TIMEOUT <= parsed <= MAX_STARTUP_TIMEOUT:
        raise argparse.ArgumentTypeError(
            f"must be a finite number from {int(MIN_STARTUP_TIMEOUT)} through "
            f"{int(MAX_STARTUP_TIMEOUT)}"
        )
    return parsed


def temperature_value(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0 or parsed > 2:
        raise argparse.ArgumentTypeError("must be a finite number from 0 through 2")
    return parsed


def format_number(value: float) -> str:
    return format(value, ".15g")


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--tag", required=True, help="safe, unique result filename prefix")
    parser.add_argument(
        "--startup-timeout",
        type=startup_timeout_value,
        default=STARTUP_TIMEOUT,
        help="server readiness timeout in seconds (30-480)",
    )
    parser.add_argument("--thinking", choices=("none", "low", "medium"), default="none")
    parser.add_argument("--temperature", type=temperature_value, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quality", action="store_true")
    parser.add_argument("--context-target", type=positive_int)
    parser.add_argument(
        "--context-output-tokens",
        type=positive_int,
        default=DEFAULT_CONTEXT_OUTPUT_TOKENS,
        help="generation token cap for the context suite",
    )
    parser.add_argument(
        "--context-timeout",
        type=positive_float,
        default=DEFAULT_CONTEXT_TIMEOUT,
        help="per-request timeout in seconds for the context suite",
    )
    parser.add_argument("--quick-repeat", type=int, choices=(1, 2), default=2)
    parser.add_argument("--skip-quick", action="store_true")
    parser.add_argument("--hard-cases", type=parse_case_ids)
    parser.add_argument("--hard-repeat", type=positive_int, choices=(1, 2, 3), default=1)
    parser.add_argument("--output-tokens", type=positive_int, default=DEFAULT_HARD_OUTPUT_TOKENS)
    parser.add_argument(
        "--request-timeout",
        type=positive_float,
        default=DEFAULT_HARD_REQUEST_TIMEOUT,
        help="per-request timeout in seconds for the hard suite",
    )
    parser.add_argument("--pool-workers", type=positive_int)
    parser.add_argument("--cpus", type=validate_cpu_list, help="Linux taskset CPU list, for example 0-7,16-23")
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
    if args.cpus is not None and not sys.platform.startswith("linux"):
        parser.error("--cpus requires Linux taskset")

    try:
        config = load_trial_config(
            args.config,
            args.tag,
            args.thinking,
            args.pool_workers,
        )
        # Validate the complete launch command before writing any trial artifact.
        server_command(config.private_path, args.cpus)
        gpu_processes = assert_preflight_clear()
        assert_results_unused(
            args.tag,
            args.quality,
            args.context_target,
            args.skip_quick,
            args.hard_cases,
        )
        write_private_config(config)
    except (TrialError, OSError) as exc:
        print(f"strata_trial: {exc}", file=sys.stderr)
        return 2

    run_id = str(uuid.uuid4())
    events = EventLog(RESULTS / f"{args.tag}-runtime-events.jsonl", run_id, args.tag)
    install_signal_handlers()
    events.write(
        "trial_started",
        source_config=str(config.source_path),
        private_config=str(config.private_path),
        engine=str(config.engine_path),
        model=str(config.model_path),
        tokenizer=str(config.tokenizer_path),
        model_name=config.model_name,
        context=config.context,
        startup_timeout=args.startup_timeout,
        thinking=args.thinking,
        temperature=args.temperature,
        seed=args.seed,
        quality=args.quality,
        context_target=args.context_target,
        context_output_tokens=args.context_output_tokens,
        context_timeout=args.context_timeout,
        quick_repeat=args.quick_repeat,
        skip_quick=args.skip_quick,
        hard_cases=args.hard_cases,
        hard_repeat=args.hard_repeat,
        output_tokens=args.output_tokens,
        request_timeout=args.request_timeout,
        pool_workers=args.pool_workers,
        cpus=args.cpus,
        engine_environment=engine_environment_metadata(),
        observed_gpu_compute_processes=gpu_processes,
    )
    try:
        passed = run_trial(config, args, events)
    except KeyboardInterrupt:
        events.write("trial_interrupted")
        return 130
    events.write("trial_complete", passed=passed)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
