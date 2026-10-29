#!/usr/bin/env python3
"""Run one isolated llama.cpp benchmark trial on loopback port 18081."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import socket
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.error import URLError
from urllib.request import Request, urlopen

if __package__:
    from . import strata_trial as safe
else:
    import strata_trial as safe


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
DEFAULT_CONFIG = ROOT / "runtime" / "qwen122-32k.json"
HOST = "127.0.0.1"
PORT = 18081
ENDPOINT = f"http://{HOST}:{PORT}/v1"
STARTUP_TIMEOUT = 180.0
MIN_STARTUP_TIMEOUT = 30.0
MAX_STARTUP_TIMEOUT = 480.0
REQUEST_TIMEOUT = 3.0
DEFAULT_HARD_OUTPUT_TOKENS = 2048
DEFAULT_HARD_REQUEST_TIMEOUT = 270.0
TAG_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")
CASE_ID_RE = re.compile(r"^[A-Za-z0-9_]{1,128}$")
MAX_HARD_CASES = 64
HASH_RE = re.compile(r"^[0-9a-f]{64}$")
CUDA_BACKEND_LIBRARY = "libggml-cuda.so"
OWNED_SERVER_OPTIONS = {
    "-m",
    "--model",
    "--host",
    "--port",
    "-a",
    "--alias",
    "-c",
    "--ctx-size",
    "--reasoning",
    "--reasoning-effort",
}

TrialError = safe.TrialError
ServerHandle = safe.ServerHandle
EventLog = safe.EventLog


@dataclass(frozen=True)
class Shard:
    name: str
    path: Path
    size: int
    sha256: str


@dataclass(frozen=True)
class TrialConfig:
    source_path: Path
    server_path: Path
    model_path: Path
    model_name: str
    context: int
    lib_dirs: tuple[Path, ...]
    args: tuple[str, ...]
    manifest_path: Path
    shards: tuple[Shard, ...]


def _read_json(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TrialError(f"{description} is unreadable or invalid: {path}") from exc
    if not isinstance(value, dict):
        raise TrialError(f"{description} must be a JSON object")
    return value


def _resolve(value: Any, base: Path, field: str) -> Path:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise TrialError(f"config {field} must be a non-empty path")
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _validate_extra_args(values: Any) -> tuple[str, ...]:
    if not isinstance(values, list) or any(
        not isinstance(value, str) or not value or "\x00" in value for value in values
    ):
        raise TrialError("config args must be a string list")
    for value in values:
        option = value.split("=", 1)[0]
        if option in OWNED_SERVER_OPTIONS:
            raise TrialError(f"config args must not override runner-owned option {option}")
    return tuple(values)


def _load_shards(values: Any, model_dir: Path) -> tuple[Shard, ...]:
    if not isinstance(values, list) or not values:
        raise TrialError("config shards must be a non-empty list")
    shards: list[Shard] = []
    names: set[str] = set()
    for index, row in enumerate(values):
        if not isinstance(row, dict):
            raise TrialError(f"config shard {index} must be an object")
        name = row.get("name")
        size = row.get("size")
        sha256 = row.get("sha256")
        if (
            not isinstance(name, str)
            or not name
            or Path(name).name != name
            or "\x00" in name
            or name in names
        ):
            raise TrialError(f"config shard {index} has an invalid or duplicate name")
        if not isinstance(size, int) or isinstance(size, bool) or size < 1:
            raise TrialError(f"config shard {name} has an invalid size")
        if not isinstance(sha256, str) or HASH_RE.fullmatch(sha256) is None:
            raise TrialError(f"config shard {name} has an invalid SHA-256")
        path = model_dir / name
        if not path.is_file() or not os.access(path, os.R_OK):
            raise TrialError(f"model shard is missing or unreadable: {path}")
        if path.stat().st_size != size:
            raise TrialError(f"model shard has the wrong size: {path}")
        names.add(name)
        shards.append(Shard(name, path, size, sha256))
    return tuple(shards)


def _verify_manifest(path: Path, shards: Sequence[Shard]) -> None:
    manifest = _read_json(path, "artifact manifest")
    verdict = manifest.get("verdict")
    if not isinstance(verdict, dict) or verdict.get("artifact_complete") is not True:
        raise TrialError("artifact manifest does not mark the split model complete")
    rows = manifest.get("shards")
    if not isinstance(rows, list):
        raise TrialError("artifact manifest has no shard table")
    by_name = {
        row.get("name"): row
        for row in rows
        if isinstance(row, dict) and isinstance(row.get("name"), str)
    }
    for shard in shards:
        row = by_name.get(shard.name)
        if (
            not isinstance(row, dict)
            or row.get("size") != shard.size
            or row.get("sha256") != shard.sha256
            or row.get("verified") is not True
        ):
            raise TrialError(f"artifact manifest does not verify shard {shard.name}")


def _option_value(args: Sequence[str], option: str) -> str | None:
    values: list[str] = []
    for index, value in enumerate(args):
        if value == option:
            if index + 1 == len(args) or args[index + 1].startswith("--"):
                raise TrialError(f"config {option} is missing its value")
            values.append(args[index + 1])
        elif value.startswith(option + "="):
            values.append(value[len(option) + 1 :])
    if len(values) > 1:
        raise TrialError(f"config repeats {option}")
    return values[0] if values else None


def _verify_eagle3_manifest(
    source: Mapping[str, Any], base: Path, args: Sequence[str], target: Shard,
) -> None:
    if _option_value(args, "--spec-type") != "draft-eagle3":
        return
    manifest_path = _resolve(source.get("draft_artifact_manifest"), base, "draft_artifact_manifest")
    manifest = _read_json(manifest_path, "EAGLE-3 artifact manifest")
    verdict = manifest.get("verdict")
    compatibility = manifest.get("compatibility")
    if (
        not isinstance(verdict, dict)
        or verdict.get("artifact_complete") is not True
        or verdict.get("full_sha256_verified") is not True
        or not isinstance(compatibility, dict)
        or compatibility.get("activation_allowed") is not True
    ):
        raise TrialError("EAGLE-3 artifact manifest does not allow activation")
    artifact = manifest.get("gguf_artifact")
    target_receipt = manifest.get("target")
    if not isinstance(artifact, dict) or not isinstance(target_receipt, dict):
        raise TrialError("EAGLE-3 artifact manifest lacks draft or target identity")
    draft_path = _resolve(_option_value(args, "--spec-draft-model"), base, "--spec-draft-model")
    receipt_path = _resolve(artifact.get("path"), manifest_path.parent, "draft artifact path")
    size = artifact.get("size")
    digest = artifact.get("sha256")
    if (
        draft_path != receipt_path
        or not isinstance(size, int) or isinstance(size, bool) or size < 1
        or not isinstance(digest, str) or HASH_RE.fullmatch(digest) is None
        or not draft_path.is_file() or not os.access(draft_path, os.R_OK)
        or draft_path.stat().st_size != size
    ):
        raise TrialError("EAGLE-3 draft does not match its verified artifact identity")
    target_path = _resolve(
        target_receipt.get("path", target_receipt.get("model")),
        manifest_path.parent, "draft target path",
    )
    if (
        target_path != target.path
        or target_receipt.get("size") != target.size
        or target_receipt.get("sha256") != target.sha256
    ):
        raise TrialError("EAGLE-3 compatibility receipt is for a different target")


def load_trial_config(source_path: Path) -> TrialConfig:
    source_path = source_path.expanduser().resolve()
    source = _read_json(source_path, "trial config")
    base = source_path.parent

    if source.get("host") != HOST or source.get("port") != PORT:
        raise TrialError(f"config must bind exactly to {HOST}:{PORT}")
    server_path = _resolve(source.get("server"), base, "server")
    if not server_path.is_file() or not os.access(server_path, os.X_OK):
        raise TrialError(f"llama-server is missing or not executable: {server_path}")
    model_path = _resolve(source.get("model"), base, "model")
    if not model_path.is_file() or not os.access(model_path, os.R_OK):
        raise TrialError(f"model is missing or unreadable: {model_path}")
    model_name = source.get("model_name")
    if (
        not isinstance(model_name, str)
        or not model_name
        or "\x00" in model_name
        or any(character.isspace() for character in model_name)
    ):
        raise TrialError("config model_name must be a non-empty name without whitespace")
    context = source.get("context")
    if not isinstance(context, int) or isinstance(context, bool) or context < 512:
        raise TrialError("config context must be an integer of at least 512")

    lib_values = source.get("lib_dirs")
    if not isinstance(lib_values, list) or not lib_values:
        raise TrialError("config lib_dirs must be a non-empty path list")
    lib_dirs = tuple(_resolve(value, base, "lib_dirs") for value in lib_values)
    for directory in lib_dirs:
        if not directory.is_dir():
            raise TrialError(f"library directory is missing: {directory}")

    args = _validate_extra_args(source.get("args", []))
    shards = _load_shards(source.get("shards"), model_path.parent)
    if shards[0].path != model_path:
        raise TrialError("config model must be the first listed split shard")
    manifest_path = _resolve(source.get("artifact_manifest"), base, "artifact_manifest")
    _verify_manifest(manifest_path, shards)
    _verify_eagle3_manifest(source, base, args, shards[0])
    return TrialConfig(
        source_path=source_path,
        server_path=server_path,
        model_path=model_path,
        model_name=model_name,
        context=context,
        lib_dirs=lib_dirs,
        args=args,
        manifest_path=manifest_path,
        shards=shards,
    )


def cuda_backend_path(config: TrialConfig) -> Path:
    backend_candidates = tuple(
        directory / CUDA_BACKEND_LIBRARY
        for directory in config.lib_dirs
        if (directory / CUDA_BACKEND_LIBRARY).is_file()
    )
    if len(backend_candidates) != 1:
        raise TrialError(
            f"expected exactly one {CUDA_BACKEND_LIBRARY} in config lib_dirs, "
            f"found {len(backend_candidates)}"
        )
    return backend_candidates[0].resolve()


def server_environment(config: TrialConfig, base_env: Mapping[str, str] | None = None) -> dict[str, str]:
    environment = dict(os.environ if base_env is None else base_env)
    backend_path = cuda_backend_path(config)
    prefix = os.pathsep.join(str(path) for path in config.lib_dirs)
    existing = environment.get("LD_LIBRARY_PATH")
    environment["LD_LIBRARY_PATH"] = prefix if not existing else prefix + os.pathsep + existing
    # GGML scans the executable directory and cwd for dynamic backends.  The
    # portable Ollama bundle keeps CUDA in cuda_v13/, so LD_LIBRARY_PATH alone
    # can resolve its dependencies but cannot make the plugin discoverable.
    environment["GGML_BACKEND_PATH"] = str(backend_path)
    return environment


def server_command(config: TrialConfig, thinking: str) -> list[str]:
    command = [
        str(config.server_path),
        "--model",
        str(config.model_path),
        "--alias",
        config.model_name,
        "--ctx-size",
        str(config.context),
        "--host",
        HOST,
        "--port",
        str(PORT),
        *config.args,
        "--reasoning",
        "off" if thinking == "none" else "on",
    ]
    if thinking == "medium":
        command.extend(("--reasoning-effort", "medium"))
    elif thinking != "none":
        raise TrialError(f"unsupported thinking mode {thinking!r}")
    return command


def port_is_occupied(timeout: float = 0.3) -> bool:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.settimeout(timeout)
        return sock.connect_ex((HOST, PORT)) == 0
    finally:
        sock.close()


def assert_preflight_clear() -> list[dict[str, Any]]:
    if port_is_occupied():
        raise TrialError(f"loopback port {PORT} is occupied; refusing to alter its process")
    processes = safe.gpu_compute_processes()
    busy = [process for process in processes if process["used_mib"] > safe.GPU_PROCESS_LIMIT_MIB]
    if busy:
        summary = ", ".join(f"pid {item['pid']} ({item['used_mib']} MiB)" for item in busy)
        raise TrialError(f"GPU has another compute process over {safe.GPU_PROCESS_LIMIT_MIB} MiB: {summary}")
    return processes


def start_server(
    config: TrialConfig,
    command: Sequence[str],
    environment: Mapping[str, str],
    log_path: Path,
) -> ServerHandle:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        log_handle = log_path.open("xb", buffering=0)
    except FileExistsError as exc:
        raise TrialError(f"server log already exists: {log_path}") from exc
    log_handle.write((f"=== llama.cpp trial server start {safe.utc_now()} ===\n").encode("utf-8"))
    try:
        process = subprocess.Popen(
            safe.unblocked_exec_command(command),
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            cwd=str(config.server_path.parent),
            env=dict(environment),
            start_new_session=True,
            close_fds=True,
        )
    except BaseException:
        log_handle.close()
        raise
    return ServerHandle(process=process, pgid=process.pid, log_handle=log_handle)


def _model_ids(payload: Any) -> list[str]:
    if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
        return []
    return [
        model["id"]
        for model in payload["data"]
        if isinstance(model, dict) and isinstance(model.get("id"), str)
    ]


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
            raise TrialError(f"llama-server exited during startup with status {returncode}")
        try:
            request = Request(ENDPOINT + "/models", headers={"Accept": "application/json"})
            with urlopen(request, timeout=min(REQUEST_TIMEOUT, max(0.1, deadline - time.monotonic()))) as response:
                payload = json.loads(response.read(2 * 1024 * 1024))
            if expected_model in _model_ids(payload):
                if handle.process.poll() is not None:
                    raise TrialError("llama-server exited as its models endpoint became ready")
                return payload
            last_problem = "models endpoint did not list the configured alias"
        except TrialError:
            raise
        except (OSError, URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_problem = type(exc).__name__
        time.sleep(1)
    raise TrialError(f"llama-server did not become ready within {int(timeout)}s ({last_problem})")


def format_number(value: float) -> str:
    return format(value, ".15g")


def bench_command(
    config: TrialConfig,
    thinking: str,
    suite: str,
    output: Path,
    *,
    quick_repeat: int = 1,
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
        "openai",
        "--model",
        config.model_name,
        "--context",
        str(config.context),
        "--thinking",
        thinking,
        "--suite",
        suite,
        "--timeout",
        format_number(request_timeout) if suite == "hard" else "300",
        "--fail-on-case",
        "--output",
        str(output),
    ]
    if suite == "quick":
        command.extend(("--repeat", str(quick_repeat), "--num-predict", "512"))
    elif suite == "hard":
        if hard_cases is None:
            raise TrialError("hard suite requires case IDs")
        command.extend(("--cases", hard_cases, "--num-predict", str(output_tokens)))
    else:
        raise TrialError(f"unsupported benchmark suite {suite!r}")
    return command


def suite_process_timeout(
    suite: str,
    *,
    hard_case_count: int = 0,
    request_timeout: float = DEFAULT_HARD_REQUEST_TIMEOUT,
) -> float:
    if suite == "quick":
        return 4 * 300 + 60
    if suite == "hard":
        if hard_case_count < 1:
            raise TrialError("hard suite requires at least one case")
        return hard_case_count * request_timeout + 60
    raise TrialError(f"unsupported benchmark suite {suite!r}")


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


def result_paths(tag: str, skip_quick: bool, hard_cases: str | None) -> list[Path]:
    paths = [RESULTS / f"{tag}-server.log", RESULTS / f"{tag}-runtime-events.jsonl"]
    if not skip_quick:
        paths.append(RESULTS / f"{tag}-quick.jsonl")
    if hard_cases is not None:
        paths.append(RESULTS / f"{tag}-hard.jsonl")
    return paths


def assert_results_unused(tag: str, skip_quick: bool, hard_cases: str | None) -> None:
    collisions = [path for path in result_paths(tag, skip_quick, hard_cases) if os.path.lexists(path)]
    if collisions:
        raise TrialError("trial tag would overwrite existing result: " + ", ".join(str(path) for path in collisions))


def run_trial(config: TrialConfig, args: argparse.Namespace, events: EventLog) -> bool:
    command = server_command(config, args.thinking)
    handle: ServerHandle | None = None
    success = True
    try:
        with safe.defer_termination_signals():
            handle = start_server(
                config,
                command,
                server_environment(config),
                RESULTS / f"{args.tag}-server.log",
            )
        events.write("server_started", pid=handle.process.pid, pgid=handle.pgid, argv=command)
        models = wait_for_server(
            handle,
            config.model_name,
            timeout=getattr(args, "startup_timeout", STARTUP_TIMEOUT),
        )
        events.write("server_ready", model=config.model_name, model_ids=_model_ids(models))

        suites: list[str] = []
        if not args.skip_quick:
            suites.append("quick")
        if args.hard_cases is not None:
            suites.append("hard")
        for suite in suites:
            output = RESULTS / f"{args.tag}-{suite}.jsonl"
            command = bench_command(
                config,
                args.thinking,
                suite,
                output,
                quick_repeat=args.quick_repeat,
                hard_cases=args.hard_cases,
                output_tokens=args.output_tokens,
                request_timeout=args.request_timeout,
            )
            events.write("suite_started", suite=suite, result=str(output.relative_to(ROOT)))
            returncode = run_suite(
                command,
                suite,
                hard_case_count=(
                    len(args.hard_cases.split(",")) if suite == "hard" and args.hard_cases else 0
                ),
                request_timeout=args.request_timeout,
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
                with safe.defer_termination_signals():
                    cleanup = safe.stop_server(handle)
                events.write("server_stopped", **cleanup)
            except (TrialError, OSError) as exc:
                events.write("server_cleanup_failed", error_type=type(exc).__name__, error=str(exc))
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
    parser.add_argument("--thinking", choices=("none", "medium"), default="none")
    parser.add_argument("--quick-repeat", type=int, choices=(1, 2), default=1)
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
    if args.skip_quick and args.hard_cases is None:
        parser.error("--skip-quick requires --hard-cases")

    try:
        config = load_trial_config(args.config)
        command = server_command(config, args.thinking)
        if not command:
            raise TrialError("empty server command")
        gpu_processes = assert_preflight_clear()
        assert_results_unused(args.tag, args.skip_quick, args.hard_cases)
    except (TrialError, OSError) as exc:
        print(f"llama_trial: {exc}", file=sys.stderr)
        return 2

    run_id = str(uuid.uuid4())
    events = EventLog(RESULTS / f"{args.tag}-runtime-events.jsonl", run_id, args.tag)
    safe.install_signal_handlers()
    events.write(
        "trial_started",
        source_config=str(config.source_path),
        server=str(config.server_path),
        model=str(config.model_path),
        model_name=config.model_name,
        context=config.context,
        startup_timeout=args.startup_timeout,
        thinking=args.thinking,
        quick_repeat=args.quick_repeat,
        skip_quick=args.skip_quick,
        hard_cases=args.hard_cases,
        output_tokens=args.output_tokens,
        request_timeout=args.request_timeout,
        artifact_manifest=str(config.manifest_path),
        shard_sha256=[shard.sha256 for shard in config.shards],
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
