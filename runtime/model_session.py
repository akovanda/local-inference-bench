#!/usr/bin/env python3
"""Run bounded benchmarks against one persistent Strata or llama.cpp server."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
import subprocess
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__:
    from . import llama_trial as llama
    from . import strata_trial as safe
else:
    import llama_trial as llama
    import strata_trial as safe


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
MAX_PLAN_STEPS = 64
MAX_OUTPUT_TOKENS = 65_536
MAX_REQUEST_TIMEOUT = 21_600.0
MAX_CONTEXT_TOKENS = 131_072
POST_STEP_HEALTH_TIMEOUT = 10.0
CONTROL_MAX_BYTES = 4096
CONTROL_INVALID_GRACE_SECONDS = 30.0
CONTROL_POLL_SECONDS = 1.0
STEP_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
SUITES = ("quick", "quality", "hard", "context")
RUNTIME_BACKENDS = ("strata", "llama")
KNOWN_CASES = {
    "quick": ("quick_generation", "quick_json_extract"),
    "quality": (
        "quality_merge_intervals",
        "quality_stable_toposort",
        "quality_deep_merge",
        "quality_retry_schedule",
        "quality_constraints",
    ),
    "hard": (
        "hard_transaction_ledger",
        "hard_weighted_intervals",
        "hard_temporal_kv",
        "hard_dependency_constraints",
    ),
    "context": (),
}
STEP_KEYS = {
    "id",
    "suite",
    "cases",
    "thinking",
    "temperature",
    "seed",
    "output_tokens",
    "request_timeout",
    "repeat",
    "context_target",
}
PLAN_KEYS = {"schema_version", "name", "steps"}


class SessionError(RuntimeError):
    pass


@dataclass(frozen=True)
class Step:
    step_id: str
    suite: str
    cases: tuple[str, ...]
    thinking: str
    temperature: float
    seed: int
    output_tokens: int
    request_timeout: float
    repeat: int
    context_target: int | None

    def controls(self, runtime_backend: str = "strata") -> dict[str, Any]:
        result = asdict(self)
        result["id"] = result.pop("step_id")
        result["cases"] = list(self.cases)
        result["seed_step"] = 1 if self.repeat > 1 else 0
        result["per_request_reasoning_effort"] = True
        if runtime_backend == "llama":
            result["chat_template_thinking"] = (
                "off" if self.thinking == "none" else "on"
            )
        return result


@dataclass(frozen=True)
class Plan:
    source_path: Path
    sha256: str
    name: str | None
    steps: tuple[Step, ...]


def _plain_int(value: Any, field: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SessionError(f"{field} must be an integer")
    if not minimum <= value <= maximum:
        raise SessionError(f"{field} must be from {minimum} through {maximum}")
    return value


def _finite_number(value: Any, field: str, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SessionError(f"{field} must be a number")
    parsed = float(value)
    if not math.isfinite(parsed) or not minimum <= parsed <= maximum:
        raise SessionError(f"{field} must be finite and from {minimum:g} through {maximum:g}")
    return parsed


def _parse_cases(value: Any, suite: str, field: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if suite == "context":
        raise SessionError(f"{field} is not supported for the generated context suite")
    if isinstance(value, str):
        items = value.split(",")
    elif isinstance(value, list) and all(isinstance(item, str) for item in value):
        items = value
    else:
        raise SessionError(f"{field} must be a case ID list or comma-separated string")
    if not items:
        raise SessionError(f"{field} must not be empty")
    try:
        normalized = safe.parse_case_ids(",".join(items)).split(",")
    except argparse.ArgumentTypeError as exc:
        raise SessionError(f"{field}: {exc}") from exc
    unknown = [item for item in normalized if item not in KNOWN_CASES[suite]]
    if unknown:
        raise SessionError(f"{field} contains case IDs outside {suite}: {', '.join(unknown)}")
    return tuple(normalized)


def _parse_step(raw: Any, index: int) -> Step:
    prefix = f"steps[{index}]"
    if not isinstance(raw, dict):
        raise SessionError(f"{prefix} must be an object")
    unknown = sorted(set(raw) - STEP_KEYS)
    if unknown:
        raise SessionError(f"{prefix} has unknown field(s): {', '.join(unknown)}")

    step_id = raw.get("id")
    if not isinstance(step_id, str) or STEP_ID_RE.fullmatch(step_id) is None:
        raise SessionError(f"{prefix}.id must be a safe 1-64 character identifier")
    suite = raw.get("suite")
    if suite not in SUITES:
        raise SessionError(f"{prefix}.suite must be one of {', '.join(SUITES)}")
    thinking = raw.get("thinking", "none")
    if thinking not in ("none", "low", "medium"):
        raise SessionError(f"{prefix}.thinking must be none, low, or medium")

    default_output = 768 if suite == "context" else 512 if suite == "quick" else 2048
    output_tokens = _plain_int(
        raw.get("output_tokens", default_output),
        f"{prefix}.output_tokens",
        1,
        MAX_OUTPUT_TOKENS,
    )
    request_timeout = _finite_number(
        raw.get("request_timeout", 300.0),
        f"{prefix}.request_timeout",
        1.0,
        MAX_REQUEST_TIMEOUT,
    )
    temperature = _finite_number(
        raw.get("temperature", 0.0), f"{prefix}.temperature", 0.0, 2.0
    )
    seed = _plain_int(raw.get("seed", 0), f"{prefix}.seed", -(2**63), 2**63 - 1)
    repeat = _plain_int(raw.get("repeat", 1), f"{prefix}.repeat", 1, 3)
    cases = _parse_cases(raw.get("cases"), suite, f"{prefix}.cases")

    context_target: int | None = None
    if suite == "context":
        if "context_target" not in raw:
            raise SessionError(f"{prefix}.context_target is required for the context suite")
        context_target = _plain_int(
            raw["context_target"], f"{prefix}.context_target", 512, 2**31 - 1
        )
    elif "context_target" in raw:
        raise SessionError(f"{prefix}.context_target is only valid for the context suite")

    return Step(
        step_id=step_id,
        suite=suite,
        cases=cases,
        thinking=thinking,
        temperature=temperature,
        seed=seed,
        output_tokens=output_tokens,
        request_timeout=request_timeout,
        repeat=repeat,
        context_target=context_target,
    )


def load_plan(path: Path) -> Plan:
    source_path = path.expanduser().resolve()
    try:
        raw_bytes = source_path.read_bytes()
        raw = json.loads(raw_bytes.decode("utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SessionError(f"plan is unreadable or invalid: {source_path}") from exc
    if not isinstance(raw, dict):
        raise SessionError("plan must be a JSON object")
    unknown = sorted(set(raw) - PLAN_KEYS)
    if unknown:
        raise SessionError(f"plan has unknown field(s): {', '.join(unknown)}")
    if raw.get("schema_version", 1) != 1:
        raise SessionError("plan schema_version must be 1")
    name = raw.get("name")
    if name is not None and (not isinstance(name, str) or not name.strip() or len(name) > 160):
        raise SessionError("plan name must be a non-empty string of at most 160 characters")
    steps_raw = raw.get("steps")
    if not isinstance(steps_raw, list) or not 1 <= len(steps_raw) <= MAX_PLAN_STEPS:
        raise SessionError(f"plan steps must contain 1-{MAX_PLAN_STEPS} entries")
    steps = tuple(_parse_step(step, index) for index, step in enumerate(steps_raw))
    step_ids = [step.step_id for step in steps]
    if len(set(step_ids)) != len(step_ids):
        raise SessionError("plan step IDs must be unique")
    return Plan(
        source_path=source_path,
        sha256=hashlib.sha256(raw_bytes).hexdigest(),
        name=name.strip() if isinstance(name, str) else None,
        steps=steps,
    )


def validate_plan_context(plan: Plan, selected_context: int) -> None:
    if selected_context > MAX_CONTEXT_TOKENS:
        raise SessionError(
            f"selected context {selected_context} exceeds the maximum {MAX_CONTEXT_TOKENS}"
        )
    for step in plan.steps:
        if step.context_target is not None and step.context_target > selected_context:
            raise SessionError(
                f"step {step.step_id} context_target {step.context_target} exceeds "
                f"the selected context {selected_context}"
            )


def validate_control_path(path: Path, results: Path = RESULTS) -> Path:
    candidate = path.expanduser()
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    resolved = candidate.resolve()
    if resolved.parent != results.resolve() or resolved.suffix != ".json":
        raise SessionError("--control-file must be a .json file directly under results")
    return resolved


def result_paths(
    tag: str,
    plan: Plan,
    results: Path = RESULTS,
    control_file: Path | None = None,
    *,
    runtime_backend: str = "strata",
) -> list[Path]:
    paths = [
        results / f"{tag}-server.log",
        results / f"{tag}-runtime-events.jsonl",
        results / f"{tag}-session-report.json",
    ]
    if runtime_backend == "strata":
        paths[:0] = [
            results / f"{tag}-config.json",
            results / f"{tag}-config.shared-settings.json",
            results / f"{tag}-engine.log",
        ]
    paths.extend(
        results / f"{tag}-{index:02d}-{step.step_id}.jsonl"
        for index, step in enumerate(plan.steps, start=1)
    )
    if control_file is not None:
        paths.append(control_file)
    return paths


def assert_results_unused(
    tag: str,
    plan: Plan,
    results: Path = RESULTS,
    control_file: Path | None = None,
    *,
    runtime_backend: str = "strata",
) -> None:
    collisions = [
        path
        for path in result_paths(
            tag,
            plan,
            results,
            control_file,
            runtime_backend=runtime_backend,
        )
        if os.path.lexists(path)
    ]
    if collisions:
        raise SessionError(
            "session tag would overwrite existing result: "
            + ", ".join(str(path) for path in collisions)
        )


def bench_command(
    config: safe.TrialConfig | llama.TrialConfig,
    step: Step,
    output: Path,
    runtime_backend: str = "strata",
) -> list[str]:
    endpoint = safe.ENDPOINT if runtime_backend == "strata" else llama.ENDPOINT
    command = [
        sys.executable,
        str(ROOT / "bench.py"),
        "--endpoint",
        endpoint,
        "--backend",
        "openai",
        "--model",
        config.model_name,
        "--context",
        str(config.context),
        "--thinking",
        step.thinking,
        "--temperature",
        safe.format_number(step.temperature),
        "--seed",
        str(step.seed),
        "--suite",
        step.suite,
        "--timeout",
        safe.format_number(step.request_timeout),
        "--repeat",
        str(step.repeat),
        "--num-predict",
        str(step.output_tokens),
        "--fail-on-case",
        "--output",
        str(output),
    ]
    if runtime_backend == "llama":
        command.extend(
            (
                "--chat-template-thinking",
                "off" if step.thinking == "none" else "on",
            )
        )
    if step.repeat > 1:
        command.extend(("--seed-step", "1"))
    if step.cases:
        command.extend(("--cases", ",".join(step.cases)))
    if step.context_target is not None:
        command.extend(("--context-target", str(step.context_target)))
    return command


def step_process_timeout(step: Step) -> float:
    case_count = len(step.cases) if step.cases else len(KNOWN_CASES[step.suite]) or 1
    return case_count * step.repeat * step.request_timeout + 90.0


def run_benchmark(command: Sequence[str], step: Step) -> tuple[int, str | None]:
    try:
        completed = subprocess.run(
            list(command),
            stdin=subprocess.DEVNULL,
            timeout=step_process_timeout(step),
            check=False,
        )
    except subprocess.TimeoutExpired:
        return 124, "benchmark process exceeded its bounded total timeout"
    except OSError as exc:
        raise SessionError("benchmark process could not be launched") from exc
    return completed.returncode, None


def initialize_control_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = b'{"stop":false}\n'
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        if os.write(descriptor, payload) != len(payload):
            raise OSError("short control-file write")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def read_control_file(path: Path) -> bool:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        details = os.fstat(descriptor)
        if not stat.S_ISREG(details.st_mode) or details.st_size > CONTROL_MAX_BYTES:
            raise SessionError("control file must be a regular file no larger than 4096 bytes")
        raw = os.read(descriptor, CONTROL_MAX_BYTES + 1)
    finally:
        os.close(descriptor)
    try:
        value = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise SessionError("control file must contain valid JSON") from exc
    if not isinstance(value, dict) or set(value) != {"stop"} or not isinstance(value["stop"], bool):
        raise SessionError('control file must contain exactly {"stop": true|false}')
    return value["stop"]


def hold_until_stopped(
    handle: safe.ServerHandle,
    control_file: Path,
    events: safe.EventLog,
) -> float:
    started = time.monotonic()
    invalid_since: float | None = None
    events.write("holding_started", control_file=str(control_file), poll_seconds=CONTROL_POLL_SECONDS)
    while True:
        now = time.monotonic()
        if handle.process.poll() is not None:
            raise SessionError(
                f"owned server exited while holding with status {handle.process.returncode}"
            )
        try:
            stop = read_control_file(control_file)
        except (OSError, SessionError) as exc:
            if invalid_since is None:
                invalid_since = now
                events.write(
                    "control_file_invalid",
                    error_type=type(exc).__name__,
                    error=str(exc),
                )
            if now - invalid_since >= CONTROL_INVALID_GRACE_SECONDS:
                raise SessionError("control file remained invalid for 30 seconds") from exc
        else:
            invalid_since = None
            if stop:
                elapsed = round(time.monotonic() - started, 3)
                events.write("holding_stopped", reason="control_file", hold_seconds=elapsed)
                return elapsed
        time.sleep(CONTROL_POLL_SECONDS)


def verify_server(
    handle: safe.ServerHandle,
    model_name: str,
    runtime_backend: str = "strata",
) -> dict[str, Any]:
    if handle.process.poll() is not None:
        raise SessionError(f"{runtime_backend} server exited with status {handle.process.returncode}")
    try:
        if runtime_backend == "strata":
            return safe.wait_for_server(handle, model_name, timeout=POST_STEP_HEALTH_TIMEOUT)
        return llama.wait_for_server(handle, model_name, timeout=POST_STEP_HEALTH_TIMEOUT)
    except safe.TrialError as exc:
        raise SessionError(f"{runtime_backend} server health was lost: {exc}") from exc


def _config_sha256(config: safe.TrialConfig | llama.TrialConfig) -> str:
    return hashlib.sha256(config.source_path.read_bytes()).hexdigest()


def _runtime_backend(args: argparse.Namespace) -> str:
    runtime_backend = getattr(args, "runtime_backend", "strata")
    if runtime_backend not in RUNTIME_BACKENDS:
        raise SessionError(f"unsupported runtime backend {runtime_backend!r}")
    return runtime_backend


def _server_command(
    config: safe.TrialConfig | llama.TrialConfig,
    args: argparse.Namespace,
    runtime_backend: str,
) -> list[str]:
    if runtime_backend == "strata":
        return safe.server_command(config.private_path, args.cpus)
    # Keep llama.cpp's reasoning response parser enabled for thinking steps.
    # Every benchmark request still explicitly selects template thinking; a
    # `none` reasoning_effort overrides this server default and disables it.
    return llama.server_command(config, "medium")


def _server_environment(
    config: safe.TrialConfig | llama.TrialConfig,
    runtime_backend: str,
) -> dict[str, str]:
    if runtime_backend == "strata":
        return safe.server_environment()
    return llama.server_environment(config)


def _start_server(
    config: safe.TrialConfig | llama.TrialConfig,
    command: Sequence[str],
    environment: Mapping[str, str],
    log_path: Path,
    runtime_backend: str,
) -> safe.ServerHandle:
    if runtime_backend == "strata":
        return safe.start_server(command, environment, log_path)
    return llama.start_server(config, command, environment, log_path)


def _wait_for_server(
    handle: safe.ServerHandle,
    model_name: str,
    timeout: float,
    runtime_backend: str,
) -> dict[str, Any]:
    if runtime_backend == "strata":
        return safe.wait_for_server(handle, model_name, timeout=timeout)
    return llama.wait_for_server(handle, model_name, timeout=timeout)


def _server_report(
    config: safe.TrialConfig | llama.TrialConfig,
    args: argparse.Namespace,
    runtime_backend: str,
    load_seconds: float | None,
    hold_seconds: float | None,
    cleanup: dict[str, Any] | None,
) -> dict[str, Any]:
    if runtime_backend == "strata":
        executable = config.engine_path
        tokenizer: str | None = str(config.tokenizer_path)
        artifact_manifest: str | None = None
        shard_sha256: list[str] = []
    else:
        executable = config.server_path
        tokenizer = None
        artifact_manifest = str(config.manifest_path)
        shard_sha256 = [shard.sha256 for shard in config.shards]
    return {
        "runtime_backend": runtime_backend,
        "engine": str(executable),
        "model": str(config.model_path),
        "tokenizer": tokenizer,
        "model_name": config.model_name,
        "context": config.context,
        "startup_timeout": args.startup_timeout,
        "initial_load_seconds": load_seconds,
        "per_request_reasoning_effort": True,
        "per_request_chat_template_thinking": runtime_backend == "llama",
        "artifact_manifest": artifact_manifest,
        "shard_sha256": shard_sha256,
        "hold_after_plan": args.hold_after_plan,
        "control_file": str(args.control_file) if args.control_file is not None else None,
        "hold_seconds": hold_seconds,
        "cleanup": cleanup,
    }


def run_session(
    config: safe.TrialConfig | llama.TrialConfig,
    plan: Plan,
    args: argparse.Namespace,
    events: safe.EventLog,
    *,
    results: Path = RESULTS,
) -> tuple[dict[str, Any], int]:
    runtime_backend = _runtime_backend(args)
    server_argv = _server_command(config, args, runtime_backend)
    handle: safe.ServerHandle | None = None
    code = 0
    status = "passed"
    step_records: list[dict[str, Any]] = []
    load_seconds: float | None = None
    cleanup: dict[str, Any] | None = None
    hold_seconds: float | None = None
    problem: dict[str, str] | None = None
    started = safe.utc_now()
    try:
        load_started = time.monotonic()
        with safe.defer_termination_signals():
            handle = _start_server(
                config,
                server_argv,
                _server_environment(config, runtime_backend),
                results / f"{args.tag}-server.log",
                runtime_backend,
            )
        events.write(
            "server_started",
            runtime_backend=runtime_backend,
            pid=handle.process.pid,
            pgid=handle.pgid,
            argv=server_argv,
        )
        models = _wait_for_server(
            handle,
            config.model_name,
            args.startup_timeout,
            runtime_backend,
        )
        load_seconds = round(time.monotonic() - load_started, 3)
        model_ids = [
            item.get("id")
            for item in models.get("data", [])
            if isinstance(item, dict) and isinstance(item.get("id"), str)
        ]
        events.write(
            "server_ready",
            runtime_backend=runtime_backend,
            model=config.model_name,
            model_ids=model_ids,
            initial_load_seconds=load_seconds,
        )

        for index, step in enumerate(plan.steps, start=1):
            verify_server(handle, config.model_name, runtime_backend)
            output = results / f"{args.tag}-{index:02d}-{step.step_id}.jsonl"
            command = bench_command(config, step, output, runtime_backend)
            controls = step.controls(runtime_backend)
            process_timeout = step_process_timeout(step)
            events.write(
                "step_started",
                index=index,
                step_id=step.step_id,
                controls=controls,
                process_timeout=process_timeout,
                result=str(output),
                argv=command,
            )
            step_started = time.monotonic()
            returncode, step_error = run_benchmark(command, step)
            elapsed = round(time.monotonic() - step_started, 3)
            record = {
                "index": index,
                "id": step.step_id,
                "controls": controls,
                "process_timeout": process_timeout,
                "result": str(output),
                "returncode": returncode,
                "passed": returncode == 0,
                "elapsed_seconds": elapsed,
                "error": step_error,
            }
            step_records.append(record)
            events.write("step_complete", **record)
            if returncode != 0:
                code = 1
                status = "completed_with_step_failures"
            # A semantic failure or truncation returns nonzero but does not end the
            # session.  Loss of the owned server is an infrastructure failure.
            verify_server(handle, config.model_name, runtime_backend)
        if args.hold_after_plan:
            hold_seconds = hold_until_stopped(
                handle,
                args.control_file,
                events,
            )
    except KeyboardInterrupt as exc:
        code = 130
        status = "interrupted"
        problem = {"error_type": type(exc).__name__, "error": str(exc)}
        events.write("session_interrupted", **problem)
    except (SessionError, safe.TrialError, OSError, subprocess.SubprocessError) as exc:
        code = 1
        status = "infrastructure_failed"
        problem = {"error_type": type(exc).__name__, "error": str(exc)}
        events.write("session_failed", **problem)
    finally:
        if handle is not None:
            try:
                with safe.defer_termination_signals():
                    cleanup = safe.stop_server(handle)
                events.write("server_stopped", **cleanup)
            except (safe.TrialError, OSError) as exc:
                cleanup = {"stopped": False, "error_type": type(exc).__name__, "error": str(exc)}
                events.write("server_cleanup_failed", **cleanup)
                code = 1
                status = "cleanup_failed"

    report = {
        "schema_version": 1,
        "session_id": events.run_id,
        "tag": args.tag,
        "runtime_backend": runtime_backend,
        "status": status,
        "started_at": started,
        "finished_at": safe.utc_now(),
        "source_config": str(config.source_path),
        "source_config_sha256": _config_sha256(config),
        "private_config": (
            str(config.private_path) if runtime_backend == "strata" else None
        ),
        "plan": {
            "path": str(plan.source_path),
            "sha256": plan.sha256,
            "name": plan.name,
            "step_count": len(plan.steps),
        },
        "server": _server_report(
            config,
            args,
            runtime_backend,
            load_seconds,
            hold_seconds,
            cleanup,
        ),
        "steps": step_records,
        "problem": problem,
    }
    return report, code


def write_report(path: Path, report: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        view = memoryview(payload)
        written = 0
        while written < len(view):
            count = os.write(descriptor, view[written:])
            if count <= 0:
                raise OSError("short session-report write")
            written += count
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runtime-backend",
        choices=RUNTIME_BACKENDS,
        default="strata",
        help="persistent server implementation (default: strata)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="runtime config (defaults to the selected backend's 122/Strata config)",
    )
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--tag", required=True, help="safe, unique result filename prefix")
    parser.add_argument(
        "--startup-timeout",
        type=safe.startup_timeout_value,
        default=safe.STARTUP_TIMEOUT,
        help="server readiness timeout in seconds (30-480)",
    )
    parser.add_argument("--pool-workers", type=safe.positive_int)
    parser.add_argument("--cpus", type=safe.validate_cpu_list)
    parser.add_argument("--hold-after-plan", action="store_true")
    parser.add_argument("--control-file", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if safe.TAG_RE.fullmatch(args.tag) is None:
        parser.error("--tag must contain only letters, digits, dot, underscore, or hyphen")
    if args.cpus is not None and not sys.platform.startswith("linux"):
        parser.error("--cpus requires Linux taskset")
    if args.hold_after_plan != (args.control_file is not None):
        parser.error("--hold-after-plan and --control-file must be used together")
    if args.runtime_backend == "llama" and (args.pool_workers is not None or args.cpus is not None):
        parser.error("--pool-workers and --cpus are only supported by the Strata backend")

    try:
        plan = load_plan(args.plan)
        control_file = (
            validate_control_path(args.control_file) if args.control_file is not None else None
        )
        args.control_file = control_file
        config_path = args.config or (
            safe.DEFAULT_CONFIG if args.runtime_backend == "strata" else llama.DEFAULT_CONFIG
        )
        if args.runtime_backend == "strata":
            config = safe.load_trial_config(
                config_path,
                args.tag,
                "none",
                args.pool_workers,
            )
            safe.server_command(config.private_path, args.cpus)
            gpu_processes = safe.assert_preflight_clear()
        else:
            config = llama.load_trial_config(config_path)
            llama.server_command(config, "medium")
            gpu_processes = llama.assert_preflight_clear()
        validate_plan_context(plan, config.context)
        assert_results_unused(
            args.tag,
            plan,
            control_file=control_file,
            runtime_backend=args.runtime_backend,
        )
        if args.runtime_backend == "strata":
            safe.write_private_config(config)
        if control_file is not None:
            initialize_control_file(control_file)
    except (SessionError, safe.TrialError, OSError) as exc:
        print(f"model_session: {exc}", file=sys.stderr, flush=True)
        return 2

    run_id = str(uuid.uuid4())
    events = safe.EventLog(RESULTS / f"{args.tag}-runtime-events.jsonl", run_id, args.tag)
    safe.install_signal_handlers()
    if args.runtime_backend == "strata":
        runtime_metadata = {
            "private_config": str(config.private_path),
            "engine": str(config.engine_path),
            "tokenizer": str(config.tokenizer_path),
            "engine_environment": safe.engine_environment_metadata(),
            "artifact_manifest": None,
            "shard_sha256": [],
        }
    else:
        runtime_metadata = {
            "private_config": None,
            "engine": str(config.server_path),
            "tokenizer": None,
            "engine_environment": {
                "library_dirs": [str(path) for path in config.lib_dirs],
                "GGML_BACKEND_PATH": str(llama.cuda_backend_path(config)),
            },
            "artifact_manifest": str(config.manifest_path),
            "shard_sha256": [shard.sha256 for shard in config.shards],
        }
    events.write(
        "session_started",
        runtime_backend=args.runtime_backend,
        source_config=str(config.source_path),
        plan=str(plan.source_path),
        plan_sha256=plan.sha256,
        step_count=len(plan.steps),
        model=str(config.model_path),
        model_name=config.model_name,
        context=config.context,
        startup_timeout=args.startup_timeout,
        hold_after_plan=args.hold_after_plan,
        control_file=str(args.control_file) if args.control_file is not None else None,
        gpu_processes_before=gpu_processes,
        **runtime_metadata,
    )
    report, code = run_session(config, plan, args, events)
    report_path = RESULTS / f"{args.tag}-session-report.json"
    try:
        write_report(report_path, report)
    except OSError as exc:
        events.write("report_write_failed", error_type=type(exc).__name__, error=str(exc))
        return 1
    events.write("session_complete", status=report["status"], returncode=code, report=str(report_path))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
