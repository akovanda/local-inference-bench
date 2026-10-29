#!/usr/bin/env python3
"""Run one bounded specialist study while keeping one owned model loaded.

This orchestrator starts the existing GPU window around ``model_session.py``,
waits for that exact session to enter its hold state, then runs the frozen
specialist code and prose groups serially.  It never starts a server directly
and never sends SIGKILL.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import signal
import socket
import stat
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__:
    from . import gpu_window
    from . import model_session
    from . import sampler_recipe_probe
    from . import strata_trial as safe
else:
    import gpu_window
    import model_session
    import sampler_recipe_probe
    import strata_trial as safe


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
GPU_WINDOW = ROOT / "runtime" / "gpu_window.py"
MODEL_SESSION = ROOT / "runtime" / "model_session.py"
SPECIALIST_BENCH = ROOT / "runtime" / "specialist_bench.py"
SAMPLER_RECIPE_PROBE = ROOT / "runtime" / "sampler_recipe_probe.py"
ENDPOINTS = {
    "llama": "http://127.0.0.1:18081/v1",
    "strata": "http://127.0.0.1:18080/v1",
}
PORTS = {"llama": 18081, "strata": 18080}
DEFAULT_MAX_WINDOW_SECONDS = 7_200.0
MAX_WINDOW_SECONDS = 86_400.0
MAX_OUTPUT_CAP = 65_536
MAX_REQUEST_TIMEOUT = 21_600.0
MAX_IDLE_WAIT_SECONDS = 900.0
RESTORATION_WAIT_SECONDS = 180.0
PROCESS_TERM_SECONDS = 20.0
EVENT_MAX_BYTES = 32 * 1024 * 1024
MANIFEST_MAX_BYTES = 4 * 1024 * 1024
SAMPLER_PLAN_MAX_BYTES = 1024 * 1024
MAX_SAMPLER_MODELS = 32
MAX_SAMPLER_STAGES_PER_MODEL = 3
TAG_RE = safe.TAG_RE
PROFILE_RE = safe.TAG_RE
SAMPLER_SUFFIX_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,47}$")


class StudyError(RuntimeError):
    """The owned study could not safely complete."""


class StudyInterrupted(BaseException):
    def __init__(self, signum: int) -> None:
        super().__init__(f"signal {signum}")
        self.signum = signum


@dataclass(frozen=True)
class HoldingSession:
    run_id: str
    runtime_backend: str
    model_name: str
    context: int
    initial_load_seconds: float
    session_started_at: str
    server_ready_at: str
    holding_started_at: str


@dataclass(frozen=True)
class SamplerStage:
    model: str
    group: str
    recipe_path: Path
    recipe_sha256: str
    recipe_label: str
    suffix: str
    prose_thinking: str | None
    code_thinking: str | None


@dataclass(frozen=True)
class SamplerPlan:
    source_path: Path
    sha256: str
    models: Mapping[str, tuple[SamplerStage, ...]]


@dataclass
class SignalLatch:
    received: int | None = None

    def handle(self, signum: int, _frame: Any) -> None:
        self.received = signum

    def check(self) -> None:
        if self.received is not None:
            raise StudyInterrupted(self.received)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def format_number(value: float) -> str:
    return format(value, ".15g")


def progress(event: str, **fields: Any) -> dict[str, Any]:
    row = {"at": utc_now(), "event": event, **fields}
    print(json.dumps(row, separators=(",", ":")), flush=True)
    return row


def _write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        view = memoryview(payload)
        offset = 0
        while offset < len(view):
            written = os.write(descriptor, view[offset:])
            if written <= 0:
                raise OSError("short study receipt write")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _read_json(path: Path, description: str) -> dict[str, Any]:
    try:
        details = path.lstat()
        if not stat.S_ISREG(details.st_mode) or path.is_symlink():
            raise StudyError(f"{description} must be a regular non-symlink file")
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StudyError(f"{description} is unreadable or invalid: {path}") from exc
    if not isinstance(value, dict):
        raise StudyError(f"{description} must be a JSON object")
    return value


def _regular_path(path: Path, description: str, *, max_bytes: int | None = None) -> Path:
    candidate = path.expanduser()
    try:
        details = candidate.lstat()
    except OSError as exc:
        raise StudyError(f"{description} is missing or inaccessible: {candidate}") from exc
    if not stat.S_ISREG(details.st_mode) or candidate.is_symlink():
        raise StudyError(f"{description} must be a regular non-symlink file: {candidate}")
    if max_bytes is not None and details.st_size > max_bytes:
        raise StudyError(f"{description} exceeds its {max_bytes}-byte bound: {candidate}")
    return candidate.resolve()


def load_sampler_plan(path: Path) -> SamplerPlan:
    source_path = _regular_path(path, "sampler plan", max_bytes=SAMPLER_PLAN_MAX_BYTES)
    document = _read_json(source_path, "sampler plan")
    unknown = sorted(set(document) - {"schema_version", "models"})
    if unknown:
        raise StudyError("unknown sampler plan fields: " + ", ".join(unknown))
    if document.get("schema_version") != 1:
        raise StudyError("sampler plan schema_version must be 1")
    models = document.get("models")
    if not isinstance(models, dict) or not models:
        raise StudyError("sampler plan models must be a non-empty object")
    if len(models) > MAX_SAMPLER_MODELS:
        raise StudyError(f"sampler plan may contain at most {MAX_SAMPLER_MODELS} models")

    normalized: dict[str, tuple[SamplerStage, ...]] = {}
    suffixes: set[str] = set()
    for model, raw_stages in models.items():
        if not isinstance(model, str) or TAG_RE.fullmatch(model) is None:
            raise StudyError(f"sampler plan model alias is unsafe: {model!r}")
        if (
            not isinstance(raw_stages, list)
            or not 1 <= len(raw_stages) <= MAX_SAMPLER_STAGES_PER_MODEL
        ):
            raise StudyError(
                f"sampler plan model {model!r} must have one to three stages"
            )
        stages: list[SamplerStage] = []
        for index, raw_stage in enumerate(raw_stages, start=1):
            if not isinstance(raw_stage, dict):
                raise StudyError(f"sampler stage {model}[{index}] must be an object")
            stage_unknown = sorted(
                set(raw_stage)
                - {"group", "recipe", "suffix", "prose_thinking", "code_thinking"}
            )
            if stage_unknown:
                raise StudyError(
                    f"unknown sampler stage fields for {model}[{index}]: "
                    + ", ".join(stage_unknown)
                )
            group = raw_stage.get("group")
            if group not in ("code", "prose"):
                raise StudyError(f"sampler stage {model}[{index}] group must be code or prose")
            suffix = raw_stage.get("suffix")
            if not isinstance(suffix, str) or SAMPLER_SUFFIX_RE.fullmatch(suffix) is None:
                raise StudyError(
                    f"sampler stage {model}[{index}] suffix must be 1-48 lowercase "
                    "letters, digits, or hyphens"
                )
            if suffix in suffixes:
                raise StudyError(f"sampler plan repeats suffix {suffix!r}")
            suffixes.add(suffix)
            prose_thinking = raw_stage.get("prose_thinking")
            if prose_thinking is not None and prose_thinking not in ("none", "low", "medium"):
                raise StudyError(
                    f"sampler stage {model}[{index}] prose_thinking must be none, low, or medium"
                )
            if group != "prose" and prose_thinking is not None:
                raise StudyError(
                    f"sampler stage {model}[{index}] prose_thinking is only valid for prose"
                )
            code_thinking = raw_stage.get("code_thinking")
            if code_thinking is not None and code_thinking not in ("none", "medium"):
                raise StudyError(
                    f"sampler stage {model}[{index}] code_thinking must be none or medium"
                )
            if group != "code" and code_thinking is not None:
                raise StudyError(
                    f"sampler stage {model}[{index}] code_thinking is only valid for code"
                )
            recipe_value = raw_stage.get("recipe")
            if (
                not isinstance(recipe_value, str)
                or not recipe_value
                or "\x00" in recipe_value
            ):
                raise StudyError(f"sampler stage {model}[{index}] recipe must be a path string")
            recipe_path = Path(recipe_value).expanduser()
            if not recipe_path.is_absolute():
                recipe_path = source_path.parent / recipe_path
            recipe_path = _regular_path(recipe_path, f"sampler recipe for {model}[{index}]")
            try:
                recipe, recipe_sha256 = sampler_recipe_probe.load_recipe(recipe_path)
            except sampler_recipe_probe.ProbeError as exc:
                raise StudyError(
                    f"sampler recipe for {model}[{index}] is invalid: {exc}"
                ) from exc
            stages.append(
                SamplerStage(
                    model=model,
                    group=group,
                    recipe_path=recipe_path,
                    recipe_sha256=recipe_sha256,
                    recipe_label=recipe["label"],
                    suffix=suffix,
                    prose_thinking=prose_thinking,
                    code_thinking=code_thinking,
                )
            )
        normalized[model] = tuple(stages)
    return SamplerPlan(
        source_path=source_path,
        sha256=sha256_file(source_path),
        models=normalized,
    )


def sampler_runtime_selection(
    sampler_plan: SamplerPlan,
    *,
    config_path: Path,
    runtime_backend: str,
) -> tuple[str, tuple[SamplerStage, ...]]:
    source = _read_json(config_path, "source config")
    model_name = source.get("model_name")
    if (
        not isinstance(model_name, str)
        or TAG_RE.fullmatch(model_name) is None
        or any(character.isspace() for character in model_name)
    ):
        raise StudyError("source config model_name is not a safe exact model alias")
    stages = sampler_plan.models.get(model_name, ())
    if not stages:
        return model_name, ()
    if runtime_backend != "llama":
        raise StudyError(
            f"sampler plan matches {model_name!r}, but extra sampler stages require native llama"
        )
    context = source.get("context")
    if (
        isinstance(context, bool)
        or not isinstance(context, int)
        or not 1 <= context <= sampler_recipe_probe.MAX_CONTEXT_TOKENS
    ):
        raise StudyError(
            f"sampler plan matches {model_name!r}, but source config context must be "
            f"from 1 through {sampler_recipe_probe.MAX_CONTEXT_TOKENS}"
        )
    return model_name, stages


def validate_note(value: str) -> str:
    if not value.strip() or len(value) > 1_000 or "\x00" in value:
        raise argparse.ArgumentTypeError("must be 1-1000 non-NUL characters")
    return value.strip()


def bounded_positive_int(value: str, maximum: int = 8_192) -> int:
    parsed = int(value)
    if not 1 <= parsed <= maximum:
        raise argparse.ArgumentTypeError(f"must be from 1 through {maximum}")
    return parsed


def bounded_seconds(value: str, maximum: float) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or not 0 < parsed <= maximum:
        raise argparse.ArgumentTypeError(f"must be finite and from >0 through {maximum:g}")
    return parsed


def _artifact_paths(
    tag: str,
    plan: model_session.Plan,
    runtime_backend: str,
    results: Path,
) -> dict[str, Path]:
    control = results / f"{tag}-control.json"
    paths = {
        "control": control,
        "events": results / f"{tag}-runtime-events.jsonl",
        "session_report": results / f"{tag}-session-report.json",
        "study_receipt": results / f"{tag}-study-trial.json",
        "code": results / f"{tag}-specialist-code.jsonl",
        "prose": results / f"{tag}-specialist-prose.jsonl",
        "code_texts": Path(str(results / f"{tag}-specialist-code.jsonl") + ".texts"),
        "prose_texts": Path(str(results / f"{tag}-specialist-prose.jsonl") + ".texts"),
        "server_log": results / f"{tag}-server.log",
        **{
            f"plan_{index:02d}_{step.step_id}": results / f"{tag}-{index:02d}-{step.step_id}.jsonl"
            for index, step in enumerate(plan.steps, start=1)
        },
    }
    if runtime_backend == "strata":
        paths.update({
            "private_config": results / f"{tag}-config.json",
            "shared_settings": results / f"{tag}-config.shared-settings.json",
            "engine_log": results / f"{tag}-engine.log",
        })
    return paths


def _sampler_artifact_paths(
    tag: str,
    stages: Sequence[SamplerStage],
    results: Path,
) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for index, stage in enumerate(stages, start=1):
        prefix = results / f"{tag}-{stage.suffix}"
        for artifact, path in sampler_recipe_probe._artifact_paths(prefix).items():
            paths[f"sampler_{index:02d}_{artifact}"] = path
    return paths


def assert_outputs_unused(
    tag: str,
    plan: model_session.Plan,
    runtime_backend: str,
    results: Path = RESULTS,
    sampler_stages: Sequence[SamplerStage] = (),
) -> dict[str, Path]:
    paths = _artifact_paths(tag, plan, runtime_backend, results)
    paths.update(_sampler_artifact_paths(tag, sampler_stages, results))
    control = paths["control"]
    model_session.assert_results_unused(
        tag,
        plan,
        results,
        control,
        runtime_backend=runtime_backend,
    )
    extras = [
        paths["study_receipt"],
        paths["code"],
        paths["prose"],
        paths["code_texts"],
        paths["prose_texts"],
        *(path for name, path in paths.items() if name.startswith("sampler_")),
    ]
    collisions = [path for path in extras if os.path.lexists(path)]
    if collisions:
        raise StudyError(
            "study tag would overwrite existing output: "
            + ", ".join(str(path) for path in collisions)
        )
    return paths


def load_specialist_manifest(
    helper: Path = SPECIALIST_BENCH,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not helper.is_file():
        raise StudyError(f"specialist benchmark helper is missing: {helper}")
    argv = [sys.executable, str(helper), "--print-manifest"]
    started = utc_now()
    try:
        completed = subprocess.run(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30.0,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise StudyError("specialist manifest command could not complete") from exc
    record = {
        "kind": "specialist_manifest",
        "argv": argv,
        "started_at": started,
        "finished_at": utc_now(),
        "returncode": completed.returncode,
    }
    if completed.returncode != 0:
        raise StudyError(f"specialist manifest command exited {completed.returncode}")
    if len(completed.stdout) > MANIFEST_MAX_BYTES:
        raise StudyError("specialist manifest exceeds the 4 MiB metadata bound")
    try:
        manifest = json.loads(completed.stdout)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise StudyError("specialist manifest command returned invalid JSON") from exc
    if not isinstance(manifest, dict):
        raise StudyError("specialist manifest must be a JSON object")
    record["stdout_sha256"] = hashlib.sha256(completed.stdout).hexdigest()
    return manifest, record


def gpu_window_command(args: argparse.Namespace, control: Path) -> list[str]:
    return [
        sys.executable,
        str(GPU_WINDOW),
        "--long-session",
        "--wait-seconds",
        format_number(args.wait_for_idle),
        "--max-seconds",
        format_number(args.max_seconds),
        "--",
        str(MODEL_SESSION),
        "--runtime-backend",
        args.runtime_backend,
        "--config",
        str(args.config),
        "--plan",
        str(args.plan),
        "--tag",
        args.tag,
        "--startup-timeout",
        format_number(args.startup_timeout),
        "--hold-after-plan",
        "--control-file",
        str(control),
    ]


def specialist_command(
    args: argparse.Namespace,
    session: HoldingSession,
    group: str,
    output: Path,
) -> list[str]:
    if group not in ("code", "prose"):
        raise StudyError(f"unsupported specialist group {group!r}")
    thinking = args.code_thinking if group == "code" else args.prose_thinking
    cap = args.code_output_cap if group == "code" else args.prose_output_cap
    timeout = args.code_timeout if group == "code" else args.prose_timeout
    temperature = 0.0 if group == "code" else 0.7
    profile = args.profile or args.tag
    argv = [
        sys.executable,
        str(SPECIALIST_BENCH),
        "--endpoint",
        ENDPOINTS[args.runtime_backend],
        "--model",
        session.model_name,
        "--context",
        str(session.context),
        "--tag",
        f"{args.tag}-{group}",
        "--profile",
        profile,
        "--jsonl",
        str(output),
        "--groups",
        group,
        "--thinking",
        thinking,
        "--output-cap",
        str(cap),
        "--timeout",
        format_number(timeout),
        "--temperature",
        format_number(temperature),
        "--seed",
        "0",
    ]
    if args.runtime_backend == "llama":
        argv.extend(("--chat-template-thinking", "off" if thinking == "none" else "on"))
    if args.reasoning_cap is not None:
        argv.extend(("--reasoning-budget-tokens", str(args.reasoning_cap)))
    if args.capability_note is not None:
        argv.extend(("--capability-note", args.capability_note))
    return argv


def sampler_command(
    args: argparse.Namespace,
    session: HoldingSession,
    stage: SamplerStage,
    output_prefix: Path,
    runtime_events: Path,
) -> list[str]:
    cap = args.code_output_cap if stage.group == "code" else args.prose_output_cap
    timeout = args.code_timeout if stage.group == "code" else args.prose_timeout
    prose_thinking = stage.prose_thinking or args.prose_thinking
    command = [
        sys.executable,
        str(SAMPLER_RECIPE_PROBE),
        "--recipe",
        str(stage.recipe_path),
        "--runtime-events",
        str(runtime_events),
        "--tag",
        args.tag,
        "--model-session-run-id",
        session.run_id,
        "--endpoint",
        ENDPOINTS[args.runtime_backend],
        "--model",
        session.model_name,
        "--expected-context",
        str(session.context),
        "--output-prefix",
        str(output_prefix),
        "--profile",
        args.profile or args.tag,
        "--groups",
        stage.group,
        "--prose-thinking",
        prose_thinking,
        "--output-cap",
        str(cap),
        "--timeout",
        format_number(timeout),
    ]
    if stage.code_thinking is not None:
        command.extend(("--code-thinking", stage.code_thinking))
    return command


def sampler_receipt_summary(
    path: Path,
    *,
    args: argparse.Namespace,
    session: HoldingSession,
    stage: SamplerStage,
    returncode: int,
) -> dict[str, Any]:
    receipt = _read_json(path, f"sampler stage receipt for {stage.suffix}")
    recipe = receipt.get("recipe")
    controls = receipt.get("controls")
    status_value = receipt.get("status")
    request_failures = receipt.get("request_failure_count")
    if (
        receipt.get("tag") != args.tag
        or receipt.get("model") != session.model_name
        or receipt.get("model_session_run_id") != session.run_id
        or receipt.get("runtime_backend") != "llama"
        or not isinstance(recipe, dict)
        or recipe.get("path") != str(stage.recipe_path)
        or recipe.get("sha256") != stage.recipe_sha256
        or status_value not in (
            "complete",
            "completed_with_request_failures",
            "interrupted",
        )
        or not isinstance(request_failures, int)
        or isinstance(request_failures, bool)
        or request_failures < 0
        or not isinstance(controls, dict)
        or controls.get("expected_context_tokens") != session.context
        or controls.get("actual_context_tokens") != session.context
    ):
        raise StudyError(f"sampler stage {stage.suffix!r} receipt identity is invalid")
    if "problem" in receipt:
        raise StudyError(f"sampler stage {stage.suffix!r} receipt has an unexpected problem field")
    if stage.code_thinking == "none" and (
        not isinstance(controls, dict)
        or controls.get("ledger_thinking") != "medium"
        or controls.get("code_thinking") != "none"
    ):
        raise StudyError(f"sampler stage {stage.suffix!r} receipt lacks code-none controls")
    if stage.code_thinking == "medium" and (
        not isinstance(controls, dict)
        or controls.get("ledger_and_code_thinking") != "medium"
    ):
        raise StudyError(f"sampler stage {stage.suffix!r} receipt lacks code-medium controls")
    if status_value == "complete" and (returncode != 0 or request_failures != 0):
        raise StudyError(f"sampler stage {stage.suffix!r} clean receipt disagrees with its exit")
    if status_value == "completed_with_request_failures" and (
        returncode != 1 or request_failures == 0
    ):
        raise StudyError(f"sampler stage {stage.suffix!r} failure receipt disagrees with its exit")
    summary = {
        "path": str(path),
        "sha256": sha256_file(path),
        "status": status_value,
        "request_failure_count": request_failures,
        "attempted_count": receipt.get("attempted_count"),
        "completed_count": receipt.get("completed_count"),
        "truncated_count": receipt.get("truncated_count"),
    }
    return summary


def _read_event_rows(path: Path) -> list[dict[str, Any]]:
    try:
        details = path.lstat()
        if not stat.S_ISREG(details.st_mode) or path.is_symlink():
            raise StudyError("session event path is not a regular file")
        if details.st_size > EVENT_MAX_BYTES:
            raise StudyError("session event file exceeds its 32 MiB bound")
        raw = path.read_bytes()
    except FileNotFoundError:
        return []
    except OSError as exc:
        raise StudyError("session event file is unreadable") from exc
    rows: list[dict[str, Any]] = []
    for line in raw.splitlines():
        try:
            row = json.loads(line)
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise StudyError("session event file contains invalid JSON") from exc
        if not isinstance(row, dict):
            raise StudyError("session event row must be a JSON object")
        rows.append(row)
    return rows


def _holding_identity(
    rows: Sequence[Mapping[str, Any]],
    *,
    tag: str,
    source_config: Path,
    plan: Path,
    control: Path,
    runtime_backend: str,
) -> HoldingSession | None:
    if not rows:
        return None
    if any(row.get("tag") != tag for row in rows):
        raise StudyError("session event file contains a different tag")
    starts = [row for row in rows if row.get("event") == "session_started"]
    if len(starts) != 1:
        if not starts:
            return None
        raise StudyError("session event file contains multiple session_started rows")
    start = starts[0]
    run_id = start.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise StudyError("session_started lacks an owned run ID")
    if any(row.get("run_id") != run_id for row in rows):
        raise StudyError("session event file mixes multiple run IDs")
    if (
        Path(str(start.get("source_config"))).resolve() != source_config.resolve()
        or Path(str(start.get("plan"))).resolve() != plan.resolve()
        or start.get("runtime_backend") != runtime_backend
        or start.get("control_file") != str(control)
    ):
        raise StudyError("session_started identity does not match this study invocation")
    ready = [row for row in rows if row.get("event") == "server_ready"]
    holding = [row for row in rows if row.get("event") == "holding_started"]
    if not holding:
        return None
    if len(ready) != 1 or len(holding) != 1:
        raise StudyError("owned session has ambiguous ready or holding lifecycle events")
    if holding[0].get("control_file") != str(control):
        raise StudyError("holding_started refers to a different control file")
    model_name = start.get("model_name")
    context = start.get("context")
    load_seconds = ready[0].get("initial_load_seconds")
    if (
        not isinstance(model_name, str)
        or not model_name
        or not isinstance(context, int)
        or isinstance(context, bool)
        or not isinstance(load_seconds, (int, float))
        or isinstance(load_seconds, bool)
        or not math.isfinite(float(load_seconds))
        or float(load_seconds) < 0
    ):
        raise StudyError("owned session lifecycle metadata is incomplete")
    return HoldingSession(
        run_id=run_id,
        runtime_backend=runtime_backend,
        model_name=model_name,
        context=context,
        initial_load_seconds=float(load_seconds),
        session_started_at=str(start.get("at")),
        server_ready_at=str(ready[0].get("at")),
        holding_started_at=str(holding[0].get("at")),
    )


def wait_for_holding(
    outer: subprocess.Popen[Any],
    events_path: Path,
    *,
    args: argparse.Namespace,
    control: Path,
    deadline: float,
    signals: SignalLatch,
) -> HoldingSession:
    while time.monotonic() < deadline:
        signals.check()
        returncode = outer.poll()
        if returncode is not None:
            raise StudyError(f"GPU window exited {returncode} before holding_started")
        identity = _holding_identity(
            _read_event_rows(events_path),
            tag=args.tag,
            source_config=args.config,
            plan=args.plan,
            control=control,
            runtime_backend=args.runtime_backend,
        )
        if identity is not None:
            return identity
        time.sleep(0.5)
    raise StudyError("owned session did not reach holding_started before the study deadline")


def _terminate_specialist(process: subprocess.Popen[Any]) -> bool:
    """Ask only the owned specialist process group to stop; never SIGKILL."""
    if process.poll() is not None:
        return True
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return True
    deadline = time.monotonic() + PROCESS_TERM_SECONDS
    while process.poll() is None and time.monotonic() < deadline:
        time.sleep(0.1)
    return process.poll() is not None


def run_specialist(
    argv: Sequence[str],
    *,
    kind: str,
    outer: subprocess.Popen[Any],
    deadline: float,
    signals: SignalLatch,
) -> dict[str, Any]:
    started_at = utc_now()
    started = time.monotonic()
    try:
        process = subprocess.Popen(
            list(argv),
            stdin=subprocess.DEVNULL,
            start_new_session=True,
            close_fds=True,
        )
    except OSError as exc:
        raise StudyError(f"{kind} specialist benchmark could not be launched") from exc
    try:
        while True:
            signals.check()
            returncode = process.poll()
            if returncode is not None:
                break
            if outer.poll() is not None:
                raise StudyError("GPU window exited while a specialist benchmark was running")
            if time.monotonic() >= deadline:
                raise StudyError("study deadline reached during specialist benchmark")
            time.sleep(0.25)
    except BaseException:
        stopped = _terminate_specialist(process)
        if not stopped:
            raise StudyError(
                f"owned {kind} specialist process did not exit after SIGTERM; SIGKILL was not sent"
            )
        raise
    return {
        "kind": kind,
        "argv": list(argv),
        "started_at": started_at,
        "finished_at": utc_now(),
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "returncode": returncode,
    }


def request_session_stop(control: Path) -> bool:
    if not os.path.lexists(control):
        return False
    before = control.lstat()
    if not stat.S_ISREG(before.st_mode) or control.is_symlink():
        raise StudyError("control file is not regular")
    flags = os.O_WRONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(control, flags)
    try:
        details = os.fstat(descriptor)
        if (
            not stat.S_ISREG(details.st_mode)
            or (details.st_dev, details.st_ino) != (before.st_dev, before.st_ino)
        ):
            raise StudyError("control file identity changed before stop request")
        os.ftruncate(descriptor, 0)
        payload = b'{"stop":true}\n'
        if os.write(descriptor, payload) != len(payload):
            raise OSError("short control-file write")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return True


def request_outer_termination(process: subprocess.Popen[Any]) -> bool:
    """Send SIGTERM only to the owned GPU-window process group."""
    if process.poll() is not None:
        return False
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return False
    return True


def wait_for_outer_exit(
    process: subprocess.Popen[Any],
    timeout: float = RESTORATION_WAIT_SECONDS,
    control: Path | None = None,
) -> tuple[int | None, bool]:
    """Wait without signalling; request control-file stop as soon as possible."""
    deadline = time.monotonic() + timeout
    stop_requested = False
    while time.monotonic() < deadline:
        returncode = process.poll()
        if returncode is not None:
            return returncode, stop_requested
        if not stop_requested and control is not None and os.path.lexists(control):
            stop_requested = request_session_stop(control)
        time.sleep(0.25)
    return process.poll(), stop_requested


def _port_clear(port: int) -> bool:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.settimeout(0.3)
        return sock.connect_ex(("127.0.0.1", port)) != 0
    finally:
        sock.close()


def restoration_evidence(runtime_backend: str) -> dict[str, Any]:
    state = gpu_window.original_state()
    return {
        "original_container": gpu_window.ORIGINAL_CONTAINER,
        "original_running": state.get("Running") is True,
        "original_paused": state.get("Paused") is True,
        "owned_loopback_port": PORTS[runtime_backend],
        "owned_loopback_port_clear": _port_clear(PORTS[runtime_backend]),
        "checked_at": utc_now(),
    }


def _artifact_inventory(paths: Mapping[str, Path]) -> dict[str, dict[str, Any]]:
    inventory: dict[str, dict[str, Any]] = {}
    for name, path in paths.items():
        try:
            details = path.lstat()
        except FileNotFoundError:
            inventory[name] = {"path": str(path), "exists": False}
            continue
        inventory[name] = {
            "path": str(path),
            "exists": True,
            "regular_file": stat.S_ISREG(details.st_mode),
            "directory": stat.S_ISDIR(details.st_mode),
            "size": details.st_size,
        }
    return inventory


def execute_study(
    args: argparse.Namespace,
    *,
    results: Path = RESULTS,
) -> tuple[dict[str, Any], int]:
    started_at = utc_now()
    started = time.monotonic()
    standard_only = bool(getattr(args, "standard_only", False))
    plan = model_session.load_plan(args.plan)
    if not args.config.is_file() or args.config.is_symlink():
        raise StudyError(f"source config is missing or not a regular file: {args.config}")
    sampler_plan: SamplerPlan | None = None
    expected_model_name: str | None = None
    sampler_stages: tuple[SamplerStage, ...] = ()
    sampler_plan_path = getattr(args, "sampler_plan", None)
    if sampler_plan_path is not None:
        sampler_plan = load_sampler_plan(sampler_plan_path)
        expected_model_name, sampler_stages = sampler_runtime_selection(
            sampler_plan,
            config_path=args.config,
            runtime_backend=args.runtime_backend,
        )
    paths = assert_outputs_unused(
        args.tag,
        plan,
        args.runtime_backend,
        results,
        sampler_stages=() if standard_only else sampler_stages,
    )
    config_sha256 = sha256_file(args.config)
    if standard_only:
        manifest = None
        manifest_process = {
            "kind": "specialist_manifest",
            "argv": None,
            "started_at": None,
            "finished_at": None,
            "returncode": None,
            "stdout_sha256": None,
            "skipped_by_standard_only": True,
        }
    else:
        manifest, manifest_process = load_specialist_manifest()
    sources = {
        str(path): sha256_file(path)
        for path in (
            (GPU_WINDOW, MODEL_SESSION)
            if standard_only
            else (GPU_WINDOW, MODEL_SESSION, SPECIALIST_BENCH)
        )
    }
    if sampler_plan is not None and not standard_only:
        if not SAMPLER_RECIPE_PROBE.is_file() or SAMPLER_RECIPE_PROBE.is_symlink():
            raise StudyError(f"sampler recipe helper is missing: {SAMPLER_RECIPE_PROBE}")
        sources[str(SAMPLER_RECIPE_PROBE)] = sha256_file(SAMPLER_RECIPE_PROBE)
    commands: list[dict[str, Any]] = [manifest_process]
    sampler_stage_records: list[dict[str, Any]] = []
    lifecycle: list[dict[str, Any]] = []
    outer: subprocess.Popen[Any] | None = None
    session: HoldingSession | None = None
    problem: dict[str, Any] | None = None
    restoration: dict[str, Any] | None = None
    session_report: dict[str, Any] | None = None
    code = 1
    signals = SignalLatch()
    previous_handlers: dict[int, Any] = {}
    cleanup = {
        "stop_requested": False,
        "outer_sigterm_sent": False,
        "outer_returncode": None,
        "outer_exited": False,
        "orchestrator_sigkill_sent": False,
        "model_session_cleanup": None,
    }
    deadline = started + args.max_seconds - RESTORATION_WAIT_SECONDS
    outer_argv = gpu_window_command(args, paths["control"])
    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, signals.handle)
        lifecycle.append(progress("study_phase", tag=args.tag, phase="window_starting"))
        outer_started_at = utc_now()
        outer_record = {
            "kind": "gpu_window",
            "argv": outer_argv,
            "started_at": outer_started_at,
            "finished_at": None,
            "returncode": None,
        }
        commands.append(outer_record)
        outer = subprocess.Popen(
            outer_argv,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
            close_fds=True,
        )
        session = wait_for_holding(
            outer,
            paths["events"],
            args=args,
            control=paths["control"],
            deadline=deadline,
            signals=signals,
        )
        lifecycle.append(
            progress(
                "study_phase",
                tag=args.tag,
                phase="holding",
                run_id=session.run_id,
                initial_load_seconds=session.initial_load_seconds,
            )
        )
        if standard_only:
            lifecycle.append(
                progress(
                    "study_phase",
                    tag=args.tag,
                    phase="standard_plan_only_complete",
                    supplemental_stages_skipped=True,
                )
            )
        for group in (() if standard_only else ("code", "prose")):
            signals.check()
            lifecycle.append(progress("study_phase", tag=args.tag, phase=f"{group}_started"))
            command = specialist_command(args, session, group, paths[group])
            command_record = {
                "kind": f"specialist_{group}",
                "argv": command,
                "started_at": None,
                "finished_at": None,
                "elapsed_seconds": None,
                "returncode": None,
            }
            commands.append(command_record)
            record = run_specialist(
                command,
                kind=f"specialist_{group}",
                outer=outer,
                deadline=deadline,
                signals=signals,
            )
            command_record.update(record)
            lifecycle.append(
                progress(
                    "study_phase",
                    tag=args.tag,
                    phase=f"{group}_complete",
                    returncode=record["returncode"],
                    elapsed_seconds=record["elapsed_seconds"],
                )
            )
            if record["returncode"] not in (0, 1):
                raise StudyError(
                    f"{group} specialist benchmark exited {record['returncode']} before a complete result set"
                )
        if sampler_stages and not standard_only:
            if session.model_name != expected_model_name:
                raise StudyError(
                    "held model alias differs from the sampler plan model selected before launch"
                )
            for index, stage in enumerate(sampler_stages, start=1):
                signals.check()
                output_prefix = results / f"{args.tag}-{stage.suffix}"
                lifecycle.append(
                    progress(
                        "study_phase",
                        tag=args.tag,
                        phase="sampler_stage_started",
                        sampler_suffix=stage.suffix,
                        sampler_group=stage.group,
                    )
                )
                command = sampler_command(
                    args,
                    session,
                    stage,
                    output_prefix,
                    paths["events"],
                )
                command_record = {
                    "kind": f"sampler_{stage.group}_{stage.suffix}",
                    "argv": command,
                    "started_at": None,
                    "finished_at": None,
                    "elapsed_seconds": None,
                    "returncode": None,
                }
                commands.append(command_record)
                stage_record: dict[str, Any] = {
                    "index": index,
                    "model": stage.model,
                    "group": stage.group,
                    "suffix": stage.suffix,
                    "recipe": {
                        "path": str(stage.recipe_path),
                        "sha256": stage.recipe_sha256,
                        "label": stage.recipe_label,
                    },
                    "prose_thinking": stage.prose_thinking,
                    "code_thinking": stage.code_thinking,
                    "command_kind": command_record["kind"],
                    "argv": command,
                    "receipt": {
                        "path": str(Path(f"{output_prefix}-sampler-receipt.json")),
                        "exists": False,
                        "status": None,
                    },
                }
                sampler_stage_records.append(stage_record)
                record = run_specialist(
                    command,
                    kind=f"sampler_{stage.group}_{stage.suffix}",
                    outer=outer,
                    deadline=deadline,
                    signals=signals,
                )
                command_record.update(record)
                stage_record["started_at"] = record["started_at"]
                stage_record["finished_at"] = record["finished_at"]
                stage_record["elapsed_seconds"] = record["elapsed_seconds"]
                stage_record["returncode"] = record["returncode"]
                receipt_path = Path(f"{output_prefix}-sampler-receipt.json")
                stage_record["receipt"]["exists"] = os.path.lexists(receipt_path)
                summary = sampler_receipt_summary(
                    receipt_path,
                    args=args,
                    session=session,
                    stage=stage,
                    returncode=record["returncode"],
                )
                stage_record["receipt"] = {"exists": True, **summary}
                lifecycle.append(
                    progress(
                        "study_phase",
                        tag=args.tag,
                        phase="sampler_stage_complete",
                        sampler_suffix=stage.suffix,
                        sampler_group=stage.group,
                        sampler_status=summary["status"],
                        returncode=record["returncode"],
                        elapsed_seconds=record["elapsed_seconds"],
                    )
                )
                if summary["status"] != "complete":
                    raise StudyError(
                        f"sampler stage {stage.suffix!r} ended with infrastructure "
                        f"status {summary['status']}"
                    )
        code = 0 if all(row.get("returncode") == 0 for row in commands[2:]) else 1
    except StudyInterrupted as exc:
        code = 128 + exc.signum
        problem = {"error_type": type(exc).__name__, "error": str(exc), "signal": exc.signum}
    except (StudyError, model_session.SessionError, OSError, subprocess.SubprocessError) as exc:
        code = 1
        problem = {"error_type": type(exc).__name__, "error": str(exc)}
    finally:
        if outer is not None:
            try:
                cleanup["stop_requested"] = request_session_stop(paths["control"])
            except (StudyError, OSError) as exc:
                cleanup["stop_error"] = {"error_type": type(exc).__name__, "error": str(exc)}
                problem = problem or cleanup["stop_error"]
                code = 1
            # Before holding, the control file cannot provide prompt cleanup:
            # model_session only reads it after every plan step.  SIGTERM the
            # owned outer group and let gpu_window forward, clean, and restore.
            # This orchestrator never follows it with SIGKILL.
            if session is None and outer.poll() is None:
                try:
                    cleanup["outer_sigterm_sent"] = request_outer_termination(outer)
                except OSError as exc:
                    cleanup["outer_sigterm_error"] = {
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                    problem = problem or cleanup["outer_sigterm_error"]
                    code = 1
            wait_seconds = RESTORATION_WAIT_SECONDS
            if not cleanup["stop_requested"] and not cleanup["outer_sigterm_sent"]:
                # Setup may fail while gpu_window is still waiting for the
                # production GPU.  Wait through its bounded lifetime and ask
                # for control-file stop if model_session later reaches hold.
                wait_seconds = max(
                    wait_seconds,
                    args.wait_for_idle + args.max_seconds + RESTORATION_WAIT_SECONDS
                    - (time.monotonic() - started),
                )
            outer_returncode, delayed_stop = wait_for_outer_exit(
                outer, timeout=wait_seconds, control=paths["control"]
            )
            cleanup["stop_requested"] = cleanup["stop_requested"] or delayed_stop
            cleanup["outer_returncode"] = outer_returncode
            cleanup["outer_exited"] = outer_returncode is not None
            commands[1]["returncode"] = outer_returncode
            commands[1]["finished_at"] = utc_now()
            if outer_returncode is None:
                problem = problem or {
                    "error_type": "StudyError",
                    "error": "GPU window did not exit within the restoration wait; no signal or SIGKILL was sent",
                }
                code = 1
            else:
                try:
                    restoration = restoration_evidence(args.runtime_backend)
                except (gpu_window.WindowError, OSError) as exc:
                    problem = problem or {"error_type": type(exc).__name__, "error": str(exc)}
                    code = 1
                else:
                    if (
                        restoration["original_running"] is not True
                        or restoration["original_paused"] is True
                        or restoration["owned_loopback_port_clear"] is not True
                    ):
                        problem = problem or {
                            "error_type": "StudyError",
                            "error": "restoration or owned-port verification failed",
                        }
                        code = 1
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)

    if paths["session_report"].is_file():
        try:
            session_report = _read_json(paths["session_report"], "model session report")
        except StudyError as exc:
            problem = problem or {"error_type": type(exc).__name__, "error": str(exc)}
            code = 1
    session_hold_seconds: float | None = None
    if isinstance(session_report, dict):
        server = session_report.get("server")
        if isinstance(server, dict):
            if isinstance(server.get("cleanup"), dict):
                cleanup["model_session_cleanup"] = server["cleanup"]
            value = server.get("hold_seconds")
            if (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value))
                and float(value) >= 0
            ):
                session_hold_seconds = float(value)
    session_cleanup = cleanup["model_session_cleanup"]
    session_cleanup_ok = (
        isinstance(session_cleanup, dict)
        and session_cleanup.get("stopped") is True
        and session_cleanup.get("kill_sent") is False
    )
    expected_outer_returncode = None
    if isinstance(session_report, dict):
        if session_report.get("status") == "passed":
            expected_outer_returncode = 0
        elif session_report.get("status") == "completed_with_step_failures":
            expected_outer_returncode = 1
    infrastructure_complete = (
        outer is not None
        and cleanup["outer_exited"] is True
        and restoration is not None
        and restoration["original_running"] is True
        and restoration["original_paused"] is False
        and restoration["owned_loopback_port_clear"] is True
        and session_report is not None
        and session_report.get("status") in ("passed", "completed_with_step_failures")
        and cleanup["outer_returncode"] == expected_outer_returncode
        and session_cleanup_ok
        and problem is None
    )
    specialist_returncodes = [
        row["returncode"] for row in commands if row["kind"].startswith("specialist_")
    ]
    plan_had_failures = (
        isinstance(session_report, dict)
        and session_report.get("status") == "completed_with_step_failures"
    )
    if infrastructure_complete:
        status = (
            "completed_with_evaluation_failures"
            if plan_had_failures or any(value == 1 for value in specialist_returncodes)
            else "complete"
        )
        code = 1 if status == "completed_with_evaluation_failures" else 0
    else:
        status = (
            "interrupted"
            if isinstance(problem, dict) and problem.get("error_type") == "StudyInterrupted"
            else "infrastructure_failed"
        )
        code = code or 1

    receipt = {
        "schema_version": 1,
        "tag": args.tag,
        "status": status,
        "started_at": started_at,
        "finished_at": utc_now(),
        "wall_seconds": round(time.monotonic() - started, 3),
        "runtime_backend": args.runtime_backend,
        "source_config": {"path": str(args.config), "sha256": config_sha256},
        "plan": {
            "path": str(plan.source_path),
            "sha256": plan.sha256,
            "name": plan.name,
            "steps": [step.controls(args.runtime_backend) for step in plan.steps],
        },
        "specialist_manifest": {
            "sha256": manifest_process.get("stdout_sha256"),
            "document": manifest,
            **({"skipped_by_standard_only": True} if standard_only else {}),
        },
        "helper_sources": sources,
        "controls": {
            "profile": args.profile or args.tag,
            "startup_timeout": args.startup_timeout,
            "wait_for_idle": args.wait_for_idle,
            "max_seconds": args.max_seconds,
            "code_thinking": args.code_thinking,
            "prose_thinking": args.prose_thinking,
            "reasoning_cap": args.reasoning_cap,
            "capability_note": args.capability_note,
            "code": {
                "output_cap": args.code_output_cap,
                "temperature": 0.0,
                "seed": 0,
                "timeout": args.code_timeout,
            },
            "prose": {
                "output_cap": args.prose_output_cap,
                "temperature": 0.7,
                "seed": 0,
                "timeout": args.prose_timeout,
            },
        },
        "session": None if session is None else {
            "run_id": session.run_id,
            "model_name": session.model_name,
            "context": session.context,
            "initial_load_seconds": session.initial_load_seconds,
            "hold_seconds": session_hold_seconds,
            "session_started_at": session.session_started_at,
            "server_ready_at": session.server_ready_at,
            "holding_started_at": session.holding_started_at,
        },
        "commands": commands,
        "lifecycle": lifecycle,
        "cleanup": cleanup,
        "restoration": restoration,
        "session_report_status": None if session_report is None else session_report.get("status"),
        "artifacts": _artifact_inventory(paths),
        "problem": problem,
        "claims": {
            "all_gpu_fit": "not claimed; use measured server/telemetry evidence",
            "performance": "no result is claimed until this receipt records a completed run",
            "model_text_in_receipt": False,
        },
    }
    if standard_only:
        receipt["controls"]["standard_only"] = True
    if sampler_plan is not None:
        receipt["sampler_plan"] = {
            "path": str(sampler_plan.source_path),
            "sha256": sampler_plan.sha256,
            "expected_model_name": expected_model_name,
            "matched": bool(sampler_stages),
            "stages": sampler_stage_records,
            **({"skipped_by_standard_only": True} if standard_only else {}),
        }
    return receipt, code


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-backend", choices=("llama", "strata"), default="llama")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument(
        "--sampler-plan",
        type=Path,
        help="optional versioned model-to-sampler-stage plan",
    )
    parser.add_argument("--tag", required=True)
    parser.add_argument("--profile", help="specialist manifest profile (defaults to tag)")
    parser.add_argument(
        "--standard-only",
        action="store_true",
        help="run only the supplied model-session plan and skip supplemental specialist/sampler stages",
    )
    parser.add_argument("--prose-thinking", choices=("none", "low", "medium"), default="medium")
    parser.add_argument("--code-thinking", choices=("none", "low", "medium"), default="medium")
    parser.add_argument("--reasoning-cap", type=bounded_positive_int)
    parser.add_argument("--capability-note", type=validate_note)
    parser.add_argument(
        "--code-output-cap",
        type=lambda value: bounded_positive_int(value, MAX_OUTPUT_CAP),
        default=4_096,
    )
    parser.add_argument(
        "--prose-output-cap",
        type=lambda value: bounded_positive_int(value, MAX_OUTPUT_CAP),
        default=2_048,
    )
    parser.add_argument(
        "--code-timeout",
        type=lambda value: bounded_seconds(value, MAX_REQUEST_TIMEOUT),
        default=900.0,
    )
    parser.add_argument(
        "--prose-timeout",
        type=lambda value: bounded_seconds(value, MAX_REQUEST_TIMEOUT),
        default=900.0,
    )
    parser.add_argument(
        "--startup-timeout", type=safe.startup_timeout_value, default=480.0
    )
    parser.add_argument(
        "--wait-for-idle",
        type=lambda value: bounded_seconds(value, MAX_IDLE_WAIT_SECONDS),
        default=MAX_IDLE_WAIT_SECONDS,
    )
    parser.add_argument(
        "--max-seconds",
        type=lambda value: bounded_seconds(value, MAX_WINDOW_SECONDS),
        default=DEFAULT_MAX_WINDOW_SECONDS,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if TAG_RE.fullmatch(args.tag) is None:
        parser.error("--tag must contain only letters, digits, dot, underscore, or hyphen")
    if args.profile is not None and PROFILE_RE.fullmatch(args.profile) is None:
        parser.error("--profile must contain only letters, digits, dot, underscore, or hyphen")
    args.config = args.config.expanduser().resolve()
    args.plan = args.plan.expanduser().resolve()
    receipt_path = RESULTS / f"{args.tag}-study-trial.json"
    if os.path.lexists(receipt_path):
        print(f"specialist study: immutable receipt already exists: {receipt_path}", file=sys.stderr)
        return 2
    try:
        receipt, code = execute_study(args)
        _write_json_exclusive(receipt_path, receipt)
    except (StudyError, model_session.SessionError, OSError, subprocess.SubprocessError) as exc:
        print(f"specialist study: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    progress("study_complete", tag=args.tag, status=receipt["status"], returncode=code)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
