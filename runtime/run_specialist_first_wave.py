#!/usr/bin/env python3
"""Run the pinned first-wave specialist cohort sequentially and fail closed."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import signal
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
    from . import model_boundary_gate, prepare_specialists as prep
else:
    import model_boundary_gate
    import prepare_specialists as prep


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
BASELINE_RECEIPT = RESULTS / "specialists-gpt20b-8k-study-trial.json"
COHORT_RECEIPT = RESULTS / "specialists-first-wave-cohort.json"
COHORT_EVENTS = RESULTS / "specialists-first-wave-cohort-events.jsonl"
PREPARE_HELPER = ROOT / "runtime" / "prepare_specialists.py"
STUDY_HELPER = ROOT / "runtime" / "specialist_study_trial.py"
SPECIALIST_BENCH = ROOT / "runtime" / "specialist_bench.py"
STANDARD_PLAN = ROOT / "runtime" / "specialists-standard-8k-plan.json"
ROLE_PLAN = ROOT / "runtime" / "specialists-role-screen-plan.json"
MAX_WAIT_SECONDS = 7_200.0
MAX_BASELINE_WAIT_SECONDS = 259_200.0
MAX_WINDOW_SECONDS = 86_400.0
MAX_OUTPUT_CAP = 65_536
MAX_REQUEST_TIMEOUT = 21_600.0
MAX_CONTEXT = 131_072
FIT_CTX_THRESHOLD = 32_768
RUN_PREFIX_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
PROCESS_TERM_SECONDS = 600.0
OOM_LOG_MAX_BYTES = 64 * 1024 * 1024
OOM_PATTERNS = (
    ("cuda_out_of_memory", re.compile(rb"cuda[^\n]{0,160}out of memory", re.IGNORECASE)),
    ("cudamalloc_failed", re.compile(rb"cudamalloc[^\n]{0,160}(?:failed|out of memory)", re.IGNORECASE)),
    ("cuda_allocation_failed", re.compile(rb"failed to allocate[^\n]{0,160}cuda", re.IGNORECASE)),
    ("cuda_error_2", re.compile(rb"cuda(?: error)?(?: code)?\s*[:=]?\s*2\b", re.IGNORECASE)),
)


class CohortError(RuntimeError):
    pass


@dataclass(frozen=True)
class Profile:
    key: str
    label: str
    plan: Path
    auto_config: Path
    strict_config: Path
    strict_tag: str
    fallback_tag: str
    default_prose_thinking: str
    capability_note: str | None = None

    @property
    def spec(self) -> prep.ModelSpec:
        return prep.SPECS[self.key]


PROFILES = (
    Profile(
        "lfm25-qad-q4_0",
        "LFM2.5-24B-A2B QAD-Q4_0",
        STANDARD_PLAN,
        prep.SPECS["lfm25-qad-q4_0"].config_path,
        ROOT / "runtime" / "lfm25-qad-q4_0-gpu-all-fit-off-8k.json",
        "specialists-lfm25-qad-q4_0-strict-8k",
        "specialists-lfm25-qad-q4_0-auto-fallback-8k",
        "medium",
        "Reasoning is always on for this model; medium is the supported prose mode.",
    ),
    Profile(
        "kat-v25-iq3_xxs",
        "KAT-Coder-V2.5-Dev IQ3_XXS",
        STANDARD_PLAN,
        prep.SPECS["kat-v25-iq3_xxs"].config_path,
        ROOT / "runtime" / "kat-v25-iq3_xxs-gpu-all-fit-off-8k.json",
        "specialists-kat-v25-iq3_xxs-strict-8k",
        "specialists-kat-v25-iq3_xxs-auto-fallback-8k",
        "none",
    ),
    Profile(
        "gemma4-style12-q5_k_m",
        "Gemma-4-12B-StyleTune Q5_K_M",
        ROLE_PLAN,
        prep.SPECS["gemma4-style12-q5_k_m"].config_path,
        ROOT / "runtime" / "gemma4-style12-q5_k_m-gpu-all-fit-off-8k.json",
        "specialists-gemma4-style12-q5_k_m-strict-8k",
        "specialists-gemma4-style12-q5_k_m-auto-fallback-8k",
        "none",
    ),
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path, description: str) -> dict[str, Any]:
    try:
        details = path.lstat()
        if not stat.S_ISREG(details.st_mode) or path.is_symlink():
            raise CohortError(f"{description} must be a regular non-symlink file")
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CohortError(f"{description} is unreadable or invalid: {path}") from exc
    if not isinstance(value, dict):
        raise CohortError(f"{description} must be a JSON object")
    return value


def _safe_cleanup(receipt: Mapping[str, Any]) -> bool:
    cleanup = receipt.get("cleanup")
    restoration = receipt.get("restoration")
    session_cleanup = cleanup.get("model_session_cleanup") if isinstance(cleanup, dict) else None
    return bool(
        isinstance(cleanup, dict)
        and cleanup.get("outer_exited") is True
        and isinstance(session_cleanup, dict)
        and session_cleanup.get("stopped") is True
        and session_cleanup.get("kill_sent") is False
        and isinstance(restoration, dict)
        and restoration.get("original_running") is True
        and restoration.get("original_paused") is False
        and restoration.get("owned_loopback_port_clear") is True
        and restoration.get("owned_loopback_port") == 18081
    )


def validate_baseline_receipt(receipt: Mapping[str, Any]) -> None:
    if receipt.get("status") not in ("complete", "completed_with_evaluation_failures"):
        raise CohortError("baseline study receipt is not evaluation-complete")
    if not _safe_cleanup(receipt):
        raise CohortError("baseline study receipt does not prove cleanup and restoration")


def wait_for_baseline(path: Path, timeout: float) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if os.path.lexists(path):
            receipt = _read_json(path, "baseline study receipt")
            validate_baseline_receipt(receipt)
            return receipt
        time.sleep(1.0)
    raise CohortError("completed baseline study receipt did not appear before the deadline")


def wait_for_model_file(spec: prep.ModelSpec, timeout: float) -> Path:
    path = spec.default_directory / spec.filename
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if os.path.lexists(path):
            try:
                details = path.lstat()
            except OSError as exc:
                raise CohortError(f"published model file became inaccessible: {path}") from exc
            if not stat.S_ISREG(details.st_mode) or path.is_symlink():
                raise CohortError(f"published model must be a regular non-symlink file: {path}")
            if details.st_size != spec.size:
                raise CohortError(
                    f"published model has size {details.st_size}; expected {spec.size}: {path}"
                )
            return path
        time.sleep(1.0)
    raise CohortError(f"verified final model publication did not appear before deadline: {path}")


class EventLog:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        self.path = path
        self._descriptor = os.open(path, flags, 0o600)

    def write(self, event: str, **fields: Any) -> dict[str, Any]:
        row = {"at": utc_now(), "event": event, **fields}
        payload = (json.dumps(row, separators=(",", ":")) + "\n").encode("utf-8")
        view = memoryview(payload)
        offset = 0
        while offset < len(view):
            written = os.write(self._descriptor, view[offset:])
            if written <= 0:
                raise OSError("short cohort event write")
            offset += written
        os.fsync(self._descriptor)
        print(json.dumps(row, separators=(",", ":")), flush=True)
        return row

    def close(self) -> None:
        if self._descriptor >= 0:
            os.close(self._descriptor)
            self._descriptor = -1


def write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}.{uuid.uuid4().hex}")
    descriptor = -1
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        view = memoryview(payload)
        offset = 0
        while offset < len(view):
            written = os.write(descriptor, view[offset:])
            if written <= 0:
                raise OSError("short cohort receipt write")
            offset += written
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise CohortError(f"immutable cohort receipt already exists: {path}") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def strict_config_document(source: Mapping[str, Any]) -> dict[str, Any]:
    document = json.loads(json.dumps(source))
    args = document.get("args")
    if not isinstance(args, list) or any(not isinstance(item, str) for item in args):
        raise CohortError("auto placement config args must be a string list")
    positions = [index for index, value in enumerate(args) if value == "--gpu-layers"]
    if len(positions) != 1 or positions[0] + 1 >= len(args) or args[positions[0] + 1] != "auto":
        raise CohortError("auto placement config must contain exactly --gpu-layers auto")
    if "--fit" in args or any(value.startswith("--fit=") for value in args):
        raise CohortError("auto placement config unexpectedly already controls --fit")
    args[positions[0] + 1] = "all"
    args.extend(("--fit", "off"))
    document["placement"] = {
        "policy": "strict_all_layers_fit_off",
        "requested_gpu_layers": "all",
        "fit": "off",
        "all_gpu_fit_claimed": False,
        "measurement_required": ["actual_gpu_layer_count", "peak_vram_bytes"],
    }
    document["runtime_validation"] = {
        **(document.get("runtime_validation") if isinstance(document.get("runtime_validation"), dict) else {}),
        "load_validated": False,
        "inference_validated": False,
        "strict_gpu_residency_validated": False,
    }
    return document


def context_config_document(source: Mapping[str, Any], context: int) -> dict[str, Any]:
    document = json.loads(json.dumps(source))
    source_context = document.get("context")
    if isinstance(source_context, bool) or not isinstance(source_context, int) or source_context <= 0:
        raise CohortError("auto placement config context must be a positive integer")
    document["context"] = context
    args = document.get("args")
    if not isinstance(args, list) or any(not isinstance(item, str) for item in args):
        raise CohortError("auto placement config args must be a string list")
    if context > FIT_CTX_THRESHOLD:
        positions = [index for index, value in enumerate(args) if value == "--fit-ctx"]
        inline = [value for value in args if value.startswith("--fit-ctx=")]
        if inline or len(positions) > 1 or (positions and positions[0] + 1 >= len(args)):
            raise CohortError("auto placement config has ambiguous --fit-ctx controls")
        if positions:
            args[positions[0] + 1] = str(context)
        else:
            args.extend(("--fit-ctx", str(context)))
    return document


def wait_at_model_boundary(
    args: argparse.Namespace,
    profile: Profile,
    study_path: Path,
    study_sha256: str,
) -> dict[str, Any] | None:
    directory = getattr(args, "model_boundary_gates", None)
    if directory is None:
        return None
    try:
        return model_boundary_gate.publish_and_wait(
            directory,
            getattr(args, "run_prefix", "specialists"),
            profile.key,
            {"path": str(study_path), "sha256": study_sha256},
            args.model_boundary_wait_seconds,
        )
    except model_boundary_gate.BoundaryGateError as exc:
        raise CohortError(f"model boundary gate failed for {profile.key}: {exc}") from exc


def publish_context_config(source_path: Path, output_path: Path, context: int) -> dict[str, Any]:
    source = _read_json(source_path, "prepared auto placement config")
    document = context_config_document(source, context)
    if source_path.resolve() == output_path.resolve():
        if source != document:
            raise CohortError("refusing to change the prepared default auto config in place")
        return document
    if os.path.lexists(output_path):
        if _read_json(output_path, "derived auto placement config") != document:
            raise CohortError(f"refusing to overwrite different derived config: {output_path}")
        return document
    write_json_exclusive(output_path, document)
    return document


def publish_strict_config(auto_path: Path, strict_path: Path) -> dict[str, Any]:
    source = _read_json(auto_path, "auto placement config")
    document = strict_config_document(source)
    if os.path.lexists(strict_path):
        if _read_json(strict_path, "strict GPU config") != document:
            raise CohortError(f"refusing to overwrite different strict config: {strict_path}")
        return document
    write_json_exclusive(strict_path, document)
    return document


def _terminate_owned(process: subprocess.Popen[Any]) -> bool:
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


def run_command(argv: Sequence[str], timeout: float, kind: str) -> dict[str, Any]:
    started_at = utc_now()
    started = time.monotonic()
    try:
        process = subprocess.Popen(
            list(argv), stdin=subprocess.DEVNULL, start_new_session=True, close_fds=True
        )
    except OSError as exc:
        raise CohortError(f"{kind} could not be launched") from exc
    try:
        while process.poll() is None:
            if time.monotonic() - started >= timeout:
                stopped = _terminate_owned(process)
                if not stopped:
                    raise CohortError(
                        f"{kind} exceeded its timeout and ignored SIGTERM; SIGKILL was not sent"
                    )
                returncode = 124
                break
            time.sleep(0.25)
        else:
            returncode = process.returncode
    except KeyboardInterrupt:
        if not _terminate_owned(process):
            raise CohortError(f"interrupted {kind} ignored SIGTERM; SIGKILL was not sent")
        raise
    return {
        "kind": kind,
        "argv": list(argv),
        "started_at": started_at,
        "finished_at": utc_now(),
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "returncode": returncode,
    }


def prepare_command(profile: Profile) -> list[str]:
    return [
        sys.executable,
        str(PREPARE_HELPER),
        "--prepare-config",
        "--model",
        profile.key,
    ]


def study_command(
    profile: Profile,
    config: Path,
    tag: str,
    prose_thinking: str,
    args: argparse.Namespace,
) -> list[str]:
    command = [
        sys.executable,
        str(STUDY_HELPER),
        "--runtime-backend",
        "llama",
        "--config",
        str(config),
        "--plan",
        str(profile.plan),
        "--tag",
        tag,
        "--profile",
        profile.key,
        "--code-thinking",
        "medium",
        "--prose-thinking",
        prose_thinking,
        "--wait-for-idle",
        format(args.wait_for_idle, ".15g"),
        "--max-seconds",
        format(args.max_window_seconds, ".15g"),
        "--code-output-cap",
        str(args.code_output_cap),
        "--prose-output-cap",
        str(args.prose_output_cap),
        "--code-timeout",
        format(args.code_timeout, ".15g"),
        "--prose-timeout",
        format(args.prose_timeout, ".15g"),
    ]
    if profile.capability_note is not None:
        command.extend(("--capability-note", profile.capability_note))
    sampler_plan = getattr(args, "sampler_plan", None)
    if sampler_plan is not None:
        command.extend(("--sampler-plan", str(sampler_plan)))
    if getattr(args, "standard_only", False):
        command.append("--standard-only")
    return command


def _study_receipt(tag: str, results: Path = RESULTS) -> tuple[Path, dict[str, Any]]:
    path = results / f"{tag}-study-trial.json"
    return path, _read_json(path, f"study receipt for {tag}")


def _study_is_complete(receipt: Mapping[str, Any]) -> bool:
    return receipt.get("status") in ("complete", "completed_with_evaluation_failures")


def _study_summary(receipt: Mapping[str, Any]) -> dict[str, Any]:
    manifest = receipt.get("specialist_manifest")
    return {
        "tag": receipt.get("tag"),
        "status": receipt.get("status"),
        "wall_seconds": receipt.get("wall_seconds"),
        "source_config": receipt.get("source_config"),
        "plan": receipt.get("plan"),
        "specialist_manifest_sha256": (
            manifest.get("sha256") if isinstance(manifest, dict) else None
        ),
        "session": receipt.get("session"),
        "cleanup": receipt.get("cleanup"),
        "restoration": receipt.get("restoration"),
    }


def validate_study_identity(
    receipt: Mapping[str, Any],
    *,
    profile: Profile,
    config: Path,
    tag: str,
    selected_prose: str,
    args: argparse.Namespace,
) -> None:
    source = receipt.get("source_config")
    plan = receipt.get("plan")
    controls = receipt.get("controls")
    session = receipt.get("session")
    sampler_plan = getattr(args, "sampler_plan", None)
    sampler_receipt = receipt.get("sampler_plan")
    sampler_identity_ok = sampler_plan is None or (
        isinstance(sampler_receipt, dict)
        and sampler_receipt.get("path") == str(sampler_plan)
        and sampler_receipt.get("sha256") == sha256_file(sampler_plan)
    )
    if (
        receipt.get("tag") != tag
        or receipt.get("runtime_backend") != "llama"
        or not isinstance(source, dict)
        or source.get("path") != str(config)
        or source.get("sha256") != sha256_file(config)
        or not isinstance(plan, dict)
        or plan.get("path") != str(profile.plan.resolve())
        or plan.get("sha256") != sha256_file(profile.plan)
        or not isinstance(controls, dict)
        or controls.get("code_thinking") != "medium"
        or controls.get("prose_thinking") != selected_prose
        or not isinstance(controls.get("code"), dict)
        or controls["code"].get("output_cap") != args.code_output_cap
        or controls["code"].get("timeout") != args.code_timeout
        or not isinstance(controls.get("prose"), dict)
        or controls["prose"].get("output_cap") != args.prose_output_cap
        or controls["prose"].get("timeout") != args.prose_timeout
        or bool(controls.get("standard_only", False))
        != bool(getattr(args, "standard_only", False))
        or (
            session is not None
            and (not isinstance(session, dict) or session.get("context") != args.context)
        )
        or not sampler_identity_ok
    ):
        raise CohortError(f"study receipt identity or request controls differ for {profile.key}")


def _verified_manifest_preflight(command: Mapping[str, Any]) -> bool:
    return bool(
        command.get("kind") == "specialist_manifest"
        and type(command.get("returncode")) is int
        and command.get("returncode") == 0
        and command.get("argv") == [
            sys.executable,
            str(SPECIALIST_BENCH),
            "--print-manifest",
        ]
    )


def _startup_oom(tag: str, receipt: Mapping[str, Any], results: Path = RESULTS) -> dict[str, Any]:
    specialist_commands = [
        row
        for row in receipt.get("commands", [])
        if (
            isinstance(row, dict)
            and str(row.get("kind", "")).startswith("specialist_")
            and not _verified_manifest_preflight(row)
        )
    ] if isinstance(receipt.get("commands"), list) else []
    if receipt.get("session") is not None or specialist_commands:
        return {"confirmed": False, "reason": "failure occurred after holding or specialist launch"}
    log = results / f"{tag}-server.log"
    try:
        details = log.lstat()
        if not stat.S_ISREG(details.st_mode) or log.is_symlink() or details.st_size > OOM_LOG_MAX_BYTES:
            return {"confirmed": False, "reason": "server log is absent, unsafe, or over its bound"}
        payload = log.read_bytes()
    except OSError:
        return {"confirmed": False, "reason": "server log is unreadable"}
    matches = [name for name, pattern in OOM_PATTERNS if pattern.search(payload)]
    return {
        "confirmed": bool(matches),
        "phase": "startup_before_holding",
        "diagnostic_classes": matches,
        "server_log": str(log),
        "server_log_sha256": hashlib.sha256(payload).hexdigest(),
    }


def prose_thinking(profile: Profile, args: argparse.Namespace) -> str:
    values = {
        "lfm25-qad-q4_0": args.lfm_prose_thinking,
        "kat-v25-iq3_xxs": args.kat_prose_thinking,
        "gemma4-style12-q5_k_m": args.style_prose_thinking,
    }
    return values[profile.key]


def _validate_static_inputs(profiles: Sequence[Profile]) -> dict[str, str]:
    paths = [PREPARE_HELPER, STUDY_HELPER]
    paths.extend({profile.plan for profile in profiles})
    hashes: dict[str, str] = {}
    for path in paths:
        if not path.is_file() or path.is_symlink():
            raise CohortError(f"required helper or plan is missing: {path}")
        hashes[str(path)] = sha256_file(path)
    return hashes


def execute_cohort(
    args: argparse.Namespace,
    *,
    profiles: Sequence[Profile] = PROFILES,
    results: Path = RESULTS,
    baseline_path: Path = BASELINE_RECEIPT,
    cohort_receipt: Path | None = None,
    cohort_events: Path | None = None,
) -> tuple[dict[str, Any], int]:
    cohort_receipt = cohort_receipt or results / COHORT_RECEIPT.name
    cohort_events = cohort_events or results / COHORT_EVENTS.name
    if os.path.lexists(cohort_receipt):
        raise CohortError("immutable first-wave cohort receipt already exists")
    events = EventLog(cohort_events)
    started_at = utc_now()
    started = time.monotonic()
    model_records: list[dict[str, Any]] = []
    problem: dict[str, str] | None = None
    code = 1
    baseline: dict[str, Any] | None = None
    baseline_sha: str | None = None
    sources: dict[str, str] = {}
    try:
        sources = _validate_static_inputs(profiles)
        sampler_plan = getattr(args, "sampler_plan", None)
        if sampler_plan is not None:
            if not sampler_plan.is_file() or sampler_plan.is_symlink():
                raise CohortError(f"sampler plan is missing or unsafe: {sampler_plan}")
            sources[str(sampler_plan)] = sha256_file(sampler_plan)
        if getattr(args, "model_boundary_gates", None) is not None:
            gate_source = Path(model_boundary_gate.__file__).resolve()
            if not gate_source.is_file() or gate_source.is_symlink():
                raise CohortError(f"model boundary gate helper is missing or unsafe: {gate_source}")
            sources[str(gate_source)] = sha256_file(gate_source)
        events.write("baseline_wait_started", path=str(baseline_path))
        baseline = wait_for_baseline(baseline_path, args.baseline_wait_seconds)
        baseline_sha = sha256_file(baseline_path)
        events.write("baseline_gate_passed", status=baseline.get("status"), sha256=baseline_sha)

        for profile in profiles:
            spec = profile.spec
            record: dict[str, Any] = {
                "key": profile.key,
                "label": profile.label,
                "started_at": utc_now(),
                "commands": [],
                "strict_tag": profile.strict_tag,
                "fallback_tag": None,
                "status": "pending",
            }
            model_records.append(record)
            events.write("model_wait_started", model=profile.key, path=str(spec.default_directory / spec.filename))
            model_path = wait_for_model_file(spec, args.download_wait_seconds)
            record["model_path"] = str(model_path)
            events.write("model_published", model=profile.key, size=spec.size)

            prepare_record = run_command(
                prepare_command(profile), args.prepare_timeout_seconds, f"prepare_{profile.key}"
            )
            record["commands"].append(prepare_record)
            events.write(
                "stage_complete",
                model=profile.key,
                kind=prepare_record["kind"],
                argv=prepare_record["argv"],
                returncode=prepare_record["returncode"],
                elapsed_seconds=prepare_record["elapsed_seconds"],
            )
            if prepare_record["returncode"] != 0:
                raise CohortError(f"prepare-config failed for {profile.key}")
            auto_started = utc_now()
            auto_document = publish_context_config(spec.config_path, profile.auto_config, args.context)
            auto_config = profile.auto_config
            auto_stage = {
                "kind": "publish_context_config",
                "argv": None,
                "started_at": auto_started,
                "finished_at": utc_now(),
                "returncode": 0,
                "source": str(spec.config_path),
                "source_sha256": sha256_file(spec.config_path),
                "output": str(auto_config),
                "output_sha256": sha256_file(auto_config),
                "context": auto_document["context"],
            }
            record["commands"].append(auto_stage)
            strict_started = utc_now()
            strict_document = publish_strict_config(auto_config, profile.strict_config)
            strict_stage = {
                "kind": "publish_strict_config",
                "argv": None,
                "started_at": strict_started,
                "finished_at": utc_now(),
                "returncode": 0,
                "source": str(auto_config),
                "source_sha256": sha256_file(auto_config),
                "output": str(profile.strict_config),
                "output_sha256": sha256_file(profile.strict_config),
                "controls": strict_document["placement"],
            }
            record["commands"].append(strict_stage)
            record["artifact_receipt"] = str(spec.receipt_path)
            record["artifact_receipt_sha256"] = sha256_file(spec.receipt_path)
            events.write(
                "strict_config_published",
                model=profile.key,
                source_sha256=strict_stage["source_sha256"],
                output_sha256=strict_stage["output_sha256"],
                artifact_receipt_sha256=record["artifact_receipt_sha256"],
            )
            record["plan"] = str(profile.plan)
            record["plan_sha256"] = sources[str(profile.plan)]
            selected_prose = prose_thinking(profile, args)
            record["controls"] = {
                "context": args.context,
                "code_thinking": "medium",
                "prose_thinking": selected_prose,
                "code_output_cap": args.code_output_cap,
                "prose_output_cap": args.prose_output_cap,
                "code_timeout": args.code_timeout,
                "prose_timeout": args.prose_timeout,
                "capability_note": profile.capability_note,
                **({"standard_only": True} if getattr(args, "standard_only", False) else {}),
            }

            strict_process = run_command(
                study_command(
                    profile, profile.strict_config, profile.strict_tag, selected_prose, args
                ),
                args.wait_for_idle + args.max_window_seconds + 300.0,
                f"study_strict_{profile.key}",
            )
            record["commands"].append(strict_process)
            events.write(
                "stage_complete",
                model=profile.key,
                kind=strict_process["kind"],
                argv=strict_process["argv"],
                returncode=strict_process["returncode"],
                elapsed_seconds=strict_process["elapsed_seconds"],
            )
            strict_receipt_path, strict_receipt = _study_receipt(profile.strict_tag, results)
            validate_study_identity(
                strict_receipt,
                profile=profile,
                config=profile.strict_config,
                tag=profile.strict_tag,
                selected_prose=selected_prose,
                args=args,
            )
            record["strict_receipt"] = str(strict_receipt_path)
            record["strict_receipt_sha256"] = sha256_file(strict_receipt_path)
            record["strict_status"] = strict_receipt.get("status")
            record["strict_study"] = _study_summary(strict_receipt)
            if _study_is_complete(strict_receipt) and _safe_cleanup(strict_receipt):
                if strict_process["returncode"] not in (0, 1):
                    raise CohortError(
                        f"strict study for {profile.key} has an inconsistent process return code"
                    )
                record["status"] = strict_receipt["status"]
                record["selected_profile"] = "strict_all_layers_fit_off"
                events.write("model_complete", model=profile.key, status=record["status"], fallback=False)
                boundary = wait_at_model_boundary(
                    args, profile, strict_receipt_path, record["strict_receipt_sha256"]
                )
                if boundary is not None:
                    record["boundary_gate"] = boundary
                    events.write("model_boundary_released", model=profile.key, **boundary)
                continue

            oom = _startup_oom(profile.strict_tag, strict_receipt, results)
            record["strict_startup_oom"] = oom
            if not (_safe_cleanup(strict_receipt) and oom.get("confirmed") is True):
                raise CohortError(
                    f"strict study for {profile.key} failed without a safely restored CUDA startup OOM"
                )
            events.write(
                "strict_startup_oom",
                model=profile.key,
                diagnostics=oom.get("diagnostic_classes", []),
            )
            fallback_process = run_command(
                study_command(profile, auto_config, profile.fallback_tag, selected_prose, args),
                args.wait_for_idle + args.max_window_seconds + 300.0,
                f"study_auto_fallback_{profile.key}",
            )
            record["commands"].append(fallback_process)
            events.write(
                "stage_complete",
                model=profile.key,
                kind=fallback_process["kind"],
                argv=fallback_process["argv"],
                returncode=fallback_process["returncode"],
                elapsed_seconds=fallback_process["elapsed_seconds"],
            )
            fallback_path, fallback_receipt = _study_receipt(profile.fallback_tag, results)
            validate_study_identity(
                fallback_receipt,
                profile=profile,
                config=auto_config,
                tag=profile.fallback_tag,
                selected_prose=selected_prose,
                args=args,
            )
            record["fallback_tag"] = profile.fallback_tag
            record["fallback_receipt"] = str(fallback_path)
            record["fallback_receipt_sha256"] = sha256_file(fallback_path)
            record["fallback_status"] = fallback_receipt.get("status")
            record["fallback_study"] = _study_summary(fallback_receipt)
            if (
                not _study_is_complete(fallback_receipt)
                or not _safe_cleanup(fallback_receipt)
                or fallback_process["returncode"] not in (0, 1)
            ):
                raise CohortError(f"auto fallback study for {profile.key} did not complete safely")
            record["status"] = fallback_receipt["status"]
            record["selected_profile"] = "auto_fallback_after_strict_cuda_oom"
            events.write("model_complete", model=profile.key, status=record["status"], fallback=True)
            boundary = wait_at_model_boundary(
                args, profile, fallback_path, record["fallback_receipt_sha256"]
            )
            if boundary is not None:
                record["boundary_gate"] = boundary
                events.write("model_boundary_released", model=profile.key, **boundary)

        code = 1 if any(row["status"] == "completed_with_evaluation_failures" for row in model_records) else 0
    except KeyboardInterrupt as exc:
        problem = {"error_type": type(exc).__name__, "error": "interrupted"}
        code = 130
        events.write("cohort_interrupted")
    except (CohortError, OSError, subprocess.SubprocessError) as exc:
        problem = {"error_type": type(exc).__name__, "error": str(exc)}
        code = 2
        events.write("cohort_failed", **problem)
    finally:
        events.close()

    complete = len(model_records) == len(profiles) and all(
        row.get("status") in ("complete", "completed_with_evaluation_failures")
        for row in model_records
    )
    status = (
        "completed_with_evaluation_failures"
        if complete and any(row["status"] == "completed_with_evaluation_failures" for row in model_records)
        else "complete" if complete else "infrastructure_failed"
    )
    receipt = {
        "schema_version": 1,
        "immutable": True,
        "status": status,
        "started_at": started_at,
        "finished_at": utc_now(),
        "wall_seconds": round(time.monotonic() - started, 3),
        "baseline": {
            "path": str(baseline_path),
            "sha256": baseline_sha,
            "status": None if baseline is None else baseline.get("status"),
            "cleanup_gate_passed": baseline is not None and _safe_cleanup(baseline),
        },
        "source_hashes": sources,
        "controls": {
            "sequence": [profile.key for profile in profiles],
            "selected_models": [profile.key for profile in profiles],
            "download_wait_seconds": args.download_wait_seconds,
            "max_window_seconds": args.max_window_seconds,
            "context": args.context,
            "code_output_cap": args.code_output_cap,
            "prose_output_cap": args.prose_output_cap,
            "code_timeout": args.code_timeout,
            "prose_timeout": args.prose_timeout,
            "strict_gpu_layers": "all",
            "strict_fit": "off",
            "strict_all_gpu_residency_claimed": False,
            "fallback_policy": "one auto placement run only after safely restored startup CUDA OOM",
            **({"standard_only": True} if getattr(args, "standard_only", False) else {}),
        },
        "models": model_records,
        "events": str(cohort_events),
        "problem": problem,
    }
    return receipt, code


def bounded_seconds(maximum: float):
    def parse(value: str) -> float:
        parsed = float(value)
        if not math.isfinite(parsed) or not 0 < parsed <= maximum:
            raise argparse.ArgumentTypeError(f"must be finite and from >0 through {maximum:g}")
        return parsed

    return parse


def bounded_int(maximum: int):
    def parse(value: str) -> int:
        parsed = int(value)
        if not 1 <= parsed <= maximum:
            raise argparse.ArgumentTypeError(f"must be from 1 through {maximum}")
        return parsed

    return parse


def run_prefix(value: str) -> str:
    if RUN_PREFIX_RE.fullmatch(value) is None:
        raise argparse.ArgumentTypeError("must be a safe 1-64 character run prefix")
    return value


def model_selection(value: str) -> tuple[str, ...]:
    values = [item.strip() for item in value.split(",")]
    if not values or any(not item for item in values):
        raise argparse.ArgumentTypeError("must contain one or more comma-separated model keys")
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("model selection must not contain duplicates")
    known = {profile.key for profile in PROFILES}
    unknown = sorted(set(values) - known)
    if unknown:
        raise argparse.ArgumentTypeError("unknown model key(s): " + ", ".join(unknown))
    requested = set(values)
    return tuple(profile.key for profile in PROFILES if profile.key in requested)


def _context_label(context: int) -> str:
    return f"{context // 1024}k" if context % 1024 == 0 else f"ctx{context}"


def profiles_for_args(args: argparse.Namespace) -> tuple[Profile, ...]:
    label = _context_label(args.context)
    plans = {
        "lfm25-qad-q4_0": args.standard_plan,
        "kat-v25-iq3_xxs": args.standard_plan,
        "gemma4-style12-q5_k_m": args.role_plan,
    }
    profiles: list[Profile] = []
    for original in PROFILES:
        default_identity = args.run_prefix == "specialists" and args.context == 8192
        auto_config = (
            original.spec.config_path
            if default_identity
            else ROOT / "runtime" / f"{args.run_prefix}-{original.key}-auto-{label}.json"
        )
        strict_config = (
            original.strict_config
            if default_identity
            else ROOT / "runtime" / f"{args.run_prefix}-{original.key}-gpu-all-fit-off-{label}.json"
        )
        profiles.append(
            Profile(
                key=original.key,
                label=original.label,
                plan=plans[original.key],
                auto_config=auto_config,
                strict_config=strict_config,
                strict_tag=f"{args.run_prefix}-{original.key}-strict-{label}",
                fallback_tag=f"{args.run_prefix}-{original.key}-auto-fallback-{label}",
                default_prose_thinking=original.default_prose_thinking,
                capability_note=original.capability_note,
            )
        )
    selected = getattr(args, "models", None)
    if selected is None:
        return tuple(profiles)
    return tuple(profile for profile in profiles if profile.key in selected)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-prefix", type=run_prefix, default="specialists")
    parser.add_argument(
        "--models",
        type=model_selection,
        help="comma-separated canonical model keys; execution retains canonical order",
    )
    parser.add_argument("--baseline-receipt", type=Path, default=BASELINE_RECEIPT)
    parser.add_argument("--context", type=bounded_int(MAX_CONTEXT), default=8192)
    parser.add_argument("--standard-plan", type=Path, default=STANDARD_PLAN)
    parser.add_argument("--role-plan", type=Path, default=ROLE_PLAN)
    parser.add_argument("--sampler-plan", type=Path)
    parser.add_argument("--standard-only", action="store_true")
    parser.add_argument(
        "--baseline-wait-seconds",
        type=bounded_seconds(MAX_BASELINE_WAIT_SECONDS),
        default=7200.0,
    )
    parser.add_argument("--download-wait-seconds", type=bounded_seconds(MAX_WAIT_SECONDS), default=7200.0)
    parser.add_argument("--prepare-timeout-seconds", type=bounded_seconds(MAX_WAIT_SECONDS), default=7200.0)
    parser.add_argument("--max-window-seconds", type=bounded_seconds(MAX_WINDOW_SECONDS), default=7200.0)
    parser.add_argument("--code-output-cap", type=bounded_int(MAX_OUTPUT_CAP), default=4096)
    parser.add_argument("--prose-output-cap", type=bounded_int(MAX_OUTPUT_CAP), default=2048)
    parser.add_argument("--code-timeout", type=bounded_seconds(MAX_REQUEST_TIMEOUT), default=900.0)
    parser.add_argument("--prose-timeout", type=bounded_seconds(MAX_REQUEST_TIMEOUT), default=900.0)
    parser.add_argument("--wait-for-idle", type=bounded_seconds(900.0), default=900.0)
    parser.add_argument("--model-boundary-gates", type=Path)
    parser.add_argument(
        "--model-boundary-wait-seconds",
        type=bounded_seconds(MAX_WINDOW_SECONDS),
        default=MAX_WINDOW_SECONDS,
    )
    parser.add_argument("--lfm-prose-thinking", choices=("none", "low", "medium"), default="medium")
    parser.add_argument("--kat-prose-thinking", choices=("none", "low", "medium"), default="none")
    parser.add_argument("--style-prose-thinking", choices=("none", "low", "medium"), default="none")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.baseline_receipt = args.baseline_receipt.expanduser().resolve()
    args.standard_plan = args.standard_plan.expanduser().resolve()
    args.role_plan = args.role_plan.expanduser().resolve()
    if args.sampler_plan is not None:
        args.sampler_plan = args.sampler_plan.expanduser().resolve()
    if args.model_boundary_gates is not None:
        args.model_boundary_gates = args.model_boundary_gates.expanduser().resolve()
    profiles = profiles_for_args(args)
    cohort_receipt = RESULTS / f"{args.run_prefix}-first-wave-cohort.json"
    cohort_events = RESULTS / f"{args.run_prefix}-first-wave-cohort-events.jsonl"
    if os.path.lexists(cohort_receipt) or os.path.lexists(cohort_events):
        print("first-wave cohort outputs already exist; refusing overwrite", file=sys.stderr)
        return 2
    try:
        receipt, code = execute_cohort(
            args,
            profiles=profiles,
            baseline_path=args.baseline_receipt,
            cohort_receipt=cohort_receipt,
            cohort_events=cohort_events,
        )
        write_json_exclusive(cohort_receipt, receipt)
    except (CohortError, OSError) as exc:
        print(f"first-wave cohort: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    return code


if __name__ == "__main__":
    raise SystemExit(main())
