#!/usr/bin/env python3
"""Inspect or run the frozen StyleTune native ``--poll 0`` diagnostic.

Inspection is the default.  It validates immutable inputs and terminal primary
metadata, writes one content-free receipt, and never opens a GPU window.
``--execute`` is required for the single owned smoke-and-prose window.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import signal
import socket
import stat
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runtime import gpu_window  # noqa: E402
from runtime import sampler_recipe_probe  # noqa: E402
from runtime import specialist_cases  # noqa: E402
from runtime import specialist_study_trial as study  # noqa: E402
import bench_cases  # noqa: E402


RESULTS = ROOT / "results"
DESIGN = RESULTS / "specialists-style-poll0-diagnostic-plan-v1.json"
CONFIG = ROOT / "runtime/gemma4-style12-poll0-gpu-72k.json"
BASE_CONFIG = (
    ROOT / "runtime/specialists-final72k64k-gemma4-style12-q5_k_m-gpu-all-fit-off-72k.json"
)
PRIMARY_PLAN = ROOT / "runtime/specialists-all-tasks-72k-64k-plan.json"
PRIMARY_SAMPLER_PLAN = ROOT / "runtime/specialists-publisher-native-sampler-plan-v3.json"
SMOKE_PLAN = ROOT / "runtime/gemma4-style12-poll0-smoke-72k-plan.json"
RECIPE = ROOT / "runtime/gemma4-style12-poll0-prose-control-recipe.json"
REFERENCE_RUNNER = ROOT / "runtime/run_style_mtp_probe.py"

MODEL = "gemma-4-12b-styletune-q5_k_m"
PRIMARY_PROFILE = "gemma4-style12-q5_k_m"
CANDIDATE_PROFILE = "gemma4-style12-q5_k_m-poll0"
PARENT_TAG = "specialists-final72k64k-gemma4-style12-q5_k_m-strict-72k"
DEFAULT_TAG = "specialists-style-poll0-72k64k-v1"
PRIMARY_RECEIPT = RESULTS / f"{PARENT_TAG}-study-trial.json"
PRIMARY_PROSE = RESULTS / f"{PARENT_TAG}-specialist-prose.jsonl"
BOUNDARY_DIRECTORY = RESULTS / "specialists-final72k64k-boundaries"
BOUNDARY_READY = (
    BOUNDARY_DIRECTORY
    / "specialists-final72k64k-gemma4-style12-q5_k_m-ready.json"
)

CONTEXT = 73_728
OUTPUT_CAP = 65_536
REQUEST_TIMEOUT = 21_600.0
PRIMARY_TOKEN_CEILING = 8_192
PRIMARY_WALL_CEILING = 3_600.0
WINDOW_SECONDS = 14_400.0
RESTORATION_RESERVE_SECONDS = 180.0
STARTUP_TIMEOUT = 480.0
IDLE_WAIT_SECONDS = 900.0
ENDPOINT = "http://127.0.0.1:18081/v1"
NATIVE_PORT = 18_081
STRATA_PORT = 18_080
MAX_JSON_BYTES = 32 * 1024 * 1024
MAX_JSONL_BYTES = 64 * 1024 * 1024
PROCESS_TERM_SECONDS = 20.0

DESIGN_SHA256 = "93e90ba5a54b68d2480d9227229a006c10ec9c80b2a66af0e9b0a896a45e4f14"
CONFIG_SHA256 = "ba5f2e452728240427e3e28da31811724f27e5de4d5392227a94a5db69dc44ec"
BASE_CONFIG_SHA256 = "3ffb305436bb7a19207064c64a9a32b9e864cbaf843a8633780d1005867b9e15"
SMOKE_PLAN_SHA256 = "96bae4d488bf95f9a8d50498db160b49ab37e4100522bfaa18dc08cf41b1c30b"
RECIPE_SHA256 = "9efc2c2544a09c28d509aa45560cd1e15a3366a87096e05dd2401a864e50cf8f"
PRIMARY_PLAN_SHA256 = "24ff6ec0986d44b82f351cf66abdb37030e2e9e52822bf854cefca251b13c97b"
PRIMARY_SAMPLER_PLAN_SHA256 = (
    "e0d2c6ba142f39956898ddb3a8b1487adf233c2aba3302c77667debf1c050b5b"
)
PUBLISHER_RECIPE_SHA256 = (
    "f7b9dcd7edaf90f0856471de9e729b9cff2cc3eb6d3a727d304ccd1db9ea881b"
)
BENCH_CASES_SHA256 = "b8f020837fd385baa38e4aed8e21d66454e98743b6f2abe5211134ca84f7f27b"
PUBLISHER_RECIPE = (
    ROOT / "runtime/specialist-sampler-recipes/styletune-prose-author-partial.json"
)

PINNED_FILES = {
    DESIGN: DESIGN_SHA256,
    CONFIG: CONFIG_SHA256,
    SMOKE_PLAN: SMOKE_PLAN_SHA256,
    RECIPE: RECIPE_SHA256,
    BASE_CONFIG: BASE_CONFIG_SHA256,
    RESULTS / "specialists-style-host-cpu-observation.json":
        "48a73e1578e596fbd9a6f7868b339e4fa5feb0229ab25c5536565f9ad5ea97c3",
    ROOT / "runtime/gpu_window.py":
        "7982aa5906fd05f6cb61c9e2b54467724755472e7d17feb485a687a142ca6898",
    ROOT / "runtime/model_session.py":
        "968b2ce078975de23dd99670c2141995f460f36ae86490ce33153e7f1ae9c43c",
    ROOT / "runtime/specialist_study_trial.py":
        "2187ee832d3785727f836d156d343865efcccb299547e877815e8ac72920e734",
    ROOT / "runtime/sampler_recipe_probe.py":
        "8543bb44ba3242240aaa5646c51b2252c9b6367f3d624ec29b0ab21695bdfb18",
    ROOT / "runtime/llama_trial.py":
        "836253f7603323b250d9df742f02bc793a0e8c3b653d193f37107dacbe06ea04",
    REFERENCE_RUNNER:
        "7c8195471bf8449ccec0b0a18cd1bfcd7d8d502eb47f37f20a17adc3758440ab",
}


class ProbeError(RuntimeError):
    """The probe cannot safely continue."""


class ProbeInterrupted(BaseException):
    def __init__(self, signum: int) -> None:
        super().__init__(f"signal {signum}")
        self.signum = signum


@dataclass
class SignalLatch:
    received: int | None = None
    cleaning: bool = False

    def handle(self, signum: int, _frame: Any) -> None:
        self.received = signum
        if not self.cleaning:
            raise ProbeInterrupted(signum)


@dataclass(frozen=True)
class ProbePaths:
    prefix: Path
    receipt: Path
    control: Path
    events: Path
    session_report: Path
    server_log: Path
    smoke: Path
    sampler_prefix: Path
    sampler_requests: Path
    sampler_manifest: Path
    sampler_receipt: Path
    sampler_visible: Path


@dataclass(frozen=True)
class Eligibility:
    eligible: bool
    failures: tuple[str, ...]
    primary_receipt: Mapping[str, Any]
    primary_prose: tuple[Mapping[str, Any], ...]
    boundary: Mapping[str, Any]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _regular(path: Path, label: str, max_bytes: int | None = None) -> os.stat_result:
    try:
        details = path.lstat()
    except OSError as exc:
        raise ProbeError(f"{label} is missing or inaccessible: {path}") from exc
    if not stat.S_ISREG(details.st_mode) or path.is_symlink():
        raise ProbeError(f"{label} must be a regular non-symlink file: {path}")
    if max_bytes is not None and details.st_size > max_bytes:
        raise ProbeError(f"{label} exceeds its {max_bytes}-byte bound: {path}")
    return details


def _json(path: Path, label: str) -> dict[str, Any]:
    _regular(path, label, MAX_JSON_BYTES)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ProbeError(f"{label} is not valid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ProbeError(f"{label} must be a JSON object: {path}")
    return value


def _jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    _regular(path, label, MAX_JSONL_BYTES)
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ProbeError(f"{label} row {line_number} is not a JSON object")
                rows.append(value)
    except ProbeError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ProbeError(f"{label} contains invalid JSONL: {path}") from exc
    return rows


GENERATED_TEXT_KEYS = {
    "response_text", "reasoning_text", "reasoning_content", "analysis_text", "raw_reasoning"
}


def _contains_generated_text(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(
            key in GENERATED_TEXT_KEYS or _contains_generated_text(nested)
            for key, nested in value.items()
        )
    if isinstance(value, list):
        return any(_contains_generated_text(item) for item in value)
    return False


def _content_free_rows(path: Path, label: str) -> list[dict[str, Any]]:
    rows = _jsonl(path, label)
    for index, row in enumerate(rows, start=1):
        if _contains_generated_text(row):
            raise ProbeError(f"{label} row {index} embeds generated text")
    return rows


def _exclusive_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        offset = 0
        while offset < len(data):
            written = os.write(descriptor, data[offset:])
            if written <= 0:
                raise OSError("short receipt write")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _integer(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def probe_paths(tag: str, prefix: Path, *, results: Path = RESULTS) -> ProbePaths:
    if study.TAG_RE.fullmatch(tag) is None:
        raise ProbeError("--tag contains unsafe characters")
    prefix = prefix.expanduser()
    if not prefix.is_absolute():
        prefix = ROOT / prefix
    prefix = prefix.resolve()
    if prefix.parent != results.resolve():
        raise ProbeError("--output-prefix must be directly under results")
    if prefix.name != tag and not prefix.name.startswith(tag + "-"):
        raise ProbeError("--output-prefix basename must equal --tag or start with '<tag>-' ")
    sampler_prefix = Path(f"{prefix}-prose")
    return ProbePaths(
        prefix=prefix,
        receipt=Path(f"{prefix}-style-poll0-probe-receipt.json"),
        control=results / f"{tag}-control.json",
        events=results / f"{tag}-runtime-events.jsonl",
        session_report=results / f"{tag}-session-report.json",
        server_log=results / f"{tag}-server.log",
        smoke=results / f"{tag}-01-poll0_smoke_json_none_256.jsonl",
        sampler_prefix=sampler_prefix,
        sampler_requests=Path(f"{sampler_prefix}-sampler-requests.jsonl"),
        sampler_manifest=Path(f"{sampler_prefix}-sampler-manifest.json"),
        sampler_receipt=Path(f"{sampler_prefix}-sampler-receipt.json"),
        sampler_visible=Path(f"{sampler_prefix}-sampler-visible-texts"),
    )


def assert_unused(paths: ProbePaths) -> None:
    candidates = tuple(asdict(paths).values())[1:]
    collisions = [str(path) for path in candidates if os.path.lexists(path)]
    if collisions:
        raise ProbeError("refusing existing output/control path: " + ", ".join(collisions))


def _assert_poll0_config(candidate: Mapping[str, Any], base: Mapping[str, Any]) -> None:
    candidate_copy = json.loads(json.dumps(candidate))
    base_copy = json.loads(json.dumps(base))
    candidate_args = candidate_copy.pop("args", None)
    base_args = base_copy.pop("args", None)
    if candidate_copy != base_copy:
        raise ProbeError("poll0 candidate differs from the base config outside args")
    if (
        not isinstance(base_args, list)
        or not all(isinstance(item, str) for item in base_args)
        or not isinstance(candidate_args, list)
        or candidate_args != base_args + ["--poll", "0"]
    ):
        raise ProbeError("poll0 candidate must differ only by appended '--poll 0'")
    lowered = [item.lower() for item in candidate_args]
    if any("draft" in item or "mtp" in item or item.startswith("--spec-") for item in lowered):
        raise ProbeError("poll0 candidate unexpectedly enables MTP or a draft model")
    if candidate_args.count("--poll") != 1:
        raise ProbeError("poll0 candidate has ambiguous poll controls")


def verify_pins() -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    for path, expected in PINNED_FILES.items():
        _regular(path, "pinned input", MAX_JSON_BYTES if path.suffix == ".json" else None)
        actual = sha256_file(path)
        if actual != expected:
            raise ProbeError(f"pinned input hash mismatch: {path}")
        files.append({"path": str(path), "sha256": actual})

    design = _json(DESIGN, "poll0 frozen design")
    candidate = _json(CONFIG, "poll0 candidate config")
    base = _json(BASE_CONFIG, "poll0 base config")
    smoke = _json(SMOKE_PLAN, "poll0 smoke plan")
    recipe = _json(RECIPE, "poll0 prose recipe")
    _assert_poll0_config(candidate, base)

    declared = design.get("source_pins")
    declared_pins = {
        str(Path(str(row.get("path"))).resolve()): row.get("sha256")
        for row in declared if isinstance(row, Mapping)
    } if isinstance(declared, list) else {}
    expected_declared = {
        str(path.resolve()): expected
        for path, expected in PINNED_FILES.items()
        if path != DESIGN and path != REFERENCE_RUNNER
    }
    if declared_pins != expected_declared:
        raise ProbeError("frozen design source pins changed")
    change = design.get("candidate_change") if isinstance(design.get("candidate_change"), Mapping) else {}
    if not (
        change.get("only_config_field_changed") == "args"
        and change.get("only_appended_args") == ["--poll", "0"]
        and change.get("no_MTP_or_other_draft") is True
    ):
        raise ProbeError("frozen design no longer declares the exact poll0-only change")

    steps = smoke.get("steps")
    if not isinstance(steps, list) or len(steps) != 1:
        raise ProbeError("poll0 smoke plan must contain exactly one step")
    step = steps[0]
    if not isinstance(step, Mapping) or (
        step.get("id") != "poll0_smoke_json_none_256"
        or step.get("suite") != "quick"
        or step.get("cases") != ["quick_json_extract"]
        or step.get("thinking") != "none"
        or step.get("temperature") != 0
        or step.get("seed") != 0
        or step.get("output_tokens") != 256
        or step.get("request_timeout") != 180
        or step.get("repeat") != 1
    ):
        raise ProbeError("poll0 smoke controls changed")
    if recipe != {
        "schema_version": 1,
        "label": "Canonical NONE prose temperature0.7 control for Style poll0",
        "claim": "custom_experiment",
        "sources": [],
        "sampler_overrides": {"temperature": 0.7},
        "notes": (
            "Matches canonical primary prose request controls; unspecified sampler fields "
            "inherit unchanged native server defaults."
        ),
    }:
        raise ProbeError("poll0 canonical prose recipe changed")
    sampler_recipe_probe.validate_recipe(recipe)
    shards = candidate.get("shards")
    if not (
        candidate.get("model_name") == MODEL
        and candidate.get("context") == CONTEXT
        and candidate.get("host") == "127.0.0.1"
        and candidate.get("port") == NATIVE_PORT
        and isinstance(shards, list)
        and len(shards) == 1
        and isinstance(shards[0], Mapping)
        and shards[0].get("sha256")
        == "14d78cfccb5f052a9c4bfcdceac30fae570a09dea4453c7d929fa57f703ff350"
    ):
        raise ProbeError("poll0 target runtime identity changed")
    return {
        "verified": True,
        "files": files,
        "candidate": {
            "config": str(CONFIG),
            "config_sha256": CONFIG_SHA256,
            "base_config": str(BASE_CONFIG),
            "base_config_sha256": BASE_CONFIG_SHA256,
            "only_appended_args": ["--poll", "0"],
            "model": MODEL,
            "context": CONTEXT,
            "target_declared_sha256": shards[0].get("sha256"),
            "mtp_or_draft_enabled": False,
        },
        "smoke": {"step_id": step.get("id"), "case_id": step["cases"][0]},
        "recipe": {"path": str(RECIPE), "sha256": RECIPE_SHA256},
    }


def _primary_receipt_evidence(receipt: Mapping[str, Any]) -> tuple[list[str], dict[str, Any]]:
    failures: list[str] = []
    if receipt.get("tag") != PARENT_TAG:
        failures.append("primary receipt tag is not the original frozen Style tag")
    if receipt.get("status") not in {
        "complete", "completed_with_evaluation_failures", "completed_with_request_failures"
    }:
        failures.append("primary study is not terminal and complete")
    if receipt.get("runtime_backend") != "llama":
        failures.append("primary study did not use native llama")
    if receipt.get("problem") not in (None, {}):
        failures.append("primary study records an infrastructure problem")
    if any(key in receipt for key in ("continuation_of", "continuation_lineage", "parent_receipt")):
        failures.append("continuation lineage is unsupported by poll0 controller v1")

    plan = receipt.get("plan") if isinstance(receipt.get("plan"), Mapping) else {}
    source = receipt.get("source_config") if isinstance(receipt.get("source_config"), Mapping) else {}
    session = receipt.get("session") if isinstance(receipt.get("session"), Mapping) else {}
    if plan.get("sha256") != PRIMARY_PLAN_SHA256:
        failures.append("primary study plan hash differs from the frozen all-task plan")
    if not _same_path(plan.get("path"), PRIMARY_PLAN):
        failures.append("primary study plan path differs from the frozen all-task plan")
    if source.get("sha256") != BASE_CONFIG_SHA256:
        failures.append("primary target config hash differs from the frozen Style config")
    if not _same_path(source.get("path"), BASE_CONFIG):
        failures.append("primary target config path differs from the frozen Style config")
    if session.get("model_name") != MODEL or session.get("context") != CONTEXT:
        failures.append("primary resident model/context identity differs")

    controls = receipt.get("controls") if isinstance(receipt.get("controls"), Mapping) else {}
    code_controls = controls.get("code") if isinstance(controls.get("code"), Mapping) else {}
    prose_controls = controls.get("prose") if isinstance(controls.get("prose"), Mapping) else {}
    if not (
        controls.get("profile") == PRIMARY_PROFILE
        and controls.get("code_thinking") == "medium"
        and controls.get("prose_thinking") == "none"
        and code_controls.get("output_cap") == OUTPUT_CAP
        and code_controls.get("temperature") == 0
        and code_controls.get("seed") == 0
        and code_controls.get("timeout") == REQUEST_TIMEOUT
        and prose_controls.get("output_cap") == OUTPUT_CAP
        and prose_controls.get("temperature") == 0.7
        and prose_controls.get("seed") == 0
        and prose_controls.get("timeout") == REQUEST_TIMEOUT
    ):
        failures.append("primary receipt lacks the frozen MED-code/NONE-prose controls")

    cleanup = receipt.get("cleanup") if isinstance(receipt.get("cleanup"), Mapping) else {}
    server_cleanup = (
        cleanup.get("model_session_cleanup")
        if isinstance(cleanup.get("model_session_cleanup"), Mapping) else {}
    )
    restoration = receipt.get("restoration") if isinstance(receipt.get("restoration"), Mapping) else {}
    if not (
        cleanup.get("outer_exited") is True
        and cleanup.get("orchestrator_sigkill_sent") is False
        and server_cleanup.get("stopped") is True
        and server_cleanup.get("kill_sent") is False
    ):
        failures.append("primary owned cleanup evidence is incomplete")
    if not (
        restoration.get("original_running") is True
        and restoration.get("original_paused") is False
        and restoration.get("owned_loopback_port_clear") is True
    ):
        failures.append("primary restoration evidence is incomplete")
    if receipt.get("session_report_status") not in {"passed", "completed_with_step_failures"}:
        failures.append("primary quick2/hard4 lifecycle is not terminal")

    commands = receipt.get("commands") if isinstance(receipt.get("commands"), list) else []
    for kind in ("specialist_code", "specialist_prose"):
        matching_commands = [
            row for row in commands
            if isinstance(row, Mapping) and row.get("kind") == kind
        ]
        terminal = len(matching_commands) == 1 and matching_commands[0].get("returncode") in {0, 1}
        if not terminal:
            failures.append(f"primary {kind} helper lifecycle is not terminal")

    sampler_plan = receipt.get("sampler_plan") if isinstance(receipt.get("sampler_plan"), Mapping) else {}
    stages = sampler_plan.get("stages") if isinstance(sampler_plan.get("stages"), list) else []
    publisher = [
        stage for stage in stages
        if isinstance(stage, Mapping)
        and stage.get("group") == "prose"
        and stage.get("suffix") == "styletune-publisher-prose"
    ]
    if not (
        sampler_plan.get("sha256") == PRIMARY_SAMPLER_PLAN_SHA256
        and _same_path(sampler_plan.get("path"), PRIMARY_SAMPLER_PLAN)
        and sampler_plan.get("matched") is True
        and sampler_plan.get("expected_model_name") == MODEL
        and len(stages) == 1
        and len(publisher) == 1
    ):
        failures.append("primary receipt lacks the one frozen publisher prose stage")
    else:
        stage = publisher[0]
        stage_recipe = stage.get("recipe") if isinstance(stage.get("recipe"), Mapping) else {}
        stage_receipt = stage.get("receipt") if isinstance(stage.get("receipt"), Mapping) else {}
        publisher_terminal = bool(
            stage.get("prose_thinking") == "none"
            and stage.get("model") == MODEL
            and stage.get("returncode") in {0, 1}
            and stage_recipe.get("sha256") == PUBLISHER_RECIPE_SHA256
            and stage_receipt.get("status") in {"complete", "completed_with_request_failures"}
        )
        if not publisher_terminal:
            failures.append("primary publisher prose helper lifecycle is not terminal")
    evidence = {
        "tag": receipt.get("tag"),
        "status": receipt.get("status"),
        "runtime_backend": receipt.get("runtime_backend"),
        "plan_sha256": plan.get("sha256"),
        "source_config_sha256": source.get("sha256"),
        "session_run_id": session.get("run_id"),
        "model": session.get("model_name"),
        "context": session.get("context"),
        "cleanup": {
            "outer_exited": cleanup.get("outer_exited"),
            "orchestrator_sigkill_sent": cleanup.get("orchestrator_sigkill_sent"),
            "model_session_stopped": server_cleanup.get("stopped"),
            "model_session_kill_sent": server_cleanup.get("kill_sent"),
        },
        "restoration": {
            "original_running": restoration.get("original_running"),
            "original_paused": restoration.get("original_paused"),
            "owned_loopback_port_clear": restoration.get("owned_loopback_port_clear"),
        },
    }
    return list(dict.fromkeys(failures)), evidence


def _fixed_primary_paths(receipt_path: Path) -> dict[str, Path]:
    directory = receipt_path.resolve().parent
    prefix = directory / PARENT_TAG
    publisher = Path(f"{prefix}-styletune-publisher-prose")
    return {
        "study_receipt": Path(f"{prefix}-study-trial.json"),
        "session_report": Path(f"{prefix}-session-report.json"),
        "quick": Path(f"{prefix}-01-quick_medium_65536.jsonl"),
        "hard": Path(f"{prefix}-02-hard_medium_65536.jsonl"),
        "code": Path(f"{prefix}-specialist-code.jsonl"),
        "prose": Path(f"{prefix}-specialist-prose.jsonl"),
        "publisher_requests": Path(f"{publisher}-sampler-requests.jsonl"),
        "publisher_manifest": Path(f"{publisher}-sampler-manifest.json"),
        "publisher_receipt": Path(f"{publisher}-sampler-receipt.json"),
        "publisher_visible": Path(f"{publisher}-sampler-visible-texts"),
    }


def _same_path(value: Any, expected: Path) -> bool:
    return isinstance(value, str) and Path(value).expanduser().resolve() == expected.resolve()


def _artifact_inventory_path(
    receipt: Mapping[str, Any], key: str, expected: Path, failures: list[str],
) -> None:
    artifacts = receipt.get("artifacts") if isinstance(receipt.get("artifacts"), Mapping) else {}
    item = artifacts.get(key) if isinstance(artifacts.get(key), Mapping) else {}
    if not (
        _same_path(item.get("path"), expected)
        and item.get("exists") is True
        and item.get("regular_file") is True
        and item.get("directory") is False
    ):
        failures.append(f"primary receipt artifact {key!r} does not identify the fixed regular file")


def _metadata_hash(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        .encode("utf-8")
    ).hexdigest()


def _project_response(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    timing = value.get("timing") if isinstance(value.get("timing"), Mapping) else {}
    return {
        "actual_prompt_tokens": value.get("actual_prompt_tokens"),
        "actual_completion_tokens": value.get("actual_completion_tokens"),
        "done_reason": value.get("done_reason"),
        "stream_complete": value.get("stream_complete"),
        "truncated": value.get("truncated"),
        "reasoning_chars": value.get("reasoning_chars"),
        "wall_seconds": timing.get("wall_seconds"),
    }


def _project_error(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    return {"type": value.get("type"), "recorded": True}


def _project_request(value: Any) -> dict[str, Any]:
    request = value if isinstance(value, Mapping) else {}
    keys = (
        "endpoint", "backend", "model", "profile", "context_max_tokens", "thinking",
        "chat_template_thinking", "reasoning_budget_tokens", "temperature", "seed",
        "output_token_limit", "timeout_seconds", "context_mode", "prompt_chars",
        "prompt_sha256", "sampler_overrides", "sampler_recipe_label",
        "sampler_recipe_sha256",
    )
    return {key: request.get(key) for key in keys if key in request}


def _standard_metadata_rows(path: Path, label: str) -> list[dict[str, Any]]:
    """Project standard bench rows without retaining or hashing generated strings."""
    _regular(path, label, MAX_JSONL_BYTES)
    projected: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ProbeError(f"{label} row {line_number} is not a JSON object")
                projected.append({
                    "schema_version": row.get("schema_version"),
                    "run_id": row.get("run_id"),
                    "call_id": row.get("call_id"),
                    "repeat_index": row.get("repeat_index"),
                    "suite": row.get("suite"),
                    "case_id": row.get("case_id"),
                    "case_version": row.get("case_version"),
                    "case_source_sha256": row.get("case_source_sha256"),
                    "request": _project_request(row.get("request")),
                    "response": _project_response(row.get("response")),
                    "error": _project_error(row.get("error")),
                })
                row.clear()
    except ProbeError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ProbeError(f"{label} contains invalid JSONL: {path}") from exc
    return projected


def _content_free_attempt_projection(row: Mapping[str, Any]) -> dict[str, Any]:
    visible = row.get("visible_text") if isinstance(row.get("visible_text"), Mapping) else {}
    metrics = row.get("response_metrics") if isinstance(row.get("response_metrics"), Mapping) else {}
    return {
        "schema_version": row.get("schema_version"),
        "run_id": row.get("run_id"),
        "call_id": row.get("call_id"),
        "repeat_index": row.get("repeat_index"),
        "tag": row.get("tag"),
        "model": row.get("model"),
        "profile": row.get("profile"),
        "suite": row.get("suite"),
        "case_id": row.get("case_id"),
        "case_group": row.get("case_group"),
        "case_version": row.get("case_version"),
        "case_sha256": row.get("case_sha256"),
        "suite_sha256": row.get("suite_sha256"),
        "request": _project_request(row.get("request")),
        "response": _project_response(row.get("response")),
        "response_metrics": {
            "finish_reason": metrics.get("finish_reason"),
            "output_tokens_reported": metrics.get("output_tokens_reported"),
            "wall_seconds": metrics.get("wall_seconds"),
        },
        "visible_text": {
            "sha256": visible.get("sha256"), "chars": visible.get("chars"),
            "bytes": visible.get("bytes"), "reasoning_excluded": visible.get("reasoning_excluded"),
        },
        "error": _project_error(row.get("error")),
        "objective_error": _project_error(row.get("objective_error")),
    }


def _terminal_attempt(projected: Mapping[str, Any]) -> bool:
    error = projected.get("error")
    if isinstance(error, Mapping):
        return isinstance(error.get("type"), str) and bool(error.get("type"))
    response = projected.get("response") if isinstance(projected.get("response"), Mapping) else {}
    tokens = _integer(response.get("actual_completion_tokens"))
    return bool(
        response.get("stream_complete") is True
        and response.get("done_reason") in {"stop", "length", "eos", "end_turn"}
        and isinstance(response.get("truncated"), bool)
        and tokens is not None and tokens >= 0
    )


def _expected_bench(suite: str) -> dict[str, dict[str, Any]]:
    return {
        case.case_id: {
            "prompt_sha256": hashlib.sha256(case.prompt.encode("utf-8")).hexdigest(),
            "prompt_chars": len(case.prompt),
        }
        for case in bench_cases.get_cases(
            suite, context_max=CONTEXT, output_tokens=OUTPUT_CAP, seed=0,
        )
    }


def _expected_specialist(group: str) -> dict[str, dict[str, Any]]:
    return {
        case.case_id: {
            "case_sha256": specialist_cases.case_sha256(case),
            "prompt_sha256": hashlib.sha256(case.prompt.encode("utf-8")).hexdigest(),
            "prompt_chars": len(case.prompt),
        }
        for case in specialist_cases.get_cases(groups=(group,))
    }


def _attempt_record(
    *, kind: str, path: Path, projected: Mapping[str, Any], valid: bool,
) -> dict[str, Any]:
    response = projected.get("response") if isinstance(projected.get("response"), Mapping) else {}
    return {
        "kind": kind,
        "path": str(path.resolve()),
        "case_id": projected.get("case_id"),
        "call_id": projected.get("call_id"),
        "terminal": _terminal_attempt(projected),
        "request_error": isinstance(projected.get("error"), Mapping),
        "stream_complete": response.get("stream_complete"),
        "truncated": response.get("truncated"),
        "finish_reason": response.get("done_reason"),
        "completion_tokens": response.get("actual_completion_tokens"),
        "metadata_sha256": _metadata_hash(projected),
        "valid": valid,
    }


def _standard_attempts(
    path: Path, *, suite: str, kind: str,
) -> tuple[list[str], list[dict[str, Any]]]:
    failures: list[str] = []
    expected = _expected_bench(suite)
    rows = _standard_metadata_rows(path, f"primary {suite} request metadata")
    seen: set[str] = set()
    run_ids = {row.get("run_id") for row in rows}
    result: list[dict[str, Any]] = []
    if len(rows) != len(expected):
        failures.append(f"primary {suite} artifact does not contain exactly {len(expected)} rows")
    if len(run_ids) != 1 or not all(isinstance(value, str) and value for value in run_ids):
        failures.append(f"primary {suite} rows do not share one nonempty run ID")
    for row in rows:
        case_id = row.get("case_id")
        request = row.get("request") if isinstance(row.get("request"), Mapping) else {}
        identity = expected.get(case_id) if isinstance(case_id, str) else None
        unique_case = identity is not None and case_id not in seen
        if unique_case:
            seen.add(case_id)
        identity_ok = bool(
            unique_case
            and row.get("schema_version") == 1
            and row.get("repeat_index") == 0
            and row.get("suite") == suite
            and row.get("case_version") == bench_cases.CASE_VERSION
            and row.get("case_source_sha256") == BENCH_CASES_SHA256
            and request.get("prompt_sha256") == identity["prompt_sha256"]
            and request.get("prompt_chars") == identity["prompt_chars"]
        )
        controls_ok = bool(
            request.get("endpoint") == ENDPOINT
            and request.get("backend") == "openai"
            and request.get("model") == MODEL
            and request.get("context_max_tokens") == CONTEXT
            and request.get("thinking") == "medium"
            and request.get("chat_template_thinking") == "on"
            and request.get("reasoning_budget_tokens") is None
            and request.get("temperature") == 0
            and request.get("seed") == 0
            and request.get("output_token_limit") == OUTPUT_CAP
            and request.get("timeout_seconds") == REQUEST_TIMEOUT
            and request.get("context_mode") == "full"
        )
        terminal = _terminal_attempt(row)
        call_id = row.get("call_id")
        call_ok = isinstance(call_id, str) and bool(call_id)
        valid = identity_ok and controls_ok and terminal and call_ok
        if not identity_ok:
            failures.append(f"primary {suite} row has unexpected or duplicate frozen case identity")
        if not controls_ok:
            failures.append(f"{suite}:{case_id}: frozen request controls mismatch")
        if not terminal:
            failures.append(f"{suite}:{case_id}: lacks terminal response or recorded request error")
        if not call_ok:
            failures.append(f"{suite}:{case_id}: lacks a nonempty call ID")
        result.append(_attempt_record(kind=kind, path=path, projected=row, valid=valid))
    if seen != set(expected):
        failures.append(f"primary {suite} artifact does not cover each frozen case exactly once")
    return failures, result


def _specialist_attempts(
    path: Path, *, group: str, kind: str, temperature: float,
) -> tuple[list[str], list[dict[str, Any]]]:
    failures: list[str] = []
    expected = _expected_specialist(group)
    rows = _content_free_rows(path, f"primary specialist {group} metadata")
    seen: set[str] = set()
    run_ids = {row.get("run_id") for row in rows}
    result: list[dict[str, Any]] = []
    thinking = "medium" if group == "code" else "none"
    template = "on" if group == "code" else "off"
    if len(rows) != len(expected):
        failures.append(f"primary specialist {group} artifact does not contain exactly three rows")
    if len(run_ids) != 1 or not all(isinstance(value, str) and value for value in run_ids):
        failures.append(f"primary specialist {group} rows do not share one nonempty run ID")
    for raw in rows:
        row = _content_free_attempt_projection(raw)
        case_id = row.get("case_id")
        request = row.get("request") if isinstance(row.get("request"), Mapping) else {}
        identity = expected.get(case_id) if isinstance(case_id, str) else None
        unique_case = identity is not None and case_id not in seen
        if unique_case:
            seen.add(case_id)
        identity_ok = bool(
            unique_case
            and row.get("schema_version") == 1
            and row.get("repeat_index") == 0
            and row.get("tag") == f"{PARENT_TAG}-{group}"
            and row.get("model") == MODEL
            and row.get("profile") == PRIMARY_PROFILE
            and row.get("suite") == f"specialist_{group}"
            and row.get("case_group") == group
            and row.get("case_version") == specialist_cases.CASE_VERSION
            and row.get("case_sha256") == identity["case_sha256"]
            and row.get("suite_sha256") == specialist_cases.FROZEN_SUITE_SHA256
            and request.get("prompt_sha256") == identity["prompt_sha256"]
            and request.get("prompt_chars") == identity["prompt_chars"]
        )
        controls_ok = bool(
            request.get("endpoint") == ENDPOINT
            and request.get("backend") == "openai"
            and request.get("model") == MODEL
            and request.get("profile") == PRIMARY_PROFILE
            and request.get("context_max_tokens") == CONTEXT
            and request.get("thinking") == thinking
            and request.get("chat_template_thinking") == template
            and request.get("reasoning_budget_tokens") is None
            and request.get("temperature") == temperature
            and request.get("seed") == 0
            and request.get("output_token_limit") == OUTPUT_CAP
            and request.get("timeout_seconds") == REQUEST_TIMEOUT
        )
        terminal = _terminal_attempt(row)
        call_ok = isinstance(row.get("call_id"), str) and bool(row.get("call_id"))
        valid = identity_ok and controls_ok and terminal and call_ok
        if not identity_ok:
            failures.append(f"primary specialist {group} row has unexpected or duplicate frozen case identity")
        if not controls_ok:
            failures.append(f"canonical_{group}:{case_id}: frozen request controls mismatch")
        if not terminal:
            failures.append(f"canonical_{group}:{case_id}: lacks terminal response or recorded request error")
        if not call_ok:
            failures.append(f"canonical_{group}:{case_id}: lacks a nonempty call ID")
        result.append(_attempt_record(kind=kind, path=path, projected=row, valid=valid))
    if seen != set(expected):
        failures.append(f"primary specialist {group} artifact does not cover each frozen case exactly once")
    return failures, result


def _validate_specialist_manifest(receipt: Mapping[str, Any]) -> list[str]:
    failures: list[str] = []
    wrapper = (
        receipt.get("specialist_manifest")
        if isinstance(receipt.get("specialist_manifest"), Mapping) else {}
    )
    document = wrapper.get("document") if isinstance(wrapper.get("document"), Mapping) else {}
    rows = document.get("cases") if isinstance(document.get("cases"), list) else []
    expected = {
        case.case_id: (case.group, specialist_cases.case_sha256(case))
        for case in specialist_cases.get_cases()
    }
    observed: dict[str, tuple[Any, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping) or not isinstance(row.get("case_id"), str):
            continue
        case_id = row["case_id"]
        if case_id in observed:
            failures.append("primary specialist manifest contains duplicate case IDs")
            continue
        observed[case_id] = (row.get("group"), row.get("case_sha256"))
    if not (
        document.get("case_version") == specialist_cases.CASE_VERSION
        and document.get("suite_sha256") == specialist_cases.FROZEN_SUITE_SHA256
        and document.get("case_count") == len(expected)
        and observed == expected
    ):
        failures.append("primary specialist manifest does not match the frozen six-case suite")
    digest = wrapper.get("sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        failures.append("primary specialist manifest lacks its recorded SHA256")
    return failures


def _validate_session_report(
    path: Path, receipt: Mapping[str, Any], paths: Mapping[str, Path],
) -> tuple[list[str], dict[str, Any]]:
    failures: list[str] = []
    report = _json(path, "primary model session report")
    session = receipt.get("session") if isinstance(receipt.get("session"), Mapping) else {}
    steps = report.get("steps") if isinstance(report.get("steps"), list) else []
    expected_steps = (
        ("quick_medium_65536", "quick", paths["quick"], 2),
        ("hard_medium_65536", "hard", paths["hard"], 4),
    )
    if not (
        report.get("schema_version") == 1
        and report.get("tag") == PARENT_TAG
        and report.get("runtime_backend") == "llama"
        and report.get("status") == receipt.get("session_report_status")
        and report.get("session_id") == session.get("run_id")
        and _same_path(report.get("source_config"), BASE_CONFIG)
        and report.get("source_config_sha256") == BASE_CONFIG_SHA256
        and isinstance(report.get("plan"), Mapping)
        and _same_path(report["plan"].get("path"), PRIMARY_PLAN)
        and report["plan"].get("sha256") == PRIMARY_PLAN_SHA256
        and len(steps) == len(expected_steps)
    ):
        failures.append("primary model session report identity or terminal status is invalid")
    for index, (step_id, suite, result_path, case_count) in enumerate(expected_steps):
        step = steps[index] if index < len(steps) and isinstance(steps[index], Mapping) else {}
        controls = step.get("controls") if isinstance(step.get("controls"), Mapping) else {}
        if not (
            step.get("index") == index + 1
            and step.get("id") == step_id
            and _same_path(step.get("result"), result_path)
            and step.get("returncode") in {0, 1}
            and step.get("error") is None
            and controls.get("id") == step_id
            and controls.get("suite") == suite
            and controls.get("thinking") == "medium"
            and controls.get("chat_template_thinking") == "on"
            and controls.get("temperature") == 0
            and controls.get("seed") == 0
            and controls.get("repeat") == 1
            and controls.get("output_tokens") == OUTPUT_CAP
            and controls.get("request_timeout") == REQUEST_TIMEOUT
            and controls.get("cases") == []
        ):
            failures.append(f"primary session step {step_id} does not match its fixed terminal controls")
    return failures, {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "status": report.get("status"),
        "session_id": report.get("session_id"),
        "steps": [
            {
                "id": row.get("id"), "result": row.get("result"),
                "returncode": row.get("returncode"), "error": row.get("error"),
            }
            for row in steps if isinstance(row, Mapping)
        ],
    }


def _validate_specialist_command_paths(
    receipt: Mapping[str, Any], paths: Mapping[str, Path],
) -> list[str]:
    failures: list[str] = []
    commands = receipt.get("commands") if isinstance(receipt.get("commands"), list) else []
    for group in ("code", "prose"):
        selected = [
            row for row in commands
            if isinstance(row, Mapping) and row.get("kind") == f"specialist_{group}"
        ]
        argv = selected[0].get("argv") if len(selected) == 1 else None
        expected = paths[group]
        try:
            output_value = argv[argv.index("--jsonl") + 1] if isinstance(argv, list) else None
            group_value = argv[argv.index("--groups") + 1] if isinstance(argv, list) else None
        except (ValueError, IndexError):
            output_value = group_value = None
        if not (
            len(selected) == 1
            and selected[0].get("returncode") in {0, 1}
            and _same_path(output_value, expected)
            and group_value == group
        ):
            failures.append(f"primary specialist {group} command does not identify its fixed result")
    return failures


def _publisher_attempts(
    receipt: Mapping[str, Any], paths: Mapping[str, Path],
) -> tuple[list[str], list[dict[str, Any]], dict[str, Any]]:
    failures: list[str] = []
    sampler_plan = receipt.get("sampler_plan") if isinstance(receipt.get("sampler_plan"), Mapping) else {}
    stages = sampler_plan.get("stages") if isinstance(sampler_plan.get("stages"), list) else []
    stage = stages[0] if len(stages) == 1 and isinstance(stages[0], Mapping) else {}
    stage_receipt = stage.get("receipt") if isinstance(stage.get("receipt"), Mapping) else {}
    stage_recipe = stage.get("recipe") if isinstance(stage.get("recipe"), Mapping) else {}
    if not (
        stage.get("group") == "prose"
        and stage.get("suffix") == "styletune-publisher-prose"
        and stage.get("prose_thinking") == "none"
        and stage.get("returncode") in {0, 1}
        and _same_path(stage_receipt.get("path"), paths["publisher_receipt"])
        and stage_receipt.get("exists") is True
        and stage_recipe.get("sha256") == PUBLISHER_RECIPE_SHA256
        and _same_path(stage_recipe.get("path"), PUBLISHER_RECIPE)
    ):
        failures.append("primary publisher stage does not identify the fixed receipt/recipe")

    sampler = _json(paths["publisher_receipt"], "primary publisher sampler receipt")
    manifest = _json(paths["publisher_manifest"], "primary publisher sampler manifest")
    rows = _content_free_rows(paths["publisher_requests"], "primary publisher prose metadata")
    session = receipt.get("session") if isinstance(receipt.get("session"), Mapping) else {}
    expected = _expected_specialist("prose")
    expected_artifacts = {
        "requests": paths["publisher_requests"],
        "manifest": paths["publisher_manifest"],
        "receipt": paths["publisher_receipt"],
        "visible_dir": paths["publisher_visible"],
    }
    sampler_artifacts = sampler.get("artifacts") if isinstance(sampler.get("artifacts"), Mapping) else {}
    if any(not _same_path(sampler_artifacts.get(key), value) for key, value in expected_artifacts.items()):
        failures.append("primary publisher sampler receipt artifact paths are not the fixed paths")
    recipe = sampler.get("recipe") if isinstance(sampler.get("recipe"), Mapping) else {}
    controls = sampler.get("controls") if isinstance(sampler.get("controls"), Mapping) else {}
    if not (
        sampler.get("schema_version") == 1
        and sampler.get("diagnostic_kind") == "resident_sampler_recipe_probe"
        and sampler.get("tag") == PARENT_TAG
        and sampler.get("model") == MODEL
        and sampler.get("model_session_run_id") == session.get("run_id")
        and sampler.get("runtime_backend") == "llama"
        and sampler.get("status") in {"complete", "completed_with_request_failures"}
        and sampler.get("case_count") == 3
        and sampler.get("prose_prompt_count") == 3
        and recipe.get("sha256") == PUBLISHER_RECIPE_SHA256
        and _same_path(recipe.get("path"), PUBLISHER_RECIPE)
        and recipe.get("sampler_overrides") == {"temperature": 1.0, "min_p": 0.1}
        and controls.get("context_max_tokens") == CONTEXT
        and controls.get("expected_context_tokens") == CONTEXT
        and controls.get("actual_context_tokens") == CONTEXT
        and controls.get("output_token_limit") == OUTPUT_CAP
        and controls.get("timeout_seconds") == REQUEST_TIMEOUT
        and controls.get("seed") == 0
        and controls.get("prose_thinking") == "none"
    ):
        failures.append("primary publisher sampler receipt identity or controls are invalid")

    manifest_recipe = manifest.get("recipe") if isinstance(manifest.get("recipe"), Mapping) else {}
    manifest_controls = manifest.get("controls") if isinstance(manifest.get("controls"), Mapping) else {}
    manifest_cases = manifest.get("cases") if isinstance(manifest.get("cases"), list) else []
    manifest_observed = {
        row.get("case_id"): (row.get("group"), row.get("case_version"), row.get("case_sha256"), row.get("prompt_sha256"))
        for row in manifest_cases if isinstance(row, Mapping) and isinstance(row.get("case_id"), str)
    }
    manifest_expected = {
        case_id: ("prose", specialist_cases.CASE_VERSION, identity["case_sha256"], identity["prompt_sha256"])
        for case_id, identity in expected.items()
    }
    if not (
        manifest.get("schema_version") == 1
        and manifest.get("diagnostic_kind") == "resident_sampler_recipe_probe"
        and manifest.get("tag") == PARENT_TAG
        and manifest.get("model_session_run_id") == session.get("run_id")
        and manifest.get("run_id") == sampler.get("run_id")
        and manifest.get("case_count") == 3
        and manifest.get("prose_prompt_count") == 3
        and len(manifest_cases) == 3
        and manifest_recipe.get("sha256") == PUBLISHER_RECIPE_SHA256
        and _same_path(manifest_recipe.get("path"), PUBLISHER_RECIPE)
        and manifest_controls == controls
        and manifest_observed == manifest_expected
    ):
        failures.append("primary publisher manifest does not match the frozen prose3 inputs")

    seen: set[str] = set()
    attempts: list[dict[str, Any]] = []
    for raw in rows:
        row = _content_free_attempt_projection(raw)
        case_id = row.get("case_id")
        request = row.get("request") if isinstance(row.get("request"), Mapping) else {}
        identity = expected.get(case_id) if isinstance(case_id, str) else None
        unique_case = identity is not None and case_id not in seen
        if unique_case:
            seen.add(case_id)
        identity_ok = bool(
            unique_case
            and row.get("schema_version") == 1
            and row.get("repeat_index") == 0
            and row.get("tag") == PARENT_TAG
            and row.get("model") == MODEL
            and row.get("profile") == PRIMARY_PROFILE
            and row.get("suite") == "specialist_prose"
            and row.get("case_group") == "prose"
            and row.get("case_version") == specialist_cases.CASE_VERSION
            and row.get("case_sha256") == identity["case_sha256"]
            and row.get("suite_sha256") == specialist_cases.FROZEN_SUITE_SHA256
            and request.get("prompt_sha256") == identity["prompt_sha256"]
            and request.get("prompt_chars") == identity["prompt_chars"]
        )
        controls_ok = bool(
            request.get("endpoint") == ENDPOINT
            and request.get("backend") == "openai"
            and request.get("model") == MODEL
            and request.get("profile") == PRIMARY_PROFILE
            and request.get("context_max_tokens") == CONTEXT
            and request.get("thinking") == "none"
            and request.get("chat_template_thinking") == "off"
            and request.get("temperature") == 1.0
            and request.get("seed") == 0
            and request.get("output_token_limit") == OUTPUT_CAP
            and request.get("timeout_seconds") == REQUEST_TIMEOUT
            and request.get("sampler_recipe_sha256") == PUBLISHER_RECIPE_SHA256
            and request.get("sampler_overrides") == {"temperature": 1.0, "min_p": 0.1}
        )
        terminal = _terminal_attempt(row)
        call_ok = isinstance(row.get("call_id"), str) and bool(row.get("call_id"))
        valid = identity_ok and controls_ok and terminal and call_ok
        if not identity_ok:
            failures.append("primary publisher row has unexpected or duplicate frozen case identity")
        if not controls_ok:
            failures.append(f"publisher_prose:{case_id}: frozen request controls mismatch")
        if not terminal:
            failures.append(f"publisher_prose:{case_id}: lacks terminal response or recorded request error")
        if not call_ok:
            failures.append(f"publisher_prose:{case_id}: lacks a nonempty call ID")
        attempts.append(_attempt_record(
            kind="publisher_none_prose", path=paths["publisher_requests"],
            projected=row, valid=valid,
        ))
    if len(rows) != 3 or seen != set(expected):
        failures.append("primary publisher artifact does not cover each frozen prose case exactly once")

    completed = sum(
        row["terminal"] and not row["request_error"] and row["truncated"] is not True
        for row in attempts
    )
    truncated = sum(row["terminal"] and row["truncated"] is True for row in attempts)
    request_failures = sum(row["request_error"] for row in attempts)
    summary_attempts = sampler.get("attempts") if isinstance(sampler.get("attempts"), list) else []
    summary_pairs = {
        (row.get("case_id"), row.get("call_id"))
        for row in summary_attempts if isinstance(row, Mapping)
    }
    row_pairs = {(row["case_id"], row["call_id"]) for row in attempts}
    if not (
        sampler.get("attempted_count") == len(attempts) == 3
        and sampler.get("completed_count") == completed
        and sampler.get("truncated_count") == truncated
        and sampler.get("request_failure_count") == request_failures
        and len(summary_attempts) == 3
        and summary_pairs == row_pairs
        and stage_receipt.get("status") == sampler.get("status")
        and stage_receipt.get("attempted_count") == 3
        and stage_receipt.get("completed_count") == completed
        and stage_receipt.get("truncated_count") == truncated
        and stage_receipt.get("request_failure_count") == request_failures
        and stage_receipt.get("sha256") == sha256_file(paths["publisher_receipt"])
    ):
        failures.append("primary publisher receipt counts do not match validated terminal rows")
    return failures, attempts, {
        "manifest": {"path": str(paths["publisher_manifest"]), "sha256": sha256_file(paths["publisher_manifest"])},
        "receipt": {"path": str(paths["publisher_receipt"]), "sha256": sha256_file(paths["publisher_receipt"])},
    }


def _primary_attempt_evidence(
    receipt_path: Path, receipt: Mapping[str, Any], prose_paths: Sequence[Path],
) -> tuple[list[str], dict[str, Any]]:
    failures: list[str] = []
    paths = _fixed_primary_paths(receipt_path)
    if sha256_file(Path(bench_cases.__file__).resolve()) != BENCH_CASES_SHA256:
        failures.append("frozen standard benchmark case source hash changed")
    if receipt_path.resolve() != paths["study_receipt"]:
        failures.append("primary receipt does not use the fixed original study receipt name")
    if len(prose_paths) != 1 or prose_paths[0].resolve() != paths["prose"]:
        failures.append("canonical prose input is not the fixed original primary artifact")
    artifact_keys = {
        "session_report": "session_report",
        "plan_01_quick_medium_65536": "quick",
        "plan_02_hard_medium_65536": "hard",
        "code": "code",
        "prose": "prose",
        "sampler_01_requests": "publisher_requests",
        "sampler_01_manifest": "publisher_manifest",
        "sampler_01_receipt": "publisher_receipt",
    }
    for inventory_key, path_key in artifact_keys.items():
        _artifact_inventory_path(receipt, inventory_key, paths[path_key], failures)
    failures.extend(_validate_specialist_manifest(receipt))
    session_failures, session_evidence = _validate_session_report(
        paths["session_report"], receipt, paths,
    )
    failures.extend(session_failures)
    failures.extend(_validate_specialist_command_paths(receipt, paths))

    attempts: list[dict[str, Any]] = []
    for path_key, suite, kind in (
        ("quick", "quick", "quick"), ("hard", "hard", "hard"),
    ):
        row_failures, rows = _standard_attempts(paths[path_key], suite=suite, kind=kind)
        failures.extend(row_failures)
        attempts.extend(rows)
    for path_key, group, kind, temperature in (
        ("code", "code", "canonical_medium_code", 0.0),
        ("prose", "prose", "canonical_none_prose", 0.7),
    ):
        row_failures, rows = _specialist_attempts(
            paths[path_key], group=group, kind=kind, temperature=temperature,
        )
        failures.extend(row_failures)
        attempts.extend(rows)
    publisher_failures, publisher_rows, publisher_inputs = _publisher_attempts(receipt, paths)
    failures.extend(publisher_failures)
    attempts.extend(publisher_rows)

    call_ids: dict[str, list[dict[str, Any]]] = {}
    for row in attempts:
        call_id = row.get("call_id")
        if isinstance(call_id, str) and call_id:
            call_ids.setdefault(call_id, []).append(row)
    duplicates = {call_id for call_id, rows in call_ids.items() if len(rows) != 1}
    if duplicates:
        failures.append("primary attempt call IDs are not globally unique")
        for row in attempts:
            if row.get("call_id") in duplicates:
                row["valid"] = False
    kinds = (
        "quick", "hard", "canonical_medium_code", "canonical_none_prose",
        "publisher_none_prose",
    )
    counts = {kind: sum(row["valid"] and row["kind"] == kind for row in attempts) for kind in kinds}
    total = sum(counts.values())
    if total != 15:
        failures.append("validated raw primary rows do not account for all 15 frozen attempts")
    evidence_rows = []
    for row in attempts:
        evidence_rows.append({key: value for key, value in row.items() if key != "valid"})
    return list(dict.fromkeys(failures)), {
        "attempt_accounting": {**counts, "total": total},
        "attempts": evidence_rows,
        "session_report": session_evidence,
        "publisher_inputs": publisher_inputs,
        "accounting_note": (
            "Counts come only from exact frozen raw request rows with unique call IDs and terminal "
            "response-or-recorded-request-error metadata. Evaluation failures and legitimate full-cap "
            "attempts remain covered; publisher prose has no additional quality gate."
        ),
        "generated_text_projected_out": True,
        "raw_standard_artifacts_hashed": False,
    }


def _expected_prose() -> dict[str, dict[str, Any]]:
    specialist_cases.assert_frozen_suite()
    return {
        case.case_id: {
            "case_sha256": specialist_cases.case_sha256(case),
            "prompt_sha256": hashlib.sha256(case.prompt.encode("utf-8")).hexdigest(),
            "prompt_chars": len(case.prompt),
        }
        for case in specialist_cases.get_cases(groups=("prose",))
    }


def _metadata_digest(row: Mapping[str, Any]) -> str:
    return _metadata_hash(_content_free_attempt_projection(row))


def _primary_prose_evidence(paths: Sequence[Path]) -> tuple[list[str], list[dict[str, Any]]]:
    failures: list[str] = []
    rows: list[tuple[Path, dict[str, Any]]] = []
    for path in paths:
        for row in _content_free_rows(path, "canonical primary prose metadata"):
            rows.append((path, row))
    expected = _expected_prose()
    if len(rows) != len(expected):
        failures.append("canonical primary prose must contain exactly the frozen first three rows")
    seen: set[str] = set()
    total_wall = 0.0
    evidence: list[dict[str, Any]] = []
    for path, row in rows:
        case_id = row.get("case_id")
        if not isinstance(case_id, str) or case_id not in expected or case_id in seen:
            failures.append("canonical primary prose case IDs are missing, duplicated, or unexpected")
            continue
        seen.add(case_id)
        expected_row = expected[case_id]
        request = row.get("request") if isinstance(row.get("request"), Mapping) else {}
        response = row.get("response") if isinstance(row.get("response"), Mapping) else {}
        metrics = row.get("response_metrics") if isinstance(row.get("response_metrics"), Mapping) else {}
        visible = row.get("visible_text") if isinstance(row.get("visible_text"), Mapping) else {}
        identity_ok = bool(
            row.get("tag") == PARENT_TAG + "-prose"
            and row.get("profile") == PRIMARY_PROFILE
            and row.get("case_group") == "prose"
            and row.get("suite") == "specialist_prose"
            and row.get("case_version") == specialist_cases.CASE_VERSION
            and row.get("case_sha256") == expected_row["case_sha256"]
            and row.get("suite_sha256") == specialist_cases.FROZEN_SUITE_SHA256
            and request.get("prompt_sha256") == expected_row["prompt_sha256"]
            and request.get("prompt_chars") == expected_row["prompt_chars"]
        )
        controls_ok = bool(
            row.get("model") == MODEL
            and request.get("model") == MODEL
            and request.get("profile") == PRIMARY_PROFILE
            and request.get("backend") == "openai"
            and request.get("thinking") == "none"
            and request.get("chat_template_thinking") == "off"
            and request.get("reasoning_budget_tokens") is None
            and request.get("temperature") == 0.7
            and request.get("seed") == 0
            and request.get("context_max_tokens") == CONTEXT
            and request.get("output_token_limit") == OUTPUT_CAP
            and request.get("timeout_seconds") == REQUEST_TIMEOUT
        )
        if not identity_ok:
            failures.append(f"{case_id}: frozen prompt/case identity mismatch")
        if not controls_ok:
            failures.append(f"{case_id}: canonical NONE prose controls mismatch")
        completion_tokens = _integer(
            metrics.get("output_tokens_reported", response.get("actual_completion_tokens"))
        )
        actual_tokens = _integer(response.get("actual_completion_tokens"))
        timing = response.get("timing") if isinstance(response.get("timing"), Mapping) else {}
        wall_seconds = _finite(metrics.get("wall_seconds", timing.get("wall_seconds")))
        finish_reason = metrics.get("finish_reason", response.get("done_reason"))
        visible_chars = _integer(visible.get("chars"))
        visible_bytes = _integer(visible.get("bytes"))
        visible_sha = visible.get("sha256")
        complete = bool(
            row.get("error") is None
            and row.get("objective_error") is None
            and response.get("stream_complete") is True
            and response.get("truncated") is False
            and finish_reason in {"stop", "eos", "end_turn"}
            and completion_tokens is not None and 0 < completion_tokens <= PRIMARY_TOKEN_CEILING
            and actual_tokens is not None and 0 < actual_tokens <= PRIMARY_TOKEN_CEILING
            and wall_seconds is not None and wall_seconds >= 0
            and visible_chars is not None and visible_chars > 0
            and visible_bytes is not None and visible_bytes > 0
            and isinstance(visible_sha, str) and len(visible_sha) == 64
            and visible.get("reasoning_excluded") is True
        )
        if not complete:
            failures.append(f"{case_id}: primary prose is not clean nonempty EOS metadata")
        if wall_seconds is not None and wall_seconds >= 0:
            total_wall += wall_seconds
        evidence.append({
            "path": str(path.resolve()),
            "case_id": case_id,
            "case_sha256": row.get("case_sha256"),
            "prompt_sha256": request.get("prompt_sha256"),
            "metadata_sha256": _metadata_digest(row),
            "visible_text": {
                "sha256": visible_sha, "chars": visible_chars, "bytes": visible_bytes,
                "reasoning_excluded": visible.get("reasoning_excluded"),
            },
            "completion_tokens": completion_tokens,
            "actual_completion_tokens": actual_tokens,
            "wall_seconds": wall_seconds,
            "stream_complete": response.get("stream_complete"),
            "truncated": response.get("truncated"),
            "finish_reason": finish_reason,
            "request_error": row.get("error") is not None,
            "controls_match": controls_ok,
            "identity_match": identity_ok,
        })
    if seen != set(expected):
        failures.append("canonical primary prose does not cover each frozen prose case exactly once")
    if total_wall > PRIMARY_WALL_CEILING:
        failures.append("canonical primary prose total wall time exceeds 3600 seconds")
    for item in evidence:
        item["prose3_total_wall_seconds"] = round(total_wall, 6)
    return list(dict.fromkeys(failures)), evidence


def validate_boundary_ready(path: Path, receipt_path: Path) -> dict[str, Any]:
    ready = _json(path, "Style model boundary READY receipt")
    expected_release = path.with_name(path.name.removesuffix("-ready.json") + "-release.json")
    expected_name = "specialists-final72k64k-gemma4-style12-q5_k_m-ready.json"
    study_identity = ready.get("study") if isinstance(ready.get("study"), Mapping) else {}
    if (
        path.name != expected_name
        or ready.get("schema_version") != 1
        or ready.get("kind") != "model_boundary_ready"
        or ready.get("run_prefix") != "specialists-final72k64k"
        or ready.get("model_key") != PRIMARY_PROFILE
        or Path(str(study_identity.get("path"))).resolve() != receipt_path.resolve()
        or study_identity.get("sha256") != sha256_file(receipt_path)
    ):
        raise ProbeError("Style READY boundary does not match the original primary receipt")
    if os.path.lexists(expected_release):
        raise ProbeError("Style READY boundary has already been released")
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "kind": ready.get("kind"),
        "run_prefix": ready.get("run_prefix"),
        "model_key": ready.get("model_key"),
        "study": dict(study_identity),
        "release_path": str(expected_release),
        "release_absent": True,
        "state": "ready_awaiting_release",
    }


def validate_eligibility(
    receipt_path: Path, prose_paths: Sequence[Path], boundary_ready_path: Path,
) -> Eligibility:
    specialist_cases.assert_frozen_suite()
    receipt = _json(receipt_path, "primary study receipt")
    receipt_failures, receipt_evidence = _primary_receipt_evidence(receipt)
    attempt_failures: list[str] = []
    attempt_evidence: dict[str, Any] = {
        "attempt_accounting": {
            "quick": 0, "hard": 0, "canonical_medium_code": 0,
            "canonical_none_prose": 0, "publisher_none_prose": 0, "total": 0,
        }
    }
    prose_failures: list[str] = []
    prose_evidence: list[dict[str, Any]] = []
    if receipt_failures:
        attempt_failures.append(
            "primary raw attempt validation deferred until the original lifecycle is terminal and clean"
        )
    else:
        try:
            attempt_failures, attempt_evidence = _primary_attempt_evidence(
                receipt_path, receipt, prose_paths,
            )
        except (ProbeError, OSError, RuntimeError) as exc:
            attempt_failures.append(f"primary raw attempt validation failed: {exc}")
        try:
            prose_failures, prose_evidence = _primary_prose_evidence(prose_paths)
        except (ProbeError, OSError, RuntimeError) as exc:
            prose_failures.append(f"canonical primary prose validation failed: {exc}")
    boundary_failures: list[str] = []
    boundary: dict[str, Any] = {"path": str(boundary_ready_path)}
    try:
        boundary = validate_boundary_ready(boundary_ready_path, receipt_path)
    except (ProbeError, OSError) as exc:
        boundary_failures.append(f"primary READY boundary validation failed: {exc}")
    failures = tuple(receipt_failures + attempt_failures + prose_failures + boundary_failures)
    return Eligibility(
        eligible=not failures,
        failures=failures,
        primary_receipt={
            "path": str(receipt_path.resolve()),
            "sha256": sha256_file(receipt_path),
            **receipt_evidence,
            **attempt_evidence,
        },
        primary_prose=tuple(prose_evidence),
        boundary=boundary,
    )


def outer_command(tag: str, control: Path) -> list[str]:
    args = argparse.Namespace(
        runtime_backend="llama", config=CONFIG, plan=SMOKE_PLAN, tag=tag,
        startup_timeout=STARTUP_TIMEOUT, wait_for_idle=IDLE_WAIT_SECONDS,
        max_seconds=WINDOW_SECONDS,
    )
    return study.gpu_window_command(args, control)


def sampler_command(tag: str, prefix: Path, run_id: str) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "runtime/sampler_recipe_probe.py"),
        "--recipe", str(RECIPE),
        "--runtime-events", str(RESULTS / f"{tag}-runtime-events.jsonl"),
        "--tag", tag,
        "--model-session-run-id", run_id,
        "--endpoint", ENDPOINT,
        "--model", MODEL,
        "--expected-context", str(CONTEXT),
        "--output-prefix", str(prefix),
        "--profile", CANDIDATE_PROFILE,
        "--groups", "prose",
        "--prose-thinking", "none",
        "--output-cap", str(OUTPUT_CAP),
        "--timeout", str(int(REQUEST_TIMEOUT)),
    ]


def _port_clear(port: int) -> bool:
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.settimeout(0.2)
    try:
        listening = probe.connect_ex(("127.0.0.1", port)) == 0
    finally:
        probe.close()
    if listening:
        return False
    binder = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        binder.bind(("127.0.0.1", port))
    except OSError:
        return False
    finally:
        binder.close()
    return True


def port_evidence() -> dict[str, bool]:
    return {"native_18081_clear": _port_clear(NATIVE_PORT), "strata_18080_clear": _port_clear(STRATA_PORT)}


def assert_ports_clear() -> dict[str, bool]:
    evidence = port_evidence()
    if not evidence["native_18081_clear"]:
        raise ProbeError(f"refusing occupied native loopback port {NATIVE_PORT}")
    if not evidence["strata_18080_clear"]:
        raise ProbeError(f"refusing occupied Strata loopback port {STRATA_PORT}")
    return evidence


def wait_for_holding(
    outer: subprocess.Popen[Any], paths: ProbePaths, deadline: float,
) -> study.HoldingSession:
    while time.monotonic() < deadline:
        returncode = outer.poll()
        if returncode is not None:
            raise ProbeError(f"GPU window exited {returncode} before exact held session")
        try:
            rows = study._read_event_rows(paths.events)
            if any(
                row.get("event") in {
                    "holding_stopped", "server_stopped", "server_cleanup_failed",
                    "session_failed", "session_interrupted", "session_complete",
                }
                for row in rows
            ):
                raise ProbeError("runtime events describe a stale or terminal held session")
            identity = study._holding_identity(
                rows, tag=paths.control.name.removesuffix("-control.json"),
                source_config=CONFIG, plan=SMOKE_PLAN, control=paths.control,
                runtime_backend="llama",
            )
        except study.StudyError as exc:
            raise ProbeError(str(exc)) from exc
        if identity is not None:
            if identity.model_name != MODEL or identity.context != CONTEXT:
                raise ProbeError("held session model/context identity mismatch")
            return identity
        time.sleep(0.5)
    raise ProbeError("whole-window guard expired before the exact held session")


def validate_smoke(path: Path, run_id: str, *, events_path: Path) -> dict[str, Any]:
    events = study._read_event_rows(events_path)
    completions = [row for row in events if row.get("event") == "step_complete"]
    if (
        any(row.get("run_id") != run_id for row in events)
        or len(completions) != 1
        or completions[0].get("id") != "poll0_smoke_json_none_256"
        or completions[0].get("result") != str(path)
        or completions[0].get("returncode") != 0
        or completions[0].get("passed") is not True
    ):
        raise ProbeError("poll0 smoke is not tied to the exact held model_session lifecycle")
    rows = _jsonl(path, "poll0 smoke result")
    if len(rows) != 1:
        raise ProbeError("poll0 smoke did not produce exactly one result")
    row = rows[0]
    request = row.get("request") if isinstance(row.get("request"), Mapping) else {}
    response = row.get("response") if isinstance(row.get("response"), Mapping) else {}
    evaluation = row.get("evaluation") if isinstance(row.get("evaluation"), Mapping) else {}
    if (
        row.get("suite") != "quick"
        or row.get("case_id") != "quick_json_extract"
        or request.get("model") != MODEL
        or request.get("context_max_tokens") != CONTEXT
        or request.get("thinking") != "none"
        or request.get("chat_template_thinking") != "off"
        or request.get("temperature") != 0
        or request.get("seed") != 0
        or request.get("output_token_limit") != 256
        or request.get("timeout_seconds") != 180
        or row.get("error") is not None
        or response.get("stream_complete") is not True
        or response.get("truncated") is True
        or evaluation.get("passed") is not True
    ):
        raise ProbeError("poll0 smoke failed identity, correctness, completion, or truncation gate")
    return {
        "path": str(path), "case_id": row.get("case_id"),
        "benchmark_run_id": row.get("run_id"), "model_session_run_id": run_id,
        "passed": True, "stream_complete": True, "truncated": False,
        "request_error": False,
    }


def _terminate_owned(process: subprocess.Popen[Any]) -> dict[str, Any]:
    result = {
        "pid": process.pid, "pgid": process.pid, "term_sent": False,
        "kill_sent": False, "stopped": process.poll() is not None,
    }
    if result["stopped"]:
        return result
    try:
        os.killpg(process.pid, signal.SIGTERM)
        result["term_sent"] = True
    except ProcessLookupError:
        result["stopped"] = True
        return result
    until = time.monotonic() + PROCESS_TERM_SECONDS
    while process.poll() is None and time.monotonic() < until:
        time.sleep(0.1)
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGKILL)
            result["kill_sent"] = True
        except ProcessLookupError:
            pass
        until = time.monotonic() + 5.0
        while process.poll() is None and time.monotonic() < until:
            time.sleep(0.1)
    result["stopped"] = process.poll() is not None
    return result


def run_owned(command: Sequence[str], deadline: float) -> tuple[subprocess.Popen[Any], int]:
    process = subprocess.Popen(
        list(command), stdin=subprocess.DEVNULL, start_new_session=True, close_fds=True,
    )
    try:
        while process.poll() is None:
            if time.monotonic() >= deadline:
                raise ProbeError("whole-window guard expired while owned prose helper was running")
            time.sleep(0.5)
    except BaseException as exc:
        cleanup = _terminate_owned(process) if process.poll() is None else {
            "pid": process.pid, "pgid": process.pid, "term_sent": False,
            "kill_sent": False, "stopped": True,
        }
        setattr(exc, "owned_process", process)
        setattr(exc, "owned_cleanup", cleanup)
        raise
    return process, int(process.returncode)


def _request_metadata_summary(path: Path, *, tag: str) -> dict[str, Any]:
    rows = _content_free_rows(path, "poll0 prose request metadata")
    expected = _expected_prose()
    if len(rows) != len(expected):
        raise ProbeError("poll0 prose request metadata must contain exactly three rows")
    seen: set[str] = set()
    summaries: list[dict[str, Any]] = []
    for row in rows:
        case_id = row.get("case_id")
        request = row.get("request") if isinstance(row.get("request"), Mapping) else {}
        response = row.get("response") if isinstance(row.get("response"), Mapping) else {}
        metrics = row.get("response_metrics") if isinstance(row.get("response_metrics"), Mapping) else {}
        if (
            not isinstance(case_id, str) or case_id not in expected or case_id in seen
            or row.get("tag") != tag or row.get("profile") != CANDIDATE_PROFILE
            or row.get("model") != MODEL or row.get("case_group") != "prose"
            or row.get("case_sha256") != expected.get(case_id, {}).get("case_sha256")
            or request.get("prompt_sha256") != expected.get(case_id, {}).get("prompt_sha256")
            or request.get("model") != MODEL or request.get("profile") != CANDIDATE_PROFILE
            or request.get("thinking") != "none"
            or request.get("chat_template_thinking") != "off"
            or request.get("temperature") != 0.7 or request.get("seed") != 0
            or request.get("context_max_tokens") != CONTEXT
            or request.get("output_token_limit") != OUTPUT_CAP
            or request.get("timeout_seconds") != REQUEST_TIMEOUT
            or request.get("sampler_recipe_sha256") != RECIPE_SHA256
        ):
            raise ProbeError("poll0 prose request metadata identity or controls mismatch")
        seen.add(case_id)
        native = response.get("openai_native_timings") if isinstance(response.get("openai_native_timings"), Mapping) else {}
        summaries.append({
            "case_id": case_id, "call_id": row.get("call_id"),
            "stream_complete": response.get("stream_complete"),
            "truncated": response.get("truncated"),
            "finish_reason": metrics.get("finish_reason", response.get("done_reason")),
            "output_tokens": metrics.get("output_tokens_reported", response.get("actual_completion_tokens")),
            "wall_seconds": metrics.get("wall_seconds"),
            "native_decode_seconds": native.get("predicted_ms") / 1000.0
            if _finite(native.get("predicted_ms")) is not None else None,
            "native_decode_tps": native.get("predicted_per_second"),
            "request_error_present": row.get("error") is not None,
        })
    if seen != set(expected):
        raise ProbeError("poll0 prose request metadata lacks exact frozen prose coverage")
    return {
        "path": str(path), "sha256": sha256_file(path), "row_count": len(rows),
        "cases": summaries,
    }


def summarize_sampler_receipt(
    path: Path, run_id: str, returncode: int, *, expected_tag: str, requests_path: Path,
) -> dict[str, Any]:
    receipt = _json(path, "poll0 sampler receipt")
    controls = receipt.get("controls") if isinstance(receipt.get("controls"), Mapping) else {}
    recipe = receipt.get("recipe") if isinstance(receipt.get("recipe"), Mapping) else {}
    attempted = _integer(receipt.get("attempted_count"))
    completed = _integer(receipt.get("completed_count"))
    truncated = _integer(receipt.get("truncated_count"))
    failures = _integer(receipt.get("request_failure_count"))
    status = receipt.get("status")
    counts_ok = bool(
        attempted == 3 and completed is not None and completed >= 0
        and truncated is not None and truncated >= 0
        and failures is not None and failures >= 0
        and completed + truncated + failures == 3
    )
    returncode_ok = (status == "complete" and returncode == 0 and failures == 0) or (
        status == "completed_with_request_failures" and returncode == 1
        and failures is not None and failures > 0
    )
    if not (
        receipt.get("tag") == expected_tag
        and receipt.get("model") == MODEL
        and receipt.get("model_session_run_id") == run_id
        and receipt.get("runtime_backend") == "llama"
        and controls.get("prose_thinking") == "none"
        and controls.get("output_token_limit") == OUTPUT_CAP
        and controls.get("timeout_seconds") == REQUEST_TIMEOUT
        and controls.get("seed") == 0
        and controls.get("expected_context_tokens") == CONTEXT
        and controls.get("actual_context_tokens") == CONTEXT
        and counts_ok and returncode_ok
        and recipe.get("path") == str(RECIPE)
        and recipe.get("sha256") == RECIPE_SHA256
        and recipe.get("sampler_overrides") == {"temperature": 0.7}
    ):
        raise ProbeError("poll0 prose sampler terminal receipt or controls are invalid")
    attempts = receipt.get("attempts") if isinstance(receipt.get("attempts"), list) else []
    if len(attempts) != 3:
        raise ProbeError("poll0 prose sampler terminal receipt lacks three attempt summaries")
    return {
        "path": str(path), "sha256": sha256_file(path), "status": status,
        "terminal_sampler_receipt_validated": True,
        "attempted_count": attempted, "completed_count": completed,
        "truncated_count": truncated, "request_failure_count": failures,
        "attempts": [
            {key: row.get(key) for key in (
                "call_id", "case_id", "outcome", "stream_complete", "truncated",
                "input_tokens", "output_tokens", "wall_seconds", "native_decode_seconds",
                "native_decode_tps", "error_type",
            )}
            for row in attempts if isinstance(row, Mapping)
        ],
        "request_metadata": _request_metadata_summary(requests_path, tag=expected_tag),
        "controls": {
            key: controls.get(key) for key in (
                "prose_thinking", "context_max_tokens", "expected_context_tokens",
                "actual_context_tokens", "output_token_limit", "timeout_seconds", "seed",
            )
        },
    }


def _observed_prose_rows(path: Path) -> dict[str, Any]:
    evidence: dict[str, Any] = {
        "path": str(path), "exists": False, "sha256": None,
        "observed_row_count": 0, "observed_completed_row_count": 0,
        "observed_request_error_row_count": 0, "rows": [],
    }
    try:
        details = path.lstat()
        if not stat.S_ISREG(details.st_mode) or path.is_symlink():
            evidence["observation_error"] = "not_a_regular_non_symlink_file"
            return evidence
        evidence["exists"] = True
        evidence["size"] = details.st_size
        if details.st_size > MAX_JSONL_BYTES:
            evidence["observation_error"] = "file_exceeds_metadata_bound"
            return evidence
        raw = path.read_bytes()
    except FileNotFoundError:
        return evidence
    except OSError:
        evidence["observation_error"] = "file_unreadable"
        return evidence
    parsed: list[dict[str, Any]] = []
    invalid = 0
    content_bearing = False
    for raw_line in raw.splitlines(keepends=True):
        if not raw_line.endswith((b"\n", b"\r")):
            evidence["ignored_incomplete_trailing_bytes"] = len(raw_line)
            continue
        try:
            row = json.loads(raw_line)
        except (UnicodeError, json.JSONDecodeError):
            invalid += 1
            continue
        if not isinstance(row, dict):
            invalid += 1
            continue
        response = row.get("response") if isinstance(row.get("response"), Mapping) else {}
        if _contains_generated_text(row):
            invalid += 1
            content_bearing = True
            continue
        metrics = row.get("response_metrics") if isinstance(row.get("response_metrics"), Mapping) else {}
        error_present = row.get("error") is not None
        complete = response.get("stream_complete") is True and not error_present
        parsed.append({
            "case_id": row.get("case_id"), "call_id": row.get("call_id"),
            "stream_complete": response.get("stream_complete"),
            "truncated": response.get("truncated"), "request_error_present": error_present,
            "output_tokens": metrics.get("output_tokens_reported", response.get("actual_completion_tokens")),
            "wall_seconds": metrics.get("wall_seconds"), "observed_complete": complete,
        })
    if not content_bearing:
        evidence["sha256"] = hashlib.sha256(raw).hexdigest()
    else:
        evidence["observation_error"] = "content_bearing_rows_ignored"
    evidence["observed_row_count"] = len(parsed)
    evidence["observed_completed_row_count"] = sum(row["observed_complete"] is True for row in parsed)
    evidence["observed_request_error_row_count"] = sum(row["request_error_present"] is True for row in parsed)
    evidence["invalid_or_content_bearing_complete_row_count"] = invalid
    evidence["rows"] = parsed
    return evidence


def _unvalidated_sampler_receipt(path: Path) -> dict[str, Any]:
    evidence: dict[str, Any] = {"path": str(path), "exists": False}
    try:
        value = _json(path, "unvalidated poll0 sampler receipt")
    except ProbeError:
        if os.path.lexists(path):
            evidence.update({"exists": True, "observation_error": "not_valid_bounded_receipt"})
        return evidence
    evidence.update({
        "exists": True, "sha256": sha256_file(path),
        "reported_status": value.get("status"),
        "reported_counts": {
            key: value.get(key) if _integer(value.get(key)) is not None else None
            for key in ("attempted_count", "completed_count", "truncated_count", "request_failure_count")
        },
        "terminal_receipt_validated": False,
    })
    return evidence


def record_unvalidated_prose(
    receipt: dict[str, Any], paths: ProbePaths, process: subprocess.Popen[Any],
    *, status: str, stop_cause: str,
) -> None:
    if status not in {"interrupted", "incomplete"}:
        raise ProbeError("unvalidated prose status must be interrupted or incomplete")
    receipt["execution"]["prose_helper_pid"] = process.pid
    receipt["execution"]["prose_helper_returncode"] = process.poll()
    receipt["prose"] = {
        "status": status, "helper_launched": True,
        "terminal_sampler_receipt_validated": False,
        "attempted_count": None, "completed_count": None,
        "truncated_count": None, "request_failure_count": None,
        "stop_cause": stop_cause,
        "observed_request_metadata": _observed_prose_rows(paths.sampler_requests),
        "unvalidated_sampler_receipt": _unvalidated_sampler_receipt(paths.sampler_receipt),
        "note": (
            "Official counts remain unknown without a validated terminal sampler receipt; "
            "observed content-free rows are separate metadata only."
        ),
    }


def _server_identity(events_path: Path, run_id: str) -> dict[str, Any]:
    rows = study._read_event_rows(events_path)
    starts = [row for row in rows if row.get("event") == "session_started"]
    servers = [row for row in rows if row.get("event") == "server_started"]
    if len(starts) != 1 or len(servers) != 1 or any(row.get("run_id") != run_id for row in rows):
        raise ProbeError("runtime events do not preserve one exact owned session/server identity")
    start = starts[0]
    server = servers[0]
    argv = server.get("argv") if isinstance(server.get("argv"), list) else []
    poll_positions = [index for index, value in enumerate(argv) if value == "--poll"]
    if not (
        start.get("source_config") == str(CONFIG)
        and start.get("source_config_sha256") == CONFIG_SHA256
        and start.get("plan") == str(SMOKE_PLAN)
        and start.get("plan_sha256") == SMOKE_PLAN_SHA256
        and start.get("runtime_backend") == "llama"
        and start.get("model_name") == MODEL
        and start.get("context") == CONTEXT
        and len(poll_positions) == 1
        and poll_positions[0] + 1 < len(argv)
        and argv[poll_positions[0] + 1] == "0"
        and not any("draft" in str(item).lower() or "mtp" in str(item).lower() for item in argv)
    ):
        raise ProbeError("held runtime identity does not prove the pinned poll0 candidate")
    return {
        "run_id": run_id,
        "model_session_pid": next(
            (row.get("pid") for row in rows if row.get("event") == "session_process_identity"), None
        ),
        "server_pid": server.get("pid"), "server_pgid": server.get("pgid"),
        "server_argv": argv, "source_config": start.get("source_config"),
        "source_config_sha256": start.get("source_config_sha256"),
        "plan": start.get("plan"), "plan_sha256": start.get("plan_sha256"),
        "runtime_backend": start.get("runtime_backend"), "model": start.get("model_name"),
        "context": start.get("context"), "events_path": str(events_path),
        "events_sha256": sha256_file(events_path),
    }


def procfs_cpu_snapshot(pid: Any, expected_start_ticks: int | None = None) -> dict[str, Any]:
    evidence: dict[str, Any] = {
        "available": False, "pid": pid if _integer(pid) is not None else None,
        "captured_at": utc_now(), "monotonic_ns": time.monotonic_ns(),
        "source": "linux_procfs_process_stat_content_free",
    }
    if _integer(pid) is None or pid <= 0:
        evidence["error"] = "invalid_server_pid"
        return evidence
    try:
        payload = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        closing = payload.rfind(")")
        if closing < 0:
            raise ValueError("missing comm terminator")
        fields = payload[closing + 1:].split()
        user_ticks = int(fields[11])
        system_ticks = int(fields[12])
        start_ticks = int(fields[19])
        processor = int(fields[36]) if len(fields) > 36 else None
        clock_ticks = int(os.sysconf("SC_CLK_TCK"))
    except FileNotFoundError:
        evidence["error"] = "server_process_not_found"
        return evidence
    except (OSError, ValueError, IndexError) as exc:
        evidence["error"] = f"procfs_stat_unavailable:{type(exc).__name__}"
        return evidence
    if expected_start_ticks is not None and start_ticks != expected_start_ticks:
        evidence.update({"start_ticks": start_ticks, "error": "server_pid_identity_changed"})
        return evidence
    evidence.update({
        "available": True, "start_ticks": start_ticks, "user_ticks": user_ticks,
        "system_ticks": system_ticks, "cpu_ticks": user_ticks + system_ticks,
        "clock_ticks_per_second": clock_ticks, "last_processor": processor,
    })
    return evidence


def summarize_cpu_snapshots(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {"before": dict(before), "after": dict(after), "comparable": False}
    if not (
        before.get("available") is True and after.get("available") is True
        and before.get("pid") == after.get("pid")
        and before.get("start_ticks") == after.get("start_ticks")
        and before.get("clock_ticks_per_second") == after.get("clock_ticks_per_second")
    ):
        summary["note"] = "CPU evidence unavailable or process identity changed"
        return summary
    delta_ticks = int(after["cpu_ticks"]) - int(before["cpu_ticks"])
    elapsed = (int(after["monotonic_ns"]) - int(before["monotonic_ns"])) / 1_000_000_000
    if delta_ticks < 0 or elapsed <= 0:
        summary["note"] = "CPU counters or elapsed time are invalid"
        return summary
    cpu_seconds = delta_ticks / int(before["clock_ticks_per_second"])
    summary.update({
        "comparable": True, "delta_cpu_ticks": delta_ticks,
        "delta_cpu_seconds": cpu_seconds, "elapsed_wall_seconds": elapsed,
        "mean_logical_cpu_equivalents": cpu_seconds / elapsed,
        "claim_scope": "process CPU use only; not proof of a GPU bottleneck or causal speedup",
    })
    return summary


def restoration_evidence() -> dict[str, Any]:
    evidence = study.restoration_evidence("llama")
    state = gpu_window.original_state()
    health = state.get("Health") if isinstance(state.get("Health"), Mapping) else {}
    ports = port_evidence()
    evidence.update({
        "original_healthy": health.get("Status") == "healthy",
        "original_health_status": health.get("Status"),
        "ports": ports,
        "both_owned_ports_clear": all(ports.values()),
    })
    return evidence


def session_cleanup_evidence(
    paths: ProbePaths, *, run_id: str, outer_returncode: int | None,
    restoration: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], list[str]]:
    failures: list[str] = []
    report = _json(paths.session_report, "poll0 model_session report")
    server = report.get("server") if isinstance(report.get("server"), Mapping) else {}
    cleanup = server.get("cleanup") if isinstance(server.get("cleanup"), Mapping) else {}
    steps = report.get("steps") if isinstance(report.get("steps"), list) else []
    if not (
        report.get("session_id") == run_id
        and report.get("tag") == paths.control.name.removesuffix("-control.json")
        and report.get("runtime_backend") == "llama"
        and report.get("status") == "passed"
        and report.get("source_config") == str(CONFIG)
        and report.get("source_config_sha256") == CONFIG_SHA256
        and isinstance(report.get("plan"), Mapping)
        and report["plan"].get("sha256") == SMOKE_PLAN_SHA256
        and server.get("model_name") == MODEL and server.get("context") == CONTEXT
        and len(steps) == 1 and isinstance(steps[0], Mapping)
        and steps[0].get("id") == "poll0_smoke_json_none_256"
        and steps[0].get("passed") is True
    ):
        failures.append("model_session report identity or smoke completion is invalid")
    if cleanup.get("stopped") is not True or cleanup.get("kill_sent") is not False:
        failures.append("owned native server did not record clean non-SIGKILL shutdown")
    if outer_returncode != 0:
        failures.append("GPU window did not exit successfully")
    if not isinstance(restoration, Mapping) or not (
        restoration.get("original_running") is True
        and restoration.get("original_paused") is False
        and restoration.get("original_healthy") is True
        and restoration.get("both_owned_ports_clear") is True
    ):
        failures.append("original healthy restoration or owned-port cleanup is unverified")
    events = study._read_event_rows(paths.events)
    if not (
        any(row.get("event") == "server_stopped" for row in events)
        and any(row.get("event") == "session_complete" for row in events)
        and all(row.get("run_id") == run_id for row in events)
    ):
        failures.append("runtime event ledger lacks exact terminal cleanup events")
    return {
        "session_report": {
            "path": str(paths.session_report), "sha256": sha256_file(paths.session_report),
            "status": report.get("status"), "session_id": report.get("session_id"),
            "source_config_sha256": report.get("source_config_sha256"),
            "plan_sha256": report.get("plan", {}).get("sha256")
            if isinstance(report.get("plan"), Mapping) else None,
        },
        "native_server": {
            "stopped": cleanup.get("stopped"), "term_sent": cleanup.get("term_sent"),
            "kill_sent": cleanup.get("kill_sent"), "pgid": cleanup.get("pgid"),
        },
        "runtime_events_sha256": sha256_file(paths.events),
    }, failures


def execute_probe(
    paths: ProbePaths,
    tag: str,
    receipt: dict[str, Any],
    *,
    boundary_ready_path: Path | None = None,
    primary_receipt_path: Path | None = None,
) -> int:
    if os.path.lexists(paths.control):
        raise ProbeError("control path must not exist before model_session starts")
    assert_unused(paths)
    if boundary_ready_path is not None and primary_receipt_path is not None:
        receipt["eligibility"]["family_boundary"] = validate_boundary_ready(
            boundary_ready_path, primary_receipt_path,
        )
    receipt["execution"]["port_preflight"] = assert_ports_clear()
    sampler_recipe_probe.validate_recipe(_json(RECIPE, "pinned poll0 prose recipe"))
    outer_argv = outer_command(tag, paths.control)
    receipt["commands"]["gpu_window"] = outer_argv
    started = time.monotonic()
    deadline = started + WINDOW_SECONDS - RESTORATION_RESERVE_SECONDS
    outer: subprocess.Popen[Any] | None = None
    prose: subprocess.Popen[Any] | None = None
    session: study.HoldingSession | None = None
    runtime_identity: dict[str, Any] | None = None
    cpu_before: dict[str, Any] | None = None
    stop_cause = "setup_failure"
    problem: dict[str, Any] | None = None
    prose_unvalidated_status: str | None = None
    terminal_profile_status: str | None = None
    code = 1
    signals = SignalLatch()
    previous_handlers: dict[int, Any] = {}
    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, signals.handle)
        outer = subprocess.Popen(
            outer_argv, stdin=subprocess.DEVNULL, start_new_session=True, close_fds=True,
        )
        receipt["execution"]["launched"] = True
        receipt["execution"]["gpu_window_pid"] = outer.pid
        session = wait_for_holding(outer, paths, deadline)
        receipt["execution"]["session"] = {
            "run_id": session.run_id, "model": session.model_name, "context": session.context,
            "initial_load_seconds": session.initial_load_seconds,
            "session_started_at": session.session_started_at,
            "server_ready_at": session.server_ready_at,
            "holding_started_at": session.holding_started_at,
        }
        runtime_identity = _server_identity(paths.events, session.run_id)
        receipt["execution"]["runtime_identity"] = runtime_identity
        cpu_before = procfs_cpu_snapshot(runtime_identity.get("server_pid"))
        receipt["observed_resources"]["server_cpu"] = {"before": cpu_before}
        receipt["smoke"] = validate_smoke(paths.smoke, session.run_id, events_path=paths.events)
        command = sampler_command(tag, paths.sampler_prefix, session.run_id)
        receipt["commands"]["sampler_recipe_probe"] = command
        prose, returncode = run_owned(command, deadline)
        receipt["execution"]["prose_helper_pid"] = prose.pid
        receipt["execution"]["prose_helper_returncode"] = returncode
        receipt["prose"] = summarize_sampler_receipt(
            paths.sampler_receipt, session.run_id, returncode,
            expected_tag=tag, requests_path=paths.sampler_requests,
        )
        terminal_profile_status = str(receipt["prose"]["status"])
        stop_cause = "probe_complete"
        code = 0 if receipt["prose"]["request_failure_count"] == 0 else 1
    except ProbeInterrupted as exc:
        prose = getattr(exc, "owned_process", prose)
        if prose is not None:
            prose_unvalidated_status = "interrupted"
        if hasattr(exc, "owned_cleanup"):
            receipt["cleanup"]["prose_helper"] = getattr(exc, "owned_cleanup")
        problem = {"error_type": type(exc).__name__, "error": str(exc), "signal": exc.signum}
        stop_cause = "interrupted"
        code = 128 + exc.signum
    except (ProbeError, study.StudyError, OSError, subprocess.SubprocessError) as exc:
        prose = getattr(exc, "owned_process", prose)
        if prose is not None:
            prose_unvalidated_status = "interrupted" if "guard expired" in str(exc) else "incomplete"
        if hasattr(exc, "owned_cleanup"):
            receipt["cleanup"]["prose_helper"] = getattr(exc, "owned_cleanup")
        problem = {"error_type": type(exc).__name__, "error": str(exc)}
        stop_cause = "profile_guard" if "guard expired" in str(exc) else "probe_failure"
    finally:
        signals.cleaning = True
        if prose is not None and prose.poll() is None:
            receipt["cleanup"]["prose_helper"] = _terminate_owned(prose)
        elif prose is not None:
            receipt["cleanup"].setdefault("prose_helper", {
                "pid": prose.pid, "pgid": prose.pid, "term_sent": False,
                "kill_sent": False, "stopped": True,
            })
        helper_cleanup = receipt["cleanup"].get("prose_helper")
        helper_stopped = not isinstance(helper_cleanup, Mapping) or helper_cleanup.get("stopped") is True
        receipt["cleanup"]["prose_helper_stopped_before_control"] = helper_stopped
        if prose is not None and prose_unvalidated_status is not None and not receipt.get("prose", {}).get("terminal_sampler_receipt_validated"):
            record_unvalidated_prose(
                receipt, paths, prose, status=prose_unvalidated_status, stop_cause=stop_cause,
            )
        if session is not None and runtime_identity is not None:
            expected_ticks = cpu_before.get("start_ticks") if isinstance(cpu_before, Mapping) else None
            cpu_after = procfs_cpu_snapshot(runtime_identity.get("server_pid"), expected_ticks)
            receipt["observed_resources"]["server_cpu"] = summarize_cpu_snapshots(
                cpu_before or {}, cpu_after,
            )
        if outer is not None:
            if not helper_stopped:
                problem = problem or {
                    "error_type": "ProbeError",
                    "error": "owned prose helper did not stop; control stop was not published",
                }
            elif session is not None and os.path.lexists(paths.control):
                try:
                    receipt["cleanup"]["control_stop_requested"] = study.request_session_stop(paths.control)
                except (study.StudyError, OSError) as exc:
                    problem = problem or {"error_type": type(exc).__name__, "error": str(exc)}
            elif outer.poll() is None:
                try:
                    receipt["cleanup"]["owned_outer_sigterm_sent"] = study.request_outer_termination(outer)
                except OSError as exc:
                    problem = problem or {"error_type": type(exc).__name__, "error": str(exc)}
            outer_returncode, delayed_stop = study.wait_for_outer_exit(
                outer, timeout=study.RESTORATION_WAIT_SECONDS,
                control=paths.control if helper_stopped else None,
            )
            receipt["cleanup"]["control_stop_requested"] |= delayed_stop
            receipt["cleanup"]["outer_returncode"] = outer_returncode
            receipt["cleanup"]["outer_exited"] = outer_returncode is not None
            if outer_returncode is not None:
                try:
                    receipt["restoration"] = restoration_evidence()
                except (OSError, RuntimeError) as exc:
                    problem = problem or {"error_type": type(exc).__name__, "error": str(exc)}
            else:
                problem = problem or {
                    "error_type": "ProbeError",
                    "error": "owned GPU window did not exit during restoration wait; no SIGKILL sent",
                }
            if session is not None and outer_returncode is not None:
                try:
                    receipt["execution"]["runtime_identity"] = _server_identity(paths.events, session.run_id)
                    cleanup_evidence, cleanup_failures = session_cleanup_evidence(
                        paths, run_id=session.run_id, outer_returncode=outer_returncode,
                        restoration=receipt.get("restoration"),
                    )
                    receipt["cleanup"].update(cleanup_evidence)
                    if cleanup_failures:
                        code = 1
                        problem = problem or {
                            "error_type": "ProbeError", "error": "; ".join(cleanup_failures),
                        }
                except (ProbeError, study.StudyError, OSError) as exc:
                    code = 1
                    problem = problem or {"error_type": type(exc).__name__, "error": str(exc)}
        receipt["cleanup"]["stop_cause"] = stop_cause
        receipt["cleanup"]["orchestrator_sigkill_sent"] = False
        receipt["problem"] = problem
        receipt["execution"]["wall_seconds"] = round(time.monotonic() - started, 3)
        if problem is not None and problem.get("error_type") != "ProbeInterrupted":
            code = 1
        receipt["status"] = (
            "complete" if terminal_profile_status == "complete" and problem is None
            else "completed_with_request_failures"
            if terminal_profile_status == "completed_with_request_failures" and problem is None
            else "interrupted" if problem and problem.get("error_type") == "ProbeInterrupted"
            else "infrastructure_failed"
        )
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
    return code


def build_receipt(
    *, args: argparse.Namespace, paths: ProbePaths, pins: Mapping[str, Any],
    eligibility: Eligibility,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "diagnostic_kind": "style_gemma4_native_poll0_prose_probe",
        "created_at": utc_now(), "mode": "execute" if args.execute else "inspect",
        "status": "eligible_not_launched" if eligibility.eligible else "ineligible_not_launched",
        "tag": args.tag, "output_prefix": str(paths.prefix),
        "wrapper_source": {"path": str(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__).resolve())},
        "pinned_inputs": pins,
        "eligibility": {
            "eligible": eligibility.eligible, "failures": list(eligibility.failures),
            "selection_basis": "first terminal completion/control/hash metadata only",
            "subjective_writing_preselection": False,
            "primary_study": dict(eligibility.primary_receipt),
            "primary_prose": [dict(row) for row in eligibility.primary_prose],
            "family_boundary": dict(eligibility.boundary),
        },
        "controls": {
            "runtime_backend": "llama", "profile": CANDIDATE_PROFILE,
            "candidate_native_args_appended": ["--poll", "0"], "mtp_or_draft_enabled": False,
            "context": CONTEXT, "thinking": "none", "temperature": 0.7, "seed": 0,
            "output_cap": OUTPUT_CAP, "per_request_timeout_seconds": REQUEST_TIMEOUT,
            "whole_window_guard_seconds": WINDOW_SECONDS,
            "idle_wait_seconds": IDLE_WAIT_SECONDS, "startup_timeout_seconds": STARTUP_TIMEOUT,
            "groups": ["prose"], "case_count": 3, "maximum_new_requests": 4,
        },
        "commands": {
            "gpu_window": outer_command(args.tag, paths.control),
            "sampler_recipe_probe_template": sampler_command(
                args.tag, paths.sampler_prefix, "<exact-held-run-id>"
            ),
        },
        "execution": {"launched": False, "launched_request_limit": 4},
        "smoke": {"status": "not_run", "attempted_count": 0},
        "prose": {
            "status": "not_run", "helper_launched": False, "attempted_count": 0,
            "completed_count": 0, "truncated_count": 0, "request_failure_count": 0,
        },
        "unexecuted_cases": {
            "hard": {"status": "not_run", "attempted_count": 0},
            "code": {"status": "not_run", "attempted_count": 0},
        },
        "observed_resources": {
            "server_cpu": {"status": "not_run"},
            "claim_scope": "process CPU evidence is not GPU bottleneck proof",
        },
        "cleanup": {
            "control_path_preexisting": os.path.lexists(paths.control),
            "control_stop_requested": False, "owned_outer_sigterm_sent": False,
            "outer_exited": None, "orchestrator_sigkill_sent": False,
        },
        "restoration": None, "problem": None,
        "limitations": {
            "gpu_inference_validated": False, "speedup_measured": False,
            "cpu_savings_measured": False, "quality_comparison_complete": False,
            "generated_writing_read_for_selection": False,
        },
        "generated_prose_in_receipt": False,
    }


def run(args: argparse.Namespace, *, results: Path = RESULTS) -> tuple[dict[str, Any], int]:
    paths = probe_paths(args.tag, args.output_prefix, results=results)
    assert_unused(paths)
    pin_failure: str | None = None
    try:
        pins = verify_pins()
    except (ProbeError, OSError) as exc:
        pin_failure = str(exc)
        pins = {"verified": False, "error": pin_failure}
    try:
        eligibility = validate_eligibility(
            args.primary_study_receipt, args.primary_prose, args.primary_boundary_ready,
        )
    except (ProbeError, RuntimeError, OSError) as exc:
        eligibility = Eligibility(
            eligible=False,
            failures=(f"primary metadata validation failed: {exc}",),
            primary_receipt={"path": str(args.primary_study_receipt)},
            primary_prose=(), boundary={"path": str(args.primary_boundary_ready)},
        )
    if pin_failure is not None:
        eligibility = Eligibility(
            eligible=False,
            failures=eligibility.failures + (f"pinned input verification failed: {pin_failure}",),
            primary_receipt=eligibility.primary_receipt,
            primary_prose=eligibility.primary_prose, boundary=eligibility.boundary,
        )
    receipt = build_receipt(args=args, paths=paths, pins=pins, eligibility=eligibility)
    if args.execute and eligibility.eligible:
        try:
            refreshed = validate_eligibility(
                args.primary_study_receipt, args.primary_prose, args.primary_boundary_ready,
            )
            if not refreshed.eligible:
                raise ProbeError(
                    "execution-time primary eligibility revalidation failed: "
                    + "; ".join(refreshed.failures)
                )
            receipt["eligibility"].update({
                "eligible": True,
                "failures": [],
                "primary_study": dict(refreshed.primary_receipt),
                "primary_prose": [dict(row) for row in refreshed.primary_prose],
                "family_boundary": dict(refreshed.boundary),
            })
            receipt["execution"]["launch_pin_revalidation"] = verify_pins()
            code = execute_probe(
                paths,
                args.tag,
                receipt,
                boundary_ready_path=args.primary_boundary_ready,
                primary_receipt_path=args.primary_study_receipt,
            )
        except (ProbeError, OSError) as exc:
            receipt["status"] = "preflight_failed_not_launched"
            receipt["problem"] = {"error_type": type(exc).__name__, "error": str(exc)}
            code = 1
    elif args.execute:
        receipt["problem"] = {
            "error_type": "EligibilityError",
            "error": "execution refused because frozen metadata eligibility failed",
        }
        code = 1
    else:
        code = 0 if eligibility.eligible else 1
    receipt["finished_at"] = utc_now()
    _exclusive_json(paths.receipt, receipt)
    return receipt, code


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-study-receipt", type=Path, default=PRIMARY_RECEIPT)
    parser.add_argument(
        "--primary-prose", action="append", type=Path, default=None,
        help="fixed content-free original canonical primary NONE prose JSONL; alternate/split inputs fail closed",
    )
    parser.add_argument("--primary-boundary-ready", type=Path, default=BOUNDARY_READY)
    parser.add_argument("--tag", default=DEFAULT_TAG)
    parser.add_argument("--output-prefix", type=Path, default=RESULTS / DEFAULT_TAG)
    parser.add_argument(
        "--execute", action="store_true",
        help="open the bounded owned GPU window; absent by default for metadata-only inspection",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.primary_prose is None:
        args.primary_prose = [PRIMARY_PROSE]
    try:
        receipt, code = run(args)
    except (ProbeError, RuntimeError, OSError, subprocess.SubprocessError) as exc:
        print(f"style poll0 probe: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({
        "status": receipt["status"], "mode": receipt["mode"],
        "eligible": receipt["eligibility"]["eligible"],
        "receipt": str(probe_paths(args.tag, args.output_prefix).receipt),
    }, sort_keys=True))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
