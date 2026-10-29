#!/usr/bin/env python3
"""Fail-closed controller for the optional Bonsai2 PTQ DSpark pilot.

The ``--check-only`` mode is CPU-only and creates no candidate files.
``--execute`` is intentionally unavailable until a parent-authored eligibility
receipt binds the completed PTQ study, every raw request source, the clean
boundary, the family-work disposition, and an unreleased READY receipt.
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
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
BASE_CONFIG = ROOT / "runtime/specialists-final72k64k-bonsai2-ptq1_0-72k.json"
BASE_CONFIG_SHA256 = "557f6f5cf23c5ac87f7a15915ca60283fa941e8a29ae516fb1919d4249ae9f42"
DS_PARK_MODEL = Path(
    "/external/storage/moe-bench/models/bonsai2-dspark/"
    "Ternary-Bonsai-2-27B-dspark-v1-experimental-Q4_0.gguf"
)
DS_PARK_SIZE = 631_713_664
DS_PARK_SHA256 = "4ead3be7e410d8ca7f65c83d25f4f599dad4797316af0351e1d6806072b01421"
DS_PARK_INVENTORY_SHA256 = "8d5dfbc314fd2191a1d3f978f17a67eeedfe7176473767b9908f1a6d76dd3a6b"

PTQ_TAG = "specialists-final72k64k-bonsai2-ptq1_0-strict-72k"
PTQ_MODEL_SESSION_RUN_ID = "e3a93f92-7ec5-4c03-8573-1395515d4fe8"
PTQ_MODEL_NAME = "bonsai2-27b-ptq1_0"
PTQ_SERVER_PID = 256_501
PTQ_SERVER_START_TICKS = "85554512"
RUN_PREFIX = "specialists-final72k64k"
MODEL_KEY = "bonsai2-ptq1_0"
CONTEXT = 73_728
ENDPOINT = "http://127.0.0.1:18081/v1"

WINDOW_SECONDS = 28_800.0
STARTUP_TIMEOUT = 480.0
IDLE_WAIT_SECONDS = 900.0
REQUEST_TIMEOUT = 21_600.0
RESTORATION_RESERVE_SECONDS = 1_800.0
BOUNDARY_WAIT_SECONDS = 86_400.0
MIN_BOUNDARY_REMAINING_SECONDS = 30_600.0
MAX_GENERATION_REQUESTS = 4

SMOKE_STEP_ID = "dspark_smoke_json_none_256"
QUICK_IDS = ("quick_generation", "quick_json_extract")
CODE_IDS = (
    "code_fresh_batch_module",
    "code_multifunction_ledger_bugfix",
    "code_edge_reading_pivot",
)
PROSE_IDS = (
    "prose_grounded_public_briefing",
    "prose_constrained_last_tram",
    "prose_structured_orion_edit",
)
HARD_IDS = (
    "hard_transaction_ledger",
    "hard_weighted_intervals",
    "hard_temporal_kv",
    "hard_dependency_constraints",
)
RAW_SOURCE_COUNTS = {
    "plan_quick": ("plan_01_quick_medium_65536", 2),
    "plan_hard": ("plan_02_hard_medium_65536", 4),
    "canonical_code": ("code", 3),
    "canonical_prose": ("prose", 3),
    "publisher_code": ("sampler_01_requests", 3),
    "shared_t0_code": ("sampler_02_requests", 3),
    "publisher_prose": ("sampler_03_requests", 3),
}

SOURCE_PINS = {
    ROOT / "runtime/model_session.py": "968b2ce078975de23dd99670c2141995f460f36ae86490ce33153e7f1ae9c43c",
    ROOT / "runtime/gpu_window.py": "7982aa5906fd05f6cb61c9e2b54467724755472e7d17feb485a687a142ca6898",
    ROOT / "runtime/specialist_study_trial.py": "2187ee832d3785727f836d156d343865efcccb299547e877815e8ac72920e734",
    ROOT / "runtime/specialist_bench.py": "fff82d7265f38af82f8b495b1da256e27e3e87ef326591ef63daa5703f76f7eb",
    ROOT / "runtime/sampler_recipe_probe.py": "8543bb44ba3242240aaa5646c51b2252c9b6367f3d624ec29b0ab21695bdfb18",
    ROOT / "runtime/native_hard_recipe_probe.py": "52f5c5c49c739e4e9643a483a485ca1afb0943bc0b4362bf0a8dddac63ab859c",
    ROOT / "bench_cases.py": "b8f020837fd385baa38e4aed8e21d66454e98743b6f2abe5211134ca84f7f27b",
    ROOT / "runtime/specialist_cases.py": "2486edf32007b35e24d1838631a267eb5ee3654afd4aad41dae0bb87020ffc43",
}

PREDECLARED_PINS = {
    ROOT / "results/specialists-bonsai-dspark-diagnostic-plan-v1.json": "bcbe9f5e0f4f5485591c09ce26e49911612110869518d1f8a686c3771ed09162",
    ROOT / "results/held-session-generation-budget-preflight-v1.json": "2c8a53e7adeb00f60338b6b189c8522c6bade1e2f5754f821423aea7ec72a4a9",
    ROOT / "results/bonsai2-speculative-metric-semantics-preflight-v1.json": "824a935af6e4eb8ebe79b2be1a74b4d305d14fda002bb4e495ed5017fc0d478a",
    ROOT / "results/bonsai2-dspark-markov-cuda-option-applicability-v1.json": "53b578a8f072bbf6255b78d07dc958505c4ace3700d6cafffffa792325aab9a1",
    ROOT / "results/bonsai2-dspark-parent-artifact-acceptance-v1.json": "3c912c9b7c81011765fe3b14d53e8f4b90a975bf7f75eb998fa0e36443e6ba33",
    ROOT / "results/bonsai2-dspark-artifact-v2.json": "f9ffa5a22d2c4e6403f5673c31f0c7a1fe844d5efba49b89dbe3e4a7148e8911",
}

CANDIDATE_ARGS = (
    "--spec-type",
    "draft-dspark",
    "--spec-draft-model",
    str(DS_PARK_MODEL),
    "--spec-draft-n-max",
    "4",
    "--spec-draft-ngl",
    "all",
)
FORBIDDEN_ENVIRONMENT_OPTION = "LLAMA_DSPARK_MARKOV_CUDA"
TAG_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
MAX_JSON_BYTES = 32 * 1024 * 1024
MAX_JSONL_BYTES = 64 * 1024 * 1024
TERMINAL_EVENTS = {
    "holding_stopped",
    "server_stopped",
    "server_cleanup_failed",
    "session_failed",
    "session_interrupted",
    "session_complete",
}


class PilotError(RuntimeError):
    """A required pilot invariant is absent or contradicted."""


class PilotGateStopped(RuntimeError):
    """A valid candidate request reached a declared semantic/performance gate."""

    def __init__(self, stage: str, reason: str, evidence: Mapping[str, Any]) -> None:
        super().__init__(reason)
        self.stage = stage
        self.evidence = dict(evidence)


class PilotStageError(PilotError):
    """A runtime-stage failure with trustworthy partial request evidence."""

    def __init__(self, stage: str, reason: str, evidence: Mapping[str, Any]) -> None:
        super().__init__(reason)
        self.stage = stage
        self.evidence = dict(evidence)


class PilotInterrupted(BaseException):
    def __init__(self, signum: int) -> None:
        super().__init__(f"signal {signum}")
        self.signum = signum


@dataclass(frozen=True)
class Binding:
    path: Path
    sha256: str
    size: int


@dataclass(frozen=True)
class Eligibility:
    source: Binding
    document: Mapping[str, Any]
    study: Binding
    ready: Binding
    family: Binding
    cleanup: Binding
    baseline_native_tps: float
    baseline_wall_seconds: float
    quality_kind: str
    quality_case_id: str
    quality_prompt_sha256: str
    boundary_remaining_seconds: float


@dataclass(frozen=True)
class RawSource:
    binding: Binding
    rows: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True)
class PilotPaths:
    config: Path
    smoke_plan: Path
    control: Path
    events: Path
    server_log: Path
    session_report: Path
    smoke: Path
    quick: Path
    quality: Path
    quality_texts: Path
    receipt: Path


@dataclass
class SignalLatch:
    received: int | None = None

    def handle(self, signum: int, _frame: Any) -> None:
        self.received = signum

    def check(self) -> None:
        if self.received is not None:
            raise PilotInterrupted(self.received)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _regular(path: Path, description: str, maximum: int | None = None) -> os.stat_result:
    try:
        details = path.lstat()
    except FileNotFoundError as exc:
        raise PilotError(f"{description} is missing: {path}") from exc
    except OSError as exc:
        raise PilotError(f"{description} is inaccessible: {path}") from exc
    if path.is_symlink() or not stat.S_ISREG(details.st_mode):
        raise PilotError(f"{description} must be a regular non-symlink file: {path}")
    if maximum is not None and details.st_size > maximum:
        raise PilotError(f"{description} exceeds its size limit: {path}")
    return details


def _json(path: Path, description: str) -> tuple[dict[str, Any], Binding]:
    details = _regular(path, description, MAX_JSON_BYTES)
    try:
        payload = path.read_bytes()
        value = json.loads(payload)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PilotError(f"{description} is malformed: {path}") from exc
    if not isinstance(value, dict):
        raise PilotError(f"{description} must be a JSON object")
    return value, Binding(path.resolve(), hashlib.sha256(payload).hexdigest(), details.st_size)


def _path(value: Any, description: str) -> Path:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise PilotError(f"{description} must be a nonempty path")
    return Path(value).expanduser().resolve()


def _same_path(value: Any, expected: Path) -> bool:
    try:
        return _path(value, "path") == expected.resolve()
    except PilotError:
        return False


def validate_binding(value: Any, description: str, *, maximum: int | None = None) -> Binding:
    if not isinstance(value, Mapping):
        raise PilotError(f"{description} binding must be an object")
    path = _path(value.get("path"), f"{description}.path")
    digest = value.get("sha256")
    size = value.get("size")
    if not isinstance(digest, str) or SHA256_RE.fullmatch(digest) is None:
        raise PilotError(f"{description}.sha256 must be a lowercase SHA-256")
    if not isinstance(size, int) or isinstance(size, bool) or size < 1:
        raise PilotError(f"{description}.size must be a positive integer")
    details = _regular(path, description, maximum)
    if details.st_size != size:
        raise PilotError(f"{description} size differs from its binding")
    actual = sha256_file(path)
    if actual != digest:
        raise PilotError(f"{description} SHA-256 differs from its binding")
    return Binding(path, actual, size)


def _parse_time(value: Any, description: str) -> datetime:
    if not isinstance(value, str):
        raise PilotError(f"{description} must be an RFC3339 timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise PilotError(f"{description} must be an RFC3339 timestamp") from exc
    if parsed.tzinfo is None:
        raise PilotError(f"{description} must include a timezone")
    return parsed.astimezone(timezone.utc)


def validate_static_sources() -> dict[str, str]:
    observed: dict[str, str] = {}
    for path, expected in {**SOURCE_PINS, **PREDECLARED_PINS, BASE_CONFIG: BASE_CONFIG_SHA256}.items():
        _regular(path, "pinned controller input")
        actual = sha256_file(path)
        if actual != expected:
            raise PilotError(f"pinned source SHA-256 changed: {path}")
        observed[str(path)] = actual
    return observed


def _expected_prompts() -> tuple[dict[str, dict[str, str]], dict[str, dict[str, str]]]:
    import bench_cases
    from runtime import specialist_cases

    benchmark: dict[str, dict[str, str]] = {}
    for suite in ("quick", "hard"):
        for case in bench_cases.get_cases(suite):
            benchmark[case.case_id] = {
                "prompt_sha256": hashlib.sha256(case.prompt.encode("utf-8")).hexdigest(),
                "case_version": bench_cases.CASE_VERSION,
            }
    specialist = {
        case.case_id: {
            "prompt_sha256": hashlib.sha256(case.prompt.encode("utf-8")).hexdigest(),
            "case_sha256": specialist_cases.case_sha256(case),
            "case_version": specialist_cases.CASE_VERSION,
        }
        for case in specialist_cases.get_cases(groups=("code",))
    }
    return benchmark, specialist


def _validate_baseline_config() -> dict[str, Any]:
    config, binding = _json(BASE_CONFIG, "baseline PTQ config")
    if binding.sha256 != BASE_CONFIG_SHA256:
        raise PilotError("baseline PTQ config SHA-256 changed")
    if (
        config.get("schema_version") != 1
        or config.get("model_name") != PTQ_MODEL_NAME
        or config.get("context") != CONTEXT
        or config.get("host") != "127.0.0.1"
        or config.get("port") != 18081
    ):
        raise PilotError("baseline PTQ config identity or context changed")
    args = config.get("args")
    if not isinstance(args, list) or any(not isinstance(item, str) for item in args):
        raise PilotError("baseline PTQ args are malformed")
    if any(item.startswith("--spec-") for item in args):
        raise PilotError("baseline PTQ config already contains speculative options")
    return config


def _line_count(path: Path) -> int:
    _regular(path, "raw request source", MAX_JSONL_BYTES)
    count = 0
    last = b""
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            count += block.count(b"\n")
            last = block[-1:]
    if last not in (b"", b"\n"):
        raise PilotError(f"raw request source has an unterminated final record: {path}")
    return count


def _validate_study(document: Mapping[str, Any], binding: Binding) -> None:
    source = document.get("source_config") if isinstance(document.get("source_config"), Mapping) else {}
    session = document.get("session") if isinstance(document.get("session"), Mapping) else {}
    cleanup = document.get("cleanup") if isinstance(document.get("cleanup"), Mapping) else {}
    server_cleanup = (
        cleanup.get("model_session_cleanup")
        if isinstance(cleanup.get("model_session_cleanup"), Mapping)
        else {}
    )
    restoration = (
        document.get("restoration") if isinstance(document.get("restoration"), Mapping) else {}
    )
    helpers = document.get("helper_sources") if isinstance(document.get("helper_sources"), Mapping) else {}
    expected_helpers = {
        str(path): digest
        for path, digest in SOURCE_PINS.items()
        if path.name in {
            "model_session.py",
            "gpu_window.py",
            "specialist_bench.py",
            "sampler_recipe_probe.py",
        }
    }
    sampler = document.get("sampler_plan") if isinstance(document.get("sampler_plan"), Mapping) else {}
    stages = sampler.get("stages") if isinstance(sampler.get("stages"), list) else []
    if not (
        document.get("schema_version") == 1
        and document.get("status") in {"complete", "completed_with_evaluation_failures"}
        and document.get("problem") is None
        and document.get("tag") == PTQ_TAG
        and document.get("runtime_backend") == "llama"
        and source.get("path") == str(BASE_CONFIG)
        and source.get("sha256") == BASE_CONFIG_SHA256
        and session.get("run_id") == PTQ_MODEL_SESSION_RUN_ID
        and session.get("model_name") == PTQ_MODEL_NAME
        and session.get("context") == CONTEXT
        and cleanup.get("outer_exited") is True
        and cleanup.get("orchestrator_sigkill_sent") is False
        and server_cleanup.get("stopped") is True
        and server_cleanup.get("kill_sent") is False
        and restoration.get("original_running") is True
        and restoration.get("original_paused") is False
        and restoration.get("owned_loopback_port") == 18081
        and restoration.get("owned_loopback_port_clear") is True
        and sampler.get("matched") is True
        and len(stages) == 3
        and all(
            isinstance(stage, Mapping)
            and stage.get("returncode") in {0, 1}
            and isinstance(stage.get("receipt"), Mapping)
            and stage["receipt"].get("status") in {"complete", "completed_with_request_failures"}
            for stage in stages
        )
        and all(helpers.get(path) == digest for path, digest in expected_helpers.items())
    ):
        raise PilotError("completed PTQ study receipt does not prove its exact terminal lifecycle")
    if binding.path.name != f"{PTQ_TAG}-study-trial.json":
        raise PilotError("completed PTQ study receipt filename does not match the frozen tag")


def _validate_raw_sources(
    eligibility: Mapping[str, Any], study: Mapping[str, Any]
) -> dict[str, RawSource]:
    rows = eligibility.get("raw_request_sources")
    if not isinstance(rows, list) or len(rows) != len(RAW_SOURCE_COUNTS):
        raise PilotError("eligibility must bind exactly seven raw request sources")
    artifacts = study.get("artifacts") if isinstance(study.get("artifacts"), Mapping) else {}
    by_key: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping) or not isinstance(row.get("key"), str):
            raise PilotError("raw request source entries must have string keys")
        key = str(row["key"])
        if key in by_key:
            raise PilotError("raw request source keys must be unique")
        by_key[key] = row
    if set(by_key) != set(RAW_SOURCE_COUNTS):
        raise PilotError("raw request source keys differ from the frozen 21-request inventory")
    result: dict[str, RawSource] = {}
    all_calls: set[str] = set()
    total = 0
    for key, (artifact_key, expected_count) in RAW_SOURCE_COUNTS.items():
        row = by_key[key]
        if row.get("artifact_key") != artifact_key or row.get("terminal_requests") != expected_count:
            raise PilotError(f"raw request source {key} has the wrong artifact or request count")
        bound = validate_binding(row, f"raw request source {key}", maximum=MAX_JSONL_BYTES)
        artifact = artifacts.get(artifact_key) if isinstance(artifacts.get(artifact_key), Mapping) else {}
        if not (
            _same_path(artifact.get("path"), bound.path)
            and artifact.get("exists") is True
            and artifact.get("regular_file") is True
            and artifact.get("directory") is False
            and artifact.get("size") == bound.size
        ):
            raise PilotError(f"completed PTQ study does not inventory raw request source {key}")
        if _line_count(bound.path) != expected_count:
            raise PilotError(f"raw request source {key} does not contain exactly {expected_count} rows")
        metadata = _metadata_rows(bound.path, f"raw request source {key}")
        expected_cases = {
            "plan_quick": QUICK_IDS,
            "plan_hard": HARD_IDS,
            "canonical_code": CODE_IDS,
            "canonical_prose": PROSE_IDS,
            "publisher_code": CODE_IDS,
            "shared_t0_code": CODE_IDS,
            "publisher_prose": PROSE_IDS,
        }[key]
        if [item.get("case_id") for item in metadata] != list(expected_cases):
            raise PilotError(f"raw request source {key} is not in its frozen case order")
        for item in metadata:
            call_id = item.get("call_id")
            response = item.get("response") if isinstance(item.get("response"), Mapping) else {}
            tokens = response.get("actual_completion_tokens")
            terminal_response = bool(
                response.get("stream_complete") is True
                and response.get("done_reason") in {"stop", "length", "eos", "end_turn"}
                and isinstance(response.get("truncated"), bool)
                and isinstance(tokens, int)
                and not isinstance(tokens, bool)
                and tokens >= 0
            )
            if not isinstance(call_id, str) or not call_id or call_id in all_calls:
                raise PilotError("raw PTQ requests lack globally unique nonempty call IDs")
            if item.get("error_present") is not True and not terminal_response:
                raise PilotError(f"raw request source {key} contains a nonterminal attempt")
            all_calls.add(call_id)
        result[key] = RawSource(bound, tuple(metadata))
        total += expected_count
    if total != 21:
        raise PilotError("raw PTQ request inventory is not exactly 21 requests")
    return result


def _clean_outcome(row: Mapping[str, Any], case_id: str, identity: Mapping[str, str]) -> None:
    if (
        row.get("case_id") != case_id
        or row.get("prompt_sha256") != identity["prompt_sha256"]
        or row.get("case_version") != identity["case_version"]
        or not isinstance(row.get("call_id"), str)
        or not row.get("call_id")
        or not isinstance(row.get("passed"), bool)
        or not isinstance(row.get("eos"), bool)
        or not isinstance(row.get("nonempty"), bool)
        or not isinstance(row.get("error_free"), bool)
        or not isinstance(row.get("truncated"), bool)
    ):
        raise PilotError(f"quality evidence is incomplete for {case_id}")
    if row.get("passed") is True and not (
        row.get("eos") is True
        and row.get("nonempty") is True
        and row.get("error_free") is True
        and row.get("truncated") is False
    ):
        raise PilotError(f"a reported pass is not a clean EOS result for {case_id}")


def _declared_outcome_matches_actual(
    declared: Mapping[str, Any], actual: Mapping[str, Any]
) -> bool:
    request = actual.get("request") if isinstance(actual.get("request"), Mapping) else {}
    response = actual.get("response") if isinstance(actual.get("response"), Mapping) else {}
    tokens = response.get("actual_completion_tokens")
    return bool(
        actual.get("case_id") == declared.get("case_id")
        and actual.get("case_version") == declared.get("case_version")
        and request.get("prompt_sha256") == declared.get("prompt_sha256")
        and actual.get("evaluation_passed") is declared.get("passed")
        and (response.get("stream_complete") is True and _eos(response.get("done_reason")))
        is declared.get("eos")
        and (isinstance(tokens, int) and not isinstance(tokens, bool) and tokens > 0)
        is declared.get("nonempty")
        and (actual.get("error_present") is False) is declared.get("error_free")
        and (response.get("truncated") is True) is declared.get("truncated")
    )


def _validate_quality(
    document: Mapping[str, Any], raw: Mapping[str, RawSource]
) -> tuple[str, str, str]:
    benchmark, specialist = _expected_prompts()
    quality = document.get("quality") if isinstance(document.get("quality"), Mapping) else {}
    code = quality.get("canonical_code")
    hard = quality.get("hard")
    if not isinstance(code, list) or not isinstance(hard, list):
        raise PilotError("eligibility quality evidence must contain code3 and hard4 rows")
    if [row.get("case_id") if isinstance(row, Mapping) else None for row in code] != list(CODE_IDS):
        raise PilotError("canonical code evidence is not in frozen suite order")
    if [row.get("case_id") if isinstance(row, Mapping) else None for row in hard] != list(HARD_IDS):
        raise PilotError("hard evidence is not in frozen suite order")
    seen_calls: set[str] = set()
    for row, case_id in zip(code, CODE_IDS, strict=True):
        assert isinstance(row, Mapping)
        _clean_outcome(row, case_id, specialist[case_id])
        if row.get("case_sha256") != specialist[case_id]["case_sha256"]:
            raise PilotError(f"canonical code case SHA-256 mismatch for {case_id}")
        if row.get("source_key") != "canonical_code":
            raise PilotError(f"canonical code source binding mismatch for {case_id}")
        actual = [item for item in raw["canonical_code"].rows if item.get("call_id") == row["call_id"]]
        if len(actual) != 1 or not _declared_outcome_matches_actual(row, actual[0]):
            raise PilotError(f"canonical code summary differs from raw source for {case_id}")
        seen_calls.add(str(row["call_id"]))
    for row, case_id in zip(hard, HARD_IDS, strict=True):
        assert isinstance(row, Mapping)
        _clean_outcome(row, case_id, benchmark[case_id])
        if row.get("source_key") != "plan_hard":
            raise PilotError(f"hard source binding mismatch for {case_id}")
        actual = [item for item in raw["plan_hard"].rows if item.get("call_id") == row["call_id"]]
        if len(actual) != 1 or not _declared_outcome_matches_actual(row, actual[0]):
            raise PilotError(f"hard summary differs from raw source for {case_id}")
        seen_calls.add(str(row["call_id"]))
    if len(seen_calls) != len(CODE_IDS) + len(HARD_IDS):
        raise PilotError("canonical quality call IDs are not unique")
    if "canonical_code" not in raw or "plan_hard" not in raw:
        raise PilotError("quality evidence lacks its immutable raw source bindings")
    code_passes = sum(row.get("passed") is True for row in code)
    hard_passes = sum(row.get("passed") is True for row in hard)
    if code_passes >= 2:
        selected = next(row for row in code if row.get("passed") is True and row.get("eos") is True)
        return "code", str(selected["case_id"]), str(selected["prompt_sha256"])
    if hard_passes >= 3:
        selected = next(row for row in hard if row.get("passed") is True and row.get("eos") is True)
        return "hard", str(selected["case_id"]), str(selected["prompt_sha256"])
    raise PilotError("PTQ quality is below both canonical-code and hard eligibility thresholds")


def native_pooled_tps(rows: Sequence[Mapping[str, Any]]) -> float:
    numerator = 0
    seconds = 0.0
    for row in rows:
        count = row.get("native_predicted_n")
        duration = row.get("native_predicted_seconds")
        if (
            not isinstance(count, int)
            or isinstance(count, bool)
            or count < 2
            or not isinstance(duration, (int, float))
            or isinstance(duration, bool)
            or not math.isfinite(float(duration))
            or float(duration) <= 0
        ):
            raise PilotError("native timing rows require predicted_n>=2 and positive finite seconds")
        numerator += count - 1
        seconds += float(duration)
    if numerator <= 0 or seconds <= 0:
        raise PilotError("pooled native timing denominator or numerator is empty")
    return numerator / seconds


def _validate_quick_baseline(
    document: Mapping[str, Any], raw: Mapping[str, RawSource]
) -> tuple[float, float]:
    benchmark, _ = _expected_prompts()
    quick = document.get("quick_baseline")
    if not isinstance(quick, list) or len(quick) != 2:
        raise PilotError("eligibility must contain exactly two baseline quick rows")
    if [row.get("case_id") if isinstance(row, Mapping) else None for row in quick] != list(QUICK_IDS):
        raise PilotError("baseline quick rows are not in frozen case order")
    calls: set[str] = set()
    for row, case_id in zip(quick, QUICK_IDS, strict=True):
        assert isinstance(row, Mapping)
        _clean_outcome(row, case_id, benchmark[case_id])
        output_tokens = row.get("output_tokens")
        wall = row.get("wall_seconds")
        if not (
            row.get("source_key") == "plan_quick"
            and row.get("passed") is True
            and row.get("eos") is True
            and row.get("nonempty") is True
            and row.get("error_free") is True
            and row.get("truncated") is False
            and isinstance(output_tokens, int)
            and not isinstance(output_tokens, bool)
            and 1 <= output_tokens <= 8192
            and isinstance(wall, (int, float))
            and not isinstance(wall, bool)
            and math.isfinite(float(wall))
            and float(wall) > 0
        ):
            raise PilotError(f"baseline quick gate failed for {case_id}")
        actual = [item for item in raw["plan_quick"].rows if item.get("call_id") == row["call_id"]]
        if len(actual) != 1 or not _declared_outcome_matches_actual(row, actual[0]):
            raise PilotError(f"baseline quick summary differs from raw source for {case_id}")
        response = actual[0].get("response") if isinstance(actual[0].get("response"), Mapping) else {}
        native = response.get("native") if isinstance(response.get("native"), Mapping) else {}
        actual_ms = native.get("predicted_ms")
        if not (
            response.get("actual_completion_tokens") == output_tokens
            and response.get("wall_seconds") == wall
            and native.get("predicted_n") == row.get("native_predicted_n")
            and isinstance(actual_ms, (int, float))
            and not isinstance(actual_ms, bool)
            and math.isclose(
                float(actual_ms) / 1000.0,
                float(row.get("native_predicted_seconds", -1)),
                rel_tol=0.0,
                abs_tol=1e-9,
            )
        ):
            raise PilotError(f"baseline quick timing summary differs from raw source for {case_id}")
        calls.add(str(row["call_id"]))
    if len(calls) != 2 or "plan_quick" not in raw:
        raise PilotError("baseline quick evidence lacks unique calls or its raw binding")
    wall = sum(float(row["wall_seconds"]) for row in quick)
    if wall > 3600:
        raise PilotError("baseline quick combined wall time exceeds 3600 seconds")
    return native_pooled_tps(quick), wall


def _validate_family(value: Mapping[str, Any], study: Binding) -> Binding:
    binding = validate_binding(value.get("receipt"), "V3/family disposition receipt")
    document, observed = _json(binding.path, "V3/family disposition receipt")
    if observed.sha256 != binding.sha256 or observed.size != binding.size:
        raise PilotError("V3/family disposition receipt changed after binding")
    subruns = document.get("diagnostic_subruns")
    if not isinstance(subruns, list):
        raise PilotError("V3/family disposition must list hash-bound diagnostic subruns")
    bound_subruns = [
        validate_binding(subrun, f"V3/family diagnostic subrun {index}")
        for index, subrun in enumerate(subruns)
    ]
    evidence_binding = validate_binding(
        document.get("v3_disposition_evidence"),
        "post-primary V3 disposition evidence",
    )
    evidence, evidence_observed = _json(
        evidence_binding.path, "post-primary V3 disposition evidence"
    )
    if (
        evidence_observed.sha256 != evidence_binding.sha256
        or evidence_observed.size != evidence_binding.size
    ):
        raise PilotError("post-primary V3 disposition evidence changed after binding")
    evidence_study = validate_binding(
        evidence.get("completed_study"), "V3 disposition completed PTQ study"
    )
    evidence_subruns = evidence.get("diagnostic_subruns")
    if not isinstance(evidence_subruns, list) or len(evidence_subruns) != len(bound_subruns):
        raise PilotError("V3 disposition evidence and family receipt disagree on diagnostic subruns")
    for index, (declared, expected) in enumerate(zip(evidence_subruns, bound_subruns, strict=True)):
        observed_subrun = validate_binding(declared, f"V3 disposition diagnostic subrun {index}")
        if observed_subrun != expected:
            raise PilotError("V3 disposition evidence and family receipt bind different subruns")
    decision = evidence.get("decision")
    reason = evidence.get("reason")
    if not (
        evidence.get("schema_version") == 1
        and evidence.get("kind") == "bonsai2_v3_post_primary_disposition_evidence"
        and evidence.get("status") == "complete"
        and evidence.get("main_ptq_tag") == PTQ_TAG
        and evidence.get("main_model_session_run_id") == PTQ_MODEL_SESSION_RUN_ID
        and evidence_study == study
        and decision in {"diagnostics_selected", "no_optional_followup_selected"}
        and isinstance(reason, str)
        and bool(reason.strip())
        and ((decision == "diagnostics_selected") is bool(bound_subruns))
    ):
        raise PilotError("post-primary V3 disposition evidence is incomplete or inconsistent")
    _parse_time(evidence.get("created_at"), "post-primary V3 disposition created_at")
    if not (
        document.get("schema_version") == 1
        and document.get("kind") == "bonsai2_ptq_v3_family_disposition"
        and document.get("status") == "complete"
        and document.get("main_ptq_tag") == PTQ_TAG
        and document.get("main_model_session_run_id") == PTQ_MODEL_SESSION_RUN_ID
        and document.get("all_v3_family_work_complete") is True
        and isinstance(document.get("in_flight_requests"), int)
        and not isinstance(document.get("in_flight_requests"), bool)
        and document.get("in_flight_requests") == 0
        and document.get("gemma_disposition") == "deferred"
        and document.get("flash_disposition") == "queued_after_ptq"
        and document.get("matching_20b_disposition") == "last"
    ):
        raise PilotError("V3/family disposition is incomplete or for another PTQ run")
    return binding


def _validate_cleanup(value: Mapping[str, Any]) -> Binding:
    binding = validate_binding(value.get("receipt"), "clean boundary receipt")
    document, observed = _json(binding.path, "clean boundary receipt")
    if observed.sha256 != binding.sha256 or observed.size != binding.size:
        raise PilotError("clean boundary receipt changed after binding")
    server = document.get("server") if isinstance(document.get("server"), Mapping) else {}
    original = document.get("original") if isinstance(document.get("original"), Mapping) else {}
    ports = document.get("ports_clear") if isinstance(document.get("ports_clear"), Mapping) else {}
    if not (
        document.get("schema_version") == 1
        and document.get("kind") == "bonsai2_ptq_clean_boundary"
        and document.get("main_ptq_tag") == PTQ_TAG
        and document.get("main_model_session_run_id") == PTQ_MODEL_SESSION_RUN_ID
        and server.get("pid") == PTQ_SERVER_PID
        and str(server.get("start_ticks")) == PTQ_SERVER_START_TICKS
        and server.get("gone") is True
        and original.get("container") == "local-codex-ollama-cpu"
        and original.get("running") is True
        and original.get("paused") is False
        and original.get("api_healthy") is True
        and ports.get("18080") is True
        and ports.get("18081") is True
    ):
        raise PilotError("clean boundary receipt does not prove exact PTQ cleanup and restoration")
    _parse_time(document.get("checked_at"), "clean boundary checked_at")
    return binding


def _validate_boundary(
    value: Mapping[str, Any], study: Binding, now: datetime
) -> tuple[Binding, float]:
    if value.get("wait_seconds") != int(BOUNDARY_WAIT_SECONDS):
        raise PilotError("eligibility boundary wait must be exactly 86400 seconds")
    ready = validate_binding(value.get("ready"), "PTQ READY receipt")
    expected_ready = (
        RESULTS
        / "specialists-final72k64k-boundaries"
        / f"{RUN_PREFIX}-{MODEL_KEY}-ready.json"
    ).resolve()
    expected_release = expected_ready.with_name(expected_ready.name.replace("-ready.json", "-release.json"))
    if ready.path != expected_ready or not _same_path(value.get("release_path"), expected_release):
        raise PilotError("eligibility boundary paths do not match the frozen PTQ boundary")
    if os.path.lexists(expected_release):
        raise PilotError("matching PTQ boundary RELEASE already exists")
    document, observed = _json(ready.path, "PTQ READY receipt")
    bound_study = document.get("study") if isinstance(document.get("study"), Mapping) else {}
    if not (
        observed.sha256 == ready.sha256
        and observed.size == ready.size
        and document.get("schema_version") == 1
        and document.get("kind") == "model_boundary_ready"
        and document.get("run_prefix") == RUN_PREFIX
        and document.get("model_key") == MODEL_KEY
        and _same_path(bound_study.get("path"), study.path)
        and bound_study.get("sha256") == study.sha256
    ):
        raise PilotError("PTQ READY receipt does not bind the exact completed study")
    published = _parse_time(document.get("published_at"), "PTQ READY published_at")
    remaining = (published + timedelta(seconds=BOUNDARY_WAIT_SECONDS) - now).total_seconds()
    if remaining < MIN_BOUNDARY_REMAINING_SECONDS:
        raise PilotError("PTQ READY boundary has less than 30600 seconds remaining")
    return ready, remaining


def validate_eligibility(path: Path, *, now: datetime | None = None) -> Eligibility:
    validate_static_sources()
    _validate_baseline_config()
    document, source = _json(path.expanduser().resolve(), "parent eligibility receipt")
    if not (
        document.get("schema_version") == 1
        and document.get("kind") == "bonsai2_dspark_pilot_eligibility"
        and document.get("decision") == "eligible"
        and document.get("pilot_warranted") is True
        and document.get("accepted_by_parent") is True
        and document.get("main_ptq_tag") == PTQ_TAG
        and document.get("main_model_session_run_id") == PTQ_MODEL_SESSION_RUN_ID
        and document.get("target_profile") == "bonsai2-ptq1_0"
        and document.get("baseline_config_sha256") == BASE_CONFIG_SHA256
    ):
        raise PilotError("parent eligibility receipt does not authorize this exact PTQ pilot")
    study = validate_binding(document.get("completed_study"), "completed PTQ study receipt")
    study_document, observed = _json(study.path, "completed PTQ study receipt")
    if observed.sha256 != study.sha256 or observed.size != study.size:
        raise PilotError("completed PTQ study receipt changed after binding")
    _validate_study(study_document, study)
    raw = _validate_raw_sources(document, study_document)
    quality_kind, quality_case, prompt_sha = _validate_quality(document, raw)
    baseline_tps, baseline_wall = _validate_quick_baseline(document, raw)
    family_value = document.get("family") if isinstance(document.get("family"), Mapping) else {}
    cleanup_value = document.get("cleanup") if isinstance(document.get("cleanup"), Mapping) else {}
    boundary_value = document.get("boundary") if isinstance(document.get("boundary"), Mapping) else {}
    family = _validate_family(family_value, study)
    cleanup = _validate_cleanup(cleanup_value)
    ready, remaining = _validate_boundary(
        boundary_value,
        study,
        now or datetime.now(timezone.utc),
    )
    return Eligibility(
        source=source,
        document=document,
        study=study,
        ready=ready,
        family=family,
        cleanup=cleanup,
        baseline_native_tps=baseline_tps,
        baseline_wall_seconds=baseline_wall,
        quality_kind=quality_kind,
        quality_case_id=quality_case,
        quality_prompt_sha256=prompt_sha,
        boundary_remaining_seconds=remaining,
    )


def pilot_paths(tag: str, *, results: Path = RESULTS) -> PilotPaths:
    prefix = results / tag
    return PilotPaths(
        config=Path(f"{prefix}-candidate-config.json"),
        smoke_plan=Path(f"{prefix}-smoke-plan.json"),
        control=Path(f"{prefix}-control.json"),
        events=Path(f"{prefix}-runtime-events.jsonl"),
        server_log=Path(f"{prefix}-server.log"),
        session_report=Path(f"{prefix}-session-report.json"),
        smoke=Path(f"{prefix}-01-{SMOKE_STEP_ID}.jsonl"),
        quick=Path(f"{prefix}-matched-quick.jsonl"),
        quality=Path(f"{prefix}-quality-pilot.jsonl"),
        quality_texts=Path(f"{prefix}-quality-pilot.jsonl.texts"),
        receipt=Path(f"{prefix}-pilot-controller.json"),
    )


def _assert_unused(paths: PilotPaths) -> None:
    collisions = [str(path) for path in paths.__dict__.values() if os.path.lexists(path)]
    if collisions:
        raise PilotError("pilot would overwrite an existing artifact: " + ", ".join(collisions))


def candidate_config(base: Mapping[str, Any]) -> dict[str, Any]:
    result = json.loads(json.dumps(base))
    args = result.get("args")
    if not isinstance(args, list) or any(not isinstance(item, str) for item in args):
        raise PilotError("baseline config args are malformed")
    if any(item.startswith("--spec-") for item in args):
        raise PilotError("baseline config already contains speculative options")
    result["args"] = [*args, *CANDIDATE_ARGS]
    result["model_name"] = "bonsai2-27b-ptq1_0-dspark-v1-experimental"
    result["pilot_metadata"] = {
        "profile": "bonsai2-ptq1_0-dspark-v1-experimental",
        "experimental_earlier_family_drafter": True,
        "trained_for_bonsai2": False,
        "draft_sha256": DS_PARK_SHA256,
        "draft_inventory_sha256": DS_PARK_INVENTORY_SHA256,
    }
    if FORBIDDEN_ENVIRONMENT_OPTION in json.dumps(result, sort_keys=True):
        raise PilotError("legacy DSpark Markov CUDA control must not enter the dflash candidate")
    if result["args"] != [*base["args"], *CANDIDATE_ARGS]:
        raise PilotError("candidate config changed an option outside the four declared DSpark flags")
    return result


def smoke_plan() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "name": "Bonsai2 PTQ DSpark sole setup generation",
        "steps": [
            {
                "id": SMOKE_STEP_ID,
                "suite": "quick",
                "cases": ["quick_json_extract"],
                "thinking": "none",
                "temperature": 0.0,
                "seed": 0,
                "output_tokens": 256,
                "request_timeout": 180.0,
                "repeat": 1,
            }
        ],
    }


def _exclusive_json(path: Path, value: Mapping[str, Any]) -> Binding:
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
                raise OSError("short exclusive JSON write")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return Binding(path.resolve(), hashlib.sha256(payload).hexdigest(), len(payload))


def verify_dspark_model(*, full_hash: bool) -> dict[str, Any]:
    details = _regular(DS_PARK_MODEL, "DSpark draft model")
    if details.st_size != DS_PARK_SIZE:
        raise PilotError("DSpark draft model size changed")
    if full_hash and sha256_file(DS_PARK_MODEL) != DS_PARK_SHA256:
        raise PilotError("DSpark draft model full SHA-256 verification failed")
    return {
        "path": str(DS_PARK_MODEL),
        "size": details.st_size,
        "sha256": DS_PARK_SHA256,
        "full_sha256_verified": full_hash,
    }


def prepare_candidate(paths: PilotPaths, *, full_hash: bool) -> dict[str, Binding]:
    _assert_unused(paths)
    base = _validate_baseline_config()
    verify_dspark_model(full_hash=full_hash)
    return {
        "config": _exclusive_json(paths.config, candidate_config(base)),
        "smoke_plan": _exclusive_json(paths.smoke_plan, smoke_plan()),
    }


def generation_request_budget() -> dict[str, int]:
    budget = {"model_session_setup": 1, "matched_quick": 2, "quality_pilot_max": 1}
    if sum(budget.values()) != MAX_GENERATION_REQUESTS:
        raise PilotError("pilot request budget is not exactly four")
    return budget


def outer_command(tag: str, paths: PilotPaths) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "runtime/gpu_window.py"),
        "--long-session",
        "--wait-seconds",
        str(int(IDLE_WAIT_SECONDS)),
        "--max-seconds",
        str(int(WINDOW_SECONDS)),
        "--",
        str(ROOT / "runtime/model_session.py"),
        "--runtime-backend",
        "llama",
        "--config",
        str(paths.config),
        "--plan",
        str(paths.smoke_plan),
        "--tag",
        tag,
        "--startup-timeout",
        str(int(STARTUP_TIMEOUT)),
        "--hold-after-plan",
        "--control-file",
        str(paths.control),
    ]


def quick_command(session: Any, paths: PilotPaths) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "bench.py"),
        "--endpoint",
        ENDPOINT,
        "--backend",
        "openai",
        "--model",
        session.model_name,
        "--context",
        str(CONTEXT),
        "--thinking",
        "medium",
        "--chat-template-thinking",
        "on",
        "--temperature",
        "0",
        "--seed",
        "0",
        "--suite",
        "quick",
        "--cases",
        ",".join(QUICK_IDS),
        "--timeout",
        str(int(REQUEST_TIMEOUT)),
        "--repeat",
        "1",
        "--num-predict",
        "65536",
        "--fail-on-case",
        "--output",
        str(paths.quick),
    ]


def quality_command(tag: str, eligibility: Eligibility, session: Any, paths: PilotPaths) -> list[str]:
    if eligibility.quality_kind == "hard":
        return [
            sys.executable,
            str(ROOT / "bench.py"),
            "--endpoint", ENDPOINT,
            "--backend", "openai",
            "--model", session.model_name,
            "--context", str(CONTEXT),
            "--thinking", "medium",
            "--chat-template-thinking", "on",
            "--temperature", "0",
            "--seed", "0",
            "--suite", "hard",
            "--cases", eligibility.quality_case_id,
            "--timeout", str(int(REQUEST_TIMEOUT)),
            "--repeat", "1",
            "--num-predict", "65536",
            "--fail-on-case",
            "--output", str(paths.quality),
        ]
    if eligibility.quality_kind == "code":
        return [
            sys.executable,
            str(ROOT / "runtime/specialist_bench.py"),
            "--endpoint", ENDPOINT,
            "--model", session.model_name,
            "--context", str(CONTEXT),
            "--tag", f"{tag}-quality",
            "--profile", "bonsai2-ptq1_0-dspark-v1-experimental",
            "--jsonl", str(paths.quality),
            "--visible-dir", str(paths.quality_texts),
            "--groups", "code",
            "--cases", eligibility.quality_case_id,
            "--thinking", "medium",
            "--chat-template-thinking", "on",
            "--temperature", "0",
            "--seed", "0",
            "--output-cap", "65536",
            "--timeout", str(int(REQUEST_TIMEOUT)),
            "--repeat", "1",
        ]
    raise PilotError("quality pilot kind is neither code nor hard")


def _metadata_rows(path: Path, description: str) -> list[dict[str, Any]]:
    _regular(path, description, MAX_JSONL_BYTES)
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise PilotError(f"{description} row {line_number} is not an object")
                request = row.get("request") if isinstance(row.get("request"), Mapping) else {}
                response = row.get("response") if isinstance(row.get("response"), Mapping) else {}
                timing = response.get("timing") if isinstance(response.get("timing"), Mapping) else {}
                native = (
                    response.get("openai_native_timings")
                    if isinstance(response.get("openai_native_timings"), Mapping)
                    else {}
                )
                visible = row.get("visible_text") if isinstance(row.get("visible_text"), Mapping) else {}
                metrics = (
                    row.get("response_metrics")
                    if isinstance(row.get("response_metrics"), Mapping)
                    else {}
                )
                rows.append(
                    {
                        "schema_version": row.get("schema_version"),
                        "run_id": row.get("run_id"),
                        "call_id": row.get("call_id"),
                        "suite": row.get("suite"),
                        "case_id": row.get("case_id"),
                        "case_version": row.get("case_version"),
                        "case_sha256": row.get("case_sha256"),
                        "request": {
                            key: request.get(key)
                            for key in (
                                "model", "context_max_tokens", "thinking",
                                "chat_template_thinking", "temperature", "seed",
                                "output_token_limit", "timeout_seconds", "prompt_sha256",
                            )
                        },
                        "response": {
                            "actual_completion_tokens": response.get("actual_completion_tokens"),
                            "done_reason": response.get("done_reason", metrics.get("finish_reason")),
                            "stream_complete": response.get("stream_complete"),
                            "truncated": response.get("truncated"),
                            "wall_seconds": timing.get("wall_seconds", metrics.get("wall_seconds")),
                            "native": dict(native),
                        },
                        "evaluation_passed": (
                            row.get("evaluation", {}).get("passed")
                            if isinstance(row.get("evaluation"), Mapping)
                            else None
                        ),
                        "error_present": row.get("error") is not None,
                        "visible_text": {
                            key: visible.get(key)
                            for key in ("path", "sha256", "chars", "bytes", "reasoning_excluded")
                        },
                    }
                )
                row.clear()
    except PilotError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PilotError(f"{description} is malformed JSONL") from exc
    return rows


def _eos(value: Any) -> bool:
    return value in {"stop", "eos", "end_turn"}


def _cap_finish(value: Any) -> bool:
    return value in {"length", "max_tokens", "max_output_tokens"}


def _validate_request_row(
    row: Mapping[str, Any], case_id: str, prompt_sha256: str, *, thinking: str, cap: int,
    timeout: float,
) -> dict[str, Any]:
    request = row.get("request") if isinstance(row.get("request"), Mapping) else {}
    response = row.get("response") if isinstance(row.get("response"), Mapping) else {}
    tokens = response.get("actual_completion_tokens")
    wall = response.get("wall_seconds")
    if not (
        row.get("schema_version") == 1
        and row.get("case_id") == case_id
        and isinstance(row.get("call_id"), str)
        and bool(row.get("call_id"))
        and request.get("context_max_tokens") == CONTEXT
        and request.get("thinking") == thinking
        and request.get("chat_template_thinking") == ("off" if thinking == "none" else "on")
        and request.get("temperature") == 0
        and request.get("seed") == 0
        and request.get("output_token_limit") == cap
        and request.get("timeout_seconds") == timeout
        and request.get("prompt_sha256") == prompt_sha256
    ):
        raise PilotError(f"candidate request identity or controls changed: {case_id}")
    evaluation_passed = row.get("evaluation_passed")
    error_present = row.get("error_present")
    stream_complete = response.get("stream_complete")
    truncated = response.get("truncated")
    finish_reason = response.get("done_reason")
    if not isinstance(evaluation_passed, bool) or not isinstance(error_present, bool):
        raise PilotError(f"candidate request outcome is malformed: {case_id}")
    if stream_complete is not None and not isinstance(stream_complete, bool):
        raise PilotError(f"candidate stream status is malformed: {case_id}")
    if truncated is not None and not isinstance(truncated, bool):
        raise PilotError(f"candidate truncation status is malformed: {case_id}")
    valid_tokens = isinstance(tokens, int) and not isinstance(tokens, bool) and tokens >= 0
    valid_wall = (
        isinstance(wall, (int, float))
        and not isinstance(wall, bool)
        and math.isfinite(float(wall))
        and float(wall) > 0
    )
    recognized_finish = _eos(finish_reason) or _cap_finish(finish_reason)
    if not error_present and stream_complete is True and recognized_finish:
        if not valid_tokens or not valid_wall or not isinstance(truncated, bool):
            raise PilotError(f"candidate terminal request metrics are malformed: {case_id}")
    nonempty = tokens > 0 if valid_tokens else None
    eos = _eos(finish_reason) if recognized_finish else None
    runtime_failure = bool(
        error_present
        or stream_complete is not True
        or not recognized_finish
    )
    clean_pass = bool(
        not runtime_failure
        and evaluation_passed is True
        and eos is True
        and nonempty is True
        and truncated is False
    )
    gate_failures: list[str] = []
    if error_present:
        gate_failures.append("request_error")
    if stream_complete is not True:
        gate_failures.append("incomplete_stream")
    if not recognized_finish:
        gate_failures.append("unknown_finish")
    if evaluation_passed is False:
        gate_failures.append("semantic_failure")
    if eos is False or truncated is True:
        gate_failures.append("output_cap_or_non_eos")
    if nonempty is False:
        gate_failures.append("empty_output")
    return {
        "case_id": case_id,
        "call_id": row.get("call_id"),
        "output_tokens": tokens if valid_tokens else None,
        "wall_seconds": float(wall) if valid_wall else None,
        "finish_reason": finish_reason,
        "stream_complete": stream_complete,
        "truncated": truncated,
        "evaluation_passed": evaluation_passed,
        "error_free": not error_present,
        "eos": eos,
        "nonempty": nonempty,
        "passed": clean_pass,
        "outcome_class": (
            "runtime_error" if runtime_failure else "clean_pass" if clean_pass else "gate_failure"
        ),
        "gate_failures": gate_failures,
    }


def validate_smoke(
    path: Path,
    events_path: Path,
    session: Any,
    *,
    gpu_processes: Callable[[], list[dict[str, Any]]],
    server_log: Path,
) -> dict[str, Any]:
    from runtime import specialist_study_trial as study

    benchmark, _ = _expected_prompts()
    events = study._read_event_rows(events_path)
    completions = [row for row in events if row.get("event") == "step_complete"]
    starts = [row for row in events if row.get("event") == "server_started"]
    if not (
        len(completions) == 1
        and completions[0].get("id") == SMOKE_STEP_ID
        and completions[0].get("result") == str(path)
        and completions[0].get("returncode") in {0, 1}
        and isinstance(completions[0].get("passed"), bool)
        and completions[0].get("passed") is (completions[0].get("returncode") == 0)
        and len(starts) == 1
        and all(row.get("run_id") == session.run_id for row in events)
    ):
        raise PilotError("smoke is not bound to the exact active held session")
    rows = _metadata_rows(path, "DSpark smoke result")
    if len(rows) != 1:
        raise PilotError("DSpark smoke must contain exactly one generation")
    summary = _validate_request_row(
        rows[0],
        "quick_json_extract",
        benchmark["quick_json_extract"]["prompt_sha256"],
        thinking="none",
        cap=256,
        timeout=180.0,
    )
    summary["step_returncode"] = completions[0].get("returncode")
    if completions[0].get("passed") is not summary["evaluation_passed"]:
        raise PilotError("smoke lifecycle result disagrees with its benchmark row")
    if summary["outcome_class"] == "runtime_error":
        raise PilotStageError("smoke", "smoke request ended with a runtime error", summary)
    native = rows[0]["response"].get("native")
    native = native if isinstance(native, Mapping) else {}
    if "draft_n" not in native or "draft_n_accepted" not in native:
        raise PilotError("smoke lacks distinct draft proposal and acceptance counters")
    proposed = native.get("draft_n")
    accepted = native.get("draft_n_accepted")
    if not (
        isinstance(proposed, (int, float))
        and not isinstance(proposed, bool)
        and math.isfinite(float(proposed))
        and float(proposed) > 0
        and isinstance(accepted, (int, float))
        and not isinstance(accepted, bool)
        and math.isfinite(float(accepted))
        and 0 <= float(accepted) <= float(proposed)
    ):
        raise PilotError("smoke draft counters are invalid")
    server_pid = starts[0].get("pid")
    gpu_rows = gpu_processes()
    resident = [
        row for row in gpu_rows
        if row.get("pid") == server_pid
        and isinstance(row.get("used_mib"), (int, float))
        and row.get("used_mib") > 0
    ]
    if len(resident) != 1:
        raise PilotError("owned candidate server has no unique GPU process residency evidence")
    _regular(server_log, "candidate server log")
    draft_name = DS_PARK_MODEL.name.lower()
    draft_logged = False
    cuda_logged = False
    with server_log.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            lowered = line.lower()
            draft_logged |= draft_name in lowered
            cuda_logged |= "cuda" in lowered and ("buffer" in lowered or "offload" in lowered)
    if not draft_logged or not cuda_logged:
        raise PilotError("server log lacks explicit draft-load and CUDA residency evidence")
    summary.update(
        draft_proposals=proposed,
        draft_accepted=accepted,
        draft_engaged=True,
        gpu_residency={"pid": server_pid, "used_mib": resident[0]["used_mib"]},
        draft_load_logged=True,
        cuda_residency_logged=True,
    )
    if summary["passed"] is not True:
        raise PilotGateStopped("smoke", "smoke stopped at its semantic or output-cap gate", summary)
    return summary


def validate_quick_candidate(path: Path, eligibility: Eligibility) -> dict[str, Any]:
    benchmark, _ = _expected_prompts()
    rows = _metadata_rows(path, "matched candidate quick result")
    if len(rows) > 2 or [row.get("case_id") for row in rows] != list(QUICK_IDS[:len(rows)]):
        raise PilotError("matched candidate quick result is not a frozen quick2 prefix")
    summaries: list[dict[str, Any]] = []
    native_rows: list[dict[str, Any]] = []
    for row, case_id in zip(rows, QUICK_IDS, strict=False):
        summary = _validate_request_row(
            row,
            case_id,
            benchmark[case_id]["prompt_sha256"],
            thinking="medium",
            cap=65_536,
            timeout=REQUEST_TIMEOUT,
        )
        native = row["response"].get("native")
        native = native if isinstance(native, Mapping) else {}
        predicted_n = native.get("predicted_n")
        predicted_ms = native.get("predicted_ms")
        valid_native = (
            isinstance(predicted_n, int)
            and not isinstance(predicted_n, bool)
            and predicted_n >= 2
            and isinstance(predicted_ms, (int, float))
            and not isinstance(predicted_ms, bool)
            and math.isfinite(float(predicted_ms))
            and float(predicted_ms) > 0
        )
        native_row = (
            {
                "native_predicted_n": predicted_n,
                "native_predicted_seconds": float(predicted_ms) / 1000.0,
            }
            if valid_native
            else {
                "native_predicted_n": None,
                "native_predicted_seconds": None,
            }
        )
        summary.update(native_row)
        if valid_native:
            native_rows.append(native_row)
        if "draft_n" in native:
            summary["draft_proposals"] = native.get("draft_n")
        if "draft_n_accepted" in native:
            summary["draft_accepted"] = native.get("draft_n_accepted")
        summaries.append(summary)
    pooled = native_pooled_tps(native_rows) if native_rows else None
    walls = [row["wall_seconds"] for row in summaries if row["wall_seconds"] is not None]
    wall = sum(walls) if walls else None
    tps_threshold = 1.05 * eligibility.baseline_native_tps
    wall_threshold = 0.95 * eligibility.baseline_wall_seconds
    full_metrics = len(rows) == 2 and len(native_rows) == 2 and len(walls) == 2
    tps_passed = pooled >= tps_threshold if full_metrics and pooled is not None else None
    wall_passed = wall <= wall_threshold if full_metrics and wall is not None else None
    result = {
        "cases": summaries,
        "observed_case_count": len(summaries),
        "expected_case_count": 2,
        "pooled_native_tps": pooled,
        "baseline_pooled_native_tps": eligibility.baseline_native_tps,
        "minimum_native_tps": tps_threshold,
        "combined_wall_seconds": wall,
        "baseline_combined_wall_seconds": eligibility.baseline_wall_seconds,
        "maximum_combined_wall_seconds": wall_threshold,
        "native_tps_gate_passed": tps_passed,
        "combined_wall_gate_passed": wall_passed,
        "passed": False,
    }
    if any(row["outcome_class"] == "runtime_error" for row in summaries):
        raise PilotStageError("quick", "matched quick ended with a runtime error", result)
    if any(row["passed"] is not True for row in summaries):
        result["stop_reason"] = "semantic_or_output_cap_gate"
        raise PilotGateStopped("quick", "matched quick stopped at a semantic or output-cap gate", result)
    if len(rows) != 2:
        raise PilotStageError("quick", "matched quick ended before both requests were recorded", result)
    if not full_metrics:
        raise PilotStageError("quick", "matched quick lacks complete native timing evidence", result)
    if tps_passed is not True or wall_passed is not True:
        result["stop_reason"] = "performance_gate"
        raise PilotGateStopped(
            "quick",
            "candidate quick2 did not meet both pooled native TPS and combined-wall continuation gates",
            result,
        )
    result["passed"] = True
    return result


def validate_quality_candidate(path: Path, eligibility: Eligibility) -> dict[str, Any]:
    benchmark, specialist = _expected_prompts()
    rows = _metadata_rows(path, "single candidate quality pilot")
    if len(rows) != 1:
        raise PilotError("quality pilot must contain exactly one request")
    identity = (
        specialist[eligibility.quality_case_id]
        if eligibility.quality_kind == "code"
        else benchmark[eligibility.quality_case_id]
    )
    row = rows[0]
    if row.get("case_version") != identity["case_version"]:
        raise PilotError("quality pilot case version changed")
    if eligibility.quality_kind == "code" and row.get("case_sha256") != identity["case_sha256"]:
        raise PilotError("quality pilot specialist case hash changed")
    result = {
        **_validate_request_row(
            row,
            eligibility.quality_case_id,
            eligibility.quality_prompt_sha256,
            thinking="medium",
            cap=65_536,
            timeout=REQUEST_TIMEOUT,
        ),
        "kind": eligibility.quality_kind,
        "scope": "one matched case only; no full code3 or hard4 candidate score",
    }
    if eligibility.quality_kind == "code":
        visible = row.get("visible_text") if isinstance(row.get("visible_text"), Mapping) else {}
        call_id = str(row.get("call_id"))
        expected_path = path.with_name(path.name + ".texts") / f"{call_id}.txt"
        size = visible.get("bytes")
        chars = visible.get("chars")
        if not (
            _same_path(visible.get("path"), expected_path)
            and isinstance(size, int)
            and not isinstance(size, bool)
            and size >= 0
            and isinstance(chars, int)
            and not isinstance(chars, bool)
            and chars >= 0
            and visible.get("reasoning_excluded") is True
        ):
            raise PilotError("quality pilot visible-answer binding is incomplete")
        visible_path = _path(visible.get("path"), "quality pilot visible-answer file.path")
        digest = visible.get("sha256")
        details = _regular(visible_path, "quality pilot visible-answer file")
        if (
            not isinstance(digest, str)
            or SHA256_RE.fullmatch(digest) is None
            or details.st_size != size
            or sha256_file(visible_path) != digest
        ):
            raise PilotError("quality pilot visible-answer file differs from its binding")
        binding = Binding(visible_path, digest, size)
        result["visible_answer"] = {
            "path": str(binding.path),
            "sha256": binding.sha256,
            "size": binding.size,
            "chars": chars,
            "reasoning_excluded": True,
        }
    if result["outcome_class"] == "runtime_error":
        raise PilotStageError("quality", "quality pilot ended with a runtime error", result)
    if result["passed"] is not True:
        raise PilotGateStopped(
            "quality", "quality pilot stopped at its semantic or output-cap gate", result
        )
    return result


def _port_clear(port: int) -> bool:
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        probe.settimeout(0.3)
        if probe.connect_ex(("127.0.0.1", port)) == 0:
            return False
    finally:
        probe.close()
    binder = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        binder.bind(("127.0.0.1", port))
    except OSError:
        return False
    finally:
        binder.close()
    return True


def _process_start_ticks(pid: int) -> str | None:
    try:
        payload = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise PilotError(f"could not inspect prior PTQ server PID {pid}") from exc
    closing = payload.rfind(")")
    if closing < 0:
        raise PilotError(f"prior PTQ server PID {pid} has malformed procfs state")
    fields = payload[closing + 1 :].split()
    try:
        return fields[19]
    except IndexError as exc:
        raise PilotError(f"prior PTQ server PID {pid} has malformed procfs state") from exc


def live_boundary_preflight() -> dict[str, Any]:
    from runtime import gpu_window

    state = gpu_window.original_state()
    health = state.get("Health") if isinstance(state.get("Health"), Mapping) else {}
    try:
        api = gpu_window._api_request("GET", "/api/tags", 5.0)
    except Exception as exc:
        raise PilotError(f"original API health probe failed: {type(exc).__name__}") from exc
    ports = {str(port): _port_clear(port) for port in (18080, 18081)}
    observed_ticks = _process_start_ticks(PTQ_SERVER_PID)
    evidence = {
        "checked_at": utc_now(),
        "original_running": state.get("Running") is True,
        "original_paused": state.get("Paused") is True,
        "original_health_status": health.get("Status", state.get("HealthStatus")),
        "original_api_healthy": isinstance(api, Mapping),
        "ports_clear": ports,
        "prior_ptq_server": {
            "pid": PTQ_SERVER_PID,
            "expected_start_ticks": PTQ_SERVER_START_TICKS,
            "observed_start_ticks": observed_ticks,
            "gone": observed_ticks != PTQ_SERVER_START_TICKS,
        },
    }
    if not (
        evidence["original_running"] is True
        and evidence["original_paused"] is False
        and evidence["original_api_healthy"] is True
        and all(ports.values())
        and evidence["prior_ptq_server"]["gone"] is True
    ):
        raise PilotError("live boundary is not running, API-healthy, unpaused, and port-clear")
    return evidence


def _run_owned(
    command: Sequence[str],
    outer: subprocess.Popen[Any],
    deadline: float,
    signals: SignalLatch,
) -> dict[str, Any]:
    started = time.monotonic()
    process = subprocess.Popen(
        list(command),
        stdin=subprocess.DEVNULL,
        start_new_session=True,
        close_fds=True,
    )
    cleanup = {"pid": process.pid, "term_sent": False, "kill_sent": False, "stopped": False}
    try:
        while process.poll() is None:
            signals.check()
            if outer.poll() is not None:
                raise PilotError("GPU window exited while a candidate request helper was running")
            if time.monotonic() >= deadline:
                raise PilotError("owned diagnostic window expired during a candidate request")
            time.sleep(0.25)
    except BaseException as exc:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
                cleanup["term_sent"] = True
            except ProcessLookupError:
                pass
            until = time.monotonic() + 35.0
            while process.poll() is None and time.monotonic() < until:
                time.sleep(0.1)
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                    cleanup["kill_sent"] = True
                except ProcessLookupError:
                    pass
                until = time.monotonic() + 5.0
                while process.poll() is None and time.monotonic() < until:
                    time.sleep(0.1)
        cleanup["stopped"] = process.poll() is not None
        setattr(exc, "owned_helper_cleanup", cleanup)
        raise
    cleanup["stopped"] = True
    return {
        "argv": list(command),
        "returncode": int(process.returncode),
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "cleanup": cleanup,
    }


def _wait_holding(
    outer: subprocess.Popen[Any], paths: PilotPaths, tag: str, deadline: float, signals: SignalLatch,
) -> Any:
    from runtime import specialist_study_trial as study

    args = argparse.Namespace(
        tag=tag,
        config=paths.config,
        plan=paths.smoke_plan,
        runtime_backend="llama",
    )
    session = study.wait_for_holding(
        outer,
        paths.events,
        args=args,
        control=paths.control,
        deadline=deadline,
        signals=signals,
    )
    if session.model_name != "bonsai2-27b-ptq1_0-dspark-v1-experimental" or session.context != CONTEXT:
        raise PilotError("held candidate session has the wrong model alias or context")
    return session


def _session_cleanup(
    outer: subprocess.Popen[Any] | None,
    paths: PilotPaths,
    session: Any | None,
) -> dict[str, Any]:
    from runtime import specialist_study_trial as study

    cleanup: dict[str, Any] = {
        "control_stop_requested": False,
        "outer_sigterm_sent": False,
        "orchestrator_sigkill_sent": False,
    }
    if outer is None:
        return cleanup
    if session is not None and os.path.lexists(paths.control):
        cleanup["control_stop_requested"] = study.request_session_stop(paths.control)
    elif outer.poll() is None:
        cleanup["outer_sigterm_sent"] = study.request_outer_termination(outer)
    returncode, delayed = study.wait_for_outer_exit(
        outer,
        timeout=study.RESTORATION_WAIT_SECONDS,
        control=paths.control,
    )
    cleanup["control_stop_requested"] |= delayed
    cleanup["outer_returncode"] = returncode
    cleanup["outer_exited"] = returncode is not None
    if returncode is None:
        raise PilotError("owned GPU window did not exit during the restoration wait")
    return cleanup


def _final_cleanup_evidence(paths: PilotPaths, session: Any, cleanup: Mapping[str, Any]) -> dict[str, Any]:
    from runtime import specialist_study_trial as study

    report, binding = _json(paths.session_report, "candidate model_session report")
    server = report.get("server") if isinstance(report.get("server"), Mapping) else {}
    server_cleanup = server.get("cleanup") if isinstance(server.get("cleanup"), Mapping) else {}
    events = study._read_event_rows(paths.events)
    restoration = live_boundary_preflight()
    session_status = report.get("status")
    if not (
        report.get("session_id") == session.run_id
        and session_status in {"passed", "completed_with_step_failures"}
        and report.get("problem") is None
        and report.get("source_config_sha256") == sha256_file(paths.config)
        and isinstance(report.get("plan"), Mapping)
        and report["plan"].get("sha256") == sha256_file(paths.smoke_plan)
        and server_cleanup.get("stopped") is True
        and server_cleanup.get("kill_sent") is False
        and cleanup.get("outer_exited") is True
        and cleanup.get("orchestrator_sigkill_sent") is False
        and any(row.get("event") == "server_stopped" for row in events)
        and any(row.get("event") == "session_complete" for row in events)
        and all(row.get("run_id") == session.run_id for row in events)
    ):
        raise PilotError("candidate lifecycle lacks exact clean terminal evidence")
    return {
        "session_report": {
            "path": str(binding.path), "sha256": binding.sha256, "size": binding.size,
        },
        "server_cleanup": dict(server_cleanup),
        "session_status": session_status,
        "semantic_step_failures": session_status == "completed_with_step_failures",
        "runtime_events": {
            "path": str(paths.events.resolve()),
            "sha256": sha256_file(paths.events),
            "size": paths.events.stat().st_size,
        },
        "restoration": restoration,
    }


def _result_row_counts(path: Path) -> dict[str, Any]:
    if not os.path.lexists(path):
        return {
            "artifact_exists": False,
            "observed_result_rows": 0,
            "observed_terminal_or_error_rows": 0,
            "observation_error": None,
        }
    try:
        rows = _metadata_rows(path, "candidate request accounting source")
    except PilotError as exc:
        return {
            "artifact_exists": True,
            "observed_result_rows": None,
            "observed_terminal_or_error_rows": None,
            "observation_error": str(exc),
        }
    terminal = 0
    for row in rows:
        response = row.get("response") if isinstance(row.get("response"), Mapping) else {}
        if row.get("error_present") is True or (
            response.get("stream_complete") is True
            and (_eos(response.get("done_reason")) or _cap_finish(response.get("done_reason")))
        ):
            terminal += 1
    return {
        "artifact_exists": True,
        "observed_result_rows": len(rows),
        "observed_terminal_or_error_rows": terminal,
        "observation_error": None,
    }


def generation_request_accounting(
    paths: PilotPaths,
    *,
    launch_attempted: Mapping[str, bool],
    stage_completed: Mapping[str, bool],
) -> dict[str, Any]:
    allocation = {
        "smoke": generation_request_budget()["model_session_setup"],
        "matched_quick": generation_request_budget()["matched_quick"],
        "quality_pilot": generation_request_budget()["quality_pilot_max"],
    }
    sources = {
        "smoke": paths.smoke,
        "matched_quick": paths.quick,
        "quality_pilot": paths.quality,
    }
    stages: dict[str, dict[str, Any]] = {}
    lower_total = 0
    upper_total = 0
    exact_values: list[int | None] = []
    violation = False
    for stage, reserved in allocation.items():
        observation = _result_row_counts(sources[stage])
        launched = launch_attempted.get(stage) is True
        completed = stage_completed.get(stage) is True
        rows = observation["observed_result_rows"]
        terminal = observation["observed_terminal_or_error_rows"]
        lower = rows if isinstance(rows, int) else 0
        if lower > reserved:
            violation = True
        if not launched and rows == 0:
            exact: int | None = 0
        elif completed and isinstance(rows, int) and rows > 0 and terminal == rows:
            exact = rows
        else:
            exact = None
        upper = exact if exact is not None else reserved if launched else max(lower, 0)
        lower_total += lower
        upper_total += upper
        exact_values.append(exact)
        stages[stage] = {
            "allocated_request_slots": reserved,
            "launch_attempted": launched,
            "stage_completed": completed,
            **observation,
            "attempts_lower_bound": lower,
            "attempts_upper_bound": upper,
            "exact_attempts": exact,
            "has_unobserved_attempt_uncertainty": exact is None,
        }
    exact_total = (
        sum(value for value in exact_values if value is not None)
        if all(value is not None for value in exact_values)
        else None
    )
    if lower_total > MAX_GENERATION_REQUESTS or upper_total > MAX_GENERATION_REQUESTS:
        violation = True
    return {
        "stages": stages,
        "observed_attempts_lower_bound": lower_total,
        "possible_attempts_upper_bound": upper_total,
        "exact_attempts": exact_total,
        "maximum": MAX_GENERATION_REQUESTS,
        "budget_violation": violation,
        "reserved_slots_are_not_reported_as_actual_attempts": True,
    }


def execute(tag: str, eligibility: Eligibility, *, paths: PilotPaths | None = None) -> tuple[dict[str, Any], int]:
    from runtime import gpu_window

    selected_paths = paths or pilot_paths(tag)
    budget = generation_request_budget()
    boundary_before = live_boundary_preflight()
    _assert_unused(selected_paths)
    verify_dspark_model(full_hash=True)
    # Full draft hashing can take long enough for the boundary state to change.
    # Revalidate both the immutable READY identity and live restoration before
    # the first process is created.
    _, remaining_before_launch = _validate_boundary(
        eligibility.document.get("boundary")
        if isinstance(eligibility.document.get("boundary"), Mapping)
        else {},
        eligibility.study,
        datetime.now(timezone.utc),
    )
    boundary_before_launch = live_boundary_preflight()
    prepared = prepare_candidate(selected_paths, full_hash=False)
    commands: list[dict[str, Any]] = []
    outer: subprocess.Popen[Any] | None = None
    session: Any | None = None
    signals = SignalLatch()
    previous = {signum: signal.getsignal(signum) for signum in (signal.SIGINT, signal.SIGTERM)}
    started = time.monotonic()
    deadline = started + WINDOW_SECONDS - RESTORATION_RESERVE_SECONDS
    status = "infrastructure_failed"
    problem: dict[str, Any] | None = None
    smoke: dict[str, Any] | None = None
    quick: dict[str, Any] | None = None
    quality: dict[str, Any] | None = None
    cleanup: dict[str, Any] = {}
    cleanup_evidence: dict[str, Any] | None = None
    gate_outcome: dict[str, Any] | None = None
    launch_attempted = {"smoke": False, "matched_quick": False, "quality_pilot": False}
    code = 1
    try:
        for signum in previous:
            signal.signal(signum, signals.handle)
        outer_argv = outer_command(tag, selected_paths)
        commands.append({"kind": "gpu_window_model_session", "argv": outer_argv})
        launch_attempted["smoke"] = True
        outer = subprocess.Popen(
            outer_argv,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
            close_fds=True,
        )
        session = _wait_holding(outer, selected_paths, tag, deadline, signals)
        smoke = validate_smoke(
            selected_paths.smoke,
            selected_paths.events,
            session,
            gpu_processes=gpu_window.gpu_compute_processes,
            server_log=selected_paths.server_log,
        )
        quick_argv = quick_command(session, selected_paths)
        launch_attempted["matched_quick"] = True
        quick_run = _run_owned(quick_argv, outer, deadline, signals)
        commands.append({"kind": "matched_quick2", **quick_run})
        quick = validate_quick_candidate(selected_paths.quick, eligibility)
        if quick_run["returncode"] != 0:
            raise PilotError("matched quick2 helper returned nonzero despite a passing result")
        quality_argv = quality_command(tag, eligibility, session, selected_paths)
        launch_attempted["quality_pilot"] = True
        quality_run = _run_owned(quality_argv, outer, deadline, signals)
        commands.append({"kind": "single_quality_pilot", **quality_run})
        quality = validate_quality_candidate(selected_paths.quality, eligibility)
        if quality_run["returncode"] != 0:
            raise PilotError("single quality helper returned nonzero despite a passing result")
        status = "complete"
        code = 0
    except PilotInterrupted as exc:
        problem = {"error_type": type(exc).__name__, "error": str(exc), "signal": exc.signum}
        status = "interrupted"
        code = 128 + exc.signum
    except PilotGateStopped as exc:
        if exc.stage == "smoke":
            smoke = exc.evidence
        elif exc.stage == "quick":
            quick = exc.evidence
        elif exc.stage == "quality":
            quality = exc.evidence
        gate_outcome = {
            "stage": exc.stage,
            "reason": str(exc),
            "classification": "expected_gate_stop",
        }
        status = "stopped_at_gate"
        code = 0
    except PilotStageError as exc:
        if exc.stage == "smoke":
            smoke = exc.evidence
        elif exc.stage == "quick":
            quick = exc.evidence
        elif exc.stage == "quality":
            quality = exc.evidence
        problem = {
            "error_type": type(exc).__name__,
            "error": str(exc),
            "stage": exc.stage,
        }
    except (PilotError, OSError, subprocess.SubprocessError) as exc:
        helper_cleanup = getattr(exc, "owned_helper_cleanup", None)
        problem = {"error_type": type(exc).__name__, "error": str(exc)}
        if helper_cleanup is not None:
            problem["owned_helper_cleanup"] = helper_cleanup
    finally:
        try:
            cleanup = _session_cleanup(outer, selected_paths, session)
        except (PilotError, OSError) as exc:
            problem = problem or {"error_type": type(exc).__name__, "error": str(exc)}
            code = 1
            status = "cleanup_failed"
        if outer is not None and session is not None and cleanup.get("outer_exited") is True:
            try:
                cleanup_evidence = _final_cleanup_evidence(selected_paths, session, cleanup)
            except (PilotError, OSError) as exc:
                problem = problem or {"error_type": type(exc).__name__, "error": str(exc)}
                code = 1
                status = "cleanup_failed"
        for signum, handler in previous.items():
            signal.signal(signum, handler)
    stage_completed = {
        "smoke": cleanup_evidence is not None,
        "matched_quick": any(row.get("kind") == "matched_quick2" for row in commands),
        "quality_pilot": any(row.get("kind") == "single_quality_pilot" for row in commands),
    }
    request_accounting = generation_request_accounting(
        selected_paths,
        launch_attempted=launch_attempted,
        stage_completed=stage_completed,
    )
    if request_accounting["budget_violation"] is True:
        problem = problem or {
            "error_type": "PilotError", "error": "generation request budget exceeded",
        }
        code = 1
        status = "infrastructure_failed"
    receipt = {
        "schema_version": 1,
        "kind": "bonsai2_dspark_pilot_execution",
        "status": status,
        "tag": tag,
        "created_at": utc_now(),
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "eligibility": {
            "path": str(eligibility.source.path),
            "sha256": eligibility.source.sha256,
            "size": eligibility.source.size,
            "study_sha256": eligibility.study.sha256,
            "ready_sha256": eligibility.ready.sha256,
            "family_sha256": eligibility.family.sha256,
            "cleanup_sha256": eligibility.cleanup.sha256,
            "boundary_remaining_seconds_at_acceptance": eligibility.boundary_remaining_seconds,
            "quality_selection": {
                "kind": eligibility.quality_kind,
                "case_id": eligibility.quality_case_id,
                "prompt_sha256": eligibility.quality_prompt_sha256,
            },
        },
        "candidate": {
            key: {"path": str(value.path), "sha256": value.sha256, "size": value.size}
            for key, value in prepared.items()
        },
        "draft": {
            "path": str(DS_PARK_MODEL),
            "size": DS_PARK_SIZE,
            "sha256": DS_PARK_SHA256,
            "full_sha256_verified_for_execution": True,
            "architecture": "dflash",
            "legacy_markov_cuda_option_used": False,
        },
        "generation_budget": {
            **budget,
            "maximum": MAX_GENERATION_REQUESTS,
            "accounting": request_accounting,
            "no_fifth_setup_generation": True,
        },
        "live_boundary_before": boundary_before,
        "live_boundary_before_launch": boundary_before_launch,
        "boundary_remaining_seconds_before_launch": remaining_before_launch,
        "session": (
            {
                "run_id": session.run_id,
                "model_name": session.model_name,
                "context": session.context,
                "initial_load_seconds": session.initial_load_seconds,
            }
            if session is not None
            else None
        ),
        "commands": commands,
        "smoke": smoke,
        "quick": quick,
        "quality": quality,
        "gate_outcome": gate_outcome,
        "cleanup": cleanup,
        "cleanup_evidence": cleanup_evidence,
        "artifacts": {
            name: {
                "path": str(path.resolve()),
                "exists": path.is_file() and not path.is_symlink(),
                "size": path.stat().st_size if path.is_file() and not path.is_symlink() else None,
                "sha256": sha256_file(path) if path.is_file() and not path.is_symlink() else None,
            }
            for name, path in {
                "smoke": selected_paths.smoke,
                "matched_quick": selected_paths.quick,
                "single_quality": selected_paths.quality,
                "runtime_events": selected_paths.events,
                "server_log": selected_paths.server_log,
            }.items()
        },
        "problem": problem,
        "claims": {
            "full_code3_or_hard4_candidate_score": False,
            "official_or_trained_bonsai2_paired_head": False,
            "private_reasoning_retained": False,
        },
    }
    _exclusive_json(selected_paths.receipt, receipt)
    return receipt, code


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eligibility", type=Path, required=True)
    parser.add_argument("--tag", required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--execute", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if TAG_RE.fullmatch(args.tag) is None:
        parser.error("--tag must contain only letters, digits, dot, underscore, or hyphen")
    try:
        eligibility = validate_eligibility(args.eligibility)
        if args.check_only:
            summary = {
                "eligible": True,
                "mode": "CPU-only validation; no candidate files or processes created",
                "eligibility_sha256": eligibility.source.sha256,
                "boundary_remaining_seconds": round(eligibility.boundary_remaining_seconds, 3),
                "quality_selection": {
                    "kind": eligibility.quality_kind,
                    "case_id": eligibility.quality_case_id,
                },
                "max_generation_requests": MAX_GENERATION_REQUESTS,
            }
            print(json.dumps(summary, sort_keys=True))
            return 0
        _, code = execute(args.tag, eligibility)
        return code
    except PilotError as exc:
        print(f"bonsai_dspark_pilot: {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
