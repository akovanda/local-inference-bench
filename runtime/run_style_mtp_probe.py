#!/usr/bin/env python3
"""Inspect or run the predeclared StyleTune Gemma-4 MTP prose probe.

The default mode is metadata-only inspection.  ``--execute`` is required to
open an owned GPU window.  Generated prose is left in the existing sampler
probe artifacts; this wrapper records only identities, counts, hashes, and
numeric request metrics.
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

import bench_cases  # noqa: E402
from runtime import model_session  # noqa: E402
from runtime import sampler_recipe_probe  # noqa: E402
from runtime import specialist_cases  # noqa: E402
from runtime import specialist_study_trial as study  # noqa: E402


RESULTS = ROOT / "results"
CONFIG = ROOT / "runtime/gemma4-style12-mtp-sharedkv-gpu-72k.json"
SMOKE_PLAN = ROOT / "runtime/gemma4-style12-mtp-smoke-72k-plan.json"
DESIGN = ROOT / "results/specialists-style-mtp-diagnostic-plan-v1.json"
PREPARATION = ROOT / "results/specialists-style-mtp-preparation-acceptance.json"
ARTIFACT_RECEIPT = ROOT / "results/gemma4-style12-mtp-artifact.json"
ASSISTANT = Path(
    "/external/storage/moe-bench/models/gemma4-12b-mtp/"
    "mtp-gemma-4-12B-it-Q8_0.gguf"
)

MODEL = "gemma-4-12b-styletune-q5_k_m"
PRIMARY_PROFILE = "gemma4-style12-q5_k_m"
PARENT_TAG = "specialists-final72k64k-gemma4-style12-q5_k_m-strict-72k"
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

PINNED_FILES = {
    DESIGN: "79c1cbc3bb4e2495704de9da5691eae8504c25acf2ab4c7a7864c01388c6a099",
    PREPARATION: "63ac473c03b3a19940cdbcb2a8564f8c060f903159427f5b0498e84fbaeedff6",
    ARTIFACT_RECEIPT: "8a0a7a1493f102133c2201846e74da77c153a9e0840b684544fb4c72f6a895eb",
    CONFIG: "09073bad22a504c24b51bca2b901cb73eb1e1823cc7f8bd2e38e97ec2cbbe436",
    SMOKE_PLAN: "3ca1886b675060e2c0f25b850a3f0db8bbdbdeadf11db971a44a058cced0d992",
    ROOT / "results/specialists-final72k64k-commands.json":
        "76a6ad2434828acec7ed5d33c249b9da12b69ba5b7cff51844e63f074702a907",
    ROOT / "runtime/run_specialist_first_wave.py":
        "63046360e0273a985b10c4dd2d33f09aea34ce3cf84deb54b628d6a26f7bf3fb",
    ROOT / "runtime/run_specialist_followups.py":
        "2d4c23390d93fe29bea051f7b7c6619534eddacb2432b67188ea43b896b65de8",
    ROOT / "runtime/gpu_window.py":
        "7982aa5906fd05f6cb61c9e2b54467724755472e7d17feb485a687a142ca6898",
    ROOT / "runtime/specialist_study_trial.py":
        "2187ee832d3785727f836d156d343865efcccb299547e877815e8ac72920e734",
    ROOT / "runtime/model_session.py":
        "968b2ce078975de23dd99670c2141995f460f36ae86490ce33153e7f1ae9c43c",
    ROOT / "runtime/sampler_recipe_probe.py":
        "8543bb44ba3242240aaa5646c51b2252c9b6367f3d624ec29b0ab21695bdfb18",
}
ASSISTANT_SIZE = 465_109_152
ASSISTANT_SHA256 = "16c90eb9f2b2891cc138f3d2b3bf11e23b2ced2aeab9a9d39d90fe446f2f0610"
PRIMARY_PLAN_SHA256 = "24ff6ec0986d44b82f351cf66abdb37030e2e9e52822bf854cefca251b13c97b"


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
    recipe: Path
    control: Path
    events: Path
    session_report: Path
    server_log: Path
    smoke: Path
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
    return ProbePaths(
        prefix=prefix,
        receipt=Path(f"{prefix}-style-mtp-probe-receipt.json"),
        recipe=Path(f"{prefix}-canonical-none-prose-recipe.json"),
        control=results / f"{tag}-control.json",
        events=results / f"{tag}-runtime-events.jsonl",
        session_report=results / f"{tag}-session-report.json",
        server_log=results / f"{tag}-server.log",
        smoke=results / f"{tag}-01-mtp_smoke_json_none_256.jsonl",
        sampler_requests=Path(f"{prefix}-sampler-requests.jsonl"),
        sampler_manifest=Path(f"{prefix}-sampler-manifest.json"),
        sampler_receipt=Path(f"{prefix}-sampler-receipt.json"),
        sampler_visible=Path(f"{prefix}-sampler-visible-texts"),
    )


def assert_unused(paths: ProbePaths) -> None:
    candidates = tuple(asdict(paths).values())[1:]
    collisions = [str(path) for path in candidates if os.path.lexists(path)]
    if collisions:
        raise ProbeError("refusing existing output/control path: " + ", ".join(collisions))


def verify_pins() -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    for path, expected in PINNED_FILES.items():
        _regular(path, "pinned input", MAX_JSON_BYTES if path.suffix == ".json" else None)
        actual = sha256_file(path)
        if actual != expected:
            raise ProbeError(f"pinned input hash mismatch: {path}")
        files.append({"path": str(path), "sha256": actual})

    details = _regular(ASSISTANT, "MTP assistant")
    if details.st_size != ASSISTANT_SIZE:
        raise ProbeError("MTP assistant size mismatch")
    artifact_sha = sha256_file(ASSISTANT)
    if artifact_sha != ASSISTANT_SHA256:
        raise ProbeError("MTP assistant SHA256 mismatch")

    artifact = _json(ARTIFACT_RECEIPT, "MTP artifact receipt")
    candidate = _json(CONFIG, "MTP candidate config")
    smoke = _json(SMOKE_PLAN, "MTP smoke plan")
    preparation = _json(PREPARATION, "MTP preparation acceptance")
    args = candidate.get("args")
    exact_tail = [
        "--spec-type", "draft-mtp", "--spec-draft-model", str(ASSISTANT),
        "--spec-draft-n-max", "3", "--spec-draft-ngl", "all",
    ]
    shards = candidate.get("shards")
    if (
        candidate.get("model_name") != MODEL
        or candidate.get("context") != CONTEXT
        or candidate.get("host") != "127.0.0.1"
        or candidate.get("port") != NATIVE_PORT
        or not isinstance(args, list)
        or args[-len(exact_tail):] != exact_tail
        or not isinstance(shards, list)
        or len(shards) != 1
        or shards[0].get("sha256")
        != "14d78cfccb5f052a9c4bfcdceac30fae570a09dea4453c7d929fa57f703ff350"
    ):
        raise ProbeError("MTP candidate config identity or controls changed")
    steps = smoke.get("steps")
    if not isinstance(steps, list) or len(steps) != 1:
        raise ProbeError("MTP smoke plan no longer has exactly one step")
    step = steps[0]
    if (
        step.get("id") != "mtp_smoke_json_none_256"
        or step.get("suite") != "quick"
        or step.get("cases") != ["quick_json_extract"]
        or step.get("thinking") != "none"
        or step.get("temperature") != 0
        or step.get("seed") != 0
        or step.get("output_tokens") != 256
        or step.get("request_timeout") != 180
        or step.get("repeat") != 1
    ):
        raise ProbeError("MTP smoke plan controls changed")
    assistant_receipt = artifact.get("artifact")
    if not isinstance(assistant_receipt, dict):
        assistant_receipt = artifact.get("gguf_artifact")
    prep_assistant = preparation.get("assistant_file")
    if not isinstance(prep_assistant, dict) or (
        prep_assistant.get("path") != str(ASSISTANT)
        or prep_assistant.get("size") != ASSISTANT_SIZE
        or prep_assistant.get("sha256_independently_reverified") != ASSISTANT_SHA256
    ):
        raise ProbeError("preparation acceptance does not pin the MTP assistant")
    return {
        "files": files,
        "assistant": {"path": str(ASSISTANT), "size": details.st_size, "sha256": artifact_sha},
        "candidate": {
            "model": candidate.get("model"),
            "model_name": candidate.get("model_name"),
            "context": candidate.get("context"),
            "target_declared_sha256": shards[0].get("sha256"),
            "spec_type": "draft-mtp",
            "spec_draft_n_max": 3,
            "spec_draft_ngl": "all",
        },
        "smoke": {"step_id": step.get("id"), "case_id": step["cases"][0]},
        "artifact_receipt_identity_present": isinstance(assistant_receipt, dict),
    }


def _primary_receipt_evidence(receipt: Mapping[str, Any]) -> tuple[list[str], dict[str, Any]]:
    failures: list[str] = []
    if receipt.get("tag") != PARENT_TAG:
        failures.append("primary receipt tag is not the predeclared Style main tag")
    if receipt.get("status") not in {"complete", "completed_with_evaluation_failures"}:
        failures.append("primary study is not complete")
    if receipt.get("runtime_backend") != "llama":
        failures.append("primary study did not use native llama")
    plan = receipt.get("plan") if isinstance(receipt.get("plan"), Mapping) else {}
    if plan.get("sha256") != PRIMARY_PLAN_SHA256:
        failures.append("primary study plan hash differs from the frozen all-task plan")
    session = receipt.get("session") if isinstance(receipt.get("session"), Mapping) else {}
    if session.get("model_name") != MODEL or session.get("context") != CONTEXT:
        failures.append("primary resident model/context identity differs")
    source_config = (
        receipt.get("source_config") if isinstance(receipt.get("source_config"), Mapping) else {}
    )
    if source_config.get("sha256") != (
        "3ffb305436bb7a19207064c64a9a32b9e864cbaf843a8633780d1005867b9e15"
    ):
        failures.append("primary target config hash differs from the frozen Style config")
    controls = receipt.get("controls") if isinstance(receipt.get("controls"), Mapping) else {}
    prose_controls = controls.get("prose") if isinstance(controls.get("prose"), Mapping) else {}
    if not (
        controls.get("profile") == PRIMARY_PROFILE
        and controls.get("prose_thinking") == "none"
        and prose_controls.get("output_cap") == OUTPUT_CAP
        and prose_controls.get("temperature") == 0.7
        and prose_controls.get("seed") == 0
        and prose_controls.get("timeout") == REQUEST_TIMEOUT
    ):
        failures.append("primary receipt lacks the canonical NONE prose controls")
    cleanup = receipt.get("cleanup") if isinstance(receipt.get("cleanup"), Mapping) else {}
    server_cleanup = (
        cleanup.get("model_session_cleanup")
        if isinstance(cleanup.get("model_session_cleanup"), Mapping) else {}
    )
    if not (
        cleanup.get("outer_exited") is True
        and cleanup.get("orchestrator_sigkill_sent") is False
        and server_cleanup.get("stopped") is True
        and server_cleanup.get("kill_sent") is False
    ):
        failures.append("primary owned-process cleanup evidence is incomplete")
    restoration = (
        receipt.get("restoration") if isinstance(receipt.get("restoration"), Mapping) else {}
    )
    if not (
        restoration.get("original_running") is True
        and restoration.get("original_paused") is False
        and restoration.get("owned_loopback_port_clear") is True
    ):
        failures.append("primary original-service/port restoration evidence is incomplete")
    if receipt.get("session_report_status") not in {"passed", "completed_with_step_failures"}:
        failures.append("primary model_session did not finish its frozen plan")

    commands = receipt.get("commands") if isinstance(receipt.get("commands"), list) else []
    by_kind = {
        row.get("kind"): row for row in commands
        if isinstance(row, Mapping) and isinstance(row.get("kind"), str)
    }
    for kind in ("specialist_code", "specialist_prose"):
        command = by_kind.get(kind)
        if not isinstance(command, Mapping) or command.get("returncode") != 0:
            failures.append(f"primary {kind} did not record a request-complete result set")

    sampler_plan = (
        receipt.get("sampler_plan") if isinstance(receipt.get("sampler_plan"), Mapping) else {}
    )
    stages = sampler_plan.get("stages") if isinstance(sampler_plan.get("stages"), list) else []
    prose_stages = [
        stage for stage in stages
        if isinstance(stage, Mapping) and stage.get("group") == "prose"
    ]
    sampler_attempts = 0
    if sampler_plan.get("matched") is not True or len(prose_stages) != 1:
        failures.append("primary receipt lacks its one matched Style prose sampler stage")
    else:
        stage_receipt = (
            prose_stages[0].get("receipt")
            if isinstance(prose_stages[0].get("receipt"), Mapping) else {}
        )
        sampler_attempts = _integer(stage_receipt.get("attempted_count")) or 0
        if not (
            stage_receipt.get("status") == "complete"
            and sampler_attempts == 3
            and stage_receipt.get("completed_count") == 3
            and stage_receipt.get("request_failure_count") == 0
            and stage_receipt.get("truncated_count") == 0
        ):
            failures.append("primary Style prose sampler stage is not three clean attempts")

    standard_attempts = len(bench_cases.get_cases("quick")) + len(bench_cases.get_cases("hard"))
    specialist_attempts = len(specialist_cases.get_cases(groups=("code",))) + len(
        specialist_cases.get_cases(groups=("prose",))
    )
    primary_attempts = standard_attempts + specialist_attempts + sampler_attempts
    if primary_attempts != 15:
        failures.append("primary receipt does not establish all 15 predeclared attempts")
    evidence = {
        "tag": receipt.get("tag"),
        "status": receipt.get("status"),
        "runtime_backend": receipt.get("runtime_backend"),
        "plan_sha256": plan.get("sha256"),
        "source_config_sha256": source_config.get("sha256"),
        "session_run_id": session.get("run_id"),
        "model": session.get("model_name"),
        "context": session.get("context"),
        "attempt_accounting": {
            "standard_frozen_cases": standard_attempts,
            "specialist_frozen_cases": specialist_attempts,
            "matched_sampler_attempts": sampler_attempts,
            "total": primary_attempts,
        },
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
    return failures, evidence


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


def _primary_prose_evidence(paths: Sequence[Path]) -> tuple[list[str], list[dict[str, Any]]]:
    failures: list[str] = []
    rows: list[tuple[Path, dict[str, Any]]] = []
    for path in paths:
        for row in _jsonl(path, "canonical primary prose JSONL"):
            rows.append((path, row))
    expected = _expected_prose()
    if len(rows) != len(expected):
        failures.append("canonical primary prose inputs must contain exactly three rows")
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
        metrics = (
            row.get("response_metrics")
            if isinstance(row.get("response_metrics"), Mapping) else {}
        )
        visible = row.get("visible_text") if isinstance(row.get("visible_text"), Mapping) else {}
        if "response_text" in response or isinstance(row.get("response_text"), str):
            failures.append(f"{case_id}: primary JSONL unexpectedly embeds generated prose")
        identity_ok = (
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
        if not identity_ok:
            failures.append(f"{case_id}: frozen prompt/case identity mismatch")
        controls_ok = (
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
        if not controls_ok:
            failures.append(f"{case_id}: canonical NONE prose controls mismatch")
        completion_tokens = _integer(
            metrics.get("output_tokens_reported", response.get("actual_completion_tokens"))
        )
        actual_completion_tokens = _integer(response.get("actual_completion_tokens"))
        wall_seconds = _finite(
            metrics.get(
                "wall_seconds",
                response.get("timing", {}).get("wall_seconds")
                if isinstance(response.get("timing"), Mapping) else None,
            )
        )
        visible_chars = _integer(visible.get("chars"))
        visible_bytes = _integer(visible.get("bytes"))
        visible_sha = visible.get("sha256")
        finish_reason = metrics.get("finish_reason", response.get("done_reason"))
        complete = (
            row.get("error") is None
            and row.get("objective_error") is None
            and response.get("stream_complete") is True
            and response.get("truncated") is False
            and finish_reason in {"stop", "eos", "end_turn"}
            and completion_tokens is not None
            and 0 < completion_tokens <= PRIMARY_TOKEN_CEILING
            and actual_completion_tokens is not None
            and 0 < actual_completion_tokens <= PRIMARY_TOKEN_CEILING
            and wall_seconds is not None
            and wall_seconds >= 0
            and visible_chars is not None
            and visible_chars > 0
            and visible_bytes is not None
            and visible_bytes > 0
            and isinstance(visible_sha, str)
            and len(visible_sha) == 64
            and visible.get("reasoning_excluded") is True
        )
        if not complete:
            failures.append(f"{case_id}: response is errored, incomplete, empty, truncated, or over limit")
        if wall_seconds is not None and wall_seconds >= 0:
            total_wall += wall_seconds
        evidence.append({
            "path": str(path.resolve()),
            "file_sha256": sha256_file(path),
            "case_id": case_id,
            "case_sha256": row.get("case_sha256"),
            "prompt_sha256": request.get("prompt_sha256"),
            "row_metadata_sha256": hashlib.sha256(
                json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                .encode("utf-8")
            ).hexdigest(),
            "visible_text": {
                "sha256": visible_sha,
                "chars": visible_chars,
                "bytes": visible_bytes,
                "reasoning_excluded": visible.get("reasoning_excluded"),
            },
            "completion_tokens": completion_tokens,
            "actual_completion_tokens": actual_completion_tokens,
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
        failures.append("canonical primary prose total request wall time exceeds 3600 seconds")
    for item in evidence:
        item["prose3_total_wall_seconds"] = round(total_wall, 6)
    return list(dict.fromkeys(failures)), evidence


def validate_eligibility(receipt_path: Path, prose_paths: Sequence[Path]) -> Eligibility:
    specialist_cases.assert_frozen_suite()
    receipt = _json(receipt_path, "primary study receipt")
    receipt_failures, receipt_evidence = _primary_receipt_evidence(receipt)
    prose_failures, prose_evidence = _primary_prose_evidence(prose_paths)
    failures = tuple(receipt_failures + prose_failures)
    return Eligibility(
        eligible=not failures,
        failures=failures,
        primary_receipt={
            "path": str(receipt_path.resolve()),
            "sha256": sha256_file(receipt_path),
            **receipt_evidence,
        },
        primary_prose=tuple(prose_evidence),
    )


def outer_command(tag: str, control: Path) -> list[str]:
    args = argparse.Namespace(
        runtime_backend="llama", config=CONFIG, plan=SMOKE_PLAN, tag=tag,
        startup_timeout=STARTUP_TIMEOUT, wait_for_idle=IDLE_WAIT_SECONDS,
        max_seconds=WINDOW_SECONDS,
    )
    return study.gpu_window_command(args, control)


def sampler_command(tag: str, prefix: Path, recipe: Path, run_id: str) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "runtime/sampler_recipe_probe.py"),
        "--recipe", str(recipe),
        "--runtime-events", str(RESULTS / f"{tag}-runtime-events.jsonl"),
        "--tag", tag,
        "--model-session-run-id", run_id,
        "--endpoint", ENDPOINT,
        "--model", MODEL,
        "--expected-context", str(CONTEXT),
        "--output-prefix", str(prefix),
        "--profile", PRIMARY_PROFILE,
        "--groups", "prose",
        "--prose-thinking", "none",
        "--output-cap", str(OUTPUT_CAP),
        "--timeout", str(int(REQUEST_TIMEOUT)),
    ]


def _recipe() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "label": "Style MTP canonical NONE prose temperature 0.7",
        "claim": "custom_experiment",
        "sources": [],
        "sampler_overrides": {"temperature": 0.7},
        "notes": "Exact canonical primary prose sampler control; unspecified fields inherit native defaults.",
    }


def assert_ports_clear() -> dict[str, bool]:
    result: dict[str, bool] = {}
    for label, port in (("native", NATIVE_PORT), ("strata", STRATA_PORT)):
        probe_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        probe_socket.settimeout(0.2)
        try:
            listening = probe_socket.connect_ex(("127.0.0.1", port)) == 0
        finally:
            probe_socket.close()
        clear = not listening
        if clear:
            bind_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                bind_socket.bind(("127.0.0.1", port))
            except OSError:
                clear = False
            finally:
                bind_socket.close()
        result[label] = clear
        if not clear:
            raise ProbeError(f"refusing occupied {label} loopback port {port}")
    return result


def wait_for_holding(
    outer: subprocess.Popen[Any], paths: ProbePaths, deadline: float,
) -> study.HoldingSession:
    args = argparse.Namespace(
        tag=paths.control.name.removesuffix("-control.json"),
        config=CONFIG,
        plan=SMOKE_PLAN,
        runtime_backend="llama",
    )
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
                rows,
                tag=args.tag,
                source_config=CONFIG,
                plan=SMOKE_PLAN,
                control=paths.control,
                runtime_backend="llama",
            )
        except study.StudyError as exc:
            raise ProbeError(str(exc)) from exc
        if identity is not None:
            if identity.model_name != MODEL or identity.context != CONTEXT:
                raise ProbeError("held session model/context identity mismatch")
            return identity
        time.sleep(0.5)
    raise ProbeError("whole-profile guard expired before the exact held session")


def validate_smoke(path: Path, run_id: str, *, events_path: Path | None = None) -> dict[str, Any]:
    if events_path is not None:
        events = study._read_event_rows(events_path)
        completions = [row for row in events if row.get("event") == "step_complete"]
        if (
            any(row.get("run_id") != run_id for row in events)
            or len(completions) != 1
            or completions[0].get("id") != "mtp_smoke_json_none_256"
            or completions[0].get("result") != str(path)
            or completions[0].get("returncode") != 0
            or completions[0].get("passed") is not True
        ):
            raise ProbeError("MTP smoke is not tied to the exact held model_session lifecycle")
    rows = _jsonl(path, "MTP smoke result")
    if len(rows) != 1:
        raise ProbeError("MTP smoke did not produce exactly one result")
    row = rows[0]
    request = row.get("request") if isinstance(row.get("request"), Mapping) else {}
    response = row.get("response") if isinstance(row.get("response"), Mapping) else {}
    evaluation = row.get("evaluation") if isinstance(row.get("evaluation"), Mapping) else {}
    if (
        not isinstance(row.get("run_id"), str)
        or not row.get("run_id")
        or row.get("suite") != "quick"
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
        raise ProbeError("MTP smoke failed identity, correctness, completion, or truncation gate")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "case_id": row.get("case_id"),
        "benchmark_run_id": row.get("run_id"),
        "model_session_run_id": run_id,
        "passed": True,
        "stream_complete": True,
        "truncated": False,
        "request_error": False,
    }


def _terminate_owned(process: subprocess.Popen[Any]) -> dict[str, Any]:
    result = {
        "pid": process.pid,
        "pgid": process.pid,
        "term_sent": False,
        "kill_sent": False,
        "stopped": process.poll() is not None,
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
                raise ProbeError("whole-profile guard expired while the owned prose helper was running")
            time.sleep(0.5)
    except BaseException as exc:
        if process.poll() is None:
            cleanup = _terminate_owned(process)
        else:
            cleanup = {
                "pid": process.pid, "pgid": process.pid, "term_sent": False,
                "kill_sent": False, "stopped": True,
            }
        setattr(exc, "owned_process", process)
        setattr(exc, "owned_cleanup", cleanup)
        raise
    return process, int(process.returncode)


def summarize_acceptance(path: Path, *, tag: str) -> dict[str, Any]:
    rows = _jsonl(path, "MTP prose request metadata")
    if len(rows) != 3:
        raise ProbeError("MTP prose request metadata must contain exactly three rows")
    cases: list[dict[str, Any]] = []
    total_proposed = 0
    total_accepted = 0
    covered = 0
    expected = _expected_prose()
    seen: set[str] = set()
    for row in rows:
        case_id = row.get("case_id")
        request = row.get("request") if isinstance(row.get("request"), Mapping) else {}
        response = row.get("response") if isinstance(row.get("response"), Mapping) else {}
        if "response_text" in response or isinstance(row.get("response_text"), str):
            raise ProbeError("MTP prose request metadata unexpectedly embeds generated prose")
        native = (
            response.get("openai_native_timings")
            if isinstance(response.get("openai_native_timings"), Mapping) else {}
        )
        proposed = _integer(native.get("draft_n"))
        accepted = _integer(native.get("draft_n_accepted"))
        if (proposed is None) != (accepted is None):
            raise ProbeError("MTP draft acceptance telemetry is only partially reported")
        if proposed is not None:
            if proposed < 0 or accepted is None or accepted < 0 or accepted > proposed:
                raise ProbeError("MTP draft acceptance telemetry is invalid")
            covered += 1
            total_proposed += proposed
            total_accepted += accepted
        if (
            not isinstance(case_id, str)
            or case_id not in expected
            or case_id in seen
            or row.get("tag") != tag
            or row.get("profile") != PRIMARY_PROFILE
            or row.get("model") != MODEL
            or row.get("case_group") != "prose"
            or row.get("case_sha256") != expected.get(case_id, {}).get("case_sha256")
            or request.get("prompt_sha256") != expected.get(case_id, {}).get("prompt_sha256")
            or request.get("model") != MODEL
            or request.get("profile") != PRIMARY_PROFILE
            or request.get("thinking") != "none"
            or request.get("chat_template_thinking") != "off"
            or request.get("temperature") != 0.7
            or request.get("seed") != 0
            or request.get("context_max_tokens") != CONTEXT
            or request.get("output_token_limit") != OUTPUT_CAP
            or request.get("timeout_seconds") != REQUEST_TIMEOUT
        ):
            raise ProbeError("MTP prose request metadata identity or controls mismatch")
        seen.add(case_id)
        cases.append({
            "case_id": case_id,
            "draft_proposed": proposed,
            "draft_accepted": accepted,
            "acceptance_rate": (
                accepted / proposed
                if proposed is not None and proposed > 0 and accepted is not None else None
            ),
        })
    if seen != set(expected):
        raise ProbeError("MTP prose request metadata does not cover each frozen prose case")
    return {
        "request_metadata_path": str(path),
        "request_metadata_sha256": sha256_file(path),
        "case_count": len(rows),
        "reported_case_count": covered,
        "missing_case_count": len(rows) - covered,
        "draft_proposed_total": total_proposed if covered else None,
        "draft_accepted_total": total_accepted if covered else None,
        "acceptance_rate": (
            total_accepted / total_proposed if covered and total_proposed > 0 else None
        ),
        "cases": cases,
        "claim_scope": "native telemetry when present; missing telemetry remains unknown",
    }


def summarize_sampler_receipt(
    path: Path,
    run_id: str,
    returncode: int,
    *,
    expected_tag: str,
    recipe_path: Path,
    requests_path: Path | None = None,
) -> dict[str, Any]:
    receipt = _json(path, "MTP sampler receipt")
    controls = receipt.get("controls") if isinstance(receipt.get("controls"), Mapping) else {}
    recipe = receipt.get("recipe") if isinstance(receipt.get("recipe"), Mapping) else {}
    if (
        receipt.get("tag") != expected_tag
        or receipt.get("model") != MODEL
        or receipt.get("model_session_run_id") != run_id
        or receipt.get("runtime_backend") != "llama"
        or controls.get("prose_thinking") != "none"
        or controls.get("output_token_limit") != OUTPUT_CAP
        or controls.get("timeout_seconds") != REQUEST_TIMEOUT
        or controls.get("seed") != 0
        or receipt.get("attempted_count") != 3
        or receipt.get("completed_count") != 3
        or receipt.get("truncated_count") != 0
        or receipt.get("request_failure_count") != 0
        or receipt.get("status") != "complete"
        or returncode != 0
        or recipe.get("path") != str(recipe_path)
        or recipe.get("sha256") != sha256_file(recipe_path)
        or recipe.get("sampler_overrides") != {"temperature": 0.7}
    ):
        raise ProbeError("MTP prose sampler receipt is incomplete or has mismatched controls")
    attempts = receipt.get("attempts") if isinstance(receipt.get("attempts"), list) else []
    summary = {
        "path": str(path),
        "sha256": sha256_file(path),
        "status": receipt.get("status"),
        "attempted_count": receipt.get("attempted_count"),
        "completed_count": receipt.get("completed_count"),
        "truncated_count": receipt.get("truncated_count"),
        "request_failure_count": receipt.get("request_failure_count"),
        "attempts": [
            {
                key: row.get(key)
                for key in (
                    "call_id", "case_id", "outcome", "stream_complete", "truncated",
                    "input_tokens", "output_tokens", "wall_seconds", "native_decode_seconds",
                    "native_decode_tps", "error_type",
                )
            }
            for row in attempts if isinstance(row, Mapping)
        ],
        "controls": {
            key: controls.get(key)
            for key in (
                "prose_thinking", "context_max_tokens", "expected_context_tokens",
                "actual_context_tokens", "output_token_limit", "timeout_seconds", "seed",
            )
        },
    }
    if requests_path is not None:
        summary["speculative_acceptance"] = summarize_acceptance(
            requests_path, tag=str(receipt.get("tag")),
        )
    return summary


def _observed_prose_rows(path: Path) -> dict[str, Any]:
    """Summarize complete metadata rows from a possibly interrupted JSONL."""
    evidence: dict[str, Any] = {
        "path": str(path),
        "exists": False,
        "sha256": None,
        "observed_row_count": 0,
        "observed_completed_row_count": 0,
        "observed_request_error_row_count": 0,
        "rows": [],
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
        evidence["sha256"] = hashlib.sha256(raw).hexdigest()
    except FileNotFoundError:
        return evidence
    except OSError:
        evidence["observation_error"] = "file_unreadable"
        return evidence

    parsed: list[dict[str, Any]] = []
    invalid_complete_rows = 0
    for raw_line in raw.splitlines(keepends=True):
        if not raw_line.endswith((b"\n", b"\r")):
            evidence["ignored_incomplete_trailing_bytes"] = len(raw_line)
            continue
        try:
            row = json.loads(raw_line)
        except (UnicodeError, json.JSONDecodeError):
            invalid_complete_rows += 1
            continue
        if not isinstance(row, dict):
            invalid_complete_rows += 1
            continue
        response = row.get("response") if isinstance(row.get("response"), Mapping) else {}
        if "response_text" in response or isinstance(row.get("response_text"), str):
            invalid_complete_rows += 1
            continue
        metrics = (
            row.get("response_metrics")
            if isinstance(row.get("response_metrics"), Mapping) else {}
        )
        native = (
            response.get("openai_native_timings")
            if isinstance(response.get("openai_native_timings"), Mapping) else {}
        )
        error_present = row.get("error") is not None
        complete = (
            response.get("stream_complete") is True
            and response.get("truncated") is not True
            and not error_present
        )
        parsed.append({
            "case_id": row.get("case_id"),
            "call_id": row.get("call_id"),
            "stream_complete": response.get("stream_complete"),
            "truncated": response.get("truncated"),
            "request_error_present": error_present,
            "output_tokens": metrics.get(
                "output_tokens_reported", response.get("actual_completion_tokens")
            ),
            "wall_seconds": metrics.get("wall_seconds"),
            "draft_proposed": native.get("draft_n"),
            "draft_accepted": native.get("draft_n_accepted"),
            "observed_complete": complete,
        })
    evidence["observed_row_count"] = len(parsed)
    evidence["observed_completed_row_count"] = sum(
        row["observed_complete"] is True for row in parsed
    )
    evidence["observed_request_error_row_count"] = sum(
        row["request_error_present"] is True for row in parsed
    )
    evidence["invalid_or_content_bearing_complete_row_count"] = invalid_complete_rows
    evidence["rows"] = parsed
    return evidence


def _unvalidated_sampler_receipt(path: Path) -> dict[str, Any]:
    evidence: dict[str, Any] = {"path": str(path), "exists": False}
    try:
        value = _json(path, "unvalidated MTP sampler receipt")
    except ProbeError:
        if os.path.lexists(path):
            evidence.update({"exists": True, "observation_error": "not_valid_bounded_receipt"})
        return evidence
    try:
        digest = sha256_file(path)
    except OSError:
        evidence.update({"exists": True, "observation_error": "receipt_changed_or_unreadable"})
        return evidence
    evidence.update({
        "exists": True,
        "sha256": digest,
        "reported_status": value.get("status"),
        "reported_counts": {
            key: value.get(key) if _integer(value.get(key)) is not None else None
            for key in (
                "attempted_count", "completed_count", "truncated_count",
                "request_failure_count",
            )
        },
        "terminal_receipt_validated": False,
    })
    return evidence


def record_unvalidated_prose(
    receipt: dict[str, Any],
    paths: ProbePaths,
    process: subprocess.Popen[Any],
    *,
    status: str,
    stop_cause: str,
) -> None:
    if status not in {"interrupted", "incomplete"}:
        raise ProbeError("unvalidated prose status must be interrupted or incomplete")
    receipt["execution"]["prose_helper_pid"] = process.pid
    receipt["execution"]["prose_helper_returncode"] = process.poll()
    receipt["prose"] = {
        "status": status,
        "helper_launched": True,
        "terminal_sampler_receipt_validated": False,
        "attempted_count": None,
        "completed_count": None,
        "truncated_count": None,
        "request_failure_count": None,
        "stop_cause": stop_cause,
        "observed_request_metadata": _observed_prose_rows(paths.sampler_requests),
        "unvalidated_sampler_receipt": _unvalidated_sampler_receipt(paths.sampler_receipt),
        "note": (
            "Official request counts remain unknown because no clean terminal sampler receipt "
            "was validated; observed rows are metadata evidence only."
        ),
    }


def _server_identity(events_path: Path, run_id: str) -> dict[str, Any]:
    rows = study._read_event_rows(events_path)
    starts = [row for row in rows if row.get("event") == "session_started"]
    servers = [row for row in rows if row.get("event") == "server_started"]
    if len(starts) != 1 or len(servers) != 1 or any(row.get("run_id") != run_id for row in rows):
        raise ProbeError("runtime events do not preserve one exact owned session/server identity")
    return {
        "run_id": run_id,
        "model_session_pid": next(
            (row.get("pid") for row in rows if row.get("event") == "session_process_identity"),
            None,
        ),
        "server_pid": servers[0].get("pid"),
        "server_pgid": servers[0].get("pgid"),
        "server_argv": servers[0].get("argv"),
        "source_config": starts[0].get("source_config"),
        "source_config_sha256": starts[0].get("source_config_sha256"),
        "plan": starts[0].get("plan"),
        "plan_sha256": starts[0].get("plan_sha256"),
        "runtime_backend": starts[0].get("runtime_backend"),
        "model": starts[0].get("model_name"),
        "context": starts[0].get("context"),
        "events_path": str(events_path),
        "events_sha256": sha256_file(events_path),
    }


def session_cleanup_evidence(
    paths: ProbePaths,
    *,
    run_id: str,
    outer_returncode: int | None,
    restoration: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], list[str]]:
    failures: list[str] = []
    report = _json(paths.session_report, "MTP model_session report")
    server = report.get("server") if isinstance(report.get("server"), Mapping) else {}
    cleanup = server.get("cleanup") if isinstance(server.get("cleanup"), Mapping) else {}
    steps = report.get("steps") if isinstance(report.get("steps"), list) else []
    if (
        report.get("session_id") != run_id
        or report.get("tag") != paths.control.name.removesuffix("-control.json")
        or report.get("runtime_backend") != "llama"
        or report.get("status") != "passed"
        or report.get("source_config") != str(CONFIG)
        or report.get("source_config_sha256") != PINNED_FILES[CONFIG]
        or not isinstance(report.get("plan"), Mapping)
        or report["plan"].get("sha256") != PINNED_FILES[SMOKE_PLAN]
        or server.get("model_name") != MODEL
        or server.get("context") != CONTEXT
        or len(steps) != 1
        or not isinstance(steps[0], Mapping)
        or steps[0].get("id") != "mtp_smoke_json_none_256"
        or steps[0].get("passed") is not True
    ):
        failures.append("model_session report identity or smoke completion is invalid")
    if cleanup.get("stopped") is not True or cleanup.get("kill_sent") is not False:
        failures.append("owned native server did not record clean non-SIGKILL shutdown")
    if outer_returncode != 0:
        failures.append("GPU window did not exit successfully")
    if not isinstance(restoration, Mapping) or not (
        restoration.get("original_running") is True
        and restoration.get("original_paused") is False
        and restoration.get("owned_loopback_port_clear") is True
    ):
        failures.append("original service or native port restoration is unverified")
    events = study._read_event_rows(paths.events)
    if not (
        any(row.get("event") == "server_stopped" for row in events)
        and any(row.get("event") == "session_complete" for row in events)
        and all(row.get("run_id") == run_id for row in events)
    ):
        failures.append("runtime event ledger lacks exact terminal cleanup events")
    return {
        "session_report": {
            "path": str(paths.session_report),
            "sha256": sha256_file(paths.session_report),
            "status": report.get("status"),
            "session_id": report.get("session_id"),
            "source_config_sha256": report.get("source_config_sha256"),
            "plan_sha256": report.get("plan", {}).get("sha256")
            if isinstance(report.get("plan"), Mapping) else None,
        },
        "native_server": {
            "stopped": cleanup.get("stopped"),
            "term_sent": cleanup.get("term_sent"),
            "kill_sent": cleanup.get("kill_sent"),
            "pgid": cleanup.get("pgid"),
        },
        "runtime_events_sha256": sha256_file(paths.events),
    }, failures


def execute_probe(paths: ProbePaths, tag: str, receipt: dict[str, Any]) -> int:
    if os.path.lexists(paths.control):
        raise ProbeError("control path must not exist before model_session starts")
    ports = assert_ports_clear()
    receipt["execution"]["port_preflight"] = ports
    recipe = _recipe()
    sampler_recipe_probe.validate_recipe(recipe)
    _exclusive_json(paths.recipe, recipe)
    outer_argv = outer_command(tag, paths.control)
    receipt["commands"]["gpu_window"] = outer_argv
    started = time.monotonic()
    deadline = started + WINDOW_SECONDS - RESTORATION_RESERVE_SECONDS
    outer: subprocess.Popen[Any] | None = None
    prose: subprocess.Popen[Any] | None = None
    session: study.HoldingSession | None = None
    stop_cause = "setup_failure"
    problem: dict[str, Any] | None = None
    prose_unvalidated_status: str | None = None
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
        receipt["execution"]["gpu_window_pid"] = outer.pid
        session = wait_for_holding(outer, paths, deadline)
        receipt["execution"]["session"] = {
            "run_id": session.run_id,
            "model": session.model_name,
            "context": session.context,
            "initial_load_seconds": session.initial_load_seconds,
            "session_started_at": session.session_started_at,
            "server_ready_at": session.server_ready_at,
            "holding_started_at": session.holding_started_at,
        }
        receipt["execution"]["runtime_identity"] = _server_identity(paths.events, session.run_id)
        receipt["smoke"] = validate_smoke(
            paths.smoke, session.run_id, events_path=paths.events,
        )
        command = sampler_command(tag, paths.prefix, paths.recipe, session.run_id)
        receipt["commands"]["sampler_recipe_probe"] = command
        prose, returncode = run_owned(command, deadline)
        receipt["execution"]["prose_helper_pid"] = prose.pid
        receipt["execution"]["prose_helper_returncode"] = returncode
        receipt["prose"] = summarize_sampler_receipt(
            paths.sampler_receipt,
            session.run_id,
            returncode,
            expected_tag=tag,
            recipe_path=paths.recipe,
            requests_path=paths.sampler_requests,
        )
        receipt["limitations"]["speculative_acceptance_measured"] = (
            receipt["prose"]["speculative_acceptance"]["reported_case_count"] > 0
        )
        stop_cause = "probe_complete"
        code = 0
    except ProbeInterrupted as exc:
        prose = getattr(exc, "owned_process", prose)
        if prose is not None:
            prose_unvalidated_status = "interrupted"
        if hasattr(exc, "owned_cleanup"):
            receipt["cleanup"]["prose_helper"] = getattr(exc, "owned_cleanup")
        problem = {
            "error_type": type(exc).__name__, "error": str(exc), "signal": exc.signum,
        }
        stop_cause = "interrupted"
        code = 128 + exc.signum
    except (ProbeError, study.StudyError, OSError, subprocess.SubprocessError) as exc:
        prose = getattr(exc, "owned_process", prose)
        if prose is not None:
            prose_unvalidated_status = (
                "interrupted"
                if "guard expired" in str(exc) or hasattr(exc, "owned_process")
                else "incomplete"
            )
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
        helper_stopped = (
            not isinstance(helper_cleanup, Mapping) or helper_cleanup.get("stopped") is True
        )
        receipt["cleanup"]["prose_helper_stopped_before_control"] = helper_stopped
        if (
            prose is not None
            and prose_unvalidated_status is not None
            and receipt.get("prose", {}).get("status") != "complete"
        ):
            record_unvalidated_prose(
                receipt,
                paths,
                prose,
                status=prose_unvalidated_status,
                stop_cause=stop_cause,
            )
        if outer is not None:
            if not helper_stopped:
                problem = problem or {
                    "error_type": "ProbeError",
                    "error": "owned prose helper did not stop; control stop was not published",
                }
            elif session is not None and os.path.lexists(paths.control):
                try:
                    receipt["cleanup"]["control_stop_requested"] = study.request_session_stop(
                        paths.control
                    )
                except (study.StudyError, OSError) as exc:
                    problem = problem or {"error_type": type(exc).__name__, "error": str(exc)}
            elif outer.poll() is None:
                try:
                    receipt["cleanup"]["owned_outer_sigterm_sent"] = (
                        study.request_outer_termination(outer)
                    )
                except OSError as exc:
                    problem = problem or {"error_type": type(exc).__name__, "error": str(exc)}
            outer_returncode, delayed_stop = study.wait_for_outer_exit(
                outer,
                timeout=study.RESTORATION_WAIT_SECONDS,
                control=paths.control if helper_stopped else None,
            )
            receipt["cleanup"]["control_stop_requested"] |= delayed_stop
            receipt["cleanup"]["outer_returncode"] = outer_returncode
            receipt["cleanup"]["outer_exited"] = outer_returncode is not None
            if outer_returncode is not None:
                try:
                    receipt["restoration"] = study.restoration_evidence("llama")
                except (OSError, RuntimeError) as exc:
                    problem = problem or {"error_type": type(exc).__name__, "error": str(exc)}
            else:
                problem = problem or {
                    "error_type": "ProbeError",
                    "error": "owned GPU window did not exit during restoration wait; no SIGKILL sent",
                }
            if session is not None and outer_returncode is not None:
                try:
                    final_identity = _server_identity(paths.events, session.run_id)
                    cleanup_evidence, cleanup_failures = session_cleanup_evidence(
                        paths,
                        run_id=session.run_id,
                        outer_returncode=outer_returncode,
                        restoration=receipt.get("restoration"),
                    )
                    receipt["execution"]["runtime_identity"] = final_identity
                    receipt["cleanup"].update(cleanup_evidence)
                    if cleanup_failures:
                        code = 1
                        problem = problem or {
                            "error_type": "ProbeError",
                            "error": "; ".join(cleanup_failures),
                        }
                except (ProbeError, study.StudyError, OSError) as exc:
                    code = 1
                    problem = problem or {"error_type": type(exc).__name__, "error": str(exc)}
        receipt["cleanup"]["stop_cause"] = stop_cause
        receipt["cleanup"]["orchestrator_sigkill_sent"] = False
        receipt["problem"] = problem
        receipt["execution"]["wall_seconds"] = round(time.monotonic() - started, 3)
        if problem is not None:
            if problem.get("error_type") != "ProbeInterrupted":
                code = 1
        receipt["status"] = (
            "complete" if code == 0 and problem is None
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
        "diagnostic_kind": "style_gemma4_mtp_prose_probe",
        "created_at": utc_now(),
        "mode": "execute" if args.execute else "inspect",
        "status": "eligible_not_launched" if eligibility.eligible else "ineligible_not_launched",
        "tag": args.tag,
        "output_prefix": str(paths.prefix),
        "wrapper_source": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "pinned_inputs": pins,
        "eligibility": {
            "eligible": eligibility.eligible,
            "failures": list(eligibility.failures),
            "selection_basis": "completion, request controls, size, and time metadata only",
            "subjective_writing_preselection": False,
            "primary_study": dict(eligibility.primary_receipt),
            "primary_prose": [dict(row) for row in eligibility.primary_prose],
        },
        "controls": {
            "runtime_backend": "llama",
            "profile": PRIMARY_PROFILE,
            "context": CONTEXT,
            "thinking": "none",
            "temperature": 0.7,
            "seed": 0,
            "output_cap": OUTPUT_CAP,
            "per_request_timeout_seconds": REQUEST_TIMEOUT,
            "whole_profile_guard_seconds": WINDOW_SECONDS,
            "idle_wait_seconds": IDLE_WAIT_SECONDS,
            "startup_timeout_seconds": STARTUP_TIMEOUT,
            "groups": ["prose"],
            "case_count": 3,
        },
        "commands": {
            "gpu_window": outer_command(args.tag, paths.control),
            "sampler_recipe_probe_template": sampler_command(
                args.tag, paths.prefix, paths.recipe, "<exact-held-run-id>"
            ),
        },
        "execution": {"launched": False},
        "smoke": {"status": "not_run"},
        "prose": {
            "status": "not_run",
            "helper_launched": False,
            "attempted_count": 0,
            "completed_count": 0,
            "truncated_count": 0,
            "request_failure_count": 0,
        },
        "cleanup": {
            "control_path_preexisting": os.path.lexists(paths.control),
            "control_stop_requested": False,
            "owned_outer_sigterm_sent": False,
            "outer_exited": None,
            "orchestrator_sigkill_sent": False,
        },
        "restoration": None,
        "problem": None,
        "limitations": {
            "gpu_inference_validated": False,
            "vram_fit_validated": False,
            "speculative_acceptance_measured": False,
            "speedup_measured": False,
            "quality_comparison_complete": False,
            "publisher_or_source_support_is_not_local_performance_evidence": True,
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
        eligibility = validate_eligibility(args.primary_study_receipt, args.primary_prose)
    except (ProbeError, RuntimeError, OSError) as exc:
        eligibility = Eligibility(
            eligible=False,
            failures=(f"primary metadata validation failed: {exc}",),
            primary_receipt={"path": str(args.primary_study_receipt)},
            primary_prose=(),
        )
    if pin_failure is not None:
        eligibility = Eligibility(
            eligible=False,
            failures=eligibility.failures + (f"pinned input verification failed: {pin_failure}",),
            primary_receipt=eligibility.primary_receipt,
            primary_prose=eligibility.primary_prose,
        )
    receipt = build_receipt(args=args, paths=paths, pins=pins, eligibility=eligibility)
    if args.execute and eligibility.eligible:
        receipt["execution"]["launched"] = True
        code = execute_probe(paths, args.tag, receipt)
    elif args.execute:
        receipt["problem"] = {
            "error_type": "EligibilityError",
            "error": "execution refused because predeclared metadata eligibility failed",
        }
        code = 1
    else:
        code = 0 if eligibility.eligible else 1
    receipt["finished_at"] = utc_now()
    _exclusive_json(paths.receipt, receipt)
    return receipt, code


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-study-receipt", required=True, type=Path)
    parser.add_argument(
        "--primary-prose", required=True, action="append", type=Path,
        help="canonical primary NONE prose JSONL; repeat when split across files",
    )
    parser.add_argument("--tag", required=True)
    parser.add_argument("--output-prefix", required=True, type=Path)
    parser.add_argument(
        "--execute", action="store_true",
        help="open the bounded GPU window; absent by default for metadata-only inspection",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        receipt, code = run(args)
    except (ProbeError, RuntimeError, OSError, subprocess.SubprocessError) as exc:
        print(f"style MTP probe: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({
        "status": receipt["status"],
        "mode": receipt["mode"],
        "eligible": receipt["eligibility"]["eligible"],
        "receipt": f"{args.output_prefix}-style-mtp-probe-receipt.json",
    }, sort_keys=True))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
