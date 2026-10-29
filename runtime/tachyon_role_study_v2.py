#!/usr/bin/env python3
"""V2 serial custom role study for fast routing, decisions, and summaries.

This is a small custom test, not a public benchmark. It waits for the GPU
trial's durable handoff, runs one CPU model server at a time, preserves first
terminal answers, and restores the LFM2.6 CPU4 service when finished. V2 uses
fresh result bindings after correcting the LFM2.6 thinking-protocol controls.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import shlex
import signal
import statistics
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
CPU_STUDY = ROOT / "results/tachyon-study-20261023-v1"
GPU_TRIAL = CPU_STUDY / "gpu-trial"
DEFAULT_STUDY = ROOT / "results/tachyon-role-study-20261023-v2"
DEFAULT_PLAN = DEFAULT_STUDY / "models-plan.json"
GPU_CONTROLLER = ROOT / "runtime/tachyon_gpu_trial.py"
DERIVED_SOURCE = ROOT / "runtime/tachyon_role_study.py"
V1_STUDY = ROOT / "results/tachyon-role-study-20261023-v1"
V1_DEFAULT_RESTORE = V1_STUDY / "default-restore-server-launch.json"
V1_CONTROLLER_LAUNCH = V1_STUDY / "controller-launch.json"
REMOTE_ROOT = "/external/device-workspace"
REMOTE_RUNTIME = REMOTE_ROOT + "/runtime/llama-cpp-cpu"
REMOTE_RESULTS = REMOTE_ROOT + "/results/tachyon-role-study-20261023-v2"
ENDPOINT = "http://127.0.0.1:18185/v1"
STUDY_KIND = "tachyon_custom_small_role_study_v2"
MANIFEST_SHA256 = "e86a95d7cb20135c9e476673439fc3ad1b4363a5cb9120fe8839d2f16e41e81e"
PLAN_SHA256 = "74d011d6fab0b2984adecec47fec4ee600af2aee95ce1e409c43a01a800c9209"
GUIDED_PLAN_SHA256 = "e77785815a718483cf3d7520ebead83225570769a9ac3743f011066f7b6dcd50"
DERIVED_SOURCE_SHA256 = "c3db80ab9c1bf015a9790ff5fe765047c648b8354c71d1a26801f766b12c4127"
HANDOFF_SHA256 = "c57a456fafb20dd2305ccef6a4fb2e145fe222d42955a6dc60ab50a58076c61c"
MODEL_ORDER = ("lfm25-26b", "bonsai2-2b", "bonsai2-4b",
               "lfm25-12b-instruct", "qwen35-08b")
PARAMETER_LABELS = {
    "lfm25-26b": "2.6B", "bonsai2-2b": "2B", "bonsai2-4b": "4B",
    "lfm25-12b-instruct": "1.2B", "qwen35-08b": "0.8B",
}
SUITES = ("routing", "decision", "summary")
GUIDED_CASE_IDS = (
    "route-summary", "route-injected_policy", "decision-cache-boundary",
    "decision-conflicting-records", "summary-unknown-eta",
    "summary-planned-not-done",
)
SSH_OPTIONS = (
    "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes",
    "-o", "ClearAllForwardings=yes", "-o", "ForwardAgent=no",
    "-o", "RequestTTY=no",
)


class RoleStudyError(RuntimeError):
    pass


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")


def json_sha(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise RoleStudyError(f"cannot hash required file: {path}") from exc
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RoleStudyError(f"invalid or unreadable JSON: {path}") from exc


def atomic_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("xb") as stream:
        stream.write(data); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_bytes(path, json.dumps(value, indent=2, sort_keys=True,
                                  ensure_ascii=False).encode("utf-8") + b"\n")


def append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")
        stream.flush(); os.fsync(stream.fileno())


def process_identity(pid: int) -> dict[str, Any]:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        tail = stat[stat.rfind(")") + 2:].split()
        argv = [item.decode("utf-8", "surrogateescape")
                for item in Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0") if item]
        return {"pid": pid, "alive": tail[0] != "Z", "state": tail[0],
                "start_ticks": int(tail[19]), "argv": argv}
    except (FileNotFoundError, ProcessLookupError):
        return {"pid": pid, "alive": False, "state": "exited", "argv": []}
    except (OSError, ValueError, IndexError) as exc:
        raise RoleStudyError(f"cannot inspect local process {pid}: {exc}") from exc


def lock_is_free(path: Path) -> bool:
    if not path.is_file():
        return False
    with path.open("r", encoding="utf-8") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
    return True


def verify_owned(receipt: Mapping[str, Any], identity: Mapping[str, Any], label: str) -> bool:
    if not identity.get("alive"):
        return False
    for field in ("pid", "start_ticks", "argv"):
        if receipt.get(field) != identity.get(field):
            raise RoleStudyError(f"{label} {field} differs from ownership receipt")
    return True


def exact_json_equal(left: Any, right: Any) -> bool:
    return canonical(left) == canonical(right)


def word_count(value: str) -> int:
    return len(value.split())


def strict_json_loads(value: str) -> Any:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in items:
            if key in result:
                raise ValueError(f"duplicate object key: {key}")
            result[key] = item
        return result

    return json.loads(value, object_pairs_hook=pairs,
                      parse_constant=lambda token: (_ for _ in ()).throw(
                          ValueError(f"non-finite JSON number: {token}")))


def evaluate_answer(case: Mapping[str, Any], final_text: str) -> dict[str, Any]:
    """Check strict syntax, exact top keys, typed field values, and summary size."""
    try:
        observed = strict_json_loads(final_text)
        json_valid, syntax_error = True, None
    except (json.JSONDecodeError, ValueError) as exc:
        observed, json_valid, syntax_error = None, False, str(exc)
    expected, required = case.get("expected"), case.get("required_keys", [])
    missing: list[str] = []; unexpected: list[str] = []; mismatches: list[str] = []
    if json_valid and isinstance(observed, dict):
        missing = [key for key in required if key not in observed]
        unexpected = [key for key in observed if key not in required]
        if isinstance(expected, dict):
            mismatches = [key for key, value in expected.items()
                          if key not in observed or not exact_json_equal(observed[key], value)]
    else:
        missing = list(required)
    keys_match = not missing and not unexpected
    values_match = bool(json_valid and isinstance(observed, dict) and not mismatches)
    limit = case.get("max_summary_words")
    summary = observed.get("summary") if isinstance(observed, dict) else None
    summary_words = word_count(summary) if isinstance(summary, str) else None
    summary_ok = True
    if limit is not None:
        summary_ok = bool(isinstance(summary, str) and summary.strip()
                          and summary_words is not None and summary_words <= limit)
    semantic_pass = bool(json_valid and keys_match and values_match and summary_ok)
    return {
        "json_valid": json_valid, "syntax_error": syntax_error,
        "observed_json": observed, "missing_keys": missing,
        "unexpected_keys": unexpected, "mismatched_fields": mismatches,
        "exact_top_level_keys": keys_match, "expected_values_match": values_match,
        "summary_word_count": summary_words, "summary_word_limit": limit,
        "summary_word_limit_pass": summary_ok, "semantic_pass": semantic_pass,
        "manual_review_required": case.get("suite") == "summary",
        "claim_scope": ("closed facts and length only; free-prose truth requires manual review"
                        if case.get("suite") == "summary" else "closed JSON rules"),
    }


def build_request(case: Mapping[str, Any], model: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "model": model["alias"], "messages": copy.deepcopy(case["messages"]),
        "max_tokens": 512, "seed": 20_261_023, **copy.deepcopy(model["recipe"]),
        "stream": True, "stream_options": {"include_usage": True},
        "timings_per_token": True,
    }


def build_guided_request(case: Mapping[str, Any], model: Mapping[str, Any],
                         guided_plan: Mapping[str, Any]) -> dict[str, Any]:
    payload = build_request(case, model)
    overrides = guided_plan["request_overrides_by_case"][case["id"]]
    payload.update(copy.deepcopy(overrides))
    return payload


def _validate_public_schema(schema: Any, label: str) -> None:
    """Allow only the small public-contract JSON Schema vocabulary used here."""
    if not isinstance(schema, dict):
        raise RoleStudyError(f"guided schema {label} is not an object")
    allowed = {"type", "properties", "required", "additionalProperties", "enum"}
    if not set(schema) <= allowed or "type" not in schema:
        raise RoleStudyError(f"guided schema {label} has unsupported keywords")
    schema_type = schema["type"]
    if isinstance(schema_type, list):
        if len(schema_type) < 2 or len(set(schema_type)) != len(schema_type) \
                or not all(item in {"string", "number", "integer", "boolean", "null"}
                           for item in schema_type):
            raise RoleStudyError(f"guided schema {label} has malformed union type")
    elif schema_type not in {"object", "string", "number", "integer", "boolean", "null"}:
        raise RoleStudyError(f"guided schema {label} has unsupported type")
    enum = schema.get("enum")
    if enum is not None and (not isinstance(enum, list) or len(enum) < 2
                             or len({canonical(item) for item in enum}) != len(enum)):
        raise RoleStudyError(f"guided schema {label} enum must contain alternatives")
    if schema_type == "object":
        properties, required = schema.get("properties"), schema.get("required")
        if not isinstance(properties, dict) or not properties \
                or not all(isinstance(key, str) for key in properties) \
                or not isinstance(required, list) or set(required) != set(properties) \
                or len(required) != len(properties) \
                or schema.get("additionalProperties") is not False:
            raise RoleStudyError(f"guided schema {label} object contract is malformed")
        for key, child in properties.items():
            _validate_public_schema(child, f"{label}.{key}")
    elif any(key in schema for key in ("properties", "required", "additionalProperties")):
        raise RoleStudyError(f"guided schema {label} non-object has object keywords")


def validate_guided_plan(value: Any, cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not isinstance(value, dict) or value.get("schema_version") != 1 \
            or value.get("kind") != "guided_public_schema_matched_six" \
            or tuple(value.get("case_ids", ())) != GUIDED_CASE_IDS \
            or value.get("max_tokens") != 512 or value.get("seed") != 20_261_023:
        raise RoleStudyError("guided public-schema probe plan differs")
    manifest_ids = {case["id"] for case in cases}
    if not set(GUIDED_CASE_IDS) <= manifest_ids:
        raise RoleStudyError("guided public-schema probe IDs are absent from the manifest")
    overrides = value.get("request_overrides_by_case")
    if not isinstance(overrides, dict) or tuple(overrides) != GUIDED_CASE_IDS:
        raise RoleStudyError("guided public-schema overrides differ from selected cases")
    for identifier, override in overrides.items():
        if not isinstance(override, dict) or set(override) != {"response_format"}:
            raise RoleStudyError(f"guided override for {identifier} changes request fields")
        response_format = override["response_format"]
        if not isinstance(response_format, dict) \
                or set(response_format) != {"type", "json_schema"} \
                or response_format.get("type") != "json_schema":
            raise RoleStudyError(f"guided response format for {identifier} differs")
        contract = response_format["json_schema"]
        if not isinstance(contract, dict) or set(contract) != {"name", "strict", "schema"} \
                or not isinstance(contract.get("name"), str) or not contract["name"] \
                or contract.get("strict") is not True:
            raise RoleStudyError(f"guided JSON schema wrapper for {identifier} differs")
        _validate_public_schema(contract["schema"], identifier)
    return copy.deepcopy(value)


def archive_empty_wait_failure(study: Path, status: Mapping[str, Any]) -> bool:
    """Consume only the old unbound waiter failure, never a bound/data run."""
    if not (
        status.get("schema_version") == 1
        and status.get("kind") == STUDY_KIND
        and status.get("status") == "failed"
        and status.get("models") == {}
        and "manifest_sha256" not in status
        and "model_plan_sha256" not in status
        and not (study / "source-pins.json").exists()
        and not (study / "models").exists()
        and not any(study.glob("*-model-verification.json"))
        and not any(study.glob("*-server-launch.json"))
        and not any(study.glob("*-resource-snapshot.json"))
        and not (study / "current-launch.json").exists()
    ):
        return False
    digest = json_sha(status); pid = status.get("controller_pid", "unknown")
    destination = study / "history" / f"unbound-wait-{pid}-{digest[:12]}.json"
    if destination.exists():
        if read_json(destination) != status:
            raise RoleStudyError("unbound wait-history receipt collision")
    else:
        atomic_json(destination, status)
    return True


def percentile(values: Sequence[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(probability * len(ordered)) - 1)]


def memory_kib(value: Any) -> int | None:
    if not isinstance(value, str):
        return None
    match = re.fullmatch(r"\s*([0-9]+)\s+kB\s*", value)
    return int(match.group(1)) if match else None


def validate_model_plan(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise RoleStudyError("model plan header differs")
    rows = value.get("models") if isinstance(value, dict) else None
    if not isinstance(rows, list) or len(rows) != 5 or not all(isinstance(row, dict) for row in rows):
        raise RoleStudyError("model plan must contain exactly five model objects")
    required = {"key", "alias", "file", "remote_path", "size", "sha256",
                "revision", "repo", "recipe", "mode_label"}
    result: list[dict[str, Any]] = []
    for row in rows:
        if not required <= set(row):
            raise RoleStudyError(f"model plan row lacks fields: {sorted(required - set(row))}")
        if not re.fullmatch(r"[a-z0-9][a-z0-9._-]*", str(row["key"])):
            raise RoleStudyError("model key is unsafe for durable paths")
        if not isinstance(row["recipe"], dict) or not isinstance(row["size"], int):
            raise RoleStudyError("model recipe/size has wrong type")
        if not re.fullmatch(r"[0-9a-f]{64}", str(row["sha256"])):
            raise RoleStudyError("model SHA-256 is malformed")
        if not str(row["remote_path"]).startswith(REMOTE_ROOT + "/models/"):
            raise RoleStudyError("model path is outside the private model root")
        forbidden = {"model", "messages", "max_tokens", "seed", "stream",
                     "stream_options", "timings_per_token", "response_format"} \
            & set(row["recipe"])
        if forbidden:
            raise RoleStudyError(f"recipe overrides controller fields: {sorted(forbidden)}")
        result.append(copy.deepcopy(row))
    if tuple(row["key"] for row in result) != MODEL_ORDER:
        raise RoleStudyError("model plan order differs")
    for index in (1, 2, 4):
        if result[index]["recipe"].get("chat_template_kwargs") != {"enable_thinking": False}:
            raise RoleStudyError(f"{result[index]['key']} must explicitly disable thinking")
    lfm26_recipe = result[0]["recipe"]
    if lfm26_recipe.get("reasoning_budget_tokens") != 0 \
            or lfm26_recipe.get("reasoning_effort") != "low" \
            or lfm26_recipe.get("chat_template_kwargs") != {"enable_thinking": True}:
        raise RoleStudyError(
            "LFM2.6 V2 must bind thinking=true, reasoning_effort=low, and zero budget"
        )
    if "chat_template_kwargs" in result[3]["recipe"]:
        raise RoleStudyError("LFM1.2 Instruct must not invent a thinking flag")
    return result


def validate_manifest(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, dict) or value.get("schema_version") != 1 \
            or value.get("kind") != "tachyon_fast_role_screening" or value.get("seed") != 20_261_023 \
            or value.get("request_settings") != {"max_tokens": 512,
                                                  "request_timeout_seconds": 180,
                                                  "json_constrained_decoding": False}:
        raise RoleStudyError("role manifest header differs")
    cases = value.get("cases")
    if not isinstance(cases, list) or len(cases) != 24:
        raise RoleStudyError("role manifest must contain exactly 24 cases")
    seen: set[str] = set(); counts = {suite: 0 for suite in SUITES}
    for case in cases:
        if not isinstance(case, dict) or not {"id", "suite", "messages", "expected", "required_keys"} <= set(case):
            raise RoleStudyError("manifest case lacks required fields")
        identifier = str(case["id"])
        if identifier in seen:
            raise RoleStudyError(f"duplicate case ID: {identifier}")
        seen.add(identifier); suite = case["suite"]
        if suite not in counts:
            raise RoleStudyError(f"unknown role suite: {suite}")
        counts[suite] += 1; messages = case["messages"]
        if (not isinstance(messages, list) or len(messages) != 2
                or [row.get("role") for row in messages if isinstance(row, dict)] != ["system", "user"]
                or any(not isinstance(row.get("content"), str) for row in messages)):
            raise RoleStudyError(f"case {identifier} must have system+user messages")
        required = case["required_keys"]
        if not isinstance(required, list) or len(set(required)) != len(required) \
                or not all(isinstance(key, str) for key in required):
            raise RoleStudyError(f"case {identifier} required_keys is malformed")
        if not isinstance(case["expected"], dict) or not set(case["expected"]) <= set(required):
            raise RoleStudyError(f"case {identifier} expected fields are malformed")
        limit = case.get("max_summary_words")
        if suite == "summary" and limit != 35:
            raise RoleStudyError(f"summary case {identifier} must bind 35 words")
        if suite != "summary" and limit is not None:
            raise RoleStudyError(f"non-summary case {identifier} has a word limit")
    if counts != {suite: 8 for suite in SUITES}:
        raise RoleStudyError(f"role suite counts differ: {counts}")
    return copy.deepcopy(cases)


def server_command(model: Mapping[str, Any]) -> list[str]:
    return ["taskset", "-c", "4-7", REMOTE_RUNTIME + "/bin/llama-server",
            "-m", str(model["remote_path"]), "--alias", str(model["alias"]),
            "--host", "127.0.0.1", "--port", "18085", "-c", "16384",
            "-np", "1", "-t", "4", "-tb", "4", "-b", "512", "-ub", "128",
            "-ngl", "0", "--jinja", "--reasoning-format", "deepseek",
            "--reasoning-effort", "none", "--threads-http", "2"]


def validate_server(receipt: Mapping[str, Any], model: Mapping[str, Any]) -> None:
    command = list(receipt.get("argv", []))
    if command[:3] == ["taskset", "-c", "4-7"]:
        command = command[3:]
    expected = server_command(model)[3:]
    if Counter(command) != Counter(expected):
        raise RoleStudyError(f"server argv differs for {model['key']}")
    for flag in ("-m", "--alias", "--host", "--port", "-c", "-np", "-t",
                 "-tb", "-b", "-ub", "-ngl", "--reasoning-format",
                 "--reasoning-effort", "--threads-http"):
        if command[command.index(flag) + 1] != expected[expected.index(flag) + 1]:
            raise RoleStudyError(f"server {flag} differs for {model['key']}")
    if receipt.get("alias") not in (None, model["alias"]):
        raise RoleStudyError("server receipt alias differs")


class Controller:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args; self.study = args.study.resolve(); self.plan_path = args.model_plan.resolve()
        self.guided_plan_path = (args.guided_probe_plan.resolve()
                                 if args.guided_probe_plan is not None else None)
        self.handoff_receipt_path = (args.handoff_receipt.resolve()
                                     if args.handoff_receipt is not None else None)
        self.status_path = self.study / "study-status.json"; self.events_path = self.study / "events.jsonl"
        self.state: dict[str, Any] = {}; self.models: list[dict[str, Any]] = []
        self.cases: list[dict[str, Any]] = []; self.manifest_sha = self.plan_sha = ""
        self.guided_plan: dict[str, Any] | None = None; self.guided_plan_sha: str | None = None
        self.handoff_sha: str | None = None
        self.gpu_restore: dict[str, Any] | None = None; self.current: dict[str, Any] | None = None
        self.lock_stream: io.TextIOWrapper | None = None; self.interrupted: int | None = None

    def event(self, name: str, **fields: Any) -> None:
        row = {"at": now(), "event": name, **fields}; append_jsonl(self.events_path, row)
        print(json.dumps(row, sort_keys=True), flush=True)

    def save(self, **fields: Any) -> None:
        self.state.update(fields, controller_pid=os.getpid(), updated_at=now()); atomic_json(self.status_path, self.state)

    def acquire_lock(self) -> None:
        self.study.mkdir(parents=True, exist_ok=True); self.lock_stream = (self.study / ".controller.lock").open("a+")
        try: fcntl.flock(self.lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc: raise RoleStudyError("another role-study controller holds the lock") from exc
        self.lock_stream.seek(0); self.lock_stream.truncate(); self.lock_stream.write(f"{os.getpid()}\n"); self.lock_stream.flush()

    def check_interrupt(self) -> None:
        if self.interrupted is not None: raise RoleStudyError(f"controller interrupted by signal {self.interrupted}")

    def handle_signal(self, signum: int, _frame: Any) -> None:
        self.interrupted = signum

    def wait_for_gpu(self) -> dict[str, Any]:
        deadline = time.monotonic() + self.args.wait_timeout; latest = "GPU trial status absent"
        while time.monotonic() < deadline:
            self.check_interrupt(); status_path = GPU_TRIAL / "trial-status.json"; launch_path = GPU_TRIAL / "controller-launch.json"
            if status_path.is_file() and launch_path.is_file():
                status, launch = read_json(status_path), read_json(launch_path)
                if not isinstance(status, dict) or not isinstance(launch, dict): raise RoleStudyError("GPU status/launch malformed")
                pid = status.get("controller_pid")
                if pid != launch.get("pid"):
                    latest = f"GPU status PID {pid!r} has not caught up to launch PID {launch.get('pid')!r}"
                    time.sleep(self.args.poll); continue
                if launch.get("source_sha256") != sha256(GPU_CONTROLLER): raise RoleStudyError("GPU launch source differs")
                identity = process_identity(pid) if isinstance(pid, int) and not isinstance(pid, bool) else {"alive": False}
                free = lock_is_free(GPU_TRIAL / ".controller.lock")
                if status.get("status") == "complete" and not identity.get("alive") and free:
                    restore = read_json(GPU_TRIAL / "cpu-restore-server-launch.json")
                    current = read_json(GPU_TRIAL / "current-launch.json")
                    if not isinstance(restore, dict) or not isinstance(current, dict) or current.get("server") != restore \
                            or current.get("default_restored") is not True:
                        raise RoleStudyError("GPU CPU restore/current receipts disagree")
                    return {"status": status, "launch": launch, "restore": restore}
                latest = f"status={status.get('status')}, alive={identity.get('alive')}, lock_free={free}"
            time.sleep(self.args.poll)
        raise RoleStudyError(f"timed out waiting for GPU handoff: {latest}")

    def recovery_handoff(self, model: Mapping[str, Any], *, require_live: bool) -> dict[str, Any]:
        """Validate the explicitly attested v1 restoration without rewriting v1/GPU state."""
        if self.handoff_receipt_path is None:
            raise RoleStudyError("recovery handoff requested without --handoff-receipt")
        if sha256(self.handoff_receipt_path) != HANDOFF_SHA256:
            raise RoleStudyError("published parent handoff SHA-256 differs")
        value = read_json(self.handoff_receipt_path)
        if not isinstance(value, dict) or value.get("schema_version") != 1 \
                or value.get("kind") != "tachyon_role_v2_parent_handoff":
            raise RoleStudyError("parent handoff header differs")
        expected_source = str(V1_DEFAULT_RESTORE.resolve())
        expected_launch = str(V1_CONTROLLER_LAUNCH.resolve())
        if value.get("source_receipt_path") != expected_source \
                or value.get("old_controller_launch_path") != expected_launch:
            raise RoleStudyError("parent handoff source paths differ")
        source_sha = sha256(V1_DEFAULT_RESTORE)
        launch_sha = sha256(V1_CONTROLLER_LAUNCH)
        if value.get("source_receipt_sha256") != source_sha \
                or value.get("old_controller_launch_sha256") != launch_sha:
            raise RoleStudyError("parent handoff source SHA-256 differs")
        source_server = read_json(V1_DEFAULT_RESTORE)
        server = value.get("server")
        if not isinstance(server, dict) or server != source_server:
            raise RoleStudyError("parent handoff server differs from the v1 receipt")
        validate_server(server, model)
        if value.get("alias") != model["alias"] or server.get("alias") != model["alias"]:
            raise RoleStudyError("parent handoff alias differs")
        old_launch = read_json(V1_CONTROLLER_LAUNCH)
        old_pid = value.get("old_controller_pid")
        if old_pid != old_launch.get("pid") or isinstance(old_pid, bool) \
                or not isinstance(old_pid, int):
            raise RoleStudyError("parent handoff old-controller PID differs")
        if old_launch.get("source_sha256") != DERIVED_SOURCE_SHA256:
            raise RoleStudyError("parent handoff old-controller source differs")
        if process_identity(old_pid).get("alive"):
            raise RoleStudyError("parent handoff old controller is still alive")
        raw = value.get("raw_server_identity")
        stable = ("pid", "start_ticks", "argv")
        if not isinstance(raw, dict) or raw.get("alive") is not True \
                or any(raw.get(field) != server.get(field) for field in stable):
            raise RoleStudyError("parent handoff raw server identity differs")
        if require_live:
            observed = self.remote_process(server["pid"])
            verify_owned(server, observed, "parent-attested recovery server")
            if any(observed.get(field) != raw.get(field) for field in stable):
                raise RoleStudyError("live server identity differs from parent attestation")
        return dict(server)

    def initialize(self, handoff: Mapping[str, Any]) -> None:
        manifest_path = self.study / "manifest.json"; self.cases = validate_manifest(read_json(manifest_path))
        self.models = validate_model_plan(read_json(self.plan_path)); self.manifest_sha = sha256(manifest_path); self.plan_sha = sha256(self.plan_path)
        if self.args.request_timeout != 180:
            raise RoleStudyError("request timeout must match the published 180-second binding")
        first_activation = not (self.study / "source-pins.json").exists()
        if self.handoff_receipt_path is not None:
            self.handoff_sha = sha256(self.handoff_receipt_path)
            self.gpu_restore = self.recovery_handoff(self.models[0],
                                                     require_live=first_activation)
        else:
            self.gpu_restore = dict(handoff["restore"])
        self.current = dict(self.gpu_restore)
        if self.manifest_sha != MANIFEST_SHA256 or self.plan_sha != PLAN_SHA256:
            raise RoleStudyError("published V2 role manifest/model plan SHA-256 differs")
        if sha256(DERIVED_SOURCE) != DERIVED_SOURCE_SHA256:
            raise RoleStudyError("pinned V1 derived source SHA-256 differs")
        if self.guided_plan_path is not None:
            self.guided_plan_sha = sha256(self.guided_plan_path)
            if self.guided_plan_sha != GUIDED_PLAN_SHA256:
                raise RoleStudyError("published guided probe plan SHA-256 differs")
            self.guided_plan = validate_guided_plan(read_json(self.guided_plan_path), self.cases)
        prior_state: dict[str, Any] | None = None
        if self.status_path.is_file():
            value = read_json(self.status_path)
            if not isinstance(value, dict):
                raise RoleStudyError("role status is malformed")
            if archive_empty_wait_failure(self.study, value):
                self.event("unbound_wait_failure_archived",
                           prior_controller_pid=value.get("controller_pid"))
            else:
                prior_state = value
        pin_paths = (Path(__file__).resolve(), DERIVED_SOURCE, manifest_path,
                     self.plan_path, GPU_CONTROLLER,
                     GPU_TRIAL / "trial-status.json", GPU_TRIAL / "controller-launch.json",
                     GPU_TRIAL / "cpu-restore-server-launch.json", GPU_TRIAL / "current-launch.json")
        if self.handoff_receipt_path is not None:
            pin_paths += (self.handoff_receipt_path, V1_DEFAULT_RESTORE,
                          V1_CONTROLLER_LAUNCH)
        pins = {str(path): sha256(path) for path in pin_paths}
        if self.guided_plan_path is not None:
            pins[str(self.guided_plan_path)] = self.guided_plan_sha
        pin_path = self.study / "source-pins.json"
        if pin_path.exists():
            if read_json(pin_path).get("sha256") != pins: raise RoleStudyError("role-study source pins changed")
        else: atomic_json(pin_path, {"schema_version": 1, "created_at": now(), "sha256": pins})
        if prior_state is not None:
            if prior_state.get("manifest_sha256") != self.manifest_sha \
                    or prior_state.get("model_plan_sha256") != self.plan_sha \
                    or prior_state.get("guided_probe_plan_sha256") != self.guided_plan_sha \
                    or prior_state.get("handoff_receipt_sha256") != self.handoff_sha:
                raise RoleStudyError("role status binding differs")
            self.state = prior_state
        if not self.state:
            self.state = {"schema_version": 1, "kind": STUDY_KIND,
                          "claim": "custom small test; not a public benchmark", "created_at": now(),
                          "status": "starting", "study": str(self.study), "manifest_sha256": self.manifest_sha,
                          "model_plan_sha256": self.plan_sha,
                          "guided_probe_plan_sha256": self.guided_plan_sha,
                          "handoff_receipt_sha256": self.handoff_sha,
                          "model_order": [row["key"] for row in self.models],
                          "models": {}, "controls": {"context_tokens": 16384, "max_tokens": 512,
                          "request_timeout_seconds": self.args.request_timeout, "seed": 20261023, "serial": True}}
        role_current_path = self.study / "current-launch.json"
        if role_current_path.is_file():
            role_current = read_json(role_current_path)
            if not isinstance(role_current, dict) or role_current.get("default_restored") is not True \
                    or not isinstance(role_current.get("server"), dict):
                raise RoleStudyError("role current-launch receipt is malformed")
            validate_server(role_current["server"], self.models[0])
            self.current = dict(role_current["server"])
        self.state["source_pins"] = pins; self.save(status="running", error=None)

    def remote_json(self, script: str, payload: Mapping[str, Any], timeout: float = 90) -> Any:
        command = shlex.join(["python3", "-c", script, json.dumps(payload, separators=(",", ":"))])
        try:
            completed = subprocess.run(["ssh", *SSH_OPTIONS, "-T", self.args.remote, command],
                                       stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                       text=True, timeout=timeout, check=False)
        except subprocess.TimeoutExpired as exc: raise RoleStudyError("SSH command timed out") from exc
        if completed.returncode: raise RoleStudyError(f"remote command failed: {completed.stderr.strip()[-800:]}")
        try: return json.loads(completed.stdout)
        except json.JSONDecodeError as exc: raise RoleStudyError("remote command returned invalid JSON") from exc

    def remote_process(self, pid: int) -> dict[str, Any]:
        script = r'''import json,pathlib,sys
p=json.loads(sys.argv[1]); pid=int(p["pid"]); b=pathlib.Path("/proc")/str(pid)
try:
 s=(b/"stat").read_text(); t=s[s.rfind(")")+2:].split(); a=[x.decode("utf-8","surrogateescape") for x in (b/"cmdline").read_bytes().split(b"\0") if x]
 print(json.dumps({"pid":pid,"alive":t[0]!="Z","state":t[0],"start_ticks":int(t[19]),"argv":a}))
except FileNotFoundError: print(json.dumps({"pid":pid,"alive":False,"state":"exited","argv":[]}))'''
        value = self.remote_json(script, {"pid": pid})
        if not isinstance(value, dict): raise RoleStudyError("remote process identity malformed")
        return value

    def remote_json_file(self, path: str) -> dict[str, Any] | None:
        script = r'''import json,pathlib,sys
p=pathlib.Path(json.loads(sys.argv[1])["path"])
if not p.is_file(): print("null")
else:
 try: print(json.dumps(json.loads(p.read_text())))
 except Exception as e: print(json.dumps({"_error":str(e)}))'''
        value = self.remote_json(script, {"path": path})
        if value is None: return None
        if not isinstance(value, dict) or value.get("_error"): raise RoleStudyError(f"invalid remote receipt: {path}")
        return value

    def verify_model(self, model: Mapping[str, Any]) -> None:
        path = self.study / f"{model['key']}-model-verification.json"
        if path.is_file():
            value = read_json(path); artifact = value.get("artifact", {})
            if value.get("plan_sha256") != self.plan_sha or value.get("path") != model["remote_path"] \
                    or value.get("size") != model["size"] or artifact.get("sha256") != model["sha256"] \
                    or artifact.get("actual_sha256") != model["sha256"] \
                    or artifact.get("path") != model["remote_path"] or artifact.get("size") != model["size"] \
                    or artifact.get("key") != model["key"] or artifact.get("file") != model["file"] \
                    or artifact.get("revision") != model["revision"]:
                raise RoleStudyError("model verification plan binding differs")
            return
        script = r'''import json,pathlib,sys
p=json.loads(sys.argv[1]); m=pathlib.Path(p["path"])
try: a=json.loads((m.parent/"artifact.json").read_text()); s=m.stat()
except Exception as e: print(json.dumps({"error":str(e)})); raise SystemExit(0)
print(json.dumps({"path":str(m),"size":s.st_size,"regular":m.is_file(),"artifact":a}))'''
        value = self.remote_json(script, {"path": model["remote_path"]})
        artifact = value.get("artifact") if isinstance(value, dict) else None
        if not isinstance(artifact, dict) or value.get("error") or not value.get("regular"):
            raise RoleStudyError(f"model observation failed for {model['key']}: {value}")
        if value.get("path") != model["remote_path"] or value.get("size") != model["size"] \
                or artifact.get("sha256") != model["sha256"] \
                or artifact.get("actual_sha256") != model["sha256"] \
                or artifact.get("path") != model["remote_path"] or artifact.get("size") != model["size"] \
                or artifact.get("key") != model["key"] or artifact.get("file") != model["file"] \
                or artifact.get("revision") != model["revision"]:
            raise RoleStudyError(f"model artifact receipt differs for {model['key']}")
        atomic_json(path, {"verified_at": now(), "plan_sha256": self.plan_sha, **value})

    def launch_server(self, model: Mapping[str, Any], label: str) -> dict[str, Any]:
        launch_argv = server_command(model); remote_receipt = REMOTE_RESULTS + f"/{label}-server-launch.json"
        script = r'''import json,os,pathlib,subprocess,sys,time
p=json.loads(sys.argv[1]); log=pathlib.Path(p["log"]); rec=pathlib.Path(p["receipt"]); log.parent.mkdir(parents=True,exist_ok=True); f=log.open("ab",buffering=0)
c=subprocess.Popen(p["argv"],stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True); b=pathlib.Path("/proc")/str(c.pid); end=time.monotonic()+2; a=[]
while time.monotonic()<end:
 if not b.exists(): raise SystemExit("server exited during launch")
 a=[x.decode("utf-8","surrogateescape") for x in (b/"cmdline").read_bytes().split(b"\0") if x]
 if a and a[0].endswith("/llama-server"): break
 time.sleep(.01)
if not a or not a[0].endswith("/llama-server"): raise SystemExit("taskset did not exec server")
s=(b/"stat").read_text(); t=s[s.rfind(")")+2:].split(); v={"schema_version":1,"kind":"tachyon_role_server","pid":c.pid,"start_ticks":int(t[19]),"argv":a,"launch_argv":p["argv"],"model_key":p["model_key"],"alias":p["alias"],"launched_at":p["at"],"log":str(log)}
x=rec.with_name(rec.name+".tmp-"+str(os.getpid())); x.write_text(json.dumps(v,sort_keys=True)+"\n"); os.replace(x,rec); print(json.dumps(v))'''
        receipt = self.remote_json(script, {
            "argv": launch_argv, "receipt": remote_receipt,
            "log": REMOTE_RESULTS + f"/{label}-server.log", "at": now(),
            "model_key": model["key"], "alias": model["alias"],
        })
        if not isinstance(receipt, dict): raise RoleStudyError("server launch receipt malformed")
        validate_server(receipt, model); verify_owned(receipt, self.remote_process(receipt["pid"]), "launched server")
        receipt["remote_receipt"] = remote_receipt
        atomic_json(self.study / f"{label}-server-launch.json", receipt)
        self.current = receipt; self.event("server_launched", model=model["key"], pid=receipt["pid"])
        return receipt

    def adopt_gpu_restore(self, model: Mapping[str, Any]) -> dict[str, Any]:
        assert self.gpu_restore is not None
        validate_server(self.gpu_restore, model); identity = self.remote_process(int(self.gpu_restore.get("pid", -1)))
        if not verify_owned(self.gpu_restore, identity, "inherited CPU server"):
            raise RoleStudyError("inherited CPU server exited before handoff")
        inherited_from = (str(self.handoff_receipt_path)
                          if self.handoff_receipt_path is not None
                          else str(GPU_TRIAL / "cpu-restore-server-launch.json"))
        receipt = {**self.gpu_restore, "role_handoff_at": now(),
                   "inherited_from": inherited_from,
                   "handoff_kind": ("parent_attested_v1_recovery"
                                    if self.handoff_receipt_path is not None
                                    else "gpu_cpu_restore")}
        atomic_json(self.study / f"{model['key']}-server-launch.json", receipt)
        self.current = receipt
        self.event("recovery_server_adopted" if self.handoff_receipt_path is not None
                   else "gpu_restore_adopted", model=model["key"], pid=receipt["pid"])
        return receipt

    def wait_endpoint(self, model: Mapping[str, Any], receipt: Mapping[str, Any]) -> None:
        deadline = time.monotonic() + self.args.server_timeout; latest = "not queried"
        while time.monotonic() < deadline:
            self.check_interrupt(); verify_owned(receipt, self.remote_process(receipt["pid"]), "role server")
            try:
                with urlopen(self.args.endpoint + "/models", timeout=10) as response: value = json.load(response)
                aliases = [row.get("id") for row in value.get("data", []) if isinstance(row, dict)]
                if aliases == [model["alias"]] or (model["alias"] in aliases and len(set(aliases)) == 1): return
                latest = repr(aliases)
            except (OSError, HTTPError, URLError, json.JSONDecodeError) as exc: latest = str(exc)
            time.sleep(2)
        raise RoleStudyError(f"endpoint did not advertise {model['alias']}: {latest}")

    def stop_current(self, model: Mapping[str, Any], label: str) -> None:
        if self.current is None: return
        receipt = self.current; identity = self.remote_process(receipt["pid"])
        if not verify_owned(receipt, identity, "role server"):
            self.current = None; return
        script = r'''import json,os,pathlib,signal,sys,time
p=json.loads(sys.argv[1]); pid=int(p["pid"]); b=pathlib.Path("/proc")/str(pid)
def ident():
 try:
  s=(b/"stat").read_text(); t=s[s.rfind(")")+2:].split(); a=[x.decode("utf-8","surrogateescape") for x in (b/"cmdline").read_bytes().split(b"\0") if x]; return {"pid":pid,"alive":t[0]!="Z","state":t[0],"start_ticks":int(t[19]),"argv":a}
 except FileNotFoundError: return {"pid":pid,"alive":False,"state":"exited","argv":[]}
i=ident()
if not i["alive"]: print(json.dumps({"stopped":True,"already_exited":True,"identity":i})); raise SystemExit(0)
if i["start_ticks"]!=p["start_ticks"] or i["argv"]!=p["argv"]: print(json.dumps({"error":"ownership mismatch","identity":i})); raise SystemExit(0)
os.kill(pid,signal.SIGTERM); end=time.monotonic()+60
while time.monotonic()<end:
 i=ident()
 if not i["alive"]: print(json.dumps({"stopped":True,"identity":i})); raise SystemExit(0)
 time.sleep(.2)
print(json.dumps({"error":"server did not stop","identity":ident()}))'''
        result = self.remote_json(script, dict(receipt), timeout=75)
        if not isinstance(result, dict) or result.get("stopped") is not True:
            raise RoleStudyError(f"failed to stop owned server: {result}")
        atomic_json(self.study / f"{label}-server-stop.json",
                    {"stopped_at": now(), "ownership": receipt, "result": result})
        self.current = None; self.event("server_stopped", model=model["key"], pid=receipt["pid"])

    def snapshot(self, model: Mapping[str, Any], receipt: Mapping[str, Any]) -> None:
        path = self.study / f"{model['key']}-resource-snapshot.json"
        if path.is_file(): return
        script = r'''import glob,json,os,pathlib,sys
p=json.loads(sys.argv[1]); pid=int(p["pid"]); b=pathlib.Path("/proc")/str(pid); s=(b/"stat").read_text(); t=s[s.rfind(")")+2:].split(); a=[x.decode("utf-8","surrogateescape") for x in (b/"cmdline").read_bytes().split(b"\0") if x]; st={}; mem={}
for line in (b/"status").read_text().splitlines():
 if ":" in line and line.split(":",1)[0] in ("VmRSS","VmHWM","VmSwap"): st[line.split(":",1)[0]]=line.split(":",1)[1].strip()
for line in pathlib.Path("/proc/meminfo").read_text().splitlines():
 if ":" in line and line.split(":",1)[0] in ("MemTotal","MemAvailable","SwapTotal","SwapFree"): mem[line.split(":",1)[0]]=line.split(":",1)[1].strip()
thermal=[]
for f in sorted(glob.glob("/sys/class/thermal/thermal_zone*/temp")):
 d=pathlib.Path(f).parent
 try: thermal.append({"zone":d.name,"type":(d/"type").read_text().strip(),"millidegrees_c":int((d/"temp").read_text())})
 except Exception: pass
cpufreq=[]
for dname in sorted(glob.glob("/sys/devices/system/cpu/cpufreq/policy*")):
 d=pathlib.Path(dname); row={"policy":d.name}
 for name in ("scaling_governor","scaling_cur_freq","scaling_max_freq","cpuinfo_max_freq"):
  try: row[name]=(d/name).read_text().strip()
  except OSError: row[name]=None
 cpufreq.append(row)
try: cpu_online=pathlib.Path("/sys/devices/system/cpu/online").read_text().strip()
except OSError: cpu_online=None
print(json.dumps({"pid":pid,"alive":t[0]!="Z","state":t[0],"start_ticks":int(t[19]),"argv":a,"process_status":st,"meminfo":mem,"thermal":thermal,"cpu_online":cpu_online,"cpu_count":os.cpu_count(),"cpufreq":cpufreq}))'''
        value = self.remote_json(script, {"pid": receipt["pid"]}); verify_owned(receipt, value, "snapshot server")
        atomic_json(path, {"captured_at": now(), "model_key": model["key"], **value})

    def binding(self, model: Mapping[str, Any]) -> dict[str, Any]:
        settings = {"model_key": model["key"], "alias": model["alias"], "recipe": model["recipe"],
                    "mode_label": model["mode_label"], "max_tokens": 512,
                    "request_timeout_seconds": self.args.request_timeout, "seed": 20261023,
                    "context_tokens": 16384, "mode": "fast_role_custom_test_v2",
                    "protocol_revision": "lfm26_thinking_true_low_zero_budget"}
        return {"schema_version": 1, "manifest_sha256": self.manifest_sha,
                "model_plan_sha256": self.plan_sha, "settings": settings,
                "settings_sha256": json_sha(settings), "created_at": now()}

    def ensure_binding(self, model: Mapping[str, Any]) -> dict[str, Any]:
        output = self.study / "models" / model["key"]; output.mkdir(parents=True, exist_ok=True)
        path = output / "run.json"; expected = self.binding(model)
        if path.is_file():
            actual = read_json(path); comparable = {k: actual.get(k) for k in expected if k != "created_at"}
            if comparable != {k: v for k, v in expected.items() if k != "created_at"}:
                raise RoleStudyError(f"resume binding differs for {model['key']}")
            return actual
        atomic_json(path, expected); return expected

    def ensure_guided_binding(self, model: Mapping[str, Any],
                              baseline: Mapping[str, Any]) -> dict[str, Any]:
        if self.guided_plan is None or self.guided_plan_sha is None:
            raise RoleStudyError("guided binding requested without a plan")
        output = self.study / "models" / model["key"] / "guided-public-schema-probe"
        output.mkdir(parents=True, exist_ok=True)
        settings = {
            **baseline["settings"],
            "experiment": "guided_public_schema_matched_six",
            "request_overrides_by_case": copy.deepcopy(
                self.guided_plan["request_overrides_by_case"]
            ),
            "guided_probe_plan_sha256": self.guided_plan_sha,
            "baseline_settings_sha256": baseline["settings_sha256"],
        }
        expected = {
            "schema_version": 1, "manifest_sha256": self.manifest_sha,
            "model_plan_sha256": self.plan_sha,
            "guided_probe_plan_sha256": self.guided_plan_sha,
            "baseline_settings_sha256": baseline["settings_sha256"],
            "settings": settings, "settings_sha256": json_sha(settings),
            "created_at": now(),
        }
        path = output / "run.json"
        if path.is_file():
            actual = read_json(path)
            comparable = {key: actual.get(key) for key in expected if key != "created_at"}
            if comparable != {key: value for key, value in expected.items()
                              if key != "created_at"}:
                raise RoleStudyError(f"guided resume binding differs for {model['key']}")
            return actual
        atomic_json(path, expected)
        return expected

    def case_dir(self, model: Mapping[str, Any], case: Mapping[str, Any],
                 *, guided: bool = False) -> Path:
        storage = hashlib.sha256(str(case["id"]).encode()).hexdigest()[:16]
        root = self.study / "models" / model["key"]
        if guided:
            root = root / "guided-public-schema-probe"
        return root / "cases" / storage

    def run_case(self, model: Mapping[str, Any], case: Mapping[str, Any],
                 binding: Mapping[str, Any], *, guided: bool = False) -> dict[str, Any]:
        directory = self.case_dir(model, case, guided=guided); result_path = directory / "result.json"
        if result_path.is_file():
            result = read_json(result_path)
            if result.get("manifest_sha256") != self.manifest_sha or result.get("settings_sha256") != binding["settings_sha256"] \
                    or result.get("case_id") != case["id"] \
                    or result.get("experiment") != ("guided_public_schema_matched_six" if guided else "baseline_free_generation"):
                raise RoleStudyError(f"terminal result binding differs for {case['id']}")
            return result
        directory.mkdir(parents=True, exist_ok=True)
        payload = (build_guided_request(case, model, self.guided_plan)
                   if guided and self.guided_plan is not None else build_request(case, model))
        request_path = directory / "request.json"
        if request_path.is_file() and read_json(request_path) != payload: raise RoleStudyError(f"saved request differs for {case['id']}")
        if not request_path.exists(): atomic_json(request_path, payload)
        started = time.monotonic(); chunks: list[dict[str, Any]] = []; final_parts: list[str] = []; reasoning_parts: list[str] = []
        first_ttft = final_ttft = None; finish_reason = None; error = None
        try:
            request = Request(self.args.endpoint + "/chat/completions", data=canonical(payload),
                              headers={"Content-Type": "application/json"}, method="POST")
            with urlopen(request, timeout=self.args.request_timeout) as response:
                for raw in response:
                    line = raw.decode("utf-8").strip()
                    if not line.startswith("data:") or line[5:].strip() == "[DONE]": continue
                    row = json.loads(line[5:].strip()); chunks.append(row)
                    for choice in row.get("choices", []):
                        if not isinstance(choice, dict): continue
                        if choice.get("finish_reason") is not None: finish_reason = choice["finish_reason"]
                        delta = choice.get("delta", {})
                        if not isinstance(delta, dict): continue
                        reasoning = delta.get("reasoning_content")
                        if not isinstance(reasoning, str): reasoning = delta.get("reasoning")
                        content = delta.get("content")
                        if isinstance(reasoning, str) and reasoning:
                            if first_ttft is None: first_ttft = time.monotonic() - started
                            reasoning_parts.append(reasoning)
                        if isinstance(content, str) and content:
                            if first_ttft is None: first_ttft = time.monotonic() - started
                            if final_ttft is None: final_ttft = time.monotonic() - started
                            final_parts.append(content)
        except (OSError, HTTPError, URLError, UnicodeError, json.JSONDecodeError) as exc:
            error = {"type": type(exc).__name__, "message": str(exc)}
        wall = time.monotonic() - started; last = chunks[-1] if chunks else {}
        response_value = {"chunks": chunks, "final_text": "".join(final_parts),
                          "reasoning_text": "".join(reasoning_parts), "wall_seconds": wall,
                          "ttft_first_seconds": first_ttft, "ttft_final_seconds": final_ttft,
                          "finish_reason": finish_reason, "last": last, "error": error}
        atomic_json(directory / "response.json", response_value); evaluation = evaluate_answer(case, response_value["final_text"])
        timing = last.get("timings", {}) if isinstance(last, dict) else {}; usage = last.get("usage", {}) if isinstance(last, dict) else {}
        result = {"schema_version": 1, "status": "api_error" if error else "completed",
                  "experiment": "guided_public_schema_matched_six" if guided else "baseline_free_generation",
                  "case_id": case["id"], "suite": case["suite"], "manifest_sha256": self.manifest_sha,
                  "settings_sha256": binding["settings_sha256"], "request_sha256": json_sha(payload),
                  "finished_at": now(), "final_text": response_value["final_text"],
                  "reasoning_text": response_value["reasoning_text"], "finish_reason": finish_reason,
                  "wall_seconds": wall, "ttft_first_seconds": first_ttft, "ttft_final_seconds": final_ttft,
                  "usage": usage, "native_timings": timing, "evaluation": evaluation, "error": error}
        atomic_json(result_path, result); return result

    def write_model_status(self, model: Mapping[str, Any], results: Sequence[Mapping[str, Any]],
                           binding: Mapping[str, Any]) -> None:
        counts = {"total": len(results), "completed": 0, "api_error": 0,
                  "semantic_pass": 0, "manual_review_required": 0}; cases = []
        for result in results:
            counts[result["status"]] = counts.get(result["status"], 0) + 1
            counts["semantic_pass"] += int(result.get("evaluation", {}).get("semantic_pass") is True)
            counts["manual_review_required"] += int(result.get("evaluation", {}).get("manual_review_required") is True)
            cases.append({"id": result["case_id"], "suite": result["suite"], "status": result["status"],
                          "semantic_pass": result.get("evaluation", {}).get("semantic_pass")})
        atomic_json(self.study / "models" / model["key"] / "status.json",
                    {"schema_version": 1, "settings_sha256": binding["settings_sha256"],
                     "counts": counts, "cases": cases, "updated_at": now()})

    def guided_cases(self) -> list[dict[str, Any]]:
        by_id = {case["id"]: case for case in self.cases}
        return [by_id[identifier] for identifier in GUIDED_CASE_IDS]

    def guided_complete(self, model: Mapping[str, Any],
                        baseline: Mapping[str, Any]) -> bool:
        if self.guided_plan is None:
            return True
        cases = self.guided_cases()
        if not all((self.case_dir(model, case, guided=True) / "result.json").is_file()
                   for case in cases):
            return False
        binding = self.ensure_guided_binding(model, baseline)
        for case in cases:
            self.run_case(model, case, binding, guided=True)
        return True

    def run_guided(self, model: Mapping[str, Any], baseline: Mapping[str, Any]) -> None:
        if self.guided_plan is None:
            return
        binding = self.ensure_guided_binding(model, baseline)
        results = [self.run_case(model, case, binding, guided=True)
                   for case in self.guided_cases()]
        counts = {
            "total": len(results),
            "completed": sum(result["status"] == "completed" for result in results),
            "api_error": sum(result["status"] == "api_error" for result in results),
            "json_valid": sum(result["evaluation"]["json_valid"] is True
                              for result in results),
            "semantic_pass": sum(result["evaluation"]["semantic_pass"] is True
                                 for result in results),
            "manual_review_required": sum(
                result["evaluation"]["manual_review_required"] is True
                for result in results),
        }
        atomic_json(self.study / "models" / model["key"] / "guided-public-schema-probe" /
                    "status.json",
                    {"schema_version": 1, "guided_probe_plan_sha256": self.guided_plan_sha,
                     "settings_sha256": binding["settings_sha256"], "counts": counts,
                     "case_ids": list(GUIDED_CASE_IDS), "updated_at": now()})

    def run_model(self, model: Mapping[str, Any], index: int) -> None:
        key = model["key"]; binding = self.ensure_binding(model)
        present = [self.run_case(model, case, binding) for case in self.cases
                   if (self.case_dir(model, case) / "result.json").is_file()]
        needs_server = (len(present) < len(self.cases) or not self.guided_complete(model, binding)
                        or not (self.study / f"{key}-resource-snapshot.json").is_file())
        receipt: dict[str, Any] | None = None; receipt_path = self.study / f"{key}-server-launch.json"
        candidate = read_json(receipt_path) if receipt_path.is_file() else None
        if candidate is None:
            remote_candidate = self.remote_json_file(REMOTE_RESULTS + f"/{key}-server-launch.json")
            if remote_candidate is not None:
                validate_server(remote_candidate, model)
                if verify_owned(remote_candidate, self.remote_process(remote_candidate["pid"]), "recovered role server"):
                    candidate = remote_candidate; atomic_json(receipt_path, candidate)
                    self.event("server_receipt_recovered", model=key, pid=candidate["pid"])
        if isinstance(candidate, dict):
            validate_server(candidate, model)
            if verify_owned(candidate, self.remote_process(candidate["pid"]), "resumed role server"):
                receipt = candidate; self.current = candidate
        if needs_server:
            fresh_gpu_handoff = index == 0 and not receipt_path.exists() and self.current == self.gpu_restore
            if receipt is None and self.current is not None and not fresh_gpu_handoff:
                current_identity = self.remote_process(int(self.current.get("pid", -1)))
                if verify_owned(self.current, current_identity, "current role service"):
                    try:
                        validate_server(self.current, model); receipt = self.current
                    except RoleStudyError:
                        active = next((row for row in self.models
                                       if row["key"] == self.current.get("model_key")
                                       or row["alias"] == self.current.get("alias")), self.models[0])
                        self.stop_current(active, "resume-current")
            if receipt is None:
                receipt = self.adopt_gpu_restore(model) if index == 0 and not receipt_path.exists() \
                    else self.launch_server(model, key)
            self.wait_endpoint(model, receipt)
            for case in self.cases:
                self.check_interrupt(); self.run_case(model, case, binding)
            self.run_guided(model, binding)
            self.snapshot(model, receipt)
        results = [read_json(self.case_dir(model, case) / "result.json") for case in self.cases]
        self.write_model_status(model, results, binding)
        if receipt is not None: self.stop_current(model, key)
        self.state["models"].setdefault(key, {}).update(
            status="complete", completed_at=now(), output=str(self.study / "models" / key))
        self.save(current_model=key); self.event("model_complete", model=key)

    def restore_default(self) -> None:
        if not self.models: return
        model = self.models[0]; receipt: dict[str, Any]
        if self.current is not None and verify_owned(
                self.current, self.remote_process(self.current["pid"]), "current role server"):
            try:
                validate_server(self.current, model); self.wait_endpoint(model, self.current); receipt = self.current
            except RoleStudyError:
                active = next((row for row in self.models if row["key"] == self.current.get("model_key")), model)
                self.stop_current(active, "failure-current"); receipt = self.launch_server(model, "default-restore")
                self.wait_endpoint(model, receipt)
        else:
            restore_path = self.study / "default-restore-server-launch.json"
            candidate = read_json(restore_path) if restore_path.is_file() else None
            if isinstance(candidate, dict) and verify_owned(
                    candidate, self.remote_process(candidate["pid"]), "restored default server"):
                validate_server(candidate, model); self.wait_endpoint(model, candidate); receipt = candidate
            else:
                receipt = self.launch_server(model, "default-restore"); self.wait_endpoint(model, receipt)
        atomic_json(self.study / "current-launch.json",
                    {"selected_at": now(), "default_restored": True,
                     "reason": "LFM2.6 CPU4 default restored after custom role study", "server": receipt})
        self.current = receipt; self.event("default_service_restored", pid=receipt["pid"], alias=model["alias"])

    def consolidate(self) -> None:
        case_rows: list[dict[str, Any]] = []; model_rows: list[dict[str, Any]] = []
        for model in self.models:
            results = [read_json(self.case_dir(model, case) / "result.json") for case in self.cases]
            snapshot = read_json(self.study / f"{model['key']}-resource-snapshot.json")
            process_status = snapshot.get("process_status", {}) if isinstance(snapshot, dict) else {}
            for sequence, result in enumerate(results):
                timing = result.get("native_timings", {}); usage = result.get("usage", {}); evaluation = result["evaluation"]
                prompt_details = usage.get("prompt_tokens_details", {}) if isinstance(usage, dict) else {}
                completion_details = usage.get("completion_tokens_details", {}) if isinstance(usage, dict) else {}
                case_rows.append({
                    "model_key": model["key"], "case_id": result["case_id"], "suite": result["suite"],
                    "request_sequence": sequence, "cold_first_request": sequence == 0,
                    "status": result["status"], "semantic_pass": evaluation["semantic_pass"],
                    "json_valid": evaluation["json_valid"],
                    "exact_top_level_keys": evaluation["exact_top_level_keys"],
                    "expected_values_match": evaluation["expected_values_match"],
                    "missing_keys": json.dumps(evaluation["missing_keys"]),
                    "unexpected_keys": json.dumps(evaluation["unexpected_keys"]),
                    "mismatched_fields": json.dumps(evaluation["mismatched_fields"]),
                    "summary_word_count": evaluation["summary_word_count"],
                    "summary_word_limit_pass": evaluation["summary_word_limit_pass"],
                    "manual_review_required": evaluation["manual_review_required"],
                    "automated_check_scope": evaluation["claim_scope"],
                    "ttft_first_seconds": result.get("ttft_first_seconds"),
                    "ttft_final_seconds": result.get("ttft_final_seconds"),
                    "wall_seconds": result.get("wall_seconds"), "prompt_tokens": usage.get("prompt_tokens"),
                    "completion_tokens": usage.get("completion_tokens"),
                    "native_prompt_tokens": timing.get("prompt_n"),
                    "native_predicted_tokens": timing.get("predicted_n"),
                    "native_cache_tokens": timing.get("cache_n"),
                    "usage_cached_prompt_tokens": prompt_details.get("cached_tokens") if isinstance(prompt_details, dict) else None,
                    "reasoning_characters": len(result.get("reasoning_text", "")),
                    "reasoning_tokens": completion_details.get("reasoning_tokens") if isinstance(completion_details, dict) else None,
                    "native_prompt_tokens_per_second": timing.get("prompt_per_second"),
                    "native_generation_tokens_per_second": timing.get("predicted_per_second"),
                    "finish_reason": result.get("finish_reason"),
                    "error": result.get("error", {}).get("message") if isinstance(result.get("error"), dict) else None,
                })
            scoped = [row for row in case_rows if row["model_key"] == model["key"]]
            first_ttfts = [float(row["ttft_first_seconds"]) for row in scoped
                           if isinstance(row["ttft_first_seconds"], (int, float))]
            final_ttfts = [float(row["ttft_final_seconds"]) for row in scoped if isinstance(row["ttft_final_seconds"], (int, float))]
            walls = [float(row["wall_seconds"]) for row in scoped if isinstance(row["wall_seconds"], (int, float))]
            pairs = [(result.get("native_timings", {}).get("predicted_per_second"),
                      result.get("native_timings", {}).get("predicted_ms")) for result in results]
            valid_pairs = [(float(rate), float(ms)) for rate, ms in pairs
                           if isinstance(rate, (int, float)) and isinstance(ms, (int, float)) and ms > 0]
            prompt_pairs = [(result.get("native_timings", {}).get("prompt_per_second"),
                             result.get("native_timings", {}).get("prompt_ms")) for result in results]
            valid_prompt_pairs = [(float(rate), float(ms)) for rate, ms in prompt_pairs
                                  if isinstance(rate, (int, float)) and isinstance(ms, (int, float)) and ms > 0]
            suite_rates = {}
            for suite in SUITES:
                rows = [row for row in scoped if row["suite"] == suite]
                suite_rates[suite] = sum(row["semantic_pass"] is True for row in rows) / len(rows) if rows else None
            model_rows.append({
                "model_key": model["key"], "alias": model["alias"],
                "parameter_label": PARAMETER_LABELS[model["key"]],
                "model_weight_bytes": model["size"],
                "model_weight_megabytes_decimal": model["size"] / 1_000_000,
                "model_weight_mib": model["size"] / (1024 * 1024),
                "server_rss_kib": memory_kib(process_status.get("VmRSS")),
                "server_hwm_kib": memory_kib(process_status.get("VmHWM")),
                "server_swap_kib": memory_kib(process_status.get("VmSwap")),
                "case_count": len(scoped), "completed": sum(row["status"] == "completed" for row in scoped),
                "api_errors": sum(row["status"] == "api_error" for row in scoped),
                "semantic_passes": sum(row["semantic_pass"] is True for row in scoped),
                "routing_success_rate": suite_rates["routing"], "decision_success_rate": suite_rates["decision"],
                "summary_fact_success_rate": suite_rates["summary"],
                "manual_review_case_count": sum(row["manual_review_required"] is True for row in scoped),
                "ttft_first_p50_seconds": statistics.median(first_ttfts) if first_ttfts else None,
                "ttft_first_p95_seconds": percentile(first_ttfts, .95),
                "ttft_final_p50_seconds": statistics.median(final_ttfts) if final_ttfts else None,
                "ttft_final_p95_seconds": percentile(final_ttfts, .95),
                "complete_answer_p50_seconds": statistics.median(walls) if walls else None,
                "complete_answer_p95_seconds": percentile(walls, .95),
                "wall_seconds_sum": sum(walls) if walls else None,
                "prompt_tokens": sum(row["prompt_tokens"] for row in scoped if isinstance(row["prompt_tokens"], int)),
                "completion_tokens": sum(row["completion_tokens"] for row in scoped if isinstance(row["completion_tokens"], int)),
                "native_generation_tokens_per_second": (sum(rate * ms for rate, ms in valid_pairs) /
                                                          sum(ms for _, ms in valid_pairs) if valid_pairs else None),
                "native_prompt_tokens_per_second": (sum(rate * ms for rate, ms in valid_prompt_pairs) /
                                                      sum(ms for _, ms in valid_prompt_pairs) if valid_prompt_pairs else None),
                "reasoning_case_count": sum(bool(result.get("reasoning_text")) for result in results),
                "reasoning_characters": sum(len(result.get("reasoning_text", "")) for result in results),
                "reasoning_tokens_reported": sum(row["reasoning_tokens"] for row in scoped
                                                 if isinstance(row["reasoning_tokens"], int)),
                "native_cache_case_count": sum(isinstance(row["native_cache_tokens"], (int, float))
                                               and row["native_cache_tokens"] > 0 for row in scoped),
                "native_cache_tokens_sum": sum(row["native_cache_tokens"] for row in scoped
                                                if isinstance(row["native_cache_tokens"], (int, float))),
                "usage_cached_prompt_tokens_sum": sum(row["usage_cached_prompt_tokens"] for row in scoped
                                                      if isinstance(row["usage_cached_prompt_tokens"], int)),
                "fast_settings": self.binding(model)["settings"],
            })
        atomic_json(self.study / "results.json", {"schema_version": 1,
                    "claim": "custom small test; not a public benchmark", "rows": case_rows})
        atomic_json(self.study / "summary.json", {"schema_version": 1,
                    "kind": STUDY_KIND, "created_at": now(),
                    "claim": "custom small test; not a public benchmark", "manifest_sha256": self.manifest_sha,
                    "model_plan_sha256": self.plan_sha, "rows": model_rows})
        for path, rows in ((self.study / "results.csv", case_rows), (self.study / "summary.csv", model_rows)):
            output = io.StringIO(newline=""); writer = csv.DictWriter(output, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows); atomic_bytes(path, output.getvalue().encode())
        self.consolidate_guided()

    def consolidate_guided(self) -> None:
        if self.guided_plan is None or self.guided_plan_sha is None:
            return
        rows: list[dict[str, Any]] = []
        baseline_sequence = {case["id"]: sequence
                             for sequence, case in enumerate(self.cases)}
        for model in self.models:
            for guided_sequence, case in enumerate(self.guided_cases(),
                                                   start=len(self.cases)):
                baseline = read_json(self.case_dir(model, case) / "result.json")
                guided = read_json(self.case_dir(model, case, guided=True) / "result.json")
                row: dict[str, Any] = {
                    "model_key": model["key"], "case_id": case["id"],
                    "suite": case["suite"],
                    "baseline_request_sequence": baseline_sequence[case["id"]],
                    "baseline_cold_first_request": baseline_sequence[case["id"]] == 0,
                    "guided_request_sequence": guided_sequence,
                    "guided_cold_first_request": False,
                    "manual_review_required": case["suite"] == "summary",
                    "comparison_scope": "same preselected case; free generation versus public-contract JSON Schema",
                }
                for prefix, result in (("baseline", baseline), ("guided", guided)):
                    evaluation = result["evaluation"]; usage = result.get("usage", {})
                    timings = result.get("native_timings", {})
                    prompt_details = usage.get("prompt_tokens_details", {}) if isinstance(usage, dict) else {}
                    completion_details = usage.get("completion_tokens_details", {}) if isinstance(usage, dict) else {}
                    row.update({
                        f"{prefix}_status": result["status"],
                        f"{prefix}_json_valid": evaluation["json_valid"],
                        f"{prefix}_semantic_pass": evaluation["semantic_pass"],
                        f"{prefix}_ttft_first_seconds": result.get("ttft_first_seconds"),
                        f"{prefix}_ttft_final_seconds": result.get("ttft_final_seconds"),
                        f"{prefix}_wall_seconds": result.get("wall_seconds"),
                        f"{prefix}_prompt_tokens": usage.get("prompt_tokens"),
                        f"{prefix}_completion_tokens": usage.get("completion_tokens"),
                        f"{prefix}_native_cache_tokens": timings.get("cache_n"),
                        f"{prefix}_usage_cached_prompt_tokens": (
                            prompt_details.get("cached_tokens")
                            if isinstance(prompt_details, dict) else None),
                        f"{prefix}_reasoning_characters": len(result.get("reasoning_text", "")),
                        f"{prefix}_reasoning_tokens": (
                            completion_details.get("reasoning_tokens")
                            if isinstance(completion_details, dict) else None),
                        f"{prefix}_native_generation_tokens_per_second": timings.get("predicted_per_second"),
                    })
                rows.append(row)
        document = {
            "schema_version": 1,
            "kind": "tachyon_guided_public_schema_matched_probe",
            "created_at": now(),
            "claim": ("six preselected matched cases only; public contract schemas encode no correct answer; "
                      "summary prose still requires manual review"),
            "guided_probe_plan_sha256": self.guided_plan_sha,
            "case_ids": list(GUIDED_CASE_IDS),
            "baseline_case_count_per_model": 6,
            "guided_case_count_per_model": 6,
            "rows": rows,
        }
        atomic_json(self.study / "guided-public-schema-probe-comparison.json", document)
        output = io.StringIO(newline=""); writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
        atomic_bytes(self.study / "guided-public-schema-probe-comparison.csv",
                     output.getvalue().encode("utf-8"))

    def run(self) -> int:
        self.acquire_lock()
        for signum in (signal.SIGINT, signal.SIGTERM): signal.signal(signum, self.handle_signal)
        if self.status_path.is_file():
            existing = read_json(self.status_path)
            if isinstance(existing, dict) and existing.get("status") == "complete":
                requested_guided_sha = (sha256(self.guided_plan_path)
                                        if self.guided_plan_path is not None else None)
                requested_handoff_sha = (sha256(self.handoff_receipt_path)
                                         if self.handoff_receipt_path is not None else None)
                if existing.get("guided_probe_plan_sha256") != requested_guided_sha \
                        or existing.get("handoff_receipt_sha256") != requested_handoff_sha:
                    raise RoleStudyError("completed study input binding differs")
                self.event("study_already_complete"); return 0
        failure: BaseException | None = None
        try:
            handoff = self.wait_for_gpu(); self.initialize(handoff)
            self.event("study_started", manifest_sha256=self.manifest_sha)
            for index, model in enumerate(self.models):
                self.save(status="running", current_model=model["key"]); self.verify_model(model); self.run_model(model, index)
            self.consolidate()
        except BaseException as exc: failure = exc
        finally:
            if self.models:
                saved_interrupt = self.interrupted
                self.interrupted = None
                try:
                    self.save(status="restoring_default", current_model=None); self.restore_default()
                except BaseException as restore_exc:
                    failure = restore_exc if failure is None else RoleStudyError(f"{failure}; restore also failed: {restore_exc}")
                finally:
                    self.interrupted = saved_interrupt
        if failure is not None:
            if not self.state:
                self.state = {"schema_version": 1, "kind": STUDY_KIND,
                              "created_at": now(), "models": {}}
            self.save(status="failed", error=str(failure), finished_at=now()); self.event("study_failed", error=str(failure))
            print(f"tachyon_role_study: {failure}", file=sys.stderr); return 1
        self.save(status="complete", error=None, current_model=None, finished_at=now()); self.event("study_complete"); return 0


def parse_args(arguments: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=DEFAULT_STUDY)
    parser.add_argument("--model-plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--guided-probe-plan", type=Path)
    parser.add_argument("--handoff-receipt", type=Path)
    parser.add_argument("--remote", default="akovanda@tachyon"); parser.add_argument("--endpoint", default=ENDPOINT)
    parser.add_argument("--wait-timeout", type=float, default=86_400); parser.add_argument("--poll", type=float, default=10)
    parser.add_argument("--server-timeout", type=float, default=600); parser.add_argument("--request-timeout", type=float, default=180)
    args = parser.parse_args(arguments)
    for name in ("wait_timeout", "poll", "server_timeout", "request_timeout"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive and finite")
    return args


def main(arguments: Iterable[str] = sys.argv[1:]) -> int:
    return Controller(parse_args(arguments)).run()


if __name__ == "__main__":
    raise SystemExit(main())
