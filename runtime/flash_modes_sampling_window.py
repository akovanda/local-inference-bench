#!/usr/bin/env python3
"""Run the accepted Flash trained-NONE and selected T1 coding window.

``--check-only`` is intentionally offline.  It validates immutable inputs and
the parent-authored acceptance document but performs no socket, process,
Docker, NVML, or model operation.  ``--execute`` is the only mode which may
enter the prepared GPU window.

The controller never publishes the main-run RELEASE marker and never emits
model text.  Result receipts contain hashes and scalar metadata only.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import http.client
import json
import math
import os
from pathlib import Path
import re
import signal
import socket
import stat
import subprocess
import sys
import time
from typing import Any, Callable, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    # Direct-path invocation sets sys.path[0] to runtime/, while execution
    # imports frozen helpers through the runtime package.
    sys.path.insert(0, str(ROOT))
RESULTS = ROOT / "results"
RUNTIME = ROOT / "runtime"
TEST_PATH = ROOT / "tests/test_flash_modes_sampling_window.py"
PREFLIGHT_PATH = RESULTS / "flash-modes-sampling-controller-preflight-v2.json"
PLAN_PATH = RESULTS / "flash-coder-modes-sampling-execution-preparation-v2.json"
PLAN_SHA256 = "e0d83249c1e1dce5c406a04a835e4f93e942357b6683f8fb7ce9d4d32f573c69"
PLAN_SIZE = 16_822
SOURCE_CONFIG = RUNTIME / "specialists-final72k64k-flash-next-coder-iq1_m-72k.json"
SOURCE_CONFIG_SHA256 = "e7daaebea57d207795b1f10bc8177333e56cd04f47cd1224f06422b986abe0f1"
SOURCE_CONFIG_SIZE = 1_350
SETUP_PLAN = RUNTIME / "specialists-flash-none-code-72k64k-setup-v1-plan.json"
SETUP_PLAN_SHA256 = "a6e35860f1ea898da323705a90233a4098a4ae10ce61cb2c2f3f7ad8e99faba6"
MAIN_TAG = "specialists-final72k64k-flash-next-coder-iq1_m-strict-72k"
RUN_PREFIX = "specialists-final72k64k"
MODEL_KEY = "flash-next-coder-iq1_m"
TAG = "specialists-flash-modes-sampling-72k64k-v2"
MODEL = "qwen3.8-flash-next-coder-iq1_m"
NONE_PROFILE = "flash-next-coder-trained-none-72k-64k"
T1_PROFILE = "flash-next-coder-publisher-thinking-t1-72k-64k"
CONTEXT = 73_728
OUTPUT_CAP = 65_536
REQUEST_TIMEOUT = 21_600.0
WINDOW_SECONDS = 86_400.0
RESTORATION_RESERVE_SECONDS = 180.0
PORTS = (18_080, 18_081)
TAIL_ENVIRONMENT = {"STRATA_ROUTE_TAIL_SKIP": "0"}
EXPECTED_ENGINE = "/external/storage/moe-bench/strata-build/strata"
MAX_JSON_BYTES = 32 * 1024 * 1024
MAX_JSONL_BYTES = 128 * 1024 * 1024
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
)

CODE_CASES = (
    "code_fresh_batch_module",
    "code_multifunction_ledger_bugfix",
    "code_edge_reading_pivot",
)
CONTINUATION_CASES = CODE_CASES[1:]
MAIN_CASES = {
    "quick2": ("quick_generation", "quick_json_extract"),
    "hard4": (
        "hard_transaction_ledger",
        "hard_weighted_intervals",
        "hard_temporal_kv",
        "hard_dependency_constraints",
    ),
    "code3": CODE_CASES,
    "prose3": (
        "prose_grounded_public_briefing",
        "prose_constrained_last_tram",
        "prose_structured_orion_edit",
    ),
}
MAIN_ARTIFACT_PATHS = {
    "quick2": RESULTS / f"{MAIN_TAG}-01-quick_medium_65536.jsonl",
    "hard4": RESULTS / f"{MAIN_TAG}-02-hard_medium_65536.jsonl",
    "code3": RESULTS / f"{MAIN_TAG}-specialist-code.jsonl",
    "prose3": RESULTS / f"{MAIN_TAG}-specialist-prose.jsonl",
}
BOUNDARY_DIR = RESULTS / f"{RUN_PREFIX}-boundaries"
MAIN_READY = BOUNDARY_DIR / f"{RUN_PREFIX}-{MODEL_KEY}-ready.json"
MAIN_RELEASE = BOUNDARY_DIR / f"{RUN_PREFIX}-{MODEL_KEY}-release.json"
MAIN_STUDY = RESULTS / f"{MAIN_TAG}-study-trial.json"

CASE_IDENTITIES = {
    "code_fresh_batch_module": {
        "case_sha256": "a8f2cc8cd45f63bff5fd2cd8ca33864004aca72a15088a9d320f239cc4ca5011",
        "prompt_sha256": "624eae1e68aca9c8510d7fbc8122b094a2222551188e7a17a24d626ced1f9998",
    },
    "code_multifunction_ledger_bugfix": {
        "case_sha256": "cb0f6864473c7b8a6ead13616ff7dc8ed73bfe76d7b0ece1a93b140a16ffac4e",
        "prompt_sha256": "364396968612a45d5572adb9f7b9139a32d3fef764af125a386df9ac7640711a",
    },
    "code_edge_reading_pivot": {
        "case_sha256": "f437e82e516c0495981e87c9b591bc4e753b0d6833297fe66259d65b0a6e0d40",
        "prompt_sha256": "4d77e4d471bf14e9b86ee0f3b4bed4f7e5de8be6f94ea2cf178f1b2652dcc793",
    },
}

STATIC_PINS = {
    RUNTIME / "model_session.py": "968b2ce078975de23dd99670c2141995f460f36ae86490ce33153e7f1ae9c43c",
    RUNTIME / "gpu_window.py": "7982aa5906fd05f6cb61c9e2b54467724755472e7d17feb485a687a142ca6898",
    RUNTIME / "strata_trial.py": "1a3d802469ae6dd5253c47bd7f0f40d8a6e03523364e867397d14b31a56dc381",
    RUNTIME / "specialist_bench.py": "fff82d7265f38af82f8b495b1da256e27e3e87ef326591ef63daa5703f76f7eb",
    RUNTIME / "specialist_cases.py": "2486edf32007b35e24d1838631a267eb5ee3654afd4aad41dae0bb87020ffc43",
    ROOT / "bench.py": "48fded3dea300e70c986c7bd75d35e909f24fb0d5040cf26369b030a8abccb7f",
    SOURCE_CONFIG: SOURCE_CONFIG_SHA256,
    SETUP_PLAN: SETUP_PLAN_SHA256,
    RESULTS / "specialists-flash-code-mode-diagnostic-plan-v2.json":
        "5ddd57284d9b11953b43d78fce60aa7ea68e41374ab3b00d63217f563015cab1",
    RESULTS / "specialists-flash-none-code-generation-accounting-v1.json":
        "028e35328deaa60eeadc042e33a39182e7a667d083c3758ba8fa9e2c70dd565f",
    RESULTS / "flash-coder-publisher-thinking-sampler-source-audit-v1.json":
        "8e225234e9385731d714345b9427902f1ebe40aff36dfb46c2a84956dc6b9eb2",
    RESULTS / "flash-none-code-owned-window-execution-preparation-v1.json":
        "acf090621d8c6a976551f22916e1c1c80ed0be8fc7030487a701f38087954a40",
    RUNTIME / "study_resource_report.py":
        "99181179d408e333735100fd56be667d457e53ad4b2b89c5150d1ecd6da7a006",
    ROOT / "vendor/strata/serve/server.py":
        "9eb13d12dc3ed0f6c94d20ac1c0018efe1e00c331deae76bd0fb9f420e005232",
    PLAN_PATH: PLAN_SHA256,
}


class ControllerError(RuntimeError):
    """An immutable or runtime safety invariant was not established."""


class ControllerInterrupted(BaseException):
    def __init__(self, signum: int) -> None:
        super().__init__(f"signal {signum}")
        self.signum = signum


class SelectedWorkDeferred(ControllerError):
    """Selected work could not retain its full allowance and cleanup reserve."""

    def __init__(self, stage: str, pending_cases: Sequence[str], reason: str) -> None:
        super().__init__(reason)
        self.stage = stage
        self.pending_cases = tuple(pending_cases)


@dataclass(frozen=True)
class Binding:
    path: Path
    sha256: str
    size: int

    def as_dict(self) -> dict[str, Any]:
        return {"path": str(self.path), "sha256": self.sha256, "size": self.size}


@dataclass(frozen=True)
class JsonlRow:
    line: int
    value: Mapping[str, Any]
    raw_sha256: str
    canonical_sha256: str


@dataclass(frozen=True)
class Acceptance:
    source: Binding
    document: Mapping[str, Any]
    ready: Binding
    study: Binding
    main_run_id: str
    main_rows: Mapping[str, tuple[JsonlRow, ...]]


@dataclass(frozen=True)
class HoldingSession:
    run_id: str
    api_pid: int
    model: str
    context: int
    session_started_line: int
    server_started_line: int


@dataclass(frozen=True)
class ProcessIdentity:
    pid: int
    parent_pid: int
    start_ticks: str
    state: str
    exe: str


@dataclass
class SignalLatch:
    received: int | None = None

    def handle(self, signum: int, _frame: Any) -> None:
        self.received = signum

    def check(self) -> None:
        if self.received is not None:
            raise ControllerInterrupted(self.received)


@dataclass
class BudgetLedger:
    maximum: int = 7
    reserved: dict[str, int] = field(default_factory=dict)
    launched: dict[str, bool] = field(default_factory=dict)
    observations: dict[str, dict[str, Any]] = field(default_factory=dict)

    def reserve(self, stage: str, slots: int) -> None:
        if stage in self.reserved:
            raise ControllerError(f"generation stage was reserved twice: {stage}")
        if isinstance(slots, bool) or not isinstance(slots, int) or slots < 1:
            raise ControllerError("generation reservation must be a positive integer")
        if sum(self.reserved.values()) + slots > self.maximum:
            raise ControllerError("generation request reservation exceeds the window maximum")
        self.reserved[stage] = slots
        self.launched[stage] = False

    def mark_launched(self, stage: str) -> None:
        if stage not in self.reserved:
            raise ControllerError(f"unreserved generation stage launched: {stage}")
        if self.launched[stage]:
            raise ControllerError(f"generation stage launched twice: {stage}")
        self.launched[stage] = True

    def observe(self, stage: str, rows: int | None, *, helper_terminal: bool) -> None:
        if stage not in self.reserved:
            raise ControllerError(f"unreserved generation stage observed: {stage}")
        slots = self.reserved[stage]
        if rows is not None and (isinstance(rows, bool) or not 0 <= rows <= slots):
            raise ControllerError(f"observed row count exceeds {stage} reservation")
        launched = self.launched[stage]
        lower = rows if isinstance(rows, int) else 0
        if not launched:
            exact: int | None = 0 if rows in (0, None) else None
            upper = 0 if exact == 0 else lower
        elif helper_terminal and isinstance(rows, int):
            exact = rows
            upper = rows
        else:
            exact = None
            # Helpers are sequential.  At most one request may have started
            # without an append-stable result row; later cases are unexecuted.
            upper = min(slots, lower + 1)
        self.observations[stage] = {
            "allocated_request_slots": slots,
            "launch_attempted": launched,
            "observed_result_rows": rows,
            "attempts_lower_bound": lower,
            "attempts_upper_bound": upper,
            "exact_attempts": exact,
            "has_uncertain_started_unrecorded_call": exact is None and upper > lower,
        }

    def summary(self) -> dict[str, Any]:
        lower = sum(item["attempts_lower_bound"] for item in self.observations.values())
        upper = sum(item["attempts_upper_bound"] for item in self.observations.values())
        exact_values = [item["exact_attempts"] for item in self.observations.values()]
        exact = sum(exact_values) if exact_values and all(v is not None for v in exact_values) else None
        return {
            "maximum_request_slots": self.maximum,
            "reserved_request_slots": sum(self.reserved.values()),
            "observed_attempts_lower_bound": lower,
            "possible_attempts_upper_bound": upper,
            "exact_attempts": exact,
            "stages": copy.deepcopy(self.observations),
            "reserved_slots_are_not_reported_as_actual_attempts": True,
            "budget_violation": upper > self.maximum,
        }


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _regular(path: Path, description: str, maximum: int | None = None) -> os.stat_result:
    try:
        details = path.lstat()
    except OSError as exc:
        raise ControllerError(f"{description} is missing or inaccessible: {path}") from exc
    if path.is_symlink() or not stat.S_ISREG(details.st_mode):
        raise ControllerError(f"{description} must be a regular non-symlink file: {path}")
    if maximum is not None and details.st_size > maximum:
        raise ControllerError(f"{description} exceeds its size bound: {path}")
    return details


def _strict_json_bytes(payload: bytes, description: str) -> Mapping[str, Any]:
    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"duplicate JSON key: {key}")
            value[key] = item
        return value

    try:
        value = json.loads(payload.decode("utf-8"), object_pairs_hook=object_pairs)
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ControllerError(f"{description} is not strict JSON") from exc
    if not isinstance(value, dict):
        raise ControllerError(f"{description} must be a JSON object")
    return value


def load_json(path: Path, description: str) -> tuple[Mapping[str, Any], Binding]:
    details = _regular(path, description, MAX_JSON_BYTES)
    payload = path.read_bytes()
    return _strict_json_bytes(payload, description), Binding(path.resolve(), hashlib.sha256(payload).hexdigest(), details.st_size)


def validate_binding(
    value: Any,
    description: str,
    *,
    expected_path: Path | None = None,
    expected_sha256: str | None = None,
    maximum: int | None = None,
) -> Binding:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256", "size"}:
        raise ControllerError(f"{description} binding must contain exactly path, sha256, and size")
    raw_path, digest, size = value.get("path"), value.get("sha256"), value.get("size")
    if not isinstance(raw_path, str) or not raw_path or not Path(raw_path).is_absolute():
        raise ControllerError(f"{description}.path must be a nonempty absolute path")
    path = Path(raw_path).resolve()
    if expected_path is not None and path != expected_path.resolve():
        raise ControllerError(f"{description} path differs from the required path")
    if not isinstance(digest, str) or SHA256_RE.fullmatch(digest) is None:
        raise ControllerError(f"{description}.sha256 is invalid")
    if expected_sha256 is not None and digest != expected_sha256:
        raise ControllerError(f"{description} does not declare the required SHA-256")
    if isinstance(size, bool) or not isinstance(size, int) or size < 1:
        raise ControllerError(f"{description}.size must be a positive integer")
    details = _regular(path, description, maximum)
    if details.st_size != size or sha256_file(path) != digest:
        raise ControllerError(f"{description} binding does not match the file")
    return Binding(path, digest, size)


def read_jsonl(path: Path, description: str) -> tuple[tuple[JsonlRow, ...], Binding]:
    details = _regular(path, description, MAX_JSONL_BYTES)
    snapshot = path.read_bytes()
    rows: list[JsonlRow] = []
    for line_number, raw in enumerate(snapshot.splitlines(), 1):
        if not raw.strip():
            continue
        value = _strict_json_bytes(raw, f"{description} line {line_number}")
        rows.append(JsonlRow(
            line=line_number,
            value=value,
            raw_sha256=hashlib.sha256(raw).hexdigest(),
            canonical_sha256=hashlib.sha256(canonical_bytes(value)).hexdigest(),
        ))
    return tuple(rows), Binding(path.resolve(), hashlib.sha256(snapshot).hexdigest(), details.st_size)


def validate_static_sources() -> dict[str, str]:
    observed: dict[str, str] = {}
    for path, expected in STATIC_PINS.items():
        _regular(path, "pinned controller input")
        actual = sha256_file(path)
        if actual != expected:
            raise ControllerError(f"pinned source SHA-256 changed: {path}")
        observed[str(path)] = actual
    if SOURCE_CONFIG.stat().st_size != SOURCE_CONFIG_SIZE or PLAN_PATH.stat().st_size != PLAN_SIZE:
        raise ControllerError("pinned source size changed")
    return observed


def _parse_time(value: Any, description: str) -> datetime:
    if not isinstance(value, str):
        raise ControllerError(f"{description} must be an RFC3339 timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ControllerError(f"{description} must be an RFC3339 timestamp") from exc
    if parsed.tzinfo is None:
        raise ControllerError(f"{description} must include a timezone")
    return parsed.astimezone(timezone.utc)


def _is_number(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(float(value))


def _terminal(row: Mapping[str, Any]) -> bool:
    if isinstance(row.get("error"), Mapping):
        return True
    response = row.get("response")
    return bool(isinstance(response, Mapping) and response.get("stream_complete") is True)


def _validate_row_pin(pin: Any, row: JsonlRow, case_id: str, description: str) -> None:
    expected = {"case_id", "source_line", "source_raw_line_sha256", "source_row_canonical_sha256"}
    if not isinstance(pin, Mapping) or set(pin) != expected:
        raise ControllerError(f"{description} row pin does not exactly match schema version 1")
    if pin.get("case_id") != case_id or row.value.get("case_id") != case_id:
        raise ControllerError(f"{description} case identity differs")
    if pin.get("source_line") != row.line:
        raise ControllerError(f"{description} source line differs")
    if pin.get("source_raw_line_sha256") != row.raw_sha256:
        raise ControllerError(f"{description} raw row hash differs")
    if pin.get("source_row_canonical_sha256") != row.canonical_sha256:
        raise ControllerError(f"{description} canonical row hash differs")


def _validate_main_specialist_code(row: Mapping[str, Any], case_id: str) -> None:
    identity = CASE_IDENTITIES[case_id]
    request = row.get("request")
    if not isinstance(request, Mapping):
        raise ControllerError(f"main code row {case_id} lacks request controls")
    required = {
        "model": MODEL,
        "thinking": "medium",
        "temperature": 0,
        "seed": 0,
        "context_max_tokens": CONTEXT,
        "output_token_limit": OUTPUT_CAP,
        "timeout_seconds": REQUEST_TIMEOUT,
        "prompt_sha256": identity["prompt_sha256"],
    }
    for key, expected in required.items():
        actual = request.get(key)
        if _is_number(expected) and _is_number(actual):
            if float(actual) != float(expected):
                raise ControllerError(f"main code row {case_id} has mismatched {key}")
        elif actual != expected:
            raise ControllerError(f"main code row {case_id} has mismatched {key}")
    if row.get("case_sha256") != identity["case_sha256"]:
        raise ControllerError(f"main code row {case_id} has mismatched case hash")
    if row.get("case_version") != "2026-10-19.specialist.2":
        raise ControllerError(f"main code row {case_id} has mismatched case version")
    if (
        row.get("tag") != f"{MAIN_TAG}-code"
        or row.get("suite") != "specialist_code"
        or row.get("case_group") != "code"
        or row.get("repeat_index") != 0
        or row.get("case_source_sha256") != STATIC_PINS[RUNTIME / "specialist_cases.py"]
    ):
        raise ControllerError(f"main code row {case_id} has mismatched run/case-source identity")


def _validate_main_artifact(
    name: str, value: Any,
) -> tuple[Binding, tuple[JsonlRow, ...]]:
    if not isinstance(value, Mapping) or set(value) != {"source", "rows"}:
        raise ControllerError(f"main artifact {name} must contain exactly source and rows")
    source = validate_binding(
        value.get("source"), f"main {name} source", expected_path=MAIN_ARTIFACT_PATHS[name],
        maximum=MAX_JSONL_BYTES,
    )
    rows, observed = read_jsonl(source.path, f"main {name} source")
    if observed.sha256 != source.sha256 or observed.size != source.size:
        raise ControllerError(f"main {name} changed during validation")
    expected_cases = MAIN_CASES[name]
    pins = value.get("rows")
    if not isinstance(pins, list) or len(pins) != len(expected_cases) or len(rows) != len(expected_cases):
        raise ControllerError(f"main {name} must contain exactly {len(expected_cases)} rows")
    for pin, row, case_id in zip(pins, rows, expected_cases):
        _validate_row_pin(pin, row, case_id, f"main {name}")
        if not _terminal(row.value):
            raise ControllerError(f"main {name} case {case_id} is not terminal")
        if name == "code3":
            _validate_main_specialist_code(row.value, case_id)
    return source, rows


def _validate_study(
    document: Mapping[str, Any], *, run_id: str, binding: Binding,
    artifact_bindings: Mapping[str, Binding],
) -> None:
    if document.get("schema_version") != 1 or document.get("tag") != MAIN_TAG:
        raise ControllerError("main study identity differs")
    if document.get("runtime_backend") != "strata":
        raise ControllerError("main study runtime backend is not strata")
    if document.get("status") not in {"complete", "completed_with_evaluation_failures"}:
        raise ControllerError("main study is not terminal at an accepted status")
    source = document.get("source_config")
    if not isinstance(source, Mapping) or source.get("path") != str(SOURCE_CONFIG) or source.get("sha256") != SOURCE_CONFIG_SHA256:
        raise ControllerError("main study source config differs")
    session = document.get("session")
    if not isinstance(session, Mapping) or session.get("run_id") != run_id:
        raise ControllerError("main study run identity differs")
    if session.get("model_name") != MODEL or session.get("context") != CONTEXT:
        raise ControllerError("main study held the wrong model or context")
    cleanup = document.get("cleanup")
    nested = cleanup.get("model_session_cleanup") if isinstance(cleanup, Mapping) else None
    if not (
        isinstance(cleanup, Mapping)
        and cleanup.get("outer_exited") is True
        and cleanup.get("orchestrator_sigkill_sent") is False
        and isinstance(nested, Mapping)
        and nested.get("stopped") is True
        and nested.get("kill_sent") is False
    ):
        raise ControllerError("main study lacks clean owned lifecycle evidence")
    restoration = document.get("restoration")
    if not (
        isinstance(restoration, Mapping)
        and restoration.get("original_running") is True
        and restoration.get("original_paused") is False
        and restoration.get("owned_loopback_port_clear") is True
    ):
        raise ControllerError("main study lacks restoration evidence")
    if document.get("problem") is not None:
        raise ControllerError("main study records an infrastructure problem")
    if document.get("session_report_status") not in {"passed", "completed_with_step_failures"}:
        raise ControllerError("main study session-report status is not terminal-safe")
    helper_sources = document.get("helper_sources")
    if not isinstance(helper_sources, Mapping):
        raise ControllerError("main study lacks helper source bindings")
    for helper in (RUNTIME / "gpu_window.py", RUNTIME / "model_session.py", RUNTIME / "specialist_bench.py"):
        if helper_sources.get(str(helper)) != STATIC_PINS[helper]:
            raise ControllerError(f"main study helper source differs: {helper}")
    controls = document.get("controls")
    code_controls = controls.get("code") if isinstance(controls, Mapping) else None
    prose_controls = controls.get("prose") if isinstance(controls, Mapping) else None
    if not (
        isinstance(controls, Mapping)
        and controls.get("code_thinking") == "medium"
        and controls.get("prose_thinking") == "none"
        and _is_number(controls.get("max_seconds"))
        and float(controls["max_seconds"]) == WINDOW_SECONDS
        and _is_number(controls.get("startup_timeout"))
        and float(controls["startup_timeout"]) == 480.0
        and _is_number(controls.get("wait_for_idle"))
        and float(controls["wait_for_idle"]) == 900.0
        and isinstance(code_controls, Mapping)
        and code_controls.get("output_cap") == OUTPUT_CAP
        and _is_number(code_controls.get("temperature"))
        and float(code_controls["temperature"]) == 0.0
        and code_controls.get("seed") == 0
        and _is_number(code_controls.get("timeout"))
        and float(code_controls["timeout"]) == REQUEST_TIMEOUT
        and isinstance(prose_controls, Mapping)
        and prose_controls.get("output_cap") == OUTPUT_CAP
        and _is_number(prose_controls.get("temperature"))
        and float(prose_controls["temperature"]) == 0.7
        and prose_controls.get("seed") == 0
        and _is_number(prose_controls.get("timeout"))
        and float(prose_controls["timeout"]) == REQUEST_TIMEOUT
    ):
        raise ControllerError("main study controls differ from the canonical main12 protocol")
    plan = document.get("plan")
    steps = plan.get("steps") if isinstance(plan, Mapping) else None
    if not isinstance(steps, list) or len(steps) != 2:
        raise ControllerError("main study plan does not contain canonical quick/hard stages")
    for step, suite, step_id in zip(steps, ("quick", "hard"), ("quick_medium_65536", "hard_medium_65536")):
        if not (
            isinstance(step, Mapping)
            and step.get("id") == step_id
            and step.get("suite") == suite
            and step.get("thinking") == "medium"
            and step.get("temperature") == 0
            and step.get("seed") == 0
            and step.get("output_tokens") == OUTPUT_CAP
            and _is_number(step.get("request_timeout"))
            and float(step["request_timeout"]) == REQUEST_TIMEOUT
            and step.get("repeat") == 1
        ):
            raise ControllerError(f"main study {suite} plan controls differ")
    commands = document.get("commands")
    if not isinstance(commands, list):
        raise ControllerError("main study commands are absent")
    for kind in ("gpu_window", "specialist_code", "specialist_prose"):
        matches = [item for item in commands if isinstance(item, Mapping) and item.get("kind") == kind]
        if len(matches) != 1 or matches[0].get("returncode") not in {0, 1}:
            raise ControllerError(f"main study command {kind} is missing or nonterminal")
    artifacts = document.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ControllerError("main study lacks artifact inventory")
    study_names = {
        "quick2": "plan_01_quick_medium_65536",
        "hard4": "plan_02_hard_medium_65536",
        "code3": "code",
        "prose3": "prose",
    }
    for name, study_name in study_names.items():
        item = artifacts.get(study_name)
        if not isinstance(item, Mapping) or item.get("exists") is not True:
            raise ControllerError(f"main study does not inventory {name}")
        if Path(str(item.get("path"))).resolve() != artifact_bindings[name].path:
            raise ControllerError(f"main study {name} path differs")
        if item.get("size") != artifact_bindings[name].size:
            raise ControllerError(f"main study {name} size differs")


def _validate_ready(
    document: Mapping[str, Any], binding: Binding, *, study: Binding,
) -> datetime:
    expected_keys = {"schema_version", "kind", "model_key", "published_at", "run_prefix", "study"}
    if set(document) != expected_keys:
        raise ControllerError("main boundary READY fields do not exactly match model_boundary_gate schema version 1")
    if (
        document.get("schema_version") != 1
        or document.get("kind") != "model_boundary_ready"
        or document.get("run_prefix") != RUN_PREFIX
        or document.get("model_key") != MODEL_KEY
    ):
        raise ControllerError("main boundary READY schema or identity differs")
    published = _parse_time(document.get("published_at"), "main boundary READY published_at")
    expected_study = {"path": str(study.path), "sha256": study.sha256}
    if document.get("study") != expected_study:
        raise ControllerError("main READY study binding differs")
    study_document, _ = load_json(study.path, "main study for READY chronology")
    finished = _parse_time(study_document.get("finished_at"), "main study finished_at")
    if published < finished:
        raise ControllerError("main boundary READY predates the bound study completion")
    return published


def _validate_restoration_attestation(value: Any) -> datetime:
    expected_keys = {
        "kind", "observed_at", "original_running", "original_paused",
        "original_api_healthy", "ports_clear", "no_foreign_gpu_process",
        "release_absent",
    }
    if not isinstance(value, Mapping) or set(value) != expected_keys:
        raise ControllerError("parent live-restoration attestation fields differ")
    if value.get("kind") != "flash_modes_sampling_parent_restoration_attestation":
        raise ControllerError("parent live-restoration attestation kind differs")
    observed_at = _parse_time(value.get("observed_at"), "parent restoration observed_at")
    if not (
        value.get("original_running") is True
        and value.get("original_paused") is False
        and value.get("original_api_healthy") is True
        and value.get("ports_clear") == {"18080": True, "18081": True}
        and value.get("no_foreign_gpu_process") is True
        and value.get("release_absent") is True
    ):
        raise ControllerError("parent live-restoration attestation is not safe")
    return observed_at


def validate_acceptance(path: Path) -> Acceptance:
    validate_static_sources()
    document, source = load_json(path.resolve(), "parent execution acceptance")
    expected_keys = {"schema_version", "kind", "accepted", "created_at", "review", "main"}
    if set(document) != expected_keys:
        raise ControllerError("parent acceptance fields do not exactly match schema version 1")
    if document.get("schema_version") != 1 or document.get("kind") != "flash_modes_sampling_parent_execution_acceptance":
        raise ControllerError("parent acceptance schema or kind differs")
    if document.get("accepted") is not True:
        raise ControllerError("parent acceptance is absent or not affirmative")
    accepted_at = _parse_time(document.get("created_at"), "parent acceptance created_at")
    review = document.get("review")
    if not isinstance(review, Mapping) or set(review) != {"plan", "controller", "tests", "preflight"}:
        raise ControllerError("parent acceptance review bindings differ")
    validate_binding(review.get("plan"), "accepted plan", expected_path=PLAN_PATH, expected_sha256=PLAN_SHA256)
    validate_binding(review.get("controller"), "accepted controller", expected_path=Path(__file__).resolve())
    validate_binding(review.get("tests"), "accepted tests", expected_path=TEST_PATH)
    validate_binding(review.get("preflight"), "accepted preflight", expected_path=PREFLIGHT_PATH)

    main = document.get("main")
    main_keys = {
        "tag", "model", "model_session_run_id", "source_config", "ready", "study",
        "artifacts", "release_path", "restoration_attestation",
    }
    if not isinstance(main, Mapping) or set(main) != main_keys:
        raise ControllerError("parent acceptance main fields differ")
    run_id = main.get("model_session_run_id")
    if main.get("tag") != MAIN_TAG or main.get("model") != MODEL:
        raise ControllerError("accepted main model or tag differs")
    if not isinstance(run_id, str) or UUID_RE.fullmatch(run_id) is None:
        raise ControllerError("accepted main run ID is not a UUID")
    validate_binding(
        main.get("source_config"), "accepted source config", expected_path=SOURCE_CONFIG,
        expected_sha256=SOURCE_CONFIG_SHA256,
    )
    if main.get("release_path") != str(MAIN_RELEASE):
        raise ControllerError("accepted main RELEASE path differs")
    if os.path.lexists(MAIN_RELEASE):
        raise ControllerError("main RELEASE is present; sampling window must precede release")
    restoration_at = _validate_restoration_attestation(main.get("restoration_attestation"))

    raw_artifacts = main.get("artifacts")
    if not isinstance(raw_artifacts, Mapping) or set(raw_artifacts) != set(MAIN_CASES):
        raise ControllerError("accepted main artifact set differs")
    artifact_bindings: dict[str, Binding] = {}
    artifact_rows: dict[str, tuple[JsonlRow, ...]] = {}
    for name in MAIN_CASES:
        artifact_bindings[name], artifact_rows[name] = _validate_main_artifact(name, raw_artifacts[name])

    study = validate_binding(
        main.get("study"), "accepted main study", expected_path=MAIN_STUDY,
        maximum=MAX_JSON_BYTES,
    )
    study_document, observed_study = load_json(study.path, "accepted main study")
    if observed_study != study:
        raise ControllerError("main study changed during acceptance validation")
    _validate_study(study_document, run_id=run_id, binding=study, artifact_bindings=artifact_bindings)
    ready = validate_binding(
        main.get("ready"), "accepted main READY", expected_path=MAIN_READY,
        maximum=MAX_JSON_BYTES,
    )
    ready_document, observed_ready = load_json(ready.path, "accepted main READY")
    if observed_ready != ready:
        raise ControllerError("main READY changed during acceptance validation")
    ready_at = _validate_ready(ready_document, ready, study=study)
    if restoration_at < ready_at or accepted_at < restoration_at:
        raise ControllerError("parent READY/restoration/acceptance chronology differs")
    return Acceptance(source, document, ready, study, run_id, artifact_rows)


def controller_paths(results: Path = RESULTS) -> dict[str, Path]:
    return {
        "private_config": results / f"{TAG}-config.json",
        "shared_settings": results / f"{TAG}-config.shared-settings.json",
        "control": results / f"{TAG}-control.json",
        "runtime_events": results / f"{TAG}-runtime-events.jsonl",
        "engine_log": results / f"{TAG}-engine.log",
        "server_log": results / f"{TAG}-server.log",
        "session_report": results / f"{TAG}-session-report.json",
        "setup": results / f"{TAG}-01-resident_setup_json_none_256.jsonl",
        "none": results / f"{TAG}-code_none_65536.jsonl",
        "none_visible": results / f"{TAG}-code_none_65536.jsonl.texts",
        "t1": results / f"{TAG}-publisher-thinking-code.jsonl",
        "t1_visible": results / f"{TAG}-publisher-thinking-code.jsonl.texts",
        "ownership": results / f"{TAG}-gpu-ownership-v1.json",
        "controller_events": results / f"{TAG}-controller-events.jsonl",
        "receipt": results / f"{TAG}-controller-receipt-v1.json",
    }


def assert_outputs_unused(
    paths: Mapping[str, Path], *, allowed_tag_paths: Sequence[Path] = (),
) -> None:
    allowed = {path.resolve() for path in allowed_tag_paths}
    collisions = {path.resolve() for path in paths.values() if os.path.lexists(path)}
    parents = {path.parent.resolve() for path in paths.values()}
    for parent in parents:
        if parent.is_dir():
            collisions.update(
                path.resolve() for path in parent.glob(f"{TAG}*")
                if path.resolve() not in allowed
            )
    if collisions:
        raise ControllerError(
            "planned tag/output is already used: " + ", ".join(sorted(map(str, collisions)))
        )


def exact_commands(plan: Mapping[str, Any]) -> dict[str, list[str]]:
    commands = {
        "window": plan.get("window_argv"),
        "none": plan.get("mandatory_NONE_code_argv"),
        "pilot": plan.get("publisher_T1_pilot_argv"),
        "continuation": plan.get("conditional_T1_remaining_code2_argv"),
    }
    if any(not isinstance(value, list) or not all(isinstance(item, str) for item in value) for value in commands.values()):
        raise ControllerError("prepared command vectors are malformed")
    return {key: list(value) for key, value in commands.items()}


def cache_compatibility(candidate: Mapping[str, Any], reference: Mapping[str, Any]) -> dict[str, Any]:
    fields = ("native_prompt_tokens", "native_cached_prompt_tokens")
    values: dict[str, dict[str, int | None]] = {}
    for field_name in fields:
        left = candidate.get(field_name)
        right = reference.get(field_name)
        left_value = left if isinstance(left, int) and not isinstance(left, bool) and left >= 0 else None
        right_value = right if isinstance(right, int) and not isinstance(right, bool) and right >= 0 else None
        values[field_name] = {"candidate": left_value, "reference": right_value}
    known = all(item[side] is not None for item in values.values() for side in ("candidate", "reference"))
    compatible = known and all(item["candidate"] == item["reference"] for item in values.values())
    return {
        "known": known,
        "compatible": compatible,
        "comparison_rule": "exact equality of reported native prompt and cached prompt token counts",
        "values": values,
    }


def _speed_claim(candidate: Mapping[str, Any], reference: Mapping[str, Any]) -> dict[str, Any]:
    cache = cache_compatibility(candidate, reference)
    candidate_wall = candidate.get("wall_seconds")
    reference_wall = reference.get("wall_seconds")
    known_wall = _is_number(candidate_wall) and _is_number(reference_wall) and float(reference_wall) > 0 and float(candidate_wall) >= 0
    reduction = (
        (float(reference_wall) - float(candidate_wall)) / float(reference_wall)
        if known_wall else None
    )
    passes = bool(cache["compatible"] and reduction is not None and reduction >= 0.05)
    return {
        "candidate_wall_seconds": float(candidate_wall) if _is_number(candidate_wall) else None,
        "reference_wall_seconds": float(reference_wall) if _is_number(reference_wall) else None,
        "relative_wall_reduction": reduction,
        "threshold": 0.05,
        "cache": cache,
        "passes": passes,
    }


def evaluate_continuation_gate(
    *,
    pilot: Mapping[str, Any],
    main_t0: Mapping[str, Any],
    none_fresh: Mapping[str, Any],
    none_pass_count: int,
) -> dict[str, Any]:
    normal_pilot = bool(
        pilot.get("complete") is True
        and pilot.get("request_error") is False
        and pilot.get("passed") is True
    )
    main_speed = _speed_claim(pilot, main_t0)
    none_speed = _speed_claim(pilot, none_fresh)
    if not normal_pilot:
        useful = False
        useful_reason = "pilot_not_normal_complete_full_case_pass"
    elif main_t0.get("passed") is not True:
        useful = True
        useful_reason = "pilot_fixes_canonical_t0_failure"
    elif main_speed["passes"]:
        useful = True
        useful_reason = "pilot_retains_t0_pass_with_at_least_5_percent_lower_wall"
    else:
        useful = False
        useful_reason = "pilot_does_not_improve_successful_t0_with_compatible_5_percent_speed_evidence"
    if none_pass_count < 3:
        mode_condition = True
        mode_reason = "mandatory_none_code3_not_3_of_3"
    elif none_fresh.get("passed") is True and none_speed["passes"]:
        mode_condition = True
        mode_reason = "pilot_beats_successful_none_fresh_by_at_least_5_percent"
    else:
        mode_condition = False
        mode_reason = "none_is_3_of_3_and_pilot_lacks_compatible_5_percent_none_speed_win"
    return {
        "pilot_normal_complete_full_case_pass": normal_pilot,
        "useful_against_canonical_t0": useful,
        "usefulness_reason": useful_reason,
        "main_t0_speed_comparison": main_speed,
        "mode_comparison_condition": mode_condition,
        "mode_comparison_reason": mode_reason,
        "none_fresh_speed_comparison": none_speed,
        "continue_remaining_two": bool(normal_pilot and useful and mode_condition),
        "selection_threshold_is_not_statistical_significance": True,
    }


def _specialist_summary(
    row: JsonlRow, *, expected_thinking: str, expected_temperature: float,
    expected_tag: str, expected_profile: str,
) -> dict[str, Any]:
    value = row.value
    case_id = value.get("case_id")
    if case_id not in CASE_IDENTITIES:
        raise ControllerError("specialist result has an unexpected coding case")
    request = value.get("request")
    if not isinstance(request, Mapping):
        raise ControllerError(f"specialist result {case_id} lacks request controls")
    identity = CASE_IDENTITIES[case_id]
    required = {
        "endpoint": "http://127.0.0.1:18080",
        "backend": "openai",
        "model": MODEL,
        "profile": expected_profile,
        "thinking": expected_thinking,
        "temperature": expected_temperature,
        "seed": 0,
        "context_max_tokens": CONTEXT,
        "output_token_limit": OUTPUT_CAP,
        "timeout_seconds": REQUEST_TIMEOUT,
        "prompt_sha256": identity["prompt_sha256"],
        "chat_template_thinking": None,
        "reasoning_budget_tokens": None,
    }
    for key, expected in required.items():
        actual = request.get(key)
        if _is_number(expected) and _is_number(actual):
            matched = float(actual) == float(expected)
        else:
            matched = actual == expected
        if not matched:
            raise ControllerError(f"specialist result {case_id} has mismatched {key}")
    if value.get("case_sha256") != identity["case_sha256"] or value.get("case_version") != "2026-10-19.specialist.2":
        raise ControllerError(f"specialist result {case_id} has mismatched frozen identity")
    if (
        value.get("tag") != expected_tag
        or value.get("profile") != expected_profile
        or value.get("suite") != "specialist_code"
        or value.get("case_group") != "code"
        or value.get("repeat_index") != 0
        or value.get("case_source_sha256") != STATIC_PINS[RUNTIME / "specialist_cases.py"]
    ):
        raise ControllerError(f"specialist result {case_id} has mismatched run/case-source identity")
    error = value.get("error")
    response = value.get("response")
    metrics = value.get("response_metrics")
    evaluation = value.get("evaluation")
    visible = value.get("visible_text")
    if not isinstance(visible, Mapping):
        raise ControllerError(f"specialist result {case_id} lacks visible text binding")
    visible_path = Path(str(visible.get("path"))).resolve()
    visible_sha = visible.get("sha256")
    visible_size = visible.get("bytes")
    if not isinstance(visible_sha, str) or SHA256_RE.fullmatch(visible_sha) is None:
        raise ControllerError(f"specialist result {case_id} visible SHA-256 is invalid")
    if isinstance(visible_size, bool) or not isinstance(visible_size, int) or visible_size < 0:
        raise ControllerError(f"specialist result {case_id} visible byte count is invalid")
    details = _regular(visible_path, f"visible result {case_id}")
    if details.st_size != visible_size or sha256_file(visible_path) != visible_sha:
        raise ControllerError(f"specialist result {case_id} visible binding differs")
    if visible.get("reasoning_excluded") is not True:
        raise ControllerError(f"specialist result {case_id} visible binding includes reasoning")
    complete = bool(
        error is None and isinstance(response, Mapping)
        and response.get("stream_complete") is True and response.get("truncated") is not True
    )
    request_error = isinstance(error, Mapping)
    metric_map = metrics if isinstance(metrics, Mapping) else {}
    return {
        "case_id": case_id,
        "source_line": row.line,
        "source_raw_line_sha256": row.raw_sha256,
        "source_row_canonical_sha256": row.canonical_sha256,
        "case_sha256": identity["case_sha256"],
        "prompt_sha256": identity["prompt_sha256"],
        "visible_text": {
            "path": str(visible_path), "sha256": visible_sha, "bytes": visible_size,
            "reasoning_excluded": visible.get("reasoning_excluded") is True,
        },
        "complete": complete,
        "request_error": request_error,
        "request_error_type": error.get("type") if isinstance(error, Mapping) else None,
        "passed": bool(isinstance(evaluation, Mapping) and evaluation.get("passed") is True),
        "objective_outcome": value.get("objective_outcome"),
        "wall_seconds": metric_map.get("wall_seconds") if _is_number(metric_map.get("wall_seconds")) else None,
        "input_tokens": metric_map.get("input_tokens_reported") if isinstance(metric_map.get("input_tokens_reported"), int) else None,
        "output_tokens": metric_map.get("output_tokens_reported") if isinstance(metric_map.get("output_tokens_reported"), int) else None,
        "native_prompt_tokens": metric_map.get("native_prompt_tokens") if isinstance(metric_map.get("native_prompt_tokens"), int) else None,
        "native_cached_prompt_tokens": metric_map.get("native_cached_prompt_tokens") if isinstance(metric_map.get("native_cached_prompt_tokens"), int) else None,
        "native_decode_tokens": metric_map.get("native_decode_tokens") if isinstance(metric_map.get("native_decode_tokens"), int) else None,
        "native_decode_seconds": metric_map.get("native_decode_seconds") if _is_number(metric_map.get("native_decode_seconds")) else None,
        "native_decode_rate": {
            "value": metric_map.get("native_decode_rate_tokens_per_second") if _is_number(metric_map.get("native_decode_rate_tokens_per_second")) else None,
            "unit": "tokens_per_second",
        },
        "wall_decode_rate": {
            "value": metric_map.get("wall_decode_rate_tokens_per_second") if _is_number(metric_map.get("wall_decode_rate_tokens_per_second")) else None,
            "unit": "tokens_per_second",
        },
        "reasoning_chars_observed": metric_map.get("reasoning_chars_observed") if isinstance(metric_map.get("reasoning_chars_observed"), int) else None,
        "reasoning_tokens_reported": metric_map.get("reasoning_tokens_reported") if isinstance(metric_map.get("reasoning_tokens_reported"), int) else None,
    }


def validate_specialist_artifact(
    path: Path,
    *,
    expected_cases: Sequence[str],
    expected_thinking: str,
    expected_temperature: float,
    expected_tag: str,
    expected_profile: str,
) -> tuple[list[dict[str, Any]], Binding]:
    rows, binding = read_jsonl(path, "specialist result")
    if len(rows) != len(expected_cases):
        raise ControllerError(
            f"specialist result row count {len(rows)} differs from expected {len(expected_cases)}"
        )
    summaries = [
        _specialist_summary(
            row, expected_thinking=expected_thinking,
            expected_temperature=expected_temperature,
            expected_tag=expected_tag, expected_profile=expected_profile,
        )
        for row in rows
    ]
    if [item["case_id"] for item in summaries] != list(expected_cases):
        raise ControllerError("specialist result case order/identity differs")
    return summaries, binding


def specialist_stage_outcome(rows: Sequence[Mapping[str, Any]]) -> str:
    """Classify result semantics without turning ordinary wrong answers into infra errors."""
    if any(row.get("request_error") is True for row in rows):
        return "infrastructure_error"
    if any(row.get("complete") is not True for row in rows):
        return "semantic_negative"
    if all(row.get("passed") is True for row in rows):
        return "pass"
    return "semantic_negative"


def assert_pilot_prefix_unchanged(before: JsonlRow, after: Sequence[JsonlRow]) -> None:
    if not after:
        raise ControllerError("T1 pilot row disappeared during continuation")
    first = after[0]
    if (
        first.line != before.line
        or first.raw_sha256 != before.raw_sha256
        or first.canonical_sha256 != before.canonical_sha256
    ):
        raise ControllerError("T1 pilot row changed or was reexecuted during continuation")


def _main_code_summary(row: JsonlRow) -> dict[str, Any]:
    value = row.value
    metrics = value.get("response_metrics")
    metric_map = metrics if isinstance(metrics, Mapping) else {}
    response = value.get("response")
    error = value.get("error")
    evaluation = value.get("evaluation")
    return {
        "case_id": value.get("case_id"),
        "complete": bool(
            error is None and isinstance(response, Mapping)
            and response.get("stream_complete") is True and response.get("truncated") is not True
        ),
        "request_error": isinstance(error, Mapping),
        "passed": bool(isinstance(evaluation, Mapping) and evaluation.get("passed") is True),
        "wall_seconds": metric_map.get("wall_seconds") if _is_number(metric_map.get("wall_seconds")) else None,
        "native_prompt_tokens": metric_map.get("native_prompt_tokens") if isinstance(metric_map.get("native_prompt_tokens"), int) else None,
        "native_cached_prompt_tokens": metric_map.get("native_cached_prompt_tokens") if isinstance(metric_map.get("native_cached_prompt_tokens"), int) else None,
        "native_decode_rate": {
            "value": metric_map.get("native_decode_rate_tokens_per_second")
            if _is_number(metric_map.get("native_decode_rate_tokens_per_second")) else None,
            "unit": "tokens_per_second",
        },
        "wall_decode_rate": {
            "value": metric_map.get("wall_decode_rate_tokens_per_second")
            if _is_number(metric_map.get("wall_decode_rate_tokens_per_second")) else None,
            "unit": "tokens_per_second",
        },
        "source_line": row.line,
        "source_raw_line_sha256": row.raw_sha256,
        "source_row_canonical_sha256": row.canonical_sha256,
    }


def mode_comparison(
    *, reference: Sequence[Mapping[str, Any]], candidate: Sequence[Mapping[str, Any]],
    reference_mode: str, candidate_mode: str,
) -> dict[str, Any]:
    """Build scalar/hash-only case comparisons for the rows actually run."""
    by_case = {row.get("case_id"): row for row in reference}
    rows: list[dict[str, Any]] = []
    for item in candidate:
        case_id = item.get("case_id")
        baseline = by_case.get(case_id)
        if not isinstance(case_id, str) or not isinstance(baseline, Mapping):
            raise ControllerError("mode comparison has no same-case reference")
        candidate_wall = item.get("wall_seconds")
        reference_wall = baseline.get("wall_seconds")
        wall_known = _is_number(candidate_wall) and _is_number(reference_wall)
        rows.append({
            "case_id": case_id,
            "reference": {
                "mode": reference_mode,
                "passed": baseline.get("passed") is True,
                "complete": baseline.get("complete") is True,
                "wall_seconds": float(reference_wall) if _is_number(reference_wall) else None,
                "native_decode_rate": baseline.get("native_decode_rate", {"value": None, "unit": "tokens_per_second"}),
            },
            "candidate": {
                "mode": candidate_mode,
                "passed": item.get("passed") is True,
                "complete": item.get("complete") is True,
                "wall_seconds": float(candidate_wall) if _is_number(candidate_wall) else None,
                "native_decode_rate": item.get("native_decode_rate", {"value": None, "unit": "tokens_per_second"}),
            },
            "candidate_minus_reference_wall": {
                "value": float(candidate_wall) - float(reference_wall) if wall_known else None,
                "unit": "seconds",
            },
            "native_prompt_cache_compatibility": cache_compatibility(item, baseline),
        })
    return {
        "reference_mode": reference_mode,
        "candidate_mode": candidate_mode,
        "actual_case_count": len(rows),
        "candidate_complete_count": sum(row["candidate"]["complete"] for row in rows),
        "candidate_pass_count": sum(row["candidate"]["passed"] for row in rows),
        "rows": rows,
    }


def validate_setup_result(path: Path) -> dict[str, Any]:
    rows, binding = read_jsonl(path, "setup result")
    if len(rows) != 1:
        raise ControllerError("setup must contain exactly one counted result row")
    row = rows[0].value
    request = row.get("request")
    if row.get("case_id") != "quick_json_extract" or not isinstance(request, Mapping):
        raise ControllerError("setup row has the wrong case identity")
    required = {
        "model": MODEL, "thinking": "none", "temperature": 0, "seed": 0,
        "context_max_tokens": CONTEXT, "output_token_limit": 256, "timeout_seconds": 180,
    }
    for key, expected in required.items():
        actual = request.get(key)
        if _is_number(expected) and _is_number(actual):
            matched = float(actual) == float(expected)
        else:
            matched = actual == expected
        if not matched:
            raise ControllerError(f"setup row has mismatched {key}")
    error = row.get("error")
    response = row.get("response")
    if error is None and not (isinstance(response, Mapping) and response.get("stream_complete") is True):
        raise ControllerError("setup row is neither terminal response nor request error")
    evaluation = row.get("evaluation")
    return {
        "source": binding.as_dict(),
        "source_line": rows[0].line,
        "source_raw_line_sha256": rows[0].raw_sha256,
        "source_row_canonical_sha256": rows[0].canonical_sha256,
        "request_error": isinstance(error, Mapping),
        "request_error_type": error.get("type") if isinstance(error, Mapping) else None,
        "semantic_pass": bool(isinstance(evaluation, Mapping) and evaluation.get("passed") is True),
        "semantic_failure_allows_mandatory_code3": error is None,
    }


def read_lifecycle(path: Path) -> tuple[tuple[JsonlRow, ...], Binding]:
    return read_jsonl(path, "owned model-session lifecycle")


def holding_session(
    path: Path, *, expected_run_id: str | None = None,
) -> HoldingSession | None:
    rows, _ = read_lifecycle(path)
    starts = [row for row in rows if row.value.get("event") == "session_started"]
    if not starts:
        return None
    if len(starts) != 1:
        raise ControllerError("owned lifecycle has ambiguous session_started rows")
    start = starts[0]
    run_id = start.value.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise ControllerError("owned lifecycle lacks a run ID")
    if expected_run_id is not None and run_id != expected_run_id:
        raise ControllerError("owned lifecycle run ID changed")
    if any(row.value.get("run_id") != run_id or row.value.get("tag") != TAG for row in rows):
        raise ControllerError("owned lifecycle mixes run or tag identities")
    if (
        Path(str(start.value.get("source_config"))).resolve() != SOURCE_CONFIG
        or Path(str(start.value.get("plan"))).resolve() != SETUP_PLAN
        or start.value.get("runtime_backend") != "strata"
        or start.value.get("model_name") != MODEL
        or start.value.get("context") != CONTEXT
        or start.value.get("control_file") != str(controller_paths()["control"])
        or start.value.get("engine_environment") != TAIL_ENVIRONMENT
    ):
        raise ControllerError("owned session_started identity differs")
    server = [row for row in rows if row.value.get("event") == "server_started"]
    ready = [row for row in rows if row.value.get("event") == "server_ready"]
    holding = [row for row in rows if row.value.get("event") == "holding_started"]
    if not holding:
        return None
    if len(server) != 1 or len(ready) != 1 or len(holding) != 1:
        raise ControllerError("owned lifecycle has ambiguous server/ready/holding rows")
    pid = server[0].value.get("pid")
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        raise ControllerError("server_started lacks a positive API PID")
    if ready[0].value.get("model") != MODEL:
        raise ControllerError("server_ready model differs")
    return HoldingSession(run_id, pid, MODEL, CONTEXT, start.line, server[0].line)


def _private_config_expected(source: Mapping[str, Any], engine_log: Path) -> Mapping[str, Any]:
    private = copy.deepcopy(dict(source))
    private["host"] = "127.0.0.1"
    private["port"] = 18_080
    private["open_browser"] = False
    private["reasoning_effort"] = "none"
    private["log"] = str(engine_log)
    return private


def validate_loaded_config(paths: Mapping[str, Path]) -> Binding:
    source, _ = load_json(SOURCE_CONFIG, "Flash source config")
    private, binding = load_json(paths["private_config"], "owned private config")
    if private != _private_config_expected(source, paths["engine_log"]):
        raise ControllerError("owned private config differs beyond per-tag log/default NONE fields")
    shared = paths["shared_settings"]
    if os.path.lexists(shared):
        details = _regular(shared, "owned shared settings")
        if details.st_size:
            payload = shared.read_bytes().strip()
            if payload not in {b"", b"{}"}:
                raise ControllerError("owned shared settings must be absent or empty")
    return binding


def _proc_identity(pid: int) -> ProcessIdentity:
    stat_path = Path(f"/proc/{pid}/stat")
    try:
        raw = stat_path.read_text(encoding="utf-8")
        close = raw.rfind(")")
        fields = raw[close + 2:].split()
        exe = os.readlink(f"/proc/{pid}/exe")
    except OSError as exc:
        raise ControllerError(f"process identity is unavailable for PID {pid}") from exc
    if close < 1 or len(fields) < 20:
        raise ControllerError(f"process stat is malformed for PID {pid}")
    return ProcessIdentity(pid, int(fields[1]), fields[19], fields[0], exe)


def _stable_process_identity_matches(
    observed: ProcessIdentity, expected: ProcessIdentity,
) -> bool:
    return bool(
        observed.state not in {"Z", "X", "x"}
        and observed.pid == expected.pid
        and observed.parent_pid == expected.parent_pid
        and observed.start_ticks == expected.start_ticks
        and observed.exe == expected.exe
    )


def _process_environment(pid: int) -> dict[str, str]:
    try:
        raw = Path(f"/proc/{pid}/environ").read_bytes()
    except OSError as exc:
        raise ControllerError(f"process environment is unavailable for PID {pid}") from exc
    values: dict[str, str] = {}
    for item in raw.split(b"\0"):
        if b"=" not in item:
            continue
        key, value = item.split(b"=", 1)
        values[key.decode("utf-8", "surrogateescape")] = value.decode("utf-8", "surrogateescape")
    return values


def _kernel_clock() -> tuple[str, int, int, int]:
    boot_id = Path("/proc/sys/kernel/random/boot_id").read_text(encoding="ascii").strip()
    boot_time: int | None = None
    for line in Path("/proc/stat").read_text(encoding="ascii").splitlines():
        if line.startswith("btime "):
            boot_time = int(line.split()[1])
            break
    if not boot_id or boot_time is None:
        raise ControllerError("kernel boot identity is unavailable")
    ticks = int(os.sysconf("SC_CLK_TCK"))
    return boot_id, boot_time, ticks, 1


def capture_ownership_document(
    *,
    session: HoldingSession,
    paths: Mapping[str, Path],
    gpu_processes: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    rows, _ = read_lifecycle(paths["runtime_events"])
    by_line = {row.line: row for row in rows}
    start = by_line.get(session.session_started_line)
    server = by_line.get(session.server_started_line)
    if start is None or server is None:
        raise ControllerError("owned lifecycle rows changed before ownership capture")
    api = _proc_identity(session.api_pid)
    candidates: list[tuple[Mapping[str, Any], ProcessIdentity]] = []
    for observation in gpu_processes:
        pid = observation.get("pid")
        if isinstance(pid, int) and not isinstance(pid, bool) and pid > 0:
            identity = _proc_identity(pid)
            if identity.parent_pid == api.pid and identity.exe == EXPECTED_ENGINE:
                candidates.append((observation, identity))
    if len(candidates) != 1:
        raise ControllerError("expected exactly one direct owned GPU-engine child")
    observation, gpu = candidates[0]
    if gpu.pid == api.pid:
        raise ControllerError("API PID must not be recorded as the GPU PID")
    environment = _process_environment(gpu.pid)
    if environment.get("STRATA_ROUTE_TAIL_SKIP") != "0":
        raise ControllerError("owned GPU engine does not have tail skip 0")
    private = validate_loaded_config(paths)
    source = Binding(SOURCE_CONFIG, SOURCE_CONFIG_SHA256, SOURCE_CONFIG_SIZE)
    boot_id, boot_time, ticks, resolution = _kernel_clock()
    gpu_birth = boot_time + int(gpu.start_ticks) / ticks
    return {
        "api_process": {
            "exe": api.exe, "parent_pid": api.parent_pid, "pid": api.pid,
            "start_ticks": api.start_ticks, "state": api.state,
        },
        "direct_child_parent_relation_verified": True,
        "engine_environment": TAIL_ENVIRONMENT,
        "expected_engine_exe": EXPECTED_ENGINE,
        "gpu_observation": {
            "name": observation.get("name"), "pid": gpu.pid,
            "used_mib": observation.get("used_mib"),
        },
        "gpu_process": {
            "exe": gpu.exe, "parent_pid": gpu.parent_pid, "pid": gpu.pid,
            "start_ticks": gpu.start_ticks, "state": gpu.state,
        },
        "kernel_clock": {
            "boot_id": boot_id, "boot_time_resolution_seconds": resolution,
            "boot_time_unix_seconds": boot_time, "clock_ticks_per_second": ticks,
            "gpu_process_start_unix_seconds": gpu_birth,
        },
        "kind": "strata_owned_gpu_process_observation",
        "lifecycle_events": [
            {
                "event": "session_started", "source_line": start.line,
                "source_raw_line_sha256": start.raw_sha256,
                "source_row_canonical_sha256": start.canonical_sha256,
            },
            {
                "event": "server_started", "source_line": server.line,
                "source_raw_line_sha256": server.raw_sha256,
                "source_row_canonical_sha256": server.canonical_sha256,
            },
        ],
        "lifecycle_source": str(paths["runtime_events"]),
        "live_kernel_start_identities_verified": True,
        "model_session_run_id": session.run_id,
        "observed_at": utc_now(),
        "observed_during_live_matching_model_session": True,
        "private_config": private.as_dict(),
        "prose_bodies_read": False,
        "runtime_backend": "strata",
        "schema_version": 1,
        "scope": (
            "API parent PID and direct owned GPU engine child PID are distinct. "
            "This observation binds ownership and does not claim a memory peak or speed result."
        ),
        "source_config": source.as_dict(),
        "tag": TAG,
    }


def validate_ownership_document(
    receipt: Mapping[str, Any],
    *,
    lifecycle_rows: Sequence[JsonlRow],
    private_config: Mapping[str, Any],
    source_config: Mapping[str, Any],
    generated_at: datetime | None = None,
) -> None:
    expected_keys = {
        "api_process", "direct_child_parent_relation_verified", "engine_environment",
        "expected_engine_exe", "gpu_observation", "gpu_process", "kernel_clock", "kind",
        "lifecycle_events", "lifecycle_source", "live_kernel_start_identities_verified",
        "model_session_run_id", "observed_at", "observed_during_live_matching_model_session",
        "private_config", "prose_bodies_read", "runtime_backend", "schema_version", "scope",
        "source_config", "tag",
    }
    if set(receipt) != expected_keys:
        raise ControllerError("ownership receipt fields do not exactly match accepted schema version 1")
    if (
        receipt.get("schema_version") != 1
        or receipt.get("kind") != "strata_owned_gpu_process_observation"
        or receipt.get("runtime_backend") != "strata"
        or receipt.get("tag") != TAG
        or receipt.get("engine_environment") != TAIL_ENVIRONMENT
        or receipt.get("expected_engine_exe") != EXPECTED_ENGINE
        or receipt.get("prose_bodies_read") is not False
    ):
        raise ControllerError("ownership receipt identity differs")
    for field_name in (
        "direct_child_parent_relation_verified", "live_kernel_start_identities_verified",
        "observed_during_live_matching_model_session",
    ):
        if receipt.get(field_name) is not True:
            raise ControllerError(f"ownership receipt {field_name} must be true")
    run_id = receipt.get("model_session_run_id")
    if not isinstance(run_id, str) or not run_id:
        raise ControllerError("ownership receipt run ID is absent")
    by_line = {row.line: row for row in lifecycle_rows}
    bindings = receipt.get("lifecycle_events")
    if not isinstance(bindings, list) or len(bindings) != 2:
        raise ControllerError("ownership receipt must bind two lifecycle events")
    bound: dict[str, JsonlRow] = {}
    for item in bindings:
        if not isinstance(item, Mapping) or set(item) != {
            "event", "source_line", "source_raw_line_sha256", "source_row_canonical_sha256",
        }:
            raise ControllerError("ownership lifecycle binding fields differ")
        event = item.get("event")
        if event not in {"session_started", "server_started"} or event in bound:
            raise ControllerError("ownership lifecycle event is unexpected or duplicated")
        row = by_line.get(item.get("source_line"))
        if row is None or row.value.get("event") != event:
            raise ControllerError("ownership lifecycle binding points to the wrong row")
        if row.value.get("tag") != TAG or row.value.get("run_id") != run_id:
            raise ControllerError("ownership lifecycle binding has a different identity")
        if item.get("source_raw_line_sha256") != row.raw_sha256 or item.get("source_row_canonical_sha256") != row.canonical_sha256:
            raise ControllerError("ownership lifecycle row hash differs")
        bound[event] = row
    api = receipt.get("api_process")
    gpu = receipt.get("gpu_process")
    observation = receipt.get("gpu_observation")
    process_keys = {"exe", "parent_pid", "pid", "start_ticks", "state"}
    if not isinstance(api, Mapping) or set(api) != process_keys or not isinstance(gpu, Mapping) or set(gpu) != process_keys:
        raise ControllerError("ownership process schema differs")
    if not isinstance(observation, Mapping) or set(observation) != {"name", "pid", "used_mib"}:
        raise ControllerError("ownership GPU observation schema differs")
    numeric = (api.get("pid"), api.get("parent_pid"), gpu.get("pid"), gpu.get("parent_pid"), observation.get("pid"))
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in numeric):
        raise ControllerError("ownership PID identities must be positive integers")
    for value in (api.get("start_ticks"), gpu.get("start_ticks")):
        if not isinstance(value, str) or not value.isdecimal() or int(value) <= 0:
            raise ControllerError("ownership start tick identity is invalid")
    if (
        api.get("pid") != bound["server_started"].value.get("pid")
        or gpu.get("pid") == api.get("pid")
        or gpu.get("parent_pid") != api.get("pid")
        or observation.get("pid") != gpu.get("pid")
        or int(gpu["start_ticks"]) <= int(api["start_ticks"])
        or gpu.get("exe") != EXPECTED_ENGINE
        or observation.get("name") != EXPECTED_ENGINE
    ):
        raise ControllerError("ownership API/GPU relationship differs")
    if not _is_number(observation.get("used_mib")) or float(observation["used_mib"]) < 0:
        raise ControllerError("ownership GPU memory observation is invalid")
    if receipt.get("private_config") != private_config or receipt.get("source_config") != source_config:
        raise ControllerError("ownership config binding differs")
    kernel = receipt.get("kernel_clock")
    if not isinstance(kernel, Mapping) or set(kernel) != {
        "boot_id", "boot_time_resolution_seconds", "boot_time_unix_seconds",
        "clock_ticks_per_second", "gpu_process_start_unix_seconds",
    }:
        raise ControllerError("ownership kernel clock schema differs")
    if not isinstance(kernel.get("boot_id"), str) or not kernel.get("boot_id"):
        raise ControllerError("ownership boot ID is absent")
    boot = kernel.get("boot_time_unix_seconds")
    resolution = kernel.get("boot_time_resolution_seconds")
    ticks = kernel.get("clock_ticks_per_second")
    supplied = kernel.get("gpu_process_start_unix_seconds")
    if not (_is_number(boot) and float(boot) > 0 and _is_number(resolution) and float(resolution) > 0):
        raise ControllerError("ownership boot time or resolution is invalid")
    if isinstance(ticks, bool) or not isinstance(ticks, int) or ticks <= 0 or not _is_number(supplied):
        raise ControllerError("ownership kernel tick/birth metadata is invalid")
    derived = float(boot) + int(gpu["start_ticks"]) / ticks
    if not math.isclose(float(supplied), derived, rel_tol=0, abs_tol=1e-6):
        raise ControllerError("ownership GPU birth time differs from kernel identities")
    observed_at = _parse_time(receipt.get("observed_at"), "ownership observed_at")
    if generated_at is not None and observed_at > generated_at.astimezone(timezone.utc):
        raise ControllerError("ownership observation is later than receipt validation time")


def _write_exclusive_json(path: Path, value: Mapping[str, Any]) -> Binding:
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise OSError("short immutable JSON write")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return Binding(path.resolve(), hashlib.sha256(payload).hexdigest(), len(payload))


class EventWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(path, flags, 0o600)
        os.close(descriptor)

    def write(self, event: str, **fields: Any) -> None:
        row = {"at": utc_now(), "event": event, **fields}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())


def _port_clear(port: int) -> bool:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.settimeout(0.3)
        return sock.connect_ex(("127.0.0.1", port)) != 0
    finally:
        sock.close()


def validate_health_payload(payload: Any, *, status_code: int = 200) -> dict[str, Any]:
    """Validate Strata's live engine-backed health metadata.

    The lifecycle context is a requested value.  ``/health.max_context`` is
    the server's report from the loaded engine, so every stage binds the
    comparison to that independently observed value.
    """
    if status_code != 200 or not isinstance(payload, Mapping):
        raise ControllerError("owned API health response is unavailable or malformed")
    if payload.get("status") != "ok" or payload.get("service") != "strata":
        raise ControllerError("owned API health response has the wrong service identity")
    if payload.get("loaded") is not True:
        raise ControllerError("owned API reports that the engine is not loaded")
    if payload.get("model") != MODEL:
        raise ControllerError("owned API health reports the wrong model")
    context = payload.get("max_context")
    if isinstance(context, bool) or not isinstance(context, int) or context != CONTEXT:
        raise ControllerError("owned API health reports the wrong loaded context")
    return {
        "status": "ok", "service": "strata", "loaded": True,
        "model": MODEL, "max_context": context,
    }


def _api_health_snapshot() -> dict[str, Any]:
    connection = http.client.HTTPConnection("127.0.0.1", 18_080, timeout=3)
    try:
        connection.request("GET", "/health", headers={"Accept": "application/json"})
        response = connection.getresponse()
        payload = json.loads(response.read(2 * 1024 * 1024))
    except (OSError, TimeoutError, http.client.HTTPException, json.JSONDecodeError) as exc:
        raise ControllerError("owned API health request failed") from exc
    finally:
        connection.close()
    return validate_health_payload(payload, status_code=response.status)


def live_preflight() -> dict[str, Any]:
    from runtime import gpu_window

    state = gpu_window.original_state()
    if state.get("Running") is not True or state.get("Paused") is True:
        raise ControllerError("original service is not running and unpaused")
    # /api/ps is a read-only health request through the frozen helper.
    gpu_window.loaded_models(time.monotonic() + 3.0)
    clear = [port for port in PORTS if _port_clear(port)]
    if clear != list(PORTS):
        raise ControllerError("temporary loopback ports are not both clear")
    processes = gpu_window.gpu_compute_processes(10.0)
    if processes:
        raise ControllerError("GPU compute process is present before the owned window")
    return {
        "checked_at": utc_now(), "original_running": True, "original_paused": False,
        "original_api_healthy": True, "ports_clear": clear,
        "gpu_compute_process_count": 0,
    }


def wait_for_holding(
    outer: subprocess.Popen[Any], paths: Mapping[str, Path], *, deadline: float, signals: SignalLatch,
) -> HoldingSession:
    while time.monotonic() < deadline:
        signals.check()
        if outer.poll() is not None:
            raise ControllerError(f"GPU window exited {outer.returncode} before holding_started")
        if os.path.lexists(paths["runtime_events"]):
            session = holding_session(paths["runtime_events"])
            if session is not None:
                return session
        time.sleep(0.5)
    raise ControllerError("owned session did not reach holding_started before its deadline")


def ensure_time_for_stage(deadline: float, case_count: int, stage: str) -> None:
    # ``deadline`` is already WINDOW_SECONDS minus the cleanup reserve.  Keep
    # 90 seconds for helper startup/terminal handling without counting the
    # restoration allowance twice.
    required = case_count * REQUEST_TIMEOUT + 90.0
    remaining = deadline - time.monotonic()
    if remaining < required:
        raise ControllerError(
            f"selected {stage} cannot fit with full per-call allowance and cleanup reserve"
        )


def run_helper(
    argv: Sequence[str], *, outer: subprocess.Popen[Any], deadline: float,
    signals: SignalLatch,
) -> dict[str, Any]:
    started_at = utc_now()
    started = time.monotonic()
    process = subprocess.Popen(
        list(argv), stdin=subprocess.DEVNULL, start_new_session=True, close_fds=True,
    )
    identity = _proc_identity(process.pid)
    identity_record = {
        "pid": identity.pid, "parent_pid": identity.parent_pid,
        "start_ticks": identity.start_ticks, "state_at_start": identity.state,
        "exe": identity.exe,
    }
    terminated = False
    try:
        while True:
            signals.check()
            returncode = process.poll()
            if returncode is not None:
                return {
                    "argv": list(argv), "started_at": started_at, "finished_at": utc_now(),
                    "elapsed_seconds": round(time.monotonic() - started, 3),
                    "returncode": returncode, "owned_sigterm_sent": terminated,
                    "owned_sigkill_sent": False, "owned_process": identity_record,
                }
            if outer.poll() is not None:
                raise ControllerError("GPU window exited while a benchmark helper was running")
            if time.monotonic() >= deadline:
                raise ControllerError("owned window deadline reached during benchmark helper")
            time.sleep(0.25)
    except BaseException as exc:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
                terminated = True
            except ProcessLookupError:
                pass
            until = time.monotonic() + 30.0
            while process.poll() is None and time.monotonic() < until:
                time.sleep(0.1)
        cleanup = {
            "owned_pid": process.pid, "owned_sigterm_sent": terminated,
            "owned_sigkill_sent": False, "stopped": process.poll() is not None,
            "owned_process": identity_record,
        }
        setattr(exc, "owned_helper_cleanup", cleanup)
        if process.poll() is None:
            failure = ControllerError("owned helper did not stop after SIGTERM; SIGKILL was not sent")
            setattr(failure, "owned_helper_cleanup", cleanup)
            raise failure from exc
        raise


def run_helper_accounted(
    argv: Sequence[str], *, stage: str, ledger: BudgetLedger, output: Path,
    expected_new_rows: int, existing_rows: int, outer: subprocess.Popen[Any],
    deadline: float, signals: SignalLatch,
    runner: Callable[..., dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], tuple[JsonlRow, ...]]:
    """Run one helper and preserve uncertainty unless every reserved row exists.

    A terminal helper process does not prove that an API call was never started
    before an absent append-stable row.  Exact accounting therefore requires
    the full expected row set, even when the helper itself has exited.
    """
    selected_runner = run_helper if runner is None else runner
    run = selected_runner(argv, outer=outer, deadline=deadline, signals=signals)
    rows, _ = read_jsonl(output, f"{stage} result")
    new_rows = max(0, len(rows) - existing_rows)
    ledger.observe(
        stage, new_rows,
        helper_terminal=new_rows == expected_new_rows,
    )
    return run, rows


def request_owned_stop(
    control: Path, events: Path, expected: HoldingSession,
    *, expected_api_identity: ProcessIdentity | None = None,
    writer: Callable[[Path], bool] | None = None,
) -> bool:
    current = holding_session(events, expected_run_id=expected.run_id)
    if current != expected:
        raise ControllerError("refusing stop because owned lifecycle identity changed")
    observed = _proc_identity(expected.api_pid)
    if observed.pid != expected.api_pid:
        raise ControllerError("refusing stop because API process identity is unavailable")
    if expected_api_identity is not None and not _stable_process_identity_matches(observed, expected_api_identity):
        raise ControllerError("refusing stop because API PID/PPID/start/executable identity changed")
    if writer is not None:
        return writer(control)
    from runtime import specialist_study_trial as study
    return study.request_session_stop(control)


def verify_live_owned_session(
    session: HoldingSession, paths: Mapping[str, Path],
    api_identity: ProcessIdentity, gpu_identity: ProcessIdentity,
) -> Mapping[str, Any]:
    from runtime import gpu_window

    current_session = holding_session(paths["runtime_events"], expected_run_id=session.run_id)
    if current_session != session:
        raise ControllerError("held session identity changed during selected work")
    current_api = _proc_identity(api_identity.pid)
    current_gpu = _proc_identity(gpu_identity.pid)
    if not _stable_process_identity_matches(current_api, api_identity):
        raise ControllerError("owned API PID/start/executable identity changed")
    if not _stable_process_identity_matches(current_gpu, gpu_identity):
        raise ControllerError("owned GPU PID/start/executable identity changed")
    if current_gpu.parent_pid != current_api.pid:
        raise ControllerError("owned GPU process is no longer a direct API child")
    if _process_environment(current_gpu.pid).get("STRATA_ROUTE_TAIL_SKIP") != "0":
        raise ControllerError("owned GPU tail environment changed")
    validate_loaded_config(paths)
    health = _api_health_snapshot()
    observations = gpu_window.gpu_compute_processes(10.0)
    matching = [item for item in observations if item.get("pid") == current_gpu.pid]
    foreign = [item for item in observations if item.get("pid") != current_gpu.pid]
    if (
        len(matching) != 1
        or matching[0].get("name") != EXPECTED_ENGINE
        or foreign
    ):
        raise ControllerError("owned GPU observation changed or a foreign GPU process appeared")
    return {
        "checked_at": utc_now(), "run_id": session.run_id,
        "api_pid": current_api.pid, "api_start_ticks": current_api.start_ticks,
        "gpu_pid": current_gpu.pid, "gpu_start_ticks": current_gpu.start_ticks,
        "gpu_used_mib": matching[0].get("used_mib"),
        "tail_skip": "0", "model": health["model"],
        "context": health["max_context"], "health_loaded": health["loaded"],
    }


def restoration_retry(
    probe: Callable[[], Mapping[str, Any]],
    *, attempts: int = 60,
    sleep: Callable[[float], None] = time.sleep,
) -> Mapping[str, Any]:
    last: Mapping[str, Any] | None = None
    problems: list[str] = []
    for attempt in range(1, attempts + 1):
        try:
            last = dict(probe())
        except Exception as exc:  # cleanup observation must retain and retry concrete errors
            problems.append(f"{type(exc).__name__}: {exc}")
        else:
            if (
                last.get("original_running") is True
                and last.get("original_paused") is False
                and last.get("original_api_healthy") is True
                and last.get("ports_clear") == [18_080, 18_081]
                and last.get("owned_identities_gone") is True
                and last.get("foreign_gpu_process_count") == 0
            ):
                return {**last, "attempt": attempt, "observation_errors": problems}
        if attempt < attempts:
            sleep(2.0)
    raise ControllerError(
        "restoration did not become healthy after bounded retries: "
        + (problems[-1] if problems else json.dumps(last, sort_keys=True))
    )


def _identity_alive(identity: ProcessIdentity) -> bool:
    try:
        observed = _proc_identity(identity.pid)
    except ControllerError:
        return False
    return observed.start_ticks == identity.start_ticks and observed.exe == identity.exe


def restoration_probe(api: ProcessIdentity, gpu: ProcessIdentity) -> Mapping[str, Any]:
    from runtime import gpu_window

    state = gpu_window.original_state()
    try:
        gpu_window.loaded_models(time.monotonic() + 3.0)
        api_healthy = True
    except Exception:
        api_healthy = False
    clear = [port for port in PORTS if _port_clear(port)]
    processes = gpu_window.gpu_compute_processes(10.0)
    return {
        "checked_at": utc_now(),
        "original_running": state.get("Running") is True,
        "original_paused": state.get("Paused") is True,
        "original_api_healthy": api_healthy,
        "ports_clear": clear,
        "owned_identities_gone": not _identity_alive(api) and not _identity_alive(gpu),
        "foreign_gpu_process_count": len(processes),
    }


def _session_cleanup_evidence(
    paths: Mapping[str, Path], session: HoldingSession, outer_returncode: int,
) -> Mapping[str, Any]:
    report, binding = load_json(paths["session_report"], "owned model-session report")
    cleanup = report.get("server", {}).get("cleanup") if isinstance(report.get("server"), Mapping) else None
    if not (
        report.get("session_id") == session.run_id
        and report.get("tag") == TAG
        and report.get("runtime_backend") == "strata"
        and report.get("status") in {"passed", "completed_with_step_failures"}
        and report.get("problem") is None
        and report.get("source_config_sha256") == SOURCE_CONFIG_SHA256
        and isinstance(cleanup, Mapping)
        and cleanup.get("stopped") is True
        and cleanup.get("kill_sent") is False
    ):
        raise ControllerError("owned model session lacks exact clean terminal evidence")
    expected_returncode = 0 if report.get("status") == "passed" else 1
    if outer_returncode != expected_returncode:
        raise ControllerError("owned GPU window return code differs from model-session outcome")
    rows, events_binding = read_lifecycle(paths["runtime_events"])
    if not any(row.value.get("event") == "server_stopped" for row in rows) or not any(row.value.get("event") == "session_complete" for row in rows):
        raise ControllerError("owned lifecycle lacks clean terminal events")
    return {
        "session_report": binding.as_dict(), "runtime_events": events_binding.as_dict(),
        "session_status": report.get("status"), "server_cleanup": dict(cleanup),
    }


def _artifact_binding_or_none(path: Path) -> Mapping[str, Any] | None:
    if not path.is_file() or path.is_symlink():
        return None
    return Binding(path.resolve(), sha256_file(path), path.stat().st_size).as_dict()


def execute(acceptance: Acceptance) -> tuple[Mapping[str, Any], int]:
    from runtime import gpu_window

    plan, _ = load_json(PLAN_PATH, "prepared execution plan")
    commands = exact_commands(plan)
    if plan.get("environment_overrides") != TAIL_ENVIRONMENT or plan.get("planned_tag") != TAG:
        raise ControllerError("prepared execution identity changed")
    paths = controller_paths()
    assert_outputs_unused(paths, allowed_tag_paths=(acceptance.source.path,))
    boundary_before = live_preflight()
    writer = EventWriter(paths["controller_events"])
    writer.write("controller_started", tag=TAG, acceptance=acceptance.source.as_dict())
    budget = BudgetLedger()
    budget.reserve("setup", 1)
    budget.reserve("none_code3", 3)
    budget.reserve("t1_pilot", 1)
    signals = SignalLatch()
    previous = {signum: signal.getsignal(signum) for signum in (signal.SIGINT, signal.SIGTERM)}
    outer: subprocess.Popen[Any] | None = None
    window_identity: ProcessIdentity | None = None
    session: HoldingSession | None = None
    api_identity: ProcessIdentity | None = None
    gpu_identity: ProcessIdentity | None = None
    started = time.monotonic()
    deadline = started + WINDOW_SECONDS - RESTORATION_RESERVE_SECONDS
    primary_problem: dict[str, Any] | None = None
    cleanup_problem: dict[str, Any] | None = None
    setup: Mapping[str, Any] | None = None
    ownership_binding: Binding | None = None
    none_rows: list[dict[str, Any]] = []
    pilot_rows: list[dict[str, Any]] = []
    continuation_rows: list[dict[str, Any]] = []
    runtime_checks: list[dict[str, Any]] = []
    gate: Mapping[str, Any] | None = None
    helper_runs: list[dict[str, Any]] = []
    cleanup: Mapping[str, Any] | None = None
    restoration: Mapping[str, Any] | None = None
    status = "infrastructure_failed"
    code = 1
    pending_selected_work: list[str] = []
    try:
        for signum in previous:
            signal.signal(signum, signals.handle)
        env = dict(os.environ)
        env.update(TAIL_ENVIRONMENT)
        budget.mark_launched("setup")
        writer.write("window_launching", argv=commands["window"], environment_overrides=TAIL_ENVIRONMENT)
        outer = subprocess.Popen(
            commands["window"], stdin=subprocess.DEVNULL, start_new_session=True,
            close_fds=True, env=env,
        )
        window_identity = _proc_identity(outer.pid)
        session = wait_for_holding(outer, paths, deadline=deadline, signals=signals)
        api_identity = _proc_identity(session.api_pid)
        setup = validate_setup_result(paths["setup"])
        budget.observe("setup", 1, helper_terminal=True)
        validate_loaded_config(paths)
        gpu_observations = gpu_window.gpu_compute_processes(10.0)
        ownership = capture_ownership_document(
            session=session, paths=paths, gpu_processes=gpu_observations,
        )
        lifecycle_rows, _ = read_lifecycle(paths["runtime_events"])
        private_binding = ownership["private_config"]
        source_binding = ownership["source_config"]
        validate_ownership_document(
            ownership, lifecycle_rows=lifecycle_rows,
            private_config=private_binding, source_config=source_binding,
            generated_at=datetime.now(timezone.utc),
        )
        ownership_binding = _write_exclusive_json(paths["ownership"], ownership)
        api_identity = ProcessIdentity(**ownership["api_process"])
        gpu_identity = ProcessIdentity(**ownership["gpu_process"])
        writer.write("ownership_captured", run_id=session.run_id, ownership=ownership_binding.as_dict())
        runtime_checks.append({
            "stage": "before_none_code3",
            **verify_live_owned_session(session, paths, api_identity, gpu_identity),
        })
        if setup["request_error"]:
            raise ControllerError("setup completed with a request/runtime error")

        ensure_time_for_stage(deadline, 3, "mandatory NONE code3")
        budget.mark_launched("none_code3")
        run, observed_rows = run_helper_accounted(
            commands["none"], stage="none_code3", ledger=budget,
            output=paths["none"], expected_new_rows=3, existing_rows=0,
            outer=outer, deadline=deadline, signals=signals,
        )
        helper_runs.append({"kind": "mandatory_none_code3", **run})
        none_rows, _ = validate_specialist_artifact(
            paths["none"], expected_cases=CODE_CASES,
            expected_thinking="none", expected_temperature=0,
            expected_tag=TAG, expected_profile=NONE_PROFILE,
        )
        if specialist_stage_outcome(none_rows) == "infrastructure_error":
            raise ControllerError("mandatory NONE code3 contains a request/runtime error")
        if run["returncode"] != 0:
            raise ControllerError("mandatory NONE helper returned nonzero without an accepted full result")
        runtime_checks.append({
            "stage": "after_none_code3",
            **verify_live_owned_session(session, paths, api_identity, gpu_identity),
        })

        try:
            ensure_time_for_stage(deadline, 1, "selected T1 pilot")
        except ControllerError as exc:
            raise SelectedWorkDeferred(
                "t1_pilot", (CODE_CASES[0],), str(exc),
            ) from exc
        budget.mark_launched("t1_pilot")
        run, observed_rows = run_helper_accounted(
            commands["pilot"], stage="t1_pilot", ledger=budget,
            output=paths["t1"], expected_new_rows=1, existing_rows=0,
            outer=outer, deadline=deadline, signals=signals,
        )
        helper_runs.append({"kind": "selected_t1_pilot", **run})
        pilot_rows, _ = validate_specialist_artifact(
            paths["t1"], expected_cases=(CODE_CASES[0],),
            expected_thinking="medium", expected_temperature=1,
            expected_tag=f"{TAG}-publisher-thinking", expected_profile=T1_PROFILE,
        )
        if any(row["request_error"] for row in pilot_rows):
            raise ControllerError("selected T1 pilot contains a request/runtime error")
        if run["returncode"] != 0:
            raise ControllerError("selected T1 helper returned nonzero without an accepted result")
        runtime_checks.append({
            "stage": "after_t1_pilot",
            **verify_live_owned_session(session, paths, api_identity, gpu_identity),
        })
        main_fresh = _main_code_summary(acceptance.main_rows["code3"][0])
        none_fresh = none_rows[0]
        gate = evaluate_continuation_gate(
            pilot=pilot_rows[0], main_t0=main_fresh, none_fresh=none_fresh,
            none_pass_count=sum(row["passed"] is True for row in none_rows),
        )
        writer.write("continuation_gate", **gate)
        if gate["continue_remaining_two"]:
            try:
                ensure_time_for_stage(deadline, 2, "conditional T1 continuation")
            except ControllerError:
                pending_selected_work.extend(CONTINUATION_CASES)
            else:
                budget.reserve("t1_continuation", 2)
                budget.mark_launched("t1_continuation")
                pilot_pin = observed_rows[0]
                run, all_rows = run_helper_accounted(
                    commands["continuation"], stage="t1_continuation", ledger=budget,
                    output=paths["t1"], expected_new_rows=2, existing_rows=1,
                    outer=outer, deadline=deadline, signals=signals,
                )
                helper_runs.append({"kind": "conditional_t1_remaining_code2", **run})
                assert_pilot_prefix_unchanged(pilot_pin, all_rows)
                all_summaries, _ = validate_specialist_artifact(
                    paths["t1"], expected_cases=CODE_CASES,
                    expected_thinking="medium", expected_temperature=1,
                    expected_tag=f"{TAG}-publisher-thinking", expected_profile=T1_PROFILE,
                )
                continuation_rows = all_summaries[1:]
                if any(row["request_error"] for row in continuation_rows):
                    raise ControllerError("conditional T1 continuation contains a request/runtime error")
                if run["returncode"] != 0:
                    raise ControllerError("conditional T1 helper returned nonzero without full results")
                runtime_checks.append({
                    "stage": "after_t1_continuation",
                    **verify_live_owned_session(session, paths, api_identity, gpu_identity),
                })
        status = "complete" if not pending_selected_work else "selected_work_pending_continuation"
        code = 0
    except ControllerInterrupted as exc:
        primary_problem = {"error_type": type(exc).__name__, "error": str(exc), "signal": exc.signum}
        status = "interrupted"
        code = 128 + exc.signum
    except SelectedWorkDeferred as exc:
        pending_selected_work.extend(exc.pending_cases)
        gate = {
            "evaluated": False,
            "reason": str(exc),
            "deferred_stage": exc.stage,
            "continuation_cases_selected": False,
        }
        status = "selected_work_pending_continuation"
        code = 0
    except (ControllerError, OSError, subprocess.SubprocessError) as exc:
        primary_problem = {"error_type": type(exc).__name__, "error": str(exc)}
        helper_cleanup = getattr(exc, "owned_helper_cleanup", None)
        if helper_cleanup is not None:
            primary_problem["owned_helper_cleanup"] = helper_cleanup
    finally:
        if outer is not None:
            try:
                if session is None:
                    # Do not signal an ambiguous window.  Wait for a late exact
                    # identity/control file, then use only the cooperative stop.
                    until = time.monotonic() + RESTORATION_RESERVE_SECONDS
                    while time.monotonic() < until and outer.poll() is None and session is None:
                        if os.path.lexists(paths["runtime_events"]):
                            session = holding_session(paths["runtime_events"])
                            if session is not None:
                                api_identity = _proc_identity(session.api_pid)
                        if session is None:
                            time.sleep(0.5)
                if session is not None and outer.poll() is None:
                    stopped = request_owned_stop(
                        paths["control"], paths["runtime_events"], session,
                        expected_api_identity=api_identity,
                    )
                    writer.write("cooperative_stop_requested", run_id=session.run_id, requested=stopped)
                until = time.monotonic() + RESTORATION_RESERVE_SECONDS
                while outer.poll() is None and time.monotonic() < until:
                    time.sleep(0.25)
                if outer.poll() is None:
                    raise ControllerError("owned GPU window did not exit; no signal or SIGKILL was sent")
                if session is not None:
                    cleanup = _session_cleanup_evidence(paths, session, int(outer.returncode))
                if api_identity is not None and gpu_identity is not None:
                    restoration = restoration_retry(
                        lambda: restoration_probe(api_identity, gpu_identity), attempts=60,
                    )
            except (ControllerError, OSError, subprocess.SubprocessError) as exc:
                cleanup_problem = {"error_type": type(exc).__name__, "error": str(exc)}
                status = "cleanup_failed"
                code = 1
        for signum, handler in previous.items():
            signal.signal(signum, handler)
    for stage, slots in budget.reserved.items():
        if stage not in budget.observations:
            path = paths["setup"] if stage == "setup" else paths["none"] if stage == "none_code3" else paths["t1"]
            row_count: int | None
            try:
                rows, _ = read_jsonl(path, f"{stage} accounting source") if os.path.lexists(path) else ((), None)
                row_count = len(rows)
                if stage == "t1_continuation":
                    row_count = max(0, row_count - 1)
            except ControllerError:
                row_count = None
            budget.observe(stage, row_count, helper_terminal=False)
    accounting = budget.summary()
    accounting["additional_t1_setup_requests"] = {
        "allocated_request_slots": 0,
        "launch_attempted": False,
        "observed_result_rows": 0,
        "attempts_lower_bound": 0,
        "attempts_upper_bound": 0,
        "exact_attempts": 0,
        "has_uncertain_started_unrecorded_call": False,
        "reason": "T1 reuses the verified healthy held session after the one counted setup",
    }
    if accounting["budget_violation"]:
        primary_problem = primary_problem or {"error_type": "ControllerError", "error": "generation budget exceeded"}
        status, code = "infrastructure_failed", 1
    main_baseline = [_main_code_summary(row) for row in acceptance.main_rows["code3"]]
    all_t1_rows = [*pilot_rows, *continuation_rows]
    receipt = {
        "schema_version": 1,
        "kind": "flash_modes_sampling_controller_execution",
        "status": status,
        "created_at": utc_now(),
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "tag": TAG,
        "model": MODEL,
        "context_tokens": CONTEXT,
        "output_cap_tokens": OUTPUT_CAP,
        "per_request_timeout_seconds": REQUEST_TIMEOUT,
        "environment_overrides": TAIL_ENVIRONMENT,
        "acceptance": acceptance.source.as_dict(),
        "main_reference": {
            "tag": MAIN_TAG, "run_id": acceptance.main_run_id,
            "ready": acceptance.ready.as_dict(), "study": acceptance.study.as_dict(),
            "release_path": str(MAIN_RELEASE), "release_created_by_controller": False,
            "baseline_code3": main_baseline,
        },
        "live_boundary_before": boundary_before,
        "owned_session": None if session is None else {
            "run_id": session.run_id, "api_pid": session.api_pid,
            "model": session.model, "context": session.context,
        },
        "owned_window_process": None if window_identity is None else {
            "pid": window_identity.pid, "parent_pid": window_identity.parent_pid,
            "start_ticks": window_identity.start_ticks,
            "state_at_start": window_identity.state, "exe": window_identity.exe,
        },
        "ownership": None if ownership_binding is None else ownership_binding.as_dict(),
        "setup": setup,
        "none_code3": none_rows,
        "selected_t1_pilot": pilot_rows,
        "conditional_t1_remaining_code2": continuation_rows,
        "mode_comparisons": {
            "main_t0_to_none": mode_comparison(
                reference=main_baseline, candidate=none_rows,
                reference_mode="canonical_main_T0_medium", candidate_mode="trained_NONE_T0",
            ),
            "main_t0_to_t1": mode_comparison(
                reference=main_baseline, candidate=all_t1_rows,
                reference_mode="canonical_main_T0_medium", candidate_mode="publisher_ON_T1",
            ),
            "none_to_t1": mode_comparison(
                reference=none_rows, candidate=all_t1_rows,
                reference_mode="trained_NONE_T0", candidate_mode="publisher_ON_T1",
            ),
        },
        "continuation_gate": gate,
        "pending_selected_work": pending_selected_work,
        "helper_runs": helper_runs,
        "runtime_identity_checks": runtime_checks,
        "generation_budget": accounting,
        "cleanup": cleanup,
        "restoration": restoration,
        "primary_problem": primary_problem,
        "cleanup_problem": cleanup_problem,
        "artifacts": {
            name: _artifact_binding_or_none(path)
            for name, path in paths.items()
            if name not in {"receipt", "none_visible", "t1_visible"}
        },
        "claims": {
            "private_reasoning_or_answer_text_in_receipt": False,
            "full_t1_code3_claimed": len(continuation_rows) == 2 and len(pilot_rows) == 1,
            "gate_is_statistical_significance": False,
            "main_release_published": False,
        },
    }
    _write_exclusive_json(paths["receipt"], receipt)
    return receipt, code


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acceptance", required=True, type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--execute", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        acceptance = validate_acceptance(args.acceptance)
        if args.check_only:
            summary = {
                "accepted": True,
                "mode": "offline CPU-only validation; no lifecycle queries or files created",
                "acceptance": acceptance.source.as_dict(),
                "main_ready": acceptance.ready.as_dict(),
                "main_study": acceptance.study.as_dict(),
                "main_run_id": acceptance.main_run_id,
                "main_terminal_case_count": sum(len(rows) for rows in acceptance.main_rows.values()),
                "main_baseline_code_case_count": len(acceptance.main_rows["code3"]),
                "planned_generation_request_slots": 5,
                "conditional_generation_request_slots": 2,
                "maximum_generation_request_slots": 7,
            }
            print(json.dumps(summary, sort_keys=True))
            return 0
        _, code = execute(acceptance)
        return code
    except ControllerError as exc:
        print(f"flash_modes_sampling_window: {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
