#!/usr/bin/env python3
"""Supervise the pinned local LFM llama.cpp backend and its exclusive GPU lease."""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import http.client
import json
import os
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
from typing import Any, BinaryIO, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = Path(__file__).with_name("lfm-backend.json")
STATE_DIR = Path("/external/user/.local/state/local-general")
LOCK_PATH = STATE_DIR / "backend.lock"
STATUS_PATH = STATE_DIR / "backend-status.json"
EVENTS_PATH = STATE_DIR / "backend-events.jsonl"
SERVER_LOG_PATH = STATE_DIR / "backend-server.log"

EXPECTED_HOST = "127.0.0.1"
EXPECTED_PORT = 18082
EXPECTED_SERVER = Path("/external/storage/moe-bench/ollama-0.40.6/lib/ollama/llama-server")
EXPECTED_MODEL = Path(
    "/external/storage/moe-bench/models/lfm25-qad-q4_0/"
    "LFM2.5-24B-A2B-QAD-Q4_0.gguf"
)
EXPECTED_MODEL_NAME = "lfm2.5-24b-a2b-qad-q4_0"
EXPECTED_MODEL_SIZE = 13_578_190_880
EXPECTED_MODEL_SHA256 = "d1c03f73ecaf61017d70419eec9e5dc0b05d50ae26c23d522829b934fbabf8a5"
EXPECTED_CONTEXT = 73_728
EXPECTED_LIB_DIRS = (
    Path("/external/storage/moe-bench/ollama-0.40.6/lib/ollama"),
    Path("/external/storage/moe-bench/ollama-0.40.6/lib/ollama/cuda_v13"),
)
EXPECTED_ARGS = (
    "--gpu-layers", "auto",
    "--parallel", "1",
    "--jinja",
    "--fit-ctx", "73728",
    "--reasoning", "on",
    "--reasoning-effort", "medium",
    "--reasoning-format", "deepseek",
    "--no-webui",
    "--no-slots",
    "--no-context-shift",
    "--metrics",
    "--api-key-file", "/external/user/.config/local-general/backend-token",
    "--temp", "0",
)
EXPECTED_ORIGINAL_CONTAINER = "local-codex-ollama-cpu"
EXPECTED_ORIGINAL_MODEL = "gpt-oss-20b:64k"
EXPECTED_ORIGINAL_HOST = "127.0.0.1"
EXPECTED_ORIGINAL_PORT = 11436
EXPECTED_COMPETING_RECEIPT = ROOT / "results/public-model-expansion-20261024-v1/launch.json"
EXPECTED_ARTIFACT_RECEIPT = ROOT / "results/lfm25-qad-q4_0-artifact.json"
EXPECTED_SOURCE_CONFIG = ROOT / "runtime/public-benchmark-pilot-lfm72k-auto-v1.json"
EXPECTED_SOURCE_SHA256 = "e42f3a1f24f6295d25328d4edc1bb9b877aeadfbd68eb0ea33c2c0ccb29395c4"
EXPECTED_GPU_LIMIT_MIB = 256
EXPECTED_DEFAULT_OUTPUT_TOKENS = 8192
EXPECTED_MAX_OUTPUT_TOKENS = 65_536
EXPECTED_API_KEY_FILE = Path("/external/user/.config/local-general/backend-token")
CUDA_BACKEND = EXPECTED_LIB_DIRS[1] / "libggml-cuda.so"
HASH_CHUNK_SIZE = 8 * 1024 * 1024
HTTP_LIMIT = 2 * 1024 * 1024
UNLOAD_TIMEOUT_SECONDS = 90.0
RESTORE_ATTEMPTS = 5


class ServiceError(RuntimeError):
    """A safety or operational condition prevents serving."""


class SafetyStop(ServiceError):
    """A destructive action was refused because ownership could not be proven."""


class ServiceInterrupted(BaseException):
    def __init__(self, signum: int) -> None:
        super().__init__(f"signal {signum}")
        self.signum = signum


class UnexpectedChildExit(ServiceError):
    pass


@dataclass(frozen=True)
class BackendConfig:
    source_path: Path
    server: Path
    model: Path
    model_name: str
    context: int
    lib_dirs: tuple[Path, ...]
    args: tuple[str, ...]
    artifact_receipt: Path
    source_config: Path
    source_sha256: str
    competing_receipt: Path
    original_container: str
    original_model: str
    original_host: str
    original_port: int
    api_key_file: Path
    startup_timeout: float
    shutdown_grace: float
    gpu_limit_mib: int

    @property
    def endpoint(self) -> str:
        return f"http://{EXPECTED_HOST}:{EXPECTED_PORT}/v1"


@dataclass(frozen=True)
class ProcessIdentity:
    pid: int
    pgid: int
    start_ticks: int
    argv: tuple[str, ...]
    state: str

    @property
    def alive(self) -> bool:
        return self.state != "Z"

    def receipt(self) -> dict[str, Any]:
        return {
            "pid": self.pid,
            "pgid": self.pgid,
            "start_ticks": self.start_ticks,
            "argv": list(self.argv),
        }


@dataclass(frozen=True)
class ContainerIdentity:
    container_id: str
    name: str
    running: bool
    paused: bool


@dataclass
class ChildHandle:
    process: subprocess.Popen[Any]
    identity: ProcessIdentity
    log_handle: BinaryIO
    stopped: bool = False


class SignalState:
    def __init__(self) -> None:
        self.received: int | None = None
        self._previous: dict[int, Any] = {}

    def _handle(self, signum: int, _frame: Any) -> None:
        if self.received is None:
            self.received = signum

    def install(self) -> None:
        for signum in (signal.SIGTERM, signal.SIGINT):
            self._previous[signum] = signal.getsignal(signum)
            signal.signal(signum, self._handle)

    def restore(self) -> None:
        for signum, handler in self._previous.items():
            signal.signal(signum, handler)
        self._previous.clear()

    def check(self) -> None:
        if self.received is not None:
            raise ServiceInterrupted(self.received)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _read_json(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ServiceError(f"{description} is unreadable or invalid: {path}") from exc
    if not isinstance(value, dict):
        raise ServiceError(f"{description} must be a JSON object")
    return value


def _resolve(value: Any, base: Path, field: str) -> Path:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ServiceError(f"config {field} must be a non-empty path")
    candidate = Path(value)
    return candidate.resolve() if candidate.is_absolute() else (base / candidate).resolve()


def _exact(actual: Any, expected: Any, field: str) -> None:
    if actual != expected:
        raise ServiceError(f"config {field} does not match the pinned service contract")


def load_config(path: Path) -> BackendConfig:
    path = path.expanduser().resolve()
    source = _read_json(path, "backend config")
    base = path.parent
    service = source.get("service")
    gateway = source.get("gateway")
    request_defaults = source.get("request_defaults")
    source_receipt = source.get("source_config")
    shards = source.get("shards")
    if not isinstance(service, dict) or not isinstance(gateway, dict):
        raise ServiceError("config requires service and gateway objects")
    if not isinstance(request_defaults, dict) or not isinstance(source_receipt, dict):
        raise ServiceError("config requires request_defaults and source_config objects")
    if not isinstance(shards, list) or len(shards) != 1 or not isinstance(shards[0], dict):
        raise ServiceError("config must pin exactly one model shard")
    raw_args = source.get("args")
    raw_lib_dirs = source.get("lib_dirs")
    if not isinstance(raw_args, list) or any(not isinstance(item, str) for item in raw_args):
        raise ServiceError("config args must be a string list")
    if not isinstance(raw_lib_dirs, list) or any(
        not isinstance(item, str) or not item for item in raw_lib_dirs
    ):
        raise ServiceError("config lib_dirs must be a non-empty path list")

    _exact(source.get("schema_version"), 1, "schema_version")
    _exact(source.get("host"), EXPECTED_HOST, "host")
    _exact(source.get("port"), EXPECTED_PORT, "port")
    _exact(_resolve(source.get("server"), base, "server"), EXPECTED_SERVER, "server")
    _exact(_resolve(source.get("model"), base, "model"), EXPECTED_MODEL, "model")
    _exact(source.get("model_name"), EXPECTED_MODEL_NAME, "model_name")
    _exact(source.get("context"), EXPECTED_CONTEXT, "context")
    _exact(tuple(raw_args), EXPECTED_ARGS, "args")
    _exact(tuple(Path(item) for item in raw_lib_dirs), EXPECTED_LIB_DIRS, "lib_dirs")
    _exact(_resolve(source.get("artifact_manifest"), base, "artifact_manifest"), EXPECTED_ARTIFACT_RECEIPT, "artifact_manifest")
    _exact(shards[0].get("name"), EXPECTED_MODEL.name, "shards[0].name")
    _exact(shards[0].get("size"), EXPECTED_MODEL_SIZE, "shards[0].size")
    _exact(shards[0].get("sha256"), EXPECTED_MODEL_SHA256, "shards[0].sha256")
    _exact(request_defaults, {"reasoning_effort": "medium", "temperature": 0}, "request_defaults")
    _exact(gateway.get("default_output_tokens"), EXPECTED_DEFAULT_OUTPUT_TOKENS, "gateway.default_output_tokens")
    _exact(gateway.get("maximum_output_tokens"), EXPECTED_MAX_OUTPUT_TOKENS, "gateway.maximum_output_tokens")
    _exact(service.get("artifact_verification"), "full_sha256_each_start", "service.artifact_verification")
    _exact(service.get("gpu_process_limit_mib"), EXPECTED_GPU_LIMIT_MIB, "service.gpu_process_limit_mib")
    _exact(service.get("original_container"), EXPECTED_ORIGINAL_CONTAINER, "service.original_container")
    _exact(service.get("original_model"), EXPECTED_ORIGINAL_MODEL, "service.original_model")
    _exact(service.get("original_ollama_url"), "http://127.0.0.1:11436", "service.original_ollama_url")
    _exact(
        _resolve(service.get("api_key_file"), base, "service.api_key_file"),
        EXPECTED_API_KEY_FILE,
        "service.api_key_file",
    )
    _exact(_resolve(service.get("state_dir"), base, "service.state_dir"), STATE_DIR, "service.state_dir")
    _exact(
        _resolve(service.get("competing_supervisor_receipt"), base, "service.competing_supervisor_receipt"),
        EXPECTED_COMPETING_RECEIPT,
        "service.competing_supervisor_receipt",
    )
    _exact(_resolve(source_receipt.get("path"), base, "source_config.path"), EXPECTED_SOURCE_CONFIG, "source_config.path")
    _exact(source_receipt.get("sha256"), EXPECTED_SOURCE_SHA256, "source_config.sha256")
    startup_timeout = service.get("startup_timeout_seconds")
    shutdown_grace = service.get("shutdown_grace_seconds")
    if not isinstance(startup_timeout, (int, float)) or isinstance(startup_timeout, bool) or not 30 <= startup_timeout <= 600:
        raise ServiceError("config startup timeout must be from 30 through 600 seconds")
    if not isinstance(shutdown_grace, (int, float)) or isinstance(shutdown_grace, bool) or not 5 <= shutdown_grace <= 120:
        raise ServiceError("config shutdown grace must be from 5 through 120 seconds")
    return BackendConfig(
        source_path=path,
        server=EXPECTED_SERVER,
        model=EXPECTED_MODEL,
        model_name=EXPECTED_MODEL_NAME,
        context=EXPECTED_CONTEXT,
        lib_dirs=EXPECTED_LIB_DIRS,
        args=EXPECTED_ARGS,
        artifact_receipt=EXPECTED_ARTIFACT_RECEIPT,
        source_config=EXPECTED_SOURCE_CONFIG,
        source_sha256=EXPECTED_SOURCE_SHA256,
        competing_receipt=EXPECTED_COMPETING_RECEIPT,
        original_container=EXPECTED_ORIGINAL_CONTAINER,
        original_model=EXPECTED_ORIGINAL_MODEL,
        original_host=EXPECTED_ORIGINAL_HOST,
        original_port=EXPECTED_ORIGINAL_PORT,
        api_key_file=EXPECTED_API_KEY_FILE,
        startup_timeout=float(startup_timeout),
        shutdown_grace=float(shutdown_grace),
        gpu_limit_mib=EXPECTED_GPU_LIMIT_MIB,
    )


def _file_sha256(path: Path, signals: SignalState | None = None) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while True:
                block = handle.read(HASH_CHUNK_SIZE)
                if not block:
                    break
                digest.update(block)
                if signals is not None:
                    signals.check()
    except OSError as exc:
        raise ServiceError(f"could not hash pinned file: {path}") from exc
    return digest.hexdigest()


def _stat_identity(stat: os.stat_result) -> dict[str, int]:
    return {
        "device": stat.st_dev,
        "inode": stat.st_ino,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "ctime_ns": stat.st_ctime_ns,
    }


def read_api_token(path: Path) -> str:
    """Load the bearer token without ever including its value in diagnostics."""
    try:
        path_identity = path.lstat()
    except OSError as exc:
        raise ServiceError("backend API key file is missing or inaccessible") from exc
    if stat.S_ISLNK(path_identity.st_mode) or not stat.S_ISREG(path_identity.st_mode):
        raise ServiceError("backend API key file must be a regular non-symlink file")
    if path_identity.st_uid != os.geteuid():
        raise ServiceError("backend API key file must be owned by the service user")
    if stat.S_IMODE(path_identity.st_mode) != 0o600:
        raise ServiceError("backend API key file permissions must be exactly 0600")
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
        with os.fdopen(descriptor, "rb") as handle:
            before = os.fstat(handle.fileno())
            raw = handle.read(4097)
            after = os.fstat(handle.fileno())
    except OSError as exc:
        raise ServiceError("backend API key file could not be read safely") from exc
    if _stat_identity(before) != _stat_identity(path_identity) or _stat_identity(after) != _stat_identity(before):
        raise ServiceError("backend API key file identity changed during validation")
    if len(raw) > 4096:
        raise ServiceError("backend API key file is too large")
    if raw.endswith(b"\r\n"):
        raw = raw[:-2]
    elif raw.endswith(b"\n"):
        raw = raw[:-1]
    if not raw:
        raise ServiceError("backend API key file is empty")
    if b"\r" in raw or b"\n" in raw or any(byte < 0x21 or byte > 0x7E for byte in raw):
        raise ServiceError("backend API key file must contain one printable ASCII token")
    return raw.decode("ascii")


def verify_artifact(config: BackendConfig, signals: SignalState | None = None) -> str:
    receipt = _read_json(config.artifact_receipt, "artifact receipt")
    shards = receipt.get("shards")
    verdict = receipt.get("verdict")
    verification = receipt.get("verification")
    model = receipt.get("model")
    if receipt.get("immutable") is not True:
        raise ServiceError("artifact receipt is not immutable")
    if not isinstance(shards, list) or len(shards) != 1 or not isinstance(shards[0], dict):
        raise ServiceError("artifact receipt must contain exactly one verified shard")
    if not isinstance(verdict, dict) or not isinstance(verification, dict) or not isinstance(model, dict):
        raise ServiceError("artifact receipt lacks verification metadata")
    shard = shards[0]
    expected_shard = {
        "name": config.model.name,
        "path": str(config.model),
        "size": EXPECTED_MODEL_SIZE,
        "sha256": EXPECTED_MODEL_SHA256,
        "verified": True,
        "full_sha256_verified": True,
    }
    for key, expected in expected_shard.items():
        if shard.get(key) != expected:
            raise ServiceError(f"artifact receipt has the wrong verified shard {key}")
    if model.get("alias") != config.model_name:
        raise ServiceError("artifact receipt is for a different model alias")
    if verdict.get("artifact_complete") is not True or verdict.get("full_sha256_verified") is not True:
        raise ServiceError("artifact receipt does not certify a complete full hash")
    if verification.get("actual_sha256") != EXPECTED_MODEL_SHA256 or verification.get("expected_sha256") != EXPECTED_MODEL_SHA256:
        raise ServiceError("artifact receipt full SHA-256 does not match the pin")
    receipt_identity = verification.get("file_identity_after_verification")
    if not isinstance(receipt_identity, dict):
        raise ServiceError("artifact receipt lacks post-verification file identity")
    try:
        with config.model.open("rb") as handle:
            before = _stat_identity(os.fstat(handle.fileno()))
            if before != receipt_identity:
                raise ServiceError("model file identity differs from its immutable receipt")
            digest = hashlib.sha256()
            while True:
                block = handle.read(HASH_CHUNK_SIZE)
                if not block:
                    break
                digest.update(block)
                if signals is not None:
                    signals.check()
            after = _stat_identity(os.fstat(handle.fileno()))
    except OSError as exc:
        raise ServiceError(f"model is missing or unreadable: {config.model}") from exc
    if before != after or _stat_identity(config.model.stat()) != after:
        raise ServiceError("model file identity changed during verification")
    actual = digest.hexdigest()
    if actual != EXPECTED_MODEL_SHA256:
        raise ServiceError("model full SHA-256 does not match the pinned artifact")
    return actual


def verify_runtime_files(config: BackendConfig, signals: SignalState) -> str:
    read_api_token(config.api_key_file)
    if not config.server.is_file() or not os.access(config.server, os.X_OK):
        raise ServiceError(f"pinned llama-server is missing or not executable: {config.server}")
    for directory in config.lib_dirs:
        if not directory.is_dir():
            raise ServiceError(f"pinned library directory is missing: {directory}")
    if not CUDA_BACKEND.is_file():
        raise ServiceError(f"pinned CUDA backend is missing: {CUDA_BACKEND}")
    if _file_sha256(config.source_config, signals) != config.source_sha256:
        raise ServiceError("source LFM config no longer matches its pinned clone receipt")
    return verify_artifact(config, signals)


def process_identity(pid: int) -> ProcessIdentity | None:
    try:
        raw = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        close = raw.rfind(")")
        if close < 0:
            raise ValueError("missing comm terminator")
        fields = raw[close + 2 :].split()
        state = fields[0]
        start_ticks = int(fields[19])
        argv = tuple(
            part.decode("utf-8", "surrogateescape")
            for part in Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
            if part
        )
        pgid = os.getpgid(pid)
    except FileNotFoundError:
        return None
    except (OSError, ValueError, IndexError) as exc:
        raise ServiceError(f"could not inspect process identity for PID {pid}") from exc
    return ProcessIdentity(pid=pid, pgid=pgid, start_ticks=start_ticks, argv=argv, state=state)


def assert_competing_supervisor_inactive(receipt_path: Path) -> None:
    receipt = _read_json(receipt_path, "competing supervisor launch receipt")
    expected = receipt.get("process")
    if not isinstance(expected, dict):
        raise ServiceError("competing supervisor launch receipt lacks process identity")
    pid = expected.get("pid")
    start_ticks = expected.get("start_ticks")
    argv = expected.get("argv")
    if not isinstance(pid, int) or isinstance(pid, bool) or pid < 1:
        raise ServiceError("competing supervisor launch receipt has invalid PID identity")
    if not isinstance(start_ticks, int) or isinstance(start_ticks, bool) or start_ticks < 1:
        raise ServiceError("competing supervisor launch receipt has invalid start-ticks identity")
    if not isinstance(argv, list) or not argv or any(not isinstance(item, str) for item in argv):
        raise ServiceError("competing supervisor launch receipt has invalid argv identity")
    current = process_identity(pid)
    if current is None or not current.alive:
        return
    if current.start_ticks == start_ticks and current.argv == tuple(argv):
        raise ServiceError("identity-verified public model expansion supervisor is still active")


def ensure_state_dir() -> None:
    STATE_DIR.mkdir(mode=0o700, parents=True, exist_ok=True)
    STATE_DIR.chmod(0o700)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()


class EventLog:
    def __init__(self, path: Path = EVENTS_PATH) -> None:
        self.path = path

    def write(self, event: str, **fields: Any) -> None:
        row = {"at": utc_now(), "event": event, **fields}
        descriptor = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())


def _safe_event(events: EventLog, event: str, **fields: Any) -> None:
    with contextlib.suppress(OSError):
        events.write(event, **fields)


def _status_payload(
    config: BackendConfig,
    owner: ProcessIdentity,
    status: str,
    *,
    child: ChildHandle | None = None,
    paused_container: ContainerIdentity | None = None,
    original_restored: bool | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "updated_at": utc_now(),
        "status": status,
        "endpoint": config.endpoint,
        "owner": owner.receipt(),
        "child": child.identity.receipt() if child is not None else None,
        "pins": {
            "server": str(config.server),
            "model": str(config.model),
            "model_size": EXPECTED_MODEL_SIZE,
            "model_sha256": EXPECTED_MODEL_SHA256,
            "model_name": config.model_name,
            "context": config.context,
            "artifact_receipt": str(config.artifact_receipt),
            "source_config_sha256": config.source_sha256,
            "api_key_file": str(config.api_key_file),
            "gateway_default_output_tokens": EXPECTED_DEFAULT_OUTPUT_TOKENS,
            "gateway_maximum_output_tokens": EXPECTED_MAX_OUTPUT_TOKENS,
        },
        "gpu_placement": {
            "policy": "auto_best_fit",
            "actual_gpu_layer_count": None,
            "peak_vram_bytes": None,
            "measured": False,
        },
        "original_container": {
            "name": config.original_container,
            "paused_instance_id": (
                paused_container.container_id if paused_container is not None else None
            ),
            "restored": original_restored,
        },
        "error": error,
    }


def write_status(config: BackendConfig, owner: ProcessIdentity, status: str, **fields: Any) -> None:
    _atomic_json(STATUS_PATH, _status_payload(config, owner, status, **fields))


class ServiceLock:
    def __init__(self, path: Path = LOCK_PATH) -> None:
        self.path = path
        self.handle: Any = None
        self.owner: ProcessIdentity | None = None

    def acquire(self) -> ProcessIdentity:
        descriptor = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o600)
        os.fchmod(descriptor, 0o600)
        self.handle = os.fdopen(descriptor, "a+", encoding="utf-8")
        try:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self.handle.close()
            self.handle = None
            raise ServiceError(f"backend singleton lock is already held: {self.path}") from exc
        owner = process_identity(os.getpid())
        if owner is None or not owner.alive:
            raise ServiceError("could not capture backend supervisor owner identity")
        self.owner = owner
        self._write("held")
        return owner

    def _write(self, status: str) -> None:
        if self.handle is None or self.owner is None:
            return
        receipt = {
            "schema_version": 1,
            "status": status,
            "updated_at": utc_now(),
            "owner": self.owner.receipt(),
        }
        self.handle.seek(0)
        self.handle.truncate()
        self.handle.write(json.dumps(receipt, separators=(",", ":")) + "\n")
        self.handle.flush()
        os.fsync(self.handle.fileno())

    def release(self) -> None:
        if self.handle is None:
            return
        with contextlib.suppress(OSError):
            self._write("released")
        with contextlib.suppress(OSError):
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
        self.handle.close()
        self.handle = None


@contextlib.contextmanager
def defer_termination_signals():
    if not hasattr(signal, "pthread_sigmask"):
        yield
        return
    watched = {signal.SIGINT, signal.SIGTERM}
    previous = signal.pthread_sigmask(signal.SIG_BLOCK, watched)
    try:
        yield
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, previous)


def assert_port_free() -> None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind((EXPECTED_HOST, EXPECTED_PORT))
    except OSError as exc:
        raise ServiceError(f"private backend port {EXPECTED_HOST}:{EXPECTED_PORT} is occupied") from exc
    finally:
        sock.close()


def _docker(*args: str, timeout: float = 15.0) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            ["docker", *args],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ServiceError(f"docker {args[0] if args else 'command'} could not complete") from exc


def inspect_original(config: BackendConfig) -> ContainerIdentity:
    completed = _docker(
        "inspect",
        "--format",
        "{{.Id}}\t{{.Name}}\t{{.State.Running}}\t{{.State.Paused}}",
        config.original_container,
    )
    if completed.returncode != 0:
        raise ServiceError("original Ollama container could not be inspected")
    fields = completed.stdout.strip().split("\t")
    if len(fields) != 4 or not fields[0] or fields[1] != "/" + config.original_container:
        raise ServiceError("original Ollama container returned invalid identity")
    if fields[2] not in {"true", "false"} or fields[3] not in {"true", "false"}:
        raise ServiceError("original Ollama container returned invalid state")
    return ContainerIdentity(
        container_id=fields[0],
        name=fields[1],
        running=fields[2] == "true",
        paused=fields[3] == "true",
    )


def require_original_available(config: BackendConfig, expected_id: str | None = None) -> ContainerIdentity:
    current = inspect_original(config)
    if expected_id is not None and current.container_id != expected_id:
        raise SafetyStop("original Ollama container identity changed")
    if not current.running:
        raise ServiceError("original Ollama container is not running")
    if current.paused:
        raise ServiceError("original Ollama container is already paused; refusing ownership")
    return current


def pause_original(config: BackendConfig) -> None:
    completed = _docker("pause", config.original_container)
    if completed.returncode != 0:
        raise ServiceError("failed to pause original Ollama container")


def pause_original_transaction(
    config: BackendConfig,
    original: ContainerIdentity,
) -> ContainerIdentity:
    """Pause and recover ownership when Docker reports an ambiguous command failure."""
    problem: ServiceError | None = None
    try:
        pause_original(config)
    except ServiceError as exc:
        problem = exc
    for attempt in range(3):
        try:
            observed = inspect_original(config)
        except ServiceError:
            if attempt < 2:
                time.sleep(0.2)
                continue
            if problem is not None:
                raise problem
            raise
        if observed.container_id != original.container_id:
            raise SafetyStop("original Ollama container identity changed during pause")
        if observed.running and observed.paused:
            return observed
        if attempt < 2:
            time.sleep(0.2)
            continue
        if problem is not None:
            raise problem
        raise ServiceError("original Ollama pause could not be confirmed")
    raise AssertionError("pause verification loop exhausted")


def restore_original(config: BackendConfig, paused: ContainerIdentity, events: EventLog) -> None:
    last_problem = "unknown"
    for attempt in range(1, RESTORE_ATTEMPTS + 1):
        _safe_event(events, "original_restore_attempt", attempt=attempt)
        try:
            current = inspect_original(config)
            if current.container_id != paused.container_id:
                raise SafetyStop("refusing to unpause a replacement Ollama container")
            if current.running and not current.paused:
                _safe_event(events, "original_restored", attempt=attempt)
                return
            completed = _docker("unpause", config.original_container)
            if completed.returncode != 0:
                last_problem = "docker unpause failed"
            current = inspect_original(config)
            if current.container_id != paused.container_id:
                raise SafetyStop("Ollama container identity changed during restore")
            if current.running and not current.paused:
                _safe_event(events, "original_restored", attempt=attempt)
                return
            last_problem = "container did not report running and unpaused"
        except SafetyStop:
            raise
        except ServiceError as exc:
            last_problem = str(exc)
        if attempt < RESTORE_ATTEMPTS:
            time.sleep(1.0)
    raise ServiceError(f"could not restore original Ollama container: {last_problem}")


def _ollama_request(
    config: BackendConfig,
    method: str,
    path: str,
    *,
    payload: Mapping[str, Any] | None = None,
    timeout: float = 5.0,
) -> Any:
    connection = http.client.HTTPConnection(config.original_host, config.original_port, timeout=timeout)
    body = None if payload is None else json.dumps(payload, separators=(",", ":")).encode("utf-8")
    headers = {"Accept": "application/json"}
    if body is not None:
        headers["Content-Type"] = "application/json"
    try:
        connection.request(method, path, body=body, headers=headers)
        response = connection.getresponse()
        data = response.read(HTTP_LIMIT)
        if response.status < 200 or response.status >= 300:
            raise ServiceError(f"original Ollama API returned HTTP {response.status}")
        return json.loads(data) if data else {}
    except (OSError, TimeoutError, http.client.HTTPException, json.JSONDecodeError) as exc:
        raise ServiceError("original Ollama API request failed") from exc
    finally:
        connection.close()


def loaded_model_names(config: BackendConfig) -> tuple[str, ...]:
    payload = _ollama_request(config, "GET", "/api/ps")
    if not isinstance(payload, dict) or not isinstance(payload.get("models"), list):
        raise ServiceError("original Ollama /api/ps returned invalid data")
    names: list[str] = []
    for row in payload["models"]:
        if not isinstance(row, dict):
            raise ServiceError("original Ollama /api/ps returned an invalid model row")
        candidates = [row.get(key) for key in ("name", "model") if row.get(key) is not None]
        if not candidates or any(not isinstance(value, str) or not value for value in candidates):
            raise ServiceError("original Ollama /api/ps returned an unnamed model")
        if len(set(candidates)) != 1:
            raise ServiceError("original Ollama /api/ps returned inconsistent model identity")
        names.append(candidates[0])
    return tuple(names)


def gpu_compute_processes(timeout: float = 10.0) -> list[dict[str, Any]]:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ServiceError("could not inspect GPU compute processes") from exc
    if completed.returncode != 0:
        raise ServiceError("nvidia-smi could not inspect GPU compute processes")
    processes: list[dict[str, Any]] = []
    for line in completed.stdout.splitlines():
        if not line.strip():
            continue
        fields = [field.strip() for field in line.rsplit(",", 2)]
        if len(fields) != 3:
            raise ServiceError("nvidia-smi returned an invalid compute-process row")
        try:
            processes.append({"pid": int(fields[0]), "name": fields[1], "used_mib": int(fields[2])})
        except ValueError as exc:
            raise ServiceError("nvidia-smi returned non-numeric compute-process data") from exc
    return processes


def assert_gpu_idle(config: BackendConfig) -> None:
    busy = [item for item in gpu_compute_processes() if item["used_mib"] > config.gpu_limit_mib]
    if busy:
        summary = ", ".join(f"pid {item['pid']} ({item['used_mib']} MiB)" for item in busy)
        raise ServiceError(f"GPU has a foreign compute process over {config.gpu_limit_mib} MiB: {summary}")


def ensure_original_idle(config: BackendConfig, signals: SignalState, events: EventLog) -> None:
    names = loaded_model_names(config)
    if any(name != config.original_model for name in names):
        raise SafetyStop("an unknown model is loaded in original Ollama; refusing to unload anything")
    if not names:
        assert_gpu_idle(config)
        return
    _safe_event(events, "original_model_unload_requested", model=config.original_model)
    _ollama_request(
        config,
        "POST",
        "/api/generate",
        payload={"model": config.original_model, "keep_alive": 0, "stream": False},
        timeout=15.0,
    )
    deadline = time.monotonic() + UNLOAD_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        signals.check()
        names = loaded_model_names(config)
        if any(name != config.original_model for name in names):
            raise SafetyStop("an unknown model appeared during original-model unload")
        busy = [item for item in gpu_compute_processes() if item["used_mib"] > config.gpu_limit_mib]
        if not names and not busy:
            _safe_event(events, "gpu_idle")
            return
        time.sleep(0.5)
    raise ServiceError("timed out waiting for original model and GPU to become idle")


def server_environment(config: BackendConfig) -> dict[str, str]:
    environment = dict(os.environ)
    environment["LD_LIBRARY_PATH"] = os.pathsep.join(str(path) for path in config.lib_dirs)
    environment["GGML_BACKEND_PATH"] = str(CUDA_BACKEND)
    return environment


def server_command(config: BackendConfig) -> list[str]:
    return [
        str(config.server),
        "--model", str(config.model),
        "--alias", config.model_name,
        "--ctx-size", str(config.context),
        "--host", EXPECTED_HOST,
        "--port", str(EXPECTED_PORT),
        *config.args,
    ]


def _open_server_log() -> BinaryIO:
    descriptor = os.open(SERVER_LOG_PATH, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    os.fchmod(descriptor, 0o600)
    handle = os.fdopen(descriptor, "ab", buffering=0)
    handle.write(f"=== LFM backend start {utc_now()} ===\n".encode("utf-8"))
    return handle


def start_child(config: BackendConfig) -> ChildHandle:
    command = server_command(config)
    log_handle = _open_server_log()
    try:
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            cwd=str(config.server.parent),
            env=server_environment(config),
            start_new_session=True,
            close_fds=True,
        )
        deadline = time.monotonic() + 5.0
        identity: ProcessIdentity | None = None
        while time.monotonic() < deadline and process.poll() is None:
            identity = process_identity(process.pid)
            if identity is not None and identity.argv == tuple(command) and identity.pgid == process.pid:
                break
            time.sleep(0.02)
        if identity is None or not identity.alive or identity.argv != tuple(command) or identity.pgid != process.pid:
            if process.poll() is not None:
                raise ServiceError(f"llama-server exited during launch with status {process.returncode}")
            raise SafetyStop("could not capture exact llama-server child identity")
        return ChildHandle(process=process, identity=identity, log_handle=log_handle)
    except BaseException:
        log_handle.close()
        raise


def _backend_json(path: str, api_token: str, timeout: float = 2.0) -> Any:
    connection = http.client.HTTPConnection(EXPECTED_HOST, EXPECTED_PORT, timeout=timeout)
    try:
        connection.request(
            "GET",
            path,
            headers={"Accept": "application/json", "Authorization": f"Bearer {api_token}"},
        )
        response = connection.getresponse()
        data = response.read(HTTP_LIMIT)
        if response.status < 200 or response.status >= 300:
            raise ServiceError(f"backend health endpoint returned HTTP {response.status}")
        return json.loads(data) if data else {}
    except (OSError, TimeoutError, http.client.HTTPException, json.JSONDecodeError) as exc:
        raise ServiceError("backend health request failed") from exc
    finally:
        connection.close()


def _model_ids(payload: Any) -> tuple[str, ...]:
    if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
        return ()
    return tuple(
        row["id"]
        for row in payload["data"]
        if isinstance(row, dict) and isinstance(row.get("id"), str)
    )


def wait_ready(config: BackendConfig, child: ChildHandle, signals: SignalState) -> None:
    deadline = time.monotonic() + config.startup_timeout
    last_problem = "not ready"
    api_token = read_api_token(config.api_key_file)
    while time.monotonic() < deadline:
        signals.check()
        returncode = child.process.poll()
        if returncode is not None:
            raise UnexpectedChildExit(f"llama-server exited during startup with status {returncode}")
        try:
            health = _backend_json("/health", api_token)
            models = _backend_json("/v1/models", api_token)
            if isinstance(health, dict) and health.get("status") == "ok" and config.model_name in _model_ids(models):
                if child.process.poll() is not None:
                    raise UnexpectedChildExit("llama-server exited as health became ready")
                return
            last_problem = "health or model identity did not match"
        except ServiceError as exc:
            last_problem = str(exc)
        time.sleep(0.5)
    raise ServiceError(f"llama-server did not become ready within {int(config.startup_timeout)}s ({last_problem})")


def _identity_matches(expected: ProcessIdentity, observed: ProcessIdentity | None) -> bool:
    return (
        observed is not None
        and observed.alive
        and observed.pid == expected.pid
        and observed.pgid == expected.pgid == expected.pid
        and observed.start_ticks == expected.start_ticks
        and observed.argv == expected.argv
    )


def stop_owned_child(child: ChildHandle, signum: int, grace: float) -> None:
    if child.stopped or child.process.poll() is not None:
        child.stopped = True
        return
    if not _identity_matches(child.identity, process_identity(child.identity.pid)):
        raise SafetyStop("refusing to signal llama-server after child identity validation failed")
    try:
        os.killpg(child.identity.pgid, signum)
    except ProcessLookupError:
        child.process.poll()
        child.stopped = True
        return
    deadline = time.monotonic() + grace
    while child.process.poll() is None and time.monotonic() < deadline:
        time.sleep(0.1)
    if child.process.poll() is None:
        if not _identity_matches(child.identity, process_identity(child.identity.pid)):
            raise SafetyStop("refusing SIGKILL after llama-server child identity changed")
        try:
            os.killpg(child.identity.pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            child.process.wait(timeout=5.0)
        except subprocess.TimeoutExpired as exc:
            raise ServiceError("owned llama-server process did not exit after SIGKILL") from exc
    child.stopped = True


def monitor_child(child: ChildHandle, signals: SignalState, grace: float) -> None:
    while True:
        if signals.received is not None:
            stop_owned_child(child, signals.received, grace)
            raise ServiceInterrupted(signals.received)
        returncode = child.process.poll()
        if returncode is not None:
            child.stopped = True
            raise UnexpectedChildExit(f"llama-server exited unexpectedly with status {returncode}")
        time.sleep(0.25)


def run_backend(
    config: BackendConfig,
    owner: ProcessIdentity,
    signals: SignalState,
    events: EventLog,
) -> int:
    child: ChildHandle | None = None
    paused: ContainerIdentity | None = None
    restored: bool | None = None
    outcome = 1
    terminal_status = "failed"
    terminal_error: str | None = None
    clean_signal = False
    try:
        write_status(config, owner, "preflight", original_restored=None)
        _safe_event(events, "preflight_started")
        digest = verify_runtime_files(config, signals)
        assert_competing_supervisor_inactive(config.competing_receipt)
        assert_port_free()
        original = require_original_available(config)
        ensure_original_idle(config, signals, events)
        signals.check()
        _safe_event(events, "artifact_verified", sha256=digest, size=EXPECTED_MODEL_SIZE)

        with defer_termination_signals():
            assert_competing_supervisor_inactive(config.competing_receipt)
            assert_port_free()
            require_original_available(config, original.container_id)
            if loaded_model_names(config):
                raise ServiceError("a model reloaded in original Ollama before pause")
            assert_gpu_idle(config)
            paused = pause_original_transaction(config, original)

        signals.check()
        assert_gpu_idle(config)
        assert_port_free()
        assert_competing_supervisor_inactive(config.competing_receipt)
        _safe_event(events, "original_paused", container=config.original_container, container_id=original.container_id)
        write_status(config, owner, "starting", paused_container=paused, original_restored=False)

        with defer_termination_signals():
            child = start_child(config)
        _safe_event(events, "backend_started", child_pid=child.identity.pid, child_start_ticks=child.identity.start_ticks)
        write_status(config, owner, "starting", child=child, paused_container=paused, original_restored=False)
        wait_ready(config, child, signals)
        _safe_event(events, "backend_ready", endpoint=config.endpoint, model=config.model_name, gpu_placement="unknown")
        write_status(config, owner, "ready", child=child, paused_container=paused, original_restored=False)
        monitor_child(child, signals, config.shutdown_grace)
        raise AssertionError("monitor_child returned unexpectedly")
    except ServiceInterrupted as exc:
        clean_signal = exc.signum == signal.SIGTERM
        outcome = 0 if clean_signal else 128 + exc.signum
        terminal_status = "stopped" if clean_signal else "interrupted"
        terminal_error = None if clean_signal else f"signal {exc.signum}"
        _safe_event(events, "shutdown_requested", signal=exc.signum)
    except (ServiceError, OSError, subprocess.SubprocessError) as exc:
        outcome = 1
        terminal_status = "failed"
        terminal_error = str(exc)
        _safe_event(events, "backend_failed", error_type=type(exc).__name__, error=str(exc))
    finally:
        if child is not None:
            try:
                with defer_termination_signals():
                    stop_owned_child(child, signals.received or signal.SIGTERM, config.shutdown_grace)
                _safe_event(events, "backend_stopped", child_pid=child.identity.pid)
            except (ServiceError, OSError, subprocess.SubprocessError) as exc:
                outcome = 1
                terminal_status = "failed"
                terminal_error = f"child cleanup failed: {exc}"
                _safe_event(events, "backend_cleanup_failed", error_type=type(exc).__name__, error=str(exc))
            finally:
                child.log_handle.close()
        if paused is not None:
            try:
                with defer_termination_signals():
                    restore_original(config, paused, events)
                restored = True
            except (ServiceError, OSError, subprocess.SubprocessError) as exc:
                restored = False
                outcome = 1
                terminal_status = "failed"
                terminal_error = f"original container restore failed: {exc}"
                _safe_event(events, "original_restore_failed", error_type=type(exc).__name__, error=str(exc))
        if paused is None:
            restored = None
        try:
            write_status(
                config,
                owner,
                terminal_status,
                child=child,
                paused_container=paused,
                original_restored=restored,
                error=terminal_error,
            )
        except OSError as exc:
            outcome = 1
            _safe_event(events, "status_write_failed", error_type=type(exc).__name__)
        _safe_event(events, "service_stopped", returncode=outcome, clean_sigterm=clean_signal, original_restored=restored)
    return outcome


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    ensure_state_dir()
    lock = ServiceLock()
    signals = SignalState()
    events = EventLog()
    owner: ProcessIdentity | None = None
    try:
        owner = lock.acquire()
        signals.install()
        config = load_config(args.config)
        _safe_event(events, "service_started", owner_pid=owner.pid, owner_start_ticks=owner.start_ticks)
        return run_backend(config, owner, signals, events)
    except ServiceError as exc:
        print(f"run_backend: {exc}", file=sys.stderr)
        _safe_event(events, "service_refused", error_type=type(exc).__name__, error=str(exc))
        if owner is not None:
            with contextlib.suppress(OSError, ServiceError):
                config = load_config(args.config)
                write_status(config, owner, "failed", error=str(exc))
        return 2
    except ServiceInterrupted as exc:
        return 0 if exc.signum == signal.SIGTERM else 128 + exc.signum
    finally:
        signals.restore()
        lock.release()


if __name__ == "__main__":
    raise SystemExit(main())
