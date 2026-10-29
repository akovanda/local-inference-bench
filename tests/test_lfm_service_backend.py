from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import signal
import sys
from pathlib import Path
from unittest import mock

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "deployment/local-general/run_backend.py"
CONFIG_PATH = ROOT / "deployment/local-general/lfm-backend.json"
SPEC = importlib.util.spec_from_file_location("lfm_service_backend", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
backend = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = backend
SPEC.loader.exec_module(backend)


def make_config(tmp_path: Path, **overrides):
    values = {
        "source_path": tmp_path / "config.json",
        "server": tmp_path / "llama-server",
        "model": tmp_path / "model.gguf",
        "model_name": backend.EXPECTED_MODEL_NAME,
        "context": backend.EXPECTED_CONTEXT,
        "lib_dirs": (tmp_path / "lib", tmp_path / "cuda"),
        "args": backend.EXPECTED_ARGS,
        "artifact_receipt": tmp_path / "artifact.json",
        "source_config": tmp_path / "source.json",
        "source_sha256": "0" * 64,
        "competing_receipt": tmp_path / "launch.json",
        "original_container": backend.EXPECTED_ORIGINAL_CONTAINER,
        "original_model": backend.EXPECTED_ORIGINAL_MODEL,
        "original_host": backend.EXPECTED_ORIGINAL_HOST,
        "original_port": backend.EXPECTED_ORIGINAL_PORT,
        "api_key_file": tmp_path / "backend-token",
        "startup_timeout": 30.0,
        "shutdown_grace": 5.0,
        "gpu_limit_mib": backend.EXPECTED_GPU_LIMIT_MIB,
    }
    values.update(overrides)
    return backend.BackendConfig(**values)


def owner_identity() -> backend.ProcessIdentity:
    return backend.ProcessIdentity(
        pid=101,
        pgid=101,
        start_ticks=202,
        argv=("python", "run_backend.py"),
        state="S",
    )


def child_handle(returncode=None) -> backend.ChildHandle:
    process = mock.Mock()
    process.pid = 303
    process.poll.return_value = returncode
    identity = backend.ProcessIdentity(
        pid=303,
        pgid=303,
        start_ticks=404,
        argv=("/pinned/llama-server", "--port", "18082"),
        state="S",
    )
    return backend.ChildHandle(process=process, identity=identity, log_handle=io.BytesIO())


def write_artifact_receipt(config: backend.BackendConfig, expected_digest: str) -> None:
    identity = backend._stat_identity(config.model.stat())
    config.artifact_receipt.write_text(
        json.dumps(
            {
                "immutable": True,
                "model": {"alias": config.model_name},
                "shards": [
                    {
                        "name": config.model.name,
                        "path": str(config.model),
                        "size": config.model.stat().st_size,
                        "sha256": expected_digest,
                        "verified": True,
                        "full_sha256_verified": True,
                    }
                ],
                "verdict": {"artifact_complete": True, "full_sha256_verified": True},
                "verification": {
                    "actual_sha256": expected_digest,
                    "expected_sha256": expected_digest,
                    "file_identity_after_verification": identity,
                },
            }
        ),
        encoding="utf-8",
    )


def test_pinned_config_and_launch_command_contract():
    config = backend.load_config(CONFIG_PATH)
    command = backend.server_command(config)

    assert config.endpoint == "http://127.0.0.1:18082/v1"
    assert command[:3] == [str(backend.EXPECTED_SERVER), "--model", str(backend.EXPECTED_MODEL)]
    assert command[command.index("--ctx-size") + 1] == "73728"
    assert command[command.index("--host") + 1] == "127.0.0.1"
    assert command[command.index("--port") + 1] == "18082"
    assert command[command.index("--fit-ctx") + 1] == "73728"
    assert command[command.index("--reasoning") + 1] == "on"
    assert command[command.index("--reasoning-effort") + 1] == "medium"
    assert command[command.index("--reasoning-format") + 1] == "deepseek"
    assert command[command.index("--api-key-file") + 1] == str(backend.EXPECTED_API_KEY_FILE)
    assert command[command.index("--temp") + 1] == "0"
    for option in ("--jinja", "--no-webui", "--no-slots", "--no-context-shift", "--metrics"):
        assert option in command
    environment = backend.server_environment(config)
    assert environment["LD_LIBRARY_PATH"] == ":".join(str(path) for path in backend.EXPECTED_LIB_DIRS)
    assert environment["GGML_BACKEND_PATH"] == str(backend.CUDA_BACKEND)


def test_api_key_file_requires_private_owned_regular_nonempty_token(tmp_path):
    key_file = tmp_path / "backend-token"
    key_file.write_bytes(b"fixture-token\n")
    key_file.chmod(0o600)
    assert backend.read_api_token(key_file) == "fixture-token"

    key_file.chmod(0o640)
    with pytest.raises(backend.ServiceError, match="exactly 0600"):
        backend.read_api_token(key_file)

    key_file.write_bytes(b"")
    key_file.chmod(0o600)
    with pytest.raises(backend.ServiceError, match="empty"):
        backend.read_api_token(key_file)

    target = tmp_path / "target-token"
    target.write_bytes(b"fixture-token")
    target.chmod(0o600)
    key_file.unlink()
    key_file.symlink_to(target)
    with pytest.raises(backend.ServiceError, match="non-symlink"):
        backend.read_api_token(key_file)


def test_backend_readiness_request_uses_bearer_auth_without_token_in_errors(monkeypatch):
    response = mock.Mock(status=200)
    response.read.return_value = b'{"status":"ok"}'
    connection = mock.Mock()
    connection.getresponse.return_value = response
    monkeypatch.setattr(backend.http.client, "HTTPConnection", mock.Mock(return_value=connection))

    assert backend._backend_json("/health", "fixture-token") == {"status": "ok"}
    connection.request.assert_called_once_with(
        "GET",
        "/health",
        headers={"Accept": "application/json", "Authorization": "Bearer fixture-token"},
    )

    response.status = 401
    with pytest.raises(backend.ServiceError) as caught:
        backend._backend_json("/v1/models", "fixture-token")
    assert "fixture-token" not in str(caught.value)


def test_artifact_requires_receipt_identity_and_fresh_full_sha256(tmp_path, monkeypatch):
    original = b"verified tiny fixture"
    expected_digest = hashlib.sha256(original).hexdigest()
    config = make_config(tmp_path)
    config.model.write_bytes(original)
    monkeypatch.setattr(backend, "EXPECTED_MODEL_SIZE", len(original))
    monkeypatch.setattr(backend, "EXPECTED_MODEL_SHA256", expected_digest)
    write_artifact_receipt(config, expected_digest)

    assert backend.verify_artifact(config) == expected_digest

    config.model.write_bytes(b"X" * len(original))
    write_artifact_receipt(config, expected_digest)
    with pytest.raises(backend.ServiceError, match="full SHA-256"):
        backend.verify_artifact(config)


def test_active_competing_supervisor_is_refused_only_on_exact_live_identity(tmp_path, monkeypatch):
    argv = ["python", "public_model_expansion.py", "supervise"]
    receipt = tmp_path / "launch.json"
    receipt.write_text(
        json.dumps({"process": {"pid": 77, "start_ticks": 88, "argv": argv}}),
        encoding="utf-8",
    )
    exact = backend.ProcessIdentity(77, 77, 88, tuple(argv), "S")
    monkeypatch.setattr(backend, "process_identity", lambda _pid: exact)
    with pytest.raises(backend.ServiceError, match="still active"):
        backend.assert_competing_supervisor_inactive(receipt)

    stale = backend.ProcessIdentity(77, 77, 89, tuple(argv), "S")
    monkeypatch.setattr(backend, "process_identity", lambda _pid: stale)
    backend.assert_competing_supervisor_inactive(receipt)


def test_unknown_loaded_model_refuses_without_unload_or_gpu_mutation(tmp_path, monkeypatch):
    config = make_config(tmp_path)
    signals = backend.SignalState()
    events = backend.EventLog(tmp_path / "events.jsonl")
    monkeypatch.setattr(backend, "loaded_model_names", lambda _config: ("not-owned:latest",))
    request = mock.Mock()
    gpu = mock.Mock()
    monkeypatch.setattr(backend, "_ollama_request", request)
    monkeypatch.setattr(backend, "assert_gpu_idle", gpu)

    with pytest.raises(backend.SafetyStop, match="unknown model"):
        backend.ensure_original_idle(config, signals, events)

    request.assert_not_called()
    gpu.assert_not_called()


def test_foreign_gpu_process_above_limit_is_refused(tmp_path, monkeypatch):
    config = make_config(tmp_path)
    monkeypatch.setattr(
        backend,
        "gpu_compute_processes",
        lambda: [{"pid": 909, "name": "foreign", "used_mib": 257}],
    )
    with pytest.raises(backend.ServiceError, match="foreign compute process"):
        backend.assert_gpu_idle(config)


def test_container_inspection_uses_exact_identity_and_tab_delimited_state(tmp_path, monkeypatch):
    config = make_config(tmp_path)
    completed = mock.Mock(
        returncode=0,
        stdout="sha256:container\t/local-codex-ollama-cpu\ttrue\tfalse\n",
    )
    docker = mock.Mock(return_value=completed)
    monkeypatch.setattr(backend, "_docker", docker)

    identity = backend.inspect_original(config)

    assert identity == backend.ContainerIdentity(
        "sha256:container", "/local-codex-ollama-cpu", True, False
    )
    assert "\t" in docker.call_args.args[2]


def test_singleton_flock_persists_owner_identity_receipt(tmp_path, monkeypatch):
    identity = backend.ProcessIdentity(
        pid=123,
        pgid=123,
        start_ticks=456,
        argv=("python", "run_backend.py"),
        state="S",
    )
    monkeypatch.setattr(backend, "process_identity", lambda _pid: identity)
    lock_path = tmp_path / "backend.lock"
    first = backend.ServiceLock(lock_path)
    second = backend.ServiceLock(lock_path)
    try:
        assert first.acquire() == identity
        receipt = json.loads(lock_path.read_text(encoding="utf-8"))
        assert receipt["status"] == "held"
        assert receipt["owner"]["pid"] == 123
        assert receipt["owner"]["start_ticks"] == 456
        with pytest.raises(backend.ServiceError, match="already held"):
            second.acquire()
    finally:
        second.release()
        first.release()


def test_ambiguous_pause_failure_is_owned_when_exact_container_reports_paused(tmp_path, monkeypatch):
    config = make_config(tmp_path)
    original = backend.ContainerIdentity("same-id", "/local-codex-ollama-cpu", True, False)
    paused = backend.ContainerIdentity("same-id", "/local-codex-ollama-cpu", True, True)
    monkeypatch.setattr(
        backend,
        "pause_original",
        mock.Mock(side_effect=backend.ServiceError("docker pause could not complete")),
    )
    monkeypatch.setattr(backend, "inspect_original", lambda _config: paused)

    assert backend.pause_original_transaction(config, original) == paused


def test_startup_failure_stops_owned_child_before_restoring_pause(tmp_path, monkeypatch):
    config = make_config(tmp_path)
    signals = backend.SignalState()
    events = backend.EventLog(tmp_path / "events.jsonl")
    original = backend.ContainerIdentity("container-id", "/local-codex-ollama-cpu", True, False)
    paused = backend.ContainerIdentity("container-id", "/local-codex-ollama-cpu", True, True)
    child = child_handle()
    order: list[str] = []

    monkeypatch.setattr(backend, "write_status", lambda *args, **kwargs: None)
    monkeypatch.setattr(backend, "verify_runtime_files", lambda *_args: backend.EXPECTED_MODEL_SHA256)
    monkeypatch.setattr(backend, "assert_competing_supervisor_inactive", lambda *_args: None)
    monkeypatch.setattr(backend, "assert_port_free", lambda: None)
    monkeypatch.setattr(backend, "require_original_available", lambda *_args: original)
    monkeypatch.setattr(backend, "ensure_original_idle", lambda *_args: None)
    monkeypatch.setattr(backend, "loaded_model_names", lambda *_args: ())
    monkeypatch.setattr(backend, "assert_gpu_idle", lambda *_args: None)
    monkeypatch.setattr(backend, "pause_original", lambda *_args: order.append("pause"))
    monkeypatch.setattr(backend, "inspect_original", lambda *_args: paused)
    monkeypatch.setattr(backend, "start_child", lambda *_args: child)
    monkeypatch.setattr(
        backend,
        "wait_ready",
        mock.Mock(side_effect=backend.ServiceError("startup health failed")),
    )

    def stop(*_args):
        order.append("stop")
        child.stopped = True

    def restore(*_args):
        order.append("restore")

    monkeypatch.setattr(backend, "stop_owned_child", stop)
    monkeypatch.setattr(backend, "restore_original", restore)

    result = backend.run_backend(config, owner_identity(), signals, events)

    assert result == 1
    assert order == ["pause", "stop", "restore"]
    assert child.log_handle.closed


def test_signal_race_after_owned_pause_restores_without_starting_child(tmp_path, monkeypatch):
    config = make_config(tmp_path)
    signals = backend.SignalState()
    events = backend.EventLog(tmp_path / "events.jsonl")
    original = backend.ContainerIdentity("container-id", "/local-codex-ollama-cpu", True, False)
    paused = backend.ContainerIdentity("container-id", "/local-codex-ollama-cpu", True, True)
    order: list[str] = []

    monkeypatch.setattr(backend, "write_status", lambda *args, **kwargs: None)
    monkeypatch.setattr(backend, "verify_runtime_files", lambda *_args: backend.EXPECTED_MODEL_SHA256)
    monkeypatch.setattr(backend, "assert_competing_supervisor_inactive", lambda *_args: None)
    monkeypatch.setattr(backend, "assert_port_free", lambda: None)
    monkeypatch.setattr(backend, "require_original_available", lambda *_args: original)
    monkeypatch.setattr(backend, "ensure_original_idle", lambda *_args: None)
    monkeypatch.setattr(backend, "loaded_model_names", lambda *_args: ())
    monkeypatch.setattr(backend, "assert_gpu_idle", lambda *_args: None)
    monkeypatch.setattr(backend, "inspect_original", lambda *_args: paused)

    def pause(*_args):
        order.append("pause")
        signals.received = signal.SIGTERM

    start = mock.Mock()
    monkeypatch.setattr(backend, "pause_original", pause)
    monkeypatch.setattr(backend, "start_child", start)
    monkeypatch.setattr(backend, "restore_original", lambda *_args: order.append("restore"))

    result = backend.run_backend(config, owner_identity(), signals, events)

    assert result == 0
    assert order == ["pause", "restore"]
    start.assert_not_called()


def test_owned_stop_validates_identity_before_forwarding_signal(monkeypatch):
    child = child_handle()
    polls = iter((None, None, 0, 0))
    child.process.poll.side_effect = lambda: next(polls, 0)
    monkeypatch.setattr(backend, "process_identity", lambda _pid: child.identity)
    sent: list[tuple[int, int]] = []
    monkeypatch.setattr(backend.os, "killpg", lambda pgid, signum: sent.append((pgid, signum)))
    monkeypatch.setattr(backend.time, "sleep", lambda _seconds: None)

    backend.stop_owned_child(child, signal.SIGTERM, 5.0)

    assert sent == [(303, signal.SIGTERM)]
    assert child.stopped is True

    other = child_handle()
    wrong = backend.ProcessIdentity(303, 303, 405, other.identity.argv, "S")
    monkeypatch.setattr(backend, "process_identity", lambda _pid: wrong)
    sent.clear()
    with pytest.raises(backend.SafetyStop, match="identity validation failed"):
        backend.stop_owned_child(other, signal.SIGTERM, 5.0)
    assert sent == []


def test_unexpected_child_exit_is_nonzero_restart_signal():
    child = child_handle(returncode=0)
    with pytest.raises(backend.UnexpectedChildExit, match="unexpectedly"):
        backend.monitor_child(child, backend.SignalState(), 5.0)
