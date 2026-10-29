import argparse
import json
import os
from pathlib import Path
from unittest import mock

import pytest

from runtime import tachyon_study as study


def model(key="bonsai2-2b"):
    return next(dict(row) for row in study.MODELS if row["key"] == key)


def controller_args(root):
    return argparse.Namespace(
        study=root,
        python=str(study.PYTHON),
        remote="akovanda@tachyon",
    )


def terminal_status(path):
    path.mkdir(parents=True, exist_ok=True)
    cases = []
    for index in range(16):
        cases.append(
            {
                "index": index,
                "suite": "humanevalplus" if index < 8 else "ifeval",
                "status": "completed" if index != 15 else "truncated",
            }
        )
    (path / "status.json").write_text(
        json.dumps(
            {
                "cases": cases,
                "counts": {
                    "total": 16,
                    "completed": 15,
                    "truncated": 1,
                    "api_failure": 0,
                    "incomplete": 0,
                    "not_started": 0,
                    "cancelled": 0,
                    "invalid": 0,
                },
                "active": None,
            }
        )
    )
    (path / "run.json").write_text(
        json.dumps(
            {
                "manifest": {"sha256": study.MANIFEST_SHA256},
                "settings": {
                    "model": "tachyon-bonsai2-2b",
                    "model_key": "bonsai2-2b",
                    "original_profile": "bonsai-nothink",
                    "max_tokens": 8192,
                    "endpoint": study.ENDPOINT,
                    "seed": 20261023,
                    "frozen_runner_sha256": study.FROZEN_RUNNER_SHA256,
                },
            }
        )
    )


def test_model_receipt_requires_verified_hash_path_and_current_size():
    entry = model()
    remote_path = (
        study.REMOTE_ROOT
        + "/models/bonsai2-2b/Ternary-Bonsai-2-2B-Q2_0.gguf"
    )
    artifact = {
        **entry,
        "path": remote_path,
        "actual_sha256": entry["sha256"],
        "verified_at": "2026-10-23T15:05:02Z",
    }
    result = study.verify_model_receipt(
        entry, {"artifact": artifact, "path": remote_path, "size": entry["size"]}
    )
    assert result["observed_size"] == entry["size"]
    assert "no redundant full rehash" in result["verification"]

    changed = dict(artifact, actual_sha256="0" * 64)
    with pytest.raises(study.StudyError, match="actual_sha256 differs"):
        study.verify_model_receipt(
            entry, {"artifact": changed, "path": remote_path, "size": entry["size"]}
        )
    with pytest.raises(study.StudyError, match="byte size differs"):
        study.verify_model_receipt(
            entry, {"artifact": artifact, "path": remote_path, "size": 1}
        )


def test_owned_process_guard_checks_birth_and_exact_cmdline_and_treats_zombie_exited():
    receipt = {"pid": 44, "start_ticks": 900, "argv": ["/pinned/bin", "--x"]}
    assert study.verify_owned_process(
        receipt,
        {"pid": 44, "start_ticks": 900, "argv": receipt["argv"], "alive": True},
        kind="fixture",
    )
    with pytest.raises(study.StudyError, match="start-tick"):
        study.verify_owned_process(
            receipt,
            {"pid": 44, "start_ticks": 901, "argv": receipt["argv"], "alive": True},
            kind="fixture",
        )
    with pytest.raises(study.StudyError, match="cmdline"):
        study.verify_owned_process(
            receipt,
            {"pid": 44, "start_ticks": 900, "argv": ["/other"], "alive": True},
            kind="fixture",
        )
    assert not study.verify_owned_process(
        receipt,
        {"pid": 44, "state": "Z", "alive": False, "argv": []},
        kind="fixture",
    )


def test_server_ownership_accepts_expected_taskset_wrapper_but_rejects_runtime_change():
    entry = model()
    command = study.server_command(entry)
    study.validate_server_argv(command, entry)
    assert command[:3] == ["taskset", "-c", "4-7"]
    assert command[3] == study.REMOTE_RUNTIME + "/bin/llama-server"

    malicious = list(command)
    malicious[3] = "/tmp/llama-server"
    with pytest.raises(study.StudyError, match="pinned CPU runtime"):
        study.validate_server_argv(malicious, entry)


def test_adopted_quality_runner_must_match_receipt_and_live_proc_argv():
    entry = model()
    expected = study.quality_command(str(study.PYTHON), study.DEFAULT_STUDY, entry)
    receipt = {
        "pid": study.INITIAL_RUNNER_PID,
        "argv": expected,
        "source_sha256": study.sha256(study.ADAPTER),
    }
    identity = {
        "pid": study.INITIAL_RUNNER_PID,
        "alive": True,
        "state": "S",
        "start_ticks": 123,
        "argv": expected,
    }
    with mock.patch.object(study, "process_identity", return_value=identity):
        assert study.verify_adopted_runner(receipt, expected) == identity

    mismatch = dict(identity, argv=expected + ["--rerun-failed"])
    with mock.patch.object(study, "process_identity", return_value=mismatch):
        with pytest.raises(study.StudyError, match="/proc argv mismatch"):
            study.verify_adopted_runner(receipt, expected)
    with pytest.raises(study.StudyError, match="receipt argv differs"):
        study.verify_adopted_runner(dict(receipt, argv=["python", "wrong.py"]), expected)


def test_grader_command_has_immutable_image_and_required_sandbox_mounts(tmp_path):
    screening = tmp_path / "tachyon-study"
    output = screening / "bonsai2-2b"
    output.mkdir(parents=True)
    (screening / "manifest.json").write_text("{}")
    command = study.grading_command(screening, output)

    assert study.GRADER_IMAGE in command
    assert command[command.index("--network") + 1] == "none"
    assert "--read-only" in command
    assert command[command.index("--cap-drop") + 1] == "ALL"
    assert command[command.index("--security-opt") + 1] == "no-new-privileges"
    assert command[command.index("--cpus") + 1] == "4"
    assert command[command.index("--memory") + 1] == "4g"
    assert "TACHYON_GRADER_SANDBOX=1" in command
    mounts = [command[index + 1] for index, value in enumerate(command) if value == "--mount"]
    assert f"type=bind,src={study.ADAPTER},dst={study.ADAPTER},readonly" in mounts
    assert f"type=bind,src={study.FROZEN_RUNNER},dst={study.FROZEN_RUNNER},readonly" in mounts
    assert f"type=bind,src={screening / 'manifest.json'},dst={screening / 'manifest.json'},readonly" in mounts
    assert f"type=bind,src={study.VENDOR},dst={study.VENDOR},readonly" in mounts
    assert f"type=bind,src={output},dst={output}" in mounts
    assert command[command.index("--ifeval-root") + 1] == str(study.IFEVAL)


def test_controller_lock_is_exclusive(tmp_path):
    first = study.Controller(controller_args(tmp_path))
    second = study.Controller(controller_args(tmp_path))
    first.acquire_lock()
    with pytest.raises(study.StudyError, match="holds the lock"):
        second.acquire_lock()
    assert (tmp_path / ".controller.lock").read_text().strip() == str(os.getpid())
    assert first.lock_stream is not None
    first.lock_stream.close()


def test_provenance_failure_is_persisted_and_returns_nonzero(tmp_path):
    controller = study.Controller(controller_args(tmp_path))
    with mock.patch.object(
        controller, "initialize", side_effect=study.StudyError("fixture provenance mismatch")
    ), mock.patch.object(study.signal, "signal"):
        assert controller.run() == 1
    status = json.loads((tmp_path / "study-status.json").read_text())
    assert status["status"] == "failed"
    assert status["error"] == "fixture provenance mismatch"
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert events[-1]["event"] == "study_failed"


def test_terminal_quality_is_preserved_without_inspecting_or_launching_runner(tmp_path):
    entry = model()
    terminal_status(tmp_path / entry["key"])
    controller = study.Controller(controller_args(tmp_path))
    controller.event = mock.Mock()
    with mock.patch.object(study, "verify_adopted_runner") as verify:
        result = controller.wait_quality(entry)
    assert result["terminal"] is True
    verify.assert_not_called()
    controller.event.assert_called_once_with(
        "quality_preserved", model=entry["key"], terminal_count=16
    )


def test_serial_model_lifecycle_stops_server_before_benchmark_and_restarts_last(tmp_path):
    calls = []
    entry = model()
    receipt = {
        "pid": 22,
        "start_ticks": 33,
        "argv": study.server_command(entry),
        "alias": "tachyon-bonsai2-2b",
    }

    class FakeController(study.Controller):
        def save(self, **fields):
            self.state.update(fields)

        def event(self, name, **fields):
            calls.append(("event", name))

        def observe_model(self, current):
            calls.append(("verify", current["key"]))
            return {"verified": True}

        def current_server(self, current):
            calls.append(("current", current["key"]))
            return None

        def ensure_server(self, current):
            calls.append(("server", current["key"]))
            return receipt

        def run_smoke(self, current):
            calls.append(("smoke", current["key"]))

        def wait_quality(self, current):
            calls.append(("quality", current["key"]))
            return {"terminal": True, "terminal_count": 16, "counts": {"completed": 16}}

        def grade(self, current):
            calls.append(("grade", current["key"]))

        def snapshot(self, current, server):
            calls.append(("snapshot", current["key"]))

        def stop_server(self, current, server):
            calls.append(("stop", current["key"]))

        def benchmark(self, current, server):
            calls.append(("bench", current["key"]))

        def summarize_one(self, current):
            calls.append(("summarize", current["key"]))

    controller = FakeController(controller_args(tmp_path))
    controller.state = {"models": {}}
    controller.run_model(entry, last=True)

    names = [name for name, _ in calls if name != "event"]
    assert names == [
        "verify",
        "current",
        "server",
        "smoke",
        "quality",
        "grade",
        "server",
        "snapshot",
        "stop",
        "bench",
        "summarize",
        "server",
    ]
    assert names.index("stop") < names.index("bench") < len(names) - 1
    assert names[-1] == "server"
    assert controller.state["models"][entry["key"]]["final_server"]["left_running"] is True


def test_benchmark_command_is_cpu_only_and_pinned_to_four_cores():
    entry = model("lfm25-26b")
    command = study.bench_command(entry)
    assert command[:3] == ["taskset", "-c", "4-7"]
    assert command[3] == study.REMOTE_RUNTIME + "/bin/llama-bench"
    assert command[command.index("-p") + 1] == "128"
    assert command[command.index("-n") + 1] == "64"
    assert command[command.index("-t") + 1] == "4"
    assert command[command.index("-r") + 1] == "3"
    assert command[command.index("-ngl") + 1] == "0"
    receipt = {
        "kind": "tachyon_llama_bench",
        "launch_argv": command,
        "argv": command[3:],
    }
    study.validate_bench_receipt(receipt, entry)
    with pytest.raises(study.StudyError, match="launch argv differs"):
        study.validate_bench_receipt({**receipt, "launch_argv": command[:-2]}, entry)


def test_native_reported_rates_are_time_weighted_and_count_rate_stays_separate(tmp_path):
    output = tmp_path / "model"
    cases = []
    for index, (prompt_rate, prompt_ms, prompt_n, generation_rate, generation_ms) in enumerate(
        ((2.0, 1000, 3, 8.0, 1000), (4.0, 3000, 9, 10.0, 3000))
    ):
        storage = f"case-{index}"
        cases.append(
            {"suite": "humanevalplus", "storage_id": storage, "status": "completed"}
        )
        attempt = output / "cases" / "humanevalplus" / storage / "attempt-0001"
        attempt.mkdir(parents=True)
        (attempt / "result.json").write_text(
            json.dumps(
                {
                    "status": "completed",
                    "native_timings": {
                        "prompt_per_second": prompt_rate,
                        "prompt_ms": prompt_ms,
                        "prompt_n": prompt_n,
                        "predicted_per_second": generation_rate,
                        "predicted_ms": generation_ms,
                    },
                }
            )
        )
    (output / "status.json").write_text(json.dumps({"cases": cases}))
    controller = study.Controller(controller_args(tmp_path))
    rates = controller._native_rates(output)
    assert rates["prompt_reported"] == pytest.approx(3.5)
    assert rates["generation_reported"] == pytest.approx(9.5)
    assert rates["prompt_count_time"] == pytest.approx(3.0)
