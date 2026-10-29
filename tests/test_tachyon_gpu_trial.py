import argparse
import json
from pathlib import Path
import subprocess
import sys
from unittest import mock

import pytest

from runtime import tachyon_gpu_trial as trial


def args(tmp_path):
    return argparse.Namespace(
        study=tmp_path / "study",
        trial_dir=tmp_path / "trial",
        remote="akovanda@tachyon",
        endpoint=trial.ENDPOINT,
        wait_timeout=1,
        poll=0.001,
        variant_timeout=600,
        server_timeout=600,
        request_timeout=900,
    )


def test_direct_script_cli_entrypoint_works_without_package_mode():
    completed = subprocess.run(
        [sys.executable, str(trial.ROOT / "runtime/tachyon_gpu_trial.py"), "--help"],
        cwd=trial.ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--variant-timeout" in completed.stdout


def test_commands_are_exact_and_never_autofit_or_full_offload():
    cpu8, gpu4, gpu8 = trial.VARIANTS
    cpu = trial.bench_command(cpu8, device_supported=True)
    assert cpu[:3] == ["taskset", "-c", "0-7"]
    assert trial.flag_value(cpu, "-t") == "8"
    assert trial.flag_value(cpu, "-ngl") == "0"
    assert "--device" not in cpu

    for variant, layers in ((gpu4, "4"), (gpu8, "8")):
        command = trial.bench_command(variant, device_supported=True)
        assert command[:2] == ["env", f"OCL_ICD_VENDORS={trial.REMOTE_ICD}"]
        assert trial.flag_value(command, "-ngl") == layers
        assert trial.flag_value(command, "--device") == "GPUOpenCL"
        assert "-v" in command
        assert "--fit" not in command and "--fit-target" not in command
        assert "99" not in command

    server = trial.server_command(
        backend="opencl", ngl=4, device_supported=True, fit_supported=True
    )
    assert trial.flag_value(server, "--fit") == "off"
    assert trial.flag_value(server, "--device") == "GPUOpenCL"
    assert trial.flag_value(server, "-ngl") == "4"


def test_handoff_uses_latest_status_and_controller_launch_then_requires_free_lock(tmp_path):
    controller = trial.Controller(args(tmp_path))
    controller.study.mkdir()
    status = {
        "status": "complete",
        "controller_pid": 2962526,
        "manifest_sha256": trial.MANIFEST_SHA256,
    }
    launch = {
        "pid": 2962526,
        "source_sha256": trial.CONTROLLER_SHA256,
        "argv": [
            str(trial.ROOT / ".venv-public-bench/bin/python"),
            "-m",
            "runtime.tachyon_study",
            "--study",
            str(controller.study),
        ],
    }
    (controller.study / "study-status.json").write_text(json.dumps(status))
    (controller.study / "controller-launch.json").write_text(json.dumps(launch))
    (controller.study / ".controller.lock").write_text("2962526\n")
    with mock.patch.object(trial, "process_identity", return_value={"alive": False}), \
         mock.patch.object(trial, "lock_is_free", return_value=True):
        assert controller.wait_for_cpu_study() == status

    launch["pid"] = 1
    (controller.study / "controller-launch.json").write_text(json.dumps(launch))
    with pytest.raises(trial.TrialError, match="PID differs"):
        controller.wait_for_cpu_study()


def test_process_and_cpu_server_guards_reject_identity_or_runtime_changes():
    receipt = {"pid": 4, "start_ticks": 9, "argv": ["/owned", "x"]}
    assert trial.verify_owned_process(
        receipt, {"pid": 4, "start_ticks": 9, "argv": receipt["argv"], "alive": True},
        kind="fixture",
    )
    with pytest.raises(trial.TrialError, match="start-tick"):
        trial.verify_owned_process(
            receipt, {"pid": 4, "start_ticks": 10, "argv": receipt["argv"], "alive": True},
            kind="fixture",
        )
    command = trial.server_command(
        backend="cpu", ngl=0, device_supported=False, fit_supported=False
    )
    trial.validate_cpu_server_argv(command)
    changed = list(command)
    changed[3] = "/tmp/llama-server"
    with pytest.raises(trial.TrialError, match="private pinned runtime"):
        trial.validate_cpu_server_argv(changed)


def test_offload_log_must_report_actual_layers_before_gpu_compute_is_proven():
    proven = trial.parse_offload_evidence(
        "load_tensors: offloading 4 repeating layers to GPU\n"
        "ggml_opencl: using GPUOpenCL\n",
        4,
    )
    assert proven["reported_offloaded_layers"] == 4
    assert proven["gpu_compute_proven"] is True
    assert proven["classification"] == "partial_gpu_offload"

    absent = trial.parse_offload_evidence("OpenCL platform detected\n", 8)
    assert absent["reported_offloaded_layers"] is None
    assert absent["gpu_compute_proven"] is False
    assert absent["classification"] == "gpu_offload_not_proven"


def test_lfm_smoke_payload_binds_native_recipe_without_bonsai_controls():
    payload = trial.smoke_payload(trial.SENSOR_PROMPT, 512, trial.CPU_ALIAS)
    assert payload["model"] == trial.CPU_ALIAS
    assert payload["max_tokens"] == 512
    assert payload["reasoning_budget_tokens"] == 0
    assert payload["temperature"] == 0.1
    assert payload["top_p"] == 1.0
    assert payload["top_k"] == 50
    assert payload["min_p"] == 0.0
    assert payload["repeat_penalty"] == 1.1
    assert payload["reasoning_effort"] == "none"
    assert payload["seed"] == 20261023
    assert "chat_template_kwargs" not in payload
    assert "presence_penalty" not in payload


def test_fast_probe_records_reasoning_final_finish_usage_and_caps(tmp_path):
    controller = trial.Controller(args(tmp_path))
    controller.trial.mkdir()

    class Response:
        lines = [
            b'data: {"choices":[{"delta":{"reasoning_content":"why"}}]}\n',
            b'data: {"choices":[{"delta":{"content":"tachyon-opencl-ok"},"finish_reason":"stop"}]}\n',
            b'data: {"choices":[],"usage":{"completion_tokens":4},"timings":{"predicted_n":4,"predicted_ms":100,"predicted_per_second":30}}\n',
            b"data: [DONE]\n",
        ]

        def __enter__(self):
            return iter(self.lines)

        def __exit__(self, *_args):
            return False

    with mock.patch.object(trial, "urlopen", return_value=Response()):
        evaluation = controller.request_smoke(
            "fixture", "exact-prose", trial.EXACT_PROMPT, 128, trial.GPU_ALIAS
        )
    request = json.loads((controller.trial / "fixture-exact-prose-request.json").read_text())
    response = json.loads((controller.trial / "fixture-exact-prose-response.json").read_text())
    assert request["reasoning_budget_tokens"] == 0
    assert request["max_tokens"] == 128
    assert evaluation["requested_reasoning_budget_tokens"] == 0
    assert evaluation["actual_reasoning_character_count"] == 3
    assert evaluation["actual_final_character_count"] == len("tachyon-opencl-ok")
    assert evaluation["finish_reason"] == "stop"
    assert evaluation["usage"] == {"completion_tokens": 4}
    assert response["reasoning_content"] == "why"
    assert (controller.trial / "fixture-exact-prose-raw-sse.txt").is_file()


def test_terminal_variant_result_resumes_without_remote_job(tmp_path):
    controller = trial.Controller(args(tmp_path))
    controller.trial.mkdir()
    result = {
        "variant": "opencl-ngl4",
        "backend": "opencl",
        "status": "failed",
        "offload": {"gpu_compute_proven": False},
    }
    (controller.trial / "opencl-ngl4-result.json").write_text(json.dumps(result))
    controller.event = mock.Mock()
    controller.remote_json_file = mock.Mock(side_effect=AssertionError("must not access remote"))
    assert controller.run_benchmark(trial.VARIANTS[1]) == result
    controller.remote_json_file.assert_not_called()


def test_summary_labels_smoke_only_and_never_claims_quality(tmp_path):
    controller = trial.Controller(args(tmp_path))
    controller.trial.mkdir()
    result = {
        "variant": "opencl-ngl4",
        "backend": "opencl",
        "status": "complete",
        "prompt_tokens_per_second": 12.0,
        "generation_tokens_per_second": 4.0,
        "offload": {
            "requested_gpu_layers": 4,
            "reported_offloaded_layers": 4,
            "classification": "partial_gpu_offload",
        },
        "error": None,
    }
    controller.consolidate(
        [result],
        {"opencl-ngl4": {"sensor_passed": True, "deterministic_prose_passed": True}},
        {
            "variant": "cpu4-fast-baseline",
            "status": "complete",
            "requested_reasoning_budget_tokens": 0,
            "quality_claim": False,
        },
    )
    summary = json.loads((controller.trial / "summary.json").read_text())
    assert summary["quality_claim"] is False
    assert "smoke-only" in summary["interpretation"]
    assert summary["rows"][0]["quality_claim"] is False
    assert summary["fast_probe_baseline"]["requested_reasoning_budget_tokens"] == 0
