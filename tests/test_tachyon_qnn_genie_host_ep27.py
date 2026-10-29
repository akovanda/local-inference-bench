from __future__ import annotations

from runtime import tachyon_qnn_genie_host_ep27 as ep27


def test_latest_handoff_is_separate_pinned_and_preflighted() -> None:
    handoff = ep27.runtime_handoff()
    assert handoff["variant"] == "latest_ep27"
    assert handoff["image"]["run_reference"] == ep27.IMAGE_DIGEST
    assert handoff["runtime"]["onnxruntime"] == "1.31.0"
    assert handoff["runtime"]["onnxruntime_qnn"] == "2.7.0"
    assert handoff["runtime"]["embedded_htp_build"] == "v2.51.0.260929181845"
    assert handoff["runtime"]["wheel"]["sha256"] == ep27.WHEEL["sha256"]
    assert handoff["runtime"]["onnxruntime_wheel"]["sha256"] == ep27.ORT_WHEEL["sha256"]
    assert handoff["dependency_preflight"]["status"] == "passed"
    assert handoff["dependency_preflight"]["image_digest"] == ep27.IMAGE_DIGEST
    assert handoff["dependency_preflight"]["unresolved"] == []


def test_latest_handoff_binds_linux_registration_workaround_without_claiming_htp() -> None:
    handoff = ep27.runtime_handoff()
    assert handoff["environment"]["ORT_QNN_ENABLE_CPU_BACKEND"] == "1"
    assert "ORT_QNN_ENABLE_CPU_BACKEND=1" in handoff["docker_argv_template"]
    registration = handoff["dependency_preflight"]["provider_registration"]
    assert registration["status"] == "passed"
    assert registration["advertised_device_type"] == "OrtHardwareDeviceType.CPU"
    assert registration["backend_execution"] == "not_tested_controller_owned"
    assert "QNN profiling > 0" in handoff["linux_npu_discovery"]["controller_requirement"]


def test_latest_runtime_preserves_locked_down_container_contract() -> None:
    argv = ep27.runtime_handoff()["docker_argv_template"]
    assert "--privileged" not in argv
    assert argv[argv.index("--network") + 1] == "none"
    assert argv[argv.index("--cap-drop") + 1] == "ALL"
    assert "no-new-privileges" in argv
    assert argv[-2:] == [ep27.IMAGE_DIGEST, "COMMAND"]
