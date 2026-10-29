from __future__ import annotations

import pytest

from runtime import tachyon_qnn_genie_host as host


PRIVATE = "/external/device-workspace"


def _argv(**overrides: object) -> list[str]:
    values: dict[str, object] = {
        "name": "moe-qnn-lfm350-pilot-20261024",
        "source": f"{PRIVATE}/runtime/lfm350/source",
        "assets": f"{PRIVATE}/runtime/lfm350/assets",
        "state": f"{PRIVATE}/results/lfm350/state",
        "output": f"{PRIVATE}/results/lfm350/output",
        "command": ["/workspace/source/runner/scripts/install.py", "--asset-dir", "/workspace/assets"],
    }
    values.update(overrides)
    return host.build_lfm350_docker_argv(**values)  # type: ignore[arg-type]


def test_private_argv_is_nonprivileged_networkless_and_has_mount_modes() -> None:
    argv = _argv()
    joined = "\n".join(argv)
    assert argv[:3] == ["docker", "run", "--rm"]
    assert "--privileged" not in argv
    assert "--publish" not in argv and "-p" not in argv
    assert argv[argv.index("--network") + 1] == "none"
    assert argv[argv.index("--cap-drop") + 1] == "ALL"
    assert "no-new-privileges" in argv
    assert "src=/external/device-workspace/runtime/lfm350/source,dst=/workspace/source,readonly" in joined
    assert "src=/external/device-workspace/runtime/lfm350/assets,dst=/workspace/assets,readonly" in joined
    assert "src=/external/device-workspace/results/lfm350/state,dst=/workspace/state" in joined
    assert "src=/external/device-workspace/results/lfm350/output,dst=/workspace/output" in joined
    assert all(device in joined for device in host.DEVICES)
    assert argv.index(host.IMAGE_DIGEST) > argv.index("--workdir")


@pytest.mark.parametrize(
    "name",
    ["unrelated", "moe-qnn-lfm350-pilot", "moe-qnn-lfm350-PILOT-20261024", "../moe-qnn-lfm350-x-20261024"],
)
def test_rejects_unowned_container_names(name: str) -> None:
    with pytest.raises(host.ContractError):
        _argv(name=name)


@pytest.mark.parametrize(
    "bad_path",
    ["relative/path", "/external/device-workspace/../private", "/etc", "/external/device-workspace/x:ro", "/external/device-workspace"],
)
def test_rejects_unsafe_or_outside_mounts(bad_path: str) -> None:
    with pytest.raises(host.ContractError):
        _argv(source=bad_path)


def test_rejects_unreviewed_docker_binary() -> None:
    with pytest.raises(host.ContractError):
        _argv(docker_binary="sudo")


def test_ldd_parser_separates_missing_and_resolved() -> None:
    parsed = host.parse_ldd(
        """libGenie.so => /sdk/libGenie.so (0x1234)
libQnnHtp.so => not found
/lib/ld-linux-aarch64.so.1 (0x4567)
"""
    )
    assert parsed["missing"] == ["libQnnHtp.so"]
    assert parsed["resolved"] == {
        "libGenie.so": "/sdk/libGenie.so",
        "ld-linux-aarch64.so.1": "/lib/ld-linux-aarch64.so.1",
    }


def test_validator_requires_return_code_and_all_hardware_markers() -> None:
    output = "\n".join(
        [
            "Core Version of the backend DSP: Hexagon Architecture V68",
            "Success in executing the sum function",
            "Unit Test on the backend DSP: Passed.",
            "QNN is supported for backend DSP on the device.",
        ]
    )
    assert host.parse_validator(output, 0)["passed"] is True
    assert host.parse_validator(output, 1)["passed"] is False
    assert host.parse_validator(output.replace("V68", "V73"), 0)["passed"] is False


def test_handoff_pins_expected_wheel_libraries_and_safe_template() -> None:
    handoff = host.runtime_handoff()
    assert handoff["schema_version"] == 1
    assert handoff["kind"] == "tachyon_qnn_ubuntu24_python312_runtime"
    assert handoff["image"]["run_reference"] == host.IMAGE_DIGEST
    assert handoff["runtime"]["wheel"]["sha256"] == host.WHEEL["sha256"]
    assert handoff["runtime"]["qairt_version"] is None
    assert handoff["runtime"]["qnn_runtime_version"] is None
    assert set(handoff["runtime"]["libraries"]) == {"provider", "htp", "v68_stub", "v68_skel"}
    assert all(v["path"].startswith(host.QNN_PACKAGE + "/") for v in host.LIBRARIES.values())
    template = handoff["docker_argv_template"]
    assert "--privileged" not in template
    assert template[template.index("--network") + 1] == "none"
    assert {"SOURCE", "ASSETS", "STATE", "OUTPUT"} <= {
        part.split(",src=", 1)[1].split(",", 1)[0]
        for part in template
        if ",src=" in part
    }


def test_dependency_preflight_is_bound_to_image_and_has_backend_proof() -> None:
    preflight = host.runtime_handoff()["dependency_preflight"]
    assert preflight["status"] == "passed"
    assert preflight["image_digest"] == host.IMAGE_DIGEST
    assert preflight["unresolved"] == []
    assert preflight["ldd"]["unresolved"] == []
    assert preflight["provider_registration"]["status"] == "passed"
    assert preflight["provider_registration"]["rtld_now_all_checked_libraries"] is True
    assert preflight["tiny_qnn_graph"]["status"] == "passed"
    assert preflight["tiny_qnn_graph"]["backend_build"] == "v2.47.0.260601114230"
    assert preflight["tiny_qnn_graph"]["context_reload"] is True
    assert preflight["libatomic"]["package"] == "libatomic1:arm64"
    assert {item["name"] for item in preflight["checked_libraries"]} == {
        "libonnxruntime_providers_qnn.so",
        "libQnnHtp.so",
        "libQnnHtpPrepare.so",
        "libQnnHtpV68Stub.so",
        "libQnnSystem.so",
    }
