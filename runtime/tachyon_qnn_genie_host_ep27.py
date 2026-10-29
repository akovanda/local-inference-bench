#!/usr/bin/env python3
"""Emit the isolated ORT 1.31 / QNN EP 2.7 Tachyon handoff."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

try:
    from runtime import tachyon_qnn_genie_host as base
except ModuleNotFoundError:  # direct ``python runtime/...py`` entrypoint
    import tachyon_qnn_genie_host as base


IMAGE_REFERENCE = "moe-qnn-lfm350-latest:ubuntu24-ort1.31-qnn2.7-20261024"
IMAGE_DIGEST = "sha256:0516b88cebfde308a28d7e702478a6b9b6d326b09433e6e63755e4c2fc54fd00"
QNN_PACKAGE = base.QNN_PACKAGE
RESULT_ROOT = "results/tachyon-qnn-investigation-20261024-v1/genie-host/latest-ep27"
REMOTE_RESULT_ROOT = f"{base.REMOTE_ROOT}/{RESULT_ROOT}"

WHEEL = {
    "filename": "onnxruntime_qnn-2.7.0-cp312-cp312-manylinux_2_34_aarch64.whl",
    "size": 88_834_007,
    "sha256": "40f6f8ab76a877b36c875122caf46189f810e52a1e7c2f75e431dfb7714168b8",
    "url": "https://files.pythonhosted.org/packages/77/f9/ea10e51868dc44ed3490f2e3dd3dda5c6b7ea6de4ae6ac9a51adde0a9cd2/onnxruntime_qnn-2.7.0-cp312-cp312-manylinux_2_34_aarch64.whl",
}

ORT_WHEEL = {
    "filename": "onnxruntime-1.31.0-cp312-cp312-manylinux_2_28_aarch64.whl",
    "size": 21_420_804,
    "sha256": "35758d7606d578ec5b9d65f6e8a1f488013194c3f6097038a3223cb26d35ef9a",
}

LIBRARIES = {
    "provider": {
        "path": f"{QNN_PACKAGE}/libonnxruntime_providers_qnn.so",
        "sha256": "74fe7c68710c859c50967c8ce374587102ba6f097a2321c3ee0e55ac8feea5de",
    },
    "htp": {
        "path": f"{QNN_PACKAGE}/libQnnHtp.so",
        "sha256": "7ec19d8002ad79509d1127cddb005e7bd07df1dfcb5ab31772078eaea5745a03",
    },
    "v68_stub": {
        "path": f"{QNN_PACKAGE}/libQnnHtpV68Stub.so",
        "sha256": "84e20670a3e25f7b7fa08b5f66134cb3bdeaa1c39f75940d4b5cce371390d53c",
    },
    "v68_skel": {
        "path": f"{QNN_PACKAGE}/libQnnHtpV68Skel.so",
        "sha256": "3f71eb57ce13a23d9dd92c6732782203ebb50245d2411812b06000008a2e1c21",
    },
}


def _evidence(name: str, sha256: str) -> dict[str, str]:
    return {
        "local_path": f"{RESULT_ROOT}/{name}",
        "remote_path": f"{REMOTE_RESULT_ROOT}/{name}",
        "sha256": sha256,
    }


def build_template_argv() -> list[str]:
    argv = [
        "docker", "run", "--rm", "--name", "CONTAINER_NAME",
        "--network", "none", "--read-only", "--tmpfs",
        "/tmp:rw,nosuid,nodev,noexec,size=512m", "--cap-drop", "ALL",
        "--security-opt", "no-new-privileges", "--user", "1007:1005",
        "--group-add", "1000",
    ]
    for device in base.DEVICES:
        argv.extend(("--device", f"{device}:{device}:rwm"))
    for host, target in base.HOST_MOUNTS:
        argv.extend(("--mount", f"type=bind,src={host},dst={target},readonly"))
    argv.extend(
        (
            "--mount", "type=bind,src=SOURCE,dst=/workspace/source,readonly",
            "--mount", "type=bind,src=ASSETS,dst=/workspace/assets,readonly",
            "--mount", "type=bind,src=STATE,dst=/workspace/state",
            "--mount", "type=bind,src=OUTPUT,dst=/workspace/output",
            "--env", "HOME=/workspace/state",
            "--env", f"LD_LIBRARY_PATH=/usr/local/lib:{QNN_PACKAGE}",
            "--env", f"ADSP_LIBRARY_PATH={QNN_PACKAGE};/usr/lib/rfsa/adsp;/workspace/assets",
            "--env", "ORT_QNN_ENABLE_CPU_BACKEND=1",
            "--env", "PYTHONDONTWRITEBYTECODE=1",
            "--workdir", "/workspace/source", IMAGE_DIGEST, "COMMAND",
        )
    )
    return argv


def runtime_handoff() -> dict[str, object]:
    return {
        "schema_version": 1,
        "kind": base.KIND,
        "variant": "latest_ep27",
        "image": {
            "reference": IMAGE_REFERENCE,
            "digest": IMAGE_DIGEST,
            "run_reference": IMAGE_DIGEST,
        },
        "runtime": {
            "python": base.PYTHON,
            "onnx": "1.22.0",
            "onnxruntime": "1.31.0",
            "onnxruntime_qnn": "2.7.0",
            "tokenizers": "0.23.1",
            "wheel": WHEEL,
            "onnxruntime_wheel": ORT_WHEEL,
            "libraries": LIBRARIES,
            "qairt_version": None,
            "qnn_runtime_version": None,
            "embedded_htp_build": "v2.51.0.260929181845",
        },
        "runtime_mounts": [
            {"source": host, "target": target, "read_only": True}
            for host, target in base.HOST_MOUNTS
        ],
        "devices": list(base.DEVICES),
        "environment": {
            "LD_LIBRARY_PATH": f"/usr/local/lib:{QNN_PACKAGE}",
            "ADSP_LIBRARY_PATH": f"{QNN_PACKAGE};/usr/lib/rfsa/adsp;/workspace/assets",
            "ORT_QNN_ENABLE_CPU_BACKEND": "1",
        },
        "docker_binary": "docker",
        "docker_argv_template": build_template_argv(),
        "dependency_preflight": {
            "schema_version": 1,
            "status": "passed",
            "scope": "host_dependencies_and_plugin_registration_only",
            "image_digest": IMAGE_DIGEST,
            "checked_at": "2026-10-24T02:44:01Z",
            "unresolved": [],
            "checked_libraries": [
                {
                    "name": "libonnxruntime_providers_qnn.so",
                    "sha256": LIBRARIES["provider"]["sha256"],
                },
                {"name": "libQnnHtp.so", "sha256": LIBRARIES["htp"]["sha256"]},
                {
                    "name": "libQnnHtpPrepare.so",
                    "sha256": "53fcebab0649bc14ee7532f268668018aac9f871b8c4a4105567bda105f85bd7",
                },
                {
                    "name": "libQnnHtpV68Stub.so",
                    "sha256": LIBRARIES["v68_stub"]["sha256"],
                },
                {
                    "name": "libQnnSystem.so",
                    "sha256": "93e25d5d8cb0a8788d336f25b9b1f71db757d5ae70a3a1ba91a8f9467d2e3de4",
                },
            ],
            "libatomic": {
                "package": "libatomic1:arm64",
                "version": "14.2.0-4ubuntu2~24.04.1",
                "path": "/lib/aarch64-linux-gnu/libatomic.so.1",
                "sha256": "2896f2608b49113496e6872c77cb591f51f4e646fec01e59f517a971199979fe",
            },
            "ldd": {"status": "passed", "unresolved": [], **_evidence("ldd.stdout", "fa1c1c61e3b3e243b5d98ad26a3b8bafa062ca65179eaa89741d081ae86c1ea0")},
            "provider_registration": {
                "status": "passed",
                "provider": "QNNExecutionProvider",
                "advertised_device_type": "OrtHardwareDeviceType.CPU",
                "backend_execution": "not_tested_controller_owned",
                "required_environment": {"ORT_QNN_ENABLE_CPU_BACKEND": "1"},
                **_evidence("provider.stdout", "54aae81c75256373564c7e307e74af6d8be435f56d0eabfdfc8a14f898e4078c"),
            },
        },
        "linux_npu_discovery": {
            "native_registration_without_opt_in": "no_QNN_OrtEpDevice",
            "workaround": "ORT_QNN_ENABLE_CPU_BACKEND=1",
            "controller_requirement": "select QNNExecutionProvider with explicit libQnnHtp.so backend and require QNN profiling > 0 with CPU profiling == 0",
            "limitation": "host preflight does not claim HTP execution; fresh model contexts and execution are controller-owned",
        },
        "build_string_provenance": {
            "inventory_path": "results/tachyon-qnn-investigation-20261024-v1/version-audit/qnn-python-wheel-inventories.json",
            "inventory_sha256": "63798780a18388fda29cdf72273365dfe679df7b6c6d7c03049bbca272e4fabe",
        },
        "claim": "host_ready_only_until_fresh_model_context_and_QNN_only_profile_pass",
    }


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> int:
    args = parse_args(arguments)
    payload = json.dumps(runtime_handoff(), indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
