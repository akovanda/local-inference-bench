#!/usr/bin/env python3
"""Build guarded Docker argv and evidence for Tachyon's private QNN userspace.

This module deliberately does not run Docker or SSH.  It emits an argv-only
contract for a pinned, non-privileged container and parses the two narrow
readiness proofs used by the setup procedure.
"""

from __future__ import annotations

import argparse
import json
import posixpath
from pathlib import PurePosixPath, Path
import re
from typing import Sequence


KIND = "tachyon_qnn_ubuntu24_python312_runtime"
IMAGE_REFERENCE = "moe-qnn-lfm350-baseline:ubuntu24-qnn2.3-libatomic-20261024"
IMAGE_DIGEST = "sha256:52bb2b57e4177a8038943581fc16fc670c4ea32fcd63702a73a68140b30ee941"
PYTHON = "/opt/qnn-venv/bin/python3.12"
REMOTE_ROOT = "/external/device-workspace"
REMOTE_WHEELHOUSE = f"{REMOTE_ROOT}/runtime/qnn-genie-host/wheelhouse"
QNN_PACKAGE = "/opt/qnn-venv/lib/python3.12/site-packages/onnxruntime_qnn"
OWNED_NAME = re.compile(r"^moe-qnn-(?:lfm350|genie)-[a-z0-9][a-z0-9-]{0,47}-20261024$")

WHEEL = {
    "filename": "onnxruntime_qnn-2.3.0-cp312-cp312-manylinux_2_34_aarch64.whl",
    "size": 78_730_222,
    "sha256": "1f0c072e1b0243f4ee5fabb9732323b20ecd6dd3af8b2fee8dbdacbd8eaf616d",
}

LIBRARIES = {
    "provider": {
        "path": f"{QNN_PACKAGE}/libonnxruntime_providers_qnn.so",
        "sha256": "ebcec5c0b52cc4bb96542accd17f4a410174bfc7a44599586f17852aa9ae78ef",
    },
    "htp": {
        "path": f"{QNN_PACKAGE}/libQnnHtp.so",
        "sha256": "9e1a73ed4f3e7cf3ef3199a23dfb3885b12a64edea30e607b0687b25b28e94f7",
    },
    "v68_stub": {
        "path": f"{QNN_PACKAGE}/libQnnHtpV68Stub.so",
        "sha256": "68e4c5f932ea006efa311c6763a4b25081f11663ae70390e6a58e41142f82c9f",
    },
    "v68_skel": {
        "path": f"{QNN_PACKAGE}/libQnnHtpV68Skel.so",
        "sha256": "1c4ef2cec89209ffb32d670cde10c3ff66733e44d34dd23c0b76f319afc5a6dd",
    },
}

HOST_MOUNTS = (
    ("/usr/lib/libcdsprpc.so", "/usr/local/lib/libcdsprpc.so"),
    ("/usr/lib/libion.so.0.0.0", "/usr/local/lib/libion.so.0"),
    ("/opt/qti/usr/lib/liblog.so.0.0.0.liblog", "/usr/local/lib/liblog.so.0"),
    ("/usr/lib/rfsa/adsp", "/usr/lib/rfsa/adsp"),
)
DEVICES = ("/dev/adsprpc-smd", "/dev/ion")

PREFLIGHT_ROOT = (
    "results/tachyon-qnn-investigation-20261024-v1/"
    "genie-host/libatomic-preflight"
)
REMOTE_PREFLIGHT_ROOT = f"{REMOTE_ROOT}/{PREFLIGHT_ROOT}"

DEPENDENCY_PREFLIGHT = {
    "schema_version": 1,
    "status": "passed",
    "image_digest": IMAGE_DIGEST,
    "checked_at": "2026-10-24T02:32:09Z",
    "unresolved": [],
    "checked_libraries": [
        {
            "name": "libonnxruntime_providers_qnn.so",
            "sha256": "ebcec5c0b52cc4bb96542accd17f4a410174bfc7a44599586f17852aa9ae78ef",
        },
        {
            "name": "libQnnHtp.so",
            "sha256": "9e1a73ed4f3e7cf3ef3199a23dfb3885b12a64edea30e607b0687b25b28e94f7",
        },
        {
            "name": "libQnnHtpPrepare.so",
            "sha256": "ec4a04f9157b5dc3a84840c7e2fbfea8affe531138129aa5271fcec2a5270a78",
        },
        {
            "name": "libQnnHtpV68Stub.so",
            "sha256": "68e4c5f932ea006efa311c6763a4b25081f11663ae70390e6a58e41142f82c9f",
        },
        {
            "name": "libQnnSystem.so",
            "sha256": "1dcc4e535f94b7d921d5c294eafc06bc0c87968077e5ee36bb14538820ef6990",
        },
    ],
    "libatomic": {
        "package": "libatomic1:arm64",
        "version": "14.2.0-4ubuntu2~24.04.1",
        "path": "/lib/aarch64-linux-gnu/libatomic.so.1",
        "sha256": "2896f2608b49113496e6872c77cb591f51f4e646fec01e59f517a971199979fe",
    },
    "ldd": {
        "status": "passed",
        "unresolved": [],
        "local_path": f"{PREFLIGHT_ROOT}/ldd.stdout",
        "remote_path": f"{REMOTE_PREFLIGHT_ROOT}/ldd.stdout",
        "sha256": "fc838e085109f009c670a34ff09b2052242ebe8c882896d251a784327a1819ff",
    },
    "provider_registration": {
        "status": "passed",
        "provider": "QNNExecutionProvider",
        "rtld_now_all_checked_libraries": True,
        "local_path": f"{PREFLIGHT_ROOT}/provider.stdout",
        "remote_path": f"{REMOTE_PREFLIGHT_ROOT}/provider.stdout",
        "sha256": "c52b97439efadbaffc279052115ee1b98673ae5b81208752ac2e79816adc7868",
    },
    "tiny_qnn_graph": {
        "status": "passed",
        "backend_build": "v2.47.0.260601114230",
        "graph": "FullyConnected+Relu",
        "original_exact_match": True,
        "restored_exact_match": True,
        "context_reload": True,
        "probe_binary_sha256": "a41d3d8ad56c222a309616444e92a365490705d0291cf0baedd6d86e75393f37",
        "context_sha256": "b293b899c4b6ab4241a89de8dd57f685d6e2ecc7dd99074907362895c3f3b5a8",
        "local_path": f"{PREFLIGHT_ROOT}/tiny-graph/stdout.jsonl",
        "remote_path": f"{REMOTE_PREFLIGHT_ROOT}/tiny-graph/stdout.jsonl",
        "sha256": "711397a9b5f6af58c6c62c3e32575f49e8f7069ec3c7820bb6678d5eae06b139",
    },
}


class ContractError(ValueError):
    """A requested container invocation is outside the private contract."""


def validate_container_name(name: str) -> str:
    if not OWNED_NAME.fullmatch(name):
        raise ContractError("container name is outside the owned moe-qnn namespace")
    return name


def validate_remote_mount(path: str, role: str) -> str:
    """Return a normalized private path, rejecting Docker-option injection."""
    if role not in {"source", "assets", "state", "output"}:
        raise ContractError(f"unknown mount role: {role}")
    if not isinstance(path, str) or not path or any(c in path for c in ":\x00\r\n"):
        raise ContractError(f"unsafe {role} mount")
    pure = PurePosixPath(path)
    if not pure.is_absolute() or ".." in pure.parts:
        raise ContractError(f"{role} mount must be an absolute non-traversing path")
    normalized = posixpath.normpath(path)
    if normalized == REMOTE_ROOT or not normalized.startswith(REMOTE_ROOT + "/"):
        raise ContractError(f"{role} mount must remain below {REMOTE_ROOT}")
    return normalized


def _validate_command(command: Sequence[str]) -> list[str]:
    if not command:
        raise ContractError("container command must not be empty")
    result = []
    for value in command:
        if not isinstance(value, str) or not value or "\x00" in value:
            raise ContractError("container command contains an unsafe argument")
        result.append(value)
    return result


def build_lfm350_docker_argv(
    *,
    name: str,
    source: str,
    assets: str,
    state: str,
    output: str,
    command: Sequence[str],
    docker_binary: str = "docker",
) -> list[str]:
    """Create the exact locked-down Docker argv for one LFM350 invocation."""
    if docker_binary != "docker":
        raise ContractError("only the audited Docker CLI is permitted")
    validate_container_name(name)
    mounts = {
        "source": validate_remote_mount(source, "source"),
        "assets": validate_remote_mount(assets, "assets"),
        "state": validate_remote_mount(state, "state"),
        "output": validate_remote_mount(output, "output"),
    }
    argv = [
        docker_binary,
        "run",
        "--rm",
        "--name",
        name,
        "--network",
        "none",
        "--read-only",
        "--tmpfs",
        "/tmp:rw,nosuid,nodev,noexec,size=512m",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--user",
        "1007:1005",
        "--group-add",
        "1000",
    ]
    for device in DEVICES:
        argv.extend(("--device", f"{device}:{device}:rwm"))
    for host, target in HOST_MOUNTS:
        argv.extend(("--mount", f"type=bind,src={host},dst={target},readonly"))
    for role, target, readonly in (
        ("source", "/workspace/source", True),
        ("assets", "/workspace/assets", True),
        ("state", "/workspace/state", False),
        ("output", "/workspace/output", False),
    ):
        suffix = ",readonly" if readonly else ""
        argv.extend(
            ("--mount", f"type=bind,src={mounts[role]},dst={target}{suffix}")
        )
    environment = {
        "HOME": "/workspace/state",
        "LD_LIBRARY_PATH": f"/usr/local/lib:{QNN_PACKAGE}",
        "ADSP_LIBRARY_PATH": f"{QNN_PACKAGE};/usr/lib/rfsa/adsp;/workspace/assets",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    for key, value in environment.items():
        argv.extend(("--env", f"{key}={value}"))
    argv.extend(("--workdir", "/workspace/source", IMAGE_DIGEST))
    argv.extend(_validate_command(command))
    return argv


def parse_ldd(text: str) -> dict[str, object]:
    """Parse the useful dependency resolution subset from ``ldd`` output."""
    resolved: dict[str, str] = {}
    missing: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        not_found = re.match(r"^(\S+)\s+=>\s+not found$", line)
        if not_found:
            missing.append(not_found.group(1))
            continue
        match = re.match(r"^(\S+)\s+=>\s+(\S+)(?:\s+\([^)]*\))?$", line)
        if match:
            library, target = match.groups()
            resolved[library] = target
            continue
        direct = re.match(r"^(\/\S+)\s+\([^)]*\)$", line)
        if direct:
            resolved[PurePosixPath(direct.group(1)).name] = direct.group(1)
    return {"resolved": resolved, "missing": sorted(set(missing))}


def parse_validator(stdout: str, returncode: int) -> dict[str, object]:
    markers = {
        "v68_detected": "Core Version of the backend DSP: Hexagon Architecture V68" in stdout,
        "sum_executed": "Success in executing the sum function" in stdout,
        "unit_test_passed": "Unit Test on the backend DSP: Passed." in stdout,
        "qnn_supported": "QNN is supported for backend DSP on the device." in stdout,
    }
    return {"passed": returncode == 0 and all(markers.values()), **markers}


def runtime_handoff() -> dict[str, object]:
    """Return the frozen schema consumed by the pilot controller."""
    mounts = [
        {"source": host, "target": target, "read_only": True}
        for host, target in HOST_MOUNTS
    ]
    return {
        "schema_version": 1,
        "kind": KIND,
        "image": {
            "reference": IMAGE_REFERENCE,
            "digest": IMAGE_DIGEST,
            "run_reference": IMAGE_DIGEST,
        },
        "runtime": {
            "python": PYTHON,
            "onnxruntime": "1.27.0",
            "onnxruntime_qnn": "2.3.0",
            "wheel": dict(WHEEL),
            "libraries": LIBRARIES,
            "qairt_version": None,
            "qnn_runtime_version": None,
            "embedded_htp_build": "v2.47.0.260601114230",
        },
        "runtime_mounts": mounts,
        "devices": list(DEVICES),
        "environment": {
            "LD_LIBRARY_PATH": f"/usr/local/lib:{QNN_PACKAGE}",
            "ADSP_LIBRARY_PATH": f"{QNN_PACKAGE};/usr/lib/rfsa/adsp;/workspace/assets",
        },
        "docker_binary": "docker",
        "docker_argv_template": build_template_argv(),
        "dependency_preflight": DEPENDENCY_PREFLIGHT,
        "repair": {
            "reason": "add Ubuntu 24 ARM64 libatomic1 required by libQnnSystem.so",
            "supersedes_image_digest": "sha256:f0dfbaf6e9c4bdb1286d880a73d12ca02ab659de00a2a8683d5e7a57c653117b",
            "supersedes_handoff_sha256": "eb12fc81b5f4883bc11ad63b6e5d8fdf2a38a6163c38da3405349c98c7bfa4da",
        },
        "claim": "host_ready_only_until_model_context_canaries_pass",
    }


def build_template_argv() -> list[str]:
    """Human/controller template; placeholders are never passed to Docker here."""
    argv = [
        "docker", "run", "--rm", "--name", "CONTAINER_NAME",
        "--network", "none", "--read-only", "--tmpfs",
        "/tmp:rw,nosuid,nodev,noexec,size=512m", "--cap-drop", "ALL",
        "--security-opt", "no-new-privileges", "--user", "1007:1005",
        "--group-add", "1000",
    ]
    for device in DEVICES:
        argv.extend(("--device", f"{device}:{device}:rwm"))
    for host, target in HOST_MOUNTS:
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
            "--env", "PYTHONDONTWRITEBYTECODE=1",
            "--workdir", "/workspace/source", IMAGE_DIGEST, "COMMAND",
        )
    )
    return argv


def _write_json(path: Path | None, value: object) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if path is None:
        print(payload, end="")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    contract = subparsers.add_parser("contract", help="emit the frozen pilot handoff")
    contract.add_argument("--output", type=Path)
    validator = subparsers.add_parser("parse-validator", help="parse saved validator output")
    validator.add_argument("--stdout", type=Path, required=True)
    validator.add_argument("--returncode", type=int, required=True)
    validator.add_argument("--output", type=Path)
    ldd = subparsers.add_parser("parse-ldd", help="parse saved ldd output")
    ldd.add_argument("--input", type=Path, required=True)
    ldd.add_argument("--output", type=Path)
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> int:
    args = parse_args(arguments)
    if args.command == "contract":
        _write_json(args.output, runtime_handoff())
    elif args.command == "parse-validator":
        _write_json(
            args.output,
            parse_validator(args.stdout.read_text(encoding="utf-8"), args.returncode),
        )
    elif args.command == "parse-ldd":
        _write_json(args.output, parse_ldd(args.input.read_text(encoding="utf-8")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
