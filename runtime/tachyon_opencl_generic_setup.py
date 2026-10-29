#!/usr/bin/env python3
"""Build and deploy a separate generic OpenCL llama.cpp runtime for Tachyon.

The build reuses the pinned focal cross-build infrastructure but disables the
Adreno-specific kernel set. It does not load a model or alter the private ICD.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import shlex
import subprocess
import sys
from types import ModuleType
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
BASE_SETUP = ROOT / "runtime/tachyon_opencl_setup.py"
BASE_SETUP_SHA256 = "fc828547bf484c7b1ca9c11eb0821df0b95cae799212dff7dc224e74eb102005"
DEFAULT_STAGING = ROOT / "vendor/tachyon-native-opencl-generic-runtime"
DEFAULT_TARGET = "/external/device-workspace/runtime/llama-cpp-opencl-generic"
PRIVATE_ICD = "/external/device-workspace/runtime/opencl/vendors"
DEFAULT_RECEIPT = ROOT / "results/tachyon-study-20261023-v1/opencl-generic-runtime-build.json"
SOURCE_COMMIT = "620d2225cc436ab0b001e3200e590e3b286ef9ab"
BINARIES = ("llama-server", "llama-bench", "llama-cli")
SSH_OPTIONS = (
    "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes",
    "-o", "ClearAllForwardings=yes", "-o", "ForwardAgent=no",
    "-o", "RequestTTY=no",
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_base() -> ModuleType:
    actual = sha256(BASE_SETUP)
    if actual != BASE_SETUP_SHA256:
        raise RuntimeError(f"base setup SHA-256 is {actual}, expected {BASE_SETUP_SHA256}")
    spec = importlib.util.spec_from_file_location("_tachyon_opencl_setup_pinned", BASE_SETUP)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import pinned OpenCL setup")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cmake_flags(base: ModuleType) -> list[str]:
    return [
        "-DCMAKE_BUILD_TYPE=Release",
        "-DCMAKE_SYSTEM_NAME=Linux",
        "-DCMAKE_SYSTEM_PROCESSOR=aarch64",
        "-DCMAKE_C_COMPILER=aarch64-linux-gnu-gcc",
        "-DCMAKE_CXX_COMPILER=aarch64-linux-gnu-g++",
        f"-DCMAKE_C_FLAGS={base.ARCH_FLAGS}",
        f"-DCMAKE_CXX_FLAGS={base.ARCH_FLAGS}",
        "-DCMAKE_BUILD_WITH_INSTALL_RPATH=ON",
        "-DCMAKE_INSTALL_RPATH=$ORIGIN/../lib",
        "-DOpenCL_LIBRARY=/opencl/libOpenCL.so",
        "-DOpenCL_INCLUDE_DIR=/opencl/include",
        "-DBUILD_SHARED_LIBS=OFF",
        "-DGGML_NATIVE=OFF",
        "-DGGML_CUDA=OFF",
        "-DGGML_VULKAN=OFF",
        "-DGGML_OPENCL=ON",
        "-DGGML_OPENCL_TARGET_VERSION=300",
        "-DGGML_OPENCL_EMBED_KERNELS=ON",
        "-DGGML_OPENCL_USE_ADRENO_KERNELS=OFF",
        "-DGGML_OPENCL_USE_ADRENO_BIN_KERNELS=OFF",
        "-DGGML_OPENCL_PROFILING=OFF",
        "-DGGML_OPENMP=ON",
        "-DLLAMA_CURL=OFF",
        "-DLLAMA_BUILD_TESTS=OFF",
        "-DLLAMA_BUILD_EXAMPLES=ON",
        "-DLLAMA_BUILD_SERVER=ON",
        "-DLLAMA_BUILD_UI=OFF",
        "-DLLAMA_USE_PREBUILT_UI=OFF",
        "-DLLAMA_BUILD_NUMBER=0",
        f"-DLLAMA_BUILD_COMMIT={SOURCE_COMMIT}",
    ]


def build(base: ModuleType, source: Path, sdk: Path, build_dir: Path, jobs: int) -> None:
    base.reset_owned_directory(build_dir)
    flags = " \\\n  ".join(shlex.quote(value) for value in cmake_flags(base))
    targets = " ".join(BINARIES)
    script = f"""
set -eux
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends ca-certificates curl g++-aarch64-linux-gnu gcc-aarch64-linux-gnu ninja-build python3
curl --fail --location --silent --show-error --output /tmp/cmake.tar.gz https://github.com/Kitware/CMake/releases/download/v{base.CMAKE_VERSION}/cmake-{base.CMAKE_VERSION}-linux-x86_64.tar.gz
echo "{base.CMAKE_SHA256}  /tmp/cmake.tar.gz" | sha256sum --check
mkdir /opt/cmake
tar -C /opt/cmake --strip-components=1 -xf /tmp/cmake.tar.gz
/opt/cmake/bin/cmake -S /src -B /build -G Ninja \\
  {flags}
/opt/cmake/bin/cmake --build /build --parallel {jobs} --target {targets}
chown -R {os.getuid()}:{os.getgid()} /build
"""
    base.run([
        "docker", "run", "--rm",
        "--volume", f"{source}:/src:ro",
        "--volume", f"{sdk}:/opencl:ro",
        "--volume", f"{build_dir}:/build",
        base.IMAGE, "bash", "-c", script,
    ])


def package(base: ModuleType, build_dir: Path, sdk: Path, bundle: Path) -> dict[str, str]:
    checksums = base.package(build_dir, sdk, bundle)
    metadata_path = bundle / "BUILD.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["opencl"].update({
        "adreno_kernels": False,
        "adreno_binary_kernels": False,
        "kernel_path": "generic embedded OpenCL kernels",
    })
    metadata["cmake"]["GGML_OPENCL_USE_ADRENO_KERNELS"] = False
    metadata["cmake"]["GGML_OPENCL_USE_ADRENO_BIN_KERNELS"] = False
    metadata["purpose"] = (
        "compatibility trial for Adreno 643 after the optimized kernel path "
        "failed in the Qualcomm OpenCL compiler before inference"
    )
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return checksums


def remote_deploy_command(target: str) -> str:
    parent, _ = target.rsplit("/", 1)
    return (
        f"set -eu; target={shlex.quote(target)}; incoming={shlex.quote(target + '.new')}; "
        f"mkdir -p {shlex.quote(parent)}; "
        'rm -rf "$incoming"; mkdir -p "$incoming"; '
        'trap \'rm -rf "$incoming"\' EXIT HUP INT TERM; '
        'tar -C "$incoming" --strip-components=1 -xf -; '
        'if test -e "$target"; then backup="$target.previous.$(date -u +%Y%m%dT%H%M%SZ)"; mv "$target" "$backup"; fi; '
        'mv "$incoming" "$target"; trap - EXIT HUP INT TERM'
    )


def deploy(bundle: Path, target: str) -> None:
    producer = subprocess.Popen(
        ["tar", "-C", str(bundle.parent), "-cf", "-", bundle.name],
        stdout=subprocess.PIPE,
    )
    assert producer.stdout is not None
    consumer = subprocess.run(
        ["ssh", *SSH_OPTIONS, "-T", "tachyon", remote_deploy_command(target)],
        stdin=producer.stdout,
        check=False,
    )
    producer.stdout.close()
    producer_status = producer.wait()
    if producer_status or consumer.returncode:
        raise subprocess.CalledProcessError(
            producer_status or consumer.returncode, "generic OpenCL deploy pipeline"
        )


def maximum_glibc_version(paths: Iterable[Path]) -> str:
    versions: list[tuple[int, ...]] = []
    for path in paths:
        output = subprocess.check_output(["readelf", "--version-info", str(path)], text=True)
        versions.extend(tuple(map(int, value.split("."))) for value in re.findall(r"GLIBC_(\d+(?:\.\d+)+)", output))
    if not versions:
        raise RuntimeError("no GLIBC symbol versions found")
    return ".".join(map(str, max(versions)))


def verify_remote(target: str, checksums: dict[str, str]) -> dict[str, Any]:
    script = r'''import hashlib,json,os,pathlib,platform,subprocess,sys
p=json.loads(sys.argv[1]); root=pathlib.Path(p["target"]); env=dict(os.environ); env["OCL_ICD_VENDORS"]=p["icd"]
def sha(path):
 d=hashlib.sha256()
 with path.open("rb") as f:
  for chunk in iter(lambda:f.read(1024*1024),b""): d.update(chunk)
 return d.hexdigest()
def run(argv,envv=None):
 c=subprocess.run(argv,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,env=envv,timeout=60)
 return {"argv":argv,"returncode":c.returncode,"output":c.stdout}
actual={name:sha(root/name) for name in p["sha256"]}
ldd={name:run(["ldd",str(root/name)]) for name in p["sha256"] if name.startswith("bin/")}
checks={
 "server_version":run([str(root/"bin/llama-server"),"--version"],env),
 "server_devices":run([str(root/"bin/llama-server"),"--list-devices"],env),
 "bench_devices":run([str(root/"bin/llama-bench"),"--list-devices"],env),
}
print(json.dumps({"machine":platform.machine(),"libc":platform.libc_ver(),"actual_sha256":actual,"expected_sha256":p["sha256"],"ldd":ldd,"checks":checks}))'''
    payload = json.dumps({"target": target, "icd": PRIVATE_ICD, "sha256": checksums}, separators=(",", ":"))
    remote = shlex.join(["python3", "-c", script, payload])
    completed = subprocess.run(
        ["ssh", *SSH_OPTIONS, "-T", "tachyon", remote],
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, timeout=180, check=False,
    )
    if completed.returncode:
        raise RuntimeError(f"remote verification failed: {completed.stderr[-1000:]}")
    value = json.loads(completed.stdout)
    if value["actual_sha256"] != checksums:
        raise RuntimeError("remote generic runtime checksums differ")
    if value["machine"] != "aarch64" or value["libc"] != ["glibc", "2.31"]:
        raise RuntimeError(f"unexpected remote ABI: {value['machine']} {value['libc']}")
    for name, check in value["checks"].items():
        if check["returncode"] != 0:
            raise RuntimeError(f"remote {name} failed: {check['output'][-1000:]}")
    device_output = value["checks"]["server_devices"]["output"] + value["checks"]["bench_devices"]["output"]
    if "GPUOpenCL" not in device_output or "Adreno(TM) 643" not in device_output:
        raise RuntimeError("generic runtime did not enumerate the expected OpenCL device")
    if any("not found" in row["output"] for row in value["ldd"].values()):
        raise RuntimeError("generic runtime has an unresolved dynamic dependency")
    return value


def write_receipt(
    path: Path,
    base: ModuleType,
    staging: Path,
    bundle: Path,
    target: str,
    jobs: int,
    checksums: dict[str, str],
    verification: dict[str, Any] | None,
) -> None:
    artifacts = {
        name: {"bytes": (bundle / name).stat().st_size, "sha256": digest}
        for name, digest in checksums.items()
    }
    document = {
        "schema_version": 1,
        "created_at_utc": now(),
        "outcome": "ready_for_parent_owned_generic_opencl_model_benchmark" if verification else "built_not_deployed",
        "scope": {
            "model_loaded": False, "inference_run": False, "server_started": False,
            "existing_server_or_container_changed": False,
            "private_icd_changed": False,
        },
        "source": {"url": base.SOURCE_URL, "commit": SOURCE_COMMIT},
        "paths": {
            "setup_script": str(Path(__file__).resolve()), "base_setup_script": str(BASE_SETUP),
            "local_staging": str(staging), "local_bundle": str(bundle),
            "remote_runtime": target, "private_icd": PRIVATE_ICD,
        },
        "toolchain": {
            "container_image": base.IMAGE, "cmake_version": base.CMAKE_VERSION,
            "parallel_jobs": jobs, "arch_flags": base.ARCH_FLAGS,
            "maximum_required_glibc_symbol": "GLIBC_" + maximum_glibc_version(
                bundle / "bin" / name for name in BINARIES
            ),
        },
        "build": {
            "cmake_flags": cmake_flags(base),
            "optimized_adreno_kernels": False,
            "generic_embedded_opencl_kernels": True,
            "fallback_compiler_option_used": False,
            "rationale": "optimized Adreno kernels failed in the Adreno 643 compiler with IndexRangeAnalyzer Stride != 0 and clBuildProgram -6",
        },
        "artifacts": artifacts,
        "remote_verification": verification,
        "source_pins": {
            str(BASE_SETUP): BASE_SETUP_SHA256,
            str(Path(__file__).resolve()): sha256(Path(__file__).resolve()),
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args(arguments: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staging", type=Path, default=DEFAULT_STAGING)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--deploy-target", default=DEFAULT_TARGET)
    parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    parser.add_argument("--no-deploy", action="store_true")
    return parser.parse_args(arguments)


def main(arguments: Iterable[str] = sys.argv[1:]) -> int:
    args = parse_args(arguments)
    if not 1 <= args.jobs <= 4:
        raise SystemExit("--jobs must be between 1 and 4")
    base = load_base()
    staging = args.staging.resolve()
    source = staging / "source"
    sdk = staging / "tachyon-opencl-dev"
    build_dir = staging / "build-aarch64-focal-opencl-generic"
    bundle = staging / "bundle"
    staging.mkdir(parents=True, exist_ok=True)
    base.fetch_source(source)
    base.fetch_opencl_development_files(sdk)
    build(base, source, sdk, build_dir, args.jobs)
    checksums = package(base, build_dir, sdk, bundle)
    verification = None
    if not args.no_deploy:
        deploy(bundle, args.deploy_target)
        verification = verify_remote(args.deploy_target, checksums)
    write_receipt(
        args.receipt.resolve(), base, staging, bundle, args.deploy_target,
        args.jobs, checksums, verification,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
