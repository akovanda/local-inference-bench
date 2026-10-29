#!/usr/bin/env python3
"""Cross-build and deploy an isolated llama.cpp CPU+OpenCL Tachyon runtime.

The build uses an Ubuntu 20.04 AArch64 cross toolchain plus Tachyon's existing
OpenCL 3.0 headers and ICD loader.  It never downloads a model, starts a model
server, or changes global ICD configuration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
from typing import Iterable


SOURCE_URL = "https://github.com/ggml-org/llama.cpp.git"
SOURCE_COMMIT = "620d2225cc436ab0b001e3200e590e3b286ef9ab"
IMAGE = "ubuntu:20.04@sha256:8feb4d8ca5354def3d8fce243717141ce31e2c428701f6682bd2fafe15388214"
CMAKE_VERSION = "3.31.8"
CMAKE_SHA256 = "630615d8e98ac33eba7fbe472626dff5c899c85af3c024585ae109166a6909d0"
ARCH_FLAGS = "-march=armv8.2-a+fp16+dotprod"
OPENCL_LOADER_REMOTE = "/usr/lib/libOpenCL.so"
OPENCL_LOADER_SHA256 = "7ed56297daac44e39ce39e3f57f1447af5068cdf9bfa8fa118b85a14b2af281b"
OPENCL_HEADERS_REMOTE = "/usr/include/adreno/CL"
ICD_LIBRARY = "/usr/lib/libOpenCL_adreno.so"
BINARIES = ("llama-server", "llama-bench", "llama-cli")
SSH_OPTIONS = (
    "-o", "BatchMode=yes",
    "-o", "StrictHostKeyChecking=yes",
    "-o", "ClearAllForwardings=yes",
    "-o", "ForwardAgent=no",
)


def run(command: list[str], *, cwd: Path | None = None) -> None:
    print("+", shlex.join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def reset_owned_directory(path: Path) -> None:
    if path.exists():
        try:
            shutil.rmtree(path)
        except PermissionError:
            run([
                "docker", "run", "--rm", "--volume", f"{path}:/owned",
                IMAGE, "chown", "-R", f"{os.getuid()}:{os.getgid()}", "/owned",
            ])
            shutil.rmtree(path)
    path.mkdir(parents=True)


def fetch_source(source: Path) -> None:
    if not source.exists():
        source.mkdir(parents=True)
        run(["git", "init", "--quiet"], cwd=source)
        run(["git", "remote", "add", "origin", SOURCE_URL], cwd=source)
    actual_url = subprocess.check_output(
        ["git", "remote", "get-url", "origin"], cwd=source, text=True
    ).strip()
    if actual_url != SOURCE_URL:
        raise RuntimeError(f"refusing unexpected source remote: {actual_url}")
    run(["git", "fetch", "--depth=1", "origin", SOURCE_COMMIT], cwd=source)
    run(["git", "checkout", "--detach", "--force", "FETCH_HEAD"], cwd=source)
    actual_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=source, text=True
    ).strip()
    if actual_commit != SOURCE_COMMIT:
        raise RuntimeError(f"source is {actual_commit}, expected {SOURCE_COMMIT}")


def fetch_opencl_development_files(sdk: Path) -> None:
    reset_owned_directory(sdk)
    include = sdk / "include"
    include.mkdir()
    run([
        "scp", *SSH_OPTIONS, "tachyon:" + OPENCL_LOADER_REMOTE,
        str(sdk / "libOpenCL.so"),
    ])
    run([
        "scp", "-r", *SSH_OPTIONS, "tachyon:" + OPENCL_HEADERS_REMOTE,
        str(include),
    ])
    actual = sha256(sdk / "libOpenCL.so")
    if actual != OPENCL_LOADER_SHA256:
        raise RuntimeError(
            f"Tachyon OpenCL loader is {actual}, expected {OPENCL_LOADER_SHA256}"
        )
    required = ("cl.h", "cl_platform.h", "cl_version.h", "opencl.h")
    missing = [name for name in required if not (include / "CL" / name).is_file()]
    if missing:
        raise RuntimeError(f"Tachyon OpenCL headers are incomplete: {', '.join(missing)}")


def build(source: Path, sdk: Path, build_dir: Path, jobs: int) -> None:
    reset_owned_directory(build_dir)
    uid_gid = f"{os.getuid()}:{os.getgid()}"
    script = f"""
set -eux
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends \
  ca-certificates curl g++-aarch64-linux-gnu gcc-aarch64-linux-gnu ninja-build python3
curl --fail --location --silent --show-error \
  --output /tmp/cmake.tar.gz \
  https://github.com/Kitware/CMake/releases/download/v{CMAKE_VERSION}/cmake-{CMAKE_VERSION}-linux-x86_64.tar.gz
echo "{CMAKE_SHA256}  /tmp/cmake.tar.gz" | sha256sum --check
mkdir /opt/cmake
tar -C /opt/cmake --strip-components=1 -xf /tmp/cmake.tar.gz
/opt/cmake/bin/cmake -S /src -B /build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_SYSTEM_NAME=Linux \
  -DCMAKE_SYSTEM_PROCESSOR=aarch64 \
  -DCMAKE_C_COMPILER=aarch64-linux-gnu-gcc \
  -DCMAKE_CXX_COMPILER=aarch64-linux-gnu-g++ \
  -DCMAKE_C_FLAGS={shlex.quote(ARCH_FLAGS)} \
  -DCMAKE_CXX_FLAGS={shlex.quote(ARCH_FLAGS)} \
  -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON \
  '-DCMAKE_INSTALL_RPATH=$ORIGIN/../lib' \
  -DOpenCL_LIBRARY=/opencl/libOpenCL.so \
  -DOpenCL_INCLUDE_DIR=/opencl/include \
  -DBUILD_SHARED_LIBS=OFF \
  -DGGML_NATIVE=OFF \
  -DGGML_CUDA=OFF \
  -DGGML_VULKAN=OFF \
  -DGGML_OPENCL=ON \
  -DGGML_OPENCL_TARGET_VERSION=300 \
  -DGGML_OPENCL_EMBED_KERNELS=ON \
  -DGGML_OPENCL_USE_ADRENO_KERNELS=ON \
  -DGGML_OPENCL_USE_ADRENO_BIN_KERNELS=OFF \
  -DGGML_OPENCL_PROFILING=OFF \
  -DGGML_OPENMP=ON \
  -DLLAMA_CURL=OFF \
  -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_EXAMPLES=ON \
  -DLLAMA_BUILD_SERVER=ON \
  -DLLAMA_BUILD_UI=OFF \
  -DLLAMA_USE_PREBUILT_UI=OFF \
  -DLLAMA_BUILD_NUMBER=0 \
  -DLLAMA_BUILD_COMMIT={SOURCE_COMMIT}
/opt/cmake/bin/cmake --build /build --parallel {jobs} --target {' '.join(BINARIES)}
chown -R {uid_gid} /build
"""
    run([
        "docker", "run", "--rm",
        "--volume", f"{source}:/src:ro",
        "--volume", f"{sdk}:/opencl:ro",
        "--volume", f"{build_dir}:/build",
        IMAGE, "bash", "-c", script,
    ])


def package(build_dir: Path, sdk: Path, bundle: Path) -> dict[str, str]:
    reset_owned_directory(bundle)
    (bundle / "bin").mkdir()
    (bundle / "lib").mkdir()
    (bundle / "vendors").mkdir()
    for name in BINARIES:
        source = build_dir / "bin" / name
        if not source.is_file():
            raise RuntimeError(f"build did not produce {source}")
        shutil.copy2(source, bundle / "bin" / name)
    shutil.copy2(sdk / "libOpenCL.so", bundle / "lib" / "libOpenCL.so")
    (bundle / "vendors" / "qualcomm.icd").write_text(ICD_LIBRARY + "\n")
    artifact_paths = [
        *(bundle / "bin" / name for name in BINARIES),
        bundle / "lib" / "libOpenCL.so",
        bundle / "vendors" / "qualcomm.icd",
    ]
    checksums = {
        str(path.relative_to(bundle)): sha256(path) for path in artifact_paths
    }
    metadata = {
        "source_url": SOURCE_URL,
        "source_commit": SOURCE_COMMIT,
        "container_image": IMAGE,
        "cmake_version": CMAKE_VERSION,
        "cmake_archive_sha256": CMAKE_SHA256,
        "arch_flags": ARCH_FLAGS,
        "opencl": {
            "enabled": True,
            "target_version": 300,
            "embedded_kernels": True,
            "adreno_kernels": True,
            "adreno_binary_kernels": False,
            "profiling": False,
            "loader_source": OPENCL_LOADER_REMOTE,
            "loader_sha256": OPENCL_LOADER_SHA256,
            "icd_library": ICD_LIBRARY,
        },
        "cmake": {
            "BUILD_SHARED_LIBS": False,
            "GGML_NATIVE": False,
            "GGML_CUDA": False,
            "GGML_VULKAN": False,
            "GGML_OPENCL": True,
            "GGML_OPENMP": True,
            "LLAMA_CURL": False,
            "LLAMA_BUILD_UI": False,
            "LLAMA_USE_PREBUILT_UI": False,
        },
        "quantization_support": {
            "Q4_0": "OpenCL MUL_MAT and MUL_MAT_ID kernels present",
            "Q2_0": "no OpenCL kernels/supports_op case; weight matmuls fall back to CPU",
        },
        "sha256": checksums,
    }
    (bundle / "BUILD.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (bundle / "SHA256SUMS").write_text(
        "".join(f"{value}  {name}\n" for name, value in sorted(checksums.items()))
    )
    return checksums


def remote_deploy_command(target: str, shared_vendors: str) -> str:
    parent, _ = target.rsplit("/", 1)
    return (
        f"set -eu; target={shlex.quote(target)}; "
        f"incoming={shlex.quote(target + '.new')}; "
        f"shared={shlex.quote(shared_vendors)}; "
        f"mkdir -p {shlex.quote(parent)}; "
        'rm -rf "$incoming"; mkdir -p "$incoming"; '
        'trap \'rm -rf "$incoming"\' EXIT HUP INT TERM; '
        'tar -C "$incoming" --strip-components=1 -xf -; '
        'if test -e "$target"; then '
        'backup="$target.previous.$(date -u +%Y%m%dT%H%M%SZ)"; '
        'mv "$target" "$backup"; fi; '
        'mv "$incoming" "$target"; trap - EXIT HUP INT TERM; '
        'mkdir -p "$shared"; shared_new="$shared/.qualcomm.icd.new.$$"; '
        'cp "$target/vendors/qualcomm.icd" "$shared_new"; '
        'mv "$shared_new" "$shared/qualcomm.icd"'
    )


def deploy(bundle: Path, target: str, shared_vendors: str) -> None:
    producer = subprocess.Popen(
        ["tar", "-C", str(bundle.parent), "-cf", "-", bundle.name],
        stdout=subprocess.PIPE,
    )
    assert producer.stdout is not None
    command = [
        "ssh", *SSH_OPTIONS, "-T", "tachyon",
        remote_deploy_command(target, shared_vendors),
    ]
    print("+ deploy", bundle, "to", f"tachyon:{target}", flush=True)
    consumer = subprocess.run(command, stdin=producer.stdout)
    producer.stdout.close()
    producer_status = producer.wait()
    if producer_status or consumer.returncode:
        raise subprocess.CalledProcessError(
            producer_status or consumer.returncode, "deploy pipeline"
        )


def parse_args(arguments: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--staging", type=Path, default=Path("vendor/tachyon-native-opencl-runtime")
    )
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument(
        "--deploy-target",
        default="/external/device-workspace/runtime/llama-cpp-opencl",
    )
    parser.add_argument(
        "--shared-vendors",
        default="/external/device-workspace/runtime/opencl/vendors",
    )
    parser.add_argument("--no-deploy", action="store_true")
    return parser.parse_args(arguments)


def main(arguments: Iterable[str] = sys.argv[1:]) -> int:
    args = parse_args(arguments)
    if not 1 <= args.jobs <= 4:
        raise SystemExit("--jobs must be between 1 and 4")
    staging = args.staging.resolve()
    source = staging / "source"
    sdk = staging / "tachyon-opencl-dev"
    build_dir = staging / "build-aarch64-focal-opencl"
    bundle = staging / "bundle"
    staging.mkdir(parents=True, exist_ok=True)
    fetch_source(source)
    fetch_opencl_development_files(sdk)
    build(source, sdk, build_dir, args.jobs)
    package(build_dir, sdk, bundle)
    if not args.no_deploy:
        deploy(bundle, args.deploy_target, args.shared_vendors)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
