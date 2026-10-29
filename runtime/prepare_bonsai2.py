#!/usr/bin/env python3
"""Stage and attest the pinned Prism llama.cpp runtime for Bonsai 2 27B.

This helper deliberately performs no GPU probe and never starts llama-server.  It
verifies the official release archive, extracts it into an isolated directory,
uses cuobjdump only as an offline fatbinary inspector, and optionally hashes one
of the two pinned Bonsai 2 model packings.  The resulting receipt can become a
``llama_trial.py`` artifact manifest once the selected model is present.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import struct
import subprocess
import sys
import tarfile
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
RECEIPT = ROOT / "results" / "bonsai2-runtime-artifact.json"

MODEL_REPO = "prism-ml/Ternary-Bonsai-2-27B-gguf"
MODEL_REVISION = "b072e1d3b35a0a630cece372c2127528e0994386"
MODEL_DIR = Path("/external/storage/moe-bench/models/bonsai2-27b")

DEMO_REPO = "https://github.com/PrismML-Eng/Bonsai-demo.git"
DEMO_REVISION = "db871ea96bffff5ce5609873c412b75d7cd5ba3c"
LLAMA_REPO = "https://github.com/PrismML-Eng/llama.cpp.git"
LLAMA_RELEASE = "prism-b10777-e3de715"
LLAMA_REVISION = "e3de715222b2667c54a267441159a5729f11155e"
RUNTIME_ROOT = Path("/external/storage/moe-bench/bonsai2-runtime-prism-b10777")
PREBUILT_DIR = RUNTIME_ROOT / "cuda-12.8"
SOURCE_DIR = Path("/external/storage/moe-bench/bonsai2-llama-source")
BUILD_DIR = Path("/external/storage/moe-bench/bonsai2-llama-build-sm75")
RUNTIME_DIR = BUILD_DIR / "bin"
ARCHIVE_NAME = "llama-prism-b10777-e3de715-bin-linux-cuda-12.8-x64.tar.gz"
ARCHIVE = RUNTIME_ROOT / ARCHIVE_NAME
ARCHIVE_SIZE = 208_198_074
ARCHIVE_SHA256 = "aa53e2256276e683386effdc8be34b3bd1cdf64ef9dc5d492da69227ac52531b"
ARCHIVE_URL = (
    "https://github.com/PrismML-Eng/llama.cpp/releases/download/"
    f"{LLAMA_RELEASE}/{ARCHIVE_NAME}"
)
CUDA_TOOLKIT = Path("/usr/local/cuda-13.0")
CUOBJDUMP = CUDA_TOOLKIT / "bin" / "cuobjdump"
CMAKE = ROOT / "vendor" / "strata" / ".venv" / "bin" / "cmake"
NINJA = ROOT / "vendor" / "strata" / ".venv" / "bin" / "ninja"
T4_CUDA_ARCHITECTURE = 75
MAX_BUILD_JOBS = 4
BUILD_OBSERVATION = {
    "observed_at": "2026-10-19T19:53:48Z",
    "logical_cpus": 56,
    "mem_available_kib": 149_335_100,
    "load_average_1m": 22.43,
    "load_average_5m": 22.30,
    "load_average_15m": 18.72,
    "active_parallel_jobs": 4,
}
HASH_RE = re.compile(r"^[0-9a-f]{64}$")
SM_RE = re.compile(r"(?:^|[^0-9])sm_([0-9]+)(?:[^0-9]|$)")
CHAT_TEMPLATE_SHA256 = "c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041"
CHAT_TEMPLATE_KEY = "tokenizer.chat_template"
GGUF_SCALAR_SIZES = {
    0: 1,   # uint8
    1: 1,   # int8
    2: 2,   # uint16
    3: 2,   # int16
    4: 4,   # uint32
    5: 4,   # int32
    6: 4,   # float32
    7: 1,   # bool
    10: 8,  # uint64
    11: 8,  # int64
    12: 8,  # float64
}


class PreparationError(RuntimeError):
    """A pinned Bonsai runtime or model identity did not verify."""


@dataclass(frozen=True)
class ModelSpec:
    quantization: str
    name: str
    size: int
    sha256: str

    @property
    def default_path(self) -> Path:
        return MODEL_DIR / self.name

    @property
    def url(self) -> str:
        return (
            f"https://huggingface.co/{MODEL_REPO}/resolve/"
            f"{MODEL_REVISION}/{self.name}"
        )


MODELS: Mapping[str, ModelSpec] = {
    "PTQ1_0": ModelSpec(
        quantization="PTQ1_0",
        name="Ternary-Bonsai-2-27B-PTQ1_0.gguf",
        size=5_946_648_928,
        sha256="53107f530aa52eb00912263ab1ee29bd199261c87cd7b4ad4ca1318c1fe33ee3",
    ),
    "PQ2_0": ModelSpec(
        quantization="PQ2_0",
        name="Ternary-Bonsai-2-27B-PQ2_0.gguf",
        size=7_206_168_928,
        sha256="3907dc1658db1f78a9826bf8d5bcb8dc65db0d466388937af57f2294fae62ec1",
    ),
}


SOURCE_BUILD_COMMANDS = (
    (
        "git clone --filter=blob:none --no-checkout "
        f"{LLAMA_REPO} {SOURCE_DIR}"
    ),
    (
        f"git -C {SOURCE_DIR} checkout --detach "
        f"{LLAMA_REVISION}"
    ),
    (
        "PATH=/usr/local/cuda-13.0/bin:$PATH "
        "CUDACXX=/usr/local/cuda-13.0/bin/nvcc "
        f"{CMAKE} -S {SOURCE_DIR} "
        f"-B {BUILD_DIR} "
        "-G Ninja -DGGML_CUDA=ON "
        "-DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.0/bin/nvcc "
        "-DCMAKE_CUDA_ARCHITECTURES=75 -DCMAKE_BUILD_TYPE=Release "
        "-DGGML_CUDA_FA_QUANTS=f16-f16 "
        "-DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF "
        "-DLLAMA_BUILD_SERVER=ON"
    ),
    (
        f"{CMAKE} --build {BUILD_DIR} "
        "--target llama-server --parallel 4"
    ),
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PreparationError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(16 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    descriptor = -1
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        view = memoryview(payload)
        written = 0
        while written < len(view):
            count = os.write(descriptor, view[written:])
            if count <= 0:
                raise OSError("short JSON write")
            written += count
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        os.replace(temporary, path)
        parent = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(parent)
        finally:
            os.close(parent)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def verify_regular_file(path: Path, *, size: int, sha256: str, label: str) -> dict[str, Any]:
    _require(HASH_RE.fullmatch(sha256) is not None, f"invalid pinned SHA-256 for {label}")
    try:
        before = path.stat(follow_symlinks=False)
    except FileNotFoundError as exc:
        raise PreparationError(f"missing {label}: {path}") from exc
    _require(stat.S_ISREG(before.st_mode), f"{label} is not a regular file: {path}")
    _require(before.st_size == size, f"{label} is {before.st_size} bytes; expected {size}")
    actual = sha256_file(path)
    after = path.stat(follow_symlinks=False)
    identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    _require(identity_before == identity_after, f"{label} changed while it was hashed: {path}")
    _require(actual == sha256, f"{label} SHA-256 is {actual}; expected {sha256}")
    return {"path": str(path), "size": size, "sha256": actual, "verified": True}


def _read_exact(handle: Any, size: int, label: str) -> bytes:
    payload = handle.read(size)
    _require(len(payload) == size, f"truncated GGUF while reading {label}")
    return payload


def _read_u32(handle: Any, label: str) -> int:
    return struct.unpack("<I", _read_exact(handle, 4, label))[0]


def _read_u64(handle: Any, label: str) -> int:
    return struct.unpack("<Q", _read_exact(handle, 8, label))[0]


def _skip_bytes(handle: Any, size: int, file_size: int, label: str) -> None:
    _require(size >= 0, f"invalid negative GGUF length for {label}")
    destination = handle.tell() + size
    _require(destination <= file_size, f"truncated GGUF while skipping {label}")
    handle.seek(destination)


def _read_gguf_bytes(handle: Any, file_size: int, *, max_size: int, label: str) -> bytes:
    size = _read_u64(handle, f"{label} length")
    _require(size <= max_size, f"GGUF {label} is unreasonably large: {size} bytes")
    _require(handle.tell() + size <= file_size, f"truncated GGUF while reading {label}")
    return _read_exact(handle, size, label)


def _skip_gguf_string(handle: Any, file_size: int, label: str) -> None:
    size = _read_u64(handle, f"{label} length")
    _skip_bytes(handle, size, file_size, label)


def _skip_gguf_value(
    handle: Any,
    value_type: int,
    file_size: int,
    *,
    label: str,
    depth: int = 0,
) -> None:
    _require(depth <= 4, f"GGUF metadata array nesting is too deep at {label}")
    if value_type in GGUF_SCALAR_SIZES:
        _skip_bytes(handle, GGUF_SCALAR_SIZES[value_type], file_size, label)
        return
    if value_type == 8:  # string
        _skip_gguf_string(handle, file_size, label)
        return
    _require(value_type == 9, f"unsupported GGUF metadata type {value_type} at {label}")
    element_type = _read_u32(handle, f"{label} array element type")
    count = _read_u64(handle, f"{label} array count")
    if element_type in GGUF_SCALAR_SIZES:
        _skip_bytes(
            handle,
            count * GGUF_SCALAR_SIZES[element_type],
            file_size,
            f"{label} array payload",
        )
        return
    _require(count <= 10_000_000, f"GGUF metadata array is unreasonably large at {label}: {count}")
    for index in range(count):
        _skip_gguf_value(
            handle,
            element_type,
            file_size,
            label=f"{label}[{index}]",
            depth=depth + 1,
        )


def read_gguf_chat_template(path: Path) -> bytes:
    """Read only tokenizer.chat_template while skipping tensor data."""

    file_size = path.stat().st_size
    with path.open("rb") as handle:
        _require(_read_exact(handle, 4, "magic") == b"GGUF", f"model is not a GGUF file: {path}")
        version = _read_u32(handle, "version")
        _require(version in (2, 3), f"unsupported GGUF version {version}: {path}")
        _read_u64(handle, "tensor count")
        metadata_count = _read_u64(handle, "metadata count")
        _require(metadata_count <= 1_000_000, f"unreasonable GGUF metadata count: {metadata_count}")
        for index in range(metadata_count):
            key_bytes = _read_gguf_bytes(
                handle,
                file_size,
                max_size=1024 * 1024,
                label=f"metadata key {index}",
            )
            try:
                key = key_bytes.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise PreparationError(f"invalid UTF-8 in GGUF metadata key {index}") from exc
            value_type = _read_u32(handle, f"metadata type for {key}")
            if key == CHAT_TEMPLATE_KEY:
                _require(value_type == 8, f"{CHAT_TEMPLATE_KEY} is not a GGUF string")
                return _read_gguf_bytes(
                    handle,
                    file_size,
                    max_size=16 * 1024 * 1024,
                    label=CHAT_TEMPLATE_KEY,
                )
            _skip_gguf_value(handle, value_type, file_size, label=key)
    raise PreparationError(f"GGUF metadata lacks {CHAT_TEMPLATE_KEY}: {path}")


def inspect_chat_template(
    path: Path,
    *,
    expected_sha256: str = CHAT_TEMPLATE_SHA256,
) -> dict[str, Any]:
    template_bytes = read_gguf_chat_template(path)
    actual_sha256 = hashlib.sha256(template_bytes).hexdigest()
    _require(
        actual_sha256 == expected_sha256,
        f"{CHAT_TEMPLATE_KEY} SHA-256 is {actual_sha256}; expected {expected_sha256}",
    )
    try:
        template = template_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise PreparationError(f"{CHAT_TEMPLATE_KEY} is not valid UTF-8") from exc
    false_condition = "{%- if enable_thinking is defined and enable_thinking is false %}"
    empty_think = "{{- '<think>\\n\\n</think>\\n\\n' }}"
    enable_thinking_false = false_condition in template and empty_think in template
    _require(
        enable_thinking_false,
        f"{CHAT_TEMPLATE_KEY} does not explicitly emit an empty think block for enable_thinking=false",
    )
    return {
        "metadata_key": CHAT_TEMPLATE_KEY,
        "sha256": actual_sha256,
        "expected_sha256": expected_sha256,
        "verified": True,
        "capabilities": {
            "enable_thinking_false": True,
            "empty_think_generation_prompt": True,
        },
        "evidence": {
            "condition": false_condition,
            "emission": empty_think,
        },
        "inspection": "static GGUF metadata inspection; model payload was not loaded",
    }


def download_archive() -> dict[str, Any]:
    if ARCHIVE.exists():
        return verify_regular_file(
            ARCHIVE, size=ARCHIVE_SIZE, sha256=ARCHIVE_SHA256, label="runtime archive"
        )
    ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
    temporary = ARCHIVE.with_name(f"{ARCHIVE.name}.download.{os.getpid()}")
    try:
        request = Request(ARCHIVE_URL, headers={"User-Agent": "moe-bonsai2-preparer/1"})
        with urlopen(request, timeout=60) as response, temporary.open("xb") as output:
            shutil.copyfileobj(response, output, length=4 * 1024 * 1024)
            output.flush()
            os.fsync(output.fileno())
        verify_regular_file(
            temporary, size=ARCHIVE_SIZE, sha256=ARCHIVE_SHA256, label="downloaded runtime archive"
        )
        os.replace(temporary, ARCHIVE)
    finally:
        temporary.unlink(missing_ok=True)
    return verify_regular_file(
        ARCHIVE, size=ARCHIVE_SIZE, sha256=ARCHIVE_SHA256, label="runtime archive"
    )


def _safe_archive_members(archive: tarfile.TarFile) -> list[tarfile.TarInfo]:
    members = archive.getmembers()
    _require(bool(members), "runtime archive is empty")
    for member in members:
        path = Path(member.name)
        _require(not path.is_absolute() and ".." not in path.parts, f"unsafe archive path: {member.name}")
        if member.issym() or member.islnk():
            target = Path(member.linkname)
            _require(
                not target.is_absolute() and ".." not in target.parts,
                f"unsafe archive link: {member.name} -> {member.linkname}",
            )
        _require(not (member.ischr() or member.isblk() or member.isfifo()), f"unsafe archive member: {member.name}")
    return members


def extract_prebuilt() -> None:
    archive_record = download_archive()
    if PREBUILT_DIR.exists():
        return
    PREBUILT_DIR.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".cuda-12.8-stage-", dir=PREBUILT_DIR.parent))
    try:
        with tarfile.open(ARCHIVE, mode="r:gz") as bundle:
            members = _safe_archive_members(bundle)
            bundle.extractall(staging, members=members, filter="data")
        _require(not PREBUILT_DIR.exists(), f"prebuilt destination appeared during extraction: {PREBUILT_DIR}")
        os.replace(staging, PREBUILT_DIR)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def _find_one(root: Path, name: str) -> Path:
    matches = [path for path in root.rglob(name) if path.is_file()]
    _require(len(matches) == 1, f"expected one {name} under {root}; found {len(matches)}")
    return matches[0]


def _binary_record(path: Path) -> dict[str, Any]:
    details = path.stat()
    return {
        "path": str(path),
        "size": details.st_size,
        "sha256": sha256_file(path),
        "executable": os.access(path, os.X_OK),
    }


def _readelf_dynamic(path: Path) -> dict[str, Any]:
    result = subprocess.run(
        ["readelf", "-d", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    needed = sorted(set(re.findall(r"\(NEEDED\).*?\[(.*?)\]", result.stdout)))
    runpaths = sorted(set(re.findall(r"\((?:RUNPATH|RPATH)\).*?\[(.*?)\]", result.stdout)))
    return {"needed": needed, "runpaths": runpaths, "probe": "offline readelf -d"}


def inspect_cuda_architectures(backend: Path, *, require_t4: bool = True) -> dict[str, Any]:
    _require(CUOBJDUMP.is_file(), f"offline CUDA inspector is missing: {CUOBJDUMP}")
    result = subprocess.run(
        [str(CUOBJDUMP), "--list-elf", str(backend)],
        check=True,
        capture_output=True,
        text=True,
    )
    output = result.stdout + result.stderr
    architectures = sorted({int(value) for value in SM_RE.findall(output)})
    _require(bool(architectures), "cuobjdump found no embedded CUDA architectures")
    t4_present = T4_CUDA_ARCHITECTURE in architectures
    if require_t4:
        _require(t4_present, f"runtime backend lacks sm_{T4_CUDA_ARCHITECTURE} code for Tesla T4")
    return {
        "architectures": architectures,
        "tesla_t4_sm75_present": t4_present,
        "command": [str(CUOBJDUMP), "--list-elf", str(backend)],
        "output_sha256": hashlib.sha256(output.encode("utf-8")).hexdigest(),
        "method": "offline fatbinary inspection; CUDA backend was not loaded",
    }


def _git_revision(path: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=path, check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def _configure_argv() -> list[str]:
    return [
        str(CMAKE), "-S", str(SOURCE_DIR), "-B", str(BUILD_DIR), "-G", "Ninja",
        "-DGGML_CUDA=ON",
        f"-DCMAKE_CUDA_COMPILER={CUDA_TOOLKIT / 'bin' / 'nvcc'}",
        f"-DCMAKE_CUDA_ARCHITECTURES={T4_CUDA_ARCHITECTURE}",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DGGML_CUDA_FA_QUANTS=f16-f16",
        "-DLLAMA_BUILD_TESTS=OFF",
        "-DLLAMA_BUILD_EXAMPLES=OFF",
        "-DLLAMA_BUILD_SERVER=ON",
    ]


def build_source_runtime() -> None:
    _require(CMAKE.is_file() and os.access(CMAKE, os.X_OK), f"existing cmake is missing: {CMAKE}")
    _require(NINJA.is_file() and os.access(NINJA, os.X_OK), f"existing ninja is missing: {NINJA}")
    if not SOURCE_DIR.exists():
        subprocess.run(
            ["git", "clone", "--filter=blob:none", "--no-checkout", LLAMA_REPO, str(SOURCE_DIR)],
            check=True,
        )
        subprocess.run(
            ["git", "checkout", "--detach", LLAMA_REVISION], cwd=SOURCE_DIR, check=True
        )
    _require(SOURCE_DIR.is_dir() and not SOURCE_DIR.is_symlink(), f"invalid source directory: {SOURCE_DIR}")
    _require(_git_revision(SOURCE_DIR) == LLAMA_REVISION, "llama.cpp source is not at the pinned revision")
    environment = dict(os.environ)
    environment["PATH"] = os.pathsep.join(
        (str(CMAKE.parent), str(CUDA_TOOLKIT / "bin"), environment.get("PATH", ""))
    )
    environment["CUDACXX"] = str(CUDA_TOOLKIT / "bin" / "nvcc")
    subprocess.run(_configure_argv(), check=True, env=environment)
    subprocess.run(
        [str(CMAKE), "--build", str(BUILD_DIR), "--target", "llama-server", "--parallel", str(MAX_BUILD_JOBS)],
        check=True,
        env=environment,
    )


def verify_source_compatibility() -> dict[str, Any]:
    requirements = {
        "ggml/include/ggml.h": ("GGML_TYPE_PQ2_0", "GGML_TYPE_PTQ1_0"),
        "ggml/src/ggml-cuda/mmvq-pq2_0.cuh": ("mul_mat_vec_pq2_0",),
        "ggml/src/ggml-cuda/mmvq-ptq1_0.cuh": ("mul_mat_vec_ptq1_0",),
        "ggml/src/ggml-cuda/fwht.cu": ("ggml_cuda_op_fwht", "ggml_cuda_op_fwht_signed"),
        "src/llama-context.cpp": ("Hadamard-folded weight", "activation transform"),
        "src/models/qwen35.cpp": ("hadamard",),
    }
    records: list[dict[str, Any]] = []
    for relative, fragments in requirements.items():
        path = SOURCE_DIR / relative
        try:
            payload = path.read_bytes()
        except OSError as exc:
            raise PreparationError(f"pinned source evidence is unreadable: {path}") from exc
        text = payload.decode("utf-8", errors="replace")
        for fragment in fragments:
            _require(fragment in text, f"pinned source lacks {fragment!r} in {relative}")
        records.append(
            {
                "path": str(path),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "required_fragments": list(fragments),
            }
        )
    return {
        "pq2_0_cuda_source_present": True,
        "ptq1_0_cuda_source_present": True,
        "hadamard_and_sign_transform_source_present": True,
        "files": records,
        "scope": "static source audit only; no model was loaded",
    }


def install_runtime() -> None:
    extract_prebuilt()
    prebuilt_backend = _find_one(PREBUILT_DIR, "libggml-cuda.so")
    prebuilt_arches = inspect_cuda_architectures(prebuilt_backend, require_t4=False)
    if prebuilt_arches["tesla_t4_sm75_present"]:
        raise PreparationError("helper invariant changed: the pinned prebuilt unexpectedly gained sm_75")
    build_source_runtime()


def inspect_runtime(archive_record: Mapping[str, Any] | None = None) -> dict[str, Any]:
    _require(RUNTIME_DIR.is_dir() and not RUNTIME_DIR.is_symlink(), f"runtime is not installed: {RUNTIME_DIR}")
    server = _find_one(RUNTIME_DIR, "llama-server")
    backend = _find_one(RUNTIME_DIR, "libggml-cuda.so")
    _require(os.access(server, os.X_OK), f"llama-server is not executable: {server}")
    candidates = sorted(
        {
            path.resolve()
            for path in RUNTIME_DIR.rglob("*.so*")
            if path.is_file()
        },
        key=str,
    )
    _require(bool(candidates), "runtime contains no llama/ggml shared libraries")
    libraries = [_binary_record(path) for path in candidates]
    library_dirs = sorted(
        {str(path.parent) for path in candidates} | {str(CUDA_TOOLKIT / "lib64")}
    )
    _require(_git_revision(SOURCE_DIR) == LLAMA_REVISION, "selected runtime source revision changed")
    cache = BUILD_DIR / "CMakeCache.txt"
    cache_text = cache.read_text(encoding="utf-8", errors="replace")
    _require(
        re.search(
            rf"^CMAKE_CUDA_ARCHITECTURES:[^=]+={T4_CUDA_ARCHITECTURE}$",
            cache_text,
            re.MULTILINE,
        )
        is not None,
        "source build cache is not pinned to sm_75",
    )
    prebuilt_backend = _find_one(PREBUILT_DIR, "libggml-cuda.so")
    prebuilt_fatbinary = inspect_cuda_architectures(prebuilt_backend, require_t4=False)
    _require(not prebuilt_fatbinary["tesla_t4_sm75_present"], "prebuilt rejection evidence changed")
    return {
        "release": LLAMA_RELEASE,
        "source_revision": LLAMA_REVISION,
        "selected_artifact": "source_build_cuda13_sm75",
        "install_dir": str(RUNTIME_DIR),
        "archive": dict(archive_record) if archive_record is not None else verify_regular_file(
            ARCHIVE, size=ARCHIVE_SIZE, sha256=ARCHIVE_SHA256, label="runtime archive"
        ),
        "server": {
            **_binary_record(server),
            "dynamic": _readelf_dynamic(server),
        },
        "cuda_backend": {
            **_binary_record(backend),
            "dynamic": _readelf_dynamic(backend),
            "fatbinary": inspect_cuda_architectures(backend),
        },
        "libraries": libraries,
        "lib_dirs": library_dirs,
        "backend_discovery": {
            "GGML_BACKEND_PATH": str(backend.resolve()),
            "LD_LIBRARY_PATH_prefix": os.pathsep.join(library_dirs),
            "llama_trial_lib_dirs": library_dirs,
        },
        "source_build": {
            "source_dir": str(SOURCE_DIR),
            "build_dir": str(BUILD_DIR),
            "revision": LLAMA_REVISION,
            "cmake": str(CMAKE),
            "ninja": str(NINJA),
            "configure_argv": _configure_argv(),
            "build_argv": [
                str(CMAKE), "--build", str(BUILD_DIR), "--target", "llama-server",
                "--parallel", str(MAX_BUILD_JOBS),
            ],
            "cuda_architectures": [T4_CUDA_ARCHITECTURE],
            "flash_attention_kv_types": ["f16-f16"],
            "max_parallel_jobs": MAX_BUILD_JOBS,
            "host_observation_during_compile": dict(BUILD_OBSERVATION),
            "compatibility_source_audit": verify_source_compatibility(),
        },
        "prebuilt_candidate": {
            "install_dir": str(PREBUILT_DIR),
            "selected": False,
            "rejection": "official CUDA 12.8 archive has no sm_75 cubin for Tesla T4",
            "cuda_backend": _binary_record(prebuilt_backend),
            "fatbinary": prebuilt_fatbinary,
        },
        "gpu_backend_loaded": False,
        "gpu_probe_performed": False,
        "measurements_performed": False,
    }


def verify_model(spec: ModelSpec, path: Path) -> dict[str, Any]:
    before = path.stat(follow_symlinks=False)
    record = verify_regular_file(path, size=spec.size, sha256=spec.sha256, label=spec.name)
    template = inspect_chat_template(path)
    after = path.stat(follow_symlinks=False)
    identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    _require(identity_before == identity_after, f"{spec.name} changed during model verification: {path}")
    template["model_file_verified_first"] = True
    return {"name": spec.name, **record, "chat_template": template}


def trial_config_template(
    spec: ModelSpec,
    model_path: Path,
    runtime: Mapping[str, Any],
    *,
    context: int = 32768,
) -> dict[str, Any]:
    _require(512 <= context <= 262_144, "context must be between 512 and 262144")
    return {
        "schema_version": 1,
        "server": runtime["server"]["path"],
        "model": str(model_path),
        "model_name": f"bonsai2-27b-{spec.quantization.lower()}",
        "context": context,
        "host": "127.0.0.1",
        "port": 18081,
        "artifact_manifest": str(RECEIPT),
        "lib_dirs": list(runtime["lib_dirs"]),
        "args": [
            "--gpu-layers", "all",
            "--fit", "off",
            "--parallel", "1",
            "--jinja",
            "--no-webui",
            "--no-slots",
        ],
        "shards": [{"name": spec.name, "size": spec.size, "sha256": spec.sha256}],
    }


def make_receipt(
    *,
    spec: ModelSpec,
    model_path: Path,
    runtime: Mapping[str, Any],
    model: Mapping[str, Any] | None,
    other_models: Sequence[Mapping[str, Any]] = (),
    context: int = 32768,
) -> dict[str, Any]:
    model_complete = model is not None
    verified_models = ([dict(model)] if model is not None else []) + [
        dict(candidate) for candidate in other_models
    ]
    for candidate in verified_models:
        template = candidate.get("chat_template")
        _require(
            isinstance(template, Mapping)
            and template.get("verified") is True
            and template.get("sha256") == CHAT_TEMPLATE_SHA256
            and isinstance(template.get("capabilities"), Mapping)
            and template["capabilities"].get("enable_thinking_false") is True,
            f"verified model record lacks enable_thinking=false template attestation: {candidate['name']}",
        )
    verified_models.sort(key=lambda candidate: candidate["name"])
    all_packing_names = {candidate.name for candidate in MODELS.values()}
    all_packings_verified = {candidate["name"] for candidate in verified_models} == all_packing_names
    shared_template_capabilities = (
        {
            "chat_template_sha256": CHAT_TEMPLATE_SHA256,
            "enable_thinking_false_supported": True,
            "enable_thinking_false_behavior": "empty_think_block",
        }
        if all_packings_verified
        else None
    )
    return {
        "schema_version": 1,
        "checked_at": utc_now(),
        "state": "ready_for_gpu_trial_unvalidated" if model_complete else "runtime_ready_model_pending",
        "source": {
            "model_repo": MODEL_REPO,
            "model_revision": MODEL_REVISION,
            "demo_repo": DEMO_REPO,
            "demo_revision": DEMO_REVISION,
            "llama_repo": LLAMA_REPO,
            "llama_release": LLAMA_RELEASE,
            "llama_revision": LLAMA_REVISION,
            "official_runtime_instructions": "https://github.com/PrismML-Eng/Bonsai-demo",
            "official_model_card": f"https://huggingface.co/{MODEL_REPO}",
        },
        "runtime": dict(runtime),
        "model": {
            "selected_quantization": spec.quantization,
            "path": str(model_path),
            "name": spec.name,
            "size": spec.size,
            "sha256": spec.sha256,
            "url": spec.url,
            "present_and_verified": model_complete,
            "chat_template": dict(model["chat_template"]) if model is not None else None,
        },
        "available_model_packings": [
            {
                "quantization": candidate.quantization,
                "name": candidate.name,
                "size": candidate.size,
                "sha256": candidate.sha256,
                "url": candidate.url,
            }
            for candidate in MODELS.values()
        ],
        "shards": verified_models,
        "template_capabilities": shared_template_capabilities,
        "trial_config_template": trial_config_template(
            spec, model_path, runtime, context=context
        ),
        "trial_config_templates": {
            candidate.quantization: trial_config_template(
                candidate, candidate.default_path, runtime, context=context
            )
            for candidate in MODELS.values()
        },
        "source_build_fallback": {
            "selected_cuda_architecture": T4_CUDA_ARCHITECTURE,
            "cuda_toolkit": str(CUDA_TOOLKIT),
            "max_parallel_jobs": MAX_BUILD_JOBS,
            "commands": list(SOURCE_BUILD_COMMANDS),
            "used": True,
        },
        "study_profile": {
            "context": context,
            "quality_tier": "medium",
            "code_output_tokens": 4096,
            "prose_output_tokens": 2048,
            "low_tier_enabled": False,
            "placement": "all_layers",
            "automatic_fit": False,
        },
        "verdict": {
            "runtime_artifact_complete": True,
            "static_tesla_t4_compatibility": True,
            "model_artifact_complete": model_complete,
            "artifact_complete": model_complete,
            "shared_template_capabilities_verified": all_packings_verified,
            "gpu_load_proven": False,
            "performance_measured": False,
        },
        "notes": [
            "PQ2_0 and PTQ1_0 require the PrismML llama.cpp fork; stock Q2 support is not sufficient.",
            "Tesla T4 compatibility is based on offline sm_75 fatbinary inspection only.",
            "No CUDA backend was loaded and no model inference or performance measurement was run.",
        ],
    }


def context_value(raw: str) -> int:
    try:
        value = int(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("context must be an integer") from exc
    if not 512 <= value <= 262_144:
        raise argparse.ArgumentTypeError("context must be between 512 and 262144")
    return value


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--install-runtime", action="store_true", help="download and extract the pinned runtime")
    parser.add_argument("--quantization", choices=tuple(MODELS), default="PQ2_0")
    parser.add_argument("--model-path", type=Path, help="override the selected model path")
    parser.add_argument("--context", type=context_value, default=8192)
    parser.add_argument("--receipt", type=Path, default=RECEIPT)
    parser.add_argument(
        "--require-model",
        action="store_true",
        help="fail instead of publishing a runtime-only receipt when the model is absent",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    spec = MODELS[args.quantization]
    model_path = (args.model_path or spec.default_path).expanduser().resolve()
    try:
        if args.install_runtime:
            install_runtime()
        archive_record = verify_regular_file(
            ARCHIVE, size=ARCHIVE_SIZE, sha256=ARCHIVE_SHA256, label="runtime archive"
        )
        runtime = inspect_runtime(archive_record)
        verified_by_quantization: dict[str, dict[str, Any]] = {}
        for quantization, candidate in MODELS.items():
            candidate_path = model_path if quantization == spec.quantization else candidate.default_path
            if candidate_path.exists():
                verified_by_quantization[quantization] = verify_model(candidate, candidate_path)
        model = verified_by_quantization.get(spec.quantization)
        if args.require_model and model is None:
            raise PreparationError(f"selected model is absent: {model_path}")
        receipt = make_receipt(
            spec=spec,
            model_path=model_path,
            runtime=runtime,
            model=model,
            other_models=[
                record
                for quantization, record in verified_by_quantization.items()
                if quantization != spec.quantization
            ],
            context=args.context,
        )
        atomic_write_json(args.receipt, receipt)
    except (OSError, subprocess.SubprocessError, tarfile.TarError, PreparationError) as exc:
        print(f"bonsai2 preparation failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(receipt["verdict"], sort_keys=True))
    print(f"receipt: {args.receipt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
