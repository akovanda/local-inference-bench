#!/usr/bin/env python3
"""Download, verify, and prepare the pinned Flash-Next IQ3_S artifacts."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import struct
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
STRATA_ROOT = ROOT / "vendor" / "strata"
STRATA_REVISION = "477ac13cbe8e582c20484871ce459d20d82df09d"
REPO = "ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF"
REVISION = "ed59f92082b1e93c0e96d60a8b11aab089b52f09"
MODEL_DIR = Path("/external/storage/moe-bench/models/flash-next-IQ3_S")
PACK_DIR = Path("/external/storage/moe-bench/packs/flash-next-iq3_s")
CONFIG = ROOT / "runtime" / "flash-next-greedy-mtp0-32k.json"
ARTIFACT = ROOT / "results" / "flash-next-artifact.json"
ENGINE = Path("/external/storage/moe-bench/strata-build/strata")
PYTHON = STRATA_ROOT / ".venv" / "bin" / "python"
GGUF_PY = STRATA_ROOT / "third_party" / "llama.cpp" / "gguf-py"
EXPERT_PROFILE = STRATA_ROOT / "data" / "expert-profile.bin"
EXPERT_PROFILE_SIZE = 196_632
EXPERT_PROFILE_SHA256 = "8f59b4aa8873209dff11c11e37bcda9529a1335b724a1afeea37bf6388975baf"
EXPERT_PROFILE_GEOMETRY = (48, 512)
EXPERT_PROFILE_PAIRS = 48 * 512


@dataclass(frozen=True)
class FileSpec:
    name: str
    size: int
    sha256: str

    @property
    def path(self) -> Path:
        return MODEL_DIR / self.name

    @property
    def url(self) -> str:
        return f"https://huggingface.co/{REPO}/resolve/{REVISION}/IQ3_S/{self.name}"


FILES = (
    FileSpec(
        "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf",
        54_817_524_224,
        "4c1eb2ceb4915e1192f4f386021897bde56a97f40a0bb78bb86465e0f7d2aca3",
    ),
    FileSpec(
        "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf",
        28_800_138_432,
        "316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113",
    ),
)

REQUIRED_PACK_FILES = (
    "dense.bin",
    "index.txt",
    "native_experts.txt",
    "tokenizer/chat_template.jinja",
    "tokenizer/merges.txt",
    "tokenizer/token_type.json",
    "tokenizer/tokenizer.json",
    "tokenizer/vocab.json",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(16 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def verify_file(spec: FileSpec) -> dict[str, Any]:
    try:
        before = spec.path.stat()
    except FileNotFoundError as exc:
        raise RuntimeError(f"missing model shard: {spec.path}") from exc
    if not spec.path.is_file() or spec.path.is_symlink():
        raise RuntimeError(f"model shard is not a regular file: {spec.path}")
    if before.st_size != spec.size:
        raise RuntimeError(
            f"{spec.name} is {before.st_size} bytes; expected {spec.size}"
        )
    actual = sha256_file(spec.path)
    after = spec.path.stat()
    identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if identity_before != identity_after:
        raise RuntimeError(f"model shard changed while it was hashed: {spec.path}")
    if actual != spec.sha256:
        raise RuntimeError(
            f"{spec.name} SHA-256 is {actual}; expected {spec.sha256}"
        )
    return {
        "path": str(spec.path),
        "size": spec.size,
        "sha256": actual,
        "verified": True,
    }


def verify_expert_profile() -> dict[str, Any]:
    try:
        profile = EXPERT_PROFILE.read_bytes()
    except OSError as exc:
        raise RuntimeError(f"cannot read the pinned Flash-Next expert profile: {EXPERT_PROFILE}") from exc
    actual_sha256 = hashlib.sha256(profile).hexdigest()
    if len(profile) != EXPERT_PROFILE_SIZE or actual_sha256 != EXPERT_PROFILE_SHA256:
        raise RuntimeError(
            f"expert profile is not the file pinned by Strata {STRATA_REVISION}: "
            f"size {len(profile)}, SHA-256 {actual_sha256}"
        )
    if profile[:4] != b"STRP" or len(profile) < 24:
        raise RuntimeError("expert profile has an invalid Strata header")
    version, layers, experts, slots, ranked = struct.unpack_from("<5I", profile, 4)
    if (
        version != 1
        or (layers, experts) != EXPERT_PROFILE_GEOMETRY
        or slots != EXPERT_PROFILE_PAIRS
        or ranked != EXPERT_PROFILE_PAIRS
    ):
        raise RuntimeError(
            "expert profile must be a complete v1 ranking for all 48 x 512 Flash-Next experts"
        )
    return {
        "path": str(EXPERT_PROFILE),
        "size": len(profile),
        "sha256": actual_sha256,
        "format_version": version,
        "layers": layers,
        "experts_per_layer": experts,
        "ranked_pairs": ranked,
        "verified": True,
    }


def download_missing(workers_per_file: int) -> set[Path]:
    missing = [spec for spec in FILES if not spec.path.exists()]
    if not missing:
        return set()
    sys.path.insert(0, str(ROOT / "runtime"))
    from range_fetch import download_file

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    def fetch(spec: FileSpec) -> Path:
        print(f"Downloading pinned shard: {spec.name}", flush=True)
        return download_file(
            spec.url,
            spec.path,
            spec.size,
            spec.sha256,
            workers=workers_per_file,
            chunk_size=32 * 1024 * 1024,
            retries=5,
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(missing)) as executor:
        futures = {executor.submit(fetch, spec): spec for spec in missing}
        for future in concurrent.futures.as_completed(futures):
            future.result()
    return {spec.path for spec in missing}


def strata_revision() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=STRATA_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def pack_provenance() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "repo": REPO,
        "revision": REVISION,
        "quantization": "IQ3_S",
        "files": [
            {"name": spec.name, "size": spec.size, "sha256": spec.sha256}
            for spec in FILES
        ],
        "strata_revision": STRATA_REVISION,
        "command": [
            str(PYTHON),
            str(STRATA_ROOT / "tools" / "iq_pack.py"),
            "--gguf",
            str(FILES[0].path),
            "--out",
            str(PACK_DIR),
        ],
    }


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
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


def validate_pack() -> list[dict[str, Any]]:
    records = []
    for relative in REQUIRED_PACK_FILES:
        path = PACK_DIR / relative
        if not path.is_file() or path.is_symlink() or path.stat().st_size <= 0:
            raise RuntimeError(f"prepared pack is missing a regular nonempty {relative}")
        records.append({"path": str(path), "size": path.stat().st_size})
    return records


def prepare_pack() -> tuple[list[dict[str, Any]], bool]:
    expected = pack_provenance()
    marker = PACK_DIR / "flash-next-source.json"
    try:
        existing = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        existing = None
    if existing == expected:
        return validate_pack(), False

    for required in (PYTHON, GGUF_PY, STRATA_ROOT / "tools" / "iq_pack.py"):
        if not required.exists():
            raise RuntimeError(f"required Strata preparation path is missing: {required}")
    PACK_DIR.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, STRATA_GGUF_PY=str(GGUF_PY))
    subprocess.run(expected["command"], cwd=STRATA_ROOT, env=env, check=True)
    records = validate_pack()
    atomic_write_json(marker, expected)
    records.append({"path": str(marker), "size": marker.stat().st_size})
    return records, True


def validate_runtime_config() -> dict[str, Any]:
    try:
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"runtime config is unreadable or invalid: {CONFIG}") from exc
    required_pairs = {
        "--pack": str(PACK_DIR),
        "--native": str(FILES[0].path),
        "--ple-gguf": str(FILES[1].path),
        "--ple-io": "ram",
        "--max-context": "32768",
        "--prefill": "auto",
        "--expert-profile": str(EXPERT_PROFILE),
        "--expert-cache": "auto",
        "--spec": "4",
        "--suffix-draft": "0",
    }
    args = config.get("args")
    if not isinstance(args, list) or any(not isinstance(value, str) for value in args):
        raise RuntimeError("runtime config args must be a list of strings")
    for flag, expected in required_pairs.items():
        positions = [index for index, value in enumerate(args) if value == flag]
        if len(positions) != 1 or positions[0] + 1 >= len(args) or args[positions[0] + 1] != expected:
            raise RuntimeError(f"runtime config must contain exactly {flag} {expected}")
    if "--mtp" in args:
        raise RuntimeError("baseline runtime config unexpectedly enables an MTP model")
    expected_fields = {
        "exe": str(ENGINE),
        "cwd": str(STRATA_ROOT),
        "tokenizer": str(PACK_DIR / "tokenizer"),
        "model_name": "qwen3.8-flash-next-iq3_s",
        "arch": "qwen4exp",
        "lib_dirs": ["/usr/local/cuda-13.0/lib64"],
    }
    for field, expected in expected_fields.items():
        if config.get(field) != expected:
            raise RuntimeError(f"runtime config {field} must be {expected!r}")
    if config.get("host") != "127.0.0.1" or config.get("port") != 18080:
        raise RuntimeError("runtime config must use loopback port 18080")
    if config.get("open_browser") is not False:
        raise RuntimeError("runtime config must leave the browser disabled")
    if config.get("sampling") != {"temperature": 0, "presence_penalty": 0}:
        raise RuntimeError("runtime config must use greedy sampling with zero presence penalty")
    return config


def build_artifact(
    verified_files: list[dict[str, Any]],
    pack_files: list[dict[str, Any]],
    pack_rebuilt: bool,
    expert_profile: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "state": "ready",
        "checked_at": utc_now(),
        "model": "Qwen3.8-Flash-Next",
        "model_name": "qwen3.8-flash-next-iq3_s",
        "arch": "qwen4exp",
        "quantization": "IQ3_S",
        "source": {
            "provider": "Hugging Face",
            "repo": REPO,
            "revision": REVISION,
            "subdirectory": "IQ3_S",
        },
        "files": verified_files,
        "pack": {
            "path": str(PACK_DIR),
            "rebuilt": pack_rebuilt,
            "experts": "read from the verified GGUF shards into Strata's resident RAM arena",
            "files": pack_files,
        },
        "runtime": {
            "config": str(CONFIG),
            "engine": str(ENGINE),
            "strata_revision": STRATA_REVISION,
            "max_context": 32768,
            "ple_io": "ram",
            "expert_cache": {
                "sizing": "auto",
                "startup_policy": "pre-filled by the complete pinned routing-frequency ranking",
                "profile": expert_profile,
            },
            "mtp": "disabled; no additional MTP checkpoint was downloaded",
            "sampling": {"temperature": 0, "presence_penalty": 0},
        },
    }


def positive_workers(value: str) -> int:
    parsed = int(value)
    if parsed < 1 or parsed > 16:
        raise argparse.ArgumentTypeError("must be between 1 and 16")
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workers-per-file",
        type=positive_workers,
        default=4,
        help="range download workers for each missing shard (default: 4)",
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="require the two final shard files to exist instead of resuming downloads",
    )
    args = parser.parse_args()

    if strata_revision() != STRATA_REVISION:
        raise RuntimeError(f"Strata checkout is not pinned to {STRATA_REVISION}")
    if not ENGINE.is_file() or not os.access(ENGINE, os.X_OK):
        raise RuntimeError(f"pinned Strata engine is missing or not executable: {ENGINE}")
    validate_runtime_config()
    expert_profile = verify_expert_profile()
    downloaded = set() if args.no_download else download_missing(args.workers_per_file)

    verified_files = []
    for spec in FILES:
        if spec.path in downloaded:
            # range_fetch publishes only after exact size and whole-file SHA-256 verification.
            verified_files.append(
                {"path": str(spec.path), "size": spec.size, "sha256": spec.sha256, "verified": True}
            )
        else:
            print(f"Checking pinned SHA-256: {spec.name}", flush=True)
            verified_files.append(verify_file(spec))

    pack_files, rebuilt = prepare_pack()
    artifact = build_artifact(verified_files, pack_files, rebuilt, expert_profile)
    atomic_write_json(ARTIFACT, artifact)
    print(f"Flash-Next IQ3_S is prepared: {ARTIFACT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
