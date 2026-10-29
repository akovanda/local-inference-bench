#!/usr/bin/env python3
"""Verify and prepare the pinned Flash-Next Coder IQ1_M Strata artifact.

The first split shard must already be downloaded.  The second split shard is
shared with the verified Flash-Next artifact and must be exposed through the
coder-named sibling symlink plus its fresh, inode-bound verification receipt.
This helper never downloads or copies either model shard and never starts a
server or GPU workload.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import struct
import subprocess
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__:
    from . import range_fetch
else:
    import range_fetch


ROOT = Path(__file__).resolve().parents[1]
STRATA_ROOT = ROOT / "vendor" / "strata"
STRATA_REVISION = "477ac13cbe8e582c20484871ce459d20d82df09d"
REPO = "ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF"
REVISION = "bba88856f8c3c0c9e40e45f82d05e228b814ac43"
SUBDIRECTORY = "IQ1_M"
MODEL_DIR = Path("/external/storage/moe-bench/models/flash-next-coder-IQ1_M")
PACK_DIR = Path("/external/storage/moe-bench/packs/flash-next-coder-iq1_m")
CONFIG = ROOT / "runtime" / "flash-next-coder-reference-8k.json"
ARTIFACT = ROOT / "results" / "flash-next-coder-artifact.json"
SHARED_RECEIPT = ROOT / "results" / "flash-next-coder-shared-ple-verification.json"
ENGINE = Path("/external/storage/moe-bench/strata-build/strata")
ENGINE_SIZE = 79_414_688
ENGINE_SHA256 = "15ba7c4dce7d08760348c16d39fb2ed0f37ac47004d898e1ea9b8d898ea72fd8"
PYTHON = STRATA_ROOT / ".venv" / "bin" / "python"
IQ_PACK = STRATA_ROOT / "tools" / "iq_pack.py"
IQ_PACK_SHA256 = "21b73cdb06031d2fe65bd87ebcd1ccfd8c40ac15cdb1c54d6dbe84d233c8049b"
GGUF_PY = STRATA_ROOT / "third_party" / "llama.cpp" / "gguf-py"
EXPERT_PROFILE = STRATA_ROOT / "data" / "expert-profile-coder.bin"
EXPERT_PROFILE_SIZE = 98_328
EXPERT_PROFILE_SHA256 = "a514661124d5305effe9c1bcecff7dc069af398b2b19ff9d11628a2cad0a8915"
PROFILE_LAYERS = 48
PROFILE_EXPERTS = 256
PROFILE_PAIRS = PROFILE_LAYERS * PROFILE_EXPERTS
MODEL_NAME = "qwen3.8-flash-next-coder-iq1_m"
CONTEXT = 8_192


@dataclass(frozen=True)
class FileSpec:
    name: str
    size: int
    sha256: str

    @property
    def path(self) -> Path:
        return MODEL_DIR / self.name


FIRST = FileSpec(
    "Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf",
    29_608_446_496,
    "bfa3644db4adfff072fc946c20994f2909d2cf89879d83f80a4c811d4016c211",
)
SECOND = FileSpec(
    "Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00002-of-00002.gguf",
    28_800_138_432,
    "316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113",
)
SHARED_SOURCE = Path(
    "/external/storage/moe-bench/models/flash-next-IQ3_S/"
    "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf"
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


class PreparationError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    return range_fetch.hash_file(path)


def _identity(details: os.stat_result) -> dict[str, int]:
    return {
        "device": details.st_dev,
        "inode": details.st_ino,
        "size": details.st_size,
        "mtime_ns": details.st_mtime_ns,
        "ctime_ns": details.st_ctime_ns,
    }


def _regular(path: Path, size: int, description: str) -> os.stat_result:
    try:
        details = path.lstat()
    except OSError as exc:
        raise PreparationError(f"{description} is inaccessible: {path}") from exc
    if not stat.S_ISREG(details.st_mode) or path.is_symlink():
        raise PreparationError(f"{description} must be a regular non-symlink file: {path}")
    if details.st_size != size:
        raise PreparationError(
            f"{description} is {details.st_size} bytes; expected exactly {size}: {path}"
        )
    return details


def verify_first_shard(spec: FileSpec = FIRST) -> dict[str, Any]:
    before = _regular(spec.path, spec.size, "coder first shard")
    actual = sha256_file(spec.path)
    after = _regular(spec.path, spec.size, "coder first shard")
    if _identity(before) != _identity(after):
        raise PreparationError("coder first shard changed while its full SHA-256 was computed")
    if actual != spec.sha256:
        raise PreparationError(
            f"coder first shard SHA-256 is {actual}; expected {spec.sha256}"
        )
    return {
        "name": spec.name,
        "path": str(spec.path),
        "size": spec.size,
        "sha256": actual,
        "verified": True,
        "full_sha256_verified": True,
        "file_identity_after_verification": _identity(after),
    }


def _read_json(path: Path, description: str) -> dict[str, Any]:
    try:
        details = path.lstat()
        if not stat.S_ISREG(details.st_mode) or path.is_symlink():
            raise PreparationError(f"{description} must be a regular non-symlink file")
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PreparationError(f"{description} is unreadable or invalid: {path}") from exc
    if not isinstance(value, dict):
        raise PreparationError(f"{description} must be a JSON object")
    return value


def verify_shared_ple(
    link: Path | None = None,
    source: Path = SHARED_SOURCE,
    receipt_path: Path = SHARED_RECEIPT,
    spec: FileSpec = SECOND,
) -> dict[str, Any]:
    link = spec.path if link is None else link
    source_details = _regular(source, spec.size, "shared PLE source")
    receipt = _read_json(receipt_path, "shared PLE verification receipt")
    identity = receipt.get("identity")
    expected_identity = {
        "device": source_details.st_dev,
        "inode": source_details.st_ino,
        "mtime_ns": source_details.st_mtime_ns,
        "ctime_ns": source_details.st_ctime_ns,
    }
    if (
        receipt.get("schema_version") != 1
        or receipt.get("verified") is not True
        or receipt.get("path") != str(source)
        or receipt.get("size") != spec.size
        or receipt.get("sha256") != spec.sha256
        or receipt.get("reuse_for") != "flash-next-coder"
        or identity != expected_identity
    ):
        raise PreparationError(
            "shared PLE receipt does not match the pinned digest and current source identity"
        )
    try:
        link_details = link.lstat()
    except OSError as exc:
        raise PreparationError(f"coder-named PLE sibling is inaccessible: {link}") from exc
    if not stat.S_ISLNK(link_details.st_mode):
        raise PreparationError("coder-named PLE sibling must be a symlink, not a copy")
    try:
        resolved = link.resolve(strict=True)
    except OSError as exc:
        raise PreparationError("coder-named PLE symlink is broken") from exc
    if resolved != source.resolve(strict=True):
        raise PreparationError("coder-named PLE symlink resolves to the wrong source")
    linked = link.stat()
    if _identity(linked) != _identity(source_details):
        raise PreparationError("coder-named PLE symlink does not expose the verified source inode")
    return {
        "name": spec.name,
        "path": str(link),
        "size": spec.size,
        "sha256": spec.sha256,
        "verified": True,
        "full_sha256_verified": True,
        "verification_reused_from": str(receipt_path),
        "verification_receipt_sha256": sha256_file(receipt_path),
        "symlink": {
            "path": str(link),
            "target": os.readlink(link),
            "resolved_source": str(source),
        },
        "source_file_identity": _identity(source_details),
    }


def verify_expert_profile(
    path: Path = EXPERT_PROFILE,
    *,
    expected_size: int = EXPERT_PROFILE_SIZE,
    expected_sha256: str = EXPERT_PROFILE_SHA256,
) -> dict[str, Any]:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise PreparationError(f"coder expert profile is unreadable: {path}") from exc
    actual_sha = hashlib.sha256(payload).hexdigest()
    if len(payload) != expected_size or actual_sha != expected_sha256:
        raise PreparationError("coder expert profile does not match its pinned size and SHA-256")
    if len(payload) < 24 or payload[:4] != b"STRP":
        raise PreparationError("coder expert profile has an invalid header")
    version, layers, experts, slots, ranked = struct.unpack_from("<5I", payload, 4)
    if (version, layers, experts, slots, ranked) != (
        1,
        PROFILE_LAYERS,
        PROFILE_EXPERTS,
        PROFILE_PAIRS,
        PROFILE_PAIRS,
    ):
        raise PreparationError("coder expert profile has the wrong v1 geometry")
    expected_bytes = 24 + ranked * 4 + layers * experts * 4
    if len(payload) != expected_bytes:
        raise PreparationError("coder expert profile length does not match its ranked list and table")
    pairs = [struct.unpack_from("<HH", payload, 24 + index * 4) for index in range(ranked)]
    if any(layer >= layers or expert >= experts for layer, expert in pairs):
        raise PreparationError("coder expert profile has an out-of-range ranked pair")
    if len(set(pairs)) != PROFILE_PAIRS:
        raise PreparationError("coder expert profile ranked pairs are not complete and unique")
    table_offset = 24 + ranked * 4
    inverse = struct.unpack_from(f"<{layers * experts}i", payload, table_offset)
    expected_inverse = [-1] * (layers * experts)
    for slot, (layer, expert) in enumerate(pairs):
        expected_inverse[layer * experts + expert] = slot
    if list(inverse) != expected_inverse:
        raise PreparationError("coder expert profile inverse-slot table disagrees with its ranking")
    return {
        "path": str(path),
        "size": len(payload),
        "sha256": actual_sha,
        "format_version": version,
        "layers": layers,
        "experts_per_layer": experts,
        "ranked_pairs": ranked,
        "unique_pairs": len(set(pairs)),
        "inverse_table_validated": True,
        "verified": True,
    }


def verify_toolchain() -> dict[str, Any]:
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=STRATA_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=10,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        raise PreparationError("pinned Strata source revision could not be verified") from exc
    if revision != STRATA_REVISION:
        raise PreparationError(f"Strata checkout is {revision}; expected {STRATA_REVISION}")
    _regular(ENGINE, ENGINE_SIZE, "Strata engine")
    if not os.access(ENGINE, os.X_OK) or sha256_file(ENGINE) != ENGINE_SHA256:
        raise PreparationError("Strata engine does not match the pinned executable identity")
    if not IQ_PACK.is_file() or IQ_PACK.is_symlink() or sha256_file(IQ_PACK) != IQ_PACK_SHA256:
        raise PreparationError("iq_pack.py does not match the pinned Strata source identity")
    if not PYTHON.is_file() or not os.access(PYTHON, os.X_OK) or not GGUF_PY.is_dir():
        raise PreparationError("Strata preparation Python or gguf-py is unavailable")
    return {
        "strata_revision": revision,
        "engine": {"path": str(ENGINE), "size": ENGINE_SIZE, "sha256": ENGINE_SHA256},
        "iq_pack": {"path": str(IQ_PACK), "sha256": IQ_PACK_SHA256},
        "python": str(PYTHON),
        "gguf_py": str(GGUF_PY),
    }


def verification_document() -> dict[str, Any]:
    first = verify_first_shard()
    second = verify_shared_ple()
    profile = verify_expert_profile()
    tools = verify_toolchain()
    return {
        "schema_version": 1,
        "checked_at": utc_now(),
        "model": {
            "repo": REPO,
            "revision": REVISION,
            "subdirectory": SUBDIRECTORY,
            "model_name": MODEL_NAME,
            "arch": "qwen4exp",
            "quantization": "IQ1_M",
        },
        "shards": [first, second],
        "expert_profile": profile,
        "toolchain": tools,
        "runtime_validation": {
            "full_weight_load_validated": False,
            "inference_validated": False,
            "gpu_placement_validated": False,
        },
    }


def pack_provenance(verification: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "repo": REPO,
        "revision": REVISION,
        "subdirectory": SUBDIRECTORY,
        "shards": [
            {"name": row["name"], "size": row["size"], "sha256": row["sha256"]}
            for row in verification["shards"]
        ],
        "shared_ple_receipt": {
            "path": str(SHARED_RECEIPT),
            "sha256": verification["shards"][1]["verification_receipt_sha256"],
        },
        "expert_profile": {
            "path": str(EXPERT_PROFILE),
            "sha256": EXPERT_PROFILE_SHA256,
        },
        "toolchain": verification["toolchain"],
        "command": [
            str(PYTHON),
            str(IQ_PACK),
            "--gguf",
            str(FIRST.path),
            "--out",
            str(PACK_DIR),
        ],
    }


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_json_exclusive(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}.{uuid.uuid4().hex}")
    descriptor = -1
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        view = memoryview(payload)
        offset = 0
        while offset < len(view):
            written = os.write(descriptor, view[offset:])
            if written <= 0:
                raise OSError("short immutable JSON write")
            offset += written
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise PreparationError(f"immutable output already exists: {path}") from exc
        _fsync_directory(path.parent)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def validate_pack(pack_dir: Path = PACK_DIR) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for relative in REQUIRED_PACK_FILES:
        path = pack_dir / relative
        try:
            details = path.lstat()
        except OSError as exc:
            raise PreparationError(f"prepared pack is missing {relative}") from exc
        if not stat.S_ISREG(details.st_mode) or path.is_symlink() or details.st_size <= 0:
            raise PreparationError(f"prepared pack {relative} must be a regular nonempty file")
        records.append({"path": str(path), "size": details.st_size})
    return records


def prepare_pack(verification: Mapping[str, Any]) -> tuple[list[dict[str, Any]], bool, dict[str, Any]]:
    provenance = pack_provenance(verification)
    marker = PACK_DIR / "flash-next-coder-source.json"
    if os.path.lexists(marker):
        existing = _read_json(marker, "Flash-Next Coder pack marker")
        if existing != provenance:
            raise PreparationError("existing Coder pack provenance differs from the pinned inputs")
        return validate_pack(PACK_DIR), False, provenance
    if PACK_DIR.exists():
        try:
            nonempty = next(PACK_DIR.iterdir(), None) is not None
        except OSError as exc:
            raise PreparationError("Coder pack directory cannot be inspected") from exc
        if nonempty:
            raise PreparationError("Coder pack directory is nonempty without its provenance marker")
    else:
        PACK_DIR.mkdir(parents=True)
    env = dict(os.environ, STRATA_GGUF_PY=str(GGUF_PY))
    try:
        subprocess.run(
            provenance["command"], cwd=STRATA_ROOT, env=env, check=True
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise PreparationError("iq_pack.py failed; partial pack was preserved for inspection") from exc
    files = validate_pack(PACK_DIR)
    write_json_exclusive(marker, provenance)
    files.append({"path": str(marker), "size": marker.stat().st_size})
    return files, True, provenance


def config_document() -> dict[str, Any]:
    return {
        "exe": str(ENGINE),
        "args": [
            "--pack", str(PACK_DIR),
            "--native", str(FIRST.path),
            "--ple-gguf", str(SECOND.path),
            "--ple-io", "ram",
            "--max-context", str(CONTEXT),
            "--prefill", "auto",
            "--expert-profile", str(EXPERT_PROFILE),
            "--expert-cache", "auto",
            "--spec", "4",
            "--suffix-draft", "0",
        ],
        "cwd": str(STRATA_ROOT),
        "tokenizer": str(PACK_DIR / "tokenizer"),
        "model_name": MODEL_NAME,
        "arch": "qwen4exp",
        "log": str(ROOT / "results" / "strata-flash-next-coder-reference-engine.log"),
        "lib_dirs": ["/usr/local/cuda-13.0/lib64"],
        "host": "127.0.0.1",
        "port": 18080,
        "open_browser": False,
        "sampling": {"temperature": 0, "presence_penalty": 0},
        "reference_environment": {
            "STRATA_ROUTE_TAIL_SKIP": "0",
            "applied_by_this_config": False,
            "caller_must_set_and_verify_in_runtime_events": True,
        },
    }


def publish_config(path: Path = CONFIG) -> dict[str, Any]:
    document = config_document()
    if os.path.lexists(path):
        if _read_json(path, "Flash-Next Coder config") != document:
            raise PreparationError(f"refusing to overwrite different existing config: {path}")
        return document
    write_json_exclusive(path, document)
    return document


def build_artifact(
    verification: Mapping[str, Any],
    pack_files: Sequence[Mapping[str, Any]],
    pack_rebuilt: bool,
    provenance: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "immutable": True,
        "state": "prepared_runtime_unvalidated",
        "checked_at": utc_now(),
        "verdict": {
            "artifact_complete": True,
            "full_sha256_verified": True,
            "shared_ple_identity_bound": True,
            "pack_complete": True,
            "runtime_load_validated": False,
            "inference_validated": False,
        },
        "model": verification["model"],
        "shards": verification["shards"],
        "expert_profile": verification["expert_profile"],
        "pack": {
            "path": str(PACK_DIR),
            "rebuilt": pack_rebuilt,
            "provenance": provenance,
            "files": list(pack_files),
        },
        "runtime": {
            "config": str(CONFIG),
            "config_sha256": sha256_file(CONFIG),
            "engine": verification["toolchain"]["engine"],
            "strata_revision": STRATA_REVISION,
            "max_context": CONTEXT,
            "full_weight_load_validated": False,
            "actual_gpu_placement_validated": False,
            "reference_environment": config["reference_environment"],
        },
        "helper": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
            "iq_pack_sha256": IQ_PACK_SHA256,
        },
        "mtp": {
            "enabled": False,
            "companion_available": False,
            "prior_fetch_result": "policy_approval_rejected",
            "detail": "approval required by policy, but AskForApproval is set to Never",
            "reviewer_origin": "unknown",
            "retried": False,
        },
    }


def prepare() -> dict[str, Any]:
    if os.path.lexists(ARTIFACT):
        raise PreparationError(f"immutable artifact receipt already exists: {ARTIFACT}")
    verification = verification_document()
    pack_files, rebuilt, provenance = prepare_pack(verification)
    config = publish_config()
    artifact = build_artifact(verification, pack_files, rebuilt, provenance, config)
    write_json_exclusive(ARTIFACT, artifact)
    return artifact


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--verify", action="store_true", help="full verification only; write nothing")
    action.add_argument("--prepare", action="store_true", help="verify, pack, and publish config/receipt")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = prepare() if args.prepare else verification_document()
    except (PreparationError, OSError, subprocess.SubprocessError) as exc:
        print(f"Flash-Next Coder preparation failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
