#!/usr/bin/env python3
"""Inspect, download, verify, and configure pinned first-wave specialist GGUFs.

The helper has no server or GPU actions.  ``--inspect`` is read-only;
``--download`` uses the repository's resumable range downloader; and
``--prepare-config`` only publishes a config after an immutable full-SHA
artifact receipt exists (or after creating that receipt from a fresh hash).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import quote

if __package__:
    from . import range_fetch
else:
    import range_fetch


ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = Path("/external/storage/moe-bench/models")
SERVER = Path("/external/storage/moe-bench/ollama-0.40.6/lib/ollama/llama-server")
LIB_DIRS = (
    Path("/external/storage/moe-bench/ollama-0.40.6/lib/ollama"),
    Path("/external/storage/moe-bench/ollama-0.40.6/lib/ollama/cuda_v13"),
)
LLAMA_REVISION = "620d2225cc436ab0b001e3200e590e3b286ef9ab"
CONTEXT = 8_192
HOST = "127.0.0.1"
PORT = 18_081
MAX_WORKERS = 16
HASH_RE = re.compile(r"^[0-9a-f]{64}$")


class PreparationError(RuntimeError):
    """Pinned metadata, local artifact, or publication validation failed."""


@dataclass(frozen=True)
class ModelSpec:
    key: str
    repo: str
    revision: str
    filename: str
    size: int
    sha256: str
    quantization: str
    alias: str
    directory_name: str
    receipt_name: str
    config_name: str
    role: str = "primary"

    @property
    def default_directory(self) -> Path:
        return MODEL_ROOT / self.directory_name

    @property
    def receipt_path(self) -> Path:
        return ROOT / "results" / self.receipt_name

    @property
    def config_path(self) -> Path:
        return ROOT / "runtime" / self.config_name

    @property
    def url(self) -> str:
        return (
            f"https://huggingface.co/{self.repo}/resolve/{self.revision}/"
            f"{quote(self.filename, safe='')}"
        )


SPECS = {
    "lfm25-qad-q4_0": ModelSpec(
        key="lfm25-qad-q4_0",
        repo="LiquidAI/LFM2.5-24B-A2B-GGUF",
        revision="0912d33141f77e10705345a6732bfd83f0252fdc",
        filename="LFM2.5-24B-A2B-QAD-Q4_0.gguf",
        size=13_578_190_880,
        sha256="d1c03f73ecaf61017d70419eec9e5dc0b05d50ae26c23d522829b934fbabf8a5",
        quantization="QAD-Q4_0",
        alias="lfm2.5-24b-a2b-qad-q4_0",
        directory_name="lfm25-qad-q4_0",
        receipt_name="lfm25-qad-q4_0-artifact.json",
        config_name="lfm25-qad-q4_0-8k.json",
    ),
    "kat-v25-iq3_xxs": ModelSpec(
        key="kat-v25-iq3_xxs",
        repo="bartowski/Kwaipilot_KAT-Coder-V2.5-Dev-GGUF",
        revision="d8f684f08d2950ea9d2db6a35ef7dada0707858b",
        filename="Kwaipilot_KAT-Coder-V2.5-Dev-IQ3_XXS.gguf",
        size=14_870_829_472,
        sha256="80453a60bfbd277ae6110413d5b7a4fef07d69997ef88e42adfe3cd650373f79",
        quantization="IQ3_XXS",
        alias="kat-coder-v2.5-dev-iq3_xxs",
        directory_name="kat-v25-iq3_xxs",
        receipt_name="kat-v25-iq3_xxs-artifact.json",
        config_name="kat-v25-iq3_xxs-8k.json",
    ),
    "gemma4-style12-q5_k_m": ModelSpec(
        key="gemma4-style12-q5_k_m",
        repo="mradermacher/Gemma-4-12B-StyleTune-GGUF",
        revision="fb2e8be6efb2905308e57ce248f209c6ddf99983",
        filename="Gemma-4-12B-StyleTune.Q5_K_M.gguf",
        size=9_239_328_384,
        sha256="14d78cfccb5f052a9c4bfcdceac30fae570a09dea4453c7d929fa57f703ff350",
        quantization="Q5_K_M",
        alias="gemma-4-12b-styletune-q5_k_m",
        directory_name="gemma4-style12-q5_k_m",
        receipt_name="gemma4-style12-q5_k_m-artifact.json",
        config_name="gemma4-style12-q5_k_m-8k.json",
    ),
}

IQ2_FALLBACK = {
    "key": "kat-v25-iq2_m",
    "repo": "bartowski/Kwaipilot_KAT-Coder-V2.5-Dev-GGUF",
    "revision": "d8f684f08d2950ea9d2db6a35ef7dada0707858b",
    "quantization": "IQ2_M",
    "role": "optional_fit_fallback",
    "availability": "metadata_unavailable",
    "activation_allowed": False,
    "reason": "exact pinned filename, LFS byte size, and SHA-256 are unavailable",
    "policy_event": {
        "action": "official Hugging Face tree API lookup for the IQ2_M LFS identity",
        "result": "policy_approval_rejected",
        "detail": "approval required by policy, but AskForApproval is set to Never",
        "reviewer_origin": "unknown",
        "retried": False,
    },
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def safe_error_detail(exc: BaseException) -> str:
    detail = " ".join(str(exc).split())
    detail = re.sub(r"https?://\S+", "[redacted-url]", detail, flags=re.IGNORECASE)
    return (detail or type(exc).__name__)[:500]


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_json_exclusive(path: Path, value: Any) -> None:
    """Durably publish a new JSON file without replacing an existing record."""
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
                raise OSError("short JSON write")
            offset += written
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise PreparationError(f"immutable record already exists: {path}") from exc
        _fsync_directory(path.parent)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def _read_json(path: Path, description: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise PreparationError(f"{description} is missing or not a regular file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PreparationError(f"{description} is unreadable or invalid: {path}") from exc
    if not isinstance(value, dict):
        raise PreparationError(f"{description} must be a JSON object: {path}")
    return value


def _identity(details: os.stat_result) -> dict[str, int]:
    return {
        "device": details.st_dev,
        "inode": details.st_ino,
        "size": details.st_size,
        "mtime_ns": details.st_mtime_ns,
        "ctime_ns": details.st_ctime_ns,
    }


def _regular_file(path: Path, expected_size: int) -> os.stat_result:
    try:
        details = path.lstat()
    except OSError as exc:
        raise PreparationError(f"artifact is inaccessible: {path}") from exc
    if not stat.S_ISREG(details.st_mode) or path.is_symlink():
        raise PreparationError(f"artifact must be a regular non-symlink file: {path}")
    if details.st_size != expected_size:
        raise PreparationError(
            f"artifact size is {details.st_size}; expected exactly {expected_size}: {path}"
        )
    return details


def verify_file(spec: ModelSpec, path: Path) -> dict[str, int]:
    before = _regular_file(path, spec.size)
    actual = range_fetch.hash_file(path)
    after = _regular_file(path, spec.size)
    if _identity(before) != _identity(after):
        raise PreparationError(f"artifact changed while it was hashed: {path}")
    if actual != spec.sha256:
        raise PreparationError(
            f"artifact SHA-256 is {actual}; expected {spec.sha256}: {path}"
        )
    return _identity(after)


def inspect_document(model: str | None = None) -> dict[str, Any]:
    keys = tuple(SPECS) if model in (None, "all") else (model,)
    primary = []
    for key in keys:
        if key not in SPECS:
            if key == IQ2_FALLBACK["key"]:
                primary.append(dict(IQ2_FALLBACK))
                continue
            raise PreparationError(f"unknown specialist model key: {key}")
        spec = SPECS[key]
        row = asdict(spec)
        row.update(
            {
                "default_directory": str(spec.default_directory),
                "receipt": str(spec.receipt_path),
                "config": str(spec.config_path),
                "metadata_pinned": True,
                "activation_allowed_after_full_sha256": True,
            }
        )
        primary.append(row)
    if model in (None, "all"):
        primary.append(dict(IQ2_FALLBACK))
    return {
        "schema_version": 1,
        "operation": "inspect",
        "mutates_files": False,
        "models": primary,
        "runtime": {
            "server": str(SERVER),
            "llama_cpp_revision": LLAMA_REVISION,
            "host": HOST,
            "port": PORT,
            "initial_context": CONTEXT,
            "gpu_placement": "auto; actual offloaded layers and peak VRAM must be measured",
        },
    }


def _artifact_path(spec: ModelSpec, model_dir: Path) -> Path:
    return model_dir / spec.filename


def resolve_model_dir(spec: ModelSpec, override: Path | None) -> Path:
    path = (spec.default_directory if override is None else override).expanduser().resolve()
    root = MODEL_ROOT.resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise PreparationError(f"model directory must remain under {root}: {path}") from exc
    if path == root:
        raise PreparationError("model directory must be a dedicated child of the model root")
    return path


def receipt_document(
    spec: ModelSpec,
    artifact: Path,
    identity: Mapping[str, int],
    verification_method: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "immutable": True,
        "checked_at": utc_now(),
        "state": "artifact_verified_runtime_unvalidated",
        "verdict": {
            "artifact_complete": True,
            "full_sha256_verified": True,
            "config_publication_allowed": True,
            "runtime_load_validated": False,
            "gpu_layer_allocation_validated": False,
        },
        "model": {
            "key": spec.key,
            "repo": spec.repo,
            "revision": spec.revision,
            "quantization": spec.quantization,
            "alias": spec.alias,
            "role": spec.role,
            "directory": str(artifact.parent),
        },
        "shards": [
            {
                "name": spec.filename,
                "path": str(artifact),
                "size": spec.size,
                "sha256": spec.sha256,
                "verified": True,
                "full_sha256_verified": True,
            }
        ],
        "verification": {
            "method": verification_method,
            "expected_sha256": spec.sha256,
            "actual_sha256": spec.sha256,
            "file_identity_after_verification": dict(identity),
        },
        "runtime": {
            "server": str(SERVER),
            "llama_cpp_revision": LLAMA_REVISION,
            "source_architecture_support_checked_by_parent": False,
            "actual_runtime_load_measured": False,
            "actual_gpu_layer_count": None,
            "peak_vram_bytes": None,
            "all_gpu_fit_claimed": False,
            "initial_context": CONTEXT,
        },
    }


def validate_receipt(receipt: Mapping[str, Any], spec: ModelSpec, artifact: Path) -> None:
    verdict = receipt.get("verdict")
    model = receipt.get("model")
    shards = receipt.get("shards")
    verification = receipt.get("verification")
    if (
        receipt.get("immutable") is not True
        or not isinstance(verdict, dict)
        or verdict.get("artifact_complete") is not True
        or verdict.get("full_sha256_verified") is not True
        or not isinstance(model, dict)
        or model.get("key") != spec.key
        or model.get("repo") != spec.repo
        or model.get("revision") != spec.revision
        or not isinstance(shards, list)
        or len(shards) != 1
        or not isinstance(shards[0], dict)
        or shards[0].get("name") != spec.filename
        or shards[0].get("path") != str(artifact)
        or shards[0].get("size") != spec.size
        or shards[0].get("sha256") != spec.sha256
        or shards[0].get("verified") is not True
        or not isinstance(verification, dict)
        or verification.get("actual_sha256") != spec.sha256
    ):
        raise PreparationError("existing immutable artifact receipt does not match the pinned model")
    current = _identity(_regular_file(artifact, spec.size))
    if verification.get("file_identity_after_verification") != current:
        raise PreparationError("artifact identity changed after its immutable full-SHA receipt")


def publish_receipt(
    spec: ModelSpec,
    artifact: Path,
    identity: Mapping[str, int],
    verification_method: str,
    receipt_path: Path | None = None,
) -> dict[str, Any]:
    path = spec.receipt_path if receipt_path is None else receipt_path
    document = receipt_document(spec, artifact, identity, verification_method)
    if os.path.lexists(path):
        existing = _read_json(path, "immutable artifact receipt")
        validate_receipt(existing, spec, artifact)
        return existing
    _write_json_exclusive(path, document)
    return document


def download_model(
    spec: ModelSpec,
    model_dir: Path,
    *,
    workers: int,
    chunk_mib: int,
    retries: int,
    receipt_path: Path | None = None,
) -> dict[str, Any]:
    if workers < 1 or workers > MAX_WORKERS:
        raise PreparationError(f"workers must be between 1 and {MAX_WORKERS}")
    artifact = _artifact_path(spec, model_dir)
    range_fetch.download_file(
        spec.url,
        artifact,
        spec.size,
        spec.sha256,
        workers=workers,
        chunk_size=chunk_mib * 1024 * 1024,
        retries=retries,
    )
    identity = _identity(_regular_file(artifact, spec.size))
    return publish_receipt(
        spec,
        artifact,
        identity,
        "range_fetch end-to-end full-file SHA-256",
        receipt_path,
    )


def config_document(
    spec: ModelSpec,
    artifact: Path,
    receipt_path: Path,
    config_path: Path,
    *,
    server: Path = SERVER,
    lib_dirs: Sequence[Path] = LIB_DIRS,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "server": str(server),
        "model": str(artifact),
        "model_name": spec.alias,
        "context": CONTEXT,
        "host": HOST,
        "port": PORT,
        "artifact_manifest": os.path.relpath(receipt_path, config_path.parent),
        "lib_dirs": [str(path) for path in lib_dirs],
        "args": [
            "--gpu-layers",
            "auto",
            "--parallel",
            "1",
            "--jinja",
            "--no-webui",
            "--no-slots",
        ],
        "placement": {
            "policy": "auto_best_fit",
            "all_gpu_fit_claimed": False,
            "measurement_required": ["actual_gpu_layer_count", "peak_vram_bytes"],
        },
        "runtime_validation": {
            "llama_cpp_revision": LLAMA_REVISION,
            "load_validated": False,
            "inference_validated": False,
        },
        "shards": [
            {"name": spec.filename, "size": spec.size, "sha256": spec.sha256}
        ],
    }


def _publish_config(path: Path, document: Mapping[str, Any]) -> None:
    if os.path.lexists(path):
        existing = _read_json(path, "specialist runtime config")
        if existing != document:
            raise PreparationError(f"refusing to overwrite different existing config: {path}")
        return
    _write_json_exclusive(path, document)


def prepare_config(
    spec: ModelSpec,
    model_dir: Path,
    *,
    receipt_path: Path | None = None,
    config_path: Path | None = None,
    server: Path = SERVER,
    lib_dirs: Sequence[Path] = LIB_DIRS,
) -> dict[str, Any]:
    artifact = _artifact_path(spec, model_dir)
    receipt = spec.receipt_path if receipt_path is None else receipt_path
    config = spec.config_path if config_path is None else config_path
    if os.path.lexists(receipt):
        validate_receipt(_read_json(receipt, "immutable artifact receipt"), spec, artifact)
    else:
        identity = verify_file(spec, artifact)
        publish_receipt(spec, artifact, identity, "fresh local full-file SHA-256", receipt)
    if not server.is_file() or not os.access(server, os.X_OK):
        raise PreparationError(f"llama-server is missing or not executable: {server}")
    if not lib_dirs or any(not path.is_dir() for path in lib_dirs):
        raise PreparationError("one or more portable llama.cpp library directories are missing")
    document = config_document(spec, artifact, receipt, config, server=server, lib_dirs=lib_dirs)
    _publish_config(config, document)
    return document


def _available_spec(key: str | None) -> ModelSpec:
    if key == IQ2_FALLBACK["key"]:
        raise PreparationError(
            "KAT IQ2_M is unavailable: exact pinned filename, LFS size, and SHA-256 are required"
        )
    if key is None or key == "all" or key not in SPECS:
        raise PreparationError("a single pinned primary --model is required")
    return SPECS[key]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--inspect", action="store_true", help="print pinned metadata; write nothing")
    action.add_argument("--download", action="store_true", help="resume, fully verify, and receipt one artifact")
    action.add_argument(
        "--prepare-config",
        action="store_true",
        help="verify/receipt an existing artifact and publish its inactive 8K config",
    )
    parser.add_argument("--model", choices=("all", *SPECS, IQ2_FALLBACK["key"]), default="all")
    parser.add_argument("--model-dir", type=Path, help="dedicated directory under the RAID model root")
    parser.add_argument("--workers", type=range_fetch.positive_int, default=4)
    parser.add_argument("--chunk-mib", type=range_fetch.positive_int, default=32)
    parser.add_argument("--retries", type=range_fetch.positive_int, default=5)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.inspect:
            print(json.dumps(inspect_document(args.model), indent=2, sort_keys=True))
            return 0
        spec = _available_spec(args.model)
        model_dir = resolve_model_dir(spec, args.model_dir)
        if args.download:
            download_model(
                spec,
                model_dir,
                workers=args.workers,
                chunk_mib=args.chunk_mib,
                retries=args.retries,
            )
        else:
            prepare_config(spec, model_dir)
        return 0
    except KeyboardInterrupt:
        print("interrupted; resumable download state was preserved", file=sys.stderr)
        return 130
    except (PreparationError, range_fetch.DownloadError, OSError, ValueError) as exc:
        print(f"specialist preparation failed: {type(exc).__name__}: {safe_error_detail(exc)}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
