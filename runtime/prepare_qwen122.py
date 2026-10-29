#!/usr/bin/env python3
"""Stage and verify the pinned Qwen3.5-122B-A10B UD-Q4_K_XL split GGUF."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import quote

if __package__:
    from . import range_fetch
else:
    import range_fetch


ROOT = Path(__file__).resolve().parents[1]
DESTINATION = Path("/external/storage/moe-bench/models/qwen122-UD-Q4_K_XL")
REPORT = ROOT / "results" / "qwen122-artifact.json"
REPOSITORY = "unsloth/Qwen3.5-122B-A10B-GGUF"
REVISION = "51eab4d59d53f573fb9206cb3ce613f1d0aa392b"
QUANT = "UD-Q4_K_XL"
BASE_URL = f"https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/{QUANT}"
LLAMA_SERVER = Path("/external/storage/moe-bench/ollama-0.40.6/lib/ollama/llama-server")
SHARDS = (
    (
        "Qwen3.5-122B-A10B-UD-Q4_K_XL-00001-of-00003.gguf",
        10_943_552,
        "467c9bd92ea518539cf75bf5a5fbfbd35e9a0b40d766ccaa67bf120e12041df3",
    ),
    (
        "Qwen3.5-122B-A10B-UD-Q4_K_XL-00002-of-00003.gguf",
        49_640_779_424,
        "ecdbd42d43b0df9fa0ef9a584e09e95a43966ef03a122aba0b87a99d44d9ad98",
    ),
    (
        "Qwen3.5-122B-A10B-UD-Q4_K_XL-00003-of-00003.gguf",
        27_378_273_056,
        "13300e0f059e6fa21aa0fabde2a554f9deea366c0e54f268045769b214b28c97",
    ),
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}.{uuid.uuid4().hex}")
    descriptor = -1
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        view = memoryview(payload)
        written = 0
        while written < len(view):
            count = os.write(descriptor, view[written:])
            if count <= 0:
                raise OSError("short provenance write")
            written += count
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def safe_error_detail(exc: BaseException) -> str:
    """Return bounded diagnostic text without exposing a request URL."""
    detail = " ".join(str(exc).split())
    detail = re.sub(r"https?://\S+", "[redacted-url]", detail, flags=re.IGNORECASE)
    return (detail or type(exc).__name__)[:500]


def artifact_report(
    destination: Path,
    verified: set[str],
    error: str | None = None,
    error_detail: str | None = None,
    resume_provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    shard_rows: list[dict[str, Any]] = []
    for name, size, sha256 in SHARDS:
        path = destination / name
        shard_rows.append(
            {
                "name": name,
                "path": str(path),
                "size": size,
                "sha256": sha256,
                "present": path.is_file() and path.stat().st_size == size,
                "verified": name in verified,
            }
        )
    complete = len(verified) == len(SHARDS)
    report: dict[str, Any] = {
        "schema_version": 2,
        "checked_at": utc_now(),
        "verdict": {
            "artifact_complete": complete,
            "llama_cpp_runnable": complete,
            "current_strata_runnable": False,
            "strata_blocking_error": "missing qwen35moe.nextn_predict_layers",
        },
        "model": {
            "repo": REPOSITORY,
            "revision": REVISION,
            "quant": QUANT,
            "directory": str(destination),
            "expected_total_size": sum(row[1] for row in SHARDS),
        },
        "shards": shard_rows,
        "parsed_metadata": {
            "general.architecture": "qwen35moe",
            "general.name": "Qwen3.5-122B-A10B",
            "split.count": 3,
            "split.tensors.count": 879,
            "qwen35moe.block_count": 48,
            "qwen35moe.context_length": 262144,
            "qwen35moe.embedding_length": 3072,
            "qwen35moe.attention.head_count": 32,
            "qwen35moe.attention.head_count_kv": 2,
            "qwen35moe.attention.key_length": 256,
            "qwen35moe.attention.value_length": 256,
            "qwen35moe.expert_count": 256,
            "qwen35moe.expert_used_count": 8,
            "qwen35moe.expert_feed_forward_length": 1024,
            "qwen35moe.expert_shared_feed_forward_length": 1024,
            "qwen35moe.ssm.conv_kernel": 4,
            "qwen35moe.ssm.state_size": 128,
            "qwen35moe.ssm.group_count": 16,
            "qwen35moe.ssm.time_step_rank": 64,
            "qwen35moe.ssm.inner_size": 8192,
            "qwen35moe.full_attention_interval": 4,
            "qwen35moe.nextn_predict_layers": None,
        },
        "llama_cpp": {
            "server": str(LLAMA_SERVER),
            "version": "0.7.0-dev",
            "commit": "620d2225c",
            "architecture_mapping": "48 qwen35moe layers map to MODEL_122B_A10B",
            "source": "https://github.com/ggml-org/llama.cpp/blob/620d2225c/src/models/qwen35moe.cpp",
            "auto_split_model_path": str(destination / SHARDS[0][0]),
            "nextn_metadata_optional": True,
            "reasoning_flag_source": "https://github.com/ggml-org/llama.cpp/blob/620d2225c/common/arg.cpp#L3602-L3630",
            "reasoning_template_source": "https://github.com/ggml-org/llama.cpp/blob/620d2225c/tools/server/server-context.cpp#L1467-L1505",
            "openai_reasoning_source": "https://github.com/ggml-org/llama.cpp/blob/620d2225c/tools/server/server-common.cpp#L1301-L1337",
            "reasoning_semantics": "--reasoning off/on sets enable_thinking false/true before Jinja rendering; an OpenAI reasoning_effort of none also forces enable_thinking false before rendering.",
        },
        "prompt_controls": {
            "template_variable": "enable_thinking",
            "none": {
                "llama_server_reasoning": "off",
                "prompt_sha256": "d954399e5692cfeb6184a50eac9f455bb99b996586eb7b5c16ae3014d9585d2b",
                "token_ids_sha256": "c1f45ea6d347b5b3ee11aa9f23f91a020d4505e0c8f5404e5ccce59d6ee30bde",
                "token_count": 16,
            },
            "medium": {
                "llama_server_reasoning": "on",
                "prompt_sha256": "c5cb3dd9432089ccf220f06700ecc5fd92193024ae670fa5b474671ae86a3e93",
                "token_ids_sha256": "146e42397bc8a82a5dd51b294d5ec3c6b8d2a571b9b5bc6e1d1db528eb66ccd5",
                "token_count": 14,
            },
            "verified_distinct": True,
            "probe_message": "Return exactly OK.",
        },
        "strata": {
            "revision": "477ac13cbe8e582c20484871ce459d20d82df09d",
            "guard_source": "vendor/strata/include/strata/artifact/gguf_reader.hpp:599",
            "reason": "The current architecture guard requires an MTP count key that this MTP-free GGUF omits.",
        },
    }
    if error is not None:
        report["error"] = error
    if error_detail is not None:
        report["error_detail"] = error_detail
    if resume_provenance is not None:
        report["resume_provenance"] = resume_provenance
    return report


def verified_prefix(
    destination: Path,
    report_path: Path,
    start_shard: int,
) -> tuple[set[str], dict[str, Any] | None]:
    """Trust an unchanged, previously hashed prefix recorded by this helper."""
    if start_shard == 1:
        return set(), None
    try:
        receipt_bytes = report_path.read_bytes()
        receipt_stat = report_path.stat()
        receipt = json.loads(receipt_bytes.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise range_fetch.IntegrityError("prior artifact receipt is unreadable or invalid") from exc
    model = receipt.get("model") if isinstance(receipt, dict) else None
    rows = receipt.get("shards") if isinstance(receipt, dict) else None
    if (
        receipt.get("schema_version") != 2
        or not isinstance(model, dict)
        or model.get("repo") != REPOSITORY
        or model.get("revision") != REVISION
        or model.get("quant") != QUANT
        or Path(str(model.get("directory", ""))).resolve() != destination.resolve()
        or not isinstance(rows, list)
    ):
        raise range_fetch.IntegrityError("prior artifact receipt does not match the pinned artifact")
    by_name = {
        row.get("name"): row
        for row in rows
        if isinstance(row, dict) and isinstance(row.get("name"), str)
    }
    verified: set[str] = set()
    file_stats: list[dict[str, Any]] = []
    for name, size, sha256 in SHARDS[: start_shard - 1]:
        row = by_name.get(name)
        if (
            not isinstance(row, dict)
            or row.get("size") != size
            or row.get("sha256") != sha256
            or row.get("verified") is not True
        ):
            raise range_fetch.IntegrityError(f"prior artifact receipt does not verify {name}")
        path = destination / name
        details = path.lstat()
        if not stat.S_ISREG(details.st_mode) or details.st_size != size:
            raise range_fetch.IntegrityError(
                f"previously verified shard changed type or size: {name}"
            )
        if details.st_mtime_ns > receipt_stat.st_mtime_ns or details.st_ctime_ns > receipt_stat.st_mtime_ns:
            raise range_fetch.IntegrityError(
                f"previously verified shard changed after its receipt: {name}"
            )
        verified.add(name)
        file_stats.append(
            {
                "name": name,
                "device": details.st_dev,
                "inode": details.st_ino,
                "mode": details.st_mode,
                "links": details.st_nlink,
                "size": details.st_size,
                "mtime_ns": details.st_mtime_ns,
                "ctime_ns": details.st_ctime_ns,
                "sha256": sha256,
            }
        )
    return verified, {
        "mode": "unchanged_verified_prefix",
        "start_shard": start_shard,
        "prior_receipt": str(report_path.resolve()),
        "prior_receipt_sha256": hashlib.sha256(receipt_bytes).hexdigest(),
        "prior_receipt_mtime_ns": receipt_stat.st_mtime_ns,
        "files": file_stats,
    }


def stage(
    destination: Path,
    report_path: Path,
    *,
    workers: int,
    chunk_mib: int,
    retries: int,
    start_shard: int = 1,
) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    verified, resume_provenance = verified_prefix(destination, report_path, start_shard)
    for name, size, sha256 in SHARDS[start_shard - 1 :]:
        output = destination / name
        try:
            range_fetch.download_file(
                f"{BASE_URL}/{quote(name)}",
                output,
                size,
                sha256,
                workers=workers,
                chunk_size=chunk_mib * 1024 * 1024,
                retries=retries,
            )
        except BaseException as exc:
            atomic_write_json(
                report_path,
                artifact_report(
                    destination,
                    verified,
                    type(exc).__name__,
                    safe_error_detail(exc),
                    resume_provenance,
                ),
            )
            raise
        verified.add(name)
        atomic_write_json(
            report_path,
            artifact_report(destination, verified, resume_provenance=resume_provenance),
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=DESTINATION)
    parser.add_argument("--report", type=Path, default=REPORT)
    parser.add_argument("--workers", type=range_fetch.positive_int, default=8)
    parser.add_argument("--chunk-mib", type=range_fetch.positive_int, default=32)
    parser.add_argument("--retries", type=range_fetch.positive_int, default=4)
    parser.add_argument(
        "--start-shard",
        type=int,
        choices=range(1, len(SHARDS) + 1),
        default=1,
        help="resume at this 1-based shard using an unchanged verified-prefix receipt",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        stage(
            args.destination.expanduser().resolve(),
            args.report.expanduser().resolve(),
            workers=args.workers,
            chunk_mib=args.chunk_mib,
            retries=args.retries,
            start_shard=args.start_shard,
        )
    except KeyboardInterrupt:
        print("interrupted; verified partial downloads were preserved", file=sys.stderr)
        return 130
    except (range_fetch.DownloadError, OSError) as exc:
        print(
            f"Qwen122 staging failed: {type(exc).__name__}: {safe_error_detail(exc)}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
