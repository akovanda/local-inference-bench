#!/usr/bin/env python3
"""Stage and verify the pinned Qwen3.6-35B-A3B UD-Q3_K_XL GGUF.

This artifact is for the existing native llama.cpp CPU-MoE comparison.  Its
published expert tensor mix is deliberately recorded as incompatible with the
pinned Strata grouped-expert kernels, so this helper never attempts an IQ pack.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

if __package__:
    from . import range_fetch
else:
    import range_fetch


ROOT = Path(__file__).resolve().parents[1]
DESTINATION = Path("/external/storage/moe-bench/models/qwen36-UD-Q3_K_XL")
REPORT = ROOT / "results" / "qwen36-q3-artifact.json"
CONFIG = ROOT / "runtime" / "qwen36-q3-llama-32k.json"
PLACEMENT_CONFIG = ROOT / "runtime" / "qwen36-q3-llama-gpu-auto-32k.json"
REPOSITORY = "unsloth/Qwen3.6-35B-A3B-MTP-GGUF"
REVISION = "5bc3e238d916f48a861bac2f8a1990a0e9b7e98d"
QUANT = "UD-Q3_K_XL"
NAME = "Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf"
SIZE = 17_227_569_440
SHA256 = "3fba9ab57290b34726837521335bf9268397174f3dd662880e2d0c5264d17b81"
URL = f"https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/{NAME}"
HEADER = Path("/external/storage/moe-bench/models/qwen36-UD-Q3_K_XL.header-0-33554431.gguf")
HEADER_SIZE = 33_554_432
HEADER_SHA256 = "1ae7604322fd49419c96d63c232e6e3d15403ca5fd0ea957af54bd6a2e5e286b"
IQ3_HEADER = Path("/external/storage/moe-bench/models/qwen36-UD-IQ3_S/header-0-33554431.gguf")
IQ3_HEADER_SHA256 = "5f38739be15505d108eb4fa44438be24dbc9e1b160056f685e1a70066b942f29"


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
    detail = " ".join(str(exc).split())
    detail = re.sub(r"https?://\S+", "[redacted-url]", detail, flags=re.IGNORECASE)
    return (detail or type(exc).__name__)[:500]


def artifact_report(destination: Path, verified: bool, error: BaseException | None = None) -> dict[str, Any]:
    path = destination / NAME
    present = path.is_file() and path.stat().st_size == SIZE
    report: dict[str, Any] = {
        "schema_version": 2,
        "checked_at": utc_now(),
        "state": "ready" if verified else "pending_download_or_verification",
        "verdict": {
            "artifact_complete": verified,
            "llama_cpp_runnable": verified,
            "current_strata_runnable": False,
            "strata_blocking_error": "The published routed-expert mix contains Q3_K, which the pinned grouped gate/up and down format tables do not support; Q6_K also requires a separately compiled opt-in.",
        },
        "model": {
            "repo": REPOSITORY,
            "revision": REVISION,
            "quant": QUANT,
            "directory": str(destination),
            "expected_total_size": SIZE,
            "publisher_source_base": "Qwen/Qwen3.6-35B-A3B",
            "same_publisher_and_revision_as_q4_control": True,
        },
        "shards": [
            {
                "name": NAME,
                "path": str(path),
                "size": SIZE,
                "sha256": SHA256,
                "present": present,
                "verified": verified,
            }
        ],
        "bounded_header_validation": {
            "tool": "vendor/strata/tools/quantscope.py",
            "bytes_read_per_candidate": HEADER_SIZE,
            "layer_by_layer_parse_attempt": {
                "result": "rejected before start",
                "reason": "approval required by policy, but AskForApproval is set to Never",
                "decision_impact": "none; the successful aggregate header scan proves that unsupported Q3_K is present in routed experts",
            },
            "q3_k_xl": {
                "path": str(HEADER),
                "range_sha256": HEADER_SHA256,
                "tensor_count": 753,
                "routed_expert_bits_per_weight": 3.54,
                "routed_expert_storage_byte_share": {
                    "IQ3_XXS": "55%",
                    "IQ4_XS": "38%",
                    "Q6_K": "5%",
                    "Q3_K": "2%",
                    "Q4_K": "1%",
                },
            },
            "iq3_s_alternative": {
                "path": str(IQ3_HEADER),
                "range_sha256": IQ3_HEADER_SHA256,
                "tensor_count": 753,
                "routed_expert_bits_per_weight": 3.20,
                "routed_expert_storage_byte_share": {
                    "IQ2_S": "51%",
                    "IQ4_XS": "40%",
                    "Q6_K": "5%",
                    "IQ3_S": "2%",
                    "Q2_K": "1%",
                    "Q4_K": "1%",
                },
                "rejected_for_strata": "Q2_K is absent from the pinned grouped expert kernels and Q6_K needs a non-baseline build.",
            },
        },
        "strata_compatibility": {
            "revision": "477ac13cbe8e582c20484871ce459d20d82df09d",
            "compatible": False,
            "pack_attempted": False,
            "patch_attempted": False,
            "source_gu_formats": "vendor/strata/src/kernels/cuda/iq_kernels.cu:645",
            "source_down_formats": "vendor/strata/src/kernels/cuda/iq_kernels.cu:651",
            "source_runtime_guard": "vendor/strata/src/qwen35/q36_engine.cu:1074",
            "geometry_note": "The Qwen35 runtime reads model geometry dynamically, but every layer still must pass native_expert_supported for its actual gate/up and down types.",
        },
        "llama_cpp": {
            "server": "/external/storage/moe-bench/ollama-0.40.6/lib/ollama/llama-server",
            "version": "0.7.0-dev",
            "upstream_revision": "620d2225cc436ab0b001e3200e590e3b286ef9ab",
            "config": str(CONFIG),
            "placement_config": str(PLACEMENT_CONFIG),
            "placement_experiment": {
                "only_argument_difference": "--cpu-moe is omitted",
                "purpose": "measure actual automatic whole-layer GPU offload with the smaller quant",
                "fit_claimed": False,
                "measurement_required": ["offloaded_layer_count", "peak_vram_bytes"],
            },
            "comparison_control": str(ROOT / "runtime" / "qwen36-llama-32k.json"),
            "config_activated": False,
        },
        "download": {
            "tool": str(ROOT / "runtime" / "range_fetch.py"),
            "workers_max": 4,
            "chunk_mib": 32,
            "seed_prefix": str(HEADER),
            "source_url_redacted_from_report": True,
        },
    }
    if error is not None:
        report["error"] = type(error).__name__
        report["error_detail"] = safe_error_detail(error)
    return report


def verify_existing(path: Path) -> None:
    if not path.is_file() or path.stat().st_size != SIZE:
        raise range_fetch.IntegrityError("model is missing or has the wrong exact size")
    if range_fetch.hash_file(path) != SHA256:
        raise range_fetch.IntegrityError("model does not match the pinned SHA-256")


def stage(destination: Path, report: Path, workers: int, chunk_mib: int, retries: int, no_download: bool) -> None:
    if workers > 4:
        raise ValueError("this staging helper is capped at four download workers")
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / NAME
    try:
        if no_download:
            verify_existing(path)
        else:
            range_fetch.download_file(
                URL,
                path,
                SIZE,
                SHA256,
                workers=workers,
                chunk_size=chunk_mib * 1024 * 1024,
                seed_prefix=HEADER,
                retries=retries,
            )
        atomic_write_json(report, artifact_report(destination, True))
    except BaseException as exc:
        atomic_write_json(report, artifact_report(destination, False, exc))
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=DESTINATION)
    parser.add_argument("--report", type=Path, default=REPORT)
    parser.add_argument("--workers", type=range_fetch.positive_int, default=4)
    parser.add_argument("--chunk-mib", type=range_fetch.positive_int, default=32)
    parser.add_argument("--retries", type=range_fetch.positive_int, default=5)
    parser.add_argument("--no-download", action="store_true", help="verify an already published file and write its receipt")
    parser.add_argument("--metadata-only", action="store_true", help="write a pending receipt without reading model payload")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    destination = args.destination.expanduser().resolve()
    report = args.report.expanduser().resolve()
    if args.workers > 4:
        print("Qwen36 Q3 staging failed: workers must not exceed 4", file=sys.stderr)
        return 2
    if args.metadata_only:
        atomic_write_json(report, artifact_report(destination, False))
        return 0
    try:
        stage(destination, report, args.workers, args.chunk_mib, args.retries, args.no_download)
    except KeyboardInterrupt:
        print("interrupted; validated partial chunks and manifest were preserved", file=sys.stderr)
        return 130
    except (range_fetch.DownloadError, OSError, ValueError) as exc:
        print(f"Qwen36 Q3 staging failed: {type(exc).__name__}: {safe_error_detail(exc)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
