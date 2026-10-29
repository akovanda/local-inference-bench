#!/usr/bin/env python3
"""Requantize the verified GPT-OSS 20B EAGLE-3 draft from Q8_0 to Q4_0.

The conversion is deliberately separate from the verified Q8_0 artifact.  It
uses the pinned Ollama llama-quantize binary, writes atomically on the RAID,
and publishes configs only after the lossy draft and its compatibility receipt
pass every check.  It never starts a model server or changes target weights.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__:
    from . import prepare_gpt20b_eagle3 as q8
else:
    import prepare_gpt20b_eagle3 as q8


ROOT = Path(__file__).resolve().parents[1]
SOURCE_MODEL = q8.DERIVED_MODEL
SOURCE_RECEIPT = q8.DERIVED_REPORT
SOURCE_SIZE = 922_272_576
SOURCE_SHA256 = "da07ec9a99c87995bdd83b6e4bea7672ed131aa6002037c3e9fbe9e6877af5aa"
SOURCE_D2T_SHA256 = "d216cee6452e34ffe59622f243fd01df01ab8300e60823b4eab9baeaf1f48cb3"

DESTINATION = Path("/external/storage/moe-bench/models/gpt20b-eagle3-Q4_0-derived")
MODEL = DESTINATION / "eagle3-gpt-oss-20b-Q4_0-target-tokenizer.gguf"
REPORT = ROOT / "results" / "gpt20b-eagle3-q4-artifact.json"
GPU_MAX3_CONFIG = ROOT / "runtime" / "gpt20b-eagle3-q4-gpu-max3-32k.json"
GPU_MAX1_CONFIG = ROOT / "runtime" / "gpt20b-eagle3-q4-gpu-max1-32k.json"

QUANTIZER = q8.SERVER.parent / "llama-quantize"
QUANTIZER_IMPL = q8.SERVER.parent / "libllama-quantize-impl.so"
QUANTIZER_SOURCE = q8.LLAMA_SOURCE / "tools" / "quantize" / "quantize.cpp"
QUANTIZER_SOURCE_REQUIREMENTS = (
    "--allow-requantize",
    "--pure",
    "--output-tensor-type",
    "--token-embedding-type",
    "model-f32.gguf [model-quant.gguf] type [nthreads]",
)

QUANTIZED_MATRIX_NAMES = tuple(
    name for name, (_shape, type_name) in q8.EXPECTED_TENSORS.items() if type_name == "Q8_0"
)
PRESERVED_F32_NAMES = tuple(
    name for name, (_shape, type_name) in q8.EXPECTED_TENSORS.items() if type_name == "F32"
)


class PreparationError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PreparationError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PreparationError(f"{description} is unreadable or invalid: {path}") from exc
    _require(isinstance(value, dict), f"{description} must be a JSON object")
    return value


def validate_source_receipt() -> dict[str, Any]:
    receipt = _read_json(SOURCE_RECEIPT, "Q8 EAGLE receipt")
    verdict = receipt.get("verdict")
    compatibility = receipt.get("compatibility")
    artifact = receipt.get("gguf_artifact")
    target = receipt.get("target")
    _require(isinstance(verdict, dict) and verdict.get("artifact_complete") is True, "Q8 source is incomplete")
    _require(verdict.get("full_sha256_verified") is True, "Q8 source lacks full-SHA proof")
    _require(
        isinstance(compatibility, dict) and compatibility.get("activation_allowed") is True,
        "Q8 source compatibility is not activation-ready",
    )
    _require(
        isinstance(artifact, dict)
        and artifact.get("path") == str(SOURCE_MODEL)
        and artifact.get("size") == SOURCE_SIZE
        and artifact.get("sha256") == SOURCE_SHA256,
        "Q8 source identity changed",
    )
    _require(
        isinstance(target, dict)
        and target.get("path") == str(q8.TARGET_MODEL)
        and target.get("size") == q8.TARGET_SIZE
        and target.get("sha256") == q8.TARGET_SHA256,
        "Q8 receipt target binding changed",
    )
    return {
        "receipt": str(SOURCE_RECEIPT),
        "path": str(SOURCE_MODEL),
        "size": SOURCE_SIZE,
        "sha256": SOURCE_SHA256,
        "target": dict(target),
    }


def validate_quantizer() -> dict[str, Any]:
    try:
        binary_stat = QUANTIZER.stat()
        impl_stat = QUANTIZER_IMPL.stat()
        source_bytes = QUANTIZER_SOURCE.read_bytes()
        source_text = source_bytes.decode("utf-8")
        binary_bytes = QUANTIZER.read_bytes()
        impl_bytes = QUANTIZER_IMPL.read_bytes()
    except (OSError, UnicodeError) as exc:
        raise PreparationError("pinned quantizer bundle or source is unreadable") from exc
    _require(stat.S_ISREG(binary_stat.st_mode) and os.access(QUANTIZER, os.X_OK), "llama-quantize is not executable")
    _require(stat.S_ISREG(impl_stat.st_mode), "quantizer implementation is not a regular file")
    for fragment in QUANTIZER_SOURCE_REQUIREMENTS:
        _require(fragment in source_text, f"pinned quantizer source lacks {fragment!r}")
    for marker in (b"allow-requantize", b"output-tensor-type", b"token-embedding-type"):
        _require(marker in binary_bytes or marker in impl_bytes, f"installed quantizer lacks marker {marker!r}")
    return {
        "revision": q8.LLAMA_REVISION,
        "binary": str(QUANTIZER),
        "binary_size": binary_stat.st_size,
        "binary_sha256": hashlib.sha256(binary_bytes).hexdigest(),
        "implementation": str(QUANTIZER_IMPL),
        "implementation_size": impl_stat.st_size,
        "implementation_sha256": hashlib.sha256(impl_bytes).hexdigest(),
        "source": str(QUANTIZER_SOURCE),
        "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "required_cli_present": True,
    }


def quantizer_argv(source: Path, output: Path, threads: int) -> list[str]:
    _require(threads == 4, "this reproducible candidate requires exactly 4 quantizer threads")
    return [
        str(QUANTIZER),
        "--allow-requantize",
        "--pure",
        "--output-tensor-type",
        "q4_0",
        "--token-embedding-type",
        "q4_0",
        str(source),
        str(output),
        "Q4_0",
        str(threads),
    ]


def _tensor_payload_sha256(path: Path, reader: Any, tensor: Any) -> str:
    byte_count = q8._tensor_byte_count(tensor, tensor.type_name)
    _require(byte_count is not None and byte_count > 0, f"unsupported tensor payload: {tensor.name}")
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            handle.seek(reader.data_start + tensor.offset)
            remaining = byte_count
            while remaining:
                block = handle.read(min(8 * 1024 * 1024, remaining))
                _require(bool(block), f"truncated tensor payload: {tensor.name}")
                digest.update(block)
                remaining -= len(block)
    except OSError as exc:
        raise PreparationError(f"tensor payload is unreadable: {tensor.name}") from exc
    return digest.hexdigest()


def inspect_q4_model(path: Path, *, full_sha256: str) -> dict[str, Any]:
    reader_type = q8._load_reader()
    model = reader_type(path)
    target = reader_type(q8.TARGET_MODEL)
    details = path.stat()
    tensors = {tensor.name: tensor for tensor in model.tensors}
    _require(len(tensors) == len(model.tensors) == len(q8.EXPECTED_TENSORS), "Q4 tensor set changed")

    ranges: list[tuple[int, int, str]] = []
    type_counts: Counter[str] = Counter()
    for name, (shape, source_type) in q8.EXPECTED_TENSORS.items():
        tensor = tensors.get(name)
        _require(tensor is not None, f"Q4 draft is missing tensor: {name}")
        expected_type = "Q4_0" if source_type == "Q8_0" else source_type
        _require(tensor.shape == shape, f"Q4 tensor {name} shape changed")
        _require(tensor.type_name == expected_type, f"Q4 tensor {name} is {tensor.type_name}, expected {expected_type}")
        byte_count = q8._tensor_byte_count(tensor, expected_type)
        _require(byte_count is not None and byte_count > 0, f"unsupported Q4 tensor layout: {name}")
        _require(tensor.offset >= 0 and tensor.offset % model.alignment == 0, f"unaligned Q4 tensor: {name}")
        start = model.data_start + tensor.offset
        end = start + byte_count
        _require(model.data_start <= start < end <= details.st_size, f"Q4 tensor exceeds bounds: {name}")
        ranges.append((start, end, name))
        type_counts[expected_type] += 1
    ranges.sort()
    for previous, current in zip(ranges, ranges[1:]):
        _require(previous[1] <= current[0], f"Q4 tensor overlap: {previous[2]}, {current[2]}")
    _require(type_counts == Counter({"Q4_0": 10, "F32": 5, "I64": 1}), "Q4 tensor type counts changed")

    source = reader_type(SOURCE_MODEL)
    source_d2t = next(tensor for tensor in source.tensors if tensor.name == "d2t")
    output_d2t = tensors["d2t"]
    source_d2t_sha = _tensor_payload_sha256(SOURCE_MODEL, source, source_d2t)
    output_d2t_sha = _tensor_payload_sha256(path, model, output_d2t)
    _require(source_d2t_sha == SOURCE_D2T_SHA256, "Q8 source d2t digest changed")
    _require(output_d2t_sha == source_d2t_sha, "Q4 conversion changed d2t mapping")

    output_header = q8.scan_raw_gguf_header(path)
    target_header = q8.scan_raw_gguf_header(q8.TARGET_MODEL)
    output_tokenizer = tuple(row for row in output_header.metadata if row.key.startswith("tokenizer.ggml."))
    target_tokenizer = tuple(row for row in target_header.metadata if row.key.startswith("tokenizer.ggml."))
    output_tokenizer_sha = q8._metadata_records_digest(output_tokenizer)
    target_tokenizer_sha = q8._metadata_records_digest(target_tokenizer)
    _require(output_tokenizer_sha == target_tokenizer_sha, "Q4 tokenizer metadata differs from target")
    _require(q8._tokenizer_digest(model.metadata) == q8._tokenizer_digest(target.metadata), "Q4 tokenizer values differ")

    expected_metadata = q8.inspect_draft(
        SOURCE_MODEL,
        q8.TARGET_MODEL,
        reader_type,
        full_sha_verified=True,
        expected_size=SOURCE_SIZE,
        expected_sha256=SOURCE_SHA256,
    )["gguf"]["metadata"]
    for key, expected in expected_metadata.items():
        _require(model.metadata.get(key) == expected, f"Q4 architecture metadata changed: {key}")

    return {
        "path": str(path),
        "size": details.st_size,
        "sha256": full_sha256,
        "fresh_full_sha256_verified": True,
        "gguf": {
            "version": model.version,
            "tensor_count": len(model.tensors),
            "tensor_type_counts": dict(sorted(type_counts.items())),
            "matrix_quantization": "Q4_0",
            "all_tensor_payload_bounds_validated": True,
            "all_offsets_aligned": True,
            "no_payload_overlaps": True,
            "architecture_metadata_preserved": True,
        },
        "compatibility": {
            "target_tokenizer_raw_metadata_sha256": target_tokenizer_sha,
            "draft_tokenizer_raw_metadata_sha256": output_tokenizer_sha,
            "target_tokenizer_exact_match": True,
            "d2t_source_sha256": source_d2t_sha,
            "d2t_output_sha256": output_d2t_sha,
            "d2t_unchanged": True,
            "target_layers": q8.TARGET_LAYERS,
            "target_hidden_size": q8.TARGET_HIDDEN_SIZE,
        },
    }


def config_document(max_draft_tokens: int) -> dict[str, Any]:
    _require(max_draft_tokens in (1, 3), "Q4 profile supports max draft tokens 1 or 3")
    config_path = GPU_MAX3_CONFIG if max_draft_tokens == 3 else GPU_MAX1_CONFIG
    value = q8.config_document("gpu", REPORT, MODEL, config_path)
    args = value["args"]
    index = args.index("--spec-draft-n-max") + 1
    args[index] = str(max_draft_tokens)
    value["draft_profile"] = f"gpu_q4_0_max{max_draft_tokens}"
    value["draft_quantization"] = "Q4_0"
    value["draft_max_tokens"] = max_draft_tokens
    return value


def publish_configs() -> None:
    q8.atomic_write_json(GPU_MAX3_CONFIG, config_document(3))
    q8.atomic_write_json(GPU_MAX1_CONFIG, config_document(1))


def artifact_report(
    *,
    source: Mapping[str, Any],
    target: Mapping[str, Any],
    quantizer: Mapping[str, Any],
    command: Sequence[str],
    candidate: Mapping[str, Any] | None = None,
    error: BaseException | None = None,
) -> dict[str, Any]:
    complete = bool(
        candidate
        and candidate.get("fresh_full_sha256_verified") is True
        and isinstance(candidate.get("compatibility"), Mapping)
        and candidate["compatibility"].get("target_tokenizer_exact_match") is True
        and candidate["compatibility"].get("d2t_unchanged") is True
    )
    value: dict[str, Any] = {
        "schema_version": 1,
        "checked_at": q8.utc_now(),
        "state": "inactive_q4_verified" if complete else "inactive_q4_blocked",
        "source_artifact": dict(source),
        "gguf_artifact": {
            "path": str(candidate.get("path", MODEL)) if candidate else str(MODEL),
            "size": candidate.get("size") if candidate else None,
            "sha256": candidate.get("sha256") if candidate else None,
            "quantization": "Q4_0",
            "lossy_requantization": True,
        },
        "target": {
            "path": str(target.get("model", q8.TARGET_MODEL)),
            "size": target.get("size", q8.TARGET_SIZE),
            "sha256": target.get("sha256", q8.TARGET_SHA256),
            "fresh_full_sha256_verified": target.get("fresh_full_sha256_verified") is True,
        },
        "quantizer": dict(quantizer),
        "conversion": {
            "argv": list(command),
            "threads": 4,
            "allow_requantize": True,
            "pure": True,
            "output_tensor_type": "Q4_0",
            "token_embedding_type": "Q4_0",
            "lossy": True,
            "target_weights_changed": False,
        },
        "candidate_validation": dict(candidate or {}),
        "compatibility": {
            "activation_allowed": complete,
            "target_identity_bound": True,
            "target_tokenizer_metadata_exact": bool(
                candidate and candidate.get("compatibility", {}).get("target_tokenizer_exact_match") is True
            ),
            "d2t_unchanged": bool(candidate and candidate.get("compatibility", {}).get("d2t_unchanged") is True),
            "lossy_draft_requantization": True,
            "target_weights_unchanged": True,
            "runtime_load_validated": False,
            "inference_validated": False,
        },
        "profiles": {
            "gpu_max3": {"config": str(GPU_MAX3_CONFIG), "max_draft_tokens": 3, "runtime_validated": False},
            "gpu_max1": {"config": str(GPU_MAX1_CONFIG), "max_draft_tokens": 1, "runtime_validated": False},
        },
        "verdict": {
            "artifact_complete": complete,
            "full_sha256_verified": bool(candidate and candidate.get("fresh_full_sha256_verified") is True),
            "header_and_tensor_directory_valid": bool(candidate),
            "source_compatible": True,
            "profile_publication_staged": complete,
            "runtime_load_validated": False,
            "inference_validated": False,
        },
    }
    if error is not None:
        value["error"] = type(error).__name__
        value["error_detail"] = q8.safe_error_detail(error)
    return value


def quantize(*, threads: int = 4) -> dict[str, Any]:
    source = validate_source_receipt()
    target = q8.validate_target_contract()
    quantizer = validate_quantizer()
    _require(SOURCE_MODEL.stat().st_size == SOURCE_SIZE, "Q8 source size changed")
    _require(_sha256(SOURCE_MODEL) == SOURCE_SHA256, "Q8 source full SHA-256 changed")
    _require(not MODEL.exists(), f"refusing to overwrite existing Q4 candidate: {MODEL}")

    DESTINATION.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{MODEL.name}.", suffix=".gguf", dir=DESTINATION)
    os.close(descriptor)
    temporary = Path(temporary_name)
    command = quantizer_argv(SOURCE_MODEL, temporary, threads)
    candidate: dict[str, Any] | None = None
    try:
        env = os.environ.copy()
        library_dir = str(q8.SERVER.parent)
        current = env.get("LD_LIBRARY_PATH")
        env["LD_LIBRARY_PATH"] = library_dir if not current else f"{library_dir}:{current}"
        completed = subprocess.run(command, env=env, check=False)
        _require(completed.returncode == 0, f"llama-quantize exited {completed.returncode}")
        output_sha256 = _sha256(temporary)
        candidate = inspect_q4_model(temporary, full_sha256=output_sha256)
        candidate["path"] = str(MODEL)
        os.replace(temporary, MODEL)
        report = artifact_report(
            source=source,
            target=target,
            quantizer=quantizer,
            command=quantizer_argv(SOURCE_MODEL, MODEL, threads),
            candidate=candidate,
        )
        _require(report["compatibility"]["activation_allowed"] is True, "Q4 activation gate failed")
        q8.atomic_write_json(REPORT, report)
        publish_configs()
        return report
    except BaseException as exc:
        temporary.unlink(missing_ok=True)
        report = artifact_report(
            source=source,
            target=target,
            quantizer=quantizer,
            command=quantizer_argv(SOURCE_MODEL, MODEL, threads),
            candidate=candidate,
            error=exc,
        )
        q8.atomic_write_json(REPORT, report)
        raise


def plan() -> dict[str, Any]:
    return {
        "source": str(SOURCE_MODEL),
        "source_size": SOURCE_SIZE,
        "source_sha256": SOURCE_SHA256,
        "output": str(MODEL),
        "receipt": str(REPORT),
        "configs": [str(GPU_MAX3_CONFIG), str(GPU_MAX1_CONFIG)],
        "command": quantizer_argv(SOURCE_MODEL, MODEL, 4),
        "conversion_started": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quantize", action="store_true", help="run the bounded Q8_0 to Q4_0 conversion")
    parser.add_argument("--threads", type=int, default=4)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        _require(args.threads == 4, "--threads must be exactly 4")
        report = quantize(threads=args.threads) if args.quantize else plan()
    except (PreparationError, q8.PreparationError, OSError, ValueError, KeyError) as exc:
        print(f"prepare_gpt20b_eagle3_q4: {type(exc).__name__}: {q8.safe_error_detail(exc)}", file=sys.stderr)
        return 1
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
