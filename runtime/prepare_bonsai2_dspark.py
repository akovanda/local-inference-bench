#!/usr/bin/env python3
"""Download, audit, and stage the experimental Bonsai2 DSpark v1 conversion.

The helper downloads one exact public BF16 source within a finite byte ceiling,
validates it, converts it with the pinned fork script, quantizes with a
separately built CPU-only ``llama-quantize``, and atomically publishes a new
artifact without overwriting an existing file.  It never starts llama-server or
model inference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = Path("/external/storage/moe-bench/models/bonsai2-dspark")
SOURCE_LOCAL = MODEL_ROOT / "Ternary-Bonsai-27B-dspark-bf16.gguf"
TARGET_MODEL = Path(
    "/external/storage/moe-bench/models/bonsai2-27b/Ternary-Bonsai-2-27B-PQ2_0.gguf"
)
DESTINATION = MODEL_ROOT / "Ternary-Bonsai-2-27B-dspark-v1-experimental-Q4_0.gguf"
REPORT = ROOT / "results" / "bonsai2-dspark-artifact-v2.json"
SOURCE_RECEIPT = ROOT / "results" / "bonsai2-dspark-source-v2.json"

LLAMA_SOURCE = Path("/external/storage/moe-bench/bonsai2-llama-source")
CONVERTER = LLAMA_SOURCE / "gguf-py/gguf/scripts/gguf_dspark_to_dflash.py"
CONVERTER_SHA256 = "1e2d90addf467abf17d1dc19744d07cd3fb6feb643ddbd7408f0f9f4bc56fe10"
LLAMA_SOURCE_REVISION = "e3de715222b2667c54a267441159a5729f11155e"
BORROWED_HADAMARD_FIX = "ede9f73442f263fe2168a8614098f00d3e90fbb3"

SOURCE_REPOSITORY = "prism-ml/Ternary-Bonsai-27B-gguf"
SOURCE_REVISION = "86e89f34c93201c3dfd5e5880fedb0022fc7e34d"
SOURCE_FILENAME = "Ternary-Bonsai-27B-dspark-bf16.gguf"
SOURCE_URL = (
    "https://huggingface.co/prism-ml/Ternary-Bonsai-27B-gguf/resolve/"
    f"{SOURCE_REVISION}/{SOURCE_FILENAME}"
)
SOURCE_SIZE = 7_291_885_792
SOURCE_SHA256 = "d5ce05b0e7e23804279fb0b451e330e71c06d977657802a0e22d71433f30dbad"
DOWNLOAD_LIMIT = 8_000_000_000

TARGET_SIZE = 7_206_168_928
TARGET_SHA256 = "3907dc1658db1f78a9826bf8d5bcb8dc65db0d466388937af57f2294fae62ec1"
TARGET_ARCH = "qwen35"
TARGET_BLOCK_COUNT = 64
TARGET_TENSOR_COUNT = 851
TARGET_EMBEDDING_LENGTH = 5120
TARGET_VOCAB_SIZE = 248_320
MASK_TOKEN_ID = 248_319
TARGET_LAYERS = [2, 17, 32, 47, 62]
BLOCK_SIZE = 4

SPECIAL_TOKENS = {
    "<|im_start|>": 248_045,
    "<|im_end|>": 248_046,
    "<tool_call>": 248_058,
    "</tool_call>": 248_059,
    "<tool_response>": 248_066,
    "</tool_response>": 248_067,
    "<think>": 248_068,
    "</think>": 248_069,
    "[PAD248319]": MASK_TOKEN_ID,
}

LEGACY_REQUIRED_TENSORS = {
    "dspark.confidence_head.bias",
    "dspark.confidence_head.weight",
    "dspark.fc.weight",
    "dspark.hidden_norm.weight",
    "dspark.log_snr_fc1.bias",
    "dspark.log_snr_fc1.weight",
    "dspark.log_snr_fc2.bias",
    "dspark.log_snr_fc2.weight",
    "dspark.markov_head_a.weight",
    "dspark.markov_head_b.weight",
    "token_embd.weight",
    "output.weight",
}

CONVERTED_REQUIRED_TENSORS = {
    "conf_proj.bias",
    "conf_proj.weight",
    "fc.weight",
    "enc.output_norm.weight",
    "log_snr_fc1.bias",
    "log_snr_fc1.weight",
    "log_snr_fc2.bias",
    "log_snr_fc2.weight",
    "markov_w1.weight",
    "markov_w2.weight",
}

# llama-quantize cannot quantize this one-row confidence projection.  It is
# only 5,376 BF16 values; every quantizable weight must still be Q4_0.
Q4_ALLOWED_BF16_TENSORS = {"conf_proj.weight": [5376]}
FINAL_Q4_TENSORS = {
    "fc.weight",
    "log_snr_fc1.weight",
    "log_snr_fc2.weight",
    "markov_w1.weight",
    "markov_w2.weight",
} | {
    f"blk.{block}.{suffix}"
    for block in range(6)
    for suffix in (
        "attn_k.weight",
        "attn_output.weight",
        "attn_q.weight",
        "attn_v.weight",
        "ffn_down.weight",
        "ffn_gate.weight",
        "ffn_up.weight",
    )
}
FINAL_F32_TENSORS = {
    "conf_proj.bias",
    "enc.output_norm.weight",
    "log_snr_fc1.bias",
    "log_snr_fc2.bias",
    "output_norm.weight",
} | {
    f"blk.{block}.{suffix}"
    for block in range(6)
    for suffix in (
        "attn_k_norm.weight",
        "attn_norm.weight",
        "attn_q_norm.weight",
        "ffn_norm.weight",
    )
}


class PreparationError(RuntimeError):
    """The candidate is unsafe or incomplete and must not be published."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PreparationError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def validate_pinned_file(path: Path, expected_size: int, expected_sha256: str) -> dict[str, Any]:
    try:
        details = path.stat()
    except OSError as exc:
        raise PreparationError(f"pinned file is inaccessible: {path}") from exc
    _require(stat.S_ISREG(details.st_mode), f"pinned path is not a regular file: {path}")
    _require(
        details.st_size == expected_size,
        f"pinned file is truncated or has the wrong size: {details.st_size} != {expected_size}",
    )
    digest = sha256_file(path)
    _require(digest == expected_sha256, f"pinned file SHA-256 mismatch: {digest}")
    return {
        "path": str(path),
        "size": details.st_size,
        "sha256": digest,
        "full_sha256_verified": True,
    }


def file_identity(path: Path) -> dict[str, Any]:
    """Return fields that bind a prior full-file verification to this inode."""
    try:
        details = path.stat()
    except OSError as exc:
        raise PreparationError(f"pinned file is inaccessible: {path}") from exc
    _require(stat.S_ISREG(details.st_mode), f"pinned path is not a regular file: {path}")
    return {
        "device": details.st_dev,
        "inode": details.st_ino,
        "size": details.st_size,
        "mtime_ns": details.st_mtime_ns,
        "ctime_ns": details.st_ctime_ns,
    }


def validate_verified_source_receipt(path: Path, receipt_path: Path) -> dict[str, Any]:
    """Reuse a full SHA verification only while the exact file identity is unchanged."""
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PreparationError(f"verified source receipt is inaccessible: {receipt_path}") from exc
    _require(receipt.get("full_sha256_verified") is True, "source receipt lacks full SHA verification")
    _require(receipt.get("sha256") == SOURCE_SHA256, "source receipt SHA-256 pin changed")
    _require(receipt.get("size") == SOURCE_SIZE, "source receipt size pin changed")
    _require(Path(receipt.get("path", "")).resolve() == path.resolve(), "source receipt path changed")
    _require(receipt.get("file_identity") == file_identity(path), "source changed after full verification")
    return receipt


def load_gguf(path: Path) -> Any:
    tools_dir = ROOT / "vendor" / "strata" / "tools"
    if str(tools_dir) not in sys.path:
        sys.path.insert(0, str(tools_dir))
    from gguf_reader import GGUFFile

    try:
        return GGUFFile(path)
    except (OSError, KeyError, UnicodeDecodeError, ValueError) as exc:
        raise PreparationError(f"GGUF header or tensor directory is invalid: {path}") from exc


def _tensor_names(reader: Any) -> set[str]:
    return {str(tensor.name) for tensor in reader.tensors}


def _tensor_inventory(reader: Any) -> tuple[dict[str, str], dict[str, list[int]], str]:
    types = {str(tensor.name): str(getattr(tensor, "type_name", "")) for tensor in reader.tensors}
    shapes = {
        str(tensor.name): list(getattr(tensor, "shape", []))
        for tensor in reader.tensors
    }
    canonical = sorted((name, types[name], shapes[name]) for name in types)
    digest = hashlib.sha256(
        json.dumps(canonical, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return types, shapes, digest


def _tokenizer_metadata(metadata: Mapping[str, Any], *, include_mask: bool) -> dict[str, Any]:
    return {
        key: value
        for key, value in metadata.items()
        if key.startswith("tokenizer.")
        and (include_mask or key != "tokenizer.ggml.mask_token_id")
    }


def _validate_special_tokens(metadata: Mapping[str, Any]) -> None:
    tokens = metadata.get("tokenizer.ggml.tokens")
    _require(isinstance(tokens, list), "tokenizer token table is missing")
    _require(len(tokens) == TARGET_VOCAB_SIZE, "tokenizer vocabulary size changed")
    for text, token_id in SPECIAL_TOKENS.items():
        _require(tokens[token_id] == text, f"tokenizer delimiter mismatch at id {token_id}: {text}")
    _require(metadata.get("tokenizer.ggml.model") == "gpt2", "target tokenizer model changed")
    _require(metadata.get("tokenizer.ggml.pre") == "qwen35", "target pre-tokenizer changed")
    _require(metadata.get("tokenizer.ggml.bos_token_id") == 248_044, "target BOS id changed")
    _require(metadata.get("tokenizer.ggml.eos_token_id") == 248_046, "target EOS id changed")
    _require(metadata.get("tokenizer.ggml.padding_token_id") == 248_044, "target padding id changed")


def validate_target_reader(reader: Any) -> dict[str, Any]:
    metadata = reader.metadata
    names = _tensor_names(reader)
    _require(metadata.get("general.architecture") == TARGET_ARCH, "wrong target architecture")
    _require(metadata.get("qwen35.block_count") == TARGET_BLOCK_COUNT, "wrong target block count")
    _require(
        metadata.get("qwen35.embedding_length") == TARGET_EMBEDDING_LENGTH,
        "wrong target embedding length",
    )
    _require(len(reader.tensors) == TARGET_TENSOR_COUNT, "wrong target tensor count")
    forbidden_keys = [key for key in metadata if "nextn" in key.lower() or "mtp" in key.lower()]
    forbidden_tensors = [name for name in names if "nextn" in name.lower() or "mtp" in name.lower()]
    _require(not forbidden_keys and not forbidden_tensors, "target unexpectedly bundles an MTP/nextn head")
    for name in names:
        if name.startswith("blk."):
            try:
                block = int(name.split(".", 2)[1])
            except (IndexError, ValueError) as exc:
                raise PreparationError(f"malformed target block tensor name: {name}") from exc
            _require(block < TARGET_BLOCK_COUNT, f"target tensor exceeds block 63: {name}")
    _validate_special_tokens(metadata)
    return {
        "architecture": TARGET_ARCH,
        "block_count": TARGET_BLOCK_COUNT,
        "embedding_length": TARGET_EMBEDDING_LENGTH,
        "tensor_count": TARGET_TENSOR_COUNT,
        "vocab_size": TARGET_VOCAB_SIZE,
        "mtp_or_nextn_present": False,
        "chat_delimiters_exact": True,
    }


def validate_legacy_reader(reader: Any) -> dict[str, Any]:
    metadata = reader.metadata
    names = _tensor_names(reader)
    _require(metadata.get("general.architecture") == "dspark", "legacy source is not arch=dspark")
    _require(metadata.get("general.file_type") == 32, "legacy source is not the pinned BF16 export")
    _require(metadata.get("dspark.block_count") == 6, "legacy DSpark layer count changed")
    _require(
        metadata.get("dspark.embedding_length") == TARGET_EMBEDDING_LENGTH,
        "legacy DSpark hidden size does not match the target",
    )
    _require(metadata.get("dspark.vocab_size") == TARGET_VOCAB_SIZE, "legacy DSpark vocab size changed")
    _require(metadata.get("dspark.dspark.block_size") == BLOCK_SIZE, "legacy block size is not 4")
    _require(
        metadata.get("dspark.dspark.target_layers") == [1, 16, 31, 46, 61],
        "legacy target layers changed",
    )
    _require(metadata.get("dspark.dspark.mask_token_id") == MASK_TOKEN_ID, "legacy mask token id changed")
    _require(len(reader.tensors) == 79, "legacy tensor count changed")
    _require(LEGACY_REQUIRED_TENSORS <= names, "legacy source lacks required DSpark/shared tensors")
    unknown = sorted(name for name in names if name.startswith("dspark.") and name not in LEGACY_REQUIRED_TENSORS)
    _require(not unknown, f"legacy source contains unmapped DSpark tensors: {unknown}")
    return {
        "architecture": "dspark",
        "block_count": 6,
        "block_size": BLOCK_SIZE,
        "embedding_length": TARGET_EMBEDDING_LENGTH,
        "target_layers_before_conversion": [1, 16, 31, 46, 61],
        "tensor_count": 79,
        "shared_embedding_and_output_present": True,
    }


def validate_converted_reader(
    reader: Any,
    target_reader: Any,
    *,
    require_q4: bool = True,
) -> dict[str, Any]:
    metadata = reader.metadata
    target_metadata = target_reader.metadata
    names = _tensor_names(reader)
    _require(metadata.get("general.architecture") == "dflash", "converted draft is not arch=dflash")
    _require(metadata.get("dflash.block_count") == 6, "converted draft layer count changed")
    _require(metadata.get("dflash.embedding_length") == TARGET_EMBEDDING_LENGTH, "draft hidden size mismatch")
    _require(metadata.get("dflash.vocab_size") == TARGET_VOCAB_SIZE, "draft vocab size mismatch")
    _require(metadata.get("dflash.block_size") == BLOCK_SIZE, "converted block size is not 4")
    _require(metadata.get("dflash.target_layers") == TARGET_LAYERS, "converted target layers are wrong")
    _require(metadata.get("tokenizer.ggml.mask_token_id") == MASK_TOKEN_ID, "converted mask id is wrong")
    _require(len(reader.tensors) == 77, "converted tensor count is not 77")
    _require(CONVERTED_REQUIRED_TENSORS <= names, "converted draft lacks required DSpark v1 tensors")
    _require(not any(name.startswith("dspark.") for name in names), "legacy tensor name survived conversion")
    _require(
        "token_embd.weight" not in names and "output.weight" not in names,
        "shared embedding/output tensors were not dropped",
    )
    _require(
        _tokenizer_metadata(metadata, include_mask=False)
        == _tokenizer_metadata(target_metadata, include_mask=False),
        "converted tokenizer metadata is not an exact target copy",
    )
    _validate_special_tokens(metadata)
    tensor_types, tensor_shapes, inventory_sha256 = _tensor_inventory(reader)
    type_counts = {
        tensor_type: sum(value == tensor_type for value in tensor_types.values())
        for tensor_type in sorted(set(tensor_types.values()))
    }
    if require_q4:
        _require(metadata.get("general.file_type") == 2, "final draft is not GGUF Q4_0")
        bf16_tensors = {
            name: tensor_shapes[name]
            for name, tensor_type in tensor_types.items()
            if tensor_type == "BF16"
        }
        _require(
            bf16_tensors == Q4_ALLOWED_BF16_TENSORS,
            f"unexpected BF16 tensors after Q4_0 quantization: {bf16_tensors}",
        )
        _require(
            {name for name, value in tensor_types.items() if value == "Q4_0"} == FINAL_Q4_TENSORS,
            "final Q4_0 tensor inventory changed",
        )
        _require(
            {name for name, value in tensor_types.items() if value == "F32"} == FINAL_F32_TENSORS,
            "final F32 tensor inventory changed",
        )
        _require(
            set(tensor_types.values()) == {"Q4_0", "F32", "BF16"},
            f"final draft contains an unexpected tensor type: {type_counts}",
        )
    else:
        _require(metadata.get("general.file_type") == 32, "converted intermediate is not GGUF BF16")
        _require(
            {name for name, value in tensor_types.items() if value == "BF16"}
            == FINAL_Q4_TENSORS | {"conf_proj.weight"},
            "converted BF16 tensor inventory changed",
        )
        _require(
            {name for name, value in tensor_types.items() if value == "F32"} == FINAL_F32_TENSORS,
            "converted F32 tensor inventory changed",
        )
        _require(
            tensor_shapes.get("conf_proj.weight") == [5376, 1],
            "converted confidence projection shape changed",
        )
    return {
        "architecture": "dflash",
        "block_count": 6,
        "block_size": BLOCK_SIZE,
        "embedding_length": TARGET_EMBEDDING_LENGTH,
        "target_layers": TARGET_LAYERS,
        "tensor_count": 77,
        "shared_embedding_and_output_dropped": True,
        "target_tokenizer_exact_copy": True,
        "q4_0": require_q4,
        "unquantized_bf16_tensors": Q4_ALLOWED_BF16_TENSORS if require_q4 else None,
        "tensor_type_counts": type_counts,
        "tensor_inventory_sha256": inventory_sha256,
    }


def conversion_commands(
    source: Path,
    donor: Path,
    intermediate: Path,
    output: Path,
    quantizer: Path,
) -> list[list[str]]:
    return [
        [
            sys.executable,
            str(CONVERTER),
            "--drop-shared-tensors",
            str(source),
            str(donor),
            str(intermediate),
        ],
        [str(quantizer), str(intermediate), str(output), "Q4_0", "4"],
    ]


def download_source(
    destination: Path = SOURCE_LOCAL,
    *,
    workers: int = 4,
    chunk_mib: int = 32,
    retries: int = 4,
) -> dict[str, Any]:
    _require(SOURCE_SIZE <= DOWNLOAD_LIMIT, "pinned source exceeds the finite download ceiling")
    _require(chunk_mib > 0, "chunk size must be positive")
    try:
        from runtime import range_fetch
    except ModuleNotFoundError:
        # Direct ``python runtime/prepare_bonsai2_dspark.py`` execution places
        # runtime/, rather than the repository root, on sys.path.
        import range_fetch

    range_fetch.download_file(
        SOURCE_URL,
        destination,
        SOURCE_SIZE,
        SOURCE_SHA256,
        workers=workers,
        chunk_size=chunk_mib * 1024 * 1024,
        retries=retries,
    )
    return validate_pinned_file(destination, SOURCE_SIZE, SOURCE_SHA256)


def validate_cpu_quantizer(path: Path) -> dict[str, Any]:
    try:
        details = path.stat()
    except OSError as exc:
        raise PreparationError(f"CPU quantizer is inaccessible: {path}") from exc
    _require(stat.S_ISREG(details.st_mode) and os.access(path, os.X_OK), "CPU quantizer is not executable")
    result = subprocess.run(
        ["ldd", str(path)],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    _require(result.returncode == 0, "could not inspect CPU quantizer linkage")
    lowered = result.stdout.lower()
    forbidden = ("cuda", "cublas", "cudart", "hip", "rocm")
    _require(not any(token in lowered for token in forbidden), "quantizer links a GPU runtime")
    return {"path": str(path), "sha256": sha256_file(path), "gpu_runtime_linked": False}


def publish_no_overwrite(temporary: Path, destination: Path) -> None:
    _require(temporary.is_file(), "temporary artifact is missing")
    _require(temporary.stat().st_dev == destination.parent.stat().st_dev, "publish is not same-filesystem")
    try:
        os.link(temporary, destination)
    except FileExistsError as exc:
        raise PreparationError(f"refusing to overwrite existing artifact: {destination}") from exc
    except OSError as exc:
        raise PreparationError(f"atomic artifact publish failed: {destination}") from exc
    temporary.unlink()


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, raw_temp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temp = Path(raw_temp)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def stage(
    source: Path,
    quantizer: Path,
    destination: Path = DESTINATION,
    source_receipt_path: Path | None = None,
) -> dict[str, Any]:
    _require(SOURCE_SIZE <= DOWNLOAD_LIMIT, "pinned source exceeds the finite download ceiling")
    _require(not destination.exists(), f"refusing to overwrite existing artifact: {destination}")
    _require(sha256_file(CONVERTER) == CONVERTER_SHA256, "pinned converter changed")
    source_receipt = (
        validate_verified_source_receipt(source, source_receipt_path)
        if source_receipt_path is not None
        else validate_pinned_file(source, SOURCE_SIZE, SOURCE_SHA256)
    )
    target_receipt = validate_pinned_file(TARGET_MODEL, TARGET_SIZE, TARGET_SHA256)
    legacy = validate_legacy_reader(load_gguf(source))
    target_reader = load_gguf(TARGET_MODEL)
    target = validate_target_reader(target_reader)
    quantizer_receipt = validate_cpu_quantizer(quantizer)

    destination.parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=".bonsai2-dspark-v1-", dir=destination.parent))
    intermediate = work / "converted-bf16.gguf"
    quantized = work / destination.name
    commands = conversion_commands(source, TARGET_MODEL, intermediate, quantized, quantizer)
    published = False
    try:
        subprocess.run(commands[0], check=True, shell=False)
        converted_bf16 = validate_converted_reader(load_gguf(intermediate), target_reader, require_q4=False)
        converted_bf16["path"] = str(intermediate)
        converted_bf16["size"] = intermediate.stat().st_size
        subprocess.run(commands[1], check=True, shell=False)
        final = validate_converted_reader(load_gguf(quantized), target_reader, require_q4=True)
        output = {
            "path": str(destination),
            "size": quantized.stat().st_size,
            "sha256": sha256_file(quantized),
            "full_sha256_verified": True,
        }
        publish_no_overwrite(quantized, destination)
        published = True
    except subprocess.CalledProcessError as exc:
        raise PreparationError(
            f"conversion command failed with exit {exc.returncode}; unpublished work preserved at {work}"
        ) from exc
    except PreparationError as exc:
        raise PreparationError(f"{exc}; unpublished work preserved at {work}") from exc
    finally:
        if published:
            shutil.rmtree(work, ignore_errors=True)

    return {
        "state": "staged_not_gpu_validated",
        "source": source_receipt,
        "target": target_receipt | target,
        "legacy_header": legacy,
        "converted_bf16": converted_bf16,
        "artifact": output | final,
        "quantizer": quantizer_receipt,
        "commands": commands,
        "gpu_validated": False,
        "inference_validated": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", help="download and fully verify the pinned BF16 source")
    parser.add_argument("--stage", action="store_true", help="convert the exact BF16 source after validation")
    parser.add_argument("--source", type=Path, default=SOURCE_LOCAL, help="exact pinned BF16 source")
    parser.add_argument(
        "--source-receipt",
        type=Path,
        default=SOURCE_RECEIPT,
        help="full-SHA receipt bound to the unchanged source inode",
    )
    parser.add_argument("--quantizer", type=Path, help="separately built CPU-only llama-quantize")
    parser.add_argument("--destination", type=Path, default=DESTINATION)
    parser.add_argument("--report", type=Path, default=REPORT)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--chunk-mib", type=int, default=32)
    parser.add_argument("--retries", type=int, default=4)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.download and not args.stage:
        print(
            json.dumps(
                {
                    "state": "ready_to_download",
                    "source_size": SOURCE_SIZE,
                    "download_limit": DOWNLOAD_LIMIT,
                    "download_supported": True,
                    "source": str(args.source),
                },
                indent=2,
            )
        )
        return 0
    try:
        download = None
        if args.download:
            download = download_source(
                args.source,
                workers=args.workers,
                chunk_mib=args.chunk_mib,
                retries=args.retries,
            )
        if not args.stage:
            print(json.dumps({"state": "source_verified", "source": download}, indent=2, sort_keys=True))
            return 0
        _require(args.quantizer is not None, "--stage requires --quantizer")
        receipt = stage(args.source, args.quantizer, args.destination, args.source_receipt)
        atomic_write_json(args.report, receipt)
    except PreparationError as exc:
        print(f"not staged: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
