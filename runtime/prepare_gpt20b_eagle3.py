#!/usr/bin/env python3
"""Stage and validate the pinned GPT-OSS 20B EAGLE-3 draft model.

The helper downloads one public GGUF into the RAID model directory, verifies
its full SHA-256 through ``range_fetch``, then validates its GGUF directory and
tokenizer against the already-verified local GPT-OSS 20B target.  It never
starts a server or touches the target model payload.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import re
import stat
import struct
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import quote

if __package__:
    from . import range_fetch
else:
    import range_fetch


ROOT = Path(__file__).resolve().parents[1]
LLAMA_SOURCE = ROOT / "vendor" / "strata" / "third_party" / "llama.cpp"
READER_PATH = ROOT / "vendor" / "strata" / "tools" / "gguf_reader.py"

REPOSITORY = "ggml-org/gpt-oss-20b-GGUF"
REVISION = "ef9b12f2ff56c69cf32153a02784e7a3c88bf524"
SOURCE_REPOSITORY = "RedHatAI/gpt-oss-20b-speculator.eagle3"
SOURCE_REVISION = "c2825cb49c32649bab8fb012df38df50a0a43db1"
FILENAME = "eagle3-gpt-oss-20b-Q8_0.gguf"
FILE_SIZE = 921_488_000
FILE_SHA256 = "6955f98a701e97a8c969f3ff33be8ba17b35bd60e5ac96258d2c42d710c7d63f"
DOWNLOAD_URL = f"https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/{quote(FILENAME)}"
DESTINATION = Path("/external/storage/moe-bench/models/gpt20b-eagle3-Q8_0")
DRAFT_MODEL = DESTINATION / FILENAME

TARGET_CONFIG = ROOT / "runtime" / "gpt20b-llama-32k.json"
TARGET_REPORT = ROOT / "results" / "gpt20b-llama-artifact.json"
TARGET_MODEL = Path(
    "/external/models/run/ollama/models/blobs/"
    "sha256-e7b273f9636059a689e3ddcab3716e4f65abe0143ac978e46673ad0e52d09efb"
)
TARGET_SHA256 = "e7b273f9636059a689e3ddcab3716e4f65abe0143ac978e46673ad0e52d09efb"
TARGET_SIZE = 13_793_422_144
REPORT = ROOT / "results" / "gpt20b-eagle3-artifact.json"
CPU_CONFIG = ROOT / "runtime" / "gpt20b-eagle3-cpu-32k.json"
GPU_CONFIG = ROOT / "runtime" / "gpt20b-eagle3-gpu-32k.json"
DERIVED_DESTINATION = Path("/external/storage/moe-bench/models/gpt20b-eagle3-Q8_0-derived")
DERIVED_MODEL = DERIVED_DESTINATION / "eagle3-gpt-oss-20b-Q8_0-target-tokenizer.gguf"
DERIVED_REPORT = ROOT / "results" / "gpt20b-eagle3-derived-artifact.json"
DERIVED_CPU_CONFIG = ROOT / "runtime" / "gpt20b-eagle3-derived-cpu-32k.json"
DERIVED_GPU_CONFIG = ROOT / "runtime" / "gpt20b-eagle3-derived-gpu-32k.json"

LLAMA_REVISION = "620d2225cc436ab0b001e3200e590e3b286ef9ab"
SERVER = Path("/external/storage/moe-bench/ollama-0.40.6/lib/ollama/llama-server")
SERVER_SHA256 = "234b05b2138264f8fb263c3205e85f4c290e8afe5067e280a4f6f90cdac5696b"
LLAMA_LIBRARY = SERVER.parent / "libllama.so.0.7.0"
COMMON_LIBRARY = SERVER.parent / "libllama-common.so.0.7.0"

CONTEXT = 32_768
TRAINED_DRAFT_TOKENS = 3
TARGET_LAYERS = [2, 12, 21]
TARGET_HIDDEN_SIZE = 2_880
TARGET_VOCAB_SIZE = 201_088
DRAFT_VOCAB_SIZE = 64_000

SOURCE_REQUIREMENTS = {
    "docs/speculative.md": (
        "RedHatAI/gpt-oss-20b-speculator.eagle3",
        "--target-model-dir",
        "--spec-type draft-eagle3",
        "number of tokens to draft for speculative decoding (default: 3)",
    ),
    "common/arg.cpp": (
        '"--spec-draft-device", "-devd", "--device-draft"',
        "none = don't offload, default: follows --device",
        '"--spec-draft-ngl", "-ngld"',
    ),
    "common/speculative.cpp": (
        "common_speculative_impl_draft_eagle3",
        "target_layer_ids_n != 3",
        "llama_set_embeddings_layer_inp(ctx_tgt",
        "features_buf.resize((size_t) n_tokens * n_embd_enc",
        "result.n_gpu_layers          = params_spec.n_gpu_layers",
    ),
    "src/models/eagle3.cpp": (
        "EAGLE3 model requires 'extract_layers' in GGUF metadata",
        "EAGLE3 requires exactly 3 entries in 'extract_layers'",
        "compatible with Readhat eagle3 speculator model",
        "GGML_ASSERT(n_layer == 1)",
    ),
    "tools/server/server-context.cpp": (
        "common_base_params_to_speculative(params_base)",
        "common_speculative_init_from_params(params_dft, model_tgt, ctx_tgt)",
    ),
}

EXPECTED_TENSORS = {
    "d2t": ([64_000], "I64"),
    "enc.output_norm.weight": ([8_640], "F32"),
    "fc.weight": ([8_640, 2_880], "Q8_0"),
    "output.weight": ([2_880, 64_000], "Q8_0"),
    "output_norm.weight": ([2_880], "F32"),
    "token_embd.weight": ([2_880, 201_088], "Q8_0"),
    "blk.0.attn_k.weight": ([5_760, 512], "Q8_0"),
    "blk.0.attn_norm.weight": ([2_880], "F32"),
    "blk.0.attn_norm_2.weight": ([2_880], "F32"),
    "blk.0.attn_output.weight": ([4_096, 2_880], "Q8_0"),
    "blk.0.attn_q.weight": ([5_760, 4_096], "Q8_0"),
    "blk.0.attn_v.weight": ([5_760, 512], "Q8_0"),
    "blk.0.ffn_down.weight": ([2_880, 2_880], "Q8_0"),
    "blk.0.ffn_gate.weight": ([2_880, 2_880], "Q8_0"),
    "blk.0.ffn_norm.weight": ([2_880], "F32"),
    "blk.0.ffn_up.weight": ([2_880, 2_880], "Q8_0"),
}

TOKENIZER_KEYS = (
    "tokenizer.ggml.model",
    "tokenizer.ggml.pre",
    "tokenizer.ggml.tokens",
    "tokenizer.ggml.token_type",
    "tokenizer.ggml.merges",
    "tokenizer.ggml.bos_token_id",
    "tokenizer.ggml.eos_token_id",
    "tokenizer.ggml.padding_token_id",
)

TARGET_TOKENIZER_KEYS = (
    "tokenizer.ggml.add_bos_token",
    "tokenizer.ggml.add_eos_token",
    "tokenizer.ggml.add_padding_token",
    "tokenizer.ggml.bos_token_id",
    "tokenizer.ggml.eos_token_id",
    "tokenizer.ggml.eos_token_ids",
    "tokenizer.ggml.merges",
    "tokenizer.ggml.model",
    "tokenizer.ggml.padding_token_id",
    "tokenizer.ggml.pre",
    "tokenizer.ggml.scores",
    "tokenizer.ggml.token_type",
    "tokenizer.ggml.tokens",
)

DERIVATION_POLICY_REJECTION = {
    "classification": "policy approval rejection; reviewer origin unknown",
    "tool": "functions.exec -> exec_command",
    "toolcall_id": None,
    "action": (
        "read-only rg lookup for EOS_IDS/eos_token_ids/special_eog_ids in pinned "
        "llama-vocab and llama-arch source"
    ),
    "reason": "approval required by policy, but AskForApproval is set to Never",
    "equivalent_retry_attempted": False,
}

_GGUF_SCALAR_BYTES = {
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
_GGUF_STRING = 8
_GGUF_ARRAY = 9


class PreparationError(RuntimeError):
    pass


@dataclasses.dataclass(frozen=True)
class RawMetadataEntry:
    key: str
    raw: bytes


@dataclasses.dataclass(frozen=True)
class RawGGUFHeader:
    version: int
    tensor_count: int
    metadata: tuple[RawMetadataEntry, ...]
    tensor_info: bytes
    header_end: int


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PreparationError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _read_exact(handle: Any, count: int, description: str) -> bytes:
    payload = handle.read(count)
    if len(payload) != count:
        raise PreparationError(f"truncated GGUF {description}")
    return payload


def _read_u32(handle: Any, description: str) -> int:
    return struct.unpack("<I", _read_exact(handle, 4, description))[0]


def _read_u64(handle: Any, description: str) -> int:
    return struct.unpack("<Q", _read_exact(handle, 8, description))[0]


def _read_gguf_string(handle: Any, description: str) -> str:
    length = _read_u64(handle, f"{description} length")
    if length > 64 * 1024 * 1024:
        raise PreparationError(f"unreasonable GGUF {description} length: {length}")
    try:
        return _read_exact(handle, length, description).decode("utf-8")
    except UnicodeDecodeError as exc:
        raise PreparationError(f"invalid UTF-8 in GGUF {description}") from exc


def _skip_gguf_value(handle: Any, value_type: int, description: str) -> None:
    scalar_bytes = _GGUF_SCALAR_BYTES.get(value_type)
    if scalar_bytes is not None:
        _read_exact(handle, scalar_bytes, description)
        return
    if value_type == _GGUF_STRING:
        _read_gguf_string(handle, description)
        return
    if value_type == _GGUF_ARRAY:
        element_type = _read_u32(handle, f"{description} array type")
        count = _read_u64(handle, f"{description} array count")
        if count > 10_000_000:
            raise PreparationError(f"unreasonable GGUF {description} array count: {count}")
        element_bytes = _GGUF_SCALAR_BYTES.get(element_type)
        if element_bytes is not None:
            _read_exact(handle, element_bytes * count, description)
            return
        if element_type == _GGUF_STRING:
            for index in range(count):
                _read_gguf_string(handle, f"{description}[{index}]")
            return
        raise PreparationError(f"unsupported GGUF {description} array element type: {element_type}")
    raise PreparationError(f"unsupported GGUF {description} value type: {value_type}")


def scan_raw_gguf_header(path: Path) -> RawGGUFHeader:
    """Read a GGUF header while retaining every metadata entry byte-for-byte."""

    try:
        file_size = path.stat().st_size
        with path.open("rb") as handle:
            _require(_read_exact(handle, 4, "magic") == b"GGUF", f"not a GGUF file: {path}")
            version = _read_u32(handle, "version")
            _require(version == 3, f"unsupported GGUF version: {version}")
            tensor_count = _read_u64(handle, "tensor count")
            metadata_count = _read_u64(handle, "metadata count")
            _require(tensor_count <= 1_000_000, "unreasonable GGUF tensor count")
            _require(metadata_count <= 1_000_000, "unreasonable GGUF metadata count")

            metadata: list[RawMetadataEntry] = []
            for index in range(metadata_count):
                start = handle.tell()
                key = _read_gguf_string(handle, f"metadata key {index}")
                value_type = _read_u32(handle, f"metadata type {key}")
                _skip_gguf_value(handle, value_type, f"metadata value {key}")
                end = handle.tell()
                handle.seek(start)
                raw = _read_exact(handle, end - start, f"metadata record {key}")
                handle.seek(end)
                metadata.append(RawMetadataEntry(key, raw))

            tensor_info_start = handle.tell()
            for index in range(tensor_count):
                _read_gguf_string(handle, f"tensor name {index}")
                dimensions = _read_u32(handle, f"tensor dimension count {index}")
                _require(1 <= dimensions <= 4, f"invalid GGUF tensor dimension count: {dimensions}")
                _read_exact(handle, dimensions * 8, f"tensor dimensions {index}")
                _read_exact(handle, 12, f"tensor type and offset {index}")
            tensor_info_end = handle.tell()
            handle.seek(tensor_info_start)
            tensor_info = _read_exact(
                handle,
                tensor_info_end - tensor_info_start,
                "tensor directory",
            )
    except OSError as exc:
        raise PreparationError(f"GGUF header is unreadable: {path}") from exc

    _require(tensor_info_end <= file_size, "GGUF header exceeds file size")
    return RawGGUFHeader(
        version=version,
        tensor_count=tensor_count,
        metadata=tuple(metadata),
        tensor_info=tensor_info,
        header_end=tensor_info_end,
    )


def _metadata_records_digest(records: Sequence[RawMetadataEntry]) -> str:
    digest = hashlib.sha256()
    for record in sorted(records, key=lambda item: item.key):
        digest.update(len(record.raw).to_bytes(8, "little"))
        digest.update(record.raw)
    return digest.hexdigest()


def _sha256_from_offset(path: Path, offset: int) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        handle.seek(offset)
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def target_aligned_metadata_records(
    source_header: RawGGUFHeader,
    target_header: RawGGUFHeader,
) -> tuple[tuple[RawMetadataEntry, ...], tuple[RawMetadataEntry, ...]]:
    """Replace all draft tokenizer metadata with the target's exact raw records."""

    target_tokenizer = tuple(
        record for record in target_header.metadata if record.key.startswith("tokenizer.ggml.")
    )
    target_keys = tuple(sorted(record.key for record in target_tokenizer))
    _require(target_keys == tuple(sorted(TARGET_TOKENIZER_KEYS)), "target tokenizer metadata key set changed")
    _require(
        len({record.key for record in target_tokenizer}) == len(target_tokenizer),
        "target tokenizer metadata contains duplicate keys",
    )
    retained = tuple(
        record
        for record in source_header.metadata
        if not record.key.startswith("tokenizer.ggml.") and record.key != "tokenizer.chat_template"
    )
    derived_metadata = retained + target_tokenizer
    _require(
        len({record.key for record in derived_metadata}) == len(derived_metadata),
        "derived metadata contains duplicate keys",
    )
    return derived_metadata, target_tokenizer


def _read_json(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PreparationError(f"{description} is unreadable or invalid: {path}") from exc
    _require(isinstance(value, dict), f"{description} must be a JSON object")
    return value


def _load_reader() -> type[Any]:
    tools_path = str(READER_PATH.parent)
    if tools_path not in sys.path:
        sys.path.insert(0, tools_path)
    from gguf_reader import GGUFFile

    return GGUFFile


def safe_error_detail(exc: BaseException) -> str:
    detail = " ".join(str(exc).split())
    detail = re.sub(r"https?://\S+", "[redacted-url]", detail, flags=re.IGNORECASE)
    return (detail or type(exc).__name__)[:500]


def validate_source(source_root: Path = LLAMA_SOURCE) -> dict[str, Any]:
    files: dict[str, dict[str, Any]] = {}
    for relative, fragments in SOURCE_REQUIREMENTS.items():
        path = source_root / relative
        try:
            payload = path.read_bytes()
            text = payload.decode("utf-8")
        except (OSError, UnicodeError) as exc:
            raise PreparationError(f"pinned llama.cpp source is unreadable: {path}") from exc
        for fragment in fragments:
            _require(fragment in text, f"pinned source support check failed: {relative}: {fragment}")
        files[relative] = {
            "sha256": hashlib.sha256(payload).hexdigest(),
            "required_fragments_present": True,
        }
    return {
        "revision": LLAMA_REVISION,
        "files": files,
        "draft_architecture": "eagle3",
        "target_layers": TARGET_LAYERS,
        "target_hidden_size": TARGET_HIDDEN_SIZE,
        "cpu_draft_semantics": "--spec-draft-device none plus --spec-draft-ngl 0",
        "gpu_draft_semantics": "omit --spec-draft-device to follow target device; --spec-draft-ngl auto",
        "target_hidden_transfer": (
            "selected target layer inputs are copied to host output buffers, concatenated as F32, "
            "then passed to the draft encoder on its selected backend"
        ),
        "source_support_validated": True,
    }


def validate_bundle(
    server: Path = SERVER,
    llama_library: Path = LLAMA_LIBRARY,
    common_library: Path = COMMON_LIBRARY,
) -> dict[str, Any]:
    try:
        server_digest = _sha256(server)
        llama_bytes = llama_library.read_bytes()
        common_bytes = common_library.read_bytes()
    except OSError as exc:
        raise PreparationError("installed llama-server bundle is unreadable") from exc
    _require(server_digest == SERVER_SHA256, "installed llama-server wrapper digest changed")
    for marker in (b"EAGLE3 model requires", b"EAGLE3 extract_layers"):
        _require(marker in llama_bytes, f"installed libllama lacks EAGLE-3 marker {marker!r}")
    for marker in (b"draft-eagle3", b"spec-draft-ngl"):
        _require(marker in common_bytes, f"installed common library lacks EAGLE-3 marker {marker!r}")
    return {
        "server": str(server),
        "server_sha256": server_digest,
        "libllama_sha256": hashlib.sha256(llama_bytes).hexdigest(),
        "libllama_common_sha256": hashlib.sha256(common_bytes).hexdigest(),
        "eagle3_loader_present": True,
        "draft_eagle3_cli_present": True,
    }


def validate_target_contract() -> dict[str, Any]:
    config = _read_json(TARGET_CONFIG, "target config")
    receipt = _read_json(TARGET_REPORT, "target artifact receipt")
    verdict = receipt.get("verdict")
    model = receipt.get("model")
    _require(isinstance(verdict, dict) and verdict.get("artifact_complete") is True, "target artifact is not verified")
    _require(isinstance(model, dict) and model.get("fresh_full_sha256_verified") is True, "target has no fresh full SHA receipt")
    _require(config.get("model") == str(TARGET_MODEL), "target config model changed")
    _require(config.get("context") == CONTEXT, "target config context changed")
    _require(config.get("host") == "127.0.0.1" and config.get("port") == 18081, "target config loopback binding changed")
    rows = config.get("shards")
    _require(isinstance(rows, list) and len(rows) == 1, "target config shard table changed")
    row = rows[0]
    _require(
        isinstance(row, dict)
        and row.get("sha256") == TARGET_SHA256
        and row.get("size") == TARGET_SIZE,
        "target config digest or size changed",
    )
    return {
        "config": str(TARGET_CONFIG),
        "artifact_manifest": str(TARGET_REPORT),
        "model": str(TARGET_MODEL),
        "size": TARGET_SIZE,
        "sha256": TARGET_SHA256,
        "fresh_full_sha256_verified": True,
        "context": CONTEXT,
    }


def _tokenizer_digest(metadata: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    for key in TOKENIZER_KEYS:
        _require(key in metadata, f"GGUF tokenizer lacks {key}")
        encoded = json.dumps(metadata[key], ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        digest.update(len(key).to_bytes(4, "little"))
        digest.update(key.encode("utf-8"))
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    return digest.hexdigest()


def _tensor_byte_count(tensor: Any, type_name: str) -> int | None:
    byte_count = tensor.expected_bytes()
    if byte_count is None and type_name == "I64":
        return tensor.elements * 8
    return byte_count


def inspect_draft(
    draft_path: Path = DRAFT_MODEL,
    target_path: Path = TARGET_MODEL,
    reader_class: type[Any] | None = None,
    *,
    full_sha_verified: bool,
    expected_size: int = FILE_SIZE,
    expected_sha256: str = FILE_SHA256,
) -> dict[str, Any]:
    draft_path = draft_path.expanduser().resolve()
    target_path = target_path.expanduser().resolve()
    try:
        details = draft_path.lstat()
        target_details = target_path.lstat()
    except OSError as exc:
        raise PreparationError("draft or target GGUF is inaccessible") from exc
    _require(stat.S_ISREG(details.st_mode), "draft model must be a regular file")
    _require(os.access(draft_path, os.R_OK), "draft model is unreadable")
    _require(details.st_size == expected_size, f"draft size is {details.st_size}, expected {expected_size}")
    _require(stat.S_ISREG(target_details.st_mode) and target_details.st_size == TARGET_SIZE, "target GGUF changed")

    reader_type = reader_class or _load_reader()
    draft = reader_type(draft_path)
    target = reader_type(target_path)
    metadata = draft.metadata
    target_metadata = target.metadata

    expected_metadata = {
        "general.architecture": "eagle3",
        "general.name": "gpt-oss-20b",
        "eagle3.target_layers": TARGET_LAYERS,
        "eagle3.target_hidden_size": TARGET_HIDDEN_SIZE,
        "eagle3.norm_before_residual": True,
        "eagle3.norm_before_fc": True,
        "eagle3.block_count": 1,
        "eagle3.context_length": 131_072,
        "eagle3.embedding_length": 2_880,
        "eagle3.feed_forward_length": 2_880,
        "eagle3.attention.head_count": 64,
        "eagle3.attention.head_count_kv": 8,
        "eagle3.attention.key_length": 64,
        "eagle3.attention.value_length": 64,
        "eagle3.vocab_size": TARGET_VOCAB_SIZE,
    }
    for key, expected in expected_metadata.items():
        _require(metadata.get(key) == expected, f"draft metadata {key} does not match {expected!r}")

    target_arch = target_metadata.get("general.architecture")
    _require(target_arch in ("gpt-oss", "gptoss"), "target architecture is not GPT-OSS")
    target_prefix = str(target_arch)
    _require(target_metadata.get(f"{target_prefix}.block_count") == 24, "target is not GPT-OSS 20B")
    _require(target_metadata.get(f"{target_prefix}.embedding_length") == TARGET_HIDDEN_SIZE, "target hidden size is not 2880")
    target_tokenizer_digest = _tokenizer_digest(target_metadata)
    draft_tokenizer_digest = _tokenizer_digest(metadata)
    _require(draft_tokenizer_digest == target_tokenizer_digest, "draft tokenizer does not exactly match target tokenizer")

    tensors = {tensor.name: tensor for tensor in draft.tensors}
    _require(len(tensors) == len(EXPECTED_TENSORS) == len(draft.tensors), "draft tensor set is not the expected 16 tensors")
    ranges: list[tuple[int, int, str]] = []
    tensor_types: Counter[str] = Counter()
    for name, (shape, type_name) in EXPECTED_TENSORS.items():
        tensor = tensors.get(name)
        _require(tensor is not None, f"draft tensor is missing: {name}")
        _require(tensor.shape == shape, f"draft tensor {name} has shape {tensor.shape}, expected {shape}")
        _require(tensor.type_name == type_name, f"draft tensor {name} has type {tensor.type_name}, expected {type_name}")
        byte_count = _tensor_byte_count(tensor, type_name)
        _require(byte_count is not None and byte_count > 0, f"unsupported tensor layout: {name}")
        _require(tensor.offset >= 0 and tensor.offset % draft.alignment == 0, f"unaligned tensor: {name}")
        start = draft.data_start + tensor.offset
        end = start + byte_count
        _require(draft.data_start <= start < end <= details.st_size, f"tensor exceeds draft bounds: {name}")
        ranges.append((start, end, name))
        tensor_types[type_name] += 1
    ranges.sort()
    for previous, current in zip(ranges, ranges[1:]):
        _require(previous[1] <= current[0], f"draft tensor payload overlap: {previous[2]}, {current[2]}")
    _require(tensor_types == Counter({"Q8_0": 10, "F32": 5, "I64": 1}), "draft tensor type counts changed")

    return {
        "path": str(draft_path),
        "size": details.st_size,
        "sha256": expected_sha256,
        "fresh_full_sha256_verified": full_sha_verified,
        "gguf": {
            "version": draft.version,
            "metadata": expected_metadata,
            "tensor_count": len(draft.tensors),
            "tensor_type_counts": dict(sorted(tensor_types.items())),
            "all_tensor_payload_bounds_validated": True,
            "all_offsets_aligned": True,
            "no_payload_overlaps": True,
        },
        "target_compatibility": {
            "target_path": str(target_path),
            "target_architecture": target_arch,
            "target_block_count": 24,
            "target_hidden_size": TARGET_HIDDEN_SIZE,
            "target_layers": TARGET_LAYERS,
            "tokenizer_keys_compared": list(TOKENIZER_KEYS),
            "target_tokenizer_sha256": target_tokenizer_digest,
            "draft_tokenizer_sha256": draft_tokenizer_digest,
            "tokenizers_exactly_match": True,
        },
        "stat": {
            "device": details.st_dev,
            "inode": details.st_ino,
            "mode": details.st_mode,
            "links": details.st_nlink,
            "mtime_ns": details.st_mtime_ns,
            "ctime_ns": details.st_ctime_ns,
        },
    }


def audit_tokenizer_and_d2t(
    draft_path: Path = DRAFT_MODEL,
    target_path: Path = TARGET_MODEL,
    reader_class: type[Any] | None = None,
) -> dict[str, Any]:
    """Prove that the published mismatch is metadata-only for trained IDs."""

    reader_type = reader_class or _load_reader()
    draft = reader_type(draft_path)
    target = reader_type(target_path)
    draft_metadata = draft.metadata
    target_metadata = target.metadata

    draft_tokens = draft_metadata.get("tokenizer.ggml.tokens")
    target_tokens = target_metadata.get("tokenizer.ggml.tokens")
    draft_types = draft_metadata.get("tokenizer.ggml.token_type")
    target_types = target_metadata.get("tokenizer.ggml.token_type")
    _require(
        isinstance(draft_tokens, list)
        and isinstance(target_tokens, list)
        and len(draft_tokens) == len(target_tokens) == TARGET_VOCAB_SIZE,
        "draft and target token tables must both contain 201088 entries",
    )
    _require(
        isinstance(draft_types, list)
        and isinstance(target_types, list)
        and len(draft_types) == len(target_types) == TARGET_VOCAB_SIZE,
        "draft and target token-type tables must both contain 201088 entries",
    )
    mismatch_ids = [
        index for index, (draft_token, target_token) in enumerate(zip(draft_tokens, target_tokens))
        if draft_token != target_token
    ]
    type_mismatch_ids = [
        index for index, (draft_type, target_type) in enumerate(zip(draft_types, target_types))
        if draft_type != target_type
    ]
    expected_tail = list(range(200_019, TARGET_VOCAB_SIZE))
    _require(mismatch_ids == expected_tail, "token mismatch is not the known contiguous padding tail")
    _require(type_mismatch_ids == expected_tail, "token-type mismatch is not the known padding tail")
    _require(
        all(
            draft_tokens[index] == f"[PAD{index}]"
            and target_tokens[index] == f"[PAD{index - 200_019}]"
            and draft_types[index] == 5
            and target_types[index] == 4
            for index in expected_tail
        ),
        "padding-tail token or type pattern changed",
    )
    _require(
        draft_metadata.get("tokenizer.ggml.merges") == target_metadata.get("tokenizer.ggml.merges"),
        "draft and target BPE merges differ",
    )
    _require(
        draft_tokens[:200_019] == target_tokens[:200_019]
        and draft_types[:200_019] == target_types[:200_019],
        "semantic tokenizer range differs",
    )

    d2t_tensor = next((tensor for tensor in draft.tensors if tensor.name == "d2t"), None)
    _require(
        d2t_tensor is not None
        and d2t_tensor.shape == [DRAFT_VOCAB_SIZE]
        and d2t_tensor.type_name == "I64",
        "draft d2t tensor contract changed",
    )
    d2t_start = draft.data_start + d2t_tensor.offset
    try:
        with draft_path.open("rb") as handle:
            handle.seek(d2t_start)
            payload = _read_exact(handle, DRAFT_VOCAB_SIZE * 8, "d2t tensor")
    except OSError as exc:
        raise PreparationError("draft d2t tensor is unreadable") from exc
    d2t = struct.unpack(f"<{DRAFT_VOCAB_SIZE}q", payload)
    _require(len(set(d2t)) == DRAFT_VOCAB_SIZE, "draft d2t mapping contains duplicates")
    _require(min(d2t) >= 0 and max(d2t) < TARGET_VOCAB_SIZE, "draft d2t mapping is out of target range")
    mapped_padding_tail = [token_id for token_id in d2t if token_id >= 200_019]
    _require(not mapped_padding_tail, "draft d2t maps an output into the mismatched padding tail")
    _require(
        all(draft_tokens[token_id] == target_tokens[token_id] for token_id in d2t),
        "draft d2t maps an output to a semantically different target token",
    )

    special_ids = {
        token: target_tokens.index(token)
        for token in ("<|channel|>", "<|start|>", "<|end|>", "<|message|>")
    }
    d2t_positions = {
        token: d2t.index(token_id)
        for token, token_id in special_ids.items()
    }
    target_eos_ids = target_metadata.get("tokenizer.ggml.eos_token_ids")
    _require(target_eos_ids == [199_999, 200_002, 200_012], "target Harmony EOG IDs changed")
    _require(target_metadata.get("tokenizer.ggml.add_eos_token") is False, "target add_eos semantics changed")
    _require(target_metadata.get("tokenizer.ggml.pre") == "default", "target pre-tokenizer changed")
    _require(draft_metadata.get("tokenizer.ggml.pre") == "gpt-4o", "published draft pre-tokenizer changed")

    return {
        "vocab_size": TARGET_VOCAB_SIZE,
        "semantic_id_range": [0, 200_018],
        "semantic_tokens_and_types_exactly_match": True,
        "padding_tail": {
            "first_id": 200_019,
            "last_id": 201_087,
            "count": len(expected_tail),
            "contiguous": True,
            "draft_pattern": "[PAD<absolute-token-id>] / token_type 5",
            "target_pattern": "[PAD<zero-based-tail-index>] / token_type 4",
        },
        "merges_exactly_match": True,
        "draft_pre": "gpt-4o",
        "target_pre": "default",
        "draft_eos_token_id": draft_metadata.get("tokenizer.ggml.eos_token_id"),
        "target_eos_token_id": target_metadata.get("tokenizer.ggml.eos_token_id"),
        "target_eos_token_ids": target_eos_ids,
        "target_add_eos_token": False,
        "d2t": {
            "sha256": hashlib.sha256(payload).hexdigest(),
            "count": len(d2t),
            "unique": len(set(d2t)),
            "minimum": min(d2t),
            "maximum": max(d2t),
            "mapped_padding_tail_count": 0,
            "all_mapped_token_strings_match": True,
            "harmony_special_output_positions": d2t_positions,
        },
        "derivation_safe": True,
        "derivation_rationale": (
            "The EAGLE tensors use target token IDs for their 201088-row embedding and a unique "
            "64000-entry d2t output map. Every mapped ID is in the unchanged semantic range; the "
            "only token-table differences are unreachable padding labels/types. Copying the exact "
            "target tokenizer metadata aligns pre-tokenization and Harmony stop semantics without "
            "renumbering or changing any trained tensor value."
        ),
    }


def derive_target_aligned_draft(
    source_path: Path = DRAFT_MODEL,
    target_path: Path = TARGET_MODEL,
    output_path: Path = DERIVED_MODEL,
    reader_class: type[Any] | None = None,
) -> dict[str, Any]:
    """Create a separate draft whose tokenizer metadata exactly matches the target."""

    source_path = source_path.expanduser().resolve()
    target_path = target_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    _require(output_path != source_path and output_path != target_path, "derived output must be a new file")

    source_sha256 = _sha256(source_path)
    _require(source_sha256 == FILE_SHA256, "published draft full SHA-256 changed")
    audit = audit_tokenizer_and_d2t(source_path, target_path, reader_class)
    source_header = scan_raw_gguf_header(source_path)
    target_header = scan_raw_gguf_header(target_path)
    reader_type = reader_class or _load_reader()
    source_reader = reader_type(source_path)

    derived_metadata, target_tokenizer = target_aligned_metadata_records(source_header, target_header)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{output_path.name}.", dir=output_path.parent)
    temporary = Path(temporary_name)
    source_data_sha = hashlib.sha256()
    try:
        with os.fdopen(descriptor, "wb") as output, source_path.open("rb") as source:
            output.write(b"GGUF")
            output.write(struct.pack("<IQQ", source_header.version, source_header.tensor_count, len(derived_metadata)))
            for record in derived_metadata:
                output.write(record.raw)
            output.write(source_header.tensor_info)
            padding = (-output.tell()) % source_reader.alignment
            output.write(b"\0" * padding)
            derived_data_start = output.tell()

            source.seek(source_reader.data_start)
            while block := source.read(8 * 1024 * 1024):
                source_data_sha.update(block)
                output.write(block)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, output_path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise

    derived_size = output_path.stat().st_size
    derived_sha256 = _sha256(output_path)
    derived_reader = reader_type(output_path)
    _require(derived_reader.data_start == derived_data_start, "derived GGUF data offset is inconsistent")
    derived_data_sha256 = _sha256_from_offset(output_path, derived_data_start)
    source_data_sha256 = source_data_sha.hexdigest()
    _require(derived_data_sha256 == source_data_sha256, "derived tensor data region differs from source")

    derived_header = scan_raw_gguf_header(output_path)
    derived_tokenizer = tuple(
        record for record in derived_header.metadata if record.key.startswith("tokenizer.ggml.")
    )
    target_tokenizer_digest = _metadata_records_digest(target_tokenizer)
    derived_tokenizer_digest = _metadata_records_digest(derived_tokenizer)
    _require(derived_tokenizer_digest == target_tokenizer_digest, "derived tokenizer metadata is not exact")
    _require(
        all(record.key != "tokenizer.chat_template" for record in derived_header.metadata),
        "draft chat template was not removed from derived artifact",
    )

    draft = inspect_draft(
        output_path,
        target_path,
        reader_type,
        full_sha_verified=True,
        expected_size=derived_size,
        expected_sha256=derived_sha256,
    )
    return {
        "source": {
            "path": str(source_path),
            "size": source_path.stat().st_size,
            "sha256": source_sha256,
            "fresh_full_sha256_verified": True,
        },
        "derived": {
            "path": str(output_path),
            "size": derived_size,
            "sha256": derived_sha256,
            "fresh_full_sha256_verified": True,
            "data_start": derived_data_start,
            "tensor_data_region_sha256": derived_data_sha256,
        },
        "tensor_payload_identity": {
            "source_data_start": source_reader.data_start,
            "derived_data_start": derived_data_start,
            "source_tensor_data_region_sha256": source_data_sha256,
            "derived_tensor_data_region_sha256": derived_data_sha256,
            "all_tensor_payload_bytes_unchanged": True,
            "tensor_directory_copied_byte_for_byte": source_header.tensor_info == derived_header.tensor_info,
        },
        "tokenizer_metadata": {
            "keys": list(TARGET_TOKENIZER_KEYS),
            "target_raw_metadata_sha256": target_tokenizer_digest,
            "derived_raw_metadata_sha256": derived_tokenizer_digest,
            "exact_raw_match": True,
            "source_chat_template_removed": True,
        },
        "semantic_audit": audit,
        "draft_validation": draft,
    }


def _profile_args(profile: str, draft_model: Path = DRAFT_MODEL) -> list[str]:
    common = [
        "--spec-type",
        "draft-eagle3",
        "--spec-draft-model",
        str(draft_model),
        "--spec-draft-n-max",
        str(TRAINED_DRAFT_TOKENS),
        "--spec-draft-n-min",
        "0",
        "--spec-draft-p-min",
        "0.0",
        "--spec-draft-type-k",
        "f16",
        "--spec-draft-type-v",
        "f16",
    ]
    if profile == "cpu":
        return common + ["--spec-draft-device", "none", "--spec-draft-ngl", "0"]
    if profile == "gpu":
        # There is no CLI "auto" sentinel for --spec-draft-device.  Omitting
        # it is the documented way to follow the target's device list.
        return common + ["--spec-draft-ngl", "auto"]
    raise PreparationError(f"unknown profile: {profile}")


def config_document(
    profile: str,
    draft_report: Path = REPORT,
    draft_model: Path = DRAFT_MODEL,
    config_path: Path = CPU_CONFIG,
) -> dict[str, Any]:
    base = _read_json(TARGET_CONFIG, "target config")
    value = json.loads(json.dumps(base))
    args = value.get("args")
    _require(isinstance(args, list), "target config args are malformed")
    value["args"] = args + _profile_args(profile, draft_model)
    value["draft_artifact_manifest"] = os.path.relpath(draft_report, config_path.parent)
    value["draft_profile"] = profile
    return value


def artifact_report(
    *,
    source: Mapping[str, Any],
    target: Mapping[str, Any],
    bundle: Mapping[str, Any] | None = None,
    download: Mapping[str, Any] | None = None,
    draft: Mapping[str, Any] | None = None,
    error: BaseException | None = None,
) -> dict[str, Any]:
    full_sha_verified = bool(
        (download and download.get("full_sha256_verified") is True)
        or (draft and draft.get("fresh_full_sha256_verified") is True)
    )
    header_valid = bool(draft)
    complete = full_sha_verified and header_valid
    if complete:
        state = "inactive_verified"
    elif full_sha_verified and error is not None:
        state = "inactive_compatibility_blocked"
    elif full_sha_verified:
        state = "inactive_validation_pending"
    else:
        state = "inactive_download_pending"
    value: dict[str, Any] = {
        "schema_version": 1,
        "checked_at": utc_now(),
        "state": state,
        "source_artifact": {
            "repository": SOURCE_REPOSITORY,
            "revision": SOURCE_REVISION,
            "checkpoint_size": 1_709_770_176,
            "checkpoint_sha256": "43c37a7067306dc0821737e734a7057aeb5ac5c9976215f21bd6fd3cab7c810e",
        },
        "gguf_artifact": {
            "repository": REPOSITORY,
            "revision": REVISION,
            "name": FILENAME,
            "path": str(DRAFT_MODEL),
            "size": FILE_SIZE,
            "sha256": FILE_SHA256,
            "conversion_source_revision": SOURCE_REVISION,
            "converted_by": "ggml-org",
            "quantization": "Q8_0",
        },
        "download_verification": dict(download or {}),
        "draft": dict(draft or {}),
        "target": dict(target),
        "llama_cpp": {
            "revision": LLAMA_REVISION,
            "source_evidence": dict(source),
            "installed_bundle_evidence": dict(bundle or {}),
        },
        "profiles": {
            "cpu": {
                "config": str(CPU_CONFIG),
                "draft_device": "none",
                "draft_gpu_layers": 0,
                "trained_max_draft_tokens": TRAINED_DRAFT_TOKENS,
                "config_candidate_written": CPU_CONFIG.is_file(),
                "runtime_validated": False,
            },
            "gpu": {
                "config": str(GPU_CONFIG),
                "draft_device": "follows target device (option omitted)",
                "draft_gpu_layers": "auto",
                "trained_max_draft_tokens": TRAINED_DRAFT_TOKENS,
                "config_candidate_written": GPU_CONFIG.is_file(),
                "runtime_validated": False,
            },
        },
        "memory_projection": {
            "q8_weight_bytes": FILE_SIZE,
            "draft_f16_kv_bytes_at_32k": 67_108_864,
            "target_hidden_features_per_token_f32_bytes": 34_560,
            "target_hidden_extraction_buffer_location": "host output buffer",
            "gpu_fit_status": "nominally within measured margin but runtime-unvalidated",
        },
        "verdict": {
            "artifact_complete": complete,
            "full_sha256_verified": full_sha_verified,
            "header_and_tensor_directory_valid": header_valid,
            "target_tokenizer_exact_match": bool(
                draft
                and isinstance(draft.get("target_compatibility"), Mapping)
                and draft["target_compatibility"].get("tokenizers_exactly_match") is True
            ),
            "source_compatible": True,
            "cpu_profile_staged": complete,
            "gpu_profile_staged": complete,
            "profile_publication_staged": complete,
            "runtime_load_validated": False,
            "inference_validated": False,
        },
    }
    if error is not None:
        value["error"] = type(error).__name__
        value["error_detail"] = safe_error_detail(error)
    return value


def derived_artifact_report(
    *,
    source_evidence: Mapping[str, Any],
    target: Mapping[str, Any],
    bundle: Mapping[str, Any],
    derivation: Mapping[str, Any] | None = None,
    error: BaseException | None = None,
) -> dict[str, Any]:
    derived = derivation.get("derived") if isinstance(derivation, Mapping) else None
    tensor_identity = derivation.get("tensor_payload_identity") if isinstance(derivation, Mapping) else None
    tokenizer = derivation.get("tokenizer_metadata") if isinstance(derivation, Mapping) else None
    semantic = derivation.get("semantic_audit") if isinstance(derivation, Mapping) else None
    draft_validation = derivation.get("draft_validation") if isinstance(derivation, Mapping) else None
    complete = bool(
        isinstance(derived, Mapping)
        and derived.get("fresh_full_sha256_verified") is True
        and isinstance(tensor_identity, Mapping)
        and tensor_identity.get("all_tensor_payload_bytes_unchanged") is True
        and tensor_identity.get("tensor_directory_copied_byte_for_byte") is True
        and isinstance(tokenizer, Mapping)
        and tokenizer.get("exact_raw_match") is True
        and isinstance(semantic, Mapping)
        and semantic.get("derivation_safe") is True
        and isinstance(draft_validation, Mapping)
    )
    value: dict[str, Any] = {
        "schema_version": 1,
        "checked_at": utc_now(),
        "state": "inactive_derived_verified" if complete else "inactive_derived_blocked",
        "original_blocked_receipt": str(REPORT),
        "source_artifact": dict(derivation.get("source", {}) if derivation else {}),
        "gguf_artifact": {
            "path": str(derived.get("path", DERIVED_MODEL)) if isinstance(derived, Mapping) else str(DERIVED_MODEL),
            "size": derived.get("size") if isinstance(derived, Mapping) else None,
            "sha256": derived.get("sha256") if isinstance(derived, Mapping) else None,
            "quantization": "Q8_0",
            "derivation": "exact target-tokenizer metadata transplant; tensor directory and data unchanged",
        },
        "target": {
            "path": str(target.get("model", TARGET_MODEL)),
            "size": target.get("size", TARGET_SIZE),
            "sha256": target.get("sha256", TARGET_SHA256),
            "fresh_full_sha256_verified": target.get("fresh_full_sha256_verified") is True,
            "config": target.get("config"),
        },
        "compatibility": {
            "activation_allowed": complete,
            "target_identity_bound": True,
            "target_tokenizer_metadata_exact_raw_match": bool(
                isinstance(tokenizer, Mapping) and tokenizer.get("exact_raw_match") is True
            ),
            "trained_tensor_payloads_unchanged": bool(
                isinstance(tensor_identity, Mapping)
                and tensor_identity.get("all_tensor_payload_bytes_unchanged") is True
            ),
            "tensor_directory_unchanged": bool(
                isinstance(tensor_identity, Mapping)
                and tensor_identity.get("tensor_directory_copied_byte_for_byte") is True
            ),
            "d2t_semantic_mapping_validated": bool(
                isinstance(semantic, Mapping)
                and isinstance(semantic.get("d2t"), Mapping)
                and semantic["d2t"].get("all_mapped_token_strings_match") is True
            ),
            "runtime_load_validated": False,
            "inference_validated": False,
        },
        "derivation_evidence": dict(derivation or {}),
        "llama_cpp": {
            "revision": LLAMA_REVISION,
            "source_evidence": dict(source_evidence),
            "installed_bundle_evidence": dict(bundle),
        },
        "profiles": {
            "cpu": {
                "config": str(DERIVED_CPU_CONFIG),
                "draft_device": "none",
                "draft_gpu_layers": 0,
                "runtime_validated": False,
            },
            "gpu": {
                "config": str(DERIVED_GPU_CONFIG),
                "draft_device": "follows target device (option omitted)",
                "draft_gpu_layers": "auto",
                "runtime_validated": False,
            },
        },
        "policy_approval_rejection": dict(DERIVATION_POLICY_REJECTION),
        "verdict": {
            "artifact_complete": complete,
            "full_sha256_verified": bool(
                isinstance(derived, Mapping) and derived.get("fresh_full_sha256_verified") is True
            ),
            "header_and_tensor_directory_valid": isinstance(draft_validation, Mapping),
            "target_tokenizer_exact_match": bool(
                isinstance(tokenizer, Mapping) and tokenizer.get("exact_raw_match") is True
            ),
            "source_compatible": True,
            "cpu_profile_staged": complete,
            "gpu_profile_staged": complete,
            "profile_publication_staged": complete,
            "runtime_load_validated": False,
            "inference_validated": False,
        },
    }
    if error is not None:
        value["error"] = type(error).__name__
        value["error_detail"] = safe_error_detail(error)
    return value


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def publish_configs(report_path: Path = REPORT) -> None:
    atomic_write_json(CPU_CONFIG, config_document("cpu", report_path, DRAFT_MODEL, CPU_CONFIG))
    atomic_write_json(GPU_CONFIG, config_document("gpu", report_path, DRAFT_MODEL, GPU_CONFIG))


def publish_derived_configs(report_path: Path = DERIVED_REPORT) -> None:
    atomic_write_json(
        DERIVED_CPU_CONFIG,
        config_document("cpu", report_path, DERIVED_MODEL, DERIVED_CPU_CONFIG),
    )
    atomic_write_json(
        DERIVED_GPU_CONFIG,
        config_document("gpu", report_path, DERIVED_MODEL, DERIVED_GPU_CONFIG),
    )


def stage(
    destination: Path = DESTINATION,
    report_path: Path = REPORT,
    *,
    workers: int,
    chunk_mib: int,
    retries: int,
) -> dict[str, Any]:
    _require(
        destination.expanduser().resolve() == DESTINATION.resolve(),
        f"destination must remain the pinned path {DESTINATION}",
    )
    source = validate_source()
    target = validate_target_contract()
    bundle = validate_bundle()
    destination.mkdir(parents=True, exist_ok=True)
    output = destination / FILENAME
    download: dict[str, Any] | None = None
    try:
        range_fetch.download_file(
            DOWNLOAD_URL,
            output,
            FILE_SIZE,
            FILE_SHA256,
            workers=workers,
            chunk_size=chunk_mib * 1024 * 1024,
            retries=retries,
        )
        download = {
            "path": str(output),
            "size": FILE_SIZE,
            "sha256": FILE_SHA256,
            "full_sha256_verified": True,
            "verified_at": utc_now(),
            "verification_method": (
                "runtime.range_fetch.download_file completed exact-size and full-SHA256 validation"
            ),
        }
        draft = inspect_draft(output, full_sha_verified=True)
        report = artifact_report(
            source=source,
            target=target,
            bundle=bundle,
            download=download,
            draft=draft,
        )
        atomic_write_json(report_path, report)
        publish_configs(report_path)
        return report
    except BaseException as exc:
        report = artifact_report(
            source=source,
            target=target,
            bundle=bundle,
            download=download,
            error=exc,
        )
        atomic_write_json(report_path, report)
        raise


def derive_stage(report_path: Path = DERIVED_REPORT) -> dict[str, Any]:
    source = validate_source()
    target = validate_target_contract()
    bundle = validate_bundle()
    derivation: dict[str, Any] | None = None
    try:
        derivation = derive_target_aligned_draft()
        report = derived_artifact_report(
            source_evidence=source,
            target=target,
            bundle=bundle,
            derivation=derivation,
        )
        _require(report["compatibility"]["activation_allowed"] is True, "derived activation gate failed")
        atomic_write_json(report_path, report)
        publish_derived_configs(report_path)
        return report
    except BaseException as exc:
        report = derived_artifact_report(
            source_evidence=source,
            target=target,
            bundle=bundle,
            derivation=derivation,
            error=exc,
        )
        atomic_write_json(report_path, report)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=DESTINATION)
    parser.add_argument("--report", type=Path, default=REPORT)
    parser.add_argument("--workers", type=range_fetch.positive_int, default=4)
    parser.add_argument("--chunk-mib", type=range_fetch.positive_int, default=16)
    parser.add_argument("--retries", type=range_fetch.positive_int, default=4)
    parser.add_argument(
        "--download",
        action="store_true",
        help="download and fully verify the pinned draft; without this flag only print the plan",
    )
    parser.add_argument(
        "--derive-target-tokenizer",
        action="store_true",
        help=(
            "create and fully verify a separate draft with exact target tokenizer metadata; "
            "the published source and target files remain untouched"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        _require(not (args.download and args.derive_target_tokenizer), "choose either download or derivation")
        if args.derive_target_tokenizer:
            _require(args.report == REPORT, "--report is only supported for the original download path")
            report = derive_stage()
        elif args.download:
            report = stage(
                args.destination.expanduser().resolve(),
                args.report.expanduser().resolve(),
                workers=args.workers,
                chunk_mib=args.chunk_mib,
                retries=args.retries,
            )
        else:
            source = validate_source()
            target = validate_target_contract()
            bundle = validate_bundle()
            report = artifact_report(source=source, target=target, bundle=bundle)
    except (PreparationError, OSError, UnicodeError, ValueError, KeyError, range_fetch.DownloadError) as exc:
        print(f"prepare_gpt20b_eagle3: {type(exc).__name__}: {safe_error_detail(exc)}", file=sys.stderr)
        return 1
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
