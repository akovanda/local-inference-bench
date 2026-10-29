#!/usr/bin/env python3
"""Download and verify the pinned Gemma 4 12B MTP assistant GGUF.

This helper only downloads, hashes, and reads GGUF metadata.  It never starts
llama-server, performs inference, or modifies the already-running StyleTune
target.  Publication of the model file is delegated to ``range_fetch``, whose
final hard-link operation refuses to overwrite an existing path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
import uuid
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__:
    from . import range_fetch
else:
    import range_fetch


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = Path("/external/storage/moe-bench/models/gemma4-12b-mtp")
FILENAME = "mtp-gemma-4-12B-it-Q8_0.gguf"
DESTINATION = MODEL_DIR / FILENAME
REPORT = ROOT / "results" / "gemma4-style12-mtp-artifact.json"

REPO = "ggml-org/gemma-4-12B-it-GGUF"
REVISION = "a29b2a6b584a5bdcb672ccdcda99ea9faeacc51a"
FILE_SIZE = 465_109_152
FILE_SHA256 = "16c90eb9f2b2891cc138f3d2b3bf11e23b2ced2aeab9a9d39d90fe446f2f0610"
FILE_XET_HASH = "8ecdf0e0c0a8120f0b123337234663abd9951e6b85378a9cdc321ed3cec5e5db"
URL = f"https://huggingface.co/{REPO}/resolve/{REVISION}/{FILENAME}"

TARGET = Path(
    "/external/storage/moe-bench/models/gemma4-style12-q5_k_m/"
    "Gemma-4-12B-StyleTune.Q5_K_M.gguf"
)
TARGET_SIZE = 9_239_328_384
TARGET_SHA256 = "14d78cfccb5f052a9c4bfcdceac30fae570a09dea4453c7d929fa57f703ff350"
TARGET_RECEIPT = ROOT / "results" / "gemma4-style12-q5_k_m-artifact.json"

LLAMA_SOURCE = ROOT / "vendor" / "strata" / "third_party" / "llama.cpp"
LLAMA_REVISION = "620d2225cc436ab0b001e3200e590e3b286ef9ab"
SERVER = Path("/external/storage/moe-bench/ollama-0.40.6/lib/ollama/llama-server")
READER_PATH = ROOT / "vendor" / "strata" / "tools" / "gguf_reader.py"

REQUESTED_CONTEXT = 73_728
REQUESTED_OUTPUT = 65_536
OBSERVED_FREE_VRAM_MIB = 4_821

UPSTREAM = {
    "converter": {
        "repository": REPO,
        "revision": REVISION,
        "repository_url": f"https://huggingface.co/{REPO}/tree/{REVISION}",
        "file_url": URL,
        "model_card_sha256": "54669a4c5924e420a9a780d7b3ef5a2d86d5f703a0161b1f3d12b26beec095f7",
        "conversion_log_sha256": "6f30d61fe6a76a08fd19e1a53f87e49056a3652143758cad42dd31414ec60aa5",
        "provenance": (
            "ggml-org model card names google/gemma-4-12B-it-assistant as a source "
            "and says the repository is automatically converted by ggml-org/convert"
        ),
    },
    "assistant_config": {
        "repository": "google/gemma-4-12B-it-assistant",
        "revision": "46d4c6f13f0ac0ad827b915669b8df9b81c64c51",
        "url": (
            "https://huggingface.co/google/gemma-4-12B-it-assistant/resolve/"
            "46d4c6f13f0ac0ad827b915669b8df9b81c64c51/config.json"
        ),
        "sha256": "b6f19209588fcefe41f65b193fad6148446253c470d36e29441ecc5158a54e6d",
        "model_type": "gemma4_unified_assistant",
        "hidden_size": 1_024,
        "backbone_hidden_size": 3_840,
        "vocab_size": 262_144,
        "layers": 4,
        "tie_word_embeddings": True,
    },
    "base_config": {
        "repository": "google/gemma-4-12B-it",
        "revision": "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7",
        "url": (
            "https://huggingface.co/google/gemma-4-12B-it/resolve/"
            "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7/config.json"
        ),
        "sha256": "478c46e8d2c52d5c2d85bf67e3b3e8c90e7c9d91086cee27e3c267907e936bd9",
        "hidden_size": 3_840,
        "vocab_size": 262_144,
        "layers": 48,
        "tie_word_embeddings": True,
    },
    "style_tune": {
        "repository": "Gryphe/Gemma-4-12B-StyleTune",
        "revision": "537724e2c41a11d0ef3142a230da3aa2c10ebd32",
        "config_url": (
            "https://huggingface.co/Gryphe/Gemma-4-12B-StyleTune/resolve/"
            "537724e2c41a11d0ef3142a230da3aa2c10ebd32/config.json"
        ),
        "config_sha256": "7bf4bceebbe3d9a77ba3a38e19236630a574dc3851bd8a0ecda03fd339dc55f7",
        "model_card_url": (
            "https://huggingface.co/Gryphe/Gemma-4-12B-StyleTune/resolve/"
            "537724e2c41a11d0ef3142a230da3aa2c10ebd32/README.md"
        ),
        "model_card_sha256": "d2dbd97a7b162485941986e5b8aac7643cdf8ae67b9b792c8626b6753f833cc4",
        "hidden_size": 3_840,
        "vocab_size": 262_144,
        "layers": 48,
        "tie_word_embeddings": False,
        "publisher_claim": "only the lm_head output projection was trained",
    },
    "mtp_documentation": "https://ai.google.dev/gemma/docs/mtp/mtp",
}

SOURCE_REQUIREMENTS = {
    "common/arg.cpp": (
        "COMMON_SPECULATIVE_TYPE_DRAFT_MTP",
        "opts.download_mtp    = spec_type_draft_mtp",
        '"--spec-type"',
    ),
    "common/speculative.cpp": (
        '{"draft-mtp",     COMMON_SPECULATIVE_TYPE_DRAFT_MTP}',
        "common_speculative_impl_draft_mtp",
        "MTP input row width must match the target h_nextn width",
        "is_mem_shared = llama_get_ctx_other(ctx_dft) == ctx_tgt",
        "with shared memory (e.g. Gemma4 assistants)",
    ),
    "src/llama-context.cpp": (
        "Gemma4Assistant requires ctx_other to be set",
        "model.arch == LLM_ARCH_GEMMA4_ASSISTANT",
    ),
    "src/llama-model.cpp": (
        "return new llama_model_gemma4_assistant(params)",
        "arch == LLM_ARCH_GEMMA4_ASSISTANT",
        "llama_memory_t mem_other = llama_get_memory(cparams.ctx_other)",
        "res = new llama_kv_cache_iswa(",
    ),
    "src/llama-vocab.cpp": (
        "bool add_eos                    = false",
        "if (ml.get_key(LLM_KV_TOKENIZER_ADD_EOS, temp, false))",
    ),
}

TOKENIZER_KEYS = (
    "tokenizer.ggml.model",
    "tokenizer.ggml.tokens",
    "tokenizer.ggml.scores",
    "tokenizer.ggml.token_type",
    "tokenizer.ggml.merges",
    "tokenizer.ggml.bos_token_id",
    "tokenizer.ggml.eos_token_id",
    "tokenizer.ggml.unknown_token_id",
    "tokenizer.ggml.padding_token_id",
    "tokenizer.ggml.mask_token_id",
    "tokenizer.ggml.add_bos_token",
    "tokenizer.ggml.add_eos_token",
    "tokenizer.ggml.add_space_prefix",
)

OPTIONAL_TOKENIZER_KEYS = {"tokenizer.ggml.add_eos_token"}


class PreparationError(RuntimeError):
    pass


@dataclass(frozen=True)
class ModelInspection:
    report: dict[str, Any]
    tokenizer: dict[str, Any]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PreparationError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _identity(details: os.stat_result) -> dict[str, int]:
    return {
        "device": details.st_dev,
        "inode": details.st_ino,
        "size": details.st_size,
        "mtime_ns": details.st_mtime_ns,
        "ctime_ns": details.st_ctime_ns,
    }


def _regular_file(path: Path, size: int, description: str) -> os.stat_result:
    try:
        details = path.lstat()
    except OSError as exc:
        raise PreparationError(f"{description} is inaccessible: {path}") from exc
    _require(stat.S_ISREG(details.st_mode) and not path.is_symlink(), f"{description} must be a regular non-symlink file")
    _require(details.st_size == size, f"{description} size is {details.st_size}, expected {size}")
    return details


def verify_pinned_file(
    path: Path = DESTINATION,
    *,
    expected_size: int = FILE_SIZE,
    expected_sha256: str = FILE_SHA256,
) -> dict[str, Any]:
    before = _regular_file(path, expected_size, "MTP artifact")
    actual = sha256_file(path)
    after = _regular_file(path, expected_size, "MTP artifact")
    _require(_identity(before) == _identity(after), "MTP artifact changed while SHA-256 was computed")
    _require(actual == expected_sha256, f"MTP artifact SHA-256 is {actual}, expected {expected_sha256}")
    return {
        "path": str(path),
        "size": expected_size,
        "sha256": actual,
        "full_sha256_verified": True,
        "file_identity_after_verification": _identity(after),
    }


def _load_reader() -> type[Any]:
    tools_path = str(READER_PATH.parent)
    if tools_path not in sys.path:
        sys.path.insert(0, tools_path)
    from gguf_reader import GGUFFile

    return GGUFFile


def _compact_digest(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _tensor_map(reader: Any, file_size: int) -> tuple[dict[str, Any], Counter[str]]:
    tensors: dict[str, Any] = {}
    ranges: list[tuple[int, int, str]] = []
    types: Counter[str] = Counter()
    for tensor in reader.tensors:
        _require(tensor.name not in tensors, f"duplicate tensor name: {tensor.name}")
        byte_count = tensor.expected_bytes()
        _require(byte_count is not None and byte_count > 0, f"unsupported tensor layout: {tensor.name}")
        _require(tensor.offset >= 0 and tensor.offset % reader.alignment == 0, f"unaligned tensor: {tensor.name}")
        start = reader.data_start + tensor.offset
        end = start + byte_count
        _require(reader.data_start <= start < end <= file_size, f"tensor exceeds file bounds: {tensor.name}")
        ranges.append((start, end, tensor.name))
        tensors[tensor.name] = tensor
        types[tensor.type_name] += 1
    ranges.sort()
    for previous, current in zip(ranges, ranges[1:]):
        _require(previous[1] <= current[0], f"tensor payloads overlap: {previous[2]} and {current[2]}")
    return tensors, types


def _tensor_record(tensor: Any) -> dict[str, Any]:
    return {
        "shape": list(tensor.shape),
        "type": tensor.type_name,
        "offset": tensor.offset,
        "bytes": tensor.expected_bytes(),
    }


def inspect_model(
    path: Path,
    role: str,
    *,
    expected_size: int,
    reader_class: type[Any] | None = None,
) -> ModelInspection:
    details = _regular_file(path, expected_size, f"{role} GGUF")
    reader = (reader_class or _load_reader())(path)
    metadata = reader.metadata
    tensors, types = _tensor_map(reader, details.st_size)
    tokenizer = {key: metadata.get(key) for key in TOKENIZER_KEYS}
    missing_tokenizer = [
        key for key, value in tokenizer.items()
        if value is None and key not in OPTIONAL_TOKENIZER_KEYS
    ]
    _require(not missing_tokenizer, f"{role} tokenizer metadata is missing: {missing_tokenizer}")

    if role == "assistant":
        prefix = "gemma4-assistant"
        expected_metadata = {
            "general.architecture": "gemma4-assistant",
            f"{prefix}.block_count": 4,
            f"{prefix}.context_length": 262_144,
            f"{prefix}.embedding_length": 1_024,
            f"{prefix}.feed_forward_length": 8_192,
            f"{prefix}.attention.head_count": 16,
            f"{prefix}.attention.head_count_kv": [8, 8, 8, 1],
            f"{prefix}.attention.shared_kv_layers": 4,
            f"{prefix}.embedding_length_out": 3_840,
            f"{prefix}.nextn_predict_layers": 4,
        }
        expected_tensors = {
            "token_embd.weight": [1_024, 262_144],
            "nextn.pre_projection.weight": [7_680, 1_024],
            "nextn.post_projection.weight": [1_024, 3_840],
        }
        _require(len(reader.tensors) == 49, "assistant GGUF does not have the pinned 49-tensor layout")
        _require(types["Q8_0"] > 0, "assistant GGUF has no Q8_0 tensor payloads")
        _require("output.weight" not in tensors, "assistant GGUF unexpectedly has an untied output tensor")
    elif role == "target":
        prefix = "gemma4"
        expected_metadata = {
            "general.architecture": "gemma4",
            f"{prefix}.block_count": 48,
            f"{prefix}.context_length": 262_144,
            f"{prefix}.embedding_length": 3_840,
            f"{prefix}.feed_forward_length": 15_360,
            f"{prefix}.attention.head_count": 16,
        }
        expected_tensors = {
            "token_embd.weight": [3_840, 262_144],
            "output.weight": [3_840, 262_144],
        }
    else:
        raise PreparationError(f"unsupported inspection role: {role}")

    for key, expected in expected_metadata.items():
        _require(metadata.get(key) == expected, f"{role} metadata {key} is {metadata.get(key)!r}, expected {expected!r}")
    selected_tensors: dict[str, Any] = {}
    for name, shape in expected_tensors.items():
        _require(name in tensors, f"{role} GGUF is missing tensor {name}")
        _require(list(tensors[name].shape) == shape, f"{role} tensor {name} has shape {list(tensors[name].shape)}, expected {shape}")
        selected_tensors[name] = _tensor_record(tensors[name])

    selected_metadata = {key: metadata[key] for key in expected_metadata}
    return ModelInspection(
        report={
            "path": str(path),
            "size": details.st_size,
            "gguf_version": reader.version,
            "alignment": reader.alignment,
            "data_start": reader.data_start,
            "tensor_count": len(reader.tensors),
            "tensor_type_counts": dict(sorted(types.items())),
            "all_tensor_payload_bounds_validated": True,
            "no_tensor_payload_overlaps": True,
            "metadata": selected_metadata,
            "tensors": selected_tensors,
            "tokenizer_metadata": {
                "keys_compared": list(TOKENIZER_KEYS),
                "digests": {key: _compact_digest(value) for key, value in tokenizer.items()},
            },
        },
        tokenizer=tokenizer,
    )


def validate_target_receipt(path: Path = TARGET_RECEIPT) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PreparationError(f"target verification receipt is unreadable: {path}") from exc
    shards = value.get("shards") if isinstance(value, dict) else None
    _require(isinstance(shards, list) and len(shards) == 1, "target receipt has an invalid shard list")
    shard = shards[0]
    _require(
        shard.get("path") == str(TARGET)
        and shard.get("size") == TARGET_SIZE
        and shard.get("sha256") == TARGET_SHA256
        and shard.get("full_sha256_verified") is True,
        "target receipt does not bind the exact verified StyleTune GGUF",
    )
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "target_path": str(TARGET),
        "target_size": TARGET_SIZE,
        "target_sha256": TARGET_SHA256,
        "full_target_sha256_reused": True,
        "target_payload_rehashed_by_this_helper": False,
    }


def compare_models(target: ModelInspection, assistant: ModelInspection) -> dict[str, Any]:
    target_metadata = target.report["metadata"]
    assistant_metadata = assistant.report["metadata"]
    target_tokens = target.tokenizer["tokenizer.ggml.tokens"]
    assistant_tokens = assistant.tokenizer["tokenizer.ggml.tokens"]
    _require(
        target.tokenizer["tokenizer.ggml.model"] == assistant.tokenizer["tokenizer.ggml.model"],
        "assistant tokenizer model type does not match the StyleTune target",
    )
    _require(
        abs(len(target_tokens) - len(assistant_tokens)) <= 128,
        "assistant vocabulary size differs from the target by more than 128 tokens",
    )
    _require(
        target_tokens[5:] == assistant_tokens[5:],
        "assistant token text differs from the StyleTune target at or after token ID 5",
    )
    target_add_bos = bool(target.tokenizer["tokenizer.ggml.add_bos_token"])
    assistant_add_bos = bool(assistant.tokenizer["tokenizer.ggml.add_bos_token"])
    _require(target_add_bos == assistant_add_bos, "assistant add_bos behavior differs from target")
    if target_add_bos:
        _require(
            target.tokenizer["tokenizer.ggml.bos_token_id"]
            == assistant.tokenizer["tokenizer.ggml.bos_token_id"],
            "assistant BOS token differs from target",
        )
    target_add_eos = bool(target.tokenizer.get("tokenizer.ggml.add_eos_token"))
    assistant_add_eos = bool(assistant.tokenizer.get("tokenizer.ggml.add_eos_token"))
    _require(target_add_eos == assistant_add_eos, "assistant add_eos behavior differs from target")
    if target_add_eos:
        _require(
            target.tokenizer["tokenizer.ggml.eos_token_id"]
            == assistant.tokenizer["tokenizer.ggml.eos_token_id"],
            "assistant EOS token differs from target while add_eos is enabled",
        )
    auxiliary_matches = {
        key: target.tokenizer[key] == assistant.tokenizer[key]
        for key in (
            "tokenizer.ggml.scores",
            "tokenizer.ggml.token_type",
            "tokenizer.ggml.merges",
        )
    }
    target_hidden = target_metadata["gemma4.embedding_length"]
    assistant_output = assistant_metadata["gemma4-assistant.embedding_length_out"]
    target_vocab = target.report["tensors"]["token_embd.weight"]["shape"][1]
    assistant_vocab = assistant.report["tensors"]["token_embd.weight"]["shape"][1]
    _require(assistant_output == target_hidden, "assistant output width does not match target hidden width")
    _require(assistant_vocab == target_vocab, "assistant vocabulary width does not match target vocabulary")
    _require(
        target_metadata["gemma4.context_length"] >= REQUESTED_CONTEXT
        and assistant_metadata["gemma4-assistant.context_length"] >= REQUESTED_CONTEXT,
        "target or assistant training context is below the requested context",
    )
    target_token_shape = target.report["tensors"]["token_embd.weight"]["shape"]
    target_output_shape = target.report["tensors"]["output.weight"]["shape"]
    _require(target_token_shape == target_output_shape, "StyleTune input/output embedding shapes differ")
    return {
        "metadata_dimension_compatible": True,
        "assistant_output_width": assistant_output,
        "target_hidden_width": target_hidden,
        "vocabulary_size": target_vocab,
        "runtime_vocabulary_compatibility_checks_pass": True,
        "token_text_id_5_plus_exact_match": True,
        "auxiliary_tokenizer_metadata_matches": auxiliary_matches,
        "auxiliary_tokenizer_metadata_required_by_pinned_runtime_gate": False,
        "tokenizer_metadata_exact_match": target.tokenizer == assistant.tokenizer,
        "bos": {
            "add_bos_target": target_add_bos,
            "add_bos_assistant": assistant_add_bos,
            "target_id": target.tokenizer["tokenizer.ggml.bos_token_id"],
            "assistant_id": assistant.tokenizer["tokenizer.ggml.bos_token_id"],
        },
        "eos": {
            "add_eos_target": target_add_eos,
            "add_eos_assistant": assistant_add_eos,
            "target_id": target.tokenizer["tokenizer.ggml.eos_token_id"],
            "assistant_id": assistant.tokenizer["tokenizer.ggml.eos_token_id"],
            "id_match_required_by_pinned_runtime": target_add_eos,
        },
        "requested_context": REQUESTED_CONTEXT,
        "requested_total_output": REQUESTED_OUTPUT,
        "context_within_both_training_limits": True,
        "target_input_embedding_shape": target_token_shape,
        "target_output_embedding_shape": target_output_shape,
        "target_has_distinct_input_and_output_tensors": True,
        "tie_word_embeddings": {
            "google_base_config": True,
            "style_tune_config": False,
            "google_assistant_config": True,
        },
        "style_tune_scope": {
            "publisher_claim": "only lm_head changed",
            "claim_source": UPSTREAM["style_tune"]["model_card_url"],
            "weights_independently_compared_by_this_helper": False,
        },
        "interpretation": (
            "architecture, hidden width, vocabulary, pinned-runtime vocabulary, and context checks pass; "
            "the StyleTune lm_head difference can still change speculative acceptance"
        ),
    }


def validate_runtime_source(source_root: Path = LLAMA_SOURCE) -> dict[str, Any]:
    files: dict[str, Any] = {}
    for relative, fragments in SOURCE_REQUIREMENTS.items():
        path = source_root / relative
        try:
            payload = path.read_bytes()
            text = payload.decode("utf-8")
        except (OSError, UnicodeError) as exc:
            raise PreparationError(f"pinned llama.cpp source is unreadable: {path}") from exc
        for fragment in fragments:
            _require(fragment in text, f"pinned runtime source lacks required fragment: {relative}: {fragment}")
        files[relative] = {
            "sha256": hashlib.sha256(payload).hexdigest(),
            "required_fragments_present": True,
        }
    return {
        "revision": LLAMA_REVISION,
        "server": str(SERVER),
        "source_snapshot": str(source_root),
        "files": files,
        "draft_mtp_present": True,
        "gemma4_assistant_shared_kv_path_present": True,
        "assistant_target_output_width_assertion_present": True,
        "runtime_binary_executed": False,
    }


def source_artifact_record() -> dict[str, Any]:
    return {
        "repository": REPO,
        "revision": REVISION,
        "filename": FILENAME,
        "url": URL,
        "size": FILE_SIZE,
        "lfs_sha256": FILE_SHA256,
        "xet_hash": FILE_XET_HASH,
        "quantization": "Q8_0",
        "role": "Gemma 4 12B assistant/MTP head; not a standalone E2B draft",
        "pin_source": f"https://huggingface.co/api/models/{REPO}/tree/{REVISION}?recursive=true&expand=true",
    }


def artifact_report(
    *,
    download: Mapping[str, Any] | None = None,
    assistant: Mapping[str, Any] | None = None,
    target: Mapping[str, Any] | None = None,
    target_receipt: Mapping[str, Any] | None = None,
    compatibility: Mapping[str, Any] | None = None,
    runtime_source: Mapping[str, Any] | None = None,
    error: Exception | None = None,
) -> dict[str, Any]:
    complete = bool(
        download
        and download.get("full_sha256_verified")
        and assistant
        and target
        and target_receipt
        and compatibility
        and compatibility.get("metadata_dimension_compatible")
        and runtime_source
        and runtime_source.get("draft_mtp_present")
        and error is None
    )
    state = "inactive_verified" if complete else "inactive_preparation_failed" if error else "inactive_download_pending"
    result: dict[str, Any] = {
        "schema_version": 1,
        "checked_at": utc_now(),
        "state": state,
        "source_artifact": source_artifact_record(),
        "upstream_provenance": UPSTREAM,
        "download_verification": dict(download) if download else None,
        "assistant_metadata": dict(assistant) if assistant else None,
        "target": {
            "path": str(TARGET),
            "size": TARGET_SIZE,
            "sha256": TARGET_SHA256,
            "verification_receipt": dict(target_receipt) if target_receipt else None,
            "metadata": dict(target) if target else None,
        },
        "compatibility": dict(compatibility) if compatibility else None,
        "runtime_source": dict(runtime_source) if runtime_source else None,
        "resource_basis": {
            "assistant_file_size_bytes": FILE_SIZE,
            "assistant_quantization": "Q8_0",
            "sampled_free_vram_mib_before_preparation": OBSERVED_FREE_VRAM_MIB,
            "sample_is_a_guaranteed_budget": False,
            "gpu_fit_validated": False,
        },
        "limitations": {
            "gpu_inference_validated": False,
            "runtime_load_validated": False,
            "vram_fit_validated": False,
            "speculative_acceptance_validated": False,
            "performance_validated": False,
            "speedup_measured": False,
            "target_style_quality_with_mtp_validated": False,
            "note": (
                "This is an offline provenance, hash, GGUF-layout, and dimensional compatibility artifact. "
                "The StyleTune lm_head is untied and differs from the base, so inference must establish "
                "acceptance, output quality, memory fit, and performance."
            ),
        },
        "verdict": {
            "artifact_complete": complete,
            "full_sha256_verified": bool(download and download.get("full_sha256_verified")),
            "gguf_metadata_validated": bool(assistant),
            "target_dimension_compatible": bool(compatibility and compatibility.get("metadata_dimension_compatible")),
            "runtime_source_support_validated": bool(runtime_source and runtime_source.get("draft_mtp_present")),
            "gpu_inference_validated": False,
            "runtime_load_validated": False,
            "speedup_measured": False,
        },
    }
    if error is not None:
        result["error"] = {"type": type(error).__name__, "message": str(error)}
    return result


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
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
                raise OSError("short report write")
            written += count
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        os.replace(temporary, path)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def stage(
    destination: Path = DESTINATION,
    report_path: Path = REPORT,
    *,
    workers: int = 2,
    chunk_mib: int = 16,
    retries: int = 3,
    request_timeout: float = 60.0,
    no_download: bool = False,
) -> dict[str, Any]:
    download: dict[str, Any] | None = None
    assistant_report: dict[str, Any] | None = None
    target_report: dict[str, Any] | None = None
    target_receipt: dict[str, Any] | None = None
    compatibility: dict[str, Any] | None = None
    runtime_source: dict[str, Any] | None = None
    try:
        runtime_source = validate_runtime_source()
        target_receipt = validate_target_receipt()
        target_inspection = inspect_model(TARGET, "target", expected_size=TARGET_SIZE)
        target_report = target_inspection.report

        if os.path.lexists(destination):
            download = verify_pinned_file(
                destination,
                expected_size=FILE_SIZE,
                expected_sha256=FILE_SHA256,
            )
            download["downloaded_by_this_run"] = False
        elif no_download:
            report = artifact_report(
                target=target_report,
                target_receipt=target_receipt,
                runtime_source=runtime_source,
            )
            atomic_write_json(report_path, report)
            return report
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            range_fetch.download_file(
                URL,
                destination,
                FILE_SIZE,
                FILE_SHA256,
                workers=workers,
                chunk_size=chunk_mib * 1024 * 1024,
                retries=retries,
                request_timeout=request_timeout,
            )
            download = verify_pinned_file(
                destination,
                expected_size=FILE_SIZE,
                expected_sha256=FILE_SHA256,
            )
            download["downloaded_by_this_run"] = True

        assistant_inspection = inspect_model(destination, "assistant", expected_size=FILE_SIZE)
        assistant_report = assistant_inspection.report
        compatibility = compare_models(target_inspection, assistant_inspection)
        report = artifact_report(
            download=download,
            assistant=assistant_report,
            target=target_report,
            target_receipt=target_receipt,
            compatibility=compatibility,
            runtime_source=runtime_source,
        )
        atomic_write_json(report_path, report)
        return report
    except Exception as exc:
        report = artifact_report(
            download=download,
            assistant=assistant_report,
            target=target_report,
            target_receipt=target_receipt,
            compatibility=compatibility,
            runtime_source=runtime_source,
            error=exc,
        )
        atomic_write_json(report_path, report)
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=DESTINATION)
    parser.add_argument("--report", type=Path, default=REPORT)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--chunk-mib", type=int, default=16)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--request-timeout", type=float, default=60.0)
    parser.add_argument("--no-download", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = stage(
            args.destination,
            args.report,
            workers=args.workers,
            chunk_mib=args.chunk_mib,
            retries=args.retries,
            request_timeout=args.request_timeout,
            no_download=args.no_download,
        )
    except Exception as exc:
        print(f"preparation failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report["verdict"], sort_keys=True))
    return 0 if report["verdict"]["artifact_complete"] or args.no_download else 1


if __name__ == "__main__":
    raise SystemExit(main())
