#!/usr/bin/env python3
"""Validate and stage the existing GPT-OSS 20B GGUF for native llama.cpp.

This reads the GGUF header and tensor directory only.  It deliberately does not
hash or copy the 13.8 GB tensor payload and never starts a server.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
MODEL_STORE = Path("/external/models/run/ollama/models")
MODEL = MODEL_STORE / "blobs" / (
    "sha256-e7b273f9636059a689e3ddcab3716e4f65abe0143ac978e46673ad0e52d09efb"
)
MODEL_DIGEST = "e7b273f9636059a689e3ddcab3716e4f65abe0143ac978e46673ad0e52d09efb"
MODEL_SIZE = 13_793_422_144
TEMPLATE_DIGEST = "fa6710a93d78da62641e192361344be7a8c0a1c3737f139cf89f20ce1626b99c"
TEMPLATE_SIZE = 7_240
TEMPLATE_BLOB = MODEL_STORE / "blobs" / f"sha256-{TEMPLATE_DIGEST}"
MODEL_NAME = "gpt-oss-20b-mxfp4"
CONTEXT = 32_768
SERVER = Path("/external/storage/moe-bench/ollama-0.40.6/lib/ollama/llama-server")
SERVER_SHA256 = "234b05b2138264f8fb263c3205e85f4c290e8afe5067e280a4f6f90cdac5696b"
LLAMA_LIBRARY = SERVER.parent / "libllama.so.0.7.0"
COMMON_LIBRARY = SERVER.parent / "libllama-common.so.0.7.0"
SERVER_LIBRARY = SERVER.parent / "libllama-server-impl.so"
LLAMA_REVISION = "620d2225cc436ab0b001e3200e590e3b286ef9ab"
LLAMA_SOURCE = ROOT / "vendor" / "strata" / "third_party" / "llama.cpp"
READER_PATH = ROOT / "vendor" / "strata" / "tools" / "gguf_reader.py"
JINJA_TEMPLATE = LLAMA_SOURCE / "models" / "templates" / "openai-gpt-oss-120b.jinja"
JINJA_TEMPLATE_SHA256 = "a4c9919cbbd4acdd51ccffe22da049264b1b73e59055fa58811a99efbd7c8146"
DEFAULT_REPORT = ROOT / "results" / "gpt20b-llama-artifact.json"
DEFAULT_CONFIG = ROOT / "runtime" / "gpt20b-llama-32k.json"
LIB_DIRS = (
    Path("/external/storage/moe-bench/ollama-0.40.6/lib/ollama"),
    Path("/external/storage/moe-bench/ollama-0.40.6/lib/ollama/cuda_v13"),
)

SOURCE_REQUIREMENTS = {
    "src/llama-arch.cpp": ('{ LLM_ARCH_OPENAI_MOE,       "gpt-oss"',),
    "src/models/openai-moe.cpp": (
        "case 24: type = LLM_TYPE_20B",
        "build_moe_ffn",
    ),
    "common/parsers/gpt-oss.cpp": (
        'data.thinking_start_tag = "<|channel|>analysis<|message|>"',
        'data.thinking_end_tags  = {"<|end|>"}',
    ),
    "tools/server/server-common.cpp": (
        'json_value(body, "reasoning_budget_tokens"',
        'if (!chat_params.thinking_end_tags.empty())',
        'llama_params["reasoning_budget_end_tags"] = chat_params.thinking_end_tags',
    ),
    "common/sampling.cpp": (
        "common_reasoning_budget_init(",
        "params.reasoning_budget_tokens < 0 ? INT_MAX : params.reasoning_budget_tokens",
    ),
    "common/reasoning-budget.cpp": (
        "budget exhausted, forcing end sequence",
        "cur_p->data[i].logit = -INFINITY",
    ),
    "common/arg.cpp": (
        'params.model.hf_file = "gpt-oss-20b-mxfp4.gguf"',
    ),
    "models/templates/openai-gpt-oss-120b.jinja": (
        'set reasoning_effort = "medium"',
        '"Reasoning: " + reasoning_effort',
        '"<|start|>assistant<|channel|>analysis<|message|>"',
    ),
}

REQUIRED_TEMPLATE_MARKERS = (
    "<|start|>",
    "<|channel|>",
    "<|message|>",
    "<|end|>",
)
REQUIRED_VOCAB_TOKENS = (
    "<|start|>",
    "<|channel|>",
    "<|message|>",
    "<|end|>",
    "<|return|>",
)


class PreparationError(RuntimeError):
    pass


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


def _load_reader() -> type[Any]:
    tools_path = str(READER_PATH.parent)
    if tools_path not in sys.path:
        sys.path.insert(0, tools_path)
    from gguf_reader import GGUFFile

    return GGUFFile


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
        "architecture": "gpt-oss / LLM_ARCH_OPENAI_MOE",
        "reasoning_budget_request_field": "reasoning_budget_tokens",
        "reasoning_start_tag": "<|channel|>analysis<|message|>",
        "reasoning_end_tags": ["<|end|>"],
        "budget_behavior": "counts tokens after the analysis start tag and forces the Harmony end token at exhaustion",
        "source_support_validated": True,
    }


def validate_bundle(
    server: Path = SERVER,
    llama_library: Path = LLAMA_LIBRARY,
    common_library: Path = COMMON_LIBRARY,
    server_library: Path = SERVER_LIBRARY,
) -> dict[str, Any]:
    """Verify the small installed binaries contain Ollama and budget support."""
    try:
        server_digest = _sha256(server)
        llama_bytes = llama_library.read_bytes()
        common_bytes = common_library.read_bytes()
        server_bytes = server_library.read_bytes()
    except OSError as exc:
        raise PreparationError("installed llama-server bundle is unreadable") from exc
    _require(server_digest == SERVER_SHA256, "installed llama-server wrapper digest changed")
    for marker in (
        b"handle_gptoss",
        b"detected Ollama-format gpt-oss GGUF; applying compatibility fixes",
        b"gptoss.",
        b"gpt-oss.",
    ):
        _require(marker in llama_bytes, f"installed libllama lacks compatibility marker {marker!r}")
    for marker in (
        b"<|channel|>analysis<|message|>",
        b"<|end|><|start|>assistant<|channel|>final<|message|>",
        b"budget exhausted, forcing end sequence",
    ):
        _require(marker in common_bytes, f"installed common library lacks Harmony marker {marker!r}")
    for marker in (b"reasoning_budget_tokens", b"reasoning_budget_end_tags"):
        _require(marker in server_bytes, f"installed server library lacks request marker {marker!r}")
    return {
        "server": str(server),
        "server_sha256": server_digest,
        "libraries": {
            str(llama_library): hashlib.sha256(llama_bytes).hexdigest(),
            str(common_library): hashlib.sha256(common_bytes).hexdigest(),
            str(server_library): hashlib.sha256(server_bytes).hexdigest(),
        },
        "ollama_gptoss_metadata_translation_present": True,
        "harmony_parser_present": True,
        "reasoning_budget_request_and_sampler_present": True,
    }


def _validate_model_file(path: Path) -> os.stat_result:
    try:
        details = path.lstat()
    except OSError as exc:
        raise PreparationError(f"model blob is inaccessible: {path}") from exc
    _require(stat.S_ISREG(details.st_mode), "model blob must be a regular file")
    _require(os.access(path, os.R_OK), "model blob is unreadable")
    _require(details.st_size == MODEL_SIZE, f"model blob size is {details.st_size}, expected {MODEL_SIZE}")
    _require(path.name == f"sha256-{MODEL_DIGEST}", "model blob name is not the pinned Ollama digest")
    return details


def load_template(path: Path = TEMPLATE_BLOB) -> tuple[str, dict[str, Any]]:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise PreparationError(f"Ollama template blob is unreadable: {path}") from exc
    _require(len(payload) == TEMPLATE_SIZE, "Ollama template blob has the wrong size")
    _require(hashlib.sha256(payload).hexdigest() == TEMPLATE_DIGEST, "Ollama template blob digest mismatch")
    try:
        template = payload.decode("utf-8")
    except UnicodeError as exc:
        raise PreparationError("Ollama template blob is not UTF-8") from exc
    for marker in REQUIRED_TEMPLATE_MARKERS:
        _require(marker in template, f"Ollama template lacks Harmony marker {marker}")
    _require("analysis" in template and "final" in template, "Ollama template lacks Harmony channels")
    return template, {
        "path": str(path),
        "size": len(payload),
        "sha256": TEMPLATE_DIGEST,
        "syntax": "jinja" if "{%" in template else "ollama_go_template" if "{{" in template else "unknown",
    }


def inspect_model(
    path: Path = MODEL,
    reader_class: type[Any] | None = None,
    template: str | None = None,
    template_receipt: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    path = path.expanduser().resolve()
    details = _validate_model_file(path)
    reader = (reader_class or _load_reader())(path)
    metadata = reader.metadata
    architecture = metadata.get("general.architecture")
    _require(
        architecture in ("gpt-oss", "gptoss"),
        f"GGUF architecture is {architecture!r}, expected 'gpt-oss' or Ollama 'gptoss'",
    )
    metadata_prefix = architecture
    block_count = metadata.get(f"{metadata_prefix}.block_count")
    _require(block_count == 24, "GGUF is not the 24-layer GPT-OSS 20B model")
    context = metadata.get(f"{metadata_prefix}.context_length")
    _require(isinstance(context, int) and context >= CONTEXT, "GGUF context is below the staged context")

    embedded_template = metadata.get("tokenizer.chat_template")
    if embedded_template is None:
        _require(isinstance(template, str) and template, "GGUF has no embedded chat template and no external template was supplied")
    else:
        _require(isinstance(embedded_template, str) and embedded_template, "GGUF embedded chat template is malformed")
        template = embedded_template
        template_receipt = {"path": None, "source": "GGUF tokenizer.chat_template"}
    assert isinstance(template, str)
    for marker in REQUIRED_TEMPLATE_MARKERS:
        _require(marker in template, f"embedded template lacks Harmony marker {marker}")
    _require("analysis" in template and "final" in template, "embedded template lacks Harmony channels")

    tokens = metadata.get("tokenizer.ggml.tokens")
    _require(isinstance(tokens, list), "GGUF has no tokenizer token array")
    token_positions: dict[str, int] = {}
    for token in REQUIRED_VOCAB_TOKENS:
        try:
            token_positions[token] = tokens.index(token)
        except ValueError as exc:
            raise PreparationError(f"GGUF tokenizer lacks Harmony token {token}") from exc

    tensor_types: Counter[str] = Counter()
    ranges: list[tuple[int, int, str]] = []
    for tensor in reader.tensors:
        byte_count = tensor.expected_bytes()
        _require(byte_count is not None and byte_count > 0, f"unsupported tensor layout: {tensor.name}")
        _require(tensor.offset >= 0 and tensor.offset % reader.alignment == 0, f"unaligned tensor: {tensor.name}")
        start = reader.data_start + tensor.offset
        end = start + byte_count
        _require(reader.data_start <= start < end <= details.st_size, f"tensor exceeds blob bounds: {tensor.name}")
        ranges.append((start, end, tensor.name))
        tensor_types[tensor.type_name] += 1
    _require(tensor_types["MXFP4"] > 0, "GGUF has no MXFP4 tensors")
    ranges.sort()
    for previous, current in zip(ranges, ranges[1:]):
        _require(previous[1] <= current[0], f"tensor payload overlap: {previous[2]}, {current[2]}")

    return {
        "path": str(path),
        "size": details.st_size,
        "sha256": MODEL_DIGEST,
        "verification_method": "pinned Ollama content-addressed blob name and exact size; tensor payload not rehashed in this preparation",
        "fresh_full_sha256_verified": False,
        "stat": {
            "device": details.st_dev,
            "inode": details.st_ino,
            "mode": details.st_mode,
            "links": details.st_nlink,
            "mtime_ns": details.st_mtime_ns,
            "ctime_ns": details.st_ctime_ns,
        },
        "gguf": {
            "version": reader.version,
            "architecture": metadata["general.architecture"],
            "name": metadata.get("general.name"),
            "context_length": context,
            "block_count": block_count,
            "expert_count": metadata.get(f"{metadata_prefix}.expert_count"),
            "expert_used_count": metadata.get(f"{metadata_prefix}.expert_used_count"),
            "requires_ollama_metadata_translation": architecture == "gptoss",
            "tensor_count": len(reader.tensors),
            "tensor_type_counts": dict(sorted(tensor_types.items())),
            "all_tensor_payload_bounds_validated": True,
            "all_offsets_aligned": True,
            "no_payload_overlaps": True,
        },
        "harmony": {
            "template_bytes": len(template.encode("utf-8")),
            "template_sha256": hashlib.sha256(template.encode("utf-8")).hexdigest(),
            "template_source": dict(template_receipt or {}),
            "required_template_markers": list(REQUIRED_TEMPLATE_MARKERS),
            "token_ids": token_positions,
            "specialized_parser_detectable": "<|channel|>" in template,
            "thinking_end_tags_nonempty": True,
        },
    }


def artifact_report(
    model: Mapping[str, Any],
    source: Mapping[str, Any],
    bundle: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    fully_verified = model.get("fresh_full_sha256_verified") is True
    return {
        "schema_version": 1,
        "checked_at": utc_now(),
        "state": "inactive_source_and_header_validated",
        "model": dict(model),
        "llama_cpp": {
            "server": str(SERVER),
            "server_sha256": SERVER_SHA256,
            "upstream_revision": LLAMA_REVISION,
            "source_evidence": dict(source),
            "installed_bundle_evidence": dict(bundle or {}),
            "ollama_gptoss_compatibility_required": True,
        },
        "shards": [
            {
                "name": MODEL.name,
                "path": str(MODEL),
                "size": MODEL_SIZE,
                "sha256": MODEL_DIGEST,
                "verified": fully_verified,
                "verification_method": model.get("verification_method"),
                "fresh_full_sha256_verified": fully_verified,
            }
        ],
        "reasoning_budget": {
            "supported_by_source": True,
            "request_field": "reasoning_budget_tokens",
            "template_parser": "GPT-OSS Harmony specialized parser",
            "start_tag": "<|channel|>analysis<|message|>",
            "end_tags": ["<|end|>"],
            "enforcement_runtime_validated": False,
            "original_ollama_template": model.get("harmony", {}).get("template_source"),
            "direct_server_template": {
                "path": str(JINJA_TEMPLATE),
                "sha256": JINJA_TEMPLATE_SHA256,
                "reasoning_effort_variable": True,
            },
        },
        "verdict": {
            "artifact_complete": fully_verified,
            "source_compatible": True,
            "header_and_tensor_directory_valid": True,
            "runtime_load_validated": False,
            "reasoning_budget_runtime_validated": False,
            "staged_for_guarded_trial": fully_verified,
        },
    }


def config_document(report_path: Path) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "server": str(SERVER),
        "model": str(MODEL),
        "model_name": MODEL_NAME,
        "context": CONTEXT,
        "host": "127.0.0.1",
        "port": 18081,
        "artifact_manifest": os.path.relpath(report_path, DEFAULT_CONFIG.parent),
        "lib_dirs": [str(path) for path in LIB_DIRS],
        "args": [
            "--gpu-layers",
            "auto",
            "--parallel",
            "1",
            "--jinja",
            "--chat-template-file",
            str(JINJA_TEMPLATE),
            "--no-webui",
            "--no-slots",
        ],
        "shards": [
            {
                "name": MODEL.name,
                "size": MODEL_SIZE,
                "sha256": MODEL_DIGEST,
            }
        ],
    }


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


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=MODEL)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument(
        "--verify-full-sha256",
        action="store_true",
        help="read and verify the full weight payload before publishing the runnable receipt",
    )
    args = parser.parse_args(argv)
    try:
        source = validate_source()
        bundle = validate_bundle()
        template, template_receipt = load_template()
        model = inspect_model(args.model, template=template, template_receipt=template_receipt)
        if args.verify_full_sha256:
            actual_digest = _sha256(args.model)
            _require(actual_digest == MODEL_DIGEST, "full model SHA-256 mismatch")
            model["fresh_full_sha256_verified"] = True
            model["verification_method"] = "fresh full-file SHA-256 plus exact size and GGUF directory validation"
        elif not args.check_only:
            raise PreparationError("refusing to publish runnable config without --verify-full-sha256")
        report_path = args.report.expanduser().resolve()
        config_path = args.config.expanduser().resolve()
        report = artifact_report(model, source, bundle)
        config = config_document(report_path)
        if not args.check_only:
            atomic_write_json(report_path, report)
            atomic_write_json(config_path, config)
    except (PreparationError, OSError, UnicodeError, ValueError, KeyError) as exc:
        print(f"prepare_gpt20b_llama: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"report": report, "config": config}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
