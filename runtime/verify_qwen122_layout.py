#!/usr/bin/env python3
"""Validate the pinned Qwen122 split GGUF directories without loading a model."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

if __package__:
    from . import prepare_qwen122 as prep
else:
    import prepare_qwen122 as prep


ROOT = Path(__file__).resolve().parents[1]
READER_PATH = ROOT / "vendor" / "strata" / "tools" / "gguf_reader.py"
DEFAULT_REPORT = ROOT / "results" / "qwen122-layout.json"


class LayoutError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise LayoutError(message)


def _load_reader() -> type[Any]:
    tools_path = str(READER_PATH.parent)
    if tools_path not in sys.path:
        sys.path.insert(0, tools_path)
    from gguf_reader import GGUFFile

    return GGUFFile


def verify_layout() -> dict[str, Any]:
    gguf_file = _load_reader()
    artifact_bytes = prep.REPORT.read_bytes()
    artifact = json.loads(artifact_bytes.decode("utf-8"))
    rows = artifact.get("shards") if isinstance(artifact, dict) else None
    _require(
        isinstance(artifact.get("verdict"), dict)
        and artifact["verdict"].get("artifact_complete") is True,
        "artifact receipt is not complete",
    )
    _require(
        isinstance(rows, list)
        and len(rows) == len(prep.SHARDS)
        and all(isinstance(row, dict) and row.get("verified") is True for row in rows),
        "artifact receipt does not verify every shard",
    )

    names: set[str] = set()
    type_counts: Counter[str] = Counter()
    shard_receipts: list[dict[str, Any]] = []
    for index, (name, expected_size, expected_sha256) in enumerate(prep.SHARDS):
        path = prep.DESTINATION / name
        details = path.stat()
        _require(details.st_size == expected_size, f"wrong size for {name}")
        row = rows[index]
        _require(
            row.get("name") == name
            and row.get("size") == expected_size
            and row.get("sha256") == expected_sha256,
            f"artifact receipt row mismatch for {name}",
        )
        gguf = gguf_file(path)
        _require(gguf.metadata.get("split.no") == index, f"wrong split.no for {name}")
        _require(gguf.metadata.get("split.count") == 3, f"wrong split.count for {name}")
        _require(
            gguf.metadata.get("split.tensors.count") == 879,
            f"wrong split tensor count for {name}",
        )
        if index == 0:
            _require(
                gguf.metadata.get("general.architecture") == "qwen35moe",
                f"wrong architecture for {name}",
            )
        else:
            _require(
                gguf.metadata.get("general.architecture") in (None, "qwen35moe"),
                f"conflicting architecture for {name}",
            )

        ranges: list[tuple[int, int, str]] = []
        for tensor in gguf.tensors:
            _require(tensor.name not in names, f"duplicate tensor name {tensor.name}")
            names.add(tensor.name)
            _require(
                bool(tensor.shape) and all(isinstance(size, int) and size > 0 for size in tensor.shape),
                f"invalid shape for {tensor.name}",
            )
            byte_count = tensor.expected_bytes()
            _require(
                byte_count is not None and byte_count > 0,
                f"unsupported or invalid quant geometry for {tensor.name}",
            )
            _require(
                tensor.offset >= 0 and tensor.offset % gguf.alignment == 0,
                f"unaligned tensor offset for {tensor.name}",
            )
            start = gguf.data_start + tensor.offset
            end = start + byte_count
            _require(
                gguf.data_start <= start < end <= details.st_size,
                f"tensor payload exceeds shard bounds for {tensor.name}",
            )
            ranges.append((start, end, tensor.name))
            type_counts[tensor.type_name] += 1

        ranges.sort()
        for previous, current in zip(ranges, ranges[1:]):
            _require(previous[1] <= current[0], f"tensor payload overlap: {previous[2]}, {current[2]}")
        shard_receipts.append(
            {
                "name": name,
                "size": details.st_size,
                "sha256": expected_sha256,
                "split_no": index,
                "data_start": gguf.data_start,
                "tensor_count": len(gguf.tensors),
                "maximum_tensor_end": max(
                    (end for _, end, _ in ranges), default=gguf.data_start
                ),
                "all_payload_bounds_valid": True,
                "all_offsets_aligned": True,
                "no_payload_overlaps": True,
            }
        )

    _require(len(names) == 879, f"split contains {len(names)} tensors, expected 879")
    return {
        "schema_version": 1,
        "checked_at": prep.utc_now(),
        "reader": str(READER_PATH),
        "reader_sha256": hashlib.sha256(READER_PATH.read_bytes()).hexdigest(),
        "artifact_manifest": str(prep.REPORT),
        "artifact_manifest_sha256": hashlib.sha256(artifact_bytes).hexdigest(),
        "split_count": len(prep.SHARDS),
        "tensor_count": len(names),
        "tensor_type_counts": dict(sorted(type_counts.items())),
        "all_tensor_names_unique": True,
        "all_tensor_directory_entries_validated": True,
        "all_tensor_payload_bounds_validated": True,
        "all_offsets_aligned": True,
        "no_payload_overlaps": True,
        "shards": shard_receipts,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args(argv)
    try:
        receipt = verify_layout()
        prep.atomic_write_json(args.report.expanduser().resolve(), receipt)
    except (LayoutError, OSError, UnicodeError, json.JSONDecodeError, ValueError, KeyError) as exc:
        print(f"verify_qwen122_layout: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
