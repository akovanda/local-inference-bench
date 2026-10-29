#!/usr/bin/env python3
"""Pin public benchmark inputs and select the same pilot items before inference."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
from datetime import datetime, timezone
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "vendor/public-benchmarks"
GOOGLE_REV = "e6890f85757dd84e27ca6df2dd30651dafad28e0"
EVALPLUS_REV = "e5d0ed0bab96280b60b637ec7f15b5e4841b0cb2"
HUMAN_URL = "https://github.com/evalplus/humanevalplus_release/releases/download/v0.1.10/HumanEvalPlus.jsonl.gz"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def exclusive(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != data:
            raise RuntimeError(f"existing staged file differs: {path}")
        return
    with path.open("xb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())


def fetch(url: str) -> bytes:
    with urlopen(url, timeout=90) as f:
        return f.read(128 * 1024 * 1024)


def prepare(output: Path, count: int, seed: int) -> dict:
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        if old["selected_count_per_suite"] != count or old["selection_seed"] != seed:
            raise RuntimeError("existing manifest belongs to another selection")
        return old
    ep = VENDOR / "evalplus-v0.3.1"
    actual = subprocess.check_output(["git", "-C", str(ep), "rev-parse", "HEAD"], text=True).strip()
    if actual != EVALPLUS_REV:
        raise RuntimeError("EvalPlus checkout does not match the pinned release")
    if subprocess.check_output(["git", "-C", str(ep), "status", "--porcelain"], text=True).strip():
        raise RuntimeError("official EvalPlus checkout has local changes")
    google_root = VENDOR / f"google-research-{GOOGLE_REV}"
    api = f"https://api.github.com/repos/google-research/google-research/contents/instruction_following_eval?ref={GOOGLE_REV}"
    entries = json.loads(fetch(api))
    files = []
    for entry in entries:
        if entry["type"] != "file":
            continue
        path = entry["path"]
        url = f"https://raw.githubusercontent.com/google-research/google-research/{GOOGLE_REV}/{path}"
        dest = google_root / path
        data = dest.read_bytes() if dest.exists() else fetch(url)
        git_blob = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
        if git_blob != entry["sha"]:
            raise RuntimeError(f"official source Git blob mismatch: {path}")
        exclusive(dest, data)
        files.append({"path": str(dest), "url": url, "git_blob_sha1": git_blob, "sha256": sha(data)})
    input_url = f"https://raw.githubusercontent.com/google-research/google-research/{GOOGLE_REV}/instruction_following_eval/data/input_data.jsonl"
    input_path = google_root / "instruction_following_eval/data/input_data.jsonl"
    if_bytes = input_path.read_bytes() if input_path.exists() else fetch(input_url)
    exclusive(input_path, if_bytes)
    license_url = f"https://raw.githubusercontent.com/google-research/google-research/{GOOGLE_REV}/LICENSE"
    license_path = google_root / "LICENSE"
    exclusive(license_path, license_path.read_bytes() if license_path.exists() else fetch(license_url))
    human_gz = VENDOR / "datasets/HumanEvalPlus-v0.1.10.jsonl.gz"
    human_bytes = human_gz.read_bytes() if human_gz.exists() else fetch(HUMAN_URL)
    exclusive(human_gz, human_bytes)
    human_jsonl = gzip.decompress(human_bytes)
    human_path = human_gz.with_suffix("")
    exclusive(human_path, human_jsonl)
    human = [json.loads(x) for x in human_jsonl.splitlines() if x.strip()]
    instruction = [json.loads(x) for x in if_bytes.splitlines() if x.strip()]
    if len(human) != 164 or len(instruction) != 541:
        raise RuntimeError("unexpected benchmark size; selection not frozen")
    cases = []
    for suite, rows, key in [("humanevalplus", human, "task_id"), ("ifeval", instruction, "key")]:
        ids = [str(x[key]) for x in rows]
        if len(ids) != len(set(ids)):
            raise RuntimeError("duplicate official task IDs")
        ordered = sorted(rows, key=lambda x: sha(f"{seed}:{suite}:{x[key]}".encode()))
        for row in ordered[:count]:
            cases.append({"suite": suite, "task_id": str(row[key]), "prompt": row["prompt"], "original": row})
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "selection_seed": seed,
        "selection_method": "First n items ordered by SHA256(seed:suite:official_task_id); chosen before inference",
        "selected_count_per_suite": count,
        "population_sizes": {"humanevalplus": len(human), "ifeval": len(instruction)},
        "scope": "pilot subset, one sample per task, not a full-benchmark leaderboard score",
        "sources": {
            "evalplus": {"repo": "https://github.com/evalplus/evalplus", "tag": "v0.3.1", "revision": actual, "root": str(ep)},
            "humanevalplus": {"version": "v0.1.10", "url": HUMAN_URL, "path": str(human_path), "compressed_sha256": sha(human_bytes), "sha256": sha(human_jsonl)},
            "ifeval": {"repo": "https://github.com/google-research/google-research", "revision": GOOGLE_REV, "root": str(google_root), "data_url": input_url, "data_path": str(input_path), "data_sha256": sha(if_bytes), "source_files": files},
        },
        "cases": cases,
    }
    data = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    exclusive(manifest_path, data)
    exclusive(output / "manifest.sha256", f"{sha(data)}  manifest.json\n".encode())
    return manifest


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--count", type=int, default=32)
    p.add_argument("--seed", type=int, default=20261023)
    args = p.parse_args()
    if not 1 <= args.count <= 164:
        p.error("count must be 1 through 164")
    m = prepare(args.output_dir.resolve(), args.count, args.seed)
    print(json.dumps({"manifest": str(args.output_dir.resolve() / "manifest.json"), "cases": len(m["cases"]), "per_suite": args.count, "seed": args.seed, "sources": {k: {f: v for f, v in x.items() if f in ("revision", "version", "sha256", "data_sha256")} for k, x in m["sources"].items()}}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
