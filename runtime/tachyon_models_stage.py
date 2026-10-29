#!/usr/bin/env python3
"""Stage immutable recent small-model artifacts on Tachyon without loading them."""
import hashlib
import json
import os
from pathlib import Path
import time
from datetime import datetime, timezone
from urllib.request import Request, urlopen

ROOT = Path.home() / "moe-bench"
MODELS = [
    {"key": "bonsai2-2b", "repo": "prism-ml/Ternary-Bonsai-2-2B-gguf", "revision": "683f7be2629342a50987627b40881e54761a2430", "file": "Ternary-Bonsai-2-2B-Q2_0.gguf", "size": 546689504, "sha256": "c2893fa6f951daee8b7263d5f49b455e57989ef56417a22125b03566dc560650", "release": "2026-10-15", "base": "Qwen3.5-2B", "packing": "Q2_0"},
    {"key": "bonsai2-4b", "repo": "prism-ml/Ternary-Bonsai-2-4B-gguf", "revision": "8d6557168168f91ea6697a4a49ed2ef622b53adb", "file": "Ternary-Bonsai-2-4B-Q2_0.gguf", "size": 1212103520, "sha256": "64a5183b44921bdb5ac4a6f5f92007821d6ebe5c66997d3c147fd214ab9433d4", "release": "2026-10-15", "base": "Qwen3.5-4B", "packing": "Q2_0"},
    {"key": "lfm25-26b", "repo": "LiquidAI/LFM2.5-2.6B-GGUF", "revision": "e7caca5d835a3901a8e0d63e94009429bafafdfc", "file": "LFM2.5-2.6B-QAD-Q4_0.gguf", "size": 1593894944, "sha256": "a247afd6414918eac8e520a9e6137dc271235461ecbe1180462221d5b8d40b03", "release": "2026-08-04", "base": "LFM2.5-2.6B", "packing": "QAD-Q4_0"},
]


def sha_file(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def status(row):
    row = {"at": datetime.now(timezone.utc).isoformat(), **row}
    p = ROOT / "results/model-download-events.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a") as f:
        f.write(json.dumps(row) + "\n")
        f.flush()
        os.fsync(f.fileno())
    print(json.dumps(row), flush=True)


def stage(m):
    directory = ROOT / "models" / m["key"]
    directory.mkdir(parents=True, exist_ok=True)
    dest = directory / m["file"]
    if dest.exists():
        if dest.stat().st_size == m["size"] and sha_file(dest) == m["sha256"]:
            status({"event": "already_verified", "model": m["key"]})
            return
        raise RuntimeError("existing complete destination differs: " + str(dest))
    partial = dest.with_suffix(dest.suffix + ".partial")
    url = "https://huggingface.co/{repo}/resolve/{revision}/{file}?download=true".format(**m)
    started = time.monotonic()
    status({"event": "download_start", "model": m["key"], "bytes": m["size"], "url": url})
    for attempt in range(1, 5):
        try:
            offset = partial.stat().st_size if partial.exists() else 0
            headers = {"User-Agent": "tachyon-model-benchmark/1"}
            if offset:
                headers["Range"] = "bytes={}-".format(offset)
            with urlopen(Request(url, headers=headers), timeout=120) as response:
                if offset and response.status != 206:
                    raise RuntimeError("server did not honor resume range")
                if offset and not response.headers.get("Content-Range", "").startswith("bytes {}-".format(offset)):
                    raise RuntimeError("unexpected Content-Range")
                with partial.open("ab" if offset else "wb") as f:
                    last = time.monotonic()
                    while True:
                        block = response.read(1024 * 1024)
                        if not block:
                            break
                        f.write(block)
                        offset += len(block)
                        if offset > m["size"]:
                            raise RuntimeError("download exceeded expected size")
                        if time.monotonic() - last >= 15:
                            f.flush()
                            status({"event": "download_progress", "model": m["key"], "bytes": offset, "total": m["size"]})
                            last = time.monotonic()
                    f.flush()
                    os.fsync(f.fileno())
            if partial.stat().st_size != m["size"]:
                raise RuntimeError("short download")
            actual = sha_file(partial)
            if actual != m["sha256"]:
                raise RuntimeError("model checksum mismatch")
            os.replace(partial, dest)
            receipt = {**m, "path": str(dest), "actual_sha256": actual, "verified_at": datetime.now(timezone.utc).isoformat(), "download_seconds": time.monotonic() - started}
            (directory / "artifact.json").write_text(json.dumps(receipt, indent=2) + "\n")
            status({"event": "model_verified", "model": m["key"], "path": str(dest), "sha256": actual, "seconds": receipt["download_seconds"]})
            return
        except Exception as exc:
            status({"event": "download_error", "model": m["key"], "attempt": attempt, "error": str(exc)})
            if attempt == 4:
                raise
            time.sleep(2)


if __name__ == "__main__":
    (ROOT / "results").mkdir(parents=True, exist_ok=True)
    (ROOT / "models-plan-20261023.json").write_text(json.dumps(MODELS, indent=2) + "\n")
    for model in MODELS:
        stage(model)
    status({"event": "all_models_verified", "count": len(MODELS)})
