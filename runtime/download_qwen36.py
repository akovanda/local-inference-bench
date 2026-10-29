#!/usr/bin/env python3
"""Fetch the exact public Qwen3.6 test artifact pinned by Strata."""
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType


ROOT = Path(__file__).resolve().parents[1]


def load_strata_setup(root: Path = ROOT) -> ModuleType:
    """Load the optional Strata setup helper only when a download is requested."""
    setup_path = root / "vendor" / "strata" / "setup.py"
    if not setup_path.is_file():
        raise RuntimeError(
            "Qwen3.6 download support requires the optional vendor/strata checkout; "
            "initialize that dependency before running this script"
        )
    spec = importlib.util.spec_from_file_location("local_inference_bench_strata_setup", setup_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load optional Strata setup helper: {setup_path}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            f"optional Strata setup helper could not be loaded from {setup_path}: {exc}"
        ) from exc
    return module


def main() -> int:
    setup = load_strata_setup()
    family = setup.OWN_PATH_FAMILIES["qwen36"]
    name = family["file"].format(q="UD-Q4_K_XL")
    target = ROOT / "runtime" / "strata-data" / "models" / "qwen36-UD-Q4_K_XL" / name
    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading pinned artifact: {name}", flush=True)
    setup.download(family["hf"] + name, target, "Qwen3.6-35B-A3B UD-Q4_K_XL")
    setup.verify_sha256(target, *family["sha256"][name])
    print(f"Verified: {target}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
