#!/usr/bin/env python3
"""Run the bounded, unscored OSS20B public-failure retry diagnostic.

This endpoint-only probe intentionally does not own a model process or a GPU.
It reuses the frozen public pilot's manifest loader and durable SSE parser while
keeping its diagnostic artifacts separate from every scored benchmark result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

if __package__:
    from . import public_benchmark_pilot as pilot
else:
    import public_benchmark_pilot as pilot


ROOT = Path(__file__).resolve().parents[1]
FROZEN_STUDY = ROOT / "results" / "public-benchmark-pilot-20261023-v1"
FROZEN_MANIFEST = FROZEN_STUDY / "manifest.json"
FROZEN_MANIFEST_SHA256 = "758607e95961b95ca98bdb2bdf11342b961239973dfcff73ff424c3a69603039"
FROZEN_SELECTION_SEED = 20261023
ALLOWED_OUTPUT_ROOT = (
    ROOT / "results" / "public-model-expansion-20261024-v1" / "oss-repair"
)
DEFAULT_ENDPOINT = "http://127.0.0.1:18081/v1"
DEFAULT_MODEL = "gpt-oss-20b-mxfp4"
DIAGNOSTIC_MAX_TOKENS = 8192
DIAGNOSTIC_TIMEOUT_SECONDS = 1200.0
FAILURE_CASES = {37: "1512", 46: "2464", 47: "2216"}
CONTROL_CASES = {
    25: ("humanevalplus", "HumanEval/52"),
    6: ("humanevalplus", "HumanEval/34"),
    62: ("ifeval", "2107"),
    39: ("ifeval", "2859"),
}


class ProbeError(RuntimeError):
    """The diagnostic contract or endpoint execution was invalid."""


@dataclass(frozen=True)
class Recipe:
    label: str
    reasoning_effort: str
    temperature: float
    top_p: float
    top_k: int
    min_p: float
    repeat_penalty: float


RECIPES = {
    "low-full-sampling": Recipe(
        label="low-full-sampling",
        reasoning_effort="low",
        temperature=1.0,
        top_p=1.0,
        top_k=0,
        min_p=0.0,
        repeat_penalty=1.0,
    ),
    "low-greedy-fallback": Recipe(
        label="low-greedy-fallback",
        reasoning_effort="low",
        temperature=0.0,
        top_p=1.0,
        top_k=0,
        min_p=0.0,
        repeat_penalty=1.0,
    ),
}
DOCUMENTATION_CLASSIFICATION = (
    "documented-compatible experiment; not uniquely publisher-recommended"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    pilot.atomic_write_json(path, value)


def load_frozen_manifest() -> pilot.Manifest:
    manifest = pilot.load_manifest(FROZEN_MANIFEST)
    if manifest.sha256 != FROZEN_MANIFEST_SHA256:
        raise ProbeError("frozen public manifest SHA-256 changed")
    if manifest.raw.get("selection_seed") != FROZEN_SELECTION_SEED:
        raise ProbeError("frozen public manifest selection seed changed")
    return manifest


def select_cases(manifest: pilot.Manifest, phase: str) -> tuple[pilot.Case, ...]:
    expected: Mapping[int, Any] = FAILURE_CASES if phase == "failures" else CONTROL_CASES
    by_index = {case.index: case for case in manifest.cases}
    selected: list[pilot.Case] = []
    for index, identity in expected.items():
        case = by_index.get(index)
        if case is None:
            raise ProbeError(f"frozen manifest is missing case index {index}")
        if phase == "failures":
            suite, task_id = "ifeval", identity
        else:
            suite, task_id = identity
        if (case.suite, case.task_id) != (suite, task_id):
            raise ProbeError(f"frozen identity changed at case index {index}")
        selected.append(case)
    return tuple(selected)


def build_payload(
    case: pilot.Case,
    recipe: Recipe,
    *,
    model: str = DEFAULT_MODEL,
    max_tokens: int = DIAGNOSTIC_MAX_TOKENS,
    seed: int = FROZEN_SELECTION_SEED,
) -> dict[str, Any]:
    """Build an answer-free payload from only the case's public prompt."""
    messages = pilot.build_messages(case)
    if case.suite == "ifeval" and messages != [{"role": "user", "content": case.prompt}]:
        raise ProbeError("IFEval payload must contain the verbatim public prompt as user-only")
    return {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": recipe.temperature,
        "top_p": recipe.top_p,
        "top_k": recipe.top_k,
        "min_p": recipe.min_p,
        "repeat_penalty": recipe.repeat_penalty,
        "seed": seed,
        "reasoning_effort": recipe.reasoning_effort,
        "n": 1,
        "stream": True,
        "stream_options": {"include_usage": True},
    }


def completion_outcome(result: Mapping[str, Any]) -> dict[str, Any]:
    status = result.get("status")
    finish_reason = result.get("finish_reason")
    completed_without_cap = (
        status == "completed"
        and finish_reason not in pilot.LENGTH_FINISH_REASONS
        and result.get("stream_complete") is True
    )
    counts = result.get("counts") if isinstance(result.get("counts"), dict) else {}
    timing = result.get("timing") if isinstance(result.get("timing"), dict) else {}
    return {
        "status": status,
        "finish_reason": finish_reason,
        "completed_without_cap": completed_without_cap,
        "cap_is_not_success": True,
        "stream_complete": result.get("stream_complete"),
        "usage": result.get("usage"),
        "counts": counts,
        "native_timings": result.get("native_timings"),
        "timing": timing,
        "error": result.get("error"),
    }


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    data = b"".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":")).encode("utf-8") + b"\n"
        for row in rows
    )
    pilot.atomic_write_bytes(path, data)


def grade_ifeval(
    selected: Sequence[pilot.Case],
    results: Sequence[Mapping[str, Any]],
    output_dir: Path,
    timeout: float = 300.0,
) -> dict[str, Any]:
    pairs = [
        (case, result)
        for case, result in zip(selected[: len(results)], results, strict=True)
        if case.suite == "ifeval"
    ]
    if not pairs:
        return {"status": "not_applicable"}
    ifeval_root = Path(
        load_frozen_manifest().raw["sources"]["ifeval"]["root"]
    ) / "instruction_following_eval"
    suite_dir = output_dir / "official-ifeval"
    official_output = suite_dir / "official-output"
    official_output.mkdir(parents=True, exist_ok=False)
    input_path = suite_dir / "selected_input_data.jsonl"
    response_path = suite_dir / "input_response_data.jsonl"
    _write_jsonl(input_path, (case.original for case, _ in pairs))
    _write_jsonl(
        response_path,
        (
            {
                "prompt": case.prompt,
                "response": result.get("final_text", "")
                if isinstance(result.get("final_text"), str)
                else "",
            }
            for case, result in pairs
        ),
    )
    command = [
        sys.executable,
        "-m",
        "instruction_following_eval.evaluation_main",
        f"--input_data={input_path}",
        f"--input_response_data={response_path}",
        f"--output_dir={official_output}",
    ]
    environment = dict(os.environ)
    package_parent = ifeval_root.parent
    current_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = str(package_parent) + (
        os.pathsep + current_pythonpath if current_pythonpath else ""
    )
    stdout_path = suite_dir / "stdout.log"
    stderr_path = suite_dir / "stderr.log"
    started = time.monotonic()
    with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
        try:
            completed = subprocess.run(
                command,
                cwd=package_parent,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                timeout=timeout,
                check=False,
            )
            returncode: int | None = completed.returncode
            error = None
        except (OSError, subprocess.TimeoutExpired) as exc:
            returncode = None
            error = f"{type(exc).__name__}: {exc}"
    strict_path = official_output / "eval_results_strict.jsonl"
    loose_path = official_output / "eval_results_loose.jsonl"
    receipt: dict[str, Any] = {
        "status": "grader_error",
        "command": command,
        "returncode": returncode,
        "wall_seconds": time.monotonic() - started,
        "error": error,
        "selected_input_data": str(input_path),
        "input_response_data": str(response_path),
        "strict_results": str(strict_path),
        "loose_results": str(loose_path),
    }
    if returncode == 0 and strict_path.is_file() and loose_path.is_file():
        receipt.update(
            {
                "status": "completed",
                "strict": pilot._aggregate_ifeval(strict_path),
                "loose": pilot._aggregate_ifeval(loose_path),
                "error": None,
            }
        )
    atomic_json(suite_dir / "receipt.json", receipt)
    return receipt


def ensure_output_path(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    try:
        resolved.relative_to(ALLOWED_OUTPUT_ROOT.resolve())
    except ValueError as exc:
        raise ProbeError(f"output must be under {ALLOWED_OUTPUT_ROOT}") from exc
    if resolved == ALLOWED_OUTPUT_ROOT.resolve():
        raise ProbeError("output must be a new child directory of the repair root")
    resolved.mkdir(parents=True, exist_ok=False)
    return resolved


def run_probe(args: argparse.Namespace) -> int:
    manifest = load_frozen_manifest()
    selected = select_cases(manifest, args.phase)
    recipe = RECIPES[args.recipe]
    output_dir = ensure_output_path(args.output_dir)
    config_sha256 = sha256_file(Path(args.runtime_config).expanduser().resolve())
    declaration = {
        "schema_version": 1,
        "kind": "unscored_failure_repair_diagnostic",
        "phase": args.phase,
        "recipe": asdict(recipe),
        "recipe_classification": DOCUMENTATION_CLASSIFICATION,
        "endpoint": args.endpoint.rstrip("/"),
        "model": args.model,
        "max_tokens": DIAGNOSTIC_MAX_TOKENS,
        "request_timeout_seconds": DIAGNOSTIC_TIMEOUT_SECONDS,
        "seed": FROZEN_SELECTION_SEED,
        "manifest": {"path": str(FROZEN_MANIFEST), "sha256": manifest.sha256},
        "runtime_config": {
            "path": str(Path(args.runtime_config).expanduser().resolve()),
            "sha256": config_sha256,
        },
        "case_indices": [case.index for case in selected],
        "task_ids": [case.task_id for case in selected],
        "scored_benchmark_eligible": False,
        "merge_into_full64": False,
        "success_requires_no_output_cap": True,
    }
    atomic_json(output_dir / "declaration.json", declaration)
    results: list[dict[str, Any]] = []
    for ordinal, case in enumerate(selected, start=1):
        case_dir = output_dir / f"case-{case.index:02d}-{case.storage_id}"
        case_dir.mkdir()
        payload = build_payload(case, recipe, model=args.model)
        request_record = {
            "schema_version": 1,
            "case": {"index": case.index, "suite": case.suite, "task_id": case.task_id},
            "manifest_sha256": manifest.sha256,
            "payload": payload,
            "url": args.endpoint.rstrip("/") + "/chat/completions",
        }
        atomic_json(case_dir / "request.json", request_record)
        print(
            json.dumps(
                {"event": "case_started", "ordinal": ordinal, "index": case.index, "task_id": case.task_id},
                separators=(",", ":"),
            ),
            flush=True,
        )
        response = pilot.request_chat_completion(
            args.endpoint,
            payload,
            DIAGNOSTIC_TIMEOUT_SECONDS,
            case_dir / "raw.sse",
            None,
        )
        result = {
            "schema_version": 1,
            "case": {"index": case.index, "suite": case.suite, "task_id": case.task_id},
            "manifest_sha256": manifest.sha256,
            "recipe": recipe.label,
            **response,
        }
        atomic_json(case_dir / "result.json", result)
        results.append(result)
        outcome = completion_outcome(result)
        print(
            json.dumps({"event": "case_finished", "index": case.index, **outcome}, separators=(",", ":")),
            flush=True,
        )
        if recipe.label == "low-greedy-fallback" and not outcome["completed_without_cap"]:
            print(
                json.dumps(
                    {
                        "event": "arm_stopped_early",
                        "reason": "first_case_without_normal_uncapped_completion",
                        "index": case.index,
                    },
                    separators=(",", ":"),
                ),
                flush=True,
            )
            break
    grading = grade_ifeval(selected, results, output_dir)
    outcomes = [
        {"case": result["case"], **completion_outcome(result)} for result in results
    ]
    summary = {
        "schema_version": 1,
        "declaration": declaration,
        "outcomes": outcomes,
        "counts": {
            "planned": len(selected),
            "total": len(outcomes),
            "completed_without_cap": sum(
                bool(outcome["completed_without_cap"]) for outcome in outcomes
            ),
            "capped_or_failed": sum(
                not bool(outcome["completed_without_cap"]) for outcome in outcomes
            ),
        },
        "official_ifeval": grading,
    }
    atomic_json(output_dir / "summary.json", summary)
    return 0 if summary["counts"]["completed_without_cap"] == len(outcomes) else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("failures", "controls"), default="failures")
    parser.add_argument("--recipe", choices=tuple(RECIPES), default="low-full-sampling")
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--runtime-config",
        type=Path,
        default=ROOT / "runtime" / "gpt20b-llama-specialists-72k-wide64k.json",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run_probe(args)
    except (ProbeError, pilot.PilotError, OSError, KeyError, ValueError) as exc:
        print(f"oss20b_public_retry_probe: {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
