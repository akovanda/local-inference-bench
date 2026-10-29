#!/usr/bin/env python3
"""Validate and summarize the Tachyon QNN HTP neural-graph probe.

The device-side probe deliberately emits JSON Lines so this module can reject
partial runs, CPU-reference mismatches, and runs that never reload the compiled
QNN context.  It has no QNN dependency and is safe to run on the controller.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable


INPUT_U8 = [126, 129, 131, 127]
WEIGHTS_U8 = [
    [129, 130, 127, 129],
    [127, 128, 130, 129],
    [128, 127, 129, 126],
]
BIAS_S32 = [1, -2, -5]
ZERO_POINT = 128
SDK_URLS = {
    version: (
        "https://softwarecenter.qualcomm.com/api/download/software/sdks/"
        f"Qualcomm_AI_Runtime_Community/All/{version}/v{version}.zip"
    )
    for version in ("2.37.0.250724", "2.50.0.260828")
}
REQUIRED_API_STEPS = {
    "getProviders",
    "logCreate",
    "backendCreate",
    "backendGetBuildId",
    "deviceCreate",
    "contextCreate",
    "graphCreate",
    "tensorCreate_input",
    "tensorCreate_weights",
    "tensorCreate_bias",
    "tensorCreate_hidden",
    "tensorCreate_output",
    "backendValidate_fullyConnected",
    "graphAddNode_fullyConnected",
    "backendValidate_relu",
    "graphAddNode_relu",
    "graphFinalize",
    "contextGetBinarySize",
    "contextGetBinary",
    "contextFree_original",
    "contextCreateFromBinary",
    "graphRetrieve",
    "contextFree_restored",
    "deviceFree",
    "backendFree",
    "logFree",
}


class ProbeValidationError(ValueError):
    """Raised when evidence does not prove the intended HTP graph run."""


def cpu_reference() -> dict[str, list[int]]:
    """Compute the fixed quantized FullyConnected + ReLU reference."""

    input_real = [value - ZERO_POINT for value in INPUT_U8]
    weights_real = [
        [value - ZERO_POINT for value in row] for row in WEIGHTS_U8
    ]
    fully_connected = [
        sum(x * weight for x, weight in zip(input_real, row)) + bias
        for row, bias in zip(weights_real, BIAS_S32)
    ]
    relu_real = [max(0, value) for value in fully_connected]
    output_u8 = [value + ZERO_POINT for value in relu_real]
    return {
        "input_u8": INPUT_U8,
        "input_real": input_real,
        "weights_u8": [value for row in WEIGHTS_U8 for value in row],
        "weights_real": [value for row in weights_real for value in row],
        "bias_s32": BIAS_S32,
        "fully_connected_real": fully_connected,
        "relu_real": relu_real,
        "output_u8": output_u8,
    }


def _only(records: Iterable[dict[str, Any]], event: str) -> dict[str, Any]:
    matches = [record for record in records if record.get("event") == event]
    if len(matches) != 1:
        raise ProbeValidationError(
            f"expected exactly one {event!r} event, found {len(matches)}"
        )
    return matches[0]


def parse_probe_output(text: str) -> dict[str, Any]:
    """Parse JSONL mixed with QNN messages and validate complete execution."""

    records: list[dict[str, Any]] = []
    non_json_lines: list[str] = []
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        if not line.startswith("{"):
            non_json_lines.append(line)
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ProbeValidationError(
                f"malformed JSON event on line {line_number}: {exc}"
            ) from exc
        if not isinstance(record, dict):
            raise ProbeValidationError(f"event on line {line_number} is not an object")
        records.append(record)

    if not records:
        raise ProbeValidationError("probe output contains no JSON events")
    failures = [record for record in records if record.get("event") == "failure"]
    if failures:
        raise ProbeValidationError(f"probe reported failure: {failures[-1]}")

    api_records = [record for record in records if record.get("event") == "api"]
    failed_calls = [record for record in api_records if record.get("result") != 0]
    if failed_calls:
        raise ProbeValidationError(f"QNN API calls failed: {failed_calls}")
    steps = {str(record.get("step")) for record in api_records}
    missing = sorted(REQUIRED_API_STEPS - steps)
    if missing:
        raise ProbeValidationError(f"missing successful QNN API steps: {missing}")
    execute_count = sum(record.get("step") == "graphExecute" for record in api_records)
    if execute_count != 6:
        raise ProbeValidationError(
            f"expected six successful graphExecute calls, found {execute_count}"
        )

    provider = _only(records, "provider")
    if provider.get("backend_id") != 6 or "HTP" not in str(provider.get("provider_name")):
        raise ProbeValidationError(f"provider is not QNN HTP backend 6: {provider}")

    finalized = _only(records, "graph_finalized")
    if finalized.get("nodes") != ["FullyConnected", "Relu"]:
        raise ProbeValidationError(f"unexpected finalized graph: {finalized}")

    reference = cpu_reference()
    executions = {
        str(record.get("phase")): record
        for record in records
        if record.get("event") == "execution"
    }
    if set(executions) != {"original", "restored"}:
        raise ProbeValidationError(
            f"expected original and restored executions, found {sorted(executions)}"
        )
    for phase, execution in executions.items():
        if execution.get("input_u8") != reference["input_u8"]:
            raise ProbeValidationError(f"{phase} input differs from CPU reference")
        if execution.get("output_u8") != reference["output_u8"]:
            raise ProbeValidationError(f"{phase} quantized output differs from CPU reference")
        if execution.get("output_real") != reference["relu_real"]:
            raise ProbeValidationError(f"{phase} real output differs from CPU reference")
        if execution.get("exact_match") is not True:
            raise ProbeValidationError(f"{phase} did not report an exact match")
        latencies = execution.get("latency_us")
        if (
            not isinstance(latencies, list)
            or len(latencies) != 3
            or any(not isinstance(value, int) or value <= 0 for value in latencies)
        ):
            raise ProbeValidationError(f"{phase} has invalid latency samples: {latencies}")

    serialized = _only(records, "context_serialized")
    if not isinstance(serialized.get("bytes"), int) or serialized["bytes"] <= 0:
        raise ProbeValidationError(f"invalid serialized context event: {serialized}")
    summary = _only(records, "summary")
    if (
        summary.get("status") != "passed"
        or summary.get("graph") != "FullyConnected+Relu"
        or summary.get("context_reload") is not True
    ):
        raise ProbeValidationError(f"probe summary did not pass: {summary}")

    return {
        "provider": provider,
        "backend": _only(records, "backend"),
        "graph": finalized,
        "executions": executions,
        "context": serialized,
        "summary": summary,
        "api_call_count": len(api_records),
        "non_json_line_count": len(non_json_lines),
        "cpu_reference": reference,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_receipt(path: Path) -> dict[str, str]:
    receipt: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition("=")
        if separator:
            receipt[key] = value
    return receipt


def _parse_sha256_manifest(path: Path) -> dict[str, str]:
    manifest: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        digest, separator, artifact_path = line.partition("  ")
        if separator and len(digest) == 64:
            manifest[artifact_path] = digest
    return manifest


def _sdk_version(build_id: str) -> str:
    normalized = build_id[1:] if build_id.startswith("v") else build_id
    for version in SDK_URLS:
        if normalized.startswith(version):
            return version
    raise ProbeValidationError(f"unrecognized QNN SDK build ID: {build_id!r}")


def validate_result_directory(result_dir: Path) -> dict[str, Any]:
    """Validate captured artifacts and return a provenance-rich report."""

    stdout_path = result_dir / "raw.stdout.jsonl"
    stderr_path = result_dir / "raw.stderr.log"
    context_path = result_dir / "context.bin"
    exit_code_path = result_dir / "exit-code.txt"
    receipt_path = result_dir / "launcher-receipt.txt"
    runtime_manifest_path = result_dir / "runtime-sha256.txt"
    required = [
        stdout_path,
        stderr_path,
        context_path,
        exit_code_path,
        receipt_path,
        runtime_manifest_path,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise ProbeValidationError(f"missing result artifacts: {missing}")
    exit_code = int(exit_code_path.read_text(encoding="utf-8").strip())
    if exit_code != 0:
        raise ProbeValidationError(f"probe process exit code was {exit_code}")

    parsed = parse_probe_output(stdout_path.read_text(encoding="utf-8"))
    context_bytes = context_path.stat().st_size
    if parsed["context"]["bytes"] != context_bytes:
        raise ProbeValidationError(
            "serialized context size in output does not match context.bin"
        )
    stderr = stderr_path.read_text(encoding="utf-8", errors="replace")
    successful_graph_log_lines = stderr.count("QnnGraph_execute done. status 0x0")
    if successful_graph_log_lines < 6:
        raise ProbeValidationError(
            "QNN log lacks six successful graph execution completions"
        )
    if "QnnContext_createFromBinary done successfully" not in stderr:
        raise ProbeValidationError("QNN log lacks context reload evidence")
    sdk_version = _sdk_version(str(parsed["backend"]["build_id"]))
    nonfatal_diagnostics = list(
        dict.fromkeys(
            line.strip()
            for line in stderr.splitlines()
            if " <W> " in line or " <E> " in line or "# LIBNATIVE:" in line
        )
    )
    repository_root = Path(__file__).resolve().parents[1]
    source_paths = [
        Path(__file__).resolve(),
        repository_root
        / "vendor"
        / "tachyon-qnn-graph-probe"
        / "src"
        / "qnn_graph_probe.cpp",
        repository_root
        / "vendor"
        / "tachyon-qnn-graph-probe"
        / "bundle"
        / "bin"
        / "qnn-graph-probe",
    ]

    report = {
        "schema_version": 1,
        "status": "passed",
        "scope": "small_neural_graph_qnn_htp_readiness",
        "target": {
            "board": "Particle Tachyon",
            "soc": "Qualcomm QCM6490",
            "accelerator": "Hexagon HTP v68",
            "os": "Ubuntu 20.04",
        },
        "claim": {
            "actual_neural_graph_execution": True,
            "backend": "QNN HTP",
            "cpu_reference_exact_match": True,
            "serialized_context_reload_execution": True,
            "llm_inference_proven": False,
        },
        "qnn": {
            "runtime_sdk_version": sdk_version,
            "runtime_sdk_source_url": SDK_URLS[sdk_version],
            "probe_compile_headers": "2.37.0.250724",
            "backend_id": parsed["provider"]["backend_id"],
            "provider_name": parsed["provider"]["provider_name"],
            "core_api": parsed["provider"].get("core_api"),
            "backend_api": parsed["provider"].get("backend_api"),
            "build_id": parsed["backend"]["build_id"],
        },
        "graph": {
            "name": "tachyon_quantized_mlp",
            "nodes": parsed["graph"]["nodes"],
            "precision": parsed["graph"]["precision"],
            "input_shape": [1, 4],
            "weight_shape": [3, 4],
            "bias_shape": [3],
            "output_shape": [1, 3],
            "finalize_latency_us": parsed["graph"]["latency_us"],
            "cpu_reference": parsed["cpu_reference"],
        },
        "execution": parsed["executions"],
        "context": {
            "bytes": context_bytes,
            "sha256": _sha256(context_path),
            "reload_succeeded": True,
        },
        "process": {
            "exit_code": exit_code,
            "api_call_count": parsed["api_call_count"],
            "launcher": _parse_receipt(receipt_path),
        },
        "artifacts": {
            path.name: {"bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in required
        },
        "runtime_artifact_sha256": _parse_sha256_manifest(runtime_manifest_path),
        "controller_artifacts": {
            str(path.relative_to(repository_root)): {
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in source_paths
            if path.is_file()
        },
        "evidence": {
            "successful_qnn_graph_log_lines": successful_graph_log_lines,
            "successful_context_reload_log_lines": stderr.count(
                "QnnContext_createFromBinary done successfully"
            ),
            "nonfatal_qnn_diagnostics": nonfatal_diagnostics,
        },
    }
    return report


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("reference", help="print the CPU reference as JSON")
    validate_parser = subparsers.add_parser(
        "validate", help="validate a captured graph-probe result directory"
    )
    validate_parser.add_argument("result_dir", type=Path)
    validate_parser.add_argument(
        "--report", type=Path, help="output path (default: RESULT_DIR/report.json)"
    )
    args = parser.parse_args()

    if args.command == "reference":
        print(json.dumps(cpu_reference(), indent=2, sort_keys=True))
        return 0
    report = validate_result_directory(args.result_dir)
    report_path = args.report or args.result_dir / "report.json"
    _write_json_atomic(report_path, report)
    print(json.dumps({"status": "passed", "report": str(report_path)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
