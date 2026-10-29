import importlib.util
import json
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).parents[1] / "runtime" / "tachyon_qnn_graph_probe.py"
SPEC = importlib.util.spec_from_file_location("tachyon_qnn_graph_probe", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


def _valid_output() -> str:
    events = [
        {"event": "provider", "backend_id": 6, "provider_name": "HTP_QTI_AISW"},
        {"event": "backend", "build_id": "v2.37.0.250724175447_124859"},
        {
            "event": "graph_finalized",
            "nodes": ["FullyConnected", "Relu"],
            "precision": "u8_scale_offset_plus_s32_bias",
            "latency_us": 100,
        },
    ]
    events.extend(
        {"event": "api", "step": step, "result": 0}
        for step in sorted(probe.REQUIRED_API_STEPS)
    )
    events.extend(
        {"event": "api", "step": "graphExecute", "result": 0} for _ in range(6)
    )
    for phase in ("original", "restored"):
        events.append(
            {
                "event": "execution",
                "phase": phase,
                "input_u8": [126, 129, 131, 127],
                "output_u8": [128, 133, 128],
                "output_real": [0, 5, 0],
                "exact_match": True,
                "latency_us": [10, 11, 12],
            }
        )
    events.extend(
        [
            {"event": "context_serialized", "path": "context.bin", "bytes": 42},
            {
                "event": "summary",
                "status": "passed",
                "graph": "FullyConnected+Relu",
                "context_reload": True,
            },
        ]
    )
    return "QNN non-JSON diagnostic\n" + "\n".join(json.dumps(event) for event in events)


def test_cpu_reference_covers_fully_connected_and_relu():
    reference = probe.cpu_reference()
    assert reference["fully_connected_real"] == [-3, 5, -1]
    assert reference["relu_real"] == [0, 5, 0]
    assert reference["output_u8"] == [128, 133, 128]


def test_parse_requires_original_and_reloaded_exact_execution():
    parsed = probe.parse_probe_output(_valid_output())
    assert parsed["provider"]["backend_id"] == 6
    assert parsed["executions"]["restored"]["output_real"] == [0, 5, 0]
    assert parsed["api_call_count"] == len(probe.REQUIRED_API_STEPS) + 6


def test_parse_rejects_numeric_mismatch():
    output = _valid_output().replace(
        '"output_u8": [128, 133, 128]', '"output_u8": [128, 132, 128]', 1
    )
    with pytest.raises(probe.ProbeValidationError, match="CPU reference"):
        probe.parse_probe_output(output)


def test_parse_rejects_non_neural_or_partial_evidence():
    calculator_only = "\n".join(
        [
            json.dumps({"event": "provider", "backend_id": 6, "provider_name": "HTP"}),
            json.dumps({"event": "summary", "status": "passed"}),
        ]
    )
    with pytest.raises(probe.ProbeValidationError, match="missing successful QNN API"):
        probe.parse_probe_output(calculator_only)


def test_captured_device_evidence_validates():
    result_dir = (
        Path(__file__).parents[1]
        / "results"
        / "tachyon-qnn-investigation-20261024-v1"
        / "graph-probe"
    )
    report = probe.validate_result_directory(result_dir)
    assert report["claim"]["actual_neural_graph_execution"] is True
    assert report["claim"]["llm_inference_proven"] is False
    assert report["context"]["bytes"] == 45056
    assert report["evidence"]["successful_qnn_graph_log_lines"] == 6


def test_captured_qnn_250_device_evidence_validates_separately():
    result_dir = (
        Path(__file__).parents[1]
        / "results"
        / "tachyon-qnn-investigation-20261024-v1"
        / "graph-probe"
        / "sdk-2.50"
    )
    report = probe.validate_result_directory(result_dir)
    assert report["qnn"]["runtime_sdk_version"] == "2.50.0.260828"
    assert report["qnn"]["probe_compile_headers"] == "2.37.0.250724"
    assert report["qnn"]["core_api"] == "2.39.0"
    assert report["execution"]["restored"]["output_real"] == [0, 5, 0]
    assert report["context"]["bytes"] == 45696
