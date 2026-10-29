import copy
import json
import subprocess
import sys

import pytest

from runtime import tachyon_worker_latency_trial as trial


def published_route_and_model():
    manifest = trial.read_json(trial.MANIFEST)
    plan = trial.read_json(trial.MODEL_PLAN)
    case = next(row for row in manifest["cases"] if row["suite"] == "routing")
    model = next(row for row in plan["models"] if row["key"] == trial.MODEL_KEY)
    return case, model


def test_ownership_guard_accepts_exact_identity_and_rejects_live_mismatch():
    receipt = {"pid": 12, "start_ticks": 34, "argv": ["/owned/server", "-m", "x"]}
    assert trial.verify_owned_process(
        receipt, {**receipt, "alive": True, "state": "S"}, "fixture"
    ) is True
    assert trial.verify_owned_process(
        receipt, {"pid": 12, "alive": False, "state": "exited", "argv": []},
        "fixture",
    ) is False
    with pytest.raises(trial.TrialError, match="start_ticks"):
        trial.verify_owned_process(
            receipt, {**receipt, "alive": True, "start_ticks": 35}, "fixture"
        )
    with pytest.raises(trial.TrialError, match="argv"):
        trial.verify_owned_process(
            receipt, {**receipt, "alive": True, "argv": ["/not-owned"]}, "fixture"
        )


def test_request_uses_public_schema_and_never_serializes_expected_answer():
    case, model = published_route_and_model()
    payload = trial.build_request(case, model)
    changed = copy.deepcopy(case)
    changed["expected"] = {"route": "hidden-different-answer"}
    assert trial.build_request(changed, model) == payload
    assert payload["messages"] == case["messages"]
    assert payload["response_format"]["json_schema"]["schema"] == case["public_schema"]
    assert payload["response_format"]["json_schema"]["strict"] is True
    assert payload["seed"] == 20261023
    assert "expected" not in payload
    assert "required_keys" not in payload
    assert "hidden-different-answer" not in json.dumps(payload)


def test_sse_parser_preserves_final_reasoning_usage_and_native_timings():
    lines = [
        b'data: {"choices":[{"delta":{"reasoning_content":"brief"}}]}\n',
        b'data: {"choices":[{"delta":{"content":"{\\"route\\":"}}]}\n',
        b'data: {"choices":[{"delta":{"content":"\\"lookup\\"}"},"finish_reason":"stop"}]}\n',
        b'data: {"choices":[],"usage":{"prompt_tokens":185,"completion_tokens":4,"prompt_tokens_details":{"cached_tokens":151}},"timings":{"cache_n":151,"prompt_n":34,"prompt_ms":620.5,"predicted_n":4,"predicted_ms":200.0,"predicted_per_second":15.0}}\n',
        b'data: [DONE]\n',
    ]
    parsed = trial.parse_sse_lines(lines)
    assert parsed["final_text"] == '{"route":"lookup"}'
    assert parsed["reasoning_text"] == "brief"
    assert parsed["finish_reason"] == "stop"
    assert parsed["usage"]["prompt_tokens_details"]["cached_tokens"] == 151
    assert parsed["native_timings"]["cache_n"] == 151
    assert parsed["native_timings"]["prompt_n"] == 34
    assert parsed["native_timings"]["prompt_ms"] == 620.5


def test_selection_requires_b_to_beat_both_a_epochs_without_quality_loss(tmp_path):
    args = trial.parse_args(["--trial", str(tmp_path)])
    controller = trial.Controller(args)
    rows = [
        {"epoch": trial.EPOCHS[0], "completed": 8, "semantic_passes": 8,
         "warm_ttft_first_median_seconds": 2.2, "warm_wall_median_seconds": 3.7},
        {"epoch": trial.EPOCHS[1], "completed": 8, "semantic_passes": 8,
         "warm_ttft_first_median_seconds": .8, "warm_wall_median_seconds": 2.1},
        {"epoch": trial.EPOCHS[2], "completed": 8, "semantic_passes": 8,
         "warm_ttft_first_median_seconds": 2.4, "warm_wall_median_seconds": 3.8},
    ]
    assert controller.choose_ubatch(rows)["selected_ubatch"] == 32
    rows[1]["semantic_passes"] = 7
    assert controller.choose_ubatch(rows)["selected_ubatch"] == 128


def test_direct_cli_help_works():
    completed = subprocess.run(
        [sys.executable, str(trial.ROOT / "runtime/tachyon_worker_latency_trial.py"), "--help"],
        cwd=trial.ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--trial" in completed.stdout
