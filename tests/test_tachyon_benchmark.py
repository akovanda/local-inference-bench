import argparse
import copy
import json
from pathlib import Path

import pytest

from runtime import tachyon_benchmark as benchmark


def run_args(profile="bonsai-nothink", model_key="bonsai2-2b"):
    return argparse.Namespace(
        endpoint="http://127.0.0.1:18185/v1",
        model="local-model",
        model_key=model_key,
        profile=profile,
        max_tokens=8192,
        request_timeout=3600.0,
        seed=20261023,
        api_key_env=None,
        rerun_failed=False,
    )


def test_recipe_is_in_settings_binding_and_changed_recipe_cannot_resume(tmp_path):
    runner = benchmark.load_frozen_runner()
    settings = benchmark.adapter_run_settings(runner, run_args())
    assert settings["original_profile"] == "bonsai-nothink"
    assert settings["native_recipe_sha256"] == benchmark._sha256_json(
        settings["native_recipe"]
    )
    manifest = runner.Manifest(tmp_path / "manifest.json", "manifest-sha", {}, ())
    first = runner.ensure_binding(tmp_path / "output", manifest, settings)
    assert first["settings"]["native_recipe"] == settings["native_recipe"]

    changed = copy.deepcopy(settings)
    changed["native_recipe"]["request_fields"]["temperature"] = 0.6
    changed["native_recipe_sha256"] = benchmark._sha256_json(
        changed["native_recipe"]
    )
    with pytest.raises(runner.PilotError, match="different run settings"):
        runner.ensure_binding(tmp_path / "output", manifest, changed)


def test_payload_never_leaks_original_grader_fields():
    runner = benchmark.load_frozen_runner()
    benchmark.install_runner_wrappers(runner)
    settings = runner.run_settings(run_args())
    case = runner.Case(
        0,
        "humanevalplus",
        "HumanEval/secret",
        "def public_prompt():\n    pass",
        {
            "prompt": "def public_prompt():\n    pass",
            "canonical_solution": "HIDDEN_CANONICAL_SOLUTION",
            "test": "HIDDEN_GRADER_TEST",
        },
    )

    payload = runner.build_request_payload(case, settings)
    serialized = json.dumps(payload)
    assert "public_prompt" in serialized
    assert "HIDDEN_CANONICAL_SOLUTION" not in serialized
    assert "HIDDEN_GRADER_TEST" not in serialized
    assert "original" not in payload


def test_thinking_template_control_is_bonsai_only_and_profiles_are_model_bound():
    runner = benchmark.load_frozen_runner()
    bonsai_settings = benchmark.adapter_run_settings(runner, run_args())
    case = runner.Case(0, "ifeval", "ifeval-0", "public instruction", {})
    bonsai = benchmark.adapter_request_payload(runner, case, bonsai_settings)
    assert bonsai["reasoning_effort"] == "none"
    assert bonsai["chat_template_kwargs"] == {"enable_thinking": False}

    lfm_settings = benchmark.adapter_run_settings(
        runner, run_args("lfm26", "lfm25-26b")
    )
    lfm = benchmark.adapter_request_payload(runner, case, lfm_settings)
    assert lfm["reasoning_effort"] == "none"
    assert "chat_template_kwargs" not in lfm
    assert lfm["temperature"] == 0.1
    assert lfm["top_p"] == 1.0
    assert lfm["top_k"] == 50
    assert lfm["min_p"] == 0.0
    assert lfm["repeat_penalty"] == 1.1

    with pytest.raises(benchmark.TachyonBenchmarkError, match="requires model-key"):
        benchmark.native_recipe("bonsai-nothink", "lfm25-26b")


def test_frozen_runner_hash_mismatch_is_rejected_before_import(tmp_path):
    hostile = tmp_path / "hostile.py"
    hostile.write_text("raise AssertionError('must never execute')\n")
    with pytest.raises(benchmark.TachyonBenchmarkError, match="SHA-256 mismatch"):
        benchmark.load_frozen_runner(hostile)


def test_summary_uses_authoritative_usage_and_weighted_native_timing(tmp_path):
    runner = benchmark.load_frozen_runner()
    settings = benchmark.adapter_run_settings(runner, run_args())
    output = tmp_path / "output"
    output.mkdir()
    run = {
        "schema_version": 1,
        "manifest": {"sha256": "manifest-sha", "case_count": 1},
        "settings": settings,
        "settings_sha256": benchmark._sha256_json(settings),
    }
    (output / "run.json").write_text(json.dumps(run))
    (output / "status.json").write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "suite": "humanevalplus",
                        "storage_id": "case-id",
                        "status": "completed",
                    }
                ]
            }
        )
    )
    attempt = output / "cases/humanevalplus/case-id/attempt-0001"
    attempt.mkdir(parents=True)
    (attempt / "result.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "manifest_sha256": "manifest-sha",
                "settings_sha256": run["settings_sha256"],
                "counts": {
                    "prompt_tokens": 5,
                    "completion_tokens": 20,
                    "total_tokens": 25,
                    "reasoning_tokens": None,
                    "token_counts_authoritative": True,
                },
                "native_timings": {
                    "predicted_n": 20,
                    "predicted_ms": 2000,
                    "predicted_per_second": 9.5,
                },
                "timing": {
                    "wall_seconds": 2.5,
                    "ttft_seconds": 0.4,
                    "ttft_final_seconds": 0.7,
                },
            }
        )
    )

    summary = benchmark.build_summary(output)
    overall = summary["rows"][0]
    assert summary["full_leaderboard_claim"] is False
    assert overall["completion_tokens"] == 20
    assert overall["authoritative_usage_case_count"] == 1
    assert overall["native_completion_tokens_per_second"] == 10.0
    assert overall["count_time_derived_generation_tokens_per_second"] == 10.0
    assert overall["server_reported_generation_timing_case_count"] == 1
    assert overall["server_reported_generation_tokens_per_second"] == 9.5
    assert overall["wall_seconds_sum"] == 2.5
    assert overall["ttft_seconds_mean"] == 0.4
