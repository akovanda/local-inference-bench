import argparse
import json
from pathlib import Path
import subprocess
import sys

import pytest

from runtime import tachyon_role_study as role


def args(root):
    return argparse.Namespace(
        study=root,
        model_plan=root / "models-plan.json",
        guided_probe_plan=None,
        remote="akovanda@tachyon",
        endpoint=role.ENDPOINT,
        wait_timeout=10.0,
        poll=0.01,
        server_timeout=10.0,
        request_timeout=180.0,
    )


def routing_case():
    return {
        "id": "route",
        "suite": "routing",
        "messages": [
            {"role": "system", "content": "Return a route."},
            {"role": "user", "content": "Summarize supplied text."},
        ],
        "expected": {"route": "local_fast"},
        "required_keys": ["route"],
    }


def summary_case():
    return {
        "id": "summary",
        "suite": "summary",
        "messages": [
            {"role": "system", "content": "Return summary and facts."},
            {"role": "user", "content": "No files were deleted."},
        ],
        "expected": {"facts": {"files_deleted": False, "count": 0}},
        "required_keys": ["summary", "facts"],
        "max_summary_words": 35,
    }


def fixture_model():
    return {
        "key": "bonsai2-2b",
        "alias": "tachyon-bonsai2-2b",
        "remote_path": "/external/device-workspace/models/bonsai2-2b/model.gguf",
        "size": 10,
        "sha256": "1" * 64,
        "revision": "rev",
        "repo": "owner/repo",
        "file": "model.gguf",
        "mode_label": "publisher_nonthinking",
        "recipe": {
            "temperature": 0.7,
            "reasoning_effort": "none",
            "chat_template_kwargs": {"enable_thinking": False},
        },
    }


def test_evaluator_accepts_exact_closed_routing_result():
    result = role.evaluate_answer(routing_case(), '{"route":"local_fast"}')
    assert result["json_valid"] is True
    assert result["exact_top_level_keys"] is True
    assert result["expected_values_match"] is True
    assert result["semantic_pass"] is True
    assert result["manual_review_required"] is False


def test_evaluator_rejects_syntax_missing_and_unexpected_keys():
    syntax = role.evaluate_answer(routing_case(), '{"route":')
    assert syntax["json_valid"] is False
    assert syntax["semantic_pass"] is False
    duplicate = role.evaluate_answer(
        routing_case(), '{"route":"lookup","route":"local_fast"}'
    )
    assert duplicate["json_valid"] is False

    missing = role.evaluate_answer(routing_case(), "{}")
    assert missing["missing_keys"] == ["route"]
    assert missing["semantic_pass"] is False

    unexpected = role.evaluate_answer(
        routing_case(), '{"route":"local_fast","confidence":1}'
    )
    assert unexpected["unexpected_keys"] == ["confidence"]
    assert unexpected["semantic_pass"] is False


def test_evaluator_uses_typed_recursive_values_for_negation_and_facts():
    case = summary_case()
    passing = role.evaluate_answer(
        case,
        '{"summary":"No files were deleted.","facts":{"files_deleted":false,"count":0}}',
    )
    assert passing["semantic_pass"] is True
    assert passing["manual_review_required"] is True
    assert "manual review" in passing["claim_scope"]

    bool_is_not_integer = role.evaluate_answer(
        case,
        '{"summary":"No files were deleted.","facts":{"files_deleted":0,"count":0}}',
    )
    assert bool_is_not_integer["mismatched_fields"] == ["facts"]
    assert bool_is_not_integer["semantic_pass"] is False

    invented_fact = role.evaluate_answer(
        case,
        '{"summary":"No files were deleted.","facts":{"files_deleted":false,"count":0,"owner":"Mina"}}',
    )
    assert invented_fact["mismatched_fields"] == ["facts"]


def test_summary_requires_nonempty_text_and_enforces_whitespace_word_limit():
    case = summary_case()
    empty = role.evaluate_answer(
        case, '{"summary":"   ","facts":{"files_deleted":false,"count":0}}'
    )
    assert empty["summary_word_limit_pass"] is False

    too_long = " ".join(["word"] * 36)
    result = role.evaluate_answer(
        case,
        json.dumps(
            {"summary": too_long, "facts": {"files_deleted": False, "count": 0}}
        ),
    )
    assert result["summary_word_count"] == 36
    assert result["summary_word_limit_pass"] is False
    assert result["semantic_pass"] is False


def test_request_never_contains_hidden_expected_or_grading_fields():
    case = summary_case()
    payload = role.build_request(case, fixture_model())
    serialized = json.dumps(payload)
    assert "files_deleted" not in serialized
    assert "required_keys" not in serialized
    assert "max_summary_words" not in serialized
    assert payload["messages"] == case["messages"]
    assert payload["max_tokens"] == 512
    assert payload["seed"] == 20261023


def test_guided_payload_adds_only_json_syntax_guidance_without_answers():
    case = routing_case()
    plan = role.validate_guided_plan(
        role.read_json(role.DEFAULT_STUDY / "guided-probe-plan.json"),
        role.validate_manifest(role.read_json(role.DEFAULT_STUDY / "manifest.json")),
    )
    payload = role.build_guided_request(case, fixture_model(), plan)
    serialized = json.dumps(payload)
    assert payload["response_format"] == {"type": "json_object"}
    assert "local_fast" not in serialized
    assert "expected" not in serialized
    assert "required_keys" not in serialized
    assert payload["max_tokens"] == 512
    assert payload["seed"] == 20261023


def test_resume_binding_rejects_recipe_or_manifest_change(tmp_path):
    controller = role.Controller(args(tmp_path))
    controller.manifest_sha = "a" * 64
    controller.plan_sha = "b" * 64
    model = fixture_model()
    first = controller.ensure_binding(model)
    assert controller.ensure_binding(model)["settings_sha256"] == first["settings_sha256"]

    changed = fixture_model()
    changed["recipe"]["temperature"] = 0.1
    with pytest.raises(role.RoleStudyError, match="resume binding differs"):
        controller.ensure_binding(changed)

    controller.manifest_sha = "c" * 64
    with pytest.raises(role.RoleStudyError, match="resume binding differs"):
        controller.ensure_binding(model)


def test_guided_binding_is_separate_and_resume_bound_to_plan(tmp_path):
    controller = role.Controller(args(tmp_path))
    controller.manifest_sha = "a" * 64
    controller.plan_sha = "b" * 64
    controller.guided_plan_sha = "c" * 64
    controller.guided_plan = {
        "request_overrides": {"response_format": {"type": "json_object"}}
    }
    model = fixture_model()
    baseline = controller.ensure_binding(model)
    guided = controller.ensure_guided_binding(model, baseline)
    path = tmp_path / "models/bonsai2-2b/guided-json-probe/run.json"
    assert path.is_file()
    assert guided["baseline_settings_sha256"] == baseline["settings_sha256"]
    assert guided["settings"]["response_format"] == {"type": "json_object"}

    controller.guided_plan_sha = "d" * 64
    with pytest.raises(role.RoleStudyError, match="guided resume binding differs"):
        controller.ensure_guided_binding(model, baseline)


def test_process_ownership_requires_pid_birth_and_exact_argv():
    receipt = {"pid": 12, "start_ticks": 99, "argv": ["/runtime/server", "-m", "x"]}
    identity = {**receipt, "alive": True, "state": "S"}
    assert role.verify_owned(receipt, identity, "fixture") is True
    with pytest.raises(role.RoleStudyError, match="start_ticks"):
        role.verify_owned(receipt, {**identity, "start_ticks": 100}, "fixture")
    with pytest.raises(role.RoleStudyError, match="argv"):
        role.verify_owned(receipt, {**identity, "argv": ["/tmp/server"]}, "fixture")
    assert role.verify_owned(receipt, {"alive": False, "state": "Z"}, "fixture") is False


def test_server_guard_accepts_gpu_restore_flag_order_but_rejects_path_change():
    model = role.validate_model_plan(role.read_json(role.DEFAULT_PLAN))[0]
    argv = role.server_command(model)[3:]
    effort = argv.index("--reasoning-effort")
    pair = argv[effort : effort + 2]
    del argv[effort : effort + 2]
    argv.extend(pair)  # GPU restore and role launch use different harmless flag order.
    receipt = {"argv": argv, "alias": model["alias"]}
    role.validate_server(receipt, model)
    changed = {**receipt, "argv": ["/tmp/llama-server", *argv[1:]]}
    with pytest.raises(role.RoleStudyError, match="server argv differs"):
        role.validate_server(changed, model)


def test_published_manifest_and_plan_validate_and_bind_fast_modes():
    manifest = role.read_json(role.DEFAULT_STUDY / "manifest.json")
    plan = role.read_json(role.DEFAULT_PLAN)
    cases = role.validate_manifest(manifest)
    models = role.validate_model_plan(plan)
    assert len(cases) == 24
    assert [model["key"] for model in models] == list(role.MODEL_ORDER)
    assert models[0]["recipe"]["reasoning_budget_tokens"] == 0
    assert "chat_template_kwargs" not in models[3]["recipe"]
    assert models[4]["recipe"]["chat_template_kwargs"] == {
        "enable_thinking": False
    }
    assert role.PARAMETER_LABELS["lfm25-12b-instruct"] == "1.2B"
    assert models[3]["size"] > models[1]["size"]  # 1.2B Q4 is larger than ternary 2B.
    assert role.memory_kib(" 123456 kB") == 123456


def test_waiting_interrupt_resumes_by_archiving_unbound_failure(
    tmp_path, monkeypatch
):
    for name in ("manifest.json", "models-plan.json", "guided-probe-plan.json"):
        (tmp_path / name).write_bytes((role.DEFAULT_STUDY / name).read_bytes())
    gpu_trial = tmp_path / "gpu-trial"
    gpu_trial.mkdir()
    gpu_controller = tmp_path / "tachyon_gpu_trial.py"
    gpu_controller.write_text("# pinned fixture\n")
    for name in (
        "trial-status.json",
        "controller-launch.json",
        "cpu-restore-server-launch.json",
        "current-launch.json",
    ):
        (gpu_trial / name).write_text("{}")
    monkeypatch.setattr(role, "GPU_TRIAL", gpu_trial)
    monkeypatch.setattr(role, "GPU_CONTROLLER", gpu_controller)
    monkeypatch.setattr(role, "MANIFEST_SHA256", role.sha256(tmp_path / "manifest.json"))
    monkeypatch.setattr(role, "PLAN_SHA256", role.sha256(tmp_path / "models-plan.json"))
    monkeypatch.setattr(
        role, "GUIDED_PLAN_SHA256", role.sha256(tmp_path / "guided-probe-plan.json")
    )
    status = {
        "schema_version": 1,
        "kind": "tachyon_custom_small_role_study",
        "status": "failed",
        "controller_pid": 123,
        "models": {},
        "error": "controller interrupted by signal 15",
    }
    (tmp_path / "study-status.json").write_text(json.dumps(status))
    parsed = args(tmp_path)
    parsed.guided_probe_plan = tmp_path / "guided-probe-plan.json"
    controller = role.Controller(parsed)
    controller.initialize({"restore": {}})

    archives = list((tmp_path / "history").glob("unbound-wait-123-*.json"))
    assert len(archives) == 1
    assert json.loads(archives[0].read_text()) == status
    resumed = json.loads((tmp_path / "study-status.json").read_text())
    assert resumed["status"] == "running"
    assert resumed["manifest_sha256"] == role.sha256(tmp_path / "manifest.json")
    assert resumed["guided_probe_plan_sha256"] == role.sha256(
        tmp_path / "guided-probe-plan.json"
    )
    assert (tmp_path / "source-pins.json").is_file()

    blocked = tmp_path / "blocked"
    (blocked / "models/model/cases/case").mkdir(parents=True)
    (blocked / "models/model/cases/case/result.json").write_text("{}")
    assert role.archive_empty_wait_failure(blocked, status) is False


def test_guided_comparison_contains_only_same_six_matched_ids(tmp_path):
    controller = role.Controller(args(tmp_path))
    controller.study = tmp_path
    controller.cases = role.validate_manifest(
        role.read_json(role.DEFAULT_STUDY / "manifest.json")
    )
    controller.models = [fixture_model()]
    controller.guided_plan = role.read_json(
        role.DEFAULT_STUDY / "guided-probe-plan.json"
    )
    controller.guided_plan_sha = role.GUIDED_PLAN_SHA256
    for case in controller.guided_cases():
        for guided in (False, True):
            directory = controller.case_dir(controller.models[0], case, guided=guided)
            directory.mkdir(parents=True)
            (directory / "result.json").write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "evaluation": {
                            "json_valid": guided,
                            "semantic_pass": guided,
                        },
                        "usage": {"prompt_tokens": 10, "completion_tokens": 4},
                        "native_timings": {"cache_n": 0, "predicted_per_second": 8.0},
                        "reasoning_text": "",
                        "wall_seconds": 1.0,
                        "ttft_first_seconds": 0.2,
                        "ttft_final_seconds": 0.2,
                    }
                )
            )
    controller.consolidate_guided()
    comparison = json.loads(
        (tmp_path / "guided-json-probe-comparison.json").read_text()
    )
    assert [row["case_id"] for row in comparison["rows"]] == list(
        role.GUIDED_CASE_IDS
    )
    assert all(row["baseline_json_valid"] is False for row in comparison["rows"])
    assert all(row["guided_json_valid"] is True for row in comparison["rows"])
    assert comparison["baseline_case_count_per_model"] == 6
    assert comparison["guided_case_count_per_model"] == 6


def test_direct_script_entrypoint_does_not_require_runtime_package_import():
    script = Path(__file__).parents[1] / "runtime/tachyon_role_study.py"
    completed = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd="/tmp",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--model-plan" in completed.stdout
