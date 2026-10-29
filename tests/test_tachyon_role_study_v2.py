import argparse
import json
from pathlib import Path
import subprocess
import sys

import pytest

from runtime import tachyon_role_study_v2 as role


def args(root):
    return argparse.Namespace(
        study=root,
        model_plan=root / "models-plan.json",
        guided_probe_plan=None,
        handoff_receipt=None,
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
    assert "response_format" not in payload


def test_guided_payload_adds_only_public_schema_and_is_independent_of_expected():
    cases = role.validate_manifest(
        role.read_json(role.DEFAULT_STUDY / "manifest.json")
    )
    case = next(row for row in cases if row["id"] == "route-summary")
    plan = role.validate_guided_plan(
        role.read_json(role.DEFAULT_STUDY / "guided-schema-probe-plan.json"), cases
    )
    baseline = role.build_request(case, fixture_model())
    payload = role.build_guided_request(case, fixture_model(), plan)
    mutated = json.loads(json.dumps(case))
    mutated["expected"] = {"route": "a-deliberately-different-answer"}
    assert role.build_guided_request(mutated, fixture_model(), plan) == payload
    without_contract = {key: value for key, value in payload.items()
                        if key != "response_format"}
    assert without_contract == baseline
    contract = payload["response_format"]
    assert contract["type"] == "json_schema"
    assert contract["json_schema"]["strict"] is True
    assert contract["json_schema"]["schema"]["properties"]["route"]["enum"] == [
        "clarify", "lookup", "code_model", "analysis_model", "local_fast"
    ]
    assert payload["max_tokens"] == 512
    assert payload["seed"] == 20261023

    cache = next(row for row in cases if row["id"] == "decision-cache-boundary")
    cache_payload = role.build_guided_request(cache, fixture_model(), plan)
    cache_properties = cache_payload["response_format"]["json_schema"]["schema"][
        "properties"
    ]
    assert cache_properties["action"]["enum"] == ["use_cache", "fetch", "inspect"]
    assert cache_properties["reason_code"]["enum"] == [
        "fresh", "stale", "unknown_age"
    ]


def test_guided_plan_rejects_extra_request_controls_and_single_answer_enums():
    cases = role.validate_manifest(
        role.read_json(role.DEFAULT_STUDY / "manifest.json")
    )
    plan = role.read_json(
        role.DEFAULT_STUDY / "guided-schema-probe-plan.json"
    )
    extra_control = json.loads(json.dumps(plan))
    extra_control["request_overrides_by_case"]["route-summary"]["temperature"] = 0
    with pytest.raises(role.RoleStudyError, match="changes request fields"):
        role.validate_guided_plan(extra_control, cases)

    singleton = json.loads(json.dumps(plan))
    route_schema = singleton["request_overrides_by_case"]["route-summary"][
        "response_format"
    ]["json_schema"]["schema"]
    route_schema["properties"]["route"]["enum"] = ["local_fast"]
    with pytest.raises(role.RoleStudyError, match="must contain alternatives"):
        role.validate_guided_plan(singleton, cases)


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
    controller.guided_plan = role.read_json(
        role.DEFAULT_STUDY / "guided-schema-probe-plan.json"
    )
    model = fixture_model()
    baseline = controller.ensure_binding(model)
    guided = controller.ensure_guided_binding(model, baseline)
    path = tmp_path / "models/bonsai2-2b/guided-public-schema-probe/run.json"
    assert path.is_file()
    assert guided["baseline_settings_sha256"] == baseline["settings_sha256"]
    assert guided["settings"]["experiment"] == "guided_public_schema_matched_six"
    assert set(guided["settings"]["request_overrides_by_case"]) == set(
        role.GUIDED_CASE_IDS
    )

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


def test_parent_handoff_requires_frozen_sources_exited_controller_and_live_identity(
    tmp_path, monkeypatch
):
    model = role.read_json(role.V1_STUDY / "models-plan.json")["models"][0]
    server = {
        "schema_version": 1,
        "kind": "tachyon_role_server",
        "pid": 222,
        "start_ticks": 333,
        "argv": role.server_command(model)[3:],
        "alias": model["alias"],
        "model_key": model["key"],
    }
    raw = {**server, "alive": True, "state": "S"}
    source = tmp_path / "default-restore-server-launch.json"
    launch = tmp_path / "controller-launch.json"
    source.write_text(json.dumps(server))
    launch.write_text(
        json.dumps({"pid": 111, "source_sha256": role.DERIVED_SOURCE_SHA256})
    )
    handoff_path = tmp_path / "parent-handoff.json"
    handoff = {
        "schema_version": 1,
        "kind": "tachyon_role_v2_parent_handoff",
        "source_receipt_path": str(source.resolve()),
        "source_receipt_sha256": role.sha256(source),
        "old_controller_launch_path": str(launch.resolve()),
        "old_controller_launch_sha256": role.sha256(launch),
        "old_controller_pid": 111,
        "alias": model["alias"],
        "server": server,
        "raw_server_identity": raw,
    }
    handoff_path.write_text(json.dumps(handoff))
    monkeypatch.setattr(role, "V1_DEFAULT_RESTORE", source)
    monkeypatch.setattr(role, "V1_CONTROLLER_LAUNCH", launch)
    monkeypatch.setattr(role, "HANDOFF_SHA256", role.sha256(handoff_path))
    monkeypatch.setattr(role, "process_identity", lambda _pid: {"alive": False})
    parsed = args(tmp_path)
    parsed.handoff_receipt = handoff_path
    controller = role.Controller(parsed)
    controller.remote_process = lambda _pid: raw
    assert controller.recovery_handoff(model, require_live=True) == server

    handoff["raw_server_identity"] = {**raw, "start_ticks": 334}
    handoff_path.write_text(json.dumps(handoff))
    monkeypatch.setattr(role, "HANDOFF_SHA256", role.sha256(handoff_path))
    with pytest.raises(role.RoleStudyError, match="raw server identity"):
        controller.recovery_handoff(model, require_live=True)

    handoff["raw_server_identity"] = raw
    handoff["alias"] = "wrong-alias"
    handoff_path.write_text(json.dumps(handoff))
    monkeypatch.setattr(role, "HANDOFF_SHA256", role.sha256(handoff_path))
    with pytest.raises(role.RoleStudyError, match="handoff alias"):
        controller.recovery_handoff(model, require_live=True)


def test_server_guard_accepts_gpu_restore_flag_order_but_rejects_path_change():
    model = role.read_json(role.V1_STUDY / "models-plan.json")["models"][0]
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
    assert models[0]["recipe"]["reasoning_effort"] == "low"
    assert models[0]["recipe"]["chat_template_kwargs"] == {
        "enable_thinking": True
    }
    assert "chat_template_kwargs" not in models[3]["recipe"]
    assert models[4]["recipe"]["chat_template_kwargs"] == {
        "enable_thinking": False
    }
    assert role.PARAMETER_LABELS["lfm25-12b-instruct"] == "1.2B"
    assert models[3]["size"] > models[1]["size"]  # 1.2B Q4 is larger than ternary 2B.
    assert role.memory_kib(" 123456 kB") == 123456


def test_v2_plan_guard_rejects_each_old_lfm26_protocol_setting():
    plan = role.read_json(role.V1_STUDY / "models-plan.json")
    plan["models"][0]["recipe"].update(
        {
            "reasoning_budget_tokens": 0,
            "reasoning_effort": "low",
            "chat_template_kwargs": {"enable_thinking": True},
        }
    )
    role.validate_model_plan(plan)
    for field, value in (
        ("reasoning_budget_tokens", 1),
        ("reasoning_effort", "none"),
        ("chat_template_kwargs", {"enable_thinking": False}),
    ):
        broken = json.loads(json.dumps(plan))
        broken["models"][0]["recipe"][field] = value
        with pytest.raises(role.RoleStudyError, match="LFM2.6 V2"):
            role.validate_model_plan(broken)


def test_waiting_interrupt_resumes_by_archiving_unbound_failure(
    tmp_path, monkeypatch
):
    for name in ("manifest.json", "models-plan.json", "guided-schema-probe-plan.json"):
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
        role, "GUIDED_PLAN_SHA256",
        role.sha256(tmp_path / "guided-schema-probe-plan.json")
    )
    status = {
        "schema_version": 1,
        "kind": role.STUDY_KIND,
        "status": "failed",
        "controller_pid": 123,
        "models": {},
        "error": "controller interrupted by signal 15",
    }
    (tmp_path / "study-status.json").write_text(json.dumps(status))
    parsed = args(tmp_path)
    parsed.guided_probe_plan = tmp_path / "guided-schema-probe-plan.json"
    controller = role.Controller(parsed)
    controller.initialize({"restore": {}})

    archives = list((tmp_path / "history").glob("unbound-wait-123-*.json"))
    assert len(archives) == 1
    assert json.loads(archives[0].read_text()) == status
    resumed = json.loads((tmp_path / "study-status.json").read_text())
    assert resumed["status"] == "running"
    assert resumed["manifest_sha256"] == role.sha256(tmp_path / "manifest.json")
    assert resumed["guided_probe_plan_sha256"] == role.sha256(
        tmp_path / "guided-schema-probe-plan.json"
    )
    assert (tmp_path / "source-pins.json").is_file()
    pins = json.loads((tmp_path / "source-pins.json").read_text())["sha256"]
    assert pins[str(role.DERIVED_SOURCE)] == role.DERIVED_SOURCE_SHA256
    assert pins[str(Path(role.__file__).resolve())] == role.sha256(
        Path(role.__file__).resolve()
    )

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
        role.DEFAULT_STUDY / "guided-schema-probe-plan.json"
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
        (tmp_path / "guided-public-schema-probe-comparison.json").read_text()
    )
    assert [row["case_id"] for row in comparison["rows"]] == list(
        role.GUIDED_CASE_IDS
    )
    assert all(row["baseline_json_valid"] is False for row in comparison["rows"])
    assert all(row["guided_json_valid"] is True for row in comparison["rows"])
    assert comparison["baseline_case_count_per_model"] == 6
    assert comparison["guided_case_count_per_model"] == 6


def test_failure_cleanup_temporarily_clears_interrupt_and_retains_primary_error(
    tmp_path, capsys, monkeypatch
):
    controller = role.Controller(args(tmp_path))
    monkeypatch.setattr(role.signal, "signal", lambda *_args: None)
    controller.acquire_lock = lambda: None
    restored_with = []
    saved_states = []

    def fail_after_interrupt():
        controller.models = [fixture_model()]
        controller.state = {
            "schema_version": 1,
            "kind": role.STUDY_KIND,
            "models": {},
        }
        controller.interrupted = 15
        raise role.RoleStudyError("primary interrupted failure")

    def restore_default():
        restored_with.append(controller.interrupted)

    controller.wait_for_gpu = fail_after_interrupt
    controller.restore_default = restore_default
    controller.save = lambda **fields: saved_states.append(fields)
    controller.event = lambda *_args, **_kwargs: None

    assert controller.run() == 1
    assert restored_with == [None]
    assert controller.interrupted == 15
    assert saved_states[-1]["status"] == "failed"
    assert saved_states[-1]["error"] == "primary interrupted failure"
    assert "primary interrupted failure" in capsys.readouterr().err


def test_direct_script_entrypoint_does_not_require_runtime_package_import():
    script = Path(__file__).parents[1] / "runtime/tachyon_role_study_v2.py"
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
