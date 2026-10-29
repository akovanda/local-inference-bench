import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from runtime import tachyon_worker_study as worker


def args(root):
    return argparse.Namespace(
        study=root,
        model_plan=worker.DEFAULT_PLAN,
        handoff_receipt=worker.DEFAULT_STUDY / "parent-handoff.json",
        remote="akovanda@tachyon",
        endpoint=worker.ENDPOINT,
        wait_timeout=10.0,
        poll=0.01,
        server_timeout=10.0,
        request_timeout=180.0,
    )


def routing_case():
    return {
        "id": "route-fixture",
        "suite": "routing",
        "messages": [
            {"role": "system", "content": "Choose one allowed route."},
            {"role": "user", "content": "Summarize supplied text."},
        ],
        "expected": {"route": "local_fast"},
        "required_keys": ["route"],
        "public_schema": {
            "type": "object",
            "properties": {
                "route": {
                    "type": "string",
                    "enum": ["clarify", "lookup", "local_fast"],
                }
            },
            "required": ["route"],
            "additionalProperties": False,
        },
    }


def summary_case():
    return {
        "id": "summary-fixture",
        "suite": "summary",
        "messages": [
            {"role": "system", "content": "Return summary and facts."},
            {"role": "user", "content": "No files were deleted."},
        ],
        "expected": {"facts": {"files_deleted": False, "count": 0}},
        "required_keys": ["summary", "facts"],
        "max_summary_words": 35,
        "public_schema": {
            "type": "object",
            "properties": {
                "summary": {"type": "string"},
                "facts": {
                    "type": "object",
                    "properties": {
                        "files_deleted": {"type": "boolean"},
                        "count": {"type": "integer"},
                    },
                    "required": ["files_deleted", "count"],
                    "additionalProperties": False,
                },
            },
            "required": ["summary", "facts"],
            "additionalProperties": False,
        },
    }


def published():
    cases = worker.validate_manifest(
        worker.read_json(worker.DEFAULT_STUDY / "manifest.json")
    )
    models = worker.validate_model_plan(worker.read_json(worker.DEFAULT_PLAN))
    return cases, models


def test_typed_evaluator_checks_syntax_keys_values_negation_and_word_limit():
    exact = worker.evaluate_answer(routing_case(), '{"route":"local_fast"}')
    assert exact["semantic_pass"] is True
    assert exact["manual_review_required"] is False

    assert worker.evaluate_answer(routing_case(), '{"route":')["json_valid"] is False
    missing = worker.evaluate_answer(routing_case(), "{}")
    assert missing["missing_keys"] == ["route"]
    extra = worker.evaluate_answer(
        routing_case(), '{"route":"local_fast","confidence":1}'
    )
    assert extra["unexpected_keys"] == ["confidence"]
    duplicate = worker.evaluate_answer(
        routing_case(), '{"route":"lookup","route":"local_fast"}'
    )
    assert duplicate["json_valid"] is False

    passing = worker.evaluate_answer(
        summary_case(),
        '{"summary":"No files were deleted.","facts":{"files_deleted":false,"count":0}}',
    )
    assert passing["semantic_pass"] is True
    assert passing["manual_review_required"] is True
    wrong_type = worker.evaluate_answer(
        summary_case(),
        '{"summary":"No files were deleted.","facts":{"files_deleted":0,"count":0}}',
    )
    assert wrong_type["mismatched_fields"] == ["facts"]
    too_long = " ".join(["word"] * 36)
    limited = worker.evaluate_answer(
        summary_case(),
        json.dumps({"summary": too_long,
                    "facts": {"files_deleted": False, "count": 0}}),
    )
    assert limited["summary_word_count"] == 36
    assert limited["semantic_pass"] is False


def test_all_requests_use_public_schema_and_never_depend_on_expected_values():
    cases, models = published()
    for case in cases:
        payload = worker.build_request(case, models[0])
        changed = copy.deepcopy(case)
        changed["expected"] = {"hidden": "different"}
        assert worker.build_request(changed, models[0]) == payload
        assert payload["messages"] == case["messages"]
        assert payload["max_tokens"] == 512
        assert payload["seed"] == 20261023
        response_format = payload["response_format"]
        assert response_format["type"] == "json_schema"
        assert response_format["json_schema"]["strict"] is True
        assert response_format["json_schema"]["schema"] == case["public_schema"]
        assert "expected" not in payload
        assert "required_keys" not in payload


def test_schema_validator_rejects_answer_specific_or_key_mismatched_contracts():
    manifest = worker.read_json(worker.DEFAULT_STUDY / "manifest.json")
    singleton = copy.deepcopy(manifest)
    singleton["cases"][0]["public_schema"]["properties"]["route"]["enum"] = [
        singleton["cases"][0]["expected"]["route"]
    ]
    with pytest.raises(worker.RoleStudyError, match="must contain alternatives"):
        worker.validate_manifest(singleton)

    hidden_const = copy.deepcopy(manifest)
    hidden_const["cases"][0]["public_schema"]["properties"]["route"]["const"] = (
        hidden_const["cases"][0]["expected"]["route"]
    )
    with pytest.raises(worker.RoleStudyError, match="unsupported keywords"):
        worker.validate_manifest(hidden_const)

    wrong_keys = copy.deepcopy(manifest)
    wrong_keys["cases"][0]["public_schema"]["properties"]["other"] = {
        "type": "string"
    }
    wrong_keys["cases"][0]["public_schema"]["required"].append("other")
    with pytest.raises(worker.RoleStudyError, match="public schema keys differ"):
        worker.validate_manifest(wrong_keys)


def test_published_manifest_plan_and_recipe_guards():
    cases, models = published()
    assert len(cases) == 24
    assert [model["key"] for model in models] == list(worker.MODEL_ORDER)
    by_key = {model["key"]: model for model in models}
    assert "chat_template_kwargs" not in by_key["lfm25-12b-instruct"]["recipe"]
    assert by_key["qwen35-08b"]["recipe"]["chat_template_kwargs"] == {
        "enable_thinking": False
    }
    assert by_key["lfm25-26b"]["recipe"]["reasoning_effort"] == "low"
    assert by_key["lfm25-26b"]["recipe"]["reasoning_budget_tokens"] == 0
    assert by_key["lfm25-26b"]["recipe"]["chat_template_kwargs"] == {
        "enable_thinking": True
    }

    reordered = worker.read_json(worker.DEFAULT_PLAN)
    reordered["models"][0], reordered["models"][1] = (
        reordered["models"][1], reordered["models"][0]
    )
    with pytest.raises(worker.RoleStudyError, match="model plan order"):
        worker.validate_model_plan(reordered)


def test_resume_binding_rejects_recipe_or_manifest_change(tmp_path):
    _, models = published()
    controller = worker.Controller(args(tmp_path))
    controller.manifest_sha = "a" * 64
    controller.plan_sha = "b" * 64
    first = controller.ensure_binding(models[0])
    assert controller.ensure_binding(models[0])["settings_sha256"] == first[
        "settings_sha256"
    ]
    changed = copy.deepcopy(models[0])
    changed["recipe"]["temperature"] = 0.9
    with pytest.raises(worker.RoleStudyError, match="resume binding differs"):
        controller.ensure_binding(changed)
    controller.manifest_sha = "c" * 64
    with pytest.raises(worker.RoleStudyError, match="resume binding differs"):
        controller.ensure_binding(models[0])


def test_process_and_server_ownership_guards():
    receipt = {"pid": 12, "start_ticks": 99,
               "argv": ["/runtime/server", "-m", "x"]}
    identity = {**receipt, "alive": True, "state": "S"}
    assert worker.verify_owned(receipt, identity, "fixture") is True
    with pytest.raises(worker.RoleStudyError, match="start_ticks"):
        worker.verify_owned(receipt, {**identity, "start_ticks": 100}, "fixture")
    with pytest.raises(worker.RoleStudyError, match="argv"):
        worker.verify_owned(receipt, {**identity, "argv": ["/tmp/server"]}, "fixture")
    assert worker.verify_owned(receipt, {"alive": False, "state": "Z"}, "fixture") is False

    _, models = published()
    model = models[0]
    argv = worker.server_command(model)[3:]
    effort = argv.index("--reasoning-effort")
    pair = argv[effort : effort + 2]
    del argv[effort : effort + 2]
    argv.extend(pair)
    worker.validate_server({"argv": argv, "alias": model["alias"]}, model)
    with pytest.raises(worker.RoleStudyError, match="server argv differs"):
        worker.validate_server(
            {"argv": ["/tmp/llama-server", *argv[1:]], "alias": model["alias"]},
            model,
        )


def test_published_parent_handoff_identifies_lfm26_not_first_plan_model():
    _, models = published()
    controller = worker.Controller(args(worker.DEFAULT_STUDY))
    receipt = controller.recovery_handoff(models, require_live=False)
    assert receipt["alias"] == "tachyon-lfm25-26b"
    assert receipt["alias"] != models[0]["alias"]
    assert receipt == worker.read_json(worker.HANDOFF_SERVER_RECEIPT)


def test_first_model_stops_alias_matched_handoff_before_loading_small_model(
    tmp_path
):
    cases, models = published()
    first = models[0]
    inherited_model = models[-1]
    inherited = {
        "pid": 222,
        "start_ticks": 333,
        "argv": worker.server_command(inherited_model)[3:],
        "alias": inherited_model["alias"],
        "model_key": inherited_model["key"],
    }
    launched = {
        "pid": 444,
        "start_ticks": 555,
        "argv": worker.server_command(first)[3:],
        "alias": first["alias"],
        "model_key": first["key"],
    }
    controller = worker.Controller(args(tmp_path))
    controller.manifest_sha = worker.MANIFEST_SHA256
    controller.plan_sha = worker.PLAN_SHA256
    controller.models = models
    controller.cases = cases[:1]
    controller.state = {"models": {}}
    controller.inherited_server = inherited
    controller.current = inherited
    stopped = []
    started = []

    controller.remote_json_file = lambda _path: None
    controller.remote_process = lambda pid: {
        **(inherited if pid == 222 else launched), "alive": True, "state": "S"
    }

    def stop_current(model, label):
        stopped.append((model["key"], label))
        controller.current = None

    def launch_server(model, label):
        started.append((model["key"], label))
        controller.current = launched
        return launched

    def run_case(model, case, binding):
        result = {
            "status": "completed", "experiment": "production_public_schema",
            "case_id": case["id"], "suite": case["suite"],
            "manifest_sha256": controller.manifest_sha,
            "settings_sha256": binding["settings_sha256"],
            "evaluation": worker.evaluate_answer(case, '{"route":"local_fast"}'),
        }
        path = controller.case_dir(model, case) / "result.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        worker.atomic_json(path, result)
        return result

    controller.stop_current = stop_current
    controller.launch_server = launch_server
    controller.wait_endpoint = lambda *_args: None
    controller.run_case = run_case
    controller.snapshot = lambda model, _receipt: worker.atomic_json(
        tmp_path / f"{model['key']}-resource-snapshot.json", {}
    )
    controller.write_model_status = lambda *_args: None
    controller.save = lambda **_fields: None
    controller.event = lambda *_args, **_fields: None

    controller.run_model(first)
    assert stopped[0] == ("lfm25-26b", "resume-current")
    assert started == [("lfm25-12b-instruct", "lfm25-12b-instruct")]
    assert stopped[-1] == ("lfm25-12b-instruct", "lfm25-12b-instruct")


def test_failure_cleanup_verifies_default_with_interrupt_temporarily_cleared(
    tmp_path, capsys, monkeypatch
):
    _, models = published()
    controller = worker.Controller(args(tmp_path))
    monkeypatch.setattr(worker.signal, "signal", lambda *_args: None)
    controller.acquire_lock = lambda: None
    restored_with = []
    saved_states = []

    def fail_after_interrupt():
        controller.models = models
        controller.state = {"schema_version": 1, "kind": worker.STUDY_KIND,
                            "models": {}}
        controller.interrupted = 15
        raise worker.RoleStudyError("primary interrupted failure")

    controller.wait_for_gpu = fail_after_interrupt
    controller.restore_default = lambda: restored_with.append(controller.interrupted)
    controller.save = lambda **fields: saved_states.append(fields)
    controller.event = lambda *_args, **_kwargs: None

    assert controller.run() == 1
    assert restored_with == [None]
    assert controller.interrupted == 15
    assert saved_states[-1]["status"] == "failed"
    assert saved_states[-1]["error"] == "primary interrupted failure"
    assert "primary interrupted failure" in capsys.readouterr().err


def test_direct_script_entrypoint_has_only_production_input_options():
    script = Path(__file__).parents[1] / "runtime/tachyon_worker_study.py"
    completed = subprocess.run(
        [sys.executable, str(script), "--help"], cwd="/tmp",
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--model-plan" in completed.stdout
    assert "--handoff-receipt" in completed.stdout
    assert "--guided-probe-plan" not in completed.stdout
    assert "--remote" not in completed.stdout
    assert "--endpoint" not in completed.stdout

