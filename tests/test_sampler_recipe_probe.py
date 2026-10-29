import argparse
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import bench
import bench_cases
from runtime import budget_prompt_probe, sampler_recipe_probe, specialist_cases


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def held_events(
    path, *, tag="study-tag", run_id="model-session", terminal=None, backend="llama",
    context=16384,
):
    rows = [
        {
            "at": "2026-10-19T00:00:00Z", "event": "session_started",
            "run_id": run_id, "tag": tag, "runtime_backend": backend,
            "model_name": "resident-alias", "context": context,
            "source_config": "/tmp/native-source.json",
        },
        {
            "at": "2026-10-19T00:01:00Z", "event": "holding_started",
            "run_id": run_id, "tag": tag, "control_file": "/tmp/control",
        },
    ]
    if terminal:
        rows.append({
            "at": "2026-10-19T00:02:00Z", "event": terminal,
            "run_id": run_id, "tag": tag,
        })
    write_jsonl(path, rows)


def fake_http(method, url, **kwargs):
    if url.endswith("/models"):
        return {"data": [{"id": "resident-alias"}]}
    if url.endswith("/props"):
        return {
            "model_alias": "resident-alias",
            "default_generation_settings": {
                "n_ctx": 16384,
                "params": {
                    "temperature": 0.8, "top_p": 0.9, "top_k": 40,
                    "reasoning_budget_tokens": -1,
                },
            },
        }
    if url.endswith("/tokenize"):
        return {"tokens": list(range(32))}
    raise AssertionError((method, url, kwargs))


def fake_call(text="fabricated visible final"):
    return {
        "response_text": text,
        "reasoning_chars": 17,
        "timing": {
            "wall_seconds": 3.0, "ttft_reasoning_seconds": 0.2,
            "ttft_final_content_seconds": 0.8,
        },
        "openai_usage": {"prompt_tokens": 100, "completion_tokens": 20},
        "openai_native_timings": {
            "prompt_n": 100, "cache_n": 0, "predicted_n": 19, "predicted_ms": 2000,
        },
        "actual_prompt_tokens": 100, "actual_completion_tokens": 20,
        "done_reason": "stop", "stream_complete": True, "truncated": False,
        "throughput": {
            "completion_tokens_per_second_native": 9.5,
            "completion_tokens_per_second_wall": 6.7,
        },
    }


def valid_ledger_schema():
    return {
        "type": "object",
        "properties": {
            "balances": {
                "type": "object",
                "properties": {"A": {"type": "integer"}},
                "required": ["A"],
                "additionalProperties": False,
            },
            "transfer_ids": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["balances", "transfer_ids"],
        "additionalProperties": False,
    }


def args_for(
    root, recipe, *, groups="ledger,code,prose", cases=None, prose="none",
    code="medium", thinking_budget_tokens=None, ledger_json_schema=None,
    expected_context=16384, output_cap=8192, timeout=2400.0,
):
    return argparse.Namespace(
        recipe=str(recipe), runtime_events=str(root / "events.jsonl"), tag="study-tag",
        model_session_run_id="model-session", endpoint="http://127.0.0.1:8080/v1",
        model="resident-alias", output_prefix=str(root / "study-tag-recipe"),
        profile="recipe-profile", groups=groups, cases=cases, prose_thinking=prose,
        code_thinking=code,
        expected_context=expected_context, output_cap=output_cap,
        timeout=timeout, holding_timeout=1.0,
        thinking_budget_tokens=thinking_budget_tokens,
        ledger_json_schema=ledger_json_schema,
    )


class SamplerRecipeProbeTests(unittest.TestCase):
    def test_recipe_validation_rejects_unknown_nonfinite_and_unjustified_full_claim(self):
        base = {
            "schema_version": 1, "label": "publisher inspired",
            "claim": "publisher_inspired", "sources": ["https://example.test/card"],
            "sampler_overrides": {"temperature": 1.0, "top_p": 0.95, "top_k": 20},
        }
        valid = sampler_recipe_probe.validate_recipe(base)
        self.assertEqual(valid["sampler_overrides"]["top_k"], 20)
        self.assertFalse(valid["allowlisted_recipe_complete"])
        self.assertIn("min_p", valid["unspecified_fields"])

        for changed in (
            base | {"sampler_overrides": {"dry_multiplier": 0.8}},
            base | {"sampler_overrides": {"temperature": float("nan")}},
            base | {"sampler_overrides": {"top_k": 2.5}},
            base | {"claim": "full_allowlisted_recipe"},
            base | {"extra": True},
        ):
            with self.subTest(changed=changed):
                with self.assertRaises(sampler_recipe_probe.ProbeError):
                    sampler_recipe_probe.validate_recipe(changed)

    def test_scoped_payload_adapter_sends_every_explicit_knob_and_restores_builder(self):
        case = bench_cases.get_cases("hard", [sampler_recipe_probe.LEDGER_CASE_ID])[0]
        config = bench.RunConfig(
            endpoint="http://127.0.0.1:8080/v1", backend="openai", model="resident",
            context=16384, thinking="medium", temperature=0.0, seed=0,
            num_predict=8192, timeout=2400, chat_template_thinking="on",
        )
        original = bench.build_openai_payload
        overrides = {
            "temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.05,
            "presence_penalty": 1.5, "repeat_penalty": 1.0,
        }
        schema = valid_ledger_schema()
        with sampler_recipe_probe.sampler_payload_overrides(
            overrides, thinking_budget_tokens=1024, ledger_json_schema=schema,
        ):
            payload = bench.build_openai_payload(config, case)
        self.assertIs(bench.build_openai_payload, original)
        for key, value in overrides.items():
            self.assertEqual(payload[key], value)
        self.assertEqual(payload["thinking_budget_tokens"], 1024)
        self.assertEqual(payload["response_format"], {
            "type": "json_schema",
            "json_schema": {
                "name": "ledger_output", "strict": True, "schema": schema,
            },
        })

        with sampler_recipe_probe.sampler_payload_overrides({"top_k": 20}):
            inherited = bench.build_openai_payload(config, case)
        self.assertNotIn("temperature", inherited)
        self.assertEqual(inherited["top_k"], 20)
        self.assertNotIn("thinking_budget_tokens", inherited)
        self.assertNotIn("response_format", inherited)

        with self.assertRaisesRegex(RuntimeError, "fixture"):
            with sampler_recipe_probe.sampler_payload_overrides(
                {"top_k": 20}, thinking_budget_tokens=1,
                ledger_json_schema=schema,
            ):
                raise RuntimeError("fixture")
        self.assertIs(bench.build_openai_payload, original)

    def test_ledger_schema_and_native_budget_compose_in_payload_and_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            held_events(root / "events.jsonl")
            recipe_path = root / "recipe.json"
            schema_path = root / "ledger.schema.json"
            write_json(recipe_path, {
                "label": "schema fixture", "sources": [],
                "sampler_overrides": {"temperature": 0.0},
            })
            write_json(schema_path, valid_ledger_schema())
            args = args_for(
                root, recipe_path, groups="ledger", prose=None,
                thinking_budget_tokens=1024, ledger_json_schema=str(schema_path),
            )
            sent = []

            def run_request(config, case):
                sent.append(bench.build_openai_payload(config, case))
                return fake_call('{"fixture":"shape only"}')

            with mock.patch.object(budget_prompt_probe, "_request_json", side_effect=fake_http), \
                    mock.patch.object(bench, "run_request", side_effect=run_request), \
                    mock.patch.object(bench_cases, "evaluate_case", return_value={
                        "passed": False, "detail": "fixture",
                    }):
                self.assertEqual(sampler_recipe_probe.run(args), 0)

            schema_sha = hashlib.sha256(schema_path.read_bytes()).hexdigest()
            self.assertEqual(len(sent), 1)
            self.assertEqual(sent[0]["thinking_budget_tokens"], 1024)
            self.assertEqual(sent[0]["response_format"], {
                "type": "json_schema",
                "json_schema": {
                    "name": "ledger_output", "strict": True,
                    "schema": valid_ledger_schema(),
                },
            })
            paths = sampler_recipe_probe._artifact_paths(Path(args.output_prefix))
            row = read_jsonl(paths["requests"])[0]
            manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
            receipt = json.loads(paths["receipt"].read_text(encoding="utf-8"))
            for controls in (row["request"], manifest["controls"], receipt["controls"]):
                self.assertEqual(controls["ledger_json_schema_path"], str(schema_path.resolve()))
                self.assertEqual(controls["ledger_json_schema_sha256"], schema_sha)
                self.assertEqual(controls["ledger_json_schema_requested"], {
                    "response_format_type": "json_schema",
                    "name": "ledger_output",
                    "strict": True,
                    "application_evidence": (
                        sampler_recipe_probe.LEDGER_JSON_SCHEMA_APPLICATION_EVIDENCE
                    ),
                })
            self.assertEqual(
                receipt["attempts"][0]["ledger_json_schema_sha256"], schema_sha,
            )
            self.assertIn("does not establish", row["request"]["capability_note"])

    def test_full_probe_uses_frozen_cases_graders_and_writes_safe_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            events = root / "events.jsonl"
            held_events(events)
            recipe_path = root / "recipe.json"
            recipe_doc = {
                "schema_version": 1, "label": "complete recipe",
                "claim": "full_allowlisted_recipe", "sources": ["https://example.test/card"],
                "sampler_overrides": {
                    "temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.05,
                    "presence_penalty": 0.0, "repeat_penalty": 1.0,
                },
            }
            write_json(recipe_path, recipe_doc)
            args = args_for(root, recipe_path, thinking_budget_tokens=1024)
            sent = []
            specialist_seen = []
            ledger_seen = []

            def run_request(config, case):
                sent.append(bench.build_openai_payload(config, case))
                call = fake_call()
                call["reasoning_text"] = "SECRET_RAW_REASONING"
                return call

            def specialist_evaluate(case, text, *, truncated):
                specialist_seen.append(case)
                if case.group == "code":
                    assertions = sum(len(item.assertions) for item in case.code_objectives)
                    return {
                        "outcome": "passed", "passed": True,
                        "objective_count": len(case.code_objectives),
                        "objectives_passed": len(case.code_objectives),
                        "assertion_count": assertions, "assertions_passed": assertions,
                    }
                return {
                    "outcome": "constraints_passed", "passed": True,
                    "deterministic_constraints_only": True,
                    "subjective_quality_status": "unscored_pending_blinded_review",
                }

            def ledger_evaluate(case, text, *, truncated):
                ledger_seen.append(case)
                return {"passed": True, "detail": "fixture"}

            original_builder = bench.build_openai_payload
            with mock.patch.object(budget_prompt_probe, "_request_json", side_effect=fake_http), \
                    mock.patch.object(bench, "run_request", side_effect=run_request), \
                    mock.patch.object(specialist_cases, "evaluate_case", side_effect=specialist_evaluate), \
                    mock.patch.object(bench_cases, "evaluate_case", side_effect=ledger_evaluate):
                exit_code = sampler_recipe_probe.run(args)

            self.assertEqual(exit_code, 0)
            self.assertIs(bench.build_openai_payload, original_builder)
            self.assertEqual(len(sent), 7)
            for payload in sent:
                for key, value in recipe_doc["sampler_overrides"].items():
                    self.assertEqual(payload[key], value)
                self.assertEqual(payload["seed"], 0)
                self.assertEqual(payload["max_tokens"], 8192)
                self.assertEqual(payload["thinking_budget_tokens"], 1024)
            self.assertEqual(ledger_seen[0].case_id, sampler_recipe_probe.LEDGER_CASE_ID)
            self.assertIs(ledger_seen[0], bench_cases.get_cases("hard", [sampler_recipe_probe.LEDGER_CASE_ID])[0])
            self.assertEqual(specialist_seen, list(specialist_cases.CASES))

            paths = sampler_recipe_probe._artifact_paths(Path(args.output_prefix))
            rows = read_jsonl(paths["requests"])
            manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
            receipt = json.loads(paths["receipt"].read_text(encoding="utf-8"))
            self.assertEqual(len(rows), 7)
            self.assertEqual(manifest["case_count"], 7)
            self.assertEqual(manifest["independent_task_count"], 4)
            self.assertEqual(manifest["prose_prompt_count"], 3)
            self.assertEqual(receipt["attempted_count"], 7)
            self.assertEqual(receipt["completed_count"], 7)
            for document in (manifest, receipt):
                self.assertEqual(document["controls"]["ledger_and_code_thinking"], "medium")
                self.assertNotIn("ledger_thinking", document["controls"])
                self.assertNotIn("code_thinking", document["controls"])
                self.assertEqual(
                    document["controls"]["native_thinking_budget_tokens_requested"], 1024,
                )
                self.assertEqual(
                    document["controls"]["native_thinking_budget_application_evidence"],
                    sampler_recipe_probe.NATIVE_THINKING_BUDGET_APPLICATION_EVIDENCE,
                )
            self.assertEqual(receipt["resident"]["sampler_defaults_before_override"]["min_p"], {
                "present": False, "value": None,
            })
            self.assertEqual(
                receipt["resident"]["native_thinking_budget_props_evidence"]
                ["reported_default_parameters"],
                {"reasoning_budget_tokens": -1},
            )
            self.assertIn(
                "does not prove",
                receipt["resident"]["native_thinking_budget_props_evidence"]["evidence_scope"],
            )
            expected_sha = hashlib.sha256(recipe_path.read_bytes()).hexdigest()
            self.assertEqual(receipt["recipe"]["sha256"], expected_sha)
            for row in rows:
                self.assertEqual(row["request"]["sampler_overrides"], recipe_doc["sampler_overrides"])
                self.assertEqual(row["request"]["sampler_recipe_sha256"], expected_sha)
                self.assertEqual(row["request"]["native_thinking_budget_tokens_requested"], 1024)
                self.assertEqual(
                    row["request"]["reasoning_controls_requested"]["thinking_budget_tokens"], 1024,
                )
                self.assertEqual(
                    row["request"]["native_thinking_budget_application_evidence"],
                    sampler_recipe_probe.NATIVE_THINKING_BUDGET_APPLICATION_EVIDENCE,
                )
                self.assertNotIn("reasoning_text", json.dumps(row))
                self.assertNotIn("SECRET_RAW_REASONING", json.dumps(row))
            specialist_rows = [row for row in rows if row["suite"].startswith("specialist_")]
            self.assertEqual(len(specialist_rows), 6)
            self.assertTrue(all("response_text" not in row["response"] for row in specialist_rows))
            visible_files = sorted(paths["visible_dir"].glob("*.txt"))
            self.assertEqual(len(visible_files), 6)
            self.assertTrue(all(path.read_text(encoding="utf-8") == "fabricated visible final" for path in visible_files))

    def test_code_none_changes_only_code_reasoning_and_records_separate_controls(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            held_events(root / "events.jsonl")
            recipe_path = root / "recipe.json"
            write_json(recipe_path, {
                "label": "code-none fixture", "sources": [],
                "sampler_overrides": {"temperature": 0.7},
            })
            cases = ",".join((
                sampler_recipe_probe.LEDGER_CASE_ID,
                "code_edge_reading_pivot",
                "prose_grounded_public_briefing",
            ))
            args = args_for(
                root, recipe_path, groups="ledger,code,prose", cases=cases,
                prose="low", code="none", thinking_budget_tokens=512,
            )
            sent = {}

            def run_request(config, case):
                sent[case.case_id] = bench.build_openai_payload(config, case)
                return fake_call()

            with mock.patch.object(budget_prompt_probe, "_request_json", side_effect=fake_http), \
                    mock.patch.object(bench, "run_request", side_effect=run_request), \
                    mock.patch.object(specialist_cases, "evaluate_case", return_value={
                        "outcome": "passed", "passed": True,
                    }), \
                    mock.patch.object(bench_cases, "evaluate_case", return_value={
                        "passed": True, "detail": "fixture",
                    }):
                self.assertEqual(sampler_recipe_probe.run(args), 0)

            self.assertEqual(sent[sampler_recipe_probe.LEDGER_CASE_ID]["reasoning_effort"], "medium")
            self.assertTrue(
                sent[sampler_recipe_probe.LEDGER_CASE_ID]["chat_template_kwargs"]["enable_thinking"]
            )
            self.assertEqual(sent["code_edge_reading_pivot"]["reasoning_effort"], "none")
            self.assertFalse(
                sent["code_edge_reading_pivot"]["chat_template_kwargs"]["enable_thinking"]
            )
            self.assertEqual(sent["code_edge_reading_pivot"]["thinking_budget_tokens"], 512)
            self.assertEqual(sent["prose_grounded_public_briefing"]["reasoning_effort"], "low")
            self.assertTrue(
                sent["prose_grounded_public_briefing"]["chat_template_kwargs"]["enable_thinking"]
            )

            paths = sampler_recipe_probe._artifact_paths(Path(args.output_prefix))
            rows = read_jsonl(paths["requests"])
            manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
            receipt = json.loads(paths["receipt"].read_text(encoding="utf-8"))
            self.assertEqual(
                {row.get("case_group", "ledger"): row["request"]["thinking"] for row in rows},
                {"ledger": "medium", "code": "none", "prose": "low"},
            )
            rows_by_group = {row.get("case_group", "ledger"): row for row in rows}
            self.assertEqual(rows_by_group["code"]["request"]["chat_template_thinking"], "off")
            self.assertEqual(
                rows_by_group["code"]["request"]["reasoning_controls_requested"],
                {
                    "reasoning_effort": "none",
                    "chat_template_thinking": "off",
                    "reasoning_budget_tokens": None,
                    "thinking_budget_tokens": 512,
                },
            )
            self.assertEqual(rows_by_group["ledger"]["request"]["thinking"], "medium")
            self.assertEqual(rows_by_group["prose"]["request"]["thinking"], "low")
            for document in (manifest, receipt):
                self.assertEqual(document["controls"]["ledger_thinking"], "medium")
                self.assertEqual(document["controls"]["code_thinking"], "none")
                self.assertEqual(document["controls"]["prose_thinking"], "low")
                self.assertNotIn("ledger_and_code_thinking", document["controls"])

    def test_selected_partial_failures_are_preserved_and_empty_answer_is_not_completed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            held_events(root / "events.jsonl")
            recipe_path = root / "recipe.json"
            write_json(recipe_path, {
                "label": "top-k experiment", "claim": "custom_experiment", "sources": [],
                "sampler_overrides": {"top_k": 20},
            })
            cases = "prose_grounded_public_briefing,prose_constrained_last_tram"
            args = args_for(root, recipe_path, groups="prose", cases=cases, prose="low")
            calls = [fake_call("partial visible answer"), TimeoutError("fixture timeout")]

            def run_request(config, case):
                value = calls.pop(0)
                if isinstance(value, BaseException):
                    raise value
                return value

            with mock.patch.object(budget_prompt_probe, "_request_json", side_effect=fake_http), \
                    mock.patch.object(bench, "run_request", side_effect=run_request), \
                    mock.patch.object(specialist_cases, "evaluate_case", return_value={
                        "outcome": "constraints_passed", "passed": True,
                    }):
                self.assertEqual(sampler_recipe_probe.run(args), 1)

            paths = sampler_recipe_probe._artifact_paths(Path(args.output_prefix))
            rows = read_jsonl(paths["requests"])
            receipt = json.loads(paths["receipt"].read_text(encoding="utf-8"))
            self.assertEqual(len(rows), 2)
            self.assertEqual(receipt["status"], "completed_with_request_failures")
            self.assertEqual(receipt["attempted_count"], 2)
            self.assertEqual(receipt["completed_count"], 1)
            self.assertEqual(receipt["request_failure_count"], 1)
            self.assertIsNone(rows[1]["response"])
            self.assertFalse(rows[1]["eligibility"]["blinded_prose_review_eligible"])
            self.assertEqual(rows[1]["visible_text"]["chars"], 0)

    def test_interrupt_writes_partial_receipt_and_restores_payload_builder(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            held_events(root / "events.jsonl")
            recipe_path = root / "recipe.json"
            write_json(recipe_path, {
                "label": "interrupt fixture", "sources": [],
                "sampler_overrides": {"temperature": 0.7},
            })
            args = args_for(root, recipe_path, groups="ledger", prose=None)
            original = bench.build_openai_payload
            with mock.patch.object(budget_prompt_probe, "_request_json", side_effect=fake_http), \
                    mock.patch.object(bench, "run_request", side_effect=KeyboardInterrupt()):
                with self.assertRaises(KeyboardInterrupt):
                    sampler_recipe_probe.run(args)
            self.assertIs(bench.build_openai_payload, original)
            receipt = json.loads(
                sampler_recipe_probe._artifact_paths(Path(args.output_prefix))["receipt"].read_text()
            )
            self.assertEqual(receipt["status"], "interrupted")
            self.assertEqual(receipt["attempted_count"], 1)
            self.assertEqual(receipt["in_flight_case"]["case_id"], sampler_recipe_probe.LEDGER_CASE_ID)

    def test_terminal_lifecycle_and_missing_tokenizer_fail_closed_before_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recipe_path = root / "recipe.json"
            write_json(recipe_path, {"label": "fixture", "sources": [], "sampler_overrides": {"top_k": 20}})
            args = args_for(root, recipe_path, groups="ledger", prose=None)
            held_events(root / "events.jsonl", terminal="holding_stopped")
            with self.assertRaisesRegex(sampler_recipe_probe.ProbeError, "no longer"):
                sampler_recipe_probe.run(args)
            self.assertFalse(sampler_recipe_probe._artifact_paths(Path(args.output_prefix))["manifest"].exists())

            held_events(root / "events.jsonl", backend="strata")
            with self.assertRaisesRegex(sampler_recipe_probe.ProbeError, "native llama"):
                sampler_recipe_probe.run(args)

            held_events(root / "events.jsonl")

            def unavailable(method, url, **kwargs):
                if url.endswith("/tokenize"):
                    raise budget_prompt_probe.ProbeHTTPError(404, "tokenize")
                return fake_http(method, url, **kwargs)

            with mock.patch.object(budget_prompt_probe, "_request_json", side_effect=unavailable):
                with self.assertRaisesRegex(sampler_recipe_probe.ProbeError, "exact.*preflight"):
                    sampler_recipe_probe.run(args)
            self.assertFalse(sampler_recipe_probe._artifact_paths(Path(args.output_prefix))["manifest"].exists())

            def does_not_fit(method, url, **kwargs):
                if url.endswith("/tokenize"):
                    return {"tokens": list(range(8200))}
                return fake_http(method, url, **kwargs)

            with mock.patch.object(budget_prompt_probe, "_request_json", side_effect=does_not_fit):
                with self.assertRaisesRegex(sampler_recipe_probe.ProbeError, "does not fit"):
                    sampler_recipe_probe.run(args)

    def test_wide_context_is_exact_across_lifecycle_props_fit_and_request(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            held_events(root / "events.jsonl", context=98304)
            recipe_path = root / "recipe.json"
            write_json(recipe_path, {
                "label": "wide fixture", "sources": [],
                "sampler_overrides": {"temperature": 0.0},
            })
            args = args_for(
                root, recipe_path, groups="code", cases="code_edge_reading_pivot",
                expected_context=98304, output_cap=65536, timeout=21600.0,
            )
            seen = []

            def wide_http(method, url, **kwargs):
                if url.endswith("/props"):
                    return {
                        "model_alias": "resident-alias",
                        "default_generation_settings": {"n_ctx": 98304, "params": {}},
                    }
                if url.endswith("/tokenize"):
                    return {"tokens": list(range(32000))}
                return fake_http(method, url, **kwargs)

            def run_request(config, _case):
                seen.append(config)
                return fake_call()

            with mock.patch.object(
                budget_prompt_probe, "_request_json", side_effect=wide_http
            ), mock.patch.object(
                bench, "run_request", side_effect=run_request
            ), mock.patch.object(
                specialist_cases, "evaluate_case", return_value={
                    "outcome": "passed", "passed": True,
                }
            ):
                self.assertEqual(sampler_recipe_probe.run(args), 0)

            receipt = json.loads(
                sampler_recipe_probe._artifact_paths(Path(args.output_prefix))["receipt"].read_text()
            )
            manifest = json.loads(
                sampler_recipe_probe._artifact_paths(Path(args.output_prefix))["manifest"].read_text()
            )
            self.assertEqual(seen[0].context, 98304)
            self.assertEqual(receipt["controls"]["expected_context_tokens"], 98304)
            self.assertEqual(receipt["controls"]["actual_context_tokens"], 98304)
            self.assertEqual(receipt["resident"]["actual_context_tokens"], 98304)
            self.assertTrue(manifest["prompt_feasibility"]["checks"][0]["fits"])

            mismatched = args_for(
                root, recipe_path, groups="code", cases="code_edge_reading_pivot",
                expected_context=131072,
            )
            mismatched.output_prefix = str(root / "study-tag-mismatch")
            with mock.patch.object(
                budget_prompt_probe, "_request_json", side_effect=wide_http
            ), self.assertRaisesRegex(sampler_recipe_probe.ProbeError, "context"):
                sampler_recipe_probe.run(mismatched)
            self.assertFalse(
                sampler_recipe_probe._artifact_paths(Path(mismatched.output_prefix))["manifest"].exists()
            )

    def test_cli_cap_timeout_and_case_selection_bounds(self):
        parser = sampler_recipe_probe.build_parser()
        required = [
            "--recipe", "r.json", "--runtime-events", "events.jsonl", "--tag", "tag",
            "--model-session-run-id", "run", "--endpoint", "http://127.0.0.1:1/v1",
            "--model", "model", "--output-prefix", "tag-probe", "--groups", "code",
            "--cases", "code_edge_reading_pivot", "--output-cap", "12288", "--timeout", "3000",
            "--thinking-budget-tokens", "1",
            "--ledger-json-schema", "ledger.schema.json",
        ]
        parsed = parser.parse_args(required)
        self.assertEqual(parsed.output_cap, 12288)
        self.assertEqual(parsed.timeout, 3000)
        self.assertEqual(parsed.thinking_budget_tokens, 1)
        self.assertEqual(parsed.ledger_json_schema, "ledger.schema.json")
        self.assertEqual(parsed.code_thinking, "medium")
        defaults = parser.parse_args(required[:-8])
        self.assertEqual(defaults.output_cap, 8192)
        self.assertEqual(defaults.timeout, 2400)
        self.assertIsNone(defaults.thinking_budget_tokens)
        self.assertIsNone(defaults.ledger_json_schema)
        self.assertEqual(defaults.code_thinking, "medium")
        self.assertEqual(defaults.expected_context, 16384)
        wide = parser.parse_args(required[:-8] + [
            "--expected-context", "131072", "--output-cap", "65536",
            "--timeout", "21600",
        ])
        self.assertEqual(wide.expected_context, 131072)
        self.assertEqual(wide.output_cap, 65536)
        self.assertEqual(wide.timeout, 21600)
        code_none = parser.parse_args(required[:-8] + ["--code-thinking", "none"])
        self.assertEqual(code_none.code_thinking, "none")
        selected = sampler_recipe_probe.selected_cases(parsed.groups, parsed.cases)
        self.assertEqual([item.case.case_id for item in selected], ["code_edge_reading_pivot"])
        with self.assertRaises(SystemExit):
            parser.parse_args(required[:-8] + ["--output-cap", "65537"])
        with self.assertRaises(SystemExit):
            parser.parse_args(required[:-8] + ["--timeout", "21601"])
        with self.assertRaises(SystemExit):
            parser.parse_args(required[:-8] + ["--expected-context", "131073"])
        for invalid in ("0", "-1", "1.5"):
            with self.subTest(invalid=invalid), self.assertRaises(SystemExit):
                parser.parse_args(required[:-8] + ["--thinking-budget-tokens", invalid])
        self.assertEqual(sampler_recipe_probe._validate_thinking_budget(1, 1), 1)
        with self.assertRaisesRegex(sampler_recipe_probe.ProbeError, "must not exceed"):
            sampler_recipe_probe._validate_thinking_budget(12289, 12288)
        with self.assertRaises(sampler_recipe_probe.ProbeError):
            sampler_recipe_probe.selected_cases("code", "prose_grounded_public_briefing")

    def test_budget_above_output_cap_fails_before_any_http_request_or_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recipe_path = root / "recipe.json"
            write_json(recipe_path, {
                "label": "fixture", "sources": [], "sampler_overrides": {"top_k": 20},
            })
            args = args_for(
                root, recipe_path, groups="ledger", prose=None, thinking_budget_tokens=8193,
            )
            with mock.patch.object(budget_prompt_probe, "_request_json") as request_json:
                with self.assertRaisesRegex(sampler_recipe_probe.ProbeError, "must not exceed"):
                    sampler_recipe_probe.run(args)
            request_json.assert_not_called()
            paths = sampler_recipe_probe._artifact_paths(Path(args.output_prefix))
            self.assertFalse(any(path.exists() for path in paths.values()))

    def test_ledger_schema_rejects_unsupported_shapes_and_nonledger_before_http(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recipe_path = root / "recipe.json"
            schema_path = root / "ledger.schema.json"
            write_json(recipe_path, {
                "label": "fixture", "sources": [], "sampler_overrides": {"top_k": 20},
            })
            invalid_schemas = (
                {"type": "object", "properties": {"x": {"type": "string", "enum": ["x"]}},
                 "required": ["x"], "additionalProperties": False},
                {"type": "object", "properties": {"x": {"$ref": "#/$defs/x"}},
                 "required": ["x"], "additionalProperties": False},
                {"type": "object", "properties": {"x": {"type": "string"}},
                 "required": [], "additionalProperties": False},
                {"type": "object", "properties": {"x": {"type": "string"}},
                 "required": ["x"], "additionalProperties": True},
                {"type": "object", "properties": {
                    "x": {"type": "array", "items": {"type": "object"}},
                 }, "required": ["x"], "additionalProperties": False},
            )
            for index, schema in enumerate(invalid_schemas):
                with self.subTest(index=index):
                    write_json(schema_path, schema)
                    args = args_for(
                        root, recipe_path, groups="ledger", prose=None,
                        ledger_json_schema=str(schema_path),
                    )
                    with mock.patch.object(budget_prompt_probe, "_request_json") as request_json:
                        with self.assertRaises(sampler_recipe_probe.ProbeError):
                            sampler_recipe_probe.run(args)
                    request_json.assert_not_called()
                    paths = sampler_recipe_probe._artifact_paths(Path(args.output_prefix))
                    self.assertFalse(any(path.exists() for path in paths.values()))

            for label, raw_schema in (
                ("non-utf8", b"\xff"),
                (
                    "duplicate-key",
                    b'{"type":"object","type":"object","properties":{},'
                    b'"required":[],"additionalProperties":false}',
                ),
            ):
                with self.subTest(label=label):
                    schema_path.write_bytes(raw_schema)
                    args = args_for(
                        root, recipe_path, groups="ledger", prose=None,
                        ledger_json_schema=str(schema_path),
                    )
                    with mock.patch.object(budget_prompt_probe, "_request_json") as request_json:
                        with self.assertRaises(sampler_recipe_probe.ProbeError):
                            sampler_recipe_probe.run(args)
                    request_json.assert_not_called()

            write_json(schema_path, valid_ledger_schema())
            args = args_for(
                root, recipe_path, groups="code", cases="code_edge_reading_pivot", prose=None,
                ledger_json_schema=str(schema_path),
            )
            with mock.patch.object(budget_prompt_probe, "_request_json") as request_json:
                with self.assertRaisesRegex(sampler_recipe_probe.ProbeError, "ledger-only"):
                    sampler_recipe_probe.run(args)
            request_json.assert_not_called()


if __name__ == "__main__":
    unittest.main()
