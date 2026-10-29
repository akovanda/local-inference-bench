import argparse
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import bench
import bench_cases
from runtime import budget_prompt_probe, native_hard_recipe_probe as probe


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def held_events(path, *, context=49152, model="resident-alias"):
    write_jsonl(path, [
        {
            "at": "start", "event": "session_started", "run_id": "session-run",
            "tag": "hard-tag", "runtime_backend": "llama", "model_name": model,
            "context": context, "source_config": "/tmp/config.json",
        },
        {
            "at": "hold", "event": "holding_started", "run_id": "session-run",
            "tag": "hard-tag", "control_file": "/tmp/control.json",
        },
    ])


def recipe(path):
    write_json(path, {
        "schema_version": 1,
        "label": "publisher fixture",
        "claim": "publisher_inspired",
        "sources": ["https://example.test/model-card"],
        "sampler_overrides": {
            "temperature": 0.7, "top_p": 0.9, "top_k": 20, "min_p": 0.05,
        },
    })
    return path


def args_for(root, recipe_path, *, cases="hard_transaction_ledger", context=49152):
    return argparse.Namespace(
        recipe=str(recipe_path), runtime_events=str(root / "events.jsonl"),
        tag="hard-tag", model_session_run_id="session-run",
        endpoint="http://127.0.0.1:18081/v1", model="resident-alias",
        expected_context=context, output_prefix=str(root / "hard-tag-publisher"),
        profile="hard-profile", cases=cases, thinking="medium",
        output_cap=32768, timeout=14400.0, holding_timeout=1.0,
    )


def fake_http(method, url, **kwargs):
    if url.endswith("/models"):
        return {"data": [{"id": "resident-alias"}]}
    if url.endswith("/props"):
        return {
            "model_alias": "resident-alias",
            "default_generation_settings": {
                "n_ctx": 49152,
                "params": {"temperature": 0.2, "top_p": 0.8},
            },
        }
    if url.endswith("/tokenize"):
        return {"tokens": list(range(1000))}
    raise AssertionError((method, url, kwargs))


def fake_call(*, truncated=False, raw_reasoning="secret-chain"):
    return {
        "response_text": "visible hard answer",
        "reasoning_chars": 42,
        "reasoning_content": raw_reasoning,
        "raw_reasoning": raw_reasoning,
        "timing": {
            "wall_seconds": 3.5,
            "ttft_reasoning_seconds": 0.2,
            "ttft_final_content_seconds": 0.9,
        },
        "openai_usage": {"prompt_tokens": 1000, "completion_tokens": 200},
        "openai_native_timings": {"predicted_ms": 2500, "predicted_n": 200},
        "actual_prompt_tokens": 1000,
        "actual_completion_tokens": 200,
        "done_reason": "length" if truncated else "stop",
        "stream_complete": True,
        "truncated": truncated,
        "throughput": {"completion_tokens_per_second_native": 80.0},
    }


class NativeHardRecipeProbeTests(unittest.TestCase):
    def test_case_selection_is_required_allowlisted_unique_and_ordered(self):
        selected = probe.selected_cases("hard_temporal_kv,hard_transaction_ledger")
        self.assertEqual(
            [case.case_id for case in selected],
            ["hard_temporal_kv", "hard_transaction_ledger"],
        )
        for value in ("", "unknown", "hard_temporal_kv,hard_temporal_kv"):
            with self.subTest(value=value), self.assertRaises(probe.HardProbeError):
                probe.selected_cases(value)
        with self.assertRaises(SystemExit):
            probe.build_parser().parse_args([])

    def test_exact_lifecycle_context_alias_and_loopback_fail_before_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recipe_path = recipe(root / "recipe.json")
            held_events(root / "events.jsonl", context=49151)
            args = args_for(root, recipe_path)
            with self.assertRaisesRegex(probe.sampler_recipe_probe.ProbeError, "context"):
                probe.run(args)
            self.assertFalse(probe._artifact_paths(Path(args.output_prefix))["manifest"].exists())

            held_events(root / "events.jsonl")
            args.endpoint = "http://example.com:18081/v1"
            with self.assertRaisesRegex(budget_prompt_probe.ProbeError, "loopback"):
                probe.run(args)

            args.endpoint = "http://127.0.0.1:18081/v1"
            def wrong_alias(method, url, **kwargs):
                if url.endswith("/models"):
                    return {"data": [{"id": "other"}]}
                return fake_http(method, url, **kwargs)
            with mock.patch.object(
                budget_prompt_probe, "_request_json", side_effect=wrong_alias
            ), self.assertRaisesRegex(probe.sampler_recipe_probe.ProbeError, "alias"):
                probe.run(args)

            def wrong_actual_context(method, url, **kwargs):
                value = fake_http(method, url, **kwargs)
                if url.endswith("/props"):
                    value["default_generation_settings"]["n_ctx"] = 98304
                return value

            with mock.patch.object(
                budget_prompt_probe, "_request_json", side_effect=wrong_actual_context
            ), self.assertRaisesRegex(probe.sampler_recipe_probe.ProbeError, "context"):
                probe.run(args)

    def test_payload_mode_metadata_quality_truncation_and_no_raw_reasoning(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recipe_path = recipe(root / "recipe.json")
            held_events(root / "events.jsonl")
            args = args_for(
                root, recipe_path,
                cases="hard_transaction_ledger,hard_dependency_constraints",
            )
            args.thinking = "none"
            payloads = []
            calls = [fake_call(), fake_call(truncated=True)]

            def run_request(config, case):
                payloads.append(bench.build_openai_payload(config, case))
                return calls[len(payloads) - 1]

            evaluations = [
                {"passed": True, "detail": "fixture pass"},
                {"passed": False, "detail": "length", "truncated": True},
            ]
            with mock.patch.object(
                budget_prompt_probe, "_request_json", side_effect=fake_http
            ), mock.patch.object(
                bench, "run_request", side_effect=run_request
            ), mock.patch.object(
                bench_cases, "evaluate_case", side_effect=evaluations
            ), mock.patch.object(
                probe.sampler_recipe_probe,
                "assert_active_hold",
                wraps=probe.sampler_recipe_probe.assert_active_hold,
            ) as active:
                self.assertEqual(probe.run(args), 0)

            self.assertEqual(active.call_count, 5)  # initial, then before/after each request
            self.assertEqual(len(payloads), 2)
            for payload in payloads:
                self.assertEqual(payload["temperature"], 0.7)
                self.assertEqual(payload["top_p"], 0.9)
                self.assertEqual(payload["top_k"], 20)
                self.assertEqual(payload["min_p"], 0.05)
                self.assertEqual(payload["max_tokens"], 32768)
                self.assertEqual(payload["seed"], 0)
                self.assertEqual(payload["reasoning_effort"], "none")
                self.assertEqual(payload["chat_template_kwargs"], {"enable_thinking": False})
                self.assertNotIn("thinking_budget_tokens", payload)

            paths = probe._artifact_paths(Path(args.output_prefix))
            rows = read_jsonl(paths["requests"])
            receipt = json.loads(paths["receipt"].read_text(encoding="utf-8"))
            manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
            self.assertEqual([row["objective_outcome"] for row in rows], ["passed", "truncated"])
            self.assertEqual(receipt["passed_count"], 1)
            self.assertEqual(receipt["truncated_count"], 1)
            self.assertEqual(receipt["wrong_answer_count"], 0)
            self.assertEqual(receipt["completed_count"], 1)
            self.assertEqual(receipt["attempts"][1]["outcome"], "truncated")
            self.assertEqual(receipt["attempts"][0]["input_tokens"], 1000)
            self.assertEqual(receipt["attempts"][0]["output_tokens"], 200)
            self.assertEqual(receipt["attempts"][0]["wall_seconds"], 3.5)
            self.assertEqual(rows[0]["response"]["reasoning_chars"], 42)
            self.assertEqual(
                rows[0]["request"]["sampler_recipe_sha256"],
                hashlib.sha256(recipe_path.read_bytes()).hexdigest(),
            )
            self.assertEqual(manifest["controls"]["thinking"], "none")
            self.assertEqual(manifest["controls"]["output_token_limit"], 32768)
            self.assertNotIn("native_thinking_budget_tokens_requested", manifest["controls"])
            self.assertNotIn("native_thinking_budget_tokens_requested", receipt["controls"])
            self.assertNotIn("native_thinking_budget_tokens_requested", receipt["attempts"][0])
            self.assertNotIn("native_thinking_budget_tokens_requested", rows[0]["request"])
            self.assertNotIn(
                "thinking_budget_tokens",
                rows[0]["request"]["reasoning_controls_requested"],
            )
            for path in paths.values():
                self.assertNotIn("secret-chain", path.read_text(encoding="utf-8"))

    def test_native_thinking_budget_is_sent_and_recorded_separately_from_output_cap(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recipe_path = recipe(root / "recipe.json")
            held_events(root / "events.jsonl")
            args = args_for(root, recipe_path)
            args.thinking_budget_tokens = 4096
            payloads = []

            def run_request(config, case):
                payloads.append(bench.build_openai_payload(config, case))
                return fake_call()

            with mock.patch.object(
                budget_prompt_probe, "_request_json", side_effect=fake_http
            ), mock.patch.object(
                bench, "run_request", side_effect=run_request
            ), mock.patch.object(
                bench_cases, "evaluate_case",
                return_value={"passed": True, "detail": "fixture pass"},
            ):
                self.assertEqual(probe.run(args), 0)

            self.assertEqual(len(payloads), 1)
            self.assertEqual(payloads[0]["thinking_budget_tokens"], 4096)
            self.assertNotIn("reasoning_budget_tokens", payloads[0])
            self.assertEqual(payloads[0]["max_tokens"], 32768)
            paths = probe._artifact_paths(Path(args.output_prefix))
            row = read_jsonl(paths["requests"])[0]
            manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
            receipt = json.loads(paths["receipt"].read_text(encoding="utf-8"))
            evidence = probe.sampler_recipe_probe.NATIVE_THINKING_BUDGET_APPLICATION_EVIDENCE
            self.assertEqual(row["request"]["output_token_limit"], 32768)
            self.assertEqual(row["request"]["native_thinking_budget_tokens_requested"], 4096)
            self.assertEqual(row["request"]["native_thinking_budget_application_evidence"], evidence)
            self.assertEqual(
                row["request"]["reasoning_controls_requested"]["thinking_budget_tokens"],
                4096,
            )
            for document in (manifest, receipt):
                self.assertEqual(
                    document["controls"]["native_thinking_budget_tokens_requested"], 4096
                )
                self.assertEqual(
                    document["controls"]["native_thinking_budget_application_evidence"],
                    evidence,
                )
                self.assertEqual(document["controls"]["output_token_limit"], 32768)
            self.assertEqual(
                receipt["attempts"][0]["native_thinking_budget_tokens_requested"], 4096
            )
            self.assertEqual(
                receipt["attempts"][0]["native_thinking_budget_application_evidence"], evidence
            )

    def test_native_thinking_budget_rejects_none_mode_and_values_above_output_cap(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recipe_path = recipe(root / "recipe.json")
            held_events(root / "events.jsonl")

            none_args = args_for(root, recipe_path)
            none_args.thinking = "none"
            none_args.thinking_budget_tokens = 1
            with self.assertRaisesRegex(probe.HardProbeError, "only with --thinking medium"):
                probe.run(none_args)

            exceeds_args = args_for(root, recipe_path)
            exceeds_args.thinking_budget_tokens = exceeds_args.output_cap + 1
            with self.assertRaisesRegex(probe.HardProbeError, "must not exceed --output-cap"):
                probe.run(exceeds_args)

            self.assertTrue(all(not path.exists() for path in probe._artifact_paths(
                Path(none_args.output_prefix)
            ).values()))

    def test_maximum_caps_context_and_timeout_are_exactly_bounded(self):
        parser = probe.build_parser()
        required = [
            "--recipe", "recipe.json", "--runtime-events", "events.jsonl",
            "--tag", "tag", "--model-session-run-id", "run",
            "--endpoint", "http://127.0.0.1:18081/v1", "--model", "model",
            "--output-prefix", "tag-hard", "--cases", "hard_transaction_ledger",
            "--thinking", "medium",
        ]
        maximum = parser.parse_args(required + [
            "--expected-context", "131072", "--output-cap", "65536",
            "--thinking-budget-tokens", "65536", "--timeout", "21600",
        ])
        self.assertEqual(maximum.expected_context, 131072)
        self.assertEqual(maximum.output_cap, 65536)
        self.assertEqual(maximum.thinking_budget_tokens, 65536)
        self.assertEqual(maximum.timeout, 21600.0)
        self.assertIsNone(parser.parse_args(required).thinking_budget_tokens)
        for option, value in (
            ("--expected-context", "131073"),
            ("--output-cap", "65537"),
            ("--thinking-budget-tokens", "0"),
            ("--thinking-budget-tokens", "65537"),
            ("--timeout", "21601"),
        ):
            with self.subTest(option=option), self.assertRaises(SystemExit):
                parser.parse_args(required + [option, value])


if __name__ == "__main__":
    unittest.main()
