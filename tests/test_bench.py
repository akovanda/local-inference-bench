import dataclasses
import hashlib
import json
import re
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import bench
import bench_cases


class EvaluationTests(unittest.TestCase):
    def test_json_fence_is_allowed_but_types_and_extra_keys_are_strict(self):
        expected = {"count": 1, "items": ["x"]}
        self.assertTrue(
            bench_cases.evaluate_json_exact('```json\n{"count":1,"items":["x"]}\n```', expected)["passed"]
        )
        self.assertFalse(
            bench_cases.evaluate_json_exact('{"count":true,"items":["x"]}', expected)["passed"]
        )
        self.assertFalse(
            bench_cases.evaluate_json_exact('{"count":1,"items":["x"],"extra":0}', expected)["passed"]
        )

    def test_length_truncation_always_fails_before_content_scoring(self):
        case = bench_cases.QUICK_CASES[1]
        perfect = json.dumps(case.expected)
        result = bench_cases.evaluate_case(case, perfect, truncated=True)
        self.assertFalse(result["passed"])
        self.assertTrue(result["truncated"])

    def test_constraint_reference_answer_matches_explicit_chain_contract(self):
        case = next(
            case for case in bench_cases.QUALITY_CASES if case.case_id == "quality_constraints"
        )
        order = case.expected["order"]
        self.assertEqual(order.index("build"), order.index("lint") + 1)
        self.assertLess(order.index("migrate"), order.index("deploy"))
        self.assertLess(order.index("deploy"), order.index("smoke"))
        self.assertEqual(case.expected["dependency_chain"], ["migrate", "deploy", "smoke"])
        self.assertIn(
            "the three strings migrate, deploy, and smoke in that stated dependency order",
            case.prompt,
        )
        self.assertTrue(bench_cases.evaluate_case(case, json.dumps(case.expected))["passed"])

    def test_ast_validation_rejects_imports_dunder_and_dangerous_builtins(self):
        candidates = (
            "import os\ndef target():\n    return 1",
            "def target(x):\n    return x.__class__",
            "def target():\n    return open('/etc/passwd').read()",
            "print('side effect')\ndef target():\n    return 1",
        )
        for source in candidates:
            with self.subTest(source=source):
                valid, errors = bench_cases.validate_python_source(source, "target")
                self.assertFalse(valid)
                self.assertTrue(errors)

    def test_approved_code_runs_in_sandbox_against_hidden_checks(self):
        source = """
def retry_delays(attempts, base, factor, cap):
    if (isinstance(attempts, bool) or not isinstance(attempts, int) or attempts < 0
            or isinstance(base, bool) or not isinstance(base, (int, float)) or base < 0
            or isinstance(factor, bool) or not isinstance(factor, (int, float)) or factor < 1
            or isinstance(cap, bool) or not isinstance(cap, (int, float)) or cap < 0):
        raise ValueError('invalid')
    return [min(cap, base * factor ** i) for i in range(max(0, attempts - 1))]
"""
        case = next(case for case in bench_cases.QUALITY_CASES if case.case_id == "quality_retry_schedule")
        result = bench_cases.run_python_checks(source, case.function_name, case.tests)
        self.assertTrue(result["passed"], result)
        self.assertEqual(result["tests_passed"], len(case.tests))

    def test_import_free_function_can_use_standard_type_and_iterator_builtins(self):
        source = """
def first_integer(values):
    return next(iter(filter(lambda value: type(value) is int, values)), None)
"""
        result = bench_cases.run_python_checks(
            source,
            "first_integer",
            (
                {"args": [[True, "skip", 4, 8]], "expected": 4},
                {"args": [[False, "skip"]], "expected": None},
            ),
        )
        self.assertTrue(result["passed"], result)


class PayloadTests(unittest.TestCase):
    def setUp(self):
        self.case = bench_cases.QUICK_CASES[0]

    def config(self, backend="ollama", model="gpt-oss-20b:64k", thinking="low"):
        return bench.RunConfig(
            endpoint="http://127.0.0.1:11436",
            backend=backend,
            model=model,
            context=32768,
            thinking=thinking,
            temperature=0.0,
            seed=17,
            num_predict=201,
            timeout=180.0,
        )

    def test_ollama_payload_uses_native_options_and_model_thinking_translation(self):
        payload = bench.build_ollama_payload(self.config(), self.case)
        self.assertEqual(payload["think"], "low")
        self.assertEqual(
            payload["options"],
            {"num_ctx": 32768, "num_predict": 201, "temperature": 0.0, "seed": 17},
        )
        self.assertFalse(
            bench.build_ollama_payload(self.config(model="qwen3.6", thinking="none"), self.case)["think"]
        )
        self.assertTrue(
            bench.build_ollama_payload(self.config(model="qwen3.6", thinking="high"), self.case)["think"]
        )

    def test_openai_payload_does_not_leak_ollama_options(self):
        payload = bench.build_openai_payload(self.config(backend="openai", thinking="medium"), self.case)
        self.assertEqual(payload["max_tokens"], 201)
        self.assertEqual(payload["reasoning_effort"], "medium")
        self.assertNotIn("options", payload)
        self.assertNotIn("num_ctx", payload)
        none_payload = bench.build_openai_payload(
            self.config(backend="openai", thinking="none"), self.case
        )
        self.assertEqual(none_payload["reasoning_effort"], "none")
        self.assertNotIn("strata_tune", payload)
        self.assertNotIn("reasoning_budget_tokens", payload)
        self.assertNotIn("chat_template_kwargs", payload)

    def test_openai_payload_adds_only_requested_strata_controls(self):
        config = dataclasses.replace(
            self.config(backend="openai", thinking="medium"),
            strata_pcie_frac=0.25,
            strata_aux_cpus=1,
            reasoning_budget_tokens=768,
        )
        payload = bench.build_openai_payload(config, self.case)
        self.assertEqual(payload["strata_tune"], {"pcie_frac": 0.25, "aux_cpus": 1})
        self.assertEqual(payload["reasoning_budget_tokens"], 768)
        self.assertNotIn("strata_tune", bench.build_openai_payload(
            dataclasses.replace(config, strata_pcie_frac=None, strata_aux_cpus=None), self.case
        ))

    def test_ollama_rejects_openai_only_request_controls(self):
        for field, value in (
            ("strata_pcie_frac", 0.5),
            ("strata_aux_cpus", 1),
            ("reasoning_budget_tokens", 512),
            ("chat_template_thinking", "off"),
        ):
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "backend openai"):
                bench.build_ollama_payload(dataclasses.replace(self.config(), **{field: value}), self.case)

    def test_openai_template_thinking_uses_json_booleans_and_keeps_effort(self):
        base = self.config(backend="openai", thinking="medium")
        enabled = bench.build_openai_payload(
            dataclasses.replace(base, chat_template_thinking="on"), self.case
        )
        disabled = bench.build_openai_payload(
            dataclasses.replace(base, chat_template_thinking="off"), self.case
        )
        self.assertIs(enabled["chat_template_kwargs"]["enable_thinking"], True)
        self.assertIs(disabled["chat_template_kwargs"]["enable_thinking"], False)
        self.assertEqual(enabled["reasoning_effort"], "medium")
        self.assertEqual(disabled["reasoning_effort"], "medium")


class RequestControlTests(unittest.TestCase):
    def test_cli_rejects_out_of_range_or_nonfinite_controls(self):
        parser = bench.build_parser()
        bad = (
            ("--strata-pcie-frac", "nan"),
            ("--strata-pcie-frac", "1.01"),
            ("--strata-aux-cpus", "2"),
            ("--reasoning-budget-tokens", "0"),
            ("--reasoning-budget-tokens", str(bench.MAX_REASONING_BUDGET_TOKENS + 1)),
            ("--chat-template-thinking", "true"),
        )
        for flag, value in bad:
            with self.subTest(flag=flag, value=value), self.assertRaises(SystemExit):
                parser.parse_args([flag, value])

    def test_run_records_exact_controls_and_nulls_without_changing_prompt(self):
        case = bench_cases.QUICK_CASES[1]
        observed = []

        def fake_request(config, selected_case):
            observed.append((config, selected_case.prompt))
            return {
                "response_text": json.dumps(selected_case.expected),
                "truncated": False,
                "stream_complete": True,
            }

        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            bench, "run_request", side_effect=fake_request
        ):
            output = Path(directory) / "controls.jsonl"
            args = bench.build_parser().parse_args([
                "--endpoint", "http://127.0.0.1:1",
                "--backend", "openai",
                "--model", "test-model",
                "--context", "32768",
                "--cases", case.case_id,
                "--output", str(output),
                "--strata-pcie-frac", "0.375",
                "--strata-aux-cpus", "0",
                "--reasoning-budget-tokens", "1024",
                "--chat-template-thinking", "off",
                "--skip-telemetry",
            ])
            self.assertEqual(bench.run_benchmark(args), 0)
            record = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(observed[0][1], case.prompt)
        self.assertEqual(observed[0][0].strata_pcie_frac, 0.375)
        self.assertEqual(record["request"]["strata_pcie_frac"], 0.375)
        self.assertEqual(record["request"]["strata_aux_cpus"], 0)
        self.assertEqual(record["request"]["reasoning_budget_tokens"], 1024)
        self.assertEqual(record["request"]["chat_template_thinking"], "off")
        default_metadata = bench._request_config(
            bench.RunConfig(
                endpoint="http://127.0.0.1:1", backend="openai", model="test-model",
                context=32768, thinking="none", temperature=0.0, seed=0,
                num_predict=None, timeout=180.0,
            ),
            case,
        )
        self.assertIsNone(default_metadata["strata_pcie_frac"])
        self.assertIsNone(default_metadata["strata_aux_cpus"])
        self.assertIsNone(default_metadata["reasoning_budget_tokens"])
        self.assertIsNone(default_metadata["chat_template_thinking"])


class StreamingTests(unittest.TestCase):
    def test_ollama_accumulator_separates_reasoning_and_final_ttft(self):
        accumulator = bench.OllamaAccumulator()
        accumulator.feed({"message": {"thinking": "private chain"}}, 0.25)
        accumulator.feed({"message": {"content": "public "}}, 0.80)
        accumulator.feed(
            {
                "message": {"thinking": "more", "content": "answer"},
                "done": True,
                "prompt_eval_count": 20,
                "eval_count": 4,
                "done_reason": "stop",
            },
            1.00,
        )
        self.assertEqual(accumulator.content, "public answer")
        self.assertNotIn("private", accumulator.content)
        self.assertEqual(accumulator.reasoning_chars, len("private chainmore"))
        self.assertEqual(accumulator.first_reasoning_seconds, 0.25)
        self.assertEqual(accumulator.first_content_seconds, 0.80)
        self.assertTrue(accumulator.done)

    def test_openai_accumulator_collects_usage_and_finish_reason(self):
        accumulator = bench.OpenAIAccumulator()
        accumulator.feed({"choices": [{"delta": {"reasoning_content": "hidden"}}]}, 0.1)
        accumulator.feed({"choices": [{"delta": {"content": "visible"}}]}, 0.4)
        accumulator.feed(
            {
                "choices": [{"delta": {}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 12, "completion_tokens": 3},
                "timings": {
                    "prompt_ms": 20.0,
                    "prompt_n": 12,
                    "cache_n": 4,
                    "prompt_per_token_ms": 2.5,
                    "prompt_per_second": 400.0,
                    "predicted_ms": 80.0,
                    "predicted_n": 2,
                    "predicted_per_token_ms": 40.0,
                    "predicted_per_second": 25.0,
                    "draft_n": 7,
                    "draft_n_accepted": 3,
                    "not_allowlisted": "model text must not persist",
                },
            },
            0.5,
        )
        accumulator.feed({"_stream_done": True}, 0.6)
        self.assertEqual(accumulator.content, "visible")
        self.assertEqual(accumulator.reasoning_chars, 6)
        self.assertEqual(accumulator.usage["prompt_tokens"], 12)
        self.assertEqual(accumulator.native_timings["draft_n"], 7)
        self.assertEqual(accumulator.native_timings["draft_n_accepted"], 3)
        self.assertNotIn("not_allowlisted", accumulator.native_timings)
        self.assertEqual(accumulator.finish_reason, "stop")
        self.assertTrue(accumulator.done_marker)

    def test_openai_native_timings_replace_omitted_fields_and_do_not_leak_between_calls(self):
        first = bench.OpenAIAccumulator()
        first.feed(
            {"timings": {"prompt_per_second": 900.0, "predicted_per_second": 20.0}}, 0.1
        )
        first.feed({"timings": {"predicted_per_second": 21.0}}, 0.2)
        self.assertEqual(first.native_timings, {"predicted_per_second": 21.0})

        generic = bench.OpenAIAccumulator()
        generic.feed({"usage": {"completion_tokens": 4}, "choices": []}, 0.1)
        self.assertIsNone(generic.native_timings)

    def test_openai_response_uses_server_native_rates_not_usage_derived_rate(self):
        case = bench_cases.QUICK_CASES[0]
        config = bench.RunConfig(
            endpoint="http://127.0.0.1:9999",
            backend="openai",
            model="strata-test",
            context=32768,
            thinking="none",
            temperature=0.0,
            seed=1,
            num_predict=100,
            timeout=5,
        )
        events = [
            {
                "choices": [{"delta": {"content": "done"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 40, "completion_tokens": 99},
                "timings": {
                    "prompt_n": 30,
                    "cache_n": 10,
                    "prompt_per_second": 750.0,
                    "predicted_n": 93,
                    "predicted_per_second": 17.25,
                    "draft_n": 31,
                    "draft_n_accepted": 18,
                },
            },
            {"_stream_done": True},
        ]
        with mock.patch.object(bench, "stream_json_events", return_value=iter(events)), mock.patch.object(
            bench.time, "monotonic", side_effect=[10.0, 10.1, 10.2, 12.0]
        ):
            response = bench.run_request(config, case)

        self.assertEqual(response["actual_completion_tokens"], 99)
        self.assertEqual(response["openai_native_timings"]["predicted_n"], 93)
        self.assertEqual(response["throughput"]["completion_tokens_per_second_native"], 17.25)
        self.assertEqual(response["throughput"]["prompt_tokens_per_second_native"], 750.0)
        self.assertAlmostEqual(response["throughput"]["completion_tokens_per_second_wall"], 49.5)

    def test_generic_openai_response_omits_native_timing_fields(self):
        case = bench_cases.QUICK_CASES[0]
        config = bench.RunConfig(
            endpoint="http://127.0.0.1:9999",
            backend="openai",
            model="generic",
            context=32768,
            thinking="none",
            temperature=0.0,
            seed=1,
            num_predict=10,
            timeout=5,
        )
        events = [
            {
                "choices": [{"delta": {"content": "done"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2},
            },
            {"_stream_done": True},
        ]
        with mock.patch.object(bench, "stream_json_events", return_value=iter(events)), mock.patch.object(
            bench.time, "monotonic", side_effect=[1.0, 1.1, 1.2, 2.0]
        ):
            response = bench.run_request(config, case)
        self.assertNotIn("openai_native_timings", response)
        self.assertNotIn("completion_tokens_per_second_native", response["throughput"])
        self.assertNotIn("prompt_tokens_per_second_native", response["throughput"])


class ExitStatusTests(unittest.TestCase):
    @staticmethod
    def args(output: Path, *, fail_on_case: bool):
        argv = [
            "--endpoint",
            "http://127.0.0.1:1",
            "--backend",
            "ollama",
            "--model",
            "test-model",
            "--context",
            "32768",
            "--cases",
            bench_cases.QUICK_CASES[1].case_id,
            "--output",
            str(output),
            "--skip-telemetry",
        ]
        if fail_on_case:
            argv.append("--fail-on-case")
        return bench.build_parser().parse_args(argv)

    def test_fail_on_case_is_opt_in_for_completed_evaluation_failure(self):
        failed_response = {
            "response_text": "{}",
            "truncated": False,
            "stream_complete": True,
        }
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            bench, "run_request", return_value=failed_response
        ):
            root = Path(directory)
            default_code = bench.run_benchmark(
                self.args(root / "default.jsonl", fail_on_case=False)
            )
            strict_code = bench.run_benchmark(
                self.args(root / "strict.jsonl", fail_on_case=True)
            )
        self.assertEqual(default_code, 0)
        self.assertEqual(strict_code, 1)

    def test_fail_on_case_rejects_incomplete_stream_even_with_perfect_text(self):
        case = bench_cases.QUICK_CASES[1]
        incomplete_response = {
            "response_text": json.dumps(case.expected),
            "truncated": True,
            "stream_complete": False,
        }
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            bench, "run_request", return_value=incomplete_response
        ):
            output = Path(directory) / "incomplete.jsonl"
            returncode = bench.run_benchmark(self.args(output, fail_on_case=True))
            record = json.loads(output.read_text())
        self.assertEqual(returncode, 1)
        self.assertFalse(record["response"]["stream_complete"])
        self.assertFalse(record["evaluation"]["passed"])
        self.assertTrue(record["evaluation"]["truncated"])


class SamplingSeedTests(unittest.TestCase):
    @staticmethod
    def args(output: Path, *, repeat: int, seed: int, seed_step=None):
        argv = [
            "--endpoint",
            "http://127.0.0.1:1",
            "--backend",
            "ollama",
            "--model",
            "test-model",
            "--context",
            "32768",
            "--cases",
            bench_cases.QUICK_CASES[1].case_id,
            "--repeat",
            str(repeat),
            "--seed",
            str(seed),
            "--output",
            str(output),
            "--skip-telemetry",
        ]
        if seed_step is not None:
            argv.extend(("--seed-step", str(seed_step)))
        return bench.build_parser().parse_args(argv)

    @staticmethod
    def successful_response(case):
        return {
            "response_text": json.dumps(case.expected),
            "truncated": False,
            "stream_complete": True,
        }

    def test_repeat_steps_sampling_seed_without_changing_selected_prompt(self):
        observed = []

        def fake_request(config, case):
            observed.append((config.seed, case.prompt))
            return self.successful_response(case)

        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            bench, "run_request", side_effect=fake_request
        ), mock.patch.object(
            bench.bench_cases, "get_cases", wraps=bench.bench_cases.get_cases
        ) as get_cases:
            output = Path(directory) / "stepped.jsonl"
            code = bench.run_benchmark(
                self.args(output, repeat=3, seed=0, seed_step=1)
            )
            records = [json.loads(line) for line in output.read_text().splitlines()]

        self.assertEqual(code, 0)
        self.assertEqual([seed for seed, _ in observed], [0, 1, 2])
        self.assertEqual([record["request"]["seed"] for record in records], [0, 1, 2])
        get_cases.assert_called_once()
        self.assertEqual(get_cases.call_args.kwargs["seed"], 0)
        self.assertEqual(len({prompt for _, prompt in observed}), 1)
        self.assertEqual(len({record["request"]["prompt_sha256"] for record in records}), 1)

    def test_default_zero_step_preserves_one_sampling_seed_across_repeats(self):
        observed_seeds = []

        def fake_request(config, case):
            observed_seeds.append(config.seed)
            return self.successful_response(case)

        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            bench, "run_request", side_effect=fake_request
        ):
            output = Path(directory) / "legacy.jsonl"
            args = self.args(output, repeat=3, seed=7)
            code = bench.run_benchmark(args)
            records = [json.loads(line) for line in output.read_text().splitlines()]

        self.assertEqual(code, 0)
        self.assertEqual(args.seed_step, 0)
        self.assertEqual(observed_seeds, [7, 7, 7])
        self.assertEqual([record["request"]["seed"] for record in records], [7, 7, 7])

    def test_seed_step_rejects_negative_and_noninteger_values(self):
        for value in ("-1", "1.5"):
            with self.subTest(value=value), self.assertRaises(SystemExit):
                bench.build_parser().parse_args(["--seed-step", value])


class ContextConstructionTests(unittest.TestCase):
    @staticmethod
    def registry_parts(case):
        prefix, remainder = case.prompt.split("BEGIN REGISTRY\n", 1)
        records, suffix = remainder.split("\nEND REGISTRY\n", 1)
        return prefix + "BEGIN REGISTRY\n", records.splitlines(), "\nEND REGISTRY\n" + suffix

    def test_context_case_is_deterministic_sized_and_has_multi_record_answer(self):
        first = bench_cases.build_context_case(1200, 4096, output_tokens=256, seed=29)
        second = bench_cases.build_context_case(1200, 4096, output_tokens=256, seed=29)
        self.assertEqual(first.prompt, second.prompt)
        self.assertEqual(first.expected, second.expected)
        self.assertLess(abs(len(first.prompt) - 1200 * 4), 250)
        anchors = first.expected["anchors"]
        locations = [first.prompt.index("record_id=" + anchor["id"]) for anchor in anchors]
        self.assertLess(locations[0], locations[1])
        self.assertLess(locations[1], locations[2])
        relationship = first.expected["relationship"]
        self.assertIn("record_id=" + relationship["partner_id"], first.prompt)
        self.assertIn("checksum=" + relationship["partner_checksum"], first.prompt)
        result = bench_cases.evaluate_case(first, "```json\n" + json.dumps(first.expected) + "\n```")
        self.assertTrue(result["passed"], result)

    def test_context_target_is_capped_after_output_and_margin_reservation(self):
        case = bench_cases.build_context_case(50_000, 4096, output_tokens=512, seed=1)
        metadata = case.metadata
        self.assertEqual(
            metadata["effective_target_tokens"],
            4096 - 512 - metadata["safety_margin_tokens"],
        )
        self.assertLessEqual(metadata["effective_target_tokens"] + 512 + metadata["safety_margin_tokens"], 4096)

    def test_full_context_prompt_remains_byte_identical_after_indexed_mode_addition(self):
        case = bench_cases.build_context_case(6000, 32768, output_tokens=768, seed=0)
        self.assertEqual(len(case.prompt), 23953)
        self.assertEqual(
            hashlib.sha256(case.prompt.encode()).hexdigest(),
            "4ce5a3046e972a12303688e1238972d5c32e8c2bbc3b63e0f5f2691d32b37697",
        )

    def test_indexed_context_keeps_requested_and_partner_rows_in_source_order(self):
        source = bench_cases.build_context_case(6000, 32768, output_tokens=768, seed=0)
        indexed = bench_cases.build_indexed_context_case(source)
        source_prefix, source_rows, source_suffix = self.registry_parts(source)
        indexed_prefix, indexed_rows, indexed_suffix = self.registry_parts(indexed)

        requested = re.findall(r"R\d{6}", source_suffix.rsplit("requested anchor ids are ", 1)[1])
        row_by_id = {row.split()[0].split("=", 1)[1]: row for row in source_rows}
        selected = set(requested)
        for record_id in requested:
            partner = re.search(r"\bpartner_id=(none|R\d{6})\b", row_by_id[record_id]).group(1)
            if partner != "none":
                selected.add(partner)
        expected_rows = [row for row in source_rows if row.split()[0].split("=", 1)[1] in selected]

        self.assertEqual(indexed.case_id, "context_retrieval_indexed")
        self.assertEqual(indexed.expected, source.expected)
        self.assertEqual(indexed_prefix, source_prefix)
        self.assertEqual(indexed_suffix, source_suffix)
        self.assertEqual(indexed_rows, expected_rows)
        self.assertEqual(indexed.metadata["retrieved_record_count"], 4)
        self.assertEqual(indexed.metadata["source_record_count"], len(source_rows))
        self.assertEqual(indexed.metadata["mode"], "indexed")
        self.assertEqual(
            indexed.metadata["retrieval_label"], "exact-ID retrieval on structured data"
        )
        self.assertEqual(
            indexed.metadata["source_prompt_sha256"],
            hashlib.sha256(source.prompt.encode()).hexdigest(),
        )
        distractor = next(row for row in source_rows if row not in expected_rows)
        self.assertNotIn(distractor, indexed.prompt)

    def test_index_selection_does_not_read_expected_scoring_truth(self):
        source = bench_cases.build_context_case(3000, 8192, output_tokens=512, seed=7)
        normal = bench_cases.build_indexed_context_case(source)
        sentinel = dataclasses.replace(source, expected={"sentinel": "not registry evidence"})
        from_sentinel = bench_cases.build_indexed_context_case(sentinel)
        self.assertEqual(from_sentinel.prompt, normal.prompt)
        self.assertEqual(from_sentinel.expected, {"sentinel": "not registry evidence"})

    def test_indexed_context_is_deterministic_and_get_cases_selects_mode(self):
        first_source = bench_cases.build_context_case(3000, 8192, output_tokens=512, seed=19)
        second_source = bench_cases.build_context_case(3000, 8192, output_tokens=512, seed=19)
        first = bench_cases.build_indexed_context_case(first_source)
        second = bench_cases.build_indexed_context_case(second_source)
        self.assertEqual(first.prompt, second.prompt)
        self.assertEqual(first.metadata["source_prompt_sha256"], second.metadata["source_prompt_sha256"])
        selected = bench_cases.get_cases(
            "context",
            context_target=3000,
            context_max=8192,
            output_tokens=512,
            seed=19,
            context_mode="indexed",
        )
        self.assertEqual([case.case_id for case in selected], ["context_retrieval_indexed"])

    def test_indexed_source_can_exceed_selected_context_and_match_128k_source(self):
        indexed = bench_cases.get_cases(
            "context",
            context_target=36000,
            context_max=32768,
            output_tokens=768,
            seed=0,
            context_mode="indexed",
        )[0]
        reference_source = bench_cases.build_context_case(36000, 131072, 768, 0)
        reference_hash = hashlib.sha256(reference_source.prompt.encode()).hexdigest()

        self.assertGreater(indexed.metadata["source_prompt_chars"], 32768 * 4)
        self.assertEqual(indexed.metadata["source_prompt_sha256"], reference_hash)
        self.assertEqual(indexed.expected, reference_source.expected)
        self.assertEqual(indexed.metadata["source_context_max_tokens"], 36000 + 768 + 4096)
        self.assertEqual(indexed.metadata["selected_context_max_tokens"], 32768)
        self.assertEqual(indexed.metadata["retrieved_record_count"], 4)
        self.assertLess(len(indexed.prompt), indexed.metadata["source_prompt_chars"])
        self.assertLessEqual(
            indexed.metadata["retrieved_prompt_estimated_tokens"]
            + 768
            + indexed.metadata["retrieved_safety_margin_tokens"],
            32768,
        )

        full = bench_cases.get_cases(
            "context",
            context_target=36000,
            context_max=32768,
            output_tokens=768,
            seed=0,
            context_mode="full",
        )[0]
        previous_capped_source = bench_cases.build_context_case(36000, 32768, 768, 0)
        self.assertEqual(full.prompt, previous_capped_source.prompt)
        self.assertLess(full.metadata["effective_target_tokens"], 36000)

    def test_oversized_index_selection_still_does_not_read_expected_truth(self):
        source = bench_cases.build_context_case(36000, 131072, 768, 0)
        normal = bench_cases.build_indexed_context_case(source)
        sentinel = dataclasses.replace(source, expected={"sentinel": "scoring only"})
        with mock.patch.object(
            bench_cases, "build_context_case", return_value=sentinel
        ) as build_source:
            indexed = bench_cases.get_cases(
                "context",
                context_target=36000,
                context_max=32768,
                output_tokens=768,
                seed=0,
                context_mode="indexed",
            )[0]
        build_source.assert_called_once_with(36000, 36000 + 768 + 4096, 768, 0)
        self.assertEqual(indexed.prompt, normal.prompt)
        self.assertEqual(indexed.expected, {"sentinel": "scoring only"})
        self.assertEqual(
            indexed.metadata["source_prompt_sha256"],
            hashlib.sha256(source.prompt.encode()).hexdigest(),
        )

    def test_indexed_context_rejects_malformed_and_missing_references(self):
        source = bench_cases.build_context_case(3000, 8192, output_tokens=512, seed=3)
        prefix, rows, suffix = self.registry_parts(source)
        late_id = source.expected["anchors"][-1]["id"]
        partner_id = source.expected["relationship"]["partner_id"]

        missing_partner_prompt = prefix + "\n".join(
            row for row in rows if not row.startswith(f"record_id={partner_id} ")
        ) + suffix
        with self.assertRaisesRegex(ValueError, r"partner record .* is missing"):
            bench_cases.build_indexed_context_case(
                dataclasses.replace(source, prompt=missing_partner_prompt)
            )

        malformed_rows = [
            re.sub(r"\bpartner_id=R\d{6}\b", "partner_id=BROKEN", row)
            if row.startswith(f"record_id={late_id} ")
            else row
            for row in rows
        ]
        malformed_prompt = prefix + "\n".join(malformed_rows) + suffix
        with self.assertRaisesRegex(ValueError, r"record .* is malformed"):
            bench_cases.build_indexed_context_case(
                dataclasses.replace(source, prompt=malformed_prompt)
            )


if __name__ == "__main__":
    unittest.main()
