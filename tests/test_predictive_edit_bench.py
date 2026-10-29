import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from runtime import predictive_edit_bench as edit_bench


class PredictiveEditEvaluationTests(unittest.TestCase):
    def test_reference_edit_passes_behavior_and_exact_fidelity_raw_or_fenced(self):
        raw = edit_bench.evaluate_response(edit_bench.EDITED_SOURCE.rstrip("\n"))
        fenced = edit_bench.evaluate_response(
            "```python\n" + edit_bench.EDITED_SOURCE + "```"
        )

        for result in (raw, fenced):
            self.assertTrue(result["passed"])
            self.assertTrue(result["semantic_passed"])
            self.assertTrue(result["fidelity_passed"])
            self.assertEqual(result["hidden_tests_passed"], len(edit_bench.BEHAVIOR_TESTS))
        self.assertEqual(raw["response_format"], "raw_python")
        self.assertEqual(fenced["response_format"], "outer_python_fence")

    def test_behavior_equivalent_extra_edit_is_a_fidelity_failure(self):
        candidate = edit_bench.EDITED_SOURCE + "\n# This extra edit must fail fidelity.\n"
        result = edit_bench.evaluate_response(candidate)

        self.assertFalse(result["passed"])
        self.assertTrue(result["semantic_passed"])
        self.assertFalse(result["fidelity_passed"])
        self.assertIn("behavior checks passed", result["detail"])

    def test_unedited_tax_rate_fails_meaningful_behavior_checks(self):
        result = edit_bench.evaluate_response(edit_bench.ORIGINAL_SOURCE)

        self.assertFalse(result["passed"])
        self.assertFalse(result["semantic_passed"])
        self.assertFalse(result["fidelity_passed"])
        self.assertIn("test 1", result["behavior_detail"])

    def test_truncated_response_is_never_executed_or_passed(self):
        with mock.patch.object(edit_bench.bench_cases, "run_python_checks") as checks:
            result = edit_bench.evaluate_response(edit_bench.EDITED_SOURCE, truncated=True)

        checks.assert_not_called()
        self.assertFalse(result["passed"])
        self.assertIsNone(result["semantic_passed"])
        self.assertFalse(result["fidelity_passed"])
        self.assertTrue(result["truncated"])


class PredictiveEditRunnerTests(unittest.TestCase):
    def _args(self, output: Path):
        return edit_bench.build_parser().parse_args([
            "--endpoint", "http://127.0.0.1:18081/v1",
            "--backend", "openai",
            "--model", "gpt-oss-20b-mxfp4",
            "--context", "32768",
            "--thinking", "none",
            "--chat-template-thinking", "off",
            "--num-predict", "2048",
            "--timeout", "90",
            "--skip-telemetry",
            "--output", str(output),
        ])

    @staticmethod
    def _response(text, *, truncated=False, complete=True):
        return {
            "response_text": text,
            "reasoning_chars": 0,
            "timing": {"wall_seconds": 1.5, "ttft_final_content_seconds": 0.25},
            "native_ollama": None,
            "openai_usage": {"prompt_tokens": 600, "completion_tokens": 580},
            "actual_prompt_tokens": 600,
            "actual_completion_tokens": 580,
            "done_reason": "length" if truncated else "stop",
            "stream_complete": complete,
            "truncated": truncated,
            "throughput": {"completion_tokens_per_second_native": 12.5},
        }

    def test_success_writes_one_standard_independent_task_row(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "copy-edit.jsonl"
            with mock.patch.object(
                edit_bench.bench, "run_request",
                return_value=self._response(edit_bench.EDITED_SOURCE),
            ):
                code = edit_bench.run(self._args(output))
            row = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(code, 0)
        self.assertEqual(row["schema_version"], 1)
        self.assertEqual(row["suite"], "quality")
        self.assertEqual(row["case_id"], edit_bench.CASE_ID)
        self.assertEqual(row["case_version"], edit_bench.CASE_VERSION)
        self.assertEqual(row["case_metadata"]["independent_task_count"], 1)
        self.assertEqual(row["case_metadata"]["hidden_test_count"], 8)
        self.assertEqual(row["request"]["chat_template_thinking"], "off")
        self.assertEqual(row["request"]["output_token_limit"], 2048)
        self.assertTrue(row["evaluation"]["semantic_passed"])
        self.assertTrue(row["evaluation"]["fidelity_passed"])
        self.assertIsNone(row["error"])

    def test_incomplete_stream_writes_failure_and_returns_nonzero(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "copy-edit.jsonl"
            with mock.patch.object(
                edit_bench.bench, "run_request",
                return_value=self._response(
                    edit_bench.EDITED_SOURCE[:300], truncated=True, complete=False
                ),
            ), mock.patch.object(edit_bench.bench_cases, "run_python_checks") as checks:
                code = edit_bench.run(self._args(output))
            row = json.loads(output.read_text(encoding="utf-8"))

        checks.assert_not_called()
        self.assertEqual(code, 1)
        self.assertFalse(row["response"]["stream_complete"])
        self.assertTrue(row["response"]["truncated"])
        self.assertFalse(row["evaluation"]["passed"])
        self.assertTrue(row["evaluation"]["truncated"])


if __name__ == "__main__":
    unittest.main()
