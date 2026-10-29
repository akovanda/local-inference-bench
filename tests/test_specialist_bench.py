import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from runtime import specialist_bench
from tests.test_specialist_cases import FRESH_REFERENCE


class SpecialistCLITests(unittest.TestCase):
    def test_default_selection_is_the_full_six_case_suite(self):
        args = specialist_bench.build_parser().parse_args([])
        self.assertEqual(len(specialist_bench._selected_cases(args)), 6)

    def test_endpoint_is_restricted_to_loopback(self):
        for endpoint in (
            "http://127.0.0.1:8080",
            "http://127.2.3.4:8080/v1",
            "http://[::1]:8080",
            "https://localhost:8443",
        ):
            with self.subTest(endpoint=endpoint):
                specialist_bench.validate_loopback_endpoint(endpoint)
        with self.assertRaisesRegex(ValueError, "loopback"):
            specialist_bench.validate_loopback_endpoint("https://example.com/v1")
        with self.assertRaisesRegex(ValueError, "credentials"):
            specialist_bench.validate_loopback_endpoint("http://user:secret@127.0.0.1:8080")

    def test_caps_and_timeouts_are_bounded(self):
        parser = specialist_bench.build_parser()
        for argv in (
            ["--output-cap", "65537"],
            ["--output-cap", "0"],
            ["--timeout", "21601"],
            ["--context", "131073"],
            ["--timeout", "nan"],
            ["--temperature", "inf"],
        ):
            with self.subTest(argv=argv), self.assertRaises(SystemExit):
                parser.parse_args(argv)
        high = parser.parse_args([
            "--output-cap", "65536", "--timeout", "21600", "--context", "131072",
        ])
        self.assertEqual(high.output_cap, 65536)
        self.assertEqual(high.timeout, 21600.0)
        self.assertEqual(high.context, 131072)

    def test_print_manifest_needs_no_server_arguments(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = specialist_bench.main(["--print-manifest"])
        manifest = json.loads(output.getvalue())
        self.assertEqual(code, 0)
        self.assertEqual(manifest["case_count"], 6)
        self.assertEqual(
            manifest["suite_sha256"],
            "683416fc8d3559adff123b78a0f2ad38cefb43422d440c557a1ccabcbdff9b26",
        )
        coding = [item for item in manifest["cases"] if item["group"] == "code"]
        self.assertTrue(all(item["default_output_cap"] == 4096 for item in coding))
        self.assertEqual(sum(item["independent_task_count"] for item in coding), 3)
        self.assertEqual(sum(item["objective_count"] for item in coding), 8)


class SpecialistRunTests(unittest.TestCase):
    @staticmethod
    def args(root: Path, *extra: str):
        argv = [
            "--endpoint",
            "http://127.0.0.1:8080",
            "--model",
            "local-test-model",
            "--tag",
            "candidate-q4",
            "--profile",
            "coding-medium",
            "--jsonl",
            str(root / "results.jsonl"),
            "--groups",
            "code",
            "--cases",
            "code_fresh_batch_module",
            "--thinking",
            "medium",
            "--output-cap",
            "4096",
            "--timeout",
            "900",
            "--temperature",
            "0",
            "--seed",
            "42",
            "--capability-note",
            "template opens a reasoning channel; this is a request label only",
        ]
        argv.extend(extra)
        return specialist_bench.build_parser().parse_args(argv)

    @staticmethod
    def completed_response():
        return {
            "response_text": FRESH_REFERENCE,
            "reasoning_chars": 17,
            "timing": {
                "wall_seconds": 12.5,
                "ttft_reasoning_seconds": 0.2,
                "ttft_final_content_seconds": 3.5,
            },
            "native_ollama": None,
            "openai_usage": {
                "prompt_tokens": 321,
                "completion_tokens": 144,
                "completion_tokens_details": {"reasoning_tokens": 37},
            },
            "actual_prompt_tokens": 321,
            "actual_completion_tokens": 144,
            "done_reason": "stop",
            "stream_complete": True,
            "truncated": False,
            "throughput": {
                "prompt_tokens_per_second_native": 800.0,
                "completion_tokens_per_second_native": 21.5,
                "completion_tokens_per_second_wall": 11.52,
            },
            "openai_native_timings": {
                "prompt_n": 300,
                "cache_n": 21,
                "predicted_n": 107,
                "predicted_ms": 4976.0,
            },
        }

    def test_run_writes_one_sanitized_row_and_separate_visible_text(self):
        observed = []

        def fake_request(config, case):
            observed.append((config, case))
            return self.completed_response()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stdout = io.StringIO()
            stderr = io.StringIO()
            with mock.patch.object(
                specialist_bench.bench, "run_request", side_effect=fake_request
            ), contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                code = specialist_bench.run(self.args(root))

            rows = [json.loads(line) for line in (root / "results.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows), 1)
            row = rows[0]
            visible_path = Path(row["visible_text"]["path"])
            self.assertEqual(visible_path.read_text(), FRESH_REFERENCE)
            serialized = (root / "results.jsonl").read_text()

        self.assertEqual(code, 0)
        self.assertEqual(stdout.getvalue(), "")
        self.assertNotIn(FRESH_REFERENCE, stderr.getvalue())
        self.assertNotIn(FRESH_REFERENCE, serialized)
        self.assertEqual(len(observed), 1)
        config, selected_case = observed[0]
        self.assertEqual(config.thinking, "medium")
        self.assertEqual(config.num_predict, 4096)
        self.assertEqual(config.seed, 42)
        self.assertEqual(selected_case.case_id, "code_fresh_batch_module")
        self.assertEqual(row["objective_outcome"], "passed")
        self.assertIsNone(row["objective_error"])
        self.assertIsNone(row["error"])
        self.assertEqual(row["suite"], "specialist_code")
        self.assertNotIn("response_text", row["response"])
        self.assertEqual(row["response"]["actual_prompt_tokens"], 321)
        self.assertEqual(row["response"]["timing"]["wall_seconds"], 12.5)
        self.assertEqual(row["request"]["backend"], "openai")
        self.assertEqual(row["request"]["thinking"], "medium")
        self.assertIsNone(row["request"]["chat_template_thinking"])
        self.assertIsNone(row["request"]["reasoning_budget_tokens"])
        self.assertEqual(row["request"]["reasoning_controls_requested"]["reasoning_effort"], "medium")
        self.assertIn("request label only", row["request"]["capability_note"])
        self.assertEqual(row["response_metrics"]["input_tokens_reported"], 321)
        self.assertEqual(row["response_metrics"]["output_tokens_reported"], 144)
        self.assertEqual(row["response_metrics"]["reasoning_tokens_reported"], 37)
        self.assertEqual(row["response_metrics"]["reasoning_chars_observed"], 17)
        self.assertEqual(row["response_metrics"]["native_decode_rate_tokens_per_second"], 21.5)
        self.assertAlmostEqual(row["response_metrics"]["native_decode_seconds"], 4.976)
        self.assertIsNone(row["response_metrics"]["load_seconds_reported"])
        self.assertTrue(row["eligibility"]["coding_objective_eligible"])
        self.assertTrue(row["visible_text"]["reasoning_excluded"])

    def test_timeout_still_writes_a_row_and_empty_visible_answer(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with mock.patch.object(
                specialist_bench.bench,
                "run_request",
                side_effect=TimeoutError("request exceeded 900s total timeout"),
            ), contextlib.redirect_stderr(io.StringIO()):
                code = specialist_bench.run(self.args(root))
            row = json.loads((root / "results.jsonl").read_text())
            visible = Path(row["visible_text"]["path"]).read_text()

        self.assertEqual(code, 1)
        self.assertEqual(row["objective_outcome"], "timeout")
        self.assertEqual(row["objective_error"]["type"], "TimeoutError")
        self.assertIsNone(row["response_metrics"])
        self.assertEqual(visible, "")
        self.assertFalse(row["eligibility"]["speed_eligible"])

    def test_reasoning_fields_remain_null_when_server_reports_nothing(self):
        call = self.completed_response()
        call["reasoning_chars"] = 0
        call["openai_usage"] = {"prompt_tokens": 5, "completion_tokens": 2}
        metrics = specialist_bench.response_metrics(call)
        self.assertIsNone(metrics["reasoning_chars_observed"])
        self.assertFalse(metrics["reasoning_stream_observed"])
        self.assertIsNone(metrics["reasoning_tokens_reported"])


if __name__ == "__main__":
    unittest.main()

