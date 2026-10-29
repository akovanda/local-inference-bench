import json
import signal
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

from runtime import public_benchmark_pilot as pilot


def _manifest(path: Path, *, ifeval_prompt: str = "Follow this exactly.") -> Path:
    value = {
        "schema_version": 1,
        "selection_seed": 20261023,
        "selected_count_per_suite": 1,
        "sources": {"evalplus": "official", "ifeval": "official"},
        "hashes": {"evalplus": "a" * 64, "ifeval": "b" * 64},
        "preparation": {"created_at": "2026-10-23T00:00:00Z"},
        "cases": [
            {
                "suite": "humanevalplus",
                "task_id": "HumanEval/0",
                "prompt": "def add(a, b):\n    \"\"\"Return the sum.\"\"\"\n",
                "original": {
                    "task_id": "HumanEval/0",
                    "prompt": "def add(a, b):\n    \"\"\"Return the sum.\"\"\"\n",
                    "hidden_tests": "GRADER_SECRET_DO_NOT_SEND",
                },
            },
            {
                "suite": "ifeval",
                "task_id": "ifeval-7",
                "prompt": ifeval_prompt,
                "original": {
                    "key": 7,
                    "prompt": ifeval_prompt,
                    "instruction_id_list": ["secret:checker"],
                    "kwargs": [{"secret": "IFEVAL_GRADER_SECRET_DO_NOT_SEND"}],
                },
            },
        ],
    }
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _run_args(manifest: Path, output: Path, endpoint: str):
    return pilot.build_parser().parse_args(
        [
            "run",
            "--manifest",
            str(manifest),
            "--endpoint",
            endpoint,
            "--model",
            "served-alias",
            "--model-key",
            "test-model",
            "--output-dir",
            str(output),
            "--max-tokens",
            "65536",
            "--request-timeout",
            "20",
            "--reasoning-effort",
            "medium",
            "--temperature",
            "0",
            "--seed",
            "20261023",
        ]
    )


class _StreamingHandler(BaseHTTPRequestHandler):
    payloads = []

    def do_POST(self):  # noqa: N802 - BaseHTTPRequestHandler API
        length = int(self.headers["Content-Length"])
        payload = json.loads(self.rfile.read(length))
        self.__class__.payloads.append(payload)
        is_humaneval = len(payload["messages"]) == 2
        if is_humaneval:
            events = [
                {"choices": [{"delta": {"reasoning_content": "private thought"}}]},
                {"choices": [{"delta": {"content": "def add(a, b):\n    return a + b\n"}}]},
                {
                    "choices": [{"delta": {}, "finish_reason": "stop"}],
                    "usage": {
                        "prompt_tokens": 21,
                        "completion_tokens": 14,
                        "total_tokens": 35,
                        "completion_tokens_details": {"reasoning_tokens": 5},
                    },
                    "timings": {
                        "prompt_n": 21,
                        "prompt_per_second": 410.5,
                        "predicted_n": 14,
                        "predicted_per_second": 19.25,
                    },
                },
            ]
        else:
            events = [
                {"choices": [{"delta": {"content": "requested response"}}]},
                {
                    "choices": [{"delta": {}, "finish_reason": "length"}],
                    "usage": {"prompt_tokens": 8, "completion_tokens": 65536},
                },
            ]
        body = b"".join(
            b"data: " + json.dumps(event).encode("utf-8") + b"\n\n" for event in events
        ) + b"data: [DONE]\n\n"
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args):
        return


class PublicBenchmarkRunTests(unittest.TestCase):
    def test_sigterm_uses_keyboard_interrupt_cancellation_path(self):
        previous = signal.getsignal(signal.SIGTERM)
        with pilot.cancel_on_sigterm():
            handler = signal.getsignal(signal.SIGTERM)
            self.assertTrue(callable(handler))
            with self.assertRaises(KeyboardInterrupt):
                handler(signal.SIGTERM, None)
        self.assertIs(signal.getsignal(signal.SIGTERM), previous)

    def test_cancellation_writes_explicit_result_and_retains_raw_artifact(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path = _manifest(root / "manifest.json")
            manifest = pilot.load_manifest(manifest_path)
            output = root / "output"
            output.mkdir()
            args = _run_args(manifest_path, output, "http://127.0.0.1:1/v1")
            settings = pilot.run_settings(args)
            binding = pilot.ensure_binding(output, manifest, settings)
            with mock.patch.object(
                pilot, "request_chat_completion", side_effect=KeyboardInterrupt
            ):
                with self.assertRaises(KeyboardInterrupt):
                    pilot.execute_case(
                        output, manifest, binding, settings, manifest.cases[0], None
                    )
            attempt = pilot._case_directory(output, manifest.cases[0]) / "attempt-0001"
            result = json.loads((attempt / "result.json").read_text(encoding="utf-8"))
            status = json.loads((output / "status.json").read_text(encoding="utf-8"))
            raw = (attempt / "raw.sse").read_bytes()

        self.assertEqual(result["status"], "cancelled")
        self.assertEqual(result["error"]["type"], "KeyboardInterrupt")
        self.assertEqual(raw, b"")
        self.assertEqual(status["counts"]["cancelled"], 1)

    def test_streaming_separates_outputs_captures_usage_and_resumes_terminal_cases(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path = _manifest(root / "manifest.json")
            output = root / "output"
            _StreamingHandler.payloads = []
            server = ThreadingHTTPServer(("127.0.0.1", 0), _StreamingHandler)
            thread = threading.Thread(target=server.serve_forever)
            thread.start()
            endpoint = f"http://127.0.0.1:{server.server_port}/v1"
            args = _run_args(manifest_path, output, endpoint)
            try:
                self.assertEqual(pilot.run_command(args), 0)
                self.assertEqual(len(_StreamingHandler.payloads), 2)
                self.assertEqual(pilot.run_command(args), 0)
                self.assertEqual(len(_StreamingHandler.payloads), 2, "resume queried the endpoint")
            finally:
                server.shutdown()
                server.server_close()
                thread.join()

            manifest = pilot.load_manifest(manifest_path)
            human_dir = pilot._case_directory(output, manifest.cases[0]) / "attempt-0001"
            ifeval_dir = pilot._case_directory(output, manifest.cases[1]) / "attempt-0001"
            human = json.loads((human_dir / "result.json").read_text(encoding="utf-8"))
            ifeval = json.loads((ifeval_dir / "result.json").read_text(encoding="utf-8"))
            status = json.loads((output / "status.json").read_text(encoding="utf-8"))
            human_raw = (human_dir / "raw.sse").read_bytes()

        self.assertEqual(human["status"], "completed")
        self.assertEqual(human["final_text"], "def add(a, b):\n    return a + b\n")
        self.assertEqual(human["reasoning_text"], "private thought")
        self.assertEqual(human["counts"]["reasoning_tokens"], 5)
        self.assertEqual(human["counts"]["completion_tokens"], 14)
        self.assertTrue(human["counts"]["token_counts_authoritative"])
        self.assertEqual(human["finish_reason"], "stop")
        self.assertEqual(human["native_timings"]["prompt_per_second"], 410.5)
        self.assertEqual(human["native_timings"]["predicted_per_second"], 19.25)
        self.assertTrue(human["stream_complete"])
        self.assertIsNotNone(human["timing"]["ttft_reasoning_seconds"])
        self.assertIsNotNone(human["timing"]["ttft_final_seconds"])
        self.assertIn(b"private thought", human_raw)

        self.assertEqual(ifeval["status"], "truncated")
        self.assertTrue(ifeval["truncated"])
        self.assertEqual(ifeval["finish_reason"], "length")
        self.assertEqual(status["counts"]["completed"], 1)
        self.assertEqual(status["counts"]["truncated"], 1)
        self.assertEqual(status["cases"][0]["attempt_count"], 1)
        self.assertEqual(status["cases"][1]["attempt_count"], 1)

    def test_requests_never_contain_original_grader_fields(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest = pilot.load_manifest(_manifest(Path(temporary) / "manifest.json"))
            settings = {
                "model": "alias",
                "max_tokens": 65536,
                "temperature": 0.0,
                "seed": 20261023,
                "reasoning_effort": "medium",
            }
            human = pilot.build_request_payload(manifest.cases[0], settings)
            ifeval = pilot.build_request_payload(manifest.cases[1], settings)

        encoded = json.dumps([human, ifeval])
        self.assertNotIn("GRADER_SECRET_DO_NOT_SEND", encoded)
        self.assertNotIn("IFEVAL_GRADER_SECRET_DO_NOT_SEND", encoded)
        self.assertNotIn("instruction_id_list", encoded)
        self.assertEqual(human["messages"][0]["content"], pilot.HUMANEVAL_SYSTEM_INSTRUCTION)
        self.assertEqual(human["messages"][1]["role"], "user")
        self.assertEqual(ifeval["messages"], [{"role": "user", "content": "Follow this exactly."}])

    def test_resume_rejects_manifest_mismatch_before_any_request(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path = _manifest(root / "manifest.json")
            output = root / "output"
            args = _run_args(manifest_path, output, "http://127.0.0.1:1/v1")
            manifest = pilot.load_manifest(manifest_path)
            settings = pilot.run_settings(args)
            output.mkdir()
            pilot.ensure_binding(output, manifest, settings)
            _manifest(manifest_path, ifeval_prompt="A changed prompt.")

            with self.assertRaisesRegex(pilot.PilotError, "different manifest"):
                pilot.run_command(args)


class PublicBenchmarkGradeTests(unittest.TestCase):
    def test_official_subprocess_failures_are_durable_grader_errors(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path = _manifest(root / "manifest.json")
            manifest = pilot.load_manifest(manifest_path)
            output = root / "output"
            output.mkdir()
            run_args = _run_args(manifest_path, output, "http://127.0.0.1:1/v1")
            binding = pilot.ensure_binding(output, manifest, pilot.run_settings(run_args))
            pilot.write_status(output, manifest, binding)
            evalplus_root = root / "evalplus"
            evalplus_root.mkdir()
            ifeval_root = root / "google" / "instruction_following_eval"
            ifeval_root.mkdir(parents=True)
            (ifeval_root / "evaluation_main.py").write_text("# placeholder\n", encoding="utf-8")
            args = pilot.build_parser().parse_args(
                [
                    "grade",
                    "--manifest",
                    str(manifest_path),
                    "--output-dir",
                    str(output),
                    "--evalplus-root",
                    str(evalplus_root),
                    "--ifeval-root",
                    str(ifeval_root),
                ]
            )
            failed_process = {
                "command": ["grader"],
                "cwd": str(root),
                "started_at": pilot.utc_now(),
                "completed_at": pilot.utc_now(),
                "wall_seconds": 0.1,
                "returncode": 7,
                "timed_out": False,
                "error": None,
                "stdout": "stdout.log",
                "stderr": "stderr.log",
            }
            with mock.patch.object(pilot, "_run_subprocess", return_value=failed_process):
                self.assertEqual(pilot.grade_command(args), 1)

            summary = json.loads(
                (output / "grading" / "summary.json").read_text(encoding="utf-8")
            )
            status = json.loads((output / "status.json").read_text(encoding="utf-8"))

        self.assertEqual(summary["status"], "grader_error")
        self.assertEqual(summary["evalplus"]["status"], "grader_error")
        self.assertEqual(summary["ifeval"]["status"], "grader_error")
        self.assertEqual(status["grading"]["status"], "grader_error")
        self.assertEqual(summary["coverage"]["not_started"], 2)


if __name__ == "__main__":
    unittest.main()
