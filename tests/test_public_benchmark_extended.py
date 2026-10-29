import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from runtime import public_benchmark_extended as extended
from runtime import public_benchmark_pilot as base


def manifest_file(path: Path) -> Path:
    human_prompt = "def add(a, b):\n    \"\"\"Return the sum.\"\"\"\n"
    value = {
        "schema_version": 1,
        "selection_seed": 20261023,
        "selected_count_per_suite": 1,
        "cases": [
            {
                "suite": "humanevalplus", "task_id": "HumanEval/0", "prompt": human_prompt,
                "original": {"task_id": "HumanEval/0", "prompt": human_prompt, "hidden": "DO_NOT_SEND_HUMAN"},
            },
            {
                "suite": "ifeval", "task_id": "ifeval-1", "prompt": "Write exactly OK.",
                "original": {"prompt": "Write exactly OK.", "instruction_id_list": ["secret"], "kwargs": [{"answer": "DO_NOT_SEND_IFEVAL"}]},
            },
        ],
    }
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def run_args(manifest: Path, output: Path):
    return extended.build_run_parser().parse_args([
        "--manifest", str(manifest), "--endpoint", "http://127.0.0.1:18081/v1",
        "--backend-adapter", "llama", "--model", "alias", "--model-key", "profile",
        "--output-dir", str(output), "--max-tokens", "65536", "--request-timeout", "21600",
        "--reasoning-effort", "none", "--temperature", "1", "--top-p", "0.95",
        "--top-k", "20", "--min-p", "0.05", "--presence-penalty", "0",
        "--repeat-penalty", "1", "--chat-template-thinking", "off",
        "--reasoning-budget-tokens", "0", "--thinking-budget-tokens", "8192",
        "--seed", "20261023", "--rerun-failed",
    ])


class ExtendedRequestTests(unittest.TestCase):
    def test_prompts_are_exact_and_allowlisted_kwargs_are_answer_free(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = base.load_manifest(manifest_file(root / "manifest.json"))
            settings = extended.run_settings(run_args(manifest.path, root / "output"))
            human = extended.build_request_payload(manifest.cases[0], settings)
            ifeval = extended.build_request_payload(manifest.cases[1], settings)
        encoded = json.dumps([human, ifeval])
        self.assertNotIn("DO_NOT_SEND_HUMAN", encoded)
        self.assertNotIn("DO_NOT_SEND_IFEVAL", encoded)
        self.assertNotIn("instruction_id_list", encoded)
        self.assertEqual(human["messages"], [
            {"role": "system", "content": base.HUMANEVAL_SYSTEM_INSTRUCTION},
            {"role": "user", "content": manifest.cases[0].prompt},
        ])
        self.assertEqual(ifeval["messages"], [{"role": "user", "content": "Write exactly OK."}])
        self.assertEqual(human["chat_template_kwargs"], {"enable_thinking": False})
        self.assertEqual(human["thinking_budget_tokens"], 8192)
        self.assertEqual(human["reasoning_budget_tokens"], 0)
        self.assertEqual(human["reasoning_effort"], "none")
        for key, expected in {"top_p": 0.95, "top_k": 20, "min_p": 0.05, "repeat_penalty": 1.0}.items():
            self.assertEqual(human[key], expected)

    def test_unknown_body_option_is_rejected_and_settings_hash_every_option(self):
        case = base.Case(0, "ifeval", "x", "prompt", {"prompt": "prompt"})
        settings = {"model": "m", "max_tokens": 1, "seed": 1, "request_options": {"temperature": 0.0, "messages": []}}
        with self.assertRaisesRegex(base.PilotError, "non-allowlisted"):
            extended.build_request_payload(case, settings)
        left = {"request_options": {"temperature": 0.0, "top_k": 20}}
        right = {"request_options": {"temperature": 0.0, "top_k": 21}}
        self.assertNotEqual(base.sha256_json(left), base.sha256_json(right))

    def test_resume_skips_terminal_cases_and_retries_only_failed_case(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path = manifest_file(root / "manifest.json")
            output = root / "output"
            args = run_args(manifest_path, output)
            calls: list[str] = []

            def response(_endpoint, payload, _timeout, raw_path, _api_key):
                calls.append(payload["messages"][-1]["content"])
                raw_path.write_bytes(b"data: [DONE]\n\n")
                status = "api_failure" if len(calls) == 2 else "completed"
                return {
                    "status": status, "response_terminal": status == "completed", "final_text": "ok",
                    "reasoning_text": "", "finish_reason": "stop" if status == "completed" else None,
                    "stream_complete": status == "completed", "truncated": False, "usage": None,
                    "native_timings": None, "counts": {}, "timing": {}, "http": {"status": 200},
                    "started_at": "x", "completed_at": "y", "error": None if status == "completed" else {"type": "test"},
                }

            with mock.patch.object(base, "request_chat_completion", side_effect=response):
                self.assertEqual(extended.run_command(args), 1)
                self.assertEqual(extended.run_command(args), 0)
                self.assertEqual(extended.run_command(args), 0)
            self.assertEqual(calls, [manifest_path and "def add(a, b):\n    \"\"\"Return the sum.\"\"\"\n", "Write exactly OK.", "Write exactly OK."])
            loaded = base.load_manifest(manifest_path)
            human = base.case_state(output, loaded.cases[0], base.read_json(output / "run.json"))
            failed = base.case_state(output, loaded.cases[1], base.read_json(output / "run.json"))
            self.assertEqual(human["attempt_count"], 1)
            self.assertEqual(failed["attempt_count"], 2)


if __name__ == "__main__":
    unittest.main()
