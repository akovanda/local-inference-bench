import argparse
import copy
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

import bench
import bench_cases
from runtime import bounded_weighted_probe as probe


TAG = "specialists-highcap16k-lfm25-qad-q4_0-strict-16k"
RUN_ID = "c9361a40-d9cf-49d7-90fe-c9361528bd8b"
MODEL = "lfm2.5-24b-a2b-qad-q4_0"


def _write_events(path: Path, *, run_id=RUN_ID, tag=TAG, terminal=None):
    rows = [
        {
            "at": "2026-10-19T22:00:00Z",
            "event": "session_started",
            "run_id": run_id,
            "tag": tag,
            "runtime_backend": "llama",
            "model_name": MODEL,
            "context": 16384,
            "source_config": "/tmp/native-source.json",
        },
        {
            "at": "2026-10-19T22:01:00Z",
            "event": "holding_started",
            "run_id": run_id,
            "tag": tag,
            "control_file": "/tmp/owned-control.json",
        },
    ]
    if terminal is not None:
        rows.append(
            {
                "at": "2026-10-19T22:02:00Z",
                "event": terminal,
                "run_id": run_id,
                "tag": tag,
            }
        )
    path.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def _args(root: Path) -> argparse.Namespace:
    events = root / "events.jsonl"
    _write_events(events)
    return argparse.Namespace(
        runtime_events=str(events),
        tag=TAG,
        model_session_run_id=RUN_ID,
        endpoint="http://127.0.0.1:18081/v1",
        model=MODEL,
        output_prefix=str(root / f"{TAG}-bounded12"),
        output_cap=8192,
        timeout=1200.0,
    )


def _fake_http(method, url, *, payload=None, timeout=10.0, label):
    if label == "models":
        return {"data": [{"id": MODEL}]}
    if label == "props":
        return {
            "model_alias": MODEL,
            "default_generation_settings": {
                "n_ctx": 16384,
                "params": {
                    "temperature": 0.8,
                    "top_p": 0.9,
                    "top_k": 40,
                    "reasoning_budget_tokens": -1,
                },
            },
        }
    if label == "tokenize":
        if payload is None or probe.INSTRUCTION_SUFFIX not in payload.get("content", ""):
            raise AssertionError("tokenize did not receive the bounded variant prompt")
        return {"tokens": list(range(400))}
    raise AssertionError((method, url, label))


def _fake_call():
    return {
        "response_text": "def select_weighted_jobs(jobs):\n    pass",
        "reasoning_chars": 321,
        "reasoning_content": "forbidden raw chain",
        "vendor": {"raw_reasoning": "also forbidden"},
        "timing": {
            "wall_seconds": 2.5,
            "ttft_reasoning_seconds": 0.1,
            "ttft_final_content_seconds": 2.0,
        },
        "openai_usage": {"prompt_tokens": 400, "completion_tokens": 90},
        "actual_prompt_tokens": 400,
        "actual_completion_tokens": 90,
        "done_reason": "stop",
        "stream_complete": True,
        "truncated": False,
        "throughput": {
            "completion_tokens_per_second_native": 40.0,
            "completion_tokens_per_second_wall": 36.0,
        },
    }


def _read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


class BoundedWeightedProbeTests(unittest.TestCase):
    def test_variant_is_exact_suffix_with_retained_frozen_grader_and_provenance(self):
        parent = probe.original_case()
        snapshot = copy.deepcopy(parent)
        variant = probe.build_variant()

        self.assertEqual(parent, snapshot)
        self.assertEqual(variant.case_id, probe.VARIANT_CASE_ID)
        self.assertEqual(variant.prompt, parent.prompt + probe.INSTRUCTION_SUFFIX)
        self.assertEqual(
            probe.INSTRUCTION_SUFFIX,
            "\n\nFor this bounded-size diagnostic, jobs contains at most 12 entries. "
            "An exact exhaustive search over all subsets is acceptable; no optimization "
            "beyond correctness is required.",
        )
        self.assertEqual(probe.sha256_text(variant.prompt), variant.metadata["full_variant_prompt_sha256"])
        self.assertEqual(probe.sha256_text(parent.prompt), variant.metadata["parent_prompt_sha256"])
        self.assertEqual(
            probe.sha256_text(probe.INSTRUCTION_SUFFIX),
            variant.metadata["instruction_suffix_sha256"],
        )
        self.assertIs(variant.tests, parent.tests)
        self.assertEqual(variant.evaluator, parent.evaluator)
        self.assertEqual(variant.function_name, parent.function_name)
        self.assertNotEqual(probe.VARIANT_CASE_VERSION, bench_cases.CASE_VERSION)
        self.assertEqual(variant.metadata["parent_case_id"], probe.PARENT_CASE_ID)
        self.assertEqual(variant.metadata["diagnostic_variant"], "bounded12_exhaustive_allowed")
        self.assertIs(variant.metadata["diagnostic_domain_changed"], True)
        self.assertIs(variant.metadata["diagnostic_algorithm_guidance"], True)

    def test_parent_valid_inputs_are_bounded_and_oversize_valid_fixture_is_rejected(self):
        parent = probe.original_case()
        summary = probe.validate_parent_test_bound(parent)
        self.assertGreater(summary["valid_input_test_count"], 0)
        self.assertLessEqual(summary["largest_valid_input_entries"], 12)

        jobs = [[f"job-{index}", index * 2, index * 2 + 1, index] for index in range(13)]
        oversized = replace(parent, tests=parent.tests + ({"args": [jobs]},))
        with self.assertRaisesRegex(probe.ProbeError, "exceeding the diagnostic bound"):
            probe.validate_parent_test_bound(oversized)

    @mock.patch.object(probe.bench, "run_request")
    @mock.patch.object(probe.budget_prompt_probe, "_request_json")
    def test_wrong_lifecycle_identity_fails_before_network_gpu_or_artifacts(
        self, request_json, run_request
    ):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = _args(root)
            _write_events(Path(args.runtime_events), run_id="different-run")
            with self.assertRaisesRegex(
                probe.sampler_recipe_probe.ProbeError, "supplied run ID"
            ):
                probe.run(args)
            self.assertTrue(
                all(not path.exists() for path in probe.artifact_paths(Path(args.output_prefix)).values())
            )
        request_json.assert_not_called()
        run_request.assert_not_called()

    @mock.patch.object(probe.budget_prompt_probe, "_request_json", side_effect=_fake_http)
    @mock.patch.object(probe.bench, "run_request")
    def test_one_request_uses_variant_standard_controls_and_writes_safe_results(
        self, run_request, _request_json
    ):
        sent_payloads = []

        def execute(config, case):
            sent_payloads.append(bench.build_openai_payload(config, case))
            return _fake_call()

        run_request.side_effect = execute
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = _args(root)
            self.assertEqual(probe.run(args), 0)
            paths = probe.artifact_paths(Path(args.output_prefix).resolve())
            rows = _read_jsonl(paths["requests"])
            manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
            receipt = json.loads(paths["receipt"].read_text(encoding="utf-8"))
            serialized = json.dumps({"rows": rows, "manifest": manifest, "receipt": receipt})

        self.assertEqual(run_request.call_count, 1)
        config, requested_case = run_request.call_args.args
        parent = probe.original_case()
        self.assertEqual(requested_case.prompt, parent.prompt + probe.INSTRUCTION_SUFFIX)
        self.assertIs(requested_case.tests, parent.tests)
        self.assertEqual(config.context, 16384)
        self.assertEqual(config.thinking, "medium")
        self.assertEqual(config.temperature, 0.0)
        self.assertEqual(config.seed, 0)
        self.assertEqual(config.num_predict, 8192)
        self.assertEqual(config.timeout, 1200.0)
        self.assertEqual(config.chat_template_thinking, "on")
        self.assertIsNone(config.reasoning_budget_tokens)
        self.assertEqual(len(sent_payloads), 1)
        self.assertEqual(sent_payloads[0]["max_tokens"], 8192)
        self.assertEqual(sent_payloads[0]["reasoning_effort"], "medium")
        self.assertEqual(sent_payloads[0]["chat_template_kwargs"], {"enable_thinking": True})
        self.assertNotIn("thinking_budget_tokens", sent_payloads[0])
        self.assertNotIn("reasoning_budget_tokens", sent_payloads[0])
        self.assertNotIn("response_format", sent_payloads[0])
        for knob in ("top_p", "top_k", "min_p", "presence_penalty", "repeat_penalty"):
            self.assertNotIn(knob, sent_payloads[0])

        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["case_id"], probe.VARIANT_CASE_ID)
        self.assertEqual(row["case_version"], probe.VARIANT_CASE_VERSION)
        self.assertEqual(row["response"]["response_text"], _fake_call()["response_text"])
        self.assertEqual(row["response"]["reasoning_chars"], 321)
        self.assertFalse(row["evaluation"]["passed"])
        request = row["request"]
        self.assertEqual(request["parent_case_id"], "hard_weighted_intervals")
        self.assertEqual(request["diagnostic_variant"], "bounded12_exhaustive_allowed")
        self.assertIs(request["diagnostic_domain_changed"], True)
        self.assertIs(request["diagnostic_algorithm_guidance"], True)
        self.assertEqual(request["parent_prompt_sha256"], probe.sha256_text(parent.prompt))
        self.assertEqual(
            request["instruction_suffix_sha256"], probe.sha256_text(probe.INSTRUCTION_SUFFIX)
        )
        self.assertEqual(request["full_variant_prompt_sha256"], probe.sha256_text(requested_case.prompt))
        self.assertRegex(request["source_case_sha256"], r"^[0-9a-f]{64}$")
        self.assertRegex(request["caller_source_sha256"], r"^[0-9a-f]{64}$")
        self.assertNotIn("forbidden raw chain", serialized)
        self.assertNotIn("also forbidden", serialized)
        self.assertNotIn('"reasoning_content"', serialized)
        self.assertNotIn('"raw_reasoning"', serialized)
        self.assertEqual(manifest["underlying_problem_count"], 1)
        self.assertEqual(manifest["planned_new_attempts"], 1)
        self.assertIn("not an original benchmark score", manifest["comparability"])
        self.assertTrue(manifest["variant"]["tests_tuple_retained"])
        self.assertEqual(receipt["status"], "complete")
        self.assertEqual(receipt["attempted_request_count"], 1)
        self.assertEqual(receipt["recorded_request_count"], 1)
        self.assertEqual(receipt["request_failure_count"], 0)
        self.assertEqual(receipt["evaluator_pass_count"], 0)
        self.assertEqual(receipt["attempts"][0]["reasoning_chars"], 321)
        self.assertEqual(receipt["provenance"]["tests_sha256"], manifest["parent"]["tests_sha256"])
        self.assertTrue(receipt["provenance"]["tests_tuple_retained"])
        self.assertNotIn("response_text", json.dumps(receipt))

    @mock.patch.object(probe.budget_prompt_probe, "_request_json", side_effect=_fake_http)
    @mock.patch.object(probe.bench, "run_request", side_effect=TimeoutError("fixture timeout"))
    def test_request_failure_is_recorded_without_becoming_an_evaluator_failure(
        self, run_request, _request_json
    ):
        with tempfile.TemporaryDirectory() as directory:
            args = _args(Path(directory))
            self.assertEqual(probe.run(args), 1)
            paths = probe.artifact_paths(Path(args.output_prefix).resolve())
            row = _read_jsonl(paths["requests"])[0]
            receipt = json.loads(paths["receipt"].read_text(encoding="utf-8"))

        self.assertEqual(run_request.call_count, 1)
        self.assertEqual(row["error"]["type"], "TimeoutError")
        self.assertEqual(row["evaluation"], {"passed": False, "detail": "request failed"})
        self.assertEqual(receipt["status"], "completed_with_request_failures")
        self.assertEqual(receipt["request_failure_count"], 1)

    @mock.patch.object(probe.budget_prompt_probe, "_request_json", side_effect=_fake_http)
    @mock.patch.object(probe.bench, "run_request", side_effect=KeyboardInterrupt())
    def test_interruption_writes_receipt_and_leaves_lifecycle_ownership_to_parent(
        self, run_request, _request_json
    ):
        with tempfile.TemporaryDirectory() as directory:
            args = _args(Path(directory))
            with self.assertRaises(KeyboardInterrupt):
                probe.run(args)
            paths = probe.artifact_paths(Path(args.output_prefix).resolve())
            receipt = json.loads(paths["receipt"].read_text(encoding="utf-8"))
            events = Path(args.runtime_events).read_text(encoding="utf-8")

        self.assertEqual(run_request.call_count, 1)
        self.assertEqual(receipt["status"], "interrupted")
        self.assertEqual(receipt["attempted_request_count"], 1)
        self.assertEqual(receipt["recorded_request_count"], 0)
        self.assertEqual(receipt["interruption"]["type"], "KeyboardInterrupt")
        self.assertNotIn("holding_stopped", events)

    def test_cli_bounds_and_required_surface(self):
        parser = probe.build_parser()
        option_dests = {
            action.dest for action in parser._actions if action.option_strings and action.dest != "help"
        }
        self.assertEqual(
            option_dests,
            {
                "runtime_events",
                "tag",
                "model_session_run_id",
                "endpoint",
                "model",
                "output_prefix",
                "output_cap",
                "timeout",
            },
        )
        self.assertEqual(parser.get_default("output_cap"), 8192)
        self.assertEqual(parser.get_default("timeout"), 1200.0)
        with self.assertRaises(argparse.ArgumentTypeError):
            probe._bounded_output_cap("12289")
        with self.assertRaises(argparse.ArgumentTypeError):
            probe._bounded_timeout("1801")


if __name__ == "__main__":
    unittest.main()
