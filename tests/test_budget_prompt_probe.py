import argparse
import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import bench_cases
from runtime import budget_prompt_probe as probe


MODEL = "resident-model"
RUN_ID = "c9361a40-d9cf-49d7-90fe-c9361528bd8b"
TAG = "specialists-highcap16k-lfm25-qad-q4_0-strict-16k"
COMPARATOR_CALL_ID = "6c2aa162-0e02-4c39-aac3-988382047b3b"


def _json_line(path: Path, value):
    path.write_text(json.dumps(value, separators=(",", ":")) + "\n", encoding="utf-8")


def _events(path: Path):
    rows = [
        {
            "at": "2026-10-19T22:00:00Z",
            "event": "session_started",
            "run_id": RUN_ID,
            "tag": TAG,
            "model_name": MODEL,
            "context": 16384,
        },
        {
            "at": "2026-10-19T22:10:00Z",
            "event": "holding_started",
            "run_id": RUN_ID,
            "tag": TAG,
            "control_file": "/tmp/owned-control.json",
        },
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _call(case, *, truncated=False, reasoning_chars=77):
    return {
        "response_text": json.dumps(case.expected, separators=(",", ":")),
        "reasoning_chars": reasoning_chars,
        "timing": {
            "wall_seconds": 3.25,
            "ttft_reasoning_seconds": 0.1,
            "ttft_final_content_seconds": 3.0,
        },
        "native_ollama": None,
        "openai_usage": {"prompt_tokens": 350, "completion_tokens": 500},
        "actual_prompt_tokens": 350,
        "actual_completion_tokens": 500,
        "done_reason": "length" if truncated else "stop",
        "stream_complete": True,
        "truncated": truncated,
        "throughput": {
            "completion_tokens_per_second_native": 17.0,
            "completion_tokens_per_second_wall": 15.0,
        },
    }


def _comparator(parent):
    config = {
        "endpoint": "http://127.0.0.1:18081/v1",
        "backend": "openai",
        "model": MODEL,
        "context_max_tokens": 16384,
        "thinking": "medium",
        "temperature": 0.0,
        "seed": 0,
        "output_token_limit": 8192,
        "timeout_seconds": 2400.0,
        "chat_template_thinking": "on",
        "prompt_sha256": hashlib.sha256(parent.prompt.encode()).hexdigest(),
    }
    call = _call(parent, truncated=True)
    return {
        "schema_version": 1,
        "run_id": "historical-run",
        "call_id": COMPARATOR_CALL_ID,
        "suite": "hard",
        "case_id": parent.case_id,
        "case_version": bench_cases.CASE_VERSION,
        "request": config,
        "response": call,
        "evaluation": bench_cases.evaluate_case(parent, call["response_text"], truncated=True),
        "error": None,
    }


def _args(root: Path):
    events = root / "runtime-events.jsonl"
    comparator = root / "comparator.jsonl"
    _events(events)
    parent = probe.original_case()
    _json_line(comparator, _comparator(parent))
    return argparse.Namespace(
        endpoint="http://127.0.0.1:18081/v1",
        model=MODEL,
        model_session_run_id=RUN_ID,
        tag=TAG,
        runtime_events=str(events),
        comparator=str(comparator),
        comparator_call_id=COMPARATOR_CALL_ID,
        output_prefix=str(root / "probe"),
        holding_timeout=0.1,
        include_recommended_temperature=False,
    )


def _fake_http(method, url, *, payload=None, timeout=10.0, label):
    if label == "models":
        return {"data": [{"id": MODEL}]}
    if label == "props":
        return {
            "model_alias": MODEL,
            "default_generation_settings": {
                "n_ctx": 16384,
                "params": {"top_k": 50, "repeat_penalty": 1.0499999523162842},
            },
        }
    if label == "tokenize":
        # Deterministic fake token counts.  The suffix changes the count but
        # remains far below the verified context with either output cap.
        return {"tokens": list(range(360 if probe.INSTRUCTION_SUFFIX in payload["content"] else 330))}
    raise AssertionError((method, url, label))


class BudgetPromptProbeTests(unittest.TestCase):
    def test_tighter_variant_changes_only_identity_and_documented_suffix(self):
        original, tighter = probe.build_variants()
        snapshot = copy.deepcopy(original)

        self.assertEqual(original, snapshot)
        self.assertEqual(tighter.prompt, original.prompt + probe.INSTRUCTION_SUFFIX)
        self.assertEqual(tighter.case_id, probe.TIGHTER_CASE_ID)
        self.assertNotEqual(probe.TIGHTER_CASE_VERSION, bench_cases.CASE_VERSION)
        self.assertNotEqual(probe.sha256_text(tighter.prompt), probe.sha256_text(original.prompt))
        self.assertIs(tighter.expected, original.expected)
        self.assertEqual(tighter.evaluator, original.evaluator)
        self.assertEqual(tighter.metadata, original.metadata)
        for field in ("suite", "max_tokens", "function_name", "tests"):
            self.assertEqual(getattr(tighter, field), getattr(original, field))
        suffix = probe.INSTRUCTION_SUFFIX.lower()
        self.assertIn("specified ledger operations", suffix)
        self.assertIn("requested json object only", suffix)
        self.assertNotIn("balance", suffix)
        self.assertNotIn("rollback", suffix)

    @mock.patch.object(probe, "_request_json", side_effect=_fake_http)
    @mock.patch.object(probe.bench, "run_request")
    def test_three_serial_cells_keep_controls_provenance_and_safe_reasoning_counts(
        self, run_request, _request_json
    ):
        original, tighter = probe.build_variants()
        run_request.side_effect = [_call(original), _call(tighter), _call(tighter)]
        with tempfile.TemporaryDirectory() as temporary:
            args = _args(Path(temporary))
            self.assertEqual(probe.run_probe(args), 0)
            paths = probe.artifact_paths(Path(args.output_prefix))
            rows = [json.loads(line) for line in paths["requests"].read_text().splitlines()]
            manifest = json.loads(paths["prompt_manifest"].read_text())
            receipt = json.loads(paths["receipt"].read_text())

        self.assertEqual(len(rows), 3)
        self.assertEqual(run_request.call_count, 3)
        configs = [call.args[0] for call in run_request.call_args_list]
        cases = [call.args[1] for call in run_request.call_args_list]
        self.assertEqual([config.num_predict for config in configs], [12288, 8192, 12288])
        for config in configs:
            self.assertEqual(config.context, 16384)
            self.assertEqual(config.thinking, "medium")
            self.assertEqual(config.temperature, 0)
            self.assertEqual(config.seed, 0)
            self.assertEqual(config.timeout, 2400)
            self.assertEqual(config.chat_template_thinking, "on")
        self.assertEqual([case.case_id for case in cases], [
            probe.PARENT_CASE_ID, probe.TIGHTER_CASE_ID, probe.TIGHTER_CASE_ID
        ])
        self.assertEqual([row["case_version"] for row in rows], [
            bench_cases.CASE_VERSION, probe.TIGHTER_CASE_VERSION, probe.TIGHTER_CASE_VERSION
        ])
        self.assertEqual([row["request"]["output_token_limit"] for row in rows], [12288, 8192, 12288])
        self.assertEqual([row["case_metadata"]["diagnostic_variant"] for row in rows], [
            "original", "tighter", "tighter"
        ])
        for row in rows:
            self.assertEqual(row["request"]["model_session_run_id"], RUN_ID)
            self.assertEqual(row["request"]["historical_comparator_call_id"], COMPARATOR_CALL_ID)
            self.assertEqual(row["response"]["reasoning_chars"], 77)
            self.assertNotIn("reasoning_content", json.dumps(row))
        self.assertEqual(manifest["tighter_variant"]["instruction_suffix"], probe.INSTRUCTION_SUFFIX)
        self.assertEqual(receipt["underlying_problem_count"], 1)
        self.assertEqual(receipt["planned_new_attempts"], 3)
        self.assertEqual(receipt["historical_comparator_count"], 1)
        self.assertEqual(receipt["completed_new_attempts"], 3)
        self.assertIn("not independent benchmark problems", receipt["attempt_relationship"])
        self.assertTrue(all(check["fits"] for check in receipt["prompt_feasibility"]["checks"]))
        self.assertNotIn("response_text", json.dumps(receipt))

    @mock.patch.object(probe, "_request_json", side_effect=_fake_http)
    @mock.patch.object(probe.bench, "run_request")
    def test_optional_recommended_temperature_appends_fourth_original_prompt_cell(
        self, run_request, _request_json
    ):
        original, tighter = probe.build_variants()
        run_request.side_effect = [_call(original), _call(tighter), _call(tighter), _call(original)]
        with tempfile.TemporaryDirectory() as temporary:
            args = _args(Path(temporary))
            args.include_recommended_temperature = True
            self.assertEqual(probe.run_probe(args), 0)
            paths = probe.artifact_paths(Path(args.output_prefix))
            rows = [json.loads(line) for line in paths["requests"].read_text().splitlines()]
            receipt = json.loads(paths["receipt"].read_text())

        configs = [call.args[0] for call in run_request.call_args_list]
        cases = [call.args[1] for call in run_request.call_args_list]
        self.assertEqual(len(rows), 4)
        self.assertEqual([config.temperature for config in configs], [0.0, 0.0, 0.0, 0.1])
        self.assertEqual([config.num_predict for config in configs], [12288, 8192, 12288, 8192])
        self.assertEqual(cases[-1].case_id, probe.PARENT_CASE_ID)
        self.assertEqual(cases[-1].prompt, original.prompt)
        self.assertEqual(rows[-1]["case_version"], bench_cases.CASE_VERSION)
        self.assertEqual(
            rows[-1]["request"]["sampling_variant"],
            "liquidai_recommended_temperature_with_verified_server_defaults",
        )
        self.assertEqual(
            rows[-1]["request"]["cell_variant"],
            rows[-1]["request"]["diagnostic_cell_id"],
        )
        declared = rows[-1]["request"]["declared_sampling"]
        self.assertEqual(declared["temperature"]["value"], 0.1)
        self.assertEqual(declared["top_k"]["value"], 50)
        self.assertEqual(declared["repeat_penalty"]["value"], 1.05)
        self.assertEqual(receipt["planned_new_attempts"], 4)
        self.assertEqual(receipt["underlying_problem_count"], 1)
        self.assertTrue(receipt["resident_identity"]["recommended_sampling_defaults_match"])
        self.assertEqual(receipt["new_attempts"][-1]["temperature"], 0.1)
        self.assertEqual(
            receipt["new_attempts"][-1]["sampling_variant"],
            "liquidai_recommended_temperature_with_verified_server_defaults",
        )

    @mock.patch.object(probe.bench, "run_request")
    @mock.patch.object(probe, "_request_json")
    def test_optional_cell_fails_closed_when_props_sampling_defaults_do_not_match(
        self, request_json, run_request
    ):
        request_json.side_effect = [
            {"data": [{"id": MODEL}]},
            {
                "model_alias": MODEL,
                "default_generation_settings": {
                    "n_ctx": 16384,
                    "params": {"top_k": 40, "repeat_penalty": 1.05},
                },
            },
        ]
        with tempfile.TemporaryDirectory() as temporary:
            args = _args(Path(temporary))
            args.include_recommended_temperature = True
            with self.assertRaisesRegex(probe.ProbeError, "top_k=50"):
                probe.run_probe(args)
            self.assertTrue(all(not path.exists() for path in probe.artifact_paths(Path(args.output_prefix)).values()))
        run_request.assert_not_called()

    @mock.patch.object(probe, "_request_json", side_effect=_fake_http)
    @mock.patch.object(probe.bench, "run_request")
    def test_interruption_preserves_completed_row_and_writes_compact_receipt(
        self, run_request, _request_json
    ):
        original, _ = probe.build_variants()
        run_request.side_effect = [_call(original), KeyboardInterrupt()]
        with tempfile.TemporaryDirectory() as temporary:
            args = _args(Path(temporary))
            with self.assertRaises(KeyboardInterrupt):
                probe.run_probe(args)
            paths = probe.artifact_paths(Path(args.output_prefix))
            rows = paths["requests"].read_text().splitlines()
            receipt = json.loads(paths["receipt"].read_text())

        self.assertEqual(len(rows), 1)
        self.assertEqual(receipt["status"], "interrupted")
        self.assertEqual(receipt["completed_new_attempts"], 1)
        self.assertEqual(receipt["problem"]["type"], "KeyboardInterrupt")
        self.assertNotIn("reasoning_content", json.dumps(receipt))

    @mock.patch.object(probe, "_request_json")
    def test_wrong_resident_context_fails_before_requests_or_artifacts(self, request_json):
        request_json.side_effect = [
            {"data": [{"id": MODEL}]},
            {"model_alias": MODEL, "default_generation_settings": {"n_ctx": 8192}},
        ]
        with tempfile.TemporaryDirectory() as temporary:
            args = _args(Path(temporary))
            with self.assertRaisesRegex(probe.ProbeError, "resident context"):
                probe.run_probe(args)
            self.assertTrue(all(not path.exists() for path in probe.artifact_paths(Path(args.output_prefix)).values()))

    def test_non_loopback_endpoint_and_existing_artifact_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = _args(root)
            args.endpoint = "https://example.com/v1"
            with self.assertRaisesRegex(probe.ProbeError, "loopback"):
                probe.run_probe(args)

            args.endpoint = "http://127.0.0.1:18081/v1"
            paths = probe.artifact_paths(Path(args.output_prefix))
            paths["receipt"].write_text("occupied")
            with self.assertRaisesRegex(probe.ProbeError, "already exists"):
                probe.run_probe(args)


if __name__ == "__main__":
    unittest.main()
