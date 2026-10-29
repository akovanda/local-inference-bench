import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

import bench_cases
from runtime import extend_hard_diagnostics as extension
from runtime import final_report


CASES = {case.case_id: case for case in bench_cases.HARD_CASES}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    return path


def write_jsonl(path, rows, *, invalid_tail=False):
    text = "".join(json.dumps(row) + "\n" for row in rows)
    if invalid_tail:
        text += "{invalid-json\n"
    path.write_text(text, encoding="utf-8")
    return path


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def baseline_document():
    return {
        "schema_version": 1,
        "generated_at": "2026-10-19T18:28:42Z",
        "attempt_count": 284,
        "independent_task_count": 4,
        "inputs": {"files": ["legacy.jsonl"], "benchmark_result_rows": 284},
        "comparison_policy": "legacy policy",
        "comparisons": [{"case_id": "legacy-case", "profiles": [{"attempt_count": 1}]}],
        "report_evaluator": {"case_version": "legacy-version"},
        "groups": [{
            "profile_id": "legacy-profile",
            "suite": "hard",
            "attempt_count": 1,
            "tasks": [{
                "call_id": "baseline-call",
                "case_id": "hard_weighted_intervals",
                "hidden_test_diagnostics": {"available": True, "hidden_test_count": 1,
                                            "hidden_tests_passed": 1},
                "rescored_outcome": "passed",
                "rescored_cause": "legacy-cause",
            }],
        }],
    }


def request_row(call_id, case_id="hard_weighted_intervals", **changes):
    case = CASES[case_id]
    row = {
        "schema_version": 1,
        "run_id": "request-run",
        "call_id": call_id,
        "started_at": "2026-10-20T01:00:00Z",
        "finished_at": "2026-10-20T01:01:00Z",
        "repeat_index": 0,
        "suite": "hard",
        "case_id": case_id,
        "case_version": bench_cases.CASE_VERSION,
        "case_source_sha256": "recorded-case-source",
        "request": {
            "model": "model-a",
            "prompt_sha256": hashlib.sha256(case.prompt.encode()).hexdigest(),
        },
        "response": {
            "response_text": "def PRIVATE_RAW_CODE_DO_NOT_COPY():\n    return 'secret'\n",
            "reasoning_content": "PRIVATE RAW REASONING DO NOT COPY",
            "stream_complete": True,
            "truncated": False,
        },
        "evaluation": {"passed": False, "detail": "fixture wrong answer"},
        "error": None,
    }
    for key, value in changes.items():
        if key == "request":
            row["request"].update(value)
        elif key == "response":
            row["response"].update(value)
        elif key == "evaluation":
            row["evaluation"].update(value)
        else:
            row[key] = value
    return row


def diagnostics(passed, total=3, *, failure_detail="semantic mismatch"):
    return {
        "available": True,
        "rescored_with_case_version": bench_cases.CASE_VERSION,
        "rescored_with_case_source_sha256": extension._CURRENT_CASE_SOURCE_SHA256,
        "hidden_test_count": total,
        "hidden_tests_passed": passed,
        "failed_tests": [] if passed == total else [{
            "index": passed, "label": "hidden", "category": "edge",
            "detail": failure_detail,
        }],
    }


class ExtendHardDiagnosticsTests(unittest.TestCase):
    def make_baseline(self, root):
        document = baseline_document()
        path = write_json(root / "baseline.json", document)
        return document, path, sha256(path)

    def test_preserves_baseline_and_adds_final_report_compatible_group(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline, baseline_path, baseline_hash = self.make_baseline(root)
            source = write_jsonl(root / "new-hard.jsonl", [
                request_row("baseline-call"),
                request_row("new-weighted"),
                request_row("new-temporal", "hard_temporal_kv"),
            ])
            evaluator = Mock(side_effect=[diagnostics(3), diagnostics(2)])
            report = extension.build_enriched_report(
                baseline_path, [source.resolve()],
                expected_baseline_sha256=baseline_hash, evaluator=evaluator,
            )

            for key, value in baseline.items():
                if key != "groups":
                    self.assertEqual(report[key], value)
            self.assertEqual(report["groups"][:-1], baseline["groups"])
            self.assertEqual(report["attempt_count"], 284)
            supplement = report["groups"][-1]
            self.assertEqual(supplement["group_name"], extension.SUPPLEMENTAL_GROUP_NAME)
            self.assertEqual(supplement["new_call_ids"], ["new-temporal", "new-weighted"])
            self.assertEqual(supplement["attempt_count"], 2)
            self.assertEqual(supplement["hidden_test_check_count"], 6)
            self.assertEqual(supplement["hidden_tests_passed"], 5)
            self.assertEqual(evaluator.call_count, 2)
            self.assertNotIn("baseline-call", [call.args[0]["call_id"] for call in evaluator.call_args_list])

            audit = report["supplemental_enrichment"]
            self.assertTrue(audit["baseline_unchanged_invariant"]["preserved"])
            self.assertIn("baseline groups only", audit["baseline_unchanged_invariant"]["note"])
            self.assertEqual(audit["skip_counts"]["baseline_call_id"], 1)
            self.assertEqual(audit["newly_rescored_attempt_count"], 2)
            self.assertEqual(audit["source_inputs"][0]["accepted_call_ids"],
                             ["new-temporal", "new-weighted"])
            self.assertEqual(len(audit["helper_source"]["sha256"]), 64)
            self.assertEqual(audit["producer_source"]["path"], str(Path(extension.__file__).resolve()))
            self.assertEqual(
                audit["producer_source"]["sha256"],
                hashlib.sha256(Path(extension.__file__).read_bytes()).hexdigest(),
            )
            self.assertEqual(len(audit["current_case_source"]["sha256"]), 64)

            output = root / "hard-round-report.json"
            extension.write_exclusive_json(output, report)
            hard_diagnostics, formats = final_report._diagnostics(root)

        self.assertEqual(formats, {})
        self.assertEqual(hard_diagnostics["baseline-call"]["rescored_cause"], "legacy-cause")
        self.assertEqual(hard_diagnostics["new-weighted"]["rescored_outcome"], "passed")
        self.assertEqual(
            hard_diagnostics["new-weighted"]["rescored_cause"],
            "recorded_failure_not_reproduced",
        )
        self.assertEqual(hard_diagnostics["new-temporal"]["rescored_outcome"], "wrong_answer")
        serialized = json.dumps(report)
        self.assertNotIn("PRIVATE_RAW_CODE_DO_NOT_COPY", serialized)
        self.assertNotIn("PRIVATE RAW REASONING DO NOT COPY", serialized)

    def test_rescore_causes_distinguish_passes_and_actual_failures(self):
        all_passed = diagnostics(3)
        failed = diagnostics(2)
        name_error_failed = diagnostics(2, failure_detail="candidate raised NameError")

        confirmed = request_row("confirmed", evaluation={"passed": True, "detail": "passed"})
        recovered = request_row("recovered", evaluation={"passed": False, "detail": "wrong"})
        builtin = request_row(
            "builtin", evaluation={"passed": False, "detail": "raised NameError in evaluator"}
        )

        self.assertEqual(extension._rescore_fields(confirmed, all_passed), (
            "passed", "recorded_pass_confirmed_by_hidden_tests",
        ))
        self.assertEqual(extension._rescore_fields(recovered, all_passed), (
            "passed", "recorded_failure_not_reproduced",
        ))
        self.assertEqual(extension._rescore_fields(builtin, all_passed), (
            "passed", "recorded_evaluator_builtin_limitation",
        ))
        self.assertEqual(extension._rescore_fields(recovered, name_error_failed), (
            "wrong_answer", "unresolved_name_requires_evaluator_review",
        ))
        self.assertEqual(extension._rescore_fields(recovered, failed), (
            "wrong_answer", "semantic_or_contract_failure",
        ))

    def test_skips_unsafe_rows_without_fabricating_zero_scores(self):
        current_prompt = hashlib.sha256(
            CASES["hard_weighted_intervals"].prompt.encode()
        ).hexdigest()
        rows = [
            request_row("bad-version", case_version="old"),
            request_row("bad-prompt", request={"prompt_sha256": "wrong"}),
            request_row("truncated", response={"truncated": True}),
            request_row("incomplete", response={"stream_complete": False}),
            request_row("request-error", error={"type": "BenchmarkHTTPError"}),
            request_row("empty", response={"response_text": "  \n"}),
            request_row("rejected"),
            request_row("accepted", request={"prompt_sha256": current_prompt}),
            request_row("other-case", "hard_transaction_ledger"),
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, baseline_path, baseline_hash = self.make_baseline(root)
            source = write_jsonl(root / "mixed.jsonl", rows, invalid_tail=True)
            evaluator = Mock(side_effect=[
                {"available": False, "reason": "candidate source was rejected"},
                diagnostics(1, total=3),
            ])
            report = extension.build_enriched_report(
                baseline_path, [source.resolve()],
                expected_baseline_sha256=baseline_hash, evaluator=evaluator,
            )

        supplement = report["groups"][-1]
        self.assertEqual(supplement["new_call_ids"], ["accepted"])
        self.assertEqual(supplement["hidden_test_check_count"], 3)
        self.assertEqual(supplement["hidden_tests_passed"], 1)
        audit = report["supplemental_enrichment"]
        expected = {
            "invalid_json_line": 1,
            "case_version_mismatch": 1,
            "prompt_sha256_mismatch": 1,
            "truncated_or_incomplete": 2,
            "request_error": 1,
            "empty_response_text": 1,
            "evaluator_rejected_source": 1,
            "ineligible_suite_or_case": 1,
        }
        for reason, count in expected.items():
            self.assertEqual(audit["skip_counts"][reason], count)
        self.assertEqual(evaluator.call_count, 2)
        self.assertNotIn("rejected", supplement["new_call_ids"])

    def test_deduplicates_identical_calls_and_rejects_conflicts_before_evaluation(self):
        row = request_row("duplicate")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, baseline_path, baseline_hash = self.make_baseline(root)
            identical = write_jsonl(root / "identical.jsonl", [row, copy.deepcopy(row)])
            evaluator = Mock(return_value=diagnostics(3))
            report = extension.build_enriched_report(
                baseline_path, [identical.resolve()],
                expected_baseline_sha256=baseline_hash, evaluator=evaluator,
            )
            self.assertEqual(report["groups"][-1]["new_call_ids"], ["duplicate"])
            self.assertEqual(report["supplemental_enrichment"]["skip_counts"]
                             ["identical_duplicate_call_id"], 1)
            self.assertEqual(evaluator.call_count, 1)

            changed = copy.deepcopy(row)
            changed["response"]["response_text"] = "different private code"
            conflicting = write_jsonl(root / "conflict.jsonl", [row, changed])
            never_called = Mock(return_value=diagnostics(3))
            with self.assertRaisesRegex(extension.ExtensionError, "conflicting duplicate"):
                extension.build_enriched_report(
                    baseline_path, [conflicting.resolve()],
                    expected_baseline_sha256=baseline_hash, evaluator=never_called,
                )
            self.assertEqual(never_called.call_count, 0)

    def test_refuses_bad_baseline_overwrite_and_symlinks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, baseline_path, baseline_hash = self.make_baseline(root)
            source = write_jsonl(root / "source.jsonl", [request_row("new")])
            with self.assertRaisesRegex(extension.ExtensionError, "SHA-256 mismatch"):
                extension.build_enriched_report(
                    baseline_path, [source.resolve()], expected_baseline_sha256="0" * 64,
                    evaluator=Mock(return_value=diagnostics(3)),
                )
            report = extension.build_enriched_report(
                baseline_path, [source.resolve()], expected_baseline_sha256=baseline_hash,
                evaluator=Mock(return_value=diagnostics(3)),
            )
            output = root / "output.json"
            extension.write_exclusive_json(output, report)
            original = output.read_bytes()
            with self.assertRaisesRegex(extension.ExtensionError, "refusing to overwrite"):
                extension.write_exclusive_json(output, report)
            self.assertEqual(output.read_bytes(), original)

            symlink_output = root / "output-link.json"
            symlink_output.symlink_to(output)
            with self.assertRaisesRegex(extension.ExtensionError, "symlink|overwrite"):
                extension.write_exclusive_json(symlink_output, report)
            source_link = root / "source-link.jsonl"
            source_link.symlink_to(source)
            with self.assertRaisesRegex(extension.ExtensionError, "symlink"):
                extension.resolve_inputs([str(source_link)], root / "unused.json")


if __name__ == "__main__":
    unittest.main()
