import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from runtime import complete_study_dashboard as dashboard


GENERATED = datetime(2026, 10, 20, 12, 0, tzinfo=timezone.utc)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    return path


def request(call_id, *, profile="profile-a", suite="hard", case_id=None, outcome="passed", **fields):
    row = {
        "call_id": call_id,
        "run_id": "request-run",
        "profile_id": profile,
        "model": "model-a",
        "runtime": "llama",
        "backend": "openai",
        "weight_quant": "Q4",
        "suite": suite,
        "case_id": case_id or f"case-{call_id}",
        "case_version": "v1",
        "prompt_sha256": f"prompt-{call_id}",
        "repeat_index": 0,
        "outcome": outcome,
        "content_passed": outcome == "passed",
        "comparison_content_passed": outcome == "passed",
        "raw_json_object": True,
        "context_max_tokens": 49152,
        "actual_context_max_tokens": 49152,
        "output_token_limit": 32768,
        "request_timeout_seconds": 14400,
        "thinking": "medium",
        "chat_template_thinking": "on",
        "temperature": 0.7,
        "sampling_seed": 0,
        "sampling_config": {"top_p": 0.9, "top_k": 20},
        "runtime_options": {"n_gpu_layers": 99},
        "source_config": "results/model-config.json",
        "source_config_sha256": "config-sha",
        "source_file": "results/requests.jsonl",
        "source_line": int(call_id.removeprefix("c")) if call_id.removeprefix("c").isdigit() else 1,
        "case_source_sha256": "case-source-sha",
        "api_input_tokens": 100,
        "completion_tokens": 200,
        "wall_seconds": 10.0,
        "native_decode_n": 200,
        "native_decode_seconds": 2.0,
        "native_decode_tps": 100.0,
        "speed_eligible": True,
        "latency_eligible": True,
    }
    row.update(fields)
    return row


def specialist(call_id, *, role="code", review_marker=None, **fields):
    row = {
        "call_id": call_id,
        "case_group": role,
        "case_id": fields.pop("case_id", f"case-{call_id}"),
        "outcome": fields.pop("outcome", "passed"),
        "completion_success": fields.pop("completion_success", True),
        "speed_eligible": fields.pop("speed_eligible", True),
        "success": fields.pop("success", True if role == "code" else None),
        "mechanical_constraints_passed": fields.pop("mechanical_constraints_passed", None),
        "blinded_prose_review_eligible": fields.pop("blinded_prose_review_eligible", role == "prose"),
        "input_tokens": fields.pop("input_tokens", 100),
        "output_tokens": fields.pop("output_tokens", 200),
        "wall_seconds": fields.pop("wall_seconds", 10.0),
        "native_decode_tokens": fields.pop("native_decode_tokens", 200),
        "native_decode_seconds": fields.pop("native_decode_seconds", 2.0),
        "native_decode_tps": fields.pop("native_decode_tps", 100.0),
        "source_file": "results/specialists.jsonl",
        "visible_text_sha256": f"visible-{call_id}",
    }
    if review_marker is not None:
        row["review"] = review_marker
    row.update(fields)
    return row


def reports(root, benchmark_rows, specialist_rows=(), resources=None, *, specialist_review=None):
    root.mkdir(parents=True, exist_ok=True)
    benchmark = {
        "schema_version": 1,
        "generated_at": "2026-10-20T10:00:00Z",
        "request_rows": list(benchmark_rows),
        "source_provenance": [{"path": "results/requests.jsonl", "sha256": "raw-sha"}],
    }
    specialist_report = {
        "schema_version": 1,
        "generated_at": "2026-10-20T11:00:00Z",
        "attempts": list(specialist_rows),
        "source_provenance": [{"path": "results/specialists.jsonl", "sha256": "specialist-sha"}],
        "review": specialist_review,
    }
    benchmark_path = write_json(root / "benchmark.json", benchmark)
    specialist_path = write_json(root / "specialist.json", specialist_report)
    resource_path = root / "resource.json"
    if resources is not None:
        write_json(resource_path, {
            "schema_version": 1,
            "generated_at": "2026-10-20T11:30:00Z",
            "servers": list(resources),
            "sources": {"lifecycle": [], "telemetry": []},
        })
    return benchmark_path, specialist_path, resource_path


def build(root, benchmark_rows, specialist_rows=(), resources=None, *, specialist_review=None):
    paths = reports(
        root, benchmark_rows, specialist_rows, resources,
        specialist_review=specialist_review,
    )
    return dashboard.build_dashboard(*paths, generated_at=GENERATED)


def evaluator_environment_note(identity):
    original = {
        "outcome": "wrong_answer",
        "passed": False,
        "objective_count": 3,
        "objectives_passed": 0,
        "assertion_count": 9,
        "assertions_passed": 1,
    }
    diagnostic = {
        "outcome": "passed",
        "passed": True,
        "objective_count": 3,
        "objectives_passed": 3,
        "assertion_count": 9,
        "assertions_passed": 9,
    }
    return {
        "note_id": "pivot-frozenset-builtin-whitelist-v1",
        "kind": "evaluator_environment_limitation",
        "scope": "one_exact_saved_attempt",
        "case_identity": {
            key: identity[key] for key in (
                "case_id", "case_sha256", "case_version", "prompt_sha256",
            )
        },
        "attempt_identity": dict(identity),
        "summary": (
            "Original strict 1/9; unchanged-source posthoc environment diagnostic 9/9. "
            "Original strict grades are retained."
        ),
        "score_adjustment": False,
        "other_case_failures_explained": False,
        "supplemental_diagnostic": {
            "note_id": "pivot-frozenset-builtin-whitelist-v1",
            "status": "posthoc_unchanged_source_environment_diagnostic",
            "source_unchanged": True,
            "primary_grade_preserved": True,
            "new_inference_requests": 0,
            "environment_delta": "safe_builtins_plus_frozenset_only",
            "original_strict": original,
            "supplemental_environment_replay": diagnostic,
        },
        "evidence": {"path": "results/frozenset-audit.json", "sha256": "audit-sha"},
        "source_evidence": {"path": "results/raw-attempts.jsonl", "sha256": "raw-sha"},
    }


class CompleteStudyDashboardTests(unittest.TestCase):
    def test_evaluator_environment_diagnostic_propagates_without_changing_scores(self):
        identity = {
            "case_id": "code_edge_reading_pivot",
            "case_sha256": "pivot-case-sha",
            "case_version": "pivot-v1",
            "case_source_sha256": "pivot-source-sha",
            "prompt_sha256": "pivot-prompt-sha",
            "call_id": "c1",
            "visible_text_sha256": "pivot-visible-sha",
        }
        primary_grade = {
            "outcome": "wrong_answer",
            "objective_count": 3,
            "objectives_passed": 0,
            "assertion_count": 9,
            "assertions_passed": 1,
        }
        benchmark_rows = [
            request(
                "c1", content_passed=False, comparison_content_passed=False,
                **{key: identity[key] for key in (
                    "case_id", "case_sha256", "case_version", "case_source_sha256",
                    "prompt_sha256",
                )},
                **primary_grade,
            ),
            request(
                "c2", content_passed=False, comparison_content_passed=False,
                **{key: identity[key] for key in (
                    "case_id", "case_sha256", "case_version", "case_source_sha256",
                    "prompt_sha256",
                )},
                **primary_grade,
            ),
        ]
        specialist_rows = [
            specialist(
                "c1", case_id=identity["case_id"], outcome="wrong_answer", success=False,
                case_sha256=identity["case_sha256"], case_version=identity["case_version"],
                case_source_sha256=identity["case_source_sha256"],
                prompt_sha256=identity["prompt_sha256"],
                visible_text_sha256=identity["visible_text_sha256"],
                objective_count=3, objectives_passed=0,
                assertion_count=9, assertions_passed=1,
            ),
            specialist(
                "c2", case_id=identity["case_id"], outcome="wrong_answer", success=False,
                case_sha256=identity["case_sha256"], case_version=identity["case_version"],
                case_source_sha256=identity["case_source_sha256"],
                prompt_sha256=identity["prompt_sha256"],
                objective_count=3, objectives_passed=0,
                assertion_count=9, assertions_passed=1,
            ),
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = reports(root, benchmark_rows, specialist_rows)
            baseline = dashboard.build_dashboard(*paths, generated_at=GENERATED)
            document = json.loads(paths[1].read_text(encoding="utf-8"))
            document["grader_contract_notes"] = [evaluator_environment_note(identity)]
            write_json(paths[1], document)
            report = dashboard.build_dashboard(*paths, generated_at=GENERATED)
            rendered = dashboard.render_html(report)
            csv_rows = {
                row["call_id"]: dashboard._csv_row(row) for row in report["attempts"]
            }

        self.assertEqual(report["inventory"], baseline["inventory"])
        self.assertEqual(report["group_summaries"], baseline["group_summaries"])
        self.assertEqual(report["policy_notes"], baseline["policy_notes"])
        attempts = {row["call_id"]: row for row in report["attempts"]}
        baseline_attempts = {row["call_id"]: row for row in baseline["attempts"]}
        for key in (
            "outcome", "outcome_bucket", "quality_passed", "objective_count",
            "objectives_passed", "assertion_count", "assertions_passed", "input_tokens",
            "output_tokens", "wall_seconds", "native_decode_seconds", "native_decode_tps",
        ):
            self.assertEqual(attempts["c1"][key], baseline_attempts["c1"][key])
        self.assertEqual(
            attempts["c1"]["grader_contract_note_ids"],
            ["pivot-frozenset-builtin-whitelist-v1"],
        )
        diagnostic = attempts["c1"]["supplemental_evaluator_diagnostics"][0]
        self.assertEqual(diagnostic["original_strict"]["assertions_passed"], 1)
        self.assertEqual(
            diagnostic["supplemental_environment_replay"]["assertions_passed"], 9
        )
        self.assertNotIn("grader_contract_note_ids", attempts["c2"])
        self.assertNotIn("supplemental_evaluator_diagnostics", attempts["c2"])
        csv_diagnostic = json.loads(
            csv_rows["c1"]["supplemental_evaluator_diagnostics_json"]
        )
        self.assertEqual(csv_diagnostic[0]["original_strict"]["assertion_count"], 9)
        self.assertEqual(
            csv_diagnostic[0]["supplemental_environment_replay"]["assertions_passed"], 9
        )
        self.assertEqual(csv_rows["c2"]["supplemental_evaluator_diagnostics_json"], "")
        self.assertIn("Evaluator environment limitation", rendered)
        self.assertIn("Original strict 1/9", rendered)
        self.assertIn("unchanged-source posthoc environment diagnostic 9/9", rendered)
        self.assertIn("Posthoc evaluator diagnostic", rendered)
        self.assertIn('href="raw-attempts.jsonl"', rendered)

    def test_evaluator_environment_diagnostic_fails_closed_on_mismatch(self):
        identity = {
            "case_id": "case-c1",
            "case_sha256": "case-sha",
            "case_version": "v1",
            "case_source_sha256": "case-source-sha",
            "prompt_sha256": "prompt-c1",
            "call_id": "c1",
            "visible_text_sha256": "visible-c1",
        }
        benchmark_rows = [request(
            "c1", outcome="wrong_answer", content_passed=False,
            comparison_content_passed=False, case_sha256="case-sha",
            objective_count=3, objectives_passed=0, assertion_count=9, assertions_passed=1,
        )]
        specialist_rows = [specialist(
            "c1", outcome="wrong_answer", success=False, case_sha256="case-sha",
            case_version="v1", prompt_sha256="prompt-c1",
            case_source_sha256="case-source-sha", objective_count=3,
            objectives_passed=0, assertion_count=9, assertions_passed=1,
        )]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = reports(root, benchmark_rows, specialist_rows)
            document = json.loads(paths[1].read_text(encoding="utf-8"))

            malformed = evaluator_environment_note(identity)
            malformed["supplemental_diagnostic"]["source_unchanged"] = False
            document["grader_contract_notes"] = [malformed]
            write_json(paths[1], document)
            with self.assertRaisesRegex(
                dashboard.DashboardError, "unchanged-source, score-preserving diagnostic"
            ):
                dashboard.build_dashboard(*paths, generated_at=GENERATED)

            mismatched = evaluator_environment_note(identity)
            mismatched["attempt_identity"]["visible_text_sha256"] = "other-visible"
            document["grader_contract_notes"] = [mismatched]
            write_json(paths[1], document)
            with self.assertRaisesRegex(dashboard.DashboardError, "matched 0 dashboard attempts"):
                dashboard.build_dashboard(*paths, generated_at=GENERATED)

            grade_mismatch = evaluator_environment_note(identity)
            grade_mismatch["supplemental_diagnostic"]["original_strict"][
                "assertions_passed"
            ] = 0
            document["grader_contract_notes"] = [grade_mismatch]
            write_json(paths[1], document)
            with self.assertRaisesRegex(dashboard.DashboardError, "original grade does not match"):
                dashboard.build_dashboard(*paths, generated_at=GENERATED)

            duplicate_root = root / "duplicate"
            duplicate_specialist = specialist(
                "c1", outcome="wrong_answer", success=False, case_sha256="case-sha",
                case_version="v1", prompt_sha256="prompt-c1",
                case_source_sha256="case-source-sha", objective_count=3,
                objectives_passed=0, assertion_count=9, assertions_passed=1,
            )
            duplicate_paths = reports(
                duplicate_root, [], [duplicate_specialist, dict(duplicate_specialist)]
            )
            duplicate_document = json.loads(
                duplicate_paths[1].read_text(encoding="utf-8")
            )
            duplicate_document["grader_contract_notes"] = [
                evaluator_environment_note(identity)
            ]
            write_json(duplicate_paths[1], duplicate_document)
            with self.assertRaisesRegex(dashboard.DashboardError, "matched 2 dashboard attempts"):
                dashboard.build_dashboard(*duplicate_paths, generated_at=GENERATED)

    def test_contract_caveat_is_supplemental_and_exact_identity_bound(self):
        case_id = "code_multifunction_ledger_bugfix"
        identity = {
            "case_id": case_id,
            "case_sha256": "ledger-case-sha",
            "case_version": "ledger-v1",
            "prompt_sha256": "ledger-prompt-sha",
        }
        benchmark_rows = [
            request("c1", **identity),
            request(
                "c2", profile="profile-b", outcome="wrong_answer", **identity,
            ),
            request(
                "c3", case_id=case_id, case_version=identity["case_version"],
                prompt_sha256=identity["prompt_sha256"],
            ),
            request(
                "c4", case_id=case_id, case_sha256="different-case",
                case_version=identity["case_version"], prompt_sha256=identity["prompt_sha256"],
            ),
            request("c5", **identity),
        ]
        specialist_rows = [
            specialist("c1", case_id=case_id),
            specialist("c2", case_id=case_id, outcome="wrong_answer", success=False),
            specialist("c3", case_id=case_id),
            specialist("c4", case_id=case_id),
            specialist("c5", case_id=case_id, case_sha256="conflicting-case"),
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = reports(root, benchmark_rows, specialist_rows)
            baseline = dashboard.build_dashboard(*paths, generated_at=GENERATED)
            specialist_document = json.loads(paths[1].read_text(encoding="utf-8"))
            specialist_document["grader_contract_notes"] = [{
                "note_id": "ledger-contract-note",
                "kind": "potential_specification_ambiguity",
                "scope": "exact_frozen_case_and_prompt_across_all_model_profiles",
                "case_identity": identity,
                "summary": (
                    "The strict check expects TypeError for surplus transfer keys; this specific "
                    "exception class is not explicitly named for that condition in the prompt. "
                    "Original strict grades are retained. <script>alert(1)</script>"
                ),
                "score_adjustment": False,
                "other_case_failures_explained": False,
                "evidence": {
                    "path": 'results/audit" onmouseover="alert(2).json',
                    "sha256": "audit-sha",
                },
            }]
            write_json(paths[1], specialist_document)
            report = dashboard.build_dashboard(*paths, generated_at=GENERATED)
            rendered = dashboard.render_html(report)
            csv_rows = {
                row["call_id"]: dashboard._csv_row(row) for row in report["attempts"]
            }

        self.assertEqual(report["inventory"], baseline["inventory"])
        self.assertEqual(report["group_summaries"], baseline["group_summaries"])
        self.assertEqual(report["policy_notes"], baseline["policy_notes"])
        self.assertEqual(len(report["grader_contract_notes"]), 1)
        self.assertFalse(report["grader_contract_notes"][0]["score_adjustment"])
        attempts = {row["call_id"]: row for row in report["attempts"]}
        for call_id in ("c1", "c2"):
            self.assertEqual(attempts[call_id]["grader_contract_note_ids"], ["ledger-contract-note"])
        for call_id in ("c3", "c4", "c5"):
            self.assertNotIn("grader_contract_note_ids", attempts[call_id])
        self.assertTrue(attempts["c1"]["quality_passed"])
        self.assertFalse(attempts["c2"]["quality_passed"])
        self.assertEqual(
            json.loads(csv_rows["c1"]["grader_contract_note_ids_json"]),
            ["ledger-contract-note"],
        )
        self.assertEqual(csv_rows["c3"]["grader_contract_note_ids_json"], "")
        self.assertIn("Potential specification ambiguity", rendered)
        self.assertIn("supplemental; no score adjustment", rendered)
        self.assertIn("Original strict grades are retained.", rendered)
        self.assertIn("&lt;script&gt;alert(1)&lt;/script&gt;", rendered)
        self.assertNotIn("<script>alert(1)</script>", rendered)
        self.assertIn('href="audit&quot; onmouseover=&quot;alert(2).json"', rendered)

    def test_deferral_coverage_stays_separate_from_attempts_and_quality_denominators(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = reports(Path(directory), [request("completed-gemma")])
            benchmark = json.loads(paths[0].read_text())
            benchmark["not_run"] = [{
                "model": "Gemma <deferred>",
                "status": "user_deferred_partial_results_preserved",
                "reason": "User deferred <remaining> tests.",
                "source_file": "results/deferral.json",
                "coverage": {
                    "planned_primary_attempt_slots": 15,
                    "terminal_completed_attempts": 2,
                    "active_request_interrupted_by_user": 1,
                    "remaining_unexecuted_primary_slots": 12,
                    "unrelated_private_content": "must not be copied",
                },
                "response_text": "must not be copied",
            }]
            write_json(paths[0], benchmark)
            report = dashboard.build_dashboard(*paths, generated_at=GENERATED)
            rendered = dashboard.render_html(report)
        self.assertEqual(report["inventory"]["attempt_count"], 1)
        self.assertEqual(len(report["attempts"]), 1)
        self.assertEqual(report["group_summaries"][0]["quality_attempt_denominator"], 1)
        self.assertEqual(report["coverage_notes"][0]["coverage"], {
            "planned_primary_attempt_slots": 15,
            "terminal_completed_attempts": 2,
            "active_request_interrupted_by_user": 1,
            "remaining_unexecuted_primary_slots": 12,
        })
        self.assertNotIn("must not be copied", json.dumps(report))
        self.assertIn("Deferred and unrun work", rendered)
        self.assertIn("Gemma &lt;deferred&gt;", rendered)
        self.assertIn("User deferred &lt;remaining&gt; tests.", rendered)
        self.assertIn("results/deferral.json", rendered)

    def test_repeats_and_parent_variants_share_one_underlying_task(self):
        rows = [
            request("c1", case_id="tie-break", repeat_index=0, prompt_sha256="prompt-base",
                    assertion_count=5, assertions_passed=5),
            request("c2", case_id="tie-break", repeat_index=1, prompt_sha256="prompt-base",
                    assertion_count=5, assertions_passed=4),
            request("c3", case_id="tie-break-guided", parent_case_id="tie-break",
                    prompt_sha256="prompt-guided"),
        ]
        with tempfile.TemporaryDirectory() as directory:
            report = build(Path(directory), rows)

        self.assertEqual(report["inventory"]["attempt_count"], 3)
        self.assertEqual(report["inventory"]["independent_underlying_task_count"], 1)
        self.assertEqual(len(report["group_summaries"]), 1)
        group = report["group_summaries"][0]
        self.assertEqual(group["attempt_count"], 3)
        self.assertEqual(group["variant_case_count"], 2)
        self.assertEqual(group["independent_underlying_task_count"], 1)
        self.assertEqual(group["quality_underlying_task_denominator"], 1)
        self.assertEqual(group["code_assertion_diagnostics"]["assertion_count_sum_across_attempts"], 10)
        self.assertEqual(group["code_assertion_diagnostics"]["assertion_count_sample_count"], 2)

    def test_profiles_suites_and_inconsistent_budget_controls_never_pool(self):
        rows = [
            request("c1", profile="p1", suite="hard", output_token_limit=4096),
            request("c2", profile="p1", suite="hard", output_token_limit=8192),
            request("c3", profile="p1", suite="canonical", output_token_limit=4096),
            request("c4", profile="p2", suite="hard", output_token_limit=4096),
        ]
        with tempfile.TemporaryDirectory() as directory:
            report = build(Path(directory), rows)

        self.assertEqual(len(report["group_summaries"]), 4)
        self.assertEqual({group["suite"] for group in report["group_summaries"]}, {"hard", "canonical"})
        hard_p1 = [g for g in report["group_summaries"] if g["profile_id"] == "p1" and g["suite"] == "hard"]
        self.assertEqual({g["controls"]["output_token_limit"] for g in hard_p1}, {4096, 8192})
        self.assertTrue(any("distinct control sets" in gap for gap in report["gaps"]))

    def test_requested_native_controls_split_groups_and_export_as_requested_only(self):
        q8_controls = {
            "cache_type_k": "q8_0", "cache_type_v": "q8_0", "flash_attn": "on",
            "fit": "off", "fit_ctx": 73728, "batch_size": 256,
            "ubatch_size": 64, "n_cpu_moe": 2, "gpu_layers": "all",
        }
        rows = [
            request("c1", profile="legacy-profile", requested_native_runtime_controls=q8_controls),
            request("c2", profile="legacy-profile", requested_native_runtime_controls={
                "cache_type_k": "f16", "cache_type_v": "f16", "gpu_layers": "all",
            }),
            request("c3", profile="legacy-profile"),
        ]
        with tempfile.TemporaryDirectory() as directory:
            report = build(Path(directory), rows)
            rendered = dashboard.render_html(report)

        self.assertEqual(len(report["group_summaries"]), 3)
        self.assertEqual({group["profile_id"] for group in report["group_summaries"]}, {
            "legacy-profile",
        })
        attempts = {item["call_id"]: item for item in report["attempts"]}
        self.assertEqual(attempts["c1"]["requested_native_runtime_controls"], q8_controls)
        self.assertNotIn("requested_native_runtime_controls", attempts["c3"])
        self.assertEqual(
            json.loads(dashboard._csv_row(attempts["c1"])[
                "requested_native_runtime_controls_json"
            ]),
            q8_controls,
        )
        self.assertEqual(
            dashboard._csv_row(attempts["c3"])["requested_native_runtime_controls_json"],
            "",
        )
        self.assertIn("Requested native runtime controls", rendered)
        self.assertIn("does not establish entirely GPU-resident execution", rendered)
        self.assertIn('"n_cpu_moe":2', rendered)
        self.assertTrue(any("distinct control sets" in gap for gap in report["gaps"]))

    def test_all_failure_outcomes_remain_in_quality_denominator_and_format_is_separate(self):
        rows = [
            request("c1", outcome="passed", content_passed=True, comparison_content_passed=True,
                    raw_json_object=False),
            request("c2", outcome="wrong_answer", content_passed=False,
                    comparison_content_passed=False, raw_json_object=True),
            request("c3", outcome="truncated", truncated=True, content_passed=None,
                    comparison_content_passed=None),
            request("c4", outcome="request_failure", request_error_type="TimeoutError",
                    content_passed=None, comparison_content_passed=None),
            request("c5", outcome="request_failure", request_error_type="BenchmarkHTTPError",
                    content_passed=None, comparison_content_passed=None),
        ]
        with tempfile.TemporaryDirectory() as directory:
            group = build(Path(directory), rows)["group_summaries"][0]

        self.assertEqual(group["attempt_count"], 5)
        self.assertEqual(group["quality_pass_count"], 1)
        self.assertEqual(group["quality_attempt_denominator"], 5)
        self.assertEqual(group["outcome_counts"], {
            "passed": 1, "completed_wrong": 1, "truncated": 1,
            "timeout": 1, "request_failure": 1, "other": 0,
        })
        self.assertEqual(group["content_pass_counts"], {"true": 1, "false": 1, "missing": 3})
        self.assertEqual(group["raw_json_format_counts"]["true"], 4)
        self.assertEqual(group["raw_json_format_counts"]["false"], 1)

    def test_prose_mechanical_pass_is_not_quality_and_missing_review_is_pending(self):
        benchmark_rows = [
            request("c1", suite="specialist_prose", content_passed=None,
                    comparison_content_passed=None),
            request("c2", suite="specialist_prose", content_passed=None,
                    comparison_content_passed=None),
        ]
        specialist_rows = [
            specialist("c1", role="prose", mechanical_constraints_passed=True,
                       success=None, review_marker=None),
            specialist("c2", role="prose", mechanical_constraints_passed=True, success=None,
                       review_marker={"total_score": 3, "max_points": 5,
                                      "rationale": "must not appear"}),
        ]
        with tempfile.TemporaryDirectory() as directory:
            report = build(
                Path(directory), benchmark_rows, specialist_rows,
                specialist_review={"reviewer_label": "blinded-assistant"},
            )

        group = report["group_summaries"][0]
        self.assertIsNone(group["quality_pass_count"])
        self.assertIsNone(group["quality_attempt_denominator"])
        self.assertEqual(group["mechanical_constraint_counts"]["passed"], 2)
        self.assertEqual(group["pending_review_attempt_count"], 1)
        self.assertEqual(group["reviewed_attempt_count"], 1)
        self.assertEqual(group["review_score_sum"], 3)
        self.assertEqual(group["review_max_points_sum"], 5)
        pending = next(row for row in report["attempts"] if row["call_id"] == "c1")
        self.assertEqual(pending["review_status"], "pending")
        self.assertIsNone(pending["review_score"])
        self.assertNotIn("rationale", json.dumps(report))

    def test_native_tps_uses_paired_native_sums_and_reports_missing_coverage(self):
        rows = [
            request("c1", native_decode_n=100, native_decode_seconds=2.0,
                    native_decode_tps=999, wall_seconds=10, latency_eligible=True),
            request("c2", native_decode_n=300, native_decode_seconds=3.0,
                    native_decode_tps=1, wall_seconds=20, latency_eligible=True),
            request("c3", native_decode_n=None, native_decode_seconds=1.0,
                    native_decode_tps=777, wall_seconds=30, latency_eligible=False),
            request("c4", native_decode_n=1000, native_decode_seconds=1.0,
                    native_decode_tps=1000, wall_seconds=40, speed_eligible=False,
                    latency_eligible=False),
        ]
        with tempfile.TemporaryDirectory() as directory:
            group = build(Path(directory), rows)["group_summaries"][0]

        native = group["native_decode_tps"]
        self.assertEqual(native["value"], 80)
        self.assertEqual(native["eligible_attempt_count"], 3)
        self.assertEqual(native["paired_sample_count"], 2)
        self.assertEqual(native["missing_pair_count"], 1)
        self.assertEqual(native["native_decode_n_sum"], 400)
        self.assertEqual(native["native_decode_seconds_sum"], 5)
        self.assertEqual(group["observed_wall_seconds"], {
            "unit": "seconds", "sum": 100.0, "sample_count": 4, "missing_count": 0,
        })
        self.assertEqual(group["median_eligible_request_latency_seconds"], 15)

    def test_resources_join_only_exact_identity_and_missing_report_stays_null(self):
        rows = [
            request("c1", study_tag="session-a", model_session_run_id="run-a",
                    actual_context_max_tokens=None),
            request("c2", study_tag="wrong-tag", model_session_run_id="run-a"),
            request("c3"),
        ]
        resource = {
            "identity": {"tag": "session-a", "run_id": "run-a"},
            "model_name": "model-a",
            "runtime_backend": "llama",
            "context_tokens": 49152,
            "initial_load_seconds": 12.5,
            "server_pid": 123,
            "server_started_at": "2026-10-20T10:00:00Z",
            "server_stopped_at": None,
            "interval_end_at": "2026-10-20T11:00:00Z",
            "interval_status": "ongoing_bounded_by_generated_at",
            "join_status": "joined",
            "process_vram": {"sample_count": 4, "median": 12000, "sampled_max": 12500},
            "gpu_metrics": {
                "utilization_percent": {"sampled_max": 99},
                "board_power_watts": {"sampled_max": 70},
                "temperature_celsius": {"sampled_max": 55},
            },
            "hardware_power_brake": {
                "sample_count": 4, "values": ["Active"],
                "observations": [{"value": "Active", "sample_count": 4}],
            },
            "notes": ["sampled interval"],
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = build(root, rows, resources=[resource])
            matched = next(row for row in report["attempts"] if row["call_id"] == "c1")
            wrong = next(row for row in report["attempts"] if row["call_id"] == "c2")
            missing = next(row for row in report["attempts"] if row["call_id"] == "c3")
            absent = build(root / "second", rows, resources=None)
            ambiguous = build(root / "third", [rows[0]], resources=[resource, resource])
            source_row = request(
                "c4", source_file="results/session-a-01-hard_medium_32768.jsonl",
                started_at="2026-10-20T11:15:00Z", finished_at="2026-10-20T11:16:00Z",
            )
            source_report = build(root / "fourth", [source_row], resources=[resource])
            stopped_resource = dict(resource) | {
                "server_stopped_at": "2026-10-20T11:00:00Z", "interval_status": "stopped"
            }
            stopped_report = build(root / "fifth", [source_row], resources=[stopped_resource])
            second_run = dict(resource) | {
                "identity": {"tag": "session-a", "run_id": "run-b"}
            }
            source_ambiguous = build(
                root / "sixth", [source_row], resources=[resource, second_run]
            )

        self.assertEqual(matched["resource_join_status"], "exact_identity_match")
        self.assertEqual(matched["resource_lifecycle_context_tokens"], 49152)
        self.assertIsNone(matched["actual_context_tokens"])
        self.assertEqual(matched["resource_process_vram_sampled_max_mib"], 12500)
        self.assertEqual(matched["resource_initial_load_seconds"], 12.5)
        self.assertEqual(
            matched["resource_scope"], "per_server_run_sampled_statistics_not_per_request_peak"
        )
        self.assertEqual(matched["resource_interval_status"], "ongoing_bounded_by_generated_at")
        self.assertEqual(matched["resource_hardware_power_brake"]["values"], ["Active"])
        matched_group = next(group for group in report["group_summaries"]
                             if matched["group_id"] == group["group_id"])
        self.assertEqual(
            matched_group["resource_observations"][0]["process_vram_sampled_max_mib"], 12500
        )
        self.assertEqual(wrong["resource_join_status"], "no_exact_resource_match")
        self.assertEqual(missing["resource_join_status"], "identity_unavailable")
        self.assertIsNone(missing["resource_process_vram_sampled_max_mib"])
        self.assertEqual(absent["attempts"][0]["resource_join_status"], "resource_report_absent")
        self.assertIsNone(absent["attempts"][0]["resource_process_vram_sampled_max_mib"])
        self.assertTrue(any("resource report is absent" in gap for gap in absent["gaps"]))
        self.assertEqual(
            ambiguous["attempts"][0]["resource_join_status"], "ambiguous_resource_identity"
        )
        self.assertIsNone(ambiguous["attempts"][0]["resource_process_vram_sampled_max_mib"])
        source_matched = source_report["attempts"][0]
        self.assertEqual(source_matched["resource_join_status"], "exact_source_tag_match")
        self.assertEqual(source_matched["resource_identity"], {"tag": "session-a", "run_id": "run-a"})
        self.assertEqual(source_matched["model_session_run_id"], "run-a")
        self.assertEqual(source_report["inventory"]["resource_exact_match_attempt_count"], 1)
        self.assertEqual(
            dashboard._csv_row(source_matched)["resource_scope"],
            "per_server_run_sampled_statistics_not_per_request_peak",
        )
        self.assertEqual(stopped_report["attempts"][0]["resource_join_status"], "identity_unavailable")
        self.assertEqual(
            source_ambiguous["attempts"][0]["resource_join_status"],
            "ambiguous_source_tag_match",
        )

    def test_html_escapes_embedded_json_and_exports_source_provenance(self):
        malicious = "</script><script>alert(1)</script>"
        row = request(
            "c1", model=malicious, sampling_config={"label": malicious},
            source_file="results/source.jsonl", source_line=77,
            response_text="TOP SECRET RAW ANSWER", reasoning_content="TOP SECRET REASONING",
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            benchmark_path, specialist_path, resource_path = reports(root, [row])
            benchmark_sha = hashlib.sha256(benchmark_path.read_bytes()).hexdigest()
            report = dashboard.build_dashboard(
                benchmark_path, specialist_path, resource_path, generated_at=GENERATED
            )
            outputs = dashboard.write_dashboard(report, root / "dashboard")
            rendered = outputs[2].read_text(encoding="utf-8")
            stored = json.loads(outputs[0].read_text(encoding="utf-8"))
            with outputs[1].open(encoding="utf-8", newline="") as handle:
                csv_rows = list(csv.DictReader(handle))

        self.assertNotIn(malicious, rendered)
        self.assertIn("\\u003c/script\\u003e", rendered)
        self.assertNotIn("TOP SECRET", rendered)
        self.assertNotIn("TOP SECRET", json.dumps(stored))
        self.assertEqual(stored["attempts"][0]["call_id"], "c1")
        self.assertEqual(stored["attempts"][0]["source_file"], "results/source.jsonl")
        self.assertEqual(csv_rows[0]["call_id"], "c1")
        self.assertEqual(csv_rows[0]["source_line"], "77")
        benchmark_source = next(x for x in stored["input_sources"] if x["kind"] == "benchmark_report")
        self.assertEqual(benchmark_source["sha256"], benchmark_sha)
        self.assertIn("No pooled global quality rank", " ".join(stored["policy_notes"]))
        self.assertIn("whole owned server run", rendered)


if __name__ == "__main__":
    unittest.main()
