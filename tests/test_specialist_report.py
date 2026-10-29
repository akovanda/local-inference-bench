import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from runtime import specialist_cases, specialist_report


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def specialist_row(
    root,
    call_id,
    case_id,
    group,
    *,
    model="resident-alpha",
    rate=10.0,
    decode_seconds=1.0,
    passed=True,
    reasoning_tokens=None,
    context=8192,
    output_cap=None,
    timeout=600.0,
    thinking=None,
    temperature=None,
    seed=0,
    sampler_recipe=None,
    native_thinking_budget=None,
    ledger_schema_sha=None,
):
    thinking = thinking if thinking is not None else ("medium" if group == "code" else "none")
    temperature = temperature if temperature is not None else (0.0 if group == "code" else 0.7)
    output_cap = output_cap if output_cap is not None else (4096 if group == "code" else 2048)
    text = "fabricated final answer"
    text_path = root / f"{call_id}.txt"
    text_path.write_text(text, encoding="utf-8")
    digest = hashlib.sha256(text.encode()).hexdigest()
    if group == "code":
        outcome = "passed" if passed else "wrong_answer"
        evaluation = {
            "outcome": outcome,
            "passed": passed,
            "validation": "accepted",
            "independent_task_count": 1,
            "objective_count": 2,
            "objectives_passed": 2 if passed else 1,
            "assertion_count": 4,
            "assertions_passed": 4 if passed else 3,
            "detail": "fabricated code evaluation",
        }
    else:
        outcome = "constraints_passed" if passed else "constraint_failed"
        case = next(item for item in specialist_cases.CASES if item.case_id == case_id)
        evaluation = {
            "outcome": outcome,
            "passed": passed,
            "deterministic_constraints_only": True,
            "constraint_count": len(case.prose_constraints),
            "constraints_passed": len(case.prose_constraints) if passed else 0,
            "subjective_quality_score": None,
            "subjective_quality_status": "unscored_pending_blinded_review",
            "blinded_rubric": [
                {
                    "criterion_id": item.criterion_id,
                    "guidance": item.guidance,
                    "max_points": item.max_points,
                }
                for item in case.rubric
            ],
            "detail": "mechanical fixture only",
        }
    row = {
        "schema_version": 1,
        "run_id": "fixture-run",
        "call_id": call_id,
        "started_at": "2026-10-19T01:00:00Z",
        "finished_at": "2026-10-19T01:00:05Z",
        "repeat_index": 0,
        "tag": f"fixture-{group}",
        "model": model,
        "profile": "fixture-profile",
        "suite": f"specialist_{group}",
        "case_id": case_id,
        "case_group": group,
        "case_category": "fixture",
        "case_version": specialist_cases.CASE_VERSION,
        "case_sha256": "case-hash",
        "suite_sha256": specialist_cases.FROZEN_SUITE_SHA256,
        "case_source_sha256": "source-hash",
        "request": {
            "endpoint": "http://127.0.0.1:8080/v1",
            "backend": "openai",
            "model": model,
            "profile": "fixture-profile",
            "thinking": thinking,
            "temperature": temperature,
            "seed": seed,
            "context_max_tokens": context,
            "output_token_limit": output_cap,
            "timeout_seconds": timeout,
            "prompt_sha256": f"prompt-{case_id}",
            "reasoning_controls_requested": {
                "reasoning_effort": thinking,
                "chat_template_thinking": "on" if group == "code" else "off",
                "reasoning_budget_tokens": 512,
            },
            "capability_note": "fixture capability",
        },
        "response": {"stream_complete": True, "truncated": False, "done_reason": "stop"},
        "response_metrics": {
            "input_tokens_reported": 100,
            "output_tokens_reported": 25,
            "reasoning_tokens_reported": reasoning_tokens,
            "reasoning_chars_observed": 12 if reasoning_tokens is not None else None,
            "reasoning_stream_observed": reasoning_tokens is not None,
            "wall_seconds": 5.0,
            "ttft_visible_seconds": 0.5,
            "native_decode_tokens": 999,
            "native_decode_seconds": decode_seconds,
            "native_decode_rate_tokens_per_second": rate,
        },
        "visible_text": {
            "path": str(text_path), "sha256": digest, "chars": len(text),
            "bytes": len(text.encode()), "reasoning_excluded": True,
        },
        "objective_outcome": outcome,
        "objective_error": None,
        "error": None,
        "evaluation": evaluation,
        "eligibility": {
            "speed_eligible": True,
            "coding_objective_eligible": group == "code",
            "prose_constraint_eligible": group == "prose",
            "blinded_prose_review_eligible": group == "prose",
            "reasoning_metric_eligible": reasoning_tokens is not None,
        },
    }
    if sampler_recipe is not None:
        row["request"].update({
            "sampler_overrides": dict(sampler_recipe["sampler_overrides"]),
            "sampler_recipe_label": sampler_recipe["label"],
            "sampler_recipe_sha256": sampler_recipe["sha256"],
            "sampler_recipe_sources": list(sampler_recipe.get("sources", [])),
            "sampler_recipe_claim": sampler_recipe.get("claim", "custom_experiment"),
            "sampler_application_evidence": "requested_in_outgoing_payload_not_independently_verified_effective",
        })
    if native_thinking_budget is not None:
        row["request"].update({
            "native_thinking_budget_tokens_requested": native_thinking_budget,
            "native_thinking_budget_application_evidence": (
                "requested_in_outgoing_payload_not_independently_verified_applied_or_enforced"
            ),
        })
        row["request"]["reasoning_controls_requested"][
            "thinking_budget_tokens"
        ] = native_thinking_budget
    if ledger_schema_sha is not None:
        row["request"].update({
            "ledger_json_schema_path": f"/schemas/{ledger_schema_sha}.json",
            "ledger_json_schema_sha256": ledger_schema_sha,
            "ledger_json_schema_requested": {
                "response_format_type": "json_schema", "name": "ledger_output",
                "strict": True,
                "application_evidence": (
                    "strict_json_schema_requested_in_outgoing_payload_"
                    "enforcement_not_independently_verified"
                ),
            },
        })
    return row


def study_receipt(path, config, code_path, prose_path, *, model="resident-alpha", actual_context=8192):
    write_json(path, {
        "schema_version": 1,
        "tag": path.stem.removesuffix("-study-trial"),
        "status": "complete",
        "runtime_backend": "llama",
        "source_config": {"path": str(config), "sha256": "config-sha"},
        "controls": {"profile": "fixture-profile"},
        "session": {
            "run_id": f"session-{model}", "model_name": model, "context": actual_context,
            "initial_load_seconds": 12.5, "hold_seconds": 30.0,
            "session_started_at": "2026-10-19T00:59:00Z",
            "server_ready_at": "2026-10-19T00:59:12Z",
            "holding_started_at": "2026-10-19T00:59:13Z",
        },
        "artifacts": {
            "code": {"path": str(code_path), "exists": code_path.is_file()},
            "prose": {"path": str(prose_path), "exists": prose_path.is_file()},
        },
    })


def frozenset_diagnostic_audit(row):
    original = {
        "outcome": "wrong_answer",
        "passed": False,
        "validation": "accepted",
        "independent_task_count": 1,
        "independent_tasks_passed": 0,
        "objective_count": 3,
        "objectives_passed": 0,
        "assertion_count": 9,
        "assertions_passed": 1,
    }
    diagnostic = {
        "outcome": "passed",
        "passed": True,
        "validation": "accepted",
        "independent_task_count": 1,
        "independent_tasks_passed": 1,
        "objective_count": 3,
        "objectives_passed": 3,
        "assertion_count": 9,
        "assertions_passed": 9,
    }
    return {
        "schema_version": 1,
        "kind": "frozen_python_builtin_environment_diagnostic",
        "call_id": row["call_id"],
        "case_id": row["case_id"],
        "case_version": row["case_version"],
        "case_sha256": row["case_sha256"],
        "case_source_sha256": row["case_source_sha256"],
        "prompt_sha256": row["request"]["prompt_sha256"],
        "visible_text": {
            "sha256": row["visible_text"]["sha256"],
            "reasoning_excluded": True,
        },
        "new_inference_requests": 0,
        "primary_grades_and_model_outputs_modified": False,
        "only_execution_environment_delta": specialist_report.FROZENSET_ENVIRONMENT_DELTA,
        "original_frozen_grade": original,
        "original_strict_replay": dict(original),
        "one_builtin_diagnostic": diagnostic,
        "frozen_source_pins": {
            "bench_cases.py": "a" * 64,
            "runtime/specialist_cases.py": row["case_source_sha256"],
        },
        "original_runner_sha256": "b" * 64,
        "diagnostic_runner_sha256": "c" * 64,
    }


class SpecialistReportTests(unittest.TestCase):
    def test_frozenset_diagnostic_is_exact_attempt_bound_and_score_neutral(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "frozenset-config.json"
            write_json(config, {"model": "/models/resident-alpha.gguf"})
            code_path = root / specialist_report.FROZENSET_DIAGNOSTIC_RAW_FILENAME
            prose_path = root / "frozenset-specialist-prose.jsonl"
            write_jsonl(prose_path, [])
            case_id = "code_edge_reading_pivot"
            target = specialist_row(root, "target-call", case_id, "code", passed=False)
            target.update({
                "case_sha256": "pivot-case-sha",
                "case_source_sha256": "pivot-source-sha",
            })
            target["request"]["prompt_sha256"] = "pivot-prompt-sha"
            target["evaluation"].update({
                "outcome": "wrong_answer",
                "passed": False,
                "validation": "accepted",
                "independent_task_count": 1,
                "independent_tasks_passed": 0,
                "objective_count": 3,
                "objectives_passed": 0,
                "assertion_count": 9,
                "assertions_passed": 1,
            })
            decoy = specialist_row(root, "decoy-call", case_id, "code", passed=False)
            decoy.update({
                "case_sha256": target["case_sha256"],
                "case_source_sha256": target["case_source_sha256"],
            })
            decoy["request"]["prompt_sha256"] = target["request"]["prompt_sha256"]
            write_jsonl(code_path, [target, decoy])
            study_receipt(root / "frozenset-study-trial.json", config, code_path, prose_path)

            baseline = specialist_report.build_report(root)
            audit_path = root / specialist_report.FROZENSET_DIAGNOSTIC_AUDIT_FILENAME
            write_json(audit_path, frozenset_diagnostic_audit(target))
            audit_sha = hashlib.sha256(audit_path.read_bytes()).hexdigest()
            raw_sha = hashlib.sha256(code_path.read_bytes()).hexdigest()
            with patch.object(
                specialist_report, "FROZENSET_DIAGNOSTIC_AUDIT_SHA256", audit_sha
            ), patch.object(
                specialist_report, "FROZENSET_DIAGNOSTIC_RAW_SHA256", raw_sha
            ):
                report = specialist_report.build_report(root)
            rendered = specialist_report.render_html(report)
            csv_rows = {
                row["call_id"]: row
                for row in csv.DictReader(specialist_report.render_csv(report).splitlines())
                if row["row_type"] == "attempt"
            }

        self.assertEqual(report["inventory"], baseline["inventory"])
        self.assertEqual(report["role_summaries"], baseline["role_summaries"])
        self.assertEqual(len(report["grader_contract_notes"]), 1)
        note = report["grader_contract_notes"][0]
        self.assertEqual(note["note_id"], specialist_report.FROZENSET_DIAGNOSTIC_NOTE_ID)
        self.assertFalse(note["score_adjustment"])
        self.assertTrue(note["supplemental_diagnostic"]["primary_grade_preserved"])
        attempts = {row["call_id"]: row for row in report["attempts"]}
        target_attempt = attempts["target-call"]
        baseline_target = next(
            row for row in baseline["attempts"] if row["call_id"] == "target-call"
        )
        for key in (
            "outcome", "success", "quality_status", "objective_count", "objectives_passed",
            "assertion_count", "assertions_passed", "input_tokens", "output_tokens",
            "wall_seconds", "native_decode_seconds", "native_decode_tps",
        ):
            self.assertEqual(target_attempt[key], baseline_target[key])
        self.assertEqual(
            target_attempt["grader_contract_note_ids"],
            [specialist_report.FROZENSET_DIAGNOSTIC_NOTE_ID],
        )
        diagnostic = target_attempt["supplemental_evaluator_diagnostics"][0]
        self.assertEqual(diagnostic["original_strict"]["assertions_passed"], 1)
        self.assertEqual(
            diagnostic["supplemental_environment_replay"]["assertions_passed"], 9
        )
        self.assertNotIn("grader_contract_note_ids", attempts["decoy-call"])
        self.assertNotIn("supplemental_evaluator_diagnostics", attempts["decoy-call"])
        csv_diagnostic = json.loads(csv_rows["target-call"]["supplemental_evaluator_diagnostics"])
        self.assertEqual(csv_diagnostic[0]["original_strict"]["assertion_count"], 9)
        self.assertEqual(
            csv_diagnostic[0]["supplemental_environment_replay"]["assertions_passed"], 9
        )
        self.assertEqual(csv_rows["decoy-call"]["supplemental_evaluator_diagnostics"], "")
        self.assertIn("Evaluator environment limitation", rendered)
        self.assertIn("frozen strict evaluator reports 1/9", rendered)
        self.assertIn("only the standard frozenset builtin added produces 9/9", rendered)
        self.assertIn(
            f'href="{specialist_report.FROZENSET_DIAGNOSTIC_AUDIT_FILENAME}"', rendered
        )
        self.assertIn(
            f'href="{specialist_report.FROZENSET_DIAGNOSTIC_RAW_FILENAME}"', rendered
        )

    def test_frozenset_diagnostic_rejects_bad_hash_schema_and_unchanged_flags(self):
        row = {
            "call_id": "call",
            "case_id": "case",
            "case_version": "version",
            "case_sha256": "case-sha",
            "case_source_sha256": "source-sha",
            "request": {"prompt_sha256": "prompt-sha"},
            "visible_text": {"sha256": "visible-sha"},
        }
        base = frozenset_diagnostic_audit(row)
        mutations = (
            ("schema", lambda value: value.update(schema_version=2)),
            ("modified-primary", lambda value: value.update(
                primary_grades_and_model_outputs_modified=True
            )),
            ("replay-grade", lambda value: value["original_strict_replay"].update(
                assertions_passed=2
            )),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audit_path = root / specialist_report.FROZENSET_DIAGNOSTIC_AUDIT_FILENAME
            raw_path = root / specialist_report.FROZENSET_DIAGNOSTIC_RAW_FILENAME
            write_jsonl(raw_path, [])
            raw_sha = hashlib.sha256(raw_path.read_bytes()).hexdigest()
            write_json(audit_path, base)
            with patch.object(
                specialist_report, "FROZENSET_DIAGNOSTIC_AUDIT_SHA256", "0" * 64
            ), patch.object(
                specialist_report, "FROZENSET_DIAGNOSTIC_RAW_SHA256", raw_sha
            ):
                with self.assertRaisesRegex(ValueError, "hash mismatch"):
                    specialist_report._load_frozenset_diagnostic_note(root)
            audit_sha = hashlib.sha256(audit_path.read_bytes()).hexdigest()
            with patch.object(
                specialist_report, "FROZENSET_DIAGNOSTIC_AUDIT_SHA256", audit_sha
            ), patch.object(
                specialist_report, "FROZENSET_DIAGNOSTIC_RAW_SHA256", "0" * 64
            ):
                with self.assertRaisesRegex(ValueError, "raw source hash mismatch"):
                    specialist_report._load_frozenset_diagnostic_note(root)
            for label, mutate in mutations:
                with self.subTest(label=label):
                    value = json.loads(json.dumps(base))
                    mutate(value)
                    write_json(audit_path, value)
                    digest = hashlib.sha256(audit_path.read_bytes()).hexdigest()
                    with patch.object(
                        specialist_report, "FROZENSET_DIAGNOSTIC_AUDIT_SHA256", digest
                    ), patch.object(
                        specialist_report, "FROZENSET_DIAGNOSTIC_RAW_SHA256", raw_sha
                    ):
                        with self.assertRaisesRegex(
                            ValueError, "unexpected schema, environment delta, or unchanged-grade status"
                        ):
                            specialist_report._load_frozenset_diagnostic_note(root)

    def test_frozenset_diagnostic_fails_closed_on_attempt_or_grade_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "mismatch-config.json"
            write_json(config, {"model": "/models/resident-alpha.gguf"})
            code_path = root / specialist_report.FROZENSET_DIAGNOSTIC_RAW_FILENAME
            prose_path = root / "mismatch-specialist-prose.jsonl"
            write_jsonl(prose_path, [])
            row = specialist_row(
                root, "mismatch-call", "code_edge_reading_pivot", "code", passed=False
            )
            write_jsonl(code_path, [row])
            study_receipt(root / "mismatch-study-trial.json", config, code_path, prose_path)
            audit = frozenset_diagnostic_audit(row)
            audit_path = root / specialist_report.FROZENSET_DIAGNOSTIC_AUDIT_FILENAME

            unmatched = json.loads(json.dumps(audit))
            unmatched["call_id"] = "other-call"
            write_json(audit_path, unmatched)
            digest = hashlib.sha256(audit_path.read_bytes()).hexdigest()
            raw_sha = hashlib.sha256(code_path.read_bytes()).hexdigest()
            with patch.object(
                specialist_report, "FROZENSET_DIAGNOSTIC_AUDIT_SHA256", digest
            ), patch.object(
                specialist_report, "FROZENSET_DIAGNOSTIC_RAW_SHA256", raw_sha
            ):
                with self.assertRaisesRegex(ValueError, "matched 0 report attempts"):
                    specialist_report.build_report(root)

            write_json(audit_path, audit)
            digest = hashlib.sha256(audit_path.read_bytes()).hexdigest()
            with patch.object(
                specialist_report, "FROZENSET_DIAGNOSTIC_AUDIT_SHA256", digest
            ), patch.object(
                specialist_report, "FROZENSET_DIAGNOSTIC_RAW_SHA256", raw_sha
            ):
                with self.assertRaisesRegex(ValueError, "original grade does not match"):
                    specialist_report.build_report(root)

    def test_grader_contract_note_requires_exact_case_and_prompt_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "contract-config.json"
            write_json(config, {"model": "/models/resident-alpha.gguf"})
            code_path = root / "contract-specialist-code.jsonl"
            prose_path = root / "contract-specialist-prose.jsonl"
            write_jsonl(prose_path, [])
            case_id = "code_multifunction_ledger_bugfix"
            matched_pass = specialist_row(root, "matched-pass", case_id, "code")
            matched_fail = specialist_row(
                root, "matched-fail", case_id, "code", model="resident-beta", passed=False
            )
            prompt_mismatch = specialist_row(root, "prompt-mismatch", case_id, "code")
            prompt_mismatch["request"]["prompt_sha256"] = "different-prompt"
            missing_identity = specialist_row(root, "missing-case-hash", case_id, "code")
            missing_identity.pop("case_sha256")
            write_jsonl(
                code_path, [matched_pass, matched_fail, prompt_mismatch, missing_identity]
            )
            study_receipt(root / "contract-study-trial.json", config, code_path, prose_path)
            audit_path = root / specialist_report.LEDGER_CONTRACT_AUDIT_FILENAME
            write_json(audit_path, {
                "kind": "frozen_case_contract_interpretation_caveat",
                "status": "original_strict_grades_preserved",
                "case_identity": {
                    "case_id": case_id,
                    "case_sha256": "case-hash",
                    "case_version": specialist_cases.CASE_VERSION,
                },
                "prompt_sha256": f"prompt-{case_id}",
            })
            audit_sha = hashlib.sha256(audit_path.read_bytes()).hexdigest()
            with patch.object(
                specialist_report, "LEDGER_CONTRACT_AUDIT_SHA256", audit_sha
            ):
                report = specialist_report.build_report(root)
            rendered = specialist_report.render_html(report)
            csv_rows = list(csv.DictReader(specialist_report.render_csv(report).splitlines()))

        self.assertEqual(report["inventory"]["attempt_count"], 4)
        self.assertEqual(report["inventory"]["code_attempt_count"], 4)
        self.assertEqual(len(report["grader_contract_notes"]), 1)
        note = report["grader_contract_notes"][0]
        self.assertEqual(note["kind"], "potential_specification_ambiguity")
        self.assertFalse(note["score_adjustment"])
        self.assertFalse(note["other_case_failures_explained"])
        self.assertEqual(note["evidence"]["sha256"], audit_sha)
        attempts = {row["call_id"]: row for row in report["attempts"]}
        for call_id in ("matched-pass", "matched-fail"):
            self.assertEqual(
                attempts[call_id]["grader_contract_note_ids"],
                [specialist_report.LEDGER_CONTRACT_NOTE_ID],
            )
        self.assertNotIn("grader_contract_note_ids", attempts["prompt-mismatch"])
        self.assertNotIn("grader_contract_note_ids", attempts["missing-case-hash"])
        self.assertTrue(attempts["matched-pass"]["success"])
        self.assertFalse(attempts["matched-fail"]["success"])
        self.assertEqual(attempts["matched-fail"]["assertions_passed"], 3)
        self.assertEqual(report["role_summaries"][0]["attempt_count"], 4)
        self.assertEqual(report["role_summaries"][0]["successful_attempt_count"], 3)
        csv_match = next(row for row in csv_rows if row["call_id"] == "matched-pass")
        self.assertEqual(csv_match["case_sha256"], "case-hash")
        self.assertIn(specialist_report.LEDGER_CONTRACT_NOTE_ID, csv_match["grader_contract_note_ids"])
        self.assertIn(specialist_report.LEDGER_CONTRACT_NOTE_SUMMARY, rendered)
        self.assertIn("Potential specification ambiguity", rendered)
        self.assertIn("supplemental; no score adjustment", rendered)
        self.assertIn(
            f'href="{specialist_report.LEDGER_CONTRACT_AUDIT_FILENAME}"', rendered
        )
        provenance = {Path(item["path"]).name: item for item in report["source_provenance"]}
        self.assertEqual(
            provenance[specialist_report.LEDGER_CONTRACT_AUDIT_FILENAME]["sha256"],
            audit_sha,
        )

    def test_ledger_schema_hashes_split_role_summaries_and_preserve_legacy_absence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "native-config.json"
            write_json(config, {"model": "/models/resident-alpha.gguf"})
            code_path = root / "schema-specialist-code.jsonl"
            prose_path = root / "schema-specialist-prose.jsonl"
            write_jsonl(prose_path, [])
            rows = [
                specialist_row(
                    root, "schema-legacy", "code_fresh_batch_module", "code",
                    context=16384, output_cap=8192, timeout=2400,
                ),
                specialist_row(
                    root, "schema-a", "code_fresh_batch_module", "code",
                    context=16384, output_cap=8192, timeout=2400,
                    ledger_schema_sha="schema-a-sha",
                ),
                specialist_row(
                    root, "schema-b", "code_fresh_batch_module", "code",
                    context=16384, output_cap=8192, timeout=2400,
                    ledger_schema_sha="schema-b-sha",
                ),
            ]
            write_jsonl(code_path, rows)
            study_receipt(
                root / "schema-study-trial.json", config, code_path, prose_path,
                actual_context=16384,
            )
            report = specialist_report.build_report(root)
            csv_rows = list(csv.DictReader(specialist_report.render_csv(report).splitlines()))

        summaries = [row for row in report["role_summaries"] if row["case_group"] == "code"]
        self.assertEqual(len(summaries), 3)
        by_schema = {row.get("ledger_json_schema_sha256"): row for row in summaries}
        self.assertEqual(set(by_schema), {None, "schema-a-sha", "schema-b-sha"})
        for summary in summaries:
            self.assertEqual(summary["attempt_count"], 1)
            self.assertEqual(summary["independent_task_count"], 1)
        attempts = {row["call_id"]: row for row in report["attempts"]}
        self.assertNotIn("ledger_json_schema_sha256", attempts["schema-legacy"])
        self.assertEqual(attempts["schema-a"]["ledger_json_schema_sha256"], "schema-a-sha")
        csv_schema = next(row for row in csv_rows if row["call_id"] == "schema-b")
        self.assertEqual(csv_schema["ledger_json_schema_sha256"], "schema-b-sha")

    def test_native_thinking_budgets_split_role_summaries_without_changing_task_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "native-config.json"
            write_json(config, {"model": "/models/resident-alpha.gguf"})
            code_path = root / "native-budget-specialist-code.jsonl"
            prose_path = root / "native-budget-specialist-prose.jsonl"
            write_jsonl(prose_path, [])
            rows = [
                specialist_row(
                    root, "legacy", "code_fresh_batch_module", "code",
                    context=16384, output_cap=8192, timeout=2400,
                ),
                specialist_row(
                    root, "native-1024", "code_fresh_batch_module", "code",
                    context=16384, output_cap=8192, timeout=2400,
                    native_thinking_budget=1024,
                ),
                specialist_row(
                    root, "native-4096", "code_fresh_batch_module", "code",
                    context=16384, output_cap=8192, timeout=2400,
                    native_thinking_budget=4096,
                ),
            ]
            write_jsonl(code_path, rows)
            study_receipt(
                root / "native-budget-study-trial.json", config, code_path, prose_path,
                actual_context=16384,
            )
            report = specialist_report.build_report(root)
            csv_rows = list(csv.DictReader(specialist_report.render_csv(report).splitlines()))

        summaries = [row for row in report["role_summaries"] if row["case_group"] == "code"]
        self.assertEqual(len(summaries), 3)
        by_budget = {
            row.get("native_thinking_budget_tokens_requested"): row for row in summaries
        }
        self.assertEqual(set(by_budget), {None, 1024, 4096})
        for summary in summaries:
            self.assertEqual(summary["attempt_count"], 1)
            self.assertEqual(summary["unique_case_count"], 1)
            self.assertEqual(summary["independent_task_count"], 1)
        attempts = {row["call_id"]: row for row in report["attempts"]}
        self.assertNotIn("native_thinking_budget_tokens_requested", attempts["legacy"])
        self.assertEqual(
            attempts["native-1024"]["native_thinking_budget_tokens_requested"], 1024,
        )
        csv_native = next(row for row in csv_rows if row["call_id"] == "native-4096")
        self.assertEqual(csv_native["native_thinking_budget_tokens_requested"], "4096")

    def test_sampler_recipes_split_role_summaries_and_do_not_leak_into_blind_view(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "native-config.json"
            write_json(config, {"model": "/models/resident-alpha.gguf"})
            code_path = root / "recipe-specialist-code.jsonl"
            prose_path = root / "recipe-specialist-prose.jsonl"
            recipe_a = {
                "label": "publisher-inspired-a", "sha256": "recipe-a",
                "sources": ["https://example.test/a"],
                "sampler_overrides": {"temperature": 1.0, "top_p": 0.95, "top_k": 20},
            }
            recipe_b = {
                "label": "publisher-inspired-b", "sha256": "recipe-b",
                "sources": ["https://example.test/b"],
                "sampler_overrides": {"temperature": 1.0, "top_p": 0.8, "top_k": 20},
            }
            code_rows = [
                specialist_row(
                    root, "recipe-a-code", "code_fresh_batch_module", "code",
                    context=16384, output_cap=8192, timeout=2400, temperature=1.0,
                    sampler_recipe=recipe_a,
                ),
                specialist_row(
                    root, "recipe-b-code", "code_fresh_batch_module", "code",
                    context=16384, output_cap=8192, timeout=2400, temperature=1.0,
                    sampler_recipe=recipe_b,
                ),
            ]
            prose_row = specialist_row(
                root, "recipe-a-prose", "prose_grounded_public_briefing", "prose",
                context=16384, output_cap=8192, timeout=2400, temperature=1.0,
                sampler_recipe=recipe_a,
            )
            write_jsonl(code_path, code_rows)
            write_jsonl(prose_path, [prose_row])
            study_receipt(
                root / "recipe-study-trial.json", config, code_path, prose_path,
                actual_context=16384,
            )

            report = specialist_report.build_report(root)
            package, mapping = specialist_report.build_blind_documents(root)
            csv_rows = list(csv.DictReader(specialist_report.render_csv(report).splitlines()))
            rendered = specialist_report.render_html(report)

        code_summaries = [row for row in report["role_summaries"] if row["case_group"] == "code"]
        self.assertEqual(len(code_summaries), 2)
        self.assertEqual({row["sampler_recipe_label"] for row in code_summaries}, {
            "publisher-inspired-a", "publisher-inspired-b",
        })
        self.assertEqual({row["sampling_config"]["request_overrides"]["top_p"] for row in code_summaries}, {
            0.95, 0.8,
        })
        attempts = {row["call_id"]: row for row in report["attempts"]}
        self.assertEqual(attempts["recipe-a-code"]["sampler_recipe_sha256"], "recipe-a")
        csv_attempt = next(row for row in csv_rows if row["call_id"] == "recipe-b-code")
        self.assertEqual(csv_attempt["sampler_recipe_label"], "publisher-inspired-b")
        self.assertIn('"top_p":0.8', csv_attempt["sampling_config"])
        self.assertIn("Sampling config", rendered)

        package_text = json.dumps(package, sort_keys=True)
        self.assertNotIn("sampler", package_text.lower())
        self.assertNotIn("publisher-inspired", package_text)
        mapping_row = next(row for row in mapping["entries"] if row["call_id"] == "recipe-a-prose")
        self.assertEqual(mapping_row["sampler_recipe_label"], "publisher-inspired-a")
        self.assertEqual(mapping_row["sampling_config"]["request_overrides"]["top_p"], 0.95)
        self.assertEqual(mapping_row["sampler_recipe_sources"], ["https://example.test/a"])

    def test_expanded_budget_controls_split_role_summaries_and_remain_visible(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "shared-config.json"
            write_json(config, {"model": "/models/resident-alpha.gguf"})
            historical_path = root / "historical-specialist-code.jsonl"
            expanded_path = root / "expanded-specialist-code.jsonl"
            prose_path = root / "unused-specialist-prose.jsonl"
            write_jsonl(prose_path, [])
            historical = specialist_row(
                root, "historical", "code_fresh_batch_module", "code",
                context=8192, output_cap=4096, timeout=900, thinking="medium", seed=0,
            )
            expanded_zero = specialist_row(
                root, "expanded-zero", "code_fresh_batch_module", "code",
                context=16384, output_cap=8192, timeout=2400, thinking="low", seed=0,
            )
            expanded_one = specialist_row(
                root, "expanded-one", "code_fresh_batch_module", "code",
                context=16384, output_cap=8192, timeout=2400, thinking="low", seed=1,
            )
            write_jsonl(historical_path, [historical])
            write_jsonl(expanded_path, [expanded_zero, expanded_one])
            study_receipt(
                root / "historical-study-trial.json", config, historical_path, prose_path,
                actual_context=8192,
            )
            study_receipt(
                root / "expanded-study-trial.json", config, expanded_path, prose_path,
                actual_context=16384,
            )
            report = specialist_report.build_report(root)
            rendered = specialist_report.render_html(report)
            csv_rows = list(csv.DictReader(specialist_report.render_csv(report).splitlines()))

        code_summaries = [row for row in report["role_summaries"] if row["case_group"] == "code"]
        self.assertEqual(len(code_summaries), 2)
        historical_summary = next(row for row in code_summaries if row["output_token_limit"] == 4096)
        expanded_summary = next(row for row in code_summaries if row["output_token_limit"] == 8192)
        self.assertEqual(historical_summary["requested_context_max_tokens"], 8192)
        self.assertEqual(historical_summary["actual_context_max_tokens"], 8192)
        self.assertEqual(historical_summary["request_timeout_seconds"], 900)
        self.assertEqual(historical_summary["thinking"], "medium")
        self.assertEqual(expanded_summary["requested_context_max_tokens"], 16384)
        self.assertEqual(expanded_summary["actual_context_max_tokens"], 16384)
        self.assertEqual(expanded_summary["request_timeout_seconds"], 2400)
        self.assertEqual(expanded_summary["thinking"], "low")
        self.assertEqual(expanded_summary["temperature"], 0.0)
        self.assertEqual(expanded_summary["sampling_seeds"], [0, 1])
        self.assertEqual(expanded_summary["attempt_count"], 2)
        self.assertEqual(expanded_summary["unique_case_count"], 1)
        self.assertEqual(expanded_summary["independent_task_count"], 1)
        expanded_attempt = next(row for row in report["attempts"] if row["call_id"] == "expanded-zero")
        self.assertEqual(expanded_attempt["output_token_limit"], 8192)
        self.assertEqual(expanded_attempt["request_timeout_seconds"], 2400)
        csv_attempt = next(row for row in csv_rows if row["call_id"] == "expanded-zero")
        self.assertEqual(csv_attempt["requested_context_max_tokens"], "16384")
        self.assertEqual(csv_attempt["actual_context_max_tokens"], "16384")
        self.assertEqual(csv_attempt["output_token_limit"], "8192")
        self.assertEqual(csv_attempt["request_timeout_seconds"], "2400")
        self.assertEqual(csv_attempt["thinking"], "low")
        self.assertEqual(csv_attempt["temperature"], "0.0")
        self.assertEqual(csv_attempt["sampling_seed"], "0")
        for expected in ("Requested context", "Actual context", "Output cap", "Request timeout s", "2400"):
            self.assertIn(expected, rendered)

    def test_standalone_report_includes_model_packing_overview_without_claiming_preview_verified(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_json(root / "flash-next-coder-header-preview.json", {
                "payload_fully_verified": False,
                "chunk_zero_manifest_hash_matches": True,
                "model_name": "Flash Coder fixture",
                "blocks": 48, "experts": 256, "experts_used": 10,
                "transformer_tensor_elements": 65_314_218_880,
                "shared_ple_tensor_elements": 51_200_245_760,
            })
            report = specialist_report.build_report(root)
            rendered = specialist_report.render_html(report)

        self.assertEqual(len(report["model_artifact_overview"]), 1)
        artifact = report["model_artifact_overview"][0]
        self.assertEqual(artifact["packing"], "IQ1_M")
        self.assertFalse(artifact["verified"])
        self.assertIsNone(artifact["recorded_weight_bytes"])
        self.assertIsNone(artifact["all_gpu_fit_claimed"])
        self.assertIn("effective compression after expert pruning", artifact["quantization_caveat"])
        self.assertIn("effective compression after expert pruning", rendered)
        self.assertIn("116514464640", rendered)
        self.assertIn("A blank all-GPU field means", rendered)
        self.assertIn(
            "flash-next-coder-header-preview.json",
            {Path(item["path"]).name for item in report["source_provenance"]},
        )

    def test_role_summary_uses_native_rate_weighting_and_keeps_prose_unscored(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "model-config.json"
            write_json(config, {"model": "/models/resident-alpha.gguf"})
            code_path = root / "alpha-specialist-code.jsonl"
            prose_path = root / "alpha-specialist-prose.jsonl"
            code_rows = [
                specialist_row(root, "code-a", "code_fresh_batch_module", "code", rate=10, decode_seconds=1, reasoning_tokens=7),
                specialist_row(root, "code-b", "code_multifunction_ledger_bugfix", "code", rate=20, decode_seconds=3),
            ]
            prose_row = specialist_row(root, "prose-a", "prose_grounded_public_briefing", "prose")
            empty_row = specialist_row(root, "prose-empty", "prose_constrained_last_tram", "prose")
            empty_path = Path(empty_row["visible_text"]["path"])
            empty_path.write_text("", encoding="utf-8")
            empty_row["visible_text"].update({
                "sha256": hashlib.sha256(b"").hexdigest(), "chars": 0, "bytes": 0,
            })
            empty_row["response"]["stream_complete"] = False
            empty_row["response_metrics"]["stream_complete"] = False
            empty_row["objective_outcome"] = "incomplete_stream"
            empty_row["evaluation"].update({"outcome": "incomplete_stream", "passed": False})
            empty_row["eligibility"].update({
                "speed_eligible": False, "prose_constraint_eligible": False,
                "blinded_prose_review_eligible": False,
            })
            write_jsonl(code_path, code_rows)
            write_jsonl(prose_path, [prose_row, empty_row])
            study_receipt(root / "alpha-study-trial.json", config, code_path, prose_path)
            report = specialist_report.build_report(root)
            rendered = specialist_report.render_html(report)

        summaries = {row["case_group"]: row for row in report["role_summaries"]}
        code = summaries["code"]
        prose = summaries["prose"]
        self.assertEqual(code["independent_task_count"], 2)
        self.assertEqual(code["successful_attempt_count"], 2)
        self.assertEqual(code["input_tokens_sum"], 200.0)
        self.assertEqual(code["output_tokens_sum"], 50.0)
        self.assertEqual(code["native_reported_decode_tps_weighted_by_native_decode_seconds"], 17.5)
        self.assertEqual(code["native_decode_weight_seconds"], 4.0)
        self.assertEqual(prose["independent_task_count"], 0)
        self.assertEqual(prose["successful_attempt_count"], 0)
        self.assertEqual(prose["unique_case_count"], 2)
        self.assertEqual(prose["completed_attempt_count"], 1)
        self.assertEqual(prose["failed_completion_attempt_count"], 1)
        attempt = next(row for row in report["attempts"] if row["call_id"] == "prose-a")
        empty_attempt = next(row for row in report["attempts"] if row["call_id"] == "prose-empty")
        self.assertTrue(attempt["mechanical_constraints_passed"])
        self.assertIsNone(attempt["success"])
        self.assertEqual(attempt["quality_status"], "unscored_pending_blinded_review")
        self.assertFalse(empty_attempt["completion_success"])
        self.assertEqual(empty_attempt["quality_status"], "failed_completion_not_review_eligible")
        self.assertEqual(report["startup_observations"][0]["startup_seconds"], 12.5)
        self.assertIn("not a cold-load claim", report["startup_observations"][0]["cache_state_note"])
        self.assertNotIn("fabricated final answer", rendered)

    def test_review_is_hash_bound_and_blind_package_separates_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "model-config.json"
            write_json(config, {"model": "/models/resident-alpha.gguf"})
            code_path = root / "alpha-specialist-code.jsonl"
            prose_path = root / "alpha-specialist-prose.jsonl"
            write_jsonl(code_path, [])
            first = specialist_row(root, "secret-call-a", "prose_grounded_public_briefing", "prose", model="resident-alpha")
            second = specialist_row(root, "secret-call-b", "prose_grounded_public_briefing", "prose", model="resident-beta")
            partial = specialist_row(root, "secret-call-partial", "prose_grounded_public_briefing", "prose", model="resident-gamma")
            partial["response"]["truncated"] = True
            partial["response"]["done_reason"] = "length"
            partial["response_metrics"]["truncated"] = True
            partial["objective_outcome"] = "truncated"
            partial["evaluation"].update({"outcome": "truncated", "passed": False})
            partial["eligibility"].update({
                "speed_eligible": False, "prose_constraint_eligible": False,
                "blinded_prose_review_eligible": False,
            })
            write_jsonl(prose_path, [first, second, partial])
            study_receipt(root / "alpha-study-trial.json", config, code_path, prose_path)

            case = next(item for item in specialist_cases.CASES if item.case_id == first["case_id"])
            rubric_scores = [
                {
                    "criterion_id": item.criterion_id,
                    "score": item.max_points,
                    "max_points": item.max_points,
                    "comment": "fixture assessment",
                }
                for item in case.rubric
            ]
            review_path = root / "specialist-prose-review.json"
            write_json(review_path, {
                "schema_version": 1,
                "reviewer_label": "assistant reviewer fixture",
                "review_method": specialist_report.REVIEW_METHOD,
                "limitations": specialist_report.REVIEW_LIMITATION,
                "reviews": {
                    first["call_id"]: {
                        "case_id": first["case_id"],
                        "visible_text_sha256": first["visible_text"]["sha256"],
                        "rubric_scores": rubric_scores,
                        "total_score": sum(item.max_points for item in case.rubric),
                        "comment": "fixture total",
                    }
                },
            })
            report = specialist_report.build_report(root, review_path)
            package, mapping = specialist_report.build_blind_documents(root)
            stale = json.loads(review_path.read_text(encoding="utf-8"))
            stale["reviews"][first["call_id"]]["visible_text_sha256"] = "stale-hash"
            write_json(review_path, stale)
            with self.assertRaisesRegex(ValueError, "text hash"):
                specialist_report.build_report(root, review_path)
            invalid_partial = json.loads(json.dumps(stale))
            invalid_partial["reviews"] = {
                partial["call_id"]: dict(stale["reviews"][first["call_id"]])
            }
            invalid_partial["reviews"][partial["call_id"]].update({
                "visible_text_sha256": partial["visible_text"]["sha256"],
            })
            write_json(review_path, invalid_partial)
            with self.assertRaisesRegex(ValueError, "not an eligible completed"):
                specialist_report.build_report(root, review_path)

        reviewed = next(row for row in report["attempts"] if row["call_id"] == first["call_id"])
        incomplete = next(row for row in report["attempts"] if row["call_id"] == partial["call_id"])
        self.assertEqual(reviewed["quality_status"], "reviewed")
        self.assertFalse(incomplete["completion_success"])
        self.assertEqual(incomplete["quality_status"], "failed_completion_not_review_eligible")
        self.assertEqual(report["review"]["reviewer_label"], "assistant reviewer fixture")
        package_text = json.dumps(package)
        self.assertNotIn("secret-call", package_text)
        self.assertNotIn("resident-alpha", package_text)
        self.assertNotIn("resident-beta", package_text)
        self.assertNotIn("resident-gamma", package_text)
        self.assertNotIn("reasoning_controls_requested", package_text)
        self.assertNotIn("source_config", package_text)
        def nested_keys(value):
            if isinstance(value, dict):
                return set(value) | set().union(*(nested_keys(item) for item in value.values()))
            if isinstance(value, list):
                return set().union(*(nested_keys(item) for item in value))
            return set()
        self.assertTrue({
            "thinking", "temperature", "sampling_seed", "sampling_seeds",
            "requested_context_max_tokens", "actual_context_max_tokens",
            "output_token_limit", "request_timeout_seconds",
        }.isdisjoint(nested_keys(package)))
        answers = package["cases"][0]["answers"]
        self.assertEqual(len(answers), 3)
        self.assertEqual(len({item["label"] for item in answers}), 3)
        self.assertTrue(all(set(item) == {"label", "text", "sha256", "review_eligible"} for item in answers))
        self.assertEqual(set(package["label_scores_template"]), {item["label"] for item in answers})
        partial_answer = next(item for item in answers if item["review_eligible"] is False)
        self.assertEqual(package["label_scores_template"][partial_answer["label"]]["rubric_scores"], [])
        self.assertEqual(
            {item["call_id"] for item in mapping["entries"]},
            {"secret-call-a", "secret-call-b", "secret-call-partial"},
        )
        self.assertTrue(all(item["visible_text_sha256"] for item in mapping["entries"]))


if __name__ == "__main__":
    unittest.main()
