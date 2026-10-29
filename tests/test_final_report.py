import csv
import json
import tempfile
import unittest
from pathlib import Path

from runtime import final_report


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def request_row(
    call_id,
    *,
    case_id="hard_transaction_ledger",
    case_version="2026-10-19.2",
    model="model<unsafe>",
    prompt_hash="prompt-a",
    repeat=0,
    temperature=0.0,
    seed=0,
    output_limit=2048,
    reasoning_budget_tokens=None,
    strata_pcie_frac=None,
    strata_aux_cpus=None,
    chat_template_thinking=None,
    passed=True,
    truncated=False,
    stream_complete=True,
    error=None,
):
    response = None if error else {
        "reasoning_chars": 31,
        "timing": {"wall_seconds": 4.5, "ttft_final_content_seconds": 1.25},
        "actual_prompt_tokens": 400,
        "actual_completion_tokens": 80,
        "done_reason": "length" if truncated else "stop",
        "stream_complete": stream_complete,
        "truncated": truncated,
        "throughput": {
            "prompt_tokens_per_second_native": 100.0,
            "completion_tokens_per_second_native": 12.0,
            "completion_tokens_per_second_wall": 17.7,
        },
        "openai_usage": {
            "prompt_tokens": 400,
            "completion_tokens": 80,
            "prompt_tokens_details": {"cached_tokens": 40},
        },
        "openai_native_timings": {
            "prompt_n": 360, "cache_n": 40, "prompt_ms": 2000,
            "predicted_n": 77, "predicted_ms": 3000,
        },
    }
    return {
        "schema_version": 1,
        "run_id": "bench-run",
        "call_id": call_id,
        "started_at": "2026-10-19T01:00:00Z",
        "finished_at": "2026-10-19T01:00:05Z",
        "repeat_index": repeat,
        "suite": "hard" if case_id.startswith("hard_") else "quality",
        "case_id": case_id,
        "case_version": case_version,
        "case_source_sha256": "case-source",
        "case_metadata": {"hidden_test_count": 37},
        "request": {
            "endpoint": "http://user:secret@127.0.0.1:18080/v1?key=secret",
            "backend": "openai",
            "model": model,
            "context_max_tokens": 32768,
            "thinking": "medium",
            "temperature": temperature,
            "seed": seed,
            "output_token_limit": output_limit,
            "reasoning_budget_tokens": reasoning_budget_tokens,
            "strata_pcie_frac": strata_pcie_frac,
            "strata_aux_cpus": strata_aux_cpus,
            "chat_template_thinking": chat_template_thinking,
            "prompt_sha256": prompt_hash,
        },
        "response": response,
        "evaluation": {"passed": passed, "detail": "fixture <detail>"},
        "error": error,
    }


def lifecycle(path, result_paths, *, environment=None, private_config=None, source_config=None):
    rows = []
    for index, result in enumerate(result_paths):
        run_id = f"{path.stem}-trial-{index}"
        rows.extend([
            {
                "at": "2026-10-19T00:59:00Z",
                "event": "trial_started",
                "run_id": run_id,
                "tag": f"tag-{index}",
                "model_name": "model<unsafe>",
                "source_config": str(source_config) if source_config else None,
                "private_config": str(private_config) if private_config else None,
                "engine": "/tmp/strata",
                "engine_environment": environment or {},
                "context": 32768,
            },
            {
                "at": "2026-10-19T01:00:00Z",
                "event": "suite_started",
                "run_id": run_id,
                "suite": "hard",
                "result": str(result),
            },
        ])
    write_jsonl(path, rows)


class FinalReportTests(unittest.TestCase):
    def test_user_deferral_preserves_completed_requests_without_fabricating_attempts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result_path = root / "gemma-quick-results.jsonl"
            write_jsonl(result_path, [
                request_row("gemma-capped", model="gemma-4-12b-styletune-q5_k_m",
                            passed=False, truncated=True),
                request_row("gemma-passed", model="gemma-4-12b-styletune-q5_k_m"),
            ])
            write_json(root / "specialists-gemma-user-deferral-v1.json", {
                "schema_version": 1, "kind": "user_requested_model_deferral",
                "deferred_models": ["gemma4-style12-q5_k_m"],
                "coverage": {
                    "planned_primary_attempt_slots": 15,
                    "terminal_completed_attempts": 2,
                    "active_request_interrupted_by_user": 1,
                    "remaining_unexecuted_primary_slots": 12,
                },
                "private_or_unrelated": "must not be copied into the note",
            })
            report = final_report.build_report(root)
        self.assertEqual({row["call_id"] for row in report["request_rows"]},
                         {"gemma-capped", "gemma-passed"})
        self.assertEqual(report["inventory"]["request_rows"], 2)
        note = next(row for row in report["not_run"] if row.get("status") ==
                    "user_deferred_partial_results_preserved")
        self.assertEqual(note["coverage"]["active_request_interrupted_by_user"], 1)
        self.assertEqual(note["coverage"]["remaining_unexecuted_primary_slots"], 12)
        self.assertNotIn("private_or_unrelated", note)
        self.assertNotIn("must not be copied", json.dumps(note))
        self.assertTrue(any(Path(source["path"]).name ==
                            "specialists-gemma-user-deferral-v1.json"
                            for source in report["source_provenance"]))

    def test_user_deferral_missing_counts_remain_unknown_and_unrelated_notes_are_ignored(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "specialists-gemma-user-deferral-v1.json"
            document = {
                "schema_version": 1, "kind": "user_requested_model_deferral",
                "deferred_models": ["gemma4-style12-q5_k_m"],
            }
            write_json(path, document)
            notes, _ = final_report._not_run(root, [])
            self.assertTrue(all(value is None for value in notes[0]["coverage"].values()))
            document["deferred_models"] = ["another-model"]
            write_json(path, document)
            self.assertEqual(final_report._not_run(root, [])[0], [])

    def test_specialist_receipt_metrics_and_prose_constraints_stay_unscored(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "specialist-source.json"
            result = root / "fixture-specialist-prose.jsonl"
            text_path = root / "visible.txt"
            text_path.write_text("fabricated visible answer", encoding="utf-8")
            write_json(config, {"model": "/models/fixture.gguf"})
            raw = {
                "schema_version": 1,
                "run_id": "specialist-run",
                "call_id": "specialist-prose-call",
                "started_at": "2026-10-19T01:00:00Z",
                "finished_at": "2026-10-19T01:00:05Z",
                "repeat_index": 0,
                "model": "resident-specialist",
                "profile": "prose-role",
                "suite": "specialist_prose",
                "case_id": "prose_grounded_public_briefing",
                "case_group": "prose",
                "case_version": "2026-10-19.specialist.2",
                "case_source_sha256": "case-source",
                "request": {
                    "endpoint": "http://127.0.0.1:8080/v1",
                    "backend": "openai",
                    "model": "resident-specialist",
                    "profile": "prose-role",
                    "thinking": "none",
                    "temperature": 0.7,
                    "seed": 0,
                    "context_max_tokens": 8192,
                    "output_token_limit": 2048,
                    "timeout_seconds": 900,
                    "prompt_sha256": "specialist-prompt",
                    "reasoning_controls_requested": {
                        "reasoning_effort": "none", "chat_template_thinking": "off",
                        "reasoning_budget_tokens": None,
                    },
                    "capability_note": "reasoning toggle is model-specific",
                },
                "response": {"stream_complete": True, "truncated": False, "done_reason": "stop"},
                "response_metrics": {
                    "input_tokens_reported": 321,
                    "output_tokens_reported": 87,
                    "reasoning_tokens_reported": 5,
                    "reasoning_chars_observed": 19,
                    "reasoning_stream_observed": True,
                    "wall_seconds": 9.5,
                    "ttft_visible_seconds": 1.5,
                    "native_decode_tokens": 86,
                    "native_decode_seconds": 4.0,
                    "native_decode_rate_tokens_per_second": 20.0,
                },
                "visible_text": {
                    "path": str(text_path), "sha256": "visible-sha", "chars": 25, "bytes": 25,
                    "reasoning_excluded": True,
                },
                "objective_outcome": "constraints_passed",
                "error": None,
                "evaluation": {
                    "outcome": "constraints_passed", "passed": True,
                    "deterministic_constraints_only": True,
                    "subjective_quality_score": None,
                    "subjective_quality_status": "unscored_pending_blinded_review",
                    "detail": "mechanical constraints checked; prose quality remains unscored",
                },
                "eligibility": {
                    "speed_eligible": True,
                    "coding_objective_eligible": False,
                    "prose_constraint_eligible": True,
                    "blinded_prose_review_eligible": True,
                    "reasoning_metric_eligible": True,
                },
            }
            write_jsonl(result, [raw])
            write_json(root / "fixture-study-trial.json", {
                "schema_version": 1,
                "tag": "fixture",
                "status": "complete",
                "runtime_backend": "llama",
                "source_config": {"path": str(config), "sha256": "config-sha"},
                "controls": {"profile": "prose-role"},
                "session": {
                    "run_id": "owned-model-session", "model_name": "resident-specialist",
                    "context": 8192, "initial_load_seconds": 12.25, "hold_seconds": 40.0,
                },
                "artifacts": {
                    "code": {"path": str(root / "fixture-specialist-code.jsonl"), "exists": False},
                    "prose": {"path": str(result), "exists": True},
                },
            })
            report = final_report.build_report(root)
            rejected_raw = json.loads(json.dumps(raw))
            rejected_raw.update({
                "call_id": "specialist-code-rejected", "suite": "specialist_code",
                "case_id": "code_fresh_batch_module", "case_group": "code",
                "objective_outcome": "evaluator_rejected",
            })
            rejected_raw["evaluation"] = {
                "outcome": "evaluator_rejected", "passed": False, "validation": "rejected",
                "detail": "candidate source used behavior blocked by the restricted evaluator",
            }
            rejected_raw["eligibility"].update({
                "coding_objective_eligible": False, "prose_constraint_eligible": False,
                "blinded_prose_review_eligible": False,
            })
            rejected = final_report.normalize_request(
                rejected_raw, root / "rejected.jsonl", 1, {}, {}, {}, {},
            )
            wrong_raw = json.loads(json.dumps(rejected_raw))
            wrong_raw["call_id"] = "specialist-code-wrong"
            wrong_raw["objective_outcome"] = "wrong_answer"
            wrong_raw["evaluation"].update({
                "outcome": "wrong_answer", "detail": "candidate source violated executable prompt requirements",
            })
            wrong_raw["eligibility"]["coding_objective_eligible"] = True
            wrong = final_report.normalize_request(
                wrong_raw, root / "wrong.jsonl", 1, {}, {}, {}, {},
            )
            expanded_raw = json.loads(json.dumps(raw))
            expanded_raw["call_id"] = "specialist-prose-expanded"
            expanded_raw["request"]["timeout_seconds"] = 2400
            normalized_standard = final_report.normalize_request(
                raw, root / "standard.jsonl", 1, {}, {}, {}, {},
            )
            normalized_expanded = final_report.normalize_request(
                expanded_raw, root / "expanded.jsonl", 1, {}, {}, {}, {},
            )
            legacy = final_report.normalize_request(
                request_row("legacy-without-timeout"), root / "legacy.jsonl", 1,
                {}, {}, {}, {},
            )

        row = report["request_rows"][0]
        self.assertEqual(row["outcome"], "constraints_passed")
        self.assertTrue(row["mechanical_constraints_passed"])
        self.assertIsNone(row["content_passed"])
        self.assertIsNone(row["comparison_content_passed"])
        self.assertFalse(row["quality_eligible"])
        self.assertTrue(row["speed_eligible"])
        self.assertEqual(row["reasoning_tokens"], 5)
        self.assertEqual(row["native_decode_n"], 86)
        self.assertEqual(row["native_decode_seconds"], 4.0)
        self.assertEqual(row["native_decode_tps"], 20.0)
        self.assertEqual(row["actual_context_max_tokens"], 8192)
        self.assertEqual(row["request_timeout_seconds"], 900)
        self.assertEqual(row["startup_seconds"], 12.25)
        self.assertEqual(row["model_session_run_id"], "owned-model-session")
        self.assertTrue(row["source_config"].endswith("specialist-source.json"))
        self.assertEqual(rejected["outcome"], "evaluator_rejected")
        self.assertFalse(rejected["quality_eligible"])
        self.assertIsNone(rejected["comparison_content_passed"])
        self.assertEqual(wrong["outcome"], "wrong_answer")
        self.assertTrue(wrong["quality_eligible"])
        self.assertFalse(wrong["comparison_content_passed"])
        self.assertNotEqual(normalized_standard["profile_id"], normalized_expanded["profile_id"])
        self.assertNotIn("request_timeout_seconds", legacy)

    def test_dunder_policy_rejection_preserves_workflow_failure_without_claiming_wrong_semantics(self):
        raw = request_row("rejected", case_id="quality_deep_merge", passed=False)
        raw["evaluation"].update(
            validation="rejected", detail="private or dunder attribute access is not allowed",
        )
        row = final_report.normalize_request(raw, Path("/tmp/rejected-quality.jsonl"), 1, {}, {}, {}, {})
        self.assertEqual(row["outcome"], "evaluator_rejected")
        self.assertFalse(row["content_passed"])
        self.assertIsNone(row["comparison_content_passed"])
        self.assertFalse(row["quality_eligible"])
        self.assertTrue(row["speed_eligible"])
        raw["evaluation"]["detail"] = "imports are not allowed"
        row = final_report.normalize_request(raw, Path("/tmp/rejected-quality.jsonl"), 1, {}, {}, {}, {})
        self.assertEqual(row["outcome"], "wrong_answer")
        self.assertFalse(row["comparison_content_passed"])

    def test_weight_quant_prefers_explicit_q3_and_preserves_unknown_future_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            q3_config = root / "opaque-runtime.json"
            q4_config = root / "qwen36-UD-Q4_K_XL-runtime.json"
            write_json(q3_config, {
                "model": "/models/Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf",
                "model_name": "qwen3.6-35b-a3b-ud-q3_k_xl",
            })
            write_json(q4_config, {"model": "/models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf"})

            from_weight_path = final_report._weight_quant(
                "qwen3.6-35b-a3b", {}, {"source_config": str(q3_config)}
            )
            explicit_model_wins = final_report._weight_quant(
                "qwen3.6-35b-a3b-ud-q3_k_xl", {}, {"source_config": str(q4_config)}
            )
            unknown_future = final_report._weight_quant(
                "qwen3.6-35b-a3b-ud-future_xl", {}, {}
            )
            architecture_only = final_report._weight_quant("qwen3.6-35b-a3b", {}, {})

        self.assertEqual(from_weight_path, "UD-Q3_K_XL")
        self.assertEqual(explicit_model_wins, "UD-Q3_K_XL")
        self.assertEqual(unknown_future, "UD-FUTURE_XL")
        self.assertNotEqual(unknown_future, "UD-Q4_K_XL")
        self.assertEqual(architecture_only, "UD-Q4_K_XL")

    def test_bonsai_request_rows_preserve_exact_packing_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for packing in ("PQ2_0", "PTQ1_0"):
                with self.subTest(packing=packing):
                    model = f"bonsai2-27b-{packing.lower()}"
                    config = root / f"{packing}.json"
                    write_json(config, {
                        "model": f"/models/Ternary-Bonsai-2-27B-{packing}.gguf",
                    })
                    row = final_report.normalize_request(
                        request_row(f"packing-{packing}", model=model),
                        root / f"{packing}.jsonl", 1,
                        {"runtime_backend": "llama", "source_config": str(config)},
                        {}, {}, {},
                    )
                    self.assertEqual(row["weight_quant"], packing)
                    self.assertEqual(final_report._weight_quant(
                        "bonsai-opaque", {}, {"source_config": str(config)},
                    ), packing)
        self.assertIsNone(final_report._explicit_quant("bonsai2-27b"))

    def test_custom_runtime_label_uses_matching_manifest_without_model_name_heuristics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "opaque-config.json"
            manifest = root / "build.json"
            server = root / "custom-bin" / "llama-server"
            write_json(config, {"server": str(server), "artifact_manifest": "build.json"})
            write_json(manifest, {
                "source": {
                    "llama_repo": "https://github.com/PrismML-Eng/llama.cpp.git",
                    "llama_revision": "e3de715222b2667c54a267441159a5729f11155e",
                },
                "runtime": {
                    "release": "prism-b10777-e3de715",
                    "server": {"path": str(server), "sha256": "a" * 64},
                },
            })
            row = final_report.normalize_request(
                request_row("custom-runtime", model="opaque-model"), root / "calls.jsonl", 1,
                {"runtime_backend": "llama", "source_config": str(config), "engine": str(server)},
                {}, {}, {},
            )
        self.assertEqual(row["runtime"], "llama.cpp server (prism-b10777-e3de715)")

    def test_custom_runtime_mismatch_or_incomplete_identity_does_not_claim_bundle_or_release(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "runtime.json"
            manifest = root / "build.json"
            server = root / "bin" / "llama-server"
            write_json(config, {"server": str(server), "artifact_manifest": str(manifest)})
            base = {
                "source": {
                    "llama_repo": "https://github.com/PrismML-Eng/llama.cpp.git",
                    "llama_revision": "e3de715222b2667c54a267441159a5729f11155e",
                },
                "runtime": {
                    "release": "prism-b10777-e3de715",
                    "server": {"path": str(server), "sha256": "a" * 64},
                },
            }
            trial = {"runtime_backend": "llama", "source_config": str(config)}
            for mismatch in ("recorded_binary", "manifest_binary", "missing_release", "missing_hash"):
                with self.subTest(mismatch=mismatch):
                    payload = json.loads(json.dumps(base))
                    observed_trial = dict(trial)
                    if mismatch == "recorded_binary":
                        observed_trial["engine"] = str(root / "other" / "llama-server")
                    elif mismatch == "manifest_binary":
                        payload["runtime"]["server"]["path"] = str(root / "other" / "llama-server")
                    elif mismatch == "missing_release":
                        del payload["runtime"]["release"]
                    else:
                        del payload["runtime"]["server"]["sha256"]
                    write_json(manifest, payload)
                    label = final_report._runtime_name(observed_trial, "opaque-model")
                    self.assertEqual(label, "llama.cpp server (Prism custom build; identity unverified)")

    def test_deduplicates_call_ids_but_preserves_repeat_attempts_and_token_semantics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "round-hard.jsonl"
            first = request_row("call-a", repeat=0, seed=10)
            second = request_row("call-b", repeat=1, seed=11)
            write_jsonl(source, [first, first, second])
            report = final_report.build_report(root)

        self.assertEqual(report["inventory"]["request_rows_before_dedup"], 3)
        self.assertEqual(report["inventory"]["request_rows"], 2)
        self.assertIn("call-a", report["inventory"]["duplicate_call_ids"])
        profile = report["profile_summaries"][0]
        self.assertEqual(profile["attempt_count"], 2)
        self.assertEqual(profile["independent_task_count"], 1)
        self.assertEqual([row["sampling_seed"] for row in report["request_rows"]], [10, 11])
        row = report["request_rows"][0]
        self.assertEqual(row["openai_prompt_tokens"], 400)
        self.assertEqual(row["native_prompt_n"], 360)
        self.assertEqual(row["native_cache_n"], 40)
        self.assertEqual(row["api_cached_prompt_tokens"], 40)
        self.assertEqual(row["native_decode_n"], 77)
        self.assertEqual(row["native_prompt_seconds"], 2.0)
        self.assertEqual(row["native_decode_seconds"], 3.0)
        self.assertIsNone(row["reasoning_tokens"])
        self.assertIsNone(row["visible_answer_tokens"])
        self.assertNotIn("secret", json.dumps(report))

    def test_native_draft_counters_and_rate_preserve_missing_and_zero_denominator(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            proposed = request_row("drafted")
            proposed["response"]["openai_native_timings"].update({"draft_n": 20, "draft_n_accepted": 5})
            zero = request_row("zero-drafts", model="zero-model")
            zero["response"]["openai_native_timings"].update({"draft_n": 0, "draft_n_accepted": 0})
            ollama = request_row("ollama-no-drafts", model="ollama-model")
            ollama["request"]["backend"] = "ollama"
            ollama["response"].pop("openai_native_timings")
            write_jsonl(root / "drafts.jsonl", [proposed, zero, ollama])
            report = final_report.build_report(root)
            parsed_csv = {row["row_id"]: row for row in csv.DictReader(final_report.render_csv(report).splitlines())}
            rendered = final_report.render_html(report)

        rows = {row["row_id"]: row for row in report["request_rows"]}
        self.assertEqual(rows["drafted"]["native_draft_n"], 20)
        self.assertEqual(rows["drafted"]["native_draft_n_accepted"], 5)
        self.assertEqual(rows["drafted"]["native_draft_acceptance_rate"], 0.25)
        self.assertEqual(parsed_csv["drafted"]["native_draft_acceptance_rate"], "0.25")
        self.assertEqual(rows["zero-drafts"]["native_draft_n"], 0)
        self.assertIsNone(rows["zero-drafts"]["native_draft_acceptance_rate"])
        self.assertIsNone(rows["ollama-no-drafts"]["native_draft_n"])
        self.assertIsNone(rows["ollama-no-drafts"]["native_draft_n_accepted"])
        self.assertIsNone(rows["ollama-no-drafts"]["native_draft_acceptance_rate"])
        self.assertIn("Draft proposed", rendered)
        self.assertIn("Draft acceptance rate", rendered)

    def test_missing_values_and_incomplete_or_timeout_outcomes_are_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            incomplete = request_row("cut", passed=False, stream_complete=False)
            timeout = request_row(
                "timeout", passed=False,
                error={"type": "TimeoutError", "message": "request timed out"},
            )
            write_jsonl(root / "fail-hard.jsonl", [incomplete, timeout])
            write_jsonl(root / "failed-runtime-events.jsonl", [
                {
                    "at": "2026-10-19T00:00:00Z", "event": "trial_started",
                    "run_id": "failed-start", "model_name": "model-a", "context": 32768,
                },
                {
                    "at": "2026-10-19T00:00:03Z", "event": "trial_failed",
                    "run_id": "failed-start", "error": "engine refused model",
                },
            ])
            write_json(root / "initial-probe.json", {
                "model": "gpt-oss-20b:64k", "runtime": "Ollama 0.20.0",
                "context": 32768, "thinking": "low", "outcome": "timeout",
                "wall_seconds": 60.0, "response_bytes": 0,
                "notes": ["service health observation"],
            })
            report = final_report.build_report(root)

        by_id = {row["row_id"]: row for row in report["request_rows"]}
        self.assertEqual(by_id["cut"]["outcome"], "incomplete_stream")
        self.assertFalse(by_id["cut"]["quality_eligible"])
        self.assertEqual(by_id["timeout"]["outcome"], "timeout")
        self.assertIsNone(by_id["timeout"]["wall_seconds"])
        self.assertIsNone(by_id["timeout"]["completion_tokens"])
        setup = next(row for row in report["supplemental_rows"] if row["row_type"] == "runtime_trial")
        self.assertEqual(setup["outcome"], "setup_error")
        self.assertEqual(setup["evaluation_detail"], "engine refused model")
        initial = next(row for row in report["supplemental_rows"] if row["source_file"].endswith("initial-probe.json"))
        self.assertEqual(initial["outcome"], "timeout")
        self.assertEqual(initial["response_bytes"], 0)
        self.assertEqual(initial["runtime"], "Ollama 0.20.0")

    def test_quality_format_and_timing_exclusion_policies_are_independent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old = request_row(
                "old-constraints", case_id="quality_constraints",
                case_version=final_report.AMBIGUOUS_CONSTRAINT_VERSION,
            )
            quick = request_row(
                "quick", case_id="quick_generation", case_version="2026-10-18.3",
                output_limit=224,
            )
            controlled = request_row("control", passed=False)
            controlled["response"]["response_text"] = "{}"
            source = root / "control-hard.jsonl"
            write_jsonl(source, [old, quick, controlled])
            write_json(root / "newer-20b-quality-control-annotation.json", {
                "timing_valid_for_speed_comparison": False,
                "result_expected": str(source),
                "reason": "concurrent CPU build",
            })
            write_json(root / "hard-json-format-audit.json", {
                "rows": [{"call_id": "control", "raw_json_object": False, "outer_code_fence": True}],
            })
            write_json(root / "hard-round-report.json", {
                "groups": [{"tasks": [{
                    "call_id": "control", "rescored_outcome": "passed",
                    "rescored_cause": "recorded_evaluator_builtin_limitation",
                }]}],
            })
            report = final_report.build_report(root)

        rows = {row["row_id"]: row for row in report["request_rows"]}
        self.assertFalse(rows["old-constraints"]["quality_eligible"])
        self.assertFalse(rows["quick"]["quality_eligible"])
        self.assertFalse(rows["quick"]["latency_eligible"])
        self.assertFalse(rows["control"]["speed_eligible"])
        self.assertFalse(rows["control"]["content_passed"])
        self.assertTrue(rows["control"]["comparison_content_passed"])
        self.assertEqual(rows["control"]["rescored_cause"], "recorded_evaluator_builtin_limitation")
        self.assertFalse(rows["control"]["raw_json_object"])
        self.assertTrue(rows["control"]["outer_code_fence"])
        self.assertIsNone(rows["control"]["empty_answer"])
        self.assertIn("concurrent CPU build", rows["control"]["exclusion_reasons"])

    def test_new_hard_json_rows_get_strict_format_audit_from_final_text(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw_object = request_row("raw-object")
            raw_object["response"]["response_text"] = '{"balances":{"a":1}}'
            fenced = request_row("fenced", case_id="hard_dependency_constraints")
            fenced["response"]["response_text"] = '```json\n{"deployment_order":[]}\n```'
            empty = request_row("empty", passed=False)
            empty["response"]["response_text"] = " \n\t"
            coding = request_row("coding", case_id="hard_weighted_intervals")
            coding["response"]["response_text"] = "{}"
            write_jsonl(root / "fresh-hard.jsonl", [raw_object, fenced, empty, coding])

            report = final_report.build_report(root)
            rendered = final_report.render_html(report)
            csv_rows = {
                row["call_id"]: row
                for row in csv.DictReader(final_report.render_csv(report).splitlines())
            }

        rows = {row["row_id"]: row for row in report["request_rows"]}
        self.assertEqual(
            (rows["raw-object"]["raw_json_object"], rows["raw-object"]["outer_code_fence"], rows["raw-object"]["empty_answer"]),
            (True, False, False),
        )
        self.assertEqual(
            (rows["fenced"]["raw_json_object"], rows["fenced"]["outer_code_fence"], rows["fenced"]["empty_answer"]),
            (False, True, False),
        )
        self.assertEqual(
            (rows["empty"]["raw_json_object"], rows["empty"]["outer_code_fence"], rows["empty"]["empty_answer"]),
            (False, False, True),
        )
        self.assertIsNone(rows["coding"]["raw_json_object"])
        self.assertIsNone(rows["coding"]["outer_code_fence"])
        self.assertIsNone(rows["coding"]["empty_answer"])
        self.assertEqual(csv_rows["fenced"]["outer_code_fence"], "True")
        self.assertEqual(csv_rows["empty"]["empty_answer"], "True")
        self.assertIn("Outer code fence", rendered)
        self.assertIn("Empty answer", rendered)

    def test_copy_edit_dimensions_and_evaluator_check_counts_are_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fidelity_failure = request_row(
                "copy-fidelity", case_id="quality_copy_edit_module",
                case_version="2026-10-19.copy-edit.1", prompt_hash="copy-prompt",
                passed=False,
            )
            fidelity_failure["case_source_sha256"] = "copy-source"
            fidelity_failure["case_metadata"] = {
                "hidden_test_count": 8, "independent_task_count": 1,
            }
            fidelity_failure["evaluation"] = {
                "passed": False,
                "detail": "behavior checks passed but source edit fidelity failed",
                "semantic_passed": True,
                "fidelity_passed": False,
                "hidden_tests_passed": 8,
            }
            standard_fallback = request_row(
                "copy-standard-count", case_id="quality_copy_edit_module",
                case_version="2026-10-19.copy-edit.1", prompt_hash="copy-prompt",
                repeat=1, passed=False,
            )
            standard_fallback["case_source_sha256"] = "copy-source"
            standard_fallback["case_metadata"] = {
                "hidden_test_count": 8, "independent_task_count": 1,
            }
            standard_fallback["evaluation"] = {
                "passed": False, "detail": "behavior test failed", "tests_passed": 5,
            }
            write_jsonl(root / "copy-edit-quality.jsonl", [fidelity_failure, standard_fallback])

            report = final_report.build_report(root)
            rendered = final_report.render_html(report)
            csv_rows = {
                row["call_id"]: row
                for row in csv.DictReader(final_report.render_csv(report).splitlines())
            }

        rows = {row["row_id"]: row for row in report["request_rows"]}
        copy_row = rows["copy-fidelity"]
        self.assertEqual(copy_row["suite"], "quality")
        self.assertEqual(copy_row["case_version"], "2026-10-19.copy-edit.1")
        self.assertEqual(copy_row["case_source_sha256"], "copy-source")
        self.assertFalse(copy_row["content_passed"])
        self.assertTrue(copy_row["semantic_behavior_passed"])
        self.assertFalse(copy_row["source_edit_fidelity_passed"])
        self.assertEqual(copy_row["hidden_tests_passed"], 8)
        self.assertEqual(copy_row["hidden_test_count"], 8)
        self.assertEqual(rows["copy-standard-count"]["hidden_tests_passed"], 5)
        self.assertIsNone(rows["copy-standard-count"]["semantic_behavior_passed"])
        self.assertIsNone(rows["copy-standard-count"]["source_edit_fidelity_passed"])
        self.assertEqual(len(report["profile_summaries"]), 1)
        self.assertEqual(report["profile_summaries"][0]["attempt_count"], 2)
        self.assertEqual(report["profile_summaries"][0]["independent_task_count"], 1)
        self.assertEqual(csv_rows["copy-fidelity"]["semantic_behavior_passed"], "True")
        self.assertEqual(csv_rows["copy-fidelity"]["source_edit_fidelity_passed"], "False")
        self.assertIn("Semantic behavior pass", rendered)
        self.assertIn("Source edit fidelity pass", rendered)

    def test_generic_timing_annotation_excludes_only_its_exact_result_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            affected = root / "flash-followups-01-quick.jsonl"
            clean = root / "flash-followups-clean-quick.jsonl"
            annotation = root / "flash-followups-quick-timing-annotation.json"
            write_jsonl(affected, [request_row("overlapped")])
            write_jsonl(clean, [request_row("clean")])
            write_json(annotation, {
                "result_expected": str(affected),
                "timing_valid_for_speed_comparison": False,
                "reason": "full-file hash overlapped this request",
                "content_outcomes_valid": True,
            })
            report = final_report.build_report(root)

        rows = {row["row_id"]: row for row in report["request_rows"]}
        self.assertFalse(rows["overlapped"]["speed_eligible"])
        self.assertFalse(rows["overlapped"]["latency_eligible"])
        self.assertTrue(rows["overlapped"]["content_passed"])
        self.assertTrue(any("full-file hash" in reason for reason in rows["overlapped"]["exclusion_reasons"]))
        self.assertTrue(rows["clean"]["speed_eligible"])
        self.assertTrue(rows["clean"]["latency_eligible"])
        self.assertIn(
            "flash-followups-quick-timing-annotation.json",
            {Path(item["path"]).name for item in report["source_provenance"]},
        )

    def test_timing_annotation_call_ids_limit_exclusion_within_one_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shared = root / "low-hard.jsonl"
            empty_target = root / "empty-hard.jsonl"
            malformed_target = root / "malformed-hard.jsonl"
            write_jsonl(shared, [request_row("queued-ledger"), request_row("clean-weighted")])
            write_jsonl(empty_target, [request_row("empty-list-stays-clean")])
            write_jsonl(malformed_target, [request_row("malformed-stays-clean")])
            write_json(root / "low-hard-timing-annotation.json", {
                "result_expected": str(shared),
                "call_ids": ["queued-ledger"],
                "timing_valid_for_speed_comparison": False,
                "reason": "three-second queue overlap",
            })
            write_json(root / "empty-timing-annotation.json", {
                "result_expected": str(empty_target),
                "call_ids": [],
                "timing_valid_for_speed_comparison": False,
                "reason": "must not become a file-wide exclusion",
            })
            write_json(root / "malformed-timing-annotation.json", {
                "result_expected": str(malformed_target),
                "call_ids": "malformed",
                "timing_valid_for_speed_comparison": False,
                "reason": "must not become a file-wide exclusion",
            })
            report = final_report.build_report(root)

        rows = {row["row_id"]: row for row in report["request_rows"]}
        self.assertFalse(rows["queued-ledger"]["speed_eligible"])
        self.assertFalse(rows["queued-ledger"]["latency_eligible"])
        for call_id in ("clean-weighted", "empty-list-stays-clean", "malformed-stays-clean"):
            self.assertTrue(rows[call_id]["speed_eligible"], call_id)
            self.assertTrue(rows[call_id]["latency_eligible"], call_id)

    def test_actual_blocked_receipts_and_q3_artifact_are_provenance_not_attempts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_json(root / "flash-next-mtp-artifact.json", {
                "state": "blocked_before_fetch",
                "source": {"revision": "mtp-rev", "raw_tensor_bytes": 5214301696},
                "payload": {"bytes_present": 0},
                "automatic_approval_rejection": {"reason": "approval rejected before launch"},
            })
            write_json(root / "flash-next-mtp-candidate.json", {
                "state": "stale-candidate", "download": {"reason": "stale reason"},
            })
            write_json(root / "flash-next-accuracy-reference.json", {
                "state": "blocked_before_launch", "model": "Flash reference",
                "requested_engine_environment": {"STRATA_ROUTE_TAIL_SKIP": "0"},
                "rejection": "approval rejected", "inference_attempts": 0,
            })
            write_json(root / "qwen36-q3-artifact.json", {
                "state": "ready",
                "model": {"quant": "UD-Q3_K_XL", "revision": "q3-rev", "repo": "publisher/q3"},
                "shards": [{"name": "q3.gguf", "size": 17, "sha256": "abc", "verified": True}],
                "llama_cpp": {"config_activated": True},
                "strata_compatibility": {"compatible": False},
            })
            report = final_report.build_report(root)

        self.assertEqual(report["inventory"]["request_rows"], 0)
        mtp_note = next(item for item in report["not_run"] if "optional MTP" in item.get("model", ""))
        self.assertEqual(mtp_note["status"], "blocked_before_fetch")
        self.assertTrue(mtp_note["source_file"].endswith("flash-next-mtp-artifact.json"))
        self.assertNotIn("stale reason", json.dumps(report["not_run"]))
        accuracy = next(item for item in report["not_run"] if item.get("model") == "Flash reference")
        self.assertEqual(accuracy["status"], "blocked_before_launch")
        q3_note = next(item for item in report["not_run"] if item.get("weight_quant") == "UD-Q3_K_XL")
        self.assertEqual(q3_note["status"], "ready")
        self.assertNotIn("No 3-bit weight comparison was run", json.dumps(report))
        mtp_provenance = next(item for item in report["artifact_provenance"] if str(item.get("source_file", "")).endswith("flash-next-mtp-artifact.json"))
        q3_provenance = next(item for item in report["artifact_provenance"] if str(item.get("source_file", "")).endswith("qwen36-q3-artifact.json"))
        self.assertEqual(mtp_provenance["payload_bytes_present"], 0)
        self.assertEqual(q3_provenance["recorded_weight_sha256"], {"q3.gguf": "abc"})
        self.assertTrue(q3_provenance["verified"])

    def test_qwen122_verified_weights_and_layout_remain_provenance_not_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_json(root / "qwen122-artifact.json", {
                "model": {
                    "repo": "publisher/qwen122", "revision": "weights-rev",
                    "quant": "UD-Q4_K_XL", "expected_total_size": 30,
                },
                "verdict": {"artifact_complete": True},
                "shards": [
                    {"name": "part-1", "size": 0, "sha256": "sha-1", "present": True, "verified": True},
                    {"name": "part-2", "size": 10, "sha256": "sha-2", "present": True, "verified": True},
                    {"name": "part-3", "size": 20, "sha256": "sha-3", "present": True, "verified": True},
                ],
            })
            write_json(root / "qwen122-layout.json", {
                "artifact_manifest_sha256": "artifact-receipt-hash",
                "split_count": 3, "tensor_count": 879,
                "all_offsets_aligned": True,
                "all_tensor_directory_entries_validated": True,
                "all_tensor_names_unique": True,
                "all_tensor_payload_bounds_validated": True,
                "no_payload_overlaps": True,
                "shards": [
                    {"tensor_count": 0}, {"tensor_count": 561}, {"tensor_count": 318},
                ],
                "tensor_type_counts": {"F32": 361, "Q4_K": 94, "Q5_K": 49, "Q6_K": 1, "Q8_0": 374},
            })
            write_json(root / "qwen122-strata-candidate.json", {
                "status": "experimental_cuda_built_unloaded",
                "artifact_evidence": {"quant": "UD-Q4_K_XL"},
                "source": {"base_revision": "runtime-rev", "patch_sha256": "patch-hash"},
                "validation": {"cuda_build": {"binary": {"sha256": "binary-hash"}}},
            })
            report = final_report.build_report(root)

        self.assertEqual(report["inventory"]["request_rows"], 0)
        self.assertEqual(report["supplemental_rows"], [])
        weights = next(
            item for item in report["artifact_provenance"]
            if str(item.get("source_file", "")).endswith("qwen122-artifact.json")
        )
        runtime = next(
            item for item in report["artifact_provenance"]
            if str(item.get("source_file", "")).endswith("qwen122-strata-candidate.json")
        )
        self.assertEqual(weights["source_repo"], "publisher/qwen122")
        self.assertEqual(weights["source_revision"], "weights-rev")
        self.assertEqual(weights["recorded_weight_bytes"], 30)
        self.assertEqual(weights["recorded_weight_sha256"], {
            "part-1": "sha-1", "part-2": "sha-2", "part-3": "sha-3",
        })
        self.assertTrue(weights["verified"])
        self.assertTrue(weights["layout_verified"])
        self.assertEqual(weights["layout_proof"]["tensor_count"], 879)
        self.assertEqual(weights["layout_proof"]["shard_tensor_counts"], [0, 561, 318])
        self.assertIn("provenance only", weights["artifact_role"])
        self.assertEqual(runtime["binary_sha256"], "binary-hash")
        self.assertNotEqual(weights["source_file"], runtime["source_file"])
        sources = {Path(item["path"]).name for item in report["source_provenance"]}
        self.assertIn("qwen122-artifact.json", sources)
        self.assertIn("qwen122-layout.json", sources)
        note = next(
            item for item in report["not_run"]
            if str(item.get("source_file", "")).endswith("qwen122-artifact.json")
        )
        self.assertIn("inference have not run", note["reason"])

    def test_native_gpt20b_and_blocked_overlap_receipts_stay_provenance_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_json(root / "gpt20b-llama-artifact.json", {
                "state": "inactive_source_and_header_validated",
                "model": {
                    "size": 13793422144, "fresh_full_sha256_verified": True,
                    "harmony": {"template_sha256": "harmony-hash"},
                },
                "shards": [{
                    "name": "gpt20b.gguf", "size": 13793422144,
                    "sha256": "weight-hash", "verified": True,
                }],
                "llama_cpp": {
                    "upstream_revision": "llama-rev", "server_sha256": "server-hash",
                },
                "reasoning_budget": {
                    "direct_server_template": {"sha256": "llama-template-hash"},
                },
                "verdict": {
                    "artifact_complete": True, "runtime_load_validated": False,
                    "reasoning_budget_runtime_validated": False,
                },
            })
            write_json(root / "flash-next-adaptive-overlap-blocked.json", {
                "status": "blocked_before_launch", "inference_requests": 0,
                "retry_attempts": 0,
                "requested_environment": {"STRATA_ADAPT_OVERLAP": "1"},
                "reason": "automatic approval rejected before launch",
            })
            report = final_report.build_report(root)

            trial = {
                "runtime_backend": "llama",
                "source_config": str(root / "gpt20b-llama-32k.json"),
                "model_name": "gpt-oss-20b-mxfp4",
            }
            normalized = final_report.normalize_request(
                request_row("native-gpt", model="gpt-oss-20b-mxfp4"),
                root / "native-gpt-hard.jsonl", 1, trial, {}, {}, {},
            )

        self.assertEqual(report["inventory"]["request_rows"], 0)
        self.assertEqual(report["supplemental_rows"], [])
        artifact = next(
            item for item in report["artifact_provenance"]
            if str(item.get("source_file", "")).endswith("gpt20b-llama-artifact.json")
        )
        self.assertEqual(artifact["weight_quant"], "MXFP4")
        self.assertEqual(artifact["recorded_weight_sha256"], {"gpt20b.gguf": "weight-hash"})
        self.assertEqual(artifact["harmony_template_sha256"], "harmony-hash")
        self.assertEqual(artifact["llama_chat_template_sha256"], "llama-template-hash")
        self.assertTrue(artifact["verified"])
        self.assertFalse(artifact["runtime_load_validated"])
        self.assertFalse(artifact["reasoning_budget_runtime_validated"])
        gpt_note = next(
            item for item in report["not_run"]
            if str(item.get("source_file", "")).endswith("gpt20b-llama-artifact.json")
        )
        self.assertIn("no native llama request rows", gpt_note["reason"])
        overlap = next(
            item for item in report["not_run"]
            if str(item.get("source_file", "")).endswith("flash-next-adaptive-overlap-blocked.json")
        )
        self.assertEqual(overlap["status"], "blocked_before_launch")
        self.assertEqual(overlap["retry_attempts"], 0)
        sources = {Path(item["path"]).name for item in report["source_provenance"]}
        self.assertIn("gpt20b-llama-artifact.json", sources)
        self.assertIn("flash-next-adaptive-overlap-blocked.json", sources)
        self.assertEqual(normalized["weight_quant"], "MXFP4")
        self.assertEqual(normalized["kv_cache"], "f16")
        self.assertEqual(normalized["runtime"], "llama.cpp server (Ollama 0.40.6 bundle)")

    def test_eagle3_draft_options_separate_profiles_and_not_run_requires_actual_draft_row(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact_path = root / "gpt20b-eagle3-artifact.json"
            write_json(artifact_path, {
                "state": "inactive_download_pending",
                "gguf_artifact": {
                    "name": "eagle3-gpt-oss-20b-Q8_0.gguf", "quantization": "Q8_0",
                    "repository": "ggml-org/gpt-oss-20b-GGUF", "revision": "draft-rev",
                    "size": 921488000, "sha256": "draft-sha",
                },
                "target": {"sha256": "target-sha"},
                "profiles": {"cpu": {"runtime_validated": False}, "gpu": {"runtime_validated": False}},
                "verdict": {
                    "artifact_complete": False, "full_sha256_verified": False,
                    "runtime_load_validated": False, "inference_validated": False,
                },
            })
            common_args = [
                "--gpu-layers", "auto", "--chat-template-file", "/templates/harmony.jinja",
                "--spec-type", "draft-eagle3", "--spec-draft-model", "/models/eagle3-Q8_0.gguf",
                "--spec-draft-n-max", "3",
                "--spec-draft-n-min", "0", "--spec-draft-p-min", "0.0",
                "--spec-draft-type-k", "f16", "--spec-draft-type-v", "f16",
            ]
            cpu_config = root / "gpt20b-eagle3-cpu-32k.json"
            gpu_config = root / "gpt20b-eagle3-gpu-32k.json"
            write_json(cpu_config, {
                "args": common_args + [
                    "--spec-draft-device", "none", "--spec-draft-ngl", "0",
                    "--spec-draft-n-ubatch", "64",
                ],
            })
            write_json(gpu_config, {"args": common_args + ["--spec-draft-ngl", "auto"]})
            raw = request_row("eagle", model="gpt-oss-20b-mxfp4")
            base_trial = {"runtime_backend": "llama", "model_name": "gpt-oss-20b-mxfp4"}
            cpu = final_report.normalize_request(
                raw, root / "cpu-hard.jsonl", 1,
                base_trial | {"source_config": str(cpu_config)}, {}, {}, {},
            )
            gpu = final_report.normalize_request(
                raw | {"call_id": "eagle-gpu"}, root / "gpu-hard.jsonl", 1,
                base_trial | {"source_config": str(gpu_config)}, {}, {}, {},
            )
            report = final_report.build_report(root)
            plain_native = {
                "model": "gpt-oss-20b-mxfp4", "runtime": "llama.cpp server (Ollama 0.40.6 bundle)",
                "runtime_options": {},
            }
            notes_without_draft = final_report._not_run(root, [plain_native])[0]
            notes_with_draft = final_report._not_run(root, [plain_native, cpu])[0]
            derived_path = root / "gpt20b-eagle3-derived-artifact.json"
            derived = json.loads(artifact_path.read_text())
            derived["state"] = "inactive_verified"
            derived["verdict"].update(artifact_complete=True, full_sha256_verified=True)
            derived["compatibility"] = {"activation_allowed": True}
            derived["gguf_artifact"]["sha256"] = "derived-draft-sha"
            write_json(derived_path, derived)
            report_with_derived = final_report.build_report(root)
            notes_with_derived = final_report._not_run(root, [plain_native])[0]

        self.assertEqual(cpu["weight_quant"], "MXFP4")
        self.assertEqual(cpu["draft_weight_quant"], "Q8_0")
        self.assertEqual(cpu["kv_cache"], "f16")
        self.assertEqual(cpu["runtime_options"]["spec_draft_model"], "/models/eagle3-Q8_0.gguf")
        self.assertEqual(cpu["runtime_options"]["spec_draft_device"], "none")
        self.assertEqual(cpu["runtime_options"]["spec_draft_ngl"], "0")
        self.assertEqual(cpu["runtime_options"]["spec_draft_n_max"], "3")
        self.assertEqual(cpu["runtime_options"]["spec_draft_n_min"], "0")
        self.assertEqual(cpu["runtime_options"]["spec_draft_p_min"], "0.0")
        self.assertEqual(cpu["runtime_options"]["spec_draft_type_k"], "f16")
        self.assertEqual(cpu["runtime_options"]["spec_draft_type_v"], "f16")
        self.assertEqual(cpu["runtime_options"]["spec_draft_n_ubatch"], "64")
        self.assertEqual(cpu["runtime_options"]["chat_template_file"], "/templates/harmony.jinja")
        self.assertIsNone(gpu["runtime_options"]["spec_draft_device"])
        self.assertEqual(gpu["runtime_options"]["spec_draft_ngl"], "auto")
        self.assertNotEqual(cpu["profile_id"], gpu["profile_id"])
        self.assertTrue(any("no actual draft-eagle3 request rows" in item.get("reason", "") for item in notes_without_draft))
        self.assertFalse(any(str(item.get("source_file", "")).endswith("gpt20b-eagle3-artifact.json") for item in notes_with_draft))
        provenance = next(
            item for item in report["artifact_provenance"]
            if str(item.get("source_file", "")).endswith("gpt20b-eagle3-artifact.json")
        )
        self.assertEqual(provenance["weight_quant"], "Q8_0")
        self.assertEqual(provenance["target_weight_quant"], "MXFP4")
        self.assertIsNone(provenance["recorded_weight_sha256"])
        self.assertFalse(provenance["verified"])
        self.assertIn(
            "gpt20b-eagle3-artifact.json",
            {Path(item["path"]).name for item in report["source_provenance"]},
        )
        self.assertTrue(any(
            str(item.get("source_file", "")).endswith("gpt20b-eagle3-derived-artifact.json")
            and item["status"] == "inactive_verified" for item in notes_with_derived
        ))
        eagle_provenance = [
            item for item in report_with_derived["artifact_provenance"]
            if "EAGLE3 draft" in item.get("model", "")
        ]
        self.assertEqual(len(eagle_provenance), 2)
        self.assertEqual(
            [item["recorded_weight_sha256"] for item in eagle_provenance],
            [None, "derived-draft-sha"],
        )

    def test_profile_identity_uses_controls_not_private_path_and_separates_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_a = root / "private-a.json"
            config_b = root / "private-b.json"
            write_json(config_a, {"reasoning_budget_tokens": 512, "args": ["--mtp", "0"]})
            write_json(config_b, {"reasoning_budget_tokens": 512, "args": ["--mtp", "0"]})
            result_a = root / "a-hard.jsonl"
            result_b = root / "b-hard.jsonl"
            result_c = root / "c-hard.jsonl"
            result_d = root / "d-hard.jsonl"
            write_jsonl(result_a, [request_row("a")])
            write_jsonl(result_b, [request_row("b")])
            write_jsonl(result_c, [request_row("c")])
            write_jsonl(result_d, [request_row("d")])
            lifecycle(root / "a-runtime-events.jsonl", [result_a], private_config=config_a)
            lifecycle(root / "b-runtime-events.jsonl", [result_b], private_config=config_b)
            lifecycle(
                root / "c-runtime-events.jsonl", [result_c], private_config=config_b,
                environment={"STRATA_SM75_PERF": "1"},
            )
            lifecycle(
                root / "d-runtime-events.jsonl", [result_d], private_config=config_b,
                source_config=root / "different-engine-source.json",
            )
            report = final_report.build_report(root)

        by_id = {row["row_id"]: row for row in report["request_rows"]}
        self.assertEqual(by_id["a"]["profile_id"], by_id["b"]["profile_id"])
        self.assertNotEqual(by_id["a"]["profile_id"], by_id["c"]["profile_id"])
        self.assertNotEqual(by_id["a"]["profile_id"], by_id["d"]["profile_id"])
        self.assertEqual(by_id["a"]["reasoning_budget_tokens"], 512)
        self.assertIsNone(by_id["a"]["reasoning_budget_tokens_requested"])
        self.assertEqual(by_id["a"]["reasoning_budget_tokens_config_default"], 512)
        merged = next(item for item in report["profile_summaries"] if item["profile_id"] == by_id["a"]["profile_id"])
        self.assertEqual(merged["attempt_count"], 2)

    def test_specialist_artifact_receipts_keep_verification_scopes_and_flash_quant_caveat(self):
        flash_names = (
            "Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf",
            "Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00002-of-00002.gguf",
        )
        flash_sizes = (29_608_446_496, 28_800_138_432)
        flash_hashes = (
            "bfa3644db4adfff072fc946c20994f2909d2cf89879d83f80a4c811d4016c211",
            "316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113",
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_json(root / "lfm25-qad-q4_0-artifact.json", {
                "state": "artifact_verified_runtime_unvalidated",
                "verdict": {
                    "artifact_complete": True, "full_sha256_verified": True,
                    "runtime_load_validated": False,
                },
                "model": {
                    "alias": "lfm-fixture", "repo": "publisher/lfm", "revision": "lfm-rev",
                    "quantization": "QAD-Q4_0",
                },
                "shards": [{
                    "name": "lfm.gguf", "size": 13_578_190_880, "sha256": "lfm-sha",
                    "verified": True, "full_sha256_verified": True,
                }],
                "runtime": {
                    "llama_cpp_revision": "llama-rev", "all_gpu_fit_claimed": False,
                },
            })
            write_json(root / "bonsai2-runtime-artifact.json", {
                "state": "ready_for_gpu_trial_unvalidated",
                "verdict": {
                    "model_artifact_complete": True, "runtime_artifact_complete": True,
                    "gpu_load_proven": False, "static_tesla_t4_compatibility": True,
                },
                "source": {
                    "model_repo": "publisher/bonsai", "model_revision": "bonsai-rev",
                    "llama_revision": "prism-rev",
                },
                "model": {"name": "bonsai-pq.gguf"},
                "available_model_packings": [
                    {"name": "bonsai-pq.gguf", "quantization": "PQ2_0"},
                    {"name": "bonsai-ptq.gguf", "quantization": "PTQ1_0"},
                ],
                "shards": [
                    {"name": "bonsai-pq.gguf", "size": 7_206_168_928, "sha256": "pq-sha", "verified": True},
                    {"name": "bonsai-ptq.gguf", "size": 5_946_648_928, "sha256": "ptq-sha", "verified": True},
                ],
                "runtime": {
                    "source_revision": "prism-rev",
                    "server": {"sha256": "server-sha"},
                    "cuda_backend": {"sha256": "cuda-sha"},
                },
            })
            write_json(root / "flash-next-coder-header-preview.json", {
                "payload_fully_verified": False,
                "chunk_zero_manifest_hash_matches": True,
                "model_name": "Flash Coder fixture",
                "blocks": 48, "experts": 256, "experts_used": 10,
                "transformer_tensor_elements": 65_314_218_880,
                "shared_ple_tensor_elements": 51_200_245_760,
            })

            preview_report = final_report.build_report(root)
            preview_flash = next(
                row for row in preview_report["artifact_provenance"]
                if row.get("artifact_family") == "specialist_flash_next_coder"
            )
            rendered = final_report.render_html(preview_report)

            self.assertFalse(preview_flash["verified"])
            self.assertIsNone(preview_flash["recorded_weight_bytes"])
            self.assertEqual(preview_flash["expected_weight_bytes"], sum(flash_sizes))
            self.assertEqual(preview_flash["header_preview"]["total_tensor_elements"], 116_514_464_640)
            self.assertFalse(preview_flash["header_preview"]["individual_iq1_m_tensor_observed_in_parsed_mixed_type_table"])
            self.assertIn("not a claim that the retained weights are 1-bit", preview_flash["quantization_caveat"])
            self.assertIn("payload full SHA-256 is not verified", preview_flash["verification_scope"])
            self.assertIsNone(preview_flash["all_gpu_fit_claimed"])
            self.assertIn("not a claim that the retained weights are 1-bit", rendered)
            self.assertEqual(
                len([row for row in preview_report["artifact_provenance"] if row.get("artifact_family") == "specialist_bonsai2"]),
                2,
            )
            first_wave = next(
                row for row in preview_report["artifact_provenance"]
                if row.get("artifact_family") == "specialist_first_wave"
            )
            self.assertTrue(first_wave["verified"])
            self.assertEqual(first_wave["runtime_source_revision"], "llama-rev")

            write_json(root / "flash-next-coder-artifact.json", {
                "state": "prepared_runtime_unvalidated",
                "verdict": {
                    "artifact_complete": True, "full_sha256_verified": True,
                    "shared_ple_identity_bound": True, "pack_complete": True,
                    "runtime_load_validated": False, "gpu_placement_validated": False,
                },
                "model": {
                    "model_name": "Flash Coder fixture", "repo": "publisher/flash",
                    "revision": "flash-rev", "quantization": "IQ1_M",
                },
                "shards": [
                    {
                        "name": name, "size": size, "sha256": digest,
                        "verified": True, "full_sha256_verified": True,
                    }
                    for name, size, digest in zip(flash_names, flash_sizes, flash_hashes)
                ],
                "runtime": {
                    "strata_revision": "strata-rev", "engine": {"sha256": "engine-sha"},
                },
            })
            final_report_value = final_report.build_report(root)

        final_flash = next(
            row for row in final_report_value["artifact_provenance"]
            if row.get("artifact_family") == "specialist_flash_next_coder"
        )
        self.assertTrue(final_flash["verified"])
        self.assertEqual(final_flash["recorded_weight_bytes"], sum(flash_sizes))
        self.assertEqual(final_flash["runtime_source_revision"], "strata-rev")
        self.assertFalse(final_flash["runtime_load_validated"])
        self.assertIsNone(final_flash["all_gpu_fit_claimed"])
        self.assertIn("full two-shard SHA-256", final_flash["verification_scope"])
        source_names = {Path(row["path"]).name for row in final_report_value["source_provenance"]}
        self.assertTrue({
            "lfm25-qad-q4_0-artifact.json", "bonsai2-runtime-artifact.json",
            "flash-next-coder-artifact.json", "flash-next-coder-header-preview.json",
        }.issubset(source_names))

    def test_q36_native_memory_and_kernel_controls_are_typed_and_separate_profiles(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def q36_config(name, *, context, expert_vram, no_mmq=True, ignored=False):
                log = root / f"{name}.log"
                args = [
                    "--native", "/models/qwen122.gguf", "--max-context", str(context),
                    "--prefill", "256", "--expert-vram", str(expert_vram),
                    "--batch", "1",
                ]
                if no_mmq:
                    args.append("--no-mmq")
                if ignored:
                    args += ["--spec", "0", "--suffix-draft", "0", "--vram-reserve-mib", "1024"]
                    log.write_text(
                        "strata-q36: ignoring --spec 0 (not an option of this engine)\n"
                        "strata-q36: ignoring --suffix-draft 0 (not an option of this engine)\n"
                        "strata-q36: ignoring --vram-reserve-mib 1024 (not an option of this engine)\n",
                        encoding="utf-8",
                    )
                config = root / f"{name}.json"
                write_json(config, {
                    "arch": "qwen35moe", "args": args, "log": str(log),
                })
                return config

            zero_config = q36_config("cpu0", context=8192, expert_vram=0)
            one_config = q36_config("cache1g", context=8192, expert_vram=1)
            full_config = q36_config("cpu0-32k", context=32768, expert_vram=0)
            mmq_config = q36_config("mmq", context=8192, expert_vram=0, no_mmq=False)
            ignored_config = q36_config("ignored-flash-options", context=32768, expert_vram=0, ignored=True)

            def normalized(call_id, config, context):
                raw = request_row(call_id, model="qwen3.5-122b-a10b-ud-q4_k_xl")
                raw["request"]["context_max_tokens"] = context
                return final_report.normalize_request(
                    raw, root / f"{call_id}.jsonl", 1,
                    {
                        "runtime_backend": "strata", "source_config": str(config),
                        "model_name": "qwen3.5-122b-a10b-ud-q4_k_xl",
                    },
                    {}, {}, {},
                )

            zero = normalized("zero", zero_config, 8192)
            one = normalized("one", one_config, 8192)
            full = normalized("full", full_config, 32768)
            mmq = normalized("mmq", mmq_config, 8192)
            ignored = normalized("ignored", ignored_config, 32768)

        self.assertEqual(zero["runtime_options"]["expert_vram_gib"], 0)
        self.assertEqual(one["runtime_options"]["expert_vram_gib"], 1)
        self.assertEqual(zero["runtime_options"]["prefill"], 256)
        self.assertEqual(zero["runtime_options"]["batch"], 1)
        self.assertIs(zero["runtime_options"]["native_prompt"], True)
        self.assertIs(zero["runtime_options"]["no_mmq"], True)
        self.assertIs(mmq["runtime_options"]["no_mmq"], False)
        self.assertNotIn("no_mmq", final_report._config_metadata({})["runtime_options"])
        self.assertNotEqual(zero["profile_id"], one["profile_id"])
        self.assertNotEqual(zero["profile_id"], full["profile_id"])
        self.assertNotIn("spec", ignored["runtime_options"])
        self.assertNotIn("suffix_draft", ignored["runtime_options"])
        self.assertNotIn("vram_reserve_mib", ignored["runtime_options"])
        self.assertEqual(ignored["ignored_runtime_options"], {
            "spec": "0", "suffix_draft": "0", "vram_reserve_mib": "1024",
        })

    def test_requested_native_runtime_controls_are_additive_and_explicit_only(self):
        expected = {
            "cache_type_k": "q8_0",
            "cache_type_v": "q8_0",
            "flash_attn": "on",
            "fit": "off",
            "fit_ctx": 73728,
            "batch_size": 256,
            "ubatch_size": 64,
            "n_cpu_moe": 2,
            "gpu_layers": "all",
            "threads": 8,
            "threads_batch": 28,
            "poll": 0,
            "poll_batch": 0,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "kat-q8kv-cpu2.json"
            write_json(config, {"args": [
                "--gpu-layers", "all", "--parallel", "1", "--fit-ctx", "73728",
                "--fit", "off", "--cache-type-k", "q8_0", "--cache-type-v", "q8_0",
                "--flash-attn", "on", "--batch-size", "256", "--ubatch-size", "64",
                "--n-cpu-moe", "2", "--threads", "8", "--threads-batch", "28",
                "--poll", "0", "--poll-batch", "0",
            ]})
            trial = {
                "runtime_backend": "llama", "source_config": str(config),
                "source_config_sha256": "config-sha", "model_name": "kat-coder",
            }
            metadata = final_report._config_metadata(trial)
            row = final_report.normalize_request(
                request_row("kat-native-controls", model="kat-coder"),
                root / "requests.jsonl", 1, trial, {}, {}, {},
            )
            missing = final_report.normalize_request(
                request_row("missing-config", model="kat-coder"),
                root / "missing.jsonl", 1,
                {"runtime_backend": "llama", "source_config": str(root / "absent.json")},
                {}, {}, {},
            )
            csv_row = next(csv.DictReader(final_report.render_csv({
                "request_rows": [row], "supplemental_rows": [],
            }).splitlines()))
            rendered = final_report.render_html({
                "generated_at": "2026-10-20T00:00:00Z", "request_rows": [row],
                "supplemental_rows": [], "profile_summaries": [],
                "matched_comparisons": [], "data_gaps": [], "not_run": [],
            })

        self.assertEqual(metadata["requested_native_runtime_controls"], expected)
        self.assertEqual(row["requested_native_runtime_controls"], expected)
        self.assertEqual(row["runtime_options"], {"gpu_layers": "all"})
        self.assertIsNone(row["kv_cache"])
        self.assertEqual(row["source_config_sha256"], "config-sha")
        self.assertFalse(any("actual" in key or "resident" in key for key in expected))
        self.assertEqual(json.loads(csv_row["requested_native_runtime_controls"]), expected)
        self.assertEqual(csv_row["source_config_sha256"], "config-sha")
        self.assertIn("Requested native runtime controls", rendered)
        self.assertIn("does not establish entirely GPU-resident execution", rendered)
        self.assertEqual(final_report._config_metadata({})["requested_native_runtime_controls"], {})
        self.assertNotIn("requested_native_runtime_controls", missing)

        without_addition = dict(row)
        without_addition.pop("requested_native_runtime_controls")
        self.assertEqual(row["profile_id"], final_report._stable_id(final_report._profile_parts(without_addition)))
        self.assertEqual(row["match_id"], final_report._stable_id(final_report._match_parts(without_addition)))

    def test_profile_identity_isolates_request_tuning_and_requested_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [
                request_row("base"),
                request_row("same-defaults"),
                request_row("pcie", strata_pcie_frac=0.25),
                request_row("aux", strata_aux_cpus=1),
                request_row("budget", reasoning_budget_tokens=768),
                request_row("template-on", chat_template_thinking="on"),
                request_row("template-off", chat_template_thinking="off"),
            ]
            write_jsonl(root / "tuned-hard.jsonl", rows)
            report = final_report.build_report(root)
            parsed_csv = list(csv.DictReader(final_report.render_csv(report).splitlines()))

        by_id = {row["row_id"]: row for row in report["request_rows"]}
        self.assertEqual(by_id["base"]["profile_id"], by_id["same-defaults"]["profile_id"])
        for row_id in ("pcie", "aux", "budget", "template-on", "template-off"):
            self.assertNotEqual(by_id["base"]["profile_id"], by_id[row_id]["profile_id"])
        self.assertNotEqual(by_id["template-on"]["profile_id"], by_id["template-off"]["profile_id"])
        self.assertEqual(by_id["pcie"]["strata_pcie_frac"], 0.25)
        self.assertEqual(by_id["aux"]["strata_aux_cpus"], 1)
        self.assertEqual(by_id["budget"]["reasoning_budget_tokens_requested"], 768)
        self.assertIsNone(by_id["budget"]["reasoning_budget_tokens_config_default"])
        self.assertIsNone(by_id["base"]["chat_template_thinking"])
        self.assertEqual(by_id["template-on"]["chat_template_thinking"], "on")
        self.assertEqual(by_id["template-off"]["chat_template_thinking"], "off")
        csv_budget = next(row for row in parsed_csv if row["row_id"] == "budget")
        self.assertEqual(csv_budget["reasoning_budget_tokens_requested"], "768")
        self.assertIn("strata_pcie_frac", csv_budget)
        self.assertIn("chat_template_thinking", csv_budget)

    def test_profile_aggregates_observed_totals_and_filters_clean_distributions(self):
        def remove_metrics(row):
            row["response"].pop("actual_prompt_tokens", None)
            row["response"].pop("actual_completion_tokens", None)
            row["response"]["timing"] = {}
            row["response"]["throughput"] = {}
            return row

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "aggregate-hard.jsonl"
            clean = request_row("clean")
            clean_two = request_row("clean-two")
            clean_two["response"]["actual_prompt_tokens"] = 500
            clean_two["response"]["actual_completion_tokens"] = 100
            clean_two["response"]["timing"]["wall_seconds"] = 5.5
            clean_two["response"]["timing"]["ttft_final_content_seconds"] = 1.75
            clean_two["response"]["throughput"]["completion_tokens_per_second_native"] = 16.0
            contaminated = request_row("contaminated")
            missing = remove_metrics(request_row("missing"))
            all_missing = remove_metrics(request_row("all-missing", model="missing-model"))
            write_jsonl(source, [clean, clean_two, contaminated, missing, all_missing])
            write_json(root / "aggregate-timing-annotation.json", {
                "result_expected": str(source),
                "call_ids": ["contaminated"],
                "timing_valid_for_speed_comparison": False,
                "reason": "fixture timing overlap",
            })
            report = final_report.build_report(root)
            rendered = final_report.render_html(report)

        profiles = {item["profile"]["model"]: item for item in report["profile_summaries"]}
        summary = profiles["model<unsafe>"]
        self.assertEqual(summary["attempt_count"], 4)
        self.assertEqual(summary["observed_api_input_tokens_sum"], 1300)
        self.assertEqual(summary["observed_api_input_tokens_sample_count"], 3)
        self.assertEqual(summary["observed_api_input_tokens_missing_count"], 1)
        self.assertEqual(summary["observed_completion_tokens_sum"], 260)
        self.assertEqual(summary["observed_completion_tokens_sample_count"], 3)
        self.assertEqual(summary["observed_completion_tokens_missing_count"], 1)
        self.assertEqual(summary["observed_request_wall_seconds_sum"], 14.5)
        self.assertEqual(summary["observed_request_wall_seconds_sample_count"], 3)
        self.assertEqual(summary["observed_request_wall_seconds_missing_count"], 1)
        self.assertEqual(summary["speed_eligible_attempts"], 3)
        self.assertEqual(summary["speed_ineligible_attempts"], 1)
        self.assertEqual(summary["clean_native_decode_tps_sample_count"], 2)
        self.assertEqual(summary["clean_native_decode_tps_missing_count"], 1)
        self.assertEqual(summary["clean_native_decode_tps_median"], 14.0)
        self.assertEqual(summary["clean_native_decode_tps_min"], 12.0)
        self.assertEqual(summary["clean_native_decode_tps_max"], 16.0)
        self.assertEqual(summary["latency_eligible_attempts"], 3)
        self.assertEqual(summary["latency_ineligible_attempts"], 1)
        self.assertEqual(summary["clean_request_wall_seconds_sample_count"], 2)
        self.assertEqual(summary["clean_request_wall_seconds_missing_count"], 1)
        self.assertEqual(summary["clean_request_wall_seconds_median"], 5.0)
        self.assertEqual(summary["clean_request_wall_seconds_min"], 4.5)
        self.assertEqual(summary["clean_request_wall_seconds_max"], 5.5)
        self.assertEqual(summary["clean_ttft_final_seconds_sample_count"], 2)
        self.assertEqual(summary["clean_ttft_final_seconds_missing_count"], 1)
        self.assertEqual(summary["clean_ttft_final_seconds_median"], 1.5)
        self.assertEqual(summary["clean_ttft_final_seconds_min"], 1.25)
        self.assertEqual(summary["clean_ttft_final_seconds_max"], 1.75)

        missing_summary = profiles["missing-model"]
        self.assertIsNone(missing_summary["observed_api_input_tokens_sum"])
        self.assertIsNone(missing_summary["observed_completion_tokens_sum"])
        self.assertIsNone(missing_summary["observed_request_wall_seconds_sum"])
        self.assertIsNone(missing_summary["clean_native_decode_tps_median"])
        self.assertEqual(missing_summary["clean_native_decode_tps_sample_count"], 0)
        self.assertEqual(missing_summary["clean_native_decode_tps_missing_count"], 1)
        self.assertIn("Observed API input tokens sum", rendered)
        self.assertIn("Decode TPS median", rendered)
        self.assertIn("Final TTFT median s", rendered)

    def test_html_escapes_values_csv_has_all_attempts_and_cache_is_supplemental(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_jsonl(root / "round-hard.jsonl", [request_row("unsafe")])
            write_json(root / "expert-cache-summary.json", {
                "method": {
                    "model": "cache<model>", "runtime": "Strata", "case": "ledger",
                    "thinking": "none", "prompt_tokens": 433, "fixed_generation_tokens": 256,
                },
                "rows": [{
                    "tag": "probe", "status": "passed", "context": 32768,
                    "use_timing_comparison": False, "contention_note": "overlap<script>",
                    "native_timing": {"generated_tokens_per_second": 11.0},
                }],
            })
            write_json(root / "flash-next-mtp-candidate.json", {
                "state": "blocked_before_fetch",
                "download": {"started": False, "reason": "approval required; no download started"},
            })
            report = final_report.build_report(root)
            rendered = final_report.render_html(report)
            parsed_csv = list(csv.DictReader(final_report.render_csv(report).splitlines()))

        self.assertNotIn("model<unsafe>", rendered)
        self.assertIn("model&lt;unsafe&gt;", rendered)
        self.assertNotIn("overlap<script>", rendered)
        self.assertIn('id="compact"', rendered)
        self.assertIn("Profile controls:", rendered)
        self.assertEqual(len(parsed_csv), 2)
        cache = report["supplemental_rows"][0]
        self.assertEqual(cache["row_type"], "expert_cache")
        self.assertFalse(cache["quality_eligible"])
        self.assertFalse(cache["speed_eligible"])
        self.assertFalse(cache["cache_experiment_timing_valid"])
        mtp = next(item for item in report["not_run"] if "optional MTP" in item.get("model", ""))
        self.assertEqual(mtp["status"], "blocked_before_fetch")
        self.assertIn("no download", mtp["reason"])

    def test_scatter_distinguishes_models_and_outcomes_without_pooling(self):
        rows = [
            {
                "model": "model-a", "runtime": "runtime-a", "thinking": "none",
                "case_id": "case", "output_token_limit": 512, "outcome": "passed",
                "comparison_content_passed": True, "wall_seconds": 10.0,
                "native_decode_tps": 8.0, "speed_eligible": True,
            },
            {
                "model": "model-b", "runtime": "runtime-b", "thinking": "medium",
                "case_id": "case", "output_token_limit": 512, "outcome": "wrong_answer",
                "comparison_content_passed": False, "wall_seconds": 8.0,
                "native_decode_tps": 12.0, "speed_eligible": True,
            },
        ]
        chart = final_report._scatter(rows)
        self.assertIn("model-a", chart)
        self.assertIn("model-b", chart)
        self.assertIn("◇ wrong answer", chart)
        self.assertIn("thinking=medium", chart)

    def test_sampler_overrides_are_optional_profile_controls_for_new_rows_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "native.json"
            write_json(config, {"sampling": {"top_p": 0.9}})
            trial = {
                "runtime_backend": "llama", "source_config": str(config),
                "model_name": "resident", "context": 16384,
            }
            base_raw = request_row("legacy")
            base = final_report.normalize_request(
                base_raw, root / "legacy.jsonl", 1, trial, {}, {}, {},
            )
            same_base = final_report.normalize_request(
                json.loads(json.dumps(base_raw)), root / "legacy-copy.jsonl", 1,
                trial, {}, {}, {},
            )
            recipe_a_raw = json.loads(json.dumps(base_raw))
            recipe_a_raw["call_id"] = "recipe-a"
            recipe_a_raw["request"].update({
                "timeout_seconds": 2400.0,
                "sampler_overrides": {"temperature": 1.0, "top_p": 0.95, "top_k": 20},
                "sampler_recipe_label": "publisher-inspired-a",
                "sampler_recipe_sha256": "recipe-a-sha",
                "sampler_recipe_sources": ["https://example.test/a"],
                "sampler_recipe_claim": "publisher_inspired",
                "sampler_application_evidence": "requested_in_outgoing_payload_not_independently_verified_effective",
            })
            recipe_b_raw = json.loads(json.dumps(recipe_a_raw))
            recipe_b_raw["call_id"] = "recipe-b"
            recipe_b_raw["request"]["sampler_overrides"]["top_p"] = 0.8
            recipe_b_raw["request"]["sampler_recipe_label"] = "publisher-inspired-b"
            recipe_b_raw["request"]["sampler_recipe_sha256"] = "recipe-b-sha"
            recipe_a = final_report.normalize_request(
                recipe_a_raw, root / "recipe-a.jsonl", 1, trial, {}, {}, {},
            )
            recipe_b = final_report.normalize_request(
                recipe_b_raw, root / "recipe-b.jsonl", 1, trial, {}, {}, {},
            )
            native_1024_raw = json.loads(json.dumps(recipe_a_raw))
            native_1024_raw["call_id"] = "native-1024"
            native_1024_raw["request"].update({
                "native_thinking_budget_tokens_requested": 1024,
                "native_thinking_budget_application_evidence": (
                    "requested_in_outgoing_payload_not_independently_verified_applied_or_enforced"
                ),
            })
            native_4096_raw = json.loads(json.dumps(native_1024_raw))
            native_4096_raw["call_id"] = "native-4096"
            native_4096_raw["request"]["native_thinking_budget_tokens_requested"] = 4096
            native_1024 = final_report.normalize_request(
                native_1024_raw, root / "native-1024.jsonl", 1, trial, {}, {}, {},
            )
            native_4096 = final_report.normalize_request(
                native_4096_raw, root / "native-4096.jsonl", 1, trial, {}, {}, {},
            )
            schema_a_raw = json.loads(json.dumps(recipe_a_raw))
            schema_a_raw["call_id"] = "schema-a"
            schema_a_raw["request"].update({
                "ledger_json_schema_path": "/schemas/a.json",
                "ledger_json_schema_sha256": "schema-sha-a",
                "ledger_json_schema_requested": {
                    "response_format_type": "json_schema", "name": "ledger_output",
                    "strict": True,
                    "application_evidence": (
                        "strict_json_schema_requested_in_outgoing_payload_"
                        "enforcement_not_independently_verified"
                    ),
                },
            })
            schema_a_moved_raw = json.loads(json.dumps(schema_a_raw))
            schema_a_moved_raw["call_id"] = "schema-a-moved"
            schema_a_moved_raw["request"]["ledger_json_schema_path"] = "/other/a.json"
            schema_b_raw = json.loads(json.dumps(schema_a_raw))
            schema_b_raw["call_id"] = "schema-b"
            schema_b_raw["request"]["ledger_json_schema_sha256"] = "schema-sha-b"
            schema_a = final_report.normalize_request(
                schema_a_raw, root / "schema-a.jsonl", 1, trial, {}, {}, {},
            )
            schema_a_moved = final_report.normalize_request(
                schema_a_moved_raw, root / "schema-a-moved.jsonl", 1, trial, {}, {}, {},
            )
            schema_b = final_report.normalize_request(
                schema_b_raw, root / "schema-b.jsonl", 1, trial, {}, {}, {},
            )

        self.assertEqual(base["profile_id"], same_base["profile_id"])
        self.assertEqual(base["sampling_config"], {"top_p": 0.9})
        for field in (
            "sampler_overrides", "sampler_recipe_label", "sampler_recipe_sha256",
            "sampler_recipe_sources", "sampler_recipe_claim", "sampler_application_evidence",
        ):
            self.assertNotIn(field, base)
        self.assertEqual(recipe_a["sampling_config"]["request_overrides"]["top_p"], 0.95)
        self.assertEqual(recipe_a["sampler_recipe_sources"], ["https://example.test/a"])
        self.assertEqual(recipe_a["actual_context_max_tokens"], 16384)
        self.assertEqual(recipe_a["request_timeout_seconds"], 2400.0)
        self.assertNotEqual(recipe_a["profile_id"], base["profile_id"])
        self.assertNotEqual(recipe_a["profile_id"], recipe_b["profile_id"])
        self.assertNotIn("native_thinking_budget_tokens_requested", base)
        self.assertNotIn("native_thinking_budget_tokens_requested", recipe_a)
        self.assertEqual(native_1024["native_thinking_budget_tokens_requested"], 1024)
        self.assertEqual(native_4096["native_thinking_budget_tokens_requested"], 4096)
        self.assertIsNone(native_1024["reasoning_budget_tokens_requested"])
        self.assertNotEqual(native_1024["profile_id"], recipe_a["profile_id"])
        self.assertNotEqual(native_1024["profile_id"], native_4096["profile_id"])
        self.assertEqual(
            final_report._profile_parts(native_1024)["native_thinking_budget_tokens_requested"],
            1024,
        )
        self.assertNotIn("ledger_json_schema_sha256", base)
        self.assertNotIn("ledger_json_schema_sha256", recipe_a)
        self.assertEqual(schema_a["ledger_json_schema_sha256"], "schema-sha-a")
        self.assertTrue(schema_a["ledger_json_schema_requested"]["strict"])
        self.assertNotEqual(schema_a["profile_id"], recipe_a["profile_id"])
        self.assertEqual(schema_a["profile_id"], schema_a_moved["profile_id"])
        self.assertNotEqual(schema_a["profile_id"], schema_b["profile_id"])

    def test_diagnostic_variants_are_exported_isolated_and_counted_by_parent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            legacy_raw = request_row("legacy")
            original_raw = json.loads(json.dumps(legacy_raw))
            original_raw["call_id"] = "ledger-original"
            original_raw["request"].update({
                "parent_case_id": "hard_transaction_ledger",
                "diagnostic_variant": "original",
            })
            tighter_raw = json.loads(json.dumps(original_raw))
            tighter_raw.update({
                "call_id": "ledger-tighter",
                "case_id": "hard_transaction_ledger_tighter",
            })
            tighter_raw["request"].update({
                "diagnostic_variant": "tighter",
                "prompt_sha256": "prompt-tighter",
            })
            weighted_control_raw = request_row(
                "weighted-control", case_id="hard_weighted_intervals",
                prompt_hash="weighted-prompt",
            )
            weighted_original_raw = json.loads(json.dumps(weighted_control_raw))
            weighted_original_raw["call_id"] = "weighted-original"
            weighted_original_raw["request"].update({
                "parent_case_id": "hard_weighted_intervals",
                "diagnostic_variant": "original",
            })
            clarified_raw = json.loads(json.dumps(weighted_control_raw))
            clarified_raw["call_id"] = "weighted-clarified"
            clarified_raw["request"].update({
                "parent_case_id": "hard_weighted_intervals",
                "diagnostic_variant": "bounded12_exhaustive_allowed",
                "diagnostic_domain_changed": True,
                "diagnostic_algorithm_guidance": True,
            })

            def normalize(raw, name):
                return final_report.normalize_request(
                    raw, root / f"{name}.jsonl", 1, {}, {}, {}, {},
                )

            legacy = normalize(legacy_raw, "legacy")
            original = normalize(original_raw, "ledger-original")
            tighter = normalize(tighter_raw, "ledger-tighter")
            weighted_control = normalize(weighted_control_raw, "weighted-control")
            weighted_original = normalize(weighted_original_raw, "weighted-original")
            clarified = normalize(clarified_raw, "weighted-clarified")

        diagnostic_fields = (
            "parent_case_id", "diagnostic_variant", "diagnostic_domain_changed",
            "diagnostic_algorithm_guidance",
        )
        for field in diagnostic_fields:
            self.assertNotIn(field, legacy)
            self.assertIn(field, final_report.CSV_COLUMNS)
        self.assertEqual(original["parent_case_id"], "hard_transaction_ledger")
        self.assertEqual(original["diagnostic_variant"], "original")
        self.assertNotIn("diagnostic_domain_changed", original)
        self.assertNotIn("diagnostic_algorithm_guidance", original)
        self.assertEqual(original["profile_id"], legacy["profile_id"])
        self.assertEqual(original["match_id"], legacy["match_id"])
        self.assertNotIn("diagnostic_variant", final_report._profile_parts(original))
        self.assertNotIn("diagnostic_variant", final_report._match_parts(original))

        self.assertNotEqual(tighter["profile_id"], original["profile_id"])
        self.assertNotEqual(tighter["match_id"], original["match_id"])
        self.assertEqual(final_report._profile_parts(tighter)["diagnostic_variant"], "tighter")
        self.assertEqual(final_report._match_parts(tighter)["parent_case_id"], "hard_transaction_ledger")

        self.assertEqual(weighted_original["profile_id"], weighted_control["profile_id"])
        self.assertEqual(weighted_original["match_id"], weighted_control["match_id"])
        self.assertNotEqual(clarified["profile_id"], weighted_control["profile_id"])
        self.assertNotEqual(clarified["match_id"], weighted_control["match_id"])
        self.assertTrue(clarified["diagnostic_domain_changed"])
        self.assertTrue(clarified["diagnostic_algorithm_guidance"])
        self.assertEqual(
            final_report._match_parts(clarified)["diagnostic_variant"],
            "bounded12_exhaustive_allowed",
        )

        family_rows = []
        for row in (original, tighter, clarified):
            family_row = dict(row)
            family_row["profile_id"] = "combined-for-count-audit"
            family_rows.append(family_row)
        summary = final_report._profile_summaries(family_rows)[0]
        self.assertEqual(summary["attempt_count"], 3)
        self.assertEqual(summary["independent_task_count"], 2)
        self.assertEqual(summary["variant_task_count"], 3)
        self.assertEqual(summary["parent_case_ids"], [
            "hard_transaction_ledger", "hard_weighted_intervals",
        ])
        self.assertEqual(summary["case_ids"], [
            "hard_transaction_ledger", "hard_transaction_ledger_tighter",
            "hard_weighted_intervals",
        ])

    def test_adaptive_result_uses_longest_unambiguous_session_tag(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "flash-bulk-adaptive-low-hard.jsonl"
            write_jsonl(source, [request_row("adaptive")])
            broad_config = root / "broad.json"
            bulk_config = root / "bulk.json"
            write_json(broad_config, {"args": ["--mtp", "2"]})
            write_json(bulk_config, {"args": ["--mtp", "0"]})
            write_jsonl(root / "session-events.jsonl", [
                {
                    "at": "2026-10-19T01:00:00Z", "event": "session_started",
                    "session_id": "broad", "tag": "flash", "model_name": "broad-model",
                    "source_config": str(broad_config), "engine": "/tmp/strata",
                },
                {
                    "at": "2026-10-19T01:00:01Z", "event": "session_started",
                    "session_id": "bulk", "tag": "flash-bulk", "model_name": "bulk-model",
                    "source_config": str(bulk_config), "engine": "/tmp/strata",
                },
            ])
            report = final_report.build_report(root)

        row = report["request_rows"][0]
        self.assertTrue(row["source_config"].endswith("bulk.json"))
        self.assertEqual(row["runtime_options"]["mtp"], "0")


if __name__ == "__main__":
    unittest.main()
