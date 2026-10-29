import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import bench_cases
from runtime import hard_report


def make_row(
    case,
    *,
    model="model-a",
    thinking="none",
    temperature=0.0,
    seed=0,
    output_limit=None,
    response_text=None,
    passed=True,
    truncated=False,
    error=None,
    prompt_sha256=None,
    reasoning_budget_tokens=None,
    strata_pcie_frac=None,
    strata_aux_cpus=None,
    chat_template_thinking=None,
):
    if response_text is None and case.evaluator == "json_exact":
        response_text = json.dumps(case.expected)
    if response_text is None:
        response_text = ""
    prompt_hash = prompt_sha256 or hashlib.sha256(case.prompt.encode()).hexdigest()
    response = None
    if error is None:
        response = {
            "response_text": response_text,
            "reasoning_chars": 17,
            "timing": {
                "wall_seconds": 4.25,
                "ttft_reasoning_seconds": 0.4,
                "ttft_final_content_seconds": 1.5,
            },
            "actual_prompt_tokens": 101,
            "actual_completion_tokens": 33,
            "done_reason": "length" if truncated else "stop",
            "stream_complete": not truncated,
            "truncated": truncated,
            "throughput": {
                "completion_tokens_per_second_wall": 7.7,
                "completion_tokens_per_second_native": 8.5,
            },
        }
    evaluation = {"passed": passed, "detail": "fixture evaluation"}
    if truncated:
        evaluation["truncated"] = True
    if case.evaluator == "python_function" and not passed and not truncated and error is None:
        evaluation.update({"validation": "accepted", "source_chars": len(response_text)})
    return {
        "schema_version": 1,
        "run_id": "bench-run",
        "call_id": f"call-{case.case_id}-{output_limit or case.max_tokens}",
        "started_at": "2026-10-19T02:00:00Z",
        "finished_at": "2026-10-19T02:00:05Z",
        "repeat_index": 0,
        "suite": case.suite,
        "case_id": case.case_id,
        "case_version": bench_cases.CASE_VERSION,
        "case_source_sha256": "recorded-source-hash",
        "case_metadata": dict(case.metadata),
        "request": {
            "endpoint": "http://user:secret@127.0.0.1:18080/v1?token=secret",
            "backend": "openai",
            "model": model,
            "context_max_tokens": 32768,
            "thinking": thinking,
            "temperature": temperature,
            "seed": seed,
            "output_token_limit": output_limit or case.max_tokens,
            "reasoning_budget_tokens": reasoning_budget_tokens,
            "strata_pcie_frac": strata_pcie_frac,
            "strata_aux_cpus": strata_aux_cpus,
            "chat_template_thinking": chat_template_thinking,
            "prompt_sha256": prompt_hash,
        },
        "response": response,
        "evaluation": evaluation,
        "error": error,
    }


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


class HardReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hard = {case.case_id: case for case in bench_cases.HARD_CASES}

    def test_full_pass_semantic_bug_and_hidden_checks_are_task_weighted(self):
        pass_row = make_row(self.hard["hard_transaction_ledger"])
        buggy_source = "def select_weighted_jobs(jobs):\n    return []\n"
        bug_row = make_row(
            self.hard["hard_weighted_intervals"],
            response_text=buggy_source,
            passed=False,
        )
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "hard.jsonl"
            write_jsonl(source, [pass_row, bug_row])
            report = hard_report.build_report([source.resolve()])

        self.assertEqual(report["independent_task_count"], 2)
        tasks = [task for group in report["groups"] for task in group["tasks"]]
        weighted = next(task for task in tasks if task["case_id"] == "hard_weighted_intervals")
        diagnostics = weighted["hidden_test_diagnostics"]
        self.assertTrue(diagnostics["available"])
        self.assertLess(diagnostics["hidden_tests_passed"], diagnostics["hidden_test_count"])
        self.assertTrue(diagnostics["failed_tests"])
        self.assertTrue(all(item["label"] and item["category"] for item in diagnostics["failed_tests"]))
        self.assertEqual(weighted["rescored_outcome"], "wrong_answer")
        self.assertEqual(weighted["rescored_cause"], "semantic_or_contract_failure")
        serialized = json.dumps(report)
        self.assertNotIn("def select_weighted_jobs", serialized)
        self.assertNotIn("user:secret", serialized)
        self.assertNotIn("token=secret", serialized)

    def test_truncation_timeout_and_request_failure_categories(self):
        truncated = make_row(
            self.hard["hard_temporal_kv"], passed=False, truncated=True, response_text="partial"
        )
        timeout = make_row(
            self.hard["hard_dependency_constraints"],
            passed=False,
            error={"type": "TimeoutError", "message": "request exceeded limit"},
        )
        request_failure = make_row(
            self.hard["hard_transaction_ledger"],
            model="model-b",
            passed=False,
            error={"type": "BenchmarkHTTPError", "message": "HTTP 500"},
        )
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "failures.jsonl"
            write_jsonl(source, [truncated, timeout, request_failure])
            report = hard_report.build_report([source.resolve()])
        outcomes = {
            task["call_id"]: task["outcome"]
            for group in report["groups"]
            for task in group["tasks"]
        }
        self.assertEqual(sorted(outcomes.values()), ["request_failure", "timeout", "truncated"])
        timeout_task = next(
            task for group in report["groups"] for task in group["tasks"] if task["outcome"] == "timeout"
        )
        self.assertEqual(timeout_task["wall_seconds"], 5.0)
        self.assertEqual(timeout_task["request_error_type"], "TimeoutError")

    def test_current_rescore_identifies_recorded_builtin_limitation(self):
        source_code = r'''
def select_weighted_jobs(jobs):
    if type(jobs) is not list:
        raise ValueError()
    seen = set()
    for job in jobs:
        if type(job) is not list or len(job) != 4:
            raise ValueError()
        name, start, end, weight = job
        if type(name) is not str or not name or name in seen:
            raise ValueError()
        if type(start) is not int or type(end) is not int or end <= start:
            raise ValueError()
        if type(weight) is not int or weight < 0:
            raise ValueError()
        seen.add(name)
    best_weight = 0
    best_ids = []
    for mask in range(1 << len(jobs)):
        chosen = [jobs[index] for index in range(len(jobs)) if mask & (1 << index)]
        okay = True
        for left_index in range(len(chosen)):
            for right_index in range(left_index + 1, len(chosen)):
                left = chosen[left_index]
                right = chosen[right_index]
                if not (left[2] <= right[1] or right[2] <= left[1]):
                    okay = False
        if okay:
            total = sum(job[3] for job in chosen)
            ids = sorted(job[0] for job in chosen)
            if total > best_weight or (total == best_weight and ids < best_ids):
                best_weight = total
                best_ids = ids
    return best_ids
'''
        case = self.hard["hard_weighted_intervals"]
        row = make_row(case, response_text=source_code, passed=False)
        row["evaluation"]["detail"] = "test 0 raised NameError"
        with tempfile.TemporaryDirectory() as directory:
            result = Path(directory) / "old-evaluator.jsonl"
            write_jsonl(result, [row])
            report = hard_report.build_report([result.resolve()])
        task = report["groups"][0]["tasks"][0]
        self.assertFalse(task["original_evaluation"]["passed"])
        self.assertEqual(task["rescored_outcome"], "passed")
        self.assertEqual(task["rescored_cause"], "recorded_evaluator_builtin_limitation")
        self.assertEqual(
            task["hidden_test_diagnostics"]["rescored_with_case_source_sha256"],
            report["report_evaluator"]["case_source_sha256"],
        )

    def test_json_failure_reports_keys_and_bounded_values(self):
        case = self.hard["hard_transaction_ledger"]
        actual = dict(case.expected)
        actual.pop("rollback_order")
        actual["extra"] = True
        actual["balances"] = dict(actual["balances"], A=100)
        row = make_row(case, response_text=json.dumps(actual), passed=False)
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "wrong.jsonl"
            write_jsonl(source, [row])
            report = hard_report.build_report([source.resolve()])
        task = report["groups"][0]["tasks"][0]
        detail = task["json_diagnostics"]
        self.assertEqual(detail["top_level_keys"]["missing"], ["rollback_order"])
        self.assertEqual(detail["top_level_keys"]["unexpected"], ["extra"])
        self.assertEqual(detail["top_level_keys"]["different_values"], ["balances"])
        self.assertEqual(detail["expected_value"], case.expected)
        self.assertEqual(detail["actual_value"], actual)

    def test_cap_profiles_lifecycle_metadata_and_exact_comparison_identity(self):
        case = self.hard["hard_transaction_ledger"]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cap_2k = root / "round-2k-hard.jsonl"
            cap_4k = root / "round-4k-hard.jsonl"
            mismatch = root / "round-mismatch-hard.jsonl"
            events = root / "runtime-events.jsonl"
            write_jsonl(cap_2k, [make_row(case, output_limit=2048)])
            write_jsonl(cap_4k, [make_row(case, output_limit=4096)])
            write_jsonl(
                mismatch,
                [make_row(case, model="model-c", output_limit=8192, prompt_sha256="different")],
            )
            write_jsonl(
                events,
                [
                    {
                        "event": "trial_started",
                        "run_id": "trial-1",
                        "tag": "round-2k",
                        "source_config": "/configs/source.json",
                        "private_config": "/results/private.json",
                    },
                    {
                        "event": "suite_started",
                        "run_id": "trial-1",
                        "suite": "hard",
                        "result": str(cap_2k),
                    },
                ],
            )
            report = hard_report.build_report(
                [cap_2k.resolve(), cap_4k.resolve(), mismatch.resolve(), events.resolve()]
            )

        caps = {group["profile"]["output_token_limit"] for group in report["groups"]}
        self.assertEqual(caps, {2048, 4096, 8192})
        enriched = next(group for group in report["groups"] if group["profile"]["output_token_limit"] == 2048)
        self.assertEqual(enriched["profile"]["source_config"], "/configs/source.json")
        self.assertNotIn("private_config", enriched["profile"])
        self.assertEqual(enriched["private_config_paths"], ["/results/private.json"])
        self.assertEqual(enriched["tasks"][0]["private_config"], "/results/private.json")
        self.assertEqual(len(report["comparisons"]), 1)
        comparison = report["comparisons"][0]
        self.assertEqual(comparison["case_id"], case.case_id)
        self.assertEqual(
            {entry["profile"]["output_token_limit"] for entry in comparison["profiles"]},
            {2048, 4096},
        )
        self.assertEqual(report["inputs"]["skipped_non_result_objects"], 2)

    def test_session_step_links_metadata_and_tail_control_splits_profile(self):
        case = self.hard["hard_transaction_ledger"]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            default = root / "session-default-01-hard.jsonl"
            tail_zero = root / "session-tail0-01-hard.jsonl"
            events = root / "session-events.jsonl"
            write_jsonl(default, [make_row(case, seed=0)])
            write_jsonl(tail_zero, [make_row(case, seed=1)])
            event_rows = []
            for suffix, result, environment in (
                ("default", default, {}),
                ("tail0", tail_zero, {"STRATA_ROUTE_TAIL_SKIP": "0", "SECRET": "omit"}),
            ):
                run_id = "session-" + suffix
                event_rows.extend([
                    {
                        "event": "session_started",
                        "run_id": run_id,
                        "tag": "session-" + suffix,
                        "source_config": "/configs/common.json",
                        "private_config": f"/results/{suffix}.json",
                        "engine": "/runtime/strata",
                        "engine_environment": environment,
                    },
                    {
                        "event": "step_started",
                        "run_id": run_id,
                        "step_id": "hard",
                        "result": str(result),
                    },
                ])
            write_jsonl(events, event_rows)
            report = hard_report.build_report([default.resolve(), tail_zero.resolve(), events.resolve()])

        self.assertEqual(len(report["groups"]), 2)
        by_environment = {json.dumps(group["profile"]["engine_environment"], sort_keys=True): group for group in report["groups"]}
        baseline = by_environment["{}"]
        tuned = by_environment['{"STRATA_ROUTE_TAIL_SKIP": "0"}']
        self.assertEqual(baseline["profile"]["source_config"], "/configs/common.json")
        self.assertEqual(baseline["profile"]["runtime"], "/runtime/strata")
        self.assertEqual(tuned["tasks"][0]["private_config"], "/results/tail0.json")
        self.assertNotIn("SECRET", json.dumps(report))

    def test_longest_unambiguous_session_tag_links_adaptive_result(self):
        case = self.hard["hard_transaction_ledger"]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            adaptive = root / "flash-bulk-adaptive-medium-hard.jsonl"
            events = root / "session-events.jsonl"
            write_jsonl(adaptive, [make_row(case)])
            write_jsonl(events, [
                {
                    "event": "session_started", "session_id": "broad", "tag": "flash",
                    "source_config": "/configs/broad.json", "private_config": "/results/broad.json",
                },
                {
                    "event": "session_started", "session_id": "bulk", "tag": "flash-bulk",
                    "source_config": "/configs/bulk.json", "private_config": "/results/bulk.json",
                },
            ])
            report = hard_report.build_report([adaptive.resolve(), events.resolve()])

        group = report["groups"][0]
        self.assertEqual(group["profile"]["source_config"], "/configs/bulk.json")
        self.assertEqual(group["tasks"][0]["private_config"], "/results/bulk.json")
        self.assertEqual(group["tasks"][0]["trial_tag"], "flash-bulk")

    def test_request_controls_are_profile_boundaries_and_preserved_per_task(self):
        case = self.hard["hard_transaction_ledger"]
        rows = [
            make_row(case, seed=0),
            make_row(
                case,
                seed=1,
                reasoning_budget_tokens=768,
                strata_pcie_frac=0.25,
                strata_aux_cpus=1,
                chat_template_thinking="on",
            ),
        ]
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "controls-hard.jsonl"
            write_jsonl(source, rows)
            report = hard_report.build_report([source.resolve()])

        self.assertEqual(len(report["groups"]), 2)
        default = next(group for group in report["groups"] if group["profile"]["reasoning_budget_tokens_requested"] is None)
        controlled = next(group for group in report["groups"] if group["profile"]["reasoning_budget_tokens_requested"] == 768)
        for name in ("strata_pcie_frac", "strata_aux_cpus", "chat_template_thinking"):
            self.assertIsNone(default["profile"][name])
        self.assertEqual(controlled["profile"]["strata_pcie_frac"], 0.25)
        self.assertEqual(controlled["profile"]["strata_aux_cpus"], 1)
        self.assertEqual(controlled["profile"]["chat_template_thinking"], "on")
        task = controlled["tasks"][0]
        self.assertEqual(task["reasoning_budget_tokens_requested"], 768)
        self.assertEqual(task["chat_template_thinking"], "on")

    def test_private_config_filenames_do_not_fragment_equivalent_profiles(self):
        case = self.hard["hard_transaction_ledger"]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "window-a-hard.jsonl"
            second = root / "window-b-hard.jsonl"
            other_source = root / "window-c-hard.jsonl"
            other_temperature = root / "window-d-hard.jsonl"
            events = root / "runtime-events.jsonl"
            write_jsonl(first, [make_row(case, seed=0)])
            write_jsonl(second, [make_row(case, seed=1)])
            write_jsonl(other_source, [make_row(case, seed=2)])
            write_jsonl(other_temperature, [make_row(case, seed=3, temperature=0.7)])
            event_rows = []
            for suffix, result, source_config, private_config, temperature in (
                ("a", first, "/configs/common.json", "/results/a-private.json", 0.0),
                ("b", second, "/configs/common.json", "/results/b-private.json", 0.0),
                ("c", other_source, "/configs/other.json", "/results/c-private.json", 0.0),
                ("d", other_temperature, "/configs/common.json", "/results/d-private.json", 0.7),
            ):
                run_id = "trial-" + suffix
                event_rows.extend(
                    [
                        {
                            "event": "trial_started",
                            "run_id": run_id,
                            "tag": "window-" + suffix,
                            "source_config": source_config,
                            "private_config": private_config,
                            "pool_workers": None,
                            "cpus": None,
                            "temperature": temperature,
                        },
                        {
                            "event": "suite_started",
                            "run_id": run_id,
                            "suite": "hard",
                            "result": str(result),
                        },
                    ]
                )
            write_jsonl(events, event_rows)
            report = hard_report.build_report(
                [first.resolve(), second.resolve(), other_source.resolve(), other_temperature.resolve(), events.resolve()]
            )

        self.assertEqual(len(report["groups"]), 3)
        common = next(
            group
            for group in report["groups"]
            if group["profile"]["source_config"] == "/configs/common.json"
            and group["profile"]["temperature"] == 0.0
        )
        self.assertEqual(common["attempt_count"], 2)
        self.assertEqual(common["independent_task_count"], 1)
        self.assertEqual(
            common["private_config_paths"],
            ["/results/a-private.json", "/results/b-private.json"],
        )
        self.assertEqual(common["coverage"]["case_ids"], [case.case_id])

    def test_sm75_performance_flag_is_profile_boundary_and_old_default_merge(self):
        case = self.hard["hard_transaction_ledger"]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old = root / "old-hard.jsonl"
            default = root / "default-hard.jsonl"
            perf = root / "perf-hard.jsonl"
            events = root / "runtime-events.jsonl"
            for result, seed in ((old, 0), (default, 1), (perf, 2)):
                write_jsonl(result, [make_row(case, seed=seed)])
            event_rows = []
            environments = (
                ("old", old, None),
                ("default", default, {}),
                ("perf", perf, {"STRATA_SM75_PERF": "1", "SECRET": "omit"}),
            )
            for suffix, result, environment in environments:
                run_id = "trial-" + suffix
                started = {
                    "event": "trial_started",
                    "run_id": run_id,
                    "tag": suffix,
                    "source_config": "/configs/common.json",
                    "private_config": f"/results/{suffix}.json",
                }
                if environment is not None:
                    started["engine_environment"] = environment
                event_rows.extend(
                    [
                        started,
                        {
                            "event": "suite_started",
                            "run_id": run_id,
                            "suite": "hard",
                            "result": str(result),
                        },
                    ]
                )
            write_jsonl(events, event_rows)
            report = hard_report.build_report(
                [old.resolve(), default.resolve(), perf.resolve(), events.resolve()]
            )

        self.assertEqual(len(report["groups"]), 2)
        baseline = next(
            group for group in report["groups"] if group["profile"]["engine_environment"] == {}
        )
        optimized = next(
            group
            for group in report["groups"]
            if group["profile"]["engine_environment"] == {"STRATA_SM75_PERF": "1"}
        )
        self.assertEqual(baseline["attempt_count"], 2)
        self.assertEqual(optimized["attempt_count"], 1)
        self.assertTrue(all(task["engine_environment"] == {} for task in baseline["tasks"]))
        self.assertEqual(
            optimized["tasks"][0]["engine_environment"],
            {"STRATA_SM75_PERF": "1"},
        )
        self.assertNotIn("SECRET", json.dumps(report))

    def test_temperature_splits_profiles_while_seeds_are_attempts(self):
        case = self.hard["hard_transaction_ledger"]
        rows = [
            make_row(case, temperature=0.7, seed=0),
            make_row(case, temperature=0.7, seed=1),
            make_row(case, temperature=0.0, seed=0),
        ]
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "sampling.jsonl"
            write_jsonl(source, rows)
            report = hard_report.build_report([source.resolve()])
        self.assertEqual(report["attempt_count"], 3)
        self.assertEqual(report["independent_task_count"], 1)
        self.assertEqual(len(report["groups"]), 2)
        stochastic = next(group for group in report["groups"] if group["profile"]["temperature"] == 0.7)
        self.assertEqual(stochastic["attempt_count"], 2)
        self.assertEqual(stochastic["independent_task_count"], 1)
        self.assertEqual(stochastic["task_count"], 2)
        self.assertEqual(
            [task["sampling_seed"] for task in stochastic["tasks"]],
            [0, 1],
        )
        self.assertTrue(all(task["temperature"] == 0.7 for task in stochastic["tasks"]))
        self.assertEqual(len(report["comparisons"]), 1)

    def test_nonhard_anchor_is_preserved_in_separate_suite_group(self):
        anchor = next(case for case in bench_cases.QUALITY_CASES if case.case_id == "quality_constraints")
        hard = self.hard["hard_transaction_ledger"]
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "mixed.jsonl"
            write_jsonl(source, [make_row(anchor), make_row(hard)])
            report = hard_report.build_report([source.resolve()])
        self.assertEqual({group["suite"] for group in report["groups"]}, {"hard", "quality"})
        quality = next(group for group in report["groups"] if group["suite"] == "quality")
        self.assertEqual(quality["coverage"]["case_ids"], ["quality_constraints"])

    def test_pretty_config_is_skipped_and_cli_writes_atomically(self):
        case = self.hard["hard_transaction_ledger"]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = root / "hard.jsonl"
            config = root / "config.json"
            output = root / "report.json"
            write_jsonl(result, [make_row(case)])
            config.write_text(json.dumps({"engine": "fixture"}, indent=2), encoding="utf-8")
            code = hard_report.main(
                ["--inputs", str(result), str(config), "--output", str(output)]
            )
            self.assertEqual(code, 0)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(report["independent_task_count"], 1)
            self.assertEqual(report["inputs"]["skipped_non_result_objects"], 1)
            with self.assertRaises(hard_report.ReportError):
                hard_report.resolve_inputs([str(result)], result)


if __name__ == "__main__":
    unittest.main()
