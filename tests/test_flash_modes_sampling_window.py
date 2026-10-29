import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from runtime import flash_modes_sampling_window as window


def write_json(path: Path, value):
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(payload).hexdigest(), "size": len(payload)}


def write_jsonl(path: Path, rows):
    raw_lines = [(json.dumps(row, separators=(",", ":")) + "\n").encode() for row in rows]
    payload = b"".join(raw_lines)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    binding = {"path": str(path.resolve()), "sha256": hashlib.sha256(payload).hexdigest(), "size": len(payload)}
    pins = []
    for index, (row, raw_line) in enumerate(zip(rows, raw_lines), 1):
        raw = raw_line.rstrip(b"\n")
        pins.append({
            "case_id": row.get("case_id"),
            "source_line": index,
            "source_raw_line_sha256": hashlib.sha256(raw).hexdigest(),
            "source_row_canonical_sha256": hashlib.sha256(window.canonical_bytes(row)).hexdigest(),
        })
    return binding, pins


def generic_main_row(case_id, *, passed=True):
    return {
        "schema_version": 1,
        "case_id": case_id,
        "request": {},
        "response": {"stream_complete": True, "truncated": False},
        "evaluation": {"passed": passed},
        "error": None,
    }


def main_code_row(case_id, *, passed=True, wall=100.0, cache=0):
    identity = window.CASE_IDENTITIES[case_id]
    return {
        "schema_version": 1,
        "tag": f"{window.MAIN_TAG}-code",
        "suite": "specialist_code",
        "case_group": "code",
        "repeat_index": 0,
        "case_id": case_id,
        "case_version": "2026-10-19.specialist.2",
        "case_sha256": identity["case_sha256"],
        "case_source_sha256": window.STATIC_PINS[window.RUNTIME / "specialist_cases.py"],
        "request": {
            "model": window.MODEL,
            "thinking": "medium",
            "temperature": 0,
            "seed": 0,
            "context_max_tokens": window.CONTEXT,
            "output_token_limit": window.OUTPUT_CAP,
            "timeout_seconds": window.REQUEST_TIMEOUT,
            "prompt_sha256": identity["prompt_sha256"],
        },
        "response": {"stream_complete": True, "truncated": False},
        "response_metrics": {
            "wall_seconds": wall,
            "native_prompt_tokens": 100,
            "native_cached_prompt_tokens": cache,
        },
        "evaluation": {"passed": passed},
        "error": None,
    }


def result_summary(*, passed=True, complete=True, request_error=False, wall=90.0,
                   prompt=100, cache=0):
    return {
        "passed": passed,
        "complete": complete,
        "request_error": request_error,
        "wall_seconds": wall,
        "native_prompt_tokens": prompt,
        "native_cached_prompt_tokens": cache,
    }


class AcceptanceFixture:
    def __init__(self, root: Path):
        self.root = root
        self.results = root / "results"
        self.runtime = root / "runtime"
        self.results.mkdir()
        self.runtime.mkdir()
        self.run_id = "0484c7f6-eb4b-4dad-8ed1-bf6b103bf226"
        self.plan = root / "plan.json"
        self.plan.write_text('{"plan":1}\n')
        self.preflight = self.results / "preflight.json"
        self.preflight.write_text('{"preflight":1}\n')
        self.source_config = self.runtime / "source.json"
        self.source_config.write_text('{"source":1}\n')
        self.boundary = self.results / f"{window.RUN_PREFIX}-boundaries"
        self.boundary.mkdir()
        self.release = self.boundary / f"{window.RUN_PREFIX}-{window.MODEL_KEY}-release.json"
        self.paths = {
            name: self.results / f"main-{name}.jsonl" for name in window.MAIN_CASES
        }
        rows = {
            "quick2": [generic_main_row(case) for case in window.MAIN_CASES["quick2"]],
            "hard4": [generic_main_row(case) for case in window.MAIN_CASES["hard4"]],
            "code3": [main_code_row(case) for case in window.MAIN_CASES["code3"]],
            "prose3": [generic_main_row(case) for case in window.MAIN_CASES["prose3"]],
        }
        self.artifacts = {}
        self.artifact_bindings = {}
        for name, artifact_rows in rows.items():
            binding, pins = write_jsonl(self.paths[name], artifact_rows)
            self.artifacts[name] = {"source": binding, "rows": pins}
            self.artifact_bindings[name] = binding
        self.study_path = self.results / f"{window.MAIN_TAG}-study-trial.json"
        self.study = {
            "schema_version": 1,
            "tag": window.MAIN_TAG,
            "status": "completed_with_evaluation_failures",
            "finished_at": "2026-10-21T01:59:00Z",
            "runtime_backend": "strata",
            "source_config": {
                "path": str(self.source_config.resolve()),
                "sha256": hashlib.sha256(self.source_config.read_bytes()).hexdigest(),
            },
            "session": {"run_id": self.run_id, "model_name": window.MODEL, "context": window.CONTEXT},
            "cleanup": {
                "outer_exited": True,
                "orchestrator_sigkill_sent": False,
                "model_session_cleanup": {"stopped": True, "kill_sent": False},
            },
            "restoration": {
                "original_running": True, "original_paused": False,
                "owned_loopback_port_clear": True,
            },
            "problem": None,
            "session_report_status": "completed_with_step_failures",
            "helper_sources": {
                str(window.RUNTIME / "gpu_window.py"): window.STATIC_PINS[window.RUNTIME / "gpu_window.py"],
                str(window.RUNTIME / "model_session.py"): window.STATIC_PINS[window.RUNTIME / "model_session.py"],
                str(window.RUNTIME / "specialist_bench.py"): window.STATIC_PINS[window.RUNTIME / "specialist_bench.py"],
            },
            "controls": {
                "code_thinking": "medium", "prose_thinking": "none",
                "max_seconds": 86400.0, "startup_timeout": 480.0, "wait_for_idle": 900.0,
                "code": {"output_cap": 65536, "temperature": 0.0, "seed": 0, "timeout": 21600.0},
                "prose": {"output_cap": 65536, "temperature": 0.7, "seed": 0, "timeout": 21600.0},
            },
            "plan": {
                "steps": [
                    {
                        "id": "quick_medium_65536", "suite": "quick", "thinking": "medium",
                        "temperature": 0.0, "seed": 0, "output_tokens": 65536,
                        "request_timeout": 21600.0, "repeat": 1,
                    },
                    {
                        "id": "hard_medium_65536", "suite": "hard", "thinking": "medium",
                        "temperature": 0.0, "seed": 0, "output_tokens": 65536,
                        "request_timeout": 21600.0, "repeat": 1,
                    },
                ],
            },
            "commands": [
                {"kind": "gpu_window", "returncode": 1},
                {"kind": "specialist_code", "returncode": 0},
                {"kind": "specialist_prose", "returncode": 0},
            ],
            "artifacts": {
                study_name: {
                    "exists": True,
                    "path": self.artifact_bindings[name]["path"],
                    "size": self.artifact_bindings[name]["size"],
                }
                for name, study_name in {
                    "quick2": "plan_01_quick_medium_65536",
                    "hard4": "plan_02_hard_medium_65536",
                    "code3": "code",
                    "prose3": "prose",
                }.items()
            },
        }
        self.study_binding = write_json(self.study_path, self.study)
        self.ready_path = self.boundary / f"{window.RUN_PREFIX}-{window.MODEL_KEY}-ready.json"
        source_binding = {
            "path": str(self.source_config.resolve()),
            "sha256": hashlib.sha256(self.source_config.read_bytes()).hexdigest(),
            "size": self.source_config.stat().st_size,
        }
        self.ready = {
            "schema_version": 1,
            "kind": "model_boundary_ready",
            "model_key": window.MODEL_KEY,
            "published_at": "2026-10-21T02:00:00Z",
            "run_prefix": window.RUN_PREFIX,
            "study": {"path": self.study_binding["path"], "sha256": self.study_binding["sha256"]},
        }
        self.ready_binding = write_json(self.ready_path, self.ready)
        self.test_review = root / "reviewed-test.py"
        self.test_review.write_text("# reviewed\n")
        self.acceptance_path = self.results / "acceptance.json"
        self.acceptance = {
            "schema_version": 1,
            "kind": "flash_modes_sampling_parent_execution_acceptance",
            "accepted": True,
            "created_at": "2026-10-21T02:01:00Z",
            "review": {
                "plan": self.binding(self.plan),
                "controller": self.binding(Path(window.__file__)),
                "tests": self.binding(self.test_review),
                "preflight": self.binding(self.preflight),
            },
            "main": {
                "tag": window.MAIN_TAG,
                "model": window.MODEL,
                "model_session_run_id": self.run_id,
                "source_config": source_binding,
                "ready": self.ready_binding,
                "study": self.study_binding,
                "artifacts": self.artifacts,
                "release_path": str(self.release.resolve()),
                "restoration_attestation": {
                    "kind": "flash_modes_sampling_parent_restoration_attestation",
                    "observed_at": "2026-10-21T02:00:30Z",
                    "original_running": True,
                    "original_paused": False,
                    "original_api_healthy": True,
                    "ports_clear": {"18080": True, "18081": True},
                    "no_foreign_gpu_process": True,
                    "release_absent": True,
                },
            },
        }
        write_json(self.acceptance_path, self.acceptance)
        self.patches = [
            mock.patch.object(window, "validate_static_sources", return_value={}),
            mock.patch.multiple(
                window,
                PLAN_PATH=self.plan,
                PLAN_SHA256=hashlib.sha256(self.plan.read_bytes()).hexdigest(),
                TEST_PATH=self.test_review,
                PREFLIGHT_PATH=self.preflight,
                SOURCE_CONFIG=self.source_config,
                SOURCE_CONFIG_SHA256=source_binding["sha256"],
                SOURCE_CONFIG_SIZE=source_binding["size"],
                MAIN_RELEASE=self.release,
                MAIN_READY=self.ready_path,
                MAIN_STUDY=self.study_path,
                MAIN_ARTIFACT_PATHS=self.paths,
            ),
        ]

    @staticmethod
    def binding(path):
        path = path.resolve()
        return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "size": path.stat().st_size}

    def start(self):
        for patcher in self.patches:
            patcher.start()

    def stop(self):
        for patcher in reversed(self.patches):
            patcher.stop()

    def rewrite_acceptance(self):
        write_json(self.acceptance_path, self.acceptance)


class AcceptanceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.fixture = AcceptanceFixture(Path(self.temp.name))
        self.fixture.start()
        self.addCleanup(self.fixture.stop)

    def test_acceptance_validates_main12_hashes_and_code_controls(self):
        accepted = window.validate_acceptance(self.fixture.acceptance_path)
        self.assertEqual(accepted.main_run_id, self.fixture.run_id)
        self.assertEqual(sum(len(rows) for rows in accepted.main_rows.values()), 12)
        self.assertEqual([row.value["case_id"] for row in accepted.main_rows["code3"]], list(window.CODE_CASES))

    def test_acceptance_fails_closed_when_not_affirmative(self):
        self.fixture.acceptance["accepted"] = False
        self.fixture.rewrite_acceptance()
        with self.assertRaisesRegex(window.ControllerError, "not affirmative"):
            window.validate_acceptance(self.fixture.acceptance_path)

    def test_acceptance_rejects_artifact_hash_mismatch(self):
        self.fixture.acceptance["main"]["artifacts"]["quick2"]["source"]["sha256"] = "0" * 64
        self.fixture.rewrite_acceptance()
        with self.assertRaisesRegex(window.ControllerError, "binding does not match"):
            window.validate_acceptance(self.fixture.acceptance_path)

    def test_acceptance_rejects_case_pin_mismatch(self):
        self.fixture.acceptance["main"]["artifacts"]["code3"]["rows"][0]["case_id"] = "wrong"
        self.fixture.rewrite_acceptance()
        with self.assertRaisesRegex(window.ControllerError, "case identity differs"):
            window.validate_acceptance(self.fixture.acceptance_path)

    def test_acceptance_rejects_main_code_control_mismatch(self):
        rows = [main_code_row(case) for case in window.CODE_CASES]
        rows[0]["request"]["context_max_tokens"] = 32768
        binding, pins = write_jsonl(self.fixture.paths["code3"], rows)
        self.fixture.acceptance["main"]["artifacts"]["code3"] = {"source": binding, "rows": pins}
        self.fixture.rewrite_acceptance()
        with self.assertRaisesRegex(window.ControllerError, "mismatched context_max_tokens"):
            window.validate_acceptance(self.fixture.acceptance_path)

    def test_invented_ready_schema_cannot_masquerade_as_boundary_owner_ready(self):
        invented = {
            "schema_version": 1,
            "kind": "flash_main_ready_for_modes_sampling",
            "ready": True,
            "study": self.fixture.study_binding,
        }
        self.fixture.ready_binding = write_json(self.fixture.ready_path, invented)
        self.fixture.acceptance["main"]["ready"] = self.fixture.ready_binding
        self.fixture.rewrite_acceptance()
        with self.assertRaisesRegex(window.ControllerError, "model_boundary_gate schema"):
            window.validate_acceptance(self.fixture.acceptance_path)

    def test_actual_lowercase_boundary_release_must_remain_absent(self):
        self.assertTrue(self.fixture.release.name.endswith("-release.json"))
        self.assertNotIn("RELEASE", self.fixture.release.name)
        self.fixture.release.write_text('{"released":true}\n')
        with self.assertRaisesRegex(window.ControllerError, "main RELEASE is present"):
            window.validate_acceptance(self.fixture.acceptance_path)

    def test_direct_path_load_bootstraps_runtime_package_without_pythonpath(self):
        source = Path(window.__file__).resolve()
        program = f"""
import importlib.util
import sys
source = {str(source)!r}
spec = importlib.util.spec_from_file_location('flash_modes_direct_test', source)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
from runtime import gpu_window, specialist_study_trial
assert gpu_window.ROOT == module.ROOT
assert specialist_study_trial.ROOT == module.ROOT
assert module.PREFLIGHT_PATH.name == 'flash-modes-sampling-controller-preflight-v2.json'
"""
        environment = dict(os.environ)
        environment.pop("PYTHONPATH", None)
        environment.pop("PYTHONHOME", None)
        with tempfile.TemporaryDirectory() as tmp:
            completed = subprocess.run(
                [sys.executable, "-I", "-c", program],
                cwd=tmp, env=environment, text=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                timeout=10, check=False,
            )
        self.assertEqual(completed.returncode, 0, completed.stderr)


class BudgetAndOutcomeTests(unittest.TestCase):
    def test_unused_tag_check_catches_unenumerated_tag_file_but_allows_acceptance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = window.controller_paths(root)
            acceptance = root / f"{window.TAG}-parent-acceptance-v1.json"
            acceptance.write_text("{}\n")
            window.assert_outputs_unused(paths, allowed_tag_paths=(acceptance,))
            unexpected = root / f"{window.TAG}-unexpected.json"
            unexpected.write_text("{}\n")
            with self.assertRaisesRegex(window.ControllerError, "tag/output is already used"):
                window.assert_outputs_unused(paths, allowed_tag_paths=(acceptance,))

    def test_selected_pilot_requires_full_six_hours_plus_reserve(self):
        outer_allowance = window.REQUEST_TIMEOUT + 90 + window.RESTORATION_RESERVE_SECONDS
        work_deadline = 1000.0 + outer_allowance - window.RESTORATION_RESERVE_SECONDS
        with mock.patch.object(window.time, "monotonic", return_value=1000.0):
            window.ensure_time_for_stage(work_deadline, 1, "selected T1 pilot")
            with self.assertRaisesRegex(window.ControllerError, "cannot fit"):
                window.ensure_time_for_stage(work_deadline - 0.001, 1, "selected T1 pilot")

    def test_budget_refuses_duplicate_setup_and_over_reservation(self):
        ledger = window.BudgetLedger()
        ledger.reserve("setup", 1)
        with self.assertRaisesRegex(window.ControllerError, "reserved twice"):
            ledger.reserve("setup", 1)
        ledger.reserve("none", 3)
        ledger.reserve("pilot", 1)
        with self.assertRaisesRegex(window.ControllerError, "exceeds"):
            ledger.reserve("too_many", 3)

    def test_interrupted_helper_reports_one_uncertain_started_call(self):
        ledger = window.BudgetLedger()
        ledger.reserve("none", 3)
        ledger.mark_launched("none")
        ledger.observe("none", 1, helper_terminal=False)
        stage = ledger.summary()["stages"]["none"]
        self.assertEqual(stage["attempts_lower_bound"], 1)
        self.assertEqual(stage["attempts_upper_bound"], 2)
        self.assertIsNone(stage["exact_attempts"])
        self.assertTrue(stage["has_uncertain_started_unrecorded_call"])

    def test_nonzero_helper_with_empty_jsonl_preserves_started_call_uncertainty(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "none.jsonl"
            output.write_bytes(b"")
            ledger = window.BudgetLedger()
            ledger.reserve("none_code3", 3)
            ledger.mark_launched("none_code3")
            runner = mock.Mock(return_value={"returncode": 1})
            run, rows = window.run_helper_accounted(
                ["python3", "helper.py"], stage="none_code3", ledger=ledger,
                output=output, expected_new_rows=3, existing_rows=0,
                outer=mock.Mock(), deadline=9999, signals=window.SignalLatch(),
                runner=runner,
            )
            self.assertEqual(run["returncode"], 1)
            self.assertEqual(rows, ())
            stage = ledger.summary()["stages"]["none_code3"]
            self.assertEqual(stage["attempts_lower_bound"], 0)
            self.assertEqual(stage["attempts_upper_bound"], 1)
            self.assertIsNone(stage["exact_attempts"])

    def test_unlaunched_stage_is_exact_zero_not_reserved_slots(self):
        ledger = window.BudgetLedger()
        ledger.reserve("continuation", 2)
        ledger.observe("continuation", 0, helper_terminal=False)
        stage = ledger.summary()["stages"]["continuation"]
        self.assertEqual(stage["exact_attempts"], 0)
        self.assertEqual(stage["attempts_upper_bound"], 0)

    def test_zero_and_unknown_attempt_counts_remain_distinct(self):
        ledger = window.BudgetLedger()
        ledger.reserve("shared_setup", 1)
        ledger.observe("shared_setup", 0, helper_terminal=False)
        self.assertEqual(ledger.summary()["stages"]["shared_setup"]["exact_attempts"], 0)

        ledger.reserve("started_helper", 1)
        ledger.mark_launched("started_helper")
        ledger.observe("started_helper", 0, helper_terminal=False)
        started = ledger.summary()["stages"]["started_helper"]
        self.assertIsNone(started["exact_attempts"])
        self.assertEqual(started["attempts_upper_bound"], 1)

    def test_semantic_negative_is_distinct_from_infrastructure_error(self):
        self.assertEqual(
            window.specialist_stage_outcome([result_summary(passed=False)]),
            "semantic_negative",
        )
        self.assertEqual(
            window.specialist_stage_outcome([result_summary(request_error=True)]),
            "infrastructure_error",
        )
        self.assertEqual(
            window.specialist_stage_outcome([result_summary(passed=True)]),
            "pass",
        )

    def test_setup_semantic_failure_can_continue_but_request_error_cannot(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "setup.jsonl"
            request = {
                "model": window.MODEL,
                "thinking": "none",
                "temperature": 0,
                "seed": 0,
                "context_max_tokens": window.CONTEXT,
                "output_token_limit": 256,
                "timeout_seconds": 180,
            }
            row = {
                "case_id": "quick_json_extract", "request": request,
                "response": {"stream_complete": True}, "evaluation": {"passed": False},
                "error": None,
            }
            write_jsonl(path, [row])
            summary = window.validate_setup_result(path)
            self.assertFalse(summary["semantic_pass"])
            self.assertTrue(summary["semantic_failure_allows_mandatory_code3"])
            row["response"] = None
            row["error"] = {"type": "TimeoutError", "message": "timeout"}
            write_jsonl(path, [row])
            summary = window.validate_setup_result(path)
            self.assertTrue(summary["request_error"])
            self.assertFalse(summary["semantic_failure_allows_mandatory_code3"])


class ContinuationGateTests(unittest.TestCase):
    def test_pilot_fix_and_none_failure_passes_gate_without_speed_claim(self):
        outcome = window.evaluate_continuation_gate(
            pilot=result_summary(passed=True, prompt=None, cache=None),
            main_t0=result_summary(passed=False, prompt=None, cache=None),
            none_fresh=result_summary(passed=False, prompt=None, cache=None),
            none_pass_count=2,
        )
        self.assertTrue(outcome["continue_remaining_two"])
        self.assertEqual(outcome["usefulness_reason"], "pilot_fixes_canonical_t0_failure")

    def test_zero_cache_is_known_and_can_support_five_percent_speed_gate(self):
        pilot = result_summary(wall=94.9, cache=0)
        reference = result_summary(wall=100.0, cache=0)
        outcome = window.evaluate_continuation_gate(
            pilot=pilot, main_t0=reference, none_fresh=reference, none_pass_count=3,
        )
        self.assertTrue(outcome["main_t0_speed_comparison"]["cache"]["known"])
        self.assertEqual(
            outcome["main_t0_speed_comparison"]["cache"]["values"]["native_cached_prompt_tokens"]["candidate"],
            0,
        )
        self.assertTrue(outcome["continue_remaining_two"])

    def test_unknown_cache_blocks_temperature_only_speed_claim(self):
        pilot = result_summary(wall=80.0, cache=None)
        reference = result_summary(wall=100.0, cache=None)
        outcome = window.evaluate_continuation_gate(
            pilot=pilot, main_t0=reference, none_fresh=reference, none_pass_count=3,
        )
        self.assertFalse(outcome["main_t0_speed_comparison"]["cache"]["known"])
        self.assertFalse(outcome["continue_remaining_two"])

    def test_wrong_or_incomplete_pilot_is_normal_negative_gate(self):
        for pilot in (
            result_summary(passed=False),
            result_summary(passed=True, complete=False),
        ):
            outcome = window.evaluate_continuation_gate(
                pilot=pilot,
                main_t0=result_summary(passed=False),
                none_fresh=result_summary(passed=False),
                none_pass_count=0,
            )
            self.assertFalse(outcome["continue_remaining_two"])
            self.assertFalse(outcome["pilot_normal_complete_full_case_pass"])

    def test_pilot_prefix_must_remain_append_stable(self):
        row = window.JsonlRow(1, {"case_id": window.CODE_CASES[0]}, "a" * 64, "b" * 64)
        window.assert_pilot_prefix_unchanged(row, [row])
        replaced = window.JsonlRow(1, row.value, "c" * 64, row.canonical_sha256)
        with self.assertRaisesRegex(window.ControllerError, "changed or was reexecuted"):
            window.assert_pilot_prefix_unchanged(row, [replaced])

    def test_partial_mode_comparison_reports_only_actual_rows_and_units(self):
        reference = [{
            "case_id": window.CODE_CASES[0], "passed": False, "complete": True,
            "wall_seconds": 100.0, "native_prompt_tokens": 100,
            "native_cached_prompt_tokens": 0,
            "native_decode_rate": {"value": None, "unit": "tokens_per_second"},
        }]
        candidate = [{
            "case_id": window.CODE_CASES[0], "passed": True, "complete": True,
            "wall_seconds": 90.0, "native_prompt_tokens": 100,
            "native_cached_prompt_tokens": 0,
            "native_decode_rate": {"value": 4.5, "unit": "tokens_per_second"},
        }]
        comparison = window.mode_comparison(
            reference=reference, candidate=candidate,
            reference_mode="T0", candidate_mode="T1",
        )
        self.assertEqual(comparison["actual_case_count"], 1)
        self.assertEqual(comparison["candidate_pass_count"], 1)
        self.assertEqual(comparison["rows"][0]["candidate_minus_reference_wall"], {
            "value": -10.0, "unit": "seconds",
        })
        self.assertEqual(
            comparison["rows"][0]["candidate"]["native_decode_rate"]["unit"],
            "tokens_per_second",
        )


class LiveHealthTests(unittest.TestCase):
    def health(self, **changes):
        payload = {
            "status": "ok", "service": "strata", "loaded": True,
            "model": window.MODEL, "max_context": window.CONTEXT,
            "images": False, "api_key": False, "image_generation": False,
        }
        payload.update(changes)
        return payload

    def test_health_requires_actual_loaded_engine_context(self):
        observed = window.validate_health_payload(self.health())
        self.assertTrue(observed["loaded"])
        self.assertEqual(observed["max_context"], window.CONTEXT)
        for payload, message in (
            (self.health(loaded=False), "not loaded"),
            (self.health(max_context=window.CONTEXT - 1), "wrong loaded context"),
            (self.health(max_context=True), "wrong loaded context"),
        ):
            with self.subTest(payload=payload):
                with self.assertRaisesRegex(window.ControllerError, message):
                    window.validate_health_payload(payload)

    def test_health_rejects_wrong_model_service_or_status(self):
        cases = (
            (self.health(model="other"), 200, "wrong model"),
            (self.health(service="other"), 200, "wrong service identity"),
            (self.health(), 503, "unavailable or malformed"),
        )
        for payload, status, message in cases:
            with self.subTest(payload=payload, status=status):
                with self.assertRaisesRegex(window.ControllerError, message):
                    window.validate_health_payload(payload, status_code=status)


class OwnershipTests(unittest.TestCase):
    def setUp(self):
        start_value = {
            "event": "session_started", "tag": window.TAG, "run_id": "run-new",
        }
        server_value = {
            "event": "server_started", "tag": window.TAG, "run_id": "run-new", "pid": 10,
        }
        self.rows = []
        for line, value in enumerate((start_value, server_value), 1):
            raw = json.dumps(value, separators=(",", ":")).encode()
            self.rows.append(window.JsonlRow(
                line, value, hashlib.sha256(raw).hexdigest(),
                hashlib.sha256(window.canonical_bytes(value)).hexdigest(),
            ))
        self.private = {"path": "/tmp/private", "sha256": "1" * 64, "size": 1}
        self.source = {"path": "/tmp/source", "sha256": "2" * 64, "size": 1}
        self.receipt = {
            "api_process": {"exe": "/usr/bin/python3", "parent_pid": 9, "pid": 10, "start_ticks": "100", "state": "S"},
            "direct_child_parent_relation_verified": True,
            "engine_environment": window.TAIL_ENVIRONMENT,
            "expected_engine_exe": window.EXPECTED_ENGINE,
            "gpu_observation": {"name": window.EXPECTED_ENGINE, "pid": 11, "used_mib": 6000},
            "gpu_process": {"exe": window.EXPECTED_ENGINE, "parent_pid": 10, "pid": 11, "start_ticks": "150", "state": "S"},
            "kernel_clock": {
                "boot_id": "boot", "boot_time_resolution_seconds": 1,
                "boot_time_unix_seconds": 1000, "clock_ticks_per_second": 100,
                "gpu_process_start_unix_seconds": 1001.5,
            },
            "kind": "strata_owned_gpu_process_observation",
            "lifecycle_events": [
                {
                    "event": row.value["event"], "source_line": row.line,
                    "source_raw_line_sha256": row.raw_sha256,
                    "source_row_canonical_sha256": row.canonical_sha256,
                }
                for row in self.rows
            ],
            "lifecycle_source": "/tmp/events.jsonl",
            "live_kernel_start_identities_verified": True,
            "model_session_run_id": "run-new",
            "observed_at": "2026-10-21T03:00:00Z",
            "observed_during_live_matching_model_session": True,
            "private_config": self.private,
            "prose_bodies_read": False,
            "runtime_backend": "strata",
            "schema_version": 1,
            "scope": "fixture ownership",
            "source_config": self.source,
            "tag": window.TAG,
        }

    def validate(self, receipt=None):
        window.validate_ownership_document(
            receipt or self.receipt,
            lifecycle_rows=self.rows,
            private_config=self.private,
            source_config=self.source,
            generated_at=datetime(2026, 10, 21, 3, 1, tzinfo=timezone.utc),
        )

    def test_exact_fake_ownership_receipt_is_accepted(self):
        self.validate()

    def test_api_pid_cannot_be_falsified_as_gpu_pid(self):
        receipt = copy.deepcopy(self.receipt)
        receipt["gpu_process"]["pid"] = 10
        receipt["gpu_observation"]["pid"] = 10
        with self.assertRaisesRegex(window.ControllerError, "relationship differs"):
            self.validate(receipt)

    def test_lifecycle_raw_hash_mismatch_is_rejected(self):
        receipt = copy.deepcopy(self.receipt)
        receipt["lifecycle_events"][0]["source_raw_line_sha256"] = "0" * 64
        with self.assertRaisesRegex(window.ControllerError, "row hash differs"):
            self.validate(receipt)

    def test_kernel_birth_mismatch_is_rejected(self):
        receipt = copy.deepcopy(self.receipt)
        receipt["kernel_clock"]["gpu_process_start_unix_seconds"] = 1002
        with self.assertRaisesRegex(window.ControllerError, "birth time differs"):
            self.validate(receipt)


class CleanupTests(unittest.TestCase):
    def lifecycle_fixture(self, root):
        source = root / "source.json"
        plan = root / "setup.json"
        control = root / "control.json"
        events = root / "events.jsonl"
        source.write_text("{}\n")
        plan.write_text("{}\n")
        control.write_text('{"stop":false}\n')
        run_id = "owned-run"
        rows = [
            {
                "event": "session_started", "tag": window.TAG, "run_id": run_id,
                "source_config": str(source.resolve()), "plan": str(plan.resolve()),
                "runtime_backend": "strata", "model_name": window.MODEL,
                "context": window.CONTEXT, "control_file": str(control.resolve()),
                "engine_environment": window.TAIL_ENVIRONMENT,
            },
            {"event": "server_started", "tag": window.TAG, "run_id": run_id, "pid": 123},
            {"event": "server_ready", "tag": window.TAG, "run_id": run_id, "model": window.MODEL},
            {"event": "holding_started", "tag": window.TAG, "run_id": run_id, "control_file": str(control.resolve())},
        ]
        write_jsonl(events, rows)
        paths = window.controller_paths(root)
        paths["control"] = control
        paths["runtime_events"] = events
        return source, plan, control, events, paths

    def test_owned_stop_requires_exact_run_process_and_control_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, plan, control, events, paths = self.lifecycle_fixture(root)
            with (
                mock.patch.multiple(window, SOURCE_CONFIG=source.resolve(), SETUP_PLAN=plan.resolve()),
                mock.patch.object(window, "controller_paths", return_value=paths),
                mock.patch.object(window, "_proc_identity", return_value=window.ProcessIdentity(123, 1, "10", "S", "/usr/bin/python3")),
            ):
                session = window.holding_session(events)
                calls = []
                identity = window.ProcessIdentity(123, 1, "10", "S", "/usr/bin/python3")
                self.assertTrue(window.request_owned_stop(
                    control, events, session, expected_api_identity=identity,
                    writer=lambda path: calls.append(path) or True,
                ))
                self.assertEqual(calls, [control])

    def test_owned_stop_refuses_reused_api_pid_with_new_start_ticks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, plan, control, events, paths = self.lifecycle_fixture(root)
            observed = window.ProcessIdentity(123, 1, "11", "S", "/usr/bin/python3")
            expected = window.ProcessIdentity(123, 1, "10", "S", "/usr/bin/python3")
            with (
                mock.patch.multiple(window, SOURCE_CONFIG=source.resolve(), SETUP_PLAN=plan.resolve()),
                mock.patch.object(window, "controller_paths", return_value=paths),
                mock.patch.object(window, "_proc_identity", return_value=observed),
            ):
                session = window.holding_session(events)
                writer = mock.Mock()
                with self.assertRaisesRegex(window.ControllerError, "identity changed"):
                    window.request_owned_stop(
                        control, events, session, expected_api_identity=expected, writer=writer,
                    )
                writer.assert_not_called()

    def test_owned_stop_allows_normal_process_state_transition(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, plan, control, events, paths = self.lifecycle_fixture(root)
            observed = window.ProcessIdentity(123, 1, "10", "R", "/usr/bin/python3")
            expected = window.ProcessIdentity(123, 1, "10", "S", "/usr/bin/python3")
            with (
                mock.patch.multiple(window, SOURCE_CONFIG=source.resolve(), SETUP_PLAN=plan.resolve()),
                mock.patch.object(window, "controller_paths", return_value=paths),
                mock.patch.object(window, "_proc_identity", return_value=observed),
            ):
                session = window.holding_session(events)
                calls = []
                self.assertTrue(window.request_owned_stop(
                    control, events, session, expected_api_identity=expected,
                    writer=lambda path: calls.append(path) or True,
                ))
                self.assertEqual(calls, [control])

    def test_stable_identity_rejects_zombie_start_exe_and_parent_changes(self):
        expected = window.ProcessIdentity(123, 1, "10", "S", "/usr/bin/python3")
        self.assertTrue(window._stable_process_identity_matches(
            window.ProcessIdentity(123, 1, "10", "R", "/usr/bin/python3"), expected,
        ))
        for observed in (
            window.ProcessIdentity(123, 1, "10", "Z", "/usr/bin/python3"),
            window.ProcessIdentity(123, 1, "11", "S", "/usr/bin/python3"),
            window.ProcessIdentity(123, 1, "10", "S", "/other/python3"),
            window.ProcessIdentity(123, 2, "10", "S", "/usr/bin/python3"),
        ):
            with self.subTest(observed=observed):
                self.assertFalse(window._stable_process_identity_matches(observed, expected))

    def test_owned_stop_refuses_changed_run_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, plan, control, events, paths = self.lifecycle_fixture(root)
            with (
                mock.patch.multiple(window, SOURCE_CONFIG=source.resolve(), SETUP_PLAN=plan.resolve()),
                mock.patch.object(window, "controller_paths", return_value=paths),
            ):
                session = window.holding_session(events)
                changed = window.HoldingSession("other-run", session.api_pid, session.model, session.context,
                                                session.session_started_line, session.server_started_line)
                writer = mock.Mock()
                with self.assertRaisesRegex(window.ControllerError, "run ID changed"):
                    window.request_owned_stop(control, events, changed, writer=writer)
                writer.assert_not_called()

    def test_restoration_retries_two_seconds_and_preserves_errors(self):
        sequence = [
            RuntimeError("transient"),
            {
                "original_running": True, "original_paused": True,
                "original_api_healthy": False, "ports_clear": [],
                "owned_identities_gone": False, "foreign_gpu_process_count": 1,
            },
            {
                "original_running": True, "original_paused": False,
                "original_api_healthy": True, "ports_clear": [18080, 18081],
                "owned_identities_gone": True, "foreign_gpu_process_count": 0,
            },
        ]
        sleeps = []

        def probe():
            item = sequence.pop(0)
            if isinstance(item, Exception):
                raise item
            return item

        result = window.restoration_retry(probe, attempts=3, sleep=sleeps.append)
        self.assertEqual(result["attempt"], 3)
        self.assertEqual(sleeps, [2.0, 2.0])
        self.assertIn("RuntimeError: transient", result["observation_errors"])


if __name__ == "__main__":
    unittest.main()
