from __future__ import annotations

import copy
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from runtime import flash_modes_sampling_parent_stage as stage
from runtime import flash_modes_sampling_window as controller


def fake_process_identity(
    *, pid=None, parent_pid=10, start_ticks="20", state="S", exe="/usr/bin/python3",
):
    return stage.ProcessIdentity(
        os.getpid() if pid is None else pid, parent_pid, start_ticks, state, exe,
    )


def fake_owner(*, start_ticks=stage.MAIN_OWNER_START_TICKS, state="S"):
    return stage.ProcessIdentity(
        stage.MAIN_OWNER_PID, 1, start_ticks, state, "/usr/bin/python3",
    )


def fake_evidence(*, ready_at="2026-10-21T02:00:00Z"):
    artifacts = {
        name: {
            "source": {"path": f"/tmp/{name}.jsonl", "sha256": str(index) * 64, "size": index},
            "rows": [
                {
                    "case_id": case_id, "source_line": line,
                    "source_raw_line_sha256": "a" * 64,
                    "source_row_canonical_sha256": "b" * 64,
                }
                for line, case_id in enumerate(cases, 1)
            ],
        }
        for index, (name, cases) in enumerate(controller.MAIN_CASES.items(), 1)
    }
    rows = {name: tuple() for name in controller.MAIN_CASES}
    return stage.MainEvidence(
        ready=controller.Binding(Path("/tmp/ready.json"), "c" * 64, 1),
        study=controller.Binding(Path("/tmp/study.json"), "d" * 64, 1),
        ready_document={"kind": "model_boundary_ready"},
        study_document={"session": {"run_id": stage.MAIN_RUN_ID}},
        ready_published_at=datetime.fromisoformat(ready_at.replace("Z", "+00:00")),
        artifacts=artifacts,
        rows=rows,
    )


def restoration_attestation(observed_at="2026-10-21T02:01:00Z"):
    return {
        "kind": "flash_modes_sampling_parent_restoration_attestation",
        "observed_at": observed_at,
        "original_running": True,
        "original_paused": False,
        "original_api_healthy": True,
        "ports_clear": {"18080": True, "18081": True},
        "no_foreign_gpu_process": True,
        "release_absent": True,
    }


def temp_stage_files(root: Path):
    return {
        "claim": root / "claim.json",
        "events": root / "events.jsonl",
        "receipt": root / "receipt.json",
    }


class StaticAndOfflineTests(unittest.TestCase):
    def test_code_review_requires_affirmative_exact_source_bindings(self):
        expected = {
            "controller": {"path": "/c", "sha256": "a" * 64, "size": 1},
            "tests": {"path": "/t", "sha256": "b" * 64, "size": 2},
            "preflight": {"path": "/p", "sha256": "c" * 64, "size": 3},
            "plan": {"path": "/q", "sha256": "d" * 64, "size": 4},
        }
        review = {
            "schema_version": 2,
            "kind": "flash_modes_sampling_parent_code_review",
            "code_review_accepted": True,
            "actual_new_window_executed": False,
            "execution_acceptance_created": False,
            "source_pins": copy.deepcopy(expected),
            "pending_actual_execution_evidence": {
                "execution_acceptance_exists": False,
                "execution_acceptance_path": str(stage.ACCEPTANCE_PATH),
                "main_ready_exists": False,
                "main_ready_path": str(controller.MAIN_READY),
                "main_release_exists": False,
                "main_release_path": str(controller.MAIN_RELEASE),
            },
        }
        stage.validate_code_review_document(review, expected)
        review["code_review_accepted"] = False
        with self.assertRaisesRegex(stage.StageError, "not affirmative"):
            stage.validate_code_review_document(review, expected)
        review["code_review_accepted"] = True
        review["source_pins"]["controller"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(stage.StageError, "controller binding differs"):
            stage.validate_code_review_document(review, expected)

    def test_check_only_never_calls_live_process_gpu_http_docker_or_popen(self):
        prerequisites = {"code_review_accepted": True}
        bomb = mock.Mock(side_effect=AssertionError("live operation called"))
        with (
            mock.patch.object(stage, "validate_prerequisites", return_value=prerequisites),
            mock.patch.object(stage.os.path, "lexists", return_value=False),
            mock.patch.object(controller, "live_preflight", bomb),
            mock.patch.object(stage, "read_process_identity", bomb),
            mock.patch.object(stage.subprocess, "Popen", bomb),
            mock.patch("sys.stdout", new_callable=io.StringIO) as output,
        ):
            self.assertEqual(stage.main(["--check-only"]), 0)
        result = json.loads(output.getvalue())
        self.assertEqual(result["status"], "waiting_for_actual_main_evidence")
        self.assertFalse(result["controller_launch_performed"])
        bomb.assert_not_called()

    def test_direct_path_import_works_without_pythonpath(self):
        source = Path(stage.__file__).resolve()
        program = f"""
import importlib.util, sys
spec=importlib.util.spec_from_file_location('parent_stage_direct', {str(source)!r})
module=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=module
spec.loader.exec_module(module)
assert module.controller.TAG == 'specialists-flash-modes-sampling-72k64k-v2'
"""
        environment = dict(os.environ)
        environment.pop("PYTHONPATH", None)
        environment.pop("PYTHONHOME", None)
        with tempfile.TemporaryDirectory() as tmp:
            completed = subprocess.run(
                [sys.executable, "-I", "-c", program], cwd=tmp, env=environment,
                text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                timeout=10, check=False,
            )
        self.assertEqual(completed.returncode, 0, completed.stderr)


class WaitAndEvidenceTests(unittest.TestCase):
    def test_absent_ready_times_out_without_live_restoration_queries(self):
        clock = iter((0.0, 0.0, 2.0, 2.0, 4.0))
        live = mock.Mock(side_effect=AssertionError("live restoration queried"))
        with mock.patch.object(controller, "live_preflight", live):
            with self.assertRaisesRegex(stage.StageError, "bounded wait expired"):
                stage.wait_for_boundary_ready(
                    signals=stage.SignalLatch(), max_wait_seconds=3.0,
                    owner_reader=lambda pid: fake_owner(),
                    exists=lambda path: False,
                    collision_probe=lambda: (),
                    monotonic=lambda: next(clock), sleep=lambda seconds: None,
                )
        live.assert_not_called()

    def test_owner_reuse_before_ready_fails_closed(self):
        with self.assertRaisesRegex(stage.StageError, "gone, reused"):
            stage.wait_for_boundary_ready(
                signals=stage.SignalLatch(), max_wait_seconds=10,
                owner_reader=lambda pid: fake_owner(start_ticks="999"),
                exists=lambda path: False, collision_probe=lambda: (),
            )

    def test_terminal_semantic_wrong_or_truncated_row_is_valid_evidence(self):
        value = {
            "case_id": controller.CODE_CASES[0],
            "response": {"stream_complete": True, "truncated": True},
            "evaluation": {"passed": False}, "error": None,
        }
        row = controller.JsonlRow(1, value, "a" * 64, "b" * 64)
        pin = stage._row_pin(row, controller.CODE_CASES[0])
        self.assertEqual(pin["case_id"], controller.CODE_CASES[0])
        value["response"]["stream_complete"] = False
        with self.assertRaisesRegex(stage.StageError, "not terminal"):
            stage._row_pin(row, controller.CODE_CASES[0])

    def test_incomplete_ready_stops_before_restoration_acceptance_or_launch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = temp_stage_files(root)
            preflight = root / "preflight.json"
            preflight.write_text("{}\n")
            acceptance = root / "acceptance.json"
            restoration = mock.Mock(side_effect=AssertionError("restoration queried"))
            launch = mock.Mock(side_effect=AssertionError("controller launched"))
            with mock.patch.object(stage, "STAGE_PREFLIGHT", preflight):
                receipt, code = stage.execute_stage(
                    signals=stage.SignalLatch(), paths=paths, acceptance_path=acceptance,
                    prerequisite_fn=lambda: {"ok": True}, unused_fn=lambda value: None,
                    wait_fn=lambda **kwargs: {"ready_observed_at": "now"},
                    evidence_fn=lambda: (_ for _ in ()).throw(stage.StageError("incomplete READY")),
                    restoration_fn=restoration,
                    launch_fn=launch,
                    stage_identity_reader=lambda pid: fake_process_identity(pid=pid),
                )
            self.assertEqual(code, 1)
            self.assertEqual(receipt["status"], "stage_failed")
            self.assertFalse(acceptance.exists())
            restoration.assert_not_called()
            launch.assert_not_called()


class AcceptanceTests(unittest.TestCase):
    def assembled(self):
        evidence = fake_evidence()
        document = stage.build_acceptance(
            evidence, restoration_attestation(), created_at="2026-10-21T02:02:00Z",
        )
        return evidence, document

    def test_acceptance_shape_rejects_wrong_run_hash_and_schema(self):
        evidence, document = self.assembled()
        with mock.patch.object(stage.os.path, "lexists", return_value=False):
            stage.validate_acceptance_shape(document, evidence)
            wrong_run = copy.deepcopy(document)
            wrong_run["main"]["model_session_run_id"] = "other"
            with self.assertRaisesRegex(stage.StageError, "identity differs"):
                stage.validate_acceptance_shape(wrong_run, evidence)
            wrong_hash = copy.deepcopy(document)
            wrong_hash["main"]["ready"]["sha256"] = "0" * 64
            with self.assertRaisesRegex(stage.StageError, "identity differs"):
                stage.validate_acceptance_shape(wrong_hash, evidence)
            wrong_schema = copy.deepcopy(document)
            wrong_schema["extra"] = True
            with self.assertRaisesRegex(stage.StageError, "top-level schema"):
                stage.validate_acceptance_shape(wrong_schema, evidence)

    def test_acceptance_chronology_and_restoration_are_strict(self):
        evidence, document = self.assembled()
        with mock.patch.object(stage.os.path, "lexists", return_value=False):
            early = copy.deepcopy(document)
            early["main"]["restoration_attestation"]["observed_at"] = "2026-10-21T01:59:00Z"
            with self.assertRaisesRegex(stage.StageError, "chronology"):
                stage.validate_acceptance_shape(early, evidence)
            unhealthy = copy.deepcopy(document)
            unhealthy["main"]["restoration_attestation"]["original_paused"] = True
            with self.assertRaises(controller.ControllerError):
                stage.validate_acceptance_shape(unhealthy, evidence)

    def test_restoration_retries_and_requires_old_api_gpu_identities_gone(self):
        live = {
            "original_running": True, "original_paused": False,
            "original_api_healthy": True, "ports_clear": [18080, 18081],
            "gpu_compute_process_count": 0,
        }
        calls = {stage.MAIN_API_PID: 0, stage.MAIN_GPU_PID: 0}

        def identity(pid):
            calls[pid] += 1
            if calls[pid] == 1:
                start = stage.MAIN_API_START_TICKS if pid == stage.MAIN_API_PID else stage.MAIN_GPU_START_TICKS
                return stage.ProcessIdentity(pid, 1, start, "S", "/owned")
            raise stage.ProcessGone("gone")

        sleeps = []
        with mock.patch.object(stage.os.path, "lexists", return_value=False):
            attestation, observation = stage.observe_restoration(
                live_probe=lambda: live, identity_reader=identity,
                attempts=2, sleep=sleeps.append,
            )
        self.assertEqual(observation["attempt"], 2)
        self.assertEqual(sleeps, [2.0])
        self.assertTrue(attestation["release_absent"])


class SingleUseAndLaunchTests(unittest.TestCase):
    def execute_with_outcome(self, root: Path, outcome):
        paths = temp_stage_files(root)
        preflight = root / "preflight.json"
        preflight.write_text("{}\n")
        acceptance = root / "acceptance.json"
        evidence = fake_evidence()
        order = []

        def prevalidate(document, observed):
            self.assertIs(observed, evidence)
            order.append("prevalidate")

        def validate(path):
            self.assertTrue(path.exists())
            order.append("validate")
            return None

        def launch(binding, **kwargs):
            self.assertEqual(binding.path, acceptance.resolve())
            order.append("launch")
            return dict(outcome)

        with (
            mock.patch.object(stage, "STAGE_PREFLIGHT", preflight),
            mock.patch.object(stage.os.path, "lexists", return_value=False),
        ):
            receipt, code = stage.execute_stage(
                signals=stage.SignalLatch(), paths=paths, acceptance_path=acceptance,
                prerequisite_fn=lambda: {"ok": True}, unused_fn=lambda value: None,
                wait_fn=lambda **kwargs: {"ready_observed_at": "now"},
                evidence_fn=lambda: evidence,
                restoration_fn=lambda: (restoration_attestation(), {"attempt": 1}),
                prevalidate_fn=prevalidate,
                acceptance_validate_fn=validate,
                launch_fn=launch,
                stage_identity_reader=lambda pid: fake_process_identity(pid=pid),
            )
        return receipt, code, order, paths, acceptance

    def test_launch_occurs_once_only_after_prevalidation_write_and_full_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            receipt, code, order, paths, acceptance = self.execute_with_outcome(
                Path(tmp), {"stage_status": "complete", "stage_interrupted_signal": None},
            )
            self.assertEqual(code, 0)
            self.assertEqual(order, ["prevalidate", "validate", "launch"])
            self.assertTrue(acceptance.exists())
            self.assertTrue(paths["receipt"].exists())
            self.assertTrue(receipt["claims"]["controller_launched_at_most_once"])
            with self.assertRaisesRegex(stage.StageError, "single-use"):
                with mock.patch.object(stage, "candidate_collisions", return_value=[]):
                    stage.assert_single_use_unused(paths, acceptance_path=acceptance)

    def test_failed_prevalidation_writes_no_acceptance_and_never_launches(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = temp_stage_files(root)
            preflight = root / "preflight.json"
            preflight.write_text("{}\n")
            acceptance = root / "acceptance.json"
            launch = mock.Mock(side_effect=AssertionError("controller launched"))
            with (
                mock.patch.object(stage, "STAGE_PREFLIGHT", preflight),
                mock.patch.object(stage.os.path, "lexists", return_value=False),
            ):
                receipt, code = stage.execute_stage(
                    signals=stage.SignalLatch(), paths=paths, acceptance_path=acceptance,
                    prerequisite_fn=lambda: {"ok": True}, unused_fn=lambda value: None,
                    wait_fn=lambda **kwargs: {"ready_observed_at": "now"},
                    evidence_fn=fake_evidence,
                    restoration_fn=lambda: (restoration_attestation(), {"attempt": 1}),
                    prevalidate_fn=lambda document, evidence: (_ for _ in ()).throw(
                        stage.StageError("acceptance mismatch")
                    ),
                    launch_fn=launch,
                    stage_identity_reader=lambda pid: fake_process_identity(pid=pid),
                )
            self.assertEqual(code, 1)
            self.assertEqual(receipt["status"], "stage_failed")
            self.assertFalse(acceptance.exists())
            launch.assert_not_called()

    def test_foreign_candidate_output_after_acceptance_prevents_launch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = temp_stage_files(root)
            preflight = root / "preflight.json"
            preflight.write_text("{}\n")
            acceptance = root / "acceptance.json"
            launch = mock.Mock(side_effect=AssertionError("controller launched"))
            probes = iter(((), (root / "foreign-candidate-output.json",)))
            with (
                mock.patch.object(stage, "STAGE_PREFLIGHT", preflight),
                mock.patch.object(stage.os.path, "lexists", return_value=False),
            ):
                receipt, code = stage.execute_stage(
                    signals=stage.SignalLatch(), paths=paths, acceptance_path=acceptance,
                    prerequisite_fn=lambda: {"ok": True}, unused_fn=lambda value: None,
                    wait_fn=lambda **kwargs: {"ready_observed_at": "now"},
                    evidence_fn=fake_evidence,
                    restoration_fn=lambda: (restoration_attestation(), {"attempt": 1}),
                    prevalidate_fn=lambda document, evidence: None,
                    acceptance_validate_fn=lambda path: None,
                    launch_fn=launch,
                    stage_identity_reader=lambda pid: fake_process_identity(pid=pid),
                    collision_fn=lambda path, allowed: next(probes),
                )
            self.assertEqual(code, 1)
            self.assertEqual(receipt["status"], "stage_failed")
            self.assertTrue(acceptance.exists())
            launch.assert_not_called()

    def test_pending_and_failure_controller_statuses_propagate_without_release(self):
        cases = (
            ({
                "stage_status": "selected_work_pending_continuation",
                "stage_interrupted_signal": None,
                "pending_selected_work": [controller.CODE_CASES[0]],
            }, 0),
            ({
                "stage_status": "cleanup_failed", "stage_interrupted_signal": None,
                "primary_problem": {"error": "runtime"},
                "cleanup_problem": {"error": "restore"},
            }, 1),
        )
        for index, (outcome, expected_code) in enumerate(cases):
            with self.subTest(outcome=outcome), tempfile.TemporaryDirectory() as tmp:
                receipt, code, _, _, _ = self.execute_with_outcome(Path(tmp), outcome)
                self.assertEqual(code, expected_code)
                self.assertEqual(receipt["status"], outcome["stage_status"])
                self.assertFalse(receipt["claims"]["main_release_created"])

    def test_actual_controller_receipt_summary_preserves_pending_and_failure_reasons(self):
        acceptance = controller.Binding(Path("/tmp/acceptance.json"), "a" * 64, 10)
        binding = controller.Binding(Path("/tmp/controller-receipt.json"), "b" * 64, 20)
        base = {
            "schema_version": 1,
            "kind": "flash_modes_sampling_controller_execution",
            "tag": controller.TAG,
            "acceptance": acceptance.as_dict(),
            "claims": {"main_release_published": False},
            "pending_selected_work": [controller.CODE_CASES[0]],
            "primary_problem": None,
            "cleanup_problem": None,
        }
        pending = {**base, "status": "selected_work_pending_continuation"}
        summary = stage._controller_receipt_summary(0, pending, binding, acceptance)
        self.assertEqual(summary["stage_status"], "selected_work_pending_continuation")
        self.assertEqual(summary["pending_selected_work"], [controller.CODE_CASES[0]])

        failed = {
            **base, "status": "cleanup_failed",
            "primary_problem": {"error": "runtime"},
            "cleanup_problem": {"error": "restore"},
        }
        summary = stage._controller_receipt_summary(1, failed, binding, acceptance)
        self.assertEqual(summary["stage_status"], "cleanup_failed")
        self.assertEqual(summary["primary_problem"], {"error": "runtime"})
        self.assertEqual(summary["cleanup_problem"], {"error": "restore"})

    def test_duplicate_stage_and_candidate_acceptance_are_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = temp_stage_files(root)
            paths["claim"].write_text("{}\n")
            with mock.patch.object(stage, "candidate_collisions", return_value=[]):
                with self.assertRaisesRegex(stage.StageError, "single-use"):
                    stage.assert_single_use_unused(paths, acceptance_path=root / "acceptance.json")
            paths["claim"].unlink()
            acceptance = root / "acceptance.json"
            acceptance.write_text("{}\n")
            with mock.patch.object(stage, "candidate_collisions", return_value=[acceptance]):
                with self.assertRaisesRegex(stage.StageError, "already used"):
                    stage.assert_single_use_unused(paths, acceptance_path=acceptance)


class InterruptTests(unittest.TestCase):
    class FakeProcess:
        def __init__(self, pid=123):
            self.pid = pid
            self.signals = []

        def poll(self):
            return None

        def send_signal(self, value):
            self.signals.append(value)

    def test_interrupt_signals_only_exact_owned_child_identity(self):
        process = self.FakeProcess()
        expected = stage.ProcessIdentity(123, os.getpid(), "10", "S", "/usr/bin/python3")
        changed = stage.ProcessIdentity(123, os.getpid(), "11", "S", "/usr/bin/python3")
        refused = stage.cooperative_interrupt_child(
            process, expected, identity_reader=lambda pid: changed,
        )
        self.assertFalse(refused["signal_sent"])
        self.assertEqual(process.signals, [])
        allowed = stage.cooperative_interrupt_child(
            process, expected,
            identity_reader=lambda pid: stage.ProcessIdentity(123, os.getpid(), "10", "R", "/usr/bin/python3"),
        )
        self.assertTrue(allowed["signal_sent"])
        self.assertEqual(process.signals, [stage.signal.SIGTERM])


if __name__ == "__main__":
    unittest.main()
