import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from runtime import public_model_expansion as expansion
from runtime import specialist_study_trial as ownership


class ExpansionContractTests(unittest.TestCase):
    def test_frozen_catalog_has_all_profiles_and_exact_caps(self):
        catalog = expansion.load_catalog()
        self.assertEqual({row["key"] for row in catalog["profiles"]}, expansion.PROFILE_KEYS)
        for row in catalog["profiles"]:
            self.assertEqual(row["controls"]["context_tokens"], 73728)
            self.assertEqual(row["controls"]["max_output_tokens"], 65536)
            self.assertEqual(row["controls"]["request_timeout_seconds"], 21600)
        self.assertEqual(expansion.endpoint_for_backend("llama"), "http://127.0.0.1:18081/v1")
        self.assertEqual(expansion.endpoint_for_backend("strata"), "http://127.0.0.1:18080/v1")

    def test_profile_binding_refuses_config_or_request_profile_change(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "profile-binding.json"
            original = {
                "manifest_sha256": expansion.MANIFEST_SHA256, "profile_key": "p", "profile_sha256": "a",
                "runtime_backend": "llama", "model_name": "m", "config": {"path": "c", "sha256": "1"},
                "runner": {"path": "r", "sha256": "2"}, "controls": {"context_tokens": 73728},
                "request_options": {"temperature": 0.0},
            }
            expansion.ensure_profile_binding(path, original)
            expansion.ensure_profile_binding(path, original | {"created_at": "later"})
            changed = original | {"request_options": {"temperature": 1.0}}
            with self.assertRaisesRegex(expansion.ExpansionError, "different executed profile"):
                expansion.ensure_profile_binding(path, changed)

    def test_process_signal_requires_pid_startticks_argv_and_session_leader(self):
        process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True)
        try:
            identity = expansion.wait_process_identity(process.pid, sys.executable)
            self.assertTrue(expansion.validate_process_identity(identity))
            tampered = dict(identity, start_ticks=identity["start_ticks"] + 1)
            self.assertFalse(expansion.validate_process_identity(tampered))
            with self.assertRaisesRegex(expansion.ExpansionError, "validation failed"):
                expansion.terminate_owned_process(process, tampered)
            self.assertIsNone(process.poll())
            self.assertTrue(expansion.terminate_owned_process(process, identity))
        finally:
            if process.poll() is None:
                os.killpg(process.pid, 15)
                process.wait(timeout=5)

    def test_failed_smoke_semantics_are_recorded_but_not_a_gate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = root / "smoke.jsonl"
            result.write_text(json.dumps({
                "request": {"output_token_limit": 32},
                "response": {"stream_complete": True, "done_reason": "length"},
                "evaluation": {"passed": False, "detail": "wrong answer"}, "error": None,
            }) + "\n")
            events = root / "events.jsonl"
            events.write_text(json.dumps({"event": "step_complete", "passed": False, "returncode": 1, "result": str(result)}) + "\n")
            config = root / "config.json"
            config.write_text("{}")
            supervisor = object.__new__(expansion.Supervisor)
            session = ownership.HoldingSession("r", "llama", "m", 73728, 1.0, "a", "b", "c")
            profile = {"key": "p", "runtime_backend": "llama", "model_name": "m"}
            result = supervisor._preflight_document(profile, config, session, events, [])
        self.assertTrue(result["health_alias_gate_passed"])
        self.assertTrue(result["transport_smoke_gate_passed"])
        self.assertFalse(result["semantic_answer_used_as_gate"])
        self.assertFalse(result["smoke_semantic_passed"])

    def test_transport_smoke_rejects_api_failure_even_if_session_reaches_step_complete(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = root / "smoke.jsonl"
            result.write_text(json.dumps({
                "request": {"output_token_limit": 32}, "response": None,
                "evaluation": {"passed": False}, "error": {"type": "TimeoutError"},
            }) + "\n")
            events = root / "events.jsonl"
            events.write_text(json.dumps({"event": "step_complete", "passed": False, "returncode": 1, "result": str(result)}) + "\n")
            with self.assertRaisesRegex(expansion.ExpansionError, "terminal 32-token API stream"):
                expansion.transport_smoke_evidence(events)

    def test_external_gate_requires_adoptable_full64_grading_and_clear_gpu(self):
        with tempfile.TemporaryDirectory() as temporary:
            study = Path(temporary)
            gate = study / expansion.OSS_HANDOFF
            gate.parent.mkdir(parents=True)
            value = {"schema_version": 1, "adoptable": True, "full64_official_grading_complete": True, "gpu": {"state": "owned"}}
            gate.write_text(json.dumps(value))
            self.assertFalse(expansion.external_gate(study)[0])
            value["gpu"]["state"] = "clear"
            gate.write_text(json.dumps(value))
            self.assertTrue(expansion.external_gate(study)[0])

    def test_external_wait_surfaces_actual_run_settings_and_live_coverage(self):
        with tempfile.TemporaryDirectory() as temporary:
            study = Path(temporary)
            output = study / "oss-repair"
            output.mkdir()
            gate = output / "handoff.json"
            gate.write_text(json.dumps({
                "schema_version": 1, "adoptable": True,
                "full64_official_grading_complete": True, "gpu": {"state": "clear"},
                "diagnostic_outcomes": {}, "grading_summary": {"status": "placeholder"},
            }))
            settings = {"reasoning_effort": "low", "temperature": 0.0, "top_p": 1.0, "top_k": 0}
            (output / "run.json").write_text(json.dumps({"settings": settings}))
            coverage = {"counts": {"total": 64, "completed": 64}}
            (output / "status.json").write_text(json.dumps(coverage))
            (output / "grading").mkdir()
            grade = {"status": "completed", "coverage": {"total": 64}}
            (output / "grading/summary.json").write_text(json.dumps(grade))
            supervisor = object.__new__(expansion.Supervisor)
            supervisor.study = study
            supervisor.state = {"models": {"oss20b_repaired": {
                "output_dir": str(output), "external_artifact_fingerprint": None,
            }}}
            supervisor.received_signal = None
            supervisor.save = lambda **fields: supervisor.state.update(fields)
            supervisor.event = lambda *_args, **_kwargs: None
            supervisor.trigger_report_update = lambda: None
            supervisor.wait_for_oss({"key": "oss20b_repaired"})
            row = supervisor.state["models"]["oss20b_repaired"]
        self.assertEqual(row["external_run_settings"], settings)
        self.assertEqual(row["run_status"], coverage)
        self.assertEqual(row["grading_summary"], grade)
        self.assertEqual(row["state"], "complete")

    def test_external_terminal_failure_advances_only_after_gpu_is_clear(self):
        with tempfile.TemporaryDirectory() as temporary:
            study = Path(temporary)
            output = study / "oss-repair"
            output.mkdir()
            (output / "handoff.json").write_text(json.dumps({
                "schema_version": 1, "status": "full64_failed", "adoptable": False,
                "full64_official_grading_complete": False, "gpu": {"state": "clear"},
                "error": {"type": "RuntimeError", "message": "fixture"},
            }))
            supervisor = object.__new__(expansion.Supervisor)
            supervisor.study = study
            supervisor.state = {"models": {"oss20b_repaired": {
                "output_dir": str(output), "external_artifact_fingerprint": None,
            }}}
            supervisor.received_signal = None
            supervisor.save = lambda **fields: supervisor.state.update(fields)
            supervisor.event = lambda *_args, **_kwargs: None
            supervisor.trigger_report_update = lambda: None
            with self.assertRaisesRegex(expansion.ExpansionError, "after GPU clear"):
                supervisor.wait_for_oss({"key": "oss20b_repaired"})
            row = supervisor.state["models"]["oss20b_repaired"]
        self.assertEqual(row["state"], "failed")
        self.assertIn("full64_failed", row["error"])

    def test_normal_external_wait_clears_stale_top_level_restart_error(self):
        with tempfile.TemporaryDirectory() as temporary:
            study = Path(temporary)
            output = study / "oss-repair"
            output.mkdir()
            (output / "handoff.json").write_text(json.dumps({
                "schema_version": 1, "status": "full64_running", "adoptable": False,
                "full64_official_grading_complete": False, "gpu": {"state": "owned_scoring"},
            }))
            supervisor = object.__new__(expansion.Supervisor)
            supervisor.study = study
            supervisor.args = SimpleNamespace(external_poll_seconds=30)
            supervisor.state = {
                "status": "interrupted", "error": "signal 15", "finished_at": "old",
                "models": {"oss20b_repaired": {
                    "output_dir": str(output), "external_artifact_fingerprint": None,
                }},
            }
            supervisor.received_signal = None
            supervisor.save = lambda **fields: supervisor.state.update(fields)
            supervisor.event = lambda *_args, **_kwargs: None
            supervisor.trigger_report_update = lambda: None
            with mock.patch.object(expansion.time, "sleep", side_effect=RuntimeError("stop fixture")):
                with self.assertRaisesRegex(RuntimeError, "stop fixture"):
                    supervisor.wait_for_oss({"key": "oss20b_repaired"})
        self.assertEqual(supervisor.state["status"], "waiting_for_oss_handoff")
        self.assertIsNone(supervisor.state["error"])
        self.assertIsNone(supervisor.state["finished_at"])

    def test_grade_coverage_requires_both_official_32_case_suites(self):
        summary = {
            "status": "completed", "coverage": {"total": 64, "completed": 63, "truncated": 1},
            "evalplus": {"status": "completed", "task_count": 32},
            "ifeval": {"status": "completed", "strict": {"prompt_count": 32}, "loose": {"prompt_count": 32}},
        }
        expansion.validate_grade_coverage(summary)
        summary["ifeval"]["strict"]["prompt_count"] = 31
        with self.assertRaisesRegex(expansion.ExpansionError, "does not cover"):
            expansion.validate_grade_coverage(summary)

    def test_grader_mounts_frozen_manifest_and_only_profile_output_is_writable(self):
        study = Path("/tmp/expansion")
        output = study / "profiles/model"
        manifest = {"sources": {"evalplus": {"root": "/public/evalplus"}, "ifeval": {"root": "/public/google"}}}
        command = expansion.grade_command("image@sha256:test", study, output, "model", manifest, Path("/tmp/runner.py"))
        mounts = [command[index + 1] for index, value in enumerate(command) if value == "--mount"]
        self.assertIn(f"type=bind,src={expansion.FROZEN_STUDY},dst={expansion.FROZEN_STUDY},readonly", mounts)
        self.assertEqual([value for value in mounts if not value.endswith(",readonly")], [f"type=bind,src={output},dst={output}"])
        self.assertEqual(command[command.index("--network") + 1], "none")

    def test_documented_profile_failure_continues_to_later_profile(self):
        supervisor = object.__new__(expansion.Supervisor)
        supervisor.args = SimpleNamespace(grade_image="image")
        supervisor.catalog = {"profiles": [
            {"key": "first", "managed_externally": False},
            {"key": "second", "managed_externally": False},
        ]}
        supervisor.state = {
            "models": {
                "first": {"output_dir": "/tmp/nonexistent-first", "state": "waiting"},
                "second": {"output_dir": "/tmp/nonexistent-second", "state": "waiting"},
            }
        }
        supervisor.received_signal = None
        supervisor.catalog_sha256 = "catalog"
        seen = []

        def run_profile(profile):
            seen.append(profile["key"])
            if profile["key"] == "first":
                raise expansion.ExpansionError("documented load failure")
            supervisor.state["models"]["second"]["state"] = "complete"

        supervisor.run_profile = run_profile
        supervisor.prepare_artifacts = lambda: None
        supervisor.record_controls = lambda: None
        supervisor.save = lambda **fields: supervisor.state.update(fields)
        supervisor.event = lambda *_args, **_kwargs: None
        supervisor.trigger_report_update = lambda: None
        with mock.patch.object(expansion.signal, "signal"), mock.patch.object(
            expansion, "process_identity", return_value={"pid": 1}
        ):
            code = supervisor.run()
        self.assertEqual(code, 1)
        self.assertEqual(seen, ["first", "second"])
        self.assertEqual(supervisor.state["models"]["first"]["state"], "failed")
        self.assertEqual(supervisor.state["models"]["second"]["state"], "complete")
        self.assertEqual(supervisor.state["status"], "complete_with_profile_failures")

    def test_safety_stop_does_not_advance_queue(self):
        supervisor = object.__new__(expansion.Supervisor)
        supervisor.args = SimpleNamespace(grade_image="image")
        supervisor.catalog = {"profiles": [
            {"key": "first", "managed_externally": False},
            {"key": "second", "managed_externally": False},
        ]}
        supervisor.state = {"models": {
            "first": {"output_dir": "/tmp/a", "state": "waiting"},
            "second": {"output_dir": "/tmp/b", "state": "waiting"},
        }}
        supervisor.received_signal = None
        supervisor.catalog_sha256 = "catalog"
        seen = []
        def run_profile(profile):
            seen.append(profile["key"])
            raise expansion.SafetyStop("restoration unsafe")
        supervisor.run_profile = run_profile
        supervisor.prepare_artifacts = lambda: None
        supervisor.record_controls = lambda: None
        supervisor.save = lambda **fields: supervisor.state.update(fields)
        supervisor.event = lambda *_args, **_kwargs: None
        supervisor.trigger_report_update = lambda: None
        with mock.patch.object(expansion.signal, "signal"), mock.patch.object(
            expansion, "process_identity", return_value={"pid": 1}
        ):
            code = supervisor.run()
        self.assertEqual(code, 1)
        self.assertEqual(seen, ["first"])
        self.assertEqual(supervisor.state["status"], "failed")


if __name__ == "__main__":
    unittest.main()
