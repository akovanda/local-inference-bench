import fcntl
import json
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest

from runtime import public_benchmark_continuation as continuation


def _warmup_report(plan_hash="a" * 64):
    return {
        "status": "completed_with_step_failures", "problem": None,
        "plan": {"sha256": plan_hash, "step_count": 1},
        "steps": [{
            "id": "ready", "returncode": 1, "passed": False, "error": None,
            "controls": {"suite": "quick", "cases": ["quick_generation"],
                         "thinking": "medium", "temperature": 0.0, "seed": 20261023,
                         "output_tokens": 32, "request_timeout": 120.0},
        }],
        "server": {"runtime_backend": "llama", "context": 73728,
                   "cleanup": {"stopped": True, "kill_sent": False, "term_sent": True}},
    }


def _warmup_result():
    return {
        "case_id": "quick_generation", "error": None,
        "evaluation": {"passed": False, "truncated": True,
                       "detail": "response ended because of the length limit"},
        "response": {"done_reason": "length", "truncated": True, "stream_complete": True},
        "request": {"output_token_limit": 32},
    }


def _manifest():
    return {"cases": [
        {"suite": "humanevalplus", "task_id": "HumanEval/0"},
        {"suite": "ifeval", "task_id": "7"},
    ]}


def _write_attempt(output, case, number, status, manifest_hash, settings_hash):
    attempt = (output / "cases" / case["suite"] /
               continuation._task_storage_id(case["task_id"]) / f"attempt-{number:04d}")
    attempt.mkdir(parents=True)
    (attempt / "result.json").write_text(json.dumps({
        "manifest_sha256": manifest_hash, "settings_sha256": settings_hash,
        "status": status, "truncated": status == "truncated",
        "finish_reason": "length" if status == "truncated" else "stop",
    }), encoding="utf-8")


class WarmupAttestationTests(unittest.TestCase):
    def test_accepts_only_exact_unscored_cap_failure(self):
        evidence = continuation.attest_warmup_only_failure(
            _warmup_report(), _warmup_result(), expected_plan_sha256="a" * 64)
        self.assertEqual(evidence["code"], "quick_generation_length_cap_semantic_failure")
        self.assertEqual(evidence["inference_score_effect"], "none")

    def test_rejects_cleanup_failure(self):
        report = _warmup_report(); report["server"]["cleanup"]["stopped"] = False
        with self.assertRaisesRegex(continuation.ContinuationError, "cleanup"):
            continuation.attest_warmup_only_failure(
                report, _warmup_result(), expected_plan_sha256="a" * 64)

    def test_rejects_unknown_failure(self):
        report = _warmup_report(); report["status"] = "infrastructure_failed"
        report["problem"] = {"error_type": "TypeError", "error": "unknown"}
        with self.assertRaisesRegex(continuation.ContinuationError, "clean completed"):
            continuation.attest_warmup_only_failure(
                report, _warmup_result(), expected_plan_sha256="a" * 64)

    def test_accepts_only_exact_owned_24h_timeout_event_chain(self):
        report = _warmup_report()
        report["status"] = "interrupted"
        report["problem"] = {"error_type": "KeyboardInterrupt", "error": "signal 15"}
        rows = [
            {"event": "trial_started", "child_pid": 41, "max_seconds": 86400.0},
            {"event": "trial_timeout", "child_pgid": 41, "max_seconds": 86400.0},
            {"event": "trial_finished", "returncode": 124},
            {"event": "window_complete", "returncode": 124},
        ]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "gpu-window.log"
            path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
            evidence = continuation.attest_owned_24h_timeout(
                report, _warmup_result(), path,
                {"resume_candidates": [{"task_id": "7"}]},
                expected_plan_sha256="a" * 64)
        self.assertEqual(evidence["code"], "gpu_window_exact_86400_second_timeout")

    def test_rejects_arbitrary_signal_as_timeout(self):
        report = _warmup_report()
        report["status"] = "interrupted"
        report["problem"] = {"error_type": "KeyboardInterrupt", "error": "signal 2"}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "gpu-window.log"
            path.write_text(json.dumps({"event": "window_complete", "returncode": 130}) + "\n")
            with self.assertRaisesRegex(continuation.ContinuationError, "owned SIGTERM"):
                continuation.attest_owned_24h_timeout(
                    report, _warmup_result(), path,
                    {"resume_candidates": [{"task_id": "7"}]},
                    expected_plan_sha256="a" * 64)


class ResumeEvidenceTests(unittest.TestCase):
    def test_truncation_untouched_and_only_nonterminal_resumed(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "model"; output.mkdir()
            manifest_hash = "m" * 64
            settings = continuation.expected_settings("alias", "model")
            settings_hash = continuation.canonical_hash(settings)
            (output / "run.json").write_text(json.dumps({
                "manifest": {"sha256": manifest_hash}, "settings": settings,
                "settings_sha256": settings_hash}), encoding="utf-8")
            cases = _manifest()["cases"]
            _write_attempt(output, cases[0], 1, "truncated", manifest_hash, settings_hash)
            _write_attempt(output, cases[1], 1, "cancelled", manifest_hash, settings_hash)
            before = continuation.audit_model_output(_manifest(), manifest_hash, output, settings)
            _write_attempt(output, cases[1], 2, "completed", manifest_hash, settings_hash)
            after = continuation.audit_model_output(_manifest(), manifest_hash, output, settings)
            transition = continuation.verify_resume_transition(before, after)
        self.assertEqual(transition["terminal_untouched"][0]["status"], "truncated")
        self.assertEqual(transition["terminal_untouched"][0]["scoring_attempt"], "attempt-0001")
        self.assertEqual(transition["resumed_nonterminal"][0]["scoring_attempt"], "attempt-0002")
        self.assertTrue(transition["resumed_nonterminal"][0]["not_best_of"])

    def test_no_new_attempt_is_rejected(self):
        value = {"cases": [{"suite": "ifeval", "task_id": "7", "status": "api_failure",
                            "attempt_count": 1, "scoring_attempt": None}]}
        with self.assertRaisesRegex(continuation.ContinuationError, "did not create"):
            continuation.verify_resume_transition(value, value)


class LockAndPreservationTests(unittest.TestCase):
    def test_waits_for_lock_without_process_queries(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "controller.lock"
            held = path.open("a+b"); fcntl.flock(held.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            waits = []
            def release():
                time.sleep(0.08); fcntl.flock(held.fileno(), fcntl.LOCK_UN); held.close()
            thread = threading.Thread(target=release); thread.start()
            acquired = continuation.acquire_controller_lock(
                path, poll_seconds=0.01, on_wait=lambda: waits.append(1))
            thread.join()
            try: self.assertTrue(waits)
            finally:
                fcntl.flock(acquired.fileno(), fcntl.LOCK_UN); acquired.close()

    def test_snapshot_preserves_original_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); source = root / "controller-status.json"
            destination = root / "evidence/controller-status.json"
            original = b'{"status":"failed","evidence":"exact"}\n'; source.write_bytes(original)
            receipt = continuation.copy_exclusive(source, destination)
            self.assertEqual(source.read_bytes(), original)
            self.assertEqual(destination.read_bytes(), original)
            self.assertEqual(receipt["sha256"], continuation.digest(source))
            with self.assertRaises(FileExistsError): continuation.copy_exclusive(source, destination)

    def test_waiting_status_is_separate_from_original_status(self):
        with tempfile.TemporaryDirectory() as temporary:
            study = Path(temporary); original = study / "controller-status.json"
            original.write_text('{"status":"running"}\n', encoding="utf-8")
            (study / "manifest.json").write_text('{}\n', encoding="utf-8")
            supervisor = continuation.Supervisor(SimpleNamespace(study_dir=study), {})
            supervisor.waiting_update()
            self.assertEqual(json.loads(original.read_text())["status"], "running")
            current = json.loads((study / "continuation-status.json").read_text())
            self.assertEqual(current["status"], "waiting_for_original_controller")


class CommandIsolationTests(unittest.TestCase):
    def test_grade_is_offline_and_only_output_writable(self):
        study = Path("/tmp/study"); output = study / "oss20b"
        manifest = {"sources": {"evalplus": {"root": "/vendor/evalplus"},
                                "ifeval": {"root": "/vendor/google"}}}
        command = continuation.grade_command(
            "sha256:image", Path("/tmp/runner.py"), study, output,
            "oss20b", manifest, "id")
        self.assertEqual(command[command.index("--network") + 1], "none")
        mounts = [command[i + 1] for i, value in enumerate(command) if value == "--mount"]
        self.assertEqual([value for value in mounts if not value.endswith(",readonly")],
                         [f"type=bind,src={output},dst={output}"])
        self.assertIn("/vendor/google/instruction_following_eval", command)


if __name__ == "__main__":
    unittest.main()
