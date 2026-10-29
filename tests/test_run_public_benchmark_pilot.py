import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from runtime import run_public_benchmark_pilot as controller


class PublicPilotOwnershipTests(unittest.TestCase):
    def test_tampered_manifest_is_rejected_before_any_model_operation(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            data = json.dumps({"selected_count_per_suite": 32, "cases": [{}] * 64}).encode()
            (p / "manifest.json").write_bytes(data)
            (p / "manifest.sha256").write_text(hashlib.sha256(data).hexdigest())
            self.assertEqual(len(controller.verify_manifest(p)["cases"]), 64)
            (p / "manifest.json").write_bytes(data + b" ")
            with self.assertRaisesRegex(RuntimeError, "SHA256 mismatch"):
                controller.verify_manifest(p)

    def test_runner_failure_still_releases_owned_session_and_checks_restoration(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            (p / "results").mkdir()
            study = p / "study"
            study.mkdir()
            (study / "manifest.json").write_text('{}')
            config = p / "config.json"
            config.write_text('{}')
            args = SimpleNamespace(study_dir=study, python="python", grade_image="test")
            obj = controller.Controller(args, {})
            outer = mock.Mock(pid=101, returncode=0)
            outer.poll.return_value = 0
            child = mock.Mock(pid=102, returncode=1)
            child.poll.return_value = 1
            identity = controller.ownership.HoldingSession("run-1", "llama", "alias", 73728, 0.1, "a", "b", "c")
            restored = {"original_running": True, "original_paused": False, "owned_loopback_port_clear": True}
            with mock.patch.object(controller, "ROOT", p), \
                 mock.patch.object(controller.llama_trial, "load_trial_config", return_value=SimpleNamespace(model_name="alias", context=73728)), \
                 mock.patch.object(controller.subprocess, "Popen", side_effect=[outer, child]), \
                 mock.patch.object(controller.ownership, "wait_for_holding", return_value=identity), \
                 mock.patch.object(controller.ownership, "request_session_stop", return_value=True) as stop, \
                 mock.patch.object(controller.ownership, "wait_for_outer_exit", return_value=(0, True)) as wait, \
                 mock.patch.object(controller.ownership, "restoration_evidence", return_value=restored) as evidence:
                with self.assertRaisesRegex(RuntimeError, "runner exited 1"):
                    obj.run_model({"key": "test", "config": config})
            stop.assert_called_once()
            wait.assert_called_once()
            evidence.assert_called_once_with("llama")
            self.assertEqual(obj.state["models"]["test"]["status"], "failed")
            self.assertTrue(obj.state["models"]["test"]["stop_requested"])

    def test_grading_container_has_no_network_and_only_model_results_writable(self):
        study = Path("/tmp/public-pilot")
        output = study / "oss20b"
        m = {"sources": {"evalplus": {"root": "/public/evalplus"}, "ifeval": {"root": "/public/google"}}}
        cmd = controller.grade_command("image@sha256:test", study, {"key": "oss20b"}, output, m)
        self.assertEqual(cmd[cmd.index("--network") + 1], "none")
        self.assertIn("--read-only", cmd)
        mounts = [cmd[i + 1] for i, x in enumerate(cmd) if x == "--mount"]
        writable = [x for x in mounts if not x.endswith(",readonly")]
        self.assertEqual(writable, [f"type=bind,src={output},dst={output}"])


if __name__ == "__main__":
    unittest.main()
