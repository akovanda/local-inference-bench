import json
import hashlib
import tempfile
import unittest
from pathlib import Path

from runtime import model_boundary_gate as gate


class Clock:
    def __init__(self):
        self.value = 0.0

    def monotonic(self):
        return self.value

    def sleep(self, seconds):
        self.value += seconds


class ModelBoundaryGateTests(unittest.TestCase):
    @staticmethod
    def _study(root):
        path = root / "study.json"
        path.write_text("{}\n", encoding="utf-8")
        return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    def test_waits_without_accepting_until_bound_release_and_safe_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            clock = Clock()
            states = []

            def sleep(seconds):
                ready = root / "run-model-ready.json"
                release = root / "run-model-release.json"
                if ready.exists() and not release.exists():
                    release.write_text(json.dumps({
                        "schema_version": 1,
                        "kind": "model_boundary_release",
                        "run_prefix": "run",
                        "model_key": "model",
                        "ready_sha256": gate.sha256_file(ready),
                    }), encoding="utf-8")
                clock.sleep(seconds)

            def boundary():
                states.append("checked")
                if len(states) == 1:
                    return {
                        "original_running": True,
                        "original_paused": False,
                        "ports_clear": {"18080": False, "18081": True},
                    }
                return {
                    "original_running": True,
                    "original_paused": False,
                    "ports_clear": {"18080": True, "18081": True},
                }

            result = gate.publish_and_wait(
                root, "run", "model", self._study(root), 5.0,
                boundary_probe=boundary, monotonic=clock.monotonic, sleep=sleep,
            )
            self.assertEqual(states, ["checked", "checked"])
            self.assertEqual(result["ready"]["sha256"], gate.sha256_file(root / "run-model-ready.json"))
            self.assertEqual(
                json.loads((root / "run-model-ready.json").read_text())["study"]["sha256"],
                hashlib.sha256((root / "study.json").read_bytes()).hexdigest(),
            )

    def test_mismatched_or_stale_release_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "run-model-release.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(gate.BoundaryGateError, "predates"):
                gate.publish_and_wait(
                    root, "run", "model", self._study(root), 1.0,
                )

    def test_timeout_publishes_ready_but_never_probes_boundary_without_release(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            clock = Clock()
            probes = []
            with self.assertRaisesRegex(gate.BoundaryGateError, "before deadline"):
                gate.publish_and_wait(
                    root, "run", "model", self._study(root), 0.2,
                    boundary_probe=lambda: probes.append(True),
                    monotonic=clock.monotonic, sleep=clock.sleep, poll_seconds=0.1,
                )
            self.assertEqual(probes, [])
            self.assertTrue((root / "run-model-ready.json").is_file())


if __name__ == "__main__":
    unittest.main()
