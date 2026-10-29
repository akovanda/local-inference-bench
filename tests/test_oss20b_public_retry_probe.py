from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "runtime"
sys.path.insert(0, str(RUNTIME))
SPEC = importlib.util.spec_from_file_location(
    "oss20b_public_retry_probe", RUNTIME / "oss20b_public_retry_probe.py"
)
assert SPEC and SPEC.loader
probe = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = probe
SPEC.loader.exec_module(probe)


class RetryProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = probe.load_frozen_manifest()

    def test_failure_selection_is_the_frozen_three(self) -> None:
        cases = probe.select_cases(self.manifest, "failures")
        self.assertEqual(
            [(case.index, case.suite, case.task_id) for case in cases],
            [(37, "ifeval", "1512"), (46, "ifeval", "2464"), (47, "ifeval", "2216")],
        )

    def test_ifeval_payload_is_verbatim_user_only_and_answer_free(self) -> None:
        case = probe.select_cases(self.manifest, "failures")[0]
        payload = probe.build_payload(case, probe.RECIPES["low-full-sampling"])
        self.assertEqual(payload["messages"], [{"role": "user", "content": case.prompt}])
        serialized = str(payload)
        self.assertNotIn("original", serialized)
        self.assertNotIn("kwargs", serialized)
        self.assertNotIn("instruction_id_list", serialized)
        self.assertEqual(
            (payload["reasoning_effort"], payload["temperature"], payload["top_p"], payload["top_k"], payload["min_p"], payload["repeat_penalty"]),
            ("low", 1.0, 1.0, 0, 0.0, 1.0),
        )

    def test_output_cap_never_counts_as_success(self) -> None:
        outcome = probe.completion_outcome(
            {
                "status": "truncated",
                "finish_reason": "length",
                "stream_complete": True,
                "counts": {"completion_tokens": 8192},
            }
        )
        self.assertFalse(outcome["completed_without_cap"])
        self.assertTrue(outcome["cap_is_not_success"])

    def test_completed_terminal_result_parsing(self) -> None:
        outcome = probe.completion_outcome(
            {
                "status": "completed",
                "finish_reason": "stop",
                "stream_complete": True,
                "usage": {"completion_tokens": 42},
                "counts": {"final_characters": 12},
                "timing": {"wall_seconds": 1.25},
            }
        )
        self.assertTrue(outcome["completed_without_cap"])
        self.assertEqual(outcome["usage"]["completion_tokens"], 42)

    def test_fallback_differs_only_in_temperature(self) -> None:
        case = probe.select_cases(self.manifest, "failures")[0]
        sampled = probe.build_payload(case, probe.RECIPES["low-full-sampling"])
        greedy = probe.build_payload(case, probe.RECIPES["low-greedy-fallback"])
        differences = {
            key for key in sampled | greedy if sampled.get(key) != greedy.get(key)
        }
        self.assertEqual(differences, {"temperature"})
        self.assertEqual((sampled["temperature"], greedy["temperature"]), (1.0, 0.0))


if __name__ == "__main__":
    unittest.main()
