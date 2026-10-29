from __future__ import annotations

import unittest

from runtime import oss20b_public_retry_full as full


class FullRetrySupervisorTests(unittest.TestCase):
    def test_full64_command_has_exact_contract_and_no_budget_override(self) -> None:
        command = full.runner_command("python", full.SELECTED_PROFILE)
        expected = {
            "--max-tokens": "65536",
            "--request-timeout": "21600",
            "--seed": "20261023",
            "--reasoning-effort": "low",
            "--temperature": "0.0",
            "--top-p": "1.0",
            "--top-k": "0",
            "--min-p": "0.0",
            "--repeat-penalty": "1.0",
        }
        for flag, value in expected.items():
            self.assertEqual(command[command.index(flag) + 1], value)
        self.assertNotIn("--reasoning-budget-tokens", command)
        self.assertNotIn("--thinking-budget-tokens", command)
        self.assertIn("--rerun-failed", command)

    def test_profile_binding_rejects_any_selected_option_change(self) -> None:
        handoff = {"profile_selected": dict(full.SELECTED_PROFILE)}
        self.assertEqual(full.validate_selected_profile(handoff), full.SELECTED_PROFILE)
        handoff["profile_selected"]["top_k"] = 1
        with self.assertRaisesRegex(full.FullRunError, "does not match"):
            full.validate_selected_profile(handoff)

    def test_final_gate_requires_completed_grading_and_clear_gpu(self) -> None:
        value = full.final_handoff(
            {"gpu": {"state": "owned_scoring"}},
            full.SELECTED_PROFILE,
            {"status": "completed", "coverage": {"total": 64}},
            {"server_process_gone": True},
        )
        self.assertTrue(value["adoptable"])
        self.assertTrue(value["full64_official_grading_complete"])
        self.assertEqual(value["gpu"]["state"], "clear")


if __name__ == "__main__":
    unittest.main()
