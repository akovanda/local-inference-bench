import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from runtime import run_specialist_followups as followups


def _write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(
    root: Path,
    *,
    study_mutator=None,
    cohort_mutator=None,
    deferral_mutator=None,
):
    study_path = root / "interrupted-study.json"
    study = {
        "schema_version": 1,
        "runtime_backend": "llama",
        "controls": {"profile": "gemma4-style12-q5_k_m"},
        "status": "interrupted",
        "problem": {"error_type": "StudyInterrupted", "error": "signal 15", "signal": 15},
        "cleanup": {
            "outer_exited": True,
            "orchestrator_sigkill_sent": False,
            "model_session_cleanup": {"stopped": True, "kill_sent": False},
        },
        "restoration": {
            "original_running": True,
            "original_paused": False,
            "owned_loopback_port": 18081,
            "owned_loopback_port_clear": True,
        },
    }
    if study_mutator is not None:
        study_mutator(study)
    _write_json(study_path, study)

    cohort_path = root / "first-wave.json"
    cohort = {
        "schema_version": 1,
        "status": "infrastructure_failed",
        "models": [{
            "key": "gemma4-style12-q5_k_m",
            "strict_status": "interrupted",
            "strict_receipt": str(study_path),
            "strict_receipt_sha256": _sha(study_path),
        }],
    }
    if cohort_mutator is not None:
        cohort_mutator(cohort)
    _write_json(cohort_path, cohort)

    deferral_path = root / "user-deferral.json"
    deferral = {
        "schema_version": 1,
        "kind": "user_requested_model_deferral",
        "deferred_models": ["gemma4-style12-q5_k_m"],
        "user_instruction": "Skip Gemma for now and revisit it later.",
        "cohort_receipt": {"path": str(cohort_path), "sha256": _sha(cohort_path)},
        "study_receipt": {"path": str(study_path), "sha256": _sha(study_path)},
        "owned_parent": {"pid": 12345, "start_ticks": 67890},
    }
    if deferral_mutator is not None:
        deferral_mutator(deferral)
    _write_json(deferral_path, deferral)
    return deferral_path, cohort_path, study_path


def _validate(deferral_path: Path, cohort_path: Path, **overrides):
    options = {
        "port_probe": lambda port: True,
        "docker_state_probe": lambda: {"Running": True, "Paused": False},
        "pid_probe": lambda pid: None,
    }
    options.update(overrides)
    return followups.validate_user_deferral(
        deferral_path, _sha(deferral_path), cohort_path, **options
    )


class UserDeferralGateTests(unittest.TestCase):
    def test_valid_deferral_proves_interruption_cleanup_and_fresh_restoration(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = _fixture(Path(directory))
            before = [path.read_bytes() for path in paths]
            evidence = _validate(*paths[:2], pid_probe=lambda pid: 99999)
            self.assertEqual(evidence["gate_kind"], "user_requested_model_deferral")
            self.assertIs(evidence["gemma_primary_complete"], False)
            self.assertEqual(evidence["source_cohort_status"], "infrastructure_failed")
            self.assertEqual(evidence["deferred_models"], ["gemma4-style12-q5_k_m"])
            self.assertEqual(evidence["interrupted_study_proof"]["signal"], 15)
            self.assertTrue(evidence["owned_parent"]["old_identity_exited"])
            self.assertEqual([path.read_bytes() for path in paths], before)

    def test_changed_deferral_sha_and_receipt_path_bindings_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            deferral, cohort, _ = _fixture(Path(directory))
            with self.assertRaises(followups.FollowupError):
                followups.validate_user_deferral(
                    deferral,
                    "0" * 64,
                    cohort,
                    port_probe=lambda port: True,
                    docker_state_probe=lambda: {"Running": True, "Paused": False},
                    pid_probe=lambda pid: None,
                )

        mutators = (
            lambda record: record["cohort_receipt"].update(path="/wrong/cohort.json"),
            lambda record: record["study_receipt"].update(path="/wrong/study.json"),
        )
        for mutate in mutators:
            with self.subTest(mutate=mutate), tempfile.TemporaryDirectory() as directory:
                deferral, cohort, _ = _fixture(Path(directory), deferral_mutator=mutate)
                with self.assertRaises(followups.FollowupError):
                    _validate(deferral, cohort)

    def test_wrong_model_or_study_status_and_incomplete_cleanup_are_rejected(self):
        mutations = (
            {"cohort_mutator": lambda value: value["models"][0].update(key="other")},
            {
                "cohort_mutator": lambda value: value["models"][0].update(
                    strict_status="complete"
                )
            },
            {"study_mutator": lambda value: value.update(status="complete")},
            {
                "study_mutator": lambda value: value["cleanup"].update(
                    outer_exited=False
                )
            },
            {
                "study_mutator": lambda value: value["cleanup"][
                    "model_session_cleanup"
                ].update(kill_sent=True)
            },
        )
        for mutation in mutations:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                deferral, cohort, _ = _fixture(Path(directory), **mutation)
                with self.assertRaises(followups.FollowupError):
                    _validate(deferral, cohort)

    def test_live_parent_paused_container_and_occupied_port_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            deferral, cohort, _ = _fixture(Path(directory))
            probes = (
                {"pid_probe": lambda pid: 67890},
                {"docker_state_probe": lambda: {"Running": True, "Paused": True}},
                {"port_probe": lambda port: port != 18080},
            )
            for probe in probes:
                with self.subTest(probe=probe), self.assertRaises(followups.FollowupError):
                    _validate(deferral, cohort, **probe)

    def test_deferral_cli_flags_are_required_together(self):
        parser = followups.build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(["--user-deferral-receipt", "/tmp/deferral.json"])
        with self.assertRaises(SystemExit):
            parser.parse_args(["--user-deferral-sha256", "a" * 64])
        args = parser.parse_args([
            "--user-deferral-receipt", "/tmp/deferral.json",
            "--user-deferral-sha256", "A" * 64,
        ])
        self.assertEqual(args.user_deferral_sha256, "a" * 64)


if __name__ == "__main__":
    unittest.main()
