import argparse
import contextlib
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from runtime import complete_specialist_study as continuation


class Clock:
    def __init__(self):
        self.value = 0.0

    def monotonic(self):
        return self.value

    def sleep(self, seconds):
        self.value += seconds


class ImmediateProcess:
    def __init__(self, pid, returncode):
        self.pid = pid
        self.returncode = returncode
        self.signals = []

    def poll(self):
        return self.returncode

    def send_signal(self, signum):
        self.signals.append(signum)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_study(
    path: Path,
    profile: str,
    *,
    status="complete",
    backend="llama",
    port=18081,
    cleanup_ok=True,
):
    returncode = 1 if status == "completed_with_evaluation_failures" else 0
    document = {
        "schema_version": 1,
        "status": status,
        "runtime_backend": backend,
        "controls": {"profile": profile},
        "cleanup": {
            "outer_exited": True,
            "outer_returncode": returncode,
            "orchestrator_sigkill_sent": False,
            "model_session_cleanup": {
                "stopped": True,
                "kill_sent": False if cleanup_ok else True,
            },
        },
        "restoration": {
            "original_running": True,
            "original_paused": False,
            "owned_loopback_port_clear": True,
            "owned_loopback_port": port,
        },
        "problem": None,
    }
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


class Fixture:
    def __init__(self, root: Path, *, prefix="study", expected_models=None):
        self.root = root
        self.results = root / "results"
        self.runtime = root / "runtime"
        self.results.mkdir()
        self.runtime.mkdir()
        self.prefix = prefix
        self.expected_models = tuple(expected_models or continuation.CANONICAL_FIRST_WAVE[1:])
        self.sampler = self.runtime / "sampler.json"
        self.standard = self.runtime / "standard.json"
        self.role = self.runtime / "role.json"
        self.config = self.runtime / "matching-config.json"
        for path in (self.sampler, self.standard, self.role, self.config):
            path.write_text("{}\n", encoding="utf-8")
        self.first_script = self.runtime / "first.py"
        self.followup_script = self.runtime / "followup.py"
        self.matching_script = self.runtime / "matching.py"
        for path in (self.first_script, self.followup_script, self.matching_script):
            path.write_text("# fixture\n", encoding="utf-8")
        self.first_wave = self.results / f"{prefix}-first-wave-cohort.json"
        self.followups = self.results / f"{prefix}-followups-cohort.json"
        self.matching = self.results / f"{prefix}-gpt20b-study-trial.json"
        commands = [
            {
                "kind": "selected_kat_style",
                "argv": [
                    "python3",
                    str(self.first_script),
                    "--run-prefix",
                    prefix,
                    "--models",
                    ",".join(self.expected_models),
                    "--sampler-plan",
                    str(self.sampler),
                ],
            },
            {
                "kind": "bonsai_and_flash",
                "argv": [
                    "python3",
                    str(self.followup_script),
                    "--run-prefix",
                    prefix,
                    "--first-wave-receipt",
                    str(self.first_wave),
                    "--standard-plan",
                    str(self.standard),
                    "--role-plan",
                    str(self.role),
                    "--sampler-plan",
                    str(self.sampler),
                ],
            },
            {
                "kind": "matching_20b_reference_LAST",
                "argv": [
                    "python3",
                    str(self.matching_script),
                    "--config",
                    str(self.config),
                    "--plan",
                    str(self.standard),
                    "--tag",
                    f"{prefix}-gpt20b",
                    "--profile",
                    "gpt20b-native-medium-16k-12k",
                ],
            },
        ]
        self.plan = self.results / "commands.json"
        self.plan.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "commands": commands,
                    "sampler_plan": str(self.sampler),
                    "sampler_plan_sha256": _sha(self.sampler),
                }
            ),
            encoding="utf-8",
        )

    def args(self, *, initial_wait=1.0):
        return argparse.Namespace(
            commands_plan=self.plan,
            commands_sha256=_sha(self.plan),
            run_prefix=self.prefix,
            results_dir=self.results,
            expected_parent_pid=77,
            expected_parent_start_ticks=1234,
            initial_wait_seconds=initial_wait,
            stage_timeout_seconds=10.0,
            boundary_wait_seconds=1.0,
            cancel_grace_seconds=1.0,
            poll_seconds=0.1,
        )

    def write_first_wave(self, *, status="complete", malformed=False):
        if malformed:
            self.first_wave.write_text("{broken", encoding="utf-8")
            return
        models = []
        for key in self.expected_models:
            path = _write_study(
                self.results / f"{key}.json", key, status=status if not models else "complete"
            )
            models.append(
                {
                    "key": key,
                    "status": status if len(models) == 0 else "complete",
                    "selected_profile": "strict_all_layers_fit_off",
                    "strict_receipt": str(path),
                    "strict_receipt_sha256": _sha(path),
                }
            )
        cohort_status = (
            "completed_with_evaluation_failures"
            if any(row["status"] == "completed_with_evaluation_failures" for row in models)
            else "complete"
        )
        self.first_wave.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "status": cohort_status,
                    "controls": {"selected_models": list(self.expected_models)},
                    "models": models,
                    "problem": None,
                }
            ),
            encoding="utf-8",
        )

    def write_followups(self, *, status="complete", cleanup_ok=True):
        models = []
        for index, (key, backend, port) in enumerate(continuation.EXPECTED_FOLLOWUPS):
            model_status = status if index == 0 else "complete"
            path = _write_study(
                self.results / f"{key}.json",
                key,
                status=model_status,
                backend=backend,
                port=port,
                cleanup_ok=cleanup_ok if index == 0 else True,
            )
            models.append(
                {
                    "key": key,
                    "status": model_status,
                    "study": {"path": str(path), "sha256": _sha(path)},
                }
            )
        cohort_status = (
            "completed_with_evaluation_failures"
            if any(row["status"] == "completed_with_evaluation_failures" for row in models)
            else "complete"
        )
        self.followups.write_text(
            json.dumps({"schema_version": 1, "status": cohort_status, "models": models}),
            encoding="utf-8",
        )

    def write_matching(self, *, status="complete"):
        _write_study(
            self.matching,
            "gpt20b-native-medium-16k-12k",
            status=status,
        )


def _boundary():
    return {
        "checked_at": "now",
        "original_running": True,
        "original_paused": False,
        "ports_clear": {"18080": True, "18081": True},
    }


def _pid_probe_after_two_parent_checks(pid):
    if not hasattr(_pid_probe_after_two_parent_checks, "calls"):
        _pid_probe_after_two_parent_checks.calls = 0
    if pid == 77:
        _pid_probe_after_two_parent_checks.calls += 1
        return 1234 if _pid_probe_after_two_parent_checks.calls <= 2 else None
    return pid * 10


class CompleteSpecialistStudyTests(unittest.TestCase):
    def setUp(self):
        _pid_probe_after_two_parent_checks.calls = 0

    def test_far_wrapper_bounds_are_opt_in_and_defaults_stay_unchanged(self):
        parser = continuation.build_parser()
        required = [
            "--commands-sha256", "a" * 64,
            "--expected-parent-pid", "1",
            "--expected-parent-start-ticks", "1",
        ]
        defaults = parser.parse_args(required)
        self.assertEqual(defaults.initial_wait_seconds, 46800.0)
        self.assertEqual(defaults.stage_timeout_seconds, 129600.0)
        wide = parser.parse_args(required + [
            "--initial-wait-seconds", "604800",
            "--stage-timeout-seconds", "604800",
        ])
        self.assertEqual(wide.initial_wait_seconds, 604800.0)
        self.assertEqual(wide.stage_timeout_seconds, 604800.0)
        with self.assertRaises(SystemExit):
            parser.parse_args(required + ["--initial-wait-seconds", "604801"])
        with self.assertRaises(SystemExit):
            parser.parse_args(required + ["--stage-timeout-seconds", "604801"])

    def test_first_wave_selection_is_derived_from_pinned_command_in_canonical_order(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(
                Path(directory), expected_models=continuation.CANONICAL_FIRST_WAVE
            )
            loaded = continuation.load_commands_plan(
                fixture.plan, _sha(fixture.plan), fixture.prefix, fixture.results
            )
            self.assertEqual(loaded.expected_first_wave, continuation.CANONICAL_FIRST_WAVE)
            fixture.write_first_wave()
            validated = continuation.validate_first_wave(
                fixture.first_wave, loaded.expected_first_wave
            )
            self.assertEqual(validated["selected_models"], list(continuation.CANONICAL_FIRST_WAVE))
            document = json.loads(fixture.plan.read_text(encoding="utf-8"))
            document["commands"][0]["argv"][
                document["commands"][0]["argv"].index("--models") + 1
            ] = "gemma4-style12-q5_k_m,kat-v25-iq3_xxs"
            fixture.plan.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(continuation.ContinuationError, "canonical order"):
                continuation.load_commands_plan(
                    fixture.plan, _sha(fixture.plan), fixture.prefix, fixture.results
                )

    def _run_full(
        self,
        fixture,
        *,
        first_status="complete",
        followup_status="complete",
        matching_status="complete",
        cleanup_ok=True,
    ):
        fixture.write_first_wave(status=first_status)
        launched = []

        def popen(argv, **_kwargs):
            launched.append(Path(argv[1]).name)
            if Path(argv[1]) == fixture.followup_script:
                fixture.write_followups(status=followup_status, cleanup_ok=cleanup_ok)
                return ImmediateProcess(901, 1 if followup_status != "complete" else 0)
            fixture.write_matching(status=matching_status)
            return ImmediateProcess(902, 1 if matching_status != "complete" else 0)

        clock = Clock()
        deps = continuation.Dependencies(
            popen=popen,
            pid_probe=_pid_probe_after_two_parent_checks,
            boundary_probe=_boundary,
            monotonic=clock.monotonic,
            sleep=clock.sleep,
        )
        with contextlib.redirect_stdout(io.StringIO()):
            document, code = continuation.execute(fixture.args(), deps=deps)
        return document, code, launched

    def test_waits_without_launching_when_parent_identity_remains_live(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(Path(directory))
            fixture.write_first_wave()
            launched = []
            clock = Clock()
            deps = continuation.Dependencies(
                popen=lambda *args, **kwargs: launched.append(args),
                pid_probe=lambda _pid: 1234,
                boundary_probe=_boundary,
                monotonic=clock.monotonic,
                sleep=clock.sleep,
            )
            with contextlib.redirect_stdout(io.StringIO()):
                document, code = continuation.execute(
                    fixture.args(initial_wait=0.3), deps=deps
                )
            self.assertEqual(code, 2)
            self.assertEqual(document["status"], "infrastructure_failed")
            self.assertEqual(launched, [])

    def test_semantic_failure_is_accepted_and_matching_runs_last(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(Path(directory))
            document, code, launched = self._run_full(
                fixture,
                first_status="completed_with_evaluation_failures",
                followup_status="completed_with_evaluation_failures",
                matching_status="completed_with_evaluation_failures",
            )
            self.assertEqual(code, 1)
            self.assertEqual(document["status"], "completed_with_evaluation_failures")
            self.assertEqual(launched, ["followup.py", "matching.py"])
            self.assertEqual([row["returncode"] for row in document["stages"]], [1, 1])
            self.assertTrue(all(row["cleanup_verified"] for row in document["receipts"]["first_wave"]["studies"]))
            self.assertEqual(document["commands"][0]["argv"][1], str(fixture.followup_script))
            self.assertRegex(document["commands"][0]["argv_sha256"], r"^[0-9a-f]{64}$")

    def test_infrastructure_cleanup_failure_stops_before_matching(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(Path(directory))
            document, code, launched = self._run_full(fixture, cleanup_ok=False)
            self.assertEqual(code, 2)
            self.assertEqual(document["status"], "infrastructure_failed")
            self.assertEqual(launched, ["followup.py"])
            self.assertIn("safe restoration", document["problem"]["error"])

    def test_malformed_first_wave_receipt_fails_without_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(Path(directory))
            fixture.write_first_wave(malformed=True)
            launched = []
            clock = Clock()
            deps = continuation.Dependencies(
                popen=lambda *args, **kwargs: launched.append(args),
                pid_probe=lambda _pid: None,
                boundary_probe=_boundary,
                monotonic=clock.monotonic,
                sleep=clock.sleep,
            )
            with contextlib.redirect_stdout(io.StringIO()):
                document, code = continuation.execute(fixture.args(), deps=deps)
            self.assertEqual(code, 2)
            self.assertIn("malformed JSON", document["problem"]["error"])
            self.assertEqual(launched, [])

    def test_existing_audit_prevents_duplicate_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(Path(directory))
            fixture.write_first_wave()
            launched = []
            clock = Clock()
            deps = continuation.Dependencies(
                popen=lambda *args, **kwargs: launched.append(args),
                pid_probe=lambda _pid: 1234,
                boundary_probe=_boundary,
                monotonic=clock.monotonic,
                sleep=clock.sleep,
            )
            with contextlib.redirect_stdout(io.StringIO()):
                _, code = continuation.execute(fixture.args(initial_wait=0.2), deps=deps)
            self.assertEqual(code, 2)
            with self.assertRaisesRegex(continuation.ContinuationError, "duplicate launch"):
                continuation.execute(fixture.args(initial_wait=0.2), deps=deps)
            self.assertEqual(launched, [])

    def test_unrestored_boundary_blocks_first_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(Path(directory))
            fixture.write_first_wave()
            launched = []
            clock = Clock()

            def bad_boundary():
                raise continuation.ContinuationError("original Docker is paused")

            deps = continuation.Dependencies(
                popen=lambda *args, **kwargs: launched.append(args),
                pid_probe=lambda _pid: None,
                boundary_probe=bad_boundary,
                monotonic=clock.monotonic,
                sleep=clock.sleep,
            )
            with contextlib.redirect_stdout(io.StringIO()):
                document, code = continuation.execute(
                    fixture.args(initial_wait=0.3), deps=deps
                )
            self.assertEqual(code, 2)
            self.assertIn("safe launch boundary", document["problem"]["error"])
            self.assertEqual(launched, [])


if __name__ == "__main__":
    unittest.main()
