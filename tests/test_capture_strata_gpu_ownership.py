import contextlib
from datetime import datetime, timedelta, timezone
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from runtime import capture_strata_gpu_ownership as capture
from runtime import study_resource_report


TAG = "specialists-flash-publisher-weighted-t1-72k64k-v1"
ENGINE = "/opt/strata/bin/strata"
MODEL = "qwen3.8-flash-next-coder-iq1_m"
NOW = datetime(2026, 10, 21, 0, 0, 13, tzinfo=timezone.utc)
BOOT = int(datetime(2026, 10, 21, 0, 0, 0, tzinfo=timezone.utc).timestamp())


class Fixture:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.source = root / "source.json"
        self.private = root / f"{TAG}-config.json"
        self.events = root / f"{TAG}-runtime-events.jsonl"
        self.output = root / f"{TAG}-gpu-ownership-v1.json"
        self.source_config = {
            "args": ["--pack", "/models/pack", "--max-context", "73728"],
            "exe": ENGINE,
            "host": "127.0.0.1",
            "log": "/previous/session-engine.log",
            "model_name": MODEL,
            "open_browser": False,
            "port": 18080,
            "reference_environment": {"STRATA_ROUTE_TAIL_SKIP": "0"},
        }
        self.private_config = dict(self.source_config)
        self.private_config["log"] = str(root / f"{TAG}-engine.log")
        self.source.write_text(json.dumps(self.source_config, indent=2) + "\n", encoding="utf-8")
        self.private.write_text(json.dumps(self.private_config, indent=2) + "\n", encoding="utf-8")
        self.rows = [
            {
                "at": "2026-10-21T00:00:10.500Z",
                "event": "session_started",
                "tag": TAG,
                "run_id": "fixture-run",
                "runtime_backend": "strata",
                "source_config": str(self.source),
                "private_config": str(self.private),
                "engine": ENGINE,
                "engine_environment": {"STRATA_ROUTE_TAIL_SKIP": "0"},
                "model_name": MODEL,
                "context": 73728,
            },
            {
                "at": "2026-10-21T00:00:10.500Z",
                "event": "server_started",
                "tag": TAG,
                "run_id": "fixture-run",
                "runtime_backend": "strata",
                "pid": 5001,
                "argv": ["python3", "-m", "serve.server", "--config", str(self.private)],
            },
            {
                "at": "2026-10-21T00:00:12.000Z",
                "event": "server_ready",
                "tag": TAG,
                "run_id": "fixture-run",
                "runtime_backend": "strata",
                "model": MODEL,
            },
            {
                "at": "2026-10-21T00:00:12.500Z",
                "event": "step_started",
                "tag": TAG,
                "run_id": "fixture-run",
            },
        ]
        self.write_rows()

    def write_rows(self) -> None:
        self.events.write_text(
            "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in self.rows),
            encoding="utf-8",
        )

    def source_sha(self) -> str:
        return hashlib.sha256(self.source.read_bytes()).hexdigest()

    def enable_model_session_reasoning_default(self) -> None:
        self.private_config["reasoning_effort"] = "none"
        self.private.write_text(json.dumps(self.private_config, indent=2) + "\n", encoding="utf-8")
        step = next(row for row in self.rows if row["event"] == "step_started")
        step.update({
            "index": 1,
            "step_id": "hard_weighted",
            "controls": {
                "per_request_reasoning_effort": True,
                "thinking": "medium",
            },
            "argv": [
                "python3", "bench.py", "--thinking", "medium", "--output", "/tmp/result.jsonl",
            ],
        })
        self.write_rows()

    def dependencies(self, *, gpu_processes=None, gpu_after_start="1050") -> capture.Dependencies:
        identities = {
            4001: [
                capture.ProcessIdentity(4001, 3001, "500", "S", "/usr/bin/python3.12"),
                capture.ProcessIdentity(4001, 3001, "500", "R", "/usr/bin/python3.12"),
            ],
            5001: [
                capture.ProcessIdentity(5001, 4001, "1000", "S", "/usr/bin/python3.12"),
                capture.ProcessIdentity(5001, 4001, "1000", "R", "/usr/bin/python3.12"),
            ],
            6001: [
                capture.ProcessIdentity(6001, 5001, "1050", "R", ENGINE),
                capture.ProcessIdentity(6001, 5001, gpu_after_start, "S", ENGINE),
            ],
        }

        def process_identity(pid):
            values = identities.get(pid)
            if not values:
                raise AssertionError(f"unexpected process identity request: {pid}")
            return values.pop(0)

        observed = gpu_processes or (lambda: [{"pid": 6001, "name": ENGINE, "used_mib": 14442}])
        return capture.Dependencies(
            process_identity=process_identity,
            process_environment=lambda pid: {"STRATA_ROUTE_TAIL_SKIP": "0"},
            gpu_processes=observed,
            kernel_clock=lambda: capture.KernelClock("fixture-boot", BOOT, 100),
            now=lambda: NOW,
        )

    def argv(self, *, allow_reasoning_default=False):
        result = [
            "--tag", TAG,
            "--source-config", str(self.source),
            "--expected-source-sha256", self.source_sha(),
            "--results-dir", str(self.root),
        ]
        if allow_reasoning_default:
            result.append("--allow-model-session-reasoning-default-none")
        return result


class CaptureStrataGpuOwnershipTests(unittest.TestCase):
    def test_capture_writes_exact_consumer_accepted_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(Path(directory))
            with mock.patch.object(capture.subprocess, "run", side_effect=AssertionError("live GPU query")):
                output, receipt = capture.capture(
                    tag=TAG,
                    source_config=fixture.source,
                    expected_source_sha256=fixture.source_sha(),
                    results_dir=fixture.root,
                    dependencies=fixture.dependencies(),
                )

            self.assertEqual(output, fixture.output)
            self.assertEqual(receipt["api_process"]["state"], "R")
            self.assertEqual(receipt["gpu_process"]["state"], "S")
            self.assertEqual(receipt["gpu_process"]["parent_pid"], receipt["api_process"]["pid"])
            self.assertEqual(receipt["gpu_observation"]["pid"], receipt["gpu_process"]["pid"])
            self.assertEqual(receipt["engine_environment"], {"STRATA_ROUTE_TAIL_SKIP": "0"})
            self.assertEqual(len(receipt), 21)
            report = study_resource_report.build_report(
                fixture.root, generated_at=NOW + timedelta(seconds=1),
            )
            ownership = next(
                item for item in report["sources"]["gpu_ownership"]
                if item["identity"] == {"tag": TAG, "run_id": "fixture-run"}
            )
            self.assertTrue(ownership["accepted"], ownership["issues"])

    def test_main_returns_nonzero_and_writes_nothing_when_identity_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(Path(directory))
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr), contextlib.redirect_stdout(io.StringIO()):
                code = capture.main(
                    fixture.argv(), dependencies=fixture.dependencies(gpu_after_start="1051"),
                )
            self.assertEqual(code, 2)
            self.assertIn("identity changed", stderr.getvalue())
            self.assertFalse(fixture.output.exists())

    def test_model_session_reasoning_default_opt_in_is_consumer_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(Path(directory))
            fixture.enable_model_session_reasoning_default()
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(io.StringIO()):
                code = capture.main(
                    fixture.argv(allow_reasoning_default=True),
                    dependencies=fixture.dependencies(),
                )
            self.assertEqual(code, 0)
            self.assertEqual(stdout.getvalue().strip(), str(fixture.output))
            receipt = json.loads(fixture.output.read_text(encoding="utf-8"))
            self.assertEqual(len(receipt), 21)
            self.assertEqual(receipt["private_config"]["sha256"], hashlib.sha256(
                fixture.private.read_bytes()
            ).hexdigest())
            report = study_resource_report.build_report(
                fixture.root, generated_at=NOW + timedelta(seconds=1),
            )
            ownership = next(
                item for item in report["sources"]["gpu_ownership"]
                if item["identity"] == {"tag": TAG, "run_id": "fixture-run"}
            )
            self.assertTrue(ownership["accepted"], ownership["issues"])

    def test_reasoning_default_stays_strict_without_opt_in(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(Path(directory))
            fixture.enable_model_session_reasoning_default()
            with self.assertRaisesRegex(capture.CaptureError, "only at the per-tag log path"):
                capture.capture(
                    tag=TAG,
                    source_config=fixture.source,
                    results_dir=fixture.root,
                    dependencies=fixture.dependencies(),
                )
            self.assertFalse(fixture.output.exists())

    def test_reasoning_default_opt_in_requires_exact_source_and_active_step_proof(self):
        variants = {
            "source reasoning_effort": lambda fixture: fixture.source_config.__setitem__(
                "reasoning_effort", "medium"
            ),
            "per_request_reasoning_effort": lambda fixture: next(
                row for row in fixture.rows if row["event"] == "step_started"
            )["controls"].__setitem__("per_request_reasoning_effort", False),
            "matching active controls": lambda fixture: next(
                row for row in fixture.rows if row["event"] == "step_started"
            )["argv"].__setitem__(3, "none"),
        }
        for message, mutate in variants.items():
            with self.subTest(message=message), tempfile.TemporaryDirectory() as directory:
                fixture = Fixture(Path(directory))
                fixture.enable_model_session_reasoning_default()
                mutate(fixture)
                if message == "source reasoning_effort":
                    fixture.source.write_text(
                        json.dumps(fixture.source_config, indent=2) + "\n", encoding="utf-8"
                    )
                else:
                    fixture.write_rows()
                with self.assertRaisesRegex(capture.CaptureError, message):
                    capture.capture(
                        tag=TAG,
                        source_config=fixture.source,
                        results_dir=fixture.root,
                        allow_model_session_reasoning_default_none=True,
                        dependencies=fixture.dependencies(),
                    )
                self.assertFalse(fixture.output.exists())
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(Path(directory))
            fixture.enable_model_session_reasoning_default()
            fixture.private_config["reasoning_effort"] = "medium"
            fixture.private.write_text(
                json.dumps(fixture.private_config, indent=2) + "\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(capture.CaptureError, "differs beyond"):
                capture.capture(
                    tag=TAG,
                    source_config=fixture.source,
                    results_dir=fixture.root,
                    allow_model_session_reasoning_default_none=True,
                    dependencies=fixture.dependencies(),
                )
            self.assertFalse(fixture.output.exists())

    def test_source_byte_change_during_snapshot_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(Path(directory))

            def mutate_source():
                fixture.source.write_bytes(fixture.source.read_bytes() + b" ")
                return [{"pid": 6001, "name": ENGINE, "used_mib": 14442}]

            with self.assertRaisesRegex(capture.CaptureError, "config bytes changed"):
                capture.capture(
                    tag=TAG,
                    source_config=fixture.source,
                    expected_source_sha256=fixture.source_sha(),
                    results_dir=fixture.root,
                    dependencies=fixture.dependencies(gpu_processes=mutate_source),
                )
            self.assertFalse(fixture.output.exists())

    def test_terminal_missing_ready_and_non_strata_lifecycles_are_rejected(self):
        variants = {
            "terminal or stopped": lambda fixture: fixture.rows.append({
                "at": "2026-10-21T00:00:12.900Z", "event": "server_stopped",
                "tag": TAG, "run_id": "fixture-run",
            }),
            "exactly one": lambda fixture: fixture.rows.__setitem__(
                slice(None), [row for row in fixture.rows if row["event"] != "server_ready"]
            ),
            "Strata backend": lambda fixture: fixture.rows[0].__setitem__("runtime_backend", "llama"),
        }
        for message, mutate in variants.items():
            with self.subTest(message=message), tempfile.TemporaryDirectory() as directory:
                fixture = Fixture(Path(directory))
                mutate(fixture)
                fixture.write_rows()
                with self.assertRaisesRegex(capture.CaptureError, message):
                    capture.capture(
                        tag=TAG,
                        source_config=fixture.source,
                        results_dir=fixture.root,
                        dependencies=fixture.dependencies(),
                    )
                self.assertFalse(fixture.output.exists())

    def test_wrong_private_config_or_tail_environment_is_rejected(self):
        variants = ("private", "environment")
        for variant in variants:
            with self.subTest(variant=variant), tempfile.TemporaryDirectory() as directory:
                fixture = Fixture(Path(directory))
                if variant == "private":
                    private = json.loads(fixture.private.read_text(encoding="utf-8"))
                    private["port"] = 18081
                    fixture.private.write_text(json.dumps(private) + "\n", encoding="utf-8")
                    message = "differ from source"
                else:
                    fixture.rows[0]["engine_environment"] = {"STRATA_ROUTE_TAIL_SKIP": "1"}
                    fixture.write_rows()
                    message = "identity differs"
                with self.assertRaisesRegex(capture.CaptureError, message):
                    capture.capture(
                        tag=TAG,
                        source_config=fixture.source,
                        results_dir=fixture.root,
                        dependencies=fixture.dependencies(),
                    )
                self.assertFalse(fixture.output.exists())

    def test_api_pid_cannot_be_used_as_gpu_pid(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(Path(directory))
            dependencies = fixture.dependencies(
                gpu_processes=lambda: [{
                    "pid": 5001, "name": "/usr/bin/python3.12", "used_mib": 14442,
                }]
            )
            with self.assertRaisesRegex(capture.CaptureError, "exactly one direct"):
                capture.capture(
                    tag=TAG,
                    source_config=fixture.source,
                    results_dir=fixture.root,
                    dependencies=dependencies,
                )
            self.assertFalse(fixture.output.exists())

    def test_existing_output_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = Fixture(Path(directory))
            fixture.output.write_bytes(b"sentinel\n")
            dependencies = fixture.dependencies(gpu_processes=lambda: self.fail("GPU query ran"))
            with self.assertRaisesRegex(capture.CaptureError, "overwrite"):
                capture.capture(
                    tag=TAG,
                    source_config=fixture.source,
                    results_dir=fixture.root,
                    dependencies=dependencies,
                )
            self.assertEqual(fixture.output.read_bytes(), b"sentinel\n")


if __name__ == "__main__":
    unittest.main()
