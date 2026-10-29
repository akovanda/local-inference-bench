import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from runtime import run_style_mtp_probe as probe
from runtime import specialist_cases
from runtime import specialist_study_trial as study


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _primary_receipt() -> dict:
    return {
        "schema_version": 1,
        "tag": probe.PARENT_TAG,
        "status": "completed_with_evaluation_failures",
        "runtime_backend": "llama",
        "source_config": {
            "sha256": "3ffb305436bb7a19207064c64a9a32b9e864cbaf843a8633780d1005867b9e15",
        },
        "plan": {"sha256": probe.PRIMARY_PLAN_SHA256},
        "session": {
            "run_id": "primary-run",
            "model_name": probe.MODEL,
            "context": probe.CONTEXT,
        },
        "commands": [
            {"kind": "specialist_code", "returncode": 0},
            {"kind": "specialist_prose", "returncode": 0},
        ],
        "cleanup": {
            "outer_exited": True,
            "orchestrator_sigkill_sent": False,
            "model_session_cleanup": {"stopped": True, "kill_sent": False},
        },
        "restoration": {
            "original_running": True,
            "original_paused": False,
            "owned_loopback_port_clear": True,
        },
        "session_report_status": "completed_with_step_failures",
        "controls": {
            "profile": probe.PRIMARY_PROFILE,
            "prose_thinking": "none",
            "prose": {
                "output_cap": probe.OUTPUT_CAP,
                "temperature": 0.7,
                "seed": 0,
                "timeout": probe.REQUEST_TIMEOUT,
            },
        },
        "sampler_plan": {
            "matched": True,
            "stages": [{
                "group": "prose",
                "receipt": {
                    "status": "complete",
                    "attempted_count": 3,
                    "completed_count": 3,
                    "truncated_count": 0,
                    "request_failure_count": 0,
                },
            }],
        },
    }


def _primary_rows(*, wall_seconds: float = 100.0) -> list[dict]:
    rows = []
    for index, case in enumerate(specialist_cases.get_cases(groups=("prose",))):
        rows.append({
            "schema_version": 1,
            "run_id": "primary-prose",
            "call_id": f"call-{index}",
            "model": probe.MODEL,
            "tag": probe.PARENT_TAG + "-prose",
            "profile": probe.PRIMARY_PROFILE,
            "suite": "specialist_prose",
            "case_id": case.case_id,
            "case_group": "prose",
            "case_version": specialist_cases.CASE_VERSION,
            "case_sha256": specialist_cases.case_sha256(case),
            "suite_sha256": specialist_cases.FROZEN_SUITE_SHA256,
            "request": {
                "model": probe.MODEL,
                "profile": probe.PRIMARY_PROFILE,
                "backend": "openai",
                "thinking": "none",
                "chat_template_thinking": "off",
                "reasoning_budget_tokens": None,
                "temperature": 0.7,
                "seed": 0,
                "context_max_tokens": probe.CONTEXT,
                "output_token_limit": probe.OUTPUT_CAP,
                "timeout_seconds": probe.REQUEST_TIMEOUT,
                "prompt_chars": len(case.prompt),
                "prompt_sha256": hashlib.sha256(case.prompt.encode()).hexdigest(),
            },
            "response": {
                "actual_completion_tokens": 200 + index,
                "done_reason": "stop",
                "stream_complete": True,
                "truncated": False,
            },
            "response_metrics": {
                "finish_reason": "stop",
                "output_tokens_reported": 200 + index,
                "wall_seconds": wall_seconds,
            },
            "visible_text": {
                "path": f"opaque-{index}.txt",
                "sha256": f"{index + 1:064x}",
                "chars": 900 + index,
                "bytes": 900 + index,
                "reasoning_excluded": True,
            },
            "objective_outcome": "wrong_answer",
            "objective_error": None,
            "error": None,
            # A mechanical/rubric failure is deliberately not an eligibility gate.
            "evaluation": {"passed": False},
        })
    return rows


def _args(root: Path, receipt: Path, prose: Path, *, execute: bool = False):
    tag = "style-mtp-test"
    return argparse.Namespace(
        primary_study_receipt=receipt,
        primary_prose=[prose],
        tag=tag,
        output_prefix=root / "results" / f"{tag}-probe",
        execute=execute,
    )


class FakeProcess:
    def __init__(self, pid: int = 4321, returncode=None):
        self.pid = pid
        self.returncode = returncode

    def poll(self):
        return self.returncode


def _execution_receipt() -> dict:
    return {
        "execution": {},
        "commands": {},
        "cleanup": {
            "control_stop_requested": False,
            "owned_outer_sigterm_sent": False,
        },
        "smoke": {"status": "not_run"},
        "prose": {
            "status": "not_run",
            "attempted_count": 0,
            "completed_count": 0,
            "truncated_count": 0,
            "request_failure_count": 0,
        },
        "limitations": {"speculative_acceptance_measured": False},
    }


class StyleMtpProbeTests(unittest.TestCase):
    def test_standalone_cli_help_bootstraps_imports_outside_repo(self):
        script = Path(probe.__file__).resolve()
        environment = os.environ.copy()
        environment.pop("PYTHONPATH", None)
        with tempfile.TemporaryDirectory() as directory:
            completed = subprocess.run(
                [sys.executable, "-I", str(script), "--help"],
                cwd=directory,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=10,
                check=False,
            )
            created = list(Path(directory).iterdir())
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--execute", completed.stdout)
        self.assertIn("--primary-study-receipt", completed.stdout)
        self.assertEqual(created, [])

    def test_ineligible_inspect_launches_nothing_and_records_no_fake_attempt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            receipt_path = root / "primary.json"
            prose_path = root / "primary-prose.jsonl"
            receipt = _primary_receipt()
            receipt["cleanup"]["outer_exited"] = False
            rows = _primary_rows()
            rows[0]["response"]["truncated"] = True
            _write_json(receipt_path, receipt)
            _write_jsonl(prose_path, rows)
            args = _args(root, receipt_path, prose_path)
            with (
                mock.patch.object(probe, "verify_pins", return_value={"verified": True}),
                mock.patch.object(probe.subprocess, "Popen") as popen,
            ):
                result, code = probe.run(args, results=root / "results")
        self.assertEqual(code, 1)
        self.assertEqual(result["status"], "ineligible_not_launched")
        self.assertFalse(result["execution"]["launched"])
        self.assertEqual(result["prose"]["attempted_count"], 0)
        self.assertIn("cleanup", " ".join(result["eligibility"]["failures"]))
        popen.assert_not_called()

    def test_matching_controls_and_commands_are_exact_and_prose_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results = root / "results"
            results.mkdir()
            paths = probe.probe_paths(
                "style-mtp-test", results / "style-mtp-test-probe", results=results
            )
            outer = probe.outer_command("style-mtp-test", paths.control)
            sampler = probe.sampler_command(
                "style-mtp-test", paths.prefix, paths.recipe, "exact-run"
            )
        self.assertIn("--long-session", outer)
        self.assertEqual(outer[outer.index("--wait-seconds") + 1], "900")
        self.assertEqual(outer[outer.index("--max-seconds") + 1], "14400")
        self.assertIn("--hold-after-plan", outer)
        self.assertEqual(outer[outer.index("--control-file") + 1], str(paths.control))
        self.assertEqual(sampler[sampler.index("--groups") + 1], "prose")
        self.assertEqual(sampler[sampler.index("--prose-thinking") + 1], "none")
        self.assertEqual(sampler[sampler.index("--output-cap") + 1], "65536")
        self.assertEqual(sampler[sampler.index("--timeout") + 1], "21600")
        self.assertEqual(sampler[sampler.index("--model-session-run-id") + 1], "exact-run")
        self.assertEqual(sampler[sampler.index("--profile") + 1], probe.PRIMARY_PROFILE)
        self.assertNotIn("code", sampler[sampler.index("--groups") + 1])

    def test_refuses_output_collision_and_precreated_control(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results = root / "results"
            results.mkdir()
            paths = probe.probe_paths("style-mtp-test", results / "style-mtp-test", results=results)
            paths.smoke.touch()
            with self.assertRaisesRegex(probe.ProbeError, "existing output/control"):
                probe.assert_unused(paths)
            paths.smoke.unlink()
            paths.control.write_text('{"stop":false}\n')
            with self.assertRaisesRegex(probe.ProbeError, "existing output/control"):
                probe.assert_unused(paths)
            receipt = {
                "execution": {}, "commands": {}, "cleanup": {}, "smoke": {}, "prose": {},
            }
            with (
                mock.patch.object(probe, "assert_ports_clear") as ports,
                self.assertRaisesRegex(probe.ProbeError, "must not exist"),
            ):
                probe.execute_probe(paths, "style-mtp-test", receipt)
            ports.assert_not_called()

    def test_exact_held_identity_rejects_mixed_and_terminal_events(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results = root / "results"
            results.mkdir()
            tag = "style-mtp-test"
            paths = probe.probe_paths(tag, results / tag, results=results)
            base = [
                {
                    "at": "start", "event": "session_started", "tag": tag,
                    "run_id": "owned", "runtime_backend": "llama",
                    "source_config": str(probe.CONFIG), "plan": str(probe.SMOKE_PLAN),
                    "control_file": str(paths.control), "model_name": probe.MODEL,
                    "context": probe.CONTEXT,
                },
                {
                    "at": "ready", "event": "server_ready", "tag": tag,
                    "run_id": "owned", "initial_load_seconds": 2.5,
                },
                {
                    "at": "hold", "event": "holding_started", "tag": tag,
                    "run_id": "owned", "control_file": str(paths.control),
                },
            ]
            _write_jsonl(paths.events, base)
            session = probe.wait_for_holding(FakeProcess(), paths, time.monotonic() + 1)
            self.assertEqual(session.run_id, "owned")

            mixed = list(base)
            mixed[1] = dict(mixed[1], run_id="foreign")
            _write_jsonl(paths.events, mixed)
            with self.assertRaisesRegex(probe.ProbeError, "multiple run IDs"):
                probe.wait_for_holding(FakeProcess(), paths, time.monotonic() + 1)

            terminal = list(base) + [{
                "at": "done", "event": "server_stopped", "tag": tag, "run_id": "owned",
            }]
            _write_jsonl(paths.events, terminal)
            with self.assertRaisesRegex(probe.ProbeError, "stale or terminal"):
                probe.wait_for_holding(FakeProcess(), paths, time.monotonic() + 1)

    def test_smoke_failure_skips_prose_and_stops_owned_session(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results = root / "results"
            results.mkdir()
            tag = "style-mtp-test"
            paths = probe.probe_paths(tag, results / tag, results=results)
            outer = FakeProcess()
            session = study.HoldingSession(
                "owned", "llama", probe.MODEL, probe.CONTEXT, 1.0, "a", "b", "c"
            )
            receipt = _execution_receipt()

            def held(*_args, **_kwargs):
                paths.control.write_text('{"stop":false}\n')
                return session

            with (
                mock.patch.object(probe, "assert_ports_clear", return_value={"native": True, "strata": True}),
                mock.patch.object(probe, "_exclusive_json"),
                mock.patch.object(probe.subprocess, "Popen", return_value=outer),
                mock.patch.object(probe, "wait_for_holding", side_effect=held),
                mock.patch.object(probe, "_server_identity", return_value={"run_id": "owned"}),
                mock.patch.object(probe, "validate_smoke", side_effect=probe.ProbeError("smoke failed")),
                mock.patch.object(probe, "run_owned") as run_owned,
                mock.patch.object(probe.study, "request_session_stop", return_value=True) as stop,
                mock.patch.object(probe.study, "wait_for_outer_exit", return_value=(0, False)),
                mock.patch.object(probe.study, "restoration_evidence", return_value={
                    "original_running": True, "original_paused": False,
                    "owned_loopback_port_clear": True,
                }),
            ):
                code = probe.execute_probe(paths, tag, receipt)
        self.assertEqual(code, 1)
        run_owned.assert_not_called()
        self.assertEqual(receipt["prose"]["status"], "not_run")
        self.assertEqual(receipt["prose"]["attempted_count"], 0)
        self.assertEqual(receipt["prose"]["completed_count"], 0)
        stop.assert_called_once_with(paths.control)
        self.assertTrue(receipt["cleanup"]["control_stop_requested"])
        self.assertFalse(receipt["cleanup"]["orchestrator_sigkill_sent"])

    def test_owned_helper_timeout_terminates_only_its_process_group(self):
        child = FakeProcess(pid=9876)
        with (
            mock.patch.object(probe.subprocess, "Popen", return_value=child),
            mock.patch.object(probe.time, "monotonic", return_value=2.0),
            mock.patch.object(probe, "_terminate_owned", return_value=True) as terminate,
            self.assertRaisesRegex(probe.ProbeError, "guard expired"),
        ):
            probe.run_owned(["synthetic-helper"], deadline=1.0)
        terminate.assert_called_once_with(child)

    def test_launched_prose_is_interrupted_or_incomplete_with_unknown_counts(self):
        scenarios = (
            ("guard", probe.ProbeError("whole-profile guard expired"), "interrupted", 1),
            ("signal", probe.ProbeInterrupted(15), "interrupted", 143),
            ("validation", probe.ProbeError("terminal receipt invalid"), "incomplete", 1),
        )
        for name, failure, expected_status, expected_code in scenarios:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                results = root / "results"
                results.mkdir()
                tag = f"style-mtp-{name}"
                paths = probe.probe_paths(tag, results / tag, results=results)
                outer = FakeProcess(pid=4100)
                helper = FakeProcess(pid=4200, returncode=-15 if name != "validation" else 0)
                session = study.HoldingSession(
                    "owned", "llama", probe.MODEL, probe.CONTEXT, 1.0, "a", "b", "c"
                )
                receipt = _execution_receipt()
                _write_jsonl(paths.sampler_requests, [{
                    "case_id": "prose_case",
                    "call_id": "call-1",
                    "response": {
                        "stream_complete": True,
                        "truncated": False,
                        "actual_completion_tokens": 17,
                        "openai_native_timings": {"draft_n": 9, "draft_n_accepted": 4},
                    },
                    "response_metrics": {"output_tokens_reported": 17, "wall_seconds": 2.5},
                    "error": None,
                }])
                with paths.sampler_requests.open("ab") as handle:
                    handle.write(b'{"partial"')

                def held(*_args, **_kwargs):
                    paths.control.write_text('{"stop":false}\n')
                    return session

                if name in {"guard", "signal"}:
                    setattr(failure, "owned_process", helper)
                    setattr(failure, "owned_cleanup", {
                        "pid": helper.pid,
                        "pgid": helper.pid,
                        "term_sent": True,
                        "kill_sent": False,
                        "stopped": True,
                    })
                    owned_result = mock.DEFAULT
                    owned_side_effect = failure
                    sampler_side_effect = None
                else:
                    owned_result = (helper, 0)
                    owned_side_effect = None
                    sampler_side_effect = failure

                with (
                    mock.patch.object(
                        probe, "assert_ports_clear", return_value={"native": True, "strata": True}
                    ),
                    mock.patch.object(probe, "_exclusive_json"),
                    mock.patch.object(probe.subprocess, "Popen", return_value=outer),
                    mock.patch.object(probe, "wait_for_holding", side_effect=held),
                    mock.patch.object(probe, "_server_identity", return_value={"run_id": "owned"}),
                    mock.patch.object(probe, "validate_smoke", return_value={"passed": True}),
                    mock.patch.object(
                        probe,
                        "run_owned",
                        return_value=owned_result,
                        side_effect=owned_side_effect,
                    ),
                    mock.patch.object(
                        probe,
                        "summarize_sampler_receipt",
                        side_effect=sampler_side_effect,
                    ),
                    mock.patch.object(probe.study, "request_session_stop", return_value=True),
                    mock.patch.object(probe.study, "wait_for_outer_exit", return_value=(0, False)),
                    mock.patch.object(probe.study, "restoration_evidence", return_value={
                        "original_running": True,
                        "original_paused": False,
                        "owned_loopback_port_clear": True,
                    }),
                    mock.patch.object(probe, "session_cleanup_evidence", return_value=({}, [])),
                ):
                    code = probe.execute_probe(paths, tag, receipt)

                self.assertEqual(code, expected_code)
                self.assertEqual(
                    receipt["status"],
                    "interrupted" if name == "signal" else "infrastructure_failed",
                )
                self.assertEqual(receipt["prose"]["status"], expected_status)
                self.assertTrue(receipt["prose"]["helper_launched"])
                self.assertFalse(receipt["prose"]["terminal_sampler_receipt_validated"])
                self.assertIsNone(receipt["prose"]["attempted_count"])
                self.assertIsNone(receipt["prose"]["completed_count"])
                self.assertIsNone(receipt["prose"]["request_failure_count"])
                observed = receipt["prose"]["observed_request_metadata"]
                self.assertEqual(observed["observed_row_count"], 1)
                self.assertEqual(observed["observed_completed_row_count"], 1)
                self.assertGreater(observed["ignored_incomplete_trailing_bytes"], 0)
                self.assertEqual(receipt["execution"]["prose_helper_pid"], helper.pid)
                self.assertTrue(receipt["cleanup"]["control_stop_requested"])

    def test_valid_primary_gate_ignores_subjective_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            receipt_path = root / "primary.json"
            prose_path = root / "primary-prose.jsonl"
            _write_json(receipt_path, _primary_receipt())
            _write_jsonl(prose_path, _primary_rows())
            eligibility = probe.validate_eligibility(receipt_path, [prose_path])
        self.assertTrue(eligibility.eligible, eligibility.failures)
        self.assertEqual(eligibility.primary_receipt["attempt_accounting"]["total"], 15)
        self.assertEqual(len(eligibility.primary_prose), 3)

    def test_acceptance_summary_reports_coverage_without_visible_text(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "requests.jsonl"
            rows = []
            for index, case in enumerate(specialist_cases.get_cases(groups=("prose",))):
                rows.append({
                    "tag": "style-mtp-test",
                    "profile": probe.PRIMARY_PROFILE,
                    "model": probe.MODEL,
                    "case_id": case.case_id,
                    "case_group": "prose",
                    "case_sha256": specialist_cases.case_sha256(case),
                    "request": {
                        "model": probe.MODEL,
                        "profile": probe.PRIMARY_PROFILE,
                        "thinking": "none",
                        "chat_template_thinking": "off",
                        "temperature": 0.7,
                        "seed": 0,
                        "context_max_tokens": probe.CONTEXT,
                        "output_token_limit": probe.OUTPUT_CAP,
                        "timeout_seconds": probe.REQUEST_TIMEOUT,
                        "prompt_sha256": hashlib.sha256(case.prompt.encode()).hexdigest(),
                    },
                    "response": {
                        "openai_native_timings": {
                            "draft_n": 10 + index,
                            "draft_n_accepted": 5 + index,
                        },
                    },
                })
            _write_jsonl(path, rows)
            summary = probe.summarize_acceptance(path, tag="style-mtp-test")
        self.assertEqual(summary["reported_case_count"], 3)
        self.assertEqual(summary["missing_case_count"], 0)
        self.assertEqual(summary["draft_proposed_total"], 33)
        self.assertEqual(summary["draft_accepted_total"], 18)
        self.assertNotIn("response_text", json.dumps(summary))


if __name__ == "__main__":
    unittest.main()
