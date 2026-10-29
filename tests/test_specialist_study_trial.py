import argparse
import contextlib
import hashlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from runtime import model_session
from runtime import specialist_study_trial as study


def _args(root: Path) -> argparse.Namespace:
    config = root / "config.json"
    plan = root / "plan.json"
    config.write_text('{"config":true}\n')
    plan.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "name": "test plan",
                "steps": [
                    {
                        "id": "quick",
                        "suite": "quick",
                        "thinking": "medium",
                        "temperature": 0,
                        "seed": 0,
                        "output_tokens": 512,
                        "request_timeout": 900,
                    }
                ],
            }
        )
    )
    return argparse.Namespace(
        runtime_backend="llama",
        config=config.resolve(),
        plan=plan.resolve(),
        tag="specialist-test",
        profile="coding",
        prose_thinking="low",
        code_thinking="medium",
        reasoning_cap=1536,
        capability_note="always-on tool use",
        code_output_cap=4096,
        prose_output_cap=2048,
        code_timeout=900.0,
        prose_timeout=900.0,
        startup_timeout=480.0,
        wait_for_idle=900.0,
        max_seconds=7200.0,
    )


def _sampler_args(root: Path) -> argparse.Namespace:
    args = _args(root)
    args.config.write_text(json.dumps({"model_name": "model", "context": 16384}))
    recipe = root / "recipe.json"
    recipe.write_text(json.dumps({
        "schema_version": 1,
        "label": "test publisher recipe",
        "claim": "publisher_inspired",
        "sources": ["https://example.test/model"],
        "sampler_overrides": {"temperature": 0.8},
    }))
    sampler_plan = root / "sampler-plan.json"
    sampler_plan.write_text(json.dumps({
        "schema_version": 1,
        "models": {
            "model": [
                {"group": "code", "recipe": "recipe.json", "suffix": "publisher-code"},
                {
                    "group": "code",
                    "recipe": "recipe.json",
                    "suffix": "publisher-code-none",
                    "code_thinking": "none",
                },
                {
                    "group": "prose",
                    "recipe": "recipe.json",
                    "suffix": "publisher-prose",
                    "prose_thinking": "none",
                },
            ]
        },
    }))
    args.sampler_plan = sampler_plan
    return args


class _Process:
    def __init__(self, returncode=None):
        self.returncode = returncode
        self.pid = 9001

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        self.returncode = 0 if self.returncode is None else self.returncode
        return self.returncode


class SpecialistStudyTrialTests(unittest.TestCase):
    def test_parser_requires_existing_plan_choice_and_defaults_thinking_medium(self):
        parser = study.build_parser()
        args = parser.parse_args(
            ["--config", "config.json", "--plan", "plan.json", "--tag", "trial"]
        )
        self.assertEqual(args.runtime_backend, "llama")
        self.assertEqual(args.prose_thinking, "medium")
        self.assertEqual(args.code_thinking, "medium")
        self.assertEqual(args.max_seconds, 7200.0)
        self.assertEqual(args.code_output_cap, 4096)
        self.assertEqual(args.prose_output_cap, 2048)
        self.assertIsNone(args.sampler_plan)
        self.assertFalse(args.standard_only)

    def test_commands_pin_window_hold_and_group_controls(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = _args(root)
            control = root / "results" / "hold.json"
            outer = study.gpu_window_command(args, control)
            session = study.HoldingSession(
                "run", "llama", "lfm", 8192, 12.5, "start", "ready", "hold"
            )
            code = study.specialist_command(args, session, "code", root / "code.jsonl")
            prose = study.specialist_command(args, session, "prose", root / "prose.jsonl")
        self.assertIn("--long-session", outer)
        self.assertEqual(outer[outer.index("--wait-seconds") + 1], "900")
        self.assertEqual(outer[outer.index("--max-seconds") + 1], "7200")
        self.assertIn("--hold-after-plan", outer)
        self.assertEqual(code[code.index("--groups") + 1], "code")
        self.assertEqual(code[code.index("--context") + 1], "8192")
        self.assertEqual(code[code.index("--output-cap") + 1], "4096")
        self.assertEqual(code[code.index("--temperature") + 1], "0")
        self.assertEqual(code[code.index("--thinking") + 1], "medium")
        self.assertEqual(prose[prose.index("--groups") + 1], "prose")
        self.assertEqual(prose[prose.index("--output-cap") + 1], "2048")
        self.assertEqual(prose[prose.index("--temperature") + 1], "0.7")
        self.assertEqual(prose[prose.index("--thinking") + 1], "low")
        self.assertEqual(prose[prose.index("--chat-template-thinking") + 1], "on")
        self.assertEqual(prose[prose.index("--reasoning-budget-tokens") + 1], "1536")
        self.assertEqual(prose[prose.index("--capability-note") + 1], "always-on tool use")

    def test_sampler_plan_validates_recipes_and_forwards_exact_resident_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = _sampler_args(root)
            sampler_plan = study.load_sampler_plan(args.sampler_plan)
            model_name, stages = study.sampler_runtime_selection(
                sampler_plan,
                config_path=args.config,
                runtime_backend="llama",
            )
            session = study.HoldingSession(
                "exact-run-id", "llama", model_name, 16384, 1.0, "start", "ready", "hold"
            )
            command = study.sampler_command(
                args,
                session,
                stages[0],
                root / f"{args.tag}-{stages[0].suffix}",
                root / f"{args.tag}-runtime-events.jsonl",
            )
            nonthinking_command = study.sampler_command(
                args,
                session,
                stages[1],
                root / f"{args.tag}-{stages[1].suffix}",
                root / f"{args.tag}-runtime-events.jsonl",
            )

            duplicate = json.loads(args.sampler_plan.read_text())
            duplicate["models"]["model"][1]["suffix"] = "publisher-code"
            duplicate_path = root / "duplicate.json"
            duplicate_path.write_text(json.dumps(duplicate))
            with self.assertRaisesRegex(study.StudyError, "repeats suffix"):
                study.load_sampler_plan(duplicate_path)
            with self.assertRaisesRegex(study.StudyError, "native llama"):
                study.sampler_runtime_selection(
                    sampler_plan,
                    config_path=args.config,
                    runtime_backend="strata",
                )
            args.config.write_text(json.dumps({"model_name": "flash-model", "context": 16384}))
            flash_model, flash_stages = study.sampler_runtime_selection(
                sampler_plan,
                config_path=args.config,
                runtime_backend="strata",
            )
            self.assertEqual((flash_model, flash_stages), ("flash-model", ()))

            three_code = json.loads(args.sampler_plan.read_text())
            three_code["models"]["model"][2]["group"] = "code"
            three_code["models"]["model"][2].pop("prose_thinking")
            three_code["models"]["model"][2]["code_thinking"] = "none"
            three_code_path = root / "three-code.json"
            three_code_path.write_text(json.dumps(three_code))
            self.assertEqual(
                len(study.load_sampler_plan(three_code_path).models["model"]), 3
            )

            too_many = json.loads(args.sampler_plan.read_text())
            too_many["models"]["model"].append({
                "group": "prose", "recipe": "recipe.json", "suffix": "fourth-stage",
            })
            too_many_path = root / "too-many.json"
            too_many_path.write_text(json.dumps(too_many))
            with self.assertRaisesRegex(study.StudyError, "one to three stages"):
                study.load_sampler_plan(too_many_path)

            invalid_mode = json.loads(args.sampler_plan.read_text())
            invalid_mode["models"]["model"][1]["code_thinking"] = "low"
            invalid_mode_path = root / "invalid-mode.json"
            invalid_mode_path.write_text(json.dumps(invalid_mode))
            with self.assertRaisesRegex(study.StudyError, "code_thinking must be none or medium"):
                study.load_sampler_plan(invalid_mode_path)

            invalid_group = json.loads(args.sampler_plan.read_text())
            invalid_group["models"]["model"][2]["code_thinking"] = "none"
            invalid_group_path = root / "invalid-group.json"
            invalid_group_path.write_text(json.dumps(invalid_group))
            with self.assertRaisesRegex(study.StudyError, "only valid for code"):
                study.load_sampler_plan(invalid_group_path)

        self.assertEqual(command[command.index("--model-session-run-id") + 1], "exact-run-id")
        self.assertEqual(command[command.index("--endpoint") + 1], "http://127.0.0.1:18081/v1")
        self.assertEqual(command[command.index("--model") + 1], "model")
        self.assertEqual(command[command.index("--groups") + 1], "code")
        self.assertEqual(command[command.index("--output-cap") + 1], "4096")
        self.assertEqual(command[command.index("--timeout") + 1], "900")
        self.assertNotIn("--code-thinking", command)
        self.assertEqual(
            nonthinking_command[nonthinking_command.index("--code-thinking") + 1], "none"
        )
        self.assertEqual(
            nonthinking_command[nonthinking_command.index("--model-session-run-id") + 1],
            "exact-run-id",
        )

    def _run_sampler_study(self, root: Path, *, request_failure: bool, semantic_wrong: bool):
        results = root / "results"
        results.mkdir()
        args = _sampler_args(root)
        outer = _Process()
        holding = study.HoldingSession(
            "resident-run", "llama", "model", 16384, 11.0, "start", "ready", "hold"
        )
        session_report = results / f"{args.tag}-session-report.json"
        calls = []

        def fake_stop(_control):
            session_report.write_text(json.dumps({
                "status": "passed",
                "server": {"cleanup": {"stopped": True, "kill_sent": False}},
            }))
            return True

        def fake_run(argv, *, kind, outer, deadline, signals):
            calls.append((kind, list(argv)))
            returncode = 0
            if kind.startswith("sampler_"):
                prefix = Path(argv[argv.index("--output-prefix") + 1])
                recipe_path = Path(argv[argv.index("--recipe") + 1])
                _, recipe_sha = study.sampler_recipe_probe.load_recipe(recipe_path)
                failed = request_failure and "--code-thinking" in argv
                returncode = 1 if failed else 0
                receipt = {
                    "status": "completed_with_request_failures" if failed else "complete",
                    "tag": args.tag,
                    "model": holding.model_name,
                    "model_session_run_id": holding.run_id,
                    "runtime_backend": "llama",
                    "recipe": {"path": str(recipe_path), "sha256": recipe_sha},
                    "request_failure_count": 1 if failed else 0,
                    "attempted_count": 1,
                    "completed_count": 0 if failed else 1,
                    "truncated_count": 0,
                    "controls": (
                    {
                        "ledger_thinking": "medium", "code_thinking": "none",
                        "expected_context_tokens": holding.context,
                        "actual_context_tokens": holding.context,
                    }
                        if "--code-thinking" in argv
                    else {
                        "ledger_and_code_thinking": "medium",
                        "expected_context_tokens": holding.context,
                        "actual_context_tokens": holding.context,
                    }
                    ),
                }
                if semantic_wrong:
                    receipt["semantic_wrong_answers"] = 1
                Path(f"{prefix}-sampler-receipt.json").write_text(json.dumps(receipt))
            return {
                "kind": kind,
                "argv": list(argv),
                "started_at": "start",
                "finished_at": "finish",
                "elapsed_seconds": 0.1,
                "returncode": returncode,
            }

        with mock.patch.object(
            study, "load_specialist_manifest", return_value=({"cases": {}}, {
                "kind": "specialist_manifest", "argv": ["manifest"], "returncode": 0,
                "stdout_sha256": "a" * 64,
            })
        ), mock.patch.object(
            study.subprocess, "Popen", return_value=outer
        ), mock.patch.object(
            study, "wait_for_holding", return_value=holding
        ), mock.patch.object(
            study, "run_specialist", side_effect=fake_run
        ), mock.patch.object(
            study, "request_session_stop", side_effect=fake_stop
        ), mock.patch.object(
            study, "wait_for_outer_exit", return_value=(0, False)
        ), mock.patch.object(
            study,
            "restoration_evidence",
            return_value={
                "original_running": True,
                "original_paused": False,
                "owned_loopback_port_clear": True,
            },
        ), contextlib.redirect_stdout(io.StringIO()):
            receipt, code = study.execute_study(args, results=results)
        return receipt, code, calls

    def test_sampler_request_failure_stops_later_stage_and_still_cleans_up(self):
        with tempfile.TemporaryDirectory() as directory:
            receipt, code, calls = self._run_sampler_study(
                Path(directory), request_failure=True, semantic_wrong=False
            )
        self.assertEqual(code, 1)
        self.assertEqual(receipt["status"], "infrastructure_failed")
        self.assertEqual([kind for kind, _ in calls], [
            "specialist_code", "specialist_prose", "sampler_code_publisher-code",
            "sampler_code_publisher-code-none",
        ])
        self.assertEqual(
            receipt["sampler_plan"]["stages"][1]["receipt"]["status"],
            "completed_with_request_failures",
        )
        self.assertTrue(receipt["cleanup"]["stop_requested"])
        self.assertTrue(receipt["cleanup"]["outer_exited"])

    def test_clean_sampler_stage_does_not_treat_semantic_wrong_answers_as_infrastructure(self):
        with tempfile.TemporaryDirectory() as directory:
            receipt, code, calls = self._run_sampler_study(
                Path(directory), request_failure=False, semantic_wrong=True
            )
        self.assertEqual(code, 0)
        self.assertEqual(receipt["status"], "complete")
        self.assertEqual([kind for kind, _ in calls][-2:], [
            "sampler_code_publisher-code-none", "sampler_prose_publisher-prose",
        ])
        self.assertEqual(
            [stage["receipt"]["status"] for stage in receipt["sampler_plan"]["stages"]],
            ["complete", "complete", "complete"],
        )

    def test_standard_only_runs_plan_then_skips_specialist_and_sampler_stages(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results = root / "results"
            results.mkdir()
            args = _sampler_args(root)
            args.standard_only = True
            outer = _Process()
            holding = study.HoldingSession(
                "resident-run", "llama", "model", 16384, 11.0, "start", "ready", "hold"
            )
            session_report = results / f"{args.tag}-session-report.json"

            def fake_stop(_control):
                session_report.write_text(json.dumps({
                    "status": "passed",
                    "server": {"cleanup": {"stopped": True, "kill_sent": False}},
                }))
                return True

            with mock.patch.object(
                study, "load_specialist_manifest"
            ) as load_manifest, mock.patch.object(
                study.subprocess, "Popen", return_value=outer
            ), mock.patch.object(
                study, "wait_for_holding", return_value=holding
            ), mock.patch.object(
                study, "run_specialist"
            ) as run_specialist, mock.patch.object(
                study, "request_session_stop", side_effect=fake_stop
            ), mock.patch.object(
                study, "wait_for_outer_exit", return_value=(0, False)
            ), mock.patch.object(
                study, "restoration_evidence", return_value={
                    "original_running": True,
                    "original_paused": False,
                    "owned_loopback_port_clear": True,
                }
            ), contextlib.redirect_stdout(io.StringIO()):
                receipt, code = study.execute_study(args, results=results)
            self.assertEqual(code, 0)
            self.assertEqual(receipt["status"], "complete")
            self.assertTrue(receipt["controls"]["standard_only"])
            self.assertTrue(receipt["specialist_manifest"]["skipped_by_standard_only"])
            self.assertTrue(receipt["sampler_plan"]["skipped_by_standard_only"])
            self.assertEqual(receipt["sampler_plan"]["stages"], [])
            self.assertIn(
                "standard_plan_only_complete",
                [row.get("phase") for row in receipt["lifecycle"]],
            )
            load_manifest.assert_not_called()
            run_specialist.assert_not_called()

    def test_explicit_high_budget_controls_are_forwarded_and_bounded(self):
        parser = study.build_parser()
        args = parser.parse_args([
            "--config", "config.json", "--plan", "plan.json", "--tag", "trial",
            "--code-output-cap", "65536", "--prose-output-cap", "65536",
            "--code-timeout", "21600", "--prose-timeout", "21600",
            "--max-seconds", "86400",
        ])
        session = study.HoldingSession(
            "run", "llama", "model", 16384, 1.0, "start", "ready", "hold"
        )
        command = study.specialist_command(args, session, "code", Path("code.jsonl"))
        self.assertEqual(command[command.index("--context") + 1], "16384")
        self.assertEqual(command[command.index("--output-cap") + 1], "65536")
        self.assertEqual(command[command.index("--timeout") + 1], "21600")
        self.assertEqual(args.max_seconds, 86400.0)
        sampler = study.SamplerStage(
            "model", "code", Path("recipe.json"), "a" * 64, "recipe", "wide", None, "medium"
        )
        sampler_argv = study.sampler_command(
            args, session, sampler, Path("wide"), Path("events.jsonl")
        )
        self.assertEqual(
            sampler_argv[sampler_argv.index("--expected-context") + 1], "16384"
        )
        for option, value in (
            ("--code-output-cap", "65537"),
            ("--code-timeout", "21601"),
            ("--max-seconds", "86401"),
        ):
            with self.subTest(option=option), self.assertRaises(SystemExit):
                parser.parse_args([
                    "--config", "config.json", "--plan", "plan.json", "--tag", "trial",
                    option, value,
                ])

    def test_strata_specialist_command_uses_18080_without_template_flag(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = _args(root)
            args.runtime_backend = "strata"
            args.code_thinking = "none"
            session = study.HoldingSession(
                "run", "strata", "flash", 32768, 20.0, "start", "ready", "hold"
            )
            command = study.specialist_command(args, session, "code", root / "code.jsonl")
        self.assertEqual(command[command.index("--endpoint") + 1], "http://127.0.0.1:18080/v1")
        self.assertNotIn("--chat-template-thinking", command)

    def test_holding_gate_requires_this_tag_run_config_plan_and_control(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = _args(root)
            control = root / "control.json"
            start = {
                "at": "start",
                "event": "session_started",
                "run_id": "ours",
                "tag": args.tag,
                "runtime_backend": "llama",
                "source_config": str(args.config),
                "plan": str(args.plan),
                "control_file": str(control),
                "model_name": "lfm",
                "context": 8192,
            }
            rows = [
                start,
                {
                    "at": "ready",
                    "event": "server_ready",
                    "run_id": "ours",
                    "tag": args.tag,
                    "initial_load_seconds": 10.5,
                },
                {
                    "at": "hold",
                    "event": "holding_started",
                    "run_id": "ours",
                    "tag": args.tag,
                    "control_file": str(control),
                },
            ]
            identity = study._holding_identity(
                rows,
                tag=args.tag,
                source_config=args.config,
                plan=args.plan,
                control=control,
                runtime_backend="llama",
            )
            self.assertEqual(identity.run_id, "ours")
            mixed = [*rows, {**rows[-1], "run_id": "stale"}]
            with self.assertRaisesRegex(study.StudyError, "multiple run IDs"):
                study._holding_identity(
                    mixed,
                    tag=args.tag,
                    source_config=args.config,
                    plan=args.plan,
                    control=control,
                    runtime_backend="llama",
                )

    def test_output_guard_rejects_stale_specialist_or_session_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = _args(root)
            plan = model_session.load_plan(args.plan)
            stale = root / f"{args.tag}-specialist-code.jsonl"
            stale.write_text("stale")
            with self.assertRaisesRegex(study.StudyError, "overwrite"):
                study.assert_outputs_unused(args.tag, plan, "llama", root)

    def test_strata_artifact_inventory_includes_private_runtime_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = _args(root)
            plan = model_session.load_plan(args.plan)
            paths = study._artifact_paths(args.tag, plan, "strata", root)
        self.assertEqual(paths["engine_log"].name, f"{args.tag}-engine.log")
        self.assertEqual(paths["private_config"].name, f"{args.tag}-config.json")
        self.assertTrue(paths["code_texts"].name.endswith(".jsonl.texts"))

    def test_manifest_capture_records_exact_stdout_hash(self):
        payload = json.dumps({"schema_version": 1, "cases": {"code": ["a"]}}).encode()
        completed = SimpleNamespace(returncode=0, stdout=payload, stderr=b"")
        with mock.patch.object(Path, "is_file", return_value=True), mock.patch.object(
            study.subprocess, "run", return_value=completed
        ):
            manifest, record = study.load_specialist_manifest(Path("/helper.py"))
        self.assertEqual(manifest["cases"]["code"], ["a"])
        self.assertEqual(record["stdout_sha256"], hashlib.sha256(payload).hexdigest())

    def test_bench_infrastructure_error_still_requests_stop_and_waits_outer(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results = root / "results"
            results.mkdir()
            args = _args(root)
            outer = _Process()
            holding = study.HoldingSession(
                "run", "llama", "lfm", 8192, 11.0, "start", "ready", "hold"
            )
            session_report = results / f"{args.tag}-session-report.json"

            def fake_stop(control):
                session_report.write_text(json.dumps({
                    "status": "passed",
                    "server": {"cleanup": {"stopped": True, "kill_sent": False}},
                }))
                return True

            with mock.patch.object(
                study, "load_specialist_manifest", return_value=({"cases": {}}, {
                    "kind": "specialist_manifest", "argv": ["manifest"], "returncode": 0,
                    "stdout_sha256": "a" * 64,
                })
            ), mock.patch.object(study, "sha256_file", return_value="b" * 64), mock.patch.object(
                study.subprocess, "Popen", return_value=outer
            ), mock.patch.object(
                study, "wait_for_holding", return_value=holding
            ), mock.patch.object(
                study, "run_specialist", side_effect=study.StudyError("bench launch failed")
            ), mock.patch.object(
                study, "request_session_stop", side_effect=fake_stop
            ) as stop, mock.patch.object(
                study, "wait_for_outer_exit", return_value=(0, False)
            ) as wait, mock.patch.object(
                study,
                "restoration_evidence",
                return_value={
                    "original_running": True,
                    "original_paused": False,
                    "owned_loopback_port_clear": True,
                },
            ), contextlib.redirect_stdout(io.StringIO()):
                receipt, code = study.execute_study(args, results=results)

        self.assertEqual(code, 1)
        self.assertEqual(receipt["status"], "infrastructure_failed")
        self.assertEqual(receipt["problem"]["error"], "bench launch failed")
        stop.assert_called_once()
        wait.assert_called_once_with(
            outer,
            timeout=180.0,
            control=results / f"{args.tag}-control.json",
        )
        self.assertFalse(receipt["cleanup"]["orchestrator_sigkill_sent"])
        self.assertFalse(receipt["cleanup"]["model_session_cleanup"]["kill_sent"])

    def test_pre_hold_abort_terms_only_owned_outer_group_then_waits_for_restoration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results = root / "results"
            results.mkdir()
            args = _args(root)
            outer = _Process()
            with mock.patch.object(
                study, "load_specialist_manifest", return_value=({"cases": {}}, {
                    "kind": "specialist_manifest", "argv": ["manifest"], "returncode": 0,
                    "stdout_sha256": "a" * 64,
                })
            ), mock.patch.object(study, "sha256_file", return_value="b" * 64), mock.patch.object(
                study.subprocess, "Popen", return_value=outer
            ), mock.patch.object(
                study, "wait_for_holding", side_effect=study.StudyError("identity mismatch")
            ), mock.patch.object(
                study, "request_session_stop", return_value=False
            ), mock.patch.object(
                study.os, "killpg"
            ) as killpg, mock.patch.object(
                study, "wait_for_outer_exit", return_value=(0, False)
            ) as wait, mock.patch.object(
                study,
                "restoration_evidence",
                return_value={
                    "original_running": True,
                    "original_paused": False,
                    "owned_loopback_port_clear": True,
                },
            ), contextlib.redirect_stdout(io.StringIO()):
                receipt, code = study.execute_study(args, results=results)

        self.assertEqual(code, 1)
        self.assertEqual(receipt["status"], "infrastructure_failed")
        killpg.assert_called_once_with(outer.pid, study.signal.SIGTERM)
        wait.assert_called_once_with(
            outer,
            timeout=180.0,
            control=results / f"{args.tag}-control.json",
        )
        self.assertTrue(receipt["cleanup"]["outer_sigterm_sent"])
        self.assertFalse(receipt["cleanup"]["orchestrator_sigkill_sent"])

    def test_request_stop_rewrites_only_regular_owned_control(self):
        with tempfile.TemporaryDirectory() as directory:
            control = Path(directory) / "control.json"
            control.write_text('{"stop":false}\n')
            self.assertTrue(study.request_session_stop(control))
            self.assertEqual(json.loads(control.read_text()), {"stop": True})
            self.assertFalse(study.request_session_stop(Path(directory) / "missing.json"))

    def test_outer_wait_requests_control_stop_without_signalling(self):
        with tempfile.TemporaryDirectory() as directory:
            control = Path(directory) / "control.json"
            control.write_text('{"stop":false}\n')

            class Process:
                def __init__(self):
                    self.polls = 0

                def poll(self):
                    self.polls += 1
                    return 0 if self.polls >= 2 else None

            with mock.patch.object(study.time, "sleep"):
                returncode, stopped = study.wait_for_outer_exit(
                    Process(), timeout=10, control=control
                )
            self.assertEqual(returncode, 0)
            self.assertTrue(stopped)
            self.assertEqual(json.loads(control.read_text()), {"stop": True})


if __name__ == "__main__":
    unittest.main()
