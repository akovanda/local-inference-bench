import argparse
import contextlib
import json
import os
import signal
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from runtime import llama_trial, model_session, strata_trial


class _Events:
    def __init__(self):
        self.run_id = "run-id"
        self.rows = []

    def write(self, event, **fields):
        self.rows.append((event, fields))


class _Process:
    pid = 9001
    returncode = None

    def poll(self):
        return self.returncode


def _step(step_id="off", **overrides):
    values = {
        "step_id": step_id,
        "suite": "hard",
        "cases": ("hard_transaction_ledger",),
        "thinking": "none",
        "temperature": 0.0,
        "seed": 41,
        "output_tokens": 4096,
        "request_timeout": 900.0,
        "repeat": 1,
        "context_target": None,
    }
    values.update(overrides)
    return model_session.Step(**values)


class ModelSessionTests(unittest.TestCase):
    def test_plan_loads_actual_per_request_controls(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "plan.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "name": "focused",
                        "steps": [
                            {
                                "id": "off",
                                "suite": "hard",
                                "cases": ["hard_transaction_ledger"],
                                "thinking": "none",
                                "temperature": 0.7,
                                "seed": 41,
                                "output_tokens": 4096,
                                "request_timeout": 900,
                                "repeat": 2,
                            },
                            {
                                "id": "medium",
                                "suite": "hard",
                                "cases": ["hard_transaction_ledger"],
                                "thinking": "medium",
                                "output_tokens": 4096,
                                "request_timeout": 1200,
                            },
                            {
                                "id": "context20k",
                                "suite": "context",
                                "thinking": "none",
                                "context_target": 20000,
                                "output_tokens": 2048,
                                "request_timeout": 1200,
                            },
                        ],
                    }
                )
            )
            plan = model_session.load_plan(path)

        self.assertEqual(plan.name, "focused")
        self.assertEqual([step.thinking for step in plan.steps], ["none", "medium", "none"])
        self.assertEqual(plan.steps[2].context_target, 20000)
        self.assertEqual(plan.steps[0].controls()["seed_step"], 1)

    def test_plan_rejects_unsafe_or_unbounded_controls(self):
        valid = {
            "schema_version": 1,
            "steps": [{"id": "one", "suite": "quick", "thinking": "none"}],
        }
        invalid_changes = (
            ("nonfinite temperature", {"temperature": float("nan")}),
            ("long timeout", {"request_timeout": 21601}),
            ("large output", {"output_tokens": 65537}),
            ("too many repeats", {"repeat": 4}),
            ("unknown case", {"cases": ["hard_transaction_ledger"]}),
            ("context target on quick", {"context_target": 1000}),
        )
        for label, changes in invalid_changes:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as temporary:
                value = json.loads(json.dumps(valid))
                value["steps"][0].update(changes)
                path = Path(temporary) / "plan.json"
                path.write_text(json.dumps(value, allow_nan=True))
                with self.assertRaises(model_session.SessionError):
                    model_session.load_plan(path)

    def test_plan_accepts_explicit_high_budget_controls(self):
        value = {
            "schema_version": 1,
            "steps": [{
                "id": "high", "suite": "hard", "thinking": "medium",
                "output_tokens": 65536, "request_timeout": 21600,
            }],
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "plan.json"
            path.write_text(json.dumps(value))
            step = model_session.load_plan(path).steps[0]
        self.assertEqual(step.output_tokens, 65536)
        self.assertEqual(step.request_timeout, 21600.0)
        plan = model_session.Plan(Path("plan.json"), "a" * 64, None, (step,))
        model_session.validate_plan_context(plan, 131072)
        with self.assertRaisesRegex(model_session.SessionError, "exceeds the maximum"):
            model_session.validate_plan_context(plan, 131073)

    def test_frozen_16k_plans_use_8192_tokens_and_2400_second_bounds(self):
        root = Path(__file__).resolve().parents[1]
        standard = model_session.load_plan(root / "runtime" / "specialists-standard-16k-plan.json")
        role = model_session.load_plan(root / "runtime" / "specialists-role-screen-16k-plan.json")
        self.assertEqual([step.suite for step in standard.steps], ["quick", "hard"])
        self.assertEqual([step.suite for step in role.steps], ["quick"])
        for step in (*standard.steps, *role.steps):
            self.assertEqual(step.output_tokens, 8192)
            self.assertEqual(step.request_timeout, 2400.0)

    def test_frozen_12k_plans_use_12288_tokens_and_3600_second_bounds(self):
        root = Path(__file__).resolve().parents[1]
        standard = model_session.load_plan(
            root / "runtime" / "specialists-standard-16k-12k-plan.json"
        )
        role = model_session.load_plan(
            root / "runtime" / "specialists-role-screen-16k-12k-plan.json"
        )
        self.assertEqual([step.suite for step in standard.steps], ["quick", "hard"])
        self.assertEqual([step.suite for step in role.steps], ["quick"])
        for step in (*standard.steps, *role.steps):
            self.assertEqual(step.output_tokens, 12288)
            self.assertEqual(step.request_timeout, 3600.0)

    def test_bench_commands_change_thinking_on_each_request(self):
        config = SimpleNamespace(model_name="qwen", context=32768)
        off = model_session.bench_command(config, _step(), Path("off.jsonl"))
        medium = model_session.bench_command(
            config,
            _step(
                "medium",
                thinking="medium",
                temperature=0.6,
                seed=42,
                repeat=2,
            ),
            Path("medium.jsonl"),
        )
        self.assertEqual(off[off.index("--thinking") + 1], "none")
        self.assertEqual(medium[medium.index("--thinking") + 1], "medium")
        self.assertEqual(medium[medium.index("--temperature") + 1], "0.6")
        self.assertEqual(medium[medium.index("--seed") + 1], "42")
        self.assertEqual(medium[medium.index("--repeat") + 1], "2")
        self.assertEqual(medium[medium.index("--seed-step") + 1], "1")
        self.assertEqual(medium[medium.index("--num-predict") + 1], "4096")
        self.assertEqual(medium[medium.index("--timeout") + 1], "900")

    def test_llama_bench_commands_set_endpoint_and_template_thinking_per_request(self):
        config = SimpleNamespace(model_name="qwen122", context=32768)
        off = model_session.bench_command(
            config,
            _step(),
            Path("off.jsonl"),
            "llama",
        )
        low = model_session.bench_command(
            config,
            _step("low", thinking="low"),
            Path("low.jsonl"),
            "llama",
        )
        medium = model_session.bench_command(
            config,
            _step("medium", thinking="medium"),
            Path("medium.jsonl"),
            "llama",
        )
        self.assertEqual(off[off.index("--endpoint") + 1], llama_trial.ENDPOINT)
        self.assertEqual(off[off.index("--chat-template-thinking") + 1], "off")
        self.assertEqual(low[low.index("--chat-template-thinking") + 1], "on")
        self.assertEqual(medium[medium.index("--chat-template-thinking") + 1], "on")
        self.assertEqual(low[low.index("--thinking") + 1], "low")
        self.assertEqual(medium[medium.index("--thinking") + 1], "medium")

    def test_parser_defaults_to_strata_and_accepts_llama(self):
        parser = model_session.build_parser()
        default = parser.parse_args(["--plan", "plan.json", "--tag", "tag"])
        llama = parser.parse_args(
            ["--runtime-backend", "llama", "--plan", "plan.json", "--tag", "tag"]
        )
        self.assertEqual(default.runtime_backend, "strata")
        self.assertEqual(llama.runtime_backend, "llama")

    def test_llama_session_loads_once_and_cleans_owned_group(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "config.json"
            source.write_text("{}")
            manifest = root / "artifact.json"
            manifest.write_text("{}")
            config = SimpleNamespace(
                source_path=source,
                server_path=root / "llama-server",
                model_path=root / "model.gguf",
                model_name="qwen122",
                context=32768,
                lib_dirs=(root / "lib",),
                manifest_path=manifest,
                shards=(SimpleNamespace(sha256="a" * 64),),
            )
            plan_path = root / "plan.json"
            plan_path.write_text("{}")
            plan = model_session.Plan(
                plan_path,
                "plan-sha",
                "llama steps",
                (_step(), _step("medium", thinking="medium")),
            )
            args = argparse.Namespace(
                tag="session",
                runtime_backend="llama",
                cpus=None,
                startup_timeout=480.0,
                hold_after_plan=False,
                control_file=None,
            )
            events = _Events()
            handle = strata_trial.ServerHandle(_Process(), 9001, mock.Mock())
            with mock.patch.object(
                model_session.llama, "server_command", return_value=["llama-server"]
            ) as server_command, mock.patch.object(
                model_session.llama, "server_environment", return_value={"LD_LIBRARY_PATH": "/lib"}
            ), mock.patch.object(
                model_session.llama, "start_server", return_value=handle
            ) as start, mock.patch.object(
                model_session.llama,
                "wait_for_server",
                return_value={"data": [{"id": "qwen122"}]},
            ) as wait, mock.patch.object(
                model_session, "verify_server", return_value={"data": [{"id": "qwen122"}]}
            ) as health, mock.patch.object(
                model_session, "run_benchmark", return_value=(0, None)
            ) as benchmark, mock.patch.object(
                model_session.safe, "stop_server", return_value={"stopped": True}
            ) as stop, mock.patch.object(
                model_session.safe, "defer_termination_signals", side_effect=contextlib.nullcontext
            ):
                report, code = model_session.run_session(config, plan, args, events, results=root)

        self.assertEqual(code, 0)
        self.assertEqual(report["runtime_backend"], "llama")
        self.assertIsNone(report["private_config"])
        self.assertEqual(report["server"]["artifact_manifest"], str(manifest))
        self.assertEqual(report["server"]["shard_sha256"], ["a" * 64])
        self.assertTrue(report["server"]["per_request_chat_template_thinking"])
        server_command.assert_called_once_with(config, "medium")
        start.assert_called_once_with(
            config,
            ["llama-server"],
            {"LD_LIBRARY_PATH": "/lib"},
            root / "session-server.log",
        )
        wait.assert_called_once_with(handle, "qwen122", timeout=480.0)
        self.assertEqual(benchmark.call_count, 2)
        commands = [call.args[0] for call in benchmark.call_args_list]
        self.assertEqual(commands[0][commands[0].index("--chat-template-thinking") + 1], "off")
        self.assertEqual(commands[1][commands[1].index("--chat-template-thinking") + 1], "on")
        self.assertEqual(health.call_count, 4)
        self.assertTrue(all(call.args[2] == "llama" for call in health.call_args_list))
        stop.assert_called_once_with(handle)

    def test_llama_result_paths_omit_strata_private_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan = model_session.Plan(root / "plan.json", "sha", None, (_step(),))
            paths = model_session.result_paths(
                "session",
                plan,
                root,
                runtime_backend="llama",
            )
        names = {path.name for path in paths}
        self.assertNotIn("session-config.json", names)
        self.assertNotIn("session-config.shared-settings.json", names)
        self.assertNotIn("session-engine.log", names)
        self.assertIn("session-server.log", names)

    def test_main_dispatches_llama_without_writing_strata_private_config(self):
        config = SimpleNamespace(
            source_path=Path("/configs/llama.json"),
            server_path=Path("/engine/llama-server"),
            model_path=Path("/models/model.gguf"),
            model_name="qwen122",
            context=32768,
            lib_dirs=(Path("/runtime/lib"),),
            manifest_path=Path("/models/artifact.json"),
            shards=(SimpleNamespace(sha256="a" * 64),),
        )
        plan = model_session.Plan(
            Path("/plans/plan.json"), "plan-sha", None, (_step(),)
        )
        events = _Events()
        with mock.patch.object(model_session, "load_plan", return_value=plan), mock.patch.object(
            model_session.llama, "load_trial_config", return_value=config
        ) as load_llama, mock.patch.object(
            model_session.safe, "load_trial_config"
        ) as load_strata, mock.patch.object(
            model_session.llama, "server_command", return_value=["llama-server"]
        ), mock.patch.object(
            model_session.llama, "assert_preflight_clear", return_value=[]
        ), mock.patch.object(
            model_session.llama,
            "cuda_backend_path",
            return_value=Path("/runtime/cuda_v13/libggml-cuda.so"),
        ), mock.patch.object(
            model_session, "assert_results_unused"
        ) as unused, mock.patch.object(
            model_session.safe, "write_private_config"
        ) as write_private, mock.patch.object(
            model_session.safe, "EventLog", return_value=events
        ), mock.patch.object(
            model_session.safe, "install_signal_handlers"
        ), mock.patch.object(
            model_session,
            "run_session",
            return_value=({"status": "passed"}, 0),
        ) as run, mock.patch.object(model_session, "write_report"):
            code = model_session.main(
                [
                    "--runtime-backend",
                    "llama",
                    "--config",
                    "runtime/qwen122-32k.json",
                    "--plan",
                    "plan.json",
                    "--tag",
                    "llama-session",
                ]
            )

        self.assertEqual(code, 0)
        load_llama.assert_called_once_with(Path("runtime/qwen122-32k.json"))
        load_strata.assert_not_called()
        write_private.assert_not_called()
        unused.assert_called_once_with(
            "llama-session",
            plan,
            control_file=None,
            runtime_backend="llama",
        )
        self.assertEqual(run.call_args.args[:3], (config, plan, mock.ANY))
        started = next(fields for event, fields in events.rows if event == "session_started")
        self.assertEqual(started["runtime_backend"], "llama")
        self.assertIsNone(started["private_config"])
        self.assertEqual(started["shard_sha256"], ["a" * 64])
        self.assertEqual(
            started["engine_environment"]["GGML_BACKEND_PATH"],
            "/runtime/cuda_v13/libggml-cuda.so",
        )

    def test_session_loads_once_and_continues_after_semantic_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "config.json"
            source.write_text("{}")
            config = SimpleNamespace(
                source_path=source,
                private_path=root / "private.json",
                engine_path=root / "engine",
                model_path=root / "model",
                tokenizer_path=root / "tokenizer",
                model_name="qwen",
                context=32768,
            )
            plan_path = root / "plan.json"
            plan_path.write_text("{}")
            plan = model_session.Plan(
                plan_path,
                "plan-sha",
                "two steps",
                (_step(), _step("medium", thinking="medium")),
            )
            args = argparse.Namespace(
                tag="session",
                cpus=None,
                startup_timeout=480.0,
                hold_after_plan=False,
                control_file=None,
            )
            events = _Events()
            handle = strata_trial.ServerHandle(_Process(), 9001, mock.Mock())
            with mock.patch.object(model_session.safe, "server_command", return_value=["server"]), mock.patch.object(
                model_session.safe, "start_server", return_value=handle
            ) as start, mock.patch.object(
                model_session.safe,
                "wait_for_server",
                return_value={"data": [{"id": "qwen"}]},
            ), mock.patch.object(
                model_session, "verify_server", return_value={"data": [{"id": "qwen"}]}
            ) as health, mock.patch.object(
                model_session, "run_benchmark", side_effect=[(1, None), (0, None)]
            ) as benchmark, mock.patch.object(
                model_session.safe, "stop_server", return_value={"stopped": True}
            ) as stop, mock.patch.object(
                model_session.safe, "defer_termination_signals", side_effect=contextlib.nullcontext
            ):
                report, code = model_session.run_session(config, plan, args, events, results=root)

        self.assertEqual(code, 1)
        self.assertEqual(report["status"], "completed_with_step_failures")
        self.assertEqual([item["returncode"] for item in report["steps"]], [1, 0])
        self.assertEqual(start.call_count, 1)
        self.assertEqual(benchmark.call_count, 2)
        self.assertEqual(health.call_count, 4)
        stop.assert_called_once_with(handle)
        self.assertIsNotNone(report["server"]["initial_load_seconds"])

    def test_infrastructure_loss_stops_later_steps_and_cleans_server(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "config.json"
            source.write_text("{}")
            config = SimpleNamespace(
                source_path=source,
                private_path=root / "private.json",
                engine_path=root / "engine",
                model_path=root / "model",
                tokenizer_path=root / "tokenizer",
                model_name="qwen",
                context=32768,
            )
            plan_path = root / "plan.json"
            plan_path.write_text("{}")
            plan = model_session.Plan(
                plan_path, "sha", None, (_step(), _step("second"))
            )
            args = argparse.Namespace(
                tag="session",
                cpus=None,
                startup_timeout=480.0,
                hold_after_plan=False,
                control_file=None,
            )
            events = _Events()
            handle = strata_trial.ServerHandle(_Process(), 9001, mock.Mock())
            with mock.patch.object(model_session.safe, "server_command", return_value=["server"]), mock.patch.object(
                model_session.safe, "start_server", return_value=handle
            ), mock.patch.object(
                model_session.safe,
                "wait_for_server",
                return_value={"data": [{"id": "qwen"}]},
            ), mock.patch.object(
                model_session,
                "verify_server",
                side_effect=[{}, model_session.SessionError("lost")],
            ), mock.patch.object(
                model_session, "run_benchmark", return_value=(0, None)
            ) as benchmark, mock.patch.object(
                model_session.safe, "stop_server", return_value={"stopped": True}
            ) as stop, mock.patch.object(
                model_session.safe, "defer_termination_signals", side_effect=contextlib.nullcontext
            ):
                report, code = model_session.run_session(config, plan, args, events, results=root)

        self.assertEqual(code, 1)
        self.assertEqual(report["status"], "infrastructure_failed")
        self.assertEqual(benchmark.call_count, 1)
        stop.assert_called_once_with(handle)

    def test_control_file_and_hold_stop_are_bounded_and_cleanup_ready(self):
        with tempfile.TemporaryDirectory() as temporary:
            results = Path(temporary)
            control = results / "control.json"
            model_session.initialize_control_file(control)
            self.assertFalse(model_session.read_control_file(control))
            self.assertEqual(control.stat().st_mode & 0o777, 0o600)
            control.write_text('{"stop":true}\n')
            events = _Events()
            handle = strata_trial.ServerHandle(_Process(), 9001, mock.Mock())
            with mock.patch.object(model_session, "verify_server") as health:
                elapsed = model_session.hold_until_stopped(handle, control, events)
            self.assertGreaterEqual(elapsed, 0)
            health.assert_not_called()
            self.assertEqual(events.rows[-1][0], "holding_stopped")

    def test_hold_is_signal_responsive(self):
        with tempfile.TemporaryDirectory() as temporary:
            control = Path(temporary) / "control.json"
            control.write_text('{"stop":false}\n')
            events = _Events()
            handle = strata_trial.ServerHandle(_Process(), 9001, mock.Mock())
            with mock.patch.object(model_session, "verify_server"), mock.patch.object(
                model_session.time, "sleep", side_effect=KeyboardInterrupt("signal 15")
            ):
                with self.assertRaises(KeyboardInterrupt):
                    model_session.hold_until_stopped(handle, control, events)

    def test_signal_during_hold_still_cleans_owned_server(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "config.json"
            source.write_text("{}")
            plan_path = root / "plan.json"
            plan_path.write_text("{}")
            control = root / "control.json"
            control.write_text('{"stop":false}\n')
            config = SimpleNamespace(
                source_path=source,
                private_path=root / "private.json",
                engine_path=root / "engine",
                model_path=root / "model",
                tokenizer_path=root / "tokenizer",
                model_name="qwen",
                context=32768,
            )
            plan = model_session.Plan(plan_path, "sha", None, (_step(),))
            args = argparse.Namespace(
                tag="session",
                cpus=None,
                startup_timeout=480.0,
                hold_after_plan=True,
                control_file=control,
            )
            events = _Events()
            handle = strata_trial.ServerHandle(_Process(), 9001, mock.Mock())
            with mock.patch.object(model_session.safe, "server_command", return_value=["server"]), mock.patch.object(
                model_session.safe, "start_server", return_value=handle
            ), mock.patch.object(
                model_session.safe,
                "wait_for_server",
                return_value={"data": [{"id": "qwen"}]},
            ), mock.patch.object(
                model_session, "verify_server", return_value={"data": [{"id": "qwen"}]}
            ), mock.patch.object(
                model_session, "run_benchmark", return_value=(0, None)
            ), mock.patch.object(
                model_session, "hold_until_stopped", side_effect=KeyboardInterrupt("signal 15")
            ), mock.patch.object(
                model_session.safe, "stop_server", return_value={"stopped": True}
            ) as stop, mock.patch.object(
                model_session.safe, "defer_termination_signals", side_effect=contextlib.nullcontext
            ):
                report, code = model_session.run_session(config, plan, args, events, results=root)

        self.assertEqual(code, 130)
        self.assertEqual(report["status"], "interrupted")
        stop.assert_called_once_with(handle)

    def test_control_path_stays_directly_under_results(self):
        with tempfile.TemporaryDirectory() as temporary:
            results = Path(temporary).resolve()
            self.assertEqual(
                model_session.validate_control_path(results / "hold.json", results),
                results / "hold.json",
            )
            for path in (results / "nested" / "hold.json", results / "hold.txt"):
                with self.assertRaises(model_session.SessionError):
                    model_session.validate_control_path(path, results)


if __name__ == "__main__":
    unittest.main()
