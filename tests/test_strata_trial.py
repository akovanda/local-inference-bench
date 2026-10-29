import argparse
import io
import json
import os
import signal
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from runtime import strata_trial


def _source_config(root: Path) -> Path:
    engine = root / "strata"
    model = root / "model.gguf"
    tokenizer = root / "tokenizer"
    engine.write_bytes(b"engine")
    engine.chmod(0o755)
    model.write_bytes(b"model")
    tokenizer.mkdir(exist_ok=True)
    (tokenizer / "vocab.json").write_text("{}")
    source = root / "source.json"
    source.write_text(
        json.dumps(
            {
                "exe": str(engine),
                "args": [
                    "--native",
                    str(model),
                    "--max-context",
                    "32768",
                    "--pool-workers",
                    "31",
                ],
                "cwd": str(root),
                "tokenizer": str(tokenizer),
                "model_name": "qwen-test",
                "lib_dirs": ["/kept/cuda/lib"],
                "host": "0.0.0.0",
                "port": 9999,
                "open_browser": True,
            }
        )
    )
    return source


class _FakeEvents:
    def __init__(self):
        self.rows = []

    def write(self, event, **fields):
        self.rows.append((event, fields))


class StrataTrialTests(unittest.TestCase):
    def test_engine_environment_event_metadata_is_strictly_whitelisted(self):
        self.assertEqual(
            strata_trial.engine_environment_metadata(
                {"HOME": "/private", "TOKEN": "secret"}
            ),
            {},
        )
        self.assertEqual(
            strata_trial.engine_environment_metadata(
                {
                    "HOME": "/private",
                    "TOKEN": "secret",
                    "STRATA_SM75_PERF": "1",
                }
            ),
            {"STRATA_SM75_PERF": "1"},
        )
        self.assertEqual(
            strata_trial.engine_environment_metadata(
                {
                    "STRATA_ROUTE_TAIL_SKIP": "0",
                    "STRATA_ADAPT_OVERLAP": "1",
                    "STRATA_SECRET": "do-not-record",
                }
            ),
            {"STRATA_ROUTE_TAIL_SKIP": "0", "STRATA_ADAPT_OVERLAP": "1"},
        )

    def test_trial_started_records_only_actual_sm75_performance_flag(self):
        config = SimpleNamespace(
            source_path=Path("/configs/source.json"),
            private_path=Path("/results/private.json"),
            engine_path=Path("/engine/strata"),
            model_path=Path("/models/model.gguf"),
            tokenizer_path=Path("/tokenizer"),
            model_name="qwen-test",
            context=32768,
        )
        events = _FakeEvents()
        with mock.patch.object(strata_trial, "load_trial_config", return_value=config), mock.patch.object(
            strata_trial, "server_command", return_value=["server"]
        ), mock.patch.object(
            strata_trial, "assert_preflight_clear", return_value=[]
        ), mock.patch.object(
            strata_trial, "assert_results_unused"
        ), mock.patch.object(
            strata_trial, "write_private_config"
        ), mock.patch.object(
            strata_trial, "EventLog", return_value=events
        ), mock.patch.object(
            strata_trial, "install_signal_handlers"
        ), mock.patch.object(
            strata_trial, "run_trial", return_value=True
        ), mock.patch.dict(
            strata_trial.os.environ,
            {"STRATA_SM75_PERF": "1", "UNRELATED_SECRET": "do-not-record"},
            clear=True,
        ):
            code = strata_trial.main(
                ["--tag", "perf-fixture", "--startup-timeout", "420"]
            )

        self.assertEqual(code, 0)
        started = next(fields for event, fields in events.rows if event == "trial_started")
        self.assertEqual(started["engine_environment"], {"STRATA_SM75_PERF": "1"})
        self.assertEqual(started["startup_timeout"], 420.0)
        self.assertNotIn("UNRELATED_SECRET", json.dumps(started))

    def test_cli_defaults_preserve_quick_repeat_two(self):
        args = strata_trial.build_parser().parse_args(["--tag", "trial"])
        self.assertFalse(args.skip_quick)
        self.assertEqual(args.quick_repeat, 2)
        self.assertIsNone(args.hard_cases)
        self.assertEqual(args.output_tokens, 2048)
        self.assertEqual(args.request_timeout, 240.0)
        self.assertEqual(args.context_output_tokens, 768)
        self.assertEqual(args.context_timeout, 300.0)
        self.assertEqual(args.startup_timeout, 180.0)
        self.assertEqual(args.temperature, 0.0)
        self.assertEqual(args.seed, 0)
        self.assertEqual(args.hard_repeat, 1)

    def test_startup_timeout_accepts_only_bounded_finite_values(self):
        parser = strata_trial.build_parser()
        for value, expected in (("30", 30.0), ("240.5", 240.5), ("480", 480.0)):
            with self.subTest(value=value):
                self.assertEqual(
                    parser.parse_args(
                        ["--tag", "trial", "--startup-timeout", value]
                    ).startup_timeout,
                    expected,
                )
        for value in ("29.9", "480.1", "nan", "inf", "-inf"):
            with self.subTest(value=value), mock.patch(
                "sys.stderr", new=io.StringIO()
            ), self.assertRaises(SystemExit):
                parser.parse_args(["--tag", "trial", "--startup-timeout", value])

    def test_skip_quick_requires_an_explicit_nonquick_suite(self):
        with mock.patch.object(strata_trial, "load_trial_config") as load, mock.patch(
            "sys.stderr", new=io.StringIO()
        ), self.assertRaises(SystemExit) as raised:
            strata_trial.main(["--tag", "trial", "--skip-quick"])
        self.assertEqual(raised.exception.code, 2)
        load.assert_not_called()

    def test_private_config_overrides_only_trial_fields_and_pool_workers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results = root / "results"
            config = strata_trial.load_trial_config(
                _source_config(root),
                "trial",
                "low",
                15,
                results=results,
                server_root=root,
            )
            strata_trial.write_private_config(config)
            written = json.loads(config.private_path.read_text())

            self.assertEqual(written["host"], "127.0.0.1")
            self.assertEqual(written["port"], 18080)
            self.assertFalse(written["open_browser"])
            self.assertEqual(written["reasoning_effort"], "low")
            self.assertEqual(written["log"], str((results / "trial-engine.log").resolve()))
            self.assertEqual(written["lib_dirs"], ["/kept/cuda/lib"])
            self.assertEqual(written["args"].count("--pool-workers"), 1)
            index = written["args"].index("--pool-workers")
            self.assertEqual(written["args"][index + 1], "15")
            self.assertEqual(config.private_path.stat().st_mode & 0o777, 0o600)

    def test_config_refuses_missing_engine_model_or_tokenizer(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = _source_config(root)
            data = json.loads(source.read_text())
            Path(data["exe"]).unlink()
            with self.assertRaisesRegex(strata_trial.TrialError, "engine executable"):
                strata_trial.load_trial_config(source, "trial", "none", None, results=root / "r", server_root=root)

            _source_config(root)
            data = json.loads(source.read_text())
            Path(data["args"][1]).unlink()
            with self.assertRaisesRegex(strata_trial.TrialError, "model is missing"):
                strata_trial.load_trial_config(source, "trial", "none", None, results=root / "r", server_root=root)

            _source_config(root)
            data = json.loads(source.read_text())
            (Path(data["tokenizer"]) / "vocab.json").unlink()
            with self.assertRaisesRegex(strata_trial.TrialError, "tokenizer is missing"):
                strata_trial.load_trial_config(source, "trial", "none", None, results=root / "r", server_root=root)

    def test_server_argv_uses_taskset_without_shell(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            python = root / "python"
            taskset = root / "taskset"
            for path in (python, taskset):
                path.write_bytes(b"#!/bin/sh\n")
                path.chmod(0o755)
            command = strata_trial.server_command(
                root / "private.json",
                "0-7,16-23",
                python=python,
                taskset_path=str(taskset),
            )
        self.assertEqual(command[:3], [str(taskset), "--cpu-list", "0-7,16-23"])
        self.assertEqual(command[3:7], [str(python), "-m", "serve.server", "--engine"])
        self.assertEqual(command[-4:], ["--host", "127.0.0.1", "--port", "18080"])

    def test_start_uses_new_session_strata_cwd_and_preserved_environment(self):
        fake_process = SimpleNamespace(pid=4321)
        base_env = {
            "HOME": "/home/test",
            "CODEX_HOME": "/codex/test",
            "CUDA_VISIBLE_DEVICES": "2",
            "LD_LIBRARY_PATH": "/caller/lib",
        }
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            strata_trial.subprocess, "Popen", return_value=fake_process
        ) as popen:
            handle = strata_trial.start_server(
                ["/venv/python", "-m", "serve.server"],
                strata_trial.server_environment(base_env),
                Path(directory) / "server.log",
            )
            handle.log_handle.close()
        self.assertEqual(
            popen.call_args.args[0],
            strata_trial.unblocked_exec_command(["/venv/python", "-m", "serve.server"]),
        )
        self.assertEqual(popen.call_args.kwargs["cwd"], str(strata_trial.STRATA_ROOT))
        self.assertEqual(popen.call_args.kwargs["env"], base_env)
        self.assertTrue(popen.call_args.kwargs["start_new_session"])
        self.assertNotIn("shell", popen.call_args.kwargs)

    def test_preflight_checks_only_strata_port_and_refuses_large_gpu_process(self):
        with mock.patch.object(strata_trial, "port_is_occupied", return_value=True), mock.patch.object(
            strata_trial, "gpu_compute_processes"
        ) as gpu:
            with self.assertRaisesRegex(strata_trial.TrialError, "port 18080"):
                strata_trial.assert_preflight_clear()
            gpu.assert_not_called()

        with mock.patch.object(strata_trial, "port_is_occupied", return_value=False), mock.patch.object(
            strata_trial,
            "gpu_compute_processes",
            return_value=[{"pid": 73, "name": "other", "used_mib": 257}],
        ):
            with self.assertRaisesRegex(strata_trial.TrialError, "pid 73"):
                strata_trial.assert_preflight_clear()

    def test_startup_waits_for_configured_model_and_checks_process(self):
        response = mock.MagicMock()
        response.__enter__.return_value.read.return_value = json.dumps(
            {"object": "list", "data": [{"id": "qwen-test"}]}
        ).encode()
        process = mock.Mock()
        process.poll.side_effect = [None, None]
        handle = strata_trial.ServerHandle(process, 4321, io.BytesIO())

        with mock.patch.object(strata_trial, "urlopen", return_value=response) as open_url:
            payload = strata_trial.wait_for_server(handle, "qwen-test", timeout=1)

        self.assertEqual(payload["data"][0]["id"], "qwen-test")
        self.assertEqual(process.poll.call_count, 2)
        request = open_url.call_args.args[0]
        self.assertEqual(request.full_url, "http://127.0.0.1:18080/v1/models")

    def test_bench_commands_use_openai_scope_and_requested_thinking(self):
        config = SimpleNamespace(model_name="qwen-test", context=32768)
        quick = strata_trial.bench_command(
            config, "none", "quick", Path("quick.jsonl"), quick_repeat=1
        )
        quality = strata_trial.bench_command(config, "medium", "quality", Path("quality.jsonl"))
        context = strata_trial.bench_command(
            config,
            "low",
            "context",
            Path("context.jsonl"),
            context_target=28000,
        )
        self.assertIn("openai", quick)
        self.assertIn("http://127.0.0.1:18080/v1", quick)
        self.assertEqual(quick[-4:], ["--repeat", "1", "--num-predict", "512"])
        self.assertEqual(quality[-2:], ["--cases", strata_trial.QUALITY_CASES])
        self.assertEqual(context[-4:], ["--context-target", "28000", "--num-predict", "768"])
        self.assertIn("--fail-on-case", quick)
        self.assertIn("--fail-on-case", quality)
        self.assertIn("--fail-on-case", context)
        for command in (quick, quality, context):
            self.assertEqual(command[command.index("--temperature") + 1], "0")
            self.assertEqual(command[command.index("--seed") + 1], "0")

    def test_context_command_forwards_generation_cap_and_bounded_timeout(self):
        config = SimpleNamespace(model_name="qwen-test", context=32768)
        command = strata_trial.bench_command(
            config,
            "medium",
            "context",
            Path("context.jsonl"),
            context_target=20000,
            context_output_tokens=2048,
            context_timeout=480.0,
        )
        self.assertEqual(command[command.index("--context-target") + 1], "20000")
        self.assertEqual(command[command.index("--num-predict") + 1], "2048")
        self.assertEqual(command[command.index("--timeout") + 1], "480")
        self.assertEqual(
            strata_trial.suite_process_timeout("context", context_timeout=480.0),
            570.0,
        )

    def test_context_controls_reject_nonpositive_or_nonfinite_values(self):
        parser = strata_trial.build_parser()
        for option, value in (
            ("--context-output-tokens", "0"),
            ("--context-output-tokens", "-1"),
            ("--context-timeout", "0"),
            ("--context-timeout", "-1"),
            ("--context-timeout", "nan"),
            ("--context-timeout", "inf"),
        ):
            with self.subTest(option=option, value=value), mock.patch(
                "sys.stderr", new=io.StringIO()
            ), self.assertRaises(SystemExit):
                parser.parse_args(["--tag", "trial", option, value])

    def test_hard_command_uses_one_bounded_subset_call(self):
        config = SimpleNamespace(model_name="qwen-test", context=32768)
        command = strata_trial.bench_command(
            config,
            "none",
            "hard",
            Path("hard.jsonl"),
            hard_cases="hard_temporal_kv,hard_dependency_constraints",
            output_tokens=2048,
            request_timeout=240.0,
        )
        self.assertEqual(command[command.index("--suite") + 1], "hard")
        self.assertEqual(
            command[command.index("--cases") + 1],
            "hard_temporal_kv,hard_dependency_constraints",
        )
        self.assertEqual(command[command.index("--num-predict") + 1], "2048")
        self.assertEqual(command[command.index("--timeout") + 1], "240")
        self.assertEqual(command[command.index("--thinking") + 1], "none")
        self.assertIn("--fail-on-case", command)
        self.assertEqual(
            strata_trial.suite_process_timeout(
                "hard", hard_case_count=2, request_timeout=240.0
            ),
            570.0,
        )

    def test_hard_repeat_steps_seed_and_multiplies_outer_timeout(self):
        config = SimpleNamespace(model_name="qwen-test", context=32768)
        command = strata_trial.bench_command(
            config,
            "none",
            "hard",
            Path("hard.jsonl"),
            hard_cases="hard_transaction_ledger,hard_temporal_kv",
            request_timeout=120.0,
            temperature=0.7,
            seed=41,
            hard_repeat=3,
        )
        self.assertEqual(command[command.index("--temperature") + 1], "0.7")
        self.assertEqual(command[command.index("--seed") + 1], "41")
        self.assertEqual(command[command.index("--repeat") + 1], "3")
        self.assertEqual(command[command.index("--seed-step") + 1], "1")
        self.assertEqual(
            strata_trial.suite_process_timeout(
                "hard", hard_case_count=2, request_timeout=120.0, hard_repeat=3
            ),
            810.0,
        )

    def test_sampling_controls_are_bounded(self):
        parser = strata_trial.build_parser()
        for value in ("-0.1", "2.1", "nan", "inf", "-inf"):
            with self.subTest(temperature=value), mock.patch(
                "sys.stderr", new=io.StringIO()
            ), self.assertRaises(SystemExit):
                parser.parse_args(["--tag", "trial", "--temperature", value])
        for value in ("0", "4"):
            with self.subTest(hard_repeat=value), mock.patch(
                "sys.stderr", new=io.StringIO()
            ), self.assertRaises(SystemExit):
                parser.parse_args(["--tag", "trial", "--hard-repeat", value])
        self.assertEqual(parser.parse_args(["--tag", "trial", "--temperature", "2"]).temperature, 2.0)

    def test_hard_case_ids_are_strict_and_deduplicated(self):
        self.assertEqual(
            strata_trial.parse_case_ids("hard_transaction_ledger,Hard2"),
            "hard_transaction_ledger,Hard2",
        )
        for value in ("", "hard-a", "hard a", "a,,b", "a,a"):
            with self.subTest(value=value), self.assertRaises(argparse.ArgumentTypeError):
                strata_trial.parse_case_ids(value)

    def test_skip_quick_hard_subset_runs_as_one_suite(self):
        config = SimpleNamespace(
            private_path=Path("/tmp/private.json"),
            model_name="qwen-test",
            context=32768,
        )
        args = SimpleNamespace(
            cpus=None,
            tag="trial",
            thinking="medium",
            quality=False,
            context_target=None,
            quick_repeat=2,
            skip_quick=True,
            hard_cases="hard_temporal_kv,hard_dependency_constraints",
            output_tokens=4096,
            request_timeout=240.0,
            startup_timeout=360.0,
        )
        process = SimpleNamespace(pid=9000)
        handle = strata_trial.ServerHandle(process, 9000, io.BytesIO())
        events = _FakeEvents()
        with mock.patch.object(strata_trial, "server_command", return_value=["server"]), mock.patch.object(
            strata_trial, "start_server", return_value=handle
        ), mock.patch.object(
            strata_trial, "wait_for_server", return_value={"data": [{"id": "qwen-test"}]}
        ) as ready, mock.patch.object(strata_trial, "run_suite", return_value=0) as suite, mock.patch.object(
            strata_trial,
            "stop_server",
            return_value={"pgid": 9000, "term_sent": True, "kill_sent": False, "stopped": True},
        ):
            passed = strata_trial.run_trial(config, args, events)
        self.assertTrue(passed)
        ready.assert_called_once_with(handle, "qwen-test", timeout=360.0)
        suite.assert_called_once()
        call = suite.call_args
        self.assertEqual(call.args[1], "hard")
        self.assertEqual(call.kwargs["hard_case_count"], 2)
        self.assertEqual(call.kwargs["request_timeout"], 240.0)
        command = call.args[0]
        self.assertEqual(command[command.index("--num-predict") + 1], "4096")
        self.assertEqual(command[command.index("--cases") + 1], "hard_temporal_kv,hard_dependency_constraints")
        self.assertTrue(any(fields.get("result") == "results/trial-hard.jsonl" for _, fields in events.rows))

    def test_skip_quick_runs_only_requested_quality_and_context_suites(self):
        config = SimpleNamespace(
            private_path=Path("/tmp/private.json"),
            model_name="qwen-test",
            context=32768,
        )
        args = SimpleNamespace(
            cpus=None,
            tag="trial",
            thinking="none",
            quality=True,
            context_target=28000,
            context_output_tokens=2048,
            context_timeout=480.0,
            quick_repeat=1,
            skip_quick=True,
        )
        process = SimpleNamespace(pid=9000)
        handle = strata_trial.ServerHandle(process, 9000, io.BytesIO())
        events = _FakeEvents()
        with mock.patch.object(strata_trial, "server_command", return_value=["server"]), mock.patch.object(
            strata_trial, "start_server", return_value=handle
        ), mock.patch.object(
            strata_trial, "wait_for_server", return_value={"data": [{"id": "qwen-test"}]}
        ), mock.patch.object(strata_trial, "run_suite", return_value=0) as suite, mock.patch.object(
            strata_trial,
            "stop_server",
            return_value={"pgid": 9000, "term_sent": True, "kill_sent": False, "stopped": True},
        ):
            passed = strata_trial.run_trial(config, args, events)
        self.assertTrue(passed)
        self.assertEqual([call.args[1] for call in suite.call_args_list], ["quality", "context"])
        context_call = suite.call_args_list[1]
        context_command = context_call.args[0]
        self.assertEqual(context_command[context_command.index("--num-predict") + 1], "2048")
        self.assertEqual(context_command[context_command.index("--timeout") + 1], "480")
        self.assertEqual(context_call.kwargs["context_timeout"], 480.0)

    def test_skip_quick_result_set_omits_quick_artifact(self):
        paths = strata_trial.result_paths("trial", True, 28000, skip_quick=True)
        names = {path.name for path in paths}
        self.assertNotIn("trial-quick.jsonl", names)
        self.assertIn("trial-quality.jsonl", names)
        self.assertIn("trial-context.jsonl", names)

    def test_hard_result_is_reserved_without_quick(self):
        paths = strata_trial.result_paths(
            "trial", False, None, skip_quick=True, hard_cases="hard_transaction_ledger"
        )
        names = {path.name for path in paths}
        self.assertNotIn("trial-quick.jsonl", names)
        self.assertIn("trial-hard.jsonl", names)

    def test_cleanup_escalates_only_the_spawned_process_group(self):
        process = mock.Mock()
        process.wait.return_value = -signal.SIGKILL
        handle = strata_trial.ServerHandle(process, 7777, io.BytesIO())
        with mock.patch.object(strata_trial, "_group_exists", return_value=True), mock.patch.object(
            strata_trial, "_wait_group_gone", side_effect=[False, True]
        ), mock.patch.object(strata_trial.os, "killpg") as killpg:
            result = strata_trial.stop_server(handle)
        self.assertEqual(
            killpg.call_args_list,
            [mock.call(7777, signal.SIGTERM), mock.call(7777, signal.SIGKILL)],
        )
        self.assertTrue(result["stopped"])
        self.assertTrue(result["kill_sent"])


if __name__ == "__main__":
    unittest.main()
