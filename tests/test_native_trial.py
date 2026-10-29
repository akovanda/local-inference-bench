import argparse
import io
import json
import os
import signal
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from runtime import native_trial


def _descriptor(binary):
    return {
        "schema_version": 1,
        "ollama_version": "0.40.6",
        "launch": {
            "argv": [str(binary), "serve"],
            "env": {
                "LD_LIBRARY_PATH": "/runtime/lib",
                "OLLAMA_HOST": "127.0.0.1:11440",
                "OLLAMA_MODELS": "/models/unchanged",
                "HOME": "/descriptor-must-not-win",
                "CODEX_HOME": "/descriptor-must-not-win",
            },
        },
    }


class _FakeEvents:
    def __init__(self):
        self.rows = []

    def write(self, event, **fields):
        self.rows.append((event, fields))


class NativeTrialTests(unittest.TestCase):
    def test_cli_defaults_preserve_quick_low_thinking(self):
        args = native_trial.build_parser().parse_args(
            [
                "--runtime",
                "/runtime/runtime.json",
                "--tag",
                "trial",
                "--cache",
                "f16",
                "--contexts",
                "32768",
            ]
        )
        self.assertFalse(args.skip_quick)
        self.assertEqual(args.quick_repeat, 2)
        self.assertEqual(args.thinking, "low")
        self.assertIsNone(args.hard_cases)
        self.assertEqual(args.output_tokens, 2048)
        self.assertEqual(args.request_timeout, 240.0)

    def test_skip_quick_requires_an_explicit_nonquick_suite(self):
        argv = [
            "--runtime",
            "/runtime/runtime.json",
            "--tag",
            "trial",
            "--cache",
            "f16",
            "--contexts",
            "32768",
            "--skip-quick",
        ]
        with mock.patch.object(native_trial, "load_descriptor") as load, mock.patch(
            "sys.stderr", new=io.StringIO()
        ), self.assertRaises(SystemExit) as raised:
            native_trial.main(argv)
        self.assertEqual(raised.exception.code, 2)
        load.assert_not_called()

    def test_descriptor_launch_and_environment_preserve_home_and_model_store(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary = root / "ollama"
            binary.write_bytes(b"#!/bin/sh\nexit 0\n")
            binary.chmod(0o755)
            descriptor_path = root / "runtime.json"
            descriptor_path.write_text(json.dumps(_descriptor(binary)))

            descriptor = native_trial.load_descriptor(descriptor_path)
            env = native_trial.build_server_env(
                descriptor,
                "q8_0",
                65536,
                {"HOME": "/real-home", "CODEX_HOME": "/real-codex", "PATH": "/bin"},
            )
            self.assertEqual(descriptor["launch"]["argv"], [str(binary), "serve"])
            self.assertEqual(env["HOME"], "/real-home")
            self.assertEqual(env["CODEX_HOME"], "/real-codex")
            self.assertEqual(env["OLLAMA_MODELS"], "/models/unchanged")
            self.assertEqual(env["OLLAMA_KV_CACHE_TYPE"], "q8_0")
            self.assertEqual(env["OLLAMA_CONTEXT_LENGTH"], "65536")
            self.assertEqual(env["OLLAMA_FLASH_ATTENTION"], "1")
            self.assertEqual(env["OLLAMA_NUM_PARALLEL"], "1")
            self.assertEqual(env["OLLAMA_KEEP_ALIVE"], "10m")
            self.assertEqual(env["OLLAMA_NOPRUNE"], "1")
            self.assertEqual(env["OLLAMA_NO_CLOUD"], "1")
            self.assertEqual(env["OLLAMA_HOST"], "127.0.0.1:11440")

    def test_preflight_refuses_occupied_port_or_large_compute_process(self):
        with mock.patch.object(native_trial, "port_is_occupied", return_value=True), mock.patch.object(
            native_trial, "gpu_compute_processes"
        ) as gpu:
            with self.assertRaisesRegex(native_trial.TrialError, "port 11440 is occupied"):
                native_trial.assert_preflight_clear()
            gpu.assert_not_called()

        with mock.patch.object(native_trial, "port_is_occupied", return_value=False), mock.patch.object(
            native_trial,
            "gpu_compute_processes",
            return_value=[{"pid": 73, "name": "other", "used_mib": 257}],
        ):
            with self.assertRaisesRegex(native_trial.TrialError, "pid 73"):
                native_trial.assert_preflight_clear()

    def test_gpu_query_parses_only_compute_process_rows(self):
        completed = SimpleNamespace(
            returncode=0,
            stdout="101, python, 128\n202, ollama, 512\n",
            stderr="",
        )
        with mock.patch.object(native_trial.subprocess, "run", return_value=completed) as run:
            processes = native_trial.gpu_compute_processes()
        self.assertEqual(processes[1], {"pid": 202, "name": "ollama", "used_mib": 512})
        command = run.call_args.args[0]
        self.assertIn("--query-compute-apps=pid,process_name,used_gpu_memory", command)

    def test_commands_are_argument_lists_with_required_benchmark_scope(self):
        quick = native_trial.bench_command(32768, "quick", Path("quick.jsonl"))
        quality = native_trial.bench_command(32768, "quality", Path("quality.jsonl"))
        context = native_trial.bench_command(
            65536,
            "context",
            Path("context.jsonl"),
            context_target=60000,
            thinking="medium",
        )
        probe = native_trial.probe_command(65536, Path("health.json"), thinking="medium")
        self.assertEqual(quick[-4:], ["--repeat", "2", "--num-predict", "512"])
        self.assertEqual(quality[-2:], ["--cases", native_trial.QUALITY_CASES])
        self.assertEqual(context[-4:], ["--context-target", "60000", "--num-predict", "768"])
        self.assertEqual(context[context.index("--thinking") + 1], "medium")
        self.assertEqual(probe[probe.index("--thinking") + 1], "medium")
        self.assertIn("--fail-on-case", quick)
        self.assertIn("--fail-on-case", quality)
        self.assertIn("--fail-on-case", context)
        self.assertNotIn("shell=True", quick)

    def test_hard_command_uses_one_bounded_subset_call(self):
        command = native_trial.bench_command(
            32768,
            "hard",
            Path("hard.jsonl"),
            thinking="medium",
            hard_cases="hard_transaction_ledger,hard_weighted_intervals",
            output_tokens=2048,
            request_timeout=270.0,
        )
        self.assertEqual(command[command.index("--suite") + 1], "hard")
        self.assertEqual(
            command[command.index("--cases") + 1],
            "hard_transaction_ledger,hard_weighted_intervals",
        )
        self.assertEqual(command[command.index("--num-predict") + 1], "2048")
        self.assertEqual(command[command.index("--timeout") + 1], "270")
        self.assertEqual(command[command.index("--thinking") + 1], "medium")
        self.assertIn("--fail-on-case", command)
        self.assertEqual(
            native_trial.suite_process_timeout(
                "hard", hard_case_count=2, request_timeout=270.0
            ),
            630.0,
        )

    def test_hard_case_ids_are_strict_and_deduplicated(self):
        self.assertEqual(
            native_trial.parse_case_ids("hard_transaction_ledger,Hard2"),
            "hard_transaction_ledger,Hard2",
        )
        for value in ("", "hard-a", "hard a", "a,,b", "a,a"):
            with self.subTest(value=value), self.assertRaises(argparse.ArgumentTypeError):
                native_trial.parse_case_ids(value)

    def test_skip_quick_hard_subset_runs_as_one_suite(self):
        fake_process = SimpleNamespace(pid=9000)
        handle = native_trial.ServerHandle(fake_process, 9000, io.BytesIO())
        events = _FakeEvents()
        descriptor = {"ollama_version": "0.40.6"}
        with mock.patch.object(native_trial, "assert_preflight_clear", return_value=[]), mock.patch.object(
            native_trial, "build_server_env", return_value={}
        ), mock.patch.object(native_trial, "start_server", return_value=handle), mock.patch.object(
            native_trial, "wait_for_server", return_value={"version": "0.40.6"}
        ), mock.patch.object(
            native_trial, "run_probe", return_value={"passed": True}
        ), mock.patch.object(native_trial, "run_suite", return_value=0) as suite, mock.patch.object(
            native_trial,
            "stop_server",
            return_value={"pgid": 9000, "term_sent": True, "kill_sent": False, "stopped": True},
        ):
            passed = native_trial.run_context(
                descriptor,
                "tag",
                "q8_0",
                32768,
                False,
                None,
                "run-id",
                events,
                skip_quick=True,
                thinking="medium",
                hard_cases="hard_transaction_ledger,hard_weighted_intervals",
                output_tokens=4096,
                request_timeout=270.0,
            )
        self.assertTrue(passed)
        suite.assert_called_once()
        call = suite.call_args
        self.assertEqual(call.args[1], "hard")
        self.assertEqual(call.kwargs["hard_case_count"], 2)
        self.assertEqual(call.kwargs["request_timeout"], 270.0)
        command = call.args[0]
        self.assertEqual(command[command.index("--num-predict") + 1], "4096")
        self.assertEqual(command[command.index("--cases") + 1], "hard_transaction_ledger,hard_weighted_intervals")
        self.assertTrue(any(fields.get("result") == "results/tag-q8_0-32k-hard.jsonl" for _, fields in events.rows))

    def test_skip_quick_runs_selected_suites_after_medium_probe(self):
        fake_process = SimpleNamespace(pid=9000)
        handle = native_trial.ServerHandle(fake_process, 9000, io.BytesIO())
        events = _FakeEvents()
        descriptor = {"ollama_version": "0.40.6"}
        with mock.patch.object(native_trial, "assert_preflight_clear", return_value=[]), mock.patch.object(
            native_trial, "build_server_env", return_value={}
        ), mock.patch.object(native_trial, "start_server", return_value=handle), mock.patch.object(
            native_trial, "wait_for_server", return_value={"version": "0.40.6"}
        ), mock.patch.object(
            native_trial, "run_probe", return_value={"passed": True}
        ) as probe, mock.patch.object(native_trial, "run_suite", return_value=0) as suite, mock.patch.object(
            native_trial,
            "stop_server",
            return_value={"pgid": 9000, "term_sent": True, "kill_sent": False, "stopped": True},
        ):
            passed = native_trial.run_context(
                descriptor,
                "tag",
                "f16",
                32768,
                True,
                28000,
                "run-id",
                events,
                quick_repeat=1,
                skip_quick=True,
                thinking="medium",
            )
        self.assertTrue(passed)
        probe.assert_called_once_with(
            32768,
            native_trial.RESULTS / "tag-f16-32k-health.json",
            "run-id",
            thinking="medium",
        )
        self.assertEqual([call.args[1] for call in suite.call_args_list], ["quality", "context"])
        for call in suite.call_args_list:
            command = call.args[0]
            self.assertEqual(command[command.index("--thinking") + 1], "medium")

    def test_start_uses_descriptor_argv_and_a_new_session(self):
        descriptor = {"launch": {"argv": ["/runtime/ollama", "serve"]}}
        fake_process = SimpleNamespace(pid=4321)
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            native_trial.subprocess, "Popen", return_value=fake_process
        ) as popen:
            handle = native_trial.start_server(
                descriptor, {"HOME": "/home/test"}, Path(directory) / "server.log"
            )
            handle.log_handle.close()
        self.assertEqual(
            popen.call_args.args[0],
            native_trial.unblocked_exec_command(["/runtime/ollama", "serve"]),
        )
        self.assertTrue(popen.call_args.kwargs["start_new_session"])
        self.assertNotIn("shell", popen.call_args.kwargs)
        self.assertEqual(handle.pgid, 4321)

    def test_false_probe_result_skips_bench_and_still_cleans_own_group(self):
        fake_process = SimpleNamespace(pid=9001)
        handle = native_trial.ServerHandle(fake_process, 9001, io.BytesIO())
        events = _FakeEvents()
        descriptor = {"ollama_version": "0.40.6"}
        with mock.patch.object(native_trial, "assert_preflight_clear", return_value=[]), mock.patch.object(
            native_trial, "build_server_env", return_value={}
        ), mock.patch.object(native_trial, "start_server", return_value=handle), mock.patch.object(
            native_trial, "wait_for_server", return_value={"version": "0.40.6"}
        ), mock.patch.object(native_trial, "run_probe", return_value={"passed": False}), mock.patch.object(
            native_trial, "run_suite"
        ) as suite, mock.patch.object(
            native_trial,
            "stop_server",
            return_value={"pgid": 9001, "term_sent": True, "kill_sent": False, "stopped": True},
        ) as stop:
            passed = native_trial.run_context(
                descriptor, "tag", "f16", 32768, True, 28000, "run-id", events
            )
        self.assertFalse(passed)
        suite.assert_not_called()
        stop.assert_called_once_with(handle)

    def test_cleanup_escalates_only_the_created_process_group(self):
        process = mock.Mock()
        process.wait.return_value = -signal.SIGKILL
        handle = native_trial.ServerHandle(process, 7777, io.BytesIO())
        with mock.patch.object(native_trial, "_group_exists", return_value=True), mock.patch.object(
            native_trial, "_wait_group_gone", side_effect=[False, True]
        ), mock.patch.object(native_trial.os, "killpg") as killpg:
            result = native_trial.stop_server(handle)
        self.assertEqual(
            killpg.call_args_list,
            [mock.call(7777, signal.SIGTERM), mock.call(7777, signal.SIGKILL)],
        )
        self.assertTrue(result["stopped"])
        self.assertTrue(result["kill_sent"])


if __name__ == "__main__":
    unittest.main()
