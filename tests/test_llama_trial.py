import argparse
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from runtime import llama_trial, prepare_qwen122


def _config(root: Path) -> Path:
    server = root / "llama-server"
    server.write_bytes(b"#!/bin/sh\n")
    server.chmod(0o755)
    lib = root / "lib"
    cuda = root / "cuda"
    lib.mkdir(exist_ok=True)
    cuda.mkdir(exist_ok=True)
    (cuda / "libggml-cuda.so").write_bytes(b"cuda backend")
    shard_specs = (
        ("model-00001-of-00002.gguf", b"first", "a" * 64),
        ("model-00002-of-00002.gguf", b"second", "b" * 64),
    )
    rows = []
    manifest_rows = []
    for name, data, digest in shard_specs:
        (root / name).write_bytes(data)
        row = {"name": name, "size": len(data), "sha256": digest}
        rows.append(row)
        manifest_rows.append({**row, "verified": True})
    manifest = root / "artifact.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "verdict": {"artifact_complete": True},
                "shards": manifest_rows,
            }
        )
    )
    config = root / "config.json"
    config.write_text(
        json.dumps(
            {
                "server": str(server),
                "model": str(root / shard_specs[0][0]),
                "model_name": "qwen-test",
                "context": 32768,
                "host": "127.0.0.1",
                "port": 18081,
                "artifact_manifest": str(manifest),
                "lib_dirs": [str(lib), str(cuda)],
                "args": ["--cpu-moe", "--parallel", "1", "--jinja"],
                "shards": rows,
            }
        )
    )
    return config


class _FakeEvents:
    def __init__(self):
        self.rows = []

    def write(self, event, **fields):
        self.rows.append((event, fields))


class LlamaTrialTests(unittest.TestCase):
    def test_staging_manifest_pins_all_shards_and_reasoning_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            report = prepare_qwen122.artifact_report(Path(directory), set())
        self.assertEqual(report["model"]["revision"], prepare_qwen122.REVISION)
        self.assertEqual(report["model"]["expected_total_size"], 77_029_996_032)
        self.assertEqual(len(report["shards"]), 3)
        self.assertFalse(report["verdict"]["artifact_complete"])
        self.assertIn("enable_thinking false/true", report["llama_cpp"]["reasoning_semantics"])

    def test_staging_can_resume_at_shard_with_unchanged_verified_prefix(self):
        shards = (
            ("one.gguf", 3, "a" * 64),
            ("two.gguf", 3, "b" * 64),
            ("three.gguf", 5, "c" * 64),
        )
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            prepare_qwen122, "SHARDS", shards
        ):
            root = Path(directory)
            (root / "one.gguf").write_bytes(b"one")
            (root / "two.gguf").write_bytes(b"two")
            report = root / "artifact.json"
            prepare_qwen122.atomic_write_json(
                report,
                prepare_qwen122.artifact_report(root, {"one.gguf", "two.gguf"}),
            )

            verified, provenance = prepare_qwen122.verified_prefix(root, report, 3)
            self.assertEqual(verified, {"one.gguf", "two.gguf"})
            self.assertEqual(provenance["start_shard"], 3)
            self.assertEqual([row["sha256"] for row in provenance["files"]], ["a" * 64, "b" * 64])
            self.assertTrue(all(row["inode"] > 0 for row in provenance["files"]))

            (root / "two.gguf").write_bytes(b"TWO")
            with self.assertRaisesRegex(
                prepare_qwen122.range_fetch.IntegrityError,
                "changed after its receipt",
            ):
                prepare_qwen122.verified_prefix(root, report, 3)

    def test_cli_defaults_and_case_validation(self):
        args = llama_trial.build_parser().parse_args(["--tag", "trial"])
        self.assertEqual(args.thinking, "none")
        self.assertEqual(args.quick_repeat, 1)
        self.assertEqual(args.output_tokens, 2048)
        self.assertEqual(args.request_timeout, 270.0)
        self.assertEqual(args.startup_timeout, 180.0)
        self.assertIsNone(args.hard_cases)
        self.assertEqual(
            llama_trial.parse_case_ids("hard_temporal_kv,hard_dependency_constraints"),
            "hard_temporal_kv,hard_dependency_constraints",
        )
        for value in ("", "hard-case", "a,,b", "a,a"):
            with self.subTest(value=value), self.assertRaises(argparse.ArgumentTypeError):
                llama_trial.parse_case_ids(value)

    def test_startup_timeout_accepts_only_bounded_finite_values(self):
        parser = llama_trial.build_parser()
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

    def test_trial_started_records_requested_startup_timeout(self):
        config = SimpleNamespace(
            source_path=Path("/configs/source.json"),
            server_path=Path("/engine/llama-server"),
            model_path=Path("/models/model.gguf"),
            model_name="qwen-test",
            context=32768,
            manifest_path=Path("/models/artifact.json"),
            shards=(SimpleNamespace(sha256="a" * 64),),
        )
        events = _FakeEvents()
        with mock.patch.object(llama_trial, "load_trial_config", return_value=config), mock.patch.object(
            llama_trial, "server_command", return_value=["server"]
        ), mock.patch.object(
            llama_trial, "assert_preflight_clear", return_value=[]
        ), mock.patch.object(
            llama_trial, "assert_results_unused"
        ), mock.patch.object(
            llama_trial, "EventLog", return_value=events
        ), mock.patch.object(
            llama_trial.safe, "install_signal_handlers"
        ), mock.patch.object(
            llama_trial, "run_trial", return_value=True
        ):
            code = llama_trial.main(
                ["--tag", "startup-fixture", "--startup-timeout", "420"]
            )

        self.assertEqual(code, 0)
        started = next(fields for event, fields in events.rows if event == "trial_started")
        self.assertEqual(started["startup_timeout"], 420.0)

    def test_skip_quick_requires_hard_cases_before_config_load(self):
        with mock.patch.object(llama_trial, "load_trial_config") as load, mock.patch(
            "sys.stderr", new=io.StringIO()
        ), self.assertRaises(SystemExit) as raised:
            llama_trial.main(["--tag", "trial", "--skip-quick"])
        self.assertEqual(raised.exception.code, 2)
        load.assert_not_called()

    def test_load_config_checks_shards_manifest_and_preserves_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            config = llama_trial.load_trial_config(_config(Path(directory)))
            environment = llama_trial.server_environment(
                config,
                {
                    "HOME": "/home/test",
                    "CODEX_HOME": "/codex/test",
                    "CUDA_VISIBLE_DEVICES": "2",
                    "LD_LIBRARY_PATH": "/caller/lib",
                },
            )
        self.assertEqual(config.model_name, "qwen-test")
        self.assertEqual(len(config.shards), 2)
        self.assertEqual(environment["HOME"], "/home/test")
        self.assertEqual(environment["CODEX_HOME"], "/codex/test")
        self.assertEqual(environment["CUDA_VISIBLE_DEVICES"], "2")
        self.assertTrue(environment["LD_LIBRARY_PATH"].endswith(":/caller/lib"))
        self.assertIn("/lib:/", environment["LD_LIBRARY_PATH"])
        self.assertEqual(
            environment["GGML_BACKEND_PATH"],
            str((config.lib_dirs[1] / "libggml-cuda.so").resolve()),
        )

    def test_server_environment_requires_one_cuda_backend_and_overrides_caller_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = llama_trial.load_trial_config(_config(root))
            environment = llama_trial.server_environment(
                config,
                {"GGML_BACKEND_PATH": "/caller/untrusted.so"},
            )
            self.assertEqual(
                environment["GGML_BACKEND_PATH"],
                str((root / "cuda/libggml-cuda.so").resolve()),
            )

            (root / "cuda/libggml-cuda.so").unlink()
            with self.assertRaisesRegex(
                llama_trial.TrialError,
                "expected exactly one libggml-cuda.so",
            ):
                llama_trial.server_environment(config, {})

            (root / "cuda/libggml-cuda.so").write_bytes(b"cuda backend")
            (root / "lib/libggml-cuda.so").write_bytes(b"duplicate")
            with self.assertRaisesRegex(
                llama_trial.TrialError,
                "found 2",
            ):
                llama_trial.server_environment(config, {})

    def test_config_rejects_unverified_or_wrong_sized_shard_and_owned_option(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = _config(root)
            manifest = json.loads((root / "artifact.json").read_text())
            manifest["shards"][1]["verified"] = False
            (root / "artifact.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(llama_trial.TrialError, "does not verify shard"):
                llama_trial.load_trial_config(path)

            _config(root)
            (root / "model-00002-of-00002.gguf").write_bytes(b"wrong")
            with self.assertRaisesRegex(llama_trial.TrialError, "wrong size"):
                llama_trial.load_trial_config(path)

            _config(root)
            source = json.loads(path.read_text())
            source["args"].extend(("--port", "9999"))
            path.write_text(json.dumps(source))
            with self.assertRaisesRegex(llama_trial.TrialError, "runner-owned option --port"):
                llama_trial.load_trial_config(path)

    def test_server_command_maps_thinking_to_template_controls(self):
        with tempfile.TemporaryDirectory() as directory:
            config = llama_trial.load_trial_config(_config(Path(directory)))
            none = llama_trial.server_command(config, "none")
            medium = llama_trial.server_command(config, "medium")
        self.assertEqual(none[none.index("--reasoning") + 1], "off")
        self.assertNotIn("--reasoning-effort", none)
        self.assertEqual(medium[medium.index("--reasoning") + 1], "on")
        self.assertEqual(medium[medium.index("--reasoning-effort") + 1], "medium")
        self.assertEqual(none[none.index("--host") + 1], "127.0.0.1")
        self.assertEqual(none[none.index("--port") + 1], "18081")
        self.assertEqual(none[none.index("--model") + 1], str(config.model_path))

    def test_eagle3_activation_requires_verified_draft_bound_to_exact_target(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = _config(root)
            source = json.loads(config_path.read_text())
            draft = root / "draft.gguf"
            draft.write_bytes(b"draft")
            receipt_path = root / "draft-artifact.json"
            source["args"].extend(("--spec-type", "draft-eagle3", "--spec-draft-model", str(draft)))
            source["draft_artifact_manifest"] = str(receipt_path)
            config_path.write_text(json.dumps(source))
            target = source["shards"][0]
            receipt = {
                "verdict": {"artifact_complete": True, "full_sha256_verified": True},
                "compatibility": {"activation_allowed": True},
                "gguf_artifact": {"path": str(draft), "size": 5, "sha256": "c" * 64},
                "target": {"path": source["model"], "size": target["size"], "sha256": target["sha256"]},
            }
            receipt_path.write_text(json.dumps(receipt))
            self.assertEqual(llama_trial.load_trial_config(config_path).model_path, Path(source["model"]))

            for section, field, value, message in (
                ("compatibility", "activation_allowed", False, "does not allow activation"),
                ("verdict", "full_sha256_verified", False, "does not allow activation"),
                ("target", "sha256", "d" * 64, "different target"),
                ("gguf_artifact", "path", str(root / "other.gguf"), "artifact identity"),
                ("gguf_artifact", "size", 6, "artifact identity"),
            ):
                changed = json.loads(json.dumps(receipt))
                changed[section][field] = value
                receipt_path.write_text(json.dumps(changed))
                with self.subTest(section=section, field=field), self.assertRaisesRegex(
                    llama_trial.TrialError, message,
                ):
                    llama_trial.load_trial_config(config_path)

            receipt_path.write_text(json.dumps(receipt))
            del source["draft_artifact_manifest"]
            config_path.write_text(json.dumps(source))
            with self.assertRaisesRegex(llama_trial.TrialError, "draft_artifact_manifest"):
                llama_trial.load_trial_config(config_path)

    def test_start_uses_new_process_group_server_cwd_and_no_shell(self):
        with tempfile.TemporaryDirectory() as directory:
            config = llama_trial.load_trial_config(_config(Path(directory)))
            process = SimpleNamespace(pid=4321)
            with mock.patch.object(llama_trial.subprocess, "Popen", return_value=process) as popen:
                handle = llama_trial.start_server(
                    config,
                    [str(config.server_path), "--version"],
                    {"HOME": "/home/test"},
                    Path(directory) / "server.log",
                )
                handle.log_handle.close()
        self.assertEqual(
            popen.call_args.args[0],
            llama_trial.safe.unblocked_exec_command([str(config.server_path), "--version"]),
        )
        self.assertEqual(popen.call_args.kwargs["cwd"], str(config.server_path.parent))
        self.assertTrue(popen.call_args.kwargs["start_new_session"])
        self.assertEqual(popen.call_args.kwargs["env"]["HOME"], "/home/test")
        self.assertNotIn("shell", popen.call_args.kwargs)

    def test_preflight_checks_only_llama_port_and_large_gpu_process(self):
        with mock.patch.object(llama_trial, "port_is_occupied", return_value=True), mock.patch.object(
            llama_trial.safe, "gpu_compute_processes"
        ) as gpu:
            with self.assertRaisesRegex(llama_trial.TrialError, "port 18081"):
                llama_trial.assert_preflight_clear()
            gpu.assert_not_called()

        with mock.patch.object(llama_trial, "port_is_occupied", return_value=False), mock.patch.object(
            llama_trial.safe,
            "gpu_compute_processes",
            return_value=[{"pid": 73, "name": "other", "used_mib": 257}],
        ):
            with self.assertRaisesRegex(llama_trial.TrialError, "pid 73"):
                llama_trial.assert_preflight_clear()

    def test_startup_waits_for_alias_and_checks_process(self):
        response = mock.MagicMock()
        response.__enter__.return_value.read.return_value = json.dumps(
            {"object": "list", "data": [{"id": "qwen-test"}]}
        ).encode()
        process = mock.Mock()
        process.poll.side_effect = [None, None]
        handle = llama_trial.ServerHandle(process, 4321, io.BytesIO())
        with mock.patch.object(llama_trial, "urlopen", return_value=response) as open_url:
            payload = llama_trial.wait_for_server(handle, "qwen-test", timeout=1)
        self.assertEqual(llama_trial._model_ids(payload), ["qwen-test"])
        self.assertEqual(process.poll.call_count, 2)
        self.assertEqual(open_url.call_args.args[0].full_url, "http://127.0.0.1:18081/v1/models")

    def test_bench_commands_are_openai_bounded_and_use_requested_thinking(self):
        config = SimpleNamespace(model_name="qwen-test", context=32768)
        quick = llama_trial.bench_command(
            config, "none", "quick", Path("quick.jsonl"), quick_repeat=1
        )
        hard = llama_trial.bench_command(
            config,
            "medium",
            "hard",
            Path("hard.jsonl"),
            hard_cases="hard_temporal_kv,hard_dependency_constraints",
            output_tokens=4096,
            request_timeout=240.0,
        )
        self.assertIn("openai", quick)
        self.assertIn("http://127.0.0.1:18081/v1", quick)
        self.assertEqual(quick[-4:], ["--repeat", "1", "--num-predict", "512"])
        self.assertEqual(hard[hard.index("--thinking") + 1], "medium")
        self.assertEqual(
            hard[hard.index("--cases") + 1],
            "hard_temporal_kv,hard_dependency_constraints",
        )
        self.assertEqual(hard[hard.index("--num-predict") + 1], "4096")
        self.assertEqual(hard[hard.index("--timeout") + 1], "240")
        self.assertEqual(
            llama_trial.suite_process_timeout(
                "hard", hard_case_count=2, request_timeout=240.0
            ),
            540.0,
        )

    def test_hard_only_trial_runs_one_suite_and_cleans_up_owned_group(self):
        config = SimpleNamespace(
            server_path=Path("/tmp/llama-server"),
            model_name="qwen-test",
            context=32768,
            lib_dirs=(),
        )
        args = SimpleNamespace(
            tag="trial",
            thinking="medium",
            quick_repeat=1,
            skip_quick=True,
            hard_cases="hard_temporal_kv,hard_dependency_constraints",
            output_tokens=4096,
            request_timeout=240.0,
            startup_timeout=360.0,
        )
        process = SimpleNamespace(pid=9000)
        handle = llama_trial.ServerHandle(process, 9000, io.BytesIO())
        events = _FakeEvents()
        with mock.patch.object(llama_trial, "server_command", return_value=["server"]), mock.patch.object(
            llama_trial, "server_environment", return_value={}
        ), mock.patch.object(
            llama_trial, "start_server", return_value=handle
        ), mock.patch.object(
            llama_trial, "wait_for_server", return_value={"data": [{"id": "qwen-test"}]}
        ) as ready, mock.patch.object(llama_trial, "run_suite", return_value=0) as suite, mock.patch.object(
            llama_trial.safe,
            "stop_server",
            return_value={"pgid": 9000, "term_sent": True, "kill_sent": False, "stopped": True},
        ) as stop:
            passed = llama_trial.run_trial(config, args, events)
        self.assertTrue(passed)
        ready.assert_called_once_with(handle, "qwen-test", timeout=360.0)
        suite.assert_called_once()
        command = suite.call_args.args[0]
        self.assertEqual(command[command.index("--suite") + 1], "hard")
        self.assertEqual(suite.call_args.kwargs["hard_case_count"], 2)
        stop.assert_called_once_with(handle)
        self.assertEqual([row[0] for row in events.rows].count("suite_started"), 1)


if __name__ == "__main__":
    unittest.main()
