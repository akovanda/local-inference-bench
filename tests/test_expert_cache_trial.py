import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from runtime import expert_cache_trial


def _fixture_config(root: Path) -> expert_cache_trial.PinnedConfig:
    return expert_cache_trial.PinnedConfig(
        source_path=root / "source.json",
        source_sha256="c" * 64,
        engine_path=root / "strata",
        model_path=root / "model.gguf",
        tokenizer_path=root / "tokenizer",
        cwd=root,
        model_name="qwen-test",
        library_dirs=(root / "lib",),
    )


def _fixture_prompt() -> expert_cache_trial.PromptTokens:
    tokenizer = SimpleNamespace(decode=lambda ids: "final answer<|im_end|>ignored")
    return expert_cache_trial.PromptTokens(
        case_id="quick_generation",
        case_prompt_sha256="a" * 64,
        rendered_prompt_sha256="b" * 64,
        rendered_prompt_chars=123,
        ids=(11, 22, 33),
        ids_sha256="d" * 64,
        tokenizer=tokenizer,
    )


class ExpertCacheTrialTests(unittest.TestCase):
    def test_parser_preserves_native_counter_semantics(self):
        output = (
            "strata-q36: hybrid experts: 12 of 256 per layer in VRAM (4.25 GiB), "
            "all 19.50 GiB in pinned host memory, the rest of a batch of <= 128 rows "
            "on 31 CPU threads, bigger prompt chunks streamed (0.75 GiB staging); 4.2 s\n"
            "gen: 41 42 43\n"
            "strata-q36: prompt 88 tokens in 100.0 ms (880.0 tok/s), "
            "3 generated in 150.0 ms (20.0 tok/s)\n"
            "strata-q36: hybrid experts: 910 of 1200 batch entries on the GPU "
            "(75.8%), 290 on the CPU pool in 4321 ms, 37 slot replacements\n"
        )
        parsed = expert_cache_trial.parse_native_output(output)

        self.assertEqual(parsed["generated_token_ids"], [41, 42, 43])
        self.assertEqual(parsed["native_timing"]["generated_tokens_per_second"], 20.0)
        placement = parsed["hybrid_expert_placement"]
        self.assertEqual(placement["slots_per_layer"], 12)
        self.assertEqual(placement["experts_per_layer"], 256)
        self.assertEqual(placement["host_memory"], "pinned")
        self.assertIn("per routed-expert layer", placement["slots_note"])
        counters = parsed["hybrid_expert_counters"]
        self.assertEqual(counters["gpu_batch_entries"], 910)
        self.assertEqual(counters["cpu_pool_batch_entries"], 290)
        self.assertEqual(counters["slot_replacements"], 37)
        self.assertIn("prompt rows and decode", counters["scope_note"])
        self.assertIn("not a wall-time fraction", counters["cpu_ms_note"])

    def test_pinned_config_rejects_argument_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            engine = root / "strata"
            engine.write_text("#!/bin/sh\n", encoding="utf-8")
            engine.chmod(0o755)
            model = root / "model.gguf"
            model.write_bytes(b"model")
            tokenizer = root / "tokenizer"
            tokenizer.mkdir()
            for name in ("chat_template.jinja", "vocab.json", "merges.txt", "token_type.json"):
                (tokenizer / name).write_text("{}", encoding="utf-8")
            config_path = root / "config.json"
            source = {
                "exe": str(engine),
                "args": [
                    "--native", str(model), "--max-context", "32768",
                    "--prefill", "auto", "--mtp", "0",
                ],
                "cwd": str(root),
                "tokenizer": str(tokenizer),
                "model_name": "qwen-test",
                "lib_dirs": [],
            }
            config_path.write_text(json.dumps(source), encoding="utf-8")
            loaded = expert_cache_trial.load_pinned_config(
                config_path,
                expected_engine=engine,
                expected_model=model,
                expected_tokenizer=tokenizer,
                expected_cwd=root,
            )
            self.assertEqual(loaded.model_path, model.resolve())

            source["args"].extend(("--cpu-threads", "1"))
            config_path.write_text(json.dumps(source), encoding="utf-8")
            with self.assertRaisesRegex(expert_cache_trial.TrialError, "arguments differ"):
                expert_cache_trial.load_pinned_config(
                    config_path,
                    expected_engine=engine,
                    expected_model=model,
                    expected_tokenizer=tokenizer,
                    expected_cwd=root,
                )

    def test_command_is_fixed_and_redacts_prompt_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = _fixture_config(root)
            prompt = _fixture_prompt()
            command = expert_cache_trial.build_engine_command(
                config,
                prompt,
                context=32768,
                generation_tokens=128,
                expert_slots=12,
            )
        self.assertEqual(command[:3], [str(config.engine_path), "--native", str(config.model_path)])
        self.assertEqual(command[-4:], ["--n", "128", "--expert-slots", "12"])
        self.assertEqual(command[command.index("--tokens") + 1], "11,22,33")
        redacted = expert_cache_trial.redacted_invocation(command, prompt)
        self.assertNotIn("11,22,33", redacted)
        self.assertIn("3 token IDs", redacted[redacted.index("--tokens") + 1])

    def test_start_uses_signal_unblock_shim_new_session_and_no_shell(self):
        fake_process = SimpleNamespace(pid=4312)
        environment = {"HOME": "/home/test", "CODEX_HOME": "/codex/test"}
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            expert_cache_trial.subprocess, "Popen", return_value=fake_process
        ) as popen:
            root = Path(directory)
            handle = expert_cache_trial.start_trial(
                ["/fixed/strata", "--native", "/fixed/model"],
                environment,
                root / "native.log",
                cwd=root,
            )
            handle.log_handle.close()

        self.assertEqual(
            popen.call_args.args[0],
            expert_cache_trial.unblocked_exec_command(
                ["/fixed/strata", "--native", "/fixed/model"]
            ),
        )
        self.assertTrue(popen.call_args.kwargs["start_new_session"])
        self.assertNotIn("shell", popen.call_args.kwargs)
        self.assertEqual(popen.call_args.kwargs["env"], environment)

    def test_cleanup_targets_only_owned_process_group(self):
        process = mock.Mock()
        process.wait.return_value = -15
        process.poll.return_value = -15
        handle = expert_cache_trial.ProcessHandle(process, 7721, io.BytesIO())
        with mock.patch.object(
            expert_cache_trial, "_group_exists", side_effect=[True, False]
        ), mock.patch.object(expert_cache_trial.os, "killpg") as killpg:
            result = expert_cache_trial.cleanup_owned_process(handle)

        killpg.assert_called_once_with(7721, expert_cache_trial.signal.SIGTERM)
        self.assertTrue(result["term_sent"])
        self.assertTrue(result["stopped"])

    def test_timeout_is_recorded_after_owned_group_cleanup(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result_path = root / "result.json"
            log_path = root / "native.log"
            log_path.write_text("partial startup\n", encoding="utf-8")
            process = mock.Mock()
            process.wait.side_effect = subprocess.TimeoutExpired(["strata"], 420)
            process.poll.return_value = -15
            handle = expert_cache_trial.ProcessHandle(process, 991, io.BytesIO())
            args = Namespace(
                tag="timeout-test", case="quick_generation", context=32768,
                tokens=128, expert_slots=None,
            )
            with mock.patch.object(
                expert_cache_trial, "start_trial", return_value=handle
            ), mock.patch.object(
                expert_cache_trial, "cleanup_owned_process",
                return_value={"pgid": 991, "term_sent": True, "kill_sent": False, "stopped": True},
            ) as cleanup:
                record, code = expert_cache_trial.execute_trial(
                    args, _fixture_config(root), _fixture_prompt(), result_path, log_path
                )

            self.assertEqual(code, 124)
            self.assertEqual(record["status"], "timed_out")
            self.assertTrue(record["timed_out"])
            cleanup.assert_called_once_with(handle)
            self.assertEqual(json.loads(result_path.read_text())["status"], "timed_out")

    def test_preflight_failure_never_launches_native_engine(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            expert_cache_trial, "RESULTS", Path(directory)
        ), mock.patch.object(
            expert_cache_trial, "load_pinned_config", return_value=_fixture_config(Path(directory))
        ), mock.patch.object(
            expert_cache_trial, "tokenize_case", return_value=_fixture_prompt()
        ), mock.patch.object(
            expert_cache_trial, "assert_preflight_clear",
            side_effect=expert_cache_trial.SharedPreflightError("GPU busy"),
        ), mock.patch.object(expert_cache_trial, "start_trial") as start, mock.patch(
            "sys.stderr", new=io.StringIO()
        ):
            code = expert_cache_trial.main(["--tag", "preflight-test"])

        self.assertEqual(code, 1)
        start.assert_not_called()

    def test_decoded_output_never_persists_unexpected_reasoning(self):
        clean = SimpleNamespace(decode=lambda ids: "answer<|im_end|>extra")
        self.assertEqual(
            expert_cache_trial.decoded_final_text(clean, [1]), ("answer", None)
        )
        reasoning = SimpleNamespace(decode=lambda ids: "<think>private chain</think>answer")
        text, reason = expert_cache_trial.decoded_final_text(reasoning, [1])
        self.assertIsNone(text)
        self.assertIn("reasoning", reason)

    def test_case_lookup_adds_repo_root_for_direct_script_launch(self):
        original = list(sys.path)
        without_root = [
            item for item in original
            if item and Path(item).resolve() != expert_cache_trial.ROOT.resolve()
        ]
        saved = sys.modules.pop("bench_cases", None)
        try:
            with mock.patch.object(sys, "path", [str(expert_cache_trial.ROOT / "runtime"), *without_root]):
                prompt = expert_cache_trial._case_prompt("hard_transaction_ledger")
                self.assertIn("Process this event ledger", prompt)
                self.assertEqual(sys.path[0], str(expert_cache_trial.ROOT))
        finally:
            sys.modules.pop("bench_cases", None)
            if saved is not None:
                sys.modules["bench_cases"] = saved


if __name__ == "__main__":
    unittest.main()
