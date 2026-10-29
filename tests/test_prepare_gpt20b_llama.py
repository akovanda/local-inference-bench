import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from runtime import prepare_gpt20b_llama as prep


class _Tensor:
    def __init__(self, name, type_name, offset, size):
        self.name = name
        self.type_name = type_name
        self.offset = offset
        self._size = size

    def expected_bytes(self):
        return self._size


class _Reader:
    def __init__(self, _path):
        self.version = 3
        self.alignment = 32
        self.data_start = 128
        self.metadata = {
            "general.architecture": "gptoss",
            "general.name": "gpt-oss-20b",
            "gptoss.block_count": 24,
            "gptoss.context_length": 131072,
            "gptoss.expert_count": 32,
            "gptoss.expert_used_count": 4,
            "tokenizer.ggml.tokens": list(prep.REQUIRED_VOCAB_TOKENS),
        }
        self.tensors = [
            _Tensor("a", "MXFP4", 0, 32),
            _Tensor("b", "F32", 32, 16),
        ]


class PrepareGpt20bLlamaTests(unittest.TestCase):
    def _model(self, root: Path) -> Path:
        path = root / f"sha256-{prep.MODEL_DIGEST}"
        path.write_bytes(b"x")
        return path

    def test_header_validation_proves_harmony_and_mxfp4_without_hashing_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            model = self._model(Path(directory))
            fake_stat = SimpleNamespace(
                st_mode=0o100444,
                st_size=prep.MODEL_SIZE,
                st_dev=1,
                st_ino=2,
                st_nlink=1,
                st_mtime_ns=3,
                st_ctime_ns=4,
            )
            with mock.patch.object(Path, "lstat", return_value=fake_stat), mock.patch(
                "runtime.prepare_gpt20b_llama.os.access", return_value=True
            ):
                receipt = prep.inspect_model(
                    model,
                    _Reader,
                    "<|start|><|channel|>analysis<|message|><|end|>"
                    "<|channel|>final<|message|>",
                    {"path": "/template"},
                )
        self.assertFalse(receipt["fresh_full_sha256_verified"])
        self.assertEqual(receipt["gguf"]["architecture"], "gptoss")
        self.assertTrue(receipt["gguf"]["requires_ollama_metadata_translation"])
        self.assertEqual(receipt["gguf"]["tensor_type_counts"]["MXFP4"], 1)
        self.assertTrue(receipt["harmony"]["thinking_end_tags_nonempty"])
        self.assertIn("<|return|>", receipt["harmony"]["token_ids"])

    def test_header_validation_rejects_missing_harmony_marker(self):
        class BadReader(_Reader):
            def __init__(self, path):
                super().__init__(path)
                self.metadata["tokenizer.chat_template"] = "analysis final"

        with tempfile.TemporaryDirectory() as directory:
            model = self._model(Path(directory))
            fake_stat = SimpleNamespace(
                st_mode=0o100444,
                st_size=prep.MODEL_SIZE,
                st_dev=1,
                st_ino=2,
                st_nlink=1,
                st_mtime_ns=3,
                st_ctime_ns=4,
            )
            with mock.patch.object(Path, "lstat", return_value=fake_stat), mock.patch(
                "runtime.prepare_gpt20b_llama.os.access", return_value=True
            ), self.assertRaisesRegex(prep.PreparationError, "Harmony marker"):
                prep.inspect_model(model, BadReader)

    def test_source_validation_requires_budget_forwarding_and_forcing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for relative, fragments in prep.SOURCE_REQUIREMENTS.items():
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("\n".join(fragments))
            receipt = prep.validate_source(root)
            self.assertTrue(receipt["source_support_validated"])
            self.assertEqual(receipt["reasoning_end_tags"], ["<|end|>"])
            target = root / "tools/server/server-common.cpp"
            target.write_text("budget field removed")
            with self.assertRaisesRegex(prep.PreparationError, "source support check failed"):
                prep.validate_source(root)

    def test_bundle_validation_requires_translation_parser_and_budget_symbols(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            server = root / "server"
            llama = root / "llama"
            common = root / "common"
            server_impl = root / "server-impl"
            server.write_bytes(b"wrapper")
            llama.write_bytes(
                b"handle_gptoss detected Ollama-format gpt-oss GGUF; applying compatibility fixes "
                b"gptoss. gpt-oss."
            )
            common.write_bytes(
                b"<|channel|>analysis<|message|> "
                b"<|end|><|start|>assistant<|channel|>final<|message|> "
                b"budget exhausted, forcing end sequence"
            )
            server_impl.write_bytes(b"reasoning_budget_tokens reasoning_budget_end_tags")
            with mock.patch.object(prep, "SERVER_SHA256", prep._sha256(server)):
                receipt = prep.validate_bundle(server, llama, common, server_impl)
            self.assertTrue(receipt["ollama_gptoss_metadata_translation_present"])
            self.assertTrue(receipt["reasoning_budget_request_and_sampler_present"])

    def test_config_and_receipt_are_inactive_and_truthful_about_full_hash(self):
        model = {
            "path": str(prep.MODEL),
            "fresh_full_sha256_verified": False,
            "harmony": {"thinking_end_tags_nonempty": True},
        }
        report = prep.artifact_report(model, {"source_support_validated": True})
        config = prep.config_document(prep.DEFAULT_REPORT)
        self.assertFalse(report["verdict"]["artifact_complete"])
        self.assertFalse(report["verdict"]["runtime_load_validated"])
        self.assertFalse(report["reasoning_budget"]["enforcement_runtime_validated"])
        self.assertFalse(report["shards"][0]["fresh_full_sha256_verified"])
        self.assertEqual(config["host"], "127.0.0.1")
        self.assertEqual(config["port"], 18081)
        self.assertNotIn("--cpu-moe", config["args"])
        self.assertIn("--chat-template-file", config["args"])


if __name__ == "__main__":
    unittest.main()
