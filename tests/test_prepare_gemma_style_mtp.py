from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from runtime import prepare_gemma_style_mtp as prep


class _Tensor:
    def __init__(self, name, shape, type_name, offset, size=32):
        self.name = name
        self.shape = shape
        self.type_name = type_name
        self.offset = offset
        self._size = size

    def expected_bytes(self):
        return self._size


def _tokenizer():
    values = {
        "tokenizer.ggml.model": "gemma4",
        "tokenizer.ggml.tokens": ["<pad>", "<eos>", "<bos>", "<unk>", "<mask>", "token-5"],
        "tokenizer.ggml.scores": [-1000.0] * 6,
        "tokenizer.ggml.token_type": [3, 3, 3, 3, 3, 1],
        "tokenizer.ggml.merges": ["a b"],
        "tokenizer.ggml.bos_token_id": 2,
        "tokenizer.ggml.eos_token_id": 1,
        "tokenizer.ggml.unknown_token_id": 3,
        "tokenizer.ggml.padding_token_id": 0,
        "tokenizer.ggml.mask_token_id": 4,
        "tokenizer.ggml.add_bos_token": True,
        "tokenizer.ggml.add_eos_token": False,
        "tokenizer.ggml.add_space_prefix": False,
    }
    return values


class _AssistantReader:
    def __init__(self, _path):
        self.version = 3
        self.alignment = 32
        self.data_start = 128
        self.metadata = {
            "general.architecture": "gemma4-assistant",
            "gemma4-assistant.block_count": 4,
            "gemma4-assistant.context_length": 262_144,
            "gemma4-assistant.embedding_length": 1_024,
            "gemma4-assistant.feed_forward_length": 8_192,
            "gemma4-assistant.attention.head_count": 16,
            "gemma4-assistant.attention.head_count_kv": [8, 8, 8, 1],
            "gemma4-assistant.attention.shared_kv_layers": 4,
            "gemma4-assistant.embedding_length_out": 3_840,
            "gemma4-assistant.nextn_predict_layers": 4,
            **_tokenizer(),
        }
        core = [
            _Tensor("token_embd.weight", [1_024, 262_144], "Q8_0", 0),
            _Tensor("nextn.pre_projection.weight", [7_680, 1_024], "Q8_0", 32),
            _Tensor("nextn.post_projection.weight", [1_024, 3_840], "Q8_0", 64),
        ]
        extras = [_Tensor(f"blk.fixture.{i}", [1], "F32", 96 + i * 32) for i in range(46)]
        self.tensors = core + extras


class _TargetReader:
    def __init__(self, _path):
        self.version = 3
        self.alignment = 32
        self.data_start = 128
        self.metadata = {
            "general.architecture": "gemma4",
            "gemma4.block_count": 48,
            "gemma4.context_length": 262_144,
            "gemma4.embedding_length": 3_840,
            "gemma4.feed_forward_length": 15_360,
            "gemma4.attention.head_count": 16,
            **_tokenizer(),
        }
        self.metadata["tokenizer.ggml.eos_token_id"] = 106
        self.tensors = [
            _Tensor("token_embd.weight", [3_840, 262_144], "Q5_K", 0),
            _Tensor("output.weight", [3_840, 262_144], "Q6_K", 32),
        ]


class MetadataTests(unittest.TestCase):
    def _file(self, root: Path, name: str, size: int = 4096) -> Path:
        path = root / name
        path.write_bytes(b"\0" * size)
        return path

    def test_metadata_only_checks_bind_assistant_output_to_untied_target(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            assistant_path = self._file(root, "assistant.gguf")
            target_path = self._file(root, "target.gguf")
            assistant = prep.inspect_model(
                assistant_path,
                "assistant",
                expected_size=assistant_path.stat().st_size,
                reader_class=_AssistantReader,
            )
            target = prep.inspect_model(
                target_path,
                "target",
                expected_size=target_path.stat().st_size,
                reader_class=_TargetReader,
            )
            compatibility = prep.compare_models(target, assistant)

        self.assertTrue(compatibility["metadata_dimension_compatible"])
        self.assertEqual(compatibility["assistant_output_width"], 3_840)
        self.assertEqual(compatibility["target_output_embedding_shape"], [3_840, 262_144])
        self.assertTrue(compatibility["target_has_distinct_input_and_output_tensors"])
        self.assertFalse(compatibility["tie_word_embeddings"]["style_tune_config"])
        self.assertFalse(compatibility["tokenizer_metadata_exact_match"])
        self.assertFalse(compatibility["eos"]["id_match_required_by_pinned_runtime"])
        self.assertTrue(compatibility["runtime_vocabulary_compatibility_checks_pass"])

    def test_auxiliary_token_type_difference_is_recorded_but_not_a_runtime_gate(self):
        class DifferentTypes(_AssistantReader):
            def __init__(self, path):
                super().__init__(path)
                self.metadata["tokenizer.ggml.token_type"][-1] = 4

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            assistant_path = self._file(root, "assistant.gguf")
            target_path = self._file(root, "target.gguf")
            assistant = prep.inspect_model(
                assistant_path,
                "assistant",
                expected_size=assistant_path.stat().st_size,
                reader_class=DifferentTypes,
            )
            target = prep.inspect_model(
                target_path,
                "target",
                expected_size=target_path.stat().st_size,
                reader_class=_TargetReader,
            )
            compatibility = prep.compare_models(target, assistant)

        self.assertTrue(compatibility["runtime_vocabulary_compatibility_checks_pass"])
        self.assertFalse(
            compatibility["auxiliary_tokenizer_metadata_matches"]["tokenizer.ggml.token_type"]
        )

    def test_tokenizer_difference_blocks_compatibility(self):
        class DifferentTokenizer(_AssistantReader):
            def __init__(self, path):
                super().__init__(path)
                self.metadata["tokenizer.ggml.tokens"][-1] = "different"

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            assistant_path = self._file(root, "assistant.gguf")
            target_path = self._file(root, "target.gguf")
            assistant = prep.inspect_model(
                assistant_path,
                "assistant",
                expected_size=assistant_path.stat().st_size,
                reader_class=DifferentTokenizer,
            )
            target = prep.inspect_model(
                target_path,
                "target",
                expected_size=target_path.stat().st_size,
                reader_class=_TargetReader,
            )
            with self.assertRaisesRegex(prep.PreparationError, "token text"):
                prep.compare_models(target, assistant)


class DownloadSafetyTests(unittest.TestCase):
    def test_verified_existing_file_is_reused_without_downloader(self):
        payload = b"verified assistant"
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "assistant.gguf"
            destination.write_bytes(payload)
            with mock.patch.object(prep, "FILE_SIZE", len(payload)), mock.patch.object(
                prep, "FILE_SHA256", hashlib.sha256(payload).hexdigest()
            ), mock.patch.object(prep.range_fetch, "download_file") as downloader, mock.patch.object(
                prep, "validate_runtime_source", return_value={"draft_mtp_present": True}
            ), mock.patch.object(prep, "validate_target_receipt", return_value={"receipt": True}), mock.patch.object(
                prep, "inspect_model"
            ) as inspect, mock.patch.object(prep, "compare_models", return_value={"metadata_dimension_compatible": True}):
                inspect.side_effect = [
                    prep.ModelInspection({"target": True}, _tokenizer()),
                    prep.ModelInspection({"assistant": True}, _tokenizer()),
                ]
                report = prep.stage(destination, Path(directory) / "report.json")

        downloader.assert_not_called()
        self.assertTrue(report["verdict"]["artifact_complete"])
        self.assertFalse(report["download_verification"]["downloaded_by_this_run"])

    def test_conflicting_existing_file_is_rejected_without_download_or_overwrite(self):
        original = b"do not overwrite"
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "assistant.gguf"
            destination.write_bytes(original)
            report_path = Path(directory) / "report.json"
            with mock.patch.object(prep, "FILE_SIZE", len(original)), mock.patch.object(
                prep, "FILE_SHA256", "0" * 64
            ), mock.patch.object(prep.range_fetch, "download_file") as downloader, mock.patch.object(
                prep, "validate_runtime_source", return_value={"draft_mtp_present": True}
            ), mock.patch.object(prep, "validate_target_receipt", return_value={"receipt": True}), mock.patch.object(
                prep, "inspect_model", return_value=prep.ModelInspection({"target": True}, _tokenizer())
            ):
                with self.assertRaisesRegex(prep.PreparationError, "SHA-256"):
                    prep.stage(destination, report_path)

            self.assertEqual(destination.read_bytes(), original)
            downloader.assert_not_called()
            report = json.loads(report_path.read_text())
            self.assertEqual(report["state"], "inactive_preparation_failed")
            self.assertFalse(report["verdict"]["artifact_complete"])

    def test_absent_file_uses_bounded_pinned_download_then_fresh_hash(self):
        payload = b"new assistant"
        digest = hashlib.sha256(payload).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "assistant.gguf"

            def fake_download(url, output, size, sha256, **options):
                self.assertEqual(url, prep.URL)
                self.assertEqual(size, len(payload))
                self.assertEqual(sha256, digest)
                self.assertEqual(options["workers"], 2)
                self.assertEqual(options["chunk_size"], 16 * 1024 * 1024)
                Path(output).write_bytes(payload)

            with mock.patch.object(prep, "FILE_SIZE", len(payload)), mock.patch.object(
                prep, "FILE_SHA256", digest
            ), mock.patch.object(prep.range_fetch, "download_file", side_effect=fake_download), mock.patch.object(
                prep, "validate_runtime_source", return_value={"draft_mtp_present": True}
            ), mock.patch.object(prep, "validate_target_receipt", return_value={"receipt": True}), mock.patch.object(
                prep, "inspect_model"
            ) as inspect, mock.patch.object(prep, "compare_models", return_value={"metadata_dimension_compatible": True}):
                inspect.side_effect = [
                    prep.ModelInspection({"target": True}, _tokenizer()),
                    prep.ModelInspection({"assistant": True}, _tokenizer()),
                ]
                report = prep.stage(destination, Path(directory) / "report.json")

        self.assertTrue(report["download_verification"]["downloaded_by_this_run"])
        self.assertEqual(report["download_verification"]["sha256"], digest)


class ReportTests(unittest.TestCase):
    def test_complete_artifact_never_claims_runtime_or_speedup(self):
        report = prep.artifact_report(
            download={"full_sha256_verified": True},
            assistant={"metadata": True},
            target={"metadata": True},
            target_receipt={"verified": True},
            compatibility={"metadata_dimension_compatible": True},
            runtime_source={"draft_mtp_present": True},
        )
        self.assertEqual(report["state"], "inactive_verified")
        self.assertTrue(report["verdict"]["artifact_complete"])
        self.assertFalse(report["verdict"]["gpu_inference_validated"])
        self.assertFalse(report["limitations"]["speculative_acceptance_validated"])
        self.assertFalse(report["limitations"]["speedup_measured"])
        self.assertFalse(report["resource_basis"]["sample_is_a_guaranteed_budget"])


if __name__ == "__main__":
    unittest.main()
