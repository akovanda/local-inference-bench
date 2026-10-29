from __future__ import annotations

import hashlib
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from runtime import prepare_bonsai2_dspark as prep


class Tensor:
    def __init__(self, name: str, type_name: str = "F32") -> None:
        self.name = name
        self.type_name = type_name


def target_reader() -> SimpleNamespace:
    tokens: list[object] = [None] * prep.TARGET_VOCAB_SIZE
    for text, token_id in prep.SPECIAL_TOKENS.items():
        tokens[token_id] = text
    metadata = {
        "general.architecture": "qwen35",
        "qwen35.block_count": 64,
        "qwen35.embedding_length": 5120,
        "tokenizer.ggml.model": "gpt2",
        "tokenizer.ggml.pre": "qwen35",
        "tokenizer.ggml.tokens": tokens,
        "tokenizer.ggml.bos_token_id": 248_044,
        "tokenizer.ggml.eos_token_id": 248_046,
        "tokenizer.ggml.padding_token_id": 248_044,
        "tokenizer.chat_template": "<|im_start|>{{ message }}<|im_end|>",
    }
    tensors = [Tensor(f"misc.{index}") for index in range(prep.TARGET_TENSOR_COUNT - 1)]
    tensors.append(Tensor("blk.63.ffn_up.weight"))
    return SimpleNamespace(metadata=metadata, tensors=tensors)


def legacy_reader() -> SimpleNamespace:
    metadata = {
        "general.architecture": "dspark",
        "general.file_type": 32,
        "dspark.block_count": 6,
        "dspark.embedding_length": 5120,
        "dspark.vocab_size": prep.TARGET_VOCAB_SIZE,
        "dspark.dspark.block_size": 4,
        "dspark.dspark.target_layers": [1, 16, 31, 46, 61],
        "dspark.dspark.mask_token_id": prep.MASK_TOKEN_ID,
    }
    names = sorted(prep.LEGACY_REQUIRED_TENSORS)
    names.extend(f"blk.{index}.fixture.{slot}" for index in range(6) for slot in range(12))
    return SimpleNamespace(metadata=metadata, tensors=[Tensor(name, "BF16") for name in names[:79]])


def converted_reader(target: SimpleNamespace) -> SimpleNamespace:
    metadata = dict(target.metadata)
    metadata.update(
        {
            "general.architecture": "dflash",
            "general.file_type": 2,
            "dflash.block_count": 6,
            "dflash.embedding_length": 5120,
            "dflash.vocab_size": prep.TARGET_VOCAB_SIZE,
            "dflash.block_size": 4,
            "dflash.target_layers": list(prep.TARGET_LAYERS),
            "tokenizer.ggml.mask_token_id": prep.MASK_TOKEN_ID,
        }
    )
    tensors = [Tensor(name, "Q4_0") for name in sorted(prep.FINAL_Q4_TENSORS)]
    tensors.extend(Tensor(name, "F32") for name in sorted(prep.FINAL_F32_TENSORS))
    confidence = Tensor("conf_proj.weight", "BF16")
    confidence.shape = [5376]
    tensors.append(confidence)
    return SimpleNamespace(metadata=metadata, tensors=tensors)


class PinnedFileTests(unittest.TestCase):
    def test_wrong_hash_is_rejected_after_exact_size(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.gguf"
            path.write_bytes(b"ABCD")
            with self.assertRaisesRegex(prep.PreparationError, "SHA-256 mismatch"):
                prep.validate_pinned_file(path, 4, "0" * 64)

    def test_truncated_download_is_rejected_before_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.gguf"
            path.write_bytes(b"ABC")
            with self.assertRaisesRegex(prep.PreparationError, "truncated or has the wrong size"):
                prep.validate_pinned_file(path, 4, hashlib.sha256(b"ABC").hexdigest())


class HeaderContractTests(unittest.TestCase):
    def test_pinned_structural_contracts_are_accepted(self) -> None:
        target = target_reader()
        legacy = legacy_reader()
        converted = converted_reader(target)

        self.assertEqual(prep.validate_target_reader(target)["tensor_count"], 851)
        self.assertEqual(prep.validate_legacy_reader(legacy)["block_size"], 4)
        self.assertTrue(
            prep.validate_converted_reader(converted, target)["shared_embedding_and_output_dropped"]
        )

    def test_wrong_target_and_wrong_converted_format_are_rejected(self) -> None:
        target = target_reader()
        target.metadata["general.architecture"] = "qwen3"
        with self.assertRaisesRegex(prep.PreparationError, "wrong target architecture"):
            prep.validate_target_reader(target)

        valid_target = target_reader()
        converted = converted_reader(valid_target)
        converted.metadata["general.architecture"] = "dspark"
        with self.assertRaisesRegex(prep.PreparationError, "arch=dflash"):
            prep.validate_converted_reader(converted, valid_target)

    def test_tokenizer_text_mismatch_is_rejected(self) -> None:
        target = target_reader()
        converted = converted_reader(target)
        converted.metadata["tokenizer.ggml.tokens"] = list(converted.metadata["tokenizer.ggml.tokens"])
        converted.metadata["tokenizer.ggml.tokens"][248_045] = "wrong delimiter"

        with self.assertRaisesRegex(prep.PreparationError, "exact target copy"):
            prep.validate_converted_reader(converted, target)

    def test_wrong_block_size_is_rejected(self) -> None:
        target = target_reader()
        converted = converted_reader(target)
        converted.metadata["dflash.block_size"] = 7
        with self.assertRaisesRegex(prep.PreparationError, "block size is not 4"):
            prep.validate_converted_reader(converted, target)

    def test_retained_shared_tensor_is_rejected(self) -> None:
        target = target_reader()
        converted = converted_reader(target)
        index = next(
            index for index, tensor in enumerate(converted.tensors) if tensor.name == "blk.0.attn_k.weight"
        )
        converted.tensors[index] = Tensor("token_embd.weight", "Q4_0")
        with self.assertRaisesRegex(prep.PreparationError, "were not dropped"):
            prep.validate_converted_reader(converted, target)

    def test_only_exact_tiny_confidence_projection_may_remain_bf16(self) -> None:
        target = target_reader()
        converted = converted_reader(target)

        receipt = prep.validate_converted_reader(converted, target)

        self.assertEqual(receipt["unquantized_bf16_tensors"], {"conf_proj.weight": [5376]})

    def test_other_bf16_tensor_or_shape_is_rejected(self) -> None:
        target = target_reader()
        converted = converted_reader(target)
        confidence = next(tensor for tensor in converted.tensors if tensor.name == "conf_proj.weight")
        confidence.type_name = "BF16"
        confidence.shape = [5376, 1]

        with self.assertRaisesRegex(prep.PreparationError, "unexpected BF16 tensors"):
            prep.validate_converted_reader(converted, target)

    def test_unexpected_final_tensor_type_is_rejected(self) -> None:
        target = target_reader()
        converted = converted_reader(target)
        fc = next(tensor for tensor in converted.tensors if tensor.name == "fc.weight")
        fc.type_name = "Q5_0"

        with self.assertRaisesRegex(prep.PreparationError, "Q4_0 tensor inventory changed"):
            prep.validate_converted_reader(converted, target)


class PublishTests(unittest.TestCase):
    def test_publish_is_atomic_and_does_not_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            temporary = root / ".candidate"
            destination = root / "draft.gguf"
            temporary.write_bytes(b"candidate")
            destination.write_bytes(b"existing")

            with self.assertRaisesRegex(prep.PreparationError, "refusing to overwrite"):
                prep.publish_no_overwrite(temporary, destination)

            self.assertEqual(destination.read_bytes(), b"existing")
            self.assertEqual(temporary.read_bytes(), b"candidate")

    def test_successful_publish_removes_temporary_link(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            temporary = root / ".candidate"
            destination = root / "draft.gguf"
            temporary.write_bytes(b"candidate")

            prep.publish_no_overwrite(temporary, destination)

            self.assertFalse(temporary.exists())
            self.assertEqual(destination.read_bytes(), b"candidate")


class CpuOnlyStageTests(unittest.TestCase):
    def test_commands_are_argument_vectors_with_no_server_or_gpu_launch(self) -> None:
        commands = prep.conversion_commands(
            Path("/models/source.gguf"),
            Path("/models/target.gguf"),
            Path("/models/intermediate.gguf"),
            Path("/models/output.gguf"),
            Path("/cpu-tools/llama-quantize"),
        )

        self.assertIn("--drop-shared-tensors", commands[0])
        self.assertEqual(commands[1][-2:], ["Q4_0", "4"])
        flattened = " ".join(part for command in commands for part in command)
        self.assertNotIn("llama-server", flattened)
        self.assertNotIn("--gpu-layers", flattened)
        self.assertNotIn("--spec-type", flattened)

    def test_stage_only_runs_converter_and_cpu_quantizer(self) -> None:
        run_commands: list[tuple[list[str], dict[str, object]]] = []

        def fake_run(command: list[str], **kwargs: object) -> SimpleNamespace:
            run_commands.append((command, kwargs))
            output = Path(command[-1] if "--drop-shared-tensors" in command else command[2])
            output.write_bytes(b"fixture")
            return SimpleNamespace(returncode=0)

        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "draft.gguf"
            with (
                mock.patch.object(prep, "validate_pinned_file", return_value={"full_sha256_verified": True}),
                mock.patch.object(prep, "validate_cpu_quantizer", return_value={"gpu_runtime_linked": False}),
                mock.patch.object(prep, "load_gguf", return_value=SimpleNamespace()),
                mock.patch.object(prep, "validate_legacy_reader", return_value={"legacy": True}),
                mock.patch.object(prep, "validate_target_reader", return_value={"target": True}),
                mock.patch.object(prep, "validate_converted_reader", side_effect=[{"bf16": True}, {"q4": True}]),
                mock.patch.object(
                    prep,
                    "sha256_file",
                    side_effect=lambda path: prep.CONVERTER_SHA256 if path == prep.CONVERTER else "artifact-sha",
                ),
                mock.patch.object(prep.subprocess, "run", side_effect=fake_run),
            ):
                receipt = prep.stage(Path("/already-present/source.gguf"), Path("/cpu/llama-quantize"), destination)

            self.assertTrue(destination.is_file())
            self.assertFalse(receipt["gpu_validated"])
            self.assertFalse(receipt["inference_validated"])
            self.assertEqual(len(run_commands), 2)
            self.assertTrue(all(kwargs.get("shell") is False for _, kwargs in run_commands))
            self.assertEqual(run_commands[0][0][1], str(prep.CONVERTER))
            self.assertEqual(run_commands[1][0][0], "/cpu/llama-quantize")

    def test_default_cli_reports_revised_finite_download_eligibility(self) -> None:
        output = io.StringIO()
        with redirect_stdout(output):
            rc = prep.main([])
        payload = output.getvalue()
        self.assertEqual(rc, 0)
        self.assertIn("ready_to_download", payload)
        self.assertIn('"download_supported": true', payload)
        self.assertLessEqual(prep.SOURCE_SIZE, prep.DOWNLOAD_LIMIT)


class DownloadBoundTests(unittest.TestCase):
    @mock.patch("runtime.range_fetch.download_file")
    @mock.patch.object(prep, "validate_pinned_file", return_value={"full_sha256_verified": True})
    def test_revised_bound_downloads_only_the_exact_pin(
        self,
        validate: mock.Mock,
        download: mock.Mock,
    ) -> None:
        destination = Path("/models/dspark/source.gguf")

        receipt = prep.download_source(destination, workers=3, chunk_mib=16, retries=2)

        self.assertTrue(receipt["full_sha256_verified"])
        download.assert_called_once_with(
            prep.SOURCE_URL,
            destination,
            prep.SOURCE_SIZE,
            prep.SOURCE_SHA256,
            workers=3,
            chunk_size=16 * 1024 * 1024,
            retries=2,
        )
        validate.assert_called_once_with(destination, prep.SOURCE_SIZE, prep.SOURCE_SHA256)

    @mock.patch("runtime.range_fetch.download_file")
    def test_bound_refusal_happens_before_network(self, download: mock.Mock) -> None:
        with (
            mock.patch.object(prep, "SOURCE_SIZE", prep.DOWNLOAD_LIMIT + 1),
            self.assertRaisesRegex(prep.PreparationError, "exceeds the finite download ceiling"),
        ):
            prep.download_source(Path("/unused/source.gguf"))

        download.assert_not_called()


if __name__ == "__main__":
    unittest.main()
