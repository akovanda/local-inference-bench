from __future__ import annotations

import struct
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from runtime import prepare_gpt20b_eagle3 as prep


class ArtifactReportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.source = {"source_support_validated": True}
        self.target = {"fresh_full_sha256_verified": True}
        self.bundle = {"eagle3_loader_present": True}

    def test_pending_report_does_not_claim_payload_or_profile_ready(self) -> None:
        report = prep.artifact_report(
            source=self.source,
            target=self.target,
            bundle=self.bundle,
        )

        self.assertEqual(report["state"], "inactive_download_pending")
        self.assertFalse(report["verdict"]["full_sha256_verified"])
        self.assertFalse(report["verdict"]["artifact_complete"])
        self.assertFalse(report["verdict"]["profile_publication_staged"])

    def test_compatibility_failure_preserves_completed_full_sha(self) -> None:
        error = prep.PreparationError("draft tokenizer does not exactly match target tokenizer")
        download = {
            "path": str(prep.DRAFT_MODEL),
            "size": prep.FILE_SIZE,
            "sha256": prep.FILE_SHA256,
            "full_sha256_verified": True,
            "verified_at": "2026-10-19T09:30:48Z",
        }

        report = prep.artifact_report(
            source=self.source,
            target=self.target,
            bundle=self.bundle,
            download=download,
            error=error,
        )

        self.assertEqual(report["state"], "inactive_compatibility_blocked")
        self.assertEqual(report["download_verification"], download)
        self.assertTrue(report["verdict"]["full_sha256_verified"])
        self.assertFalse(report["verdict"]["header_and_tensor_directory_valid"])
        self.assertFalse(report["verdict"]["target_tokenizer_exact_match"])
        self.assertFalse(report["verdict"]["artifact_complete"])
        self.assertFalse(report["verdict"]["profile_publication_staged"])
        self.assertEqual(report["error"], "PreparationError")

    def test_complete_report_requires_validated_draft(self) -> None:
        download = {"full_sha256_verified": True}
        draft = {
            "fresh_full_sha256_verified": True,
            "target_compatibility": {"tokenizers_exactly_match": True},
        }

        report = prep.artifact_report(
            source=self.source,
            target=self.target,
            download=download,
            draft=draft,
        )

        self.assertEqual(report["state"], "inactive_verified")
        self.assertTrue(report["verdict"]["artifact_complete"])
        self.assertTrue(report["verdict"]["target_tokenizer_exact_match"])
        self.assertTrue(report["verdict"]["profile_publication_staged"])


class StageTests(unittest.TestCase):
    @mock.patch.object(prep, "publish_configs")
    @mock.patch.object(prep, "atomic_write_json")
    @mock.patch.object(prep, "inspect_draft")
    @mock.patch.object(prep.range_fetch, "download_file")
    @mock.patch.object(prep, "validate_bundle", return_value={"bundle": True})
    @mock.patch.object(prep, "validate_target_contract", return_value={"target": True})
    @mock.patch.object(prep, "validate_source", return_value={"source": True})
    def test_stage_retains_sha_proof_when_compatibility_inspection_fails(
        self,
        _validate_source: mock.Mock,
        _validate_target: mock.Mock,
        _validate_bundle: mock.Mock,
        download_file: mock.Mock,
        inspect_draft: mock.Mock,
        atomic_write_json: mock.Mock,
        publish_configs: mock.Mock,
    ) -> None:
        del download_file
        inspect_draft.side_effect = prep.PreparationError(
            "draft tokenizer does not exactly match target tokenizer"
        )

        with self.assertRaisesRegex(prep.PreparationError, "tokenizer"):
            prep.stage(
                prep.DESTINATION,
                Path("/tmp/unused-eagle3-report.json"),
                workers=4,
                chunk_mib=16,
                retries=4,
            )

        publish_configs.assert_not_called()
        written = atomic_write_json.call_args.args[1]
        self.assertEqual(written["state"], "inactive_compatibility_blocked")
        self.assertTrue(written["download_verification"]["full_sha256_verified"])
        self.assertEqual(written["download_verification"]["sha256"], prep.FILE_SHA256)
        self.assertTrue(written["verdict"]["full_sha256_verified"])
        self.assertFalse(written["verdict"]["artifact_complete"])
        self.assertFalse(written["verdict"]["profile_publication_staged"])


class ProfileTests(unittest.TestCase):
    def test_cpu_profile_keeps_draft_off_gpu(self) -> None:
        args = prep._profile_args("cpu")
        self.assertEqual(args[-4:], ["--spec-draft-device", "none", "--spec-draft-ngl", "0"])
        self.assertIn("draft-eagle3", args)

    def test_gpu_profile_follows_target_device(self) -> None:
        args = prep._profile_args("gpu")
        self.assertNotIn("--spec-draft-device", args)
        self.assertEqual(args[-2:], ["--spec-draft-ngl", "auto"])

    def test_derived_profile_binds_exact_draft_and_receipt(self) -> None:
        config = prep.config_document(
            "cpu",
            prep.DERIVED_REPORT,
            prep.DERIVED_MODEL,
            prep.DERIVED_CPU_CONFIG,
        )
        args = config["args"]
        model_index = args.index("--spec-draft-model") + 1
        self.assertEqual(args[model_index], str(prep.DERIVED_MODEL))
        self.assertEqual(config["draft_artifact_manifest"], "../results/gpt20b-eagle3-derived-artifact.json")


class DerivedMetadataTests(unittest.TestCase):
    def test_exact_target_tokenizer_records_replace_all_draft_tokenizer_metadata(self) -> None:
        source = prep.RawGGUFHeader(
            version=3,
            tensor_count=1,
            metadata=(
                prep.RawMetadataEntry("general.architecture", b"source-architecture"),
                prep.RawMetadataEntry("tokenizer.chat_template", b"source-template"),
                prep.RawMetadataEntry("tokenizer.ggml.tokens", b"source-tokens"),
            ),
            tensor_info=b"tensor-directory",
            header_end=123,
        )
        target_records = tuple(
            prep.RawMetadataEntry(key, f"target:{key}".encode())
            for key in prep.TARGET_TOKENIZER_KEYS
        )
        target = prep.RawGGUFHeader(
            version=3,
            tensor_count=459,
            metadata=(prep.RawMetadataEntry("general.architecture", b"target-architecture"),) + target_records,
            tensor_info=b"target-directory",
            header_end=456,
        )

        derived, selected = prep.target_aligned_metadata_records(source, target)

        self.assertEqual(selected, target_records)
        self.assertIn(source.metadata[0], derived)
        self.assertNotIn(source.metadata[1], derived)
        self.assertNotIn(source.metadata[2], derived)
        self.assertEqual(derived[1:], target_records)

    def test_raw_header_scanner_handles_string_array_without_tensor_payload(self) -> None:
        def text(value: str) -> bytes:
            encoded = value.encode()
            return struct.pack("<Q", len(encoded)) + encoded

        def entry(key: str, value_type: int, value: bytes) -> bytes:
            return text(key) + struct.pack("<I", value_type) + value

        tokens = struct.pack("<IQ", 8, 2) + text("alpha") + text("beta")
        metadata = (
            entry("general.alignment", 4, struct.pack("<I", 32)),
            entry("tokenizer.ggml.tokens", 9, tokens),
        )
        tensor_info = text("d2t") + struct.pack("<IQIQ", 1, 1, 27, 0)
        payload = b"GGUF" + struct.pack("<IQQ", 3, 1, len(metadata)) + b"".join(metadata) + tensor_info
        payload += b"\0" * ((-len(payload)) % 32) + b"payload"

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tiny.gguf"
            path.write_bytes(payload)
            header = prep.scan_raw_gguf_header(path)

        self.assertEqual(header.version, 3)
        self.assertEqual(header.tensor_count, 1)
        self.assertEqual([row.key for row in header.metadata], ["general.alignment", "tokenizer.ggml.tokens"])
        self.assertEqual(header.tensor_info, tensor_info)


class SemanticCompatibilityTests(unittest.TestCase):
    def test_i64_d2t_payload_size_uses_primitive_width(self) -> None:
        class Tensor:
            elements = 64_000

            @staticmethod
            def expected_bytes() -> None:
                return None

        self.assertEqual(prep._tensor_byte_count(Tensor(), "I64"), 512_000)

    @staticmethod
    def _metadata(*, target: bool) -> dict[str, object]:
        semantic = [f"token-{index}" for index in range(200_019)]
        semantic[199_998] = "<|startoftext|>"
        semantic[199_999] = "<|endoftext|>"
        semantic[200_002] = "<|return|>"
        semantic[200_005] = "<|channel|>"
        semantic[200_006] = "<|start|>"
        semantic[200_007] = "<|end|>"
        semantic[200_008] = "<|message|>"
        tail = (
            [f"[PAD{index}]" for index in range(1_069)]
            if target
            else [f"[PAD{index}]" for index in range(200_019, 201_088)]
        )
        return {
            "tokenizer.ggml.tokens": semantic + tail,
            "tokenizer.ggml.token_type": [1] * 200_019 + ([4] * 1_069 if target else [5] * 1_069),
            "tokenizer.ggml.merges": ["a b"],
            "tokenizer.ggml.pre": "default" if target else "gpt-4o",
            "tokenizer.ggml.eos_token_id": 199_999 if target else 200_002,
            "tokenizer.ggml.eos_token_ids": [199_999, 200_002, 200_012] if target else None,
            "tokenizer.ggml.add_eos_token": False if target else None,
        }

    def _audit(self, last_id: int) -> dict[str, object]:
        with tempfile.TemporaryDirectory() as directory:
            draft_path = Path(directory) / "draft.gguf"
            target_path = Path(directory) / "target.gguf"
            d2t = list(range(63_996)) + [200_005, 200_006, 200_007, last_id]
            draft_path.write_bytes(struct.pack("<64000q", *d2t))
            target_path.write_bytes(b"target")
            draft_metadata = self._metadata(target=False)
            target_metadata = self._metadata(target=True)

            class Tensor:
                name = "d2t"
                shape = [64_000]
                type_name = "I64"
                offset = 0

            class Reader:
                def __init__(self, path: Path) -> None:
                    self.metadata = target_metadata if path.name == "target.gguf" else draft_metadata
                    self.data_start = 0
                    self.tensors = [] if path.name == "target.gguf" else [Tensor()]

            return prep.audit_tokenizer_and_d2t(draft_path, target_path, Reader)

    def test_d2t_only_maps_exact_semantic_ids(self) -> None:
        audit = self._audit(200_008)
        self.assertTrue(audit["derivation_safe"])
        self.assertEqual(audit["padding_tail"]["count"], 1_069)
        self.assertEqual(audit["d2t"]["mapped_padding_tail_count"], 0)
        self.assertEqual(audit["d2t"]["harmony_special_output_positions"]["<|message|>"], 63_999)

    def test_d2t_mapping_into_padding_tail_is_rejected(self) -> None:
        with self.assertRaisesRegex(prep.PreparationError, "padding tail"):
            self._audit(200_019)


class DerivedReceiptTests(unittest.TestCase):
    def test_activation_requires_hash_tokenizer_and_tensor_identity(self) -> None:
        derivation = {
            "source": {"sha256": prep.FILE_SHA256},
            "derived": {
                "path": str(prep.DERIVED_MODEL),
                "size": 123,
                "sha256": "derived-sha",
                "fresh_full_sha256_verified": True,
            },
            "tensor_payload_identity": {
                "all_tensor_payload_bytes_unchanged": True,
                "tensor_directory_copied_byte_for_byte": True,
            },
            "tokenizer_metadata": {"exact_raw_match": True},
            "semantic_audit": {
                "derivation_safe": True,
                "d2t": {"all_mapped_token_strings_match": True},
            },
            "draft_validation": {"gguf": {"tensor_count": 16}},
        }
        report = prep.derived_artifact_report(
            source_evidence={"source_support_validated": True},
            target={
                "model": str(prep.TARGET_MODEL),
                "size": prep.TARGET_SIZE,
                "sha256": prep.TARGET_SHA256,
                "fresh_full_sha256_verified": True,
            },
            bundle={"eagle3_loader_present": True},
            derivation=derivation,
        )

        self.assertTrue(report["verdict"]["artifact_complete"])
        self.assertTrue(report["verdict"]["full_sha256_verified"])
        self.assertTrue(report["compatibility"]["activation_allowed"])
        self.assertEqual(report["target"]["sha256"], prep.TARGET_SHA256)
        self.assertEqual(report["gguf_artifact"]["path"], str(prep.DERIVED_MODEL))

    def test_incomplete_derivation_stays_blocked(self) -> None:
        report = prep.derived_artifact_report(
            source_evidence={},
            target={},
            bundle={},
            error=prep.PreparationError("not exact"),
        )
        self.assertFalse(report["verdict"]["artifact_complete"])
        self.assertFalse(report["compatibility"]["activation_allowed"])


if __name__ == "__main__":
    unittest.main()
