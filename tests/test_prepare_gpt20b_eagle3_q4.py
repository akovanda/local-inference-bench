from __future__ import annotations

import unittest
from unittest import mock

from runtime import prepare_gpt20b_eagle3_q4 as prep


class QuantizerCommandTests(unittest.TestCase):
    def test_command_is_bounded_pure_q4_requantization(self) -> None:
        command = prep.quantizer_argv(prep.SOURCE_MODEL, prep.MODEL, 4)
        self.assertEqual(command[0], str(prep.QUANTIZER))
        self.assertEqual(
            command[1:8],
            [
                "--allow-requantize",
                "--pure",
                "--output-tensor-type",
                "q4_0",
                "--token-embedding-type",
                "q4_0",
                str(prep.SOURCE_MODEL),
            ],
        )
        self.assertEqual(command[-3:], [str(prep.MODEL), "Q4_0", "4"])

    def test_command_rejects_unplanned_thread_count(self) -> None:
        with self.assertRaisesRegex(prep.PreparationError, "exactly 4"):
            prep.quantizer_argv(prep.SOURCE_MODEL, prep.MODEL, 8)


class SourceReceiptTests(unittest.TestCase):
    @mock.patch.object(prep, "_read_json")
    def test_source_receipt_requires_activation_and_exact_identities(self, read_json: mock.Mock) -> None:
        read_json.return_value = {
            "verdict": {"artifact_complete": True, "full_sha256_verified": True},
            "compatibility": {"activation_allowed": True},
            "gguf_artifact": {
                "path": str(prep.SOURCE_MODEL),
                "size": prep.SOURCE_SIZE,
                "sha256": prep.SOURCE_SHA256,
            },
            "target": {
                "path": str(prep.q8.TARGET_MODEL),
                "size": prep.q8.TARGET_SIZE,
                "sha256": prep.q8.TARGET_SHA256,
            },
        }
        source = prep.validate_source_receipt()
        self.assertEqual(source["sha256"], prep.SOURCE_SHA256)
        self.assertEqual(source["target"]["sha256"], prep.q8.TARGET_SHA256)

    @mock.patch.object(prep, "_read_json")
    def test_source_receipt_rejects_blocked_activation(self, read_json: mock.Mock) -> None:
        read_json.return_value = {
            "verdict": {"artifact_complete": True, "full_sha256_verified": True},
            "compatibility": {"activation_allowed": False},
            "gguf_artifact": {},
            "target": {},
        }
        with self.assertRaisesRegex(prep.PreparationError, "activation-ready"):
            prep.validate_source_receipt()


class ConfigTests(unittest.TestCase):
    def test_gpu_max3_config_binds_q4_artifact_and_receipt(self) -> None:
        config = prep.config_document(3)
        args = config["args"]
        self.assertEqual(args[args.index("--spec-draft-model") + 1], str(prep.MODEL))
        self.assertEqual(args[args.index("--spec-draft-n-max") + 1], "3")
        self.assertEqual(config["draft_artifact_manifest"], "../results/gpt20b-eagle3-q4-artifact.json")
        self.assertEqual(config["draft_quantization"], "Q4_0")
        self.assertEqual(config["draft_profile"], "gpu_q4_0_max3")

    def test_gpu_max1_changes_only_bounded_draft_length(self) -> None:
        max3 = prep.config_document(3)
        max1 = prep.config_document(1)
        args3 = list(max3["args"])
        args1 = list(max1["args"])
        index = args3.index("--spec-draft-n-max") + 1
        self.assertEqual(args3[index], "3")
        self.assertEqual(args1[index], "1")
        args3[index] = args1[index]
        self.assertEqual(args3, args1)

    def test_config_rejects_unplanned_max(self) -> None:
        with self.assertRaisesRegex(prep.PreparationError, "1 or 3"):
            prep.config_document(2)


class ReceiptTests(unittest.TestCase):
    def setUp(self) -> None:
        self.source = {
            "path": str(prep.SOURCE_MODEL),
            "size": prep.SOURCE_SIZE,
            "sha256": prep.SOURCE_SHA256,
        }
        self.target = {
            "model": str(prep.q8.TARGET_MODEL),
            "size": prep.q8.TARGET_SIZE,
            "sha256": prep.q8.TARGET_SHA256,
            "fresh_full_sha256_verified": True,
        }
        self.quantizer = {"binary_sha256": "binary", "implementation_sha256": "impl"}
        self.command = prep.quantizer_argv(prep.SOURCE_MODEL, prep.MODEL, 4)

    def test_complete_candidate_activates_only_with_hash_tokenizer_and_d2t(self) -> None:
        candidate = {
            "path": str(prep.MODEL),
            "size": 123,
            "sha256": "candidate-sha",
            "fresh_full_sha256_verified": True,
            "compatibility": {
                "target_tokenizer_exact_match": True,
                "d2t_unchanged": True,
            },
        }
        report = prep.artifact_report(
            source=self.source,
            target=self.target,
            quantizer=self.quantizer,
            command=self.command,
            candidate=candidate,
        )
        self.assertTrue(report["verdict"]["artifact_complete"])
        self.assertTrue(report["verdict"]["full_sha256_verified"])
        self.assertTrue(report["compatibility"]["activation_allowed"])
        self.assertTrue(report["conversion"]["lossy"])
        self.assertFalse(report["conversion"]["target_weights_changed"])
        self.assertEqual(report["target"]["sha256"], prep.q8.TARGET_SHA256)

    def test_missing_d2t_identity_stays_blocked(self) -> None:
        candidate = {
            "path": str(prep.MODEL),
            "size": 123,
            "sha256": "candidate-sha",
            "fresh_full_sha256_verified": True,
            "compatibility": {
                "target_tokenizer_exact_match": True,
                "d2t_unchanged": False,
            },
        }
        report = prep.artifact_report(
            source=self.source,
            target=self.target,
            quantizer=self.quantizer,
            command=self.command,
            candidate=candidate,
        )
        self.assertFalse(report["verdict"]["artifact_complete"])
        self.assertFalse(report["compatibility"]["activation_allowed"])
        self.assertFalse(report["verdict"]["profile_publication_staged"])

    def test_error_receipt_preserves_lossy_scope_without_activation(self) -> None:
        report = prep.artifact_report(
            source=self.source,
            target=self.target,
            quantizer=self.quantizer,
            command=self.command,
            error=prep.PreparationError("quantizer failed"),
        )
        self.assertEqual(report["state"], "inactive_q4_blocked")
        self.assertEqual(report["error"], "PreparationError")
        self.assertFalse(report["compatibility"]["activation_allowed"])


if __name__ == "__main__":
    unittest.main()
