from __future__ import annotations

import hashlib
import json
import os
import stat
import struct
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from runtime import prepare_bonsai2 as prep


class ModelIdentityTests(unittest.TestCase):
    def test_exact_published_model_identities_are_pinned(self) -> None:
        self.assertEqual(prep.MODEL_REVISION, "b072e1d3b35a0a630cece372c2127528e0994386")
        self.assertEqual(prep.MODELS["PTQ1_0"].size, 5_946_648_928)
        self.assertEqual(
            prep.MODELS["PTQ1_0"].sha256,
            "53107f530aa52eb00912263ab1ee29bd199261c87cd7b4ad4ca1318c1fe33ee3",
        )
        self.assertEqual(prep.MODELS["PQ2_0"].size, 7_206_168_928)
        self.assertEqual(
            prep.MODELS["PQ2_0"].sha256,
            "3907dc1658db1f78a9826bf8d5bcb8dc65db0d466388937af57f2294fae62ec1",
        )

    def test_model_verification_rejects_same_size_wrong_content(self) -> None:
        payload = b"pinned model fixture"
        spec = prep.ModelSpec("FIXTURE", "model.gguf", len(payload), hashlib.sha256(payload).hexdigest())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / spec.name
            path.write_bytes(payload)
            template = {
                "verified": True,
                "capabilities": {"enable_thinking_false": True},
            }
            with mock.patch.object(prep, "inspect_chat_template", return_value=template) as inspect:
                self.assertTrue(prep.verify_model(spec, path)["verified"])
                inspect.assert_called_once_with(path)
                path.write_bytes(b"x" * len(payload))
                with self.assertRaisesRegex(prep.PreparationError, "SHA-256"):
                    prep.verify_model(spec, path)
                inspect.assert_called_once_with(path)

    def test_chat_template_attestation_reads_gguf_metadata(self) -> None:
        template = (
            "{%- if enable_thinking is defined and enable_thinking is false %}"
            "{{- '<think>\\n\\n</think>\\n\\n' }}"
        ).encode("utf-8")

        def gguf_string(value: bytes) -> bytes:
            return struct.pack("<Q", len(value)) + value

        payload = b"".join(
            (
                b"GGUF",
                struct.pack("<IQQ", 3, 0, 2),
                gguf_string(b"general.name"),
                struct.pack("<I", 8),
                gguf_string(b"fixture"),
                gguf_string(prep.CHAT_TEMPLATE_KEY.encode("utf-8")),
                struct.pack("<I", 8),
                gguf_string(template),
            )
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.gguf"
            path.write_bytes(payload)
            result = prep.inspect_chat_template(
                path,
                expected_sha256=hashlib.sha256(template).hexdigest(),
            )
        self.assertTrue(result["verified"])
        self.assertTrue(result["capabilities"]["enable_thinking_false"])


class RuntimeInspectionTests(unittest.TestCase):
    def test_cuda_arch_check_requires_static_sm75_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            backend = Path(directory) / "libggml-cuda.so"
            backend.write_bytes(b"fixture")
            completed = mock.Mock(stdout="ELF file 1: sm_75.cubin\nELF file 2: sm_89.cubin\n", stderr="")
            with mock.patch("runtime.prepare_bonsai2.subprocess.run", return_value=completed):
                with mock.patch.object(prep, "CUOBJDUMP", Path("/bin/true")):
                    result = prep.inspect_cuda_architectures(backend)
            self.assertEqual(result["architectures"], [75, 89])
            self.assertTrue(result["tesla_t4_sm75_present"])
            self.assertIn("not loaded", result["method"])

    def test_cuda_arch_check_rejects_release_without_t4_code(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            backend = Path(directory) / "libggml-cuda.so"
            backend.write_bytes(b"fixture")
            completed = mock.Mock(stdout="ELF file 1: sm_80.cubin\n", stderr="")
            with mock.patch("runtime.prepare_bonsai2.subprocess.run", return_value=completed):
                with mock.patch.object(prep, "CUOBJDUMP", Path("/bin/true")):
                    with self.assertRaisesRegex(prep.PreparationError, "lacks sm_75"):
                        prep.inspect_cuda_architectures(backend)

    def test_regular_file_verification_rejects_symlink(self) -> None:
        payload = b"archive"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "target"
            target.write_bytes(payload)
            link = root / "link"
            link.symlink_to(target)
            with self.assertRaisesRegex(prep.PreparationError, "not a regular file"):
                prep.verify_regular_file(
                    link,
                    size=len(payload),
                    sha256=hashlib.sha256(payload).hexdigest(),
                    label="fixture",
                )

    def test_source_audit_requires_both_fork_formats_and_transforms(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixtures = {
                "ggml/include/ggml.h": "GGML_TYPE_PQ2_0 GGML_TYPE_PTQ1_0",
                "ggml/src/ggml-cuda/mmvq-pq2_0.cuh": "mul_mat_vec_pq2_0",
                "ggml/src/ggml-cuda/mmvq-ptq1_0.cuh": "mul_mat_vec_ptq1_0",
                "ggml/src/ggml-cuda/fwht.cu": "ggml_cuda_op_fwht ggml_cuda_op_fwht_signed",
                "src/llama-context.cpp": "Hadamard-folded weight activation transform",
                "src/models/qwen35.cpp": "hadamard",
            }
            for relative, content in fixtures.items():
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content, encoding="utf-8")
            with mock.patch.object(prep, "SOURCE_DIR", root):
                result = prep.verify_source_compatibility()
                self.assertTrue(result["pq2_0_cuda_source_present"])
                (root / "ggml/include/ggml.h").write_text("GGML_TYPE_PQ2_0", encoding="utf-8")
                with self.assertRaisesRegex(prep.PreparationError, "PTQ1_0"):
                    prep.verify_source_compatibility()


class ReceiptTests(unittest.TestCase):
    def setUp(self) -> None:
        self.spec = prep.MODELS["PTQ1_0"]
        self.template = {
            "metadata_key": prep.CHAT_TEMPLATE_KEY,
            "sha256": prep.CHAT_TEMPLATE_SHA256,
            "expected_sha256": prep.CHAT_TEMPLATE_SHA256,
            "verified": True,
            "capabilities": {
                "enable_thinking_false": True,
                "empty_think_generation_prompt": True,
            },
        }
        self.runtime = {
            "server": {"path": "/runtime/llama-server"},
            "lib_dirs": ["/runtime"],
            "cuda_backend": {"fatbinary": {"tesla_t4_sm75_present": True}},
        }

    def test_runtime_only_receipt_does_not_claim_model_or_gpu_readiness(self) -> None:
        report = prep.make_receipt(
            spec=self.spec,
            model_path=self.spec.default_path,
            runtime=self.runtime,
            model=None,
        )
        self.assertEqual(report["state"], "runtime_ready_model_pending")
        self.assertTrue(report["verdict"]["runtime_artifact_complete"])
        self.assertFalse(report["verdict"]["artifact_complete"])
        self.assertFalse(report["verdict"]["gpu_load_proven"])
        self.assertEqual(report["shards"], [])

    def test_verified_model_receipt_is_llama_trial_compatible_at_32k(self) -> None:
        model = {
            "name": self.spec.name,
            "path": str(self.spec.default_path),
            "size": self.spec.size,
            "sha256": self.spec.sha256,
            "verified": True,
            "chat_template": self.template,
        }
        report = prep.make_receipt(
            spec=self.spec,
            model_path=self.spec.default_path,
            runtime=self.runtime,
            model=model,
        )
        config = report["trial_config_template"]
        self.assertTrue(report["verdict"]["artifact_complete"])
        self.assertFalse(report["verdict"]["gpu_load_proven"])
        self.assertEqual(config["context"], 32768)
        self.assertEqual(config["host"], "127.0.0.1")
        self.assertEqual(config["port"], 18081)
        self.assertEqual(config["shards"][0]["sha256"], self.spec.sha256)
        self.assertEqual(config["args"][config["args"].index("--gpu-layers") + 1], "all")
        self.assertEqual(config["args"][config["args"].index("--fit") + 1], "off")
        self.assertEqual(report["model"]["chat_template"]["sha256"], prep.CHAT_TEMPLATE_SHA256)
        self.assertIsNone(report["template_capabilities"])

    def test_one_receipt_can_attest_both_packings(self) -> None:
        selected = {
            "name": prep.MODELS["PQ2_0"].name,
            "path": str(prep.MODELS["PQ2_0"].default_path),
            "size": prep.MODELS["PQ2_0"].size,
            "sha256": prep.MODELS["PQ2_0"].sha256,
            "verified": True,
            "chat_template": self.template,
        }
        paired = {
            "name": prep.MODELS["PTQ1_0"].name,
            "path": str(prep.MODELS["PTQ1_0"].default_path),
            "size": prep.MODELS["PTQ1_0"].size,
            "sha256": prep.MODELS["PTQ1_0"].sha256,
            "verified": True,
            "chat_template": self.template,
        }
        report = prep.make_receipt(
            spec=prep.MODELS["PQ2_0"],
            model_path=prep.MODELS["PQ2_0"].default_path,
            runtime=self.runtime,
            model=selected,
            other_models=[paired],
        )
        self.assertEqual({row["name"] for row in report["shards"]}, {selected["name"], paired["name"]})
        self.assertEqual(set(report["trial_config_templates"]), {"PQ2_0", "PTQ1_0"})
        self.assertEqual(
            report["template_capabilities"],
            {
                "chat_template_sha256": prep.CHAT_TEMPLATE_SHA256,
                "enable_thinking_false_supported": True,
                "enable_thinking_false_behavior": "empty_think_block",
            },
        )

    def test_source_fallback_is_pinned_to_sm75_and_four_jobs(self) -> None:
        commands = "\n".join(prep.SOURCE_BUILD_COMMANDS)
        self.assertIn(prep.LLAMA_REVISION, commands)
        self.assertIn("CMAKE_CUDA_ARCHITECTURES=75", commands)
        self.assertIn("GGML_CUDA_FA_QUANTS=f16-f16", commands)
        self.assertIn("--parallel 4", commands)
        self.assertEqual(prep.BUILD_OBSERVATION["active_parallel_jobs"], 4)

    def test_cli_defaults_to_frozen_study_context_and_allows_32k(self) -> None:
        self.assertEqual(prep.parse_args([]).context, 8192)
        self.assertEqual(prep.parse_args(["--context", "32768"]).context, 32768)
        with self.assertRaises(SystemExit):
            prep.parse_args(["--context", "128"])

    def test_receipt_records_medium_output_profile(self) -> None:
        report = prep.make_receipt(
            spec=self.spec,
            model_path=self.spec.default_path,
            runtime=self.runtime,
            model=None,
            context=8192,
        )
        self.assertEqual(report["trial_config_template"]["context"], 8192)
        self.assertEqual(report["study_profile"]["quality_tier"], "medium")
        self.assertEqual(report["study_profile"]["code_output_tokens"], 4096)
        self.assertEqual(report["study_profile"]["prose_output_tokens"], 2048)
        self.assertFalse(report["study_profile"]["low_tier_enabled"])

    def test_both_packings_share_parent_staging_directory(self) -> None:
        self.assertEqual(prep.MODELS["PQ2_0"].default_path.parent, prep.MODEL_DIR)
        self.assertEqual(prep.MODELS["PTQ1_0"].default_path.parent, prep.MODEL_DIR)


if __name__ == "__main__":
    unittest.main()
