import contextlib
import hashlib
import io
import json
import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

from runtime import llama_trial
from runtime import prepare_specialists as prep


class PrepareSpecialistsTests(unittest.TestCase):
    def _spec(self, root: Path, payload: bytes = b"specialist") -> tuple[prep.ModelSpec, Path]:
        digest = hashlib.sha256(payload).hexdigest()
        spec = replace(
            prep.SPECS["lfm25-qad-q4_0"],
            filename="fixture.gguf",
            size=len(payload),
            sha256=digest,
            receipt_name="fixture-artifact.json",
            config_name="fixture.json",
        )
        model_dir = root / "models" / "fixture"
        model_dir.mkdir(parents=True)
        (model_dir / spec.filename).write_bytes(payload)
        return spec, model_dir

    def test_primary_identities_and_optional_fallback_are_explicit(self):
        document = prep.inspect_document()
        by_key = {row["key"]: row for row in document["models"]}
        self.assertEqual(by_key["lfm25-qad-q4_0"]["size"], 13_578_190_880)
        self.assertEqual(
            by_key["kat-v25-iq3_xxs"]["sha256"],
            "80453a60bfbd277ae6110413d5b7a4fef07d69997ef88e42adfe3cd650373f79",
        )
        self.assertEqual(by_key["gemma4-style12-q5_k_m"]["revision"],
                         "fb2e8be6efb2905308e57ce248f209c6ddf99983")
        fallback = by_key["kat-v25-iq2_m"]
        self.assertFalse(fallback["activation_allowed"])
        self.assertEqual(fallback["availability"], "metadata_unavailable")
        self.assertEqual(fallback["policy_event"]["reviewer_origin"], "unknown")
        self.assertIn("AskForApproval is set to Never", fallback["policy_event"]["detail"])

    def test_inspect_cli_is_read_only_and_prints_json(self):
        output = io.StringIO()
        with mock.patch.object(prep, "_write_json_exclusive") as write, contextlib.redirect_stdout(output):
            self.assertEqual(prep.main(["--inspect", "--model", "kat-v25-iq3_xxs"]), 0)
        write.assert_not_called()
        self.assertEqual(json.loads(output.getvalue())["models"][0]["key"], "kat-v25-iq3_xxs")

    def test_iq2_fallback_fails_closed_for_mutating_actions(self):
        with mock.patch.object(prep.range_fetch, "download_file") as download:
            self.assertEqual(prep.main(["--download", "--model", "kat-v25-iq2_m"]), 1)
        download.assert_not_called()

    def test_verify_file_checks_full_digest_and_rejects_wrong_content(self):
        with tempfile.TemporaryDirectory() as directory:
            spec, model_dir = self._spec(Path(directory))
            identity = prep.verify_file(spec, model_dir / spec.filename)
            self.assertEqual(identity["size"], spec.size)
            (model_dir / spec.filename).write_bytes(b"SPECIALIST")
            with self.assertRaisesRegex(prep.PreparationError, "SHA-256"):
                prep.verify_file(spec, model_dir / spec.filename)

    def test_download_uses_exact_pin_and_publishes_full_sha_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = b"specialist"
            spec, model_dir = self._spec(root, payload)
            receipt = root / "receipt.json"

            def fake_download(url, output, size, sha256, **kwargs):
                self.assertEqual(url, spec.url)
                self.assertEqual((size, sha256), (spec.size, spec.sha256))
                self.assertEqual(kwargs["workers"], 3)
                self.assertEqual(kwargs["chunk_size"], 16 * 1024 * 1024)
                return output

            with mock.patch.object(prep.range_fetch, "download_file", side_effect=fake_download):
                result = prep.download_model(
                    spec, model_dir, workers=3, chunk_mib=16, retries=2, receipt_path=receipt
                )
            self.assertTrue(result["verdict"]["artifact_complete"])
            self.assertTrue(result["verdict"]["full_sha256_verified"])
            self.assertEqual(result["shards"][0]["sha256"], spec.sha256)
            self.assertEqual(json.loads(receipt.read_text()), result)

    def test_immutable_receipt_is_reused_but_conflict_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec, model_dir = self._spec(root)
            artifact = model_dir / spec.filename
            receipt = root / "receipt.json"
            identity = prep.verify_file(spec, artifact)
            first = prep.publish_receipt(spec, artifact, identity, "test", receipt)
            before = receipt.read_bytes()
            second = prep.publish_receipt(spec, artifact, identity, "different label", receipt)
            self.assertEqual(first, second)
            self.assertEqual(receipt.read_bytes(), before)
            bad = json.loads(receipt.read_text())
            bad["shards"][0]["sha256"] = "0" * 64
            receipt.write_text(json.dumps(bad))
            with self.assertRaisesRegex(prep.PreparationError, "does not match"):
                prep.publish_receipt(spec, artifact, identity, "test", receipt)

    def test_config_is_llama_trial_compatible_and_marks_auto_placement_unmeasured(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec, model_dir = self._spec(root)
            artifact = model_dir / spec.filename
            receipt = root / "receipt.json"
            config = root / "runtime" / "config.json"
            server = root / "bundle" / "llama-server"
            lib_root = server.parent
            cuda = lib_root / "cuda_v13"
            cuda.mkdir(parents=True)
            server.write_text("#!/bin/sh\n")
            server.chmod(0o755)
            (cuda / llama_trial.CUDA_BACKEND_LIBRARY).write_bytes(b"cuda")
            identity = prep.verify_file(spec, artifact)
            prep.publish_receipt(spec, artifact, identity, "test full SHA", receipt)
            document = prep.config_document(
                spec, artifact, receipt, config, server=server, lib_dirs=(lib_root, cuda)
            )
            config.parent.mkdir()
            config.write_text(json.dumps(document))
            loaded = llama_trial.load_trial_config(config)
            self.assertEqual(loaded.model_name, spec.alias)
            self.assertEqual(loaded.context, 8192)
            self.assertEqual(loaded.args[:2], ("--gpu-layers", "auto"))
            self.assertFalse(document["placement"]["all_gpu_fit_claimed"])
            self.assertEqual(document["placement"]["measurement_required"],
                             ["actual_gpu_layer_count", "peak_vram_bytes"])

    def test_prepare_config_hashes_unreceipted_file_then_publishes_both_records(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec, model_dir = self._spec(root)
            receipt = root / "receipt.json"
            config = root / "config.json"
            server = root / "llama-server"
            server.write_text("server")
            server.chmod(0o755)
            lib = root / "lib"
            lib.mkdir()
            document = prep.prepare_config(
                spec,
                model_dir,
                receipt_path=receipt,
                config_path=config,
                server=server,
                lib_dirs=(lib,),
            )
            self.assertTrue(receipt.is_file())
            self.assertEqual(json.loads(config.read_text()), document)
            self.assertFalse(document["runtime_validation"]["load_validated"])

    def test_config_publication_refuses_to_replace_different_existing_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text("{}")
            with self.assertRaisesRegex(prep.PreparationError, "refusing to overwrite"):
                prep._publish_config(path, {"schema_version": 1})
            self.assertEqual(path.read_text(), "{}")

    def test_override_directory_must_remain_under_raid_model_root(self):
        spec = prep.SPECS["lfm25-qad-q4_0"]
        with self.assertRaisesRegex(prep.PreparationError, "must remain under"):
            prep.resolve_model_dir(spec, Path("/external/models/forbidden"))


if __name__ == "__main__":
    unittest.main()
