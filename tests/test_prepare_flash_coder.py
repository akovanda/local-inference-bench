import hashlib
import json
import os
import struct
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

from runtime import prepare_flash_coder as prep


def _profile_bytes(pairs):
    layers, experts = prep.PROFILE_LAYERS, prep.PROFILE_EXPERTS
    inverse = [-1] * (layers * experts)
    for slot, (layer, expert) in enumerate(pairs):
        inverse[layer * experts + expert] = slot
    return (
        b"STRP"
        + struct.pack("<5I", 1, layers, experts, len(pairs), len(pairs))
        + b"".join(struct.pack("<HH", *pair) for pair in pairs)
        + struct.pack(f"<{layers * experts}i", *inverse)
    )


class PrepareFlashCoderTests(unittest.TestCase):
    def test_profile_validates_all_pruned_pairs_and_inverse_bounds(self):
        pairs = [
            (layer, expert)
            for expert in range(prep.PROFILE_EXPERTS)
            for layer in range(prep.PROFILE_LAYERS)
        ]
        payload = _profile_bytes(pairs)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.bin"
            path.write_bytes(payload)
            record = prep.verify_expert_profile(
                path,
                expected_size=len(payload),
                expected_sha256=hashlib.sha256(payload).hexdigest(),
            )
            self.assertEqual(record["ranked_pairs"], 12_288)
            self.assertEqual(record["unique_pairs"], 12_288)
            self.assertTrue(record["inverse_table_validated"])

            malformed = bytearray(payload)
            struct.pack_into("<HH", malformed, 24, prep.PROFILE_LAYERS, 0)
            path.write_bytes(malformed)
            with self.assertRaisesRegex(prep.PreparationError, "out-of-range"):
                prep.verify_expert_profile(
                    path,
                    expected_size=len(malformed),
                    expected_sha256=hashlib.sha256(malformed).hexdigest(),
                )

    def test_shared_ple_requires_exact_receipt_identity_and_symlink(self):
        payload = b"shared PLE"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.gguf"
            link = root / "coder-00002.gguf"
            receipt = root / "receipt.json"
            source.write_bytes(payload)
            link.symlink_to(source)
            details = source.stat()
            digest = hashlib.sha256(payload).hexdigest()
            spec = prep.FileSpec(link.name, len(payload), digest)
            receipt.write_text(json.dumps({
                "schema_version": 1,
                "verified": True,
                "path": str(source),
                "size": len(payload),
                "sha256": digest,
                "identity": {
                    "device": details.st_dev,
                    "inode": details.st_ino,
                    "mtime_ns": details.st_mtime_ns,
                    "ctime_ns": details.st_ctime_ns,
                },
                "reuse_for": "flash-next-coder",
            }))
            record = prep.verify_shared_ple(link, source, receipt, spec)
            self.assertTrue(record["verified"])
            self.assertEqual(record["source_file_identity"]["inode"], details.st_ino)

            wrong = root / "wrong.gguf"
            wrong.write_bytes(payload)
            link.unlink()
            link.symlink_to(wrong)
            with self.assertRaisesRegex(prep.PreparationError, "wrong source"):
                prep.verify_shared_ple(link, source, receipt, spec)

    def test_shared_ple_rejects_stale_identity_without_rehashing_payload(self):
        payload = b"shared PLE"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.gguf"
            link = root / "coder-00002.gguf"
            receipt = root / "receipt.json"
            source.write_bytes(payload)
            link.symlink_to(source)
            details = source.stat()
            spec = prep.FileSpec(link.name, len(payload), hashlib.sha256(payload).hexdigest())
            receipt.write_text(json.dumps({
                "schema_version": 1,
                "verified": True,
                "path": str(source),
                "size": len(payload),
                "sha256": spec.sha256,
                "identity": {
                    "device": details.st_dev,
                    "inode": details.st_ino + 1,
                    "mtime_ns": details.st_mtime_ns,
                    "ctime_ns": details.st_ctime_ns,
                },
                "reuse_for": "flash-next-coder",
            }))
            with mock.patch.object(prep, "sha256_file", wraps=prep.sha256_file) as hasher:
                with self.assertRaisesRegex(prep.PreparationError, "receipt"):
                    prep.verify_shared_ple(link, source, receipt, spec)
            hasher.assert_not_called()

    def test_first_shard_requires_stable_full_sha(self):
        payload = b"coder first shard"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = prep.FileSpec("first.gguf", len(payload), hashlib.sha256(payload).hexdigest())
            path = root / spec.name
            path.write_bytes(payload)
            with mock.patch.object(prep, "MODEL_DIR", root):
                record = prep.verify_first_shard(spec)
                self.assertTrue(record["full_sha256_verified"])
                path.write_bytes(b"x" * len(payload))
                with self.assertRaisesRegex(prep.PreparationError, "SHA-256"):
                    prep.verify_first_shard(spec)

    def test_config_is_8k_greedy_pruned_reference_with_external_tail_env(self):
        config = prep.config_document()
        args = config["args"]
        self.assertEqual(args[args.index("--max-context") + 1], "8192")
        self.assertEqual(args[args.index("--ple-io") + 1], "ram")
        self.assertEqual(args[args.index("--prefill") + 1], "auto")
        self.assertEqual(args[args.index("--expert-cache") + 1], "auto")
        self.assertEqual(args[args.index("--suffix-draft") + 1], "0")
        self.assertNotIn("--mtp", args)
        self.assertEqual(config["host"], "127.0.0.1")
        self.assertEqual(config["port"], 18080)
        self.assertEqual(config["sampling"], {"temperature": 0, "presence_penalty": 0})
        self.assertEqual(config["reference_environment"]["STRATA_ROUTE_TAIL_SKIP"], "0")
        self.assertFalse(config["reference_environment"]["applied_by_this_config"])

    def test_existing_pack_reused_only_with_exact_provenance(self):
        verification = {
            "shards": [
                {"name": prep.FIRST.name, "size": prep.FIRST.size, "sha256": prep.FIRST.sha256},
                {
                    "name": prep.SECOND.name,
                    "size": prep.SECOND.size,
                    "sha256": prep.SECOND.sha256,
                    "verification_receipt_sha256": "a" * 64,
                },
            ],
            "toolchain": {"strata_revision": prep.STRATA_REVISION},
        }
        with tempfile.TemporaryDirectory() as directory:
            pack_dir = Path(directory)
            for relative in prep.REQUIRED_PACK_FILES:
                path = pack_dir / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"fixture")
            with mock.patch.object(prep, "PACK_DIR", pack_dir):
                marker = pack_dir / "flash-next-coder-source.json"
                marker.write_text(json.dumps(prep.pack_provenance(verification)))
                with mock.patch.object(prep.subprocess, "run") as run:
                    files, rebuilt, provenance = prep.prepare_pack(verification)
            run.assert_not_called()
            self.assertFalse(rebuilt)
            self.assertEqual(provenance["revision"], prep.REVISION)
            self.assertEqual(len(files), len(prep.REQUIRED_PACK_FILES))

    def test_artifact_keeps_runtime_claims_false_and_mtp_unavailable(self):
        verification = {
            "model": {"model_name": prep.MODEL_NAME},
            "shards": [{"name": "one"}, {"name": "two"}],
            "expert_profile": {"verified": True},
            "toolchain": {"engine": {"sha256": prep.ENGINE_SHA256}},
        }
        config = prep.config_document()
        with mock.patch.object(prep, "sha256_file", return_value="b" * 64):
            artifact = prep.build_artifact(
                verification, [], False, {"revision": prep.REVISION}, config
            )
        self.assertTrue(artifact["verdict"]["artifact_complete"])
        self.assertFalse(artifact["verdict"]["runtime_load_validated"])
        self.assertFalse(artifact["runtime"]["actual_gpu_placement_validated"])
        self.assertEqual(artifact["runtime"]["config_sha256"], "b" * 64)
        self.assertFalse(artifact["mtp"]["enabled"])
        self.assertFalse(artifact["mtp"]["retried"])


if __name__ == "__main__":
    unittest.main()
