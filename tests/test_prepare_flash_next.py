import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from runtime import prepare_flash_next


class PrepareFlashNextTests(unittest.TestCase):
    def test_verify_file_accepts_only_exact_size_and_hash(self):
        payload = b"pinned flash-next shard fixture"
        with tempfile.TemporaryDirectory() as directory:
            model_dir = Path(directory)
            spec = prepare_flash_next.FileSpec(
                "shard.gguf", len(payload), hashlib.sha256(payload).hexdigest()
            )
            (model_dir / spec.name).write_bytes(payload)
            with mock.patch.object(prepare_flash_next, "MODEL_DIR", model_dir):
                record = prepare_flash_next.verify_file(spec)
            self.assertTrue(record["verified"])
            self.assertEqual(record["size"], len(payload))
            self.assertEqual(record["sha256"], spec.sha256)

            (model_dir / spec.name).write_bytes(payload + b"corrupt")
            with mock.patch.object(prepare_flash_next, "MODEL_DIR", model_dir):
                with self.assertRaisesRegex(RuntimeError, "expected"):
                    prepare_flash_next.verify_file(spec)

    def test_runtime_config_requires_greedy_no_mtp_baseline(self):
        config = {
            "exe": str(prepare_flash_next.ENGINE),
            "args": [
                "--pack", str(prepare_flash_next.PACK_DIR),
                "--native", str(prepare_flash_next.FILES[0].path),
                "--ple-gguf", str(prepare_flash_next.FILES[1].path),
                "--ple-io", "ram",
                "--max-context", "32768",
                "--prefill", "auto",
                "--expert-profile", str(prepare_flash_next.EXPERT_PROFILE),
                "--expert-cache", "auto",
                "--spec", "4",
                "--suffix-draft", "0",
            ],
            "cwd": str(prepare_flash_next.STRATA_ROOT),
            "tokenizer": str(prepare_flash_next.PACK_DIR / "tokenizer"),
            "model_name": "qwen3.8-flash-next-iq3_s",
            "arch": "qwen4exp",
            "lib_dirs": ["/usr/local/cuda-13.0/lib64"],
            "host": "127.0.0.1",
            "port": 18080,
            "open_browser": False,
            "sampling": {"temperature": 0, "presence_penalty": 0},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            with mock.patch.object(prepare_flash_next, "CONFIG", path):
                self.assertEqual(prepare_flash_next.validate_runtime_config(), config)
                config["args"].extend(("--mtp", "/tmp/mtp"))
                path.write_text(json.dumps(config), encoding="utf-8")
                with self.assertRaisesRegex(RuntimeError, "MTP"):
                    prepare_flash_next.validate_runtime_config()

    def test_expert_profile_requires_complete_pinned_flash_ranking(self):
        with tempfile.TemporaryDirectory() as directory:
            profile = Path(directory) / "expert-profile.bin"
            profile.write_bytes(prepare_flash_next.EXPERT_PROFILE.read_bytes())
            with mock.patch.object(prepare_flash_next, "EXPERT_PROFILE", profile):
                record = prepare_flash_next.verify_expert_profile()
                self.assertEqual(record["layers"], 48)
                self.assertEqual(record["experts_per_layer"], 512)
                self.assertEqual(record["ranked_pairs"], 24_576)
                self.assertTrue(record["verified"])

                profile.write_bytes(profile.read_bytes()[:-1] + b"x")
                with self.assertRaisesRegex(RuntimeError, "not the file pinned"):
                    prepare_flash_next.verify_expert_profile()

    def test_existing_pack_is_reused_only_with_exact_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            pack_dir = Path(directory)
            for relative in prepare_flash_next.REQUIRED_PACK_FILES:
                path = pack_dir / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"fixture")
            marker = pack_dir / "flash-next-source.json"
            with mock.patch.object(prepare_flash_next, "PACK_DIR", pack_dir):
                marker.write_text(
                    json.dumps(prepare_flash_next.pack_provenance()), encoding="utf-8"
                )
                with mock.patch("runtime.prepare_flash_next.subprocess.run") as run:
                    files, rebuilt = prepare_flash_next.prepare_pack()
            run.assert_not_called()
            self.assertFalse(rebuilt)
            self.assertEqual(len(files), len(prepare_flash_next.REQUIRED_PACK_FILES))


if __name__ == "__main__":
    unittest.main()
