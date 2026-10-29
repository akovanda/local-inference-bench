import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from runtime import diagnose_native


class DiagnoseNativeTests(unittest.TestCase):
    def test_cpu_environment_preserves_identity_and_disables_accelerators(self):
        descriptor = {
            "launch": {
                "env": {
                    "HOME": "/wrong",
                    "CODEX_HOME": "/wrong-codex",
                    "LD_LIBRARY_PATH": "/runtime/lib",
                    "OLLAMA_LLM_LIBRARY": "cuda_v13",
                    "OLLAMA_MODELS": "/models",
                }
            }
        }
        environment = diagnose_native.cpu_only_environment(
            descriptor,
            base_env={"HOME": "/home/test", "CODEX_HOME": "/codex/test"},
        )
        self.assertEqual(environment["HOME"], "/home/test")
        self.assertEqual(environment["CODEX_HOME"], "/codex/test")
        self.assertEqual(environment["LD_LIBRARY_PATH"], "/runtime/lib")
        self.assertEqual(environment["OLLAMA_MODELS"], "/models")
        self.assertEqual(environment["OLLAMA_LLM_LIBRARY"], "cpu")
        self.assertEqual(environment["CUDA_VISIBLE_DEVICES"], "-1")
        self.assertEqual(environment["OLLAMA_VULKAN"], "false")

    def test_verified_copy_refuses_large_blob_and_preserves_digest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = b"metadata"
            digest = __import__("hashlib").sha256(payload).hexdigest()
            source = root / f"sha256-{digest}"
            destination = root / "copy"
            source.write_bytes(payload)
            source.chmod(0o444)
            with mock.patch.object(os, "access", side_effect=lambda p, mode: not bool(mode & os.W_OK)):
                diagnose_native._copy_verified(source, destination, digest, len(payload))
                with self.assertRaisesRegex(diagnose_native.DiagnosisError, "large model blob"):
                    diagnose_native._copy_verified(source, root / "large", digest, 17 * 1024 * 1024)
            self.assertEqual(destination.read_bytes(), payload)

    def test_fixture_descriptor_only_redirects_model_store(self):
        descriptor = {
            "schema_version": 1,
            "launch": {
                "argv": ["/runtime/ollama", "serve"],
                "env": {
                    "LD_LIBRARY_PATH": "/runtime/lib",
                    "OLLAMA_MODELS": "/original/models",
                },
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            fixture = Path(directory)
            path = diagnose_native.write_fixture_descriptor(descriptor, fixture)
            written = json.loads(path.read_text())
            second = diagnose_native.write_fixture_descriptor(descriptor, fixture)
        self.assertEqual(path, second)
        self.assertEqual(written["launch"]["env"]["OLLAMA_MODELS"], str(fixture))
        self.assertEqual(written["launch"]["env"]["LD_LIBRARY_PATH"], "/runtime/lib")
        self.assertEqual(written["launch"]["argv"], ["/runtime/ollama", "serve"])
        self.assertEqual(set(written), set(descriptor))
        self.assertEqual(descriptor["launch"]["env"]["OLLAMA_MODELS"], "/original/models")

    def test_model_inspection_checks_manifest_names_sizes_and_read_only_state(self):
        with tempfile.TemporaryDirectory() as directory:
            models = Path(directory)
            blobs = models / "blobs"
            manifest_dir = models / "manifests/registry.ollama.ai/library/gpt-oss-20b"
            blobs.mkdir()
            manifest_dir.mkdir(parents=True)
            digest = "sha256:" + "a" * 64
            blob = blobs / digest.replace(":", "-")
            blob.write_bytes(b"data")
            config_digest = "sha256:" + "b" * 64
            config_blob = blobs / config_digest.replace(":", "-")
            config_blob.write_bytes(b"{}")
            manifest = manifest_dir / "64k"
            manifest.write_text(
                json.dumps(
                    {
                        "schemaVersion": 2,
                        "config": {"digest": config_digest, "size": 2},
                        "layers": [{"digest": digest, "size": 4}],
                    }
                )
            )
            for path in (models, blobs, models / "manifests", manifest, blob, config_blob):
                path.chmod(0o555 if path.is_dir() else 0o444)
            with mock.patch.object(os, "access", side_effect=lambda p, mode: not bool(mode & os.W_OK)):
                files = diagnose_native.inspect_model_files(models)
                diagnose_native.assert_model_files_unchanged(files)
            self.assertEqual(files.manifest, manifest)
            self.assertEqual(set(files.referenced), {blob, config_blob})

    def test_diagnosis_saves_only_filtered_show_and_chat_fields(self):
        show = {
            "model_info": {"general.architecture": "gptoss"},
            "template": "must not be saved",
            "license": "must not be saved",
        }
        chat = {"message": {"content": "must not be saved"}}
        with mock.patch.object(
            diagnose_native,
            "request_json",
            side_effect=[(200, show, None), (200, chat, None)],
        ) as request:
            result = diagnose_native.diagnose()
        self.assertEqual(
            result,
            {
                "show": {"status": 200, "error": None, "architecture": "gptoss"},
                "chat": {"status": 200, "error": None},
            },
        )
        chat_payload = request.call_args_list[1].args[1]
        self.assertEqual(chat_payload["options"]["num_gpu"], 0)
        self.assertEqual(chat_payload["options"]["num_thread"], 1)
        self.assertEqual(chat_payload["options"]["num_ctx"], 2048)
        self.assertEqual(chat_payload["options"]["num_predict"], 1)
        self.assertFalse(chat_payload["stream"])

    def test_failed_show_does_not_attempt_chat(self):
        with mock.patch.object(
            diagnose_native,
            "request_json",
            return_value=(500, None, "manifest failed"),
        ) as request:
            result = diagnose_native.diagnose()
        self.assertEqual(request.call_count, 1)
        self.assertEqual(result["show"]["error"], "manifest failed")
        self.assertNotIn("chat", result)


if __name__ == "__main__":
    unittest.main()
