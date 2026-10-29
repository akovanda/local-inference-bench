import hashlib
import io
import json
import os
import tarfile
import tempfile
import unittest
from pathlib import Path

from runtime import native_ollama


class NativeOllamaTests(unittest.TestCase):
    def test_overlay_resolution_honors_precedence_whiteouts_and_opaque_directories(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            high, low = root / "high", root / "low"
            for layer in (high, low):
                (layer / "usr/lib/ollama/cuda_v13").mkdir(parents=True)
            (low / "usr/lib/ollama/common.so").write_bytes(b"old")
            (high / "usr/lib/ollama/common.so").write_bytes(b"new")
            (low / "usr/lib/ollama/deleted.so").write_bytes(b"hidden")
            (high / "usr/lib/ollama/.wh.deleted.so").touch()
            (low / "usr/lib/ollama/cuda_v13/lower.so").write_bytes(b"hidden by opaque")
            (high / "usr/lib/ollama/cuda_v13/.wh..wh..opq").touch()
            (high / "usr/lib/ollama/cuda_v13/libggml-cuda.so").write_bytes(b"cuda")

            common = native_ollama.resolve_overlay([high, low], "/usr/lib/ollama/common.so")
            deleted = native_ollama.resolve_overlay([high, low], "/usr/lib/ollama/deleted.so")
            backend = native_ollama.resolve_overlay([high, low], "/usr/lib/ollama/cuda_v13")

            self.assertIsInstance(common, native_ollama.OverlayNode)
            self.assertEqual(common.source_path.read_bytes(), b"new")
            self.assertIsNone(deleted)
            self.assertIsInstance(backend, native_ollama.OverlayDirectory)
            self.assertEqual(
                [Path(child.logical_path).name for child in native_ollama.merged_children(backend)],
                ["libggml-cuda.so"],
            )

    def test_stream_extract_preserves_symlinks_and_compares_source_digests(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            high = root / "layer"
            (high / "usr/bin").mkdir(parents=True)
            (high / "usr/lib/ollama/cuda_v13").mkdir(parents=True)
            (high / "usr/bin/ollama").write_bytes(b"binary")
            os.chmod(high / "usr/bin/ollama", 0o755)
            (high / "usr/lib/ollama/libbase.so.1").write_bytes(b"base")
            (high / "usr/lib/ollama/libbase.so").symlink_to("libbase.so.1")
            (high / "usr/lib/ollama/cuda_v13/libggml-cuda.so").write_bytes(b"cuda")

            stream = io.BytesIO()
            native_ollama.emit_bundle([high], "auto", stream)
            stream.seek(0)
            output = root / "output"
            output.mkdir()
            manifest = native_ollama.extract_bundle(stream, output)

            self.assertEqual(manifest["backend"], "cuda_v13")
            self.assertEqual((output / "bin/ollama").read_bytes(), b"binary")
            self.assertTrue((output / "lib/ollama/libbase.so").is_symlink())
            files = {record["path"]: record for record in manifest["files"]}
            self.assertEqual(files["bin/ollama"]["sha256"], hashlib.sha256(b"binary").hexdigest())
            self.assertNotIn("lib/ollama/.wh.deleted.so", files)

    def test_descriptor_launch_is_loopback_and_does_not_override_home(self):
        destination = Path("/external/storage/example")
        descriptor = native_ollama.make_descriptor(
            destination=destination,
            image_reference="example",
            image_record={"Id": "sha256:abc", "RepoDigests": ["example@sha256:def"]},
            backend="cuda_v13",
            version="0.40.6",
            version_output="ollama version is 0.40.6",
            source_files=[],
            models=Path("/models"),
            host=native_ollama._validate_loopback_host("127.0.0.1:11440"),
        )
        env = descriptor["launch"]["env"]
        self.assertNotIn("HOME", env)
        self.assertNotIn("CODEX_HOME", env)
        self.assertEqual(env["OLLAMA_HOST"], "127.0.0.1:11440")
        self.assertEqual(descriptor["launch"]["argv"][-1], "serve")
        identity = descriptor["launch"]["home_identity"]
        self.assertEqual(identity["private_key"], str(Path.home() / ".ollama/id_ed25519"))
        self.assertIn("Ollama 0.40.6", identity["serve_behavior"])
        self.assertIn("inherited HOME", identity["serve_behavior"])
        with self.assertRaises(native_ollama.NativeOllamaError):
            native_ollama._validate_loopback_host("0.0.0.0:11440")


if __name__ == "__main__":
    unittest.main()
