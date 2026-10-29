import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from runtime import download_supervisor


class DownloadSupervisorTests(unittest.TestCase):
    def test_only_explicit_range_fetch_transients_are_retried(self):
        transient = (
            "range fetch failed: request reached its 60 second deadline",
            "range fetch failed: transient HTTP status 503",
            "range fetch failed: request failed with TimeoutError",
            "range fetch failed: ranged response byte count was incomplete",
        )
        for message in transient:
            with self.subTest(message=message):
                self.assertEqual(download_supervisor.classify_failure(message), "transient")
        permanent = (
            "range fetch failed: completed partial does not match the expected SHA-256",
            "range fetch failed: response Content-Range did not exactly match the request",
            "range fetch failed with local OSError",
            "range fetch failed: existing range manifest does not exactly match this request",
        )
        for message in permanent:
            with self.subTest(message=message):
                self.assertEqual(download_supervisor.classify_failure(message), "permanent")

    def test_commands_keep_pinned_identity_and_worker_allocation(self):
        commands = [download_supervisor.download_command(job) for job in download_supervisor.JOBS]
        self.assertEqual([command[command.index("--workers") + 1] for command in commands], ["8", "4"])
        for job, command in zip(download_supervisor.JOBS, commands):
            self.assertEqual(command[command.index("--url") + 1], job.file.url)
            self.assertEqual(command[command.index("--output") + 1], str(job.file.path))
            self.assertEqual(command[command.index("--size") + 1], str(job.file.size))
            self.assertEqual(command[command.index("--sha256") + 1], job.file.sha256)

    def test_progress_rejects_a_manifest_for_another_request(self):
        original = download_supervisor.JOBS[0]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / original.file.name
            file_spec = type(original.file)(original.file.name, original.file.size, original.file.sha256)
            job = type(original)(original.label, file_spec, original.workers)
            manifest_path = Path(str(output) + ".range-part.manifest.json")
            manifest_path.write_text(
                json.dumps(
                    {
                        "url": file_spec.url,
                        "expected_size": file_spec.size + 1,
                        "expected_sha256": file_spec.sha256,
                        "chunk_size": 32 * 1024 * 1024,
                        "chunks": {},
                    }
                ),
                encoding="utf-8",
            )
            with mock.patch.object(download_supervisor.prepare_flash_next, "MODEL_DIR", Path(directory)):
                progress = download_supervisor.manifest_progress(job)
            self.assertIn("progress_error", progress)


if __name__ == "__main__":
    unittest.main()
