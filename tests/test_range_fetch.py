import contextlib
import hashlib
import http.server
import io
import json
import os
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from runtime import range_fetch


class _RangeHandler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self):
        range_value = self.headers.get("Range", "")
        if not range_value.startswith("bytes=") or "-" not in range_value:
            self.send_response(200)
            self.send_header("Content-Length", str(len(self.server.data)))
            self.end_headers()
            self.wfile.write(self.server.data)
            return
        start_text, end_text = range_value[6:].split("-", 1)
        start, end = int(start_text), int(end_text)
        with self.server.state_lock:
            self.server.request_counts[start] = self.server.request_counts.get(start, 0) + 1
        payload = self.server.data[start : end + 1]
        self.send_response(206)
        if start in self.server.invalid_content_range_starts:
            advertised_start = start + 1
        else:
            advertised_start = start
        self.send_header(
            "Content-Range", f"bytes {advertised_start}-{end}/{len(self.server.data)}"
        )
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Content-Encoding", "identity")
        self.send_header("Connection", "close")
        self.end_headers()
        try:
            self.wfile.write(payload)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def log_message(self, format, *args):
        pass


class LocalRangeServer:
    def __init__(self, data):
        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _RangeHandler)
        self.server.daemon_threads = True
        self.server.data = data
        self.server.invalid_content_range_starts = set()
        self.server.request_counts = {}
        self.server.state_lock = threading.Lock()
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def url(self):
        return f"http://127.0.0.1:{self.server.server_port}/artifact.bin"

    @property
    def invalid_content_range_starts(self):
        return self.server.invalid_content_range_starts

    @property
    def request_counts(self):
        return self.server.request_counts

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)


def _digest(data):
    return hashlib.sha256(data).hexdigest()


class RangeFetchTests(unittest.TestCase):
    def run_download(self, server, output, data, **overrides):
        options = {
            "workers": 3,
            "chunk_size": 32 * 1024,
            "retries": 1,
            "request_timeout": 5,
            "progress_interval": 1000,
        }
        options.update(overrides)
        # Production fsync calls are deliberately frequent. Patching only the syscall keeps
        # these tiny offline protocol tests independent of host writeback pressure.
        with contextlib.redirect_stderr(io.StringIO()), mock.patch(
            "runtime.range_fetch.os.fsync"
        ):
            return range_fetch.download_file(
                server.url, output, len(data), _digest(data), **options
            )

    def test_valid_parallel_range_download_publishes_exact_file(self):
        data = bytes((index * 37 + 11) % 256 for index in range(160_123))
        with tempfile.TemporaryDirectory() as directory, LocalRangeServer(data) as server:
            output = Path(directory) / "model.gguf"
            result = self.run_download(server, output, data)
            self.assertEqual(result, output)
            self.assertEqual(output.read_bytes(), data)
            self.assertFalse(range_fetch.part_path_for(output).exists())
            self.assertFalse(range_fetch.manifest_path_for(output).exists())
            self.assertGreater(len(server.request_counts), 1)

    def test_invalid_content_range_is_refused_and_partial_is_preserved(self):
        chunk_size = 16 * 1024
        data = bytes((index * 19 + 3) % 256 for index in range(chunk_size * 2 + 91))
        with tempfile.TemporaryDirectory() as directory, LocalRangeServer(data) as server:
            output = Path(directory) / "model.gguf"
            server.invalid_content_range_starts.add(0)
            with self.assertRaises(range_fetch.RangeProtocolError):
                self.run_download(
                    server, output, data, workers=1, chunk_size=chunk_size
                )
            self.assertFalse(output.exists())
            self.assertTrue(range_fetch.part_path_for(output).exists())
            self.assertTrue(range_fetch.manifest_path_for(output).exists())

    def test_resume_revalidates_chunks_and_refetches_corruption(self):
        chunk_size = 12 * 1024
        data = bytes((index * 13 + 7) % 256 for index in range(chunk_size * 4 + 17))
        with tempfile.TemporaryDirectory() as directory, LocalRangeServer(data) as server:
            output = Path(directory) / "model.gguf"
            with mock.patch("runtime.range_fetch.os.fsync"):
                part_path, manifest_path, manifest, _ = range_fetch.load_or_create_state(
                    output, server.url, len(data), _digest(data), chunk_size
                )
                descriptor = os.open(part_path, os.O_RDWR)
                try:
                    for index in (0, 1):
                        start, end = range_fetch.chunk_bounds(index, len(data), chunk_size)
                        payload = data[start : end + 1]
                        range_fetch._record_chunk(
                            descriptor,
                            manifest_path,
                            manifest,
                            index,
                            start,
                            payload,
                            _digest(payload),
                        )
                finally:
                    os.close(descriptor)

            corrupt_index, retained_index = 0, 1
            incomplete_indices = [2, 3, 4]
            with part_path.open("r+b") as handle:
                handle.seek(corrupt_index * chunk_size + 9)
                handle.write(b"X")

            counts_before = dict(server.request_counts)
            server.invalid_content_range_starts.clear()
            self.run_download(server, output, data, workers=2, chunk_size=chunk_size)
            counts_after = server.request_counts

            self.assertEqual(output.read_bytes(), data)
            corrupt_start = corrupt_index * chunk_size
            retained_start = retained_index * chunk_size
            incomplete_start = incomplete_indices[0] * chunk_size
            self.assertGreater(
                counts_after.get(corrupt_start, 0), counts_before.get(corrupt_start, 0)
            )
            self.assertEqual(
                counts_after.get(retained_start, 0), counts_before.get(retained_start, 0)
            )
            self.assertGreater(
                counts_after.get(incomplete_start, 0), counts_before.get(incomplete_start, 0)
            )

    def test_seed_prefix_is_hash_verified_and_only_complete_chunks_are_counted(self):
        chunk_size = 10 * 1024
        data = bytes((index * 29 + 5) % 256 for index in range(chunk_size * 4 + 123))
        seed_data = data[: chunk_size * 2 + 777]
        with tempfile.TemporaryDirectory() as directory, LocalRangeServer(data) as server:
            root = Path(directory)
            output = root / "model.gguf"
            seed = root / "existing-download.part"
            seed.write_bytes(seed_data)
            seed_before = seed.read_bytes()

            server.invalid_content_range_starts.add(chunk_size * 2)
            with self.assertRaises(range_fetch.RangeProtocolError):
                self.run_download(
                    server,
                    output,
                    data,
                    workers=1,
                    chunk_size=chunk_size,
                    seed_prefix=seed,
                )

            manifest = json.loads(range_fetch.manifest_path_for(output).read_text())
            copied = chunk_size * 2
            self.assertEqual(manifest["seed"]["copied_bytes"], copied)
            self.assertEqual(manifest["seed"]["prefix_sha256"], _digest(seed_data[:copied]))
            self.assertTrue({"0", "1"}.issubset(manifest["chunks"]))
            self.assertEqual(
                range_fetch.part_path_for(output).read_bytes()[:copied], seed_data[:copied]
            )
            self.assertEqual(seed.read_bytes(), seed_before)

            server.invalid_content_range_starts.clear()
            counts_before = dict(server.request_counts)
            self.run_download(
                server,
                output,
                data,
                workers=2,
                chunk_size=chunk_size,
                seed_prefix=seed,
            )
            self.assertEqual(output.read_bytes(), data)
            self.assertEqual(server.request_counts.get(0, 0), counts_before.get(0, 0))
            self.assertEqual(
                server.request_counts.get(chunk_size, 0), counts_before.get(chunk_size, 0)
            )
            self.assertEqual(seed.read_bytes(), seed_before)


if __name__ == "__main__":
    unittest.main()
