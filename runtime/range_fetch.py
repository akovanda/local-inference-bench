#!/usr/bin/env python3
"""Resumable parallel HTTP Range downloader with end-to-end integrity checks."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import http.client
import json
import os
import re
import socket
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urljoin, urlsplit


MANIFEST_VERSION = 1
REQUEST_TIMEOUT_SECONDS = 60.0
MAX_REDIRECTS = 5
READ_BLOCK = 1024 * 1024
CONTENT_RANGE_RE = re.compile(r"^bytes (\d+)-(\d+)/(\d+)$", re.IGNORECASE)
TRANSIENT_HTTP_STATUSES = {408, 425, 429, 500, 502, 503, 504}


class DownloadError(RuntimeError):
    """Base class for safe, URL-free user-facing download errors."""


class RangeProtocolError(DownloadError):
    """The origin did not honor the exact requested byte range."""


class IntegrityError(DownloadError):
    """Local or final integrity validation failed."""


class TransientDownloadError(DownloadError):
    """A request may succeed if retried."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def part_path_for(output: Path) -> Path:
    return Path(str(output) + ".range-part")


def manifest_path_for(output: Path) -> Path:
    return Path(str(output) + ".range-part.manifest.json")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_all(descriptor: int, data: bytes) -> None:
    view = memoryview(data)
    written = 0
    while written < len(view):
        count = os.write(descriptor, view[written:])
        if count <= 0:
            raise OSError("short local write")
        written += count


def atomic_write_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    payload = (json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode("utf-8")
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}.{uuid.uuid4().hex}")
    descriptor = -1
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        _write_all(descriptor, payload)
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _new_manifest(url: str, expected_size: int, expected_sha256: str, chunk_size: int) -> dict[str, Any]:
    timestamp = utc_now()
    return {
        "schema_version": MANIFEST_VERSION,
        "url": url,
        "expected_size": expected_size,
        "expected_sha256": expected_sha256,
        "chunk_size": chunk_size,
        "created_at": timestamp,
        "updated_at": timestamp,
        "chunks": {},
    }


def _manifest_identity(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": manifest.get("schema_version"),
        "url": manifest.get("url"),
        "expected_size": manifest.get("expected_size"),
        "expected_sha256": manifest.get("expected_sha256"),
        "chunk_size": manifest.get("chunk_size"),
    }


def _expected_identity(url: str, expected_size: int, expected_sha256: str, chunk_size: int) -> dict[str, Any]:
    return {
        "schema_version": MANIFEST_VERSION,
        "url": url,
        "expected_size": expected_size,
        "expected_sha256": expected_sha256,
        "chunk_size": chunk_size,
    }


def load_or_create_state(
    output: Path,
    url: str,
    expected_size: int,
    expected_sha256: str,
    chunk_size: int,
) -> tuple[Path, Path, dict[str, Any], bool]:
    part_path = part_path_for(output)
    manifest_path = manifest_path_for(output)
    part_exists = os.path.lexists(part_path)
    manifest_exists = os.path.lexists(manifest_path)

    if manifest_exists:
        if manifest_path.is_symlink() or not manifest_path.is_file():
            raise DownloadError("refusing non-regular range manifest")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise DownloadError("range manifest is unreadable or invalid") from exc
        if not isinstance(manifest, dict) or _manifest_identity(manifest) != _expected_identity(
            url, expected_size, expected_sha256, chunk_size
        ):
            raise DownloadError("existing range manifest does not exactly match this request")
        if not isinstance(manifest.get("chunks"), dict):
            raise DownloadError("range manifest has an invalid chunks table")
        if not part_exists:
            if manifest["chunks"]:
                raise DownloadError("range manifest records chunks but its partial file is missing")
            descriptor = os.open(part_path, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o644)
            os.close(descriptor)
        elif part_path.is_symlink() or not part_path.is_file():
            raise DownloadError("refusing non-regular range partial")
        if part_path.stat().st_size > expected_size:
            raise DownloadError("range partial is larger than the expected artifact")
        return part_path, manifest_path, manifest, False

    if part_exists:
        raise DownloadError("range partial exists without its matching manifest; refusing to overwrite it")

    manifest = _new_manifest(url, expected_size, expected_sha256, chunk_size)
    atomic_write_manifest(manifest_path, manifest)
    try:
        descriptor = os.open(part_path, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o644)
        os.close(descriptor)
        _fsync_directory(part_path.parent)
    except Exception:
        # The persisted empty manifest makes this initialization safely resumable.
        raise
    return part_path, manifest_path, manifest, True


def chunk_bounds(index: int, expected_size: int, chunk_size: int) -> tuple[int, int]:
    start = index * chunk_size
    end = min(expected_size, start + chunk_size) - 1
    return start, end


def chunk_count(expected_size: int, chunk_size: int) -> int:
    return (expected_size + chunk_size - 1) // chunk_size


def _sha256_range(descriptor: int, start: int, length: int) -> str | None:
    digest = hashlib.sha256()
    offset = 0
    while offset < length:
        block = os.pread(descriptor, min(READ_BLOCK, length - offset), start + offset)
        if not block:
            return None
        digest.update(block)
        offset += len(block)
    return digest.hexdigest()


def validate_completed_chunks(
    descriptor: int,
    manifest: dict[str, Any],
    expected_size: int,
    chunk_size: int,
) -> tuple[set[int], bool]:
    valid: set[int] = set()
    changed = False
    total_chunks = chunk_count(expected_size, chunk_size)
    chunks = manifest["chunks"]
    for key, entry in list(chunks.items()):
        try:
            index = int(key)
        except (TypeError, ValueError):
            del chunks[key]
            changed = True
            continue
        if str(index) != key or index < 0 or index >= total_chunks or not isinstance(entry, dict):
            del chunks[key]
            changed = True
            continue
        start, end = chunk_bounds(index, expected_size, chunk_size)
        length = end - start + 1
        digest = entry.get("sha256")
        structurally_valid = (
            entry.get("start") == start
            and entry.get("end") == end
            and entry.get("bytes") == length
            and isinstance(digest, str)
            and re.fullmatch(r"[0-9a-f]{64}", digest) is not None
        )
        local_digest = _sha256_range(descriptor, start, length) if structurally_valid else None
        if not structurally_valid or local_digest != digest:
            del chunks[key]
            changed = True
            continue
        valid.add(index)
    return valid, changed


def _pwrite_all(descriptor: int, data: bytes, offset: int) -> None:
    view = memoryview(data)
    written = 0
    while written < len(view):
        count = os.pwrite(descriptor, view[written:], offset + written)
        if count <= 0:
            raise OSError("short positional write")
        written += count


def _record_chunk(
    descriptor: int,
    manifest_path: Path,
    manifest: dict[str, Any],
    index: int,
    start: int,
    data: bytes,
    digest: str,
) -> None:
    _pwrite_all(descriptor, data, start)
    os.fsync(descriptor)
    manifest["chunks"][str(index)] = {
        "start": start,
        "end": start + len(data) - 1,
        "bytes": len(data),
        "sha256": digest,
    }
    manifest["updated_at"] = utc_now()
    atomic_write_manifest(manifest_path, manifest)


def seed_from_prefix(
    seed_path: Path,
    part_path: Path,
    descriptor: int,
    manifest_path: Path,
    manifest: dict[str, Any],
    expected_size: int,
    chunk_size: int,
) -> set[int]:
    if seed_path.resolve() in {part_path.resolve(), manifest_path.resolve()}:
        raise DownloadError("seed prefix must be a different file")
    if seed_path.is_symlink() or not seed_path.is_file():
        raise DownloadError("seed prefix must be a regular file")
    before = seed_path.stat()
    available = min(before.st_size, expected_size)
    complete_indices = [
        index
        for index in range(chunk_count(expected_size, chunk_size))
        if chunk_bounds(index, expected_size, chunk_size)[1] < available
    ]
    copied_bytes = 0
    source_digest = hashlib.sha256()
    entries: dict[str, dict[str, Any]] = {}

    source_descriptor = os.open(seed_path, os.O_RDONLY)
    try:
        for index in complete_indices:
            start, end = chunk_bounds(index, expected_size, chunk_size)
            length = end - start + 1
            data = os.pread(source_descriptor, length, start)
            if len(data) != length:
                raise IntegrityError("seed prefix changed or ended during copy")
            digest = hashlib.sha256(data).hexdigest()
            source_digest.update(data)
            _pwrite_all(descriptor, data, start)
            entries[str(index)] = {
                "start": start,
                "end": end,
                "bytes": length,
                "sha256": digest,
            }
            copied_bytes += length
    finally:
        os.close(source_descriptor)
    os.fsync(descriptor)

    after = seed_path.stat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise IntegrityError("seed prefix changed during copy")

    destination_digest = _sha256_range(descriptor, 0, copied_bytes)
    source_hexdigest = source_digest.hexdigest()
    if destination_digest != source_hexdigest:
        raise IntegrityError("copied seed prefix failed destination verification")

    manifest["chunks"].update(entries)
    manifest["seed"] = {
        "path": str(seed_path.resolve()),
        "source_size": before.st_size,
        "copied_bytes": copied_bytes,
        "prefix_sha256": source_hexdigest,
        "verified_at": utc_now(),
    }
    manifest["updated_at"] = utc_now()
    atomic_write_manifest(manifest_path, manifest)
    return set(complete_indices)


def _remaining(deadline: float) -> float:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TransientDownloadError("request reached its 60 second deadline")
    return remaining


def _connection_for_url(url: str, timeout: float) -> tuple[http.client.HTTPConnection, str]:
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise RangeProtocolError("URL must use http or https and include a host")
    if parsed.username is not None or parsed.password is not None:
        raise RangeProtocolError("credentials in URLs are not supported")
    if parsed.scheme == "https":
        connection: http.client.HTTPConnection = http.client.HTTPSConnection(
            parsed.hostname, parsed.port, timeout=timeout
        )
    else:
        connection = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=timeout)
    path = parsed.path or "/"
    if parsed.query:
        path += "?" + parsed.query
    return connection, path


def fetch_range_once(url: str, start: int, end: int, total: int, timeout: float) -> tuple[bytes, str]:
    deadline = time.monotonic() + min(timeout, REQUEST_TIMEOUT_SECONDS)
    current_url = url
    expected_length = end - start + 1

    for redirect_count in range(MAX_REDIRECTS + 1):
        connection: http.client.HTTPConnection | None = None
        try:
            connection, path = _connection_for_url(current_url, _remaining(deadline))
            headers = {
                "Range": f"bytes={start}-{end}",
                "Accept-Encoding": "identity",
                "User-Agent": "range-fetch/1",
            }
            connection.request("GET", path, headers=headers)
            if connection.sock is not None:
                connection.sock.settimeout(_remaining(deadline))
            response = connection.getresponse()

            if response.status in {301, 302, 303, 307, 308}:
                location = response.getheader("Location")
                if not location:
                    raise RangeProtocolError("redirect response omitted Location")
                if redirect_count >= MAX_REDIRECTS:
                    raise RangeProtocolError("too many redirects")
                current_url = urljoin(current_url, location)
                continue

            if response.status == 200:
                raise RangeProtocolError("server ignored Range and returned HTTP 200")
            if response.status != 206:
                if response.status in TRANSIENT_HTTP_STATUSES:
                    raise TransientDownloadError(f"transient HTTP status {response.status}")
                raise RangeProtocolError(f"unexpected HTTP status {response.status}")

            content_range = response.getheader("Content-Range", "")
            match = CONTENT_RANGE_RE.fullmatch(content_range.strip())
            if not match or tuple(map(int, match.groups())) != (start, end, total):
                raise RangeProtocolError("response Content-Range did not exactly match the request")
            content_encoding = response.getheader("Content-Encoding")
            if content_encoding and content_encoding.lower() != "identity":
                raise RangeProtocolError("ranged response used an unexpected content encoding")
            content_length = response.getheader("Content-Length")
            if content_length is not None:
                try:
                    if int(content_length) != expected_length:
                        raise RangeProtocolError("response Content-Length did not match the requested range")
                except ValueError as exc:
                    raise RangeProtocolError("response Content-Length was invalid") from exc

            blocks: list[bytes] = []
            received = 0
            while received <= expected_length:
                if connection.sock is not None:
                    connection.sock.settimeout(_remaining(deadline))
                block = response.read(min(READ_BLOCK, expected_length + 1 - received))
                if not block:
                    break
                blocks.append(block)
                received += len(block)
            data = b"".join(blocks)
            if len(data) != expected_length:
                raise TransientDownloadError("ranged response byte count was incomplete")
            return data, hashlib.sha256(data).hexdigest()
        except RangeProtocolError:
            raise
        except TransientDownloadError:
            raise
        except (OSError, socket.timeout, TimeoutError, http.client.HTTPException) as exc:
            raise TransientDownloadError(f"request failed with {type(exc).__name__}") from exc
        finally:
            if connection is not None:
                connection.close()
    raise RangeProtocolError("too many redirects")


def fetch_range_with_retries(
    url: str,
    start: int,
    end: int,
    total: int,
    retries: int,
    timeout: float,
) -> tuple[bytes, str]:
    for attempt in range(retries):
        try:
            return fetch_range_once(url, start, end, total, timeout)
        except TransientDownloadError:
            if attempt + 1 >= retries:
                raise
            time.sleep(min(8.0, 0.5 * (2**attempt)))
    raise TransientDownloadError("request exhausted its retry budget")


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(4 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _format_bytes(value: int) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    amount = float(value)
    unit = units[0]
    for unit in units:
        if amount < 1024.0 or unit == units[-1]:
            break
        amount /= 1024.0
    return f"{amount:.1f} {unit}"


def _print_progress(complete: int, total: int, new_bytes: int, elapsed: float) -> None:
    rate = new_bytes / elapsed if elapsed > 0 else 0.0
    percent = (complete * 100.0 / total) if total else 100.0
    print(
        f"progress {_format_bytes(complete)}/{_format_bytes(total)} ({percent:.1f}%) "
        f"rate={_format_bytes(int(rate))}/s elapsed={elapsed:.1f}s",
        file=sys.stderr,
        flush=True,
    )


def _publish_no_overwrite(part_path: Path, output: Path, manifest_path: Path) -> None:
    if os.path.lexists(output):
        raise DownloadError("output appeared during download; refusing to overwrite it")
    try:
        os.link(part_path, output)
    except FileExistsError as exc:
        raise DownloadError("output appeared during download; refusing to overwrite it") from exc
    _fsync_directory(output.parent)
    # Publication has succeeded. Removing downloader-owned state does not affect the final inode.
    try:
        part_path.unlink()
        manifest_path.unlink()
        _fsync_directory(output.parent)
    except OSError:
        # A valid, fully hashed final file is already atomically visible; stale state is harmless.
        pass


def download_file(
    url: str,
    output: Path | str,
    expected_size: int,
    expected_sha256: str,
    *,
    workers: int = 4,
    chunk_size: int = 32 * 1024 * 1024,
    seed_prefix: Path | str | None = None,
    retries: int = 4,
    request_timeout: float = REQUEST_TIMEOUT_SECONDS,
    progress_interval: float = 5.0,
) -> Path:
    output = Path(output)
    expected_sha256 = expected_sha256.lower()
    if expected_size <= 0:
        raise DownloadError("expected size must be positive")
    if re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None:
        raise DownloadError("expected SHA-256 must be exactly 64 hexadecimal characters")
    if workers < 1 or workers > 64:
        raise DownloadError("workers must be between 1 and 64")
    if chunk_size < 1:
        raise DownloadError("chunk size must be positive")
    if retries < 1 or retries > 10:
        raise DownloadError("retries must be between 1 and 10")
    if request_timeout <= 0 or request_timeout > REQUEST_TIMEOUT_SECONDS:
        raise DownloadError("per-request timeout must be greater than 0 and at most 60 seconds")
    _connection_for_url(url, request_timeout)[0].close()

    output.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(output):
        if not output.is_file() or output.stat().st_size != expected_size:
            raise DownloadError("existing output does not have the exact expected size")
        if hash_file(output) != expected_sha256:
            raise IntegrityError("existing output does not match the expected SHA-256")
        print("existing output matches expected size and SHA-256", file=sys.stderr, flush=True)
        return output

    part_path, manifest_path, manifest, created = load_or_create_state(
        output, url, expected_size, expected_sha256, chunk_size
    )
    descriptor = os.open(part_path, os.O_RDWR)
    try:
        complete, changed = validate_completed_chunks(descriptor, manifest, expected_size, chunk_size)
        if changed:
            manifest["updated_at"] = utc_now()
            atomic_write_manifest(manifest_path, manifest)

        if seed_prefix is not None:
            seed_path = Path(seed_prefix)
            recorded_seed = manifest.get("seed")
            if recorded_seed is not None:
                if recorded_seed.get("path") != str(seed_path.resolve()):
                    raise DownloadError("existing manifest was initialized from a different seed prefix")
            elif complete:
                raise DownloadError("seed prefix can only initialize a partial before chunks are completed")
            else:
                complete.update(
                    seed_from_prefix(
                        seed_path,
                        part_path,
                        descriptor,
                        manifest_path,
                        manifest,
                        expected_size,
                        chunk_size,
                    )
                )
        elif created and manifest.get("seed") is not None:
            raise DownloadError("invalid new manifest seed state")

        total_chunks = chunk_count(expected_size, chunk_size)
        missing = [index for index in range(total_chunks) if index not in complete]
        completed_bytes = sum(
            chunk_bounds(index, expected_size, chunk_size)[1]
            - chunk_bounds(index, expected_size, chunk_size)[0]
            + 1
            for index in complete
        )
        new_bytes = 0
        started = time.monotonic()
        last_progress = started

        executor = concurrent.futures.ThreadPoolExecutor(max_workers=workers, thread_name_prefix="range-fetch")
        futures: dict[concurrent.futures.Future[tuple[bytes, str]], tuple[int, int, int]] = {}
        try:
            missing_iterator = iter(missing)

            def submit_next() -> concurrent.futures.Future[tuple[bytes, str]] | None:
                try:
                    index = next(missing_iterator)
                except StopIteration:
                    return None
                start, end = chunk_bounds(index, expected_size, chunk_size)
                future = executor.submit(
                    fetch_range_with_retries,
                    url,
                    start,
                    end,
                    expected_size,
                    retries,
                    request_timeout,
                )
                futures[future] = (index, start, end)
                return future

            # Bound completed byte buffers even when the network outruns durable writes.
            for _ in range(min(len(missing), workers * 2)):
                submit_next()

            pending = set(futures)
            while pending:
                done, pending = concurrent.futures.wait(
                    pending, timeout=1.0, return_when=concurrent.futures.FIRST_COMPLETED
                )
                for future in done:
                    index, start, end = futures.pop(future)
                    data, digest = future.result()
                    if len(data) != end - start + 1:
                        raise IntegrityError("worker returned a chunk with an invalid byte count")
                    _record_chunk(descriptor, manifest_path, manifest, index, start, data, digest)
                    complete.add(index)
                    completed_bytes += len(data)
                    new_bytes += len(data)
                while len(pending) < workers * 2:
                    submitted = submit_next()
                    if submitted is None:
                        break
                    pending.add(submitted)
                now = time.monotonic()
                if now - last_progress >= progress_interval:
                    _print_progress(completed_bytes, expected_size, new_bytes, now - started)
                    last_progress = now
        except BaseException:
            for future in list(futures):
                future.cancel()
            raise
        finally:
            executor.shutdown(wait=True, cancel_futures=True)

        elapsed = time.monotonic() - started
        _print_progress(completed_bytes, expected_size, new_bytes, elapsed)
        if len(complete) != total_chunks or completed_bytes != expected_size:
            raise IntegrityError("not all chunks were durably completed")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)

    if part_path.stat().st_size != expected_size:
        raise IntegrityError("completed partial does not have the exact expected size")
    actual_sha256 = hash_file(part_path)
    if actual_sha256 != expected_sha256:
        # Per-chunk hashes cannot identify a same-length corruption delivered by the origin.
        # Clear completion claims so a subsequent invocation refetches rather than looping.
        manifest["chunks"] = {}
        manifest["last_integrity_failure_at"] = utc_now()
        manifest["updated_at"] = utc_now()
        atomic_write_manifest(manifest_path, manifest)
        raise IntegrityError("completed partial does not match the expected SHA-256")

    _publish_no_overwrite(part_path, output, manifest_path)
    print("download complete; size and SHA-256 verified", file=sys.stderr, flush=True)
    return output


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def sha256_value(value: str) -> str:
    normalized = value.lower()
    if re.fullmatch(r"[0-9a-f]{64}", normalized) is None:
        raise argparse.ArgumentTypeError("must be exactly 64 hexadecimal characters")
    return normalized


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True, help="public artifact URL (never printed)")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--size", required=True, type=positive_int, help="exact expected bytes")
    parser.add_argument("--sha256", required=True, type=sha256_value, help="exact expected SHA-256")
    parser.add_argument("--workers", type=positive_int, default=4)
    parser.add_argument("--chunk-mib", type=positive_int, default=32)
    parser.add_argument("--seed-prefix", type=Path)
    parser.add_argument("--retries", type=positive_int, default=4, help="attempts per chunk, maximum 10")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        download_file(
            args.url,
            args.output,
            args.size,
            args.sha256,
            workers=args.workers,
            chunk_size=args.chunk_mib * 1024 * 1024,
            seed_prefix=args.seed_prefix,
            retries=args.retries,
        )
    except KeyboardInterrupt:
        print("interrupted; validated partial chunks and manifest were preserved", file=sys.stderr)
        return 130
    except DownloadError as exc:
        print(f"range fetch failed: {exc}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"range fetch failed with local {type(exc).__name__}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
