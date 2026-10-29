#!/usr/bin/env python3
"""Selectively extract a Linux AArch64 QNN runtime from Qualcomm's public SDK.

The Qualcomm AI Runtime community ZIP is larger than 1 GiB.  This helper reads
its ZIP central directory with exact HTTP Range requests and downloads only
explicitly named members.  It intentionally does not accept credentials or an
SDK license agreement and never installs files outside the requested output
directory.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import struct
import sys
import urllib.error
import urllib.request
import zlib


DEFAULT_VERSION = "2.37.0.250724"
URL_TEMPLATE = (
    "https://softwarecenter.qualcomm.com/api/download/software/sdks/"
    "Qualcomm_AI_Runtime_Community/All/{version}/v{version}.zip"
)
EOCD_SIGNATURE = b"PK\x05\x06"
CENTRAL_SIGNATURE = b"PK\x01\x02"
LOCAL_SIGNATURE = b"PK\x03\x04"
CONTENT_RANGE_RE = re.compile(r"^bytes (\d+)-(\d+)/(\d+)$", re.IGNORECASE)


class ArchiveError(RuntimeError):
    """The remote archive or requested extraction is unsafe or malformed."""


@dataclass(frozen=True)
class Member:
    name: str
    method: int
    flags: int
    crc32: int
    compressed_size: int
    size: int
    local_offset: int
    external_attributes: int


class RangeZip:
    def __init__(self, url: str, timeout: float = 60.0) -> None:
        self.url = url
        self.timeout = timeout
        first, total = self._range(0, 0)
        if len(first) != 1 or total <= 0:
            raise ArchiveError("archive did not return a valid first byte")
        self.size = total
        self.members = self._read_directory()

    def _range(self, start: int, end: int) -> tuple[bytes, int]:
        if start < 0 or end < start:
            raise ArchiveError(f"invalid byte range {start}-{end}")
        request = urllib.request.Request(
            self.url,
            headers={
                "Range": f"bytes={start}-{end}",
                "User-Agent": "tachyon-qnn-setup/1",
                "Accept-Encoding": "identity",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                data = response.read()
                status = response.status
                content_range = response.headers.get("Content-Range", "")
        except urllib.error.URLError as exc:
            raise ArchiveError(f"range request failed: {exc.reason}") from exc
        match = CONTENT_RANGE_RE.fullmatch(content_range.strip())
        if status != 206 or match is None:
            raise ArchiveError("origin did not honor an exact HTTP byte range")
        actual_start, actual_end, total = map(int, match.groups())
        if (actual_start, actual_end) != (start, end):
            raise ArchiveError(
                f"origin returned bytes {actual_start}-{actual_end}, expected {start}-{end}"
            )
        if len(data) != end - start + 1:
            raise ArchiveError("range response body length does not match Content-Range")
        return data, total

    def _read_directory(self) -> dict[str, Member]:
        tail_size = min(self.size, 65_557)
        tail_start = self.size - tail_size
        tail, _ = self._range(tail_start, self.size - 1)
        marker = tail.rfind(EOCD_SIGNATURE)
        if marker < 0 or len(tail) - marker < 22:
            raise ArchiveError("ZIP end-of-central-directory record was not found")
        (
            signature,
            disk_number,
            directory_disk,
            disk_entries,
            total_entries,
            directory_size,
            directory_offset,
            comment_length,
        ) = struct.unpack_from("<4s4H2LH", tail, marker)
        if signature != EOCD_SIGNATURE:
            raise ArchiveError("invalid ZIP end-of-central-directory signature")
        if marker + 22 + comment_length != len(tail):
            raise ArchiveError("ZIP end record is malformed or trailing data is present")
        if disk_number or directory_disk or disk_entries != total_entries:
            raise ArchiveError("multi-disk ZIP archives are unsupported")
        if 0xFFFF in (disk_entries, total_entries) or 0xFFFFFFFF in (
            directory_size,
            directory_offset,
        ):
            raise ArchiveError("ZIP64 archives are unsupported")
        directory_end = directory_offset + directory_size
        if directory_end > self.size or directory_size <= 0:
            raise ArchiveError("ZIP central directory bounds are invalid")
        raw, _ = self._range(directory_offset, directory_end - 1)
        members: dict[str, Member] = {}
        offset = 0
        for _ in range(total_entries):
            if offset + 46 > len(raw):
                raise ArchiveError("ZIP central directory is truncated")
            fields = struct.unpack_from("<4s6H3L5H2L", raw, offset)
            if fields[0] != CENTRAL_SIGNATURE:
                raise ArchiveError("invalid ZIP central-directory member signature")
            (
                _,
                _version_made,
                _version_needed,
                flags,
                method,
                _mtime,
                _mdate,
                crc32,
                compressed_size,
                size,
                name_length,
                extra_length,
                member_comment_length,
                _member_disk,
                _internal_attributes,
                external_attributes,
                local_offset,
            ) = fields
            record_end = offset + 46 + name_length + extra_length + member_comment_length
            if record_end > len(raw):
                raise ArchiveError("ZIP central-directory member is truncated")
            name_bytes = raw[offset + 46 : offset + 46 + name_length]
            encoding = "utf-8" if flags & 0x800 else "cp437"
            try:
                name = name_bytes.decode(encoding)
            except UnicodeDecodeError as exc:
                raise ArchiveError("ZIP member name cannot be decoded") from exc
            if name in members:
                raise ArchiveError(f"duplicate ZIP member: {name}")
            members[name] = Member(
                name=name,
                method=method,
                flags=flags,
                crc32=crc32,
                compressed_size=compressed_size,
                size=size,
                local_offset=local_offset,
                external_attributes=external_attributes,
            )
            offset = record_end
        if offset != len(raw):
            raise ArchiveError("ZIP central-directory size does not match its entries")
        return members

    def extract(self, name: str, destination: Path) -> dict[str, object]:
        try:
            member = self.members[name]
        except KeyError as exc:
            raise ArchiveError(f"member is absent from archive: {name}") from exc
        member_path = PurePosixPath(name)
        if member_path.is_absolute() or ".." in member_path.parts or name.endswith("/"):
            raise ArchiveError(f"refusing unsafe or non-file member: {name}")
        unix_mode = member.external_attributes >> 16
        if unix_mode and (unix_mode & 0o170000) not in (0, 0o100000):
            raise ArchiveError(f"refusing non-regular member: {name}")
        local_header, _ = self._range(member.local_offset, member.local_offset + 29)
        fields = struct.unpack("<4s5H3L2H", local_header)
        if fields[0] != LOCAL_SIGNATURE:
            raise ArchiveError(f"invalid local header for {name}")
        flags, method, name_length, extra_length = fields[2], fields[3], fields[9], fields[10]
        if flags & 0x1:
            raise ArchiveError(f"encrypted ZIP member is unsupported: {name}")
        if method != member.method:
            raise ArchiveError(f"compression method mismatch for {name}")
        data_offset = member.local_offset + 30 + name_length + extra_length
        if member.compressed_size:
            compressed, _ = self._range(
                data_offset, data_offset + member.compressed_size - 1
            )
        else:
            compressed = b""
        if member.method == 0:
            payload = compressed
        elif member.method == 8:
            try:
                payload = zlib.decompress(compressed, -zlib.MAX_WBITS)
            except zlib.error as exc:
                raise ArchiveError(f"deflate decompression failed for {name}") from exc
        else:
            raise ArchiveError(f"unsupported compression method {member.method} for {name}")
        if len(payload) != member.size:
            raise ArchiveError(f"uncompressed size mismatch for {name}")
        actual_crc = zlib.crc32(payload) & 0xFFFFFFFF
        if actual_crc != member.crc32:
            raise ArchiveError(f"CRC32 mismatch for {name}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + f".tmp.{os.getpid()}")
        if temporary.exists():
            raise ArchiveError(f"temporary output already exists: {temporary}")
        try:
            with temporary.open("xb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.chmod(temporary, unix_mode & 0o777 if unix_mode else 0o644)
            os.replace(temporary, destination)
        finally:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
        return {
            "member": name,
            "output": str(destination),
            "size": member.size,
            "compressed_size": member.compressed_size,
            "crc32": f"{member.crc32:08x}",
            "sha256": hashlib.sha256(payload).hexdigest(),
        }


def parse_args(arguments: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", default=DEFAULT_VERSION)
    parser.add_argument("--url", help="override the version-derived public SDK URL")
    parser.add_argument("--timeout", type=float, default=60.0)
    subparsers = parser.add_subparsers(dest="command", required=True)

    listing = subparsers.add_parser("list", help="list matching central-directory members")
    listing.add_argument("patterns", nargs="*", help="case-insensitive substrings")

    extracting = subparsers.add_parser("extract", help="extract exact named members")
    extracting.add_argument("--output", type=Path, required=True)
    extracting.add_argument("--manifest", type=Path)
    extracting.add_argument("members", nargs="+")
    return parser.parse_args(arguments)


def main(arguments: list[str] = sys.argv[1:]) -> int:
    args = parse_args(arguments)
    url = args.url or URL_TEMPLATE.format(version=args.version)
    archive = RangeZip(url, timeout=args.timeout)
    if args.command == "list":
        patterns = [pattern.casefold() for pattern in args.patterns]
        for member in archive.members.values():
            if all(pattern in member.name.casefold() for pattern in patterns):
                print(f"{member.size}\t{member.compressed_size}\t{member.name}")
        return 0

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for name in args.members:
        relative = PurePosixPath(name)
        records.append(archive.extract(name, output.joinpath(*relative.parts)))
    manifest = {
        "schema_version": 1,
        "source_url": url,
        "source_version": args.version,
        "archive_size": archive.size,
        "files": records,
    }
    manifest_path = (args.manifest or output / "SELECTIVE-EXTRACT.json").resolve()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ArchiveError as exc:
        raise SystemExit(f"error: {exc}") from exc
