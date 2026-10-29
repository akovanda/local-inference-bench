#!/usr/bin/env python3
"""Verify and resolve files in a relocated public evidence checkout.

The public archive is intentionally self-contained. Paths found in evidence
files are resolved only below the explicitly supplied archive root; the process
working directory and historical absolute paths are never used.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath
import stat
from typing import Any, Iterable, Mapping, Sequence


class EvidenceError(RuntimeError):
    """The published evidence manifest or one of its files is invalid."""


SELECTION_IDENTITY_FIELDS = (
    "cases", "population_sizes", "scope", "selected_count_per_suite",
    "selection_method", "selection_seed",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _published_path(value: Any) -> PurePosixPath:
    if not isinstance(value, str) or not value:
        raise EvidenceError("evidence manifest path must be a non-empty string")
    if "\\" in value:
        raise EvidenceError(f"evidence manifest path is not POSIX-relative: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise EvidenceError(f"unsafe evidence manifest path: {value!r}")
    return path


def _hex_sha256(value: Any, *, field: str) -> str:
    if (not isinstance(value, str) or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)):
        raise EvidenceError(f"{field} must be a lowercase SHA-256 hex digest")
    return value


def _regular_file_without_symlinks(root: Path, relative: PurePosixPath) -> Path:
    candidate = root.joinpath(*relative.parts)
    current = root
    for part in relative.parts:
        current = current / part
        try:
            metadata = current.lstat()
        except FileNotFoundError as exc:
            raise EvidenceError(f"published evidence file is missing: {relative}") from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise EvidenceError(f"published evidence path contains a symlink: {relative}")
    try:
        candidate.resolve(strict=True).relative_to(root)
    except (OSError, ValueError) as exc:
        raise EvidenceError(f"published evidence escapes its root: {relative}") from exc
    if not stat.S_ISREG(candidate.stat().st_mode):
        raise EvidenceError(f"published evidence is not a regular file: {relative}")
    return candidate


@dataclass(frozen=True)
class EvidenceEntry:
    path: str
    sha256: str
    size: int
    original_sha256: str | None = None
    original_size: int | None = None


class VerifiedEvidence:
    """A completely verified public evidence manifest and its archive root."""

    def __init__(self, root: Path, manifest_path: Path, manifest: Mapping[str, Any],
                 entries: Mapping[str, EvidenceEntry], manifest_sha256: str) -> None:
        self.root = root
        self.manifest_path = manifest_path
        self.manifest = manifest
        self.entries = dict(entries)
        self.manifest_sha256 = manifest_sha256
        selection = manifest.get("selection_identity_sha256")
        self.selection_identity_sha256 = (
            _hex_sha256(selection, field="selection_identity_sha256")
            if selection is not None else None
        )

    def relative(self, path: Path) -> str:
        try:
            return path.resolve(strict=False).relative_to(self.root).as_posix()
        except (OSError, ValueError) as exc:
            raise EvidenceError(f"path is outside the evidence root: {path}") from exc

    def require(self, value: str | Path) -> Path:
        """Return a verified manifest member, failing if it is not published."""
        path = Path(value)
        if path.is_absolute():
            try:
                relative = path.resolve(strict=False).relative_to(self.root).as_posix()
            except (OSError, ValueError) as exc:
                raise EvidenceError(f"required evidence is outside the archive: {path}") from exc
        else:
            relative = _published_path(path.as_posix()).as_posix()
        entry = self.entries.get(relative)
        if entry is None:
            raise EvidenceError(f"required evidence is not in evidence-manifest.json: {relative}")
        return self.root / relative

    def ensure_published_file(self, path: Path) -> Path:
        """Reject reads of files that are outside or absent from the manifest."""
        return self.require(path)

    def resolve(self, value: Any, *, bases: Sequence[Path] = (),
                required: bool = False) -> Path | None:
        """Resolve a nested evidence path without cwd or historical fallbacks.

        ``/external/...`` is publication metadata, not a readable path. It
        resolves to ``None`` when optional and fails when required.
        """
        if not isinstance(value, str) or not value:
            if required:
                raise EvidenceError("required evidence reference is empty")
            return None
        if value == "/external" or value.startswith("/external/"):
            if required:
                raise EvidenceError(f"required evidence was recorded as external metadata: {value}")
            return None

        raw = Path(value)
        candidates: list[Path] = []
        if raw.is_absolute():
            try:
                raw.resolve(strict=False).relative_to(self.root)
            except (OSError, ValueError):
                if required:
                    raise EvidenceError(f"absolute evidence reference is outside the archive: {value}")
                return None
            candidates.append(raw)
        elif raw.parts and raw.parts[0] in {"results", "runtime", "deployment", "tests"}:
            candidates.append(self.root / raw)
        else:
            for base in bases:
                base_path = Path(base)
                if not base_path.is_absolute():
                    base_path = self.root / base_path
                try:
                    base_path.resolve(strict=False).relative_to(self.root)
                except (OSError, ValueError) as exc:
                    raise EvidenceError(f"reference base is outside the archive: {base}") from exc
                candidates.append(base_path / raw)
            candidates.append(self.root / raw)

        contained: list[Path] = []
        for candidate in candidates:
            try:
                relative = candidate.resolve(strict=False).relative_to(self.root)
            except (OSError, ValueError):
                continue
            normalized = self.root / relative
            contained.append(normalized)
            if normalized.exists():
                if normalized.is_file():
                    return self.ensure_published_file(normalized)
                if normalized.is_dir():
                    # Directories are containers and are not individual manifest entries.
                    return normalized
                raise EvidenceError(f"evidence reference is not a file or directory: {value}")
        if required:
            rendered = contained[0] if contained else value
            raise EvidenceError(f"required evidence reference is missing: {rendered}")
        return None


def verify_manifest(evidence_root: Path | str, *,
                    manifest_path: Path | str | None = None,
                    required_paths: Iterable[str | Path] = ()) -> VerifiedEvidence:
    """Verify every published byte and return a containment-aware resolver."""
    root = Path(evidence_root).resolve(strict=True)
    if not root.is_dir():
        raise EvidenceError(f"evidence root is not a directory: {root}")
    if manifest_path is None:
        manifest_file = root / "evidence-manifest.json"
    else:
        manifest_file = Path(manifest_path)
        if not manifest_file.is_absolute():
            manifest_file = root / manifest_file
    try:
        relative_manifest = manifest_file.absolute().relative_to(root)
    except (OSError, ValueError) as exc:
        raise EvidenceError("evidence manifest must be a file inside the evidence root") from exc
    manifest_file = _regular_file_without_symlinks(
        root, PurePosixPath(relative_manifest.as_posix())
    )
    try:
        raw = manifest_file.read_bytes()
        manifest = json.loads(raw)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise EvidenceError(f"invalid evidence manifest: {manifest_file}") from exc
    if not isinstance(manifest, dict):
        raise EvidenceError("evidence manifest must be a JSON object")
    if manifest.get("schema_version") != 1 or manifest.get("transformation_version") != 1:
        raise EvidenceError("unsupported evidence manifest schema/transformation version")
    files = manifest.get("files")
    if not isinstance(files, list):
        raise EvidenceError("evidence manifest files must be a list")

    entries: dict[str, EvidenceEntry] = {}
    for index, source in enumerate(files):
        if not isinstance(source, dict):
            raise EvidenceError(f"evidence manifest file {index} must be an object")
        relative = _published_path(source.get("path")).as_posix()
        if relative in entries:
            raise EvidenceError(f"duplicate evidence manifest path: {relative}")
        size = source.get("size")
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise EvidenceError(f"invalid published size for {relative}")
        original_size = source.get("original_size")
        if (original_size is not None
                and (not isinstance(original_size, int) or isinstance(original_size, bool)
                     or original_size < 0)):
            raise EvidenceError(f"invalid original size for {relative}")
        original_sha = source.get("original_sha256")
        entries[relative] = EvidenceEntry(
            path=relative,
            sha256=_hex_sha256(source.get("sha256"), field=f"files[{index}].sha256"),
            size=size,
            original_sha256=(
                _hex_sha256(original_sha, field=f"files[{index}].original_sha256")
                if original_sha is not None else None
            ),
            original_size=original_size,
        )

    for entry in entries.values():
        relative = PurePosixPath(entry.path)
        path = _regular_file_without_symlinks(root, relative)
        actual_size = path.stat().st_size
        if actual_size != entry.size:
            raise EvidenceError(
                f"published evidence size differs for {entry.path}: "
                f"expected {entry.size}, found {actual_size}"
            )
        actual_sha = sha256_file(path)
        if actual_sha != entry.sha256:
            raise EvidenceError(
                f"published evidence SHA-256 differs for {entry.path}: "
                f"expected {entry.sha256}, found {actual_sha}"
            )

    verified = VerifiedEvidence(root, manifest_file, manifest, entries,
                                hashlib.sha256(raw).hexdigest())
    for required in required_paths:
        verified.require(required)
    return verified


def selection_identity_sha256(manifest: Mapping[str, Any]) -> str:
    """Hash immutable selection fields using the archive producer's contract."""
    value = {key: manifest[key] for key in SELECTION_IDENTITY_FIELDS}
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
