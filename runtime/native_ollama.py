#!/usr/bin/env python3
"""Prepare a portable Ollama runtime from an existing overlay2 image.

The public Ollama executable, common runtime libraries, and one CUDA backend
are streamed directly from Docker's read-only image layer directories.  This
does not create, start, stop, or remove containers.
"""

from __future__ import annotations

import argparse
import hashlib
import ipaddress
import io
import json
import os
import posixpath
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import BinaryIO, Iterable, Mapping, Sequence


DEFAULT_IMAGE = "ollama/ollama:0.40.6"
DEFAULT_DEST = Path("/external/storage/moe-bench/ollama-0.40.6")
DEFAULT_MODELS = Path("/external/models/run/ollama/models")
DEFAULT_HOST = "127.0.0.1:11440"
DESCRIPTOR_NAME = "runtime.json"
SCHEMA_VERSION = 1
READ_SIZE = 4 * 1024 * 1024
STREAM_MANIFEST = ".__native_ollama_source_manifest__.json"
EXPECTED_IMAGE_ID = "sha256:dbbbdcfe875a5e686e5d7651d0901ebe5a73a87f26cd948a4199717f9802bddd"
EXPECTED_REPO_DIGEST = "ollama/ollama@sha256:265409e1d931028a8f7d5a32bcbafddd7bffbc6cc629d77efe9560adf692a5e7"
BACKEND_RE = re.compile(r"^cuda_v(\d+)$")
VERSION_RE = re.compile(r"\b(?:version(?: is)?\s+)?v?(\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.-]+)?)\b", re.I)


class NativeOllamaError(RuntimeError):
    """The portable runtime could not be prepared or verified safely."""


@dataclass(frozen=True)
class OverlayNode:
    logical_path: str
    source_path: Path
    layer_index: int
    mode: int

    @property
    def kind(self) -> str:
        if stat.S_ISDIR(self.mode):
            return "directory"
        if stat.S_ISREG(self.mode):
            return "file"
        if stat.S_ISLNK(self.mode):
            return "symlink"
        return "special"


@dataclass(frozen=True)
class OverlayDirectory:
    logical_path: str
    sources: tuple[tuple[int, Path], ...]
    node: OverlayNode


def _run(command: Sequence[str], *, timeout: float = 30.0, env: Mapping[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command,
            check=True,
            text=True,
            capture_output=True,
            timeout=timeout,
            env=env,
        )
    except FileNotFoundError as exc:
        raise NativeOllamaError(f"required command is unavailable: {command[0]}") from exc
    except subprocess.TimeoutExpired as exc:
        raise NativeOllamaError(f"command timed out: {command[0]}") from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip()
        raise NativeOllamaError(f"command failed: {command[0]}: {detail}") from exc


def inspect_image(reference: str) -> dict:
    """Return one exact Docker image record using read-only image inspection."""
    result = _run(["docker", "image", "inspect", reference], timeout=60)
    try:
        records = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise NativeOllamaError("docker image inspect returned invalid JSON") from exc
    if not isinstance(records, list) or len(records) != 1 or not isinstance(records[0], dict):
        raise NativeOllamaError("docker image inspect did not resolve exactly one image")
    return records[0]


def image_layers(record: Mapping[str, object]) -> list[Path]:
    graph = record.get("GraphDriver")
    if not isinstance(graph, dict) or graph.get("Name") != "overlay2":
        raise NativeOllamaError("the image must use Docker's overlay2 graph driver")
    data = graph.get("Data")
    if not isinstance(data, dict):
        raise NativeOllamaError("image inspection omitted overlay2 layer data")
    raw_layers: list[str] = []
    upper = data.get("UpperDir")
    if isinstance(upper, str) and upper:
        raw_layers.append(upper)
    lower = data.get("LowerDir")
    if isinstance(lower, str) and lower:
        raw_layers.extend(value for value in lower.split(":") if value)
    if not raw_layers:
        raise NativeOllamaError("image inspection reported no overlay2 layers")
    layers = [Path(value) for value in raw_layers]
    if any(not value.is_absolute() or value.name != "diff" for value in layers):
        raise NativeOllamaError("image inspection reported an unexpected overlay2 layer path")
    if len(set(layers)) != len(layers):
        raise NativeOllamaError("image inspection reported duplicate overlay2 layers")
    return layers


def validate_pinned_image(reference: str, record: Mapping[str, object]) -> None:
    """Keep the default tag pinned while still allowing explicit older images."""
    if reference != DEFAULT_IMAGE:
        return
    image_id = record.get("Id")
    digests = record.get("RepoDigests")
    if image_id != EXPECTED_IMAGE_ID:
        raise NativeOllamaError(
            f"{DEFAULT_IMAGE} resolved to {image_id!r}, expected {EXPECTED_IMAGE_ID}"
        )
    if not isinstance(digests, list) or EXPECTED_REPO_DIGEST not in digests:
        raise NativeOllamaError(
            f"{DEFAULT_IMAGE} is missing expected repository digest {EXPECTED_REPO_DIGEST}"
        )


def _logical_parts(logical_path: str) -> tuple[str, ...]:
    path = PurePosixPath(logical_path)
    if not path.is_absolute() or ".." in path.parts:
        raise NativeOllamaError(f"invalid image path: {logical_path!r}")
    return tuple(part for part in path.parts if part != "/")


def _opaque(directory: Path) -> bool:
    if (directory / ".wh..wh..opq").exists():
        return True
    for name in ("trusted.overlay.opaque", "user.overlay.opaque"):
        try:
            if os.getxattr(directory, name) in (b"y", b"Y", b"x", b"X"):
                return True
        except (AttributeError, OSError):
            pass
    return False


def _whiteout_names(directory: Path) -> set[str]:
    hidden: set[str] = set()
    try:
        children = list(os.scandir(directory))
    except FileNotFoundError:
        return hidden
    for child in children:
        if child.name == ".wh..wh..opq":
            continue
        if child.name.startswith(".wh."):
            hidden.add(child.name[4:])
            continue
        try:
            child_stat = child.stat(follow_symlinks=False)
        except FileNotFoundError:
            continue
        if stat.S_ISCHR(child_stat.st_mode) and child_stat.st_rdev == os.makedev(0, 0):
            hidden.add(child.name)
    return hidden


def _child_directory(parent: OverlayDirectory, name: str) -> OverlayDirectory | OverlayNode | None:
    selected: OverlayNode | None = None
    directory_sources: list[tuple[int, Path]] = []
    for layer_index, parent_source in parent.sources:
        hidden = name in _whiteout_names(parent_source)
        candidate = parent_source / name
        try:
            candidate_stat = candidate.lstat()
        except FileNotFoundError:
            candidate_stat = None

        if selected is None and hidden:
            return None
        if candidate_stat is not None and not (
            stat.S_ISCHR(candidate_stat.st_mode) and candidate_stat.st_rdev == os.makedev(0, 0)
        ):
            node = OverlayNode(
                logical_path=posixpath.join(parent.logical_path, name),
                source_path=candidate,
                layer_index=layer_index,
                mode=candidate_stat.st_mode,
            )
            if selected is None:
                selected = node
                if not stat.S_ISDIR(node.mode):
                    return node
            if selected.kind == "directory" and stat.S_ISDIR(node.mode):
                directory_sources.append((layer_index, candidate))

        # A whiteout below a higher directory stops directory merging, while
        # preserving the higher directory itself.
        if hidden or _opaque(parent_source):
            break

    if selected is None:
        return None
    if selected.kind != "directory":
        return selected
    return OverlayDirectory(
        logical_path=selected.logical_path,
        sources=tuple(directory_sources),
        node=selected,
    )


def resolve_overlay(layers: Sequence[Path], logical_path: str) -> OverlayDirectory | OverlayNode | None:
    """Resolve a path over ordered overlay2 diffs, including whiteouts."""
    root_sources = tuple((index, layer) for index, layer in enumerate(layers) if layer.is_dir())
    if not root_sources:
        raise NativeOllamaError("none of the inspected overlay2 layer directories is readable")
    root_stat = root_sources[0][1].lstat()
    current: OverlayDirectory | OverlayNode = OverlayDirectory(
        logical_path="/",
        sources=root_sources,
        node=OverlayNode("/", root_sources[0][1], root_sources[0][0], root_stat.st_mode),
    )
    for part in _logical_parts(logical_path):
        if not isinstance(current, OverlayDirectory):
            return None
        child = _child_directory(current, part)
        if child is None:
            return None
        current = child
    return current


def merged_children(directory: OverlayDirectory) -> list[OverlayDirectory | OverlayNode]:
    names: set[str] = set()
    hidden: set[str] = set()
    for _, source in directory.sources:
        local_hidden = _whiteout_names(source)
        try:
            entries = list(os.scandir(source))
        except FileNotFoundError:
            entries = []
        for entry in entries:
            if entry.name.startswith(".wh.") or entry.name in hidden or entry.name in local_hidden:
                continue
            try:
                entry_stat = entry.stat(follow_symlinks=False)
            except FileNotFoundError:
                continue
            if stat.S_ISCHR(entry_stat.st_mode) and entry_stat.st_rdev == os.makedev(0, 0):
                continue
            names.add(entry.name)
        hidden.update(local_hidden)
        if _opaque(source):
            break
    result = []
    for name in sorted(names):
        child = _child_directory(directory, name)
        if child is not None:
            result.append(child)
    return result


def walk_overlay(node: OverlayDirectory | OverlayNode) -> Iterable[OverlayNode]:
    yield node.node if isinstance(node, OverlayDirectory) else node
    if isinstance(node, OverlayDirectory):
        for child in merged_children(node):
            yield from walk_overlay(child)


def available_cuda_backends(layers: Sequence[Path]) -> list[str]:
    library_root = resolve_overlay(layers, "/usr/lib/ollama")
    if not isinstance(library_root, OverlayDirectory):
        raise NativeOllamaError("image does not contain /usr/lib/ollama")
    result = []
    for child in merged_children(library_root):
        if isinstance(child, OverlayDirectory) and BACKEND_RE.fullmatch(PurePosixPath(child.logical_path).name):
            marker = resolve_overlay(layers, child.logical_path + "/libggml-cuda.so")
            if isinstance(marker, OverlayNode) and marker.kind == "file":
                result.append(PurePosixPath(child.logical_path).name)
    return sorted(result, key=lambda item: int(BACKEND_RE.fullmatch(item).group(1)))


def select_bundle(layers: Sequence[Path], requested_backend: str) -> tuple[str, list[OverlayNode]]:
    backends = available_cuda_backends(layers)
    if requested_backend == "auto":
        if not backends:
            raise NativeOllamaError("image has no CUDA backend")
        backend = backends[-1]
    else:
        backend = requested_backend
        if backend not in backends:
            available = ", ".join(backends) or "none"
            raise NativeOllamaError(f"CUDA backend {backend!r} is unavailable (available: {available})")

    binary = resolve_overlay(layers, "/usr/bin/ollama")
    if not isinstance(binary, OverlayNode) or binary.kind != "file":
        raise NativeOllamaError("image does not contain a regular /usr/bin/ollama")
    library_root = resolve_overlay(layers, "/usr/lib/ollama")
    if not isinstance(library_root, OverlayDirectory):
        raise NativeOllamaError("image does not contain /usr/lib/ollama")

    selected = [binary]
    for child in merged_children(library_root):
        name = PurePosixPath(child.logical_path).name
        if isinstance(child, OverlayDirectory):
            if name == backend:
                selected.extend(walk_overlay(child))
            continue
        # The include symlink targets the excluded MLX bundle and is not used
        # by the executable or CUDA runner.
        if name != "include":
            selected.append(child)
    return backend, selected


def archive_name(node: OverlayNode) -> str:
    if node.logical_path == "/usr/bin/ollama":
        return "bin/ollama"
    prefix = "/usr/lib/ollama/"
    if node.logical_path.startswith(prefix):
        return "lib/ollama/" + node.logical_path[len(prefix) :]
    raise NativeOllamaError(f"selected path is outside the portable bundle: {node.logical_path}")


class _DigestingReader:
    def __init__(self, source: BinaryIO):
        self.source = source
        self.digest = hashlib.sha256()
        self.count = 0

    def read(self, size: int = -1) -> bytes:
        data = self.source.read(size)
        self.digest.update(data)
        self.count += len(data)
        return data


def _tar_info(node: OverlayNode, name: str) -> tuple[tarfile.TarInfo, os.stat_result]:
    source_stat = node.source_path.lstat()
    info = tarfile.TarInfo(name)
    info.mode = stat.S_IMODE(source_stat.st_mode)
    info.mtime = int(source_stat.st_mtime)
    info.uid = 0
    info.gid = 0
    if stat.S_ISDIR(source_stat.st_mode):
        info.type = tarfile.DIRTYPE
        info.size = 0
    elif stat.S_ISREG(source_stat.st_mode):
        info.type = tarfile.REGTYPE
        info.size = source_stat.st_size
    elif stat.S_ISLNK(source_stat.st_mode):
        info.type = tarfile.SYMTYPE
        info.size = 0
        info.linkname = os.readlink(node.source_path)
    else:
        raise NativeOllamaError(f"refusing special image entry: {node.logical_path}")
    return info, source_stat


def emit_bundle(layers: Sequence[Path], requested_backend: str, output: BinaryIO) -> None:
    """Privileged half: resolve image files and write a tar stream to stdout."""
    backend, nodes = select_bundle(layers, requested_backend)
    records: list[dict[str, object]] = []
    with tarfile.open(fileobj=output, mode="w|") as archive:
        for node in nodes:
            name = archive_name(node)
            info, source_stat = _tar_info(node, name)
            record: dict[str, object] = {
                "path": name,
                "source_path": node.logical_path,
                "type": node.kind,
                "mode": stat.S_IMODE(source_stat.st_mode),
                "size": source_stat.st_size if node.kind == "file" else 0,
            }
            if node.kind == "file":
                with node.source_path.open("rb") as source:
                    reader = _DigestingReader(source)
                    archive.addfile(info, reader)
                if reader.count != source_stat.st_size:
                    raise NativeOllamaError(f"source changed while copying {node.logical_path}")
                record["sha256"] = reader.digest.hexdigest()
            elif node.kind == "symlink":
                record["target"] = info.linkname
                archive.addfile(info)
            else:
                archive.addfile(info)
            records.append(record)

        payload = (json.dumps({"backend": backend, "files": records}, sort_keys=True) + "\n").encode()
        info = tarfile.TarInfo(STREAM_MANIFEST)
        info.type = tarfile.REGTYPE
        info.mode = 0o600
        info.size = len(payload)
        info.mtime = int(datetime.now(timezone.utc).timestamp())
        archive.addfile(info, io.BytesIO(payload))


def _safe_member_path(root: Path, name: str) -> Path:
    pure = PurePosixPath(name)
    if pure.is_absolute() or not pure.parts or ".." in pure.parts:
        raise NativeOllamaError(f"unsafe archive member: {name!r}")
    return root.joinpath(*pure.parts)


def _safe_link_target(member_name: str, target: str) -> None:
    if not target or posixpath.isabs(target):
        raise NativeOllamaError(f"non-portable symlink in image: {member_name} -> {target}")
    resolved = posixpath.normpath(posixpath.join(posixpath.dirname(member_name), target))
    if resolved == ".." or resolved.startswith("../"):
        raise NativeOllamaError(f"escaping symlink in image: {member_name} -> {target}")


def extract_bundle(stream: BinaryIO, destination: Path) -> dict:
    """Unpack the controlled stream as the invoking user and verify every byte."""
    observed: dict[str, dict[str, object]] = {}
    directory_metadata: list[tuple[Path, int, int]] = []
    source_manifest: dict | None = None
    with tarfile.open(fileobj=stream, mode="r|*") as archive:
        for member in archive:
            if member.name == STREAM_MANIFEST:
                source = archive.extractfile(member)
                if source is None:
                    raise NativeOllamaError("source manifest was unreadable")
                try:
                    source_manifest = json.load(source)
                except json.JSONDecodeError as exc:
                    raise NativeOllamaError("source manifest was invalid") from exc
                continue
            if member.name in observed:
                raise NativeOllamaError(f"duplicate archive member: {member.name}")
            target = _safe_member_path(destination, member.name)
            target.parent.mkdir(parents=True, exist_ok=True)
            if member.isdir():
                target.mkdir(exist_ok=True)
                os.chmod(target, member.mode)
                directory_metadata.append((target, member.mode, member.mtime))
                observed[member.name] = {"type": "directory", "mode": member.mode, "size": 0}
            elif member.isreg():
                source = archive.extractfile(member)
                if source is None:
                    raise NativeOllamaError(f"archive member was unreadable: {member.name}")
                digest = hashlib.sha256()
                size = 0
                with target.open("xb") as output:
                    while True:
                        block = source.read(READ_SIZE)
                        if not block:
                            break
                        output.write(block)
                        digest.update(block)
                        size += len(block)
                os.chmod(target, member.mode)
                os.utime(target, (member.mtime, member.mtime))
                observed[member.name] = {
                    "type": "file",
                    "mode": member.mode,
                    "size": size,
                    "sha256": digest.hexdigest(),
                }
            elif member.issym():
                _safe_link_target(member.name, member.linkname)
                target.symlink_to(member.linkname)
                observed[member.name] = {
                    "type": "symlink",
                    "mode": member.mode,
                    "size": 0,
                    "target": member.linkname,
                }
            else:
                raise NativeOllamaError(f"refusing unsupported archive member: {member.name}")

    for directory, mode, mtime in reversed(directory_metadata):
        os.chmod(directory, mode)
        os.utime(directory, (mtime, mtime))
    if not isinstance(source_manifest, dict):
        raise NativeOllamaError("copy stream ended without its source manifest")
    expected_files = source_manifest.get("files")
    if not isinstance(expected_files, list):
        raise NativeOllamaError("source manifest has no files table")
    expected_by_name = {
        record.get("path"): record for record in expected_files if isinstance(record, dict)
    }
    if len(expected_by_name) != len(expected_files) or set(expected_by_name) != set(observed):
        raise NativeOllamaError("copied file set does not match the resolved image file set")
    for name, expected in expected_by_name.items():
        actual = observed[name]
        for field in ("type", "mode", "size", "sha256", "target"):
            if field in expected and expected[field] != actual.get(field):
                raise NativeOllamaError(f"copied {name} failed source {field} verification")
    return source_manifest


def _version(binary: Path, library_root: Path, backend: str) -> tuple[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "LD_LIBRARY_PATH": os.pathsep.join((str(library_root), str(library_root / backend))),
            "OLLAMA_LLM_LIBRARY": backend,
            "OLLAMA_HOST": "127.0.0.1:0",
            "OLLAMA_NO_CLOUD": "1",
            "OLLAMA_NOPRUNE": "1",
        }
    )
    result = _run([str(binary), "--version"], timeout=20, env=env)
    text = "\n".join(value.strip() for value in (result.stdout, result.stderr) if value.strip())
    match = VERSION_RE.search(text)
    if match is None:
        raise NativeOllamaError(f"could not parse Ollama version output: {text!r}")
    return match.group(1), text


def _validate_loopback_host(value: str) -> str:
    if "://" in value:
        raise NativeOllamaError("OLLAMA_HOST must be a loopback address and port without a URL scheme")
    if value.startswith("["):
        closing = value.find("]")
        if closing < 0 or closing + 1 >= len(value) or value[closing + 1] != ":":
            raise NativeOllamaError("invalid loopback host")
        host, port_text = value[1:closing], value[closing + 2 :]
    else:
        try:
            host, port_text = value.rsplit(":", 1)
        except ValueError as exc:
            raise NativeOllamaError("OLLAMA_HOST must include a port") from exc
    try:
        address = ipaddress.ip_address(host)
        port = int(port_text)
    except ValueError as exc:
        raise NativeOllamaError("OLLAMA_HOST must use a numeric loopback address and port") from exc
    if not address.is_loopback or not 1 <= port <= 65535:
        raise NativeOllamaError("OLLAMA_HOST must use a loopback address and port 1-65535")
    return value


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    try:
        with temporary.open("x", encoding="utf-8") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def make_descriptor(
    *,
    destination: Path,
    image_reference: str,
    image_record: Mapping[str, object],
    backend: str,
    version: str,
    version_output: str,
    source_files: Sequence[Mapping[str, object]],
    models: Path,
    host: str,
) -> dict:
    binary = destination / "bin" / "ollama"
    library_root = destination / "lib" / "ollama"
    library_path = os.pathsep.join((str(library_root), str(library_root / backend)))
    env = {
        "LD_LIBRARY_PATH": library_path,
        "OLLAMA_HOST": host,
        "OLLAMA_LLM_LIBRARY": backend,
        "OLLAMA_MODELS": str(models),
        "OLLAMA_NOPRUNE": "1",
        "OLLAMA_NO_CLOUD": "1",
    }
    home = Path.home()
    private_key = home / ".ollama" / "id_ed25519"
    public_key = home / ".ollama" / "id_ed25519.pub"
    repo_digests = image_record.get("RepoDigests")
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "source_image": {
            "reference": image_reference,
            "id": image_record.get("Id"),
            "repo_digests": repo_digests if isinstance(repo_digests, list) else [],
        },
        "ollama_version": version,
        "version_validation_output": version_output,
        "binary": str(binary),
        "library_root": str(library_root),
        "cuda_backend": backend,
        "files": list(source_files),
        "launch": {
            "argv": [str(binary), "serve"],
            "env": env,
            "endpoint": "http://" + host,
            "environment_policy": "overlay these values on the current environment; preserve HOME and CODEX_HOME",
            "home_identity": {
                "private_key": str(private_key),
                "private_key_present": private_key.is_file(),
                "public_key": str(public_key),
                "public_key_present": public_key.is_file(),
                "serve_behavior": (
                    f"Ollama {version} initializes this Ed25519 key pair before binding; "
                    "if the private key is absent, serve creates both files under the inherited HOME. "
                    "Ollama exposes no separate identity-path setting."
                ),
            },
        },
    }


def verify_descriptor(descriptor_path: Path, *, check_version: bool = True) -> dict:
    try:
        descriptor = json.loads(descriptor_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise NativeOllamaError(f"cannot read descriptor: {descriptor_path}") from exc
    if not isinstance(descriptor, dict) or descriptor.get("schema_version") != SCHEMA_VERSION:
        raise NativeOllamaError("unsupported runtime descriptor")
    binary = Path(str(descriptor.get("binary", "")))
    library_root = Path(str(descriptor.get("library_root", "")))
    backend = descriptor.get("cuda_backend")
    files = descriptor.get("files")
    if not binary.is_absolute() or not library_root.is_absolute() or not isinstance(backend, str) or not isinstance(files, list):
        raise NativeOllamaError("runtime descriptor is incomplete")
    root = binary.parent.parent
    for record in files:
        if not isinstance(record, dict) or not isinstance(record.get("path"), str):
            raise NativeOllamaError("runtime descriptor contains an invalid file record")
        path = _safe_member_path(root, record["path"])
        kind = record.get("type")
        try:
            path_stat = path.lstat()
        except FileNotFoundError as exc:
            raise NativeOllamaError(f"runtime file is missing: {path}") from exc
        if kind == "file":
            if not stat.S_ISREG(path_stat.st_mode) or path_stat.st_size != record.get("size"):
                raise NativeOllamaError(f"runtime file metadata mismatch: {path}")
            if stat.S_IMODE(path_stat.st_mode) != record.get("mode"):
                raise NativeOllamaError(f"runtime file mode mismatch: {path}")
            digest = hashlib.sha256()
            with path.open("rb") as source:
                while block := source.read(READ_SIZE):
                    digest.update(block)
            if digest.hexdigest() != record.get("sha256"):
                raise NativeOllamaError(f"runtime file digest mismatch: {path}")
        elif kind == "symlink":
            if not stat.S_ISLNK(path_stat.st_mode) or os.readlink(path) != record.get("target"):
                raise NativeOllamaError(f"runtime symlink mismatch: {path}")
        elif kind == "directory":
            if not stat.S_ISDIR(path_stat.st_mode):
                raise NativeOllamaError(f"runtime directory mismatch: {path}")
            if stat.S_IMODE(path_stat.st_mode) != record.get("mode"):
                raise NativeOllamaError(f"runtime directory mode mismatch: {path}")
        else:
            raise NativeOllamaError(f"unknown runtime entry type: {kind!r}")
    if check_version:
        version, _ = _version(binary, library_root, backend)
        if version != descriptor.get("ollama_version"):
            raise NativeOllamaError(
                f"runtime version changed: descriptor={descriptor.get('ollama_version')!r}, binary={version!r}"
            )
    return descriptor


def prepare(args: argparse.Namespace) -> dict:
    destination = args.dest.expanduser().resolve()
    models = args.models.expanduser().resolve()
    host = _validate_loopback_host(args.host)
    if destination.exists() or os.path.lexists(destination):
        raise NativeOllamaError(f"destination already exists; refusing to overwrite it: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    record = inspect_image(args.image)
    validate_pinned_image(args.image, record)
    layers = image_layers(record)

    staging = Path(tempfile.mkdtemp(prefix="." + destination.name + ".tmp.", dir=destination.parent))
    process: subprocess.Popen[bytes] | None = None
    try:
        command = [
            "sudo",
            "-n",
            sys.executable,
            str(Path(__file__).resolve()),
            "_emit",
            "--layers-json",
            json.dumps([str(layer) for layer in layers]),
            "--backend",
            args.backend,
        ]
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if process.stdout is None:
            raise NativeOllamaError("could not open the privileged copy stream")
        try:
            source_manifest = extract_bundle(process.stdout, staging)
        finally:
            process.stdout.close()
        returncode = process.wait(timeout=60)
        stderr = process.stderr.read().decode(errors="replace").strip() if process.stderr else ""
        if returncode != 0:
            raise NativeOllamaError(f"privileged image copy failed: {stderr or f'exit {returncode}'}")

        backend = source_manifest.get("backend")
        source_files = source_manifest.get("files")
        if not isinstance(backend, str) or not isinstance(source_files, list):
            raise NativeOllamaError("copy stream did not identify its CUDA backend and files")
        version, version_output = _version(
            staging / "bin" / "ollama", staging / "lib" / "ollama", backend
        )
        if args.image == DEFAULT_IMAGE and version != "0.40.6":
            raise NativeOllamaError(f"expected Ollama 0.40.6, copied binary reports {version}")
        descriptor = make_descriptor(
            destination=destination,
            image_reference=args.image,
            image_record=record,
            backend=backend,
            version=version,
            version_output=version_output,
            source_files=source_files,
            models=models,
            host=host,
        )
        _write_json(staging / DESCRIPTOR_NAME, descriptor)
        staging.rename(destination)
        verify_descriptor(destination / DESCRIPTOR_NAME)
        return descriptor
    except Exception:
        if process is not None and process.poll() is None:
            process.kill()
            process.wait()
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare", help="copy and verify a portable runtime")
    prepare_parser.add_argument("--image", default=DEFAULT_IMAGE, help="existing local Docker image reference")
    prepare_parser.add_argument("--dest", type=Path, default=DEFAULT_DEST, help="new portable runtime directory")
    prepare_parser.add_argument("--backend", default="auto", help="CUDA bundle such as cuda_v13 (default: newest present)")
    prepare_parser.add_argument("--models", type=Path, default=DEFAULT_MODELS, help="read-only model store for the launch descriptor")
    prepare_parser.add_argument("--host", default=DEFAULT_HOST, help="numeric loopback bind address and port")

    verify_parser = subparsers.add_parser("verify", help="rehash a prepared runtime")
    verify_parser.add_argument("descriptor", type=Path, help="runtime.json path")

    internal = subparsers.add_parser("_emit", help=argparse.SUPPRESS)
    internal.add_argument("--layers-json", required=True)
    internal.add_argument("--backend", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "_emit":
            try:
                raw_layers = json.loads(args.layers_json)
            except json.JSONDecodeError as exc:
                raise NativeOllamaError("invalid internal layer list") from exc
            if not isinstance(raw_layers, list) or not all(isinstance(value, str) for value in raw_layers):
                raise NativeOllamaError("invalid internal layer list")
            emit_bundle([Path(value) for value in raw_layers], args.backend, sys.stdout.buffer)
            return 0
        if args.command == "prepare":
            descriptor = prepare(args)
        else:
            descriptor = verify_descriptor(args.descriptor.expanduser().resolve())
        print(json.dumps(descriptor, indent=2, sort_keys=True))
        return 0
    except NativeOllamaError as exc:
        print(f"native_ollama: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
