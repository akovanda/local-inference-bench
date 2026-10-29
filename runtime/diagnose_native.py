#!/usr/bin/env python3
"""Diagnose one native Ollama model with a strictly CPU-only local server."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import stat
import sys
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__:
    from . import native_trial
else:
    import native_trial


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
DEFAULT_RUNTIME = Path("/external/storage/moe-bench/ollama-0.40.6/runtime.json")
DEFAULT_FIXTURE = Path("/external/storage/moe-bench/ollama-model-fixture")
MODEL = "gpt-oss-20b:64k"
HOST = "127.0.0.1"
PORT = 11440
ENDPOINT = f"http://{HOST}:{PORT}"
REQUEST_TIMEOUT = 15.0
MANIFEST_SHA256 = "a57efc7be387985a7da1e5894b5bf15994d4c7bc58a0396b36281392b3bee11a"
WEIGHT_DIGEST = "e7b273f9636059a689e3ddcab3716e4f65abe0143ac978e46673ad0e52d09efb"
PINNED_BLOBS = {
    "90845e31c8b5f1d2910d67f91dc7d401680d7dd8be247a23a72dddedb68ee76a": 489,
    WEIGHT_DIGEST: 13_793_422_144,
    "fa6710a93d78da62641e192361344be7a8c0a1c3737f139cf89f20ce1626b99c": 7_240,
    "f60356777647e927149cbd4c0ec1314a90caba9400ad205ddc4ce47ed001c2d6": 11_353,
    "869b7510d8f3b21f9fde08447a42469ff0bde2357eb270160e6db31fb68ccc9a": 48,
}


class DiagnosisError(RuntimeError):
    pass


@dataclass(frozen=True)
class ModelFiles:
    manifest: Path
    referenced: tuple[Path, ...]
    fingerprints: Mapping[str, tuple[int, int, int]]


def _regular_read_only_file(path: Path, expected_size: int | None = None) -> None:
    try:
        info = path.lstat()
    except OSError as exc:
        raise DiagnosisError(f"model file is inaccessible: {path}") from exc
    if not stat.S_ISREG(info.st_mode):
        raise DiagnosisError(f"model file is not a regular file: {path}")
    if not os.access(path, os.R_OK):
        raise DiagnosisError(f"model file is unreadable: {path}")
    if os.access(path, os.W_OK):
        raise DiagnosisError(f"model file is writable; refusing shared-store diagnosis: {path}")
    if expected_size is not None and info.st_size != expected_size:
        raise DiagnosisError(
            f"model blob size mismatch for {path.name}: {info.st_size} != {expected_size}"
        )


def inspect_model_files(models: Path, model: str = MODEL) -> ModelFiles:
    """Validate the selected legacy manifest without reading the large model blob."""
    if model != MODEL:
        raise DiagnosisError(f"this diagnostic is pinned to {MODEL}")
    models = models.expanduser().resolve()
    for directory in (models, models / "blobs", models / "manifests"):
        if not directory.is_dir() or not os.access(directory, os.R_OK | os.X_OK):
            raise DiagnosisError(f"model-store directory is inaccessible: {directory}")
        if os.access(directory, os.W_OK):
            raise DiagnosisError(f"model-store directory is writable: {directory}")

    manifest = (
        models
        / "manifests"
        / "registry.ollama.ai"
        / "library"
        / "gpt-oss-20b"
        / "64k"
    )
    _regular_read_only_file(manifest)
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DiagnosisError(f"model manifest is unreadable or invalid: {manifest}") from exc
    if not isinstance(payload, dict) or payload.get("schemaVersion") != 2:
        raise DiagnosisError("model manifest must use schemaVersion 2")

    records: list[Mapping[str, Any]] = []
    config = payload.get("config")
    layers = payload.get("layers")
    if not isinstance(config, dict) or not isinstance(layers, list) or not layers:
        raise DiagnosisError("model manifest is missing config or layers")
    records.append(config)
    if any(not isinstance(layer, dict) for layer in layers):
        raise DiagnosisError("model manifest contains a malformed layer")
    records.extend(layers)

    referenced: list[Path] = []
    for record in records:
        digest = record.get("digest")
        size = record.get("size")
        if (
            not isinstance(digest, str)
            or not digest.startswith("sha256:")
            or len(digest) != 71
            or any(character not in "0123456789abcdef" for character in digest[7:])
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size < 0
        ):
            raise DiagnosisError("model manifest contains an invalid digest or size")
        blob = models / "blobs" / digest.replace(":", "-")
        _regular_read_only_file(blob, size)
        referenced.append(blob)

    paths = (manifest, *referenced)
    fingerprints = {
        str(path): (path.stat().st_ino, path.stat().st_size, path.stat().st_mtime_ns)
        for path in paths
    }
    return ModelFiles(manifest, tuple(referenced), fingerprints)


def assert_model_files_unchanged(files: ModelFiles) -> None:
    for name, expected in files.fingerprints.items():
        path = Path(name)
        try:
            info = path.stat()
        except OSError as exc:
            raise DiagnosisError(f"selected model file changed or disappeared: {path}") from exc
        actual = (info.st_ino, info.st_size, info.st_mtime_ns)
        if actual != expected:
            raise DiagnosisError(f"selected model file changed during diagnosis: {path}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _copy_verified(source: Path, destination: Path, digest: str, size: int) -> None:
    if size > 16 * 1024 * 1024:
        raise DiagnosisError("refusing to copy a large model blob into the metadata fixture")
    _regular_read_only_file(source, size)
    if source.name != f"sha256-{digest}" or _sha256(source) != digest:
        raise DiagnosisError(f"source blob digest mismatch: {source}")
    payload = source.read_bytes()
    descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        written = 0
        while written < len(payload):
            count = os.write(descriptor, payload[written:])
            if count <= 0:
                raise OSError("short fixture write")
            written += count
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    if destination.stat().st_size != size or _sha256(destination) != digest:
        raise DiagnosisError(f"copied blob verification failed: {destination}")


def _validate_fixture(source_models: Path, fixture: Path, manifest_payload: bytes) -> None:
    manifest = fixture / "manifests/registry.ollama.ai/library/gpt-oss-20b/64k"
    if (
        not manifest.is_file()
        or manifest.stat().st_size != len(manifest_payload)
        or _sha256(manifest) != MANIFEST_SHA256
    ):
        raise DiagnosisError("fixture manifest verification failed")
    for digest, size in PINNED_BLOBS.items():
        source = source_models / "blobs" / f"sha256-{digest}"
        destination = fixture / "blobs" / f"sha256-{digest}"
        if digest == WEIGHT_DIGEST:
            if (
                not destination.is_symlink()
                or destination.resolve() != source
                or destination.stat().st_size != size
            ):
                raise DiagnosisError("fixture weight symlink verification failed")
        elif (
            not destination.is_file()
            or destination.stat().st_size != size
            or _sha256(destination) != digest
        ):
            raise DiagnosisError(f"fixture metadata blob verification failed: {destination}")


def prepare_fixture(source_models: Path, fixture: Path) -> Path:
    """Copy only metadata and link the pinned public weight into a new store."""
    source_models = source_models.expanduser().resolve()
    fixture = fixture.expanduser().resolve()
    files = inspect_model_files(source_models)
    manifest_payload = files.manifest.read_bytes()
    if len(manifest_payload) != 923 or hashlib.sha256(manifest_payload).hexdigest() != MANIFEST_SHA256:
        raise DiagnosisError("source 64k manifest does not match the pinned model")
    manifest_json = json.loads(manifest_payload)
    described = {
        record["digest"].removeprefix("sha256:"): record["size"]
        for record in (manifest_json["config"], *manifest_json["layers"])
    }
    if described != PINNED_BLOBS:
        raise DiagnosisError("source 64k manifest blob set does not match the pinned model")
    if os.path.lexists(fixture):
        if not fixture.is_dir():
            raise DiagnosisError(f"fixture destination is not a directory: {fixture}")
        _validate_fixture(source_models, fixture, manifest_payload)
        return fixture

    blobs = fixture / "blobs"
    manifest_dir = fixture / "manifests/registry.ollama.ai/library/gpt-oss-20b"
    manifest_dir.mkdir(parents=True, mode=0o755)
    blobs.mkdir(mode=0o755)
    destination_manifest = manifest_dir / "64k"
    descriptor = os.open(destination_manifest, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        written = 0
        while written < len(manifest_payload):
            count = os.write(descriptor, manifest_payload[written:])
            if count <= 0:
                raise OSError("short fixture manifest write")
            written += count
        os.fsync(descriptor)
    finally:
        os.close(descriptor)

    for digest, size in PINNED_BLOBS.items():
        source = source_models / "blobs" / f"sha256-{digest}"
        destination = blobs / f"sha256-{digest}"
        if digest == WEIGHT_DIGEST:
            _regular_read_only_file(source, size)
            os.symlink(source, destination)
            if not destination.is_symlink() or destination.resolve() != source:
                raise DiagnosisError("weight symlink verification failed")
            if destination.stat().st_size != size:
                raise DiagnosisError("weight symlink target size changed")
        else:
            _copy_verified(source, destination, digest, size)

    _validate_fixture(source_models, fixture, manifest_payload)
    return fixture


def write_fixture_descriptor(descriptor: Mapping[str, Any], fixture: Path) -> Path:
    """Write a native_trial descriptor that changes only the model-store path."""
    result = copy.deepcopy(dict(descriptor))
    result["launch"]["env"]["OLLAMA_MODELS"] = str(fixture)
    payload = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode("utf-8")
    path = fixture / "runtime.json"
    if path.exists():
        if path.read_bytes() != payload:
            raise DiagnosisError(f"fixture runtime descriptor differs: {path}")
        return path
    descriptor_fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        written = 0
        while written < len(payload):
            count = os.write(descriptor_fd, payload[written:])
            if count <= 0:
                raise OSError("short fixture descriptor write")
            written += count
        os.fsync(descriptor_fd)
    finally:
        os.close(descriptor_fd)
    return path


def write_fixture_report(source_models: Path, fixture: Path) -> Path:
    metadata = []
    for digest, size in PINNED_BLOBS.items():
        if digest == WEIGHT_DIGEST:
            continue
        metadata.append(
            {
                "sha256": digest,
                "size": size,
                "source": str(source_models / "blobs" / f"sha256-{digest}"),
                "copy": str(fixture / "blobs" / f"sha256-{digest}"),
            }
        )
    weight_source = source_models / "blobs" / f"sha256-{WEIGHT_DIGEST}"
    weight_link = fixture / "blobs" / f"sha256-{WEIGHT_DIGEST}"
    report = {
        "schema_version": 1,
        "manifest": {
            "sha256": MANIFEST_SHA256,
            "size": 923,
            "source": str(
                source_models
                / "manifests/registry.ollama.ai/library/gpt-oss-20b/64k"
            ),
            "copy": str(
                fixture / "manifests/registry.ollama.ai/library/gpt-oss-20b/64k"
            ),
        },
        "metadata_blobs": metadata,
        "weight": {
            "sha256": WEIGHT_DIGEST,
            "size": PINNED_BLOBS[WEIGHT_DIGEST],
            "source": str(weight_source),
            "link": str(weight_link),
            "link_target": os.readlink(weight_link),
        },
    }
    payload = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode("utf-8")
    path = fixture / "diagnostic-report.json"
    if path.exists():
        if path.read_bytes() != payload:
            raise DiagnosisError(f"fixture diagnostic report differs: {path}")
        return path
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        written = 0
        while written < len(payload):
            count = os.write(descriptor, payload[written:])
            if count <= 0:
                raise OSError("short fixture report write")
            written += count
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return path


def cpu_only_environment(
    descriptor: Mapping[str, Any],
    *,
    base_env: Mapping[str, str] | None = None,
    models: Path | None = None,
) -> dict[str, str]:
    source = dict(os.environ if base_env is None else base_env)
    preserved = {key: source.get(key) for key in ("HOME", "CODEX_HOME")}
    source.update(descriptor["launch"]["env"])
    for key, value in preserved.items():
        if value is None:
            source.pop(key, None)
        else:
            source[key] = value
    source.update(
        {
            "OLLAMA_HOST": f"{HOST}:{PORT}",
            "OLLAMA_NOPRUNE": "1",
            "OLLAMA_NO_CLOUD": "1",
            "OLLAMA_VULKAN": "false",
            "OLLAMA_LLM_LIBRARY": "cpu",
            "CUDA_VISIBLE_DEVICES": "-1",
            "HIP_VISIBLE_DEVICES": "-1",
            "ROCR_VISIBLE_DEVICES": "-1",
            "GGML_VK_VISIBLE_DEVICES": "-1",
        }
    )
    if models is not None:
        source["OLLAMA_MODELS"] = str(models)
    return source


def _error_text(body: bytes, fallback: str) -> str:
    try:
        payload = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return fallback[:2000]
    if isinstance(payload, dict) and isinstance(payload.get("error"), str):
        return payload["error"][:2000]
    return fallback[:2000]


def request_json(path: str, payload: Mapping[str, Any]) -> tuple[int, dict[str, Any] | None, str | None]:
    request = urllib.request.Request(
        ENDPOINT + path,
        data=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT) as response:
            body = response.read(8 * 1024 * 1024)
            decoded = json.loads(body)
            if not isinstance(decoded, dict):
                raise DiagnosisError(f"{path} returned a non-object JSON response")
            return response.status, decoded, None
    except urllib.error.HTTPError as exc:
        body = exc.read(64 * 1024)
        return exc.code, None, _error_text(body, str(exc))
    except (OSError, TimeoutError, json.JSONDecodeError) as exc:
        return 0, None, str(exc)[:2000]


def architecture_from_show(payload: Mapping[str, Any]) -> str | None:
    model_info = payload.get("model_info")
    if isinstance(model_info, dict):
        architecture = model_info.get("general.architecture")
        if isinstance(architecture, str):
            return architecture
    details = payload.get("details")
    if isinstance(details, dict):
        family = details.get("family")
        if isinstance(family, str):
            return family
    return None


def diagnose() -> dict[str, Any]:
    show_status, show_payload, show_error = request_json("/api/show", {"model": MODEL})
    result: dict[str, Any] = {
        "show": {
            "status": show_status,
            "error": show_error,
            "architecture": architecture_from_show(show_payload or {}),
        }
    }
    if show_status != 200 or show_payload is None:
        return result

    chat_status, _chat_payload, chat_error = request_json(
        "/api/chat",
        {
            "model": MODEL,
            "messages": [{"role": "user", "content": "Reply OK."}],
            "think": "low",
            "stream": False,
            "options": {
                "num_gpu": 0,
                "num_thread": 1,
                "num_ctx": 2048,
                "num_predict": 1,
            },
        },
    )
    result["chat"] = {"status": chat_status, "error": chat_error}
    return result


def output_paths(tag: str) -> tuple[Path, Path]:
    return RESULTS / f"{tag}-diagnosis.json", RESULTS / f"{tag}-server.log"


def write_result(path: Path, result: Mapping[str, Any]) -> None:
    payload = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        written = 0
        while written < len(payload):
            count = os.write(descriptor, payload[written:])
            if count <= 0:
                raise OSError("short diagnosis write")
            written += count
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, default=DEFAULT_RUNTIME)
    parser.add_argument(
        "--fixture",
        type=Path,
        help=f"create and use a private metadata store (recommended: {DEFAULT_FIXTURE})",
    )
    parser.add_argument(
        "--prepare-fixture-only",
        action="store_true",
        help="validate/write the fixture and GPU trial descriptor without starting Ollama",
    )
    parser.add_argument("--tag", default=f"native-diagnosis-{uuid.uuid4().hex[:8]}")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if native_trial.TAG_RE.fullmatch(args.tag) is None:
        raise SystemExit("diagnose_native: invalid --tag")
    if args.prepare_fixture_only:
        if args.fixture is None:
            raise SystemExit("diagnose_native: --prepare-fixture-only requires --fixture")
        try:
            descriptor = native_trial.load_descriptor(args.runtime)
            models = Path(descriptor["launch"]["env"]["OLLAMA_MODELS"])
            original_files = inspect_model_files(models)
            fixture = prepare_fixture(models, args.fixture)
            runtime_path = write_fixture_descriptor(descriptor, fixture)
            report_path = write_fixture_report(models.resolve(), fixture)
            assert_model_files_unchanged(original_files)
        except (DiagnosisError, native_trial.TrialError, OSError) as exc:
            print(f"diagnose_native: {exc}", file=sys.stderr)
            return 1
        print(
            json.dumps(
                {"fixture": str(fixture), "runtime": str(runtime_path), "report": str(report_path)},
                separators=(",", ":"),
            ),
            flush=True,
        )
        return 0
    output, server_log = output_paths(args.tag)
    if os.path.lexists(output) or os.path.lexists(server_log):
        raise SystemExit("diagnose_native: tag would overwrite an existing artifact")

    handle: native_trial.ServerHandle | None = None
    model_files: ModelFiles | None = None
    exit_code = 1
    try:
        descriptor = native_trial.load_descriptor(args.runtime)
        models = Path(descriptor["launch"]["env"]["OLLAMA_MODELS"])
        model_files = inspect_model_files(models)
        if args.fixture is not None:
            active_models = prepare_fixture(models, args.fixture)
            write_fixture_descriptor(descriptor, active_models)
            write_fixture_report(models, active_models)
        else:
            active_models = models
        if native_trial.port_is_occupied():
            raise DiagnosisError(f"loopback port {PORT} is occupied")
        environment = cpu_only_environment(descriptor, models=active_models)
        with native_trial.defer_termination_signals():
            handle = native_trial.start_server(descriptor, environment, server_log)
        native_trial.wait_for_server(handle, descriptor.get("ollama_version"))
        result = diagnose()
        write_result(output, result)
        show = result["show"]
        chat = result.get("chat")
        print(json.dumps(result, separators=(",", ":")), flush=True)
        exit_code = 0 if show["status"] == 200 and chat and chat["status"] == 200 else 1
    except (DiagnosisError, native_trial.TrialError, OSError) as exc:
        print(f"diagnose_native: {exc}", file=sys.stderr)
    finally:
        if handle is not None:
            try:
                with native_trial.defer_termination_signals():
                    native_trial.stop_server(handle)
            except (native_trial.TrialError, OSError) as exc:
                print(f"diagnose_native: cleanup failed: {exc}", file=sys.stderr)
                exit_code = 1
        if model_files is not None:
            try:
                assert_model_files_unchanged(model_files)
            except DiagnosisError as exc:
                print(f"diagnose_native: {exc}", file=sys.stderr)
                exit_code = 1
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
