#!/usr/bin/env python3
"""Audit, stage, and qualify the LFM2.5-350M QNN community runner on Tachyon.

The local controller is conservative: model assets are pinned and hashed,
QNN work is serialized by the shared device lock, an owned CPU service is
stopped only after exact /proc identity validation, and performance claims are
gated on post-run QNN-only provider profiles.

The ``container-pilot`` subcommand runs inside the private ARM64 userspace. It
starts the audited loopback server, runs deterministic canaries plus three
short requests, shuts it down, and validates its final profiles.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import http.client
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tarfile
import time
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VENDOR = ROOT / "vendor/tachyon-qnn-lfm350-pilot"
DEFAULT_STUDY = ROOT / "results/tachyon-qnn-investigation-20261024-v1/lfm350-pilot"
DEFAULT_CPU_RECEIPT = (
    ROOT / "results/tachyon-worker-latency-20261023-v1/greedy-diagnostic/"
    "final-service-greedy-fast-default.json"
)
DEFAULT_RUNTIME_HANDOFF = (
    ROOT / "results/tachyon-qnn-investigation-20261024-v1/"
    "ubuntu24-qnn-runtime-handoff.json"
)
REMOTE_ROOT = "/external/device-workspace"
REMOTE_PRIVATE = REMOTE_ROOT + "/runtime/tachyon-qnn-lfm350-pilot"
REMOTE_RESULTS = REMOTE_ROOT + "/results/tachyon-qnn-investigation-20261024-v1/lfm350-pilot"
REMOTE_DEVICE_LOCK = REMOTE_ROOT + "/runtime/qnn-device.lock"
REMOTE_SOURCE = REMOTE_PRIVATE + "/source-49824267326c975a-audited"
REMOTE_ASSETS = REMOTE_PRIVATE + "/assets-773ff42cc383cb61"
REMOTE_STATE = REMOTE_PRIVATE + "/state-773ff42-4982426"
REMOTE_PILOT_OUTPUT = REMOTE_RESULTS + "/container-pilot"
UPSTREAM_COMMIT = "49824267326c975afb14f426f6a665f63f0f0762"
HF_REVISION = "773ff42cc383cb61ecf32eb13d1f828634fbd0e1"
HF_REPOSITORY = "PrismPhi/lfm2.5-350m-q6a-qcs6490-qnn-npu"
MODEL_SOURCE = "LiquidAI/LFM2.5-350M"
EXPECTED_FIRST_TOKEN_ID = 40550
QNN_WHEEL = {
    "filename": "onnxruntime_qnn-2.3.0-cp312-cp312-manylinux_2_34_aarch64.whl",
    "version": "2.3.0",
    "size": 78730222,
    "sha256": "1f0c072e1b0243f4ee5fabb9732323b20ecd6dd3af8b2fee8dbdacbd8eaf616d",
    "url": ("https://files.pythonhosted.org/packages/0c/59/"
            "d9026853ef37e4824b7668e78740fbeb503ce8f3a4cdd47313c57a84f7d8/"
            "onnxruntime_qnn-2.3.0-cp312-cp312-manylinux_2_34_aarch64.whl"),
}
EXPECTED_QNN_LIBRARIES = {
    "provider": "ebcec5c0b52cc4bb96542accd17f4a410174bfc7a44599586f17852aa9ae78ef",
    "htp": "9e1a73ed4f3e7cf3ef3199a23dfb3885b12a64edea30e607b0687b25b28e94f7",
    "v68_stub": "68e4c5f932ea006efa311c6763a4b25081f11663ae70390e6a58e41142f82c9f",
    "v68_skel": "1c4ef2cec89209ffb32d670cde10c3ff66733e44d34dd23c0b76f319afc5a6dd",
}
LATEST_QNN_WHEEL = {
    "filename": "onnxruntime_qnn-2.7.0-cp312-cp312-manylinux_2_34_aarch64.whl",
    "version": "2.7.0",
    "size": 88834007,
    "sha256": "40f6f8ab76a877b36c875122caf46189f810e52a1e7c2f75e431dfb7714168b8",
}
LATEST_ORT_WHEEL = {
    "filename": "onnxruntime-1.31.0-cp312-cp312-manylinux_2_28_aarch64.whl",
    "size": 21420804,
    "sha256": "35758d7606d578ec5b9d65f6e8a1f488013194c3f6097038a3223cb26d35ef9a",
}
LATEST_QNN_LIBRARIES = {
    "provider": "74fe7c68710c859c50967c8ce374587102ba6f097a2321c3ee0e55ac8feea5de",
    "htp": "7ec19d8002ad79509d1127cddb005e7bd07df1dfcb5ab31772078eaea5745a03",
    "v68_stub": "84e20670a3e25f7b7fa08b5f66134cb3bdeaa1c39f75940d4b5cce371390d53c",
    "v68_skel": "3f71eb57ce13a23d9dd92c6732782203ebb50245d2411812b06000008a2e1c21",
}
SSH_OPTIONS = (
    "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes",
    "-o", "ClearAllForwardings=yes", "-o", "ForwardAgent=no",
    "-o", "RequestTTY=no",
)


class PilotError(RuntimeError):
    pass


def remote_layout(variant: str) -> dict[str, str]:
    """Return disjoint mutable paths while sharing immutable source/assets."""
    if variant == "baseline":
        return {"private": REMOTE_PRIVATE, "results": REMOTE_RESULTS,
                "source": REMOTE_SOURCE, "assets": REMOTE_ASSETS,
                "state": REMOTE_STATE, "output": REMOTE_PILOT_OUTPUT}
    if variant == "latest-pair":
        private = REMOTE_PRIVATE + "/latest-pair-ort131-qnn27"
        results = REMOTE_RESULTS + "/latest-pair"
        return {"private": private, "results": results,
                "source": REMOTE_SOURCE, "assets": REMOTE_ASSETS,
                "state": private + "/state-773ff42-4982426-fresh",
                "output": results + "/container-pilot"}
    raise PilotError(f"unknown runtime variant: {variant}")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")


def json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PilotError(f"invalid JSON: {path}") from exc


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("xb") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2,
                                sort_keys=True).encode("utf-8") + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def append_event(path: Path, event: str, **fields: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {"at": utc_now(), "event": event, **fields}
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _run(argv: Sequence[str], *, timeout: float = 60,
         input_data: bytes | None = None) -> subprocess.CompletedProcess[bytes]:
    try:
        completed = subprocess.run(
            list(argv), input=input_data, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, timeout=timeout, check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise PilotError(f"command timed out: {argv[0]}") from exc
    if completed.returncode:
        detail = (completed.stdout + b"\n" + completed.stderr).decode("utf-8", "replace")[-4000:]
        raise PilotError(f"command failed ({completed.returncode}): {argv[0]}: {detail}")
    return completed


def source_manifest(vendor: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(vendor.rglob("*")):
        relative = path.relative_to(vendor)
        if ".git" in relative.parts or "__pycache__" in relative.parts \
                or path.suffix in {".pyc", ".pyo"} or path.is_dir():
            continue
        if path.is_symlink() or not path.is_file():
            raise PilotError(f"unsupported vendored source entry: {path}")
        rows.append({"path": relative.as_posix(), "size": path.stat().st_size,
                     "sha256": sha256(path)})
    if not rows:
        raise PilotError("vendored source tree is empty")
    return rows


def validate_asset_manifest(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, dict) or value.get("schema_version") != 2:
        raise PilotError("asset manifest schema differs")
    source = value.get("huggingface")
    if not isinstance(source, dict) or source.get("repository") != HF_REPOSITORY \
            or source.get("pinned_revision") != HF_REVISION \
            or source.get("source_revision") != HF_REVISION:
        raise PilotError("asset manifest source/revision differs")
    if value.get("source_model") != MODEL_SOURCE:
        raise PilotError("asset manifest is not derived from the expected instruct checkpoint")
    files = value.get("files")
    if not isinstance(files, list) or len(files) != 11:
        raise PilotError("asset manifest must contain exactly 11 files")
    seen: set[str] = set()
    normalized: list[dict[str, Any]] = []
    for item in files:
        if not isinstance(item, dict):
            raise PilotError("asset manifest entry is not an object")
        path = PurePosixPath(str(item.get("path", "")))
        digest = str(item.get("sha256", ""))
        size = item.get("size")
        if not path.parts or path.is_absolute() or ".." in path.parts \
                or path.as_posix() in seen:
            raise PilotError(f"unsafe/duplicate asset path: {path}")
        if not re.fullmatch(r"[0-9a-f]{64}", digest) \
                or isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise PilotError(f"invalid asset metadata: {path}")
        seen.add(path.as_posix())
        normalized.append({"path": path.as_posix(), "sha256": digest, "size": size})
    return normalized


def audit_vendor(vendor: Path = DEFAULT_VENDOR) -> dict[str, Any]:
    if not vendor.is_dir():
        raise PilotError(f"vendored source is absent: {vendor}")
    commit = _run(["git", "-C", str(vendor), "rev-parse", "HEAD"]).stdout.decode().strip()
    if commit != UPSTREAM_COMMIT:
        raise PilotError(f"vendored commit differs: {commit}")
    dirty = _run(["git", "-C", str(vendor), "status", "--porcelain"]).stdout.decode()
    if dirty:
        raise PilotError("vendored source has local modifications")
    manifest_path = vendor / "runner/config/model-assets.json"
    assets = validate_asset_manifest(read_json(manifest_path))
    canary = read_json(vendor / "runner/config/install-canary.json")
    if canary.get("expected_first_token_id") != EXPECTED_FIRST_TOKEN_ID \
            or canary.get("expected_subject") != "Tokyo" \
            or canary.get("evidence") != "records/evidence/install-canary-golden.json":
        raise PilotError("upstream semantic canary differs")
    licenses = {
        "code": {"path": "LICENSE", "sha256": sha256(vendor / "LICENSE")},
        "model_derivatives": {
            "path": "MODEL_LICENSE", "sha256": sha256(vendor / "MODEL_LICENSE"),
            "commercial_threshold_usd": 10_000_000,
        },
    }
    return {
        "schema_version": 1, "audited_at": utc_now(),
        "upstream_repository": "https://github.com/PrismPhi/radxa-dragon-q6a-qcs6490-lfm2.5-350m-qnn-npu",
        "upstream_commit": commit, "huggingface_repository": HF_REPOSITORY,
        "huggingface_revision": HF_REVISION, "source_checkpoint": MODEL_SOURCE,
        "source_checkpoint_kind": "general-purpose instruction-tuned; not Base",
        "converted_graph": "A16W8 QDQ model body; CPU embedding/tokenization/sampling",
        "converted_context_tokens": 2048,
        "asset_manifest_path": str(manifest_path.relative_to(ROOT)),
        "asset_manifest_sha256": sha256(manifest_path), "assets": assets,
        "source_files": source_manifest(vendor), "licenses": licenses,
        "qnn_python_runtime": {
            "wheel": QNN_WHEEL,
            "expected_embedded_library_sha256": EXPECTED_QNN_LIBRARIES,
            "qairt_version": None,
            "qnn_runtime_version": None,
            "version_note": "The Python distribution version is not treated as the QAIRT/QNN SDK runtime version.",
        },
        "matched_cpu_q8_control": {
            "implementation": "runner/scripts/run_practical_hybrid_prompt_v0.py:run_official_cpu_reference",
            "required_artifact": "official full Q8 ONNX checkpoint",
            "artifact_in_pinned_11_file_release": False,
            "status": "unavailable_skip",
            "reason": "The published runner exposes the control code but does not publish its required full Q8 ONNX checkpoint in the pinned release assets; this pilot will not invent or download a different control model.",
        },
        "audit_findings": [
            "No sudo, package-manager, credential, cloud-login, or arbitrary shell execution is used by the public installer.",
            "Network acquisition is isolated from execution: pinned assets are staged first and containers use network=none.",
            "Community claims require target-generated contexts, deterministic canaries, and post-profile QNN-only proof.",
            "No license click-through or gated repository access is performed.",
        ],
    }


def process_matches(receipt: Mapping[str, Any], identity: Mapping[str, Any]) -> bool:
    if not identity.get("alive"):
        return False
    for key in ("pid", "start_ticks", "argv"):
        if receipt.get(key) != identity.get(key):
            raise PilotError(f"owned process {key} differs")
    return True


def provider_proof(server_result: Mapping[str, Any]) -> dict[str, Any]:
    profiles = server_result.get("profiles")
    if not isinstance(profiles, list):
        raise PilotError("server profile evidence is absent")
    by_label = {str(item.get("label")): item for item in profiles if isinstance(item, dict)}
    proof: dict[str, Any] = {}
    for label in ("chunk", "decode"):
        item = by_label.get(label)
        counts = item.get("provider_counts") if isinstance(item, dict) else None
        if not isinstance(counts, dict):
            raise PilotError(f"{label} provider counts are absent")
        qnn = int(counts.get("QNNExecutionProvider", 0))
        cpu = int(counts.get("CPUExecutionProvider", 0))
        if qnn <= 0 or cpu != 0:
            raise PilotError(f"{label} is not QNN-only: {counts}")
        proof[label] = {"qnn_execution_count": qnn, "cpu_execution_count": cpu,
                        "profile": item.get("profile")}
    if server_result.get("qnn_only_verified") is not True:
        raise PilotError("server did not finalize qnn_only_verified=true")
    qnn_only = server_result.get("qnn_only_by_profile")
    if not isinstance(qnn_only, dict) or not all(qnn_only.get(x) is True for x in ("chunk", "decode")):
        raise PilotError("server QNN-only profile summary differs")
    return proof


def htp_execution_proof(server_result: Mapping[str, Any], server_log: str) -> dict[str, Any]:
    """Require evidence that a QNN profile represents HTP, not QNN's CPU backend."""
    profiles = provider_proof(server_result)
    fingerprint = server_result.get("runtime_fingerprint")
    if not isinstance(fingerprint, dict):
        raise PilotError("runtime fingerprint is absent")
    identity = fingerprint.get("identity")
    qnn = fingerprint.get("qnn")
    if not isinstance(identity, dict) or not isinstance(qnn, dict):
        raise PilotError("runtime QNN identity is absent")
    expected_backend = "/opt/qnn-venv/lib/python3.12/site-packages/onnxruntime_qnn/libQnnHtp.so"
    for label in ("execution_contract", "decode_execution_contract"):
        contract = fingerprint.get(label)
        if (not isinstance(contract, dict)
                or contract.get("provider_options", {}).get("backend_path") != expected_backend
                or contract.get("session_config", {}).get("session.disable_cpu_ep_fallback") != "1"):
            raise PilotError(f"{label} does not require the HTP backend with CPU fallback disabled")
    required_identity = {
        "onnxruntime": "1.31.0", "onnxruntime_qnn": "2.7.0",
        "provider_name": "QNNExecutionProvider", "htp_generation": "v68", "soc": "QCS6490",
        "qnn_ep_library_sha256": LATEST_QNN_LIBRARIES["provider"],
        "qnn_htp_library_sha256": LATEST_QNN_LIBRARIES["htp"],
        "htp_stub_sha256": LATEST_QNN_LIBRARIES["v68_stub"],
        "htp_skel_sha256": LATEST_QNN_LIBRARIES["v68_skel"],
    }
    if any(identity.get(key) != expected for key, expected in required_identity.items()):
        raise PilotError("latest HTP runtime identity differs")
    mapped = qnn.get("mapped_libraries")
    if not isinstance(mapped, list):
        raise PilotError("runtime mapped-library evidence is absent")
    mapped_by_name = {item.get("name"): item for item in mapped if isinstance(item, dict)}
    required_mapped = {
        "libQnnHtp.so": LATEST_QNN_LIBRARIES["htp"],
        "libQnnHtpV68Stub.so": LATEST_QNN_LIBRARIES["v68_stub"],
        "libQnnSystem.so": "93e25d5d8cb0a8788d336f25b9b1f71db757d5ae70a3a1ba91a8f9467d2e3de4",
        "libonnxruntime_providers_qnn.so": LATEST_QNN_LIBRARIES["provider"],
    }
    for name, digest in required_mapped.items():
        record = mapped_by_name.get(name)
        if not isinstance(record, dict) or record.get("exists") is not True or record.get("sha256") != digest:
            raise PilotError(f"mapped HTP runtime library differs: {name}")
    if "libQnnCpu.so" in mapped_by_name:
        raise PilotError("QNN CPU backend library was mapped")
    fastrpc_marker = "rpcmem APIs will be used from libxdsprpc"
    if fastrpc_marker not in server_log:
        raise PilotError("FastRPC HTP initialization evidence is absent")
    return {
        "schema_version": 1,
        "status": "passed",
        "claim": "QNN HTP/V68 execution; no ORT CPU EP and no QNN CPU backend",
        "provider_profiles": profiles,
        "backend_path": expected_backend,
        "runtime_identity": required_identity,
        "mapped_libraries": {name: {"sha256": digest, "path": mapped_by_name[name].get("path")}
                             for name, digest in required_mapped.items()},
        "qnn_cpu_backend_mapped": False,
        "fastrpc_marker_observed": True,
    }


def semantic_canary(normal: Mapping[str, Any], structured: Mapping[str, Any]) -> dict[str, Any]:
    try:
        text = normal["choices"][0]["message"]["content"]
        metrics = normal["qnn_metrics"]
        json_text = structured["choices"][0]["message"]["content"]
        json_value = json.loads(json_text)
        structured_metrics = structured["qnn_metrics"]
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
        raise PilotError("semantic canary response is malformed") from exc
    if not isinstance(text, str) or "tokyo" not in text.casefold():
        raise PilotError(f"normal semantic canary failed: {text!r}")
    if metrics.get("first_token_id") != EXPECTED_FIRST_TOKEN_ID:
        raise PilotError("normal first-token identity differs")
    if json_value != {"answer": "Tokyo"}:
        raise PilotError(f"JSON semantic canary failed: {json_value!r}")
    if metrics.get("all_outputs_finite") is not True or structured_metrics.get("all_outputs_finite") is not True:
        raise PilotError("semantic canary reports non-finite QNN output")
    return {"passed": True, "normal_text": text,
            "normal_first_token_id": metrics.get("first_token_id"), "json_object": json_value}


def request_cases() -> list[dict[str, Any]]:
    return [
        {"id": "sentiment-positive", "messages": [{"role": "user", "content":
         "Classify the sentiment. Return only positive, negative, or neutral: I loved the quick and careful support."}],
         "expected_contains": ["positive"]},
        {"id": "short-summary", "messages": [{"role": "user", "content":
         "Summarize in one short sentence: The launch moved from Monday to Thursday because the security review needs two more days."}],
         "expected_contains": ["thursday", "security"]},
        {"id": "route-security", "messages": [{"role": "user", "content":
         "Route this request to exactly one team. Return only security, billing, or general: My password was exposed in a public log."}],
         "expected_contains": ["security"]},
    ]


def _http_json(host: str, port: int, method: str, path: str,
               body: Mapping[str, Any] | None = None, timeout: float = 180) -> dict[str, Any]:
    connection = http.client.HTTPConnection(host, port, timeout=timeout)
    payload = None if body is None else json.dumps(body, ensure_ascii=False).encode("utf-8")
    headers = {} if payload is None else {"Content-Type": "application/json; charset=utf-8"}
    try:
        connection.request(method, path, body=payload, headers=headers)
        response = connection.getresponse(); raw = response.read(); status = response.status
    finally:
        connection.close()
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PilotError(f"HTTP {path} returned invalid JSON") from exc
    if status // 100 != 2 or not isinstance(value, dict):
        raise PilotError(f"HTTP {path} failed ({status}): {value}")
    return value


def _stream_request(host: str, port: int, payload: Mapping[str, Any], timeout: float = 180) -> dict[str, Any]:
    connection = http.client.HTTPConnection(host, port, timeout=timeout)
    encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    started = time.monotonic(); first_content: float | None = None
    fragments: list[str] = []; final: dict[str, Any] | None = None; events: list[dict[str, Any]] = []
    try:
        connection.request("POST", "/v1/chat/completions", body=encoded,
                           headers={"Content-Type": "application/json; charset=utf-8"})
        response = connection.getresponse()
        if response.status != 200:
            raise PilotError(f"stream request failed with HTTP {response.status}: {response.read()!r}")
        while True:
            line = response.readline()
            if not line: break
            decoded = line.decode("utf-8", "strict").strip()
            if not decoded or decoded.startswith("event:"): continue
            if not decoded.startswith("data: "): raise PilotError(f"invalid SSE line: {decoded!r}")
            data = decoded[6:]
            if data == "[DONE]": break
            event = json.loads(data); events.append(event); choices = event.get("choices") or []
            if choices:
                content = (choices[0].get("delta") or {}).get("content")
                if isinstance(content, str) and content:
                    if first_content is None: first_content = time.monotonic() - started
                    fragments.append(content)
                if choices[0].get("finish_reason") is not None: final = event
    finally:
        connection.close()
    if final is None: raise PilotError("stream ended without a terminal completion chunk")
    return {"text": "".join(fragments), "ttft_seconds": first_content,
            "wall_seconds": time.monotonic() - started, "terminal": final, "events": events}


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_container_pilot(args: argparse.Namespace) -> int:
    state = args.state_dir.resolve(); source = args.source_dir.resolve(); output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True); port = int(args.port); log_root = output / "server-logs"
    command = [sys.executable, str(source / "runner/scripts/server.py"), "--host", "127.0.0.1",
        "--port", str(port), "--timestamp", "tachyon_lfm350_pilot", "--log-root", str(log_root),
        "--chunk-context", str(state / "contexts/chunk/chunk_epcontext.onnx"),
        "--decode-context", str(state / "contexts/decode/decode_epcontext.onnx"),
        "--tokenizer", str(state / "models/tokenizer/tokenizer.json"),
        "--rope-cache", str(state / "models/host/rope_cache.npz"),
        "--embedding-int8-dir", str(state / "models/host/embedding_int8_rowwise"),
        "--v0-runner-dir", str(source / "runner/scripts"), "--chunk", "16", "--total-len", "2048",
        "--default-profile", "extraction"]
    server_log = output / "server.log"; started = time.monotonic()
    result: dict[str, Any] = {"schema_version": 1, "kind": "tachyon_lfm350_qnn_pilot",
        "started_at": utc_now(), "status": "running", "server_argv": command,
        "model": "LiquidAI/LFM2.5-350M instruct derivative A16W8 QDQ", "context_tokens": 2048,
        "requests": []}
    process: subprocess.Popen[Any] | None = None
    try:
        with server_log.open("wb") as log:
            process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            deadline = time.monotonic() + 180; health: dict[str, Any] | None = None
            while time.monotonic() < deadline:
                if process.poll() is not None: raise PilotError(f"server exited during initialization: {process.returncode}")
                try:
                    health = _http_json("127.0.0.1", port, "GET", "/health", timeout=2)
                    if health.get("ready"): break
                except (OSError, PilotError): time.sleep(0.25)
            if not health or not health.get("ready"): raise PilotError("server did not become ready")
            result["cold_server_init_seconds"] = time.monotonic() - started; result["health_at_ready"] = health
            common = {"model": "lfm2.5-350m-qnn-ctx2048", "messages": [{"role": "user", "content":
                "Complete this sentence with the city name only: The capital of Japan is"}],
                "temperature": 0, "seed": 2468}
            normal_request = {**common, "max_tokens": 8}
            json_request = {**common, "max_tokens": 32, "response_format": {"type": "json_object"}}
            normal = _http_json("127.0.0.1", port, "POST", "/v1/chat/completions", normal_request)
            structured = _http_json("127.0.0.1", port, "POST", "/v1/chat/completions", json_request)
            result["semantic_canary"] = semantic_canary(normal, structured)
            for name, value in (("canary-normal-request", normal_request), ("canary-normal-response", normal),
                                ("canary-json-request", json_request), ("canary-json-response", structured)):
                _write_json(output / f"{name}.json", value)
            for case in request_cases():
                payload = {"model": "lfm2.5-350m-qnn-ctx2048", "messages": case["messages"], "max_tokens": 64,
                    "temperature": 0.1, "top_k": 50, "top_p": 1.0, "repetition_penalty": 1.05,
                    "profile": "extraction", "seed": 20261024, "stream": True}
                observed = _stream_request("127.0.0.1", port, payload)
                evaluation = {"required_terms": case["expected_contains"],
                    "contains_required_terms": all(x in observed["text"].casefold() for x in case["expected_contains"])}
                record = {"id": case["id"], "request": payload, "response": observed, "evaluation": evaluation}
                raw_path = output / "requests" / f"{case['id']}.json"; _write_json(raw_path, record)
                result["requests"].append({"id": case["id"], "status": "completed",
                    "semantic_pass": evaluation["contains_required_terms"], "text": observed["text"],
                    "ttft_seconds": observed["ttft_seconds"], "wall_seconds": observed["wall_seconds"],
                    "usage": observed["terminal"].get("usage"),
                    "qnn_metrics": observed["terminal"].get("qnn_metrics"), "raw": str(raw_path)})
        if process.poll() is None: process.send_signal(signal.SIGTERM); process.wait(timeout=60)
        if process.returncode != 0: raise PilotError(f"server exited with {process.returncode}")
        server_result_path = log_root / "openai_server_tachyon_lfm350_pilot/server_result.json"
        server_result = read_json(server_result_path); result["provider_proof"] = provider_proof(server_result)
        result["server_result"] = str(server_result_path); result["status"] = "complete"
        result["finished_at"] = utc_now(); result["elapsed_seconds"] = time.monotonic() - started
        _write_json(output / "pilot-result.json", result); return 0
    except Exception as exc:
        result.update({"status": "failed", "finished_at": utc_now(), "error_type": type(exc).__name__, "error": str(exc)})
        _write_json(output / "pilot-result.json", result); raise
    finally:
        if process is not None and process.poll() is None:
            process.send_signal(signal.SIGTERM)
            try: process.wait(timeout=60)
            except subprocess.TimeoutExpired: process.kill(); process.wait(timeout=10)


class Remote:
    def __init__(self, destination: str) -> None: self.destination = destination
    def run(self, argv: Sequence[str], *, timeout: float = 90) -> bytes:
        command = shlex.join([str(item) for item in argv])
        return _run(["ssh", *SSH_OPTIONS, "-T", self.destination, command], timeout=timeout).stdout
    def python(self, script: str, payload: Mapping[str, Any], *, timeout: float = 90) -> Any:
        raw = self.run(["python3", "-c", script, json.dumps(payload, separators=(",", ":"))], timeout=timeout)
        try: return json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc: raise PilotError("remote Python returned invalid JSON") from exc
    def copy_to(self, local: Path, remote: str, *, recursive: bool = False, timeout: float = 300) -> None:
        argv = ["scp", *SSH_OPTIONS]; argv += ["-r"] if recursive else []; argv += [str(local), f"{self.destination}:{remote}"]
        _run(argv, timeout=timeout)
    def copy_from(self, remote: str, local: Path, *, recursive: bool = False, timeout: float = 300) -> None:
        argv = ["scp", *SSH_OPTIONS]; argv += ["-r"] if recursive else []; argv += [f"{self.destination}:{remote}", str(local)]
        _run(argv, timeout=timeout)


REMOTE_PREPARE = r'''import json,pathlib,sys
p=json.loads(sys.argv[1]); root=pathlib.Path(p["root"]); results=pathlib.Path(p["results"]); allowed=pathlib.Path("/external/device-workspace")
for x in (root,results,*[pathlib.Path(v) for v in p.get("directories",[])]):
 x=x.resolve()
 if allowed not in (x,*x.parents): raise SystemExit("unsafe path")
 x.mkdir(parents=True,exist_ok=True)
print(json.dumps({"root":str(root.resolve()),"results":str(results.resolve())}))'''

REMOTE_STAGE_ASSETS = r'''import hashlib,json,os,pathlib,sys,urllib.request
p=json.loads(sys.argv[1]); root=pathlib.Path(p["root"]).resolve(); allowed=pathlib.Path("/external/device-workspace/runtime/tachyon-qnn-lfm350-pilot")
if allowed not in (root,*root.parents): raise SystemExit("unsafe asset root")
root.mkdir(parents=True,exist_ok=True); out=[]
def digest(path):
 h=hashlib.sha256()
 with path.open("rb") as f:
  while True:
   b=f.read(1048576)
   if not b: break
   h.update(b)
 return h.hexdigest()
for item in p["files"]:
 rel=pathlib.PurePosixPath(item["path"])
 if rel.is_absolute() or ".." in rel.parts: raise SystemExit("unsafe asset path")
 target=root.joinpath(*rel.parts); target.parent.mkdir(parents=True,exist_ok=True)
 if target.is_file() and target.stat().st_size==item["size"] and digest(target)==item["sha256"]:
  out.append({"path":item["path"],"status":"reused","size":item["size"],"sha256":item["sha256"]}); continue
 if target.exists(): raise SystemExit("existing asset differs: "+str(target))
 partial=target.with_name(target.name+".partial")
 if partial.exists(): partial.unlink()
 url=p["base_url"].rstrip("/")+"/"+item["path"]
 try:
  with urllib.request.urlopen(url,timeout=180) as src,partial.open("xb") as dst:
   while True:
    block=src.read(1048576)
    if not block: break
    dst.write(block)
  if partial.stat().st_size!=item["size"] or digest(partial)!=item["sha256"]: raise RuntimeError("size/hash mismatch")
  os.replace(str(partial),str(target))
 except Exception:
  try: partial.unlink()
  except FileNotFoundError: pass
  raise
 out.append({"path":item["path"],"status":"downloaded","size":item["size"],"sha256":item["sha256"]})
print(json.dumps({"schema_version":1,"repository":p["repository"],"revision":p["revision"],"files":out}))'''

REMOTE_PROCESS = r'''import json,pathlib,sys
p=json.loads(sys.argv[1]); pid=int(p["pid"]); root=pathlib.Path("/proc")/str(pid)
try:
 stat=(root/"stat").read_text(); tail=stat[stat.rfind(")")+2:].split(); raw=(root/"cmdline").read_bytes(); argv=[x.decode("utf-8","surrogateescape") for x in raw.split(bytes([0])) if x]
 print(json.dumps({"pid":pid,"alive":tail[0]!="Z","state":tail[0],"start_ticks":int(tail[19]),"argv":argv}))
except FileNotFoundError: print(json.dumps({"pid":pid,"alive":False,"state":"exited","argv":[]}))'''

REMOTE_STOP = r'''import json,os,pathlib,signal,sys,time
p=json.loads(sys.argv[1]); e=p["expected"]; pid=int(e["pid"])
def ident():
 root=pathlib.Path("/proc")/str(pid)
 try:
  stat=(root/"stat").read_text(); t=stat[stat.rfind(")")+2:].split(); a=[x.decode("utf-8","surrogateescape") for x in (root/"cmdline").read_bytes().split(bytes([0])) if x]; return {"pid":pid,"alive":t[0]!="Z","state":t[0],"start_ticks":int(t[19]),"argv":a}
 except FileNotFoundError:return {"pid":pid,"alive":False,"state":"exited","argv":[]}
before=ident()
if not before["alive"]: print(json.dumps({"status":"already_exited","before":before})); raise SystemExit(0)
for k in ("pid","start_ticks","argv"):
 if before.get(k)!=e.get(k): print(json.dumps({"status":"ownership_mismatch","field":k,"before":before})); raise SystemExit(3)
os.kill(pid,signal.SIGTERM); deadline=time.monotonic()+30
while time.monotonic()<deadline:
 after=ident()
 if not after["alive"]: print(json.dumps({"status":"stopped","signal":"SIGTERM","before":before,"after":after})); raise SystemExit(0)
 time.sleep(.2)
print(json.dumps({"status":"timeout","before":before,"after":ident()})); raise SystemExit(4)'''

REMOTE_RESTORE = r'''import json,pathlib,subprocess,sys,time,urllib.request
p=json.loads(sys.argv[1]); argv=p["launch_argv"]; expected_argv=p["server_argv"]; log=pathlib.Path(p["log"]); receipt=pathlib.Path(p["receipt"])
log.parent.mkdir(parents=True,exist_ok=True); out=log.open("ab",buffering=0); proc=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT,cwd=p["cwd"],start_new_session=True)
def ident():
 root=pathlib.Path("/proc")/str(proc.pid)
 try:
  stat=(root/"stat").read_text(); t=stat[stat.rfind(")")+2:].split(); a=[x.decode("utf-8","surrogateescape") for x in (root/"cmdline").read_bytes().split(bytes([0])) if x]; return {"pid":proc.pid,"alive":t[0]!="Z","state":t[0],"start_ticks":int(t[19]),"argv":a}
 except FileNotFoundError:return {"pid":proc.pid,"alive":False,"argv":[]}
deadline=time.monotonic()+120; identity=None; endpoint=None
while time.monotonic()<deadline:
 identity=ident()
 if not identity["alive"]: print(json.dumps({"status":"exited","identity":identity})); raise SystemExit(3)
 if identity["argv"]==expected_argv:
  try:
   with urllib.request.urlopen(p["models_url"],timeout=2) as response: endpoint=json.loads(response.read().decode())
   ids=[x.get("id") for x in endpoint.get("data",[])]
   if p["alias"] in ids: break
  except Exception: pass
 time.sleep(.5)
else: print(json.dumps({"status":"not_ready","identity":identity,"endpoint":endpoint})); raise SystemExit(4)
value={"schema_version":1,"kind":"tachyon_qnn_lfm350_pilot_cpu_restore","launched_at":p["launched_at"],"pid":proc.pid,"start_ticks":identity["start_ticks"],"argv":identity["argv"],"launch_argv":argv,"alias":p["alias"],"log":str(log),"endpoint":endpoint,"status":"ready"}
receipt.write_text(json.dumps(value,indent=2,sort_keys=True)+"\n"); print(json.dumps(value))'''


def load_cpu_service(receipt_path: Path) -> dict[str, Any]:
    value = read_json(receipt_path); server = value.get("server")
    if server is None and value.get("kind") == "tachyon_qnn_lfm350_pilot_cpu_restore":
        server = dict(value)
        if not server.get("alias"):
            data = (server.get("endpoint") or {}).get("data") or []
            if any(isinstance(item, dict) and item.get("id") == "tachyon-lfm25-26b"
                   for item in data):
                server["alias"] = "tachyon-lfm25-26b"
    if not isinstance(server, dict): raise PilotError("CPU service receipt lacks server object")
    for key in ("pid", "start_ticks", "argv", "launch_argv"):
        if not server.get(key): raise PilotError(f"CPU service receipt lacks {key}")
    if server.get("alias") != "tachyon-lfm25-26b" or server["argv"][0] != REMOTE_ROOT + "/runtime/llama-cpp-cpu/bin/llama-server":
        raise PilotError("CPU service receipt has an unexpected runtime/alias")
    return dict(server)


def validate_runtime_handoff(value: Any, runtime_variant: str = "baseline") -> dict[str, Any]:
    if not isinstance(value, dict) or value.get("schema_version") != 1: raise PilotError("runtime handoff schema differs")
    if value.get("kind") != "tachyon_qnn_ubuntu24_python312_runtime": raise PilotError("runtime handoff kind differs")
    image = value.get("image")
    if not isinstance(image, dict) or not image.get("reference") or not re.fullmatch(r"sha256:[0-9a-f]{64}", str(image.get("digest", ""))):
        raise PilotError("runtime handoff image is not digest-pinned")
    if image.get("run_reference") != image.get("digest"):
        raise PilotError("runtime handoff must execute the exact local image content address")
    if runtime_variant == "baseline":
        expected_ort, expected_qnn = "1.27.0", "2.3.0"
        expected_wheel = QNN_WHEEL
        expected_libraries = EXPECTED_QNN_LIBRARIES
        expected_build = "v2.47.0.260601114230"
    elif runtime_variant == "latest-pair":
        expected_ort, expected_qnn = "1.31.0", "2.7.0"
        expected_wheel = LATEST_QNN_WHEEL
        expected_libraries = LATEST_QNN_LIBRARIES
        expected_build = "v2.51.0.260929181845"
        if value.get("variant") != "latest_ep27":
            raise PilotError("latest runtime handoff variant differs")
    else:
        raise PilotError(f"unknown runtime variant: {runtime_variant}")
    runtime = value.get("runtime")
    if not isinstance(runtime, dict) or runtime.get("python") != "/opt/qnn-venv/bin/python3.12": raise PilotError("runtime handoff Python differs")
    if runtime.get("onnxruntime") != expected_ort or runtime.get("onnxruntime_qnn") != expected_qnn:
        raise PilotError("runtime handoff ORT/QNN package versions differ")
    wheel = runtime.get("wheel")
    if not isinstance(wheel, dict) or any(wheel.get(key) != expected_wheel[key]
                                           for key in ("filename", "size", "sha256")):
        raise PilotError("runtime handoff QNN wheel pin differs")
    if runtime.get("embedded_htp_build") != expected_build:
        raise PilotError("runtime handoff embedded HTP build differs")
    if runtime_variant == "latest-pair":
        ort_wheel = runtime.get("onnxruntime_wheel")
        if not isinstance(ort_wheel, dict) or any(ort_wheel.get(key) != LATEST_ORT_WHEEL[key]
                                                 for key in ("filename", "size", "sha256")):
            raise PilotError("latest runtime ORT wheel pin differs")
        if value.get("environment", {}).get("ORT_QNN_ENABLE_CPU_BACKEND") != "1":
            raise PilotError("latest runtime provider discovery opt-in is absent")
    libraries = runtime.get("libraries")
    if not isinstance(libraries, dict):
        raise PilotError("runtime handoff library records are absent")
    for label, expected_hash in expected_libraries.items():
        record = libraries.get(label)
        if not isinstance(record, dict) or not record.get("path") \
                or record.get("sha256") != expected_hash:
            raise PilotError(f"runtime handoff {label} library identity differs")
    if runtime.get("qairt_version") is not None or runtime.get("qnn_runtime_version") is not None:
        raise PilotError("runtime handoff inferred an unsupported QAIRT/QNN runtime version")
    mounts = value.get("runtime_mounts")
    if not isinstance(mounts, list) or not mounts: raise PilotError("runtime handoff mount list is absent")
    for mount in mounts:
        if not isinstance(mount, dict) or not mount.get("source") or not mount.get("target") or mount.get("read_only") is not True:
            raise PilotError("runtime handoff mount is incomplete/not read-only")
    preflight = value.get("dependency_preflight")
    if not isinstance(preflight, dict) or preflight.get("schema_version") != 1 or preflight.get("status") != "passed":
        raise PilotError("runtime handoff dependency preflight is absent/not passed")
    if preflight.get("image_digest") != image["digest"] or preflight.get("unresolved") != []:
        raise PilotError("runtime dependency preflight is not clean/image-bound")
    checked = preflight.get("checked_libraries")
    if not isinstance(checked, list):
        raise PilotError("runtime dependency preflight library list is absent")
    by_name = {item.get("name"): item for item in checked if isinstance(item, dict)}
    required = {
        "libonnxruntime_providers_qnn.so": expected_libraries["provider"],
        "libQnnHtp.so": expected_libraries["htp"],
        "libQnnHtpV68Stub.so": expected_libraries["v68_stub"],
        "libQnnHtpPrepare.so": ("53fcebab0649bc14ee7532f268668018aac9f871b8c4a4105567bda105f85bd7"
                                   if runtime_variant == "latest-pair" else None),
        "libQnnSystem.so": ("93e25d5d8cb0a8788d336f25b9b1f71db757d5ae70a3a1ba91a8f9467d2e3de4"
                            if runtime_variant == "latest-pair" else None),
    }
    if set(by_name) != set(required):
        raise PilotError("runtime dependency preflight library set differs")
    for name, expected in required.items():
        digest = by_name[name].get("sha256")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest) or (expected and digest != expected):
            raise PilotError(f"runtime dependency preflight identity differs for {name}")
    ldd = preflight.get("ldd")
    provider = preflight.get("provider_registration")
    if not isinstance(ldd, dict) or ldd.get("status") != "passed" or ldd.get("unresolved") != []:
        raise PilotError("runtime dependency ldd preflight is not clean")
    if not isinstance(provider, dict) or provider.get("status") != "passed" or provider.get("provider") != "QNNExecutionProvider":
        raise PilotError("runtime provider preflight is not passed")
    evidence = [("ldd", ldd), ("provider", provider)]
    if runtime_variant == "baseline":
        graph = preflight.get("tiny_qnn_graph")
        if provider.get("rtld_now_all_checked_libraries") is not True:
            raise PilotError("runtime provider/RTLD preflight is not passed")
        if (not isinstance(graph, dict) or graph.get("status") != "passed"
                or graph.get("backend_build") != runtime.get("embedded_htp_build")
                or graph.get("context_reload") is not True
                or graph.get("original_exact_match") is not True
                or graph.get("restored_exact_match") is not True):
            raise PilotError("runtime tiny-QNN-graph preflight is not passed")
        evidence.append(("tiny graph", graph))
    else:
        if (provider.get("backend_execution") != "not_tested_controller_owned"
                or provider.get("required_environment") != {"ORT_QNN_ENABLE_CPU_BACKEND": "1"}
                or preflight.get("scope") != "host_dependencies_and_plugin_registration_only"):
            raise PilotError("latest runtime preflight overstates/misstates execution scope")
    libatomic = preflight.get("libatomic")
    if (not isinstance(libatomic, dict) or libatomic.get("path") != "/lib/aarch64-linux-gnu/libatomic.so.1"
            or not re.fullmatch(r"[0-9a-f]{64}", str(libatomic.get("sha256", "")))):
        raise PilotError("runtime libatomic preflight identity is absent")
    for label, record in evidence:
        if (not isinstance(record.get("local_path"), str)
                or not re.fullmatch(r"[0-9a-f]{64}", str(record.get("sha256", "")))):
            raise PilotError(f"runtime {label} preflight evidence pin is absent")
    return dict(value)


def verify_dependency_preflight_evidence(handoff: Mapping[str, Any], root: Path = ROOT) -> None:
    """Verify local evidence referenced by the image-bound preflight receipt."""
    root = root.resolve()
    preflight = handoff["dependency_preflight"]
    labels = ["ldd", "provider_registration"]
    if "tiny_qnn_graph" in preflight:
        labels.append("tiny_qnn_graph")
    for label in labels:
        record = preflight[label]
        path = (root / record["local_path"]).resolve()
        if path != root and root not in path.parents:
            raise PilotError(f"runtime {label} preflight evidence escapes repository root")
        if not path.is_file() or sha256(path) != record["sha256"]:
            raise PilotError(f"runtime {label} preflight evidence is missing or changed")


def build_container_command(handoff: Mapping[str, Any], *, mode: str, controller_path: str,
                            layout: Mapping[str, str] | None = None) -> list[str]:
    if mode not in {"install", "pilot"}: raise PilotError(f"unknown container mode: {mode}")
    layout = dict(layout or remote_layout("baseline"))
    image = handoff["image"]; runtime = handoff["runtime"]; name = f"tachyon-qnn-lfm350-{mode}-{os.getpid()}"
    command = [str(handoff.get("docker_binary", "/usr/bin/docker")), "run", "--rm", "--name", name,
        "--network", "none", "--read-only", "--cap-drop", "ALL",
        "--security-opt", "no-new-privileges", "--user", "1007:1005",
        "--group-add", "1000",
        "--device", "/dev/adsprpc-smd:/dev/adsprpc-smd:rwm",
        "--device", "/dev/ion:/dev/ion:rwm",
        "--tmpfs", "/tmp:rw,nosuid,nodev,noexec,size=512m",
        "--mount", f"type=bind,src={layout['source']},dst=/workspace/source,readonly",
        "--mount", f"type=bind,src={layout['assets']},dst=/workspace/assets,readonly",
        "--mount", f"type=bind,src={layout['state']},dst=/workspace/state",
        "--mount", f"type=bind,src={layout['output']},dst=/workspace/output",
        "--mount", f"type=bind,src={controller_path},dst=/workspace/controller.py,readonly"]
    for mount in handoff["runtime_mounts"]:
        option = f"type=bind,src={mount['source']},dst={mount['target']}"
        if mount.get("read_only"): option += ",readonly"
        command.extend(["--mount", option])
    environment = {"HOME": "/workspace/state", "PYTHONDONTWRITEBYTECODE": "1",
                   **(handoff.get("environment") or {})}
    for key, value in sorted(environment.items()): command.extend(["--env", f"{key}={value}"])
    command.extend(["--workdir", "/workspace/source"])
    # The pinned image has Python as its audited ENTRYPOINT. Arguments after the
    # image are the script and its arguments, not another interpreter.
    command.append(str(image["run_reference"]))
    if mode == "install":
        command.extend(["/workspace/source/runner/scripts/install.py", "--state-dir", "/workspace/state",
            "--asset-dir", "/workspace/assets", "--manifest", "/workspace/source/runner/config/model-assets.json",
            "--canary", "/workspace/source/runner/config/install-canary.json", "--smoke-port", "18086", "--force-context"])
    else:
        command.extend(["/workspace/controller.py", "container-pilot", "--state-dir", "/workspace/state",
            "--source-dir", "/workspace/source", "--output-dir", "/workspace/output", "--port", "18086"])
    return command


def create_source_archive(vendor: Path, audit: Mapping[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True); temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    with tarfile.open(temporary, "w") as archive:
        for item in audit["source_files"]: archive.add(vendor / item["path"], arcname=item["path"], recursive=False)
        payload = canonical({"commit": UPSTREAM_COMMIT, "source_manifest": audit["source_files"]})
        info = tarfile.TarInfo("PILOT-SOURCE-MANIFEST.json"); info.size = len(payload); info.mtime = 0; info.mode = 0o644
        archive.addfile(info, io.BytesIO(payload))
    os.replace(temporary, output)


REMOTE_EXTRACT_SOURCE = r'''import hashlib,json,os,pathlib,shutil,sys,tarfile
p=json.loads(sys.argv[1]); archive=pathlib.Path(p["archive"]).resolve(); target=pathlib.Path(p["target"]).resolve(); allowed=pathlib.Path("/external/device-workspace/runtime/tachyon-qnn-lfm350-pilot")
if allowed not in (target,*target.parents): raise SystemExit("unsafe source target")
def digest(path):
 h=hashlib.sha256()
 with path.open("rb") as f:
  while True:
   b=f.read(1048576)
   if not b: break
   h.update(b)
 return h.hexdigest()
def valid(root): return root.is_dir() and all((root/x["path"]).is_file() and (root/x["path"]).stat().st_size==x["size"] and digest(root/x["path"])==x["sha256"] for x in p["files"])
if target.exists():
 if not valid(target): raise SystemExit("existing source target differs")
 print(json.dumps({"status":"reused","target":str(target),"commit":p["commit"]})); raise SystemExit(0)
tmp=target.with_name(target.name+".partial-"+str(os.getpid())); tmp.mkdir(parents=True)
try:
 with tarfile.open(str(archive),"r") as tf:
  for member in tf.getmembers():
   rel=pathlib.PurePosixPath(member.name)
   if rel.is_absolute() or ".." in rel.parts or not member.isfile(): raise RuntimeError("unsafe archive member")
   dst=tmp.joinpath(*rel.parts); dst.parent.mkdir(parents=True,exist_ok=True); src=tf.extractfile(member)
   with dst.open("xb") as out: shutil.copyfileobj(src,out)
 if not valid(tmp): raise RuntimeError("deployed source hash differs")
 os.replace(str(tmp),str(target))
finally:
 if tmp.exists(): shutil.rmtree(tmp)
print(json.dumps({"status":"installed","target":str(target),"commit":p["commit"]}))'''

REMOTE_GATHER_INSTALL = r'''import json,pathlib,shutil,sys
p=json.loads(sys.argv[1]); state=pathlib.Path(p["state"]).resolve(); results=pathlib.Path(p["results"]).resolve(); allowed=pathlib.Path("/external/device-workspace")
for x in (state,results):
 if allowed not in (x,*x.parents): raise SystemExit("unsafe gather path")
logs=sorted((state/"logs").glob("install_*"),key=lambda x:x.stat().st_mtime)
if not logs: raise SystemExit("install logs absent")
source=logs[-1]; target=results/("install-logs-"+source.name)
if not target.exists(): shutil.copytree(str(source),str(target))
print(json.dumps({"source":str(source),"target":str(target),"status":"ready"}))'''


def remote_snapshot(remote: Remote) -> dict[str, Any]:
    script = r'''import glob,json,pathlib,sys
p=json.loads(sys.argv[1]); out={"memory":{},"thermal":[],"cpu_online":None}
for line in pathlib.Path("/proc/meminfo").read_text().splitlines():
 if ":" in line:
  k,v=line.split(":",1)
  if k in ("MemTotal","MemAvailable","SwapTotal","SwapFree"): out["memory"][k]=v.strip()
try: out["cpu_online"]=pathlib.Path("/sys/devices/system/cpu/online").read_text().strip()
except OSError: pass
for x in glob.glob("/sys/class/thermal/thermal_zone*"):
 try: out["thermal"].append({"path":x,"type":pathlib.Path(x,"type").read_text().strip(),"temp_millic":int(pathlib.Path(x,"temp").read_text())})
 except Exception: pass
print(json.dumps(out))'''
    value = remote.python(script, {})
    if not isinstance(value, dict): raise PilotError("remote snapshot malformed")
    return value


def stage_remote(remote: Remote, study: Path, vendor: Path, audit: Mapping[str, Any],
                 layout: Mapping[str, str]) -> tuple[dict[str, Any], str]:
    remote.python(REMOTE_PREPARE, {"root": layout["private"], "results": layout["results"],
                                  "directories": [layout["state"], layout["output"]]})
    archive = study / "staging/source.tar"; create_source_archive(vendor, audit, archive)
    remote_archive = layout["results"] + "/source.tar"; remote.copy_to(archive, remote_archive, timeout=300)
    source_receipt = remote.python(REMOTE_EXTRACT_SOURCE, {"archive": remote_archive, "target": layout["source"],
        "commit": UPSTREAM_COMMIT, "files": audit["source_files"]}, timeout=300)
    controller_hash = sha256(Path(__file__)); remote_controller = layout["results"] + f"/tachyon_qnn_lfm350_pilot-{controller_hash[:16]}.py"
    remote.copy_to(Path(__file__), remote_controller); observed = remote.run(["sha256sum", remote_controller]).decode().split()[0]
    if observed != controller_hash: raise PilotError("remote controller source hash differs")
    return source_receipt, remote_controller


def stage_assets(remote: Remote, audit: Mapping[str, Any], layout: Mapping[str, str]) -> dict[str, Any]:
    base = f"https://huggingface.co/{HF_REPOSITORY}/resolve/{HF_REVISION}"
    value = remote.python(REMOTE_STAGE_ASSETS, {"root": layout["assets"], "base_url": base,
        "repository": HF_REPOSITORY, "revision": HF_REVISION, "files": audit["assets"]}, timeout=3600)
    if not isinstance(value, dict) or len(value.get("files", [])) != 11: raise PilotError("remote asset staging receipt is incomplete")
    return value


def collect_evidence(remote: Remote, study: Path, layout: Mapping[str, str]) -> None:
    study.mkdir(parents=True, exist_ok=True)
    gathered = remote.python(REMOTE_GATHER_INSTALL, {"state": layout["state"], "results": layout["results"]})
    if not isinstance(gathered, dict) or gathered.get("status") != "ready":
        raise PilotError("install-log evidence gather failed")
    remote.copy_from(layout["output"], study, recursive=True, timeout=600)
    remote.copy_from(str(gathered["target"]), study, recursive=True, timeout=600)
    for name, path in {"install-result.json": layout["state"] + "/install-result.json",
        "chunk-source-stamp.json": layout["state"] + "/contexts/chunk/source-stamp.json",
        "decode-source-stamp.json": layout["state"] + "/contexts/decode/source-stamp.json"}.items():
        remote.copy_from(path, study / name, timeout=300)


def archive_prior_attempt(study: Path) -> Path | None:
    status_path = study / "pilot-status.json"
    if not status_path.is_file():
        return None
    prior = read_json(status_path)
    stamp = str(prior.get("finished_at") or prior.get("started_at") or utc_now())
    stamp = re.sub(r"[^0-9A-Za-z]+", "-", stamp).strip("-")
    target = study / "history" / f"attempt-{stamp}"
    suffix = 1
    while target.exists():
        target = study / "history" / f"attempt-{stamp}-{suffix}"
        suffix += 1
    target.mkdir(parents=True)
    for name in (
        "pilot-status.json", "source-pins.json", "install-container-command.json",
        "pilot-container-command.json", "cpu-service-stop.json",
        "restored-cpu-service.json", "install-container-stdout.log",
        "pilot-container-stdout.log",
    ):
        source = study / name
        if source.is_file():
            shutil.copy2(source, target / name)
    return target


def run_controller(args: argparse.Namespace) -> int:
    study = args.study.resolve(); vendor = args.vendor.resolve(); layout = remote_layout(args.variant)
    study.mkdir(parents=True, exist_ok=True); lock_path = study / ".controller.lock"
    with lock_path.open("a+", encoding="utf-8") as lock:
        try: fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc: raise PilotError("another LFM350 pilot controller holds the lock") from exc
        events = study / "events.jsonl"; status_path = study / "pilot-status.json"
        archived = archive_prior_attempt(study)
        if archived is not None:
            append_event(events, "prior_attempt_archived", archive_path=str(archived))
        audit = audit_vendor(vendor); atomic_json(study / "source-audit.json", audit)
        atomic_json(study / "matched-cpu-q8-control.json", audit["matched_cpu_q8_control"])
        handoff = validate_runtime_handoff(read_json(args.runtime_handoff.resolve()), args.variant)
        verify_dependency_preflight_evidence(handoff)
        cpu = load_cpu_service(args.cpu_receipt.resolve())
        pins = {"controller": {"path": str(Path(__file__)), "sha256": sha256(Path(__file__))},
            "vendor_commit": UPSTREAM_COMMIT, "vendor_manifest_sha256": audit["asset_manifest_sha256"],
            "huggingface_revision": HF_REVISION,
            "runtime_variant": args.variant,
            "runtime_handoff": {"path": str(args.runtime_handoff.resolve()), "sha256": sha256(args.runtime_handoff.resolve())},
            "cpu_service_receipt": {"path": str(args.cpu_receipt.resolve()), "sha256": sha256(args.cpu_receipt.resolve())}}
        atomic_json(study / "source-pins.json", pins)
        state: dict[str, Any] = {"schema_version": 1, "kind": "tachyon_qnn_lfm350_pilot", "status": "running",
            "started_at": utc_now(), "pins": pins, "runtime_variant": args.variant,
            "claim_scope": "community QNN runner qualification; no broader benchmark claim"}
        atomic_json(status_path, state); remote = Remote(args.remote); cpu_stopped = False
        try:
            append_event(events, "source_audit_complete", commit=UPSTREAM_COMMIT); state["snapshot_before"] = remote_snapshot(remote)
            source_receipt, remote_controller = stage_remote(remote, study, vendor, audit, layout)
            atomic_json(study / "remote-source-receipt.json", source_receipt); append_event(events, "source_staged", remote_source=layout["source"])
            assets = stage_assets(remote, audit, layout); atomic_json(study / "remote-assets-receipt.json", assets)
            append_event(events, "assets_staged", count=len(assets["files"]))
            identity = remote.python(REMOTE_PROCESS, {"pid": cpu["pid"]})
            if not isinstance(identity, dict) or not process_matches(cpu, identity): raise PilotError("expected CPU service is not alive")
            stop = remote.python(REMOTE_STOP, {"expected": cpu}, timeout=60)
            if stop.get("status") != "stopped": raise PilotError(f"CPU service did not stop cleanly: {stop}")
            cpu_stopped = True; atomic_json(study / "cpu-service-stop.json", stop); append_event(events, "cpu_service_stopped", pid=cpu["pid"])
            for mode, timeout in (("install", 2400), ("pilot", 900)):
                command = build_container_command(handoff, mode=mode, controller_path=remote_controller, layout=layout)
                atomic_json(study / f"{mode}-container-command.json", {"argv": command, "argv_sha256": json_sha256(command)})
                completed = remote.run(["flock", "-w", "600", REMOTE_DEVICE_LOCK, *command], timeout=timeout)
                (study / f"{mode}-container-stdout.log").write_bytes(completed); append_event(events, f"{mode}_container_complete")
            collect_evidence(remote, study, layout); pilot_result = read_json(study / "container-pilot/pilot-result.json")
            if pilot_result.get("status") != "complete" or pilot_result.get("semantic_canary", {}).get("passed") is not True:
                raise PilotError("container pilot did not pass its canary")
            state["pilot_result"] = pilot_result; state["snapshot_after_pilot"] = remote_snapshot(remote)
            state["status"] = "complete"; append_event(events, "pilot_complete")
        except Exception as exc:
            state.update({"status": "failed", "error_type": type(exc).__name__, "error": str(exc)})
            append_event(events, "pilot_failed", error_type=type(exc).__name__, error=str(exc)); raise
        finally:
            if cpu_stopped:
                try:
                    restore = remote.python(REMOTE_RESTORE, {"launch_argv": cpu["launch_argv"], "server_argv": cpu["argv"],
                        "cwd": REMOTE_ROOT, "log": layout["results"] + "/restored-cpu-service.log",
                        "receipt": layout["results"] + "/restored-cpu-service.json",
                        "models_url": "http://127.0.0.1:18085/v1/models", "alias": cpu["alias"], "launched_at": utc_now()}, timeout=150)
                    atomic_json(study / "restored-cpu-service.json", restore)
                    append_event(events, "cpu_service_restored", pid=restore.get("pid"))
                    state["restored_cpu_service"] = restore
                except Exception as restore_exc:
                    state.update({"status": "failed", "restoration_error_type": type(restore_exc).__name__,
                                  "restoration_error": str(restore_exc)})
                    append_event(events, "cpu_service_restore_failed", error=str(restore_exc))
                    atomic_json(status_path, state)
                    raise
            state["finished_at"] = utc_now(); atomic_json(status_path, state)
        return 0


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest="command", required=True)
    audit = sub.add_parser("audit"); audit.add_argument("--vendor", type=Path, default=DEFAULT_VENDOR); audit.add_argument("--output", type=Path)
    container = sub.add_parser("container-pilot"); container.add_argument("--state-dir", type=Path, required=True)
    container.add_argument("--source-dir", type=Path, required=True); container.add_argument("--output-dir", type=Path, required=True)
    container.add_argument("--port", type=int, default=18086)
    run = sub.add_parser("run"); run.add_argument("--study", type=Path, default=DEFAULT_STUDY)
    run.add_argument("--vendor", type=Path, default=DEFAULT_VENDOR); run.add_argument("--runtime-handoff", type=Path, default=DEFAULT_RUNTIME_HANDOFF)
    run.add_argument("--cpu-receipt", type=Path, default=DEFAULT_CPU_RECEIPT); run.add_argument("--remote", default="akovanda@tachyon")
    run.add_argument("--variant", choices=("baseline", "latest-pair"), default="baseline")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    if args.command == "audit":
        value = audit_vendor(args.vendor.resolve())
        if args.output: atomic_json(args.output.resolve(), value)
        print(json.dumps(value, sort_keys=True)); return 0
    if args.command == "container-pilot": return run_container_pilot(args)
    return run_controller(args)


if __name__ == "__main__":
    try: raise SystemExit(main())
    except PilotError as exc:
        print(f"error: {exc}", file=sys.stderr); raise SystemExit(1) from exc
