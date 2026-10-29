import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from runtime import tachyon_qnn_lfm350_pilot as pilot


def handoff():
    libraries = {}
    names = {
        "provider": "libonnxruntime_providers_qnn.so",
        "htp": "libQnnHtp.so",
        "v68_stub": "libQnnHtpV68Stub.so",
        "v68_skel": "libQnnHtpV68Skel.so",
    }
    for label, digest in pilot.EXPECTED_QNN_LIBRARIES.items():
        libraries[label] = {
            "path": "/opt/qnn-venv/lib/python3.12/site-packages/onnxruntime_qnn/" + names[label],
            "sha256": digest,
        }
    image_digest = "sha256:" + "a" * 64
    return {
        "schema_version": 1,
        "kind": "tachyon_qnn_ubuntu24_python312_runtime",
        "image": {
            "reference": "tachyon-qnn-python312",
            "digest": image_digest,
            "run_reference": image_digest,
        },
        "runtime": {
            "python": "/opt/qnn-venv/bin/python3.12",
            "onnxruntime": "1.27.0",
            "onnxruntime_qnn": "2.3.0",
            "wheel": dict(pilot.QNN_WHEEL),
            "libraries": libraries,
            "qairt_version": None,
            "qnn_runtime_version": None,
            "embedded_htp_build": "v2.47.0.260601114230",
        },
        "runtime_mounts": [
            {"source": "/usr/lib/libcdsprpc.so", "target": "/usr/lib/libcdsprpc.so",
             "read_only": True},
        ],
        "environment": {"LD_LIBRARY_PATH": "/usr/lib"},
        "dependency_preflight": {
            "schema_version": 1,
            "status": "passed",
            "image_digest": image_digest,
            "unresolved": [],
            "checked_libraries": [
                {"name": "libonnxruntime_providers_qnn.so", "sha256": pilot.EXPECTED_QNN_LIBRARIES["provider"]},
                {"name": "libQnnHtp.so", "sha256": pilot.EXPECTED_QNN_LIBRARIES["htp"]},
                {"name": "libQnnHtpPrepare.so", "sha256": "4" * 64},
                {"name": "libQnnHtpV68Stub.so", "sha256": pilot.EXPECTED_QNN_LIBRARIES["v68_stub"]},
                {"name": "libQnnSystem.so", "sha256": "5" * 64},
            ],
            "ldd": {"status": "passed", "unresolved": [], "local_path": "evidence/ldd.txt", "sha256": "1" * 64},
            "provider_registration": {"status": "passed", "provider": "QNNExecutionProvider",
                "rtld_now_all_checked_libraries": True, "local_path": "evidence/provider.txt", "sha256": "2" * 64},
            "tiny_qnn_graph": {"status": "passed", "backend_build": "v2.47.0.260601114230", "context_reload": True,
                "original_exact_match": True, "restored_exact_match": True,
                "local_path": "evidence/graph.txt", "sha256": "3" * 64},
            "libatomic": {"path": "/lib/aarch64-linux-gnu/libatomic.so.1", "sha256": "6" * 64},
        },
    }


def test_actual_vendor_audit_pins_instruct_source_and_exact_assets():
    value = pilot.audit_vendor()
    assert value["upstream_commit"] == pilot.UPSTREAM_COMMIT
    assert value["huggingface_revision"] == pilot.HF_REVISION
    assert value["source_checkpoint"] == "LiquidAI/LFM2.5-350M"
    assert value["source_checkpoint_kind"] == "general-purpose instruction-tuned; not Base"
    assert value["converted_context_tokens"] == 2048
    assert len(value["assets"]) == 11
    assert sum(item["size"] for item in value["assets"]) == 820_636_910
    assert value["qnn_python_runtime"]["wheel"]["sha256"] == (
        "1f0c072e1b0243f4ee5fabb9732323b20ecd6dd3af8b2fee8dbdacbd8eaf616d"
    )
    assert value["matched_cpu_q8_control"]["status"] == "unavailable_skip"
    assert value["matched_cpu_q8_control"]["artifact_in_pinned_11_file_release"] is False


def test_asset_manifest_rejects_traversal_even_with_valid_hash_and_count():
    manifest = pilot.read_json(
        pilot.DEFAULT_VENDOR / "runner/config/model-assets.json"
    )
    changed = copy.deepcopy(manifest)
    changed["files"][0]["path"] = "../MODEL_LICENSE"
    with pytest.raises(pilot.PilotError, match="unsafe/duplicate"):
        pilot.validate_asset_manifest(changed)


def test_provider_proof_requires_both_graphs_and_rejects_cpu_fallback():
    complete = {
        "qnn_only_verified": True,
        "qnn_only_by_profile": {"chunk": True, "decode": True},
        "profiles": [
            {"label": "chunk", "provider_counts": {
                "QNNExecutionProvider": 4, "CPUExecutionProvider": 0}},
            {"label": "decode", "provider_counts": {
                "QNNExecutionProvider": 19, "CPUExecutionProvider": 0}},
        ],
    }
    proof = pilot.provider_proof(complete)
    assert proof["chunk"]["qnn_execution_count"] == 4
    assert proof["decode"]["cpu_execution_count"] == 0

    fallback = copy.deepcopy(complete)
    fallback["profiles"][1]["provider_counts"]["CPUExecutionProvider"] = 1
    with pytest.raises(pilot.PilotError, match="not QNN-only"):
        pilot.provider_proof(fallback)

    missing_decode = copy.deepcopy(complete)
    missing_decode["profiles"] = missing_decode["profiles"][:1]
    with pytest.raises(pilot.PilotError, match="decode provider counts"):
        pilot.provider_proof(missing_decode)


def test_semantic_canary_checks_first_token_subject_json_and_finite_outputs():
    normal = {
        "choices": [{"message": {"content": "Tokyo"}}],
        "qnn_metrics": {"first_token_id": 40550, "all_outputs_finite": True},
    }
    structured = {
        "choices": [{"message": {"content": '{"answer":"Tokyo"}'}}],
        "qnn_metrics": {"all_outputs_finite": True},
    }
    assert pilot.semantic_canary(normal, structured)["passed"] is True
    wrong = copy.deepcopy(normal)
    wrong["qnn_metrics"]["first_token_id"] = 1
    with pytest.raises(pilot.PilotError, match="first-token"):
        pilot.semantic_canary(wrong, structured)


def test_runtime_handoff_and_container_argv_pin_security_and_library_hashes():
    value = pilot.validate_runtime_handoff(handoff())
    command = pilot.build_container_command(
        value, mode="install", controller_path="/remote/controller.py"
    )
    joined = " ".join(command)
    assert "--network none" in joined
    assert "--read-only" in command
    assert "--cap-drop ALL" in joined
    assert "--security-opt no-new-privileges" in joined
    assert "--user 1007:1005" in joined
    assert "--device /dev/adsprpc-smd" in joined
    assert "--device /dev/ion" in joined
    assert "readonly" in joined
    assert "noexec,size=512m" in joined
    assert "/workspace/source" in joined
    assert "--privileged" not in command
    assert value["image"]["run_reference"] in command
    image_index = command.index(value["image"]["run_reference"])
    assert command[image_index + 1] == "/workspace/source/runner/scripts/install.py"
    assert value["runtime"]["python"] not in command

    changed = handoff()
    changed["runtime"]["libraries"]["provider"]["sha256"] = "0" * 64
    with pytest.raises(pilot.PilotError, match="provider library identity"):
        pilot.validate_runtime_handoff(changed)

    changed = handoff()
    changed["dependency_preflight"]["image_digest"] = "sha256:" + "0" * 64
    with pytest.raises(pilot.PilotError, match="not clean/image-bound"):
        pilot.validate_runtime_handoff(changed)


def test_dependency_preflight_evidence_hashes_are_required_before_cpu_stop(tmp_path):
    value = pilot.validate_runtime_handoff(handoff())
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    payloads = {
        "ldd": b"ldd clean\n",
        "provider_registration": b"provider passed\n",
        "tiny_qnn_graph": b"graph passed\n",
    }
    for label, payload in payloads.items():
        path = evidence / Path(value["dependency_preflight"][label]["local_path"]).name
        path.write_bytes(payload)
        value["dependency_preflight"][label]["sha256"] = pilot.sha256(path)
    pilot.verify_dependency_preflight_evidence(value, root=tmp_path)
    (evidence / "provider.txt").write_text("changed")
    with pytest.raises(pilot.PilotError, match="missing or changed"):
        pilot.verify_dependency_preflight_evidence(value, root=tmp_path)


def test_actual_latest_pair_handoff_is_exact_and_requires_discovery_opt_in():
    path = (pilot.ROOT / "results/tachyon-qnn-investigation-20261024-v1/"
            "ubuntu24-qnn-runtime-handoff-latest-ep27.json")
    value = pilot.validate_runtime_handoff(pilot.read_json(path), "latest-pair")
    pilot.verify_dependency_preflight_evidence(value)
    command = pilot.build_container_command(
        value, mode="install", controller_path="/remote/controller.py",
        layout=pilot.remote_layout("latest-pair"),
    )
    assert "ORT_QNN_ENABLE_CPU_BACKEND=1" in command
    assert value["image"]["digest"] in command
    assert pilot.remote_layout("latest-pair")["state"] in " ".join(command)
    changed = copy.deepcopy(value)
    changed["environment"].pop("ORT_QNN_ENABLE_CPU_BACKEND")
    with pytest.raises(pilot.PilotError, match="discovery opt-in"):
        pilot.validate_runtime_handoff(changed, "latest-pair")
    with pytest.raises(pilot.PilotError, match="package versions differ"):
        pilot.validate_runtime_handoff(value, "baseline")


def test_actual_latest_result_proves_htp_and_rejects_qnn_cpu_mapping():
    result_dir = (pilot.ROOT / "results/tachyon-qnn-investigation-20261024-v1/"
                  "lfm350-pilot/latest-pair/container-pilot")
    server_result = pilot.read_json(
        result_dir / "server-logs/openai_server_tachyon_lfm350_pilot/server_result.json"
    )
    server_log = (result_dir / "server.log").read_text(encoding="utf-8")
    proof = pilot.htp_execution_proof(server_result, server_log)
    assert proof["status"] == "passed"
    assert proof["provider_profiles"]["decode"]["cpu_execution_count"] == 0
    assert proof["qnn_cpu_backend_mapped"] is False
    changed = copy.deepcopy(server_result)
    changed["runtime_fingerprint"]["qnn"]["mapped_libraries"].append({
        "name": "libQnnCpu.so", "exists": True, "sha256": "0" * 64,
    })
    with pytest.raises(pilot.PilotError, match="CPU backend"):
        pilot.htp_execution_proof(changed, server_log)


def test_process_ownership_guard_and_bounded_short_request_plan():
    receipt = {"pid": 10, "start_ticks": 20, "argv": ["server", "--port", "18085"]}
    assert pilot.process_matches(receipt, {**receipt, "alive": True}) is True
    mismatch = {**receipt, "alive": True, "start_ticks": 21}
    with pytest.raises(pilot.PilotError, match="start_ticks differs"):
        pilot.process_matches(receipt, mismatch)
    assert pilot.process_matches(receipt, {"alive": False}) is False

    cases = pilot.request_cases()
    assert [item["id"] for item in cases] == [
        "sentiment-positive", "short-summary", "route-security"
    ]
    assert all("expected_contains" in item for item in cases)


def test_latest_pair_uses_disjoint_mutable_remote_state():
    baseline = pilot.remote_layout("baseline")
    latest = pilot.remote_layout("latest-pair")
    assert latest["source"] == baseline["source"]
    assert latest["assets"] == baseline["assets"]
    assert latest["state"] != baseline["state"]
    assert latest["output"] != baseline["output"]
    assert latest["results"] != baseline["results"]
    assert "/latest-pair" in latest["results"]


def test_flat_restored_cpu_receipt_can_be_adopted_and_prior_attempt_is_archived(tmp_path):
    restored = {
        "schema_version": 1,
        "kind": "tachyon_qnn_lfm350_pilot_cpu_restore",
        "pid": 44,
        "start_ticks": 55,
        "alias": "tachyon-lfm25-26b",
        "argv": [pilot.REMOTE_ROOT + "/runtime/llama-cpp-cpu/bin/llama-server"],
        "launch_argv": ["taskset", "-c", "4-7", "server"],
    }
    receipt = tmp_path / "restored.json"
    receipt.write_text(json.dumps(restored))
    assert pilot.load_cpu_service(receipt)["pid"] == 44

    study = tmp_path / "study"
    study.mkdir()
    (study / "pilot-status.json").write_text(json.dumps({
        "started_at": "2026-10-24T02:21:32+00:00", "status": "failed",
    }))
    (study / "install-container-command.json").write_text('{"failed":true}\n')
    archived = pilot.archive_prior_attempt(study)
    assert archived is not None
    assert json.loads((archived / "pilot-status.json").read_text())["status"] == "failed"
    assert (archived / "install-container-command.json").is_file()
    events = study / "events.jsonl"
    pilot.append_event(events, "prior_attempt_archived", archive_path=str(archived))
    assert json.loads(events.read_text())["archive_path"] == str(archived)


def test_direct_cli_audit_entrypoint(tmp_path):
    output = tmp_path / "audit.json"
    completed = subprocess.run(
        [sys.executable, str(Path(pilot.__file__)), "audit", "--output", str(output)],
        cwd=pilot.ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, timeout=30, check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(output.read_text())["upstream_commit"] == pilot.UPSTREAM_COMMIT
