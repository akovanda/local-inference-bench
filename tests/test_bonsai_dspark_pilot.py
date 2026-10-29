import hashlib
import json
import signal
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from runtime import bonsai_dspark_pilot as pilot


def write_json(path: Path, value):
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size": len(payload),
    }


def write_jsonl(path: Path, rows):
    payload = b"".join(
        (json.dumps(row, separators=(",", ":")) + "\n").encode() for row in rows
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size": len(payload),
    }


def benchmark_row(
    case_id,
    prompt_sha,
    *,
    thinking="medium",
    cap=65536,
    timeout=pilot.REQUEST_TIMEOUT,
    wall=10.0,
    predicted_n=101,
    predicted_ms=1000.0,
    passed=True,
    draft_n=None,
    draft_accepted=None,
):
    benchmark, specialist = pilot._expected_prompts()
    identity = specialist.get(case_id, benchmark.get(case_id))
    if identity is None:
        raise AssertionError(f"unknown fixture case: {case_id}")
    native = {"predicted_n": predicted_n, "predicted_ms": predicted_ms}
    if draft_n is not None:
        native["draft_n"] = draft_n
    if draft_accepted is not None:
        native["draft_n_accepted"] = draft_accepted
    return {
        "schema_version": 1,
        "run_id": "benchmark-run",
        "call_id": f"call-{case_id}",
        "suite": (
            "quick" if case_id.startswith("quick_")
            else "specialist_code" if case_id.startswith("code_")
            else "hard"
        ),
        "case_id": case_id,
        "case_version": identity["case_version"],
        "case_sha256": identity.get("case_sha256"),
        "request": {
            "model": "bonsai2-27b-ptq1_0-dspark-v1-experimental",
            "context_max_tokens": pilot.CONTEXT,
            "thinking": thinking,
            "chat_template_thinking": "off" if thinking == "none" else "on",
            "temperature": 0.0,
            "seed": 0,
            "output_token_limit": cap,
            "timeout_seconds": timeout,
            "prompt_sha256": prompt_sha,
        },
        "response": {
            "response_text": "fixture visible answer",
            "actual_completion_tokens": 100,
            "done_reason": "stop",
            "stream_complete": True,
            "truncated": False,
            "timing": {"wall_seconds": wall},
            "openai_native_timings": native,
        },
        "evaluation": {"passed": passed},
        "error": None,
    }


def eligibility_stub(*, baseline_tps=50.0, baseline_wall=30.0, kind="hard", case=None):
    case = case or pilot.HARD_IDS[0]
    benchmark, specialist = pilot._expected_prompts()
    prompt = (specialist if kind == "code" else benchmark)[case]["prompt_sha256"]
    binding = pilot.Binding(Path("/tmp/x"), "a" * 64, 1)
    return pilot.Eligibility(
        source=binding,
        document={},
        study=binding,
        ready=binding,
        family=binding,
        cleanup=binding,
        baseline_native_tps=baseline_tps,
        baseline_wall_seconds=baseline_wall,
        quality_kind=kind,
        quality_case_id=case,
        quality_prompt_sha256=prompt,
        boundary_remaining_seconds=40_000.0,
    )


def quality_rows(code_passes, hard_passes):
    benchmark, specialist = pilot._expected_prompts()
    code = []
    for index, case_id in enumerate(pilot.CODE_IDS):
        identity = specialist[case_id]
        passed = index < code_passes
        code.append(
            {
                "case_id": case_id,
                "case_version": identity["case_version"],
                "case_sha256": identity["case_sha256"],
                "prompt_sha256": identity["prompt_sha256"],
                "call_id": f"code-{index}",
                "passed": passed,
                "eos": True,
                "nonempty": True,
                "error_free": True,
                "truncated": False,
                "source_key": "canonical_code",
            }
        )
    hard = []
    for index, case_id in enumerate(pilot.HARD_IDS):
        identity = benchmark[case_id]
        passed = index < hard_passes
        hard.append(
            {
                "case_id": case_id,
                "case_version": identity["case_version"],
                "prompt_sha256": identity["prompt_sha256"],
                "call_id": f"hard-{index}",
                "passed": passed,
                "eos": True,
                "nonempty": True,
                "error_free": True,
                "truncated": False,
                "source_key": "plan_hard",
            }
        )
    return {"quality": {"canonical_code": code, "hard": hard}}


def raw_terminal_row(case_id, *, call_id=None, passed=True):
    benchmark, specialist = pilot._expected_prompts()
    identity = specialist.get(case_id, benchmark.get(case_id))
    if identity is None:
        from runtime import specialist_cases

        case = specialist_cases.get_cases(groups=("prose",), case_ids=(case_id,))[0]
        identity = {
            "case_version": specialist_cases.CASE_VERSION,
            "case_sha256": specialist_cases.case_sha256(case),
            "prompt_sha256": hashlib.sha256(case.prompt.encode()).hexdigest(),
        }
    return {
        "schema_version": 1,
        "run_id": "raw-run",
        "call_id": call_id or f"raw-{case_id}",
        "suite": "specialist_code" if case_id in pilot.CODE_IDS else (
            "specialist_prose" if case_id in pilot.PROSE_IDS else
            "quick" if case_id in pilot.QUICK_IDS else "hard"
        ),
        "case_id": case_id,
        "case_version": identity["case_version"],
        "case_sha256": identity.get("case_sha256"),
        "request": {
            "prompt_sha256": identity["prompt_sha256"],
            "context_max_tokens": pilot.CONTEXT,
        },
        "response": {
            "actual_completion_tokens": 10,
            "done_reason": "stop",
            "stream_complete": True,
            "truncated": False,
            "timing": {"wall_seconds": 1.0},
            "openai_native_timings": {"predicted_n": 10, "predicted_ms": 1000.0},
        },
        "evaluation": {"passed": passed},
        "error": None,
    }


def project_raw(row):
    response = row["response"]
    return {
        "schema_version": row["schema_version"],
        "run_id": row["run_id"],
        "call_id": row["call_id"],
        "suite": row["suite"],
        "case_id": row["case_id"],
        "case_version": row["case_version"],
        "case_sha256": row.get("case_sha256"),
        "request": row["request"],
        "response": {
            "actual_completion_tokens": response["actual_completion_tokens"],
            "done_reason": response["done_reason"],
            "stream_complete": response["stream_complete"],
            "truncated": response["truncated"],
            "wall_seconds": response["timing"]["wall_seconds"],
            "native": response["openai_native_timings"],
        },
        "evaluation_passed": row["evaluation"]["passed"],
        "error_present": row["error"] is not None,
        "visible_chars": None,
    }


class FakeProcess:
    def __init__(self, pid=9001, returncode=None):
        self.pid = pid
        self.returncode = returncode

    def poll(self):
        return self.returncode


class BonsaiDSparkPilotTests(unittest.TestCase):
    def test_candidate_delta_appends_only_declared_flags(self):
        base = {
            "schema_version": 1,
            "model_name": pilot.PTQ_MODEL_NAME,
            "args": ["--gpu-layers", "all", "--fit", "off", "--jinja"],
            "context": pilot.CONTEXT,
        }
        candidate = pilot.candidate_config(base)
        self.assertEqual(candidate["args"], [*base["args"], *pilot.CANDIDATE_ARGS])
        self.assertEqual(candidate["args"][-8:], list(pilot.CANDIDATE_ARGS))
        self.assertNotIn(pilot.FORBIDDEN_ENVIRONMENT_OPTION, json.dumps(candidate))
        self.assertEqual(base["model_name"], pilot.PTQ_MODEL_NAME)

    def test_one_setup_step_and_exact_four_request_budget(self):
        plan = pilot.smoke_plan()
        self.assertEqual(len(plan["steps"]), 1)
        self.assertEqual(plan["steps"][0]["cases"], ["quick_json_extract"])
        self.assertEqual(plan["steps"][0]["output_tokens"], 256)
        self.assertEqual(plan["steps"][0]["thinking"], "none")
        budget = pilot.generation_request_budget()
        self.assertEqual(budget, {
            "model_session_setup": 1, "matched_quick": 2, "quality_pilot_max": 1,
        })
        self.assertEqual(sum(budget.values()), 4)

    def test_gpu_window_accepts_exact_model_session_child_command(self):
        from runtime import gpu_window

        paths = pilot.pilot_paths("fixture-only", results=Path("/tmp/fixture-results"))
        command = pilot.outer_command("fixture-only", paths)
        child = command[command.index("--") + 1:]
        gpu_window.validate_trial(child)
        self.assertEqual(child[0], str(pilot.ROOT / "runtime/model_session.py"))
        self.assertEqual(child[child.index("--control-file") + 1], str(paths.control))
        self.assertNotIn(pilot.FORBIDDEN_ENVIRONMENT_OPTION, command)

    def test_native_metric_uses_predicted_n_minus_one_symmetrically(self):
        rows = [
            {"native_predicted_n": 101, "native_predicted_seconds": 2.0},
            {"native_predicted_n": 51, "native_predicted_seconds": 1.0},
        ]
        self.assertEqual(pilot.native_pooled_tps(rows), 50.0)
        with self.assertRaisesRegex(pilot.PilotError, "predicted_n"):
            pilot.native_pooled_tps([
                {"native_predicted_n": 1, "native_predicted_seconds": 1.0}
            ])

    def test_quality_selection_is_one_first_passing_frozen_case(self):
        first = quality_rows(2, 4)
        raw = {
            "canonical_code": pilot.RawSource(
                pilot.Binding(Path("code"), "a" * 64, 1),
                tuple(project_raw(raw_terminal_row(row["case_id"], call_id=row["call_id"], passed=row["passed"]))
                      for row in first["quality"]["canonical_code"]),
            ),
            "plan_hard": pilot.RawSource(
                pilot.Binding(Path("hard"), "b" * 64, 1),
                tuple(project_raw(raw_terminal_row(row["case_id"], call_id=row["call_id"], passed=row["passed"]))
                      for row in first["quality"]["hard"]),
            ),
        }
        kind, case_id, _ = pilot._validate_quality(first, raw)
        self.assertEqual((kind, case_id), ("code", pilot.CODE_IDS[0]))
        second = quality_rows(1, 3)
        raw = {
            "canonical_code": pilot.RawSource(
                pilot.Binding(Path("code"), "a" * 64, 1),
                tuple(project_raw(raw_terminal_row(row["case_id"], call_id=row["call_id"], passed=row["passed"]))
                      for row in second["quality"]["canonical_code"]),
            ),
            "plan_hard": pilot.RawSource(
                pilot.Binding(Path("hard"), "b" * 64, 1),
                tuple(project_raw(raw_terminal_row(row["case_id"], call_id=row["call_id"], passed=row["passed"]))
                      for row in second["quality"]["hard"]),
            ),
        }
        kind, case_id, _ = pilot._validate_quality(second, raw)
        self.assertEqual((kind, case_id), ("hard", pilot.HARD_IDS[0]))
        third = quality_rows(1, 2)
        raw = {
            "canonical_code": pilot.RawSource(
                pilot.Binding(Path("code"), "a" * 64, 1),
                tuple(project_raw(raw_terminal_row(row["case_id"], call_id=row["call_id"], passed=row["passed"]))
                      for row in third["quality"]["canonical_code"]),
            ),
            "plan_hard": pilot.RawSource(
                pilot.Binding(Path("hard"), "b" * 64, 1),
                tuple(project_raw(raw_terminal_row(row["case_id"], call_id=row["call_id"], passed=row["passed"]))
                      for row in third["quality"]["hard"]),
            ),
        }
        with self.assertRaisesRegex(pilot.PilotError, "below both"):
            pilot._validate_quality(third, raw)

    def test_binding_rejects_hash_size_and_identity_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bound.json"
            binding = write_json(path, {"value": 1})
            self.assertEqual(pilot.validate_binding(binding, "fixture").sha256, binding["sha256"])
            bad_hash = dict(binding, sha256="0" * 64)
            with self.assertRaisesRegex(pilot.PilotError, "SHA-256"):
                pilot.validate_binding(bad_hash, "fixture")
            bad_size = dict(binding, size=binding["size"] + 1)
            with self.assertRaisesRegex(pilot.PilotError, "size"):
                pilot.validate_binding(bad_size, "fixture")
            path.write_text("changed", encoding="utf-8")
            with self.assertRaises(pilot.PilotError):
                pilot.validate_binding(binding, "fixture")

    def test_study_wrong_main_identity_fails_closed(self):
        helpers = {
            str(path): digest
            for path, digest in pilot.SOURCE_PINS.items()
            if path.name in {
                "model_session.py", "gpu_window.py", "specialist_bench.py",
                "sampler_recipe_probe.py",
            }
        }
        stages = [
            {"returncode": 0, "receipt": {"status": "complete"}}
            for _ in range(3)
        ]
        study = {
            "schema_version": 1,
            "status": "complete",
            "problem": None,
            "tag": pilot.PTQ_TAG,
            "runtime_backend": "llama",
            "source_config": {"path": str(pilot.BASE_CONFIG), "sha256": pilot.BASE_CONFIG_SHA256},
            "session": {
                "run_id": pilot.PTQ_MODEL_SESSION_RUN_ID,
                "model_name": pilot.PTQ_MODEL_NAME,
                "context": pilot.CONTEXT,
            },
            "cleanup": {
                "outer_exited": True,
                "orchestrator_sigkill_sent": False,
                "model_session_cleanup": {"stopped": True, "kill_sent": False},
            },
            "restoration": {
                "original_running": True, "original_paused": False,
                "owned_loopback_port": 18081, "owned_loopback_port_clear": True,
            },
            "helper_sources": helpers,
            "sampler_plan": {"matched": True, "stages": stages},
        }
        bound = pilot.Binding(Path(f"/tmp/{pilot.PTQ_TAG}-study-trial.json"), "a" * 64, 1)
        pilot._validate_study(study, bound)
        study["session"]["run_id"] = "foreign"
        with self.assertRaisesRegex(pilot.PilotError, "terminal lifecycle"):
            pilot._validate_study(study, bound)

    def test_raw_source_inventory_is_exactly_21_and_hash_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = []
            artifacts = {}
            for key, (artifact_key, count) in pilot.RAW_SOURCE_COUNTS.items():
                path = root / f"{key}.jsonl"
                case_ids = {
                    "plan_quick": pilot.QUICK_IDS,
                    "plan_hard": pilot.HARD_IDS,
                    "canonical_code": pilot.CODE_IDS,
                    "canonical_prose": pilot.PROSE_IDS,
                    "publisher_code": pilot.CODE_IDS,
                    "shared_t0_code": pilot.CODE_IDS,
                    "publisher_prose": pilot.PROSE_IDS,
                }[key]
                binding = write_jsonl(
                    path,
                    [raw_terminal_row(case_id, call_id=f"{key}-{index}")
                     for index, case_id in enumerate(case_ids)],
                )
                rows.append({
                    "key": key,
                    "artifact_key": artifact_key,
                    "terminal_requests": count,
                    **binding,
                })
                artifacts[artifact_key] = {
                    "path": binding["path"], "size": binding["size"],
                    "exists": True, "regular_file": True, "directory": False,
                }
            validated = pilot._validate_raw_sources(
                {"raw_request_sources": rows}, {"artifacts": artifacts}
            )
            self.assertEqual(len(validated), 7)
            self.assertEqual(sum(item[1] for item in pilot.RAW_SOURCE_COUNTS.values()), 21)
            rows[0]["sha256"] = "0" * 64
            with self.assertRaisesRegex(pilot.PilotError, "SHA-256"):
                pilot._validate_raw_sources(
                    {"raw_request_sources": rows}, {"artifacts": artifacts}
                )

    def test_boundary_requires_unreleased_matching_ready_and_time_reserve(self):
        with tempfile.TemporaryDirectory() as directory:
            results = Path(directory) / "results"
            ready = (
                results / "specialists-final72k64k-boundaries"
                / f"{pilot.RUN_PREFIX}-{pilot.MODEL_KEY}-ready.json"
            )
            study_path = results / f"{pilot.PTQ_TAG}-study-trial.json"
            study_path.parent.mkdir(parents=True)
            study_path.write_text("{}\n", encoding="utf-8")
            study = pilot.Binding(
                study_path.resolve(), hashlib.sha256(b"{}\n").hexdigest(), 3
            )
            published = "2026-10-20T20:00:00Z"
            binding = write_json(ready, {
                "schema_version": 1,
                "kind": "model_boundary_ready",
                "run_prefix": pilot.RUN_PREFIX,
                "model_key": pilot.MODEL_KEY,
                "published_at": published,
                "study": {"path": str(study.path), "sha256": study.sha256},
            })
            release = ready.with_name(ready.name.replace("-ready.json", "-release.json"))
            value = {
                "wait_seconds": 86400,
                "ready": binding,
                "release_path": str(release),
            }
            with mock.patch.object(pilot, "RESULTS", results):
                _, remaining = pilot._validate_boundary(
                    value, study, datetime(2026, 10, 20, 21, tzinfo=timezone.utc)
                )
                self.assertGreater(remaining, pilot.MIN_BOUNDARY_REMAINING_SECONDS)
                release.write_text("{}\n", encoding="utf-8")
                with self.assertRaisesRegex(pilot.PilotError, "RELEASE"):
                    pilot._validate_boundary(
                        value, study, datetime(2026, 10, 20, 21, tzinfo=timezone.utc)
                    )

    def test_family_allows_evidenced_no_optional_followup(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            study_value = write_json(root / "study.json", {"status": "complete"})
            study = pilot.Binding(
                Path(study_value["path"]), study_value["sha256"], study_value["size"]
            )

            def family_binding(decision):
                evidence = write_json(root / f"evidence-{decision}.json", {
                    "schema_version": 1,
                    "kind": "bonsai2_v3_post_primary_disposition_evidence",
                    "status": "complete",
                    "created_at": "2026-10-20T21:00:00Z",
                    "main_ptq_tag": pilot.PTQ_TAG,
                    "main_model_session_run_id": pilot.PTQ_MODEL_SESSION_RUN_ID,
                    "completed_study": study_value,
                    "decision": decision,
                    "reason": "No optional V3 followup was selected after primary review.",
                    "diagnostic_subruns": [],
                })
                return write_json(root / f"family-{decision}.json", {
                    "schema_version": 1,
                    "kind": "bonsai2_ptq_v3_family_disposition",
                    "status": "complete",
                    "main_ptq_tag": pilot.PTQ_TAG,
                    "main_model_session_run_id": pilot.PTQ_MODEL_SESSION_RUN_ID,
                    "all_v3_family_work_complete": True,
                    "in_flight_requests": 0,
                    "gemma_disposition": "deferred",
                    "flash_disposition": "queued_after_ptq",
                    "matching_20b_disposition": "last",
                    "v3_disposition_evidence": evidence,
                    "diagnostic_subruns": [],
                })

            pilot._validate_family(
                {"receipt": family_binding("no_optional_followup_selected")}, study
            )
            with self.assertRaisesRegex(pilot.PilotError, "incomplete or inconsistent"):
                pilot._validate_family(
                    {"receipt": family_binding("diagnostics_selected")}, study
                )

    def test_smoke_accepts_zero_accepted_but_rejects_absent_counter(self):
        benchmark, _ = pilot._expected_prompts()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            smoke = root / "smoke.jsonl"
            events = root / "events.jsonl"
            log = root / "server.log"
            session = SimpleNamespace(run_id="owned")
            row = benchmark_row(
                "quick_json_extract",
                benchmark["quick_json_extract"]["prompt_sha256"],
                thinking="none",
                cap=256,
                timeout=180.0,
                draft_n=10,
                draft_accepted=0,
            )
            write_jsonl(smoke, [row])
            write_jsonl(events, [
                {"event": "server_started", "run_id": "owned", "pid": 4321},
                {
                    "event": "step_complete", "run_id": "owned", "id": pilot.SMOKE_STEP_ID,
                    "result": str(smoke), "returncode": 0, "passed": True,
                },
            ])
            log.write_text(
                f"loading {pilot.DS_PARK_MODEL.name}\nCUDA buffer residency established\n",
                encoding="utf-8",
            )
            result = pilot.validate_smoke(
                smoke, events, session,
                gpu_processes=lambda: [{"pid": 4321, "used_mib": 11000}],
                server_log=log,
            )
            self.assertEqual(result["draft_accepted"], 0)
            del row["response"]["openai_native_timings"]["draft_n_accepted"]
            write_jsonl(smoke, [row])
            with self.assertRaisesRegex(pilot.PilotError, "distinct draft"):
                pilot.validate_smoke(
                    smoke, events, session,
                    gpu_processes=lambda: [{"pid": 4321, "used_mib": 11000}],
                    server_log=log,
                )

    def test_smoke_semantic_failure_is_a_gate_stop(self):
        benchmark, _ = pilot._expected_prompts()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            smoke = root / "smoke.jsonl"
            events = root / "events.jsonl"
            log = root / "server.log"
            session = SimpleNamespace(run_id="owned")
            row = benchmark_row(
                "quick_json_extract",
                benchmark["quick_json_extract"]["prompt_sha256"],
                thinking="none", cap=256, timeout=180.0, passed=False,
                draft_n=10, draft_accepted=1,
            )
            write_jsonl(smoke, [row])
            write_jsonl(events, [
                {"event": "server_started", "run_id": "owned", "pid": 4321},
                {
                    "event": "step_complete", "run_id": "owned", "id": pilot.SMOKE_STEP_ID,
                    "result": str(smoke), "returncode": 1, "passed": False,
                },
            ])
            log.write_text(
                f"loading {pilot.DS_PARK_MODEL.name}\nCUDA buffer residency established\n",
                encoding="utf-8",
            )
            with self.assertRaises(pilot.PilotGateStopped) as stopped:
                pilot.validate_smoke(
                    smoke, events, session,
                    gpu_processes=lambda: [{"pid": 4321, "used_mib": 11000}],
                    server_log=log,
                )
            self.assertEqual(stopped.exception.stage, "smoke")
            self.assertEqual(stopped.exception.evidence["outcome_class"], "gate_failure")
            self.assertEqual(stopped.exception.evidence["step_returncode"], 1)

    def test_clean_semantic_failure_session_is_not_cleanup_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = pilot.pilot_paths("semantic-cleanup", results=Path(directory))
            paths.config.write_text("{}\n", encoding="utf-8")
            paths.smoke_plan.write_text("{}\n", encoding="utf-8")
            session = SimpleNamespace(run_id="owned")
            write_json(paths.session_report, {
                "schema_version": 1,
                "session_id": "owned",
                "status": "completed_with_step_failures",
                "problem": None,
                "source_config_sha256": pilot.sha256_file(paths.config),
                "plan": {"sha256": pilot.sha256_file(paths.smoke_plan)},
                "server": {"cleanup": {"stopped": True, "kill_sent": False}},
            })
            write_jsonl(paths.events, [
                {"event": "server_stopped", "run_id": "owned"},
                {"event": "session_complete", "run_id": "owned"},
            ])
            cleanup = {"outer_exited": True, "orchestrator_sigkill_sent": False}
            with mock.patch.object(pilot, "live_boundary_preflight", return_value={"clean": True}):
                evidence = pilot._final_cleanup_evidence(paths, session, cleanup)
            self.assertEqual(evidence["session_status"], "completed_with_step_failures")
            self.assertTrue(evidence["semantic_step_failures"])

    def test_live_boundary_does_not_require_docker_healthcheck(self):
        from runtime import gpu_window

        with mock.patch.object(
            gpu_window, "original_state", return_value={"Running": True, "Paused": False}
        ), mock.patch.object(
            gpu_window, "_api_request", return_value={"version": "0.20.0"}
        ), mock.patch.object(
            pilot, "_port_clear", return_value=True
        ), mock.patch.object(
            pilot, "_process_start_ticks", return_value=None
        ):
            evidence = pilot.live_boundary_preflight()
        self.assertIsNone(evidence["original_health_status"])
        self.assertTrue(evidence["original_api_healthy"])

        with mock.patch.object(
            gpu_window, "original_state", return_value={"Running": True, "Paused": False}
        ), mock.patch.object(
            gpu_window, "_api_request", side_effect=OSError("unhealthy")
        ), mock.patch.object(
            pilot, "_port_clear", return_value=True
        ), mock.patch.object(
            pilot, "_process_start_ticks", return_value=None
        ):
            with self.assertRaisesRegex(pilot.PilotError, "API health probe failed"):
                pilot.live_boundary_preflight()

    def test_quick_requires_both_clean_and_both_speed_gates(self):
        benchmark, _ = pilot._expected_prompts()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "quick.jsonl"
            rows = [
                benchmark_row(
                    case_id, benchmark[case_id]["prompt_sha256"],
                    wall=9.0, predicted_n=601, predicted_ms=10_000.0,
                )
                for case_id in pilot.QUICK_IDS
            ]
            write_jsonl(path, rows)
            result = pilot.validate_quick_candidate(
                path, eligibility_stub(baseline_tps=50.0, baseline_wall=30.0)
            )
            self.assertEqual(result["pooled_native_tps"], 60.0)
            rows[1]["evaluation"]["passed"] = False
            write_jsonl(path, rows)
            with self.assertRaisesRegex(pilot.PilotGateStopped, "semantic or output-cap") as gate:
                pilot.validate_quick_candidate(
                    path, eligibility_stub(baseline_tps=50.0, baseline_wall=30.0)
                )
            self.assertEqual(gate.exception.evidence["observed_case_count"], 2)
            rows[1]["evaluation"]["passed"] = True
            write_jsonl(path, rows)
            with self.assertRaisesRegex(pilot.PilotGateStopped, "continuation gates") as gate:
                pilot.validate_quick_candidate(
                    path, eligibility_stub(baseline_tps=70.0, baseline_wall=30.0)
                )
            self.assertEqual(gate.exception.evidence["pooled_native_tps"], 60.0)
            self.assertFalse(gate.exception.evidence["native_tps_gate_passed"])
            self.assertTrue(gate.exception.evidence["combined_wall_gate_passed"])

    def test_slower_clean_quick_stops_before_quality_with_metrics(self):
        benchmark, _ = pilot._expected_prompts()
        with tempfile.TemporaryDirectory() as directory:
            paths = pilot.pilot_paths("slower-quick", results=Path(directory))
            write_jsonl(paths.smoke, [
                benchmark_row(
                    "quick_json_extract",
                    benchmark["quick_json_extract"]["prompt_sha256"],
                    thinking="none", cap=256, timeout=180.0,
                    draft_n=10, draft_accepted=2,
                )
            ])
            write_jsonl(paths.quick, [
                benchmark_row(
                    case_id, benchmark[case_id]["prompt_sha256"],
                    wall=9.0, predicted_n=601, predicted_ms=10_000.0,
                )
                for case_id in pilot.QUICK_IDS
            ])
            eligibility = eligibility_stub(baseline_tps=70.0, baseline_wall=30.0)
            prepared = {
                "config": pilot.Binding(paths.config, "a" * 64, 1),
                "smoke_plan": pilot.Binding(paths.smoke_plan, "b" * 64, 1),
            }
            session = SimpleNamespace(
                run_id="owned", model_name="candidate", context=pilot.CONTEXT,
                initial_load_seconds=1.0,
            )
            with mock.patch.object(pilot, "live_boundary_preflight", return_value={}), mock.patch.object(
                pilot, "_assert_unused"
            ), mock.patch.object(pilot, "verify_dspark_model"), mock.patch.object(
                pilot, "_validate_boundary", return_value=(eligibility.ready, 40_000.0)
            ), mock.patch.object(
                pilot, "prepare_candidate", return_value=prepared
            ), mock.patch.object(
                pilot.subprocess, "Popen", return_value=FakeProcess()
            ), mock.patch.object(
                pilot, "_wait_holding", return_value=session
            ), mock.patch.object(
                pilot, "validate_smoke", return_value={"passed": True}
            ), mock.patch.object(
                pilot, "quick_command", return_value=["quick-helper"]
            ), mock.patch.object(
                pilot, "_run_owned", return_value={
                    "argv": ["quick-helper"], "returncode": 0,
                    "elapsed_seconds": 1.0, "cleanup": {"stopped": True},
                }
            ), mock.patch.object(
                pilot, "quality_command"
            ) as quality_command, mock.patch.object(
                pilot, "_session_cleanup", return_value={
                    "outer_exited": True, "orchestrator_sigkill_sent": False,
                }
            ), mock.patch.object(
                pilot, "_final_cleanup_evidence", return_value={"restoration": {}}
            ), mock.patch.object(pilot.signal, "signal"):
                receipt, code = pilot.execute("slower-quick", eligibility, paths=paths)
            self.assertEqual(code, 0)
            self.assertEqual(receipt["status"], "stopped_at_gate")
            self.assertEqual(receipt["gate_outcome"]["stage"], "quick")
            self.assertEqual(receipt["quick"]["pooled_native_tps"], 60.0)
            self.assertFalse(receipt["quick"]["native_tps_gate_passed"])
            self.assertIsNone(receipt["problem"])
            self.assertIsNone(receipt["quality"])
            quality_command.assert_not_called()

    def test_request_accounting_uses_rows_and_preserves_unknown_start(self):
        benchmark, _ = pilot._expected_prompts()
        with tempfile.TemporaryDirectory() as directory:
            paths = pilot.pilot_paths("accounting", results=Path(directory))
            row = benchmark_row(
                pilot.QUICK_IDS[0], benchmark[pilot.QUICK_IDS[0]]["prompt_sha256"],
                passed=False,
            )
            write_jsonl(paths.quick, [row])
            with self.assertRaises(pilot.PilotGateStopped) as stopped:
                pilot.validate_quick_candidate(
                    paths.quick, eligibility_stub(baseline_tps=50.0, baseline_wall=30.0)
                )
            self.assertEqual(stopped.exception.evidence["observed_case_count"], 1)
            accounting = pilot.generation_request_accounting(
                paths,
                launch_attempted={"smoke": False, "matched_quick": True, "quality_pilot": False},
                stage_completed={"smoke": False, "matched_quick": True, "quality_pilot": False},
            )
            quick = accounting["stages"]["matched_quick"]
            self.assertEqual(quick["allocated_request_slots"], 2)
            self.assertEqual(quick["observed_result_rows"], 1)
            self.assertEqual(quick["exact_attempts"], 1)

            paths.quick.unlink()
            accounting = pilot.generation_request_accounting(
                paths,
                launch_attempted={"smoke": True, "matched_quick": False, "quality_pilot": False},
                stage_completed={"smoke": False, "matched_quick": False, "quality_pilot": False},
            )
            smoke = accounting["stages"]["smoke"]
            self.assertEqual(smoke["observed_result_rows"], 0)
            self.assertIsNone(smoke["exact_attempts"])
            self.assertEqual(smoke["attempts_upper_bound"], 1)
            self.assertIsNone(accounting["exact_attempts"])

    def test_quality_semantic_failure_is_a_gate_stop(self):
        benchmark, _ = pilot._expected_prompts()
        case_id = pilot.HARD_IDS[0]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "quality.jsonl"
            write_jsonl(path, [
                benchmark_row(case_id, benchmark[case_id]["prompt_sha256"], passed=False)
            ])
            with self.assertRaises(pilot.PilotGateStopped) as stopped:
                pilot.validate_quality_candidate(
                    path, eligibility_stub(kind="hard", case=case_id)
                )
            self.assertEqual(stopped.exception.stage, "quality")
            self.assertEqual(stopped.exception.evidence["outcome_class"], "gate_failure")

    def test_owned_helper_signal_terminates_only_owned_group(self):
        helper = FakeProcess(pid=8123)
        outer = FakeProcess(pid=7000)
        latch = pilot.SignalLatch(received=signal.SIGTERM)

        def terminate(pid, signum):
            self.assertEqual((pid, signum), (helper.pid, signal.SIGTERM))
            helper.returncode = -signal.SIGTERM

        with mock.patch.object(pilot.subprocess, "Popen", return_value=helper), mock.patch.object(
            pilot.os, "killpg", side_effect=terminate
        ) as killpg:
            with self.assertRaises(pilot.PilotInterrupted) as raised:
                pilot._run_owned(["fixture"], outer, 10**12, latch)
        self.assertEqual(killpg.call_count, 1)
        cleanup = getattr(raised.exception, "owned_helper_cleanup")
        self.assertTrue(cleanup["term_sent"])
        self.assertTrue(cleanup["stopped"])
        self.assertFalse(cleanup["kill_sent"])

    def test_failed_preboundary_validation_creates_nothing_and_never_executes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tag = "preboundary-reject"
            paths = pilot.pilot_paths(tag, results=root)
            with mock.patch.object(
                pilot, "validate_eligibility", side_effect=pilot.PilotError("full21 absent")
            ), mock.patch.object(pilot, "execute") as execute:
                code = pilot.main([
                    "--eligibility", str(root / "ineligible.json"),
                    "--tag", tag,
                    "--execute",
                ])
            self.assertEqual(code, 2)
            execute.assert_not_called()
            self.assertTrue(all(not path.exists() for path in paths.__dict__.values()))

    def test_quality_commands_contain_only_one_case(self):
        session = SimpleNamespace(model_name="candidate")
        paths = pilot.pilot_paths("test", results=Path("/tmp/results"))
        hard = pilot.quality_command(
            "test", eligibility_stub(kind="hard", case=pilot.HARD_IDS[1]), session, paths
        )
        self.assertEqual(hard[hard.index("--cases") + 1], pilot.HARD_IDS[1])
        code = pilot.quality_command(
            "test",
            eligibility_stub(kind="code", case=pilot.CODE_IDS[1]),
            session,
            paths,
        )
        self.assertEqual(code[code.index("--cases") + 1], pilot.CODE_IDS[1])

    def test_code_quality_hash_binds_visible_answer_without_reasoning(self):
        _, specialist = pilot._expected_prompts()
        case_id = pilot.CODE_IDS[0]
        identity = specialist[case_id]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "quality.jsonl"
            row = benchmark_row(case_id, identity["prompt_sha256"])
            visible_dir = path.with_name(path.name + ".texts")
            visible_dir.mkdir()
            visible_path = visible_dir / f"{row['call_id']}.txt"
            payload = b"fixture visible answer"
            visible_path.write_bytes(payload)
            row["visible_text"] = {
                "path": str(visible_path.resolve()),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "chars": len(payload.decode()),
                "bytes": len(payload),
                "reasoning_excluded": True,
            }
            write_jsonl(path, [row])
            result = pilot.validate_quality_candidate(
                path, eligibility_stub(kind="code", case=case_id)
            )
            self.assertTrue(result["visible_answer"]["reasoning_excluded"])
            self.assertEqual(result["visible_answer"]["sha256"], hashlib.sha256(payload).hexdigest())
            row["visible_text"]["reasoning_excluded"] = False
            write_jsonl(path, [row])
            with self.assertRaisesRegex(pilot.PilotError, "visible-answer binding"):
                pilot.validate_quality_candidate(
                    path, eligibility_stub(kind="code", case=case_id)
                )


if __name__ == "__main__":
    unittest.main()
