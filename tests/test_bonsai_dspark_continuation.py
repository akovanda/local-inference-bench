import hashlib
import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from runtime import bonsai_dspark_continuation as continuation
from runtime import bonsai_dspark_pilot as pilot


def write_json(path: Path, value):
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(payload).hexdigest(),
            "size": len(payload)}


def write_jsonl(path: Path, rows):
    payload = b"".join((json.dumps(row, separators=(",", ":")) + "\n").encode()
                       for row in rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(payload).hexdigest(),
            "size": len(payload)}


def quick_row(case_id, *, passed=True, wall=10.0, predicted_n=101,
              predicted_ms=1000.0, draft_n=20, draft_accepted=10):
    benchmark, _ = pilot._expected_prompts()
    identity = benchmark[case_id]
    return {
        "schema_version": 1,
        "run_id": "fixture-benchmark",
        "call_id": f"call-{case_id}",
        "suite": "quick",
        "case_id": case_id,
        "case_version": identity["case_version"],
        "case_sha256": identity.get("case_sha256"),
        "request": {
            "model": "bonsai2-27b-ptq1_0-dspark-v1-experimental",
            "context_max_tokens": pilot.CONTEXT,
            "thinking": "medium",
            "chat_template_thinking": "on",
            "temperature": 0,
            "seed": 0,
            "output_token_limit": 65_536,
            "timeout_seconds": pilot.REQUEST_TIMEOUT,
            "prompt_sha256": identity["prompt_sha256"],
        },
        "response": {
            "actual_completion_tokens": 100,
            "done_reason": "stop",
            "stream_complete": True,
            "truncated": False,
            "timing": {"wall_seconds": wall},
            "openai_native_timings": {
                "predicted_n": predicted_n,
                "predicted_ms": predicted_ms,
                "draft_n": draft_n,
                "draft_n_accepted": draft_accepted,
            },
        },
        "evaluation": {"passed": passed},
        "error": None,
    }


def eligibility_stub(*, baseline_tps=50.0, baseline_wall=30.0):
    _, specialist = pilot._expected_prompts()
    binding = pilot.Binding(Path("/tmp/fixture"), "a" * 64, 1)
    return pilot.Eligibility(
        source=binding, document={}, study=binding, ready=binding, family=binding,
        cleanup=binding, baseline_native_tps=baseline_tps,
        baseline_wall_seconds=baseline_wall, quality_kind="code",
        quality_case_id="code_fresh_batch_module",
        quality_prompt_sha256=specialist["code_fresh_batch_module"]["prompt_sha256"],
        boundary_remaining_seconds=40_000,
    )


def input_stub(acceptance: Path) -> continuation.ContinuationInputs:
    binding = pilot.Binding(acceptance.resolve(), "a" * 64, 1)
    return continuation.ContinuationInputs(
        acceptance=binding, adjudication=binding, plan=binding,
        eligibility=eligibility_stub(), original_config=binding,
        deadline=continuation.ORIGINAL_DEADLINE,
    )


class FakeProcess:
    def __init__(self):
        self.pid = 12345
        self.returncode = None

    def poll(self):
        return self.returncode


class BonsaiDSparkContinuationTests(unittest.TestCase):
    def test_setup_is_exact_quick2_and_has_no_smoke(self):
        plan = continuation.EXPECTED_SETUP_PLAN
        self.assertEqual(len(plan["steps"]), 1)
        step = plan["steps"][0]
        self.assertEqual(step["id"], "matched_quick_medium_65536")
        self.assertEqual(step["cases"], ["quick_generation", "quick_json_extract"])
        self.assertEqual(step["thinking"], "medium")
        self.assertEqual(step["output_tokens"], 65_536)
        self.assertNotIn("dspark_smoke", json.dumps(plan))
        self.assertEqual(continuation.ALREADY_SPENT + 2 + 1, continuation.TOTAL_MAX)

    def test_outer_command_uses_remaining_window_and_no_extra_helper(self):
        paths = continuation.continuation_paths(results=Path("/tmp/continuation-fixture"))
        command = continuation.outer_command(paths, 1234)
        self.assertEqual(command[command.index("--max-seconds") + 1], "1234")
        child = command[command.index("--") + 1:]
        self.assertEqual(child[0], str(continuation.ROOT / "runtime/model_session.py"))
        self.assertEqual(child[child.index("--plan") + 1], str(paths.setup_plan))
        self.assertEqual(child[child.index("--control-file") + 1], str(paths.control))
        self.assertNotIn("smoke", " ".join(command))

    def test_prepare_copies_config_byte_identically_and_materializes_only_nested_plan(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "original.json"
            raw = b'{"exact":"bytes"}\n'
            source.write_bytes(raw)
            paths = continuation.continuation_paths(results=root / "results")
            document = {"setup_plan": continuation.EXPECTED_SETUP_PLAN, "wrapper": "excluded"}
            with mock.patch.object(continuation, "ORIGINAL_CONFIG", source), \
                    mock.patch.object(continuation, "ORIGINAL_CONFIG_SIZE", len(raw)), \
                    mock.patch.object(continuation, "ORIGINAL_CONFIG_SHA256",
                                      hashlib.sha256(raw).hexdigest()):
                prepared = continuation.prepare_runtime(paths, document)
            self.assertEqual(paths.config.read_bytes(), raw)
            self.assertEqual(json.loads(paths.setup_plan.read_text()),
                             continuation.EXPECTED_SETUP_PLAN)
            self.assertNotIn("wrapper", paths.setup_plan.read_text())
            self.assertFalse(paths.control.exists())
            self.assertEqual(prepared["config"].sha256, hashlib.sha256(raw).hexdigest())

    def _quick_fixture(self, root, rows):
        paths = continuation.continuation_paths(results=Path(root))
        write_jsonl(paths.quick, rows)
        session = SimpleNamespace(run_id="owned", model_name="candidate",
                                  context=pilot.CONTEXT, initial_load_seconds=1.0)
        write_jsonl(paths.events, [
            {"event": "server_started", "run_id": "owned", "pid": 4321},
            {"event": "step_complete", "run_id": "owned",
             "id": continuation.SETUP_STEP_ID, "result": str(paths.quick),
             "returncode": 0 if all(row["evaluation"]["passed"] for row in rows) else 1,
             "passed": all(row["evaluation"]["passed"] for row in rows)},
        ])
        paths.server_log.write_text(
            f"loading {pilot.DS_PARK_MODEL.name}\n"
            "using dflash implementation\n"
            "speculative decoding enabled: draft-dspark\n",
            encoding="utf-8",
        )
        return paths, session

    def test_clean_but_slower_quick_retains_comparisons_and_stops_quality(self):
        with tempfile.TemporaryDirectory() as directory:
            rows = [quick_row(case, wall=10, predicted_n=51, predicted_ms=1000)
                    for case in continuation.QUICK_IDS]
            paths, session = self._quick_fixture(directory, rows)
            eligibility = eligibility_stub(baseline_tps=70, baseline_wall=30)
            with self.assertRaises(pilot.PilotGateStopped) as stopped:
                continuation.validate_quick(
                    paths, eligibility, session,
                    lambda: [{"pid": 4321, "used_mib": 12000}],
                )
            evidence = stopped.exception.evidence
            self.assertEqual(stopped.exception.stage, "quick")
            self.assertEqual(evidence["pooled_native_tps"], 50.0)
            self.assertFalse(evidence["native_tps_gate_passed"])
            self.assertTrue(evidence["combined_wall_gate_passed"])
            self.assertEqual(len(evidence["draft_counters"]), 2)
            self.assertEqual(evidence["runtime_engagement"]["inferred_loaded_layer_count"], None)

    def test_first_quick_semantic_failure_is_exactly_one_new_attempt(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = continuation.continuation_paths(results=Path(directory))
            write_jsonl(paths.quick, [quick_row(continuation.QUICK_IDS[0], passed=False)])
            accounting = continuation.request_accounting(
                paths, quick_launched=True, quick_completed=True,
                quality_launched=False, quality_completed=False,
            )
            stage = accounting["continuation_stages"]["matched_quick_setup"]
            self.assertEqual(stage["exact_attempts"], 1)
            self.assertEqual(accounting["exact_total_attempts"], 2)
            self.assertEqual(accounting["possible_total_attempts_upper_bound"], 2)
            self.assertEqual(accounting["duplicate_smoke_requests"], 0)

    def test_interrupted_pre_hold_setup_does_not_claim_zero_requests(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = continuation.continuation_paths(results=Path(directory))
            accounting = continuation.request_accounting(
                paths, quick_launched=True, quick_completed=False,
                quality_launched=False, quality_completed=False,
            )
            stage = accounting["continuation_stages"]["matched_quick_setup"]
            self.assertIsNone(stage["exact_attempts"])
            self.assertEqual(stage["attempts_lower_bound"], 0)
            self.assertEqual(stage["attempts_upper_bound"], 2)
            self.assertIsNone(accounting["exact_total_attempts"])
            self.assertEqual(accounting["possible_total_attempts_upper_bound"], 3)

    def test_completed_setup_without_rows_keeps_attempts_unknown(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = continuation.continuation_paths(results=Path(directory))
            accounting = continuation.request_accounting(
                paths, quick_launched=True, quick_completed=True,
                quality_launched=False, quality_completed=False,
            )
            stage = accounting["continuation_stages"]["matched_quick_setup"]
            self.assertIsNone(stage["exact_attempts"])
            self.assertEqual(stage["attempts_lower_bound"], 0)
            self.assertEqual(stage["attempts_upper_bound"], 2)
            self.assertIsNone(accounting["exact_total_attempts"])
            self.assertEqual(accounting["possible_total_attempts_upper_bound"], 3)

    def test_gate_in_execute_skips_quality_and_keeps_expected_status(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = continuation.continuation_paths(results=root)
            acceptance = root / "acceptance.json"
            acceptance.write_text("{}\n")
            inputs = input_stub(acceptance)
            prepared = {
                "config": pilot.Binding(paths.config, "a" * 64, 1),
                "setup_plan": pilot.Binding(paths.setup_plan, "b" * 64, 1),
            }
            session = SimpleNamespace(run_id="owned", model_name="candidate",
                                      context=pilot.CONTEXT, initial_load_seconds=1.0)
            write_jsonl(paths.quick, [quick_row(continuation.QUICK_IDS[0], passed=False)])
            gate = pilot.PilotGateStopped("quick", "first quick failed",
                                          {"observed_case_count": 1,
                                           "stop_reason": "semantic_or_output_cap_gate"})
            with mock.patch.object(continuation, "remaining_time", return_value={
                "remaining_total_seconds": 20_000, "usable_request_seconds": 18_200,
                "window_max_seconds": 20_000, "restoration_reserve_seconds": 1800,
                "original_deadline": continuation.ORIGINAL_DEADLINE_TEXT,
            }), mock.patch.object(pilot, "live_boundary_preflight", return_value={"clean": True}), \
                    mock.patch.object(continuation, "_assert_unused"), \
                    mock.patch.object(pilot, "verify_dspark_model"), \
                    mock.patch.object(continuation, "validate_inputs", return_value=inputs), \
                    mock.patch.object(continuation, "_load_pinned",
                                      return_value=({"setup_plan": continuation.EXPECTED_SETUP_PLAN}, inputs.plan)), \
                    mock.patch.object(continuation, "prepare_runtime", return_value=prepared), \
                    mock.patch.object(continuation.subprocess, "Popen", return_value=FakeProcess()), \
                    mock.patch.object(pilot, "_wait_holding", return_value=session), \
                    mock.patch.object(continuation, "validate_quick", side_effect=gate), \
                    mock.patch.object(pilot, "quality_command") as quality_command, \
                    mock.patch.object(pilot, "_session_cleanup", return_value={
                        "outer_exited": True, "orchestrator_sigkill_sent": False}), \
                    mock.patch.object(continuation, "retry_cleanup_evidence",
                                      return_value=({"restoration": {}}, {"attempts": 1})), \
                    mock.patch.object(continuation.signal, "signal"):
                receipt, code = continuation.execute(inputs, paths=paths)
            self.assertEqual(code, 0)
            self.assertEqual(receipt["status"], "stopped_at_gate")
            self.assertEqual(receipt["quick"]["observed_case_count"], 1)
            self.assertIsNone(receipt["quality"])
            self.assertFalse(receipt["first_smoke_rerun"])
            quality_command.assert_not_called()

    def test_hash_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bound.json"
            binding = write_json(path, {"value": 1})
            with self.assertRaisesRegex(pilot.PilotError, "SHA-256"):
                continuation._require_binding(
                    dict(binding, sha256="0" * 64), path,
                    binding["sha256"], binding["size"], "fixture",
                )

    def test_parent_acceptance_binds_controller_preflight_and_immutable_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            preflight = root / "preflight.json"
            controller_path = Path(continuation.__file__).resolve()
            preflight_binding = write_json(preflight, {
                "schema_version": 1,
                "kind": "bonsai2_dspark_continuation_preflight",
                "status": "passed",
                "gpu_or_model_execution_performed": False,
                "controller_sha256": pilot.sha256_file(controller_path),
            })
            acceptance = root / "acceptance.json"
            controller_binding = {
                "path": str(controller_path),
                "sha256": pilot.sha256_file(controller_path),
                "size": controller_path.stat().st_size,
            }

            def pinned(path, digest, size):
                return {"path": str(path.resolve()), "sha256": digest, "size": size}

            document = {
                "schema_version": 1,
                "kind": "bonsai2_dspark_continuation_parent_acceptance",
                "decision": "execute",
                "accepted_by_parent": True,
                "tag": continuation.TAG,
                "original_tag": continuation.ORIGINAL_TAG,
                "original_model_session_run_id": continuation.ORIGINAL_RUN_ID,
                "original_operational_deadline": continuation.ORIGINAL_DEADLINE_TEXT,
                "generation_budget": {"already_spent": 1, "continuation_maximum": 3,
                                      "total_maximum": 4, "duplicate_smoke": False},
                "controller": controller_binding,
                "preflight": preflight_binding,
                "continuation_plan": pinned(continuation.PLAN, continuation.PLAN_SHA256,
                                             continuation.PLAN_SIZE),
                "first_smoke_adjudication": pinned(
                    continuation.ADJUDICATION, continuation.ADJUDICATION_SHA256,
                    continuation.ADJUDICATION_SIZE),
                "eligibility": pinned(continuation.ELIGIBILITY,
                                      continuation.ELIGIBILITY_SHA256,
                                      continuation.ELIGIBILITY_SIZE),
                "original_candidate_config": pinned(
                    continuation.ORIGINAL_CONFIG, continuation.ORIGINAL_CONFIG_SHA256,
                    continuation.ORIGINAL_CONFIG_SIZE),
            }
            write_json(acceptance, document)
            with mock.patch.object(continuation, "PREFLIGHT", preflight), \
                    mock.patch.object(continuation, "ACCEPTANCE", acceptance):
                bound = continuation.validate_acceptance(acceptance)
                self.assertEqual(bound.path, acceptance.resolve())
                document["controller"]["sha256"] = "0" * 64
                write_json(acceptance, document)
                with self.assertRaisesRegex(pilot.PilotError, "SHA-256"):
                    continuation.validate_acceptance(acceptance)

    def test_time_gate_uses_original_deadline_and_reserve(self):
        accepted = continuation.remaining_time(
            continuation.ORIGINAL_DEADLINE - timedelta(seconds=10_000)
        )
        self.assertEqual(accepted["window_max_seconds"], 10_000)
        self.assertEqual(accepted["usable_request_seconds"], 8_200)
        with self.assertRaisesRegex(continuation.ContinuationError, "insufficient time"):
            continuation.remaining_time(
                continuation.ORIGINAL_DEADLINE
                - timedelta(seconds=continuation.RESTORATION_RESERVE_SECONDS
                            + pilot.STARTUP_TIMEOUT)
            )

    def test_ready_or_release_failure_propagates_before_execution(self):
        now = datetime(2026, 10, 20, 22, tzinfo=timezone.utc)
        binding = pilot.Binding(Path("/tmp/x"), "a" * 64, 1)
        with mock.patch.object(continuation, "validate_adjudication",
                               return_value=({"original_operational_deadline": continuation.ORIGINAL_DEADLINE_TEXT}, binding)), \
                mock.patch.object(continuation, "validate_plan",
                                  return_value=({"candidate_config": {
                                      "path": str(continuation.ORIGINAL_CONFIG),
                                      "sha256": continuation.ORIGINAL_CONFIG_SHA256,
                                      "size": continuation.ORIGINAL_CONFIG_SIZE}}, binding)), \
                mock.patch.object(continuation, "validate_acceptance", return_value=binding), \
                mock.patch.object(pilot, "validate_static_sources"), \
                mock.patch.object(pilot, "sha256_file", return_value=continuation.PILOT_SOURCE_SHA256), \
                mock.patch.object(Path, "stat", return_value=SimpleNamespace(st_size=continuation.PILOT_SOURCE_SIZE)), \
                mock.patch.object(pilot, "validate_eligibility",
                                  side_effect=pilot.PilotError("READY released")):
            with self.assertRaisesRegex(pilot.PilotError, "READY released"):
                continuation.validate_inputs(Path("/tmp/acceptance"), now=now)

    def test_adjudication_rejects_unclean_original_cleanup(self):
        document = json.loads(continuation.ADJUDICATION.read_text())
        document["independent_cleanup_verification"]["server_cleanup"]["kill_sent"] = True
        binding = pilot.Binding(continuation.ADJUDICATION.resolve(),
                                continuation.ADJUDICATION_SHA256,
                                continuation.ADJUDICATION_SIZE)
        with mock.patch.object(continuation, "_load_pinned", return_value=(document, binding)):
            with self.assertRaisesRegex(continuation.ContinuationError, "cleanup is not clean"):
                continuation.validate_adjudication()

    def test_delayed_restoration_retries_without_signaling_any_process(self):
        calls = []

        def validator(_paths, _session, _cleanup):
            calls.append(1)
            if len(calls) < 3:
                raise pilot.PilotError("transient restore probe")
            return {"restoration": {"clean": True}}

        with mock.patch.object(continuation.os, "killpg") as killpg:
            evidence, retry = continuation.retry_cleanup_evidence(
                continuation.continuation_paths(results=Path("/tmp/x")).pilot_paths(),
                SimpleNamespace(run_id="owned"), {"outer_exited": True},
                validator=validator, sleeper=lambda _delay: None,
            )
        self.assertEqual(retry["attempts"], 3)
        self.assertTrue(retry["delayed_restoration_observed"])
        self.assertEqual(evidence, {"restoration": {"clean": True}})
        killpg.assert_not_called()

    def test_actual_adjudication_and_plan_validate_without_reading_prose(self):
        adjudication, _ = continuation.validate_adjudication()
        plan, _ = continuation.validate_plan()
        self.assertFalse(adjudication["prose_bodies_read"])
        self.assertFalse(plan["prose_bodies_read"])
        self.assertEqual(plan["setup_plan"], continuation.EXPECTED_SETUP_PLAN)


if __name__ == "__main__":
    unittest.main()
