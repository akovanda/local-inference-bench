import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import bench_cases
from runtime import run_style_poll0_probe as probe
from runtime import specialist_cases
from runtime import specialist_study_trial as study


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _primary_receipt() -> dict:
    return {
        "schema_version": 1,
        "tag": probe.PARENT_TAG,
        "status": "completed_with_evaluation_failures",
        "runtime_backend": "llama",
        "problem": None,
        "source_config": {"path": str(probe.BASE_CONFIG), "sha256": probe.BASE_CONFIG_SHA256},
        "plan": {"path": str(probe.PRIMARY_PLAN), "sha256": probe.PRIMARY_PLAN_SHA256},
        "session": {
            "run_id": "primary-run", "model_name": probe.MODEL, "context": probe.CONTEXT,
        },
        "commands": [
            # Return code 1 is terminal evaluation failure, not a missing attempt.
            {"kind": "specialist_code", "returncode": 1},
            {"kind": "specialist_prose", "returncode": 0},
        ],
        "cleanup": {
            "outer_exited": True,
            "orchestrator_sigkill_sent": False,
            "model_session_cleanup": {"stopped": True, "kill_sent": False},
        },
        "restoration": {
            "original_running": True, "original_paused": False,
            "owned_loopback_port_clear": True,
        },
        # Standard-plan failures include legitimate 64K full-cap rows.
        "session_report_status": "completed_with_step_failures",
        "controls": {
            "profile": probe.PRIMARY_PROFILE,
            "code_thinking": "medium",
            "prose_thinking": "none",
            "code": {
                "output_cap": probe.OUTPUT_CAP, "temperature": 0.0,
                "seed": 0, "timeout": probe.REQUEST_TIMEOUT,
            },
            "prose": {
                "output_cap": probe.OUTPUT_CAP, "temperature": 0.7,
                "seed": 0, "timeout": probe.REQUEST_TIMEOUT,
            },
        },
        "sampler_plan": {
            "path": str(probe.PRIMARY_SAMPLER_PLAN),
            "sha256": probe.PRIMARY_SAMPLER_PLAN_SHA256,
            "expected_model_name": probe.MODEL,
            "matched": True,
            "stages": [{
                "model": probe.MODEL,
                "group": "prose",
                "suffix": "styletune-publisher-prose",
                "prose_thinking": "none",
                "returncode": 0,
                "recipe": {"sha256": probe.PUBLISHER_RECIPE_SHA256},
                # Publisher full caps are terminal/accounted, not a quality gate.
                "receipt": {
                    "status": "complete", "attempted_count": 3,
                    "completed_count": 1, "truncated_count": 2,
                    "request_failure_count": 0,
                },
            }],
        },
    }


def _primary_rows(*, wall_seconds: float = 100.0) -> list[dict]:
    rows = []
    for index, case in enumerate(specialist_cases.get_cases(groups=("prose",))):
        rows.append({
            "schema_version": 1,
            "run_id": "primary-prose",
            "call_id": f"call-{index}",
            "repeat_index": 0,
            "model": probe.MODEL,
            "tag": probe.PARENT_TAG + "-prose",
            "profile": probe.PRIMARY_PROFILE,
            "suite": "specialist_prose",
            "case_id": case.case_id,
            "case_group": "prose",
            "case_version": specialist_cases.CASE_VERSION,
            "case_sha256": specialist_cases.case_sha256(case),
            "suite_sha256": specialist_cases.FROZEN_SUITE_SHA256,
            "request": {
                "endpoint": probe.ENDPOINT,
                "model": probe.MODEL,
                "profile": probe.PRIMARY_PROFILE,
                "backend": "openai",
                "thinking": "none",
                "chat_template_thinking": "off",
                "reasoning_budget_tokens": None,
                "temperature": 0.7,
                "seed": 0,
                "context_max_tokens": probe.CONTEXT,
                "output_token_limit": probe.OUTPUT_CAP,
                "timeout_seconds": probe.REQUEST_TIMEOUT,
                "prompt_chars": len(case.prompt),
                "prompt_sha256": hashlib.sha256(case.prompt.encode()).hexdigest(),
            },
            "response": {
                "actual_completion_tokens": 200 + index,
                "done_reason": "stop",
                "stream_complete": True,
                "truncated": False,
            },
            "response_metrics": {
                "finish_reason": "stop",
                "output_tokens_reported": 200 + index,
                "wall_seconds": wall_seconds,
            },
            "visible_text": {
                "path": f"opaque-{index}.txt",
                "sha256": f"{index + 1:064x}",
                "chars": 900 + index,
                "bytes": 900 + index,
                "reasoning_excluded": True,
            },
            "objective_outcome": "wrong_answer",
            "objective_error": None,
            "error": None,
            "evaluation": {"passed": False},
        })
    return rows


def _standard_rows(suite: str) -> list[dict]:
    rows = []
    for index, case in enumerate(bench_cases.get_cases(suite)):
        full_cap = suite == "quick" and case.case_id == "quick_generation"
        tokens = probe.OUTPUT_CAP if full_cap else 200 + index
        rows.append({
            "schema_version": 1,
            "run_id": f"primary-{suite}",
            "call_id": f"{suite}-call-{index}",
            "repeat_index": 0,
            "suite": suite,
            "case_id": case.case_id,
            "case_version": bench_cases.CASE_VERSION,
            "case_source_sha256": probe.BENCH_CASES_SHA256,
            "request": {
                "endpoint": probe.ENDPOINT,
                "backend": "openai",
                "model": probe.MODEL,
                "context_max_tokens": probe.CONTEXT,
                "thinking": "medium",
                "chat_template_thinking": "on",
                "reasoning_budget_tokens": None,
                "temperature": 0.0,
                "seed": 0,
                "output_token_limit": probe.OUTPUT_CAP,
                "timeout_seconds": probe.REQUEST_TIMEOUT,
                "context_mode": "full",
                "prompt_chars": len(case.prompt),
                "prompt_sha256": hashlib.sha256(case.prompt.encode()).hexdigest(),
            },
            "response": {
                # The real standard bench format contains generated strings.  The
                # controller must project these fields out of all evidence.
                "response_text": "fixture generated text must never enter the receipt",
                "reasoning_text": "fixture private reasoning must never enter the receipt",
                "reasoning_chars": 25,
                "actual_prompt_tokens": 20,
                "actual_completion_tokens": tokens,
                "done_reason": "length" if full_cap else "stop",
                "stream_complete": True,
                "truncated": full_cap,
                "timing": {"wall_seconds": 10.0 + index},
            },
            "evaluation": {"passed": not full_cap},
            "error": None,
        })
    return rows


def _specialist_rows(group: str) -> list[dict]:
    if group == "prose":
        return _primary_rows()
    rows = []
    for index, case in enumerate(specialist_cases.get_cases(groups=(group,))):
        rows.append({
            "schema_version": 1,
            "run_id": "primary-code",
            "call_id": f"code-call-{index}",
            "repeat_index": 0,
            "model": probe.MODEL,
            "tag": probe.PARENT_TAG + "-code",
            "profile": probe.PRIMARY_PROFILE,
            "suite": "specialist_code",
            "case_id": case.case_id,
            "case_group": "code",
            "case_version": specialist_cases.CASE_VERSION,
            "case_sha256": specialist_cases.case_sha256(case),
            "suite_sha256": specialist_cases.FROZEN_SUITE_SHA256,
            "request": {
                "endpoint": probe.ENDPOINT,
                "model": probe.MODEL,
                "profile": probe.PRIMARY_PROFILE,
                "backend": "openai",
                "thinking": "medium",
                "chat_template_thinking": "on",
                "reasoning_budget_tokens": None,
                "temperature": 0.0,
                "seed": 0,
                "context_max_tokens": probe.CONTEXT,
                "output_token_limit": probe.OUTPUT_CAP,
                "timeout_seconds": probe.REQUEST_TIMEOUT,
                "prompt_chars": len(case.prompt),
                "prompt_sha256": hashlib.sha256(case.prompt.encode()).hexdigest(),
            },
            "response": {
                "actual_completion_tokens": 300 + index,
                "done_reason": "stop",
                "stream_complete": True,
                "truncated": False,
            },
            "response_metrics": {
                "finish_reason": "stop", "output_tokens_reported": 300 + index,
                "wall_seconds": 20.0,
            },
            "visible_text": {
                "path": f"opaque-code-{index}.txt", "sha256": f"{index + 4:064x}",
                "chars": 100, "bytes": 100, "reasoning_excluded": True,
            },
            "objective_outcome": "wrong_answer",
            "objective_error": None,
            "error": None,
            "evaluation": {"passed": False},
        })
    return rows


def _publisher_rows() -> list[dict]:
    rows = []
    for index, case in enumerate(specialist_cases.get_cases(groups=("prose",))):
        rows.append({
            "schema_version": 1,
            "run_id": "publisher-run",
            "call_id": f"publisher-call-{index}",
            "repeat_index": 0,
            "model": probe.MODEL,
            "tag": probe.PARENT_TAG,
            "profile": probe.PRIMARY_PROFILE,
            "suite": "specialist_prose",
            "case_id": case.case_id,
            "case_group": "prose",
            "case_version": specialist_cases.CASE_VERSION,
            "case_sha256": specialist_cases.case_sha256(case),
            "suite_sha256": specialist_cases.FROZEN_SUITE_SHA256,
            "request": {
                "endpoint": probe.ENDPOINT,
                "backend": "openai",
                "model": probe.MODEL,
                "profile": probe.PRIMARY_PROFILE,
                "thinking": "none",
                "chat_template_thinking": "off",
                "temperature": 1.0,
                "seed": 0,
                "context_max_tokens": probe.CONTEXT,
                "output_token_limit": probe.OUTPUT_CAP,
                "timeout_seconds": probe.REQUEST_TIMEOUT,
                "prompt_chars": len(case.prompt),
                "prompt_sha256": hashlib.sha256(case.prompt.encode()).hexdigest(),
                "sampler_recipe_label": "StyleTune author temperature/min-p check",
                "sampler_recipe_sha256": probe.PUBLISHER_RECIPE_SHA256,
                "sampler_overrides": {"temperature": 1.0, "min_p": 0.1},
            },
            "response": {
                "actual_completion_tokens": 400 + index,
                "done_reason": "stop", "stream_complete": True, "truncated": False,
            },
            "response_metrics": {
                "finish_reason": "stop", "output_tokens_reported": 400 + index,
                "wall_seconds": 30.0,
            },
            "visible_text": {
                "path": f"opaque-publisher-{index}.txt", "sha256": f"{index + 7:064x}",
                "chars": 100, "bytes": 100, "reasoning_excluded": True,
            },
            "objective_outcome": "constraint_failed",
            "objective_error": None,
            "error": None,
            "evaluation": {"passed": False},
        })
    return rows


def _materialize_primary(
    root: Path, *, hard_rows: list[dict] | None = None,
) -> tuple[Path, Path, Path]:
    receipt_path = root / f"{probe.PARENT_TAG}-study-trial.json"
    paths = probe._fixed_primary_paths(receipt_path)
    prose_path = paths["prose"]
    boundary_path = root / "specialists-final72k64k-gemma4-style12-q5_k_m-ready.json"

    _write_jsonl(paths["quick"], _standard_rows("quick"))
    hard_data = _standard_rows("hard") if hard_rows is None else hard_rows
    _write_jsonl(paths["hard"], hard_data)
    _write_jsonl(paths["code"], _specialist_rows("code"))
    _write_jsonl(prose_path, _specialist_rows("prose"))
    publisher_rows = _publisher_rows()
    _write_jsonl(paths["publisher_requests"], publisher_rows)

    expected_prose = {
        case.case_id: case for case in specialist_cases.get_cases(groups=("prose",))
    }
    recipe = {
        "schema_version": 1,
        "label": "StyleTune author temperature/min-p check",
        "claim": "publisher_inspired",
        "sources": ["https://huggingface.co/Gryphe/Gemma-4-12B-StyleTune"],
        "sampler_overrides": {"temperature": 1.0, "min_p": 0.1},
        "path": str(probe.PUBLISHER_RECIPE),
        "sha256": probe.PUBLISHER_RECIPE_SHA256,
    }
    controls = {
        "context_max_tokens": probe.CONTEXT,
        "expected_context_tokens": probe.CONTEXT,
        "actual_context_tokens": probe.CONTEXT,
        "output_token_limit": probe.OUTPUT_CAP,
        "timeout_seconds": probe.REQUEST_TIMEOUT,
        "seed": 0,
        "prose_thinking": "none",
        "ledger_and_code_thinking": "medium",
    }
    manifest = {
        "schema_version": 1,
        "diagnostic_kind": "resident_sampler_recipe_probe",
        "run_id": "publisher-run",
        "tag": probe.PARENT_TAG,
        "model_session_run_id": "primary-run",
        "recipe": recipe,
        "controls": controls,
        "case_count": 3,
        "prose_prompt_count": 3,
        "cases": [
            {
                "case_id": case_id,
                "group": "prose",
                "case_version": specialist_cases.CASE_VERSION,
                "case_sha256": specialist_cases.case_sha256(case),
                "prompt_sha256": hashlib.sha256(case.prompt.encode()).hexdigest(),
            }
            for case_id, case in expected_prose.items()
        ],
    }
    _write_json(paths["publisher_manifest"], manifest)
    sampler_receipt = {
        "schema_version": 1,
        "diagnostic_kind": "resident_sampler_recipe_probe",
        "run_id": "publisher-run",
        "tag": probe.PARENT_TAG,
        "model": probe.MODEL,
        "model_session_run_id": "primary-run",
        "runtime_backend": "llama",
        "status": "complete",
        "recipe": recipe,
        "controls": controls,
        "case_count": 3,
        "prose_prompt_count": 3,
        "attempted_count": 3,
        "completed_count": 3,
        "truncated_count": 0,
        "request_failure_count": 0,
        "artifacts": {
            "requests": str(paths["publisher_requests"]),
            "manifest": str(paths["publisher_manifest"]),
            "receipt": str(paths["publisher_receipt"]),
            "visible_dir": str(paths["publisher_visible"]),
        },
        "attempts": [
            {"case_id": row["case_id"], "call_id": row["call_id"]}
            for row in publisher_rows
        ],
    }
    _write_json(paths["publisher_receipt"], sampler_receipt)

    step_controls = []
    for step_id, suite in (("quick_medium_65536", "quick"), ("hard_medium_65536", "hard")):
        step_controls.append({
            "id": step_id,
            "suite": suite,
            "cases": [],
            "thinking": "medium",
            "chat_template_thinking": "on",
            "temperature": 0.0,
            "seed": 0,
            "repeat": 1,
            "output_tokens": probe.OUTPUT_CAP,
            "request_timeout": probe.REQUEST_TIMEOUT,
        })
    _write_json(paths["session_report"], {
        "schema_version": 1,
        "session_id": "primary-run",
        "tag": probe.PARENT_TAG,
        "runtime_backend": "llama",
        "status": "completed_with_step_failures",
        "source_config": str(probe.BASE_CONFIG),
        "source_config_sha256": probe.BASE_CONFIG_SHA256,
        "plan": {"path": str(probe.PRIMARY_PLAN), "sha256": probe.PRIMARY_PLAN_SHA256},
        "steps": [
            {
                "index": 1, "id": "quick_medium_65536", "controls": step_controls[0],
                "result": str(paths["quick"]), "returncode": 1, "error": None,
            },
            {
                "index": 2, "id": "hard_medium_65536", "controls": step_controls[1],
                "result": str(paths["hard"]),
                "returncode": 1 if any(row.get("error") is not None for row in hard_data) else 0,
                "error": None,
            },
        ],
    })

    receipt = _primary_receipt()
    receipt["plan"]["steps"] = step_controls
    receipt["commands"] = [
        {
            "kind": "specialist_code", "returncode": 1,
            "argv": ["helper", "--jsonl", str(paths["code"]), "--groups", "code"],
        },
        {
            "kind": "specialist_prose", "returncode": 0,
            "argv": ["helper", "--jsonl", str(paths["prose"]), "--groups", "prose"],
        },
    ]
    receipt["specialist_manifest"] = {
        "sha256": "1" * 64,
        "document": {
            "case_version": specialist_cases.CASE_VERSION,
            "suite_sha256": specialist_cases.FROZEN_SUITE_SHA256,
            "case_count": len(specialist_cases.get_cases()),
            "cases": [
                {
                    "case_id": case.case_id,
                    "group": case.group,
                    "case_sha256": specialist_cases.case_sha256(case),
                }
                for case in specialist_cases.get_cases()
            ],
        },
    }
    stage = receipt["sampler_plan"]["stages"][0]
    stage["recipe"]["path"] = str(probe.PUBLISHER_RECIPE)
    stage["receipt"] = {
        "exists": True,
        "path": str(paths["publisher_receipt"]),
        "sha256": probe.sha256_file(paths["publisher_receipt"]),
        "status": "complete",
        "attempted_count": 3,
        "completed_count": 3,
        "truncated_count": 0,
        "request_failure_count": 0,
    }
    artifact_paths = {
        "session_report": paths["session_report"],
        "plan_01_quick_medium_65536": paths["quick"],
        "plan_02_hard_medium_65536": paths["hard"],
        "code": paths["code"],
        "prose": paths["prose"],
        "sampler_01_requests": paths["publisher_requests"],
        "sampler_01_manifest": paths["publisher_manifest"],
        "sampler_01_receipt": paths["publisher_receipt"],
    }
    receipt["artifacts"] = {
        key: {
            "path": str(path), "exists": True, "regular_file": True,
            "directory": False, "size": path.stat().st_size,
        }
        for key, path in artifact_paths.items()
    }
    _write_json(receipt_path, receipt)
    _boundary(boundary_path, receipt_path)
    return receipt_path, prose_path, boundary_path


def _boundary(path: Path, receipt_path: Path) -> None:
    _write_json(path, {
        "schema_version": 1,
        "kind": "model_boundary_ready",
        "run_prefix": "specialists-final72k64k",
        "model_key": probe.PRIMARY_PROFILE,
        "published_at": "now",
        "study": {"path": str(receipt_path), "sha256": probe.sha256_file(receipt_path)},
    })


def _args(
    root: Path, receipt: Path, prose: Path, boundary: Path, *, execute: bool = False,
) -> argparse.Namespace:
    tag = "style-poll0-test"
    return argparse.Namespace(
        primary_study_receipt=receipt,
        primary_prose=[prose],
        primary_boundary_ready=boundary,
        tag=tag,
        output_prefix=root / "results" / tag,
        execute=execute,
    )


class FakeProcess:
    def __init__(self, pid: int = 4321, returncode=None):
        self.pid = pid
        self.returncode = returncode

    def poll(self):
        return self.returncode


def _execution_receipt() -> dict:
    return {
        "execution": {},
        "commands": {},
        "cleanup": {"control_stop_requested": False, "owned_outer_sigterm_sent": False},
        "smoke": {"status": "not_run", "attempted_count": 0},
        "prose": {
            "status": "not_run", "attempted_count": 0, "completed_count": 0,
            "truncated_count": 0, "request_failure_count": 0,
        },
        "observed_resources": {"server_cpu": {}},
        "limitations": {},
    }


class StylePoll0ProbeTests(unittest.TestCase):
    def test_standalone_cli_help_bootstraps_imports_outside_repo(self):
        script = Path(probe.__file__).resolve()
        environment = os.environ.copy()
        environment.pop("PYTHONPATH", None)
        with tempfile.TemporaryDirectory() as directory:
            completed = subprocess.run(
                [sys.executable, "-I", str(script), "--help"],
                cwd=directory,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=10,
                check=False,
            )
            created = list(Path(directory).iterdir())
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--execute", completed.stdout)
        self.assertIn("--primary-boundary-ready", completed.stdout)
        self.assertEqual(created, [])

    def test_frozen_pins_and_exact_only_poll0_config_difference(self):
        pins = probe.verify_pins()
        self.assertTrue(pins["verified"])
        self.assertEqual(pins["candidate"]["only_appended_args"], ["--poll", "0"])
        base = json.loads(probe.BASE_CONFIG.read_text())
        candidate = json.loads(probe.CONFIG.read_text())
        probe._assert_poll0_config(candidate, base)
        changed = json.loads(json.dumps(candidate))
        changed["context"] = 8192
        with self.assertRaisesRegex(probe.ProbeError, "outside args"):
            probe._assert_poll0_config(changed, base)
        changed = json.loads(json.dumps(candidate))
        changed["args"].extend(["--spec-type", "draft-mtp"])
        with self.assertRaisesRegex(probe.ProbeError, "appended"):
            probe._assert_poll0_config(changed, base)

    def test_missing_or_incomplete_primary_inspect_writes_only_receipt_and_never_launches(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results = root / "results"
            results.mkdir()
            receipt_path = root / "primary.json"
            prose_path = root / "primary-prose.jsonl"
            boundary_path = root / "specialists-final72k64k-gemma4-style12-q5_k_m-ready.json"
            receipt = _primary_receipt()
            receipt["session_report_status"] = "running"
            _write_json(receipt_path, receipt)
            _write_jsonl(prose_path, _primary_rows())
            _boundary(boundary_path, receipt_path)
            args = _args(root, receipt_path, prose_path, boundary_path)
            with (
                mock.patch.object(probe, "verify_pins", return_value={"verified": True}),
                mock.patch.object(probe.subprocess, "Popen") as popen,
            ):
                result, code = probe.run(args, results=results)
            paths = probe.probe_paths(args.tag, args.output_prefix, results=results)
            existing = [path for path in as_path_values(paths) if os.path.lexists(path)]
        self.assertEqual(code, 1)
        self.assertEqual(result["status"], "ineligible_not_launched")
        self.assertFalse(result["execution"]["launched"])
        self.assertEqual(result["prose"]["attempted_count"], 0)
        self.assertEqual(existing, [paths.receipt])
        popen.assert_not_called()

    def test_missing_primary_metadata_is_a_receipted_ineligible_inspection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results = root / "results"
            results.mkdir()
            args = _args(
                root,
                root / "missing-primary.json",
                root / "missing-prose.jsonl",
                root / "specialists-final72k64k-gemma4-style12-q5_k_m-ready.json",
            )
            with (
                mock.patch.object(probe, "verify_pins", return_value={"verified": True}),
                mock.patch.object(probe.subprocess, "Popen") as popen,
            ):
                result, code = probe.run(args, results=results)
            paths = probe.probe_paths(args.tag, args.output_prefix, results=results)
            self.assertTrue(paths.receipt.is_file())
            self.assertFalse(paths.control.exists())
            self.assertFalse(paths.sampler_manifest.exists())
        self.assertEqual(code, 1)
        self.assertEqual(result["status"], "ineligible_not_launched")
        self.assertEqual(result["prose"]["attempted_count"], 0)
        popen.assert_not_called()

    def test_primary_accounting_keeps_full_cap_and_evaluation_failures_covered(self):
        with tempfile.TemporaryDirectory() as directory:
            receipt_path, prose_path, boundary_path = _materialize_primary(Path(directory))
            eligibility = probe.validate_eligibility(
                receipt_path, [prose_path], boundary_path,
            )
        self.assertTrue(eligibility.eligible, eligibility.failures)
        self.assertEqual(eligibility.primary_receipt["attempt_accounting"], {
            "quick": 2, "hard": 4, "canonical_medium_code": 3,
            "canonical_none_prose": 3, "publisher_none_prose": 3, "total": 15,
        })
        full_cap = next(
            row for row in eligibility.primary_receipt["attempts"]
            if row["case_id"] == "quick_generation"
        )
        self.assertTrue(full_cap["terminal"])
        self.assertTrue(full_cap["truncated"])
        self.assertEqual(full_cap["completion_tokens"], probe.OUTPUT_CAP)
        serialized = json.dumps(eligibility.primary_receipt)
        self.assertNotIn("fixture generated text", serialized)
        self.assertNotIn("fixture private reasoning", serialized)

        receipt = _primary_receipt()
        receipt["commands"][0]["returncode"] = -15
        failures, _ = probe._primary_receipt_evidence(receipt)
        self.assertTrue(any("specialist_code" in item for item in failures))

    def test_recorded_primary_request_error_is_terminal_and_accounted(self):
        hard_rows = _standard_rows("hard")
        hard_rows[-1]["response"] = None
        hard_rows[-1]["error"] = {"type": "TimeoutError", "message": "recorded only"}
        with tempfile.TemporaryDirectory() as directory:
            receipt_path, prose_path, boundary_path = _materialize_primary(
                Path(directory), hard_rows=hard_rows,
            )
            eligibility = probe.validate_eligibility(
                receipt_path, [prose_path], boundary_path,
            )
        self.assertTrue(eligibility.eligible, eligibility.failures)
        request_error = next(
            row for row in eligibility.primary_receipt["attempts"]
            if row["case_id"] == "hard_dependency_constraints"
        )
        self.assertTrue(request_error["terminal"])
        self.assertTrue(request_error["request_error"])
        self.assertEqual(eligibility.primary_receipt["attempt_accounting"]["total"], 15)

    def test_exact_eligibility_rejects_prose_and_boundary_mismatches(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            receipt_path, prose_path, boundary_path = _materialize_primary(root)
            eligibility = probe.validate_eligibility(receipt_path, [prose_path], boundary_path)
            self.assertTrue(eligibility.eligible, eligibility.failures)
            self.assertEqual(eligibility.primary_receipt["attempt_accounting"]["total"], 15)
            rows = _primary_rows()
            rows[0]["request"]["prompt_sha256"] = "0" * 64
            _write_jsonl(prose_path, rows)
            eligibility = probe.validate_eligibility(receipt_path, [prose_path], boundary_path)
            self.assertFalse(eligibility.eligible)
            self.assertTrue(any("identity mismatch" in item for item in eligibility.failures))
            _write_jsonl(prose_path, _primary_rows())
            boundary_path.with_name(
                "specialists-final72k64k-gemma4-style12-q5_k_m-release.json"
            ).write_text("{}\n")
            eligibility = probe.validate_eligibility(receipt_path, [prose_path], boundary_path)
            self.assertFalse(eligibility.eligible)
            self.assertTrue(any("released" in item for item in eligibility.failures))

    def test_terminal_receipt_with_one_missing_raw_primary_row_is_ineligible(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            incomplete_hard = _standard_rows("hard")[:-1]
            receipt_path, prose_path, boundary_path = _materialize_primary(
                root, hard_rows=incomplete_hard,
            )
            eligibility = probe.validate_eligibility(
                receipt_path, [prose_path], boundary_path,
            )
        self.assertFalse(eligibility.eligible)
        self.assertEqual(eligibility.primary_receipt["attempt_accounting"]["hard"], 3)
        self.assertEqual(eligibility.primary_receipt["attempt_accounting"]["total"], 14)
        self.assertTrue(any("exactly 4 rows" in item for item in eligibility.failures))

    def test_commands_pin_window_smoke_profile_and_prose_only_controls(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results = root / "results"
            results.mkdir()
            paths = probe.probe_paths("style-poll0-test", results / "style-poll0-test", results=results)
            outer = probe.outer_command("style-poll0-test", paths.control)
            sampler = probe.sampler_command("style-poll0-test", paths.sampler_prefix, "exact-run")
        self.assertIn("--long-session", outer)
        self.assertEqual(outer[outer.index("--wait-seconds") + 1], "900")
        self.assertEqual(outer[outer.index("--max-seconds") + 1], "14400")
        self.assertEqual(outer[outer.index("--config") + 1], str(probe.CONFIG))
        self.assertEqual(outer[outer.index("--plan") + 1], str(probe.SMOKE_PLAN))
        self.assertIn("--hold-after-plan", outer)
        self.assertEqual(sampler[sampler.index("--recipe") + 1], str(probe.RECIPE))
        self.assertEqual(sampler[sampler.index("--profile") + 1], probe.CANDIDATE_PROFILE)
        self.assertEqual(sampler[sampler.index("--groups") + 1], "prose")
        self.assertEqual(sampler[sampler.index("--prose-thinking") + 1], "none")
        self.assertEqual(sampler[sampler.index("--output-cap") + 1], "65536")
        self.assertEqual(sampler[sampler.index("--timeout") + 1], "21600")
        self.assertEqual(sampler[sampler.index("--model-session-run-id") + 1], "exact-run")
        self.assertNotIn("code", sampler[sampler.index("--groups") + 1])

    def test_refuses_collisions_control_and_occupied_port_before_window_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results = root / "results"
            results.mkdir()
            paths = probe.probe_paths("style-poll0-test", results / "style-poll0-test", results=results)
            paths.smoke.touch()
            with self.assertRaisesRegex(probe.ProbeError, "existing output/control"):
                probe.assert_unused(paths)
            paths.smoke.unlink()
            paths.control.write_text('{"stop":false}\n')
            with (
                mock.patch.object(probe, "assert_ports_clear") as ports,
                mock.patch.object(probe.subprocess, "Popen") as popen,
                self.assertRaisesRegex(probe.ProbeError, "must not exist"),
            ):
                probe.execute_probe(paths, "style-poll0-test", _execution_receipt())
            ports.assert_not_called()
            popen.assert_not_called()
            paths.control.unlink()
            with (
                mock.patch.object(probe, "assert_ports_clear", side_effect=probe.ProbeError("occupied")),
                mock.patch.object(probe.subprocess, "Popen") as popen,
                self.assertRaisesRegex(probe.ProbeError, "occupied"),
            ):
                probe.execute_probe(paths, "style-poll0-test", _execution_receipt())
            popen.assert_not_called()

    def test_exact_held_identity_and_smoke_reject_foreign_or_stale_lifecycle(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results = root / "results"
            results.mkdir()
            tag = "style-poll0-test"
            paths = probe.probe_paths(tag, results / tag, results=results)
            base = [
                {
                    "at": "start", "event": "session_started", "tag": tag,
                    "run_id": "owned", "runtime_backend": "llama",
                    "source_config": str(probe.CONFIG), "plan": str(probe.SMOKE_PLAN),
                    "control_file": str(paths.control), "model_name": probe.MODEL,
                    "context": probe.CONTEXT,
                },
                {
                    "at": "ready", "event": "server_ready", "tag": tag,
                    "run_id": "owned", "initial_load_seconds": 2.5,
                },
                {
                    "at": "hold", "event": "holding_started", "tag": tag,
                    "run_id": "owned", "control_file": str(paths.control),
                },
            ]
            _write_jsonl(paths.events, base)
            session = probe.wait_for_holding(FakeProcess(), paths, time.monotonic() + 1)
            self.assertEqual(session.run_id, "owned")
            mixed = list(base)
            mixed[1] = dict(mixed[1], run_id="foreign")
            _write_jsonl(paths.events, mixed)
            with self.assertRaisesRegex(probe.ProbeError, "multiple run IDs"):
                probe.wait_for_holding(FakeProcess(), paths, time.monotonic() + 1)
            terminal = list(base) + [{
                "at": "done", "event": "server_stopped", "tag": tag, "run_id": "owned",
            }]
            _write_jsonl(paths.events, terminal)
            with self.assertRaisesRegex(probe.ProbeError, "stale or terminal"):
                probe.wait_for_holding(FakeProcess(), paths, time.monotonic() + 1)

    def test_smoke_failure_skips_prose_and_stops_session(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results = root / "results"
            results.mkdir()
            tag = "style-poll0-test"
            paths = probe.probe_paths(tag, results / tag, results=results)
            outer = FakeProcess()
            session = study.HoldingSession(
                "owned", "llama", probe.MODEL, probe.CONTEXT, 1.0, "a", "b", "c"
            )
            receipt = _execution_receipt()
            actions = []

            def held(*_args, **_kwargs):
                paths.control.write_text('{"stop":false}\n')
                return session

            def stop(_path):
                actions.append("control_stop")
                return True

            with (
                mock.patch.object(probe, "assert_ports_clear", return_value={"native": True, "strata": True}),
                mock.patch.object(probe.subprocess, "Popen", return_value=outer),
                mock.patch.object(probe, "wait_for_holding", side_effect=held),
                mock.patch.object(probe, "_server_identity", return_value={"run_id": "owned", "server_pid": 99}),
                mock.patch.object(probe, "procfs_cpu_snapshot", return_value={"available": False}),
                mock.patch.object(probe, "validate_smoke", side_effect=probe.ProbeError("smoke failed")),
                mock.patch.object(probe, "run_owned") as run_owned,
                mock.patch.object(probe.study, "request_session_stop", side_effect=stop),
                mock.patch.object(probe.study, "wait_for_outer_exit", return_value=(0, False)),
                mock.patch.object(probe, "restoration_evidence", return_value={
                    "original_running": True, "original_paused": False,
                    "original_healthy": True, "both_owned_ports_clear": True,
                }),
                mock.patch.object(probe, "session_cleanup_evidence", return_value=({}, [])),
            ):
                code = probe.execute_probe(paths, tag, receipt)
        self.assertEqual(code, 1)
        run_owned.assert_not_called()
        self.assertEqual(receipt["prose"]["status"], "not_run")
        self.assertEqual(receipt["prose"]["attempted_count"], 0)
        self.assertEqual(actions, ["control_stop"])
        self.assertTrue(receipt["cleanup"]["control_stop_requested"])
        self.assertFalse(receipt["cleanup"]["orchestrator_sigkill_sent"])

    def test_interrupted_helper_counts_unknown_and_cleanup_precedes_control(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results = root / "results"
            results.mkdir()
            tag = "style-poll0-test"
            paths = probe.probe_paths(tag, results / tag, results=results)
            outer = FakeProcess(pid=4100)
            helper = FakeProcess(pid=4200, returncode=-15)
            session = study.HoldingSession(
                "owned", "llama", probe.MODEL, probe.CONTEXT, 1.0, "a", "b", "c"
            )
            receipt = _execution_receipt()
            case = specialist_cases.get_cases(groups=("prose",))[0]
            partial_rows = [{
                "case_id": case.case_id,
                "call_id": "call-1",
                "response": {"stream_complete": True, "truncated": False, "actual_completion_tokens": 17},
                "response_metrics": {"output_tokens_reported": 17, "wall_seconds": 2.5},
                "error": None,
            }]
            failure = probe.ProbeError("whole-window guard expired")
            setattr(failure, "owned_process", helper)
            setattr(failure, "owned_cleanup", {
                "pid": helper.pid, "pgid": helper.pid, "term_sent": True,
                "kill_sent": False, "stopped": True,
            })
            actions = []

            def held(*_args, **_kwargs):
                paths.control.write_text('{"stop":false}\n')
                return session

            def stop(_path):
                actions.append("control_stop")
                return True

            def interrupted_helper(*_args, **_kwargs):
                _write_jsonl(paths.sampler_requests, partial_rows)
                with paths.sampler_requests.open("ab") as handle:
                    handle.write(b'{"partial"')
                actions.append("helper_stop")
                raise failure

            with (
                mock.patch.object(probe, "assert_ports_clear", return_value={"native": True, "strata": True}),
                mock.patch.object(probe.subprocess, "Popen", return_value=outer),
                mock.patch.object(probe, "wait_for_holding", side_effect=held),
                mock.patch.object(probe, "_server_identity", return_value={"run_id": "owned", "server_pid": 99}),
                mock.patch.object(probe, "procfs_cpu_snapshot", return_value={"available": False}),
                mock.patch.object(probe, "validate_smoke", return_value={"passed": True}),
                mock.patch.object(probe, "run_owned", side_effect=interrupted_helper),
                mock.patch.object(probe.study, "request_session_stop", side_effect=stop),
                mock.patch.object(probe.study, "wait_for_outer_exit", return_value=(0, False)),
                mock.patch.object(probe, "restoration_evidence", return_value={
                    "original_running": True, "original_paused": False,
                    "original_healthy": True, "both_owned_ports_clear": True,
                }),
                mock.patch.object(probe, "session_cleanup_evidence", return_value=({}, [])),
            ):
                code = probe.execute_probe(paths, tag, receipt)
        self.assertEqual(code, 1)
        self.assertEqual(receipt["status"], "infrastructure_failed")
        self.assertEqual(receipt["prose"]["status"], "interrupted")
        self.assertIsNone(receipt["prose"]["attempted_count"])
        self.assertIsNone(receipt["prose"]["completed_count"])
        observed = receipt["prose"]["observed_request_metadata"]
        self.assertEqual(observed["observed_row_count"], 1)
        self.assertGreater(observed["ignored_incomplete_trailing_bytes"], 0)
        self.assertTrue(receipt["cleanup"]["prose_helper_stopped_before_control"])
        self.assertEqual(actions, ["helper_stop", "control_stop"])

    def test_procfs_snapshot_comparison_binds_pid_and_start_ticks(self):
        before = {
            "available": True, "pid": 9, "start_ticks": 100, "cpu_ticks": 200,
            "clock_ticks_per_second": 100, "monotonic_ns": 1_000_000_000,
        }
        after = {
            "available": True, "pid": 9, "start_ticks": 100, "cpu_ticks": 350,
            "clock_ticks_per_second": 100, "monotonic_ns": 2_000_000_000,
        }
        summary = probe.summarize_cpu_snapshots(before, after)
        self.assertTrue(summary["comparable"])
        self.assertEqual(summary["delta_cpu_seconds"], 1.5)
        self.assertEqual(summary["mean_logical_cpu_equivalents"], 1.5)
        after["start_ticks"] = 101
        self.assertFalse(probe.summarize_cpu_snapshots(before, after)["comparable"])

    def test_terminal_sampler_receipt_validates_exact_poll0_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            requests_path = root / "requests.jsonl"
            receipt_path = root / "receipt.json"
            rows = []
            attempts = []
            for index, case in enumerate(specialist_cases.get_cases(groups=("prose",))):
                truncated = index == 2
                rows.append({
                    "tag": "style-poll0-test",
                    "profile": probe.CANDIDATE_PROFILE,
                    "model": probe.MODEL,
                    "case_id": case.case_id,
                    "case_group": "prose",
                    "case_sha256": specialist_cases.case_sha256(case),
                    "call_id": f"call-{index}",
                    "request": {
                        "model": probe.MODEL,
                        "profile": probe.CANDIDATE_PROFILE,
                        "thinking": "none",
                        "chat_template_thinking": "off",
                        "temperature": 0.7,
                        "seed": 0,
                        "context_max_tokens": probe.CONTEXT,
                        "output_token_limit": probe.OUTPUT_CAP,
                        "timeout_seconds": probe.REQUEST_TIMEOUT,
                        "prompt_sha256": hashlib.sha256(case.prompt.encode()).hexdigest(),
                        "sampler_recipe_sha256": probe.RECIPE_SHA256,
                    },
                    "response": {
                        "stream_complete": True,
                        "truncated": truncated,
                        "actual_completion_tokens": 100 + index,
                        "done_reason": "length" if truncated else "stop",
                        "openai_native_timings": {"predicted_ms": 1000, "predicted_per_second": 10},
                    },
                    "response_metrics": {
                        "output_tokens_reported": 100 + index,
                        "wall_seconds": 10 + index,
                    },
                    "error": None,
                })
                attempts.append({
                    "call_id": f"call-{index}", "case_id": case.case_id,
                    "outcome": "truncated" if truncated else "wrong_answer",
                    "stream_complete": True, "truncated": truncated,
                    "output_tokens": 100 + index, "wall_seconds": 10 + index,
                })
            _write_jsonl(requests_path, rows)
            _write_json(receipt_path, {
                "tag": "style-poll0-test",
                "model": probe.MODEL,
                "model_session_run_id": "held-run",
                "runtime_backend": "llama",
                "status": "complete",
                "attempted_count": 3,
                "completed_count": 2,
                "truncated_count": 1,
                "request_failure_count": 0,
                "attempts": attempts,
                "recipe": {
                    "path": str(probe.RECIPE),
                    "sha256": probe.RECIPE_SHA256,
                    "sampler_overrides": {"temperature": 0.7},
                },
                "controls": {
                    "prose_thinking": "none",
                    "output_token_limit": probe.OUTPUT_CAP,
                    "timeout_seconds": probe.REQUEST_TIMEOUT,
                    "seed": 0,
                    "expected_context_tokens": probe.CONTEXT,
                    "actual_context_tokens": probe.CONTEXT,
                },
            })
            summary = probe.summarize_sampler_receipt(
                receipt_path,
                "held-run",
                0,
                expected_tag="style-poll0-test",
                requests_path=requests_path,
            )
        self.assertTrue(summary["terminal_sampler_receipt_validated"])
        self.assertEqual(summary["attempted_count"], 3)
        self.assertEqual(summary["truncated_count"], 1)
        self.assertEqual(summary["request_metadata"]["row_count"], 3)


def as_path_values(paths: probe.ProbePaths):
    return [
        paths.receipt, paths.control, paths.events, paths.session_report, paths.server_log,
        paths.smoke, paths.sampler_requests, paths.sampler_manifest, paths.sampler_receipt,
        paths.sampler_visible,
    ]


if __name__ == "__main__":
    unittest.main()
