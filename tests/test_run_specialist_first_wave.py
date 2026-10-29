import argparse
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

from runtime import run_specialist_first_wave as cohort


def _safe_receipt(status="complete"):
    return {
        "tag": "test-study",
        "status": status,
        "runtime_backend": "llama",
        "wall_seconds": 12.5,
        "source_config": {"path": "/tmp/config.json", "sha256": "c" * 64},
        "plan": {"path": "/tmp/plan.json", "sha256": "d" * 64},
        "specialist_manifest": {"sha256": "e" * 64},
        "cleanup": {
            "outer_exited": True,
            "model_session_cleanup": {"stopped": True, "kill_sent": False},
        },
        "restoration": {
            "original_running": True,
            "original_paused": False,
            "owned_loopback_port_clear": True,
            "owned_loopback_port": 18081,
        },
        "session": {"run_id": "run", "context": 8192},
        "controls": {
            "code_thinking": "medium",
            "prose_thinking": "medium",
            "code": {"output_cap": 4096, "timeout": 900.0},
            "prose": {"output_cap": 2048, "timeout": 900.0},
        },
        "commands": [],
    }


def _args():
    return argparse.Namespace(
        baseline_wait_seconds=1.0,
        download_wait_seconds=1.0,
        prepare_timeout_seconds=1.0,
        max_window_seconds=60.0,
        wait_for_idle=1.0,
        context=8192,
        code_output_cap=4096,
        prose_output_cap=2048,
        code_timeout=900.0,
        prose_timeout=900.0,
        lfm_prose_thinking="medium",
        kat_prose_thinking="none",
        style_prose_thinking="none",
    )


def _manifest_preflight_command():
    return {
        "kind": "specialist_manifest",
        "argv": [
            cohort.sys.executable,
            str(cohort.SPECIALIST_BENCH),
            "--print-manifest",
        ],
        "returncode": 0,
    }


class RunSpecialistFirstWaveTests(unittest.TestCase):
    def test_defaults_preserve_original_8k_cohort_identity(self):
        args = cohort.build_parser().parse_args([])
        profiles = cohort.profiles_for_args(args)
        self.assertEqual(args.context, 8192)
        self.assertEqual(args.code_output_cap, 4096)
        self.assertEqual(args.prose_output_cap, 2048)
        self.assertEqual(args.max_window_seconds, 7200.0)
        self.assertEqual(args.baseline_wait_seconds, 7200.0)
        self.assertEqual(
            [(item.auto_config, item.strict_config, item.strict_tag) for item in profiles],
            [(item.auto_config, item.strict_config, item.strict_tag) for item in cohort.PROFILES],
        )

    def test_baseline_wait_has_separate_long_bound_without_changing_other_waits(self):
        parser = cohort.build_parser()
        extended = parser.parse_args(["--baseline-wait-seconds", "259200"])
        self.assertEqual(extended.baseline_wait_seconds, cohort.MAX_BASELINE_WAIT_SECONDS)
        self.assertEqual(extended.download_wait_seconds, 7200.0)
        self.assertEqual(extended.prepare_timeout_seconds, 7200.0)

        for value in ("259200.001", "inf", "nan"):
            with self.subTest(value=value), self.assertRaises(SystemExit):
                parser.parse_args(["--baseline-wait-seconds", value])

        with self.assertRaises(SystemExit):
            parser.parse_args(["--download-wait-seconds", "259200"])

    def test_model_selection_rejects_invalid_values_and_keeps_canonical_order(self):
        parser = cohort.build_parser()
        selected = parser.parse_args([
            "--models", "gemma4-style12-q5_k_m,kat-v25-iq3_xxs"
        ])
        self.assertEqual(
            selected.models,
            ("kat-v25-iq3_xxs", "gemma4-style12-q5_k_m"),
        )
        self.assertEqual(
            [profile.key for profile in cohort.profiles_for_args(selected)],
            ["kat-v25-iq3_xxs", "gemma4-style12-q5_k_m"],
        )
        for value in ("", ",", "kat-v25-iq3_xxs,", "unknown", "kat-v25-iq3_xxs,kat-v25-iq3_xxs"):
            with self.subTest(value=value), self.assertRaises(SystemExit):
                parser.parse_args(["--models", value])

    def test_baseline_gate_requires_complete_cleanup_restoration_and_clear_port(self):
        cohort.validate_baseline_receipt(_safe_receipt("completed_with_evaluation_failures"))
        changes = (
            ("status", lambda row: row.update(status="infrastructure_failed")),
            ("outer", lambda row: row["cleanup"].update(outer_exited=False)),
            ("paused", lambda row: row["restoration"].update(original_paused=True)),
            ("port", lambda row: row["restoration"].update(owned_loopback_port_clear=False)),
        )
        for label, change in changes:
            with self.subTest(label=label):
                row = json.loads(json.dumps(_safe_receipt()))
                change(row)
                with self.assertRaises(cohort.CohortError):
                    cohort.validate_baseline_receipt(row)

    def test_strict_config_requests_all_layers_and_disables_fit_without_claiming_success(self):
        source = {
            "args": ["--gpu-layers", "auto", "--parallel", "1"],
            "placement": {"policy": "auto_best_fit", "all_gpu_fit_claimed": False},
            "runtime_validation": {"load_validated": False},
        }
        strict = cohort.strict_config_document(source)
        self.assertEqual(strict["args"][:2], ["--gpu-layers", "all"])
        self.assertEqual(strict["args"][-2:], ["--fit", "off"])
        self.assertEqual(strict["placement"]["policy"], "strict_all_layers_fit_off")
        self.assertFalse(strict["placement"]["all_gpu_fit_claimed"])
        self.assertFalse(strict["runtime_validation"]["strict_gpu_residency_validated"])

    def test_context_derivation_preserves_source_and_high_budget_cli_is_explicit(self):
        source = {
            "context": 8192,
            "artifact_manifest": "../results/pinned.json",
            "args": ["--gpu-layers", "auto"],
        }
        derived = cohort.context_config_document(source, 131072)
        self.assertEqual(source["context"], 8192)
        self.assertEqual(derived["context"], 131072)
        self.assertEqual(derived["artifact_manifest"], source["artifact_manifest"])
        fit_index = derived["args"].index("--fit-ctx")
        self.assertEqual(derived["args"][fit_index + 1], "131072")
        unchanged = cohort.context_config_document(source, 32768)
        self.assertNotIn("--fit-ctx", unchanged["args"])
        parser = cohort.build_parser()
        args = parser.parse_args([
            "--run-prefix", "specialists-highcap", "--context", "131072",
            "--code-output-cap", "65536", "--prose-output-cap", "65536",
            "--code-timeout", "21600", "--prose-timeout", "21600",
            "--max-window-seconds", "86400",
        ])
        profiles = cohort.profiles_for_args(args)
        self.assertTrue(all("specialists-highcap" in str(item.auto_config) for item in profiles))
        self.assertTrue(all(item.plan == args.standard_plan for item in profiles[:2]))
        self.assertEqual(profiles[2].plan, args.role_plan)
        command = cohort.study_command(
            profiles[0], profiles[0].strict_config, profiles[0].strict_tag, "medium", args
        )
        self.assertEqual(command[command.index("--code-output-cap") + 1], "65536")
        self.assertEqual(command[command.index("--prose-output-cap") + 1], "65536")
        self.assertEqual(command[command.index("--code-timeout") + 1], "21600")
        self.assertEqual(command[command.index("--max-seconds") + 1], "86400")
        with self.assertRaises(SystemExit):
            parser.parse_args(["--max-window-seconds", "86401"])
        with self.assertRaises(SystemExit):
            parser.parse_args(["--context", "131073"])

    def test_sampler_plan_is_forwarded_only_when_explicit(self):
        parser = cohort.build_parser()
        legacy = parser.parse_args([])
        profile = cohort.profiles_for_args(legacy)[0]
        legacy_command = cohort.study_command(
            profile, profile.strict_config, profile.strict_tag, "medium", legacy
        )
        self.assertNotIn("--sampler-plan", legacy_command)

        opted_in = parser.parse_args(["--sampler-plan", "publisher-plan.json"])
        profile = cohort.profiles_for_args(opted_in)[0]
        command = cohort.study_command(
            profile, profile.strict_config, profile.strict_tag, "medium", opted_in
        )
        self.assertEqual(command[command.index("--sampler-plan") + 1], "publisher-plan.json")
        standard_only = parser.parse_args(["--standard-only"])
        profile = cohort.profiles_for_args(standard_only)[0]
        self.assertIn(
            "--standard-only",
            cohort.study_command(
                profile, profile.strict_config, profile.strict_tag, "medium", standard_only
            ),
        )

    def test_model_boundary_gate_is_opt_in_and_forwards_selected_study_identity(self):
        parser = cohort.build_parser()
        legacy = parser.parse_args([])
        self.assertIsNone(legacy.model_boundary_gates)
        opted_in = parser.parse_args([
            "--model-boundary-gates", "/tmp/gates",
            "--model-boundary-wait-seconds", "86400",
        ])
        profile = cohort.profiles_for_args(opted_in)[0]
        with mock.patch.object(
            cohort.model_boundary_gate, "publish_and_wait", return_value={"accepted": True}
        ) as publish:
            result = cohort.wait_at_model_boundary(
                opted_in, profile, Path("/tmp/study.json"), "a" * 64
            )
        self.assertEqual(result, {"accepted": True})
        publish.assert_called_once_with(
            Path("/tmp/gates"), "specialists", profile.key,
            {"path": "/tmp/study.json", "sha256": "a" * 64}, 86400.0,
        )
        with self.assertRaises(SystemExit):
            parser.parse_args(["--model-boundary-wait-seconds", "86401"])

    def _run_fixture(self, root, profiles, strict_receipts, strict_codes=None, fallback_receipts=None):
        results = root / "results"
        results.mkdir()
        baseline_path = results / "baseline.json"
        baseline_path.write_text(json.dumps(_safe_receipt()))
        strict_codes = strict_codes or [0] * len(profiles)
        fallback_receipts = fallback_receipts or {}
        calls = []

        def fake_run(argv, timeout, kind):
            calls.append((kind, list(argv)))
            if kind.startswith("prepare_"):
                return {
                    "kind": kind,
                    "argv": list(argv),
                    "returncode": 0,
                    "elapsed_seconds": 0.1,
                }
            key = kind.removeprefix("study_strict_")
            if kind.startswith("study_strict_"):
                index = [profile.key for profile in profiles].index(key)
                tag = profiles[index].strict_tag
                receipt = json.loads(json.dumps(strict_receipts[index]))
                config = Path(argv[argv.index("--config") + 1])
                plan = Path(argv[argv.index("--plan") + 1]).resolve()
                receipt["tag"] = tag
                receipt["source_config"] = {"path": str(config), "sha256": "b" * 64}
                receipt["plan"] = {"path": str(plan), "sha256": "b" * 64}
                receipt["controls"]["prose_thinking"] = argv[argv.index("--prose-thinking") + 1]
                (results / f"{tag}-study-trial.json").write_text(json.dumps(receipt))
                if not strict_receipts[index].get("session"):
                    (results / f"{tag}-server.log").write_bytes(b"cudaMalloc failed: out of memory")
                return {
                    "kind": kind,
                    "argv": list(argv),
                    "returncode": strict_codes[index],
                    "elapsed_seconds": 0.2,
                }
            key = kind.removeprefix("study_auto_fallback_")
            profile = next(item for item in profiles if item.key == key)
            receipt = fallback_receipts[key]
            receipt = json.loads(json.dumps(receipt))
            config = Path(argv[argv.index("--config") + 1])
            plan = Path(argv[argv.index("--plan") + 1]).resolve()
            receipt["tag"] = profile.fallback_tag
            receipt["source_config"] = {"path": str(config), "sha256": "b" * 64}
            receipt["plan"] = {"path": str(plan), "sha256": "b" * 64}
            receipt["controls"]["prose_thinking"] = argv[argv.index("--prose-thinking") + 1]
            (results / f"{profile.fallback_tag}-study-trial.json").write_text(json.dumps(receipt))
            return {
                "kind": kind,
                "argv": list(argv),
                "returncode": 0,
                "elapsed_seconds": 0.3,
            }

        def fake_publish(auto, strict):
            return {"placement": {"policy": "strict_all_layers_fit_off"}}

        with mock.patch.object(cohort, "_validate_static_inputs", return_value={str(p.plan): "a" * 64 for p in profiles}), mock.patch.object(
            cohort, "wait_for_model_file", side_effect=lambda spec, timeout: spec.default_directory / spec.filename
        ), mock.patch.object(cohort, "run_command", side_effect=fake_run), mock.patch.object(
            cohort, "publish_context_config", return_value={"context": 8192}
        ), mock.patch.object(
            cohort, "publish_strict_config", side_effect=fake_publish
        ), mock.patch.object(cohort, "sha256_file", return_value="b" * 64):
            receipt, code = cohort.execute_cohort(
                _args(), profiles=profiles, results=results, baseline_path=baseline_path
            )
        return receipt, code, calls

    def test_semantic_failures_proceed_through_all_profiles_in_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profiles = tuple(
                replace(
                    profile,
                    strict_config=root / f"{profile.key}-strict.json",
                    plan=root / f"{profile.key}-plan.json",
                )
                for profile in cohort.PROFILES
            )
            for profile in profiles:
                profile.plan.write_text("{}")
            receipts = [_safe_receipt("completed_with_evaluation_failures") for _ in profiles]
            result, code, calls = self._run_fixture(root, profiles, receipts, [1, 1, 1])
        self.assertEqual(code, 1)
        self.assertEqual(result["status"], "completed_with_evaluation_failures")
        self.assertEqual(result["models"][0]["strict_study"]["wall_seconds"], 12.5)
        self.assertEqual(
            result["models"][0]["strict_study"]["specialist_manifest_sha256"],
            "e" * 64,
        )
        strict_order = [kind.removeprefix("study_strict_") for kind, _ in calls if kind.startswith("study_strict_")]
        self.assertEqual(strict_order, [profile.key for profile in profiles])

    def test_selected_subset_runs_once_in_canonical_order_and_is_recorded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profiles = tuple(
                replace(
                    profile,
                    strict_config=root / f"{profile.key}-strict.json",
                    plan=root / f"{profile.key}-plan.json",
                )
                for profile in cohort.PROFILES[1:]
            )
            for profile in profiles:
                profile.plan.write_text("{}")
            receipts = [_safe_receipt("complete") for _ in profiles]
            result, code, calls = self._run_fixture(root, profiles, receipts)
        self.assertEqual(code, 0)
        expected = ["kat-v25-iq3_xxs", "gemma4-style12-q5_k_m"]
        self.assertEqual(result["controls"]["selected_models"], expected)
        self.assertEqual(result["controls"]["sequence"], expected)
        self.assertEqual(
            [kind.removeprefix("study_strict_") for kind, _ in calls if kind.startswith("study_strict_")],
            expected,
        )

    def test_safe_startup_cuda_oom_allows_one_auto_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = cohort.PROFILES[0]
            profile = replace(base, strict_config=root / "strict.json", plan=root / "plan.json")
            profile.plan.write_text("{}")
            strict = _safe_receipt("infrastructure_failed")
            strict["session"] = None
            strict["commands"] = [
                _manifest_preflight_command(),
                {
                    "kind": "gpu_window",
                    "argv": ["python3", "gpu_window.py", "--long-session"],
                    "returncode": 1,
                },
            ]
            fallback = _safe_receipt("complete")
            result, code, calls = self._run_fixture(
                root,
                (profile,),
                [strict],
                [1],
                {profile.key: fallback},
            )
        self.assertEqual(code, 0)
        self.assertEqual(result["models"][0]["selected_profile"], "auto_fallback_after_strict_cuda_oom")
        self.assertTrue(result["models"][0]["strict_startup_oom"]["confirmed"])
        self.assertEqual(sum(kind.startswith("study_auto_fallback_") for kind, _ in calls), 1)

    def test_startup_oom_manifest_exception_is_exact_and_conservative(self):
        tag = "kat-strict-72k"
        receipt = {
            "session": None,
            "commands": [
                _manifest_preflight_command(),
                {"kind": "gpu_window", "argv": ["gpu_window.py"], "returncode": 1},
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            results = Path(directory)
            log = results / f"{tag}-server.log"
            log.write_bytes(b"cudaMalloc failed: out of memory allocating 1440 MiB")
            confirmed = cohort._startup_oom(tag, receipt, results)
            self.assertTrue(confirmed["confirmed"])
            self.assertIn("cudamalloc_failed", confirmed["diagnostic_classes"])

            rejected_commands = []
            for change in (
                {"argv": [cohort.sys.executable, str(cohort.SPECIALIST_BENCH),
                           "--print-manifest", "--extra"]},
                {"argv": [cohort.sys.executable, "/tmp/spoofed-specialist_bench.py",
                           "--print-manifest"]},
                {"returncode": 1},
                {"kind": "specialist_code", "argv": ["python3", "specialist_bench.py"]},
                {"kind": "specialist_sampler", "argv": ["python3", "sampler_recipe_probe.py"]},
            ):
                command = _manifest_preflight_command()
                command.update(change)
                rejected_commands.append(command)
            for command in rejected_commands:
                with self.subTest(command=command):
                    rejected = cohort._startup_oom(
                        tag, {"session": None, "commands": [command]}, results
                    )
                    self.assertFalse(rejected["confirmed"])
                    self.assertIn("specialist launch", rejected["reason"])

            held = cohort._startup_oom(
                tag, {"session": {"run_id": "held"}, "commands": [_manifest_preflight_command()]},
                results,
            )
            self.assertFalse(held["confirmed"])

            log.write_bytes(b"server failed during startup without a CUDA allocation diagnostic")
            self.assertFalse(cohort._startup_oom(tag, receipt, results)["confirmed"])

            log.unlink()
            target = results / "other.log"
            target.write_bytes(b"cudaMalloc failed: out of memory")
            log.symlink_to(target)
            unsafe = cohort._startup_oom(tag, receipt, results)
            self.assertFalse(unsafe["confirmed"])
            self.assertIn("unsafe", unsafe["reason"])

    def test_manifest_preflight_does_not_bypass_safe_cleanup_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = cohort.PROFILES[0]
            profile = replace(base, strict_config=root / "strict.json", plan=root / "plan.json")
            profile.plan.write_text("{}")
            strict = _safe_receipt("infrastructure_failed")
            strict["session"] = None
            strict["commands"] = [_manifest_preflight_command()]
            strict["cleanup"]["outer_exited"] = False
            result, code, calls = self._run_fixture(root, (profile,), [strict], [1])

        self.assertEqual(code, 2)
        self.assertEqual(result["status"], "infrastructure_failed")
        self.assertEqual(sum(kind.startswith("study_auto_fallback_") for kind, _ in calls), 0)

    def test_unknown_infrastructure_failure_stops_before_later_models(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profiles = tuple(
                replace(profile, strict_config=root / f"{profile.key}.json", plan=root / f"{profile.key}.plan")
                for profile in cohort.PROFILES[:2]
            )
            for profile in profiles:
                profile.plan.write_text("{}")
            unknown = _safe_receipt("infrastructure_failed")
            unknown["session"] = None

            # Override fixture OOM log after it is written by using a non-OOM
            # session marker: this makes fallback categorically ineligible.
            unknown["session"] = {"run_id": "reached-hold"}
            result, code, calls = self._run_fixture(root, profiles, [unknown, _safe_receipt()])
        self.assertEqual(code, 2)
        self.assertEqual(result["status"], "infrastructure_failed")
        self.assertEqual(sum(kind.startswith("study_strict_") for kind, _ in calls), 1)
        self.assertEqual(sum(kind.startswith("study_auto_fallback_") for kind, _ in calls), 0)


if __name__ == "__main__":
    unittest.main()
