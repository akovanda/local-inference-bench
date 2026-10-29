import argparse
import hashlib
import json
import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

from runtime import run_specialist_followups as followups


def _args():
    return argparse.Namespace(
        first_wave_wait_seconds=1.0,
        download_wait_seconds=1.0,
        prepare_timeout_seconds=1.0,
        max_window_seconds=60.0,
        wait_for_idle=1.0,
        context=8192,
        code_output_cap=4096,
        prose_output_cap=2048,
        code_timeout=900.0,
        prose_timeout=900.0,
    )


def _safe_study(tag="study", status="complete", port=18081):
    return {
        "tag": tag,
        "status": status,
        "wall_seconds": 10.0,
        "runtime_backend": "llama" if port == 18081 else "strata",
        "source_config": {"path": "/tmp/config", "sha256": "c" * 64},
        "plan": {"path": "/tmp/plan", "sha256": "d" * 64},
        "specialist_manifest": {"sha256": "e" * 64},
        "controls": {
            "code_thinking": "medium",
            "prose_thinking": "none",
            "code": {"output_cap": 4096, "timeout": 900.0},
            "prose": {"output_cap": 2048, "timeout": 900.0},
        },
        "session": {"run_id": "run", "context": 8192},
        "commands": [],
        "cleanup": {
            "outer_exited": True,
            "orchestrator_sigkill_sent": False,
            "model_session_cleanup": {"stopped": True, "kill_sent": False},
        },
        "restoration": {
            "original_running": True,
            "original_paused": False,
            "owned_loopback_port_clear": True,
            "owned_loopback_port": port,
        },
    }


def _write_first_wave(root: Path, *, study=None):
    results = root / "results"
    results.mkdir(parents=True, exist_ok=True)
    study_path = results / "last-study.json"
    study = study or _safe_study()
    study_path.write_text(json.dumps(study), encoding="utf-8")
    digest = hashlib.sha256(study_path.read_bytes()).hexdigest()
    cohort = {
        "status": "completed_with_evaluation_failures",
        "models": [{
            "selected_profile": "strict_all_layers_fit_off",
            "strict_receipt": str(study_path),
            "strict_receipt_sha256": digest,
        }],
    }
    cohort_path = results / "first-wave.json"
    cohort_path.write_text(json.dumps(cohort), encoding="utf-8")
    return cohort_path, cohort


def _fixture(root: Path):
    runtime = root / "runtime"
    results = root / "results"
    models = root / "models"
    runtime.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    models.mkdir(parents=True, exist_ok=True)
    standard = runtime / "standard.json"
    role = runtime / "role.json"
    standard.write_text("{}", encoding="utf-8")
    role.write_text("{}", encoding="utf-8")
    profiles = (
        replace(
            followups.PROFILES[0],
            model_path=models / "pq.gguf",
            model_size=3,
            config=runtime / "pq.json",
            plan=standard,
            tag="test-pq",
        ),
        replace(
            followups.PROFILES[1],
            model_path=models / "ptq.gguf",
            model_size=4,
            config=runtime / "ptq.json",
            plan=role,
            tag="test-ptq",
        ),
        replace(
            followups.PROFILES[2],
            model_path=models / "flash.gguf",
            model_size=5,
            config=runtime / "flash.json",
            plan=role,
            tag="test-flash",
        ),
    )
    first_wave, _ = _write_first_wave(root)
    helpers = []
    for name in ("bonsai.py", "flash.py", "study.py"):
        path = runtime / name
        path.write_text("# fixture\n", encoding="utf-8")
        helpers.append(path)
    paths = followups.Paths(
        results=results,
        first_wave_receipt=first_wave,
        cohort_receipt=results / "cohort.json",
        cohort_events=results / "events.jsonl",
        bonsai_helper=helpers[0],
        flash_helper=helpers[1],
        study_helper=helpers[2],
        bonsai_receipt=results / "bonsai.json",
        flash_receipt=results / "flash.json",
    )
    return profiles, paths


class RunSpecialistFollowupsTests(unittest.TestCase):
    def test_defaults_preserve_original_8k_cohort_identity(self):
        args = followups.build_parser().parse_args([])
        profiles = followups.profiles_for_args(args)
        self.assertEqual(args.context, 8192)
        self.assertEqual(args.code_output_cap, 4096)
        self.assertEqual(args.prose_output_cap, 2048)
        self.assertEqual(args.max_window_seconds, 7200.0)
        self.assertEqual(
            [(item.config, item.plan, item.tag) for item in profiles],
            [(item.config, item.plan, item.tag) for item in followups.PROFILES],
        )

    def test_first_wave_gate_requires_last_safe_study_and_both_ports_clear(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, receipt = _write_first_wave(root)
            gate = followups.validate_first_wave(receipt, port_probe=lambda port: True)
            self.assertEqual(gate["ports_clear"], {"18080": True, "18081": True})

            with self.assertRaises(followups.FollowupError):
                followups.validate_first_wave(
                    receipt, port_probe=lambda port: port == 18081
                )

            unsafe = _safe_study()
            unsafe["cleanup"]["outer_exited"] = False
            _, unsafe_receipt = _write_first_wave(root / "unsafe", study=unsafe)
            with self.assertRaises(followups.FollowupError):
                followups.validate_first_wave(unsafe_receipt, port_probe=lambda port: True)

    def test_execute_selects_explicit_user_deferral_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profiles, paths = _fixture(root)
            args = _args()
            args.user_deferral_receipt = root / "deferral.json"
            args.user_deferral_sha256 = "a" * 64
            evidence = {
                "gate_kind": "user_requested_model_deferral",
                "gemma_primary_complete": False,
                "deferred_models": ["gemma4-style12-q5_k_m"],
            }
            with mock.patch.object(
                followups, "validate_user_deferral", return_value=evidence
            ) as validate, mock.patch.object(
                followups, "wait_for_first_wave"
            ) as normal_gate, mock.patch.object(
                followups,
                "wait_for_atomic_file",
                side_effect=followups.FollowupError("stop after gate"),
            ):
                receipt, code = followups.execute_cohort(
                    args, profiles=profiles, paths=paths, port_probe=lambda port: True
                )
            validate.assert_called_once()
            normal_gate.assert_not_called()
            self.assertEqual(receipt["first_wave_gate"], evidence)
            self.assertEqual(receipt["problem"]["error"], "stop after gate")
            self.assertEqual(code, 2)

    def test_bonsai_receipt_publishes_only_strict_sm75_templates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profiles, paths = _fixture(root)
            bonsai = profiles[:2]
            templates = {}
            shards = []
            for profile in bonsai:
                templates[profile.quantization] = {
                    "schema_version": 1,
                    "server": str(root / "llama-server"),
                    "model": str(profile.model_path),
                    "model_name": profile.key,
                    "context": 8192,
                    "host": "127.0.0.1",
                    "port": 18081,
                    "artifact_manifest": os.path.relpath(paths.bonsai_receipt, profile.config.parent),
                    "lib_dirs": [str(root / "lib")],
                    "args": [
                        "--gpu-layers", "all", "--fit", "off", "--parallel", "1",
                        "--jinja", "--no-webui", "--no-slots",
                    ],
                }
                shards.append({
                    "path": str(profile.model_path),
                    "size": profile.model_size,
                    "sha256": profile.model_sha256,
                    "verified": True,
                })
            receipt = {
                "verdict": {"artifact_complete": True, "static_tesla_t4_compatibility": True},
                "runtime": {
                    "selected_artifact": "source_build_cuda13_sm75",
                    "cuda_backend": {"fatbinary": {
                        "tesla_t4_sm75_present": True,
                        "architectures": [75],
                    }},
                    "prebuilt_candidate": {"fatbinary": {"tesla_t4_sm75_present": False}},
                },
                "shards": shards,
                "trial_config_templates": templates,
                "template_capabilities": {
                    "chat_template_sha256": followups.BONSAI_TEMPLATE_SHA256,
                    "enable_thinking_false_supported": True,
                    "enable_thinking_false_behavior": "empty_think_block",
                },
            }
            paths.bonsai_receipt.write_text(json.dumps(receipt), encoding="utf-8")
            commands, prepared = followups.ensure_bonsai_prepared(bonsai, paths, 1.0)
            self.assertEqual(commands, [])
            self.assertEqual(prepared["runtime_gate"], "source_build_cuda13_sm75")
            for profile in bonsai:
                document = json.loads(profile.config.read_text(encoding="utf-8"))
                self.assertEqual(document["args"][:4], ["--gpu-layers", "all", "--fit", "off"])
            unsupported = json.loads(json.dumps(receipt))
            unsupported["template_capabilities"]["enable_thinking_false_supported"] = False
            with self.assertRaises(followups.FollowupError):
                followups.validate_bonsai_receipt(
                    unsupported, bonsai, paths.bonsai_receipt
                )

    def test_high_budget_profiles_are_unique_and_flash_context_is_derived(self):
        parser = followups.build_parser()
        args = parser.parse_args([
            "--run-prefix", "specialists-highcap", "--context", "131072",
            "--code-output-cap", "65536", "--prose-output-cap", "65536",
            "--code-timeout", "21600", "--prose-timeout", "21600",
            "--max-window-seconds", "86400",
        ])
        profiles = followups.profiles_for_args(args)
        self.assertTrue(all("specialists-highcap" in str(item.config) for item in profiles))
        self.assertEqual(profiles[0].plan, args.standard_plan)
        source = {
            "args": ["--pack", "/pack", "--max-context", "8192", "--suffix-draft", "0"],
            "log": "/old.log",
        }
        derived = followups.flash_context_config_document(source, 131072, profiles[2].tag)
        index = derived["args"].index("--max-context")
        self.assertEqual(derived["args"][index + 1], "131072")
        self.assertEqual(source["args"][index + 1], "8192")
        self.assertIn(profiles[2].tag, derived["log"])
        command = followups.study_command(profiles[2], Path("study.py"), args)
        self.assertEqual(command[command.index("--code-output-cap") + 1], "65536")
        self.assertEqual(command[command.index("--prose-output-cap") + 1], "65536")
        self.assertEqual(command[command.index("--prose-timeout") + 1], "21600")
        self.assertEqual(command[command.index("--max-seconds") + 1], "86400")
        with self.assertRaises(SystemExit):
            parser.parse_args(["--max-window-seconds", "86401"])
        with self.assertRaises(SystemExit):
            parser.parse_args(["--context", "131073"])

    def test_wide_bonsai_clone_sets_fit_context_floor(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profiles, paths = _fixture(root)
            templates = {}
            for profile in profiles[:2]:
                templates[profile.key] = {
                    "context": 8192,
                    "args": ["--gpu-layers", "all", "--fit", "off"],
                }
            with mock.patch.object(
                followups, "validate_bonsai_receipt", return_value=templates
            ), mock.patch.object(followups, "sha256_file", return_value="a" * 64):
                paths.bonsai_receipt.write_text("{}", encoding="utf-8")
                _, evidence = followups.ensure_bonsai_prepared(
                    profiles[:2], paths, 1.0, context=98_304
                )
            for profile in profiles[:2]:
                document = json.loads(profile.config.read_text(encoding="utf-8"))
                index = document["args"].index("--fit-ctx")
                self.assertEqual(document["args"][index + 1], "98304")
                self.assertEqual(evidence["configs"][profile.key]["context"], 98_304)

    def test_model_boundary_gate_is_opt_in_and_forwards_study_identity(self):
        parser = followups.build_parser()
        self.assertIsNone(parser.parse_args([]).model_boundary_gates)
        args = parser.parse_args([
            "--model-boundary-gates", "/tmp/gates",
            "--model-boundary-wait-seconds", "86400",
        ])
        profile = followups.profiles_for_args(args)[0]
        study = {"path": "/tmp/study.json", "sha256": "b" * 64}
        with mock.patch.object(
            followups.model_boundary_gate, "publish_and_wait", return_value={"accepted": True}
        ) as publish:
            result = followups.wait_at_model_boundary(args, profile, study)
        self.assertEqual(result, {"accepted": True})
        publish.assert_called_once_with(
            Path("/tmp/gates"), "specialists", profile.key, study, 86400.0,
        )
        with self.assertRaises(SystemExit):
            parser.parse_args(["--model-boundary-wait-seconds", "86401"])

    def test_sampler_plan_is_forwarded_only_when_explicit(self):
        parser = followups.build_parser()
        legacy = parser.parse_args([])
        profile = followups.profiles_for_args(legacy)[0]
        self.assertNotIn("--sampler-plan", followups.study_command(profile, Path("study.py"), legacy))

        opted_in = parser.parse_args(["--sampler-plan", "publisher-plan.json"])
        profile = followups.profiles_for_args(opted_in)[0]
        command = followups.study_command(profile, Path("study.py"), opted_in)
        self.assertEqual(command[command.index("--sampler-plan") + 1], "publisher-plan.json")
        standard_only = parser.parse_args(["--standard-only"])
        profile = followups.profiles_for_args(standard_only)[0]
        self.assertIn(
            "--standard-only",
            followups.study_command(profile, Path("study.py"), standard_only),
        )

    def test_semantic_failures_advance_in_strict_sequence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profiles, paths = _fixture(root)
            order = []

            def fake_study(profile, args, used_paths, port_probe):
                order.append(profile.key)
                status = "completed_with_evaluation_failures" if len(order) != 2 else "complete"
                process = {
                    "kind": f"study_{profile.key}",
                    "argv": [profile.key],
                    "environment": dict(profile.environment),
                    "returncode": 1 if status != "complete" else 0,
                    "elapsed_seconds": 1.0,
                }
                return process, {
                    "path": str(root / f"{profile.key}.json"),
                    "sha256": "f" * 64,
                    "summary": {"status": status},
                    "ports_clear_after": {"18080": True, "18081": True},
                }

            bonsai_prepared = {
                "receipt": str(paths.bonsai_receipt),
                "receipt_sha256": "a" * 64,
                "configs": {
                    profile.key: {"path": str(profile.config), "sha256": "b" * 64}
                    for profile in profiles[:2]
                },
                "runtime_gate": "source_build_cuda13_sm75",
            }
            flash_prepared = {
                "receipt": str(paths.flash_receipt),
                "receipt_sha256": "c" * 64,
                "config": str(profiles[2].config),
                "config_sha256": "d" * 64,
                "mtp_enabled": False,
                "reference_environment": {"STRATA_ROUTE_TAIL_SKIP": "0"},
            }
            with mock.patch.object(followups, "wait_for_atomic_file", return_value={"size": 1}), mock.patch.object(
                followups, "ensure_bonsai_prepared", return_value=([], bonsai_prepared)
            ), mock.patch.object(
                followups, "ensure_flash_prepared", return_value=([], flash_prepared)
            ), mock.patch.object(followups, "run_study", side_effect=fake_study):
                receipt, code = followups.execute_cohort(
                    _args(), profiles=profiles, paths=paths, port_probe=lambda port: True
                )
            self.assertEqual(order, [profile.key for profile in profiles])
            self.assertEqual(code, 1)
            self.assertEqual(receipt["status"], "completed_with_evaluation_failures")
            self.assertEqual(
                receipt["models"][2]["controls"]["environment"],
                {"STRATA_ROUTE_TAIL_SKIP": "0"},
            )

    def test_flash_study_applies_tail_skip_environment_to_owned_process(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profiles, paths = _fixture(root)
            flash = profiles[2]
            receipt_path = paths.results / f"{flash.tag}-study-trial.json"
            study_receipt = _safe_study(flash.tag, port=18080)
            study_receipt["source_config"] = {
                "path": str(flash.config),
                "sha256": hashlib.sha256(flash.config.read_bytes()).hexdigest()
                if flash.config.exists() else None,
            }
            flash.config.write_text("{}", encoding="utf-8")
            study_receipt["source_config"]["sha256"] = hashlib.sha256(
                flash.config.read_bytes()
            ).hexdigest()
            study_receipt["plan"] = {
                "path": str(flash.plan),
                "sha256": hashlib.sha256(flash.plan.read_bytes()).hexdigest(),
            }
            receipt_path.write_text(json.dumps(study_receipt), encoding="utf-8")
            (paths.results / f"{flash.tag}-runtime-events.jsonl").write_text(
                json.dumps({
                    "event": "session_started",
                    "runtime_backend": "strata",
                    "engine_environment": {"STRATA_ROUTE_TAIL_SKIP": "0"},
                }) + "\n",
                encoding="utf-8",
            )
            seen = {}

            def fake_run(argv, timeout, kind, environment=None):
                seen.update(environment or {})
                return {
                    "kind": kind,
                    "argv": list(argv),
                    "environment": dict(environment or {}),
                    "returncode": 0,
                    "elapsed_seconds": 1.0,
                }

            with mock.patch.object(followups, "run_command", side_effect=fake_run):
                process, study = followups.run_study(
                    flash, _args(), paths, port_probe=lambda port: True
                )
            self.assertEqual(seen, {"STRATA_ROUTE_TAIL_SKIP": "0"})
            self.assertEqual(process["environment"], seen)
            self.assertTrue(all(study["ports_clear_after"].values()))
            self.assertEqual(
                study["environment_evidence"]["engine_environment"],
                {"STRATA_ROUTE_TAIL_SKIP": "0"},
            )

    def test_unknown_infrastructure_failure_stops_without_advancing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profiles, paths = _fixture(root)
            calls = []
            bonsai_prepared = {
                "receipt": str(paths.bonsai_receipt),
                "receipt_sha256": "a" * 64,
                "configs": {
                    profile.key: {"path": str(profile.config), "sha256": "b" * 64}
                    for profile in profiles[:2]
                },
                "runtime_gate": "source_build_cuda13_sm75",
            }

            def fail_first(profile, args, used_paths, port_probe):
                calls.append(profile.key)
                raise followups.FollowupError("unknown infrastructure failure")

            with mock.patch.object(followups, "wait_for_atomic_file", return_value={"size": 1}), mock.patch.object(
                followups, "ensure_bonsai_prepared", return_value=([], bonsai_prepared)
            ), mock.patch.object(followups, "run_study", side_effect=fail_first):
                receipt, code = followups.execute_cohort(
                    _args(), profiles=profiles, paths=paths, port_probe=lambda port: True
                )
            self.assertEqual(calls, [profiles[0].key])
            self.assertEqual(code, 2)
            self.assertEqual(receipt["status"], "infrastructure_failed")
            self.assertEqual(len(receipt["models"]), 1)


if __name__ == "__main__":
    unittest.main()
