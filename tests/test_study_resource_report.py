import csv
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from runtime import study_resource_report


def write_jsonl(path, rows, *, trailing=""):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows) + trailing, encoding="utf-8")


def event(event_name, at, tag="model-a", run_id="run-a", **fields):
    return {"event": event_name, "at": at, "tag": tag, "run_id": run_id, **fields}


def lifecycle(tag, run_id, pid, start, stop=None, *, terminal=None, context=49152):
    rows = [
        event("session_started", start, tag, run_id, runtime_backend="llama", context=context,
              model_name=f"{tag}-model"),
        event("server_started", start, tag, run_id, pid=pid),
        event("server_ready", start, tag, run_id, initial_load_seconds=12.5),
    ]
    if stop:
        rows.append(event("server_stopped", stop, tag, run_id,
                          stopped=True, term_sent=True, kill_sent=False))
    if terminal:
        rows.append(event(terminal, stop or start, tag, run_id, status=terminal))
    return rows


def gpu(*, util=50, power=40, sm=600, mem=5000, temp=45, brake="Active"):
    return {
        "utilization.gpu": str(util),
        "power.draw": str(power),
        "clocks.sm": str(sm),
        "clocks.mem": str(mem),
        "temperature.gpu": str(temp),
        "clocks_event_reasons.hw_power_brake_slowdown": brake,
    }


def old_sample(at, pid, vram, **gpu_fields):
    return {"at": at, "gpu": gpu(**gpu_fields),
            "gpu_processes": [{"pid": str(pid), "used_mib": str(vram)}]}


def new_sample(at, pid, vram, **gpu_fields):
    return {"at": at, "gpu": [gpu(**gpu_fields)],
            "gpu_processes": [{"pid": str(pid), "used_gpu_memory": str(vram)}]}


GENERATED = datetime(2026, 10, 20, 1, 0, tzinfo=timezone.utc)


def write_receipt(path, receipt):
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def strata_child_fixture(
    root, *, tag="strata-a", run_id="run-strata-a", api_pid=5001, gpu_pid=6001,
    gpu_start_ticks=1100, receipt_version=1,
):
    engine = "/opt/strata/bin/strata"
    environment = {"STRATA_ROUTE_TAIL_SKIP": "0"}
    private = root / f"{tag}-config.json"
    source = root / f"{tag}-source.json"
    config = {"exe": engine, "reference_environment": environment}
    private.write_text(json.dumps(config, sort_keys=True) + "\n", encoding="utf-8")
    source.write_text(json.dumps(config, sort_keys=True) + "\n", encoding="utf-8")
    life = root / f"{tag}-runtime-events.jsonl"
    rows = [
        event(
            "session_started", "2026-10-20T00:00:10.500Z", tag, run_id,
            runtime_backend="strata", context=73728, model_name=f"{tag}-model",
            engine=engine, engine_environment=environment,
            private_config=str(private), source_config=str(source),
        ),
        event(
            "server_started", "2026-10-20T00:00:10.500Z", tag, run_id,
            runtime_backend="strata", pid=api_pid,
            argv=["python", "-m", "serve.server", "--config", str(private)],
        ),
        event("server_ready", "2026-10-20T00:00:12Z", tag, run_id,
              initial_load_seconds=1.5),
        event("server_stopped", "2026-10-20T00:00:20Z", tag, run_id,
              stopped=True, term_sent=True, kill_sent=False),
    ]
    write_jsonl(life, rows)
    raw_lines = life.read_bytes().splitlines()
    bindings = []
    for line, event_name in ((1, "session_started"), (2, "server_started")):
        value = json.loads(raw_lines[line - 1])
        canonical = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
        bindings.append({
            "event": event_name,
            "source_line": line,
            "source_raw_line_sha256": hashlib.sha256(raw_lines[line - 1]).hexdigest(),
            "source_row_canonical_sha256": hashlib.sha256(canonical).hexdigest(),
        })
    boot_time = int(datetime(2026, 10, 20, tzinfo=timezone.utc).timestamp())

    def file_binding(path):
        snapshot = path.read_bytes()
        return {"path": str(path), "sha256": hashlib.sha256(snapshot).hexdigest(),
                "size": len(snapshot)}

    receipt = {
        "api_process": {
            "exe": "/usr/bin/python3", "parent_pid": 4001, "pid": api_pid,
            "start_ticks": "1000", "state": "S",
        },
        "direct_child_parent_relation_verified": True,
        "engine_environment": environment,
        "expected_engine_exe": engine,
        "gpu_observation": {"name": engine, "pid": gpu_pid, "used_mib": 256},
        "gpu_process": {
            "exe": engine, "parent_pid": api_pid, "pid": gpu_pid,
            "start_ticks": str(gpu_start_ticks), "state": "S",
        },
        "kernel_clock": {
            "boot_id": "fixture-boot-id", "boot_time_resolution_seconds": 1,
            "boot_time_unix_seconds": boot_time, "clock_ticks_per_second": 100,
            "gpu_process_start_unix_seconds": boot_time + gpu_start_ticks / 100,
        },
        "kind": "strata_owned_gpu_process_observation",
        "lifecycle_events": bindings,
        "lifecycle_source": str(life),
        "live_kernel_start_identities_verified": True,
        "model_session_run_id": run_id,
        "observed_at": "2026-10-20T00:00:13Z",
        "observed_during_live_matching_model_session": True,
        "private_config": file_binding(private),
        "prose_bodies_read": False,
        "runtime_backend": "strata",
        "schema_version": 1,
        "scope": "fixture ownership only",
        "source_config": file_binding(source),
        "tag": tag,
    }
    receipt_path = root / f"{tag}-gpu-ownership-v{receipt_version}.json"
    write_receipt(receipt_path, receipt)
    return receipt_path, receipt


class StudyResourceReportTests(unittest.TestCase):
    def test_both_telemetry_formats_produce_sampled_stats_and_source_truth(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            life = root / "alpha-runtime-events.jsonl"
            write_jsonl(life, lifecycle("alpha", "run-1", 101,
                                        "2026-10-20T00:00:00Z", "2026-10-20T00:03:00Z"))
            old = root / "specialists-machine-telemetry.jsonl"
            write_jsonl(old, [
                old_sample("2026-10-20T00:00:00Z", 101, 100, util=20, power=30, brake="Active"),
                old_sample("2026-10-20T00:03:00Z", 101, 300, util=80, power=50, brake="Not Active"),
            ])
            newer = root / "specialists-machine-telemetry-part2.jsonl"
            write_jsonl(newer, [new_sample("2026-10-20T00:01:00Z", 101, 200,
                                               util=50, power=40, brake="Active")])

            report = study_resource_report.build_report(root, generated_at=GENERATED)

        server = report["servers"][0]
        self.assertEqual(server["join_status"], "joined")
        self.assertEqual(server["context_tokens"], 49152)
        self.assertEqual(server["initial_load_seconds"], 12.5)
        self.assertEqual(server["cleanup"], {"stopped": True, "term_sent": True, "kill_sent": False})
        self.assertEqual(server["cleanup_events"], [{
            "event": "server_stopped", "at": "2026-10-20T00:03:00.000Z",
            "booleans": {"stopped": True, "term_sent": True, "kill_sent": False},
        }])
        self.assertEqual(server["process_vram"]["sample_count"], 3)
        self.assertEqual(server["process_vram"]["min"], 100)
        self.assertEqual(server["process_vram"]["median"], 200)
        self.assertEqual(server["process_vram"]["sampled_max"], 300)
        self.assertEqual(server["process_vram"]["first_sample"]["at"], "2026-10-20T00:00:00.000Z")
        self.assertEqual(server["process_vram"]["last_sample"]["at"], "2026-10-20T00:03:00.000Z")
        self.assertEqual(server["gpu_metrics"]["utilization_percent"]["median"], 50)
        self.assertEqual(server["gpu_metrics"]["board_power_watts"]["sampled_max"], 50)
        self.assertEqual(server["hardware_power_brake"]["values"], ["Active", "Not Active"])
        self.assertEqual(server["hardware_power_brake"]["observations"], [
            {"value": "Active", "sample_count": 2},
            {"value": "Not Active", "sample_count": 1},
        ])
        self.assertTrue(all(source["sha256"] for group in report["sources"].values() for source in group))
        self.assertTrue(any("not summed" in note for note in report["coverage_notes"]))
        self.assertIn("not a true peak", " ".join(report["disclaimers"]))
        self.assertIn("neither model energy", " ".join(report["disclaimers"]))

    def test_pid_reuse_is_scoped_to_nonoverlapping_inclusive_intervals(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_jsonl(root / "one-runtime-events.jsonl", lifecycle(
                "one", "run-1", 44, "2026-10-20T00:00:00Z", "2026-10-20T00:01:00Z"))
            write_jsonl(root / "two-runtime-events.jsonl", lifecycle(
                "two", "run-2", 44, "2026-10-20T00:02:00Z", "2026-10-20T00:03:00Z"))
            write_jsonl(root / "specialists-machine-telemetry-part2.jsonl", [
                new_sample("2026-10-20T00:00:00Z", 44, 100),
                new_sample("2026-10-20T00:01:00Z", 44, 110),
                new_sample("2026-10-20T00:01:30Z", 44, 999),
                new_sample("2026-10-20T00:02:00Z", 44, 200),
                new_sample("2026-10-20T00:03:00Z", 44, 210),
            ])
            report = study_resource_report.build_report(root, generated_at=GENERATED)

        by_tag = {server["identity"]["tag"]: server for server in report["servers"]}
        self.assertEqual(by_tag["one"]["process_vram"]["sample_count"], 2)
        self.assertEqual(by_tag["one"]["process_vram"]["sampled_max"], 110)
        self.assertEqual(by_tag["two"]["process_vram"]["sample_count"], 2)
        self.assertEqual(by_tag["two"]["process_vram"]["min"], 200)

    def test_legacy_server_events_join_without_inventing_session_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_jsonl(root / "legacy-runtime-events.jsonl", [
                event("server_started", "2026-10-20T00:00:00Z", "legacy", "old-run", pid=45),
                event("server_stopped", "2026-10-20T00:01:00Z", "legacy", "old-run",
                      stopped=True, kill_sent=False),
            ])
            write_jsonl(root / "specialists-machine-telemetry.jsonl", [
                old_sample("2026-10-20T00:00:30Z", 45, 321)
            ])
            report = study_resource_report.build_report(root, generated_at=GENERATED)

        server = report["servers"][0]
        self.assertEqual(server["join_status"], "joined")
        self.assertEqual(server["process_vram"]["sampled_max"], 321)
        self.assertIsNone(server["context_tokens"])
        self.assertIsNone(server["runtime_backend"])
        self.assertIsNone(server["model_name"])
        self.assertTrue(any("metadata are unavailable" in note for note in server["notes"]))

    def test_ongoing_is_bounded_but_terminal_without_stop_is_not_joined(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_jsonl(root / "ongoing-runtime-events.jsonl", lifecycle(
                "ongoing", "run-live", 51, "2026-10-20T00:00:00Z"))
            terminal_rows = lifecycle("terminal", "run-done", 52, "2026-10-20T00:00:00Z")
            terminal_rows.append(event("session_failed", "2026-10-20T00:05:00Z",
                                       "terminal", "run-done", status="failed"))
            terminal_rows.append(event("server_cleanup_failed", "2026-10-20T00:05:01Z",
                                       "terminal", "run-done", stopped=False))
            write_jsonl(root / "terminal-runtime-events.jsonl", terminal_rows)
            write_jsonl(root / "absent-runtime-events.jsonl", lifecycle(
                "absent", "run-empty", 53, "2026-10-19T23:00:00Z", "2026-10-19T23:05:00Z"))
            write_jsonl(root / "specialists-machine-telemetry.jsonl", [
                old_sample("2026-10-20T00:10:00Z", 51, 500),
                old_sample("2026-10-20T00:02:00Z", 52, 600),
            ])
            report = study_resource_report.build_report(root, generated_at=GENERATED)

        by_tag = {server["identity"]["tag"]: server for server in report["servers"]}
        ongoing = by_tag["ongoing"]
        self.assertEqual(ongoing["interval_status"], "ongoing_bounded_by_generated_at")
        self.assertEqual(ongoing["interval_end_at"], "2026-10-20T01:00:00.000Z")
        self.assertEqual(ongoing["join_status"], "joined")
        terminal = by_tag["terminal"]
        self.assertEqual(terminal["interval_status"], "missing_stop_terminal_session")
        self.assertEqual(terminal["join_status"], "not_joined")
        self.assertEqual(terminal["coverage"]["interval_sample_count"], 0)
        self.assertEqual(terminal["cleanup_events"], [{
            "event": "server_cleanup_failed", "at": "2026-10-20T00:05:01.000Z",
            "booleans": {"stopped": False},
        }])
        absent = by_tag["absent"]
        self.assertEqual(absent["join_status"], "joined_without_usable_samples")
        self.assertIsNone(absent["process_vram"])
        self.assertTrue(all(value is None for value in absent["gpu_metrics"].values()))

    def test_contaminated_multi_gpu_and_multi_process_samples_are_excluded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_jsonl(root / "alpha-runtime-events.jsonl", lifecycle(
                "alpha", "run-a", 71, "2026-10-20T00:00:00Z", "2026-10-20T00:05:00Z"))
            good = new_sample("2026-10-20T00:01:00Z", 71, 700, util=10)
            multi_gpu = new_sample("2026-10-20T00:02:00Z", 71, 800, util=99)
            multi_gpu["gpu"].append(gpu(util=99))
            multi_process = new_sample("2026-10-20T00:03:00Z", 71, 900, util=99)
            multi_process["gpu_processes"].append({"pid": "72", "used_gpu_memory": "1"})
            wrong_pid = new_sample("2026-10-20T00:04:00Z", 72, 1000, util=99)
            write_jsonl(root / "specialists-machine-telemetry-part3.jsonl",
                        [good, multi_gpu, multi_process, wrong_pid])
            report = study_resource_report.build_report(root, generated_at=GENERATED)

        server = report["servers"][0]
        self.assertEqual(server["coverage"]["interval_sample_count"], 4)
        self.assertEqual(server["coverage"]["owned_pid_observed_sample_count"], 3)
        self.assertEqual(server["coverage"]["isolated_sample_count"], 1)
        self.assertEqual(server["coverage"]["excluded_sample_count"], 3)
        self.assertEqual(server["process_vram"]["sampled_max"], 700)
        self.assertEqual(server["gpu_metrics"]["utilization_percent"]["sampled_max"], 10)
        reasons = server["coverage"]["excluded_reasons"]
        self.assertEqual(reasons["gpu_record_count_not_one"], 1)
        self.assertEqual(reasons["gpu_process_count_not_one"], 1)
        self.assertEqual(reasons["sole_process_pid_mismatch"], 1)

    def test_malformed_partial_lines_and_ambiguous_lifecycle_are_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ambiguous = lifecycle("alpha", "same-run", 80,
                                  "2026-10-20T00:00:00Z", "2026-10-20T00:10:00Z")
            ambiguous.insert(2, event("server_started", "2026-10-20T00:00:01Z",
                                      "alpha", "same-run", pid=81))
            write_jsonl(root / "ambiguous-runtime-events.jsonl", ambiguous)
            telemetry = root / "specialists-machine-telemetry.jsonl"
            write_jsonl(telemetry, [old_sample("2026-10-20T00:01:00Z", 80, 100)],
                        trailing='{"at":"partial"')
            report = study_resource_report.build_report(root, generated_at=GENERATED)

        self.assertEqual(report["summary"]["malformed_or_non_object_line_count"], 1)
        self.assertEqual(report["servers"][0]["join_status"], "not_joined")
        self.assertTrue(any("server_started" in note for note in report["servers"][0]["notes"]))
        telemetry_source = report["sources"]["telemetry"][0]
        self.assertEqual(telemetry_source["invalid_record_count"], 1)
        self.assertEqual(telemetry_source["issues"][0]["line"], 2)

    def test_overlapping_same_pid_intervals_are_rejected_as_ambiguous(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_jsonl(root / "one-runtime-events.jsonl", lifecycle(
                "one", "run-one", 99, "2026-10-20T00:00:00Z", "2026-10-20T00:10:00Z"))
            write_jsonl(root / "two-runtime-events.jsonl", lifecycle(
                "two", "run-two", 99, "2026-10-20T00:05:00Z", "2026-10-20T00:15:00Z"))
            write_jsonl(root / "specialists-machine-telemetry.jsonl", [
                old_sample("2026-10-20T00:06:00Z", 99, 1000)
            ])
            report = study_resource_report.build_report(root, generated_at=GENERATED)

        self.assertTrue(all(server["join_status"] == "not_joined" for server in report["servers"]))
        self.assertTrue(all(any("overlapping lifecycle" in note for note in server["notes"])
                            for server in report["servers"]))

    def test_valid_strata_child_receipt_joins_child_and_retains_api_pid(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            receipt_path, receipt = strata_child_fixture(root)
            write_jsonl(root / "specialists-machine-telemetry-part3b.jsonl", [
                new_sample("2026-10-20T00:00:11.500Z", 6001, 50),
                new_sample("2026-10-20T00:00:12Z", 6001, 100, util=20),
                new_sample("2026-10-20T00:00:14Z", 6001, 300, util=60),
                new_sample("2026-10-20T00:00:15Z", 7001, 999, util=99),
            ])
            report = study_resource_report.build_report(root, generated_at=GENERATED)

        server = report["servers"][0]
        self.assertEqual(server["server_pid"], 5001)
        self.assertEqual(server["gpu_process_pid"], 6001)
        self.assertEqual(server["gpu_process_started_at"], "2026-10-20T00:00:11.000Z")
        self.assertEqual(server["gpu_process_start_time_resolution_seconds"], 1)
        self.assertEqual(server["join_status"], "joined")
        self.assertEqual(server["process_vram"]["sample_count"], 2)
        self.assertEqual(server["process_vram"]["sampled_max"], 300)
        self.assertEqual(server["coverage"]["interval_sample_count"], 3)
        self.assertEqual(server["coverage"]["owned_pid_observed_sample_count"], 2)
        self.assertEqual(server["coverage"]["excluded_reasons"]["sole_process_pid_mismatch"], 1)
        self.assertEqual(server["gpu_ownership"]["receipt_path"], str(receipt_path))
        self.assertEqual(
            server["gpu_ownership"]["gpu_process_start_ticks"],
            receipt["gpu_process"]["start_ticks"],
        )
        self.assertTrue(report["sources"]["gpu_ownership"][0]["accepted"])

    def test_missing_or_invalid_strata_child_proof_never_joins_child(self):
        mutations = {
            "row": lambda value: value["lifecycle_events"][0].update(source_line=99),
            "hash": lambda value: value["lifecycle_events"][1].update(
                source_raw_line_sha256="0" * 64
            ),
            "pid": lambda value: value["api_process"].update(pid=9999),
            "run": lambda value: value.update(model_session_run_id="wrong-run"),
            "ppid": lambda value: value["gpu_process"].update(parent_pid=9999),
            "config": lambda value: value["private_config"].update(sha256="0" * 64),
            "exe": lambda value: value.update(expected_engine_exe="/opt/strata/bin/wrong"),
            "timestamp": lambda value: value["kernel_clock"].update(
                gpu_process_start_unix_seconds=1
            ),
        }
        for label in ["missing", "receipt_symlink", "config_symlink", *mutations]:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                receipt_path, original = strata_child_fixture(root)
                if label == "missing":
                    receipt_path.unlink()
                elif label == "receipt_symlink":
                    target = root / "ownership-receipt-target.json"
                    receipt_path.rename(target)
                    receipt_path.symlink_to(target.name)
                elif label == "config_symlink":
                    private = Path(original["private_config"]["path"])
                    target = root / "private-config-target.json"
                    private.rename(target)
                    private.symlink_to(target.name)
                else:
                    changed = copy.deepcopy(original)
                    mutations[label](changed)
                    write_receipt(receipt_path, changed)
                write_jsonl(root / "specialists-machine-telemetry-part3b.jsonl", [
                    new_sample("2026-10-20T00:00:14Z", 6001, 300),
                ])
                report = study_resource_report.build_report(root, generated_at=GENERATED)

                server = report["servers"][0]
                self.assertEqual(server["server_pid"], 5001)
                self.assertIsNone(server["gpu_process_pid"])
                self.assertNotEqual(server["join_status"], "joined")
                self.assertIsNone(server["process_vram"])
                self.assertTrue(any("ownership receipt" in note for note in server["notes"]))
                if label != "missing":
                    source = report["sources"]["gpu_ownership"][0]
                    self.assertFalse(source["accepted"])
                    self.assertTrue(source["issues"])
                    self.assertTrue(any("receipt rejected" in note for note in report["coverage_notes"]))

    def test_duplicate_and_cross_identity_engine_ownership_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, receipt = strata_child_fixture(root)
            write_receipt(root / "strata-a-gpu-ownership-v2.json", receipt)
            write_jsonl(root / "specialists-machine-telemetry-part3b.jsonl", [
                new_sample("2026-10-20T00:00:14Z", 6001, 300),
            ])
            duplicate_report = study_resource_report.build_report(root, generated_at=GENERATED)

        duplicate_server = duplicate_report["servers"][0]
        self.assertIsNone(duplicate_server["gpu_process_pid"])
        self.assertNotEqual(duplicate_server["join_status"], "joined")
        self.assertTrue(any("multiple GPU ownership receipts" in note
                            for note in duplicate_server["notes"]))
        self.assertTrue(all(not source["accepted"]
                            for source in duplicate_report["sources"]["gpu_ownership"]))

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            strata_child_fixture(root, tag="strata-a", run_id="run-a", api_pid=5001,
                                 gpu_pid=6001, gpu_start_ticks=1100)
            strata_child_fixture(root, tag="strata-b", run_id="run-b", api_pid=5002,
                                 gpu_pid=6001, gpu_start_ticks=1100)
            write_jsonl(root / "specialists-machine-telemetry-part3b.jsonl", [
                new_sample("2026-10-20T00:00:14Z", 6001, 300),
            ])
            conflict_report = study_resource_report.build_report(root, generated_at=GENERATED)

        self.assertTrue(all(server["join_status"] == "not_joined"
                            for server in conflict_report["servers"]))
        self.assertTrue(all(any("claimed by multiple lifecycle identities" in note
                                for note in server["notes"])
                            for server in conflict_report["servers"]))
        self.assertTrue(all(not source["accepted"]
                            for source in conflict_report["sources"]["gpu_ownership"]))

    def test_real_strata_child_receipt_is_accepted_and_joins_persisted_telemetry(self):
        root = Path(study_resource_report.__file__).resolve().parents[1]
        report = study_resource_report.build_report(root / "results")
        tag = "specialists-final72k64k-flash-next-coder-iq1_m-strict-72k"
        server = next(item for item in report["servers"] if item["identity"]["tag"] == tag)

        self.assertEqual(server["server_pid"], 1777545)
        self.assertEqual(server["gpu_process_pid"], 1777616)
        self.assertEqual(server["join_status"], "joined")
        self.assertGreater(server["process_vram"]["sample_count"], 0)
        self.assertTrue(any(path.endswith("specialists-machine-telemetry-part3b.jsonl")
                            for path in server["telemetry_sources"]))
        source = next(item for item in report["sources"]["gpu_ownership"]
                      if item["identity"] and item["identity"]["tag"] == tag)
        self.assertTrue(source["accepted"])
        self.assertEqual(
            source["sha256"],
            "ea55f1ea9e5cbaea2f7ba7b26abe771b2454f6b24b85ed1641b304d603f24af2",
        )

    def test_outputs_are_exclusive_and_csv_uses_sampled_max_label(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_jsonl(root / "one-runtime-events.jsonl", lifecycle(
                "one", "run-one", 7, "2026-10-20T00:00:00Z", "2026-10-20T00:01:00Z"))
            write_jsonl(root / "specialists-machine-telemetry.jsonl", [
                old_sample("2026-10-20T00:00:30Z", 7, 123)
            ])
            report = study_resource_report.build_report(root, generated_at=GENERATED)
            prefix = root / "custom.report"
            json_path, csv_path = study_resource_report.write_report(report, prefix)
            with csv_path.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(json_path.name, "custom.report.json")
            self.assertEqual(csv_path.name, "custom.report.csv")
            self.assertEqual(rows[0]["process_vram_sampled_max_mib"], "123.0")
            self.assertNotIn("process_vram_peak_mib", rows[0])
            with self.assertRaises(study_resource_report.ReportInputError):
                study_resource_report.write_report(report, prefix)

    def test_help_is_offline_and_documents_explicit_output_prefix(self):
        script = Path(study_resource_report.__file__).resolve()
        result = subprocess.run([sys.executable, str(script), "--help"], capture_output=True,
                                text=True, check=False, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--output-prefix", result.stdout)
        self.assertIn("--results-dir", result.stdout)


if __name__ == "__main__":
    unittest.main()
