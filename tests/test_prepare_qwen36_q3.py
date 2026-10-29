import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from runtime import prepare_qwen36_q3 as q3


class Qwen36Q3PreparationTests(unittest.TestCase):
    def test_pending_report_records_llama_only_compatibility_decision(self):
        with tempfile.TemporaryDirectory() as directory:
            report = q3.artifact_report(Path(directory), False)
        self.assertFalse(report["verdict"]["artifact_complete"])
        self.assertFalse(report["verdict"]["current_strata_runnable"])
        self.assertFalse(report["strata_compatibility"]["pack_attempted"])
        self.assertIn("Q3_K", report["verdict"]["strata_blocking_error"])
        self.assertEqual(report["bounded_header_validation"]["layer_by_layer_parse_attempt"]["result"],
                         "rejected before start")
        self.assertEqual(report["shards"][0]["sha256"], q3.SHA256)
        self.assertFalse(report["llama_cpp"]["placement_experiment"]["fit_claimed"])

    def test_placement_config_only_removes_cpu_moe(self):
        baseline = json.loads(q3.CONFIG.read_text(encoding="utf-8"))
        placement = json.loads(q3.PLACEMENT_CONFIG.read_text(encoding="utf-8"))
        expected_args = baseline["args"].copy()
        expected_args.remove("--cpu-moe")
        self.assertEqual(placement["args"], expected_args)
        baseline["args"] = expected_args
        self.assertEqual(placement, baseline)

    def test_metadata_only_writes_valid_pending_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            destination = root / "model"
            report_path = root / "receipt.json"
            rc = q3.main([
                "--metadata-only",
                "--destination",
                str(destination),
                "--report",
                str(report_path),
            ])
            self.assertEqual(rc, 0)
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["state"], "pending_download_or_verification")
            self.assertFalse(report["verdict"]["llama_cpp_runnable"])

    def test_stage_rejects_more_than_four_workers_before_downloading(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(q3.range_fetch, "download_file") as download:
            with self.assertRaisesRegex(ValueError, "capped at four"):
                q3.stage(Path(directory), Path(directory) / "receipt.json", 5, 32, 5, False)
            download.assert_not_called()


if __name__ == "__main__":
    unittest.main()
