import csv
from collections import Counter

import pytest

from runtime import model_evidence_report_20261023 as report


def t4_rows(model_key):
    return report.public_rows(
        report.PUBLIC_T4,
        model_key,
        study_family="public_benchmark_t4_64case",
        hardware="NVIDIA T4",
        context_tokens=73728,
        output_tokens=65536,
        subset_note="frozen selected32+32 public subset",
    )


def test_public_grades_join_by_task_and_prompt_with_distinct_definitions():
    oss = t4_rows("oss20b")
    lfm = t4_rows("lfm25")

    assert len(oss) == len(lfm) == 64
    for rows, expected in ((oss, (30, 26)), (lfm, (31, 31))):
        code = [row for row in rows if row["suite"] == "humanevalplus"]
        instructions = [row for row in rows if row["suite"] == "ifeval"]
        assert {row["success_definition"] for row in code} == {
            "evalplus_enhanced_plus_per_task_pass"
        }
        assert {row["success_definition"] for row in instructions} == {
            "ifeval_strict_prompt_pass"
        }
        assert sum(row["pass"] is True for row in code) == expected[0]
        assert sum(row["pass"] is True for row in instructions) == expected[1]

    mechanical = next(
        row for row in lfm
        if row["suite"] == "ifeval" and row["case_id"] == "3071"
    )
    assert mechanical["pass"] is False


def test_grouping_uses_predicted_ms_weighted_server_rate_and_keeps_unknown():
    base = {
        "model": "fixture", "profile": "p", "suite": "routing",
        "study_family": "family", "hardware": "cpu",
        "success_definition": "exact",
    }
    rows = [
        {**base, "pass": True, "capped": False, "output_tokens": 10,
         "native_generation_tps": 10.0, "native_predicted_ms": 1000.0,
         "wall_seconds": 1.0},
        {**base, "pass": None, "capped": True, "output_tokens": 30,
         "native_generation_tps": 20.0, "native_predicted_ms": 3000.0,
         "wall_seconds": 9.0},
    ]

    grouped = report.group_rows(rows)
    assert len(grouped) == 1
    assert grouped[0]["weighted_native_tps"] == pytest.approx(17.5)
    assert grouped[0]["median_wall_seconds"] == pytest.approx(5.0)
    assert grouped[0]["p95_wall_seconds"] == pytest.approx(9.0)
    assert grouped[0]["pass"] == 1
    assert grouped[0]["unknown"] == 1
    assert grouped[0]["caps"] == 1


def test_generate_has_exact_nonoverlapping_families_and_separate_evidence(tmp_path):
    generated = report.generate(tmp_path)
    rows = generated["rows"]
    counts = Counter(row["study_family"] for row in rows)

    assert len(rows) == 817
    assert len({row["record_id"] for row in rows}) == 817
    assert counts == {
        "legacy_complete_study_dashboard": 480,
        "public_benchmark_t4_64case": 128,
        "public_screen_tachyon_16case": 48,
        "tachyon_worker_production_schema_24": 120,
        "tachyon_worker_latency_ab": 24,
        "tachyon_worker_latency_greedy_diagnostic": 16,
        "tachyon_worker_latency_ordinary_default_probe": 1,
    }
    assert len(generated["manual"]) == 40
    assert len(generated["accelerator"]) == 6
    assert all(not item["included_in_817_stable_rows"]
               for item in generated["exclusions"])

    assert report.sha256(tmp_path / "legacy-complete-study-dashboard.csv") == report.sha256(
        report.LEGACY
    )
    with (tmp_path / "accelerator-evidence.csv").open(newline="", encoding="utf-8") as stream:
        accelerator = list(csv.DictReader(stream))
    failed_opencl = [row for row in accelerator
                     if row["backend"].startswith("opencl") and row["status"] == "failed"]
    assert len(failed_opencl) == 3
    assert all(row["generation_tps"] == "" for row in failed_opencl)

    html = (tmp_path / "report.html").read_text(encoding="utf-8")
    assert "Do not read this as one leaderboard" in html
    assert "final-paired-comparison.json" in html
    assert "official-grader-caveats.json" in html
    assert "Tachyon role study v1" in html
    assert "Gemma follow-up" in html


def test_t4_aggregate_metrics_come_from_detail_records():
    expected = {
        "oss20b": (340350, 3, 7.438757472554905),
        "lfm25": (121660, 0, 13.59204176310363),
    }
    for key, (tokens, caps, weighted_tps) in expected.items():
        rows = t4_rows(key)
        weighted = [
            (row["native_generation_tps"], row["native_predicted_ms"])
            for row in rows
            if row["native_generation_tps"] is not None
            and row["native_predicted_ms"] is not None
        ]
        actual_tps = sum(rate * milliseconds for rate, milliseconds in weighted) / sum(
            milliseconds for _, milliseconds in weighted
        )
        assert sum(row["output_tokens"] for row in rows) == tokens
        assert sum(row["capped"] is True for row in rows) == caps
        assert actual_tps == pytest.approx(weighted_tps)
