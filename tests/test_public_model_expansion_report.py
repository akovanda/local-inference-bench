import csv
import hashlib
import json
from pathlib import Path

import pytest

from runtime import public_model_expansion_report as report


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def fixture_tree(tmp_path: Path, *, strict_prompt="Follow exactly."):
    manifest_path = tmp_path / "manifest.json"
    cases = [
        {"suite": "humanevalplus", "task_id": "HumanEval/0", "prompt": "def zero():\n", "original": {}},
        {"suite": "humanevalplus", "task_id": "HumanEval/1", "prompt": "def one():\n", "original": {}},
        {"suite": "ifeval", "task_id": "ifeval-7", "prompt": "Follow exactly.", "original": {}},
    ]
    write_json(manifest_path, {"schema_version": 1, "selection_seed": 1,
                               "selected_count_per_suite": 1, "cases": cases})
    manifest_sha = report.sha256_file(manifest_path)
    catalog_path = tmp_path / "catalog.json"
    write_json(catalog_path, {
        "schema_version": 1,
        "profiles": {
            "model-a": {
                "display_name": "Unsafe <script>alert(1)</script>",
                "model_name": "model-a.gguf",
                "runtime_backend": "llama.cpp",
                "profile_tag": "matched-medium",
                "parameters": {
                    "context_tokens": 73728,
                    "max_output_tokens": 65536,
                    "request_timeout_seconds": 21600,
                    "seed": 20261023,
                    "request_options": {"reasoning_effort": "medium", "temperature": 0},
                    "placement": {"gpu_layers": 99},
                },
            }
        },
    })
    study = tmp_path / "study"
    output = study / "profiles/model-a"
    settings_sha = "a" * 64
    write_json(study / "supervisor-status.json", {
        "schema_version": 1, "status": "running", "manifest_sha256": manifest_sha,
        "catalog_sha256": report.sha256_file(catalog_path),
        "models": {"model-a": {"key": "model-a", "state": "running",
                                  "output_dir": str(output),
                                  "grading_summary": str(output / "grading/summary.json")}},
    })
    write_json(output / "run.json", {
        "manifest": {"sha256": manifest_sha}, "settings_sha256": settings_sha,
        "settings": {"model": "model-a", "max_tokens": 65536,
                     "request_timeout_seconds": 21600, "seed": 20261023},
    })

    def result(task_id, suite, attempt, status, *, wall, completion, rate=None,
               predicted_ms=None, truncated=False):
        return {
            "manifest_sha256": manifest_sha, "settings_sha256": settings_sha,
            "case": {"task_id": task_id, "suite": suite}, "attempt": attempt,
            "status": status, "truncated": truncated,
            "finish_reason": "length" if truncated else "stop",
            "counts": {"prompt_tokens": 5, "completion_tokens": completion,
                       "total_tokens": 5 + completion, "reasoning_tokens": 2},
            "timing": {"wall_seconds": wall, "ttft_seconds": wall / 10,
                       "ttft_final_seconds": wall / 5},
            "native_timings": {"predicted_per_second": rate, "predicted_ms": predicted_ms,
                               "predicted_n": completion, "prompt_per_second": 100,
                               "prompt_ms": 50, "prompt_n": 5},
        }

    def attempt(task_id, suite, number, value):
        sid = hashlib.sha256(task_id.encode()).hexdigest()[:16]
        write_json(output / f"cases/{suite}/{sid}/attempt-{number:04d}/result.json", value)

    attempt("HumanEval/0", "humanevalplus", 1,
            result("HumanEval/0", "humanevalplus", 1, "api_failure", wall=3, completion=2))
    attempt("HumanEval/0", "humanevalplus", 2,
            result("HumanEval/0", "humanevalplus", 2, "completed", wall=4,
                   completion=10, rate=10, predicted_ms=1000))
    # This later completion must remain retained as cost, never replace attempt 2.
    attempt("HumanEval/0", "humanevalplus", 3,
            result("HumanEval/0", "humanevalplus", 3, "completed", wall=5,
                   completion=100, rate=100, predicted_ms=1000))
    attempt("HumanEval/1", "humanevalplus", 1,
            result("HumanEval/1", "humanevalplus", 1, "completed", wall=9,
                   completion=60, rate=20, predicted_ms=3000))
    attempt("ifeval-7", "ifeval", 1,
            result("ifeval-7", "ifeval", 1, "truncated", wall=8,
                   completion=65536, rate=8, predicted_ms=8000, truncated=True))

    strict_path = output / "grading/ifeval/hash/official-output/eval_results_strict.jsonl"
    write_jsonl(strict_path, [{"prompt": strict_prompt, "follow_all_instructions": True,
                               "follow_instruction_list": [True, False]}])
    write_json(output / "grading/summary.json", {
        "status": "running",
        "evalplus": {"status": "running", "per_task": [
            {"task_id": "HumanEval/0", "base_status": "pass", "plus_status": "pass"}
        ]},
        "ifeval": {"status": "running", "strict_results": str(strict_path)},
    })
    return study, catalog_path, manifest_path


def generate_fixture(tmp_path: Path, **kwargs):
    study, catalog, manifest = fixture_tree(tmp_path, **kwargs)
    return report.generate(study_dir=study, output_dir=study / "report",
                           catalog_path=catalog, manifest_path=manifest,
                           expected_manifest_sha256=None, include_original_controls=False)


def test_grader_joins_keep_partial_unknown_and_map_ifeval_by_exact_prompt(tmp_path):
    generated = generate_fixture(tmp_path)
    human = next(row for row in generated["summaries"] if row["suite"] == "humanevalplus")
    instructions = next(row for row in generated["summaries"] if row["suite"] == "ifeval")

    assert human["terminal_cases"] == 2
    assert human["quality_passes"] == human["quality_denominator"] == 1
    assert human["graded_unknown_cases"] == 1
    assert human["humaneval_base_passes"] == human["humaneval_base_graded"] == 1
    assert instructions["quality_passes"] == instructions["quality_denominator"] == 1
    assert instructions["ifeval_strict_instruction_passes"] == 1
    assert instructions["ifeval_strict_instruction_total"] == 2
    assert instructions["ifeval_loose_prompt_graded"] == 0


def test_ifeval_grade_rejects_order_or_fuzzy_prompt_join(tmp_path):
    study, catalog, manifest = fixture_tree(tmp_path, strict_prompt="Almost the same prompt.")
    with pytest.raises(report.ReportError, match="exactly match frozen manifest"):
        report.generate(study_dir=study, output_dir=study / "report", catalog_path=catalog,
                        manifest_path=manifest, expected_manifest_sha256=None,
                        include_original_controls=False)


def test_first_terminal_wins_retries_cost_and_caps_count_only_selected_terminal(tmp_path):
    generated = generate_fixture(tmp_path)
    rows = [row for row in generated["details"] if row["case_id"] == "HumanEval/0"]
    assert [row["attempt_number"] for row in rows if row["selected_for_benchmark"]] == [2]
    assert next(row for row in rows if row["attempt_number"] == 3)["selected_for_benchmark"] is False

    human = next(row for row in generated["summaries"] if row["suite"] == "humanevalplus")
    instructions = next(row for row in generated["summaries"] if row["suite"] == "ifeval")
    assert human["selected_completion_tokens"] == 70
    assert human["all_request_completion_tokens"] == 172
    assert human["all_request_wall_seconds"] == 21
    assert human["cap_cases"] == 0
    assert instructions["cap_cases"] == 1
    assert instructions["terminal_cases"] == 1


def test_generation_tps_is_predicted_ms_weighted_and_html_is_escaped(tmp_path):
    generated = generate_fixture(tmp_path)
    human = next(row for row in generated["summaries"] if row["suite"] == "humanevalplus")
    assert human["weighted_server_generation_tps"] == pytest.approx(17.5)
    assert human["weighted_server_prompt_tps"] == pytest.approx(100)
    html_text = (generated["output_dir"] / "report.html").read_text(encoding="utf-8")
    assert "Unsafe &lt;script&gt;alert(1)&lt;/script&gt;" in html_text
    assert "Unsafe <script>alert(1)</script>" not in html_text
    assert "matched T0/medium baseline" in html_text

    with (generated["output_dir"] / "detailed-cases.csv").open(newline="", encoding="utf-8") as stream:
        csv_rows = list(csv.DictReader(stream))
    assert len(csv_rows) == 5
    assert sum(row["selected_for_benchmark"] == "true" for row in csv_rows) == 3


def test_missing_result_is_pending_not_error_and_report_remains_partial(tmp_path):
    study, catalog, manifest = fixture_tree(tmp_path)
    pending_id = hashlib.sha256("HumanEval/1".encode()).hexdigest()[:16]
    result_path = study / f"profiles/model-a/cases/humanevalplus/{pending_id}/attempt-0001/result.json"
    result_path.unlink()
    generated = report.generate(study_dir=study, output_dir=study / "report",
                                catalog_path=catalog, manifest_path=manifest,
                                expected_manifest_sha256=None, include_original_controls=False)
    human = next(row for row in generated["summaries"] if row["suite"] == "humanevalplus")
    assert human["terminal_cases"] == 1
    assert human["pending_cases"] == 1
    assert human["error_cases"] == 0
    assert generated["status"]["phase"] == "partial"


def test_public_mode_rejects_a_referenced_grade_file_that_is_missing(monkeypatch, tmp_path):
    class MissingGradeEvidence:
        def resolve(self, value, *, bases=(), required=False):
            assert required is True
            raise report.public_evidence.EvidenceError("required evidence reference is missing")

    monkeypatch.setattr(report, "_ACTIVE_EVIDENCE", MissingGradeEvidence())
    with pytest.raises(report.ReportError, match="required evidence reference is missing"):
        report._official_path(
            "results/study/grading/strict.jsonl", tmp_path / "summary.json",
            tmp_path / "profile", tmp_path / "study",
        )
