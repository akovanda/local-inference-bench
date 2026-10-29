# Validation scope

The portable CI selection tests contract validation, request capture, durable attempts and cancellation, exact task/prompt grading joins, partial denominators, cap handling, retry accounting, HTML escaping, and fixture-based report generation. It does not download a model or require a GPU:

```bash
python -m pytest -q \
  tests/test_bench.py \
  tests/test_hard_cases.py \
  tests/test_specialist_cases.py \
  tests/test_public_benchmark_pilot.py \
  tests/test_public_model_expansion_report.py \
  tests/test_final_report.py \
  tests/test_public_evidence.py \
  tests/test_runtime_import_safety.py
```

The CI workflow also compiles Python source and checks documented CLI entry points. Additional release evidence checks are documented in the README; run them after extracting the evidence archive.

The original lab environment collected 761 tests. Collection is not a claim that all tests passed. A targeted baseline before public packaging passed 41 tests with 16 subtests. These historical counts describe that environment and selection, not the current CI result. Current CI status is established by the workflow run for the exact commit.

`python -m pytest` defaults to the same source-only selection as CI. The gateway has a separate CI job and requires `.[test,gateway]`. After extracting the evidence and installing the gateway extra, `python -m pytest tests` runs the broader portable selection. Checks tied to original lab source/configuration pins are marked `historical`; checks needing excluded upstream code are marked `optional_upstream`. These guards deliberately reject modified historical receipts. Tests marked `evidence_integration` specifically exercise archive reconstruction. All three marker groups are excluded by default and can be selected explicitly with `-m historical`, `-m optional_upstream`, or `-m evidence_integration` once their documented prerequisites are available. Historical checks belong in the original lab, not in the sanitized export.

For a release, verify the evidence manifest, rebuild both public reports, compare their supported invariants, audit the export, and record the commands and outcomes in the release notes. These checks validate transformations of saved observations; they do not reproduce inference or establish fresh model-quality measurements.

## Initial release verification

On October 29, 2026, the release export was checked with Python 3.12. A fresh environment installed the package with its test and gateway extras. The default fixture selection passed 100 tests; the earlier combined fixture/gateway run passed 107 tests with one optional SDK test skipped. The broader archived-evidence suite passed 710 tests and 194 subtests, with 51 original-lab/upstream checks deselected. Two dedicated evidence integration tests passed separately.

Both public reports rebuilt from 3,017 verified manifest entries. Comparison preserved all 817 cross-study rows, 216 grouped rows, 704 expansion detail rows (including baseline references), and 22 expansion summaries. Expansion status remained 168 terminal cases, 407 pending, and one error. All 32 local links in rebuilt reports resolved. The publication scan found no configured secret patterns or private host identifiers; it is a pattern scan, not a proof that arbitrary content can never contain sensitive information.
