import csv
from html.parser import HTMLParser
from pathlib import Path

import pytest

from runtime import model_evidence_report_20261023 as cross_report
from runtime import public_model_expansion_report as expansion_report


pytestmark = pytest.mark.evidence_integration
ROOT = Path(__file__).resolve().parents[1]


class _Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.hrefs = []

    def handle_starttag(self, tag, attrs):
        if tag == "a" and (href := dict(attrs).get("href")):
            self.hrefs.append(href)


def _assert_local_links_exist(report: Path):
    parser = _Links()
    parser.feed(report.read_text(encoding="utf-8"))
    for href in parser.hrefs:
        if "://" not in href and not href.startswith("#"):
            assert (report.parent / href.split("#", 1)[0]).resolve().exists(), href


def test_relocated_cross_study_rebuild_preserves_817_rows(tmp_path):
    generated = cross_report.generate(tmp_path / "cross-study", evidence_root=ROOT)
    assert len(generated["rows"]) == 817
    assert generated["provenance"]["evidence_manifest"]["sha256"]
    with (tmp_path / "cross-study/evidence.csv").open(newline="", encoding="utf-8") as stream:
        assert sum(1 for _ in csv.DictReader(stream)) == 817
    html = (tmp_path / "cross-study/report.html").read_text(encoding="utf-8")
    assert "2026-10-24 investigation supersedes those conclusions" in html
    _assert_local_links_exist(tmp_path / "cross-study/report.html")


def test_relocated_partial_expansion_preserves_progress_and_unknown_grades(tmp_path):
    generated = expansion_report.generate(
        output_dir=tmp_path / "expansion", evidence_root=ROOT,
    )
    status = generated["status"]
    assert status["queue_cases_terminal"] == 168
    assert status["queue_cases_pending"] == 407
    assert status["queue_cases_error"] == 1
    assert status["selection_identity_sha256"]
    assert sum(row["graded_unknown_cases"] for row in generated["summaries"]) > 0
    _assert_local_links_exist(tmp_path / "expansion/report.html")

