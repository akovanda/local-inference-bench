import hashlib
import json
from pathlib import Path

import pytest

from runtime import public_evidence


def _write_manifest(root: Path, files: dict[str, bytes], **extra) -> Path:
    records = []
    for relative, data in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        digest = hashlib.sha256(data).hexdigest()
        records.append({
            "path": relative, "sha256": digest, "size": len(data),
            "original_sha256": digest, "original_size": len(data),
        })
    manifest = {
        "schema_version": 1, "transformation_version": 1,
        "files": records, **extra,
    }
    path = root / "evidence-manifest.json"
    path.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")
    return path


def test_verifies_bytes_and_resolves_only_inside_relocated_root(tmp_path, monkeypatch):
    root = tmp_path / "relocated"
    root.mkdir()
    _write_manifest(root, {
        "results/study/summary.json": b'{"grade":"known"}\n',
        "results/study/grading/strict.jsonl": b'{"pass":true}\n',
    })
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    monkeypatch.chdir(unrelated)

    verified = public_evidence.verify_manifest(
        root, required_paths=("results/study/summary.json",)
    )
    assert verified.resolve("results/study/summary.json", required=True) == (
        root / "results/study/summary.json"
    )
    assert verified.resolve(
        "grading/strict.jsonl", bases=(root / "results/study",), required=True
    ) == root / "results/study/grading/strict.jsonl"
    assert verified.resolve("/historical/private/results.json") is None
    with pytest.raises(public_evidence.EvidenceError, match="outside the archive"):
        verified.resolve("/historical/private/results.json", required=True)


@pytest.mark.parametrize("mutation", ["corrupt", "missing"])
def test_verification_rejects_corrupt_or_missing_published_bytes(tmp_path, mutation):
    root = tmp_path / "release"
    root.mkdir()
    _write_manifest(root, {"results/value.json": b'{"value":1}\n'})
    evidence = root / "results/value.json"
    if mutation == "corrupt":
        evidence.write_bytes(b'{"value":2}\n')
        match = "SHA-256 differs"
    else:
        evidence.unlink()
        match = "is missing"
    with pytest.raises(public_evidence.EvidenceError, match=match):
        public_evidence.verify_manifest(root)


def test_verification_rejects_symlink_even_when_target_bytes_match(tmp_path):
    root = tmp_path / "release"
    root.mkdir()
    data = b"published\n"
    _write_manifest(root, {"results/value.txt": data})
    external = tmp_path / "external.txt"
    external.write_bytes(data)
    published = root / "results/value.txt"
    published.unlink()
    published.symlink_to(external)

    with pytest.raises(public_evidence.EvidenceError, match="symlink"):
        public_evidence.verify_manifest(root)


def test_verification_rejects_a_symlinked_manifest(tmp_path):
    root = tmp_path / "release"
    root.mkdir()
    manifest = _write_manifest(root, {"results/value.txt": b"published\n"})
    external = tmp_path / "manifest.json"
    manifest.replace(external)
    manifest.symlink_to(external)

    with pytest.raises(public_evidence.EvidenceError, match="symlink"):
        public_evidence.verify_manifest(root)


def test_required_file_must_be_selected_by_frozen_manifest(tmp_path):
    root = tmp_path / "release"
    root.mkdir()
    _write_manifest(root, {"results/selected.json": b"{}\n"})
    (root / "results/unselected.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(public_evidence.EvidenceError, match="not in evidence-manifest"):
        public_evidence.verify_manifest(root, required_paths=("results/unselected.json",))


def test_selection_identity_uses_the_publisher_contract_fields():
    selection = {
        "cases": [{"suite": "ifeval", "task_id": "1", "prompt": "Do it", "original": {}}],
        "population_sizes": {"ifeval": 1}, "scope": "fixture",
        "selected_count_per_suite": 1, "selection_method": "seeded",
        "selection_seed": 7, "sources": {"private_path": "/external/redacted"},
    }
    expected_value = {
        key: selection[key] for key in public_evidence.SELECTION_IDENTITY_FIELDS
    }
    expected = hashlib.sha256(json.dumps(
        expected_value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()
    assert public_evidence.selection_identity_sha256(selection) == expected

