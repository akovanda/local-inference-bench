#!/usr/bin/env python3
"""Verify, audit, and package the public research evidence (standard library only)."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import re
import tarfile

ARCHIVE_NAME = 'local-inference-bench-evidence-v0.1.0.tar.gz'
SKIP = {'.git', '.venv', '__pycache__', '.pytest_cache', 'build', '.cache'}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def verify(root: Path) -> dict:
    root = root.resolve()
    data = json.loads((root / 'evidence-manifest.json').read_text())
    if data.get('schema_version') != 1 or not isinstance(data.get('files'), list):
        raise ValueError('unsupported evidence manifest')
    seen = set()
    for item in data['files']:
        rel = Path(item['path'])
        path = root / rel
        if rel.is_absolute() or '..' in rel.parts or path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError(f'unsafe evidence path: {rel}')
        if rel.as_posix() in seen:
            raise ValueError(f'duplicate evidence path: {rel}')
        seen.add(rel.as_posix())
        if not path.is_file() or path.stat().st_size != item['size'] or digest(path) != item['sha256']:
            raise ValueError(f'missing or changed public evidence: {rel}')
    original = json.loads((root / 'results/public-benchmark-pilot-20261023-v1/manifest.json').read_text())
    identity = {key: original[key] for key in data['selection_identity_keys']}
    selected_sha = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()
    if selected_sha != data['selection_identity_sha256']:
        raise ValueError('frozen benchmark selection changed')
    return data


def audit(root: Path) -> dict:
    findings = []
    scanned = 0
    # Test fixture strings are still scanned; these patterns target complete token
    # shapes rather than auth-related keywords or secret-file *locations*.
    patterns = {
        'private key': re.compile(r'-----BEGIN (?:[A-Z]+ )?PRIVATE KEY-----'),
        'GitHub token': re.compile(r'\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})\b'),
        'Hugging Face token': re.compile(r'\bhf_[A-Za-z0-9]{30,}\b'),
        'AWS key': re.compile(r'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b'),
        'provider token': re.compile(r'\bsk-(?:proj-|ant-)?[A-Za-z0-9_-]{30,}\b'),
        'private user path': re.compile('/home/' + 'akovanda'),
        'tailnet DNS': re.compile(r'\b(?:[A-Za-z0-9-]+\.)+ts\.net\b'),
        'credential URL': re.compile(r'https?://[^\s/:<>]+:[^\s/@<>]+@'),
    }
    forbidden = {'.gguf', '.part', '.safetensors', '.onnx', '.so', '.whl', '.pkl', '.pickle', '.pyc'}
    for path in sorted(root.rglob('*')):
        rel = path.relative_to(root)
        if SKIP.intersection(rel.parts):
            continue
        if path.is_symlink():
            findings.append({'path': rel.as_posix(), 'reason': 'symlink in publication tree'})
            continue
        if not path.is_file():
            continue
        scanned += 1
        if path.suffix in forbidden or path.stat().st_size > 50 * 1024 * 1024:
            findings.append({'path': rel.as_posix(), 'reason': 'excluded binary or oversized file'})
            continue
        try:
            value = path.read_text(encoding='utf-8')
        except UnicodeError:
            if rel.as_posix() != 'vendor/strata/data/expert-profile.bin':
                findings.append({'path': rel.as_posix(), 'reason': 'unreviewed binary'})
            continue
        for name, pattern in patterns.items():
            if pattern.search(value):
                # Explicit URL-redaction fixtures contain fake credentials.
                if name == 'credential URL' and rel.parts[0] == 'tests':
                    continue
                findings.append({'path': rel.as_posix(), 'reason': name})
    if findings:
        # Do not print matched credential text.
        raise ValueError(json.dumps({'audit_findings': findings}, indent=2))
    return {'scanned_files': scanned, 'findings': 0}


def archive(root: Path, output: Path) -> dict:
    data = verify(root)
    audit(root)
    output.mkdir(parents=True, exist_ok=True)
    dest = output / ARCHIVE_NAME
    notices = ['LICENSE', 'THIRD_PARTY_NOTICES.md'] + [
        path.relative_to(root).as_posix() for path in sorted((root / 'docs/licenses').glob('*')) if path.is_file()
    ]
    files = ['evidence-manifest.json'] + sorted(set(item['path'] for item in data['files']) | set(notices))
    # Stable metadata makes this archive byte reproducible from its manifest.
    with dest.open('wb') as raw:
        with gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode='w') as tar:
                for name in files:
                    path = root / name
                    info = tar.gettarinfo(str(path), arcname=name)
                    info.uid = info.gid = 0
                    info.uname = info.gname = ''
                    info.mtime = 0
                    info.mode = 0o644
                    with path.open('rb') as stream:
                        tar.addfile(info, stream)
    checksum = digest(dest)
    (output / 'SHA256SUMS').write_text(f'{checksum}  {ARCHIVE_NAME}\n')
    return {'archive': str(dest), 'sha256': checksum, 'size': dest.stat().st_size, 'evidence_files': len(data['files'])}


def read_csv(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def compare_reports(root: Path, rebuilt: Path) -> dict:
    pairs = [
        ('results/model-evidence-20261023-v2/evidence.csv', 'cross-study/evidence.csv'),
        ('results/model-evidence-20261023-v2/group-summary.csv', 'cross-study/group-summary.csv'),
        ('results/public-model-expansion-20261024-v1/report/detailed-cases.csv', 'expansion/detailed-cases.csv'),
        ('results/public-model-expansion-20261024-v1/report/model-suite-summary.csv', 'expansion/model-suite-summary.csv'),
    ]
    checked = {}
    # Paths, provenance hashes, and display notes change in public derivatives;
    # compare every other field, including identities, treatment and metrics.
    ignored = {'source_path', 'grade_source_path', 'notes', 'config_sha256', 'catalog_sha256',
               'runner_sha256', 'profile_sha256', 'profile_tag'}
    for before, after in pairs:
        expected, actual = read_csv(root / before), read_csv(rebuilt / after)
        if len(expected) != len(actual):
            raise ValueError(f'row count differs: {after}')
        for index, (a, b) in enumerate(zip(expected, actual)):
            for key in a:
                if key in ignored or key.endswith('_path') or key in {'output_dir', 'config_path', 'profile_parameters_json'}:
                    continue
                if a[key] != b.get(key):
                    raise ValueError(f'rebuilt metric differs in {after}, row {index}, field {key}')
        checked[after] = len(actual)
    original_status = json.loads((root / 'results/public-model-expansion-20261024-v1/report/report-status.json').read_text())
    new_status = json.loads((rebuilt / 'expansion/report-status.json').read_text())
    for key in ('phase', 'queue_cases_expected', 'queue_cases_terminal', 'queue_cases_pending',
                'queue_cases_error', 'queue_profiles_terminal'):
        if original_status[key] != new_status[key]:
            raise ValueError(f'expansion status changed: {key}')
    return checked


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('verify', 'audit', 'archive', 'check-reports'):
        p = sub.add_parser(name)
        p.add_argument('--root', type=Path, default=Path('.'))
        if name == 'archive':
            p.add_argument('--output', type=Path, required=True)
        if name == 'check-reports':
            p.add_argument('--rebuilt', type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == 'verify':
            result = {'verified_files': len(verify(args.root)['files'])}
        elif args.command == 'audit':
            result = audit(args.root)
        elif args.command == 'archive':
            result = archive(args.root, args.output)
        else:
            result = compare_reports(args.root, args.rebuilt)
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f'{exc}\n')
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
