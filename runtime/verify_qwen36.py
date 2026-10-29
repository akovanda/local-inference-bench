#!/usr/bin/env python3
"""Verify the pinned public model after the active HF download publishes it."""
import datetime
import hashlib
import json
import pathlib
import time

MODEL = pathlib.Path('/external/storage/moe-bench/models/qwen36-UD-Q4_K_XL/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf')
RESULT = pathlib.Path(__file__).resolve().parents[1] / 'results/qwen36-artifact.json'
EXPECTED_SIZE = 22853663008
EXPECTED_SHA = '55983c5a75a1ab969824077b3bb3de4146e82a9234072b48ad4e8f92ad3fe9f1'


def main():
    deadline = time.monotonic() + 3600
    print('Waiting for the pinned Qwen model download to publish its final file.', flush=True)
    while not MODEL.is_file():
        if time.monotonic() >= deadline:
            raise TimeoutError('The final model file was not published within one hour.')
        time.sleep(2)
    started = time.monotonic()
    result = {
        'path': str(MODEL),
        'repo': 'unsloth/Qwen3.6-35B-A3B-MTP-GGUF',
        'revision': '5bc3e238d916f48a861bac2f8a1990a0e9b7e98d',
        'expected_size': EXPECTED_SIZE,
        'expected_sha256': EXPECTED_SHA,
        'verified': False,
    }
    try:
        before = MODEL.stat()
        result['size'] = before.st_size
        if before.st_size != EXPECTED_SIZE:
            raise ValueError('Model file size does not match the pinned size.')
        digest = hashlib.sha256()
        read_bytes = 0
        next_report = time.monotonic() + 10
        with MODEL.open('rb') as handle:
            while chunk := handle.read(8 * 1024 * 1024):
                digest.update(chunk)
                read_bytes += len(chunk)
                if time.monotonic() >= next_report:
                    print(f'Hash verification: {100 * read_bytes / EXPECTED_SIZE:.1f}%', flush=True)
                    next_report = time.monotonic() + 10
        after = MODEL.stat()
        if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
            raise ValueError('Model file changed during verification.')
        result['sha256'] = digest.hexdigest()
        if read_bytes != EXPECTED_SIZE or result['sha256'] != EXPECTED_SHA:
            raise ValueError('Model checksum does not match the pinned SHA-256.')
        result['verified'] = True
    except Exception as exc:
        result['error'] = str(exc)
    result['checked_at'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    result['verification_seconds'] = time.monotonic() - started
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    temporary = RESULT.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(result, indent=2) + '\n')
    temporary.replace(RESULT)
    print(json.dumps(result), flush=True)
    return 0 if result['verified'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
