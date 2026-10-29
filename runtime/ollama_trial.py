#!/usr/bin/env python3
"""Run one isolated Ollama configuration and restore the original service."""
import argparse
import datetime
import json
import pathlib
import signal
import subprocess
import sys
import time
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
ORIGINAL = 'local-codex-ollama-cpu'
OLD = 'sha256:f6722249524a681e2862797973f8a808abf4ada2ae166023b15b56f7398bad9e'
NEW = 'sha256:dbbbdcfe875a5e686e5d7651d0901ebe5a73a87f26cd948a4199717f9802bddd'
PROFILES = {'old-f16': (OLD, 'f16'), 'old-q8': (OLD, 'q8_0'), 'new-f16': (NEW, 'f16'), 'new-q8': (NEW, 'q8_0')}
NAMES = ['moe-test-ollama-' + k for k in PROFILES]

def docker(*args, check=True):
    return subprocess.run(['docker', *args], check=check, text=True, capture_output=True, timeout=180)

def event(kind, **fields):
    row = {'at': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'event': kind, **fields}
    with (ROOT / 'results' / 'runtime-events.jsonl').open('a') as f:
        f.write(json.dumps(row) + '\n')
        f.flush()
    print(json.dumps(row), flush=True)

def interrupted(sig, frame):
    raise KeyboardInterrupt(f'signal {sig}')

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profile', choices=PROFILES, required=True)
    p.add_argument('--contexts', default='32768')
    p.add_argument('--quality', action='store_true')
    p.add_argument('--context-target', type=int)
    p.add_argument('--repeat', type=int, default=2)
    p.add_argument('--skip-quick', action='store_true')
    a = p.parse_args()
    contexts = [int(c) for c in a.contexts.split(',')]
    if not contexts or any(c not in (16384, 32768, 65536, 131072) for c in contexts):
        p.error('contexts must be 16384,32768,65536,131072')
    image, cache = PROFILES[a.profile]
    name = 'moe-test-ollama-' + a.profile
    docker('image', 'inspect', image, '--format', '{{.Id}}')
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    event('trial_start', profile=a.profile, image=image, kv_cache=cache, contexts=contexts)
    try:
        docker('stop', '--timeout', '30', ORIGINAL)
        # The parent serializes trials. A stopped container may still be completing
        # Docker metadata writes, so do not stop unrelated inactive trial names.
        memory_deadline = time.monotonic() + 90
        while True:
            memory = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True, timeout=10)
            if int(memory.strip()) < 200:
                break
            if time.monotonic() >= memory_deadline:
                raise RuntimeError('GPU is still occupied; refusing overlapping inference')
            time.sleep(1)
        exists = docker('container', 'inspect', name, '--format', '{{.State.Running}}', check=False)
        if exists.returncode:
            docker('run', '-d', '--name', name, '--gpus', 'all', '-p', '127.0.0.1:11440:11434',
                   '-v', '/external/models/run/ollama:/root/.ollama:ro',
                   '-e', 'OLLAMA_HOST=0.0.0.0:11434', '-e', 'OLLAMA_KEEP_ALIVE=10m',
                   '-e', 'OLLAMA_NUM_PARALLEL=1', '-e', 'OLLAMA_CONTEXT_LENGTH=32768',
                   '-e', 'OLLAMA_KV_CACHE_TYPE=' + cache, '-e', 'OLLAMA_FLASH_ATTENTION=1',
                   '-e', 'OLLAMA_NOPRUNE=1', image)
        else:
            docker('start', name)
        deadline = time.monotonic() + 60
        while True:
            try:
                with urllib.request.urlopen('http://127.0.0.1:11440/api/version', timeout=3) as r:
                    version = json.load(r)
                break
            except Exception:
                if time.monotonic() >= deadline:
                    raise RuntimeError('trial server did not become ready in 60s')
                time.sleep(1)
        event('server_ready', profile=a.profile, version=version)
        for ctx in contexts:
            stem = f'{a.profile}-{ctx//1024}k'
            health = ROOT / 'results' / (stem + '-health.json')
            subprocess.run([sys.executable, str(ROOT/'runtime/probe.py'), '--context', str(ctx), '--timeout', '180', '--output', str(health)], check=True)
            if not json.loads(health.read_text()).get('passed'):
                event('context_health_failed', profile=a.profile, context=ctx)
                continue
            def bench(suite, extra):
                output = ROOT/'results'/(stem+'-'+suite+'.jsonl')
                cmd = [sys.executable, str(ROOT/'bench.py'), '--endpoint', 'http://127.0.0.1:11440', '--backend', 'ollama', '--model', 'gpt-oss-20b:64k', '--context', str(ctx), '--thinking', 'low', '--suite', suite, '--timeout', '300', '--output', str(output), *extra]
                event('suite_start', profile=a.profile, context=ctx, suite=suite)
                code = subprocess.run(cmd).returncode
                event('suite_end', profile=a.profile, context=ctx, suite=suite, returncode=code)
            if not a.skip_quick:
                bench('quick', ['--repeat', str(a.repeat), '--num-predict', '512'])
            if a.quality:
                bench('quality', ['--cases', 'quality_merge_intervals,quality_stable_toposort,quality_constraints'])
            if a.context_target:
                bench('context', ['--context-target', str(a.context_target), '--num-predict', '768'])
        return 0
    finally:
        try:
            logs = docker('logs', name, check=False)
            (ROOT/'results'/(a.profile+'-server.log')).write_text(logs.stdout + logs.stderr)
        except Exception as exc:
            event('log_capture_failed', profile=a.profile, error=str(exc))
        try:
            docker('stop', '--timeout', '20', name, check=False)
        except Exception as exc:
            event('trial_stop_failed', profile=a.profile, error=str(exc))
        finally:
            restored = docker('start', ORIGINAL, check=False)
            event('original_service_restored', profile=a.profile, success=restored.returncode == 0)
            if restored.returncode:
                print(restored.stderr, file=sys.stderr)

if __name__ == '__main__':
    raise SystemExit(main())
