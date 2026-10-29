#!/usr/bin/env python3
"""Run the two frozen public pilots serially, with owned GPU windows and grading."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

if __package__:
    from . import gpu_window, llama_trial, specialist_study_trial as ownership
    from .signal_exec import unblocked_exec_command
else:
    import gpu_window, llama_trial, specialist_study_trial as ownership
    from signal_exec import unblocked_exec_command

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "vendor/public-benchmarks"
PLAN = ROOT / "runtime/public-benchmark-pilot-ready-v1-plan.json"
RUNNER = ROOT / "runtime/public_benchmark_pilot.py"
MODELS = (
    {"key": "oss20b", "config": ROOT / "runtime/gpt20b-llama-specialists-72k-wide64k.json"},
    {"key": "lfm25", "config": ROOT / "runtime/public-benchmark-pilot-lfm72k-auto-v1.json"},
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp-{os.getpid()}")
    with tmp.open("x") as f:
        json.dump(value, f, indent=2, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def verify_manifest(study: Path) -> dict:
    expected = (study / "manifest.sha256").read_text().split()[0]
    if digest(study / "manifest.json") != expected:
        raise RuntimeError("frozen benchmark manifest SHA256 mismatch")
    data = json.loads((study / "manifest.json").read_text())
    if data.get("selected_count_per_suite") != 32 or len(data.get("cases", [])) != 64:
        raise RuntimeError("this controller requires the frozen 32+32 pilot")
    return data


def runner_command(python: str, study: Path, model: dict, output: Path) -> list[str]:
    return [python, str(RUNNER), "run", "--manifest", str(study / "manifest.json"),
            "--endpoint", "http://127.0.0.1:18081/v1", "--model", model["alias"],
            "--model-key", model["key"], "--output-dir", str(output), "--max-tokens", "65536",
            "--request-timeout", "21600", "--reasoning-effort", "medium", "--temperature", "0",
            "--seed", "20261023"]


def grade_command(image: str, study: Path, model: dict, output: Path, manifest: dict) -> list[str]:
    # Only public sources/inputs are readable; only this model's results are writable.
    return ["docker", "run", "--rm", "--name", f"moe-public-grade-{model['key']}-{os.getpid()}",
            "--network", "none", "--read-only", "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
            "--pids-limit", "256", "--cpus", "4", "--memory", "4g", "--shm-size", "256m",
            "--user", f"{os.getuid()}:{os.getgid()}",
            "--tmpfs", "/tmp:rw,nosuid,size=2g", "--env", "HOME=/tmp", "--env", "XDG_CACHE_HOME=/tmp/cache",
            "--env", "PYTHONDONTWRITEBYTECODE=1", "--env", "OMP_NUM_THREADS=1", "--env", "OPENBLAS_NUM_THREADS=1",
            "--mount", f"type=bind,src={RUNNER},dst=/app/public_benchmark_pilot.py,readonly",
            "--mount", f"type=bind,src={VENDOR},dst={VENDOR},readonly",
            "--mount", f"type=bind,src={study},dst={study},readonly",
            "--mount", f"type=bind,src={output},dst={output}", image,
            "python", "/app/public_benchmark_pilot.py", "grade", "--manifest", str(study / "manifest.json"),
            "--output-dir", str(output), "--evalplus-root", manifest["sources"]["evalplus"]["root"],
            "--ifeval-root", str(Path(manifest["sources"]["ifeval"]["root"]) / "instruction_following_eval")]


class Controller:
    def __init__(self, args: argparse.Namespace, manifest: dict) -> None:
        self.args, self.manifest = args, manifest
        self.study = args.study_dir.resolve()
        self.received_signal: int | None = None
        self.state = {"schema_version": 1, "study": "OSS20B versus LFM2.5 public pilot",
                      "started_at": now(), "controller_pid": os.getpid(), "status": "starting",
                      "manifest_sha256": digest(self.study / "manifest.json"),
                      "planned_requests": 128, "planned_per_model": 64,
                      "model_order": [m["key"] for m in MODELS], "models": {},
                      "controls": {"context_tokens": 73728, "max_output_tokens": 65536,
                                   "request_timeout_seconds": 21600, "temperature": 0,
                                   "seed": 20261023, "reasoning_effort": "medium",
                                   "note": "LFM has native always-on reasoning; effort values need not mean equal compute. One sample per task."}}

    def save(self, **fields) -> None:
        self.state.update(fields)
        self.state["updated_at"] = now()
        atomic_json(self.study / "controller-status.json", self.state)

    def event(self, name: str, **fields) -> None:
        row = {"at": now(), "event": name, **fields}
        with (self.study / "controller-events.jsonl").open("a") as f:
            f.write(json.dumps(row) + "\n")
            f.flush()
            os.fsync(f.fileno())
        print(json.dumps(row), flush=True)

    def check_signal(self) -> None:
        if self.received_signal is not None:
            raise RuntimeError(f"controller interrupted by signal {self.received_signal}")

    def handle_signal(self, signum, _frame) -> None:
        self.received_signal = signum

    def run_model(self, entry: dict) -> None:
        self.check_signal()
        for filename, expected in self.state.get("runtime_file_sha256", {}).items():
            if digest(Path(filename)) != expected:
                raise RuntimeError(f"pinned runtime changed before model boundary: {filename}")
        cfg = llama_trial.load_trial_config(entry["config"])
        model = {**entry, "config": str(entry["config"]), "alias": cfg.model_name,
                 "config_sha256": digest(entry["config"]), "context": cfg.context}
        if cfg.context != 73728:
            raise RuntimeError("public pilot requires matched 72K context configurations")
        key = model["key"]
        output = self.study / key
        output.mkdir(exist_ok=True)
        tag = f"public-pilot-20261023-v1-{key}"
        control = ROOT / "results" / f"{tag}-control.json"
        events = ROOT / "results" / f"{tag}-runtime-events.jsonl"
        if events.exists() or control.exists():
            raise RuntimeError(f"session tag already exists; refusing to overwrite: {tag}")
        row = {**model, "status": "loading", "started_at": now(), "output_dir": str(output),
               "session_tag": tag, "control_file": str(control), "runtime_events": str(events)}
        self.state["models"][key] = row
        self.save(status="running", current_model=key)
        window_cmd = [self.args.python, str(ROOT / "runtime/gpu_window.py"), "--wait-seconds", "900",
                      "--max-seconds", "86400", "--long-session", "--", str(ROOT / "runtime/model_session.py"),
                      "--runtime-backend", "llama", "--config", model["config"], "--plan", str(PLAN),
                      "--tag", tag, "--startup-timeout", "480", "--hold-after-plan", "--control-file", str(control)]
        outer = child = None
        stage_error = None
        with (output / "gpu-window.log").open("x") as window_log, (output / "runner.log").open("x") as runner_log:
            try:
                outer = subprocess.Popen(unblocked_exec_command(window_cmd), cwd=ROOT, stdin=subprocess.DEVNULL,
                                         stdout=window_log, stderr=subprocess.STDOUT, start_new_session=True)
                row["gpu_window_pid"] = outer.pid
                self.save()
                latch = ownership.SignalLatch()
                holder_args = SimpleNamespace(tag=tag, config=entry["config"], plan=PLAN, runtime_backend="llama")
                identity = ownership.wait_for_holding(outer, events, args=holder_args, control=control,
                                                       deadline=time.monotonic() + 1800, signals=latch)
                self.check_signal()
                row.update(status="generating", session=asdict(identity), inference_started_at=now())
                self.save()
                self.event("model_inference_started", model=key, session=asdict(identity))
                cmd = runner_command(self.args.python, self.study, model, output)
                child = subprocess.Popen(unblocked_exec_command(cmd), cwd=ROOT, stdin=subprocess.DEVNULL,
                                         stdout=runner_log, stderr=subprocess.STDOUT, start_new_session=True)
                row["runner_pid"] = child.pid
                self.save()
                while child.poll() is None:
                    self.check_signal()
                    if outer.poll() is not None:
                        raise RuntimeError(f"owned GPU window exited during inference ({outer.returncode})")
                    self.save()
                    time.sleep(5)
                row["inference_returncode"] = child.returncode
                row["inference_finished_at"] = now()
                if child.returncode != 0:
                    raise RuntimeError(f"public runner exited {child.returncode}; saved attempts retained")
            except BaseException as exc:
                stage_error = exc
            finally:
                if child is not None and child.poll() is None:
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        # Keep the owned server alive rather than leave a request running after restore.
                        row["runner_cleanup_error"] = "runner did not exit after SIGTERM"
                if outer is not None:
                    row["stop_requested"] = ownership.request_session_stop(control)
                    code, _ = ownership.wait_for_outer_exit(outer, timeout=180, control=control)
                    if code is None:
                        ownership.request_outer_termination(outer)
                        code, _ = ownership.wait_for_outer_exit(outer, timeout=180, control=control)
                    row["gpu_window_returncode"] = code
                    row["restoration"] = ownership.restoration_evidence("llama")
                    self.save()
                    if code != 0 or row["restoration"]["original_paused"] or not row["restoration"]["owned_loopback_port_clear"]:
                        stage_error = RuntimeError(f"owned model cleanup/restoration not clean: {code}")
        if stage_error is not None:
            row.update(status="failed", error=str(stage_error), finished_at=now())
            self.save()
            raise stage_error
        row["status"] = "grading"
        self.save()
        self.event("model_grading_started", model=key)
        cmd = grade_command(self.args.grade_image, self.study, model, output, self.manifest)
        atomic_json(output / "grading-command.json", {"argv": cmd, "isolation": "network disabled; read-only root/public inputs; only this model output writable"})
        with (output / "grading.log").open("x") as f:
            completed = subprocess.run(cmd, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=f, stderr=subprocess.STDOUT, timeout=7200)
        row["grading_returncode"] = completed.returncode
        row["finished_at"] = now()
        row["status"] = "complete" if completed.returncode == 0 else "grading_failed"
        summary = output / "grading/summary.json"
        if summary.exists():
            row["grading_summary"] = json.loads(summary.read_text())
        self.save()
        self.event("model_finished", model=key, status=row["status"])
        if completed.returncode:
            raise RuntimeError(f"official grading failed for {key}; answers retained for offline repair")

    def run(self) -> int:
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, self.handle_signal)
        self.state["original_container_state"] = gpu_window.original_state()
        self.state["original_resident_models"] = gpu_window.loaded_models(time.monotonic() + 10)
        self.state["grade_image"] = self.args.grade_image
        self.state["runner_sha256"] = digest(RUNNER)
        pins = [ROOT / "runtime" / x for x in ["gpu_window.py", "model_session.py", "llama_trial.py", "strata_trial.py", "signal_exec.py", "specialist_study_trial.py"]]
        pins.extend(x["config"] for x in MODELS)
        pins.extend([PLAN, RUNNER, Path(__file__).resolve()])
        self.state["runtime_file_sha256"] = {str(p): digest(p) for p in pins}
        self.save()
        self.event("pilot_started", planned_requests=128, manifest_sha256=self.state["manifest_sha256"])
        try:
            for model in MODELS:
                self.run_model(model)
            self.save(status="complete", finished_at=now(), current_model=None)
            self.event("pilot_complete")
            return 0
        except BaseException as exc:
            self.save(status="interrupted" if self.received_signal else "failed", error=str(exc), finished_at=now())
            self.event("pilot_failed", error=str(exc))
            return 1
        finally:
            # Reinstate the original resident model only after our own GPU process is gone.
            evidence = ownership.restoration_evidence("llama")
            self.state["final_restoration"] = evidence
            if evidence["original_running"] and not evidence["original_paused"] and evidence["owned_loopback_port_clear"]:
                original = self.state.get("original_resident_models", [])
                if len(original) == 1 and original[0].get("name") == gpu_window.ORIGINAL_MODEL:
                    try:
                        row = original[0]
                        payload = {"model": gpu_window.ORIGINAL_MODEL, "keep_alive": -1, "stream": False,
                                   "options": {"num_ctx": row.get("context_length", 32768)}}
                        gpu_window._api_request("POST", "/api/generate", 180, payload)
                        self.state["original_residency_restored"] = gpu_window.loaded_models(time.monotonic() + 10)
                    except Exception as exc:
                        self.state["residency_restore_error"] = str(exc)
            self.save()


def main() -> int:
    global RUNNER
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, default=ROOT / "results/public-benchmark-pilot-20261023-v1")
    parser.add_argument("--python", default=str(ROOT / ".venv-public-bench/bin/python"))
    parser.add_argument("--grade-image", default="moe-public-benchmark-grader:20261023-v1")
    parser.add_argument("--runner", type=Path, help="immutable snapshot of the benchmark runner")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.runner is not None:
        RUNNER = args.runner.resolve()
    args.study_dir = args.study_dir.resolve()
    manifest = verify_manifest(args.study_dir)
    if args.dry_run:
        print(json.dumps({"planned_requests": 128, "manifest_sha256": digest(args.study_dir / "manifest.json"),
                          "models": [{"key": x["key"], "config": str(x["config"]),
                                      "context": llama_trial.load_trial_config(x["config"]).context} for x in MODELS]}, indent=2))
        return 0
    if not RUNNER.is_file():
        raise RuntimeError("benchmark runner has not been prepared")
    subprocess.run(["docker", "image", "inspect", "--format", "{{.Id}}", args.grade_image], check=True, stdout=subprocess.DEVNULL)
    with (args.study_dir / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (args.study_dir / "controller-status.json").exists():
            raise RuntimeError("study controller already has saved state; use a new version or explicit recovery")
        return Controller(args, manifest).run()


if __name__ == "__main__":
    raise SystemExit(main())
