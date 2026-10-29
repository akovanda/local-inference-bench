#!/usr/bin/env python3
"""Bounded post-study LFM2.6 recurrent-cache and generic OpenCL trial.

The controller waits for the frozen worker study to release ownership, runs a
CPU A/B/A comparison over its eight routing requests, makes one generic
OpenCL llama-bench attempt, and leaves a verified LFM2.6 CPU service running.
It never serializes manifest ``expected`` values into model requests.
"""
from __future__ import annotations

import argparse
import base64
import copy
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import shlex
import signal
import statistics
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "results/tachyon-worker-study-20261023-v1"
DEFAULT_TRIAL = ROOT / "results/tachyon-worker-latency-20261023-v1"
MANIFEST = BASELINE / "manifest.json"
MODEL_PLAN = BASELINE / "models-plan.json"
BASELINE_SOURCE = ROOT / "runtime/tachyon_worker_study.py"
CPU_BUILD_RECEIPT = ROOT / "results/tachyon-native-runtime-20261023-v1.json"
OPENCL_BUILD_RECEIPT = ROOT / "results/tachyon-study-20261023-v1/opencl-generic-runtime-build.json"

REMOTE_ROOT = "/external/device-workspace"
REMOTE_RESULTS = REMOTE_ROOT + "/results/tachyon-worker-latency-20261023-v1"
REMOTE_CPU_RUNTIME = REMOTE_ROOT + "/runtime/llama-cpp-cpu"
REMOTE_OPENCL_RUNTIME = REMOTE_ROOT + "/runtime/llama-cpp-opencl-generic"
REMOTE_ICD = REMOTE_ROOT + "/runtime/opencl/vendors"
REMOTE_MODEL = REMOTE_ROOT + "/models/lfm25-26b/LFM2.5-2.6B-QAD-Q4_0.gguf"
ENDPOINT = "http://127.0.0.1:18185/v1"
MODEL_KEY = "lfm25-26b"
ALIAS = "tachyon-lfm25-26b"
STUDY_KIND = "tachyon_worker_lfm26_latency_trial"

MANIFEST_SHA256 = "c5ef1735ae4163a077b9033c11910e42939ffb6e4a34293abf50cb58c13ddc0d"
MODEL_PLAN_SHA256 = "555b08874f5849b8530492ad0de343e60e4936c11b765d7c7a1484714f7578f2"
BASELINE_SOURCE_SHA256 = "168a29529ce4bce20140e4be3f160624e460dfd9db5c42a0f49236c07a0acae0"
CPU_BUILD_RECEIPT_SHA256 = "4dc3392552106c6838ce2ad9c383938b5408a80ddbc90d604bcc77ce1997a05e"
OPENCL_BUILD_RECEIPT_SHA256 = "5361184de9611a881c02aaf34f70379dd99865ecb1951780546d4a3985974ec2"
CPU_SERVER_SHA256 = "f153cc81dfbb20c3fe8086a668c1bbf388961329f34577628e66318b9dadb5ee"
OPENCL_BENCH_SHA256 = "f9c41e67809015815083a14441eca28e88ca49dc74686401c7a9fa3e5e83ce5c"
MODEL_SHA256 = "a247afd6414918eac8e520a9e6137dc271235461ecbe1180462221d5b8d40b03"
MODEL_SIZE = 1_593_894_944
SOURCE_COMMIT = "620d2225cc436ab0b001e3200e590e3b286ef9ab"
EXPECTED_BASELINE_CONTROLLER_PID = 3_722_122

EPOCHS = (
    {"key": "a1-ub128", "arm": "A", "ubatch": 128},
    {"key": "b-ub32", "arm": "B", "ubatch": 32},
    {"key": "a2-ub128", "arm": "A", "ubatch": 128},
)
SSH_OPTIONS = (
    "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes",
    "-o", "ClearAllForwardings=yes", "-o", "ForwardAgent=no",
    "-o", "RequestTTY=no",
)


class TrialError(RuntimeError):
    pass


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")


def json_sha(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise TrialError(f"cannot hash required file: {path}") from exc
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TrialError(f"invalid or unreadable JSON: {path}") from exc


def atomic_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("xb") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_bytes(path, json.dumps(value, indent=2, sort_keys=True,
                                  ensure_ascii=False).encode("utf-8") + b"\n")


def append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def process_identity(pid: int) -> dict[str, Any]:
    try:
        base = Path("/proc") / str(pid)
        stat = (base / "stat").read_text(encoding="utf-8")
        tail = stat[stat.rfind(")") + 2:].split()
        argv = [item.decode("utf-8", "surrogateescape")
                for item in (base / "cmdline").read_bytes().split(b"\0") if item]
        return {"pid": pid, "alive": tail[0] != "Z", "state": tail[0],
                "start_ticks": int(tail[19]), "argv": argv}
    except (FileNotFoundError, ProcessLookupError):
        return {"pid": pid, "alive": False, "state": "exited", "argv": []}
    except (OSError, ValueError, IndexError) as exc:
        raise TrialError(f"cannot inspect local process {pid}: {exc}") from exc


def lock_is_free(path: Path) -> bool:
    if not path.is_file():
        return False
    with path.open("r", encoding="utf-8") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
    return True


def verify_owned_process(receipt: Mapping[str, Any], identity: Mapping[str, Any],
                         label: str) -> bool:
    """Return false for an exited process; reject every live identity mismatch."""
    if not identity.get("alive"):
        return False
    for field in ("pid", "start_ticks", "argv"):
        if receipt.get(field) != identity.get(field):
            raise TrialError(f"{label} {field} differs from ownership receipt")
    return True


def percentile(values: Sequence[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(probability * len(ordered)) - 1)]


def kib_value(value: Any) -> int | None:
    match = re.fullmatch(r"\s*([0-9]+)\s+kB\s*", value) if isinstance(value, str) else None
    return int(match.group(1)) if match else None


def build_request(case: Mapping[str, Any], model: Mapping[str, Any]) -> dict[str, Any]:
    """Build only from public prompt/schema fields; expected answers stay local."""
    safe_id = re.sub(r"[^A-Za-z0-9_-]", "_", str(case["id"]))
    return {
        "model": model["alias"],
        "messages": copy.deepcopy(case["messages"]),
        "max_tokens": 512,
        "seed": 20_261_023,
        **copy.deepcopy(model["recipe"]),
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "worker_latency_" + safe_id,
                "strict": True,
                "schema": copy.deepcopy(case["public_schema"]),
            },
        },
        "stream": True,
        "stream_options": {"include_usage": True},
        "timings_per_token": True,
    }


def evaluate_route(case: Mapping[str, Any], final_text: str) -> dict[str, Any]:
    try:
        observed = json.loads(final_text)
        valid = isinstance(observed, dict)
        error = None if valid else "top-level JSON is not an object"
    except json.JSONDecodeError as exc:
        observed, valid, error = None, False, str(exc)
    expected = case["expected"]
    semantic = bool(valid and set(observed) == {"route"}
                    and observed.get("route") == expected.get("route"))
    return {"json_valid": valid, "observed_json": observed,
            "exact_top_level_keys": bool(valid and set(observed) == {"route"}),
            "semantic_pass": semantic, "syntax_error": error}


def parse_sse_lines(lines: Sequence[bytes]) -> dict[str, Any]:
    chunks: list[dict[str, Any]] = []
    final_parts: list[str] = []
    reasoning_parts: list[str] = []
    finish_reason: str | None = None
    usage: Mapping[str, Any] = {}
    timings: Mapping[str, Any] = {}
    for raw in lines:
        line = raw.decode("utf-8").strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            continue
        row = json.loads(payload)
        if not isinstance(row, dict):
            raise TrialError("SSE data row is not an object")
        chunks.append(row)
        if isinstance(row.get("usage"), dict):
            usage = row["usage"]
        if isinstance(row.get("timings"), dict):
            timings = row["timings"]
        for choice in row.get("choices", []):
            if not isinstance(choice, dict):
                continue
            if isinstance(choice.get("finish_reason"), str):
                finish_reason = choice["finish_reason"]
            delta = choice.get("delta")
            if not isinstance(delta, dict):
                continue
            content = delta.get("content")
            reasoning = delta.get("reasoning_content", delta.get("reasoning"))
            if isinstance(content, str):
                final_parts.append(content)
            if isinstance(reasoning, str):
                reasoning_parts.append(reasoning)
    return {"chunks": chunks, "final_text": "".join(final_parts),
            "reasoning_text": "".join(reasoning_parts),
            "finish_reason": finish_reason, "usage": dict(usage),
            "native_timings": dict(timings)}


def server_command(ubatch: int) -> list[str]:
    return [
        "taskset", "-c", "4-7", REMOTE_CPU_RUNTIME + "/bin/llama-server",
        "-m", REMOTE_MODEL, "--alias", ALIAS, "--host", "127.0.0.1",
        "--port", "18085", "-c", "16384", "-np", "1", "-t", "4",
        "-tb", "4", "-b", "512", "-ub", str(ubatch), "-ngl", "0",
        "-ctxcp", "32", "-cms", "8192", "--cache-prompt", "--jinja",
        "--reasoning-format", "deepseek", "--reasoning-effort", "none",
        "--threads-http", "2",
    ]


def benchmark_command() -> list[str]:
    return [
        "env", f"OCL_ICD_VENDORS={REMOTE_ICD}", "taskset", "-c", "4-7",
        REMOTE_OPENCL_RUNTIME + "/bin/llama-bench", "-m", REMOTE_MODEL,
        "-p", "128", "-n", "32", "-t", "4", "-r", "2", "-ngl", "4",
        "--device", "GPUOpenCL", "-v", "-o", "json",
    ]


def actual_exec_argv(command: Sequence[str]) -> list[str]:
    value = list(command)
    if value[:1] == ["env"]:
        value = value[2:]
    if value[:1] == ["taskset"]:
        value = value[3:]
    return value


def benchmark_rates(value: Any) -> dict[str, float | None]:
    rows = value if isinstance(value, list) else []
    prompt = [float(row["avg_ts"]) for row in rows if isinstance(row, dict)
              and row.get("n_prompt") == 128 and row.get("n_gen") == 0
              and isinstance(row.get("avg_ts"), (int, float))]
    generation = [float(row["avg_ts"]) for row in rows if isinstance(row, dict)
                  and row.get("n_prompt") == 0 and row.get("n_gen") == 32
                  and isinstance(row.get("avg_ts"), (int, float))]
    return {"prompt_tokens_per_second": prompt[0] if len(prompt) == 1 else None,
            "generation_tokens_per_second": generation[0] if len(generation) == 1 else None}


class Controller:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.trial = args.trial.resolve()
        self.status_path = self.trial / "trial-status.json"
        self.events_path = self.trial / "events.jsonl"
        self.state: dict[str, Any] = {}
        self.cases: list[dict[str, Any]] = []
        self.model: dict[str, Any] = {}
        self.current: dict[str, Any] | None = None
        self.lock_stream: io.TextIOWrapper | None = None
        self.interrupted: int | None = None
        self.handoff_acquired = False
        self.service_taken_over = False

    def event(self, name: str, **fields: Any) -> None:
        row = {"at": now(), "event": name, **fields}
        append_jsonl(self.events_path, row)
        print(json.dumps(row, sort_keys=True), flush=True)

    def save(self, **fields: Any) -> None:
        self.state.update(fields, controller_pid=os.getpid(), updated_at=now())
        atomic_json(self.status_path, self.state)

    def acquire_lock(self) -> None:
        self.trial.mkdir(parents=True, exist_ok=True)
        self.lock_stream = (self.trial / ".controller.lock").open("a+")
        try:
            fcntl.flock(self.lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise TrialError("another latency-trial controller holds the lock") from exc
        self.lock_stream.seek(0)
        self.lock_stream.truncate()
        self.lock_stream.write(f"{os.getpid()}\n")
        self.lock_stream.flush()
        identity = process_identity(os.getpid())
        atomic_json(self.trial / "controller-launch.json", {
            "schema_version": 1, "kind": STUDY_KIND, "at": now(),
            "pid": os.getpid(), "start_ticks": identity["start_ticks"],
            "argv": identity["argv"], "source_sha256": sha256(Path(__file__)),
        })

    def check_interrupt(self) -> None:
        if self.interrupted is not None:
            raise TrialError(f"controller interrupted by signal {self.interrupted}")

    def handle_signal(self, signum: int, _frame: Any) -> None:
        self.interrupted = signum

    def remote_json(self, script: str, payload: Mapping[str, Any],
                    timeout: float = 90) -> Any:
        command = shlex.join(["python3", "-c", script,
                              json.dumps(payload, separators=(",", ":"))])
        try:
            completed = subprocess.run(
                ["ssh", *SSH_OPTIONS, "-T", self.args.remote, command],
                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, text=True, timeout=timeout, check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise TrialError("SSH command timed out") from exc
        if completed.returncode:
            raise TrialError(f"remote command failed: {completed.stderr.strip()[-1200:]}")
        try:
            return json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise TrialError("remote command returned invalid JSON") from exc

    def remote_process(self, pid: int) -> dict[str, Any]:
        script = r'''import json,pathlib,sys
p=json.loads(sys.argv[1]); pid=int(p["pid"]); b=pathlib.Path("/proc")/str(pid)
try:
 s=(b/"stat").read_text(); t=s[s.rfind(")")+2:].split(); a=[x.decode("utf-8","surrogateescape") for x in (b/"cmdline").read_bytes().split(b"\0") if x]
 print(json.dumps({"pid":pid,"alive":t[0]!="Z","state":t[0],"start_ticks":int(t[19]),"argv":a}))
except FileNotFoundError: print(json.dumps({"pid":pid,"alive":False,"state":"exited","argv":[]}))'''
        value = self.remote_json(script, {"pid": pid})
        if not isinstance(value, dict):
            raise TrialError("remote process identity is malformed")
        return value

    def remote_bytes(self, path: str) -> bytes:
        script = r'''import base64,json,pathlib,sys
p=pathlib.Path(json.loads(sys.argv[1])["path"])
print(json.dumps({"data":base64.b64encode(p.read_bytes()).decode("ascii")}))'''
        value = self.remote_json(script, {"path": path})
        if not isinstance(value, dict) or not isinstance(value.get("data"), str):
            raise TrialError(f"remote file is unreadable: {path}")
        return base64.b64decode(value["data"], validate=True)

    def resource_snapshot(self, name: str,
                          receipt: Mapping[str, Any] | None = None) -> dict[str, Any]:
        path = self.trial / name
        if path.is_file():
            value = read_json(path)
            if not isinstance(value, dict):
                raise TrialError(f"resource snapshot is malformed: {path}")
            return value
        script = r'''import glob,json,pathlib,sys
p=json.loads(sys.argv[1]); mem={}; thermal=[]; policies={}
for line in pathlib.Path("/proc/meminfo").read_text().splitlines():
 if ":" in line and line.split(":",1)[0] in ("MemTotal","MemAvailable","SwapTotal","SwapFree"): mem[line.split(":",1)[0]]=line.split(":",1)[1].strip()
for f in sorted(glob.glob("/sys/class/thermal/thermal_zone*/temp")):
 d=pathlib.Path(f).parent
 try: thermal.append({"zone":d.name,"type":(d/"type").read_text().strip(),"millidegrees_c":int((d/"temp").read_text())})
 except Exception: pass
for cpu in (4,7):
 d=pathlib.Path("/sys/devices/system/cpu/cpufreq")/("policy"+str(cpu)); row={}
 for key in ("scaling_cur_freq","scaling_max_freq","cpuinfo_max_freq","scaling_governor"):
  try: row[key]=(d/key).read_text().strip()
  except OSError: row[key]=None
 policies[str(cpu)]=row
proc=None
if p.get("pid") is not None:
 b=pathlib.Path("/proc")/str(p["pid"])
 try:
  s=(b/"stat").read_text(); t=s[s.rfind(")")+2:].split(); a=[x.decode("utf-8","surrogateescape") for x in (b/"cmdline").read_bytes().split(b"\0") if x]; st={}
  for line in (b/"status").read_text().splitlines():
   if ":" in line and line.split(":",1)[0] in ("VmRSS","VmHWM","VmSwap"): st[line.split(":",1)[0]]=line.split(":",1)[1].strip()
  proc={"pid":p["pid"],"alive":t[0]!="Z","state":t[0],"start_ticks":int(t[19]),"argv":a,"status":st}
 except FileNotFoundError: proc={"pid":p["pid"],"alive":False,"state":"exited","argv":[]}
print(json.dumps({"meminfo":mem,"thermal":thermal,"thermal_max_millidegrees_c":max([x["millidegrees_c"] for x in thermal],default=None),"cpufreq_policy4_policy7":policies,"process":proc}))'''
        value = self.remote_json(script, {"pid": receipt.get("pid") if receipt else None})
        if not isinstance(value, dict):
            raise TrialError("remote resource snapshot is malformed")
        if receipt is not None:
            process = value.get("process")
            if not isinstance(process, dict):
                raise TrialError("server resource snapshot lacks process identity")
            verify_owned_process(receipt, process, "snapshot server")
        result = {"schema_version": 1, "captured_at": now(), **value}
        atomic_json(path, result)
        return result

    def wait_for_baseline(self) -> dict[str, Any]:
        deadline = time.monotonic() + self.args.wait_timeout
        latest = "baseline status absent"
        while time.monotonic() < deadline:
            self.check_interrupt()
            if self.status_path.exists() and read_json(self.status_path).get("status") == "complete":
                raise TrialError("latency trial is already complete")
            status_path = BASELINE / "study-status.json"
            launch_path = BASELINE / "controller-launch.json"
            if status_path.is_file() and launch_path.is_file():
                status, launch = read_json(status_path), read_json(launch_path)
                pid = launch.get("pid") if isinstance(launch, dict) else None
                if pid != EXPECTED_BASELINE_CONTROLLER_PID \
                        or status.get("controller_pid") != pid:
                    raise TrialError("baseline controller PID differs from recorded root launch")
                if launch.get("source_sha256") != BASELINE_SOURCE_SHA256:
                    raise TrialError("baseline controller source SHA-256 differs")
                identity = process_identity(pid)
                free = lock_is_free(BASELINE / ".controller.lock")
                if status.get("status") == "complete" and not identity.get("alive") and free:
                    current = read_json(BASELINE / "current-launch.json")
                    if not isinstance(current, dict) or current.get("default_restored") is not True \
                            or not isinstance(current.get("server"), dict):
                        raise TrialError("baseline final current-launch receipt is malformed")
                    self.handoff_acquired = True
                    return {"status": status, "launch": launch, "current": current}
                latest = f"status={status.get('status')}, alive={identity.get('alive')}, lock_free={free}"
            time.sleep(self.args.poll)
        raise TrialError(f"timed out waiting for baseline handoff: {latest}")

    def initialize(self, handoff: Mapping[str, Any]) -> None:
        pins = {
            BASELINE_SOURCE: BASELINE_SOURCE_SHA256,
            MANIFEST: MANIFEST_SHA256,
            MODEL_PLAN: MODEL_PLAN_SHA256,
            CPU_BUILD_RECEIPT: CPU_BUILD_RECEIPT_SHA256,
            OPENCL_BUILD_RECEIPT: OPENCL_BUILD_RECEIPT_SHA256,
        }
        for path, expected in pins.items():
            if sha256(path) != expected:
                raise TrialError(f"pinned input SHA-256 differs: {path}")
        manifest, plan = read_json(MANIFEST), read_json(MODEL_PLAN)
        cases = manifest.get("cases") if isinstance(manifest, dict) else None
        models = plan.get("models") if isinstance(plan, dict) else None
        self.cases = [copy.deepcopy(row) for row in cases or []
                      if isinstance(row, dict) and row.get("suite") == "routing"]
        self.model = next((copy.deepcopy(row) for row in models or []
                           if isinstance(row, dict) and row.get("key") == MODEL_KEY), {})
        if len(self.cases) != 8 or not self.model \
                or self.model.get("remote_path") != REMOTE_MODEL \
                or self.model.get("sha256") != MODEL_SHA256 \
                or self.model.get("size") != MODEL_SIZE \
                or self.model.get("alias") != ALIAS:
            raise TrialError("frozen routing cases or LFM2.6 model binding differs")
        for case in self.cases:
            if not {"id", "messages", "expected", "public_schema"} <= set(case):
                raise TrialError("routing case lacks required fields")
            payload = build_request(case, self.model)
            if any(key in payload for key in ("expected", "required_keys")):
                raise TrialError("answer data leaked into request payload")
        cpu_receipt = read_json(CPU_BUILD_RECEIPT)
        opencl_receipt = read_json(OPENCL_BUILD_RECEIPT)
        if cpu_receipt.get("source", {}).get("commit") != SOURCE_COMMIT \
                or cpu_receipt.get("artifacts", {}).get("bin/llama-server", {}).get("sha256") != CPU_SERVER_SHA256:
            raise TrialError("CPU runtime build receipt differs")
        if opencl_receipt.get("source", {}).get("commit") != SOURCE_COMMIT \
                or opencl_receipt.get("artifacts", {}).get("bin/llama-bench", {}).get("sha256") != OPENCL_BENCH_SHA256 \
                or opencl_receipt.get("build", {}).get("optimized_adreno_kernels") is not False:
            raise TrialError("generic OpenCL runtime build receipt differs")
        dynamic = {
            str(Path(__file__).resolve()): sha256(Path(__file__).resolve()),
            **{str(path): expected for path, expected in pins.items()},
            str(BASELINE / "study-status.json"): sha256(BASELINE / "study-status.json"),
            str(BASELINE / "current-launch.json"): sha256(BASELINE / "current-launch.json"),
            str(BASELINE / "controller-launch.json"): sha256(BASELINE / "controller-launch.json"),
        }
        pin_path = self.trial / "source-pins.json"
        if pin_path.is_file():
            if read_json(pin_path).get("sha256") != dynamic:
                raise TrialError("latency trial source pins changed")
        else:
            atomic_json(pin_path, {"schema_version": 1, "created_at": now(), "sha256": dynamic})
        self.state = {
            "schema_version": 1, "kind": STUDY_KIND,
            "claim": "small same-model routing latency trial; not a general benchmark",
            "created_at": now(), "status": "starting", "trial": str(self.trial),
            "baseline_controller_pid": handoff["launch"]["pid"],
            "manifest_sha256": MANIFEST_SHA256,
            "model_plan_sha256": MODEL_PLAN_SHA256,
            "model_key": MODEL_KEY, "epochs": {},
        }
        self.save(status="running")

    def verify_remote_artifacts(self) -> None:
        script = r'''import hashlib,json,pathlib,sys
p=json.loads(sys.argv[1]); out={}
for path in p["paths"]:
 h=hashlib.sha256(); f=pathlib.Path(path)
 with f.open("rb") as s:
  for chunk in iter(lambda:s.read(1024*1024),b""): h.update(chunk)
 out[path]={"sha256":h.hexdigest(),"size":f.stat().st_size}
artifact=json.loads((pathlib.Path(p["model"]).parent/"artifact.json").read_text())
print(json.dumps({"files":out,"model_stat":pathlib.Path(p["model"]).stat().st_size,"model_artifact":artifact}))'''
        cpu = REMOTE_CPU_RUNTIME + "/bin/llama-server"
        gpu = REMOTE_OPENCL_RUNTIME + "/bin/llama-bench"
        value = self.remote_json(script, {"paths": [cpu, gpu], "model": REMOTE_MODEL}, timeout=120)
        if value.get("files", {}).get(cpu, {}).get("sha256") != CPU_SERVER_SHA256 \
                or value.get("files", {}).get(gpu, {}).get("sha256") != OPENCL_BENCH_SHA256 \
                or value.get("model_stat") != MODEL_SIZE \
                or value.get("model_artifact", {}).get("actual_sha256") != MODEL_SHA256:
            raise TrialError("remote runtime/model artifact verification differs")
        atomic_json(self.trial / "remote-artifact-verification.json",
                    {"verified_at": now(), **value})

    def validate_baseline_server(self, receipt: Mapping[str, Any]) -> None:
        models = read_json(MODEL_PLAN)["models"]
        model = next(row for row in models if row["key"] == "lfm25-12b-instruct")
        expected = [
            REMOTE_CPU_RUNTIME + "/bin/llama-server", "-m", model["remote_path"],
            "--alias", model["alias"], "--host", "127.0.0.1", "--port", "18085",
            "-c", "16384", "-np", "1", "-t", "4", "-tb", "4", "-b", "512",
            "-ub", "128", "-ngl", "0", "--jinja", "--reasoning-format", "deepseek",
            "--reasoning-effort", "none", "--threads-http", "2",
        ]
        if receipt.get("argv") != expected or receipt.get("model_key") != model["key"] \
                or receipt.get("alias") != model["alias"]:
            raise TrialError("baseline restored LFM1.2 server argv differs")

    def stop_owned(self, receipt: Mapping[str, Any], label: str) -> dict[str, Any]:
        identity = self.remote_process(int(receipt.get("pid", -1)))
        if not verify_owned_process(receipt, identity, label):
            return {"stopped": True, "already_exited": True, "identity": identity}
        script = r'''import json,os,pathlib,signal,sys,time
p=json.loads(sys.argv[1]); pid=int(p["pid"]); b=pathlib.Path("/proc")/str(pid)
def ident():
 try:
  s=(b/"stat").read_text(); t=s[s.rfind(")")+2:].split(); a=[x.decode("utf-8","surrogateescape") for x in (b/"cmdline").read_bytes().split(b"\0") if x]; return {"pid":pid,"alive":t[0]!="Z","state":t[0],"start_ticks":int(t[19]),"argv":a}
 except FileNotFoundError: return {"pid":pid,"alive":False,"state":"exited","argv":[]}
i=ident()
if not i["alive"]: print(json.dumps({"stopped":True,"already_exited":True,"identity":i})); raise SystemExit(0)
if i["start_ticks"]!=p["start_ticks"] or i["argv"]!=p["argv"]: print(json.dumps({"error":"ownership mismatch","identity":i})); raise SystemExit(0)
os.kill(pid,signal.SIGTERM); end=time.monotonic()+60
while time.monotonic()<end:
 i=ident()
 if not i["alive"]: print(json.dumps({"stopped":True,"identity":i})); raise SystemExit(0)
 time.sleep(.2)
print(json.dumps({"error":"server did not stop","identity":ident()}))'''
        value = self.remote_json(script, dict(receipt), timeout=75)
        if not isinstance(value, dict) or value.get("stopped") is not True:
            raise TrialError(f"failed to stop {label}: {value}")
        return value

    def launch_server(self, ubatch: int, label: str) -> dict[str, Any]:
        command = server_command(ubatch)
        receipt_path = REMOTE_RESULTS + f"/{label}-server-launch.json"
        log_path = REMOTE_RESULTS + f"/{label}-server.log"
        script = r'''import json,os,pathlib,subprocess,sys,time
p=json.loads(sys.argv[1]); log=pathlib.Path(p["log"]); rec=pathlib.Path(p["receipt"]); log.parent.mkdir(parents=True,exist_ok=True); f=log.open("ab",buffering=0)
c=subprocess.Popen(p["argv"],stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True); b=pathlib.Path("/proc")/str(c.pid); end=time.monotonic()+3; a=[]
while time.monotonic()<end:
 if not b.exists(): break
 a=[x.decode("utf-8","surrogateescape") for x in (b/"cmdline").read_bytes().split(b"\0") if x]
 if a and a[0].endswith("/llama-server"): break
 time.sleep(.01)
if not a or not a[0].endswith("/llama-server"): print(json.dumps({"error":"server exited or wrapper did not exec"})); raise SystemExit(0)
s=(b/"stat").read_text(); t=s[s.rfind(")")+2:].split(); v={"schema_version":1,"kind":"tachyon_worker_latency_server","pid":c.pid,"start_ticks":int(t[19]),"argv":a,"launch_argv":p["argv"],"model_key":p["model_key"],"alias":p["alias"],"ubatch":p["ubatch"],"launched_at":p["at"],"log":str(log)}
x=rec.with_name(rec.name+".tmp-"+str(os.getpid())); x.write_text(json.dumps(v,sort_keys=True)+"\n"); os.replace(x,rec); print(json.dumps(v))'''
        value = self.remote_json(script, {"argv": command, "receipt": receipt_path,
            "log": log_path, "model_key": MODEL_KEY, "alias": ALIAS,
            "ubatch": ubatch, "at": now()})
        if not isinstance(value, dict) or value.get("error"):
            raise TrialError(f"server launch failed: {value}")
        if value.get("argv") != actual_exec_argv(command) \
                or value.get("launch_argv") != command:
            raise TrialError("launched server argv differs")
        verify_owned_process(value, self.remote_process(value["pid"]), "launched latency server")
        value["remote_receipt"] = receipt_path
        atomic_json(self.trial / f"{label}-server-launch.json", value)
        self.current = value
        self.event("server_launched", label=label, ubatch=ubatch, pid=value["pid"])
        return value

    def wait_endpoint(self, receipt: Mapping[str, Any]) -> None:
        deadline, latest = time.monotonic() + self.args.server_timeout, "not queried"
        while time.monotonic() < deadline:
            self.check_interrupt()
            if not verify_owned_process(receipt, self.remote_process(receipt["pid"]), "latency server"):
                raise TrialError("latency server exited during load")
            try:
                with urlopen(self.args.endpoint.rstrip("/") + "/models", timeout=10) as response:
                    value = json.load(response)
                aliases = [row.get("id") for row in value.get("data", []) if isinstance(row, dict)]
                if aliases == [ALIAS] or (ALIAS in aliases and len(set(aliases)) == 1):
                    return
                latest = repr(aliases)
            except (OSError, HTTPError, URLError, json.JSONDecodeError) as exc:
                latest = str(exc)
            time.sleep(2)
        raise TrialError(f"endpoint did not advertise {ALIAS}: {latest}")

    def run_case(self, epoch: Mapping[str, Any], sequence: int,
                 case: Mapping[str, Any]) -> dict[str, Any]:
        directory = self.trial / "epochs" / str(epoch["key"]) / f"{sequence:02d}-{case['id']}"
        result_path = directory / "result.json"
        payload = build_request(case, self.model)
        binding = {"epoch": epoch, "case_id": case["id"],
                   "request_sha256": json_sha(payload),
                   "manifest_sha256": MANIFEST_SHA256,
                   "model_plan_sha256": MODEL_PLAN_SHA256}
        if result_path.is_file():
            value = read_json(result_path)
            if any(value.get(key) != expected for key, expected in binding.items()):
                raise TrialError(f"saved case binding differs: {case['id']}")
            return value
        directory.mkdir(parents=True, exist_ok=True)
        request_path = directory / "request.json"
        if request_path.is_file() and read_json(request_path) != payload:
            raise TrialError(f"saved request differs: {case['id']}")
        if not request_path.exists():
            atomic_json(request_path, payload)
        attempt = 1
        while (directory / f"attempt-{attempt}-raw-sse.txt").exists():
            attempt += 1
        raw_path = directory / f"attempt-{attempt}-raw-sse.txt"
        request = Request(self.args.endpoint.rstrip("/") + "/chat/completions",
                          data=canonical(payload), headers={"Content-Type": "application/json"},
                          method="POST")
        raw_lines: list[bytes] = []
        first_ttft = final_ttft = None
        started = time.monotonic()
        error: dict[str, str] | None = None
        with raw_path.open("xb", buffering=0) as raw_stream:
            try:
                with urlopen(request, timeout=self.args.request_timeout) as response:
                    for raw in response:
                        raw_stream.write(raw)
                        raw_lines.append(raw)
                        parsed = parse_sse_lines([raw])
                        if first_ttft is None and (parsed["reasoning_text"] or parsed["final_text"]):
                            first_ttft = time.monotonic() - started
                        if final_ttft is None and parsed["final_text"]:
                            final_ttft = time.monotonic() - started
            except (OSError, HTTPError, URLError, UnicodeError,
                    json.JSONDecodeError, TrialError) as exc:
                error = {"type": type(exc).__name__, "message": str(exc)}
            os.fsync(raw_stream.fileno())
        wall = time.monotonic() - started
        parsed = parse_sse_lines(raw_lines)
        response_value = {**parsed, "wall_seconds": wall,
                          "ttft_first_seconds": first_ttft,
                          "ttft_final_seconds": final_ttft, "error": error,
                          "attempt": attempt}
        atomic_json(directory / "response.json", response_value)
        evaluation = evaluate_route(case, parsed["final_text"])
        result = {"schema_version": 1, "status": "api_error" if error else "completed",
                  **binding, "sequence": sequence, "cold_first_request": sequence == 0,
                  "shared_prefix_warm_request": sequence > 0,
                  "final_text": parsed["final_text"],
                  "reasoning_text": parsed["reasoning_text"],
                  "finish_reason": parsed["finish_reason"], "usage": parsed["usage"],
                  "native_timings": parsed["native_timings"],
                  "ttft_first_seconds": first_ttft, "ttft_final_seconds": final_ttft,
                  "wall_seconds": wall, "evaluation": evaluation, "error": error,
                  "finished_at": now()}
        atomic_json(result_path, result)
        self.event("case_terminal", epoch=epoch["key"], case=case["id"],
                   status=result["status"], cache_n=parsed["native_timings"].get("cache_n"))
        return result

    def summarize_epoch(self, epoch: Mapping[str, Any],
                        results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        warm = list(results[1:])
        def numbers(field: str) -> list[float]:
            return [float(row[field]) for row in warm
                    if isinstance(row.get(field), (int, float))]
        def timings(field: str) -> list[float]:
            return [float(row.get("native_timings", {}).get(field)) for row in warm
                    if isinstance(row.get("native_timings", {}).get(field), (int, float))]
        value = {
            "schema_version": 1, "epoch": dict(epoch), "request_count": len(results),
            "cold_request_count": 1, "warm_shared_prefix_request_count": len(warm),
            "prompt_characterization": "eight distinct routing prompts with one repeated system-policy prefix",
            "semantic_passes": sum(row.get("evaluation", {}).get("semantic_pass") is True for row in results),
            "completed": sum(row.get("status") == "completed" for row in results),
            "warm_ttft_first_median_seconds": statistics.median(numbers("ttft_first_seconds")) if numbers("ttft_first_seconds") else None,
            "warm_ttft_first_p95_seconds": percentile(numbers("ttft_first_seconds"), .95),
            "warm_wall_median_seconds": statistics.median(numbers("wall_seconds")) if numbers("wall_seconds") else None,
            "warm_wall_p95_seconds": percentile(numbers("wall_seconds"), .95),
            "warm_prompt_ms_median": statistics.median(timings("prompt_ms")) if timings("prompt_ms") else None,
            "warm_cache_n_median": statistics.median(timings("cache_n")) if timings("cache_n") else None,
            "warm_prompt_n_median": statistics.median(timings("prompt_n")) if timings("prompt_n") else None,
            "created_at": now(),
        }
        atomic_json(self.trial / "epochs" / str(epoch["key"]) / "summary.json", value)
        return value

    def run_epoch(self, epoch: Mapping[str, Any]) -> dict[str, Any]:
        summary_path = self.trial / "epochs" / str(epoch["key"]) / "summary.json"
        if summary_path.is_file():
            return read_json(summary_path)
        receipt = self.launch_server(int(epoch["ubatch"]), str(epoch["key"]))
        try:
            self.wait_endpoint(receipt)
            before = self.resource_snapshot(
                f"epochs/{epoch['key']}/resource-before.json", receipt)
            results = [self.run_case(epoch, sequence, case)
                       for sequence, case in enumerate(self.cases)]
            after = self.resource_snapshot(
                f"epochs/{epoch['key']}/resource-after.json", receipt)
            summary = self.summarize_epoch(epoch, results)
            summary["resource_before"] = before
            summary["resource_after"] = after
            atomic_json(summary_path, summary)
        finally:
            stop = self.stop_owned(receipt, f"{epoch['key']} server")
            atomic_json(self.trial / f"{epoch['key']}-server-stop.json",
                        {"stopped_at": now(), "ownership": receipt, "result": stop})
            atomic_bytes(self.trial / f"{epoch['key']}-server.log",
                         self.remote_bytes(str(receipt["log"])))
            self.current = None
        self.state["epochs"][str(epoch["key"])] = summary
        self.save(status="running")
        return summary

    def run_opencl_benchmark(self) -> dict[str, Any]:
        result_path = self.trial / "opencl-generic-ngl4-result.json"
        if result_path.is_file():
            return read_json(result_path)
        command = benchmark_command()
        receipt_path = REMOTE_RESULTS + "/opencl-generic-ngl4-bench.json"
        stdout_path = REMOTE_RESULTS + "/opencl-generic-ngl4-stdout.json"
        stderr_path = REMOTE_RESULTS + "/opencl-generic-ngl4-stderr.log"
        before = self.resource_snapshot("opencl-generic-ngl4-resource-before.json")
        script = r'''import json,os,pathlib,signal,subprocess,sys,time
p=json.loads(sys.argv[1]); root=pathlib.Path(p["stdout"]).parent; root.mkdir(parents=True,exist_ok=True); out=open(p["stdout"],"wb",buffering=0); err=open(p["stderr"],"wb",buffering=0)
started=time.monotonic(); c=subprocess.Popen(p["argv"],stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True); b=pathlib.Path("/proc")/str(c.pid); time.sleep(.05)
try:
 s=(b/"stat").read_text(); t=s[s.rfind(")")+2:].split(); a=[x.decode("utf-8","surrogateescape") for x in (b/"cmdline").read_bytes().split(b"\0") if x]; start_ticks=int(t[19])
except FileNotFoundError: a=[]; start_ticks=None
timed_out=False
try: rc=c.wait(timeout=p["timeout"])
except subprocess.TimeoutExpired:
 timed_out=True
 try:
  s=(b/"stat").read_text(); t=s[s.rfind(")")+2:].split(); cur=[x.decode("utf-8","surrogateescape") for x in (b/"cmdline").read_bytes().split(b"\0") if x]
  if int(t[19])!=start_ticks or cur!=a: raise RuntimeError("benchmark ownership changed before timeout stop")
  os.kill(c.pid,signal.SIGTERM); rc=c.wait(timeout=15)
 except subprocess.TimeoutExpired:
  os.kill(c.pid,signal.SIGKILL); rc=c.wait(timeout=5)
out.close(); err.close(); v={"schema_version":1,"kind":"tachyon_worker_latency_opencl_bench","launch_argv":p["argv"],"pid":c.pid,"start_ticks":start_ticks,"argv":a,"returncode":rc,"timed_out":timed_out,"wall_seconds":time.monotonic()-started,"stdout":p["stdout"],"stderr":p["stderr"],"finished_at":p["at"]}
rec=pathlib.Path(p["receipt"]); tmp=rec.with_name(rec.name+".tmp-"+str(os.getpid())); tmp.write_text(json.dumps(v,sort_keys=True)+"\n"); os.replace(tmp,rec); print(json.dumps(v))'''
        receipt = self.remote_json(script, {"argv": command, "timeout": self.args.bench_timeout,
            "receipt": receipt_path, "stdout": stdout_path, "stderr": stderr_path,
            "at": now()}, timeout=self.args.bench_timeout + 30)
        if not isinstance(receipt, dict) or receipt.get("launch_argv") != command:
            raise TrialError("OpenCL benchmark receipt differs")
        if receipt.get("argv") != actual_exec_argv(command):
            raise TrialError("OpenCL benchmark executable argv differs")
        stdout, stderr = self.remote_bytes(stdout_path), self.remote_bytes(stderr_path)
        atomic_bytes(self.trial / "opencl-generic-ngl4-stdout.json", stdout)
        atomic_bytes(self.trial / "opencl-generic-ngl4-stderr.log", stderr)
        atomic_json(self.trial / "opencl-generic-ngl4-bench.json", receipt)
        parsed: Any = None
        parse_error: str | None = None
        try:
            parsed = json.loads(stdout.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError) as exc:
            parse_error = str(exc)
        rates = benchmark_rates(parsed)
        log = stderr.decode("utf-8", "replace")
        offload = re.findall(r"offload(?:ing|ed)?\s+(\d+)\s+(?:repeating\s+)?layers?", log, re.I)
        result = {"schema_version": 1,
                  "status": "complete" if receipt.get("returncode") == 0 and all(value is not None for value in rates.values()) else "failed",
                  "variant": "generic-opencl-ngl4", "requested_gpu_layers": 4,
                  "command": command, "receipt": receipt, **rates,
                  "reported_offloaded_layers": int(offload[-1]) if offload else None,
                  "parse_error": parse_error,
                  "compiler_failure_detected": bool(re.search(r"clBuildProgram|FATAL ERROR|Assertion failed|Stride != 0", log, re.I)),
                  "quality_claim": False,
                  "interpretation": "llama-bench speed probe only; no model answer quality was tested",
                  "resource_before": before,
                  "resource_after": self.resource_snapshot(
                      "opencl-generic-ngl4-resource-after.json"),
                  "finished_at": now()}
        atomic_json(result_path, result)
        self.event("opencl_benchmark_terminal", status=result["status"],
                   returncode=receipt.get("returncode"))
        return result

    def choose_ubatch(self, summaries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        by_key = {row["epoch"]["key"]: row for row in summaries}
        a1, b, a2 = by_key["a1-ub128"], by_key["b-ub32"], by_key["a2-ub128"]
        metrics = ("warm_ttft_first_median_seconds", "warm_wall_median_seconds")
        complete = all(row.get("completed") == 8 for row in (a1, b, a2))
        quality = b.get("semantic_passes") >= max(a1.get("semantic_passes", 0), a2.get("semantic_passes", 0))
        clearly_faster = all(isinstance(row.get(metric), (int, float)) for row in (a1, b, a2) for metric in metrics) \
            and all(b[metric] < a1[metric] and b[metric] < a2[metric] for metric in metrics)
        selected = 32 if complete and quality and clearly_faster else 128
        return {"selected_ubatch": selected, "complete": complete,
                "no_quality_regression": quality, "b_beats_both_a_epochs_on_warm_ttft_and_wall": clearly_faster,
                "selection_rule": "ub32 only when all epochs complete, semantic passes do not regress, and both warm TTFT and wall medians beat each bracketing ub128 epoch"}

    def restore_final(self, ubatch: int, selection: Mapping[str, Any]) -> dict[str, Any]:
        receipt = self.launch_server(ubatch, "final-lfm26")
        self.wait_endpoint(receipt)
        identity = self.remote_process(receipt["pid"])
        verify_owned_process(receipt, identity, "final LFM2.6 server")
        value = {"schema_version": 1, "kind": "tachyon_worker_latency_final_service",
                 "selected_at": now(), "endpoint": self.args.endpoint,
                 "remote_endpoint": "http://127.0.0.1:18085/v1",
                 "model_key": MODEL_KEY, "alias": ALIAS, "ubatch": ubatch,
                 "server": receipt, "verified_identity": identity,
                 "selection": dict(selection), "fast_profile": copy.deepcopy(self.model["recipe"]),
                 "request_controls": {"max_tokens": 512, "seed": 20261023,
                                      "response_format": "case_public_json_schema"}}
        atomic_json(self.trial / "final-service.json", value)
        atomic_json(self.trial / "current-launch.json", value)
        self.current = receipt
        self.event("final_service_ready", pid=receipt["pid"], ubatch=ubatch)
        return value

    def consolidate(self, summaries: Sequence[Mapping[str, Any]],
                    opencl: Mapping[str, Any], selection: Mapping[str, Any],
                    final: Mapping[str, Any]) -> None:
        rows = []
        for summary in summaries:
            before = summary["resource_before"]
            after = summary["resource_after"]
            rows.append({"epoch": summary["epoch"]["key"], "arm": summary["epoch"]["arm"],
                         "ubatch": summary["epoch"]["ubatch"],
                         "request_count": summary["request_count"],
                         "cold_request_count": summary["cold_request_count"],
                         "warm_request_count": summary["warm_shared_prefix_request_count"],
                         "semantic_passes": summary["semantic_passes"],
                         "warm_cache_n_median": summary["warm_cache_n_median"],
                         "warm_prompt_n_median": summary["warm_prompt_n_median"],
                         "warm_prompt_ms_median": summary["warm_prompt_ms_median"],
                         "warm_ttft_first_median_seconds": summary["warm_ttft_first_median_seconds"],
                         "warm_wall_median_seconds": summary["warm_wall_median_seconds"],
                         "thermal_max_c_before": before.get("thermal_max_millidegrees_c") / 1000
                         if isinstance(before.get("thermal_max_millidegrees_c"), int) else None,
                         "thermal_max_c_after": after.get("thermal_max_millidegrees_c") / 1000
                         if isinstance(after.get("thermal_max_millidegrees_c"), int) else None,
                         "mem_available_kib_before": kib_value(before.get("meminfo", {}).get("MemAvailable")),
                         "mem_available_kib_after": kib_value(after.get("meminfo", {}).get("MemAvailable")),
                         "server_rss_kib_before": kib_value((before.get("process") or {}).get("status", {}).get("VmRSS")),
                         "server_rss_kib_after": kib_value((after.get("process") or {}).get("status", {}).get("VmRSS")),
                         "policy4_cur_khz_before": before.get("cpufreq_policy4_policy7", {}).get("4", {}).get("scaling_cur_freq"),
                         "policy4_cur_khz_after": after.get("cpufreq_policy4_policy7", {}).get("4", {}).get("scaling_cur_freq"),
                         "policy4_max_khz": after.get("cpufreq_policy4_policy7", {}).get("4", {}).get("scaling_max_freq"),
                         "policy7_cur_khz_before": before.get("cpufreq_policy4_policy7", {}).get("7", {}).get("scaling_cur_freq"),
                         "policy7_cur_khz_after": after.get("cpufreq_policy4_policy7", {}).get("7", {}).get("scaling_cur_freq"),
                         "policy7_max_khz": after.get("cpufreq_policy4_policy7", {}).get("7", {}).get("scaling_max_freq")})
        atomic_json(self.trial / "summary.json", {
            "schema_version": 1, "kind": STUDY_KIND, "created_at": now(),
            "claim": "small A/B/A test of eight distinct shared-prefix routing prompts on one LFM2.6 model; cold request excluded from warm medians",
            "limitations": ["not a general workload benchmark", "only one eight-prompt ordering per epoch",
                            "wall time includes answer-length and scheduler variation",
                            "OpenCL result is a speed/load probe without answer-quality evidence"],
            "rows": rows, "selection": dict(selection), "opencl": dict(opencl),
            "final_service": dict(final),
        })
        output = io.StringIO(newline="")
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
        atomic_bytes(self.trial / "summary.csv", output.getvalue().encode("utf-8"))

    def run(self) -> int:
        self.acquire_lock()
        for signum in (signal.SIGINT, signal.SIGTERM):
            signal.signal(signum, self.handle_signal)
        if self.status_path.is_file() and read_json(self.status_path).get("status") == "complete":
            return 0
        failure: BaseException | None = None
        summaries: list[dict[str, Any]] = []
        selection: dict[str, Any] = {"selected_ubatch": 128, "reason": "safe failure fallback"}
        final: dict[str, Any] | None = None
        try:
            handoff = self.wait_for_baseline()
            self.initialize(handoff)
            self.verify_remote_artifacts()
            inherited = handoff["current"]["server"]
            self.validate_baseline_server(inherited)
            verify_owned_process(inherited, self.remote_process(inherited["pid"]),
                                 "baseline restored LFM1.2 server")
            stopped = self.stop_owned(inherited, "baseline restored LFM1.2 server")
            self.service_taken_over = True
            atomic_json(self.trial / "baseline-lfm12-server-stop.json",
                        {"stopped_at": now(), "ownership": inherited, "result": stopped})
            self.event("baseline_handoff_owned", stopped_pid=inherited["pid"])
            for epoch in EPOCHS:
                self.check_interrupt()
                summaries.append(self.run_epoch(epoch))
            self.save(status="opencl_benchmark")
            opencl = self.run_opencl_benchmark()
            selection = self.choose_ubatch(summaries)
            final = self.restore_final(int(selection["selected_ubatch"]), selection)
            self.consolidate(summaries, opencl, selection, final)
        except BaseException as exc:
            failure = exc
        finally:
            if self.service_taken_over and final is None:
                try:
                    if self.current is not None:
                        stopped = self.stop_owned(self.current, "failed-trial current server")
                        atomic_json(self.trial / "failure-current-server-stop.json",
                                    {"stopped_at": now(), "ownership": self.current, "result": stopped})
                        self.current = None
                    final = self.restore_final(128, selection)
                except BaseException as restore_exc:
                    failure = restore_exc if failure is None else TrialError(
                        f"{failure}; final LFM2.6 restore also failed: {restore_exc}")
        if failure is not None:
            if not self.state:
                self.state = {"schema_version": 1, "kind": STUDY_KIND,
                              "created_at": now(), "epochs": {}}
            self.save(status="failed", error=str(failure), finished_at=now(),
                      final_service=str(self.trial / "final-service.json") if final else None)
            self.event("trial_failed", error=str(failure))
            print(f"tachyon_worker_latency_trial: {failure}", file=sys.stderr)
            return 1
        self.save(status="complete", error=None, finished_at=now(),
                  selected_ubatch=selection["selected_ubatch"],
                  final_service=str(self.trial / "final-service.json"))
        self.event("trial_complete", selected_ubatch=selection["selected_ubatch"])
        return 0


def parse_args(arguments: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial", type=Path, default=DEFAULT_TRIAL)
    args = parser.parse_args(arguments)
    args.remote = "akovanda@tachyon"
    args.endpoint = ENDPOINT
    args.wait_timeout = 86_400.0
    args.poll = 10.0
    args.server_timeout = 600.0
    args.request_timeout = 180.0
    args.bench_timeout = 120.0
    return args


def main(arguments: Iterable[str] = sys.argv[1:]) -> int:
    return Controller(parse_args(arguments)).run()


if __name__ == "__main__":
    raise SystemExit(main())
