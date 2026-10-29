#!/usr/bin/env python3
"""Append read-only machine observations while benchmark models are running."""
import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time


def query_gpu(fields, query):
    result = subprocess.run(
        ["nvidia-smi", f"--{query}={','.join(fields)}", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=10, check=False,
    )
    if result.returncode:
        return {"query_error": result.stderr.strip(), "returncode": result.returncode}
    return [dict(zip(fields, (part.strip() for part in row)))
            for row in csv.reader(result.stdout.splitlines()) if row]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stop-file", type=Path, required=True)
    parser.add_argument("--start-at", help="ISO UTC timestamp; otherwise start immediately")
    parser.add_argument("--seconds", type=int, default=64800)
    parser.add_argument("--interval", type=int, default=30)
    args = parser.parse_args()
    if args.seconds <= 0 or args.interval <= 0:
        parser.error("duration and interval must be positive")
    start_at = datetime.fromisoformat(args.start_at.replace("Z", "+00:00")).timestamp() if args.start_at else time.time()
    while time.time() < start_at and not args.stop_file.exists():
        time.sleep(max(0, min(30, start_at - time.time())))
    started = time.monotonic()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["timestamp", "utilization.gpu", "utilization.memory", "memory.used", "memory.total",
              "power.draw", "clocks.sm", "clocks.mem", "temperature.gpu",
              "clocks_event_reasons.hw_power_brake_slowdown"]
    while not args.stop_file.exists() and time.monotonic() - started < args.seconds:
        record = {"at": datetime.now(timezone.utc).isoformat(), "study": "specialists-2026-10-19",
                  "note": "read-only continuation; other machine workloads are not controlled"}
        try:
            record["gpu"] = query_gpu(fields, "query-gpu")
            record["gpu_processes"] = query_gpu(["pid", "used_gpu_memory"], "query-compute-apps")
            record["load_average"] = list(os.getloadavg())
            memory = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
            record["meminfo"] = {key: memory[key].strip() for key in ("MemFree", "MemAvailable", "SwapFree")}
        except Exception as exc:
            record["observation_error"] = str(exc)
        with args.output.open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
