#!/usr/bin/env python3
"""Print benchmark metadata without showing generated answers or reasoning."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", default="specialists-final12k16k")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--recent", type=int, default=4)
    args = parser.parse_args()
    paths = set()
    for pattern in ("-0*-*.jsonl", "-specialist-code.jsonl", "-specialist-prose.jsonl", "-sampler-requests.jsonl"):
        paths.update(args.results_dir.glob(args.prefix + "*" + pattern))
    paths = sorted(paths, key=lambda p: p.stat().st_mtime, reverse=True)[:max(1, args.recent)]
    summaries = []
    for path in paths:
        rows, malformed = [], 0
        for line in path.read_text().splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue
            if row.get("case_id"):
                rows.append(row)
        outcomes = Counter()
        for row in rows:
            grade = row.get("evaluation") or {}
            outcomes[grade.get("outcome") or ("passed" if grade.get("passed") is True else "not_passed" if grade.get("passed") is False else "ungraded")] += 1
        latest = None
        if rows:
            row = rows[-1]
            response = row.get("response") or {}
            grade = row.get("evaluation") or {}
            latest = {"case_id": row["case_id"], "call_id": row.get("call_id"),
                      "finished_at": row.get("finished_at"), "reported_grade": grade.get("outcome") or grade.get("passed"),
                      "tokens": response.get("actual_completion_tokens"),
                      "wall_seconds": response.get("timing", {}).get("wall_seconds"),
                      "native_tps": response.get("throughput", {}).get("completion_tokens_per_second_native"),
                      "truncated": response.get("truncated"), "error_type": type(row["error"]).__name__ if row.get("error") else None}
        summaries.append({"source": str(path), "completed_rows": len(rows), "reported_outcomes": dict(outcomes),
                          "unparsed_lines_possibly_current_write": malformed, "latest": latest})
    print(json.dumps({"at": datetime.now(timezone.utc).isoformat(), "scope": "metadata only; prose grades are mechanical constraints, not writing-quality scores", "files": summaries}))


if __name__ == "__main__":
    main()
