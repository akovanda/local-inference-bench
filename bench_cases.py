"""Versioned benchmark cases and offline-safe response evaluators.

The prompts and hidden checks live in this module so benchmark results can record
the module's SHA-256 hash and remain attributable to an exact case definition.
Only Python's standard library is used.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import os
import random
import re
import resource
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Sequence


CASE_VERSION = "2026-10-19.2"


@dataclass(frozen=True)
class BenchmarkCase:
    case_id: str
    suite: str
    prompt: str
    max_tokens: int
    evaluator: str
    expected: Any = None
    function_name: str | None = None
    tests: tuple[Mapping[str, Any], ...] = field(default_factory=tuple)
    metadata: Mapping[str, Any] = field(default_factory=dict)


QUICK_CASES = (
    BenchmarkCase(
        case_id="quick_generation",
        suite="quick",
        max_tokens=224,
        evaluator="generation",
        prompt=(
            "Write exactly eight numbered sentences explaining how a bounded work queue "
            "prevents overload in a small web service. Each sentence must contain 15 to "
            "22 words. Cover admission, capacity, backpressure, workers, timeouts, metrics, "
            "shutdown, and one tradeoff, in that order. Stop after sentence 8."
        ),
        metadata={"purpose": "short generation throughput"},
    ),
    BenchmarkCase(
        case_id="quick_json_extract",
        suite="quick",
        max_tokens=256,
        evaluator="json_exact",
        expected={
            "failed_services": ["api", "worker"],
            "retry_total": 7,
            "latest_success_id": "evt-105",
            "p95_ms": 240,
        },
        prompt=(
            "Extract facts from the records below. Return only one JSON object with exactly "
            "these keys and types: failed_services (array of strings, sorted), retry_total "
            "(integer sum for failed records), latest_success_id (string for the newest "
            "successful timestamp), and p95_ms (integer from the summary record).\n\n"
            "2026-04-03T09:00:00Z id=evt-101 service=api status=failed retries=2\n"
            "2026-04-03T09:01:00Z id=evt-102 service=cache status=ok retries=0\n"
            "2026-04-03T09:03:00Z id=evt-103 service=worker status=failed retries=4\n"
            "2026-04-03T09:04:00Z id=evt-104 service=api status=failed retries=1\n"
            "2026-04-03T09:07:00Z id=evt-105 service=cache status=ok retries=0\n"
            "summary window=5m p50_ms=81 p95_ms=240 p99_ms=410"
        ),
    ),
)


QUALITY_CASES = (
    BenchmarkCase(
        case_id="quality_merge_intervals",
        suite="quality",
        max_tokens=1536,
        evaluator="python_function",
        function_name="merge_half_open",
        prompt=(
            "Implement this Python function without imports:\n\n"
            "    def merge_half_open(intervals):\n\n"
            "Input is an iterable of two-item [start, end] integer sequences representing "
            "half-open intervals. Raise ValueError if an item is malformed, either endpoint "
            "is a bool or non-integer, or end < start. Discard empty intervals. Sort the "
            "remaining intervals and merge only genuine overlaps; touching intervals such as "
            "[1, 3] and [3, 5] must remain separate. Return a new list of [start, end] lists "
            "and do not mutate the input. Return only Python code (a code fence is allowed)."
        ),
        tests=(
            {"args": [[]], "expected": []},
            {"args": [[[1, 3], [3, 5]]], "expected": [[1, 3], [3, 5]]},
            {
                "args": [[[5, 8], [1, 4], [2, 6], [10, 10], [-2, 0]]],
                "expected": [[-2, 0], [1, 8]],
            },
            {"args": [[[2, 7], [2, 7], [3, 4]]], "expected": [[2, 7]]},
            {"args": [[[4, 2]]], "raises": "ValueError"},
            {"args": [[[False, 2]]], "raises": "ValueError"},
            {"args": [[[1, 2, 3]]], "raises": "ValueError"},
        ),
    ),
    BenchmarkCase(
        case_id="quality_stable_toposort",
        suite="quality",
        max_tokens=1536,
        evaluator="python_function",
        function_name="stable_toposort",
        prompt=(
            "Implement this Python function without imports:\n\n"
            "    def stable_toposort(nodes, edges):\n\n"
            "nodes is a sequence of unique hashable values in preferred order. edges contains "
            "(before, after) pairs. Return a topological ordering. Whenever several nodes are "
            "ready, choose the one earliest in the original nodes sequence. Duplicate edges "
            "have no additional effect. Raise ValueError for duplicate nodes, an edge endpoint "
            "not in nodes, or a cycle. Do not mutate either input. Return only Python code "
            "(a code fence is allowed)."
        ),
        tests=(
            {"args": [[], []], "expected": []},
            {
                "args": [["fetch", "lint", "build", "test", "ship"], [["build", "test"], ["fetch", "build"], ["lint", "test"], ["test", "ship"]]],
                "expected": ["fetch", "lint", "build", "test", "ship"],
            },
            {
                "args": [["c", "a", "b", "d"], [["a", "d"], ["b", "d"], ["a", "d"]]],
                "expected": ["c", "a", "b", "d"],
            },
            {"args": [["a", "a"], []], "raises": "ValueError"},
            {"args": [["a"], [["a", "missing"]]], "raises": "ValueError"},
            {"args": [["a", "b"], [["a", "b"], ["b", "a"]]], "raises": "ValueError"},
        ),
    ),
    BenchmarkCase(
        case_id="quality_deep_merge",
        suite="quality",
        max_tokens=1536,
        evaluator="python_function",
        function_name="deep_merge",
        prompt=(
            "Implement this Python function without imports:\n\n"
            "    def deep_merge(base, overlay):\n\n"
            "Both arguments must be dictionaries or the function raises TypeError. Merge "
            "dictionary values recursively. An overlay value of None deletes that key. Every "
            "other overlay value replaces the base value, including lists. Preserve normal "
            "dict insertion order: surviving base keys keep their positions and new overlay "
            "keys appear in overlay order. Return a fully independent copy of all nested dict "
            "and list containers and do not mutate either input. Values need only contain "
            "dicts, lists, and JSON scalar types. Return only Python code (a code fence is allowed)."
        ),
        tests=(
            {"args": [{}, {}], "expected": {}},
            {
                "args": [
                    {"service": {"port": 80, "flags": ["a"], "old": True}, "region": "us"},
                    {"service": {"port": 443, "old": None, "new": {"x": [1]}}, "zone": "a"},
                ],
                "expected": {"service": {"port": 443, "flags": ["a"], "new": {"x": [1]}}, "region": "us", "zone": "a"},
                "assert_inputs_unchanged": True,
                "mutate_result_path": ["service", "flags"],
            },
            {"args": [{"x": {"a": 1}}, {"x": [2, {"b": 3}]}], "expected": {"x": [2, {"b": 3}]}, "assert_inputs_unchanged": True, "mutate_result_path": ["x"]},
            {"args": [[], {}], "raises": "TypeError"},
            {"args": [{}, []], "raises": "TypeError"},
        ),
    ),
    BenchmarkCase(
        case_id="quality_retry_schedule",
        suite="quality",
        max_tokens=1280,
        evaluator="python_function",
        function_name="retry_delays",
        prompt=(
            "Implement this Python function without imports:\n\n"
            "    def retry_delays(attempts, base, factor, cap):\n\n"
            "attempts is the total number of attempts, so return max(attempts - 1, 0) delays. "
            "Delay i is min(cap, base * factor**i), beginning with i=0. attempts must be a "
            "non-boolean integer >= 0. base and cap must be non-boolean int or float >= 0, "
            "and factor must be non-boolean int or float >= 1; otherwise raise ValueError. "
            "Return a list and do not round values. Return only Python code (a code fence is allowed)."
        ),
        tests=(
            {"args": [0, 1, 2, 10], "expected": []},
            {"args": [1, 1, 2, 10], "expected": []},
            {"args": [6, 0.5, 2, 5], "expected": [0.5, 1.0, 2.0, 4.0, 5]},
            {"args": [5, 3, 1, 2], "expected": [2, 2, 2, 2]},
            {"args": [True, 1, 2, 3], "raises": "ValueError"},
            {"args": [3, -1, 2, 3], "raises": "ValueError"},
            {"args": [3, 1, 0.5, 3], "raises": "ValueError"},
        ),
    ),
    BenchmarkCase(
        case_id="quality_constraints",
        suite="quality",
        max_tokens=1024,
        evaluator="json_exact",
        expected={
            "order": ["lint", "build", "migrate", "deploy", "smoke"],
            "immediate_pair": ["lint", "build"],
            "dependency_chain": ["migrate", "deploy", "smoke"],
        },
        prompt=(
            "Five release jobs occupy distinct slots 1 through 5: lint, build, migrate, "
            "deploy, and smoke. Build is immediately after lint. Migrate is before deploy. "
            "Deploy is before smoke. Migrate is not first. Smoke is fifth. Determine the "
            "unique order. Return only JSON with exactly these keys and types: order (array "
            "of all five job strings), immediate_pair (the two strings lint and build in "
            "order), and dependency_chain (the three strings migrate, deploy, and smoke in "
            "that stated dependency order)."
        ),
    ),
)


def _hard_test(
    label: str,
    category: str,
    args: list[Any],
    *,
    expected: Any = None,
    raises: str | None = None,
    assert_inputs_unchanged: bool = False,
) -> Mapping[str, Any]:
    test: dict[str, Any] = {"label": label, "category": category, "args": args}
    if raises is not None:
        test["raises"] = raises
    else:
        test["expected"] = expected
        if assert_inputs_unchanged:
            test["assert_inputs_unchanged"] = True
    return test


def _weighted_reference(jobs: Sequence[Sequence[Any]]) -> list[str]:
    best_weight = -1
    best_ids: list[str] | None = None
    for mask in range(1 << len(jobs)):
        chosen = [jobs[index] for index in range(len(jobs)) if mask & (1 << index)]
        compatible = all(
            left[2] <= right[1] or right[2] <= left[1]
            for left_index, left in enumerate(chosen)
            for right in chosen[left_index + 1 :]
        )
        if not compatible:
            continue
        total = sum(job[3] for job in chosen)
        ids = sorted(job[0] for job in chosen)
        if total > best_weight or (total == best_weight and (best_ids is None or ids < best_ids)):
            best_weight = total
            best_ids = ids
    return best_ids if best_ids is not None else []


def _build_weighted_tests() -> tuple[Mapping[str, Any], ...]:
    tests: list[Mapping[str, Any]] = []
    fixed = (
        ("empty", "boundary", []),
        ("one", "boundary", [["solo", 2, 5, 7]]),
        ("touching_allowed", "boundary_touching", [["left", 0, 3, 4], ["right", 3, 6, 5]]),
        ("overlap_heavy_wins", "optimization", [["a", 0, 2, 4], ["b", 2, 5, 5], ["heavy", 0, 5, 10]]),
        ("lex_single_tie", "tie_break", [["z", 0, 4, 8], ["a", 0, 4, 8]]),
        ("lex_composite_tie", "tie_break", [["a", 0, 2, 5], ["d", 2, 4, 5], ["b", 0, 4, 10]]),
        ("all_zero_prefers_empty", "tie_break", [["a", 0, 1, 0], ["b", 1, 2, 0]]),
        ("zero_weight_lexical_prefix", "tie_break", [["a", 0, 1, 0], ["z", 1, 2, 4]]),
        ("negative_times", "boundary", [["past", -5, -2, 3], ["now", -2, 1, 4], ["wide", -4, 0, 8]]),
    )
    for label, category, jobs in fixed:
        tests.append(
            _hard_test(
                label,
                category,
                [jobs],
                expected=_weighted_reference(jobs),
                assert_inputs_unchanged=True,
            )
        )

    rng = random.Random(0x57A11)
    for case_index in range(18):
        jobs: list[list[Any]] = []
        for job_index in range(5 + case_index % 4):
            start = rng.randrange(-3, 13)
            end = start + rng.randrange(1, 6)
            jobs.append(
                [f"g{case_index:02d}_{chr(97 + job_index)}", start, end, rng.randrange(0, 13)]
            )
        rng.shuffle(jobs)
        tests.append(
            _hard_test(
                f"generated_{case_index:02d}",
                "generated_bruteforce",
                [jobs],
                expected=_weighted_reference(jobs),
                assert_inputs_unchanged=True,
            )
        )

    invalid = (
        ("jobs_not_list", "bad"),
        ("malformed_job", [["a", 0, 1]]),
        ("empty_id", [["", 0, 1, 2]]),
        ("non_string_id", [[7, 0, 1, 2]]),
        ("duplicate_id", [["a", 0, 1, 2], ["a", 1, 2, 3]]),
        ("bool_start", [["a", False, 1, 2]]),
        ("bool_end", [["a", 0, True, 2]]),
        ("nonpositive_span", [["a", 2, 2, 3]]),
        ("bool_weight", [["a", 0, 1, True]]),
        ("negative_weight", [["a", 0, 1, -1]]),
    )
    for label, jobs in invalid:
        tests.append(_hard_test(label, "validation", [jobs], raises="ValueError"))
    return tuple(tests)


def _temporal_reference(writes: Sequence[Sequence[Any]], queries: Sequence[Sequence[Any]]) -> list[Any]:
    answers: list[Any] = []
    for key, query_time in queries:
        latest: tuple[int, int, Any, Any] | None = None
        for position, (write_key, timestamp, value, ttl) in enumerate(writes):
            if write_key == key and timestamp <= query_time:
                candidate = (timestamp, position, value, ttl)
                if latest is None or candidate[:2] > latest[:2]:
                    latest = candidate
        if latest is None or latest[2] is None:
            answers.append(None)
        elif query_time >= latest[0] + latest[3]:
            answers.append(None)
        else:
            answers.append(latest[2])
    return answers


def _build_temporal_tests() -> tuple[Mapping[str, Any], ...]:
    tests: list[Mapping[str, Any]] = []
    fixed = (
        ("empty", "boundary", [], [["a", 0]]),
        ("unknown_key", "boundary", [["a", 0, "A", 5]], [["b", 2]]),
        ("exact_and_expiry", "ttl_boundary", [["a", 4, "A", 3]], [["a", 3], ["a", 4], ["a", 6], ["a", 7]]),
        ("expired_no_resurrection", "no_resurrection", [["a", 0, "old", 100], ["a", 10, "new", 2]], [["a", 9], ["a", 10], ["a", 12], ["a", 50]]),
        ("tombstone_no_resurrection", "no_resurrection", [["a", 0, "old", 100], ["a", 7, None, None]], [["a", 6], ["a", 7], ["a", 50]]),
        ("same_timestamp_later_wins", "same_timestamp", [["a", 3, "first", 10], ["a", 3, "second", 2]], [["a", 3], ["a", 5]]),
        ("timestamp_beats_input_position", "ordering", [["a", 8, "newer", 10], ["a", 2, "older", 100]], [["a", 9]]),
        ("keys_are_isolated", "key_isolation", [["a", 1, "A", 5], ["b", 1, "B", 8]], [["a", 2], ["b", 2]]),
        ("future_write_ignored", "ordering", [["a", 10, "future", 5], ["a", 1, "now", 20]], [["a", 5]]),
        ("same_timestamp_tombstone", "same_timestamp", [["a", 2, "A", 10], ["a", 2, None, None]], [["a", 2]]),
        ("new_value_after_tombstone", "no_resurrection", [["a", 0, "old", 20], ["a", 4, None, None], ["a", 7, "fresh", 4]], [["a", 5], ["a", 7], ["a", 11]]),
    )
    for label, category, writes, queries in fixed:
        tests.append(
            _hard_test(
                label,
                category,
                [writes, queries],
                expected=_temporal_reference(writes, queries),
                assert_inputs_unchanged=True,
            )
        )

    rng = random.Random(0x7E4F0)
    keys = ("alpha", "beta", "gamma")
    for case_index in range(18):
        writes: list[list[Any]] = []
        for write_index in range(7 + case_index % 4):
            key = rng.choice(keys)
            timestamp = rng.randrange(0, 16)
            if rng.randrange(5) == 0:
                writes.append([key, timestamp, None, None])
            else:
                writes.append([key, timestamp, f"v{case_index}_{write_index}", rng.randrange(1, 8)])
        queries = [[rng.choice(keys), rng.randrange(0, 23)] for _ in range(8)]
        tests.append(
            _hard_test(
                f"generated_{case_index:02d}",
                "generated_reference",
                [writes, queries],
                expected=_temporal_reference(writes, queries),
                assert_inputs_unchanged=True,
            )
        )

    invalid = (
        ("writes_not_list", "bad", []),
        ("queries_not_list", [], "bad"),
        ("malformed_write", [["a", 0, "A"]], []),
        ("empty_write_key", [["", 0, "A", 2]], []),
        ("bool_write_time", [["a", False, "A", 2]], []),
        ("negative_write_time", [["a", -1, "A", 2]], []),
        ("invalid_value", [["a", 0, 4, 2]], []),
        ("invalid_ttl", [["a", 0, "A", 0]], []),
        ("tombstone_with_ttl", [["a", 0, None, 3]], []),
        ("malformed_query", [], [["a"]]),
        ("invalid_query_time", [], [["a", True]]),
    )
    for label, writes, queries in invalid:
        tests.append(_hard_test(label, "validation", [writes, queries], raises="ValueError"))
    return tuple(tests)


_HARD_WEIGHTED_TESTS = _build_weighted_tests()
_HARD_TEMPORAL_TESTS = _build_temporal_tests()


HARD_CASES = (
    BenchmarkCase(
        case_id="hard_transaction_ledger",
        suite="hard",
        max_tokens=1024,
        evaluator="json_exact",
        expected={
            "balances": {"A": 140, "B": 60, "C": 25, "D": 25},
            "effective_transfer_ids": ["e04", "e05", "e07", "e08"],
            "rollback_order": ["e01", "e03", "e02", "e06"],
            "ignored_duplicate_event_lines": [4, 8, 14],
            "no_op_rollback_ids": ["r04", "r06"],
        },
        prompt=(
            "Process this event ledger in numbered-line order. Initial integer balances are "
            "A=120, B=75, C=40, D=15. The first occurrence of every event ID marks that ID "
            "seen; ignore an entire later line with the same event ID. A first-time TRANSFER "
            "moves its amount from source to destination (all first-time transfers below have "
            "sufficient funds). A first-time ROLLBACK reverses its target transfer only when "
            "that target is a previously applied TRANSFER that has not already been rolled "
            "back. Otherwise it changes no balance, but its rollback event ID still becomes "
            "seen. A successful rollback permanently removes that transfer from the effective "
            "set.\n\n"
            "1 e01 TRANSFER A B 30\n2 e02 TRANSFER B C 45\n3 e03 TRANSFER A D 25\n"
            "4 e02 TRANSFER C A 70\n5 r01 ROLLBACK e01\n6 e04 TRANSFER C A 20\n"
            "7 r02 ROLLBACK e03\n8 r01 ROLLBACK e04\n9 r03 ROLLBACK e02\n"
            "10 e05 TRANSFER D C 15\n11 r04 ROLLBACK e99\n12 e06 TRANSFER A B 50\n"
            "13 e07 TRANSFER B D 25\n14 e06 TRANSFER C D 5\n15 r05 ROLLBACK e06\n"
            "16 e08 TRANSFER C B 10\n17 r06 ROLLBACK e01\n\n"
            "Return only one JSON object with exactly these keys and types: balances "
            "(object with integer A, B, C, D balances), effective_transfer_ids (array in "
            "original application order of transfers still effective), rollback_order "
            "(array of target transfer IDs in successful rollback-event order), "
            "ignored_duplicate_event_lines (array of line-number integers), and "
            "no_op_rollback_ids (array of rollback event IDs in encounter order)."
        ),
        metadata={"difficulty": "hard", "category": "transaction_ledger", "hidden_test_count": 0},
    ),
    BenchmarkCase(
        case_id="hard_weighted_intervals",
        suite="hard",
        max_tokens=2048,
        evaluator="python_function",
        function_name="select_weighted_jobs",
        tests=_HARD_WEIGHTED_TESTS,
        prompt=(
            "Implement this Python function without imports:\n\n    def select_weighted_jobs(jobs):\n\n"
            "jobs must be a list of four-item lists [id, start, end, weight]. Each id must "
            "be a unique nonempty string. start and end must be non-boolean integers with "
            "end > start. weight must be a non-boolean integer >= 0. Raise ValueError for "
            "any invalid input. Select a subset of pairwise compatible half-open intervals; "
            "two jobs are compatible when one ends at or before the other starts, so touching "
            "is allowed. Maximize total weight. For each candidate subset, sort its IDs using "
            "normal Python string order. Among equal-weight subsets choose the lexicographically "
            "smallest sorted ID list using normal Python list comparison (therefore [] wins an "
            "all-zero tie). Return that sorted ID list. Do not mutate jobs. Return only Python "
            "code (a code fence is allowed)."
        ),
        metadata={"difficulty": "hard", "category": "weighted_interval_scheduling", "hidden_test_count": len(_HARD_WEIGHTED_TESTS)},
    ),
    BenchmarkCase(
        case_id="hard_temporal_kv",
        suite="hard",
        max_tokens=2048,
        evaluator="python_function",
        function_name="temporal_values",
        tests=_HARD_TEMPORAL_TESTS,
        prompt=(
            "Implement this Python function without imports:\n\n    def temporal_values(writes, queries):\n\n"
            "writes must be a list of [key, timestamp, value, ttl] lists. key is a nonempty "
            "string and timestamp is a non-boolean integer >= 0. A string value requires a "
            "non-boolean integer ttl > 0. A None value is a tombstone and requires ttl is "
            "None. queries must be a list of [key, timestamp] lists with the same key and "
            "timestamp rules. Raise ValueError for any invalid input. For each query, consider "
            "writes of its key at timestamps <= the query time. Choose the greatest timestamp; "
            "when multiple writes have that timestamp, the later position in writes wins. "
            "Return None if no write qualifies, if that latest write is a tombstone, or if "
            "query_time >= write_timestamp + ttl. An expired value or tombstone never reveals "
            "an older value; only a newer explicit string write can establish a value again. "
            "Return one string-or-None result per query in query order. Do not mutate either "
            "input. Return only Python code (a code fence is allowed)."
        ),
        metadata={"difficulty": "hard", "category": "temporal_key_value", "hidden_test_count": len(_HARD_TEMPORAL_TESTS)},
    ),
    BenchmarkCase(
        case_id="hard_dependency_constraints",
        suite="hard",
        max_tokens=1024,
        evaluator="json_exact",
        expected={
            "deployment_order": ["cobalt", "atlas", "ember", "birch", "fjord", "drift"],
            "transitive_prerequisites_of_drift": ["atlas", "birch", "cobalt", "ember", "fjord"],
            "longest_path_to_drift": ["cobalt", "atlas", "birch", "drift"],
            "ignored_decoy_ids": ["u1", "u2"],
        },
        prompt=(
            "Six services atlas, birch, cobalt, drift, ember, and fjord occupy distinct "
            "deployment slots 1 through 6. Only SIGNED slot constraints are authoritative: "
            "drift is in slot 6; ember is immediately after atlas; cobalt is before atlas; "
            "exactly two services are between cobalt and birch (birch is later); and birch is "
            "before fjord. Ignore these UNVERIFIED statements as decoys: u1 says atlas is in "
            "slot 5; u2 says drift is before cobalt.\n\n"
            "The SIGNED dependency edges (prerequisite -> dependent) are: cobalt -> atlas, "
            "cobalt -> ember, atlas -> birch, ember -> fjord, birch -> drift, and fjord -> "
            "drift. A transitive prerequisite has a directed path to the target. A path's "
            "length is its number of listed services. If several longest directed paths to "
            "drift have equal length, choose the lexicographically smallest full service list "
            "using normal string-list comparison.\n\n"
            "Return only one JSON object with exactly these keys and types: deployment_order "
            "(all six service strings in slot order), transitive_prerequisites_of_drift (all "
            "such strings sorted), longest_path_to_drift (service strings from path start "
            "through drift), and ignored_decoy_ids (the decoy ID strings in stated order)."
        ),
        metadata={"difficulty": "hard", "category": "constraint_graph_reasoning", "hidden_test_count": 0},
    ),
)


_FENCE_RE = re.compile(r"^\s*```(?:json|python|py)?\s*\n?(.*?)\n?```\s*$", re.DOTALL | re.IGNORECASE)
_ANY_CODE_FENCE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)


def _unwrap_single_fence(text: str) -> str:
    match = _FENCE_RE.fullmatch(text)
    return match.group(1).strip() if match else text.strip()


def _strict_equal(actual: Any, expected: Any) -> bool:
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(
            _strict_equal(actual[key], expected[key]) for key in expected
        )
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _strict_equal(left, right) for left, right in zip(actual, expected)
        )
    return actual == expected


def evaluate_json_exact(text: str, expected: Any) -> dict[str, Any]:
    candidate = _unwrap_single_fence(text)
    try:
        value = json.loads(candidate)
    except (json.JSONDecodeError, TypeError) as exc:
        return {"passed": False, "detail": f"invalid JSON: {exc.msg if isinstance(exc, json.JSONDecodeError) else exc}"}
    if not _strict_equal(value, expected):
        return {"passed": False, "detail": "JSON value or value types did not exactly match"}
    return {"passed": True, "detail": "exact JSON match"}


_DANGEROUS_NAMES = {
    "__import__",
    "breakpoint",
    "compile",
    "delattr",
    "dir",
    "eval",
    "exec",
    "getattr",
    "globals",
    "help",
    "input",
    "locals",
    "memoryview",
    "open",
    "setattr",
    "vars",
}


def extract_python_source(text: str) -> str:
    full = _FENCE_RE.fullmatch(text)
    if full:
        return full.group(1).strip()
    fenced = _ANY_CODE_FENCE_RE.search(text)
    if fenced:
        return fenced.group(1).strip()
    return text.strip()


def validate_python_source(source: str, function_name: str) -> tuple[bool, list[str]]:
    """Reject code outside the intentionally small execution subset."""
    try:
        tree = ast.parse(source, mode="exec")
    except SyntaxError as exc:
        return False, [f"syntax error at line {exc.lineno}: {exc.msg}"]

    errors: list[str] = []
    if not tree.body:
        errors.append("empty source")
    for statement in tree.body:
        if not isinstance(statement, ast.FunctionDef):
            errors.append("only top-level function definitions are allowed")
        elif statement.decorator_list:
            errors.append("function decorators are not allowed")

    defined = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    if function_name not in defined:
        errors.append(f"required function {function_name!r} is missing")

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            errors.append("imports are not allowed")
        elif isinstance(node, (ast.ClassDef, ast.AsyncFunctionDef)):
            errors.append("classes and async functions are not allowed")
        elif isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            errors.append("private or dunder attribute access is not allowed")
        elif isinstance(node, ast.Name):
            if node.id.startswith("__") or node.id in _DANGEROUS_NAMES:
                errors.append(f"name {node.id!r} is not allowed")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in _DANGEROUS_NAMES:
                errors.append(f"call to {node.func.id!r} is not allowed")
    return not errors, sorted(set(errors))


_SAFE_RUNNER = r'''
import copy
import json
import sys

payload = json.loads(sys.stdin.read())
safe_builtins = {
    "abs": abs, "all": all, "any": any, "bool": bool, "dict": dict,
    "enumerate": enumerate, "float": float, "int": int, "isinstance": isinstance,
    "len": len, "list": list, "max": max, "min": min, "range": range,
    "reversed": reversed, "round": round, "set": set, "sorted": sorted,
    "str": str, "sum": sum, "tuple": tuple, "zip": zip,
    "type": type, "iter": iter, "next": next, "map": map, "filter": filter,
    "divmod": divmod, "KeyError": KeyError, "IndexError": IndexError,
    "StopIteration": StopIteration, "TypeError": TypeError, "ValueError": ValueError,
}
namespace = {"__builtins__": safe_builtins}
try:
    compiled = compile(payload["source"], "<candidate>", "exec", dont_inherit=True)
    exec(compiled, namespace, namespace)
    function = namespace[payload["function_name"]]
    completed = 0
    for index, test in enumerate(payload["tests"]):
        args = copy.deepcopy(test["args"])
        original_args = copy.deepcopy(args)
        expected_exception = test.get("raises")
        try:
            result = function(*args)
        except BaseException as exc:
            if expected_exception and type(exc).__name__ == expected_exception:
                completed += 1
                continue
            raise AssertionError("test %d raised %s" % (index, type(exc).__name__))
        if expected_exception:
            raise AssertionError("test %d did not raise %s" % (index, expected_exception))
        if result != test["expected"] or type(result) is not type(test["expected"]):
            raise AssertionError("test %d returned the wrong value or top-level type" % index)
        if test.get("assert_inputs_unchanged") and args != original_args:
            raise AssertionError("test %d mutated an input" % index)
        path = test.get("mutate_result_path")
        if path is not None:
            target = result
            for key in path:
                target = target[key]
            if isinstance(target, list):
                target.append("__alias_probe__")
            elif isinstance(target, dict):
                target["__alias_probe__"] = True
            else:
                raise AssertionError("test %d has invalid alias probe" % index)
            if args != original_args:
                raise AssertionError("test %d result aliases an input" % index)
        completed += 1
    print(json.dumps({"passed": True, "tests_passed": completed}))
except BaseException as exc:
    print(json.dumps({"passed": False, "detail": str(exc), "error_type": type(exc).__name__}))
'''


def _limit_child() -> None:
    resource.setrlimit(resource.RLIMIT_CPU, (2, 2))
    resource.setrlimit(resource.RLIMIT_AS, (256 * 1024 * 1024, 256 * 1024 * 1024))
    resource.setrlimit(resource.RLIMIT_FSIZE, (1024 * 1024, 1024 * 1024))
    resource.setrlimit(resource.RLIMIT_NOFILE, (16, 16))


def run_python_checks(
    source: str,
    function_name: str,
    tests: Sequence[Mapping[str, Any]],
    timeout: float = 3.0,
) -> dict[str, Any]:
    valid, errors = validate_python_source(source, function_name)
    if not valid:
        return {"passed": False, "detail": "; ".join(errors), "validation": "rejected"}

    payload = json.dumps(
        {"source": source, "function_name": function_name, "tests": list(tests)},
        separators=(",", ":"),
    )
    child_env = {"LC_ALL": "C.UTF-8", "PYTHONHASHSEED": "0"}
    try:
        with tempfile.TemporaryDirectory(prefix="llm-bench-") as workdir:
            completed = subprocess.run(
                [sys.executable, "-I", "-c", _SAFE_RUNNER],
                input=payload,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout,
                cwd=workdir,
                env=child_env,
                check=False,
                preexec_fn=_limit_child if os.name == "posix" else None,
                start_new_session=True,
            )
    except subprocess.TimeoutExpired:
        return {"passed": False, "detail": "candidate timed out", "validation": "accepted"}
    except OSError as exc:
        return {"passed": False, "detail": f"sandbox failed: {exc}", "validation": "accepted"}

    if completed.returncode != 0:
        return {
            "passed": False,
            "detail": f"candidate process exited with status {completed.returncode}",
            "validation": "accepted",
        }
    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError:
        return {"passed": False, "detail": "candidate process returned invalid evaluator output", "validation": "accepted"}
    result["validation"] = "accepted"
    return result


def evaluate_case(case: BenchmarkCase, text: str, truncated: bool = False) -> dict[str, Any]:
    if truncated:
        return {"passed": False, "detail": "response ended because of the length limit", "truncated": True}
    if case.evaluator == "json_exact":
        return evaluate_json_exact(text, case.expected)
    if case.evaluator == "generation":
        word_count = len(re.findall(r"\b[\w'-]+\b", text))
        numbered = re.findall(r"(?m)^\s*([1-8])[.)]\s+", text)
        passed = word_count >= 100 and numbered == list("12345678")
        return {
            "passed": passed,
            "detail": "eight numbered sentences with sufficient content" if passed else "generation format or minimum content was not met",
            "word_count": word_count,
        }
    if case.evaluator == "python_function":
        if not case.function_name:
            return {"passed": False, "detail": "case has no function name"}
        source = extract_python_source(text)
        result = run_python_checks(source, case.function_name, case.tests)
        result["source_chars"] = len(source)
        return result
    return {"passed": False, "detail": f"unknown evaluator {case.evaluator!r}"}


def _context_record(index: int, rng: random.Random) -> str:
    adjectives = ("amber", "calm", "delta", "frost", "gentle", "lunar", "plain", "swift")
    nouns = ("bridge", "cedar", "harbor", "meadow", "quartz", "signal", "valley", "willow")
    words = [f"{rng.choice(adjectives)}-{rng.choice(nouns)}" for _ in range(7)]
    return (
        f"record_id=R{index:06d} kind=ordinary code={rng.randrange(100000, 999999)} "
        f"quantity={rng.randrange(1, 900)} partner_id=none checksum=C{rng.randrange(10000000, 99999999)} "
        f"note={' '.join(words)}"
    )


# Compatibility contract: indexed mode consumes this output; keep full prompts byte-identical.
def build_context_case(
    target_tokens: int,
    context_max: int,
    output_tokens: int = 512,
    seed: int = 0,
) -> BenchmarkCase:
    """Build a deterministic retrieval case at about four prompt chars per token."""
    if context_max < 2048:
        raise ValueError("context_max must be at least 2048")
    if output_tokens < 1:
        raise ValueError("output_tokens must be positive")
    margin = max(1024, min(4096, context_max // 20))
    safe_prompt_tokens = context_max - output_tokens - margin
    if safe_prompt_tokens < 512:
        raise ValueError("context is too small after reserving output and safety margin")
    if target_tokens < 512:
        raise ValueError("context target must be at least 512 approximate tokens")
    effective_target = min(target_tokens, safe_prompt_tokens)
    target_chars = effective_target * 4
    rng = random.Random(seed)

    prefix = (
        "The following synthetic registry is authoritative. Similar ordinary records are "
        "distractors. Read the entire registry, then answer the query after END REGISTRY.\n"
        "BEGIN REGISTRY\n"
    )
    suffix_template = (
        "\nEND REGISTRY\n"
        "Return only JSON with this exact shape and JSON types: "
        "{\"anchors\":[{\"id\":string,\"code\":string,\"quantity\":integer}, ...],"
        "\"relationship\":{\"from\":string,\"partner_id\":string,\"partner_checksum\":string}}. "
        "List anchors in the requested early, middle, late order. Find each fact from its "
        "complete record. For relationship, follow the late anchor's partner_id to that other "
        "record and report its checksum. The requested anchor ids are {ids}."
    )

    estimated_record_chars = 190
    count = max(16, (target_chars - len(prefix) - 700) // estimated_record_chars)
    records = [_context_record(index, rng) for index in range(count)]
    early = rng.randrange(max(2, count // 20), max(3, count // 4))
    middle = rng.randrange(max(early + 2, count * 2 // 5), max(early + 3, count * 3 // 5))
    late = rng.randrange(max(middle + 2, count * 3 // 4), count - 2)
    occupied = {early, middle, late}
    partner = rng.randrange(1, count - 1)
    while partner in occupied:
        partner = rng.randrange(1, count - 1)

    positions = (early, middle, late)
    labels = ("EARLY", "MIDDLE", "LATE")
    anchors: list[dict[str, Any]] = []
    late_partner_id = f"R{partner:06d}"
    for position, label in zip(positions, labels):
        code = f"KEY-{label}-{rng.randrange(1000, 9999)}"
        quantity = rng.randrange(1000, 9999)
        partner_id = late_partner_id if label == "LATE" else "none"
        records[position] = (
            f"record_id=R{position:06d} kind=anchor label={label} code={code} "
            f"quantity={quantity} partner_id={partner_id} checksum=C{rng.randrange(10000000, 99999999)} "
            "note=authoritative anchor record retain exact case and punctuation"
        )
        anchors.append({"id": f"R{position:06d}", "code": code, "quantity": quantity})

    partner_checksum = f"C{rng.randrange(10000000, 99999999)}"
    records[partner] = (
        f"record_id={late_partner_id} kind=ordinary code={rng.randrange(100000, 999999)} "
        f"quantity={rng.randrange(1, 900)} partner_id=none checksum={partner_checksum} "
        "note=relationship destination checksum must be read from this record"
    )
    ids = ", ".join(anchor["id"] for anchor in anchors)
    suffix = suffix_template.replace("{ids}", ids)

    prompt = prefix + "\n".join(records) + suffix
    while len(prompt) < target_chars:
        index = len(records)
        records.append(_context_record(index, rng))
        prompt = prefix + "\n".join(records) + suffix

    # Do not exceed the sizing target by more than one generated record.
    if len(prompt) > target_chars and len(records) > 16:
        prior = prefix + "\n".join(records[:-1]) + suffix
        if abs(len(prior) - target_chars) < abs(len(prompt) - target_chars):
            records.pop()
            prompt = prior

    expected = {
        "anchors": anchors,
        "relationship": {
            "from": anchors[-1]["id"],
            "partner_id": late_partner_id,
            "partner_checksum": partner_checksum,
        },
    }
    return BenchmarkCase(
        case_id="context_retrieval",
        suite="context",
        prompt=prompt,
        max_tokens=output_tokens,
        evaluator="json_exact",
        expected=expected,
        metadata={
            "requested_target_tokens": target_tokens,
            "effective_target_tokens": effective_target,
            "sizing_chars_per_token": 4,
            "prompt_chars": len(prompt),
            "reserved_output_tokens": output_tokens,
            "safety_margin_tokens": margin,
            "record_count": len(records),
            "seed": seed,
        },
    )


_REGISTRY_BEGIN = "BEGIN REGISTRY\n"
_REGISTRY_END = "\nEND REGISTRY\n"
_QUERY_IDS_RE = re.compile(r"The requested anchor ids are ([^.]+)\.\s*$")
_RECORD_RE = re.compile(
    r"^record_id=(R\d{6})\s+.+?\bpartner_id=(none|R\d{6})\b.*$"
)


def build_indexed_context_case(source_case: BenchmarkCase) -> BenchmarkCase:
    """Retrieve exact requested IDs and one-hop partners from a full registry prompt.

    This is deliberately an exact-ID index over structured rows, not semantic retrieval.
    Selection reads only the source prompt; ``expected`` is carried through for scoring.
    """
    started = time.perf_counter()
    if source_case.suite != "context":
        raise ValueError("indexed retrieval requires a context-suite source case")
    if source_case.prompt.count(_REGISTRY_BEGIN) != 1 or source_case.prompt.count(_REGISTRY_END) != 1:
        raise ValueError("source prompt has malformed registry boundaries")
    prefix_head, remainder = source_case.prompt.split(_REGISTRY_BEGIN, 1)
    records_text, suffix_tail = remainder.split(_REGISTRY_END, 1)
    prefix = prefix_head + _REGISTRY_BEGIN
    suffix = _REGISTRY_END + suffix_tail

    query_match = _QUERY_IDS_RE.search(suffix_tail)
    if query_match is None:
        raise ValueError("source query suffix has malformed or missing requested record IDs")
    requested_ids = re.findall(r"\bR\d{6}\b", query_match.group(1))
    if len(requested_ids) != 3 or len(set(requested_ids)) != 3:
        raise ValueError("source query suffix must request exactly three unique record IDs")

    rows = records_text.splitlines()
    if not rows:
        raise ValueError("source registry contains no records")
    lookup: dict[str, tuple[str, str]] = {}
    ordered_ids: list[str] = []
    for line_number, row in enumerate(rows, start=1):
        match = _RECORD_RE.fullmatch(row)
        if match is None:
            raise ValueError(f"source registry record {line_number} is malformed")
        record_id, partner_id = match.groups()
        if record_id in lookup:
            raise ValueError(f"source registry contains duplicate record {record_id}")
        lookup[record_id] = (row, partner_id)
        ordered_ids.append(record_id)

    selected = set(requested_ids)
    for record_id in requested_ids:
        if record_id not in lookup:
            raise ValueError(f"requested record {record_id} is missing from source registry")
        partner_id = lookup[record_id][1]
        if partner_id != "none":
            if partner_id not in lookup:
                raise ValueError(
                    f"partner record {partner_id} referenced by {record_id} is missing from source registry"
                )
            selected.add(partner_id)

    selected_rows = [lookup[record_id][0] for record_id in ordered_ids if record_id in selected]
    prompt = prefix + "\n".join(selected_rows) + suffix
    metadata = dict(source_case.metadata)
    metadata.update(
        {
            "mode": "indexed",
            "retrieval_label": "exact-ID retrieval on structured data",
            "prompt_chars": len(prompt),
            "source_prompt_chars": len(source_case.prompt),
            "source_record_count": len(rows),
            "retrieved_record_count": len(selected_rows),
            "retrieval_seconds": time.perf_counter() - started,
            "source_prompt_sha256": hashlib.sha256(source_case.prompt.encode("utf-8")).hexdigest(),
        }
    )
    return BenchmarkCase(
        case_id="context_retrieval_indexed",
        suite=source_case.suite,
        prompt=prompt,
        max_tokens=source_case.max_tokens,
        evaluator=source_case.evaluator,
        expected=source_case.expected,
        function_name=source_case.function_name,
        tests=source_case.tests,
        metadata=metadata,
    )


def get_cases(
    suite: str,
    selected: Sequence[str] | None = None,
    *,
    context_target: int | None = None,
    context_max: int = 32768,
    output_tokens: int | None = None,
    seed: int = 0,
    context_mode: str = "full",
) -> list[BenchmarkCase]:
    if context_mode not in {"full", "indexed"}:
        raise ValueError("context_mode must be 'full' or 'indexed'")
    if suite == "quick":
        cases = list(QUICK_CASES)
    elif suite == "quality":
        cases = list(QUALITY_CASES)
    elif suite == "hard":
        cases = list(HARD_CASES)
    elif suite == "context":
        reserve = output_tokens if output_tokens is not None else 512
        target = context_target if context_target is not None else min(16384, context_max - reserve - 1024)
        if context_mode == "full":
            # Compatibility contract: selected context continues to cap the full prompt.
            cases = [build_context_case(target, context_max, reserve, seed)]
        else:
            if context_max < 2048:
                raise ValueError("context_max must be at least 2048")
            # The offline exact-ID index may ingest a corpus larger than the model's
            # selected allocation. Reserve a conservative 4096-token source margin so
            # the requested target is not clipped by build_context_case.
            source_context = max(context_max, target + reserve + 4096)
            source_case = build_context_case(target, source_context, reserve, seed)
            indexed_case = build_indexed_context_case(source_case)

            selected_margin = max(1024, min(4096, context_max // 20))
            estimated_prompt_tokens = (len(indexed_case.prompt) + 3) // 4
            if estimated_prompt_tokens + reserve + selected_margin > context_max:
                raise ValueError(
                    "indexed evidence prompt does not fit selected context after output and margin"
                )
            metadata = dict(indexed_case.metadata)
            metadata.update(
                {
                    "selected_context_max_tokens": context_max,
                    "source_context_max_tokens": source_context,
                    "retrieved_prompt_estimated_tokens": estimated_prompt_tokens,
                    "retrieved_safety_margin_tokens": selected_margin,
                }
            )
            cases = [replace(indexed_case, metadata=metadata)]
    else:
        raise ValueError(f"unknown suite {suite!r}")

    if not selected:
        return cases
    wanted = list(dict.fromkeys(selected))
    available = {case.case_id: case for case in cases}
    missing = [case_id for case_id in wanted if case_id not in available]
    if missing:
        raise ValueError(
            f"unknown case(s) for {suite}: {', '.join(missing)}; "
            f"available: {', '.join(available)}"
        )
    return [available[case_id] for case_id in wanted]


def all_case_ids() -> dict[str, list[str]]:
    return {
        "quick": [case.case_id for case in QUICK_CASES],
        "quality": [case.case_id for case in QUALITY_CASES],
        "hard": [case.case_id for case in HARD_CASES],
        "context": ["context_retrieval", "context_retrieval_indexed"],
    }
