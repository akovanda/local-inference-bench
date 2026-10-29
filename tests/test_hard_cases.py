import itertools
import json
import unittest

from bench_cases import CASE_VERSION, HARD_CASES, all_case_ids, evaluate_case, get_cases, run_python_checks


WEIGHTED_REFERENCE_SOURCE = r'''
def select_weighted_jobs(jobs):
    if not isinstance(jobs, list):
        raise ValueError()
    seen = set()
    for job in jobs:
        if not isinstance(job, list) or len(job) != 4:
            raise ValueError()
        name, start, end, weight = job
        if not isinstance(name, str) or not name or name in seen:
            raise ValueError()
        if (not isinstance(start, int) or isinstance(start, bool) or
                not isinstance(end, int) or isinstance(end, bool) or end <= start):
            raise ValueError()
        if not isinstance(weight, int) or isinstance(weight, bool) or weight < 0:
            raise ValueError()
        seen.add(name)
    best_weight = 0
    best_ids = []
    for mask in range(1 << len(jobs)):
        chosen = []
        for index in range(len(jobs)):
            if mask & (1 << index):
                chosen.append(jobs[index])
        okay = True
        for left_index in range(len(chosen)):
            for right_index in range(left_index + 1, len(chosen)):
                left = chosen[left_index]
                right = chosen[right_index]
                if not (left[2] <= right[1] or right[2] <= left[1]):
                    okay = False
        if okay:
            total = sum(job[3] for job in chosen)
            ids = sorted(job[0] for job in chosen)
            if total > best_weight or (total == best_weight and ids < best_ids):
                best_weight = total
                best_ids = ids
    return best_ids
'''


TEMPORAL_REFERENCE_SOURCE = r'''
def temporal_values(writes, queries):
    if not isinstance(writes, list) or not isinstance(queries, list):
        raise ValueError()
    for write in writes:
        if not isinstance(write, list) or len(write) != 4:
            raise ValueError()
        key, timestamp, value, ttl = write
        if not isinstance(key, str) or not key:
            raise ValueError()
        if not isinstance(timestamp, int) or isinstance(timestamp, bool) or timestamp < 0:
            raise ValueError()
        if value is None:
            if ttl is not None:
                raise ValueError()
        elif isinstance(value, str):
            if not isinstance(ttl, int) or isinstance(ttl, bool) or ttl <= 0:
                raise ValueError()
        else:
            raise ValueError()
    for query in queries:
        if not isinstance(query, list) or len(query) != 2:
            raise ValueError()
        key, timestamp = query
        if not isinstance(key, str) or not key:
            raise ValueError()
        if not isinstance(timestamp, int) or isinstance(timestamp, bool) or timestamp < 0:
            raise ValueError()
    answers = []
    for key, query_time in queries:
        latest = None
        for position, write in enumerate(writes):
            write_key, timestamp, value, ttl = write
            if write_key == key and timestamp <= query_time:
                if latest is None or timestamp > latest[0] or (timestamp == latest[0] and position > latest[1]):
                    latest = [timestamp, position, value, ttl]
        if latest is None or latest[2] is None or query_time >= latest[0] + latest[3]:
            answers.append(None)
        else:
            answers.append(latest[2])
    return answers
'''


def independent_weighted_reference(jobs):
    best = None
    for size in range(len(jobs) + 1):
        for chosen in itertools.combinations(jobs, size):
            if any(
                not (left[2] <= right[1] or right[2] <= left[1])
                for left, right in itertools.combinations(chosen, 2)
            ):
                continue
            candidate = (sum(job[3] for job in chosen), sorted(job[0] for job in chosen))
            if best is None or candidate[0] > best[0] or (
                candidate[0] == best[0] and candidate[1] < best[1]
            ):
                best = candidate
    return best[1]


def independent_temporal_reference(writes, queries):
    result = []
    for key, query_time in queries:
        eligible = [
            (timestamp, position, value, ttl)
            for position, (write_key, timestamp, value, ttl) in enumerate(writes)
            if write_key == key and timestamp <= query_time
        ]
        if not eligible:
            result.append(None)
            continue
        timestamp, _, value, ttl = max(eligible, key=lambda item: (item[0], item[1]))
        result.append(
            value if value is not None and query_time < timestamp + ttl else None
        )
    return result


class HardCaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = {case.case_id: case for case in HARD_CASES}

    def test_suite_ids_metadata_and_labeled_checks(self):
        expected_ids = [
            "hard_transaction_ledger",
            "hard_weighted_intervals",
            "hard_temporal_kv",
            "hard_dependency_constraints",
        ]
        self.assertEqual(CASE_VERSION, "2026-10-19.2")
        self.assertEqual([case.case_id for case in get_cases("hard")], expected_ids)
        self.assertEqual(all_case_ids()["hard"], expected_ids)
        for case in HARD_CASES:
            self.assertEqual(case.metadata["difficulty"], "hard")
            self.assertIn("category", case.metadata)
            self.assertEqual(case.metadata["hidden_test_count"], len(case.tests))
            for check in case.tests:
                self.assertIsInstance(check.get("label"), str)
                self.assertTrue(check["label"])
                self.assertIsInstance(check.get("category"), str)
                self.assertTrue(check["category"])
        self.assertGreaterEqual(len(self.cases["hard_weighted_intervals"].tests), 20)
        self.assertGreaterEqual(len(self.cases["hard_temporal_kv"].tests), 20)
        weighted_by_label = {
            check["label"]: check for check in self.cases["hard_weighted_intervals"].tests
        }
        self.assertEqual(
            weighted_by_label["zero_weight_lexical_prefix"]["expected"],
            ["a", "z"],
        )

    def test_transaction_reference_and_rollback_regression(self):
        balances = {"A": 120, "B": 75, "C": 40, "D": 15}
        events = [
            ("e01", "transfer", "A", "B", 30),
            ("e02", "transfer", "B", "C", 45),
            ("e03", "transfer", "A", "D", 25),
            ("e02", "transfer", "C", "A", 70),
            ("r01", "rollback", "e01"),
            ("e04", "transfer", "C", "A", 20),
            ("r02", "rollback", "e03"),
            ("r01", "rollback", "e04"),
            ("r03", "rollback", "e02"),
            ("e05", "transfer", "D", "C", 15),
            ("r04", "rollback", "e99"),
            ("e06", "transfer", "A", "B", 50),
            ("e07", "transfer", "B", "D", 25),
            ("e06", "transfer", "C", "D", 5),
            ("r05", "rollback", "e06"),
            ("e08", "transfer", "C", "B", 10),
            ("r06", "rollback", "e01"),
        ]
        seen = set()
        applied = {}
        application_order = []
        rollback_order = []
        duplicate_lines = []
        no_ops = []
        for line, event in enumerate(events, 1):
            event_id, operation, *fields = event
            if event_id in seen:
                duplicate_lines.append(line)
                continue
            seen.add(event_id)
            if operation == "transfer":
                source, destination, amount = fields
                balances[source] -= amount
                balances[destination] += amount
                applied[event_id] = [source, destination, amount, True]
                application_order.append(event_id)
            else:
                target = fields[0]
                transfer = applied.get(target)
                if transfer is None or not transfer[3]:
                    no_ops.append(event_id)
                    continue
                source, destination, amount, _ = transfer
                balances[source] += amount
                balances[destination] -= amount
                transfer[3] = False
                rollback_order.append(target)
        actual = {
            "balances": balances,
            "effective_transfer_ids": [event_id for event_id in application_order if applied[event_id][3]],
            "rollback_order": rollback_order,
            "ignored_duplicate_event_lines": duplicate_lines,
            "no_op_rollback_ids": no_ops,
        }
        case = self.cases["hard_transaction_ledger"]
        self.assertEqual(actual, case.expected)
        self.assertTrue(evaluate_case(case, json.dumps(actual))["passed"])
        broken = dict(actual)
        broken["balances"] = dict(actual["balances"], A=90)
        self.assertFalse(evaluate_case(case, json.dumps(broken))["passed"])

    def test_weighted_generated_answers_have_independent_oracle(self):
        case = self.cases["hard_weighted_intervals"]
        for check in case.tests:
            if "expected" in check:
                self.assertEqual(
                    independent_weighted_reference(check["args"][0]),
                    check["expected"],
                    check["label"],
                )
        result = run_python_checks(WEIGHTED_REFERENCE_SOURCE, case.function_name, case.tests)
        self.assertTrue(result["passed"], result)
        self.assertEqual(result["tests_passed"], len(case.tests))

    def test_weighted_boundary_and_tie_bugs_are_caught(self):
        case = self.cases["hard_weighted_intervals"]
        touching_bug = WEIGHTED_REFERENCE_SOURCE.replace(
            "left[2] <= right[1] or right[2] <= left[1]",
            "left[2] < right[1] or right[2] < left[1]",
        )
        tie_bug = WEIGHTED_REFERENCE_SOURCE.replace("ids < best_ids", "ids > best_ids")
        self.assertFalse(run_python_checks(touching_bug, case.function_name, case.tests)["passed"])
        self.assertFalse(run_python_checks(tie_bug, case.function_name, case.tests)["passed"])

    def test_temporal_generated_answers_have_independent_oracle(self):
        case = self.cases["hard_temporal_kv"]
        for check in case.tests:
            if "expected" in check:
                self.assertEqual(
                    independent_temporal_reference(*check["args"]),
                    check["expected"],
                    check["label"],
                )
        result = run_python_checks(TEMPORAL_REFERENCE_SOURCE, case.function_name, case.tests)
        self.assertTrue(result["passed"], result)
        self.assertEqual(result["tests_passed"], len(case.tests))

    def test_temporal_ttl_fallback_bug_is_caught(self):
        case = self.cases["hard_temporal_kv"]
        fallback_bug = TEMPORAL_REFERENCE_SOURCE.replace(
            "if write_key == key and timestamp <= query_time:",
            "if write_key == key and timestamp <= query_time and (value is None or query_time < timestamp + ttl):",
        )
        result = run_python_checks(fallback_bug, case.function_name, case.tests)
        self.assertFalse(result["passed"], result)

    def test_constraint_solution_is_unique_and_graph_answer_is_derived(self):
        names = ("atlas", "birch", "cobalt", "drift", "ember", "fjord")
        orders = []
        for order in itertools.permutations(names):
            position = {name: index + 1 for index, name in enumerate(order)}
            if (
                position["drift"] == 6
                and position["ember"] == position["atlas"] + 1
                and position["cobalt"] < position["atlas"]
                and position["birch"] == position["cobalt"] + 3
                and position["birch"] < position["fjord"]
            ):
                orders.append(list(order))
        self.assertEqual(orders, [["cobalt", "atlas", "ember", "birch", "fjord", "drift"]])

        edges = [
            ("cobalt", "atlas"),
            ("cobalt", "ember"),
            ("atlas", "birch"),
            ("ember", "fjord"),
            ("birch", "drift"),
            ("fjord", "drift"),
        ]
        paths = []

        def visit(path):
            followers = [right for left, right in edges if left == path[-1]]
            if path[-1] == "drift":
                paths.append(path)
            for follower in followers:
                visit(path + [follower])

        incoming = {right for _, right in edges}
        for root in sorted(set(names) - incoming):
            visit([root])
        drift_paths = [path for path in paths if path[-1] == "drift"]
        longest_length = max(map(len, drift_paths))
        longest = min(path for path in drift_paths if len(path) == longest_length)
        prerequisites = sorted({node for path in drift_paths for node in path[:-1]})
        actual = {
            "deployment_order": orders[0],
            "transitive_prerequisites_of_drift": prerequisites,
            "longest_path_to_drift": longest,
            "ignored_decoy_ids": ["u1", "u2"],
        }
        case = self.cases["hard_dependency_constraints"]
        self.assertEqual(actual, case.expected)
        self.assertTrue(evaluate_case(case, json.dumps(actual))["passed"])
        broken = dict(actual, deployment_order=["cobalt", "birch", "ember", "fjord", "atlas", "drift"])
        self.assertFalse(evaluate_case(case, json.dumps(broken))["passed"])


if __name__ == "__main__":
    unittest.main()
