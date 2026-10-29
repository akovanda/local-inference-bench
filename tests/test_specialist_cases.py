import unittest

from runtime import specialist_cases


FRESH_REFERENCE = '''def parse_size(text):
    if type(text) is not str:
        raise ValueError("size must be str")
    if text.endswith("MiB"):
        digits = text[:-3]
        factor = 1048576
    elif text.endswith("KiB"):
        digits = text[:-3]
        factor = 1024
    elif text.endswith("B"):
        digits = text[:-1]
        factor = 1
    else:
        raise ValueError("invalid unit")
    if not digits or any(char not in "0123456789" for char in digits):
        raise ValueError("invalid digits")
    if len(digits) > 1 and digits[0] == "0":
        raise ValueError("leading zero")
    return int(digits) * factor


def pack_batches(sizes, limit):
    if type(sizes) is not list or type(limit) is not int:
        raise TypeError("wrong type")
    if limit <= 0:
        raise ValueError("limit must be positive")
    result = []
    current = []
    total = 0
    for size in sizes:
        if type(size) is not int:
            raise TypeError("size must be int")
        if size <= 0 or size > limit:
            raise ValueError("invalid size")
        if current and total + size > limit:
            result.append(current)
            current = []
            total = 0
        current.append(size)
        total += size
    if current:
        result.append(current)
    return result
'''


BUGFIX_REFERENCE = '''def _require_nonnegative_int(value, label):
    if type(value) is not int:
        raise TypeError(label + " must be int")
    if value < 0:
        raise ValueError(label + " must be nonnegative")
    return value


def apply_delta(balance, debit):
    balance = _require_nonnegative_int(balance, "balance")
    debit = _require_nonnegative_int(debit, "debit")
    if debit > balance:
        raise ValueError("insufficient balance")
    return balance - debit


def transfer(balances, source, destination, amount):
    if type(balances) is not dict:
        raise TypeError("balances must be dict")
    for account, balance in balances.items():
        if type(account) is not str or type(balance) is not int:
            raise TypeError("invalid account")
        if not account or balance < 0:
            raise ValueError("invalid account")
    if type(source) is not str or type(destination) is not str:
        raise TypeError("accounts must be str")
    if type(amount) is not int:
        raise TypeError("amount must be int")
    if amount <= 0:
        raise ValueError("amount must be positive")
    if source == destination:
        raise ValueError("accounts must differ")
    if source not in balances or destination not in balances:
        raise ValueError("unknown account")
    updated = dict(balances)
    updated[source] = apply_delta(updated[source], amount)
    updated[destination] += amount
    return updated


def settle_batch(balances, transfers):
    if type(balances) is not dict or type(transfers) is not list:
        raise TypeError("wrong container type")
    current = dict(balances)
    for account, balance in current.items():
        if type(account) is not str or type(balance) is not int:
            raise TypeError("invalid account")
        if not account or balance < 0:
            raise ValueError("invalid account")
    for item in transfers:
        if type(item) is not dict or set(item) != {"source", "destination", "amount"}:
            raise TypeError("invalid transfer")
        current = transfer(current, item["source"], item["destination"], item["amount"])
    return current
'''


PIVOT_REFERENCE = '''def pivot_readings(rows, sensor_order):
    if type(rows) is not list or type(sensor_order) is not list:
        raise TypeError("containers must be lists")
    sensors = set()
    for sensor in sensor_order:
        if type(sensor) is not str:
            raise TypeError("sensor must be str")
        if not sensor or sensor in sensors:
            raise ValueError("invalid sensor order")
        sensors.add(sensor)
    readings = {}
    timestamps = set()
    for row in rows:
        if type(row) is not dict or set(row) != {"timestamp", "sensor", "value"}:
            raise TypeError("invalid row")
        timestamp = row["timestamp"]
        sensor = row["sensor"]
        value = row["value"]
        if type(timestamp) is not str or type(sensor) is not str:
            raise TypeError("labels must be str")
        if not timestamp or not sensor:
            raise ValueError("labels must be nonempty")
        if type(value) is not int and type(value) is not float:
            raise TypeError("value must be numeric")
        if sensor not in sensors:
            raise ValueError("unknown sensor")
        key = (timestamp, sensor)
        if key in readings:
            raise ValueError("duplicate reading")
        readings[key] = value
        timestamps.add(timestamp)
    ordered_timestamps = sorted(timestamps)
    values = []
    for timestamp in ordered_timestamps:
        values.append([readings[(timestamp, sensor)] if (timestamp, sensor) in readings else None for sensor in sensor_order])
    return {"timestamps": ordered_timestamps, "sensors": list(sensor_order), "values": values}
'''


class FrozenSuiteTests(unittest.TestCase):
    def test_manifest_hashes_and_counts_are_pinned(self):
        expected = {
            "code_fresh_batch_module": "a8f2cc8cd45f63bff5fd2cd8ca33864004aca72a15088a9d320f239cc4ca5011",
            "code_multifunction_ledger_bugfix": "cb0f6864473c7b8a6ead13616ff7dc8ed73bfe76d7b0ece1a93b140a16ffac4e",
            "code_edge_reading_pivot": "f437e82e516c0495981e87c9b591bc4e753b0d6833297fe66259d65b0a6e0d40",
            "prose_grounded_public_briefing": "d17c04c79118f4d0a703416593f8542e2b6e07a7bb4797f2185fa8fdc43bb4db",
            "prose_constrained_last_tram": "87b138d1daaee485aac170c96326818e0a3482d064883eb51d2fb79d8114ab81",
            "prose_structured_orion_edit": "d700051411b4779a90f4d09e154fc7724277c84ba2f825f1f7faaecead10d1f3",
        }
        self.assertEqual(specialist_cases.CASE_VERSION, "2026-10-19.specialist.2")
        self.assertEqual(
            specialist_cases.FROZEN_SUITE_SHA256,
            "683416fc8d3559adff123b78a0f2ad38cefb43422d440c557a1ccabcbdff9b26",
        )
        specialist_cases.assert_frozen_suite()
        self.assertEqual(
            {case.case_id: specialist_cases.case_sha256(case) for case in specialist_cases.CASES},
            expected,
        )
        self.assertEqual(len(specialist_cases.get_cases(["code"])), 3)
        self.assertEqual(len(specialist_cases.get_cases(["prose"])), 3)
        self.assertTrue(all(case.max_tokens == 4096 for case in specialist_cases.get_cases(["code"])))
        self.assertTrue(all(case.max_tokens == 1536 for case in specialist_cases.get_cases(["prose"])))
        self.assertEqual(
            sum(1 for _case in specialist_cases.get_cases(["code"])), 3
        )
        self.assertEqual(
            sum(len(case.code_objectives) for case in specialist_cases.get_cases(["code"])), 8
        )

    def test_group_and_case_selection_is_ordered_and_strict(self):
        selected = specialist_cases.get_cases(
            ["code"], ["code_edge_reading_pivot", "code_fresh_batch_module"]
        )
        self.assertEqual(
            [case.case_id for case in selected],
            ["code_edge_reading_pivot", "code_fresh_batch_module"],
        )
        with self.assertRaisesRegex(ValueError, "unknown or excluded"):
            specialist_cases.get_cases(["code"], ["prose_constrained_last_tram"])


class CodeEvaluationTests(unittest.TestCase):
    def test_reference_modules_pass_every_independent_objective_and_assertion(self):
        references = {
            "code_fresh_batch_module": FRESH_REFERENCE,
            "code_multifunction_ledger_bugfix": BUGFIX_REFERENCE,
            "code_edge_reading_pivot": PIVOT_REFERENCE,
        }
        for case in specialist_cases.get_cases(["code"]):
            with self.subTest(case=case.case_id):
                result = specialist_cases.evaluate_code(case, references[case.case_id])
                self.assertEqual(result["outcome"], "passed", result)
                self.assertEqual(
                    result["independent_tasks_passed"], result["independent_task_count"]
                )
                self.assertEqual(result["independent_task_count"], 1)
                self.assertEqual(result["objectives_passed"], result["objective_count"])
                self.assertEqual(result["assertions_passed"], result["assertion_count"])
                self.assertEqual(
                    result["assertion_count"],
                    sum(item["assertion_count"] for item in result["objective_outcomes"]),
                )

    def test_restricted_evaluator_rejection_is_distinct_from_wrong_answer(self):
        case = specialist_cases.get_cases(["code"], ["code_fresh_batch_module"])[0]
        rejected = specialist_cases.evaluate_code(
            case,
            "def parse_size(text):\n    return open(text)\n\ndef pack_batches(sizes, limit):\n    return []\n",
        )
        wrong = specialist_cases.evaluate_code(
            case,
            "def parse_size(text):\n    return 0\n\ndef pack_batches(sizes, limit):\n    return []\n",
        )
        self.assertEqual(rejected["outcome"], "evaluator_rejected")
        self.assertEqual(rejected["validation"], "rejected")
        self.assertEqual(wrong["outcome"], "wrong_answer")
        self.assertEqual(wrong["validation"], "accepted")
        self.assertGreater(wrong["assertion_count"], wrong["assertions_passed"])

    def test_prompt_invalid_structure_is_wrong_answer_not_evaluator_rejection(self):
        case = specialist_cases.get_cases(["code"], ["code_fresh_batch_module"])[0]
        missing_function = specialist_cases.evaluate_code(
            case, "def parse_size(text):\n    return 1\n"
        )
        imported = specialist_cases.evaluate_code(
            case,
            "import math\n\ndef parse_size(text):\n    return 1\n\ndef pack_batches(sizes, limit):\n    return []\n",
        )
        self.assertEqual(missing_function["outcome"], "wrong_answer")
        self.assertEqual(imported["outcome"], "wrong_answer")

    def test_function_definition_order_is_not_a_source_constraint(self):
        parse_source, pack_source = FRESH_REFERENCE.split("\n\ndef pack_batches", 1)
        reordered = "def pack_batches" + pack_source + "\n\n" + parse_source + "\n"
        case = specialist_cases.get_cases(["code"], ["code_fresh_batch_module"])[0]
        result = specialist_cases.evaluate_code(case, reordered)
        self.assertEqual(result["outcome"], "passed", result)
        self.assertTrue(result["function_name_constraint_passed"])
        self.assertTrue(result["function_signature_constraint_passed"])

    def test_expected_exception_mutation_probe_detects_mutate_then_raise(self):
        prefix = BUGFIX_REFERENCE.split("\ndef settle_batch", 1)[0]
        mutation_source = prefix + '''

def settle_batch(balances, transfers):
    balances["mutated"] = 1
    raise ValueError("later failure")
'''
        case = specialist_cases.get_cases(
            ["code"], ["code_multifunction_ledger_bugfix"]
        )[0]
        result = specialist_cases.evaluate_code(case, mutation_source)
        settle = next(
            item for item in result["objective_outcomes"] if item["objective_id"] == "settle_batch"
        )
        late_failure = next(
            item
            for item in settle["assertion_outcomes"]
            if item["assertion_id"] == "later_failure_atomic"
        )
        self.assertEqual(result["outcome"], "wrong_answer")
        self.assertFalse(late_failure["passed"])
        self.assertIn("mutated an input", late_failure["detail"])


class ProseEvaluationTests(unittest.TestCase):
    def test_mechanical_failures_leave_subjective_quality_explicitly_unscored(self):
        case = specialist_cases.get_cases(
            ["prose"], ["prose_grounded_public_briefing"]
        )[0]
        result = specialist_cases.evaluate_prose(
            case, "Briefing:\nHarbor Link served 99 magical islands."
        )
        self.assertEqual(result["outcome"], "constraint_failed")
        self.assertTrue(result["deterministic_constraints_only"])
        self.assertIn("not a prose-quality", result["constraint_interpretation"])
        self.assertIsNone(result["subjective_quality_score"])
        self.assertEqual(
            result["subjective_quality_status"], "unscored_pending_blinded_review"
        )
        self.assertEqual(result["blinded_rubric_max_points"], 10)
        self.assertEqual(len(result["blinded_rubric"]), 3)

    def test_truncation_does_not_claim_prose_quality_or_constraint_success(self):
        case = specialist_cases.get_cases(
            ["prose"], ["prose_constrained_last_tram"]
        )[0]
        result = specialist_cases.evaluate_prose(case, "The Last Tram", truncated=True)
        self.assertEqual(result["outcome"], "truncated")
        self.assertEqual(result["constraints_passed"], 0)
        self.assertIsNone(result["subjective_quality_score"])


if __name__ == "__main__":
    unittest.main()

