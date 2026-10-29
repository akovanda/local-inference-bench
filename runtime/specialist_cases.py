"""Frozen coding and prose cases for the local specialist-model comparison.

The suite is deliberately static.  Any change to a prompt, assertion, constraint,
or rubric must also change ``CASE_VERSION`` and ``FROZEN_SUITE_SHA256`` before a
benchmark run.  Coding answers execute only through ``bench_cases``' existing
restricted subprocess evaluator.  Prose checks cover mechanical constraints;
they never stand in for the later blinded quality review.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping, Sequence

import bench_cases


CASE_VERSION = "2026-10-19.specialist.2"


@dataclass(frozen=True)
class CodeAssertion:
    assertion_id: str
    spec_json: str

    def spec(self) -> Mapping[str, Any]:
        value = json.loads(self.spec_json)
        if not isinstance(value, dict):  # guarded by frozen suite construction
            raise ValueError(f"assertion {self.assertion_id!r} is not a JSON object")
        return value


@dataclass(frozen=True)
class CodeObjective:
    objective_id: str
    function_name: str
    assertions: tuple[CodeAssertion, ...]


@dataclass(frozen=True)
class ProseConstraint:
    constraint_id: str
    kind: str
    value_json: str

    def value(self) -> Any:
        return json.loads(self.value_json)


@dataclass(frozen=True)
class RubricCriterion:
    criterion_id: str
    guidance: str
    max_points: int


@dataclass(frozen=True)
class SpecialistCase:
    case_id: str
    group: str
    prompt: str
    max_tokens: int
    category: str
    code_objectives: tuple[CodeObjective, ...] = ()
    required_top_level_functions: tuple[str, ...] = ()
    required_function_signatures: tuple[tuple[str, tuple[str, ...]], ...] = ()
    prose_constraints: tuple[ProseConstraint, ...] = ()
    rubric: tuple[RubricCriterion, ...] = ()


def _assertion(assertion_id: str, spec: Mapping[str, Any]) -> CodeAssertion:
    return CodeAssertion(
        assertion_id,
        json.dumps(spec, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
    )


def _constraint(constraint_id: str, kind: str, value: Any) -> ProseConstraint:
    return ProseConstraint(
        constraint_id,
        kind,
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
    )


FRESH_MODULE_PROMPT = '''Create a small Python module from scratch. Implement exactly these two top-level functions and no others:

    def parse_size(text):
    def pack_batches(sizes, limit):

parse_size accepts a string consisting only of one or more ASCII decimal digits followed immediately by B, KiB, or MiB. Return the byte count using factors 1, 1024, and 1048576. Reject every other input, including bools, non-strings, signs, whitespace, decimals, missing units, and leading zeroes on multi-digit numbers, by raising ValueError.

pack_batches accepts a list of positive integers and a positive integer limit. bool is not an integer for this task. Preserve input order and greedily return a new list of batches, where each batch is a new list whose sum is at most limit. Begin a new batch exactly when the next size would exceed the limit. Raise TypeError for a non-list sizes input or any bool/non-integer size or limit. Raise ValueError for non-positive values or an item above limit. Do not mutate or alias the input.

Use no imports, decorators, classes, globals, or top-level statements other than the two function definitions. Return only the complete Python module, either as raw code or in one outer Python code fence.'''


BUGFIX_SOURCE = '''def _require_nonnegative_int(value, label):
    if not isinstance(value, int):
        raise TypeError(label + " must be int")
    if value < 0:
        raise ValueError(label + " must be nonnegative")
    return value


def apply_delta(balance, debit):
    balance = _require_nonnegative_int(balance, "balance")
    debit = _require_nonnegative_int(debit, "debit")
    if debit > balance:
        raise ValueError("insufficient balance")
    return balance + debit


def transfer(balances, source, destination, amount):
    if type(balances) is not dict:
        raise TypeError("balances must be dict")
    amount = _require_nonnegative_int(amount, "amount")
    if amount == 0:
        raise ValueError("amount must be positive")
    if source == destination:
        raise ValueError("accounts must differ")
    if source not in balances or destination not in balances:
        raise ValueError("unknown account")
    updated = balances
    updated[source] = apply_delta(updated[source], amount)
    updated[destination] -= amount
    return updated


def settle_batch(balances, transfers):
    if type(transfers) is not list:
        raise TypeError("transfers must be list")
    current = balances
    for item in transfers:
        current = transfer(current, item["destination"], item["source"], item["amount"])
    return current
'''


BUGFIX_PROMPT = (
    "Repair the supplied Python module. Its four functions have interacting bugs, so return the entire corrected module rather than a patch. Preserve all four function names and signatures, add no top-level functions, and use no imports, decorators, classes, globals, or non-function top-level statements.\n\n"
    "Required behavior:\n"
    "- _require_nonnegative_int accepts only type(value) is int, rejecting bool and other types with TypeError and negative integers with ValueError.\n"
    "- apply_delta returns balance minus debit after validation and raises ValueError when debit exceeds balance.\n"
    "- transfer requires a dict whose keys are account strings and values are nonnegative integers, two distinct string account names present in the dict, and a positive integer amount. It returns a new dict with amount debited from source and credited to destination. It never mutates or aliases the input. Invalid types raise TypeError; missing accounts, equal accounts, non-positive amounts, or insufficient funds raise ValueError.\n"
    "- settle_batch requires a list of transfer dictionaries having exactly source, destination, and amount keys. Apply transfers in order using the same rules. Return a new final dict and leave all inputs unchanged, including when a later transfer fails.\n\n"
    "Return only raw Python or one outer Python code fence.\n\nBEGIN MODULE\n"
    + BUGFIX_SOURCE
    + "END MODULE"
)


TRANSFORMATION_PROMPT = '''Implement this Python function without imports:

    def pivot_readings(rows, sensor_order):

rows must be a list of dictionaries with exactly the keys timestamp, sensor, and value. timestamp and sensor must be non-empty strings. value must be an int or float but not bool. sensor_order must be a list of unique, non-empty strings, and every row sensor must occur in it. Duplicate timestamp/sensor pairs are invalid.

Return a new dictionary with exactly these keys:
- timestamps: all distinct timestamps in ascending string order;
- sensors: a new list preserving sensor_order;
- values: one new row per timestamp, in timestamps order, with values in sensor_order and None where a reading is absent.

Raise TypeError for a wrong container, key, or scalar type. Raise ValueError for empty strings, duplicate sensor_order entries, an unknown sensor, or a duplicate timestamp/sensor pair. Empty rows and sensor_order produce {"timestamps": [], "sensors": [], "values": []}. Do not mutate or alias either input. Return only Python code, raw or in one outer Python code fence.'''


FACTUAL_REWRITE_PROMPT = '''Rewrite the source note as a concise public briefing. Use only facts in the source: do not infer causes, benefits, trends, or opinions. Output a line containing exactly "Briefing:" followed by one paragraph of 95 to 130 words. Use no bullets.

SOURCE NOTE
The City of Bellweather opened the Harbor Link e-bike pilot on May 14, 2025. The pilot serves 18 apartment buildings. By June 30, 2025, 612 residents had registered and the system had logged 4,260 trips. The median trip lasted 11 minutes, and 92% of rides began between 6 a.m. and 10 p.m. The city received 17 incident reports: 14 concerned improper parking and 3 concerned app unlock failures. No injuries were reported. The pilot cost $184,000, funded by a two-year mobility grant. The council review is scheduled for September 9, 2025. Staff has not recommended either extending or cancelling the pilot.
END SOURCE'''


FICTION_PROMPT = '''Write a complete short scene with the title "The Last Tram" on its own first line and exactly three body paragraphs totaling 180 to 230 words. Narrate in first-person plural and present tense, with a restrained, quietly melancholy voice. The setting is an empty seaside tram depot at dawn after the final scheduled service. Include the exact lowercase phrase "the rails remember rain" exactly once. Include a red ticket punch, salt on a window, and a timetable being folded. Use no dialogue, no named characters, no quotation marks, and none of the words ghost, dream, memory, or nostalgic in any capitalization. End with exactly: We leave the light on.'''


EDIT_SOURCE = '''Project Orion migration began on March 3. Staging migration is complete. Production migration is paused because a vendor archive checksum mismatch was found on March 7. The team agreed that production will remain paused until the replacement archive is verified, and support will be notified only after a new deployment date is approved. Ana Ibarra owns archive verification and is due to report by March 12. Ravi Chen will schedule production deployment after verification is approved. No customer data has moved, and support has not been notified. The vendor has not given an arrival time for the corrected archive. The March 15 deployment window might no longer be available.'''


STRUCTURED_EDIT_PROMPT = (
    "Edit the supplied project note into exactly four sections in this order: Status, Decisions, Actions, Open questions. Put each heading alone on a line with that exact capitalization and no Markdown heading marks. Status must be one paragraph of 35 to 60 words with no bullets. Decisions must have exactly two '-' bullets. Actions must have exactly two '-' bullets, one for Ana Ibarra with the March 12 due date and one for Ravi Chen. Open questions must have exactly two '-' bullets: corrected archive timing and March 15 window availability. Preserve source facts, add none, and use no other text.\n\nSOURCE NOTE\n"
    + EDIT_SOURCE
    + "\nEND SOURCE"
)


CASES: tuple[SpecialistCase, ...] = (
    SpecialistCase(
        case_id="code_fresh_batch_module",
        group="code",
        category="fresh_module_generation",
        prompt=FRESH_MODULE_PROMPT,
        max_tokens=4096,
        required_top_level_functions=("parse_size", "pack_batches"),
        required_function_signatures=(
            ("parse_size", ("text",)),
            ("pack_batches", ("sizes", "limit")),
        ),
        code_objectives=(
            CodeObjective(
                "parse_size",
                "parse_size",
                (
                    _assertion("bytes", {"args": ["17B"], "expected": 17}),
                    _assertion("kibibytes", {"args": ["2KiB"], "expected": 2048}),
                    _assertion("mebibytes", {"args": ["3MiB"], "expected": 3145728}),
                    _assertion("leading_zero", {"args": ["01KiB"], "raises": "ValueError"}),
                    _assertion("whitespace", {"args": [" 2KiB"], "raises": "ValueError"}),
                    _assertion("wrong_type", {"args": [2], "raises": "ValueError"}),
                ),
            ),
            CodeObjective(
                "pack_batches",
                "pack_batches",
                (
                    _assertion(
                        "greedy_boundary",
                        {
                            "args": [[4, 3, 5, 2, 2], 7],
                            "expected": [[4, 3], [5, 2], [2]],
                            "assert_inputs_unchanged": True,
                        },
                    ),
                    _assertion(
                        "empty",
                        {
                            "args": [[], 5],
                            "expected": [],
                            "assert_inputs_unchanged": True,
                            "mutate_result_path": [],
                        },
                    ),
                    _assertion("exact_limit", {"args": [[5, 5], 5], "expected": [[5], [5]]}),
                    _assertion("bool_item", {"args": [[True], 5], "raises": "TypeError"}),
                    _assertion("oversized", {"args": [[6], 5], "raises": "ValueError"}),
                    _assertion("non_list", {"args": ["12", 5], "raises": "TypeError"}),
                ),
            ),
        ),
    ),
    SpecialistCase(
        case_id="code_multifunction_ledger_bugfix",
        group="code",
        category="multifunction_bugfix",
        prompt=BUGFIX_PROMPT,
        max_tokens=4096,
        required_top_level_functions=(
            "_require_nonnegative_int",
            "apply_delta",
            "transfer",
            "settle_batch",
        ),
        required_function_signatures=(
            ("_require_nonnegative_int", ("value", "label")),
            ("apply_delta", ("balance", "debit")),
            ("transfer", ("balances", "source", "destination", "amount")),
            ("settle_batch", ("balances", "transfers")),
        ),
        code_objectives=(
            CodeObjective(
                "apply_delta",
                "apply_delta",
                (
                    _assertion("debit", {"args": [100, 37], "expected": 63}),
                    _assertion("exact_balance", {"args": [8, 8], "expected": 0}),
                    _assertion("bool_rejected", {"args": [True, 1], "raises": "TypeError"}),
                    _assertion("insufficient", {"args": [3, 4], "raises": "ValueError"}),
                ),
            ),
            CodeObjective(
                "transfer",
                "transfer",
                (
                    _assertion(
                        "debit_credit_copy",
                        {
                            "args": [{"cash": 90, "reserve": 10}, "cash", "reserve", 25],
                            "expected": {"cash": 65, "reserve": 35},
                            "assert_inputs_unchanged": True,
                        },
                    ),
                    _assertion(
                        "third_account_preserved",
                        {
                            "args": [{"a": 4, "b": 1, "c": 9}, "a", "b", 4],
                            "expected": {"a": 0, "b": 5, "c": 9},
                            "assert_inputs_unchanged": True,
                        },
                    ),
                    _assertion("unknown", {"args": [{"a": 2}, "a", "b", 1], "raises": "ValueError"}),
                    _assertion("bool_balance", {"args": [{"a": True, "b": 0}, "a", "b", 1], "raises": "TypeError"}),
                    _assertion("non_string_account", {"args": [{"a": 2, "b": 0}, 1, "b", 1], "raises": "TypeError"}),
                ),
            ),
            CodeObjective(
                "settle_batch",
                "settle_batch",
                (
                    _assertion(
                        "ordered_transfers",
                        {
                            "args": [
                                {"a": 9, "b": 1, "c": 0},
                                [
                                    {"source": "a", "destination": "b", "amount": 4},
                                    {"source": "b", "destination": "c", "amount": 3},
                                ],
                            ],
                            "expected": {"a": 5, "b": 2, "c": 3},
                            "assert_inputs_unchanged": True,
                        },
                    ),
                    _assertion(
                        "empty_batch_copy",
                        {
                            "args": [{"a": 2}, []],
                            "expected": {"a": 2},
                            "assert_inputs_unchanged": True,
                            "mutate_result_path": [],
                        },
                    ),
                    _assertion(
                        "later_failure_atomic",
                        {
                            "args": [
                                {"a": 5, "b": 0},
                                [
                                    {"source": "a", "destination": "b", "amount": 2},
                                    {"source": "a", "destination": "b", "amount": 9},
                                ],
                            ],
                            "raises": "ValueError",
                            "assert_inputs_unchanged": True,
                        },
                    ),
                    _assertion(
                        "exact_transfer_keys",
                        {
                            "args": [
                                {"a": 5, "b": 0},
                                [{"source": "a", "destination": "b", "amount": 1, "note": "x"}],
                            ],
                            "raises": "TypeError",
                        },
                    ),
                ),
            ),
        ),
    ),
    SpecialistCase(
        case_id="code_edge_reading_pivot",
        group="code",
        category="edge_heavy_data_transformation",
        prompt=TRANSFORMATION_PROMPT,
        max_tokens=4096,
        required_top_level_functions=("pivot_readings",),
        required_function_signatures=(("pivot_readings", ("rows", "sensor_order")),),
        code_objectives=(
            CodeObjective(
                "core_pivot",
                "pivot_readings",
                (
                    _assertion(
                        "sort_fill_and_order",
                        {
                            "args": [
                                [
                                    {"timestamp": "2026-03-02", "sensor": "b", "value": 4},
                                    {"timestamp": "2026-03-01", "sensor": "a", "value": 1.5},
                                    {"timestamp": "2026-03-02", "sensor": "a", "value": 3},
                                ],
                                ["b", "a", "c"],
                            ],
                            "expected": {
                                "timestamps": ["2026-03-01", "2026-03-02"],
                                "sensors": ["b", "a", "c"],
                                "values": [[None, 1.5, None], [4, 3, None]],
                            },
                            "assert_inputs_unchanged": True,
                            "mutate_result_path": ["sensors"],
                        },
                    ),
                    _assertion(
                        "empty",
                        {
                            "args": [[], []],
                            "expected": {"timestamps": [], "sensors": [], "values": []},
                            "assert_inputs_unchanged": True,
                            "mutate_result_path": ["sensors"],
                        },
                    ),
                ),
            ),
            CodeObjective(
                "edge_values",
                "pivot_readings",
                (
                    _assertion(
                        "zero_and_negative",
                        {
                            "args": [
                                [
                                    {"timestamp": "t", "sensor": "x", "value": 0},
                                    {"timestamp": "t", "sensor": "y", "value": -2.25},
                                ],
                                ["x", "y"],
                            ],
                            "expected": {"timestamps": ["t"], "sensors": ["x", "y"], "values": [[0, -2.25]]},
                        },
                    ),
                    _assertion(
                        "sparse_timestamps",
                        {
                            "args": [
                                [
                                    {"timestamp": "z", "sensor": "only", "value": 9},
                                    {"timestamp": "a", "sensor": "only", "value": 2},
                                ],
                                ["only", "missing"],
                            ],
                            "expected": {"timestamps": ["a", "z"], "sensors": ["only", "missing"], "values": [[2, None], [9, None]]},
                        },
                    ),
                ),
            ),
            CodeObjective(
                "input_validation",
                "pivot_readings",
                (
                    _assertion(
                        "duplicate_pair",
                        {
                            "args": [
                                [
                                    {"timestamp": "t", "sensor": "x", "value": 1},
                                    {"timestamp": "t", "sensor": "x", "value": 2},
                                ],
                                ["x"],
                            ],
                            "raises": "ValueError",
                        },
                    ),
                    _assertion(
                        "duplicate_sensor_order",
                        {"args": [[], ["x", "x"]], "raises": "ValueError"},
                    ),
                    _assertion(
                        "extra_key",
                        {
                            "args": [[{"timestamp": "t", "sensor": "x", "value": 1, "unit": "C"}], ["x"]],
                            "raises": "TypeError",
                        },
                    ),
                    _assertion(
                        "bool_value",
                        {"args": [[{"timestamp": "t", "sensor": "x", "value": True}], ["x"]], "raises": "TypeError"},
                    ),
                    _assertion(
                        "unknown_sensor",
                        {"args": [[{"timestamp": "t", "sensor": "x", "value": 1}], ["y"]], "raises": "ValueError"},
                    ),
                ),
            ),
        ),
    ),
    SpecialistCase(
        case_id="prose_grounded_public_briefing",
        group="prose",
        category="source_grounded_factual_rewrite",
        prompt=FACTUAL_REWRITE_PROMPT,
        max_tokens=1536,
        prose_constraints=(
            _constraint("heading", "first_line_exact", "Briefing:"),
            _constraint("body_paragraph", "body_paragraph_count", 1),
            _constraint("word_count", "body_word_count_range", [95, 130]),
            _constraint("no_bullets", "forbidden_line_prefixes", ["-", "*", "•"]),
            _constraint(
                "source_numbers_only",
                "allowed_number_tokens",
                ["14", "2025", "18", "30", "612", "4,260", "11", "92", "6", "10", "17", "3", "184,000", "two", "9"],
            ),
            _constraint(
                "core_source_lexical_proxy",
                "required_casefold_substrings",
                ["harbor link", "bellweather", "no injuries", "$184,000", "september 9"],
            ),
        ),
        rubric=(
            RubricCriterion("factual_fidelity", "Every claim is supported by the supplied note; omissions do not distort it.", 5),
            RubricCriterion("clarity", "The briefing is concise, readable, and orders facts for a public audience.", 3),
            RubricCriterion("prose_control", "Sentences flow naturally without hype, redundancy, or source-like choppiness.", 2),
        ),
    ),
    SpecialistCase(
        case_id="prose_constrained_last_tram",
        group="prose",
        category="constrained_voice_style_fiction",
        prompt=FICTION_PROMPT,
        max_tokens=1536,
        prose_constraints=(
            _constraint("title", "first_line_exact", "The Last Tram"),
            _constraint("paragraphs", "body_paragraph_count", 3),
            _constraint("word_count", "body_word_count_range", [180, 230]),
            _constraint("required_phrase_once", "exact_substring_count", ["the rails remember rain", 1]),
            _constraint("exact_ending", "ends_with", "We leave the light on."),
            _constraint("required_objects", "required_casefold_substrings", ["red ticket punch", "salt on a window", "timetable"]),
            _constraint("forbidden_words", "forbidden_words_casefold", ["ghost", "dream", "memory", "nostalgic"]),
            _constraint("no_quotation_marks", "forbidden_substrings", ["\"", "“", "”"]),
        ),
        rubric=(
            RubricCriterion("voice", "First-person plural and present tense remain natural and consistent.", 3),
            RubricCriterion("style", "The scene sustains restrained melancholy through precise imagery rather than explanation.", 4),
            RubricCriterion("scene_arc", "The three paragraphs form a complete scene with movement and a resonant ending.", 3),
        ),
    ),
    SpecialistCase(
        case_id="prose_structured_orion_edit",
        group="prose",
        category="structured_editing",
        prompt=STRUCTURED_EDIT_PROMPT,
        max_tokens=1536,
        prose_constraints=(
            _constraint("section_order", "section_order_exact", ["Status", "Decisions", "Actions", "Open questions"]),
            _constraint("status_length", "section_word_count_range", ["Status", 35, 60]),
            _constraint("status_no_bullets", "section_forbidden_line_prefixes", ["Status", ["-", "*", "•"]]),
            _constraint("bullet_counts", "section_bullet_counts", {"Decisions": 2, "Actions": 2, "Open questions": 2}),
            _constraint("owners", "required_casefold_substrings", ["ana ibarra", "march 12", "ravi chen"]),
            _constraint("dates", "required_casefold_substrings", ["march 3", "march 7", "march 15"]),
            _constraint("no_extra_numbers", "allowed_number_tokens", ["3", "7", "12", "15"]),
        ),
        rubric=(
            RubricCriterion("source_fidelity", "All source facts are preserved accurately and no new status or commitment appears.", 5),
            RubricCriterion("information_design", "Facts are placed in the correct section with decisions, owners, and unknowns easy to scan.", 3),
            RubricCriterion("editing_quality", "Wording is concise, parallel, and free of ambiguity or needless repetition.", 2),
        ),
    ),
)


def _case_payload(case: SpecialistCase) -> dict[str, Any]:
    return asdict(case)


def case_sha256(case: SpecialistCase) -> str:
    encoded = json.dumps(
        _case_payload(case), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def suite_sha256(cases: Sequence[SpecialistCase] = CASES) -> str:
    encoded = json.dumps(
        [_case_payload(case) for case in cases],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


# This literal pins every prompt, expected assertion, prose constraint, and rubric.
FROZEN_SUITE_SHA256 = "683416fc8d3559adff123b78a0f2ad38cefb43422d440c557a1ccabcbdff9b26"


def get_cases(
    groups: Sequence[str] | None = None,
    case_ids: Sequence[str] | None = None,
) -> list[SpecialistCase]:
    selected_groups = tuple(dict.fromkeys(groups or ("code", "prose")))
    invalid_groups = [group for group in selected_groups if group not in {"code", "prose"}]
    if invalid_groups:
        raise ValueError("unknown group(s): " + ", ".join(invalid_groups))
    available = {case.case_id: case for case in CASES if case.group in selected_groups}
    if not case_ids:
        return [case for case in CASES if case.group in selected_groups]
    wanted = tuple(dict.fromkeys(case_ids))
    missing = [case_id for case_id in wanted if case_id not in available]
    if missing:
        raise ValueError(
            "unknown or excluded case(s): "
            + ", ".join(missing)
            + "; available: "
            + ", ".join(available)
        )
    return [available[case_id] for case_id in wanted]


def _top_level_function_names(source: str) -> tuple[str, ...]:
    tree = ast.parse(source, mode="exec")
    return tuple(node.name for node in tree.body if isinstance(node, ast.FunctionDef))


def _top_level_function_signatures(
    source: str,
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    tree = ast.parse(source, mode="exec")
    signatures: list[tuple[str, tuple[str, ...]]] = []
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        arguments = node.args
        if (
            arguments.posonlyargs
            or arguments.vararg is not None
            or arguments.kwonlyargs
            or arguments.kwarg is not None
            or arguments.defaults
            or arguments.kw_defaults
            or node.returns is not None
            or any(arg.annotation is not None for arg in arguments.args)
        ):
            signatures.append((node.name, ("<non-exact-signature>",)))
        else:
            signatures.append((node.name, tuple(arg.arg for arg in arguments.args)))
    return tuple(signatures)


def _safety_policy_only(errors: Sequence[str]) -> bool:
    if not errors:
        return False
    return all(
        error == "private or dunder attribute access is not allowed"
        or (error.startswith("name ") and error.endswith(" is not allowed"))
        or (error.startswith("call to ") and error.endswith(" is not allowed"))
        for error in errors
    )


def _run_code_assertion(
    source: str,
    function_name: str,
    assertion: CodeAssertion,
) -> dict[str, Any]:
    spec = dict(assertion.spec())
    expected_exception = spec.get("raises")
    if expected_exception and spec.get("assert_inputs_unchanged"):
        # bench_cases' runner intentionally short-circuits after an expected
        # exception.  Catch it inside evaluator-generated restricted code so the
        # outer runner can still compare its deep-copied arguments for mutation.
        allowed_exceptions = {"IndexError", "KeyError", "StopIteration", "TypeError", "ValueError"}
        if expected_exception not in allowed_exceptions:
            return {
                "passed": False,
                "detail": f"unsupported expected exception {expected_exception!r}",
                "validation": "rejected",
            }
        args = spec.get("args")
        if not isinstance(args, list):
            return {"passed": False, "detail": "assertion args must be a list", "validation": "rejected"}
        parameters = [f"probe_arg_{index}" for index in range(len(args))]
        joined = ", ".join(parameters)
        call_args = joined
        sentinel = "specialist_expected_exception_observed"
        wrapper = (
            f"\n\ndef specialist_exception_probe({joined}):\n"
            f"    try:\n"
            f"        {function_name}({call_args})\n"
            f"    except {expected_exception}:\n"
            f"        return {sentinel!r}\n"
            f"    return 'specialist_expected_exception_missing'\n"
        )
        wrapped_spec = {
            "args": args,
            "expected": sentinel,
            "assert_inputs_unchanged": True,
        }
        return bench_cases.run_python_checks(
            source + wrapper,
            "specialist_exception_probe",
            [wrapped_spec],
        )
    return bench_cases.run_python_checks(source, function_name, [spec])


def _execution_semantics_unknown(result: Mapping[str, Any]) -> bool:
    if result.get("validation") == "rejected":
        return True
    detail = str(result.get("detail", ""))
    return detail.startswith(
        (
            "sandbox failed:",
            "candidate process exited with status",
            "candidate process returned invalid evaluator output",
            "candidate timed out",
        )
    )


def evaluate_code(case: SpecialistCase, text: str, *, truncated: bool = False) -> dict[str, Any]:
    assertion_count = sum(len(objective.assertions) for objective in case.code_objectives)
    if truncated:
        return {
            "outcome": "truncated",
            "passed": False,
            "validation": "not_run",
            "independent_task_count": 1,
            "independent_tasks_passed": 0,
            "objective_count": len(case.code_objectives),
            "objectives_passed": 0,
            "assertion_count": assertion_count,
            "assertions_passed": 0,
            "objective_outcomes": [],
            "detail": "response was incomplete or length-truncated",
        }

    source = bench_cases.extract_python_source(text)
    validation_errors: list[str] = []
    for function_name in case.required_top_level_functions:
        valid, errors = bench_cases.validate_python_source(source, function_name)
        if not valid:
            validation_errors.extend(errors)
    if validation_errors:
        unique_errors = sorted(set(validation_errors))
        outcome = "evaluator_rejected" if _safety_policy_only(unique_errors) else "wrong_answer"
        return {
            "outcome": outcome,
            "passed": False,
            "validation": "rejected",
            "validation_errors": unique_errors,
            "independent_task_count": 1,
            "independent_tasks_passed": 0,
            "objective_count": len(case.code_objectives),
            "objectives_passed": 0,
            "assertion_count": assertion_count,
            "assertions_passed": 0,
            "objective_outcomes": [],
            "source_chars": len(source),
            "detail": (
                "candidate source used behavior blocked by the restricted evaluator; semantic correctness is unknown"
                if outcome == "evaluator_rejected"
                else "candidate source violated executable prompt requirements"
            ),
        }

    actual_functions = _top_level_function_names(source)
    actual_signatures = _top_level_function_signatures(source)
    expected_names = case.required_top_level_functions
    name_constraint_passed = (
        len(actual_functions) == len(expected_names)
        and set(actual_functions) == set(expected_names)
    )
    signature_constraint_passed = (
        len(actual_signatures) == len(case.required_function_signatures)
        and dict(actual_signatures) == dict(case.required_function_signatures)
    )
    source_constraint_passed = name_constraint_passed and signature_constraint_passed
    objective_outcomes: list[dict[str, Any]] = []
    assertions_passed = 0
    evaluator_unknown = False
    for objective in case.code_objectives:
        assertion_outcomes: list[dict[str, Any]] = []
        for assertion in objective.assertions:
            result = _run_code_assertion(source, objective.function_name, assertion)
            evaluator_unknown = evaluator_unknown or _execution_semantics_unknown(result)
            passed = result.get("passed") is True
            assertions_passed += int(passed)
            assertion_outcomes.append(
                {
                    "assertion_id": assertion.assertion_id,
                    "passed": passed,
                    "detail": result.get("detail"),
                }
            )
        objective_outcomes.append(
            {
                "objective_id": objective.objective_id,
                "function_name": objective.function_name,
                "passed": all(item["passed"] for item in assertion_outcomes),
                "assertion_count": len(assertion_outcomes),
                "assertions_passed": sum(int(item["passed"]) for item in assertion_outcomes),
                "assertion_outcomes": assertion_outcomes,
            }
        )

    tasks_passed = sum(int(item["passed"]) for item in objective_outcomes)
    passed = (
        source_constraint_passed
        and tasks_passed == len(case.code_objectives)
        and assertions_passed == assertion_count
    )
    outcome = "passed" if passed else ("evaluator_rejected" if evaluator_unknown else "wrong_answer")
    return {
        "outcome": outcome,
        "passed": passed,
        "validation": "accepted",
        "source_constraint_passed": source_constraint_passed,
        "function_name_constraint_passed": name_constraint_passed,
        "function_signature_constraint_passed": signature_constraint_passed,
        "required_top_level_functions": list(case.required_top_level_functions),
        "actual_top_level_functions": list(actual_functions),
        "required_function_signatures": [
            [name, list(arguments)] for name, arguments in case.required_function_signatures
        ],
        "actual_function_signatures": [
            [name, list(arguments)] for name, arguments in actual_signatures
        ],
        "independent_task_count": 1,
        "independent_tasks_passed": int(passed),
        "objective_count": len(case.code_objectives),
        "objectives_passed": tasks_passed,
        "assertion_count": assertion_count,
        "assertions_passed": assertions_passed,
        "objective_outcomes": objective_outcomes,
        "source_chars": len(source),
        "detail": (
            "all source constraints and objective assertions passed"
            if passed
            else (
                "restricted execution could not establish semantic correctness"
                if outcome == "evaluator_rejected"
                else "one or more source constraints or objective assertions failed"
            )
        ),
    }


_WORD_RE = re.compile(r"\b[\w'’-]+\b", re.UNICODE)
_NUMBER_RE = re.compile(r"(?<![\w])\d[\d,]*(?![\w])")


def _body_after_first_line(text: str) -> str:
    parts = text.strip().splitlines()
    return "\n".join(parts[1:]).strip() if parts else ""


def _paragraphs(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"\n\s*\n", text.strip()) if part.strip()]


def _sections(text: str, headings: Sequence[str]) -> tuple[list[str], dict[str, str]]:
    lines = text.strip().splitlines()
    found = [line.strip() for line in lines if line.strip() in headings]
    sections: dict[str, list[str]] = {heading: [] for heading in headings}
    current: str | None = None
    for line in lines:
        stripped = line.strip()
        if stripped in headings:
            current = stripped
        elif current is not None:
            sections[current].append(line)
    return found, {key: "\n".join(value).strip() for key, value in sections.items()}


def _evaluate_constraint(constraint: ProseConstraint, text: str) -> tuple[bool, str]:
    value = constraint.value()
    stripped = text.strip()
    lines = stripped.splitlines()
    casefolded = stripped.casefold()
    if constraint.kind == "first_line_exact":
        passed = bool(lines) and lines[0].strip() == value
    elif constraint.kind == "body_paragraph_count":
        passed = len(_paragraphs(_body_after_first_line(stripped))) == value
    elif constraint.kind == "body_word_count_range":
        count = len(_WORD_RE.findall(_body_after_first_line(stripped)))
        passed = value[0] <= count <= value[1]
        return passed, f"body word count {count}; required {value[0]}..{value[1]}"
    elif constraint.kind == "forbidden_line_prefixes":
        passed = not any(
            line.lstrip().startswith(tuple(value)) for line in lines if line.strip()
        )
    elif constraint.kind == "allowed_number_tokens":
        actual = _NUMBER_RE.findall(stripped)
        passed = all(token in value for token in actual)
        return passed, f"number tokens {actual!r}; allowed {value!r}"
    elif constraint.kind == "required_casefold_substrings":
        missing = [item for item in value if item.casefold() not in casefolded]
        passed = not missing
        return passed, "all required text present" if passed else f"missing required text: {missing!r}"
    elif constraint.kind == "exact_substring_count":
        actual = stripped.count(value[0])
        passed = actual == value[1]
        return passed, f"substring count {actual}; required {value[1]}"
    elif constraint.kind == "ends_with":
        passed = stripped.endswith(value)
    elif constraint.kind == "forbidden_words_casefold":
        words = {word.casefold() for word in _WORD_RE.findall(stripped)}
        found = [word for word in value if word.casefold() in words]
        passed = not found
        return passed, "no forbidden words" if passed else f"forbidden words present: {found!r}"
    elif constraint.kind == "forbidden_substrings":
        found = [item for item in value if item in stripped]
        passed = not found
        return passed, "no forbidden substrings" if passed else f"forbidden substrings present: {found!r}"
    elif constraint.kind == "section_order_exact":
        found, _ = _sections(stripped, value)
        passed = found == value
        return passed, f"section headings {found!r}; required {value!r}"
    elif constraint.kind == "section_word_count_range":
        headings = ("Status", "Decisions", "Actions", "Open questions")
        _, sections = _sections(stripped, headings)
        count = len(_WORD_RE.findall(sections.get(value[0], "")))
        passed = value[1] <= count <= value[2]
        return passed, f"{value[0]} word count {count}; required {value[1]}..{value[2]}"
    elif constraint.kind == "section_bullet_counts":
        headings = tuple(value)
        # Include all expected document headings so section bodies stop correctly.
        all_headings = ("Status", "Decisions", "Actions", "Open questions")
        _, sections = _sections(stripped, all_headings)
        actual = {
            heading: sum(1 for line in sections.get(heading, "").splitlines() if line.lstrip().startswith("- "))
            for heading in headings
        }
        only_bullets = all(
            all(
                not line.strip() or line.lstrip().startswith("- ")
                for line in sections.get(heading, "").splitlines()
            )
            for heading in headings
        )
        passed = actual == value and only_bullets
        return passed, f"bullet counts {actual!r}; required {value!r}"
    elif constraint.kind == "section_forbidden_line_prefixes":
        headings = ("Status", "Decisions", "Actions", "Open questions")
        _, sections = _sections(stripped, headings)
        found = [
            line
            for line in sections.get(value[0], "").splitlines()
            if line.lstrip().startswith(tuple(value[1]))
        ]
        passed = not found
        return passed, "no forbidden section line prefixes" if passed else f"forbidden section lines: {found!r}"
    else:
        return False, f"unknown constraint kind {constraint.kind!r}"
    return passed, "constraint met" if passed else "constraint not met"


def evaluate_prose(case: SpecialistCase, text: str, *, truncated: bool = False) -> dict[str, Any]:
    if truncated:
        outcomes: list[dict[str, Any]] = []
        outcome = "truncated"
        passed = False
    else:
        outcomes = []
        for constraint in case.prose_constraints:
            passed_one, detail = _evaluate_constraint(constraint, text)
            outcomes.append(
                {
                    "constraint_id": constraint.constraint_id,
                    "passed": passed_one,
                    "detail": detail,
                }
            )
        passed = all(item["passed"] for item in outcomes)
        outcome = "constraints_passed" if passed else "constraint_failed"
    return {
        "outcome": outcome,
        "passed": passed,
        "deterministic_constraints_only": True,
        "constraint_interpretation": "mechanical format and lexical proxies only; these are not a prose-quality or factual-correctness score",
        "constraint_count": len(case.prose_constraints),
        "constraints_passed": sum(int(item["passed"]) for item in outcomes),
        "constraint_outcomes": outcomes,
        "subjective_quality_score": None,
        "subjective_quality_status": "unscored_pending_blinded_review",
        "blinded_rubric": [asdict(item) for item in case.rubric],
        "blinded_rubric_max_points": sum(item.max_points for item in case.rubric),
        "detail": (
            "mechanical constraints checked; prose quality remains unscored"
            if not truncated
            else "response was incomplete or length-truncated; prose quality remains unscored"
        ),
    }


def evaluate_case(case: SpecialistCase, text: str, *, truncated: bool = False) -> dict[str, Any]:
    if case.group == "code":
        return evaluate_code(case, text, truncated=truncated)
    if case.group == "prose":
        return evaluate_prose(case, text, truncated=truncated)
    raise ValueError(f"unknown specialist case group {case.group!r}")


def assert_frozen_suite() -> None:
    actual = suite_sha256()
    if actual != FROZEN_SUITE_SHA256:
        raise RuntimeError(
            "specialist suite changed without updating CASE_VERSION and FROZEN_SUITE_SHA256: "
            f"expected {FROZEN_SUITE_SHA256}, got {actual}"
        )


def all_case_ids() -> tuple[str, ...]:
    return tuple(case.case_id for case in CASES)

