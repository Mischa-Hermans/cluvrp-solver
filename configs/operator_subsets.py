"""Settings for neighborhood subset analysis."""

from itertools import combinations

ANALYSIS_INSTANCE_NAMES = ["D", "G", "F", "K"]
ANALYSIS_SEEDS = [11, 22, 33]
ANALYSIS_TIME_LIMIT_SECONDS = 60.0

ALL_OPERATORS = [
    "relocate_best",
    "remove_reinsert_two",
    "pair_relocate_best",
    "swap_two_one",
    "remove_reinsert_three",
]


def build_all_operator_subsets(operators: list[str]) -> dict:
    operator_sets = {}

    for r in range(1, len(operators) + 1):
        for subset in combinations(operators, r):
            name = "_".join(subset)
            operator_sets[name] = {op: 1.0 for op in subset}

    return operator_sets


OPERATOR_SETS = build_all_operator_subsets(ALL_OPERATORS)