"""Settings for neighborhood ablation analysis."""

ANALYSIS_INSTANCE_NAMES = ["D", "G", "F", "K"]
ANALYSIS_SEEDS = [11, 22, 33, 44, 55, 66, 77]
ANALYSIS_TIME_LIMIT_SECONDS = 60.0

BASE_NEIGHBORHOOD_WEIGHTS = {
    "relocate_best": 1.0,
    "swap_restricted": 1.0,
    "remove_reinsert_two": 1.0,
    "ejection_chain_light": 1.0,
    "pair_relocate_best": 1.0,
    "swap_two_one": 1.0,
    "remove_reinsert_three": 1.0,
}