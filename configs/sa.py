"""Settings for simulated annealing."""

ALPHA_BALANCE = 0.15
CONSTRUCTION_ITERATIONS = 50

SA_INITIAL_TEMP = 433
SA_COOLING_RATE = 0.9757
SA_ITERATIONS_PER_TEMP = 96
SA_MIN_TEMP = 1e-3
SA_MAX_NEIGHBOR_ATTEMPTS = 250

NEIGHBORHOOD_WEIGHTS = {
    "relocate_best": 0.0,
    "swap_restricted": 0.0,
    "remove_reinsert_two": 1.0,
    "ejection_chain_light": 0.0,
    "pair_relocate_best": 1.0,
    "swap_two_one": 0.0,
    "remove_reinsert_three": 1.0,
}