"""Settings for iterated local search."""

ILS_CONSTRUCTION_ITERATIONS = 45

ILS_NEIGHBORHOOD_WEIGHTS = {
    "relocate_best": 0.0,
    "swap_restricted": 0.0,
    "remove_reinsert_two": 1.0,
    "ejection_chain_light": 0.0,
    "pair_relocate_best": 0.0,
    "swap_two_one": 0.0,
    "remove_reinsert_three": 1.0,
}

ILS_PERTURBATION_STEPS = 7