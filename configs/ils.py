"""Settings for iterated local search."""

ILS_CONSTRUCTION_ITERATIONS = 40

ILS_NEIGHBORHOOD_WEIGHTS = {
    "relocate_best": 1.0,
    "remove_reinsert_three": 1.0,
}

ILS_PERTURBATION_STEPS = 9 