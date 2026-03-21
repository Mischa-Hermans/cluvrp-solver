"""Settings for simulated annealing."""

ALPHA_BALANCE = 0.15
CONSTRUCTION_ITERATIONS = 50 # 50

SA_INITIAL_TEMP = 200 # 200.0
SA_COOLING_RATE = 0.95 # 0.95
SA_ITERATIONS_PER_TEMP = 50 # 50
SA_MIN_TEMP = 1e-3
SA_MAX_NEIGHBOR_ATTEMPTS = 250

NEIGHBORHOOD_WEIGHTS = {
    "relocate_best": 1.0,
    "swap_restricted": 1.0,
    "remove_reinsert_two": 1.0,
    "ejection_chain_light": 1.0,
}