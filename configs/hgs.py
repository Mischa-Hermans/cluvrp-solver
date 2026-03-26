"""Settings for hybrid genetic search."""

HGS_POPULATION_SIZE = 11
HGS_ELITE_SIZE = 4
HGS_TOURNAMENT_SIZE = 2

# Time spent to improve each initial individual.
HGS_INITIAL_INDIVIDUAL_TIME_SECONDS = 1.6467

# Time spent to improve each offspring.
HGS_OFFSPRING_IMPROVEMENT_TIME_SECONDS = 0.24

# Probability of inheriting a whole supercluster from parent 1.
HGS_PARENT1_ROUTE_INHERIT_PROB = 0.37

# HGS uses its own embedded local search settings.
HGS_CONSTRUCTION_ITERATIONS = 22

HGS_NEIGHBORHOOD_WEIGHTS = {
    "relocate_best": 1.0,
    "remove_reinsert_three": 1.0,
}

HGS_PERTURBATION_STEPS = 1