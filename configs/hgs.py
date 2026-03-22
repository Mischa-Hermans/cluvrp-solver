"""Settings for hybrid genetic search."""

HGS_POPULATION_SIZE = 12
HGS_ELITE_SIZE = 3
HGS_TOURNAMENT_SIZE = 3

# Time spent to improve each initial individual.
HGS_INITIAL_INDIVIDUAL_TIME_SECONDS = 4.0

# Time spent to improve each offspring.
HGS_OFFSPRING_IMPROVEMENT_TIME_SECONDS = 1.5

# Probability of inheriting a whole supercluster from parent 1.
HGS_PARENT1_ROUTE_INHERIT_PROB = 0.5