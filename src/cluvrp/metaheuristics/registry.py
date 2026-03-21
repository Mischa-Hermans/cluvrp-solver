"""Map a method name to the matching optimizer."""

from src.cluvrp.metaheuristics.simulated_annealing import optimize_with_simulated_annealing
from src.cluvrp.metaheuristics.iterated_local_search import optimize_with_iterated_local_search

OPTIMIZER_REGISTRY = {
    "sa": optimize_with_simulated_annealing,
    "ils": optimize_with_iterated_local_search,
}