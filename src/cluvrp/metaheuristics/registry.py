"""Map a method name to the matching optimizer."""

from src.cluvrp.metaheuristics.simulated_annealing import optimize_with_simulated_annealing
from src.cluvrp.metaheuristics.iterated_local_search import optimize_with_iterated_local_search
from src.cluvrp.metaheuristics.genetic_search import optimize_with_genetic_search
from src.cluvrp.metaheuristics.hybrid_genetic_search import optimize_with_hybrid_genetic_search

OPTIMIZER_REGISTRY = {
    "sa": optimize_with_simulated_annealing,
    "ils": optimize_with_iterated_local_search,
    "ga": optimize_with_genetic_search,
    "hgs": optimize_with_hybrid_genetic_search,
}