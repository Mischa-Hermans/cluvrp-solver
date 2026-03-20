"""Map a method name to the matching optimizer."""

from src.cluvrp.metaheuristics.simulated_annealing import optimize_with_simulated_annealing

OPTIMIZER_REGISTRY = {
    "sa": optimize_with_simulated_annealing,
}