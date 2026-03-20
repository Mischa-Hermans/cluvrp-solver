"""Run one instance once and print the result."""

from configs.default import INSTANCE_DIRS
from configs.sa import (
    ALPHA_BALANCE,
    CONSTRUCTION_ITERATIONS,
    SA_INITIAL_TEMP,
    SA_COOLING_RATE,
    SA_ITERATIONS_PER_TEMP,
    SA_MIN_TEMP,
    SA_MAX_NEIGHBOR_ATTEMPTS,
    NEIGHBORHOOD_WEIGHTS,
)
from src.cluvrp.io.instance_reader import get_instance_path, read_gvrp_instance
from src.cluvrp.experiments.run_single_instance import run_single_instance 

if __name__ == "__main__":
    instance_name = "C"
    instance = read_gvrp_instance(get_instance_path(instance_name, INSTANCE_DIRS))

    result = run_single_instance(
        instance=instance,
        time_limit_seconds=50.0,
        base_seed=42,
        alpha_balance=ALPHA_BALANCE,
        construction_iterations=CONSTRUCTION_ITERATIONS,
        initial_temp=SA_INITIAL_TEMP,
        cooling_rate=SA_COOLING_RATE,
        iterations_per_temp=SA_ITERATIONS_PER_TEMP,
        min_temp=SA_MIN_TEMP,
        max_neighbor_attempts=SA_MAX_NEIGHBOR_ATTEMPTS,
        neighborhood_weights=NEIGHBORHOOD_WEIGHTS,
    )

    best = result["best_solution"]
    stats = result["stats"]

    print("Instance:", instance.name)
    print("Vehicles:", instance.vehicles)
    print("Capacity:", instance.capacity)
    print("Best solution cost found:", round(best.total_cost, 3))
    print("Iterations:", stats.iterations)
    print("Accepted moves:", stats.accepted_moves)
    print("Improving moves:", stats.improving_moves)
    print("Superclusters:", best.superclusters)
    print("Loads:", best.loads)