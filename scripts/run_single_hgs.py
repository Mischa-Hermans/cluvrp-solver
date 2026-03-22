"""Run one instance once with hybrid genetic search and print the result."""

from configs.default import INSTANCE_DIRS
from configs.sa import ALPHA_BALANCE
from configs.ils import (
    ILS_CONSTRUCTION_ITERATIONS,
    ILS_NEIGHBORHOOD_WEIGHTS,
    ILS_PERTURBATION_STEPS,
)
from src.cluvrp.io.instance_reader import get_instance_path, read_gvrp_instance
from src.cluvrp.experiments.run_single_instance import run_single_instance

if __name__ == "__main__":
    instance_name = "D"
    instance = read_gvrp_instance(get_instance_path(instance_name, INSTANCE_DIRS))

    result = run_single_instance(
        instance=instance,
        time_limit_seconds=200.0,
        base_seed=42,
        alpha_balance=ALPHA_BALANCE,
        construction_iterations=ILS_CONSTRUCTION_ITERATIONS,
        initial_temp=0.0,
        cooling_rate=0.0,
        iterations_per_temp=0,
        min_temp=0.0,
        max_neighbor_attempts=0,
        neighborhood_weights=ILS_NEIGHBORHOOD_WEIGHTS,
        method="hgs",
        optimizer_kwargs={
            "perturbation_steps": ILS_PERTURBATION_STEPS,
        },
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