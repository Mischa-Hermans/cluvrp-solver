"""Run one instance once with the selected method."""

from configs.default import INSTANCE_DIRS
from configs.methods import SINGLE_METHOD
from configs.run_single import (
    SINGLE_INSTANCE_NAME,
    SINGLE_TIME_LIMIT_SECONDS,
    SINGLE_BASE_SEED,
)
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
from configs.ils import (
    ILS_CONSTRUCTION_ITERATIONS,
    ILS_NEIGHBORHOOD_WEIGHTS,
    ILS_PERTURBATION_STEPS,
)
from configs.hgs import (
    HGS_CONSTRUCTION_ITERATIONS,
    HGS_NEIGHBORHOOD_WEIGHTS,
    HGS_PERTURBATION_STEPS,
)
from src.cluvrp.io.instance_reader import get_instance_path, read_gvrp_instance
from src.cluvrp.experiments.run_single_instance import run_single_instance


def get_run_settings(method: str) -> tuple[dict, dict]:
    if method == "sa":
        return (
            {
                "alpha_balance": ALPHA_BALANCE,
                "construction_iterations": CONSTRUCTION_ITERATIONS,
                "initial_temp": SA_INITIAL_TEMP,
                "cooling_rate": SA_COOLING_RATE,
                "iterations_per_temp": SA_ITERATIONS_PER_TEMP,
                "min_temp": SA_MIN_TEMP,
                "max_neighbor_attempts": SA_MAX_NEIGHBOR_ATTEMPTS,
                "neighborhood_weights": NEIGHBORHOOD_WEIGHTS,
            },
            {},
        )

    if method == "ils":
        return (
            {
                "alpha_balance": ALPHA_BALANCE,
                "construction_iterations": ILS_CONSTRUCTION_ITERATIONS,
                "initial_temp": 0.0,
                "cooling_rate": 0.0,
                "iterations_per_temp": 0,
                "min_temp": 0.0,
                "max_neighbor_attempts": 0,
                "neighborhood_weights": ILS_NEIGHBORHOOD_WEIGHTS,
            },
            {
                "perturbation_steps": ILS_PERTURBATION_STEPS,
            },
        )

    if method == "hgs":
        return (
            {
                "alpha_balance": ALPHA_BALANCE,
                "construction_iterations": HGS_CONSTRUCTION_ITERATIONS,
                "initial_temp": 0.0,
                "cooling_rate": 0.0,
                "iterations_per_temp": 0,
                "min_temp": 0.0,
                "max_neighbor_attempts": 0,
                "neighborhood_weights": HGS_NEIGHBORHOOD_WEIGHTS,
            },
            {
                "perturbation_steps": HGS_PERTURBATION_STEPS,
            },
        )

    raise ValueError(f"Unknown method: {method}")


if __name__ == "__main__":
    instance = read_gvrp_instance(
        get_instance_path(SINGLE_INSTANCE_NAME, INSTANCE_DIRS)
    )

    base_kwargs, optimizer_kwargs = get_run_settings(SINGLE_METHOD)

    print(f"Method: {SINGLE_METHOD}")
    print(f"Instance: {SINGLE_INSTANCE_NAME}")
    print(f"Time limit: {SINGLE_TIME_LIMIT_SECONDS}s")
    print(f"Seed: {SINGLE_BASE_SEED}")

    result = run_single_instance(
        instance=instance,
        time_limit_seconds=SINGLE_TIME_LIMIT_SECONDS,
        base_seed=SINGLE_BASE_SEED,
        method=SINGLE_METHOD,
        optimizer_kwargs=optimizer_kwargs,
        **base_kwargs,
    )

    initial_solution = result["initial_solution"]
    best_solution = result["best_solution"]
    stats = result["stats"]

    print(f"Initial cost: {initial_solution.total_cost:.3f}")
    print(f"Final cost: {best_solution.total_cost:.3f}")
    print(f"Elapsed time: {stats.elapsed_time:.3f}s")