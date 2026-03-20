"""Run all instances in parallel and collect results."""

from __future__ import annotations

from multiprocessing import Pool, cpu_count

from src.cluvrp.experiments.run_single_instance import run_single_instance
from src.cluvrp.experiments.tables import build_results_table
from src.cluvrp.tracking.checkpoints import best_cost_at_time


def _run_single_wrapper(args):
    name, instance, i, kwargs = args

    print(f"Running instance {name} ...")

    result = run_single_instance(
        instance=instance,
        base_seed=kwargs["base_seed"] + i,
        time_limit_seconds=kwargs["time_limit_seconds"],
        alpha_balance=kwargs["alpha_balance"],
        construction_iterations=kwargs["construction_iterations"],
        initial_temp=kwargs["initial_temp"],
        cooling_rate=kwargs["cooling_rate"],
        iterations_per_temp=kwargs["iterations_per_temp"],
        min_temp=kwargs["min_temp"],
        max_neighbor_attempts=kwargs["max_neighbor_attempts"],
        neighborhood_weights=kwargs["neighborhood_weights"],
    )

    history = result["history"]
    checkpoint_costs = {
        t: best_cost_at_time(history, float(t))
        for t in kwargs["checkpoint_seconds"]
    }

    return name, {
        "initial_solution": result["initial_solution"],
        "best_solution": result["best_solution"],
        "history": history,
        "stats": {
            "elapsed_time": result["stats"].elapsed_time,
            "iterations": result["stats"].iterations,
            "accepted_moves": result["stats"].accepted_moves,
            "improving_moves": result["stats"].improving_moves,
            "final_temperature": result["stats"].final_temperature,
        },
        "checkpoint_costs": checkpoint_costs,
    }


def run_benchmark(
    instances: dict,
    instance_names: list[str],
    checkpoint_seconds: list[int],
    time_limit_seconds: float,
    base_seed: int,
    best_known_soft: dict,
    best_known_hard: dict,
    alpha_balance: float,
    construction_iterations: int,
    initial_temp: float,
    cooling_rate: float,
    iterations_per_temp: int,
    min_temp: float,
    max_neighbor_attempts: int,
    neighborhood_weights: dict,
):
    # pack shared kwargs
    kwargs = {
        "checkpoint_seconds": checkpoint_seconds,
        "time_limit_seconds": time_limit_seconds,
        "base_seed": base_seed,
        "alpha_balance": alpha_balance,
        "construction_iterations": construction_iterations,
        "initial_temp": initial_temp,
        "cooling_rate": cooling_rate,
        "iterations_per_temp": iterations_per_temp,
        "min_temp": min_temp,
        "max_neighbor_attempts": max_neighbor_attempts,
        "neighborhood_weights": neighborhood_weights,
    }

    tasks = [
        (name, instances[name], i, kwargs)
        for i, name in enumerate(instance_names)
    ]

    # use all cores minus one
    n_workers = max(1, cpu_count() - 1)

    with Pool(processes=n_workers) as pool:
        results = pool.map(_run_single_wrapper, tasks)

    # collect results
    all_runs = {name: res for name, res in results}

    results_df = build_results_table(
        all_runs=all_runs,
        instance_names=instance_names,
        checkpoint_seconds=checkpoint_seconds,
        best_known_soft=best_known_soft,
        best_known_hard=best_known_hard,
    )

    return results_df, all_runs