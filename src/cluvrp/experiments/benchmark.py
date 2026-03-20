"""Run all instances once and read off checkpoint values from one history."""

from __future__ import annotations

from src.cluvrp.experiments.run_single_instance import run_single_instance
from src.cluvrp.experiments.tables import build_results_table
from src.cluvrp.tracking.checkpoints import best_cost_at_time


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
    all_runs = {}

    for i, name in enumerate(instance_names):
        instance = instances[name]
        print(f"Running instance {name} ...")

        result = run_single_instance(
            instance=instance,
            time_limit_seconds=time_limit_seconds,
            base_seed=base_seed + i,
            alpha_balance=alpha_balance,
            construction_iterations=construction_iterations,
            initial_temp=initial_temp,
            cooling_rate=cooling_rate,
            iterations_per_temp=iterations_per_temp,
            min_temp=min_temp,
            max_neighbor_attempts=max_neighbor_attempts,
            neighborhood_weights=neighborhood_weights,
        )

        history = result["history"]
        checkpoint_costs = {t: best_cost_at_time(history, float(t)) for t in checkpoint_seconds}

        all_runs[name] = {
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

    results_df = build_results_table(
        all_runs=all_runs,
        instance_names=instance_names,
        checkpoint_seconds=checkpoint_seconds,
        best_known_soft=best_known_soft,
        best_known_hard=best_known_hard,
    )

    return results_df, all_runs