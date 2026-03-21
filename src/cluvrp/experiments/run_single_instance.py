"""Run one instance with the selected optimizer."""

from __future__ import annotations

from src.cluvrp.metaheuristics.registry import OPTIMIZER_REGISTRY


def run_single_instance(
    instance,
    time_limit_seconds: float,
    base_seed: int,
    alpha_balance: float,
    construction_iterations: int,
    initial_temp: float,
    cooling_rate: float,
    iterations_per_temp: int,
    min_temp: float,
    max_neighbor_attempts: int,
    neighborhood_weights: dict,
    method: str = "sa",
    optimizer_kwargs: dict | None = None,
):
    if optimizer_kwargs is None:
        optimizer_kwargs = {}

    optimizer = OPTIMIZER_REGISTRY[method]

    return optimizer(
        instance=instance,
        time_limit_seconds=time_limit_seconds,
        base_seed=base_seed,
        alpha_balance=alpha_balance,
        construction_iterations=construction_iterations,
        initial_temp=initial_temp,
        cooling_rate=cooling_rate,
        iterations_per_temp=iterations_per_temp,
        min_temp=min_temp,
        max_neighbor_attempts=max_neighbor_attempts,
        neighborhood_weights=neighborhood_weights,
        **optimizer_kwargs,
    )