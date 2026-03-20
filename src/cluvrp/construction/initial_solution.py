"""Build the full initial solution from constructed superclusters."""

from __future__ import annotations

import random

from src.cluvrp.construction.superclusters import construct_superclusters
from src.cluvrp.routing.route_builder import build_solution_from_superclusters
from src.cluvrp.core.distances import build_node_distance_matrix
from src.cluvrp.types import GVRPInstance, Solution


def construct_initial_solution(
    instance: GVRPInstance,
    rng: random.Random,
    node_dist,
    alpha_balance: float,
) -> Solution:
    superclusters, _, _ = construct_superclusters(
        instance=instance,
        rng=rng,
        alpha_balance=alpha_balance,
    )
    return build_solution_from_superclusters(instance, superclusters, node_dist)


def construct_best_initial_solution(
    instance: GVRPInstance,
    base_seed: int,
    construction_iterations: int,
    alpha_balance: float,
) -> Solution:
    if construction_iterations < 1:
        raise ValueError("construction_iterations must be at least 1")

    rng_master = random.Random(base_seed)
    node_dist = build_node_distance_matrix(instance.coords)

    best_solution = None
    best_cost = float("inf")

    for _ in range(construction_iterations):
        seed = rng_master.randint(0, 10**9)
        rng = random.Random(seed)

        try:
            solution = construct_initial_solution(
                instance=instance,
                rng=rng,
                node_dist=node_dist,
                alpha_balance=alpha_balance,
            )
            solution.construction_seed = seed

            if solution.total_cost < best_cost:
                best_cost = solution.total_cost
                best_solution = solution
        except RuntimeError:
            pass

    if best_solution is None:
        raise RuntimeError("No feasible initial solution could be constructed.")

    return best_solution