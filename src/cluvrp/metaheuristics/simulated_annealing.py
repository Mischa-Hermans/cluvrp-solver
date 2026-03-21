"""Simulated annealing with best-so-far tracking over time."""

from __future__ import annotations

import math
import random
import time
from typing import Optional

from src.cluvrp.construction.initial_solution import construct_best_initial_solution
from src.cluvrp.core.distances import build_node_distance_matrix
from src.cluvrp.core.utils import weighted_choice
from src.cluvrp.neighborhoods.relocate import neighborhood_relocate_best
from src.cluvrp.neighborhoods.swap import neighborhood_swap_restricted
from src.cluvrp.neighborhoods.remove_reinsert import neighborhood_remove_reinsert_two
from src.cluvrp.neighborhoods.ejection_chain import neighborhood_ejection_chain_light
from src.cluvrp.neighborhoods.pair_relocate import neighborhood_pair_relocate_best
from src.cluvrp.neighborhoods.swap_two_one import neighborhood_swap_two_one
from src.cluvrp.neighborhoods.remove_reinsert_three import neighborhood_remove_reinsert_three
from src.cluvrp.tracking.history import initialize_history, record_step
from src.cluvrp.types import GVRPInstance, RunStats


def propose_neighbor(
    instance,
    solution,
    rng,
    node_dist,
    neighborhood_weights,
):
    move_names = [k for k, v in neighborhood_weights.items() if v > 0]
    if not move_names:
        return None, None

    move_type = weighted_choice(move_names, neighborhood_weights, rng)

    if move_type == "relocate_best":
        return neighborhood_relocate_best(instance, solution, rng, node_dist), move_type
    if move_type == "swap_restricted":
        return neighborhood_swap_restricted(instance, solution, rng, node_dist), move_type
    if move_type == "remove_reinsert_two":
        return neighborhood_remove_reinsert_two(instance, solution, rng, node_dist), move_type
    if move_type == "ejection_chain_light":
        return neighborhood_ejection_chain_light(instance, solution, rng, node_dist), move_type
    if move_type == "pair_relocate_best":
        return neighborhood_pair_relocate_best(instance, solution, rng, node_dist), move_type
    if move_type == "swap_two_one":
        return neighborhood_swap_two_one(instance, solution, rng, node_dist), move_type
    if move_type == "remove_reinsert_three":
        return neighborhood_remove_reinsert_three(instance, solution, rng, node_dist), move_type

    raise ValueError(f"Unknown neighborhood: {move_type}")


def optimize_with_simulated_annealing(
    instance: GVRPInstance,
    time_limit_seconds: float,
    base_seed: int,
    alpha_balance: float,
    construction_iterations: int,
    initial_temp: float,
    cooling_rate: float,
    iterations_per_temp: int,
    min_temp: float,
    max_neighbor_attempts: int,
    neighborhood_weights: Optional[dict] = None,
):
    if neighborhood_weights is None:
        neighborhood_weights = {
            "relocate_best": 1.0,
            "swap_restricted": 1.0,
            "remove_reinsert_two": 1.0,
            "ejection_chain_light": 1.0,
            "pair_relocate_best": 1.0,
            "swap_two_one": 1.0,
            "remove_reinsert_three": 1.0,
        }

    del max_neighbor_attempts

    start_time = time.perf_counter()
    rng = random.Random(base_seed)
    node_dist = build_node_distance_matrix(instance.coords)

    initial_solution = construct_best_initial_solution(
        instance=instance,
        base_seed=base_seed,
        construction_iterations=construction_iterations,
        alpha_balance=alpha_balance,
    )

    current_solution = initial_solution.copy()
    best_solution = initial_solution.copy()

    history = initialize_history(best_solution.total_cost)
    temperature = initial_temp
    iterations = 0
    accepted_moves = 0
    improving_moves = 0

    operator_stats = {
        name: {
            "proposed": 0,
            "returned_candidate": 0,
            "accepted": 0,
            "improving": 0,
            "new_global_best": 0,
            "accepted_delta_sum": 0.0,
        }
        for name, weight in neighborhood_weights.items() if weight > 0
    }

    while time.perf_counter() - start_time < time_limit_seconds and temperature > min_temp:
        for _ in range(iterations_per_temp):
            elapsed = time.perf_counter() - start_time
            if elapsed >= time_limit_seconds:
                break

            candidate_solution, move_type = propose_neighbor(
                instance=instance,
                solution=current_solution,
                rng=rng,
                node_dist=node_dist,
                neighborhood_weights=neighborhood_weights,
            )

            iterations += 1

            if move_type is not None:
                operator_stats[move_type]["proposed"] += 1

            if candidate_solution is None:
                continue

            operator_stats[move_type]["returned_candidate"] += 1

            delta = candidate_solution.total_cost - current_solution.total_cost
            improving = delta < -1e-12

            accept = False
            if improving:
                accept = True
                improving_moves += 1
            else:
                prob = math.exp(-delta / max(temperature, 1e-12))
                if rng.random() < prob:
                    accept = True

            if accept:
                current_solution = candidate_solution
                accepted_moves += 1

                operator_stats[move_type]["accepted"] += 1
                operator_stats[move_type]["accepted_delta_sum"] += delta

                if improving:
                    operator_stats[move_type]["improving"] += 1

                if current_solution.total_cost < best_solution.total_cost - 1e-12:
                    best_solution = current_solution.copy()
                    operator_stats[move_type]["new_global_best"] += 1

            record_step(
                history=history,
                elapsed_time=time.perf_counter() - start_time,
                current_cost=current_solution.total_cost,
                best_cost=best_solution.total_cost,
                accepted=accept,
                improving=improving,
                move_type=None if candidate_solution is None else candidate_solution.last_move_type,
            )

        temperature *= cooling_rate

    elapsed = time.perf_counter() - start_time
    stats = RunStats(
        elapsed_time=elapsed,
        iterations=iterations,
        accepted_moves=accepted_moves,
        improving_moves=improving_moves,
        final_temperature=temperature,
    )

    return {
        "initial_solution": initial_solution,
        "best_solution": best_solution,
        "history": history,
        "stats": stats,
        "operator_stats": operator_stats,
    }