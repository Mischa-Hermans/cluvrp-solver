"""Iterated local search with cluster-level neighborhoods."""

from __future__ import annotations

import random
import time
from typing import Optional

from src.cluvrp.construction.initial_solution import construct_best_initial_solution
from src.cluvrp.core.distances import build_node_distance_matrix
from src.cluvrp.neighborhoods.relocate import neighborhood_relocate_best
from src.cluvrp.neighborhoods.swap import neighborhood_swap_restricted
from src.cluvrp.neighborhoods.remove_reinsert import neighborhood_remove_reinsert_two
from src.cluvrp.neighborhoods.ejection_chain import neighborhood_ejection_chain_light
from src.cluvrp.neighborhoods.pair_relocate import neighborhood_pair_relocate_best
from src.cluvrp.neighborhoods.swap_two_one import neighborhood_swap_two_one
from src.cluvrp.neighborhoods.remove_reinsert_three import neighborhood_remove_reinsert_three
from src.cluvrp.tracking.history import initialize_history, record_step
from src.cluvrp.types import GVRPInstance, RunStats


def apply_named_neighborhood(
    move_type: str,
    instance,
    solution,
    rng,
    node_dist,
):
    if move_type == "relocate_best":
        return neighborhood_relocate_best(instance, solution, rng, node_dist)
    if move_type == "swap_restricted":
        return neighborhood_swap_restricted(instance, solution, rng, node_dist)
    if move_type == "remove_reinsert_two":
        return neighborhood_remove_reinsert_two(instance, solution, rng, node_dist)
    if move_type == "ejection_chain_light":
        return neighborhood_ejection_chain_light(instance, solution, rng, node_dist)
    if move_type == "pair_relocate_best":
        return neighborhood_pair_relocate_best(instance, solution, rng, node_dist)
    if move_type == "swap_two_one":
        return neighborhood_swap_two_one(instance, solution, rng, node_dist)
    if move_type == "remove_reinsert_three":
        return neighborhood_remove_reinsert_three(instance, solution, rng, node_dist)

    raise ValueError(f"Unknown neighborhood: {move_type}")


def enabled_move_names(neighborhood_weights: dict) -> list[str]:
    return [name for name, weight in neighborhood_weights.items() if weight > 0]


def run_local_search(
    instance,
    current_solution,
    best_solution,
    rng,
    node_dist,
    neighborhood_weights,
    operator_stats,
    history,
    start_time,
    time_limit_seconds,
):
    active_moves = enabled_move_names(neighborhood_weights)

    iterations = 0
    accepted_moves = 0
    improving_moves = 0

    while time.perf_counter() - start_time < time_limit_seconds:
        move_order = active_moves[:]
        rng.shuffle(move_order)

        best_candidate = None
        best_move = None
        best_delta = 0.0

        for move_type in move_order:
            if time.perf_counter() - start_time >= time_limit_seconds:
                break

            operator_stats[move_type]["proposed"] += 1
            iterations += 1

            candidate_solution = apply_named_neighborhood(
                move_type=move_type,
                instance=instance,
                solution=current_solution,
                rng=rng,
                node_dist=node_dist,
            )

            if candidate_solution is None:
                continue

            operator_stats[move_type]["returned_candidate"] += 1
            delta = candidate_solution.total_cost - current_solution.total_cost

            if delta < best_delta - 1e-12:
                best_delta = delta
                best_candidate = candidate_solution
                best_move = move_type

        if best_candidate is None:
            break

        current_solution = best_candidate
        accepted_moves += 1
        improving_moves += 1

        operator_stats[best_move]["accepted"] += 1
        operator_stats[best_move]["improving"] += 1
        operator_stats[best_move]["accepted_delta_sum"] += best_delta

        if current_solution.total_cost < best_solution.total_cost - 1e-12:
            best_solution = current_solution.copy()
            operator_stats[best_move]["new_global_best"] += 1

        record_step(
            history=history,
            elapsed_time=time.perf_counter() - start_time,
            current_cost=current_solution.total_cost,
            best_cost=best_solution.total_cost,
            accepted=True,
            improving=True,
            move_type=best_move,
        )

    return current_solution, best_solution, iterations, accepted_moves, improving_moves


def apply_perturbation(
    instance,
    current_solution,
    best_solution,
    rng,
    node_dist,
    neighborhood_weights,
    operator_stats,
    history,
    start_time,
    time_limit_seconds,
    perturbation_steps: int,
):
    active_moves = enabled_move_names(neighborhood_weights)

    preferred_moves = [
        "remove_reinsert_three",
        "pair_relocate_best",
        "remove_reinsert_two",
        "swap_two_one",
        "relocate_best",
        "ejection_chain_light",
        "swap_restricted",
    ]
    perturbation_moves = [m for m in preferred_moves if m in active_moves]

    iterations = 0
    accepted_moves = 0
    improving_moves = 0

    for _ in range(perturbation_steps):
        if time.perf_counter() - start_time >= time_limit_seconds:
            break

        move_order = perturbation_moves[:]
        rng.shuffle(move_order)

        accepted_candidate = None
        accepted_move = None
        accepted_delta = None

        for move_type in move_order:
            if time.perf_counter() - start_time >= time_limit_seconds:
                break

            operator_stats[move_type]["proposed"] += 1
            iterations += 1

            candidate_solution = apply_named_neighborhood(
                move_type=move_type,
                instance=instance,
                solution=current_solution,
                rng=rng,
                node_dist=node_dist,
            )

            if candidate_solution is None:
                continue

            operator_stats[move_type]["returned_candidate"] += 1
            accepted_candidate = candidate_solution
            accepted_move = move_type
            accepted_delta = candidate_solution.total_cost - current_solution.total_cost
            break

        if accepted_candidate is None:
            break

        current_solution = accepted_candidate
        accepted_moves += 1

        operator_stats[accepted_move]["accepted"] += 1
        operator_stats[accepted_move]["accepted_delta_sum"] += accepted_delta

        improving = accepted_delta < -1e-12
        if improving:
            improving_moves += 1
            operator_stats[accepted_move]["improving"] += 1

        if current_solution.total_cost < best_solution.total_cost - 1e-12:
            best_solution = current_solution.copy()
            operator_stats[accepted_move]["new_global_best"] += 1

        record_step(
            history=history,
            elapsed_time=time.perf_counter() - start_time,
            current_cost=current_solution.total_cost,
            best_cost=best_solution.total_cost,
            accepted=True,
            improving=improving,
            move_type=accepted_move,
        )

    return current_solution, best_solution, iterations, accepted_moves, improving_moves


def optimize_with_iterated_local_search(
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
    perturbation_steps: int = 2,
):
    del initial_temp
    del cooling_rate
    del iterations_per_temp
    del min_temp
    del max_neighbor_attempts

    if neighborhood_weights is None:
        neighborhood_weights = {
            "relocate_best": 0.0,
            "swap_restricted": 0.0,
            "remove_reinsert_two": 1.0,
            "ejection_chain_light": 0.0,
            "pair_relocate_best": 1.0,
            "swap_two_one": 0.0,
            "remove_reinsert_three": 1.0,
        }

    start_time = time.perf_counter()
    rng = random.Random(base_seed)
    node_dist = build_node_distance_matrix(instance.coords)

    initial_solution = construct_best_initial_solution(
        instance=instance,
        base_seed=base_seed,
        construction_iterations=construction_iterations,
        alpha_balance=alpha_balance,
    )

    history = initialize_history(initial_solution.total_cost)

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

    current_solution = initial_solution.copy()
    best_solution = initial_solution.copy()

    total_iterations = 0
    total_accepted_moves = 0
    total_improving_moves = 0

    current_solution, best_solution, iters, acc, imp = run_local_search(
        instance=instance,
        current_solution=current_solution,
        best_solution=best_solution,
        rng=rng,
        node_dist=node_dist,
        neighborhood_weights=neighborhood_weights,
        operator_stats=operator_stats,
        history=history,
        start_time=start_time,
        time_limit_seconds=time_limit_seconds,
    )
    total_iterations += iters
    total_accepted_moves += acc
    total_improving_moves += imp

    while time.perf_counter() - start_time < time_limit_seconds:
        candidate_solution = current_solution.copy()

        candidate_solution, best_solution, iters, acc, imp = apply_perturbation(
            instance=instance,
            current_solution=candidate_solution,
            best_solution=best_solution,
            rng=rng,
            node_dist=node_dist,
            neighborhood_weights=neighborhood_weights,
            operator_stats=operator_stats,
            history=history,
            start_time=start_time,
            time_limit_seconds=time_limit_seconds,
            perturbation_steps=perturbation_steps,
        )
        total_iterations += iters
        total_accepted_moves += acc
        total_improving_moves += imp

        if time.perf_counter() - start_time >= time_limit_seconds:
            break

        candidate_solution, best_solution, iters, acc, imp = run_local_search(
            instance=instance,
            current_solution=candidate_solution,
            best_solution=best_solution,
            rng=rng,
            node_dist=node_dist,
            neighborhood_weights=neighborhood_weights,
            operator_stats=operator_stats,
            history=history,
            start_time=start_time,
            time_limit_seconds=time_limit_seconds,
        )
        total_iterations += iters
        total_accepted_moves += acc
        total_improving_moves += imp

        current_solution = candidate_solution

    elapsed = time.perf_counter() - start_time
    stats = RunStats(
        elapsed_time=elapsed,
        iterations=total_iterations,
        accepted_moves=total_accepted_moves,
        improving_moves=total_improving_moves,
        final_temperature=0.0,
    )

    return {
        "initial_solution": initial_solution,
        "best_solution": best_solution,
        "history": history,
        "stats": stats,
        "operator_stats": operator_stats,
    }