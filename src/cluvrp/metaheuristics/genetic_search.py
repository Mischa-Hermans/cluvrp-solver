"""Simple hybrid genetic search using ILS as local improvement."""

from __future__ import annotations

import random
import time
from typing import Optional

from src.cluvrp.core.distances import build_node_distance_matrix
from src.cluvrp.metaheuristics.iterated_local_search import optimize_with_iterated_local_search
from src.cluvrp.routing.route_builder import build_solution_from_superclusters
from src.cluvrp.tracking.history import initialize_history, record_step
from src.cluvrp.types import GVRPInstance, RunStats
from configs.ga import (
    GA_POPULATION_SIZE,
    GA_ELITE_SIZE,
    GA_TOURNAMENT_SIZE,
    GA_CHILD_IMPROVEMENT_TIME_SECONDS,
    GA_INITIAL_SOLUTION_TIME_SECONDS,
)


def tournament_selection(population, rng: random.Random, k: int):
    """Select the best among k sampled individuals."""
    candidates = rng.sample(population, min(k, len(population)))
    return min(candidates, key=lambda s: s.total_cost)


def crossover(parent1, parent2, instance, rng: random.Random):
    """Assign each cluster to the vehicle chosen from one of the parents."""
    n_vehicles = len(parent1.superclusters)
    child_superclusters = [[] for _ in range(n_vehicles)]

    for cluster in instance.clusters.keys():
        if rng.random() < 0.5:
            sc_id = parent1.cluster_to_supercluster[cluster]
        else:
            sc_id = parent2.cluster_to_supercluster[cluster]
        child_superclusters[sc_id].append(cluster)

    return child_superclusters


def repair_superclusters(instance, child_superclusters, rng: random.Random):
    """Repair infeasible crossover output by removing overload and reinserting greedily."""
    n_vehicles = len(child_superclusters)
    demands = instance.cluster_demands
    capacity = instance.capacity

    superclusters = [sc[:] for sc in child_superclusters]
    loads = [sum(demands[r] for r in sc) for sc in superclusters]

    unassigned = []

    for sc_id in range(n_vehicles):
        while loads[sc_id] > capacity and superclusters[sc_id]:
            cluster = rng.choice(superclusters[sc_id])
            superclusters[sc_id].remove(cluster)
            loads[sc_id] -= demands[cluster]
            unassigned.append(cluster)

    for cluster in sorted(unassigned, key=lambda r: demands[r], reverse=True):
        feasible_destinations = [
            sc_id for sc_id in range(n_vehicles)
            if loads[sc_id] + demands[cluster] <= capacity
        ]
        if not feasible_destinations:
            return None

        best_sc = min(feasible_destinations, key=lambda sc_id: loads[sc_id])
        superclusters[best_sc].append(cluster)
        loads[best_sc] += demands[cluster]

    assigned = [r for sc in superclusters for r in sc]
    if sorted(assigned) != sorted(instance.clusters.keys()):
        return None

    return [sorted(sc) for sc in superclusters]


def improve_child_with_ils(
    instance,
    child_superclusters,
    node_dist,
    rng_seed: int,
    alpha_balance: float,
    construction_iterations: int,
    neighborhood_weights: dict,
    perturbation_steps: int,
):
    """Evaluate repaired child and improve it briefly with ILS."""
    child_solution = build_solution_from_superclusters(
        instance=instance,
        superclusters=child_superclusters,
        node_dist=node_dist,
    )

    ils_result = optimize_with_iterated_local_search(
        instance=instance,
        time_limit_seconds=GA_CHILD_IMPROVEMENT_TIME_SECONDS,
        base_seed=rng_seed,
        alpha_balance=alpha_balance,
        construction_iterations=construction_iterations,
        initial_temp=0.0,
        cooling_rate=0.0,
        iterations_per_temp=0,
        min_temp=0.0,
        max_neighbor_attempts=0,
        neighborhood_weights=neighborhood_weights,
        perturbation_steps=perturbation_steps,
    )

    ils_solution = ils_result["best_solution"]

    if child_solution.total_cost < ils_solution.total_cost:
        return child_solution
    return ils_solution


def optimize_with_genetic_search(
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

    rng = random.Random(base_seed)
    start_time = time.perf_counter()
    node_dist = build_node_distance_matrix(instance.coords)

    population = []
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

    best_solution = None
    history = None

    # Initial population from repeated short ILS runs.
    for i in range(GA_POPULATION_SIZE):
        if time.perf_counter() - start_time >= time_limit_seconds:
            break

        result = optimize_with_iterated_local_search(
            instance=instance,
            time_limit_seconds=min(GA_INITIAL_SOLUTION_TIME_SECONDS, time_limit_seconds),
            base_seed=base_seed + i,
            alpha_balance=alpha_balance,
            construction_iterations=construction_iterations,
            initial_temp=0.0,
            cooling_rate=0.0,
            iterations_per_temp=0,
            min_temp=0.0,
            max_neighbor_attempts=0,
            neighborhood_weights=neighborhood_weights,
            perturbation_steps=perturbation_steps,
        )

        sol = result["best_solution"].copy()
        population.append(sol)

        if best_solution is None or sol.total_cost < best_solution.total_cost:
            best_solution = sol.copy()

    if not population or best_solution is None:
        raise RuntimeError("Genetic search could not build an initial population.")

    initial_solution = best_solution.copy()
    history = initialize_history(best_solution.total_cost)

    generations = 0

    while time.perf_counter() - start_time < time_limit_seconds:
        population.sort(key=lambda s: s.total_cost)
        new_population = [sol.copy() for sol in population[: min(GA_ELITE_SIZE, len(population))]]

        while (
            len(new_population) < GA_POPULATION_SIZE
            and time.perf_counter() - start_time < time_limit_seconds
        ):
            parent1 = tournament_selection(population, rng, GA_TOURNAMENT_SIZE)
            parent2 = tournament_selection(population, rng, GA_TOURNAMENT_SIZE)

            child_superclusters = crossover(parent1, parent2, instance, rng)
            child_superclusters = repair_superclusters(instance, child_superclusters, rng)

            if child_superclusters is None:
                continue

            child = improve_child_with_ils(
                instance=instance,
                child_superclusters=child_superclusters,
                node_dist=node_dist,
                rng_seed=rng.randint(0, 1_000_000),
                alpha_balance=alpha_balance,
                construction_iterations=construction_iterations,
                neighborhood_weights=neighborhood_weights,
                perturbation_steps=perturbation_steps,
            )

            new_population.append(child)

            improving = child.total_cost < best_solution.total_cost - 1e-12
            if improving:
                best_solution = child.copy()

            record_step(
                history=history,
                elapsed_time=time.perf_counter() - start_time,
                current_cost=child.total_cost,
                best_cost=best_solution.total_cost,
                accepted=True,
                improving=improving,
                move_type="ga_child",
            )

        population = new_population
        generations += 1

    elapsed = time.perf_counter() - start_time
    stats = RunStats(
        elapsed_time=elapsed,
        iterations=generations,
        accepted_moves=0,
        improving_moves=0,
        final_temperature=0.0,
    )

    return {
        "initial_solution": initial_solution,
        "best_solution": best_solution,
        "history": history,
        "stats": stats,
        "operator_stats": operator_stats,
    }