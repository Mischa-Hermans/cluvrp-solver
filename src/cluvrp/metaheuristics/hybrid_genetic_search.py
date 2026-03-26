"""Hybrid genetic search using ILS as local improvement."""

from __future__ import annotations

import random
import time
from typing import Optional

from configs.hgs import (
    HGS_POPULATION_SIZE,
    HGS_ELITE_SIZE,
    HGS_TOURNAMENT_SIZE,
    HGS_INITIAL_INDIVIDUAL_TIME_SECONDS,
    HGS_OFFSPRING_IMPROVEMENT_TIME_SECONDS,
    HGS_PARENT1_ROUTE_INHERIT_PROB,
)
from src.cluvrp.metaheuristics.iterated_local_search import optimize_with_iterated_local_search
from src.cluvrp.tracking.history import initialize_history, record_step
from src.cluvrp.types import GVRPInstance, RunStats


def assignment_signature(solution) -> tuple[int, ...]:
    """Encode the cluster-to-vehicle assignment."""
    return tuple(
        solution.cluster_to_supercluster[r]
        for r in sorted(solution.cluster_to_supercluster)
    )


def hamming_distance(sig_a: tuple[int, ...], sig_b: tuple[int, ...]) -> int:
    """Count cluster assignments that differ."""
    return sum(a != b for a, b in zip(sig_a, sig_b))


def diversity_score(solution, population) -> float:
    """Average assignment distance to the rest of the population."""
    if len(population) <= 1:
        return 0.0

    sig = assignment_signature(solution)
    distances = [
        hamming_distance(sig, assignment_signature(other))
        for other in population
        if other is not solution
    ]
    return float(sum(distances) / len(distances)) if distances else 0.0


def biased_fitness(solution, population) -> tuple[float, float]:
    """Lower is better; break objective ties using diversity."""
    return (solution.total_cost, -diversity_score(solution, population))


def tournament_selection(population, rng: random.Random, k: int):
    """Select one parent by tournament on biased fitness."""
    candidates = rng.sample(population, min(k, len(population)))
    return min(candidates, key=lambda s: biased_fitness(s, population))


def seed_individual_with_ils(
    instance,
    seed: int,
    alpha_balance: float,
    construction_iterations: int,
    neighborhood_weights: dict,
    perturbation_steps: int,
    time_budget: float,
):
    """Build one individual via a short ILS run."""
    result = optimize_with_iterated_local_search(
        instance=instance,
        time_limit_seconds=time_budget,
        base_seed=seed,
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
    return result["best_solution"].copy()


def crossover_superclusters(
    parent1,
    parent2,
    instance,
    rng: random.Random,
    parent1_route_inherit_prob: float,
):
    """
    Inherit a random subset of superclusters from parent1, then fill the rest from parent2.
    """
    n_vehicles = len(parent1.superclusters)
    child_superclusters = [[] for _ in range(n_vehicles)]

    all_clusters = list(instance.clusters.keys())
    assigned = set()

    parent1_route_ids = list(range(n_vehicles))
    rng.shuffle(parent1_route_ids)

    for sc_id in parent1_route_ids:
        if rng.random() <= parent1_route_inherit_prob:
            for r in parent1.superclusters[sc_id]:
                if r not in assigned:
                    child_superclusters[sc_id].append(r)
                    assigned.add(r)

    remaining = [r for r in all_clusters if r not in assigned]
    rng.shuffle(remaining)

    loads = [
        sum(instance.cluster_demands[r] for r in sc)
        for sc in child_superclusters
    ]

    for r in remaining:
        preferred_sc = parent2.cluster_to_supercluster[r]
        demand = instance.cluster_demands[r]

        feasible = [
            sc_id for sc_id in range(n_vehicles)
            if loads[sc_id] + demand <= instance.capacity
        ]

        if not feasible:
            return None

        if preferred_sc in feasible:
            chosen_sc = preferred_sc
        else:
            chosen_sc = min(feasible, key=lambda sc_id: loads[sc_id])

        child_superclusters[chosen_sc].append(r)
        loads[chosen_sc] += demand

    assigned_final = sorted(r for sc in child_superclusters for r in sc)
    if assigned_final != sorted(all_clusters):
        return None

    return [sorted(sc) for sc in child_superclusters]


def mutate_superclusters(instance, superclusters, rng: random.Random):
    """Small mutation before local improvement: move one random cluster if capacity allows."""
    superclusters = [sc[:] for sc in superclusters]
    n_vehicles = len(superclusters)
    loads = [
        sum(instance.cluster_demands[r] for r in sc)
        for sc in superclusters
    ]

    non_empty = [sc_id for sc_id in range(n_vehicles) if superclusters[sc_id]]
    if not non_empty:
        return superclusters

    src = rng.choice(non_empty)
    cluster = rng.choice(superclusters[src])
    demand = instance.cluster_demands[cluster]

    feasible_destinations = [
        dst for dst in range(n_vehicles)
        if dst != src and loads[dst] + demand <= instance.capacity
    ]
    if not feasible_destinations:
        return superclusters

    dst = rng.choice(feasible_destinations)
    superclusters[src].remove(cluster)
    superclusters[dst].append(cluster)

    return [sorted(sc) for sc in superclusters]


def survivor_selection(population, max_size: int):
    """Keep unique individuals first, then rank by objective and diversity."""
    unique = []
    seen = set()

    for sol in sorted(population, key=lambda s: s.total_cost):
        sig = assignment_signature(sol)
        if sig not in seen:
            unique.append(sol)
            seen.add(sig)

    ranked = sorted(unique, key=lambda s: biased_fitness(s, unique))
    return ranked[:max_size]


def optimize_with_hybrid_genetic_search(
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
    population_size: int = HGS_POPULATION_SIZE,
    elite_size: int = HGS_ELITE_SIZE,
    tournament_size: int = HGS_TOURNAMENT_SIZE,
    initial_individual_time_seconds: float = HGS_INITIAL_INDIVIDUAL_TIME_SECONDS,
    offspring_improvement_time_seconds: float = HGS_OFFSPRING_IMPROVEMENT_TIME_SECONDS,
    parent1_route_inherit_prob: float = HGS_PARENT1_ROUTE_INHERIT_PROB,
):
    # Shared interface with SA / ILS; unused here.
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

    population_size = max(2, int(population_size))
    elite_size = max(1, min(int(elite_size), population_size - 1))
    tournament_size = max(2, int(tournament_size))
    initial_individual_time_seconds = max(0.1, float(initial_individual_time_seconds))
    offspring_improvement_time_seconds = max(0.1, float(offspring_improvement_time_seconds))
    parent1_route_inherit_prob = min(1.0, max(0.0, float(parent1_route_inherit_prob)))

    start_time = time.perf_counter()
    rng = random.Random(base_seed)

    history = None
    population = []

    # ---- Initial population ----
    for i in range(population_size):
        if time.perf_counter() - start_time >= time_limit_seconds:
            break

        try:
            individual = seed_individual_with_ils(
                instance=instance,
                seed=base_seed + i,
                alpha_balance=alpha_balance,
                construction_iterations=construction_iterations,
                neighborhood_weights=neighborhood_weights,
                perturbation_steps=perturbation_steps,
                time_budget=initial_individual_time_seconds,
            )
            population.append(individual)
        except RuntimeError:
            continue

    if not population:
        raise RuntimeError("HGS could not build an initial population.")

    population = survivor_selection(population, population_size)

    best_solution = min(population, key=lambda s: s.total_cost).copy()
    initial_solution = best_solution.copy()
    history = initialize_history(best_solution.total_cost)

    generations = 0
    offspring_created = 0
    offspring_improved_best = 0

    # ---- Evolution loop ----
    while time.perf_counter() - start_time < time_limit_seconds:
        new_population = population[: min(elite_size, len(population))]

        while (
            len(new_population) < population_size
            and time.perf_counter() - start_time < time_limit_seconds
        ):
            parent1 = tournament_selection(population, rng, tournament_size)
            parent2 = tournament_selection(population, rng, tournament_size)

            offspring_superclusters = crossover_superclusters(
                parent1=parent1,
                parent2=parent2,
                instance=instance,
                rng=rng,
                parent1_route_inherit_prob=parent1_route_inherit_prob,
            )
            if offspring_superclusters is None:
                continue

            offspring_superclusters = mutate_superclusters(instance, offspring_superclusters, rng)

            try:
                offspring = seed_individual_with_ils(
                    instance=instance,
                    seed=rng.randint(0, 1_000_000),
                    alpha_balance=alpha_balance,
                    construction_iterations=construction_iterations,
                    neighborhood_weights=neighborhood_weights,
                    perturbation_steps=perturbation_steps,
                    time_budget=offspring_improvement_time_seconds,
                )
            except RuntimeError:
                continue

            new_population.append(offspring)
            offspring_created += 1

            improving = offspring.total_cost < best_solution.total_cost - 1e-12
            if improving:
                best_solution = offspring.copy()
                offspring_improved_best += 1

            record_step(
                history=history,
                elapsed_time=time.perf_counter() - start_time,
                current_cost=offspring.total_cost,
                best_cost=best_solution.total_cost,
                accepted=True,
                improving=improving,
                move_type="hgs_offspring",
            )

        population.extend(new_population)
        population = survivor_selection(population, population_size)
        generations += 1

    elapsed = time.perf_counter() - start_time
    stats = RunStats(
        elapsed_time=elapsed,
        iterations=generations,
        accepted_moves=offspring_created,
        improving_moves=offspring_improved_best,
        final_temperature=0.0,
    )

    return {
        "initial_solution": initial_solution,
        "best_solution": best_solution,
        "history": history,
        "stats": stats,
        "operator_stats": {},
    }