"""Construct capacity-feasible superclusters."""

from __future__ import annotations

import random
from typing import Dict, List, Tuple

from src.cluvrp.core.distances import compute_cluster_centroids, build_cluster_distance_matrix
from src.cluvrp.core.feasibility import check_instance_feasibility
from src.cluvrp.construction.seeds import choose_seeds
from src.cluvrp.types import GVRPInstance


def assignment_score(
    cluster_id: int,
    sc_clusters: List[int],
    sc_load: int,
    instance: GVRPInstance,
    cluster_dist: Dict[int, Dict[int, float]],
    alpha_balance: float,
) -> float:
    dist_term = min(cluster_dist[cluster_id][s] for s in sc_clusters)
    projected_load = sc_load + instance.cluster_demands[cluster_id]
    load_ratio = projected_load / instance.capacity
    return dist_term + alpha_balance * load_ratio * 100.0


def construct_superclusters(
    instance: GVRPInstance,
    rng: random.Random,
    alpha_balance: float,
) -> Tuple[List[List[int]], List[int], Dict[int, int]]:
    check_instance_feasibility(instance)

    centroids = compute_cluster_centroids(instance)
    cluster_dist = build_cluster_distance_matrix(centroids)
    seeds = choose_seeds(instance, centroids, cluster_dist, rng)

    superclusters = [[r] for r in seeds]
    loads = [instance.cluster_demands[r] for r in seeds]

    assigned = set(seeds)
    unassigned = set(instance.clusters) - assigned

    while unassigned:
        candidate_moves = []

        for r in unassigned:
            for sc_id in range(instance.vehicles):
                if loads[sc_id] + instance.cluster_demands[r] <= instance.capacity:
                    score = assignment_score(
                        cluster_id=r,
                        sc_clusters=superclusters[sc_id],
                        sc_load=loads[sc_id],
                        instance=instance,
                        cluster_dist=cluster_dist,
                        alpha_balance=alpha_balance,
                    )
                    candidate_moves.append((score, r, sc_id))

        if not candidate_moves:
            raise RuntimeError("Construction got stuck; try another seed.")

        candidate_moves.sort(key=lambda x: x[0])
        _, r, sc_id = rng.choice(candidate_moves[: min(4, len(candidate_moves))])

        superclusters[sc_id].append(r)
        loads[sc_id] += instance.cluster_demands[r]
        unassigned.remove(r)

    superclusters = [sorted(sc) for sc in superclusters]
    cluster_to_supercluster = {}
    for sc_id, sc in enumerate(superclusters):
        for r in sc:
            cluster_to_supercluster[r] = sc_id

    return superclusters, loads, cluster_to_supercluster