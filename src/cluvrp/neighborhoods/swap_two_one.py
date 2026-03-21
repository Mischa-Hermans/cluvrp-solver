"""Swap two clusters from one vehicle with one from another."""

from __future__ import annotations

import random
from itertools import combinations

from src.cluvrp.neighborhoods.helpers import cluster_misplacement_scores, copy_superclusters
from src.cluvrp.routing.route_builder import reoptimize_affected_superclusters
from src.cluvrp.types import GVRPInstance, Solution


def neighborhood_swap_two_one(
    instance: GVRPInstance,
    solution: Solution,
    rng: random.Random,
    node_dist,
    top_k_clusters: int = 8,
):
    scores = cluster_misplacement_scores(instance, solution)
    if len(scores) < 2:
        return None

    candidate_clusters = scores[: min(top_k_clusters, len(scores))]
    rng.shuffle(candidate_clusters)

    best_candidate = None
    best_delta = float("inf")

    for _, ra, a in candidate_clusters:
        same_source = [r for _, r, src in candidate_clusters if src == a and r != ra]
        for rb in same_source:
            pair = tuple(sorted((ra, rb)))
            pair_demand = instance.cluster_demands[pair[0]] + instance.cluster_demands[pair[1]]

            for b in range(len(solution.superclusters)):
                if b == a:
                    continue

                for rc in solution.superclusters[b]:
                    new_load_a = solution.loads[a] - pair_demand + instance.cluster_demands[rc]
                    new_load_b = solution.loads[b] - instance.cluster_demands[rc] + pair_demand

                    if new_load_a > instance.capacity or new_load_b > instance.capacity:
                        continue

                    new_superclusters = copy_superclusters(solution.superclusters)
                    new_superclusters[a].remove(pair[0])
                    new_superclusters[a].remove(pair[1])
                    new_superclusters[b].remove(rc)

                    new_superclusters[a].append(rc)
                    new_superclusters[b].append(pair[0])
                    new_superclusters[b].append(pair[1])

                    candidate_solution = reoptimize_affected_superclusters(
                        instance=instance,
                        current_solution=solution,
                        new_superclusters=new_superclusters,
                        affected_ids=[a, b],
                        node_dist=node_dist,
                    )

                    delta = candidate_solution.total_cost - solution.total_cost
                    if delta < best_delta:
                        best_delta = delta
                        best_candidate = candidate_solution
                        best_candidate.last_move_type = "swap_two_one"

    return best_candidate