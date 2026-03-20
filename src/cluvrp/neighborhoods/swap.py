"""Swap one promising cluster with a cluster from another vehicle."""

from __future__ import annotations

import random

from src.cluvrp.neighborhoods.helpers import cluster_misplacement_scores, copy_superclusters
from src.cluvrp.routing.route_builder import reoptimize_affected_superclusters
from src.cluvrp.types import GVRPInstance, Solution


def neighborhood_swap_restricted(
    instance: GVRPInstance,
    solution: Solution,
    rng: random.Random,
    node_dist,
    top_k_clusters: int = 6,
):
    scores = cluster_misplacement_scores(instance, solution)
    if not scores:
        return None

    candidate_clusters = scores[: min(top_k_clusters, len(scores))]
    rng.shuffle(candidate_clusters)

    best_candidate = None
    best_delta = float("inf")

    for _, ra, a in candidate_clusters:
        for b in range(len(solution.superclusters)):
            if b == a:
                continue

            for rb in solution.superclusters[b]:
                load_a = solution.loads[a] - instance.cluster_demands[ra] + instance.cluster_demands[rb]
                load_b = solution.loads[b] - instance.cluster_demands[rb] + instance.cluster_demands[ra]

                if load_a > instance.capacity or load_b > instance.capacity:
                    continue

                new_superclusters = copy_superclusters(solution.superclusters)
                new_superclusters[a].remove(ra)
                new_superclusters[b].remove(rb)
                new_superclusters[a].append(rb)
                new_superclusters[b].append(ra)

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
                    best_candidate.last_move_type = "swap_restricted"

    return best_candidate