"""Move one promising cluster to another vehicle."""

from __future__ import annotations

import random

from src.cluvrp.neighborhoods.helpers import cluster_misplacement_scores, copy_superclusters
from src.cluvrp.routing.route_builder import reoptimize_affected_superclusters
from src.cluvrp.types import GVRPInstance, Solution


def neighborhood_relocate_best(
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

    for _, cluster, src in candidate_clusters:
        demand = instance.cluster_demands[cluster]

        for dst in range(len(solution.superclusters)):
            if dst == src:
                continue
            if solution.loads[dst] + demand > instance.capacity:
                continue

            new_superclusters = copy_superclusters(solution.superclusters)
            new_superclusters[src].remove(cluster)
            new_superclusters[dst].append(cluster)

            candidate_solution = reoptimize_affected_superclusters(
                instance=instance,
                current_solution=solution,
                new_superclusters=new_superclusters,
                affected_ids=[src, dst],
                node_dist=node_dist,
            )

            delta = candidate_solution.total_cost - solution.total_cost
            if delta < best_delta:
                best_delta = delta
                best_candidate = candidate_solution
                best_candidate.last_move_type = "relocate_best"

    return best_candidate