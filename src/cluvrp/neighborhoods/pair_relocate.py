"""Move a nearby pair of clusters together to another vehicle."""

from __future__ import annotations

import random

from src.cluvrp.neighborhoods.helpers import (
    cluster_misplacement_scores,
    copy_superclusters,
    nearest_partners_within_supercluster,
)
from src.cluvrp.routing.route_builder import reoptimize_affected_superclusters
from src.cluvrp.types import GVRPInstance, Solution


def neighborhood_pair_relocate_best(
    instance: GVRPInstance,
    solution: Solution,
    rng: random.Random,
    node_dist,
    top_k_clusters: int = 8,
    max_partners: int = 4,
):
    scores = cluster_misplacement_scores(instance, solution)
    if not scores:
        return None

    candidate_clusters = scores[: min(top_k_clusters, len(scores))]
    rng.shuffle(candidate_clusters)

    best_candidate = None
    best_delta = float("inf")

    for _, ra, src in candidate_clusters:
        if len(solution.superclusters[src]) <= 2:
            continue

        partner_candidates = nearest_partners_within_supercluster(
            instance=instance,
            solution=solution,
            anchor_cluster=ra,
            src_sc=src,
            max_partners=max_partners,
        )

        for rb in partner_candidates:
            demand_pair = instance.cluster_demands[ra] + instance.cluster_demands[rb]

            for dst in range(len(solution.superclusters)):
                if dst == src:
                    continue
                if solution.loads[dst] + demand_pair > instance.capacity:
                    continue

                new_superclusters = copy_superclusters(solution.superclusters)
                new_superclusters[src].remove(ra)
                new_superclusters[src].remove(rb)
                new_superclusters[dst].append(ra)
                new_superclusters[dst].append(rb)

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
                    best_candidate.last_move_type = "pair_relocate_best"

    return best_candidate