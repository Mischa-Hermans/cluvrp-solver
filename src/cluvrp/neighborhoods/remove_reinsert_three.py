"""Remove three promising clusters and greedily reinsert them."""

from __future__ import annotations

import random

from src.cluvrp.neighborhoods.helpers import cluster_misplacement_scores
from src.cluvrp.routing.route_builder import reoptimize_affected_superclusters
from src.cluvrp.types import GVRPInstance, Solution
from src.cluvrp.neighborhoods.remove_reinsert import greedy_reinsert_cluster


def neighborhood_remove_reinsert_three(
    instance: GVRPInstance,
    solution: Solution,
    rng: random.Random,
    node_dist,
    top_k_clusters: int = 10,
):
    scores = cluster_misplacement_scores(instance, solution)
    if len(scores) < 3:
        return None

    candidate_clusters = [r for _, r, _ in scores[: min(top_k_clusters, len(scores))]]
    if len(candidate_clusters) < 3:
        return None

    remove_triplet = rng.sample(candidate_clusters, 3)

    remaining_counts = {
        sc_id: len(sc)
        for sc_id, sc in enumerate(solution.superclusters)
    }

    for cluster in remove_triplet:
        src = solution.cluster_to_supercluster[cluster]
        remaining_counts[src] -= 1
        if remaining_counts[src] <= 0:
            return None

    current = solution.copy()
    affected = set()

    for cluster in remove_triplet:
        src = current.cluster_to_supercluster[cluster]
        current.superclusters[src].remove(cluster)
        affected.add(src)

    current = reoptimize_affected_superclusters(
        instance=instance,
        current_solution=solution,
        new_superclusters=current.superclusters,
        affected_ids=sorted(affected),
        node_dist=node_dist,
    )

    # Reinsert large-demand clusters first.
    ordered_triplet = sorted(
        remove_triplet,
        key=lambda r: instance.cluster_demands[r],
        reverse=True,
    )

    for cluster in ordered_triplet:
        candidate = greedy_reinsert_cluster(
            instance=instance,
            partial_solution=current,
            cluster=cluster,
            node_dist=node_dist,
        )
        if candidate is None:
            return None
        current = candidate

    current.last_move_type = "remove_reinsert_three"
    return current