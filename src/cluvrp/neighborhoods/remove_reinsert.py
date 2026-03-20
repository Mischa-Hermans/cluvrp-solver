"""Remove two clusters and greedily reinsert them."""

from __future__ import annotations

import random

from src.cluvrp.neighborhoods.helpers import cluster_misplacement_scores, copy_superclusters
from src.cluvrp.routing.route_builder import reoptimize_affected_superclusters
from src.cluvrp.types import GVRPInstance, Solution


def greedy_reinsert_cluster(
    instance: GVRPInstance,
    partial_solution: Solution,
    cluster: int,
    node_dist,
):
    best_candidate = None
    best_cost = float("inf")

    for dst in range(len(partial_solution.superclusters)):
        if partial_solution.loads[dst] + instance.cluster_demands[cluster] > instance.capacity:
            continue

        new_superclusters = copy_superclusters(partial_solution.superclusters)
        new_superclusters[dst].append(cluster)

        candidate_solution = reoptimize_affected_superclusters(
            instance=instance,
            current_solution=partial_solution,
            new_superclusters=new_superclusters,
            affected_ids=[dst],
            node_dist=node_dist,
        )

        if candidate_solution.total_cost < best_cost:
            best_cost = candidate_solution.total_cost
            best_candidate = candidate_solution

    return best_candidate


def neighborhood_remove_reinsert_two(
    instance: GVRPInstance,
    solution: Solution,
    rng: random.Random,
    node_dist,
    top_k_clusters: int = 8,
):
    scores = cluster_misplacement_scores(instance, solution)
    if len(scores) < 2:
        return None

    candidate_clusters = [r for _, r, _ in scores[: min(top_k_clusters, len(scores))]]
    if len(candidate_clusters) < 2:
        return None

    remove_pair = rng.sample(candidate_clusters, 2)
    current = solution.copy()
    affected = set()

    for cluster in remove_pair:
        src = current.cluster_to_supercluster[cluster]
        if len(current.superclusters[src]) <= 1:
            return None
        current.superclusters[src].remove(cluster)
        affected.add(src)

    current = reoptimize_affected_superclusters(
        instance=instance,
        current_solution=solution,
        new_superclusters=current.superclusters,
        affected_ids=sorted(affected),
        node_dist=node_dist,
    )

    for cluster in remove_pair:
        candidate = greedy_reinsert_cluster(
            instance=instance,
            partial_solution=current,
            cluster=cluster,
            node_dist=node_dist,
        )
        if candidate is None:
            return None
        current = candidate

    current.last_move_type = "remove_reinsert_two"
    return current