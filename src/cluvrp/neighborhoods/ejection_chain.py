"""Small two-step ejection chain move."""

from __future__ import annotations

import random

from src.cluvrp.neighborhoods.helpers import cluster_misplacement_scores, copy_superclusters
from src.cluvrp.routing.route_builder import reoptimize_affected_superclusters
from src.cluvrp.types import GVRPInstance, Solution


def neighborhood_ejection_chain_light(
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
    m = len(solution.superclusters)

    for _, ra, a in candidate_clusters:
        demand_ra = instance.cluster_demands[ra]

        for b in range(m):
            if b == a:
                continue
            if solution.loads[b] + demand_ra > instance.capacity:
                continue

            temp_superclusters = copy_superclusters(solution.superclusters)
            temp_superclusters[a].remove(ra)
            temp_superclusters[b].append(ra)

            temp_solution = reoptimize_affected_superclusters(
                instance=instance,
                current_solution=solution,
                new_superclusters=temp_superclusters,
                affected_ids=[a, b],
                node_dist=node_dist,
            )

            for rb in temp_solution.superclusters[b]:
                if rb == ra:
                    continue
                demand_rb = instance.cluster_demands[rb]

                for c in range(m):
                    if c == b:
                        continue
                    if temp_solution.loads[c] + demand_rb > instance.capacity:
                        continue

                    new_superclusters = copy_superclusters(temp_solution.superclusters)
                    new_superclusters[b].remove(rb)
                    new_superclusters[c].append(rb)

                    candidate_solution = reoptimize_affected_superclusters(
                        instance=instance,
                        current_solution=temp_solution,
                        new_superclusters=new_superclusters,
                        affected_ids=sorted(set([b, c])),
                        node_dist=node_dist,
                    )

                    delta = candidate_solution.total_cost - solution.total_cost
                    if delta < best_delta:
                        best_delta = delta
                        best_candidate = candidate_solution
                        best_candidate.last_move_type = "ejection_chain_light"

    return best_candidate