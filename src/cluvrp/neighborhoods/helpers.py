"""Shared helpers for neighborhood evaluation."""

from __future__ import annotations

from typing import List, Tuple

from src.cluvrp.core.distances import compute_cluster_centroids, compute_supercluster_centroids, euclidean
from src.cluvrp.types import GVRPInstance, Solution


def copy_superclusters(superclusters: List[List[int]]) -> List[List[int]]:
    return [sc[:] for sc in superclusters]


def cluster_misplacement_scores(
    instance: GVRPInstance,
    solution: Solution,
) -> List[Tuple[float, int, int]]:
    cluster_centroids = compute_cluster_centroids(instance)
    sc_centroids = compute_supercluster_centroids(instance, solution.superclusters)

    scores = []
    for sc_id, sc in enumerate(solution.superclusters):
        if len(sc) <= 1:
            continue

        current_centroid = sc_centroids[sc_id]

        for r in sc:
            r_centroid = cluster_centroids[r]
            current_dist = euclidean(r_centroid, current_centroid)

            best_other_dist = float("inf")
            for other_sc in range(len(solution.superclusters)):
                if other_sc == sc_id:
                    continue
                d = euclidean(r_centroid, sc_centroids[other_sc])
                if d < best_other_dist:
                    best_other_dist = d

            scores.append((current_dist - best_other_dist, r, sc_id))

    scores.sort(reverse=True)
    return scores


def nearest_partners_within_supercluster(
    instance: GVRPInstance,
    solution: Solution,
    anchor_cluster: int,
    src_sc: int,
    max_partners: int = 4,
) -> List[int]:
    cluster_centroids = compute_cluster_centroids(instance)
    anchor_xy = cluster_centroids[anchor_cluster]

    candidates = [
        r for r in solution.superclusters[src_sc]
        if r != anchor_cluster
    ]

    ranked = sorted(
        candidates,
        key=lambda r: euclidean(anchor_xy, cluster_centroids[r]),
    )

    return ranked[: min(max_partners, len(ranked))]