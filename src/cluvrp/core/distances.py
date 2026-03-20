"""Distance calculations for nodes and clusters."""

from __future__ import annotations

import math
from typing import Dict, Tuple

from src.cluvrp.types import GVRPInstance


def euclidean(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def build_node_distance_matrix(coords: Dict[int, Tuple[float, float]]) -> Dict[int, Dict[int, float]]:
    return {i: {j: euclidean(coords[i], coords[j]) for j in coords} for i in coords}


def compute_cluster_centroids(instance: GVRPInstance) -> Dict[int, Tuple[float, float]]:
    centroids = {}
    for r, customers in instance.clusters.items():
        xs = [instance.coords[i][0] for i in customers]
        ys = [instance.coords[i][1] for i in customers]
        centroids[r] = (sum(xs) / len(xs), sum(ys) / len(ys))
    return centroids


def build_cluster_distance_matrix(centroids: Dict[int, Tuple[float, float]]) -> Dict[int, Dict[int, float]]:
    return {r1: {r2: euclidean(centroids[r1], centroids[r2]) for r2 in centroids} for r1 in centroids}


def compute_supercluster_centroids(
    instance: GVRPInstance,
    superclusters: list[list[int]],
) -> Dict[int, Tuple[float, float]]:
    cluster_centroids = compute_cluster_centroids(instance)
    depot_xy = instance.coords[instance.depot]
    sc_centroids = {}

    for sc_id, sc in enumerate(superclusters):
        if not sc:
            sc_centroids[sc_id] = depot_xy
            continue
        xs = [cluster_centroids[r][0] for r in sc]
        ys = [cluster_centroids[r][1] for r in sc]
        sc_centroids[sc_id] = (sum(xs) / len(xs), sum(ys) / len(ys))

    return sc_centroids