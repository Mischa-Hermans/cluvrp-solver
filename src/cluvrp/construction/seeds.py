"""Choose starting seed clusters for the vehicles."""

from __future__ import annotations

import random
from typing import Dict, List, Tuple

from src.cluvrp.core.distances import euclidean
from src.cluvrp.types import GVRPInstance


def choose_seeds(
    instance: GVRPInstance,
    cluster_centroids: Dict[int, Tuple[float, float]],
    cluster_dist: Dict[int, Dict[int, float]],
    rng: random.Random,
) -> List[int]:
    all_clusters = list(instance.clusters.keys())
    depot = instance.coords[instance.depot]

    farthest = sorted(
        all_clusters,
        key=lambda r: euclidean(cluster_centroids[r], depot),
        reverse=True,
    )
    seeds = [rng.choice(farthest[: min(3, len(farthest))])]

    while len(seeds) < instance.vehicles:
        candidates = [r for r in all_clusters if r not in seeds]
        ranked = sorted(
            candidates,
            key=lambda r: min(cluster_dist[r][s] for s in seeds),
            reverse=True,
        )
        seeds.append(rng.choice(ranked[: min(3, len(ranked))]))

    return seeds