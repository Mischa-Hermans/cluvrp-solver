"""Simple TSP heuristics used when needed."""

from __future__ import annotations

import random
from typing import Dict, List

from src.cluvrp.core.evaluation import route_length


def nearest_neighbor_tour(
    customers: List[int],
    depot: int,
    dist: Dict[int, Dict[int, float]],
    rng: random.Random,
) -> List[int]:
    if not customers:
        return [depot, depot]

    unvisited = set(customers)
    ranked = sorted(unvisited, key=lambda i: dist[depot][i])
    start = rng.choice(ranked[: min(3, len(ranked))])

    route = [depot, start]
    unvisited.remove(start)
    current = start

    while unvisited:
        nearest = min(unvisited, key=lambda j: dist[current][j])
        route.append(nearest)
        unvisited.remove(nearest)
        current = nearest

    route.append(depot)
    return route


def two_opt(route: List[int], dist: Dict[int, Dict[int, float]]) -> List[int]:
    best = route[:]
    improved = True

    while improved:
        improved = False
        for i in range(1, len(best) - 2):
            for j in range(i + 1, len(best) - 1):
                if j - i == 1:
                    continue

                a = best[i - 1]
                b = best[i]
                c = best[j - 1]
                d = best[j]
                delta = dist[a][c] + dist[b][d] - dist[a][b] - dist[c][d]

                if delta < -1e-12:
                    best = best[:i] + best[i:j][::-1] + best[j:]
                    improved = True
                    break
            if improved:
                break

    return best