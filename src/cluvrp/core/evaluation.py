"""Cost calculations and benchmark summaries."""

from __future__ import annotations

from typing import Dict, List, Optional

from src.cluvrp.types import Solution


def route_length(route: List[int], dist: Dict[int, Dict[int, float]]) -> float:
    return sum(dist[route[i]][route[i + 1]] for i in range(len(route) - 1))


def compute_gap_percent(obj_val: float, best_known: float) -> float:
    return 100.0 * (obj_val - best_known) / best_known


def summarize_run(instance_name: str, solution: Solution, best_known_soft: Dict[str, float]) -> dict:
    obj = solution.total_cost
    best = best_known_soft.get(instance_name)
    gap_pct = None if best is None else round(compute_gap_percent(obj, best), 3)

    return {
        "obj_val": round(obj, 3),
        "gap_pct": gap_pct,
    }


def rename_result_columns(df):
    return df.rename(columns={
        "best_known_hard": "best known hard",
        "best_known_soft": "best known soft",
        "1s_obj_val": "within 1s obj. val.",
        "1s_gap_pct": "within 1s % gap",
        "10s_obj_val": "within 10s obj. val.",
        "10s_gap_pct": "within 10s % gap",
        "60s_obj_val": "within 60s obj. val.",
        "60s_gap_pct": "within 60s % gap",
        "300s_obj_val": "within 300s obj. val.",
        "300s_gap_pct": "within 300s % gap",
        "elapsed_time_s": "run time (s)",
    })