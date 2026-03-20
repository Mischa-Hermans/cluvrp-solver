"""Save and load result objects to disk."""

from __future__ import annotations

from pathlib import Path
import json
import pickle
import pandas as pd

from src.cluvrp.types import Solution, RunHistory, RunStats


def save_dataframe_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def save_run_json(path: Path, solution: Solution, history: RunHistory, stats: RunStats) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "best_solution": {
            "superclusters": solution.superclusters,
            "loads": solution.loads,
            "cluster_to_supercluster": solution.cluster_to_supercluster,
            "supercluster_customers": solution.supercluster_customers,
            "routes": solution.routes,
            "route_costs": solution.route_costs,
            "total_cost": solution.total_cost,
            "construction_seed": solution.construction_seed,
            "last_move_type": solution.last_move_type,
        },
        "history": [
            {
                "elapsed_time": r.elapsed_time,
                "current_cost": r.current_cost,
                "best_cost": r.best_cost,
                "accepted": r.accepted,
                "improving": r.improving,
                "move_type": r.move_type,
            }
            for r in history.records
        ],
        "stats": {
            "elapsed_time": stats.elapsed_time,
            "iterations": stats.iterations,
            "accepted_moves": stats.accepted_moves,
            "improving_moves": stats.improving_moves,
            "final_temperature": stats.final_temperature,
        },
    }
    path.write_text(json.dumps(payload, indent=2))


def save_pickle(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)