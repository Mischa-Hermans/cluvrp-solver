"""Turn benchmark outputs into a clean results table."""

from __future__ import annotations

import pandas as pd

from src.cluvrp.core.evaluation import compute_gap_percent, rename_result_columns


def build_results_table(
    all_runs: dict,
    instance_names: list[str],
    checkpoint_seconds: list[int],
    best_known_soft: dict,
    best_known_hard: dict,
):
    rows = []

    for name in instance_names:
        run = all_runs[name]
        record = {
            "instance": name,
            "best_known_hard": best_known_hard.get(name),
            "best_known_soft": best_known_soft.get(name),
        }

        for t in checkpoint_seconds:
            obj = round(run["checkpoint_costs"][t], 3)
            best = best_known_soft.get(name)
            gap = None if best is None else round(compute_gap_percent(obj, best), 3)
            record[f"{t}s_obj_val"] = obj
            record[f"{t}s_gap_pct"] = gap

        record["elapsed_time_s"] = round(run["stats"]["elapsed_time"], 3)
        rows.append(record)

    df = pd.DataFrame(rows)
    return rename_result_columns(df)