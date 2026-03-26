"""Run ablation experiments for neighborhood operators."""

from __future__ import annotations

from copy import deepcopy
from multiprocessing import Pool, cpu_count

import pandas as pd
from scipy import stats

from src.cluvrp.experiments.run_single_instance import run_single_instance
from src.cluvrp.core.evaluation import compute_gap_percent


def build_operator_weight_sets(base_weights: dict) -> dict:
    weight_sets = {"all": deepcopy(base_weights)}

    for op in base_weights:
        weights = deepcopy(base_weights)
        weights[op] = 0.0
        weight_sets[f"without_{op}"] = weights

    return weight_sets


def compute_ablation_pvalues(run_df: pd.DataFrame) -> pd.DataFrame:
    base_df = run_df[run_df["setting"] == "all"].copy()
    pvalue_rows = []

    for setting_name in sorted(run_df["setting"].unique()):
        if setting_name == "all":
            continue

        compare_df = run_df[run_df["setting"] == setting_name].copy()

        merged = base_df.merge(
            compare_df,
            on=["instance", "seed"],
            suffixes=("_all", "_other"),
        )

        if merged.empty:
            continue

        diffs = merged["obj_val_other"] - merged["obj_val_all"]

        if len(diffs) >= 2:
            t_stat, p_value = stats.ttest_rel(
                merged["obj_val_other"],
                merged["obj_val_all"],
                nan_policy="omit",
            )
        else:
            t_stat, p_value = None, None

        pvalue_rows.append({
            "setting": setting_name,
            "mean_diff_obj_val": diffs.mean(),
            "median_diff_obj_val": diffs.median(),
            "t_statistic": t_stat,
            "p_value": p_value,
            "is_significant_5pct": None if p_value is None else p_value < 0.05,
            "is_significant_1pct": None if p_value is None else p_value < 0.01,
            "pairs": len(diffs),
        })

    pvalues_df = pd.DataFrame(pvalue_rows)

    if not pvalues_df.empty:
        pvalues_df = pvalues_df.sort_values("p_value", na_position="last").reset_index(drop=True)

    return pvalues_df


def _run_single_analysis_task(args):
    (
        setting_name,
        neighborhood_weights,
        instance_name,
        instance,
        seed,
        time_limit_seconds,
        best_known_soft,
        alpha_balance,
        construction_iterations,
        initial_temp,
        cooling_rate,
        iterations_per_temp,
        min_temp,
        max_neighbor_attempts,
    ) = args

    print(
        f"Running setting={setting_name}, instance={instance_name}, seed={seed} ...",
        flush=True,
    )

    result = run_single_instance(
        instance=instance,
        time_limit_seconds=time_limit_seconds,
        base_seed=seed,
        alpha_balance=alpha_balance,
        construction_iterations=construction_iterations,
        initial_temp=initial_temp,
        cooling_rate=cooling_rate,
        iterations_per_temp=iterations_per_temp,
        min_temp=min_temp,
        max_neighbor_attempts=max_neighbor_attempts,
        neighborhood_weights=neighborhood_weights,
        method="sa",
    )

    best_known = best_known_soft.get(instance_name)
    obj_val = result["best_solution"].total_cost
    gap_pct = None if best_known is None else compute_gap_percent(obj_val, best_known)

    run_row = {
        "setting": setting_name,
        "instance": instance_name,
        "seed": seed,
        "obj_val": round(obj_val, 3),
        "gap_pct": None if gap_pct is None else round(gap_pct, 3),
    }

    operator_rows = []
    for op_name, op_stats in result["operator_stats"].items():
        proposed = op_stats["proposed"]
        returned_candidate = op_stats["returned_candidate"]
        accepted = op_stats["accepted"]
        improving = op_stats["improving"]
        new_global_best = op_stats["new_global_best"]
        avg_accepted_delta = (
            op_stats["accepted_delta_sum"] / accepted if accepted > 0 else None
        )

        operator_rows.append({
            "setting": setting_name,
            "instance": instance_name,
            "seed": seed,
            "operator": op_name,
            "proposed": proposed,
            "returned_candidate": returned_candidate,
            "accepted": accepted,
            "improving": improving,
            "new_global_best": new_global_best,
            "candidate_rate": None if proposed == 0 else returned_candidate / proposed,
            "accept_rate": None if proposed == 0 else accepted / proposed,
            "improve_rate": None if proposed == 0 else improving / proposed,
            "avg_accepted_delta": avg_accepted_delta,
        })

    return run_row, operator_rows


def run_neighborhood_analysis(
    instances: dict,
    instance_names: list[str],
    seeds: list[int],
    time_limit_seconds: float,
    best_known_soft: dict,
    alpha_balance: float,
    construction_iterations: int,
    initial_temp: float,
    cooling_rate: float,
    iterations_per_temp: int,
    min_temp: float,
    max_neighbor_attempts: int,
    base_neighborhood_weights: dict,
    n_workers: int | None = None,
):
    weight_sets = build_operator_weight_sets(base_neighborhood_weights)

    tasks = []
    for setting_name, neighborhood_weights in weight_sets.items():
        for instance_name in instance_names:
            instance = instances[instance_name]
            for seed in seeds:
                tasks.append((
                    setting_name,
                    neighborhood_weights,
                    instance_name,
                    instance,
                    seed,
                    time_limit_seconds,
                    best_known_soft,
                    alpha_balance,
                    construction_iterations,
                    initial_temp,
                    cooling_rate,
                    iterations_per_temp,
                    min_temp,
                    max_neighbor_attempts,
                ))

    if n_workers is None:
        n_workers = max(1, cpu_count() - 1)

    with Pool(processes=n_workers) as pool:
        results = pool.map(_run_single_analysis_task, tasks)

    run_rows = []
    operator_rows = []

    for run_row, op_rows in results:
        run_rows.append(run_row)
        operator_rows.extend(op_rows)

    run_df = pd.DataFrame(run_rows)
    operator_df = pd.DataFrame(operator_rows)

    summary_df = (
        run_df.groupby("setting", as_index=False)
        .agg(
            mean_obj_val=("obj_val", "mean"),
            mean_gap_pct=("gap_pct", "mean"),
            std_gap_pct=("gap_pct", "std"),
            runs=("obj_val", "count"),
        )
        .sort_values("mean_gap_pct")
        .reset_index(drop=True)
    )

    operator_summary_df = (
        operator_df.groupby(["setting", "operator"], as_index=False)
        .agg(
            mean_proposed=("proposed", "mean"),
            mean_candidate_rate=("candidate_rate", "mean"),
            mean_accept_rate=("accept_rate", "mean"),
            mean_improve_rate=("improve_rate", "mean"),
            mean_new_global_best=("new_global_best", "mean"),
            mean_avg_accepted_delta=("avg_accepted_delta", "mean"),
        )
        .sort_values(["setting", "operator"])
        .reset_index(drop=True)
    )

    pvalues_df = compute_ablation_pvalues(run_df)

    return run_df, summary_df, operator_df, operator_summary_df, pvalues_df