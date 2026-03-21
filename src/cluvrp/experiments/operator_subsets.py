"""Run experiments for different neighborhood subsets."""

from __future__ import annotations

from multiprocessing import Pool, cpu_count
import pandas as pd

from src.cluvrp.experiments.run_single_instance import run_single_instance
from src.cluvrp.core.evaluation import compute_gap_percent


def _run_single_subset_task(args):
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
        f"Running subset={setting_name}, instance={instance_name}, seed={seed} ...",
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
    for op_name, stats in result["operator_stats"].items():
        proposed = stats["proposed"]
        returned_candidate = stats["returned_candidate"]
        accepted = stats["accepted"]
        improving = stats["improving"]
        new_global_best = stats["new_global_best"]
        avg_accepted_delta = (
            stats["accepted_delta_sum"] / accepted if accepted > 0 else None
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


def run_neighborhood_subset_analysis(
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
    operator_sets: dict,
    n_workers: int | None = None,
):
    print("Preparing neighborhood subset analysis...", flush=True)

    tasks = []
    for setting_name, neighborhood_weights in operator_sets.items():
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

    print(f"Total tasks: {len(tasks)}", flush=True)

    if n_workers is None:
        n_workers = max(1, cpu_count() - 1)

    print(f"Using {n_workers} worker processes", flush=True)

    with Pool(processes=n_workers) as pool:
        results = pool.map(_run_single_subset_task, tasks)

    print("All subset tasks finished, building tables...", flush=True)

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

    return run_df, summary_df, operator_df, operator_summary_df