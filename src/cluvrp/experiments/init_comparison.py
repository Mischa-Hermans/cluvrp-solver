"""Compare initialization strategies under simulated annealing."""

from __future__ import annotations

from multiprocessing import Pool, cpu_count

import pandas as pd
from scipy import stats

from src.cluvrp.core.evaluation import compute_gap_percent
from src.cluvrp.experiments.run_single_instance import run_single_instance


def _run_single_init_task(args):
    (
        init_method,
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
        neighborhood_weights,
    ) = args

    print(
        f"Running init={init_method}, instance={instance_name}, seed={seed} ...",
        flush=True,
    )

    try:
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
            optimizer_kwargs={"init_mode": init_method},
        )
    except RuntimeError as e:
        print(
            f"FAILED init={init_method}, instance={instance_name}, seed={seed}: {e}",
            flush=True,
        )
        return None

    best_known = best_known_soft.get(instance_name)

    initial_obj = result["initial_solution"].total_cost
    final_obj = result["best_solution"].total_cost

    initial_gap = None if best_known is None else compute_gap_percent(initial_obj, best_known)
    final_gap = None if best_known is None else compute_gap_percent(final_obj, best_known)

    return {
        "initialization": init_method,
        "instance": instance_name,
        "seed": seed,
        "initial_obj_val": round(initial_obj, 3),
        "initial_gap_pct": None if initial_gap is None else round(initial_gap, 3),
        "final_obj_val": round(final_obj, 3),
        "final_gap_pct": None if final_gap is None else round(final_gap, 3),
    }


def compute_pvalues(run_df: pd.DataFrame) -> pd.DataFrame:
    proposed = run_df[run_df["initialization"] == "proposed"].copy()
    rows = []

    for other_name in ["random", "greedy"]:
        other = run_df[run_df["initialization"] == other_name].copy()

        merged = proposed.merge(
            other,
            on=["instance", "seed"],
            suffixes=("_proposed", "_other"),
        )

        if merged.empty:
            continue

        diffs = merged["final_gap_pct_other"] - merged["final_gap_pct_proposed"]

        if len(diffs) >= 2:
            t_stat, p_value = stats.ttest_rel(
                merged["final_gap_pct_other"],
                merged["final_gap_pct_proposed"],
                nan_policy="omit",
            )
        else:
            t_stat, p_value = None, None

        rows.append({
            "comparison": f"proposed_vs_{other_name}",
            "mean_diff_final_gap": diffs.mean(),
            "t_statistic": t_stat,
            "p_value": p_value,
            "pairs": len(diffs),
        })

    return pd.DataFrame(rows)


def run_init_comparison(
    instances: dict,
    instance_names: list[str],
    seeds: list[int],
    init_methods: list[str],
    time_limit_seconds: float,
    best_known_soft: dict,
    alpha_balance: float,
    construction_iterations: int,
    initial_temp: float,
    cooling_rate: float,
    iterations_per_temp: int,
    min_temp: float,
    max_neighbor_attempts: int,
    neighborhood_weights: dict,
    n_workers: int | None = None,
):
    tasks = []
    for init_method in init_methods:
        for instance_name in instance_names:
            instance = instances[instance_name]
            for seed in seeds:
                tasks.append((
                    init_method,
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
                    neighborhood_weights,
                ))

    if n_workers is None:
        n_workers = max(1, cpu_count() - 1)

    with Pool(processes=n_workers) as pool:
        rows = pool.map(_run_single_init_task, tasks)

    # Drop failed runs
    rows = [row for row in rows if row is not None]

    if not rows:
        raise RuntimeError("All initialization comparison runs failed.")

    run_df = pd.DataFrame(rows)

    summary_df = (
        run_df.groupby("initialization", as_index=False)
        .agg(
            initial_mean_gap=("initial_gap_pct", "mean"),
            initial_std_gap=("initial_gap_pct", "std"),
            final_mean_gap=("final_gap_pct", "mean"),
            final_std_gap=("final_gap_pct", "std"),
            final_min_gap=("final_gap_pct", "min"),
            final_max_gap=("final_gap_pct", "max"),
            runs=("final_gap_pct", "count"),
        )
        .reset_index(drop=True)
    )

    order = pd.Categorical(
        summary_df["initialization"],
        categories=["random", "greedy", "proposed"],
        ordered=True,
    )
    summary_df["initialization"] = order
    summary_df = summary_df.sort_values("initialization").reset_index(drop=True)
    summary_df["initialization"] = summary_df["initialization"].astype(str)

    pvalues_df = compute_pvalues(run_df)

    return run_df, summary_df, pvalues_df