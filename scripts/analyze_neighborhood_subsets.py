"""Run neighborhood subset analysis and save the results."""

from configs.default import (
    INSTANCE_DIRS,
    BEST_KNOWN_SOFT,
    TABLES_DIR,
)
from configs.sa import (
    ALPHA_BALANCE,
    CONSTRUCTION_ITERATIONS,
    SA_INITIAL_TEMP,
    SA_COOLING_RATE,
    SA_ITERATIONS_PER_TEMP,
    SA_MIN_TEMP,
    SA_MAX_NEIGHBOR_ATTEMPTS,
)
from configs.operator_subsets import (
    ANALYSIS_INSTANCE_NAMES,
    ANALYSIS_SEEDS,
    ANALYSIS_TIME_LIMIT_SECONDS,
    OPERATOR_SETS,
)
from src.cluvrp.io.instance_reader import get_instance_path, read_gvrp_instance
from src.cluvrp.io.result_io import save_dataframe_csv
from src.cluvrp.experiments.operator_subsets import run_neighborhood_subset_analysis


if __name__ == "__main__":
    print("Starting neighborhood subset analysis...", flush=True)

    instances = {
        name: read_gvrp_instance(get_instance_path(name, INSTANCE_DIRS))
        for name in ANALYSIS_INSTANCE_NAMES
    }

    print(f"Loaded instances: {ANALYSIS_INSTANCE_NAMES}", flush=True)
    print(f"Seeds: {ANALYSIS_SEEDS}", flush=True)
    print(f"Time limit per run: {ANALYSIS_TIME_LIMIT_SECONDS}", flush=True)

    run_df, summary_df, operator_df, operator_summary_df = run_neighborhood_subset_analysis(
        instances=instances,
        instance_names=ANALYSIS_INSTANCE_NAMES,
        seeds=ANALYSIS_SEEDS,
        time_limit_seconds=ANALYSIS_TIME_LIMIT_SECONDS,
        best_known_soft=BEST_KNOWN_SOFT,
        alpha_balance=ALPHA_BALANCE,
        construction_iterations=CONSTRUCTION_ITERATIONS,
        initial_temp=SA_INITIAL_TEMP,
        cooling_rate=SA_COOLING_RATE,
        iterations_per_temp=SA_ITERATIONS_PER_TEMP,
        min_temp=SA_MIN_TEMP,
        max_neighbor_attempts=SA_MAX_NEIGHBOR_ATTEMPTS,
        operator_sets=OPERATOR_SETS,
        n_workers=4,
    )

    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    save_dataframe_csv(run_df, TABLES_DIR / "neighborhood_subset_runs.csv")
    save_dataframe_csv(summary_df, TABLES_DIR / "neighborhood_subset_summary.csv")
    save_dataframe_csv(operator_df, TABLES_DIR / "neighborhood_subset_operator_stats_runs.csv")
    save_dataframe_csv(operator_summary_df, TABLES_DIR / "neighborhood_subset_operator_stats_summary.csv")

    print("\nSubset summary:", flush=True)
    print(summary_df, flush=True)
    print("\nSubset operator summary:", flush=True)
    print(operator_summary_df, flush=True)