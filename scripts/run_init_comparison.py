"""Run initialization comparison under simulated annealing."""

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
    NEIGHBORHOOD_WEIGHTS,
)
from configs.init_comparison import (
    INIT_COMPARISON_INSTANCE_NAMES,
    INIT_COMPARISON_SEEDS,
    INIT_COMPARISON_TIME_LIMIT_SECONDS,
    INIT_METHODS,
)
from src.cluvrp.io.instance_reader import get_instance_path, read_gvrp_instance
from src.cluvrp.io.result_io import save_dataframe_csv
from src.cluvrp.experiments.init_comparison import run_init_comparison


if __name__ == "__main__":
    instances = {
        name: read_gvrp_instance(get_instance_path(name, INSTANCE_DIRS))
        for name in INIT_COMPARISON_INSTANCE_NAMES
    }

    run_df, summary_df, pvalues_df = run_init_comparison(
        instances=instances,
        instance_names=INIT_COMPARISON_INSTANCE_NAMES,
        seeds=INIT_COMPARISON_SEEDS,
        init_methods=INIT_METHODS,
        time_limit_seconds=INIT_COMPARISON_TIME_LIMIT_SECONDS,
        best_known_soft=BEST_KNOWN_SOFT,
        alpha_balance=ALPHA_BALANCE,
        construction_iterations=CONSTRUCTION_ITERATIONS,
        initial_temp=SA_INITIAL_TEMP,
        cooling_rate=SA_COOLING_RATE,
        iterations_per_temp=SA_ITERATIONS_PER_TEMP,
        min_temp=SA_MIN_TEMP,
        max_neighbor_attempts=SA_MAX_NEIGHBOR_ATTEMPTS,
        neighborhood_weights=NEIGHBORHOOD_WEIGHTS,
    )

    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    save_dataframe_csv(run_df, TABLES_DIR / "init_comparison_runs.csv")
    save_dataframe_csv(summary_df, TABLES_DIR / "init_comparison_summary.csv")
    save_dataframe_csv(pvalues_df, TABLES_DIR / "init_comparison_pvalues.csv")

    print("\nInitialization comparison summary:")
    print(summary_df)
    print("\nPaired t-test results:")
    print(pvalues_df)