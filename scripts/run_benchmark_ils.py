"""Run all benchmark instances with iterated local search and save the final csv table."""

from configs.default import (
    INSTANCE_DIRS,
    INSTANCE_NAMES,
    BEST_KNOWN_HARD,
    BEST_KNOWN_SOFT,
    TABLES_DIR,
    LOGS_DIR,
)
from configs.sa import (
    ALPHA_BALANCE,
    SA_INITIAL_TEMP,
    SA_COOLING_RATE,
    SA_ITERATIONS_PER_TEMP,
    SA_MIN_TEMP,
    SA_MAX_NEIGHBOR_ATTEMPTS,
)
from configs.benchmark import CHECKPOINT_SECONDS, BENCHMARK_TIME_LIMIT_SECONDS, BASE_SEED
from configs.ils import (
    ILS_CONSTRUCTION_ITERATIONS,
    ILS_NEIGHBORHOOD_WEIGHTS,
    ILS_PERTURBATION_STEPS,
)
from src.cluvrp.io.instance_reader import get_instance_path, read_gvrp_instance
from src.cluvrp.experiments.benchmark import run_benchmark
from src.cluvrp.io.result_io import save_dataframe_csv, save_pickle

if __name__ == "__main__":
    instances = {
        name: read_gvrp_instance(get_instance_path(name, INSTANCE_DIRS))
        for name in INSTANCE_NAMES
    }

    results_df, all_runs = run_benchmark(
        instances=instances,
        instance_names=INSTANCE_NAMES,
        checkpoint_seconds=CHECKPOINT_SECONDS,
        time_limit_seconds=BENCHMARK_TIME_LIMIT_SECONDS,
        base_seed=BASE_SEED,
        best_known_soft=BEST_KNOWN_SOFT,
        best_known_hard=BEST_KNOWN_HARD,
        alpha_balance=ALPHA_BALANCE,
        construction_iterations=ILS_CONSTRUCTION_ITERATIONS,
        initial_temp=SA_INITIAL_TEMP,
        cooling_rate=SA_COOLING_RATE,
        iterations_per_temp=SA_ITERATIONS_PER_TEMP,
        min_temp=SA_MIN_TEMP,
        max_neighbor_attempts=SA_MAX_NEIGHBOR_ATTEMPTS,
        neighborhood_weights=ILS_NEIGHBORHOOD_WEIGHTS,
        method="ils",
        optimizer_kwargs={
            "perturbation_steps": ILS_PERTURBATION_STEPS,
        },
    )

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = TABLES_DIR / "hard_cluvrp_ils_results_A_to_K.csv"
    pkl_path = LOGS_DIR / "hard_benchmark_runs_ils.pkl"

    save_dataframe_csv(results_df, csv_path)
    save_pickle(all_runs, pkl_path)

    print(f"Saved table to: {csv_path.resolve()}")
    print(f"Saved full benchmark results to: {pkl_path.resolve()}")
    print(results_df)