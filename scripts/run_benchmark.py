"""Run all benchmark instances and save the final csv table."""

from configs.default import (
    INSTANCE_DIRS,
    INSTANCE_NAMES,
    BEST_KNOWN_HARD,
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
from configs.benchmark import CHECKPOINT_SECONDS, BENCHMARK_TIME_LIMIT_SECONDS, BASE_SEED
from src.cluvrp.io.instance_reader import get_instance_path, read_gvrp_instance
from src.cluvrp.experiments.benchmark import run_benchmark
from src.cluvrp.io.result_io import save_dataframe_csv

if __name__ == "__main__":
    instances = {
        name: read_gvrp_instance(get_instance_path(name, INSTANCE_DIRS))
        for name in INSTANCE_NAMES
    }

    results_df, _ = run_benchmark(
        instances=instances,
        instance_names=INSTANCE_NAMES,
        checkpoint_seconds=CHECKPOINT_SECONDS,
        time_limit_seconds=BENCHMARK_TIME_LIMIT_SECONDS,
        base_seed=BASE_SEED,
        best_known_soft=BEST_KNOWN_SOFT,
        best_known_hard=BEST_KNOWN_HARD,
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
    out_path = TABLES_DIR / "soft_cluvrp_sa_exact_results_A_to_K.csv"
    save_dataframe_csv(results_df, out_path)
    print(f"Saved results to: {out_path.resolve()}")
    print(results_df)