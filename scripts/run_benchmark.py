"""Run all benchmark instances with the selected method and save the results."""

from configs.default import (
    INSTANCE_DIRS,
    INSTANCE_NAMES,
    BEST_KNOWN_HARD,
    BEST_KNOWN_SOFT,
    TABLES_DIR,
    LOGS_DIR,
)
from configs.methods import BENCHMARK_METHOD
from configs.routing import ROUTING_VARIANT
from configs.benchmark import (
    CHECKPOINT_SECONDS,
    BENCHMARK_TIME_LIMIT_SECONDS,
    BASE_SEED,
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
from configs.ils import (
    ILS_CONSTRUCTION_ITERATIONS,
    ILS_NEIGHBORHOOD_WEIGHTS,
    ILS_PERTURBATION_STEPS,
)
from configs.hgs import (
    HGS_CONSTRUCTION_ITERATIONS,
    HGS_NEIGHBORHOOD_WEIGHTS,
    HGS_PERTURBATION_STEPS,
)
from src.cluvrp.io.instance_reader import get_instance_path, read_gvrp_instance
from src.cluvrp.experiments.benchmark import run_benchmark
from src.cluvrp.io.result_io import save_dataframe_csv, save_pickle


def get_benchmark_settings(method: str) -> tuple[dict, dict]:
    if method == "sa":
        return (
            {
                "alpha_balance": ALPHA_BALANCE,
                "construction_iterations": CONSTRUCTION_ITERATIONS,
                "initial_temp": SA_INITIAL_TEMP,
                "cooling_rate": SA_COOLING_RATE,
                "iterations_per_temp": SA_ITERATIONS_PER_TEMP,
                "min_temp": SA_MIN_TEMP,
                "max_neighbor_attempts": SA_MAX_NEIGHBOR_ATTEMPTS,
                "neighborhood_weights": NEIGHBORHOOD_WEIGHTS,
            },
            {},
        )

    if method == "ils":
        return (
            {
                "alpha_balance": ALPHA_BALANCE,
                "construction_iterations": ILS_CONSTRUCTION_ITERATIONS,
                "initial_temp": 0.0,
                "cooling_rate": 0.0,
                "iterations_per_temp": 0,
                "min_temp": 0.0,
                "max_neighbor_attempts": 0,
                "neighborhood_weights": ILS_NEIGHBORHOOD_WEIGHTS,
            },
            {
                "perturbation_steps": ILS_PERTURBATION_STEPS,
            },
        )

    if method == "hgs":
        return (
            {
                "alpha_balance": ALPHA_BALANCE,
                "construction_iterations": HGS_CONSTRUCTION_ITERATIONS,
                "initial_temp": 0.0,
                "cooling_rate": 0.0,
                "iterations_per_temp": 0,
                "min_temp": 0.0,
                "max_neighbor_attempts": 0,
                "neighborhood_weights": HGS_NEIGHBORHOOD_WEIGHTS,
            },
            {
                "perturbation_steps": HGS_PERTURBATION_STEPS,
            },
        )

    raise ValueError(f"Unknown method: {method}")


if __name__ == "__main__":
    instances = {
        name: read_gvrp_instance(get_instance_path(name, INSTANCE_DIRS))
        for name in INSTANCE_NAMES
    }

    base_kwargs, optimizer_kwargs = get_benchmark_settings(BENCHMARK_METHOD)

    print(f"Routing variant: {ROUTING_VARIANT}")
    print(f"Method: {BENCHMARK_METHOD}")

    results_df, all_runs = run_benchmark(
        instances=instances,
        instance_names=INSTANCE_NAMES,
        checkpoint_seconds=CHECKPOINT_SECONDS,
        time_limit_seconds=BENCHMARK_TIME_LIMIT_SECONDS,
        base_seed=BASE_SEED,
        best_known_soft=BEST_KNOWN_SOFT,
        best_known_hard=BEST_KNOWN_HARD,
        method=BENCHMARK_METHOD,
        optimizer_kwargs=optimizer_kwargs,
        **base_kwargs,
    )

    results_df.insert(0, "method", BENCHMARK_METHOD)
    results_df.insert(1, "routing_variant", ROUTING_VARIANT)

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = TABLES_DIR / f"{ROUTING_VARIANT}_cluvrp_{BENCHMARK_METHOD}_results_A_to_K.csv"
    pkl_path = LOGS_DIR / f"{ROUTING_VARIANT}_benchmark_runs_{BENCHMARK_METHOD}.pkl"

    save_dataframe_csv(results_df, csv_path)
    save_pickle(all_runs, pkl_path)

    print(f"Saved table to: {csv_path.resolve()}")
    print(f"Saved full benchmark results to: {pkl_path.resolve()}")
    print(results_df)