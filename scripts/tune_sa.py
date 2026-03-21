"""Tune simulated annealing parameters with Optuna and save the results."""

from configs.default import (
    INSTANCE_DIRS,
    BEST_KNOWN_SOFT,
    TABLES_DIR,
    LOGS_DIR,
)
from configs.sa import (
    ALPHA_BALANCE,
    SA_MIN_TEMP,
    SA_MAX_NEIGHBOR_ATTEMPTS,
    NEIGHBORHOOD_WEIGHTS,
)
from configs.tuning import (
    TUNING_INSTANCE_NAMES,
    TUNING_SEEDS,
    TUNING_TIME_LIMIT_SECONDS,
    OPTUNA_N_TRIALS,
    OPTUNA_STUDY_NAME,
    INITIAL_TEMP_MIN,
    INITIAL_TEMP_MAX,
    COOLING_RATE_MIN,
    COOLING_RATE_MAX,
    ITERATIONS_PER_TEMP_MIN,
    ITERATIONS_PER_TEMP_MAX,
    CONSTRUCTION_ITERATIONS_MIN,
    CONSTRUCTION_ITERATIONS_MAX,
)
from src.cluvrp.io.instance_reader import get_instance_path, read_gvrp_instance
from src.cluvrp.io.result_io import save_dataframe_csv
from src.cluvrp.experiments.tuning import run_optuna_tuning, save_best_params_json


if __name__ == "__main__":
    instances = {
        name: read_gvrp_instance(get_instance_path(name, INSTANCE_DIRS))
        for name in TUNING_INSTANCE_NAMES
    }

    study, trials_df, detailed_df, best_params = run_optuna_tuning(
        instances=instances,
        tuning_instance_names=TUNING_INSTANCE_NAMES,
        seeds=TUNING_SEEDS,
        time_limit_seconds=TUNING_TIME_LIMIT_SECONDS,
        n_trials=OPTUNA_N_TRIALS,
        best_known_soft=BEST_KNOWN_SOFT,
        alpha_balance=ALPHA_BALANCE,
        min_temp=SA_MIN_TEMP,
        max_neighbor_attempts=SA_MAX_NEIGHBOR_ATTEMPTS,
        neighborhood_weights=NEIGHBORHOOD_WEIGHTS,
        initial_temp_min=INITIAL_TEMP_MIN,
        initial_temp_max=INITIAL_TEMP_MAX,
        cooling_rate_min=COOLING_RATE_MIN,
        cooling_rate_max=COOLING_RATE_MAX,
        iterations_per_temp_min=ITERATIONS_PER_TEMP_MIN,
        iterations_per_temp_max=ITERATIONS_PER_TEMP_MAX,
        construction_iterations_min=CONSTRUCTION_ITERATIONS_MIN,
        construction_iterations_max=CONSTRUCTION_ITERATIONS_MAX,
        study_name=OPTUNA_STUDY_NAME,
    )

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    trials_path = TABLES_DIR / "sa_tuning_trials.csv"
    detailed_path = TABLES_DIR / "sa_tuning_detailed.csv"
    best_params_path = LOGS_DIR / "sa_best_params.json"

    save_dataframe_csv(trials_df, trials_path)
    save_dataframe_csv(detailed_df, detailed_path)
    save_best_params_json(best_params, best_params_path)

    print(f"Saved Optuna trial summary to: {trials_path.resolve()}")
    print(f"Saved Optuna detailed results to: {detailed_path.resolve()}")
    print(f"Saved best parameters to: {best_params_path.resolve()}")
    print("Best parameters found:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")