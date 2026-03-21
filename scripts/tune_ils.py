"""Tune iterated local search with Optuna and save the results."""

from configs.default import (
    INSTANCE_DIRS,
    BEST_KNOWN_SOFT,
    TABLES_DIR,
    LOGS_DIR,
)
from configs.sa import ALPHA_BALANCE
from configs.tuning_ils import (
    TUNING_INSTANCE_NAMES,
    TUNING_SEEDS,
    TUNING_TIME_LIMIT_SECONDS,
    OPTUNA_N_TRIALS,
    OPTUNA_N_JOBS,
    OPTUNA_STUDY_NAME,
    OPTUNA_SEED,
    OPTUNA_RESET_STUDY,
    PERTURBATION_STEPS_MIN,
    PERTURBATION_STEPS_MAX,
    CONSTRUCTION_ITERATIONS_MIN,
    CONSTRUCTION_ITERATIONS_MAX,
    TUNE_RELOCATE_BEST,
    TUNE_REMOVE_REINSERT_TWO,
    TUNE_PAIR_RELOCATE_BEST,
    TUNE_REMOVE_REINSERT_THREE,
)
from src.cluvrp.io.instance_reader import get_instance_path, read_gvrp_instance
from src.cluvrp.io.result_io import save_dataframe_csv
from src.cluvrp.experiments.tuning_ils import run_optuna_tuning_ils, save_best_params_json


if __name__ == "__main__":
    instances = {
        name: read_gvrp_instance(get_instance_path(name, INSTANCE_DIRS))
        for name in TUNING_INSTANCE_NAMES
    }

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    storage_path = LOGS_DIR / f"{OPTUNA_STUDY_NAME}.db"
    storage_url = f"sqlite:///{storage_path.resolve().as_posix()}"

    print(f"Starting ILS tuning with {OPTUNA_N_JOBS} parallel trial workers", flush=True)
    print(f"Study storage: {storage_path.resolve()}", flush=True)

    study, trials_df, detailed_df, best_params = run_optuna_tuning_ils(
        instances=instances,
        tuning_instance_names=TUNING_INSTANCE_NAMES,
        seeds=TUNING_SEEDS,
        time_limit_seconds=TUNING_TIME_LIMIT_SECONDS,
        n_trials=OPTUNA_N_TRIALS,
        n_jobs=OPTUNA_N_JOBS,
        best_known_soft=BEST_KNOWN_SOFT,
        alpha_balance=ALPHA_BALANCE,
        perturbation_steps_min=PERTURBATION_STEPS_MIN,
        perturbation_steps_max=PERTURBATION_STEPS_MAX,
        construction_iterations_min=CONSTRUCTION_ITERATIONS_MIN,
        construction_iterations_max=CONSTRUCTION_ITERATIONS_MAX,
        tune_relocate_best=TUNE_RELOCATE_BEST,
        tune_remove_reinsert_two=TUNE_REMOVE_REINSERT_TWO,
        tune_pair_relocate_best=TUNE_PAIR_RELOCATE_BEST,
        tune_remove_reinsert_three=TUNE_REMOVE_REINSERT_THREE,
        study_name=OPTUNA_STUDY_NAME,
        storage_url=storage_url,
        optuna_seed=OPTUNA_SEED,
        reset_existing_study=OPTUNA_RESET_STUDY,
    )

    trials_path = TABLES_DIR / "ils_tuning_trials.csv"
    detailed_path = TABLES_DIR / "ils_tuning_detailed.csv"
    best_params_path = LOGS_DIR / "ils_best_params.json"

    save_dataframe_csv(trials_df, trials_path)
    save_dataframe_csv(detailed_df, detailed_path)
    save_best_params_json(best_params, best_params_path)

    print(f"Saved ILS trial summary to: {trials_path.resolve()}")
    print(f"Saved ILS detailed results to: {detailed_path.resolve()}")
    print(f"Saved best parameters to: {best_params_path.resolve()}")
    print("Best parameters found:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")