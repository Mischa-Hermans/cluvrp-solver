"""Tune HGS with Optuna and save the results."""

from configs.default import (
    INSTANCE_DIRS,
    BEST_KNOWN_SOFT,
    TABLES_DIR,
    LOGS_DIR,
)
from configs.sa import ALPHA_BALANCE
from configs.hgs import HGS_NEIGHBORHOOD_WEIGHTS
from configs.tuning_hgs import (
    TUNING_INSTANCE_NAMES,
    TUNING_SEEDS,
    TUNING_TIME_LIMIT_SECONDS,
    OPTUNA_N_TRIALS,
    OPTUNA_N_JOBS,
    OPTUNA_STUDY_NAME,
    OPTUNA_SEED,
    OPTUNA_RESET_STUDY,
    POPULATION_SIZE_MIN,
    POPULATION_SIZE_MAX,
    ELITE_SIZE_MIN,
    ELITE_SIZE_MAX,
    TOURNAMENT_SIZE_MIN,
    TOURNAMENT_SIZE_MAX,
    INITIAL_INDIVIDUAL_TIME_MIN,
    INITIAL_INDIVIDUAL_TIME_MAX,
    OFFSPRING_IMPROVEMENT_TIME_MIN,
    OFFSPRING_IMPROVEMENT_TIME_MAX,
    PARENT1_ROUTE_INHERIT_PROB_MIN,
    PARENT1_ROUTE_INHERIT_PROB_MAX,
    PERTURBATION_STEPS_MIN,
    PERTURBATION_STEPS_MAX,
    CONSTRUCTION_ITERATIONS_MIN,
    CONSTRUCTION_ITERATIONS_MAX,
)
from src.cluvrp.io.instance_reader import get_instance_path, read_gvrp_instance
from src.cluvrp.io.result_io import save_dataframe_csv
from src.cluvrp.experiments.tuning_hgs import run_optuna_tuning_hgs, save_best_params_json

if __name__ == "__main__":
    instances = {
        name: read_gvrp_instance(get_instance_path(name, INSTANCE_DIRS))
        for name in TUNING_INSTANCE_NAMES
    }

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    storage_path = LOGS_DIR / f"{OPTUNA_STUDY_NAME}.db"
    storage_url = f"sqlite:///{storage_path.resolve().as_posix()}"

    print(f"Starting HGS tuning with {OPTUNA_N_JOBS} parallel trial workers", flush=True)
    print(f"Study storage: {storage_path.resolve()}", flush=True)

    study, trials_df, detailed_df, best_params = run_optuna_tuning_hgs(
        instances=instances,
        tuning_instance_names=TUNING_INSTANCE_NAMES,
        seeds=TUNING_SEEDS,
        time_limit_seconds=TUNING_TIME_LIMIT_SECONDS,
        n_trials=OPTUNA_N_TRIALS,
        n_jobs=OPTUNA_N_JOBS,
        best_known_soft=BEST_KNOWN_SOFT,
        alpha_balance=ALPHA_BALANCE,
        neighborhood_weights=HGS_NEIGHBORHOOD_WEIGHTS,
        population_size_min=POPULATION_SIZE_MIN,
        population_size_max=POPULATION_SIZE_MAX,
        elite_size_min=ELITE_SIZE_MIN,
        elite_size_max=ELITE_SIZE_MAX,
        tournament_size_min=TOURNAMENT_SIZE_MIN,
        tournament_size_max=TOURNAMENT_SIZE_MAX,
        initial_individual_time_min=INITIAL_INDIVIDUAL_TIME_MIN,
        initial_individual_time_max=INITIAL_INDIVIDUAL_TIME_MAX,
        offspring_improvement_time_min=OFFSPRING_IMPROVEMENT_TIME_MIN,
        offspring_improvement_time_max=OFFSPRING_IMPROVEMENT_TIME_MAX,
        parent1_route_inherit_prob_min=PARENT1_ROUTE_INHERIT_PROB_MIN,
        parent1_route_inherit_prob_max=PARENT1_ROUTE_INHERIT_PROB_MAX,
        perturbation_steps_min=PERTURBATION_STEPS_MIN,
        perturbation_steps_max=PERTURBATION_STEPS_MAX,
        construction_iterations_min=CONSTRUCTION_ITERATIONS_MIN,
        construction_iterations_max=CONSTRUCTION_ITERATIONS_MAX,
        study_name=OPTUNA_STUDY_NAME,
        storage_url=storage_url,
        optuna_seed=OPTUNA_SEED,
        reset_existing_study=OPTUNA_RESET_STUDY,
    )

    trials_path = TABLES_DIR / "hgs_tuning_trials.csv"
    detailed_path = TABLES_DIR / "hgs_tuning_detailed.csv"
    best_params_path = LOGS_DIR / "hgs_best_params.json"

    save_dataframe_csv(trials_df, trials_path)
    save_dataframe_csv(detailed_df, detailed_path)
    save_best_params_json(best_params, best_params_path)

    print(f"Saved HGS trial summary to: {trials_path.resolve()}")
    print(f"Saved HGS detailed results to: {detailed_path.resolve()}")
    print(f"Saved best parameters to: {best_params_path.resolve()}")
    print("Best parameters found:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")