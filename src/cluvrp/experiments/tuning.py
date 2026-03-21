"""Run Optuna tuning for simulated annealing parameters."""

from __future__ import annotations

import json
from pathlib import Path

import optuna
import pandas as pd

from src.cluvrp.core.evaluation import compute_gap_percent
from src.cluvrp.experiments.run_single_instance import run_single_instance


def evaluate_sa_config(
    instances: dict,
    tuning_instance_names: list[str],
    seeds: list[int],
    time_limit_seconds: float,
    best_known_soft: dict,
    alpha_balance: float,
    min_temp: float,
    max_neighbor_attempts: int,
    neighborhood_weights: dict,
    initial_temp: float,
    cooling_rate: float,
    iterations_per_temp: int,
    construction_iterations: int,
):
    rows = []
    gaps = []

    for instance_name in tuning_instance_names:
        instance = instances[instance_name]
        best_known = best_known_soft.get(instance_name)

        for seed in seeds:
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

            obj_val = result["best_solution"].total_cost
            gap_pct = None if best_known is None else compute_gap_percent(obj_val, best_known)

            rows.append({
                "instance": instance_name,
                "seed": seed,
                "obj_val": round(obj_val, 3),
                "gap_pct": None if gap_pct is None else round(gap_pct, 3),
            })

            if gap_pct is not None:
                gaps.append(gap_pct)

    mean_gap = float(sum(gaps) / len(gaps)) if gaps else float("inf")
    detailed_df = pd.DataFrame(rows)
    return mean_gap, detailed_df


def run_optuna_tuning(
    instances: dict,
    tuning_instance_names: list[str],
    seeds: list[int],
    time_limit_seconds: float,
    n_trials: int,
    best_known_soft: dict,
    alpha_balance: float,
    min_temp: float,
    max_neighbor_attempts: int,
    neighborhood_weights: dict,
    initial_temp_min: float,
    initial_temp_max: float,
    cooling_rate_min: float,
    cooling_rate_max: float,
    iterations_per_temp_min: int,
    iterations_per_temp_max: int,
    construction_iterations_min: int,
    construction_iterations_max: int,
    study_name: str = "sa_tuning",
):
    trial_rows = []
    detailed_rows = []

    def objective(trial: optuna.Trial) -> float:
        initial_temp = trial.suggest_float(
            "initial_temp",
            initial_temp_min,
            initial_temp_max,
            log=True,
        )
        cooling_rate = trial.suggest_float(
            "cooling_rate",
            cooling_rate_min,
            cooling_rate_max,
        )
        iterations_per_temp = trial.suggest_int(
            "iterations_per_temp",
            iterations_per_temp_min,
            iterations_per_temp_max,
        )
        construction_iterations = trial.suggest_int(
            "construction_iterations",
            construction_iterations_min,
            construction_iterations_max,
        )

        print(
            f"\nTrial {trial.number}: "
            f"T0={initial_temp:.1f}, "
            f"cool={cooling_rate:.4f}, "
            f"iters={iterations_per_temp}, "
            f"construct={construction_iterations}"
        )

        mean_gap, detailed_df = evaluate_sa_config(
            instances=instances,
            tuning_instance_names=tuning_instance_names,
            seeds=seeds,
            time_limit_seconds=time_limit_seconds,
            best_known_soft=best_known_soft,
            alpha_balance=alpha_balance,
            min_temp=min_temp,
            max_neighbor_attempts=max_neighbor_attempts,
            neighborhood_weights=neighborhood_weights,
            initial_temp=initial_temp,
            cooling_rate=cooling_rate,
            iterations_per_temp=iterations_per_temp,
            construction_iterations=construction_iterations,
        )

        print(f" -> mean gap = {mean_gap:.3f}")

        trial_rows.append({
            "trial_number": trial.number,
            "initial_temp": initial_temp,
            "cooling_rate": cooling_rate,
            "iterations_per_temp": iterations_per_temp,
            "construction_iterations": construction_iterations,
            "mean_gap_pct": mean_gap,
        })

        if not detailed_df.empty:
            detailed_df = detailed_df.copy()
            detailed_df["trial_number"] = trial.number
            detailed_df["initial_temp"] = initial_temp
            detailed_df["cooling_rate"] = cooling_rate
            detailed_df["iterations_per_temp"] = iterations_per_temp
            detailed_df["construction_iterations"] = construction_iterations
            detailed_rows.extend(detailed_df.to_dict(orient="records"))

        return mean_gap

    study = optuna.create_study(
        direction="minimize",
        study_name=study_name,
    )
    study.optimize(objective, n_trials=n_trials)

    trials_df = pd.DataFrame(trial_rows).sort_values("mean_gap_pct").reset_index(drop=True)
    detailed_df = pd.DataFrame(detailed_rows)

    best_params = dict(study.best_params)
    best_params["best_mean_gap_pct"] = study.best_value

    return study, trials_df, detailed_df, best_params


def save_best_params_json(best_params: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(best_params, indent=2))