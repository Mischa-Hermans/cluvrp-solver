"""Run Optuna tuning for hybrid genetic search."""

from __future__ import annotations

import json
from pathlib import Path

import optuna
import pandas as pd

from src.cluvrp.core.evaluation import compute_gap_percent
from src.cluvrp.experiments.run_single_instance import run_single_instance


def evaluate_hgs_config(
    instances: dict,
    tuning_instance_names: list[str],
    seeds: list[int],
    time_limit_seconds: float,
    best_known_soft: dict,
    alpha_balance: float,
    neighborhood_weights: dict,
    construction_iterations: int,
    perturbation_steps: int,
    population_size: int,
    elite_size: int,
    tournament_size: int,
    initial_individual_time_seconds: float,
    offspring_improvement_time_seconds: float,
    parent1_route_inherit_prob: float,
):
    detailed_rows = []
    gaps = []
    obj_vals = []

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
                initial_temp=0.0,
                cooling_rate=0.0,
                iterations_per_temp=0,
                min_temp=0.0,
                max_neighbor_attempts=0,
                neighborhood_weights=neighborhood_weights,
                method="hgs",
                optimizer_kwargs={
                    "perturbation_steps": perturbation_steps,
                    "population_size": population_size,
                    "elite_size": elite_size,
                    "tournament_size": tournament_size,
                    "initial_individual_time_seconds": initial_individual_time_seconds,
                    "offspring_improvement_time_seconds": offspring_improvement_time_seconds,
                    "parent1_route_inherit_prob": parent1_route_inherit_prob,
                },
            )

            obj_val = result["best_solution"].total_cost
            gap_pct = None if best_known is None else compute_gap_percent(obj_val, best_known)

            obj_vals.append(obj_val)
            if gap_pct is not None:
                gaps.append(gap_pct)

            detailed_rows.append({
                "instance": instance_name,
                "seed": seed,
                "obj_val": round(obj_val, 3),
                "gap_pct": None if gap_pct is None else round(gap_pct, 3),
            })

    mean_gap = float(sum(gaps) / len(gaps)) if gaps else float("inf")
    mean_obj = float(sum(obj_vals) / len(obj_vals)) if obj_vals else float("inf")

    return mean_gap, mean_obj, detailed_rows


def run_optuna_tuning_hgs(
    instances: dict,
    tuning_instance_names: list[str],
    seeds: list[int],
    time_limit_seconds: float,
    n_trials: int,
    n_jobs: int,
    best_known_soft: dict,
    alpha_balance: float,
    neighborhood_weights: dict,
    population_size_min: int,
    population_size_max: int,
    elite_size_min: int,
    elite_size_max: int,
    tournament_size_min: int,
    tournament_size_max: int,
    initial_individual_time_min: float,
    initial_individual_time_max: float,
    offspring_improvement_time_min: float,
    offspring_improvement_time_max: float,
    parent1_route_inherit_prob_min: float,
    parent1_route_inherit_prob_max: float,
    perturbation_steps_min: int,
    perturbation_steps_max: int,
    construction_iterations_min: int,
    construction_iterations_max: int,
    study_name: str = "hgs_tuning",
    storage_url: str | None = None,
    optuna_seed: int | None = None,
    reset_existing_study: bool = False,
):
    sampler = optuna.samplers.TPESampler(seed=optuna_seed)

    if storage_url is not None and reset_existing_study:
        try:
            optuna.delete_study(study_name=study_name, storage=storage_url)
        except KeyError:
            pass

    study = optuna.create_study(
        direction="minimize",
        study_name=study_name,
        storage=storage_url,
        load_if_exists=not reset_existing_study,
        sampler=sampler,
    )

    def objective(trial: optuna.Trial) -> float:
        population_size = trial.suggest_int("population_size", population_size_min, population_size_max)
        elite_size = trial.suggest_int("elite_size", elite_size_min, min(elite_size_max, population_size - 1))
        tournament_size = trial.suggest_int("tournament_size", tournament_size_min, tournament_size_max)
        initial_individual_time_seconds = trial.suggest_float(
            "initial_individual_time_seconds",
            initial_individual_time_min,
            initial_individual_time_max,
        )
        offspring_improvement_time_seconds = trial.suggest_float(
            "offspring_improvement_time_seconds",
            offspring_improvement_time_min,
            offspring_improvement_time_max,
        )
        parent1_route_inherit_prob = trial.suggest_float(
            "parent1_route_inherit_prob",
            parent1_route_inherit_prob_min,
            parent1_route_inherit_prob_max,
        )
        perturbation_steps = trial.suggest_int(
            "perturbation_steps",
            perturbation_steps_min,
            perturbation_steps_max,
        )
        construction_iterations = trial.suggest_int(
            "construction_iterations",
            construction_iterations_min,
            construction_iterations_max,
        )

        print(
            f"Trial {trial.number}: "
            f"pop={population_size}, elite={elite_size}, tourn={tournament_size}, "
            f"init={initial_individual_time_seconds:.2f}, child={offspring_improvement_time_seconds:.2f}, "
            f"inherit={parent1_route_inherit_prob:.2f}, perturb={perturbation_steps}, "
            f"construct={construction_iterations}",
            flush=True,
        )

        mean_gap, mean_obj, detailed_rows = evaluate_hgs_config(
            instances=instances,
            tuning_instance_names=tuning_instance_names,
            seeds=seeds,
            time_limit_seconds=time_limit_seconds,
            best_known_soft=best_known_soft,
            alpha_balance=alpha_balance,
            neighborhood_weights=neighborhood_weights,
            construction_iterations=construction_iterations,
            perturbation_steps=perturbation_steps,
            population_size=population_size,
            elite_size=elite_size,
            tournament_size=tournament_size,
            initial_individual_time_seconds=initial_individual_time_seconds,
            offspring_improvement_time_seconds=offspring_improvement_time_seconds,
            parent1_route_inherit_prob=parent1_route_inherit_prob,
        )

        trial.set_user_attr("mean_obj_val", mean_obj)
        trial.set_user_attr("detailed_rows", detailed_rows)

        print(f" -> mean gap = {mean_gap:.3f}", flush=True)
        return mean_gap

    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs)

    trial_rows = []
    detailed_rows = []

    for trial in study.trials:
        if trial.state != optuna.trial.TrialState.COMPLETE:
            continue

        params = trial.params
        trial_rows.append({
            "trial_number": trial.number,
            "population_size": params.get("population_size"),
            "elite_size": params.get("elite_size"),
            "tournament_size": params.get("tournament_size"),
            "initial_individual_time_seconds": params.get("initial_individual_time_seconds"),
            "offspring_improvement_time_seconds": params.get("offspring_improvement_time_seconds"),
            "parent1_route_inherit_prob": params.get("parent1_route_inherit_prob"),
            "perturbation_steps": params.get("perturbation_steps"),
            "construction_iterations": params.get("construction_iterations"),
            "mean_gap_pct": trial.value,
            "mean_obj_val": trial.user_attrs.get("mean_obj_val"),
        })

        for row in trial.user_attrs.get("detailed_rows", []):
            detailed_rows.append({
                "trial_number": trial.number,
                **{k: params.get(k) for k in params},
                **row,
            })

    trials_df = pd.DataFrame(trial_rows).sort_values("mean_gap_pct").reset_index(drop=True)
    detailed_df = pd.DataFrame(detailed_rows)

    best_params = dict(study.best_params)
    best_params["best_mean_gap_pct"] = study.best_value

    return study, trials_df, detailed_df, best_params


def save_best_params_json(best_params: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(best_params, indent=2))