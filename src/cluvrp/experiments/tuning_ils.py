"""Run Optuna tuning for iterated local search."""

from __future__ import annotations

import json
from pathlib import Path

import optuna
import pandas as pd

from src.cluvrp.core.evaluation import compute_gap_percent
from src.cluvrp.experiments.run_single_instance import run_single_instance


def build_neighborhood_weights(
    use_relocate_best: int,
    use_remove_reinsert_two: int,
    use_pair_relocate_best: int,
    use_remove_reinsert_three: int,
) -> dict:
    return {
        "relocate_best": float(use_relocate_best),
        "swap_restricted": 0.0,
        "remove_reinsert_two": float(use_remove_reinsert_two),
        "ejection_chain_light": 0.0,
        "pair_relocate_best": float(use_pair_relocate_best),
        "swap_two_one": 0.0,
        "remove_reinsert_three": float(use_remove_reinsert_three),
    }


def evaluate_ils_config(
    instances: dict,
    tuning_instance_names: list[str],
    seeds: list[int],
    time_limit_seconds: float,
    best_known_soft: dict,
    alpha_balance: float,
    construction_iterations: int,
    perturbation_steps: int,
    neighborhood_weights: dict,
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
                method="ils",
                optimizer_kwargs={
                    "perturbation_steps": perturbation_steps,
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


def run_optuna_tuning_ils(
    instances: dict,
    tuning_instance_names: list[str],
    seeds: list[int],
    time_limit_seconds: float,
    n_trials: int,
    n_jobs: int,
    best_known_soft: dict,
    alpha_balance: float,
    perturbation_steps_min: int,
    perturbation_steps_max: int,
    construction_iterations_min: int,
    construction_iterations_max: int,
    tune_relocate_best: bool,
    tune_remove_reinsert_two: bool,
    tune_pair_relocate_best: bool,
    tune_remove_reinsert_three: bool,
    study_name: str = "ils_tuning",
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

        use_relocate_best = (
            trial.suggest_categorical("use_relocate_best", [0, 1])
            if tune_relocate_best else 0
        )
        use_remove_reinsert_two = (
            trial.suggest_categorical("use_remove_reinsert_two", [0, 1])
            if tune_remove_reinsert_two else 0
        )
        use_pair_relocate_best = (
            trial.suggest_categorical("use_pair_relocate_best", [0, 1])
            if tune_pair_relocate_best else 1
        )
        use_remove_reinsert_three = (
            trial.suggest_categorical("use_remove_reinsert_three", [0, 1])
            if tune_remove_reinsert_three else 1
        )

        if (
            use_relocate_best
            + use_remove_reinsert_two
            + use_pair_relocate_best
            + use_remove_reinsert_three
        ) == 0:
            raise optuna.TrialPruned()

        neighborhood_weights = build_neighborhood_weights(
            use_relocate_best=use_relocate_best,
            use_remove_reinsert_two=use_remove_reinsert_two,
            use_pair_relocate_best=use_pair_relocate_best,
            use_remove_reinsert_three=use_remove_reinsert_three,
        )

        print(
            f"Trial {trial.number}: "
            f"perturb={perturbation_steps}, "
            f"construct={construction_iterations}, "
            f"relocate={use_relocate_best}, "
            f"remove2={use_remove_reinsert_two}, "
            f"pair={use_pair_relocate_best}, "
            f"remove3={use_remove_reinsert_three}",
            flush=True,
        )

        mean_gap, mean_obj, detailed_rows = evaluate_ils_config(
            instances=instances,
            tuning_instance_names=tuning_instance_names,
            seeds=seeds,
            time_limit_seconds=time_limit_seconds,
            best_known_soft=best_known_soft,
            alpha_balance=alpha_balance,
            construction_iterations=construction_iterations,
            perturbation_steps=perturbation_steps,
            neighborhood_weights=neighborhood_weights,
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
            "perturbation_steps": params.get("perturbation_steps"),
            "construction_iterations": params.get("construction_iterations"),
            "use_relocate_best": params.get("use_relocate_best", 0),
            "use_remove_reinsert_two": params.get("use_remove_reinsert_two", 0),
            "use_pair_relocate_best": params.get("use_pair_relocate_best", 1),
            "use_remove_reinsert_three": params.get("use_remove_reinsert_three", 1),
            "mean_gap_pct": trial.value,
            "mean_obj_val": trial.user_attrs.get("mean_obj_val"),
        })

        for row in trial.user_attrs.get("detailed_rows", []):
            detailed_rows.append({
                "trial_number": trial.number,
                "perturbation_steps": params.get("perturbation_steps"),
                "construction_iterations": params.get("construction_iterations"),
                "use_relocate_best": params.get("use_relocate_best", 0),
                "use_remove_reinsert_two": params.get("use_remove_reinsert_two", 0),
                "use_pair_relocate_best": params.get("use_pair_relocate_best", 1),
                "use_remove_reinsert_three": params.get("use_remove_reinsert_three", 1),
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