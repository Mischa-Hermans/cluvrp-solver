"""Make plots for all benchmark instances."""

import matplotlib.pyplot as plt

from configs.default import (
    INSTANCE_DIRS,
    INSTANCE_NAMES,
    BEST_KNOWN_HARD,
    BEST_KNOWN_SOFT,
    PLOTS_DIR,
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
from src.cluvrp.visualization.clusters import plot_original_clusters, plot_superclusters
from src.cluvrp.visualization.routes import plot_final_routes
from src.cluvrp.visualization.convergence import plot_convergence


if __name__ == "__main__":
    instances = {
        name: read_gvrp_instance(get_instance_path(name, INSTANCE_DIRS))
        for name in INSTANCE_NAMES
    }

    _, all_runs = run_benchmark(
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

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    for instance_name in INSTANCE_NAMES:
        instance = instances[instance_name]
        run = all_runs[instance_name]

        initial_solution = run["initial_solution"]
        best_solution = run["best_solution"]

        init_cost = initial_solution.total_cost
        final_cost = best_solution.total_cost

        fig, axes = plt.subplots(2, 2, figsize=(18, 14))

        # Original clusters
        plot_original_clusters(instance, ax=axes[0, 0])
        axes[0, 0].set_title(f"{instance.name} – original clusters")

        # Superclusters (from initial solution)
        plot_superclusters(instance, initial_solution, ax=axes[0, 1])
        axes[0, 1].set_title(f"{instance.name} – superclusters (construction)")

        # Initial solution routes
        plot_final_routes(instance, initial_solution, ax=axes[1, 0])
        axes[1, 0].set_title(
            f"{instance.name} – initial routes (before SA)\nCost = {init_cost:.2f}"
        )

        # Final solution routes
        plot_final_routes(instance, best_solution, ax=axes[1, 1])
        axes[1, 1].set_title(
            f"{instance.name} – final routes (after SA)\nCost = {final_cost:.2f}"
        )

        plt.tight_layout()
        plt.savefig(PLOTS_DIR / f"{instance_name}_overview.png", dpi=200)
        plt.close()

        # Convergence plot
        fig, ax = plt.subplots(figsize=(8, 5))
        plot_convergence(
            run["history"],
            ax=ax,
            title=f"{instance.name} – convergence (best-so-far)",
        )
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / f"{instance_name}_convergence.png", dpi=200)
        plt.close()

        print(f"Saved plots for instance {instance_name}")