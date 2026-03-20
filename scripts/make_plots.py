"""Make plots from saved benchmark runs."""

import matplotlib.pyplot as plt

from configs.default import (
    INSTANCE_DIRS,
    INSTANCE_NAMES,
    PLOTS_DIR,
    LOGS_DIR,
)
from src.cluvrp.io.instance_reader import get_instance_path, read_gvrp_instance
from src.cluvrp.io.result_io import load_pickle
from src.cluvrp.visualization.clusters import plot_original_clusters, plot_superclusters
from src.cluvrp.visualization.routes import plot_final_routes
from src.cluvrp.visualization.convergence import plot_convergence


if __name__ == "__main__":
    instances = {
        name: read_gvrp_instance(get_instance_path(name, INSTANCE_DIRS))
        for name in INSTANCE_NAMES
    }

    pkl_path = LOGS_DIR / "benchmark_runs.pkl"
    all_runs = load_pickle(pkl_path)

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    for instance_name in INSTANCE_NAMES:
        instance = instances[instance_name]
        run = all_runs[instance_name]

        initial_solution = run["initial_solution"]
        best_solution = run["best_solution"]

        init_cost = initial_solution.total_cost
        final_cost = best_solution.total_cost

        fig, axes = plt.subplots(2, 2, figsize=(18, 14))

        plot_original_clusters(instance, ax=axes[0, 0])
        axes[0, 0].set_title(f"{instance.name} – original clusters")

        plot_superclusters(instance, initial_solution, ax=axes[0, 1])
        axes[0, 1].set_title(f"{instance.name} – superclusters (construction)")

        plot_final_routes(instance, initial_solution, ax=axes[1, 0])
        axes[1, 0].set_title(
            f"{instance.name} – initial routes (before SA)\nCost = {init_cost:.2f}"
        )

        plot_final_routes(instance, best_solution, ax=axes[1, 1])
        axes[1, 1].set_title(
            f"{instance.name} – final routes (after SA)\nCost = {final_cost:.2f}"
        )

        plt.tight_layout()
        plt.savefig(PLOTS_DIR / f"{instance_name}_overview.png", dpi=200)
        plt.close()

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