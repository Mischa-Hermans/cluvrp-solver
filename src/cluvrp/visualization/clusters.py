"""Plot original clusters and final superclusters."""

from __future__ import annotations

import matplotlib.pyplot as plt

from src.cluvrp.core.distances import compute_cluster_centroids
from src.cluvrp.types import GVRPInstance, Solution


def plot_original_clusters(instance: GVRPInstance, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))

    colors = plt.cm.tab20.colors

    for idx, (cluster_id, customers) in enumerate(instance.clusters.items()):
        color = colors[idx % len(colors)]
        xs = [instance.coords[i][0] for i in customers]
        ys = [instance.coords[i][1] for i in customers]
        ax.scatter(xs, ys, color=color, s=55, label=f"C{cluster_id}")
        for i in customers:
            ax.text(instance.coords[i][0] + 0.5, instance.coords[i][1] + 0.5, str(i), fontsize=8)

    depot_x, depot_y = instance.coords[instance.depot]
    ax.scatter(depot_x, depot_y, color="black", marker="s", s=140)
    ax.text(depot_x + 1, depot_y + 1, "Depot", fontsize=10, weight="bold")

    ax.set_title(f"{instance.name}: original clusters")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True)
    return ax


def plot_superclusters(instance: GVRPInstance, solution: Solution, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))

    colors = plt.cm.tab10.colors
    centroids = compute_cluster_centroids(instance)

    for r, customers in instance.clusters.items():
        sc_id = solution.cluster_to_supercluster[r]
        color = colors[sc_id % len(colors)]

        xs = [instance.coords[i][0] for i in customers]
        ys = [instance.coords[i][1] for i in customers]
        ax.scatter(xs, ys, color=color, s=60)

        for i in customers:
            ax.text(instance.coords[i][0] + 0.5, instance.coords[i][1] + 0.5, str(i), fontsize=8)

    for r, (cx, cy) in centroids.items():
        sc_id = solution.cluster_to_supercluster[r]
        color = colors[sc_id % len(colors)]
        ax.scatter(cx, cy, color=color, marker="x", s=130, linewidths=2)
        ax.text(cx + 1.0, cy + 1.0, f"C{r}", fontsize=9, weight="bold")

    depot_x, depot_y = instance.coords[instance.depot]
    ax.scatter(depot_x, depot_y, color="black", marker="s", s=140)
    ax.text(depot_x + 1, depot_y + 1, "Depot", fontsize=10, weight="bold")

    for sc_id, load in enumerate(solution.loads):
        ax.scatter([], [], color=colors[sc_id % len(colors)], label=f"SC {sc_id+1} (load={load})")

    ax.set_title(f"{instance.name}: superclusters")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    ax.grid(True)
    return ax