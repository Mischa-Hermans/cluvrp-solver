"""Plot final vehicle routes."""

from __future__ import annotations

import matplotlib.pyplot as plt

from src.cluvrp.types import GVRPInstance, Solution


def plot_final_routes(instance: GVRPInstance, solution: Solution, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))

    colors = plt.cm.tab10.colors

    for node, (x, y) in instance.coords.items():
        if node == instance.depot:
            continue
        ax.scatter(x, y, s=40, alpha=0.7)
        ax.text(x + 0.5, y + 0.5, str(node), fontsize=8)

    for route_id, route in enumerate(solution.routes):
        color = colors[route_id % len(colors)]
        xs = [instance.coords[i][0] for i in route]
        ys = [instance.coords[i][1] for i in route]
        ax.plot(xs, ys, marker="o", linewidth=2, color=color, label=f"Vehicle {route_id+1}")

    depot_x, depot_y = instance.coords[instance.depot]
    ax.scatter(depot_x, depot_y, color="black", marker="s", s=160)
    ax.text(depot_x + 1, depot_y + 1, "Depot", fontsize=10, weight="bold")

    ax.set_title(f"{instance.name}: final routes")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    ax.grid(True)
    return ax