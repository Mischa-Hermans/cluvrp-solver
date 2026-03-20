"""Plot how the best-so-far value changes over time."""

from __future__ import annotations

import matplotlib.pyplot as plt

from src.cluvrp.types import RunHistory


def plot_convergence(history: RunHistory, ax=None, title: str = "Convergence"):
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    xs = [r.elapsed_time for r in history.records]
    ys = [r.best_cost for r in history.records]

    ax.plot(xs, ys)
    ax.set_title(title)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("best-so-far cost")
    ax.grid(True)
    return ax