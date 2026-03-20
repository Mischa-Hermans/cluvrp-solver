"""Read the incumbent value at chosen time checkpoints."""

from __future__ import annotations

from src.cluvrp.types import RunHistory


def best_cost_at_time(history: RunHistory, checkpoint_time: float) -> float:
    best = history.records[0].best_cost
    for record in history.records:
        if record.elapsed_time <= checkpoint_time:
            best = record.best_cost
        else:
            break
    return best