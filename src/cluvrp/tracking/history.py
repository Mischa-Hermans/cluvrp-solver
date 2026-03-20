"""Store the best-so-far path during one run."""

from __future__ import annotations

from src.cluvrp.types import RunHistory


def initialize_history(initial_cost: float) -> RunHistory:
    history = RunHistory()
    history.add(
        elapsed_time=0.0,
        current_cost=initial_cost,
        best_cost=initial_cost,
        accepted=True,
        improving=False,
        move_type="initial_solution",
    )
    return history


def record_step(
    history: RunHistory,
    elapsed_time: float,
    current_cost: float,
    best_cost: float,
    accepted: bool,
    improving: bool,
    move_type,
) -> None:
    history.add(
        elapsed_time=elapsed_time,
        current_cost=current_cost,
        best_cost=best_cost,
        accepted=accepted,
        improving=improving,
        move_type=move_type,
    )