"""Shared optimizer result format."""

from __future__ import annotations

from typing import TypedDict

from src.cluvrp.types import Solution, RunHistory, RunStats


class OptimizerResult(TypedDict):
    best_solution: Solution
    history: RunHistory
    stats: RunStats