"""Common type alias for neighborhood functions."""

from __future__ import annotations

from typing import Callable, Optional

from src.cluvrp.types import GVRPInstance, Solution

NeighborhoodFn = Callable[[GVRPInstance, Solution, object, object], Optional[Solution]]