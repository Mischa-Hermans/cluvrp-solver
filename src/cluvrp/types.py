"""Data classes for instances, solutions, and run history."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class GVRPInstance:
    name: str
    comment: str
    dimension: int
    vehicles: int
    n_clusters: int
    capacity: int
    coords: Dict[int, Tuple[float, float]]
    clusters: Dict[int, List[int]]
    cluster_demands: Dict[int, int]
    depot: int = 1


@dataclass
class Solution:
    superclusters: List[List[int]]
    loads: List[int]
    cluster_to_supercluster: Dict[int, int]
    supercluster_customers: List[List[int]]
    routes: List[List[int]]
    route_costs: List[float]
    total_cost: float
    construction_seed: Optional[int] = None
    last_move_type: Optional[str] = None

    def copy(self) -> "Solution":
        return Solution(
            superclusters=[sc[:] for sc in self.superclusters],
            loads=self.loads[:],
            cluster_to_supercluster=dict(self.cluster_to_supercluster),
            supercluster_customers=[custs[:] for custs in self.supercluster_customers],
            routes=[route[:] for route in self.routes],
            route_costs=self.route_costs[:],
            total_cost=self.total_cost,
            construction_seed=self.construction_seed,
            last_move_type=self.last_move_type,
        )


@dataclass
class HistoryRecord:
    elapsed_time: float
    current_cost: float
    best_cost: float
    accepted: bool
    improving: bool
    move_type: Optional[str]


@dataclass
class RunHistory:
    records: List[HistoryRecord] = field(default_factory=list)

    def add(
        self,
        elapsed_time: float,
        current_cost: float,
        best_cost: float,
        accepted: bool,
        improving: bool,
        move_type: Optional[str],
    ) -> None:
        self.records.append(
            HistoryRecord(
                elapsed_time=elapsed_time,
                current_cost=current_cost,
                best_cost=best_cost,
                accepted=accepted,
                improving=improving,
                move_type=move_type,
            )
        )


@dataclass
class RunStats:
    elapsed_time: float
    iterations: int
    accepted_moves: int
    improving_moves: int
    final_temperature: float