"""Feasibility checks for capacities and instance sanity."""

from __future__ import annotations

from src.cluvrp.types import GVRPInstance


def check_instance_feasibility(instance: GVRPInstance) -> None:
    oversized = [r for r, q in instance.cluster_demands.items() if q > instance.capacity]
    if oversized:
        raise ValueError(f"Clusters that exceed capacity by themselves: {oversized}")

    total_demand = sum(instance.cluster_demands.values())
    total_capacity = instance.vehicles * instance.capacity
    if total_demand > total_capacity:
        raise ValueError(
            f"Instance infeasible: total demand {total_demand} > total fleet capacity {total_capacity}"
        )


def feasible_load(instance: GVRPInstance, sc_clusters: list[int]) -> int:
    return sum(instance.cluster_demands[r] for r in sc_clusters)