"""Build full solutions by routing each supercluster."""

from __future__ import annotations

import random

from configs.routing import ROUTING_SOLVER, ROUTING_VARIANT, HYBRID_EXACT_MAX_CUSTOMERS
from src.cluvrp.core.feasibility import feasible_load
from src.cluvrp.core.evaluation import route_length
from src.cluvrp.types import GVRPInstance, Solution
from src.cluvrp.routing.tsp_exact import exact_tsp_gurobi
from src.cluvrp.routing.tsp_heuristics import nearest_neighbor_tour, two_opt


def customers_of_supercluster(instance: GVRPInstance, sc_clusters: list[int]) -> list[int]:
    customers = []
    for r in sc_clusters:
        customers.extend(instance.clusters[r])
    return customers


def build_cluster_to_supercluster(superclusters: list[list[int]]) -> dict[int, int]:
    cluster_to_supercluster = {}
    for sc_id, sc in enumerate(superclusters):
        for r in sc:
            cluster_to_supercluster[r] = sc_id
    return cluster_to_supercluster


def compute_loads(instance: GVRPInstance, superclusters: list[list[int]]) -> list[int]:
    return [sum(instance.cluster_demands[r] for r in sc) for sc in superclusters]


def heuristic_tsp_route(
    customers: list[int],
    depot: int,
    node_dist,
):
    # Fixed seed so route evaluation stays deterministic.
    rng = random.Random(0)
    route = nearest_neighbor_tour(customers, depot, node_dist, rng)
    route = two_opt(route, node_dist)
    cost = route_length(route, node_dist)
    return route, cost


def evaluate_supercluster_route(
    instance: GVRPInstance,
    sc_clusters: list[int],
    node_dist,
):
    customers = customers_of_supercluster(instance, sc_clusters)
    cluster_customer_sets = [instance.clusters[r] for r in sc_clusters]

    if ROUTING_VARIANT == "soft":
        hard_cluster_sets = None
    elif ROUTING_VARIANT == "hard":
        hard_cluster_sets = cluster_customer_sets
    else:
        raise ValueError(f"Unknown ROUTING_VARIANT: {ROUTING_VARIANT}")

    if ROUTING_SOLVER == "exact":
        route, cost = exact_tsp_gurobi(
            customers,
            instance.depot,
            node_dist,
            cluster_customer_sets=hard_cluster_sets,
        )

    elif ROUTING_SOLVER == "heuristic":
        if ROUTING_VARIANT == "hard":
            raise NotImplementedError(
                "Hard CluVRP is currently only supported with exact routing."
            )
        route, cost = heuristic_tsp_route(customers, instance.depot, node_dist)

    elif ROUTING_SOLVER == "hybrid":
        if len(customers) <= HYBRID_EXACT_MAX_CUSTOMERS:
            route, cost = exact_tsp_gurobi(
                customers,
                instance.depot,
                node_dist,
                cluster_customer_sets=hard_cluster_sets,
            )
        else:
            if ROUTING_VARIANT == "hard":
                raise NotImplementedError(
                    "Hard CluVRP with hybrid routing is only supported when the "
                    "supercluster is small enough for exact routing."
                )
            route, cost = heuristic_tsp_route(customers, instance.depot, node_dist)

    else:
        raise ValueError(f"Unknown ROUTING_SOLVER: {ROUTING_SOLVER}")

    return route, cost, customers


def build_solution_from_superclusters(
    instance: GVRPInstance,
    superclusters: list[list[int]],
    node_dist,
) -> Solution:
    superclusters = [sorted(sc) for sc in superclusters]
    loads = compute_loads(instance, superclusters)
    cluster_to_supercluster = build_cluster_to_supercluster(superclusters)

    routes = []
    route_costs = []
    supercluster_customers = []

    for sc in superclusters:
        route, cost, customers = evaluate_supercluster_route(instance, sc, node_dist)
        routes.append(route)
        route_costs.append(cost)
        supercluster_customers.append(customers)

    return Solution(
        superclusters=superclusters,
        loads=loads,
        cluster_to_supercluster=cluster_to_supercluster,
        supercluster_customers=supercluster_customers,
        routes=routes,
        route_costs=route_costs,
        total_cost=sum(route_costs),
    )


def reoptimize_affected_superclusters(
    instance: GVRPInstance,
    current_solution: Solution,
    new_superclusters: list[list[int]],
    affected_ids: list[int],
    node_dist,
) -> Solution:
    new_superclusters = [sorted(sc) for sc in new_superclusters]
    new_loads = current_solution.loads[:]
    new_routes = [r[:] for r in current_solution.routes]
    new_route_costs = current_solution.route_costs[:]
    new_supercluster_customers = [c[:] for c in current_solution.supercluster_customers]

    for sc_id in affected_ids:
        sc = new_superclusters[sc_id]
        load = feasible_load(instance, sc)
        if load > instance.capacity:
            raise ValueError(f"Infeasible move: load {load} exceeds capacity {instance.capacity}")

        route, cost, customers = evaluate_supercluster_route(instance, sc, node_dist)
        new_loads[sc_id] = load
        new_routes[sc_id] = route
        new_route_costs[sc_id] = cost
        new_supercluster_customers[sc_id] = customers

    return Solution(
        superclusters=new_superclusters,
        loads=new_loads,
        cluster_to_supercluster=build_cluster_to_supercluster(new_superclusters),
        supercluster_customers=new_supercluster_customers,
        routes=new_routes,
        route_costs=new_route_costs,
        total_cost=sum(new_route_costs),
    )