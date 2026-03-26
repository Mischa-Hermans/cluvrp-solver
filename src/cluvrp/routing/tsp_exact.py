"""Exact TSP solver for one supercluster using Gurobi."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import gurobipy as gp
from gurobipy import GRB

from src.cluvrp.core.evaluation import route_length


def exact_tsp_gurobi(
    customers: List[int],
    depot: int,
    dist: Dict[int, Dict[int, float]],
    time_limit: Optional[float] = None,
    cluster_customer_sets: Optional[List[List[int]]] = None,
) -> Tuple[List[int], float]:
    if not customers:
        return [depot, depot], 0.0

    if len(customers) == 1:
        route = [depot, customers[0], depot]
        return route, route_length(route, dist)

    nodes = [depot] + customers
    n = len(nodes)
    idx_to_node = {i: node for i, node in enumerate(nodes)}
    node_to_idx = {node: i for i, node in idx_to_node.items()}

    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()

    model = gp.Model("tsp", env=env)
    if time_limit is not None:
        model.Params.TimeLimit = time_limit
    model.Params.LazyConstraints = 1

    x = {}
    for i in range(n):
        for j in range(i + 1, n):
            x[i, j] = model.addVar(
                vtype=GRB.BINARY,
                obj=dist[idx_to_node[i]][idx_to_node[j]],
                name=f"x_{i}_{j}",
            )

    model.update()

    # Degree constraints
    for i in range(n):
        expr = gp.quicksum(x[min(i, j), max(i, j)] for j in range(n) if j != i)
        model.addConstr(expr == 2, name=f"deg_{i}")

    # Hard-CluVRP cluster contiguity constraints:
    # each cluster must be entered once and exited once,
    # which in an undirected Hamiltonian cycle means exactly two cut edges.
    if cluster_customer_sets is not None:
        all_idx = set(range(n))

        for cluster_id, cluster_customers in enumerate(cluster_customer_sets):
            cluster_idx = {node_to_idx[c] for c in cluster_customers if c in node_to_idx}

            # Ignore empty clusters just in case
            if not cluster_idx:
                continue

            outside_idx = all_idx - cluster_idx

            expr = gp.quicksum(
                x[min(i, j), max(i, j)]
                for i in cluster_idx
                for j in outside_idx
            )
            model.addConstr(expr == 2, name=f"cluster_cut_{cluster_id}")

    def selected_edges():
        return [(i, j) for (i, j), var in x.items() if var.X > 0.5]

    def find_subtour(edges):
        adj = {i: [] for i in range(n)}
        for i, j in edges:
            adj[i].append(j)
            adj[j].append(i)

        unvisited = set(range(n))
        shortest = None

        while unvisited:
            start = next(iter(unvisited))
            comp = []
            stack = [start]
            while stack:
                u = stack.pop()
                if u in unvisited:
                    unvisited.remove(u)
                    comp.append(u)
                    for v in adj[u]:
                        if v in unvisited:
                            stack.append(v)
            if shortest is None or len(comp) < len(shortest):
                shortest = comp

        return shortest

    def callback(model, where):
        if where == GRB.Callback.MIPSOL:
            vals = model.cbGetSolution(model._x)
            edges = [(i, j) for (i, j), val in vals.items() if val > 0.5]
            cycle = find_subtour(edges)

            if len(cycle) < n:
                model.cbLazy(
                    gp.quicksum(
                        model._x[min(i, j), max(i, j)]
                        for idx_i, i in enumerate(cycle)
                        for j in cycle[idx_i + 1:]
                    ) <= len(cycle) - 1
                )

    model._x = x
    model.optimize(callback)

    if model.Status not in {GRB.OPTIMAL, GRB.TIME_LIMIT}:
        raise RuntimeError(f"Gurobi TSP solve failed with status {model.Status}")

    edges = selected_edges()
    adj = {i: [] for i in range(n)}
    for i, j in edges:
        adj[i].append(j)
        adj[j].append(i)

    tour_idx = [0]
    prev = None
    current = 0
    while True:
        neighbors = adj[current]
        nxt = neighbors[0] if neighbors[0] != prev else (neighbors[1] if len(neighbors) > 1 else None)
        if nxt is None:
            break
        if nxt == 0:
            tour_idx.append(0)
            break
        tour_idx.append(nxt)
        prev, current = current, nxt

    route = [idx_to_node[i] for i in tour_idx]
    cost = route_length(route, dist)
    return route, cost