"""Read GVRP instance files from disk."""

from __future__ import annotations

from pathlib import Path
from typing import List

from src.cluvrp.types import GVRPInstance


def read_gvrp_instance(file_path: Path) -> GVRPInstance:
    lines = [line.strip() for line in file_path.read_text().splitlines() if line.strip()]

    name = ""
    comment = ""
    dimension = None
    vehicles = None
    n_clusters = None
    capacity = None

    coords = {}
    clusters = {}
    cluster_demands = {}

    i = 0
    while i < len(lines):
        line = lines[i]

        if line.startswith("NAME"):
            name = line.split(":", 1)[1].strip()

        elif line.startswith("COMMENT"):
            comment = line.split(":", 1)[1].strip()

        elif line.startswith("DIMENSION"):
            dimension = int(line.split(":", 1)[1].strip())

        elif line.startswith("VEHICLES"):
            vehicles = int(line.split(":", 1)[1].strip())

        elif line.startswith("GVRP_SETS"):
            n_clusters = int(line.split(":", 1)[1].strip())

        elif line.startswith("CAPACITY"):
            capacity = int(line.split(":", 1)[1].strip())

        elif line == "NODE_COORD_SECTION":
            i += 1
            while i < len(lines) and lines[i] != "GVRP_SET_SECTION":
                node, x, y = lines[i].split()
                coords[int(node)] = (float(x), float(y))
                i += 1
            continue

        elif line == "GVRP_SET_SECTION":
            i += 1
            while i < len(lines) and lines[i] != "DEMAND_SECTION":
                parts = lines[i].split()
                cluster_id = int(parts[0])
                customers = [int(x) for x in parts[1:] if x != "-1"]
                clusters[cluster_id] = customers
                i += 1
            continue

        elif line == "DEMAND_SECTION":
            i += 1
            while i < len(lines) and lines[i] != "EOF":
                cluster_id, demand = lines[i].split()
                cluster_demands[int(cluster_id)] = int(demand)
                i += 1
            continue

        i += 1

    if dimension is None or vehicles is None or n_clusters is None or capacity is None:
        raise ValueError(f"Failed to parse header in {file_path}")

    return GVRPInstance(
        name=name or file_path.stem,
        comment=comment,
        dimension=dimension,
        vehicles=vehicles,
        n_clusters=n_clusters,
        capacity=capacity,
        coords=coords,
        clusters=clusters,
        cluster_demands=cluster_demands,
        depot=1,
    )


def get_instance_path(instance_name: str, instance_dirs: List[Path]) -> Path:
    for folder in instance_dirs:
        candidate = folder / f"{instance_name}.gvrp"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find {instance_name}.gvrp in any configured instance folder.")