"""Small sanity check for construction output shape."""

import random

from src.cluvrp.construction.superclusters import construct_superclusters
from src.cluvrp.types import GVRPInstance


def test_construct_superclusters_basic():
    instance = GVRPInstance(
        name="X",
        comment="",
        dimension=5,
        vehicles=2,
        n_clusters=4,
        capacity=10,
        coords={1: (0, 0), 2: (1, 0), 3: (2, 0), 4: (0, 2), 5: (2, 2)},
        clusters={1: [2], 2: [3], 3: [4], 4: [5]},
        cluster_demands={1: 2, 2: 2, 3: 2, 4: 2},
        depot=1,
    )

    superclusters, loads, mapping = construct_superclusters(instance, random.Random(1), 0.15)

    assert len(superclusters) == instance.vehicles
    assert len(loads) == instance.vehicles
    assert set(mapping.keys()) == set(instance.clusters.keys())