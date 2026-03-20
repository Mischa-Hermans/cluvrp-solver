"""Small sanity check for feasibility logic."""

from src.cluvrp.core.feasibility import feasible_load
from src.cluvrp.types import GVRPInstance


def test_feasible_load():
    instance = GVRPInstance(
        name="X",
        comment="",
        dimension=3,
        vehicles=2,
        n_clusters=2,
        capacity=10,
        coords={1: (0, 0), 2: (1, 0), 3: (0, 1)},
        clusters={1: [2], 2: [3]},
        cluster_demands={1: 3, 2: 4},
        depot=1,
    )
    assert feasible_load(instance, [1, 2]) == 7