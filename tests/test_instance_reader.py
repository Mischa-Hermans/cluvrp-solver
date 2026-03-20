"""Small sanity check for instance loading."""

from pathlib import Path

from src.cluvrp.io.instance_reader import read_gvrp_instance


def test_reader_returns_instance(tmp_path: Path):
    text = """NAME : X
COMMENT : test
TYPE : GVRP
DIMENSION : 3
VEHICLES : 1
GVRP_SETS : 1
CAPACITY : 10
EDGE_WEIGHT_TYPE : EUC_2D
NODE_COORD_SECTION
1 0 0
2 1 0
3 0 1
GVRP_SET_SECTION
1 2 3 -1
DEMAND_SECTION
1 5
EOF
"""
    path = tmp_path / "X.gvrp"
    path.write_text(text)
    instance = read_gvrp_instance(path)

    assert instance.name == "X"
    assert instance.vehicles == 1
    assert instance.capacity == 10
    assert instance.cluster_demands[1] == 5