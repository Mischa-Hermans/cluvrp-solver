"""Small sanity check for neighborhood helper output."""

from src.cluvrp.neighborhoods.helpers import copy_superclusters


def test_copy_superclusters():
    sc = [[1, 2], [3]]
    copied = copy_superclusters(sc)
    copied[0].append(99)

    assert sc != copied
    assert sc == [[1, 2], [3]]