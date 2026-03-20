"""Small sanity check for checkpoint extraction."""

from src.cluvrp.tracking.checkpoints import best_cost_at_time
from src.cluvrp.tracking.history import initialize_history, record_step


def test_best_cost_at_time():
    history = initialize_history(100.0)
    record_step(history, 1.0, 95.0, 95.0, True, True, "move")
    record_step(history, 2.0, 97.0, 95.0, True, False, "move")

    assert best_cost_at_time(history, 0.5) == 100.0
    assert best_cost_at_time(history, 1.5) == 95.0
    assert best_cost_at_time(history, 3.0) == 95.0