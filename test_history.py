"""Tests for the run history recorder."""

from hill_climb import Result
from history import History


def _result(value: float) -> Result:
    return Result(x=value, value=value, iterations=10)


def test_add_and_len():
    h = History()
    assert len(h) == 0
    h.add("parabola", "basic", _result(1.0))
    h.add("bumpy", "basic", _result(2.0))
    assert len(h) == 2


def test_best_returns_highest_value():
    h = History()
    assert h.best() is None
    h.add("a", "basic", _result(1.0))
    h.add("b", "basic", _result(5.0))
    h.add("c", "basic", _result(3.0))
    best = h.best()
    assert best is not None
    assert best.objective == "b"
    assert best.value == 5.0


def test_round_trips_through_json(tmp_path):
    h = History()
    h.add("gaussian", "random restart x10", _result(0.9))
    path = tmp_path / "history.json"
    h.save(path)

    loaded = History.load(path)
    assert len(loaded) == 1
    assert loaded.records[0].objective == "gaussian"
    assert loaded.records[0].mode == "random restart x10"
    assert loaded.records[0].value == 0.9


if __name__ == "__main__":
    test_add_and_len()
    test_best_returns_highest_value()
    print("Run `pytest test_history.py` for the tmp_path-based test.")
    print("Core history tests passed.")
