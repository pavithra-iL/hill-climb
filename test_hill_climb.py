"""Tests for the hill climbing optimizer."""

import math

from hill_climb import hill_climb, random_restart_hill_climb


def test_finds_peak_of_simple_parabola():
    # -(x - 2)^2 peaks at x = 2.
    result = hill_climb(lambda x: -((x - 2) ** 2), start=0.0, step_size=0.01)
    assert math.isclose(result.x, 2.0, abs_tol=0.05)
    assert result.value <= 0.0


def test_stops_at_local_maximum():
    result = hill_climb(lambda x: -(x ** 2), start=5.0, step_size=0.1)
    assert result.iterations > 0
    assert math.isclose(result.x, 0.0, abs_tol=0.2)


def test_random_restart_is_deterministic_with_seed():
    def f(x: float) -> float:
        return math.sin(x)

    a = random_restart_hill_climb(f, bounds=(-5, 5), seed=1, step_size=0.01)
    b = random_restart_hill_climb(f, bounds=(-5, 5), seed=1, step_size=0.01)
    assert math.isclose(a.x, b.x)
    assert math.isclose(a.value, b.value)


def test_invalid_arguments_raise():
    for bad in (0, -1):
        try:
            hill_climb(lambda x: x, start=0.0, step_size=bad)
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError for step_size")


if __name__ == "__main__":
    test_finds_peak_of_simple_parabola()
    test_stops_at_local_maximum()
    test_random_restart_is_deterministic_with_seed()
    test_invalid_arguments_raise()
    print("All tests passed.")
