"""Tests for the simulated annealing optimizer."""

import math

from annealing import anneal
from objectives import bumpy, gaussian, parabola


def test_finds_parabola_peak():
    result = anneal(parabola, start=-5.0, bounds=(-10, 10), seed=1)
    assert math.isclose(result.x, 2.0, abs_tol=0.3)


def test_respects_bounds():
    result = anneal(gaussian, start=0.0, bounds=(-1.0, 1.0), seed=2)
    assert -1.0 <= result.x <= 1.0


def test_deterministic_with_seed():
    a = anneal(bumpy, start=-4.0, bounds=(-5, 5), seed=42)
    b = anneal(bumpy, start=-4.0, bounds=(-5, 5), seed=42)
    assert math.isclose(a.x, b.x)
    assert math.isclose(a.value, b.value)


def test_escapes_a_poor_local_optimum():
    # Starting on a lower hump of a multi-modal function, annealing should
    # still reach a value near the global max (~1.19 for bumpy on [-5, 5]).
    result = anneal(bumpy, start=-4.0, bounds=(-5, 5), seed=3)
    assert result.value > 1.0


def test_invalid_arguments_raise():
    cases = [
        {"step_size": 0},
        {"initial_temperature": 0},
        {"cooling": 1.0},
        {"cooling": 0.0},
        {"max_iterations": 0},
    ]
    for kwargs in cases:
        try:
            anneal(parabola, start=0.0, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {kwargs}")


if __name__ == "__main__":
    test_finds_parabola_peak()
    test_respects_bounds()
    test_deterministic_with_seed()
    test_escapes_a_poor_local_optimum()
    test_invalid_arguments_raise()
    print("All annealing tests passed.")
