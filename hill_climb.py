"""Hill climbing optimization algorithm.

A simple local-search optimizer that iteratively moves toward better
neighboring solutions. Includes both basic and random-restart variants.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable


@dataclass
class Result:
    """Outcome of an optimization run."""

    x: float
    value: float
    iterations: int


def hill_climb(
    objective: Callable[[float], float],
    start: float,
    *,
    step_size: float = 0.1,
    max_iterations: int = 1000,
) -> Result:
    """Maximize ``objective`` starting from ``start``.

    At each step it looks one ``step_size`` to the left and right and moves
    to whichever neighbor is better. It stops when neither neighbor improves
    on the current position (a local maximum) or when ``max_iterations`` is
    reached.
    """
    if step_size <= 0:
        raise ValueError("step_size must be positive")
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")

    current = start
    current_value = objective(current)

    for iteration in range(1, max_iterations + 1):
        left = current - step_size
        right = current + step_size
        left_value = objective(left)
        right_value = objective(right)

        if right_value > current_value and right_value >= left_value:
            current, current_value = right, right_value
        elif left_value > current_value:
            current, current_value = left, left_value
        else:
            return Result(current, current_value, iteration)

    return Result(current, current_value, max_iterations)


def random_restart_hill_climb(
    objective: Callable[[float], float],
    *,
    bounds: tuple[float, float],
    restarts: int = 10,
    step_size: float = 0.1,
    max_iterations: int = 1000,
    seed: int | None = None,
) -> Result:
    """Run ``hill_climb`` from several random starts and keep the best.

    Helps escape poor local maxima by sampling multiple starting points
    within ``bounds``.
    """
    if restarts <= 0:
        raise ValueError("restarts must be positive")

    low, high = bounds
    if low >= high:
        raise ValueError("bounds must be (low, high) with low < high")

    rng = random.Random(seed)
    best: Result | None = None

    for _ in range(restarts):
        start = rng.uniform(low, high)
        result = hill_climb(
            objective,
            start,
            step_size=step_size,
            max_iterations=max_iterations,
        )
        if best is None or result.value > best.value:
            best = result

    assert best is not None  # restarts > 0 guarantees at least one result
    return best


if __name__ == "__main__":
    # Maximize a bumpy function with several local maxima.
    import math

    def f(x: float) -> float:
        return math.sin(x) + 0.3 * math.sin(3 * x)

    basic = hill_climb(f, start=0.0, step_size=0.01)
    print(f"Basic hill climb:  x={basic.x:.4f} value={basic.value:.4f}")

    best = random_restart_hill_climb(
        f, bounds=(-5.0, 5.0), restarts=25, step_size=0.01, seed=42
    )
    print(f"Random restart:    x={best.x:.4f} value={best.value:.4f}")
