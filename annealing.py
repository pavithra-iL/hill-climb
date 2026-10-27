"""Simulated annealing optimizer.

Unlike plain hill climbing, simulated annealing can accept worse moves
early on, which lets it escape local maxima. The willingness to accept
worse moves shrinks as the "temperature" cools, so the search settles
down over time.
"""

from __future__ import annotations

import math
import random
from typing import Callable

from hill_climb import Result


def anneal(
    objective: Callable[[float], float],
    start: float,
    *,
    bounds: tuple[float, float] | None = None,
    step_size: float = 0.5,
    initial_temperature: float = 10.0,
    cooling: float = 0.995,
    max_iterations: int = 10_000,
    seed: int | None = None,
) -> Result:
    """Maximize ``objective`` via simulated annealing.

    At each step a random neighbor within ``step_size`` is proposed. A
    better neighbor is always accepted; a worse one is accepted with
    probability ``exp(delta / temperature)``. The temperature is scaled by
    ``cooling`` each iteration (0 < cooling < 1).

    ``bounds`` optionally clamps the search to ``(low, high)``. The best
    point ever visited is returned, not merely the final one.
    """
    if step_size <= 0:
        raise ValueError("step_size must be positive")
    if initial_temperature <= 0:
        raise ValueError("initial_temperature must be positive")
    if not 0.0 < cooling < 1.0:
        raise ValueError("cooling must be in the open interval (0, 1)")
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")

    low = high = None
    if bounds is not None:
        low, high = bounds
        if low >= high:
            raise ValueError("bounds must be (low, high) with low < high")
        start = min(max(start, low), high)

    rng = random.Random(seed)

    current = start
    current_value = objective(current)

    best = current
    best_value = current_value

    temperature = initial_temperature

    for iteration in range(1, max_iterations + 1):
        candidate = current + rng.uniform(-step_size, step_size)
        if low is not None:
            candidate = min(max(candidate, low), high)

        candidate_value = objective(candidate)
        delta = candidate_value - current_value

        if delta >= 0 or rng.random() < math.exp(delta / temperature):
            current, current_value = candidate, candidate_value
            if current_value > best_value:
                best, best_value = current, current_value

        temperature *= cooling
        if temperature < 1e-12:
            return Result(best, best_value, iteration)

    return Result(best, best_value, max_iterations)


if __name__ == "__main__":
    from objectives import bumpy

    result = anneal(bumpy, start=-4.0, bounds=(-5.0, 5.0), seed=7)
    print(f"Simulated annealing: x={result.x:.4f} value={result.value:.4f}")
