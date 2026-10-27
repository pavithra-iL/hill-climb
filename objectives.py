"""A small library of 1-D objective functions for optimization.

Each function maps a float to a float and is meant to be maximized.
These are handy for exercising the hill climbing optimizer.
"""

from __future__ import annotations

import math


def parabola(x: float) -> float:
    """Smooth single peak at x = 2."""
    return -((x - 2.0) ** 2)


def bumpy(x: float) -> float:
    """Several local maxima — good for testing random restarts."""
    return math.sin(x) + 0.3 * math.sin(3.0 * x)


def gaussian(x: float, *, mu: float = 0.0, sigma: float = 1.0) -> float:
    """Classic bell curve peaking at ``mu``."""
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    return math.exp(-((x - mu) ** 2) / (2.0 * sigma ** 2))


def sawtooth_envelope(x: float) -> float:
    """A decaying oscillation; peak near the origin."""
    return math.cos(x) * math.exp(-0.1 * abs(x))


# Registry so callers can look functions up by name.
OBJECTIVES = {
    "parabola": parabola,
    "bumpy": bumpy,
    "gaussian": gaussian,
    "sawtooth_envelope": sawtooth_envelope,
}
