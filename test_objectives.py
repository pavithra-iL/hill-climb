"""Tests for the objective function library."""

import math

from objectives import OBJECTIVES, bumpy, gaussian, parabola, sawtooth_envelope


def test_parabola_peaks_at_two():
    assert parabola(2.0) == 0.0
    assert parabola(1.0) < parabola(2.0)
    assert parabola(3.0) < parabola(2.0)


def test_gaussian_peaks_at_mu():
    assert math.isclose(gaussian(0.0), 1.0)
    assert gaussian(1.0, mu=1.0) > gaussian(0.0, mu=1.0)


def test_gaussian_rejects_bad_sigma():
    for bad in (0.0, -1.0):
        try:
            gaussian(0.0, sigma=bad)
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError for sigma")


def test_sawtooth_peaks_near_origin():
    assert sawtooth_envelope(0.0) >= sawtooth_envelope(2.0)


def test_registry_is_complete_and_callable():
    expected = {"parabola", "bumpy", "gaussian", "sawtooth_envelope"}
    assert set(OBJECTIVES) == expected
    for fn in OBJECTIVES.values():
        assert isinstance(fn(0.5), float)
    # bumpy is referenced to keep the import meaningful.
    assert isinstance(bumpy(1.0), float)


if __name__ == "__main__":
    test_parabola_peaks_at_two()
    test_gaussian_peaks_at_mu()
    test_gaussian_rejects_bad_sigma()
    test_sawtooth_peaks_near_origin()
    test_registry_is_complete_and_callable()
    print("All objective tests passed.")
