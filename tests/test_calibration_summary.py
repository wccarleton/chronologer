"""Weighted mean regression tests; calibration density values remain untouched."""

import numpy as np
import pytest

import chronologer
import chronologer.calibration as calibration


def test_real_mean_is_within_retained_support_and_arrays_are_preserved():
    result = chronologer.calibrate(
        [-3240], [25], chronologer.load_calcurve("intcal20", quiet=True),
        as_pandas=False,
    )[0]
    t, weights = result["t_values"], result["pdf_values"]
    assert t[0] <= result["mean"] <= t[-1]
    assert result["mean"] == pytest.approx(np.average(t, weights=weights))
    distribution = result["calibrated_distribution"]
    grid = np.linspace(distribution.a, distribution.b, 10000)
    raw = distribution.pdf(grid)
    np.testing.assert_array_equal(t, grid[raw > 1e-7])
    np.testing.assert_array_equal(weights, raw[raw > 1e-7])


@pytest.mark.parametrize("as_pandas", [False, True])
def test_mean_is_invariant_to_weight_scale_with_trimmed_gaps(monkeypatch, as_pandas):
    class Distribution:
        a, b = -100, 0

        def __init__(self, curve, c14_mean, c14_err):
            self.scale = c14_err

        def pdf(self, t):
            return self.scale * 1e-5 * np.where((t < -70) | (t > -20), t + 101, 0)

    monkeypatch.setattr(calibration, "calrcarbon", Distribution)
    result = calibration.calibrate([0, 0, 0], [0.1, 1, 10], {}, tol=0, as_pandas=as_pandas)
    means = result["Mean Calibrated Age (BP)"] if as_pandas else [r["mean"] for r in result]
    np.testing.assert_allclose(means, np.repeat(means[0], 3), rtol=1e-14)
    assert -100 <= means[0] <= 0
