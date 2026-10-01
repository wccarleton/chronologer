import numpy as np
import pytest
from chronologer import hdi, calibrate, load_calcurve


@pytest.mark.parametrize("scale", [1e-12, 1, 1e12])
def test_hdi_coverage_and_scale_invariance(scale):
    t = np.arange(7, dtype=float)
    w = np.array([1, 4, 8, 2, 9, 5, 1], dtype=float)
    intervals = hdi(t, w * scale, .8)
    assert intervals == [(1, 2), (4, 5)]
    mask = np.zeros(len(t), dtype=bool)
    for a, b in intervals:
        mask |= (t >= a) & (t <= b)
    assert w[mask].sum() / w.sum() >= .8
    assert (w[mask].sum() - min(w[mask])) / w.sum() < .8


def test_first_retained_gap_does_not_join_distinct_intervals():
    assert hdi([-10, -5, -4, -3], [1, 4, 4, 1], 1, grid_spacing=1) == [(-10, -10), (-5, -3)]


def test_cutoff_includes_dominant_cell_and_ties():
    assert hdi([0, 1, 2], [100, 1, 1], .95) == [(0, 0)]
    assert hdi([0, 1, 2, 3], [1, 2, 2, 1], .5) == [(1, 2)]


def test_normal_density_interval():
    t = np.linspace(-5, 5, 10001)
    intervals = hdi(t, np.exp(-.5 * t*t))
    np.testing.assert_allclose(intervals, [(-1.96, 1.96)], atol=.002)


def test_narrow_real_batch_has_valid_coverage():
    results = calibrate([-(1800 + i*137) for i in range(2, 24)], [25]*22,
                        load_calcurve('intcal20', quiet=True), as_pandas=False)
    for result in results:
        t, w = result['t_values'], result['pdf_values']
        mask = np.zeros(len(t), dtype=bool)
        for a, b in result['hdi_intervals']:
            mask |= (t >= a) & (t <= b)
        assert w[mask].sum()/w.sum() >= .95 - 1e-12


@pytest.mark.parametrize('t,w,p', [([], [], .95), ([0,1], [0,0], .95),
    ([1,0], [1,1], .95), ([0,1], [-1,2], .95), ([0,1], [1,1], 0)])
def test_invalid_inputs(t,w,p):
    with pytest.raises(ValueError):
        hdi(t,w,p)
