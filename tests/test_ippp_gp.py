import numpy as np
import pytest
import pytensor
import pytensor.tensor as pt
from scipy.stats import norm, uniform
import xarray as xr

import chronologer as ch
from chronologer.models.ippp import build_gp, _grid_logp, GPFit
from chronologer.distributions import calrcarbon


def test_full_point_process_logp_and_gradient_include_empty_exposure():
    a = pt.dscalar('a')
    grid = np.linspace(-10, 0, 8)
    logp = _grid_logp([-8., -3.], grid, pt.ones(8) * a)
    evaluate = pytensor.function([a], [logp, pt.grad(logp, a)])
    value, grad = evaluate(.7)
    assert value == pytest.approx(2 * np.log(.7) - 10 * .7)
    assert grad == pytest.approx(2 / .7 - 10)
    empty = pytensor.function([a], _grid_logp(np.array([]), grid, pt.ones(8) * a))
    assert empty(.7) == pytest.approx(-7.)
    # Keeping all events but extending the observed empty window penalizes rate.
    wider = pytensor.function([a], _grid_logp([-8., -3.], np.linspace(-20, 0, 8), pt.ones(8) * a))
    assert wider(.7) - value == pytest.approx(-7.)


def test_grid_likelihood_matches_its_interpolated_intensity():
    grid = np.array([-10., -7., -2., 0.])
    rate = np.array([.1, .8, .3, .6])
    times = np.array([-10., -8., -4., 0.])
    symbolic = pt.dvector('rate')
    logp = _grid_logp(times, grid, symbolic)
    f = pytensor.function([symbolic], [logp, pt.grad(logp, symbolic)])
    expected = np.log(np.interp(times, grid, rate)).sum() - np.trapezoid(rate, grid)
    value, gradient = f(rate)
    assert value == pytest.approx(expected)
    numerical = [(f(rate + np.eye(4)[i] * 1e-6)[0] - f(rate - np.eye(4)[i] * 1e-6)[0]) / 2e-6 for i in range(4)]
    np.testing.assert_allclose(gradient, numerical, rtol=1e-6)


@pytest.mark.parametrize('params', [{}, {'start': -10}, {'end': 0}, {'start': None, 'end': 0},
                                  {'start': 0, 'end': -10}, {'start': -10, 'end': -10},
                                  {'start': -np.inf, 'end': 0}])
def test_observation_period_must_be_explicit_finite_and_ordered(params):
    with pytest.raises((ValueError, TypeError)):
        ch.fit([norm(-5, 1)], model=ch.models.ippp.gp, params=params)


def test_disjoint_measurements_and_exact_events_outside_window_rejected():
    for data in [[-11.], [uniform(-20, 3)]]:
        with pytest.raises(ValueError):
            build_gp(data, start=-10, end=0)


def test_uncertain_and_mixed_curve_graph_compiles_with_finite_gradients():
    t = np.linspace(-20, 0, 21)
    curve = dict(calbp=t, c14bp=t, c14_sigma=np.full_like(t, .1))
    shifted = {**curve, 'c14bp': t + .5}
    data = [calrcarbon(curve, -8., .2), calrcarbon(shifted, -5., .2), norm(-6., .3), uniform(-4., 1), -3.]
    model = build_gp(data, start=-12., end=-1., grid_size=8)
    point = model.initial_point()
    assert np.isfinite(model.compile_logp()(point))
    assert np.isfinite(model.compile_dlogp()(point)).all()
    assert model.ippp_spec['start'] == -12. and model.ippp_spec['end'] == -1.
    assert model.ippp_spec['length_scale_median'] == pytest.approx(11 / 5)


def test_empty_observed_window_model_compiles():
    model = build_gp([], start=-10., end=0., grid_size=8)
    point = model.initial_point()
    assert np.isfinite(model.compile_logp()(point))
    assert np.isfinite(model.compile_dlogp()(point)).all()


@pytest.mark.parametrize('exact', [False, True])
@pytest.mark.parametrize('cores', [1, 2])
def test_gp_nuts_and_result_intensity_are_not_area_normalized(exact, cores):
    progress = []
    data = [-8., -5., -3.] if exact else [norm(-8, .2), norm(-5, .3), norm(-3, .2)]
    result = ch.fit(data, model=ch.models.ippp.gp,
                    params={'start': -10., 'end': 0., 'grid_size': 8},
                    mcmc_config={'draws': 4, 'tune': 4, 'chains': cores, 'cores': cores, 'random_seed': 14},
                    progress_callback=progress.append)
    assert isinstance(result, GPFit) and isinstance(result.posterior, xr.DataTree)
    posterior = result.posterior['posterior']
    assert posterior['tau'].shape == (cores, 4, 3)
    assert posterior['intensity'].shape == (cores, 4, 8)
    assert all(np.isfinite(v).all() for v in result.intensity.values())
    assert np.all(result.intensity['rate_values'] > 0)
    assert np.all(result.intensity['lower_values'] <= result.intensity['upper_values'])
    assert np.trapezoid(result.intensity['rate_values'], result.intensity['t_values']) == pytest.approx(
        float(posterior['integrated_intensity'].mean()))
    assert progress[-1] == dict(stage='Evaluating intensity', completed=8 * cores, total=8 * cores)
    counts = [p['completed'] for p in progress]
    assert counts == sorted(counts)
