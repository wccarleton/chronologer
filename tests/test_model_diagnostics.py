import numpy as np
import xarray as xr
from scipy.stats import norm, uniform
from chronologer.model_diagnostics import model_diagnostics, waic_from_log_likelihood


def test_waic_formula_and_warning():
    values = np.array([[-1., -2.], [-3., -2.]])
    score = waic_from_log_likelihood(values, n_events=2, likelihood='event_marginal')
    expected = np.log(np.exp(values).mean(axis=0)).sum() - values.var(axis=0).sum()
    assert np.isclose(score['waic'], -2 * expected)
    assert score['warning'] and score['p_waic'] == 1


def test_mixture_integrates_measurement_not_latent_date():
    ds = xr.Dataset({k: (('chain', 'draw', 'component'), np.array(v).reshape(1, 2, 1))
                     for k, v in dict(means=[0., 1.], scales=[2., 3.], weights=[1., 1.]).items()})
    events = [norm(1., .5), uniform(-1., 2.)]
    score = model_diagnostics(ds, events, model='mixture')
    logp = np.stack([norm.logpdf(1., [0., 1.], np.sqrt(np.array([2., 3.])**2 + .5**2)),
                    np.log((norm.cdf(1., [0., 1.], [2., 3.]) - norm.cdf(-1., [0., 1.], [2., 3.])) / 2)], axis=1)
    expected = waic_from_log_likelihood(logp, n_events=2, likelihood='event_marginal')
    assert np.isclose(score['waic'], expected['waic'])


def test_ippp_includes_count_and_empty_window():
    ds = xr.Dataset({'intensity': (('chain', 'draw', 'grid'), [[[1., 1., 1., 1.], [2., 2., 2., 2.]]])})
    score = model_diagnostics(ds, [uniform(0., 2.)], model='ippp_gp', grid=[0., 1., 2., 3.])
    expected = waic_from_log_likelihood(np.array([-3., np.log(2.) - 6.])[:, None],
                                         n_events=1, likelihood='observation_window')
    assert np.isclose(score['waic'], expected['waic'])
    assert score['n_units'] == 1 and score['se'] is None
    empty = model_diagnostics(ds, [], model='ippp_gp', grid=[0., 1., 2., 3.])
    assert empty['n_events'] == 0 and np.isfinite(empty['waic'])


def test_single_bounds_and_insufficient_draws():
    ds = xr.Dataset({'tau_mu': (('chain', 'draw'), [[0., 1.]]),
                     'tau_sd': (('chain', 'draw'), [[1., 2.]])})
    score = model_diagnostics(ds, [uniform(-1., 2.)], model='single_density', lower=-1., upper=1.)
    assert np.isclose(score['elpd_waic'], -np.log(2))
    assert score['p_waic'] < 1e-15
    assert 'unavailable' in model_diagnostics(ds.isel(draw=[0]), [norm()], model='single_density', lower=-1., upper=1.)


def test_radiocarbon_uses_each_curve_and_its_uncertainty():
    from chronologer.distributions import calrcarbon
    grid = np.linspace(-100., 100., 41)
    events = [calrcarbon(dict(calbp=grid, c14bp=grid + shift, c14_sigma=np.ones(41)),
                        c14_mean=2., c14_err=2.) for shift in (0., 10.)]
    ds = xr.Dataset({k: (('chain', 'draw', 'component'), np.array(v).reshape(1, 2, 1))
                     for k, v in dict(means=[0., 1.], scales=[2., 3.], weights=[1., 1.]).items()})
    score = model_diagnostics(ds, events, model='mixture')
    values = np.stack([norm.logpdf(2. - shift, [0., 1.], np.sqrt(np.array([2., 3.])**2 + 5))
                       for shift in (0., 10.)], axis=1)
    expected = waic_from_log_likelihood(values, n_events=2, likelihood='event_marginal')
    assert np.isclose(score['waic'], expected['waic'])
