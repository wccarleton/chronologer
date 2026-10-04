"""Predictive integration and WAIC checks without sampling."""
import numpy as np
import pytest
import xarray as xr
from scipy.special import logsumexp
from scipy.stats import norm, uniform

from chronologer import Phase
from chronologer.phases import waic
from chronologer.phase_stats import _loglik
from chronologer.distributions import calrcarbon
from test_curve_references import curve


def test_predictive_integrals():
    mu, scale = np.array([-50.]), np.array([8.])
    observation = norm(-48, 3)
    assert _loglik(observation, 'normal', mu, scale)[0] == pytest.approx(norm.logpdf(-48, -50, np.hypot(8, 3)))
    expected = (norm.cdf(-46, -48, 3) - norm.cdf(-54, -48, 3)) / 8
    assert np.exp(_loglik(observation, 'uniform', mu, scale))[0] == pytest.approx(expected)
    assert np.exp(_loglik(uniform(-52, 4), 'uniform', mu, scale))[0] == pytest.approx(1 / 8)
    assert np.exp(_loglik(uniform(-52, 4), 'normal', mu, scale))[0] == pytest.approx((norm.cdf(-48, -50, 8) - norm.cdf(-52, -50, 8)) / 4)
    # Linear calibration fixture: existing curve error adds in quadrature.
    radiocarbon = calrcarbon(curve(), -48, 3)
    error = np.hypot(3, 2)
    assert _loglik(radiocarbon, 'normal', mu, scale)[0] == pytest.approx(norm.logpdf(-48, -50, np.hypot(8, error)), abs=1e-7)
    expected = (norm.cdf(-46, -48, error) - norm.cdf(-54, -48, error)) / 8
    assert np.exp(_loglik(radiocarbon, 'uniform', mu, scale))[0] == pytest.approx(expected, rel=1e-6)


def test_waic_matches_pointwise_definition():
    mu = np.array([[[-50.], [-49.], [-51.], [-48.]]])
    posterior = xr.Dataset({'mu': (('chain', 'draw', 'phase'), mu),
                            'scale': (('chain', 'draw', 'phase'), np.full_like(mu, 8.))},
                           coords={'phase': ['A']})
    observations = [norm(-48, 3), norm(-55, 4)]
    score = waic([{'label': 'A'}, {'label': 'A'}], {'A': Phase()},
                 measurements=observations, posterior=posterior)
    ll = np.stack([norm.logpdf(e.mean(), mu.reshape(-1), np.hypot(8, e.std())) for e in observations], axis=-1)
    penalty = ll.var(axis=0)
    pointwise = logsumexp(ll, axis=0) - np.log(4) - penalty
    assert score['waic'] == pytest.approx(-2 * pointwise.sum())
    assert score['p_waic'] == pytest.approx(penalty.sum())
    assert score['se'] == pytest.approx(2 * np.sqrt(2 * pointwise.var()))
    assert score['n_events'] == 2 and score['n_samples'] == 4
