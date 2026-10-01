import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt
import pytest
from scipy.stats import norm, uniform

import chronologer
from chronologer.density import _measurement
from chronologer.distributions import calrcarbon


@pytest.mark.parametrize('k', [1, 2, 5])
def test_construct_logp_gradient_and_nuts(k):
    model = chronologer.build_gaussian_mixture([norm(-20, 2), norm(-10, 1)], k)
    point = model.initial_point()
    assert np.isfinite(model.compile_logp()(point))
    assert np.isfinite(model.compile_dlogp()(point)).all()
    with model:
        trace = pm.sample(draws=4, tune=4, chains=1, cores=1, init='adapt_diag',
                          random_seed=24, progressbar=False, compute_convergence_checks=False)
    posterior = trace['posterior'].to_dataset()
    assert posterior['tau'].shape == (1, 4, 2)
    assert (np.diff(posterior['means'].values, axis=-1) > 0).all()
    weights = posterior['weights'].values
    assert (weights >= 0).all()
    np.testing.assert_allclose(weights.sum(axis=-1), 1)


def test_public_fit_density_and_requested_grid():
    progress = []
    fit = chronologer.fit_gaussian_mixture([norm(-20, 2), norm(-10, 1)], 2,
        draws=6, tune=6, chains=1, random_seed=24, progress_callback=progress.append)
    data = fit.posterior['posterior'].to_dataset()
    means, scales = data['means'].values, data['scales'].values
    grid = np.linspace(np.min(means - 8*scales), np.max(means + 8*scales), 20000)
    density = fit.evaluate_density(grid)
    assert np.array_equal(density['t_values'], grid)
    for key in ('pdf_values', 'lower_values', 'upper_values'):
        assert np.isfinite(density[key]).all() and (density[key] >= 0).all()
    assert np.all(density['lower_values'] <= density['upper_values'])
    assert np.trapezoid(density['pdf_values'], grid) == pytest.approx(1, abs=.001)
    assert progress[-1]['completed'] == progress[-1]['total'] == 12
    assert [p['completed'] for p in progress] == sorted(p['completed'] for p in progress)
    # Direct weighted Normal PDFs, not calibrated PDFs or a moment approximation.
    expected = sum(data.weights.values[0, j, k] * norm.pdf(grid, means[0, j, k], scales[0, j, k])
                   for j in range(6) for k in range(2)) / 6
    np.testing.assert_allclose(density['pdf_values'], expected)


def test_measurement_bridge_matches_existing_distributions_and_derivatives(monkeypatch):
    monkeypatch.setattr(calrcarbon, '_interp_mean', None)
    monkeypatch.setattr(calrcarbon, '_interp_error', None)
    curve = dict(calbp=np.linspace(-100, 0, 11), c14bp=np.linspace(-101, -1, 11),
                 c14_sigma=np.linspace(1, 2, 11))
    radiocarbon = calrcarbon(curve, -50, 3)
    for event in [radiocarbon, norm(-50, 3), uniform(-70, 40)]:
        logp, _, _ = _measurement(event)
        t = pt.dscalar('t')
        f = pytensor.function([t], [logp(t), pt.grad(logp(t), t)])
        value, gradient = f(-48.)
        assert value == pytest.approx(float(event.logpdf(-48.)))
        step = 1e-4
        expected = (event.logpdf(-48+step)-event.logpdf(-48-step))/(2*step)
        assert gradient == pytest.approx(float(expected), abs=1e-6)
    model = chronologer.build_gaussian_mixture([radiocarbon, norm(-40, 3)], 2)
    assert np.isfinite(model.compile_logp()(model.initial_point()))
    assert np.isfinite(model.compile_dlogp()(model.initial_point())).all()


@pytest.mark.parametrize('k', [0, 21, 1.5, True])
def test_invalid_maximum(k):
    with pytest.raises(ValueError):
        chronologer.build_gaussian_mixture([norm(-20, 2)], k)
