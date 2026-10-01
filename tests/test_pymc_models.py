"""Execution smoke tests, not convergence or posterior-accuracy tests.

Model structures follow example.py and density_model.ipynb; IPPP models use
the current public likelihood signatures in chronologer.models. Synthetic
curves avoid downloads and place initial dates strictly inside curve bins.
"""

import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt
import pytest
import xarray as xr

from chronologer.models import approx_integral, ippp_logp_lm, ippp_logp_sine
from chronologer.pymccarbon import compute_bin_index, interpolate_calcurve


CURVE = tuple(pt.as_tensor_variable(a) for a in (
    np.array([-4., -3., -2., -1., 0.]),
    np.array([-4.2, -3.1, -2.3, -1.1, -.2]),
    np.array([.12, .15, .1, .13, .12]),
))


@pytest.mark.parametrize('shape, values', [((), -2.5), ((1,), [-2.5]), ((2,), [-2.5, -1.5]), ((None,), [-2.5, -1.5])])
def test_symbolic_interpolation_construction_values_and_gradient(shape, values):
    tau = pt.tensor('tau', shape=shape, dtype='float64')
    index = compute_bin_index(tau, CURVE[0])
    mean, sigma = interpolate_calcurve(tau, *CURVE)
    gradient = pt.grad(pt.sum(mean + sigma), tau)
    evaluate = pytensor.function([tau], [index, mean, sigma, gradient])
    indices, means, sigmas, gradients = evaluate(np.asarray(values))
    grid, radiocarbon, errors = [v.data for v in CURVE]
    # Preserve the helper's existing scalar -> length-one output convention.
    expected_values = np.atleast_1d(values)
    np.testing.assert_array_equal(indices, np.searchsorted(grid, expected_values, side='right') - 1)
    np.testing.assert_allclose(means, np.interp(expected_values, grid, radiocarbon))
    np.testing.assert_allclose(sigmas, np.interp(expected_values, grid, errors))
    slopes = (np.diff(radiocarbon) + np.diff(errors)) / np.diff(grid)
    np.testing.assert_allclose(gradients, slopes[indices].reshape(np.asarray(values).shape))
    if shape == (None,):
        assert evaluate(np.array([-2.5]))[1].shape == (1,)


def radiocarbon_model(family):
    scalar = family == 'scalar'
    shape = () if scalar else (2,)
    with pm.Model() as model:
        if family == 'hierarchical':
            mu = pm.TruncatedNormal('tau_mu', mu=-2., sigma=.5, lower=-3.8, upper=-.2)
            sd = pm.HalfNormal('tau_sd', sigma=.5)
            tau = pm.TruncatedNormal('tau', mu=mu, sigma=sd, lower=-3.8, upper=-.2, shape=shape)
        elif family == 'mixture':
            weights = pm.Dirichlet('weights', a=np.ones(2))
            means = pm.Uniform('means', lower=-3.8, upper=-.2, shape=2)
            sds = pm.HalfNormal('sds', sigma=.5, shape=2)
            components = [pm.TruncatedNormal.dist(mu=means[i], sigma=sds[i], lower=-3.8, upper=-.2) for i in range(2)]
            tau = pm.Mixture('tau', w=weights, comp_dists=components, shape=shape)
        else:
            tau = pm.Uniform('tau', lower=-3.8, upper=-.2, shape=shape)
        mean, sigma = interpolate_calcurve(tau, *CURVE)
        if scalar:
            # This is how test_drive.ipynb consumes scalar interpolation.
            mean, sigma = mean[0], sigma[0]
        latent = pm.Normal('r_latent', mu=mean, sigma=sigma, shape=shape)
        pm.Normal('r_measured', mu=latent, sigma=.2,
                  observed=-2.7 if scalar else [-2.7, -1.7], shape=shape)
    expected = {'tau': shape, 'r_latent': shape}
    if family == 'hierarchical':
        expected.update(tau_mu=(), tau_sd=())
    if family == 'mixture':
        expected.update(weights=(2,), means=(2,), sds=(2,))
    return model, expected


def ippp_model(family):
    with pm.Model() as model:
        tau = pm.Uniform('tau', lower=1., upper=2., shape=2)
        pm.Normal('observed_times', mu=tau, sigma=.1, observed=[1.3, 1.7])
        domain = pt.as_tensor_variable(np.linspace(1., 2., 16))
        if family == 'sine':
            a = pm.HalfNormal('a', sigma=2.)
            b = pm.Uniform('b', lower=8., upper=12.)
            likelihood = ippp_logp_sine(tau, a, b, domain)
            expected = {'tau': (2,), 'a': (), 'b': ()}
        else:
            # The existing one-covariate/scalar-coefficient path has scalar
            # logp. This preserves its linear (not exponential) rate function.
            beta = pm.HalfNormal('beta', sigma=2.)
            likelihood = ippp_logp_lm(tau, beta, domain)
            expected = {'tau': (2,), 'beta': ()}
        pm.Potential('ippp_likelihood', likelihood)
    return model, expected


@pytest.mark.parametrize('family', ['scalar', 'vector', 'hierarchical', 'mixture', 'sine', 'linear'])
def test_model_logp_gradient_and_sampling(family):
    model, expected = ippp_model(family) if family in ('sine', 'linear') else radiocarbon_model(family)
    with model:
        point = model.initial_point()
        logp = model.compile_logp()(point)
        assert np.ndim(logp) == 0 and np.isfinite(logp)
        gradient = model.compile_dlogp()(point)
        assert gradient.size == sum(np.size(v) for v in point.values())
        assert np.all(np.isfinite(gradient))
        # Explicitly use PyMC's own NUTS, independent of optional installed
        # samplers. Tiny counts test execution/structure, not convergence.
        trace = pm.sample(draws=8, tune=8, chains=1, cores=1, random_seed=912,
                          nuts_sampler='pymc', init='adapt_diag',
                          progressbar=False, compute_convergence_checks=False)
    assert isinstance(trace, xr.DataTree)
    assert {'posterior', 'sample_stats', 'observed_data'} <= set(trace.children)
    posterior = trace['posterior'].to_dataset()
    assert posterior.sizes['chain'] == 1 and posterior.sizes['draw'] == 8
    for name, shape in expected.items():
        assert posterior[name].dims[:2] == ('chain', 'draw')
        assert posterior[name].shape == (1, 8, *shape)
        assert np.all(np.isfinite(posterior[name].values))
    assert 'diverging' in trace['sample_stats']


@pytest.mark.parametrize('family', ['sine', 'linear'])
def test_ippp_numeric_logp_and_gradient(family):
    # Check the existing rectangular integral, without changing its convention.
    domain = np.linspace(1., 2., 16)
    events = np.array([1.3, 1.7])
    a = pt.scalar('a')
    times = pt.as_tensor_variable(events)
    grid = pt.as_tensor_variable(domain)
    logp = ippp_logp_sine(times, a, 10., grid) if family == 'sine' else ippp_logp_lm(times, a, grid)
    fn = pytensor.function([a], [logp, pt.grad(logp, a)])
    rate = lambda t: 2. * (1 + np.sin(2*np.pi*t/10)) if family == 'sine' else 2.*t
    expected = np.log(rate(events)).sum() - rate(domain).sum()*(domain[-1]-domain[0])/len(domain)
    value, derivative = fn(2.)
    np.testing.assert_allclose(value, expected)
    delta = 1e-5
    np.testing.assert_allclose(derivative, (fn(2.+delta)[0]-fn(2.-delta)[0])/(2*delta), rtol=1e-6)
