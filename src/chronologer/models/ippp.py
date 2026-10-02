"""IPPP likelihood helpers and a finite-grid Gaussian-process intensity model."""
from __future__ import annotations

from typing import Callable
from dataclasses import dataclass
from numbers import Real

import numpy as np
import pytensor.tensor as pt


def _grid_logp(times, grid, rate):
    """Full point-process log likelihood for a piecewise-linear intensity.

    The trapezoid integral is exact for this interpolation, including endpoints.
    No division by integrated intensity (conditioning on count) is performed.
    """
    times, grid, rate = map(pt.as_tensor_variable, (times, grid, rate))
    index = pt.clip(pt.searchsorted(grid, times, side='right') - 1, 0, grid.shape[0] - 2)
    fraction = (times - grid[index]) / (grid[index + 1] - grid[index])
    at_event = rate[index] + fraction * (rate[index + 1] - rate[index])
    integral = pt.sum((rate[:-1] + rate[1:]) * pt.diff(grid) / 2)
    value = pt.sum(pt.log(at_event)) - integral
    return pt.switch(pt.all((times >= grid[0]) & (times <= grid[-1])), value, -np.inf)


def build_gp(data, *, start, end, grid_size=32, baseline_count=10.,
             log_rate_sd=1.5, amplitude_scale=1., length_scale_median=None,
             length_scale_log_sd=.5):
    """Build the basic GP IPPP on an explicitly supplied observation period.

    start < end in increasing native calendar coordinates (negative BP for C14).
    data contains exact numeric times and/or supported measurement distributions:
    calrcarbon, frozen scipy normal/uniform. Empty data means an observed empty
    window. Assumes complete observation throughout the entire declared period.

    The GP defines log intensity at grid nodes; positive intensity is interpolated
    linearly between them. This is a finite-grid approximation, not an exact
    continuous GP. Priors are documented in docs/ippp-gp.md and model.ippp_spec.
    """
    import pymc as pm
    from ..density import _measurement
    from ..distributions import calrcarbon
    if (not isinstance(start, Real) or isinstance(start, (bool, np.bool_))
            or not isinstance(end, Real) or isinstance(end, (bool, np.bool_))
            or not np.isfinite([start, end]).all() or start >= end):
        raise ValueError('Explicit finite observation start < end is required; dates are never inferred.')
    duration = float(end - start)
    if not np.isfinite(duration):
        raise ValueError('Observation duration must be finite.')
    if type(grid_size) is not int or not 4 <= grid_size <= 256:
        raise ValueError('grid_size must be an integer from 4 to 256 for this dense-GP benchmark.')
    length_scale_median = duration / 5 if length_scale_median is None else length_scale_median
    prior_values = [baseline_count, log_rate_sd, amplitude_scale, length_scale_median, length_scale_log_sd]
    if any(not isinstance(v, Real) or isinstance(v, (bool, np.bool_)) or not np.isfinite(v) or v <= 0 for v in prior_values):
        raise ValueError('GP prior scales and baseline_count must be finite and positive.')
    data = list(data)
    grid = np.linspace(start, end, grid_size)
    if np.any(np.diff(grid) <= 0):
        raise ValueError('Observation window is too narrow for a distinct numerical grid.')
    unknown, lower, upper, initial = [], [], [], []
    known = {}
    for i, event in enumerate(data):
        if isinstance(event, Real) and not isinstance(event, (bool, np.bool_)):
            if not np.isfinite(event) or not start <= event <= end:
                raise ValueError('Exact event dates must lie within the declared observation period.')
            known[i] = float(event)
            continue
        logp, centre, _ = _measurement(event)
        support = (event.a, event.b) if isinstance(event, calrcarbon) else event.support()
        lo, hi = max(start, support[0]), min(end, support[1])
        if not lo < hi:
            raise ValueError('An event measurement has no support within the declared observation period.')
        unknown.append((i, logp)); lower.append(lo); upper.append(hi)
        initial.append(float(np.clip(centre, lo + .05 * (hi - lo), hi - .05 * (hi - lo))))
    with pm.Model(coords={'grid': grid, 'event': np.arange(len(data)),
                          'uncertain_event': [i for i, _ in unknown]}) as model:
        log_rate = pm.Normal('log_rate', mu=np.log(baseline_count) - np.log(duration), sigma=log_rate_sd)
        amplitude = pm.HalfNormal('amplitude', sigma=amplitude_scale)
        length_scale = pm.LogNormal('length_scale', mu=np.log(length_scale_median), sigma=length_scale_log_sd)
        process = pm.gp.Latent(cov_func=amplitude**2 * pm.gp.cov.ExpQuad(1, ls=length_scale / duration))
        offset = process.prior('gp_offset', X=np.linspace(0., 1., grid_size)[:, None], dims='grid')
        rate = pm.Deterministic('intensity', pt.exp(log_rate + offset), dims='grid')
        integral = pm.Deterministic('integrated_intensity', pt.sum((rate[:-1] + rate[1:]) * np.diff(grid) / 2))
        if unknown:
            latent = pm.Uniform('tau_unknown', lower=np.array(lower), upper=np.array(upper),
                                dims='uncertain_event', initval=np.array(initial))
            # Uniforms provide bounded coordinates only. Cancel their constants so
            # the joint measure is likelihood * IPPP intensity, not an extra prior.
            pm.Potential('measurements', pt.sum(pt.stack([logp(latent[j]) for j, (_, logp) in enumerate(unknown)]))
                         + np.log(np.array(upper) - np.array(lower)).sum())
        values = dict(known)
        values.update({i: latent[j] for j, (i, _) in enumerate(unknown)})
        tau = (pm.Deterministic('tau', pt.stack([values[i] for i in range(len(data))]), dims='event')
               if data else pt.as_tensor_variable(np.array([], dtype=float)))
        pm.Potential('ippp_likelihood', _grid_logp(tau, grid, rate))
    model.ippp_spec = dict(start=float(start), end=float(end), grid_size=grid_size,
                           baseline_count=float(baseline_count), log_rate_sd=float(log_rate_sd),
                           amplitude_scale=float(amplitude_scale), length_scale_median=float(length_scale_median),
                           length_scale_log_sd=float(length_scale_log_sd))
    return model


@dataclass
class GPFit:
    posterior: object
    intensity: dict
    specification: dict


def gp(data, *, params=None, mcmc_config=None, progress_callback=None):
    """Fit the basic GP IPPP. params must explicitly contain start and end.

    Returns GPFit with raw PyMC DataTree, intensity (events per time unit) posterior
    mean and pointwise 95% band on the grid, and the resolved model specification.
    """
    import pymc as pm
    from .density import _options, _MCMC_KEYS
    settings = _options(params, {'start', 'end', 'grid_size', 'baseline_count', 'log_rate_sd',
                                'amplitude_scale', 'length_scale_median', 'length_scale_log_sd'}, 'params')
    if 'start' not in settings or 'end' not in settings:
        raise ValueError('Declare observation start and end explicitly in params.')
    sampling = dict(draws=250, tune=250, chains=2, random_seed=912, cores=1)
    sampling.update(_options(mcmc_config, _MCMC_KEYS, 'mcmc_config'))
    for key in ('draws', 'tune', 'chains', 'cores'):
        if type(sampling[key]) is not int or sampling[key] < (0 if key == 'tune' else 1):
            raise ValueError('MCMC counts must be integers: positive draws/chains, nonnegative tune.')
    total = (sampling['draws'] + sampling['tune']) * sampling['chains']
    sampling['cores'] = min(sampling['cores'], sampling['chains'])
    counts = [0] * sampling['chains']
    def report(stage, completed=0):
        if progress_callback:
            progress_callback(dict(stage=stage, completed=completed, total=total))
    def on_draw(trace, draw):
        counts[draw.chain] = draw.draw_idx + 1
        report(f"{'Tuning' if draw.tuning else 'Sampling'} · chain {draw.chain + 1}/{sampling['chains']}",
               sum(counts))
    report('Building GP IPPP model')
    model = build_gp(data, **settings)
    with model:
        report('Compiling and initializing')
        posterior = pm.sample(**sampling, blas_cores='auto', nuts_sampler='pymc', init='adapt_diag',
                              target_accept=.95, progressbar=False, compute_convergence_checks=False,
                              callback=on_draw if progress_callback else None)
    report('Evaluating intensity', total)
    rates = posterior['posterior']['intensity'].transpose('chain', 'draw', 'grid').values
    rates = rates.reshape(-1, rates.shape[-1])
    if not np.isfinite(rates).all():
        raise ValueError('Posterior intensity contains nonfinite values.')
    low, high = np.quantile(rates, [.025, .975], axis=0)
    intensity = dict(t_values=np.array(model.coords['grid']), rate_values=rates.mean(axis=0),
                     lower_values=low, upper_values=high)
    return GPFit(posterior, intensity, model.ippp_spec)


def approx_integral(
    rate_func: Callable[[pt.TensorVariable], pt.TensorVariable],
    domain: pt.TensorVariable,
) -> pt.TensorVariable:
    """
    Approximates the integral of the rate function over a given domain.

    Parameters:
    -----------
    rate_func : callable
        The rate function to be integrated, should take a tensor as input.
    domain : tensor
        A sequence of points over which the rate function is evaluated.

    Returns:
    --------
    TensorVariable
        The approximate integral of the rate function over the domain.
    """
    # Evaluate the rate function at the points in the domain
    rate_values = rate_func(domain)

    # Number of evaluation points (inferred from the domain shape)
    eval_n = domain.shape[0]

    # Approximate the integral using the sum of the rate values times the step size
    # Assume equally spaced points in domain unless provided differently
    integral_rate = pt.sum(rate_values) * (domain[-1] - domain[0]) / eval_n

    return integral_rate


def ippp_logp_sine(
    value: pt.TensorVariable,
    a: float,
    b: float,
    domain: pt.TensorVariable,
) -> pt.TensorVariable:
    """
    Log-likelihood for an inhomogeneous Poisson process with a sine rate
    function and tensor-compatible parameters.

    Parameters:
    -----------
    value : tensor
        Observed event times as a PyTensor tensor.
    a : float
        Amplitude of the sine wave.
    b : float
        Period of the sine wave.
    domain : tensor
        The sequence of regularly-spaced points over which the GP or other
        covariate function is evaluated (for integral approximation).

    Returns:
    --------
    TensorVariable
        Log-likelihood of observing the event times based on the IPPP model.
    """
    # Define the rate function using a and b
    def rate_func(t: pt.TensorVariable) -> pt.TensorVariable:
        return a * (1 + pt.sin(2 * pt.pi * t / b))

    # Log-likelihood: sum of log(rate) at event times
    log_rate_sum = pt.sum(pt.log(rate_func(value)))

    # Approximate the integral over the interval [start, end]
    integral_rate = approx_integral(rate_func, domain)

    # Return the log-likelihood
    return log_rate_sum - integral_rate


def ippp_logp_lm(
    X_tau: pt.TensorVariable,
    Beta: pt.TensorVariable,
    domain: pt.TensorVariable,
) -> pt.TensorVariable:
    """
    Log-likelihood for an inhomogeneous Poisson process using a linear model
    and a Gaussian process sample for the covariate process.

    Parameters:
    -----------
    X_tau : tensor
        Covariate matrix (design matrix) evaluated at the observed and
        uncertain event times, shape ``(n_events, n_covariates)``.
    Beta : tensor
        Regression coefficient vector of length n_covariates.
    domain : tensor
        The sequence of regularly-spaced points over which the GP or other
        covariate function is evaluated (for integral approximation).

    Returns:
    --------
    TensorVariable
        Log-likelihood of observing the event times based on the IPPP model.
    """
    # Define the rate function as λ_t = X_tau * Beta
    def rate_func(X_t: pt.TensorVariable) -> pt.TensorVariable:
        return pt.dot(X_t, Beta)

    # Log-likelihood: sum of log(rate) at event times τ
    log_rate_sum = pt.sum(pt.log(rate_func(X_tau)))

    # Approximate the integral over the domain
    integral_rate = approx_integral(rate_func, domain)

    # Return the log-likelihood
    return log_rate_sum - integral_rate
