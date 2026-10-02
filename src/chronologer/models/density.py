"""Concise fitting entry points for the existing density models.

These functions delegate without changing scientific or sampling defaults.
The original builders, fit functions and result classes remain in chronologer.density.
"""
from collections.abc import Mapping

from .. import density as _implementation

__all__ = ['single', 'gmixture']
_MCMC_KEYS = {'draws', 'tune', 'chains', 'random_seed'}


def _options(value, allowed, name):
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f'{name} must be a mapping or None.')
    unknown = set(value) - allowed
    if unknown:
        raise TypeError(f'Unknown {name} option(s): {", ".join(sorted(map(str, unknown)))}')
    return dict(value)


def single(data, *, params=None, mcmc_config=None, progress_callback=None):
    """Fit the existing single truncated-normal radiocarbon hierarchy.

    data: mapping with radiocarbon_ages, radiocarbon_errors and calcurve, using
        the existing engine's negative-BP coordinates (no sign conversion).
    params: lower, upper, mean_prior, mean_prior_sd, sd_prior_scale (all required).
    mcmc_config: optional draws, tune, chains, random_seed. Omitted values retain
        the engine defaults: 250 draws, 250 tuning iterations, 2 chains, seed 912.
    progress_callback: optional callable receiving stage/completed/total updates.

    Returns the unchanged chronologer.density.DensityFit object.
    """
    keys = {'radiocarbon_ages', 'radiocarbon_errors', 'calcurve'}
    if not isinstance(data, Mapping) or set(data) != keys:
        raise TypeError('single data must contain radiocarbon_ages, radiocarbon_errors and calcurve.')
    scientific = _options(params, {'lower', 'upper', 'mean_prior', 'mean_prior_sd', 'sd_prior_scale'}, 'params')
    sampling = _options(mcmc_config, _MCMC_KEYS, 'mcmc_config')
    return _implementation.fit_radiocarbon_density(
        data['radiocarbon_ages'], data['radiocarbon_errors'], data['calcurve'],
        **scientific, **sampling, progress_callback=progress_callback)


def gmixture(data, *, params=None, mcmc_config=None, progress_callback=None):
    """Fit the existing sparse Gaussian mixture to measurement distributions.

    data: existing sequence of calrcarbon or scipy frozen normal/uniform objects.
        Each radiocarbon distribution retains its own calibration-curve splines.
    params: optional K_max (default 5), prior_center, prior_scale and output grid.
    mcmc_config: optional draws, tune, chains, random_seed. Omitted values retain
        the engine defaults: 250 draws, 250 tuning iterations, 2 chains, seed 912.
    progress_callback: optional callable receiving stage/completed/total updates.

    Returns the unchanged chronologer.density.GaussianMixtureFit object.
    """
    scientific = _options(params, {'K_max', 'prior_center', 'prior_scale', 'grid'}, 'params')
    sampling = _options(mcmc_config, _MCMC_KEYS, 'mcmc_config')
    return _implementation.fit_gaussian_mixture(
        data, **scientific, **sampling, progress_callback=progress_callback)
