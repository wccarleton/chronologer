"""Small generic-density checks, without MCMC."""
import numpy as np
import pytest
import pytensor
import xarray as xr
from scipy.stats import norm, uniform

import chronologer as ch
from chronologer.distributions import calrcarbon
from test_curve_references import curve

SETTINGS = dict(lower=-90, upper=-10, mean_prior=-50, mean_prior_sd=20, sd_prior_scale=15)


def test_mixed_measurements_use_existing_likelihoods():
    events = [norm(-50, 3), uniform(-65, 20),
              calrcarbon(curve(), -50, 3), calrcarbon(curve(20), -30, 4)]
    model = ch.build_single_density(events, **SETTINGS)
    assert model.coords['event'] == (0, 1, 2, 3)
    assert np.isfinite(model.compile_logp(mode='FAST_COMPILE')(model.initial_point()))
    logp = pytensor.function([model['tau']], model.potentials[0], mode='FAST_COMPILE')
    dates = np.array([-49., -55., -48., -49.])
    assert logp(dates) == pytest.approx(sum(e.logpdf(t) for e, t in zip(events, dates)))
    with pytest.raises(ValueError, match='intersect'):
        ch.build_single_density([uniform(-200, 10)], **SETTINGS)


def test_fit_reuses_sampling_and_normalized_density(monkeypatch):
    import pymc as pm
    from chronologer.density import DensityFit
    trace = xr.DataTree.from_dict({'posterior': xr.Dataset({
        'tau_mu': (('chain', 'draw'), [[-50., -49., -51., -48.]]),
        'tau_sd': (('chain', 'draw'), [[10., 11., 12., 13.]]),
    })})
    calls = []
    def sample(**kwargs):
        calls.append(kwargs); return trace
    monkeypatch.setattr(pm, 'sample', sample)
    progress = []
    fit = ch.models.density.single_density([norm(-50, 3)], params=SETTINGS,
        mcmc_config=dict(draws=4, tune=4, chains=1, cores=1), progress_callback=progress.append)
    assert isinstance(fit, DensityFit) and fit.posterior is trace
    assert calls[0]['nuts_sampler'] == 'pymc' and calls[0]['cores'] == 1
    assert progress[-1]['completed'] == progress[-1]['total'] == 8
    assert np.trapezoid(fit.density['pdf_values'], fit.density['t_values']) == pytest.approx(1, abs=1e-4)
