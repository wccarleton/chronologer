"""Concise model calls must be transparent adapters to the existing API."""
import numpy as np
import pytest
from scipy.stats import norm
import xarray as xr

import chronologer as ch
from chronologer import density as implementation
from chronologer.models import approx_integral, ippp_logp_lm, ippp_logp_sine
from test_density_api import CURVE_DATA, SETTINGS


def single_data():
    return dict(radiocarbon_ages=[-2.7, -1.7], radiocarbon_errors=[.2, .2], calcurve=CURVE_DATA)


def test_legacy_imports_and_result_classes_remain_available():
    assert ch.fit_radiocarbon_density is implementation.fit_radiocarbon_density
    assert ch.fit_gaussian_mixture is implementation.fit_gaussian_mixture
    assert approx_integral is ch.models.ippp.approx_integral
    assert ippp_logp_lm is ch.models.ippp.ippp_logp_lm
    assert ippp_logp_sine is ch.models.ippp.ippp_logp_sine
    assert callable(ch.models.ippp.gp)


@pytest.mark.parametrize('family', ['single', 'gmixture'])
@pytest.mark.parametrize('dispatch', [False, True])
def test_adapters_forward_exact_inputs_and_return_identity(monkeypatch, family, dispatch):
    data = single_data() if family == 'single' else [norm(-2, .2), norm(-3, .3)]
    params = SETTINGS.copy() if family == 'single' else dict(K_max=2, prior_center=-2.5, prior_scale=1, grid=np.linspace(-5, 0, 10))
    mcmc = dict(draws=17, tune=9, chains=3, random_seed=41)
    before_params, before_mcmc = params.copy(), mcmc.copy()
    progress = lambda update: None
    result = object()
    captured = []
    def fake(*args, **kwargs):
        captured.append((args, kwargs))
        return result
    monkeypatch.setattr(implementation, 'fit_radiocarbon_density' if family == 'single' else 'fit_gaussian_mixture', fake)
    model = getattr(ch.models.density, family)
    options = dict(params=params, mcmc_config=mcmc, progress_callback=progress)
    actual = ch.fit(data, model=model, **options) if dispatch else model(data, **options)
    assert actual is result
    args, kwargs = captured[0]
    assert args[0] is (data['radiocarbon_ages'] if family == 'single' else data)
    if family == 'single':
        assert args[1] is data['radiocarbon_errors'] and args[2] is data['calcurve']
    assert kwargs == {**params, **mcmc, 'progress_callback': progress}
    assert params == before_params and mcmc == before_mcmc


def test_defaults_are_left_to_legacy_implementation(monkeypatch):
    captured = []
    monkeypatch.setattr(implementation, 'fit_gaussian_mixture', lambda *args, **kwargs: captured.append(kwargs))
    ch.models.density.gmixture([])
    assert captured == [{'progress_callback': None}]
    def custom(data, *, params, mcmc_config):
        return data, params, mcmc_config
    assert ch.fit('data', model=custom) == ('data', None, None)


@pytest.mark.parametrize('family', ['single', 'gmixture'])
def test_wrong_option_groups_rejected_before_fitting(family):
    model = getattr(ch.models.density, family)
    data = single_data() if family == 'single' else []
    for kwargs in [dict(params={'draws': 10}), dict(mcmc_config={'mean_prior': 0}),
                   dict(mcmc_config={'nuts_sampler': 'nutpie'}), dict(params=['K_max'])]:
        with pytest.raises(TypeError):
            model(data, **kwargs)
    with pytest.raises(TypeError, match='callable'):
        ch.fit(data, model='density.gmixture')


def test_single_requires_explicit_input_mapping():
    with pytest.raises(TypeError, match='radiocarbon_ages'):
        ch.models.density.single([1, 2])


@pytest.mark.parametrize('family', ['single', 'gmixture'])
@pytest.mark.parametrize('cores', [1, 2])
def test_switchboard_executes_existing_models_with_small_samples(family, cores):
    progress = []
    data = single_data() if family == 'single' else [norm(-2.7, .2), norm(-1.7, .2)]
    params = SETTINGS if family == 'single' else {'K_max': 1}
    result = ch.fit(data, model=getattr(ch.models.density, family), params=params,
                    mcmc_config=dict(draws=4, tune=4, chains=cores, cores=cores, random_seed=19),
                    progress_callback=progress.append)
    assert isinstance(result, implementation.DensityFit if family == 'single' else implementation.GaussianMixtureFit)
    assert isinstance(result.posterior, xr.DataTree)
    assert result.posterior['posterior']['tau'].shape == (cores, 4, 2)
    assert all(np.isfinite(values).all() for values in result.density.values())
    assert progress[-1]['completed'] == progress[-1]['total'] == 8 * cores
    counts = [p['completed'] for p in progress]
    assert counts == sorted(counts)
