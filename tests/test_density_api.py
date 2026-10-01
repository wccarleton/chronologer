import numpy as np
import pytest
import xarray as xr

import chronologer
from test_pymc_models import CURVE, radiocarbon_model


CURVE_DATA = dict(zip(('calbp', 'c14bp', 'c14_sigma'), [a.data for a in CURVE]))
SETTINGS = dict(lower=-3.8, upper=-.2, mean_prior=-2., mean_prior_sd=.5, sd_prior_scale=.5)


def test_public_density_model_preserves_tested_hierarchy_logp_and_gradient():
    reference, _ = radiocarbon_model('hierarchical')
    model = chronologer.build_radiocarbon_density([-2.7, -1.7], [.2, .2], CURVE_DATA, **SETTINGS)
    point = reference.initial_point()
    assert model.initial_point().keys() == point.keys()
    np.testing.assert_allclose(model.compile_logp()(point), reference.compile_logp()(point))
    np.testing.assert_allclose(model.compile_dlogp()(point), reference.compile_dlogp()(point))


def test_public_density_fit_returns_posterior_and_population_density():
    progress = []
    fit = chronologer.fit_radiocarbon_density([-2.7, -1.7], [.2, .2], CURVE_DATA,
                                             **SETTINGS, draws=8, tune=8, chains=1,
                                             progress_callback=progress.append)
    assert progress[0]['stage'] == 'Building model'
    assert progress[-1] == dict(stage='Evaluating density', completed=16, total=16)
    assert [p['completed'] for p in progress] == sorted(p['completed'] for p in progress)
    assert any(p['stage'].startswith('Tuning') for p in progress)
    assert any(p['stage'].startswith('Sampling') for p in progress)
    assert isinstance(fit.posterior, xr.DataTree)
    posterior = fit.posterior['posterior'].to_dataset()
    assert {'tau_mu', 'tau_sd', 'tau', 'r_latent'} <= set(posterior)
    assert posterior['tau'].shape == (1, 8, 2)
    density = fit.density
    assert all(values.shape == (512,) and np.isfinite(values).all() for values in density.values())
    assert np.all(density['lower_values'] <= density['upper_values'])
    assert np.all(density['pdf_values'] >= 0)
    assert np.trapezoid(density['pdf_values'], density['t_values']) == pytest.approx(1, abs=.01)


def test_density_rejects_final_curve_endpoint():
    with pytest.raises(ValueError, match='final grid point'):
        chronologer.build_radiocarbon_density([-2.7], [.2], CURVE_DATA, **{**SETTINGS, 'upper': 0})
