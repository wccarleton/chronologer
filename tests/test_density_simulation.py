"""Small real prior-predictive checks; no MCMC."""
import numpy as np
import pytest

from chronologer.density import DensitySim, simulate_single_density, simulate_radiocarbon_density
from test_curve_references import curve

SETTINGS = dict(lower=-90, upper=-10, mean_prior=-50, mean_prior_sd=10, sd_prior_scale=15)


def test_calendar_simulation_has_prior_dates_and_density():
    result = simulate_single_density(3, distribution='normal', error=4, draws=4, **SETTINGS)
    assert isinstance(result, DensitySim) and 'posterior' not in result.prior.children
    prior = result.prior['prior'].to_dataset()
    assert prior['tau'].shape == prior['measured'].shape == (1, 4, 3)
    assert np.all((prior['tau'].values >= -90) & (prior['tau'].values <= -10))
    assert np.isfinite(prior['measured']).all()
    assert np.trapezoid(result.density['pdf_values'], result.density['t_values']) == pytest.approx(1, abs=.001)
    uniform = simulate_single_density(2, distribution='uniform', error=4, draws=4, **SETTINGS).prior['prior'].to_dataset()
    assert np.all(np.abs(uniform['measured'] - uniform['tau']) <= np.sqrt(3) * 4)


def test_radiocarbon_simulation_uses_curve_and_reports_progress():
    progress = []
    result = simulate_radiocarbon_density(3, curve(), error=3, draws=4,
                                          progress_callback=progress.append, **SETTINGS)
    assert np.isfinite(result.prior['prior']['measured'].values).all()
    assert progress[-1]['completed'] == progress[-1]['total'] == 4
    with pytest.raises(ValueError, match='curve support'):
        simulate_radiocarbon_density(3, curve(), draws=4, **{**SETTINGS, 'lower': -101})
