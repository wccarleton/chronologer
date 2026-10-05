"""Small real prior-predictive checks; no MCMC."""
import numpy as np
import pytest

from chronologer.density import DensitySim, simulate_single_density, simulate_radiocarbon_density, simulate_gaussian_mixture
from test_curve_references import curve

SETTINGS = dict(lower=-90, upper=-10, mean_prior=-50, mean_prior_sd=10, sd_prior_scale=15)


def test_one_parameter_draw_can_generate_large_datasets():
    single = simulate_single_density(1000, draws=1, **SETTINGS)
    mixture = simulate_gaussian_mixture(10000, K_max=2, prior_center=-50, prior_scale=5, draws=1)
    for result, n in ((single, 1000), (mixture, 10000)):
        assert result.prior['prior']['tau'].shape == result.prior['prior']['measured'].shape == (1, 1, n)
        assert np.isfinite(result.prior['prior']['measured']).all()
        assert np.allclose(result.density['lower_values'], result.density['pdf_values'])
        assert np.allclose(result.density['upper_values'], result.density['pdf_values'])


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


def test_forward_radiocarbon_mean_and_combined_error():
    """A large nonidentity curve offset must reach the measurement distribution."""
    import pymc as pm
    import pytensor.tensor as pt
    from scipy.stats import norm
    from chronologer.density import _simulate_measurement
    synthetic = dict(calbp=np.array([-100., -50., 0.]),
        c14bp=np.array([-900., -700., -500.]), c14_sigma=np.full(3, 4.))
    with pm.Model(coords={'event': np.arange(2)}) as model:
        _simulate_measurement(pt.as_tensor_variable(np.array([-75., -25.])),
                              'calrcarbon', 3., synthetic)
    measured = model['measured']
    # Analytic curve means are -800/-600, with sqrt(3² + 4²) = 5 SD.
    expected = np.array([-800., -600.])
    assert np.allclose(pm.logp(measured, expected).eval(), norm.logpdf(expected, expected, 5))
    assert np.allclose(pm.logp(measured, expected + 5).eval(), norm.logpdf(expected + 5, expected, 5))


def test_mixture_simulation_reuses_priors_and_density():
    from scipy.stats import norm
    result = simulate_gaussian_mixture(3, K_max=3, prior_center=-50, prior_scale=5, error=3, draws=4)
    prior = result.prior['prior'].to_dataset()
    assert prior['tau'].shape == prior['measured'].shape == (1, 4, 3)
    assert np.all(np.diff(prior['means'].values, axis=-1) >= 0)
    assert np.allclose(prior['weights'].sum('component'), 1)
    grid = result.density['t_values']
    values = (prior['weights'].values[..., None] * norm.pdf(grid,
        loc=prior['means'].values[..., None], scale=prior['scales'].values[..., None])).sum(axis=2)
    assert np.allclose(result.density['pdf_values'], values.mean(axis=(0, 1)))
    assert result.priors['concentration'] == .3
    radiocarbon = simulate_gaussian_mixture(2, K_max=1, prior_center=-50, prior_scale=5,
        distribution='calrcarbon', calcurve=curve(), draws=2)
    assert np.isfinite(radiocarbon.prior['prior']['measured']).all()
    assert np.all(radiocarbon.prior['prior']['weights'] == 1)
    with pytest.raises(ValueError, match='outside calibration curve support'):
        simulate_gaussian_mixture(2, K_max=1, prior_center=-500, prior_scale=5,
            distribution='calrcarbon', calcurve=curve(), draws=2)
