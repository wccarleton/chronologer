"""Post-fit predictive scores; no model construction or sampling.

Measurement likelihoods integrate out event dates. IPPP treats the entire
observation window as one predictive unit, preserving the count likelihood.
"""
import numpy as np
from scipy.integrate import quad_vec
from scipy.special import logsumexp
from scipy.stats import norm, truncnorm

from .distributions import calrcarbon


def waic_from_log_likelihood(values, *, n_events, likelihood, notes=()):
    """WAIC from a retained-draw by predictive-unit log-likelihood matrix."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 1 or not np.isfinite(values).all():
        raise ValueError('WAIC requires at least two finite posterior draws and one predictive unit.')
    variance = values.var(axis=0)
    pointwise = logsumexp(values, axis=0) - np.log(len(values)) - variance
    elpd = float(pointwise.sum())
    score = dict(waic=-2 * elpd, elpd_waic=elpd, p_waic=float(variance.sum()),
                 se=float(2 * np.sqrt(len(pointwise) * pointwise.var())) if len(pointwise) > 1 else None,
                 n_events=n_events, n_samples=len(values), n_units=len(pointwise),
                 warning=bool(np.any(variance > .4)), likelihood=likelihood, notes=list(notes))
    if not all(np.isfinite(score[k]) for k in ('waic', 'elpd_waic', 'p_waic')):
        raise ValueError('Nonfinite WAIC estimate.')
    return score


def model_diagnostics(posterior, measurements, *, model, lower=None, upper=None, grid=None):
    """Score single density, mixture or GP IPPP without changing fitted results.

    posterior is an xarray Dataset. Adaptive quadrature uses the original
    measurement likelihood (including each radiocarbon curve), never a
    normalized calibrated-date density. Failures return an explicit reason.
    """
    try:
        return _score(posterior, list(measurements), model, lower, upper, grid)
    except (ValueError, FloatingPointError) as error:
        return dict(unavailable=str(error))


def _score(posterior, measurements, model, lower, upper, grid):
    def draws(name):
        x = posterior[name].transpose('chain', 'draw', ...).values
        return x.reshape((-1, *x.shape[2:]))

    if model == 'ippp_gp':
        grid = np.asarray(grid, dtype=float)
        rates = draws('intensity')
        if grid.ndim != 1 or len(grid) != rates.shape[1] or not np.all(np.diff(grid) > 0):
            raise ValueError('IPPP diagnostics require the fitted observation grid.')
        lower, upper = grid[0], grid[-1]
        def density(t):
            j = min(max(np.searchsorted(grid, t, side='right') - 1, 0), len(grid) - 2)
            return rates[:, j] + (rates[:, j + 1] - rates[:, j]) * (t - grid[j]) / (grid[j + 1] - grid[j])
        points = grid[1:-1]
    elif model == 'mixture':
        means, scales, weights = (draws(k) for k in ('means', 'scales', 'weights'))
        def density(t):
            return np.sum(weights * norm.pdf(t, means, scales), axis=1)
        points = np.unique(np.quantile(np.r_[means.ravel(), (means - scales).ravel(), (means + scales).ravel()], np.linspace(0, 1, 61)))
    elif model == 'single_density':
        means, scales = draws('tau_mu'), draws('tau_sd')
        if lower is None or upper is None or lower >= upper:
            raise ValueError('Single-density diagnostics require the fitted bounds.')
        def density(t):
            return truncnorm.pdf(t, (lower - means) / scales, (upper - means) / scales, means, scales)
        points = np.unique(np.quantile(np.r_[means, means - scales, means + scales], np.linspace(0, 1, 61)))
    else:
        raise ValueError('Predictive diagnostics are unavailable for this model.')

    columns = []
    for event in measurements:
        if isinstance(event, (int, float, np.number)):
            columns.append(np.log(density(float(event))))
            continue
        support = (event.a, event.b) if isinstance(event, calrcarbon) else event.support()
        lo = max(support[0], lower if lower is not None else -np.inf)
        hi = min(support[1], upper if upper is not None else np.inf)
        if not lo < hi:
            raise ValueError('Measurement support does not intersect the model domain.')
        # Scale the likelihood for numerical integration, retaining its constant.
        if isinstance(event, calrcarbon):
            candidates = np.linspace(lo, hi, max(257, int(np.ceil((hi - lo) / 20)) + 1))
        else:
            candidates = np.array([np.clip(event.mean(), lo, hi)])
        offset = (-np.log(event.c14_err * np.sqrt(2 * np.pi)) if isinstance(event, calrcarbon)
                  else float(np.max(event.logpdf(candidates))))
        cuts = points[(points > lo) & (points < hi)]
        if not isinstance(event, calrcarbon) and event.dist.name == 'norm':
            cuts = np.unique(np.r_[cuts, event.mean() + event.std() * np.arange(-8, 9)])
            cuts = cuts[(cuts > lo) & (cuts < hi)]
        if isinstance(event, calrcarbon):
            cuts = np.unique(np.r_[cuts, candidates[1:-1]])
        # quad_vec points require finite endpoints; split infinite domains.
        edges = np.r_[lo, cuts, hi]
        value = 0
        for a, b in zip(edges[:-1], edges[1:]):
            integral, error = quad_vec(lambda t: np.exp(event.logpdf(t) - offset) * density(t),
                                       a, b, epsabs=1e-12 / len(edges), epsrel=1e-7)
            if not np.isfinite(error) or error > max(1e-10, np.linalg.norm(integral) * 1e-5):
                raise ValueError('Measurement integration did not converge; WAIC is unavailable.')
            value = value + integral
        if np.any(value <= 0) or not np.isfinite(value).all():
            raise ValueError('Measurement integration underflowed or returned nonfinite values.')
        columns.append(np.log(value) + offset)
    if model == 'ippp_gp':
        integrated = np.sum((rates[:, :-1] + rates[:, 1:]) * np.diff(grid) / 2, axis=1)
        values = (np.sum(columns, axis=0) - integrated)[:, None] if columns else -integrated[:, None]
        return waic_from_log_likelihood(values, n_events=len(measurements), likelihood='observation_window', notes=[
            'The complete observation window is one predictive unit, including the integrated-intensity count term. Latent dates are integrated out; intensity is not normalized.',
            'One observed window provides no between-unit standard error. WAIC from a single window has limited predictive interpretation.',
            'Scores are for this model and observation window; no cross-tab comparability is assumed.'])
    return waic_from_log_likelihood(np.stack(columns, axis=1), n_events=len(measurements), likelihood='event_marginal', notes=[
        'Event-level measurement likelihoods integrate out latent dates using adaptive quadrature.',
        'WAIC assesses prediction, not convergence or chronological correctness. No cross-tab comparability is assumed.'])
