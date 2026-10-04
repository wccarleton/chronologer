"""Event-level predictive assessment, integrating out latent event dates."""
import numpy as np
from scipy.special import logsumexp
from scipy.stats import norm

from .distributions import calrcarbon


def _mass(lower, upper):
    """Stable log probability of a standard-normal interval."""
    a = np.where(lower > 0, norm.logsf(lower), norm.logcdf(upper))
    b = np.where(lower > 0, norm.logsf(upper), norm.logcdf(lower))
    with np.errstate(divide='ignore', invalid='ignore'):
        return a + np.log(-np.expm1(b - a))


def _radiocarbon(event, family, mu, scale):
    """Composite Gaussian quadrature on the existing curve's spline intervals.

    Integrate the unnormalized measurement likelihood, never a calibrated PDF.
    Normal integration is restricted to ten sigma (omitted mass < 2e-23).
    Doubling quadrature order must converge relatively to 1e-6 for every draw.
    """
    radius = scale / 2 if family == 'uniform' else 10 * scale
    start, end = np.maximum(mu - radius, event.a), np.minimum(mu + radius, event.b)
    if np.any(start >= end):
        raise ValueError('Phase support does not intersect the calibration curve.')
    knots = event._interp_mean.x
    active = (knots[:-1] < end.max()) & (knots[1:] > start.min())
    left = np.maximum(start[:, None], knots[:-1][active])
    right = np.minimum(end[:, None], knots[1:][active])
    half = np.maximum(right - left, 0) / 2
    midpoint = np.clip((left + right) / 2, event.a, event.b)
    previous = None
    for order in (8, 16, 32, 64):
        nodes, weights = np.polynomial.legendre.leggauss(order)
        t = midpoint[..., None] + half[..., None] * nodes
        logp = event.logpdf(t)
        logp += (-np.log(scale)[:, None, None] if family == 'uniform'
                 else norm.logpdf(t, mu[:, None, None], scale[:, None, None]))
        with np.errstate(divide='ignore'):
            value = logsumexp(logp + np.log(half)[..., None] + np.log(weights), axis=(1, 2))
        if previous is not None and np.all(np.abs(value - previous) < 1e-6):
            return value
        previous = value
    raise ValueError('Radiocarbon predictive integration did not converge.')


def _loglik(event, family, mu, scale):
    if isinstance(event, calrcarbon):
        return _radiocarbon(event, family, mu, scale)
    _, loc, error = event.dist._parse_args(*event.args, **event.kwds)
    if event.dist.name == 'norm':
        if family == 'normal':
            return norm.logpdf(loc, mu, np.hypot(scale, error))
        return _mass((mu - scale / 2 - loc) / error,
                     (mu + scale / 2 - loc) / error) - np.log(scale)
    if event.dist.name == 'uniform':
        if family == 'normal':
            return _mass((loc - mu) / scale, (loc + error - mu) / scale) - np.log(error)
        overlap = np.maximum(0, np.minimum(mu + scale / 2, loc + error)
                             - np.maximum(mu - scale / 2, loc))
        with np.errstate(divide='ignore'):
            return np.log(overlap) - np.log(scale) - np.log(error)
    raise ValueError('Unsupported phase measurement.')


def waic(data, phases, *, measurements, posterior):
    """Whole-model WAIC for predicting measured events in their supplied phases.

    One pointwise contribution per event; integrate out its latent date given
    each posterior location/scale draw. Label membership is held fixed. Uses
    The standard pointwise definition is WAIC = -2 * elpd_waic (smaller is better).
    Compare only identical observations, membership and measurement conventions.
    Returns summaries only; does not modify the model or retain likelihood draws.
    """
    records = data.to_dict('records') if hasattr(data, 'columns') else list(data)
    measurements = list(measurements)
    if not records or len(records) != len(measurements):
        raise ValueError('Supply one measurement per labelled event.')
    shape = (posterior.sizes['chain'], posterior.sizes['draw'], len(records))
    if shape[0] * shape[1] < 2:
        raise ValueError('WAIC requires at least two retained draws.')
    loglik = np.empty(shape)
    for i, (row, event) in enumerate(zip(records, measurements)):
        label = row['label']
        mu = posterior['mu'].sel(phase=label).transpose('chain', 'draw').values.reshape(-1)
        scale = posterior['scale'].sel(phase=label).transpose('chain', 'draw').values.reshape(-1)
        values = np.empty_like(mu)
        for start in range(0, len(mu), 32):
            stop = start + 32
            values[start:stop] = _loglik(event, phases[label].distribution, mu[start:stop], scale[start:stop])
        loglik[:, :, i] = values.reshape(shape[:2])
    if not np.isfinite(loglik).all():
        raise ValueError('Predictive log likelihood is nonfinite for some retained draws.')
    # ArviZ 1.x no longer exposes waic; use its conventional pointwise formula.
    samples = loglik.reshape(-1, len(records))
    penalty = samples.var(axis=0)
    pointwise = logsumexp(samples, axis=0) - np.log(len(samples)) - penalty
    elpd = float(pointwise.sum())
    unreliable = bool(np.any(penalty > .4))
    notes = ['Pointwise log-likelihood variance exceeds 0.4; WAIC may be unreliable.'] if unreliable else []
    if len(records) < 2:
        notes.append('With one event, the between-event standard error cannot be estimated meaningfully.')
    return dict(waic=-2 * elpd, se=float(2 * np.sqrt(len(records) * pointwise.var())),
                elpd_waic=elpd, p_waic=float(penalty.sum()),
                n_events=len(records), n_samples=shape[0] * shape[1],
                warning=unreliable, notes=notes,
                likelihood='event_marginal')
