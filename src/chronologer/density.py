"""Callable form of the radiocarbon hierarchy exercised by test_pymc_models.

All times use negative BP. Bounds and prior scales are scientific inputs, not
sampling defaults. This is the common-bound variant covered by the execution
tests, not the notebook's optional per-event, calibration-derived truncation.
"""

from dataclasses import dataclass

import numpy as np
from scipy.stats import truncnorm

# Mixture defaults are intentionally centralized. The scale is independent of K.
MIXTURE_CONCENTRATION = 0.3
MIXTURE_LOG_SCALE_SD = 0.75
MIXTURE_SCALE_FRACTION = 0.2


def _measurement(event):
    """Bridge existing distribution objects to differentiable measurement logp.

    calrcarbon uses its existing cubic splines, including combined curve/error
    uncertainty. Frozen scipy normal/uniform distributions use their exact PDFs.
    All coordinates must already use the caller's common calendar convention.
    """
    import pymc as pm
    import pytensor.tensor as pt
    from .distributions import calrcarbon

    if isinstance(event, calrcarbon):
        if (event.c14_mean is None or event.c14_err is None
                or not np.isfinite([event.c14_mean, event.c14_err]).all() or event.c14_err <= 0):
            raise ValueError("Radiocarbon observations require finite mean and positive error.")
        mean_spline, error_spline = event._interp_mean, event._interp_error
        # Use the same spline evaluation as distributions.py, without a new
        # interpolation approximation or a non-differentiable Python callback.
        def spline(t, curve):
            knots = pt.as_tensor_variable(curve.x)
            coefficients = pt.as_tensor_variable(curve.c)
            index = pt.clip(pt.searchsorted(knots, t, side="right") - 1, 0, len(curve.x) - 2)
            delta = t - knots[index]
            return ((coefficients[0, index] * delta + coefficients[1, index]) * delta
                    + coefficients[2, index]) * delta + coefficients[3, index]
        def logp(t):
            safe = pt.clip(t, event.a, event.b)
            mean, error = spline(safe, mean_spline), spline(safe, error_spline)
            value = pm.logp(pm.Normal.dist(mu=mean, sigma=pt.sqrt(event.c14_err**2 + error**2)), event.c14_mean)
            return pt.switch((t >= event.a) & (t <= event.b), value, -np.inf)
        grid = np.linspace(event.a, event.b, 10000)
        log_weights = event.logpdf(grid)
        weights = np.exp(log_weights - np.max(log_weights))
        weights /= weights.sum()
        centre = float(grid @ weights)
        width = float(np.sqrt((grid - centre)**2 @ weights))
        return logp, centre, width
    name = getattr(getattr(event, "dist", None), "name", None)
    if name not in {"norm", "uniform"}:
        raise ValueError("Supported measurements are calrcarbon and frozen scipy normal/uniform distributions.")
    _, loc, scale = event.dist._parse_args(*event.args, **event.kwds)
    if not np.isfinite([loc, scale]).all() or scale <= 0:
        raise ValueError("Measurement location must be finite and scale positive.")
    if name == "norm":
        return lambda t: pm.logp(pm.Normal.dist(mu=loc, sigma=scale), t), float(loc), float(scale)
    return (lambda t: pm.logp(pm.Uniform.dist(lower=loc, upper=loc + scale), t),
            float(loc + scale / 2), float(scale / np.sqrt(12)))


def build_gaussian_mixture(events, K_max=5, *, prior_center=None, prior_scale=None):
    """Build an overfitted Gaussian mixture for latent event times.

    Accepts existing calrcarbon instances or scipy.stats frozen norm/uniform
    measurement distributions. Their PDFs are likelihoods for each latent time,
    not additional event-time priors. K_max is an upper complexity allowance.

    Means have an ordered iid Normal(center, scale) prior; weights have symmetric
    Dirichlet(0.3) prior; component SDs have LogNormal(log(0.2*scale), 0.75) prior.
    Defaults: center=mean measurement centres; scale=max(range of centres,
    median measurement SD). These data-scaled hyperpriors are empirical defaults,
    not a claim of noninformativeness. Explicit overrides support sensitivity work.
    """
    import pymc as pm
    import pytensor.tensor as pt
    events = list(events)
    if not events or type(K_max) is not int or not 1 <= K_max <= 20:
        raise ValueError("Supply at least one event and an integer K_max from 1 to 20.")
    measurements = [_measurement(event) for event in events]
    centres = np.array([item[1] for item in measurements])
    widths = np.array([item[2] for item in measurements])
    centre = float(centres.mean()) if prior_center is None else float(prior_center)
    scale = float(max(np.ptp(centres), np.median(widths))) if prior_scale is None else float(prior_scale)
    if not np.isfinite([centre, scale]).all() or scale <= 0:
        raise ValueError("Mixture prior center must be finite and prior scale positive.")
    with pm.Model(coords={"component": np.arange(K_max), "event": np.arange(len(events))}) as model:
        means = pm.Normal("means", mu=centre, sigma=scale, dims="component",
                          transform=pm.distributions.transforms.ordered,
                          initval=centre + scale * (np.linspace(-.5, .5, K_max) if K_max > 1 else np.zeros(1)))
        scales = pm.LogNormal("scales", mu=np.log(MIXTURE_SCALE_FRACTION * scale),
                              sigma=MIXTURE_LOG_SCALE_SD, dims="component")
        weights = (pm.Dirichlet("weights", a=np.full(K_max, MIXTURE_CONCENTRATION), dims="component")
                   if K_max > 1 else pm.Deterministic("weights", pt.ones(1), dims="component"))
        tau = pm.Mixture("tau", w=weights, comp_dists=pm.Normal.dist(mu=means, sigma=scales),
                         dims="event", initval=centres)
        pm.Potential("measurements", pt.sum(pt.stack([item[0](tau[i]) for i, item in enumerate(measurements)])))
    model.mixture_priors = dict(center=centre, scale=scale, concentration=MIXTURE_CONCENTRATION,
                                log_scale_sd=MIXTURE_LOG_SCALE_SD, scale_fraction=MIXTURE_SCALE_FRACTION)
    return model


def evaluate_mixture_density(posterior, grid):
    """Evaluate the actual Normal mixture for every draw, then summarize.

    Never renormalizes a cropped grid. The returned band is pointwise 95%, not
    simultaneous. PyMC's DataTree stays inside the standalone engine API.
    """
    from scipy.stats import norm
    grid = np.asarray(grid, dtype=float)
    if grid.ndim != 1 or grid.size < 2 or not np.isfinite(grid).all() or np.any(np.diff(grid) <= 0):
        raise ValueError("Density grid must be a finite, strictly increasing vector.")
    dataset = posterior["posterior"].to_dataset()
    arrays = [dataset[name].transpose("chain", "draw", "component").values.reshape(-1, dataset.sizes["component"])
              for name in ("means", "scales", "weights")]
    means, scales, weights = arrays
    values = np.zeros((len(means), len(grid)))
    for k in range(means.shape[1]):
        values += weights[:, k, None] * norm.pdf(grid[None, :], means[:, k, None], scales[:, k, None])
    if not np.isfinite(values).all():
        raise ValueError("Mixture density contains nonfinite values.")
    low, high = np.quantile(values, [.025, .975], axis=0)
    return dict(t_values=grid, pdf_values=values.mean(axis=0), lower_values=low, upper_values=high)


@dataclass
class GaussianMixtureFit:
    posterior: object
    density: dict
    priors: dict

    def evaluate_density(self, grid):
        return evaluate_mixture_density(self.posterior, grid)


def fit_gaussian_mixture(events, K_max=5, *, grid=None, prior_center=None, prior_scale=None,
                         draws=250, tune=250, chains=2, random_seed=912, cores=1, progress_callback=None):
    """Fit ordinary continuous PyMC NUTS; return posterior and density summary.

    Component allocations are analytically marginalized by pm.Mixture. No
    discrete sampler or inferred integer K is used. Default grid covers all
    retained component means +/- six SDs; supply a grid for a chosen domain.
    Short default runs establish execution, not convergence.
    """
    import pymc as pm
    if type(cores) is not int or cores < 1:
        raise ValueError('cores must be a positive integer.')
    total = chains * (tune + draws)
    counts = [0] * chains
    def report(stage, completed=0):
        if progress_callback:
            progress_callback(dict(stage=stage, completed=completed, total=total))
    def on_draw(trace, draw):
        counts[draw.chain] = draw.draw_idx + 1
        report(f"{'Tuning' if draw.tuning else 'Sampling'} · chain {draw.chain + 1}/{chains}",
               sum(counts))
    report("Building mixture model")
    model = build_gaussian_mixture(events, K_max, prior_center=prior_center, prior_scale=prior_scale)
    with model:
        report("Compiling and initializing")
        trace = pm.sample(draws=draws, tune=tune, chains=chains, cores=min(cores, chains), blas_cores='auto', random_seed=random_seed,
                          nuts_sampler="pymc", init="adapt_diag", target_accept=.95,
                          progressbar=False, compute_convergence_checks=False,
                          callback=on_draw if progress_callback else None)
    report("Evaluating density", total)
    if grid is None:
        data = trace["posterior"].to_dataset()
        means, scales = data["means"].values, data["scales"].values
        grid = np.linspace(np.min(means - 6 * scales), np.max(means + 6 * scales), 2048)
    return GaussianMixtureFit(trace, evaluate_mixture_density(trace, grid), model.mixture_priors)


@dataclass
class DensityFit:
    posterior: object  # PyMC 6 xarray.DataTree, kept in Python only
    density: dict      # arrays derived from the sampled population parameters


def build_radiocarbon_density(radiocarbon_ages, radiocarbon_errors, calcurve, *,
                              lower, upper, mean_prior, mean_prior_sd, sd_prior_scale):
    """Build the existing single truncated-normal hierarchy (no sampling).

    tau_mu ~ TruncatedNormal(mean_prior, mean_prior_sd, lower, upper)
    tau_sd ~ HalfNormal(sd_prior_scale)
    tau[i] ~ TruncatedNormal(tau_mu, tau_sd, lower, upper)
    r_latent[i] ~ Normal(interpolated curve mean, interpolated curve sigma)
    r_measured[i] ~ Normal(r_latent[i], radiocarbon_errors[i])
    """
    import pymc as pm
    import pytensor.tensor as pt
    from .pymccarbon import interpolate_calcurve

    ages = np.asarray(radiocarbon_ages, dtype=float)
    errors = np.asarray(radiocarbon_errors, dtype=float)
    curve = [np.asarray(calcurve[key], dtype=float) for key in ("calbp", "c14bp", "c14_sigma")]
    if ages.ndim != 1 or not ages.size or errors.shape != ages.shape:
        raise ValueError("Supply equally sized, nonempty age and error vectors.")
    if not np.isfinite(ages).all() or not np.isfinite(errors).all() or np.any(errors <= 0):
        raise ValueError("Ages and errors must be finite, with positive errors.")
    if any(a.ndim != 1 or a.shape != curve[0].shape or not np.isfinite(a).all() for a in curve):
        raise ValueError("Curve arrays must be finite vectors of equal length.")
    if curve[0].size < 2 or np.any(np.diff(curve[0]) <= 0) or np.any(curve[2] <= 0):
        raise ValueError("Curve times must increase and curve errors must be positive.")
    if not np.isfinite([lower, upper, mean_prior, mean_prior_sd, sd_prior_scale]).all():
        raise ValueError("Bounds and prior settings must be finite.")
    if not curve[0][0] <= lower < upper < curve[0][-1]:
        raise ValueError("Bounds must increase within the curve; the younger bound must precede its final grid point.")
    if mean_prior_sd <= 0 or sd_prior_scale <= 0:
        raise ValueError("Prior scales must be positive.")
    with pm.Model() as model:
        mu = pm.TruncatedNormal("tau_mu", mu=mean_prior, sigma=mean_prior_sd, lower=lower, upper=upper)
        sd = pm.HalfNormal("tau_sd", sigma=sd_prior_scale)
        tau = pm.TruncatedNormal("tau", mu=mu, sigma=sd, lower=lower, upper=upper, shape=ages.shape)
        mean, sigma = interpolate_calcurve(tau, *(pt.as_tensor_variable(a) for a in curve))
        latent = pm.Normal("r_latent", mu=mean, sigma=sigma, shape=ages.shape)
        pm.Normal("r_measured", mu=latent, sigma=errors, observed=ages, shape=ages.shape)
    return model


def fit_radiocarbon_density(radiocarbon_ages, radiocarbon_errors, calcurve, *,
                            lower, upper, mean_prior, mean_prior_sd, sd_prior_scale,
                            draws=250, tune=250, chains=2, random_seed=912, cores=1,
                            progress_callback=None):
    """Fit with ordinary PyMC NUTS and return posterior plus density arrays.

    The curve is the posterior mean of the common-bound truncated-normal
    population density. The band is its pointwise 95% equal-tail credible
    interval, not an event-date HDI or a simultaneous credible band.
    Defaults are an execution benchmark, not a convergence guarantee.
    Optional progress_callback receives stage/completed/total dictionaries.
    Counts include tuning and posterior draws; compilation has no percentage.
    """
    import pymc as pm

    if type(cores) is not int or cores < 1:
        raise ValueError('cores must be a positive integer.')
    total = chains * (tune + draws)
    counts = [0] * chains
    def report(stage, completed=0):
        if progress_callback:
            progress_callback(dict(stage=stage, completed=completed, total=total))

    def on_draw(trace, draw):
        counts[draw.chain] = draw.draw_idx + 1
        completed = sum(counts)
        report(f"{'Tuning' if draw.tuning else 'Sampling'} · chain {draw.chain + 1}/{chains}", completed)

    report("Building model")
    model = build_radiocarbon_density(
        radiocarbon_ages, radiocarbon_errors, calcurve, lower=lower, upper=upper,
        mean_prior=mean_prior, mean_prior_sd=mean_prior_sd, sd_prior_scale=sd_prior_scale)
    with model:
        report("Compiling and initializing")
        trace = pm.sample(draws=draws, tune=tune, chains=chains, cores=min(cores, chains), blas_cores='auto',
                          random_seed=random_seed, nuts_sampler="pymc", init="adapt_diag",
                          progressbar=False, compute_convergence_checks=False,
                          callback=on_draw if progress_callback else None)
    report("Evaluating density", total)
    posterior = trace["posterior"].to_dataset()
    means = posterior["tau_mu"].values.reshape(-1, 1)
    scales = posterior["tau_sd"].values.reshape(-1, 1)
    grid = np.linspace(lower, upper, 512)
    densities = truncnorm.pdf(grid[None, :], (lower - means) / scales,
                             (upper - means) / scales, loc=means, scale=scales)
    if not np.isfinite(densities).all():
        raise ValueError("Posterior density contains non-finite values.")
    low, high = np.quantile(densities, [.025, .975], axis=0)
    return DensityFit(trace, {"t_values": grid, "pdf_values": densities.mean(axis=0),
                              "lower_values": low, "upper_values": high})
