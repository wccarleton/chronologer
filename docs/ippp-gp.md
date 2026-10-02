# Basic GP IPPP benchmark

```python
import chronologer as ch
from scipy.stats import norm

fit = ch.fit(
    [norm(-2400, 40), norm(-2000, 30)],
    model=ch.models.ippp.gp,
    params={"start": -3000, "end": -1000, "grid_size": 32},
    mcmc_config={"draws": 1000, "tune": 1000, "chains": 4},
)
# Equivalent: ch.models.ippp.gp(data, params=..., mcmc_config=...)
```

`start` and `end` are mandatory, finite, increasing calendar coordinates. They are
never derived from input dates, calibrated tails, or prior summaries. For negative
BP coordinates, -3000 is the older start and -1000 the younger end. The declared
window describes when events could have been observed, including observed empty
time; it is not a display crop or an inferred activity boundary.

Data may contain exact numeric calendar times, existing `calrcarbon` objects, or
frozen scipy normal/uniform measurement distributions. Per-event calibration
curves use their existing shared spline references. Exact times outside the window
and measurements with disjoint support are rejected. No dates are silently dropped.
Uncertain dates retain their measurement likelihoods within the declared window.
The engine also accepts an empty list to represent a genuinely observed empty
window; the initial app benchmark requires at least one selected event.

## Model and approximation

This is a log-Gaussian Cox process: conditional on the sampled intensity, events
follow an inhomogeneous Poisson point process. The likelihood is

`sum(log(lambda(tau_i))) - integral_start^end lambda(t) dt`.

It retains count information. There is no division by total intensity or
conditioning on the observed count. The likelihood assumes complete observation
across the entire period. Preservation, detection probability, sampling effort,
selection and duplicate observations of the same event are not modeled.

An exponentiated-quadratic GP defines log intensity at `grid_size` equally spaced
nodes, including both observation endpoints. Positive node intensities are
linearly interpolated in calendar time. The trapezoid integral is **exact for this
piecewise-linear intensity**, but that intensity is a finite-grid approximation to
a continuous GP process. Event likelihood evaluation and integration use the same
interpolation. Check grid resolution before scientific interpretation. Increasing
nodes changes the approximation and increases dense-GP computation; the benchmark
accepts 4–256 nodes, default 32. No bins/count aggregation replace event dates.

Let `T = end - start`. Defaults, independent of the observed count, are:

- `log_rate ~ Normal(log(baseline_count / T), log_rate_sd)`, with
  `baseline_count=10`, `log_rate_sd=1.5`.
- `amplitude ~ HalfNormal(amplitude_scale)`, with `amplitude_scale=1`.
- `length_scale ~ LogNormal(log(length_scale_median), length_scale_log_sd)`,
  with median `T/5` years and log SD `0.5`.
- Zero-mean GP offset with covariance
  `amplitude² * exp(-(t-t')² / (2*length_scale²))`.
- Node intensity `exp(log_rate + offset)` in events per calendar unit.

`baseline_count` centers the baseline log-rate prior; it is **not** the prior mean
of the integrated intensity. GP variation and lognormal transformations matter.
All five prior settings can be overridden through `params` (the app displays the
defaults but does not yet expose prior editing). Changing the window changes the
default time scale and baseline-rate prior too; explicit prior overrides allow
controlled sensitivity analyses.

Uncertain dates use bounded Uniform coordinates over window/measurement-support
intersections. Their normalization constants are cancelled, leaving measurement
likelihoods times the point-process likelihood, without an extra event-time prior.
These intersections never change the observation window used for the rate integral.

Implementation uses [PyMC's latent GP](https://www.pymc.io/projects/docs/en/stable/api/gp/generated/pymc.gp.Latent.html)
with its reparameterized representation and default numerical jitter. Ordinary
PyMC NUTS uses one core, `adapt_diag`, target acceptance 0.95, seed 912. Standalone
sampling defaults remain 250/250/2; the app supplies its editable 1000/1000/4 defaults.
No new sampler or backend is added.

## Results and verification

`GPFit` contains `posterior` (PyMC's DataTree), `intensity` (grid, posterior mean
rate, pointwise 95% bounds), and `specification` (resolved window/grid/prior settings).
Intensity is **not area-normalized**: its integral is the expected event count.
Posterior variables include `log_rate`, `amplitude`, `length_scale`, `gp_offset`,
node `intensity`, `integrated_intensity`, and `tau` for nonempty data. Uncertain dates
also have `tau_unknown`. `build_gp(data, start=..., end=..., ...)` exposes the model
for logp/gradient checks or advanced Python use; existing IPPP helpers are unchanged.

Tests check the full likelihood against analytic constant intensity, gradients
against analytic/finite-difference values, empty exposure, missing/reversed bounds,
mixed calibration curves, logp/gradient compilation, and tiny ordinary NUTS runs.
These are execution and numerical-coherence checks, not evidence of convergence
or validated prior/grid choices for a particular archaeological dataset.
