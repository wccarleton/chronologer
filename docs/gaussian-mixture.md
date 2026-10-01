# Gaussian mixture density API

`build_gaussian_mixture(events, K_max=5, *, prior_center=None, prior_scale=None)`
returns a PyMC Model. `fit_gaussian_mixture` accepts the same inputs plus `grid`,
`draws`, `tune`, `chains`, `random_seed`, and `progress_callback`; it returns a
`GaussianMixtureFit` with `posterior` (PyMC 6 DataTree), `density`, and `priors`.
`fit.evaluate_density(grid)` / `evaluate_mixture_density(posterior, grid)`
evaluate the retained posterior on a strictly increasing calendar-time grid.
The existing radiocarbon density functions are unchanged.

Events are existing `calrcarbon` objects or frozen SciPy `norm` / `uniform`
measurement distributions, in consistent time coordinates. Each object's PDF
is used as the measurement likelihood for its latent time. The API does not
apply an additional event-specific date prior. Do not supply posterior densities
with substantive priors already incorporated as if they were likelihoods.
Unsupported distribution families raise an explicit error.

For radiocarbon, the symbolic likelihood evaluates the **existing cubic spline
coefficients** and combines measurement and curve errors in quadrature, matching
`calrcarbon.logpdf`. Curve-support violations have log likelihood minus infinity.
There is no calibration-grid approximation to this likelihood. The spline class
cache in `distributions.py` is unchanged: all radiocarbon objects in a process
must use the same curve. ChronoApp checks this and uses fresh worker processes.
Adding further measurement families is confined to the engine measurement bridge.

## Model and exact defaults

Let `c_i` and `s_i` be each measurement distribution's centre and SD. For
radiocarbon these summaries use its existing likelihood on 10,000 points over
the calibration curve's support, normalized only for these prior summaries.
For normal/uniform observations they are analytic.

- `C = mean(c_i)`; `S = max(ptp(c_i), median(s_i))`.
- Component means: iid `Normal(C, S)`, restricted to ascending order.
- Component SDs: `LogNormal(log(0.2*S), 0.75)` (natural log; 0.75 is log-SD).
- Weights: `Dirichlet([0.3] * K_max)`. At K=1 the weight is identically one.
- Each latent time: `Mixture(weights, Normal(means, SDs))`.
- Measurement log likelihoods are added with `pm.Potential`.

These empirical, data-scaled hyperpriors are defaults, **not noninformative
priors**. `prior_center` and `prior_scale` override C/S for sensitivity analyses.
The constants live together in `density.py`; component scale does not change
when K changes. The lognormal discourages arbitrarily collapsing scales while
remaining positive. Concentration 0.3 encourages sparse weights, without a
guarantee of a particular number of active components.

PyMC's ordered transform enforces ascending means throughout sampling, using
positive exponential increments; ordered initial values alone would not suffice.
Allocations are analytically marginalized by `pm.Mixture`; there is no discrete K,
allocation sampler, RJMCMC, or model-per-K fitting. Ordering prevents ordinary
label switching but does not eliminate weakly identified or near-empty components.

Fits use ordinary PyMC NUTS, `init='adapt_diag'`, one core, `target_accept=0.95`,
250 tune + 250 retained draws per chain, two chains, seed 912 by default.
These are short development runs; convergence is not assessed automatically.

## Density output

Each retained draw supplies a full weighted sum of Normal PDFs on the requested
grid. `pdf_values` averages those densities; `lower_values` / `upper_values` are
pointwise 2.5% / 97.5% quantiles. This is neither a sum of calibrated PDFs nor a
Normal evaluated at posterior-mean parameters. No finite-grid renormalization is
performed. Default grid: 2,048 points spanning all retained means +/- six SDs.
For very narrow components across a wide domain, request a denser grid and check
numerical integration. Gaussian support is unbounded; the measurement curve's
finite support is not a truncation of the mixture itself.

Component indices, even with ordering, are computational devices. The total
posterior temporal density is the scientific output; weights do not by themselves
establish archaeological groups, occupations, or phases.
