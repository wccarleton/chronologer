# Concise model API

Model families live under `chronologer.models`. `density.single_density` and
`density.gmixture` are fitting functions, not classes or model-name strings.
Use them directly or through the small `chronologer.fit` delegator:

Both models live in the same `chronologer.density` implementation module.
`build_single_density(events, **params)` and `fit_single_density(events, **params)`
accept existing `calrcarbon` and frozen SciPy `norm`/`uniform` measurements,
including mixed types and different calibration curves in one fit. The event
distribution is one truncated normal within explicitly supplied bounds, with
`mean_prior`, `mean_prior_sd` and `sd_prior_scale` unchanged. The generic single
model reuses the mixture's `_measurement` likelihoods, including cubic curve
splines and combined curve/laboratory error. This differs from the legacy
radiocarbon hierarchy's linear interpolation and explicit `r_latent` variable.

The old `build_radiocarbon_density`, `fit_radiocarbon_density`, and mapping-based
`models.density.single` calls preserve that original hierarchy and return type.
The new `models.density.single_density` also accepts legacy mappings, delegating
to the original fitter. Sequence input uses the new generic builder. Both return
`DensityFit` and share the same sampler, density calculation, priors and native
calendar convention. There is no simulation path in this patch.

```python
import chronologer as ch
from scipy.stats import norm

events = [norm(-2500, 30), norm(-2550, 40)]
result = ch.fit(
    events,
    model=ch.models.density.gmixture,
    params={"K_max": 5},
    mcmc_config={"draws": 1000, "tune": 1000, "chains": 4},
)
# Equivalent direct call:
result = ch.models.density.gmixture(
    events, params={"K_max": 5},
    mcmc_config={"draws": 1000, "tune": 1000, "chains": 4},
)
```

The single-density model retains its existing radiocarbon-specific inputs, grouped
in a mapping rather than converted into a new event representation:

```python
data = {
    "radiocarbon_ages": [-2500, -2550],
    "radiocarbon_errors": [30, 40],
    "calcurve": ch.load_calcurve("intcal20"),
}
result = ch.models.density.single(
    data,
    params={"lower": -3500, "upper": -1500, "mean_prior": -2500,
            "mean_prior_sd": 500, "sd_prior_scale": 400},
    mcmc_config={"draws": 1000, "tune": 1000, "chains": 4},
)
```

Inputs retain native negative-BP coordinates. Mixture data remains a sequence of
the supported measurement distribution objects, including `calrcarbon` with its
own curve's spline references. There is no app-specific event schema in the engine.

| Argument | Meaning |
| --- | --- |
| `data` | Existing model-specific inputs, as described above |
| `params` | Single: the five required bounds/prior arguments; mixture: optional `K_max`, `prior_center`, `prior_scale`, `grid` |
| `mcmc_config` | Optional `draws`, `tune`, `chains`, `random_seed`, `cores` |
| `progress_callback` | Optional separate runtime callback receiving stage/completed/total dictionaries |

Unknown or misplaced options raise `TypeError`; scientific input validation
remains in the existing implementation. The caller copies option mappings and
does not mutate data. Omitted sampling options retain **standalone engine** defaults:
250 draws, 250 tuning iterations, two chains, seed 912, one core. Positive integer
`cores` enables PyMC parallel chains (capped by chain count), with `blas_cores="auto"`.
Progress counts sum completed iterations across chains. ChronoApp supplies its own
explicit user-selected settings (new summaries start at 1000/1000/4).

Return objects are unchanged: `chronologer.density.DensityFit` and
`chronologer.density.GaussianMixtureFit`, including their PyMC DataTree, density
arrays, and mixture-specific priors/evaluation method. No priors, likelihoods,
parameterizations, samplers or calibration behavior change.

Existing top-level `build_*`/`fit_*` functions and `chronologer.density` imports
continue working without deprecation warnings. The old `chronologer.models`
likelihood imports (`approx_integral`, `ippp_logp_sine`, `ippp_logp_lm`) remain
available and also live under `chronologer.models.ippp`. The new
[`ippp.gp` benchmark](ippp-gp.md) uses the same fitting signature and requires
explicit observation `start` and `end`. No registry,
backend abstraction or new inference dependency is introduced.

## Single-density simulation

`simulate_single_density` generates independent prior-predictive datasets with
PyMC, using the same truncated-normal population and hyperpriors as inference:

```python
simulation = ch.simulate_single_density(
    6, distribution="calrcarbon", error=30,
    calcurve=ch.load_calcurve("intcal20"),
    lower=-3500, upper=-1500, mean_prior=-2500,
    mean_prior_sd=500, sd_prior_scale=400, draws=1000,
)
```

Supported measurement families are `calrcarbon`, `normal`, and `uniform`.
`error` is measurement SD; uniform half-width is `sqrt(3) * error`.
Radiocarbon simulation reuses the measurement splines and combines laboratory
and curve uncertainty. Bounds must lie inside the chosen calibration curve.
All dates use native negative-BP coordinates.

Each replicate draws new `tau_mu` and `tau_sd`, latent `tau` dates, and noisy
`measured` dates. There is no conditioning on observations, MCMC, or tuning.
The `DensitySim` result contains `prior` (a PyMC DataTree with a `prior` group)
and `density` (the existing mean and pointwise 95% density summary arrays).
`simulate_radiocarbon_density(n, calcurve, ...)` is a convenience wrapper.
