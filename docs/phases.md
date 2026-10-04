# Phases

A phase is the distribution of a labelled group's latent event dates. Its
boundaries are derived distribution quantiles, not parameters that define the
phase. Version 1 supports uniform and normal (`gaussian` is an alias) phases.

```python
from scipy.stats import norm
from chronologer import Phase, build_phase, fit_phase, group_phases

rows = [{"label": "occupation"}, {"label": "occupation"}, {"label": "repair"}]
measurements = [norm(-2500, 30), norm(-2550, 40), norm(-2400, 20)]
phases = {
    "occupation": Phase("uniform", prior_center=-2525, prior_scale=100),
    "repair": Phase("normal", prior_center=-2400, prior_scale=100),
}
groups = group_phases(rows)  # also accepts a DataFrame with a label column
model = build_phase(rows, phases, measurements=measurements)  # no sampling
# When ready to sample:
# result = fit_phase(rows, phases, measurements=measurements)
# posterior = result["posterior"].to_dataset()
# label = "occupation"
# limits = phases[label].interval(
#     .05, .95, posterior.mu.sel(phase=label), posterior.scale.sel(phase=label))
```

Measurements use the existing `calrcarbon` or frozen SciPy normal/uniform objects,
aligned with row order. Each radiocarbon event retains its calibration curve and
combined curve/measurement uncertainty. All coordinates pass through unchanged;
radiocarbon uses the engine's native negative-BP convention. App-side date and
curve conversion remains with the caller.

`mu` is the center of a uniform phase or the mean of a normal phase. `scale` is
the full positive width or positive sigma, respectively. Uniform limits are
`mu - scale/2` and `mu + scale/2`, available through `interval(0, 1, mu, scale)`.
Normal limits are infinite; use any chosen finite quantiles instead.
Queries on posterior arrays return a quantile for each draw. An interval between
these quantiles describes the fitted event distribution, not uncertainty on a
boundary estimate. Summarizing uncertainty across those draws is a separate query.

The location prior is Normal(prior_center, prior_scale). Positive scales use the
mixture model's LogNormal reference SD and log dispersion, converting SD to full
width for uniform phases. Omitted prior settings reuse the mixture's empirical
rule within each group. See `Phase`'s docstring for the exact priors; pass explicit
settings for sensitivity analyses.

The mapping keys are the existing labels, not a separate membership system.
The joint model retains a labelled `phase` coordinate and original `event`
order. First-occurrence label order is indexing only; it implies no precedence.

Optional ordering is expressed separately from the phase distributions:

```python
from chronologer import Order

orders = [Order("occupation", "repair", anchors=(.95, .05), delta_scale=100)]
model = build_phase(rows, phases, measurements=measurements, orders=orders)
# fit_phase accepts the same orders argument.
```

For `Order(A, B, anchors=(p, q))`, the model enforces the relationship by
reparameterization:

```
Q_B(q) = Q_A(p) + delta
mu_B = mu_A + scale_A * z_A(p) + delta - scale_B * z_B(q)
```

Here `z(p)` is `p - .5` for uniform phases and the standard normal quantile for
normal phases. Thus the anchors depend on phase width/sigma, not only location.
Default anchors `(.5, .5)` order centers; `(1, 0)` orders exact uniform end/start
limits; `(.95, .05)` orders effective normal end/start quantiles. Normal anchors
must have finite quantiles, so probabilities 0 and 1 are rejected for ordering.

`delta ~ HalfNormal(delta_scale)` is a positive anchor separation in the existing
calendar units. With native negative BP, adding delta moves toward younger
dates. An omitted `delta_scale` uses the larger resolved prior scale of the two
phases. The downstream independent location prior is **replaced**, not retained;
only each chain's root has a free Normal location prior. Width/sigma priors are
unchanged. The posterior's `delta` coordinate `order` indexes the input relationship
sequence; `mu` still covers all labels, including derived locations.

Simple disjoint chains such as `A -> B -> C` are supported, even when the input
relationships are not listed chronologically. Cycles, branching, and multiple
predecessors are rejected. There are no direct inequality potentials, overlap
constraints or Harris-matrix logic. Delta represents an intervening period only
when the selected anchors are end/start; an explicit intervening period can
instead be represented as another labelled phase.
