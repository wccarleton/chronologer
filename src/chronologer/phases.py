"""Labelled event distributions; boundaries are queries, not model parameters."""
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm, uniform

from .density import (_measurement, _priors, MIXTURE_LOG_SCALE_SD,
                      MIXTURE_SCALE_FRACTION)

__all__ = ["Phase", "Order", "group_phases", "build_phase", "fit_phase", "waic"]


def waic(data, phases, *, measurements, posterior):
    """Event-level WAIC with latent dates integrated out; see phase_stats.waic."""
    from .phase_stats import waic as evaluate
    return evaluate(data, phases, measurements=measurements, posterior=posterior)


@dataclass(frozen=True)
class Phase:
    """A uniform or normal event distribution and its hyperprior settings.

    The phase name is the label used as its key in build_phase's phases mapping.
    Coordinates are unchanged (native negative BP for radiocarbon); no datum
    conversion occurs. ``gaussian`` is an alias for ``normal``.

    Unordered/root mu has Normal(prior_center, prior_scale) prior; an Order
    replaces the downstream location prior with a derived location. The positive scale has a
    LogNormal prior with log SD 0.75, reusing the mixture's 0.2 * prior_scale
    reference SD. Scale means sigma for normal phases and full width for uniform
    phases; the latter's reference width is sqrt(12) times the reference SD.
    Omitted settings use the mixture's empirical rule within this label group:
    mean measurement centres and max(centre range, median measurement SD).
    These are data-scaled priors, not noninformative priors.

    Each non-root phase owns one input delta with HalfNormal(delta_scale) prior.
    At a merge its older/receiving anchor is the exact maximum selected
    predecessor anchor plus that delta. Root phases have no delta; their
    delta_scale setting is unused. Omitted delta_scale falls back to legacy
    Order settings, then the largest resolved prior scale of related phases.
    """
    distribution: str = "normal"
    prior_center: float | None = None
    prior_scale: float | None = None
    delta_scale: float | None = None

    def __post_init__(self):
        if self.distribution == "gaussian":
            object.__setattr__(self, "distribution", "normal")
        if self.distribution not in {"uniform", "normal"}:
            raise ValueError("Phase distribution must be uniform or normal.")
        if self.prior_center is not None and not np.isfinite(self.prior_center):
            raise ValueError("Phase prior center must be finite.")
        if self.prior_scale is not None and (
                not np.isfinite(self.prior_scale) or self.prior_scale <= 0):
            raise ValueError("Phase prior scale must be finite and positive.")
        if self.delta_scale is not None and (not np.isfinite(self.delta_scale) or self.delta_scale <= 0):
            raise ValueError('Phase input delta scale must be finite and positive.')

    def quantile(self, p, mu, scale):
        """Query any p in [0, 1] for numeric parameters or posterior arrays.

        mu is the center/mean; scale is full width/sigma, respectively. For
        uniform phases p=0 and p=1 return the endpoints. Normal limits are
        infinite. Posterior queries are per draw, not quantiles of a pooled
        posterior predictive distribution or credible intervals on parameters.
        """
        standardized = self._offset(p)
        if (not np.isfinite(mu).all() or not np.isfinite(scale).all()
                or np.any(np.asarray(scale) <= 0)):
            raise ValueError("Phase parameters must be finite with positive scale.")
        return mu + scale * standardized

    def _offset(self, p):
        """Standardized quantile shared by numeric queries and model algebra."""
        p = np.asarray(p, dtype=float)
        if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
            raise ValueError("Quantile probabilities must lie in [0, 1].")
        return uniform.ppf(p) - .5 if self.distribution == "uniform" else norm.ppf(p)

    def interval(self, p, q, mu, scale):
        """Return the two distribution quantiles for scalar p <= q."""
        if np.ndim(p) or np.ndim(q) or p > q:
            raise ValueError("Interval probabilities must be scalars with p <= q.")
        return self.quantile(p, mu, scale), self.quantile(q, mu, scale)


@dataclass(frozen=True)
class Order:
    """Relate quantile anchors on two labelled phases by a positive delta.

    Order(before, after, anchors=(p, q)) selects predecessor and receiver anchors.
    The receiver owns one delta: Q_after(q) = max(predecessor anchors) + delta.
    For a chain this is Q_after(q) = Q_before(p) + delta
    in the caller's existing coordinates. In native negative BP, before is
    older and after is younger. Defaults order medians; (1, 0) orders uniform
    endpoints and (.95, .05) orders normal effective end/start quantiles.
    Normal anchors must be strictly inside (0, 1), since limits are infinite.

    delta ~ HalfNormal(Phase.delta_scale), in calendar-time units. This Order's
    delta_scale remains a legacy fallback; incoming edges must agree on it.
    If omitted, its scale is max(resolved prior_scale of all related phases), using the same
    explicit or empirical settings as Phase. delta is an anchor separation,
    not necessarily an intervening period. The downstream mu is derived;
    its former independent Normal prior is not also imposed.
    """
    before: str
    after: str
    anchors: tuple[float, float] = (.5, .5)
    delta_scale: float | None = None

    def __post_init__(self):
        if (not isinstance(self.before, str) or not self.before.strip()
                or not isinstance(self.after, str) or not self.after.strip()
                or self.before == self.after):
            raise ValueError("Order requires two distinct nonempty phase labels.")
        anchors = np.asarray(self.anchors, dtype=float)
        if (anchors.shape != (2,) or not np.isfinite(anchors).all()
                or np.any((anchors < 0) | (anchors > 1))):
            raise ValueError("Order anchors must be two probabilities in [0, 1].")
        object.__setattr__(self, "anchors", tuple(anchors))
        if self.delta_scale is not None and (
                not np.isfinite(self.delta_scale) or self.delta_scale <= 0):
            raise ValueError("delta_scale must be finite and positive.")


def group_phases(data):
    """Group records (or DataFrame rows) by their existing nonempty label.

    Return a dict of label -> records in first-occurrence order. Order is
    retained for stable identity/indexing, not interpreted as chronology.
    No membership identifiers are added and records are not modified.
    """
    records = data.to_dict("records") if hasattr(data, "columns") else list(data)
    groups = {}
    for event in records:
        label = event["label"]
        if not isinstance(label, str) or not label.strip():
            raise ValueError("Phase events require a nonempty string label.")
        groups.setdefault(label, []).append(event)
    return groups


def _orders(phases, orders):
    """Resolve a DAG from specs, independent of event membership or layout."""
    incoming = {label: [] for label in phases}
    pairs = set()
    for order in orders:
        if not isinstance(order, Order):
            raise TypeError('orders must contain Order objects.')
        if order.before not in phases or order.after not in phases:
            raise ValueError('Order labels must refer to existing phases.')
        pair = (order.before, order.after)
        if pair in pairs:
            raise ValueError('Duplicate phase orders are not supported.')
        pairs.add(pair)
        incoming[order.after].append(order)
        for label, p in zip(pair, order.anchors):
            if not np.isfinite(phases[label]._offset(p)):
                raise ValueError('Order anchors must be finite; normal anchors require 0 < p < 1.')
    for label, edges in incoming.items():
        if len({edge.anchors[1] for edge in edges}) > 1:
            raise ValueError('Incoming orders must use the same receiving anchor.')
        if phases[label].delta_scale is None and len({edge.delta_scale for edge in edges if edge.delta_scale is not None}) > 1:
            raise ValueError('Incoming legacy delta scales must agree; set Phase.delta_scale explicitly.')
    roots = [label for label, edges in incoming.items() if not edges]
    sequence, resolved = [], set(roots)
    pending = [label for label in phases if label not in resolved]
    while pending:
        ready = [label for label in pending if all(edge.before in resolved for edge in incoming[label])]
        if not ready:
            raise ValueError('Phase orders must not contain cycles.')
        for label in ready:
            sequence.append(label); resolved.add(label); pending.remove(label)
    return incoming, roots, sequence


def build_phase(data, phases, *, measurements, orders=()):
    """Build a joint PyMC hierarchy without sampling.

    data contains labelled records or a DataFrame. phases maps each label to a
    Phase. measurements is an aligned sequence of existing calrcarbon or frozen
    scipy normal/uniform objects, preserving each event's curve and uncertainty.
    Conversion of app rows to these existing objects stays with the caller.

    mu and scale have a labelled ``phase`` dimension; scale is full width for
    uniform phases and sigma for normal phases. tau has an ``event`` dimension
    in original row order. Each tau_i has its group's event distribution and
    the existing measurement log likelihood. tau_0, tau_1, ... are internal
    group variables, indexed by first label occurrence. Optional orders is a
    sequence of Order relationships forming a DAG. Each non-root phase owns
    one positive delta from the exact youngest selected predecessor anchor.
    delta has a labelled ``input_phase`` dimension. Input row and relationship
    order need not be chronological. Root locations remain free; downstream locations
    are derived by quantile algebra, without inequality potentials.
    """
    import pymc as pm
    import pytensor.tensor as pt

    records = data.to_dict("records") if hasattr(data, "columns") else list(data)
    groups = group_phases(records)
    events = list(measurements)
    if not groups or len(events) != len(records):
        raise ValueError("Supply nonempty labelled data and one measurement per row.")
    if not all(isinstance(s, Phase) for s in phases.values()):
        raise ValueError("Supply exactly one Phase specification per event label.")
    labels = list(groups) + [label for label in phases if label not in groups]
    orders = list(orders)
    incoming, roots, sequence = _orders(phases, orders)
    if set(phases) != set(groups):
        raise ValueError('Every phase currently requires labelled measurements; unobserved phase inference is deferred.')
    receivers = [label for label in labels if incoming[label]]
    indices = {label: [i for i, row in enumerate(records) if row["label"] == label]
               for label in labels}
    bridges = [_measurement(event) for event in events]
    priors = [_priors([bridges[i] for i in indices[label]],
                      phases[label].prior_center, phases[label].prior_scale)
              for label in labels]
    centers, spreads = np.asarray(priors).T
    reference = MIXTURE_SCALE_FRACTION * spreads * np.array(
        [np.sqrt(12) if phases[label].distribution == "uniform" else 1 for label in labels])
    initial = reference.copy()
    for j, label in enumerate(labels):
        if phases[label].distribution == "uniform":
            initial[j] = max(initial[j], 2 * max(abs(bridges[i][1] - centers[j])
                                               + bridges[i][2] for i in indices[label]))
    coords = {"phase": labels, "event": np.arange(len(records))}
    if orders:
        coords.update(root=roots, input_phase=receivers)
    mu_initial = centers.copy()
    with pm.Model(coords=coords) as model:
        if not orders:
            mu = pm.Normal("mu", mu=centers, sigma=spreads, dims="phase", initval=centers)
        scale = pm.LogNormal("scale", mu=np.log(reference), sigma=MIXTURE_LOG_SCALE_SD,
                             dims="phase", initval=initial)
        if orders:
            positions = {label: j for j, label in enumerate(labels)}
            root_indices = [positions[label] for label in roots]
            root_mu = pm.Normal("mu_root", mu=centers[root_indices], sigma=spreads[root_indices],
                                dims="root", initval=centers[root_indices])
            delta_scales = np.array([
                phases[label].delta_scale if phases[label].delta_scale is not None else
                next((edge.delta_scale for edge in incoming[label] if edge.delta_scale is not None),
                     max(spreads[positions[name]] for name in [label, *[edge.before for edge in incoming[label]]]))
                for label in receivers])
            delta_initial = delta_scales.copy()
            delta_indices = {label: k for k, label in enumerate(receivers)}
            for label in sequence:
                b, k = positions[label], delta_indices[label]
                offset_b = phases[label]._offset(incoming[label][0].anchors[1])
                reference_anchor = max(mu_initial[positions[edge.before]] + initial[positions[edge.before]]
                                       * phases[edge.before]._offset(edge.anchors[0]) for edge in incoming[label])
                separation = centers[b] + initial[b] * offset_b - reference_anchor
                # Start at measured centers when they satisfy the chosen order.
                # This changes initialization only, never the delta prior.
                if separation > 0:
                    delta_initial[k] = separation
                mu_initial[b] = reference_anchor + delta_initial[k] - initial[b] * offset_b
            delta = pm.HalfNormal("delta", sigma=delta_scales, dims="input_phase", initval=delta_initial)
            locations = {label: root_mu[j] for j, label in enumerate(roots)}
            for label in sequence:
                b, k = positions[label], delta_indices[label]
                anchors = pt.stack([locations[edge.before] + scale[positions[edge.before]]
                                    * phases[edge.before]._offset(edge.anchors[0]) for edge in incoming[label]])
                locations[label] = (pt.max(anchors) + delta[k]
                                    - scale[b] * phases[label]._offset(incoming[label][0].anchors[1]))
            mu = pm.Deterministic("mu", pt.stack([locations[label] for label in labels]), dims="phase")
        latent = [None] * len(records)
        for j, label in enumerate(labels):
            rows = indices[label]
            kwargs = dict(shape=len(rows), initval=[bridges[i][1] for i in rows])
            if phases[label].distribution == "uniform":
                # Derived locations can move the interval away from the data;
                # initialize latent dates strictly inside its actual support.
                if orders:
                    kwargs["initval"] = np.clip(kwargs["initval"],
                                               mu_initial[j] - .49 * initial[j],
                                               mu_initial[j] + .49 * initial[j])
                times = pm.Uniform(f"tau_{j}", lower=mu[j] - scale[j] / 2,
                                   upper=mu[j] + scale[j] / 2, **kwargs)
            else:
                times = pm.Normal(f"tau_{j}", mu=mu[j], sigma=scale[j], **kwargs)
            for k, i in enumerate(rows):
                latent[i] = times[k]
        tau = pm.Deterministic("tau", pt.stack(latent), dims="event")
        pm.Potential("measurements", pt.sum(pt.stack(
            [bridge[0](tau[i]) for i, bridge in enumerate(bridges)])))
    return model


def fit_phase(data, phases, *, measurements, orders=(), draws=250, tune=250, chains=2,
              random_seed=912, cores=1, progress_callback=None):
    """Fit build_phase with the existing density models' PyMC NUTS settings.

    Return PyMC's posterior DataTree. Query a label's distribution per draw with
    spec.quantile(p, posterior.mu.sel(phase=label), posterior.scale.sel(phase=label)),
    where posterior = result['posterior'].to_dataset(). Short default runs are
    execution defaults, not a convergence guarantee.
    """
    import pymc as pm

    if type(cores) is not int or cores < 1:
        raise ValueError("cores must be a positive integer.")
    total = chains * (draws + tune)
    counts = [0] * chains
    def report(stage, completed=0):
        if progress_callback:
            progress_callback(dict(stage=stage, completed=completed, total=total))
    def on_draw(trace, draw):
        counts[draw.chain] = draw.draw_idx + 1
        report(f"{'Tuning' if draw.tuning else 'Sampling'} · chain {draw.chain + 1}/{chains}", sum(counts))
    report("Building phase model")
    model = build_phase(data, phases, measurements=measurements, orders=orders)
    with model:
        report("Compiling and initializing")
        result = pm.sample(draws=draws, tune=tune, chains=chains, cores=min(cores, chains),
                         blas_cores="auto", random_seed=random_seed, nuts_sampler="pymc",
                         init="adapt_diag", target_accept=.95, progressbar=False,
                         compute_convergence_checks=False,
                         callback=on_draw if progress_callback else None)
    report("Summarizing phases", total)
    return result
