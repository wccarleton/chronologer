import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm, uniform

from chronologer import Phase, Order, build_phase, group_phases
from chronologer.distributions import calrcarbon
from test_curve_references import curve


def test_uniform_construction():
    spec = Phase("uniform", prior_center=-50, prior_scale=20)
    model = build_phase([{"label": "A"}, {"label": "A"}], {"A": spec},
                        measurements=[calrcarbon(curve(), -50, 3), norm(-40, 2)])
    assert model.coords["phase"] == ("A",)
    assert model["tau"].type.shape == (2,)
    # Python graph mode avoids native compilation and sampling for this check.
    assert np.isfinite(model.compile_logp(mode="FAST_COMPILE")(model.initial_point()))


def test_normal_construction():
    model = build_phase([{"label": "A"}, {"label": "B"}, {"label": "A"}],
                        {"A": Phase("gaussian"), "B": Phase("normal")},
                        measurements=[norm(-50, 3), norm(-30, 2), norm(-40, 2)])
    assert model.coords["phase"] == ("A", "B")
    assert model["tau_0"].type.shape == (2,)
    assert model["tau_1"].type.shape == (1,)
    assert np.isfinite(model.compile_logp(mode="FAST_COMPILE")(model.initial_point()))


def test_group_labels():
    rows = [{"label": "older", "id": 1}, {"label": "younger", "id": 2},
            {"label": "older", "id": 3}]
    groups = group_phases(rows)
    assert list(groups) == ["older", "younger"]
    assert groups["older"] == [rows[0], rows[2]]
    assert groups["older"][0] is rows[0]
    assert group_phases(pd.DataFrame(rows)) == groups


def test_queries():
    uniform = Phase("uniform")
    assert uniform.interval(0, 1, -50, 20) == (-60, -40)
    assert uniform.quantile(.25, -50, 20) == -55
    normal = Phase("normal")
    np.testing.assert_allclose(normal.interval(.05, .95, -50, 2), norm(-50, 2).ppf([.05, .95]))
    np.testing.assert_allclose(normal.quantile(.5, np.array([-50, -40]), np.array([2, 3])), [-50, -40])
    assert normal.quantile(0, -50, 2) == -np.inf
    with pytest.raises(ValueError):
        uniform.quantile(1.1, -50, 20)
    with pytest.raises(ValueError):
        normal.interval(.95, .05, -50, 2)


def check_order(phases, orders):
    # Deliberately place younger labels first to check identity is label-based.
    labels = list(reversed(phases))
    rows = [{"label": label} for label in labels]
    model = build_phase(rows, phases, orders=orders,
                        measurements=[norm(-70 + 30 * list(phases).index(label), 2)
                                      for label in labels])
    values = model.replace_rvs_by_values([model["mu"], model["scale"], model["delta"]])
    evaluate = model.compile_fn(values, inputs=model.value_vars, mode="FAST_COMPILE",
                                on_unused_input="ignore")
    point = model.initial_point()
    assert "mu" not in {rv.name for rv in model.free_RVs}
    assert [rv.name for rv in model.potentials] == ["measurements"]
    assert np.isfinite(model.compile_logp(mode="FAST_COMPILE")(point))
    # Two fixed parameter points exercise the actual model algebra, no sampling.
    for multiplier in (1., 2.):
        point["scale_log__"] = point["scale_log__"] + np.log(multiplier)
        point["delta_log__"] = np.log(np.arange(1, len(model.coords['input_phase']) + 1) * 7 * multiplier)
        mu, scale, delta = evaluate(point)
        assert (delta > 0).all()
        for k, order in enumerate(orders):
            a, b = labels.index(order.before), labels.index(order.after)
            p, q = order.anchors
            upstream = phases[order.before].quantile(p, mu[a], scale[a])
            downstream = phases[order.after].quantile(q, mu[b], scale[b])
            assert downstream - upstream == pytest.approx(delta[model.coords['input_phase'].index(order.after)])


def test_center_order_chain():
    phases = {label: Phase("normal", prior_center=-70 + 30 * i, prior_scale=10)
              for i, label in enumerate(("A", "B", "C"))}
    check_order(phases, [Order("B", "C", delta_scale=20), Order("A", "B")])


def test_uniform_endpoint_order():
    phases = {label: Phase("uniform", prior_center=-70 + 30 * i, prior_scale=10)
              for i, label in enumerate(("A", "B"))}
    check_order(phases, [Order("A", "B", anchors=(1, 0), delta_scale=15)])


def test_normal_quantile_order():
    phases = {label: Phase("normal", prior_center=-70 + 30 * i, prior_scale=10)
              for i, label in enumerate(("A", "B"))}
    check_order(phases, [Order("A", "B", anchors=(.95, .05), delta_scale=15)])


def test_invalid_order():
    phases = {"A": Phase(), "B": Phase()}
    rows = [{"label": label} for label in phases]
    measurements = [norm(-50, 2), norm(-30, 2)]
    with pytest.raises(ValueError, match="cycles"):
        build_phase(rows, phases, measurements=measurements,
                    orders=[Order("A", "B"), Order("B", "A")])
    with pytest.raises(ValueError, match="finite"):
        build_phase(rows, phases, measurements=measurements,
                    orders=[Order("A", "B", anchors=(1, 0))])


def test_order_initialization_with_bounded_measurements():
    model = build_phase([{'label': 'A'}, {'label': 'B'}],
                        {'A': Phase('uniform'), 'B': Phase('uniform')},
                        measurements=[uniform(-2510, 20), uniform(-2210, 20)],
                        orders=[Order('A', 'B')])
    assert np.isfinite(model.compile_logp(mode='FAST_COMPILE')(model.initial_point()))
