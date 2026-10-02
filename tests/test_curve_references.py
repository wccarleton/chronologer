import numpy as np
import pytest
import pymc as pm
import pytensor
import pytensor.tensor as pt
from chronologer.distributions import calrcarbon, _curve_splines
from chronologer.density import _measurement, build_gaussian_mixture


def curve(offset=0):
    return dict(calbp=np.linspace(-100, 0, 11),
                c14bp=np.linspace(-100, 0, 11) + offset,
                c14_sigma=np.full(11, 2.))


def test_reuse_distinct_curves_and_input_mutation():
    data = curve()
    a = calrcarbon(data, -50, 3)
    same = calrcarbon(curve(), -45, 4)
    other = calrcarbon(curve(20), -50, 3)
    assert a._interp_mean is same._interp_mean
    assert a._interp_error is same._interp_error
    assert a._interp_mean is not other._interp_mean
    assert a._calc_curve_params(-50)[0] == pytest.approx(-50)
    assert other._calc_curve_params(-50)[0] == pytest.approx(-30)
    original = a.pdf(-50)
    data['c14bp'] += 40
    changed = calrcarbon(data, -50, 3)
    assert changed._interp_mean is not a._interp_mean
    assert a.pdf(-50) == original
    _curve_splines.cache_clear()
    assert a.pdf(-50) == original
    assert a._interp_mean is same._interp_mean


def test_curve_uncertainty_is_part_of_identity():
    data = curve()
    a = calrcarbon(data, -50, 3)
    data['c14_sigma'] *= 2
    b = calrcarbon(data, -50, 3)
    assert a._interp_error is not b._interp_error
    assert a.pdf(-50) != b.pdf(-50)


def test_mixed_curve_symbolic_likelihood_and_gradient():
    a, b = calrcarbon(curve(), -50, 3), calrcarbon(curve(20), -50, 3)
    t = pt.dscalar('t')
    for event in (a, b):
        logp = _measurement(event)[0](t)
        evaluate = pytensor.function([t], [logp, pt.grad(logp, t)])
        value, gradient = evaluate(-48.)
        assert value == pytest.approx(event.logpdf(-48.))
        h = .0001
        assert gradient == pytest.approx((event.logpdf(-48+h)-event.logpdf(-48-h))/(2*h), abs=1e-6)
    model = build_gaussian_mixture([a, b], 2)
    assert np.isfinite(model.compile_logp()(model.initial_point()))
    assert np.isfinite(model.compile_dlogp()(model.initial_point())).all()
    with model:
        trace = pm.sample(draws=4, tune=4, chains=1, cores=1, init='adapt_diag',
                          random_seed=42, progressbar=False, compute_convergence_checks=False)
    assert np.isfinite(trace['posterior']['tau'].values).all()
    # Building another curve later cannot change the already-compiled model.
    evaluate = model.compile_logp()
    before = evaluate(model.initial_point())
    calrcarbon(curve(40), -50, 3)
    assert evaluate(model.initial_point()) == before
