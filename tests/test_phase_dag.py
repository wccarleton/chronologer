"""Small DAG algebra checks and one short NUTS execution, not convergence tests."""
import numpy as np
import pymc as pm
import pytest
from scipy.stats import norm
from chronologer import Phase, Order, build_phase


def diamond():
    specs = {label: Phase('normal', prior_center=-100 + 20*i, prior_scale=10,
                          delta_scale=7 if label != 'A' else None)
             for i, label in enumerate('ABCD')}
    edges = [Order('C', 'D', (.95, .05)), Order('A', 'C', (.95, .05)),
             Order('B', 'D', (.95, .05)), Order('A', 'B', (.95, .05))]
    rows = [{'label': label} for label in specs]
    model = build_phase(rows, specs, measurements=[norm(-100 + 20*i, 5) for i in range(4)], orders=edges)
    return model, specs, edges


def test_diamond_uses_one_delta_per_receiver_and_exact_max():
    model, specs, edges = diamond()
    assert set(model.coords['input_phase']) == set('BCD')
    values = model.replace_rvs_by_values([model['mu'], model['scale'], model['delta']])
    evaluate = model.compile_fn(values, inputs=model.value_vars, mode='FAST_COMPILE', on_unused_input='ignore')
    point = model.initial_point()
    for scales in ([2., 3., 8., 4.], [2., 8., 3., 4.]):
        point['scale_log__'] = np.log(scales)
        point['delta_log__'] = np.log([5., 7., 11.])
        mu, scale, delta = evaluate(point)
        for label in 'BCD':
            incoming = [edge for edge in edges if edge.after == label]
            reference = max(specs[e.before].quantile(e.anchors[0], mu['ABCD'.index(e.before)], scale['ABCD'.index(e.before)]) for e in incoming)
            anchor = specs[label].quantile(.05, mu['ABCD'.index(label)], scale['ABCD'.index(label)])
            assert anchor - reference == pytest.approx(delta[model.coords['input_phase'].index(label)])
    assert np.isfinite(model.compile_logp(mode='FAST_COMPILE')(model.initial_point()))


def test_dag_rejects_cycles_and_inconsistent_receiver_anchors():
    model, specs, edges = diamond()
    rows = [{'label': label} for label in specs]
    measurements = [norm(-100 + 20*i, 5) for i in range(4)]
    with pytest.raises(ValueError, match='cycles'):
        build_phase(rows, specs, measurements=measurements, orders=edges + [Order('D','A',(.95,.05))])
    with pytest.raises(ValueError, match='same receiving anchor'):
        build_phase(rows, specs, measurements=measurements, orders=[Order('A','D'),Order('B','D',(.5,.2))])
    with pytest.raises(ValueError, match='Duplicate'):
        build_phase(rows, specs, measurements=measurements, orders=edges + edges[:1])


def test_exact_max_nuts_smoke():
    model, _, _ = diamond()
    with model:
        trace = pm.sample(draws=8, tune=12, chains=1, cores=1, random_seed=123,
                          progressbar=False, compute_convergence_checks=False, init='adapt_diag', target_accept=.95)
    data = trace['posterior'].to_dataset()
    assert data['delta'].shape == (1, 8, 3)
    assert np.isfinite(data['mu']).all() and (data['delta'] > 0).all()
