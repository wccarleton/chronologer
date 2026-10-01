import numpy as np
import pytest
from chronologer.utils import simulate_c14


def test_numeric_interpolation_and_curve_noise():
    old_state = np.random.get_state()
    try:
        np.random.seed(17)
        expected = np.random.normal([-90, -45, 0], [2, 3, 4])
        np.random.seed(17)
        actual = simulate_c14([-100, -50, 0], np.array([-100, 0]),
                              np.array([-90, 0]), np.array([2, 4]))
        np.testing.assert_allclose(actual, expected)
    finally:
        np.random.set_state(old_state)


def test_scalar_and_domain_checks():
    assert simulate_c14(-50, [-100, 0], [-90, 0], [0, 0]).tolist() == [-45]
    with pytest.raises(ValueError, match='support'):
        simulate_c14(1, [-100, 0], [-90, 0], [2, 4])
    with pytest.raises(ValueError, match='increasing'):
        simulate_c14(-50, [0, -100], [0, -90], [2, 4])
