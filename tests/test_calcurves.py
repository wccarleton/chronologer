import os
import pandas as pd
import numpy as np
import pytest

from chronologer.calcurves import load_calcurve


# Helper function to create temp calibration curve CSV

def _create_curve_file(tmp_path, calbp, c14bp, sigma=None):
    if sigma is None:
        sigma = np.ones_like(calbp)
    df = pd.DataFrame({'calbp': calbp, 'c14bp': c14bp, 'c14_sigma': sigma})
    file_path = tmp_path / 'curve.csv'
    df.to_csv(file_path, index=False)
    return file_path


def test_load_custom_curve_descending(tmp_path):
    calbp = [100, 90, 80]
    c14bp = [110, 100, 90]
    path = _create_curve_file(tmp_path, calbp, c14bp)
    curve = load_calcurve(custom_path=str(path), quiet=True)
    assert np.all(curve['calbp'] == np.array([-100, -90, -80]))
    assert np.all(curve['c14bp'] == np.array([-110, -100, -90]))


def test_load_custom_curve_ascending_negative(tmp_path):
    calbp = [-100, -90, -80]
    c14bp = [-110, -100, -90]
    path = _create_curve_file(tmp_path, calbp, c14bp)
    curve = load_calcurve(custom_path=str(path), quiet=True)
    assert np.all(curve['calbp'] == np.array(calbp))
    assert np.all(curve['c14bp'] == np.array(c14bp))


def test_load_builtin_curve():
    curve = load_calcurve('intcal20', quiet=True)
    for key in ['calbp', 'c14bp', 'c14_sigma']:
        assert key in curve
        assert isinstance(curve[key], np.ndarray)
    assert len(curve['calbp']) > 0


def test_custom_curve_missing_columns(tmp_path):
    df = pd.DataFrame({'calbp': [1, 2], 'c14bp': [3, 4]})
    path = tmp_path / 'bad.csv'
    df.to_csv(path, index=False)
    with pytest.raises(ValueError):
        load_calcurve(custom_path=str(path), quiet=True)


def test_unknown_curve_name():
    with pytest.raises(ValueError):
        load_calcurve('unknown_curve', quiet=True)
