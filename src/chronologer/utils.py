from __future__ import annotations

from typing import Iterable

import numpy as np

def simulate_c14(
    tau: Iterable[float],
    calbp: np.ndarray,
    c14bp: np.ndarray,
    c14_sigma: np.ndarray,
) -> np.ndarray:
    """
    Simulate radiocarbon measurements for given calendar dates based on the calibration curve.

    Args:
    - tau: array-like calendar ages, in the same signed coordinates as calbp.
      load_calcurve returns increasing negative BP coordinates.
    - calbp: array-like, calendar years (BP) from the calibration curve.
    - c14bp: array-like, radiocarbon years from the calibration curve.
    - c14_sigma: array-like, radiocarbon year uncertainties from the calibration curve.

    Returns:
    - simulated_radiocarbon: array of sampled radiocarbon ages for the given calendar dates.
    """
    # This numerical helper must not index NumPy arrays with symbolic tensors.
    tau = np.atleast_1d(np.asarray(tau, dtype=float))
    calbp, c14bp, c14_sigma = [np.asarray(a, dtype=float) for a in (calbp, c14bp, c14_sigma)]
    if (tau.ndim != 1 or calbp.ndim != 1 or len(calbp) < 2
            or c14bp.shape != calbp.shape or c14_sigma.shape != calbp.shape
            or not all(np.isfinite(a).all() for a in (tau, calbp, c14bp, c14_sigma))
            or np.any(np.diff(calbp) <= 0) or np.any(c14_sigma < 0)):
        raise ValueError("Supply finite vectors, increasing curve times, and nonnegative curve errors.")
    if np.any((tau < calbp[0]) | (tau > calbp[-1])):
        raise ValueError("Calendar dates must be within the calibration curve's support.")
    mean_eval = np.interp(tau, calbp, c14bp)
    sigma_eval = np.interp(tau, calbp, c14_sigma)
    radiocarbon_sample = np.random.normal(
        loc=mean_eval, scale=sigma_eval, size=len(mean_eval)
    )

    return np.array(radiocarbon_sample)
