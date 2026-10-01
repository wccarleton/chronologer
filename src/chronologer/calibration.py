from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from .distributions import calrcarbon


def hdi(
    t_values: np.ndarray,
    pdf_values: np.ndarray,
    hdi_prob: float = 0.95,
    *,
    grid_spacing: float | None = None,
) -> List[Tuple[float, float]]:
    """
    Computes highest density interval (HDI) from a calibrated PDF.

    Parameters
    ----------
    t_values : np.ndarray
        Array of calendar ages (time domain).
    pdf_values : np.ndarray
        Array of calibrated densities.
    hdi_prob : float, optional
        Desired HDI probability mass (default = 0.95).
    grid_spacing : float, optional
        Original uniform grid spacing before trimming. If omitted, the smallest
        retained spacing is used. Values must be on a uniform grid, possibly
        with missing cells; weights need not be normalized.

    Returns
    -------
    hdi_intervals : list of tuples
        Inclusive grid-node endpoints for the highest-density cells containing
        at least the requested fraction of retained weight. All cells tied at
        the cutoff are included, so discrete coverage can exceed hdi_prob.
    """

    t_values = np.asarray(t_values, dtype=float)
    pdf_values = np.asarray(pdf_values, dtype=float)
    if (t_values.ndim != 1 or pdf_values.shape != t_values.shape
            or not len(t_values) or not np.all(np.isfinite(t_values))
            or not np.all(np.isfinite(pdf_values)) or np.any(pdf_values < 0)
            or not np.any(pdf_values > 0) or not 0 < hdi_prob <= 1):
        raise ValueError("HDI requires finite coordinates, nonnegative weights, and 0 < probability <= 1")
    differences = np.diff(t_values)
    if np.any(differences <= 0):
        raise ValueError("HDI coordinates must be strictly increasing")
    step = grid_spacing if grid_spacing is not None else (np.min(differences) if len(differences) else 1.0)
    if not np.isfinite(step) or step <= 0:
        raise ValueError("Grid spacing must be finite and positive")
    if not np.allclose(differences / step, np.round(differences / step), rtol=1e-7, atol=1e-7):
        raise ValueError("HDI requires a uniform grid, optionally with missing cells")

    # Equal-width cells: spacing cancels. Scale locally to avoid overflow;
    # the original density array is never modified.
    weights = pdf_values / np.max(pdf_values)
    sorted_weights = np.sort(weights[weights > 0])[::-1]
    cumulative = np.cumsum(sorted_weights)
    cutoff_index = min(np.searchsorted(cumulative, hdi_prob * cumulative[-1]), len(sorted_weights) - 1)
    selected = (weights > 0) & (weights >= sorted_weights[cutoff_index])
    ages = t_values[selected]
    # Original spacing distinguishes real missing cells from rounding noise.
    groups = np.split(ages, np.flatnonzero(np.diff(ages) > step * (1 + 1e-7)) + 1)
    return [(float(group[0]), float(group[-1])) for group in groups]


def calibrate(
    radiocarbon_ages: Iterable[float],
    radiocarbon_errors: Iterable[float],
    calcurve: Dict[str, np.ndarray],
    hdi_prob: float = 0.95,
    tol: float = 1e-7,
    as_pandas: bool = True,
) -> List[Dict[str, object]] | pd.DataFrame:
    """
    Calibrates one or more radiocarbon ages using the calrcarbon distribution.

    Args:
    - radiocarbon_ages: array-like, radiocarbon ages to calibrate (negative BP convention).
    - radiocarbon_errors: array-like, errors associated with the radiocarbon ages.
    - calcurve: dict containing 'calbp', 'c14bp', and 'c14_sigma' from calibration curve.
    - hdi_prob: float, probability for the HDI (default is 0.95).
    - as_pandas: logical, return a pandas dataframe summary instead of full densities?

    Returns:
    - DataFrame if as_pandas=True, otherwise list of dicts (one per date).
    """

    results = []

    for age, error in zip(radiocarbon_ages, radiocarbon_errors):
        cal = calrcarbon(calcurve, c14_mean=age, c14_err=error)

        # Sample PDF over fine grid in the curve range
        t_values = np.linspace(cal.a, cal.b, 10000)
        grid_spacing = t_values[1] - t_values[0]
        pdf_values = cal.pdf(t_values)

        # Trim to just the part where the density is meaningful
        mask = pdf_values > tol
        t_values = t_values[mask]
        pdf_values = pdf_values[mask]
        if len(t_values) < 2:
            raise ValueError("Calibration returned fewer than two usable grid points")

        # The uniform grid spacing cancels in the weighted mean. Divide by
        # retained weight without rescaling the returned density values.
        mean_age = np.sum(t_values * pdf_values) / np.sum(pdf_values)
        variance_age = np.sum(((t_values - mean_age) ** 2) * pdf_values) * (
            t_values[1] - t_values[0]
        )
        std_age = np.sqrt(variance_age)

        # Compute proper HDI (potentially discontinuous)
        hdi_intervals = hdi(t_values, pdf_values, hdi_prob=hdi_prob, grid_spacing=grid_spacing)

        # Store results
        results.append(
            {
                "radiocarbon_age": age,
                "mean": mean_age,
                "std": std_age,
                "hdi_intervals": hdi_intervals,
                "calibrated_distribution": cal,
                "t_values": t_values,
                "pdf_values": pdf_values,
            }
        )

    if as_pandas:
        df = pd.DataFrame(
            {
                "Radiocarbon Age": [r["radiocarbon_age"] for r in results],
                "Mean Calibrated Age (BP)": [r["mean"] for r in results],
                "Std Dev (BP)": [r["std"] for r in results],
                "HDI Intervals": [r["hdi_intervals"] for r in results],
                "Calibrated Distribution": [
                    r["calibrated_distribution"] for r in results
                ],
                "CalBP Domain": [r["t_values"] for r in results],
                "Calibrated PDF": [r["pdf_values"] for r in results],
            }
        )
        return df

    return results
