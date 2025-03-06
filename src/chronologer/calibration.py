import pandas as pd
import numpy as np
from .distributions import calrcarbon

def hdi(t_values, 
        pdf_values, 
        hdi_prob=0.95):
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

    Returns
    -------
    hdi_intervals : list of tuples
        List of (start, end) intervals covering the HDI.
    """
    
    # Sort by descending density (highest first)
    idx = np.argsort(-pdf_values)
    sorted_pdf = pdf_values[idx]
    sorted_t = t_values[idx]

    # Cumulative mass until desired probability reached
    cumulative_mass = np.cumsum(sorted_pdf) * (t_values[1] - t_values[0])
    within_hdi = cumulative_mass <= hdi_prob

    # Extract HDI ages and sort back into time order
    hdi_ages = np.sort(sorted_t[within_hdi])

    # Find contiguous runs (this handles multimodal intervals)
    gaps = np.where(np.diff(hdi_ages) > (t_values[1] - t_values[0]))[0]
    intervals = []

    start = hdi_ages[0]
    for gap in gaps:
        end = hdi_ages[gap]
        intervals.append((start, end))
        start = hdi_ages[gap + 1]

    intervals.append((start, hdi_ages[-1]))
    return intervals

def calibrate(radiocarbon_ages, 
              radiocarbon_errors, 
              calcurve, 
              hdi_prob=0.95,
              tol = 1e-7, 
              as_pandas=True):
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
        pdf_values = cal.pdf(t_values)
        
        # Trim to just the part where the density is meaningful
        mask = pdf_values > tol
        t_values = t_values[mask]
        pdf_values = pdf_values[mask]

        # Compute mean & std (this part's fine)
        mean_age = np.sum(t_values * pdf_values) * (t_values[1] - t_values[0])
        variance_age = np.sum(((t_values - mean_age)**2) * pdf_values) * (t_values[1] - t_values[0])
        std_age = np.sqrt(variance_age)

        # Compute proper HDI (potentially discontinuous)
        hdi_intervals = hdi(t_values, pdf_values, hdi_prob=hdi_prob)

        # Store results
        results.append({
            "radiocarbon_age": age,
            "mean": mean_age,
            "std": std_age,
            "hdi_intervals": hdi_intervals,
            "calibrated_distribution": cal,
            "t_values": t_values,
            "pdf_values": pdf_values,
        })

    if as_pandas:
        df = pd.DataFrame({
            "Radiocarbon Age": [r["radiocarbon_age"] for r in results],
            "Mean Calibrated Age (BP)": [r["mean"] for r in results],
            "Std Dev (BP)": [r["std"] for r in results],
            "HDI Intervals": [r["hdi_intervals"] for r in results],
            "Calibrated Distribution": [r["calibrated_distribution"] for r in results],
            "CalBP Domain": [r["t_values"] for r in results],
            "Calibrated PDF": [r["pdf_values"] for r in results],
        })
        return df


    return results