from scipy.stats import norm
import numpy as np
import pandas as pd

def calibrate(radiocarbon_ages, 
                radiocarbon_errors, 
                calbp, 
                c14bp, 
                c14_sigma, 
                hdi_prob = 0.95,
                as_pandas = True):
    """
    Calibrates one or more radiocarbon ages and returns the calibrated date density, 
    along with summary statistics such as the mean, standard deviation, and HDIs.
    
    Args:
    - radiocarbon_ages: array-like, radiocarbon ages to calibrate.
    - radiocarbon_errors: array-like, errors associated with the radiocarbon ages.
    - calbp: array-like, calendar years (BP) from the calibration curve.
    - c14bp: array-like, radiocarbon years from the calibration curve.
    - c14_sigma: array-like, radiocarbon year uncertainties from the calibration curve.
    - hdi_prob: float, probability for the HDI (default is 0.95).
    - as_pandas: logical, return a pandas dataframe summary instead of full densities?
    
    Returns:
    - results: a dictionary with calibrated densities and summary statistics for each input date, or pandas dataframe summary.
    """
    
    results = []

    for age, error in zip(radiocarbon_ages, radiocarbon_errors):
        # Calculate likelihood for each calendar year based on the calibration curve
        likelihoods = norm.pdf(age, loc=c14bp, scale=np.sqrt(c14_sigma**2 + error**2))
        
        # Normalize the likelihood to get a proper probability density function
        likelihoods /= np.sum(likelihoods)

        # Compute summary statistics
        mean_age = np.sum(calbp * likelihoods)
        variance_age = np.sum(((calbp - mean_age)**2) * likelihoods)
        std_age = np.sqrt(variance_age)

        # Compute the cumulative density
        cumulative_density = np.cumsum(likelihoods)

        # Find HDI
        lower_idx = np.searchsorted(cumulative_density, (1 - hdi_prob) / 2)
        upper_idx = np.searchsorted(cumulative_density, 1 - (1 - hdi_prob) / 2)
        hdi_lower = calbp[lower_idx]
        hdi_upper = calbp[upper_idx]

        # Store results
        results.append({
            "radiocarbon_age": age,
            "mean": mean_age,
            "std": std_age,
            "hdi_lower": hdi_lower,
            "hdi_upper": hdi_upper,
            "calibrated_density": likelihoods,
            "calbp_domain": calbp
        })
    
    if as_pandas:
        # Prepare data for the DataFrame
        data = {
            "Radiocarbon Age": [],
            "Mean Calibrated Age (BP)": [],
            "Std Dev (BP)": [],
            "HDI Lower (BP)": [],
            "HDI Upper (BP)": []
        }
        
        for result in results:
            data["Radiocarbon Age"].append(result["radiocarbon_age"])
            data["Mean Calibrated Age (BP)"].append(result["mean"])
            data["Std Dev (BP)"].append(result["std"])
            data["HDI Lower (BP)"].append(result["hdi_lower"])
            data["HDI Upper (BP)"].append(result["hdi_upper"])
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        return df
    else:
        return results

def compute_bin_index(tau, calbp):
    """
    Vectorized computation of bin indices where tau falls between calbp[i] and calbp[i+1].
    Assumes that tau draws are strictly inside (min(calbp), max(calbp)) and that
    the calbp vector (calibration curve) is strictly increasing in time with
    smaller or more negative values into the past and larger or more positive
    values toward the present.

    Parameters:
    - tau: Tensor, array-like or scalar. Values to be located within the bins.
    - calbp: Tensor, array-like. Calibration curve time points.

    Returns:
    - bin_index: Tensor. The bin indices for each tau.
    """
    # Logical comparison to find where each tau is within the bins
    logical_result = (tau[:, None] - calbp[None, :]) >= 0  # Shape broadcasting to compare each tau with calbp
    # Explicitly cast boolean to integers (0 or 1)
    numeric_result = pt.cast(logical_result, 'int32')
    # Sum the number of true conditions along the calibration curve axis and subtract 1
    bin_index = pt.sum(numeric_result, axis=1) - 1
    return bin_index

def interpolate_calcurve(tau, 
                        calbp, 
                        c14bp, 
                        c14_sigma):
    """
    Linear interpolation for calibration curve using PyTensor operations.
    """
    # Compute bin index where tau falls
    bin_idx = compute_bin_index(tau, calbp)

    # Compute right bin edge (bin_idx + 1)
    bin_idx_right = bin_idx + 1

    # Get the corresponding values for the surrounding bins
    calbp_i = calbp[bin_idx]
    calbp_ip1 = calbp[bin_idx_right]
    
    c14bp_i = c14bp[bin_idx]
    c14bp_ip1 = c14bp[bin_idx_right]
    
    sigma_i = c14_sigma[bin_idx]
    sigma_ip1 = c14_sigma[bin_idx_right]

    # Compute slope and interpolate
    slope_mean = (c14bp_ip1 - c14bp_i) / (calbp_ip1 - calbp_i)
    slope_sigma = (sigma_ip1 - sigma_i) / (calbp_ip1 - calbp_i)

    mean_interpolated = c14bp_i + slope_mean * (tau - calbp_i)
    sigma_interpolated = sigma_i + slope_sigma * (tau - calbp_i)
    
    return mean_interpolated, sigma_interpolated
