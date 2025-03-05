import pytensor.tensor as pt

def compute_bin_index(tau, calbp, pyt = True):
    """
    Vectorized computation of bin indices where tau falls between calbp[i] and calbp[i+1].
    Assumes that tau draws are strictly inside (min(calbp), max(calbp)) and that
    the calbp vector (calibration curve) is strictly increasing in time with
    smaller or more negative values into the past and larger or more positive
    values toward the present.

    Parameters:
    - tau: Tensor, array-like or scalar. Values to be located within the bins.
    - calbp: Tensor, array-like. Calibration curve time points.
    - pyt: Logical, if True (defult) return a tensor object; otherwise return an integer.

    Returns:
    - bin_index: Tensor. The bin indices for each tau.
    """
    # Logical comparison to find where each tau is within the bins
    logical_result = (tau[:, None] - calbp[None, :]) >= 0  # Shape broadcasting to compare each tau with calbp
    # Explicitly cast boolean to integers (0 or 1)
    numeric_result = pt.cast(logical_result, 'int32')
    # Sum the number of true conditions along the calibration curve axis and subtract 1
    bin_index = pt.sum(numeric_result, axis=1) - 1

    # check whether to return tensor or int
    if(pyt):
        return bin_index
    else: 
        return bin_index.eval()

def interpolate_calcurve(tau, 
                        calbp, 
                        c14bp, 
                        c14_sigma,
                        pyt = True):
    """
    Linear interpolation for calibration curve using PyTensor operations.
    """
    # Compute bin index where tau falls
    bin_idx = compute_bin_index(tau, calbp, pyt = pyt)

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
