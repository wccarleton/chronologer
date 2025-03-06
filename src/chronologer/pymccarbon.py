import pytensor.tensor as pt

def compute_bin_index(tau, calbp, pyt=True):
    """
    Compute bin indices where each tau falls between calbp[i] and calbp[i+1].

    Works for scalar tau (single date) and vector tau (many dates).

    Parameters
    ----------
    tau : scalar or 1D array-like (pytensor variable)
        Calendar age(s) to locate within calibration bins.
    calbp : 1D array-like (pytensor variable)
        Calibration curve time points (assumed strictly ordered).

    pyt : bool, default=True
        If True, return PyTensor object.
        If False, evaluate and return numpy (for testing/debugging outside PyMC model).

    Returns
    -------
    bin_index : scalar or 1D array-like
        Bin index for each tau.
    """
    # Ensure tau is at least 1D (if scalar, becomes shape (1,))
    if tau.ndim == 0:
        tau = tau[None]

    # Shape broadcasting works correctly for both scalar and vector tau now
    logical_result = (tau[:, None] - calbp[None, :]) >= 0
    numeric_result = pt.cast(logical_result, 'int32')
    bin_index = pt.sum(numeric_result, axis=1) - 1

    # If pyt=False, evaluate result for standalone use
    if not pyt:
        return bin_index.eval()

    # If originally scalar, return scalar index, else return array
    return bin_index[0] if bin_index.shape[0] == 1 else bin_index

def interpolate_calcurve(tau, calbp, c14bp, c14_sigma, pyt=True):
    """
    Linearly interpolate the calibration curve at given tau(s).
    
    Works for scalar tau (single date) and vector tau (many dates).

    Parameters
    ----------
    tau : scalar or 1D array-like (pytensor variable)
        Calendar age(s) to interpolate.
    calbp, c14bp, c14_sigma : 1D array-like (pytensor variable)
        Calibration curve points (calendar age, radiocarbon age, error).
    
    pyt : bool, default=True
        If True, return PyTensor objects.
        If False, evaluate and return numpy (for standalone testing).

    Returns
    -------
    mean_interpolated, sigma_interpolated : scalar or array-like
        Interpolated radiocarbon mean and sigma for each tau.
    """
    # Compute bin index (this now works for scalar or vector tau)
    bin_idx = compute_bin_index(tau, calbp, pyt=pyt)

    # Right-hand bin edges (clip to avoid out-of-bounds)
    bin_idx_right = pt.minimum(bin_idx + 1, calbp.shape[0] - 1)

    # Gather surrounding points
    calbp_i = calbp[bin_idx]
    calbp_ip1 = calbp[bin_idx_right]

    c14bp_i = c14bp[bin_idx]
    c14bp_ip1 = c14bp[bin_idx_right]

    sigma_i = c14_sigma[bin_idx]
    sigma_ip1 = c14_sigma[bin_idx_right]

    # Linear interpolation (vector-safe)
    slope_mean = (c14bp_ip1 - c14bp_i) / (calbp_ip1 - calbp_i)
    slope_sigma = (sigma_ip1 - sigma_i) / (calbp_ip1 - calbp_i)

    mean_interpolated = c14bp_i + slope_mean * (tau - calbp_i)
    sigma_interpolated = sigma_i + slope_sigma * (tau - calbp_i)

    if not pyt:
        return mean_interpolated.eval(), sigma_interpolated.eval()

    return mean_interpolated, sigma_interpolated
