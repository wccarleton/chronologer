from .pymccarbon import interpolate_calcurve

def simulate_c14(tau, 
                 calbp, 
                 c14bp, 
                 c14_sigma):
    """
    Simulate radiocarbon measurements for given calendar dates based on the calibration curve.
    
    Args:
    - tau: array-like, calendar ages (BP) to back-calibrate.
    - calbp: array-like, calendar years (BP) from the calibration curve.
    - c14bp: array-like, radiocarbon years from the calibration curve.
    - c14_sigma: array-like, radiocarbon year uncertainties from the calibration curve.
    
    Returns:
    - simulated_radiocarbon: array of sampled radiocarbon ages for the given calendar dates.
    """
    # Ensure calendar_dates is an array for vectorization
    tau = np.atleast_1d(tau)
    
    # Interpolate the calibration curve for each calendar date
    simulated_radiocarbon = []
    mean, sigma = interpolate_calcurve(tau, calbp, c14bp, c14_sigma, pyt = False)
    # if mean and sigma come in as tensors, evaluate them for this
    mean_eval = mean.eval() if hasattr(mean, "eval") else mean
    sigma_eval = sigma.eval() if hasattr(sigma, "eval") else sigma
    radiocarbon_sample = np.random.normal(loc=mean_eval, scale=sigma_eval, size=len(mean_eval))
    
    return np.array(radiocarbon_sample)