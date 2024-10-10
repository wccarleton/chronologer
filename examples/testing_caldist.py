import pymc as pm
import pytensor.tensor as pt
import numpy as np
import os
import pandas as pd

def download_intcal20(cache=True):
    cache_file = os.path.join(CACHE_DIR, "intcal20_cache.csv")
    
    # Check if cache exists and return cached data if available
    if cache and os.path.exists(cache_file):
        return pd.read_csv(cache_file).to_dict('list')
    
    # Download the calibration curve data
    url = "https://intcal.org/curves/intcal20.14c"
    intcal20 = pd.read_csv(url, skiprows=10, delimiter=",")
    intcal20.columns = ["calbp", "c14bp", "c14_sigma", "f14c", "f14c_sigma"]
    
    # Cache the data for future use
    if cache:
        intcal20.to_csv(cache_file, index=False)

    return {
        'calbp': intcal20['calbp'].values,
        'c14bp': intcal20['c14bp'].values,
        'c14_sigma': intcal20['c14_sigma'].values
    }

def add_custom_curve(name, calbp, c14bp, c14_sigma):
    calibration_curves[name] = {
        'calbp': calbp,
        'c14bp': c14bp,
        'c14_sigma': c14_sigma
    }


cache_base_dir = os.getcwd()
CACHE_DIR = os.path.join(cache_base_dir, "calcurves")

# Example usage to create initial intcal20 dictionary
intcal20 = download_intcal20()

# Users can add their custom curves to this dictionary
calibration_curves = {
    'intcal20': intcal20
}

def pytensor_linear_interp(x, xp, fp):
    """
    Custom linear interpolation for PyTensor to avoid index errors.
    """
    # Create conditions for each segment of the interpolation
    conditions = [(x >= xp[i]) & (x < xp[i + 1]) for i in range(len(xp) - 1)]

    # Interpolate values for each segment
    interpolated_values = [
        fp[i] + (fp[i + 1] - fp[i]) * (x - xp[i]) / (xp[i + 1] - xp[i]) 
        for i in range(len(xp) - 1)
    ]

    # Use pt.switch to return the interpolated value based on conditions
    result = pt.switch(conditions[0], interpolated_values[0], interpolated_values[-1])
    for cond, interp_val in zip(conditions[1:], interpolated_values[1:]):
        result = pt.switch(cond, interp_val, result)
    
    return result

# Use the IntCal20 calibration curve from calibration_curves dictionary
intcal20 = calibration_curves['intcal20']
calbp = intcal20['calbp']  # Calendar years
c14bp = intcal20['c14bp']  # Radiocarbon years
c14_sigma = intcal20['c14_sigma']  # Error in radiocarbon years

# PyMC model definition
with pm.Model() as model:
    # 1. Define prior for calendar time (tau)
    tau = pm.Normal('tau', mu=2500, sigma=200)
    
    # 2. Interpolate the calibration curve to get the latent radiocarbon mean and error
    calcurve_mean = pytensor_linear_interp(tau, calbp, c14bp)
    calcurve_error = pytensor_linear_interp(tau, calbp, c14_sigma)
    
    # 3. Latent radiocarbon date (uncalibrated)
    r_latent = pm.Normal('r_latent', mu=calcurve_mean, sigma=calcurve_error, observed=2400)
    
    # 4. Observed radiocarbon date with measurement error (lab error)
    #r_measured = pm.Normal('r_measured', mu=r_latent, sigma=30, observed=2400)
    
    # 5. Sample from the posterior
    trace = pm.sample(draws=5000, chains=1)

# After running, you can inspect the results as follows:
pm.plot_trace(trace)
