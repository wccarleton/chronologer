import pymc as pm
import pytensor.tensor as pt
import numpy as np

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


# Example calibration curve data
calbp = np.array([1000, 2000, 3000])  # Calendar years
c14bp = np.array([1200, 2100, 2500])  # Radiocarbon years
c14_sigma = np.array([30, 40, 50])    # Error in radiocarbon years

# PyMC model definition
with pm.Model() as model:
    # 1. Define prior for calendar time (tau)
    tau = pm.Normal('tau', mu=2500, sigma=200)
    
    # 2. Interpolate the calibration curve based on sampled tau
    mean = pytensor_linear_interp(tau, calbp, c14bp)
    sigma = pytensor_linear_interp(tau, calbp, c14_sigma)
    
    # 3. Define a normal distribution for the observed radiocarbon age
    radiocarbon_time = pm.Normal('R_measured', mu=mean, sigma=sigma, observed=2400)
    
    # 4. Sample from the posterior
    trace = pm.sample()

# After running, you can inspect the results as follows:
pm.plot_trace(trace)


import pymc as pm
import pytensor.tensor as pt
import numpy as np
from chronologer.calcurves import calibration_curves

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
    r_latent = pm.Normal('r_latent', mu=calcurve_mean, sigma=calcurve_error)
    
    # 4. Observed radiocarbon date with measurement error (lab error)
    r_measured = pm.Normal('r_measured', mu=r_latent, sigma=30, observed=2400)
    
    # 5. Sample from the posterior
    trace = pm.sample()

# After running, you can inspect the results as follows:
pm.plot_trace(trace)
