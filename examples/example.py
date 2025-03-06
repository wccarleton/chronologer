import pymc as pm
import pytensor.tensor as pt
import numpy as np
from chronologer.calcurves import download_intcal20
from chronologer.calibration import calibrate, interpolate_calcurve

# Example usage to create initial intcal20 dictionary
intcal20 = download_intcal20()

# Example calibration curve data with negative values
radiocarbon_age = np.array([-2500, -2000])            # Measured radiocarbon age
radiocarbon_error = np.array([30, 30])                # Lab error

# Precompute calibration limits
cal_dates = calibrate(radiocarbon_ages = radiocarbon_age, 
                        radiocarbon_errors = radiocarbon_error, 
                        calbp = intcal20['calbp'], 
                        c14bp = intcal20['c14bp'], 
                        c14_sigma = intcal20['c14_sigma'], 
                        hdi_prob = 0.99)

N = len(radiocarbon_age)

lower_bound = cal_dates['HDI Lower (BP)'].values
upper_bound = cal_dates['HDI Upper (BP)'].values

mids = (upper_bound + lower_bound) / 2
prior_sd = np.repeat(200, N)

# recast calibration curve data as tensors for use in PyMC model context
calbp_tensor = pt.as_tensor_variable(intcal20['calbp'])
c14bp_tensor = pt.as_tensor_variable(intcal20['c14bp'])
c14_sigma_tensor = pt.as_tensor_variable(intcal20['c14_sigma'])

with pm.Model() as model:
    tau = pm.Uniform('tau', 
                    lower=lower_bound, 
                    upper=upper_bound,
                    shape=N)

    # calibration model
    mean, error = interpolate_calcurve(tau, 
                                       calbp_tensor, 
                                       c14bp_tensor, 
                                       c14_sigma_tensor)
    r_latent = pm.Normal('r_latent', 
                         mu=mean, 
                         sigma=error, 
                         shape=N)
    r_measured = pm.Normal('r_measured', 
                           mu=r_latent, 
                           sigma=radiocarbon_error, 
                           observed=radiocarbon_age,
                           shape=N)

    # Sample from the posterior
    trace = pm.sample(draws=5000, chains = 1, init="adapt_diag")

pm.summary(trace)

# compare with dates calibrated quickly using the standard approach
cal_dates

# Plot the results
pm.plot_trace(trace)