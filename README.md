# Chronologer

**Chronologer** is a Python package designed to support **Bayesian radiocarbon date calibration** and **time-series modeling** using PyMC. It provides flexible tools for integrating chronological uncertainty into Bayesian models, allowing researchers to handle calibrated radiocarbon dates and related time-series data with ease.

## Features

- **Bayesian Radiocarbon Calibration**: Incorporate radiocarbon dating uncertainties directly into Bayesian models using pytensor operations compatible with PyMC.
- **Inhomogeneous Poisson Process (IPPP)**: Model event counts over time with varying intensity functions that account for radiocarbon dating and calibration uncertainties.
-**Density-based Models**: Model event densities with standard models like Gaussian mixtures while
accounting for radiocarbon dating and calibration uncertainties.
- **Customize Phase Models**: Implement prior information such as `date_a > date_b` in Bayesian models using standard PyMC approaches.

## Installation

You can install **Chronologer** by cloning this repo and using `pip`:

```bash
git clone https://github.com/yourusername/chronologer.git
cd chronologer
pip install -e .
```

## Requirements
- Python >= 3.6
- NumPy
- SciPy
- PyMC
- Sphinx (for documentation)

The environment can be set up using the provided environment.yml file:

```bash
conda env create -f environment.yml
conda activate chronologer
```

## Usage

### Basic radiocarbon date calibration:

```python
import chronologer as cg

# Example usage for radiocarbon calibration
c14_mean = 2500  # Radiocarbon age
c14_err = 30     # Error in radiocarbon age

calcurve = cg.load_intcal_curve()  # Load IntCal20 calibration curve
cal_radiocarbon = cg.CalRadiocarbon(calcurve, 
                        c14_mean=c14_mean, 
                        c14_err=c14_err)

# Sample a calibrated date
sample = cal_radiocarbon.sample(size=1000)
```

### Inhomogeneous Poisson Process (IPPP) with Radiocarbon Uncertainty

```python
# Example usage to create initial intcal20 dictionary
intcal20 = download_intcal20()

# Users can add their custom curves to this dictionary
calibration_curves = {
    'intcal20': intcal20
}

# Extract the vectors from the calibration_curves dictionary
calbp_tensor = pt.as_tensor_variable(calibration_curves['intcal20']['calbp'])
c14bp_tensor = pt.as_tensor_variable(calibration_curves['intcal20']['c14bp'])
c14_sigma_tensor = pt.as_tensor_variable(calibration_curves['intcal20']['c14_sigma'])

# Example calibration curve data with negative values
radiocarbon_age = np.array([-2500, -2000])                      # Measured radiocarbon age
radiocarbon_error = np.array([30, 30])                          # Lab error

# Precompute calibration limits
cal_dates = calibrate(radiocarbon_ages = radiocarbon_age, 
                        radiocarbon_errors = radiocarbon_error, 
                        calbp = calbp, 
                        c14bp = c14bp, 
                        c14_sigma = c14_sigma, 
                        hdi_prob = 0.99)

N = len(radiocarbon_age)

lower_bound = cal_dates['HDI Lower (BP)'].values
upper_bound = cal_dates['HDI Upper (BP)'].values

mids = (upper_bound + lower_bound) / 2
prior_sd = np.repeat(200, N)

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
```

## Documentation

Complete documentation is available at [Read the Docs](https://chronologer.readthedocs.io).

## Contributing

We welcome contributions! Please see the [contributing guidelines](CONTRIBUTING.md) for more details.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
