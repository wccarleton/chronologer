# Chronologer

**Chronologer** is a Python package designed to support **Bayesian radiocarbon calibration** and **time-series modeling** using PyMC. It provides flexible tools for integrating chronological uncertainty into Bayesian models, allowing researchers to handle calibrated radiocarbon dates and related time-series data with ease.

## Features

- **Bayesian Radiocarbon Calibration**: Incorporate radiocarbon dating uncertainties directly into Bayesian models.
- **Inhomogeneous Poisson Process (IPPP)**: Model event counts over time with varying intensity functions that account for radiocarbon calibration uncertainty.
- **Custom Distributions**: Define and use custom radiocarbon calibration distributions with PyMC.
- **Pre-Specified Chronological Orders**: Implement prior information such as `date_a > date_b` in Bayesian models.

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
import pymc as pm
import chronologer as cg

# Set up the IPPP model
with pm.Model() as model:
    beta_0 = pm.Normal('beta_0', mu=0, sigma=10)
    beta_1 = pm.Normal('beta_1', mu=0, sigma=10)
    
    # Latent event times with radiocarbon calibration uncertainty
    t_i = cg.CalRadiocarbon('t_i', calcurve=calcurve, c14_mean=2500, c14_err=30)
    
    # Intensity function for the Poisson process
    def lambda_t(t):
        return pm.math.exp(beta_0 + beta_1 * t)
    
    likelihood = pm.Poisson('obs', mu=lambda_t(t_i), observed=your_data)
    
    trace = pm.sample()
```

## Documentation

Complete documentation is available at [Read the Docs](https://chronologer.readthedocs.io).

## Contributing

We welcome contributions! Please see the [contributing guidelines](CONTRIBUTING.md) for more details.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
