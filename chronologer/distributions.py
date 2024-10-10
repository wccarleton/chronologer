import pymc as pm
import pytensor.tensor as pt
import numpy as np
from pymc.distributions.continuous import Continuous
from pytensor.tensor.random.op import RandomVariable
from chronologer.calcurves import calibration_curves

class CalRadiocarbonRV(RandomVariable):
    """
    Custom random variable for CalRadiocarbon.
    """
    name = "calradiocarbon"
    ndim_supp = 0  # Support for a scalar
    ndims_params = [0, 0, 0]  # Dimension for each parameter (scalar in this case)
    dtype = "floatX"  # Data type
    _print_name = ("CalRadiocarbon",)

    @classmethod
    def rng_fn(cls, rng, calbp, c14bp, c14_sigma, c14_mean, c14_err, size=None):
        """
        Random variable function to generate samples based on tau.
        """
        # Use inverse transform sampling to sample tau
        t_values, pdf_values = cls._get_pdf_values(c14_mean, c14_err, calbp, c14bp, c14_sigma)
        cdf_values = np.cumsum(pdf_values) * (t_values[1] - t_values[0])
        cdf_values /= cdf_values[-1]
        uniform_samples = rng.uniform(0, 1, size=size)
        tau_samples = np.interp(uniform_samples, cdf_values, t_values)
        return tau_samples

    @staticmethod
    def _get_pdf_values(c14_mean, c14_err, calbp, c14bp, c14_sigma, threshold=1e-7):
        """
        Helper function to calculate PDF values over the range.
        """
        t_values = np.linspace(min(calbp), max(calbp), 10000)
        pdf_values = pm.Normal.dist(mu=c14bp, sigma=np.sqrt(c14_err**2 + c14_sigma**2)).logp(c14_mean).eval()

        mask = pdf_values > threshold
        t_min = t_values[mask].min()
        t_max = t_values[mask].max()
        t_values = np.linspace(t_min, t_max, 10000)
        pdf_values = pm.Normal.dist(mu=c14bp, sigma=np.sqrt(c14_err**2 + c14_sigma**2)).logp(c14_mean).eval()

        pdf_values /= np.trapz(pdf_values, t_values)
        return t_values, pdf_values


class CalRadiocarbon(Continuous):
    """
    Custom calibrated radiocarbon date distribution using PyMC.
    """
    rv_op = CalRadiocarbonRV()

    def __init__(self, name, calcurve_name, c14_mean, c14_err, *args, **kwargs):
        super().__init__(name, *args, **kwargs)
        
        # Load the calibration curve data (calbp, c14bp, c14_sigma)
        self.calcurve = self.load_calcurve(calcurve_name)
        self.c14_mean = pt.as_tensor_variable(c14_mean)
        self.c14_err = pt.as_tensor_variable(c14_err)

    def load_calcurve(self, calcurve_name):
        """
        Load the calibration curve and extract the relevant numeric arrays.
        """
        calcurve = calibration_curves[calcurve_name]
        calbp = calcurve['calbp']
        c14bp = calcurve['c14bp']
        c14_sigma = calcurve['c14_sigma']
        return calbp, c14bp, c14_sigma

    def _calc_curve_params(self, tau):
        """
        Calculate curve parameters based on the input tau (time).
        """
        calbp, c14bp, c14_sigma = self.calcurve
        curve_mean = pt.extra_ops.interp(tau, calbp, c14bp)
        curve_error = pt.extra_ops.interp(tau, calbp, c14_sigma)
        return curve_mean, curve_error

    def logp(self, tau):
        """
        Log-probability function for the calibrated radiocarbon distribution.
        """
        curve_mean, curve_error = self._calc_curve_params(tau)
        combined_error = pt.sqrt(self.c14_err**2 + curve_error**2)
        
        # Log-probability based on the normal distribution
        return pm.logp(pm.Normal.dist(mu=curve_mean, sigma=combined_error), self.c14_mean)

    @classmethod
    def dist(cls, calcurve_name, c14_mean, c14_err, *args, **kwargs):
        """
        The dist method is used to create the distribution outside of a model context.
        This method returns the distribution as a standalone.
        """
        # Only unpack the relevant arrays
        calbp, c14bp, c14_sigma = calibration_curves[calcurve_name]['calbp'], calibration_curves[calcurve_name]['c14bp'], calibration_curves[calcurve_name]['c14_sigma']
        return cls.rv_op(calbp, c14bp, c14_sigma, c14_mean, c14_err, *args, **kwargs)

    def random(self, point=None, size=None, random_state=None):
        """
        Method for drawing random samples of tau from the distribution.
        """
        calbp, c14bp, c14_sigma = self.calcurve
        return self.rv_op.rng_fn(np.random.default_rng(), calbp, c14bp, c14_sigma, self.c14_mean.eval(), self.c14_err.eval(), size=size)
