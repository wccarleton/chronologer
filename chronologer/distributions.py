import pymc as pm
import pytensor.tensor as pt  # Use pytensor's tensor operations
import numpy as np
from chronologer.calcurves import calibration_curves, download_intcal20

class CalRadiocarbon(pm.Continuous):
    """Custom Calibrated Radiocarbon Date Distribution"""

    def __init__(self, calcurve_name='intcal20', c14_mean=None, c14_err=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Load the calibration curve from the available curves or download it
        if calcurve_name not in calibration_curves:
            raise ValueError(f"Calibration curve '{calcurve_name}' not found.")
        
        self.calcurve = calibration_curves[calcurve_name]
        self.c14_mean = pt.as_tensor_variable(c14_mean)  # Convert to pytensor variable
        self.c14_err = pt.as_tensor_variable(c14_err)  # Convert to pytensor variable

        # Set up calibration curve ranges
        self.a = -np.max(self.calcurve['calbp'])
        self.b = -np.min(self.calcurve['calbp'])

        # Use PyTensor's linear interpolation (equivalent to Aesara's interp)
        self._interp_mean = self.calcurve['calbp'], self.calcurve['c14bp']
        self._interp_error = self.calcurve['calbp'], self.calcurve['c14_sigma']

    def _calc_curve_params(self, tau):
        curve_mean = pt.extra_ops.interp(tau, self._interp_mean[0], self._interp_mean[1])
        curve_error = pt.extra_ops.interp(tau, self._interp_error[0], self._interp_error[1])
        return curve_mean, curve_error

    def logp(self, tau):
        """
        Calculate the log-probability of a given value tau.
        """
        curve_mean, curve_error = self._calc_curve_params(tau)
        combined_error = pt.sqrt(self.c14_err**2 + curve_error**2)

        # Use PyMC's Normal distribution for log-probability calculation
        return pm.logp(pm.Normal.dist(mu=curve_mean, sigma=combined_error), self.c14_mean)

    def random(self, point=None, size=None):
        """
        Random variates for generating random samples from the distribution.
        """
        t_values, pdf_values = self._get_pdf_values(self.c14_mean, self.c14_err)
        cdf_values = np.cumsum(pdf_values) * (t_values[1] - t_values[0])
        cdf_values /= cdf_values[-1]
        uniform_samples = np.random.uniform(0, 1, size=size)
        inverse_cdf = np.interp(uniform_samples, cdf_values, t_values)
        return inverse_cdf

    def _get_pdf_values(self, c14_mean, c14_err, threshold=1e-7):
        """Helper function to calculate the PDF values over the range."""
        t_values = np.linspace(self.a, self.b, 10000)
        pdf_values = pm.logp(pm.Normal.dist(mu=self._interp_mean(t_values),
                                            sigma=pt.sqrt(c14_err**2 + self._interp_error(t_values)**2)),
                                            c14_mean).eval()

        mask = pdf_values > threshold
        t_min = t_values[mask].min()
        t_max = t_values[mask].max()
        t_values = np.linspace(t_min, t_max, 10000)
        pdf_values = pm.logp(pm.Normal.dist(mu=self._interp_mean(t_values),
                                            sigma=pt.sqrt(c14_err**2 + self._interp_error(t_values)**2)),
                                            c14_mean).eval()

        pdf_values /= np.trapz(pdf_values, t_values)
        return t_values, pdf_values
