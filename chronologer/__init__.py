"""
Chronologer: A package for Bayesian radiocarbon calibration and time-series modeling.
"""

# Get calibration data
from .calcurves import download_intcal20, calibration_curves
download_intcal20()

# Import other core modules
from .core import *
from .calibration import *
from .distributions import *

# Define version
__version__ = "0.1.0"
