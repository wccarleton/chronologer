"""Chronologer: tools for Bayesian radiocarbon date calibration."""

from __future__ import annotations

from .calcurves import load_calcurve
from .calibration import calibrate, hdi
from .density import (build_radiocarbon_density, fit_radiocarbon_density,
                      build_gaussian_mixture, fit_gaussian_mixture, evaluate_mixture_density)
from . import models
from .fitting import fit
from .phases import Phase, Order, group_phases, build_phase, fit_phase

__all__ = ["load_calcurve", "calibrate", "hdi", "build_radiocarbon_density", "fit_radiocarbon_density"]
__all__ += ["build_gaussian_mixture", "fit_gaussian_mixture", "evaluate_mixture_density"]
__all__ += ["models", "fit"]
__all__ += ["Phase", "Order", "group_phases", "build_phase", "fit_phase"]

# Define version
__version__ = "0.2.0"
