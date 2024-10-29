import os
import pandas as pd
import numpy as np
import warnings

# Path to the calcurves folder
CACHE_DIR = os.path.join(os.path.dirname(__file__), "calcurves")

# Ensure the calcurves directory exists
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

def ensure_strictly_increasing(calbp, c14bp):
    """
    Check if calbp and c14bp are strictly increasing.
    If not, issue a warning and reverse the sign for each element.
    """
    if not (np.all(np.diff(calbp) > 0) and np.all(np.diff(c14bp) > 0)):
        warnings.warn("Calibration curve data is not strictly increasing. Adjusting sign to correct...")
        calbp = -1 * calbp
        c14bp = -1 * c14bp
    return calbp, c14bp

def download_intcal20(cache=True):
    cache_file = os.path.join(CACHE_DIR, "intcal20_cache.csv")
    
    # Check if cache exists and return cached data if available
    if cache and os.path.exists(cache_file):
        cached_data = pd.read_csv(cache_file)
        calbp, c14bp = ensure_strictly_increasing(cached_data['calbp'].values, cached_data['c14bp'].values)
        return {
            'calbp': calbp,
            'c14bp': c14bp,
            'c14_sigma': cached_data['c14_sigma'].values
        }
    
    # Download the calibration curve data
    url = "https://intcal.org/curves/intcal20.14c"
    intcal20 = pd.read_csv(url, skiprows=10, delimiter=",")
    intcal20.columns = ["calbp", "c14bp", "c14_sigma", "f14c", "f14c_sigma"]
    
    # Ensure calbp and c14bp are strictly increasing
    calbp, c14bp = ensure_strictly_increasing(intcal20['calbp'].values, intcal20['c14bp'].values)
    
    # Cache the data for future use
    if cache:
        intcal20['calbp'] = calbp
        intcal20['c14bp'] = c14bp
        intcal20.to_csv(cache_file, index=False)

    return {
        'calbp': calbp,
        'c14bp': c14bp,
        'c14_sigma': intcal20['c14_sigma'].values
    }

def add_custom_curve(name, calbp, c14bp, c14_sigma):
    # Ensure calbp and c14bp are strictly increasing
    calbp, c14bp = ensure_strictly_increasing(calbp, c14bp)
    
    # Add the curve to the dictionary
    calibration_curves[name] = {
        'calbp': calbp,
        'c14bp': c14bp,
        'c14_sigma': c14_sigma
    }