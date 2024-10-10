import os
import pandas as pd

# Path to the calcurves folder
CACHE_DIR = os.path.join(os.path.dirname(__file__), "calcurves")

# Ensure the calcurves directory exists
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

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

# Example usage to create initial intcal20 dictionary
intcal20 = download_intcal20()

# Users can add their custom curves to this dictionary
calibration_curves = {
    'intcal20': intcal20
}

# Users can update this dictionary with their custom curves
# Example: calibration_curves['custom_curve'] = custom_curve_data