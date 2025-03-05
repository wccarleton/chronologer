#!/usr/bin/env python3
# calcurve.py - Calibration curve loader for calrcarbon
# Author: Christopher Carleton
# GitHub: https://github.com/wccarleton/calrcarbon

import os
import pandas as pd

# Predefined calibration curves
DEFAULT_CURVES = {
    "intcal20": "https://intcal.org/curves/intcal20.14c",
    "shcal20": "https://intcal.org/curves/shcal20.14c",
    "marine20": "https://intcal.org/curves/marine20.14c",
}

CACHE_DIR = os.path.join(os.path.dirname(__file__), "calibration_curves")
os.makedirs(CACHE_DIR, exist_ok=True)

def load_calcurve(curve_name="intcal20", custom_path=None, quiet=False):
    """
    Loads a calibration curve either from the web (if built-in) or from a provided file path.

    Parameters
    ----------
    curve_name : str
        Name of the predefined calibration curve (e.g., "intcal20").
    custom_path : str, optional
        Path to a custom calibration curve file.

    Returns
    -------
    dict
        Dictionary with keys "calbp", "c14bp", "c14_sigma".

    Raises
    ------
    ValueError
        If the curve name is unknown and no custom file path is provided.
    """

    if custom_path is not None:
        curve_path = custom_path
        if not os.path.exists(curve_path):
            raise FileNotFoundError(f"Custom curve file not found: {curve_path}")
    elif curve_name in DEFAULT_CURVES:
        cached_file = os.path.join(CACHE_DIR, f"{curve_name}.14c")
        if not os.path.exists(cached_file):
            if not quiet:
                print(f"Downloading {curve_name}...")
            url = DEFAULT_CURVES[curve_name]
            df = pd.read_csv(url, skiprows=10, delimiter=",")
            df.columns = ["calbp", "c14bp", "c14_sigma", "f14c", "f14c_sigma"]
            df.to_csv(cached_file, index=False)
        else:
            if not quiet:
                print(f"Loading {curve_name} from cache.")
        curve_path = cached_file
    else:
        raise ValueError(f"Unknown curve '{curve_name}', and no custom_path provided.")

    # Load the curve
    df = pd.read_csv(curve_path)
    if not set(["calbp", "c14bp", "c14_sigma"]).issubset(df.columns):
        raise ValueError(f"Curve file {curve_path} does not contain required columns.")
    
    # Check if time in the file runs old to young (ascending) or young to old (descending)
    if df["calbp"].iloc[0] > df["calbp"].iloc[-1]:
        if not quiet:
            print(f"{curve_name} has descending calbp (older to younger) implying positive BP values. Converting to negative BP (older more negative).")
        df["calbp"] *= -1
        df["c14bp"] *= -1
    else:
        pass

    # Return as dict for compatibility with existing code
    return {
        "calbp": df["calbp"].values,
        "c14bp": df["c14bp"].values,
        "c14_sigma": df["c14_sigma"].values,
    }