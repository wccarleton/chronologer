import os
import sys

# Print the current working directory
print(f"Current working directory: {os.getcwd()}")

def in_interactive_mode():
    return hasattr(sys, 'ps1') or sys.flags.interactive

if in_interactive_mode():
    cache_base_dir = os.getcwd()  # Current working directory for interactive mode
else:
    cache_base_dir = os.path.dirname(__file__)

CACHE_DIR = os.path.join(cache_base_dir, "calcurves")

# Ensure the directory exists
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)
