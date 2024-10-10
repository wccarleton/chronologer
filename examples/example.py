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

with pm.Model() as model:
    # Use the custom CalRadiocarbon distribution
    tau = CalRadiocarbon('tau', 
                         calcurve_name='intcal20', 
                         c14_mean=2500, 
                         c14_err=30)
    
    # Define a simple model (e.g., Poisson model)
    likelihood = pm.Poisson('likelihood', 
                            mu=tau, 
                            observed=[5, 10, 15])
    
    # Sample from the posterior
    trace = pm.sample()
    
    # Print a summary of the trace
    print(pm.summary(trace))
    
    
# Import necessary modules
import matplotlib.pyplot as plt

# Instantiate the CalRadiocarbon distribution
cal_rad = CalRadiocarbon(name = 'tau', calcurve_name='intcal20', c14_mean=2500, c14_err=30)

# Generate samples from the random method
samples = cal_rad.random(size=1000)

# Plot a histogram of the tau samples
plt.hist(samples, bins=50, color='blue', alpha=0.7)
plt.xlabel('Calendar Age (tau)')
plt.ylabel('Frequency')
plt.title('Histogram of Calibrated Radiocarbon Samples (tau)')
plt.show()

# Create an instance of your custom distribution
cal_rad = CalRadiocarbon.dist(name='tau', 
                         calcurve_name='intcal20', 
                         c14_mean=2500, 
                         c14_err=30)

# Generate random samples
samples = cal_rad.random(size=10)
print("Random samples of tau:", samples)

# Evaluate the log-probability for a specific value of tau
tau_value = 1500
log_prob = cal_rad.logp(tau_value).eval()
print("Log probability of tau:", log_prob)

# Instantiate the distribution
cal_rad_dist = CalRadiocarbon.dist(calcurve_name='intcal20', c14_mean=2500, c14_err=30)

# Generate random samples from the distribution
samples = cal_rad_dist.random(size=10)
print("Random samples of tau:", samples)