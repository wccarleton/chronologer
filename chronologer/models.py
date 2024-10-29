import pymc as pm
import pytensor.tensor as pt

def approx_integral(rate_func, domain):
    """
    Approximates the integral of the rate function over a given domain.

    Parameters:
    -----------
    rate_func : callable
        The rate function to be integrated, should take a tensor as input.
    domain : tensor
        A sequence of points over which the rate function is evaluated.

    Returns:
    --------
    TensorVariable
        The approximate integral of the rate function over the domain.
    """
    # Evaluate the rate function at the points in the domain
    rate_values = rate_func(domain)

    # Number of evaluation points (inferred from the domain shape)
    eval_n = domain.shape[0]
    
    # Approximate the integral using the sum of the rate values times the step size
    # Assume equally spaced points in domain unless provided differently
    integral_rate = pt.sum(rate_values) * (domain[-1] - domain[0]) / eval_n
    
    return integral_rate

def ippp_logp_sine(value, a, b, domain):
    """
    Log-likelihood function for IPPP using a sine rate function with tensor-compatible parameters.

    Parameters:
    -----------
    value : tensor
        Observed event times as a PyTensor tensor.
    a : float
        Amplitude of the sine wave.
    b : float
        Period of the sine wave.
    domain : tensor
        The sequence of regularly-spaced points over which the GP or other 
        covariate function is evaluated (for integral approximation).

    Returns:
    --------
    TensorVariable
        Log-likelihood of observing the event times based on the IPPP model.
    """
    # Define the rate function using a and b
    rate_func = lambda t: a * (1 + pt.sin(2 * pt.pi * t / b))

    # Log-likelihood: sum of log(rate) at event times
    log_rate_sum = pt.sum(pt.log(rate_func(value)))

    # Approximate the integral over the interval [start, end]
    integral_rate = approx_integral(rate_func, domain)

    # Return the log-likelihood
    return log_rate_sum - integral_rate

def ippp_logp_lm(X_tau, Beta, domain):
    """
    Log-likelihood function for IPPP using a linear model, covariates, and 
    assuming a Gaussian Process sample for the covariate process with 
    tensor-compatible parameters.

    Parameters:
    -----------
    X_tau : tensor
        Covariate matrix (design matrix) evaluated at the observed (and uncertain) event times, 
        shape (n_events, n_covariates).
    Beta : tensor
        Regression coefficient vector of length n_covariates.
    domain : tensor
        The sequence of regularly-spaced points over which the GP or other 
        covariate function is evaluated (for integral approximation).

    Returns:
    --------
    TensorVariable
        Log-likelihood of observing the event times based on the IPPP model.
    """
    # Define the rate function as λ_t = X_tau * Beta
    rate_func = lambda X_t: pt.dot(X_t, Beta)

    # Log-likelihood: sum of log(rate) at event times τ
    log_rate_sum = pt.sum(pt.log(rate_func(X_tau)))

    # Approximate the integral over the domain
    integral_rate = approx_integral(rate_func, domain)

    # Return the log-likelihood
    return log_rate_sum - integral_rate
