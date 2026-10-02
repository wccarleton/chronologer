"""Small callable-based convenience entry point; models own their validation."""


def fit(data, *, model, params=None, mcmc_config=None, progress_callback=None):
    """Delegate to a model callable, returning its result unchanged.

    Example: fit(events, model=chronologer.models.density.gmixture,
                 params={'K_max': 5}, mcmc_config={'draws': 1000, 'chains': 4}).
    No model-name registry, model conversion or sampling defaults are added here.
    """
    if not callable(model):
        raise TypeError('model must be a callable, such as chronologer.models.density.gmixture.')
    runtime = {} if progress_callback is None else {'progress_callback': progress_callback}
    return model(data, params=params, mcmc_config=mcmc_config, **runtime)
