# Ordinary PyMC execution baseline

Tested in `chronoapp` with Python 3.12.13, PyMC 6.3.2, PyTensor 3.3.2,
and ArviZ 1.3.0. No alternate backend or compiler was installed.

`tests/test_pymc_models.py` uses small synthetic curves and datasets. The
radiocarbon hierarchy and mixture structures follow `examples/example.py`,
`notebooks/test_drive.ipynb`, and `notebooks/density_model.ipynb`. IPPP models
exercise the functions currently exported by `chronologer.models`, rather
than obsolete notebook function signatures. Test-only priors and domains keep
rates positive and dates inside the synthetic curve. Production models and
their priors, parameterizations, integration convention, and settings are unchanged.

## Coverage

All six variants construct successfully, compile/evaluate finite scalar model
logp and finite continuous gradients, and execute PyMC's own NUTS through
`pm.sample(nuts_sampler="pymc")`:

- Single-date radiocarbon calibration with a latent radiocarbon measurement.
- Two-date radiocarbon calibration.
- Hierarchical truncated-normal calendar-date density with radiocarbon measurements.
- Two-component truncated-normal mixture with radiocarbon measurements.
- Sine-rate IPPP Potential with uncertain event dates and observation error.
- Linear-rate IPPP Potential with a scalar coefficient and uncertain event dates.

Each sampling test runs one chain, eight tuning steps, and eight retained draws,
with convergence checks disabled. These are execution tests, not evidence of
convergence or accurate posterior estimates. Gradients of the interpolation and
both scalar IPPP likelihoods are also checked against analytic slopes or finite
differences. Their existing numerical integration convention is preserved.

## Sampling result

Actual sampling results are `xarray.DataTree` objects. Tests verify `posterior`,
`sample_stats`, and `observed_data` children; posterior variables have leading
`chain` and `draw` dimensions of length 1 and 8, followed by their model shape.
Use `trace["posterior"].to_dataset()` for an xarray Dataset. Sampler statistics
include `diverging`.

Package source contains tensor likelihood/interpolation helpers, not a
`pm.sample` wrapper, and neither exposes nor consumes `InferenceData`.
Examples/notebooks retain raw PyMC sampler output in `trace` and pass it to
plotting/summary functions. No return-type shim or plotting migration was added.

## Symbolic interpolation

The Python expression `bin_index.shape[0] == 1` evaluates to Python `False`
under PyTensor 3.3.2; it does not create a symbolic conditional that fails during
construction. Consequently, even a scalar input returns length-one arrays.
That is the existing behavior used by `test_drive.ipynb` through `[0]` indexing.
It is now tested explicitly, together with length-one, length-two, and dynamic
vectors, evaluated values, and gradients. Model compilation and sampling pass.
The misleading scalar-return comment was not used as a reason to change the API.

## Known limitations and subsequent backend work

- **Multivariate linear IPPP is not certified by this baseline.** With a design
  matrix of shape `(2, 2)`, a coefficient vector of length 2, and the documented
  one-dimensional time grid of length 16, `ippp_logp_lm` raises a dot-product
  shape error during construction. Passing a matrix-valued domain instead
  produces a vector logp because `approx_integral` multiplies by the vector
  `domain[-1] - domain[0]`. This existing interface/integration problem needs a
  separate scientific decision; it was not rewritten as a compatibility fix.
- Interpolation uses comparison-derived integer indices and piecewise-linear
  slopes. Gradients are checked inside bins; knots, support boundaries, and
  the exact final grid point require care. Current interpolation divides by
  zero at the final point because both selected endpoints coincide. Tests keep
  priors strictly inside the curve and do not alter this boundary behavior.
- Future nutpie/Numba checks should separately test these indexing graphs,
  dynamic shapes, truncated-normal/mixture operations, and Potential gradients.
  Ordinary PyMC success does not establish alternate-backend support.
- IPPP rates must remain positive. Tests use valid positive supports; no link
  function or new positivity constraint was added to the library.
- Stale notebook imports/signatures and deprecated plotting aliases were not
  executed or modernized. No claim is made that entire old notebooks run.

## Validation

The new model suite passed: **12 passed, 5 warnings in 149.79 seconds**.
The final complete suite passed: **32 passed, 5 warnings in 19.07 seconds**
(after the first run had populated compilation caches). `pip check` reported
**No broken requirements found**.
Warnings were PyTensor's `Loop fusion failed because the resulting node would
exceed the kernel argument limit` (two hierarchical, three mixture).
PyTensor also logs the existing missing-g++ notice on import in this environment.
The full run logged six expected notices about only eight samples per chain
and unreliable R-hat/ESS. Convergence is intentionally not assessed. No
deprecation warnings were emitted by these tests. No package source changes
were necessary for the tested execution paths.

Run the baseline and complete suite with:

```powershell
conda run --no-capture-output -n chronoapp python -B -m pytest -q tests/test_pymc_models.py
conda run --no-capture-output -n chronoapp python -B -m pytest -q --log-cli-level=WARNING
conda run --no-capture-output -n chronoapp python -m pip check
```
