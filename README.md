# cbc_spin_priors

Spin priors for compact binary coalescences in [bilby](https://git.ligo.org/lscsoft/bilby).

All of them reproduce spins with uniform magnitudes up to `amax` (default 0.99) and isotropic orientations. They differ only in which parameters the sampler sees.

The documentation, including the derivations, is at https://colmtalbot.github.io/cbc_spin_priors/ and builds with `sphinx-build docs docs/_build/html` after `pip install ".[docs]"`.

| module | priors | sampled parameters |
|---|---|---|
| `cbc_spin_priors.aligned` | `ChiEffQ`, `Chi2`, `Chi1` | χ_eff given q, χ₂ given (q, χ_eff); χ₁ is a conditional delta function |
| `cbc_spin_priors.inplane` | `ChiInPlaneTotal`, `ChiInPlane2GivenTotal`, `PhiS2`, `ChiInPlane1FromTotal`, `Phi12FromTotal` | the total in-plane spin \|S\|, the secondary in-plane spin and φ_S2; χ₁⊥ and φ₁₂ are conditional delta functions |
| `cbc_spin_priors.spins` | `set_spin_priors`, `SPINS` | sets one of five parameterisations in a `BBHPriorDict` |

The five parameterisations in `SPINS` are as follows.

- **aligned:** χ₁ and χ₂ use `AlignedSpin`. The in-plane spins use `ConditionalChiInPlane`.
- **effective:** χ_eff, χ₂ and χ₁ come from `cbc_spin_priors.aligned`. The in-plane spins are the same as for aligned.
- **spherical:** the default magnitudes and tilts.
- **aligned_perp:** the aligned spins of aligned, with the total in-plane spin priors.
- **effective_perp:** the aligned spins of effective, with the total in-plane spin priors.

The CDFs of `ChiEffQ`, `Chi2` and the in-plane priors are evaluated in closed form or by quadrature, without lookup tables; `ChiEffQ` and `Chi2` are inverted with a safeguarded Newton iteration, and `qmin`/`qmax` on `ChiEffQ` are metadata only.

### Other array libraries

Numpy arrays and Python scalars use the numba kernels. Arrays from any other library that supports the Python array API standard, such as torch, JAX or array-api-strict, use a generic implementation in `_xp.py`, `_inplane_xp.py` and `_aligned_xp.py`. `array_api_compat` detects the library from the inputs, or from the `xp` that bilby passes. The generic path computes in float64, agrees with numba to 1e-10, and can be differentiated: torch autograd is tested. It is several times slower than numba on CPU, about 15× for the total in-plane spin.

Limitations of the generic path:

- **jit:** it has not been run under `jax.jit`. Newton iterations stop early only for eager arrays. Under tracing, every iteration up to `maxiter` unrolls, so compilation is slow. The iteration cap is `maxiter=100`, an argument of the internal rescale functions that the prior classes do not expose yet; about 15 suffices for the in-plane and `ChiEffQ` priors, and about 50 for `Chi2`.
- **Gradients:** these go through the iterations, so the backward pass costs about as much as the forward one.
- **bilby's priors:** some of bilby's own priors, such as `AlignedSpin` and `UniformInComponentsMassRatio`, do not yet accept non-numpy arrays. A full `BBHPriorDict` therefore still needs numpy inputs.

## Install

```bash
pip install ".[fast,test]"
```

- **bilby:** the package needs a bilby with the array-API interface (`bilby.compat.utils`), version 3.0.0 or later.
- **numba (`fast` extra):** compiles the `Chi2` and in-plane kernels. Without it, the same code runs as plain Python and numpy, which is much slower.

## Use

```python
from bilby.gw.prior import BBHPriorDict
from cbc_spin_priors import set_spin_priors

priors = set_spin_priors(BBHPriorDict(), "effective_perp")
```

To use the in-plane priors with your own aligned-spin priors, add `total_in_plane_spin_priors(amax)` to a `ConditionalPriorDict`. It must already contain `mass_ratio`, `chi_1`, `chi_2` and `phi_jl`, and must not contain `phi_12`.

## Tests

```bash
python -m pytest
```

- `tests/test_aligned.py`: the effective-spin aligned priors, checked against uniform magnitudes and isotropic tilts, and `Chi2` checked against quadrature.
- `tests/test_inplane.py`: the in-plane priors. It checks CDF/rescale round trips, normalisation, densities against histograms and finite differences, and the joint distribution against the standard in-plane priors. It also checks the numba kernels against numpy, and serialisation.
- `tests/test_spins.py`: `set_spin_priors` with every parameterisation. It checks the keys, `amax`, the errors, that the spin distributions match, that the unit cube maps to finite parameters and density, and JSON round trips.
- `tests/test_array_api.py`: the generic path against the numba path on numpy, array-api-strict, torch and jax, as available. It also checks gradients against 1/pdf and finite differences. Backends that are not installed are skipped.

## Examples

- `examples/run_injection.py`: analyses a simulated binary black hole signal with dynesty, using any of the parameterisations. For example:

  ```bash
  python run_injection.py bbh effective_perp act-walk --naccept 5 --nact 3 --npool 16
  ```

  `bbh` uses the full likelihood with IMRPhenomXPHM. `relbin` uses relative binning with IMRPhenomXP on a binary-neutron-star-mass signal. Both marginalise over distance with `{kind}_lookup.npz` in the working directory, which is built if it is missing.
- `examples/draw_prior_samples.py`: draws from all five parameterisations and prints KS p-values for a₁, a₂, cos θ₁, cos θ₂ and φ₁₂. With `--plot`, it also writes histograms.
- `examples/inplane_unit_cube.py`: maps one unit-cube point through the in-plane priors to χ₁⊥ and φ₁₂, and back.
