Usage
=====

Installation
------------

Install from a clone of the repository,

.. code-block:: bash

   pip install ".[fast]"

or directly from GitHub,

.. code-block:: bash

   pip install "cbc_spin_priors[fast] @ git+https://github.com/ColmTalbot/cbc_spin_priors.git"

The extras are:

- ``fast``: numba, which compiles the CDF and inverse-CDF kernels. Without
  it the same code runs as plain Python and numpy, which is much slower. This
  is recommended for any real analysis.
- ``test``: pytest and array-api-strict, for the test suite.
- ``examples``: lalsuite, dynesty and matplotlib, for
  ``examples/run_injection.py``.
- ``docs``: Sphinx and the theme, to build this documentation.

The package needs a version of bilby with the array-API interface
(``bilby.compat.utils``), 3.0.0 or later. If your package index does not have
it yet, install bilby from its repository first:

.. code-block:: bash

   pip install "bilby @ git+https://github.com/bilby-dev/bilby.git"

Setting the spin priors of a ``BBHPriorDict``
---------------------------------------------

:func:`~cbc_spin_priors.spins.set_spin_priors` replaces the spin priors of a
``bilby.gw.prior.BBHPriorDict`` in place and returns it:

.. code-block:: python

   from bilby.gw.prior import BBHPriorDict
   from cbc_spin_priors import set_spin_priors

   priors = set_spin_priors(BBHPriorDict(), "effective_perp")
   samples = priors.sample(1000)

The second argument is one of :data:`~cbc_spin_priors.spins.SPINS`:

.. list-table::
   :header-rows: 1
   :widths: 18 40 42

   * - ``spin``
     - aligned components
     - in-plane components
   * - ``aligned``
     - ``chi_1``, ``chi_2`` with bilby's ``AlignedSpin``
     - ``chi_1_in_plane``, ``chi_2_in_plane`` with ``ConditionalChiInPlane``,
       ``phi_12`` uniform
   * - ``effective``
     - ``chi_eff`` (:class:`~cbc_spin_priors.aligned.ChiEffQ`), ``chi_2``
       (:class:`~cbc_spin_priors.aligned.Chi2`), ``chi_1`` a delta function
       (:class:`~cbc_spin_priors.aligned.Chi1`)
     - as for ``aligned``
   * - ``spherical``
     - bilby's default ``a_1``, ``a_2``, ``tilt_1``, ``tilt_2``
     - bilby's default ``phi_12``
   * - ``aligned_perp``
     - as for ``aligned``
     - ``chi_in_plane_total``, ``chi_2_in_plane``, ``phi_s2``; ``chi_1_in_plane``
       and ``phi_12`` delta functions
   * - ``effective_perp``
     - as for ``effective``
     - as for ``aligned_perp``

All five describe the same distribution of spins, uniform in magnitude up to
``amax`` and isotropic in orientation; see :doc:`theory`. ``phi_jl`` keeps
its uniform prior in every case.

The other arguments are:

- ``amax``: the maximum spin magnitude, 0.99
  (:data:`~cbc_spin_priors.aligned.AMAX`) by default.
- ``aligned``: set this to ``True`` for aligned-spin analyses, with a
  ``BBHPriorDict(aligned_spin=True)``. The ``_perp`` parameterisations need
  precessing spins and raise a ``ValueError`` if it is set.

For ``effective`` and ``effective_perp``, ``priors["mass_ratio"]`` must
already be set, since its bounds are stored on the ``chi_eff`` prior. They
are metadata only and do not change the density.

The conditional delta functions (``chi_1``, ``chi_1_in_plane`` and
``phi_12``) set ``is_fixed = False``, so bilby treats them as search
parameters: they are computed from the other parameters for every sample and
passed to the likelihood. Each still has a unit-cube coordinate, which does
not change its value.

Adding the in-plane priors to your own prior dict
-------------------------------------------------

:func:`~cbc_spin_priors.inplane.total_in_plane_spin_priors` returns the five
in-plane priors as a dict. Add them to a
``bilby.core.prior.ConditionalPriorDict`` that already has ``mass_ratio``,
``chi_1``, ``chi_2`` and ``phi_jl``, and that does not have ``phi_12``
(``phi_12`` is one of the returned priors):

.. code-block:: python

   import numpy as np
   from bilby.core.prior import Uniform
   from bilby.gw.prior import AlignedSpin, BBHPriorDict
   from cbc_spin_priors import total_in_plane_spin_priors

   priors = BBHPriorDict(aligned_spin=True)
   priors["chi_1"] = AlignedSpin(Uniform(0, 0.99), name="chi_1")
   priors["chi_2"] = AlignedSpin(Uniform(0, 0.99), name="chi_2")
   priors["phi_jl"] = Uniform(0, 2 * np.pi, boundary="periodic", name="phi_jl")
   priors.update(total_in_plane_spin_priors(amax=0.99))

The ``amax`` of the in-plane priors must match the maximum magnitude used
for the aligned spins. The in-plane priors give the standard prior's
distribution of the in-plane spins given the aligned spins. With a different
prior on the aligned spins, the in-plane spins still follow that conditional
distribution, but the spins as a whole are no longer uniform in magnitude and
isotropic.

The helper functions :func:`~cbc_spin_priors.inplane.in_plane_from_total` and
:func:`~cbc_spin_priors.inplane.total_from_in_plane` convert between
``(chi_in_plane_total, chi_2_in_plane, phi_s2)`` and
``(chi_1_in_plane, chi_2_in_plane, phi_12)``, for example to convert an
injection or a set of posterior samples.

Other array libraries
---------------------

Every prior accepts arrays from any library that supports the Python array
API standard, such as torch, JAX or array-api-strict, and returns an array of
the same library. Numpy arrays and Python scalars use the numba kernels; other
arrays use a generic implementation. ``array_api_compat`` detects the library
from the inputs, or from the ``xp`` that bilby passes. No array library other
than numpy is imported by the package.

.. code-block:: python

   import torch
   from cbc_spin_priors import ChiInPlaneTotal

   prior = ChiInPlaneTotal()
   uu = torch.tensor([0.1, 0.5, 0.9], dtype=torch.float64, requires_grad=True)
   rho = prior.rescale(uu, chi_1=0.3, chi_2=-0.6, mass_ratio=0.4)
   rho.sum().backward()  # uu.grad is 1 / prob(rho)

Numpy conditioning variables can be mixed with arrays of one other library;
the result is in the other library. The generic path computes in float64. It
agrees with the numba kernels to a relative tolerance of :math:`10^{-10}`,
and it can be differentiated. It is several times slower than numba on CPU,
about 15 times for :class:`~cbc_spin_priors.inplane.ChiInPlaneTotal`.

Its limitations are:

- **float64:** everything is computed in float64, so JAX needs
  ``jax.config.update("jax_enable_x64", True)``.
- **jit:** the CDFs and densities trace under ``jax.jit``. The inverse CDFs
  (``rescale``) use Newton iterations that stop early only for eager arrays.
  Under tracing, every iteration up to the cap of 100 is unrolled, so
  compilation is slow. About 15 iterations suffice for the in-plane priors
  and :class:`~cbc_spin_priors.aligned.ChiEffQ`, and about 50 for
  :class:`~cbc_spin_priors.aligned.Chi2`, but the cap is not yet exposed by the prior
  classes.
- **Gradients:** these go through the iterations, so the backward pass costs
  about as much as the forward one.
- **bilby's own priors:** some of them, such as ``AlignedSpin`` and
  ``UniformInComponentsMassRatio``, do not accept non-numpy arrays yet, so a
  full ``BBHPriorDict`` still needs numpy inputs.

See :doc:`implementation` for how the dispatch works.

Examples
--------

The ``examples`` directory has three scripts.

``examples/inplane_unit_cube.py``
   Maps one unit-cube point through the in-plane priors by hand to
   ``chi_in_plane_total``, ``chi_2_in_plane`` and ``phi_s2``, recovers
   ``chi_1_in_plane`` and ``phi_12``, and maps back with the CDFs.

   .. code-block:: bash

      python examples/inplane_unit_cube.py

``examples/draw_prior_samples.py``
   Draws from all five parameterisations and prints KS p-values for
   :math:`a_1`, :math:`a_2`, :math:`\cos\theta_1`, :math:`\cos\theta_2` and
   :math:`\phi_{12}` against uniform distributions. With ``--plot``, it also
   writes histograms.

   .. code-block:: bash

      python examples/draw_prior_samples.py --n 20000 --plot spin_priors.png

``examples/run_injection.py``
   Analyses a simulated binary black hole signal in simulated Gaussian noise
   with dynesty, using any of the parameterisations. It needs the
   ``examples`` extra.

   .. code-block:: bash

      python examples/run_injection.py bbh effective_perp act-walk --naccept 5 --nact 3 --npool 16

   ``bbh`` uses the full likelihood with IMRPhenomXPHM. ``relbin`` uses
   relative binning with IMRPhenomXP on a signal with binary-neutron-star
   masses. Both marginalise over distance with a lookup table,
   ``{kind}_lookup.npz`` in the working directory, which is built if it is
   missing.
