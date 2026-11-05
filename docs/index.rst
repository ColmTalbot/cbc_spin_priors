cbc_spin_priors
===============

Spin priors for compact binary coalescences in
`bilby <https://github.com/bilby-dev/bilby>`_.

Every prior in this package describes the same spins: uniform magnitudes up
to :math:`\amax` (0.99 by default) and isotropic orientations. They differ
only in the coordinates the sampler sees.

- :mod:`cbc_spin_priors.aligned` samples the aligned spins as the effective
  spin :math:`\chieff` given the mass ratio and :math:`\chi_2` given
  :math:`(q, \chieff)`, with :math:`\chi_1` a conditional delta function.
- :mod:`cbc_spin_priors.inplane` samples the in-plane spins as the
  magnitude of the total in-plane spin :math:`|S|`, the secondary in-plane
  spin and the angle :math:`\phi_{S2}` between them, with :math:`\chi_{1\perp}`
  and :math:`\phi_{12}` conditional delta functions.
- :func:`~cbc_spin_priors.spins.set_spin_priors` sets one of five parameterisations
  in a :code:`bilby.gw.prior.BBHPriorDict`.

The CDFs are evaluated in closed form or by fixed-order quadrature, without
lookup tables, and are inverted with safeguarded Newton iterations. Numpy
inputs use compiled numba kernels; arrays from other libraries that support
the Python array API standard, such as torch and JAX, use a generic
implementation.

.. toctree::
   :maxdepth: 2

   usage
   theory
   implementation
   api
