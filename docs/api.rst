API reference
=============

Every name below is also importable from the top-level package, for example
``from cbc_spin_priors import ChiEffQ``.

All the prior classes are bilby conditional priors. They provide
``rescale``, ``cdf``, ``prob`` and ``ln_prob``, which take the value and the
required variables as keyword arguments, and an optional ``xp`` array
namespace. See :doc:`usage` for the array libraries that are supported.

``cbc_spin_priors.spins``
-------------------------

.. automodule:: cbc_spin_priors.spins
   :no-members:

.. autosummary::
   :nosignatures:

   set_spin_priors
   SPINS

.. autofunction:: cbc_spin_priors.spins.set_spin_priors

.. autodata:: cbc_spin_priors.spins.SPINS
   :no-value:

``cbc_spin_priors.aligned``
---------------------------

.. automodule:: cbc_spin_priors.aligned
   :no-members:

.. autosummary::
   :nosignatures:

   ChiEffQ
   Chi2
   Chi1
   AMAX

.. autoclass:: cbc_spin_priors.aligned.ChiEffQ

.. autoclass:: cbc_spin_priors.aligned.Chi2

.. autoclass:: cbc_spin_priors.aligned.Chi1

.. autodata:: cbc_spin_priors.aligned.AMAX

``cbc_spin_priors.inplane``
---------------------------

.. automodule:: cbc_spin_priors.inplane
   :no-members:

.. autosummary::
   :nosignatures:

   total_in_plane_spin_priors
   ChiInPlaneTotal
   ChiInPlane2GivenTotal
   PhiS2
   ChiInPlane1FromTotal
   Phi12FromTotal
   in_plane_from_total
   total_from_in_plane

.. autofunction:: cbc_spin_priors.inplane.total_in_plane_spin_priors

.. autoclass:: cbc_spin_priors.inplane.ChiInPlaneTotal

.. autoclass:: cbc_spin_priors.inplane.ChiInPlane2GivenTotal

.. autoclass:: cbc_spin_priors.inplane.PhiS2

.. autoclass:: cbc_spin_priors.inplane.ChiInPlane1FromTotal

.. autoclass:: cbc_spin_priors.inplane.Phi12FromTotal

.. autofunction:: cbc_spin_priors.inplane.in_plane_from_total

.. autofunction:: cbc_spin_priors.inplane.total_from_in_plane

Internal array-API helpers
--------------------------

These are not part of the public interface. They are listed because
:doc:`implementation` refers to them.

.. automodule:: cbc_spin_priors._xp
   :no-members:

.. autofunction:: cbc_spin_priors._xp.namespace

.. autofunction:: cbc_spin_priors._xp.as_float

.. autofunction:: cbc_spin_priors._xp.acos_clip

.. autofunction:: cbc_spin_priors._xp.newton_bisect

.. autofunction:: cbc_spin_priors._xp.spence

.. automodule:: cbc_spin_priors._inplane_xp
   :no-members:

.. automodule:: cbc_spin_priors._aligned_xp
   :no-members:
