"""
Spin priors for compact binary coalescences in bilby: the effective-spin
aligned priors and the total in-plane spin priors.
"""

from . import aligned, inplane, spins
from .aligned import AMAX, Chi1, Chi2, ChiEffQ
from .inplane import (
    ChiInPlane1FromTotal,
    ChiInPlane2GivenTotal,
    ChiInPlaneTotal,
    Phi12FromTotal,
    PhiS2,
    in_plane_from_total,
    total_from_in_plane,
    total_in_plane_spin_priors,
)
from .spins import SPINS, set_spin_priors

__version__ = "0.1.0"

__all__ = aligned.__all__ + inplane.__all__ + spins.__all__ + ["AMAX"]
