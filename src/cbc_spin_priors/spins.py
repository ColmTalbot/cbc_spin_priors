"""
Set the spin priors of a :code:`bilby.gw.prior.BBHPriorDict` to one of the
the parameterisations in :code:`SPINS`.

- :code:`aligned`: :math:`\\chi_1, \\chi_2` with :code:`AlignedSpin`, in-plane
  spins with :code:`ConditionalChiInPlane`.
- :code:`effective`: :math:`\\chi_\\mathrm{eff} | q` (:code:`ChiEffQ`),
  :math:`\\chi_2 | q, \\chi_\\mathrm{eff}` (:code:`Chi2`) and
  :math:`\\chi_1` as a conditional delta function, in-plane spins as for
  :code:`aligned`.
- :code:`spherical`: the default spin magnitudes and tilts.
- :code:`aligned_perp`, :code:`effective_perp`: as :code:`aligned` and
  :code:`effective`, but the in-plane spins and :math:`\\phi_{12}` are
  replaced by the total in-plane spin, the secondary in-plane spin and
  :math:`\\phi_{S2}` (see :mod:`cbc_spin_priors.inplane`).
"""

from bilby.core.prior import Uniform
from bilby.gw.prior import AlignedSpin, ConditionalChiInPlane

from .aligned import AMAX, Chi1, Chi2, ChiEffQ
from .inplane import total_in_plane_spin_priors

__all__ = ["SPINS", "set_spin_priors"]

SPINS = ("aligned", "effective", "spherical", "aligned_perp", "effective_perp")


def set_spin_priors(priors, spin, amax=AMAX, aligned=False):
    """
    Replace the spin priors of :code:`priors` in place.

    Parameters
    ==========
    priors: bilby.gw.prior.BBHPriorDict
        Created with :code:`aligned_spin=aligned`; :code:`effective` priors
        read the mass-ratio bounds from :code:`priors["mass_ratio"]`.
    spin: str
        One of :code:`SPINS`.
    amax: float
        The maximum spin magnitude.
    aligned: bool
        Whether the spins are aligned with the orbital angular momentum.
        The :code:`_perp` priors require precessing spins.

    Returns
    =======
    priors: the modified prior dict
    """
    # the "_perp" variants sample the in-plane spins as the total in-plane spin
    # and the secondary in-plane spin, see inplane.py
    in_plane_total = spin in ("effective_perp", "aligned_perp")
    base_spin = spin.removesuffix("_perp") if in_plane_total else spin

    if base_spin == "effective":
        priors["chi_eff"] = ChiEffQ(
            amax, priors["mass_ratio"].minimum, priors["mass_ratio"].maximum, name="chi_eff"
        )
        priors["chi_2"] = Chi2(amax)
        priors["chi_1"] = Chi1()
    elif base_spin == "aligned":
        priors["chi_1"] = AlignedSpin(Uniform(0, amax), name="chi_1")
        priors["chi_2"] = AlignedSpin(Uniform(0, amax), name="chi_2")
    elif base_spin == "spherical":
        priors["a_1"].maximum = amax
        priors["a_2"].maximum = amax
    else:
        raise ValueError(f"Unknown spin type: {spin}")

    if not aligned and spin != "spherical":
        del priors["a_1"], priors["a_2"], priors["tilt_1"], priors["tilt_2"]
        if in_plane_total:
            # replaces chi_1_in_plane, chi_2_in_plane and phi_12, phi_jl is unchanged
            del priors["phi_12"]
            for key, prior in total_in_plane_spin_priors(amax).items():
                priors[key] = prior
        else:
            priors["chi_1_in_plane"] = ConditionalChiInPlane(1e-5, amax, name="chi_1_in_plane")
            priors["chi_2_in_plane"] = ConditionalChiInPlane(1e-5, amax, name="chi_2_in_plane")
    elif in_plane_total:
        raise ValueError(f"{spin} requires precessing spins")
    return priors
