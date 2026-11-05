#!/usr/bin/env python
"""
Walk one point through the total in-plane spin priors by hand: map a point in
the unit cube to (chi_in_plane_total, chi_2_in_plane, phi_s2), recover the
component in-plane spins, and map back with the CDFs.

usage:
    python inplane_unit_cube.py
"""

import numpy as np

from cbc_spin_priors import (
    AMAX,
    ChiInPlane2GivenTotal,
    ChiInPlaneTotal,
    PhiS2,
    in_plane_from_total,
    total_from_in_plane,
)

# aligned spins and mass ratio, as sampled before the in-plane spins
cond = dict(chi_1=0.3, chi_2=-0.2, mass_ratio=0.7)
uu = np.array([0.4, 0.7, 0.2])

total, secondary, angle = ChiInPlaneTotal(AMAX), ChiInPlane2GivenTotal(AMAX), PhiS2(AMAX)
rho = total.rescale(uu[0], **cond)
r2 = secondary.rescale(uu[1], chi_in_plane_total=rho, **cond)
phi_s2 = angle.rescale(uu[2], chi_in_plane_total=rho, chi_2_in_plane=r2, **cond)
chi_1_in_plane, phi_12 = in_plane_from_total(rho, r2, phi_s2, cond["mass_ratio"])

print(f"unit cube              {uu}")
print(f"chi_in_plane_total     {rho:.6f}")
print(f"chi_2_in_plane         {r2:.6f}")
print(f"phi_s2                 {phi_s2:.6f}")
print(f"chi_1_in_plane         {chi_1_in_plane:.6f}")
print(f"phi_12                 {phi_12:.6f}")
a_1 = np.hypot(cond["chi_1"], chi_1_in_plane)
a_2 = np.hypot(cond["chi_2"], r2)
print(f"a_1, a_2               {a_1:.6f}, {a_2:.6f}  (both <= {AMAX})")

back = [
    total.cdf(rho, **cond),
    secondary.cdf(r2, chi_in_plane_total=rho, **cond),
    angle.cdf(phi_s2, chi_in_plane_total=rho, chi_2_in_plane=r2, **cond),
]
print(f"cdf of each            {np.array(back)}")
print(f"total_from_in_plane    {total_from_in_plane(chi_1_in_plane, r2, phi_12, cond['mass_ratio'])}")
