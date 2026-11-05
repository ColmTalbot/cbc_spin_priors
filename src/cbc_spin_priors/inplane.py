r"""
Conditional priors for the in-plane spins parameterised by the total in-plane
spin and the secondary in-plane spin.

Write each in-plane spin as a 2D vector :math:`v_i = \chi_{i\perp} e^{i\phi_i}`.
Under the standard prior (uniform magnitude, isotropic orientation, sampled as
the aligned component followed by :code:`ConditionalChiInPlane`), each vector
is radially symmetric given its aligned component :math:`\chi_i`

.. math::

    p(v_i | \chi_i) = \frac{1}{2\pi L_i (\chi_i^2 + |v_i|^2)},
    \quad |v_i| \leq R_i = \sqrt{a_{\max}^2 - \chi_i^2},
    \quad L_i = \ln(a_{\max} / |\chi_i|).

We sample, after the aligned spins,

1. :code:`chi_in_plane_total`: :math:`|S|` with
   :math:`S = (v_1 + q^2 v_2) / (1 + q^2)`, the mass-squared-weighted total
   in-plane spin (:math:`m_1^2 (1 + q^2) S` is the in-plane part of
   :math:`J = L + S_1 + S_2`). The azimuth of :math:`S` is uniform and plays the
   role of :code:`phi_jl`.
2. :code:`chi_2_in_plane`: :math:`|v_2|` given :math:`|S|`.
3. :code:`phi_s2`: the angle of :math:`v_2` relative to :math:`S`.
4. :code:`chi_1_in_plane` and :code:`phi_12` as conditional delta functions
   through :math:`v_1 = (1 + q^2) S - q^2 v_2`.

All CDFs are evaluated without lookup tables. Working in the space of
:math:`v_1` with :math:`a = (1 + q^2)|S|` and :math:`d = q^2 |v_2|`:

- the probability that :math:`v_1` lies in a disk of radius :math:`a` centred a
  distance :math:`d` from the origin is closed form before truncation at
  :math:`R_1`; the truncated part is a smooth 1D Gauss-Legendre integral;
- the arc integral of :math:`p(v_1)` along a circle (giving the densities of
  :math:`|S|` and :math:`|v_2|`) and the conditional CDF of the angle are
  closed form (arctangents);
- the remaining integral over :math:`|v_2|` is done with Gauss-Legendre
  quadrature in :math:`t = \mathrm{CDF}(|v_2| | \chi_2)` split at the known
  kinks, using a sinh map around the near-singular point :math:`d = a`;
- inverse CDFs use vectorised safeguarded Newton iterations.

Numpy arrays and Python scalars use the compiled numba kernels (or the
vectorised numpy fallback), arrays from any other library supporting the
Python array API standard use the generic implementation in
:mod:`cbc_spin_priors._inplane_xp`.
"""

import math

import numpy as np

try:
    import numba as _numba
except ImportError:
    _numba = None

from bilby.compat.utils import xp_wrap
from bilby.core.prior import ConditionalBasePrior, ConditionalDeltaFunction, Prior

from . import _inplane_xp, _xp

__all__ = [
    "ChiInPlaneTotal",
    "ChiInPlane2GivenTotal",
    "PhiS2",
    "ChiInPlane1FromTotal",
    "Phi12FromTotal",
    "total_in_plane_spin_priors",
    "in_plane_from_total",
    "total_from_in_plane",
]

AMAX = 0.99
# floor on |chi| to keep the logarithmic scales of the quadrature bounded,
# this modifies the prior for |chi| < CMIN which has probability ~1e-9
CMIN = 1e-10
# "numba" (compiled scalar kernels, much faster) or "numpy" (vectorised reference)
# for numpy inputs, "array_api" forces the generic implementation
BACKEND = "numba" if _numba is not None else "numpy"
N_OUTER = 20
N_INNER = 16

_X_OUTER, _W_OUTER = np.polynomial.legendre.leggauss(N_OUTER)
_X_OUTER = (_X_OUTER + 1) / 2
_W_OUTER = _W_OUTER / 2
_X_INNER, _W_INNER = np.polynomial.legendre.leggauss(N_INNER)
_X_INNER = (_X_INNER + 1) / 2
_W_INNER = _W_INNER / 2


# ---------------------------------------------------------------------------
# single-spin quantities
# ---------------------------------------------------------------------------

def _spin_constants(chi, amax):
    """Return (c, R, L): the floored |chi|, the in-plane radius and log(amax / c)."""
    c = np.clip(np.abs(chi), CMIN, amax * (1 - 1e-10))
    R = np.sqrt((amax - c) * (amax + c))
    L = np.log(amax / c)
    return c, R, L


def _r_to_t(r, c, L):
    """CDF of the in-plane magnitude given the aligned component."""
    return np.log1p((r / c) ** 2) / (2 * L)


def _t_to_r(t, c, L):
    """Inverse CDF of the in-plane magnitude given the aligned component."""
    return c * np.sqrt(np.expm1(2 * L * t))


def _p_r(r, c, L):
    """Density of the in-plane magnitude given the aligned component."""
    return r / (L * (c**2 + r**2))


# ---------------------------------------------------------------------------
# disk and arc integrals of p(v_1)
# ---------------------------------------------------------------------------

def _arc_integral(a, d, c, R):
    r"""
    :math:`J = \int_0^{\psi_c} d\psi / (c^2 + a^2 + d^2 - 2 a d \cos\psi)` where
    :math:`\psi_c` is the half-opening angle of the arc of the circle of radius
    :math:`a` centred a distance :math:`d` from the origin that lies inside
    the disk of radius :math:`R`.

    Also returns :math:`\psi_c`, :math:`A - B` and :math:`A + B`.
    """
    am = c**2 + (a - d) ** 2
    ap = c**2 + (a + d) ** 2
    ad2 = 2 * a * d
    with np.errstate(divide="ignore", invalid="ignore"):
        cos_c = np.where(
            ad2 > 0,
            (a**2 + d**2 - R**2) / ad2,
            np.where(a**2 + d**2 <= R**2, -1.0, 1.0),
        )
    psi_c = np.arccos(np.clip(cos_c, -1, 1))
    half = psi_c / 2
    sqm = np.sqrt(am)
    sqp = np.sqrt(ap)
    J = 2 / (sqm * sqp) * np.arctan2(sqp * np.sin(half), sqm * np.cos(half))
    return J, psi_c, am, ap


def _disk_probability(a, d, c, R, L):
    r"""
    Probability that :math:`v_1` lies inside the disk of radius :math:`a`
    centred a distance :math:`d` from the origin.

    This is the closed-form integral of the untruncated density over the disk
    minus the (smooth) part of that disk outside :math:`|v_1| = R`.
    Valid for :math:`a - R \leq d \leq a + R`.
    """
    a2 = a**2
    d2 = d**2
    c2 = c**2
    xx = a2 + c2 - d2
    zz = c2 + d2 - a2
    sq = np.sqrt(xx**2 + 4 * c2 * d2)
    with np.errstate(divide="ignore", invalid="ignore"):
        mm = np.where(zz > 0, 2 * a2 / (sq + zz), (sq - zz) / (2 * c2))
    p_full = np.log1p(mm) / (2 * L)

    # truncated part: s = |v_1|^2 from max(R^2, (a - d)^2) to (a + d)^2,
    # only needed where the disk crosses |v_1| = R
    trunc = np.zeros_like(p_full)
    crosses = a + d > R
    a, d, c2, R, L = [np.broadcast_to(xx, crosses.shape)[crosses] for xx in (a, d, c2, R, L)]
    mid = a**2 + d**2
    half = 2 * a * d
    with np.errstate(divide="ignore", invalid="ignore"):
        cos0 = np.where(half > 0, (mid - R**2) / half, -1.0)
    tau0 = np.arccos(np.clip(cos0, -1, 1))[..., None]
    width = np.pi - tau0
    tau = tau0 + width * _X_INNER
    ss = mid[..., None] - half[..., None] * np.cos(tau)
    with np.errstate(divide="ignore", invalid="ignore"):
        xarg = (d[..., None] - a[..., None] * np.cos(tau)) / np.sqrt(ss)
    phi = np.arccos(np.clip(np.nan_to_num(xarg, nan=1.0), -1, 1)) / np.pi
    integrand = half[..., None] * np.sin(tau) / (c2[..., None] + ss) * phi
    trunc[crosses] = np.sum(width * _W_INNER * integrand, axis=-1) / (2 * L)
    return np.clip(p_full - trunc, 0, 1)


def _disk_probability_derivative(a, d, c, R, L):
    """Derivative of :code:`_disk_probability` with respect to a."""
    J = _arc_integral(a, d, c, R)[0]
    return a * J / (np.pi * L)


# ---------------------------------------------------------------------------
# quadrature over t = CDF(|v_2|)
# ---------------------------------------------------------------------------

def _half_segment_nodes(anchor, end, eps, frac=1.0):
    r"""
    Nodes and weights for integrating from :code:`anchor` towards :code:`end`
    (either direction) over a fraction of the way, using the map
    :math:`x = x_0 \pm \epsilon \sinh(Y s^2)` with :math:`s \in [0, s_{\max}]`.

    The quadratic part removes square-root endpoint behaviour and the sinh
    part resolves near-singular features of width :math:`\epsilon` at the
    anchor. :code:`eps=inf` gives the pure quadratic map.
    """
    delta = end - anchor
    sgn = np.sign(delta)
    span = np.abs(delta)
    eps = np.minimum(np.maximum(eps, 1e-14 * span), 1e3 * span + 1e-300)
    Y = np.arcsinh(span / eps)
    smax = np.sqrt(np.arcsinh(frac * span / eps) / np.where(Y > 0, Y, 1.0))
    ss = smax[..., None] * _X_OUTER
    ww = smax[..., None] * _W_OUTER
    yy = Y[..., None] * ss**2
    xx = anchor[..., None] + (sgn * eps)[..., None] * np.sinh(yy)
    jac = (eps * 2 * Y)[..., None] * ss * np.cosh(yy)
    return xx, ww * jac


class _Geometry:
    """Pre-computed quantities for a given |S| and conditioning variables."""

    def __init__(self, rho, chi_1, chi_2, mass_ratio, amax):
        rho, chi_1, chi_2, mass_ratio = np.broadcast_arrays(
            *[np.asarray(xx, dtype=float) for xx in (rho, chi_1, chi_2, mass_ratio)]
        )
        self.shape = rho.shape
        self.amax = amax
        self.c1, self.R1, self.L1 = _spin_constants(chi_1.ravel(), amax)
        self.c2, self.R2, self.L2 = _spin_constants(chi_2.ravel(), amax)
        self.q2 = mass_ratio.ravel() ** 2
        self.rho_max = (self.R1 + self.q2 * self.R2) / (1 + self.q2)
        self.set_rho(rho.ravel())

    def set_rho(self, rho):
        self.rho = np.clip(rho, 0, self.rho_max)
        q2 = self.q2
        a = (1 + q2) * self.rho
        self.a = a

        def d_to_t(dd):
            return _r_to_t(np.clip(dd / q2, 0, self.R2), self.c2, self.L2)

        d_lo = np.maximum(0, a - self.R1)
        d_hi = np.minimum(q2 * self.R2, a + self.R1)
        t_lo = d_to_t(d_lo)
        t_hi = np.maximum(d_to_t(d_hi), t_lo)
        t_k = np.clip(d_to_t(np.maximum(self.R1 - a, 0)), t_lo, t_hi)
        # width of the near-singular feature at d = a mapped to t, if the
        # feature lies outside the range we cluster nodes at the nearest end
        # with a width set by the distance to the feature
        r_a = a / q2
        t_a_raw = _r_to_t(r_a, self.c2, self.L2)
        t_a = np.clip(t_a_raw, t_lo, t_hi)
        eps_a = np.hypot(self.c1 / q2 * _p_r(r_a, self.c2, self.L2), t_a_raw - t_a)
        swap = t_k < t_a
        knots = np.stack([
            t_lo, np.where(swap, t_k, t_a), np.where(swap, t_a, t_k), t_hi
        ], axis=-1)
        inf = np.full_like(t_lo, np.inf)
        eps = np.stack([
            inf, np.where(swap, inf, eps_a), np.where(swap, eps_a, inf), inf
        ], axis=-1)
        mids = (knots[:, 1:] + knots[:, :-1]) / 2
        # six half segments ordered in increasing t
        self.anchor = np.stack([
            knots[:, 0], knots[:, 1], knots[:, 1], knots[:, 2], knots[:, 2], knots[:, 3]
        ], axis=-1)
        self.end = np.repeat(mids, 2, axis=-1)
        self.eps = np.stack([
            eps[:, 0], eps[:, 1], eps[:, 1], eps[:, 2], eps[:, 2], eps[:, 3]
        ], axis=-1)
        # +1 for halves anchored at their lower end
        self.direction = np.array([1, -1, 1, -1, 1, -1])
        self.t_lo = t_lo
        self.t_hi = t_hi

    def subset(self, idx):
        """A copy restricted to the points idx."""
        new = object.__new__(_Geometry)
        npts = len(self.q2)
        for key, value in self.__dict__.items():
            if isinstance(value, np.ndarray) and value.ndim > 0 and value.shape[0] == npts and key != "direction":
                value = value[idx]
            setattr(new, key, value)
        new.shape = (len(np.atleast_1d(idx)),)
        return new

    def _expand(self, *args):
        return [xx[:, None, None] for xx in args]

    def d_of_t(self, tt):
        return self.q2[:, None, None] * _t_to_r(
            tt, self.c2[:, None, None], self.L2[:, None, None]
        )

    def outer_nodes(self):
        return _half_segment_nodes(self.anchor, self.end, self.eps)

    def cdf_and_pdf(self):
        """CDF and density of |S|."""
        tt, ww = self.outer_nodes()
        dd = self.d_of_t(tt)
        a, c1, R1, L1 = self._expand(self.a, self.c1, self.R1, self.L1)
        G = _disk_probability(a, dd, c1, R1, L1)
        dG = _disk_probability_derivative(a, dd, c1, R1, L1)
        cdf = self.t_lo + np.sum(ww * G, axis=(-1, -2))
        pdf = (1 + self.q2) * np.sum(ww * dG, axis=(-1, -2))
        return np.clip(cdf, 0, 1), pdf

    def r2_density_t(self, tt):
        """Unnormalised density of t = CDF(|v_2| | chi_2) given |S|."""
        dd = self.q2[..., None] * _t_to_r(tt, self.c2[..., None], self.L2[..., None])
        expand = [xx[..., None] for xx in (self.a, self.c1, self.R1, self.L1)]
        return _disk_probability_derivative(expand[0], dd, *expand[1:])

    def r2_half_integrals(self):
        tt, ww = self.outer_nodes()
        dd = self.d_of_t(tt)
        a, c1, R1, L1 = self._expand(self.a, self.c1, self.R1, self.L1)
        return np.sum(ww * _disk_probability_derivative(a, dd, c1, R1, L1), axis=-1)

    def r2_partial(self, idx, frac):
        """Integral of the t density over a fraction of half segment idx from its anchor."""
        rows = np.arange(len(idx))
        anchor = self.anchor[rows, idx]
        end = self.end[rows, idx]
        eps = self.eps[rows, idx]
        tt, ww = _half_segment_nodes(anchor, end, eps, frac=frac)
        dd = self.q2[:, None] * _t_to_r(tt, self.c2[:, None], self.L2[:, None])
        dG = _disk_probability_derivative(
            self.a[:, None], dd, self.c1[:, None], self.R1[:, None], self.L1[:, None]
        )
        return np.sum(ww * dG, axis=-1)

    def r2_cdf(self, r2):
        """CDF of |v_2| given |S|."""
        t_target = np.clip(_r_to_t(r2, self.c2, self.L2), self.t_lo, self.t_hi)
        halves = self.r2_half_integrals()
        total = np.sum(halves, axis=-1)
        lower = np.minimum(self.anchor, self.end)
        upper = np.maximum(self.anchor, self.end)
        cumulative = np.zeros_like(total)
        for idx in range(6):
            col = np.full(len(total), idx)
            span = upper[:, idx] - lower[:, idx]
            inside = (t_target > lower[:, idx]) & (t_target < upper[:, idx]) & (span > 0)
            full = t_target >= upper[:, idx]
            cumulative += np.where(full, halves[:, idx], 0.0)
            if np.any(inside):
                if self.direction[idx] == 1:
                    frac = (t_target - lower[:, idx]) / np.where(span > 0, span, 1)
                    part = self.r2_partial(col, np.clip(frac, 0, 1))
                else:
                    frac = (upper[:, idx] - t_target) / np.where(span > 0, span, 1)
                    part = halves[:, idx] - self.r2_partial(col, np.clip(frac, 0, 1))
                cumulative += np.where(inside, part, 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            out = np.where(total > 0, cumulative / total, (r2 >= _t_to_r(self.t_lo, self.c2, self.L2)) * 1.0)
        return np.clip(out, 0, 1), total

    def r2_rescale(self, uu):
        """Inverse CDF of |v_2| given |S|."""
        halves = self.r2_half_integrals()
        total = np.sum(halves, axis=-1)
        cumulative = np.cumsum(halves, axis=-1)
        target = uu * total
        idx = np.minimum(np.sum(cumulative < target[:, None], axis=-1), 5)
        rows = np.arange(len(uu))
        before = np.where(idx > 0, cumulative[rows, np.maximum(idx - 1, 0)], 0.0)
        this = halves[rows, idx]
        direction = self.direction[idx]
        # integral needed from the anchor of the selected half segment
        needed = np.where(direction == 1, target - before, before + this - target)
        needed = np.clip(needed, 0, this)

        anchor = self.anchor[rows, idx]
        end = self.end[rows, idx]
        eps = self.eps[rows, idx]
        span = np.abs(end - anchor)
        eps_eff = np.minimum(np.maximum(eps, 1e-14 * span), 1e3 * span + 1e-300)
        Y = np.arcsinh(span / eps_eff)

        def func(ss, act):
            # ss is the mapped variable in [0, 1]; frac in t follows from the map
            geom = self.subset(act)
            sp, ee, yy = span[act], eps_eff[act], Y[act]
            mapped = ee * np.sinh(yy * ss**2)
            frac = np.where(sp > 0, mapped / np.where(sp > 0, sp, 1), 0.0)
            value = geom.r2_partial(idx[act], np.clip(frac, 0, 1))
            tt = anchor[act] + np.sign(end[act] - anchor[act]) * mapped
            dens = geom.r2_density_t(tt[:, None])[:, 0] * ee * 2 * yy * ss * np.cosh(yy * ss**2)
            return value, dens

        with np.errstate(divide="ignore", invalid="ignore"):
            guess = np.clip(np.sqrt(np.where(this > 0, needed / this, 0.0)), 0, 1)
        ss = _newton_bisect(func, needed, np.zeros_like(guess), np.ones_like(guess), guess,
                            atol=1e-13 * np.maximum(total, 1e-300))
        tt = anchor + np.sign(end - anchor) * eps_eff * np.sinh(Y * ss**2)
        tt = np.clip(tt, self.t_lo, self.t_hi)
        return _t_to_r(tt, self.c2, self.L2)


def _newton_bisect(func, target, lo, hi, x0, atol, maxiter=100):
    """
    Vectorised safeguarded Newton iteration for an increasing function.
    :code:`func` returns the value and derivative.
    """
    xx = x0.copy()
    lo = lo.copy()
    hi = hi.copy()
    atol = np.broadcast_to(atol, xx.shape)
    previous = np.full(xx.shape, np.inf)
    active = np.arange(len(xx))
    for _ in range(maxiter):
        x_act = xx[active]
        value, deriv = func(x_act, active)
        resid = value - target[active]
        lo[active] = np.where(resid <= 0, x_act, lo[active])
        hi[active] = np.where(resid >= 0, x_act, hi[active])
        lo_act, hi_act = lo[active], hi[active]
        with np.errstate(divide="ignore", invalid="ignore"):
            new = x_act - resid / deriv
        # bisect if the step leaves the bracket or the last step did not halve the residual
        slow = np.abs(resid) > 0.5 * previous[active]
        previous[active] = np.abs(resid)
        bad = ~np.isfinite(new) | (new <= lo_act) | (new >= hi_act) | slow
        new = np.where(bad, (lo_act + hi_act) / 2, new)
        done = (np.abs(resid) <= atol[active]) | (
            hi_act - lo_act <= 1e-14 * np.maximum(np.abs(hi_act), 1e-300)
        )
        xx[active] = np.where(done, x_act, new)
        active = active[~done]
        if len(active) == 0:
            break
    return xx


# ---------------------------------------------------------------------------
# public functional interface
# ---------------------------------------------------------------------------

def _total_cdf_pdf_numpy(rho, chi_1, chi_2, mass_ratio, amax=AMAX):
    """CDF and density of |S| given the aligned spins and mass ratio."""
    geom = _Geometry(rho, chi_1, chi_2, mass_ratio, amax)
    cdf, pdf = geom.cdf_and_pdf()
    rho = np.broadcast_to(np.asarray(rho, dtype=float), geom.shape).ravel()
    pdf = np.where((rho >= 0) & (rho <= geom.rho_max), pdf, 0.0)
    return cdf.reshape(geom.shape), pdf.reshape(geom.shape)


def _total_rescale_numpy(uu, chi_1, chi_2, mass_ratio, amax=AMAX):
    """Inverse CDF of |S| given the aligned spins and mass ratio."""
    uu, chi_1, chi_2, mass_ratio = np.broadcast_arrays(
        *[np.asarray(xx, dtype=float) for xx in (uu, chi_1, chi_2, mass_ratio)]
    )
    shape = uu.shape
    uu = uu.ravel()
    geom = _Geometry(np.zeros(uu.shape), chi_1, chi_2, mass_ratio, amax)
    rho_max = geom.rho_max
    uu = np.clip(uu, 1e-300, 1 - 1e-16)

    # Newton iteration for logit(F) in terms of logit(rho / rho_max), the CDF
    # behaves as a power law in both tails so this is close to linear
    def func(zz, act):
        sub = geom.subset(act)
        frac = 1 / (1 + np.exp(-zz))
        rho = rho_max[act] * frac
        sub.set_rho(rho)
        cdf, pdf = sub.cdf_and_pdf()
        cdf = np.clip(cdf, 1e-300, 1 - 1e-16)
        value = np.log(cdf) - np.log1p(-cdf)
        deriv = pdf * rho * (1 - frac) / (cdf * (1 - cdf))
        return value, deriv

    target = np.log(uu) - np.log1p(-uu)
    guess = np.clip(target / 2, -300, 40)
    zz = _newton_bisect(func, target, np.full(uu.shape, -700.0), np.full(uu.shape, 40.0), guess, atol=1e-10)
    rho = rho_max / (1 + np.exp(-zz))
    return np.minimum(rho, rho_max).reshape(shape)


def _secondary_cdf_pdf_numpy(r2, rho, chi_1, chi_2, mass_ratio, amax=AMAX):
    """CDF and density of |v_2| given |S|, the aligned spins and mass ratio."""
    r2, rho, chi_1, chi_2, mass_ratio = np.broadcast_arrays(
        *[np.asarray(xx, dtype=float) for xx in (r2, rho, chi_1, chi_2, mass_ratio)]
    )
    geom = _Geometry(rho, chi_1, chi_2, mass_ratio, amax)
    r2 = r2.ravel()
    cdf, total = geom.r2_cdf(r2)
    tt = _r_to_t(r2, geom.c2, geom.L2)
    inside = (tt >= geom.t_lo) & (tt <= geom.t_hi) & (r2 <= geom.R2)
    with np.errstate(divide="ignore", invalid="ignore"):
        pdf = np.where(
            inside & (total > 0),
            geom.r2_density_t(tt[:, None])[:, 0] * _p_r(r2, geom.c2, geom.L2) / total,
            0.0,
        )
    return cdf.reshape(geom.shape), pdf.reshape(geom.shape)


def _secondary_rescale_numpy(uu, rho, chi_1, chi_2, mass_ratio, amax=AMAX):
    """Inverse CDF of |v_2| given |S|, the aligned spins and mass ratio."""
    uu, rho, chi_1, chi_2, mass_ratio = np.broadcast_arrays(
        *[np.asarray(xx, dtype=float) for xx in (uu, rho, chi_1, chi_2, mass_ratio)]
    )
    geom = _Geometry(rho, chi_1, chi_2, mass_ratio, amax)
    uu = uu.ravel()
    return geom.r2_rescale(uu).reshape(geom.shape)


# ---------------------------------------------------------------------------
# compiled scalar kernels (numba), mirroring the numpy implementation above
# ---------------------------------------------------------------------------

if _numba is not None:
    _jit = _numba.njit(cache=True, fastmath=False, error_model="numpy")
    _INF = np.inf
    _PI = np.pi

    @_jit
    def _nb_spin(chi, amax):
        c = min(max(abs(chi), CMIN), amax * (1 - 1e-10))
        return c, np.sqrt((amax - c) * (amax + c)), np.log(amax / c)

    @_jit
    def _nb_r_to_t(r, c, L):
        return np.log1p((r / c) ** 2) / (2 * L)

    @_jit
    def _nb_t_to_r(t, c, L):
        return c * np.sqrt(np.expm1(2 * L * t))

    @_jit
    def _nb_dgda(a, d, c, R, L):
        am = c * c + (a - d) ** 2
        ap = c * c + (a + d) ** 2
        ad2 = 2 * a * d
        if ad2 > 0:
            cos_c = (a * a + d * d - R * R) / ad2
        elif a * a + d * d <= R * R:
            cos_c = -1.0
        else:
            cos_c = 1.0
        half = np.arccos(min(max(cos_c, -1.0), 1.0)) / 2
        sqm = np.sqrt(am)
        sqp = np.sqrt(ap)
        J = 2 / (sqm * sqp) * np.arctan2(sqp * np.sin(half), sqm * np.cos(half))
        return a * J / (_PI * L)

    @_jit
    def _nb_disk(a, d, c, R, L):
        a2 = a * a
        d2 = d * d
        c2 = c * c
        xx = a2 + c2 - d2
        zz = c2 + d2 - a2
        sq = np.sqrt(xx * xx + 4 * c2 * d2)
        if zz > 0:
            mm = 2 * a2 / (sq + zz)
        else:
            mm = (sq - zz) / (2 * c2)
        out = np.log1p(mm) / (2 * L)
        if a + d > R:
            mid = a2 + d2
            half = 2 * a * d
            if half > 0:
                cos0 = (mid - R * R) / half
            else:
                cos0 = -1.0
            tau0 = np.arccos(min(max(cos0, -1.0), 1.0))
            width = _PI - tau0
            trunc = 0.0
            for i in range(N_INNER):
                tau = tau0 + width * _X_INNER[i]
                ss = mid - half * np.cos(tau)
                if ss > 0:
                    xarg = (d - a * np.cos(tau)) / np.sqrt(ss)
                else:
                    xarg = 1.0
                phi = np.arccos(min(max(xarg, -1.0), 1.0)) / _PI
                trunc += _W_INNER[i] * half * np.sin(tau) / (c2 + ss) * phi
            out -= width * trunc / (2 * L)
        return min(max(out, 0.0), 1.0)

    @_jit
    def _nb_knots(rho, c1, R1, L1, c2, R2, L2, q2):
        a = (1 + q2) * rho
        d_lo = max(0.0, a - R1)
        d_hi = min(q2 * R2, a + R1)
        t_lo = _nb_r_to_t(min(max(d_lo / q2, 0.0), R2), c2, L2)
        t_hi = max(_nb_r_to_t(min(max(d_hi / q2, 0.0), R2), c2, L2), t_lo)
        t_k = _nb_r_to_t(min(max(max(R1 - a, 0.0) / q2, 0.0), R2), c2, L2)
        t_k = min(max(t_k, t_lo), t_hi)
        r_a = a / q2
        t_a_raw = _nb_r_to_t(r_a, c2, L2)
        t_a = min(max(t_a_raw, t_lo), t_hi)
        eps_a = np.hypot(c1 / q2 * r_a / (L2 * (c2 * c2 + r_a * r_a)), t_a_raw - t_a)
        if t_k < t_a:
            k1, k2, e1, e2 = t_k, t_a, _INF, eps_a
        else:
            k1, k2, e1, e2 = t_a, t_k, eps_a, _INF
        m0 = (t_lo + k1) / 2
        m1 = (k1 + k2) / 2
        m2 = (k2 + t_hi) / 2
        anchor = np.array([t_lo, k1, k1, k2, k2, t_hi])
        end = np.array([m0, m0, m1, m1, m2, m2])
        eps = np.array([_INF, e1, e1, e2, e2, _INF])
        return a, t_lo, t_hi, anchor, end, eps

    @_jit
    def _nb_map(anchor, end, eps):
        span = abs(end - anchor)
        sgn = 1.0 if end >= anchor else -1.0
        e = min(max(eps, 1e-14 * span), 1e3 * span + 1e-300)
        return sgn, span, e, np.arcsinh(span / e)

    @_jit
    def _nb_total_cdf_pdf(rho, c1, R1, L1, c2, R2, L2, q2):
        a, t_lo, t_hi, anchor, end, eps = _nb_knots(rho, c1, R1, L1, c2, R2, L2, q2)
        cdf = t_lo
        pdf = 0.0
        for h in range(6):
            sgn, span, e, Y = _nb_map(anchor[h], end[h], eps[h])
            if span <= 0:
                continue
            for i in range(N_OUTER):
                s = _X_OUTER[i]
                y = Y * s * s
                tt = anchor[h] + sgn * e * np.sinh(y)
                w = _W_OUTER[i] * e * 2 * Y * s * np.cosh(y)
                d = q2 * _nb_t_to_r(tt, c2, L2)
                cdf += w * _nb_disk(a, d, c1, R1, L1)
                pdf += w * _nb_dgda(a, d, c1, R1, L1)
        return min(max(cdf, 0.0), 1.0), (1 + q2) * pdf

    @_jit
    def _nb_total_rescale(u, chi_1, chi_2, mass_ratio, amax):
        c1, R1, L1 = _nb_spin(chi_1, amax)
        c2, R2, L2 = _nb_spin(chi_2, amax)
        q2 = mass_ratio * mass_ratio
        rho_max = (R1 + q2 * R2) / (1 + q2)
        u = min(max(u, 1e-300), 1 - 1e-16)
        target = np.log(u) - np.log1p(-u)
        lo, hi = -700.0, 40.0
        zz = min(max(target / 2, -300.0), 40.0)
        previous = _INF
        for _ in range(100):
            frac = 1 / (1 + np.exp(-zz))
            rho = rho_max * frac
            cdf, pdf = _nb_total_cdf_pdf(rho, c1, R1, L1, c2, R2, L2, q2)
            cdf = min(max(cdf, 1e-300), 1 - 1e-16)
            resid = np.log(cdf) - np.log1p(-cdf) - target
            if abs(resid) <= 1e-10:
                break
            if resid < 0:
                lo = zz
            else:
                hi = zz
            if hi - lo <= 1e-14 * max(abs(hi), 1e-300):
                break
            new = zz - resid * cdf * (1 - cdf) / (pdf * rho * (1 - frac))
            if not (lo < new < hi) or abs(resid) > 0.5 * previous:
                new = (lo + hi) / 2
            previous = abs(resid)
            zz = new
        return min(rho_max / (1 + np.exp(-zz)), rho_max)

    @_jit
    def _nb_r2_half_integral(h, frac, a, c1, R1, L1, c2, L2, q2, anchor, end, eps):
        sgn, span, e, Y = _nb_map(anchor[h], end[h], eps[h])
        if span <= 0 or frac <= 0:
            return 0.0
        smax = np.sqrt(np.arcsinh(frac * span / e) / Y)
        out = 0.0
        for i in range(N_OUTER):
            s = smax * _X_OUTER[i]
            y = Y * s * s
            tt = anchor[h] + sgn * e * np.sinh(y)
            d = q2 * _nb_t_to_r(tt, c2, L2)
            out += smax * _W_OUTER[i] * e * 2 * Y * s * np.cosh(y) * _nb_dgda(a, d, c1, R1, L1)
        return out

    @_jit
    def _nb_secondary_cdf_pdf(r2, rho, chi_1, chi_2, mass_ratio, amax):
        c1, R1, L1 = _nb_spin(chi_1, amax)
        c2, R2, L2 = _nb_spin(chi_2, amax)
        q2 = mass_ratio * mass_ratio
        rho = min(max(rho, 0.0), (R1 + q2 * R2) / (1 + q2))
        a, t_lo, t_hi, anchor, end, eps = _nb_knots(rho, c1, R1, L1, c2, R2, L2, q2)
        t_raw = _nb_r_to_t(r2, c2, L2)
        t_target = min(max(t_raw, t_lo), t_hi)
        total = 0.0
        cumulative = 0.0
        for h in range(6):
            full = _nb_r2_half_integral(h, 1.0, a, c1, R1, L1, c2, L2, q2, anchor, end, eps)
            total += full
            lower = min(anchor[h], end[h])
            upper = max(anchor[h], end[h])
            span = upper - lower
            if t_target >= upper:
                cumulative += full
            elif t_target > lower and span > 0:
                if h % 2 == 0:
                    cumulative += _nb_r2_half_integral(
                        h, (t_target - lower) / span, a, c1, R1, L1, c2, L2, q2, anchor, end, eps)
                else:
                    cumulative += full - _nb_r2_half_integral(
                        h, (upper - t_target) / span, a, c1, R1, L1, c2, L2, q2, anchor, end, eps)
        if total > 0:
            cdf = min(max(cumulative / total, 0.0), 1.0)
        else:
            cdf = 1.0 if r2 >= _nb_t_to_r(t_lo, c2, L2) else 0.0
        if t_lo <= t_raw <= t_hi and r2 <= R2 and total > 0:
            pdf = _nb_dgda(a, q2 * r2, c1, R1, L1) * r2 / (L2 * (c2 * c2 + r2 * r2)) / total
        else:
            pdf = 0.0
        return cdf, pdf

    @_jit
    def _nb_secondary_rescale(u, rho, chi_1, chi_2, mass_ratio, amax):
        c1, R1, L1 = _nb_spin(chi_1, amax)
        c2, R2, L2 = _nb_spin(chi_2, amax)
        q2 = mass_ratio * mass_ratio
        rho = min(max(rho, 0.0), (R1 + q2 * R2) / (1 + q2))
        a, t_lo, t_hi, anchor, end, eps = _nb_knots(rho, c1, R1, L1, c2, R2, L2, q2)
        halves = np.empty(6)
        for h in range(6):
            halves[h] = _nb_r2_half_integral(h, 1.0, a, c1, R1, L1, c2, L2, q2, anchor, end, eps)
        total = halves.sum()
        target = u * total
        cumulative = 0.0
        idx = 5
        before = 0.0
        for h in range(6):
            if cumulative + halves[h] >= target:
                idx = h
                break
            cumulative += halves[h]
        before = cumulative
        this = halves[idx]
        if idx % 2 == 0:
            needed = target - before
        else:
            needed = before + this - target
        needed = min(max(needed, 0.0), this)
        sgn, span, e, Y = _nb_map(anchor[idx], end[idx], eps[idx])
        atol = 1e-13 * max(total, 1e-300)
        lo, hi = 0.0, 1.0
        ss = np.sqrt(needed / this) if this > 0 else 0.0
        ss = min(max(ss, 0.0), 1.0)
        previous = _INF
        if span > 0:
            for _ in range(100):
                mapped = e * np.sinh(Y * ss * ss)
                frac = min(max(mapped / span, 0.0), 1.0)
                value = _nb_r2_half_integral(idx, frac, a, c1, R1, L1, c2, L2, q2, anchor, end, eps)
                resid = value - needed
                if abs(resid) <= atol:
                    break
                if resid < 0:
                    lo = ss
                else:
                    hi = ss
                if hi - lo <= 1e-14:
                    break
                tt = anchor[idx] + sgn * mapped
                dens = _nb_dgda(a, q2 * _nb_t_to_r(tt, c2, L2), c1, R1, L1)
                deriv = dens * e * 2 * Y * ss * np.cosh(Y * ss * ss)
                new = ss - resid / deriv if deriv > 0 else -1.0
                if not (lo < new < hi) or abs(resid) > 0.5 * previous:
                    new = (lo + hi) / 2
                previous = abs(resid)
                ss = new
        tt = anchor[idx] + sgn * e * np.sinh(Y * ss * ss)
        tt = min(max(tt, t_lo), t_hi)
        return _nb_t_to_r(tt, c2, L2)

    @_jit
    def _nb_total_cdf_pdf_vec(rho, chi_1, chi_2, mass_ratio, amax):
        cdf = np.empty(rho.shape)
        pdf = np.empty(rho.shape)
        for i in range(rho.size):
            c1, R1, L1 = _nb_spin(chi_1[i], amax)
            c2, R2, L2 = _nb_spin(chi_2[i], amax)
            q2 = mass_ratio[i] ** 2
            rho_max = (R1 + q2 * R2) / (1 + q2)
            cdf[i], pdf[i] = _nb_total_cdf_pdf(min(max(rho[i], 0.0), rho_max), c1, R1, L1, c2, R2, L2, q2)
            if rho[i] < 0 or rho[i] > rho_max:
                pdf[i] = 0.0
        return cdf, pdf

    @_jit
    def _nb_total_rescale_vec(uu, chi_1, chi_2, mass_ratio, amax):
        out = np.empty(uu.shape)
        for i in range(uu.size):
            out[i] = _nb_total_rescale(uu[i], chi_1[i], chi_2[i], mass_ratio[i], amax)
        return out

    @_jit
    def _nb_secondary_cdf_pdf_vec(r2, rho, chi_1, chi_2, mass_ratio, amax):
        cdf = np.empty(r2.shape)
        pdf = np.empty(r2.shape)
        for i in range(r2.size):
            cdf[i], pdf[i] = _nb_secondary_cdf_pdf(r2[i], rho[i], chi_1[i], chi_2[i], mass_ratio[i], amax)
        return cdf, pdf

    @_jit
    def _nb_secondary_rescale_vec(uu, rho, chi_1, chi_2, mass_ratio, amax):
        out = np.empty(uu.shape)
        for i in range(uu.size):
            out[i] = _nb_secondary_rescale(uu[i], rho[i], chi_1[i], chi_2[i], mass_ratio[i], amax)
        return out


def _flat_inputs(*args):
    arrays = np.broadcast_arrays(*[np.asarray(xx, dtype=float) for xx in args])
    return arrays[0].shape, [np.ascontiguousarray(xx).ravel() for xx in arrays]


def _use_numba(backend):
    if backend is None:
        backend = BACKEND
    if backend == "numba" and _numba is None:
        raise ImportError("numba is not available")
    return backend == "numba"


def _generic(backend, xp, *args):
    """The namespace to use for the generic implementation, or None for numpy/numba."""
    xp = _xp.namespace(*args, xp=xp)
    if backend == "array_api" or not _xp.is_numpy(xp):
        return xp
    return None


def total_cdf_pdf(rho, chi_1, chi_2, mass_ratio, amax=AMAX, backend=None, xp=None):
    """CDF and density of |S| given the aligned spins and mass ratio."""
    if (xp := _generic(backend, xp, rho, chi_1, chi_2, mass_ratio)) is not None:
        return _inplane_xp.total_cdf_pdf(xp, rho, chi_1, chi_2, mass_ratio, amax)
    if not _use_numba(backend):
        return _total_cdf_pdf_numpy(rho, chi_1, chi_2, mass_ratio, amax)
    shape, args = _flat_inputs(rho, chi_1, chi_2, mass_ratio)
    cdf, pdf = _nb_total_cdf_pdf_vec(*args, float(amax))
    return cdf.reshape(shape), pdf.reshape(shape)


def total_rescale(uu, chi_1, chi_2, mass_ratio, amax=AMAX, backend=None, xp=None):
    """Inverse CDF of |S| given the aligned spins and mass ratio."""
    if (xp := _generic(backend, xp, uu, chi_1, chi_2, mass_ratio)) is not None:
        return _inplane_xp.total_rescale(xp, uu, chi_1, chi_2, mass_ratio, amax)
    if not _use_numba(backend):
        return _total_rescale_numpy(uu, chi_1, chi_2, mass_ratio, amax)
    shape, args = _flat_inputs(uu, chi_1, chi_2, mass_ratio)
    return _nb_total_rescale_vec(*args, float(amax)).reshape(shape)


def secondary_cdf_pdf(r2, rho, chi_1, chi_2, mass_ratio, amax=AMAX, backend=None, xp=None):
    """CDF and density of |v_2| given |S|, the aligned spins and mass ratio."""
    if (xp := _generic(backend, xp, r2, rho, chi_1, chi_2, mass_ratio)) is not None:
        return _inplane_xp.secondary_cdf_pdf(xp, r2, rho, chi_1, chi_2, mass_ratio, amax)
    if not _use_numba(backend):
        return _secondary_cdf_pdf_numpy(r2, rho, chi_1, chi_2, mass_ratio, amax)
    shape, args = _flat_inputs(r2, rho, chi_1, chi_2, mass_ratio)
    cdf, pdf = _nb_secondary_cdf_pdf_vec(*args, float(amax))
    return cdf.reshape(shape), pdf.reshape(shape)


def secondary_rescale(uu, rho, chi_1, chi_2, mass_ratio, amax=AMAX, backend=None, xp=None):
    """Inverse CDF of |v_2| given |S|, the aligned spins and mass ratio."""
    if (xp := _generic(backend, xp, uu, rho, chi_1, chi_2, mass_ratio)) is not None:
        return _inplane_xp.secondary_rescale(xp, uu, rho, chi_1, chi_2, mass_ratio, amax)
    if not _use_numba(backend):
        return _secondary_rescale_numpy(uu, rho, chi_1, chi_2, mass_ratio, amax)
    shape, args = _flat_inputs(uu, rho, chi_1, chi_2, mass_ratio)
    return _nb_secondary_rescale_vec(*args, float(amax)).reshape(shape)


# the closed-form angle priors and conversions have a single implementation
# for every namespace (numpy via array_api_compat)

def _angle_terms(xp, r2, rho, chi_1, mass_ratio, amax):
    _, (r2, rho, chi_1, mass_ratio), _ = _xp.as_float(xp, r2, rho, chi_1, mass_ratio)
    q2 = mass_ratio**2
    c1, R1, _ = _inplane_xp._spin(xp, chi_1, amax)
    a = (1 + q2) * rho
    d = q2 * r2
    am = c1**2 + (a - d) ** 2
    ap = c1**2 + (a + d) ** 2
    ad2 = 2 * a * d
    pos = ad2 > 0
    cos_c = xp.where(
        pos,
        (a**2 + d**2 - R1**2) / xp.where(pos, ad2, 1.0),
        1 - 2 * xp.astype(a**2 + d**2 <= R1**2, xp.float64),
    )
    psi_c = _xp.acos_clip(xp, cos_c)
    kk = xp.sqrt(ap / am)
    half_norm = xp.atan2(kk * xp.sin(psi_c / 2), xp.cos(psi_c / 2))
    return psi_c, kk, half_norm, am, ap


def angle_cdf_pdf(theta, r2, rho, chi_1, mass_ratio, amax=AMAX, xp=None):
    """CDF and density of the angle of v_2 relative to S, on (-pi, pi]."""
    xp = _xp.namespace(theta, r2, rho, chi_1, mass_ratio, xp=xp)
    shape, (theta, r2, rho, chi_1, mass_ratio), _ = _xp.as_float(
        xp, theta, r2, rho, chi_1, mass_ratio
    )
    psi_c, kk, half_norm, am, ap = _angle_terms(xp, r2, rho, chi_1, mass_ratio, amax)
    inside = xp.abs(theta) <= psi_c
    th = xp.clip(theta, -psi_c, psi_c)
    phase = xp.atan2(kk * xp.sin(th / 2), xp.cos(th / 2))
    positive = half_norm > 0
    cdf = xp.where(
        positive,
        (phase + half_norm) / (2 * xp.where(positive, half_norm, 1.0)),
        xp.astype(theta >= 0, xp.float64),
    )
    # A - B cos(theta) = (A - B) cos^2(theta / 2) + (A + B) sin^2(theta / 2)
    denom = am * xp.cos(th / 2) ** 2 + ap * xp.sin(th / 2) ** 2
    norm = 2 * half_norm * 2 / xp.sqrt(am * ap)
    keep = inside & (norm > 0)
    pdf = xp.where(keep, 1 / (denom * xp.where(keep, norm, 1.0)), 0.0)
    return xp.reshape(xp.clip(cdf, 0.0, 1.0), shape), xp.reshape(pdf, shape)


def angle_rescale(uu, r2, rho, chi_1, mass_ratio, amax=AMAX, xp=None):
    """Inverse CDF of the angle of v_2 relative to S."""
    xp = _xp.namespace(uu, r2, rho, chi_1, mass_ratio, xp=xp)
    shape, (uu, r2, rho, chi_1, mass_ratio), _ = _xp.as_float(xp, uu, r2, rho, chi_1, mass_ratio)
    psi_c, kk, half_norm, *_ = _angle_terms(xp, r2, rho, chi_1, mass_ratio, amax)
    phase = (2 * uu - 1) * half_norm
    return xp.reshape(2 * xp.atan2(xp.sin(phase), kk * xp.cos(phase)), shape)


def in_plane_from_total(rho, r2, theta, mass_ratio, xp=None):
    """Return (chi_1_in_plane, phi_12) from |S|, |v_2| and the angle of v_2 relative to S."""
    xp = _xp.namespace(rho, r2, theta, mass_ratio, xp=xp)
    shape, (rho, r2, theta, mass_ratio), _ = _xp.as_float(xp, rho, r2, theta, mass_ratio)
    q2 = mass_ratio**2
    a = (1 + q2) * rho
    d = q2 * r2
    sin_half = xp.sin(theta / 2)
    chi_1_in_plane = xp.sqrt((a - d) ** 2 + 4 * a * d * sin_half**2)
    phi_1 = xp.atan2(-d * xp.sin(theta), a - d * xp.cos(theta))
    phi_12 = xp.remainder(theta - phi_1, 2 * math.pi)
    return xp.reshape(chi_1_in_plane, shape), xp.reshape(phi_12, shape)


def total_from_in_plane(chi_1_in_plane, chi_2_in_plane, phi_12, mass_ratio, xp=None):
    """Return (|S|, |v_2|, angle of v_2 relative to S) from the standard in-plane parameters."""
    xp = _xp.namespace(chi_1_in_plane, chi_2_in_plane, phi_12, mass_ratio, xp=xp)
    shape, (chi_1_in_plane, chi_2_in_plane, phi_12, mass_ratio), _ = _xp.as_float(
        xp, chi_1_in_plane, chi_2_in_plane, phi_12, mass_ratio
    )
    q2 = mass_ratio**2
    sx = (chi_1_in_plane + q2 * chi_2_in_plane * xp.cos(phi_12)) / (1 + q2)
    sy = q2 * chi_2_in_plane * xp.sin(phi_12) / (1 + q2)
    rho = xp.hypot(sx, sy)
    delta = phi_12 - xp.atan2(sy, sx)
    theta = xp.atan2(xp.sin(delta), xp.cos(delta))
    return xp.reshape(rho, shape), xp.reshape(chi_2_in_plane, shape), xp.reshape(theta, shape)


# ---------------------------------------------------------------------------
# bilby prior classes
# ---------------------------------------------------------------------------

def _shape_output(val, out):
    """Return numpy scalars for scalar input, other namespaces are returned as is."""
    if isinstance(out, np.ndarray) and np.ndim(val) == 0 and out.size == 1:
        return out.reshape(())[()]
    return out


class _TotalBase(ConditionalBasePrior):
    """Shared boilerplate for the conditional priors in this module."""

    _required = []

    def __init__(self, amax, minimum, maximum, name, latex_label=None, unit=None, boundary=None):
        class_name = type(self).__name__
        super().__init__(
            minimum=minimum, maximum=maximum, name=name, latex_label=latex_label,
            unit=unit, boundary=boundary, condition_func=self._condition_function,
        )
        self.amax = amax
        self._required_variables = list(self._required)
        self.__class__.__name__ = class_name
        self.__class__.__qualname__ = class_name

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(amax={self.amax}, name={self.name!r}, "
            f"latex_label={self.latex_label!r}, unit={self.unit!r}, "
            f"boundary={self.boundary!r})"
        )

    def get_instantiation_dict(self):
        instantiation_dict = Prior.get_instantiation_dict(self)
        for key in ["minimum", "maximum", "condition_func"]:
            instantiation_dict.pop(key, None)
        instantiation_dict["amax"] = self.amax
        return instantiation_dict

    def _condition_function(self, reference_params, **required_variables):
        return dict()

    # the functional interface, set by subclasses
    _rescale = None
    _cdf_pdf = None

    def _call(self, func, val, xp, required_variables):
        values = [required_variables[key] for key in self._required]
        return func(val, *values, amax=self.amax, xp=xp)

    @xp_wrap
    def rescale(self, val, *, xp=None, **required_variables):
        return _shape_output(val, self._call(self._rescale, val, xp, required_variables))

    @xp_wrap
    def cdf(self, val, *, xp=None, **required_variables):
        return _shape_output(val, self._call(self._cdf_pdf, val, xp, required_variables)[0])

    @xp_wrap
    def prob(self, val, *, xp=None, **required_variables):
        return _shape_output(val, self._call(self._cdf_pdf, val, xp, required_variables)[1])

    @xp_wrap
    def ln_prob(self, val, *, xp=None, **required_variables):
        prob = self.prob(val, xp=xp, **required_variables)
        xp = _xp.namespace(prob)
        if _xp.is_numpy(xp):
            with np.errstate(divide="ignore"):
                return np.log(prob)
        return xp.log(prob)


class ChiInPlaneTotal(_TotalBase):
    r"""
    Prior on :math:`|S|`, the magnitude of the mass-squared-weighted total
    in-plane spin :math:`S = (v_1 + q^2 v_2) / (1 + q^2)`, given the aligned
    spins and mass ratio, implied by uniform spin magnitudes and isotropic
    orientations.
    """

    _required = ["chi_1", "chi_2", "mass_ratio"]
    _rescale = staticmethod(total_rescale)
    _cdf_pdf = staticmethod(total_cdf_pdf)

    def __init__(self, amax=AMAX, name="chi_in_plane_total", latex_label=r"$\chi_{\perp}$",
                 unit=None, boundary=None):
        super().__init__(amax, 0, amax, name, latex_label, unit, boundary)


class ChiInPlane2GivenTotal(_TotalBase):
    r"""
    Prior on the secondary in-plane spin magnitude :math:`|v_2|` given
    :math:`|S|`, the aligned spins and mass ratio.
    """

    _required = ["chi_in_plane_total", "chi_1", "chi_2", "mass_ratio"]
    _rescale = staticmethod(secondary_rescale)
    _cdf_pdf = staticmethod(secondary_cdf_pdf)

    def __init__(self, amax=AMAX, name="chi_2_in_plane", latex_label=r"$\chi_{2\perp}$",
                 unit=None, boundary=None):
        super().__init__(amax, 0, amax, name, latex_label, unit, boundary)


class PhiS2(_TotalBase):
    r"""
    Prior on the angle of the secondary in-plane spin relative to the total
    in-plane spin :math:`S`, on :math:`(-\pi, \pi]`.

    .. math::

        p(\theta) \propto \left(\chi_1^2 + |(1 + q^2) S - q^2 v_2|^2\right)^{-1}

    on the arc where :math:`|v_1| \leq \sqrt{a_{\max}^2 - \chi_1^2}`.
    """

    _required = ["chi_2_in_plane", "chi_in_plane_total", "chi_1", "mass_ratio"]
    _rescale = staticmethod(angle_rescale)
    _cdf_pdf = staticmethod(angle_cdf_pdf)

    def __init__(self, amax=AMAX, name="phi_s2", latex_label=r"$\phi_{S2}$",
                 unit=None, boundary=None):
        super().__init__(amax, -np.pi, np.pi, name, latex_label, unit, boundary)


class _InPlaneDelta(ConditionalDeltaFunction):
    """Conditional delta functions for chi_1_in_plane and phi_12."""

    _index = 0

    def __init__(self, name, latex_label=None, unit=None, boundary=None):
        class_name = type(self).__name__
        super().__init__(
            name=name, latex_label=latex_label, unit=unit, boundary=boundary,
            condition_func=self._condition_function, peak=0.1,
        )
        self._required_variables = ["chi_in_plane_total", "chi_2_in_plane", "phi_s2", "mass_ratio"]
        self.__class__.__name__ = class_name
        self.__class__.__qualname__ = class_name

    is_fixed = False

    def get_instantiation_dict(self):
        instantiation_dict = Prior.get_instantiation_dict(self)
        for key in ["peak", "condition_func"]:
            instantiation_dict.pop(key, None)
        return instantiation_dict

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(name={self.name!r}, "
            f"latex_label={self.latex_label!r}, unit={self.unit!r})"
        )

    def _condition_function(self, reference_params, **required_variables):
        values = in_plane_from_total(
            required_variables["chi_in_plane_total"],
            required_variables["chi_2_in_plane"],
            required_variables["phi_s2"],
            required_variables["mass_ratio"],
        )
        return dict(peak=_shape_output(required_variables["chi_2_in_plane"], values[self._index]))


class ChiInPlane1FromTotal(_InPlaneDelta):
    """chi_1_in_plane = |(1 + q^2) S - q^2 v_2| as a conditional delta function."""

    _index = 0

    def __init__(self, name="chi_1_in_plane", latex_label=r"$\chi_{1\perp}$", unit=None, boundary=None):
        super().__init__(name=name, latex_label=latex_label, unit=unit, boundary=boundary)


class Phi12FromTotal(_InPlaneDelta):
    """phi_12, the azimuth of v_2 relative to v_1, as a conditional delta function."""

    _index = 1

    def __init__(self, name="phi_12", latex_label=r"$\phi_{12}$", unit=None, boundary=None):
        super().__init__(name=name, latex_label=latex_label, unit=unit, boundary=boundary)


def total_in_plane_spin_priors(amax=AMAX):
    """
    The priors replacing chi_1_in_plane, chi_2_in_plane and phi_12.
    phi_jl stays uniform and plays the role of the azimuth of S.
    """
    return dict(
        chi_in_plane_total=ChiInPlaneTotal(amax),
        chi_2_in_plane=ChiInPlane2GivenTotal(amax),
        phi_s2=PhiS2(amax),
        chi_1_in_plane=ChiInPlane1FromTotal(),
        phi_12=Phi12FromTotal(),
    )
