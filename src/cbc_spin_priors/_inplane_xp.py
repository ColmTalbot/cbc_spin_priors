"""
Generic array API implementation of the total in-plane spin CDFs and inverse
CDFs in :mod:`cbc_spin_priors.inplane`.

This is an element-wise transcription of the numba scalar kernels: inputs are
flattened to a batch axis and the six half segments, the outer quadrature
nodes and the inner quadrature nodes are trailing axes. Branches are replaced
by :code:`xp.where`, the data-dependent subsets of the numpy implementation are
replaced by evaluating everything and masking, and the Newton iterations use
:func:`cbc_spin_priors._xp.newton_bisect`.
"""

import math

import numpy as np

from ._xp import acos_clip, as_float, newton_bisect

PI = math.pi
INF = math.inf


def _nodes(xp, device):
    # imported here to avoid a circular import
    from .inplane import _W_INNER, _W_OUTER, _X_INNER, _X_OUTER

    return [
        xp.asarray(np.asarray(arr), dtype=xp.float64, device=device)
        for arr in (_X_OUTER, _W_OUTER, _X_INNER, _W_INNER)
    ]


def _e(xx):
    return xx[..., None]


def _spin(xp, chi, amax):
    from .inplane import CMIN

    c = xp.clip(xp.abs(chi), CMIN, amax * (1 - 1e-10))
    return c, xp.sqrt((amax - c) * (amax + c)), xp.log(amax / c)


def _r_to_t(xp, r, c, L):
    return xp.log1p((r / c) ** 2) / (2 * L)


def _t_to_r(xp, t, c, L):
    return c * xp.sqrt(xp.expm1(2 * L * t))


def _dgda(xp, a, d, c, R, L):
    am = c * c + (a - d) ** 2
    ap = c * c + (a + d) ** 2
    ad2 = 2 * a * d
    pos = ad2 > 0
    cos_c = xp.where(
        pos,
        (a * a + d * d - R * R) / xp.where(pos, ad2, 1.0),
        1 - 2 * xp.astype(a * a + d * d <= R * R, xp.float64),
    )
    half = acos_clip(xp, cos_c) / 2
    sqm = xp.sqrt(am)
    sqp = xp.sqrt(ap)
    J = 2 / (sqm * sqp) * xp.atan2(sqp * xp.sin(half), sqm * xp.cos(half))
    return a * J / (PI * L)


def _disk(xp, a, d, c, R, L, x_inner, w_inner):
    a2 = a * a
    d2 = d * d
    c2 = c * c
    xx = a2 + c2 - d2
    zz = c2 + d2 - a2
    sq = xp.sqrt(xx * xx + 4 * c2 * d2)
    pos = zz > 0
    mm = xp.where(pos, 2 * a2 / xp.where(pos, sq + zz, 1.0), (sq - zz) / (2 * c2))
    out = xp.log1p(mm) / (2 * L)

    # the truncated part, evaluated everywhere and only used where the disk
    # crosses |v_1| = R
    mid = a2 + d2
    half = 2 * a * d
    pos = half > 0
    cos0 = xp.where(pos, (mid - R * R) / xp.where(pos, half, 1.0), -1.0)
    tau0 = acos_clip(xp, cos0)
    width = PI - tau0
    tau = _e(tau0) + _e(width) * x_inner
    ss = _e(mid) - _e(half) * xp.cos(tau)
    pos = ss > 0
    xarg = xp.where(pos, (_e(d) - _e(a) * xp.cos(tau)) / xp.sqrt(xp.where(pos, ss, 1.0)), 1.0)
    phi = acos_clip(xp, xarg) / PI
    trunc = xp.sum(w_inner * _e(half) * xp.sin(tau) / (_e(c2) + ss) * phi, axis=-1)
    out = out - xp.where(a + d > R, width * trunc / (2 * L), 0.0)
    return xp.clip(out, 0.0, 1.0)


def _knots(xp, rho, c1, R1, L1, c2, R2, L2, q2):
    a = (1 + q2) * rho
    d_lo = xp.clip(a - R1, min=0.0)
    d_hi = xp.minimum(q2 * R2, a + R1)
    t_lo = _r_to_t(xp, xp.clip(d_lo / q2, 0.0, R2), c2, L2)
    t_hi = xp.maximum(_r_to_t(xp, xp.clip(d_hi / q2, 0.0, R2), c2, L2), t_lo)
    t_k = _r_to_t(xp, xp.clip(xp.clip(R1 - a, min=0.0) / q2, 0.0, R2), c2, L2)
    t_k = xp.clip(t_k, t_lo, t_hi)
    r_a = a / q2
    t_a_raw = _r_to_t(xp, r_a, c2, L2)
    t_a = xp.clip(t_a_raw, t_lo, t_hi)
    ex = c1 / q2 * r_a / (L2 * (c2 * c2 + r_a * r_a))
    ey = t_a_raw - t_a
    # hypot has a NaN gradient at the origin (rho = 0)
    nonzero = (ex != 0) | (ey != 0)
    eps_a = xp.where(nonzero, xp.hypot(xp.where(nonzero, ex, 1.0), ey), 0.0)
    swap = t_k < t_a
    k1 = xp.where(swap, t_k, t_a)
    k2 = xp.where(swap, t_a, t_k)
    e1 = xp.where(swap, INF, eps_a)
    e2 = xp.where(swap, eps_a, INF)
    inf = xp.full_like(t_lo, INF)
    m0 = (t_lo + k1) / 2
    m1 = (k1 + k2) / 2
    m2 = (k2 + t_hi) / 2
    anchor = xp.stack([t_lo, k1, k1, k2, k2, t_hi], axis=-1)
    end = xp.stack([m0, m0, m1, m1, m2, m2], axis=-1)
    eps = xp.stack([inf, e1, e1, e2, e2, inf], axis=-1)
    return a, t_lo, t_hi, anchor, end, eps


def _map(xp, anchor, end, eps):
    span = xp.abs(end - anchor)
    sgn = 2 * xp.astype(end >= anchor, xp.float64) - 1
    e = xp.minimum(xp.maximum(eps, 1e-14 * span), 1e3 * span + 1e-300)
    # e = 0 only for empty half segments (span = 0), which are masked by the callers
    return sgn, span, e, xp.asinh(span / xp.where(span > 0, e, 1.0))


def _constants(xp, chi_1, chi_2, mass_ratio, amax):
    c1, R1, L1 = _spin(xp, chi_1, amax)
    c2, R2, L2 = _spin(xp, chi_2, amax)
    q2 = mass_ratio * mass_ratio
    rho_max = (R1 + q2 * R2) / (1 + q2)
    return c1, R1, L1, c2, R2, L2, q2, rho_max


def _total_cdf_pdf_core(xp, rho, c1, R1, L1, c2, R2, L2, q2, nodes):
    x_outer, w_outer, x_inner, w_inner = nodes
    a, t_lo, t_hi, anchor, end, eps = _knots(xp, rho, c1, R1, L1, c2, R2, L2, q2)
    sgn, span, e, Y = _map(xp, anchor, end, eps)
    # (batch, half segment, outer node)
    y = _e(Y) * x_outer * x_outer
    tt = _e(anchor) + _e(sgn * e) * xp.sinh(y)
    ww = xp.where(_e(span > 0), w_outer * _e(e * 2 * Y) * x_outer * xp.cosh(y), 0.0)
    a, c1, R1, L1, c2, L2, q2_ = [xx[:, None, None] for xx in (a, c1, R1, L1, c2, L2, q2)]
    d = q2_ * _t_to_r(xp, tt, c2, L2)
    cdf = t_lo + xp.sum(ww * _disk(xp, a, d, c1, R1, L1, x_inner, w_inner), axis=(-1, -2))
    pdf = (1 + q2) * xp.sum(ww * _dgda(xp, a, d, c1, R1, L1), axis=(-1, -2))
    return xp.clip(cdf, 0.0, 1.0), pdf


def total_cdf_pdf(xp, rho, chi_1, chi_2, mass_ratio, amax):
    shape, (rho, chi_1, chi_2, mass_ratio), device = as_float(xp, rho, chi_1, chi_2, mass_ratio)
    c1, R1, L1, c2, R2, L2, q2, rho_max = _constants(xp, chi_1, chi_2, mass_ratio, amax)
    cdf, pdf = _total_cdf_pdf_core(
        xp, xp.clip(rho, 0.0, rho_max), c1, R1, L1, c2, R2, L2, q2, _nodes(xp, device)
    )
    pdf = xp.where((rho < 0) | (rho > rho_max), 0.0, pdf)
    return xp.reshape(cdf, shape), xp.reshape(pdf, shape)


def total_rescale(xp, uu, chi_1, chi_2, mass_ratio, amax, maxiter=100):
    shape, (uu, chi_1, chi_2, mass_ratio), device = as_float(xp, uu, chi_1, chi_2, mass_ratio)
    c1, R1, L1, c2, R2, L2, q2, rho_max = _constants(xp, chi_1, chi_2, mass_ratio, amax)
    nodes = _nodes(xp, device)
    uu = xp.clip(uu, 1e-300, 1 - 1e-16)
    target = xp.log(uu) - xp.log1p(-uu)

    # Newton iteration for logit(F) in terms of logit(rho / rho_max)
    def func(zz):
        frac = 1 / (1 + xp.exp(-zz))
        rho = rho_max * frac
        cdf, pdf = _total_cdf_pdf_core(xp, rho, c1, R1, L1, c2, R2, L2, q2, nodes)
        cdf = xp.clip(cdf, 1e-300, 1 - 1e-16)
        resid = xp.log(cdf) - xp.log1p(-cdf) - target
        return resid, resid * cdf * (1 - cdf) / (pdf * rho * (1 - frac))

    zz = newton_bisect(
        xp, func, xp.clip(target / 2, -300.0, 40.0),
        xp.full_like(uu, -700.0), xp.full_like(uu, 40.0),
        tol=1e-10, xrtol=1e-14, xfloor=1e-300, maxiter=maxiter,
    )
    rho = xp.minimum(rho_max / (1 + xp.exp(-zz)), rho_max)
    return xp.reshape(rho, shape)


def _half_integrals(xp, frac, anchor, end, eps, a, c1, R1, L1, c2, L2, q2, x_outer, w_outer):
    """
    Integral of the t density from the anchor of each half segment over the
    fraction :code:`frac` of it. The geometry has a trailing half segment
    axis, the per-point constants have shape (batch, 1).
    """
    sgn, span, e, Y = _map(xp, anchor, end, eps)
    valid = (span > 0) & (frac > 0)
    frac = xp.where(valid, frac, 1.0)
    smax = xp.sqrt(xp.asinh(frac * span / e) / xp.where(Y > 0, Y, 1.0))
    ss = _e(smax) * x_outer
    y = _e(Y) * ss * ss
    tt = _e(anchor) + _e(sgn * e) * xp.sinh(y)
    a, c1, R1, L1, c2, L2, q2 = [_e(xx) for xx in (a, c1, R1, L1, c2, L2, q2)]
    d = q2 * _t_to_r(xp, tt, c2, L2)
    integrand = _e(smax) * w_outer * _e(e * 2 * Y) * ss * xp.cosh(y) * _dgda(xp, a, d, c1, R1, L1)
    return xp.where(valid, xp.sum(integrand, axis=-1), 0.0)


def _secondary_setup(xp, rho, chi_1, chi_2, mass_ratio, amax):
    c1, R1, L1, c2, R2, L2, q2, rho_max = _constants(xp, chi_1, chi_2, mass_ratio, amax)
    rho = xp.clip(rho, 0.0, rho_max)
    a, t_lo, t_hi, anchor, end, eps = _knots(xp, rho, c1, R1, L1, c2, R2, L2, q2)
    return c1, R1, L1, c2, R2, L2, q2, a, t_lo, t_hi, anchor, end, eps


def secondary_cdf_pdf(xp, r2, rho, chi_1, chi_2, mass_ratio, amax):
    shape, (r2, rho, chi_1, chi_2, mass_ratio), device = as_float(
        xp, r2, rho, chi_1, chi_2, mass_ratio
    )
    x_outer, w_outer, _, _ = _nodes(xp, device)
    c1, R1, L1, c2, R2, L2, q2, a, t_lo, t_hi, anchor, end, eps = _secondary_setup(
        xp, rho, chi_1, chi_2, mass_ratio, amax
    )
    consts = [_e(xx) for xx in (a, c1, R1, L1, c2, L2, q2)]

    def integrals(frac):
        return _half_integrals(xp, frac, anchor, end, eps, *consts, x_outer, w_outer)

    full = integrals(xp.ones_like(anchor))
    total = xp.sum(full, axis=-1)
    t_raw = _r_to_t(xp, r2, c2, L2)
    t_target = _e(xp.clip(t_raw, t_lo, t_hi))
    lower = xp.minimum(anchor, end)
    upper = xp.maximum(anchor, end)
    span = upper - lower
    safe_span = xp.where(span > 0, span, 1.0)
    # halves 0, 2, 4 are anchored at their lower end, 1, 3, 5 at their upper end
    from_lower = xp.asarray([True, False, True, False, True, False], device=device)
    frac = xp.where(from_lower, t_target - lower, upper - t_target) / safe_span
    partial = integrals(frac)
    partial = xp.where(from_lower, partial, full - partial)
    contribution = xp.where(
        t_target >= upper, full,
        xp.where((t_target > lower) & (span > 0), partial, 0.0),
    )
    cumulative = xp.sum(contribution, axis=-1)
    positive = total > 0
    safe_total = xp.where(positive, total, 1.0)
    cdf = xp.where(
        positive,
        xp.clip(cumulative / safe_total, 0.0, 1.0),
        xp.astype(r2 >= _t_to_r(xp, t_lo, c2, L2), xp.float64),
    )
    inside = (t_lo <= t_raw) & (t_raw <= t_hi) & (r2 <= R2) & positive
    pdf = xp.where(
        inside,
        _dgda(xp, a, q2 * r2, c1, R1, L1) * r2 / (L2 * (c2 * c2 + r2 * r2)) / safe_total,
        0.0,
    )
    return xp.reshape(cdf, shape), xp.reshape(pdf, shape)


def secondary_rescale(xp, uu, rho, chi_1, chi_2, mass_ratio, amax, maxiter=100):
    shape, (uu, rho, chi_1, chi_2, mass_ratio), device = as_float(
        xp, uu, rho, chi_1, chi_2, mass_ratio
    )
    x_outer, w_outer, _, _ = _nodes(xp, device)
    c1, R1, L1, c2, R2, L2, q2, a, t_lo, t_hi, anchor, end, eps = _secondary_setup(
        xp, rho, chi_1, chi_2, mass_ratio, amax
    )
    consts = [_e(xx) for xx in (a, c1, R1, L1, c2, L2, q2)]
    halves = _half_integrals(xp, xp.ones_like(anchor), anchor, end, eps, *consts, x_outer, w_outer)
    total = xp.sum(halves, axis=-1)
    target = uu * total

    # select the first half segment whose cumulative integral reaches the target
    found = target != target
    cumulative = xp.zeros_like(total)
    before = xp.zeros_like(total)
    columns = []
    for h in range(6):
        hit = ~found & (cumulative + halves[:, h] >= target)
        before = xp.where(hit, cumulative, before)
        columns.append(hit)
        found = found | hit
        cumulative = cumulative + halves[:, h]
    columns[5] = columns[5] | ~found
    before = xp.where(found, before, cumulative)
    select = xp.stack(columns, axis=-1)

    def pick(arr):
        return xp.sum(xp.where(select, arr, 0.0), axis=-1)

    from_lower = pick(xp.asarray([1.0, 0.0, 1.0, 0.0, 1.0, 0.0], dtype=xp.float64, device=device)) > 0.5
    this = pick(halves)
    needed = xp.where(from_lower, target - before, before + this - target)
    needed = xp.minimum(xp.clip(needed, min=0.0), this)
    anchor, end, eps = _e(pick(anchor)), _e(pick(end)), _e(pick(eps))
    sgn, span, e, Y = [xx[:, 0] for xx in _map(xp, anchor, end, eps)]
    safe_span = xp.where(span > 0, span, 1.0)
    atol = 1e-13 * xp.clip(total, min=1e-300)
    positive = this > 0
    ss = xp.where(positive, xp.sqrt(xp.where(positive, needed / xp.where(positive, this, 1.0), 0.0)), 0.0)
    ss = xp.clip(ss, 0.0, 1.0)

    def func(ss):
        mapped = e * xp.sinh(Y * ss * ss)
        frac = xp.clip(mapped / safe_span, 0.0, 1.0)
        value = _half_integrals(xp, _e(frac), anchor, end, eps, *consts, x_outer, w_outer)[:, 0]
        resid = value - needed
        tt = anchor[:, 0] + sgn * mapped
        dens = _dgda(xp, a, q2 * _t_to_r(xp, tt, c2, L2), c1, R1, L1)
        deriv = dens * e * 2 * Y * ss * xp.cosh(Y * ss * ss)
        usable = deriv > 0
        return resid, xp.where(usable, resid / xp.where(usable, deriv, 1.0), math.nan)

    ss = newton_bisect(
        xp, func, ss, xp.zeros_like(ss), xp.ones_like(ss),
        tol=atol, xrtol=1e-14, xfloor=1.0, done=~(span > 0), maxiter=maxiter,
    )
    tt = anchor[:, 0] + sgn * e * xp.sinh(Y * ss * ss)
    tt = xp.clip(tt, t_lo, t_hi)
    return xp.reshape(_t_to_r(xp, tt, c2, L2), shape)
