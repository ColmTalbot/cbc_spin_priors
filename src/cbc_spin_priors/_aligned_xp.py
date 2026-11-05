"""
Generic array API implementation of the :code:`ChiEffQ` and :code:`Chi2` CDFs
and inverse CDFs in :mod:`cbc_spin_priors.aligned`, an element-wise
transcription of the numba scalar kernels (see
:mod:`cbc_spin_priors._inplane_xp`).
"""

import math

from ._xp import as_float, newton_bisect, real_dilog


def _safe_log_abs(xp, x):
    """log|x|, with 0 where x == 0."""
    nonzero = x != 0
    return xp.where(nonzero, xp.log(xp.abs(xp.where(nonzero, x, 1.0))), 0.0)


def _chi2_antiderivative(xp, x, beta, log_q, log_amax):
    lx = _safe_log_abs(xp, x)
    xl = x * lx
    pp = xl - x
    k2 = log_q - log_amax
    zero = beta == 0
    hh_zero = xl * lx - 2 * xl + 2 * x
    safe_beta = xp.where(zero, 1.0, beta)
    diff = beta - x
    l2 = _safe_log_abs(xp, diff)
    hh = (
        (pp + beta - beta * xp.log(xp.abs(safe_beta))) * l2
        - xl + 2 * x + beta * real_dilog(xp, 1.0 - x / safe_beta)
    )
    hh = xp.where(zero, hh_zero, hh)
    qq = xp.where(zero, pp, -diff * l2 - x)
    return hh + k2 * pp - log_amax * qq - log_amax * k2 * x


def _chi2_setup(xp, chi_eff, mass_ratio, amax):
    sgn = 2 * xp.astype(chi_eff >= 0, xp.float64) - 1
    alt = xp.clip(((1 + mass_ratio) * xp.abs(chi_eff) - amax) / mass_ratio, min=-amax)
    lower = xp.minimum(sgn * alt, sgn * amax)
    upper = xp.maximum(sgn * alt, sgn * amax)
    beta = (1 + mass_ratio) * chi_eff / mass_ratio
    log_q = xp.log(mass_ratio)
    log_amax = math.log(amax)
    a_lo = _chi2_antiderivative(xp, lower, beta, log_q, log_amax)
    norm = _chi2_antiderivative(xp, upper, beta, log_q, log_amax) - a_lo
    return lower, upper, beta, log_q, log_amax, a_lo, norm


def _chi2_unnormalised_pdf(xp, x, chi_eff, mass_ratio, amax):
    return (
        xp.log(xp.abs(x) / amax)
        * xp.log(xp.abs((1 + mass_ratio) * chi_eff - mass_ratio * x) / amax)
    )


def chi2_cdf_pdf(xp, xx, chi_eff, mass_ratio, amax):
    shape, (xx, chi_eff, mass_ratio), _ = as_float(xp, xx, chi_eff, mass_ratio)
    lower, upper, beta, log_q, log_amax, a_lo, norm = _chi2_setup(xp, chi_eff, mass_ratio, amax)
    inside = (xx > lower) & (xx < upper)
    cdf = (_chi2_antiderivative(xp, xx, beta, log_q, log_amax) - a_lo) / norm
    cdf = xp.where(inside, xp.clip(cdf, 0.0, 1.0), xp.astype(xx >= upper, xp.float64))
    pdf = xp.where(inside, _chi2_unnormalised_pdf(xp, xx, chi_eff, mass_ratio, amax) / norm, 0.0)
    return xp.reshape(cdf, shape), xp.reshape(pdf, shape)


def chi2_rescale(xp, uu, chi_eff, mass_ratio, amax, maxiter=100):
    shape, (uu, chi_eff, mass_ratio), _ = as_float(xp, uu, chi_eff, mass_ratio)
    lower, upper, beta, log_q, log_amax, a_lo, norm = _chi2_setup(xp, chi_eff, mass_ratio, amax)
    target = uu * norm

    def func(xx):
        resid = _chi2_antiderivative(xp, xx, beta, log_q, log_amax) - a_lo - target
        deriv = _chi2_unnormalised_pdf(xp, xx, chi_eff, mass_ratio, amax)
        usable = (deriv > 0) & xp.isfinite(deriv)
        return resid, xp.where(usable, resid / xp.where(usable, deriv, 1.0), math.nan)

    xx = newton_bisect(
        xp, func, lower + uu * (upper - lower), lower, upper,
        tol=1e-14 * xp.abs(norm), xrtol=1e-15, xfloor=1e-300, maxiter=maxiter,
    )
    return xp.reshape(xx, shape)


def _xlogx(xp, x):
    return x * _safe_log_abs(xp, x)


def _chi_eff_antiderivative(xp, z, t, log_q):
    w = z - t
    phi = z * _xlogx(xp, z) / 2 - z * z / 4
    phi_t = t * _xlogx(xp, t) / 2 - t * t / 4
    log_w = _safe_log_abs(xp, w)
    positive = t * t > 0
    safe_t = xp.where(positive, t, 1.0)
    dilog = xp.where(positive, t * t * real_dilog(xp, 1.0 - z / safe_t), 0.0)
    linear = w * _xlogx(xp, w) / 2 - w * w / 4 + t * (_xlogx(xp, w) - w)
    double_log = (
        (phi - phi_t) * log_w
        - (phi + t * (_xlogx(xp, z) - z) - dilog) / 2
        + (z * z / 2 + t * z) / 4
    )
    return (linear - double_log - log_q * (3 * z * z / 4 - z * _xlogx(xp, z) / 2)) / 2


def _chi_eff_half(xp, t, mass_ratio, log_q):
    """
    P(0 < t' < t) and its density, see :code:`aligned._chi_eff_half`.

    Below t = 1e-300 the linear limit is used (numba sets t = 0 there), which
    keeps the derivative at t = 0.
    """
    tiny = t < 1e-300
    slope = 1 - log_q / 2
    t_lin = t
    t = xp.where(tiny, 0.0, t)
    upper = xp.clip(t + mass_ratio, max=1.0)
    lower = t - mass_ratio
    half = _chi_eff_antiderivative(xp, upper, t, log_q) - _chi_eff_antiderivative(xp, lower, t, log_q)
    w_hi = mass_ratio
    w_lo = 1 - t
    half = half + xp.where(
        t + mass_ratio > 1,
        (_xlogx(xp, w_hi) - _xlogx(xp, w_lo) - (w_hi - w_lo) * (1 + log_q)) / 2,
        0.0,
    )
    half = half / (-2 * mass_ratio)
    density = (
        _chi2_antiderivative(xp, upper, t, -log_q, 0.0)
        - _chi2_antiderivative(xp, lower, t, -log_q, 0.0)
    ) / (4 * mass_ratio)
    return xp.where(tiny, t_lin * slope, half), xp.where(tiny, slope, density)


def chi_eff_cdf_pdf(xp, xx, mass_ratio, amax):
    shape, (xx, mass_ratio), _ = as_float(xp, xx, mass_ratio)
    upper = 1 + mass_ratio
    # where rather than abs, so the derivative at chi_eff = 0 is not zero
    positive = xx >= 0
    absx = xp.where(positive, xx, -xx)
    inside = absx < amax
    t = upper * xp.where(inside, absx, 0.0) / amax
    half, density = _chi_eff_half(xp, t, mass_ratio, xp.log(mass_ratio))
    half = xp.where(inside, half, 0.5)
    cdf = xp.where(positive, 0.5 + half, 0.5 - half)
    pdf = xp.where(inside, density * upper / amax, 0.0)
    return xp.reshape(xp.clip(cdf, 0.0, 1.0), shape), xp.reshape(pdf, shape)


def chi_eff_rescale(xp, uu, mass_ratio, amax, maxiter=100):
    shape, (uu, mass_ratio), _ = as_float(xp, uu, mass_ratio)
    # where rather than abs, so the derivative at u = 1/2 is not zero
    centred = uu - 0.5
    positive = centred >= 0
    target = xp.where(positive, centred, -centred)
    log_q = xp.log(mass_ratio)
    upper = 1 + mass_ratio

    def func(tt):
        half, deriv = _chi_eff_half(xp, tt, mass_ratio, log_q)
        resid = half - target
        usable = (deriv > 0) & xp.isfinite(deriv)
        return resid, xp.where(usable, resid / xp.where(usable, deriv, 1.0), math.nan)

    # linear in t near zero, where the density is 1 - ln(q) / 2
    tt = xp.minimum(target / (1 - log_q / 2), upper / 2)
    # the initial guess is exact at u = 1/2, where the iteration stops at once
    trivial = (uu <= 0) | (uu >= 1)
    tt = newton_bisect(
        xp, func, tt, xp.zeros_like(tt), upper,
        tol=1e-16, xrtol=1e-15, xfloor=0.0, done=trivial, maxiter=maxiter, step_rtol=1e-10,
    )
    out = amax * tt / upper
    out = xp.where(positive, out, -out)
    out = xp.where(uu <= 0, -amax, xp.where(uu >= 1, amax, out))
    return xp.reshape(out, shape)
