r"""
Conditional priors for sampling the aligned spins as the effective spin
:math:`\chi_\mathrm{eff}` given the mass ratio, then :math:`\chi_2` given
:math:`(q, \chi_\mathrm{eff})`, with :math:`\chi_1` a conditional delta function.

Together they reproduce the aligned components of spins with uniform magnitude
up to :math:`a_{\max}` and isotropic orientation.
"""

import numpy as np

from bilby.compat.utils import xp_wrap
from bilby.core.prior import (
    ConditionalBasePrior,
    ConditionalDeltaFunction,
    Prior,
)

from . import _aligned_xp, _xp

__all__ = ["ChiEffQ", "Chi1", "Chi2"]

AMAX = 0.99


class ChiEffQ(ConditionalBasePrior):
    r"""
    The conditional prior for chi effective given mass ratio as derived in
    `Callister (2021) <https://arxiv.org/abs/2104.09508>`_ for a fixed
    maximum spin magnitude. The CDF is evaluated in closed form (via the
    dilogarithm) and inverted with a safeguarded Newton iteration.
    :code:`qmin` and :code:`qmax` are metadata only.

    Numpy arrays and Python scalars use compiled numba kernels, other array
    API namespaces use :mod:`cbc_spin_priors._aligned_xp`.

    .. math::

        p(\chi_\mathrm{eff} | q) = \begin{cases}
            (1 + q) (2 - \ln q) / (2 a_{\max}) & \chi_\mathrm{eff} = 0 \\
            ... & |\chi_\mathrm{eff}| < a_{\max} \frac{1 - q}{1 + q}, |\chi_\mathrm{eff}| < q a_{\max} / (1 + q) \\
            ... & |\chi_\mathrm{eff}| < a_{\max} \frac{1 - q}{1 + q}, |\chi_\mathrm{eff}| > q a_{\max} / (1 + q) \\
            ... & |\chi_\mathrm{eff}| > a_{\max} \frac{1 - q}{1 + q}, |\chi_\mathrm{eff}| < q a_{\max} / (1 + q) \\
            ... & |\chi_\mathrm{eff}| > a_{\max} \frac{1 - q}{1 + q}, |\chi_\mathrm{eff}| > q a_{\max} / (1 + q), |\chi_\mathrm{eff}| \leq a_{\max} / (1 + q) \\
            ... & |\chi_\mathrm{eff}| > a_{\max} / (1 + q), |\chi_\mathrm{eff}| < a_{\max} \\
            0 & |\chi_\mathrm{eff}| \geq a_{\max}
        \end{cases}
    """

    def __init__(self, amax, qmin, qmax, name="chi_eff", latex_label=None, unit=None, boundary=None):
        super().__init__(
            minimum=-amax, maximum=amax, name=name, latex_label=latex_label,
            unit=unit, boundary=boundary, condition_func=self._condition_function
        )
        self.amax = amax
        self.qmin = qmin
        self.qmax = qmax
        self._required_variables = ["mass_ratio"]
        self.__class__.__name__ = "ChiEffQ"
        self.__class__.__qualname__ = "ChiEffQ"

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(amax={self.amax}, qmin={self.qmin}, "
            f"qmax={self.qmax}, name={self.name}, latex_label={self.latex_label}, "
            f"unit={self.unit}, boundary={self.boundary})"
        )

    def get_instantiation_dict(self):
        instantiation_dict = Prior.get_instantiation_dict(self)
        instantiation_dict["amax"] = self.amax
        instantiation_dict["qmin"] = self.qmin
        instantiation_dict["qmax"] = self.qmax
        return instantiation_dict

    def _condition_function(self, reference_params, **required_variables):
        return dict(amax=self.amax)

    def _inputs(self, val, required_variables):
        arrays = np.broadcast_arrays(*[
            np.asarray(xx, dtype=float) for xx in (val, required_variables["mass_ratio"])
        ])
        return arrays[0].shape, [np.ascontiguousarray(xx).ravel() for xx in arrays]

    def _generic(self, val, xp, required_variables):
        """The namespace for the generic implementation, or None for numba."""
        return _generic_namespace(xp, val, required_variables["mass_ratio"])

    @xp_wrap
    def cdf(self, val, *, xp=None, **required_variables):
        if (xp := self._generic(val, xp, required_variables)) is not None:
            return _aligned_xp.chi_eff_cdf_pdf(xp, val, required_variables["mass_ratio"], self.amax)[0]
        shape, args = self._inputs(val, required_variables)
        return Chi2._output(val, _chi_eff_cdf_vec(*args, float(self.amax)), shape)

    @xp_wrap
    def rescale(self, val, *, xp=None, **required_variables):
        if (xp := self._generic(val, xp, required_variables)) is not None:
            return _aligned_xp.chi_eff_rescale(xp, val, required_variables["mass_ratio"], self.amax)
        shape, args = self._inputs(val, required_variables)
        return Chi2._output(val, _chi_eff_rescale_vec(*args, float(self.amax)), shape)

    @xp_wrap
    def prob(self, val, *, xp=None, **required_variables):
        # the generic density is the derivative of the closed-form CDF, the
        # numpy one uses scipy's complex dilogarithm, not in the standard
        if (xp := self._generic(val, xp, required_variables)) is not None:
            return _aligned_xp.chi_eff_cdf_pdf(xp, val, required_variables["mass_ratio"], self.amax)[1]
        return self._pdf(val, xp=np, **required_variables)

    @xp_wrap
    def _pdf(self, val, *, xp=np, **required_variables):
        from scipy.special import spence

        q = required_variables.get("mass_ratio", 1.0)
        amax = self.amax
        chi_eff = xp.abs(val)

        # sometimes we have to fight with pandas
        if not hasattr(val, "real"):
            chi_eff = chi_eff.values
            q = q.values

        max_primary = amax / (1 + q)
        max_secondary = q * amax / (1 + q)
        max_difference = amax * (1 - q) / (1 + q)

        # Set up various piecewise cases, these are applied consecutively so lower bounds are implicit
        opposite_signs_allowed = chi_eff < max_difference
        same_sign_required = chi_eff < amax
        zero_chi_eff = chi_eff == 0

        with np.errstate(divide="ignore", invalid="ignore"):
            secondary_ratio = max_secondary / chi_eff
            primary_ratio = max_primary / chi_eff

            lower = (
                (4 - xp.log(q**2) - xp.log(xp.abs(1 / secondary_ratio**2 - 1)))
                + xp.nan_to_num(
                    xp.log(xp.abs(1 - secondary_ratio) / (1 + secondary_ratio))
                    + (spence(1 + secondary_ratio + 0j) - spence(1 - secondary_ratio + 0j)).real
                )
                / secondary_ratio
            ) / (4 * max_primary)

            # these terms diverge on boundaries and so we manually regularize
            primary_term = xp.log(xp.abs(1 / primary_ratio - 1) + (primary_ratio == 1))
            secondary_term = xp.log(
                xp.abs(1 / secondary_ratio - 1) + (secondary_ratio == 1)
            )

            upper = (
                2 * (amax - chi_eff)
                + max_difference * xp.log(q)
                + (chi_eff - max_secondary) * secondary_term
                + (chi_eff - max_primary) * primary_term
                + chi_eff * xp.log(primary_ratio) * (primary_term - xp.log(q))
                + chi_eff * (spence(primary_ratio + 0j) - spence(1 - secondary_ratio + 0j)).real
            ) / (4 * max_primary * max_secondary)

            equal = (1 + q) / (2 * amax) * (2 - xp.log(q))

        return xp.maximum(0.0, xp.select(
            [zero_chi_eff, opposite_signs_allowed, same_sign_required],
            [equal, lower, upper],
            0.0,
        ))

    @xp_wrap
    def ln_prob(self, val, *, xp=None, **required_variables):
        return _log(self.prob(val, xp=xp, **required_variables))


try:
    import numba as _numba
except ImportError:
    _numba = None


def _generic_namespace(xp, *args):
    """The namespace for the generic implementation, or None for numpy/numba."""
    xp = _xp.namespace(*args, xp=xp)
    return None if _xp.is_numpy(xp) else xp


def _log(prob):
    xp = _xp.namespace(prob)
    if _xp.is_numpy(xp):
        with np.errstate(divide="ignore"):
            return np.log(prob)
    return xp.log(prob)


def _maybe_njit(func):
    if _numba is None:
        return func
    return _numba.njit(cache=True, error_model="numpy")(func)


# Cephes coefficients for the dilogarithm, shared with the generic version
_SPENCE_A = _xp._SPENCE_A
_SPENCE_B = _xp._SPENCE_B


@_maybe_njit
def _spence(x):
    """Li2(1 - x) for x >= 0, matches scipy.special.spence (Cephes)."""
    pi2 = np.pi**2 / 6
    if x == 1.0:
        return 0.0
    if x == 0.0:
        return pi2
    flag = 0
    if x > 2.0:
        x = 1.0 / x
        flag |= 2
    if x > 1.5:
        w = 1.0 / x - 1.0
        flag |= 2
    elif x < 0.5:
        w = -x
        flag |= 1
    else:
        w = x - 1.0
    pa = 0.0
    for c in _SPENCE_A:
        pa = pa * w + c
    pb = 0.0
    for c in _SPENCE_B:
        pb = pb * w + c
    y = -w * pa / pb
    if flag & 1:
        y = pi2 - np.log(x) * np.log(1.0 - x) - y
    if flag & 2:
        z = np.log(x)
        y = -0.5 * z * z - y
    return y


@_maybe_njit
def _real_dilog(z):
    """Real part of Li2(z) for real z."""
    if z <= 1.0:
        return _spence(1.0 - z)
    lz = np.log(z)
    return np.pi**2 / 3 - 0.5 * lz * lz - _spence(1.0 - 1.0 / z)


@_maybe_njit
def _chi2_antiderivative(x, beta, log_q, log_amax):
    r"""
    Antiderivative of :math:`\ln(|x| / a) \ln(q |\beta - x| / a)` with
    :math:`\beta = (1 + q) \chi_\mathrm{eff} / q`, using
    :math:`\int \ln|t| / (1 - t) dt = \Re\,\mathrm{Li}_2(1 - t)`.
    """
    xl = x * np.log(abs(x)) if x != 0 else 0.0
    pp = xl - x
    k2 = log_q - log_amax
    if beta == 0:
        lx = np.log(abs(x)) if x != 0 else 0.0
        hh = xl * lx - 2 * xl + 2 * x
        qq = pp
    else:
        diff = beta - x
        l2 = np.log(abs(diff)) if diff != 0 else 0.0
        hh = (
            (pp + beta - beta * np.log(abs(beta))) * l2
            - xl + 2 * x + beta * _real_dilog(1.0 - x / beta)
        )
        qq = -diff * l2 - x
    return hh + k2 * pp - log_amax * qq - log_amax * k2 * x


@_maybe_njit
def _xlogx(x):
    return x * np.log(abs(x)) if x != 0 else 0.0


@_maybe_njit
def _chi_eff_antiderivative(z, t, log_q):
    r"""
    Antiderivative of :math:`(\ln|z - t| - \ln q) z (1 - \ln|z|) / 2` for
    :math:`|z| \leq 1`, :math:`t \geq 0`. The coefficient of :math:`\ln|z - t|`
    vanishes at :math:`z = t`, so the logarithmic singularity is removable.
    """
    w = z - t
    phi = z * _xlogx(z) / 2 - z * z / 4
    phi_t = t * _xlogx(t) / 2 - t * t / 4
    log_w = np.log(abs(w)) if w != 0 else 0.0
    dilog = t * t * _real_dilog(1.0 - z / t) if t * t > 0 else 0.0
    linear = w * _xlogx(w) / 2 - w * w / 4 + t * (_xlogx(w) - w)
    double_log = (
        (phi - phi_t) * log_w
        - (phi + t * (_xlogx(z) - z) - dilog) / 2
        + (z * z / 2 + t * z) / 4
    )
    return (linear - double_log - log_q * (3 * z * z / 4 - z * _xlogx(z) / 2)) / 2


@_maybe_njit
def _chi_eff_half(t, mass_ratio, log_q):
    r"""
    :math:`P(0 < t' < t)` and its density for :math:`t = (1 + q)\chi_\mathrm{eff} / a_{\max} \geq 0`.

    With :math:`x = \chi_1 / a_{\max}`, :math:`y = \chi_2 / a_{\max}` each with
    density :math:`f(x) = -\ln|x| / 2` and CDF :math:`1/2 + G(x)`, this is
    :math:`\int f(y) G(t - q y) dy`, which after substituting :math:`z = t - qy`
    integrates in closed form over :math:`|z| \leq 1` (where
    :math:`G = z (1 - \ln|z|) / 2`) and :math:`z > 1` (where :math:`G = 1/2`).
    """
    if t < 1e-300:
        t = 0.0
    upper = min(t + mass_ratio, 1.0)
    lower = t - mass_ratio
    half = _chi_eff_antiderivative(upper, t, log_q) - _chi_eff_antiderivative(lower, t, log_q)
    if t + mass_ratio > 1:
        # antiderivative of (ln|z - t| - ln q) / 2 between z = 1 and t + q
        w_hi = mass_ratio
        w_lo = 1 - t
        half += (
            _xlogx(w_hi) - _xlogx(w_lo) - (w_hi - w_lo) * (1 + log_q)
        ) / 2
    half /= -2 * mass_ratio
    # the density is (1 / 4q) int ln|z| ln(|t - z| / q) dz over the same range
    density = (
        _chi2_antiderivative(upper, t, -log_q, 0.0)
        - _chi2_antiderivative(lower, t, -log_q, 0.0)
    ) / (4 * mass_ratio)
    return half, density


@_maybe_njit
def _chi_eff_cdf_scalar(x, mass_ratio, amax):
    upper = 1 + mass_ratio
    t = upper * abs(x) / amax
    if t == 0:
        return 0.5
    if abs(x) >= amax:
        half = 0.5
    else:
        half = _chi_eff_half(t, mass_ratio, np.log(mass_ratio))[0]
    cdf = 0.5 + half if x > 0 else 0.5 - half
    return min(max(cdf, 0.0), 1.0)


@_maybe_njit
def _chi_eff_rescale_scalar(u, mass_ratio, amax):
    if u <= 0:
        return -amax
    if u >= 1:
        return amax
    target = abs(u - 0.5)
    if target == 0:
        return 0.0
    log_q = np.log(mass_ratio)
    upper = 1 + mass_ratio
    lo, hi = 0.0, upper
    # linear in t near zero, where the density is 1 - ln(q) / 2
    tt = min(target / (1 - log_q / 2), upper / 2)
    previous = np.inf
    for _ in range(100):
        half, deriv = _chi_eff_half(tt, mass_ratio, log_q)
        resid = half - target
        if abs(resid) <= 1e-16:
            break
        if resid < 0:
            lo = tt
        else:
            hi = tt
        if hi - lo <= 1e-15 * hi:
            break
        new = tt - resid / deriv if deriv > 0 and np.isfinite(deriv) else lo - 1.0
        if not (lo < new < hi) or abs(resid) > 0.5 * previous:
            new = (lo + hi) / 2
        elif abs(new - tt) <= 1e-10 * new:
            # quadratic convergence, the remaining error is far below rounding
            tt = new
            break
        previous = abs(resid)
        tt = new
    out = amax * tt / upper
    return out if u > 0.5 else -out


@_maybe_njit
def _chi_eff_cdf_vec(xx, mass_ratio, amax):
    out = np.empty(xx.shape)
    for i in range(xx.size):
        out[i] = _chi_eff_cdf_scalar(xx[i], mass_ratio[i], amax)
    return out


@_maybe_njit
def _chi_eff_rescale_vec(uu, mass_ratio, amax):
    out = np.empty(uu.shape)
    for i in range(uu.size):
        out[i] = _chi_eff_rescale_scalar(uu[i], mass_ratio[i], amax)
    return out


@_maybe_njit
def _chi2_setup(chi_eff, mass_ratio, amax):
    sgn = 1.0 if chi_eff >= 0 else -1.0
    alt = max(((1 + mass_ratio) * abs(chi_eff) - amax) / mass_ratio, -amax)
    lower = min(sgn * alt, sgn * amax)
    upper = max(sgn * alt, sgn * amax)
    beta = (1 + mass_ratio) * chi_eff / mass_ratio
    log_q = np.log(mass_ratio)
    log_amax = np.log(amax)
    a_lo = _chi2_antiderivative(lower, beta, log_q, log_amax)
    norm = _chi2_antiderivative(upper, beta, log_q, log_amax) - a_lo
    return lower, upper, beta, log_q, log_amax, a_lo, norm


@_maybe_njit
def _chi2_unnormalised_pdf(x, chi_eff, mass_ratio, amax):
    return (
        np.log(abs(x) / amax)
        * np.log(abs((1 + mass_ratio) * chi_eff - mass_ratio * x) / amax)
    )


@_maybe_njit
def _chi2_cdf_pdf_scalar(x, chi_eff, mass_ratio, amax):
    lower, upper, beta, log_q, log_amax, a_lo, norm = _chi2_setup(chi_eff, mass_ratio, amax)
    if x <= lower:
        return 0.0, 0.0
    if x >= upper:
        return 1.0, 0.0
    cdf = (_chi2_antiderivative(x, beta, log_q, log_amax) - a_lo) / norm
    pdf = _chi2_unnormalised_pdf(x, chi_eff, mass_ratio, amax) / norm
    return min(max(cdf, 0.0), 1.0), pdf


@_maybe_njit
def _chi2_rescale_scalar(u, chi_eff, mass_ratio, amax):
    lower, upper, beta, log_q, log_amax, a_lo, norm = _chi2_setup(chi_eff, mass_ratio, amax)
    target = u * norm
    lo, hi = lower, upper
    xx = lower + u * (upper - lower)
    previous = np.inf
    for _ in range(100):
        resid = _chi2_antiderivative(xx, beta, log_q, log_amax) - a_lo - target
        if abs(resid) <= 1e-14 * abs(norm):
            break
        if resid < 0:
            lo = xx
        else:
            hi = xx
        if hi - lo <= 1e-15 * max(abs(hi), 1e-300):
            break
        deriv = _chi2_unnormalised_pdf(xx, chi_eff, mass_ratio, amax)
        new = xx - resid / deriv if deriv > 0 and np.isfinite(deriv) else lo - 1.0
        if not (lo < new < hi) or abs(resid) > 0.5 * previous:
            new = (lo + hi) / 2
        previous = abs(resid)
        xx = new
    return xx


@_maybe_njit
def _chi2_cdf_pdf_vec(xx, chi_eff, mass_ratio, amax):
    cdf = np.empty(xx.shape)
    pdf = np.empty(xx.shape)
    for i in range(xx.size):
        cdf[i], pdf[i] = _chi2_cdf_pdf_scalar(xx[i], chi_eff[i], mass_ratio[i], amax)
    return cdf, pdf


@_maybe_njit
def _chi2_rescale_vec(uu, chi_eff, mass_ratio, amax):
    out = np.empty(uu.shape)
    for i in range(uu.size):
        out[i] = _chi2_rescale_scalar(uu[i], chi_eff[i], mass_ratio[i], amax)
    return out


class Chi2(ConditionalBasePrior):
    r"""
    The conditional prior for the aligned spin of the less massive object
    given :math:`\chi_\mathrm{eff}` and :math:`q`, with the closed-form CDF
    (via the dilogarithm) and a Newton inverse. :code:`prob` is normalised.

    Numpy arrays and Python scalars use compiled numba kernels, other array
    API namespaces use :mod:`cbc_spin_priors._aligned_xp`.

    .. math::

        p(\chi_2 | q, \chi_\mathrm{eff}) \propto
        \ln\left(\frac{|\chi_2|}{a_{\max}}\right)
        \ln\left(\frac{|(1 + q)\chi_{\mathrm{eff}} - q\chi_2|}{a_{\max}}\right)
    """

    def __init__(self, amax, name="chi_2", latex_label=None, unit=None, boundary=None):
        super().__init__(
            minimum=-amax, maximum=amax, name=name, latex_label=latex_label,
            unit=unit, boundary=boundary, condition_func=self._condition_function,
        )
        self.amax = amax
        self._required_variables = ["mass_ratio", "chi_eff"]
        self.__class__.__name__ = "Chi2"
        self.__class__.__qualname__ = "Chi2"

    def get_instantiation_dict(self):
        instantiation_dict = Prior.get_instantiation_dict(self)
        instantiation_dict["amax"] = self.amax
        return instantiation_dict

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(amax={self.amax}, name={self.name}, "
            f"latex_label={self.latex_label}, unit={self.unit}, "
            f"boundary={self.boundary})"
        )

    def _condition_function(self, reference_params, **required_variables):
        return dict()

    def _inputs(self, val, required_variables):
        arrays = np.broadcast_arrays(*[
            np.asarray(xx, dtype=float)
            for xx in (val, required_variables["chi_eff"], required_variables["mass_ratio"])
        ])
        return arrays[0].shape, [np.ascontiguousarray(xx).ravel() for xx in arrays]

    @staticmethod
    def _output(val, out, shape):
        out = out.reshape(shape)
        return out[()] if np.ndim(val) == 0 and out.size == 1 else out

    def _generic(self, val, xp, required_variables):
        """The namespace for the generic implementation, or None for numba."""
        return _generic_namespace(xp, val, required_variables["chi_eff"], required_variables["mass_ratio"])

    def _generic_args(self, val, required_variables):
        return val, required_variables["chi_eff"], required_variables["mass_ratio"], self.amax

    @xp_wrap
    def rescale(self, val, *, xp=None, **required_variables):
        if (xp := self._generic(val, xp, required_variables)) is not None:
            return _aligned_xp.chi2_rescale(xp, *self._generic_args(val, required_variables))
        shape, args = self._inputs(val, required_variables)
        return self._output(val, _chi2_rescale_vec(*args, float(self.amax)), shape)

    @xp_wrap
    def cdf(self, val, *, xp=None, **required_variables):
        if (xp := self._generic(val, xp, required_variables)) is not None:
            return _aligned_xp.chi2_cdf_pdf(xp, *self._generic_args(val, required_variables))[0]
        shape, args = self._inputs(val, required_variables)
        return self._output(val, _chi2_cdf_pdf_vec(*args, float(self.amax))[0], shape)

    @xp_wrap
    def prob(self, val, *, xp=None, **required_variables):
        if (xp := self._generic(val, xp, required_variables)) is not None:
            return _aligned_xp.chi2_cdf_pdf(xp, *self._generic_args(val, required_variables))[1]
        shape, args = self._inputs(val, required_variables)
        return self._output(val, _chi2_cdf_pdf_vec(*args, float(self.amax))[1], shape)

    @xp_wrap
    def ln_prob(self, val, *, xp=None, **required_variables):
        return _log(self.prob(val, xp=xp, **required_variables))


class Chi1(ConditionalDeltaFunction):

    def __init__(self, name="chi_1", latex_label=None, unit=None, boundary="periodic"):
        super().__init__(
            name=name, latex_label=latex_label, unit=unit, boundary=boundary,
            condition_func=self._condition_function, peak=0.1,
        )
        self._required_variables = ["chi_eff", "mass_ratio", "chi_2"]
        self.__class__.__name__ = "Chi1"
        self.__class__.__qualname__ = "Chi1"

    is_fixed = False

    def get_instantiation_dict(self):
        instantiation_dict = Prior.get_instantiation_dict(self)
        return instantiation_dict

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(name={self.name}, "
            f"latex_label={self.latex_label}, unit={self.unit}, "
            f"boundary={self.boundary})"
        )

    def _condition_function(self, reference_params, **required_variables):
        chi_eff = required_variables["chi_eff"]
        mass_ratio = required_variables["mass_ratio"]
        chi_2 = required_variables["chi_2"]
        chi_1 = (1 + mass_ratio) * chi_eff - mass_ratio * chi_2
        return dict(peak=chi_1)
