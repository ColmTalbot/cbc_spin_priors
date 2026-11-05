"""
Helpers for the generic Python array API code paths.

The priors keep two implementations: compiled numba kernels (or a vectorised
numpy fallback) for numpy arrays and Python scalars, and a generic
implementation written only in terms of the array API standard for every
other namespace (e.g. JAX, PyTorch, CuPy), detected with
:code:`array_api_compat`. No array library other than numpy is imported here.

The generic code is written so that it can be traced and differentiated:

- no boolean-mask indexing or in-place assignment, branches are evaluated on
  every element and combined with :code:`xp.where` (with the denominators and
  arguments of the unused branch made safe so they do not produce NaN
  gradients);
- iterations have a fixed maximum count, with an early exit only for eager
  (non-lazy) arrays, see :func:`newton_bisect`.
"""

import math

import array_api_compat as aac
import array_api_compat.numpy as aac_numpy
import numpy as np

__all__ = ["namespace", "is_numpy", "as_float", "acos_clip", "newton_bisect", "spence", "real_dilog"]

_SCALARS = (bool, int, float, complex)


def _namespace_of(arg):
    """The namespace of a single argument, None for scalars and unknown types."""
    if arg is None or isinstance(arg, _SCALARS):
        return None
    try:
        return aac.array_namespace(arg)
    except TypeError:
        # e.g. pandas series, handled by the numpy path
        return aac_numpy


def namespace(*args, xp=None):
    """
    The array namespace of :code:`args`.

    Python scalars and :code:`None` are ignored, if there are no arrays the
    namespace is numpy. Objects :code:`array_api_compat` does not recognise
    (e.g. pandas series) are treated as numpy. Numpy arrays mixed with arrays
    of one other namespace (e.g. conditioning variables from priors that only
    return numpy) use the other namespace. If the arguments are all numpy but
    :code:`xp` (e.g. from :code:`bilby.compat.utils.xp_wrap`) is another
    namespace, :code:`xp` is used.
    """
    found = [ns for ns in map(_namespace_of, args) if ns is not None]
    others = [ns for ns in found if not is_numpy(ns)]
    if not others:
        if xp is not None and not is_numpy(xp):
            return xp
        return aac_numpy
    if any(ns is not others[0] for ns in others):
        raise TypeError(f"Multiple array namespaces: {set(ns.__name__ for ns in others)}")
    return others[0]


def is_numpy(xp):
    return aac.is_numpy_namespace(xp)


def _device(xp, args):
    """The device of the first argument in the namespace xp."""
    for arg in args:
        if _namespace_of(arg) is xp:
            return aac.device(arg)
    return None


def _to_float(xp, arg, device):
    # arrays already in xp are cast rather than passed through asarray, which
    # for some libraries (e.g. torch) drops the autograd history
    if aac.is_array_api_obj(arg) and _namespace_of(arg) is xp:
        return arg if arg.dtype == xp.float64 else xp.astype(arg, xp.float64)
    return xp.asarray(arg, dtype=xp.float64, device=device)


def as_float(xp, *args):
    """
    Broadcast :code:`args` to float64 arrays of :code:`xp` on the device of
    the first array, flattened to 1D.

    Returns the broadcast shape, the flat arrays and the device.
    """
    device = _device(xp, args)
    arrays = xp.broadcast_arrays(*[_to_float(xp, arg, device) for arg in args])
    shape = arrays[0].shape
    return shape, [xp.reshape(arr, (-1,)) for arr in arrays], device


def acos_clip(xp, x):
    """
    :code:`acos(clip(x, -1, 1))` with a zero rather than NaN gradient where
    the argument is clipped (the derivative of acos is infinite at +-1).
    """
    inside = xp.abs(x) < 1
    return xp.where(
        inside,
        xp.acos(xp.where(inside, x, 0.0)),
        math.pi * xp.astype(x <= -1, x.dtype),
    )


def _is_eager(arr):
    return not aac.is_lazy_array(arr)


def newton_bisect(xp, func, x0, lo, hi, tol, xrtol, xfloor, done=None, maxiter=100, step_rtol=None):
    """
    Vectorised safeguarded Newton iteration for an increasing function,
    mirroring the scalar numba kernels element by element.

    :code:`func(x)` returns :code:`(resid, step)` with the Newton step
    :code:`resid / deriv` (NaN or infinite if the derivative is unusable).
    Each element stops when :code:`|resid| <= tol` or when the bracket is
    narrower than :code:`xrtol * max(|hi|, xfloor)`, the step falls back to
    bisection if it leaves the bracket or the last step did not halve the
    residual. If :code:`step_rtol` is given, an accepted Newton step smaller
    than :code:`step_rtol * new` is taken and the element stops.

    Every iteration evaluates :code:`func` on all elements, converged ones are
    frozen with :code:`xp.where`. The loop runs at most :code:`maxiter` times
    and exits early once everything has converged, unless the arrays are lazy
    (e.g. traced under :code:`jax.jit`) in which case all :code:`maxiter`
    iterations are unrolled.
    """
    xx = x0
    if done is None:
        done = xx != xx
    previous = xp.full_like(xx, math.inf)
    for _ in range(maxiter):
        resid, step = func(xx)
        stop = done | (xp.abs(resid) <= tol)
        lo = xp.where(~stop & (resid < 0), xx, lo)
        hi = xp.where(~stop & ~(resid < 0), xx, hi)
        stop = stop | (hi - lo <= xrtol * xp.clip(xp.abs(hi), min=xfloor))
        new = xx - step
        bad = (
            ~xp.isfinite(new) | (new <= lo) | (new >= hi)
            | (xp.abs(resid) > 0.5 * previous)
        )
        small = ~bad & (xp.abs(new - xx) <= step_rtol * new) if step_rtol is not None else stop
        new = xp.where(bad, (lo + hi) / 2, new)
        previous = xp.where(stop, previous, xp.abs(resid))
        xx = xp.where(stop, xx, new)
        done = stop | small
        if _is_eager(done) and bool(xp.all(done)):
            break
    return xx


# Cephes coefficients for the dilogarithm
_SPENCE_A = (
    4.65128586073990045278E-5, 7.31589045238094711071E-3, 1.33847639578309018650E-1,
    8.79691311754530315341E-1, 2.71149851196553469920E0, 4.25697156008121755724E0,
    3.29771340985225106936E0, 1.00000000000000000126E0,
)
_SPENCE_B = (
    6.90990488912553276999E-4, 2.54043763932544379113E-2, 2.82974860602568089943E-1,
    1.41172597751831069617E0, 3.63800533345137075418E0, 5.03278880143316990390E0,
    3.54771340985225096217E0, 9.99999999999999998740E-1,
)


def spence(xp, x):
    """Li2(1 - x) for x >= 0, matches scipy.special.spence (Cephes)."""
    pi2 = math.pi**2 / 6
    special = (x == 0) | (x == 1)
    xs = xp.where(special, 0.75, x)
    big = xs > 2.0
    xs = xp.where(big, 1.0 / xs, xs)
    upper = xs > 1.5
    lower = ~upper & (xs < 0.5)
    w = xp.where(upper, 1.0 / xs - 1.0, xp.where(lower, -xs, xs - 1.0))
    pa = xp.zeros_like(w)
    for c in _SPENCE_A:
        pa = pa * w + c
    pb = xp.zeros_like(w)
    for c in _SPENCE_B:
        pb = pb * w + c
    y = -w * pa / pb
    # 1 - xs is only small where xs is close to 1, which is not the lower branch
    xl = xp.where(lower, xs, 0.25)
    y = xp.where(lower, pi2 - xp.log(xl) * xp.log(1.0 - xl) - y, y)
    z = xp.log(xs)
    y = xp.where(big | upper, -0.5 * z * z - y, y)
    return xp.where(x == 0, pi2, xp.where(x == 1, 0.0, y))


def real_dilog(xp, z):
    """Real part of Li2(z) for real z."""
    below = z <= 1.0
    z_lo = xp.where(below, z, 0.0)
    z_hi = xp.where(below, 2.0, z)
    lz = xp.log(z_hi)
    return xp.where(
        below,
        spence(xp, 1.0 - z_lo),
        math.pi**2 / 3 - 0.5 * lz * lz - spence(xp, 1.0 - 1.0 / z_hi),
    )
