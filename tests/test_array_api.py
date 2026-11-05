"""
Tests for the generic array API implementations, compared against the numba
kernels. Optional array libraries are skipped if they are not installed.
"""

import numpy as np
import pytest

import array_api_compat as aac

from cbc_spin_priors import _xp
from cbc_spin_priors import inplane as ipp
from cbc_spin_priors.aligned import (
    AMAX, Chi2, ChiEffQ, _chi2_cdf_pdf_vec, _chi2_rescale_vec, _chi_eff_cdf_vec, _chi_eff_rescale_vec,
)
from cbc_spin_priors.inplane import total_in_plane_spin_priors

RTOL = 1e-10


def _numpy():
    import array_api_compat.numpy as xp

    return xp


def _strict():
    return pytest.importorskip("array_api_strict")


def _torch():
    torch = pytest.importorskip("torch")
    return aac.array_namespace(torch.ones(1))


def _jax():
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    return aac.array_namespace(jax.numpy.ones(1))


NAMESPACES = dict(numpy=_numpy, array_api_strict=_strict, torch=_torch, jax=_jax)


@pytest.fixture(params=list(NAMESPACES))
def xp(request):
    return NAMESPACES[request.param]()


@pytest.fixture(params=[name for name in NAMESPACES if name != "numpy"])
def other_xp(request):
    """Namespaces that are dispatched to the generic path automatically."""
    return NAMESPACES[request.param]()


def to_numpy(arr):
    if aac.is_torch_array(arr):
        arr = arr.detach().cpu()
    return np.asarray(arr)


def assert_close(actual, expected, rtol=RTOL, atol=1e-300):
    np.testing.assert_allclose(to_numpy(actual), expected, rtol=rtol, atol=atol)


def asarray(xp, arr):
    # explicit float64, torch defaults to float32 for Python floats
    return xp.asarray(arr, dtype=xp.float64)


def assert_matches_numba(actual, numba, reference, regular):
    """
    Agreement with numba (or numpy, without numba) to RTOL for generic
    points. The hand-picked edge cases (aligned spins below the floor CMIN or
    at the maximum, equal masses) are ill-conditioned and numba, the numpy
    implementation and the generic path in different libraries differ there
    at the 1e-9 level.

    The numpy reference is skipped where it is not finite.
    """
    assert_close(actual[regular], numba[regular])
    assert_close(actual, numba, rtol=1e-7)
    if reference is not None:
        finite = np.isfinite(reference)
        assert_close(actual[regular & finite], reference[regular & finite])
        assert_close(actual[finite], reference[finite], rtol=1e-7)


@pytest.fixture(scope="module")
def conditions():
    rng = np.random.default_rng(10)
    n = 200
    chi_1 = rng.uniform(-AMAX, AMAX, n)
    chi_2 = rng.uniform(-AMAX, AMAX, n)
    mass_ratio = rng.uniform(0.05, 1, n)
    # edge cases: tiny and near-maximal aligned spins, equal masses
    chi_1[:4] = [1e-12, -0.98999, 0.5, 0.0]
    chi_2[:4] = [0.3, 1e-12, -0.98999, 0.0]
    mass_ratio[:4] = [1.0, 0.1, 1.0, 0.5]
    uu = rng.uniform(0, 1, n)
    uu[4:7] = [0.0, 1.0, 1e-12]
    return chi_1, chi_2, mass_ratio, uu


@pytest.fixture(scope="module")
def regular(conditions):
    """Everything but the hand-picked edge cases."""
    chi_1, *_ = conditions
    out = np.ones(len(chi_1), dtype=bool)
    out[:7] = False
    return out


def test_total_matches_numba(xp, conditions, regular):
    chi_1, chi_2, mass_ratio, uu = conditions
    args = [asarray(xp, arr) for arr in (chi_1, chi_2, mass_ratio)]
    rho = ipp.total_rescale(uu, chi_1, chi_2, mass_ratio, backend=ipp.BACKEND)
    actual = to_numpy(ipp.total_rescale(asarray(xp, uu), *args, backend="array_api"))
    assert_matches_numba(actual, rho, None, regular)
    # include points outside the support
    rho_eval = np.concatenate([rho, [-0.1, 1.5, 0.0]])
    full = [np.concatenate([arr, arr[:3]]) for arr in (chi_1, chi_2, mass_ratio)]
    expected = ipp.total_cdf_pdf(rho_eval, *full, backend=ipp.BACKEND)
    reference = ipp.total_cdf_pdf(rho_eval, *full, backend="numpy")
    actual = ipp.total_cdf_pdf(asarray(xp, rho_eval), *[asarray(xp, arr) for arr in full], backend="array_api")
    for a, b, c in zip(actual, expected, reference):
        assert_matches_numba(to_numpy(a), b, c, np.concatenate([regular, regular[:3]]))


def test_secondary_matches_numba(xp, conditions, regular):
    chi_1, chi_2, mass_ratio, uu = conditions
    rho = ipp.total_rescale(uu, chi_1, chi_2, mass_ratio, backend=ipp.BACKEND)
    u2 = uu[::-1].copy()
    r2 = ipp.secondary_rescale(u2, rho, chi_1, chi_2, mass_ratio, backend=ipp.BACKEND)
    args = [asarray(xp, arr) for arr in (rho, chi_1, chi_2, mass_ratio)]
    actual = to_numpy(ipp.secondary_rescale(asarray(xp, u2), *args, backend="array_api"))
    assert_matches_numba(actual, r2, None, regular)
    r2_eval = np.concatenate([r2[:-2], [0.0, AMAX]])
    expected = ipp.secondary_cdf_pdf(r2_eval, rho, chi_1, chi_2, mass_ratio, backend=ipp.BACKEND)
    reference = ipp.secondary_cdf_pdf(r2_eval, rho, chi_1, chi_2, mass_ratio, backend="numpy")
    actual = ipp.secondary_cdf_pdf(asarray(xp, r2_eval), *args, backend="array_api")
    for a, b, c in zip(actual, expected, reference):
        assert_matches_numba(to_numpy(a), b, c, regular)


def test_angle_matches_numpy(xp, conditions):
    """The angle prior has a single implementation, check other namespaces against numpy."""
    chi_1, chi_2, mass_ratio, uu = conditions
    rho = ipp.total_rescale(uu, chi_1, chi_2, mass_ratio)
    r2 = ipp.secondary_rescale(uu[::-1].copy(), rho, chi_1, chi_2, mass_ratio)
    theta = ipp.angle_rescale(uu, r2, rho, chi_1, mass_ratio)
    args = [asarray(xp, arr) for arr in (r2, rho, chi_1, mass_ratio)]
    assert_close(ipp.angle_rescale(asarray(xp, uu), *args), theta)
    for a, b in zip(ipp.angle_cdf_pdf(asarray(xp, theta), *args),
                    ipp.angle_cdf_pdf(theta, r2, rho, chi_1, mass_ratio)):
        assert_close(a, b)
    chi_1_in_plane, phi_12 = ipp.in_plane_from_total(rho, r2, theta, mass_ratio)
    converted = ipp.in_plane_from_total(*[asarray(xp, arr) for arr in (rho, r2, theta, mass_ratio)])
    assert_close(converted[0], chi_1_in_plane)
    assert_close(converted[1], phi_12)
    # round trip, the angle is undefined if either in-plane vector vanishes
    back = ipp.total_from_in_plane(*[asarray(xp, arr) for arr in (chi_1_in_plane, r2, phi_12, mass_ratio)])
    keep = (rho > 1e-6) & (r2 > 1e-6)
    assert_close(to_numpy(back[0])[keep], rho[keep], rtol=1e-8)
    assert_close(back[1], r2)
    wrapped = np.remainder(to_numpy(back[2]) - theta + np.pi, 2 * np.pi) - np.pi
    assert np.max(np.abs(wrapped[keep])) < 1e-8


def test_chi2_matches_numba(xp):
    from cbc_spin_priors import _aligned_xp

    rng = np.random.default_rng(11)
    n = 200
    mass_ratio = rng.uniform(0.05, 1, n)
    chi_eff = rng.uniform(-1, 1, n) * AMAX
    chi_eff[:3] = [0.0, 1e-9, -0.9]
    uu = rng.uniform(0, 1, n)
    uu[3:5] = [0.0, 1.0]
    x2 = _chi2_rescale_vec(uu, chi_eff, mass_ratio, AMAX)
    args = [asarray(xp, arr) for arr in (chi_eff, mass_ratio)]
    assert_close(_aligned_xp.chi2_rescale(xp, asarray(xp, uu), *args, AMAX), x2)
    x_eval = np.concatenate([x2, [-AMAX, AMAX, 0.0]])
    full = [np.concatenate([arr, arr[:3]]) for arr in (chi_eff, mass_ratio)]
    expected = _chi2_cdf_pdf_vec(x_eval, *full, AMAX)
    actual = _aligned_xp.chi2_cdf_pdf(xp, asarray(xp, x_eval), *[asarray(xp, arr) for arr in full], AMAX)
    for a, b in zip(actual, expected):
        assert_close(a, b)


def test_chi_eff_matches_numba(xp):
    from cbc_spin_priors import _aligned_xp

    rng = np.random.default_rng(12)
    n = 300
    mass_ratio = rng.uniform(0.05, 1, n)
    mass_ratio[:3] = [1.0, 0.999, 0.05]
    uu = rng.uniform(0, 1, n)
    uu[3:8] = [0.0, 1.0, 0.5, 1e-10, 1 - 1e-10]
    expected = _chi_eff_rescale_vec(uu, mass_ratio, AMAX)
    actual = _aligned_xp.chi_eff_rescale(xp, asarray(xp, uu), asarray(xp, mass_ratio), AMAX)
    assert_close(actual, expected, atol=1e-15)
    xx = np.concatenate([expected, [0.0, -AMAX, AMAX, 1.5, -1.5, 1e-12]])
    qq = np.concatenate([mass_ratio, mass_ratio[:6]])
    cdf, pdf = _aligned_xp.chi_eff_cdf_pdf(xp, asarray(xp, xx), asarray(xp, qq), AMAX)
    # absolute roundoff in the CDF near 0 differs between libraries
    assert_close(cdf, _chi_eff_cdf_vec(xx, qq, AMAX), atol=1e-15)
    # the numpy density uses scipy's complex dilogarithm and a different closed form
    reference = ChiEffQ(AMAX, 0.05, 1.0).prob(xx, mass_ratio=qq)
    assert_close(pdf, reference, rtol=1e-9, atol=1e-12)


def test_spence_matches_scipy(xp):
    from scipy.special import spence

    xx = np.concatenate([[0.0, 1.0, 0.5, 1.5, 2.0], np.geomspace(1e-8, 1e3, 200)])
    assert_close(_xp.spence(xp, asarray(xp, xx)), spence(xx), rtol=1e-13, atol=1e-15)


def test_priors_dispatch_on_namespace(other_xp):
    """Prior methods return arrays of the input namespace and agree with numba."""
    xp = other_xp
    priors = total_in_plane_spin_priors()
    conditions = dict(chi_1=0.3, chi_2=-0.6, mass_ratio=0.4)
    uu = np.array([0.1, 0.5, 0.9])
    rho = priors["chi_in_plane_total"].rescale(asarray(xp, uu), **conditions)
    assert aac.array_namespace(rho) is xp
    assert_close(rho, priors["chi_in_plane_total"].rescale(uu, **conditions))
    for method in ("cdf", "prob", "ln_prob"):
        value = getattr(priors["chi_in_plane_total"], method)(rho, **conditions)
        assert aac.array_namespace(value) is xp
        assert_close(value, getattr(priors["chi_in_plane_total"], method)(to_numpy(rho), **conditions))
    # numpy conditioning variables mixed with xp arrays, as from priors that only return numpy
    conditions["chi_in_plane_total"] = to_numpy(rho)
    r2 = priors["chi_2_in_plane"].rescale(asarray(xp, uu[::-1].copy()), **conditions)
    assert aac.array_namespace(r2) is xp
    assert_close(r2, priors["chi_2_in_plane"].rescale(uu[::-1].copy(), **conditions))
    # scalar input gives a 0-d array
    scalar = priors["chi_in_plane_total"].prob(asarray(xp, 0.2), chi_1=0.3, chi_2=-0.6, mass_ratio=0.4)
    assert scalar.shape == ()
    # the conditional delta functions
    peaks = priors["chi_1_in_plane"].condition_func(
        {}, chi_in_plane_total=rho, chi_2_in_plane=r2, phi_s2=asarray(xp, [0.1, -2.0, 3.0]), mass_ratio=0.4,
    )["peak"]
    assert aac.array_namespace(peaks) is xp
    # ChiEffQ
    prior = ChiEffQ(AMAX, 0.05, 1.0)
    chi_eff = prior.rescale(asarray(xp, uu), mass_ratio=0.7)
    assert aac.array_namespace(chi_eff) is xp
    assert_close(chi_eff, prior.rescale(uu, mass_ratio=0.7))
    for method in ("cdf", "prob", "ln_prob"):
        value = getattr(prior, method)(chi_eff, mass_ratio=0.7)
        assert aac.array_namespace(value) is xp
        assert_close(value, getattr(prior, method)(to_numpy(chi_eff), mass_ratio=0.7), rtol=1e-9)
    # Chi2
    prior = Chi2(AMAX)
    x2 = prior.rescale(asarray(xp, uu), chi_eff=0.2, mass_ratio=asarray(xp, 0.7))
    assert aac.array_namespace(x2) is xp
    assert_close(x2, prior.rescale(uu, chi_eff=0.2, mass_ratio=0.7))
    assert_close(prior.prob(x2, chi_eff=0.2, mass_ratio=0.7), prior.prob(to_numpy(x2), chi_eff=0.2, mass_ratio=0.7))


def test_xp_keyword_from_prior_dict(other_xp):
    """bilby passes xp explicitly, numpy inputs are then converted to that namespace."""
    xp = other_xp
    prior = total_in_plane_spin_priors()["chi_in_plane_total"]
    out = prior.prob(np.array([0.1, 0.3]), chi_1=0.3, chi_2=-0.6, mass_ratio=0.4, xp=xp)
    assert aac.array_namespace(out) is xp
    # numpy with numpy xp stays on the numba path
    out = prior.prob(np.array([0.1, 0.3]), chi_1=0.3, chi_2=-0.6, mass_ratio=0.4, xp=np)
    assert isinstance(out, np.ndarray)


def test_torch_gradients():
    """Derivatives through the quadrature and the Newton inverses are consistent."""
    torch = pytest.importorskip("torch")
    priors = total_in_plane_spin_priors()
    kwargs = dict(dtype=torch.float64)
    chi_1 = torch.tensor([0.3, -0.5, 0.01, 0.9], **kwargs)
    chi_2 = torch.tensor([0.2, 0.7, -0.9, 0.1], **kwargs)
    mass_ratio = torch.tensor([0.5, 0.9, 0.2, 1.0], **kwargs)
    uu = torch.tensor([0.2, 0.5, 0.97, 0.01], requires_grad=True, **kwargs)

    def check(prior, value, **conditions):
        # d rescale / du = 1 / pdf and d cdf / dx = pdf
        grad, = torch.autograd.grad(value.sum(), uu)
        pdf = prior.prob(value.detach(), **conditions)
        assert_close(grad, to_numpy(1 / pdf), rtol=1e-6)
        xx = value.detach().clone().requires_grad_(True)
        grad, = torch.autograd.grad(prior.cdf(xx, **conditions).sum(), xx)
        assert_close(grad, to_numpy(pdf), rtol=1e-8)

    conditions = dict(chi_1=chi_1, chi_2=chi_2, mass_ratio=mass_ratio)
    rho = priors["chi_in_plane_total"].rescale(uu, **conditions)
    check(priors["chi_in_plane_total"], rho, **conditions)
    conditions["chi_in_plane_total"] = rho.detach()
    r2 = priors["chi_2_in_plane"].rescale(uu, **conditions)
    check(priors["chi_2_in_plane"], r2, **conditions)
    conditions = dict(chi_2_in_plane=r2.detach(), chi_in_plane_total=rho.detach(), chi_1=chi_1, mass_ratio=mass_ratio)
    check(priors["phi_s2"], priors["phi_s2"].rescale(uu, **conditions), **conditions)
    conditions = dict(chi_eff=torch.tensor([0.1, -0.4, 0.0, 0.6], **kwargs), mass_ratio=mass_ratio)
    check(Chi2(AMAX), Chi2(AMAX).rescale(uu, **conditions), **conditions)
    prior = ChiEffQ(AMAX, 0.05, 1.0)
    check(prior, prior.rescale(uu, mass_ratio=mass_ratio), mass_ratio=mass_ratio)

    # gradient with respect to a conditioning variable against finite differences
    chi_1 = chi_1.clone().requires_grad_(True)
    rho = rho.detach()
    grad, = torch.autograd.grad(
        priors["chi_in_plane_total"].ln_prob(rho, chi_1=chi_1, chi_2=chi_2, mass_ratio=mass_ratio).sum(), chi_1
    )
    step = 1e-6
    args = [to_numpy(arr) for arr in (rho, chi_1, chi_2, mass_ratio)]
    plus = ipp.total_cdf_pdf(args[0], args[1] + step, *args[2:])[1]
    minus = ipp.total_cdf_pdf(args[0], args[1] - step, *args[2:])[1]
    assert_close(grad, (np.log(plus) - np.log(minus)) / (2 * step), rtol=1e-5)


def test_jax_jit_and_grad():
    """The closed-form and quadrature parts trace under jit."""
    _jax()
    import jax

    jnp = jax.numpy
    conditions = (jnp.asarray([0.3, -0.5]), jnp.asarray([0.2, 0.7]), jnp.asarray([0.5, 0.9]))
    rho = jnp.asarray([0.2, 0.4])
    cdf_pdf = jax.jit(lambda rho: ipp.total_cdf_pdf(rho, *conditions))
    cdf, pdf = cdf_pdf(rho)
    expected = ipp.total_cdf_pdf(np.asarray(rho), *[np.asarray(arr) for arr in conditions], backend=ipp.BACKEND)
    assert_close(cdf, expected[0])
    assert_close(pdf, expected[1])
    grad = jax.grad(lambda rho: ipp.total_cdf_pdf(rho, *conditions)[0].sum())(rho)
    assert_close(grad, expected[1], rtol=1e-8)
