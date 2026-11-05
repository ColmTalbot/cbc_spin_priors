"""
Tests for cbc_spin_priors.inplane.
"""

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import chisquare, ks_1samp, uniform

from bilby.core.prior import ConditionalPriorDict, PriorDict, Uniform
from bilby.gw.conversion import convert_to_lal_binary_black_hole_parameters
from bilby.gw.prior import AlignedSpin, ConditionalChiInPlane, UniformInComponentsMassRatio

from cbc_spin_priors.aligned import AMAX, ChiEffQ, Chi1, Chi2
from cbc_spin_priors import inplane as ipp
from cbc_spin_priors.inplane import (
    ChiInPlane1FromTotal,
    ChiInPlane2GivenTotal,
    ChiInPlaneTotal,
    Phi12FromTotal,
    PhiS2,
    total_from_in_plane,
    total_in_plane_spin_priors,
)

THRESHOLD = 1e-3
QMIN = 0.125


def _prior_dict(aligned_chain, in_plane):
    priors = dict(mass_ratio=UniformInComponentsMassRatio(QMIN, 1.0))
    if aligned_chain == "effective":
        priors["chi_eff"] = ChiEffQ(AMAX, QMIN, 1.0, name="chi_eff")
        priors["chi_2"] = Chi2(AMAX)
        priors["chi_1"] = Chi1()
    else:
        priors["chi_1"] = AlignedSpin(Uniform(0, AMAX), name="chi_1")
        priors["chi_2"] = AlignedSpin(Uniform(0, AMAX), name="chi_2")
    if in_plane == "total":
        priors.update(total_in_plane_spin_priors(AMAX))
    else:
        priors["chi_1_in_plane"] = ConditionalChiInPlane(1e-5, AMAX, name="chi_1_in_plane")
        priors["chi_2_in_plane"] = ConditionalChiInPlane(1e-5, AMAX, name="chi_2_in_plane")
        priors["phi_12"] = Uniform(0, 2 * np.pi, name="phi_12", boundary="periodic")
    priors["phi_jl"] = Uniform(0, 2 * np.pi, name="phi_jl", boundary="periodic")
    return ConditionalPriorDict(priors)


@pytest.fixture(scope="module", params=["aligned", "effective"])
def aligned_chain(request):
    return request.param


@pytest.fixture(scope="module")
def total_samples(aligned_chain):
    priors = _prior_dict(aligned_chain, "total")
    rng = np.random.default_rng(42)
    return priors.sample(20000, random_state=rng)


@pytest.fixture(scope="module")
def standard_samples():
    priors = _prior_dict("aligned", "standard")
    rng = np.random.default_rng(43)
    return priors.sample(20000, random_state=rng)


def _standard_unit_cube(samples):
    """Push samples through the CDF chain of the standard aligned prior."""
    aligned = AlignedSpin(Uniform(0, AMAX), name="chi_1")
    in_plane_1 = ConditionalChiInPlane(1e-5, AMAX, name="chi_1_in_plane")
    in_plane_2 = ConditionalChiInPlane(1e-5, AMAX, name="chi_2_in_plane")
    return dict(
        chi_1=aligned.cdf(samples["chi_1"]),
        chi_2=aligned.cdf(samples["chi_2"]),
        chi_1_in_plane=in_plane_1.cdf(samples["chi_1_in_plane"], chi_1=samples["chi_1"]),
        chi_2_in_plane=in_plane_2.cdf(samples["chi_2_in_plane"], chi_2=samples["chi_2"]),
        phi_12=np.mod(samples["phi_12"], 2 * np.pi) / (2 * np.pi),
    )


def _total_unit_cube(samples):
    """Push samples through the CDF chain of the total in-plane spin prior."""
    rho, r2, theta = total_from_in_plane(
        samples["chi_1_in_plane"], samples["chi_2_in_plane"], samples["phi_12"], samples["mass_ratio"]
    )
    common = dict(chi_1=samples["chi_1"], mass_ratio=samples["mass_ratio"])
    return dict(
        chi_in_plane_total=ChiInPlaneTotal(AMAX).cdf(rho, chi_2=samples["chi_2"], **common),
        chi_2_in_plane=ChiInPlane2GivenTotal(AMAX).cdf(
            r2, chi_in_plane_total=rho, chi_2=samples["chi_2"], **common
        ),
        phi_s2=PhiS2(AMAX).cdf(theta, chi_2_in_plane=r2, chi_in_plane_total=rho, **common),
    )


def _assert_independent_uniform(columns, bins=5):
    """KS test each column and a chi-squared test of the joint histogram."""
    for key, values in columns.items():
        result = ks_1samp(values, uniform(0, 1).cdf)
        assert result.pvalue > THRESHOLD, f"{key}: {result}"
    data = np.column_stack(list(columns.values()))
    counts, _ = np.histogramdd(data, bins=bins, range=[(0, 1)] * data.shape[1])
    result = chisquare(counts.ravel())
    assert result.pvalue > THRESHOLD, f"joint {list(columns)}: {result}"


def test_can_be_sampled_single(aligned_chain):
    samples = _prior_dict(aligned_chain, "total").sample()
    for key in ["chi_in_plane_total", "chi_2_in_plane", "phi_s2", "chi_1_in_plane", "phi_12"]:
        assert np.ndim(samples[key]) == 0


def test_rescale_matches_sample(aligned_chain):
    priors = _prior_dict(aligned_chain, "total")
    keys = priors.sorted_keys_without_fixed_parameters
    rng = np.random.default_rng(1)
    theta = rng.uniform(size=len(keys))
    values = dict(zip(keys, priors.rescale(keys, theta)))
    rho, r2, angle = total_from_in_plane(
        values["chi_1_in_plane"], values["chi_2_in_plane"], values["phi_12"], values["mass_ratio"]
    )
    assert np.isclose(rho, values["chi_in_plane_total"], atol=1e-12)
    assert np.isclose(np.mod(angle - values["phi_s2"] + np.pi, 2 * np.pi), np.pi, atol=1e-12)


def test_spin_distribution_correct(total_samples):
    samples, _ = convert_to_lal_binary_black_hole_parameters(total_samples)
    for key, bounds in [
        ("a_1", (0, AMAX)),
        ("a_2", (0, AMAX)),
        ("cos_tilt_1", (-1, 2)),
        ("cos_tilt_2", (-1, 2)),
        ("phi_12", (0, 2 * np.pi)),
        ("phi_jl", (0, 2 * np.pi)),
    ]:
        result = ks_1samp(samples[key], uniform(*bounds).cdf)
        assert result.pvalue > THRESHOLD, f"Failed for key: {key} ({result})"
    assert np.all(samples["a_1"] <= AMAX + 1e-12)
    assert np.all(samples["a_2"] <= AMAX + 1e-12)


def test_joint_distribution_matches_standard(total_samples):
    """The standard prior's unit-cube coordinates of samples from the new prior are iid uniform."""
    unit = _standard_unit_cube(total_samples)
    unit["mass_ratio"] = UniformInComponentsMassRatio(QMIN, 1.0).cdf(total_samples["mass_ratio"])
    _assert_independent_uniform(
        {key: unit[key] for key in ["chi_1_in_plane", "chi_2_in_plane", "phi_12"]}
    )
    _assert_independent_uniform(
        {key: unit[key] for key in ["chi_1", "chi_1_in_plane", "chi_2_in_plane"]}
    )
    _assert_independent_uniform(
        {key: unit[key] for key in ["mass_ratio", "chi_1_in_plane", "phi_12"]}
    )


def test_standard_samples_uniform_in_new_cube(standard_samples):
    """The reverse: samples from the standard prior are iid uniform in the new unit cube."""
    unit = _total_unit_cube(standard_samples)
    _assert_independent_uniform(unit)
    q_unit = UniformInComponentsMassRatio(QMIN, 1.0).cdf(standard_samples["mass_ratio"])
    _assert_independent_uniform(dict(mass_ratio=q_unit, chi_in_plane_total=unit["chi_in_plane_total"]))


CONDITIONS = [
    dict(chi_1=0.3, chi_2=-0.5, mass_ratio=0.8),
    dict(chi_1=1e-5, chi_2=0.2, mass_ratio=0.5),
    dict(chi_1=0.7, chi_2=-1e-5, mass_ratio=1.0),
    dict(chi_1=-1e-5, chi_2=1e-5, mass_ratio=0.9),
    dict(chi_1=1e-8, chi_2=0.3, mass_ratio=QMIN),
    dict(chi_1=0.98, chi_2=0.1, mass_ratio=0.3),
    dict(chi_1=0.1, chi_2=0.98, mass_ratio=QMIN),
    dict(chi_1=-0.5, chi_2=0.5, mass_ratio=QMIN),
    dict(chi_1=-0.56, chi_2=3e-8, mass_ratio=0.53),
]
UNITS = np.concatenate([[1e-10, 1e-6, 1e-3], np.linspace(0.01, 0.99, 25), [1 - 1e-3, 1 - 1e-6, 1 - 1e-10]])


def _chain(cond, u_total=0.5, u_r2=0.5):
    rho = ChiInPlaneTotal(AMAX).rescale(u_total, **cond)
    r2 = ChiInPlane2GivenTotal(AMAX).rescale(u_r2, chi_in_plane_total=rho, **cond)
    angle_cond = dict(
        chi_2_in_plane=r2, chi_in_plane_total=rho, chi_1=cond["chi_1"], mass_ratio=cond["mass_ratio"]
    )
    return rho, r2, angle_cond


@pytest.mark.parametrize("cond", CONDITIONS)
@pytest.mark.parametrize("u_total", [1e-6, 0.3, 0.9, 1 - 1e-6])
def test_cdf_rescale_roundtrip(cond, u_total):
    prior = ChiInPlaneTotal(AMAX)
    uu = UNITS
    vals = prior.rescale(uu, **cond)
    assert np.all(np.diff(vals) > 0)
    assert np.allclose(prior.cdf(vals, **cond), uu, atol=1e-9, rtol=0)
    # x-space round trip away from the extreme tails, where tiny CDF errors are amplified
    inner = slice(1, -1)
    assert np.allclose(prior.rescale(prior.cdf(vals[inner], **cond), **cond), vals[inner], rtol=1e-6, atol=1e-12)

    rho, _, _ = _chain(cond, u_total)
    prior = ChiInPlane2GivenTotal(AMAX)
    kwargs = dict(chi_in_plane_total=rho, **cond)
    vals = prior.rescale(uu, **kwargs)
    assert np.all(np.diff(vals) >= 0)
    assert np.allclose(prior.cdf(vals, **kwargs), uu, atol=1e-9, rtol=0)
    assert np.allclose(prior.rescale(prior.cdf(vals[inner], **kwargs), **kwargs), vals[inner], rtol=1e-6, atol=1e-12)

    for u_r2 in [1e-8, 0.5, 1 - 1e-8]:
        _, r2, angle_cond = _chain(cond, u_total, u_r2)
        prior = PhiS2(AMAX)
        vals = prior.rescale(uu, **angle_cond)
        assert np.all(np.diff(vals) > 0)
        assert np.allclose(prior.cdf(vals, **angle_cond), uu, atol=1e-10, rtol=0)
        assert np.allclose(prior.rescale(prior.cdf(vals[inner], **angle_cond), **angle_cond), vals[inner], atol=1e-8)
        # the implied primary in-plane spin is always allowed
        chi_1p, _ = ipp.in_plane_from_total(rho, r2, vals, cond["mass_ratio"])
        assert np.all(chi_1p**2 + cond["chi_1"] ** 2 <= AMAX**2 * (1 + 1e-10))


def _check_density_matches_cdf(prior, kwargs, atol=1e-7):
    """The integral of prob between consecutive quantiles equals the CDF difference."""
    uu = np.concatenate([[1e-9, 1e-4], np.linspace(0.01, 0.99, 50), [1 - 1e-4, 1 - 1e-9]])
    xx = prior.rescale(uu, **kwargs)
    pieces = np.array([
        quad(lambda x: prior.prob(x, **kwargs), lo, hi, limit=200, epsabs=1e-12, epsrel=1e-10)[0]
        if hi > lo else 0.0
        for lo, hi in zip(xx[:-1], xx[1:])
    ])
    assert np.allclose(pieces, np.diff(uu), atol=atol, rtol=0), np.max(np.abs(pieces - np.diff(uu)))
    assert np.isclose(np.sum(pieces), uu[-1] - uu[0], atol=atol * 10)


@pytest.mark.parametrize("cond", CONDITIONS)
def test_prob_normalised(cond):
    _check_density_matches_cdf(ChiInPlaneTotal(AMAX), cond)
    for u_total in [0.05, 0.5, 0.95]:
        rho, _, angle_cond = _chain(cond, u_total)
        _check_density_matches_cdf(ChiInPlane2GivenTotal(AMAX), dict(chi_in_plane_total=rho, **cond))
        _check_density_matches_cdf(PhiS2(AMAX), angle_cond)
        prior = PhiS2(AMAX)
        assert np.isclose(np.exp(prior.ln_prob(0.1, **angle_cond)), prior.prob(0.1, **angle_cond))


def _histogram_check(samples, prob, bins):
    counts, edges = np.histogram(samples, bins=bins)
    expected = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        fine = np.linspace(lo, hi, 201)
        expected.append(np.trapezoid(prob(fine), fine))
    expected = np.array(expected) * len(samples)
    use = expected > 50
    zscores = (counts[use] - expected[use]) / np.sqrt(expected[use])
    assert np.max(np.abs(zscores)) < 5, zscores


@pytest.mark.parametrize("cond", CONDITIONS[:4] + CONDITIONS[-1:])
def test_prob_matches_histogram(cond):
    rng = np.random.default_rng(7)
    n = 20000
    prior = ChiInPlaneTotal(AMAX)
    samples = prior.rescale(rng.uniform(size=n), **cond)
    _histogram_check(samples, lambda x: prior.prob(x, **cond), np.quantile(samples, np.linspace(0.02, 0.98, 25)))

    rho, _, _ = _chain(cond, 0.6)
    prior = ChiInPlane2GivenTotal(AMAX)
    kwargs = dict(chi_in_plane_total=rho, **cond)
    samples = prior.rescale(rng.uniform(size=n), **kwargs)
    _histogram_check(samples, lambda x: prior.prob(x, **kwargs), np.quantile(samples, np.linspace(0.02, 0.98, 25)))

    _, _, angle_cond = _chain(cond, 0.6)
    prior = PhiS2(AMAX)
    samples = prior.rescale(rng.uniform(size=n), **angle_cond)
    _histogram_check(samples, lambda x: prior.prob(x, **angle_cond), np.quantile(samples, np.linspace(0.02, 0.98, 25)))


def test_joint_density_matches_standard_with_jacobian():
    """p(|S|) p(|v_2| | |S|) p(theta | ...) = p(chi_1p) p(chi_2p) / (2 pi) |d(chi_1p, chi_2p, phi_12) / d(|S|, |v_2|, theta)|"""
    rng = np.random.default_rng(3)
    in_plane = ConditionalChiInPlane(1e-5, AMAX, name="chi_1_in_plane")
    in_plane_2 = ConditionalChiInPlane(1e-5, AMAX, name="chi_2_in_plane")
    for _ in range(50):
        cond = dict(chi_1=rng.uniform(-0.9, 0.9), chi_2=rng.uniform(-0.9, 0.9), mass_ratio=rng.uniform(QMIN, 1))
        rho, r2, angle_cond = _chain(cond, *rng.uniform(0.05, 0.95, 2))
        theta = PhiS2(AMAX).rescale(rng.uniform(0.05, 0.95), **angle_cond)
        new = (
            ChiInPlaneTotal(AMAX).prob(rho, **cond)
            * ChiInPlane2GivenTotal(AMAX).prob(r2, chi_in_plane_total=rho, **cond)
            * PhiS2(AMAX).prob(theta, **angle_cond)
        )
        q = cond["mass_ratio"]
        chi_1p, phi_12 = ipp.in_plane_from_total(rho, r2, theta, q)
        old = (
            in_plane.prob(chi_1p, chi_1=cond["chi_1"])
            * in_plane_2.prob(r2, chi_2=cond["chi_2"])
            / (2 * np.pi)
        )
        h = 1e-6

        def unwrap(x):
            return np.angle(np.exp(1j * x))

        d_rho = [(a - b) / (2 * h) for a, b in zip(
            ipp.in_plane_from_total(rho + h, r2, theta, q), ipp.in_plane_from_total(rho - h, r2, theta, q))]
        d_theta = [(a - b) / (2 * h) for a, b in zip(
            ipp.in_plane_from_total(rho, r2, theta + h, q), ipp.in_plane_from_total(rho, r2, theta - h, q))]
        d_rho[1] = unwrap(d_rho[1] * 2 * h) / (2 * h)
        d_theta[1] = unwrap(d_theta[1] * 2 * h) / (2 * h)
        jacobian = abs(d_rho[0] * d_theta[1] - d_theta[0] * d_rho[1])
        assert np.isclose(new, old * jacobian, rtol=1e-5), (cond, rho, r2, theta, new, old * jacobian)


@pytest.mark.skipif(ipp._numba is None, reason="numba not available")
def test_numba_matches_numpy():
    rng = np.random.default_rng(5)
    n = 500
    chi_1 = rng.uniform(-0.98, 0.98, n)
    chi_2 = rng.uniform(-0.98, 0.98, n)
    chi_1[:50] = 10 ** rng.uniform(-9, -3, 50)
    chi_2[25:75] = -(10 ** rng.uniform(-9, -3, 50))
    q = rng.uniform(QMIN, 1, n)
    uu = rng.uniform(size=n)
    args = (chi_1, chi_2, q)
    rho = ipp.total_rescale(uu, *args, backend="numba")
    assert np.allclose(rho, ipp.total_rescale(uu, *args, backend="numpy"), rtol=1e-7, atol=1e-12)
    for a, b in zip(ipp.total_cdf_pdf(rho, *args, backend="numba"), ipp.total_cdf_pdf(rho, *args, backend="numpy")):
        assert np.allclose(a, b, rtol=1e-9, atol=1e-12)
    r2 = ipp.secondary_rescale(uu[::-1], rho, *args, backend="numba")
    assert np.allclose(r2, ipp.secondary_rescale(uu[::-1], rho, *args, backend="numpy"), rtol=1e-7, atol=1e-12)
    for a, b in zip(ipp.secondary_cdf_pdf(r2, rho, *args, backend="numba"),
                    ipp.secondary_cdf_pdf(r2, rho, *args, backend="numpy")):
        assert np.allclose(a, b, rtol=1e-8, atol=1e-10)


def test_outside_support_has_zero_prob():
    cond = dict(chi_1=0.3, chi_2=-0.5, mass_ratio=0.8)
    assert ChiInPlaneTotal(AMAX).prob(1.0, **cond) == 0
    assert ChiInPlaneTotal(AMAX).cdf(1.0, **cond) == 1
    assert ChiInPlaneTotal(AMAX).cdf(0.0, **cond) == 0
    rho, _, angle_cond = _chain(cond)
    lo, hi = ChiInPlane2GivenTotal(AMAX).rescale(np.array([0.0, 1.0]), chi_in_plane_total=rho, **cond)
    assert ChiInPlane2GivenTotal(AMAX).prob(hi * 1.01 + 1e-3, chi_in_plane_total=rho, **cond) == 0
    assert np.isneginf(ChiInPlane2GivenTotal(AMAX).ln_prob(hi * 1.01 + 1e-3, chi_in_plane_total=rho, **cond))


def test_prior_dict_round_trip(tmp_path):
    priors = PriorDict(total_in_plane_spin_priors(AMAX))
    for key, prior in priors.items():
        assert prior.__class__.__name__ == type(prior).__name__
        assert type(prior).__name__ in repr(prior)
    priors.to_json(outdir=str(tmp_path), label="test")
    loaded = PriorDict.from_json(str(tmp_path / "test_prior.json"))
    for key in priors:
        assert type(loaded[key]) is type(priors[key])
        assert repr(loaded[key]) == repr(priors[key])
    namespace = {cls.__name__: cls for cls in [
        ChiInPlaneTotal, ChiInPlane2GivenTotal, PhiS2, ChiInPlane1FromTotal, Phi12FromTotal]}
    for key in priors:
        assert repr(eval(repr(priors[key]), namespace)) == repr(priors[key])


def test_conditional_delta_functions_are_not_fixed():
    priors = _prior_dict("effective", "total")
    keys = priors.sorted_keys_without_fixed_parameters
    for key in ["chi_1_in_plane", "phi_12", "chi_in_plane_total", "chi_2_in_plane", "phi_s2"]:
        assert key in keys
        assert not priors[key].is_fixed
    order = priors.sorted_keys
    assert order.index("chi_in_plane_total") < order.index("chi_2_in_plane") < order.index("phi_s2")
    assert order.index("phi_s2") < order.index("chi_1_in_plane")
    assert order.index("phi_s2") < order.index("phi_12")
