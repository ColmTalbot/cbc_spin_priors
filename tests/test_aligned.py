"""
Tests for cbc_spin_priors.aligned.
"""

import numpy as np
from pytest import fixture
from scipy.stats import ks_1samp, uniform

from bilby.core.prior import ConditionalPriorDict
from bilby.gw.conversion import convert_to_lal_binary_black_hole_parameters
from bilby.gw.prior import ConditionalChiInPlane, UniformInComponentsMassRatio

from cbc_spin_priors.aligned import AMAX, Chi1, Chi2, ChiEffQ

THRESHOLD = 1e-3


@fixture
def effective_spin_prior_dict():
    mass_ratio = UniformInComponentsMassRatio(0.1, 1.0)
    chi_eff = ChiEffQ(AMAX, 0.1, 1.0, name="chi_eff")
    chi_1 = Chi1()
    chi_2 = Chi2(AMAX)
    chi_1p = ConditionalChiInPlane(1e-5, AMAX, name="chi_1_in_plane")
    chi_2p = ConditionalChiInPlane(1e-5, AMAX, name="chi_2_in_plane")

    priors = ConditionalPriorDict(dict(
        mass_ratio=mass_ratio,
        chi_eff=chi_eff,
        chi_1=chi_1,
        chi_2=chi_2,
        chi_1_in_plane=chi_1p,
        chi_2_in_plane=chi_2p,
    ))
    return priors


def test_effective_spin_can_be_sampled_many(effective_spin_prior_dict):
    samples = effective_spin_prior_dict.sample(10)
    assert "chi_eff" in samples
    assert "chi_1" in samples
    assert "chi_2" in samples
    assert "chi_1_in_plane" in samples
    assert "chi_2_in_plane" in samples


def test_effective_spin_can_be_sampled_single(effective_spin_prior_dict):
    samples = effective_spin_prior_dict.sample()
    assert "chi_eff" in samples
    assert "chi_1" in samples
    assert "chi_2" in samples
    assert "chi_1_in_plane" in samples
    assert "chi_2_in_plane" in samples


def test_effective_spin_distribution_correct(effective_spin_prior_dict):
    rng = np.random.default_rng(123)
    samples = effective_spin_prior_dict.sample(10000, random_state=rng)
    samples, _ = convert_to_lal_binary_black_hole_parameters(samples)

    for key, bounds in [
        ("a_1", (0, AMAX)),
        ("a_2", (0, AMAX)),
        ("cos_tilt_1", (-1, 2)),
        ("cos_tilt_2", (-1, 2)),
    ]:
        dist = uniform(*bounds)
        ks_result = ks_1samp(samples[key], dist.cdf)
        assert ks_result.pvalue > THRESHOLD, f"Failed for key: {key} ({ks_result})"


def test_chi2_matches_quadrature():
    from scipy.integrate import quad

    prior = Chi2(AMAX)
    for chi_eff, mass_ratio in [(0.1, 0.8), (-0.4, 0.3), (0.02, 1.0), (0.7, 0.5), (0.0, 0.5)]:
        lower = min(prior.rescale(0.0, chi_eff=chi_eff, mass_ratio=mass_ratio), 0)
        upper = prior.rescale(1.0, chi_eff=chi_eff, mass_ratio=mass_ratio)

        def pdf(x):
            # unnormalised density
            return (
                np.log(np.abs(x) / AMAX)
                * np.log(np.abs((1 + mass_ratio) * chi_eff - mass_ratio * x) / AMAX)
            )

        singular = [x for x in (0.0, (1 + mass_ratio) * chi_eff / mass_ratio) if lower < x < upper]
        lower = prior.rescale(0.0, chi_eff=chi_eff, mass_ratio=mass_ratio)
        norm = quad(pdf, lower, upper, points=singular or None, limit=500)[0]
        assert np.isclose(
            quad(lambda x: prior.prob(x, chi_eff=chi_eff, mass_ratio=mass_ratio), lower, upper,
                 points=singular or None, limit=500)[0],
            1,
        )
        for xx in np.linspace(lower, upper, 9)[1:-1]:
            points = [x for x in singular if x < xx]
            expected = quad(pdf, lower, xx, points=points or None, limit=500)[0] / norm
            assert np.isclose(prior.cdf(xx, chi_eff=chi_eff, mass_ratio=mass_ratio), expected, atol=1e-7)
        uu = np.linspace(0, 1, 101)
        values = prior.rescale(uu, chi_eff=chi_eff, mass_ratio=mass_ratio)
        assert np.all(np.diff(values) >= 0)
        assert np.allclose(prior.cdf(values, chi_eff=chi_eff, mass_ratio=mass_ratio), uu, atol=1e-10)


CHI_EFF_MASS_RATIOS = [0.05, 0.125, 0.3, 0.5, 0.8, 0.999, 1.0]


def _chi_eff_breakpoints(mass_ratio):
    """Case boundaries of the piecewise chi_eff density, for |chi_eff|."""
    return [
        AMAX * (1 - mass_ratio) / (1 + mass_ratio),
        mass_ratio * AMAX / (1 + mass_ratio),
        AMAX / (1 + mass_ratio),
    ]


class _InterpolatedChiEffQ(ChiEffQ):
    """The previous lookup-table implementation, kept as a reference."""

    def __init__(self, amax, qmin, qmax, **kwargs):
        super().__init__(amax, qmin, qmax, **kwargs)
        self._setup_interpolants()

    def _setup_interpolants(self, *, xp=np):
        from scipy.integrate import cumulative_trapezoid
        from scipy.interpolate import RegularGridInterpolator

        qdomain = np.linspace(self.qmin, self.qmax, 800)
        cdomain = np.linspace(0, self.amax - 1e-8, 1000)
        chieff, qgrid = np.meshgrid(cdomain, qdomain, indexing="ij")
        pdfs = self._pdf(chieff, xp=xp, mass_ratio=qgrid)
        cdfs = cumulative_trapezoid(pdfs, cdomain, axis=0, initial=0)
        cdfs /= cdfs[-1]
        self._cdf_interpolant = RegularGridInterpolator((cdomain, qdomain), cdfs, method="linear")
        icdfs = np.array([np.interp(np.linspace(0, 1, 500), cdf, cdomain) for cdf in cdfs.T]).T
        self._icdf_interpolant = RegularGridInterpolator((np.linspace(0, 1, 500), qdomain), icdfs, method="linear")

    def cdf(self, val, **required_variables):
        val = np.asarray(val)
        sgn = np.sign(val)
        val = np.abs(val)
        qs = required_variables["mass_ratio"] * np.ones(val.shape)
        pts = np.array([np.minimum(val, self.amax), qs]).T
        return np.clip((1 + sgn * self._cdf_interpolant(pts).squeeze()) / 2, 0, 1)

    def rescale(self, val, **required_variables):
        val = np.asarray(val)
        sgn = np.sign(val - 0.5)
        icdf = 2 * np.abs(val - 0.5)
        qs = required_variables["mass_ratio"] * np.ones(val.shape)
        pts = np.array([icdf, qs]).T
        return sgn * self._icdf_interpolant(pts).squeeze()


def test_chi_eff_cdf_matches_quadrature():
    from scipy.integrate import quad

    prior = ChiEffQ(AMAX, 0.05, 1.0)
    for mass_ratio in CHI_EFF_MASS_RATIOS:
        breakpoints = _chi_eff_breakpoints(mass_ratio)
        values = list(np.linspace(0, AMAX, 12)[1:-1]) + [1e-12, AMAX - 1e-6] + [
            bb + dd for bb in breakpoints for dd in (-1e-4, -1e-9, 0, 1e-9, 1e-4) if 0 < bb + dd < AMAX
        ]

        def pdf(x):
            return prior.prob(x, mass_ratio=mass_ratio)

        for xx in values:
            points = [bb for bb in breakpoints if 0 < bb < xx] or None
            half = quad(pdf, 0, xx, points=points, limit=500, epsabs=1e-13, epsrel=1e-13)[0]
            for sgn in (1, -1):
                assert np.isclose(
                    prior.cdf(sgn * xx, mass_ratio=mass_ratio), 0.5 + sgn * half, rtol=0, atol=1e-10
                ), (mass_ratio, sgn * xx)
        assert prior.cdf(0.0, mass_ratio=mass_ratio) == 0.5
        assert prior.cdf(AMAX, mass_ratio=mass_ratio) == 1
        assert prior.cdf(-AMAX, mass_ratio=mass_ratio) == 0
        assert prior.cdf(2.0, mass_ratio=mass_ratio) == 1


def test_chi_eff_prob_normalised():
    from scipy.integrate import quad

    prior = ChiEffQ(AMAX, 0.05, 1.0)
    for mass_ratio in CHI_EFF_MASS_RATIOS:
        points = sorted({0.0} | {sgn * bb for bb in _chi_eff_breakpoints(mass_ratio) for sgn in (1, -1)})
        total = quad(
            lambda x: prior.prob(x, mass_ratio=mass_ratio), -AMAX, AMAX,
            points=points, limit=500, epsabs=1e-13,
        )[0]
        assert np.isclose(total, 1, rtol=0, atol=1e-10), mass_ratio
        assert np.isclose(
            prior.prob(0.0, mass_ratio=mass_ratio),
            (1 + mass_ratio) * (2 - np.log(mass_ratio)) / (2 * AMAX),
        )


def test_chi_eff_rescale_round_trip_and_monotonic():
    prior = ChiEffQ(AMAX, 0.05, 1.0)
    uu = np.concatenate([[0, 1e-10, 1e-6], np.linspace(0, 1, 2001)[1:-1], [1 - 1e-6, 1 - 1e-10, 1]])
    for mass_ratio in CHI_EFF_MASS_RATIOS:
        values = prior.rescale(uu, mass_ratio=mass_ratio)
        assert values[0] == -AMAX and values[-1] == AMAX
        assert np.all(np.diff(values) > 0), mass_ratio
        assert np.allclose(prior.cdf(values, mass_ratio=mass_ratio), uu, rtol=0, atol=1e-13), mass_ratio
        breakpoints = _chi_eff_breakpoints(mass_ratio)
        xx = np.array([sgn * (bb + dd) for bb in breakpoints for dd in (-1e-6, 0, 1e-6) for sgn in (1, -1)])
        xx = xx[np.abs(xx) < AMAX]
        assert np.allclose(prior.rescale(prior.cdf(xx, mass_ratio=mass_ratio), mass_ratio=mass_ratio), xx, atol=1e-12)


def test_chi_eff_scalar_and_broadcast():
    prior = ChiEffQ(AMAX, 0.05, 1.0)
    assert isinstance(prior.cdf(0.3, mass_ratio=0.5), float)
    assert isinstance(prior.rescale(0.3, mass_ratio=0.5), float)
    qs = np.array([0.1, 0.5, 1.0])
    assert np.allclose(
        prior.rescale(0.7, mass_ratio=qs),
        [prior.rescale(0.7, mass_ratio=qq) for qq in qs],
    )
    assert prior.cdf(np.zeros((2, 3)), mass_ratio=qs).shape == (2, 3)


def test_chi_eff_matches_interpolated_reference():
    prior = ChiEffQ(AMAX, 0.1, 1.0)
    reference = _InterpolatedChiEffQ(AMAX, 0.1, 1.0)
    rng = np.random.default_rng(42)
    qs = rng.uniform(0.1, 1.0, 5000)
    uu = rng.uniform(0, 1, 5000)
    xx = rng.uniform(-AMAX, AMAX, 5000)
    # the old tables were linear interpolants on 1000 x 800 (cdf) and 500 x 800 (inverse)
    # grids, good to ~1e-6 in the cdf and ~5e-4 in u for the inverse; the inverse table is
    # coarse in the tails, where chi_eff changes quickly with u, so compare it in u
    assert np.max(np.abs(prior.cdf(xx, mass_ratio=qs) - reference.cdf(xx, mass_ratio=qs))) < 1e-5
    old_values = reference.rescale(uu, mass_ratio=qs)
    assert np.max(np.abs(prior.cdf(old_values, mass_ratio=qs) - uu)) < 1e-3
    bulk = np.abs(uu - 0.5) < 0.49
    assert np.max(np.abs(prior.rescale(uu, mass_ratio=qs) - old_values)[bulk]) < 1e-3


def test_chi_eff_timing_against_interpolated_reference():
    from timeit import timeit

    prior = ChiEffQ(AMAX, 0.1, 1.0)
    reference = _InterpolatedChiEffQ(AMAX, 0.1, 1.0)
    rng = np.random.default_rng(1)
    qs = rng.uniform(0.1, 1.0, 10000)
    uu = rng.uniform(0, 1, 10000)
    xx = rng.uniform(-AMAX, AMAX, 10000)
    for func in (prior.cdf, prior.rescale):
        func(xx[:2], mass_ratio=qs[:2])  # compile
    timings = dict(
        cdf_new=timeit(lambda: prior.cdf(xx, mass_ratio=qs), number=5) / 5,
        cdf_old=timeit(lambda: reference.cdf(xx, mass_ratio=qs), number=5) / 5,
        rescale_new=timeit(lambda: prior.rescale(uu, mass_ratio=qs), number=5) / 5,
        rescale_old=timeit(lambda: reference.rescale(uu, mass_ratio=qs), number=5) / 5,
    )
    print({key: f"{value * 1e3:.2f} ms" for key, value in timings.items()})
    # generous bounds, also hold without numba
    assert timings["cdf_new"] < 100 * timings["cdf_old"] + 0.5
    assert timings["rescale_new"] < 100 * timings["rescale_old"] + 0.5
