"""
Tests for cbc_spin_priors.spins.set_spin_priors and the package interface.
"""

import numpy as np
import pytest
from scipy.stats import ks_1samp, uniform

from bilby.core.prior import PriorDict
from bilby.gw.conversion import convert_to_lal_binary_black_hole_parameters
from bilby.gw.prior import BBHPriorDict

import cbc_spin_priors
from cbc_spin_priors import AMAX, SPINS, set_spin_priors

THRESHOLD = 1e-3
IN_PLANE = ["chi_1_in_plane", "chi_2_in_plane"]
PERP = ["chi_in_plane_total", "chi_2_in_plane", "phi_s2", "chi_1_in_plane", "phi_12"]
EXPECTED = dict(
    aligned=["chi_1", "chi_2", "phi_12", "phi_jl"] + IN_PLANE,
    effective=["chi_eff", "chi_1", "chi_2", "phi_12", "phi_jl"] + IN_PLANE,
    spherical=["a_1", "a_2", "tilt_1", "tilt_2", "phi_12", "phi_jl"],
    aligned_perp=["chi_1", "chi_2", "phi_jl"] + PERP,
    effective_perp=["chi_eff", "chi_1", "chi_2", "phi_jl"] + PERP,
)
SPIN_KEYS = {key for keys in EXPECTED.values() for key in keys}


def _priors(spin, aligned=False):
    return set_spin_priors(BBHPriorDict(aligned_spin=aligned), spin, aligned=aligned)


def test_public_names():
    for name in cbc_spin_priors.__all__:
        assert hasattr(cbc_spin_priors, name)


@pytest.mark.parametrize("spin", SPINS)
def test_spin_keys(spin):
    priors = _priors(spin)
    assert {key for key in priors if key in SPIN_KEYS} == set(EXPECTED[spin])


@pytest.mark.parametrize("spin", ["effective", "effective_perp"])
def test_chi_2_choice(spin):
    assert type(_priors(spin)["chi_2"]) is cbc_spin_priors.Chi2


def test_effective_uses_mass_ratio_bounds():
    priors = BBHPriorDict()
    priors["mass_ratio"].minimum = 0.2
    set_spin_priors(priors, "effective")
    assert priors["chi_eff"].qmin == 0.2
    assert priors["chi_eff"].qmax == priors["mass_ratio"].maximum


@pytest.mark.parametrize("spin", SPINS)
def test_amax(spin):
    priors = set_spin_priors(BBHPriorDict(), spin, amax=0.5)
    samples, _ = convert_to_lal_binary_black_hole_parameters(priors.sample(200))
    assert np.all(samples["a_1"] <= 0.5 + 1e-12)
    assert np.all(samples["a_2"] <= 0.5 + 1e-12)


@pytest.mark.parametrize("spin", ["aligned", "effective"])
def test_aligned_spin_only(spin):
    priors = _priors(spin, aligned=True)
    assert not any(key in priors for key in IN_PLANE + ["chi_in_plane_total", "phi_12"])
    assert "chi_1" in priors and "chi_2" in priors


@pytest.mark.parametrize("spin", ["aligned_perp", "effective_perp"])
def test_perp_requires_precession(spin):
    with pytest.raises(ValueError, match="requires precessing spins"):
        _priors(spin, aligned=True)


def test_unknown_spin():
    with pytest.raises(ValueError, match="Unknown spin type"):
        _priors("isotropic")


@pytest.mark.parametrize("spin", SPINS)
def test_spin_distribution_correct(spin):
    """All five parameterisations give the same uniform-magnitude isotropic spins."""
    priors = _priors(spin)
    samples = priors.sample(4000, random_state=np.random.default_rng(42))
    samples, _ = convert_to_lal_binary_black_hole_parameters(samples)
    for ii in [1, 2]:
        samples[f"cos_tilt_{ii}"] = np.cos(samples[f"tilt_{ii}"])
    for key, bounds in [
        ("a_1", (0, AMAX)),
        ("a_2", (0, AMAX)),
        ("cos_tilt_1", (-1, 2)),
        ("cos_tilt_2", (-1, 2)),
        ("phi_12", (0, 2 * np.pi)),
    ]:
        ks_result = ks_1samp(np.mod(samples[key], 2 * np.pi) if key == "phi_12" else samples[key],
                             uniform(*bounds).cdf)
        assert ks_result.pvalue > THRESHOLD, f"Failed for {spin}, {key} ({ks_result})"


@pytest.mark.parametrize("spin", SPINS)
def test_unit_cube_maps_to_support(spin):
    """Every point of the unit cube maps to finite parameters with finite prior density."""
    priors = _priors(spin)
    # without the mass constraints, ln_prob would estimate their normalisation by sampling the whole prior
    for key in ["mass_1", "mass_2"]:
        del priors[key]
    keys = priors.non_fixed_keys
    uu = np.random.default_rng(1).uniform(0.02, 0.98, len(keys))
    theta = dict(zip(keys, priors.rescale(keys, uu)))
    assert np.all(np.isfinite(list(theta.values())))
    assert np.isfinite(priors.ln_prob({key: theta[key] for key in keys}))


@pytest.mark.parametrize("spin", ["effective_perp", "aligned_perp"])
def test_json_round_trip(spin, tmp_path):
    priors = PriorDict({key: prior for key, prior in _priors(spin).items() if key in SPIN_KEYS})
    priors.to_json(outdir=str(tmp_path), label=spin)
    loaded = PriorDict.from_json(str(tmp_path / f"{spin}_prior.json"))
    for key in priors:
        assert type(loaded[key]) is type(priors[key])
        assert repr(loaded[key]) == repr(priors[key])
