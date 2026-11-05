#!/usr/bin/env python
"""
Draw from each spin parameterisation with the same BBH prior dict as
run_injection.py and check that they all describe the same spins: uniform
magnitudes up to AMAX and isotropic orientations.

usage:
    python draw_prior_samples.py [--n 20000] [--plot spin_priors.png]

Prints the KS p-value of each spin parameter against the expected uniform
distribution. With --plot, also histograms each parameter for every prior.
"""

import argparse

import numpy as np
from scipy.stats import ks_1samp, uniform

from bilby.gw.conversion import convert_to_lal_binary_black_hole_parameters
from bilby.gw.prior import BBHPriorDict

from cbc_spin_priors import AMAX, SPINS, set_spin_priors

EXPECTED = dict(
    a_1=(0, AMAX),
    a_2=(0, AMAX),
    cos_tilt_1=(-1, 2),
    cos_tilt_2=(-1, 2),
    phi_12=(0, 2 * np.pi),
)


def draw(spin, n, rng):
    priors = set_spin_priors(BBHPriorDict(), spin)
    samples, _ = convert_to_lal_binary_black_hole_parameters(priors.sample(n, random_state=rng))
    samples["phi_12"] = np.mod(samples["phi_12"], 2 * np.pi)
    for ii in [1, 2]:
        samples[f"cos_tilt_{ii}"] = np.cos(samples[f"tilt_{ii}"])
    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--plot", default=None, help="write histograms to this file")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    draws = {spin: draw(spin, args.n, rng) for spin in SPINS}

    print(f"KS p-values against the expected uniform distributions ({args.n} samples)")
    print(f"{'':16s}" + "".join(f"{key:>12s}" for key in EXPECTED))
    for spin, samples in draws.items():
        pvalues = [ks_1samp(samples[key], uniform(*bounds).cdf).pvalue for key, bounds in EXPECTED.items()]
        print(f"{spin:16s}" + "".join(f"{p:12.3f}" for p in pvalues))

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, len(EXPECTED), figsize=(3 * len(EXPECTED), 3), constrained_layout=True)
        for ax, (key, (lo, width)) in zip(axes, EXPECTED.items()):
            bins = np.linspace(lo, lo + width, 31)
            for spin, samples in draws.items():
                ls = "--" if spin.endswith("_perp") else "-"
                ax.hist(samples[key], bins=bins, density=True, histtype="step", ls=ls, label=spin)
            ax.axhline(1 / width, color="k", lw=0.8)
            ax.set_xlabel(key)
        axes[0].legend(fontsize=7)
        fig.savefig(args.plot, dpi=120)
        print("wrote", args.plot)


if __name__ == "__main__":
    main()
