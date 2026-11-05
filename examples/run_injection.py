#!/usr/bin/env python
"""
Analyse a simulated binary black hole signal in simulated Gaussian noise with
dynesty, using one of the spin parameterisations in cbc_spin_priors.SPINS.

usage:
    python run_injection.py KIND SPIN SAMPLE --naccept N --nact N [options]

KIND is "bbh" (GravitationalWaveTransient, IMRPhenomXPHM) or "relbin"
(relative binning, IMRPhenomXP), SPIN is one of cbc_spin_priors.SPINS and
SAMPLE is "acceptance-walk" or "act-walk". The distance-marginalisation lookup
table {KIND}_lookup.npz is read from (or written to) the working directory.
For example,

    python run_injection.py bbh effective_perp act-walk --naccept 5 --nact 3 \
        --npool 16 --restart-time 7200

and a quick, unconverged test is

    python run_injection.py bbh aligned_perp acceptance-walk --naccept 5 --nact 3 \
        --nlive 100 --npool 4
"""

import argparse
import signal

import numpy as np

from bilby.core.prior import Gaussian, Sine, Uniform
from bilby.core.sampler import run_sampler
from bilby.core.utils import random
from bilby.gw.conversion import (
    component_masses_to_chirp_mass,
    convert_to_lal_binary_black_hole_parameters,
    generate_all_bbh_parameters,
)
from bilby.gw.detector import InterferometerList
from bilby.gw.detector.calibration import CubicSpline
from bilby.gw.likelihood import (
    GravitationalWaveTransient,
    RelativeBinningGravitationalWaveTransient,
)
from bilby.gw.prior import BBHPriorDict
from bilby.gw.result import CBCResult
from bilby.gw.source import lal_binary_black_hole, lal_binary_black_hole_relative_binning
from bilby.gw.waveform_generator import WaveformGenerator

from cbc_spin_priors import AMAX, SPINS, set_spin_priors

EXIT_CODE = 130


def run(sample, naccept, nact, spin="effective", aligned=False, kind="relbin", *, args):
    ifos, waveform_generator, injection_parameters = setup_data(kind)

    priors = BBHPriorDict(aligned_spin=aligned)

    priors["H1_time"] = Uniform(
        minimum=injection_parameters["H1_time"] - 3e-3,
        maximum=injection_parameters["H1_time"] + 3e-3,
        name="H1_time",
        latex_label="$t_H$",
        unit="$s$",
    )

    set_spin_priors(priors, spin, amax=AMAX, aligned=aligned)

    del priors["ra"], priors["dec"]
    priors["zenith"] = Sine(name="zenith", latex_label="$\\theta$")
    priors["azimuth"] = Uniform(
        name="azimuth", minimum=0, maximum=2 * np.pi, latex_label="$\\phi$", boundary="periodic"
    )

    if kind == "relbin":
        del priors["mass_1"], priors["mass_2"]
        chirp_mass = (1.3 * 1.5)**0.6 / (2.8)**0.2
        priors["chirp_mass"].minimum = chirp_mass - 1e-3
        priors["chirp_mass"].maximum = chirp_mass + 1e-3
    else:
        priors["delta_phase"] = priors.pop("phase")

    for ifo in ifos:
        det = ifo.name
        ifo.calibration_model = CubicSpline(
            prefix=f"recalib_{det}_",
            n_points=10,
            minimum_frequency=ifo.minimum_frequency,
            maximum_frequency=ifo.maximum_frequency,
        )
        freqs = np.logspace(
            np.log10(ifo.minimum_frequency),
            np.log10(ifo.maximum_frequency),
            10,
        )
        for ii in range(10):
            priors[f"recalib_{det}_amplitude_{ii}"] = Gaussian(mu=0, sigma=0.1, boundary="reflective")
            priors[f"recalib_{det}_phase_{ii}"] = Gaussian(mu=0, sigma=0.1, boundary="reflective")
            priors[f"recalib_{det}_frequency_{ii}"] = freqs[ii]

    if kind == "relbin":
        fiducial_parameters = injection_parameters.copy()
        m1 = fiducial_parameters.pop("mass_1")
        m2 = fiducial_parameters.pop("mass_2")
        fiducial_parameters["chirp_mass"] = component_masses_to_chirp_mass(m1, m2)
        fiducial_parameters["mass_ratio"] = m2 / m1
        for ifo in ifos:
            det = ifo.name
            freqs = np.logspace(
                np.log10(ifo.minimum_frequency),
                np.log10(ifo.maximum_frequency),
                10,
            )
            for ii in range(10):
                fiducial_parameters[f"recalib_{det}_amplitude_{ii}"] = 0.0
                fiducial_parameters[f"recalib_{det}_phase_{ii}"] = 0.0
                fiducial_parameters[f"recalib_{det}_frequency_{ii}"] = freqs[ii]
        cls_ = RelativeBinningGravitationalWaveTransient
        kwargs = dict(
            phase_marginalization=True,
            fiducial_parameters=fiducial_parameters,
        )
    else:
        cls_ = GravitationalWaveTransient
        kwargs = dict(
            phase_marginalization=False,
        )

    likelihood = cls_(
        interferometers=ifos,
        waveform_generator=waveform_generator,
        priors=priors,
        distance_marginalization=True,
        distance_marginalization_lookup_table=f"{kind}_lookup.npz",
        reference_frame="H1L1",
        time_reference="H1",
        **kwargs,
    )

    if sample == "acceptance-walk":
        full_label = f"{kind}_{spin}_acceptance_{naccept}"
    elif sample == "act-walk":
        full_label = f"{kind}_{spin}_act_{nact}"
    else:
        raise ValueError(f"Unknown sample method: {sample}")

    # Periodically checkpoint and exit with EXIT_CODE so HTCondor can save the
    # checkpoint and restart the job. Bilby only installs its SIGALRM handler
    # while sampling, so ignore the alarm during setup and post-processing.
    if args.restart_time > 0:
        signal.signal(signal.SIGALRM, signal.SIG_IGN)
        signal.alarm(args.restart_time)

    run_sampler(
        likelihood=likelihood,
        priors=priors,
        sampler="dynesty",
        sample=sample,
        naccept=naccept,
        nact=nact,
        npoints=args.nlive,
        walks=args.walks,
        npool=args.npool,
        injection_parameters=injection_parameters,
        outdir=args.outdir,
        label=full_label,
        rstate=random.rng,
        print_method="interval-30",
        exit_code=EXIT_CODE,
        conversion_function=generate_all_bbh_parameters,
        result_class=CBCResult,
        save="hdf5",
    )


def setup_data(kind):
    if kind == "relbin":
        duration = 256.0
        sampling_frequency = 2048.0
        minimum_frequency = 20
        waveform_arguments = dict(
            waveform_approximant="IMRPhenomXP",
            reference_frequency=50.0,
            minimum_frequency=20.0,
            fiducial=1,
        )
        fdsm = lal_binary_black_hole_relative_binning
    else:
        duration = 4.0
        sampling_frequency = 1024.0
        waveform_arguments = dict(
            waveform_approximant="IMRPhenomXPHM",
            reference_frequency=50.0,
            minimum_frequency=20.0,
            catch_waveform_errors=True,
        )
        fdsm = lal_binary_black_hole

    waveform_generator = WaveformGenerator(
        duration=duration,
        sampling_frequency=sampling_frequency,
        frequency_domain_source_model=fdsm,
        parameter_conversion=convert_to_lal_binary_black_hole_parameters,
        waveform_arguments=waveform_arguments,
    )

    injection_parameters = INJECTION_PARAMETERS[kind].copy()

    ifos = InterferometerList(["H1", "L1"])
    ifos.set_strain_data_from_power_spectral_densities(
        sampling_frequency=sampling_frequency,
        duration=duration,
        start_time=injection_parameters["geocent_time"] - 2,
    )

    ifos.inject_signal(
        waveform_generator=waveform_generator, parameters=injection_parameters
    )

    time_delay = ifos[0].time_delay_from_geocenter(
        injection_parameters["ra"],
        injection_parameters["dec"],
        injection_parameters["geocent_time"],
    )

    injection_parameters["H1_time"] = (
        injection_parameters["geocent_time"] + time_delay
    )
    return ifos, waveform_generator, injection_parameters


COMMON_INJECTION = dict(
    tilt_1=0.5,
    tilt_2=1.0,
    phi_12=1.7,
    phi_jl=0.3,
    theta_jn=0.4,
    psi=2.659,
    phase=1.3,
    geocent_time=1126259642.413,
    ra=1.375,
    dec=-1.2108,
)
INJECTION_PARAMETERS = dict(
    relbin=dict(
        mass_1=1.5,
        mass_2=1.3,
        a_1=0.03,
        a_2=0.01,
        luminosity_distance=50.0,
        **COMMON_INJECTION,
    ),
    bbh=dict(
        mass_1=36.0,
        mass_2=29.0,
        a_1=0.4,
        a_2=0.3,
        luminosity_distance=2000.0,
        **COMMON_INJECTION,
    ),
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("kind")
    parser.add_argument("spin", choices=SPINS)
    parser.add_argument("sample", choices=["acceptance-walk", "act-walk"])
    parser.add_argument("--naccept", type=int, required=True)
    parser.add_argument("--nact", type=int, required=True)
    parser.add_argument("--outdir", default="outdir")
    parser.add_argument("--nlive", type=int, default=1000)
    parser.add_argument("--walks", type=int, default=1000)
    parser.add_argument("--npool", type=int, default=16)
    parser.add_argument("--aligned", action="store_true")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--restart-time", type=int, default=0,
        help="Seconds after sampler start-up before checkpointing and exiting with "
             f"code {EXIT_CODE} (0 to disable)",
    )

    args = parser.parse_args()

    random.seed(args.seed)
    run(
        sample=args.sample,
        spin=args.spin,
        naccept=args.naccept,
        nact=args.nact,
        kind=args.kind,
        aligned=args.aligned,
        args=args,
    )


if __name__ == "__main__":
    main()

