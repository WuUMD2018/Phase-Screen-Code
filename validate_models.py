"""Additional independent structure-function checks for supported variants."""
import argparse
import json
from pathlib import Path
import numpy as np
from phase_screens import (Spectrum, randomized_phase_screen, structure_function,
                           theoretical_structure, unresolved_structure_fraction)
from plot_results import add_plot_argument, plot_phase_screen


def run_checks(*, plot=True, plot_dir=Path("model_validation_plots")):
    """Return the model validation report and, by default, save phase plots."""
    wavelength, dz = 1.55e-6, 250.0
    lags = np.array([2, 4, 8, 12, 16])
    late_width = 2.34
    channel_spectrum = Spectrum(cn2=1e-16)
    cases = [
        ("random_shift_core", Spectrum(cn2=1e-14, inner_scale=0.02, outer_scale=0.15),
         0, 64, 0.002, 400),
        ("von_karman_hybrid", Spectrum(cn2=1e-14, model="von_karman",
                                       inner_scale=0.02, outer_scale=0.4), 4, 64, 0.002, 400),
        ("anisotropic_power_law", Spectrum(cn2=1e-14, model="power_law",
                                          alpha=3.5, mu_x=1, mu_y=2), 8, 64, 0.002, 400),
        ("modified_atmosphere_late_channel_window", channel_spectrum,
         4, 512, late_width / 512, 160),
    ]
    report = {}
    plot_screen = None
    for i, (name, spec, levels, n, dx, trials) in enumerate(cases):
        rng = np.random.default_rng(540 + i)
        target = np.stack([theoretical_structure(lags * dx, wavelength, dz, spec, axis=axis)
                           for axis in ("x", "y")], axis=1)
        # CW comments: Paulson et al. §2.2, Eqs. (11)–(15), compare ensemble
        # CW comments: phase increments against theory. Each row is one independent screen.
        samples = np.empty((trials, lags.size, 2))
        for trial in range(trials):
            phase = randomized_phase_screen(n, dx, wavelength, dz, spec, rng=rng,
                                            subharmonics=levels)
            samples[trial] = structure_function(phase, lags)
            if plot and trial == 0 and name == "modified_atmosphere_late_channel_window":
                # CW comments: Keep a single §3.1 hybrid realization from the tested
                # CW comments: ensemble for a visual PSD check without advancing its RNG.
                plot_screen = phase, dx, spec
        mean = samples.mean(axis=0)
        error = mean / target - 1
        record = {"independent_screens": trials, "n": n, "dx_m": dx,
                  "outer_scale_m": None if spec.model == "power_law" else spec.outer_scale,
                  "relative_errors_xy": error.tolist(),
                  "rms_relative_error": float(np.sqrt(np.mean(error**2)))}
        if spec.model == "power_law":
            # With uniform jitter near kappa=0, the increment second moment
            # converges and its fourth moment diverges for 3 < alpha < 4.
            # This seeded check uses a fixed tolerance and makes no standard
            # error claim. The high-frequency tail is deliberately uncorrected.
            tolerance = 0.15
            assert np.all(abs(error) < tolerance), (name, error)
            record["seeded_smoke_tolerance"] = tolerance
        else:
            se = samples.std(axis=0, ddof=1) / np.sqrt(trials)
            assert np.all(abs(error) < 5 * se / target + 0.025), (name, error)
            record["relative_standard_errors_xy"] = (se / target).tolist()
        report[name] = record
    # Keep the physical width constant while changing the sampled bandwidth.
    counts = (128, 256, 512)
    fractions = np.array([unresolved_structure_fraction(
        n, late_width / n, wavelength, dz, channel_spectrum) for n in counts])
    assert np.all((fractions >= 0) & (fractions <= 1))
    assert np.all(np.diff(fractions, axis=0) < 0), fractions
    assert np.max(fractions[-1]) < 0.001, fractions
    report["late_window_fft_cell_tail"] = {"width_m": late_width, "n": counts,
        "lag_one_increment_variance_fraction_xy": fractions.tolist()}
    report["status"] = "PASS"
    if plot_screen is not None:
        phase, dx, spec = plot_screen
        plot_phase_screen(phase, dx, wavelength, dz, spec, plot_dir)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("model_validation_results.json"))
    parser.add_argument("--plot-dir", type=Path, default=Path("model_validation_plots"))
    add_plot_argument(parser)
    args = parser.parse_args(argv)
    report = run_checks(plot=args.plot, plot_dir=args.plot_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
