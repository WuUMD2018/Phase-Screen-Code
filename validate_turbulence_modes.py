"""Monte Carlo phase-increment checks for the Normal and Advanced modes."""
import argparse
import hashlib
import json
from pathlib import Path
from secrets import randbits

import numpy as np
from threadpoolctl import threadpool_limits

from advanced_turbulence import TurbulenceSettings
from phase_screens import randomized_phase_screen, structure_function, theoretical_structure
from studio import StudioConfig


def validate_modes(realizations=30, *, seed=None):
    """Compare independent draws with each mode's analytic structure function."""
    if realizations < 30:
        raise ValueError("Use at least 30 independent realizations per mode.")
    seed = randbits(52) if seed is None else seed
    mode_seeds = np.random.SeedSequence(seed).spawn(2)
    n, width = 256, 2.0
    pitch = width / n
    lags = np.array([2, 4, 8, 16, 32])
    settings = TurbulenceSettings(alpha=3.4, inner_scale_mm=8, outer_scale_mm=350,
                                  major_scale=1.65, minor_scale=0.8)
    reports = {}
    for mode, option, mode_seed in zip(("normal", "advanced"), (None, settings), mode_seeds):
        channel = StudioConfig(resolution=n, cn2=1e-15, fwhm_nm=0, turbulence=option).channel_config()
        thickness = channel.length / channel.n_screens
        spectrum = channel.spectrum
        expected = np.stack([theoretical_structure(lags * pitch, channel.wavelength, thickness,
                                                     spectrum, axis=axis) for axis in ("x", "y")], axis=1)
        sample = []
        hashes = set()
        with threadpool_limits(limits=1, user_api="blas"):
            for child in mode_seed.spawn(realizations):
                phase = randomized_phase_screen(n, pitch, channel.wavelength, thickness, spectrum,
                                                rng=np.random.default_rng(child))
                hashes.add(hashlib.sha256(phase.tobytes()).digest())
                sample.append(structure_function(phase, lags))
        assert len(hashes) == realizations, f"Duplicate phase screen in {mode} mode"
        values = np.asarray(sample)
        measured = values.mean(axis=0)
        standard_error = values.std(axis=0, ddof=1) / np.sqrt(realizations)
        # CW comments: Compare the Monte Carlo mean with Eq. (11). A fixed allowance
        # CW comments: covers finite screen bandwidth and the rest is sampling error.
        assert np.all(abs(measured - expected) <= 5 * standard_error + 0.08 * expected), mode
        relative_error = measured / expected - 1
        reports[mode] = {
            "model": spectrum.model, "alpha": spectrum.alpha,
            "inner_scale_mm": spectrum.inner_scale * 1000, "outer_scale_mm": spectrum.outer_scale * 1000,
            "axis_scales": [spectrum.mu_x, spectrum.mu_y], "independent_realizations": realizations,
            "unique_phase_screens": len(hashes), "rms_relative_error": float(np.sqrt(np.mean(relative_error**2))),
            "tested_separations_m": (lags * pitch).tolist(),
            "theory_rad2_xy": expected.tolist(), "measured_rad2_xy": measured.tolist(),
            "standard_error_rad2_xy": standard_error.tolist(),
            "axis_y_over_axis_x_at_four_pixels": float(measured[1, 1] / measured[1, 0]),
        }
    assert reports["advanced"]["axis_y_over_axis_x_at_four_pixels"] > 1.4
    assert 0.6 < reports["normal"]["axis_y_over_axis_x_at_four_pixels"] < 1.5
    return {"status": "PASS", "seed": seed, "grid_size": n, "width_m": width, "modes": reports}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--realizations", type=int, default=30)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--out", type=Path, default=Path("turbulence_modes_validation.json"))
    args = parser.parse_args()
    report = validate_modes(args.realizations, seed=args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for mode, result in report["modes"].items():
        print(f"{mode.capitalize()}: {result['independent_realizations']} independent screens, "
              f"{result['rms_relative_error']:.2%} relative RMS against theory")
    print(f"PASS. Report saved to {args.out}")


if __name__ == "__main__":
    main()
