"""Numerical checks, run with: python validate.py --out validation_results.json

Tests include independent analytic and direct-integral references; no pytest
dependency. Monte Carlo checks use independent screens, not pixel counts as
independent replicates. Thresholds are smoke-test tolerances, not accuracy
guarantees for arbitrary channels.
"""
import argparse
import json
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch
import numpy as np
from phase_screens import (Spectrum, randomized_phase_screen, structure_function,
                           theoretical_structure, unresolved_noise_variances)
from plot_results import (BEAM_PLOT_FRACTIONS, add_plot_argument, plot_beam_snapshots,
                          plot_phase_screen)
from propagation import (ChannelConfig, beam_metrics, coordinates, fraunhofer,
                         fraunhofer_relative_error, fresnel_fft, gaussian_beam,
                         power, propagate_channel, scaled_fresnel)


def relative_l2(actual, expected):
    return float(np.linalg.norm(actual - expected) / np.linalg.norm(expected))


def check_channel_screens(wavelength):
    """Check screen location, phase sign, and per-slab reproducibility."""
    config = ChannelConfig(n=128, length=4000, n_screens=1, waist=0.035,
                           wavelength=wavelength, spectrum=Spectrum(cn2=0))
    tilt_x, tilt_y = 7e-6, -13e-6
    calls = []

    def tilt_screen(n, dx, wavelength, thickness, spectrum, **kwargs):
        assert thickness == config.length
        x = coordinates(n, dx)
        return 2 * np.pi / wavelength * (tilt_x * x[None, :] + tilt_y * x[:, None])

    def capture_screen(index, z, dx, phase):
        calls.append((index, z, dx, phase.shape))
        if phase.flags.writeable:
            phase.fill(0)

    with patch("propagation.randomized_phase_screen", side_effect=tilt_screen):
        result = propagate_channel(config, screen_callback=capture_screen, emit_warnings=False)

    expected_midpoint = config.length / 2
    [(index, z, pitch, shape)] = calls
    assert index == 0 and z == expected_midpoint and shape == (config.n, config.n)
    assert np.isclose(pitch * config.n, config.width_at(expected_midpoint), rtol=1e-13)
    final = result.history[-1]
    centroid_error = float(np.max(np.abs(
        np.array([final["centroid_x_m"], final["centroid_y_m"]])
        - expected_midpoint * np.array([tilt_x, tilt_y])
    )))
    assert centroid_error < 1e-8
    assert abs(final["power"] / result.history[0]["power"] - 1) < 1e-12

    config = replace(config, n=64, n_screens=3, spectrum=Spectrum(cn2=1e-16))

    def collect_last_screen(profile=None):
        captured = []

        def retain(index, z, dx, phase):
            if index == config.n_screens - 1:
                captured.append(phase.copy())

        propagate_channel(config, cn2_profile=profile, screen_callback=retain, emit_warnings=False)
        return captured[0]

    baseline = collect_last_screen()
    earlier_vacuum = collect_last_screen(
        lambda z: 0.0 if z < 2 * config.length / 3 else config.spectrum.cn2
    )
    assert np.any(baseline != 0)
    assert np.array_equal(baseline, earlier_vacuum)
    return {"midpoint_tilt_centroid_max_error_m": centroid_error,
            "screen_callback_mutation_isolated": True,
            "last_screen_unchanged_when_previous_slabs_are_vacuum": True}


def check_residual_chirp(wavelength):
    """A quadratic phase can fold to a constant on a coarse sampled grid."""
    n, waist, z = 128, 0.01, 100_000.0
    zr = np.pi * waist**2 / wavelength
    wavefront_radius = z + zr**2 / z
    midpoint_pitch = np.sqrt(2 * wavelength * wavefront_radius)
    config = ChannelConfig(n=n, length=2 * z, n_screens=1, waist=waist,
                           wavelength=wavelength, spectrum=Spectrum(cn2=0))

    def plateau_width(position):
        return 0.12 if position == 0 else n * midpoint_pitch

    def expanding_width(position):
        return 0.12 if position == 0 else n * midpoint_pitch * position / z

    plateau = propagate_channel(config, width_profile=plateau_width, emit_warnings=False)
    expanding = propagate_channel(config, width_profile=expanding_width, emit_warnings=False)
    plateau_cycles = plateau.diagnostics[-1]["residual_chirp_cycles_per_pixel"]
    expanding_cycles = expanding.diagnostics[-1]["residual_chirp_cycles_per_pixel"]
    assert plateau_cycles > 0.5
    assert any(message.startswith("Residual scaling chirp") for message in plateau.warnings)
    assert expanding_cycles < 0.5
    assert not any(message.startswith("Residual scaling chirp") for message in expanding.warnings)
    plateau_analytic = gaussian_beam(n, midpoint_pitch, wavelength, waist, z_from_waist=2 * z)
    plateau_error = relative_l2(abs(plateau.field)**2, abs(plateau_analytic)**2)
    analytic = gaussian_beam(n, 2 * midpoint_pitch, wavelength, waist, z_from_waist=2 * z)
    err = relative_l2(expanding.field, analytic)
    assert err < 1e-9, err
    return {"plateau_residual_cycles_per_pixel": plateau_cycles,
            "plateau_aliasing_warning_emitted": True,
            "plateau_intensity_relative_l2": plateau_error,
            "expanding_residual_cycles_per_pixel": expanding_cycles,
            "expanding_complex_field_relative_l2": err}


def run_checks(trials=400, *, plot=True, plot_dir=Path("validation_plots")):
    """Run numerical validation and save figures. Set plot=False to skip figures."""
    if trials < 2:
        raise ValueError("Validation requires at least two independent screens")
    report = {}
    n, wavelength, waist = 256, 1.55e-6, 0.025
    dx, dxout, distance = 0.24 / n, 0.48 / n, 2500.0
    u = gaussian_beam(n, dx, wavelength, waist)
    actual = scaled_fresnel(u, wavelength, dx, dxout, distance)
    theory = gaussian_beam(n, dxout, wavelength, waist, z_from_waist=distance)
    err = relative_l2(actual, theory)
    assert err < 1e-6, f"Scaled Gaussian complex-field error {err}"
    report["scaled_gaussian_complex_field_relative_l2"] = err
    perr = abs(power(actual, dxout) / power(u, dx) - 1)
    assert perr < 1e-12
    report["scaled_power_relative_error"] = perr
    back = scaled_fresnel(actual, wavelength, dxout, dx, -distance)
    report["roundtrip_relative_l2"] = relative_l2(back, u)
    assert report["roundtrip_relative_l2"] < 1e-12

    # A tilted beam tests frequency/coordinate signs independently of widths.
    tx, ty = 12e-6, -8e-6
    tilted = gaussian_beam(n, dx, wavelength, waist, tilt=(tx, ty))
    tilted_out = scaled_fresnel(tilted, wavelength, dx, dxout, distance)
    met = beam_metrics(tilted_out, dxout)
    cerr = max(abs(met["centroid_x_m"] - tx * distance),
               abs(met["centroid_y_m"] - ty * distance))
    assert cerr < 1e-6, cerr
    report["tilted_beam_centroid_error_m"] = cerr

    # Direct 2-D Fresnel quadrature on a deliberately small, well-sampled grid.
    nd, d1, d2, z = 64, 0.004, 0.003, 800.0
    ud = gaussian_beam(nd, d1, wavelength, 0.02)
    x1, x2 = coordinates(nd, d1), coordinates(nd, d2)
    kernel = np.exp(1j * np.pi * (x2[:, None] - x1[None, :])**2 / (wavelength * z))
    direct = kernel @ ud @ kernel.T * d1**2 / (1j * wavelength * z)
    scaled = scaled_fresnel(ud, wavelength, d1, d2, z)
    # Compare the well-supported interior; the scaled FFT has finite periodic boundaries.
    interior = np.abs(x2) < 0.06
    sl = np.ix_(interior, interior)
    report["direct_fresnel_interior_relative_l2"] = relative_l2(scaled[sl], direct[sl])
    assert report["direct_fresnel_interior_relative_l2"] < 1e-3

    # Vacuum multi-screen channel must recover analytic propagation on its final grid.
    cfg = ChannelConfig(n=256, n_screens=20, length=10000, waist=waist,
                        spectrum=Spectrum(cn2=0), extra_spread_angle=0)
    channel = propagate_channel(cfg, emit_warnings=False)
    outdx = cfg.width_at(cfg.length) / cfg.n
    analytic = gaussian_beam(cfg.n, outdx, wavelength, waist, z_from_waist=cfg.length)
    report["vacuum_channel_complex_field_relative_l2"] = relative_l2(channel.field, analytic)
    assert report["vacuum_channel_complex_field_relative_l2"] < 1e-5
    report["vacuum_channel_warnings"] = channel.warnings
    report["screen_channel"] = check_channel_screens(wavelength)
    report["residual_chirp_aliasing"] = check_residual_chirp(wavelength)

    # CW comments: Paper §3.5 discusses propagation on expanding grids.
    # CW comments: Seven slabs put 20% between simulation planes and 50% on a screen.
    snapshot_cfg = replace(cfg, n=128, n_screens=7)
    vacuum_reference = propagate_channel(snapshot_cfg, emit_warnings=False)
    snapshot_channel = propagate_channel(snapshot_cfg, snapshot_fractions=BEAM_PLOT_FRACTIONS,
                                         emit_warnings=False)
    assert tuple(snapshot_channel.snapshots) == BEAM_PLOT_FRACTIONS
    assert np.array_equal(snapshot_channel.field, vacuum_reference.field)
    assert np.array_equal(snapshot_channel.snapshots[1.0][2], snapshot_channel.field)
    snapshot_errors = []
    for fraction, (z_m, pitch, field) in snapshot_channel.snapshots.items():
        assert z_m == fraction * snapshot_cfg.length
        assert np.isclose(pitch, snapshot_cfg.width_at(z_m) / snapshot_cfg.n)
        reference = gaussian_beam(snapshot_cfg.n, pitch, wavelength, waist, z_from_waist=z_m)
        field_error = relative_l2(field, reference)
        assert field_error < 1e-5, (fraction, field_error)
        snapshot_errors.append({"path_fraction": fraction, "z_m": z_m, "dx_m": pitch,
                                "analytic_complex_field_relative_l2": field_error})
    report["beam_snapshots"] = snapshot_errors

    # Independent one-step Fresnel FFT on its natural grid.
    natural, x_nat = fresnel_fft(u, wavelength, dx, distance)
    dx_nat = wavelength * distance / (n * dx)
    assert np.array_equal(x_nat, coordinates(n, dx_nat))
    natural_scaled = scaled_fresnel(u, wavelength, dx, dx_nat, distance)
    report["natural_grid_fresnel_complex_field_relative_l2"] = relative_l2(natural_scaled, natural)
    assert report["natural_grid_fresnel_complex_field_relative_l2"] < 1e-10

    # Parseval diagnostic must agree with explicitly propagating both fields.
    near_ff, _ = fraunhofer(u, wavelength, dx, distance)
    measured_error = relative_l2(near_ff, natural)
    estimated_error = fraunhofer_relative_error(u, wavelength, dx, distance)
    report["fraunhofer_complex_field_relative_error_at_2500_m"] = estimated_error
    report["fraunhofer_error_estimate_absolute_difference"] = abs(estimated_error - measured_error)
    assert report["fraunhofer_error_estimate_absolute_difference"] < 1e-12

    # A Gaussian Fraunhofer field has radius lambda*z/(pi*w0) exactly in that approximation.
    zfar = 1e6
    uff, xf = fraunhofer(u, wavelength, dx, zfar)
    dxf = xf[1] - xf[0]
    far_radius = wavelength * zfar / (np.pi * waist)
    measured = beam_metrics(uff, dxf)["radius_x_m"]
    report["fraunhofer_radius_relative_error"] = abs(measured / far_radius - 1)
    assert report["fraunhofer_radius_relative_error"] < 1e-8
    report["fraunhofer_power_relative_error"] = abs(power(uff, dxf) / power(u, dx) - 1)
    assert report["fraunhofer_power_relative_error"] < 1e-12

    # Fresnel intensity approaches Fraunhofer intensity at long distance.
    fres = scaled_fresnel(u, wavelength, dx, dxf, zfar)
    report["far_field_limit_intensity_relative_l2"] = relative_l2(abs(fres)**2, abs(uff)**2)
    assert report["far_field_limit_intensity_relative_l2"] < 1e-4

    # Monte Carlo: bounded-spectrum screen structure function vs Eq. (11).
    ns, ds, thickness = 64, 0.002, 250.0
    spec = Spectrum(cn2=1e-14, inner_scale=0.02, outer_scale=0.4)
    lags = np.array([1, 2, 4, 8, 12, 16])
    theory_s = theoretical_structure(lags * ds, wavelength, thickness, spec)
    refined_s = theoretical_structure(lags * ds, wavelength, thickness, spec,
                                      n_radial=16384, n_angle=768)
    report["theory_quadrature_relative_change"] = float(np.max(abs(refined_s / theory_s - 1)))
    assert report["theory_quadrature_relative_change"] < 1e-4
    rng = np.random.default_rng(98421)
    samples = []
    for _ in range(trials):
        phi = randomized_phase_screen(ns, ds, wavelength, thickness, spec,
                                       subharmonics=4, rng=rng)
        samples.append(structure_function(phi, lags).mean(axis=1))
    samples = np.asarray(samples)
    avg, se = samples.mean(axis=0), samples.std(axis=0, ddof=1) / np.sqrt(trials)
    relative_error = avg / theory_s - 1
    assert np.all(abs(avg - theory_s) < 5 * se + 0.025 * theory_s), relative_error
    report["phase_structure_monte_carlo"] = {
        "independent_screens": trials, "separation_m": (lags * ds).tolist(),
        "theory_rad2": theory_s.tolist(), "mean_rad2": avg.tolist(),
        "standard_error_rad2": se.tolist(), "relative_error": relative_error.tolist(),
        "relative_rms_error": float(np.sqrt(np.mean(relative_error**2)))}

    # Variance normalization: same random draws, twice thickness -> sqrt(2) phase.
    p1 = randomized_phase_screen(ns, ds, wavelength, thickness, spec, rng=np.random.default_rng(31))
    p2 = randomized_phase_screen(ns, ds, wavelength, 2 * thickness, spec, rng=np.random.default_rng(31))
    report["thickness_scaling_relative_l2"] = relative_l2(p2, np.sqrt(2) * p1)
    assert report["thickness_scaling_relative_l2"] < 1e-12
    zero = randomized_phase_screen(ns, ds, wavelength, thickness, replace(spec, cn2=0))
    assert np.count_nonzero(zero) == 0

    # Independent inertial-range normalization cross-check.
    ps = Spectrum(cn2=1e-14, model="power_law")
    rho = np.array([0.001, 0.01, 0.1])
    r0 = (0.423 * (2 * np.pi / wavelength)**2 * ps.cn2 * thickness)**(-3 / 5)
    target = 6.88 * (rho / r0)**(5 / 3)
    ptheory = theoretical_structure(rho, wavelength, thickness, ps)
    report["kolmogorov_constant_relative_difference"] = float(np.max(abs(ptheory / target - 1)))
    assert report["kolmogorov_constant_relative_difference"] < 0.005

    # Optional whitening: finite/nonnegative, isotropy and quadrature convergence.
    vx, vy = unresolved_noise_variances(ns, ds, wavelength, thickness, ps)
    vx2, vy2 = unresolved_noise_variances(ns, ds, wavelength, thickness, ps,
                                        n_angle=512, n_radial=4096)
    report["white_noise_variance_rad2"] = [vx, vy]
    report["white_noise_quadrature_relative_change"] = max(abs(vx2 / vx - 1), abs(vy2 / vy - 1))
    assert vx >= 0 and vy >= 0 and abs(vx / vy - 1) < 1e-8
    assert report["white_noise_quadrature_relative_change"] < 0.02
    white_screen = randomized_phase_screen(ns, ds, wavelength, thickness, ps,
                                           whiten=True, rng=np.random.default_rng(9))
    assert np.isfinite(white_screen).all()
    assert abs(white_screen.mean()) < 1e-12
    if plot:
        # CW comments: Paper §2.2 shows a screen realization and validates D_phi.
        # CW comments: Use a wider window to expose the inertial band in the PSD plot.
        display_n, display_dx = 512, 2.34 / 512
        display_spectrum = Spectrum(cn2=spec.cn2, inner_scale=0.01, outer_scale=30.0)
        screen = randomized_phase_screen(display_n, display_dx, wavelength, thickness,
                                         display_spectrum, rng=np.random.default_rng(98421))
        plot_phase_screen(screen, display_dx, wavelength, thickness, display_spectrum, plot_dir)
        plot_beam_snapshots(snapshot_channel.snapshots, plot_dir, title="Vacuum validation beam")
    report["status"] = "PASS"
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=400)
    parser.add_argument("--out", type=Path, default=Path("validation_results.json"))
    parser.add_argument("--plot-dir", type=Path,
                        help="Plot directory (defaults to validation_plots beside --out)")
    add_plot_argument(parser)
    args = parser.parse_args()
    plot_dir = args.out.parent / "validation_plots" if args.plot_dir is None else args.plot_dir
    report = run_checks(args.trials, plot=args.plot, plot_dir=plot_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
