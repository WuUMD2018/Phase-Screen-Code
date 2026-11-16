"""Run a multi-screen channel and optionally observe it farther downstream."""
import argparse
import csv
import json
from dataclasses import asdict, replace
from pathlib import Path
import warnings
import numpy as np
from phase_screens import Spectrum
from plot_results import (BEAM_PLOT_FRACTIONS, add_plot_argument, plot_beam_snapshots,
                          plot_phase_screen)
from propagation import (ChannelConfig, fraunhofer, fraunhofer_relative_error,
                         fresnel_fft, propagate_channel)


def write_csv(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=512)
    parser.add_argument("--screens", type=int, default=40)
    parser.add_argument("--length", type=float, default=10000)
    parser.add_argument("--wavelength", type=float, default=1.55e-6)
    parser.add_argument("--waist", type=float, default=0.03)
    parser.add_argument("--cn2", type=float, default=1e-16)
    parser.add_argument("--inner-scale", type=float, default=0.01)
    parser.add_argument("--outer-scale", type=float, default=30.0)
    parser.add_argument("--model", choices=["modified_atmosphere", "von_karman", "power_law"],
                        default="modified_atmosphere")
    parser.add_argument("--alpha", type=float, default=11 / 3)
    parser.add_argument("--mu-x", type=float, default=1.0)
    parser.add_argument("--mu-y", type=float, default=1.0)
    parser.add_argument("--subharmonics", type=int, default=4)
    parser.add_argument("--margin", type=float, default=6.0)
    parser.add_argument("--spread-angle", type=float, default=10e-6)
    parser.add_argument("--seed", type=int, default=20261115)
    parser.add_argument("--whiten", action="store_true",
                        help="Approximate subresolution correction; normally leave disabled")
    parser.add_argument("--save-screens", action="store_true")
    parser.add_argument("--observation-distance", type=float,
                        help="Distance in metres beyond the receiver for a Fresnel and Fraunhofer comparison")
    parser.add_argument("--out", type=Path, default=Path("results"))
    add_plot_argument(parser)
    args = parser.parse_args(argv)
    if args.observation_distance is not None and (
        not np.isfinite(args.observation_distance) or args.observation_distance <= 0
    ):
        parser.error("--observation-distance must be finite and positive")
    spectrum = Spectrum(cn2=args.cn2, model=args.model, inner_scale=args.inner_scale,
                        outer_scale=args.outer_scale, alpha=args.alpha,
                        mu_x=args.mu_x, mu_y=args.mu_y)
    cfg = ChannelConfig(n=args.n, n_screens=args.screens, length=args.length,
        wavelength=args.wavelength, waist=args.waist, spectrum=spectrum,
        subharmonics=args.subharmonics, margin=args.margin,
        extra_spread_angle=args.spread_angle, seed=args.seed, whiten=args.whiten)
    summary = run_demo(cfg, output_dir=args.out, plot=args.plot,
                       save_screens=args.save_screens, observation_distance=args.observation_distance)
    print(json.dumps(summary, indent=2))
    print(f"Saved data to {args.out.resolve()}")


def run_demo(config=None, *, output_dir=Path("results"), plot=True,
             save_screens=False, observation_distance=None):
    """Run the example and write data and PNG figures. Set plot=False for data only."""
    cfg = ChannelConfig() if config is None else config
    spectrum = cfg.spectrum
    if observation_distance is not None and (
        not np.isfinite(observation_distance) or observation_distance <= 0
    ):
        raise ValueError("observation_distance must be finite and positive")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    last_screen = None
    thickness = cfg.length / cfg.n_screens

    def save_screen(j, z, dx, phi):
        nonlocal last_screen
        if j == 0 or save_screens:
            np.savez_compressed(out / f"screen_{j:04d}.npz", phase_rad=phi,
                                z_m=z, dx_m=dx, thickness_m=thickness)
        if plot and j == cfg.n_screens - 1:
            last_screen = (z, dx, phi.copy())

    # CW comments: Paper §§2.1 and 3.5 use thin screens in a split-step channel.
    # CW comments: Capture the three physical planes without changing those slabs.
    result = propagate_channel(cfg, screen_callback=save_screen,
                               snapshot_fractions=BEAM_PLOT_FRACTIONS if plot else ())
    vacuum = propagate_channel(replace(cfg, spectrum=replace(spectrum, cn2=0)), emit_warnings=False)
    np.savez_compressed(out / "channel.npz", field=result.field,
        intensity=abs(result.field)**2, vacuum_field=vacuum.field, x_m=result.x_m,
        wavelength_m=cfg.wavelength, length_m=cfg.length)
    write_csv(out / "beam_history.csv", result.history)
    write_csv(out / "vacuum_history.csv", vacuum.history)
    write_csv(out / "screen_geometry.csv", result.screen_geometry)
    write_csv(out / "sampling_diagnostics.csv", result.diagnostics)
    (out / "config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")
    last = result.history[-1]
    summary = {"first_plane_width_m": cfg.width_at(0), "last_plane_width_m": cfg.width_at(cfg.length),
        "input_power": result.history[0]["power"], "output_power": last["power"],
        "final_radius_x_m": last["radius_x_m"], "final_radius_y_m": last["radius_y_m"],
        "final_centroid_x_m": last["centroid_x_m"], "final_centroid_y_m": last["centroid_y_m"],
        "max_edge_power_fraction": max(result.history[0]["edge_power_fraction"],
            max(row["edge_power_fraction"] for row in result.diagnostics)),
        "warnings": result.warnings, "vacuum_warnings": vacuum.warnings,
        "description": "One illustrative turbulence realization"}
    if observation_distance is not None:
        distance = observation_distance
        pitch = last["dx_m"]
        u_fresnel, x_observation = fresnel_fft(result.field, cfg.wavelength, pitch, distance)
        u_far, _ = fraunhofer(result.field, cfg.wavelength, pitch, distance)
        error = fraunhofer_relative_error(result.field, cfg.wavelength, pitch, distance)
        summary["observation"] = {"distance_beyond_receiver_m": distance,
            "output_pitch_m": cfg.wavelength * distance / (cfg.n * pitch),
            "fraunhofer_vs_fresnel_relative_complex_field_l2": error}
        np.savez_compressed(out / "observation.npz", fresnel_field=u_fresnel,
            fraunhofer_field=u_far, x_m=x_observation, distance_beyond_receiver_m=distance)
        if error > 0.05:
            warnings.warn(f"Dropping the source chirp changes the complex field by {error:.1%} "
                          "on the sampled observation grid. Use the Fresnel result for this distance.",
                          RuntimeWarning, stacklevel=2)
    if plot:
        # CW comments: Paper §2.2 compares phase structure functions with theory.
        # CW comments: For the PSD plot, the last screen resolves more low-frequency bins.
        z_m, dx_m, phase = last_screen
        plot_phase_screen(phase, dx_m, cfg.wavelength, thickness, spectrum, out, z_m=z_m)
        plot_beam_snapshots(result.snapshots, out)
        np.savez_compressed(out / "visualized_screen.npz", phase_rad=phase, z_m=z_m,
                            dx_m=dx_m, thickness_m=thickness, wavelength_m=cfg.wavelength)
        positions, pitches, fields = zip(*result.snapshots.values())
        np.savez_compressed(out / "beam_snapshots.npz", fractions=list(result.snapshots),
                            z_m=positions, dx_m=pitches, fields=np.stack(fields))
        summary["plots"] = ["phase_screen.png", "phase_spectrum.png", "beam_snapshots.png"]
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    main()
