"""Save headless Matplotlib figures for the simulator and its validation runs."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from phase_screens import Spectrum


BEAM_PLOT_FRACTIONS = (0.2, 0.5, 1.0)


def _plot_bool(value):
    value = value.lower()
    if value in ("true", "false"):
        return value == "true"
    raise argparse.ArgumentTypeError("expected True or False")


def add_plot_argument(parser):
    """Add default-on plotting with either --plot False or --no-plot to disable it."""
    options = parser.add_mutually_exclusive_group()
    options.add_argument("--plot", metavar="BOOL", type=_plot_bool, default=True,
                         help="Save diagnostic figures (default True)")
    options.add_argument("--no-plot", dest="plot", action="store_false",
                         help="Skip diagnostic figures")


def _figure(*, figsize):
    try:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
    except ImportError as error:
        raise RuntimeError("Install the plotting dependency with python -m pip install matplotlib") from error
    figure = Figure(figsize=figsize, layout="constrained")
    FigureCanvasAgg(figure)
    return figure


def _extent(n, dx):
    return [(-(n // 2) - 0.5) * dx, (n - n // 2 - 0.5) * dx] * 2


def _radial_periodogram(phase, dx):
    """Return log-binned frequency and angular-mean 2-D PSD per (cycles/m)^2."""
    n = phase.shape[0]
    df, nyquist = 1 / (n * dx), 1 / (2 * dx)
    freq = np.fft.fftfreq(n, dx)
    radial_frequency = np.hypot(freq[:, None], freq[None, :])

    # CW comments: Paulson et al. §2.1, Eqs. (4)–(7) synthesize a nonperiodic screen.
    # CW comments: Hann tapering reduces leakage from the finite crop. Divide by mean(window**2)
    # CW comments: so the periodogram retains the phase PSD normalization.
    taper = np.hanning(n)
    window = taper[:, None] * taper[None, :]
    centered = phase - np.sum(phase * window) / np.sum(window)
    psd = abs(np.fft.fft2(centered * window))**2 * (dx / n)**2 / np.mean(window**2)

    # CW comments: Average Fourier pixels in annuli to retain a 2-D spectral density.
    # CW comments: Integrating rings would add one to the power-law exponent.
    # CW comments: Complete annuli extend up to the axis Nyquist frequency.
    valid = (radial_frequency > 0) & (radial_frequency <= nyquist)
    frequencies = radial_frequency[valid]
    edges = np.geomspace(df * (1 - 1e-10), nyquist * (1 + 1e-10), 45)
    bin_index = np.searchsorted(edges, frequencies, side="right") - 1
    counts = np.bincount(bin_index, minlength=edges.size - 1)
    sum_freq = np.bincount(bin_index, weights=frequencies, minlength=counts.size)
    sum_psd = np.bincount(bin_index, weights=psd[valid], minlength=counts.size)
    populated = counts > 0
    counts = counts[populated]
    return sum_freq[populated] / counts, sum_psd[populated] / counts, counts


def _model_psd(frequency, wavelength, thickness, spectrum):
    angles = np.linspace(0, 2 * np.pi, 128, endpoint=False)
    kappa = (2 * np.pi * frequency)[:, None]
    model = spectrum.psd(kappa * np.cos(angles), kappa * np.sin(angles)).mean(axis=1)
    # CW comments: Eq. (11) gives S_phi = 2π k² Δz Φ_n in angular-frequency coordinates.
    # CW comments: Multiply by the Jacobian d²κ/d²f = (2π)² to plot in cycles/m.
    return (2 * np.pi)**3 * (2 * np.pi / wavelength)**2 * thickness * model


def plot_phase_screen(phase, dx, wavelength, thickness, spectrum, output_dir, *, z_m=None):
    """Write the unwrapped phase image and its radial phase PSD as PNGs."""
    phase = np.asarray(phase)
    if phase.ndim != 2 or phase.shape[0] != phase.shape[1] or phase.shape[0] < 8:
        raise ValueError("Expected a square phase screen with at least 8 points per axis")
    if not np.all(np.isfinite(phase)) or not all(
        np.isfinite(value) and value > 0 for value in (dx, wavelength, thickness)
    ):
        raise ValueError("Expected finite phase samples and positive finite dx, wavelength and thickness")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    image_path, spectrum_path = output / "phase_screen.png", output / "phase_spectrum.png"
    n = phase.shape[0]

    fig = _figure(figsize=(6.3, 5.2))
    ax = fig.subplots()
    span = max(float(np.max(np.abs(phase))), np.finfo(float).eps)
    im = ax.imshow(phase, extent=_extent(n, dx), origin="lower", cmap="RdBu_r",
                   vmin=-span, vmax=span, interpolation="nearest")
    position = "" if z_m is None else f" at z = {z_m:g} m"
    ax.set(xlabel="x (m)", ylabel="y (m)", title=f"Phase screen{position}")
    fig.colorbar(im, ax=ax, label="Unwrapped phase (rad)")
    fig.savefig(image_path, dpi=160)

    fig = _figure(figsize=(8.4, 5.4))
    ax = fig.subplots()
    frequency, estimate, counts = _radial_periodogram(phase, dx)
    df, nyquist = 1 / (n * dx), 1 / (2 * dx)
    xmin, xmax = df * 0.8, nyquist * 1.12
    fit_min, fit_max = 3 * df, 0.65 * nyquist
    outer = inner = None
    if spectrum.model != "power_law":
        # CW comments: The Andrews bump model cutoffs follow Eqs. (9)–(10).
        # CW comments: The markers are κ_l/(2π) and κ_0/(2π) = 1/L_0 in cycles/m.
        # CW comments: Both markers may lie outside the band resolved by the screen FFT.
        outer = 1 / spectrum.outer_scale
        inner = spectrum.inner_wavenumber / (2 * np.pi)
        xmin, xmax = min(xmin, outer / 3), max(xmax, 2 * inner)
        fit_min, fit_max = max(fit_min, 3 * outer), min(fit_max, inner / 2)

    model_f = np.geomspace(xmin, xmax, 512)
    model = _model_psd(model_f, wavelength, thickness, spectrum)
    model_ok = np.isfinite(model) & (model > 0)
    good = np.isfinite(estimate) & (estimate > 0)
    if np.any(good):
        # CW comments: Keep the plotted model tail within one decade of the measured floor.
        model_ok &= model >= 0.1 * np.min(estimate[good])
    if np.any(model_ok):
        ax.loglog(model_f[model_ok], model[model_ok], color="#1d567e", lw=1.8,
                  label=f"Input model (inertial exponent −{spectrum.alpha:.2f})")
    if np.any(good):
        ax.loglog(frequency[good], estimate[good], "o-", color="#bc4b35", lw=1.0,
                  markersize=3.0, label="Screen estimate (Hann taper)")

    # CW comments: Fit populated bins between model roll-offs and away from the lowest FFT bins.
    # CW comments: Tapering and finite screen width affect the low bins most strongly.
    fit = good & (counts >= 6) & (frequency >= fit_min) & (frequency <= fit_max)
    if np.count_nonzero(fit) >= 4 and frequency[fit][-1] / frequency[fit][0] >= 2:
        log_frequency, log_psd = np.log10(frequency[fit]), np.log10(estimate[fit])
        slope, intercept = np.polyfit(log_frequency, log_psd, 1)
        ax.loglog(frequency[fit], 10**(intercept + slope * log_frequency), "--",
                  color="#151719", lw=2.0, label=f"Inertial-range fit (slope {slope:.2f})")
    elif np.any(good):
        ax.text(0.02, 0.03, "Limited resolved inertial range for fitting",
                transform=ax.transAxes, fontsize=9, color="#3e4b5a")
    else:
        ax.text(0.5, 0.45, "Zero phase variance", ha="center", transform=ax.transAxes)

    if outer is not None:
        ax.axvline(outer, color="#67803a", ls="--", lw=1.2,
                   label=f"Outer roll-off ({outer:.3g} cycles/m)")
        ax.axvline(inner, color="#8b588f", ls="--", lw=1.2,
                   label=f"Inner roll-off ({inner:.3g} cycles/m)")
    ax.axvline(df, color="#778392", ls=":", lw=1.1, label="First FFT frequency")
    ax.axvline(nyquist, color="#778392", ls="-.", lw=1.1, label="Axis Nyquist")
    ax.set(xscale="log", yscale="log", xlim=(xmin, xmax),
           xlabel="Radial spatial frequency (cycles/m)", ylabel="2-D phase PSD (rad² m²)",
           title="Phase-screen spatial power spectrum")
    ax.grid(True, which="both", alpha=0.17)
    ax.legend(fontsize=8, loc="best")
    fig.savefig(spectrum_path, dpi=170)
    return image_path, spectrum_path


def plot_beam_snapshots(snapshots, output_dir, *, title="Beam amplitude and phase along the propagation path"):
    """Write amplitude and wrapped phase at fraction -> (z_m, dx_m, field) checkpoints."""
    if not snapshots:
        raise ValueError("At least one beam snapshot is required")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "beam_snapshots.png"
    fig = _figure(figsize=(3.55 * len(snapshots) + 1.2, 7.3))
    from matplotlib import colormaps
    from matplotlib.colors import PowerNorm
    from matplotlib.ticker import MaxNLocator

    axes = fig.subplots(2, len(snapshots), squeeze=False)
    peak = max(float(np.max(np.abs(frame[2]))) for frame in snapshots.values())
    amp_norm = PowerNorm(gamma=0.5, vmin=0, vmax=peak if peak > 0 else 1)
    phase_map = colormaps["twilight_shifted"].with_extremes(bad="#e6e8ec")
    for index, (fraction, (z_m, dx, field)) in enumerate(sorted(snapshots.items())):
        amp = np.abs(field)
        extent = _extent(field.shape[0], dx)
        amp_im = axes[0, index].imshow(amp, origin="lower", extent=extent, cmap="inferno",
                                      norm=amp_norm, interpolation="nearest")
        # CW comments: The input snapshots include the restored scaling chirp.
        # CW comments: Mask pixels below 1% of the local peak amplitude to show phase in the illuminated region.
        phase = np.ma.masked_where(amp <= 0.01 * amp.max(), np.angle(field))
        phase_im = axes[1, index].imshow(phase, origin="lower", extent=extent, cmap=phase_map,
                                        vmin=-np.pi, vmax=np.pi, interpolation="nearest")
        axes[0, index].set_title(f"{fraction:.0%} of path  ·  {z_m / 1000:.3g} km")
    for ax in axes.flat:
        ax.set(xlabel="x (m)", ylabel="y (m)")
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.yaxis.set_major_locator(MaxNLocator(5))
    fig.suptitle(title)
    fig.colorbar(amp_im, ax=list(axes[0]), label="Field amplitude (√P, common scale)", shrink=0.86)
    cb = fig.colorbar(phase_im, ax=list(axes[1]), label="Wrapped phase (rad)",
                     ticks=[-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi], shrink=0.86)
    cb.set_ticklabels(["−π", "−π/2", "0", "π/2", "π"])
    fig.supxlabel("Phase is gray below 1% of each plane’s peak amplitude.", fontsize=9)
    fig.savefig(path, dpi=170)
    return path


def _plot_receiver_summary(root):
    """Retain the original receiver and grid-size comparison for saved runs."""
    with np.load(root / "channel.npz") as data:
        x = data["x_m"]
        turbulent, vacuum = data["intensity"], abs(data["vacuum_field"])**2
    with (root / "beam_history.csv").open() as stream:
        history = list(csv.DictReader(stream))
    z = np.array([float(row["z_m"]) / 1000 for row in history])
    fig = _figure(figsize=(13, 4.3))
    axes = fig.subplots(1, 3)
    peak = max(turbulent.max(), vacuum.max())
    for ax, intensity, title in zip(axes[:2], (vacuum, turbulent), ("Vacuum", "Turbulence")):
        im = ax.imshow(intensity / peak, extent=_extent(x.size, x[1] - x[0]), origin="lower",
                       cmap="inferno", vmin=0, vmax=1)
        ax.set(xlabel="x (m)", ylabel="y (m)", title=title)
    fig.colorbar(im, ax=list(axes[:2]), label="Intensity / common peak", shrink=0.8)
    for key, label in [("radius_x_m", "2σ x radius"), ("radius_y_m", "2σ y radius")]:
        axes[2].plot(z, [float(row[key]) for row in history], label=label)
    axes[2].plot(z, [float(row["grid_width_m"]) / 2 for row in history], "--", label="Grid half-width")
    axes[2].set(xlabel="Distance (km)", ylabel="Length (m)", title="Expanding computation window")
    axes[2].grid(alpha=0.2)
    axes[2].legend(fontsize=8)
    fig.savefig(root / "propagation.png", dpi=180)
    fig.savefig(root / "propagation.pdf")


def main():
    parser = argparse.ArgumentParser(description="Regenerate plots from saved demo data")
    parser.add_argument("--input", type=Path, default=Path("results"))
    root = parser.parse_args().input
    if (root / "channel.npz").is_file():
        _plot_receiver_summary(root)
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    chosen_screen = root / "visualized_screen.npz"
    screen_path = chosen_screen if chosen_screen.is_file() else root / "screen_0000.npz"
    with np.load(screen_path) as screen:
        plot_phase_screen(screen["phase_rad"], float(screen["dx_m"]), config["wavelength"],
                          float(screen["thickness_m"]), Spectrum(**config["spectrum"]), root,
                          z_m=float(screen["z_m"]) if "z_m" in screen else None)
    if (root / "beam_snapshots.npz").is_file():
        with np.load(root / "beam_snapshots.npz") as data:
            snapshots = dict(zip(data["fractions"], zip(data["z_m"], data["dx_m"], data["fields"])))
            plot_beam_snapshots(snapshots, root)
    print(f"Figures saved in {root.resolve()}")


if __name__ == "__main__":
    main()
