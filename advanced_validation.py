"""Validate a configured elliptical phase spectrum on the channel's screen grid."""
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from phase_screens import randomized_phase_screen
from plot_results import _extent, _figure
from spectrum_band import band_width_profile, gaussian_wavelengths


_SAMPLES = 30
_SECTOR_DEG = 12


@dataclass(frozen=True)
class AdvancedValidation:
    screen_image: Path
    spectrum_image: Path
    warnings: tuple[str, ...]
    report: dict


@dataclass(frozen=True)
class _AxisBins:
    mask: np.ndarray
    indices: np.ndarray
    counts: np.ndarray
    frequency: np.ndarray
    model: np.ndarray

    def average(self, psd):
        sums = np.bincount(self.indices, weights=psd[self.mask], minlength=self.counts.size)
        populated = self.counts > 0
        return sums[populated] / self.counts[populated]


def _phase_psd(spectrum, fx, fy, wavelength, dz):
    k = 2 * np.pi / wavelength
    # CW comments: Paper Eq. (11) gives the 2-D PSD 2πk²ΔzΦn in angular spatial frequency.
    # CW comments: Multiply by (2π)² to express the spectral density per (cycles/m)².
    return (2 * np.pi)**3 * k**2 * dz * spectrum.psd(2 * np.pi * fx, 2 * np.pi * fy)


def _axis_bins(n, dx, wavelength, dz, spectrum):
    fx, fy = np.meshgrid(np.fft.fftfreq(n, dx), np.fft.fftfreq(n, dx), sparse=True)
    frequency = np.hypot(fx, fy)
    df, nyquist = 1 / (n * dx), 1 / (2 * dx)
    inside = (frequency > 0) & (frequency <= nyquist)
    tangent = np.tan(np.deg2rad(_SECTOR_DEG))
    edges = np.geomspace(df * (1 - 1e-10), nyquist * (1 + 1e-10), 49)
    theory = _phase_psd(spectrum, fx, fy, wavelength, dz)
    axes = []
    # CW comments: Paper §3.2 uses an elliptical spectrum in κx and κy. Keep the two axes
    # CW comments: separate and average theory over the same Fourier modes as the estimates.
    for sector in (abs(fy) <= tangent * abs(fx), abs(fx) <= tangent * abs(fy)):
        mask = inside & sector
        chosen_f = frequency[mask]
        indices = np.searchsorted(edges, chosen_f, side="right") - 1
        counts = np.bincount(indices, minlength=edges.size - 1)
        populated = counts > 0
        sum_f = np.bincount(indices, weights=chosen_f, minlength=counts.size)
        sum_model = np.bincount(indices, weights=theory[mask], minlength=counts.size)
        axes.append(_AxisBins(mask, indices, counts, sum_f[populated] / counts[populated],
                              sum_model[populated] / counts[populated]))
    return axes


def _plot_maps(frames, dx, z_m, destination):
    fig = _figure(figsize=(11, 4.0))
    axes = fig.subplots(1, len(frames), squeeze=False)[0]
    limit = max(max(float(np.max(abs(frame))) for frame in frames), np.finfo(float).eps)
    for i, (ax, phase) in enumerate(zip(axes, frames), 1):
        image = ax.imshow(phase, extent=_extent(phase.shape[0], dx), origin="lower",
                          vmin=-limit, vmax=limit, cmap="RdBu_r", interpolation="nearest")
        ax.set(xlabel="x (m)", ylabel="y (m)", title=f"Screen {i}")
    position = f"{z_m:,.0f}" if z_m >= 1000 else f"{z_m:.3g}"
    fig.suptitle(f"Independent realizations at the final screen plane ({position} m)")
    fig.colorbar(image, ax=list(axes), shrink=0.82, label="Unwrapped phase (rad)")
    fig.savefig(destination, dpi=140)


def _flag_spectrum(bins, mean, sem, comparison):
    # CW comments: A deviation needs practical size (>25%) and support from the 30-screen
    # CW comments: ensemble (>3 standard errors). Require three consecutive resolved bins.
    absolute_error = abs(mean - bins.model)
    flagged = comparison & (absolute_error > 0.25 * bins.model) & (absolute_error > 3 * sem)
    contiguous = np.diff(np.flatnonzero(bins.counts)) == 1
    sustained = bool(np.any(flagged[2:] & flagged[1:-1] & flagged[:-2] & contiguous[1:] & contiguous[:-1]))
    return sustained, int(np.count_nonzero(flagged))


def _plot_spectra(axis_data, spectrum, n, dx, destination):
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

    fig = _figure(figsize=(12, 4.8))
    axes = fig.subplots(1, 2)
    df, nyquist = 1 / (n * dx), 1 / (2 * dx)
    messages, reports = [], {}
    names = ("x / major axis", "y / minor axis")
    stretches = (spectrum.mu_x, spectrum.mu_y)
    for ax, (bins, samples), label, stretch in zip(axes, axis_data, names, stretches):
        mean = np.mean(samples, axis=0)
        sem = np.std(samples, axis=0, ddof=1) / np.sqrt(samples.shape[0])
        measured = np.isfinite(mean) & (mean > 0)
        outer = 1 / (spectrum.outer_scale * stretch)
        inner = spectrum.inner_wavenumber / (2 * np.pi * stretch)
        selected_counts = bins.counts[bins.counts > 0]
        # CW comments: Set comparison bins from the grid and model alone, including resolved
        # CW comments: scale turnovers. Exclude leakage-prone low bins, near-Nyquist bins,
        # CW comments: and theory more than ten decades below the eligible band's peak.
        candidate = ((selected_counts >= 8) & (bins.frequency >= 4 * df)
                     & (bins.frequency <= 0.9 * nyquist) & np.isfinite(bins.model))
        reference = np.max(bins.model[candidate], initial=0.0)
        comparison = candidate & (bins.model > 0) & (bins.model >= 1e-10 * reference)
        compare_f = bins.frequency[comparison]
        sustained_deviation, flagged_bins = _flag_spectrum(bins, mean, sem, comparison)
        if sustained_deviation:
            messages.append(f"{label}: PSD differs from the model by >25% and >3 standard errors "
                            "in at least three consecutive comparison bins.")
        if compare_f.size < 3:
            messages.append(f"{label}: too few resolved bins for a PSD deviation check.")
        fit = (measured & (bins.model > 0) & (selected_counts >= 4)
               & (bins.frequency >= max(3 * df, 3 * outer))
               & (bins.frequency <= min(0.65 * nyquist, inner / 3)))
        f_fit = bins.frequency[fit]
        measured_slope = model_slope = None
        robust_fit = f_fit.size >= 4 and f_fit[-1] >= 2 * f_fit[0]
        # CW comments: Display the common measured and modeled inertial fit bins only.
        # CW comments: Keep the complete PSD and the broader turnover checks in the report.
        visible = fit if robust_fit else (comparison & measured)
        if not np.any(visible):
            visible = measured & (bins.frequency >= 2 * df) & (bins.frequency <= 0.9 * nyquist)
        if not np.any(visible):
            visible = measured
        shown_f = bins.frequency[visible]
        if shown_f.size:
            padx = max(1.09, (shown_f[-1] / shown_f[0])**0.09)
            lower, upper = max(df, shown_f[0] / padx), min(nyquist, shown_f[-1] * padx)
        else:
            lower, upper = df, nyquist

        mean_label = f"Mean of {_SAMPLES} screens"
        model_label = "Model on same FFT modes"
        if robust_fit:
            logs = np.log10(f_fit)
            slope, intercept = np.polyfit(logs, np.log10(mean[fit]), 1)
            measured_slope = float(slope)
            model_slope = float(np.polyfit(logs, np.log10(bins.model[fit]), 1)[0])
            model_label = f"Model on same FFT modes (slope {model_slope:.2f})"
        else:
            diagnostic = "No resolved inertial fit · showing sampled diagnostics"
            if not shown_f.size:
                diagnostic = "No positive sampled PSD"
            ax.text(0.025, 0.04, diagnostic, va="bottom", transform=ax.transAxes, fontsize=8,
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "pad": 3})
            messages.append(f"{label}: too little resolved inertial range to fit a slope.")
        lower_ci = mean[visible] - 1.96 * sem[visible]
        upper_ci = mean[visible] + 1.96 * sem[visible]
        ax.loglog(shown_f, mean[visible], "o", color="#c26040", markeredgecolor="white", markeredgewidth=0.5,
                  ms=5.3, zorder=4, label=mean_label)
        model_visible = visible & np.isfinite(bins.model) & (bins.model > 0)
        ax.loglog(bins.frequency[model_visible], bins.model[model_visible], color="#274f77",
                  lw=2.0, label=model_label)
        if robust_fit:
            ax.loglog(f_fit, 10**(slope * logs + intercept), "--", color="#17202b", lw=1.35,
                      label=f"Measured fit (slope {measured_slope:.2f})")
        ax.fill_between(shown_f, lower_ci, upper_ci, where=lower_ci > 0, color="#c26040", alpha=0.19,
                        label="Pointwise ±1.96 SEM")
        for cutoff, cutoff_label in ((outer, "Outer"), (inner, "Inner")):
            if not df <= cutoff <= nyquist:
                messages.append(f"{label}: {cutoff_label.lower()} roll-off is outside the sampled band.")
        ax.set_title(f"{label}\nOuter {outer:.3g} · inner {inner:.3g} cycles/m", fontsize=11)
        ax.set(xlim=(lower, upper), xlabel="Spatial frequency (cycles/m)",
               ylabel="2-D phase PSD (rad² m²)")
        ax.margins(y=0.12)
        ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6, 8), numticks=12))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.3g}"))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.grid(which="both", alpha=0.15)
        if shown_f.size:
            ax.legend(fontsize=8, loc="best")
        reports[label] = {"outer_rolloff_cycles_m": float(outer), "inner_rolloff_cycles_m": float(inner),
                          "outer_rolloff_sampled": bool(df <= outer <= nyquist),
                          "inner_rolloff_sampled": bool(df <= inner <= nyquist),
                          "measured_slope": measured_slope, "model_fit_slope": model_slope,
                          "fit_band_cycles_m": [float(f_fit[0]), float(f_fit[-1])] if robust_fit else None,
                          "display_mode": "inertial_fit" if robust_fit else "resolved_diagnostics",
                          "display_limits_cycles_m": [float(lower), float(upper)],
                          "comparison_bins": int(compare_f.size),
                          "comparison_band_cycles_m": [float(compare_f[0]), float(compare_f[-1])] if compare_f.size else None,
                          "sustained_psd_deviation": sustained_deviation, "flagged_bins": flagged_bins,
                          "frequency_cycles_m": bins.frequency.tolist(),
                          "mean_psd_rad2_m2": mean.tolist(), "psd_sem_rad2_m2": sem.tolist(),
                          "model_psd_rad2_m2": bins.model.tolist()}
    fig.suptitle(f"Directional phase PSD within ±{_SECTOR_DEG}° of each axis · nominal slope −{spectrum.alpha:.3g}")
    fig.supxlabel("Fits use only the displayed common bins. Roll-off frequencies and full PSD are recorded in the validation report.",
                  fontsize=8)
    fig.savefig(destination, dpi=150)
    return reports, messages


def validate_advanced(config, settings, output_dir, *, seed: int | None = None, cancel_event=None):
    """Generate 30 independent screens and compare their measured PSD with theory.

    The final-screen width follows the channel's common wavelength grid. The
    estimator uses the central wavelength and the final slab's full thickness.
    With no validation seed, use fresh system entropy.
    """
    from studio import _available_memory

    def check_cancel():
        if cancel_event is not None and cancel_event.is_set():
            raise RuntimeError("Phase-screen validation was cancelled.")

    check_cancel()
    channel = config.channel_config()
    n, wavelength = config.resolution, channel.wavelength
    # CW comments: Budget for the three saved maps, transient periodograms, FFT buffers,
    # CW comments: and figure rendering, then use the requested grid at its full resolution.
    available = _available_memory()
    estimate = 144 * 1024**2 + 192 * n**2
    if available is not None and estimate > 0.65 * available:
        raise ValueError(f"A {n} × {n} validation grid needs about {estimate / 1024**2:,.0f} MiB of memory. "
                         "Reduce the resolution and try again.")
    dz = config.distance_m / config.screens
    z_m = config.distance_m - dz / 2
    wavelengths, _ = gaussian_wavelengths(config.wavelength_nm, config.fwhm_nm)
    dx = band_width_profile(channel, wavelengths)(z_m) / n
    spectrum = settings.to_spectrum(config.cn2)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    screen_image, spectrum_image = output / "advanced_phase_screens.png", output / "advanced_phase_spectrum.png"
    axes = _axis_bins(n, dx, wavelength, dz, spectrum)
    samples = [np.empty((_SAMPLES, axis.frequency.size)) for axis in axes]
    taper = np.hanning(n)
    window = taper[:, None] * taper[None, :]
    window_sum = np.sum(window)
    scale = (dx / n)**2 / np.mean(window**2)
    stream = np.random.SeedSequence(seed)
    first_three = []
    with threadpool_limits(limits=1, user_api="blas"):
        for idx, child in enumerate(stream.spawn(_SAMPLES)):
            check_cancel()
            phase = randomized_phase_screen(n, dx, wavelength, dz, spectrum,
                                            subharmonics=channel.subharmonics, rng=np.random.default_rng(child),
                                            whiten=channel.whiten)
            if idx < 3:
                first_three.append(phase)
            # CW comments: Paper §2.1 produces nonperiodic screens. The Hann window suppresses
            # CW comments: edge leakage in the periodogram. Correct for its mean square amplitude.
            centered = phase - np.sum(phase * window) / window_sum
            psd = abs(np.fft.fft2(centered * window))**2 * scale
            if not np.isfinite(psd).all():
                raise ValueError("The chosen turbulence parameters produced a nonfinite phase PSD.")
            for axis, results in zip(axes, samples):
                results[idx] = axis.average(psd)
    check_cancel()
    _plot_maps(first_three, dx, z_m, screen_image)
    check_cancel()
    directions, warnings = _plot_spectra(list(zip(axes, samples)), spectrum, n, dx, spectrum_image)
    check_cancel()
    report = {"samples": _SAMPLES, "validation_seed": int(stream.entropy), "resolution": n,
              "channel_resolution": config.resolution, "screen_z_m": z_m, "screen_width_m": n * dx,
              "dx_m": dx, "wavelength_m": wavelength, "slab_thickness_m": dz,
              "sampled_fft_band_cycles_m": [1 / (n * dx), 1 / (2 * dx)],
              "sector_half_angle_degrees": _SECTOR_DEG, "spectrum": asdict(spectrum),
              "comparison": "Heuristic: flag three adjacent comparison bins beyond 25% and 3 SEM from same-mode "
                            "theory. Use frequencies 4df to 0.9 Nyquist with at least eight Fourier modes per bin, "
                            "excluding modeled PSD more than ten decades below the eligible peak. "
                            "Tapering correlates nearby bins.",
              "directions": directions, "warnings": warnings}
    (output / "advanced_validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return AdvancedValidation(screen_image, spectrum_image, tuple(warnings), report)
