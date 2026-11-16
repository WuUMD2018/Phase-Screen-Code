"""Ensemble runner and data staging for the desktop simulator."""
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict, dataclass
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import secrets
import shutil
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits

from advanced_turbulence import TurbulenceSettings
from phase_screens import Spectrum
from propagation import ChannelConfig, coordinates, propagate_channel
from spectrum_band import band_width_profile, gaussian_negative_fraction, gaussian_wavelengths
from studio_io import export_data, export_images, render_beam


WAVELENGTH_M = 1550e-9
DEFAULT_WAVELENGTH_NM = 1550.0
_MIB = 1024 ** 2


def _positive(value, name):
    try:
        value = float(value)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(f"{name} must be a positive finite number.") from exc
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite number.")
    return value


def _from_log(log_value, name):
    try:
        return _positive(math.exp(log_value), name)
    except OverflowError as exc:
        raise ValueError(f"{name} is outside the numerical range of this simulation.") from exc


def _fried_log_coefficient(wavelength_nm):
    wavelength_nm = _positive(wavelength_nm, "Central wavelength")
    return math.log(0.423) + 2 * (math.log(2 * math.pi * 1e9) - math.log(wavelength_nm))


def cn2_to_r0_cm(cn2, distance_m, wavelength_nm=DEFAULT_WAVELENGTH_NM):
    """Equivalent plane-wave Kolmogorov Fried diameter in cm for uniform Cn²."""
    cn2 = _positive(cn2, "Cn²")
    distance_m = _positive(distance_m, "Propagation distance")
    # CW comments: Schmidt (2010), Eqs. (9.70)–(9.71), combine the slab
    # CW comments: strengths as r0^(-5/3) = 0.423 k² Cn² L for a uniform path.
    log_strength = _fried_log_coefficient(wavelength_nm) + math.log(cn2) + math.log(distance_m)
    return _from_log(math.log(100.0) - 0.6 * log_strength, "Equivalent Fried parameter")


def r0_cm_to_cn2(r0_cm, distance_m, wavelength_nm=DEFAULT_WAVELENGTH_NM):
    """Uniform Cn² in m^(-2/3) corresponding to the given Fried diameter."""
    r0_cm = _positive(r0_cm, "Fried parameter")
    distance_m = _positive(distance_m, "Propagation distance")
    log_cn2 = (math.log(100.0) - math.log(r0_cm)) * (5 / 3)
    log_cn2 -= _fried_log_coefficient(wavelength_nm) + math.log(distance_m)
    return _from_log(log_cn2, "Equivalent Cn²")


@dataclass(frozen=True)
class StudioConfig:
    distance_m: float = 10_000.0
    beam_diameter_cm: float = 6.0
    screens: int = 12
    cn2: float = 1e-16
    resolution: int = 512
    iterations: int = 6
    seed: int | None = None
    wavelength_nm: float = DEFAULT_WAVELENGTH_NM
    fwhm_nm: float = 10.0
    turbulence: TurbulenceSettings | None = None

    def __post_init__(self):
        if self.turbulence is not None and not isinstance(self.turbulence, TurbulenceSettings):
            raise ValueError("turbulence must be a TurbulenceSettings object or None for Normal mode.")
        for name, label in (("distance_m", "Propagation distance"),
                            ("beam_diameter_cm", "Initial beam diameter"), ("cn2", "Cn²"),
                            ("wavelength_nm", "Central wavelength")):
            object.__setattr__(self, name, _positive(getattr(self, name), label))
        wavelengths, _ = gaussian_wavelengths(self.wavelength_nm, self.fwhm_nm)
        object.__setattr__(self, "fwhm_nm", float(self.fwhm_nm))
        for name, label, minimum in (("screens", "Phase screens", 1),
                                     ("resolution", "Resolution", 8),
                                     ("iterations", "Iterations", 1)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
                raise ValueError(f"{label} must be an integer of at least {minimum}.")
            object.__setattr__(self, name, int(value))
        if self.resolution % 2:
            raise ValueError("Resolution must be even.")
        if self.seed is not None:
            if isinstance(self.seed, bool) or not isinstance(self.seed, (int, np.integer)) or self.seed < 0:
                raise ValueError("The random seed must be a nonnegative integer.")
            object.__setattr__(self, "seed", int(self.seed))
        cn2_to_r0_cm(self.cn2, self.distance_m, self.wavelength_nm)
        channel = self.channel_config()
        width = band_width_profile(channel, wavelengths)
        # CW comments: Check one common transverse sampling grid at both ends.
        # CW comments: The shortest color also sets the most restrictive phase sampling.
        try:
            with np.errstate(all="ignore"):
                dx0, dx1 = (width(z) / self.resolution for z in (0, self.distance_m))
                pixel_areas = (dx0 * dx0, dx1 * dx1)
                span = (self.distance_m * wavelengths[0], channel.waist * channel.waist,
                        2 * math.pi / wavelengths[0])
        except (ValueError, OverflowError, ZeroDivisionError) as exc:
            raise ValueError("Distance, beam size and wavelength band produce an invalid grid.") from exc
        if not all(math.isfinite(value) and value > 0 for value in (*pixel_areas, *span)):
            raise ValueError("Distance, beam size and wavelength band produce an invalid grid.")

    def channel_config(self, seed=0):
        spectrum = Spectrum(cn2=self.cn2) if self.turbulence is None else self.turbulence.to_spectrum(self.cn2)
        return ChannelConfig(wavelength=self.wavelength_nm / 1e9, length=self.distance_m,
                             waist=self.beam_diameter_cm / 200.0, n=self.resolution,
                             n_screens=self.screens, spectrum=spectrum, seed=seed)


@dataclass(frozen=True)
class StudioResult:
    config: StudioConfig
    work_dir: Path
    warnings: tuple[str, ...]
    elapsed_s: float
    workers: int
    seed: int


class SimulationCancelled(RuntimeError):
    """The studio run was stopped before its exports were ready."""


def _available_memory():
    if os.name == "nt":
        import ctypes

        class MemoryStatus(ctypes.Structure):
            _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
                (name, ctypes.c_uint64) for name in ("total", "available", "page_total", "page_available",
                                                    "virtual_total", "virtual_available", "extended")
            ]

        status = MemoryStatus()
        status.length = ctypes.sizeof(status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return int(status.available)
    else:
        try:
            return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_AVPHYS_PAGES")
        except (AttributeError, OSError, ValueError):
            pass
    return None


def _worker_count(config, folder, requested, component_count):
    n2 = config.resolution ** 2
    reference_export = int(config.fwhm_nm > 0)
    values_per_realization = (3 + reference_export) * config.screens + 2 + reference_export
    output_bytes = n2 * (4 * values_per_realization * config.iterations
                         + 16 * (config.screens + 1) + 4 * (2 + reference_export))
    output_bytes += 2 * _MIB * (config.iterations + 1)
    free_disk = shutil.disk_usage(folder).free
    if output_bytes + 32 * _MIB > free_disk:
        raise ValueError(f"This run needs about {output_bytes / _MIB:,.0f} MiB of temporary disk space. "
                         f"Only {free_disk / _MIB:,.0f} MiB is available.")
    if requested is not None and (isinstance(requested, bool) or not isinstance(requested, int) or requested < 1):
        raise ValueError("workers must be a positive integer.")
    cpu_count = (getattr(os, "process_cpu_count", os.cpu_count)() or 1)
    workers = min(config.iterations, cpu_count, 4 if requested is None else requested)
    # CW comments: Each process works on one realization. Budget for FFT temporaries
    # CW comments: and for one active slice of the disk-backed phase/field arrays.
    per_worker = 144 * _MIB + (160 + 16 * (component_count - 1)) * n2
    available = _available_memory()
    if available is not None:
        budget = int(0.65 * available)
        if budget < per_worker:
            raise ValueError("Resolution is too high for the available memory. Reduce the grid resolution.")
        workers = min(workers, max(1, budget // per_worker))
    return workers


def _write_coordinates(config, work_dir, wavelengths, check_cancel):
    channel = config.channel_config()
    width = band_width_profile(channel, wavelengths)
    out = work_dir / "coordinates"
    out.mkdir()
    n, s = config.resolution, config.screens
    edges = np.linspace(0, config.distance_m, s + 1)
    midpoints = (edges[:-1] + edges[1:]) / 2
    width_positions = np.linspace(0, config.distance_m, 2 * s + 1)[1::2]
    x_cube = np.lib.format.open_memmap(out / "screen_x_m.npy", mode="w+", dtype=np.float64, shape=(s, n, n))
    y_cube = np.lib.format.open_memmap(out / "screen_y_m.npy", mode="w+", dtype=np.float64, shape=(s, n, n))
    for i, z in enumerate(width_positions):
        check_cancel()
        x = coordinates(n, width(z) / n)
        x_cube[i] = x[None, :]
        y_cube[i] = x[:, None]
    x_cube.flush()
    y_cube.flush()
    del x_cube, y_cube
    x = coordinates(n, width(config.distance_m) / n)
    np.save(out / "screen_z_m.npy", midpoints)
    np.save(out / "final_x_m.npy", np.broadcast_to(x[None, :], (n, n)))
    np.save(out / "final_y_m.npy", np.broadcast_to(x[:, None], (n, n)))
    return x


def _check_cancel(marker):
    if marker.is_file():
        raise SimulationCancelled("Simulation cancelled.")


def _render_options(config, center_amplitude):
    options = {"phase_label": f"Phase at {config.wavelength_nm:g} nm"}
    if config.fwhm_nm > 0:
        options.update(phase_amplitude=center_amplitude, amplitude_label="Spectral amplitude")
    return options


def _single_realization(config, work_dir, index, seed, band, cancel_event=None):
    out = work_dir / f"realization_{index + 1:04d}"
    cancel_marker = work_dir / ".cancel"
    _check_cancel(cancel_marker)
    out.mkdir()
    n, s = config.resolution, config.screens
    broadband = config.fwhm_nm > 0

    def screen_array(name):
        return np.lib.format.open_memmap(out / f"{name}.npy", mode="w+", dtype=np.float32, shape=(s, n, n))

    screens = screen_array("phase_screens_rad")
    amps = screen_array("after_screen_amplitude")
    phases = screen_array("after_screen_phase_rad")
    center_amps = screen_array("after_screen_center_amplitude") if broadband else None

    def save_screen(i, z_m, dx_m, phi):
        if cancel_event is not None and cancel_event.is_set():
            raise SimulationCancelled("Simulation cancelled.")
        _check_cancel(cancel_marker)
        screens[i] = phi

    def save_field(i, z_m, dx_m, field):
        amplitude_array = center_amps if broadband else amps
        amplitude_array[i] = np.abs(field)
        phases[i] = np.angle(field)

    def save_intensity(i, z_m, dx_m, intensity):
        amps[i] = np.sqrt(intensity, out=intensity)

    # CW comments: Paper §3.5 uses separate turbulence realizations. Parallelism
    # CW comments: is across independent runs; successive screens stay in axial order.
    with threadpool_limits(limits=1, user_api="blas"):
        channel = config.channel_config(seed)
        result = propagate_channel(channel, screen_callback=save_screen, field_callback=save_field,
            width_profile=band_width_profile(channel, band[0]), spectral_band=band if broadband else None,
            screen_intensity_callback=save_intensity if broadband else None, emit_warnings=False)
    for data in (screens, amps, phases, center_amps):
        if data is not None:
            data.flush()
    del data, screens, amps, phases, center_amps
    _check_cancel(cancel_marker)
    center_amplitude = np.abs(result.field).astype(np.float32)
    amplitude = np.sqrt(result.intensity).astype(np.float32) if broadband else center_amplitude
    phase = np.angle(result.field).astype(np.float32)
    np.save(out / "final_amplitude.npy", amplitude)
    np.save(out / "final_phase_rad.npy", phase)
    if broadband:
        np.save(out / "final_center_amplitude.npy", center_amplitude)
    render_beam(amplitude, phase, result.x_m, out, title=f"Realization {index + 1} of {config.iterations}",
                **_render_options(config, center_amplitude))
    return index, tuple(result.warnings)


def _average_results(config, work_dir, x, check_cancel):
    ndim = config.resolution
    intensity = np.zeros((ndim, ndim), dtype=np.float64)
    phasors = np.zeros((ndim, ndim), dtype=np.complex128)
    broadband = config.fwhm_nm > 0
    center_intensity = np.zeros_like(intensity) if broadband else None
    for i in range(config.iterations):
        check_cancel()
        src = work_dir / f"realization_{i + 1:04d}"
        amplitude = np.load(src / "final_amplitude.npy", mmap_mode="r")
        phase = np.load(src / "final_phase_rad.npy", mmap_mode="r")
        center_amplitude = np.load(src / "final_center_amplitude.npy", mmap_mode="r") if broadband else amplitude
        # CW comments: Average intensity before taking its square root. Average
        # CW comments: wrapped angles on the unit circle to preserve the ±π seam.
        intensity += np.square(amplitude, dtype=np.float64)
        if broadband:
            center_intensity += np.square(center_amplitude, dtype=np.float64)
        phasors += np.where(center_amplitude > 0, np.exp(1j * phase.astype(np.float64)), 0.0)
        del amplitude, phase, center_amplitude
    amplitude = np.sqrt(intensity / config.iterations).astype(np.float32)
    phase = np.angle(phasors).astype(np.float32)
    phase[np.abs(phasors) <= 1e-12 * config.iterations] = np.nan
    out = work_dir / "average"
    out.mkdir()
    np.save(out / "final_amplitude.npy", amplitude)
    np.save(out / "final_phase_rad.npy", phase)
    center_amplitude = None
    if broadband:
        center_amplitude = np.sqrt(center_intensity / config.iterations).astype(np.float32)
        np.save(out / "final_center_amplitude.npy", center_amplitude)
    render_beam(amplitude, phase, x, out, title=f"Average of {config.iterations} realizations",
                **_render_options(config, center_amplitude))


def _write_manifest(result, seeds, band, negative_fraction):
    cfg = result.config
    manifest = {
        "format_version": 3, "parameters": asdict(cfg), "wavelength_m": cfg.wavelength_nm / 1e9,
        "equivalent_plane_wave_fried_cm": cn2_to_r0_cm(cfg.cn2, cfg.distance_m, cfg.wavelength_nm),
        "atmospheric_turbulence": {
            "mode": "normal" if cfg.turbulence is None else "advanced",
            "phase_screen_spectrum": asdict(cfg.channel_config().spectrum),
            "fried_parameter_convention": "Kolmogorov equivalent at the center wavelength",
        },
        "wavelength_spectrum": {
            "density": "Gaussian optical power per unit wavelength with positive-wavelength support",
            "center_nm": cfg.wavelength_nm, "fwhm_nm": cfg.fwhm_nm,
            "wavelength_samples_nm": (band[0] * 1e9).tolist(), "power_weights": band[1].tolist(),
            "gaussian_fraction_at_nonpositive_wavelength": negative_fraction,
            "phase_reference_nm": cfg.wavelength_nm,
            "turbulence": "One common non-dispersive optical path per slab and realization",
        },
        "beam_size_definition": "Initial 1/e^2 intensity diameter",
        "base_seed": result.seed, "realization_seeds": [str(seed) for seed in seeds],
        "workers": result.workers, "elapsed_s": result.elapsed_s, "warnings": list(result.warnings),
        "data": {
            "indexing": "Arrays use [screen, y, x] or [y, x]. Realization folders are one-based.",
            "realization_NNNN/phase_screens_rad.npy": "float32 [screens, n, n], unwrapped screen phase at the center wavelength in rad",
            "realization_NNNN/after_screen_amplitude.npy": "float32 [screens, n, n], sqrt(spectral intensity) after modulation",
            "realization_NNNN/after_screen_phase_rad.npy": "float32 [screens, n, n], center-wavelength phase after modulation in rad",
            "realization_NNNN/final_amplitude.npy": "float32 [n, n], sqrt(spectral intensity) at the receiver",
            "realization_NNNN/final_phase_rad.npy": "float32 [n, n], center-wavelength receiver phase in rad",
            "coordinates/screen_x_m.npy": "float64 [screens, n, n], x in m, shared by all realizations",
            "coordinates/screen_y_m.npy": "float64 [screens, n, n], y in m, shared by all realizations",
            "coordinates/screen_z_m.npy": "float64 [screens], screen depth in m",
            "coordinates/final_x_m.npy": "float64 [n, n], receiver x in m",
            "coordinates/final_y_m.npy": "float64 [n, n], receiver y in m",
            "average/final_amplitude.npy": "float32 [n, n], sqrt(ensemble mean of spectral intensity)",
            "average/final_phase_rad.npy": "float32 [n, n], circular mean center phase in rad; NaN for zero resultant",
        },
        "images": "Amplitude and phase PNGs for every realization and the average. Read the colorbars for scales.",
    }
    if cfg.fwhm_nm > 0:
        manifest["data"].update({
            "realization_NNNN/after_screen_center_amplitude.npy": "float32 [screens, n, n], center-wavelength amplitude after modulation",
            "realization_NNNN/final_center_amplitude.npy": "float32 [n, n], center-wavelength receiver amplitude",
            "average/final_center_amplitude.npy": "float32 [n, n], sqrt(ensemble mean of center-wavelength intensity)",
        })
    (result.work_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def run_studio(config, work_dir, *, on_result=None, cancel_event=None, workers=None):
    """Run independent realizations and stage exports in an empty directory.

    Call on_result(index, total) after each realization's PNGs are ready. Index
    is zero-based. Caller owns the work directory and its eventual cleanup.
    """
    started = perf_counter()
    work_dir = Path(work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    if any(work_dir.iterdir()):
        raise ValueError("Choose an empty work directory for this simulation.")
    band = gaussian_wavelengths(config.wavelength_nm, config.fwhm_nm)
    used_workers = _worker_count(config, work_dir, workers, band[0].size)
    seed = secrets.randbits(52) if config.seed is None else config.seed
    subseeds = np.random.SeedSequence(seed).spawn(config.iterations)
    seeds = [sum(int(word) << (32 * i) for i, word in enumerate(s.generate_state(4))) for s in subseeds]
    warning_set = set()
    negative_fraction = gaussian_negative_fraction(config.wavelength_nm, config.fwhm_nm)
    if negative_fraction > 1e-6:
        warning_set.add(f"The input Gaussian puts {negative_fraction:.4%} of its power at nonpositive wavelengths. "
                        "The simulation renormalized its positive portion.")
    marker = work_dir / ".cancel"

    def check_cancel():
        if cancel_event is not None and cancel_event.is_set():
            marker.touch()
            raise SimulationCancelled("Simulation cancelled.")

    def completed(reply):
        index, messages = reply
        warning_set.update(messages)
        if on_result is not None:
            on_result(index, config.iterations)

    check_cancel()
    x = _write_coordinates(config, work_dir, band[0], check_cancel)
    if used_workers == 1:
        for i, child_seed in enumerate(seeds):
            check_cancel()
            completed(_single_realization(config, work_dir, i, child_seed, band, cancel_event))
    else:
        with ProcessPoolExecutor(max_workers=used_workers, mp_context=mp.get_context("spawn")) as pool:
            pending = set()
            remaining = iter(enumerate(seeds))
            try:
                for _ in range(used_workers):
                    i, child_seed = next(remaining)
                    pending.add(pool.submit(_single_realization, config, work_dir, i, child_seed, band))
                while pending:
                    check_cancel()
                    done, pending = wait(pending, timeout=0.1, return_when=FIRST_COMPLETED)
                    for future in done:
                        completed(future.result())
                        task = next(remaining, None)
                        if task is not None:
                            i, child_seed = task
                            pending.add(pool.submit(_single_realization, config, work_dir, i, child_seed, band))
            except BaseException:
                marker.touch()
                for future in pending:
                    future.cancel()
                raise
    _average_results(config, work_dir, x, check_cancel)
    result = StudioResult(config, work_dir, tuple(sorted(warning_set)), perf_counter() - started, used_workers, seed)
    _write_manifest(result, seeds, band, negative_fraction)
    return result
