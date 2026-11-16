"""Linear scalar paraxial propagation on physically expanding square grids.

Scaled Fresnel/angular-spectrum factorization; the e^(ikz) carrier is omitted.
This is not a nonparaxial angular-spectrum solver. See README.md.
"""
from dataclasses import dataclass, field, replace
import warnings
import numpy as np
from phase_screens import Spectrum, randomized_phase_screen


def coordinates(n, dx):
    return (np.arange(n) - n // 2) * dx


def fft2c(u):
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(u)))


def ifft2c(u):
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(u)))


def power(u, dx):
    return float(np.sum(abs(u)**2) * dx**2)


def gaussian_beam(n, dx, wavelength, waist, *, z_from_waist=0.0,
                  total_power=1.0, center=(0.0, 0.0), tilt=(0.0, 0.0)):
    """waist is the 1/e amplitude (=1/e^2 intensity) radius, in m.

    center is in metres, tilt is paraxial radians. Analytic normalization;
    a clipped input is NOT silently normalized back to total_power.
    """
    if min(n, dx, wavelength, waist, total_power) <= 0:
        raise ValueError("Positive beam and grid parameters required")
    x = coordinates(n, dx)
    r2 = (x[None, :] - center[0])**2 + (x[:, None] - center[1])**2
    # CW comments: Schmidt (2010), Ch. 6: the analytical Fresnel solution for a Gaussian
    # CW comments: sets its Rayleigh range, divergence and quadratic wavefront.
    zr = np.pi * waist**2 / wavelength
    q = 1 + 1j * z_from_waist / zr
    amplitude = np.sqrt(2 * total_power / (np.pi * waist**2))
    u = amplitude / q * np.exp(-r2 / (waist**2 * q))
    return u * np.exp(2j * np.pi / wavelength *
                      (tilt[0] * x[None, :] + tilt[1] * x[:, None]))


def _field_check(u, dx, wavelength):
    if u.ndim != 2 or u.shape[0] != u.shape[1] or u.shape[0] < 2 or u.shape[0] % 2:
        raise ValueError("Field must be square with an even side length of at least two")
    if (not np.isfinite(u).all() or not np.isfinite(dx) or dx <= 0
            or not np.isfinite(wavelength) or wavelength <= 0):
        raise ValueError("Require finite field, positive dx and wavelength")


def scaled_fresnel(u, wavelength, dx_in, dx_out, distance):
    """Fresnel propagation to arbitrary positive output pitch, same N.

    m = dx_out/dx_in. U2 = Q3/m * IFFT[Q2 * FFT[Q1*U1]].
    Pixel-area power sum(|U|^2)*dx^2 is preserved to roundoff, but
    conservation alone does not demonstrate freedom from aliasing.
    At distance=0 changing pitch is not a propagation: rejected.
    """
    u = np.asarray(u, dtype=complex)
    _field_check(u, dx_in, wavelength)
    if not np.isfinite(dx_out) or dx_out <= 0 or not np.isfinite(distance):
        raise ValueError("Invalid output pitch or distance")
    if distance == 0:
        if dx_in != dx_out:
            raise ValueError("Cannot change sampling at zero propagation distance")
        return u.copy()
    envelope, curvature, _ = _scaled_step(u, wavelength, dx_in, dx_out, distance)
    return _restore_curvature(envelope, wavelength, dx_out, curvature)


def _squared_radius(x):
    return x[None, :]**2 + x[:, None]**2


def _restore_curvature(envelope, wavelength, dx, curvature):
    if curvature == 0:
        return envelope.copy()
    x = coordinates(envelope.shape[0], dx)
    chirp = np.exp(1j * np.pi * curvature * x**2 / wavelength)
    field = envelope * chirp[None, :]
    field *= chirp[:, None]
    return field


def _scaled_step(u, wavelength, dx_in, dx_out, distance, *, curvature=0.0, diagnose=False):
    n, m, k = u.shape[0], dx_out / dx_in, 2 * np.pi / wavelength
    x1 = coordinates(n, dx_in)
    f = np.fft.fftfreq(n, dx_in)
    # CW comments: Paulson et al., §3.5, cite Schmidt [19] for FFT propagation between grid sizes.
    # CW comments: Schmidt (2010), Ch. 6: this scaled Fresnel step combines the prior output chirp
    # CW comments: with the current input chirp. The 1/m amplitude accounts for the m² pixel area.
    residual_curvature = curvature + (1 - m) / distance
    output_curvature = (m - 1) / (m * distance)
    # CW comments: The Fresnel chirps separate along x and y, so compute only
    # CW comments: their 1-D exponentials. Keep the transform in FFT order.
    if residual_curvature == 0:
        prechirped = u
    else:
        q1 = np.exp(1j * k * residual_curvature * x1**2 / 2)
        prechirped = u * q1[None, :]
        prechirped *= q1[:, None]
    q2 = np.exp(-1j * np.pi * wavelength * distance * f**2 / m)
    transformed = np.fft.fft2(np.fft.ifftshift(prechirped))
    diag = None
    if diagnose:
        diag = _transfer_diagnostics(transformed, f, wavelength, dx_in, m, distance)
    transformed *= q2[None, :]
    transformed *= q2[:, None]
    out = np.fft.fftshift(np.fft.ifft2(transformed))
    out /= m
    if diagnose:
        for label, field, x, pitch, coefficient in (
            ("residual", u, x1, dx_in, residual_curvature),
            ("output", out, coordinates(n, dx_out), dx_out, output_curvature),
        ):
            intensity = abs(field)**2
            radius = max(_occupied_radius(x, intensity.sum(axis=axis)) for axis in (0, 1))
            diag[f"{label}_chirp_cycles_per_pixel"] = float(
                abs(coefficient) * (radius + pitch / 2) * pitch / wavelength
            )
    return out, output_curvature, diag


def fraunhofer(u, wavelength, dx, distance):
    """One free-space far-field observation, not a repeated channel step.

    Returns (field, x_m). Natural pitch = wavelength*distance/(N*dx).
    Requires the neglected source quadratic phase pi*a^2/(lambda*z) << 1
    over the illuminated source radius a. The final output chirp is retained.
    """
    return _diffraction_fft(u, wavelength, dx, distance, include_input_chirp=False)


def fresnel_fft(u, wavelength, dx, distance):
    """One-step Fresnel diffraction on the natural FFT output grid.

    Returns (field, x_m) with pitch wavelength*distance/(N*dx).
    Use scaled_fresnel for a specifically chosen output pitch.
    """
    return _diffraction_fft(u, wavelength, dx, distance, include_input_chirp=True)


def _source_chirp_arguments(u, wavelength, dx, distance):
    u = np.asarray(u, dtype=complex)
    _field_check(u, dx, wavelength)
    if not np.isfinite(distance) or distance <= 0:
        raise ValueError("Observation distance must be finite and positive")
    x = coordinates(u.shape[0], dx)
    phase = np.pi * _squared_radius(x) / (wavelength * distance)
    return u, phase


def _diffraction_fft(u, wavelength, dx, distance, *, include_input_chirp):
    u, phase = _source_chirp_arguments(u, wavelength, dx, distance)
    # CW comments: Schmidt (2010), Chs. 4 and 6: the natural FFT grid is x = λzf.
    # CW comments: Fraunhofer drops the source chirp and keeps the observation chirp.
    x = coordinates(u.shape[0], wavelength * distance / (u.shape[0] * dx))
    q = np.exp(1j * np.pi * _squared_radius(x) / (wavelength * distance))
    if include_input_chirp:
        u = u * np.exp(1j * phase)
    return q * fft2c(u) * dx**2 / (1j * wavelength * distance), x


def fraunhofer_relative_error(u, wavelength, dx, distance):
    """Relative complex-field L2 difference from one-step discrete Fresnel.

    Parseval gives the difference on the common natural FFT grid from the
    dropped source chirp alone. Both methods still require adequate sampling.
    """
    u, phase = _source_chirp_arguments(u, wavelength, dx, distance)
    norm = np.linalg.norm(u)
    if norm == 0:
        raise ValueError("Fraunhofer relative error requires a nonzero field")
    return float(np.linalg.norm(u * np.expm1(1j * phase)) / norm)


def beam_metrics(u, dx):
    intensity = abs(u)**2
    total = intensity.sum()
    if total <= 0:
        raise ValueError("Zero-power field")
    x = coordinates(u.shape[0], dx)
    px, py = intensity.sum(axis=0) / total, intensity.sum(axis=1) / total
    cx, cy = float(px @ x), float(py @ x)
    wx, wy = 2 * np.sqrt(px @ (x - cx)**2), 2 * np.sqrt(py @ (x - cy)**2)
    edge = abs(x) >= 0.4 * u.shape[0] * dx
    border = edge[None, :] | edge[:, None]
    return {"power": float(total * dx**2), "centroid_x_m": cx, "centroid_y_m": cy,
            "radius_x_m": float(wx), "radius_y_m": float(wy),
            "edge_power_fraction": float(intensity[border].sum() / total)}


def sampling_diagnostics(u, wavelength, dx_in, dx_out, distance):
    """Heuristics for aliasing; NOT a rigorous convergence certificate.

    Transfer-function phase sampling is checked over frequencies carrying
    99.9999% of each marginal spectral energy, after the input chirp.
    """
    u = np.asarray(u, dtype=complex)
    _field_check(u, dx_in, wavelength)
    if not np.isfinite(dx_out) or dx_out <= 0 or not np.isfinite(distance) or distance == 0:
        raise ValueError("Diagnostics require positive output pitch and nonzero finite distance")
    n, m = u.shape[0], dx_out / dx_in
    x = coordinates(n, dx_in)
    r2 = x[None, :]**2 + x[:, None]**2
    q1 = np.exp(1j * np.pi * (1 - m) * r2 / (wavelength * distance))
    f = np.fft.fftshift(np.fft.fftfreq(n, dx_in))
    return _transfer_diagnostics(fft2c(q1 * u), f, wavelength, dx_in, m, distance)


def _occupied_radius(axis, marginal):
    order = np.argsort(abs(axis))
    cumulative = np.cumsum(marginal[order])
    i = np.searchsorted(cumulative, 0.999999 * cumulative[-1])
    return abs(axis[order[i]])


def _transfer_diagnostics(transformed, f, wavelength, dx_in, m, distance):
    spec = abs(transformed)**2
    if spec.sum() == 0:
        raise ValueError("Sampling diagnostics require a nonzero field")
    edge = abs(f) >= 0.8 / (2 * dx_in)
    high = float(spec[edge[:, None] | edge[None, :]].sum() / spec.sum())
    occupied = max(_occupied_radius(f, spec.sum(axis=axis)) for axis in (0, 1))
    cycles = wavelength * abs(distance) * occupied / (m * f.size * dx_in)
    return {"dechirped_high_frequency_fraction": high,
            "transfer_phase_cycles_per_bin": float(cycles)}


@dataclass(frozen=True)
class ChannelConfig:
    wavelength: float = 1.55e-6
    length: float = 10_000.0
    n_screens: int = 40
    n: int = 512
    waist: float = 0.03
    margin: float = 6.0
    extra_spread_angle: float = 10e-6
    subharmonics: int = 4
    seed: int = 20261115
    spectrum: Spectrum = field(default_factory=lambda: Spectrum(cn2=1e-16))
    whiten: bool = False
    absorber_start: float | None = None

    def __post_init__(self):
        if not isinstance(self.n, (int, np.integer)) or self.n < 8 or self.n % 2:
            raise ValueError("n must be an even integer >= 8")
        if not isinstance(self.n_screens, (int, np.integer)) or self.n_screens < 1:
            raise ValueError("n_screens must be a positive integer")
        if not all(np.isfinite(v) and v > 0 for v in
                   (self.wavelength, self.length, self.waist, self.margin)):
            raise ValueError("Positive finite channel parameters required")
        if not np.isfinite(self.extra_spread_angle) or self.extra_spread_angle < 0:
            raise ValueError("extra_spread_angle must be nonnegative")
        if self.absorber_start is not None and not 0 < self.absorber_start < 1:
            raise ValueError("absorber_start must be None or in (0,1)")

    def width_at(self, z):
        """Scheduled width (metres), not automatic turbulence prediction.

        Half-width = margin * nominal radius. Extra angular allowance is
        user-set and combined in quadrature with vacuum Gaussian spreading.
        """
        # CW comments: Paulson et al., §3.5, describe propagation between physical domains.
        # CW comments: This schedule uses vacuum Gaussian spreading and an adjustable angular margin.
        zr = np.pi * self.waist**2 / self.wavelength
        radius = np.sqrt(self.waist**2 * (1 + (z / zr)**2)
                         + (self.extra_spread_angle * z)**2)
        return 2 * self.margin * radius


@dataclass
class ChannelResult:
    field: np.ndarray
    x_m: np.ndarray
    history: list
    screen_geometry: list
    diagnostics: list
    warnings: list
    snapshots: dict[float, tuple[float, float, np.ndarray]] = field(default_factory=dict)
    spectral_intensity: np.ndarray | None = None

    @property
    def intensity(self):
        """Receiver intensity, integrated over wavelength for a spectral run."""
        return abs(self.field)**2 if self.spectral_intensity is None else self.spectral_intensity


def _spectral_components(reference, spectral_band):
    if spectral_band is None:
        return 1.0, []
    wavelengths, weights = (np.asarray(values, dtype=float) for values in spectral_band)
    if (wavelengths.ndim != 1 or wavelengths.size == 0 or wavelengths.shape != weights.shape
            or not np.isfinite(wavelengths).all() or np.any(wavelengths <= 0)
            or not np.isfinite(weights).all() or np.any(weights < 0) or not np.any(weights > 0)):
        raise ValueError("A spectral band needs positive wavelengths and nonnegative weights with positive sum")
    weights = weights / weights.max()
    weights /= weights.sum()
    at_center = wavelengths == reference
    auxiliary = [(float(wl), float(w)) for wl, w in zip(wavelengths, weights)
                 if wl != reference and w > 0]
    return float(weights[at_center].sum()), auxiliary


def _integrated_intensity(reference_field, reference_weight, auxiliary):
    intensity = reference_weight * abs(reference_field)**2
    for (_, weight), field_at_wavelength in auxiliary:
        intensity += weight * abs(field_at_wavelength)**2
    return intensity


def _snapshot_planes(snapshot_fractions, sample_z):
    """Map requested fractional locations to existing planes or free-space gaps."""
    try:
        if isinstance(snapshot_fractions, (str, bytes)):
            raise TypeError
        fractions = sorted({float(fraction) for fraction in snapshot_fractions})
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("snapshot_fractions must be an iterable of numbers in [0, 1]") from exc
    grid, interior = {}, {}
    for fraction in fractions:
        if not np.isfinite(fraction) or not 0 <= fraction <= 1:
            raise ValueError("snapshot_fractions must be finite numbers in [0, 1]")
        z = fraction * float(sample_z[-1])
        upper = int(np.searchsorted(sample_z, z))
        # Roundoff in fraction * length can put a matching plane on either side.
        for plane in (upper, max(upper - 1, 0)):
            if abs(sample_z[plane] - z) <= 2 * np.spacing(z):
                grid.setdefault(plane, []).append((fraction, z))
                break
        else:
            interior.setdefault(upper - 1, []).append((fraction, z))
    return grid, interior


def propagate_channel(config=None, *, initial_field=None, width_profile=None,
                      cn2_profile=None, screen_callback=None, field_callback=None, emit_warnings=True,
                      snapshot_fractions=(), spectral_band=None, screen_intensity_callback=None):
    """Symmetric split-step channel with independent midpoint phase screens.

    Per slab: propagate dz/2 -> apply exp(i*phi) -> propagate dz/2.
    phi represents cn2(midpoint)*dz. cn2_profile(z) is sampled at midpoint;
    refine the slabs to resolve a varying profile. width_profile(z), if given,
    specifies the full physical width in metres at any propagation plane.
    screen_callback(index, z_m, dx_m, phase) permits streaming screens to disk.
    field_callback(index, z_m, dx_m, field) receives an independent copy of the
    physical complex field immediately after screen modulation and before any
    absorbing mask. The field includes the current quadratic phase.
    N remains constant while the scheduled physical widths and pixel pitches grow.
    Each slab has its own RNG stream. The phase is applied before the callback.
    The quadratic scaling phase is tracked across steps and restored at exit.
    snapshot_fractions selects fractional z locations in [0, 1]. Entries in
    the returned snapshots mapping are (z_m, dx_m, complex_field). Existing
    screen planes contain the field after phase modulation and any absorber.
    Interior planes are sampled by branching a free-space propagation from
    the previous simulation plane, without changing the main channel steps.

    spectral_band=(wavelengths_m, power_weights) propagates incoherently summed
    components through the same non-dispersive optical-path screens. The
    receiver intensity includes every component. The complex field, history,
    screen phases, and snapshots refer to config.wavelength. All components
    share one grid and the same initial field profile. By default the longest
    wavelength determines the width schedule. Sampling diagnostics retain the
    worst values over the band. screen_intensity_callback(index, z_m, dx_m,
    intensity) receives the integrated intensity immediately after each screen.
    """
    c = ChannelConfig() if config is None else config
    reference_weight, components = _spectral_components(c.wavelength, spectral_band)
    if components:
        longest = max(c.wavelength, max(wl for wl, _ in components))
        shortest = min(c.wavelength, min(wl for wl, _ in components))
        default_width = replace(c, wavelength=longest).width_at
    else:
        shortest, default_width = c.wavelength, c.width_at
    width = default_width if width_profile is None else width_profile
    edges = np.linspace(0, c.length, c.n_screens + 1)
    dz = c.length / c.n_screens
    sample_z = np.linspace(0, c.length, 2 * c.n_screens + 1)
    widths = np.asarray([width(z) for z in sample_z], dtype=float)
    if widths.shape != sample_z.shape or not np.all(np.isfinite(widths)) or np.any(widths <= 0):
        raise ValueError("Width profile must return a positive finite scalar at each z")
    grid_snapshots, interior_snapshots = _snapshot_planes(snapshot_fractions, sample_z)
    snapshots = {}
    # CW comments: Paulson et al., §3.5, generate separate realizations for different domains.
    # CW comments: A child RNG per slab preserves its screen when earlier slabs are disabled.
    slab_seeds = np.random.SeedSequence(c.seed).spawn(c.n_screens)
    dx = widths[0] / c.n
    u = (gaussian_beam(c.n, dx, c.wavelength, c.waist) if initial_field is None
         else np.asarray(initial_field, dtype=complex).copy())
    if u.shape != (c.n, c.n):
        raise ValueError("Initial field must have shape (config.n, config.n)")
    _field_check(u, dx, c.wavelength)
    initial_power = power(u, dx)
    if initial_power <= 0:
        raise ValueError("Initial field must have nonzero power")
    auxiliary_fields = [u.copy() for _ in components]
    history, geometry, diagnostics, flags = [], [], [], set()
    curvature = 0.0
    mask = None
    if c.absorber_start is not None:
        t = abs(coordinates(c.n, 2 / c.n))
        ramp = np.clip((t - c.absorber_start) / (1 - c.absorber_start), 0, 1)
        one = np.cos(np.pi * ramp / 2)**8
        mask = one[:, None] * one[None, :]
    def record(z, pitch):
        met = beam_metrics(u, pitch)
        met.update(z_m=float(z), grid_width_m=c.n * pitch, dx_m=pitch)
        if met["edge_power_fraction"] > 1e-4:
            flags.add("Power reached the outer 10% border: enlarge the width/margin and check convergence.")
        history.append(met)
    record(0, dx)
    for fraction, z in grid_snapshots.pop(0, ()):
        snapshots[fraction] = (z, float(dx), u.copy())
    receiver_snapshots = grid_snapshots.pop(sample_z.size - 1, ())
    for j in range(c.n_screens):
        zmid = (edges[j] + edges[j + 1]) / 2
        dxmid, dxnext = widths[2 * j + 1] / c.n, widths[2 * j + 2] / c.n
        spectrum = (c.spectrum if cn2_profile is None else
                    replace(c.spectrum, cn2=float(cn2_profile(zmid))))
        slab_rytov = None
        if spectrum.alpha == 11 / 3 and spectrum.mu_x == spectrum.mu_y == 1:
            # Plane-wave Kolmogorov indicator only, not exact finite-scale
            # beam scintillation. A large value flags coarse axial sampling.
            slab_rytov = 1.23 * spectrum.cn2 * (2 * np.pi / shortest)**(7 / 6) * dz**(11 / 6)
            if slab_rytov > 0.1:
                flags.add("Large per-slab Kolmogorov Rytov indicator: increase the screen count and test axial convergence.")
        for half, pitch_out in enumerate((dxmid, dxnext)):
            step = 2 * j + half
            # CW comments: Schmidt (2010), Ch. 9: the main channel advances through midpoint screens.
            # CW comments: A diagnostic plane in a gap uses a separate Fresnel step from upstream.
            for fraction, z in interior_snapshots.get(step, ()):
                snapshot_width = np.asarray(width(z), dtype=float)
                if snapshot_width.ndim != 0 or not np.isfinite(snapshot_width) or snapshot_width <= 0:
                    raise ValueError("Width profile must return a positive finite scalar at snapshot z")
                snapshot_dx = float(snapshot_width) / c.n
                branch, branch_curvature, _ = _scaled_step(
                    u, c.wavelength, dx, snapshot_dx, z - sample_z[step], curvature=curvature,
                )
                snapshots[fraction] = (z, snapshot_dx, _restore_curvature(
                    branch, c.wavelength, snapshot_dx, branch_curvature,
                ))
            # CW comments: Paulson et al., §3.5, alternate FFT propagation and screen modulation.
            # CW comments: Here each slab places two equal free-space steps around its screen.
            previous_curvature = curvature
            u, curvature, diag = _scaled_step(
                u, c.wavelength, dx, pitch_out, dz / 2, curvature=curvature, diagnose=True,
            )
            edge_power = beam_metrics(u, pitch_out)["edge_power_fraction"]
            for i, (wavelength, _) in enumerate(components):
                # CW comments: With a common grid the geometric curvature is shared.
                # CW comments: Each component still has its own wavelength-dependent Fresnel chirps.
                auxiliary_fields[i], _, chromatic_diag = _scaled_step(
                    auxiliary_fields[i], wavelength, dx, pitch_out, dz / 2,
                    curvature=previous_curvature, diagnose=True,
                )
                for key, value in chromatic_diag.items():
                    diag[key] = max(diag[key], value)
                edge_power = max(edge_power, beam_metrics(auxiliary_fields[i], pitch_out)["edge_power_fraction"])
            diag.update(screen_index=j, half_step=half, z_end_m=float(zmid if half == 0 else edges[j + 1]))
            if diag["dechirped_high_frequency_fraction"] > 1e-5:
                flags.add("Significant energy approaches Nyquist: increase N and inspect screen/field sampling.")
            if diag["transfer_phase_cycles_per_bin"] > 0.5:
                flags.add("Transfer chirp is undersampled over occupied bandwidth: refine propagation steps or grid/window.")
            if diag["residual_chirp_cycles_per_pixel"] > 0.5:
                flags.add("Residual scaling chirp exceeds half a cycle per pixel over the occupied field: check the width schedule and resolution.")
            dx = pitch_out
            diag["edge_power_fraction"] = edge_power
            diagnostics.append(diag)
            if edge_power > 1e-4:
                flags.add("Power reached the outer 10% border: enlarge the width/margin and check convergence.")
            if half == 0:
                # CW comments: Paulson et al., §2.1, Eqs. (4)–(7): use the full slab thickness
                # CW comments: for screen variance, then multiply the field by exp(iφ).
                phi = randomized_phase_screen(c.n, dx, c.wavelength, dz, spectrum,
                    subharmonics=c.subharmonics, rng=np.random.default_rng(slab_seeds[j]), whiten=c.whiten)
                intensity = abs(u)**2
                lit = intensity > intensity.max() * 1e-6
                for other_field in auxiliary_fields:
                    other_intensity = abs(other_field)**2
                    lit |= other_intensity > other_intensity.max() * 1e-6
                step_x = np.abs(np.diff(phi, axis=1))[lit[:, 1:] & lit[:, :-1]]
                step_y = np.abs(np.diff(phi, axis=0))[lit[1:, :] & lit[:-1, :]]
                max_step = max(np.max(step_x, initial=0), np.max(step_y, initial=0)) * (c.wavelength / shortest)
                if max_step > np.pi / 2:
                    flags.add("Large adjacent screen-phase increments: increase N and check exp(i*phi) sampling.")
                geometry.append({"screen_index": j, "z_m": float(zmid), "thickness_m": dz,
                    "width_m": c.n * dx, "dx_m": dx, "cn2": spectrum.cn2,
                    "kolmogorov_slab_rytov_indicator": slab_rytov,
                    "max_lit_phase_increment_rad": float(max_step)})
                u *= np.exp(1j * phi)
                # CW comments: Paulson et al., Eq. (4), makes phase proportional to k.
                # CW comments: Reuse one physical OPD screen and rescale its phase for each wavelength.
                for i, (wavelength, _) in enumerate(components):
                    auxiliary_fields[i] *= np.exp(1j * (c.wavelength / wavelength) * phi)
                if screen_callback is not None:
                    screen_callback(j, float(zmid), dx, phi)
                if field_callback is not None:
                    field_callback(j, float(zmid), dx, _restore_curvature(u, c.wavelength, dx, curvature))
                if screen_intensity_callback is not None:
                    screen_intensity_callback(j, float(zmid), dx,
                        _integrated_intensity(u, reference_weight, zip(components, auxiliary_fields)))
            if mask is not None:
                u *= mask
                for field_at_wavelength in auxiliary_fields:
                    field_at_wavelength *= mask
            for fraction, z in grid_snapshots.get(step + 1, ()):
                snapshots[fraction] = (z, float(dx), _restore_curvature(u, c.wavelength, dx, curvature))
        record(edges[j + 1], dx)
    if c.absorber_start is not None and power(u, dx) < 0.999 * initial_power:
        flags.add("Absorbing border removed >0.1% of input power; enlarge the computational window.")
    if diagnostics[-1]["output_chirp_cycles_per_pixel"] > 0.5:
        flags.add("Receiver curvature exceeds half a cycle per pixel over the occupied field: refine the receiver pitch before propagating its sampled complex field again.")
    messages = sorted(flags)
    if emit_warnings:
        for message in messages:
            warnings.warn(message, RuntimeWarning, stacklevel=2)
    field_at_receiver = _restore_curvature(u, c.wavelength, dx, curvature)
    spectral_intensity = (_integrated_intensity(field_at_receiver, reference_weight,
                          zip(components, auxiliary_fields)) if components else None)
    for fraction, z in receiver_snapshots:
        snapshots[fraction] = (z, float(dx), field_at_receiver.copy())
    return ChannelResult(field_at_receiver, coordinates(c.n, dx), history, geometry,
                         diagnostics, messages, dict(sorted(snapshots.items())), spectral_intensity)
