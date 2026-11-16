"""Random-shift FFT phase screens and randomized subharmonics.

Based on Paulson, Wu & Davis, arXiv:1905.07074v2, Eqs. (4)-(10),
(17)-(30). See README.md for explicit corrections to the printed
subharmonic weights and central-cell convention. All lengths are metres;
kappa is radians/metre; output is unwrapped phase in radians.
"""
from dataclasses import dataclass, replace
from math import gamma
import numpy as np


_KOLMOGOROV_ALPHA = 11 / 3
_CN2_REFERENCE_M = 1.0


def _generalized_normalization(alpha):
    return gamma(alpha - 1) * np.cos(np.pi * alpha / 2) / (4 * np.pi**2)


_KOLMOGOROV_NORMALIZATION = _generalized_normalization(_KOLMOGOROV_ALPHA)


@dataclass(frozen=True)
class Spectrum:
    cn2: float = 1e-15
    model: str = "modified_atmosphere"
    inner_scale: float = 0.01
    outer_scale: float = 30.0
    alpha: float = _KOLMOGOROV_ALPHA
    mu_x: float = 1.0
    mu_y: float = 1.0

    def __post_init__(self):
        if not np.isfinite(self.cn2) or self.cn2 < 0:
            raise ValueError("cn2 must be finite and nonnegative")
        if self.model not in {"modified_atmosphere", "von_karman", "power_law", "bounded_power_law"}:
            raise ValueError("Unknown spectrum model")
        if not 3 < self.alpha < 4:
            raise ValueError("Require 3 < alpha < 4")
        if not all(np.isfinite(v) and v > 0 for v in (self.mu_x, self.mu_y)):
            raise ValueError("Anisotropy factors must be finite and positive")
        if self.model != "power_law":
            if self.model != "bounded_power_law" and (
                self.mu_x != 1 or self.mu_y != 1 or self.alpha != _KOLMOGOROV_ALPHA
            ):
                raise ValueError("Modified atmosphere and von Kármán models are isotropic Kolmogorov")
            if not all(np.isfinite(v) and v > 0 for v in
                       (self.inner_scale, self.outer_scale)):
                raise ValueError("Bounded models require finite positive inner/outer scales")
            if self.inner_scale >= self.outer_scale:
                raise ValueError("inner_scale must be smaller than outer_scale")

    @property
    def inner_wavenumber(self):
        """Inner cutoff of the stretched wavenumber, in rad/m, for finite spectra."""
        if self.model == "power_law":
            return None
        return (5.92 if self.model == "von_karman" else 3.3) / self.inner_scale

    def psd(self, kx, ky):
        """3-D refractive-index PSD evaluated at kz=0 (not phase PSD).

        For power_law, inner_scale/outer_scale are deliberately unused and
        cn2 has units m**(3-alpha). Other models use cn2 in m**(-2/3).
        The bounded_power_law coefficients use this convention at a 1 m
        reference separation before applying the inner and outer cutoffs.
        """
        q2 = (self.mu_x * np.asarray(kx)) ** 2 + (self.mu_y * np.asarray(ky)) ** 2
        if self.cn2 == 0:
            return np.zeros_like(q2, dtype=float)
        if self.model == "power_law":
            # CW comments: Paper §3.2, Eqs. (21)–(23). Use the anisotropic refractive-index PSD
            # CW comments: at κz = 0, with slope −α and the A(α) normalization.
            a = _generalized_normalization(self.alpha)
            with np.errstate(divide="ignore"):
                return a * self.cn2 * self.mu_x * self.mu_y * q2 ** (-self.alpha / 2)
        # CW comments: Paper §2.2, Eqs. (9)–(10). The outer scale flattens the spectrum near
        # CW comments: zero and the inner scale drives the high-frequency exponential cutoff.
        q0 = 2 * np.pi / self.outer_scale
        ql = self.inner_wavenumber
        if self.model != "von_karman":
            v = np.sqrt(q2) / ql
            # The fitted polynomial becomes negative only deep in the
            # exponentially negligible tail; a PSD cannot be negative.
            bump = np.maximum(1 + 1.802 * v - 0.254 * v ** (7 / 6), 0)
        else:
            # CW comments: The modified von Kármán option uses its own inner cutoff and omits
            # CW comments: the bump in paper Eq. (10).
            bump = 1.0
        strength = 0.033 * self.cn2
        if self.model == "bounded_power_law":
            # CW comments: Eq. (23) sets A(alpha); calibrate it to the 0.033 in Eq. (9).
            # CW comments: Match the structure coefficient units at 1 m, then apply Eqs. (9)–(10)
            # CW comments: to q = hypot(mu_x*kx, mu_y*ky). The PSD carries its mu_x*mu_y Jacobian.
            strength *= _generalized_normalization(self.alpha) / _KOLMOGOROV_NORMALIZATION
            strength *= _CN2_REFERENCE_M ** (_KOLMOGOROV_ALPHA - self.alpha) * self.mu_x * self.mu_y
        return strength * bump * np.exp(-q2 / ql**2) / (q2 + q0**2) ** (self.alpha / 2)


def _check_grid(n, dx, wavelength, thickness):
    if not isinstance(n, (int, np.integer)) or n < 8 or n % 2:
        raise ValueError("n must be an even integer >= 8")
    if not all(np.isfinite(v) and v > 0 for v in (dx, wavelength, thickness)):
        raise ValueError("dx, wavelength, and thickness must be finite and positive")


def _add_low_modes(phase, x, kx, ky, coefficients):
    """Accumulate separable, piston-removed modes in one real matrix multiply."""
    # CW comments: Paper §2.1 discusses piston. Keep expm1 in the identity
    # CW comments: exp(i k·r)-1 = (1+expm1(i ky y))*expm1(i kx x)+expm1(i ky y).
    # CW comments: Evaluate all 1-D factors, then multiply two thin matrices.
    ex = np.expm1(1j * np.asarray(kx)[:, None] * x[None, :])
    ey = np.expm1(1j * x[:, None] * np.asarray(ky)[None, :])
    coefficients = np.asarray(coefficients)
    weighted = (1 + ey) * coefficients
    y_term = np.sum(ey * coefficients, axis=1).real
    left = np.concatenate((weighted.real, -weighted.imag), axis=1)
    right = np.concatenate((ex.real, ex.imag), axis=0)
    phase += left @ right
    phase += y_term[:, None]


def randomized_phase_screen(n, dx, wavelength, thickness, spectrum=None,
                            *, subharmonics=4, rng=None, whiten=False):
    """Return one n x n phase screen; array axes are [y, x].

    subharmonics=0: core randomly shifted FFT, including its shifted central
    sample. P>0: replace that cell by P nested 3x3 grids (8 samples per level)
    and ONE central sample at the finest level, with independent cell jitter.
    Each real and imaginary coefficient part has unit-variance Gaussian noise;
    do NOT divide the complex coefficient by sqrt(2).

    thickness is the full slab represented by this one thin screen, even
    when the screen is placed at the slab midpoint. Remove piston only,
    retaining physical tilt, defocus and beam wander. No phase wrapping.
    whiten implements the approximate short-lag correction, OFF by default.
    """
    _check_grid(n, dx, wavelength, thickness)
    if not isinstance(subharmonics, (int, np.integer)) or not 0 <= subharmonics <= 20:
        raise ValueError("subharmonics must be an integer in [0, 20]")
    spectrum = Spectrum() if spectrum is None else spectrum
    rng = np.random.default_rng() if rng is None else rng
    if spectrum.cn2 == 0:
        return np.zeros((n, n))
    # CW comments: Paper §2.1, Eqs. (3)–(5). A screen of width n*dx has wavenumber spacing
    # CW comments: 2π/(n*dx). Shift the whole lattice by one uniform offset per axis.
    x = (np.arange(n) - n // 2) * dx
    dk = 2 * np.pi / (n * dx)
    k = 2 * np.pi / wavelength
    f = (np.arange(n) - n // 2) * dk
    shift_x, shift_y = rng.uniform(-0.5, 0.5, 2) * dk
    psd = spectrum.psd(f[None, :] + shift_x, f[:, None] + shift_y)
    # CW comments: Paper §3.1, Eq. (17). Exclude the central cell from the FFT and add it
    # CW comments: separately below so a large piston does not erase small phase variations.
    psd[n // 2, n // 2] = 0.0
    # CW comments: Paper §2.1, Eq. (4). Both Gaussian components have unit variance. Multiply
    # CW comments: each coefficient by the square root of the phase PSD times the cell area dk².
    coef = (rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n)))
    coef *= k * np.sqrt(2 * np.pi * thickness * psd) * dk
    # CW comments: Paper §2.1, Eqs. (6)–(7). Undo NumPy's 1/n² inverse FFT normalization and
    # CW comments: apply the spatial phase ramp for the shared wavenumber shift. Keep one real
    # CW comments: phase realization.
    base = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(coef)))
    base *= n**2
    base *= np.exp(1j * shift_y * x)[:, None]
    base *= np.exp(1j * shift_x * x)[None, :]
    phase = base.real.copy()

    mode_x, mode_y, coefficients = [], [], []

    def draw_mode(kx, ky, cell_width):
        # CW comments: Paper Eqs. (4), (18) and Paulson dissertation Eq. (2-35). The square root
        # CW comments: of a subcell's area is its width, giving the amplitude factor
        # CW comments: 3**(-level).
        amplitude = k * np.sqrt(2 * np.pi * thickness * spectrum.psd(kx, ky)) * cell_width
        c = (rng.standard_normal() + 1j * rng.standard_normal()) * amplitude
        mode_x.append(kx)
        mode_y.append(ky)
        coefficients.append(c)

    # CW comments: Paper §3.1, Eqs. (17)–(20). Draw an independent sample in each of eight outer
    # CW comments: subcells and subdivide the central subcell again at the next level.
    for level in range(1, subharmonics + 1):
        dkp = dk / 3**level
        for iy in (-1, 0, 1):
            for ix in (-1, 0, 1):
                if ix == 0 and iy == 0:
                    continue
                jitter = rng.uniform(-0.5, 0.5, 2)
                draw_mode((ix + jitter[0]) * dkp, (iy + jitter[1]) * dkp, dkp)
    if subharmonics == 0:
        # CW comments: Paper §2.1, Eq. (5). The core algorithm uses the original shared shift
        # CW comments: for the central sample too.
        draw_mode(shift_x, shift_y, dk)
    else:
        # CW comments: Paper §3.1, Eq. (19). Sample the full central cell left after the final
        # CW comments: subdivision, of width dk/3**subharmonics. The README explains this
        # CW comments: indexing choice.
        dkp = dk / 3**subharmonics
        kx, ky = rng.uniform(-0.5, 0.5, 2) * dkp
        draw_mode(kx, ky, dkp)
    _add_low_modes(phase, x, mode_x, mode_y, coefficients)
    if whiten:
        # CW comments: Paper §3.2, Eqs. (24)–(30). Add shared pixel noise and an axis-specific
        # CW comments: variance difference to match unresolved nearest-neighbor increments.
        vx, vy = unresolved_noise_variances(n, dx, wavelength, thickness, spectrum)
        phase += np.sqrt(min(vx, vy)) * rng.standard_normal((n, n))
        # Independent columns vary in x; independent rows vary in y.
        if vx > vy:
            phase += np.sqrt(vx - vy) * rng.standard_normal((1, n))
        elif vy > vx:
            phase += np.sqrt(vy - vx) * rng.standard_normal((n, 1))
    # CW comments: Paper §2.1. Remove the final mean phase (piston) while retaining tilt and
    # CW comments: higher spatial variation.
    phase -= phase.mean()
    return phase


def unresolved_noise_variances(n, dx, wavelength, thickness, spectrum,
                               *, n_angle=256, n_radial=2048, tail_ratio=1e4):
    """Numerical Eqs. (24)-(30), with explicit FFT CELL boundaries.

    K_in = [-N/2-1/2, N/2-1/2]**2 * dk, the union of the cells sampled
    by the jittered even-N lattice. These cell edges are used in place of
    the bounds printed in Eq. (27).
    Polar/log-radial quadrature integrates outside this square. Results
    are approximate: increase n_angle/n_radial/tail_ratio for convergence.
    White noise matches nearest-neighbour missing increments, not the full
    unresolved covariance or physical subpixel diffraction.
    """
    _check_grid(n, dx, wavelength, thickness)
    if (not isinstance(n_angle, (int, np.integer)) or n_angle < 16
            or not isinstance(n_radial, (int, np.integer)) or n_radial < 32
            or not np.isfinite(tail_ratio) or tail_ratio <= 1):
        raise ValueError("Require quadrature counts >= 16 angles and 32 radii, and finite tail_ratio > 1")
    if spectrum.cn2 == 0:
        return 0.0, 0.0
    # CW comments: Paper §3.2, Eqs. (24)–(28). Integrate the unsampled region beyond the
    # CW comments: even-grid FFT cell edges with polar quadrature.
    dk = 2 * np.pi / (n * dx)
    lower, upper = (n / 2 + 0.5) * dk, (n / 2 - 0.5) * dk
    theta = (np.arange(n_angle) + 0.5) * 2 * np.pi / n_angle
    cs, sn = np.cos(theta), np.sin(theta)
    rx = np.where(cs >= 0, upper, lower) / np.maximum(abs(cs), 1e-15)
    ry = np.where(sn >= 0, upper, lower) / np.maximum(abs(sn), 1e-15)
    rmin = np.minimum(rx, ry)
    t = np.linspace(0, np.log(tail_ratio), n_radial)
    radius = rmin[:, None] * np.exp(t)[None, :]
    kx, ky = radius * cs[:, None], radius * sn[:, None]
    weight = spectrum.psd(kx, ky) * radius**2
    prefactor = 2 * np.pi * (2 * np.pi / wavelength)**2 * thickness
    # CW comments: Paper §3.2, Eqs. (24)–(25). The squared sine evaluates 1 − cos(κ·dx) stably.
    # CW comments: The log-radius Jacobian is r².
    vx = prefactor * 2 * np.pi * np.mean(np.trapezoid(
        weight * 2 * np.sin(dx * kx / 2)**2, x=t, axis=1))
    vy = prefactor * 2 * np.pi * np.mean(np.trapezoid(
        weight * 2 * np.sin(dx * ky / 2)**2, x=t, axis=1))
    return float(max(vx, 0)), float(max(vy, 0))


def structure_function(phase, pixel_lags):
    """Unbiased-by-wrap spatial increments along x and y (no np.roll)."""
    n = phase.shape[0]
    values = []
    # CW comments: Paper §2.2, Eqs. (11)–(15). Estimate the phase structure function by
    # CW comments: averaging squared phase differences over all valid x and y pixel pairs at
    # CW comments: each lag.
    for lag in pixel_lags:
        if int(lag) != lag or not 0 < lag < n:
            raise ValueError("Each lag must be an integer in (0, n)")
        lag = int(lag)
        values.append([np.mean((phase[:, lag:] - phase[:, :-lag])**2),
                       np.mean((phase[lag:, :] - phase[:-lag, :])**2)])
    return np.asarray(values)


def theoretical_structure(r, wavelength, thickness, spectrum, *, axis="x",
                          n_radial=8192, n_angle=512):
    """Eq. (11) integrated over all spatial frequencies.

    Bounded models use polar quadrature with the axis separation stretched by
    1/mu for the anisotropic model. Unbounded power laws use an analytic
    gamma-function integral, including anisotropy.
    """
    r = np.atleast_1d(np.asarray(r, dtype=float))
    if r.ndim != 1 or not np.isfinite(r).all() or np.any(r < 0) or axis not in ("x", "y"):
        raise ValueError("Require finite nonnegative separations and axis='x' or 'y'")
    if not all(np.isfinite(v) and v > 0 for v in (wavelength, thickness)):
        raise ValueError("wavelength and thickness must be finite and positive")
    if spectrum.cn2 == 0:
        return np.zeros_like(r)
    k = 2 * np.pi / wavelength
    if spectrum.model == "power_law":
        # CW comments: Paper §3.4, Eqs. (32)–(33). Use the closed-form gamma-function integral
        # CW comments: and rescale the separation by the chosen axis's anisotropy factor.
        a = spectrum.alpha
        A = _generalized_normalization(a)
        integral = -gamma(1 - a / 2) / (2 ** (a - 1) * gamma(a / 2))
        mu = spectrum.mu_x if axis == "x" else spectrum.mu_y
        return 8 * np.pi**2 * k**2 * thickness * spectrum.cn2 * A * integral * (r / mu)**(a - 2)
    if (not isinstance(n_radial, (int, np.integer)) or n_radial < 32
            or not isinstance(n_angle, (int, np.integer)) or n_angle < 16):
        raise ValueError("Require integer quadrature counts >= 32 radii and 16 angles")
    if spectrum.mu_x != 1.0 or spectrum.mu_y != 1.0:
        # CW comments: In Eq. (11), q_x = mu_x*kx and q_y = mu_y*ky transfer
        # CW comments: anisotropy to r/mu. The measure cancels the PSD's mu_x*mu_y factor.
        r = r / (spectrum.mu_x if axis == "x" else spectrum.mu_y)
        spectrum = replace(spectrum, mu_x=1.0, mu_y=1.0)
    q0 = 2 * np.pi / spectrum.outer_scale
    ql = spectrum.inner_wavenumber
    q = np.geomspace(q0 * 1e-6, max(12 * ql, 12 * q0), n_radial)
    psd = spectrum.psd(q, 0)
    ct = np.cos((np.arange(n_angle) + 0.5) * 2 * np.pi / n_angle)
    answer = []
    for rho in r:
        # CW comments: Paper §2.2, Eqs. (11)–(12). Recover 1 − J0 by averaging 1 − cos over
        # CW comments: angle. The squared-sine form keeps the integral accurate at short
        # CW comments: separations.
        kernel = np.mean(2 * np.sin(q[:, None] * rho * ct[None, :] / 2)**2, axis=1)
        answer.append(8 * np.pi**2 * k**2 * thickness * np.trapezoid(psd * q * kernel, x=q))
    return np.asarray(answer)


def unresolved_structure_fraction(n, dx, wavelength, thickness, spectrum):
    """Return (x, y) fractions of D_phi(dx) outside the sampled FFT cells.

    This compares the missing adjacent-pixel increment variance (2 times
    the equivalent white-noise variance) to the full theoretical increment
    variance. The even-N FFT cell edges are those in
    unresolved_noise_variances(), not a symmetric Nyquist square. It estimates
    spectral truncation before any optional whitening. Returns (0, 0) for a
    vacuum slab and does not quantify propagation aliasing or scintillation.
    """
    # CW comments: Compare twice the white-noise variance in paper Eqs. (24)–(25) with the full
    # CW comments: one-pixel structure function of Eq. (11). Two independent noise samples
    # CW comments: contribute to each increment.
    vx, vy = unresolved_noise_variances(n, dx, wavelength, thickness, spectrum)
    if spectrum.cn2 == 0:
        return 0.0, 0.0
    full_x = theoretical_structure(dx, wavelength, thickness, spectrum, axis="x")[0]
    full_y = full_x if spectrum.mu_x == spectrum.mu_y else theoretical_structure(
        dx, wavelength, thickness, spectrum, axis="y")[0]
    return float(2 * vx / full_x), float(2 * vy / full_y)
