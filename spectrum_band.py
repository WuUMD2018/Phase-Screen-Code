"""Wavelength quadrature and common spatial sampling for the optical band."""

from dataclasses import replace
from math import erfc, isfinite, log, sqrt

import numpy as np


_FWHM_PER_SIGMA = 2 * sqrt(2 * log(2))
_NEGATIVE_TAIL_SWITCH = 1e-6
_STANDARD_LIMIT = 8.0
_HERMITE_X, _HERMITE_WEIGHTS = np.polynomial.hermite.hermgauss(9)
_STANDARD_HERMITE_X = sqrt(2) * _HERMITE_X
_LEGENDRE_X, _LEGENDRE_WEIGHTS = np.polynomial.legendre.leggauss(25)


def _band_parameters(wavelength_nm, fwhm_nm):
    """Validate display units and return center and Gaussian sigma in metres."""
    try:
        center = float(wavelength_nm) / 1e9
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Central wavelength must be a positive finite number in nm.") from exc
    if not isfinite(center) or center <= 0:
        raise ValueError("Central wavelength must be a positive finite number in nm.")

    try:
        width = float(fwhm_nm)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Wavelength FWHM must be a nonnegative finite number in nm.") from exc
    if not isfinite(width) or width < 0:
        raise ValueError("Wavelength FWHM must be a nonnegative finite number in nm.")
    return center, width / 1e9 / _FWHM_PER_SIGMA


def gaussian_negative_fraction(wavelength_nm, fwhm_nm):
    """Fraction of the input Gaussian's power formally at λ ≤ 0."""
    center, sigma = _band_parameters(wavelength_nm, fwhm_nm)
    return 0.0 if sigma == 0 else 0.5 * erfc((center / sigma) / sqrt(2))


def gaussian_wavelengths(wavelength_nm, fwhm_nm):
    """Return sorted positive wavelength nodes (m) and normalized power weights.

    The input is a Gaussian power spectral density per unit *wavelength*.
    FWHM is the full width at half maximum of that density, in nm. The weights
    directly integrate intensity over wavelength. The zero-bandwidth result has
    one exact center node with unit weight.

    Nine Gauss-Hermite nodes integrate ordinary bands while keeping the center
    as a positive-weight node. If below-zero power exceeds 1 ppm or the node
    grid reaches zero, use 25 Gauss-Legendre nodes on the positive Gaussian,
    retaining wavelengths within 8 standard deviations of center. Weights are
    normalized on the sampled interval. In that case the exact center can be
    included with zero weight for a separate reference field or phase. A narrow
    positive band that is indistinguishable in float64 reduces to one node.
    """
    center, sigma = _band_parameters(wavelength_nm, fwhm_nm)
    if sigma == 0 or center + sigma == center:
        return np.array([center]), np.ones(1)

    # CW comments: Hermite quadrature integrates the Gaussian power density.
    # CW comments: Use wavelengths directly, with no frequency Jacobian.
    lambda_hermite = center + sigma * _STANDARD_HERMITE_X
    center_in_sigma = center / sigma
    negative_fraction = 0.5 * erfc(center_in_sigma / sqrt(2))
    if lambda_hermite[0] > 0 and negative_fraction <= _NEGATIVE_TAIL_SWITCH:
        nodes = lambda_hermite
        weights = _HERMITE_WEIGHTS.copy()
        nodes[nodes.size // 2] = center
    else:
        # CW comments: A broad Gaussian may assign power to negative wavelength.
        # CW comments: Integrate on its positive part and renormalize the weights.
        lo, hi = max(-center_in_sigma, -_STANDARD_LIMIT), _STANDARD_LIMIT
        t = (hi - lo) / 2 * _LEGENDRE_X + (lo + hi) / 2
        nodes = center + sigma * t
        weights = _LEGENDRE_WEIGHTS * np.exp(-0.5 * t**2)
        central_index = int(np.searchsorted(nodes, center))
        if central_index == nodes.size or nodes[central_index] != center:
            # CW comments: A zero-weight center is a reference field for phase.
            nodes = np.insert(nodes, central_index, center)
            weights = np.insert(weights, central_index, 0.0)

    if np.any(nodes <= 0) or not np.isfinite(nodes).all():
        raise ValueError("The wavelength band exceeds the numerical range of the simulation.")
    weights /= weights.sum()
    return nodes, weights


def band_width_profile(channel_config, wavelengths_m):
    """Return the shared physical domain width at any z, in metres.

    Use the longest sampled wavelength to allow for its stronger Gaussian
    spreading. Every wavelength then lands on the same transverse coordinates,
    so spectral intensity can be combined directly. With only the original
    monochromatic wavelength the returned schedule is the original method.
    """
    wavelengths = np.asarray(wavelengths_m, dtype=float)
    if (wavelengths.ndim != 1 or wavelengths.size == 0 or
            not np.isfinite(wavelengths).all() or np.any(wavelengths <= 0)):
        raise ValueError("The band must contain at least one positive finite wavelength.")
    longest = float(wavelengths.max())
    if longest == channel_config.wavelength:
        return channel_config.width_at
    return replace(channel_config, wavelength=longest).width_at
