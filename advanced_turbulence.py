"""Validated advanced phase-screen settings in display units."""
from dataclasses import dataclass
import math

from phase_screens import Spectrum


@dataclass(frozen=True)
class TurbulenceSettings:
    alpha: float = 11 / 3
    inner_scale_mm: float = 10.0
    outer_scale_mm: float = 30_000.0
    major_scale: float = 1.0
    minor_scale: float = 1.0

    def __post_init__(self):
        names = ("alpha", "inner_scale_mm", "outer_scale_mm", "major_scale", "minor_scale")
        for name in names:
            try:
                value = float(getattr(self, name))
            except (ValueError, TypeError, OverflowError) as exc:
                raise ValueError("Advanced settings must be finite numbers.") from exc
            if not math.isfinite(value):
                raise ValueError("Advanced settings must be finite numbers.")
            object.__setattr__(self, name, value)
        if not 3 < self.alpha < 4:
            raise ValueError("Power-law index α must be greater than 3 and less than 4.")
        if not 0 < self.inner_scale_mm < self.outer_scale_mm:
            raise ValueError("Enter positive scales with the inner scale smaller than the outer scale.")
        if self.inner_scale_mm / 1000 == 0:
            raise ValueError("The inner scale is too small for the numerical model.")
        if not 0.5 <= self.minor_scale <= self.major_scale <= 2:
            raise ValueError("Axis scales must be between 0.5 and 2, with major at least as large as minor.")

    def to_spectrum(self, cn2):
        """Build the finite anisotropic phase-screen model from display units."""
        return Spectrum(cn2=cn2, model="bounded_power_law", alpha=self.alpha,
                        inner_scale=self.inner_scale_mm / 1000, outer_scale=self.outer_scale_mm / 1000,
                        mu_x=self.major_scale, mu_y=self.minor_scale)
