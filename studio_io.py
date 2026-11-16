"""Render standalone scientific panels and export complete simulation archives."""

from functools import lru_cache
import math
from pathlib import Path
import tempfile
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile

import numpy as np


_CANVAS = (760, 690)
_PLOT_LEFT, _PLOT_TOP, _PLOT_SIZE = 102, 108, 476
_BACKGROUND = "#0d1827"
_TEXT = "#eef4ff"
_SECONDARY = "#b7c7da"
_BORDER = "#52657e"
_MASK = (43, 56, 73)

_SCREEN_ARRAYS = (
    "phase_screens_rad.npy",
    "after_screen_amplitude.npy",
    "after_screen_phase_rad.npy",
)
_FINAL_ARRAYS = ("final_amplitude.npy", "final_phase_rad.npy")
_COORDINATES = ("screen_x_m.npy", "screen_y_m.npy", "screen_z_m.npy",
                "final_x_m.npy", "final_y_m.npy")
_IMAGES = ("amplitude.png", "phase.png")


@lru_cache(maxsize=1)
def _graphics():
    from matplotlib import colormaps, get_data_path
    from PIL import Image, ImageDraw, ImageFont

    font_dir = Path(get_data_path()) / "fonts" / "ttf"
    fonts = {
        "heading": ImageFont.truetype(str(font_dir / "DejaVuSans-Bold.ttf"), 27),
        "subtitle": ImageFont.truetype(str(font_dir / "DejaVuSans.ttf"), 15),
        "axis": ImageFont.truetype(str(font_dir / "DejaVuSans.ttf"), 16),
        "tick": ImageFont.truetype(str(font_dir / "DejaVuSans.ttf"), 14),
        "caption": ImageFont.truetype(str(font_dir / "DejaVuSans.ttf"), 13),
    }
    palettes = tuple(colormaps[name](np.linspace(0, 1, 256), bytes=True)[:, :3].copy()
                     for name in ("inferno", "twilight_shifted"))
    return Image, ImageDraw, fonts, palettes


def _axis_ticks(low, high):
    rough_spacing = (high - low) / 4
    decade = 10.0 ** math.floor(math.log10(rough_spacing))
    spacing = next(step * decade for step in (1, 2, 2.5, 5, 10)
                   if step * decade >= rough_spacing)
    first = math.ceil(low / spacing)
    last = math.floor(high / spacing)
    return (i * spacing for i in range(first, last + 1))


def _number(value):
    return format(value, ".3g").replace("-", "−")


def _render_panel(rgb, low, high, palette, color_ticks, path, *, heading, title,
                  bar_label, note, resample):
    image, image_draw, fonts, _ = _graphics()
    canvas = image.new("RGB", _CANVAS, _BACKGROUND)
    draw = image_draw.Draw(canvas)
    left, top, size = _PLOT_LEFT, _PLOT_TOP, _PLOT_SIZE
    bottom = top + size
    draw.text((36, 31), heading, font=fonts["heading"], fill=_TEXT)
    draw.text((38, 73), title, font=fonts["subtitle"], fill=_SECONDARY)

    # CW comments: The stored coordinates increase with array row number.
    # CW comments: Reverse the rows so the largest physical y is at the top.
    spatial = image.fromarray(rgb[::-1]).resize((size, size), resample=resample)
    canvas.paste(spatial, (left, top))
    draw.rectangle((left - 1, top - 1, left + size, bottom), outline=_BORDER)

    for value in _axis_ticks(low, high):
        point = round((value - low) / (high - low) * (size - 1))
        x, y = left + point, top + size - 1 - point
        label = _number(value)
        draw.line((x, bottom, x, bottom + 6), fill=_BORDER)
        draw.text((x, bottom + 10), label, font=fonts["tick"], fill=_SECONDARY, anchor="mt")
        draw.line((left - 7, y, left, y), fill=_BORDER)
        draw.text((left - 11, y), label, font=fonts["tick"], fill=_SECONDARY, anchor="rm")

    draw.text((left + size // 2, bottom + 37), "x (m)", font=fonts["axis"],
              fill=_TEXT, anchor="mt")
    label_image = image.new("RGBA", (70, 25))
    image_draw.Draw(label_image).text((35, 12), "y (m)", font=fonts["axis"],
                                      fill=_TEXT, anchor="mm")
    label_image = label_image.rotate(90, expand=True)
    canvas.paste(label_image, (19, top + (size - label_image.height) // 2), label_image)

    bar_x, bar_width = 618, 23
    gradient_indices = np.linspace(255, 0, size).round().astype(np.uint8)
    gradient = np.repeat(palette[gradient_indices, None, :], bar_width, axis=1)
    canvas.paste(image.fromarray(gradient), (bar_x, top))
    draw.rectangle((bar_x - 1, top - 1, bar_x + bar_width, bottom), outline=_BORDER)
    draw.text((bar_x + 5, top - 14), bar_label, font=fonts["axis"], fill=_TEXT, anchor="ls")
    for fraction, label in color_ticks:
        y = top + round((1 - fraction) * (size - 1))
        draw.line((bar_x + bar_width, y, bar_x + bar_width + 5, y), fill=_BORDER)
        draw.text((bar_x + bar_width + 10, y), label, font=fonts["tick"],
                  fill=_SECONDARY, anchor="lm")
    draw.text((38, 658), note, font=fonts["caption"], fill=_SECONDARY)
    canvas.save(path, format="PNG", compress_level=3)


def render_beam(amplitude, phase, x_m, directory, *, title, amplitude_max=None,
                phase_amplitude=None, phase_label=None, amplitude_label=None):
    """Write amplitude.png and phase.png from a square sampled optical field.

    Phase is in radians. NaN phase values are shown with the phase mask.
    The color scale is local to the image unless amplitude_max is supplied.
    When phase and amplitude describe different spectral quantities,
    phase_amplitude supplies the magnitude of the field that defines phase.
    """
    amplitude = np.asarray(amplitude)
    phase = np.asarray(phase)
    axis = np.asarray(x_m, dtype=float)
    if axis.ndim != 1 or len(axis) < 2 or amplitude.shape != (axis.size, axis.size):
        raise ValueError("Beam amplitude must be square and match the coordinate axis")
    if phase.shape != amplitude.shape or np.iscomplexobj(amplitude) or np.iscomplexobj(phase):
        raise ValueError("Amplitude and phase must be real arrays of the same shape")
    delta = np.diff(axis)
    if (not np.all(np.isfinite(axis)) or not np.all(delta > 0)
            or not np.allclose(delta, delta[0], rtol=1e-7, atol=0)):
        raise ValueError("Beam coordinates must form a finite, ascending uniform grid")
    if not np.all(np.isfinite(amplitude)) or np.any(amplitude < 0):
        raise ValueError("Amplitude must be finite and nonnegative")
    peak = float(np.max(amplitude))
    phase_peak = peak
    if phase_amplitude is not None:
        phase_amplitude = np.asarray(phase_amplitude)
        if (phase_amplitude.shape != amplitude.shape or np.iscomplexobj(phase_amplitude)
                or not np.all(np.isfinite(phase_amplitude)) or np.any(phase_amplitude < 0)):
            raise ValueError("phase_amplitude must be finite, nonnegative and match the phase shape")
        phase_peak = float(np.max(phase_amplitude))
    if amplitude_max is None:
        ceiling = peak if peak > 0 else 1.0
        amplitude_note = "Linear scale for this field." if peak > 0 else "Zero field amplitude."
    else:
        ceiling = float(amplitude_max)
        if not math.isfinite(ceiling) or ceiling <= 0:
            raise ValueError("amplitude_max must be finite and positive")
        amplitude_note = ("Shared linear scale. Values above the scale are saturated."
                          if peak > ceiling else "Shared linear amplitude scale.")

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    image, _, _, (amp_palette, phase_palette) = _graphics()
    low, high = axis[0] - delta[0] / 2, axis[-1] + delta[0] / 2

    if axis.size > _PLOT_SIZE:
        # CW comments: Only the preview is reduced to its display resolution.
        # CW comments: The array archive retains every simulation grid point.
        screen_index = ((np.arange(_PLOT_SIZE) + 0.5) * axis.size / _PLOT_SIZE).astype(np.intp)
        subset = np.ix_(screen_index, screen_index)
        amplitude, phase = amplitude[subset], phase[subset]
        if phase_amplitude is not None:
            phase_amplitude = phase_amplitude[subset]

    amp_index = np.clip(amplitude / ceiling, 0, 1)
    amplitude_ticks = [(i / 4, _number(ceiling * i / 4)) for i in range(5)]
    amp_path = directory / _IMAGES[0]
    _render_panel(amp_palette[np.rint(amp_index * 255).astype(np.uint8)], low, high, amp_palette,
                  amplitude_ticks, amp_path, heading=amplitude_label or "Beam amplitude", title=title,
                  bar_label="√P", note=amplitude_note, resample=image.Resampling.BILINEAR)

    # CW comments: The phase is displayed modulo 2π. Mask undefined phase
    # CW comments: and places below 1% of peak field amplitude in the preview.
    # CW comments: Use the magnitude at the phase reference wavelength for a
    # CW comments: broadband run because the total intensity can fill its nulls.
    reference_magnitude = amplitude if phase_amplitude is None else phase_amplitude
    phase_mask = ~np.isfinite(phase) | (reference_magnitude <= 0.01 * phase_peak)
    phase = np.where(phase_mask, 0, phase)
    outside_range = (phase < -np.pi) | (phase > np.pi)
    if np.any(outside_range):
        phase[outside_range] = (phase[outside_range] + np.pi) % (2 * np.pi) - np.pi
    phase_index = np.clip(np.rint((phase / np.pi + 1) * 127.5), 0, 255).astype(np.uint8)
    phase_rgb = phase_palette[phase_index]
    phase_rgb[phase_mask] = _MASK
    phase_path = directory / _IMAGES[1]
    phase_ticks = list(zip((0, 0.25, 0.5, 0.75, 1), ("−π", "−π/2", "0", "π/2", "π")))
    label = "amplitude" if phase_amplitude is None else "reference amplitude"
    _render_panel(phase_rgb, low, high, phase_palette, phase_ticks, phase_path,
                  heading=phase_label or "Beam phase", title=title, bar_label="rad",
                  note=f"Gray marks undefined phase or {label} at or below 1% of its peak.",
                  resample=image.Resampling.NEAREST)
    return amp_path, phase_path


def _archive_names(config, images):
    yield "manifest.json"
    final_arrays = _FINAL_ARRAYS
    screen_arrays = _SCREEN_ARRAYS
    if not images and getattr(config, "fwhm_nm", 0) > 0:
        final_arrays += ("final_center_amplitude.npy",)
        screen_arrays += ("after_screen_center_amplitude.npy",)
    realization_files = _IMAGES if images else screen_arrays + final_arrays
    if not images:
        for name in _COORDINATES:
            yield f"coordinates/{name}"
    for i in range(1, config.iterations + 1):
        for name in realization_files:
            yield f"realization_{i:04d}/{name}"
    for name in (_IMAGES if images else final_arrays):
        yield f"average/{name}"


def _export_zip(result, destination, *, images):
    root = Path(result.work_dir)
    destination = Path(destination).expanduser()
    names = tuple(_archive_names(result.config, images))
    for name in names:
        if not (root / name).is_file():
            raise FileNotFoundError(f"Simulation output is missing {name}")
    compression = ZIP_STORED if images else ZIP_DEFLATED
    # CW comments: Create the partial archive beside the requested file. Replace
    # CW comments: it after closing so an existing download survives a failed export.
    with tempfile.NamedTemporaryFile(prefix=".optics-", suffix=".zip",
                                     dir=destination.parent, delete=False) as stream:
        working_zip = Path(stream.name)
    try:
        with ZipFile(working_zip, "w", compression=compression,
                     compresslevel=None if images else 1, allowZip64=True) as archive:
            for name in names:
                archive.write(root / name, arcname=name)
        working_zip.replace(destination)
    finally:
        working_zip.unlink(missing_ok=True)
    return destination


def export_data(result, destination):
    """Write all realization and average NumPy arrays to a ZIP64-capable archive."""
    return _export_zip(result, destination, images=False)


def export_images(result, destination):
    """Write the GUI's final-beam PNGs for every realization and the average."""
    return _export_zip(result, destination, images=True)
