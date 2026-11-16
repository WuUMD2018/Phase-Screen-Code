"""Tk desktop interface for the atmospheric optics simulator."""
from collections import OrderedDict
from math import floor, isfinite
import multiprocessing
from pathlib import Path
from queue import Empty, SimpleQueue
import re
from tempfile import TemporaryDirectory
from threading import Event, Thread
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk

from advanced_turbulence import TurbulenceSettings
from studio import (StudioConfig, cn2_to_r0_cm, export_data, export_images,
                    r0_cm_to_cn2, run_studio)


SURFACE = "#f1f5f9"
WHITE = "#ffffff"
INK = "#17293c"
MUTED = "#52677a"
LINE = "#d9e3ed"
PLOT_BG = "#0d1827"
PLOT_TEXT = "#a7bdd4"
ACCENT = "#234fc4"
OUT_OF_RANGE = "#a03622"
FONT = "Segoe UI"
MONO = "Consolas"
FRAME_MS = 300


def scaled_value(text, percentage, *, kind="float"):
    """Parse a typed maximum and map a 1–100% slider to a valid setting."""
    if not isfinite(percentage) or not 1 <= percentage <= 100:
        raise ValueError("The slider must be between 1% and 100%.")
    value = str(text).strip()
    if kind in ("float", "nonnegative"):
        allow_zero = kind == "nonnegative"
        message = ("Enter a finite number of at least 0." if allow_zero
                   else "Enter a positive finite number.")
        try:
            maximum = float(value)
        except ValueError:
            raise ValueError(message) from None
        actual = maximum * (percentage / 100)
        if (not isfinite(maximum) or not isfinite(actual) or maximum < 0
                or (actual <= 0 and (not allow_zero or maximum > 0))):
            raise ValueError(message)
        return actual
    if not re.fullmatch(r"\+?[0-9]+", value):
        raise ValueError("Enter a whole number.")
    maximum = int(value)
    if kind == "even":
        if maximum < 8 or maximum % 2:
            raise ValueError("Enter an even number of at least 8.")
        if percentage == 100:
            return maximum
        minimum = max(8, 2 * ((maximum + 199) // 200))
        return min(maximum, max(minimum, 2 * floor(maximum * (percentage / 200) + 0.5)))
    if maximum < 1:
        raise ValueError("Enter a whole number of at least 1.")
    if percentage == 100:
        return maximum
    return min(maximum, max((maximum + 99) // 100, floor(maximum * (percentage / 100) + 0.5)))


def _display_number(value):
    if isinstance(value, int):
        return f"{value:,}"
    if abs(value) < 0.001 or abs(value) >= 1e6:
        return f"{value:.4g}"
    return f"{value:,.5g}"


def _make_styles(root):
    style = ttk.Style(root)
    style.theme_use("clam")
    style.configure("TButton", font=(FONT, 10, "bold"), padding=(14, 8), relief="flat",
                    borderwidth=0, foreground=INK, background="#e5edf6")
    style.map("TButton", background=[("disabled", "#e8edf2"), ("active", "#d9e6f4")],
              foreground=[("disabled", "#8190a0")])
    style.configure("Primary.TButton", foreground=WHITE, background=ACCENT)
    style.map("Primary.TButton", background=[("disabled", "#c7d3e5"), ("active", "#153ca7")],
              foreground=[("disabled", "#61768f")])
    style.configure("Selected.TButton", foreground=WHITE, background="#294264")
    style.map("Selected.TButton", background=[("active", "#233958")])
    style.configure("Quiet.TButton", padding=(10, 6))
    style.configure("TEntry", padding=(7, 4), foreground=INK, fieldbackground=WHITE,
                    bordercolor=LINE, lightcolor=LINE, darkcolor=LINE)
    style.map("TEntry", fieldbackground=[("disabled", "#f0f4f9"), ("invalid", "#fff4ed")],
              bordercolor=[("invalid", "#db8670"), ("focus", "#7193e0")])
    style.configure("Horizontal.TProgressbar", background="#4671de", troughcolor="#e5edf6",
                    bordercolor=WHITE, lightcolor="#4671de", darkcolor="#4671de", thickness=7)


class PercentageSlider(tk.Canvas):
    def __init__(self, parent, variable):
        super().__init__(parent, height=17, width=140, bg=WHITE, highlightthickness=0,
                         takefocus=True, cursor="hand2")
        self.variable = variable
        self.editable = True
        self.bind("<Configure>", self._draw)
        self.bind("<FocusIn>", self._draw)
        self.bind("<FocusOut>", self._draw)
        self.bind("<Button-1>", self._click)
        self.bind("<B1-Motion>", self._drag)
        self.bind("<Left>", lambda _event: self._move(self.variable.get() - 1))
        self.bind("<Right>", lambda _event: self._move(self.variable.get() + 1))
        self.bind("<Home>", lambda _event: self._move(1))
        self.bind("<End>", lambda _event: self._move(100))
        self.variable.trace_add("write", self._draw)

    def set_editable(self, editable):
        self.editable = editable
        self.configure(takefocus=editable, cursor="hand2" if editable else "arrow")
        self._draw()

    def _bounds(self):
        return 7, max(8, self.winfo_width() - 8)

    def _move(self, percentage):
        if self.editable:
            self.variable.set(max(1, min(100, int(floor(percentage + 0.5)))))
        return "break"

    def _click(self, event):
        if self.editable:
            self.focus_set()
        return self._drag(event)

    def _drag(self, event):
        left, right = self._bounds()
        return self._move(1 + 99 * (event.x - left) / (right - left))

    def _draw(self, *_args):
        left, right = self._bounds()
        center = 8
        thumb = left + (right - left) * (self.variable.get() - 1) / 99
        self.delete("all")
        self.create_line(left, center, right, center, fill="#dfe8f3", width=4, capstyle="round")
        accent = "#5277db" if self.editable else "#a9b5c4"
        self.create_line(left, center, thumb, center, fill=accent, width=4, capstyle="round")
        if self.focus_get() == self and self.editable:
            self.create_oval(thumb - 7, center - 7, thumb + 7, center + 7, outline="#a1b7f0", width=2)
        self.create_oval(thumb - 5, center - 5, thumb + 5, center + 5, fill=accent, outline=WHITE)


class ParameterRow(tk.Frame):
    def __init__(self, parent, label, maximum, changed, *, kind="float", use_slider=True):
        super().__init__(parent, bg=WHITE)
        self.label = label
        self.kind = kind
        self.maximum = tk.StringVar(value=str(maximum))
        self.percentage = tk.DoubleVar(value=100)
        self.percent_label = tk.StringVar(value="100%")
        self.actual_label = tk.StringVar()
        self.columnconfigure(0, weight=1)
        self.name = tk.Label(self, text=label, bg=WHITE, fg=INK, font=(FONT, 9, "bold"), anchor="w")
        self.name.grid(row=0, column=0, columnspan=2, sticky="w")
        self.entry = ttk.Entry(self, textvariable=self.maximum, justify="right", font=(MONO, 10), width=13)
        self.entry.grid(row=0, column=2, sticky="e")
        self.slider = None
        if use_slider:
            self.slider = PercentageSlider(self, self.percentage)
            self.slider.grid(row=1, column=0, sticky="ew", pady=(1, 2))
            tk.Label(self, textvariable=self.percent_label, bg=WHITE, fg=MUTED, font=(MONO, 9), width=5,
                     anchor="e").grid(row=1, column=1, sticky="e", padx=(3, 4))
            self.actual = tk.Label(self, textvariable=self.actual_label, bg=WHITE, fg=ACCENT, width=14,
                                   font=(MONO, 9, "bold"), anchor="e")
            self.actual.grid(row=1, column=2, sticky="e")
        self.maximum.trace_add("write", changed)
        self.percentage.trace_add("write", changed)

    def set_active(self, active, editable):
        state = ["!disabled"] if active and editable else ["disabled"]
        self.entry.state(state)
        if self.slider is not None:
            self.slider.set_editable(active and editable)
            self.actual.configure(fg=ACCENT if active else MUTED)
        self.name.configure(fg=INK if active else MUTED)

    def read(self):
        percentage = 100 if self.slider is None else int(floor(self.percentage.get() + 0.5))
        self.percent_label.set(f"{percentage}%")
        try:
            value = scaled_value(self.maximum.get(), percentage, kind=self.kind)
        except (ValueError, OverflowError):
            self.actual_label.set("—")
            self.entry.state(["invalid"])
            raise
        self.entry.state(["!invalid"])
        self.actual_label.set(_display_number(value))
        return value

    def set_equivalent(self, value):
        self.maximum.set("—" if value is None else f"{value:.7g}")
        self.percentage.set(100)
        self.percent_label.set("100%")
        self.actual_label.set("Equivalent" if value is not None else "—")
        self.entry.state(["!invalid"])


class PlotPane(tk.Frame):
    """A fitted PNG with a small cache for the 300 ms playback loop."""

    def __init__(self, parent, title, placeholder):
        super().__init__(parent, bg=PLOT_BG, highlightthickness=1, highlightbackground="#1c2f48")
        self.title_label = tk.Label(self, text=title, bg=PLOT_BG, fg="#edf4fc",
                                    font=(FONT, 11, "bold"), anchor="w")
        self.title_label.pack(fill="x", padx=12, pady=(10, 2))
        self.canvas = tk.Canvas(self, bg=PLOT_BG, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True, padx=4, pady=(0, 4))
        self.cache = OrderedDict()
        self.path = None
        self.placeholder = placeholder
        self.pending = None
        self.canvas.bind("<Configure>", self._schedule_resize)

    def _schedule_resize(self, _event):
        if self.path is None:
            self._draw()
            return
        if self.pending is not None:
            self.after_cancel(self.pending)
        self.pending = self.after(70, self._resize)

    def _resize(self):
        self.pending = None
        self.cache.clear()
        self._draw()

    def show(self, path=None, *, placeholder=None):
        if placeholder is not None:
            self.placeholder = placeholder
        if path is None or path != self.path:
            self.path = path
            if path is None and not self.title_label.winfo_manager():
                self.title_label.pack(fill="x", padx=12, pady=(10, 2), before=self.canvas)
            elif path is not None and self.title_label.winfo_manager():
                self.title_label.pack_forget()
            self._draw()

    def _draw(self):
        width, height = self.canvas.winfo_width(), self.canvas.winfo_height()
        self.canvas.delete("all")
        if width < 2 or height < 2:
            return
        if self.path is None:
            self.canvas.create_text(width / 2, height / 2, text=self.placeholder,
                                    width=max(50, width - 50), justify="center", fill=PLOT_TEXT,
                                    font=(FONT, 11))
            return
        key = self.path, width, height
        try:
            if key not in self.cache:
                with Image.open(self.path) as source:
                    image = source.convert("RGB")
                image.thumbnail((max(1, width - 4), max(1, height - 4)), Image.Resampling.LANCZOS)
                self.cache[key] = ImageTk.PhotoImage(image, master=self.canvas)
                if len(self.cache) > 8:
                    self.cache.popitem(last=False)
            photo = self.cache[key]
            self.cache.move_to_end(key)
            self.canvas.create_image(width / 2, height / 2, image=photo)
        except (OSError, ValueError) as exc:
            self.canvas.create_text(width / 2, height / 2, text=f"Preview unavailable\n{exc}",
                                    width=max(50, width - 50), justify="center", fill=PLOT_TEXT,
                                    font=(FONT, 10))


class WavefrontStudio:
    """Controller for Tk widgets, workers, and playback."""

    def __init__(self, root):
        self.root = root
        root.title("Wavefront Studio · Atmospheric optics")
        root.configure(bg=SURFACE)
        root.geometry("1220x830")
        root.minsize(980, 740)
        root.protocol("WM_DELETE_WINDOW", self._close)
        _make_styles(root)
        self.mode = "cn2"
        defaults = StudioConfig()
        self.turbulence_defaults = {"cn2": (f"{defaults.cn2:g}", 100.0), "fried": None}
        self.advanced_settings = TurbulenceSettings()
        self.advanced_active = False
        self.view = "single"
        self.busy = None
        self.closing = False
        self.updating = True
        self.cancel_event = Event()
        self.events = SimpleQueue()
        self.temp = None
        self.result = None
        self.run_config = None
        self.ready = set()
        self.frame_index = None
        self.warnings = ()
        self.progress = tk.DoubleVar(value=0)
        self.progress_text = tk.StringVar(value="Ready to simulate")
        self.frame_text = tk.StringVar(value="Receiver images appear as runs finish.")
        self.note = tk.StringVar(value="")
        self.source_text = tk.StringVar()
        self._layout(defaults)
        self.updating = False
        self._settings_changed()
        self._update_controls()
        root.after(45, self._poll)
        root.after(FRAME_MS, self._tick)

    def _layout(self, defaults):
        header = tk.Frame(self.root, bg="#122236", padx=24, pady=13)
        header.pack(fill="x")
        tk.Label(header, text="Wavefront Studio", font=(FONT, 21, "bold"), bg="#122236", fg=WHITE,
                 anchor="w").pack(side="left")
        tk.Label(header, textvariable=self.source_text, font=(FONT, 11),
                 bg="#122236", fg="#bdd0e6").pack(side="right")

        body = tk.Frame(self.root, bg=SURFACE, padx=16, pady=14)
        body.pack(fill="both", expand=True)
        body.grid_columnconfigure(1, weight=1)
        body.grid_rowconfigure(0, weight=1)
        settings = tk.Frame(body, bg=WHITE, padx=14, pady=12,
                            highlightthickness=1, highlightbackground=LINE)
        settings.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        settings.columnconfigure(0, weight=1)
        heading = tk.Frame(settings, bg=WHITE)
        heading.grid(row=0, column=0, sticky="ew")
        tk.Label(heading, text="Simulation settings", bg=WHITE, fg=INK, font=(FONT, 14, "bold"),
                 anchor="w").pack(side="left", fill="x", expand=True)
        self.advanced_button = ttk.Button(heading, text="Advanced…", style="Quiet.TButton",
                                          command=self._open_advanced)
        self.advanced_button.pack(side="right")
        settings_caption = tk.Frame(settings, bg=WHITE)
        settings_caption.grid(row=1, column=0, sticky="ew", pady=(1, 6))
        tk.Label(settings_caption, text="Edit maxima and select 1–100%.", bg=WHITE, fg=MUTED,
                 font=(FONT, 9), anchor="w").pack(side="left", fill="x", expand=True)
        self.advanced_indicator = tk.Label(settings_caption, text="Normal mode", bg="#edf3fa", fg=MUTED,
                                           font=(FONT, 8, "bold"), padx=7, pady=2)
        self.advanced_indicator.pack(side="right")
        self.rows = {}

        def add(key, title, maximum, row, kind="float", *, use_slider=True):
            widget = ParameterRow(settings, title, maximum, self._settings_changed,
                                  kind=kind, use_slider=use_slider)
            widget.grid(row=row, column=0, sticky="ew", pady=(0, 4))
            self.rows[key] = widget

        add("wavelength", "Center wavelength (nm)", f"{defaults.wavelength_nm:g}", 2, use_slider=False)
        add("fwhm", "Maximum FWHM (nm)", f"{defaults.fwhm_nm:g}", 3, "nonnegative")
        add("distance", "Propagation distance (m)", f"{defaults.distance_m:g}", 4)
        add("diameter", "Input beam diameter (cm)", f"{defaults.beam_diameter_cm:g}", 5)
        add("screens", "Phase screens (count)", defaults.screens, 6, "int")
        switcher = tk.Frame(settings, bg=WHITE)
        switcher.grid(row=7, column=0, sticky="ew", pady=(3, 4))
        tk.Label(switcher, text="Turbulence input", bg=WHITE, fg=INK,
                 font=(FONT, 9, "bold")).pack(side="left", padx=(0, 7))
        self.cn2_button = ttk.Button(switcher, text="Cn²", width=5, style="Selected.TButton",
                                     command=lambda: self._change_turbulence("cn2"))
        self.cn2_button.pack(side="left", padx=(0, 3))
        self.fried_button = ttk.Button(switcher, text="Fried r₀", width=6,
                                       command=lambda: self._change_turbulence("fried"))
        self.fried_button.pack(side="left")
        add("cn2", "Cn² (m⁻²⁄³)", f"{defaults.cn2:g}", 8)
        add("fried", "Fried parameter r₀ (cm)", "", 9)
        add("resolution", "Beam grid (samples per side)", defaults.resolution, 10, "even")
        add("iterations", "Realizations (count)", defaults.iterations, 11, "int")
        self.run_button = ttk.Button(settings, text="Simulate", style="Primary.TButton", command=self._start)
        self.run_button.grid(row=12, column=0, sticky="ew", pady=(6, 2))
        tk.Label(settings, text="FWHM 0 selects monochromatic propagation.\nThe input diameter is measured at 1/e² intensity.",
                 bg=WHITE, fg=MUTED, justify="left", anchor="w", font=(FONT, 8)).grid(
                     row=13, column=0, sticky="ew", pady=(3, 0))

        right = tk.Frame(body, bg=SURFACE)
        right.grid(row=0, column=1, sticky="nsew")
        right.grid_columnconfigure(0, weight=1)
        right.grid_rowconfigure(1, weight=1)
        results_top = tk.Frame(right, bg=WHITE, padx=12, pady=9,
                               highlightthickness=1, highlightbackground=LINE)
        results_top.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        labels = tk.Frame(results_top, bg=WHITE)
        labels.pack(side="left", fill="x", expand=True)
        tk.Label(labels, text="Receiver field", bg=WHITE, fg=INK, font=(FONT, 14, "bold"),
                 anchor="w").pack(fill="x")
        tk.Label(labels, textvariable=self.frame_text, bg=WHITE, fg=MUTED, font=(FONT, 9),
                 anchor="w").pack(fill="x")
        self.average_button = ttk.Button(results_top, text="Average", width=7,
                                         command=lambda: self._change_view("average"))
        self.average_button.pack(side="right")
        self.single_button = ttk.Button(results_top, text="Single", width=6, style="Selected.TButton",
                                        command=lambda: self._change_view("single"))
        self.single_button.pack(side="right", padx=(5, 3))

        panes = tk.Frame(right, bg=SURFACE)
        panes.grid(row=1, column=0, sticky="nsew")
        panes.grid_rowconfigure(0, weight=1)
        panes.grid_columnconfigure((0, 1), weight=1, uniform="plot_panes")
        self.amplitude = PlotPane(panes, "Spectral amplitude (√P)", "Run a simulation to see the receiver amplitude.")
        self.amplitude.grid(row=0, column=0, sticky="nsew", padx=(0, 4))
        self.phase = PlotPane(panes, "Phase at center wavelength (rad)",
                              "Run a simulation to see the phase at the center wavelength.")
        self.phase.grid(row=0, column=1, sticky="nsew", padx=(4, 0))

        footer = tk.Frame(right, bg=WHITE, padx=12, pady=10,
                          highlightthickness=1, highlightbackground=LINE)
        footer.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        footer.columnconfigure(0, weight=1)
        tk.Label(footer, textvariable=self.progress_text, bg=WHITE, fg=INK,
                 font=(FONT, 10, "bold"), anchor="w").grid(row=0, column=0, sticky="ew", padx=(0, 8))
        self.download_data = ttk.Button(footer, text="Download data", command=lambda: self._export("data"))
        self.download_data.grid(row=0, column=1, padx=(0, 4))
        self.download_images = ttk.Button(footer, text="Download images", command=lambda: self._export("images"))
        self.download_images.grid(row=0, column=2)
        ttk.Progressbar(footer, variable=self.progress, mode="determinate").grid(
            row=1, column=0, columnspan=3, sticky="ew", pady=(7, 3))
        tk.Label(footer, textvariable=self.note, bg=WHITE, fg=OUT_OF_RANGE, font=(FONT, 9),
                 anchor="w").grid(row=2, column=0, columnspan=2, sticky="ew")
        self.details = ttk.Button(footer, text="Details", style="Quiet.TButton", command=self._show_warnings)
        self.details.grid(row=2, column=2, sticky="e")
        self.details.grid_remove()

    def _settings_changed(self, *_args):
        if self.updating:
            return
        self.updating = True
        try:
            values = {}
            for key, row in self.rows.items():
                if key in ("cn2", "fried") and key != self.mode:
                    continue
                try:
                    values[key] = row.read()
                except (ValueError, OverflowError):
                    pass
            inactive = "fried" if self.mode == "cn2" else "cn2"
            equivalent = None
            if self.mode in values and "distance" in values and "wavelength" in values:
                convert = cn2_to_r0_cm if self.mode == "cn2" else r0_cm_to_cn2
                try:
                    equivalent = convert(values[self.mode], values["distance"], values["wavelength"])
                except (ValueError, OverflowError, ZeroDivisionError):
                    pass
                if equivalent is not None and (not isfinite(equivalent) or equivalent <= 0):
                    equivalent = None
            self.rows[inactive].set_equivalent(equivalent)
            center, fwhm = values.get("wavelength"), values.get("fwhm")
            wavelength_label = "—" if center is None else f"{_display_number(center)} nm"
            bandwidth_label = "—" if fwhm is None else f"{_display_number(fwhm)} nm"
            self.source_text.set(f"Center λ {wavelength_label}  ·  FWHM {bandwidth_label}")
        finally:
            self.updating = False

    def _change_turbulence(self, mode):
        if mode == self.mode or self.busy:
            return
        old = self.rows[self.mode]
        self.turbulence_defaults[self.mode] = (old.maximum.get(), old.percentage.get())
        current = self.rows[mode]
        fallback = "10" if mode == "fried" else "1e-16"
        shown = current.maximum.get()
        saved = self.turbulence_defaults[mode] or (shown if shown != "—" else fallback, 100.0)
        self.updating = True
        self.mode = mode
        current.maximum.set(saved[0])
        current.percentage.set(saved[1])
        self.updating = False
        self._settings_changed()
        self._update_controls()

    def _change_view(self, view):
        self.view = view
        self._update_controls()
        self._render()

    def _open_advanced(self):
        if self.busy:
            return
        try:
            config = self._build_config(include_advanced=False)
        except (ValueError, OverflowError, ZeroDivisionError):
            config = None
        from advanced_gui import AdvancedDialog

        AdvancedDialog(self.root, settings=self.advanced_settings, active=self.advanced_active,
                       config=config, on_apply=self._apply_advanced)

    def _apply_advanced(self, settings, active):
        self.advanced_settings = settings
        self.advanced_active = bool(active)
        self.advanced_indicator.configure(text="Advanced mode" if active else "Normal mode",
                                          bg="#e5edff" if active else "#edf3fa",
                                          fg=ACCENT if active else MUTED)

    def _build_config(self, *, include_advanced=True):
        values = {}
        for key, row in self.rows.items():
            if key in ("cn2", "fried") and key != self.mode:
                continue
            try:
                values[key] = row.read()
            except (ValueError, OverflowError) as exc:
                row.entry.focus_set()
                raise ValueError(f"{row.label}: {exc}") from None
        cn2 = (values["cn2"] if self.mode == "cn2"
               else r0_cm_to_cn2(values["fried"], values["distance"], values["wavelength"]))
        turbulence = self.advanced_settings if include_advanced and self.advanced_active else None
        return StudioConfig(distance_m=values["distance"], beam_diameter_cm=values["diameter"],
                            screens=values["screens"], cn2=cn2, resolution=values["resolution"],
                            iterations=values["iterations"], wavelength_nm=values["wavelength"],
                            fwhm_nm=values["fwhm"], turbulence=turbulence)

    def _update_controls(self):
        state = ["disabled"] if self.busy else ["!disabled"]
        for key, row in self.rows.items():
            row.set_active(key not in ("cn2", "fried") or key == self.mode, not self.busy)
        self.cn2_button.configure(style="Selected.TButton" if self.mode == "cn2" else "TButton")
        self.fried_button.configure(style="Selected.TButton" if self.mode == "fried" else "TButton")
        for button in (self.run_button, self.cn2_button, self.fried_button, self.advanced_button):
            button.state(state)
        self.single_button.configure(style="Selected.TButton" if self.view == "single" else "TButton")
        self.average_button.configure(style="Selected.TButton" if self.view == "average" else "TButton")
        export_state = ["!disabled"] if self.result is not None and not self.busy else ["disabled"]
        self.download_data.state(export_state)
        self.download_images.state(export_state)

    def _show_warnings(self):
        messagebox.showwarning("Sampling diagnostics", "\n\n".join(self.warnings), parent=self.root)

    def _start(self):
        if self.busy:
            return
        try:
            config = self._build_config()
            temp = TemporaryDirectory(prefix="wavefront-studio-")
        except (ValueError, OverflowError, OSError, ZeroDivisionError) as exc:
            messagebox.showwarning("Check simulation settings", str(exc), parent=self.root)
            return
        old_temp, self.temp = self.temp, temp
        self.result = None
        self.run_config = config
        self.ready.clear()
        self.frame_index = None
        self.cancel_event = Event()
        self.busy = "simulation"
        self.warnings = ()
        self.details.grid_remove()
        self.note.set("")
        self.progress.set(0)
        self.progress_text.set(f"Starting {config.iterations:,} realizations…")
        self._update_controls()
        self._render()
        Thread(target=self._run_background, args=(config, Path(temp.name), old_temp), daemon=False).start()

    def _run_background(self, config, work_dir, old_temp):
        try:
            if old_temp is not None:
                old_temp.cleanup()
            result = run_studio(config, work_dir, cancel_event=self.cancel_event,
                                on_result=lambda index, total: self.events.put(("frame", index, total)))
            self.events.put(("done", result))
        except Exception as exc:
            self.events.put(("error", str(exc)))

    def _poll(self):
        while True:
            try:
                kind, *payload = self.events.get_nowait()
            except Empty:
                break
            if kind == "frame":
                index, total = payload
                self.ready.add(index)
                self.progress.set(100 * len(self.ready) / total)
                self.progress_text.set(f"Simulating {len(self.ready):,} of {total:,}")
                if self.frame_index is None and not self.closing:
                    self.frame_index = index
                    self._render()
            elif kind == "done":
                self.busy = None
                self.result = payload[0]
                self.warnings = tuple(self.result.warnings)
                self.progress.set(100)
                self.progress_text.set(f"Complete · {self.result.elapsed_s:.2f} s · {self.result.workers} worker(s)")
                if self.warnings:
                    self.note.set(f"{len(self.warnings)} sampling diagnostic(s) recorded.")
                    self.details.grid()
                self._update_controls()
                self._render()
            elif kind == "exported":
                self.busy = None
                self.progress_text.set(f"Saved {Path(payload[0]).name}")
                self._update_controls()
            elif kind == "error":
                was_run = self.busy == "simulation"
                self.busy = None
                self.progress_text.set("Simulation stopped" if was_run else "Export stopped")
                self._update_controls()
                if not self.closing:
                    messagebox.showerror("Wavefront Studio", payload[0] or "The operation did not finish.",
                                         parent=self.root)
        if self.closing and self.busy is None:
            self._close()
            return
        self.root.after(45, self._poll)

    def _render(self):
        if self.closing:
            return
        center_label = ("center λ" if self.run_config is None
                        else f"{_display_number(self.run_config.wavelength_nm)} nm")
        if self.view == "average":
            if self.result is None:
                prompt = "The ensemble average appears when all realizations are complete."
                self.frame_text.set(f"RMS amplitude · circular mean phase at {center_label}")
                self.amplitude.show(placeholder=prompt)
                self.phase.show(placeholder=prompt)
                return
            folder = Path(self.result.work_dir) / "average"
            self.frame_text.set(f"RMS amplitude · circular mean phase at {center_label}")
        else:
            if self.frame_index is None:
                prompt = "Waiting for the first realization." if self.busy else "Run a simulation to see the receiver field."
                self.frame_text.set(prompt)
                self.amplitude.show(placeholder=prompt)
                self.phase.show(placeholder=prompt)
                return
            folder = Path(self.temp.name) / f"realization_{self.frame_index + 1:04d}"
            self.frame_text.set(f"Realization {self.frame_index + 1:,} · Phase at {center_label} · 0.3 s playback")
        self.amplitude.show(folder / "amplitude.png")
        self.phase.show(folder / "phase.png")

    def _tick(self):
        if not self.closing and self.view == "single" and self.ready:
            completed = sorted(self.ready)
            if self.frame_index in self.ready:
                self.frame_index = completed[(completed.index(self.frame_index) + 1) % len(completed)]
            else:
                self.frame_index = completed[0]
            self._render()
        self.root.after(FRAME_MS, self._tick)

    def _export(self, kind):
        if self.result is None or self.busy:
            return
        name = "wavefront_data.zip" if kind == "data" else "wavefront_images.zip"
        destination = filedialog.asksaveasfilename(parent=self.root, title=f"Save {kind} archive",
                                                  initialfile=name, defaultextension=".zip",
                                                  filetypes=[("ZIP archive", "*.zip")])
        if not destination:
            return
        self.busy = "export"
        self.progress_text.set(f"Saving {kind} ZIP…")
        self._update_controls()
        export = export_data if kind == "data" else export_images

        def write():
            try:
                export(self.result, Path(destination))
                self.events.put(("exported", destination))
            except Exception as exc:
                self.events.put(("error", str(exc)))

        Thread(target=write, daemon=False).start()

    def _close(self):
        if self.busy is not None:
            self.closing = True
            if self.busy == "simulation":
                self.cancel_event.set()
            self.progress_text.set("Finishing background work before closing…")
            return
        if self.temp is not None:
            self.temp.cleanup()
            self.temp = None
        self.root.destroy()


def main():
    multiprocessing.freeze_support()
    root = tk.Tk()
    WavefrontStudio(root)
    root.mainloop()


if __name__ == "__main__":
    main()
