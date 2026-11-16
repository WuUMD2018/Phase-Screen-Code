"""Modal spectrum controls and validation previews for Wavefront Studio."""
import json
from math import isfinite
from pathlib import Path
from queue import Empty, SimpleQueue
from tempfile import TemporaryDirectory
from threading import Event, Thread
import tkinter as tk
from tkinter import ttk

from PIL import Image, ImageTk

from advanced_turbulence import TurbulenceSettings


SURFACE = "#f1f5f9"
WHITE = "#ffffff"
INK = "#17293c"
MUTED = "#52677a"
LINE = "#d9e3ed"
ACCENT = "#234fc4"
ISSUE = "#a03622"
PLOT_BG = "#0d1827"
FONT = "Segoe UI"
MONO = "Consolas"


class _LimitControl(tk.Frame):
    def __init__(self, parent, title, limit, percent, changed, *, alpha_floor=None):
        super().__init__(parent, bg=WHITE)
        self.title = title
        self.alpha_floor = alpha_floor
        self.limit = tk.StringVar(self, value=f"{limit:.16g}")
        self.percent = tk.DoubleVar(self, value=percent)
        self.value_text = tk.StringVar(self)
        self.columnconfigure(0, weight=1)
        tk.Label(self, text=title, bg=WHITE, fg=INK, font=(FONT, 10, "bold"), anchor="w").grid(
            row=0, column=0, sticky="ew")
        self.entry = ttk.Entry(self, textvariable=self.limit, font=(MONO, 10), justify="right", width=11)
        self.entry.grid(row=0, column=1, sticky="e")
        self.slider = ttk.Scale(self, variable=self.percent, from_=0 if alpha_floor is not None else 1,
                                to=100, orient="horizontal", style="Advanced.Horizontal.TScale")
        self.slider.grid(row=1, column=0, sticky="ew", pady=(4, 0), padx=(0, 8))
        tk.Label(self, textvariable=self.value_text, bg=WHITE, fg=ACCENT, font=(MONO, 9, "bold"),
                 anchor="e").grid(row=1, column=1, sticky="ew", pady=(4, 0))
        self.limit.trace_add("write", changed)
        self.percent.trace_add("write", changed)

    def set_enabled(self, enabled):
        state = ["!disabled"] if enabled else ["disabled"]
        self.entry.state(state)
        self.slider.state(state)

    def read(self):
        try:
            maximum = float(self.limit.get())
        except (ValueError, OverflowError):
            maximum = float("nan")
        percent = self.percent.get()
        if self.alpha_floor is None:
            value = maximum * percent / 100
            problem = None if isfinite(value) and value > 0 else f"{self.title} must be positive and finite."
        else:
            value = min(maximum, self.alpha_floor + (maximum - self.alpha_floor) * (percent / 100))
            problem = (None if isfinite(maximum) and self.alpha_floor <= maximum < 4 else
                       f"Max α must be at least {self.alpha_floor:.6g} and below 4.")
        self.entry.state(["invalid"] if problem else ["!invalid"])
        self.value_text.set("—" if problem else f"{percent:.0f}% · {value:.5g}")
        if problem:
            raise ValueError(problem)
        return value

    def draft(self):
        return self.limit.get(), self.percent.get()


class _AxisControl(tk.Frame):
    def __init__(self, parent, title, value, changed):
        super().__init__(parent, bg=WHITE)
        self.enabled = tk.BooleanVar(self, value=value != 1)
        self.value = tk.DoubleVar(self, value=value)
        self.value_text = tk.StringVar(self)
        self.columnconfigure(0, weight=1)
        self.toggle = ttk.Checkbutton(self, text=title, variable=self.enabled, style="Advanced.TCheckbutton")
        self.toggle.grid(row=0, column=0, sticky="w")
        tk.Label(self, textvariable=self.value_text, bg=WHITE, fg=ACCENT, font=(MONO, 9, "bold"),
                 anchor="e").grid(row=0, column=1, sticky="e")
        self.slider = ttk.Scale(self, variable=self.value, from_=0.5, to=2,
                                orient="horizontal", style="Advanced.Horizontal.TScale")
        self.slider.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(2, 0))
        self.enabled.trace_add("write", changed)
        self.value.trace_add("write", changed)

    def set_enabled(self, enabled):
        self.toggle.state(["!disabled"] if enabled else ["disabled"])
        self.slider.state(["!disabled"] if enabled and self.enabled.get() else ["disabled"])

    def read(self):
        value = self.value.get() if self.enabled.get() else 1.0
        self.value_text.set(f"{value:.3f}×")
        return value

    def draft(self):
        return self.enabled.get(), self.value.get()


class _Preview(tk.Canvas):
    def __init__(self, parent, placeholder):
        super().__init__(parent, width=450, height=350, bg=PLOT_BG, highlightthickness=0)
        self.placeholder = placeholder
        self.source = self.photo = self._pending = None
        self.actual_size = False
        self.bind("<Configure>", self._schedule)

    def show(self, image=None):
        self.source = image
        self._schedule()

    def set_actual_size(self, actual_size):
        self.actual_size = actual_size
        self.xview_moveto(0)
        self.yview_moveto(0)
        self._schedule()

    def _schedule(self, _event=None):
        if self._pending is not None:
            self.after_cancel(self._pending)
        self._pending = self.after(45, self._draw)

    def _draw(self):
        self._pending = None
        width, height = self.winfo_width(), self.winfo_height()
        if width < 4 or height < 4:
            return
        self.delete("all")
        if self.source is None:
            self.photo = None
            self.create_text(width / 2, height / 2, text=self.placeholder,
                             width=max(25, width - 40), justify="center", fill="#bdd0e6", font=(FONT, 11))
            return
        if self.actual_size:
            self.photo = ImageTk.PhotoImage(self.source, master=self)
            self.configure(scrollregion=(0, 0, *self.source.size))
            self.create_image(0, 0, anchor="nw", image=self.photo)
        else:
            image = self.source.copy()
            image.thumbnail((max(1, width - 6), max(1, height - 6)), Image.Resampling.LANCZOS)
            self.photo = ImageTk.PhotoImage(image, master=self)
            self.configure(scrollregion="")
            self.create_image(width / 2, height / 2, image=self.photo)

    def cancel_draw(self):
        if self._pending is not None:
            self.after_cancel(self._pending)
            self._pending = None


def _validate_in_background(config, settings, cancel, messages):
    """Load private PNGs before exiting the temporary folder and notifying Tk."""
    try:
        from advanced_validation import validate_advanced

        with TemporaryDirectory(prefix="wavefront-validation-") as directory:
            result = validate_advanced(config, settings, Path(directory), cancel_event=cancel)
            if not cancel.is_set():
                with Image.open(result.screen_image) as screen, Image.open(result.spectrum_image) as spectrum:
                    payload = (screen.convert("RGB"), spectrum.convert("RGB"),
                               tuple(str(item) for item in result.warnings), result.report)
        if cancel.is_set():
            messages.put(("canceled", None))
        else:
            messages.put(("result", payload))
    except Exception as exc:
        messages.put(("canceled", None) if cancel.is_set() else ("error", str(exc) or type(exc).__name__))


class AdvancedDialog(tk.Toplevel):
    """Edit turbulence inputs and check their spectra without changing the main mode."""

    def __init__(self, parent, *, settings, active, config, on_apply):
        super().__init__(parent)
        self.withdraw()
        self.parent = parent
        self.settings = settings
        self.config = config
        self.on_apply = on_apply
        self.busy = self.closing = self.initializing = True
        self._poll_id = None
        self._result = None
        self._cancel = Event()
        self._messages = SimpleQueue()
        self._last_valid = settings
        self._last_form = None
        self.title("Advanced Turbulence Settings · Wavefront Studio")
        self.transient(parent)
        self.configure(bg=SURFACE)
        self.protocol("WM_DELETE_WINDOW", self._close)
        self.bind("<Escape>", lambda _event: self._close())
        self.minsize(810, 595)
        style = ttk.Style(self)
        style.configure("Advanced.Horizontal.TScale", background="#5277db", troughcolor="#dce7f3",
                        bordercolor="#a1b7f0", lightcolor="#a1b7f0", darkcolor="#365ac7",
                        sliderthickness=15)
        style.map("Advanced.Horizontal.TScale", background=[("disabled", "#a9b5c4"),
                                                             ("active", "#365ac7")])
        style.configure("Advanced.TCheckbutton", background=WHITE, foreground=INK, font=(FONT, 10, "bold"))
        style.map("Advanced.TCheckbutton", background=[("active", WHITE)],
                  foreground=[("disabled", MUTED)])
        style.configure("Advanced.TNotebook", background=SURFACE, borderwidth=0, padding=0)
        style.configure("Advanced.TNotebook.Tab", background="#e5edf6", foreground=MUTED,
                        font=(FONT, 10, "bold"), padding=(14, 7))
        style.map("Advanced.TNotebook.Tab", background=[("selected", PLOT_BG)],
                  foreground=[("selected", WHITE)])
        self.status = tk.StringVar(self, value="")
        self.form_error = tk.StringVar(self, value="")
        self._layout(active)
        previous = getattr(parent, "_wavefront_advanced_form", None)
        if previous is not None and previous[0] == settings:
            self._restore_draft(previous[1])
        self.initializing = self.closing = self.busy = False
        self._changed()
        self.update_idletasks()
        width = min(1120, max(810, self.winfo_screenwidth() - 30))
        height = min(710, max(595, self.winfo_screenheight() - 85))
        left = max(0, min(parent.winfo_rootx() + (parent.winfo_width() - width) // 2,
                          self.winfo_screenwidth() - width - 10))
        top = max(0, min(parent.winfo_rooty() + (parent.winfo_height() - height) // 2,
                         self.winfo_screenheight() - height - 50))
        self.geometry(f"{width}x{height}+{left}+{top}")
        self.deiconify()
        self.wait_visibility()
        self.grab_set()
        self.limits[0].entry.focus_set()

    def _layout(self, active):
        settings = self.settings
        header = tk.Frame(self, bg="#122236", padx=20, pady=11)
        header.pack(fill="x")
        tk.Label(header, text="Advanced Turbulence Settings", bg="#122236", fg=WHITE,
                 font=(FONT, 18, "bold"), anchor="w").pack(side="left")
        tk.Label(header, text=f"Current mode  ·  {'Advanced' if active else 'Normal'}", bg="#122236",
                 fg="#bdd0e6", font=(FONT, 10), anchor="e").pack(side="right")

        footer = tk.Frame(self, bg=WHITE, padx=14, pady=10, highlightthickness=1, highlightbackground=LINE)
        footer.pack(side="bottom", fill="x", padx=14, pady=(0, 12))
        self.close_button = ttk.Button(footer, text="Close", style="Quiet.TButton", command=self._close)
        self.close_button.pack(side="left")
        self.apply_advanced = ttk.Button(footer, text="Apply Advanced Setting", style="Primary.TButton",
                                         command=lambda: self._apply(True))
        self.apply_advanced.pack(side="right")
        self.apply_normal = ttk.Button(footer, text="Apply Normal Setting", command=lambda: self._apply(False))
        self.apply_normal.pack(side="right", padx=(0, 7))

        body = tk.Frame(self, bg=SURFACE, padx=14, pady=12)
        body.pack(fill="both", expand=True)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)
        form = tk.Frame(body, bg=WHITE, padx=13, pady=11, highlightthickness=1, highlightbackground=LINE)
        form.grid(row=0, column=0, sticky="nsew", padx=(0, 11))
        form.columnconfigure(0, minsize=292, weight=1)
        tk.Label(form, text="Spectrum parameters", bg=WHITE, fg=INK, font=(FONT, 13, "bold"),
                 anchor="w").grid(row=0, sticky="ew")
        alpha_floor = min(3.001, settings.alpha)
        alpha_ceiling = max(3.999, settings.alpha)
        alpha_percent = 100 * (settings.alpha - alpha_floor) / (alpha_ceiling - alpha_floor)
        alpha_help = f"The α slider runs from {alpha_floor:.6g} to Max α. Scale sliders select 1–100% of each maximum."
        tk.Label(form, text=alpha_help, bg=WHITE, fg=MUTED, font=(FONT, 9), anchor="w",
                 wraplength=285, justify="left").grid(row=1, sticky="ew", pady=(1, 8))
        self.limits = (
            _LimitControl(form, "Max α", alpha_ceiling, alpha_percent, self._changed, alpha_floor=alpha_floor),
            _LimitControl(form, "Maximum inner scale (mm)", settings.inner_scale_mm, 100, self._changed),
            _LimitControl(form, "Maximum outer scale (mm)", settings.outer_scale_mm, 100, self._changed),
        )
        for index, row in enumerate(self.limits, 2):
            row.grid(row=index, sticky="ew", pady=(0, 8))
        ttk.Separator(form).grid(row=5, sticky="ew", pady=(2, 7))
        tk.Label(form, text="Anisotropy", bg=WHITE, fg=INK, font=(FONT, 11, "bold"),
                 anchor="w").grid(row=6, sticky="ew")
        tk.Label(form, text="Axis scales run from 0.5× to 2×. An unchecked axis uses 1×.", bg=WHITE, fg=MUTED,
                 font=(FONT, 9), anchor="w", wraplength=285, justify="left").grid(row=7, sticky="ew", pady=(0, 5))
        self.axes = (
            _AxisControl(form, "Scale major axis", settings.major_scale, self._changed),
            _AxisControl(form, "Scale minor axis", settings.minor_scale, self._changed),
        )
        for index, row in enumerate(self.axes, 8):
            row.grid(row=index, sticky="ew", pady=(0, 6))
        tk.Label(form, textvariable=self.form_error, bg=WHITE, fg=ISSUE, font=(FONT, 9), anchor="w",
                 justify="left", wraplength=285).grid(row=10, sticky="ew", pady=(3, 0))

        validation = tk.Frame(body, bg=SURFACE)
        validation.grid(row=0, column=1, sticky="nsew")
        validation.rowconfigure(1, weight=1)
        validation.columnconfigure(0, weight=1)
        controls = tk.Frame(validation, bg=WHITE, padx=11, pady=8, highlightthickness=1, highlightbackground=LINE)
        controls.grid(row=0, sticky="ew", pady=(0, 8))
        tk.Label(controls, text="Compare turbulence models", bg=WHITE, fg=INK, font=(FONT, 13, "bold"),
                 anchor="w").pack(side="left")
        self.validate_button = ttk.Button(controls, text="Validate Settings", style="Primary.TButton",
                                          command=self._start_validation)
        self.validate_button.pack(side="right")
        self.tabs = ttk.Notebook(validation, style="Advanced.TNotebook")
        self.tabs.grid(row=1, sticky="nsew")
        self.screen_preview = _Preview(self.tabs, "Validate Settings to compare phase screens on the same grid.")
        self.spectrum_preview = _Preview(self.tabs, "Validate Settings to compare directional spectral slopes.")
        self.tabs.add(self.screen_preview, text="Phase screens")
        self.tabs.add(self.spectrum_preview, text="Power spectra")

        self.spectrum_guide = tk.Frame(validation, bg=WHITE, padx=10, pady=5,
                                       highlightthickness=1, highlightbackground=LINE)
        self.spectrum_guide.grid(row=2, sticky="ew", pady=(5, 0))
        self.spectrum_guide.grid_remove()
        self.enlarge_button = ttk.Button(self.spectrum_guide, text="Enlarge PSD…", style="Quiet.TButton",
                                         command=self._enlarge_spectrum)
        self.enlarge_button.pack(side="right", padx=(6, 0))
        guide_text = tk.Label(self.spectrum_guide, bg=WHITE, fg=MUTED, font=(FONT, 9), anchor="w",
                              text="Measured and model fits use matching bins on each axis. Details lists the cutoffs and sampling limits.",
                              justify="left", wraplength=290)
        guide_text.pack(side="left", expand=True, fill="x")
        self.spectrum_guide.bind("<Configure>", lambda event: guide_text.configure(
            wraplength=max(80, event.width - self.enlarge_button.winfo_reqwidth() - 32)))
        self.tabs.bind("<<NotebookTabChanged>>", self._tab_changed)

        status_area = tk.Frame(validation, bg=WHITE, padx=10, pady=8, highlightthickness=1, highlightbackground=LINE)
        status_area.grid(row=3, sticky="ew", pady=(8, 0))
        self.status_label = tk.Label(status_area, textvariable=self.status, bg=WHITE, fg=MUTED, font=(FONT, 9),
                                     justify="left", anchor="w", wraplength=390)
        self.status_label.pack(side="left", expand=True, fill="x")
        self.details_button = ttk.Button(status_area, text="Details…", style="Quiet.TButton",
                                         command=self._show_details)
        self.details_button.pack(side="right")
        self.progress = ttk.Progressbar(validation, mode="indeterminate")
        self.progress.grid(row=4, sticky="ew", pady=(5, 0))
        self.progress.grid_remove()

    def _tab_changed(self, _event):
        if self.tabs.select() == str(self.spectrum_preview):
            self.spectrum_guide.grid()
        else:
            self.spectrum_guide.grid_remove()

    def _read(self):
        values, problems = [], []
        for row in self.limits:
            try:
                values.append(row.read())
            except ValueError as exc:
                problems.append(str(exc))
        major, minor = (row.read() for row in self.axes)
        if problems:
            raise ValueError(problems[0])
        alpha, inner, outer = values
        if outer <= inner:
            raise ValueError("Outer scale must be larger than inner scale.")
        if major < minor:
            raise ValueError("Major axis scale must be at least the minor axis scale.")
        return TurbulenceSettings(alpha=alpha, inner_scale_mm=inner, outer_scale_mm=outer,
                                  major_scale=major, minor_scale=minor)

    def _changed(self, *_args):
        if self.initializing or self.busy:
            return
        try:
            self._last_valid = self._read()
        except ValueError as exc:
            self.form_error.set(str(exc))
        else:
            self.form_error.set("")
            self._last_form = tuple(row.draft() for row in (*self.limits, *self.axes))
        if self._result is not None:
            self._result = None
            self.screen_preview.show()
            self.spectrum_preview.show()
        self._set_status("Choose Validate Settings to compare these parameters." if self.config is not None else
                         "Correct the main simulation settings to enable validation.")
        self._update_controls()

    def _restore_draft(self, draft):
        for row, (limit, percent) in zip(self.limits, draft[:3]):
            row.limit.set(limit)
            row.percent.set(percent)
        for row, (enabled, value) in zip(self.axes, draft[3:]):
            row.enabled.set(enabled)
            row.value.set(value)

    def _set_status(self, message, *, warning=False):
        self.status.set(message)
        self.status_label.configure(fg=ISSUE if warning else MUTED)

    def _update_controls(self):
        enabled = not self.busy
        valid = not self.form_error.get()
        for row in (*self.limits, *self.axes):
            row.set_enabled(enabled)
        self.validate_button.state(["!disabled"] if enabled and valid and self.config is not None else ["disabled"])
        self.apply_advanced.state(["!disabled"] if enabled and valid else ["disabled"])
        self.apply_normal.state(["!disabled"] if enabled else ["disabled"])
        self.details_button.state(["!disabled"] if enabled and self._result is not None else ["disabled"])
        self.enlarge_button.state(["!disabled"] if enabled and self._result is not None else ["disabled"])
        self.close_button.state(["disabled"] if self.closing else ["!disabled"])

    def _start_validation(self):
        if self.busy or self.form_error.get() or self.config is None:
            return
        self.busy = True
        self._result = None
        self.screen_preview.show()
        self.spectrum_preview.show()
        self._set_status("Generating phase screens and power spectra…")
        self._update_controls()
        self.progress.grid()
        self.progress.start(12)
        Thread(target=_validate_in_background, args=(self.config, self._last_valid, self._cancel, self._messages),
               daemon=False, name="wavefront-validation").start()
        self._poll_id = self.after(45, self._poll)

    def _poll(self):
        self._poll_id = None
        try:
            kind, payload = self._messages.get_nowait()
        except Empty:
            self._poll_id = self.after(45, self._poll)
            return
        self.busy = False
        self.progress.stop()
        self.progress.grid_remove()
        if self.closing:
            self._destroy_dialog()
            return
        if kind == "result":
            screen, spectrum, warnings, report = payload
            self._result = warnings, report
            self.screen_preview.show(screen)
            self.spectrum_preview.show(spectrum)
            if warnings:
                suffix = f" (+{len(warnings) - 1} more)" if len(warnings) > 1 else ""
                first = warnings[0] if len(warnings[0]) <= 125 else warnings[0][:122] + "…"
                self._set_status(first + suffix, warning=True)
            else:
                self._set_status("Validation complete. Sampling diagnostics are clear.")
        elif kind == "error":
            self._set_status(f"Validation could not finish. {payload}", warning=True)
        self._update_controls()

    def _enlarge_spectrum(self):
        image = self.spectrum_preview.source
        if image is None:
            return
        dialog = tk.Toplevel(self, bg=SURFACE)
        dialog.title("Directional power spectra · Wavefront Studio")
        dialog.transient(self)
        screen_width, screen_height = self.winfo_screenwidth(), self.winfo_screenheight()
        width = min(image.width + 28, screen_width - 42)
        height = min(image.height + 88, screen_height - 110)
        dialog.geometry(f"{width}x{height}+{(screen_width - width) // 2}+{(screen_height - height) // 2}")
        dialog.minsize(580, 360)
        footer = tk.Frame(dialog, bg=SURFACE, padx=10, pady=8)
        footer.pack(side="bottom", fill="x")
        body = tk.Frame(dialog, bg=SURFACE, padx=10, pady=10)
        body.pack(fill="both", expand=True)
        body.columnconfigure(0, weight=1)
        body.rowconfigure(0, weight=1)
        preview = _Preview(body, "")
        preview.grid(row=0, column=0, sticky="nsew")
        preview.show(image)
        vertical = ttk.Scrollbar(body, orient="vertical", command=preview.yview)
        horizontal = ttk.Scrollbar(body, orient="horizontal", command=preview.xview)
        preview.configure(yscrollcommand=vertical.set, xscrollcommand=horizontal.set)

        def toggle_scale():
            actual = not preview.actual_size
            if actual:
                vertical.grid(row=0, column=1, sticky="ns")
                horizontal.grid(row=1, column=0, sticky="ew")
            else:
                vertical.grid_remove()
                horizontal.grid_remove()
            preview.set_actual_size(actual)
            scale_button.configure(text="Fit to window" if actual else "Actual size (100%)")

        def close():
            preview.cancel_draw()
            dialog.destroy()
            self.grab_set()
            self.enlarge_button.focus_set()

        scale_button = ttk.Button(footer, text="Actual size (100%)", command=toggle_scale)
        scale_button.pack(side="left")
        ttk.Button(footer, text="Done", command=close).pack(side="right")
        dialog.protocol("WM_DELETE_WINDOW", close)
        dialog.bind("<Escape>", lambda _event: close())
        dialog.wait_visibility()
        dialog.grab_set()
        scale_button.focus_set()

    def _show_details(self):
        if self._result is None:
            return
        warnings, report = self._result
        dialog = tk.Toplevel(self, bg=SURFACE)
        dialog.title("Validation details")
        dialog.transient(self)
        dialog.geometry("740x450")
        frame = tk.Frame(dialog, bg=SURFACE, padx=13, pady=11)
        frame.pack(fill="both", expand=True)
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)
        box = tk.Text(frame, wrap="none", font=(MONO, 10), fg=INK, bg=WHITE, relief="flat", padx=9, pady=9)
        box.grid(row=0, column=0, sticky="nsew")
        vertical = ttk.Scrollbar(frame, orient="vertical", command=box.yview)
        vertical.grid(row=0, column=1, sticky="ns")
        horizontal = ttk.Scrollbar(frame, orient="horizontal", command=box.xview)
        horizontal.grid(row=1, column=0, sticky="ew")
        box.configure(yscrollcommand=vertical.set, xscrollcommand=horizontal.set)
        summary = "\n".join(f"• {item}" for item in warnings) if warnings else "No sampling warnings."
        box.insert("1.0", f"Sampling diagnostics\n{summary}\n\nValidation report\n"
                   + json.dumps(report, indent=2, ensure_ascii=False, default=str))
        box.configure(state="disabled")

        def close():
            dialog.destroy()
            self.grab_set()
            self.details_button.focus_set()

        ttk.Button(frame, text="Done", command=close).grid(row=2, column=0, sticky="e", pady=(9, 0))
        dialog.protocol("WM_DELETE_WINDOW", close)
        dialog.bind("<Escape>", lambda _event: close())
        dialog.wait_visibility()
        dialog.grab_set()
        box.focus_set()

    def _apply(self, active):
        if self.busy or (active and self.form_error.get()):
            return
        self.on_apply(self._last_valid, active)
        if self._last_form is not None:
            self.parent._wavefront_advanced_form = (self._last_valid, self._last_form)
        self._destroy_dialog()

    def _close(self):
        if self.busy:
            self.closing = True
            self._cancel.set()
            self._set_status("Stopping validation and closing…")
            self._update_controls()
            return
        self._destroy_dialog()

    def _destroy_dialog(self):
        if self._poll_id is not None:
            self.after_cancel(self._poll_id)
        self.screen_preview.cancel_draw()
        self.spectrum_preview.cancel_draw()
        self.grab_release()
        self.destroy()
