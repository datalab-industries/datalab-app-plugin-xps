"""XPS block with Shirley background subtraction and Voigt peak fitting."""

import warnings
from pathlib import Path

import numpy as np
from bokeh.layouts import column
from bokeh.models import Button, CustomJS, Div, TextInput
from pydatalab.blocks.base import DataBlock, event, generate_js_callback_single_float_parameter
from pydatalab.bokeh_plots import DATALAB_BOKEH_THEME

from datalab_app_plugin_digibat._version import __version__


def shirley_background(x, y, tol=1e-5, max_iter=100):
    """Compute iterative Shirley background for XPS data.

    Parameters:
        x: Binding energy array.
        y: Intensity array.
        tol: Convergence tolerance.
        max_iter: Maximum number of iterations.

    Returns:
        background: The Shirley background array.
    """
    x = np.array(x)
    y = np.array(y)

    background = np.linspace(y[0], y[-1], len(y))

    for _ in range(max_iter):
        prev = background.copy()

        diff = y - background
        integral = np.cumsum(diff[::-1])[::-1]

        background = y[-1] + (y[0] - y[-1]) * integral / integral[0]

        if np.linalg.norm(background - prev) < tol:
            break

    return background


class XPSBlock(DataBlock):
    version = __version__
    blocktype: str = "xps"
    name: str = "XPS Block"
    description: str = (
        "A block that loads an XPS .VGD file, subtracts a Shirley background, and fits Voigt peaks."
    )
    accepted_file_extensions = (".VGD", ".vgd")

    defaults = {
        "num_peaks": 3,
        "peak_centers": "284, 286, 290",
        "run_fit": False,
    }

    @property
    def plot_functions(self):
        return (self.plot_XPS,)

    @event()
    def set_num_peaks(self, num_peaks: str):
        """Store the user-supplied number of peaks."""
        try:
            n = int(num_peaks)
        except ValueError:
            raise ValueError(f"num_peaks must be an integer, got {num_peaks!r}")
        if not (1 <= n <= 20):
            raise ValueError(f"num_peaks must be between 1 and 20, got {n}")
        self.data["num_peaks"] = n

    @event()
    def set_peak_centers(self, peak_centers: str):
        """Store the user-supplied peak center positions as a comma-separated string."""
        parts = [p.strip() for p in peak_centers.split(",")]
        try:
            centers = [float(p) for p in parts if p]
        except ValueError:
            raise ValueError(f"peak_centers must be comma-separated numbers, got {peak_centers!r}")
        if not centers:
            raise ValueError("peak_centers must not be empty")
        self.data["peak_centers"] = ", ".join(str(c) for c in centers)

    @event()
    def trigger_fit(self, trigger: str):
        """Enable peak fitting on next plot render (called by the Fit button)."""
        self.data["run_fit"] = True

    def plot_XPS(self, filename: str | Path | None = None):
        """Creates an XPS plot with Shirley background and Voigt peak fitting.

        Parameters:
            filename: Path to the .VGD file. If None, retrieves from the database
                using the `file_id` stored in `self.data`.
        """
        import bokeh.embed
        from bokeh.plotting import figure
        from vgd_reader import read_vgd

        file_info = None
        if not filename:
            try:
                from pydatalab.file_utils import get_file_info_by_id
            except ImportError:
                raise RuntimeError(
                    "The `datalab-server[server]` extra must be installed to use this block with a database."
                )

            if "file_id" not in self.data:
                return

            try:
                file_info = get_file_info_by_id(self.data["file_id"], update_if_live=True)
            except StopIteration:
                return
            if not file_info:
                return
            filename = Path(file_info["location"])

        data = read_vgd(filename)
        x = data.binding_energy
        y = data.corrected_intensity

        plot_title = file_info["name"] if file_info else Path(filename).name

        # --- Shirley background ---
        bg = None
        y_sub = y.copy()
        try:
            bg = shirley_background(x, y)
            y_sub = y - bg
        except Exception as exc:
            warnings.warn(f"Shirley background failed: {exc}")

        # --- Parameters ---
        num_peaks = int(self.data.get("num_peaks", self.defaults["num_peaks"]))
        peak_centers_str = self.data.get("peak_centers", self.defaults["peak_centers"])
        run_fit = bool(self.data.get("run_fit", self.defaults["run_fit"]))
        try:
            centers = [float(p.strip()) for p in peak_centers_str.split(",") if p.strip()]
        except ValueError:
            centers = []

        # Reconcile centers list with num_peaks
        if len(centers) < num_peaks:
            extra = np.linspace(x.min(), x.max(), num_peaks - len(centers) + 2)[1:-1]
            centers = centers + list(extra[: num_peaks - len(centers)])
        centers = centers[:num_peaks]

        # --- Voigt peak fitting (only when explicitly requested) ---
        fit_result = None
        components = {}
        if run_fit:
            try:
                from lmfit.models import VoigtModel

                model = None
                for i in range(1, num_peaks + 1):
                    vm = VoigtModel(prefix=f"p{i}_")
                    model = vm if model is None else model + vm

                if model:
                    params = model.make_params()
                    for i in range(1, num_peaks + 1):
                        params[f"p{i}_center"].set(value=centers[i - 1])
                        params[f"p{i}_amplitude"].set(min=0)

                    fit_result = model.fit(y_sub, params, x=x)
                    components = fit_result.eval_components(x=x)
            except Exception as exc:
                warnings.warn(f"Voigt peak fitting failed: {exc}")
            finally:
                # Reset so re-renders don't re-run the fit automatically
                self.data["run_fit"] = False

        # --- Bokeh figure ---
        p = figure(
            title=plot_title,
            x_axis_label="Binding Energy (eV)",
            y_axis_label="Intensity",
        )
        p.x_range.flipped = True  # XPS convention: high BE on left

        p.line(x, y, legend_label="Raw data", line_width=2, color="black")

        if bg is not None:
            p.line(
                x,
                bg,
                legend_label="Shirley background",
                line_width=2,
                color="gray",
                line_dash="dashed",
            )
            p.line(
                x,
                y_sub,
                legend_label="Background subtracted",
                line_width=2,
                color="navy",
            )

        if fit_result is not None:
            p.line(
                x,
                fit_result.best_fit,
                legend_label="Total fit",
                line_width=2,
                color="crimson",
            )
            _colors = [
                "#1f77b4",
                "#ff7f0e",
                "#2ca02c",
                "#9467bd",
                "#8c564b",
                "#e377c2",
                "#7f7f7f",
                "#bcbd22",
                "#17becf",
                "#d62728",
            ]
            for idx, (name, comp) in enumerate(components.items()):
                p.line(
                    x,
                    comp,
                    legend_label=name.rstrip("_"),
                    line_width=2,
                    color=_colors[idx % len(_colors)],
                    line_dash="dotted",
                )

        p.legend.click_policy = "hide"

        # --- Widgets ---
        def _make_input(label, event_name, param_name, current_value, numeric_only=True):
            inp = TextInput(value=str(current_value), title=f"{label} (current: {current_value})")
            hidden = Div(text=str(current_value), visible=False)
            if numeric_only:
                validation_code = """
                    var v = cb_obj.value;
                    if (v === null) return;
                    v = v.trim();
                    if (v === "" || isNaN(Number(v))) return;
                    hidden.text = v;
                """
            else:
                # Comma-separated numerics
                validation_code = """
                    var v = cb_obj.value;
                    if (v === null) return;
                    v = v.trim();
                    if (v === "") return;
                    var parts = v.split(",");
                    for (var i = 0; i < parts.length; i++) {
                        var part = parts[i].trim();
                        if (part === "" || isNaN(Number(part))) return;
                    }
                    hidden.text = v;
                """
            inp.js_on_change(
                "value",
                CustomJS(args=dict(hidden=hidden), code=validation_code),
            )
            hidden.js_on_change(
                "text",
                CustomJS(
                    code=generate_js_callback_single_float_parameter(
                        event_name, param_name, self.block_id, throttled=False
                    )
                ),
            )
            return inp, hidden

        num_peaks_input, num_peaks_hidden = _make_input(
            "Number of peaks", "set_num_peaks", "num_peaks", num_peaks, numeric_only=True
        )
        peak_centers_input, peak_centers_hidden = _make_input(
            "Peak centers (eV)",
            "set_peak_centers",
            "peak_centers",
            peak_centers_str,
            numeric_only=False,
        )

        fit_button = Button(
            label="Fit peaks", button_type="primary", width=150, sizing_mode="fixed"
        )
        fit_button.js_on_click(
            CustomJS(
                code=generate_js_callback_single_float_parameter(
                    "trigger_fit", "trigger", self.block_id
                ).replace("event.target.value", '"1"')
            )
        )

        layout = column(
            num_peaks_input,
            peak_centers_input,
            fit_button,
            p,
            num_peaks_hidden,
            peak_centers_hidden,
            sizing_mode="stretch_width",
        )

        if fit_result is not None:
            self.data["fit_report"] = fit_result.fit_report()

        self.data["bokeh_plot_data"] = bokeh.embed.json_item(layout, theme=DATALAB_BOKEH_THEME)
