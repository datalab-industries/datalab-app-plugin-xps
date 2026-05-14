"""XPS block with Shirley background subtraction and Voigt peak fitting."""

import warnings
from pathlib import Path

import numpy as np
from bokeh.layouts import column
from bokeh.models import Button, CustomJS, Div, TextInput
from bokeh.plotting import figure
from pydatalab.blocks.base import DataBlock, event, generate_js_callback_single_float_parameter
from pydatalab.bokeh_plots import DATALAB_BOKEH_THEME
from vgd_reader import read_vgd

from datalab_app_plugin_xps._version import __version__
from datalab_app_plugin_xps.xps.utils import shirley_background


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
        "peak_centers": "",
        "run_fit": False,
    }

    _prefers_async = True
    multi_file = True

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

    def _make_subfigure(self, filename: Path, plot_title=None):
        background_subtraction = True

        if "XPS_Survey" in filename.name:
            background_subtraction = False

        data = read_vgd(filename)
        x = data.binding_energy
        y = data.corrected_intensity

        # --- Shirley background ---
        bg = None
        if background_subtraction:
            y_sub = y.copy()
            try:
                bg = shirley_background(x, y)
                y_sub = y - bg
            except Exception as exc:
                warnings.warn(f"Shirley background failed: {exc} for {filename}")

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
        num_peaks = len(centers)

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
                warnings.warn(f"Voigt peak fitting failed: {exc} for {filename}")
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

        return (
            fit_result,
            column(
                num_peaks_input,
                peak_centers_input,
                fit_button,
                p,
                num_peaks_hidden,
                peak_centers_hidden,
                sizing_mode="stretch_width",
            ),
        )

    def plot_XPS(self, filename: str | Path | None = None):
        """Creates an XPS plot with Shirley background and Voigt peak fitting.

        Parameters:
            filename: Path to the .VGD file. If None, retrieves from the database
                using the `file_id` stored in `self.data`.
        """

        import bokeh.embed

        if filename:
            file_infos = [{"location": filename, "name": Path(filename).name}]

        else:
            try:
                from pydatalab.file_utils import get_file_info_by_id
            except ImportError:
                raise RuntimeError(
                    "The `datalab-server[server]` extra must be installed to use this block with a database."
                )

            if "file_ids" not in self.data:
                return

            file_ids = self.data.get("file_ids", [])
            if not file_ids and "file_id" in self.data:
                file_ids = [self.data["file_id"]]

            file_infos = []

            for fid in file_ids:
                try:
                    file_infos.append(get_file_info_by_id(fid, update_if_live=True))
                except StopIteration:
                    return

            if not file_infos:
                return

        subfigures = []
        for file in file_infos:
            plot_title = file["name"]
            fit, subfigure = self._make_subfigure(Path(file["location"]), plot_title=plot_title)
            if fit is not None:
                if "fit_report" not in self.data:
                    self.data["fit_report"] = []

                self.data["fit_report"].append(fit.fit_report())
            if subfigure is not None:
                subfigures.append(subfigure)

        if len(subfigures) == 1:
            layout = subfigures[0]
        else:
            layout = column(*subfigures)

        self.data["bokeh_plot_data"] = bokeh.embed.json_item(layout, theme=DATALAB_BOKEH_THEME)
