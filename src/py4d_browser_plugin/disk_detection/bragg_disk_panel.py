import os
import time
from functools import partial

import numpy as np
import pyqtgraph as pg
import py4DSTEM
from PyQt5 import QtCore
from PyQt5.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from py4D_browser.utils import DetectorShape, pg_point_roi

from . import saving

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from py4D_browser import DataViewer
    from .disk_detection import DiskDetectionWindow


def _format_duration(seconds):
    if seconds < 1:
        return f"{seconds * 1000:.2f} ms"
    if seconds < 60:
        return f"{seconds:.1f} s"
    minutes, seconds = divmod(round(seconds), 60)
    if minutes < 60:
        return f"{minutes} min {seconds:02d} s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours} h {minutes:02d} min {seconds:02d} s"


class BraggDiskSettingsPane(QGroupBox):
    def __init__(self):
        super().__init__("Detection Parameters")

        form = QFormLayout()

        self.corr_power_spin = QDoubleSpinBox()
        self.corr_power_spin.setRange(0.0, 1.0)
        self.corr_power_spin.setSingleStep(0.02)
        self.corr_power_spin.setValue(1.0)
        self.corr_power_spin.setToolTip(
            "Correlation power (corrPower): 1 = cross correlation, 0 = phase correlation"
        )
        form.addRow("Corr. Power", self.corr_power_spin)

        self.sigma_spin = QDoubleSpinBox()
        self.sigma_spin.setRange(0.0, 100.0)
        self.sigma_spin.setValue(0.0)
        self.sigma_spin.setToolTip(
            "Gaussian smoothing of the diffraction pattern before correlation (sigma), in pixels"
        )
        form.addRow("Sigma (0 = off)", self.sigma_spin)

        self.sigma_cc_spin = QDoubleSpinBox()
        self.sigma_cc_spin.setRange(0.0, 100.0)
        self.sigma_cc_spin.setValue(0.0)
        self.sigma_cc_spin.setToolTip(
            "Gaussian smoothing of the correlation map (sigma_cc), in pixels"
        )
        form.addRow("Corr. Sigma", self.sigma_cc_spin)

        self.subpixel_combo = QComboBox()
        self.subpixel_combo.addItems(["pixel", "poly", "multicorr"])
        self.subpixel_combo.setCurrentText("poly")
        self.subpixel_combo.setToolTip("Subpixel refinement mode (subpixel)")
        form.addRow("Subpixel", self.subpixel_combo)

        self.upsample_factor_spin = QSpinBox()
        self.upsample_factor_spin.setRange(1, 256)
        self.upsample_factor_spin.setValue(16)
        self.upsample_factor_spin.setToolTip(
            "Upsampling factor for multicorr subpixel refinement (upsample_factor)"
        )
        form.addRow("Upsample", self.upsample_factor_spin)

        self.min_abs_intensity_spin = QDoubleSpinBox()
        self.min_abs_intensity_spin.setRange(0.0, 1e6)
        self.min_abs_intensity_spin.setDecimals(4)
        self.min_abs_intensity_spin.setSingleStep(0.1)
        self.min_abs_intensity_spin.setValue(0.0)
        self.min_abs_intensity_spin.setToolTip(
            "Minimum absolute peak intensity (minAbsoluteIntensity)"
        )
        form.addRow("Min Abs. Int.", self.min_abs_intensity_spin)

        self.min_rel_intensity_spin = QDoubleSpinBox()
        self.min_rel_intensity_spin.setRange(0.0, 1.0)
        self.min_rel_intensity_spin.setDecimals(5)
        self.min_rel_intensity_spin.setSingleStep(0.001)
        self.min_rel_intensity_spin.setValue(0.0)
        self.min_rel_intensity_spin.setToolTip(
            "Minimum peak intensity relative to the reference peak (minRelativeIntensity)"
        )
        form.addRow("Min Rel. Int.", self.min_rel_intensity_spin)

        self.relative_to_peak_spin = QSpinBox()
        self.relative_to_peak_spin.setRange(0, 20)
        self.relative_to_peak_spin.setValue(0)
        self.relative_to_peak_spin.setToolTip(
            "Index of the reference peak for the relative intensity cutoff (relativeToPeak)"
        )
        form.addRow("Rel. to Peak #", self.relative_to_peak_spin)

        self.min_peak_spacing_spin = QSpinBox()
        self.min_peak_spacing_spin.setRange(0, 1000)
        self.min_peak_spacing_spin.setValue(20)
        self.min_peak_spacing_spin.setToolTip(
            "Minimum spacing between detected peaks, in pixels (minPeakSpacing)"
        )
        form.addRow("Min Spacing (px)", self.min_peak_spacing_spin)

        self.edge_boundary_spin = QSpinBox()
        self.edge_boundary_spin.setRange(0, 1000)
        self.edge_boundary_spin.setValue(0)
        self.edge_boundary_spin.setToolTip(
            "Ignore peaks within this many pixels of the pattern edge (edgeBoundary)"
        )
        form.addRow("Edge (px)", self.edge_boundary_spin)

        self.max_num_peaks_spin = QSpinBox()
        self.max_num_peaks_spin.setRange(1, 1000)
        self.max_num_peaks_spin.setValue(50)
        self.max_num_peaks_spin.setToolTip(
            "Maximum number of peaks to return (maxNumPeaks)"
        )
        form.addRow("Max Peaks", self.max_num_peaks_spin)

        self.cuda_checkbox = QCheckBox()
        self.cuda_checkbox.setToolTip("Run detection on the GPU via CUDA (CUDA)")
        form.addRow("Use CUDA", self.cuda_checkbox)

        self.setLayout(form)

        self._all_spinboxes = [
            self.corr_power_spin,
            self.sigma_spin,
            self.sigma_cc_spin,
            self.upsample_factor_spin,
            self.min_abs_intensity_spin,
            self.min_rel_intensity_spin,
            self.relative_to_peak_spin,
            self.min_peak_spacing_spin,
            self.edge_boundary_spin,
            self.max_num_peaks_spin,
        ]

    def connect_changed(self, slot):
        for spin in self._all_spinboxes:
            spin.valueChanged.connect(slot)
        self.subpixel_combo.currentTextChanged.connect(slot)
        self.cuda_checkbox.stateChanged.connect(slot)

    def get_params(self):
        return dict(
            corrPower=self.corr_power_spin.value(),
            sigma=self.sigma_spin.value() or None,
            sigma_cc=self.sigma_cc_spin.value(),
            subpixel=self.subpixel_combo.currentText(),
            upsample_factor=self.upsample_factor_spin.value(),
            minAbsoluteIntensity=self.min_abs_intensity_spin.value(),
            minRelativeIntensity=self.min_rel_intensity_spin.value(),
            relativeToPeak=self.relative_to_peak_spin.value(),
            minPeakSpacing=self.min_peak_spacing_spin.value(),
            edgeBoundary=self.edge_boundary_spin.value(),
            maxNumPeaks=self.max_num_peaks_spin.value(),
            CUDA=self.cuda_checkbox.isChecked(),
        )


class BraggPreviewPane(QGroupBox):
    def __init__(self, title):
        super().__init__(title)

        self.last_dp = None
        self._has_shown_dp = False

        self.rs_view = pg.ImageView()
        self.rs_view.setImage(np.zeros((25, 25)))
        self.rs_selector = pg_point_roi(self.rs_view.getView())

        self.dp_view = pg.ImageView()
        self.dp_view.setImage(np.zeros((512, 512)))

        # Bragg disk positions detected at this pane's scan position, drawn
        # as open rings sized to match the measured bright-field disk
        # diameter (set via set_marker_diameter once a probe is accepted).
        self.scatter = pg.ScatterPlotItem(
            size=12,
            pen=pg.mkPen("g", width=2),
            brush=None,
            symbol="o",
            pxMode=False,  # size is in data (pixel) units, not screen pixels,
            # so markers scale with the image the same way the disks do
        )
        self.dp_view.getView().addItem(self.scatter)

        layout = QVBoxLayout()
        layout.addWidget(self.rs_view)
        layout.addWidget(self.dp_view)
        self.setLayout(layout)

    def set_realspace_image(self, image: Optional[np.ndarray]):
        if image is not None:
            # Transposed to match the main window's real_space_widget
            # convention (see _render_virtual_image in update_views.py) --
            # otherwise non-square scan shapes appear flipped/rotated
            # relative to the main virtual image view, and get_scan_position
            # below (which mirrors the main window's own point-selector
            # reading convention) would read back the wrong (xc, yc).
            # Zoom is only reset when the shape changes, so refreshing on
            # every kernel update doesn't undo the user's panning/zooming.
            self.rs_view.setImage(
                image.T,
                autoLevels=True,
                autoRange=self.rs_view.image.shape != image.T.shape,
            )

    def set_colormaps(self, realspace_cmap, diffraction_cmap):
        self.rs_view.setColorMap(realspace_cmap)
        self.dp_view.setColorMap(diffraction_cmap)

    def set_marker_diameter(self, diameter):
        self.scatter.setSize(diameter)

    def get_scan_position(self, datacube):
        roi_state = self.rs_selector.saveState()
        y0, x0 = roi_state["pos"]
        xc, yc = int(x0 + 1), int(y0 + 1)
        xc = int(np.clip(xc, 0, datacube.R_Nx - 1))
        yc = int(np.clip(yc, 0, datacube.R_Ny - 1))
        return xc, yc

    def set_scan_position(self, xc, yc):
        # The inverse of get_scan_position. Signals are blocked, so the
        # caller is responsible for updating the pane.
        self.rs_selector.blockSignals(True)
        self.rs_selector.setPos((yc - 0.5, xc - 0.5))
        self.rs_selector.blockSignals(False)

    def update_dp(self, dp, scale_fn, relevel):
        self.last_dp = dp
        # Transposed to match the main window's diffraction_space_widget
        # convention (see _render_diffraction_image in update_views.py).
        scaled = scale_fn(dp).T

        if relevel or not self._has_shown_dp:
            levels = tuple(np.percentile(scaled, [0.1, 99.9]))
            self.dp_view.setImage(
                scaled,
                autoLevels=False,
                levels=levels,
                autoRange=not self._has_shown_dp,
            )
            self._has_shown_dp = True
        else:
            # Leave levels untouched -- pyqtgraph does not reset them when
            # autoLevels=False and no explicit levels are given, so any
            # level range the user dragged in by hand on the histogram
            # widget survives a scan-position-only update.
            self.dp_view.setImage(scaled, autoLevels=False, autoRange=False)

    def update_scatter(self, qx, qy):
        # dp is displayed transposed (see update_dp), so marker positions
        # need the same (y, x) swap to stay aligned with the disks --
        # verified empirically with an offscreen render against a synthetic
        # test image.
        spots = [{"pos": [y, x], "data": 1} for x, y in zip(qx, qy)]
        self.scatter.setData(spots)


class BraggDiskPanel(QWidget):
    """
    Bragg disk detection controls. The panel itself is the controls column;
    its preview panes live in `self.views`, for the window to lay out
    alongside the probe/kernel views.
    """

    # Start with a single preview pane; this caps how many more the user
    # can add via the "Add Preview Position" button.
    MAX_ADDITIONAL_PANES = 3

    def __init__(self, window: "DiskDetectionWindow"):
        super().__init__()

        self.window = window
        self.panes = []

        self.settings_pane = BraggDiskSettingsPane()
        self.settings_pane.connect_changed(self._on_detection_params_changed)

        # Previews use the same colormaps as the main window's views, and
        # follow them if they're changed there (via the gradient's
        # right-click menu).
        parent = self.window.parent
        for widget in (parent.real_space_widget, parent.diffraction_space_widget):
            widget.ui.histogram.gradient.sigGradientChangeFinished.connect(
                self.sync_colormaps
            )

        scaling_box = self._build_scaling_box()

        self.find_all_button = QPushButton("Find All Bragg Disks")
        self.find_all_button.clicked.connect(self.find_all)

        self.find_all_progress = QProgressBar()
        self.find_all_progress.setVisible(False)

        self.add_pane_button = QPushButton("Add Preview Position")
        self.add_pane_button.clicked.connect(self.add_pane)

        # For applying detection parameter changes when live update is off.
        self.update_previews_button = QPushButton("Update Previews")
        self.update_previews_button.clicked.connect(lambda: self.update_previews())

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.settings_pane)
        layout.addWidget(scaling_box)
        layout.addWidget(self.update_previews_button)
        layout.addWidget(self.add_pane_button)
        layout.addWidget(self.find_all_button)
        layout.addWidget(self.find_all_progress)
        self.setLayout(layout)

        self.previews_layout = QHBoxLayout()
        self.views = QWidget()
        self.views.setLayout(self.previews_layout)

        self._append_pane()

    def _build_scaling_box(self):
        scaling_box = QGroupBox("Display Scaling")
        scaling_layout = QVBoxLayout()

        self.scaling_group = QButtonGroup(self)
        self.linear_radio = QRadioButton("Linear")
        self.log_radio = QRadioButton("Log")
        self.power_radio = QRadioButton("Power")
        self.power_radio.setChecked(True)
        for button in (self.linear_radio, self.log_radio, self.power_radio):
            self.scaling_group.addButton(button)
            scaling_layout.addWidget(button)
            # only redraw once, when a button becomes checked -- QButtonGroup
            # toggles the old and new selection in the same click
            button.toggled.connect(
                lambda checked: self.on_scaling_changed() if checked else None
            )

        gamma_row = QHBoxLayout()
        gamma_row.addWidget(QLabel("Power"))
        self.gamma_spin = QDoubleSpinBox()
        self.gamma_spin.setRange(0.01, 2.0)
        self.gamma_spin.setSingleStep(0.05)
        self.gamma_spin.setValue(0.5)
        self.gamma_spin.valueChanged.connect(self.on_scaling_changed)
        gamma_row.addWidget(self.gamma_spin)
        scaling_layout.addLayout(gamma_row)

        scaling_box.setLayout(scaling_layout)
        return scaling_box

    def get_scaling_fn(self):
        gamma = self.gamma_spin.value()
        if self.linear_radio.isChecked():
            return lambda dp: dp.astype(np.float64, copy=False)
        elif self.log_radio.isChecked():
            return lambda dp: np.log(np.maximum(dp, 1e-6))
        else:
            return lambda dp: np.power(np.maximum(dp, 0), gamma)

    def _append_pane(self):
        if len(self.panes) - 1 >= self.MAX_ADDITIONAL_PANES:
            return

        pane = BraggPreviewPane(f"Preview {len(self.panes) + 1}")
        # Always update once the drag is released. Also update continuously
        # during the drag (sigRegionChanged), but only if the user has
        # opted into that via the "Live Update" checkbox.
        pane.rs_selector.sigRegionChangeFinished.connect(
            partial(self.update_previews, panes=[pane])
        )
        pane.rs_selector.sigRegionChanged.connect(
            partial(self._on_pane_region_changed, pane)
        )
        self.panes.append(pane)
        self.previews_layout.addWidget(pane)
        self.window.disable_auto_default(pane)
        self.sync_colormaps(panes=[pane])

        if self.window.probe is not None:
            pane.set_marker_diameter(2 * self.window.probe_radius)
        self.show_main_view(panes=[pane])

        if len(self.panes) - 1 >= self.MAX_ADDITIONAL_PANES:
            self.add_pane_button.setEnabled(False)

    def add_pane(self):
        self._append_pane()

    def _on_pane_region_changed(self, pane, *_):
        if self.window.is_live():
            self.update_previews(panes=[pane])

    def _on_detection_params_changed(self, *_):
        if self.window.is_live():
            self.on_detection_params_changed()

    def on_kernel_updated(self):
        parent = self.window.parent
        vimg = parent.unscaled_realspace_image
        diameter = 2 * self.window.probe_radius
        for pane in self.panes:
            pane.set_realspace_image(vimg)
            pane.set_marker_diameter(diameter)
        self.update_previews()

    def _find_peaks(self, dp, xc, yc, probe, params):
        parent = self.window.parent
        try:
            peaks = parent.datacube.find_Bragg_disks(
                template=probe.kernel, data=(xc, yc), **params
            )
            return peaks.qx, peaks.qy
        except Exception as exc:
            parent.statusBar().showMessage(f"Peak finding failed: {exc}", 5_000)
            return np.array([]), np.array([])

    def sync_colormaps(self, *_, panes=None):
        parent = self.window.parent
        realspace_cmap = parent.real_space_widget.ui.histogram.gradient.colorMap()
        diffraction_cmap = (
            parent.diffraction_space_widget.ui.histogram.gradient.colorMap()
        )
        for pane in panes if panes is not None else self.panes:
            pane.set_colormaps(realspace_cmap, diffraction_cmap)

    def _main_scan_position(self, parent):
        # Where the main window's real-space cursor is -- or, for a
        # rectangular selection, its centre.
        detector = parent.get_virtual_image_detector()
        if detector["shape"] is DetectorShape.POINT:
            xc, yc = detector["point"]
        else:
            slice_x, slice_y = detector["slice"]
            xc = (slice_x.start + slice_x.stop - 1) // 2
            yc = (slice_y.start + slice_y.stop - 1) // 2
        return (
            int(np.clip(xc, 0, parent.datacube.R_Nx - 1)),
            int(np.clip(yc, 0, parent.datacube.R_Ny - 1)),
        )

    def show_main_view(self, panes=None):
        """
        Mirror the main window's current view in the given (default: all)
        panes: its virtual image, with the pane's cursor at the main
        window's scan position and that position's diffraction pattern.
        """
        parent = self.window.parent
        if parent.datacube is None:
            return

        panes = panes if panes is not None else self.panes
        xc, yc = self._main_scan_position(parent)
        for pane in panes:
            pane.set_realspace_image(parent.unscaled_realspace_image)
            pane.set_scan_position(xc, yc)
        self.update_previews(panes=panes)

    def update_previews(self, panes=None, relevel=False):
        # Called when a pane's point selector moves (position + peaks +
        # display all need refreshing for that pane). Until there's a
        # kernel, just shows the diffraction pattern.
        parent = self.window.parent
        probe = self.window.probe
        if parent.datacube is None:
            return

        panes = panes if panes is not None else self.panes
        params = self.settings_pane.get_params()
        scale_fn = self.get_scaling_fn()

        for pane in panes:
            xc, yc = pane.get_scan_position(parent.datacube)
            dp = parent.datacube.data[xc, yc, :, :]
            pane.update_dp(dp, scale_fn, relevel=relevel)
            if probe is not None:
                qx, qy = self._find_peaks(dp, xc, yc, probe, params)
                pane.update_scatter(qx, qy)

    def on_detection_params_changed(self, *_):
        # Detection-only parameters (thresholds, corrPower, etc.) don't
        # change the diffraction pattern or its display scaling, only which
        # peaks are found -- so just rerun peak-finding against each pane's
        # cached DP, without touching the displayed image or its levels.
        parent = self.window.parent
        probe = self.window.probe
        if probe is None or parent.datacube is None:
            return

        params = self.settings_pane.get_params()
        for pane in self.panes:
            if pane.last_dp is None:
                continue
            xc, yc = pane.get_scan_position(parent.datacube)
            qx, qy = self._find_peaks(pane.last_dp, xc, yc, probe, params)
            pane.update_scatter(qx, qy)

    def on_scaling_changed(self, *_):
        # Scaling mode/gamma changed -- redisplay each pane's cached DP with
        # freshly computed percentile levels for the new scaling function.
        scale_fn = self.get_scaling_fn()
        for pane in self.panes:
            if pane.last_dp is not None:
                pane.update_dp(pane.last_dp, scale_fn, relevel=True)

    def find_all(self):
        parent = self.window.parent
        probe = self.window.probe
        if probe is None or parent.datacube is None:
            parent.statusBar().showMessage("Generate a probe kernel first!", 5_000)
            return

        self.window.finding_all = True
        try:
            self._find_all(parent, probe)
        finally:
            self.window.finding_all = False
            self.window.sig_find_all_finished.emit()

    def _find_all(self, parent, probe):
        params = self.settings_pane.get_params()
        datacube = parent.datacube
        R_Nx, R_Ny = datacube.R_Nx, datacube.R_Ny

        # find_Bragg_disks(data=None) runs the whole dataset internally with
        # no way to hook a Qt progress bar into its console-only tqdm loop,
        # so loop over scan positions ourselves (via the same sanctioned
        # single-position call the preview panes already use) and assemble
        # the results into a BraggVectors object by hand, using only public
        # py4DSTEM API (PointListArray / set_raw_vectors).
        self.find_all_button.setEnabled(False)
        self.find_all_progress.setMinimum(0)
        self.find_all_progress.setMaximum(R_Nx * R_Ny)
        self.find_all_progress.setValue(0)
        self.find_all_progress.setVisible(True)
        parent.statusBar().showMessage("Running disk detection on the full dataset...")

        pla = None
        count = 0
        start_time = time.perf_counter()
        try:
            for rx in range(R_Nx):
                for ry in range(R_Ny):
                    peaks = datacube.find_Bragg_disks(
                        template=probe.kernel, data=(rx, ry), **params
                    )
                    if pla is None:
                        pla = py4DSTEM.PointListArray(
                            dtype=peaks.data.dtype, shape=datacube.Rshape
                        )
                    pla.get_pointlist(rx, ry).add(peaks.data)

                    count += 1
                    self.find_all_progress.setValue(count)
                    parent.qtapp.processEvents()

                    # A new dataset may have been loaded while events were
                    # processed -- these results would belong to neither.
                    if parent.datacube is not datacube:
                        self.find_all_progress.setVisible(False)
                        self.find_all_button.setEnabled(True)
                        parent.statusBar().showMessage(
                            "Find All Bragg Disks stopped: a new dataset was "
                            "loaded.",
                            10_000,
                        )
                        return

            braggvectors = py4DSTEM.BraggVectors(datacube.Rshape, datacube.Qshape)
            braggvectors.set_raw_vectors(pla)
            datacube.attach(braggvectors)

            bvm = braggvectors.get_bvm(mode="raw")
            bvm.name = "bvm"
            datacube.attach(bvm)
            elapsed = time.perf_counter() - start_time
        except Exception as exc:
            parent.statusBar().showMessage(f"Disk detection failed: {exc}", 5_000)
            self.find_all_progress.setVisible(False)
            self.find_all_button.setEnabled(True)
            raise

        self.find_all_progress.setVisible(False)
        self.find_all_button.setEnabled(True)

        saved = self._save_results(parent, braggvectors, params)
        self._show_summary(datacube, elapsed, saved)

    def _show_summary(self, datacube, elapsed, saved):
        n_patterns = datacube.R_Nx * datacube.R_Ny
        lines = [
            f"Patterns: {n_patterns:,} ({datacube.R_Nx} × {datacube.R_Ny} scan)",
            f"Total time: {_format_duration(elapsed)}",
            f"Average per pattern: {_format_duration(elapsed / n_patterns)}",
            "",
        ]
        if saved is not None:
            filepath, datapath, params_filepath = saved
            lines += [
                "Bragg vectors saved to:",
                filepath,
                f"at {datapath}",
                "",
                "Parameters saved to:",
                params_filepath,
            ]
        else:
            lines.append(
                "The results were not saved; they remain attached to the "
                "dataset in memory."
            )

        box = QMessageBox(self)
        box.setIcon(QMessageBox.Information)
        box.setWindowTitle("Disk Detection Complete")
        box.setText("Bragg disk detection completed successfully.")
        box.setInformativeText("\n".join(lines))
        # So the paths can be copied.
        for label in box.findChildren(QLabel):
            label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        box.exec_()

    def _save_results(self, parent, braggvectors, detect_params):
        """
        Save the Bragg vectors into the file the raw data came from when
        py4DSTEM can append to it, otherwise export the whole dataset (with
        the results attached) to a new file -- then write the parameters
        used next to whichever file that was. Returns (filepath, datapath,
        params_filepath), or None if nothing was saved.
        """
        source = saving.find_source_emd_group(parent)
        saved = None
        if source is not None:
            saved = self._append_to_source_file(parent, braggvectors, *source)
            if saved is False:
                return None
        if saved is None:
            saved = self._save_datacube_as(parent)
            if saved is None:
                return None
        filepath, datapath = saved

        probe_kernel_panel = self.window.probe_kernel_panel
        params_filepath = saving.write_params_file(
            filepath,
            data=dict(
                filepath=saving.loaded_filepath(parent),
                datapath=source[1] if source is not None else None,
                shape=parent.datacube.data.shape,
            ),
            probe_params=probe_kernel_panel.probe_params,
            kernel_params=probe_kernel_panel.kernel_params,
            detect_params=detect_params,
            braggvectors_datapath=datapath,
        )

        parent.statusBar().showMessage(
            f"Saved Bragg vectors to {filepath} (at {datapath}), and the "
            f"parameters used to {params_filepath}",
            15_000,
        )
        return filepath, datapath, params_filepath

    def _append_to_source_file(self, parent, braggvectors, filepath, group):
        """
        Returns (filepath, datapath) once saved, False if the user declined
        to replace existing results, or None if the file couldn't be
        written to (so the caller can fall back to exporting a new file).
        """
        datapath = f"{group}/{braggvectors.name}"
        replace = saving.has_node(filepath, datapath)
        if replace:
            answer = QMessageBox.question(
                self,
                "Replace Bragg Vectors?",
                f"{os.path.basename(filepath)} already contains Bragg vectors "
                f"at {datapath}.\n\nReplace them with the new results?",
            )
            if answer != QMessageBox.Yes:
                parent.statusBar().showMessage(
                    "Not saved. Results remain attached to the dataset in " "memory.",
                    10_000,
                )
                return False

        parent.statusBar().showMessage(f"Saving Bragg vectors to {filepath}...")
        try:
            saving.append_braggvectors(braggvectors, filepath, group, replace)
        except Exception as exc:
            QMessageBox.warning(
                self,
                "Couldn't Save Into Source File",
                f"Couldn't save the Bragg vectors into {filepath}:\n\n{exc}"
                "\n\nChoose a new file to export the dataset and results to "
                "instead.",
            )
            return None
        return filepath, datapath

    def _save_datacube_as(self, parent):
        # For data that didn't come from a py4DSTEM/EMD file, which
        # py4DSTEM.save can't append to. Reuses the main window's own
        # save-file dialog and the same py4DSTEM.save call as File > Export
        # Datacube > py4DSTEM HDF5 -- since braggvectors/bvm are attached to
        # the datacube's tree, they're written out along with it.
        parent.statusBar().showMessage(
            "Disk detection complete. Choose where to save the results...", 5_000
        )
        try:
            filename = parent.get_savefile_name("py4DSTEM HDF5")
        except ValueError:
            parent.statusBar().showMessage(
                "Save cancelled. Results remain attached to the dataset in "
                "memory (export later via File > Export Datacube > "
                "py4DSTEM HDF5).",
                10_000,
            )
            return None

        try:
            py4DSTEM.save(filename, parent.datacube, mode="o")
        except Exception as exc:
            parent.statusBar().showMessage(f"Saving failed: {exc}", 5_000)
            raise

        return filename, saving.find_braggvectors_datapath(filename)
