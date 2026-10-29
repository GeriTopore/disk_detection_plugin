import os

import h5py
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
    QPushButton,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from py4D_browser.dialogs import ResizeDialog
from py4D_browser.menu_actions import find_calibrations, get_ND
from py4D_browser.utils import DetectorInfo, DetectorShape

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from py4D_browser import DataViewer
    from .disk_detection import DiskDetectionWindow


def _vacuum_probe_from_single_pixel(datacube, xc, yc, threshold, expansion, opening):
    """
    Work around a py4DSTEM bug (as of 0.14.18): DataCube.get_vacuum_probe's
    internal averaging loop is skipped entirely when the ROI contains
    exactly one scan position, leaving its `probe` variable as a raw view
    into datacube.data (sometimes read-only, and the wrong dtype for the
    in-place `probe *= mask` that follows) -- both fail. Replicate the same
    thresholding/expansion/opening masking on a writable float copy
    instead, since averaging over a single pattern is just that pattern.
    """
    from scipy.ndimage import binary_dilation, binary_opening, distance_transform_edt

    probe = datacube.data[xc, yc, :, :].astype(np.float64)
    mask = probe > np.max(probe) * threshold
    mask = binary_opening(mask, iterations=opening)
    mask = binary_dilation(mask, iterations=1)
    mask = (
        np.cos(
            (np.pi / 2)
            * np.minimum(distance_transform_edt(np.logical_not(mask)) / expansion, 1)
        )
        ** 2
    )
    probe = py4DSTEM.Probe(probe * mask)
    datacube.attach(probe)
    return probe


def _load_datacube_from_file(
    filepath: str, dialog_parent: QWidget
) -> py4DSTEM.DataCube:
    """
    Load a standalone DataCube from a file on disk, using the same
    format-detection logic as py4D_browser.menu_actions.load_file.
    """
    extension = os.path.splitext(filepath)[-1].lower()

    if extension in (".h5", ".hdf5", ".py4dstem", ".emd", ".mat"):
        file = h5py.File(filepath, "r")
        datacubes = get_ND(file)
        if len(datacubes) >= 1:
            parent_group = "/".join(datacubes[0].name.split("/")[:-1])
            if len(parent_group) > 1 and "emd_group_type" in file[parent_group].attrs:
                return py4DSTEM.DataCube.from_h5(datacubes[0].file[parent_group])

            datacube = py4DSTEM.DataCube(datacubes[0][()])
            R_size, R_units, Q_size, Q_units = find_calibrations(datacubes[0])
            datacube.calibration.set_R_pixel_size(R_size)
            datacube.calibration.set_R_pixel_units(R_units)
            datacube.calibration.set_Q_pixel_size(Q_size)
            datacube.calibration.set_Q_pixel_units(Q_units)
            return datacube

        datacubes = get_ND(file, N=3)
        if len(datacubes) >= 1:
            array = datacubes[0][()]
            new_shape = ResizeDialog.get_new_size(
                [1, array.shape[0]], parent=dialog_parent
            )
            return py4DSTEM.DataCube(array.reshape(*new_shape, *array.shape[1:]))

        raise ValueError("No 4D (or even 3D) data detected in the H5 file!")

    elif extension == ".npy":
        return py4DSTEM.DataCube(np.load(filepath))

    else:
        return py4DSTEM.import_file(filepath, mem="RAM")


class ProbeKernelPanel(QWidget):
    """
    Probe source/mask/kernel controls. The panel itself is the controls
    column; its probe and kernel image views live in `self.views`, for the
    window to lay out alongside the Bragg disk previews.
    """

    def __init__(self, window: "DiskDetectionWindow"):
        super().__init__()

        self.window = window

        self.probe = None
        self.alpha = None
        self.qx0 = None
        self.qy0 = None
        self._source = None
        self._vacuum_datacube = None
        self._vacuum_filepath = None
        self._measured_radius = None  # most recently measured BF disk radius

        # Exactly what the current probe/kernel were generated with, as
        # py4DSTEM keyword arguments -- saved alongside the detection results
        # so the same probe/kernel can be rebuilt outside the GUI.
        self.probe_params = None
        self.kernel_params = None

        # ---- vacuum region source ----
        source_box = QGroupBox("Probe Source")
        source_layout = QVBoxLayout()

        button_row = QHBoxLayout()
        use_selection_button = QPushButton("From selection")
        use_selection_button.setToolTip(
            "Measure the probe from the current point or rectangular selection "
            "on the main window's virtual image"
        )
        use_selection_button.clicked.connect(self.use_current_selection)
        button_row.addWidget(use_selection_button)

        load_vacuum_button = QPushButton("Vacuum File...")
        load_vacuum_button.setToolTip("Measure the probe from a separate vacuum file")
        load_vacuum_button.clicked.connect(self.load_vacuum_file)
        button_row.addWidget(load_vacuum_button)

        use_synthetic_button = QPushButton("Synthetic")
        use_synthetic_button.setToolTip("Generate a synthetic probe")
        use_synthetic_button.clicked.connect(self.use_synthetic_probe)
        button_row.addWidget(use_synthetic_button)
        source_layout.addLayout(button_row)

        # Checkable, so the source in use stays pressed in (and none is
        # until one has been chosen).
        self._source_buttons = {
            "selection": use_selection_button,
            "vacuum_file": load_vacuum_button,
            "synthetic": use_synthetic_button,
        }
        self._source_group = QButtonGroup(self)
        for button in self._source_buttons.values():
            button.setCheckable(True)
            self._source_group.addButton(button)

        self.source_label = QLabel("No vacuum region selected yet")
        self.source_label.setWordWrap(True)
        source_layout.addWidget(self.source_label)

        # Only used when the source is "Synthetic", and greyed out otherwise.
        synth_form = QFormLayout()
        synth_form.setContentsMargins(0, 0, 0, 0)
        self.synth_radius_spin = QDoubleSpinBox()
        self.synth_radius_spin.setRange(0.5, 1000.0)
        self.synth_radius_spin.setValue(10.0)
        self.synth_radius_spin.setToolTip("Synthetic probe radius, in pixels")
        synth_form.addRow("Radius (px)", self.synth_radius_spin)

        # When checked, the radius follows (and is greyed out to) the
        # bright-field disk radius measured from the current selection;
        # otherwise the user's own value is used as-is.
        self.use_measured_radius_checkbox = QCheckBox("Use measured radius")
        self.use_measured_radius_checkbox.setChecked(True)
        self.use_measured_radius_checkbox.toggled.connect(
            self.on_use_measured_radius_toggled
        )
        synth_form.addRow(self.use_measured_radius_checkbox)

        self.synth_width_spin = QDoubleSpinBox()
        self.synth_width_spin.setRange(0.1, 100.0)
        self.synth_width_spin.setValue(4.0)
        self.synth_width_spin.setToolTip("Synthetic probe edge width, in pixels")
        synth_form.addRow("Edge Width (px)", self.synth_width_spin)

        for spin in (self.synth_radius_spin, self.synth_width_spin):
            spin.valueChanged.connect(self.on_probe_params_changed)

        self.synth_params = QWidget()
        self.synth_params.setLayout(synth_form)
        self.synth_params.setEnabled(False)
        self.synth_radius_spin.setEnabled(False)

        source_layout.addWidget(self.synth_params)
        source_box.setLayout(source_layout)

        # ---- mask settings + probe generation ----
        mask_box = QGroupBox("Mask Settings")
        mask_form = QFormLayout()

        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0.0, 1.0)
        self.threshold_spin.setSingleStep(0.01)
        self.threshold_spin.setDecimals(2)
        self.threshold_spin.setValue(0.0)
        self.threshold_spin.setToolTip(
            "Pixels above this fraction of the probe's maximum intensity count\n"
            "as probe; the rest is masked out as background. 0 keeps nearly\n"
            "everything; 0.1-0.3 is typical for noisy data."
        )
        mask_form.addRow("Threshold", self.threshold_spin)

        self.expansion_spin = QSpinBox()
        # py4DSTEM divides by this, so 0 would make the probe NaN.
        self.expansion_spin.setRange(1, 500)
        self.expansion_spin.setValue(12)
        self.expansion_spin.setToolTip(
            "Width in pixels of the smooth fall-off applied outside the mask\n"
            "edge, instead of a hard cut. Larger = softer, wider edge."
        )
        mask_form.addRow("Expansion", self.expansion_spin)

        self.opening_spin = QSpinBox()
        self.opening_spin.setRange(0, 500)
        self.opening_spin.setValue(3)
        self.opening_spin.setToolTip(
            "Binary opening iterations: removes specks and thin features\n"
            "from the mask, leaving the solid bright-field disk. Too high\n"
            "starts to erode the disk edge."
        )
        mask_form.addRow("Opening", self.opening_spin)

        for spin in (self.threshold_spin, self.expansion_spin, self.opening_spin):
            spin.valueChanged.connect(self.on_probe_params_changed)

        mask_box.setLayout(mask_form)

        self.probe_view = pg.ImageView()
        self.probe_view.setImage(np.zeros((256, 256)))

        # ---- kernel settings ----
        kernel_box = QGroupBox("Kernel Settings")
        kernel_form = QFormLayout()

        self.kernel_mode_combo = QComboBox()
        self.kernel_mode_combo.addItems(["flat", "gaussian", "sigmoid", "sigmoid_log"])
        self.kernel_mode_combo.currentTextChanged.connect(
            self.update_kernel_controls_enabled
        )
        self.kernel_mode_combo.setToolTip("Kernel mode")
        kernel_form.addRow("Mode", self.kernel_mode_combo)

        self.sigma_spin = QDoubleSpinBox()
        self.sigma_spin.setRange(0.0, 100.0)
        self.sigma_spin.setValue(4.0)
        self.sigma_spin.setToolTip("Gaussian kernel sigma, in pixels")
        kernel_form.addRow("Sigma", self.sigma_spin)

        self.r_inner_spin = QDoubleSpinBox()
        self.r_inner_spin.setRange(0.0, 10.0)
        self.r_inner_spin.setSingleStep(0.1)
        self.r_inner_spin.setValue(1.0)
        self.r_inner_spin.setToolTip(
            "Inner sigmoid radius, as a multiple of the probe radius"
        )
        kernel_form.addRow("Inner Radius (×r)", self.r_inner_spin)

        self.r_outer_spin = QDoubleSpinBox()
        self.r_outer_spin.setRange(0.0, 10.0)
        self.r_outer_spin.setSingleStep(0.1)
        self.r_outer_spin.setValue(4.0)
        self.r_outer_spin.setToolTip(
            "Outer sigmoid radius, as a multiple of the probe radius"
        )
        kernel_form.addRow("Outer Radius (×r)", self.r_outer_spin)

        self.kernel_mode_combo.currentTextChanged.connect(self.on_kernel_params_changed)
        for spin in (self.sigma_spin, self.r_inner_spin, self.r_outer_spin):
            spin.valueChanged.connect(self.on_kernel_params_changed)

        kernel_box.setLayout(kernel_form)

        # Generating the probe generates its kernel too (see generate_probe).
        generate_button = QPushButton("Generate Probe && Kernel")
        generate_button.clicked.connect(self.generate_probe)

        self.kernel_view = pg.ImageView()
        self.kernel_view.setImage(np.zeros((256, 256)))

        # Line profiles through the kernel center, along x and along y,
        # so the falloff shape (e.g. gaussian/sigmoid rolloff) can be judged.
        self.kernel_profile_plot = pg.PlotWidget()
        self.kernel_profile_plot.addLegend()
        self.kernel_profile_plot.setMaximumHeight(150)
        self._kernel_x_curve = self.kernel_profile_plot.plot(
            pen=pg.mkPen("y", width=2), name="x profile"
        )
        self._kernel_y_curve = self.kernel_profile_plot.plot(
            pen=pg.mkPen("c", width=2), name="y profile"
        )

        # ---- overall layout ----
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(source_box)
        layout.addWidget(mask_box)
        layout.addWidget(kernel_box)
        layout.addWidget(generate_button)
        self.setLayout(layout)

        # Splitters (rather than plain layouts) so the panes can be resized
        # by dragging their borders, same as the main window's own
        # real-space/FFT split.
        kernel_splitter = QSplitter(QtCore.Qt.Vertical)
        kernel_splitter.addWidget(self.kernel_view)
        kernel_splitter.addWidget(self.kernel_profile_plot)
        kernel_splitter.setSizes([500, 150])

        self.views = QSplitter(QtCore.Qt.Horizontal)
        self.views.addWidget(self.probe_view)
        self.views.addWidget(kernel_splitter)

        self.update_kernel_controls_enabled(self.kernel_mode_combo.currentText())

    def use_current_selection(self):
        self._vacuum_datacube = None
        self._set_source("selection", "Source: selection on current dataset")

    def load_vacuum_file(self):
        parent = self.window.parent
        try:
            filepath = parent.show_file_dialog()
        except ValueError:
            self._sync_source_buttons()
            return

        try:
            datacube = _load_datacube_from_file(filepath, dialog_parent=self)
        except Exception as exc:
            parent.statusBar().showMessage(f"Couldn't load vacuum file: {exc}", 5_000)
            self._sync_source_buttons()
            return

        self._vacuum_datacube = datacube
        self._vacuum_filepath = os.path.abspath(filepath)
        self._set_source("vacuum_file", f"Source: {os.path.basename(filepath)}")

    def use_synthetic_probe(self):
        self._vacuum_datacube = None
        self._set_source("synthetic", "Source: synthetic probe")

    def _set_source(self, source, label_text):
        self._source = source
        self.source_label.setText(label_text)
        self.synth_params.setEnabled(source == "synthetic")
        self._sync_source_buttons()
        self.on_probe_params_changed()

    def _sync_source_buttons(self):
        # Clicking a button checks it straight away, so put the check back
        # on the source actually in use when that didn't change (e.g. a
        # cancelled vacuum file load). Exclusivity is lifted meanwhile, as
        # an exclusive group won't uncheck its last checked button.
        self._source_group.setExclusive(False)
        for source, button in self._source_buttons.items():
            button.setChecked(source == self._source)
        self._source_group.setExclusive(True)

    def on_probe_params_changed(self, *_):
        # A new probe means a new kernel (and so new detections), which
        # generate_probe carries through on its own.
        if self.window.is_live() and self._source is not None:
            self.generate_probe()

    def on_kernel_params_changed(self, *_):
        if self.window.is_live() and self.probe is not None:
            self.generate_kernel()

    def on_use_measured_radius_toggled(self, checked):
        self.synth_radius_spin.setEnabled(not checked)
        if checked and self._measured_radius is not None:
            self._set_synth_radius(self._measured_radius)
        self.on_probe_params_changed()

    def _record_measured_radius(self, radius):
        self._measured_radius = radius
        if self.use_measured_radius_checkbox.isChecked():
            self._set_synth_radius(radius)

    def _set_synth_radius(self, radius):
        # Programmatic sync to the measured radius -- blocked so it doesn't
        # count as a user edit and re-trigger a live probe regeneration.
        self.synth_radius_spin.blockSignals(True)
        self.synth_radius_spin.setValue(radius)
        self.synth_radius_spin.blockSignals(False)

    def _selection_roi(self, parent) -> Optional[tuple]:
        """
        Get the (rx_min, rx_max, ry_min, ry_max) bounds of whatever is
        currently selected on the main window's virtual image -- a
        rectangular region, or a single scan position (point selector).
        """
        detector: DetectorInfo = parent.get_virtual_image_detector()

        if detector["shape"] is DetectorShape.RECTANGULAR:
            slice_x, slice_y = detector["slice"]
            return (
                int(slice_x.start),
                int(slice_x.stop),
                int(slice_y.start),
                int(slice_y.stop),
            )

        if detector["shape"] is DetectorShape.POINT:
            xc, yc = (int(i) for i in detector["point"])
            return (xc, xc + 1, yc, yc + 1)

        parent.statusBar().showMessage(
            "Select a point or rectangular region on the virtual image to "
            "use as the vacuum region.",
            5_000,
        )
        return None

    def _mask_params(self):
        return dict(
            threshold=self.threshold_spin.value(),
            expansion=self.expansion_spin.value(),
            opening=self.opening_spin.value(),
        )

    def _measure_probe_from_current_selection(self, parent):
        """
        Measure a (probe, alpha, qx0, qy0, probe_params) from whatever is
        currently selected on the main window (point or rectangular), or
        return None (with a status bar message already shown) if there's no
        usable selection.
        """
        roi = self._selection_roi(parent)
        if roi is None:
            return None
        rx_min, rx_max, ry_min, ry_max = roi
        mask_params = self._mask_params()

        if (rx_max - rx_min) * (ry_max - ry_min) == 1:
            probe = _vacuum_probe_from_single_pixel(
                parent.datacube, rx_min, ry_min, **mask_params
            )
        else:
            # Passed as a boolean mask rather than as the bounds tuple
            # get_vacuum_probe also accepts: as of py4DSTEM 0.14.18 a tuple
            # ROI is read relative to its own corner, i.e. always averages
            # the patterns at the top-left of the scan.
            mask = np.zeros(parent.datacube.Rshape, dtype=np.bool_)
            mask[rx_min:rx_max, ry_min:ry_max] = True
            probe = parent.datacube.get_vacuum_probe(ROI=mask, **mask_params)

        alpha, qx0, qy0 = parent.datacube.get_probe_size(probe.probe)
        probe_params = dict(source="selection", ROI=roi, get_vacuum_probe=mask_params)
        return probe, alpha, qx0, qy0, probe_params

    def generate_probe(self):
        parent = self.window.parent
        if parent.datacube is None:
            parent.statusBar().showMessage(
                "Load a dataset in the main window first!", 5_000
            )
            return

        if self._source == "selection":
            result = self._measure_probe_from_current_selection(parent)
            if result is None:
                return
            self.probe, self.alpha, self.qx0, self.qy0, self.probe_params = result
            self._record_measured_radius(self.alpha)

        elif self._source == "vacuum_file":
            if self._vacuum_datacube is None:
                parent.statusBar().showMessage("Load a vacuum file first!", 5_000)
                return
            mask_params = self._mask_params()
            self.probe = self._vacuum_datacube.get_vacuum_probe(**mask_params)
            self.alpha, self.qx0, self.qy0 = parent.datacube.get_probe_size(
                self.probe.probe
            )
            self._record_measured_radius(self.alpha)
            self.probe_params = dict(
                source="vacuum_file",
                filepath=self._vacuum_filepath,
                get_vacuum_probe=mask_params,
            )

        elif self._source == "synthetic":
            # With "Use measured radius" checked, re-measure the BF disk
            # radius from the current selection every time, so the synthetic
            # probe reflects it exactly rather than a possibly-stale value
            # (falling back to the last measured radius if there's no usable
            # selection to measure from). Otherwise use the user's radius.
            if self.use_measured_radius_checkbox.isChecked():
                result = self._measure_probe_from_current_selection(parent)
                if result is not None:
                    _, measured_alpha, _, _, _ = result
                    self._record_measured_radius(measured_alpha)

            Qshape = (parent.datacube.Q_Nx, parent.datacube.Q_Ny)
            synthetic_params = dict(
                radius=self.synth_radius_spin.value(),
                width=self.synth_width_spin.value(),
                Qshape=Qshape,
            )
            self.probe = py4DSTEM.Probe.generate_synthetic_probe(**synthetic_params)
            self.alpha = synthetic_params["radius"]
            self.qx0, self.qy0 = Qshape[0] / 2.0, Qshape[1] / 2.0
            self.probe_params = dict(
                source="synthetic", generate_synthetic_probe=synthetic_params
            )

        else:
            parent.statusBar().showMessage(
                "Choose a vacuum region source first!", 5_000
            )
            return

        self.probe_view.setImage(
            self.probe.probe,
            autoLevels=True,
            autoRange=self.probe_view.image.shape != self.probe.probe.shape,
        )

        # Keep the kernel (and so the detections) in step with the probe.
        self.generate_kernel()

    def update_kernel_controls_enabled(self, mode):
        self.sigma_spin.setEnabled(mode == "gaussian")
        self.r_inner_spin.setEnabled(mode in ("sigmoid", "sigmoid_log"))
        self.r_outer_spin.setEnabled(mode in ("sigmoid", "sigmoid_log"))

    def generate_kernel(self):
        parent = self.window.parent
        if self.probe is None:
            parent.statusBar().showMessage("Generate a probe first!", 5_000)
            return

        mode = self.kernel_mode_combo.currentText()
        kernel_params = dict(mode=mode, origin=(self.qx0, self.qy0))
        if mode == "gaussian":
            kernel_params["sigma"] = self.sigma_spin.value()
        elif mode in ("sigmoid", "sigmoid_log"):
            kernel_params["radii"] = (
                self.alpha * self.r_inner_spin.value(),
                self.alpha * self.r_outer_spin.value(),
            )

        self.probe.get_kernel(**kernel_params)
        self.kernel_params = kernel_params
        kernel_shifted = np.fft.fftshift(self.probe.kernel)
        # Only reset the zoom when the shape changes, so live updates don't
        # undo any panning/zooming the user has done.
        self.kernel_view.setImage(
            kernel_shifted,
            autoLevels=True,
            autoRange=self.kernel_view.image.shape != kernel_shifted.shape,
        )

        cx, cy = kernel_shifted.shape[0] // 2, kernel_shifted.shape[1] // 2
        self._kernel_x_curve.setData(
            np.arange(kernel_shifted.shape[0]), kernel_shifted[:, cy]
        )
        self._kernel_y_curve.setData(
            np.arange(kernel_shifted.shape[1]), kernel_shifted[cx, :]
        )

        self.window.kernel_updated(self.probe, self.alpha)
