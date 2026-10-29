from PyQt5 import QtCore
from PyQt5.QtWidgets import (
    QCheckBox,
    QDialog,
    QScrollArea,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from .bragg_disk_panel import BraggDiskPanel
from .probe_kernel_panel import ProbeKernelPanel

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from py4D_browser import DataViewer


class DiskDetectionPlugin(QWidget):

    # required for py4DGUI to recognize this as a plugin.
    plugin_id = "py4DGUI.external.disk_detection"

    uses_single_action = True
    display_name = "Disk Detection..."

    def __init__(self, parent: "DataViewer", plugin_action, **kwargs):
        super().__init__()

        self.parent = parent
        self.window = None

        plugin_action.triggered.connect(self.launch_window)

    def close(self):
        if self.window is not None:
            self.window.close()

    def launch_window(self):
        if self.window is None:
            self.window = DiskDetectionWindow(parent=self.parent)

        self.window.show()
        self.window.raise_()
        self.window.activateWindow()


class DiskDetectionWindow(QDialog):
    def __init__(self, parent: "DataViewer"):
        super().__init__(parent=parent)

        self.parent = parent
        self.probe = None  # the current Probe object, once a kernel is generated
        self.probe_radius = None  # measured bright-field disk radius, in pixels

        self.setWindowTitle("Disk Detection")

        # Off by default: changes only take effect via the Generate/Update
        # buttons, and previews only update when a point selector drag is
        # released. When checked, any probe, kernel or detection parameter
        # change regenerates everything downstream of it, and previews also
        # update continuously while dragging (can be too slow/jumpy to want
        # on all the time).
        self.live_update_checkbox = QCheckBox("Live Update")
        self.live_update_checkbox.setChecked(False)
        self.live_update_checkbox.toggled.connect(self.on_live_update_toggled)

        self.probe_kernel_panel = ProbeKernelPanel(window=self)
        self.bragg_disk_panel = BraggDiskPanel(window=self)

        controls_layout = QVBoxLayout()
        controls_layout.addWidget(self.live_update_checkbox)
        controls_layout.addWidget(self.probe_kernel_panel)
        controls_layout.addWidget(self.bragg_disk_panel)
        controls_layout.addStretch()
        controls_widget = QWidget()
        controls_widget.setLayout(controls_layout)

        controls_scroll = QScrollArea()
        controls_scroll.setWidgetResizable(True)
        controls_scroll.setWidget(controls_widget)
        # Scroll vertically only -- never squeeze the controls narrower than
        # they need to be.
        controls_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        controls_scroll.setMinimumWidth(
            controls_widget.sizeHint().width()
            + controls_scroll.verticalScrollBar().sizeHint().width()
        )

        views_splitter = QSplitter(QtCore.Qt.Vertical)
        views_splitter.addWidget(self.probe_kernel_panel.views)
        views_splitter.addWidget(self.bragg_disk_panel.views)
        views_splitter.setSizes([300, 600])

        main_splitter = QSplitter(QtCore.Qt.Horizontal)
        main_splitter.addWidget(controls_scroll)
        main_splitter.addWidget(views_splitter)
        main_splitter.setSizes([360, 1040])

        layout = QVBoxLayout(self)
        layout.addWidget(main_splitter)

        self.resize(1400, 900)

    def is_live(self):
        return self.live_update_checkbox.isChecked()

    def on_live_update_toggled(self, checked):
        # Catch the views up with any changes made while live update was off.
        if checked:
            self.probe_kernel_panel.on_probe_params_changed()

    def kernel_updated(self, probe, probe_radius):
        self.probe = probe
        self.probe_radius = probe_radius
        self.bragg_disk_panel.on_kernel_updated()
