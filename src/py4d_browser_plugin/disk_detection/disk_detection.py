from PyQt5 import QtCore
from PyQt5.QtWidgets import (
    QCheckBox,
    QDialog,
    QHBoxLayout,
    QPushButton,
    QScrollArea,
    QSplitter,
    QToolButton,
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
        self._reload_pending = False

        plugin_action.triggered.connect(self.launch_window)
        parent.signal_datacube_changed.connect(self.on_datacube_changed)

    def close(self):
        if self.window is not None:
            self.window.close()

    def launch_window(self):
        if self.window is None:
            self.window = DiskDetectionWindow(parent=self.parent)
            self.window.sig_find_all_finished.connect(self._on_find_all_finished)

        if not self.window.isVisible():
            # (Re)opening: start the previews from the main window's view.
            self.window.bragg_disk_panel.show_main_view()

        self.window.show()
        self.window.raise_()
        self.window.activateWindow()

    def on_datacube_changed(self):
        if self.window is None:
            return
        if self.window.finding_all:
            # Find All Bragg Disks keeps the GUI responsive while it runs,
            # which is how a new dataset can get loaded mid-run. It stops
            # itself when it notices; reload once it has.
            self._reload_pending = True
            return
        self.reload_window()

    def _on_find_all_finished(self):
        if self._reload_pending:
            self._reload_pending = False
            self.reload_window()

    def reload_window(self):
        # The probe, kernel, previews and results all belong to the old
        # dataset (whose scan/pattern shapes may differ), so start over with
        # a fresh window -- reopened in the same place if it was open.
        old_window, self.window = self.window, None
        was_visible = old_window.isVisible()
        geometry = old_window.saveGeometry()
        old_window.close()
        old_window.deleteLater()

        if was_visible:
            self.launch_window()
            self.window.restoreGeometry(geometry)


class DiskDetectionWindow(QDialog):
    # Emitted when a Find All Bragg Disks run ends, however it ends.
    sig_find_all_finished = QtCore.pyqtSignal()

    def __init__(self, parent: "DataViewer"):
        super().__init__(parent=parent)

        self.parent = parent
        self.probe = None  # the current Probe object, once a kernel is generated
        self.probe_radius = None  # measured bright-field disk radius, in pixels
        self.finding_all = False  # a Find All Bragg Disks run is in progress

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

        self.controls_scroll = QScrollArea()
        self.controls_scroll.setWidgetResizable(True)
        self.controls_scroll.setWidget(controls_widget)
        # Scroll vertically only -- never squeeze the controls narrower than
        # they need to be.
        self.controls_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        controls_width = (
            controls_widget.minimumSizeHint().width()
            + self.controls_scroll.verticalScrollBar().sizeHint().width()
        )
        self.controls_scroll.setMinimumWidth(controls_width)

        views_splitter = QSplitter(QtCore.Qt.Vertical)
        views_splitter.addWidget(self.probe_kernel_panel.views)
        views_splitter.addWidget(self.bragg_disk_panel.views)
        views_splitter.setSizes([300, 600])

        # Retracts the controls column so the views fill the window. Lives
        # outside the column itself so it stays reachable while retracted.
        self.retract_button = QToolButton()
        self.retract_button.setAutoRaise(True)
        self.retract_button.clicked.connect(self.toggle_controls)

        retract_strip = QVBoxLayout()
        retract_strip.setContentsMargins(0, 0, 0, 0)
        retract_strip.addWidget(self.retract_button)
        retract_strip.addStretch()

        views_layout = QHBoxLayout()
        views_layout.setContentsMargins(0, 0, 0, 0)
        views_layout.setSpacing(0)
        views_layout.addLayout(retract_strip)
        views_layout.addWidget(views_splitter)
        views_widget = QWidget()
        views_widget.setLayout(views_layout)

        main_splitter = QSplitter(QtCore.Qt.Horizontal)
        main_splitter.addWidget(self.controls_scroll)
        main_splitter.addWidget(views_widget)
        # Collapsing is the retract button's job; dragging the handle all
        # the way left would leave it out of step.
        main_splitter.setCollapsible(0, False)
        main_splitter.setSizes([controls_width, 1400 - controls_width])

        self._set_controls_visible(True)

        layout = QVBoxLayout(self)
        layout.addWidget(main_splitter)

        self.resize(1400, 900)

        self.disable_auto_default(self)

    def disable_auto_default(self, widget):
        # QDialog makes every push button in it "auto-default", drawing the
        # first one (the "From selection" source button) highlighted as if
        # already chosen, and having Enter in any spinbox click it. Needs
        # calling again for buttons added later (e.g. new preview panes).
        for button in widget.findChildren(QPushButton):
            button.setAutoDefault(False)

    def toggle_controls(self):
        self._set_controls_visible(self.controls_scroll.isHidden())

    def _set_controls_visible(self, visible):
        self.controls_scroll.setVisible(visible)
        if visible:
            self.retract_button.setArrowType(QtCore.Qt.LeftArrow)
            self.retract_button.setToolTip("Hide parameters")
        else:
            self.retract_button.setArrowType(QtCore.Qt.RightArrow)
            self.retract_button.setToolTip("Show parameters")

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
