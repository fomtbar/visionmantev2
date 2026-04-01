from __future__ import annotations

import numpy as np
from PyQt6.QtCore import Qt, QRect
from PyQt6.QtGui import QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QPushButton, QLabel, QStatusBar, QToolBar,
    QMessageBox, QInputDialog, QSizePolicy,
    QGroupBox, QCheckBox, QSpinBox, QFormLayout,
    QDoubleSpinBox,
)
from loguru import logger

from src.core.inspection_engine import InspectionEngine, EngineState
from src.core.result_model import InspectionResult
from src.core.config_manager import ConfigManager, ROIZone
from src.gui.widgets.camera_view import CameraView
from src.gui.widgets.result_panel import ResultPanel
from src.gui.dialogs.plc_config_dialog import PLCConfigDialog


class MainWindow(QMainWindow):
    def __init__(self, engine: InspectionEngine, config: ConfigManager, parent=None):
        super().__init__(parent)
        self._engine = engine
        self._cfg = config
        self._roi_counter = 0

        self.setWindowTitle(f"VisionMante v2 — {config.app.job_name}")
        self.setMinimumSize(1100, 700)
        self._apply_dark_theme()
        self._build_ui()
        self._connect_signals()

        logger.info("MainWindow inicializado")

    # ── UI Construction ───────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(6, 6, 6, 6)
        main_layout.setSpacing(8)

        # ── Columna izquierda: cámara + controles ─────────────────────────────
        left_col = QVBoxLayout()
        left_col.setSpacing(6)

        # Cámara
        self._camera_view = CameraView()
        self._camera_view.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._camera_view.roi_defined.connect(self._on_roi_defined)
        left_col.addWidget(self._camera_view)

        # Controles principales
        controls = self._build_controls()
        left_col.addLayout(controls)

        main_layout.addLayout(left_col, stretch=3)

        # ── Columna derecha: resultado + config ───────────────────────────────
        right_col = QVBoxLayout()
        right_col.setSpacing(6)

        self._result_panel = ResultPanel(self._engine.history)
        self._result_panel.setFixedWidth(300)
        right_col.addWidget(self._result_panel)

        # Panel de configuración rápida
        quick_cfg = self._build_quick_config()
        right_col.addWidget(quick_cfg)
        right_col.addStretch()

        main_layout.addLayout(right_col, stretch=0)

        # ── Status bar ────────────────────────────────────────────────────────
        self._status_bar = self.statusBar()
        self._lbl_state = QLabel("Estado: IDLE")
        self._lbl_plc = QLabel("PLC: desconectado")
        self._lbl_algo = QLabel(f"Algoritmo: {self._engine.algorithm_name}")
        for lbl in (self._lbl_state, self._lbl_plc, self._lbl_algo):
            lbl.setStyleSheet("color: #aaa; margin: 0 10px;")
        self._status_bar.addWidget(self._lbl_state)
        self._status_bar.addWidget(self._lbl_plc)
        self._status_bar.addPermanentWidget(self._lbl_algo)

        # ── Toolbar ───────────────────────────────────────────────────────────
        toolbar = QToolBar("Principal")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        btn_plc = QPushButton("Config PLC")
        btn_plc.clicked.connect(self._open_plc_config)
        toolbar.addWidget(btn_plc)

        toolbar.addSeparator()

        btn_save_cfg = QPushButton("Guardar config")
        btn_save_cfg.clicked.connect(self._save_config)
        toolbar.addWidget(btn_save_cfg)

    def _build_controls(self) -> QHBoxLayout:
        row = QHBoxLayout()

        # Botón TRIGGER (grande, llamativo)
        self._btn_trigger = QPushButton("▶  INSPECCIONAR")
        self._btn_trigger.setFixedHeight(60)
        self._btn_trigger.setStyleSheet(
            "QPushButton { background-color: #1e6fe0; color: white; font-size: 18px; "
            "font-weight: bold; border-radius: 8px; }"
            "QPushButton:hover { background-color: #2a80f5; }"
            "QPushButton:pressed { background-color: #1450a0; }"
            "QPushButton:disabled { background-color: #444; color: #777; }"
        )
        self._btn_trigger.clicked.connect(self._engine.trigger)
        row.addWidget(self._btn_trigger, stretch=3)

        # Shortcut: Espacio = trigger
        shortcut = QShortcut(QKeySequence(Qt.Key.Key_Space), self)
        shortcut.activated.connect(self._engine.trigger)

        # Botón definir ROI
        self._btn_roi = QPushButton("✏  Definir ROI")
        self._btn_roi.setFixedHeight(60)
        self._btn_roi.setCheckable(True)
        self._btn_roi.setStyleSheet(
            "QPushButton { background-color: #2c2c3e; color: #ffd700; font-size: 14px; "
            "border-radius: 8px; border: 2px solid #ffd700; }"
            "QPushButton:checked { background-color: #ffd700; color: #1a1a2e; }"
        )
        self._btn_roi.toggled.connect(self._camera_view.set_draw_mode)
        row.addWidget(self._btn_roi, stretch=1)

        # Botón capturar referencia ORB
        self._btn_ref = QPushButton("📷  Capturar referencia OK")
        self._btn_ref.setFixedHeight(60)
        self._btn_ref.setStyleSheet(
            "QPushButton { background-color: #2c2c3e; color: #00dc50; font-size: 13px; "
            "border-radius: 8px; border: 2px solid #00dc50; }"
            "QPushButton:hover { background-color: #003322; }"
        )
        self._btn_ref.clicked.connect(self._capture_reference)
        row.addWidget(self._btn_ref, stretch=1)

        # Botón limpiar ROIs
        btn_clear_roi = QPushButton("🗑  Limpiar ROIs")
        btn_clear_roi.setFixedHeight(60)
        btn_clear_roi.setStyleSheet(
            "QPushButton { background-color: #2c2c3e; color: #ff6b6b; font-size: 13px; "
            "border-radius: 8px; border: 2px solid #ff6b6b; }"
        )
        btn_clear_roi.clicked.connect(self._clear_rois)
        row.addWidget(btn_clear_roi, stretch=1)

        return row

    def _build_quick_config(self) -> QGroupBox:
        group = QGroupBox("Configuración rápida")
        group.setStyleSheet("QGroupBox { color: #ccc; border: 1px solid #444; border-radius: 4px; margin-top: 6px; }"
                            "QGroupBox::title { subcontrol-origin: margin; left: 8px; }")
        form = QFormLayout(group)
        form.setSpacing(6)

        self._sb_threshold = QDoubleSpinBox()
        self._sb_threshold.setRange(0.3, 0.99)
        self._sb_threshold.setSingleStep(0.05)
        self._sb_threshold.setDecimals(2)
        self._sb_threshold.setValue(self._cfg.vision.confidence_threshold)
        self._sb_threshold.valueChanged.connect(
            lambda v: self._cfg.update_vision(confidence_threshold=v)
        )
        form.addRow("Umbral confianza:", self._sb_threshold)

        self._sb_pieces = QSpinBox()
        self._sb_pieces.setRange(1, 20)
        self._sb_pieces.setValue(self._cfg.vision.expected_pieces)
        self._sb_pieces.valueChanged.connect(
            lambda v: self._cfg.update_vision(expected_pieces=v)
        )
        form.addRow("Piezas esperadas:", self._sb_pieces)

        return group

    # ── Signal connections ────────────────────────────────────────────────────

    def _connect_signals(self) -> None:
        # Cámara → vista
        self._engine.camera.frame_ready.connect(self._camera_view.update_frame)

        # Resultado → vista y panel
        self._engine.inspection_complete.connect(self._on_inspection_complete)

        # Estado del motor
        self._engine.state_changed.connect(self._on_state_changed)

        # PLC status
        self._engine.plc_status_changed.connect(self._on_plc_status)

    # ── Slots ─────────────────────────────────────────────────────────────────

    def _on_inspection_complete(self, result: InspectionResult) -> None:
        self._camera_view.update_result(result)
        self._result_panel.update_result(result)
        self._lbl_algo.setText(f"Algoritmo: {self._engine.algorithm_name}")

    def _on_state_changed(self, state_name: str) -> None:
        self._lbl_state.setText(f"Estado: {state_name}")
        busy = state_name not in ("IDLE",)
        self._btn_trigger.setEnabled(not busy)

    def _on_plc_status(self, connected: bool) -> None:
        if connected:
            info = self._cfg.plc
            self._lbl_plc.setText(f"PLC: {info.brand.upper()} {info.ip} ✓")
            self._lbl_plc.setStyleSheet("color: #00dc50; margin: 0 10px;")
        else:
            self._lbl_plc.setText("PLC: desconectado")
            self._lbl_plc.setStyleSheet("color: #ff6b6b; margin: 0 10px;")

    def _on_roi_defined(self, rect: QRect) -> None:
        self._roi_counter += 1
        zone_id, ok = QInputDialog.getText(
            self, "Nueva zona ROI",
            "Nombre de la zona:",
            text=f"zona_{self._roi_counter}"
        )
        if not ok or not zone_id.strip():
            return

        zone = ROIZone(
            id=zone_id.strip(),
            x=rect.x(), y=rect.y(),
            w=rect.width(), h=rect.height(),
        )
        self._engine.roi_manager.add_zone(zone)
        zones = self._engine.roi_manager.get_zones()
        self._camera_view.set_roi_zones(zones)
        self._cfg.update_rois([z.model_dump() for z in zones])
        logger.info(f"ROI agregada: {zone_id} ({rect.width()}x{rect.height()})")

        # Desactivar modo dibujo automáticamente
        self._btn_roi.setChecked(False)

    def _capture_reference(self) -> None:
        frame = self._engine.camera.get_frame()
        if frame is None:
            QMessageBox.warning(self, "Sin cámara", "No hay frame disponible.")
            return
        if self._engine.load_orb_reference(frame):
            QMessageBox.information(self, "Referencia OK",
                                    "Imagen de referencia capturada correctamente.\n"
                                    "El sistema comparará contra esta imagen.")
            logger.info("Referencia ORB capturada desde GUI")
        else:
            QMessageBox.warning(self, "Error", "No se pudo extraer features de la imagen.")

    def _clear_rois(self) -> None:
        reply = QMessageBox.question(
            self, "Limpiar ROIs",
            "¿Eliminar todas las zonas de inspección?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self._engine.roi_manager.set_zones([])
            self._camera_view.set_roi_zones([])
            self._cfg.update_rois([])
            self._roi_counter = 0

    def _open_plc_config(self) -> None:
        dlg = PLCConfigDialog(self._cfg.plc, self)
        if dlg.exec():
            new_plc_cfg = dlg.get_config()
            self._cfg.update_plc(**new_plc_cfg.model_dump())
            QMessageBox.information(self, "PLC", "Config guardada. Reinicia para aplicar.")

    def _save_config(self) -> None:
        self._cfg.save()
        self.statusBar().showMessage("Config guardada.", 3000)

    # ── Theme ─────────────────────────────────────────────────────────────────

    def _apply_dark_theme(self) -> None:
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #1a1a2e;
                color: #e0e0e0;
            }
            QPushButton {
                background-color: #2c2c3e;
                color: #e0e0e0;
                border: 1px solid #444;
                border-radius: 4px;
                padding: 4px 10px;
            }
            QPushButton:hover { background-color: #3a3a50; }
            QLabel { color: #e0e0e0; }
            QGroupBox { color: #ccc; }
            QSpinBox, QDoubleSpinBox, QLineEdit, QComboBox {
                background-color: #2c2c3e;
                color: #e0e0e0;
                border: 1px solid #555;
                border-radius: 3px;
                padding: 2px 4px;
            }
            QToolBar {
                background-color: #12121e;
                border-bottom: 1px solid #333;
                padding: 4px;
                spacing: 6px;
            }
            QStatusBar { background-color: #12121e; color: #888; }
        """)

    def closeEvent(self, event) -> None:
        self._engine.shutdown()
        event.accept()
