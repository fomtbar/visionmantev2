from __future__ import annotations

import cv2
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
from src.gui.dialogs.plc_test_dialog import PLCTestDialog
from src.gui.dialogs.reference_dialog import ReferenceSelectionDialog
from src.gui.dialogs.patterns_gallery_dialog import PatternsGalleryDialog
from src.gui.dialogs.camera_select_dialog import CameraSelectDialog
from src.gui.dialogs.multi_zone_setup_dialog import MultiZoneSetupDialog
from src.vision.windowed_matcher import SearchWindow
from src.utils.paths import get_app_root

_REFERENCE_IMAGES_DIR = get_app_root() / "data" / "reference_images"


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
        self._lbl_state.setStyleSheet("color: #aaa; margin: 0 10px;")
        self._status_bar.addWidget(self._lbl_state)

        self._lbl_plc = QLabel()
        self._lbl_plc.setStyleSheet("color: #aaa; margin: 0 6px;")
        self._status_bar.addWidget(self._lbl_plc)

        # Botón reconectar — solo visible cuando PLC habilitado y desconectado
        self._btn_plc_reconnect = QPushButton("↺ Reconectar")
        self._btn_plc_reconnect.setFixedHeight(22)
        self._btn_plc_reconnect.setStyleSheet(
            "QPushButton { color: #ffa040; border: 1px solid #ffa040; border-radius: 3px;"
            " padding: 1px 8px; font-size: 11px; margin: 2px; }"
            "QPushButton:hover { background-color: #2a1800; }"
        )
        self._btn_plc_reconnect.setVisible(False)
        self._btn_plc_reconnect.clicked.connect(self._on_plc_reconnect)
        self._status_bar.addWidget(self._btn_plc_reconnect)

        self._lbl_algo = QLabel(f"Algoritmo: {self._engine.algorithm_name}")
        self._lbl_algo.setStyleSheet("color: #aaa; margin: 0 10px;")
        self._status_bar.addPermanentWidget(self._lbl_algo)

        # Estado inicial del PLC en la barra
        self._refresh_plc_status_bar()

        # ── Toolbar ───────────────────────────────────────────────────────────
        toolbar = QToolBar("Principal")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        btn_plc = QPushButton("Config PLC")
        btn_plc.clicked.connect(self._open_plc_config)
        toolbar.addWidget(btn_plc)

        btn_plc_test = QPushButton("Probar PLC")
        btn_plc_test.setStyleSheet(
            "QPushButton { color: #ffd700; border: 1px solid #ffd700; border-radius: 3px;"
            " padding: 4px 10px; }"
            "QPushButton:hover { background-color: #2a2a10; }"
        )
        btn_plc_test.clicked.connect(self._open_plc_test)
        toolbar.addWidget(btn_plc_test)

        toolbar.addSeparator()

        self._btn_camera = QPushButton(f"Camara {self._cfg.camera.device_index}")
        self._btn_camera.setStyleSheet(
            "QPushButton { color: #7ec8e3; border: 1px solid #7ec8e3; border-radius: 3px;"
            " padding: 4px 10px; }"
            "QPushButton:hover { background-color: #0a2233; }"
        )
        self._btn_camera.clicked.connect(self._open_camera_select)
        toolbar.addWidget(self._btn_camera)

        toolbar.addSeparator()

        btn_save_cfg = QPushButton("Guardar config")
        btn_save_cfg.clicked.connect(self._save_config)
        toolbar.addWidget(btn_save_cfg)

        toolbar.addSeparator()

        btn_gallery = QPushButton("Ver patrones")
        btn_gallery.setStyleSheet(
            "QPushButton { color: #00bfff; border: 1px solid #00bfff; border-radius: 3px;"
            " padding: 4px 10px; }"
            "QPushButton:hover { background-color: #003355; }"
        )
        btn_gallery.clicked.connect(self._open_patterns_gallery)
        toolbar.addWidget(btn_gallery)

    def _build_controls(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(6)

        # ── Producción ────────────────────────────────────────────────────────
        self._btn_trigger = QPushButton("▶  INSPECCIONAR")
        self._btn_trigger.setFixedHeight(60)
        self._btn_trigger.setToolTip("Dispara una inspección (también: Espacio)")
        self._btn_trigger.setStyleSheet(
            "QPushButton { background-color: #1e6fe0; color: white; font-size: 18px; "
            "font-weight: bold; border-radius: 8px; }"
            "QPushButton:hover { background-color: #2a80f5; }"
            "QPushButton:pressed { background-color: #1450a0; }"
            "QPushButton:disabled { background-color: #444; color: #777; }"
        )
        self._btn_trigger.clicked.connect(self._engine.trigger)
        row.addWidget(self._btn_trigger, stretch=3)

        shortcut = QShortcut(QKeySequence(Qt.Key.Key_Space), self)
        shortcut.activated.connect(self._engine.trigger)

        # ── Configuración de zonas ────────────────────────────────────────────
        # Paso 1: setup de todas las piezas de una vez
        self._btn_ref = QPushButton("Configurar piezas")
        self._btn_ref.setFixedHeight(60)
        self._btn_ref.setToolTip(
            "Congela el frame y te deja marcar\n"
            "todas las piezas a buscar en un solo paso."
        )
        self._btn_ref.setStyleSheet(
            "QPushButton { background-color: #2c2c3e; color: #00dc50; font-size: 13px; "
            "font-weight: bold; border-radius: 8px; border: 2px solid #00dc50; }"
            "QPushButton:hover { background-color: #003322; }"
        )
        self._btn_ref.clicked.connect(self._setup_zones)
        row.addWidget(self._btn_ref, stretch=1)

        # Paso 2 (opcional): agregar más fotos de referencia a una zona ya creada
        self._btn_add_pattern = QPushButton("+ Variante")
        self._btn_add_pattern.setFixedHeight(60)
        self._btn_add_pattern.setToolTip(
            "Paso 2 (opcional) — Agregá otra foto de referencia\n"
            "a una zona existente (ej: pieza con otra orientación)."
        )
        self._btn_add_pattern.setStyleSheet(
            "QPushButton { background-color: #2c2c3e; color: #7ec8e3; font-size: 13px; "
            "border-radius: 8px; border: 2px solid #7ec8e3; }"
            "QPushButton:hover { background-color: #0a2233; }"
        )
        self._btn_add_pattern.clicked.connect(self._add_pattern_to_zone)
        row.addWidget(self._btn_add_pattern, stretch=1)

        # Limpiar todo
        btn_clear = QPushButton("Limpiar zonas")
        btn_clear.setFixedHeight(60)
        btn_clear.setToolTip("Elimina todas las zonas y sus patrones de referencia")
        btn_clear.setStyleSheet(
            "QPushButton { background-color: #2c2c3e; color: #ff6b6b; font-size: 13px; "
            "border-radius: 8px; border: 2px solid #ff6b6b; }"
            "QPushButton:hover { background-color: #330a0a; }"
        )
        btn_clear.clicked.connect(self._clear_rois)
        row.addWidget(btn_clear, stretch=1)

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
        self._engine.camera.frame_ready.connect(self._camera_view.update_frame)
        self._engine.inspection_complete.connect(self._on_inspection_complete)
        self._engine.state_changed.connect(self._on_state_changed)
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
        self._refresh_plc_status_bar(connected)

    def _refresh_plc_status_bar(self, connected: bool | None = None) -> None:
        plc_cfg = self._cfg.plc

        if not plc_cfg.enabled:
            self._lbl_plc.setText("PLC: deshabilitado")
            self._lbl_plc.setStyleSheet("color: #666; margin: 0 6px;")
            self._btn_plc_reconnect.setVisible(False)
            return

        if connected is None:
            connected = self._engine._plc.is_connected() if self._engine._plc else False

        if connected:
            self._lbl_plc.setText(
                f"PLC: {plc_cfg.brand.upper()}  {plc_cfg.ip}:{plc_cfg.port}  ✓"
            )
            self._lbl_plc.setStyleSheet("color: #00dc50; margin: 0 6px;")
            self._btn_plc_reconnect.setVisible(False)
        else:
            self._lbl_plc.setText(
                f"PLC: {plc_cfg.brand.upper()}  {plc_cfg.ip}:{plc_cfg.port}  ✗"
            )
            self._lbl_plc.setStyleSheet("color: #ff6b6b; margin: 0 6px;")
            self._btn_plc_reconnect.setVisible(True)

    def _on_plc_reconnect(self) -> None:
        self._btn_plc_reconnect.setEnabled(False)
        self._btn_plc_reconnect.setText("Conectando…")
        ok = self._engine.reconnect_plc()
        self._btn_plc_reconnect.setEnabled(True)
        self._btn_plc_reconnect.setText("↺ Reconectar")
        self._refresh_plc_status_bar(ok)

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

    def _setup_zones(self) -> None:
        """Abre el diálogo multi-zona: congela frame, marca N piezas, crea todo de una vez."""
        frame = self._engine.camera.get_frame()
        if frame is None:
            QMessageBox.warning(self, "Sin cámara", "No hay frame disponible.")
            return

        # Avisar si ya hay zonas configuradas
        existing = self._engine.roi_manager.get_zones()
        if existing:
            reply = QMessageBox.question(
                self, "Reemplazar zonas",
                f"Ya hay {len(existing)} zona(s) definida(s).\n"
                "¿Reemplazarlas con la nueva configuración?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return

        dlg = MultiZoneSetupDialog(frame, parent=self)
        if not dlg.exec() or not dlg.zones:
            return

        # Limpiar zonas anteriores (config + referencias ORB en memoria)
        self._engine.roi_manager.set_zones([])
        self._engine.clear_all_references()
        self._roi_counter = 0

        errors = []
        for draft in dlg.zones:
            zone_id = draft.name or f"pieza_{draft.index}"
            self._roi_counter += 1

            zone = ROIZone(
                id=zone_id,
                x=draft.rect.x(), y=draft.rect.y(),
                w=draft.rect.width(), h=draft.rect.height(),
            )
            self._engine.roi_manager.add_zone(zone)

            # Guardar patrón en disco
            zone_dir = _REFERENCE_IMAGES_DIR / zone_id
            zone_dir.mkdir(parents=True, exist_ok=True)
            ref_path = zone_dir / "001.png"
            cv2.imwrite(str(ref_path), draft.image)

            # Crear ventana de búsqueda PRIMERO para que add_zone_reference
            # cargue el patrón en el WindowedMatcher recién creado
            if draft.search_window is not None:
                sw = draft.search_window
                self._engine._classifier.set_zone_search_window(
                    zone_id,
                    SearchWindow(sw.x(), sw.y(), sw.width(), sw.height()),
                )

            # Cargar patrón en memoria (va al WindowedMatcher si existe, y a Template+ORB)
            if not self._engine.add_zone_reference(zone_id, draft.image):
                errors.append(zone_id)

        zones = self._engine.roi_manager.get_zones()
        self._camera_view.set_roi_zones(zones)
        self._cfg.update_rois([z.model_dump() for z in zones])
        self._cfg.save()

        n = len(dlg.zones)
        if errors:
            QMessageBox.warning(
                self, "Advertencia",
                f"{len(errors)} zona(s) sin features suficientes: {', '.join(errors)}\n"
                "Intentá con mejor iluminación o más textura en esa área."
            )
        self.statusBar().showMessage(
            f"{n} zona(s) configuradas — patrones guardados.", 5000
        )
        logger.info(f"Setup multi-zona: {n} zonas creadas")

    def _add_pattern_to_zone(self) -> None:
        """
        + Variante: agrega una foto de referencia adicional a una zona existente.
        Útil cuando la misma pieza puede aparecer en distintas orientaciones o
        condiciones de iluminación.
        """
        zones = self._engine.roi_manager.get_zones()
        if not zones:
            QMessageBox.information(
                self, "Sin zonas",
                "Primero configurá las piezas con 'Configurar piezas'."
            )
            return

        # Elegir zona — si hay más de una, mostrar selector con conteo actual
        if len(zones) == 1:
            zone_id = zones[0].id
        else:
            # Construir descripción con conteo de patrones por zona
            items = []
            for z in zones:
                n = self._engine._classifier.zone_pattern_count(z.id)
                items.append(f"{z.id}  ({n} foto{'s' if n != 1 else ''} actuales)")

            from PyQt6.QtWidgets import QInputDialog
            chosen, ok = QInputDialog.getItem(
                self,
                "¿A qué pieza agregar la variante?",
                "Elegí la zona a la que querés agregar una foto adicional.\n"
                "Cada foto extra ayuda cuando la pieza aparece en distintas posiciones:",
                items, 0, False
            )
            if not ok:
                return
            zone_id = zones[items.index(chosen)].id

        frame = self._engine.camera.get_frame()
        if frame is None:
            QMessageBox.warning(self, "Sin cámara", "No hay frame disponible.")
            return

        n_actual = self._engine._classifier.zone_pattern_count(zone_id)
        dlg = ReferenceSelectionDialog(frame, parent=self)
        dlg.setWindowTitle(
            f"Agregar variante a '{zone_id}'  (ya tiene {n_actual} foto{'s' if n_actual != 1 else ''})"
        )
        if not dlg.exec():
            return

        zone_dir = _REFERENCE_IMAGES_DIR / zone_id
        zone_dir.mkdir(parents=True, exist_ok=True)
        existing = list(zone_dir.glob("*.png"))
        ref_path = zone_dir / f"{len(existing) + 1:03d}.png"
        cv2.imwrite(str(ref_path), dlg.cropped_image)

        if not self._engine.add_zone_reference(zone_id, dlg.cropped_image):
            QMessageBox.warning(self, "Error",
                                "No se pudieron extraer features.\n"
                                "Intentá con mejor iluminación o más textura.")
            return

        total = n_actual + 1
        self.statusBar().showMessage(
            f"Variante agregada a '{zone_id}' — ahora tiene {total} foto(s) de referencia.", 5000
        )
        logger.info(f"Variante #{total} agregada a zona '{zone_id}'")

    def _clear_rois(self) -> None:
        zones = self._engine.roi_manager.get_zones()
        if not zones:
            self.statusBar().showMessage("No hay zonas definidas.", 2000)
            return

        from PyQt6.QtWidgets import QMessageBox as MB
        msg = MB(self)
        msg.setWindowTitle("Limpiar zonas")
        msg.setText(
            f"Hay {len(zones)} zona(s) configurada(s).\n\n"
            "¿Qué querés hacer?"
        )
        btn_all  = msg.addButton("Borrar todo (zonas + fotos del disco)", MB.ButtonRole.DestructiveRole)
        btn_cfg  = msg.addButton("Solo limpiar configuración (mantener fotos)", MB.ButtonRole.AcceptRole)
        btn_no   = msg.addButton("Cancelar", MB.ButtonRole.RejectRole)
        msg.setDefaultButton(btn_no)
        msg.exec()

        clicked = msg.clickedButton()
        if clicked == btn_no:
            return

        # Limpiar memoria y config
        self._engine.roi_manager.set_zones([])
        self._camera_view.set_roi_zones([])
        self._cfg.update_rois([])
        self._cfg.save()
        self._roi_counter = 0

        if clicked == btn_all:
            import shutil
            deleted = 0
            for z in zones:
                zone_dir = _REFERENCE_IMAGES_DIR / z.id
                if zone_dir.exists():
                    shutil.rmtree(zone_dir, ignore_errors=True)
                    deleted += 1
            self.statusBar().showMessage(
                f"Zonas eliminadas + {deleted} carpeta(s) de fotos borradas.", 4000
            )
        else:
            self.statusBar().showMessage(
                "Configuración limpiada. Las fotos siguen en disco.", 4000
            )

    def _open_patterns_gallery(self) -> None:
        zones = self._engine.roi_manager.get_zones()
        dlg = PatternsGalleryDialog(zones, self._engine.reload_zone_references, self)
        dlg.exec()

    def _open_plc_config(self) -> None:
        dlg = PLCConfigDialog(self._cfg.plc, self)
        if not dlg.exec():
            return
        new_cfg = dlg.get_config()
        self._cfg.update_plc(**new_cfg.model_dump())
        self._cfg.save()
        connected = self._engine.apply_plc_config(new_cfg)
        self._refresh_plc_status_bar(connected if new_cfg.enabled else None)
        msg = (
            f"PLC {new_cfg.brand.upper()} {new_cfg.ip}:{new_cfg.port} — conectado."
            if connected else
            f"Config guardada. PLC {new_cfg.brand.upper()} no conectó — reintentando en segundo plano."
            if new_cfg.enabled else
            "PLC deshabilitado."
        )
        self.statusBar().showMessage(msg, 4000)

    def _open_plc_test(self) -> None:
        dlg = PLCTestDialog(self)
        dlg.exec()

    def _open_camera_select(self) -> None:
        dlg = CameraSelectDialog(
            current_index=self._cfg.camera.device_index,
            config=self._cfg.camera,
            on_selected=self._switch_camera,
            parent=self,
        )
        dlg.exec()

    def _switch_camera(self, index: int) -> None:
        self._cfg.update_camera(device_index=index)
        self._engine.camera.update_config(self._cfg.camera)
        self._btn_camera.setText(f"Camara {index}")
        self._cfg.save()
        self.statusBar().showMessage(f"Camara {index} activa.", 3000)
        logger.info(f"Camara cambiada a indice {index}")

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
