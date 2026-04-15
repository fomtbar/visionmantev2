from __future__ import annotations

import concurrent.futures
import cv2
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QDialogButtonBox, QFrame, QScrollArea, QWidget,
)
from loguru import logger

from src.core.config_manager import CameraConfig


# ── Escaneo de cámaras en hilo ────────────────────────────────────────────────

def _probe_camera(index: int) -> dict | None:
    """Prueba un índice de cámara. Retorna info o None si no existe."""
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap.release()
        return None
    ret, _ = cap.read()
    if not ret:
        cap.release()
        return None
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return {"index": index, "width": w, "height": h}


def scan_cameras(max_index: int = 5) -> list[dict]:
    """Detecta cámaras disponibles (índices 0..max_index-1) en paralelo."""
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_index) as ex:
        futures = {ex.submit(_probe_camera, i): i for i in range(max_index)}
        for fut in concurrent.futures.as_completed(futures):
            info = fut.result()
            if info:
                results.append(info)
    return sorted(results, key=lambda x: x["index"])


class _ScanWorker(QThread):
    done = pyqtSignal(list)   # list[dict]

    def run(self):
        cameras = scan_cameras()
        self.done.emit(cameras)


# ── Diálogo ───────────────────────────────────────────────────────────────────

class CameraSelectDialog(QDialog):
    """
    Muestra las cámaras disponibles y permite cambiar la activa sin reiniciar.
    Llama a on_selected(index) cuando el usuario elige una.
    """

    def __init__(self, current_index: int, config: CameraConfig,
                 on_selected, parent=None):
        super().__init__(parent)
        self._current = current_index
        self._config = config
        self._on_selected = on_selected   # Callable[[int], None]
        self._cameras: list[dict] = []

        self.setWindowTitle("Seleccionar cámara")
        self.setMinimumWidth(420)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowType.WindowContextHelpButtonHint)
        self._build_ui()
        self._start_scan()

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        self._lbl_status = QLabel("Buscando cámaras…")
        self._lbl_status.setStyleSheet("color: #ffd700; font-size: 12px;")
        layout.addWidget(self._lbl_status)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("color: #333;")
        layout.addWidget(sep)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        self._list_widget = QWidget()
        self._list_widget.setStyleSheet("background: transparent;")
        self._list_layout = QVBoxLayout(self._list_widget)
        self._list_layout.setContentsMargins(0, 0, 0, 0)
        self._list_layout.setSpacing(6)
        scroll.setWidget(self._list_widget)
        layout.addWidget(scroll)

        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def _start_scan(self) -> None:
        self._worker = _ScanWorker()
        self._worker.done.connect(self._on_scan_done)
        self._worker.start()

    def _on_scan_done(self, cameras: list[dict]) -> None:
        self._cameras = cameras
        # Limpiar lista previa
        while self._list_layout.count():
            item = self._list_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        if not cameras:
            lbl = QLabel("No se detectaron cámaras.")
            lbl.setStyleSheet("color: #ff6b6b; padding: 8px;")
            self._list_layout.addWidget(lbl)
            self._lbl_status.setText("Sin cámaras detectadas.")
            return

        self._lbl_status.setText(f"{len(cameras)} cámara(s) encontrada(s)")

        for cam in cameras:
            self._list_layout.addWidget(self._make_camera_row(cam))

        self._list_layout.addStretch()

    def _make_camera_row(self, cam: dict) -> QFrame:
        idx = cam["index"]
        is_current = (idx == self._current)

        row = QFrame()
        row.setFrameShape(QFrame.Shape.StyledPanel)
        border_color = "#00dc50" if is_current else "#444"
        row.setStyleSheet(
            f"QFrame {{ background-color: #2c2c3e; border: 1px solid {border_color};"
            f" border-radius: 6px; }}"
        )
        h = QHBoxLayout(row)
        h.setContentsMargins(12, 8, 12, 8)

        # Icono + info
        lbl_icon = QLabel("📷")
        lbl_icon.setStyleSheet("font-size: 20px;")
        h.addWidget(lbl_icon)

        info_col = QVBoxLayout()
        lbl_name = QLabel(f"Cámara {idx}")
        lbl_name.setStyleSheet("color: #e0e0e0; font-size: 13px; font-weight: bold;")
        info_col.addWidget(lbl_name)
        lbl_res = QLabel(f"{cam['width']} × {cam['height']} px")
        lbl_res.setStyleSheet("color: #888; font-size: 11px;")
        info_col.addWidget(lbl_res)
        h.addLayout(info_col)
        h.addStretch()

        if is_current:
            lbl_active = QLabel("● Activa")
            lbl_active.setStyleSheet("color: #00dc50; font-size: 11px; font-weight: bold;")
            h.addWidget(lbl_active)
        else:
            btn = QPushButton("Usar esta")
            btn.setFixedHeight(30)
            btn.setStyleSheet(
                "QPushButton { background-color: #1e6fe0; color: white; font-weight: bold;"
                " border-radius: 4px; padding: 2px 14px; }"
                "QPushButton:hover { background-color: #2a80f5; }"
            )
            btn.clicked.connect(lambda _, i=idx: self._select(i))
            h.addWidget(btn)

        return row

    def _select(self, index: int) -> None:
        self._current = index
        self._on_selected(index)
        # Refrescar para marcar la nueva activa
        self._on_scan_done(self._cameras)
