from __future__ import annotations

import cv2
from pathlib import Path

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QScrollArea, QWidget, QFrame,
    QMessageBox,
)

from src.core.config_manager import ROIZone
from src.utils.paths import get_app_root

_REFERENCE_IMAGES_DIR = get_app_root() / "data" / "reference_images"


def _get_zone_patterns(zone_id: str) -> list[Path]:
    """Retorna lista ordenada de archivos de patrón para la zona."""
    zone_dir = _REFERENCE_IMAGES_DIR / zone_id
    if zone_dir.is_dir():
        return sorted(zone_dir.glob("*.png"))
    # Formato legacy: archivo único
    legacy = _REFERENCE_IMAGES_DIR / f"{zone_id}.png"
    if legacy.exists():
        return [legacy]
    return []


class _PatternThumbnail(QFrame):
    """Miniatura de un patrón individual con botón de eliminar."""

    deleted = pyqtSignal(Path)  # ruta del archivo eliminado

    def __init__(self, path: Path, index: int, parent=None):
        super().__init__(parent)
        self._path = path
        self._build_ui(index)

    def _build_ui(self, index: int) -> None:
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setStyleSheet(
            "_PatternThumbnail { background-color: #1e1e30; border: 1px solid #555;"
            " border-radius: 4px; }"
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        thumb = QLabel()
        thumb.setFixedSize(120, 90)
        thumb.setAlignment(Qt.AlignmentFlag.AlignCenter)

        pix = QPixmap(str(self._path))
        if not pix.isNull():
            thumb.setPixmap(
                pix.scaled(120, 90,
                           Qt.AspectRatioMode.KeepAspectRatio,
                           Qt.TransformationMode.SmoothTransformation)
            )
            thumb.setStyleSheet("background-color: #111; border-radius: 3px;")
        else:
            thumb.setText("Error")
            thumb.setStyleSheet("background-color: #111; color: #555; font-size: 10px;")
        layout.addWidget(thumb)

        lbl = QLabel(f"#{index}")
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl.setStyleSheet("color: #aaa; font-size: 10px;")
        layout.addWidget(lbl)

        btn_del = QPushButton("✕")
        btn_del.setFixedHeight(22)
        btn_del.setToolTip("Eliminar este patrón")
        btn_del.setStyleSheet(
            "QPushButton { background-color: #3a1a1a; color: #ff6b6b;"
            " border: 1px solid #ff6b6b; border-radius: 3px; font-size: 11px; }"
            "QPushButton:hover { background-color: #5a2a2a; }"
        )
        btn_del.clicked.connect(self._on_delete)
        layout.addWidget(btn_del)

    def _on_delete(self) -> None:
        reply = QMessageBox.question(
            self, "Eliminar patrón",
            f"¿Eliminar el patrón '{self._path.name}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            try:
                self._path.unlink()
                self.deleted.emit(self._path)
            except Exception as exc:
                QMessageBox.warning(self, "Error", f"No se pudo eliminar:\n{exc}")


class _PatternCard(QFrame):
    """Tarjeta con todos los patrones de una zona ROI."""

    reference_deleted = pyqtSignal(str)   # zone_id (algún patrón fue eliminado)
    add_pattern_requested = pyqtSignal(str)  # zone_id

    def __init__(self, zone: ROIZone, parent=None):
        super().__init__(parent)
        self._zone = zone
        self._patterns = _get_zone_patterns(zone.id)
        self._build_ui()

    def _build_ui(self) -> None:
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setStyleSheet(
            "_PatternCard { background-color: #2c2c3e; border: 1px solid #444;"
            " border-radius: 6px; }"
        )
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 10, 12, 10)
        root.setSpacing(8)

        # ── Encabezado ─────────────────────────────────────────────────────────
        header = QHBoxLayout()

        lbl_id = QLabel(f"Zona: {self._zone.id}")
        lbl_id.setStyleSheet("color: #ffd700; font-size: 14px; font-weight: bold;")
        header.addWidget(lbl_id)

        header.addStretch()

        n = len(self._patterns)
        status_color = "#00dc50" if n > 0 else "#ff6b6b"
        status_text = f"{n} patrón{'es' if n != 1 else ''}" if n > 0 else "Sin patrones"
        lbl_count = QLabel(status_text)
        lbl_count.setStyleSheet(f"color: {status_color}; font-size: 12px;")
        header.addWidget(lbl_count)

        root.addLayout(header)

        lbl_roi = QLabel(
            f"ROI: {self._zone.w} × {self._zone.h} px   origen ({self._zone.x}, {self._zone.y})"
        )
        lbl_roi.setStyleSheet("color: #888; font-size: 11px;")
        root.addWidget(lbl_roi)

        # ── Miniaturas de patrones ─────────────────────────────────────────────
        thumbs_row = QHBoxLayout()
        thumbs_row.setSpacing(8)
        thumbs_row.setAlignment(Qt.AlignmentFlag.AlignLeft)

        for i, path in enumerate(self._patterns, start=1):
            thumb = _PatternThumbnail(path, i)
            thumb.deleted.connect(self._on_pattern_deleted)
            thumbs_row.addWidget(thumb)

        if not self._patterns:
            lbl_empty = QLabel("Sin imágenes de referencia.\nCaptura una con el botón principal.")
            lbl_empty.setStyleSheet("color: #555; font-size: 11px;")
            thumbs_row.addWidget(lbl_empty)

        root.addLayout(thumbs_row)

    def _on_pattern_deleted(self, path: Path) -> None:
        self.reference_deleted.emit(self._zone.id)


class PatternsGalleryDialog(QDialog):
    """
    Galería de patrones de referencia.
    Muestra todas las zonas ROI con sus imágenes OK guardadas en disco.
    Soporta múltiples patrones por zona.
    """

    def __init__(self, zones: list[ROIZone], parent=None):
        super().__init__(parent)
        self._zones = zones
        self.setWindowTitle("Galería de patrones de referencia")
        self.setModal(True)
        self.setMinimumSize(780, 540)
        self.resize(820, 600)
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 14, 16, 12)
        root.setSpacing(10)

        # Header
        lbl_title = QLabel("Patrones de referencia")
        lbl_title.setStyleSheet("color: #e0e0e0; font-size: 16px; font-weight: bold;")
        root.addWidget(lbl_title)

        n_zones = len(self._zones)
        total_patterns = sum(len(_get_zone_patterns(z.id)) for z in self._zones)
        lbl_sub = QLabel(
            f"{n_zones} zona{'s' if n_zones != 1 else ''} definida{'s' if n_zones != 1 else ''}"
            f"  ·  {total_patterns} patrón(es) guardado(s)"
        )
        lbl_sub.setStyleSheet("color: #888; font-size: 11px;")
        root.addWidget(lbl_sub)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("color: #333;")
        root.addWidget(sep)

        # Scroll
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet(
            "QScrollArea { border: none; background-color: #1a1a2e; }"
            "QScrollBar:vertical { background: #1a1a2e; width: 10px; border-radius: 5px; }"
            "QScrollBar::handle:vertical { background: #444; border-radius: 5px; }"
        )

        container = QWidget()
        container.setStyleSheet("background-color: #1a1a2e;")
        vbox = QVBoxLayout(container)
        vbox.setContentsMargins(4, 6, 4, 6)
        vbox.setSpacing(10)

        if not self._zones:
            lbl_empty = QLabel(
                "No hay zonas ROI definidas.\n\n"
                "Usa el botón  '📷 Definir zona + Referencia OK'  para comenzar."
            )
            lbl_empty.setStyleSheet("color: #555; font-size: 13px;")
            lbl_empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
            vbox.addWidget(lbl_empty)
        else:
            for zone in self._zones:
                card = _PatternCard(zone)
                card.reference_deleted.connect(self._on_reference_deleted)
                vbox.addWidget(card)

        vbox.addStretch()
        scroll.setWidget(container)
        root.addWidget(scroll)

        # Footer
        footer = QHBoxLayout()
        lbl_dir = QLabel(f"Directorio: {_REFERENCE_IMAGES_DIR}")
        lbl_dir.setStyleSheet("color: #555; font-size: 10px;")
        footer.addWidget(lbl_dir, stretch=1)

        btn_close = QPushButton("Cerrar")
        btn_close.setFixedSize(100, 34)
        btn_close.setStyleSheet(
            "QPushButton { background-color: #2c2c3e; color: #e0e0e0;"
            " border: 1px solid #555; border-radius: 4px; }"
            "QPushButton:hover { background-color: #3a3a50; }"
        )
        btn_close.clicked.connect(self.accept)
        footer.addWidget(btn_close)
        root.addLayout(footer)

    def _on_reference_deleted(self, zone_id: str) -> None:
        QMessageBox.information(
            self,
            "Patrón eliminado",
            f"Patrón de '{zone_id}' eliminado del disco.\n"
            "Cierra y vuelve a abrir la galería para ver los cambios.\n"
            "Reinicia la app para que el motor refresque los patrones en memoria.",
        )
