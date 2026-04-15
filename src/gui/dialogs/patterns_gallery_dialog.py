from __future__ import annotations

from pathlib import Path
from typing import Callable

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
    zone_dir = _REFERENCE_IMAGES_DIR / zone_id
    if zone_dir.is_dir():
        return sorted(zone_dir.glob("*.png"))
    legacy = _REFERENCE_IMAGES_DIR / f"{zone_id}.png"
    if legacy.exists():
        return [legacy]
    return []


# ── Miniatura individual ──────────────────────────────────────────────────────

class _PatternThumbnail(QFrame):
    deleted = pyqtSignal(Path)

    def __init__(self, path: Path, index: int, parent=None):
        super().__init__(parent)
        self._path = path
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
        pix = QPixmap(str(path))
        if not pix.isNull():
            thumb.setPixmap(pix.scaled(120, 90,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation))
            thumb.setStyleSheet("background-color: #111; border-radius: 3px;")
        else:
            thumb.setText("Error")
            thumb.setStyleSheet("background-color: #111; color: #555; font-size: 10px;")
        layout.addWidget(thumb)

        lbl = QLabel(f"#{index}")
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl.setStyleSheet("color: #aaa; font-size: 10px;")
        layout.addWidget(lbl)

        btn_del = QPushButton("✕ Eliminar")
        btn_del.setFixedHeight(22)
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
            f"¿Eliminar '{self._path.name}'?\n"
            "El motor actualizará los patrones automáticamente.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            try:
                self._path.unlink()
                self.deleted.emit(self._path)
            except Exception as exc:
                QMessageBox.warning(self, "Error", f"No se pudo eliminar:\n{exc}")


# ── Tarjeta de zona ───────────────────────────────────────────────────────────

class _PatternCard(QFrame):
    """
    Tarjeta de una zona ROI con sus miniaturas.
    Se auto-actualiza cuando se elimina un patrón sin necesidad de recrear el diálogo.
    """

    def __init__(self, zone: ROIZone,
                 on_deleted: Callable[[str], int],   # (zone_id) → patrones restantes en memoria
                 parent=None):
        super().__init__(parent)
        self._zone = zone
        self._on_deleted = on_deleted
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setStyleSheet(
            "_PatternCard { background-color: #2c2c3e; border: 1px solid #444;"
            " border-radius: 6px; }"
        )
        self._build_ui()

    def _build_ui(self) -> None:
        self._root = QVBoxLayout(self)
        self._root.setContentsMargins(12, 10, 12, 10)
        self._root.setSpacing(8)
        self._render_content()

    def _render_content(self) -> None:
        # Limpiar widgets previos (para refrescar sin recrear la card)
        while self._root.count():
            item = self._root.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        patterns = _get_zone_patterns(self._zone.id)

        # Encabezado
        header = QHBoxLayout()
        lbl_id = QLabel(f"Zona: {self._zone.id}")
        lbl_id.setStyleSheet("color: #ffd700; font-size: 14px; font-weight: bold;")
        header.addWidget(lbl_id)
        header.addStretch()

        n = len(patterns)
        color = "#00dc50" if n > 0 else "#ff6b6b"
        self._lbl_count = QLabel(f"{n} patrón{'es' if n != 1 else ''}" if n else "Sin patrones")
        self._lbl_count.setStyleSheet(f"color: {color}; font-size: 12px;")
        header.addWidget(self._lbl_count)

        header_w = QWidget()
        header_w.setLayout(header)
        self._root.addWidget(header_w)

        lbl_roi = QLabel(
            f"ROI: {self._zone.w} × {self._zone.h} px   origen ({self._zone.x}, {self._zone.y})"
        )
        lbl_roi.setStyleSheet("color: #888; font-size: 11px;")
        self._root.addWidget(lbl_roi)

        # Miniaturas
        thumbs_w = QWidget()
        thumbs_row = QHBoxLayout(thumbs_w)
        thumbs_row.setContentsMargins(0, 0, 0, 0)
        thumbs_row.setSpacing(8)
        thumbs_row.setAlignment(Qt.AlignmentFlag.AlignLeft)

        if patterns:
            for i, path in enumerate(patterns, start=1):
                thumb = _PatternThumbnail(path, i)
                thumb.deleted.connect(self._on_pattern_deleted)
                thumbs_row.addWidget(thumb)
        else:
            lbl_empty = QLabel("Sin imágenes de referencia.\nUsá 'Nueva zona' para capturar.")
            lbl_empty.setStyleSheet("color: #555; font-size: 11px;")
            thumbs_row.addWidget(lbl_empty)

        self._root.addWidget(thumbs_w)

    def _on_pattern_deleted(self, path: Path) -> None:
        # Recargar patrones en el motor ORB (sin reiniciar)
        remaining = self._on_deleted(self._zone.id)
        # Refrescar la card visualmente
        self._render_content()


# ── Diálogo principal ─────────────────────────────────────────────────────────

class PatternsGalleryDialog(QDialog):
    """
    Galería de patrones de referencia.
    Eliminación en vivo: actualiza UI y motor ORB sin reiniciar la app.
    """

    def __init__(self, zones: list[ROIZone],
                 reload_callback: Callable[[str], int],   # (zone_id) → patrones en memoria
                 parent=None):
        super().__init__(parent)
        self._zones = zones
        self._reload_callback = reload_callback
        self.setWindowTitle("Patrones de referencia")
        self.setModal(True)
        self.setMinimumSize(800, 540)
        self.resize(840, 600)
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 14, 16, 12)
        root.setSpacing(10)

        lbl_title = QLabel("Patrones de referencia")
        lbl_title.setStyleSheet("color: #e0e0e0; font-size: 16px; font-weight: bold;")
        root.addWidget(lbl_title)

        total = sum(len(_get_zone_patterns(z.id)) for z in self._zones)
        lbl_sub = QLabel(
            f"{len(self._zones)} zona(s)  ·  {total} patrón(es) guardado(s)  "
            f"·  Eliminación activa sin reiniciar"
        )
        lbl_sub.setStyleSheet("color: #888; font-size: 11px;")
        root.addWidget(lbl_sub)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("color: #333;")
        root.addWidget(sep)

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
                "Usá el botón 'Nueva zona' para comenzar."
            )
            lbl_empty.setStyleSheet("color: #555; font-size: 13px;")
            lbl_empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
            vbox.addWidget(lbl_empty)
        else:
            for zone in self._zones:
                card = _PatternCard(zone, self._reload_callback)
                vbox.addWidget(card)

        vbox.addStretch()
        scroll.setWidget(container)
        root.addWidget(scroll)

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
