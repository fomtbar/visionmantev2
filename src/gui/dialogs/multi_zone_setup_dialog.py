from __future__ import annotations

from dataclasses import dataclass, field
import cv2
import numpy as np
from PyQt6.QtCore import Qt, QRect, QPoint
from PyQt6.QtGui import (
    QImage, QPixmap, QPainter, QPen, QColor, QFont, QBrush,
)
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QScrollArea, QWidget, QFrame, QLineEdit, QSizePolicy,
)

from src.gui.dialogs.reference_dialog import ReferenceSelectionDialog

# Colores para cada zona (cíclicos)
_ZONE_COLORS = [
    QColor(255, 215,   0),   # amarillo
    QColor(  0, 200, 255),   # cyan
    QColor(255,  80,  80),   # rojo
    QColor( 80, 255, 120),   # verde
    QColor(220,  80, 255),   # violeta
    QColor(255, 160,   0),   # naranja
]


@dataclass
class ZoneDraft:
    """
    Zona marcada antes de confirmar.

    Modo directo (search_window=None):
        rect = zona ROI = área del patrón (comportamiento clásico)

    Modo ventana de búsqueda (search_window=rect):
        rect         = ventana de búsqueda grande
        search_window = igual a rect (marca que usa windowed matching)
        image        = patrón más pequeño seleccionado dentro de la ventana
    """
    index: int
    name: str
    rect: QRect           # coordenadas del FRAME
    widget_rect: QRect    # coordenadas del canvas (para dibujado)
    image: np.ndarray     # recorte de referencia (o patrón sub-seleccionado)
    search_window: QRect | None = field(default=None)  # None = modo directo

    @property
    def color(self) -> QColor:
        return _ZONE_COLORS[(self.index - 1) % len(_ZONE_COLORS)]


class MultiZoneSetupDialog(QDialog):
    """
    Flujo de setup unificado:
    1. Muestra frame congelado.
    2. El usuario dibuja N rectángulos (uno por pieza a detectar).
    3. Cada rectángulo = zona de búsqueda + imagen de referencia OK.
    4. Al confirmar devuelve la lista de ZoneDraft.

    Acceder al resultado: dlg.zones  (list[ZoneDraft]) tras accept().
    """

    def __init__(self, frame: np.ndarray, parent=None):
        super().__init__(parent)
        self._frame = frame.copy()
        self._zones: list[ZoneDraft] = []
        self._drawing = False
        self._start: QPoint | None = None
        self._live_rect: QRect | None = None
        self._windowed_mode = False   # toggle: False=directo  True=ventana+patrón

        self.setWindowTitle("Configurar piezas a inspeccionar")
        self.setModal(True)
        self.setMinimumSize(1060, 620)
        self._build_ui()
        self._render()

    # ── Resultado ─────────────────────────────────────────────────────────────

    @property
    def zones(self) -> list[ZoneDraft]:
        return self._zones

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Panel izquierdo: canvas ───────────────────────────────────────────
        left = QVBoxLayout()
        left.setContentsMargins(10, 10, 6, 10)
        left.setSpacing(6)

        inst_row = QHBoxLayout()

        self._lbl_inst = QLabel("Dibujá un rectángulo por pieza · clic y arrastrá")
        self._lbl_inst.setStyleSheet(
            "color: #ffd700; font-size: 13px; font-weight: bold;"
            " background: #1a1a2e; padding: 4px;"
        )
        self._lbl_inst.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        inst_row.addWidget(self._lbl_inst, stretch=1)

        self._btn_mode = QPushButton("Modo: Zona directa")
        self._btn_mode.setCheckable(True)
        self._btn_mode.setFixedHeight(30)
        self._btn_mode.setStyleSheet(
            "QPushButton { background-color: #2c2c3e; color: #aaa; border: 1px solid #555;"
            " border-radius: 4px; padding: 2px 10px; font-size: 11px; }"
            "QPushButton:checked { background-color: #003355; color: #7ec8e3;"
            " border-color: #7ec8e3; }"
            "QPushButton:hover { background-color: #3a3a50; }"
        )
        self._btn_mode.toggled.connect(self._on_mode_toggled)
        inst_row.addWidget(self._btn_mode)

        left.addLayout(inst_row)

        self._canvas = QLabel()
        self._canvas.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._canvas.setStyleSheet("background-color: #111;")
        self._canvas.setCursor(Qt.CursorShape.CrossCursor)
        self._canvas.mousePressEvent   = self._on_press
        self._canvas.mouseMoveEvent    = self._on_move
        self._canvas.mouseReleaseEvent = self._on_release
        left.addWidget(self._canvas, stretch=1)

        root.addLayout(left, stretch=3)

        # ── Panel derecho: lista de zonas + acciones ──────────────────────────
        right_w = QFrame()
        right_w.setFixedWidth(270)
        right_w.setStyleSheet("background-color: #12121e; border-left: 1px solid #333;")
        right = QVBoxLayout(right_w)
        right.setContentsMargins(12, 12, 12, 12)
        right.setSpacing(10)

        self._lbl_count = QLabel("0 piezas marcadas")
        self._lbl_count.setStyleSheet("color: #888; font-size: 12px;")
        right.addWidget(self._lbl_count)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("color: #333;")
        right.addWidget(sep)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet(
            "QScrollArea { border: none; background: transparent; }"
            "QScrollBar:vertical { background: #1a1a2e; width: 8px; }"
            "QScrollBar::handle:vertical { background: #444; border-radius: 4px; }"
        )
        self._zone_list_w = QWidget()
        self._zone_list_w.setStyleSheet("background: transparent;")
        self._zone_list = QVBoxLayout(self._zone_list_w)
        self._zone_list.setContentsMargins(0, 0, 0, 0)
        self._zone_list.setSpacing(6)
        self._zone_list.addStretch()
        scroll.setWidget(self._zone_list_w)
        right.addWidget(scroll, stretch=1)

        sep2 = QFrame()
        sep2.setFrameShape(QFrame.Shape.HLine)
        sep2.setStyleSheet("color: #333;")
        right.addWidget(sep2)

        self._btn_confirm = QPushButton("Confirmar (0 piezas)")
        self._btn_confirm.setFixedHeight(44)
        self._btn_confirm.setEnabled(False)
        self._btn_confirm.setStyleSheet(
            "QPushButton { background-color: #00a040; color: white; font-size: 14px;"
            " font-weight: bold; border-radius: 6px; }"
            "QPushButton:hover { background-color: #00c050; }"
            "QPushButton:disabled { background-color: #333; color: #555; }"
        )
        self._btn_confirm.clicked.connect(self.accept)
        right.addWidget(self._btn_confirm)

        btn_cancel = QPushButton("Cancelar")
        btn_cancel.setFixedHeight(34)
        btn_cancel.setStyleSheet(
            "QPushButton { background-color: #2c2c3e; color: #aaa; border-radius: 4px; }"
            "QPushButton:hover { background-color: #3a3a50; }"
        )
        btn_cancel.clicked.connect(self.reject)
        right.addWidget(btn_cancel)

        root.addWidget(right_w)

    # ── Mouse ─────────────────────────────────────────────────────────────────

    def _on_press(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._drawing = True
            self._start = event.pos()
            self._live_rect = None

    def _on_move(self, event) -> None:
        if self._drawing and self._start:
            self._live_rect = QRect(self._start, event.pos()).normalized()
            self._render()

    def _on_release(self, event) -> None:
        if not self._drawing:
            return
        self._drawing = False
        r = self._live_rect
        self._live_rect = None
        if r and r.width() > 15 and r.height() > 15:
            self._add_zone(r)
        self._render()

    # ── Modo toggle ───────────────────────────────────────────────────────────

    def _on_mode_toggled(self, checked: bool) -> None:
        self._windowed_mode = checked
        if checked:
            self._btn_mode.setText("Modo: Ventana de búsqueda")
            self._lbl_inst.setText(
                "1º dibujá la ventana de búsqueda (grande)  →  "
                "2º seleccioná el patrón dentro de ella"
            )
            self._lbl_inst.setStyleSheet(
                "color: #7ec8e3; font-size: 12px; font-weight: bold;"
                " background: #1a1a2e; padding: 4px;"
            )
        else:
            self._btn_mode.setText("Modo: Zona directa")
            self._lbl_inst.setText("Dibujá un rectángulo por pieza · clic y arrastrá")
            self._lbl_inst.setStyleSheet(
                "color: #ffd700; font-size: 13px; font-weight: bold;"
                " background: #1a1a2e; padding: 4px;"
            )

    # ── Zona ─────────────────────────────────────────────────────────────────

    def _add_zone(self, widget_rect: QRect) -> None:
        frame_rect = self._widget_rect_to_frame(widget_rect)
        if frame_rect.width() < 10 or frame_rect.height() < 10:
            return

        x, y, w, h = frame_rect.x(), frame_rect.y(), frame_rect.width(), frame_rect.height()
        idx = len(self._zones) + 1
        name = f"pieza_{idx}"

        if self._windowed_mode:
            # Modo ventana: pedir al usuario que marque el patrón dentro de la ventana
            window_crop = self._frame[y:y + h, x:x + w].copy()
            dlg = ReferenceSelectionDialog(
                window_crop,
                instruction="Marcá el PATRÓN a buscar dentro de la ventana (puede ser más pequeño)",
                parent=self,
            )
            dlg.setWindowTitle(f"Seleccionar patrón — pieza {idx}")
            if not dlg.exec() or dlg.cropped_image is None:
                return   # usuario canceló → no agregar zona
            pattern_image = dlg.cropped_image
            search_window = frame_rect
        else:
            pattern_image = self._frame[y:y + h, x:x + w].copy()
            search_window = None

        zone = ZoneDraft(
            index=idx,
            name=name,
            rect=frame_rect,
            widget_rect=widget_rect,
            image=pattern_image,
            search_window=search_window,
        )
        self._zones.append(zone)
        self._add_zone_row(zone)
        self._update_count()

    def _remove_zone(self, zone: ZoneDraft) -> None:
        self._zones.remove(zone)
        # Renumerar
        for i, z in enumerate(self._zones, start=1):
            z.index = i
        self._rebuild_zone_list()
        self._update_count()
        self._render()

    # ── Panel derecho ─────────────────────────────────────────────────────────

    def _add_zone_row(self, zone: ZoneDraft) -> None:
        row = self._make_zone_row(zone)
        # Insertar antes del stretch final
        self._zone_list.insertWidget(self._zone_list.count() - 1, row)

    def _rebuild_zone_list(self) -> None:
        while self._zone_list.count() > 1:   # dejar el stretch
            item = self._zone_list.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        for zone in self._zones:
            self._zone_list.insertWidget(self._zone_list.count() - 1, self._make_zone_row(zone))

    def _make_zone_row(self, zone: ZoneDraft) -> QFrame:
        c = zone.color
        row = QFrame()
        row.setStyleSheet(
            f"QFrame {{ background-color: #1e1e30; border: 1px solid "
            f"rgb({c.red()},{c.green()},{c.blue()}); border-radius: 5px; }}"
        )
        h = QHBoxLayout(row)
        h.setContentsMargins(8, 6, 8, 6)

        # Swatch de color
        swatch = QLabel()
        swatch.setFixedSize(14, 14)
        swatch.setStyleSheet(
            f"background-color: rgb({c.red()},{c.green()},{c.blue()});"
            " border-radius: 3px;"
        )
        h.addWidget(swatch)

        # Nombre editable
        ed = QLineEdit(zone.name)
        ed.setStyleSheet(
            "QLineEdit { background: transparent; color: #e0e0e0; border: none; font-size: 12px; }"
        )
        ed.textChanged.connect(lambda txt, z=zone: setattr(z, 'name', txt.strip() or z.name))
        h.addWidget(ed, stretch=1)

        # Dimensiones + indicador de modo
        fr = zone.rect
        mode_tag = " [V]" if zone.search_window is not None else ""
        lbl_size = QLabel(f"{fr.width()}×{fr.height()}{mode_tag}")
        lbl_size.setStyleSheet(
            "color: #7ec8e3; font-size: 10px;" if zone.search_window else "color: #666; font-size: 10px;"
        )
        h.addWidget(lbl_size)

        # Botón eliminar
        btn_del = QPushButton("✕")
        btn_del.setFixedSize(22, 22)
        btn_del.setStyleSheet(
            "QPushButton { background-color: #3a1a1a; color: #ff6b6b;"
            " border: none; border-radius: 3px; font-size: 11px; }"
            "QPushButton:hover { background-color: #5a2a2a; }"
        )
        btn_del.clicked.connect(lambda _, z=zone: self._remove_zone(z))
        h.addWidget(btn_del)

        return row

    def _update_count(self) -> None:
        n = len(self._zones)
        plural = "s" if n != 1 else ""
        self._lbl_count.setText(f"{n} pieza{plural} marcada{plural}")
        self._btn_confirm.setText(f"Confirmar ({n} pieza{plural})")
        self._btn_confirm.setEnabled(n > 0)

    # ── Render canvas ─────────────────────────────────────────────────────────

    def _letterbox_offset(self) -> tuple[int, int]:
        """
        Retorna el offset (ox, oy) del pixmap dentro del QLabel canvas.
        El pixmap se escala con KeepAspectRatio y queda centrado, dejando
        márgenes ("letterbox") en el eje que no llena. Los eventos de mouse
        vienen en coordenadas del QLabel; al dibujar sobre el pixmap hay que
        restar este offset para que los rects queden alineados con el cursor.
        """
        fh, fw = self._frame.shape[:2]
        cw, ch = self._canvas.width(), self._canvas.height()
        if cw <= 0 or ch <= 0:
            return 0, 0
        scale = min(cw / fw, ch / fh)
        return int((cw - fw * scale) / 2), int((ch - fh * scale) / 2)

    def _render(self) -> None:
        frame = self._frame.copy()
        # Oscurecer ligeramente
        overlay = np.zeros_like(frame)
        frame = cv2.addWeighted(frame, 0.88, overlay, 0.12, 0)

        pixmap = self._array_to_pixmap(frame)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # El pixmap NO tiene letterbox (está justamente del tamaño del área de imagen),
        # pero los coords almacenados en widget_rect son del QLabel (incluyen el offset).
        # Hay que restar el offset para dibujar en el sistema de coordenadas del pixmap.
        ox, oy = self._letterbox_offset()

        # Zonas confirmadas
        for zone in self._zones:
            pr = QRect(zone.widget_rect.x() - ox, zone.widget_rect.y() - oy,
                       zone.widget_rect.width(), zone.widget_rect.height())
            self._draw_zone_rect(
                painter, pr, zone.color, zone.index, zone.name,
                is_window=zone.search_window is not None,
            )

        # Rectángulo en curso
        if self._live_rect:
            next_idx = len(self._zones) + 1
            color = _ZONE_COLORS[(next_idx - 1) % len(_ZONE_COLORS)]
            pr = QRect(self._live_rect.x() - ox, self._live_rect.y() - oy,
                       self._live_rect.width(), self._live_rect.height())
            self._draw_zone_rect(painter, pr, color, next_idx, "", live=True)

        painter.end()
        self._canvas.setPixmap(pixmap)

    def _draw_zone_rect(
        self, painter: QPainter, rect: QRect,
        color: QColor, index: int, name: str,
        live: bool = False, is_window: bool = False,
    ) -> None:
        style = Qt.PenStyle.DashLine if live else Qt.PenStyle.SolidLine
        width = 2 if not is_window else 3
        pen = QPen(color, width, style)
        painter.setPen(pen)
        painter.drawRect(rect)

        # Ventana de búsqueda: relleno más tenue + borde interior punteado
        if is_window:
            painter.fillRect(rect, QBrush(QColor(color.red(), color.green(), color.blue(), 18)))
            inner_pen = QPen(color, 1, Qt.PenStyle.DotLine)
            painter.setPen(inner_pen)
            margin = 6
            painter.drawRect(rect.adjusted(margin, margin, -margin, -margin))
        else:
            painter.fillRect(rect, QBrush(QColor(color.red(), color.green(), color.blue(), 25 if live else 35)))

        # Etiqueta
        label = (name if name else f"pieza_{index}") + (" [ventana]" if is_window else "")
        painter.setPen(QPen(QColor(0, 0, 0, 160)))
        painter.setFont(QFont("Arial", 9, QFont.Weight.Bold))
        painter.drawText(rect.x() + 5, rect.y() + 16, label)
        painter.setPen(QPen(color))
        painter.drawText(rect.x() + 4, rect.y() + 15, label)

    # ── Coord conversion ──────────────────────────────────────────────────────

    def _widget_rect_to_frame(self, rect: QRect) -> QRect:
        fh, fw = self._frame.shape[:2]
        cw, ch = self._canvas.width(), self._canvas.height()
        scale = min(cw / fw, ch / fh)
        offset_x = (cw - fw * scale) / 2
        offset_y = (ch - fh * scale) / 2
        x = int((rect.x() - offset_x) / scale)
        y = int((rect.y() - offset_y) / scale)
        w = int(rect.width() / scale)
        h = int(rect.height() / scale)
        return QRect(
            max(0, x), max(0, y),
            min(w, fw - max(0, x)),
            min(h, fh - max(0, y)),
        )

    def _array_to_pixmap(self, frame: np.ndarray) -> QPixmap:
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, w, h, w * 3, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        return pixmap.scaled(
            self._canvas.width(), self._canvas.height(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._render()
