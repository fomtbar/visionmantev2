from __future__ import annotations

import cv2
import numpy as np
from PyQt6.QtCore import Qt, QRect, QPoint
from PyQt6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QFont, QCursor
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QMessageBox,
)


class ReferenceSelectionDialog(QDialog):
    """
    Muestra el frame actual congelado y permite al usuario
    dibujar un rectángulo que define simultáneamente:
      - La zona ROI (dónde mirar)
      - La imagen de referencia OK (cómo se ve una pieza buena)
    """

    def __init__(
        self,
        frame: np.ndarray,
        existing_zone_id: str | None = None,
        instruction: str | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self._frame = frame.copy()
        self._existing_zone_id = existing_zone_id
        self._custom_instruction = instruction

        self.selected_rect: QRect | None = None      # rect en coordenadas del FRAME
        self.cropped_image: np.ndarray | None = None  # recorte de la referencia

        self._drawing = False
        self._start: QPoint | None = None
        self._current_rect: QRect | None = None

        self.setWindowTitle("Definir zona de referencia OK")
        self.setModal(True)
        self._build_ui()
        self._render()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        # Instrucción
        text = self._custom_instruction or "Dibuja un rectángulo alrededor de la pieza en estado OK"
        lbl = QLabel(text)
        lbl.setStyleSheet("color: #ffd700; font-size: 13px; font-weight: bold;")
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(lbl)

        lbl2 = QLabel("Haz clic y arrastra · Suelta para confirmar la zona")
        lbl2.setStyleSheet("color: #aaa; font-size: 11px;")
        lbl2.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(lbl2)

        # Canvas con el frame congelado
        self._canvas = QLabel()
        self._canvas.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._canvas.setCursor(QCursor(Qt.CursorShape.CrossCursor))
        self._canvas.setMinimumSize(800, 500)
        self._canvas.setStyleSheet("background-color: #111;")
        self._canvas.mousePressEvent = self._on_press
        self._canvas.mouseMoveEvent = self._on_move
        self._canvas.mouseReleaseEvent = self._on_release
        layout.addWidget(self._canvas)

        # Info del recorte seleccionado
        self._lbl_info = QLabel("Sin selección")
        self._lbl_info.setStyleSheet("color: #888; font-size: 11px;")
        self._lbl_info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._lbl_info)

        # Botones
        btn_row = QHBoxLayout()
        self._btn_confirm = QPushButton("✔  Confirmar zona y capturar referencia")
        self._btn_confirm.setFixedHeight(44)
        self._btn_confirm.setEnabled(False)
        self._btn_confirm.setStyleSheet(
            "QPushButton { background-color: #00a040; color: white; font-size: 14px; "
            "font-weight: bold; border-radius: 6px; }"
            "QPushButton:hover { background-color: #00c050; }"
            "QPushButton:disabled { background-color: #333; color: #666; }"
        )
        self._btn_confirm.clicked.connect(self._confirm)

        btn_cancel = QPushButton("Cancelar")
        btn_cancel.setFixedHeight(44)
        btn_cancel.setFixedWidth(100)
        btn_cancel.setStyleSheet(
            "QPushButton { background-color: #444; color: #ccc; border-radius: 6px; }"
            "QPushButton:hover { background-color: #555; }"
        )
        btn_cancel.clicked.connect(self.reject)

        btn_row.addWidget(self._btn_confirm)
        btn_row.addWidget(btn_cancel)
        layout.addLayout(btn_row)

    # ── Mouse events ──────────────────────────────────────────────────────────

    def _on_press(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._drawing = True
            self._start = event.pos()
            self._current_rect = None

    def _on_move(self, event) -> None:
        if self._drawing and self._start:
            self._current_rect = QRect(self._start, event.pos()).normalized()
            self._render()

    def _on_release(self, event) -> None:
        if self._drawing:
            self._drawing = False
            if self._current_rect and self._current_rect.width() > 15 and self._current_rect.height() > 15:
                self._lbl_info.setText(
                    f"Zona seleccionada: {self._current_rect.width()}×{self._current_rect.height()} px (en pantalla)"
                )
                self._btn_confirm.setEnabled(True)
            else:
                self._current_rect = None
                self._lbl_info.setText("Rectángulo muy pequeño, intenta de nuevo")
                self._btn_confirm.setEnabled(False)
            self._render()

    # ── Render ────────────────────────────────────────────────────────────────

    def _render(self) -> None:
        frame = self._frame.copy()
        h, w = frame.shape[:2]

        # Oscurecer ligeramente para destacar la selección
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.85, overlay, 0.15, 0)

        pixmap = self._array_to_pixmap(frame)

        if self._current_rect:
            painter = QPainter(pixmap)
            # Sombra exterior
            pen_shadow = QPen(QColor(0, 0, 0, 120), 4)
            painter.setPen(pen_shadow)
            painter.drawRect(self._current_rect.adjusted(2, 2, 2, 2))
            # Rectángulo principal
            pen = QPen(QColor(255, 215, 0), 2, Qt.PenStyle.SolidLine)
            painter.setPen(pen)
            painter.drawRect(self._current_rect)
            # Relleno semitransparente
            painter.fillRect(
                self._current_rect,
                QColor(255, 215, 0, 30),
            )
            # Dimensiones
            painter.setPen(QPen(QColor(255, 215, 0)))
            font = QFont("Arial", 10)
            font.setBold(True)
            painter.setFont(font)
            label = f"{self._current_rect.width()}×{self._current_rect.height()}"
            painter.drawText(
                self._current_rect.x() + 4,
                self._current_rect.y() - 6,
                label,
            )
            painter.end()

        self._canvas.setPixmap(pixmap)

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

    # ── Confirm ───────────────────────────────────────────────────────────────

    def _confirm(self) -> None:
        if not self._current_rect:
            return

        frame_rect = self._widget_rect_to_frame(self._current_rect)
        if frame_rect.width() < 10 or frame_rect.height() < 10:
            QMessageBox.warning(self, "Zona inválida", "La zona seleccionada es demasiado pequeña.")
            return

        x, y = frame_rect.x(), frame_rect.y()
        w, h = frame_rect.width(), frame_rect.height()
        self.cropped_image = self._frame[y:y + h, x:x + w].copy()
        self.selected_rect = frame_rect
        self.accept()

    def _widget_rect_to_frame(self, rect: QRect) -> QRect:
        fh, fw = self._frame.shape[:2]
        cw, ch = self._canvas.width(), self._canvas.height()
        scale = min(cw / fw, ch / fh)
        scaled_w = fw * scale
        scaled_h = fh * scale
        offset_x = (cw - scaled_w) / 2
        offset_y = (ch - scaled_h) / 2

        x = int((rect.x() - offset_x) / scale)
        y = int((rect.y() - offset_y) / scale)
        w = int(rect.width() / scale)
        h = int(rect.height() / scale)
        fh_orig, fw_orig = self._frame.shape[:2]
        return QRect(
            max(0, x), max(0, y),
            min(w, fw_orig - max(0, x)),
            min(h, fh_orig - max(0, y)),
        )
