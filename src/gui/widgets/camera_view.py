from __future__ import annotations

import cv2
import numpy as np
from PyQt6.QtCore import Qt, QRect, QPoint, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QFont
from PyQt6.QtWidgets import QLabel

from src.core.result_model import InspectionResult, PieceResult
from src.core.config_manager import ROIZone


class CameraView(QLabel):
    """
    Widget de visualización de cámara en vivo.
    - Muestra el feed de cámara como QPixmap.
    - Dibuja ROIs y resultados de inspección como overlay.
    - Permite definir ROIs arrastrando con el ratón.
    """

    roi_defined = pyqtSignal(QRect)       # nueva ROI dibujada por el usuario

    COLOR_OK = QColor(0, 220, 80)
    COLOR_NG = QColor(220, 40, 40)
    COLOR_ROI = QColor(255, 200, 0)
    COLOR_ABSENT = QColor(150, 150, 150)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(640, 480)
        self.setStyleSheet("background-color: #1a1a2e;")

        self._current_frame: np.ndarray | None = None
        self._last_result: InspectionResult | None = None
        self._roi_zones: list[ROIZone] = []
        self._draw_rois_mode = False

        # Para dibujar ROI con mouse
        self._drawing = False
        self._roi_start: QPoint | None = None
        self._roi_current: QRect | None = None

    # ── Feed de cámara ────────────────────────────────────────────────────────

    def update_frame(self, frame: np.ndarray) -> None:
        self._current_frame = frame
        self._render()

    def update_result(self, result: InspectionResult) -> None:
        self._last_result = result
        self._render()

    def clear_result(self) -> None:
        self._last_result = None
        self._render()

    def set_roi_zones(self, zones: list[ROIZone]) -> None:
        self._roi_zones = zones
        self._render()

    def set_draw_mode(self, enabled: bool) -> None:
        self._draw_rois_mode = enabled
        self.setCursor(Qt.CursorShape.CrossCursor if enabled else Qt.CursorShape.ArrowCursor)

    # ── Render ────────────────────────────────────────────────────────────────

    def _render(self) -> None:
        if self._current_frame is None:
            return

        frame = self._current_frame.copy()
        h, w = frame.shape[:2]

        # Dibujar ROIs definidas
        for zone in self._roi_zones:
            cv2.rectangle(frame,
                          (zone.x, zone.y),
                          (zone.x + zone.w, zone.y + zone.h),
                          (255, 200, 0), 2)
            cv2.putText(frame, zone.id,
                        (zone.x + 4, zone.y + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 200, 0), 1)

        # Overlay de resultado
        if self._last_result:
            self._draw_result_overlay(frame)

        pixmap = self._array_to_pixmap(frame)

        # ROI en dibujo activo
        if self._drawing and self._roi_current and self._draw_rois_mode:
            painter = QPainter(pixmap)
            pen = QPen(self.COLOR_ROI, 2, Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.drawRect(self._roi_current)
            painter.end()

        self.setPixmap(pixmap)

    def _draw_result_overlay(self, frame: np.ndarray) -> None:
        result = self._last_result
        color = (0, 220, 80) if result.is_ok else (220, 40, 40)

        # Badge global OK/NG
        label = result.global_status
        cv2.rectangle(frame, (8, 8), (110, 52), color, -1)
        cv2.putText(frame, label, (16, 44),
                    cv2.FONT_HERSHEY_DUPLEX, 1.4, (255, 255, 255), 2)

        # Bounding boxes por pieza
        for piece in result.pieces:
            if piece.bounding_box:
                x, y, bw, bh = piece.bounding_box
                pc = (0, 220, 80) if piece.is_ok else (220, 40, 40)
                cv2.rectangle(frame, (x, y), (x + bw, y + bh), pc, 2)
                cv2.putText(frame,
                            f"{piece.zone_id}: {piece.status} {piece.confidence:.0%}",
                            (x, y - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, pc, 1)

        # Tiempo de inspección
        cv2.putText(frame,
                    f"{result.inference_time_ms:.0f}ms",
                    (frame.shape[1] - 90, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

    def _array_to_pixmap(self, frame: np.ndarray) -> QPixmap:
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, w, h, w * 3, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        # Escalar al widget preservando aspect ratio
        return pixmap.scaled(self.width(), self.height(),
                             Qt.AspectRatioMode.KeepAspectRatio,
                             Qt.TransformationMode.SmoothTransformation)

    # ── Mouse: dibujar ROI ────────────────────────────────────────────────────

    def _frame_to_widget_scale(self):
        if self._current_frame is None:
            return 1.0, 1.0
        fh, fw = self._current_frame.shape[:2]
        scale = min(self.width() / fw, self.height() / fh)
        return fw * scale, fh * scale, scale

    def mousePressEvent(self, event):
        if self._draw_rois_mode and event.button() == Qt.MouseButton.LeftButton:
            self._drawing = True
            self._roi_start = event.pos()
            self._roi_current = QRect(self._roi_start, self._roi_start)

    def mouseMoveEvent(self, event):
        if self._drawing and self._draw_rois_mode:
            self._roi_current = QRect(self._roi_start, event.pos()).normalized()
            self._render()

    def mouseReleaseEvent(self, event):
        if self._drawing and self._draw_rois_mode:
            self._drawing = False
            if self._roi_current and self._roi_current.width() > 10:
                # Convertir coordenadas widget → coordenadas frame
                roi_in_frame = self._widget_rect_to_frame(self._roi_current)
                self.roi_defined.emit(roi_in_frame)
            self._roi_current = None

    def _widget_rect_to_frame(self, rect: QRect) -> QRect:
        if self._current_frame is None:
            return rect
        fh, fw = self._current_frame.shape[:2]
        scale = min(self.width() / fw, self.height() / fh)
        scaled_w = fw * scale
        scaled_h = fh * scale
        offset_x = (self.width() - scaled_w) / 2
        offset_y = (self.height() - scaled_h) / 2

        x = int((rect.x() - offset_x) / scale)
        y = int((rect.y() - offset_y) / scale)
        w = int(rect.width() / scale)
        h = int(rect.height() / scale)
        return QRect(max(0, x), max(0, y), w, h)
