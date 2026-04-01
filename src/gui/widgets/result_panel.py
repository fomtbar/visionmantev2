from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QFont
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QTableWidget, QTableWidgetItem, QHeaderView,
    QPushButton, QFrame,
)

from src.core.result_model import InspectionResult, ResultHistory


class ResultPanel(QWidget):
    """Panel lateral con indicador OK/NG, detalle por pieza e historial."""

    def __init__(self, history: ResultHistory, parent=None):
        super().__init__(parent)
        self._history = history
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # ── Indicador global ─────────────────────────────────────────────────
        self._status_label = QLabel("---")
        self._status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._status_label.setFont(QFont("Arial", 42, QFont.Weight.Bold))
        self._status_label.setFixedHeight(100)
        self._status_label.setStyleSheet(
            "background-color: #2c2c3e; border-radius: 8px; color: #888;"
        )
        layout.addWidget(self._status_label)

        # ── Stats rápidas ─────────────────────────────────────────────────────
        stats_row = QHBoxLayout()
        self._lbl_total = self._stat_label("Total: 0")
        self._lbl_ok = self._stat_label("OK: 0", "#00dc50")
        self._lbl_ng = self._stat_label("NG: 0", "#dc2828")
        self._lbl_rate = self._stat_label("Tasa: -")
        for lbl in (self._lbl_total, self._lbl_ok, self._lbl_ng, self._lbl_rate):
            stats_row.addWidget(lbl)
        layout.addLayout(stats_row)

        # ── Tiempo de ciclo ───────────────────────────────────────────────────
        self._lbl_time = QLabel("Ciclo: --ms  |  Algoritmo: --")
        self._lbl_time.setStyleSheet("color: #aaa; font-size: 11px;")
        layout.addWidget(self._lbl_time)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("color: #444;")
        layout.addWidget(sep)

        # ── Tabla de piezas ───────────────────────────────────────────────────
        lbl = QLabel("Detalle por zona:")
        lbl.setStyleSheet("color: #ccc; font-size: 12px;")
        layout.addWidget(lbl)

        self._pieces_table = QTableWidget(0, 3)
        self._pieces_table.setHorizontalHeaderLabels(["Zona", "Estado", "Confianza"])
        self._pieces_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._pieces_table.setMaximumHeight(180)
        self._pieces_table.setStyleSheet(
            "QTableWidget { background: #1e1e2e; color: #ddd; gridline-color: #444; }"
            "QHeaderView::section { background: #2c2c3e; color: #aaa; }"
        )
        layout.addWidget(self._pieces_table)

        sep2 = QFrame()
        sep2.setFrameShape(QFrame.Shape.HLine)
        sep2.setStyleSheet("color: #444;")
        layout.addWidget(sep2)

        # ── Historial ─────────────────────────────────────────────────────────
        hist_header = QHBoxLayout()
        lbl2 = QLabel("Historial:")
        lbl2.setStyleSheet("color: #ccc; font-size: 12px;")
        hist_header.addWidget(lbl2)
        hist_header.addStretch()
        btn_clear = QPushButton("Limpiar")
        btn_clear.setFixedWidth(70)
        btn_clear.clicked.connect(self._clear_history)
        hist_header.addWidget(btn_clear)
        layout.addLayout(hist_header)

        self._history_table = QTableWidget(0, 3)
        self._history_table.setHorizontalHeaderLabels(["Hora", "Estado", "ms"])
        self._history_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._history_table.setStyleSheet(
            "QTableWidget { background: #1e1e2e; color: #ddd; gridline-color: #444; }"
            "QHeaderView::section { background: #2c2c3e; color: #aaa; }"
        )
        layout.addWidget(self._history_table)

    # ── Update ────────────────────────────────────────────────────────────────

    def update_result(self, result: InspectionResult) -> None:
        # Indicador global
        if result.is_ok:
            self._status_label.setText("OK")
            self._status_label.setStyleSheet(
                "background-color: #00dc50; border-radius: 8px; color: white;"
            )
        else:
            self._status_label.setText("NG")
            self._status_label.setStyleSheet(
                "background-color: #dc2828; border-radius: 8px; color: white;"
            )

        self._lbl_time.setText(
            f"Ciclo: {result.inference_time_ms:.0f}ms  |  {result.timestamp.strftime('%H:%M:%S')}"
        )

        # Tabla de piezas
        self._pieces_table.setRowCount(len(result.pieces))
        for row, piece in enumerate(result.pieces):
            color = QColor("#00dc50") if piece.is_ok else QColor("#dc2828")
            items = [
                QTableWidgetItem(piece.zone_id),
                QTableWidgetItem(piece.status),
                QTableWidgetItem(f"{piece.confidence:.0%}"),
            ]
            for col, item in enumerate(items):
                item.setForeground(color)
                item.setFlags(Qt.ItemFlag.ItemIsEnabled)
                self._pieces_table.setItem(row, col, item)

        # Stats
        h = self._history
        self._lbl_total.setText(f"Total: {h.total}")
        self._lbl_ok.setText(f"OK: {h.ok_count}")
        self._lbl_ng.setText(f"NG: {h.ng_count}")
        self._lbl_rate.setText(f"Tasa: {h.ok_rate:.1f}%")

        # Historial (primera fila)
        self._history_table.insertRow(0)
        hora_item = QTableWidgetItem(result.timestamp.strftime("%H:%M:%S"))
        estado_item = QTableWidgetItem(result.global_status)
        ms_item = QTableWidgetItem(f"{result.inference_time_ms:.0f}")
        status_color = QColor("#00dc50") if result.is_ok else QColor("#dc2828")
        for item in (hora_item, estado_item, ms_item):
            item.setFlags(Qt.ItemFlag.ItemIsEnabled)
            item.setForeground(status_color)
        self._history_table.setItem(0, 0, hora_item)
        self._history_table.setItem(0, 1, estado_item)
        self._history_table.setItem(0, 2, ms_item)

        # Limitar historial visual a 100 filas
        while self._history_table.rowCount() > 100:
            self._history_table.removeRow(self._history_table.rowCount() - 1)

    def _clear_history(self) -> None:
        self._history.clear()
        self._history_table.setRowCount(0)
        self._status_label.setText("---")
        self._status_label.setStyleSheet(
            "background-color: #2c2c3e; border-radius: 8px; color: #888;"
        )
        self._lbl_total.setText("Total: 0")
        self._lbl_ok.setText("OK: 0")
        self._lbl_ng.setText("NG: 0")
        self._lbl_rate.setText("Tasa: -")

    def _stat_label(self, text: str, color: str = "#aaa") -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet(f"color: {color}; font-size: 12px; font-weight: bold;")
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        return lbl
