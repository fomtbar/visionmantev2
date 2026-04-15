from __future__ import annotations

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout,
    QGroupBox, QLabel, QLineEdit, QComboBox, QSpinBox,
    QPushButton, QDialogButtonBox, QMessageBox,
)
from loguru import logger

from src.core.config_manager import PLCConfig
from src.plc.plc_factory import create_plc
from src.plc.base import AbstractPLC


# ── Hilo de conexión (no bloquear la GUI) ─────────────────────────────────────

class _ConnectWorker(QThread):
    done = pyqtSignal(bool, str)   # (éxito, mensaje)

    def __init__(self, plc: AbstractPLC):
        super().__init__()
        self._plc = plc

    def run(self):
        try:
            ok = self._plc.connect()
            msg = "Conectado" if ok else "No se pudo conectar"
            self.done.emit(ok, msg)
        except Exception as e:
            self.done.emit(False, str(e))


# ── Diálogo principal ─────────────────────────────────────────────────────────

class PLCTestDialog(QDialog):
    """Ventana de prueba para conectar al PLC y togglear una marca (bit) a voluntad."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Prueba de comunicación PLC")
        self.setMinimumWidth(460)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowType.WindowContextHelpButtonHint)

        self._plc: AbstractPLC | None = None
        self._bit_state: bool = False
        self._worker: _ConnectWorker | None = None

        self._build_ui()

    # ── Construcción UI ───────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # ── Grupo conexión ────────────────────────────────────────────────────
        conn_group = QGroupBox("Conexión")
        conn_form = QFormLayout(conn_group)
        conn_form.setSpacing(6)

        self._cb_brand = QComboBox()
        self._cb_brand.addItems(["mitsubishi", "siemens", "mock"])
        self._cb_brand.currentTextChanged.connect(self._on_brand_changed)
        conn_form.addRow("Marca:", self._cb_brand)

        self._ed_ip = QLineEdit("192.168.0.1")
        self._ed_ip.setPlaceholderText("ej: 192.168.0.10")
        conn_form.addRow("IP:", self._ed_ip)

        self._sb_port = QSpinBox()
        self._sb_port.setRange(1, 65535)
        self._sb_port.setValue(1025)  # MC Protocol default Mitsubishi Q Serie
        conn_form.addRow("Puerto:", self._sb_port)

        self._cb_commtype = QComboBox()
        self._cb_commtype.addItems(["binary", "ascii"])
        self._cb_commtype.setToolTip(
            "binary: trama binaria (default pymcprotocol)\n"
            "ascii: trama ASCII — usar si read/write dan timeout tras conectar"
        )
        conn_form.addRow("Modo trama:", self._cb_commtype)

        btn_row = QHBoxLayout()
        self._btn_connect = QPushButton("Conectar")
        self._btn_connect.setStyleSheet(
            "QPushButton { background-color: #1e6fe0; color: white; font-weight: bold; "
            "border-radius: 4px; padding: 6px 18px; }"
            "QPushButton:hover { background-color: #2a80f5; }"
            "QPushButton:disabled { background-color: #444; color: #777; }"
        )
        self._btn_connect.clicked.connect(self._on_connect)

        self._btn_disconnect = QPushButton("Desconectar")
        self._btn_disconnect.setEnabled(False)
        self._btn_disconnect.setStyleSheet(
            "QPushButton { background-color: #8b2020; color: white; font-weight: bold; "
            "border-radius: 4px; padding: 6px 18px; }"
            "QPushButton:hover { background-color: #c03030; }"
            "QPushButton:disabled { background-color: #444; color: #777; }"
        )
        self._btn_disconnect.clicked.connect(self._on_disconnect)

        btn_row.addWidget(self._btn_connect)
        btn_row.addWidget(self._btn_disconnect)
        conn_form.addRow(btn_row)

        # Indicador de estado
        self._lbl_status = QLabel("Sin conexión")
        self._lbl_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._lbl_status.setStyleSheet(
            "background-color: #3a1010; color: #ff6b6b; border-radius: 4px; "
            "padding: 4px 10px; font-weight: bold;"
        )
        conn_form.addRow("Estado:", self._lbl_status)

        layout.addWidget(conn_group)

        # ── Grupo testigo de marca ─────────────────────────────────────────────
        bit_group = QGroupBox("Testigo — escritura de marca")
        bit_layout = QVBoxLayout(bit_group)
        bit_layout.setSpacing(8)

        addr_row = QHBoxLayout()
        lbl_addr = QLabel("Dirección:")
        self._ed_address = QLineEdit("M1200")
        self._ed_address.setPlaceholderText("ej: M1200 / DB1.DBX0.0")
        self._ed_address.setMaximumWidth(160)
        addr_row.addWidget(lbl_addr)
        addr_row.addWidget(self._ed_address)
        addr_row.addStretch()
        bit_layout.addLayout(addr_row)

        # Testigo visual grande
        self._lbl_bit = QLabel("●  APAGADO")
        self._lbl_bit.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._lbl_bit.setStyleSheet(
            "font-size: 28px; font-weight: bold; color: #555; "
            "background-color: #222; border-radius: 8px; padding: 14px;"
        )
        bit_layout.addWidget(self._lbl_bit)

        btn_bit_row = QHBoxLayout()

        self._btn_toggle = QPushButton("Encender / Apagar  (toggle)")
        self._btn_toggle.setFixedHeight(50)
        self._btn_toggle.setEnabled(False)
        self._btn_toggle.setStyleSheet(
            "QPushButton { background-color: #2c4a2c; color: #00dc50; font-size: 15px; "
            "font-weight: bold; border-radius: 6px; border: 2px solid #00dc50; }"
            "QPushButton:hover { background-color: #3a6a3a; }"
            "QPushButton:disabled { background-color: #2c2c2c; color: #555; border-color: #444; }"
        )
        self._btn_toggle.clicked.connect(self._on_toggle)
        btn_bit_row.addWidget(self._btn_toggle, stretch=3)

        self._btn_read = QPushButton("Leer")
        self._btn_read.setFixedHeight(50)
        self._btn_read.setEnabled(False)
        self._btn_read.setStyleSheet(
            "QPushButton { background-color: #2c3a4a; color: #7ec8e3; font-size: 13px; "
            "border-radius: 6px; border: 1px solid #7ec8e3; }"
            "QPushButton:hover { background-color: #3a4a5a; }"
            "QPushButton:disabled { background-color: #2c2c2c; color: #555; border-color: #444; }"
        )
        self._btn_read.clicked.connect(self._on_read)
        btn_bit_row.addWidget(self._btn_read, stretch=1)

        bit_layout.addLayout(btn_bit_row)
        layout.addWidget(bit_group)

        # ── Botón Cerrar ──────────────────────────────────────────────────────
        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    # ── Slots ─────────────────────────────────────────────────────────────────

    def _on_brand_changed(self, brand: str) -> None:
        """Actualiza puerto default y placeholder de dirección según marca."""
        if brand == "mitsubishi":
            self._sb_port.setValue(1025)
            self._ed_address.setPlaceholderText("ej: M1200, D100")
        elif brand == "siemens":
            self._sb_port.setValue(102)
            self._ed_address.setPlaceholderText("ej: DB1.DBX0.0")
        else:  # mock
            self._sb_port.setValue(502)
            self._ed_address.setPlaceholderText("cualquier texto")

    def _on_connect(self) -> None:
        ip = self._ed_ip.text().strip()
        port = self._sb_port.value()
        brand = self._cb_brand.currentText()

        if not ip:
            QMessageBox.warning(self, "IP requerida", "Ingresá la IP del PLC.")
            return

        cfg = PLCConfig(
            enabled=True,
            brand=brand,
            ip=ip,
            port=port,
            commtype=self._cb_commtype.currentText(),
            trigger_address="M0",
            result_ok_address="M100",
            result_ng_address="M101",
        )

        self._plc = create_plc(cfg)
        self._btn_connect.setEnabled(False)
        self._lbl_status.setText("Conectando…")
        self._lbl_status.setStyleSheet(
            "background-color: #3a3a10; color: #ffd700; border-radius: 4px; "
            "padding: 4px 10px; font-weight: bold;"
        )

        self._worker = _ConnectWorker(self._plc)
        self._worker.done.connect(self._on_connect_done)
        self._worker.start()

    def _on_connect_done(self, ok: bool, msg: str) -> None:
        if ok:
            self._lbl_status.setText(f"Conectado  ({self._cb_brand.currentText().upper()}  {self._ed_ip.text()})")
            self._lbl_status.setStyleSheet(
                "background-color: #103a10; color: #00dc50; border-radius: 4px; "
                "padding: 4px 10px; font-weight: bold;"
            )
            self._btn_disconnect.setEnabled(True)
            self._btn_toggle.setEnabled(True)
            self._btn_read.setEnabled(True)
            self._ed_ip.setEnabled(False)
            self._sb_port.setEnabled(False)
            self._cb_brand.setEnabled(False)
            self._cb_commtype.setEnabled(False)
            # Leer estado inicial
            self._on_read()
        else:
            self._lbl_status.setText(f"Error: {msg}")
            self._lbl_status.setStyleSheet(
                "background-color: #3a1010; color: #ff6b6b; border-radius: 4px; "
                "padding: 4px 10px; font-weight: bold;"
            )
            self._btn_connect.setEnabled(True)
            self._plc = None
            logger.error(f"PLCTestDialog: error de conexión — {msg}")

    def _on_disconnect(self) -> None:
        if self._plc:
            self._plc.disconnect()
            self._plc = None

        self._lbl_status.setText("Sin conexión")
        self._lbl_status.setStyleSheet(
            "background-color: #3a1010; color: #ff6b6b; border-radius: 4px; "
            "padding: 4px 10px; font-weight: bold;"
        )
        self._btn_connect.setEnabled(True)
        self._btn_disconnect.setEnabled(False)
        self._btn_toggle.setEnabled(False)
        self._btn_read.setEnabled(False)
        self._ed_ip.setEnabled(True)
        self._sb_port.setEnabled(True)
        self._cb_brand.setEnabled(True)
        self._cb_commtype.setEnabled(True)
        self._bit_state = False
        self._update_bit_indicator(False)

    def _on_toggle(self) -> None:
        if not self._plc:
            return
        address = self._ed_address.text().strip()
        if not address:
            QMessageBox.warning(self, "Dirección", "Ingresá la dirección del bit a escribir.")
            return
        try:
            new_state = not self._bit_state
            self._plc.write_bit(address, new_state)
            self._bit_state = new_state
            self._update_bit_indicator(new_state)
        except Exception as e:
            QMessageBox.critical(self, "Error de escritura", str(e))
            logger.error(f"PLCTestDialog write_bit error: {e}")

    def _on_read(self) -> None:
        if not self._plc:
            return
        address = self._ed_address.text().strip()
        if not address:
            return
        try:
            val = self._plc.read_bit(address)
            self._bit_state = val
            self._update_bit_indicator(val)
        except Exception as e:
            logger.warning(f"PLCTestDialog read_bit error: {e}")

    def _update_bit_indicator(self, state: bool) -> None:
        if state:
            self._lbl_bit.setText("●  ENCENDIDO")
            self._lbl_bit.setStyleSheet(
                "font-size: 28px; font-weight: bold; color: #00dc50; "
                "background-color: #0a2a0a; border-radius: 8px; padding: 14px;"
            )
        else:
            self._lbl_bit.setText("●  APAGADO")
            self._lbl_bit.setStyleSheet(
                "font-size: 28px; font-weight: bold; color: #555; "
                "background-color: #222; border-radius: 8px; padding: 14px;"
            )

    def reject(self) -> None:
        """Al cerrar desconectamos limpiamente si está conectado."""
        if self._plc and self._plc.is_connected():
            self._plc.disconnect()
        super().reject()
