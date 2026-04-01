from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog, QFormLayout, QLineEdit, QComboBox,
    QSpinBox, QCheckBox, QDialogButtonBox, QGroupBox, QVBoxLayout,
)

from src.core.config_manager import PLCConfig


class PLCConfigDialog(QDialog):
    def __init__(self, config: PLCConfig, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Configuración PLC")
        self.setMinimumWidth(420)
        self._config = config
        self._build_ui()
        self._load_values()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        # ── Conexión ─────────────────────────────────────────────────────────
        conn_group = QGroupBox("Conexión")
        form = QFormLayout(conn_group)

        self._cb_enabled = QCheckBox("Habilitar comunicación PLC")
        form.addRow(self._cb_enabled)

        self._cb_brand = QComboBox()
        self._cb_brand.addItems(["mock", "siemens", "mitsubishi"])
        form.addRow("Marca:", self._cb_brand)

        self._ed_ip = QLineEdit()
        self._ed_ip.setPlaceholderText("192.168.0.1")
        form.addRow("IP:", self._ed_ip)

        self._sb_port = QSpinBox()
        self._sb_port.setRange(1, 65535)
        form.addRow("Puerto:", self._sb_port)

        layout.addWidget(conn_group)

        # ── Direcciones ───────────────────────────────────────────────────────
        addr_group = QGroupBox("Direcciones")
        addr_form = QFormLayout(addr_group)

        self._ed_trigger = QLineEdit()
        self._ed_trigger.setPlaceholderText("DB1.DBX0.0 / M0")
        addr_form.addRow("Trigger (entrada):", self._ed_trigger)

        self._ed_ok = QLineEdit()
        self._ed_ok.setPlaceholderText("DB1.DBX0.1 / M100")
        addr_form.addRow("Resultado OK:", self._ed_ok)

        self._ed_ng = QLineEdit()
        self._ed_ng.setPlaceholderText("DB1.DBX0.2 / M101")
        addr_form.addRow("Resultado NG:", self._ed_ng)

        self._sb_hold = QSpinBox()
        self._sb_hold.setRange(50, 5000)
        self._sb_hold.setSuffix(" ms")
        addr_form.addRow("Hold resultado:", self._sb_hold)

        layout.addWidget(addr_group)

        # ── Botones ───────────────────────────────────────────────────────────
        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def _load_values(self) -> None:
        self._cb_enabled.setChecked(self._config.enabled)
        idx = self._cb_brand.findText(self._config.brand)
        self._cb_brand.setCurrentIndex(idx if idx >= 0 else 0)
        self._ed_ip.setText(self._config.ip)
        self._sb_port.setValue(self._config.port)
        self._ed_trigger.setText(self._config.trigger_address)
        self._ed_ok.setText(self._config.result_ok_address)
        self._ed_ng.setText(self._config.result_ng_address)
        self._sb_hold.setValue(self._config.result_hold_ms)

    def get_config(self) -> PLCConfig:
        return PLCConfig(
            enabled=self._cb_enabled.isChecked(),
            brand=self._cb_brand.currentText(),
            ip=self._ed_ip.text().strip(),
            port=self._sb_port.value(),
            trigger_address=self._ed_trigger.text().strip(),
            result_ok_address=self._ed_ok.text().strip(),
            result_ng_address=self._ed_ng.text().strip(),
            result_hold_ms=self._sb_hold.value(),
        )
