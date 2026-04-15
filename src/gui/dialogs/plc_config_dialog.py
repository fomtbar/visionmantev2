from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog, QFormLayout, QLineEdit, QComboBox,
    QSpinBox, QCheckBox, QDialogButtonBox, QGroupBox,
    QVBoxLayout, QLabel,
)

from src.core.config_manager import PLCConfig

# Hints por marca: (puerto_default, commtype_visible, trigger, ok, ng)
_BRAND_HINTS: dict[str, dict] = {
    "mitsubishi": {
        "port": 1025,
        "commtype": True,
        "trigger":    ("Trigger (bit entrada)",  "ej: M0, M100"),
        "ok":         ("Resultado OK (salida)",  "ej: M1200, M1000"),
        "ng":         ("Resultado NG (salida)",  "ej: M1201, M1001"),
        "note":       "Marcas: M0–M9999  |  Registros: D0–D9999  |  Entradas: X0  |  Salidas: Y0",
    },
    "siemens": {
        "port": 102,
        "commtype": False,
        "trigger":    ("Trigger (bit entrada)",  "ej: DB1.DBX0.0"),
        "ok":         ("Resultado OK (salida)",  "ej: DB1.DBX0.1"),
        "ng":         ("Resultado NG (salida)",  "ej: DB1.DBX0.2"),
        "note":       "Formato S7: DB<n>.DBX<byte>.<bit>",
    },
    "mock": {
        "port": 502,
        "commtype": False,
        "trigger":    ("Trigger",  "cualquier texto"),
        "ok":         ("OK",       "cualquier texto"),
        "ng":         ("NG",       "cualquier texto"),
        "note":       "Simulación en memoria — sin hardware real",
    },
}


class PLCConfigDialog(QDialog):
    def __init__(self, config: PLCConfig, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Configuración PLC")
        self.setMinimumWidth(460)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowType.WindowContextHelpButtonHint)
        self._config = config
        self._build_ui()
        self._load_values()
        self._on_brand_changed(self._cb_brand.currentText())

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # ── Conexión ─────────────────────────────────────────────────────────
        conn_group = QGroupBox("Conexión")
        form = QFormLayout(conn_group)
        form.setSpacing(6)

        self._cb_enabled = QCheckBox("Habilitar comunicación PLC")
        form.addRow(self._cb_enabled)

        self._cb_brand = QComboBox()
        self._cb_brand.addItems(["mock", "siemens", "mitsubishi"])
        self._cb_brand.currentTextChanged.connect(self._on_brand_changed)
        form.addRow("Marca:", self._cb_brand)

        self._ed_ip = QLineEdit()
        self._ed_ip.setPlaceholderText("192.168.3.39")
        form.addRow("IP:", self._ed_ip)

        self._sb_port = QSpinBox()
        self._sb_port.setRange(1, 65535)
        form.addRow("Puerto:", self._sb_port)

        # commtype — solo visible para Mitsubishi
        self._lbl_commtype = QLabel("Modo trama:")
        self._cb_commtype = QComboBox()
        self._cb_commtype.addItems(["binary", "ascii"])
        self._cb_commtype.setToolTip(
            "binary: modo binario (default)\n"
            "ascii:  usar si read/write dan timeout tras conectar"
        )
        form.addRow(self._lbl_commtype, self._cb_commtype)

        layout.addWidget(conn_group)

        # ── Direcciones ───────────────────────────────────────────────────────
        self._addr_group = QGroupBox("Direcciones")
        addr_form = QFormLayout(self._addr_group)
        addr_form.setSpacing(6)

        self._lbl_trigger = QLabel()
        self._ed_trigger = QLineEdit()
        addr_form.addRow(self._lbl_trigger, self._ed_trigger)

        self._lbl_ok = QLabel()
        self._ed_ok = QLineEdit()
        addr_form.addRow(self._lbl_ok, self._ed_ok)

        self._lbl_ng = QLabel()
        self._ed_ng = QLineEdit()
        addr_form.addRow(self._lbl_ng, self._ed_ng)

        self._sb_hold = QSpinBox()
        self._sb_hold.setRange(50, 5000)
        self._sb_hold.setSuffix(" ms")
        addr_form.addRow("Hold resultado:", self._sb_hold)

        # Nota de formato
        self._lbl_note = QLabel()
        self._lbl_note.setStyleSheet("color: #888; font-size: 11px;")
        self._lbl_note.setWordWrap(True)
        addr_form.addRow(self._lbl_note)

        layout.addWidget(self._addr_group)

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
        idx_ct = self._cb_commtype.findText(getattr(self._config, "commtype", "binary"))
        self._cb_commtype.setCurrentIndex(idx_ct if idx_ct >= 0 else 0)
        self._ed_trigger.setText(self._config.trigger_address)
        self._ed_ok.setText(self._config.result_ok_address)
        self._ed_ng.setText(self._config.result_ng_address)
        self._sb_hold.setValue(self._config.result_hold_ms)

    # ── Slot: adaptar UI al cambio de marca ───────────────────────────────────

    def _on_brand_changed(self, brand: str) -> None:
        hints = _BRAND_HINTS.get(brand, _BRAND_HINTS["mock"])

        # Puerto default solo si el usuario no cambió el valor manualmente
        # (si el puerto actual es el default de la OTRA marca, lo pisamos)
        current_port = self._sb_port.value()
        known_ports = {h["port"] for h in _BRAND_HINTS.values()}
        if current_port in known_ports:
            self._sb_port.setValue(hints["port"])

        # commtype visible solo en Mitsubishi
        show_ct = hints["commtype"]
        self._lbl_commtype.setVisible(show_ct)
        self._cb_commtype.setVisible(show_ct)

        # Labels y placeholders de direcciones
        lbl_t, ph_t = hints["trigger"]
        lbl_o, ph_o = hints["ok"]
        lbl_n, ph_n = hints["ng"]

        self._lbl_trigger.setText(lbl_t + ":")
        self._ed_trigger.setPlaceholderText(ph_t)

        self._lbl_ok.setText(lbl_o + ":")
        self._ed_ok.setPlaceholderText(ph_o)

        self._lbl_ng.setText(lbl_n + ":")
        self._ed_ng.setPlaceholderText(ph_n)

        self._lbl_note.setText(hints["note"])

    # ── Resultado ─────────────────────────────────────────────────────────────

    def get_config(self) -> PLCConfig:
        brand = self._cb_brand.currentText()
        commtype = self._cb_commtype.currentText() if brand == "mitsubishi" else "binary"
        return PLCConfig(
            enabled=self._cb_enabled.isChecked(),
            brand=brand,
            ip=self._ed_ip.text().strip(),
            port=self._sb_port.value(),
            commtype=commtype,
            trigger_address=self._ed_trigger.text().strip(),
            result_ok_address=self._ed_ok.text().strip(),
            result_ng_address=self._ed_ng.text().strip(),
            result_hold_ms=self._sb_hold.value(),
        )
