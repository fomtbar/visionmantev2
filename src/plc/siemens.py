from __future__ import annotations

import re
from loguru import logger
from src.plc.base import AbstractPLC
from src.core.config_manager import PLCConfig


def _parse_s7_address(address: str) -> tuple[int, int, int]:
    """
    Parsea dirección S7 formato 'DB1.DBX0.0' → (db, byte, bit).
    También acepta 'DB1.DBW0' o 'DB1.DBD0' (retorna bit=0).
    """
    m = re.match(r"DB(\d+)\.DBX(\d+)\.(\d+)", address.upper())
    if m:
        return int(m.group(1)), int(m.group(2)), int(m.group(3))
    raise ValueError(f"Formato de dirección S7 inválido: {address!r} (esperado: DB1.DBX0.0)")


class SiemensPLC(AbstractPLC):
    """Driver Siemens S7-1200/S7-1500 via python-snap7 (puerto TCP 102)."""

    def __init__(self, config: PLCConfig):
        self._config = config
        self._client = None
        self._connected = False

    def connect(self) -> bool:
        try:
            import snap7
            self._client = snap7.client.Client()
            self._client.connect(self._config.ip, 0, 1)
            self._connected = self._client.get_connected()
            if self._connected:
                logger.info(f"Siemens S7: conectado a {self._config.ip}")
            else:
                logger.error(f"Siemens S7: no se pudo conectar a {self._config.ip}")
            return self._connected
        except Exception as e:
            logger.error(f"Siemens S7 connect error: {e}")
            self._connected = False
            return False

    def disconnect(self) -> None:
        if self._client:
            try:
                self._client.disconnect()
            except Exception:
                pass
        self._connected = False
        logger.info("Siemens S7: desconectado")

    def read_trigger_bit(self) -> bool:
        if not self._connected:
            return False
        try:
            db, byte_num, bit_num = _parse_s7_address(self._config.trigger_address)
            data = self._client.db_read(db, byte_num, 1)
            return bool((data[0] >> bit_num) & 1)
        except Exception as e:
            logger.error(f"Siemens read_trigger error: {e}")
            self._connected = False
            return False

    def _write_bit(self, address: str, value: bool) -> None:
        db, byte_num, bit_num = _parse_s7_address(address)
        data = self._client.db_read(db, byte_num, 1)
        if value:
            data[0] |= (1 << bit_num)
        else:
            data[0] &= ~(1 << bit_num)
        self._client.db_write(db, byte_num, data)

    def write_result(self, ok: bool) -> None:
        if not self._connected:
            return
        try:
            self._write_bit(self._config.result_ok_address, ok)
            self._write_bit(self._config.result_ng_address, not ok)
        except Exception as e:
            logger.error(f"Siemens write_result error: {e}")

    def write_result_batch(self, results: dict[str, bool]) -> None:
        # Por ahora escribe solo el resultado global
        all_ok = all(results.values())
        self.write_result(all_ok)

    def is_connected(self) -> bool:
        if self._client:
            try:
                self._connected = self._client.get_connected()
            except Exception:
                self._connected = False
        return self._connected

    def get_info(self) -> dict:
        return {
            "brand": "Siemens S7",
            "ip": self._config.ip,
            "port": self._config.port,
            "connected": self._connected,
        }
