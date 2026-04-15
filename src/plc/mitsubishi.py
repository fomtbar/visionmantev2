from __future__ import annotations

from loguru import logger
from src.plc.base import AbstractPLC
from src.core.config_manager import PLCConfig


class MitsubishiPLC(AbstractPLC):
    """Driver Mitsubishi Q/iQ-R/FX via pymcprotocol (MC Protocol 3E frame)."""

    def __init__(self, config: PLCConfig):
        self._config = config
        self._plc = None
        self._connected = False

    def connect(self) -> bool:
        try:
            import pymcprotocol
            commtype = getattr(self._config, "commtype", "binary")
            self._plc = pymcprotocol.Type3E(plctype="Q")
            self._plc.commtype = commtype    # "binary" o "ascii" — asignar ANTES de connect()
            self._plc.soc_timeout = 10.0    # ídem
            self._plc.connect(self._config.ip, self._config.port)
            self._connected = True
            logger.info(
                f"Mitsubishi MC: conectado a {self._config.ip}:{self._config.port} "
                f"[{commtype}]"
            )
            return True
        except Exception as e:
            logger.error(f"Mitsubishi connect error: {e}")
            self._connected = False
            return False

    def disconnect(self) -> None:
        if self._plc:
            try:
                self._plc.close()
            except Exception:
                pass
            self._plc = None   # destruir el objeto — connect() creará uno nuevo limpio
        self._connected = False
        logger.info("Mitsubishi MC: desconectado")

    def read_trigger_bit(self) -> bool:
        if not self._connected:
            return False
        try:
            values = self._plc.batchread_bitunits(
                headdevice=self._config.trigger_address.upper(), readsize=1
            )
            return bool(values[0])
        except Exception as e:
            logger.error(f"Mitsubishi read_trigger error: {e}")
            self._connected = False
            return False

    def write_result(self, ok: bool) -> None:
        if not self._connected:
            return
        try:
            self._plc.batchwrite_bitunits(
                headdevice=self._config.result_ok_address.upper(), values=[1 if ok else 0]
            )
            self._plc.batchwrite_bitunits(
                headdevice=self._config.result_ng_address.upper(), values=[0 if ok else 1]
            )
        except Exception as e:
            logger.error(f"Mitsubishi write_result error: {e}")

    def write_result_batch(self, results: dict[str, bool]) -> None:
        all_ok = all(results.values())
        self.write_result(all_ok)

    def write_bit(self, address: str, value: bool) -> None:
        if not self._connected:
            raise ConnectionError("Mitsubishi no conectado")
        self._plc.batchwrite_bitunits(headdevice=address.upper(), values=[1 if value else 0])
        logger.debug(f"Mitsubishi write_bit {address} = {value}")

    def read_bit(self, address: str) -> bool:
        if not self._connected:
            raise ConnectionError("Mitsubishi no conectado")
        values = self._plc.batchread_bitunits(headdevice=address.upper(), readsize=1)
        return bool(values[0])

    def is_connected(self) -> bool:
        return self._connected

    def get_info(self) -> dict:
        return {
            "brand": "Mitsubishi Q",
            "ip": self._config.ip,
            "port": self._config.port,
            "connected": self._connected,
        }
