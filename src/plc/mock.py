from __future__ import annotations

from loguru import logger
from src.plc.base import AbstractPLC


class MockPLC(AbstractPLC):
    """PLC simulado para pruebas en escritorio sin hardware real."""

    def __init__(self, *args, **kwargs):
        self._connected = False
        self._trigger_value = False
        self._last_result: bool | None = None
        self._write_log: list[dict] = []
        self._bits: dict[str, bool] = {}

    def connect(self) -> bool:
        self._connected = True
        logger.info("MockPLC: conectado (simulación)")
        return True

    def disconnect(self) -> None:
        self._connected = False
        logger.info("MockPLC: desconectado")

    def read_trigger_bit(self) -> bool:
        val = self._trigger_value
        if val:
            self._trigger_value = False   # auto-reset (flanco)
        return val

    def write_result(self, ok: bool) -> None:
        self._last_result = ok
        self._write_log.append({"type": "global", "ok": ok})
        logger.info(f"MockPLC: resultado escrito → {'OK' if ok else 'NG'}")

    def write_result_batch(self, results: dict[str, bool]) -> None:
        self._write_log.append({"type": "batch", "results": results})
        logger.info(f"MockPLC: resultado batch → {results}")

    def is_connected(self) -> bool:
        return self._connected

    def write_bit(self, address: str, value: bool) -> None:
        self._bits[address.upper()] = value
        logger.debug(f"MockPLC: write_bit {address} = {value}")

    def read_bit(self, address: str) -> bool:
        return self._bits.get(address.upper(), False)

    def get_info(self) -> dict:
        return {"brand": "mock", "status": "simulación", "last_result": self._last_result}

    # ── Helpers para testing / interfaz GUI ──────────────────────────────────

    def simulate_trigger(self) -> None:
        """Simula un pulso de trigger desde el PLC."""
        self._trigger_value = True
        logger.debug("MockPLC: trigger simulado")
