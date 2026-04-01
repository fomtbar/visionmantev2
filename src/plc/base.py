from __future__ import annotations

from abc import ABC, abstractmethod


class AbstractPLC(ABC):
    """Interfaz unificada para cualquier marca de PLC."""

    @abstractmethod
    def connect(self) -> bool: ...

    @abstractmethod
    def disconnect(self) -> None: ...

    @abstractmethod
    def read_trigger_bit(self) -> bool:
        """Lee el bit de trigger configurado. Retorna True si hay disparo."""
        ...

    @abstractmethod
    def write_result(self, ok: bool) -> None:
        """Escribe el resultado global (OK=True / NG=False)."""
        ...

    @abstractmethod
    def write_result_batch(self, results: dict[str, bool]) -> None:
        """Escribe resultados por zona {zone_id: ok}."""
        ...

    @abstractmethod
    def is_connected(self) -> bool: ...

    @abstractmethod
    def get_info(self) -> dict: ...
