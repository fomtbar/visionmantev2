from __future__ import annotations

from src.plc.base import AbstractPLC
from src.core.config_manager import PLCConfig


def create_plc(config: PLCConfig) -> AbstractPLC:
    brand = config.brand.lower()

    if brand == "siemens":
        from src.plc.siemens import SiemensPLC
        return SiemensPLC(config)

    if brand == "mitsubishi":
        from src.plc.mitsubishi import MitsubishiPLC
        return MitsubishiPLC(config)

    if brand == "mock":
        from src.plc.mock import MockPLC
        return MockPLC(config)

    raise ValueError(f"Marca de PLC no soportada: {brand!r}")
