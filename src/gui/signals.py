from __future__ import annotations

import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal

from src.core.result_model import InspectionResult


class AppSignals(QObject):
    """Bus central de señales de la aplicación."""

    # Cámara
    frame_ready = pyqtSignal(object)            # np.ndarray

    # Inspección
    trigger_requested = pyqtSignal()
    inspection_started = pyqtSignal()
    inspection_complete = pyqtSignal(object)    # InspectionResult

    # PLC
    plc_connected = pyqtSignal(bool)
    plc_trigger_received = pyqtSignal()

    # Modelos / entrenamiento
    model_loaded = pyqtSignal(str)              # model_path
    training_progress = pyqtSignal(int, float)  # epoch, mAP
    training_complete = pyqtSignal(str)         # model_path

    # Config
    config_changed = pyqtSignal()

    # Estado general
    status_message = pyqtSignal(str)
    error_occurred = pyqtSignal(str)


# Singleton global
_signals_instance: AppSignals | None = None


def get_signals() -> AppSignals:
    global _signals_instance
    if _signals_instance is None:
        _signals_instance = AppSignals()
    return _signals_instance
