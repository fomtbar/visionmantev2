from __future__ import annotations

import time
from enum import Enum, auto

import numpy as np
from PyQt6.QtCore import QObject, QThread, pyqtSignal
from loguru import logger

from src.core.config_manager import ConfigManager
from src.core.result_model import InspectionResult, ResultHistory
from src.vision.capture import CameraCapture
from src.vision.classifier import InspectionClassifier
from src.vision.roi_manager import ROIManager
from src.plc.base import AbstractPLC
from src.plc.plc_factory import create_plc


class EngineState(Enum):
    IDLE = auto()
    TRIGGERED = auto()
    INSPECTING = auto()
    DONE = auto()


class InspectionEngine(QObject):
    """
    Máquina de estados central:
    IDLE → TRIGGERED → INSPECTING → DONE → IDLE

    Recibe trigger por señal Qt (botón o PLC), ejecuta inspección,
    emite resultado y escribe al PLC.
    """

    inspection_complete = pyqtSignal(object)   # InspectionResult
    state_changed = pyqtSignal(str)
    plc_status_changed = pyqtSignal(bool)

    def __init__(self, config_manager: ConfigManager, parent=None):
        super().__init__(parent)
        self._cfg = config_manager
        self._state = EngineState.IDLE
        self._history = ResultHistory(config_manager.app.result_history_max)

        # Vision
        self._roi_manager = ROIManager(config_manager.rois.zones)
        self._classifier = InspectionClassifier(
            config_manager.vision, self._roi_manager, config_manager.app.job_name
        )

        # Camera
        self._camera = CameraCapture(config_manager.camera)
        self._camera.frame_ready.connect(self._on_frame)
        self._camera.start()

        # PLC
        self._plc: AbstractPLC | None = None
        self._plc_thread: PLCWatchdog | None = None
        if config_manager.plc.enabled:
            self._setup_plc()

        self._last_trigger_time: float = 0.0

    # ── Trigger entry points ─────────────────────────────────────────────────

    def trigger(self) -> None:
        """Disparo manual (botón en GUI o señal de PLC watchdog)."""
        now = time.perf_counter()
        min_gap = self._cfg.vision.min_cycle_time_ms / 1000.0
        if self._state != EngineState.IDLE:
            logger.debug("Trigger ignorado — motor ocupado")
            return
        if (now - self._last_trigger_time) < min_gap:
            logger.debug("Trigger ignorado — ciclo mínimo no cumplido")
            return

        self._last_trigger_time = now
        self._set_state(EngineState.TRIGGERED)
        self._run_inspection()

    # ── Inspection cycle ─────────────────────────────────────────────────────

    def _run_inspection(self) -> None:
        self._set_state(EngineState.INSPECTING)
        frame = self._camera.get_frame()

        if frame is None:
            logger.warning("Inspección cancelada — sin frame de cámara")
            self._set_state(EngineState.IDLE)
            return

        result = self._classifier.inspect(frame)
        self._history.add(result)

        if self._plc:
            self._send_to_plc(result)

        self.inspection_complete.emit(result)
        logger.info(result.summary())
        self._set_state(EngineState.IDLE)

    def _send_to_plc(self, result: InspectionResult) -> None:
        try:
            if self._cfg.plc.result_mode if hasattr(self._cfg.plc, 'result_mode') else True:
                self._plc.write_result(result.is_ok)
            # batch por pieza
            batch = {p.zone_id: p.is_ok for p in result.pieces}
            self._plc.write_result_batch(batch)
        except Exception as e:
            logger.error(f"Error escribiendo resultado al PLC: {e}")

    # ── Frame handling ────────────────────────────────────────────────────────

    def _on_frame(self, frame: np.ndarray) -> None:
        pass  # Los frames se consumen on-demand en get_frame()

    # ── PLC setup & watchdog ─────────────────────────────────────────────────

    def _setup_plc(self) -> None:
        self._plc = create_plc(self._cfg.plc)
        connected = self._plc.connect()
        self.plc_status_changed.emit(connected)

        if connected:
            self._plc_thread = PLCWatchdog(self._plc, self._cfg.plc.poll_interval_ms)
            self._plc_thread.trigger_received.connect(self.trigger)
            self._plc_thread.connection_lost.connect(lambda: self.plc_status_changed.emit(False))
            self._plc_thread.start()

    def reconnect_plc(self) -> bool:
        if self._plc:
            ok = self._plc.connect()
            self.plc_status_changed.emit(ok)
            return ok
        return False

    # ── Config updates ────────────────────────────────────────────────────────

    def reload_config(self) -> None:
        self._cfg.load()
        self._roi_manager.set_zones(self._cfg.rois.zones)
        self._classifier.update_config(self._cfg.vision)
        self._classifier.update_job(self._cfg.app.job_name)

    def load_orb_reference(self, image: np.ndarray) -> bool:
        return self._classifier.load_orb_reference(image)

    # ── Accessors ─────────────────────────────────────────────────────────────

    @property
    def history(self) -> ResultHistory:
        return self._history

    @property
    def roi_manager(self) -> ROIManager:
        return self._roi_manager

    @property
    def camera(self) -> CameraCapture:
        return self._camera

    @property
    def state(self) -> EngineState:
        return self._state

    @property
    def algorithm_name(self) -> str:
        return self._classifier.algorithm_name

    def _set_state(self, state: EngineState) -> None:
        self._state = state
        self.state_changed.emit(state.name)

    def shutdown(self) -> None:
        self._camera.stop()
        if self._plc_thread:
            self._plc_thread.stop()
        if self._plc:
            self._plc.disconnect()


class PLCWatchdog(QThread):
    """Hilo de polling al PLC: detecta trigger y vigila conexión."""

    trigger_received = pyqtSignal()
    connection_lost = pyqtSignal()

    def __init__(self, plc: AbstractPLC, poll_ms: int = 20, parent=None):
        super().__init__(parent)
        self._plc = plc
        self._poll_ms = poll_ms
        self._running = False

    def run(self) -> None:
        self._running = True
        while self._running:
            if not self._plc.is_connected():
                self.connection_lost.emit()
                self.msleep(2000)
                self._plc.connect()
                continue

            try:
                if self._plc.read_trigger_bit():
                    self.trigger_received.emit()
            except Exception as e:
                logger.error(f"PLCWatchdog error: {e}")
                self.connection_lost.emit()

            self.msleep(self._poll_ms)

    def stop(self) -> None:
        self._running = False
        self.wait(2000)
