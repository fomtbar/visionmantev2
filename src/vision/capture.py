from __future__ import annotations

import cv2
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal
from loguru import logger

from src.core.config_manager import CameraConfig


class CameraCapture(QThread):
    """Captura de cámara en hilo separado. Emite frames como numpy arrays."""

    frame_ready = pyqtSignal(object)   # np.ndarray BGR

    def __init__(self, config: CameraConfig, parent=None):
        super().__init__(parent)
        self._config = config
        self._cap: cv2.VideoCapture | None = None
        self._running = False
        self._last_frame: np.ndarray | None = None

    def run(self) -> None:
        self._running = True
        self._open_camera()

        while self._running:
            if self._cap and self._cap.isOpened():
                ret, frame = self._cap.read()
                if ret:
                    frame = self._apply_flips(frame)
                    self._last_frame = frame
                    self.frame_ready.emit(frame)
                else:
                    logger.warning("Cámara: frame fallido, intentando reconectar...")
                    self._open_camera()
            else:
                self.msleep(500)
                self._open_camera()

            self.msleep(max(1, 1000 // self._config.fps))

        if self._cap:
            self._cap.release()
            self._cap = None

    def _open_camera(self) -> None:
        if self._cap:
            self._cap.release()
        self._cap = cv2.VideoCapture(self._config.device_index, cv2.CAP_DSHOW)
        if self._cap.isOpened():
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._config.width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._config.height)
            self._cap.set(cv2.CAP_PROP_FPS, self._config.fps)
            logger.info(
                f"Cámara {self._config.device_index} abierta "
                f"{self._config.width}x{self._config.height}@{self._config.fps}fps"
            )
        else:
            logger.error(f"No se pudo abrir la cámara {self._config.device_index}")

    def _apply_flips(self, frame: np.ndarray) -> np.ndarray:
        if self._config.flip_horizontal and self._config.flip_vertical:
            return cv2.flip(frame, -1)
        if self._config.flip_horizontal:
            return cv2.flip(frame, 1)
        if self._config.flip_vertical:
            return cv2.flip(frame, 0)
        return frame

    def get_frame(self) -> np.ndarray | None:
        """Retorna el último frame capturado (para inspección bajo demanda)."""
        return self._last_frame.copy() if self._last_frame is not None else None

    def stop(self) -> None:
        self._running = False
        self.wait(3000)

    def update_config(self, config: CameraConfig) -> None:
        self._config = config
        self._open_camera()
