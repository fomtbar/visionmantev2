from __future__ import annotations

import time
from pathlib import Path
from dataclasses import dataclass

import cv2
import numpy as np
from loguru import logger


@dataclass
class Detection:
    x: int
    y: int
    w: int
    h: int
    confidence: float
    class_id: int
    class_name: str


class YOLODetector:
    """
    Detector basado en YOLOv8-nano exportado a ONNX.
    Corre completamente en CPU usando ONNX Runtime.
    """

    INPUT_SIZE = 640

    def __init__(self, model_path: str | Path, confidence_threshold: float = 0.5):
        self._model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold
        self._session = None
        self._class_names: list[str] = []
        self._input_name: str = ""
        self._last_inference_ms: float = 0.0

    def load(self) -> bool:
        if not self._model_path.exists():
            logger.error(f"YOLO: modelo no encontrado: {self._model_path}")
            return False
        try:
            import onnxruntime as ort
            providers = self._get_providers()
            self._session = ort.InferenceSession(str(self._model_path), providers=providers)
            self._input_name = self._session.get_inputs()[0].name
            logger.info(f"YOLO: modelo cargado desde {self._model_path.name} | providers: {providers}")
            return True
        except Exception as e:
            logger.error(f"YOLO: error cargando modelo: {e}")
            return False

    def _get_providers(self) -> list[str]:
        try:
            import onnxruntime as ort
            available = ort.get_available_providers()
            if "OpenVINOExecutionProvider" in available:
                logger.info("YOLO: usando OpenVINO EP (aceleración Intel)")
                return ["OpenVINOExecutionProvider", "CPUExecutionProvider"]
        except Exception:
            pass
        return ["CPUExecutionProvider"]

    def detect(self, image: np.ndarray) -> tuple[list[Detection], float]:
        """
        Ejecuta inferencia sobre la imagen.
        Retorna (lista de detecciones, tiempo_ms).
        """
        if self._session is None:
            return [], 0.0

        input_tensor, scale_x, scale_y, pad_x, pad_y = self._preprocess(image)

        t0 = time.perf_counter()
        outputs = self._session.run(None, {self._input_name: input_tensor})
        inference_ms = (time.perf_counter() - t0) * 1000
        self._last_inference_ms = inference_ms

        detections = self._postprocess(outputs[0], scale_x, scale_y, pad_x, pad_y)
        return detections, inference_ms

    def _preprocess(self, image: np.ndarray):
        h, w = image.shape[:2]
        scale = self.INPUT_SIZE / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(image, (new_w, new_h))

        pad_x = (self.INPUT_SIZE - new_w) // 2
        pad_y = (self.INPUT_SIZE - new_h) // 2
        padded = cv2.copyMakeBorder(resized, pad_y, self.INPUT_SIZE - new_h - pad_y,
                                     pad_x, self.INPUT_SIZE - new_w - pad_x,
                                     cv2.BORDER_CONSTANT, value=(114, 114, 114))

        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        tensor = rgb.astype(np.float32) / 255.0
        tensor = np.transpose(tensor, (2, 0, 1))[np.newaxis]

        return tensor, scale, scale, pad_x, pad_y

    def _postprocess(self, output: np.ndarray, scale_x: float, scale_y: float,
                     pad_x: int, pad_y: int) -> list[Detection]:
        detections: list[Detection] = []
        # YOLOv8 output shape: [1, 84, 8400] -> transpose -> [8400, 84]
        predictions = output[0].T if output.ndim == 3 else output

        for pred in predictions:
            if pred.ndim == 0:
                continue
            scores = pred[4:] if len(pred) > 5 else pred[4:5]
            class_id = int(np.argmax(scores))
            confidence = float(scores[class_id])

            if confidence < self.confidence_threshold:
                continue

            cx, cy, bw, bh = pred[:4]
            x = int((cx - pad_x) / scale_x - bw / 2 / scale_x)
            y = int((cy - pad_y) / scale_y - bh / 2 / scale_y)
            w = int(bw / scale_x)
            h = int(bh / scale_y)

            class_name = (self._class_names[class_id]
                          if class_id < len(self._class_names)
                          else str(class_id))

            detections.append(Detection(
                x=max(0, x), y=max(0, y), w=w, h=h,
                confidence=confidence,
                class_id=class_id,
                class_name=class_name,
            ))

        return detections

    def set_class_names(self, names: list[str]) -> None:
        self._class_names = names

    @property
    def is_loaded(self) -> bool:
        return self._session is not None

    @property
    def last_inference_ms(self) -> float:
        return self._last_inference_ms
