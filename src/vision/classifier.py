from __future__ import annotations

import time
from datetime import datetime

import numpy as np
from loguru import logger

from src.core.config_manager import VisionConfig
from src.core.result_model import InspectionResult, PieceResult
from src.vision.roi_manager import ROIManager, ROICrop
from src.vision.detector import YOLODetector, Detection
from src.vision.fallback_orb import ORBMatcher


class InspectionClassifier:
    """
    Orquesta la visión: toma el frame, aplica ROIs,
    ejecuta el algoritmo (YOLO u ORB) y produce InspectionResult.
    """

    def __init__(self, config: VisionConfig, roi_manager: ROIManager, job_id: str = "default"):
        self._config = config
        self._roi_manager = roi_manager
        self._job_id = job_id
        self._yolo: YOLODetector | None = None
        self._orb: ORBMatcher = ORBMatcher(config.confidence_threshold)

        if config.algorithm == "yolo" and config.model_path:
            self._yolo = YOLODetector(config.model_path, config.confidence_threshold)
            if not self._yolo.load():
                logger.warning("YOLO no cargó — usando ORB como fallback")
                self._yolo = None

    def update_config(self, config: VisionConfig) -> None:
        self._config = config
        self._orb.confidence_threshold = config.confidence_threshold

    def update_job(self, job_id: str) -> None:
        self._job_id = job_id

    def inspect(self, frame: np.ndarray, save_snapshot_on_ng: bool = True) -> InspectionResult:
        t0 = time.perf_counter()
        crops = self._roi_manager.get_crops(frame)
        pieces: list[PieceResult] = []

        for crop in crops:
            piece = self._inspect_crop(crop)
            pieces.append(piece)

        global_status = self._compute_global(pieces)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        snapshot = frame.copy() if (save_snapshot_on_ng and global_status == "NG") else None

        return InspectionResult(
            timestamp=datetime.now(),
            global_status=global_status,
            pieces=pieces,
            inference_time_ms=elapsed_ms,
            job_id=self._job_id,
            frame_snapshot=snapshot,
        )

    def _inspect_crop(self, crop: ROICrop) -> PieceResult:
        zone = crop.zone

        # --- YOLO ---
        if self._yolo and self._yolo.is_loaded:
            detections, _ = self._yolo.detect(crop.image)
            if not detections:
                return PieceResult(zone_id=zone.id, status="ABSENT", confidence=0.0)

            best = max(detections, key=lambda d: d.confidence)
            # Clases: 0=ok, 1=ng (convención del entrenamiento)
            if best.class_name.lower() in ("ok", "0") or best.class_id == 0:
                status = "OK" if best.confidence >= self._config.confidence_threshold else "NG"
            else:
                status = "NG"

            bbox = (zone.x + best.x, zone.y + best.y, best.w, best.h)
            return PieceResult(zone_id=zone.id, status=status,
                               confidence=best.confidence, bounding_box=bbox)

        # --- ORB fallback ---
        if self._orb.is_ready:
            status_str, confidence = self._orb.match(crop.image)
            return PieceResult(zone_id=zone.id, status=status_str, confidence=confidence)

        # Sin modelo ni referencia — siempre NG
        logger.warning(f"Zona '{zone.id}': sin algoritmo configurado")
        return PieceResult(zone_id=zone.id, status="NG", confidence=0.0)

    def _compute_global(self, pieces: list[PieceResult]) -> str:
        if not pieces:
            return "NG"

        if self._config.any_ng_is_global_ng:
            return "NG" if any(not p.is_ok for p in pieces) else "OK"
        else:
            ok_count = sum(1 for p in pieces if p.is_ok)
            return "OK" if ok_count == len(pieces) else "NG"

    def load_orb_reference(self, image: np.ndarray) -> bool:
        return self._orb.load_reference_from_array(image)

    @property
    def algorithm_ready(self) -> bool:
        if self._config.algorithm == "yolo":
            return self._yolo is not None and self._yolo.is_loaded
        return self._orb.is_ready

    @property
    def algorithm_name(self) -> str:
        if self._config.algorithm == "yolo" and self._yolo and self._yolo.is_loaded:
            return "YOLOv8"
        if self._orb.is_ready:
            return "ORB"
        return "Sin modelo"
