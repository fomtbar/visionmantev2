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

    Cada zona ROI tiene su propio ORBMatcher con sus patrones de referencia.
    Al inspeccionar, se compara contra TODOS los patrones de la zona;
    si alguno coincide → OK.
    """

    def __init__(self, config: VisionConfig, roi_manager: ROIManager, job_id: str = "default"):
        self._config = config
        self._roi_manager = roi_manager
        self._job_id = job_id
        self._yolo: YOLODetector | None = None

        # Matcher global (legacy / fallback cuando no hay matcher por zona)
        self._orb: ORBMatcher = ORBMatcher(config.confidence_threshold)

        # Matchers por zona: { zone_id: ORBMatcher }
        self._orb_by_zone: dict[str, ORBMatcher] = {}

        if config.algorithm == "yolo" and config.model_path:
            self._yolo = YOLODetector(config.model_path, config.confidence_threshold)
            if not self._yolo.load():
                logger.warning("YOLO no cargó — usando ORB como fallback")
                self._yolo = None

    def update_config(self, config: VisionConfig) -> None:
        self._config = config
        self._orb.confidence_threshold = config.confidence_threshold
        for matcher in self._orb_by_zone.values():
            matcher.confidence_threshold = config.confidence_threshold

    def update_job(self, job_id: str) -> None:
        self._job_id = job_id

    # ── Inspección ────────────────────────────────────────────────────────────

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
            if best.class_name.lower() in ("ok", "0") or best.class_id == 0:
                status = "OK" if best.confidence >= self._config.confidence_threshold else "NG"
            else:
                status = "NG"

            bbox = (zone.x + best.x, zone.y + best.y, best.w, best.h)
            return PieceResult(zone_id=zone.id, status=status,
                               confidence=best.confidence, bounding_box=bbox)

        # --- ORB: matcher por zona (con fallback al global) ---
        matcher = self._orb_by_zone.get(zone.id)
        using_fallback = False
        if matcher is None or not matcher.is_ready:
            matcher = self._orb  # fallback legacy
            using_fallback = True

        if matcher.is_ready:
            n_refs = matcher.reference_count
            logger.debug(
                f"Zona '{zone.id}': ORB {'(global fallback)' if using_fallback else ''} "
                f"{n_refs} patrón(es), crop={crop.image.shape[1]}x{crop.image.shape[0]}"
            )
            status_str, confidence = matcher.match(crop.image)
            logger.debug(f"Zona '{zone.id}': resultado={status_str} conf={confidence:.3f}")
            return PieceResult(zone_id=zone.id, status=status_str, confidence=confidence)

        # Sin modelo ni referencia
        logger.warning(
            f"Zona '{zone.id}': sin patrones de referencia — zonas cargadas: "
            f"{list(self._orb_by_zone.keys())}"
        )
        return PieceResult(zone_id=zone.id, status="NG", confidence=0.0)

    def _compute_global(self, pieces: list[PieceResult]) -> str:
        if not pieces:
            return "NG"

        if self._config.any_ng_is_global_ng:
            return "NG" if any(not p.is_ok for p in pieces) else "OK"
        else:
            ok_count = sum(1 for p in pieces if p.is_ok)
            return "OK" if ok_count == len(pieces) else "NG"

    # ── Gestión de patrones por zona ──────────────────────────────────────────

    def add_zone_reference(self, zone_id: str, image: np.ndarray) -> bool:
        """Agrega un patrón de referencia para una zona específica."""
        if zone_id not in self._orb_by_zone:
            self._orb_by_zone[zone_id] = ORBMatcher(self._config.confidence_threshold)
        return self._orb_by_zone[zone_id].add_reference_from_array(image)

    def add_zone_reference_from_path(self, zone_id: str, path) -> bool:
        """Agrega un patrón de referencia desde archivo para una zona específica."""
        if zone_id not in self._orb_by_zone:
            self._orb_by_zone[zone_id] = ORBMatcher(self._config.confidence_threshold)
        return self._orb_by_zone[zone_id].add_reference(path)

    def clear_zone_references(self, zone_id: str) -> None:
        """Limpia todos los patrones de una zona en memoria."""
        if zone_id in self._orb_by_zone:
            self._orb_by_zone[zone_id].clear_references()

    def clear_all_zone_references(self) -> None:
        """Limpia todos los patrones de TODAS las zonas (usar antes de re-setup)."""
        self._orb_by_zone.clear()
        self._orb.clear_references()

    def zone_pattern_count(self, zone_id: str) -> int:
        """Retorna cuántos patrones tiene cargados la zona."""
        matcher = self._orb_by_zone.get(zone_id)
        return matcher.reference_count if matcher else 0

    # ── Compatibilidad legacy ─────────────────────────────────────────────────

    def load_orb_reference(self, image: np.ndarray) -> bool:
        """Legacy: carga referencia en el matcher global."""
        return self._orb.load_reference_from_array(image)

    def load_orb_reference_from_path(self, path) -> bool:
        """Legacy: carga referencia en el matcher global."""
        return self._orb.load_reference(path)

    # ── Propiedades ───────────────────────────────────────────────────────────

    @property
    def algorithm_ready(self) -> bool:
        if self._config.algorithm == "yolo":
            return self._yolo is not None and self._yolo.is_loaded
        # ORB listo si hay al menos una zona con patrones o el global tiene referencia
        return (
            self._orb.is_ready
            or any(m.is_ready for m in self._orb_by_zone.values())
        )

    @property
    def algorithm_name(self) -> str:
        if self._config.algorithm == "yolo" and self._yolo and self._yolo.is_loaded:
            return "YOLOv8"
        if self._orb.is_ready or any(m.is_ready for m in self._orb_by_zone.values()):
            return "ORB"
        return "Sin modelo"
