from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

import numpy as np
from loguru import logger

from src.core.config_manager import VisionConfig
from src.core.result_model import InspectionResult, PieceResult
from src.vision.roi_manager import ROIManager, ROICrop
from src.vision.detector import YOLODetector, Detection
from src.vision.template_matcher import TemplateMatcher
from src.vision.fallback_orb import ORBMatcher


# Píxeles extra alrededor de la zona ROI para la búsqueda por template.
# Permite que la pieza esté ligeramente desplazada y el sistema igual la encuentre.
_SEARCH_PADDING = 30


class InspectionClassifier:
    """
    Orquesta la visión: aplica ROIs, ejecuta el detector y produce InspectionResult.

    Jerarquía de detección (por zona):
      1. YOLO    — si está cargado y configurado
      2. Template matching — detector primario ORB-free; busca el objeto
                             dentro de la ventana de zona + margen de búsqueda
      3. ORB     — fallback para casos con poca correlación de pixel
                   (p.ej. piezas con orientación variable)
    """

    def __init__(self, config: VisionConfig, roi_manager: ROIManager, job_id: str = "default"):
        self._config = config
        self._roi_manager = roi_manager
        self._job_id = job_id
        self._yolo: YOLODetector | None = None

        # ── Matchers por zona ─────────────────────────────────────────────────
        # Template matcher (primario): TM_CCOEFF_NORMED — busca en ventana
        self._tmpl_by_zone: dict[str, TemplateMatcher] = {}
        # ORB matcher (fallback): feature matching — robusto a rotación
        self._orb_by_zone: dict[str, ORBMatcher] = {}

        if config.algorithm == "yolo" and config.model_path:
            self._yolo = YOLODetector(config.model_path, config.confidence_threshold)
            if not self._yolo.load():
                logger.warning("YOLO no cargó — usando Template+ORB como fallback")
                self._yolo = None

    def update_config(self, config: VisionConfig) -> None:
        self._config = config
        for m in self._tmpl_by_zone.values():
            m.confidence_threshold = config.confidence_threshold
        for m in self._orb_by_zone.values():
            m.confidence_threshold = config.confidence_threshold

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

        # ── YOLO ─────────────────────────────────────────────────────────────
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

        # ── Template matching (primario) ──────────────────────────────────────
        tmpl = self._tmpl_by_zone.get(zone.id)
        if tmpl and tmpl.is_ready:
            search_img = self._padded_crop(crop, _SEARCH_PADDING)
            logger.debug(
                f"Zona '{zone.id}': Template {tmpl.reference_count} patrón(es) "
                f"zona={zone.w}×{zone.h} búsqueda={search_img.shape[1]}×{search_img.shape[0]}"
            )
            t_status, t_conf = tmpl.match(search_img)
            logger.debug(f"Zona '{zone.id}': Template → {t_status} {t_conf:.3f}")

            # Si hay coincidencia clara → listo
            if t_status == "OK":
                return PieceResult(zone_id=zone.id, status="OK", confidence=t_conf)

            # Resultado ambiguo: intentar refinar con ORB
            orb = self._orb_by_zone.get(zone.id)
            if orb and orb.is_ready:
                o_status, o_conf = orb.match(crop.image)
                logger.debug(f"Zona '{zone.id}': ORB (refinado) → {o_status} {o_conf:.3f}")
                # Usar el mejor de los dos
                if o_conf > t_conf:
                    return PieceResult(zone_id=zone.id, status=o_status, confidence=o_conf)

            return PieceResult(zone_id=zone.id, status=t_status, confidence=t_conf)

        # ── ORB solo (sin template) ───────────────────────────────────────────
        orb = self._orb_by_zone.get(zone.id)
        if orb and orb.is_ready:
            logger.debug(f"Zona '{zone.id}': solo ORB {orb.reference_count} patrón(es)")
            o_status, o_conf = orb.match(crop.image)
            logger.debug(f"Zona '{zone.id}': ORB → {o_status} {o_conf:.3f}")
            return PieceResult(zone_id=zone.id, status=o_status, confidence=o_conf)

        # ── Sin patrones configurados ─────────────────────────────────────────
        logger.warning(
            f"Zona '{zone.id}': sin patrones de referencia — "
            f"zonas tmpl={list(self._tmpl_by_zone.keys())} "
            f"orb={list(self._orb_by_zone.keys())}"
        )
        return PieceResult(zone_id=zone.id, status="NG", confidence=0.0)

    def _padded_crop(self, crop: ROICrop, padding: int) -> np.ndarray:
        """Extrae un crop de la zona con margen extra en los 4 bordes."""
        z = crop.zone
        fh, fw = crop.frame_shape
        x1 = max(0, z.x - padding)
        y1 = max(0, z.y - padding)
        x2 = min(fw, z.x + z.w + padding)
        y2 = min(fh, z.y + z.h + padding)
        return crop.frame[y1:y2, x1:x2]

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
        """
        Agrega un patrón de referencia para una zona.
        Carga en TemplateMatcher (primario) y ORBMatcher (fallback).
        """
        ok_tmpl = self._get_or_create_tmpl(zone_id).add_reference_from_array(image)
        ok_orb  = self._get_or_create_orb(zone_id).add_reference_from_array(image)
        return ok_tmpl or ok_orb   # basta con que uno lo acepte

    def add_zone_reference_from_path(self, zone_id: str, path: Path) -> bool:
        """Agrega un patrón desde archivo."""
        ok_tmpl = self._get_or_create_tmpl(zone_id).add_reference(path)
        ok_orb  = self._get_or_create_orb(zone_id).add_reference(path)
        return ok_tmpl or ok_orb

    def clear_zone_references(self, zone_id: str) -> None:
        """Limpia patrones de una zona."""
        if zone_id in self._tmpl_by_zone:
            self._tmpl_by_zone[zone_id].clear_references()
        if zone_id in self._orb_by_zone:
            self._orb_by_zone[zone_id].clear_references()

    def clear_all_zone_references(self) -> None:
        """Limpia todos los patrones de todas las zonas."""
        self._tmpl_by_zone.clear()
        self._orb_by_zone.clear()

    def zone_pattern_count(self, zone_id: str) -> int:
        """Cuántos patrones tiene la zona (usa template como referencia)."""
        m = self._tmpl_by_zone.get(zone_id)
        return m.reference_count if m else 0

    # ── Helpers internos ──────────────────────────────────────────────────────

    def _get_or_create_tmpl(self, zone_id: str) -> TemplateMatcher:
        if zone_id not in self._tmpl_by_zone:
            self._tmpl_by_zone[zone_id] = TemplateMatcher(self._config.confidence_threshold)
        return self._tmpl_by_zone[zone_id]

    def _get_or_create_orb(self, zone_id: str) -> ORBMatcher:
        if zone_id not in self._orb_by_zone:
            self._orb_by_zone[zone_id] = ORBMatcher(self._config.confidence_threshold)
        return self._orb_by_zone[zone_id]

    # ── Compatibilidad legacy ─────────────────────────────────────────────────

    def load_orb_reference(self, image: np.ndarray) -> bool:
        return False  # legacy — no-op (usar add_zone_reference)

    def load_orb_reference_from_path(self, path) -> bool:
        return False  # legacy — no-op

    # ── Propiedades ───────────────────────────────────────────────────────────

    @property
    def algorithm_ready(self) -> bool:
        if self._config.algorithm == "yolo":
            return self._yolo is not None and self._yolo.is_loaded
        return (
            any(m.is_ready for m in self._tmpl_by_zone.values())
            or any(m.is_ready for m in self._orb_by_zone.values())
        )

    @property
    def algorithm_name(self) -> str:
        if self._config.algorithm == "yolo" and self._yolo and self._yolo.is_loaded:
            return "YOLOv8"
        has_tmpl = any(m.is_ready for m in self._tmpl_by_zone.values())
        has_orb  = any(m.is_ready for m in self._orb_by_zone.values())
        if has_tmpl and has_orb:
            return "Template+ORB"
        if has_tmpl:
            return "Template"
        if has_orb:
            return "ORB"
        return "Sin modelo"
