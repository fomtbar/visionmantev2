from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from loguru import logger


class ORBMatcher:
    """
    Detector de presencia/ausencia basado en feature matching ORB.
    Soporta múltiples imágenes de referencia por instancia.
    Retorna OK si la imagen inspeccionada coincide con ALGUNO de los patrones cargados.
    """

    MIN_MATCH_COUNT = 10

    def __init__(self, confidence_threshold: float = 0.65):
        self.confidence_threshold = confidence_threshold
        self._orb = cv2.ORB_create(nfeatures=500)
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        # Lista de (keypoints, descriptors) — un elemento por patrón cargado
        self._references: list[tuple] = []

    # ── Carga de referencias ──────────────────────────────────────────────────

    def add_reference(self, image_path: str | Path) -> bool:
        """Agrega un patrón de referencia desde archivo."""
        path = Path(image_path)
        if not path.exists():
            logger.error(f"ORB: referencia no encontrada: {path}")
            return False

        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            logger.error(f"ORB: no se pudo leer imagen: {path}")
            return False

        kp, des = self._orb.detectAndCompute(img, None)
        if des is None or len(kp) < 5:
            logger.warning(f"ORB: pocos features en patrón ({len(kp) if kp else 0})")
            return False

        self._references.append((kp, des))
        logger.info(
            f"ORB: patrón #{len(self._references)} cargado con {len(kp)} keypoints "
            f"desde {path.name}"
        )
        return True

    def add_reference_from_array(self, image: np.ndarray) -> bool:
        """Agrega un patrón de referencia desde numpy array (captura directa desde GUI)."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        kp, des = self._orb.detectAndCompute(gray, None)
        if des is None or len(kp) < 5:
            logger.warning(f"ORB: pocos features en patrón desde array ({len(kp) if kp else 0})")
            return False
        self._references.append((kp, des))
        logger.info(
            f"ORB: patrón #{len(self._references)} cargado desde array con {len(kp)} keypoints"
        )
        return True

    def clear_references(self) -> None:
        """Elimina todos los patrones cargados en memoria."""
        self._references.clear()

    # ── Compatibilidad hacia atrás ────────────────────────────────────────────

    def load_reference(self, image_path: str | Path) -> bool:
        """Reemplaza todos los patrones con uno nuevo (compatibilidad legacy)."""
        self.clear_references()
        return self.add_reference(image_path)

    def load_reference_from_array(self, image: np.ndarray) -> bool:
        """Reemplaza todos los patrones con uno nuevo (compatibilidad legacy)."""
        self.clear_references()
        return self.add_reference_from_array(image)

    # ── Comparación ───────────────────────────────────────────────────────────

    def match(self, image: np.ndarray) -> tuple[str, float]:
        """
        Compara la imagen contra TODOS los patrones cargados.
        Retorna ("OK", confidence) si alguno coincide; ("NG"/"ABSENT", confidence) si no.
        """
        if not self._references:
            return "NG", 0.0

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        kp, des = self._orb.detectAndCompute(gray, None)

        if des is None or len(kp) < 3:
            return "ABSENT", 0.0

        best_confidence = 0.0

        for ref_kp, ref_des in self._references:
            matches = self._matcher.knnMatch(ref_des, des, k=2)

            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < 0.75 * n.distance:
                        good_matches.append(m)

            if len(good_matches) >= self.MIN_MATCH_COUNT:
                confidence = min(1.0, len(good_matches) / (len(ref_kp) * 0.5))
                if confidence >= self.confidence_threshold:
                    return "OK", round(confidence, 3)  # Coincidencia encontrada
                best_confidence = max(best_confidence, confidence)
            else:
                partial = len(good_matches) / self.MIN_MATCH_COUNT * self.confidence_threshold
                best_confidence = max(best_confidence, partial)

        return "NG", round(best_confidence, 3)

    # ── Propiedades ───────────────────────────────────────────────────────────

    @property
    def is_ready(self) -> bool:
        return len(self._references) > 0

    @property
    def reference_count(self) -> int:
        return len(self._references)
