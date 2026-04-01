from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from loguru import logger


class ORBMatcher:
    """
    Detector de presencia/ausencia basado en feature matching ORB.
    No requiere entrenamiento previo — usa imágenes de referencia OK.
    Ideal para arranque rápido sin dataset etiquetado.
    """

    MIN_MATCH_COUNT = 10

    def __init__(self, confidence_threshold: float = 0.65):
        self.confidence_threshold = confidence_threshold
        self._orb = cv2.ORB_create(nfeatures=500)
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self._reference_descriptors: np.ndarray | None = None
        self._reference_keypoints = None
        self._reference_loaded = False

    def load_reference(self, image_path: str | Path) -> bool:
        """Carga imagen de referencia OK para comparar."""
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
            logger.warning(f"ORB: pocos features en referencia ({len(kp) if kp else 0})")
            return False

        self._reference_keypoints = kp
        self._reference_descriptors = des
        self._reference_loaded = True
        logger.info(f"ORB: referencia cargada con {len(kp)} keypoints desde {path.name}")
        return True

    def load_reference_from_array(self, image: np.ndarray) -> bool:
        """Carga referencia desde numpy array (captura directa desde GUI)."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        kp, des = self._orb.detectAndCompute(gray, None)
        if des is None or len(kp) < 5:
            logger.warning(f"ORB: pocos features en imagen de referencia ({len(kp) if kp else 0})")
            return False
        self._reference_keypoints = kp
        self._reference_descriptors = des
        self._reference_loaded = True
        logger.info(f"ORB: referencia cargada desde array con {len(kp)} keypoints")
        return True

    def match(self, image: np.ndarray) -> tuple[str, float]:
        """
        Compara imagen con referencia.
        Retorna: ("OK"|"NG"|"ABSENT", confidence 0.0-1.0)
        """
        if not self._reference_loaded:
            return "NG", 0.0

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        kp, des = self._orb.detectAndCompute(gray, None)

        if des is None or len(kp) < 3:
            return "ABSENT", 0.0

        matches = self._matcher.knnMatch(self._reference_descriptors, des, k=2)

        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)

        if len(good_matches) < self.MIN_MATCH_COUNT:
            confidence = len(good_matches) / self.MIN_MATCH_COUNT * self.confidence_threshold
            return "NG", round(confidence, 3)

        ref_count = len(self._reference_keypoints)
        confidence = min(1.0, len(good_matches) / (ref_count * 0.5))
        status = "OK" if confidence >= self.confidence_threshold else "NG"
        return status, round(confidence, 3)

    @property
    def is_ready(self) -> bool:
        return self._reference_loaded
