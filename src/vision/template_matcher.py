from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from loguru import logger


class TemplateMatcher:
    """
    Detector de presencia basado en template matching (TM_CCOEFF_NORMED).

    Ventajas frente a ORB:
    - Funciona con superficies lisas/uniformes (sin keypoints)
    - Encuentra el objeto en CUALQUIER posición del área de búsqueda
      → tolerante a pequeños desplazamientos de la pieza dentro de la zona
    - Normalizado: robusto a variaciones de iluminación global
    - Más rápido que feature matching para imágenes pequeñas

    Uso típico:
        matcher = TemplateMatcher(confidence_threshold=0.55)
        matcher.add_reference_from_array(reference_crop)   # al hacer setup
        status, conf = matcher.match(search_area)           # al inspeccionar
    """

    def __init__(self, confidence_threshold: float = 0.55):
        self.confidence_threshold = confidence_threshold
        self._templates: list[np.ndarray] = []   # grises

    # ── Carga de referencias ──────────────────────────────────────────────────

    def add_reference_from_array(self, image: np.ndarray) -> bool:
        """Agrega un patrón de referencia desde array (BGR o gris)."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
        if gray.shape[0] < 5 or gray.shape[1] < 5:
            logger.warning(
                f"TemplateMatcher: patrón demasiado pequeño "
                f"({gray.shape[1]}×{gray.shape[0]}px) — ignorado"
            )
            return False
        self._templates.append(gray)
        logger.info(
            f"TemplateMatcher: patrón #{len(self._templates)} cargado "
            f"({gray.shape[1]}×{gray.shape[0]}px)"
        )
        return True

    def add_reference(self, image_path: str | Path) -> bool:
        """Agrega un patrón de referencia desde archivo."""
        path = Path(image_path)
        if not path.exists():
            logger.error(f"TemplateMatcher: no encontrado: {path}")
            return False
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            logger.error(f"TemplateMatcher: no se pudo leer: {path}")
            return False
        self._templates.append(img)
        logger.info(
            f"TemplateMatcher: patrón #{len(self._templates)} cargado desde {path.name} "
            f"({img.shape[1]}×{img.shape[0]}px)"
        )
        return True

    def clear_references(self) -> None:
        """Elimina todos los patrones cargados."""
        self._templates.clear()

    # ── Comparación ───────────────────────────────────────────────────────────

    def match(self, search_image: np.ndarray) -> tuple[str, float]:
        """
        Busca cada patrón almacenado dentro de search_image.

        - search_image puede ser MÁS GRANDE que el patrón: la función localiza
          la mejor posición → tolerancia posicional automática.
        - Si search_image == tamaño del patrón: comparación directa (score único).
        - Devuelve ("OK", conf) si algún patrón supera el umbral.

        Args:
            search_image: área de búsqueda (BGR o gris), puede ser el crop
                          de la zona ROI o un crop ligeramente expandido.
        """
        if not self._templates:
            return "NG", 0.0

        gray = (
            cv2.cvtColor(search_image, cv2.COLOR_BGR2GRAY)
            if len(search_image.shape) == 3
            else search_image
        )

        best_conf = 0.0
        best_loc: tuple | None = None

        for i, tmpl in enumerate(self._templates):
            hs, ws = gray.shape
            ht, wt = tmpl.shape

            # Si el patrón no cabe en el área de búsqueda, escalarlo
            if wt > ws or ht > hs:
                scale = min(ws / wt, hs / ht) * 0.9
                new_w = max(3, int(wt * scale))
                new_h = max(3, int(ht * scale))
                t = cv2.resize(tmpl, (new_w, new_h))
            else:
                t = tmpl

            result = cv2.matchTemplate(gray, t, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)
            conf = float(max_val)
            logger.debug(
                f"TemplateMatcher patrón#{i+1}: conf={conf:.3f} "
                f"best_pos={max_loc} "
                f"search={ws}×{hs} tmpl={t.shape[1]}×{t.shape[0]}"
            )
            if conf > best_conf:
                best_conf = conf
                best_loc = max_loc

        if best_conf >= self.confidence_threshold:
            return "OK", round(best_conf, 3)
        return "NG", round(best_conf, 3)

    # ── Propiedades ───────────────────────────────────────────────────────────

    @property
    def is_ready(self) -> bool:
        return len(self._templates) > 0

    @property
    def reference_count(self) -> int:
        return len(self._templates)
