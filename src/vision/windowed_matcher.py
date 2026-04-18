"""
WindowedMatcher — template matching con ventana de búsqueda explícita.

Concepto:
  - Se define una VENTANA DE BÚSQUEDA (área mayor que el patrón) en coordenadas
    de frame. El patrón puede aparecer en cualquier posición dentro de ella.
  - Se captura un PATRÓN de referencia (imagen del objeto/característica a buscar).
  - En cada inspección, se busca el patrón DENTRO de la ventana → posición +
    confianza + status OK/NG.

Ventajas sobre matching directo:
  - Tolera desplazamientos sin necesidad de recapturar referencia.
  - La ventana puede ser 2-5× mayor que el patrón → amplia tolerancia posicional.
  - Multi-escala opcional → tolera pequeñas variaciones de distancia/zoom.
  - Muestra DÓNDE encontró el patrón (bbox en frame coords).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import cv2
import numpy as np
from loguru import logger


# ── Tipos auxiliares ────────────────────────────────────────────────────────────

@dataclass
class SearchWindow:
    """
    Ventana de búsqueda rectangular en coordenadas de frame.

    Puede construirse de varias formas:
        SearchWindow(x, y, w, h)
        SearchWindow.from_center(cx, cy, w, h)
        SearchWindow.from_roi(roi_x, roi_y, roi_w, roi_h, expand_px=60)
    """
    x: int
    y: int
    w: int
    h: int

    @classmethod
    def from_center(cls, cx: int, cy: int, w: int, h: int) -> "SearchWindow":
        return cls(x=cx - w // 2, y=cy - h // 2, w=w, h=h)

    @classmethod
    def from_roi(
        cls,
        roi_x: int, roi_y: int, roi_w: int, roi_h: int,
        expand_px: int = 60,
    ) -> "SearchWindow":
        """Expande un ROI existente en todos los bordes."""
        return cls(
            x=roi_x - expand_px,
            y=roi_y - expand_px,
            w=roi_w + expand_px * 2,
            h=roi_h + expand_px * 2,
        )

    def clamped(self, frame_w: int, frame_h: int) -> "SearchWindow":
        """Retorna una copia recortada a los límites del frame."""
        x = max(0, self.x)
        y = max(0, self.y)
        x2 = min(frame_w, self.x + self.w)
        y2 = min(frame_h, self.y + self.h)
        return SearchWindow(x=x, y=y, w=x2 - x, h=y2 - y)

    @property
    def area(self) -> int:
        return self.w * self.h

    def as_slice(self) -> tuple[slice, slice]:
        return slice(self.y, self.y + self.h), slice(self.x, self.x + self.w)


@dataclass
class WindowedMatchResult:
    status: str           # "OK" / "NG" / "NO_WINDOW" / "NO_PATTERN"
    confidence: float
    # Posición del match en coordenadas de frame (None si no encontró)
    found_bbox: tuple[int, int, int, int] | None = None   # x, y, w, h
    scale_used: float = 1.0
    search_window: SearchWindow | None = None
    debug_frame: np.ndarray | None = field(default=None, repr=False)


# ── Clase principal ─────────────────────────────────────────────────────────────

class WindowedMatcher:
    """
    Busca un patrón de referencia dentro de una ventana de búsqueda configurable.

    Flujo típico:
        matcher = WindowedMatcher(confidence_threshold=0.60)
        matcher.set_search_window(SearchWindow(x=100, y=80, w=400, h=250))
        matcher.set_pattern_from_array(reference_crop)   # al hacer setup

        result = matcher.match(frame)                    # en cada inspección
        if result.status == "OK":
            print(f"Encontrado en {result.found_bbox} conf={result.confidence:.2f}")

    Parámetros
    ----------
    confidence_threshold : umbral mínimo para OK (TM_CCOEFF_NORMED 0-1)
    scale_range          : (min, max, steps) para búsqueda multi-escala.
                           Usar (1.0, 1.0, 1) para escala única (más rápido).
    min_pattern_px       : tamaño mínimo en px que debe tener el patrón
    """

    def __init__(
        self,
        confidence_threshold: float = 0.60,
        scale_range: tuple[float, float, int] = (0.85, 1.15, 7),
        min_pattern_px: int = 10,
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.scale_range = scale_range
        self.min_pattern_px = min_pattern_px

        self._window: SearchWindow | None = None
        self._patterns: list[np.ndarray] = []   # grises

    # ── Configuración ──────────────────────────────────────────────────────────

    def set_search_window(self, window: SearchWindow) -> None:
        self._window = window
        logger.info(
            f"WindowedMatcher: ventana ({window.x},{window.y}) "
            f"{window.w}×{window.h}px"
        )

    def set_pattern_from_array(self, image: np.ndarray) -> bool:
        """Agrega un patrón de referencia (puede haber varios)."""
        gray = _to_gray(image)
        if gray.shape[0] < self.min_pattern_px or gray.shape[1] < self.min_pattern_px:
            logger.warning(f"WindowedMatcher: patrón demasiado pequeño {gray.shape} — ignorado")
            return False
        self._patterns.append(gray)
        logger.info(
            f"WindowedMatcher: patrón #{len(self._patterns)} "
            f"cargado {gray.shape[1]}×{gray.shape[0]}px"
        )
        return True

    def set_pattern_from_file(self, path: str) -> bool:
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            logger.error(f"WindowedMatcher: no se pudo leer {path}")
            return False
        self._patterns.append(img)
        logger.info(f"WindowedMatcher: patrón #{len(self._patterns)} cargado desde {path}")
        return True

    def clear_patterns(self) -> None:
        self._patterns.clear()

    # ── Inspección ─────────────────────────────────────────────────────────────

    def match(self, frame: np.ndarray, debug: bool = False) -> WindowedMatchResult:
        """
        Busca el patrón dentro de la ventana de búsqueda.

        Args:
            frame: imagen BGR completa del frame
            debug: si True, agrega debug_frame con anotaciones
        """
        if self._window is None:
            return WindowedMatchResult(status="NO_WINDOW", confidence=0.0)
        if not self._patterns:
            return WindowedMatchResult(status="NO_PATTERN", confidence=0.0)

        # Recortar ventana del frame
        win = self._window.clamped(frame.shape[1], frame.shape[0])
        if win.w < self.min_pattern_px or win.h < self.min_pattern_px:
            logger.warning("WindowedMatcher: ventana fuera del frame o demasiado pequeña")
            return WindowedMatchResult(status="NG", confidence=0.0, search_window=win)

        search_crop = frame[win.as_slice()]
        search_gray = _to_gray(search_crop)

        best_conf = 0.0
        best_loc: tuple[int, int] | None = None
        best_scale = 1.0
        best_pat_size: tuple[int, int] = (0, 0)

        scales = self._build_scales()

        for tmpl_orig in self._patterns:
            for scale in scales:
                tmpl = self._scaled_template(tmpl_orig, scale, search_gray.shape)
                if tmpl is None:
                    continue

                result = cv2.matchTemplate(search_gray, tmpl, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(result)
                conf = float(max_val)

                logger.debug(
                    f"WindowedMatcher scale={scale:.2f} "
                    f"tmpl={tmpl.shape[1]}×{tmpl.shape[0]} "
                    f"search={search_gray.shape[1]}×{search_gray.shape[0]} "
                    f"conf={conf:.3f}"
                )

                if conf > best_conf:
                    best_conf = conf
                    best_loc = max_loc
                    best_scale = scale
                    best_pat_size = (tmpl.shape[1], tmpl.shape[0])

        # Convertir localización a coordenadas de frame
        found_bbox: tuple[int, int, int, int] | None = None
        if best_loc is not None:
            fx = win.x + best_loc[0]
            fy = win.y + best_loc[1]
            found_bbox = (fx, fy, best_pat_size[0], best_pat_size[1])

        status = "OK" if best_conf >= self.confidence_threshold else "NG"

        debug_frame = None
        if debug:
            debug_frame = self._draw_debug(frame.copy(), win, found_bbox, best_conf, status)

        logger.debug(
            f"WindowedMatcher → {status} conf={best_conf:.3f} "
            f"scale={best_scale:.2f} loc={found_bbox}"
        )

        return WindowedMatchResult(
            status=status,
            confidence=round(best_conf, 3),
            found_bbox=found_bbox,
            scale_used=best_scale,
            search_window=win,
            debug_frame=debug_frame,
        )

    def get_debug_view(self, frame: np.ndarray) -> np.ndarray:
        """Retorna frame anotado con ventana, match y confianza. Útil para tuning."""
        result = self.match(frame, debug=True)
        return result.debug_frame if result.debug_frame is not None else frame.copy()

    # ── Helpers internos ───────────────────────────────────────────────────────

    def _build_scales(self) -> list[float]:
        lo, hi, steps = self.scale_range
        if steps <= 1 or lo == hi:
            return [1.0]
        return [lo + (hi - lo) * i / (steps - 1) for i in range(steps)]

    def _scaled_template(
        self,
        tmpl: np.ndarray,
        scale: float,
        search_shape: tuple[int, int],
    ) -> np.ndarray | None:
        """Escala el patrón y verifica que quepa en el área de búsqueda."""
        new_w = max(self.min_pattern_px, int(tmpl.shape[1] * scale))
        new_h = max(self.min_pattern_px, int(tmpl.shape[0] * scale))

        # El patrón debe ser más pequeño que el área de búsqueda
        if new_w >= search_shape[1] or new_h >= search_shape[0]:
            return None

        if scale == 1.0 and tmpl.shape[1] == new_w:
            return tmpl
        return cv2.resize(tmpl, (new_w, new_h))

    def _draw_debug(
        self,
        frame: np.ndarray,
        win: SearchWindow,
        found_bbox: tuple | None,
        conf: float,
        status: str,
    ) -> np.ndarray:
        # Ventana de búsqueda — azul
        cv2.rectangle(
            frame,
            (win.x, win.y), (win.x + win.w, win.y + win.h),
            (200, 130, 0), 2,
        )
        cv2.putText(
            frame, "ventana",
            (win.x + 4, win.y + 16),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 130, 0), 1,
        )

        # Match encontrado — verde/rojo
        if found_bbox:
            fx, fy, fw, fh = found_bbox
            color = (0, 210, 0) if status == "OK" else (0, 0, 210)
            cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), color, 2)

        # Etiqueta de resultado
        label = f"[{status}] {conf:.2f}"
        color = (0, 210, 0) if status == "OK" else (0, 0, 210)
        cv2.putText(frame, label, (win.x, win.y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        return frame

    # ── Propiedades ────────────────────────────────────────────────────────────

    @property
    def is_ready(self) -> bool:
        return bool(self._patterns) and self._window is not None

    @property
    def pattern_count(self) -> int:
        return len(self._patterns)

    @property
    def search_window(self) -> SearchWindow | None:
        return self._window


# ── Utilidad ────────────────────────────────────────────────────────────────────

def _to_gray(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
