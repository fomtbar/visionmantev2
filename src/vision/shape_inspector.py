"""
ShapeInspector — detección geométrica robusta para piezas rectangulares.

Pipeline:
  1. Localizar la pieza (butaca) → cuadrilátero principal en frame
  2. Warp de perspectiva → vista normalizada invariante a posición/ángulo
  3. Buscar rectángulo interno en zona central → OK / NG

Diseñado para:
  - Piezas colocadas a mano (posición/orientación variable)
  - Modelos con/sin rectángulo central como feature discriminante
  - Sin imágenes de referencia: puramente geométrico
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from loguru import logger


@dataclass
class ShapeResult:
    piece_found: bool
    inner_rect_found: bool
    status: str           # "OK" / "NG" / "NO_PIECE"
    confidence: float
    warped: np.ndarray | None = field(default=None, repr=False)
    piece_corners: np.ndarray | None = field(default=None, repr=False)
    # (x, y, w, h) en coordenadas del warp
    inner_rect_bbox: tuple[int, int, int, int] | None = None
    debug_frame: np.ndarray | None = field(default=None, repr=False)


class ShapeInspector:
    """
    Inspector geométrico para piezas rectangulares (ej. butaca 1m × 30cm).

    No requiere imágenes de referencia: detecta la pieza por contorno y
    busca el rectángulo interno por análisis de forma.

    Parámetros
    ----------
    piece_aspect_ratio  : relación largo/ancho de la pieza (1m/30cm ≈ 3.33)
    aspect_tolerance    : ± tolerancia sobre piece_aspect_ratio
    min_piece_area_frac : fracción mínima del frame que debe ocupar la pieza
    center_zone_frac    : fracción del warp que define la "zona central" a analizar
    min_inner_area_frac : fracción mínima de la zona central para el rect interno
    max_inner_area_frac : fracción máxima (evita detectar la pieza entera)
    canny_low/high      : umbrales de Canny para ambos pasos
    warp_w / warp_h     : dimensiones de salida del warp (px)
    """

    def __init__(
        self,
        piece_aspect_ratio: float = 3.33,
        aspect_tolerance: float = 0.8,
        min_piece_area_frac: float = 0.04,
        center_zone_frac: float = 0.55,
        min_inner_area_frac: float = 0.015,
        max_inner_area_frac: float = 0.70,
        canny_low: int = 25,
        canny_high: int = 90,
        warp_w: int = 660,
        warp_h: int = 200,
    ) -> None:
        self.piece_aspect_ratio = piece_aspect_ratio
        self.aspect_tolerance = aspect_tolerance
        self.min_piece_area_frac = min_piece_area_frac
        self.center_zone_frac = center_zone_frac
        self.min_inner_area_frac = min_inner_area_frac
        self.max_inner_area_frac = max_inner_area_frac
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.warp_w = warp_w
        self.warp_h = warp_h

    # ── API principal ──────────────────────────────────────────────────────────

    def inspect(self, frame: np.ndarray, debug: bool = False) -> ShapeResult:
        """
        Analiza el frame y retorna ShapeResult con status OK/NG/NO_PIECE.

        Args:
            frame: imagen BGR del frame de cámara
            debug: si True, agrega debug_frame con anotaciones visuales
        """
        corners = self._find_piece(frame)
        if corners is None:
            logger.debug("ShapeInspector: pieza no localizada")
            return ShapeResult(
                piece_found=False,
                inner_rect_found=False,
                status="NO_PIECE",
                confidence=0.0,
            )

        warped = self._warp_perspective(frame, corners)
        inner_bbox, inner_conf = self._find_inner_rect(warped)
        inner_found = inner_bbox is not None

        logger.debug(
            f"ShapeInspector: pieza OK warp={warped.shape[1]}×{warped.shape[0]} "
            f"rect_interno={'SI' if inner_found else 'NO'} conf={inner_conf:.3f}"
        )

        debug_frame = self._draw_debug(frame.copy(), corners, warped, inner_bbox) if debug else None

        return ShapeResult(
            piece_found=True,
            inner_rect_found=inner_found,
            status="OK" if inner_found else "NG",
            confidence=inner_conf,
            warped=warped,
            piece_corners=corners,
            inner_rect_bbox=inner_bbox,
            debug_frame=debug_frame,
        )

    def get_debug_view(self, frame: np.ndarray) -> np.ndarray:
        """
        Retorna una imagen de debug compuesta: frame anotado + warp + zona central.
        Útil para ajustar parámetros en tiempo real.
        """
        result = self.inspect(frame, debug=True)

        ann = result.debug_frame if result.debug_frame is not None else frame.copy()

        if result.warped is not None:
            warped_resized = cv2.resize(result.warped, (self.warp_w, self.warp_h))
            self._draw_center_zone(warped_resized)
            if result.inner_rect_bbox:
                x, y, bw, bh = result.inner_rect_bbox
                cv2.rectangle(warped_resized, (x, y), (x + bw, y + bh), (0, 220, 0), 2)

            # Ajustar alturas para concatenar horizontalmente
            target_h = ann.shape[0]
            scale = target_h / warped_resized.shape[0]
            tw = int(warped_resized.shape[1] * scale)
            warped_resized = cv2.resize(warped_resized, (tw, target_h))
            ann = np.hstack([ann, warped_resized])

        label = f"[{result.status}] conf={result.confidence:.2f}"
        color = (0, 200, 0) if result.status == "OK" else (0, 0, 220)
        cv2.putText(ann, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        return ann

    # ── Paso 1: localizar la pieza ─────────────────────────────────────────────

    def _find_piece(self, frame: np.ndarray) -> np.ndarray | None:
        """
        Encuentra el cuadrilátero principal (la butaca) en el frame.
        Retorna array float32 (4,2) con esquinas, o None.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)

        # Combinar Canny + umbral adaptativo para mayor robustez a distintas iluminaciones
        edges_c = cv2.Canny(blurred, self.canny_low, self.canny_high)
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 21, 4
        )
        edges = cv2.bitwise_or(edges_c, thresh)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.dilate(edges, kernel, iterations=1)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        frame_area = frame.shape[0] * frame.shape[1]
        min_area = frame_area * self.min_piece_area_frac

        best_corners: np.ndarray | None = None
        best_area = 0.0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue

            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

            if len(approx) != 4:
                continue
            if not cv2.isContourConvex(approx):
                continue

            # Relación de aspecto del rectángulo mínimo envolvente
            _, (w, h), _ = cv2.minAreaRect(approx)
            if w == 0 or h == 0:
                continue
            ratio = max(w, h) / min(w, h)

            # Aceptar si la relación está dentro de la tolerancia
            if abs(ratio - self.piece_aspect_ratio) > self.aspect_tolerance:
                continue

            if area > best_area:
                best_area = area
                best_corners = approx.reshape(4, 2).astype(np.float32)

        return best_corners

    # ── Paso 2: warp de perspectiva ────────────────────────────────────────────

    def _warp_perspective(self, frame: np.ndarray, corners: np.ndarray) -> np.ndarray:
        ordered = _order_corners(corners)

        # Detectar orientación para que el warp use la dimensión correcta
        tl, tr, br, bl = ordered
        w_top = float(np.linalg.norm(tr - tl))
        w_bot = float(np.linalg.norm(br - bl))
        h_left = float(np.linalg.norm(bl - tl))
        h_right = float(np.linalg.norm(br - tr))
        avg_w = (w_top + w_bot) / 2
        avg_h = (h_left + h_right) / 2

        # Si la pieza está en vertical (alto > ancho), rotar el warp
        if avg_h > avg_w:
            out_w, out_h = self.warp_h, self.warp_w
        else:
            out_w, out_h = self.warp_w, self.warp_h

        dst = np.array(
            [[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]],
            dtype=np.float32,
        )
        M = cv2.getPerspectiveTransform(ordered, dst)
        return cv2.warpPerspective(frame, M, (out_w, out_h))

    # ── Paso 3: buscar rectángulo interno ─────────────────────────────────────

    def _find_inner_rect(
        self, warped: np.ndarray
    ) -> tuple[tuple[int, int, int, int] | None, float]:
        """
        Busca un rectángulo en la zona central del warp.
        Retorna (bbox, confidence) con bbox=(x, y, w, h) o (None, 0.0).
        """
        h, w = warped.shape[:2]

        # Recortar zona central
        mx = int(w * (1 - self.center_zone_frac) / 2)
        my = int(h * (1 - self.center_zone_frac) / 2)
        cx1, cy1 = mx, my
        cx2, cy2 = w - mx, h - my
        center_crop = warped[cy1:cy2, cx1:cx2]

        if center_crop.size == 0:
            return None, 0.0

        gray = (
            cv2.cvtColor(center_crop, cv2.COLOR_BGR2GRAY)
            if len(center_crop.shape) == 3
            else center_crop
        )
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        center_area = center_crop.shape[0] * center_crop.shape[1]
        min_area = center_area * self.min_inner_area_frac
        max_area = center_area * self.max_inner_area_frac

        candidates: list[tuple[float, tuple[int, int, int, int]]] = []

        for binary in self._edge_maps(blurred):
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
            contours, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < min_area or area > max_area:
                    continue

                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
                if len(approx) != 4:
                    continue
                if not cv2.isContourConvex(approx):
                    continue

                bx, by, bw, bh = cv2.boundingRect(approx)
                rect_area = bw * bh
                if rect_area == 0:
                    continue

                rectangularity = area / rect_area
                if rectangularity < 0.55:
                    continue

                # Score: rectangularidad × proporción de área sobre la zona
                area_score = min(1.0, area / (center_area * 0.25))
                conf = rectangularity * 0.6 + area_score * 0.4
                candidates.append((conf, (bx + cx1, by + cy1, bw, bh)))

        if not candidates:
            return None, 0.0

        best_conf, best_bbox = max(candidates, key=lambda c: c[0])
        return best_bbox, round(best_conf, 3)

    def _edge_maps(self, gray: np.ndarray) -> list[np.ndarray]:
        """Genera múltiples mapas de bordes con distintas estrategias."""
        maps = []
        # Canny estándar
        maps.append(cv2.Canny(gray, self.canny_low, self.canny_high))
        # Canny más agresivo para bordes suaves
        maps.append(cv2.Canny(gray, self.canny_low // 2, self.canny_high // 2))
        # Umbral adaptativo (bueno para diferencias de textura)
        maps.append(
            cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV, 11, 3,
            )
        )
        return maps

    # ── Debug ──────────────────────────────────────────────────────────────────

    def _draw_debug(
        self,
        frame: np.ndarray,
        corners: np.ndarray | None,
        warped: np.ndarray | None,
        inner_bbox: tuple | None,
    ) -> np.ndarray:
        if corners is not None:
            pts = corners.astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], True, (0, 255, 0), 3)
            for i, pt in enumerate(corners.astype(int)):
                cv2.circle(frame, tuple(pt), 6, (0, 200, 255), -1)
                cv2.putText(frame, str(i), tuple(pt + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        if warped is not None:
            self._draw_center_zone(warped)

        return frame

    def _draw_center_zone(self, img: np.ndarray) -> None:
        h, w = img.shape[:2]
        mx = int(w * (1 - self.center_zone_frac) / 2)
        my = int(h * (1 - self.center_zone_frac) / 2)
        cv2.rectangle(img, (mx, my), (w - mx, h - my), (255, 165, 0), 1)


# ── Utilidades ─────────────────────────────────────────────────────────────────

def _order_corners(pts: np.ndarray) -> np.ndarray:
    """Ordena 4 puntos: top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1).flatten()
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect
