"""
Herramienta de prueba para WindowedMatcher.

Flujo interactivo:
  1. Se inicia la cámara.
  2. Se dibuja la VENTANA DE BÚSQUEDA arrastrando el mouse (rectángulo grande).
  3. Dentro de la ventana, se dibuja el PATRÓN (rectángulo pequeño sobre la pieza).
  4. Se captura el patrón y comienza la inspección en tiempo real.

Teclas:
  1  — modo "dibujar ventana de búsqueda"
  2  — modo "dibujar patrón" (debe caber dentro de la ventana)
  c  — capturar patrón del área dibujada
  r  — resetear (borrar patrón y ventana)
  d  — toggle debug (muestra dónde encontró el patrón)
  +/-  — ajustar umbral de confianza ±0.05
  q / ESC — salir

Uso:
    python tools/test_windowed_matcher.py
    python tools/test_windowed_matcher.py --cam 1
    python tools/test_windowed_matcher.py --image foto.jpg
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np

from src.vision.windowed_matcher import WindowedMatcher, SearchWindow


# ── Estado de la UI ─────────────────────────────────────────────────────────────

class _DrawState:
    def __init__(self) -> None:
        self.mode = "window"    # "window" | "pattern"
        self.drawing = False
        self.start: tuple[int, int] = (0, 0)
        self.end: tuple[int, int] = (0, 0)
        self.window_rect: tuple[int, int, int, int] | None = None   # x,y,w,h
        self.pattern_rect: tuple[int, int, int, int] | None = None  # x,y,w,h

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start = (x, y)
            self.end = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.end = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.end = (x, y)
            x1, y1 = min(self.start[0], self.end[0]), min(self.start[1], self.end[1])
            x2, y2 = max(self.start[0], self.end[0]), max(self.start[1], self.end[1])
            w, h = x2 - x1, y2 - y1
            if w > 5 and h > 5:
                if self.mode == "window":
                    self.window_rect = (x1, y1, w, h)
                    print(f"  [ventana] ({x1},{y1}) {w}×{h}px")
                else:
                    self.pattern_rect = (x1, y1, w, h)
                    print(f"  [patrón]  ({x1},{y1}) {w}×{h}px  — presiona [c] para capturar")

    def current_rect(self) -> tuple[int, int, int, int] | None:
        if not self.drawing:
            return None
        x1, y1 = min(self.start[0], self.end[0]), min(self.start[1], self.end[1])
        x2, y2 = max(self.start[0], self.end[0]), max(self.start[1], self.end[1])
        return (x1, y1, x2 - x1, y2 - y1)


def _draw_overlay(frame: np.ndarray, state: _DrawState, matcher: WindowedMatcher, debug: bool) -> np.ndarray:
    vis = frame.copy()

    # Ventana de búsqueda confirmada
    if state.window_rect:
        x, y, w, h = state.window_rect
        cv2.rectangle(vis, (x, y), (x + w, y + h), (200, 130, 0), 2)
        cv2.putText(vis, "ventana busqueda", (x + 4, y + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 130, 0), 1)

    # Patrón confirmado
    if state.pattern_rect:
        x, y, w, h = state.pattern_rect
        cv2.rectangle(vis, (x, y), (x + w, y + h), (255, 80, 0), 2)
        cv2.putText(vis, "patron", (x + 4, y + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 80, 0), 1)

    # Rectángulo en curso
    cur = state.current_rect()
    if cur:
        x, y, w, h = cur
        color = (200, 130, 0) if state.mode == "window" else (255, 80, 0)
        cv2.rectangle(vis, (x, y), (x + w, y + h), color, 1)

    # Resultado de inspección
    if matcher.is_ready:
        result = matcher.match(frame, debug=debug)
        if debug and result.debug_frame is not None:
            vis = result.debug_frame
        else:
            label = f"[{result.status}] {result.confidence:.2f}"
            color = (0, 210, 0) if result.status == "OK" else (0, 0, 220)
            cv2.putText(vis, label, (10, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 2)
            if result.found_bbox:
                fx, fy, fw, fh = result.found_bbox
                cv2.rectangle(vis, (fx, fy), (fx + fw, fy + fh), color, 2)

    # HUD
    mode_label = "[1] VENTANA" if state.mode == "window" else "[2] PATRON"
    th_label = f"umbral={matcher.confidence_threshold:.2f}"
    pats = f"patrones={matcher.pattern_count}"
    cv2.putText(vis, f"Modo: {mode_label}  {th_label}  {pats}",
                (8, vis.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)

    return vis


def run_live(cam_index: int, matcher: WindowedMatcher) -> None:
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        print(f"[ERROR] No se pudo abrir cámara {cam_index}")
        return

    state = _DrawState()
    debug = False

    cv2.namedWindow("WindowedMatcher", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("WindowedMatcher", state.on_mouse)

    print("\n=== WindowedMatcher Tester ===")
    print("  IMPORTANTE: haz clic en la ventana OpenCV antes de usar teclado")
    print("  [1] Dibujar ventana de búsqueda (grande)")
    print("  [2] Dibujar área de patrón (dentro de la ventana)")
    print("  [c] Capturar patrón")
    print("  [r] Resetear  [d] Debug  [+/-] Umbral  [q] Salir\n")

    last_frame = None

    while True:
        # Leer teclado ANTES de cap.read para no perder eventos
        key = cv2.waitKey(30) & 0xFF

        ret, frame = cap.read()
        if ret:
            last_frame = frame

        if last_frame is None:
            continue

        vis = _draw_overlay(last_frame, state, matcher, debug)
        cv2.imshow("WindowedMatcher", vis)
        if key in (ord("q"), 27):
            break
        elif key == ord("1"):
            state.mode = "window"
            print("Modo: dibujar VENTANA DE BÚSQUEDA")
        elif key == ord("2"):
            state.mode = "pattern"
            print("Modo: dibujar PATRÓN")
        elif key == ord("c"):
            if state.pattern_rect and last_frame is not None:
                x, y, w, h = state.pattern_rect
                crop = last_frame[y:y + h, x:x + w]
                if matcher.set_pattern_from_array(crop):
                    print(f"  Patrón capturado ({w}×{h}px). Total: {matcher.pattern_count}")
                else:
                    print("  [WARN] No se pudo capturar patrón (demasiado pequeño)")
                if state.window_rect and matcher.search_window is None:
                    wx, wy, ww, wh = state.window_rect
                    matcher.set_search_window(SearchWindow(wx, wy, ww, wh))
                    print(f"  Ventana asignada automáticamente desde dibujo")
            elif state.window_rect and matcher.search_window is None:
                wx, wy, ww, wh = state.window_rect
                matcher.set_search_window(SearchWindow(wx, wy, ww, wh))
                print("  Ventana asignada (sin patrón aún — dibuja patrón y presiona c)")
            else:
                print("  [WARN] Dibuja primero la ventana [1] y luego el patrón [2]")
        elif key == ord("r"):
            matcher.clear_patterns()
            matcher._window = None  # type: ignore[attr-defined]
            state.window_rect = None
            state.pattern_rect = None
            print("  Reset completo")
        elif key == ord("d"):
            debug = not debug
            print(f"  Debug: {'ON' if debug else 'OFF'}")
        elif key == ord("+"):
            matcher.confidence_threshold = round(min(0.99, matcher.confidence_threshold + 0.05), 2)
            print(f"  Umbral → {matcher.confidence_threshold}")
        elif key == ord("-"):
            matcher.confidence_threshold = round(max(0.10, matcher.confidence_threshold - 0.05), 2)
            print(f"  Umbral → {matcher.confidence_threshold}")

    cap.release()
    cv2.destroyAllWindows()


def run_image(path: str, matcher: WindowedMatcher) -> None:
    frame = cv2.imread(path)
    if frame is None:
        print(f"[ERROR] {path}")
        return

    state = _DrawState()
    debug = True

    cv2.namedWindow("WindowedMatcher")
    cv2.setMouseCallback("WindowedMatcher", state.on_mouse)

    print("Imagen estática. Dibuja ventana [1] → patrón [2] → captura [c] → inspecciona.")

    while True:
        vis = _draw_overlay(frame, state, matcher, debug)
        cv2.imshow("WindowedMatcher", vis)
        key = cv2.waitKey(30) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord("1"):
            state.mode = "window"
        elif key == ord("2"):
            state.mode = "pattern"
        elif key == ord("c") and state.pattern_rect:
            x, y, w, h = state.pattern_rect
            crop = frame[y:y + h, x:x + w]
            matcher.set_pattern_from_array(crop)
            if state.window_rect:
                wx, wy, ww, wh = state.window_rect
                matcher.set_search_window(SearchWindow(wx, wy, ww, wh))
        elif key == ord("r"):
            matcher.clear_patterns()
            matcher._window = None  # type: ignore[attr-defined]
            state.window_rect = None
            state.pattern_rect = None

    cv2.destroyAllWindows()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--image", type=str, default=None)
    ap.add_argument("--threshold", type=float, default=0.60)
    ap.add_argument("--scale-min", type=float, default=0.85, dest="scale_min")
    ap.add_argument("--scale-max", type=float, default=1.15, dest="scale_max")
    ap.add_argument("--scale-steps", type=int, default=7, dest="scale_steps")
    args = ap.parse_args()

    matcher = WindowedMatcher(
        confidence_threshold=args.threshold,
        scale_range=(args.scale_min, args.scale_max, args.scale_steps),
    )
    print(f"WindowedMatcher  threshold={matcher.confidence_threshold}  "
          f"scale={args.scale_min:.2f}–{args.scale_max:.2f} ({args.scale_steps} pasos)")

    if args.image:
        run_image(args.image, matcher)
    else:
        run_live(args.cam, matcher)


if __name__ == "__main__":
    main()
