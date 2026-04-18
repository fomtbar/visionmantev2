"""
Herramienta de prueba para ShapeInspector.

Uso:
    python tools/test_shape_inspector.py                 # cámara viva (índice 0)
    python tools/test_shape_inspector.py --cam 1         # otra cámara
    python tools/test_shape_inspector.py --image foto.jpg  # imagen estática

Teclas en vivo:
    d  — toggle modo debug (muestra warp y zona central)
    +/-  — ajusta tolerancia de aspecto ±0.1
    q / ESC — salir
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Asegurar que el raíz del proyecto esté en sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2

from src.vision.shape_inspector import ShapeInspector


def run_live(cam_index: int, inspector: ShapeInspector, debug: bool) -> None:
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        print(f"[ERROR] No se pudo abrir cámara índice {cam_index}")
        return

    print("Cámara abierta. Teclas: [d] debug  [+/-] tolerancia  [q/ESC] salir")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Frame vacío")
            continue

        if debug:
            view = inspector.get_debug_view(frame)
        else:
            result = inspector.inspect(frame)
            view = frame.copy()
            label = f"[{result.status}] conf={result.confidence:.2f}"
            color = (0, 200, 0) if result.status == "OK" else (0, 0, 220) if result.status == "NG" else (0, 180, 220)
            cv2.putText(view, label, (10, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 2)
            if result.piece_corners is not None:
                pts = result.piece_corners.astype(int).reshape((-1, 1, 2))
                cv2.polylines(view, [pts], True, (0, 255, 0), 2)

        cv2.imshow("ShapeInspector", view)
        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), 27):
            break
        elif key == ord("d"):
            debug = not debug
            print(f"Debug: {'ON' if debug else 'OFF'}")
        elif key == ord("+"):
            inspector.aspect_tolerance = round(inspector.aspect_tolerance + 0.1, 2)
            print(f"aspect_tolerance → {inspector.aspect_tolerance}")
        elif key == ord("-"):
            inspector.aspect_tolerance = max(0.1, round(inspector.aspect_tolerance - 0.1, 2))
            print(f"aspect_tolerance → {inspector.aspect_tolerance}")

    cap.release()
    cv2.destroyAllWindows()


def run_image(path: str, inspector: ShapeInspector) -> None:
    frame = cv2.imread(path)
    if frame is None:
        print(f"[ERROR] No se pudo leer imagen: {path}")
        return

    view = inspector.get_debug_view(frame)
    result = inspector.inspect(frame)
    print(f"Resultado: {result.status}  conf={result.confidence:.3f}  "
          f"pieza={result.piece_found}  rect_interno={result.inner_rect_found}")

    cv2.imshow("ShapeInspector", view)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--image", type=str, default=None)
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--aspect", type=float, default=3.33, help="Relación largo/ancho pieza")
    ap.add_argument("--tolerance", type=float, default=0.8)
    ap.add_argument("--min-area", type=float, default=0.04, dest="min_area")
    args = ap.parse_args()

    inspector = ShapeInspector(
        piece_aspect_ratio=args.aspect,
        aspect_tolerance=args.tolerance,
        min_piece_area_frac=args.min_area,
    )
    print(f"ShapeInspector  aspect={inspector.piece_aspect_ratio}  "
          f"tolerance=±{inspector.aspect_tolerance}  "
          f"min_area={inspector.min_piece_area_frac*100:.1f}%")

    if args.image:
        run_image(args.image, inspector)
    else:
        run_live(args.cam, inspector, debug=args.debug)


if __name__ == "__main__":
    main()
