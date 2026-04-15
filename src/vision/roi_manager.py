from __future__ import annotations

import numpy as np
from dataclasses import dataclass

from src.core.config_manager import ROIZone


@dataclass
class ROICrop:
    zone: ROIZone
    image: np.ndarray              # crop ajustado a la zona (copia)
    frame: np.ndarray              # frame completo (referencia — para búsqueda expandida)
    frame_shape: tuple[int, int]   # (h, w) del frame original


class ROIManager:
    """Gestiona zonas de inspección (ROIs)."""

    def __init__(self, zones: list[ROIZone] | None = None):
        self._zones: list[ROIZone] = zones or []

    def set_zones(self, zones: list[ROIZone]) -> None:
        self._zones = zones

    def add_zone(self, zone: ROIZone) -> None:
        self._zones = [z for z in self._zones if z.id != zone.id]
        self._zones.append(zone)

    def remove_zone(self, zone_id: str) -> None:
        self._zones = [z for z in self._zones if z.id != zone_id]

    def get_zones(self) -> list[ROIZone]:
        return list(self._zones)

    def apply(self, frame: np.ndarray) -> list[ROICrop]:
        """Recorta el frame según cada zona definida."""
        h, w = frame.shape[:2]
        crops: list[ROICrop] = []

        for zone in self._zones:
            x1 = max(0, zone.x)
            y1 = max(0, zone.y)
            x2 = min(w, zone.x + zone.w)
            y2 = min(h, zone.y + zone.h)
            if x2 > x1 and y2 > y1:
                crops.append(ROICrop(
                    zone=zone,
                    image=frame[y1:y2, x1:x2].copy(),
                    frame=frame,   # referencia al frame — sin copiar
                    frame_shape=(h, w),
                ))

        return crops

    def full_frame_zone(self, frame: np.ndarray) -> list[ROICrop]:
        """Si no hay ROIs definidas, usa el frame completo como una zona."""
        h, w = frame.shape[:2]
        default_zone = ROIZone(id="full", x=0, y=0, w=w, h=h)
        return [ROICrop(zone=default_zone, image=frame.copy(), frame=frame, frame_shape=(h, w))]

    def get_crops(self, frame: np.ndarray) -> list[ROICrop]:
        if self._zones:
            return self.apply(frame)
        return self.full_frame_zone(frame)

    def has_zones(self) -> bool:
        return len(self._zones) > 0
