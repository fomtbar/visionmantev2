from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal
from collections import deque

import numpy as np


@dataclass
class PieceResult:
    zone_id: str
    status: Literal["OK", "NG", "ABSENT"]
    confidence: float
    bounding_box: tuple[int, int, int, int] | None = None  # x, y, w, h

    @property
    def is_ok(self) -> bool:
        return self.status == "OK"


@dataclass
class InspectionResult:
    timestamp: datetime
    global_status: Literal["OK", "NG"]
    pieces: list[PieceResult]
    inference_time_ms: float
    job_id: str
    frame_snapshot: np.ndarray | None = field(default=None, repr=False)

    @property
    def is_ok(self) -> bool:
        return self.global_status == "OK"

    @property
    def ng_pieces(self) -> list[PieceResult]:
        return [p for p in self.pieces if not p.is_ok]

    def summary(self) -> str:
        total = len(self.pieces)
        ok_count = sum(1 for p in self.pieces if p.is_ok)
        return (
            f"{self.global_status} | {ok_count}/{total} piezas OK | "
            f"{self.inference_time_ms:.1f}ms | {self.timestamp.strftime('%H:%M:%S')}"
        )


class ResultHistory:
    def __init__(self, max_size: int = 200):
        self._history: deque[InspectionResult] = deque(maxlen=max_size)

    def add(self, result: InspectionResult) -> None:
        self._history.appendleft(result)

    def get_all(self) -> list[InspectionResult]:
        return list(self._history)

    def get_last_ng(self, n: int = 10) -> list[InspectionResult]:
        return [r for r in self._history if not r.is_ok][:n]

    @property
    def total(self) -> int:
        return len(self._history)

    @property
    def ok_count(self) -> int:
        return sum(1 for r in self._history if r.is_ok)

    @property
    def ng_count(self) -> int:
        return self.total - self.ok_count

    @property
    def ok_rate(self) -> float:
        return (self.ok_count / self.total * 100) if self.total > 0 else 0.0

    def clear(self) -> None:
        self._history.clear()
