from __future__ import annotations

from pathlib import Path
from typing import Literal

import tomllib
import tomli_w
from pydantic import BaseModel, Field, field_validator
from loguru import logger

from src.utils.paths import get_app_root


# ── Pydantic models ──────────────────────────────────────────────────────────

class AppConfig(BaseModel):
    job_name: str = "default"
    language: str = "es"
    log_level: str = "INFO"
    result_history_max: int = 200


class CameraConfig(BaseModel):
    device_index: int = 0
    width: int = 1280
    height: int = 720
    fps: int = 30
    flip_horizontal: bool = False
    flip_vertical: bool = False


class VisionConfig(BaseModel):
    algorithm: Literal["yolo", "orb"] = "orb"
    model_path: str = ""
    confidence_threshold: float = Field(default=0.65, ge=0.0, le=1.0)
    expected_pieces: int = Field(default=1, ge=1, le=20)
    result_mode: Literal["global", "per_piece"] = "global"
    any_ng_is_global_ng: bool = True
    min_cycle_time_ms: int = 200


class PLCConfig(BaseModel):
    enabled: bool = False
    brand: Literal["siemens", "mitsubishi", "mock"] = "mock"
    ip: str = "192.168.0.1"
    port: int = 1025   # Mitsubishi Q MC Protocol 3E. Siemens S7 usa 102.
    commtype: Literal["binary", "ascii"] = "binary"   # Mitsubishi: binario o ASCII
    trigger_address: str = "DB1.DBX0.0"
    result_ok_address: str = "DB1.DBX0.1"
    result_ng_address: str = "DB1.DBX0.2"
    result_hold_ms: int = 500
    poll_interval_ms: int = 20

    @field_validator("ip")
    @classmethod
    def validate_ip(cls, v: str) -> str:
        import re
        pattern = r"^(\d{1,3}\.){3}\d{1,3}$"
        if not re.match(pattern, v):
            raise ValueError(f"IP inválida: {v}")
        return v


class ROIZone(BaseModel):
    id: str
    x: int
    y: int
    w: int
    h: int
    expected_class: str = "ok"
    expected_count: int = 1


class ROIsConfig(BaseModel):
    zones: list[ROIZone] = Field(default_factory=list)


class FullConfig(BaseModel):
    app: AppConfig = Field(default_factory=AppConfig)
    camera: CameraConfig = Field(default_factory=CameraConfig)
    vision: VisionConfig = Field(default_factory=VisionConfig)
    plc: PLCConfig = Field(default_factory=PLCConfig)
    rois: ROIsConfig = Field(default_factory=ROIsConfig)


# ── Manager ──────────────────────────────────────────────────────────────────

class ConfigManager:
    _DEFAULT_PATH = get_app_root() / "config" / "app_config.toml"

    def __init__(self, config_path: Path | None = None):
        self._path = config_path or self._DEFAULT_PATH
        self._config: FullConfig = FullConfig()
        self.load()

    def load(self) -> None:
        if not self._path.exists():
            logger.warning(f"Config no encontrada en {self._path}, usando defaults")
            self._config = FullConfig()
            return

        try:
            with open(self._path, "rb") as f:
                raw = tomllib.load(f)
            self._config = FullConfig.model_validate(raw)
            logger.info(f"Config cargada desde {self._path}")
        except Exception as e:
            logger.error(f"Error cargando config: {e} — usando defaults")
            self._config = FullConfig()

    def save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        raw = self._config.model_dump()
        with open(self._path, "wb") as f:
            tomli_w.dump(raw, f)
        logger.info(f"Config guardada en {self._path}")

    @property
    def app(self) -> AppConfig:
        return self._config.app

    @property
    def camera(self) -> CameraConfig:
        return self._config.camera

    @property
    def vision(self) -> VisionConfig:
        return self._config.vision

    @property
    def plc(self) -> PLCConfig:
        return self._config.plc

    @property
    def rois(self) -> ROIsConfig:
        return self._config.rois

    def update_camera(self, **kwargs) -> None:
        self._config.camera = self._config.camera.model_copy(update=kwargs)

    def update_vision(self, **kwargs) -> None:
        self._config.vision = self._config.vision.model_copy(update=kwargs)

    def update_plc(self, **kwargs) -> None:
        self._config.plc = self._config.plc.model_copy(update=kwargs)

    def update_rois(self, zones: list[dict]) -> None:
        self._config.rois = ROIsConfig(zones=[ROIZone(**z) for z in zones])

    def get_full(self) -> FullConfig:
        return self._config
