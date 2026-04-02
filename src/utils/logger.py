import sys
from pathlib import Path
from loguru import logger

from src.utils.paths import get_app_root


def setup_logger(log_level: str = "INFO", log_dir: Path | None = None) -> None:
    logger.remove()

    # sys.stdout es None cuando se compila con PyInstaller console=False
    if sys.stdout is not None:
        logger.add(
            sys.stdout,
            level=log_level,
            format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> - {message}",
            colorize=True,
        )

    if log_dir is None:
        log_dir = get_app_root() / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    logger.add(
        log_dir / "app.log",
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{line} - {message}",
        rotation="5 MB",
        retention="10 days",
        encoding="utf-8",
    )


__all__ = ["logger", "setup_logger"]
