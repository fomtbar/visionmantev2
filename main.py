"""
VisionMante v2 - Sistema de visión industrial
Punto de entrada principal.
"""

import sys
from pathlib import Path

# Asegurar que src/ esté en el path
sys.path.insert(0, str(Path(__file__).parent))

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt

from src.utils.logger import setup_logger
from src.core.config_manager import ConfigManager
from src.core.inspection_engine import InspectionEngine
from src.gui.main_window import MainWindow


def main() -> int:
    setup_logger()

    from loguru import logger
    logger.info("=" * 50)
    logger.info("VisionMante v2 — iniciando")

    app = QApplication(sys.argv)
    app.setApplicationName("VisionMante v2")
    app.setOrganizationName("VisionMante")
    config = ConfigManager()
    engine = InspectionEngine(config)
    window = MainWindow(engine, config)
    window.show()

    logger.info("Aplicación lista")
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
