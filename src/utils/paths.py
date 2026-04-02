from __future__ import annotations

import sys
from pathlib import Path


def get_app_root() -> Path:
    """
    Retorna el directorio raíz de la aplicación.

    - Cuando corre como exe empaquetado con PyInstaller:
        directorio que contiene el .exe  (dist/VisionMante/)
    - Cuando corre como código fuente:
        raíz del proyecto  (2 niveles sobre src/utils/)
    """
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).parents[2]
