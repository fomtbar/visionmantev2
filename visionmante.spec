# -*- mode: python ; coding: utf-8 -*-
#
# VisionMante v2 — PyInstaller spec
# Genera un bundle onedir en dist/VisionMante/
#
# NOTA: config/ y data/ NO se incluyen aquí;
# el script release.bat los copia al lado del .exe después de compilar.

import sys
import os
import glob

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

# ── python3XX.dll — copiar explícitamente al nivel raíz del exe ───────────────
# El bootloader de PyInstaller puede quedar con un path absoluto del directorio
# de build embebido; copiar la DLL junto al .exe garantiza que siempre la encuentre.
_py_ver = f"python3{sys.version_info.minor}.dll"

# En un venv, sys.executable apunta a env1\Scripts\python.exe
# La DLL real está en el Python base (sys.base_prefix)
_base_prefix = getattr(sys, 'base_prefix', sys.prefix)
_dll_candidates = (
    glob.glob(os.path.join(_base_prefix, _py_ver)) +
    glob.glob(os.path.join(os.path.dirname(sys.executable), _py_ver)) +
    glob.glob(os.path.join(sys.prefix, _py_ver))
)
_python_dll = next((p for p in _dll_candidates if os.path.exists(p)), None)

if _python_dll:
    print(f"[spec] python DLL encontrada: {_python_dll}")
    _dll_binaries = [(_python_dll, '.')]   # '.' = junto al .exe
else:
    print(f"[spec] AVISO: no se encontro {_py_ver} — el exe puede fallar al arrancar")
    _dll_binaries = []

# ── Binarios nativos ──────────────────────────────────────────────────────────
# onnxruntime DLLs — necesarias para inferencia YOLO
ort_dlls = collect_dynamic_libs('onnxruntime')

# snap7.dll — requerida si se usa PLC Siemens.
# Colócala manualmente en dist/VisionMante/ si usas PLC Siemens.
extra_binaries = ort_dlls + _dll_binaries

# ── Datos embebidos (solo recursos de solo lectura) ───────────────────────────
# cv2 datos internos (cascades, haarcascades, etc.)
cv2_datas = collect_data_files('cv2')

all_datas = cv2_datas

# ── Hidden imports ────────────────────────────────────────────────────────────
hidden = [
    # pydantic v2
    'pydantic',
    'pydantic.v1',
    'pydantic_core',
    # config
    'tomllib',
    'tomli_w',
    # logging
    'loguru',
    # PLC
    'snap7',
    'snap7.client',
    'snap7.type',
    'snap7.util',
    'snap7.util.getters',
    'snap7.util.setters',
    'pymcprotocol',
    # vision
    'cv2',
    'onnxruntime',
    'onnxruntime.capi',
    # Qt
    'PyQt6',
    'PyQt6.QtCore',
    'PyQt6.QtWidgets',
    'PyQt6.QtGui',
    'PyQt6.sip',
    # stdlib (por si acaso con frozen)
    'pathlib',
    'ctypes',
    'ctypes.util',
]

a = Analysis(
    ['main.py'],
    pathex=['.'],
    binaries=extra_binaries,
    datas=all_datas,
    hiddenimports=hidden,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # No se usa en runtime — solo para entrenar/exportar modelos
        'ultralytics',
        # torch ecosystem (pesado, no necesario con onnxruntime)
        'torch',
        'torchvision',
        'torchaudio',
        'tensorflow',
        # GUI alternativas
        'tkinter',
        'wx',
        # Dev / ciencia de datos
        'pytest',
        'matplotlib',
        'pandas',
        'scipy',
        'IPython',
        'jupyter',
        'notebook',
    ],
    noarchive=False,
    optimize=1,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='VisionMante',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=['vcruntime140.dll', 'msvcp140.dll'],
    console=False,          # Sin ventana de consola
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    # icon='assets/icon.ico',  # Descomenta si tienes un icono
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='VisionMante',
)
