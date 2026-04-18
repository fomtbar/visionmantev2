"""Helper para release.bat: imprime la ruta de python3XX.dll del Python base."""
import sys
import os
import glob

base = getattr(sys, 'base_prefix', sys.prefix)
pattern = os.path.join(base, f'python3{sys.version_info.minor}.dll')
matches = glob.glob(pattern)
if matches:
    print(matches[0])
