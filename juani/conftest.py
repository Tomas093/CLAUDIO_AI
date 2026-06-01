"""
conftest.py — Configuración de pytest para el paquete juani.

Inserta el directorio raíz de juani/ al inicio de sys.path para que los
imports internos (from detection.coordinates import ...) resuelvan correctamente
desde este paquete y no desde el directorio padre de CLAUDIO_AI.
"""
import sys
from pathlib import Path

# Asegura que juani/ esté al INICIO del path, antes que CLAUDIO_AI/
# para que nuestros módulos tengan prioridad sobre los del directorio padre.
juani_root = str(Path(__file__).parent)
if juani_root not in sys.path:
    sys.path.insert(0, juani_root)
