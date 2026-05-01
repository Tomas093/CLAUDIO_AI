# config.py
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# ── Rutas de entrada ──────────────────────────────────────────────────────
INPUT_DIR        = BASE_DIR / "input"
DXF_FILE         = INPUT_DIR / "component.dxf"
BACKGROUNDS_DIR  = INPUT_DIR / "backgrounds"

# ── Rutas de salida ───────────────────────────────────────────────────────
OUTPUT_DIR       = BASE_DIR / "output"
SPRITES_DIR      = OUTPUT_DIR / "sprites"
SYNTHETIC_DIR    = OUTPUT_DIR / "synthetic"
DATASET_DIR      = BASE_DIR / "dataset"

# ── Fase 1: Generación de Sprites ─────────────────────────────────────────
N_SPRITE_VARIATIONS = 50    # Número de variaciones de grosor de línea
DILATION_KERNEL_MIN = 1     # Kernel mínimo (línea delgada, sin cambio)
DILATION_KERNEL_MAX = 9     # Kernel máximo (línea gruesa)
RENDER_DPI          = 200   # DPI del renderizado matplotlib
BINARIZE_THRESHOLD  = 200   # Umbral para separar líneas del fondo blanco

# ── Fase 2/3: Generación de Imágenes Sintéticas ───────────────────────────
N_SYNTHETIC_TOTAL      = 500  # Total de imágenes sintéticas a generar
SPRITE_SCALE_MIN       = 0.08 # Tamaño mínimo del sprite (fracción del ancho del BG)
SPRITE_SCALE_MAX       = 0.25 # Tamaño máximo del sprite
COMPONENTS_PER_IMG_MIN = 1    # Mínimo de componentes por imagen
COMPONENTS_PER_IMG_MAX = 3    # Máximo de componentes por imagen
ALLOW_RANDOM_ROTATION  = True # Rota el sprite antes de pegarlo al plano

# ── Fase 4: Ensamblaje del Dataset ────────────────────────────────────────
TRAIN_RATIO    = 0.80   # 80% entrenamiento
NEGATIVE_RATIO = 0.15   # 15% del dataset total serán negativos (sin componente)
CLASS_ID       = 0      # Índice de clase YOLO
CLASS_NAME     = "componente_electrico"  # Nombre del componente (usado en data.yaml)