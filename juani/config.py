"""
config.py — Centralización de constantes y configuración del pipeline.

Elimina números mágicos dispersos y provee un único objeto de configuración
(PipelineConfig) en lugar de pasar 18+ parámetros sueltos entre funciones.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


# ── Renderizado ───────────────────────────────────────────────────────────────

RENDER_PAD_PX               = 48       # padding en px alrededor del contenido
RENDER_MIN_SYM_PX           = 32       # mínimo de px por símbolo antes de ajustar escala
RENDER_MAX_DIM_PX           = 16_000   # límite de px por lado (protección de memoria)
RENDER_FILTER_BUFFER_FACTOR = 8.0      # buffer del filtro espacial (× tamaño símbolo CAD)

# ── Análisis de escala ────────────────────────────────────────────────────────

SCALE_TARGET_SYMBOL_PX = 64    # tamaño objetivo de un símbolo INSERT en px
SCALE_TARGET_TEXT_PX   = 16    # tamaño objetivo de texto en px
SCALE_OUTLIER_FACTOR   = 5.0   # descarta INSERTs > mediana × este factor
SCALE_EXCLUDED_LAYERS  = frozenset({
    "CAJETIN", "TITLE", "TITULO", "BORDE", "FRAME", "DEFPOINTS", "VIEWPORT",
})

# ── Detección / post-proceso ──────────────────────────────────────────────────

DETECT_CONF_RAW         = 0.25   # umbral interno SAHI (bajo → mejor recall en tiles)
DETECT_CONF_MIN         = 0.50   # filtro final post-ensemble
DETECT_IOU_NMS          = 0.50   # IoU para NMS agnóstico de clase
DETECT_IOS_NESTED       = 0.70   # IOS para suprimir cajas anidadas
DETECT_BORDER_MARGIN_PX = 2      # px de margen para detectar bbox pegado al borde
DETECT_BORDER_CONF_SAFE = 0.90   # conf a partir de la cual no se descarta por borde
DETECT_SLICE_SIZE       = 640    # tamaño de tile en px
DETECT_SLICE_OVERLAP    = 0.20   # solapamiento entre tiles
DETECT_SLICE_MIN        = 160    # slice_size mínimo de seguridad
DETECT_SLICE_MAX        = 1_280  # slice_size máximo de seguridad
DETECT_BATCH_SIZE       = 16     # tiles por batch de inferencia


@dataclass
class PipelineConfig:
    """
    Objeto único de configuración que reemplaza los 18+ parámetros sueltos
    de pipeline_v2.py. Se construye en el CLI y se pasa a Pipeline/EnsemblePipeline.

    Separar configuración de lógica (DIP) permite cambiar parámetros sin
    tocar las clases que los consumen.
    """
    dxf_path:           str
    output_dir:         str             = "./pipeline_out"
    model_paths:        List[str]       = field(default_factory=list)

    # render
    target_px:          int             = SCALE_TARGET_SYMBOL_PX
    max_dim_px:         int             = RENDER_MAX_DIM_PX
    layers_include:     Optional[List[str]] = None
    color_mode:         str             = "color"   # color | grayscale | binary | mono

    # detection
    conf_raw:           float           = DETECT_CONF_RAW
    conf_min:           float           = DETECT_CONF_MIN
    iou_nms:            float           = DETECT_IOU_NMS
    ios_nested:         float           = DETECT_IOS_NESTED
    border_margin_px:   int             = DETECT_BORDER_MARGIN_PX
    border_conf_safe:   float           = DETECT_BORDER_CONF_SAFE
    slice_size:         int             = DETECT_SLICE_SIZE
    overlap:            float           = DETECT_SLICE_OVERLAP
    zoom:               float           = 1.0
    min_tiles_per_axis: Optional[int]   = None
    batch_size:         int             = DETECT_BATCH_SIZE
    save_slices:        bool            = False
    device:             str             = "cpu"
    model_type:         str             = "yolov8"
