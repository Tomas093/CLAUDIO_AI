# Refactorización del Pipeline CLAUDIO_AI
> Análisis completo con perspectiva production-ready: SOLID, bajo acoplamiento, alta cohesión.

---

## 1. Mapa del problema actual

### Anti-patterns identificados

#### 🔴 CRÍTICO

**1. Versionado en nombres de archivo**
`dxf_to_image.py` → `dxf_to_image_v2.py`, `scale_analyzer.py` → `scale_analyzer_v2.py`, `pipeline.py` → `pipeline_v2.py`.
Esto no escala. ¿Cuándo aparece `_v3`? ¿Cuál usar? El código cliente no puede abstraerse del "qué versión".
El correcto: una sola clase `DxfRenderer` con configuración, y Git para el historial.

**2. God function: `ejecutar()` en `inference_sahi.py`**
Una sola función hace: cargar modelo, calcular slice_size, guardar slices, hacer inferencia en batch, aplicar 4 filtros diferentes, guardar JSON, dibujar PNG, y loggear todo. Viola SRP en cada línea.

**3. Duplicación masiva: `correr_pipeline` vs `correr_pipeline_ensemble`**
Las dos funciones comparten ~80% del código. La diferencia es que ensemble itera sobre modelos y acumula detecciones. Esto es un caso textbook de Template Method o composición. Cualquier fix (añadir un filtro, cambiar el path de output) se tiene que hacer dos veces.

**4. Doble lectura del DXF**
`spec_extractor.py::extraer_textos_dxf()` abre el mismo archivo DXF que el pipeline ya parseó con `ezdxf.readfile()`. En planos grandes esto duplica el tiempo de I/O sin necesidad.

**5. Lógica de coordenadas duplicada**
La conversión px→CAD aparece en `inference_sahi.py::correr_sahi_batched()`, en `inference_sahi.py::mapear_a_cad()`, y en `spec_extractor.py::_bbox_cad()`. Tres implementaciones del mismo algoritmo — cuando hay un bug, hay que corregirlo en tres lugares.

#### 🟡 IMPORTANTE

**6. Números mágicos dispersos en todo el código**
```python
# inference_sahi.py
ios_thresh=0.7    # ← ¿por qué 0.7 y no 0.75?
margen_px=2       # ← constante sin nombre
conf_safe=0.9     # ← umbral "safe" sin contexto

# dxf_to_image_v2.py
PAD_PX = 48
MIN_SYM_PX = 32
FILTRO_BUFFER_FACTOR = 8.0

# scale_analyzer_v2.py
mask = arr <= mediana * 5   # ← ¿por qué 5?
```
Todos deberían vivir en un único `config.py` con nombres descriptivos y documentación.

**7. Logging con `print()` sin estructura**
No hay niveles (DEBUG/INFO/WARNING), no hay timestamps, no hay forma de silenciar el output en tests. Imposible filtrar qué módulo generó cada mensaje.

**8. `run.py` con paths hardcodeados**
```python
ruta_modelo = r"C:\Users\juani\faculty\4Ano\CLAUDIO_AI\dxf\best.pt"
```
Código muerto con rutas absolutas personales. No tiene lugar en el repo.

**9. `bare except: pass` como estrategia de error**
`scale_analyzer_v2.py::_block_bbox()` tiene 6 `except: pass`. Silencia errores reales sin traza. Si un bloque DXF falla misteriosamente, no hay forma de saberlo.

**10. Parámetros repetidos en cascada**
`correr_pipeline_ensemble()` acepta 18 parámetros. Los pasa todos a sub-funciones que a su vez los pasan a otras. Esto es un code smell clásico: falta un objeto de configuración.

#### 🟢 MENOR

**11. `scale_analyzer_v2` ignora el `if/elif` implícito como Strategy**
La lógica de "INSERT → CIRCLE → TEXT → BBOX" es un patrón Strategy sin nombre, mezclado con la función principal.

**12. `dxf_to_image_v2.py` mezcla 6 responsabilidades en una función**
`renderizar_dxf()` hace: lectura DXF, análisis de escala, manipulación de capas, política de color, filtro espacial, cálculo de bounds, render matplotlib, post-procesado PIL, y serialización de metadata.

---

## 2. Estructura de carpetas propuesta

```
claudio_ai/
│
├── config.py                       # ← NUEVO: toda la config centralizada
│
├── rendering/
│   ├── __init__.py
│   ├── renderer.py                 # DxfRenderer — orquesta el render
│   ├── color_policy.py             # Protocol + implementaciones (TextWhite, TextBlack, Mono)
│   ├── spatial_filter.py           # SpatialFilter — filtra entidades fuera de región
│   └── scale/
│       ├── __init__.py
│       ├── analyzer.py             # ScaleAnalyzer — corre strategies en orden
│       └── strategies.py           # InsertStrategy, CircleStrategy, TextStrategy, BboxFallback
│
├── detection/
│   ├── __init__.py
│   ├── model.py                    # DetectionModel — wrappea SAHI/YOLO
│   ├── slicer.py                   # ImageSlicer — slicing + batch inference
│   ├── postprocess.py              # PostProcessorChain — NMS, border, nested, conf_min
│   └── coordinates.py              # CoordinateMapper — única fuente de verdad px↔CAD
│
├── extraction/
│   ├── __init__.py
│   └── spec_extractor.py           # SpecExtractor — extrae specs de texto
│
├── pipeline/
│   ├── __init__.py
│   ├── base.py                     # Pipeline — orquestador principal
│   └── ensemble.py                 # EnsemblePipeline — extiende Pipeline
│
├── io/
│   ├── __init__.py
│   ├── result_writer.py            # ResultWriter — guarda JSON, copia archivos
│   └── visualizer.py              # Visualizer — dibuja bboxes sobre imagen
│
├── cli.py                          # Único punto de entrada CLI
│
└── tests/
    ├── test_coordinates.py
    ├── test_postprocess.py
    ├── test_spec_extractor.py
    ├── test_scale_analyzer.py
    └── test_pipeline_integration.py
```

---

## 3. Diseño módulo por módulo

### 3.1 `config.py` — Configuración centralizada

**Por qué:** Elimina todos los números mágicos. Un solo lugar para cambiar parámetros. Hace el código autodocumentado.

```python
# config.py
from dataclasses import dataclass, field
from typing import Optional, List


# ── Render ────────────────────────────────────────────────────────────────────
RENDER_PAD_PX         = 48     # padding en pixeles alrededor del contenido
RENDER_MIN_SYM_PX     = 32     # mínimo de px para un símbolo antes de escalar
RENDER_MAX_DIM_PX     = 16_000 # límite de px por lado (protección de memoria)
RENDER_FILTER_BUFFER  = 8.0    # factor de buffer para filtro espacial (× tamaño símbolo)

# ── Scale analyzer ────────────────────────────────────────────────────────────
SCALE_TARGET_SYMBOL_PX  = 64   # tamaño objetivo de un símbolo en px
SCALE_TARGET_TEXT_PX    = 16   # tamaño objetivo de texto en px
SCALE_OUTLIER_FACTOR    = 5.0  # filtro de bloques gigantes: max = mediana × este factor
SCALE_EXCLUDED_LAYERS   = frozenset({
    "CAJETIN", "TITLE", "TITULO", "BORDE", "FRAME", "DEFPOINTS", "VIEWPORT"
})

# ── Detection / postprocess ───────────────────────────────────────────────────
DETECT_CONF_RAW         = 0.25  # umbral interno SAHI (bajo → mejor recall en tiles)
DETECT_CONF_MIN         = 0.50  # filtro final post-ensemble
DETECT_IOU_NMS          = 0.50  # IoU para NMS agnóstico de clase
DETECT_IOS_NESTED       = 0.70  # IOS para suprimir cajas anidadas
DETECT_BORDER_MARGIN_PX = 2     # margen para detectar cajas pegadas al borde
DETECT_BORDER_CONF_SAFE = 0.90  # confianza a partir de la cual no se descarta por borde
DETECT_SLICE_SIZE       = 640
DETECT_SLICE_OVERLAP    = 0.20
DETECT_SLICE_MIN        = 160
DETECT_SLICE_MAX        = 1_280
DETECT_BATCH_SIZE       = 16


@dataclass
class PipelineConfig:
    """Objeto de configuración que reemplaza los 18 parámetros sueltos."""
    dxf_path:           str
    output_dir:         str          = "./pipeline_out"
    model_paths:        List[str]    = field(default_factory=list)

    # render
    target_px:          int          = SCALE_TARGET_SYMBOL_PX
    max_dim_px:         int          = RENDER_MAX_DIM_PX
    layers_include:     Optional[List[str]] = None
    color_mode:         str          = "color"   # color | grayscale | binary | mono

    # detection
    conf_raw:           float        = DETECT_CONF_RAW
    conf_min:           float        = DETECT_CONF_MIN
    iou_nms:            float        = DETECT_IOU_NMS
    slice_size:         int          = DETECT_SLICE_SIZE
    overlap:            float        = DETECT_SLICE_OVERLAP
    zoom:               float        = 1.0
    min_tiles_per_axis: Optional[int]= None
    batch_size:         int          = DETECT_BATCH_SIZE
    save_slices:        bool         = False
    device:             str          = "cpu"
    model_type:         str          = "yolov8"
```

---

### 3.2 `rendering/scale/strategies.py` — Strategy para escala

**Por qué:** La cadena `if/elif` oculta el patrón Strategy. Con Protocol explícito es fácil añadir una nueva estrategia (e.g., basada en dimensiones del cajetín) sin tocar el analizador.

```python
# rendering/scale/strategies.py
from typing import Protocol, Optional, Tuple
import numpy as np
import ezdxf

ScaleResult = Tuple[float, str]   # (px_per_cad, descripcion)


class ScaleStrategy(Protocol):
    def estimate(self, msp, doc) -> Optional[ScaleResult]:
        """Retorna (px_per_cad, descripcion) o None si no aplica."""
        ...


class InsertScaleStrategy:
    """Usa la mediana del lado mayor de los INSERTs significativos."""
    def __init__(self, target_px: int, excluded_layers, outlier_factor: float):
        self.target_px       = target_px
        self.excluded_layers = excluded_layers
        self.outlier_factor  = outlier_factor

    def estimate(self, msp, doc) -> Optional[ScaleResult]:
        sizes = self._collect_sizes(msp, doc)
        if not sizes:
            return None
        arr     = np.array(sizes)
        median  = float(np.median(arr))
        arr_f   = arr[arr <= median * self.outlier_factor]
        tam     = float(np.median(arr_f))
        return (self.target_px / tam, f"INSERT_lado={tam:.4f}")

    def _collect_sizes(self, msp, doc):
        ...   # lógica extraída de scale_analyzer_v2.analizar_inserts_v2


class CircleScaleStrategy:
    """Usa la mediana del radio de círculos (filtrado IQR)."""
    ...


class TextScaleStrategy:
    """Usa la moda de altura de texto."""
    ...


class BboxFallbackStrategy:
    """Último recurso: escala para que el plano completo entre en 8000px."""
    def estimate(self, msp, doc) -> Optional[ScaleResult]:
        # siempre retorna algo
        ...
```

---

### 3.3 `rendering/color_policy.py` — Policy para color de texto

**Por qué:** Actualmente la lógica de "texto negro vs blanco" está inline en `renderizar_dxf()` con un parámetro booleano `texto_negro`. Extraerla como Policy permite añadir futuros modos (e.g., texto por capa) sin tocar el renderer.

```python
# rendering/color_policy.py
from typing import Protocol
import ezdxf


class ColorPolicy(Protocol):
    def apply(self, doc: ezdxf.document.Drawing) -> None:
        """Modifica el doc en memoria para aplicar la política de color."""
        ...


class TextInvisiblePolicy:
    """Fuerza texto blanco (invisible sobre fondo blanco) — para render de detección."""
    def apply(self, doc):
        for e in doc.modelspace():
            if e.dxftype() in ("TEXT", "MTEXT", "ATTRIB", "ATTDEF"):
                try: e.rgb = (255, 255, 255)
                except: pass
            if e.dxftype() == "INSERT":
                try:
                    for att in e.attribs: att.rgb = (255, 255, 255)
                except: pass


class TextBlackPolicy:
    """Fuerza texto negro — para render de visualización humana."""
    def apply(self, doc):
        for e in doc.modelspace():
            if e.dxftype() in ("TEXT", "MTEXT", "ATTRIB", "ATTDEF"):
                try: e.rgb = (0, 0, 0)
                except: pass
            if e.dxftype() == "INSERT":
                try:
                    for att in e.attribs: att.rgb = (0, 0, 0)
                except: pass


class MonoPolicy:
    """Fuerza todo a negro (para entrenamiento en escala de grises)."""
    def apply(self, doc):
        for layer in doc.layers:
            try: layer.color = 7
            except: pass
        for e in doc.modelspace():
            try:
                if hasattr(e.dxf, "color"): e.dxf.color = 256
            except: pass


class NoOpPolicy:
    """No modifica nada — render en color original."""
    def apply(self, doc):
        pass
```

---

### 3.4 `detection/coordinates.py` — Fuente única de verdad para coordenadas

**Por qué:** La misma conversión px↔CAD existe en 3 lugares. Si el sistema de coordenadas cambia, sólo se edita aquí.

```python
# detection/coordinates.py
from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class CanvasMeta:
    """Metadatos del canvas de renderizado. Inmutable para evitar mutaciones accidentales."""
    px_per_cad:        float
    x_min_cad:         float
    y_min_cad:         float
    x_max_cad:         float
    y_max_cad:         float
    image_width_px:    int
    image_height_px:   int

    @classmethod
    def from_dict(cls, d: dict) -> "CanvasMeta":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__})


class CoordinateMapper:
    """
    Única fuente de verdad para conversiones entre espacio de píxeles y CAD.

    Convención:
      - Píxeles: origen en esquina superior-izquierda, Y crece hacia abajo.
      - CAD:     origen en esquina inferior-izquierda, Y crece hacia arriba.
    """
    def __init__(self, meta: CanvasMeta):
        self._meta = meta

    def px_to_cad(self, cx_px: float, cy_px: float) -> Tuple[float, float]:
        m = self._meta
        x_cad = m.x_min_cad + cx_px / m.px_per_cad
        y_cad = m.y_max_cad - cy_px / m.px_per_cad
        return x_cad, y_cad

    def bbox_px_to_cad(self, x1: float, y1: float, x2: float, y2: float
                       ) -> Tuple[float, float, float, float]:
        """Retorna (x_min_cad, y_min_cad, x_max_cad, y_max_cad)."""
        m = self._meta
        x_min_cad = m.x_min_cad + x1 / m.px_per_cad
        x_max_cad = m.x_min_cad + x2 / m.px_per_cad
        # Y invertido: y1_px (arriba) → y_max_cad, y2_px (abajo) → y_min_cad
        y_max_cad = m.y_min_cad + (m.image_height_px - y1) / m.px_per_cad
        y_min_cad = m.y_min_cad + (m.image_height_px - y2) / m.px_per_cad
        return x_min_cad, y_min_cad, x_max_cad, y_max_cad
```

---

### 3.5 `detection/postprocess.py` — Chain of Responsibility para filtros

**Por qué:** Los 4 filtros actuales (`border`, `nms`, `nested`, `conf_min`) están pegados con `print()` y lógica propia en `ejecutar()`. Chain of Responsibility permite reordenarlos, desactivarlos, o añadir nuevos sin tocar el pipeline.

```python
# detection/postprocess.py
from typing import List, Protocol
import logging

logger = logging.getLogger(__name__)

Detection = dict   # {clase, conf, bbox_px, centro_px, x_cad, y_cad}


class DetectionFilter(Protocol):
    def filter(self, detections: List[Detection], **ctx) -> List[Detection]: ...


class BorderFilter:
    def __init__(self, margin_px: int = 2, conf_safe: float = 0.9):
        self.margin_px = margin_px
        self.conf_safe = conf_safe

    def filter(self, detections, *, meta, **_):
        W, H = meta["image_width_px"], meta["image_height_px"]
        before = len(detections)
        kept = [
            d for d in detections
            if not self._touches_border(d["bbox_px"], W, H) or d["conf"] >= self.conf_safe
        ]
        logger.info("BorderFilter: %d → %d", before, len(kept))
        return kept

    def _touches_border(self, bbox, W, H):
        x1, y1, x2, y2 = bbox
        m = self.margin_px
        return x1 <= m or y1 <= m or x2 >= W - m or y2 >= H - m


class AgnosticNMSFilter:
    def __init__(self, iou_thresh: float = 0.5): ...
    def filter(self, detections, **_): ...   # lógica de nms_agnostico_clase


class NestedBoxFilter:
    def __init__(self, ios_thresh: float = 0.7): ...
    def filter(self, detections, **_): ...   # lógica de eliminar_anidadas


class ConfidenceFilter:
    def __init__(self, conf_min: float = 0.5): ...
    def filter(self, detections, **_): ...   # lógica de filtrar_por_confianza


class PostProcessorChain:
    """Aplica filtros en secuencia. Fácil de reconfigurar o extender."""
    def __init__(self, filters: List[DetectionFilter]):
        self._filters = filters

    @classmethod
    def default(cls, cfg) -> "PostProcessorChain":
        return cls([
            BorderFilter(cfg.border_margin_px, cfg.border_conf_safe),
            AgnosticNMSFilter(cfg.iou_nms),
            NestedBoxFilter(cfg.ios_nested),
            ConfidenceFilter(cfg.conf_min),
        ])

    def run(self, detections: List[Detection], **ctx) -> List[Detection]:
        for f in self._filters:
            detections = f.filter(detections, **ctx)
        return detections
```

---

### 3.6 `pipeline/base.py` — Pipeline unificado (elimina la duplicación)

**Por qué:** Las dos funciones `correr_pipeline` y `correr_pipeline_ensemble` deben ser una sola clase. El caso ensemble es simplemente "correr el paso de inferencia con N modelos en vez de 1".

```python
# pipeline/base.py
import logging
from pathlib import Path
from typing import List

from config import PipelineConfig
from rendering.renderer import DxfRenderer
from rendering.color_policy import TextInvisiblePolicy, TextBlackPolicy
from detection.model import DetectionModel
from detection.slicer import ImageSlicer
from detection.postprocess import PostProcessorChain
from detection.coordinates import CanvasMeta, CoordinateMapper
from extraction.spec_extractor import SpecExtractor
from io.result_writer import ResultWriter
from io.visualizer import Visualizer

logger = logging.getLogger(__name__)


class Pipeline:
    """
    Orquestador principal. Separa claramente las etapas:
      1. Render (detección)  → imagen sin texto visible
      2. Inferencia          → detecciones crudas
      3. Post-proceso        → filtros en cadena
      4. Render (viz)        → imagen con texto negro
      5. Visualización       → PNG con bboxes
      6. Extracción de specs → JSON enriquecido
      7. Escritura           → archivos de salida
    """
    def __init__(self, cfg: PipelineConfig):
        self.cfg       = cfg
        self.renderer  = DxfRenderer(cfg)
        self.slicer    = ImageSlicer(cfg)
        self.postproc  = PostProcessorChain.default(cfg)
        self.extractor = SpecExtractor()
        self.writer    = ResultWriter(cfg.output_dir)
        self.viz       = Visualizer()

    def run(self) -> dict:
        out = Path(self.cfg.output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # ── Etapa 1: render limpio para detección ────────────────────────────
        logger.info("Etapa 1: render de detección")
        detect_img, meta = self.renderer.render(
            self.cfg.dxf_path,
            out / "plano_render.png",
            color_policy=TextInvisiblePolicy(),
        )
        canvas = CanvasMeta.from_dict(meta)

        # ── Etapa 2: inferencia ───────────────────────────────────────────────
        logger.info("Etapa 2: inferencia")
        raw_dets = self._run_inference(detect_img, meta)

        # ── Etapa 3: post-proceso ─────────────────────────────────────────────
        logger.info("Etapa 3: post-proceso (%d detecciones crudas)", len(raw_dets))
        dets = self.postproc.run(raw_dets, meta=meta)
        logger.info("Post-proceso: %d detecciones finales", len(dets))

        # ── Etapa 4: render de visualización ─────────────────────────────────
        logger.info("Etapa 4: render de visualización")
        viz_img, _ = self.renderer.render(
            self.cfg.dxf_path,
            out / "plano_render_viz.png",
            color_policy=TextBlackPolicy(),
        )

        # ── Etapa 5: visualización con bboxes ────────────────────────────────
        self.viz.draw(viz_img, dets, out / "deteccion_visual.png")

        # ── Etapa 6: extracción de specs ──────────────────────────────────────
        logger.info("Etapa 6: extracción de specs")
        dets_with_specs = self.extractor.extract(
            self.cfg.dxf_path, dets, canvas
        )

        # ── Etapa 7: escritura de resultados ──────────────────────────────────
        self.writer.write(dets, dets_with_specs, meta)

        return {"detections": dets_with_specs, "meta": meta}

    def _run_inference(self, image_path, meta) -> list:
        """Hook para subclases (ej: EnsemblePipeline)."""
        model = DetectionModel.load(self.cfg.model_paths[0], self.cfg)
        return self.slicer.run_batched(model, image_path, meta)


class EnsemblePipeline(Pipeline):
    """
    Extiende Pipeline para correr múltiples modelos y agregar sus detecciones.
    Sólo sobreescribe el paso de inferencia — todo lo demás es idéntico.
    """
    def _run_inference(self, image_path, meta) -> list:
        all_dets = []
        for i, model_path in enumerate(self.cfg.model_paths, 1):
            logger.info("Ensemble: modelo %d/%d: %s", i, len(self.cfg.model_paths), model_path)
            model = DetectionModel.load(model_path, self.cfg)
            dets  = self.slicer.run_batched(model, image_path, meta)
            logger.info("  → %d detecciones crudas", len(dets))
            all_dets.extend(dets)
        logger.info("Ensemble: total crudas: %d", len(all_dets))
        return all_dets
```

**Resultado de la refactorización del pipeline:**
- De 260 líneas duplicadas → 30 líneas en `Pipeline` + 8 en `EnsemblePipeline`.
- Añadir un nuevo paso (e.g., OCR post-detección) = 3 líneas en `run()`.
- Reemplazar el renderer = cambiar 1 import.

---

### 3.7 `extraction/spec_extractor.py` — Sin re-lectura del DXF

**Por qué:** La versión actual abre el archivo DXF una segunda vez. El `SpecExtractor` debe recibir el doc ya cargado desde el pipeline.

```python
# extraction/spec_extractor.py
import logging
from typing import List, Dict, Any
import ezdxf
from detection.coordinates import CanvasMeta, CoordinateMapper

logger = logging.getLogger(__name__)


class SpecExtractor:
    def extract(
        self,
        dxf_path: str,
        detections: List[Dict],
        canvas: CanvasMeta,
    ) -> List[Dict[str, Any]]:
        """
        Enriquece cada detección con los textos DXF que le corresponden.
        Recibe el canvas para reusar el CoordinateMapper — no vuelve a
        leer el DXF si el pipeline ya lo tiene en memoria (ver nota abajo).
        """
        doc = ezdxf.readfile(dxf_path)   # ← ver mejora futura en sección 5
        texts = self._collect_texts(doc.modelspace())
        logger.info("SpecExtractor: %d textos encontrados", len(texts))

        mapper = CoordinateMapper(canvas)
        return [
            {**det, "specs": self._assign_specs(det, detections, texts, mapper)}
            for det in detections
        ]

    def _collect_texts(self, msp) -> List[Dict]:
        ...   # extraer_textos_dxf actual

    def _assign_specs(self, det, all_dets, texts, mapper) -> List[str]:
        ...   # asignar_specs actual, usando mapper.bbox_px_to_cad() en vez de _bbox_cad()
```

**Nota sobre la doble lectura:** La solución ideal es que el `DxfRenderer` devuelva el `doc` junto con la imagen y metadata, y el `Pipeline` lo pase al `SpecExtractor`. Esto requiere que `DxfRenderer` no cierre el doc al terminar el render. Implementación en fases: primero extraer la clase, luego eliminar la re-lectura.

---

### 3.8 `cli.py` — Punto de entrada único

**Por qué:** Actualmente hay 5 archivos con `if __name__ == "__main__"`, cada uno con su propio `argparse`. Un CLI unificado con subcomandos es más mantenible.

```python
# cli.py
import argparse
import logging
from config import PipelineConfig
from pipeline.base import Pipeline, EnsemblePipeline


def cmd_run(args):
    cfg = PipelineConfig(
        dxf_path    = args.dxf,
        output_dir  = args.out,
        model_paths = [args.modelo] if args.modelo else args.modelos,
        target_px   = args.target_px,
        conf_min    = args.conf_min,
        # ...
    )
    pipeline = EnsemblePipeline(cfg) if len(cfg.model_paths) > 1 else Pipeline(cfg)
    pipeline.run()


def cmd_render(args):
    from rendering.renderer import DxfRenderer
    ...


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s"
    )
    parser = argparse.ArgumentParser(prog="claudio")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # claudio run --dxf plano.dxf --modelos best1.pt best2.pt
    p_run = sub.add_parser("run", help="Detectar componentes en un DXF")
    p_run.add_argument("--dxf",        required=True)
    p_run.add_argument("--modelo",     default=None)
    p_run.add_argument("--modelos",    nargs="+", default=None)
    p_run.add_argument("--out",        default="./pipeline_out")
    p_run.add_argument("--target-px",  type=int,   default=64)
    p_run.add_argument("--conf-min",   type=float, default=0.5)
    # ...
    p_run.set_defaults(func=cmd_run)

    # claudio render --dxf plano.dxf --out render.png
    p_render = sub.add_parser("render", help="Solo renderizar un DXF")
    p_render.set_defaults(func=cmd_render)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
```

**Uso después de refactorizar:**
```bash
# Antes (confuso, ¿cuál pipeline usar?):
python pipeline_v2.py --dxf plano.dxf --modelos m1.pt m2.pt

# Después (claro y unificado):
python cli.py run --dxf plano.dxf --modelos m1.pt m2.pt
```

---

## 4. Logging estructurado

Reemplazar todos los `print()` por:

```python
import logging
logger = logging.getLogger(__name__)

# En vez de:
print(f"[v2/read  ] {dxf_path}  ({t:.1f}s)")

# Usar:
logger.info("DXF leído: %s (%.1fs)", dxf_path, t)
```

Configuración centralizada en `cli.py`:
```python
logging.basicConfig(
    level=logging.DEBUG if args.verbose else logging.INFO,
    format="%(asctime)s [%(name)-20s] %(levelname)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("claudio.log"),  # opcional
    ]
)
```

**Beneficios:** silenciar en tests con `logging.disable()`, filtrar por módulo, guardar a archivo sin cambiar código.

---

## 5. Tests unitarios necesarios

### `tests/test_coordinates.py` — crítico, lógica matemática
```python
def test_px_to_cad_identity():
    meta = CanvasMeta(px_per_cad=10.0, x_min_cad=0, y_min_cad=0,
                      x_max_cad=100, y_max_cad=100,
                      image_width_px=1000, image_height_px=1000)
    mapper = CoordinateMapper(meta)
    # origen px → esquina inferior-izquierda CAD
    x, y = mapper.px_to_cad(0, 1000)
    assert x == pytest.approx(0.0)
    assert y == pytest.approx(0.0)

def test_y_axis_inversion():
    meta = CanvasMeta(px_per_cad=1.0, x_min_cad=0, y_min_cad=0,
                      x_max_cad=100, y_max_cad=100,
                      image_width_px=100, image_height_px=100)
    mapper = CoordinateMapper(meta)
    _, y_top = mapper.px_to_cad(0, 0)    # pixel arriba → CAD alto
    _, y_bot = mapper.px_to_cad(0, 100)  # pixel abajo → CAD bajo
    assert y_top > y_bot
```

### `tests/test_postprocess.py`
```python
def test_confidence_filter_removes_low_conf():
    dets = [{"conf": 0.8}, {"conf": 0.3}, {"conf": 0.6}]
    result = ConfidenceFilter(conf_min=0.5).filter(dets)
    assert len(result) == 2
    assert all(d["conf"] >= 0.5 for d in result)

def test_chain_is_composable():
    chain = PostProcessorChain([ConfidenceFilter(0.5), NestedBoxFilter(0.7)])
    # verificar que se aplican en orden correcto
    ...
```

### `tests/test_spec_extractor.py`
```python
def test_gap_cutting_stops_at_large_gap():
    # dado un componente de alto=2.0 y candidatos en x=5, 6, 10
    # el gap 6→10 es > alto (2.0), por lo que sólo toma x=5,6
    ...

def test_no_leak_to_right_component():
    # si hay otro componente detectado a la derecha, x_limite lo bloquea
    ...
```

### `tests/test_pipeline_integration.py`
```python
@pytest.fixture
def minimal_dxf(tmp_path):
    # crear un DXF mínimo con ezdxf programáticamente
    doc = ezdxf.new()
    msp = doc.modelspace()
    msp.add_line((0, 0, 0), (10, 10, 0))
    path = tmp_path / "test.dxf"
    doc.saveas(path)
    return path

def test_pipeline_runs_without_error(minimal_dxf, tmp_path):
    cfg = PipelineConfig(
        dxf_path    = str(minimal_dxf),
        output_dir  = str(tmp_path / "out"),
        model_paths = ["dummy.pt"],   # mockear DetectionModel
    )
    # mock DetectionModel.load para no necesitar GPU/pesos reales
    with patch("pipeline.base.DetectionModel.load") as mock:
        mock.return_value.predict.return_value = []
        result = Pipeline(cfg).run()
    assert "detections" in result
```

---

## 6. Plan de migración gradual

La refactorización completa no tiene por qué hacerse de un tirón. Orden sugerido:

| Fase | Qué hacer | Riesgo | Impacto |
|------|-----------|--------|---------|
| 1 | Crear `config.py` con todos los magic numbers. Reemplazar constantes en todos los archivos. | Muy bajo | Alto: elimina dispersión |
| 2 | Extraer `CoordinateMapper` y actualizar `inference_sahi` y `spec_extractor` para usarlo | Bajo | Elimina bug surface |
| 3 | Extraer `PostProcessorChain` de `inference_sahi.py::ejecutar()` | Medio | Testabilidad |
| 4 | Extraer `ColorPolicy` de `dxf_to_image_v2.py` | Bajo | Extensibilidad |
| 5 | Extraer strategies de `scale_analyzer_v2` | Bajo | Claridad |
| 6 | Crear `Pipeline` y `EnsemblePipeline` unificados | Medio | Elimina duplicación |
| 7 | CLI unificado (`cli.py`) | Bajo | UX |
| 8 | Reemplazar `print()` por `logging` | Bajo | Operabilidad |
| 9 | Tests unitarios por módulo | — | Calidad a largo plazo |
| 10 | Eliminar archivos obsoletos (`v1`, `run.py`, CLIs individuales) | Bajo | Limpieza |

> La Fase 1 y 2 se pueden hacer en < 2 horas y tienen el mayor ROI inmediato.

---

## 7. Resumen de principios SOLID aplicados

| Principio | Problema actual | Solución |
|-----------|----------------|----------|
| **S** Single Responsibility | `ejecutar()` hace 8 cosas, `renderizar_dxf()` hace 6 | Extraer clases con una responsabilidad cada una |
| **O** Open/Closed | Añadir filtro = editar `ejecutar()` | `PostProcessorChain`: agregar filtro sin tocar los existentes |
| **L** Liskov | `EnsemblePipeline` puede sustituir a `Pipeline` sin sorpresas | Template Method: sólo overridear `_run_inference()` |
| **I** Interface Segregation | Clientes de `inference_sahi` importan todo el módulo | Interfaces pequeñas: `ScaleStrategy`, `ColorPolicy`, `DetectionFilter` |
| **D** Dependency Inversion | `pipeline_v2` depende de funciones concretas de `inference_sahi` | `Pipeline.__init__` recibe `renderer`, `slicer`, `postproc` como dependencias |
