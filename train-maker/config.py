# config.py — Motor de configuración del pipeline.
# Parsea components_config.yaml y expone settings tipados.
#
# Cambios respecto de la versión anterior (auditoría H9, H19):
#   * Se agregaron los campos de learning rate, que el YAML declaraba y el
#     loader descartaba en silencio (lr0 nunca llegaba a Ultralytics).
#   * Se valida que no haya claves desconocidas en el YAML: antes cualquier
#     typo o campo no soportado se ignoraba sin avisar.
#   * Se eliminó el bloque de "aliases legacy" del final, que ejecutaba
#     load_config() en tiempo de import y exponía constantes derivadas del
#     PRIMER componente de la lista como defaults de medio pipeline.
from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Optional

import yaml

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
_CONFIG_FILE = BASE_DIR / "components_config.yaml"


# ── Dataclasses tipadas ───────────────────────────────────────────────────────

@dataclass
class BackgroundsConfig:
    """Generación de fondos desde planos DXF completos."""

    enabled: bool = False
    dxf_sources_dir: Path = Path("input/backgrounds")
    tile_size: int = 640
    overlap: int = 320
    min_std_dev: float = 6.0
    # Fracción mínima de píxeles con tinta para que un tile cuente como
    # contexto de plano y no como margen en blanco.
    min_ink_fraction: float = 0.010

    # Escala de render: cuántos píxeles debe medir un símbolo típico del plano.
    # Debe coincidir con el --target-px que se usa en inferencia.
    target_symbol_px: int = 64

    # Rechazo de símbolos (ver H3/H1): qué se considera "puede ser un símbolo".
    min_circle_radius: float = 0.15   # unidades CAD; menor = punto de conexión
    min_hatch_side: float = 0.15      # unidades CAD
    symbol_margin_px: int = 24        # margen extra alrededor de cada símbolo
    fallback_symbol_cad: float = 1.19  # si el plano no tiene símbolos medibles

    # Control de memoria y volumen
    render_window_px: int = 2560
    max_render_windows: int = 60
    max_tiles_per_plan: int = 400

    # Planos que nunca deben usarse como fondo (set de evaluación).
    exclude_stems: tuple[str, ...] = ("test1", "test_2", "test2")


@dataclass
class AugmentationConfig:
    """Hiperparámetros de augmentation de Ultralytics.

    Los defaults están pensados para DIBUJO TÉCNICO monocromático, no para
    fotografía natural (ver H11):
      * los símbolos eléctricos son quirales -> nada de flips;
      * mixup promedia imágenes y sobre líneas negras genera trazos fantasma;
      * copy_paste requiere máscaras y en `detect` se ignora;
      * erasing alto borra el único rasgo distintivo de un símbolo que es
        90 % espacio en blanco.
    """

    hsv_h: float = 0.0
    hsv_s: float = 0.1
    hsv_v: float = 0.4
    degrees: float = 3.0
    translate: float = 0.1
    scale: float = 0.5
    shear: float = 2.0
    perspective: float = 0.0
    flipud: float = 0.0
    fliplr: float = 0.0
    mosaic: float = 1.0
    mixup: float = 0.0
    copy_paste: float = 0.0
    erasing: float = 0.1


@dataclass
class ModifiersConfig:
    """Sprites de modificadores (polos, anotaciones) compuestos sobre la base."""

    dir: Path = Path("input/modifiers")
    probability: float = 0.0
    count_min: int = 1
    count_max: int = 1
    allow_rotation: bool = True
    thickness_dilation: list[int] = field(default_factory=lambda: [1, 3])


@dataclass
class GlobalConfig:
    """Bloque ``global:`` del YAML."""

    backgrounds_dir: Path
    output_dir: Path
    dataset_dir: Path

    yolo_workspace: Path = Path("yolo_workspace")
    training_mode: str = "per_component"

    # Splits — ahora sí se usan (antes se validaban y se ignoraban, ver H2)
    train_ratio: float = 0.80
    val_ratio: float = 0.10
    test_ratio: float = 0.10

    # Proporción de imágenes sin objetos. Presupuesto ÚNICO: incluye los
    # fondos vacíos, los componentes confusos y los negativos manuales.
    negative_ratio: float = 0.20
    # Fracción del presupuesto que se cubre con fondos vacíos; el resto va a
    # componentes confusos / negativos manuales.
    negative_empty_fraction: float = 0.33
    # Tope de negativos sintéticos inyectados en el fine-tune real,
    # como fracción de las imágenes reales de train (ver H6).
    finetune_negative_ratio: float = 0.15

    batch_size: int = 16
    workers: int = 4

    render_dpi: int = 200
    binarize_threshold: int = 200

    # Composición del sprite. AHORA es fracción del LADO MENOR DEL FONDO,
    # no del propio sprite (ver H8).
    sprite_scale_min: float = 0.06
    sprite_scale_max: float = 0.16
    components_per_img_min: int = 1
    components_per_img_max: int = 3
    allow_random_rotation: bool = True

    # Fracción de instancias que se colocan cortadas por el borde, para que
    # el modelo vea lo mismo que le llega desde el slicing de SAHI (ver H12).
    edge_crop_ratio: float = 0.15
    # Área visible mínima para que una instancia cortada se etiquete.
    min_visible_area: float = 0.40
    # Solape máximo permitido entre instancias (IoU). 0 = prohibido (ver H14).
    max_placement_iou: float = 0.15

    # Entrenamiento
    yolo_model: str = "yolo11s.pt"
    epochs: int = 300
    imgsz: int = 640
    patience: int = 50
    project: str = "Liard_Detection"
    lr0: float = 0.01
    lrf: float = 0.01
    optimizer: str = "auto"
    cos_lr: bool = False

    # Fine-tuning
    epochs_finetune: int = 100
    lr0_finetune: float = 0.001
    lrf_finetune: float = 0.01
    freeze_finetune: int = 0   # 0 = backbone descongelado (ver H6)

    seed: int = 42
    max_missing_labels_pct: float = 5.0


@dataclass
class ComponentConfig:
    name: str
    dxf_paths: list[Path]
    images_to_generate: int = 10_000
    sprite_variations: int = 150
    line_thickness_range: list[int] = field(default_factory=lambda: [3, 20])
    polarity_filters: list[str] = field(default_factory=list)
    class_id: int = 0
    roboflow_zip_path: Optional[Path] = None
    skip_training: bool = False
    hard_negatives: dict[str, float] = field(default_factory=dict)
    # Componentes que NUNCA deben usarse como negativo de éste porque son
    # visualmente casi el mismo símbolo (ver H1).
    exclude_as_negative: list[str] = field(default_factory=list)


@dataclass
class PipelineConfig:
    g: GlobalConfig
    backgrounds: BackgroundsConfig
    augmentation: AugmentationConfig
    modifiers: ModifiersConfig
    components: list[ComponentConfig]

    def component_sprites_dir(self, comp: ComponentConfig, variant_idx: int = 0) -> Path:
        if len(comp.dxf_paths) == 1:
            return self.g.output_dir / comp.name / "sprites"
        return self.g.output_dir / comp.name / f"sprites_v{variant_idx}"

    def component_synthetic_dir(self, comp: ComponentConfig, variant_idx: int = 0) -> Path:
        if len(comp.dxf_paths) == 1:
            return self.g.output_dir / comp.name / "synthetic"
        return self.g.output_dir / comp.name / f"synthetic_v{variant_idx}"

    def component_variant_count(self, comp: ComponentConfig) -> int:
        return len(comp.dxf_paths)

    def get(self, name: str) -> ComponentConfig:
        for c in self.components:
            if c.name == name:
                return c
        raise KeyError(f"Componente '{name}' no está en components_config.yaml")

    def class_names(self) -> dict[int, str]:
        return {c.class_id: c.name for c in self.components}


# ── Validación ────────────────────────────────────────────────────────────────

def _check_unknown_keys(raw: dict, dc, block: str, extra: set[str] = frozenset()) -> None:
    """Avisa si el YAML trae claves que el loader no conoce.

    Antes, cualquier clave no soportada (por ejemplo ``lr0`` en el bloque
    global) se descartaba en silencio y daba la falsa impresión de estar
    configurando algo.
    """
    known = {f.name for f in fields(dc)} | set(extra)
    unknown = set(raw) - known
    if unknown:
        raise ValueError(
            f"components_config.yaml: claves desconocidas en el bloque "
            f"'{block}': {sorted(unknown)}.\n"
            f"  Claves válidas: {sorted(known)}"
        )


_COMPONENT_KEYS = {
    "name", "dxf_path", "dxf_paths", "images_to_generate", "sprite_variations",
    "line_thickness_range", "polarity_filters", "roboflow_zip_path",
    "skip_training", "hard_negatives", "exclude_as_negative",
}


def _normalize_dxf_paths(raw_value, component_name: str) -> list[Path]:
    if raw_value is None:
        raise ValueError(
            f"El componente '{component_name}' debe definir 'dxf_path' o 'dxf_paths'."
        )
    if isinstance(raw_value, (str, Path)):
        raw_items = [raw_value]
    elif isinstance(raw_value, (list, tuple)):
        raw_items = list(raw_value)
    else:
        raise TypeError(
            f"El componente '{component_name}' tiene un tipo inválido de ruta DXF: "
            f"{type(raw_value).__name__}."
        )

    normalized: list[Path] = []
    for item in raw_items:
        if isinstance(item, (str, Path)):
            normalized.append(BASE_DIR / item)
        elif isinstance(item, (list, tuple)) and len(item) == 1 and isinstance(item[0], (str, Path)):
            normalized.append(BASE_DIR / item[0])
        else:
            raise TypeError(
                f"El componente '{component_name}' tiene una ruta DXF inválida: {item!r}"
            )
    return normalized


# ── Loader ────────────────────────────────────────────────────────────────────

def load_config(path: Optional[Path] = None) -> PipelineConfig:
    path = path or _CONFIG_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo de configuración: {path}\n"
            "Creá components_config.yaml en train-maker/."
        )

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    top_level = {"global", "backgrounds", "augmentation", "modifiers",
                 "validation", "components"}
    unknown_top = set(raw) - top_level
    if unknown_top:
        raise ValueError(
            f"components_config.yaml: bloques desconocidos {sorted(unknown_top)}. "
            f"Válidos: {sorted(top_level)}"
        )

    # ── global ────────────────────────────────────────────────────────────
    g_raw = dict(raw.get("global", {}) or {})
    _check_unknown_keys(g_raw, GlobalConfig, "global")

    yolo_workspace = (PROJECT_ROOT / g_raw.get("yolo_workspace", "yolo_workspace")).resolve()

    def gnum(key, cast, default):
        return cast(g_raw[key]) if key in g_raw else default

    g = GlobalConfig(
        backgrounds_dir=BASE_DIR / g_raw.get("backgrounds_dir", "output/backgrounds"),
        output_dir=BASE_DIR / g_raw.get("output_dir", "output"),
        dataset_dir=BASE_DIR / g_raw.get("dataset_dir", "dataset"),
        yolo_workspace=yolo_workspace,
        training_mode=str(g_raw.get("training_mode", "per_component")),
        train_ratio=gnum("train_ratio", float, 0.80),
        val_ratio=gnum("val_ratio", float, 0.10),
        test_ratio=gnum("test_ratio", float, 0.10),
        negative_ratio=gnum("negative_ratio", float, 0.20),
        negative_empty_fraction=gnum("negative_empty_fraction", float, 0.33),
        finetune_negative_ratio=gnum("finetune_negative_ratio", float, 0.15),
        batch_size=gnum("batch_size", int, 16),
        workers=gnum("workers", int, 4),
        render_dpi=gnum("render_dpi", int, 200),
        binarize_threshold=gnum("binarize_threshold", int, 200),
        sprite_scale_min=gnum("sprite_scale_min", float, 0.06),
        sprite_scale_max=gnum("sprite_scale_max", float, 0.16),
        components_per_img_min=gnum("components_per_img_min", int, 1),
        components_per_img_max=gnum("components_per_img_max", int, 3),
        allow_random_rotation=bool(g_raw.get("allow_random_rotation", True)),
        edge_crop_ratio=gnum("edge_crop_ratio", float, 0.15),
        min_visible_area=gnum("min_visible_area", float, 0.40),
        max_placement_iou=gnum("max_placement_iou", float, 0.15),
        yolo_model=str(g_raw.get("yolo_model", "yolo11s.pt")),
        epochs=gnum("epochs", int, 300),
        imgsz=gnum("imgsz", int, 640),
        patience=gnum("patience", int, 50),
        project=str(g_raw.get("project", "Liard_Detection")),
        lr0=gnum("lr0", float, 0.01),
        lrf=gnum("lrf", float, 0.01),
        optimizer=str(g_raw.get("optimizer", "auto")),
        cos_lr=bool(g_raw.get("cos_lr", False)),
        epochs_finetune=gnum("epochs_finetune", int, 100),
        lr0_finetune=gnum("lr0_finetune", float, 0.001),
        lrf_finetune=gnum("lrf_finetune", float, 0.01),
        freeze_finetune=gnum("freeze_finetune", int, 0),
        seed=gnum("seed", int, 42),
        max_missing_labels_pct=float(
            (raw.get("validation", {}) or {}).get("max_missing_labels_pct", 5.0)
        ),
    )

    ratio_sum = g.train_ratio + g.val_ratio + g.test_ratio
    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError(
            f"Los ratios de split deben sumar 1.0, suman {ratio_sum:.4f} "
            f"(train={g.train_ratio}, val={g.val_ratio}, test={g.test_ratio})"
        )
    if g.val_ratio <= 0:
        raise ValueError(
            "val_ratio debe ser > 0. Sin set de validación separado las "
            "métricas de entrenamiento no significan nada (ver auditoría H2)."
        )
    if g.sprite_scale_min <= 0 or g.sprite_scale_max > 1.0:
        raise ValueError(
            "sprite_scale_min/max son fracciones del lado menor del fondo y "
            "deben estar en (0, 1]."
        )

    # ── backgrounds ───────────────────────────────────────────────────────
    bg_raw = dict(raw.get("backgrounds", {}) or {})
    _check_unknown_keys(bg_raw, BackgroundsConfig, "backgrounds")
    bg_defaults = BackgroundsConfig()
    backgrounds = BackgroundsConfig(
        enabled=bool(bg_raw.get("enabled", False)),
        dxf_sources_dir=BASE_DIR / bg_raw.get("dxf_sources_dir", "input/backgrounds"),
        tile_size=int(bg_raw.get("tile_size", bg_defaults.tile_size)),
        overlap=int(bg_raw.get("overlap", bg_defaults.overlap)),
        min_std_dev=float(bg_raw.get("min_std_dev", bg_defaults.min_std_dev)),
        min_ink_fraction=float(bg_raw.get("min_ink_fraction", bg_defaults.min_ink_fraction)),
        target_symbol_px=int(bg_raw.get("target_symbol_px", bg_defaults.target_symbol_px)),
        min_circle_radius=float(bg_raw.get("min_circle_radius", bg_defaults.min_circle_radius)),
        min_hatch_side=float(bg_raw.get("min_hatch_side", bg_defaults.min_hatch_side)),
        symbol_margin_px=int(bg_raw.get("symbol_margin_px", bg_defaults.symbol_margin_px)),
        fallback_symbol_cad=float(bg_raw.get("fallback_symbol_cad", bg_defaults.fallback_symbol_cad)),
        render_window_px=int(bg_raw.get("render_window_px", bg_defaults.render_window_px)),
        max_render_windows=int(bg_raw.get("max_render_windows", bg_defaults.max_render_windows)),
        max_tiles_per_plan=int(bg_raw.get("max_tiles_per_plan", bg_defaults.max_tiles_per_plan)),
        exclude_stems=tuple(
            s.lower() for s in bg_raw.get("exclude_stems", bg_defaults.exclude_stems)
        ),
    )

    # ── augmentation ──────────────────────────────────────────────────────
    aug_raw = dict(raw.get("augmentation", {}) or {})
    _check_unknown_keys(aug_raw, AugmentationConfig, "augmentation")
    aug_defaults = AugmentationConfig()
    augmentation = AugmentationConfig(**{
        f.name: float(aug_raw.get(f.name, getattr(aug_defaults, f.name)))
        for f in fields(AugmentationConfig)
    })

    # ── modifiers ─────────────────────────────────────────────────────────
    mod_raw = dict(raw.get("modifiers", {}) or {})
    _check_unknown_keys(mod_raw, ModifiersConfig, "modifiers")
    td = mod_raw.get("thickness_dilation", [1, 3])
    modifiers = ModifiersConfig(
        dir=BASE_DIR / mod_raw.get("dir", "input/modifiers"),
        probability=float(mod_raw.get("probability", 0.0)),
        count_min=int(mod_raw.get("count_min", 1)),
        count_max=int(mod_raw.get("count_max", 1)),
        allow_rotation=bool(mod_raw.get("allow_rotation", True)),
        thickness_dilation=[int(td[0]), int(td[1])],
    )

    # ── components ────────────────────────────────────────────────────────
    components: list[ComponentConfig] = []
    seen_names: set[str] = set()

    for idx, c_raw in enumerate(raw.get("components", []) or []):
        component_name = c_raw.get("name", "?")
        unknown = set(c_raw) - _COMPONENT_KEYS
        if unknown:
            raise ValueError(
                f"Componente '{component_name}': claves desconocidas {sorted(unknown)}"
            )
        if component_name in seen_names:
            raise ValueError(f"Componente duplicado en el YAML: '{component_name}'")
        if any(ch in component_name for ch in ' ()"\'\\/'):
            raise ValueError(
                f"El nombre de componente '{component_name}' tiene caracteres que "
                f"se usan para construir rutas y nombres de run. Usá solo "
                f"[a-z0-9_]."
            )
        seen_names.add(component_name)

        lt = c_raw.get("line_thickness_range", [3, 20])
        dxf_paths = _normalize_dxf_paths(
            c_raw.get("dxf_paths", c_raw.get("dxf_path")), component_name
        )

        zip_path_str = c_raw.get("roboflow_zip_path")
        roboflow_zip_path = (BASE_DIR / zip_path_str).resolve() if zip_path_str else None

        components.append(ComponentConfig(
            name=component_name,
            dxf_paths=dxf_paths,
            images_to_generate=int(c_raw.get("images_to_generate", 10_000)),
            sprite_variations=int(c_raw.get("sprite_variations", 150)),
            line_thickness_range=[int(lt[0]), int(lt[1])],
            polarity_filters=list(c_raw.get("polarity_filters", [])),
            class_id=0 if g.training_mode == "per_component" else idx,
            roboflow_zip_path=roboflow_zip_path,
            skip_training=bool(c_raw.get("skip_training", False)),
            hard_negatives=dict(c_raw.get("hard_negatives", {}) or {}),
            exclude_as_negative=list(c_raw.get("exclude_as_negative", []) or []),
        ))

    if not components:
        raise ValueError("Tiene que haber al menos un componente en components_config.yaml")

    # Referencias cruzadas: que hard_negatives / exclude_as_negative apunten a
    # componentes que existen. Antes un typo acá se ignoraba en silencio.
    for comp in components:
        for ref_name, where in (
            [(n, "hard_negatives") for n in comp.hard_negatives]
            + [(n, "exclude_as_negative") for n in comp.exclude_as_negative]
        ):
            if ref_name not in seen_names:
                raise ValueError(
                    f"Componente '{comp.name}': {where} referencia a "
                    f"'{ref_name}', que no existe en el YAML."
                )

    return PipelineConfig(
        g=g, backgrounds=backgrounds, augmentation=augmentation,
        modifiers=modifiers, components=components,
    )
