# config.py — Phase 1: Configuration Engine
# Parses components_config.yaml and exposes typed settings for the entire pipeline.
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

BASE_DIR = Path(__file__).resolve().parent
_CONFIG_FILE = BASE_DIR / "components_config.yaml"


# ── Typed dataclasses ─────────────────────────────────────────────────────────

@dataclass
class BackgroundsConfig:
    """Settings for automatic background generation from DXF floor plans."""

    enabled: bool = False
    dxf_sources_dir: Path = Path("input/planos_completos")
    tile_size: int = 640
    overlap: int = 320
    render_dpi: int = 4098
    min_std_dev: float = 10.0


@dataclass
class AugmentationConfig:
    """YOLO training augmentation hyperparameters.

    Defaults are tuned aggressively for synthetic datasets, which are
    highly prone to overfitting (the model memorises sprite patterns).
    All values map directly to Ultralytics ``model.train()`` kwargs.
    """

    hsv_h: float = 0.015       # Hue shift (fraction of 360°)
    hsv_s: float = 0.7         # Saturation shift
    hsv_v: float = 0.4         # Value/brightness shift
    degrees: float = 15.0      # Random rotation ±degrees
    translate: float = 0.1     # Random translate ±fraction
    scale: float = 0.5         # Random scale ±fraction
    shear: float = 2.0         # Random shear ±degrees
    perspective: float = 0.0   # Perspective distortion
    flipud: float = 0.1        # Vertical flip probability
    fliplr: float = 0.5        # Horizontal flip probability
    mosaic: float = 1.0        # Mosaic augmentation probability
    mixup: float = 0.15        # MixUp augmentation probability
    copy_paste: float = 0.1    # Copy-paste augmentation probability
    erasing: float = 0.4       # Random erasing probability


@dataclass
class ModifiersConfig:
    """Settings for modifier sprites (pole symbols, annotations, etc.).

    Modifiers are PNG images composited on top of the base sprite
    before pasting onto the background.  Each parameter controls
    a different axis of randomness.
    """

    dir: Path = Path("input/modifiers")
    probability: float = 0.70        # prob. of attaching ANY modifier
    count_min: int = 1               # min modifiers when used
    count_max: int = 1               # max modifiers when used
    allow_rotation: bool = True      # randomly rotate modifier 0/90/180/270
    thickness_dilation: list[int] = field(default_factory=lambda: [1, 3])


@dataclass
class GlobalConfig:
    """Global pipeline settings parsed from the YAML ``global:`` block."""

    # I/O
    backgrounds_dir: Path
    output_dir: Path
    dataset_dir: Path

    # Split ratios
    train_ratio: float = 0.80
    val_ratio: float = 0.10
    test_ratio: float = 0.10

    # Negatives
    negative_ratio: float = 0.15

    # Hardware constraints (prevents OOM on 8 GB VRAM)
    batch_size: int = 16
    workers: int = 4

    # Rendering
    render_dpi: int = 200
    binarize_threshold: int = 200

    # Sprite composition
    sprite_scale_min: float = 0.08
    sprite_scale_max: float = 0.25
    components_per_img_min: int = 1
    components_per_img_max: int = 5
    allow_random_rotation: bool = True

    # YOLO training
    yolo_model: str = "yolov8m.pt"
    epochs: int = 100
    imgsz: int = 640
    patience: int = 20
    project: str = "Liard_Detection"

    # Validation safety
    max_missing_labels_pct: float = 5.0


@dataclass
class ComponentConfig:
    """Per-component settings parsed from each item in ``components:``."""

    name: str
    dxf_path: Path
    images_to_generate: int = 10_000
    sprite_variations: int = 150
    line_thickness_range: list[int] = field(default_factory=lambda: [2, 150])
    polarity_filters: list[str] = field(default_factory=list)
    class_id: int = 0  # assigned at load time


@dataclass
class PipelineConfig:
    """Top-level container holding the full pipeline configuration."""

    g: GlobalConfig
    backgrounds: BackgroundsConfig
    augmentation: AugmentationConfig
    modifiers: ModifiersConfig
    components: list[ComponentConfig]

    # ── Convenience helpers ────────────────────────────────────────────────

    def component_sprites_dir(self, comp: ComponentConfig) -> Path:
        """Return the directory where sprites for *comp* are stored."""
        return self.g.output_dir / comp.name / "sprites"

    def component_synthetic_dir(self, comp: ComponentConfig) -> Path:
        """Return the directory where synthetic images for *comp* are stored."""
        return self.g.output_dir / comp.name / "synthetic"

    def class_names(self) -> dict[int, str]:
        """Return ``{class_id: name}`` mapping for all components."""
        return {c.class_id: c.name for c in self.components}


# ── Loader ────────────────────────────────────────────────────────────────────

def load_config(path: Optional[Path] = None) -> PipelineConfig:
    """Parse *components_config.yaml* and return a fully-resolved config."""

    path = path or _CONFIG_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {path}\n"
            "Create components_config.yaml in train-maker/."
        )

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))

    # ── Global block ──────────────────────────────────────────────────────
    g_raw = raw.get("global", {})
    g = GlobalConfig(
        backgrounds_dir=BASE_DIR / g_raw.get("backgrounds_dir", "input/backgrounds"),
        output_dir=BASE_DIR / g_raw.get("output_dir", "output"),
        dataset_dir=BASE_DIR / g_raw.get("dataset_dir", "dataset"),
        train_ratio=float(g_raw.get("train_ratio", 0.80)),
        val_ratio=float(g_raw.get("val_ratio", 0.10)),
        test_ratio=float(g_raw.get("test_ratio", 0.10)),
        negative_ratio=float(g_raw.get("negative_ratio", 0.15)),
        batch_size=int(g_raw.get("batch_size", 16)),
        workers=int(g_raw.get("workers", 4)),
        render_dpi=int(g_raw.get("render_dpi", 200)),
        binarize_threshold=int(g_raw.get("binarize_threshold", 200)),
        sprite_scale_min=float(g_raw.get("sprite_scale_min", 0.08)),
        sprite_scale_max=float(g_raw.get("sprite_scale_max", 0.25)),
        components_per_img_min=int(g_raw.get("components_per_img_min", 1)),
        components_per_img_max=int(g_raw.get("components_per_img_max", 5)),
        allow_random_rotation=bool(g_raw.get("allow_random_rotation", True)),
        yolo_model=str(g_raw.get("yolo_model", "yolov8m.pt")),
        epochs=int(g_raw.get("epochs", 100)),
        imgsz=int(g_raw.get("imgsz", 640)),
        patience=int(g_raw.get("patience", 20)),
        project=str(g_raw.get("project", "Liard_Detection")),
        max_missing_labels_pct=float(
            raw.get("validation", {}).get("max_missing_labels_pct", 5.0)
        ),
    )

    # Validate ratios
    ratio_sum = g.train_ratio + g.val_ratio + g.test_ratio
    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError(
            f"Split ratios must sum to 1.0, got {ratio_sum:.4f} "
            f"(train={g.train_ratio}, val={g.val_ratio}, test={g.test_ratio})"
        )

    # ── Backgrounds block ─────────────────────────────────────────────────
    bg_raw = raw.get("backgrounds", {})
    backgrounds = BackgroundsConfig(
        enabled=bool(bg_raw.get("enabled", False)),
        dxf_sources_dir=BASE_DIR / bg_raw.get("dxf_sources_dir", "input/planos_completos"),
        tile_size=int(bg_raw.get("tile_size", 640)),
        overlap=int(bg_raw.get("overlap", 320)),
        render_dpi=int(bg_raw.get("render_dpi", 4098)),
        min_std_dev=float(bg_raw.get("min_std_dev", 10.0)),
    )

    # ── Augmentation block ────────────────────────────────────────────────
    aug_raw = raw.get("augmentation", {})
    augmentation = AugmentationConfig(
        hsv_h=float(aug_raw.get("hsv_h", 0.015)),
        hsv_s=float(aug_raw.get("hsv_s", 0.7)),
        hsv_v=float(aug_raw.get("hsv_v", 0.4)),
        degrees=float(aug_raw.get("degrees", 15.0)),
        translate=float(aug_raw.get("translate", 0.1)),
        scale=float(aug_raw.get("scale", 0.5)),
        shear=float(aug_raw.get("shear", 2.0)),
        perspective=float(aug_raw.get("perspective", 0.0)),
        flipud=float(aug_raw.get("flipud", 0.1)),
        fliplr=float(aug_raw.get("fliplr", 0.5)),
        mosaic=float(aug_raw.get("mosaic", 1.0)),
        mixup=float(aug_raw.get("mixup", 0.15)),
        copy_paste=float(aug_raw.get("copy_paste", 0.1)),
        erasing=float(aug_raw.get("erasing", 0.4)),
    )

    # ── Modifiers block ───────────────────────────────────────────────────
    mod_raw = raw.get("modifiers", {})
    td = mod_raw.get("thickness_dilation", [1, 3])
    modifiers = ModifiersConfig(
        dir=BASE_DIR / mod_raw.get("dir", "input/modifiers"),
        probability=float(mod_raw.get("probability", 0.70)),
        count_min=int(mod_raw.get("count_min", 1)),
        count_max=int(mod_raw.get("count_max", 1)),
        allow_rotation=bool(mod_raw.get("allow_rotation", True)),
        thickness_dilation=[int(td[0]), int(td[1])],
    )

    # ── Components list ───────────────────────────────────────────────────
    components: list[ComponentConfig] = []
    for idx, c_raw in enumerate(raw.get("components", [])):
        lt = c_raw.get("line_thickness_range", [2, 150])
        comp = ComponentConfig(
            name=c_raw["name"],
            dxf_path=BASE_DIR / c_raw["dxf_path"],
            images_to_generate=int(c_raw.get("images_to_generate", 10_000)),
            sprite_variations=int(c_raw.get("sprite_variations", 150)),
            line_thickness_range=[int(lt[0]), int(lt[1])],
            polarity_filters=list(c_raw.get("polarity_filters", [])),
            class_id=idx,
        )
        components.append(comp)

    if not components:
        raise ValueError("At least one component must be defined in components_config.yaml")

    return PipelineConfig(
        g=g, backgrounds=backgrounds, augmentation=augmentation,
        modifiers=modifiers, components=components,
    )


# ── Backward-compatible aliases (used by legacy imports) ──────────────────────
# These are populated at import time so older scripts that do
#   ``from config import DXF_FILE, SPRITES_DIR, ...``
# continue to work during the migration.

_cfg = load_config()

# Paths
INPUT_DIR = BASE_DIR / "input"
DXF_FILE = _cfg.components[0].dxf_path if _cfg.components else INPUT_DIR / "component.dxf"
BACKGROUNDS_DIR = _cfg.g.backgrounds_dir
OUTPUT_DIR = _cfg.g.output_dir
SPRITES_DIR = _cfg.component_sprites_dir(_cfg.components[0]) if _cfg.components else OUTPUT_DIR / "sprites"
SYNTHETIC_DIR = _cfg.component_synthetic_dir(_cfg.components[0]) if _cfg.components else OUTPUT_DIR / "synthetic"
DATASET_DIR = _cfg.g.dataset_dir

# Phase 1
N_SPRITE_VARIATIONS = _cfg.components[0].sprite_variations if _cfg.components else 150
DILATION_KERNEL_MIN = _cfg.components[0].line_thickness_range[0] if _cfg.components else 2
DILATION_KERNEL_MAX = _cfg.components[0].line_thickness_range[1] if _cfg.components else 150
RENDER_DPI = _cfg.g.render_dpi
BINARIZE_THRESHOLD = _cfg.g.binarize_threshold

# Phase 2/3
N_SYNTHETIC_TOTAL = _cfg.components[0].images_to_generate if _cfg.components else 500
SPRITE_SCALE_MIN = _cfg.g.sprite_scale_min
SPRITE_SCALE_MAX = _cfg.g.sprite_scale_max
COMPONENTS_PER_IMG_MIN = _cfg.g.components_per_img_min
COMPONENTS_PER_IMG_MAX = _cfg.g.components_per_img_max
ALLOW_RANDOM_ROTATION = _cfg.g.allow_random_rotation

# Phase 4
TRAIN_RATIO = _cfg.g.train_ratio
NEGATIVE_RATIO = _cfg.g.negative_ratio
CLASS_ID = 0
CLASS_NAME = _cfg.components[0].name if _cfg.components else "component"
