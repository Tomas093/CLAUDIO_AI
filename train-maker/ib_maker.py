"""Compatibilidad con la versión monolítica anterior.

El pipeline fue separado en módulos individuales:
- config.py                → Configuration engine (YAML-based)
- phase1_extractor.py      → Sprite extraction from DXF
- phase2_3_fusion_labeler.py → Synthetic image fusion + YOLO labelling
- phase4_assembler.py      → Dataset assembly (80/10/10 split)
- validate_dataset.py      → Pre-training validation & bbox clipping
- train.py                 → YOLO training with VRAM control
- run_pipeline.py          → Master controller
"""

from config import *  # noqa: F403,F401
from phase1_extractor import apply_dilation, crop_to_content, generate_sprite_variations, render_dxf_to_rgba
from phase2_3_fusion_labeler import (
    YoloBBox,
    calculate_yolo_bbox,
    composite_sprite_on_bg,
    generate_one_sample,
    generate_synthetic_dataset,
    load_background_paths,
    load_rgba_images,
    rotate_sprite_and_track_bbox,
    scale_canvas_and_roi,
)
from phase4_assembler import (
    assemble_dataset,
    create_data_yaml,
    create_yolo_structure,
    generate_yolo_yaml,
    inject_negatives,
    print_dataset_summary,
    split_and_copy,
)
from validate_dataset import validate_dataset
from run_pipeline import run_pipeline


if __name__ == "__main__":
    run_pipeline()
