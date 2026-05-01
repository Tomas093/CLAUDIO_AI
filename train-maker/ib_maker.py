"""Compatibilidad con la versión monolítica anterior.

El pipeline fue separado en módulos individuales:
- config.py
- phase1_extractor.py
- phase2_3_fusion_labeler.py
- phase4_assembler.py
- run_pipeline.py
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
    load_sprites,
    rotate_sprite_bound,
    scale_sprite,
)
from phase4_assembler import (
    assemble_dataset,
    create_data_yaml,
    create_yolo_structure,
    inject_negatives,
    print_dataset_summary,
    split_and_copy,
)
from run_pipeline import run_pipeline


if __name__ == "__main__":
    run_pipeline()

