import yaml
import os

base_config = {
    "global": {
        "backgrounds_dir": "output/backgrounds",
        "output_dir": "output",
        "dataset_dir": "dataset",
        "training_mode": "per_component",
        "train_ratio": 0.8,
        "val_ratio": 0.1,
        "test_ratio": 0.1,
        "components_per_img_min": 1,
        "components_per_img_max": 3,
        "allow_random_rotation": True,
        "require_full_visibility": True,
        "sprite_scale_min": 0.07,
        "sprite_scale_max": 0.15,
        "negative_ratio": 0.30,
        "binarize_threshold": 200,
        "render_dpi": 200,
        "yolo_workspace": "yolo_workspace",
        "yolo_model": "yolo11m.pt",
        "epochs": 300,
        "patience": 100,
        "batch_size": 16,
        "workers": 0,
        "imgsz": 640,
        "epochs_finetune": 100,
        "lr0": 0.001,
        "lr0_finetune": 0.001,
        "lrf_finetune": 0.001,
        "project": "Liard_Detection"
    },
    "backgrounds": {
        "enabled": False,
        "dxf_sources_dir": "input/backgrounds",
        "render_dpi": 4098,
        "tile_size": 640,
        "overlap": 320,
        "min_std_dev": 10
    },
    "augmentation": {
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
        "degrees": 0.0,
        "translate": 0.1,
        "scale": 0.5,
        "shear": 2.0,
        "perspective": 0.0,
        "flipud": 0.1,
        "fliplr": 0.5,
        "mosaic": 1.0,
        "mixup": 0.15,
        "copy_paste": 0.1,
        "erasing": 0.4
    },
    "modifiers": {
        "dir": "input/modifiers",
        "probability": 0.0,
        "count_min": 1,
        "count_max": 1,
        "allow_rotation": True,
        "thickness_dilation": [1, 3]
    },
    "validation": {
        "max_missing_labels_pct": 5.0
    },
    "components": []
}

dxfs = [
    "interruptor_diferencial.dxf"
]

for dxf in dxfs:
    name = dxf.replace(".dxf", "")
    images = 5000
    comp = {
        "name": name,
        "dxf_path": [f"input/components/{dxf}"],
        "images_to_generate": images,
        "line_thickness_range": [10, 50],
        "polarity_filters": ["invert", "threshold"],
        "sprite_variations": 90
    }
    base_config["components"].append(comp)

with open("c:/Users/Tomas/Documents/LAB3/CLAUDIO_AI/train-maker/components_config.yaml", "w", encoding="utf-8") as f:
    yaml.dump(base_config, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
