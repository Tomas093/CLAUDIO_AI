# train.py — Phase 5: YOLO training in two phases (Synthetic + Fine-tune)
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch
from ultralytics import YOLO

from config import PipelineConfig, load_config, BASE_DIR


def train_synthetic(component_name: str, data_yaml: Path, cfg: PipelineConfig) -> Path:
    """Entrena un modelo YOLO desde cero usando el dataset sintético."""
    device = 0 if torch.cuda.is_available() else "cpu"
    run_name = f"phase1_{component_name}"
    project_dir = cfg.g.yolo_workspace
    
    last_pt = project_dir / run_name / "weights" / "last.pt"
    resume = False
    
    if last_pt.exists():
        print(f"\n[Phase 1] Reanudando entrenamiento para '{component_name}' desde {last_pt}")
        model = YOLO(str(last_pt))
        resume = True
    else:
        model_path = Path(__file__).resolve().parent / cfg.g.yolo_model
        model = YOLO(str(model_path))
        print(f"\n[Phase 1] Entrenando modelo sintético desde cero para '{component_name}'")
    
    aug = cfg.augmentation
    
    model.train(
        data=str(data_yaml.resolve()),
        epochs=cfg.g.epochs,
        imgsz=cfg.g.imgsz,
        batch=cfg.g.batch_size,
        workers=cfg.g.workers,
        name=run_name,
        project=str(project_dir.resolve()),
        exist_ok=True,
        resume=resume,
        device=device,
        patience=cfg.g.patience,
        # lr0 no se especifica, usa default de Ultralytics
        
        # Augmentations para evitar overfitting en datos sintéticos
        hsv_h=aug.hsv_h,
        hsv_s=aug.hsv_s,
        hsv_v=aug.hsv_v,
        degrees=aug.degrees,
        translate=aug.translate,
        scale=aug.scale,
        shear=aug.shear,
        perspective=aug.perspective,
        flipud=aug.flipud,
        fliplr=aug.fliplr,
        mosaic=aug.mosaic,
        mixup=aug.mixup,
        copy_paste=aug.copy_paste,
        erasing=aug.erasing,
    )
    
    best_pt = project_dir / run_name / "weights" / "best.pt"
    if not best_pt.exists():
        raise FileNotFoundError(f"Entrenamiento sintético falló para {component_name}, no se generó {best_pt}")
        
    return best_pt


def train_finetune(component_name: str, data_yaml: Path, base_weights: Path, cfg: PipelineConfig) -> Path:
    """Hace fine-tuning del modelo sintético usando datos reales."""
    model = YOLO(str(base_weights))

    device = 0 if torch.cuda.is_available() else "cpu"
    run_name = f"phase2_{component_name}"
    project_dir = Path("C:/temp/yolo_workspace")
    
    print(f"\n[Phase 2] Fine-tuning con datos reales para '{component_name}'")
    
    model.train(
        data=str(data_yaml.resolve()),
        epochs=cfg.g.epochs_finetune,
        imgsz=cfg.g.imgsz,
        batch=cfg.g.batch_size,
        workers=0,
        amp=False,
        name=run_name,
        project=str(project_dir.resolve()),
        exist_ok=True,
        device=device,
        patience=cfg.g.patience,
        lr0=cfg.g.lr0_finetune,
        lrf=cfg.g.lrf_finetune,
        freeze=10,  # Congelar backbone
        save_period=100,
    )
    
    best_pt = project_dir / run_name / "weights" / "best.pt"
    if not best_pt.exists():
        raise FileNotFoundError(f"Fine-tuning falló para {component_name}, no se generó {best_pt}")
        
    # Copiar a carpeta models/ en BASE_DIR (train-maker/models)
    models_dir = BASE_DIR / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    final_model_path = models_dir / f"best_{component_name}.pt"
    shutil.copy2(best_pt, final_model_path)
    
    return final_model_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--component", required=True)
    parser.add_argument("--phase", choices=["1", "2"], required=True)
    parser.add_argument("--data-yaml", required=True)
    parser.add_argument("--base-weights", required=False)
    args = parser.parse_args()

    cfg = load_config()

    if args.phase == "1":
        train_synthetic(args.component, Path(args.data_yaml), cfg)
    elif args.phase == "2":
        if not args.base_weights:
            raise ValueError("--base-weights es requerido para la fase 2")
        train_finetune(args.component, Path(args.data_yaml), Path(args.base_weights), cfg)