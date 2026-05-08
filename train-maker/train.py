# train.py — Phase 5: YOLO training with strict VRAM control
# Uses the Ultralytics Python API with hardware constraints from
# components_config.yaml.  Disables auto-batching.
from __future__ import annotations

from pathlib import Path

import torch
from ultralytics import YOLO

from config import load_config


def train_model(data_yaml: Path | None = None) -> None:
    """Launch YOLOv8 training using settings from the pipeline config.

    Parameters
    ----------
    data_yaml : Path, optional
        Explicit path to ``data.yaml``.  When *None* the path is
        resolved from ``components_config.yaml → dataset_dir/data.yaml``.
    """

    cfg = load_config()

    if data_yaml is None:
        data_yaml = cfg.g.dataset_dir / "data.yaml"

    if not data_yaml.exists():
        raise FileNotFoundError(
            f"data.yaml no encontrado: {data_yaml}\n"
            "Ejecutá el pipeline completo antes de entrenar."
        )

    model_path = Path(__file__).resolve().parent / cfg.g.yolo_model
    model = YOLO(str(model_path))

    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"[Training] Dispositivo: {device}")
    print(f"[Training] Modelo: {cfg.g.yolo_model}")
    print(f"[Training] Batch size: {cfg.g.batch_size} (auto-batch DESACTIVADO)")
    print(f"[Training] Workers: {cfg.g.workers}")
    print(f"[Training] data.yaml: {data_yaml.resolve()}")

    aug = cfg.augmentation
    print(f"[Training] Augmentation: degrees={aug.degrees}, mosaic={aug.mosaic}, "
          f"mixup={aug.mixup}, scale={aug.scale}, erasing={aug.erasing}")

    model.train(
        data=str(data_yaml.resolve()),
        epochs=cfg.g.epochs,
        imgsz=cfg.g.imgsz,
        batch=cfg.g.batch_size,          # Strict — no auto-batch
        workers=cfg.g.workers,           # Prevents OOM on dataloaders
        name=f"{cfg.g.project}_v1",
        project=cfg.g.project,
        device=device,
        patience=cfg.g.patience,
        # ── Augmentation (from components_config.yaml) ────────────────
        # Controlled via YAML to prevent overfitting on synthetic sprites.
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
        # Tracking (W&B / MLflow)
        # Set WANDB_MODE=online or configure mlflow before running.
    )


if __name__ == "__main__":
    print("CUDA disponible:", torch.cuda.is_available())
    print("Versión CUDA de PyTorch:", torch.version.cuda)
    print("Cantidad de GPUs:", torch.cuda.device_count())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    train_model()