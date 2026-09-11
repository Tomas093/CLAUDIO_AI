# train.py — Entrenamiento YOLO en dos fases (sintética + fine-tune real).
#
# Cambios respecto de la versión anterior (auditoría H5, H9, H10):
#   * `resume` es un flag EXPLÍCITO. Antes bastaba con que existiera last.pt
#     para que se reanudara el run anterior, y Ultralytics con resume=True
#     restaura los argumentos viejos e ignora el data.yaml, las épocas y las
#     augmentations nuevas. Ese era el motivo de "arreglé algo, reentrené y
#     no cambió nada".
#   * Sin --resume, el run anterior se archiva con timestamp en vez de
#     escribirse encima con exist_ok=True.
#   * La fase 2 usa cfg.g.yolo_workspace igual que la fase 1; antes tenía
#     hardcodeado C:/temp/yolo_workspace.
#   * lr0 / lrf / optimizer / cos_lr salen de la configuración.
from __future__ import annotations

import argparse
import shutil
import sys
import torch

import pathlib
import time

_original_write_bytes = pathlib.Path.write_bytes

def _patched_write_bytes(self, data):
    retries = 5
    for i in range(retries):
        try:
            return _original_write_bytes(self, data)
        except OSError as e:
            if i == retries - 1:
                raise
            print(f"[HOTFIX] Transient OSError {e} on {self}. Retrying {i+1}/{retries}...")
            time.sleep(2)

pathlib.Path.write_bytes = _patched_write_bytes

import time
from pathlib import Path

import ctypes

def _prevent_windows_sleep():
    """Evita que Windows entre en suspensión o hibernación durante el entrenamiento."""
    try:
        ES_CONTINUOUS = 0x80000000
        ES_SYSTEM_REQUIRED = 0x00000001
        ES_AWAYMODE_REQUIRED = 0x00000040
        ctypes.windll.kernel32.SetThreadExecutionState(
            ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_AWAYMODE_REQUIRED
        )
        print("[HOTFIX] Windows Sleep Prevention ACTIVADA (SetThreadExecutionState).")
    except Exception as e:
        print(f"[WARN] No se pudo activar Sleep Prevention: {e}")

_prevent_windows_sleep()

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import torch
from ultralytics import YOLO

from config import BASE_DIR, PipelineConfig, load_config


def _archive_previous_run(run_dir: Path) -> None:
    """Mueve un run anterior a <nombre>_<timestamp> en vez de pisarlo."""
    if not run_dir.exists():
        return
    stamp = time.strftime("%Y%m%d_%H%M%S")
    archived = run_dir.with_name(f"{run_dir.name}_prev_{stamp}")
    run_dir.rename(archived)
    print(f"[train] Run anterior archivado en '{archived.name}'")


def _resolve_start_weights(cfg: PipelineConfig, run_dir: Path, resume: bool) -> tuple[str, bool]:
    last_pt = run_dir / "weights" / "last.pt"
    if resume:
        if not last_pt.exists():
            print(f"⚠️ Se pidió --resume pero no existe {last_pt}. Arrancando de cero.")
            _archive_previous_run(run_dir)
            model_path = BASE_DIR / cfg.g.yolo_model
            return str(model_path), False
        print(f"[train] REANUDANDO desde {last_pt}.\n"
              f"        Ultralytics va a restaurar los argumentos del run anterior: "
              f"el data.yaml y los hiperparámetros que pases ahora se IGNORAN.")
        return str(last_pt), True

    _archive_previous_run(run_dir)
    model_path = BASE_DIR / cfg.g.yolo_model
    if not model_path.exists():
        # Ultralytics lo descarga solo si no está en disco
        return cfg.g.yolo_model, False
    return str(model_path), False


def _common_kwargs(cfg: PipelineConfig) -> dict:
    aug = cfg.augmentation
    return dict(
        imgsz=cfg.g.imgsz,
        batch=cfg.g.batch_size,
        workers=cfg.g.workers,
        seed=cfg.g.seed,
        deterministic=True,
        patience=cfg.g.patience,
        optimizer=cfg.g.optimizer,
        cos_lr=cfg.g.cos_lr,
        hsv_h=aug.hsv_h, hsv_s=aug.hsv_s, hsv_v=aug.hsv_v,
        degrees=aug.degrees, translate=aug.translate, scale=aug.scale,
        shear=aug.shear, perspective=aug.perspective,
        flipud=aug.flipud, fliplr=aug.fliplr,
        mosaic=aug.mosaic, mixup=aug.mixup, copy_paste=aug.copy_paste,
        erasing=aug.erasing,
    )


def train_synthetic(component_name: str, data_yaml: Path, cfg: PipelineConfig,
                    resume: bool = False) -> Path:
    """Fase 1: entrena desde el checkpoint base con el dataset sintético."""
    device = 0 if torch.cuda.is_available() else "cpu"
    run_name = f"phase1_{component_name}"
    project_dir = cfg.g.yolo_workspace
    run_dir = project_dir / run_name

    weights, resuming = _resolve_start_weights(cfg, run_dir, resume)
    
    if resuming:
        try:
            ckpt = torch.load(weights, map_location='cpu')
            epoch_val = ckpt.get('epoch', 0)
            if epoch_val >= cfg.g.epochs - 1 or epoch_val == -1:
                print(f"✅ Fase 1 ya completó sus {cfg.g.epochs} épocas. Salteando.")
                return Path(weights)
        except Exception:
            pass

    model = YOLO(weights)

    print(f"\n[Fase 1] '{component_name}' | pesos: {weights} | "
          f"datos: {data_yaml} | device: {device}")

    model.train(
        data=str(data_yaml.resolve()),
        epochs=cfg.g.epochs,
        lr0=cfg.g.lr0,
        lrf=cfg.g.lrf,
        name=run_name,
        project=str(project_dir.resolve()),
        exist_ok=True,
        resume=resuming,
        device=device,
        **_common_kwargs(cfg),
    )

    best_pt = run_dir / "weights" / "best.pt"
    if not best_pt.exists():
        raise FileNotFoundError(
            f"El entrenamiento sintético de {component_name} no generó {best_pt}"
        )
    return best_pt


def train_finetune(component_name: str, data_yaml: Path, base_weights: Path,
                   cfg: PipelineConfig, resume: bool = False) -> Path:
    """Fase 2: fine-tuning del modelo sintético con los datos reales."""
    if not base_weights.exists():
        raise FileNotFoundError(
            f"No están los pesos de la fase 1: {base_weights}. Corré la fase 1 primero."
        )

    device = 0 if torch.cuda.is_available() else "cpu"
    run_name = f"phase2_{component_name}"
    project_dir = cfg.g.yolo_workspace     # antes: hardcodeado a C:/temp
    run_dir = project_dir / run_name

    if resume:
        last_pt = run_dir / "weights" / "last.pt"
        if not last_pt.exists():
            print(f"⚠️ Se pidió --resume pero no existe {last_pt}. Arrancando de cero con weights base.")
            _archive_previous_run(run_dir)
            model = YOLO(str(base_weights))
            resume = False
        else:
            try:
                ckpt = torch.load(str(last_pt), map_location='cpu')
                epoch_val = ckpt.get('epoch', 0)
                if epoch_val >= cfg.g.epochs_finetune - 1 or epoch_val == -1:
                    print(f"✅ Fase 2 ya completó sus {cfg.g.epochs_finetune} épocas. Salteando.")
                    return
            except Exception:
                pass
            model = YOLO(str(last_pt))
    else:
        _archive_previous_run(run_dir)
        model = YOLO(str(base_weights))

    print(f"\n[Fase 2] Fine-tune de '{component_name}' desde {base_weights.name} | "
          f"freeze={cfg.g.freeze_finetune} | device: {device}")

    kwargs = _common_kwargs(cfg)
    kwargs["workers"] = 0        # el dataset real es chico; evita overhead
    kwargs["optimizer"] = "SGD"  # Fuerza SGD para Fase 2
    model.train(
        data=str(data_yaml.resolve()),
        epochs=cfg.g.epochs_finetune,
        lr0=cfg.g.lr0_finetune,
        lrf=cfg.g.lrf_finetune,
        freeze=cfg.g.freeze_finetune or None,
        amp=False,
        name=run_name,
        project=str(project_dir.resolve()),
        exist_ok=True,
        resume=resume,
        device=device,
        **kwargs,
    )

    best_pt = run_dir / "weights" / "best.pt"
    if not best_pt.exists():
        raise FileNotFoundError(
            f"El fine-tuning de {component_name} no generó {best_pt}"
        )

    models_dir = BASE_DIR / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    final = models_dir / f"best_{component_name}.pt"
    shutil.copy2(best_pt, final)
    print(f"[Fase 2] Modelo final: {final}")
    return final


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenamiento YOLO de un componente")
    parser.add_argument("--component", required=True)
    parser.add_argument("--phase", choices=["1", "2"], required=True)
    parser.add_argument("--data-yaml", required=True)
    parser.add_argument("--base-weights")
    parser.add_argument(
        "--resume", action="store_true",
        help="Reanuda el run anterior. OJO: Ultralytics restaura los argumentos "
             "guardados en el checkpoint e ignora el data.yaml y los "
             "hiperparámetros que pases ahora.",
    )
    args = parser.parse_args()

    cfg = load_config()
    if args.phase == "1":
        train_synthetic(args.component, Path(args.data_yaml), cfg, resume=args.resume)
    else:
        if not args.base_weights:
            raise ValueError("--base-weights es obligatorio para la fase 2")
        train_finetune(args.component, Path(args.data_yaml),
                       Path(args.base_weights), cfg, resume=args.resume)
