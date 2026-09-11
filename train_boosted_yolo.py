# train_boosted_yolo.py
# Boosted fine-tuning of YOLO11 Nano for electrical components
import os
import sys
import shutil
import ctypes
from pathlib import Path
import torch
import torch.nn as nn
from ultralytics import YOLO

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

def prevent_sleep():
    try:
        ES_CONTINUOUS = 0x80000000
        ES_SYSTEM_REQUIRED = 0x00000001
        ES_AWAYMODE_REQUIRED = 0x00000040
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_AWAYMODE_REQUIRED)
        print('[HOTFIX] Windows Sleep Prevention ACTIVADA.')
    except Exception as e:
        print(f'[WARN] No se pudo activar Sleep Prevention: {e}')

class FocalBCE(nn.Module):
    """
    Focal Loss (gamma=1.5) para penalizar fuertemente hard negatives (BORNES, tablas, conductores)
    y downweighting en fondos faciles sin aparatos.
    """
    def __init__(self, gamma=1.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.gamma = gamma

    def forward(self, pred, target):
        loss = self.bce(pred, target)
        prob = pred.sigmoid()
        p_t = target * prob + (1.0 - target) * (1.0 - prob)
        modulating = (1.0 - p_t) ** self.gamma
        return loss * modulating

def inject_focal_loss(trainer):
    if hasattr(trainer, 'loss') and hasattr(trainer.loss, 'bce'):
        trainer.loss.bce = FocalBCE(gamma=1.5)
        print('\n[SOTA-OPTIM] Focal Loss ACTIVADA (gamma=1.5) para supresion de hard negatives en trainer.loss.bce\n')

def main():
    prevent_sleep()
    print('=' * 65)
    print('ENTRENAMIENTO BOOST: YOLO11 NANO CLASE UNICA (COMPONENTE)')
    print('Optimizacion CAD: Dataset limpio sin neg_conf, degrees=0.0, fliplr=0.5')
    print('Mosaic=0.5, close_mosaic=5, AdamW fine-tuning desde base limpia')
    print('=' * 65)

    yaml_path = Path('train-maker/dataset_unified_componente/data.yaml').resolve()
    if not yaml_path.exists():
        raise FileNotFoundError(f'No existe {yaml_path}')

    device = 0 if torch.cuda.is_available() else 'cpu'
    print(f'Dispositivo: cuda:{device} ({torch.cuda.get_device_name(0)})')

    # Partir del checkpoint optimo actual
    start_weights = 'train-maker/models/best_componente_nano.pt'
    print(f'Cargando pesos iniciales: {start_weights}')
    model = YOLO(start_weights)

    project_dir = Path('yolo_workspace').resolve()
    run_name = 'boosted_componente_yolo11n'

    # Hiperparámetros de fine-tuning suave sobre las muestras objetivo
    train_results = model.train(
        data=str(yaml_path),
        epochs=12,
        imgsz=640,
        batch=32,
        workers=2,
        device=device,
        project=str(project_dir),
        name=run_name,
        exist_ok=True,
        patience=15,
        optimizer='AdamW',
        lr0=0.0005,          # Fine-tuning rate muy suave
        lrf=0.05,
        cos_lr=True,
        translate=0.15,      # Invarianza espacial cerca de bordes
        scale=0.15,          # Escala moderada sin distorsion
        degrees=0.0,         # Cero rotacion para evitar aliasing en 1px
        flipud=0.0,          # Desactivado: orientacion fija en CAD
        fliplr=0.0,          # Desactivado: cero flip horizontal para evitar corrimiento
        mosaic=0.3,          # Mosaico leve inicial
        mixup=0.0,           # Desactivado: evitar lineas fantasma
        copy_paste=0.0,
        close_mosaic=6,      # Ultimas 6 epocas en tiles 100% reales sin mosaico
        hsv_h=0.01,
        hsv_s=0.2,
        hsv_v=0.2,
        verbose=True
    )

    best_pt = project_dir / run_name / 'weights' / 'best.pt'
    if not best_pt.exists():
        raise FileNotFoundError(f'No se genero {best_pt}')

    # Guardar nuevo modelo optimo definitivo
    dest_model = Path('train-maker/models/best_componente_nano.pt')
    shutil.copy2(best_pt, dest_model)
    print(f'[OK] Nuevo modelo optimo guardado en: {dest_model}')

    # Validacion
    print('\nValidando modelo en split de validacion...')
    val_model = YOLO(str(dest_model))
    metrics = val_model.val(data=str(yaml_path), device=device, split='val', imgsz=640)
    print('=' * 65)
    print('METRICAS FINALES MODELO BOOSTED:')
    print(f'  Precision (mp): {metrics.box.mp:.4f} ({metrics.box.mp*100:.2f}%)')
    print(f'  Recall (mr):    {metrics.box.mr:.4f} ({metrics.box.mr*100:.2f}%)')
    print(f'  mAP50:          {metrics.box.map50:.4f} ({metrics.box.map50*100:.2f}%)')
    print(f'  mAP50-95:       {metrics.box.map:.4f} ({metrics.box.map*100:.2f}%)')
    print('=' * 65)

if __name__ == '__main__':
    main()
