# train_unified_yolo.py
# Entrenamiento del modelo YOLO11 Nano de clase unica: 'componente'
import os
import sys
import shutil
import ctypes
from pathlib import Path
import torch
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

def main():
    prevent_sleep()
    print('=' * 60)
    print('INICIANDO ENTRENAMIENTO YOLO11 NANO CLASE UNICA (COMPONENTE)')
    print('=' * 60)

    yaml_path = Path('train-maker/dataset_unified_componente/data.yaml').resolve()
    if not yaml_path.exists():
        raise FileNotFoundError(f'No existe el archivo de configuracion: {yaml_path}')

    device = 0 if torch.cuda.is_available() else 'cpu'
    dev_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'
    print(f'Dispositivo: cuda:{device} ({dev_name})')

    # Cargar YOLO11 Nano base
    base_model = 'yolo11n.pt'
    print(f'Cargando checkpoint base: {base_model}...')
    model = YOLO(base_model)

    project_dir = Path('yolo_workspace').resolve()
    run_name = 'unified_componente_yolo11n'

    # Entrenar
    print('Iniciando ciclo de entrenamiento...')
    train_results = model.train(
        data=str(yaml_path),
        epochs=35,
        imgsz=640,
        batch=32,
        workers=2,
        device=device,
        project=str(project_dir),
        name=run_name,
        exist_ok=True,
        patience=10,
        optimizer='AdamW',
        lr0=0.005,
        lrf=0.01,
        degrees=5.0,
        scale=0.5,
        flipud=0.5,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.1,
        hsv_h=0.0,
        hsv_s=0.1,
        hsv_v=0.3,
        verbose=True
    )

    best_pt = project_dir / run_name / 'weights' / 'best.pt'
    if not best_pt.exists():
        raise FileNotFoundError(f'No se genero {best_pt}')

    # Copiar a carpeta models
    dest_pt = Path('train-maker/models/best_componente_nano.pt').resolve()
    dest_pt.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_pt, dest_pt)
    print(f'\n[OK] Modelo optimo guardado exitosamente en: {dest_pt}')

    # Validacion final con metricas
    print('\nEjecutando validacion final sobre el conjunto de validacion...')
    val_model = YOLO(str(dest_pt))
    metrics = val_model.val(data=str(yaml_path), device=device, split='val', imgsz=640)
    
    p = metrics.box.mp
    r = metrics.box.mr
    map50 = metrics.box.map50
    map5095 = metrics.box.map
    print('\n' + '=' * 60)
    print('METRICAS DE VALIDACION DEL MODELO ENTRENADO:')
    print(f'  Precision (mp):     {p:.4f} ({p*100:.2f}%)')
    print(f'  Recall (mr):        {r:.4f} ({r*100:.2f}%)')
    print(f'  mAP50:              {map50:.4f} ({map50*100:.2f}%)')
    print(f'  mAP50-95:           {map5095:.4f} ({map5095*100:.2f}%)')
    print('=' * 60)

if __name__ == '__main__':
    main()
