"""RF-DETR Nano sobre el mismo dataset que los YOLO (ds18 en formato COCO). Corre en venv_rfdetr.

24/09, pedido de Tomas: "proba tambien el rf-detr, el mas chico". rfdetr 1.3.0 es la ultima que
acepta Python 3.9 y ya trae el Nano. Diferencias a tener en cuenta:
  - El Nano entrena por defecto a 384 px. Aca va a 640, la resolucion de los tiles: a 384 el
    amperimetro (~17 px) quedaria en ~10 px.
  - Es un DETR: salida uno a uno sin NMS (como YOLO26), 300 consultas por imagen.
  - Deja `best_limpio.pth` solo con los pesos y el nombre de la clase (sin args de entrenamiento,
    rutas ni optimizer), la misma regla de "sin rastros" que `train.clean`.

    venv_rfdetr/Scripts/python.exe -u src/train_rfdetr.py
25/09: rfdetr trae lr_drop=100, o sea que con 40-60 epocas el LR NUNCA baja (RF_nano seguia
mejorando en la 39 sin haber tenido fase de LR bajo). RFDETR_LR_DROP = epoca en la que el LR baja a 1/10.
Entorno: RFDETR_NAME (RF_nano), RFDETR_DS (ds18_coco), RFDETR_PATIENCE (10), RFDETR_EPOCHS (40), RFDETR_BATCH (4), RFDETR_ACCUM (4). Con batch 8 se llenaba la GPU (7,8 de 8 GB) y desbordaba a RAM: ~40 s por iteracion, RFDETR_RES (640).
"""
import os, sys, argparse, torch
sys.path.insert(0, os.path.dirname(__file__))
from paths import WORK
from rfdetr import RFDETRNano


def limpiar(src, dst):
    ck = torch.load(src, map_location='cpu', weights_only=False)
    torch.save({'model': ck['model'], 'args': argparse.Namespace(class_names=['componente'])}, dst)


if __name__ == '__main__':
    out = os.path.join(WORK, 'runs', os.environ.get('RFDETR_NAME', 'RF_nano'))
    os.makedirs(out, exist_ok=True)
    pre = os.path.join(WORK, 'rfdetr', 'rf-detr-nano.pth')
    m = RFDETRNano(pretrain_weights=pre, resolution=int(os.environ.get('RFDETR_RES', '640')))
    # 26/09: si la corrida se corto (se cerro la sesion y murio el proceso), retoma desde checkpoint.pth
    # (modelo, EMA, optimizer, scheduler y epoca) en vez de empezar de cero.
    ck = os.path.join(out, 'checkpoint.pth')
    extra = dict(resume=ck) if os.path.exists(ck) else {}
    if extra: print('[rfdetr] retoma desde', ck)
    m.train(**extra,dataset_dir=os.path.join(WORK, os.environ.get('RFDETR_DS', 'ds18_coco')), output_dir=out,
            epochs=int(os.environ.get('RFDETR_EPOCHS', '40')),
            batch_size=int(os.environ.get('RFDETR_BATCH', '4')),
            grad_accum_steps=int(os.environ.get('RFDETR_ACCUM', '4')),
            num_workers=4, early_stopping=True,
            early_stopping_patience=int(os.environ.get('RFDETR_PATIENCE', '10')),
            lr_drop=int(os.environ.get('RFDETR_LR_DROP', '100')),
            checkpoint_interval=5, tensorboard=False, run_test=False)
    best = os.path.join(out, 'checkpoint_best_total.pth')
    limpiar(best, os.path.join(out, 'best_limpio.pth'))
    print('[rfdetr] pesos limpios ->', os.path.join(out, 'best_limpio.pth'))
