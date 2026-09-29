"""RF-DETR Nano con rfdetr 1.11 (Python 3.11 en D:\\CLAUDIO_RFDETR\\venv311). Plan RF3.

25/09, pedido de Tomas: probar la version nueva de rfdetr (la 1.10 entrena ~23% mas rapido; las notas
no prometen mas precision). Para comparar SOLO la version, todo igual que RF2 (src/train_rfdetr.py con
rfdetr 1.3.0): ds19_coco, 640 px, 60 epocas, LR a 1/10 en la 48, batch 4 x accum 4 (efectivo 16).
Cambios de API en 1.11: el LR se baja con `lr_scheduler_kwargs={'lr_drop': N}` y el checkpoint trae
estado de lightning; `best_limpio.pth` deja solo pesos + nombre de clase (sin rastros), igual que antes.
Windows: correr con PYTHONUTF8=1 (si no, el log revienta con UnicodeEncodeError).

    D:\\CLAUDIO_RFDETR\\venv311\\Scripts\\python.exe -u src/train_rfdetr11.py
Entorno: RFDETR_NAME (RF3_nano), RFDETR_DS (ds19_coco), RFDETR_OUT (D:\\CLAUDIO_RFDETR\\runs),
RFDETR_EPOCHS (60), RFDETR_LR_DROP (48), RFDETR_BATCH (4), RFDETR_ACCUM (4), RFDETR_RES (640).
"""
import os, sys, argparse, torch
sys.path.insert(0, os.path.dirname(__file__))
from paths import WORK
from rfdetr import RFDETRNano


def limpiar(src, dst):
    ck = torch.load(src, map_location='cpu', weights_only=False)
    torch.save({'model': ck['model'], 'args': argparse.Namespace(class_names=['componente'])}, dst)


if __name__ == '__main__':
    out = os.path.join(os.environ.get('RFDETR_OUT', r'D:\CLAUDIO_RFDETR\runs'), os.environ.get('RFDETR_NAME', 'RF3_nano'))
    os.makedirs(out, exist_ok=True)
    m = RFDETRNano(pretrain_weights=os.path.join(WORK, 'rfdetr', 'rf-detr-nano.pth'),
                   resolution=int(os.environ.get('RFDETR_RES', '640')))
    # 28/09: tras un corte de luz retoma desde last.ckpt (checkpoint completo de lightning: pesos, EMA,
    # optimizer, scheduler y epoca) en vez de empezar de cero.
    ck = os.path.join(out, 'last.ckpt')
    extra = dict(resume=ck) if os.path.exists(ck) else {}
    if extra: print('[rfdetr11] retoma desde', ck)
    m.train(**extra, dataset_dir=os.path.join(WORK, os.environ.get('RFDETR_DS', 'ds19_coco')), output_dir=out,
            epochs=int(os.environ.get('RFDETR_EPOCHS', '60')),
            batch_size=int(os.environ.get('RFDETR_BATCH', '4')),
            grad_accum_steps=int(os.environ.get('RFDETR_ACCUM', '4')),
            lr_scheduler='step', lr_scheduler_kwargs={'lr_drop': int(os.environ.get('RFDETR_LR_DROP', '48'))},
            num_workers=4, early_stopping=True, early_stopping_patience=60,
            checkpoint_interval=5, tensorboard=False, progress_bar=None)
    limpiar(os.path.join(out, 'checkpoint_best_total.pth'), os.path.join(out, 'best_limpio.pth'))
    print('[rfdetr11] pesos limpios ->', os.path.join(out, 'best_limpio.pth'))
