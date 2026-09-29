"""Reanuda una corrida cortada, opcionalmente cambiando el total de epocas, y deja best_limpio.pt.

23/09: Tomas pidio bajar X_26p2 de 200 a 100 epocas con la corrida en la epoca 65. Ultralytics
reanuda con los argumentos guardados en last.pt, asi que se edita `train_args['epochs']` antes.
El coseno de la lr se recalcula con el total nuevo: la lr baja de golpe al reanudar (esperable).

    py -3 -u src/reanudar.py X_26p2 --epochs 100
"""
import sys, os, argparse, torch
sys.path.insert(0, os.path.dirname(__file__))
from paths import WORK
from train import clean
from ultralytics import YOLO

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('name'); ap.add_argument('--epochs', type=int, default=0)
    a = ap.parse_args()
    run = os.path.join(WORK, 'runs', a.name)
    last = os.path.join(run, 'weights', 'last.pt')
    if a.epochs:
        ck = torch.load(last, map_location='cpu', weights_only=False)
        print('[reanudar] %s: epoca %d, epocas %s -> %d' % (a.name, ck['epoch'] + 1, ck['train_args']['epochs'], a.epochs))
        ck['train_args']['epochs'] = a.epochs
        torch.save(ck, last)
    try:
        YOLO(last).train(resume=True)
    except Exception as e:
        # 28/09: si el corte fue entre el final del entrenamiento y la limpieza, ultralytics dice que no hay
        # nada que reanudar; en ese caso alcanza con limpiar best.pt.
        if 'nothing to resume' not in str(e) or not os.path.exists(os.path.join(run, 'weights', 'best.pt')):
            raise
        print('[reanudar] la corrida ya habia terminado:', e)
    clean(os.path.join(run, 'weights', 'best.pt'), os.path.join(run, 'best_limpio.pt'))
    print('[reanudar] pesos limpios ->', os.path.join(run, 'best_limpio.pt'))
