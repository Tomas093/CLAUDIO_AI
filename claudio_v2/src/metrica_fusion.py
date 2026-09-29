"""Las dos metricas propias del plan R: amperimetros localizados y pulsadores con caja propia.

Son las que motivaron el plan, asi que son las que dicen si funciono. El total de FN no alcanza:
un pulsador tragado adentro de la caja del contactor y un pulsador que no se detecto cuentan
igual en el total, y son problemas distintos.

  - `A$C45664E16` (amperimetro): cuantos de los 16 tienen ALGUNA deteccion encima, y con que
    confianza. N localiza 10, Q los 16. Lo que faltaba no era detectarlos sino la confianza.
  - `PULS` (pulsador): cuantos de los 25 tienen caja PROPIA y cuantos quedan solamente adentro
    de una caja mas grande. Q saca 6 propias y traga 18; N saca 21. Una caja con dos componentes
    adentro es lo que mas complica la etapa 2.

    py -3 src/metrica_fusion.py V3_v9_N_marcelo V3_v12_Q_marcelo V4_v13_R_marcelo
"""
import sys, os, csv, glob
sys.path.insert(0, os.path.dirname(__file__))
from paths import WORK

PLANOS = ['LU-UN-01 Esquemas Unifilares', 'nyw-un-01 esquemas unifilares']
GT = 'data/test_marcelo/%s_gt_v4.csv'


def cargar(plano, bloque):
    f = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), GT % plano)
    if not os.path.exists(f):
        return []
    return [r for r in csv.DictReader(open(f, encoding='utf-8', errors='ignore'))
            if r['block_name'].split('$')[-1] == bloque]


def dets(tag, plano):
    f = os.path.join(WORK, 'eval', tag, plano + '_detecciones.csv')
    if not os.path.exists(f):
        return None
    return [(float(r['x1']), float(r['y1']), float(r['x2']), float(r['y2']), float(r['conf']))
            for r in csv.DictReader(open(f))]


def main():
    tags = sys.argv[1:] or ['V3_v9_N_marcelo']
    print('%-22s %-10s | %s' % ('tag', 'bloque', 'resultado'))
    for tag in tags:
        tot_a = tot_a_ok = 0; confs = []
        tot_p = tot_p_ok = tot_p_trag = 0
        for plano in PLANOS:
            d = dets(tag, plano)
            if d is None:
                continue
            for g in cargar(plano, 'C45664E16'):
                tot_a += 1
                gx, gy = float(g['x_cad']), float(g['y_cad'])
                mejor = None
                for x0, y0, x1, y1, c in d:
                    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
                    if abs(cx - gx) < .25 and abs(cy - gy) < .25 and (mejor is None or c > mejor):
                        mejor = c
                if mejor is not None:
                    tot_a_ok += 1; confs.append(mejor)
            for g in cargar(plano, 'PULS'):
                tot_p += 1
                gx, gy = float(g['x_cad']), float(g['y_cad'])
                gw = abs(float(g['x2']) - float(g['x1']))
                propia = tragado = False
                for x0, y0, x1, y1, c in d:
                    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
                    if abs(cx - gx) < gw * .6 and abs(cy - gy) < gw * .6:
                        propia = True
                    if x0 <= gx <= x1 and y0 <= gy <= y1 and (x1 - x0) > gw * 1.8:
                        tragado = True
                if propia:
                    tot_p_ok += 1
                elif tragado:
                    tot_p_trag += 1
        confs.sort()
        med = confs[len(confs) // 2] if confs else -1
        bajo = sum(1 for c in confs if c < .20)
        print('%-22s %-10s | %d/%d localizados, conf p50 %.3f, %d por debajo de 0,20'
              % (tag[:22], 'amperim.', tot_a_ok, tot_a, med, bajo))
        print('%-22s %-10s | %d/%d con caja propia, %d tragados en una caja mas grande'
              % ('', 'PULS', tot_p_ok, tot_p, tot_p_trag))


if __name__ == '__main__':
    main()
