"""Vuelve a puntuar evaluaciones ya corridas contra otra version del GT, sin re-inferir.

`evaluate.py` guarda en `<tag>/<plano>_detecciones.csv` TODAS las detecciones fusionadas en
coordenadas CAD (umbral interno 0.05). El matching es puro calculo, asi que cambiar el GT no
obliga a volver a pasar el modelo por los planos: con los CSV alcanza. Doce modelos x 7 planos
pasan de horas de GPU a segundos.

    py -3 src/rescore.py V3               # re-puntua todos los tags que empiezan con V3_
    GT_V4=1 py -3 src/rescore.py V3 V4    # varios prefijos, contra el GT v4
"""
import sys, os, csv, json, glob, collections
sys.path.insert(0, os.path.dirname(__file__))
from paths import BASE, WORK
import evaluate as ev

TH = (0.05, 0.10, 0.20, 0.25, 0.30)

import numpy as np
from scipy.optimize import linear_sum_assignment


def match_rapido(gt, dets):
    """Mismo criterio y mismo resultado que `evaluate.match`, pero vectorizado.

    `evaluate.match` arma la matriz de costos con un doble bucle en Python. Para 1240 GT x 4000
    detecciones son 5 millones de iteraciones por umbral, por plano y por modelo: re-puntuar los
    12 modelos tardaba mas que volver a inferir. Aca la matriz se arma con numpy y ademas se
    recorta a las filas y columnas que tienen algun par posible, que es lo que hace que el
    asignamiento hungaro sea rapido.
    """
    nG, nD = len(gt), len(dets)
    if nG == 0 or nD == 0:
        return [], list(range(nG)), list(range(nD))
    D = np.asarray(dets, float)[:, :4]
    gc = np.array([g['c'] for g in gt], float)
    x0, y0, x1, y1 = D[:, 0], D[:, 1], D[:, 2], D[:, 3]
    mx = (x1 - x0) * .2; my = (y1 - y0) * .2
    gx = gc[:, 0][:, None]; gy = gc[:, 1][:, None]
    dentro = ((x0 - mx)[None, :] <= gx) & (gx <= (x1 + mx)[None, :]) &              ((y0 - my)[None, :] <= gy) & (gy <= (y1 + my)[None, :])
    cx = (x0 + x1) / 2; cy = (y0 + y1) / 2
    tiene_b = np.array([g['b'] is not None for g in gt])
    if tiene_b.any():
        B = np.array([g['b'] if g['b'] is not None else (np.inf, np.inf, -np.inf, -np.inf) for g in gt], float)
        en_caja = (B[:, 0][:, None] <= cx[None, :]) & (cx[None, :] <= B[:, 2][:, None]) &                   (B[:, 1][:, None] <= cy[None, :]) & (cy[None, :] <= B[:, 3][:, None])
        dentro = dentro | en_caja
    if not dentro.any():
        return [], list(range(nG)), list(range(nD))
    fg = np.where(dentro.any(1))[0]; fd = np.where(dentro.any(0))[0]
    sub = dentro[np.ix_(fg, fd)]
    dist = np.hypot(cx[None, :] - gx, cy[None, :] - gy)[np.ix_(fg, fd)]
    BIG = 1e6
    C = np.where(sub, dist, BIG)
    r, c = linear_sum_assignment(C)
    tp = [(int(fg[i]), int(fd[j])) for i, j in zip(r, c) if C[i, j] < BIG]
    mg = {g for g, _ in tp}; md = {d for _, d in tp}
    return tp, [i for i in range(nG) if i not in mg], [j for j in range(nD) if j not in md]


def metrics_rapido(gt, dc, th):
    d = [x for x in dc if x[4] >= th]
    tp, fn, fp = match_rapido(gt, d)
    return dict(th=th, tp=len(tp), fn=len(fn), fp=len(fp))


def gt_de(nombre):
    for n, dxf, gtp in ev.PLANOS:
        if n == nombre:
            return ev.load_gt(n, os.path.join(BASE, gtp))
    return None


def cargar_todos_los_gt():
    gts = {}
    for n, dxf, gtp in ev.PLANOS:
        gts[n] = ev.load_gt(n, os.path.join(BASE, gtp))
    os.environ['EVAL_SET'] = 'qet'; os.environ['QET_DIR'] = 'test_marcelo'
    import importlib
    importlib.reload(ev)
    for n, dxf, gtp in ev.PLANOS:
        gts[n] = ev.load_gt(n, os.path.join(BASE, gtp))
    return gts


def main():
    prefs = [a for a in sys.argv[1:] if not a.startswith('-')] or ['V3']
    gts = cargar_todos_los_gt()
    tags = []
    for pref in prefs:
        for p_ in glob.glob(os.path.join(WORK, 'eval', pref + '_*')):
            tags.append((pref, os.path.basename(p_)))
    modelos = collections.defaultdict(dict)
    for pref, t in sorted(set(tags)):
        m = t[len(pref) + 1:].rsplit('_', 1)[0]
        for csvf in glob.glob(os.path.join(WORK, 'eval', t, '*_detecciones.csv')):
            plano = os.path.basename(csvf)[:-len('_detecciones.csv')]
            if plano not in gts:
                continue
            dc = [[float(r['x1']), float(r['y1']), float(r['x2']), float(r['y2']), float(r['conf'])]
                  for r in csv.DictReader(open(csvf))]
            modelos[m][plano] = dc
    filas = []
    for m, planos in modelos.items():
        por_th = {}
        for th in TH:
            gt_n = tp = fn = fp = 0
            for plano, dc in planos.items():
                gt = gts[plano]
                r = metrics_rapido(gt, dc, th)
                gt_n += len(gt); tp += r['tp']; fn += r['fn']; fp += r['fp']
            por_th[th] = (gt_n, tp, fn, fp, tp / max(1, gt_n), tp / max(1, tp + fp))
        # detalle de FN por bloque al umbral mas bajo
        fnb = collections.Counter()
        for plano, dc in planos.items():
            gt = gts[plano]
            d = [x for x in dc if x[4] >= 0.05]
            _tp, f, _fp = match_rapido(gt, d)
            for i in f:
                fnb[gt[i]['n'].split('$')[-1]] += 1
        filas.append((m, por_th, fnb))
    filas.sort(key=lambda r: (r[1][0.05][2], -r[1][0.05][5]))
    print('%-8s %6s %5s %8s %7s %8s   | %5s %8s %8s' %
          ('modelo', 'GT', 'FN', 'recall', 'FP', 'prec', 'FN25', 'rec@.25', 'prec@.25'))
    for m, p, fnb in filas:
        g, tp, fn, fp, rec, pre = p[0.05]
        _g2, _t2, fn2, _f2, rec2, pre2 = p[0.25]
        print('%-8s %6d %5d %8.4f %7d %8.4f   | %5d %8.4f %8.4f' % (m, g, fn, rec, fp, pre, fn2, rec2, pre2))
    print()
    for m, p, fnb in filas[:4]:
        print('FN de %s (conf 0.05): %s' % (m, ', '.join('%s=%d' % (k, v) for k, v in fnb.most_common(8))))


if __name__ == '__main__':
    main()
