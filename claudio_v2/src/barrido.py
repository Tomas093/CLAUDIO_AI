"""Tabla FN/FP por umbral de un tag (GT_V6), con la columna que se elija como puntuacion (29/09)."""
import os, sys, csv, importlib, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault(os.environ.get('GT_BARRIDO', 'GT_V9'), '1')
import evaluate as ev, rescore as rs
def cargar(tag):
    out = []
    for suf, qet in (('005', 0), ('marcelo', 1)):
        if qet: os.environ['EVAL_SET'] = 'qet'; os.environ['QET_DIR'] = 'test_marcelo'
        else: os.environ.pop('EVAL_SET', None)
        importlib.reload(ev)
        for pl, dxf, gtp in ev.PLANOS:
            gt = ev.load_gt(pl, os.path.join(ev.BASE, gtp))
            rows = list(csv.DictReader(open(os.path.join(ev.WORK, 'eval', '%s_%s' % (tag, suf), pl + '_detecciones.csv'))))
            z = ev.neutras(os.path.join(ev.BASE, gtp))   # GT v8: zonas neutras
            if z: rows = [r for r in rows if ev.fuera_de_neutras([[float(r['x1']), float(r['y1']), float(r['x2']), float(r['y2'])]], z)]
            out.append((pl, gt, rows))
    return out
def tabla(planos, score, ths, filtro=None):
    res = {}
    for th in ths:
        fn = fp = 0
        for pl, gt, rows in planos:
            d = [[float(r['x1']), float(r['y1']), float(r['x2']), float(r['y2']), score(r)] for r in rows if (filtro is None or filtro(r))]
            m = rs.metrics_rapido(gt, d, th); fn += m['fn']; fp += m['fp']
        res[th] = (fn, fp)
    return res
# 29/09: planos de DESARROLLO (se mira para decidir) y de RESERVA (solo la medicion final), para no sobreajustar
DESARROLLO = {'LU-UN-01 Esquemas Unifilares', 'test_2', 'tsss_2'}

if __name__ == '__main__':
    tag = sys.argv[1]; P = cargar(tag)
    cols = [('p', lambda r: float(r['p'])), ('conf*p', lambda r: float(r['conf_det']) * float(r['p'])),
            ('conf', lambda r: float(r['conf_det']))] if 'p' in P[0][2][0] else [('conf', lambda r: float(r['conf']))]
    for grupo, sel in (('DESARROLLO', lambda pl: pl in DESARROLLO), ('RESERVA', lambda pl: pl not in DESARROLLO)):
        if os.environ.get('SOLO_DESARROLLO') == '1' and grupo == 'RESERVA': continue
        Q = [x for x in P if sel(x[0])]
        print('== %s (%d componentes)' % (grupo, sum(len(x[1]) for x in Q)))
        for nombre, sc in cols:
            t = tabla(Q, sc, [0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9])
            print('%-7s' % nombre, ' | '.join('%.2f:%d/%d' % (k, *v) for k, v in t.items()))
