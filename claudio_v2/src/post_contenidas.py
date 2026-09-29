"""Supresion de detecciones CONTENIDAS en otra mas grande (30/09, con GT v8).

Con el GT v8 (TOMA = gabinete entero, grillas de accesorios neutras) casi no hay componentes anidados en el GT
(2 de 3443), asi que una deteccion que cae casi entera dentro de otra mas grande es un pedazo o un duplicado.
Regla: se saca d si existe k con area(k) >= R*area(d), inter(d,k)/area(d) >= T y conf(k) >= A*conf(d).
Se decide en DESARROLLO; RESERVA solo se mide.
    GT_V8=1 py -3 src/post_contenidas.py <tag> [col]
"""
import os, sys
os.environ.setdefault('GT_V8', '1'); sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import barrido as B


def suprimir(d, T=.85, R=1.3, A=.5, th0=0.0, P=9.0, K=0.0):
    """d: lista [x1,y1,x2,y2,conf]; devuelve los que quedan."""
    d = [x for x in d if x[4] >= th0]
    if not d: return d
    D = np.array(d); ar = (D[:, 2] - D[:, 0]) * (D[:, 3] - D[:, 1]); keep = []
    for i, x in enumerate(D):
        ix = np.clip(np.minimum(x[2], D[:, 2]) - np.maximum(x[0], D[:, 0]), 0, None)
        iy = np.clip(np.minimum(x[3], D[:, 3]) - np.maximum(x[1], D[:, 1]), 0, None)
        m = (ar >= R * ar[i]) & (ix * iy / ar[i] >= T) & (D[:, 4] >= A * x[4]) & (D[:, 4] >= K)
        if x[4] >= P or not m.any(): keep.append(d[i])   # P: conf propia que protege
    return keep


if __name__ == '__main__':
    tag = sys.argv[1]; col = sys.argv[2] if len(sys.argv) > 2 else 'conf'
    P = B.cargar(tag)
    import rescore as rs
    ths = [0.05, 0.1, 0.15, 0.2, 0.3]
    for T, R, A, Pp, K in [(t, r, 0, 9, k) for t in (.8, .9) for r in (2.5, 3.5, 5.0) for k in (.3, .5, .8)]:
        out = []
        for grupo, sel in (('DES', lambda pl: pl in B.DESARROLLO), ('RES', lambda pl: pl not in B.DESARROLLO)):
            res = []
            for th in ths:
                fn = fp = 0
                for pl, gt, rows in P:
                    if not sel(pl): continue
                    d = [[float(r['x1']), float(r['y1']), float(r['x2']), float(r['y2']), float(r[col])] for r in rows if float(r[col]) >= th]
                    if T < 1: d = suprimir(d, T, R, A, P=Pp, K=K)
                    m = rs.metrics_rapido(gt, d, th); fn += m['fn']; fp += m['fp']
                res.append('%.2f:%d/%d' % (th, fn, fp))
            out.append(grupo + ' ' + ' '.join(res))
        print('T%.2f R%.1f K%.1f | %s' % (T, R, K, ' || '.join(out)), flush=True)
