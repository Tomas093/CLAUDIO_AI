"""Umbral maximo con recall 100% por modelo y por ensamble, y cuantos FP cuesta (sin GPU).

Objetivo de Tomas (22/09): 100% de recall con UN modelo a conf ~0,25; si no, con dos a ~0,25, y con
menos FP que la union R+U a 0,05. Esto recorre los CSV de `work/eval/<tag>_{005,marcelo}` (umbral
interno 0,05) y para cada modelo o par informa:
  - FN y FP a 0,25
  - th100: el umbral mas alto con FN=0 (y los FP a ese umbral)
  - el componente que limita (la GT con la peor confianza maxima)

Reglas de fusion de dos modelos (cluster = detecciones de los dos que se solapan como en `fuse`):
  max    la union de siempre (conf = la mayor)
  por    O probabilistico: 1-(1-a)(1-b). Sube lo que ven los dos; lo de uno solo queda igual.
  media  (a+b)/2 con 0 si un modelo no lo ve: baja lo que ve uno solo (menos FP).

    GT_V4=1 py -3 src/umbral_recall.py V4_v13_R V4_v16_U V4_v14_S ...
"""
import sys, os, csv, importlib, itertools
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from paths import BASE, WORK
import evaluate as ev
import rescore as rs

THS = np.round(np.arange(.05, .701, .01), 2)


def cargar(tags):
    """{plano: (gt, {tag: dets})} para los 7 planos."""
    out = {}
    for suf, qet in (('005', False), ('marcelo', True)):
        if qet:
            os.environ['EVAL_SET'] = 'qet'; os.environ['QET_DIR'] = 'test_marcelo'
        else:
            os.environ.pop('EVAL_SET', None)
        importlib.reload(ev)
        for pl, dxf, gtp in ev.PLANOS:
            gt = ev.load_gt(pl, os.path.join(BASE, gtp))
            d = {}
            for t in tags:
                f = os.path.join(WORK, 'eval', '%s_%s' % (t, suf), pl + '_detecciones.csv')
                if os.path.exists(f):
                    d[t] = np.array([[float(r['x1']), float(r['y1']), float(r['x2']), float(r['y2']), float(r['conf'])]
                                     for r in csv.DictReader(open(f))]).reshape(-1, 5)
            out[pl] = (gt, d)
    return out


def iou_ios(a, B):
    ix = np.clip(np.minimum(a[2], B[:, 2]) - np.maximum(a[0], B[:, 0]), 0, None)
    iy = np.clip(np.minimum(a[3], B[:, 3]) - np.maximum(a[1], B[:, 1]), 0, None)
    I = ix * iy
    A = (a[2] - a[0]) * (a[3] - a[1]); Bb = (B[:, 2] - B[:, 0]) * (B[:, 3] - B[:, 1])
    return I / np.maximum(1e-12, A + Bb - I), I / np.maximum(1e-12, np.minimum(A, Bb)), Bb / max(1e-12, A)


def clusters(dets_por_modelo, iou=.45, ios=.7, ar=4.0):
    """Agrupa detecciones de varios modelos. Devuelve [caja, [conf max por modelo]]."""
    todas = [np.c_[d, np.full(len(d), k)] for k, d in enumerate(dets_por_modelo) if len(d)]
    if not todas:
        return []
    D = np.vstack(todas); D = D[np.argsort(-D[:, 4])]
    usado = np.zeros(len(D), bool); res = []
    for i in range(len(D)):
        if usado[i]:
            continue
        o, s, r = iou_ios(D[i], D)
        m = (~usado) & ((o > iou) | ((s > ios) & (r < ar) & (r > 1 / ar)))
        m[i] = True
        cs = np.zeros(len(dets_por_modelo))
        for j in np.where(m)[0]:
            cs[int(D[j, 5])] = max(cs[int(D[j, 5])], D[j, 4])
        usado |= m
        res.append((D[i, :4], cs))
    return res


REGLAS = {'max': lambda c: c.max(), 'por': lambda c: 1 - np.prod(1 - c), 'media': lambda c: c.mean()}


def barrer(datos, fuente):
    """fuente(pl) -> array Nx5. Devuelve {th: (fn, fp)} y la lista de FN a 0,25."""
    cache = {pl: fuente(pl) for pl in datos}
    res = {}; fn25 = []
    for th in THS:
        fn = fp = 0
        for pl, (gt, _) in datos.items():
            d = cache[pl]; d = d[d[:, 4] >= th] if len(d) else d
            tp, f, p = rs.match_rapido(gt, [list(x) for x in d])
            fn += len(f); fp += len(p)
            if th == .25:
                fn25 += ['%s:%s' % (pl[:10], gt[i]['n'].split('$')[-1][:16]) for i in f]
        res[th] = (fn, fp)
    return res, fn25


def informe(nombre, res, fn25):
    ok = [th for th in THS if res[th][0] == 0]
    th100 = max(ok) if ok else None
    s = '%-28s @0.05 FN=%2d FP=%5d | @0.25 FN=%2d FP=%5d | th100=%s' % (
        nombre, res[.05][0], res[.05][1], res[.25][0], res[.25][1],
        '%.2f (FP=%d)' % (th100, res[th100][1]) if th100 else '-')
    print(s)
    if fn25 and len(fn25) <= 12:
        print('      FN@0.25:', ', '.join(fn25))


def main():
    tags = sys.argv[1:] or ['V4_v13_R', 'V4_v16_U']
    datos = cargar(tags)
    print('GT total:', sum(len(g) for g, _ in datos.values()), '(tiene que ser 3143)')
    for t in tags:
        informe(t, *barrer(datos, lambda pl, t=t: datos[pl][1].get(t, np.zeros((0, 5)))))
    for a, b in itertools.combinations(tags, 2):
        for regla, f in REGLAS.items():
            def fuente(pl, a=a, b=b, f=f):
                cl = clusters([datos[pl][1].get(a, np.zeros((0, 5))), datos[pl][1].get(b, np.zeros((0, 5)))])
                return np.array([list(c) + [f(cs)] for c, cs in cl]).reshape(-1, 5)
            informe('%s+%s %s' % (a[3:], b[3:], regla), *barrer(datos, fuente))


if __name__ == '__main__':
    main()
