"""Compara modelos por puntaje: 0.7 recall (fuerte: cada 1% de FN resta 5% del termino), 0.2 umbral de confianza, 0.1 FP.
Uso: comparar.py tag1 tag2 ...   (lee work/eval/<tag>/*_detecciones.csv y recalcula con matching optimo)"""
import sys, os, csv, json
sys.path.insert(0, os.path.dirname(__file__))
from paths import WORK, BASE
WORK = os.environ.get("CLAUDIO_WORK", WORK)
from evaluate import PLANOS, load_gt, match
TH = (0.25, 0.20, 0.15, 0.10, 0.05)
def puntaje(R, th, fp, ngt):
    return 0.7*max(0, 1-5*(1-R)) + 0.2*(th/0.25) + 0.1*max(0, 1-fp/ngt)
def evaluar(tag):
    out = {'planos': {}, 'total': []}; G = {}
    for name, _, gtp in PLANOS:
        p = os.path.join(WORK, 'eval', tag, f'{name}_detecciones.csv')
        if not os.path.exists(p): continue
        gt = load_gt(name, os.path.join(BASE, gtp))
        dc = [[float(r['x1']), float(r['y1']), float(r['x2']), float(r['y2']), float(r['conf'])] for r in csv.DictReader(open(p))]
        filas = []
        for th in TH:
            tp, fn, fp = match(gt, [d for d in dc if d[4] >= th])
            filas.append(dict(th=th, gt=len(gt), tp=len(tp), fn=len(fn), fp=len(fp), fn_bloques=[gt[i]['n'] for i in fn]))
        out['planos'][name] = filas; G[name] = filas
    for k, th in enumerate(TH):
        g = sum(G[n][k]['gt'] for n in G); tp = sum(G[n][k]['tp'] for n in G); fp = sum(G[n][k]['fp'] for n in G)
        R = tp/max(1, g); out['total'].append(dict(th=th, recall=round(R, 4), fn=g-tp, fp=fp, puntaje=round(puntaje(R, th, fp, g), 4)))
    best = max(out['total'], key=lambda r: r['puntaje']); out['mejor'] = best
    return out
if __name__ == '__main__':
    res = {t: evaluar(t) for t in sys.argv[1:]}
    for t, r in res.items():
        print(f'\n=== {t}  mejor puntaje {r["mejor"]["puntaje"]} @ conf {r["mejor"]["th"]}')
        print('conf   ' + '  '.join(f'{n:>16}' for n in r['planos']) + '     TOTAL recall/FN/FP  puntaje')
        for k, th in enumerate(TH):
            cells = '  '.join(f'{r["planos"][n][k]["tp"]/r["planos"][n][k]["gt"]:6.1%} fn{r["planos"][n][k]["fn"]:3d} fp{r["planos"][n][k]["fp"]:3d}' for n in r['planos'])
            T = r['total'][k]; print(f'{th:.2f}  {cells}   {T["recall"]:6.1%} {T["fn"]:3d} {T["fp"]:3d}  {T["puntaje"]}')
    json.dump(res, open(os.path.join(WORK, 'reporte', 'comparacion.json'), 'w'), indent=1, ensure_ascii=False)
