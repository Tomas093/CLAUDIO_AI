"""Busca componentes que faltan en el GT por COPIA de simbolos ya etiquetados (30/09, GT v8).

Tomas: "hay un monton de filas de interruptores termomagneticos que no los toma algunos". Cada caja del GT es
una plantilla; se busca (NCC en el render limpio) otra aparicion identica que no tenga caja. Lo que aparece
es candidato a faltante y va a la grilla de revision (no entra solo al GT).

    py -3 src/gt_plantillas.py <plano> <prefijo_render>     # usa work/gt_v8/<prefijo>_limpio.png + _meta.pkl
"""
import sys, os, csv, json, pickle, collections
sys.path.insert(0, os.path.dirname(__file__))
import cv2, numpy as np
from paths import BASE
from render import cad2px, px2cad
import gt_v8

pl, pre = sys.argv[1], sys.argv[2]
UMB = float(os.environ.get('UMB_NCC', '0.88'))
gtp = [os.path.join(BASE, g) for n, d, g in gt_v8.planos() if n == pl][0]
G = list(csv.DictReader(open(gtp[:-4] + '_v8.csv', encoding='utf-8')))
N = list(csv.DictReader(open(gtp[:-4] + '_v8_neutras.csv', encoding='utf-8')))
meta = pickle.load(open('work/gt_v8/%s_meta.pkl' % pre, 'rb'))
im = cv2.imread('work/gt_v8/%s_limpio.png' % pre, cv2.IMREAD_GRAYSCALE)
if im.ndim == 3: im = im[:, :, 0]   # ultralytics parchea cv2.imread (llega por evaluate)
tinta = (255 - im).astype(np.float32)


def px(r):
    x0, y0 = cad2px(meta, float(r['x1']), float(r['y2'])); x1, y1 = cad2px(meta, float(r['x2']), float(r['y1']))
    return int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1))


cajas = [px(r) for r in G] + [px(r) for r in N]
ocup = np.zeros(im.shape, np.uint8)
for x0, y0, x1, y1 in cajas: ocup[max(0, y0):y1, max(0, x0):x1] = 1
# plantillas: por nombre de bloque hasta 3 ejemplares de tamano distinto
por = collections.defaultdict(list)
for r in G: por[r['block_name'].split('-')[0] if r['block_name'].startswith(('AUDIT7', 'V8')) else r['block_name']].append(px(r))
cand = []
for n, L in por.items():
    vistos = []
    for b in L:
        w, h = b[2] - b[0], b[3] - b[1]
        if w < 8 or h < 8 or w > 600 or h > 600: continue
        if any(abs(w - a) < 3 and abs(h - c) < 3 for a, c in vistos) and len(vistos) >= 1: continue
        vistos.append((w, h))
        if len(vistos) > 4: break
        T = tinta[b[1]:b[3], b[0]:b[2]]
        if T.std() < 5 or (T > 60).mean() < .03: continue
        R = cv2.matchTemplate(tinta, T, cv2.TM_CCOEFF_NORMED)
        ys, xs = np.where(R >= UMB)
        for y, x in zip(ys, xs):
            cy, cx = y + h // 2, x + w // 2
            if ocup[cy, cx]: continue
            cand.append((float(R[y, x]), x, y, x + w, y + h, n))
# NMS
cand.sort(key=lambda c: -c[0]); keep = []
for c in cand:
    if all(max(0, min(c[3], k[3]) - max(c[1], k[1])) * max(0, min(c[4], k[4]) - max(c[2], k[2])) < .3 * (c[3] - c[1]) * (c[4] - c[2]) for k in keep):
        keep.append(c)
out = []
for s, x0, y0, x1, y1, n in keep:
    a = px2cad(meta, x0, y1); b = px2cad(meta, x1, y0)
    out.append(dict(plano=pl, ncc=round(s, 3), plantilla=n, caja_cad=[a[0], a[1], b[0], b[1]], caja_px=[int(x0), int(y0), int(x1), int(y1)]))
json.dump(out, open('work/gt_v8/cand_plantillas_%s.json' % pre, 'w'), indent=0)
print(pl, len(out), collections.Counter(o['plantilla'] for o in out).most_common(15))
