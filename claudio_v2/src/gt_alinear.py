"""Detecta cajas del GT CORRIDAS respecto del simbolo (30/09, GT v8; Tomas: "hay varios bb mal en el nyw").

Por nombre de bloque (>= 5 cajas del mismo tamano): plantilla = mediana pixel a pixel de los recortes (el simbolo
bien centrado domina). Cada caja se busca en una ventana de +-60% alrededor; si la mejor posicion tiene NCC >= 0,9,
esta corrida >= 12% del lado y mejora al menos 0,15 la NCC de la posicion actual, se propone moverla.
Tambien confirma los candidatos por plantilla (gt_plantillas.py) con NCC >= 0,95: si pisan una caja corrida la
reemplazan; si no pisan nada son faltantes.
    py -3 src/gt_alinear.py <plano> <prefijo_render>   -> work/gt_v8/realineos_<prefijo>.json
"""
import sys, os, csv, json, pickle, collections
sys.path.insert(0, os.path.dirname(__file__))
import cv2, numpy as np
from paths import BASE
from render import cad2px, px2cad
import gt_v8

pl, pre = sys.argv[1], sys.argv[2]
gtp = [os.path.join(BASE, g) for n, d, g in gt_v8.planos() if n == pl][0]
G = list(csv.DictReader(open(gtp[:-4] + '_v8.csv', encoding='utf-8')))
meta = pickle.load(open('work/gt_v8/%s_meta.pkl' % pre, 'rb'))
im = cv2.imread('work/gt_v8/%s_limpio.png' % pre, cv2.IMREAD_GRAYSCALE)
if im.ndim == 3: im = im[:, :, 0]   # ultralytics parchea cv2.imread (llega por evaluate)
tinta = (255 - im).astype(np.float32)
H, Wimg = im.shape


def px(r):
    x0, y0 = cad2px(meta, float(r['x1']), float(r['y2'])); x1, y1 = cad2px(meta, float(r['x2']), float(r['y1']))
    return int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1))


por = collections.defaultdict(list)
for i, r in enumerate(G):
    b = px(r); por[(r['block_name'] if not r['block_name'].startswith(('AUDIT7', 'V8')) else 'x', b[2] - b[0], b[3] - b[1])].append((i, b))
# agrupar tamanos casi iguales (+-2 px) del mismo bloque
grupos = collections.defaultdict(list)
for (n, w, h), L in por.items(): grupos[(n, round(w / 3), round(h / 3))] += L
plantillas = []
for (n, _, _), L in grupos.items():
    if n == 'x' or len(L) < 5: continue
    w = min(b[2] - b[0] for _, b in L); h = min(b[3] - b[1] for _, b in L)
    if w < 10 or h < 10: continue
    crops = [tinta[b[1]:b[1] + h, b[0]:b[0] + w] for _, b in L if b[1] >= 0 and b[0] >= 0 and b[1] + h <= H and b[0] + w <= Wimg]
    T = np.median(np.stack(crops), 0).astype(np.float32)
    if T.std() >= 3: plantillas.append((n, w, h, T, L))
# las cajas agregadas a mano/auditoria ('x') se prueban contra las plantillas de tamano parecido (+-15%)
sueltas = [(i, b) for (n, _, _), L in grupos.items() if n == 'x' for i, b in L]
movs = []
def probar(i, b, T, w, h):
    mx, my = int(.6 * w), int(.6 * h)
    X0, Y0 = max(0, b[0] - mx), max(0, b[1] - my); X1, Y1 = min(Wimg, b[0] + w + mx), min(H, b[1] + h + my)
    V = tinta[Y0:Y1, X0:X1]
    if V.shape[0] < h or V.shape[1] < w: return None
    R = cv2.matchTemplate(V, T, cv2.TM_CCOEFF_NORMED)
    y, x = np.unravel_index(np.argmax(R), R.shape); best = float(R[y, x])
    cy, cx = b[1] - Y0, b[0] - X0
    aqui = float(R[cy, cx]) if 0 <= cy < R.shape[0] and 0 <= cx < R.shape[1] else -1
    return best, aqui, X0 + x - b[0], Y0 + y - b[1]
for n, w, h, T, L in plantillas:
    todos = L + [(i, b) for i, b in sueltas if abs((b[2] - b[0]) - w) <= .15 * w and abs((b[3] - b[1]) - h) <= .15 * h]
    for i, b in todos:
        res = probar(i, b, T, w, h)
        if not res: continue
        best, aqui, dx, dy = res
        if best >= .9 and (abs(dx) >= .12 * w or abs(dy) >= .12 * h) and best - aqui >= .15:
            nb = (b[0] + dx, b[1] + dy, b[0] + dx + (b[2] - b[0]), b[1] + dy + (b[3] - b[1]))
            a = px2cad(meta, nb[0], nb[3]); c = px2cad(meta, nb[2], nb[1])
            r = G[i]
            movs.append(dict(de=[float(r[k]) for k in ('x1', 'y1', 'x2', 'y2')], a=[a[0], a[1], c[0], c[1]], bloque=r['block_name'],
                             plantilla=n, ncc=round(best, 3), ncc_antes=round(aqui, 3), dpx=[int(dx), int(dy)]))
# una caja puede salir con dos plantillas: quedarse con la de mayor NCC
m2 = {}
for m in movs:
    k = tuple(m['de'])
    if k not in m2 or m['ncc'] > m2[k]['ncc']: m2[k] = m
movs = list(m2.values())
json.dump(movs, open('work/gt_v8/realineos_%s.json' % pre, 'w'), indent=0)
print(pl, len(movs), collections.Counter(m['bloque'] for m in movs).most_common(10))
