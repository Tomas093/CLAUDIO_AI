"""Confirma con RF4 las pseudo-etiquetas de los zips manuales (data/zips_pseudo.json, modelo K del 18/09).

28/09, pedido de Tomas (opcion "b"): en una muestra de 48 pseudo-etiquetas >= 0,80, 2 no eran componentes
(circulo numerado "17", texto "2x25A 30mA") y 2 marcaban solo la cruz de un interruptor cortado por el borde
de la captura (Tomas: "que este todo el componente, o el 90% por lo menos"). Para cada candidata >= ALTA se
guarda la mayor confianza de RF4 sobre ella (IoU >= 0,5) y si toca el borde de la captura. build_all
(manual_tiles, env PSEUDO_CONFIRMA) deja como etiqueta solo las confirmadas que no tocan el borde; el resto
pasa a zona neutra (blanqueada), igual que lo dudoso.

La captura se pasa a la escala del armado (mediana de las cajas = 70 px, `resize_keep_strokes` sobre
`ink_map`) y se infiere en tiles de 640 con paso 320, de a 4 (la GPU esta compartida con un entrenamiento).

    venv_rfdetr/Scripts/python.exe src/confirmar_pseudo.py best_componente_v24_RF4.pth
Salida: data/zips_pseudo_confirmadas.json  {clave: [[x0,y0,x1,y1, confK, confRF4, toca_borde], ...]}
"""
import os, sys, json, cv2, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from paths import DATA, WORK
from postproc import ink_map, resize_keep_strokes

ALTA = float(os.environ.get('PSEUDO_ALTA', '0.80'))


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1])); I = ix * iy
    return I / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - I + 1e-9)


def inferir(modelo, img, lote=4, T=640, P=320):
    H, W = img.shape
    ys = list(range(0, max(1, H - T) + 1, P)); xs = list(range(0, max(1, W - T) + 1, P))
    if ys[-1] + T < H: ys.append(H - T)
    if xs[-1] + T < W: xs.append(W - T)
    tiles, offs, dets = [], [], []

    def correr():
        for r, (ox, oy) in zip(modelo.predict(tiles, conf=.05), offs):
            for b, c in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist()):
                dets.append([b[0] + ox, b[1] + oy, b[2] + ox, b[3] + oy, c])
        tiles.clear(); offs.clear()
    for y in ys:
        for x in xs:
            t = img[max(0, y):y + T, max(0, x):x + T]
            if t.shape != (T, T):
                t = cv2.copyMakeBorder(t, 0, T - t.shape[0], 0, T - t.shape[1], cv2.BORDER_CONSTANT, value=255)
            if (t < 128).sum() == 0: continue
            tiles.append(cv2.cvtColor(t, cv2.COLOR_GRAY2BGR)); offs.append((max(0, x), max(0, y)))
            if len(tiles) == lote: correr()
    if tiles: correr()
    return dets


def main():
    from evaluate import RFDETRComoYOLO
    modelo = RFDETRComoYOLO(sys.argv[1])
    ps = json.load(open(os.path.join(DATA, 'zips_pseudo.json')))
    res = json.load(open(os.path.join(DATA, 'zips_merged.json')))
    zdir = os.path.join(WORK, 'zips'); out = {}; n = 0
    claves = [k for k, v in ps.items() if any(len(c) >= 5 and c[4] >= ALTA for c in v.get('cand', []))]
    for i, key in enumerate(claves):
        pv = ps[key]; W0, H0 = float(pv['w']), float(pv['h'])
        im = cv2.imread(os.path.join(zdir, *key.split('/')))
        if im is None: continue
        H, W = im.shape[:2]
        cand = [c for c in pv['cand'] if len(c) >= 5 and c[4] >= ALTA]
        g, a, _ = res.get(key, [[], [], []])
        lados = [max((b[2] - b[0]) * W, (b[3] - b[1]) * H) for b in g + a] + \
                [max((c[2] - c[0]) * W / W0, (c[3] - c[1]) * H / H0) for c in cand]
        s = 70.0 / max(1.0, float(np.median(lados)))
        img = resize_keep_strokes(ink_map(im), s)
        dets = inferir(modelo, img)
        fila = []
        for c in cand:
            q = [c[0] / W0 * W * s, c[1] / H0 * H * s, c[2] / W0 * W * s, c[3] / H0 * H * s]
            rf = max([d[4] for d in dets if iou(d, q) >= .5], default=0.0)
            e = 3.0 / W0
            borde = c[0] / W0 <= e or c[1] / H0 <= e or c[2] / W0 >= 1 - e or c[3] / H0 >= 1 - e
            fila.append([c[0], c[1], c[2], c[3], c[4], round(rf, 3), bool(borde)])
            n += 1
        out[key] = fila
        if i % 100 == 0: print('[confirmar] %d/%d capturas, %d candidatas' % (i, len(claves), n), flush=True)
    dst = os.path.join(DATA, 'zips_pseudo_confirmadas.json')
    json.dump(out, open(dst + '.tmp', 'w')); os.replace(dst + '.tmp', dst)
    todas = [f for v in out.values() for f in v]
    ok = [f for f in todas if f[5] >= .3 and not f[6]]
    print('[confirmar] candidatas %d | confirmadas por RF4 (>=0,3) y lejos del borde %d | tocan borde %d | RF4 no las ve %d'
          % (len(todas), len(ok), sum(f[6] for f in todas), sum(f[5] < .3 for f in todas)))


if __name__ == '__main__':
    main()
