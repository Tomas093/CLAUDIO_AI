"""Grilla numerada de los cambios del GT v8 para que Tomas los revise (30/09).

Cada celda: recorte del plano con el GT v8 en verde fino y el cambio en grueso:
  rojo = caja nueva / modificada en v8, azul = candidata por plantilla (NO esta en v8 todavia),
  gris = neutra, naranja punteado = caja que v8 SACO (estaba en lo que Tomas vio).
    py -3 src/grilla_v8.py
"""
import sys, os, csv, json, pickle, glob, collections
sys.path.insert(0, os.path.dirname(__file__))
import cv2, numpy as np
from paths import BASE
from render import cad2px
import gt_v8

W = 220
OUT = 'work/revision_tomas/GT_v8'
PRE = {'nyw-un-01 esquemas unifilares': 'nyw-un-01_v8', 'LU-UN-01 Esquemas Unifilares': 'LU-UN-01_v8',
       'fl_un_02': 'fl_un_02_v8', 'tsss_2': 'tsss_2_v8', 'test1': 'test1_v8', 'test_2': 'test_2_v8',
       'EZE4077-IE-UF001-02': 'EZE4077-IE-UF001-02_v8'}
cajas = lambda r: tuple(float(r[k]) for k in ('x1', 'y1', 'x2', 'y2'))


def main():
    os.makedirs(OUT, exist_ok=True)
    items = []   # (plano, caja, color, texto, grupo)
    for pl, dxf, gtp in gt_v8.planos():
        gtp = os.path.join(BASE, gtp)
        v8 = list(csv.DictReader(open(gtp[:-4] + '_v8.csv', encoding='utf-8')))
        neu = list(csv.DictReader(open(gtp[:-4] + '_v8_neutras.csv', encoding='utf-8')))
        b8 = gt_v8.base_de(pl); vio = gt_v8.leer(b8) if b8 else []
        vset = {cajas(r) for r in vio}
        for r in v8:
            n = r['block_name']
            if n.startswith('V8-toma'): items.append((pl, cajas(r), 'r', 'TOMA', 1))
            elif n.startswith('V8-'): items.append((pl, cajas(r), 'r', n[3:], 3))
            elif n in gt_v8.RES and cajas(r) not in vset: items.append((pl, cajas(r), 'r', 'RES', 4))
            elif n == 'AUDIT-contactor' and cajas(r) not in vset: items.append((pl, cajas(r), 'r', 'KC', 4))
        v8set = [cajas(r) for r in v8]
        for r in vio:
            b = cajas(r)
            if not any(gt_v8.iou(b, c) > .5 for c in v8set) and r['block_name'] not in gt_v8.RES and r['block_name'] != 'AUDIT-contactor':
                items.append((pl, b, 'o', 'sacada', 2))
        for r in neu: items.append((pl, cajas(r), 'g', 'neutra', 5))
        f = 'work/gt_v8/cand_plantillas_%s.json' % PRE[pl]
        if os.path.exists(f):
            for c in json.load(open(f)): items.append((pl, tuple(c['caja_cad']), 'b', 'cand %.2f' % c['ncc'], 6))
    items.sort(key=lambda x: (x[4], x[0], -x[1][3], x[1][0]))
    cache = {}; celdas = []; lista = []
    for i, (pl, b, col, txt, grp) in enumerate(items, 1):
        if pl not in cache:
            cache = {pl: (cv2.imread('work/gt_v8/%s.png' % PRE[pl].replace('_v8', '_v8')), pickle.load(open('work/gt_v8/%s_meta.pkl' % PRE[pl], 'rb')))}
        im, meta = cache[pl]
        x0, y0 = cad2px(meta, b[0], b[3]); x1, y1 = cad2px(meta, b[2], b[1])
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2; lado = max(90, 1.8 * max(x1 - x0, y1 - y0))
        a, c = int(cx - lado / 2), int(cy - lado / 2); L = int(lado)
        crop = np.full((L, L, 3), 255, np.uint8)
        sy0, sx0 = max(0, c), max(0, a); sy1, sx1 = min(im.shape[0], c + L), min(im.shape[1], a + L)
        crop[sy0 - c:sy1 - c, sx0 - a:sx1 - a] = im[sy0:sy1, sx0:sx1]
        # el render _v8.png ya trae el GT v8 en magenta; lo paso a verde fino
        m = (crop[:, :, 0] == 255) & (crop[:, :, 1] == 0) & (crop[:, :, 2] == 255)
        crop[m] = (0, 170, 0)
        m = (crop[:, :, 0] == 0) & (crop[:, :, 1] == 0) & (crop[:, :, 2] == 255)
        crop[m] = (0, 170, 0)
        s = W / L; crop = cv2.resize(crop, (W, W), interpolation=cv2.INTER_AREA)
        C = {'r': (0, 0, 255), 'b': (255, 0, 0), 'g': (130, 130, 130), 'o': (0, 140, 255)}[col]
        p0 = (int((x0 - a) * s), int((y0 - c) * s)); p1 = (int((x1 - a) * s), int((y1 - c) * s))
        cv2.rectangle(crop, p0, p1, C, 2)
        cv2.rectangle(crop, (0, 0), (46, 20), (255, 255, 255), -1)
        cv2.putText(crop, str(i), (2, 15), 0, .5, (200, 0, 0), 1)
        cv2.putText(crop, txt, (50, 15), 0, .4, C, 1)
        celdas.append(crop); lista.append('%d\t%s\t%s\t%s' % (i, pl, txt, ','.join('%.4f' % v for v in b)))
    for pg in range(0, len(celdas), 64):
        cel = celdas[pg:pg + 64]; cols = 8
        G = np.full(((len(cel) + cols - 1) // cols * (W + 4), cols * (W + 4), 3), 90, np.uint8)
        for j, cc in enumerate(cel): G[(j // cols) * (W + 4):(j // cols) * (W + 4) + W, (j % cols) * (W + 4):(j % cols) * (W + 4) + W] = cc
        cv2.imwrite(os.path.join(OUT, 'v8_pag%02d.png' % (pg // 64 + 1)), G)
    open(os.path.join(OUT, 'lista.txt'), 'w', encoding='utf-8').write('\n'.join(lista))
    print(len(items), collections.Counter(x[3] if x[4] != 6 else 'cand' for x in items))


if __name__ == '__main__':
    main()
