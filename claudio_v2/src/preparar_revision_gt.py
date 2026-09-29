# -*- coding: utf-8 -*-
"""Prepara material para auditar la VERDAD DE TERRENO de los planos de evaluacion.

Por que importa mas que auditar las detecciones: el GT es la regla con la que se decide que
modelo es mejor. Si esta torcida, todas las comparaciones entre modelos lo estan y no hay
forma de darse cuenta desde adentro del bucle. Ya hubo dos precedentes: el GT de test_2 estaba
sin cajas y real_labels.json tenia 24% de cajas con aire.

Dos tareas, mismo criterio que preparar_revision.py:

  C (precision del GT): un recorte por caja del GT, con la caja en ROJO. Cuantos componentes
    encierra y que tan bien encuadra.

  D (completitud del GT): una celda del plano con TODAS las cajas del GT en AZUL. Que
    componente quedo SIN caja. Esta es la pregunta nueva: si el GT tiene faltantes, el recall
    de 99,66% es ficticio, porque lo que falta en el GT no se le reclama a ningun modelo.

  py -3 src/preparar_revision_gt.py

Salida en revision_gt/: tarea_c/, tarea_d/, manifest_c.jsonl, manifest_d.jsonl.

Nota de implementacion: se procesa UN plano por vez y el render se mantiene en gris, pasando a
color solo el recorte que se guarda. LU-UN-01 renderiza a 5076x15192, que en color son 231 MB;
manteniendo los siete planos a la vez el proceso se quedaba sin memoria y moria sin escribir
nada.
"""
import os, sys, csv, json, random
import numpy as np, cv2, ezdxf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import BASE
from render import render_doc, cad2px
from scale import auto_ppc
from postproc import darken

OUT = os.path.join(BASE, 'revision_gt')

N_C = int(os.environ.get('N_C', '300'))       # recortes de la tarea C, repartidos entre planos
N_D = int(os.environ.get('N_D', '170'))       # celdas de la tarea D, repartidas entre planos
CELDA = int(os.environ.get('CELDA', '1100'))
LADO_C = 640

PLANOS = [
    ('test1', 'test1.dxf', 'test/test_1/verdad_terreno/test1_completo.csv'),
    ('test_2', 'test_2.dxf', 'test/test_2/verdad_terreno/test_2_completo.csv'),
    ('fl_un_02', 'dxf/FL-UN-02_tablero_1.dxf', 'dxf/fl_un_02_gt_completo.csv'),
    ('tsss_2', 'TSSS_2 (1).dxf', 'dxf/tsss_2_gt_completo.csv'),
    ('EZE4077', 'claudio_v2/data/test_marcelo/EZE4077-IE-UF001-02.dxf',
     'claudio_v2/data/test_marcelo/EZE4077-IE-UF001-02_gt.csv'),
    ('LU-UN-01', 'claudio_v2/data/test_marcelo/LU-UN-01 Esquemas Unifilares.dxf',
     'claudio_v2/data/test_marcelo/LU-UN-01 Esquemas Unifilares_gt.csv'),
    ('nyw-un-01', 'claudio_v2/data/test_marcelo/nyw-un-01 esquemas unifilares.dxf',
     'claudio_v2/data/test_marcelo/nyw-un-01 esquemas unifilares_gt.csv'),
]


def inter(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    return 0. if (x1 <= x0 or y1 <= y0) else (x1 - x0) * (y1 - y0)


def area(a):
    return max(1e-9, (a[2] - a[0]) * (a[3] - a[1]))


def cargar_uno(nom, dxf, gtp):
    """(imagen_gris, [caja_px,...], [bloque,...]) o None."""
    fd, fg = os.path.join(BASE, dxf), os.path.join(BASE, gtp)
    if not (os.path.exists(fd) and os.path.exists(fg)):
        print('[aviso] falta %s' % nom)
        return None
    doc = ezdxf.readfile(fd)
    ppc = auto_ppc(doc)
    if ppc is None:
        print('[aviso] %s: sin texto para autoescalar' % nom)
        return None
    img, meta = render_doc(doc, ppc)
    img = darken(img)
    cajas, nombres = [], []
    with open(fg, newline='', encoding='utf-8', errors='ignore') as f:
        for r in csv.DictReader(f):
            if not r.get('x1'):
                continue      # fila sin caja: no se puede recortar
            ax, ay = cad2px(meta, float(r['x1']), float(r['y1']))
            bx, by = cad2px(meta, float(r['x2']), float(r['y2']))
            cajas.append([min(ax, bx), min(ay, by), max(ax, bx), max(ay, by)])
            nombres.append(r.get('block_name', '?'))
    if not cajas:
        print('[aviso] %s: el GT no tiene ninguna caja' % nom)
        return None
    print('  %-12s %5d cajas de GT | render %dx%d' % (nom, len(cajas), img.shape[1], img.shape[0]))
    return img, cajas, nombres


def recortes_c(nom, img, cajas, nombres, cupo, r, d_out, k0):
    idx = list(range(len(cajas)))
    r.shuffle(idx)
    H, W = img.shape[:2]
    man = []
    k = k0
    for i in idx[:cupo]:
        b = cajas[i]
        cx, cy = (b[0] + b[2]) / 2., (b[1] + b[3]) / 2.
        lado = max(b[2] - b[0], b[3] - b[1])
        m = max(110., lado * 2.6)
        x0, y0 = int(max(0, cx - m / 2)), int(max(0, cy - m / 2))
        x1, y1 = int(min(W, cx + m / 2)), int(min(H, cy + m / 2))
        if x1 - x0 < 12 or y1 - y0 < 12:
            continue
        cr = cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_GRAY2BGR)
        gr = max(1, int(round(lado * .035)))
        cv2.rectangle(cr, (int(b[0]) - x0, int(b[1]) - y0), (int(b[2]) - x0, int(b[3]) - y0),
                      (0, 0, 255), gr)
        f = float(LADO_C) / max(cr.shape[0], cr.shape[1])
        cr = cv2.resize(cr, None, fx=f, fy=f,
                        interpolation=cv2.INTER_CUBIC if f > 1 else cv2.INTER_AREA)
        fn = 'c_%04d.png' % k
        k += 1
        cv2.imwrite(os.path.join(d_out, fn), cr)
        man.append({'id': fn, 'plano': nom, 'idx': i, 'bloque': nombres[i],
                    'caja': [round(v, 1) for v in b]})
    return man, k


def celdas_d(nom, img, cajas, cupo, r, d_out, k0):
    H, W = img.shape[:2]
    paso = max(1, int(CELDA * .88))
    pos = []
    for y0 in range(0, max(1, H - 1), paso):
        for x0 in range(0, max(1, W - 1), paso):
            x1, y1 = min(W, x0 + CELDA), min(H, y0 + CELDA)
            if x1 - x0 < CELDA // 2 or y1 - y0 < CELDA // 2:
                continue
            if (img[y0:y1, x0:x1] < 160).mean() < .004:
                continue      # celda casi en blanco
            pos.append((x0, y0, x1, y1))
    r.shuffle(pos)
    man = []
    k = k0
    for x0, y0, x1, y1 in pos[:cupo]:
        cr = cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_GRAY2BGR)
        dentro = 0
        for b in cajas:
            if inter(b, [x0, y0, x1, y1]) / area(b) < .55:
                continue
            cv2.rectangle(cr, (int(b[0]) - x0, int(b[1]) - y0),
                          (int(b[2]) - x0, int(b[3]) - y0), (255, 40, 0), 2)
            dentro += 1
        f = 1000.0 / max(cr.shape[0], cr.shape[1])
        if f < 1:
            cr = cv2.resize(cr, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
        else:
            f = 1.0
        fn = 'd_%04d.png' % k
        k += 1
        cv2.imwrite(os.path.join(d_out, fn), cr)
        man.append({'id': fn, 'plano': nom, 'origen': [x0, y0], 'escala': round(f, 4),
                    'cajas_dibujadas': dentro})
    return man, k


if __name__ == '__main__':
    dc = os.path.join(OUT, 'tarea_c')
    dd = os.path.join(OUT, 'tarea_d')
    os.makedirs(dc, exist_ok=True)
    os.makedirs(dd, exist_ok=True)
    cupo_c = max(1, N_C // len(PLANOS))
    cupo_d = max(1, N_D // len(PLANOS))
    print('renderizando los planos de evaluacion (uno por vez)...')
    print('cupo: %d recortes y %d celdas por plano' % (cupo_c, cupo_d))
    man_c, man_d = [], []
    kc = kd = 0
    r = random.Random(0)
    for nom, dxf, gtp in PLANOS:
        d = cargar_uno(nom, dxf, gtp)
        if d is None:
            continue
        img, cajas, nombres = d
        mc, kc = recortes_c(nom, img, cajas, nombres, cupo_c, r, dc, kc)
        md, kd = celdas_d(nom, img, cajas, cupo_d, r, dd, kd)
        man_c.extend(mc)
        man_d.extend(md)
        print('     -> %d recortes, %d celdas' % (len(mc), len(md)))
        del img, cajas, nombres, d      # el render grande se libera antes del proximo plano
    with open(os.path.join(OUT, 'manifest_c.jsonl'), 'w', encoding='utf-8') as fh:
        for x in man_c:
            fh.write(json.dumps(x, ensure_ascii=False) + '\n')
    with open(os.path.join(OUT, 'manifest_d.jsonl'), 'w', encoding='utf-8') as fh:
        for x in man_d:
            fh.write(json.dumps(x, ensure_ascii=False) + '\n')
    print('tarea C: %d recortes -> revision_gt/tarea_c' % len(man_c))
    print('tarea D: %d celdas   -> revision_gt/tarea_d' % len(man_d))
