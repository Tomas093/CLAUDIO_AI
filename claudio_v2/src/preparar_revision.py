# -*- coding: utf-8 -*-
"""Prepara el material para que un modelo de vision audite las detecciones sobre los PDF.

Dos tareas, separadas a proposito:

  A (precision / cajas dobles): un recorte por deteccion, con la caja en ROJO y contexto
    alrededor. La pregunta es cerrada -- cuantos componentes hay dentro del rojo -- que es
    lo que un VLM contesta bien.

  B (recall / faltantes): una celda del plano con TODAS las detecciones ya dibujadas en
    AZUL. La pregunta es que quedo SIN marcar. Pedirle que detecte desde cero en un plano
    denso lo hace alucinar; pedirle que busque huecos sobre un trabajo ya hecho, no.

Salida: revision_ia/tarea_a/*.png, revision_ia/tarea_b/*.png y los manifest .jsonl con las
coordenadas para poder mapear las respuestas de vuelta al plano.
"""
import os, sys, csv, glob, json, random
import numpy as np, cv2

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAIZ = os.path.dirname(BASE)
DET = os.path.join(BASE, 'work', 'det_N_pdf')
PDFS = sys.argv[1] if len(sys.argv) > 1 else os.path.join(BASE, 'data', 'planos_marcelo')
OUT = os.path.join(RAIZ, 'revision_ia')

sys.path.insert(0, BASE)
import detectar as DT

# El csv guarda las cajas en PUNTOS PDF (`px2pdf`), pero el render con el que trabaja el
# detector esta en pixeles. Se vuelve a renderizar cada PDF con la misma funcion del
# detector -- es CPU puro, no toca la GPU -- y se invierte px2pdf con su propia meta, en
# vez de adivinar la escala a partir del visual.png ya dibujado.
DPI_FIJO = {'ByA-IE-EU-Transferencia Unidades-Model': 160}   # PDF con el texto vectorizado


def pdf2px(meta, X, Y):
    """Inversa de detectar.px2pdf."""
    a, b = X * meta['esc'], Y * meta['esc']
    if meta.get('giro') == 90:
        return (meta['Wr'] - 1) - b, a
    return a, b

N_A = int(os.environ.get('N_A', '260'))       # recortes de la tarea A
CELDA = int(os.environ.get('CELDA', '1100'))  # lado de la celda de la tarea B, en px del plano
SOLAPE = 0.12
LADO_A = 640                                  # lado del png que se le manda al modelo


def inter(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    return 0. if (x1 <= x0 or y1 <= y0) else (x1 - x0) * (y1 - y0)


def area(a):
    return max(1e-9, (a[2] - a[0]) * (a[3] - a[1]))


def cargar():
    """{nombre: (imagen_gris_del_render, [deteccion_en_px,...])}."""
    disp = {}
    for p in glob.glob(os.path.join(PDFS, '*.pdf')):
        disp[os.path.splitext(os.path.basename(p))[0]] = p
    out = {}
    for f in sorted(glob.glob(os.path.join(DET, '*_detecciones.csv'))):
        nom = os.path.basename(f).replace('_detecciones.csv', '')
        if nom not in disp:
            print('[aviso] sin pdf para %s' % nom)
            continue
        D = [[float(r['x1']), float(r['y1']), float(r['x2']), float(r['y2']), float(r['conf'])]
             for r in csv.DictReader(open(f, encoding='utf-8'))]
        if not D:
            continue
        try:
            img, meta = DT.render_pdf(disp[nom], dpi=DPI_FIJO.get(nom))
        except Exception as e:
            print('[aviso] no renderiza %s: %s' % (nom, str(e)[:90]))
            continue
        px = []
        for d in D:
            ax, ay = pdf2px(meta, d[0], d[1])
            bx, by = pdf2px(meta, d[2], d[3])
            px.append([min(ax, bx), min(ay, by), max(ax, bx), max(ay, by), d[4]])
        out[nom] = (cv2.cvtColor(DT.darken(img), cv2.COLOR_GRAY2BGR), px)
    return out


def sospechosas(D):
    """Razon por la que cada deteccion es sospechosa (cadena vacia si no lo es)."""
    med = np.median([max(d[2] - d[0], d[3] - d[1]) for d in D])
    raz = [''] * len(D)
    for i, d in enumerate(D):
        hijos = [j for j, e in enumerate(D)
                 if j != i and inter(d, e) / area(e) > .80 and area(e) < .6 * area(d)]
        indep = []
        for j in hijos:
            if all(inter(D[j], D[k]) / min(area(D[j]), area(D[k])) < .3 for k in indep):
                indep.append(j)
        if len(indep) >= 2:
            raz[i] = 'engloba'
        elif max(d[2] - d[0], d[3] - d[1]) > 4 * med:
            raz[i] = 'gigante'
        else:
            for j, e in enumerate(D):
                if j == i:
                    continue
                I = inter(d, e)
                if I > 0 and I / (area(d) + area(e) - I) > .35:
                    raz[i] = 'solapa'
                    break
    return raz


def tarea_a(datos, r):
    d_out = os.path.join(OUT, 'tarea_a')
    os.makedirs(d_out, exist_ok=True)
    cand = []
    for nom, (im, D) in datos.items():
        raz = sospechosas(D)
        for i, d in enumerate(D):
            cand.append((nom, i, d, raz[i]))
    sosp = [c for c in cand if c[3]]
    resto = [c for c in cand if not c[3]]
    r.shuffle(resto)
    # todas las sospechosas + una muestra al azar, para tener tasa base con que comparar
    sel = sosp + resto[:max(0, N_A - len(sosp))]
    print('[A] %d sospechosas + %d al azar = %d recortes' % (len(sosp), len(sel) - len(sosp), len(sel)))
    man = []
    for k, (nom, i, d, raz) in enumerate(sel):
        im = datos[nom][0]
        H, W = im.shape[:2]
        cx, cy = (d[0] + d[2]) / 2., (d[1] + d[3]) / 2.
        lado = max(d[2] - d[0], d[3] - d[1])
        m = max(110., lado * 2.6)
        x0, y0 = int(max(0, cx - m / 2)), int(max(0, cy - m / 2))
        x1, y1 = int(min(W, cx + m / 2)), int(min(H, cy + m / 2))
        if x1 - x0 < 12 or y1 - y0 < 12:
            continue
        cr = im[y0:y1, x0:x1].copy()
        gr = max(1, int(round(lado * 0.035)))
        cv2.rectangle(cr, (int(d[0]) - x0, int(d[1]) - y0), (int(d[2]) - x0, int(d[3]) - y0),
                      (0, 0, 255), gr)
        f = float(LADO_A) / max(cr.shape[0], cr.shape[1])
        cr = cv2.resize(cr, None, fx=f, fy=f,
                        interpolation=cv2.INTER_CUBIC if f > 1 else cv2.INTER_AREA)
        fn = 'a_%04d.png' % k
        cv2.imwrite(os.path.join(d_out, fn), cr)
        man.append({'id': fn, 'plano': nom, 'idx': i, 'caja': [round(v, 1) for v in d[:4]],
                    'conf': round(d[4], 3), 'motivo': raz or 'muestra'})
    with open(os.path.join(OUT, 'manifest_a.jsonl'), 'w', encoding='utf-8') as fh:
        for m in man:
            fh.write(json.dumps(m, ensure_ascii=False) + '\n')
    return len(man)


def tarea_b(datos):
    d_out = os.path.join(OUT, 'tarea_b')
    os.makedirs(d_out, exist_ok=True)
    man = []
    k = 0
    for nom, (im, D) in datos.items():
        H, W = im.shape[:2]
        paso = max(1, int(CELDA * (1 - SOLAPE)))
        for y0 in range(0, max(1, H - 1), paso):
            for x0 in range(0, max(1, W - 1), paso):
                x1, y1 = min(W, x0 + CELDA), min(H, y0 + CELDA)
                if x1 - x0 < CELDA // 2 or y1 - y0 < CELDA // 2:
                    continue
                cr = im[y0:y1, x0:x1]
                # celda casi en blanco: no tiene sentido preguntarla
                if (cv2.cvtColor(cr, cv2.COLOR_BGR2GRAY) < 160).mean() < .004:
                    continue
                cr = cr.copy()
                dentro = 0
                for d in D:
                    if inter(d, [x0, y0, x1, y1]) / area(d) < .55:
                        continue
                    cv2.rectangle(cr, (int(d[0]) - x0, int(d[1]) - y0),
                                  (int(d[2]) - x0, int(d[3]) - y0), (255, 40, 0), 2)
                    dentro += 1
                f = 1000.0 / max(cr.shape[0], cr.shape[1])
                if f < 1:
                    cr = cv2.resize(cr, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
                else:
                    f = 1.0
                fn = 'b_%04d.png' % k
                k += 1
                cv2.imwrite(os.path.join(d_out, fn), cr)
                man.append({'id': fn, 'plano': nom, 'origen': [x0, y0], 'escala': round(f, 4),
                            'cajas_dibujadas': dentro})
    with open(os.path.join(OUT, 'manifest_b.jsonl'), 'w', encoding='utf-8') as fh:
        for m in man:
            fh.write(json.dumps(m, ensure_ascii=False) + '\n')
    return k


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    datos = cargar()
    print('planos con detecciones e imagen: %d' % len(datos))
    r = random.Random(0)
    na = tarea_a(datos, r)
    nb = tarea_b(datos)
    print('tarea A: %d recortes -> revision_ia/tarea_a' % na)
    print('tarea B: %d celdas   -> revision_ia/tarea_b' % nb)
