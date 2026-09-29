"""Auditoria automatica de un dataset YOLO (work/dsNN): busca etiquetas raras y componentes sin etiqueta.

28/09, pedido de Tomas ("revisa el dataset si hay algun problema mas o cosa rara"). Chequeos por caja:
  vacia      menos de 2% de tinta adentro (la caja quedo sobre un hueco)
  suelta     la tinta ocupa < 55% del ancho o del alto de la caja (caja inflada, no cenida)
  diminuta   lado mayor < 6 px            gigante   lado mayor > 400 px
  alargada   relacion de lados > 8
  pisada     otra caja la cubre > 50% (anidadas / encimadas)
  cortada    toca el borde del tile (puede ser un simbolo a medias)
Por imagen:
  sin_etiqueta   un modelo ya entrenado (--modelo, por defecto R) detecta algo con conf >= 0,6 que no
                 coincide con ninguna caja (IoU < 0,3): candidato a componente sin etiquetar (o FP del modelo).
  neg_con_simbolo  lo mismo en tiles negativos (n*, rn*), donde no deberia haber ningun componente.
  no_la_ve       caja etiquetada que el modelo no ve ni a conf 0,05 (etiqueta sospechosa: texto, hueco...).
  (La primera version usaba una heuristica de manchas de tinta: marcaba 485.000 en ds20, inservible.)
Salida: <salida>/resumen.txt, problemas.csv y grillas numeradas por tipo (<tipo>.png) para revisar a ojo.

    py -3 src/auditar_ds.py work/ds21_muestra work/auditoria_ds21
"""
import os, sys, glob, csv, collections, random
import cv2, numpy as np

S = 640


def leer(lbl):
    out = []
    if os.path.exists(lbl):
        for ln in open(lbl):
            p = ln.split()
            if len(p) == 5:
                cx, cy, w, h = [float(v) * S for v in p[1:]]
                out.append([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
    return out


def gris(f):
    # ultralytics reemplaza cv2.imread al importarse y devuelve BGR aunque se pida gris
    a = cv2.imread(f, 0)
    if a is not None and a.ndim == 3:
        a = a[:, :, 0] if a.shape[2] == 1 else cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    return a


def fuente(nombre):
    for pre, f in (('rn', 'real_neg'), ('rp', 'real_pos'), ('n', 'sint_neg'), ('p', 'sint_pos'), ('m', 'manual')):
        if nombre.startswith(pre):
            return f
    return 'otro'


def candidatos_sin_etiqueta(img, boxes):
    tinta = (img < 140).astype(np.uint8)
    for b in boxes:          # lo etiquetado (con margen) no cuenta
        tinta[max(0, int(b[1]) - 3):int(b[3]) + 3, max(0, int(b[0]) - 3):int(b[2]) + 3] = 0
    n, lab, st, _ = cv2.connectedComponentsWithStats(cv2.dilate(tinta, np.ones((3, 3), np.uint8)), connectivity=8)
    out = []
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if not (12 <= max(w, h) <= 160 and min(w, h) >= 8):
            continue
        sub = tinta[y:y + h, x:x + w]
        dens = sub.mean()
        if dens < .06 or dens > .6:
            continue
        # texto suelto: muchas columnas vacias intercaladas y alto chico -> se descarta
        cols = (sub.sum(0) > 0).mean(); filas = (sub.sum(1) > 0).mean()
        if h < 16 and w > 2.5 * h:
            continue
        if cols < .5 or filas < .5:
            continue
        out.append([x, y, x + w, y + h, dens])
    return out


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1])); I = ix * iy
    return I / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - I + 1e-9)


def main():
    ds, sal = sys.argv[1], sys.argv[2]
    lim = int(sys.argv[3]) if len(sys.argv) > 3 else 0          # auditar solo N imagenes al azar (0 = todas)
    pesos = sys.argv[4] if len(sys.argv) > 4 else 'best_componente_v13_R.pt'
    from ultralytics import YOLO
    modelo = YOLO(pesos)
    os.makedirs(sal, exist_ok=True)
    probs = []; stats = collections.defaultdict(collections.Counter); tam = collections.defaultdict(list)
    ejemplos = collections.defaultdict(list)
    imgs = sorted(glob.glob(os.path.join(ds, 'images', '*', '*.png')))
    random.Random(0).shuffle(imgs)
    if lim: imgs = imgs[:lim]
    for f in imgs:
        split = os.path.basename(os.path.dirname(f)); nom = os.path.splitext(os.path.basename(f))[0]
        src = fuente(nom); img = gris(f)
        boxes = leer(os.path.join(ds, 'labels', split, nom + '.txt'))
        stats[src]['imagenes'] += 1; stats[src]['cajas'] += len(boxes)
        if not boxes and src in ('sint_pos', 'real_pos', 'manual'):
            stats[src]['pos_sin_cajas'] += 1
        for i, b in enumerate(boxes):
            x0, y0, x1, y1 = [int(round(v)) for v in b]
            w, h = x1 - x0, y1 - y0; tam[src].append(max(w, h))
            sub = img[max(0, y0):y1, max(0, x0):x1]
            tipos = []
            if sub.size == 0 or (sub < 128).mean() < .02:
                tipos.append('vacia')
            else:
                ys, xs = np.where(sub < 140)
                if len(xs) and ((xs.max() - xs.min() + 1) < .55 * sub.shape[1] or (ys.max() - ys.min() + 1) < .55 * sub.shape[0]):
                    tipos.append('suelta')
            if max(w, h) < 6: tipos.append('diminuta')
            if max(w, h) > 400: tipos.append('gigante')
            if max(w, h) > 8 * max(1, min(w, h)): tipos.append('alargada')
            if x0 <= 0 or y0 <= 0 or x1 >= S or y1 >= S: tipos.append('cortada')
            for j, q in enumerate(boxes):
                if j == i: continue
                ix = max(0, min(b[2], q[2]) - max(b[0], q[0])); iy = max(0, min(b[3], q[3]) - max(b[1], q[1]))
                if ix * iy > .5 * max(1, w * h):
                    tipos.append('pisada'); break
            for t in tipos:
                stats[src][t] += 1; probs.append([f, src, t, x0, y0, x1, y1])
                ejemplos[t].append((f, [x0, y0, x1, y1], boxes))
        r = modelo.predict(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), imgsz=S, conf=.05, verbose=False)[0]
        dets = [d + [c] for d, c in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist())]
        t = 'neg_con_simbolo' if src in ('sint_neg', 'real_neg') else 'sin_etiqueta'
        for d in dets:
            if d[4] >= .6 and all(iou(d, b) < .3 for b in boxes):
                c = [int(v) for v in d[:4]]
                stats[src][t] += 1; probs.append([f, src, t] + c)
                ejemplos[t].append((f, c, boxes))
        for b in boxes:
            if not any(iou(d, b) >= .3 for d in dets):
                c = [int(round(v)) for v in b]
                stats[src]['no_la_ve'] += 1; probs.append([f, src, 'no_la_ve'] + c)
                ejemplos['no_la_ve'].append((f, c, boxes))
    with open(os.path.join(sal, 'problemas.csv'), 'w', newline='') as h:
        w = csv.writer(h); w.writerow(['imagen', 'fuente', 'tipo', 'x0', 'y0', 'x1', 'y1']); w.writerows(probs)
    lin = ['dataset: %s  (%d imagenes)' % (ds, len(imgs)), '']
    tipos = ['cajas', 'pos_sin_cajas', 'vacia', 'suelta', 'diminuta', 'gigante', 'alargada', 'pisada', 'cortada', 'sin_etiqueta', 'neg_con_simbolo', 'no_la_ve']
    lin.append('%-10s %8s ' % ('fuente', 'imagenes') + ' '.join('%9s' % t[:9] for t in tipos) + '  p50px  <20px')
    for src, c in sorted(stats.items()):
        T = np.array(tam[src]) if tam[src] else np.array([0])
        lin.append('%-10s %8d ' % (src, c['imagenes']) + ' '.join('%9d' % c[t] for t in tipos) + '  %5.0f  %4.1f%%' % (np.median(T), 100 * (T < 20).mean()))
    open(os.path.join(sal, 'resumen.txt'), 'w').write('\n'.join(lin)); print('\n'.join(lin))
    # grillas numeradas por tipo (hasta 30 ejemplos): rojo = lo senalado, verde = las otras cajas
    for t, L in ejemplos.items():
        cel = []; ref = []
        for f, b, boxes in L[:int(os.environ.get('AUD_MAX', '30'))]:
            img = cv2.cvtColor(gris(f), cv2.COLOR_GRAY2BGR)
            cx, cy = (b[0] + b[2]) // 2, (b[1] + b[3]) // 2; R = max(60, int(max(b[2] - b[0], b[3] - b[1]) * .9))
            for q in boxes: cv2.rectangle(img, (int(q[0]), int(q[1])), (int(q[2]), int(q[3])), (0, 170, 0), 1)
            cv2.rectangle(img, (b[0] - 1, b[1] - 1), (b[2] + 1, b[3] + 1), (0, 0, 255), 1)
            p = cv2.copyMakeBorder(img, R, R, R, R, cv2.BORDER_CONSTANT, value=(255, 255, 255))
            cr = cv2.resize(p[cy:cy + 2 * R, cx:cx + 2 * R], (220, 220), interpolation=cv2.INTER_AREA)
            cv2.putText(cr, str(len(cel) + 1), (3, 16), 0, .55, (255, 0, 0), 2)
            cel.append(cr); ref.append('%d %s %s' % (len(cel), os.path.basename(f), b))
        cols = 6; rows = (len(cel) + cols - 1) // cols
        g = np.full((rows * 224, cols * 224, 3), 90, np.uint8)
        for i, c in enumerate(cel): g[(i // cols) * 224:(i // cols) * 224 + 220, (i % cols) * 224:(i % cols) * 224 + 220] = c
        cv2.imwrite(os.path.join(sal, t + '.png'), g)
        open(os.path.join(sal, t + '_ref.txt'), 'w').write('\n'.join(ref))
    print('grillas en', sal)


if __name__ == '__main__':
    main()
