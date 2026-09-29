"""Datos para el VERIFICADOR de segunda etapa (29/09, goal de Tomas: 100% de recall con < 700 FP).

Idea: el detector (RF-DETR) se usa con umbral bajo para no perder nada, y un clasificador chico mira cada caja
propuesta con su contexto y decide si es un componente bien encuadrado (positivo) o un pedazo, duplicado, texto,
circulo numerado, etc. (negativo). De los 6.592 FP de RF4 a 0,05, 2.424 son pedazos/duplicados sobre componentes
ya detectados: un clasificador que ve la caja Y su entorno los puede separar.

SIN SOBREAJUSTE: los 7 planos de test no se usan aca. Las muestras salen de
  - tiles sinteticos NUEVOS (receta ds21, semillas desde 5.000.000: el detector nunca los vio; GT exacto)
  - tiles de validacion de ds21 manuales (m) y de planos reales (rp) (val: no entrenan al detector de ds21)
Etiqueta de cada caja del detector: 1 si IoU >= 0,5 con alguna caja del GT del tile, 0 si no. Ademas se agregan
las cajas del GT (con jitter) como positivos, para que vea tambien lo que el detector casi no ve.

Salida: work/verif/<nombre>.npz con X (N,64,64) uint8 recorte con contexto, M (N,4) caja relativa al recorte,
y (N,) etiqueta, conf (N,) confianza del detector (-1 para las del GT), src (N,) fuente.

    venv_rfdetr/Scripts/python.exe -u src/verif_datos.py best_componente_v24_RF4.pth 6000
"""
import os, sys, json, random, glob
import cv2, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from paths import WORK

R = 64          # lado del recorte
CTX = 2.2       # el recorte abarca CTX veces el lado mayor de la caja


def recorte(img, b):
    """Recorte cuadrado con contexto alrededor de la caja b, a RxR, y la caja en coordenadas del recorte (0-1)."""
    x0, y0, x1, y1 = b[:4]; cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    L = max(24.0, max(x1 - x0, y1 - y0) * CTX)
    X0, Y0 = int(round(cx - L / 2)), int(round(cy - L / 2)); Li = int(round(L))
    p = Li
    im = cv2.copyMakeBorder(img, p, p, p, p, cv2.BORDER_CONSTANT, value=255)
    c = im[Y0 + p:Y0 + p + Li, X0 + p:X0 + p + Li]
    c = cv2.resize(c, (R, R), interpolation=cv2.INTER_AREA if Li > R else cv2.INTER_LINEAR)
    m = [(x0 - X0) / Li, (y0 - Y0) / Li, (x1 - X0) / Li, (y1 - Y0) / Li]
    return c, m


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1])); I = ix * iy
    return I / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - I + 1e-9)


def main():
    pesos, n_sint = sys.argv[1], int(sys.argv[2])
    nombre = os.environ.get('VERIF_NOMBRE', 'rf4_ds21')
    from receta_ds21 import RECETA
    os.environ.update(RECETA)
    import build_all as B
    from compose import make_tile
    from evaluate import RFDETRComoYOLO
    modelo = RFDETRComoYOLO(pesos)
    X, M, Y, C, SRC = [], [], [], [], []
    rng = random.Random(7)

    def procesar(img, gt, fuente):
        det = modelo.predict([cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)], conf=.05)[0]
        for b, c in zip(det.boxes.xyxy.tolist(), det.boxes.conf.tolist()):
            mi = max([iou(b, g) for g in gt], default=0.0)
            if .3 <= mi < .5: continue          # v2: ambiguas (ni claramente bien ni claramente mal) fuera
            y = int(mi >= .5)
            cr, m = recorte(img, b); X.append(cr); M.append(m); Y.append(y); C.append(c); SRC.append(fuente)
        def poner(b, y, tag):
            cr, m = recorte(img, b); X.append(cr); M.append(m); Y.append(y); C.append(-1.0); SRC.append(fuente + tag)
        for g in gt:                     # positivos del GT, con jitter chico de caja
            w, h = g[2] - g[0], g[3] - g[1]
            # v2 (29/09): jitter +-15% por lado. Con +-8% el verificador rechazaba el TM-DIN de LU/nyw (0,98 de RF4,
            # 0,001 de verificador): en esos planos la caja cubre el cuerpo sin el contacto de abajo.
            for _ in range(2):
                poner([g[0] + rng.uniform(-.15, .15) * w, g[1] + rng.uniform(-.15, .15) * h,
                       g[2] + rng.uniform(-.15, .15) * w, g[3] + rng.uniform(-.15, .15) * h], 1, '_gt')
            # NEGATIVOS sinteticos del tipo de FP que se ve en los planos: pedazo del componente y caja corrida
            for _ in range(8):   # v2: solo pedazos claros (<= 30% del area, como la cruz sola)
                fw, fh = rng.uniform(.15, .7), rng.uniform(.15, .7)
                if fw * fh > .3: continue
                px, py = rng.uniform(0, 1 - fw), rng.uniform(0, 1 - fh)
                p = [g[0] + px * w, g[1] + py * h, g[0] + (px + fw) * w, g[1] + (py + fh) * h]
                if max(iou(p, q) for q in gt) < .3: poner(p, 0, '_pedazo'); break
            if rng.random() < .5:
                for _ in range(8):
                    dx, dy = rng.choice([-1, 1]) * rng.uniform(.6, 1.0) * w, rng.uniform(-.3, .3) * h
                    if rng.random() < .5: dx, dy = rng.uniform(-.3, .3) * w, rng.choice([-1, 1]) * rng.uniform(.6, 1.0) * h
                    p = [g[0] + dx, g[1] + dy, g[2] + dx, g[3] + dy]
                    if max(iou(p, q) for q in gt) < .3: poner(p, 0, '_corrida'); break
        # union de dos componentes vecinos (una caja para dos)
        for _ in range(min(3, len(gt) // 2)):
            a, b = rng.sample(gt, 2)
            if abs((a[0] + a[2]) - (b[0] + b[2])) / 2 < 3 * max(a[2] - a[0], b[2] - b[0]) and \
               abs((a[1] + a[3]) - (b[1] + b[3])) / 2 < 3 * max(a[3] - a[1], b[3] - b[1]):
                u = [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]
                if max(iou(u, q) for q in gt) < .45: poner(u, 0, '_union')
        # cajas sobre tinta que no es componente (texto, tablas, circulos, marcas): centro en un pixel de tinta
        ys, xs = np.where(img < 128)
        if len(xs):
            lados = [max(g[2] - g[0], g[3] - g[1]) for g in gt] or [30.0]
            for _ in range(6):
                k = rng.randrange(len(xs)); L = rng.choice(lados) * rng.uniform(.6, 1.4); asp = rng.uniform(.4, 2.5)
                w2, h2 = L * min(1, asp) , L * min(1, 1 / asp)
                p = [xs[k] - w2 / 2, ys[k] - h2 / 2, xs[k] + w2 / 2, ys[k] + h2 / 2]
                if all(iou(p, q) < .1 for q in gt): poner(p, 0, '_fondo')

    lote = []
    # v4 (30/09): VERIF_NEGOBJ=n dibuja hasta n negativos "de objeto" por tile (PAT suelta / en cuadrado, flecha
    # de alimentacion, rotulo vertical; src/neg_objetos.py) y los agrega como negativos explicitos (+ jitter).
    # Semillas desde VERIF_SEMILLA (v4: 6.000.000) para que sean tiles que nadie vio.
    n_obj = int(os.environ.get('VERIF_NEGOBJ', '0')); semilla = int(os.environ.get('VERIF_SEMILLA', '5000000'))
    import neg_objetos
    for i in range(n_sint):
        neg = (i % 5 == 4)
        r_t = random.Random(semilla + i)
        img, boxes = make_tile(B.LIB, r_t, negative=neg)
        if img.ndim == 3: img = img[:, :, 0].copy()
        gt = [list(map(float, b[:4])) for b in boxes if b[2] - b[0] >= 3 and b[3] - b[1] >= 3]
        objs = neg_objetos.agregar(img, r_t, r_t.uniform(9, 25), n_obj) if n_obj else []
        procesar(img, gt, 'sint')
        for b, tipo in objs:
            w, h = b[2] - b[0], b[3] - b[1]
            for k in range(3):
                j = 0 if k == 0 else .12
                q = [b[0] + rng.uniform(-j, j) * w, b[1] + rng.uniform(-j, j) * h, b[2] + rng.uniform(-j, j) * w, b[3] + rng.uniform(-j, j) * h]
                cr, m = recorte(img, q); X.append(cr); M.append(m); Y.append(0); C.append(-1.0); SRC.append('negobj_' + tipo)
        if i % 500 == 0: print('[verif] sinteticos %d/%d, muestras %d' % (i, n_sint, len(Y)), flush=True)
    # tiles de validacion de ds21 (manuales y planos reales)
    # v3 (29/09): VERIF_SPLITS=train,val agrega TODOS los tiles manuales y de planos reales de ds21 (estilo real)
    splits = [x for x in os.environ.get('VERIF_SPLITS', 'val').split(',') if x]
    for f in sorted(sum([glob.glob(os.path.join(WORK, 'ds21', 'images', sp, '*.png')) for sp in splits], [])):
        nom = os.path.basename(f)
        if not (nom.startswith('m') or nom.startswith('rp')): continue
        img = cv2.imread(f, 0)
        if img.ndim == 3: img = img[:, :, 0] if img.shape[2] == 1 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)   # ultralytics parchea imread
        lf = f.replace(os.sep + 'images' + os.sep, os.sep + 'labels' + os.sep)[:-4] + '.txt'
        gt = []
        if os.path.exists(lf):
            for ln in open(lf):
                _, cx, cy, w, h = map(float, ln.split())
                gt.append([(cx - w / 2) * 640, (cy - h / 2) * 640, (cx + w / 2) * 640, (cy + h / 2) * 640])
        procesar(img, gt, 'm' if nom.startswith('m') else 'real')
    os.makedirs(os.path.join(WORK, 'verif'), exist_ok=True)
    out = os.path.join(WORK, 'verif', nombre + '.npz')
    np.savez_compressed(out, X=np.array(X, np.uint8), M=np.array(M, np.float32), y=np.array(Y, np.uint8),
                        conf=np.array(C, np.float32), src=np.array(SRC))
    y = np.array(Y); c = np.array(C)
    print('[verif] listo %s: %d muestras | det positivas %d negativas %d | gt %d'
          % (out, len(y), ((c >= 0) & (y == 1)).sum(), ((c >= 0) & (y == 0)).sum(), (c < 0).sum()))


if __name__ == '__main__':
    main()
