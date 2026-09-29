"""Completa las etiquetas faltantes de los zips manuales (Roboflow).

Problema (18/09): cada zip se anoto para UN tipo de componente. En una captura que muestra
medio tablero, se etiqueto solo el interruptor motorizado (o la fotocelula, o el medidor) y
TODOS los demas componentes del tile quedaron sin etiqueta. Medido con el modelo K sobre 40
tiles por zip, las detecciones firmes (conf >= 0.55) que no caen sobre ninguna etiqueta son
hasta 6,6 veces mas que las etiquetas existentes. Cada una es un negativo falso: el
entrenamiento le ensena al modelo que ese componente NO es un componente, que es justo lo
contrario de lo que queremos con recall 100%.

Solucion, en dos umbrales:
  - conf >= ALTA  -> se agrega como etiqueta (el modelo tiene ~100% de recall en planos reales,
                     asi que una deteccion firme sobre un simbolo es casi seguro un componente)
  - DUDA..ALTA    -> no se etiqueta ni se deja como fondo: la zona se marca para blanquear,
                     que es el mecanismo que el generador ya usa para lo ambiguo.
Se escribe data/zips_pseudo.json para no depender del modelo en cada build.
"""
import os, sys, glob, json, cv2, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import BASE, DATA, WORK

ALTA = float(os.environ.get('PSEUDO_ALTA', '0.55'))
DUDA = float(os.environ.get('PSEUDO_DUDA', '0.12'))
PESOS = os.environ.get('PSEUDO_PESOS', 'best_componente_v6_K.pt')

def inter(a, b):
    x0 = max(a[0], b[0]); y0 = max(a[1], b[1]); x1 = min(a[2], b[2]); y1 = min(a[3], b[3])
    return 0.0 if (x1 <= x0 or y1 <= y0) else (x1 - x0) * (y1 - y0)

def main():
    from ultralytics import YOLO
    w = PESOS if os.path.exists(PESOS) else os.path.join(BASE, 'claudio_v2', PESOS)
    m = YOLO(w)
    zdir = os.path.join(WORK, 'zips')
    out = {}
    tot_or = tot_ps = tot_ig = 0
    for d in sorted(os.listdir(zdir)):
        ims = glob.glob(os.path.join(zdir, d, '**', 'images', '*.*'), recursive=True)
        for f in ims:
            g = cv2.imread(f)
            if g is None: continue
            H, W = g.shape[:2]
            lb = f.replace('images', 'labels').rsplit('.', 1)[0] + '.txt'
            orig = []
            if os.path.exists(lb):
                for ln in open(lb):
                    p = ln.split()
                    if len(p) < 5: continue
                    x, y, ww, hh = [float(v) for v in p[1:5]]
                    orig.append([(x-ww/2)*W, (y-hh/2)*H, (x+ww/2)*W, (y+hh/2)*H])
            gris = cv2.cvtColor(g, cv2.COLOR_BGR2GRAY)
            tinta = (gris < 170)
            r = m.predict(g, conf=DUDA, verbose=False, device=0)[0]
            nuevas = []
            for b, c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
                ab = (b[2]-b[0]) * (b[3]-b[1])
                if ab <= 0: continue
                if max([inter(b, q)/ab for q in orig], default=0) >= .25: continue   # ya etiquetado
                # una caja sin tinta adentro no encierra ningun simbolo: es ruido del modelo
                # sobre una zona vacia. Sin este filtro entraron 878 etiquetas casi vacias.
                x0, y0 = max(0, int(b[0])), max(0, int(b[1]))
                x1, y1 = min(W, int(b[2])), min(H, int(b[3]))
                if x1 - x0 < 3 or y1 - y0 < 3: continue
                if tinta[y0:y1, x0:x1].mean() < .02: continue
                nuevas.append([float(v) for v in b] + [float(c)])
            key = '/'.join(os.path.relpath(f, zdir).split(os.sep))
            out[key] = {'w': W, 'h': H, 'orig': orig, 'cand': nuevas}
            tot_or += len(orig); tot_ps += len(nuevas)
    p = os.path.join(DATA, 'zips_pseudo.json')
    json.dump(out, open(p, 'w'))
    print('[pseudo] %d tiles | etiquetas originales %d | candidatas %d (con su confianza)'
          % (len(out), tot_or, tot_ps))
    print('[pseudo] escrito', p)

if __name__ == '__main__':
    main()
