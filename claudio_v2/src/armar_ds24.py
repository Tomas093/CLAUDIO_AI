"""ds24 (30/09): ds23 + SUCIEDAD encima de los componentes (circulos grandes, rotulos cortos, lineas; RF7 perdia los
interruptores de EZE cruzados por circulos). Base: ds22 + CLASES AUXILIARES de negativos duros (idea del informe de modelos, practica estandar).

Clase 0 = componente (igual que siempre). Clase 1 = "no-componente que parece componente": PAT (suelta y en
cuadrado), flecha de alimentacion y rotulo vertical, dibujados con src/neg_objetos.py en lugares libres de los
tiles SINTETICOS (p*, n*) sin pisar cajas. La cabeza aprende a separarlos en vez de tratarlos como fondo; a la
salida se usa solo la clase 0. Los tiles manuales/reales (m*, r*) se copian igual (sin clase 1).
    py -3 src/armar_ds24.py
"""
import os, sys, glob, random, shutil, zlib
sys.path.insert(0, os.path.dirname(__file__))
import cv2, numpy as np
from paths import WORK
import neg_objetos

SRC, DST = os.path.join(WORK, 'ds22'), os.path.join(WORK, 'ds24')


def main():
    n1 = 0
    for sp in ('train', 'val'):
        os.makedirs(os.path.join(DST, 'images', sp), exist_ok=True); os.makedirs(os.path.join(DST, 'labels', sp), exist_ok=True)
        for i, f in enumerate(sorted(glob.glob(os.path.join(SRC, 'images', sp, '*.png')))):
            nom = os.path.basename(f); lf = os.path.join(SRC, 'labels', sp, nom[:-4] + '.txt')
            di, dl = os.path.join(DST, 'images', sp, nom), os.path.join(DST, 'labels', sp, nom[:-4] + '.txt')
            lab = open(lf).read().strip().splitlines() if os.path.exists(lf) else []
            if nom[0] in 'mr':
                shutil.copyfile(f, di); open(dl, 'w').write('\n'.join(lab) + ('\n' if lab else '')); continue
            img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
            if img.ndim == 3: img = img[:, :, 0]
            img = img.copy(); H, W = img.shape
            cajas = []
            for l in lab:
                _, cx, cy, w, h = map(float, l.split()[:5])
                cajas.append([(cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H])
            r = random.Random(23_000_000 + zlib.crc32((sp + nom).encode()))   # reproducible
            th = r.uniform(9, 25)
            if r.random() < .5: neg_objetos.ensuciar(img, r, th, cajas)       # ds24: suciedad sobre positivos
            objs = neg_objetos.agregar(img, r, th, r.choice([0, 1, 2, 2, 3, 4]), evitar=cajas)
            for b, _ in objs:
                x0, y0, x1, y1 = [max(0, min(W if k % 2 == 0 else H, v)) for k, v in enumerate(b)]
                if x1 - x0 < 3 or y1 - y0 < 3: continue
                lab.append('1 %.6f %.6f %.6f %.6f' % ((x0 + x1) / 2 / W, (y0 + y1) / 2 / H, (x1 - x0) / W, (y1 - y0) / H)); n1 += 1
            cv2.imwrite(di, img); open(dl, 'w').write('\n'.join(lab) + ('\n' if lab else ''))
            if i % 2000 == 0: print('[ds24] %s %d | clase1 %d' % (sp, i, n1), flush=True)
    open(os.path.join(DST, 'data.yaml'), 'w').write('path: %s\ntrain: images/train\nval: images/val\nnames:\n  0: componente\n  1: no_componente\n' % DST.replace('\\', '/'))
    print('[ds24] listo, cajas clase 1:', n1)


if __name__ == '__main__':
    main()
