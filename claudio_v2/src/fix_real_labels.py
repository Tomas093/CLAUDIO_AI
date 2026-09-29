"""Cine las cajas de real_labels.json al simbolo, ignorando el cable que las atraviesa.

El 24% de esas etiquetas tienen mas de 18% de aire (el plano `vyre` llega al 37%): traen un
margen fijo de 4-5 px alrededor del simbolo. No alcanza con recortar a la tinta porque el
cable vertical cruza la caja de arriba a abajo y su tinta llega a los bordes. Se proyecta la
tinta sobre cada eje y se toma el tramo donde la cuenta supera un umbral: el cable aporta 1-2
pixeles por fila y queda debajo, el simbolo aporta muchos mas y queda dentro.
"""
import os, sys, json, cv2, numpy as np, shutil
P = r'C:\Users\Tomas\Documents\LAB3\CLAUDIO_AI\claudio_v2'
os.chdir(P); sys.path.insert(0,'.'); sys.path.insert(0,'src')
B = r'C:\Users\Tomas\Documents\LAB3\CLAUDIO_AI'
SOLO_MEDIR = os.environ.get('APLICAR','0') != '1'

def cenir_eje(cnt, minimo=3):
    """tramo del eje con al menos `minimo` pixeles de tinta.

    Un cable deja 1-2 pixeles por fila, un trazo del simbolo deja mas. Se usa un minimo
    ABSOLUTO y no una fraccion del maximo: con la fraccion se recortaban partes legitimas
    de los simbolos que tienen una zona muy densa y otra de trazo fino (el diferencial
    quedaba partido al medio).
    """
    if cnt.max() <= 0: return None
    idx = np.where(cnt >= min(minimo, max(1, cnt.max())))[0]
    if not len(idx): return None
    return int(idx.min()), int(idx.max()) + 1

lab = json.load(open('data/real_labels.json'))
nuevo = {}; ant = []; des = []
for n, bx in lab.items():
    im = cv2.imread(os.path.join('data','renders', n + '_d.png'), 0)
    if im is None: nuevo[n] = bx; continue
    H, W = im.shape; out = []
    for b in bx:
        x0, y0, x1, y1 = [int(round(v)) for v in b[:4]]
        x0, y0 = max(0,x0), max(0,y0); x1, y1 = min(W,x1), min(H,y1)
        w, h = x1-x0, y1-y0
        if w < 6 or h < 6: out.append(list(b)); continue
        t = (im[y0:y1, x0:x1] < 160)
        if not t.any(): out.append(list(b)); continue
        ys, xs = np.where(t)
        hol = max((xs.min()+(w-1-xs.max()))/float(w), (ys.min()+(h-1-ys.max()))/float(h))
        ant.append(hol)
        # Solo se tocan las que tienen aire de sobra. Las que ya ajustan se dejan como estan:
        # cenir una caja correcta solo puede empeorarla.
        if hol < .15: des.append(hol); out.append(list(b)); continue
        rx = cenir_eje(t.sum(axis=0)); ry = cenir_eje(t.sum(axis=1))
        if rx is None or ry is None: out.append(list(b)); continue
        nb = [x0+rx[0], y0+ry[0], x0+rx[1], y0+ry[1]]
        if nb[2]-nb[0] < 4 or nb[3]-nb[1] < 4: out.append(list(b)); continue
        # y nunca se recorta mas de la mitad del area: eso seria estar cortando el simbolo
        if (nb[2]-nb[0])*(nb[3]-nb[1]) < .30*w*h: out.append(list(b)); des.append(hol); continue
        nw, nh = nb[2]-nb[0], nb[3]-nb[1]
        t2 = (im[nb[1]:nb[3], nb[0]:nb[2]] < 160)
        ys2, xs2 = np.where(t2)
        des.append(max((xs2.min()+(nw-1-xs2.max()))/float(nw), (ys2.min()+(nh-1-ys2.max()))/float(nh)))
        out.append(nb)
    nuevo[n] = out
a, d = np.array(ant), np.array(des)
print('%d cajas' % len(a))
print('  holgura ANTES : mediana %.3f  p90 %.3f  >18%%: %.1f%%' % (np.median(a), np.percentile(a,90), 100*(a>.18).mean()))
print('  holgura DESPUES: mediana %.3f  p90 %.3f  >18%%: %.1f%%' % (np.median(d), np.percentile(d,90), 100*(d>.18).mean()))
if not SOLO_MEDIR:
    shutil.copy2('data/real_labels.json', os.path.join(B,'_para_borrar','real_labels_18092026.json'))
    json.dump(nuevo, open('data/real_labels.json','w'))
    print('  aplicado (el original quedo en _para_borrar)')
else:
    print('  (solo medicion, no se escribio nada)')
