"""Grillas numeradas de los generadores nuevos del plan ds21 (28/09), para que Tomas las revise ANTES de
armar el dataset. Verde = caja positiva. Uso: py -3 src/grilla_ds21.py <carpeta_salida>"""
import os, sys, random, cv2, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from receta_ds21 import RECETA
os.environ.update({k: os.environ.get(k, v) for k, v in RECETA.items()})
sys.path.insert(0, os.path.dirname(__file__))
import build_all as B
from compose import make_tile

def tiles(forzar, n, semilla, neg=False):
    viejo = {k: os.environ[k] for k in forzar}
    os.environ.update(forzar)
    out = []
    for i in range(n):
        img, boxes = make_tile(B.LIB, random.Random(semilla + i), negative=neg)
        out.append((img, boxes))
    os.environ.update(viejo)
    return out

def dibujar(img, boxes, k, esc=.75):
    v = cv2.resize(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), None, fx=esc, fy=esc, interpolation=cv2.INTER_AREA)
    for b in boxes:   # 2 px por fuera: si no, la caja se esconde bajo el borde negro de los rectangulos
        cv2.rectangle(v, (int(b[0]*esc) - 2, int(b[1]*esc) - 2), (int(b[2]*esc) + 2, int(b[3]*esc) + 2), (0, 170, 0), 1)
    cv2.rectangle(v, (0, 0), (46, 22), (255, 255, 255), -1); cv2.putText(v, str(k), (3, 17), 0, .6, (0, 0, 255), 2)
    return v

def grilla(celdas, cols, f):
    H = max(c.shape[0] for c in celdas); W = max(c.shape[1] for c in celdas)
    g = np.full(((len(celdas) + cols - 1) // cols * (H + 4), cols * (W + 4), 3), 90, np.uint8)
    for i, c in enumerate(celdas):
        y, x = (i // cols) * (H + 4), (i % cols) * (W + 4); g[y:y + c.shape[0], x:x + c.shape[1]] = c
    cv2.imwrite(f, g)

def recortes(lista, filtro, k0, R=48, zoom=3):
    """Recortes ampliados alrededor de las cajas que cumplen `filtro` (para ver lo chico)."""
    cel = []
    for img, boxes in lista:
        for b in boxes:
            if not filtro(b): continue
            cx, cy = (b[0] + b[2]) // 2, (b[1] + b[3]) // 2
            p = cv2.copyMakeBorder(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), R, R, R, R, cv2.BORDER_CONSTANT, value=(255, 255, 255))
            cr = p[cy:cy + 2 * R, cx:cx + 2 * R].copy()
            cv2.rectangle(cr, (int(b[0] - cx + R), int(b[1] - cy + R)), (int(b[2] - cx + R), int(b[3] - cy + R)), (0, 170, 0), 1)
            cr = cv2.resize(cr, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_NEAREST)
            cv2.putText(cr, str(k0 + len(cel)), (3, 17), 0, .6, (0, 0, 255), 2)
            cel.append(cr)
            if len(cel) >= 24: return cel
    return cel

if __name__ == '__main__':
    out = sys.argv[1]; os.makedirs(out, exist_ok=True)
    apagar = dict(P_LETRA='0', P_ROTULO='0', P_ROTULO_NEG='0', P_NEGEXTRA='0')
    # 1) letra en recuadro: recortes ampliados de las cajas chicas cuadradas
    import compose
    marcadas = []; _orig = compose.letra_recuadro
    def _espia(c, r, txt_h, xs, boxes):
        n0 = len(boxes); k = _orig(c, r, txt_h, xs, boxes); marcadas.extend(tuple(b[:4]) for b in boxes[n0:]); return k
    compose.letra_recuadro = _espia
    L = tiles(dict(apagar, P_LETRA='1'), 12, 1000)
    compose.letra_recuadro = _orig
    grilla(recortes(L, lambda b: tuple(b[:4]) in marcadas, 1), 6, os.path.join(out, '1_letra_recuadro.png'))
    # 2) rotulos verticales: tiles enteros
    T = tiles(dict(apagar, P_ROTULO_NEG='1'), 8, 2000)   # 28/09: ahora NEGATIVO (Tomas)
    grilla([dibujar(i, b, k + 1, .6) for k, (i, b) in enumerate(T)], 4, os.path.join(out, '2_rotulo_vertical_NEGATIVO.png'))
    # 3) negativos extra: tiles NEGATIVOS (sin simbolos) para que se vea solo el fondo nuevo
    N = tiles(dict(apagar, P_NEGEXTRA='1'), 8, 3000, neg=True)
    grilla([dibujar(i, b, k + 1, .6) for k, (i, b) in enumerate(N)], 4, os.path.join(out, '3_negativos_extra.png'))
    # 5) filas y parejas (28/09: Tomas marco que en las ternas los simbolos quedaban encimados)
    Fi = tiles(dict(apagar, P_TERNA='1', P_DENSA='1', P_TRAFO='1'), 8, 5000)
    grilla([dibujar(i, b, k + 1, .6) for k, (i, b) in enumerate(Fi)], 4, os.path.join(out, '5_filas_ternas_parejas.png'))
    # 4) receta completa ds21
    F = tiles({}, 8, 4000)
    grilla([dibujar(i, b, k + 1, .6) for k, (i, b) in enumerate(F)], 4, os.path.join(out, '4_receta_completa.png'))
    print('grillas en', out)
