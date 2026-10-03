"""Negativos "de objeto" CON CAJA para el verificador v4 (30/09, goal: 0 FN con < 350 FP).

Con el GT v8 la mayoria de los FP aislados de RF4 con conf 0,95-0,99 son cosas que Tomas dijo que NO son
componentes: la PAT (sola o dentro de un cuadrado con el punto de union en la esquina), la flecha de
alimentacion con el nombre del tablero en vertical y los rotulos verticales. La receta ds21 tenia P_PAT=0 y la
PAT en cuadrado no se generaba nunca. Aca se dibujan sobre un tile ya hecho, en un lugar en blanco, y se
devuelve la caja de cada uno para usarla como negativo explicito (mas su jitter).
Dibujo propio (no se copia geometria de los planos de test).
"""
import cv2, numpy as np
from compose import put_text, ROTULOS, S


def _blanco(img, b, m=4):
    x0, y0, x1, y1 = [int(v) for v in b]
    x0, y0 = max(0, x0 - m), max(0, y0 - m); x1, y1 = min(S, x1 + m), min(S, y1 + m)
    if x1 - x0 < 4 or y1 - y0 < 4: return False
    return (img[y0:y1, x0:x1] < 200).mean() < .002


def _tierra(img, cx, ytop, w, h, t):
    """Bajada + tres rayas decrecientes. Devuelve la caja."""
    ymed = ytop + h // 2
    cv2.line(img, (cx, ytop), (cx, ymed), 0, t)
    for k, f in enumerate([1.0, .66, .33]):
        L = int(w * .45 * f); yy = ymed + k * max(2, h // 7)
        cv2.line(img, (cx - L, yy), (cx + L, yy), 0, t)
    return [cx - int(w * .45), ytop, cx + int(w * .45), ymed + 2 * max(2, h // 7)]


def pat_suelta(img, r, txt_h):
    h = int(txt_h * r.uniform(1.6, 3.2)); w = int(h * r.uniform(.6, 1.0))
    x = r.randint(8, S - w - 8); y = r.randint(8, S - h - 8)
    b = [x, y, x + w, y + h]
    if not _blanco(img, b): return None
    t = r.choice([1, 1, 2]); cx = x + w // 2
    if r.random() < .55: cv2.circle(img, (cx, y + max(2, h // 6)), max(2, int(h * .09)), 0, -1)
    bb = _tierra(img, cx, y, w, h, t)
    if r.random() < .4: put_text(img, cx + int(w * .6), y + int(txt_h * .9), r.choice(['PAT', 'PE', 'TT']), int(txt_h * r.uniform(.8, 1.1)), r)
    return [min(bb[0], x), y, max(bb[2], x + w), bb[3]]


def pat_cuadrado(img, r, txt_h):
    """Cuadrado (a veces rectangulo) con el punto de union gordo en una esquina de arriba, el cable que baja
    por dentro y la tierra en un recuadro chico abajo. Todo el conjunto es UN negativo."""
    s = int(txt_h * r.uniform(2.6, 4.5)); w = int(s * r.uniform(.8, 1.15))
    x = r.randint(10, S - w - 10); y = r.randint(10, S - s - 10)
    b = [x, y, x + w, y + s]
    if not _blanco(img, b, 6): return None
    t = 1 if r.random() < .8 else 2
    if r.random() < .75: cv2.rectangle(img, (x, y), (x + w, y + s), 0, t)
    der = r.random() < .7
    px = x + w if der else x
    cv2.circle(img, (px, y), max(2, int(s * r.uniform(.06, .1))), 0, -1)
    # cable: desde el punto hacia el otro lado y baja
    ix = x + int(w * r.uniform(.15, .3)) if der else x + int(w * r.uniform(.7, .85))
    iy = y + int(s * r.uniform(.12, .25))
    cv2.line(img, (px, iy), (ix, iy), 0, 1); cv2.line(img, (ix, iy), (ix, y + int(s * .55)), 0, 1)
    # recuadro chico con la tierra
    sw = int(w * r.uniform(.35, .5)); sh = int(s * r.uniform(.3, .42))
    sx = ix - sw // 2; sy = y + s - sh - max(1, int(s * .04))
    if r.random() < .8: cv2.rectangle(img, (sx, sy), (sx + sw, sy + sh), 0, 1)
    _tierra(img, ix, sy + max(1, sh // 8), int(sw * .8), int(sh * .8), 1)
    return b


def flecha(img, r, txt_h):
    L = int(txt_h * r.uniform(2, 6)); a = int(txt_h * r.uniform(.5, .8))
    x = r.randint(12, S - 40); y = r.randint(10, max(11, S - int(txt_h * 10)))
    alto = L + int(a * 1.8)
    con_txt = r.random() < .8
    b = [x - a, y, x + a + (int(txt_h * 1.3) if con_txt else 0), y + max(alto, int(txt_h * 5) if con_txt else 0)]
    if not _blanco(img, b, 5): return None
    cv2.line(img, (x, y), (x, y + L), 0, 1)
    pts = np.array([[x - a, y + L], [x + a, y + L], [x, y + L + int(a * 1.8)]], np.int32)
    cv2.polylines(img, [pts], True, 0, 1)
    if con_txt: put_text(img, x + a + 2, y + L - int(txt_h * .5), r.choice(ROTULOS)[:10], int(txt_h * .9), r, rot=True)
    return [x - a, y + L - int(txt_h * .3), x + a, y + L + int(a * 1.8)] if r.random() < .5 else b


def rotulo(img, r, txt_h):
    w = int(txt_h * r.uniform(1.7, 2.8)); h = int(txt_h * r.uniform(6.5, 13.0))
    x = r.randint(6, S - w - 6); y = r.randint(int(txt_h * 3), max(int(txt_h * 3) + 1, S - h - 6))
    b = [x, y, x + w, y + h]
    if h >= S - 20 or not _blanco(img, b, 4): return None
    cv2.rectangle(img, (x, y), (x + w, y + h), 0, 1 if r.random() < .8 else 2)
    nom = r.choice(ROTULOS); hh = int(txt_h * r.uniform(.75, .95))
    while len(nom) > 2 and len(nom) * hh * .62 > h - 6: nom = nom[:-1]
    put_text(img, x + 2, y + max(2, int((h - len(nom) * hh * .62) / 2)), nom, hh, r, rot=True)
    return b


GENS = [(pat_cuadrado, 3), (pat_suelta, 2), (flecha, 2), (rotulo, 1)]


def _pisa(b, evitar, m):
    return any(b[0] - m < q[2] and b[2] + m > q[0] and b[1] - m < q[3] and b[3] + m > q[1] for q in evitar)


def agregar(img, r, txt_h, n, evitar=(), gens=None):
    """Dibuja hasta n negativos de objeto en lugares en blanco que NO pisen ninguna caja de `evitar` (las cajas
    de los componentes del tile; 30/09, Tomas: un rotulo quedaba encima de la caja del medidor Wh).
    Se dibuja en una copia y solo se pega si la caja final esta libre. Devuelve [(caja, tipo)]."""
    G = gens or GENS
    tot = sum(p for _, p in G); out = []; ocup = [list(q[:4]) for q in evitar]
    m = int(txt_h * .6) + 4
    for _ in range(n * 6):
        if len(out) >= n: break
        k = r.random() * tot
        for g, p in G:
            k -= p
            if k <= 0: break
        tmp = img.copy(); b = g(tmp, r, txt_h)
        if b is None: continue
        dib = np.where((tmp != img).any(-1) if tmp.ndim == 3 else tmp != img)
        if len(dib[0]):
            tb = [int(dib[1].min()), int(dib[0].min()), int(dib[1].max()) + 1, int(dib[0].max()) + 1]
            if _pisa(tb, ocup, m): continue
        img[:] = tmp; out.append((b, g.__name__)); ocup.append(list(b))
    return out


def ensuciar(img, r, txt_h, boxes):
    """v5 (30/09): suciedad ENCIMA de los componentes, que siguen siendo positivos. Con el v4 como veto, el
    verificador rechazaba (p < 0,001) interruptores cruzados por circulos grandes o con un rotulo corto pegado
    ("Q6", "K1"): habia aprendido que circulo grande = negativo. Generico: circulos grandes que cruzan el tile,
    rotulos cortos junto a algunos componentes y alguna linea de cota."""
    for _ in range(r.randint(1, 4)):                     # circulos grandes (burbujas, circulos de construccion)
        R = int(txt_h * r.uniform(2.5, 10)); cx = r.randint(0, S); cy = r.randint(0, S)
        cv2.circle(img, (cx, cy), R, int(r.uniform(0, 140)), 1)
    for b in r.sample(list(boxes), min(len(boxes), r.randint(0, 6))):   # rotulo corto pegado al componente
        t = r.choice(['Q%d' % r.randint(1, 30), 'K%d' % r.randint(1, 9), 'F%d' % r.randint(1, 20), 'X%d' % r.randint(1, 9),
                      'KM%d' % r.randint(1, 5), 'ID%02d' % r.randint(1, 12)])
        x = int(b[0] - txt_h * r.uniform(.2, 1.5)) if r.random() < .5 else int(b[2] + 2)
        y = int(r.uniform(b[1], max(b[1] + 1, b[3] - txt_h)))
        put_text(img, max(0, x), max(0, y), t, int(txt_h * r.uniform(.6, .9)), r, rot=r.random() < .4)
    if r.random() < .3:                                  # linea de cota / cable que cruza
        y = r.randint(0, S - 1); cv2.line(img, (0, y), (S, y + r.randint(-40, 40)), 0, 1)
