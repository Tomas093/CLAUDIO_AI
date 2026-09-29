"""Simbolos procedurales (sin fuente externa): caja en T, borneras DIN/circulares, PAT, fusibles, lamparas, etc."""
import os
import numpy as np, cv2, random
def _c(h, w): return np.full((h, w), 255, np.uint8)
def caja_t(r):
    s = r.randint(70, 110); im = _c(s+40, s); t = r.randint(1, 2)
    im = _c(s, s); cv2.rectangle(im, (3, 3), (s-4, s-4), 0, t)
    cv2.line(im, (s//4, s//4), (3*s//4, s//4), 0, t+1); cv2.line(im, (s//2, s//4), (s//2, 3*s//4), 0, t+1)
    return im
def bornera_din(r):
    w, h = r.randint(30, 60), r.randint(60, 110); im = _c(h, w); t = r.randint(1, 2)
    cv2.rectangle(im, (2, h//5), (w-3, 4*h//5), 0, t); cv2.line(im, (w//2, 0), (w//2, h//5), 0, t); cv2.line(im, (w//2, 4*h//5), (w//2, h-1), 0, t)
    if r.random() < .5: cv2.line(im, (2, h//2), (w-3, h//2), 0, t)
    if r.random() < .5: cv2.circle(im, (w//2, h//2), max(3, w//6), 0, t)
    return im
def bornera_circ(r):
    s = r.randint(40, 80); im = _c(s, s); t = r.randint(1, 2)
    cv2.circle(im, (s//2, s//2), s//3, 0, t)
    if r.random() < .5: cv2.line(im, (2, s-2), (s-2, 2), 0, t)
    else: cv2.line(im, (s//4, 3*s//4), (3*s//4, s//4), 0, t); cv2.line(im, (s//2, 0), (s//2, s//2), 0, t)
    if r.random() < .6:
        im2 = _c(s, int(s*1.6)); im2[:, int(s*.6):] = im
        cv2.putText(im2, r.choice(['X1','X2','X3','X4']), (1, s//2), cv2.FONT_HERSHEY_SIMPLEX, s/90, 0, 1, cv2.LINE_AA); im = im2
    return im
def bornera_strip(r):
    n = r.randint(3, 8); cw = r.randint(18, 30); h = r.randint(40, 70); im = _c(h, n*cw+4); t = r.randint(1, 2)
    for i in range(n):
        cv2.rectangle(im, (2+i*cw, 2), (2+(i+1)*cw, h-3), 0, t)
        if r.random() < .6: cv2.circle(im, (2+i*cw+cw//2, h//2), cw//4, 0, 1)
    return im
def pat(r):
    w = r.randint(50, 80); h = r.randint(60, 100); im = _c(h, w); t = r.randint(1, 2)
    y = h//2; cv2.line(im, (w//2, 0), (w//2, y), 0, t)
    if r.random() < .5: cv2.circle(im, (w//2, y//2), r.randint(4, 8), 0, -1)
    for k, f in enumerate([1, .66, .33]):
        L = int(w*.45*f); yy = y + k*h//7; cv2.line(im, (w//2-L, yy), (w//2+L, yy), 0, t)
    return im
def fusible(r):
    w, h = r.randint(20, 35), r.randint(60, 100); im = _c(h, w); t = r.randint(1, 2)
    cv2.rectangle(im, (2, h//5), (w-3, 4*h//5), 0, t); cv2.line(im, (w//2, 0), (w//2, h-1), 0, t)
    return im
def lampara(r):
    s = r.randint(40, 80); im = _c(s, s); t = r.randint(1, 2); c = s//2; R = s//2-3
    cv2.circle(im, (c, c), R, 0, t); d = int(R*.7)
    cv2.line(im, (c-d, c-d), (c+d, c+d), 0, t); cv2.line(im, (c-d, c+d), (c+d, c-d), 0, t)
    return im
# 20/09: el bloque A$C45664E16 de LU-UN-01 y nyw-un-01 es el falso negativo mas grande que
# queda (15 de los 19 de N). Se lo habia tratado como "simbolo chico subrepresentado" y el
# plan O fallo intentando arreglarlo por ahi. Al mirarlo por fin renderizado resulto ser algo
# mucho mas simple: un cuadradito con la letra 'A' en CURSIVA, pegado a un cable horizontal.
# Mide 17.9 px, que no es chico; el dataset ya tenia 11-13% de cajas de ese tamano.
# Lo que si estaba mal representado es el caso en si: una caja con UNA sola letra salia en el
# 0.24% de los simbolos, siempre con fuente recta y siempre suelta, sin el cable pegado.
_TXT_LETRAS = ['IM', 'CT', 'GE', 'TTA', 'PLC', 'M', 'V', 'A', 'kWh', 'UPS', 'SPM', 'PF',
               'DPS', 'VFD', 'T', 'TI', 'TV']
_TXT_UNA = ['A', 'V', 'M', 'T', 'W', 'S', 'F', 'kWh', 'Hz', 'cos']


def caja_letras(r):
    # la mitad de las veces una sola letra, que es el caso real que se perdia
    txt = r.choice(_TXT_UNA) if r.random() < .5 else r.choice(_TXT_LETRAS)
    w = r.randint(60, 110); h = r.randint(50, 90); im = _c(h, w); t = r.randint(1, 2)
    if r.random() < .5: cv2.rectangle(im, (2, 2), (w-3, h-3), 0, t)
    else:
        s = min(w, h); im = _c(s, s); w = h = s; cv2.circle(im, (s//2, s//2), s//2-3, 0, t)
    fuente = cv2.FONT_HERSHEY_SIMPLEX
    if r.random() < .35: fuente = fuente | cv2.FONT_ITALIC      # la 'A' del plano es cursiva
    fs = 0.4 + 0.25*min(w, h)/60 * (2/len(txt))**.5
    (tw, th), _ = cv2.getTextSize(txt, fuente, fs, max(1, t-1))
    cv2.putText(im, txt, ((w-tw)//2, (h+th)//2), fuente, fs, 0, max(1, t-1), cv2.LINE_AA)
    if r.random() < .45:
        # cable que sale del costado, como en el plano. Va DENTRO del sprite para que el
        # modelo vea el simbolo con su cable pegado, que es como aparece siempre.
        L = r.randint(4, 12)
        lado = r.random()
        im2 = _c(h, w + L)
        if lado < .5:
            im2[:, L:] = im; cv2.line(im2, (0, h//2), (L, h//2), 0, 1)
        else:
            im2[:, :w] = im; cv2.line(im2, (w-1, h//2), (w+L-1, h//2), 0, 1)
        im = im2
    return im
def contactor(r):
    w, h = r.randint(40, 70), r.randint(70, 110); im = _c(h, w); t = r.randint(1, 2)
    cv2.line(im, (w//2, 0), (w//2, h//3), 0, t); cv2.line(im, (w//2, h//3), (w//5, 2*h//3), 0, t); cv2.line(im, (w//2, 2*h//3), (w//2, h-1), 0, t)
    cv2.circle(im, (w//2, h//3), 4, 0, 1)
    if r.random() < .5: cv2.rectangle(im, (w-18, h//2-8), (w-3, h//2+8), 0, t)
    return im
def caja_punteada(r):
    """Recuadro de linea PUNTEADA con el nombre de un equipo adentro. ES un componente.

    21/09. Es el ultimo falso negativo que le queda al plan R: el `PLC / LOGICAS: / TRANSF AUT`
    de nyw-un-01, que ninguno de los 13 modelos detecta (sus detecciones mas cercanas llegan a
    confianza 0,087). En el dataset el recuadro punteado aparece SOLO como negativo -los marcos
    punteados vacios no son componentes-, asi que el modelo aprendio que punteado = ignorar.

    La diferencia entre los dos casos es el contenido: un marco punteado VACIO (o que solo
    agrupa otros simbolos) no es un componente; uno con el nombre de un equipo adentro si lo es,
    igual que su equivalente de linea llena. Este generador cubre ese segundo caso.
    """
    lineas = r.choice([
        ['PLC', 'LOGICAS:', 'TRANSF AUT'],
        ['PLC', 'Comando'],
        ['UPS', '10 kVA'],
        ['Control de', 'Iluminacion'],
        ['Central de', 'Incendio'],
        ['Tablero de', 'Transferencia'],
        ['Banco de', 'Capacitores'],
        ['Modulo', 'de Control'],
    ])
    fs = r.uniform(.32, .5); t = 1
    med = [cv2.getTextSize(x, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)[0] for x in lineas]
    tw = max(m[0] for m in med); th = med[0][1]
    pad = r.randint(6, 16); inter = int(th * r.uniform(1.6, 2.2))
    w = tw + 2 * pad; h = inter * len(lineas) + 2 * pad
    im = _c(h, w)
    # rectangulo punteado: trazos cortos con hueco, como lo dibuja el CAD
    paso = r.randint(5, 10); trazo = max(2, int(paso * r.uniform(.45, .7)))
    for x in range(1, w - 1, paso):
        cv2.line(im, (x, 1), (min(w - 2, x + trazo), 1), 0, t)
        cv2.line(im, (x, h - 2), (min(w - 2, x + trazo), h - 2), 0, t)
    for y in range(1, h - 1, paso):
        cv2.line(im, (1, y), (1, min(h - 2, y + trazo)), 0, t)
        cv2.line(im, (w - 2, y), (w - 2, min(h - 2, y + trazo)), 0, t)
    for i, x in enumerate(lineas):
        sw = med[i][0]
        cv2.putText(im, x, ((w - sw) // 2, pad + inter * i + th), cv2.FONT_HERSHEY_SIMPLEX, fs, 0, 1, cv2.LINE_AA)
    return im


def caja_nombre(r):
    """Caja rectangular grande con un nombre largo en 2-3 lineas.

    Tipo 'Controlador para / Transferencia / Automatica' (IE-UNI-01), que D no detecta.
    Se diferencia de caja_letras en que el texto es largo y la caja mucho mas ancha.
    """
    # 19/09: se amplio con los rotulos que aparecen en los planos de Marcelo. El modelo N
    # ponia 3 o 4 cajas superpuestas y desalineadas sobre estos recuadros en vez de una sola
    # ('Iso-Gard IG6', 'Fuente 24 VCC 2 A', 'UPS 6 kVA 15 min', 'VigilOhm IM400'): son la
    # mayor parte de las cajas mal puestas que Tomas marco sobre los PDF.
    lineas = r.choice([
        ['Controlador para', 'Transferencia', 'Automatica'],
        ['Power Meter', 'Schneider Electric'],
        ['Central de', 'Incendio'],
        ['Tablero de', 'Comando'],
        ['Medicion', 'de Energia'],
        ['Banco de', 'Capacitores'],
        ['Grupo', 'Electrogeno'],
        ['UPS', '10 kVA'],
        ['Iso-Gard', 'IG6'],
        ['VigilOhm', 'IM400'],
        ['Fuente', '24 VCC', '2 A'],
        ['Fuente', '24 VCC', '5 A'],
        ['UPS', '6 kVA', '15 min'],
        ['UPS', '2.5 kVA', '30 min'],
        ['UPS', '1 kVA', '15 min'],
        ['Conexiones de', 'comando desde', 'TGE'],
        ['Modulo de', 'Señalizacion'],
        ['Rele de', 'Proteccion'],
        ['PLC', 'Comando'],
        ['Analizador', 'de Redes'],
    ])
    fs = r.uniform(.32, .5); t = r.randint(1, 2)
    med = [cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)[0] for s in lineas]
    tw = max(m[0] for m in med); th = med[0][1]
    pad = r.randint(6, 14); inter = int(th * r.uniform(1.6, 2.1))
    w = tw + 2*pad; h = inter*len(lineas) + 2*pad
    im = _c(h, w)
    cv2.rectangle(im, (1, 1), (w-2, h-2), 0, t)
    for i, s in enumerate(lineas):
        sw = med[i][0]
        cv2.putText(im, s, ((w-sw)//2, pad + inter*i + th), cv2.FONT_HERSHEY_SIMPLEX, fs, 0, 1, cv2.LINE_AA)
    return im

# 16/09 (Tomas): el PAT sale del dataset del plan F. Se deja la funcion `pat` definida
# porque los pesos de D si se entrenaron con ella y los GT actuales lo cuentan como componente.
# `caja_nombre` y `caja_letras` van repetidos: los recuadros con texto son de lo que peor
# encuadra el modelo (varias cajas superpuestas sobre el mismo rotulo) y necesitan mas peso.
# 20/09 (plan R): `caja_nombre` vuelve a peso simple si GENS_NOMBRE_SIMPLE=1. Con peso doble
# (plan P/Q) el modelo generalizo "recuadro con texto adentro = componente" hasta las celdas de
# la planilla de circuitos de LU-UN-01, y esos falsos positivos le robaban asignaciones a los
# contactores en el matching 1-a-1. `caja_letras` se deja en doble: es el generador del
# amperimetro `A$C45664E16` y no es el que se desbordo.
GENS = [caja_t, bornera_din, bornera_circ, bornera_strip, fusible, lampara, caja_letras,
        contactor, caja_nombre, caja_nombre, caja_letras, caja_punteada]
if os.environ.get('GENS_NOMBRE_SIMPLE') == '1':
    GENS = [g for i, g in enumerate(GENS) if i != 9]
if os.environ.get('GENS_SIN_PUNTEADA') == '1':
    # el plan T la saca: no cumplio su proposito y `compose.marco_punteado` cubre el caso a la
    # escala real, que era lo que faltaba.
    GENS = [g for g in GENS if g.__name__ != 'caja_punteada']
