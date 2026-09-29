"""Ajusta al dibujo real las cajas que la auditoria (`corregir_gt.py`) agrego al GT.

`corregir_gt.py` solo tenia el CENTRO que dio el revisor, asi que le armo a cada faltante una
caja del tamano tipico del plano. Eso deja dos defectos que se ven apenas se renderiza (§5.6,
"antes de perseguir un FN verifica que el GT este bien"):

  1. La caja no coincide con el simbolo: los rotulos de tablero de test1 son rectangulos altos
     y la caja tipica del plano es la de un interruptor, mucho mas chica y corrida.
  2. Hay duplicados. Los tiles de la revision se solapan, asi que un componente que cae en el
     borde se marco dos veces, una por tile. Como el matching es 1-a-1 por distancia de
     centros, el duplicado es un FN que ningun modelo puede evitar.

Arreglo: se busca en la imagen renderizada el contorno de tinta que contiene al centro y se usa
su bbox. Si el contorno tiene tamano razonable se reemplaza la caja; si no (quedo enganchado a
un cable y se comio medio plano) se deja la sintetica. Despues se fusionan las filas AUDIT que
quedan sobre la misma caja.

    py -3 src/snap_audit_gt.py            # solo informa
    py -3 src/snap_audit_gt.py --aplicar  # escribe <nombre>_v4.csv
"""
import sys, os, csv, glob, shutil, math
import cv2, numpy as np, ezdxf

sys.path.insert(0, os.path.dirname(__file__))
from paths import BASE
from render import render_doc, cad2px, px2cad
from scale import auto_ppc

PLANOS = {
    'test1': ('test1.dxf', 'test/test_1/verdad_terreno/test1_completo.csv'),
    'test_2': ('test_2.dxf', 'test/test_2/verdad_terreno/test_2_completo.csv'),
    'fl_un_02': ('dxf/FL-UN-02_tablero_1.dxf', 'dxf/fl_un_02_gt_completo.csv'),
    'tsss_2': ('TSSS_2 (1).dxf', 'dxf/tsss_2_gt_completo.csv'),
}
for _p in glob.glob(os.path.join(BASE, 'claudio_v2', 'data', 'test_marcelo', '*.dxf')):
    _n = os.path.basename(_p)[:-4]
    PLANOS[_n] = (os.path.relpath(_p, BASE), os.path.relpath(_p[:-4] + '_gt.csv', BASE))

MIN_F, MAX_F = 0.35, 4.0     # tamano aceptable del contorno, en lados tipicos del plano
# 21/09, CAUSA RAIZ de varios errores de este script: el render dibuja los CONTORNOS de los
# componentes en gris claro (167-211) y solo el TEXTO en negro. Con el umbral en 128 los
# recuadros eran invisibles para el script y lo unico "con tinta" era el texto, asi que la
# busqueda terminaba enganchando cajas del GT a las palabras. Todo lo que pregunte "hay algo
# dibujado aca" tiene que usar este umbral, no 128.
TINTA = 230
MAX_AR = 8.0
EXPLOTAR = int(os.environ.get('SNAP_EXPLOTAR', '0'))   # profundidad de explosion de INSERT                 # relacion de aspecto maxima: mas que esto es un cable


def lado_tipico(filas):
    ws = sorted(abs(float(r['x2']) - float(r['x1'])) for r in filas if r.get('x1'))
    hs = sorted(abs(float(r['y2']) - float(r['y1'])) for r in filas if r.get('y1'))
    if not ws:
        return .3, .3
    return ws[len(ws) // 2], hs[len(hs) // 2]


def contorno_en(img, px, py, lado_px):
    """bbox del contorno de tinta que contiene (px,py), en pixeles. None si no sirve."""
    r = int(lado_px * MAX_F)
    x0, y0 = max(0, int(px) - r), max(0, int(py) - r)
    x1, y1 = min(img.shape[1], int(px) + r), min(img.shape[0], int(py) + r)
    sub = img[y0:y1, x0:x1]
    if sub.size == 0:
        return None
    tinta = (sub < TINTA).astype(np.uint8)
    if not tinta.any():
        return None
    # dilatar un poco para cerrar el trazo del rectangulo antes de buscar el contorno
    cerr = cv2.morphologyEx(tinta, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    cnts, _ = cv2.findContours(cerr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cx, cy = px - x0, py - y0
    mejor = None
    for c in cnts:
        bx, by, bw, bh = cv2.boundingRect(c)
        if not (bx - 2 <= cx <= bx + bw + 2 and by - 2 <= cy <= by + bh + 2):
            continue
        if bw < 3 or bh < 3:
            continue
        if max(bw, bh) / max(1, min(bw, bh)) > MAX_AR:
            continue
        if not (lado_px * MIN_F <= max(bw, bh) <= lado_px * MAX_F):
            continue
        area = bw * bh
        if mejor is None or area < mejor[0]:
            mejor = (area, bx + x0, by + y0, bw, bh)
    if mejor is None:
        return None
    return mejor[1], mejor[2], mejor[3], mejor[4]



# --- reconstruccion del recuadro a partir de las LINE del DXF -------------------------------
# Los rotulos de tablero de test1 no son polilineas cerradas: son cuatro LINE sueltas. Buscar
# un contorno en la imagen no alcanza (el trazo del recuadro y el texto de adentro quedan como
# contornos separados, y el mas chico es el texto). Con las lineas el rectangulo es exacto.
# Para estos el componente ES, por definicion, un rectangulo dibujado, y el metodo los encuentra
# (19 de 20 en test1). Si no aparece ninguno, la marca fue sobre aire y la fila sale: una caja de
# referencia inventada es peor que un faltante. NO incluir aqui `bobina_apertura`: esos recuadros
# viven adentro de un bloque y el script no los ve, asi que "no lo encontre" no prueba nada.
TIPOS_SIN_RECUADRO_FUERA = {'rotulo_con_nombre'}
TIPOS_RECUADRO = {'rotulo_con_nombre', 'bobina_apertura', 'toma_fusibles', 'rectificador',
                  'ondulador', 'plc', 'ups', 'tablero_transferencia', 'bateria', 'medidor'}
EPS = 1e-6


def lineas_del_doc(doc, hueco=0.25):
    """Segmentos rectos del plano, incluidos los de adentro de los bloques.

    Ojo: iterar solo el modelspace deja afuera casi todo. Los recuadros `BA` de LU-UN-01 son
    geometria de un bloque, no del modelspace, y al no verlos el script daba por inexistentes
    123 componentes reales y los borraba del GT. Hay que explotar los INSERT.
    """
    ver, hor = [], []

    def agregar(ax, ay, bx, by):
        if abs(ax - bx) < 1e-4 and abs(ay - by) > 1e-4:
            ver.append((ax, min(ay, by), max(ay, by)))
        elif abs(ay - by) < 1e-4 and abs(ax - bx) > 1e-4:
            hor.append((ay, min(ax, bx), max(ax, bx)))

    def de_entidad(e, prof=0):
        t = e.dxftype()
        try:
            if t == 'LINE':
                a, b = e.dxf.start, e.dxf.end
                agregar(a.x, a.y, b.x, b.y)
            elif t == 'LWPOLYLINE':
                pts = [(q[0], q[1]) for q in e.get_points()]
                if getattr(e, 'closed', False) and len(pts) > 2:
                    pts = pts + [pts[0]]
                for i in range(len(pts) - 1):
                    agregar(pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1])
            elif t == 'POLYLINE':
                pts = [(v.dxf.location.x, v.dxf.location.y) for v in e.vertices]
                if e.is_closed and len(pts) > 2:
                    pts = pts + [pts[0]]
                for i in range(len(pts) - 1):
                    agregar(pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1])
            elif t == 'INSERT' and prof < EXPLOTAR:
                # Explotar todos los INSERT de un plano grande tarda mas que el resto del
                # script junto. No hace falta: si el recuadro no aparece, la fila conserva la
                # caja que ya tenia (que es lo que habia antes) y no se pierde nada.
                for ve in e.virtual_entities():
                    de_entidad(ve, prof + 1)
        except Exception:
            pass

    for e in doc.modelspace():
        de_entidad(e)
    # 21/09: unir los segmentos colineales separados por huecos. Un marco PUNTEADO no es una
    # linea larga sino decenas de trazos cortos, y `recuadro_en` pide una vertical cuyo tramo
    # cubra el punto: con los trazos sueltos nunca lo encuentra. Por eso el recuadro del
    # `PLC / LOGICAS / TRANSF AUT` de nyw-un-01 se le escapaba y la caja terminaba sobre el
    # texto de adentro.
    if hueco > 0:      # con hueco 0 NO se une nada: unir aunque sea trazos que se tocan ya
        ver = _unir(ver, hueco)   # alarga los tramos y la busqueda encuentra el marco del
        hor = _unir(hor, hueco)   # tablero en vez de la caja del componente (55 -> 40 en LU)
    # Como arrays: un plano grande tiene cientos de miles de segmentos y recorrerlos en Python
    # por cada fila de la auditoria tarda minutos. Vectorizado son milisegundos.
    V = np.array(ver, float).reshape(-1, 3) if ver else np.zeros((0, 3))
    H = np.array(hor, float).reshape(-1, 3) if hor else np.zeros((0, 3))
    return V, H


def _unir(segs, hueco):
    """Une segmentos con la misma coordenada cuyos tramos se tocan o casi."""
    if not segs:
        return segs
    segs.sort()
    out = []
    ci, a, b = segs[0]
    for c, x, y in segs[1:]:
        if abs(c - ci) < 1e-4 and x <= b + hueco:
            b = max(b, y)
        else:
            out.append((ci, a, b)); ci, a, b = c, x, y
    out.append((ci, a, b))
    return out


def hay_tinta(img, px, py, lado_px, frac=.6, minimo=12):
    """True si hay algo dibujado alrededor de (px,py).

    21/09, ERROR CORREGIDO: antes esto pedia que mas del 1% de la VENTANA fuera tinta, y la
    ventana es grande. Un simbolo chico -una bornera de 12x12 px- adentro de una ventana de
    72x72 da 0,8% y quedaba descartado como "sobre aire". Asi se borraron del GT componentes
    reales (la fotocelula de test_2 y las borneras de salida de fl_un_02, verificados a ojo).
    El umbral tiene que ser ABSOLUTO: importa que haya algo dibujado, no que ocupe un
    porcentaje de un area que elegimos nosotros.
    """
    r = max(4, int(lado_px * frac))
    x0, y0 = max(0, int(px) - r), max(0, int(py) - r)
    x1, y1 = min(img.shape[1], int(px) + r), min(img.shape[0], int(py) + r)
    sub = img[y0:y1, x0:x1]
    return sub.size > 0 and int((sub < TINTA).sum()) >= minimo


def _borde_pintado(img, meta, X1, Y1, X2, Y2, minimo=.35):
    """Fraccion de cada borde del rectangulo que tiene tinta encima.

    21/09: sin esto, `recuadro_en` acepta rectangulos que no existen como dibujo, formados por
    segmentos sueltos que casualmente encierran el punto. Asi le puso al `PLC` de nyw-un-01 una
    caja sobre un pedazo del interior ("PLC" y "AUT.") en vez del marco punteado entero, y ese
    era el ultimo falso negativo que quedaba. Un rectangulo de verdad tiene tinta en sus cuatro
    lados; uno inventado, no. El umbral es bajo (35%) porque los marcos PUNTEADOS tienen huecos.
    """
    a = cad2px(meta, X1, Y1); b = cad2px(meta, X2, Y2)
    x0, y0 = int(min(a[0], b[0])), int(min(a[1], b[1]))
    x1, y1 = int(max(a[0], b[0])), int(max(a[1], b[1]))
    if x1 - x0 < 3 or y1 - y0 < 3:
        return False
    H, W = img.shape
    x0 = max(0, x0); y0 = max(0, y0); x1 = min(W - 1, x1); y1 = min(H - 1, y1)
    tol = 2
    def frac(sub):
        return 0. if sub.size == 0 else float((sub < TINTA).any(axis=0).mean())
    arriba = frac(img[max(0, y0 - tol):y0 + tol + 1, x0:x1 + 1])
    abajo = frac(img[max(0, y1 - tol):y1 + tol + 1, x0:x1 + 1])
    izq = frac(img[y0:y1 + 1, max(0, x0 - tol):x0 + tol + 1].T)
    der = frac(img[y0:y1 + 1, max(0, x1 - tol):x1 + tol + 1].T)
    return min(arriba, abajo, izq, der) >= minimo


def recuadro_en(V, H, x, y, lado, img=None, meta=None, min_borde=.35):
    """Rectangulo dibujado que encierra a (x,y). None si no hay.

    Toma las lineas mas cercanas de cada lado -que es lo que da la caja del componente- y, si
    se le pasan `img` y `meta`, verifica que los cuatro bordes tengan algo dibujado encima.
    Esa validacion solo RECHAZA: sin ella el script acepta cualquier combinacion de segmentos
    que casualmente encierre el punto, y asi le puso al `PLC` de nyw-un-01 una caja sobre un
    pedazo del interior en vez del marco entero.
    """
    if len(V) == 0 or len(H) == 0:
        return None
    cubre = (V[:, 1] - .02 <= y) & (y <= V[:, 2] + .02)
    vx = V[cubre, 0]
    if vx.size == 0:
        return None
    izq = vx[vx <= x + EPS]; der = vx[vx >= x - EPS]
    if izq.size == 0 or der.size == 0:
        return None
    xl = izq.max(); xr = der.min()
    # El tope era lado*3 y el marco punteado del PLC de nyw-un-01 mide 2,778 con lado 0,913:
    # lo rechazaba por 1,4%. Con la validacion de bordes ya no hace falta ser tan estricto.
    if xr - xl < 1e-4 or xr - xl > lado * 4:
        return None
    ch = (H[:, 1] - .02 <= xl) & (H[:, 2] + .02 >= xr)
    hy = H[ch, 0]
    if hy.size == 0:
        return None
    aba = hy[hy <= y + EPS]; arr = hy[hy >= y - EPS]
    if aba.size == 0 or arr.size == 0:
        return None
    yb = aba.max(); yt = arr.min()
    if yt - yb < 1e-4 or yt - yb > lado * 4:
        return None
    w_, h_ = xr - xl, yt - yb
    if max(w_, h_) / max(1e-9, min(w_, h_)) > 6.0:
        return None
    if img is not None and meta is not None and not _borde_pintado(img, meta, xl, yb, xr, yt, min_borde):
        return None
    return float(xl), float(yb), float(xr), float(yt)



def hay_tinta(img, px, py, lado_px, frac=.6, minimo=12):
    """True si hay algo dibujado alrededor de (px,py).

    21/09, ERROR CORREGIDO: antes esto pedia que mas del 1% de la VENTANA fuera tinta, y la
    ventana es grande. Un simbolo chico -una bornera de 12x12 px- adentro de una ventana de
    72x72 da 0,8% y quedaba descartado como "sobre aire". Asi se borraron del GT componentes
    reales (la fotocelula de test_2 y las borneras de salida de fl_un_02, verificados a ojo).
    El umbral tiene que ser ABSOLUTO: importa que haya algo dibujado, no que ocupe un
    porcentaje de un area que elegimos nosotros.
    """
    r = max(4, int(lado_px * frac))
    x0, y0 = max(0, int(px) - r), max(0, int(py) - r)
    x1, y1 = min(img.shape[1], int(px) + r), min(img.shape[0], int(py) + r)
    sub = img[y0:y1, x0:x1]
    return sub.size > 0 and int((sub < TINTA).sum()) >= minimo


def blob_cercano(img, px, py, lado_px, radio=1.8):
    """bbox de la mancha de tinta mas cercana a (px,py). None si no hay nada cerca.

    21/09. Las coordenadas que dio el revisor traen un desfasaje sistematico de unos 39 px
    (0,67 unidades CAD) contra la tinta real: no estan sobre aire, estan corridas. Se vio al
    renderizar las filas que el script descartaba -caian al lado de una fotocelula, de recuadros
    chicos y de las borneras de salida, todos componentes de verdad-.

    Descartarlas borra componentes reales del GT, que es justo lo que la auditoria vino a
    arreglar. Engancharlas a la mancha mas cercana recupera el componente Y corrige la caja.
    """
    r = max(8, int(lado_px * radio))
    x0, y0 = max(0, int(px) - r), max(0, int(py) - r)
    x1, y1 = min(img.shape[1], int(px) + r), min(img.shape[0], int(py) + r)
    sub = img[y0:y1, x0:x1]
    if sub.size == 0:
        return None
    tinta = (sub < TINTA).astype(np.uint8)
    if not tinta.any():
        return None
    cerr = cv2.morphologyEx(tinta, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, lab, stats, cent = cv2.connectedComponentsWithStats(cerr, 8)
    cx, cy = px - x0, py - y0
    mejor = None
    for i in range(1, n):
        bx, by, bw, bh, area = stats[i]
        if area < 6 or bw < 3 or bh < 3:
            continue
        if max(bw, bh) > lado_px * 3.5:          # un cable largo, no un simbolo
            continue
        dx = max(bx - cx, 0, cx - (bx + bw)); dy = max(by - cy, 0, cy - (by + bh))
        d = (dx * dx + dy * dy) ** .5
        if mejor is None or d < mejor[0]:
            mejor = (d, bx + x0, by + y0, bw, bh)
    if mejor is None or mejor[0] > lado_px * radio:
        return None
    return mejor[1], mejor[2], mejor[3], mejor[4]


def textos_del_doc(doc):
    """Cajas aproximadas de los textos, para no engancharle una caja del GT a una palabra.

    21/09: sin esto, la busqueda de la mancha mas cercana pegaba la caja al numero de circuito,
    a la etiqueta del cable y a la palabra "GOLPE". Un texto no es un componente, asi que un
    candidato que cae encima de uno se descarta y se sigue buscando.
    """
    T = []
    for e in doc.modelspace():
        t = e.dxftype()
        if t not in ('TEXT', 'MTEXT'):
            continue
        try:
            txt = e.dxf.text if t == 'TEXT' else e.text
            h = float(getattr(e.dxf, 'height', 0) or 0.2)
            p = e.dxf.insert
            w = max(1, len(str(txt))) * h * .75
            T.append((p.x - h * .4, p.y - h * .5, p.x + w, p.y + h * 1.3))
        except Exception:
            pass
    return np.array(T, float).reshape(-1, 4) if T else np.zeros((0, 4))


def sobre_texto(T, x, y):
    if len(T) == 0:
        return False
    return bool(np.any((T[:, 0] <= x) & (x <= T[:, 2]) & (T[:, 1] <= y) & (y <= T[:, 3])))


def main():
    aplicar = '--aplicar' in sys.argv
    tot_snap = tot_dup = tot_aud = 0
    solo = None
    if '--solo' in sys.argv:
        solo = sys.argv[sys.argv.index('--solo') + 1]
    for nom, (dxf, gtp) in sorted(PLANOS.items()):
        if solo and solo.lower() not in nom.lower():
            continue
        print('[%s] leyendo...' % nom[:28], flush=True)
        fg = os.path.join(BASE, gtp)
        for suf in ('_v3.csv', '_v2.csv'):
            if os.path.exists(fg[:-4] + suf):
                fg = fg[:-4] + suf
                break
        else:
            continue
        filas = list(csv.DictReader(open(fg, newline='', encoding='utf-8', errors='ignore')))
        campos = list(csv.DictReader(open(fg, newline='', encoding='utf-8', errors='ignore')).fieldnames or [])
        aud = [r for r in filas if r['block_name'].startswith('AUDIT-')]
        if not aud:
            continue
        doc = ezdxf.readfile(os.path.join(BASE, dxf))
        print('[%s] renderizando...' % nom[:28], flush=True)
        img, meta = render_doc(doc, auto_ppc(doc))
        print('[%s] render %s' % (nom[:28], img.shape), flush=True)
        lw, lh = lado_tipico([r for r in filas if not r['block_name'].startswith('AUDIT-')])
        lado_px = max(lw, lh) * meta['ppc']
        # Dos versiones de las lineas. La primera SIN unir es la precisa: encuentra la caja
        # del componente. La segunda une los trazos separados por huecos y sirve solo para los
        # marcos PUNTEADOS, que son decenas de trazos cortos. Se prueba en ese orden porque
        # unir de entrada hace que la busqueda encuentre rectangulos mas grandes -el marco del
        # tablero en vez del componente- y los rectangulos hallados en LU-UN-01 caian de 55 a 40.
        ver, hor = lineas_del_doc(doc, hueco=0.0)
        verm, horm = lineas_del_doc(doc, hueco=0.25)
        TXT = textos_del_doc(doc)
        snap = rect = sin_rect = 0
        fuera_falso = set()
        for r in aud:
            tipo = r['block_name'][len('AUDIT-'):]
            X, Y = float(r['x_cad']), float(r['y_cad'])
            px, py = cad2px(meta, X, Y)
            if True:
                # 21/09: se prueba para TODOS los tipos, no solo los "recuadro". Medido: en
                # test_2 las 7 filas de la auditoria encuentran un rectangulo que las encierra
                # con un error mediano de 0,042 unidades CAD -la fotocelula y los contactores
                # tambien estan dibujados dentro de un recuadro-. Restringirlo por tipo las
                # mandaba al camino de respaldo, que es mucho menos preciso.
                # El test es exigente (cuatro lineas que encierran el punto, con proporcion
                # razonable), asi que probarlo de mas no inventa cajas: o encuentra o no.
                # La pasada SIN unir se le exige mucha mas tinta en los bordes (0,70). En un
                # marco PUNTEADO los trazos sueltos hacen de verticales cortas y arman un
                # rectangulo falso en el interior: al `PLC` de nyw-un-01 le ponia una caja sobre
                # las palabras "PLC" y "AUT." en vez del marco entero, y pasaba la validacion
                # floja porque el texto cruza esos bordes. Un rectangulo de linea llena tiene
                # los cuatro lados casi completos; uno inventado, no.
                box = recuadro_en(ver, hor, X, Y, max(lw, lh), img, meta, min_borde=.70)
                if box is None:
                    box = recuadro_en(verm, horm, X, Y, max(lw, lh), img, meta, min_borde=.35)
                if box is not None:
                    X1, Y1, X2, Y2 = box
                    r['x1'], r['y1'], r['x2'], r['y2'] = ('%.4f' % X1, '%.4f' % Y1, '%.4f' % X2, '%.4f' % Y2)
                    r['x_cad'], r['y_cad'] = '%.4f' % ((X1 + X2) / 2), '%.4f' % ((Y1 + Y2) / 2)
                    rect += 1
                    continue
                if tipo in TIPOS_SIN_RECUADRO_FUERA:
                    fuera_falso.add(id(r)); sin_rect += 1
                    continue
            # Sin rectangulo: NO se descarta la fila por eso. Que el script no encuentre el
            # recuadro no prueba que no este (los `BA` viven dentro de un bloque). Lo unico que
            # justifica sacar una fila es que ahi no haya NADA dibujado: eso si es un faltante
            # inventado, y una caja de referencia sobre aire es peor que un faltante.
            got = contorno_en(img, px, py, lado_px)
            if got is not None:
                bx, by, bw, bh = got
                mx, my = px2cad(meta, bx + bw / 2., by + bh / 2.)
                if sobre_texto(TXT, mx, my):
                    got = None          # es una palabra, no un componente
            if got is None:
                # sin contorno en el punto: buscar la mancha de tinta mas cercana. Solo si no
                # hay NADA en todo el radio se considera que la marca fue sobre aire.
                # Radio corto a proposito. Con 1,8 lados llegaba hasta el texto de al lado y
                # engancho la caja al numero de circuito, a la etiqueta del cable y a la
                # palabra "GOLPE": peor que no hacer nada. Con 0,8 alcanza para el desfasaje
                # real observado (39 px en fl_un_02) y no llega al texto vecino.
                got = blob_cercano(img, px, py, lado_px, radio=0.8)
                if got is not None:
                    bx, by, bw, bh = got
                    mx, my = px2cad(meta, bx + bw / 2., by + bh / 2.)
                    if sobre_texto(TXT, mx, my):
                        got = None
                if got is None:
                    # Ni rectangulo ni contorno ni mancha cerca: se DEJA la caja sintetica en
                    # el punto original. Las coordenadas del revisor resultaron precisas
                    # (error mediano 0,042), asi que el punto es mejor informacion que
                    # cualquier cosa a la que podamos saltar, y borrar la fila esconderia un
                    # componente real, que es lo que esta auditoria vino a evitar.
                    sin_rect += 1
                    continue
            bx, by, bw, bh = got
            X1, Y2 = px2cad(meta, bx, by)
            X2, Y1 = px2cad(meta, bx + bw, by + bh)
            r['x1'], r['y1'], r['x2'], r['y2'] = ('%.4f' % X1, '%.4f' % Y1, '%.4f' % X2, '%.4f' % Y2)
            r['x_cad'], r['y_cad'] = '%.4f' % ((X1 + X2) / 2), '%.4f' % ((Y1 + Y2) / 2)
            snap += 1
        aud = [r for r in aud if id(r) not in fuera_falso]
        filas = [r for r in filas if id(r) not in fuera_falso]
        # Fusionar duplicados. Dos fuentes distintas:
        #   a) los tiles de la revision se solapan, asi que un componente del borde se marco
        #      dos veces y quedan dos filas AUDIT sobre lo mismo;
        #   b) el revisor marco como "faltante" algo que el GT YA TENIA. Eso no se veia
        #      comparando las filas AUDIT entre si: hay que compararlas contra TODAS.
        # El criterio es sin umbral inventado: es duplicado si el centro de una cae adentro de
        # la caja de la otra. Un umbral por distancia deja pasar casos por centimetros (el
        # acople de nyw-un-01 quedaba a 0,253 con un corte en 0,187).
        def caja(r):
            return (min(float(r['x1']), float(r['x2'])), min(float(r['y1']), float(r['y2'])),
                    max(float(r['x1']), float(r['x2'])), max(float(r['y1']), float(r['y2'])))

        def centro(r):
            return float(r['x_cad']), float(r['y_cad'])

        def solapan(a, b):
            ax, ay = centro(a); bx, by = centro(b)
            A = caja(a); B = caja(b)
            if (B[0] <= ax <= B[2] and B[1] <= ay <= B[3]) or                (A[0] <= bx <= A[2] and A[1] <= by <= A[3]):
                return True
            # El centro que dio el revisor es aproximado por diseno, asi que puede caer justo
            # afuera de la caja del bloque que ya estaba: el acople de nyw-un-01 quedaba a 0,25
            # de un `INTERRUPTOR` real. Se compara la distancia contra el TAMANO de las cajas,
            # no contra un numero fijo. Con 0,5 diagonales dos TM-DIN vecinos (separados 0,72,
            # diagonal 1,05 -> corte 0,53) NO se fusionan, que es el caso a no romper.
            da = math.hypot(A[2] - A[0], A[3] - A[1])
            db = math.hypot(B[2] - B[0], B[3] - B[1])
            return math.hypot(ax - bx, ay - by) < 0.5 * (da + db) / 2

        previas = [r for r in filas if not r['block_name'].startswith('AUDIT-')]
        vistos, fuera = list(previas), set()
        for r in aud:
            if any(solapan(r, v) for v in vistos):
                fuera.add(id(r))
            else:
                vistos.append(r)
        # Cajas degeneradas: ancho o alto cero. No son componentes (en nyw-un-01 `*U57` es una
        # linea suelta encima de un diferencial que ya tiene su propia fila) y ningun detector
        # de cajas puede acertarle a algo de area cero: es un FN garantizado para todos.
        degen = 0
        for r in filas:
            if r.get('x1') and (abs(float(r['x2']) - float(r['x1'])) < 1e-9 or
                                abs(float(r['y2']) - float(r['y1'])) < 1e-9):
                fuera.add(id(r)); degen += 1
        filas = [r for r in filas if id(r) not in fuera]
        tot_snap += snap + rect; tot_dup += len(fuera); tot_aud += len(aud) + sin_rect
        print('  %-32s AUDIT=%3d  recuadro=%3d  contorno=%3d  sin ajustar (caja original)=%3d  dup=%3d (degen %d)  -> %d filas'
              % (nom[:32], len(aud), rect, snap, sin_rect, len(fuera) - degen, degen, len(filas)))
        if aplicar:
            base_v = os.path.join(BASE, gtp)[:-4]
            shutil.copy2(fg, os.path.join(BASE, '_para_borrar',
                                          os.path.basename(fg) + '.antes_de_snap'))
            with open(base_v + '_v4.csv', 'w', newline='', encoding='utf-8') as f:
                w = csv.DictWriter(f, fieldnames=campos)
                w.writeheader(); w.writerows(filas)
    print('TOTAL  AUDIT=%d  ajustadas=%d  duplicadas quitadas=%d' % (tot_aud, tot_snap, tot_dup))
    if not aplicar:
        print('\n(sin --aplicar no se escribe nada)')


if __name__ == '__main__':
    main()
