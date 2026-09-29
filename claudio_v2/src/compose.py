"""Generador de tiles sinteticos 640x640 estilo plano unifilar.
Positivos: simbolos de la biblioteca (elmt->dxf, componentes manuales, bloques de planos de train, procedurales).
Negativos (fondo): cables, marcas de conductores, textos, recuadros punteados, flechas, tablas, fondo real de planos de train con componentes borrados.
Escala: altura de texto ~11 px (misma normalizacion que la inferencia)."""
import numpy as np, cv2, random, glob, json, os
from PIL import Image, ImageDraw, ImageFont
import sys; sys.path.insert(0, os.path.dirname(__file__))
from procsym import GENS

S = 640
import matplotlib as _mpl
_md = os.path.join(os.path.dirname(_mpl.__file__), 'mpl-data', 'fonts', 'ttf')
FONTS = [os.path.join(_md, 'DejaVuSans.ttf'), 'C:/Windows/Fonts/arial.ttf', 'C:/Windows/Fonts/arialn.ttf', 'C:/Windows/Fonts/ARIALN.TTF',
         '/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed.ttf', '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf']
FONTS = [f for f in FONTS if os.path.exists(f)]
_fc = {}
def font(sz, r):
    f = r.choice(FONTS); k = (f, sz)
    if k not in _fc: _fc[k] = ImageFont.truetype(f, sz)
    return _fc[k]

WORDS = ['4x25A', '2x10A', '2x16A', '4x40A', '4x63A', 'Icc=6kA', 'Curva C', 'Curva D', '30mA', '300mA', 'ID01', 'ID02', 'TS1A', 'TUG', 'TUE',
 'ILUMINAC.', 'RESERVA', 'LSOH 4x10mm+PE', 'BAND. PORTAC.', 'DE TGBT', '(TAB. GEN. BARRA NORMAL)', 'Seccionador Manual Bajo Carga', 'x3', 'R', 'S', 'T', 'N', 'PE',
 'TOMAS AA', 'PT. EST.', 'MOD. EMERG.', '25mm²', 'NSX160B', 'TMD', '3x380V', '50Hz', 'TABLERO SECCIONAL', 'EQ. AA', 'BOMBA', 'ASCENSOR', 'kW', '15 kVA',
 'cos φ=0.85', 'TS-TL1', 'BARRA NORMAL', 'EMERGENCIA', 'NORMAL', 'CIRCUITO', 'POTENCIA (kW)', 'DESTINO', 'L1', 'L2', 'L3', 'Q1', 'K1M', 'F2', 'X1', '1', '2', '3', '5', '6', '11',
 'comunicacion modbus', 'IN=100A', 'Ir=0.8In', '220V', 'TSFM-1/N', 'UPS COMANDO', 'PLANTA SUB-SUELO', 'NOTA:', 'VER DETALLE', 'Rev. A', 'ESC 1:50']

def rand_text(r):
    n = r.choice([1, 1, 1, 2, 3])
    return ' '.join(r.choice(WORDS) for _ in range(n)) if r.random() < .75 else ''.join(r.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-./x') for _ in range(r.randint(2, 9)))

# Simbolos que NO deben entrar al dataset. 16/09 (Tomas): la puesta a tierra sale del plan F.
# Son los dos unicos PAT de la biblioteca (revisados uno por uno sobre 1005 simbolos).
# Sprites que NO son componentes y por lo tanto no pueden llevar etiqueta.
# 18/09, revisados uno por uno por Tomas sobre la grilla de los 42 extras sacados de planos:
# p0246 es la palabra "CT" renderizada, p0245/p0247/p0259/p0260 son circulos y elipses de
# recorte, p0253/p0261/p0262 son lineas y palancas sueltas. Cada uno traia una caja que
# encerraba texto o cable, y de ahi salian las cajas mal encuadradas del dataset.
_EXCL_TOMAS = ('p0245.png,p0247.png,p0253.png,p0259.png,p0260.png,p0261.png,'
               'p0262.png,p0281.png,p0286.png,e00232.png')
EXCLUIR_SIM = set(os.environ.get('EXCLUIR_SIM', _EXCL_TOMAS).split(','))

def es_basura(g):
    """True solo si el sprite esta literalmente vacio: no hay tinta que aprender.

    18/09. La primera version filtraba tambien por forma (relacion de lados mayor a 6,
    densidad de tinta > 0.85, marco con el contenido chico). Tomas reviso uno por uno lo que
    ese filtro descartaba y el veredicto fue que casi todo eran componentes de verdad: las
    barras de arcos son borneras, los rectangulos macizos son aparatos y el simbolo dentro de
    su recuadro tambien lo es. La heuristica por forma estaba mal planteada: en un unifilar un
    componente puede ser largo y fino, o una mancha negra, sin dejar de ser un componente.
    Quedan afuera a mano los pocos que Tomas marco (ver EXCLUIR_SIM), y aca solo el caso en
    que no hay nada dibujado.
    """
    return ink_bbox(g) is None


class Lib:
    def __init__(self, lib_dir, bg_imgs, filtrar=True):
        # `filtrar=False` para las carpetas curadas a mano (los simbolos que paso Tomas,
        # recortados uno por uno del plano al bbox del GT): ahi el filtro automatico sobra
        # y de hecho se comia uno de los 21.
        self.syms = []
        for f in sorted(glob.glob(os.path.join(lib_dir, '*.png'))):
            if os.path.basename(f) in EXCLUIR_SIM: continue
            g = cv2.imread(f, 0)
            if g is None: continue
            if filtrar and es_basura(g): continue
            self.syms.append(g)
        self.bgs = bg_imgs  # lista de (img_gray_darkened, mask_uint8 donde 255=zona borrada)

def ink_bbox(a, thr=160):
    ys, xs = np.where(a < thr)
    if len(xs) == 0: return None
    return xs.min(), ys.min(), xs.max() + 1, ys.max() + 1

def stroke_aug(g, r):
    g = 255 - np.clip((255.0 - g) * r.uniform(1.5, 4), 0, 255).astype(np.uint8)
    p = r.random()
    if p < .25: g = cv2.erode(g, np.ones((2, 2), np.uint8))   # engrosa trazo
    return g

def put_text(canvas, x, y, txt, h, r, rot=False):
    f = font(max(7, int(h * 1.25)), r)
    img = Image.new('L', (int(len(txt) * h * 1.1) + 10, int(h * 2.2)), 255)
    ImageDraw.Draw(img).text((2, 1), txt, font=f, fill=0)
    a = np.array(img); bb = ink_bbox(a, 200)
    if bb is None: return
    a = a[:bb[3] + 1, :bb[2] + 2]
    if r.random() < .5: a = cv2.resize(a, (max(1, int(a.shape[1] * r.uniform(.7, 1.0))), a.shape[0]))
    if rot: a = cv2.rotate(a, cv2.ROTATE_90_COUNTERCLOCKWISE)
    blit_min(canvas, a, x, y)

def blit_min(canvas, a, x, y):
    H, W = canvas.shape; h, w = a.shape
    x0, y0 = max(0, x), max(0, y); x1, y1 = min(W, x + w), min(H, y + h)
    if x1 <= x0 or y1 <= y0: return False
    canvas[y0:y1, x0:x1] = np.minimum(canvas[y0:y1, x0:x1], a[y0 - y:y1 - y, x0 - x:x1 - x]); return True

def dashed_line(c, p0, p1, t, r, dash=None):
    L = int(np.hypot(p1[0]-p0[0], p1[1]-p0[1]))
    if L < 2: return
    d = dash or r.randint(6, 14); g = r.randint(3, 7)
    for s in range(0, L, d + g):
        e = min(L, s + d); a = s / L; b = e / L
        cv2.line(c, (int(p0[0]+(p1[0]-p0[0])*a), int(p0[1]+(p1[1]-p0[1])*a)), (int(p0[0]+(p1[0]-p0[0])*b), int(p0[1]+(p1[1]-p0[1])*b)), 0, t)

def hash_marks(c, x, y, r, t=1, esc=1.0):
    """Marcas /// del conductor. NEGATIVO.

    18 de los 59 falsos positivos de test5 son este patron, asi que conviene que el
    sintetico se parezca al real: 3-4 barras paralelas bien inclinadas, separacion
    proporcional al largo, y el nodo lleno al lado.
    """
    n = r.randint(3, 4); L = max(6, int(r.randint(10, 20) * esc))
    paso = max(3, int(L * r.uniform(.28, .45)))
    dx, dy = L // 2, int(L * r.uniform(.30, .55))
    for k in range(n):
        yy = y + k * paso
        cv2.line(c, (x - dx, yy + dy), (x + dx, yy - dy), 0, t)
    if r.random() < .7: cv2.circle(c, (x - dx - 2, y + n*paso + 2), max(2, int(2*esc)), 0, -1)

def clutter(c, r, txt_h, n_scale=1.0):
    """elementos negativos: cables, textos, marcas, recuadros, flechas, tablas, circulos numerados"""
    H, W = c.shape
    t = r.choice([1, 1, 1, 1, 2])
    for _ in range(int(r.randint(0, 3) * n_scale)):  # recuadro punteado / marco
        x0, y0 = r.randint(-100, W-50), r.randint(-100, H-50); x1, y1 = x0 + r.randint(150, 700), y0 + r.randint(150, 700)
        for p, q in [((x0,y0),(x1,y0)), ((x1,y0),(x1,y1)), ((x1,y1),(x0,y1)), ((x0,y1),(x0,y0))]:
            (dashed_line if r.random() < .6 else (lambda c_, a_, b_, t_, r_: cv2.line(c_, a_, b_, 0, t_)))(c, p, q, t, r)
    for _ in range(int(r.randint(2, 14) * n_scale)):  # textos sueltos
        put_text(c, r.randint(-40, W), r.randint(-10, H), rand_text(r), int(txt_h * r.uniform(.7, 1.8)), r, rot=r.random() < .15)
    for _ in range(int(r.randint(0, 3) * n_scale)):  # NEGATIVO: nota de 2-4 lineas alineadas, sin recuadro
        # 19/09: sobre los PDF de Marcelo el modelo N ponia 3 o 4 cajas superpuestas encima de
        # cada rotulo de varias lineas. `caja_nombre` le ensena que un recuadro con texto largo
        # es UN componente; esto le ensena lo contrario: texto alineado SIN recuadro no lleva
        # ninguna. Sin el par, el modelo solo ve "varias lineas juntas" y no sabe que hacer.
        nl = r.randint(2, 4); hh = int(txt_h * r.uniform(.8, 1.4)); inter = int(hh * r.uniform(1.25, 1.8))
        x, y = r.randint(-30, W-40), r.randint(-10, H-nl*inter)
        for i in range(nl):
            put_text(c, x, y + i*inter, rand_text(r), hh, r)
    for _ in range(int(r.randint(0, 5) * n_scale)):  # numeros en circulo
        x, y = r.randint(0, W), r.randint(0, H); R = int(txt_h * r.uniform(.9, 1.3))
        cv2.circle(c, (x, y), R, 0, 1); put_text(c, x - R//2, y - R//2, str(r.randint(1, 16)), int(R*.9), r)
    for _ in range(int(r.randint(0, 6) * n_scale)):  # marcas de conductores ///
        hash_marks(c, r.randint(0, W), r.randint(0, H), r, t)
    for _ in range(int(r.randint(0, 3) * n_scale)):  # flechas
        x, y = r.randint(0, W), r.randint(0, H); L = r.randint(30, 150)
        cv2.line(c, (x, y), (x, y + L), 0, t); cv2.line(c, (x-6, y+L-14), (x, y+L), 0, t); cv2.line(c, (x+6, y+L-14), (x, y+L), 0, t)
    if r.random() < .3 * n_scale:  # planilla de circuitos con texto vertical
        x0, y0 = r.randint(-50, W-100), r.randint(0, H-100); cw = r.randint(28, 60); nc = r.randint(4, 14)
        hs = [r.randint(12, 18) for _ in range(r.randint(3, 7))] + [r.randint(60, 150)]
        yy = y0
        for hh in hs:
            cv2.line(c, (x0, yy), (x0+nc*cw, yy), 0, 1)
            for j in range(nc):
                if hh < 30 and r.random() < .8: put_text(c, x0+j*cw+3, yy+2, r.choice(['TM C-60','2x16','4x25','R-N','LS0H','2x2,5+T','--','RES','0,72','S-N'])[:8], int(hh*.55), r)
                elif hh >= 30 and r.random() < .7:
                    for q in range(r.randint(1, 3)): put_text(c, x0+j*cw+3+q*int(txt_h), yy+4, r.choice(['ALIMENTACION','RESERVA','TOMAS USOS','GENERALES','UNIDAD CONDENSADORA','SIN EQUIPAR']), int(txt_h*.8), r, rot=True)
            yy += hh
        cv2.line(c, (x0, yy), (x0+nc*cw, yy), 0, 1)
        for j in range(nc+1): cv2.line(c, (x0+j*cw, y0), (x0+j*cw, yy), 0, 1)
    if r.random() < .2 * n_scale:  # tabla
        x0, y0 = r.randint(0, W-100), r.randint(0, H-60); cw, ch = r.randint(25, 60), r.randint(18, 30); nc, nr = r.randint(3, 10), r.randint(2, 6)
        for i in range(nr+1): cv2.line(c, (x0, y0+i*ch), (x0+nc*cw, y0+i*ch), 0, 1)
        for j in range(nc+1): cv2.line(c, (x0+j*cw, y0), (x0+j*cw, y0+nr*ch), 0, 1)
        for i in range(nr):
            for j in range(nc):
                if r.random() < .6: put_text(c, x0+j*cw+3, y0+i*ch+4, r.choice(WORDS)[:5], int(ch*.45), r)
    for _ in range(int(r.randint(0, 8) * n_scale)):  # puntos/nodos de union
        x, y = r.randint(0, W), r.randint(0, H)
        if r.random() < .5: cv2.circle(c, (x, y), r.randint(2, 4), 0, -1)
        elif r.random() < 0.20: cv2.circle(c, (x, y), r.randint(3, 6), 0, 1)

def sym_instance(lib, r, txt_h):
    proc = r.random() < .22
    gen = GENS[r.randrange(len(GENS))] if proc else None
    if proc: a = gen(r)
    else: a = lib.syms[r.randrange(len(lib.syms))]
    # 28/09: con P_ROTULO_NEG el rectangulo alto con texto vertical es NEGATIVO (Tomas); un `caja_nombre`
    # girado 90 grados es exactamente eso pero con caja. No se gira mientras ese negativo este activo.
    no_girar = gen is not None and gen.__name__ == 'caja_nombre' and float(os.environ.get('P_ROTULO_NEG', '0')) > 0
    a = a.copy()
    # tamano: lado mayor ~ 6.5 * altura de texto (rango amplio)
    # 19/09: el limite inferior baja de 1.8 a 1.15. El dataset tenia solo 13% de cajas de
    # menos de 20 px cuando en los planos reales son el 27-32%, asi que los simbolos chicos
    # estaban muy poco representados. El amperimetro de LU-UN-01 y nyw-un-01 mide 13 px a la
    # escala del detector: el modelo lo encuentra bien (IoU 0.82-0.93) pero con confianza
    # 0.06-0.17, y por eso se perdia. Eran 7 de los 10 falsos negativos que quedaban.
    # 20/09: REVERTIDO al valor del plan N. El plan O habia bajado el piso a 1.10 con un tramo
    # chico explicito (18% de las veces), con la hipotesis de que el amperimetro A$C45664E16 se
    # perdia por falta de ejemplos chicos. La hipotesis era FALSA y el remedio salio caro:
    # O duplico los falsos negativos (39 contra 19 de N, misma confianza y mismo GT), no
    # arreglo el amperimetro (21 FN contra 15) y encima rompio tres bloques que N detectaba
    # perfecto: PULS 1->7, A$C4C456AEF 0->5, TRAFOIN 0->3. Llenar el dataset de simbolos
    # diminutos degrado los medianos sin ganar nada en los chicos.
    # Este valor se reconstruyo midiendo ds9 (la base de N, que quedo en disco): sus tiles
    # sinteticos dan p50=45 px con 12.8% de cajas de menos de 20 px, y con 1.80 se reproduce
    # en 44 px y 11.4%. Con el tramo chico de O daba 37 px y 17.6%.
    lo, hi = 1.80, 12.0
    target = txt_h * np.exp(r.uniform(np.log(lo), np.log(hi)))
    h, w = a.shape; s = target / max(h, w)
    ink = np.clip((255.0 - a) / 225.0, 0, 1).astype(np.float32)
    if s < 1:
        k = max(1, int(round(.8 / s)))
        if k > 1: ink = cv2.dilate(ink, np.ones((k, k), np.uint8))
    ink = cv2.resize(ink, (max(3, int(w*s)), max(3, int(h*s))), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    a = (255 - np.clip(ink * 1.6, 0, 1) * 255).astype(np.uint8)
    # 28/09 (revision): tampoco se giran sprites apaisados (w > 1,5 h, p. ej. "Fuente 24 VCC 5A"): girados
    # quedan como un rotulo vertical, que ahora es negativo.
    if float(os.environ.get('P_ROTULO_NEG', '0')) > 0 and a.shape[1] > 1.5 * a.shape[0]: no_girar = True
    if r.random() < .12 and not no_girar: a = cv2.rotate(a, r.choice([cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE]))
    a = stroke_aug(a, r)
    bb = ink_bbox(a)
    if bb is None: return None
    a = a[bb[1]:bb[3], bb[0]:bb[2]]
    if min(a.shape) < 3:
        a = cv2.copyMakeBorder(a, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=255)
    return a

def cenir(c, b):
    """Ajusta la caja a la tinta que quedo realmente dibujada adentro.

    18/09. Los generadores que dibujan el simbolo a mano (`spm_branch`, `fusible_solo`,
    `tablero_row`) armaban la caja con margenes fijos en pixeles (-3, +4) calculados de la
    geometria, no de lo dibujado. En un simbolo de 20 px eso es un tercio de aire, y el modelo
    aprende la caja estirada: medido sobre 1200 tiles, el 43% de las cajas de `spm_branch` y
    el 44% de las de `fusible_solo` tenian mas de 18% de holgura, contra 0.4% en los sprites
    de biblioteca (que si se recortan a su tinta). Es el mismo defecto que el sprite del
    diferencial con cables, y se ve en el modelo: SPM y lampara de senalizacion son los dos
    peores IoU (0.42 y 0.46) con el alto predicho en 1.74x y 1.66x.

    Se ajusta DESPUES de dibujar, asi la caja termina en la tinta como en un sprite recortado.
    Si adentro no hay tinta (no deberia pasar) se devuelve la caja original.
    """
    x0, y0, x1, y1 = [int(v) for v in b[:4]]
    X0, Y0, X1, Y1 = max(0, x0), max(0, y0), min(S, x1), min(S, y1)
    if X1 - X0 < 3 or Y1 - Y0 < 3: return b
    ys, xs = np.where(c[Y0:Y1, X0:X1] < 160)
    if len(xs) == 0: return b
    return [X0 + int(xs.min()), Y0 + int(ys.min()), X0 + int(xs.max()) + 1, Y0 + int(ys.max()) + 1]

def visible(b):
    """28/09 (Tomas: "que se genere al menos el 90% del componente"): True si la parte de la caja que queda
    dentro del tile es >= VIS_MIN de su area (0,9 en ds21). Sin VIS_MIN no filtra (comportamiento viejo)."""
    vm = os.environ.get('VIS_SINT') or os.environ.get('VIS_MIN')   # 28/09: VIS_SINT solo para sinteticos
    if not vm: return True
    A = max(1e-9, (b[2] - b[0]) * (b[3] - b[1]))
    ix = max(0, min(S, b[2]) - max(0, b[0])); iy = max(0, min(S, b[3]) - max(0, b[1]))
    return ix * iy >= float(vm) * A


def overlap(b, boxes, m=6):
    for q in boxes:
        if b[0] < q[2]+m and b[2]+m > q[0] and b[1] < q[3]+m and b[3]+m > q[1]: return True
    return False

def real_bg(lib, r, txt_h):
    img, mask = lib.bgs[r.randrange(len(lib.bgs))]
    s = txt_h / 11.0 * r.uniform(.85, 1.2)
    H, W = img.shape; ch = int(S / s)
    if H < 50 or W < 50: return None
    y = r.randint(0, max(0, H - ch)); x = r.randint(0, max(0, W - ch))
    crop = img[y:y+ch, x:x+ch]
    if crop.size == 0: return None
    crop = cv2.resize(crop, (int(crop.shape[1]*s), int(crop.shape[0]*s)), interpolation=cv2.INTER_AREA)
    c = np.full((S, S), 255, np.uint8); c[:crop.shape[0], :crop.shape[1]] = crop[:S, :S]
    return c

def tablero_row(c, r, txt_h, xs, boxes):
    """Fila de borneras/selectoras/pulsadores chicos sobre el borde punteado del tablero, con texto pegado (estilo FL-UN / TSSS)."""
    if not xs: return
    y = r.randint(30, S - 30); t = r.choice([1, 1, 2])
    dashed_line(c, (0, y), (S, y), 1, r, dash=r.randint(8, 22))
    kind = r.choices(['born', 'sel', 'puls'], weights=[.6, .25, .15])[0]
    lab = r.choice(['X1', 'X2', 'X3', 'X4', 'XB', 'X'])
    for x in xs:
        if r.random() < .15: continue
        if kind == 'born' or (kind != 'born' and r.random() < .2):
            R = max(3, int(round(txt_h * r.uniform(.4, .75))))
            ang = np.deg2rad(r.uniform(20, 50)); L = R * r.uniform(1.3, 1.7)
            dx, dy = int(L*np.cos(ang)), int(L*np.sin(ang))
            b = [x-max(R, dx)-2, y-max(R, dy)-2, x+max(R, dx)+3, y+max(R, dy)+3]
            if overlap(b, boxes, 2) or not visible(b): continue
            if r.random() < .3: c[max(0,b[1]):b[3], max(0,b[0]):b[2]] = 255
            cv2.line(c, (x, b[1]-6), (x, b[3]+6), 0, t)
            cv2.circle(c, (x, y), R, 0, 1); cv2.line(c, (x-dx, y+dy), (x+dx, y-dy), 0, 1)
            if r.random() < .8: put_text(c, x - int(R*r.uniform(2.6, 4.0)), y - int(R*r.uniform(2.2, 3.0)), lab, int(txt_h*r.uniform(.9, 1.2)), r)
        elif kind == 'sel':
            L = int(txt_h * r.uniform(2.0, 3.2)); k = r.randint(2, 3)
            x0, y0 = x - int(L*.5), y - L
            b = [x0-2, y0-2, x+k+3, y+k+3]
            if overlap(b, boxes, 2) or not visible(b): continue
            if r.random() < .3: c[max(0,b[1]):b[3], max(0,b[0]):b[2]] = 255
            cv2.line(c, (x0, y0), (x, y), 0, t + (1 if r.random() < .5 else 0)); cv2.circle(c, (x, y), k, 0, 1)
            cv2.line(c, (x, y+k), (x, y+k+int(txt_h*3)), 0, t)
            if r.random() < .85: put_text(c, x0 - int(txt_h*.5), y0 - int(txt_h*1.3), r.choice(['M-0-A', 'S-M-0-A', 'M-0-A', '0-1', 'L-0-R']), int(txt_h*r.uniform(.8, 1.1)), r)
        else:
            h = int(txt_h * r.uniform(2.2, 3.2)); w = int(h * .6)
            b = [x - w - 2, y - h//2 - 2, x + 4, y + h//2 + 3]
            if overlap(b, boxes, 2) or not visible(b): continue
            if r.random() < .3: c[max(0,b[1]):b[3], max(0,b[0]):b[2]] = 255
            cv2.line(c, (x, b[1]-8), (x, y - h//2), 0, t); cv2.line(c, (x, y + h//2), (x, b[3]+8), 0, t)
            cv2.line(c, (x, y + h//2), (x - w//2, y - h//3), 0, t)
            dashed_line(c, (x - w//4, y), (x - w, y), 1, r, dash=3); cv2.line(c, (x - w, y - h//4), (x - w, y + h//4), 0, 1)
            if r.random() < .7: put_text(c, x + 4, y - h//2, r.choice(['SPM', 'SP', 'S1', 'PM', 'PE']), int(txt_h), r)
        boxes.append(cenir(c, b))

def spm_branch(c, r, txt_h, xs, boxes):
    """Rama de comando, en sus TRES variantes, cada una con su propia caja.

    17/09, segunda vuelta. La primera version generaba SIEMPRE fusible + ojo de buey juntos,
    y el modelo (H) dejo de reconocer el ojo de buey cuando aparece solo: en test1, que tiene
    11 ojos de buey y ningun fusible, perdio uno. Ahora se generan los tres casos que existen
    en los planos, para que la diferencia la haga el contexto y no la forma:
       - 'conjunto': fusible + ojo de buey juntos -> DOS cajas, una por componente
       - 'ojo'     : ojo de buey solo       -> su propia caja
       - 'fusible' : fusible solo           -> su propia caja
    El texto ('SPM/3P', 'R: 4A', 'LED 220Vca') siempre queda AFUERA de la caja.
    """
    if not xs: return
    y = r.randint(40, S - 70); t = r.choice([1, 1, 2])
    if r.random() < .5: dashed_line(c, (0, y), (S, y), 1, r, dash=r.randint(8, 22))
    else: cv2.line(c, (0, y), (S, y), 0, t)
    for x in xs:
        if r.random() < .3: continue
        modo = r.choices(['conjunto', 'ojo', 'fusible'], weights=[.45, .35, .20])[0]
        cy = y + int(txt_h * r.uniform(1.0, 2.4))
        R = max(3, int(txt_h * r.uniform(.55, 1.0)))
        L = txt_h * r.uniform(1.1, 2.0); w = L * r.uniform(.28, .42)
        # 18/09: el cartucho va DERECHO la mayor parte de las veces. Antes salia siempre
        # inclinado 35-55 grados, y al inclinarlo su caja se vuelve casi cuadrada; en los planos
        # de Tomas el SPM es un rectangulo horizontal de proporcion ~3:1 (medido en fl_un_02:
        # 0.65 x 0.21). Por eso el modelo le predecia 0.32 x 0.32 y el SPM quedaba con el peor
        # IoU de todos los componentes (0.436, con el ancho en 0.43x del real).
        rectito = r.random() < .6
        ang = 0.0 if rectito else np.deg2rad(r.uniform(35, 55) * (1 if r.random() < .5 else -1))
        ca, sa = np.cos(ang), np.sin(ang)

        def _pt(ux, uy, fx):
            return (int(fx + ux*ca - uy*sa), int(cy + ux*sa + uy*ca))

        def caja_fusible(fx):
            pts = [list(_pt(ux, uy, fx)) for ux, uy in
                   ((-L/2, -w/2), (L/2, -w/2), (L/2, w/2), (-L/2, w/2))]
            return np.array(pts, np.int32)

        def marca_fusible(fx):
            # la raya de adentro: en el cartucho derecho va de esquina a esquina, como se ve en
            # fl_un_02 y tsss_2; en el inclinado, a lo largo del eje.
            if rectito: cv2.line(c, _pt(-L/2, -w/2, fx), _pt(L/2, w/2, fx), 0, 1)
            else: cv2.line(c, _pt(-L/2, 0, fx), _pt(L/2, 0, fx), 0, 1)

        if modo == 'ojo':
            b = [x - R - 3, cy - R - 3, x + R + 4, cy + R + 4]
            if b[0] < 0 or b[1] < 0 or b[2] > S or b[3] > S or overlap(b, boxes, 2): continue
            if r.random() < .3: c[b[1]:b[3], b[0]:b[2]] = 255
            cv2.line(c, (x, y), (x, cy - R), 0, t)
            cv2.circle(c, (x, cy), R, 0, 1)
            dd = int(R * .7)
            cv2.line(c, (x-dd, cy-dd), (x+dd, cy+dd), 0, 1)
            cv2.line(c, (x-dd, cy+dd), (x+dd, cy-dd), 0, 1)
            boxes.append([x - R, cy - R, x + R + 1, cy + R + 1])
            if r.random() < .7:
                put_text(c, b[2] + 2, b[1], r.choice(['LED 220Vca', 'x3 220V', '220Vca', 'x1']), int(txt_h*r.uniform(.7, .95)), r)
            continue

        if modo == 'fusible':
            pts = caja_fusible(x)
            b = [int(pts[:, 0].min())-3, int(pts[:, 1].min())-3, int(pts[:, 0].max())+4, int(pts[:, 1].max())+4]
            if b[0] < 0 or b[1] < 0 or b[2] > S or b[3] > S or overlap(b, boxes, 2): continue
            if r.random() < .3: c[b[1]:b[3], b[0]:b[2]] = 255
            # el cable entra y sale POR EL BORDE DEL CARTUCHO, no por el borde de la caja:
            # si llega hasta b[1]/b[3] (que traen margen) su tinta queda adentro y al cenir
            # la caja se estira a lo alto, que es lo que aplastaba la proporcion del SPM.
            py0, py1 = int(pts[:, 1].min()), int(pts[:, 1].max())
            cv2.line(c, (x, y), (x, py0), 0, t)
            cv2.polylines(c, [pts], True, 0, 1)
            if r.random() < .75:
                marca_fusible(x)
            cv2.line(c, (x, py1), (x, min(S-1, py1 + int(txt_h*2))), 0, t)
            if r.random() < .6: hash_marks(c, x, min(S-5, b[3] + int(txt_h*1.3)), r, 1)
            boxes.append([int(pts[:, 0].min()), py0, int(pts[:, 0].max()) + 1, py1 + 1])
            if r.random() < .8:
                put_text(c, b[2] + 2, b[1], r.choice(['SPM/1P', 'SPM/3P', 'R: 4A', '4A']), int(txt_h*r.uniform(.75, 1.0)), r)
            continue

        # conjunto: fusible + ojo de buey en UNA caja
        izq = r.random() < .5
        pts = caja_fusible(x)
        fx0, fy0 = int(pts[:, 0].min()), int(pts[:, 1].min())
        fx1, fy1 = int(pts[:, 0].max()), int(pts[:, 1].max())
        sep = int(txt_h * r.uniform(2.0, 5.0))
        lx = (fx0 - sep - R) if izq else (fx1 + sep + R)
        ly = cy
        # DOS cajas: el fusible y el ojo de buey son componentes distintos (Tomas, 17/09),
        # aunque esten pegados en la misma rama de comando.
        bf = [fx0 - 3, fy0 - 3, fx1 + 4, fy1 + 4]
        bo = [lx - R - 3, ly - R - 3, lx + R + 4, ly + R + 4]
        b = [min(bf[0], bo[0]), min(bf[1], bo[1]), max(bf[2], bo[2]), max(bf[3], bo[3])]
        if b[0] < 0 or b[1] < 0 or b[2] > S or b[3] > S or overlap(b, boxes, 2): continue
        if r.random() < .3: c[b[1]:b[3], b[0]:b[2]] = 255
        cv2.line(c, (x, y), (x, fy0), 0, t)
        cv2.polylines(c, [pts], True, 0, 1)
        if r.random() < .75:
            marca_fusible(x)
        cv2.circle(c, (lx, ly), R, 0, 1)
        dd = int(R * .7)
        cv2.line(c, (lx-dd, ly-dd), (lx+dd, ly+dd), 0, 1)
        cv2.line(c, (lx-dd, ly+dd), (lx+dd, ly-dd), 0, 1)
        if izq: cv2.line(c, (lx + R, ly), (fx0, ly), 0, t)
        else:   cv2.line(c, (fx1, ly), (lx - R, ly), 0, t)
        if r.random() < .5:
            cv2.line(c, (x, fy1), (x, min(S-1, fy1 + int(txt_h*2))), 0, t)
            if r.random() < .6: hash_marks(c, x, min(S-5, fy1 + int(txt_h*1.3)), r, 1)
        # La etiqueta es el contorno exacto del simbolo, no `b` con su margen: el cable vertical
        # atraviesa la caja de arriba a abajo, asi que cenir() no puede achicarla en Y y los
        # 3-4 px de margen quedaban adentro. Medido: el cartucho salia 25x14 px en vez de 25x8,
        # o sea proporcion 1.8 en lugar de la 3.1 que tiene el SPM real.
        boxes.append([fx0, fy0, fx1 + 1, fy1 + 1])
        boxes.append([lx - R, ly - R, lx + R + 1, ly + R + 1])
        if r.random() < .85:
            put_text(c, b[0], b[3] + 2, r.choice(['SPM/1P', 'SPM/3P', 'SPM/2P', 'SPM']), int(txt_h*r.uniform(.75, 1.0)), r)
            if r.random() < .5:
                put_text(c, b[0], b[3] + 2 + int(txt_h*1.2), r.choice(['R: 4A', 'R:4A', '4A', '2A']), int(txt_h*r.uniform(.7, .95)), r)

def _cargar_fusibles_dxf():
    """Geometria de los fusibles que paso Tomas (data/fusibles_tomas/*.dxf), en unidades CAD.

    Devuelve [(segmentos, caja_cuerpo)]: `segmentos` son polilineas [(x,y),...] (lineas, polilineas y
    circulos discretizados) y `caja_cuerpo` = (x0,y0,x1,y1) del CUERPO del fusible, que es lo que se
    etiqueta (el GT no incluye los cables). Cuerpo, por archivo:
      Fusible.dxf   : las lineas no horizontales (rectangulo a 45 grados + el cable que lo atraviesa)
      Fusible_2.dxf : todo (rectangulo con circulo)
      Fusible_3.dxf : la polilinea cerrada de 4 vertices (el mismo dibujo que los de test1)
    """
    import ezdxf, glob
    out = []
    carpeta = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'fusibles_tomas')
    for f in sorted(glob.glob(os.path.join(carpeta, '*.dxf'))):
        nom = os.path.basename(f)
        segs, cuerpo = [], []
        for e in ezdxf.readfile(f).modelspace():
            t = e.dxftype()
            if t == 'LINE':
                pl = [(e.dxf.start.x, e.dxf.start.y), (e.dxf.end.x, e.dxf.end.y)]
            elif t == 'LWPOLYLINE':
                pl = [tuple(q) for q in e.get_points('xy')]
                nv = len(pl)                              # vertices antes de cerrar
                if e.closed:
                    pl.append(pl[0])
            elif t == 'CIRCLE':
                a = np.linspace(0, 2 * np.pi, 40)
                cx, cy, rr = e.dxf.center.x, e.dxf.center.y, e.dxf.radius
                pl = list(zip(cx + rr * np.cos(a), cy + rr * np.sin(a)))
            else:
                continue
            segs.append(pl)
            horiz = t == 'LINE' and abs(pl[0][1] - pl[1][1]) < 1e-6
            if (nom == 'Fusible.dxf' and not horiz) or nom == 'Fusible_2.dxf' or \
               (nom == 'Fusible_3.dxf' and t == 'LWPOLYLINE' and nv == 4):
                cuerpo += pl
        if segs and cuerpo:
            P = np.array(cuerpo)
            out.append((nom, segs, (P[:, 0].min(), P[:, 1].min(), P[:, 0].max(), P[:, 1].max())))
    return out


_FUSIBLES_DXF = None
_PESO_FUS = {'Fusible.dxf': 1.0, 'Fusible_2.dxf': .5, 'Fusible_3.dxf': 2.0}


def fusible_dxf(c, r, txt_h, boxes):
    """Fusible de los DXF de Tomas en una rama hacia la lampara piloto (26/09, pedido de Tomas).

    En test1 hay 5 fusibles "x3" (uno por tablero TSAsc4) y ningun modelo los detectaba: RF-DETR les da
    0,06-0,10 y los YOLO nada. No estaban en el GT (lineas sueltas, sin bloque): el 100% estaba inflado.
    Los sprites de Tomas (sym_lib_tomas/u_fusible_1..3) son ESTE fusible pero no enseñaban a detectarlo:
    traen los cables adentro (la etiqueta sale 84x28, alargada, y el GT es solo el cuerpo), sym_instance
    los escala a >= 1,8 alturas de texto (el real mide ~1,1) y en chico no se parecen al render real.
    Aca se usa la geometria EXACTA de sus DXF (Fusible_3 es el mismo dibujo que test1, de otra hoja del
    proyecto: x 2245 contra 1877-1938 de test1): se dibuja a 4x y se reduce con INTER_AREA, que da el gris
    moteado del render (lo que en test1 parece "rayado" es el cable que atraviesa el cuerpo). Caja = solo
    el cuerpo. Fusible y lampara llevan cajas separadas (regla de Tomas, 17/09).
    """
    global _FUSIBLES_DXF
    if _FUSIBLES_DXF is None:
        _FUSIBLES_DXF = _cargar_fusibles_dxf()
    if not _FUSIBLES_DXF:
        return
    K = 4
    for _ in range(r.randint(1, 3)):
        nom, segs, cb = r.choices(_FUSIBLES_DXF, weights=[_PESO_FUS.get(f[0], 1.0) for f in _FUSIBLES_DXF])[0]
        # escala: el cuerpo mide 0,9-1,3 alturas de texto en su lado mayor (test1: 1,15)
        esc = txt_h * r.uniform(.9, 1.3) / max(cb[2] - cb[0], cb[3] - cb[1])
        espejo = r.random() < .5
        vertical = r.random() < .2
        todos = np.array([q for pl in segs for q in pl])
        gx0, gx1, gy1 = todos[:, 0].min(), todos[:, 0].max(), todos[:, 1].max()

        def tr(x, y):                                   # CAD -> px relativos al sprite (y hacia abajo)
            u, v = (x - gx0) * esc, (gy1 - y) * esc
            if espejo:
                u = (gx1 - gx0) * esc - u
            return (v, u) if vertical else (u, v)

        uv = np.array([tr(*q) for q in todos])
        W, H = int(uv[:, 0].max()) + 3, int(uv[:, 1].max()) + 3
        pre, post = int(txt_h * r.uniform(1.0, 3.5)), int(txt_h * r.uniform(.8, 2.5))
        con_lamp = r.random() < .7
        R = max(3, int(txt_h * r.uniform(.65, 1.0)))
        largo = pre + (H if vertical else W) + post + (2 * R if con_lamp else 0) + 8
        ancho = int(2.4 * txt_h) + (W if vertical else H)
        if vertical:
            x0 = r.randint(ancho // 2 + 5, max(ancho // 2 + 6, S - ancho // 2 - 5)); y0 = r.randint(5, max(6, S - largo - 5))
            g = [x0 - ancho // 2, y0, x0 + ancho // 2, y0 + largo]
        else:
            x0 = r.randint(5, max(6, S - largo - 5)); y0 = r.randint(ancho // 2 + 5, max(ancho // 2 + 6, S - ancho // 2 - 5))
            g = [x0, y0 - ancho // 2, x0 + largo, y0 + ancho // 2]
        if g[0] < 0 or g[1] < 0 or g[2] > S or g[3] > S or overlap(g, boxes, 2):
            continue
        # sprite a 4x con los segmentos reales (incluye sus cables cortos, el codo y la patita)
        big = np.full((H * K, W * K), 255, np.uint8)
        for pl in segs:
            P = (np.array([tr(*q) for q in pl]) * K).astype(np.int32)
            cv2.polylines(big, [P], False, 0, max(1, int(round(K * r.uniform(.8, 1.3)))))
        spr = cv2.resize(big, (W, H), interpolation=cv2.INTER_AREA)
        ys, xs = np.where(spr < 200)
        if not len(xs):
            continue
        if vertical:
            sx, sy = x0 - W // 2, y0 + pre
        else:
            sx, sy = x0 + pre, y0 - H // 2
        if sy < 0 or sx < 0 or sy + H > S or sx + W > S:
            continue
        c[g[1]:g[3], g[0]:g[2]] = 255
        c[sy:sy + H, sx:sx + W] = np.minimum(c[sy:sy + H, sx:sx + W], spr)
        # extremos de los cables del sprite (tinta mas a la izquierda/derecha, o arriba/abajo)
        if vertical:
            a_in = (sx + int(xs[ys == ys.min()].mean()), sy + int(ys.min()))
            a_out = (sx + int(xs[ys == ys.max()].mean()), sy + int(ys.max()))
            ini, fin = (a_in[0], y0), (a_out[0], a_out[1] + post)
        else:
            a_in = (sx + int(xs.min()), sy + int(ys[xs == xs.min()].mean()))
            a_out = (sx + int(xs.max()), sy + int(ys[xs == xs.max()].mean()))
            ini, fin = (x0, a_in[1]), (a_out[0] + post, a_out[1])
        t = r.choice([1, 1, 2])
        if r.random() < .6:                              # cable del que sale la rama
            if vertical:
                cv2.line(c, (max(0, ini[0] - int(txt_h * 4)), ini[1]), (min(S - 1, ini[0] + int(txt_h * 4)), ini[1]), 0, t)
            else:
                cv2.line(c, (ini[0], max(0, ini[1] - int(txt_h * 4))), (ini[0], min(S - 1, ini[1] + int(txt_h * 4))), 0, t)
        if r.random() < .6:
            cv2.circle(c, ini, max(2, int(txt_h * .3)), 0, 1)
        cv2.line(c, ini, a_in, 0, t)
        cv2.line(c, a_out, fin, 0, 1 if r.random() < .5 else t)
        (u0, v0), (u1, v1) = tr(cb[0], cb[3]), tr(cb[2], cb[1])
        bx0, bx1 = sorted((u0, u1)); by0, by1 = sorted((v0, v1))
        boxes.append([int(sx + bx0), int(sy + by0), int(sx + bx1) + 1, int(sy + by1) + 1])
        if con_lamp:
            lx, ly = (fin[0], fin[1] + R) if vertical else (fin[0] + R, fin[1])
            cv2.circle(c, (lx, ly), R, 0, 1)
            dd = int(R * .7)
            cv2.line(c, (lx - dd, ly - dd), (lx + dd, ly + dd), 0, 1)
            cv2.line(c, (lx - dd, ly + dd), (lx + dd, ly - dd), 0, 1)
            boxes.append([lx - R, ly - R, lx + R + 1, ly + R + 1])
            if r.random() < .5:
                put_text(c, lx - R, ly + R + 2, r.choice(['x3', 'x1', 'x3', '3x']), int(txt_h * r.uniform(.7, .95)), r)
        if r.random() < .7:
            put_text(c, int(sx + bx0) - int(txt_h * .3), int(sy + by0) - int(txt_h * 1.1),
                     r.choice(['x3', 'x3', 'x1', '3x', '2A']), int(txt_h * r.uniform(.7, .95)), r)


def fusible_solo(c, r, txt_h, xs, boxes):
    """Fusible suelto sobre un cable: cartucho inclinado, recto o en caja.

    Tomas pidio mas enfasis en los fusibles (17/09): era el simbolo que peor se detectaba.
    """
    if not xs: return
    for x in r.sample(xs, min(len(xs), r.randint(1, 3))):
        y = r.randint(40, S - 40)
        t = r.choice([1, 1, 2])
        L = txt_h * r.uniform(1.2, 2.6); w = L * r.uniform(.28, .5)
        modo = r.choice(['inclinado', 'recto', 'caja'])
        ang = np.deg2rad(r.uniform(30, 60) * (1 if r.random() < .5 else -1)) if modo == 'inclinado' else np.deg2rad(90 if modo == 'recto' else 0)
        ca, sa = np.cos(ang), np.sin(ang)
        pts = []
        for ux, uy in ((-L/2, -w/2), (L/2, -w/2), (L/2, w/2), (-L/2, w/2)):
            pts.append([int(x + ux*ca - uy*sa), int(y + ux*sa + uy*ca)])
        pts = np.array(pts, np.int32)
        b = [int(pts[:, 0].min())-3, int(pts[:, 1].min())-3, int(pts[:, 0].max())+4, int(pts[:, 1].max())+4]
        if b[0] < 0 or b[1] < 0 or b[2] > S or b[3] > S or overlap(b, boxes, 3): continue
        cv2.line(c, (x, max(0, b[1]-int(txt_h*2))), (x, b[1]), 0, t)
        cv2.line(c, (x, b[3]), (x, min(S-1, b[3]+int(txt_h*2))), 0, t)
        cv2.polylines(c, [pts], True, 0, 1)
        if modo == 'caja' and r.random() < .6:
            cv2.line(c, (int(x - L/2*ca), int(y - L/2*sa)), (int(x + L/2*ca), int(y + L/2*sa)), 0, 1)
        boxes.append(cenir(c, b))
        if r.random() < .7:
            put_text(c, b[2] + 2, b[1], r.choice(['4A', '6A', '2A', 'gG 25A', 'NH 63A', 'F1/2/3']), int(txt_h*r.uniform(.7, 1.0)), r)

def compuesto_vertical(c, r, txt_h, xs, boxes):
    """Aparato de dos piezas separadas: palanca arriba + toroidal abajo = UNA caja.

    Es el diferencial de los planos reales. El modelo lo parte en dos porque en el dataset
    casi todos los simbolos son sprites compactos de una pieza; nunca vio que dos formas
    separadas por un hueco son un solo aparato. A veces lleva la 'S' de super inmunizado
    entre las dos piezas, que hoy baja el IoU de 0.68 a 0.42.
    """
    if not xs: return
    for x in r.sample(xs, min(len(xs), r.randint(1, 3))):
        t = r.choice([1, 1, 2])
        H = int(txt_h * r.uniform(2.6, 4.2))
        y0 = r.randint(20, max(21, S - H - 30))
        wpal = int(txt_h * r.uniform(.8, 1.4))
        ytop = y0 + int(H * .25)
        cv2.line(c, (x, max(0, y0 - int(txt_h))), (x, y0), 0, t)
        cv2.line(c, (x - wpal//3, ytop), (x + wpal//2, y0), 0, max(2, t+1))
        cv2.circle(c, (x, y0), max(2, t), 0, 1)
        cv2.circle(c, (x, ytop), max(2, t), 0, 1)
        yov = y0 + int(H * .78)
        rx = int(txt_h * r.uniform(.7, 1.2)); ry = max(2, int(rx * r.uniform(.25, .45)))
        cv2.line(c, (x, ytop), (x, yov - ry), 0, t)
        cv2.ellipse(c, (x, yov), (rx, ry), 0, 0, 360, 0, 1)
        cv2.line(c, (x, yov + ry), (x, min(S-1, yov + ry + int(txt_h*1.5))), 0, t)
        b = [x - max(wpal, rx) - 4, y0 - 4, x + max(wpal, rx) + 5, yov + ry + 4]
        if b[0] < 0 or b[1] < 0 or b[2] > S or b[3] > S or overlap(b, boxes, 3): continue
        if r.random() < .45:
            ss = int(txt_h * .9)
            sx = x + max(wpal, rx) - int(txt_h*.2); sy = (ytop + yov)//2
            if sx + ss + 3 < S:
                cv2.rectangle(c, (sx, sy - ss//2), (sx + ss, sy + ss//2), 0, 1)
                put_text(c, sx + 1, sy - ss//2 + 1, 'S', int(ss*.8), r)
                b[2] = max(b[2], sx + ss + 3)
        boxes.append(b)
        if r.random() < .8:
            put_text(c, b[2] + 3, b[1] + int(H*.3), r.choice(['25A', '2x25A', '4x40A']), int(txt_h*.9), r)
            put_text(c, b[2] + 3, b[1] + int(H*.3) + int(txt_h*1.2), r.choice(['30mA', '300mA']), int(txt_h*.85), r)

def polo_suelto(c, r, txt_h, xs):
    """Palanca sola sobre un cable, SIN el resto del aparato. NEGATIVO (no agrega caja).

    Pedido de Tomas (17/09): el modelo marca muchos 'polos' sueltos como componentes.
    Un polo DENTRO de un aparato es parte del positivo; solo, no lo es.
    """
    if not xs: return
    for x in r.sample(xs, min(len(xs), r.randint(1, 3))):
        y = r.randint(40, S - 40); t = r.choice([1, 1, 2])
        h = int(txt_h * r.uniform(1.0, 1.8)); w = int(h * r.uniform(.5, .9))
        cv2.line(c, (x, max(0, y - h - int(txt_h))), (x, y - h), 0, t)
        cv2.line(c, (x - w//3, y), (x + w//2, y - h), 0, max(2, t+1))
        cv2.circle(c, (x, y - h), max(2, t), 0, 1)
        cv2.circle(c, (x, y), max(2, t), 0, 1)
        cv2.line(c, (x, y), (x, min(S-1, y + int(txt_h*2))), 0, t)

def barra_colectora(c, r, txt_h):
    """Barra colectora: trazo NEGRO GRUESO (relleno). Es NEGATIVO.

    En los planos del cliente (IE-UNI-01..04) el modelo D dispara una cadena de falsos
    positivos a lo largo de estas barras. Se agregan como fondo, sin caja, para que aprenda
    que una banda negra maciza no es un componente.
    """
    grosor = max(3, int(txt_h * r.uniform(.5, 1.6)))
    if r.random() < .5:      # vertical
        x = r.randint(20, S-20); y0, y1 = r.randint(-40, S//3), r.randint(2*S//3, S+40)
        cv2.rectangle(c, (x - grosor//2, y0), (x + grosor//2, y1), 0, -1)
        for _ in range(r.randint(0, 5)):   # derivaciones que salen de la barra
            yy = r.randint(max(0, y0), min(S, y1))
            cv2.line(c, (x, yy), (x + r.choice([-1, 1]) * r.randint(30, 200), yy), 0, r.choice([1, 1, 2]))
    else:                    # horizontal
        y = r.randint(20, S-20); x0, x1 = r.randint(-40, S//3), r.randint(2*S//3, S+40)
        cv2.rectangle(c, (x0, y - grosor//2), (x1, y + grosor//2), 0, -1)
        for _ in range(r.randint(0, 5)):
            xx = r.randint(max(0, x0), min(S, x1))
            cv2.line(c, (xx, y), (xx, y + r.choice([-1, 1]) * r.randint(30, 200)), 0, r.choice([1, 1, 2]))

def terna_rst(c, r, txt_h, lib, boxes):
    """Terna R/S/T: dos o tres columnas iguales, cada una con su simbolo y su caja.

    19/09. Un revisor de vision audito las 2106 detecciones sobre los PDF de Marcelo. En los
    planos bien orientados quedaban 196 componentes sin detectar y el 73% eran fusibles (92)
    y lamparas (51); al mismo tiempo, los casos donde una caja abarcaba varios componentes
    eran justo "tres fusibles + tres lamparas R/S/T adentro de una sola caja". Es el mismo
    problema contado de dos formas: en la terna el detector saca UNA caja para las tres
    columnas, que cuenta como una acertada y dos perdidas.

    Se diferencia de `fila_densa` en que copia la forma real del patron: pocas columnas
    (2-4), todas del MISMO simbolo, colgando cada una de su propio cable vertical desde una
    barra comun, con la letra de fase debajo y, muy seguido, un segundo simbolo mas abajo
    (arriba el fusible, abajo la lampara). Cada simbolo lleva su caja: una terna de dos pisos
    son SEIS cajas, no una.
    """
    prev = [list(b[:4]) for b in boxes]
    arriba = sym_instance(lib, r, txt_h)
    if arriba is None: return 0
    h, w = arriba.shape
    if w < 6 or h < 6 or w > S // 5: return 0
    # 28/09 (Tomas, grilla G 4-6): tres sprites muy alargados en fila (dos bobinas sobre un cable) parecen UN
    # componente cortado, no tres. En la terna no se usan simbolos con relacion de lados > 3,5.
    if max(w, h) > 3.5 * min(w, h): return 0
    dos_pisos = r.random() < .65
    abajo = sym_instance(lib, r, txt_h) if dos_pisos else None
    if abajo is not None and (abajo.shape[0] < 6 or abajo.shape[1] > S // 5 or max(abajo.shape) > 3.5 * min(abajo.shape)):
        abajo = None
    n = r.choice([3, 3, 3, 2, 4])
    # 28/09 (Tomas): la separacion salia del ancho del simbolo de ARRIBA; si el de abajo era mas ancho
    # (lamparas debajo de fusibles) los de abajo quedaban encimados y no se veia nada. Ahora se usa el
    # mas ancho de los dos, con un espacio libre de al menos media altura de texto entre columnas.
    wmax = max(w, abajo.shape[1] if abajo is not None else 0)
    sep = max(wmax + max(3, int(txt_h * .5)), int(wmax * r.uniform(1.25, 2.4)))   # separadas, no pegadas
    # 28/09 (revision): con 4 columnas anchas la terna no entraba y las que se salian quedaban como
    # positivos recortados o cajas degeneradas en x=640. Se sacan columnas hasta que entre.
    while n > 2 and sep * (n - 1) + wmax + 12 > S - 8:
        n -= 1
    if sep * (n - 1) + wmax + 12 > S - 8:
        return 0
    hb = abajo.shape[0] if abajo is not None else 0
    tramo = int(txt_h * r.uniform(1.2, 2.6))              # cable entre los dos pisos
    alto = h + (tramo + hb if abajo is not None else 0) + int(txt_h * 2.2)
    ancho = sep * (n - 1) + wmax
    for _ in range(8):
        x = r.randint(4, max(5, S - ancho - 4))
        y = r.randint(int(txt_h * 2), max(int(txt_h * 2) + 1, S - alto - 4))
        ext = (wmax - w) // 2 + 2          # el simbolo de abajo va centrado y puede ser mas ancho
        if not overlap([x - ext, y - int(txt_h * 1.6), x + sep * (n - 1) + w + ext, y + alto], prev, 2): break
    else:
        return 0
    ybarra = y - int(txt_h * r.uniform(.8, 1.5))
    if r.random() < .8:
        cv2.line(c, (max(0, x - int(w * .8)), ybarra), (min(S, x + ancho + int(w * .8)), ybarra),
                 0, r.choice([1, 2]))
    fases = ['R', 'S', 'T', 'N'][:n] if r.random() < .8 else [str(i + 1) for i in range(n)]
    puestos = 0
    for k in range(n):
        xk = x + k * sep
        cx = xk + w // 2
        cv2.line(c, (cx, ybarra), (cx, y), 0, 1)                      # baja de la barra
        if overlap([xk, y, xk + w, y + h], prev, 2) or not visible([xk, y, xk + w, y + h]): continue
        blit_min(c, arriba, xk, y)
        boxes.append([xk, y, min(S, xk + w), min(S, y + h)])
        puestos += 1
        yy = y + h
        if abajo is not None:
            cv2.line(c, (cx, yy), (cx, yy + tramo), 0, 1)
            wb, hb2 = abajo.shape[1], abajo.shape[0]
            xb = cx - wb // 2
            if not overlap([xb, yy + tramo, xb + wb, yy + tramo + hb2], prev, 2) and visible([xb, yy + tramo, xb + wb, yy + tramo + hb2]):
                blit_min(c, abajo, xb, yy + tramo)
                boxes.append([max(0, xb), yy + tramo, min(S, xb + wb), min(S, yy + tramo + hb2)])
                puestos += 1
            yy = yy + tramo + hb2
        if r.random() < .85:      # la letra de fase, siempre AFUERA de las cajas
            put_text(c, cx - int(txt_h * .3), yy + 2, fases[k], int(txt_h * r.uniform(.8, 1.1)), r)
    return puestos

def puls_contactor(c, r, txt_h, lib, boxes):
    """Contactor con su pulsador al lado: DOS componentes, DOS cajas.

    20/09. Medido sobre LU-UN-01 y nyw-un-01 con el GT v4: el plan Q no "pierde" los `PULS`,
    se los TRAGA. De los 25 pulsadores del GT, Q saca caja propia para 6 y mete los otros 18
    adentro de la caja del contactor; N saca 21 propias. Es el defecto que mas complica la
    etapa 2 (una caja con dos componentes no se puede clasificar) y es justo lo que Tomas
    pidio arreglar desde el principio.

    En el plano el par es siempre igual: la bobina (rectangulo con diagonal) con su contacto
    cruzandola, y abajo a la izquierda un triangulo con un punto adentro unido por una linea
    corta. Son dos bloques INSERT distintos en el DXF, no uno.

    El generador dibuja exactamente esa configuracion y etiqueta los dos por separado, para
    que el modelo vea la separacion en el caso concreto en el que la esta perdiendo.
    """
    prev = [list(b[:4]) for b in boxes]
    t = r.choice([1, 1, 2])
    esc = txt_h / 11.0
    # bobina + contacto (el contactor)
    bw = int(r.uniform(2.0, 3.2) * txt_h); bh = int(r.uniform(1.6, 2.6) * txt_h)
    # pulsador: triangulo con punto
    pw = int(r.uniform(1.4, 2.2) * txt_h); ph = int(pw * r.uniform(.75, 1.0))
    sep_x = int(txt_h * r.uniform(.8, 2.0))
    sep_y = int(txt_h * r.uniform(.6, 1.8))
    ancho = bw + sep_x + pw + int(txt_h * 3)
    alto = bh + sep_y + ph + int(txt_h * 2)
    for _ in range(8):
        x = r.randint(4, max(5, S - ancho - 4))
        y = r.randint(4, max(5, S - alto - 4))
        if not overlap([x - 2, y - 2, x + ancho + 2, y + alto + 2], prev, 2):
            break
    else:
        return 0
    # ---- contactor
    bx, by = x + pw + sep_x, y
    cv2.rectangle(c, (bx, by), (bx + bw, by + bh), 0, t)
    cv2.line(c, (bx, by + bh), (bx + bw, by), 0, t)                 # la diagonal de la bobina
    cv2.line(c, (bx + bw // 2, max(0, by - int(txt_h * 1.2))), (bx + bw // 2, by), 0, t)
    cv2.line(c, (bx + bw // 2, by + bh), (bx + bw // 2, min(S - 1, by + bh + int(txt_h * 1.2))), 0, t)
    caja_c = cenir(c, [bx, by, bx + bw, by + bh]) if 'cenir' in globals() else [bx, by, bx + bw, by + bh]
    boxes.append([max(0, caja_c[0]), max(0, caja_c[1]), min(S, caja_c[2]), min(S, caja_c[3])])
    # ---- pulsador (triangulo con punto)
    px, py = x, y + bh + sep_y - ph
    py = max(0, py)
    tri = np.array([[px + pw // 2, py], [px, py + ph], [px + pw, py + ph]], np.int32)
    cv2.polylines(c, [tri], True, 0, t)
    cv2.circle(c, (px + pw // 2, py + int(ph * .62)), max(1, int(ph * .10)), 0, -1)
    cv2.line(c, (px + pw, py + int(ph * .62)), (bx, py + int(ph * .62)), 0, 1)   # union al contactor
    boxes.append([max(0, px), max(0, py), min(S, px + pw), min(S, py + ph)])
    # ---- el texto del contactor, SIEMPRE afuera de las dos cajas
    if r.random() < .85:
        for i, lin in enumerate(r.choice([['12A', 'AC3', 'S1'], ['18A', 'AC3', 'S1'],
                                          ['16A', 'CT', '2NA'], ['40A', 'AC3', 'NC']])):
            put_text(c, bx + bw + int(txt_h * .4), by + int(txt_h * (.9 + 1.1 * i)),
                     lin, int(txt_h * r.uniform(.8, 1.05)), r)
    if r.random() < .5:
        put_text(c, max(0, px - int(txt_h * 1.2)), py + ph + int(txt_h * .9), 'A',
                 int(txt_h * r.uniform(.8, 1.1)), r)
    return 2


def fila_densa(c, r, txt_h, lib, boxes):
    """Fila de simbolos PEGADOS, iguales o distintos, cada uno con su caja propia.

    Hallazgo del 16/09 sobre EZE4077 (plano real del cliente): cuando los simbolos estan
    tan juntos que se tocan o se solapan, el modelo saca UNA caja que abarca 2 o 3 de ellos
    en vez de una por simbolo (24 cajas asi a conf 0.20). Como el evaluador asigna 1-a-1,
    los sobrantes cuentan como falsos negativos: es el techo real del recall en planos densos.
    Aca se generan filas apretadas etiquetando cada simbolo por separado para que aprenda a
    separarlos.

    18/09, dos cambios pedidos por Tomas, porque la etapa 2 de clasificacion se ensucia con
    las cajas que abarcan dos componentes:
      - La fila ya no se encima con lo que dibujaron los generadores anteriores. Antes se
        salteaba el chequeo de solapamiento por completo y terminaba pisando las cajas de
        `spm_branch` y `tablero_row`: de ahi salian 53 de los 58 pares de etiquetas anidadas
        medidos sobre 1500 tiles. El salteo se mantiene SOLO entre los simbolos de la propia
        fila, que es su razon de ser.
      - La fila puede ser de simbolos distintos, no solo iguales: el caso "dos componentes
        diferentes pegados" tambien tiene que quedar como dos cajas.
    """
    prev = [list(b[:4]) for b in boxes]          # lo que ya esta dibujado: no se pisa
    a = sym_instance(lib, r, txt_h)
    if a is None: return 0
    h, w = a.shape
    if w < 10 or h < 10: return 0   # con simbolos diminutos la fila queda un amasijo ilegible
    mezclada = r.random() < .35     # fila de simbolos distintos
    n = r.randint(3, 8)
    # 28/09 (Tomas: "se pegan todos y no se ve nada"): antes el paso era 0,75-1,05 del ancho del PRIMER
    # simbolo, o sea solapados hasta 25%, y en las filas mezcladas el vecino podia ser el doble de ancho
    # con el mismo paso. Ahora cada simbolo arranca donde termina el anterior mas un hueco de 0 a 20% de
    # su ancho: se tocan pero no se pisan (sigue siendo la fila apretada de EZE4077).
    # 28/09, segunda y tercera vuelta (Tomas): con hueco 0 y despues con 3-4 px se seguian viendo pegados.
    # Hueco = max(5 px, 0,6 alturas de texto, 30-60% del ancho).
    paso = max(3, int(w * 1.6))                       # solo para estimar el largo de la fila
    # se busca una franja libre; si en 8 intentos no hay lugar, la fila no se dibuja
    for _ in range(8):
        y = r.randint(10, max(11, S - h - 10))
        x = r.randint(0, max(1, S - paso * n))
        if not overlap([x, y, x + paso * n + w, y + h], prev, 2): break
    else:
        return 0
    eje = y + h // 2
    if r.random() < .6:
        cv2.line(c, (0, eje), (S, eje), 0, r.choice([1, 1, 2]))     # la barra de la que cuelgan
    puestos = 0
    xk_sig = x
    for k in range(n):
        ak, hk, wk = a, h, w
        if mezclada and k:
            otro = sym_instance(lib, r, txt_h)
            if otro is not None and otro.shape[0] >= 10 and otro.shape[1] >= 10 and otro.shape[1] <= 2 * w:
                ak = otro; hk, wk = otro.shape
        xk = xk_sig
        xk_sig = xk + wk + max(5, int(txt_h * .6), int(wk * r.uniform(.3, .6)))   # 28/09: siempre con aire (Tomas)
        if xk + wk > S: break
        yk = y + (h - hk) // 2
        x0, y0, x1, y1 = max(0, xk), max(0, yk), min(S, xk + wk), min(S, yk + hk)
        # 28/09 (revision): misma regla de visibilidad que el resto (VIS_MIN de area); antes 0,6 por lado
        if (x1 - x0) * (y1 - y0) < max(.36, float(os.environ.get('VIS_SINT') or os.environ.get('VIS_MIN') or 0)) * wk * hk: continue
        if overlap([x0, y0, x1, y1], prev, 2): continue      # no se pisa con lo anterior
        c[y0:y1, x0:x1] = 255     # 28/09: que ningun cable de la grilla atraviese el simbolo por el medio
        blit_min(c, ak, xk, yk)
        boxes.append([x0, y0, x1, y1])
        puestos += 1
        if r.random() < .5:     # sin texto al costado: caeria encima del vecino
            cv2.line(c, (xk + wk // 2, yk + hk), (xk + wk // 2, min(S, yk + hk + int(txt_h * 3))), 0, 1)
    return puestos

def trafo_o_pareja(c, r, txt_h, boxes):
    """Dos circulos que se tocan: o es UN transformador, o son DOS interruptores contiguos.

    Idea de Tomas (16/09): el modelo fusiona de a dos porque dos circulos solapados se
    parecen al simbolo del transformador. Aca se generan los dos casos explicitamente para
    que aprenda a distinguirlos por el contexto:
      - trafo    -> dos circulos solapados, SIN palanca ni derivacion propia = UNA caja
      - pareja   -> dos circulos, cada uno con su palanca y su cable vertical = DOS cajas
    """
    R = max(4, int(txt_h * r.uniform(.7, 1.3)))
    es_trafo = r.random() < .45
    # 28/09: el transformador SI son dos circulos solapados; la pareja de interruptores iba igual de
    # encimada y no se veia nada (Tomas). Ahora la pareja queda tangente o apenas separada.
    sep = int(R * r.uniform(1.0, 1.7)) if es_trafo else int(R * r.uniform(2.4, 3.0)) + 5
    # 18/09: antes elegia la posicion sin mirar lo ya dibujado y se encimaba con las filas
    # densas y las ramas de comando, dejando una etiqueta adentro de la otra.
    for _ in range(8):
        x = r.randint(R + 8, S - 2 * R - sep - 8)
        y = r.randint(R + 20, S - R - 30)
        if not overlap([x - R - int(txt_h * 2) - 2, y - R - int(txt_h * 2) - 2,
                        x + sep + R + int(txt_h * 2) + 3, y + R + int(txt_h * 2) + 3], boxes, 2): break
    else:
        return
    t = r.choice([1, 1, 2])
    c1, c2 = (x, y), (x + sep, y)
    cv2.circle(c, c1, R, 0, t); cv2.circle(c, c2, R, 0, t)
    if es_trafo:
        # entra por un lado y sale por el otro: es un solo aparato
        cv2.line(c, (c1[0] - R - int(txt_h * 2), y), (c1[0] - R, y), 0, t)
        cv2.line(c, (c2[0] + R, y), (c2[0] + R + int(txt_h * 2), y), 0, t)
        b = [c1[0] - R - 2, y - R - 2, c2[0] + R + 3, y + R + 3]
        boxes.append(b)
        if r.random() < .7:
            put_text(c, b[0], b[3] + 2, r.choice(['TRAFO', 'T1', '220/12Vca', 'TI', 'TC']), int(txt_h * .9), r)
    else:
        # cada uno con su palanca y su bajada: son dos aparatos
        for cc in (c1, c2):
            cv2.line(c, (cc[0], cc[1] - R - int(txt_h * 2)), (cc[0], cc[1] - R), 0, t)
            cv2.line(c, (cc[0], cc[1] + R), (cc[0], cc[1] + R + int(txt_h * 2)), 0, t)
            d = int(R * .8)
            cv2.line(c, (cc[0] - d // 2, cc[1] + d), (cc[0] + d, cc[1] - d), 0, t)   # palanca
            boxes.append([cc[0] - R - 2, cc[1] - R - 2, cc[0] + R + 3, cc[1] + R + 3])
            if r.random() < .6:   # 28/09 (revision): al costado se metia en la caja del vecino; va abajo
                put_text(c, cc[0] + 3, cc[1] + R + 6, r.choice(['10A', '16A', '25A', '2x10A']), int(txt_h * .85), r)

def marco_punteado(c, r, txt_h, boxes):
    """Marco de linea PUNTEADA, GRANDE, con el nombre de un equipo adentro. ES un componente.

    21/09. `procsym.caja_punteada` no alcanzo: pasa por `sym_instance`, que escala el sprite a
    ~6,5 alturas de texto, o sea al tamano de un componente comun. El marco real del
    `PLC / LOGICAS: / - TRANSF AUT / - ACOPLE DE BARRAS` de nyw-un-01 mide 2,778 x 1,320
    unidades CAD cuando el componente tipico de ese plano mide 0,913: es TRES VECES mas ancho.
    El modelo nunca vio uno asi, y por eso ni R ni S tienen una sola deteccion encima.

    Este generador lo dibuja a nivel de tile, con el tamano y la proporcion reales (entre 2 y
    4 veces el componente tipico, relacion de lados cerca de 2:1) y con varias lineas de texto
    adentro, una de ellas centrada arriba como titulo.
    """
    prev = [list(b[:4]) for b in boxes]
    w = int(txt_h * r.uniform(9.0, 16.0))
    h = int(w / r.uniform(1.6, 2.6))
    if w > S - 20 or h > S - 20:
        return 0
    for _ in range(8):
        x = r.randint(4, max(5, S - w - 4)); y = r.randint(4, max(5, S - h - 4))
        if not overlap([x - 2, y - 2, x + w + 2, y + h + 2], prev, 2):
            break
    else:
        return 0
    # 28/09 (revision visual): el marco quedaba atravesado por los cables de la grilla; un equipo
    # (TTA, PLC, banco de capacitores) no tiene cables pasando por adentro.
    c[max(0, y - 1):y + h + 2, max(0, x - 1):x + w + 2] = 255
    t = 1
    paso = max(4, int(txt_h * r.uniform(.5, 1.0))); trazo = max(2, int(paso * r.uniform(.45, .7)))
    for xx in range(x, x + w, paso):
        cv2.line(c, (xx, y), (min(x + w, xx + trazo), y), 0, t)
        cv2.line(c, (xx, y + h), (min(x + w, xx + trazo), y + h), 0, t)
    for yy in range(y, y + h, paso):
        cv2.line(c, (x, yy), (x, min(y + h, yy + trazo)), 0, t)
        cv2.line(c, (x + w, yy), (x + w, min(y + h, yy + trazo)), 0, t)
    titulo, cuerpo = r.choice([
        ('PLC', ['LOGICAS:', '- TRANSF AUT', '- ACOPLE DE BARRAS']),
        ('PLC', ['COMANDO', '- ARRANQUE', '- PARADA']),
        ('UPS', ['10 kVA', 'AUTONOMIA 15 min']),
        ('TTA', ['TRANSFERENCIA', 'AUTOMATICA']),
        ('CONTROL', ['DE ILUMINACION', '- HORARIO']),
        ('BANCO', ['DE CAPACITORES', '- 5 PASOS']),
    ])
    # el texto tiene que entrar ADENTRO del marco: si no, la caja no encierra lo que dice
    # encerrar y el modelo aprende un encuadre mal.
    put_text(c, x + w // 2 - int(len(titulo) * txt_h * .3), y + int(txt_h * 1.1), titulo, int(txt_h), r)
    caben = max(0, int((h - txt_h * 2.2) / (txt_h * 1.25)))
    for i, lin in enumerate(cuerpo[:caben]):
        if x + int(txt_h * .5) + int(len(lin) * txt_h * .6) > x + w - 2:
            lin = lin[:max(1, int((w - txt_h) / (txt_h * .6)))]
        put_text(c, x + int(txt_h * .5), y + int(txt_h * (2.5 + 1.25 * i)), lin, int(txt_h * .9), r)
    boxes.append([x, y, min(S, x + w), min(S, y + h)])
    return 1


def pat_negativo(c, r, txt_h):
    """Puestas a tierra dibujadas SIN caja. NEGATIVO.

    21/09 (Tomas): "el PAT no es un componente". Es el falso positivo mas confiado que queda:
    en test1, 8 de los 9 FP por encima de 0,9 son PAT, todos a 0,93-0,94.

    El sprite ya estaba fuera de la biblioteca (EXCLUIR_SIM) y `procsym.pat` no esta en GENS
    desde el plan F, asi que el modelo no lo aprende de ahi: lo aprende de las PSEUDO-ETIQUETAS.
    Se generaron con el modelo L a conf >= 0,80 y L ya detectaba el PAT con 0,94, con lo cual
    entro como etiqueta positiva y se refuerza solo en cada generacion. Filtrarlo de las
    pseudo-etiquetas requeriria un clasificador; el negativo explicito es directo y es el mismo
    patron que ya se uso para las celdas de planilla (`tabla_celdas`).

    Se dibuja como en el plano: la bajada vertical, muchas veces con un punto de union gordo
    arriba, y las tres rayas horizontales decrecientes. A veces con la etiqueta PAT o PE al lado.
    """
    for _ in range(r.randint(1, 3)):
        h = int(txt_h * r.uniform(1.6, 3.2)); w = int(h * r.uniform(.6, 1.0))
        x = r.randint(8, max(9, S - w - 8)); y = r.randint(8, max(9, S - h - 8))
        t = r.choice([1, 1, 2])
        cx = x + w // 2; ymed = y + h // 2
        cv2.line(c, (cx, y), (cx, ymed), 0, t)
        if r.random() < .55:
            cv2.circle(c, (cx, y + max(2, h // 6)), max(2, int(h * .09)), 0, -1)
        for k, f in enumerate([1.0, .66, .33]):
            L = int(w * .45 * f); yy = ymed + k * max(2, h // 7)
            cv2.line(c, (cx - L, yy), (cx + L, yy), 0, t)
        if r.random() < .5:
            put_text(c, cx + int(w * .6), y + int(txt_h * .9),
                     r.choice(['PAT', 'PE', 'TT']), int(txt_h * r.uniform(.8, 1.1)), r)


def libre_de(b, evitar, m):
    """True si la caja b (con margen m) no toca ninguna de `evitar`."""
    return not overlap(b, [q[:4] for q in evitar], m)


def tabla_celdas(c, r, txt_h, evitar=()):
    """Grilla de celdas chicas y cuadradas con texto corto adentro. NEGATIVO.

    En LU-UN-01 el modelo detecta las celdas de las tablas 'FUNCION / DESTINO' como si
    fueran cajas con letras (que si son componentes). La diferencia esta en el contexto:
    la celda vive dentro de una grilla de celdas iguales, el componente cuelga de un cable.
    """
    cw = int(txt_h * r.uniform(1.8, 3.4)); ch = int(txt_h * r.uniform(1.5, 2.6))
    nc = r.randint(4, 14); nr = r.randint(1, 4)
    M = int(txt_h * .6) + 4
    for _ in range(8):   # 28/09: busca un lugar que no toque ninguna caja positiva (FONDO_BLANCO la borraria entera)
        x0 = r.randint(-cw, max(1, S - nc * cw)); y0 = r.randint(0, max(1, S - nr * ch - 10))
        if libre_de([x0, y0, x0 + nc * cw, y0 + nr * ch], evitar, M): break
    else:
        return
    for i in range(nr + 1):
        cv2.line(c, (x0, y0 + i * ch), (x0 + nc * cw, y0 + i * ch), 0, 1)
    for j in range(nc + 1):
        cv2.line(c, (x0 + j * cw, y0), (x0 + j * cw, y0 + nr * ch), 0, 1)
    # 20/09 (plan R): el vocabulario de antes eran cuatro etiquetas inventadas y las planillas
    # reales de LU-UN-01 / nyw-un-01 numeran los circuitos hasta C60 y repiten RESERVA. Q marcaba
    # como componente justo esas celdas (`C7`, `C21`, `C23`, `C25`...), asi que el negativo tiene
    # que parecerse a lo que el modelo esta confundiendo, no a otra cosa.
    corto = (['C%d' % i for i in range(1, 61)] + ['RES', 'RESERVA', 'N', 'E1', 'T1', 'T2',
             'A1', 'A2', 'LV1', 'OCE1', 'TAB', 'X', 'M'])
    enc = ['N', 'Circuito', 'Tipo', 'Fases', 'P Tot', 'Destino', 'Funcion']
    for i in range(nr):
        for j in range(nc):
            if r.random() < .8:
                voc = enc if (i == 0 and r.random() < .5) else corto
                put_text(c, x0 + j * cw + int(cw * .2), y0 + i * ch + int(ch * .2), r.choice(voc), int(ch * .5), r)

LETRAS_RECUADRO = ['A', 'A', 'A', 'V', 'V', 'R', 'W', 'M', 'E', 'H', 'T', 'S', 'I', 'Hz', 'kW', 'SI', 'AD', 'BA']


def letra_recuadro(c, r, txt_h, xs, boxes):
    """Letra dentro de un recuadro chico, del tamano del texto. ES un componente (28/09).

    El "A" en un cuadradito de LU-UN-01 / nyw-un-01 (bloque A$C45664E16, amperimetro) es la falla mas
    repetida de RF4: 25 casos que ve con 0,05-0,4, porque mide lo mismo que una letra de texto (lado
    ~1,3 alturas de texto) y el modelo lo confunde con texto. `sym_instance` nunca lo genera a ese
    tamano (piso 1,8 alturas). Aca se dibuja a nivel de tile, colgado de un cable o pegado a otro
    aparato por un cable corto, y SIEMPRE con letras sueltas SIN recuadro cerca (el par negativo):
    la diferencia que tiene que aprender es el recuadro, no la letra.
    """
    puestos = 0
    for _ in range(r.randint(1, 4)):
        L = max(7, int(txt_h * r.uniform(1.1, 1.7)))
        txt = r.choice(LETRAS_RECUADRO)
        w = L if len(txt) == 1 else int(L * r.uniform(1.4, 1.8))
        for _ in range(8):
            if xs and r.random() < .6:
                x = min(max(4, r.choice(xs) - w // 2), S - w - 4)   # 28/09 (revision): que no se salga
            else:
                x = r.randint(4, S - w - 4)
            y = r.randint(4, S - L - 4)
            if not overlap([x - 3, y - 3, x + w + 3, y + L + 3], boxes, 2): break
        else:
            continue
        c[max(0, y - 1):y + L + 2, max(0, x - 1):x + w + 2] = 255
        t = 1 if r.random() < .8 else 2
        cv2.rectangle(c, (x, y), (x + w, y + L), 0, t)
        put_text(c, x + max(1, int(w * .18)), y + max(1, int(L * .12)), txt, int(L * .62), r)
        if r.random() < .6:                    # cable corto a un costado (hacia el aparato vecino)
            lado = r.choice([-1, 1]); yy = y + L // 2
            x_a = x if lado < 0 else x + w
            cv2.line(c, (x_a, yy), (x_a + lado * int(txt_h * r.uniform(1.5, 5)), yy), 0, 1)
        elif r.random() < .5:                  # colgado: cable arriba y abajo
            cv2.line(c, (x + w // 2, max(0, y - int(txt_h * 2))), (x + w // 2, y), 0, 1)
            cv2.line(c, (x + w // 2, y + L), (x + w // 2, min(S - 1, y + L + int(txt_h * 2))), 0, 1)
        boxes.append(cenir(c, [x, y, x + w + 1, y + L + 1]))
        puestos += 1
        # par negativo: la misma letra (u otra) suelta, sin recuadro, cerca
        for _ in range(r.randint(1, 3)):
            tx = x + r.randint(-int(txt_h * 6), int(txt_h * 6)); ty = y + r.randint(-int(txt_h * 4), int(txt_h * 4))
            if not overlap([tx - 2, ty - 2, tx + int(txt_h * 2), ty + int(txt_h * 1.6)], boxes, 2):
                put_text(c, tx, ty, r.choice(LETRAS_RECUADRO + ['12A', 'AC3', 'S1', '16A', 'CT', '2NA']), int(txt_h * r.uniform(.8, 1.1)), r)
    return puestos


ROTULOS = ['TSASC4-6', 'TSASC4-5 LUZ CAB.', 'TSASC4-7 LUZ CAB.', 'TS4A', 'TSSB-T2', 'TS-D3', 'TS1A', 'TSFM-1/N',
           'TS-TL1', 'TUE OFFICE', 'TAB. BOMBAS', 'TS ASCENSOR', 'TSPB', 'TSC-2', 'UPS COMANDO', 'TS AA']


def rotulo_vertical(c, r, txt_h, boxes, positivo=True):
    """Fila de rectangulos altos con el nombre del tablero en VERTICAL, al final de los cables.

    28/09, Tomas: NO son componentes (son la referencia al tablero al que va el cable). Se sacaron los 9
    `AUDIT-rotulo_con_nombre` del GT de test1 (v6) y esto se usa como NEGATIVO (positivo=False, P_ROTULO_NEG):
    no agrega cajas. RF4 les daba 0,1-0,45. Lo de abajo es la motivacion original, cuando se crearon como positivos.

    28/09. En test1 las referencias a tableros ("TSASC4-6 LUZ CAB.") son 9 componentes grandes (2,2
    veces la mediana) y alargados: RF4 los ve con 0,1-0,45 y con la caja cubriendo solo la parte de
    arriba. En los 7 planos hay solo 18 componentes de mas del doble de la mediana; el dataset casi no
    tiene cajas de ese tamano con esa forma. Caja = el rectangulo entero.
    """
    n = r.randint(2, 6)
    w = int(txt_h * r.uniform(1.7, 2.8)); h = int(txt_h * r.uniform(6.5, 13.0))
    paso = int(w * r.uniform(1.6, 3.2))
    if n * paso + 10 > S or h + int(txt_h * 4) > S - 10:
        n = max(1, (S - 10) // max(1, paso))
    prev = [list(b[:4]) for b in boxes]
    for _ in range(8 if not positivo else 1):   # 28/09: como negativo prueba 8 lugares (con 1 salia el 18% de las veces)
        x0 = r.randint(4, max(5, S - n * paso - 4)); y0 = r.randint(int(txt_h * 3), max(int(txt_h * 3) + 1, S - h - 6))
        if not overlap([x0 - 2, y0 - int(txt_h * 3), x0 + n * paso, y0 + h + 3], prev, 2 if positivo else int(txt_h * .6) + 4):
            break
    else:
        return 0
    t = 1 if r.random() < .8 else 2
    for k in range(n):
        x = x0 + k * paso
        c[y0:y0 + h + 1, x:x + w + 1] = 255
        cv2.rectangle(c, (x, y0), (x + w, y0 + h), 0, t)
        cv2.line(c, (x + w // 2, max(0, y0 - int(txt_h * r.uniform(2, 6)))), (x + w // 2, y0), 0, 1)
        nom = r.choice(ROTULOS)
        partes = nom.split(' ', 1) if (' ' in nom and w > txt_h * 2.2 and r.random() < .6) else [nom]
        hh = int(txt_h * r.uniform(.75, .95))
        largo_max = h - 6
        for q, p in enumerate(partes):
            while len(p) > 2 and len(p) * hh * .62 > largo_max: p = p[:-1]
            # put_text con rot=True gira 90 antihorario: el texto se lee de abajo hacia arriba
            px = x + 2 + q * int(hh * 1.25)
            py = y0 + max(2, int((h - len(p) * hh * .62) / 2))
            put_text(c, px, py, p, hh, r, rot=True)
        if positivo:
            boxes.append([x, y0, x + w + 1, y0 + h + 1])
    return n


def negativos_extra(c, r, txt_h, evitar=()):
    """Cosas que RF4 marca con >= 0,5 y NO son componentes (28/09, muestra de sus FP). NEGATIVO.

    - circulos de referencia numerados del tamano de un componente (los de `clutter` son mas chicos)
    - circulos grandes que se cortan (burbujas de detalle de EZE4077): marcaba la lente del cruce
    - flecha de alimentacion con punta de triangulo abierto y el nombre del tablero en vertical
    28/09 (revision): se dibujan DESPUES de los simbolos y cada uno busca un lugar que no toque ninguna
    caja (margen `M`): antes iban antes y FONDO_BLANCO borraba casi todos los que caian cerca de un
    componente, que es justo donde RF4 se equivoca.
    """
    M = int(txt_h * .6) + 4

    def lugar(f):          # f() -> (bbox, params); hasta 8 intentos
        for _ in range(8):
            b, p = f()
            if libre_de(b, evitar, M): return p
        return None
    for _ in range(r.randint(0, 3)):           # circulo numerado grande, a veces con linea guia
        R = int(txt_h * r.uniform(1.2, 2.1))
        p = lugar(lambda: (lambda x, y: ([x - R, y - R, x + R, y + R], (x, y)))(r.randint(R, S - R), r.randint(R, S - R)))
        if p is None: continue
        x, y = p
        cv2.circle(c, (x, y), R, 0, 1)
        num = str(r.randint(1, 16))
        put_text(c, x - int(R * .35 * len(num)), y - int(R * .45), num, int(R * .95), r)
        if r.random() < .4:
            ang = r.uniform(0, 2 * np.pi); L = int(txt_h * r.uniform(2, 6))
            p = (int(x + R * np.cos(ang)), int(y + R * np.sin(ang)))
            cv2.line(c, p, (int(p[0] + L * np.cos(ang)), int(p[1] + L * np.sin(ang))), 0, 1)
    if r.random() < .35:                       # 2-3 circulos grandes que se cortan
        R = int(txt_h * r.uniform(4.5, 11)); n = r.randint(2, 3)
        sep = int(R * r.uniform(1.25, 1.85))
        p = lugar(lambda: (lambda x, y: ([x - R, y - R, x + (n - 1) * sep + R, y + R], (x, y)))(
            r.randint(-R // 2, max(-R // 2 + 1, S - n * sep)), r.randint(R // 2, max(R // 2 + 1, S - R // 2))))
        if p is not None:
            x, y = p
            for k in range(n):
                cv2.circle(c, (x + k * sep, y), R, 0, 1)
    for _ in range(r.randint(0, 2)):           # flecha de alimentacion con nombre vertical
        L = int(txt_h * r.uniform(2, 6)); a = int(txt_h * r.uniform(.5, .8))
        p = lugar(lambda: (lambda x, y: ([x - a, y, x + a + int(txt_h * 2), y + L + int(a * 1.8) + int(txt_h * 6)], (x, y)))(
            r.randint(10, S - 30), r.randint(10, max(11, S - int(txt_h * 10)))))
        if p is None: continue
        x, y = p
        cv2.line(c, (x, y), (x, y + L), 0, 1)
        pts = np.array([[x - a, y + L], [x + a, y + L], [x, y + L + int(a * 1.8)]], np.int32)
        cv2.polylines(c, [pts], True, 0, 1)
        if r.random() < .8:
            put_text(c, x + a + 2, y + L - int(txt_h * .5), r.choice(ROTULOS)[:10], int(txt_h * .9), r, rot=True)


def make_tile(lib, r, negative=False):
    txt_h = 11.0 * np.exp(r.uniform(np.log(.8), np.log(2.3)))   # cubre inferencia a escala 1.0-2.2 (texto 9-25 px)
    # 28/09, pedido de Tomas: FONDO_BLANCO=1 -> sin recortes de planos reales de fondo, y todo lo negativo
    # (textos, tablas, marcas, circulos, PAT...) se dibuja en una capa aparte que al final se pega SOLO fuera
    # de las cajas positivas (con margen). Ningun simbolo queda encima de una tabla o de texto: en la grilla
    # de ds21 habia un transformador sobre una planilla. Los simbolos siguen sobre sus cables.
    fb = os.environ.get('FONDO_BLANCO') == '1'
    c = None
    if not fb and r.random() < .3 and lib.bgs: c = real_bg(lib, r, txt_h)
    if c is None: c = np.full((S, S), 255, np.uint8)
    cn = np.full((S, S), 255, np.uint8) if fb else c     # capa de lo negativo
    boxes = []
    t = r.choice([1, 1, 1, 1, 2])
    layout = r.random()
    # red de cables: verticales con buses horizontales (layout unifilar)
    xs = []
    if layout < .75:
        pitch = int(txt_h * r.uniform(5, 15)); x = r.randint(10, pitch)
        while x < S:
            xs.append(x); x += int(pitch * r.uniform(.8, 1.3))
        ybus = sorted(r.sample(range(20, S-20), r.randint(1, 3)))
        for yb in ybus:
            cv2.line(c, (max(0, xs[0] - r.randint(0, 80)), yb), (min(S, xs[-1] + r.randint(0, 80)), yb), 0, t)
        for x in xs:
            ya, yb2 = (r.randint(-50, S//2), r.randint(S//2, S+50)) if r.random() < .7 else (0, S)
            cv2.line(c, (x, ya), (x, yb2), 0, t)
            if r.random() < .5: hash_marks(cn, x, r.randint(0, S), r, 1)
            if r.random() < .5: cv2.circle(c, (x, r.choice(ybus)), r.randint(2, 4), 0, -1 if r.random() < .85 else 1)
    else:
        for _ in range(r.randint(2, 8)):
            if r.random() < .5:
                y = r.randint(0, S); cv2.line(c, (r.randint(-50, S), y), (r.randint(0, S+50), y), 0, t)
            else:
                x = r.randint(0, S); cv2.line(c, (x, r.randint(-50, S)), (x, r.randint(0, S+50)), 0, t)
    clutter(cn, r, txt_h, n_scale=r.uniform(.5, 1.5))
    # barra colectora gruesa: siempre NEGATIVO (va antes de los simbolos, es fondo)
    if r.random() < float(os.environ.get('P_BARRA', '0')):
        barra_colectora(cn, r, txt_h)
    if not negative and xs and r.random() < float(os.environ.get('P_TABLERO', '0')):
        tablero_row(c, r, txt_h, xs, boxes)
    if not negative and xs and r.random() < float(os.environ.get('P_SPM', '0')):
        spm_branch(c, r, txt_h, xs, boxes)
    if not negative and r.random() < float(os.environ.get('P_FUSDXF', '0')):
        fusible_dxf(c, r, txt_h, boxes)
    if not negative and r.random() < float(os.environ.get('P_DENSA', '0')):
        fila_densa(c, r, txt_h, lib, boxes)
    if not negative and r.random() < float(os.environ.get('P_TERNA', '0')):
        terna_rst(c, r, txt_h, lib, boxes)
    if not negative and r.random() < float(os.environ.get('P_PULS', '0')):
        puls_contactor(c, r, txt_h, lib, boxes)
    if not negative and r.random() < float(os.environ.get('P_MARCO', '0')):
        marco_punteado(c, r, txt_h, boxes)
    if not negative and r.random() < float(os.environ.get('P_TRAFO', '0')):
        trafo_o_pareja(c, r, txt_h, boxes)
    if not negative and xs and r.random() < float(os.environ.get('P_FUSIBLE', '0')):
        fusible_solo(c, r, txt_h, xs, boxes)
    if not negative and xs and r.random() < float(os.environ.get('P_COMPUESTO', '0')):
        compuesto_vertical(c, r, txt_h, xs, boxes)
    if xs and r.random() < float(os.environ.get('P_POLO', '0')):
        polo_suelto(cn, r, txt_h, xs)      # negativo: no agrega caja
    if not negative and r.random() < float(os.environ.get('P_LETRA', '0')):
        letra_recuadro(c, r, txt_h, xs, boxes)
    if not negative and r.random() < float(os.environ.get('P_ROTULO', '0')):
        rotulo_vertical(c, r, txt_h, boxes)
    if negative and r.random() < float(os.environ.get('P_TABLERO', '0')) * .5:
        dashed_line(cn, (0, r.randint(20, S-20)), (S, r.randint(20, S-20)), 1, r)
    textos = []   # 28/09 (revision visual): lugar reservado de las etiquetas de texto de los simbolos
    if not negative:
        n = r.randint(1, 14)
        for _ in range(n * 3):
            if len(boxes) >= n: break
            a = sym_instance(lib, r, txt_h)
            if a is None: continue
            h, w = a.shape
            if xs and r.random() < .7:   # sobre un cable vertical
                cx = r.choice(xs); x = cx - w // 2 + r.randint(-2, 2)
            else:
                x = r.randint(-w // 3, S - 2 * w // 3)
            y = r.randint(-h // 3, S - 2 * h // 3)
            b = [x, y, x + w, y + h]
            # margen proporcional: con el fijo de 6 px un simbolo de 13 px casi no encontraba
            # lugar, y por eso los chicos quedaban subrepresentados frente a los planos reales.
            # El margen proporcional viene del plan O y SE DEJA: al revertir el tramo chico de
            # O, medir con este margen deja la mediana en 44 px contra los 45 de ds9, y con el
            # margen fijo de 6 se va a 39. Lo que hundio al plan O fue el tramo chico, no esto.
            if overlap(b, boxes, max(2, min(6, int(min(w, h) * .3)))): continue
            if overlap(b, textos, 2): continue          # 28/09: no se pega encima del texto de otro simbolo
            x0, y0, x1, y1 = max(0, x), max(0, y), min(S, x + w), min(S, y + h)
            if x1 - x0 > 2 and y1 - y0 > 2 and (c[y0:y1, x0:x1] < 128).mean() > (.25 if max(w, h) < 40 else .08): continue
            vis_w, vis_h = x1 - x0, y1 - y0
            vm = os.environ.get('VIS_SINT') or os.environ.get('VIS_MIN')
            if vm and vis_w * vis_h < float(vm) * w * h:
                # 28/09 (revision visual): con VIS_MIN el simbolo cortado directamente NO se dibuja. Antes se
                # dibujaba, se marcaba 'ign' y al final se tapaba con un parche blanco que cortaba cables y
                # palabras ("Secci____dor"); el resultado para el modelo es el mismo, sin el parche.
                continue
            # el simbolo corta el cable: blanqueo parcial interior
            if r.random() < .4:
                c[y0 + 2:y1 - 2, x0 + 2:x1 - 2] = np.maximum(c[y0 + 2:y1 - 2, x0 + 2:x1 - 2], 255 if r.random() < .7 else c[y0 + 2:y1 - 2, x0 + 2:x1 - 2])
            blit_min(c, a, x, y)
            # 28/09: VIS_MIN = fraccion minima de AREA visible para que un simbolo cortado por el
            # borde siga siendo positivo. Con el 50% por lado (hasta 25% del area) el modelo aprendia
            # que medio simbolo tambien es un componente: la mitad de los FP >= 0,5 de RF4 son pedazos
            # (la cruz del interruptor, media lampara). Sin VIS_MIN queda la regla vieja.
            corto = (vis_w * vis_h < float(vm) * w * h) if vm else (vis_w < .5 * w or vis_h < .5 * h)
            if corto:
                boxes.append([x0, y0, x1, y1, 'ign']);  continue
            boxes.append([x0, y0, x1, y1])
            if r.random() < .7:  # etiqueta al costado
                tx = x1 + r.randint(2, 8); ty = y0 + r.randint(0, max(1, h // 2))
                for k in range(r.randint(1, 3)):
                    pal = r.choice(WORDS); yk = ty + int(k * txt_h * 1.35)
                    zt = [tx, yk, tx + int(len(pal) * txt_h * .75) + 2, yk + int(txt_h * 1.3)]
                    if overlap(zt, boxes, 2): break      # 28/09: la etiqueta no se mete en otra caja
                    put_text(c, tx, yk, pal, int(txt_h), r)
                    textos.append(zt)
    # 28/09 (revision): los negativos "de objeto" van DESPUES de los positivos y eligen lugar evitando las
    # cajas; asi sobreviven a FONDO_BLANCO y aparecen al lado de componentes (donde RF4 se confunde).
    if r.random() < float(os.environ.get('P_CELDAS', '0')):
        tabla_celdas(cn, r, txt_h, boxes)          # siempre negativo: no agrega cajas
    if r.random() < float(os.environ.get('P_PAT', '0')):
        pat_negativo(cn, r, txt_h)          # siempre negativo: no agrega cajas
    if r.random() < float(os.environ.get('P_NEGEXTRA', '0')):
        negativos_extra(cn, r, txt_h, boxes)       # siempre negativo: no agrega cajas
    if r.random() < float(os.environ.get('P_ROTULO_NEG', '0')):
        rotulo_vertical(cn, r, txt_h, boxes, positivo=False)   # negativo: no agrega cajas
    # red de seguridad (18/09): ninguna etiqueta puede quedar adentro de otra.
    # Una caja que engloba a otra le ensena al modelo justo lo que no queremos, que es sacar
    # una sola caja para dos componentes, y eso despues le ensucia la etapa 2 de clasificacion.
    # Si algun generador igual las produjo, la zona es ambigua: se marcan las dos como 'ign'
    # y mas abajo se pintan de blanco, que es el mecanismo que ya existe para los recortes.
    for i in range(len(boxes)):
        if len(boxes[i]) == 5: continue
        for j in range(i + 1, len(boxes)):
            if len(boxes[j]) == 5: continue
            a, b = boxes[i], boxes[j]
            ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
            if not ix or not iy: continue
            aa = (a[2]-a[0]) * (a[3]-a[1]); ab = (b[2]-b[0]) * (b[3]-b[1])
            if ix * iy > .75 * min(aa, ab):
                boxes[i] = list(a[:4]) + ['ign']; boxes[j] = list(b[:4]) + ['ign']
                break

    if fb:
        # La capa negativa se pega por OBJETO: se agrupa su tinta en manchas (dilatando ~media altura de
        # texto, asi una palabra o una tabla entera queda en una sola mancha) y se descarta toda mancha que
        # toque la zona de una caja positiva (caja + margen). Borrar solo los pixeles cercanos dejaba el
        # simbolo metido en el medio de la planilla, que es justo lo que Tomas marco.
        m = max(3, int(txt_h * .6)); zona = np.zeros((S, S), np.uint8)
        for b in boxes:
            zona[max(0, int(b[1]) - m):int(b[3]) + m, max(0, int(b[0]) - m):int(b[2]) + m] = 1
        tinta = (cn < 250).astype(np.uint8)          # incluye el gris del antialiasing (si no, quedan fantasmas)
        if tinta.any():
            k = max(3, int(txt_h * .5)) | 1
            n_obj, lab = cv2.connectedComponents(cv2.dilate(tinta, np.ones((k, k), np.uint8)), connectivity=8)
            malos = np.unique(lab[(zona > 0) & (lab > 0)])
            cn = np.where(np.isin(lab, malos[malos > 0]), 255, cn).astype(np.uint8)   # la mancha entera
        c = np.minimum(c, cn)
    # aug global
    if r.random() < .3: c = cv2.GaussianBlur(c, (3, 3), r.uniform(.3, .9))
    if r.random() < .2: c = np.where(c < r.randint(200, 235), 0, 255).astype(np.uint8)
    if r.random() < .15:
        q = r.randint(40, 90); c = cv2.imdecode(cv2.imencode('.jpg', c, [cv2.IMWRITE_JPEG_QUALITY, q])[1], 0)
    # simbolos recortados al borde (<50% visibles) se pintan de blanco para no dejar ambiguedad
    out = []
    for b in boxes:
        if len(b) == 5: c[b[1]:b[3], b[0]:b[2]] = 255
        else: out.append(b)
    return c, out
