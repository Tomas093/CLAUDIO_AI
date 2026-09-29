"""Construye el dataset completo en work/ds (clase unica 'componente')."""
import sys, os, random, json, glob, zipfile, cv2, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from paths import BASE, DATA, WORK, DS
from compose import Lib, make_tile, S
from postproc import ink_map, resize_keep_strokes
from multiprocessing import Pool

# 28/09: fraccion minima de area visible de un componente cortado por el borde del tile para que siga
# siendo positivo (si no, se blanquea). Era 0,6 fijo; VIS_MIN (el mismo que usa compose) la cambia.
VIS_REAL = float(os.environ.get('VIS_MIN') or 0.6)
NP, NN, NR =int(os.environ.get('NP', 4500)), int(os.environ.get('NN', 1500)), int(os.environ.get('NR', 600))   # sinteticos positivos, negativos, tiles de planos reales

def _ensure_lib():
    d = os.path.join(DATA, 'sym_lib')
    if not os.path.isdir(d): zipfile.ZipFile(os.path.join(DATA, 'sym_lib.zip')).extractall(DATA)
_ensure_lib()

def _real():
    lab = json.load(open(os.path.join(DATA, 'real_labels.json')))
    return {n: (cv2.imread(os.path.join(DATA, 'renders', f'{n}_d.png'), 0), lab[n]) for n in lab}
REAL = _real()
# 28/09: revision visual de las etiquetas ORIGINALES (zips a mano y planos reales) con subagentes, reglas de
# Tomas: lo marcado (no componente / cortado / caja mal) no se etiqueta ni es negativo: pasa a zona neutra.
_ro = os.path.join(DATA, 'revision_originales_agentes.json')
REV_ORIG = json.load(open(_ro)) if os.path.exists(_ro) and os.environ.get('REVISION_ORIG', '1') == '1' else {'zip': [], 'real': []}
_RO_REAL = {(k, tuple(round(v, 1) for v in b[:4])) for k, b in REV_ORIG['real']}
_RO_ZIP = {(k, tuple(round(v, 4) for v in b[:4])) for k, b in REV_ORIG['zip']}
_bgs = []
for n, (im, bx) in REAL.items():
    im = im.copy()
    for b in bx: im[max(0,int(b[1])-4):int(b[3])+4, max(0,int(b[0])-4):int(b[2])+4] = 255
    _bgs.append((im, None))
LIB = Lib(os.path.join(DATA, 'sym_lib'), _bgs)
_full = os.path.join(DATA, 'sym_lib_full')
if os.environ.get('LIB_FULL', '1') == '1' and os.path.isdir(_full) and len(os.listdir(_full)) > 100:
    # biblioteca completa + extras no-elmt (componentes manuales y bloques de planos: p0245-p0286)
    extra = Lib(os.path.join(DATA, 'sym_lib'), _bgs).syms[245:]
    LIB = Lib(_full, _bgs); LIB.syms += extra * 3
# Simbolos reales que paso Tomas (data/diferencial.dxf, 17/09): el diferencial completo, el
# testigo de tension, el fusible y los demas bloques de sus planos. Se repiten varias veces
# para que pesen en el fine-tuning, que es corto y busca corregir justo estos simbolos.
_tomas = os.path.join(DATA, 'sym_lib_tomas')
if os.path.isdir(_tomas):
    _st = Lib(_tomas, _bgs, filtrar=False).syms   # curados a mano: no pasan por es_basura
    _peso = int(os.environ.get('PESO_TOMAS', '1'))
    if _st and _peso > 0:
        LIB.syms += _st * _peso
        print('[lib] simbolos de Tomas: %d x%d' % (len(_st), _peso))
print('[lib] simbolos en uso:', len(LIB.syms))

def save(split, name, img, boxes):
    cv2.imwrite(os.path.join(DS, 'images', split, name + '.png'), img)
    with open(os.path.join(DS, 'labels', split, name + '.txt'), 'w') as f:
        for b in boxes:
            x0, y0, x1, y1 = [float(v) for v in b[:4]]
            x0, y0, x1, y1 = max(0,x0), max(0,y0), min(S,x1), min(S,y1)
            if x1 - x0 < 3 or y1 - y0 < 3: continue
            f.write(f'0 {(x0+x1)/2/S:.6f} {(y0+y1)/2/S:.6f} {(x1-x0)/S:.6f} {(y1-y0)/S:.6f}\n')

def synth_job(a):
    split, i, neg = a
    r = random.Random((i * 7919 + (13 if neg else 0) + (1 if split == 'val' else 0)) & 0xffffffff)
    img, boxes = make_tile(LIB, r, negative=neg)
    save(split, f'{"n" if neg else "p"}{i:06d}', img, boxes)

def real_job(a):
    split, i = a
    r = random.Random(10_000 + i)
    # 22/09. Antes el plano se elegia uniforme: plano2 (4 cajas) recibia 121 tiles y vyre (86)
    # 147, o sea 30 repeticiones por caja contra 2. Ahora cada caja etiquetada pesa lo mismo.
    # REAL_UNIFORME=1 vuelve al sorteo viejo (el de ds13).
    ns = sorted(REAL)
    if os.environ.get('REAL_UNIFORME') == '1': n = r.choice(ns)
    else: n = r.choices(ns, weights=[len(REAL[k][1]) for k in ns])[0]
    im, bx = REAL[n]
    s = r.uniform(.75, 1.35); H, W = im.shape; cs = int(S / s); neg = (i % 4 == 3)
    for _ in range(20):
        x = r.randint(-cs//4, max(0, W - 3*cs//4)); y = r.randint(-cs//4, max(0, H - 3*cs//4))
        c = np.full((cs, cs), 255, np.uint8)
        x0, y0, x1, y1 = max(0,x), max(0,y), min(W,x+cs), min(H,y+cs)
        c[y0-y:y1-y, x0-x:x1-x] = im[y0:y1, x0:x1]
        c = cv2.resize(c, (S, S), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
        out = []
        for b in bx:
            q = [(b[0]-x)*s, (b[1]-y)*s, (b[2]-x)*s, (b[3]-y)*s]
            if (n, tuple(round(v, 1) for v in b[:4])) in _RO_REAL:   # revision: zona neutra
                v = [max(0,q[0]), max(0,q[1]), min(S,q[2]), min(S,q[3])]
                if v[2] > v[0] and v[3] > v[1]: c[int(v[1]):int(v[3])+1, int(v[0]):int(v[2])+1] = 255
                continue
            v = [max(0,q[0]), max(0,q[1]), min(S,q[2]), min(S,q[3])]
            if v[2] <= v[0] or v[3] <= v[1]: continue
            fr = (v[2]-v[0])*(v[3]-v[1]) / max(1e-6,(q[2]-q[0])*(q[3]-q[1]))
            if fr >= VIS_REAL: out.append(v)
            else: c[int(v[1]):int(v[3])+1, int(v[0]):int(v[2])+1] = 255
        if neg:
            for q in out: c[max(0,int(q[1])-3):int(q[3])+3, max(0,int(q[0])-3):int(q[2])+3] = 255
            out = []
        if (np.mean(c < 128) > .004) and (neg or out): break
    if r.random() < .2: c = cv2.GaussianBlur(c, (3,3), .6)
    save(split, f'{"rn" if neg else "rp"}{i:06d}', c, out)

def unzip_manual():
    dst = os.path.join(WORK, 'zips'); os.makedirs(dst, exist_ok=True)
    zs = glob.glob(os.path.join(BASE, 'zips_datasets', '*.zip')) + [os.path.join(BASE, z) for z in ['Bornera.v2i.yolov11.zip', 'Diferencial.v3i.yolov11.zip', 'Termomagnetica.v2i.yolov11.zip']]
    for z in zs:
        if not os.path.exists(z): print('[aviso] falta', z); continue
        d = os.path.join(dst, os.path.basename(z)[:-4])
        if not os.path.isdir(d): zipfile.ZipFile(z).extractall(d)
    return dst

def manual_tiles():
    """Tiles de los zips anotados a mano, con las etiquetas que faltaban completadas.

    18/09. Cada zip se anoto para UN tipo de componente: en una captura que muestra medio
    tablero se etiqueto solo el interruptor motorizado (o la fotocelula, o el medidor) y todos
    los demas componentes quedaron sin etiqueta. Medido con el modelo K, las detecciones firmes
    que no caen sobre ninguna etiqueta eran hasta 6,6 veces mas que las etiquetas existentes:
    cada una le ensenaba al modelo que ese componente NO es un componente, justo lo contrario
    de lo que buscamos con recall 100%. `src/pseudo_zips.py` deja en data/zips_pseudo.json una
    caja candidata por cada una, con su confianza; aca las firmes entran como etiqueta y las
    dudosas se blanquean, que es el mecanismo que ya existia para lo ambiguo.
    """
    zdir = unzip_manual()
    res = json.load(open(os.path.join(DATA, 'zips_merged.json')))
    ps = {}
    _pp = os.path.join(DATA, 'zips_pseudo.json')
    if os.environ.get('USAR_PSEUDO', '1') == '1' and os.path.exists(_pp):
        ps = json.load(open(_pp))
        print('[manual] etiquetas completadas desde zips_pseudo.json')
    ALTA = float(os.environ.get('PSEUDO_ALTA', '0.80'))
    # 28/09: PSEUDO_CONFIRMA=1 usa data/zips_pseudo_confirmadas.json (src/confirmar_pseudo.py, RF4)
    conf_rf4 = None
    _pc = os.path.join(DATA, 'zips_pseudo_confirmadas.json')
    if os.environ.get('PSEUDO_CONFIRMA') == '1' and os.path.exists(_pc):
        conf_rf4 = {(k, round(f[0], 2), round(f[1], 2), round(f[2], 2), round(f[3], 2)): f
                    for k, v in json.load(open(_pc)).items() for f in v}
        print('[manual] pseudo-etiquetas filtradas con RF4:', len(conf_rf4))
    n_rf4_fuera = 0
    # 28/09: revision a mano de Tomas (grillas A y B de work/revision_tomas): lo que marco como componente
    # queda positivo aunque RF4 no lo vea o toque el borde; lo no nombrado sigue en zona neutra.
    rev_pos = set()
    _pr = os.path.join(DATA, 'zips_pseudo_revision_tomas.json')
    if conf_rf4 is not None and os.path.exists(_pr) and os.environ.get('PSEUDO_REVISION', '1') == '1':
        rev_pos = {(p[0], round(p[1], 1), round(p[2], 1), round(p[3], 1), round(p[4], 1))
                   for p in json.load(open(_pr))['positivo']}
        print('[manual] pseudo-etiquetas confirmadas a mano por Tomas:', len(rev_pos))
    rev_neutra = set(); n_rev_neutra = 0
    _pa = os.path.join(DATA, 'zips_pseudo_revision_agentes.json')
    if conf_rf4 is not None and os.path.exists(_pa) and os.environ.get('PSEUDO_REVISION', '1') == '1':
        rev_neutra = {(p[0], round(p[1], 1), round(p[2], 1), round(p[3], 1), round(p[4], 1))
                      for p in json.load(open(_pa))['neutra']}
        print('[manual] pseudo-etiquetas a zona neutra por la revision visual:', len(rev_neutra))

    def roza(q, acept):
        """True si la caja toca aunque sea un poco alguna etiqueta ya aceptada.

        Lo que se blanquea desaparece del tile. Una zona dudosa que apenas roza un componente
        etiquetado le borraria un pedazo y la etiqueta quedaria sobre un hueco: asi quedaron
        572 etiquetas casi vacias. Si roza algo, se descarta en silencio en vez de blanquearse.
        """
        aq = max(1e-12, (q[2]-q[0]) * (q[3]-q[1]))
        for o in acept:
            ix = max(0.0, min(q[2], o[2]) - max(q[0], o[0]))
            iy = max(0.0, min(q[3], o[3]) - max(q[1], o[1]))
            if ix > 0 and iy > 0 and ix * iy > .05 * min(aq, max(1e-12, (o[2]-o[0])*(o[3]-o[1]))):
                return True
        return False

    r = random.Random(123); n = 0; st = [0, 0]
    origen = {}   # 28/09: de que captura y si es etiqueta original o pseudo, por cada tile manual (revision de Tomas)
    n_ps = n_ig = n_bajo_ign = n_vacias = 0
    for key in sorted(res):
        g, a, ign = res[key]; boxes = g + a
        malas = [b for b in boxes if (key, tuple(round(v, 4) for v in b[:4])) in _RO_ZIP]
        if malas:   # revision de originales: a zona neutra
            boxes = [b for b in boxes if b not in malas]; ign = list(ign) + malas
        n_orig = len(boxes)
        pv = ps.get(key)
        if pv:
            W0, H0 = float(pv['w']), float(pv['h'])
            ign0 = list(ign); ign = list(ign)
            # Las candidatas se aceptan de mayor a menor confianza y solo si no chocan con una
            # caja ya aceptada. Sin este filtro el modelo aporta la misma pieza dos veces (el
            # aparato entero y su polo) y quedan etiquetas anidadas: la primera version del ds8
            # llego al 3,0% de pares anidados, cuando el ds7 tenia 0,0%. Una caja adentro de
            # otra es justo lo que no queremos que aprenda, asi que lo descartado se blanquea.
            for b in sorted(pv['cand'], key=lambda z: -z[4]):
                q = [b[0]/W0, b[1]/H0, b[2]/W0, b[3]/H0]
                aq = max(1e-12, (q[2]-q[0]) * (q[3]-q[1]))
                choca = False
                for o in boxes:
                    ix = max(0.0, min(q[2], o[2]) - max(q[0], o[0]))
                    iy = max(0.0, min(q[3], o[3]) - max(q[1], o[1]))
                    if ix <= 0 or iy <= 0: continue
                    I = ix * iy; ao = max(1e-12, (o[2]-o[0]) * (o[3]-o[1]))
                    if I / (aq + ao - I) > .40 or I / min(aq, ao) > .70:
                        choca = True; break
                # Ojo con que se blanquea: una candidata que CHOCA con una caja ya aceptada cae
                # justo encima de un componente etiquetado, asi que blanquearla le borraria el
                # simbolo y la etiqueta quedaria sobre un hueco. Eso dejo 4.582 etiquetas casi
                # vacias en la primera version. Solo se blanquea lo dudoso que no pisa nada.
                if choca: continue
                # 22/09. Una candidata que cae adentro de una zona neutra ORIGINAL del zip se
                # blanquea mas abajo junto con la zona, y la etiqueta queda sobre un hueco: asi
                # habia 492 cajas vacias en ds13 (393 candidatas firmes en esa situacion). La zona
                # es un componente cortado por el borde de la captura: manda la zona.
                if any(max(0.0, min(q[2], z[2]) - max(q[0], z[0])) * max(0.0, min(q[3], z[3]) - max(q[1], z[1]))
                       > .5 * aq for z in ign0):
                    n_bajo_ign += 1; continue
                firme = b[4] >= ALTA
                if firme and conf_rf4 is not None:
                    # 28/09 (Tomas, opcion b): solo queda como etiqueta si RF4 tambien la ve (>= 0,3) y no toca
                    # el borde de la captura (componente posiblemente cortado: "que se vea el 90%"). Si no, neutra.
                    f4 = conf_rf4.get((key, round(b[0], 2), round(b[1], 2), round(b[2], 2), round(b[3], 2)))
                    k1 = (key, round(b[0], 1), round(b[1], 1), round(b[2], 1), round(b[3], 1))
                    if (f4 is None or f4[5] < .3 or f4[6]) and k1 not in rev_pos:
                        firme = False; n_rf4_fuera += 1
                    elif k1 in rev_neutra and k1 not in rev_pos:
                        # revision visual con subagentes (28/09): no componente / cortado / caja mal -> neutra
                        firme = False; n_rev_neutra += 1
                if firme: boxes.append(q); n_ps += 1
                elif not roza(q, boxes): ign.append(q); n_ig += 1
        if 'v2i' in key or 'v3i' in key: continue   # exportes Roboflow 'Resize 640x640 (Stretch)': simbolos deformados, se excluyen
        f = os.path.join(zdir, *key.split('/'))
        if not boxes or not os.path.exists(f): continue
        split = 'val' if r.random() < .1 else 'train'
        im = cv2.imread(f)
        if im is None: continue
        ink = ink_map(im); H, W = ink.shape
        ms = np.median([max((b[2]-b[0])*W, (b[3]-b[1])*H) for b in boxes])
        for k in range(1 if ('v2i' in key or 'v3i' in key) else 2):
            s = 70.0 * np.exp(r.uniform(np.log(.7), np.log(1.4))) / ms
            x = resize_keep_strokes(ink, s); h, w = x.shape
            ox = r.randint(0, max(0, w - S)); oy = r.randint(0, max(0, h - S))
            c = np.full((S, S), 255, np.uint8); cr = x[oy:oy+S, ox:ox+S]; c[:cr.shape[0], :cr.shape[1]] = cr
            for b in ign:   # componentes cortados por el borde de la captura: zona neutra (blanco)
                x0_, y0_, x1_, y1_ = int(b[0]*W*s-ox)-2, int(b[1]*H*s-oy)-2, int(b[2]*W*s-ox)+2, int(b[3]*H*s-oy)+2
                c[max(0,y0_):max(0,y1_), max(0,x0_):max(0,x1_)] = 255
            lab = []; fuente_lab = []
            for ib, b in enumerate(boxes):
                q = [b[0]*W*s-ox, b[1]*H*s-oy, b[2]*W*s-ox, b[3]*H*s-oy]
                v = [max(0,q[0]), max(0,q[1]), min(S,q[2]), min(S,q[3])]
                if v[2] <= v[0] or v[3] <= v[1]: continue
                if (v[2]-v[0])*(v[3]-v[1]) / max(1e-6,(q[2]-q[0])*(q[3]-q[1])) >= VIS_REAL:
                    lab.append(v); fuente_lab.append(['orig' if ib < n_orig else 'pseudo'] + [round(t, 4) for t in b[:4]])
                else: c[int(v[1]):int(v[3])+1, int(v[0]):int(v[2])+1] = 255
            # Red final: lo blanqueado (zonas neutras, cortes de borde) puede haberse llevado el
            # simbolo de otra etiqueta. Una caja con menos de 2% de tinta es un hueco: se saca.
            ok = []; ok_f = []
            for v, fu in zip(lab, fuente_lab):
                sub = c[int(v[1]):int(np.ceil(v[3])), int(v[0]):int(np.ceil(v[2]))]
                if sub.size and (sub < 128).mean() >= .02: ok.append(v); ok_f.append(fu)
                else: n_vacias += 1
            lab = ok
            if not lab: continue
            save(split, f'm{n:06d}', c, lab)
            origen[f'm{n:06d}'] = dict(captura=key, cajas=[[round(t, 1) for t in v] + fu for v, fu in zip(lab, ok_f)])
            n += 1; st[split == 'val'] += 1
    json.dump(origen, open(os.path.join(DS, 'origen_manual.json'), 'w'))
    print('[manual] tiles train/val', st, '| etiquetas agregadas', n_ps, '| zonas neutras', n_ig,
          '| candidatas bajo zona neutra', n_bajo_ign, '| etiquetas vacias sacadas', n_vacias, '| pseudo a neutra por RF4/borde', n_rf4_fuera, '| por revision visual', n_rev_neutra)

if __name__ == '__main__':
    for sp in ('train', 'val'):
        os.makedirs(os.path.join(DS, 'images', sp), exist_ok=True); os.makedirs(os.path.join(DS, 'labels', sp), exist_ok=True)
    jobs = [('train', i, False) for i in range(NP)] + [('train', i, True) for i in range(NN)] + \
           [('val', i, False) for i in range(NP // 10)] + [('val', i, True) for i in range(NN // 10)]
    with Pool(max(1, os.cpu_count() - 1)) as p:
        p.map(synth_job, jobs, chunksize=32); print('[synth] ok')
        p.map(real_job, [('train', i) for i in range(NR)] + [('val', 900000 + i) for i in range(NR // 10)], chunksize=16); print('[real] ok')
    manual_tiles()
    with open(os.path.join(DS, 'data.yaml'), 'w') as f:
        f.write(f"path: {DS.replace(os.sep, '/')}\ntrain: images/train\nval: images/val\nnames:\n  0: componente\n")
    for sp in ('train', 'val'): print(sp, len(os.listdir(os.path.join(DS, 'images', sp))))
