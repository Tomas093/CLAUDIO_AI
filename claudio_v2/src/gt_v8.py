"""GT v8 de los planos de test (30/09). Une las dos "v7" que habia y aplica la revision de Tomas.

Habia DOS v7 hechas en paralelo:
  - la de la otra sesion (work/gt_v8/base_otra_sesion, ver CAMBIOS_v7.txt): saca PAT, divide los testigos
    de tension, reajusta EZE4077/test1 y cajas infladas. Es la que Tomas vio en los PDF.
  - la mia (<gt>_v7.csv): v6 + 289 faltantes `AUDIT7-faltante` (grilla work/revision_tomas/GT_v7_agregados).
v8 = base de la otra sesion + mis AUDIT7 con la revision de Tomas del 30/09:
  - fuera: 75 y 167 (PAT en cuadrado), 288, 170 (caja roja mal; la verde ya estaba).
  - neutras (ni TP ni FP ni FN): 129, 274, 275 (Tomas no esta seguro) y toda grilla de accesorios
    (BA|BC|NCx2 / MO|U<|NAx2, como la 129): ni la grilla ni sus celdas se cuentan.
  - TOMA: el componente es el GABINETE entero (como la 146), no los fusibles de adentro. Se toma el
    rectangulo exterior que encierra el texto "TOMA" y se sacan las cajas que quedan adentro.
  - BA (bobina de apertura) y los demas accesorios con recuadro propio (BC, MO, U<, AD, SI, SCI, Fc, MC)
    SI son componentes: caja = recuadro chico que encierra el texto.
  - RES (reservas, bloques *U52/*U56/*U57/*U46): la caja se comia el tramo punteado de abajo; se recorta a
    la altura del interruptor termomagnetico vecino de la misma fila.
  - KC (contactor de barra normal, fl_un_02): la caja AUDIT-contactor solo cubria la bobina; pasa a cubrir
    bobina + contacto con sus dos bornes (union con el bloque A$C40E17AD0).
Salida: <gt>_v8.csv y <gt>_v8_neutras.csv al lado de cada GT.

    py -3 src/gt_v8.py            # informa
    py -3 src/gt_v8.py --aplicar  # escribe
"""
import sys, os, csv, json, importlib, collections
sys.path.insert(0, os.path.dirname(__file__))
import ezdxf
from paths import BASE
import evaluate as ev

AQUI = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE8 = os.path.join(AQUI, 'work', 'gt_v8', 'base_otra_sesion')
FUERA = {75, 167, 288, 170, 138}   # 138: bobina suelta dentro del contactor (Tomas 30/09)
NEUTRAS = {129, 274, 275}
RES = {'*U52', '*U56', '*U57', '*U46'}
PRE = {'nyw-un-01 esquemas unifilares': 'nyw-un-01_v8', 'LU-UN-01 Esquemas Unifilares': 'LU-UN-01_v8', 'fl_un_02': 'fl_un_02_v8',
       'tsss_2': 'tsss_2_v8', 'test1': 'test1_v8', 'test_2': 'test_2_v8', 'EZE4077-IE-UF001-02': 'EZE4077-IE-UF001-02_v8'}
ACCES = {'BA', 'BC', 'MO', 'U<', 'AD', 'SI', 'SCI', 'FC', 'MC'}


def planos():
    os.environ.pop('EVAL_SET', None); importlib.reload(ev); P = list(ev.PLANOS)
    os.environ['EVAL_SET'] = 'qet'; os.environ['QET_DIR'] = 'test_marcelo'; importlib.reload(ev); P += ev.PLANOS
    os.environ.pop('EVAL_SET', None)
    return P


def base_de(pl):
    for f in os.listdir(BASE8):
        if f.endswith('.csv') and f.lower().startswith(pl.lower()[:8]):
            return os.path.join(BASE8, f)


def leer(p):
    return [r for r in csv.DictReader(open(p, encoding='utf-8', errors='ignore')) if r.get('x1')]


def caja(r): return tuple(float(r[k]) for k in ('x1', 'y1', 'x2', 'y2'))


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1])); I = ix * iy
    return I / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - I + 1e-12)


def dentro(p, b, m=0.0): return b[0] - m <= p[0] <= b[2] + m and b[1] - m <= p[1] <= b[3] + m


def centro(b): return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)


def fila(n, b): return dict(block_name=n, x_cad=centro(b)[0], y_cad=centro(b)[1], x1=b[0], y1=b[1], x2=b[2], y2=b[3])


def rects(msp):
    R = []
    for e in msp.query('LWPOLYLINE'):
        p = [(x, y) for x, y, *_ in e.get_points()]
        if 4 <= len(p) <= 5:
            xs = [a for a, _ in p]; ys = [b for _, b in p]
            b = (min(xs), min(ys), max(xs), max(ys))
            if b[2] - b[0] > 0 and b[3] - b[1] > 0: R.append(b)
    return R


def area(b): return (b[2] - b[0]) * (b[3] - b[1])


def main(aplicar):
    num = {int(k): v for k, v in json.load(open(os.path.join(AQUI, 'work', 'gt_v8', 'grilla_v7_num.json'))).items()}
    tot = collections.Counter()
    for pl, dxf, gtp in planos():
        gtp = os.path.join(BASE, gtp); b8 = base_de(pl)
        filas = leer(b8) if b8 else leer(gtp[:-4] + '_v6.csv' if os.path.exists(gtp[:-4] + '_v6.csv') else gtp)
        neut = []
        # mis AUDIT7 con la revision de Tomas
        mias = [f for f in num.items() if f[1]['plano'] == pl]
        for k, r in mias:
            b = tuple(r['caja_cad'])
            if k in FUERA: tot['fuera'] += 1; continue
            if k in NEUTRAS: neut.append(fila('NEUTRA-%d' % k, b)); continue
            if any(iou(b, caja(f)) > .5 for f in filas): tot['dup'] += 1; continue
            filas.append(fila('AUDIT7-%d' % k, b))
        doc = ezdxf.readfile(os.path.join(BASE, dxf)); msp = doc.modelspace(); R = rects(msp)
        textos = []
        for e in msp.query('TEXT MTEXT'):
            t = (e.dxf.text if e.dxftype() == 'TEXT' else e.text).strip().upper()
            textos.append((t, (e.dxf.insert.x, e.dxf.insert.y)))
        # TOMA: gabinete entero = rectangulo mas grande que encierra el texto, sin ser una hoja
        for t, p in textos:
            if not t.startswith('TOMA') or t.startswith('TOMAS') or 'TRANSF' in t: continue
            C = sorted([r for r in R if dentro(p, r)], key=area)
            if not C: print('  TOMA sin recuadro', pl, p); continue
            chico = C[0]; marco = chico
            for r in C[1:]:
                if area(r) < 2.0 * area(chico): marco = r      # marco doble: el exterior
            n0 = len(filas)
            filas = [f for f in filas if not (dentro(centro(caja(f)), marco) and area(caja(f)) < .9 * area(marco))]
            tot['sacadas_en_toma'] += n0 - len(filas)
            if not any(iou(marco, caja(f)) > .7 for f in filas): filas.append(fila('V8-toma', marco)); tot['toma'] += 1
        # accesorios con recuadro propio. Si la celda tiene otra celda de accesorio pegada es una GRILLA
        # (BA|BC / MO|U<...): la grilla entera va neutra (Tomas dudo con la 129) y no se agregan las celdas.
        # Si el recuadro cae adentro de otra caja del GT (el "SI" de un diferencial) tampoco es aparte.
        EXTRA = ACCES | {'NCX2', 'NAX2', 'NCX4', 'NAX4', '2CA', 'NCX1', 'NAX1'}
        celdas = []
        for t, p in textos:
            if t not in EXTRA: continue
            C = sorted([r for r in R if dentro(p, r)], key=area)
            if C: celdas.append((t, C[0]))
        med = sorted(area(caja(f)) for f in filas)[len(filas) // 2] if filas else 1e9
        pegadas = lambda a, b: a is not b and max(a[0], b[0]) - min(a[2], b[2]) < .1 * (a[2] - a[0]) and max(a[1], b[1]) - min(a[3], b[3]) < .1 * (a[3] - a[1])
        grupos = []
        for t, r in celdas:
            if area(r) > 4 * med: continue
            g = [x for x in grupos if any(pegadas(r, q) for q in x)]
            for x in g: grupos.remove(x)
            grupos.append(sum(g, []) + [r])
        for gr in grupos:
            if len(gr) >= 2:
                u = (min(q[0] for q in gr), min(q[1] for q in gr), max(q[2] for q in gr), max(q[3] for q in gr))
                if not any(iou(u, caja(f)) > .5 for f in neut): neut.append(fila('NEUTRA-grilla', u))
                n0 = len(filas); filas = [f for f in filas if not (iou(caja(f), u) > .3 or dentro(centro(caja(f)), u))]
                tot['grilla_a_neutra'] += n0 - len(filas)
        sueltas = [r for gr in grupos if len(gr) == 1 for r in gr]
        for t, r in celdas:
            if t not in ACCES or r not in sueltas: continue
            if any(dentro(centro(r), caja(f)) and area(caja(f)) > 1.5 * area(r) for f in filas): continue
            ya = [f for f in filas if iou(r, caja(f)) > .3 or dentro(centro(caja(f)), r)]
            if ya:
                for f in ya:
                    if f['block_name'].startswith('AUDIT7'): f.update(fila(f['block_name'], r))
                continue
            filas.append(fila('V8-' + t, r)); tot['acc_' + t] += 1
        # RES: recortar a la fila de termomagneticas
        tm = [caja(f) for f in filas if f['block_name'] == 'TM-DIN']
        delta = collections.defaultdict(list); sin = []
        for f in filas:
            if f['block_name'] not in RES: continue
            b = caja(f); cx = centro(b)[0]
            C = [t for t in tm if t[1] >= b[1] - .05 and t[3] <= b[3] + .05 and abs(centro(t)[0] - cx) < 5]
            if not C: sin.append(f); continue
            t = min(C, key=lambda t: abs(centro(t)[0] - cx)); w = t[2] - t[0]
            delta[f['block_name']].append((b[3] - t[3], t[3] - t[1], w))
            f.update(fila(f['block_name'], (cx - w / 2, t[1], cx + w / 2, t[3]))); tot['res'] += 1
        # fila entera de reservas (sin termomagnetica al lado): offset tipico del mismo bloque
        todos = [d for L in delta.values() for d in L]
        for f in sin:
            L = delta.get(f['block_name']) or todos
            if not L: continue
            dt, h, w = [sorted(v)[len(v) // 2] for v in zip(*L)]
            b = caja(f); cx = centro(b)[0]
            f.update(fila(f['block_name'], (cx - w / 2, b[3] - dt - h, cx + w / 2, b[3] - dt))); tot['res_tipico'] += 1
        # KC: bobina + contacto
        for e in msp.query('INSERT[name=="A$C40E17AD0"]'):
            x, y = e.dxf.insert.x, e.dxf.insert.y; kb = (x - .005, y - .049, x + .870, y + .519)
            for f in filas:
                if f['block_name'] == 'AUDIT-contactor' and iou(caja(f), kb) > 0 or (f['block_name'] == 'AUDIT-contactor' and dentro(centro(caja(f)), kb, .2)):
                    b = caja(f); f.update(fila('AUDIT-contactor', (min(b[0], kb[0]), min(b[1], kb[1]), max(b[2], kb[2]), max(b[3], kb[3])))); tot['kc'] += 1
        # las cajas de GT que cayeron sobre una grilla neutra pasan a neutras
        g = [n for n in neut if n['block_name'] == 'NEUTRA-grilla']
        n0 = len(filas); filas = [f for f in filas if not any(iou(caja(f), caja(n)) > .5 for n in g)]; tot['a_neutra'] += n0 - len(filas)
        # cajas corridas (gt_alinear.py) y faltantes por copia exacta (gt_plantillas.py, NCC >= 0,95)
        pre = PRE.get(pl)
        f = os.path.join(AQUI, 'work', 'gt_v8', 'realineos_%s.json' % pre)
        if pre and os.path.exists(f):
            for m in json.load(open(f)):
                for fl in filas:
                    if max(abs(a - b) for a, b in zip(caja(fl), m['de'])) < 1e-3:
                        fl.update(fila(fl['block_name'], tuple(m['a']))); tot['realineada'] += 1
        f = os.path.join(AQUI, 'work', 'gt_v8', 'cand_plantillas_%s.json' % pre)
        if pre and os.path.exists(f):
            for c in json.load(open(f)):
                b = tuple(c['caja_cad'])
                if c['ncc'] < .95 or any(iou(b, caja(x)) > .3 or dentro(centro(b), caja(x)) for x in filas): continue
                filas.append(fila('V8-copia', b)); tot['copia'] += 1
        print('%-32s base %s -> %d cajas, %d neutras' % (pl, os.path.basename(b8) if b8 else '-', len(filas), len(neut)))
        if aplicar:
            for suf, F in (('_v8.csv', filas), ('_v8_neutras.csv', neut)):
                with open(gtp[:-4] + suf, 'w', newline='', encoding='utf-8') as fh:
                    w = csv.DictWriter(fh, ['block_name', 'x_cad', 'y_cad', 'x1', 'y1', 'x2', 'y2'], extrasaction='ignore')
                    w.writeheader(); w.writerows(F)
    print(dict(tot))


if __name__ == '__main__':
    main('--aplicar' in sys.argv)
