"""Tiles REALES de entrenamiento a partir del GT nuevo de Tomas (Ground-Truth-Planos.zip, 30/09).

Usa los planos que NO son de test: los PDF convertidos a DXF (scripts/pdf_dxf, GT en puntos PDF -> caja CAD
(x1, -y2, x2, -y1)) y los DXF nativos que tenemos (background, hola). Los 7 planos de test quedan afuera.
Render igual que evaluate.py (render_doc + darken), a 2 escalas (x1,0 y x1,6, con +-15% de jitter), tiles de 640
con paso 448. Reglas de Tomas: solo se etiqueta lo visible >= 90%; lo cortado y las zonas neutras se pintan de
blanco (ni positivo ni negativo). Escala base: la que deja el lado mediano de los componentes igual que en los
DXF con texto (background/hola, via auto_ppc), asi no se usa nada de los planos de test.
Los circulos de marcado de SAMPLE B (polilineas cerradas redondas de 44-96 pt) se sacan antes de renderizar.
Validacion: planos enteros apartados (VAL).

    py -3 -u src/reales_tomas.py        -> work/reales_tomas/{images,labels}/{train,val}
"""
import os, sys, csv, random
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, cv2, ezdxf
from paths import BASE, WORK
from render import render_doc, cad2px
from scale import auto_ppc
from postproc import darken

Z = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'gt_planos_tomas', 'Ground-Truth-Planos')
PD = os.path.join(Z, 'scripts', 'pdf_dxf')
OUT = os.path.join(WORK, 'reales_tomas'); S = 640; PASO = 448
# PDF que NO tienen DXF nativo (los que si tienen, se usan desde el DXF: sin duplicar planos)
PDFS = {'03.10- IE-UNI-02 rev1': 'IE-UNI-02', '03.11- IE-UNI-03 rev1': 'IE-UNI-03', '03.12- IE-UNI-04 rev1': 'IE-UNI-04',
        'ByA-IE-EU Transf y TSSG-TSSG': 'bya_tssg', 'ByA-IE-EU Transf y TSSG-TSTransferenia': 'bya_tstransf',
        'ByA-IE-EU-Transferencia Unidades-Model': 'bya_unidades', 'Plano 1': 'Plano_1', 'Plano 2': 'Plano_2',
        'Plano 3': 'Plano_3', 'Planos tableros': 'Planos_tableros', 'PR-EE-001-TGBT0101-Rev1 p1': 'pree001'}
# DXF nativos (Planos_Marcelo.zip de Tomas, 30/09): nombre del GT -> prefijo del archivo
DXFN = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'planos_marcelo_nuevo', 'Planos_Marcelo', 'DXF')
_pref = {'116-BRANDSEN 923- IE REP- BELLONI': '116-BRANDSEN', '13-LJN-E-07-01-075-013-Rev.C': '13-LJN', '14-LJN-E-02-12-077-014-Rev.A': '14-LJN',
         '16-LJN-E-02-12-077-016-Rev.A': '16-LJN', 'BSAR-184-1126-1695 Extrusora Filler': 'BSAR-184-1126', 'BSAR-184-1661-1801 TABLEROS ELECTRICOS': 'BSAR-184-1661',
         'IE-UNI-01 rev1': 'IE-UNI-01', 'SAMPLE B-24537-ASS-IE-GE-EU010-r00': 'SAMPLE B', 'Tablero de bombas elevadora de agua': 'Tablero de bombas',
         'background': 'background', 'hola': 'hola'}
NATIVOS = {}
for _n, _p in _pref.items():
    _f = [f for f in os.listdir(DXFN) if f.startswith(_p) and f.lower().endswith('.dxf')] if os.path.isdir(DXFN) else []
    if _f: NATIVOS[_n] = os.path.join(DXFN, _f[0])
# 01/10 (Tomas: "usa los pdfs tambien, usa todos"): versiones PDF de planos que ya estan como DXF nativo (otro estilo
# de dibujo: texto como trazos, grosores del PDF) + hojas sueltas de SAMPLE B. Con RT_EXTRA=1 se arman SOLO estos, en
# work/reales_tomas2 (prefijo rp_).
PDFS_EXTRA = {'03.9- IE-UNI-01 rev1': 'IE-UNI-01', '13-LJN-E-07-01-075-013-Rev.C': '13-LJN', '14-LJN-E-02-12-077-014-Rev.A': '14-LJN',
              '16-LJN-E-02-12-077-016-Rev.A': '16-LJN', 'ByA-IE-EU Transf y TSSG-TSTransferenia Editado': 'bya_tstransf_ed',
              **{'SAMPLE B %d-24537-ASS-IE-GE-EU010-r00' % k: 'SAMPLEB%d' % k for k in range(1, 9)}}
EXTRA = os.environ.get('RT_EXTRA') == '1'
if EXTRA: OUT = os.path.join(WORK, 'reales_tomas2')
TEST = {'test1', 'test_2', 'FL-UN-02', 'TSSS_2', 'EZE4077-IE-UF001-02', 'LU-UN-01 Esquemas Unifilares', 'nyw-un-01 esquemas unifilares'}
assert not TEST & set(NATIVOS) and not TEST & set(PDFS) and not TEST & set(PDFS_EXTRA)
VAL = {'03.12- IE-UNI-04 rev1', 'Plano 3', 'hola', 'BSAR-184-1126-1695 Extrusora Filler'}
TOPE = int(os.environ.get('RT_TOPE', '600'))   # tiles maximos por plano y escala (SAMPLE B no tiene que tapar al resto)


def leer(p, pdf):
    if not os.path.exists(p): return []
    out = []
    for r in csv.DictReader(open(p, encoding='utf-8')):
        x1, y1, x2, y2 = (float(r[k]) for k in ('x1', 'y1', 'x2', 'y2'))
        out.append((x1, -y2, x2, -y1) if pdf else (x1, y1, x2, y2))
    return out


def sin_circulos_marcado(doc, B=None):
    """PDF convertido: polilineas cerradas redondas de 44-96 pt. DXF nativo (B = cajas del GT): CIRCLE amarillo
    (ACI 2) de radio >= 0,5 x lado mediano con el centro de alguna caja adentro (= gtlib.drop_markup_circles)."""
    msp = doc.modelspace(); fuera = []
    if B:
        lado = np.median([max(b[2] - b[0], b[3] - b[1]) for b in B]); C = np.array([((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for b in B])
        for e in msp.query('CIRCLE'):
            if e.dxf.color != 2 or e.dxf.radius < .5 * lado: continue
            if (np.hypot(C[:, 0] - e.dxf.center.x, C[:, 1] - e.dxf.center.y) <= e.dxf.radius).any(): fuera.append(e)
        for e in fuera: msp.delete_entity(e)
        return len(fuera)
    for e in msp.query('LWPOLYLINE'):
        p = [(x, y) for x, y, *_ in e.get_points()]
        if len(p) < 9: continue
        xs = [a for a, _ in p]; ys = [b for _, b in p]; w, h = max(xs) - min(xs), max(ys) - min(ys)
        if 40 <= w <= 100 and .9 < w / max(h, 1e-9) < 1.1 and np.hypot(p[0][0] - p[-1][0], p[0][1] - p[-1][1]) < .05 * w:
            fuera.append(e)
    for e in fuera: msp.delete_entity(e)
    return len(fuera)


def objetivo():
    """lado mediano en px de los componentes cuando el plano se renderiza con auto_ppc (DXF con texto)."""
    v = []
    for n, f in NATIVOS.items():
        doc = ezdxf.readfile(f); ppc = auto_ppc(doc); B = leer(os.path.join(Z, 'DXFs', n, n + '_gt.csv'), False)
        if ppc and B: v.append(np.median([max(b[2] - b[0], b[3] - b[1]) for b in B]) * ppc)
    return float(np.median(v))


def main():
    rng = random.Random(0); tgt = objetivo(); print('[rt] lado objetivo %.1f px' % tgt, flush=True)
    for sp in ('train', 'val'):
        for t in ('images', 'labels'): os.makedirs(os.path.join(OUT, t, sp), exist_ok=True)
    planos = [(n, os.path.join(PD, d + '.p0.dxf'), os.path.join(Z, 'PDFs', n, n + '_gt.csv'), os.path.join(Z, 'PDFs', n, n + '_gt_neutras.csv'), True) for n, d in (PDFS_EXTRA if EXTRA else PDFS).items()]
    if not EXTRA: planos += [(n, f, os.path.join(Z, 'DXFs', n, n + '_gt.csv'), os.path.join(Z, 'DXFs', n, n + '_gt_neutras.csv'), False) for n, f in NATIVOS.items()]
    tot = {'train': [0, 0], 'val': [0, 0]}
    for n, dxf, g, ne, pdf in planos:
        B = leer(g, pdf); N = leer(ne, pdf)
        slug = ('rp_' if EXTRA else 'rt_') + '%s_' % ''.join(ch if ch.isalnum() else '_' for ch in n)[:30]
        if any(f.startswith(slug) for sp_ in ('train', 'val') for f in os.listdir(os.path.join(OUT, 'images', sp_))):
            print('[rt] ya estaba', n, flush=True); continue          # retoma: planos ya hechos
        if not B or not os.path.exists(dxf): print('[rt] salteo', n); continue
        sp = 'val' if n in VAL else 'train'
        doc = ezdxf.readfile(dxf); nc = sin_circulos_marcado(doc, None if pdf else B) if (pdf and n.startswith('SAMPLE')) or not pdf else 0
        lado = np.median([max(b[2] - b[0], b[3] - b[1]) for b in B])
        ppc0 = (None if pdf else auto_ppc(doc)) or tgt / lado      # DXF nativo: igual que evaluate.py
        tope = int(np.clip(len(B) // 3, 40, TOPE))                   # tiles por escala segun cuantos componentes tiene
        from render import plan_extents
        ex = plan_extents(doc)
        for esc in (1.0, 1.6):
            s = esc * rng.uniform(.85, 1.15); ppc = ppc0 * s
            # planos enormes (Planos tableros: 28.520 x 86.348 px): se renderiza por ventanas de ~12.000 px
            L = 12000 / ppc; sol = S / ppc
            vents = [(x, y, min(ex[2], x + L), min(ex[3], y + L)) for x in np.arange(ex[0], ex[2], L - sol) for y in np.arange(ex[1], ex[3], L - sol)]
            # solo ventanas con algun componente (BSAR-1126: 350.000 x 188.000 px con 43 cajas -> cientos de ventanas vacias)
            vents = [v for v in vents if any(b[0] < v[2] and b[2] > v[0] and b[1] < v[3] and b[3] > v[1] for b in B)]
            rng.shuffle(vents); k = 0
            for win in vents:
                if k >= tope: break
                img, meta = render_doc(doc, ppc, window=win); img = darken(img)
                if img.ndim == 3: img = img[:, :, 0]
                img = img.copy(); H, W = img.shape
                px = lambda b: (*cad2px(meta, b[0], b[3]), *cad2px(meta, b[2], b[1]))
                for b in N:
                    x0, y0, x1, y1 = [int(v) for v in px(b)]; img[max(0, y0):max(0, y1 + 1), max(0, x0):max(0, x1 + 1)] = 255
                Bp = [px(b) for b in B]
                pos = [(x, y) for y in range(0, max(1, H - S) + 1, PASO) for x in range(0, max(1, W - S) + 1, PASO)]
                rng.shuffle(pos)
                for x, y in pos:
                    if k >= tope: break
                    c = np.full((S, S), 255, np.uint8); t = img[y:y + S, x:x + S]; c[:t.shape[0], :t.shape[1]] = t
                    lab = []
                    for b in Bp:
                        q = [b[0] - x, b[1] - y, b[2] - x, b[3] - y]
                        v = [max(0, q[0]), max(0, q[1]), min(S, q[2]), min(S, q[3])]
                        if v[2] <= v[0] or v[3] <= v[1]: continue
                        fr = (v[2] - v[0]) * (v[3] - v[1]) / max(1e-6, (q[2] - q[0]) * (q[3] - q[1]))
                        if fr >= .9: lab.append(v)
                        else: c[int(v[1]):int(v[3]) + 1, int(v[0]):int(v[2]) + 1] = 255
                    if not lab and (np.mean(c < 128) < .004 or rng.random() < .7): continue   # pocos tiles vacios
                    nom = ('rp_' if EXTRA else 'rt_') + '%s_%s_%d' % (''.join(ch if ch.isalnum() else '_' for ch in n)[:30], str(esc).replace('.', ''), k)
                    cv2.imwrite(os.path.join(OUT, 'images', sp, nom + '.png'), c)
                    with open(os.path.join(OUT, 'labels', sp, nom + '.txt'), 'w') as h:
                        for v in lab: h.write('0 %.6f %.6f %.6f %.6f\n' % ((v[0] + v[2]) / 2 / S, (v[1] + v[3]) / 2 / S, (v[2] - v[0]) / S, (v[3] - v[1]) / S))
                    k += 1; tot[sp][0] += 1; tot[sp][1] += len(lab)
        print('[rt] %-45s %s  cajas %4d  neutras %d  circ %d' % (n, sp, len(B), len(N), nc), flush=True)
    print('[rt] tiles/cajas', tot); open(os.path.join(OUT, 'LISTO'), 'w').write(str(tot))


if __name__ == '__main__':
    main()
