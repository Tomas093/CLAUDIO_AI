# -*- coding: utf-8 -*-
"""Ciñe las cajas del GT al simbolo, sacandoles el texto de atributo del bloque.

20/09. El GT de LU-UN-01 y nyw-un-01 calculo la caja de cada bloque INCLUYENDO sus ATTRIB
(`12A`, `AC3`, `S1`). El proyecto ya tenia definido que el GT correcto excluye
ATTRIB/ATTDEF/TEXT/MTEXT, pero en estos dos planos no se aplico. Medido sobre LU-UN-01:

    S-M-0-A      area GT / real  3.21x
    CONTACTOR                    2.35x
    SECC-BC-FUS                  2.31x
    TM-DIN (490 cajas)           1.46x
    MEDIDOR (sin atributos)      1.00x

Lo grave no es el tamano sino que **el centro queda corrido hacia el texto** (0.263 unidades en
CONTACTOR, sobre un simbolo de 0.569 de ancho). El matching de `evaluate.py` asigna por
distancia de centros, asi que un modelo que cine bien la caja al simbolo queda lejos del centro
del GT y se cuenta como fallo: **penaliza justo al que encuadra mejor**.

No se regenera el GT desde cero: se **corrige la caja** de las filas que corresponden a un
INSERT del DXF, dejando intactas las demas (geometria suelta y las 221 que agrego la auditoria).

  py -3 src/cenir_gt.py [--aplicar]

Escribe `<nombre>_v3.csv`. Sin `--aplicar` solo mide cuanto cambiaria.
"""
import os, sys, csv, collections, statistics, shutil

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import BASE
import ezdxf
from ezdxf import bbox as ebbox

SKIP = {'ATTRIB', 'ATTDEF', 'TEXT', 'MTEXT'}

# solo estos dos planos tienen el problema; los demas se revisaron y estan bien
PLANOS = [
    ('LU-UN-01', 'claudio_v2/data/test_marcelo/LU-UN-01 Esquemas Unifilares.dxf',
     'claudio_v2/data/test_marcelo/LU-UN-01 Esquemas Unifilares_gt.csv'),
    ('nyw-un-01', 'claudio_v2/data/test_marcelo/nyw-un-01 esquemas unifilares.dxf',
     'claudio_v2/data/test_marcelo/nyw-un-01 esquemas unifilares_gt.csv'),
]


def caja_sin_texto(e):
    try:
        vs = [v for v in e.virtual_entities() if v.dxftype() not in SKIP]
    except Exception:
        return None
    if not vs:
        return None
    b = ebbox.extents(vs, fast=True)
    return (b.extmin.x, b.extmin.y, b.extmax.x, b.extmax.y)


def procesar(nom, dxf, gtp, aplicar, sufijo='_v3'):
    fd, fg = os.path.join(BASE, dxf), os.path.join(BASE, gtp)
    if not (os.path.exists(fd) and os.path.exists(fg)):
        print('[aviso] falta %s' % nom)
        return
    doc = ezdxf.readfile(fd)
    msp = doc.modelspace()
    porb = collections.defaultdict(list)
    for e in msp.query('INSERT'):
        c = caja_sin_texto(e)
        if c:
            porb[e.dxf.name].append(c)
    with open(fg, newline='', encoding='utf-8', errors='ignore') as f:
        rd = csv.DictReader(f)
        campos = list(rd.fieldnames or [])
        filas = list(rd)
    cambios = 0
    desv = []
    for r in filas:
        n = r.get('block_name', '')
        if n not in porb or not r.get('x1'):
            continue
        gx, gy = float(r['x_cad']), float(r['y_cad'])
        # el INSERT mas cercano de ese mismo bloque
        best = min(porb[n], key=lambda c: ((c[0]+c[2])/2. - gx)**2 + ((c[1]+c[3])/2. - gy)**2)
        cx, cy = (best[0]+best[2])/2., (best[1]+best[3])/2.
        if ((cx - gx)**2 + (cy - gy)**2) ** .5 > 1.5:
            continue          # no se pudo emparejar con confianza: se deja como esta
        ax, ay, bx, by = float(r['x1']), float(r['y1']), float(r['x2']), float(r['y2'])
        d = (((ax+bx)/2. - cx)**2 + ((ay+by)/2. - cy)**2) ** .5
        if d > 1e-4 or abs((bx-ax) - (best[2]-best[0])) > 1e-4:
            desv.append(d)
            cambios += 1
            r['x1'], r['y1'] = '%.4f' % best[0], '%.4f' % best[1]
            r['x2'], r['y2'] = '%.4f' % best[2], '%.4f' % best[3]
            r['x_cad'], r['y_cad'] = '%.4f' % cx, '%.4f' % cy
    print('  %-11s %5d filas | %4d cajas cenidas | desvio del centro: med %.3f max %.3f' %
          (nom, len(filas), cambios,
           statistics.median(desv) if desv else 0, max(desv) if desv else 0))
    if aplicar:
        sal = fg[:-4] + sufijo + '.csv'
        with open(sal, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=campos)
            w.writeheader()
            w.writerows(filas)
        print('             -> %s' % os.path.basename(sal))
    return cambios


def main():
    aplicar = '--aplicar' in sys.argv
    print('ciniendo las cajas del GT (sacando ATTRIB/ATTDEF/TEXT/MTEXT):')
    tot = 0
    for nom, dxf, gtp in PLANOS:
        # se cine sobre la version _v2 (la que ya tiene los 221 componentes que agrego la
        # auditoria), asi el _v3 queda con las DOS correcciones: completo y cenido.
        v2 = gtp[:-4] + '_v2.csv'
        base = v2 if os.path.exists(os.path.join(BASE, v2)) else gtp
        n = procesar(nom, dxf, base, aplicar) or 0
        tot += n
        if aplicar:
            # el nombre que deja `procesar` depende de la base; se normaliza a <orig>_v3.csv
            hecho = os.path.join(BASE, base[:-4] + '_v3.csv')
            quiero = os.path.join(BASE, gtp[:-4] + '_v3.csv')
            if hecho != quiero and os.path.exists(hecho):
                shutil.move(hecho, quiero)
                print('             -> %s' % os.path.basename(quiero))
    print('\ntotal de cajas corregidas: %d' % tot)
    if not aplicar:
        print('(sin --aplicar no se escribe nada)')


if __name__ == '__main__':
    main()
