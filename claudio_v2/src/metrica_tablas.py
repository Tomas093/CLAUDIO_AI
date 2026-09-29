"""Cuenta detecciones que caen sobre celdas de la planilla de circuitos (OBJETIVOS.md §0).

El plan Q gano precision con las ternas R/S/T pero empezo a marcar como componente las celdas
de la planilla (`C7`, `C21`, `C23`...). Como el matching es 1-a-1 global, esos falsos positivos
le roban asignaciones a los componentes de verdad: asi "perdia" 10 CONTACTOR. El total de FP no
lo muestra porque se diluye entre miles.

Esta es la metrica propia de ese defecto: se ubican los textos de celda en el DXF y se cuenta
cuantas detecciones caen encima.

    py -3 src/metrica_tablas.py V3_v9_N_marcelo V3_v12_Q_marcelo
"""
import sys, os, csv, re, glob
import ezdxf

sys.path.insert(0, os.path.dirname(__file__))
from paths import BASE, WORK

# lo que hay adentro de una celda de la planilla de circuitos
CELDA = re.compile(r'^\s*(C\d{1,3}|RES|RESERVA|N|CIRCUITO|FASES|DESTINO|FUNCION|TIPO DE CABLE|P TOT.*)\s*$', re.I)

DXFS = {}
for p in glob.glob(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'test_marcelo', '*.dxf')):
    DXFS[os.path.basename(p)[:-4]] = p


def textos_de_celda(dxf):
    """Textos que parecen celda de planilla, incluidos los de adentro de los bloques.

    Ojo: iterar solo el modelspace da CERO en LU-UN-01. Las celdas de la planilla de circuitos
    son texto de un bloque, no del modelspace, y sin explotar los INSERT la metrica mide 0
    siempre y parece que el defecto no existe.
    """
    out = []
    doc = ezdxf.readfile(dxf)

    def ver(e, prof=0):
        t = e.dxftype()
        try:
            if t in ('TEXT', 'MTEXT', 'ATTRIB'):
                s_ = e.dxf.text if t in ('TEXT', 'ATTRIB') else e.text
                h = float(getattr(e.dxf, 'height', 0) or 0.2)
                p_ = e.dxf.insert
                if CELDA.match(str(s_).replace('\P', ' ')):
                    out.append((p_.x, p_.y, h))
            elif t == 'INSERT' and prof < 3:
                for a in e.attribs:
                    ver(a, prof + 1)
                for ve in e.virtual_entities():
                    ver(ve, prof + 1)
        except Exception:
            pass

    for e in doc.modelspace():
        ver(e)
    return out


def main():
    tags = sys.argv[1:] or ['V3_v9_N_marcelo']
    cache = {}
    print('%-26s %-22s %8s %8s %7s' % ('tag', 'plano', 'dets', 'en celda', '%'))
    for tag in tags:
        for csvf in sorted(glob.glob(os.path.join(WORK, 'eval', tag, '*_detecciones.csv'))):
            plano = os.path.basename(csvf)[:-len('_detecciones.csv')]
            if plano not in DXFS:
                continue
            if plano not in cache:
                cache[plano] = textos_de_celda(DXFS[plano])
            cel = cache[plano]
            dets = [r for r in csv.DictReader(open(csvf)) if float(r['conf']) >= 0.25]
            n = 0
            for r in dets:
                cx = (float(r['x1']) + float(r['x2'])) / 2
                cy = (float(r['y1']) + float(r['y2'])) / 2
                for (tx, ty, h) in cel:
                    if abs(cx - tx) < h * 4 and abs(cy - ty) < h * 1.6:
                        n += 1
                        break
            print('%-26s %-22s %8d %8d %6.1f%%' % (tag[:26], plano[:22], len(dets), n,
                                                   100. * n / max(1, len(dets))))


if __name__ == '__main__':
    main()
