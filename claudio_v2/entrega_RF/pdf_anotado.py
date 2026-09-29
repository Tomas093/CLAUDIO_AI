"""PDF vectorial con las detecciones dibujadas encima (zoom sin perder calidad).

25/09, pedido de Tomas: ver las detecciones sobre los PDF de Marcelo "sin que pierda calidad y
pueda hacer zoom", con el plano en blanco y negro. Se parte del PDF ORIGINAL (no de un render):
  1. `Document.recolor(1)` pasa todo a escala de grises sin rasterizar.
  2. `--bn` lleva a negro los trazos grises y los rellenos oscuros (texto), editando los operadores
     de color de los content streams. Los rellenos claros quedan claros: pasarlos a negro taparia
     el plano (el mismo problema de los WIPEOUT/HATCH en el render).
  3. Cada deteccion va como rectangulo vectorial: rojo si conf >= --alta, naranja si no. Las
     coordenadas vienen de `detectar.py` (puntos de la pagina tal como se ve, ya rotada), asi que
     se pasan por `derotation_matrix` para dibujar en el sistema sin rotar del PDF.

    py -3 src/pdf_anotado.py plano.pdf plano_detecciones.csv salida.pdf [--bn] [--alta 0.25]
"""
import sys, re, csv, argparse
import fitz

_COLOR = re.compile(rb'(?<![\w.])(\d*\.?\d+)\s+(g|G)(?![\w])')


def a_negro(doc):
    """Operadores de gris: todo lo que no es (casi) blanco pasa a negro, trazos y rellenos.

    Con un corte mas bajo para rellenos (0,6) los textos que eran amarillos o celestes quedaban
    en gris 0,9: invisibles. Umbral `BN_CORTE` (0,995; con 0,97 quedaban trazos amarillos en `.9728 G`): lo de arriba se considera fondo blanco.
    """
    corte = float(__import__('os').environ.get('BN_CORTE', '0.995'))
    def sub(m):
        v = float(m.group(1)); op = m.group(2)
        if v < corte:
            return b'0 ' + op
        return m.group(0)
    hechos = set()
    for page in doc:
        xrefs = list(page.get_contents())
        xrefs += [x[0] for x in page.get_xobjects()]
        for xr in xrefs:
            if xr in hechos or not doc.xref_is_stream(xr):
                continue
            hechos.add(xr)
            doc.update_stream(xr, _COLOR.sub(sub, doc.xref_stream(xr)))


def anotar(pdf, csv_det, salida, bn=False, alta=.25, pagina=0):
    doc = fitz.open(pdf)
    doc.recolor(1)
    if bn:
        a_negro(doc)
    page = doc[pagina]
    m = page.derotation_matrix
    ancho = max(.25, min(page.rect.width, page.rect.height) / 1500.)
    n = [0, 0]
    for r in csv.DictReader(open(csv_det, encoding='utf-8')):
        c = float(r['conf'])
        # con el plano girado (`giro` de detectar.py) las esquinas pueden venir invertidas
        x1, y1, x2, y2 = float(r['x1']), float(r['y1']), float(r['x2']), float(r['y2'])
        rect = fitz.Rect(min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)) * m
        rect.normalize()
        col = (1, 0, 0) if c >= alta else (1, .55, 0)
        page.draw_rect(rect, color=col, width=ancho, overlay=True)
        n[c >= alta] += 1
    doc.save(salida, garbage=3, deflate=True)
    print('%s: %d cajas (>=%.2f: %d, debajo: %d) -> %s' % (pdf, sum(n), alta, n[1], n[0], salida))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('pdf'); ap.add_argument('csv'); ap.add_argument('salida')
    ap.add_argument('--bn', action='store_true'); ap.add_argument('--alta', type=float, default=.25)
    a = ap.parse_args()
    anotar(a.pdf, a.csv, a.salida, a.bn, a.alta)
