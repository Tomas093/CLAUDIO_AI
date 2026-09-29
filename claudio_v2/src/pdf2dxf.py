"""PDF vectorial de CAD -> DXF, conservando trazos y texto.

Los planos que exporta el cliente suelen venir como PDF vectorial (no escaneado): adentro
estan las mismas lineas y textos del CAD original. PyMuPDF los lee y aca se vuelcan a DXF
para poder usar el mismo pipeline de inferencia (que necesita el texto para `auto_ppc`).

Uso:
    py -3 src/pdf2dxf.py <archivo.pdf|carpeta> <carpeta_salida> [--dpi-curvas 12]

Una pagina = un DXF (`<nombre>_p01.dxf`). Se saltean las paginas sin vectores (escaneadas),
avisando, porque de esas no se puede sacar un DXF util.
"""
import argparse, glob, math, os, sys
import ezdxf

try:
    import fitz  # PyMuPDF
except ImportError:
    raise SystemExit("falta PyMuPDF: py -3 -m pip install pymupdf")


def bezier(p0, p1, p2, p3, n):
    """Aproxima una curva de Bezier cubica con n segmentos."""
    out = []
    for i in range(n + 1):
        t = i / n; u = 1 - t
        x = u*u*u*p0[0] + 3*u*u*t*p1[0] + 3*u*t*t*p2[0] + t*t*t*p3[0]
        y = u*u*u*p0[1] + 3*u*u*t*p1[1] + 3*u*t*t*p2[1] + t*t*t*p3[1]
        out.append((x, y))
    return out


def pagina_a_doc(page, seg_curva=12):
    """Convierte una pagina en un ezdxf doc. Devuelve (doc, n_trazos, n_textos).

    Ojo con `page.rotation`: los planos de obra suelen venir con la pagina rotada 90/270
    (formato apaisado guardado en vertical). `get_drawings()` y `get_text()` entregan las
    coordenadas SIN rotar, asi que hay que pasarlas por `page.rotation_matrix`; si no, el
    DXF sale girado y el detector no reconoce nada (se entreno con degrees=0).
    """
    doc = ezdxf.new('R2010', setup=True)
    msp = doc.modelspace()
    M = page.rotation_matrix
    rot_pag = page.rotation or 0
    H = page.rect.height          # PDF: y hacia abajo; DXF: y hacia arriba -> y_dxf = H - y_pdf

    def P(p):
        q = fitz.Point(p[0], p[1]) * M
        return (q.x, H - q.y)

    n_tr = 0
    for d in page.get_drawings():
        for item in d['items']:
            k = item[0]
            try:
                if k == 'l':                      # linea
                    msp.add_line(P(item[1]), P(item[2])); n_tr += 1
                elif k == 'c':                    # bezier cubica
                    pts = bezier(P(item[1]), P(item[2]), P(item[3]), P(item[4]), seg_curva)
                    msp.add_lwpolyline(pts); n_tr += 1
                elif k == 're':                   # rectangulo
                    r = item[1]
                    pts = [P((r.x0, r.y0)), P((r.x1, r.y0)), P((r.x1, r.y1)), P((r.x0, r.y1))]
                    msp.add_lwpolyline(pts, close=True); n_tr += 1
                elif k == 'qu':                   # quad
                    q = item[1]
                    pts = [P((q.ul.x, q.ul.y)), P((q.ur.x, q.ur.y)), P((q.lr.x, q.lr.y)), P((q.ll.x, q.ll.y))]
                    msp.add_lwpolyline(pts, close=True); n_tr += 1
            except Exception:
                pass

    n_tx = 0
    d = page.get_text('dict')
    for blk in d.get('blocks', []):
        for line in blk.get('lines', []):
            # angulo de la linea de texto (dir es el vector de direccion)
            # El vector de direccion viene sin rotar: hay que pasarlo por la misma matriz que
            # los puntos y recien ahi calcular el angulo en el espacio DXF (con la y invertida).
            dx, dy = line.get('dir', (1, 0))
            o = fitz.Point(0, 0) * M
            v = fitz.Point(dx, dy) * M
            ang = math.degrees(math.atan2(-(v.y - o.y), v.x - o.x))
            for sp in line.get('spans', []):
                txt = sp['text'].strip()
                if not txt:
                    continue
                x0, y0, x1, y1 = sp['bbox']
                alto = sp['size']
                try:
                    t = msp.add_text(txt, dxfattribs={'height': alto, 'rotation': ang})
                    t.set_placement(P((x0, y1)))   # y1 es el borde inferior en coords PDF
                    n_tx += 1
                except Exception:
                    pass
    return doc, n_tr, n_tx


def convertir(pdf_path, out_dir, seg_curva=12):
    base = os.path.splitext(os.path.basename(pdf_path))[0]
    src = fitz.open(pdf_path)
    hechos = []
    for i, page in enumerate(src):
        doc, n_tr, n_tx = pagina_a_doc(page, seg_curva)
        if n_tr < 20:
            print('   p%02d: solo %d trazos -> parece escaneada o vacia, se saltea' % (i + 1, n_tr))
            continue
        dst = os.path.join(out_dir, '%s_p%02d.dxf' % (base, i + 1))
        try:
            doc.saveas(dst)
        except PermissionError:
            print('   p%02d: NO se pudo escribir (abierto en otro programa): %s' % (i + 1, os.path.basename(dst)))
            continue
        print('   p%02d: %6d trazos, %5d textos -> %s' % (i + 1, n_tr, n_tx, os.path.basename(dst)))
        hechos.append(dst)
    src.close()
    return hechos


def main():
    ap = argparse.ArgumentParser(description='PDF vectorial de CAD -> DXF')
    ap.add_argument('entrada', help='archivo .pdf o carpeta con .pdf')
    ap.add_argument('salida', help='carpeta de salida')
    ap.add_argument('--seg-curva', type=int, default=12, help='segmentos por curva bezier')
    a = ap.parse_args()
    pdfs = sorted(glob.glob(os.path.join(a.entrada, '*.pdf'))) if os.path.isdir(a.entrada) else [a.entrada]
    if not pdfs:
        raise SystemExit('no hay PDF en %s' % a.entrada)
    os.makedirs(a.salida, exist_ok=True)
    total = []
    for p in pdfs:
        print(os.path.basename(p))
        total += convertir(p, a.salida, a.seg_curva)
    print('\n%d DXF generados en %s' % (len(total), os.path.abspath(a.salida)))


if __name__ == '__main__':
    main()
