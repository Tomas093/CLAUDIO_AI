"""PDF VECTORIAL de un plano de test con el GT dibujado (30/09; mismo metodo que los _con_GT_v7.pdf).

No se rasteriza nada: se agregan las cajas del GT como LWPOLYLINE al modelspace del DXF (capa GT_BOX, magenta,
grosor 0,5 mm; neutras en capa GT_NEUTRA, gris) y se dibuja todo con el backend PyMuPDF de ezdxf. El PDF queda
con lineas y texto vectoriales: se puede hacer zoom sin perder calidad. Escala: el lado mayor queda en ~4500 mm.
    py -3 src/gt_pdf.py <plano> <salida_dir>
"""
import sys, os, csv, time
sys.path.insert(0, os.path.dirname(__file__))
import ezdxf
from ezdxf.addons.drawing import Frontend, RenderContext, layout, config
from ezdxf.addons.drawing.pymupdf import PyMuPdfBackend
from ezdxf import bbox
from paths import BASE
import gt_v8

pl, out = sys.argv[1], sys.argv[2]
dxf, gtp = [(d, os.path.join(BASE, g)) for n, d, g in gt_v8.planos() if n == pl][0]
cajas = lambda p: [tuple(float(r[k]) for k in ('x1', 'y1', 'x2', 'y2')) for r in csv.DictReader(open(p, encoding='utf-8')) if r.get('x1')]
t = time.time()
doc = ezdxf.readfile(os.path.join(BASE, dxf)); msp = doc.modelspace()
ext = bbox.extents(msp); w, h = ext.size.x, ext.size.y
for capa, col, p in (('GT_BOX', 6, gtp[:-4] + '_v8.csv'), ('GT_NEUTRA', 8, gtp[:-4] + '_v8_neutras.csv')):
    if capa not in doc.layers: doc.layers.add(capa, color=col)
    for x1, y1, x2, y2 in cajas(p):
        msp.add_lwpolyline([(x1, y1), (x2, y1), (x2, y2), (x1, y2)], close=True, dxfattribs={'layer': capa, 'color': col, 'lineweight': 50})
cfg = config.Configuration(background_policy=config.BackgroundPolicy.WHITE, color_policy=config.ColorPolicy.COLOR)
be = PyMuPdfBackend()
Frontend(RenderContext(doc), be, config=cfg).draw_layout(msp, finalize=True)
esc = max(1.0, min(4500 / max(w, h), 60))           # mm de papel por unidad CAD
pag = layout.Page(0, 0, layout.Units.mm, margins=layout.Margins.all(2))
os.makedirs(out, exist_ok=True)
nombre = os.path.join(out, pl.split(' ')[0] + '_con_GT_v8.pdf')
open(nombre, 'wb').write(be.get_pdf_bytes(pag, settings=layout.Settings(scale=esc)))
print(pl, '->', nombre, round(time.time() - t), 's', flush=True)
