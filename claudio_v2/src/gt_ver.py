"""Render de un plano de test con el GT dibujado (30/09, revision GT v8). Guarda PNG grande + meta.
    py -3 src/gt_ver.py <plano> <csv_gt> <salida.png> [escala]
"""
import sys, os, csv, json, pickle
sys.path.insert(0, os.path.dirname(__file__))
import cv2, ezdxf
from paths import BASE
from render import render_doc, cad2px
from scale import auto_ppc
pl, gtp, sal = sys.argv[1:4]; esc = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
import gt_v8
dxf = [d for n, d, g in gt_v8.planos() if n == pl][0]
doc = ezdxf.readfile(os.path.join(BASE, dxf)); ppc = auto_ppc(doc) * esc
img, meta = render_doc(doc, ppc)
pickle.dump(meta, open(sal[:-4] + '_meta.pkl', 'wb'))
cv2.imwrite(sal[:-4] + '_limpio.png', img)
im = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img.copy()
for r in csv.DictReader(open(gtp, encoding='utf-8', errors='ignore')):
    if not r.get('x1'): continue
    x0, y0 = cad2px(meta, float(r['x1']), float(r['y2'])); x1, y1 = cad2px(meta, float(r['x2']), float(r['y1']))
    col = (0, 0, 255) if r['block_name'].startswith('AUDIT7') else (255, 0, 255)
    cv2.rectangle(im, (int(x0), int(y0)), (int(x1), int(y1)), col, 2)
cv2.imwrite(sal, im); print(im.shape, ppc)
