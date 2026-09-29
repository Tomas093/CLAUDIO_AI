"""Evalua pesos con escalas de render [1.0] y [1.0,+S] fusionadas. Guarda detecciones en CAD por escala."""
import sys, os, json, ezdxf, numpy as np
sys.path.insert(0, '/home/claude/pack/claudio_v2/src')
os.environ.setdefault('CLAUDIO_BASE', '/mnt/user-data/uploads/CLAUDIO_AI')
from ultralytics import YOLO
from render import render_doc, px2cad
from scale import auto_ppc
from postproc import darken
from evaluate import PLANOS, infer, load_gt
from paths import BASE
w, tag = sys.argv[1], sys.argv[2]; scales = [float(s) for s in sys.argv[3].split(',')]
m = YOLO(w); out = {}
for name, dxf, gtp in PLANOS:
    doc = ezdxf.readfile(os.path.join(BASE, dxf)); p0 = auto_ppc(doc); out[name] = {}
    for s in scales:
        img, meta = render_doc(doc, p0 * s); img = darken(img)
        dets = infer(m, img, 0.05); dc = []
        for d in dets:
            xa, ya = px2cad(meta, d[0], d[3]); xb, yb = px2cad(meta, d[2], d[1]); dc.append([xa, ya, xb, yb, d[4]])
        out[name][str(s)] = dc; print(name, s, img.shape, len(dc), flush=True)
    json.dump(out, open(f'/home/claude/exp/{tag}.json', 'w'))
