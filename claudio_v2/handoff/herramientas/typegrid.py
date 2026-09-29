import sys, json, glob, cv2, ezdxf, numpy as np, os
sys.path.insert(0, '/home/claude/pack/claudio_v2/src')
from render import render_doc, cad2px
from postproc import darken
from numgrid import numbered_grid
seen = {}; order = []
files = [j for j in sorted(glob.glob('render/*.json')) if len(json.load(open(j))) >= 3]
for j in files:
    for b in json.load(open(j)):
        t = b[4].split('/', 3)[-1] if b[4].startswith('embed://') else b[4]
        if t not in seen: seen[t] = (j, b); order.append(t)
crops = []; cache = {}
for t in order:
    j, b = seen[t]
    if j not in cache:
        doc = ezdxf.readfile(j[:-5] + '.dxf'); img, meta = render_doc(doc, 3.0); cache = {j: (darken(img), meta)}
    img, meta = cache[j]
    p0 = cad2px(meta, b[0], b[3]); p1 = cad2px(meta, b[2], b[1]); x0, y0, x1, y1 = int(p0[0]), int(p0[1]), int(p1[0]), int(p1[1])
    pw, ph = int((x1-x0)*.4)+10, int((y1-y0)*.4)+10
    c = cv2.cvtColor(img[max(0,y0-ph):y1+ph, max(0,x0-pw):x1+pw], cv2.COLOR_GRAY2BGR).copy()
    cv2.rectangle(c, (min(pw,x0), min(ph,y0)), (min(pw,x0)+x1-x0, min(ph,y0)+y1-y0), (0,0,220), 2)
    crops.append(c)
json.dump(order, open('render/tipos.json', 'w'), indent=0)
O = '/mnt/user-data/outputs/revision_planos/'
for p in range(0, len(crops), 150):
    G = numbered_grid(crops[p:p+150], '', cell=150, cols=15, strip=16, colors=[(0,0,220)]*len(crops[p:p+150]), start=p, digits=3,
                      title=f'QElectroTech: un ejemplo por tipo de elemento marcado como componente [{p+1}-{min(len(crops),p+150)}]')
    cv2.imwrite(O + f'qet_tipos_{p//150+1}.jpg', G, [cv2.IMWRITE_JPEG_QUALITY, 88])
print(len(order), len(files))
