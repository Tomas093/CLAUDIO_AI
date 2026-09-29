import sys, json, csv, cv2, numpy as np, os
sys.path.insert(0, '/home/claude/pack/claudio_v2/src')
from numgrid import numbered_grid
from render import cad2px
def iou(a, b):
    ix = max(0, min(a[2], b[2])-max(a[0], b[0])); iy = max(0, min(a[3], b[3])-max(a[1], b[1])); I = ix*iy
    A = (a[2]-a[0])*(a[3]-a[1]); B = (b[2]-b[0])*(b[3]-b[1]); return I/(A+B-I+1e-9), I/(min(A, B)+1e-9)
O = '/mnt/user-data/outputs/revision_planos/'; os.makedirs(O, exist_ok=True)
def build(n, base=None, conf_min=.10):
    img = cv2.imread(f'{n}.png', 0); meta = json.load(open(f'{n}.json'))
    items = []   # (box, fuente, conf)
    for b in (base or []): items.append((b[:4], 'dxf', 1.0))
    for d in json.load(open(f'{n}_prop.json')):
        if d[4] < conf_min: continue
        if any(iou(d, b)[0] > .3 or iou(d, b)[1] > .6 for b, _, _ in items): continue
        items.append((d[:4], 'modelo', d[4]))
    items.sort(key=lambda it: ((it[0][1]+it[0][3])//400, it[0][0]))
    v = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR); crops = []; cols = []; idx = []
    for i, (b, src, c) in enumerate(items):
        col = (0, 0, 220) if src == 'dxf' else ((0, 160, 0) if c >= .4 else (0, 150, 255))
        x0, y0, x1, y1 = map(int, b); cv2.rectangle(v, (x0, y0), (x1, y1), col, 2)
        cv2.putText(v, str(i+1), (x0, max(10, y0-3)), cv2.FONT_HERSHEY_SIMPLEX, .45, col, 1, cv2.LINE_AA)
        pw, ph = int((x1-x0)*.4)+8, int((y1-y0)*.4)+8
        cc = cv2.cvtColor(img[max(0, y0-ph):y1+ph, max(0, x0-pw):x1+pw], cv2.COLOR_GRAY2BGR).copy()
        cv2.rectangle(cc, (min(pw, x0), min(ph, y0)), (min(pw, x0)+x1-x0, min(ph, y0)+y1-y0), col, 1)
        crops.append(cc); cols.append(col)
        idx.append(dict(id=i+1, fuente=src, conf=round(c, 3), bbox_px=[round(t, 1) for t in b]))
    json.dump(dict(plano=n, meta=meta, cajas=idx), open(O + f'{n}_cajas.json', 'w'), indent=0)
    H, W = v.shape[:2]; s = min(1.0, 7000 / max(H, W))
    cv2.imwrite(O + f'{n}_A_plano_numerado.jpg', cv2.resize(v, None, fx=s, fy=s, interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 90])
    for p in range(0, len(crops), 150):
        G = numbered_grid(crops[p:p+150], '', cell=150, cols=15, strip=16, colors=cols[p:p+150], start=p, digits=3,
                          title=f'{n}: rojo=del DXF, verde=modelo conf>=0.40, naranja=modelo conf 0.10-0.40  [{p+1}-{min(len(crops), p+150)}]')
        cv2.imwrite(O + f'{n}_B_grilla_{p//150+1}.jpg', G, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(n, len(items), sum(1 for it in items if it[1] == 'dxf'))
if __name__ == '__main__':
    build('plano5', base=json.load(open('plano5_inserts.json')))
    # vyre: GT previo (csv) como base
    meta = json.load(open('vyre.json')); base = []
    for r in csv.DictReader(open('/mnt/user-data/uploads/CLAUDIO_AI/dxf/vyre_gt_completo.csv', encoding='utf-8', errors='ignore')):
        a = cad2px(meta, float(r['x1']), float(r['y2'])); b = cad2px(meta, float(r['x2']), float(r['y1'])); base.append([a[0], a[1], b[0], b[1]])
    build('vyre', base=base)
    for n in ['sld_pabrik_gula_iec60617', '01_diagrama_unifilar_ccm', '05_diagrama_unifilar', 'substation_110_33kv_sld', 'grid_utility_to_mdp_iec60617']:
        build(n)
