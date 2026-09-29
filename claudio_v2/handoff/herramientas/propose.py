import sys, json, csv, cv2, numpy as np, os
sys.path.insert(0, '/home/claude/pack/claudio_v2/src')
from numgrid import numbered_grid
from render import cad2px
from ultralytics import YOLO
sys.argv += []
W = sys.argv[1]; names = sys.argv[2].split(','); conf = float(sys.argv[3]) if len(sys.argv) > 3 else 0.10
m = YOLO(W)
def infer(img, tile=640, stride=320):
    H, Wd = img.shape; dets = []
    ys = list(range(0, max(1, H-tile)+1, stride)); xs = list(range(0, max(1, Wd-tile)+1, stride))
    if ys[-1]+tile < H: ys.append(max(0, H-tile))
    if xs[-1]+tile < Wd: xs.append(max(0, Wd-tile))
    for y in ys:
        batch, offs = [], []
        for x in xs:
            t = img[y:y+tile, x:x+tile]
            if t.shape != (tile, tile): t = cv2.copyMakeBorder(t, 0, tile-t.shape[0], 0, tile-t.shape[1], cv2.BORDER_CONSTANT, value=255)
            if (t < 128).mean() < .0005: continue
            batch.append(cv2.cvtColor(t, cv2.COLOR_GRAY2BGR)); offs.append((x, y))
        if not batch: continue
        for r, (ox, oy) in zip(m.predict(batch, imgsz=tile, conf=conf, verbose=False), offs):
            for b, c in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist()): dets.append([b[0]+ox, b[1]+oy, b[2]+ox, b[3]+oy, c])
    dets.sort(key=lambda d: -d[4]); keep = []
    for d in dets:
        ok = True
        for k in keep:
            ix = max(0, min(d[2], k[2])-max(d[0], k[0])); iy = max(0, min(d[3], k[3])-max(d[1], k[1])); I = ix*iy
            A = (d[2]-d[0])*(d[3]-d[1]); B = (k[2]-k[0])*(k[3]-k[1])
            if I/(A+B-I+1e-9) > .45 or I/(min(A, B)+1e-9) > .7: ok = False; break
        if ok: keep.append(d)
    return keep
for n in names:
    img = cv2.imread(f'{n}.png', 0); dets = infer(img)
    json.dump(dets, open(f'{n}_prop.json', 'w'))
    print(n, len(dets), flush=True)
