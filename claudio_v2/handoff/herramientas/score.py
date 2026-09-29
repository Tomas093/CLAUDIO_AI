import sys, json, os
sys.path.insert(0, '/home/claude/pack/claudio_v2/src'); os.environ.setdefault('CLAUDIO_BASE', '/mnt/user-data/uploads/CLAUDIO_AI')
from evaluate import PLANOS, load_gt, match
from paths import BASE
def nms(D):
    D = sorted(D, key=lambda d: -d[4]); keep = []
    for d in D:
        ok = True
        for k in keep:
            ix = max(0, min(d[2], k[2]) - max(d[0], k[0])); iy = max(0, min(d[3], k[3]) - max(d[1], k[1])); I = ix*iy
            A = (d[2]-d[0])*(d[3]-d[1]); B = (k[2]-k[0])*(k[3]-k[1])
            if I/(A+B-I+1e-12) > float(os.environ.get("IOU",".45")) or (I/(min(A, B)+1e-12) > float(os.environ.get("IOS",".7")) and max(A, B) < float(os.environ.get("AR","4"))*min(A, B)): ok = False; break
        if ok: keep.append(d)
    return keep
R = json.load(open(sys.argv[1])); combos = [c.split('+') for c in sys.argv[2:]]
for combo in combos:
    print('== escalas', combo); tot = {}
    for name, _, gtp in PLANOS:
        if name not in R or not all(s in R[name] for s in combo): continue
        gt = load_gt(name, os.path.join(BASE, gtp)); D = nms([d for s in combo for d in R[name][s]]); row = []
        for th in (.25, .2, .15, .1, .05):
            tp, fn, fp = match(gt, [d for d in D if d[4] >= th]); row.append(f'{th}:fn{len(fn)}/fp{len(fp)}')
            t = tot.setdefault(th, [0, 0, 0]); t[0] += len(gt); t[1] += len(fn); t[2] += len(fp)
        print(f'  {name:9s} gt{len(gt):4d}', ' '.join(row))
    print('  TOTAL', ' '.join(f'{th}: R={1-v[1]/v[0]:.3f} fn{v[1]} fp{v[2]}' for th, v in tot.items()))
