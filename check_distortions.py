from pathlib import Path
from collections import defaultdict
import numpy as np

d = Path('train-maker/dataset_unified_componente/train')
aspects_by_type = defaultdict(list)
w_by_type = defaultdict(list)
h_by_type = defaultdict(list)

for lbl in (d / 'labels').glob('*.txt'):
    prefix = lbl.name.split('_')[0]
    with open(lbl, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            pts = line.strip().split()
            if len(pts) >= 5:
                try:
                    w, h = float(pts[3]), float(pts[4])
                    aspects_by_type[prefix].append(w / (h + 1e-6))
                    w_by_type[prefix].append(w)
                    h_by_type[prefix].append(h)
                except Exception:
                    pass

print(f"{'Tipo':<15} | {'Total Cajas':<11} | {'W prom':<8} | {'H prom':<8} | {'Aspect Ratio (W/H)':<22} | {'Distorsionadas (W/H > 3 o < 0.2)'}")
print('-' * 95)
for k in sorted(aspects_by_type.keys()):
    arr = np.array(aspects_by_type[k])
    w_arr = np.array(w_by_type[k])
    h_arr = np.array(h_by_type[k])
    if len(arr) == 0:
        continue
    distorted = np.sum((arr > 3.0) | (arr < 0.2))
    print(f"{k:<15} | {len(arr):<11} | {w_arr.mean():.3f}    | {h_arr.mean():.3f}    | med={np.median(arr):.2f} (std={arr.std():.2f})     | {distorted} ({distorted/len(arr)*100:.1f}%)")
