import json
from statistics import median

with open("out_jijiji_v4/detecciones.json") as f:
    dets = json.load(f)

print(f"Total detections in file: {len(dets)}")

# We can group them by true/false according to size
# True ones are tall (e.g. w around 30, h around 60)
# False ones are wide (e.g. w around 57, h around 41)

true_dets = []
false_dets = []

for d in dets:
    w = d["bbox_px"][2] - d["bbox_px"][0]
    h = d["bbox_px"][3] - d["bbox_px"][1]
    if w < h: # Tall is true differential
        true_dets.append((d, w, h))
    else: # Wide is false positive termomagnetico
        false_dets.append((d, w, h))

print("\n--- TRUE DETECTIONS (interruptor_diferencial) ---")
print(f"Count: {len(true_dets)}")
confs = [d["conf"] for d, _, _ in true_dets]
if confs:
    print(f"Confidence - Min: {min(confs):.4f}, Max: {max(confs):.4f}, Mean: {sum(confs)/len(confs):.4f}")
    for idx, (d, w, h) in enumerate(true_dets):
        print(f"  {idx+1:2d}. conf={d['conf']:.4f} size=({w:.1f}x{h:.1f}) center=({d['centro_px'][0]:.1f}, {d['centro_px'][1]:.1f})")

print("\n--- FALSE POSITIVES (termomagnetico horizontal distractor) ---")
print(f"Count: {len(false_dets)}")
for idx, (d, w, h) in enumerate(false_dets):
    print(f"  {idx+1:2d}. conf={d['conf']:.4f} size=({w:.1f}x{h:.1f}) center=({d['centro_px'][0]:.1f}, {d['centro_px'][1]:.1f})")
