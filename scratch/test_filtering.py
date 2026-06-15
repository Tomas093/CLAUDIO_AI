import json
from statistics import median

with open("out_jijiji_v4/detecciones.json") as f:
    dets = json.load(f)

print(f"Total detections: {len(dets)}")

widths = [d["bbox_px"][2] - d["bbox_px"][0] for d in dets]
heights = [d["bbox_px"][3] - d["bbox_px"][1] for d in dets]
med_w = median(widths)
med_h = median(heights)

print(f"Median width: {med_w:.2f}")
print(f"Median height: {med_h:.2f}")

factor = 1.5
sobrevivientes = []
descartados = []

for d in dets:
    w = d["bbox_px"][2] - d["bbox_px"][0]
    h = d["bbox_px"][3] - d["bbox_px"][1]
    if (w > med_w * factor or w < med_w / factor or
        h > med_h * factor or h < med_h / factor):
        descartados.append((d, w, h))
    else:
        sobrevivientes.append(d)

print(f"\nSurvival count with factor={factor}: {len(sobrevivientes)}")
confs = [d["conf"] for d in sobrevivientes]
print(f"Confidence stats of survivors:")
print(f"  Min: {min(confs):.4f}")
print(f"  Max: {max(confs):.4f}")
print(f"  Mean: {sum(confs)/len(confs):.4f}")

print(f"\nDiscarded count: {len(descartados)}")
for idx, (d, w, h) in enumerate(descartados):
    print(f"  {idx+1}. conf={d['conf']:.4f} size=({w:.1f}x{h:.1f})")
