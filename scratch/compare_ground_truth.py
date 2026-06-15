import json
import math

def distance(x1, y1, x2, y2):
    return math.sqrt((x1-x2)**2 + (y1-y2)**2)

with open(r'c:\Users\Tomas\Documents\LAB3\CLAUDIO_AI\pipeline_out\dif_ground_truth\ground_truth.json', 'r') as f:
    gt_data = json.load(f)

with open(r'c:\Users\Tomas\Documents\LAB3\CLAUDIO_AI\pipeline_out\dif_jijiji_phase2_z10_o80\detecciones.json', 'r') as f:
    pred_data = json.load(f)

print(f"Ground Truth count: {len(gt_data)}")
print(f"Predictions count: {len(pred_data)}")

# Match predictions to GT based on CAD coordinates distance
threshold_cad = 2.0  # Allow some CAD unit tolerance
matched_gt = set()
matched_pred = set()

for p_i, p in enumerate(pred_data):
    best_gt_i = -1
    min_dist = float('inf')
    for g_i, g in enumerate(gt_data):
        d = distance(p['x_cad'], p['y_cad'], g['x_cad'], g['y_cad'])
        if d < min_dist:
            min_dist = d
            best_gt_i = g_i
    
    if min_dist <= threshold_cad:
        matched_gt.add(best_gt_i)
        matched_pred.add(p_i)

true_positives = len(matched_pred)
false_positives = len(pred_data) - true_positives
false_negatives = len(gt_data) - len(matched_gt)

print(f"True Positives (Correct): {true_positives}")
print(f"False Positives (Invented): {false_positives}")
print(f"False Negatives (Missed): {false_negatives}")

if false_positives > 0:
    print("\n[False Positives]")
    for i, p in enumerate(pred_data):
        if i not in matched_pred:
            print(f"  - Conf: {p['conf']:.3f} en CAD ({p['x_cad']:.2f}, {p['y_cad']:.2f})")

if false_negatives > 0:
    print("\n[False Negatives]")
    for i, g in enumerate(gt_data):
        if i not in matched_gt:
            print(f"  - Perdido en CAD ({g['x_cad']:.2f}, {g['y_cad']:.2f})")
