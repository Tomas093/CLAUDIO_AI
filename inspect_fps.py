import cv2
import json
import os
from eval_tsss_2 import run_eval

with open('evaluation_tsss_2/tsss_2_meta.json') as f:
    meta = json.load(f)
img = cv2.imread('evaluation_tsss_2/tsss_2_render.png')

res = run_eval(
    model_path='train-maker/models/best_componente_nano.pt',
    gt_csv_path='dxf/tsss_2_gt_completo.csv',
    render_path='evaluation_tsss_2/tsss_2_render.png',
    meta_path='evaluation_tsss_2/tsss_2_meta.json',
    out_dir='evaluation_tsss_2'
)

unmatched_dets = res[0.20]['unmatched_det']
print(f'Total unmatched detections at conf=0.20: {len(unmatched_dets)}')
os.makedirs('evaluation_tsss_2/fps', exist_ok=True)
for i, d in enumerate(unmatched_dets):
    px = d['bbox_px']
    conf = d['conf']
    crop = img[max(0, px[1]-20):min(meta['H_px'], px[3]+20), max(0, px[0]-20):min(meta['W_px'], px[2]+20)]
    filename = f'evaluation_tsss_2/fps/fp_{i:02d}_conf_{conf:.2f}.png'
    cv2.imwrite(filename, crop)
    print(f"FP {i:02d}: conf={conf:.2f}, cad=({d['xc']:.2f}, {d['yc']:.2f}), px={px}")
