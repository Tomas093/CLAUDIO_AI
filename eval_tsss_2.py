import os
import sys
import csv
import json
import time
import math
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from vector_inference import (
    nms_agnostico_clase_cad,
    eliminar_anidadas_cad,
    nms_distancia_cad
)

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

def slice_coords(H, W, slice_size=640, overlap=0.80):
    step = int(round(slice_size * (1.0 - overlap)))
    slices = []
    y = 0
    while y < H:
        x = 0
        y2 = min(y + slice_size, H)
        y1 = max(0, y2 - slice_size)
        while x < W:
            x2 = min(x + slice_size, W)
            x1 = max(0, x2 - slice_size)
            slices.append((x1, y1, x2, y2))
            if x2 >= W:
                break
            x += step
        if y2 >= H:
            break
        y += step
    return slices

def run_eval(model_path, gt_csv_path, render_path, meta_path, out_dir, dist_tol=1.0, conf_sweep=[0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]):
    os.makedirs(out_dir, exist_ok=True)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"Loading model {model_path} on {device}...")
    model = YOLO(model_path)

    print(f"Loading image {render_path}...")
    img_bgr = cv2.imread(render_path)
    with open(meta_path, 'r', encoding='utf-8') as f:
        meta = json.load(f)

    W_px, H_px = img_bgr.shape[1], img_bgr.shape[0]
    x_min, y_min = meta['x_min_cad'], meta['y_min_cad']
    x_max, y_max = meta['x_max_cad'], meta['y_max_cad']
    W_cad, H_cad = meta['W_cad'], meta['H_cad']
    px_per_cad = meta['px_per_cad']

    gt_list = []
    with open(gt_csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            gt_dict = {
                'name': row.get('block_name', row.get('tipo', 'comp')),
                'xc': float(row['x_cad']),
                'yc': float(row['y_cad'])
            }
            if 'x1' in row and 'y1' in row and 'x2' in row and 'y2' in row:
                try:
                    x1, y1 = float(row['x1']), float(row['y1'])
                    x2, y2 = float(row['x2']), float(row['y2'])
                    if x1 != x2 and y1 != y2:
                        gt_dict['xc'] = (x1 + x2) / 2.0
                        gt_dict['yc'] = (y1 + y2) / 2.0
                    gt_dict['x1'] = x1
                    gt_dict['y1'] = y1
                    gt_dict['x2'] = x2
                    gt_dict['y2'] = y2
                except (ValueError, TypeError):
                    pass
            gt_list.append(gt_dict)

    pad = 320
    slice_size = 640
    padded_img = cv2.copyMakeBorder(img_bgr, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    pad_h, pad_w = padded_img.shape[:2]
    slices = slice_coords(pad_h, pad_w, slice_size=slice_size, overlap=0.80)

    print(f"Slicing: {len(slices)} tiles (img={W_px}x{H_px}, pad={pad}). Running batch inference...")
    raw_dets = []
    batch_size = 32
    t0 = time.time()
    for i in range(0, len(slices), batch_size):
        b_slices = slices[i:i+batch_size]
        b_imgs = [padded_img[y1:y2, x1:x2] for (x1, y1, x2, y2) in b_slices]
        results = model(b_imgs, verbose=False, conf=0.01, device=device)
        for s_idx, res in enumerate(results):
            x1_tile, y1_tile, _, _ = b_slices[s_idx]
            if res.boxes:
                xyxy = res.boxes.xyxy.cpu().numpy()
                confs = res.boxes.conf.cpu().numpy()
                for b, c in zip(xyxy, confs):
                    px_x1 = (x1_tile + b[0]) - pad
                    px_y1 = (y1_tile + b[1]) - pad
                    px_x2 = (x1_tile + b[2]) - pad
                    px_y2 = (y1_tile + b[3]) - pad
                    if px_x2 <= 0 or px_x1 >= W_px or px_y2 <= 0 or px_y1 >= H_px:
                        continue
                    px_x1 = max(0, min(W_px, px_x1))
                    px_x2 = max(0, min(W_px, px_x2))
                    px_y1 = max(0, min(H_px, px_y1))
                    px_y2 = max(0, min(H_px, px_y2))

                    cad_x1 = x_min + (px_x1 / W_px) * W_cad
                    cad_x2 = x_min + (px_x2 / W_px) * W_cad
                    cad_y1 = y_max - (px_y2 / H_px) * H_cad
                    cad_y2 = y_max - (px_y1 / H_px) * H_cad
                    cx = (cad_x1 + cad_x2) / 2.0
                    cy = (cad_y1 + cad_y2) / 2.0

                    raw_dets.append({
                        'clase': 'componente',
                        'conf': float(c),
                        'bbox_cad': [cad_x1, min(cad_y1, cad_y2), cad_x2, max(cad_y1, cad_y2)],
                        'xc': cx, 'yc': cy,
                        'bbox_px': [int(px_x1), int(px_y1), int(px_x2), int(px_y2)]
                    })

    t_inf = time.time() - t0
    print(f"Inference completed in {t_inf:.1f}s. Raw dets: {len(raw_dets)}")

    print(f"\n{'Conf':>6} | {'Dets':>6} | {'GT':>6} | {'TP':>6} | {'FN':>6} | {'FP':>6} | {'Recall':>8} | {'Precision':>10} | {'F1':>8}")
    print('-' * 78)

    results_by_conf = {}
    for th in conf_sweep:
        dets_th = [d for d in raw_dets if d['conf'] >= th]
        dets_th = nms_agnostico_clase_cad(dets_th, iou_thresh=0.45)
        dets_th = eliminar_anidadas_cad(dets_th, ios_thresh=0.60)
        dets_th = nms_distancia_cad(dets_th, d_min=0.45)

        candidates = []
        for g_idx, g in enumerate(gt_list):
            for d_idx, d in enumerate(dets_th):
                dist = math.hypot(g['xc'] - d['xc'], g['yc'] - d['yc'])
                b = d['bbox_cad']
                inside = (b[0] <= g['xc'] <= b[2] and b[1] <= g['yc'] <= b[3])
                if dist <= dist_tol or inside:
                    candidates.append((dist, g_idx, d_idx))

        candidates.sort(key=lambda x: x[0])
        matched_gt = set()
        matched_det = set()
        for dist, g_idx, d_idx in candidates:
            if g_idx not in matched_gt and d_idx not in matched_det:
                matched_gt.add(g_idx)
                matched_det.add(d_idx)

        tp = len(matched_gt)
        fn = len(gt_list) - tp
        fp = len(dets_th) - len(matched_det)
        rec = (tp / len(gt_list)) * 100 if gt_list else 100.0
        prec = (tp / (tp + fp)) * 100 if (tp + fp) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        unmatched_gt_items = [gt_list[i] for i in range(len(gt_list)) if i not in matched_gt]
        unmatched_det_items = [dets_th[i] for i in range(len(dets_th)) if i not in matched_det]

        results_by_conf[th] = {
            'tp': tp, 'fn': fn, 'fp': fp,
            'recall': rec, 'precision': prec, 'f1': f1,
            'dets': len(dets_th), 'gt': len(gt_list),
            'unmatched_gt': unmatched_gt_items,
            'unmatched_det': unmatched_det_items,
            'dets_list': dets_th,
            'matched_det': matched_det
        }
        print(f"{th:6.2f} | {len(dets_th):6d} | {len(gt_list):6d} | {tp:6d} | {fn:6d} | {fp:6d} | {rec:7.1f}% | {prec:9.1f}% | {f1/100:8.3f}")

    # Breakdown by block name at conf=0.15 and conf=0.20
    for ref_th in [0.15, 0.20]:
        res = results_by_conf[ref_th]
        print(f"\n================ BREAKDOWN AT CONF = {ref_th} ================")
        gt_by_type = defaultdict(int)
        for g in gt_list:
            gt_by_type[g['name']] += 1

        fn_by_type = defaultdict(int)
        for g in res['unmatched_gt']:
            fn_by_type[g['name']] += 1

        print(f"{'Block Name':<25} | {'Total GT':>10} | {'Detected (TP)':>15} | {'Missed (FN)':>12} | {'Recall %':>10}")
        print('-' * 80)
        for bname in sorted(gt_by_type.keys()):
            tot = gt_by_type[bname]
            miss = fn_by_type.get(bname, 0)
            det = tot - miss
            rec = (det / tot) * 100
            print(f"{bname:<25} | {tot:10d} | {det:15d} | {miss:12d} | {rec:9.1f}%")

    # Visual validation output
    res20 = results_by_conf[0.20]
    vis_canvas = img_bgr.copy()
    matched_set = res20['matched_det']
    for idx, d in enumerate(res20['dets_list']):
        x1, y1, x2, y2 = map(int, d['bbox_px'])
        color = (0, 200, 0) if idx in matched_set else (0, 140, 255)
        cv2.rectangle(vis_canvas, (x1, y1), (x2, y2), color, 2)
        cv2.putText(vis_canvas, f"{d['conf']:.2f}", (x1, max(12, y1 - 3)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)

    for g in res20['unmatched_gt']:
        px_c = int(round((g['xc'] - x_min) / W_cad * W_px))
        px_r = int(round((y_max - g['yc']) / H_cad * H_px))
        cv2.circle(vis_canvas, (px_c, px_r), 20, (0, 0, 255), 3)
        cv2.putText(vis_canvas, f"FN:{g['name'][:10]}", (px_c - 20, px_r - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

    vis_path = os.path.join(out_dir, "tsss_2_visual_validation.png")
    cv2.imwrite(vis_path, vis_canvas)
    print(f"\n[Visual] Validation image saved to: {vis_path}")

    # Also save detail of unmatched GT items to json for inspection & dataset injection
    fn_export_path = os.path.join(out_dir, "unmatched_gt_conf0.20.json")
    with open(fn_export_path, 'w', encoding='utf-8') as f:
        json.dump(res20['unmatched_gt'], f, indent=2)
    print(f"[Export] Missed GT exported to {fn_export_path}")

    return results_by_conf

if __name__ == '__main__':
    run_eval(
        model_path='train-maker/models/best_componente_nano.pt',
        gt_csv_path='dxf/tsss_2_gt_completo.csv',
        render_path='evaluation_tsss_2/tsss_2_render.png',
        meta_path='evaluation_tsss_2/tsss_2_meta.json',
        out_dir='evaluation_tsss_2',
        dist_tol=1.0
    )
