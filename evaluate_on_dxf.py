import os
import sys
import csv
import json
import time
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from dxf_to_image import renderizar_dxf
from vector_inference import nms_agnostico_clase_cad, eliminar_anidadas_cad

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')


def slice_image_coords(H, W, slice_size=640, overlap=0.80):
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


def run_inference_on_dxf(dxf_path, model_path, output_dir, target_px=100,
                         conf_thresh=0.02, iou_thresh=0.45, overlap=0.80,
                         pad=320, batch_size=32, device='cuda:0'):
    os.makedirs(output_dir, exist_ok=True)
    dxf_stem = Path(dxf_path).stem
    render_img_path = os.path.join(output_dir, f'{dxf_stem}_render.png')
    render_json_path = os.path.join(output_dir, f'{dxf_stem}_render.json')

    if not os.path.exists(render_img_path) or not os.path.exists(render_json_path):
        print(f'[render] Renderizando {dxf_path}...')
        meta = renderizar_dxf(dxf_path, render_img_path, target_px=target_px)
    else:
        print(f'[render] Usando render existente: {render_img_path}')
        with open(render_json_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)

    print(f'[model] Cargando modelo {model_path} en {device}...')
    model = YOLO(model_path)
    full_img = cv2.imread(render_img_path)
    if full_img is None:
        raise FileNotFoundError(f'No se pudo leer {render_img_path}')
    H, W = full_img.shape[:2]

    # Padding perimetral
    if pad > 0:
        padded_img = cv2.copyMakeBorder(full_img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    else:
        padded_img = full_img
    pH, pW = padded_img.shape[:2]

    slices = slice_image_coords(pH, pW, slice_size=640, overlap=overlap)
    print(f'[slicing] {len(slices)} tiles generados para imagen {W}x{H} (pad={pad}px, overlap={overlap*100:.0f}%, step={int(round(640*(1.0-overlap)))}px)')

    all_raw_dets = []
    px_per_cad = meta['px_per_cad']
    x_min_cad = meta['x_min_cad']
    y_max_cad = meta['y_max_cad']

    t0 = time.time()
    for i in range(0, len(slices), batch_size):
        batch_slices = slices[i:i + batch_size]
        batch_imgs = [padded_img[y1:y2, x1:x2] for (x1, y1, x2, y2) in batch_slices]
        results = model(batch_imgs, conf=conf_thresh, verbose=False, device=device)
        for (x1_tile, y1_tile, _, _), res in zip(batch_slices, results):
            if res.boxes is None or len(res.boxes) == 0:
                continue
            xyxy = res.boxes.xyxy.cpu().numpy()
            confs = res.boxes.conf.cpu().numpy()
            for b, c in zip(xyxy, confs):
                px_x1 = x1_tile + b[0] - pad
                px_y1 = y1_tile + b[1] - pad
                px_x2 = x1_tile + b[2] - pad
                px_y2 = y1_tile + b[3] - pad

                cad_x1 = x_min_cad + px_x1 / px_per_cad
                cad_x2 = x_min_cad + px_x2 / px_per_cad
                cad_y1 = y_max_cad - px_y2 / px_per_cad
                cad_y2 = y_max_cad - px_y1 / px_per_cad

                cx = (cad_x1 + cad_x2) / 2.0
                cy = (cad_y1 + cad_y2) / 2.0

                all_raw_dets.append({
                    'clase': 'componente',
                    'conf': float(c),
                    'bbox_cad': [cad_x1, min(cad_y1, cad_y2), cad_x2, max(cad_y1, cad_y2)],
                    'centro_cad': [cx, cy],
                    'x_cad': cx,
                    'y_cad': cy,
                    'bbox_px': [px_x1, px_y1, px_x2, px_y2]
                })

    t_inf = time.time() - t0
    print(f'[infer] Detecciones crudas: {len(all_raw_dets)} en {t_inf:.2f}s ({len(slices)/(t_inf+1e-6):.1f} tiles/s)')

    nms_dets = nms_agnostico_clase_cad(all_raw_dets, iou_thresh=iou_thresh)
    final_dets = eliminar_anidadas_cad(nms_dets, ios_thresh=0.60)
    print(f'[post] Detecciones tras NMS e inclusion: {len(final_dets)}')
    return final_dets, meta, full_img


def evaluate_matches(gt_csv_path, detections, dist_thresh_cad=0.85):
    gt_list = []
    with open(gt_csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            gt_list.append({
                'block_name': row.get('block_name', row.get('tipo', 'componente')),
                'x_cad': float(row['x_cad']),
                'y_cad': float(row['y_cad']),
            })

    total_gt = len(gt_list)
    candidates = []
    for g_idx, g in enumerate(gt_list):
        for d_idx, d in enumerate(detections):
            dist = np.sqrt((g['x_cad'] - d['x_cad'])**2 + (g['y_cad'] - d['y_cad'])**2)
            b = d['bbox_cad']
            inside = (b[0] <= g['x_cad'] <= b[2] and b[1] <= g['y_cad'] <= b[3])
            if dist <= dist_thresh_cad or inside:
                candidates.append((dist, g_idx, d_idx))

    # Greedy 1-to-1 matching
    candidates.sort(key=lambda x: x[0])
    matched_gt = set()
    matched_det = set()
    for dist, g_idx, d_idx in candidates:
        if g_idx not in matched_gt and d_idx not in matched_det:
            matched_gt.add(g_idx)
            matched_det.add(d_idx)

    tp = len(matched_gt)
    fn = total_gt - tp
    fp = len(detections) - len(matched_det)

    recall = tp / total_gt if total_gt > 0 else 1.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    per_type = defaultdict(lambda: {'total': 0, 'detected': 0})
    for g_idx, g in enumerate(gt_list):
        b_name = g['block_name'].split('$')[-1]
        per_type[b_name]['total'] += 1
        if g_idx in matched_gt:
            per_type[b_name]['detected'] += 1

    return {
        'total_gt': total_gt,
        'tp': tp,
        'fn': fn,
        'fp': fp,
        'recall': recall,
        'precision': precision,
        'f1': f1,
        'per_type': dict(per_type),
        'unmatched_gt': [gt_list[i] for i in range(total_gt) if i not in matched_gt],
        'unmatched_dets': [detections[i] for i in range(len(detections)) if i not in matched_det],
        'matched_det_indices': matched_det
    }


def draw_visual_validation(full_img, meta, detections, eval_res, out_path):
    canvas = full_img.copy()
    px_per_cad = meta['px_per_cad']
    x_min_cad = meta['x_min_cad']
    y_max_cad = meta['y_max_cad']

    matched_det_indices = eval_res.get('matched_det_indices', set())
    for idx, d in enumerate(detections):
        x1, y1, x2, y2 = map(int, d['bbox_px'])
        color = (0, 200, 0) if idx in matched_det_indices else (0, 140, 255)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        cv2.putText(canvas, f"{d['conf']:.2f}", (x1, max(15, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

    for g in eval_res['unmatched_gt']:
        px_c = int(round((g['x_cad'] - x_min_cad) * px_per_cad))
        px_r = int(round((y_max_cad - g['y_cad']) * px_per_cad))
        cv2.circle(canvas, (px_c, px_r), 25, (0, 0, 255), 3)
        cv2.putText(canvas, 'MISSING', (px_c - 20, px_r - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    cv2.imwrite(out_path, canvas)
    print(f'[visual] Visualizacion guardada en: {out_path}')


def test_plan(dxf_path, gt_csv, model_path, output_dir, target_px=100, overlap=0.80, pad=320):
    print('\n' + '=' * 75)
    print(f'EVALUANDO: {dxf_path}')
    print('=' * 75)

    raw_dets, meta, img = run_inference_on_dxf(
        dxf_path=dxf_path,
        model_path=model_path,
        output_dir=output_dir,
        target_px=target_px,
        conf_thresh=0.02,
        overlap=overlap,
        pad=pad,
        device='cuda:0' if torch.cuda.is_available() else 'cpu'
    )

    print(f'\n--- BARRIDO CONTRA GROUND TRUTH BASE ({gt_csv}) ---')
    print('  Conf | Dets |   GT |   TP |   FN |   FP |    Recall |  Precision |       F1')
    print('-' * 75)

    best_res = None
    best_conf = 0.25

    for conf in [0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60]:
        filtered = [d for d in raw_dets if d['conf'] >= conf]
        res = evaluate_matches(gt_csv, filtered)
        rec_s = f"{res['recall']*100:.1f}%"
        prec_s = f"{res['precision']*100:.1f}%"
        f1_s = f"{res['f1']:.3f}"
        print(f"  {conf:4.2f} | {len(filtered):4d} | {res['total_gt']:4d} | {res['tp']:4d} | {res['fn']:4d} | {res['fp']:4d} | {rec_s:>9s} | {prec_s:>10s} | {f1_s:>8s}")

        if best_res is None:
            best_res = res
            best_conf = conf
        else:
            if res['fn'] < best_res['fn']:
                best_res = res
                best_conf = conf
            elif res['fn'] == best_res['fn'] and res['precision'] > best_res['precision']:
                best_res = res
                best_conf = conf

    print('\n' + '-' * 75)
    print(f'RESULTADO BASE: Conf >= {best_conf:.2f}')
    print(f"  Total Ground Truth:  {best_res['total_gt']}")
    print(f"  True Positives (TP): {best_res['tp']}")
    print(f"  False Negatives (FN):{best_res['fn']}  (cero falsos negativos objetivo)")
    print(f"  False Positives (FP):{best_res['fp']}")
    print(f"  Recall:              {best_res['recall']*100:.2f}%")
    print(f"  Precision:           {best_res['precision']*100:.2f}%")
    print(f"  F1-Score:            {best_res['f1']:.4f}")

    gt_completo_csv = str(gt_csv).replace('_inserts.csv', '_completo.csv')
    if os.path.exists(gt_completo_csv):
        print('\n' + '=' * 75)
        print(f'--- BARRIDO CONTRA GROUND TRUTH COMPLETO ({gt_completo_csv}) ---')
        print('  (Incluye aparatos dibujados con CIRCLE/TEXT/LINE en capa 01-ELE-APARATOS)')
        print('  Conf | Dets |   GT |   TP |   FN |   FP |    Recall |  Precision |       F1')
        print('-' * 75)
        best_comp_res = None
        best_comp_conf = 0.25
        for conf in [0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60]:
            filtered = [d for d in raw_dets if d['conf'] >= conf]
            res_c = evaluate_matches(gt_completo_csv, filtered)
            rec_s = f"{res_c['recall']*100:.1f}%"
            prec_s = f"{res_c['precision']*100:.1f}%"
            f1_s = f"{res_c['f1']:.3f}"
            print(f"  {conf:4.2f} | {len(filtered):4d} | {res_c['total_gt']:4d} | {res_c['tp']:4d} | {res_c['fn']:4d} | {res_c['fp']:4d} | {rec_s:>9s} | {prec_s:>10s} | {f1_s:>8s}")
            if best_comp_res is None or (res_c['fn'] < best_comp_res['fn']) or (res_c['fn'] == best_comp_res['fn'] and res_c['precision'] > best_comp_res['precision']):
                best_comp_res = res_c
                best_comp_conf = conf

        print('\n' + '-' * 75)
        print(f'RESULTADO COMPLETO (REAL-WORLD APPARATUS): Conf >= {best_comp_conf:.2f}')
        print(f"  Total Ground Truth:  {best_comp_res['total_gt']}")
        print(f"  True Positives (TP): {best_comp_res['tp']}")
        print(f"  False Negatives (FN):{best_comp_res['fn']}")
        print(f"  False Positives (FP):{best_comp_res['fp']}")
        print(f"  Recall:              {best_comp_res['recall']*100:.2f}%")
        print(f"  Precision:           {best_comp_res['precision']*100:.2f}%")
        print(f"  F1-Score:            {best_comp_res['f1']:.4f}")

    print('=' * 75)
    out_vis = os.path.join(output_dir, f'{Path(dxf_path).stem}_eval_visual.png')
    best_dets = [d for d in raw_dets if d['conf'] >= best_conf]
    draw_visual_validation(img, meta, best_dets, best_res, out_vis)
    return best_res


def main():
    model_path = 'train-maker/models/best_componente_nano.pt'
    if not os.path.exists(model_path):
        model_path = 'yolo_workspace/boosted_componente_yolo11n/weights/best.pt'
    if not os.path.exists(model_path):
        model_path = 'yolo_workspace/unified_componente_yolo11n/weights/best.pt'
    if not os.path.exists(model_path):
        print(f'Esperando a que exista el modelo en {model_path}...')
        return

    print(f'Evaluando modelo: {model_path}')
    print('\n=== EVALUACION SOBRE TEST 2 (test_2.dxf) ===')
    res2 = test_plan('test_2.dxf', 'test/test_2/verdad_terreno/test_2_inserts.csv', model_path, 'evaluation_test2', 100, overlap=0.80, pad=320)
    print('\n=== EVALUACION SOBRE TEST 1 (test1.dxf) ===')
    res1 = test_plan('test1.dxf', 'test/test_1/verdad_terreno/test1_inserts.csv', model_path, 'evaluation_test1', 100, overlap=0.80, pad=320)


if __name__ == '__main__':
    main()
