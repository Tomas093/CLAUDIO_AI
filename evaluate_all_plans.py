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
import ezdxf, ezdxf.bbox
from ezdxf.addons.drawing import RenderContext, Frontend
from ezdxf.addons.drawing.matplotlib import MatplotlibBackend
from ezdxf.addons.drawing.config import Configuration, ColorPolicy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ultralytics import YOLO

from vector_inference import (
    nms_agnostico_clase_cad,
    eliminar_anidadas_cad,
    nms_distancia_cad,
    fusionar_multipolares_cad,
    snap_a_bloques_cad
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

def evaluate_dxf(dxf_path, gt_csv_path, model, out_dir, px_per_cad_default=75.0,
                 is_industrial=False, conf_sweep=[0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50],
                 dist_tol=1.0, batch_size=32, device='cuda:0', force_render=False):
    os.makedirs(out_dir, exist_ok=True)
    dxf_stem = Path(dxf_path).stem
    render_path = os.path.join(out_dir, f"{dxf_stem}_render.png")
    meta_path = os.path.join(out_dir, f"{dxf_stem}_meta.json")

    # Renderizado inteligente
    if force_render or not os.path.exists(render_path) or not os.path.exists(meta_path):
        print(f"\n[Render] Renderizando {dxf_path}...")
        doc = ezdxf.readfile(dxf_path)
        msp = doc.modelspace()
        ctx = RenderContext(doc)
        
        # Filtro de entidades: excluir textos y cotas
        # Para planos industriales, excluir la capa IE-UN-TEXTOS donde reside la tabla PLANILLA-UNI
        if is_industrial:
            geom_entities = [e for e in msp if e.dxftype() not in ('TEXT', 'MTEXT', 'HATCH', 'DIMENSION', 'LEADER')
                             and e.dxf.layer.upper() not in ('IE-UN-TEXTOS', 'FORMATO', 'CARATULA')]
        else:
            geom_entities = [e for e in msp if e.dxftype() not in ('TEXT', 'MTEXT', 'HATCH', 'DIMENSION', 'LEADER')]

        bbox = ezdxf.bbox.extents(geom_entities)
        x_min, y_min = bbox.extmin.x, bbox.extmin.y
        x_max, y_max = bbox.extmax.x, bbox.extmax.y
        W_cad = x_max - x_min
        H_cad = y_max - y_min
        px_per_cad = px_per_cad_default
        W_px = int(round(W_cad * px_per_cad))
        H_px = int(round(H_cad * px_per_cad))

        dpi = 100
        fig = plt.figure(figsize=(W_px/dpi, H_px/dpi), dpi=dpi)
        ax = fig.add_axes([0, 0, 1, 1]); ax.axis('off')
        cfg = Configuration(color_policy=ColorPolicy.COLOR, custom_bg_color='#ffffff')
        fe = Frontend(ctx, MatplotlibBackend(ax), config=cfg)
        fe.draw_entities(geom_entities)
        ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max); ax.set_aspect('equal')
        fig.canvas.draw()
        buf = np.array(fig.canvas.buffer_rgba())[:, :, :3]
        plt.close(fig)
        img_bgr = cv2.cvtColor(buf, cv2.COLOR_RGB2BGR)
        cv2.imwrite(render_path, img_bgr)
        meta = {
            'px_per_cad': px_per_cad,
            'x_min_cad': x_min, 'y_min_cad': y_min,
            'x_max_cad': x_max, 'y_max_cad': y_max,
            'W_cad': W_cad, 'H_cad': H_cad,
            'W_px': W_px, 'H_px': H_px
        }
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(meta, f, indent=2)
    else:
        print(f"\n[Render] Usando render existente: {render_path}")
        img_bgr = cv2.imread(render_path)
        with open(meta_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)

    W_px, H_px = img_bgr.shape[1], img_bgr.shape[0]
    x_min, y_min = meta['x_min_cad'], meta['y_min_cad']
    x_max, y_max = meta['x_max_cad'], meta['y_max_cad']
    W_cad, H_cad = meta['W_cad'], meta['H_cad']
    px_per_cad = meta['px_per_cad']

    # Cargar Ground Truth
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

    # Slicing con padding
    pad = 320
    slice_size = 640
    padded_img = cv2.copyMakeBorder(img_bgr, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    pad_h, pad_w = padded_img.shape[:2]
    slices = slice_coords(pad_h, pad_w, slice_size=slice_size, overlap=0.80)

    # Inferencia batch
    raw_dets = []
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
    print(f"Inferencia en {t_inf:.1f}s ({len(slices)/(t_inf+1e-6):.1f} tiles/s). Detecciones crudas: {len(raw_dets)}")

    # Barrido por umbrales con métricas completas (TP, FN, FP, Recall, Precision, F1)
    results_by_conf = {}
    print(f"\n{'Conf':>6} | {'Dets':>6} | {'GT':>6} | {'TP':>6} | {'FN':>6} | {'FP':>6} | {'Recall':>8} | {'Precision':>10} | {'F1':>8}")
    print('-' * 78)

    for th in conf_sweep:
        dets_th = [d for d in raw_dets if d['conf'] >= th]
        dets_th = nms_agnostico_clase_cad(dets_th, iou_thresh=0.45)
        dets_th = eliminar_anidadas_cad(dets_th, ios_thresh=0.60)
        dets_th = nms_distancia_cad(dets_th, d_min=0.45)

        # Emparejamiento greedy 1-a-1
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

    # Análisis detallado a conf=0.20
    ref_th = 0.20
    ref_res = results_by_conf.get(ref_th) or list(results_by_conf.values())[0]
    if ref_res['unmatched_gt']:
        fn_counts = defaultdict(int)
        for g in ref_res['unmatched_gt']:
            fn_counts[g['name'].split('$')[-1]] += 1
        print(f"  [Detalle FN a conf={ref_th}]: {dict(fn_counts)}")

    # Guardar crops de FPs principales
    fp_crop_dir = os.path.join(out_dir, "fp_crops")
    os.makedirs(fp_crop_dir, exist_ok=True)
    top_fps = sorted(ref_res['unmatched_det'], key=lambda d: -d['conf'])[:8]
    for idx, fp_d in enumerate(top_fps):
        px = fp_d['bbox_px']
        pad_c = 15
        x1 = max(0, px[0] - pad_c); y1 = max(0, px[1] - pad_c)
        x2 = min(W_px, px[2] + pad_c); y2 = min(H_px, px[3] + pad_c)
        crop = img_bgr[y1:y2, x1:x2].copy()
        cv2.rectangle(crop, (px[0]-x1, px[1]-y1), (px[2]-x1, px[3]-y1), (0, 0, 255), 2)
        cv2.imwrite(os.path.join(fp_crop_dir, f"fp_{idx:02d}_conf_{fp_d['conf']:.2f}.png"), crop)

    # Dibujar lámina de validación visual completa
    vis_canvas = img_bgr.copy()
    matched_set = ref_res.get('matched_det', set())
    for idx, d in enumerate(ref_res['dets_list']):
        x1, y1, x2, y2 = map(int, d['bbox_px'])
        color = (0, 200, 0) if idx in matched_set else (0, 140, 255)
        cv2.rectangle(vis_canvas, (x1, y1), (x2, y2), color, 2)
        cv2.putText(vis_canvas, f"{d['conf']:.2f}", (x1, max(12, y1 - 3)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)

    for g in ref_res['unmatched_gt']:
        px_c = int(round((g['xc'] - x_min) / W_cad * W_px))
        px_r = int(round((y_max - g['yc']) / H_cad * H_px))
        cv2.circle(vis_canvas, (px_c, px_r), 20, (0, 0, 255), 3)
        cv2.putText(vis_canvas, 'FN', (px_c - 10, px_r - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

    vis_path = os.path.join(out_dir, f"{dxf_stem}_visual_validation.png")
    cv2.imwrite(vis_path, vis_canvas)
    print(f"  [Visual] Lamina guardada en: {vis_path}")

    return results_by_conf, img_bgr, meta

def main():
    MODEL_PATH = 'train-maker/models/best_componente_nano.pt'
    print('=' * 80)
    print(f'EVALUACION GLOBAL MULTI-PLANO DE COMPONENTES: {MODEL_PATH}')
    print('=' * 80)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = YOLO(MODEL_PATH)

    plans = [
        {
            'name': 'TEST 2 (Distribucion - GT Base INSERTs)',
            'dxf': 'test/test_2/test_2.dxf' if os.path.exists('test/test_2/test_2.dxf') else 'test_2.dxf',
            'gt': 'test/test_2/verdad_terreno/test_2_inserts.csv',
            'out': 'eval_multitest/test2_base',
            'scale': 84.31,
            'is_ind': False,
            'dist_tol': 0.85
        },
        {
            'name': 'TEST 2 (Distribucion - GT Completo Aparatos)',
            'dxf': 'test/test_2/test_2.dxf' if os.path.exists('test/test_2/test_2.dxf') else 'test_2.dxf',
            'gt': 'test/test_2/verdad_terreno/test_2_completo.csv',
            'out': 'eval_multitest/test2_comp',
            'scale': 84.31,
            'is_ind': False,
            'dist_tol': 0.85
        },
        {
            'name': 'TEST 1 (Distribucion - GT Base INSERTs)',
            'dxf': 'test/test_1/test1.dxf' if os.path.exists('test/test_1/test1.dxf') else 'test1.dxf',
            'gt': 'test/test_1/verdad_terreno/test1_inserts.csv',
            'out': 'eval_multitest/test1_base',
            'scale': 75.0,
            'is_ind': False,
            'dist_tol': 0.85
        },
        {
            'name': 'TEST 1 (Distribucion - GT Completo Aparatos)',
            'dxf': 'test/test_1/test1.dxf' if os.path.exists('test/test_1/test1.dxf') else 'test1.dxf',
            'gt': 'test/test_1/verdad_terreno/test1_completo.csv',
            'out': 'eval_multitest/test1_comp',
            'scale': 75.0,
            'is_ind': False,
            'dist_tol': 0.85
        },
        {
            'name': 'FL-UN-02 (Industrial Schneider - 258 Aparatos)',
            'dxf': 'dxf/FL-UN-02_tablero_1.dxf',
            'gt': 'dxf/fl_un_02_gt.csv',
            'out': 'eval_multitest/fl_un_02',
            'scale': 75.0,
            'is_ind': True,
            'dist_tol': 1.2
        },
        {
            'name': 'FL-UN-02 (Total con BORNES - 363 Componentes)',
            'dxf': 'dxf/FL-UN-02_tablero_1.dxf',
            'gt': 'dxf/fl_un_02_gt_completo.csv',
            'out': 'eval_multitest/fl_un_02_completo',
            'scale': 75.0,
            'is_ind': True,
            'dist_tol': 1.2
        },
        {
            'name': 'TSSS_2 (Industrial Schneider - 206 Componentes con BORNES)',
            'dxf': 'TSSS_2 (1).dxf',
            'gt': 'dxf/tsss_2_gt_completo.csv',
            'out': 'evaluation_tsss_2',
            'scale': 75.0,
            'is_ind': True,
            'dist_tol': 1.0
        }
    ]

    force_render = '--force-render' in sys.argv
    all_summaries = {}
    for p in plans:
        print('\n' + '=' * 80)
        print(f"PLAN: {p['name']}")
        print(f"DXF: {p['dxf']} | GT: {p['gt']}")
        print('=' * 80)
        res_conf, img, meta = evaluate_dxf(
            dxf_path=p['dxf'],
            gt_csv_path=p['gt'],
            model=model,
            out_dir=p['out'],
            px_per_cad_default=p['scale'],
            is_industrial=p['is_ind'],
            dist_tol=p['dist_tol'],
            device=device,
            force_render=force_render
        )
        all_summaries[p['name']] = res_conf

    # Resumen final comparativo
    print('\n' + '#' * 85)
    print('TABLA RESUMEN FINAL - RENDIMIENTO POR PLANO A UMBRAL OPTIMO (Conf >= 0.20)')
    print('#' * 85)
    print(f"{'Plano / Benchmark':<45} | {'GT':>5} | {'TP':>5} | {'FN':>4} | {'FP':>4} | {'Recall':>8} | {'Precision':>10}")
    print('-' * 85)
    for name, res in all_summaries.items():
        # Tomar conf=0.20
        r20 = res.get(0.20, res.get(0.15))
        print(f"{name:<45} | {r20['gt']:5d} | {r20['tp']:5d} | {r20['fn']:4d} | {r20['fp']:4d} | {r20['recall']:7.1f}% | {r20['precision']:9.1f}%")
    print('#' * 85)

if __name__ == '__main__':
    main()
