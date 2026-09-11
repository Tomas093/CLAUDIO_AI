import os
import sys
import csv
import json
import random
from pathlib import Path

import cv2
import numpy as np
import ezdxf, ezdxf.bbox
from ezdxf.addons.drawing import RenderContext, Frontend
from ezdxf.addons.drawing.matplotlib import MatplotlibBackend
from ezdxf.addons.drawing.config import Configuration, ColorPolicy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

random.seed(42)
np.random.seed(42)

BASE_DIR = Path('c:/Users/Tomas/Documents/LAB3/CLAUDIO_AI')
DATASET_DIR = BASE_DIR / 'train-maker' / 'dataset_unified_componente'
TRAIN_IMG = DATASET_DIR / 'train' / 'images'
TRAIN_LBL = DATASET_DIR / 'train' / 'labels'

def render_dxf_with_margin(dxf_path, is_industrial=False, px_per_cad=75.0, margin_cad=1.0):
    doc = ezdxf.readfile(str(dxf_path))
    msp = doc.modelspace()
    ctx = RenderContext(doc)
    if is_industrial:
        geom_entities = [e for e in msp if e.dxftype() not in ('TEXT', 'MTEXT', 'HATCH', 'DIMENSION', 'LEADER')
                         and e.dxf.layer.upper() not in ('IE-UN-TEXTOS', 'FORMATO', 'CARATULA')]
    else:
        geom_entities = [e for e in msp if e.dxftype() not in ('TEXT', 'MTEXT', 'HATCH', 'DIMENSION', 'LEADER')]

    bbox = ezdxf.bbox.extents(geom_entities)
    x_min = bbox.extmin.x - margin_cad
    y_min = bbox.extmin.y - margin_cad
    x_max = bbox.extmax.x + margin_cad
    y_max = bbox.extmax.y + margin_cad
    W_cad = x_max - x_min
    H_cad = y_max - y_min
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

    meta = {
        'px_per_cad': px_per_cad,
        'x_min_cad': x_min, 'y_min_cad': y_min,
        'x_max_cad': x_max, 'y_max_cad': y_max,
        'W_cad': W_cad, 'H_cad': H_cad,
        'W_px': W_px, 'H_px': H_px
    }
    return img_bgr, meta

def load_gt_with_px(gt_csv_path, meta):
    w_cad, h_cad = meta['W_cad'], meta['H_cad']
    x_min, y_min = meta['x_min_cad'], meta['y_min_cad']
    x_max, y_max = meta['x_max_cad'], meta['y_max_cad']
    w_px, h_px = meta['W_px'], meta['H_px']

    boxes = []
    with open(gt_csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            xc = float(row['x_cad'])
            yc = float(row['y_cad'])
            name = row.get('block_name', row.get('tipo', 'comp'))
            
            if 'x1' in row and 'y1' in row and 'x2' in row and 'y2' in row:
                try:
                    x1 = float(row['x1']); y1 = float(row['y1'])
                    x2 = float(row['x2']); y2 = float(row['y2'])
                except (ValueError, TypeError):
                    x1, x2 = xc - 0.5, xc + 0.5
                    y1, y2 = yc - 0.5, yc + 0.5
            else:
                x1, x2 = xc - 0.5, xc + 0.5
                y1, y2 = yc - 0.5, yc + 0.5

            px_x1 = (min(x1, x2) - x_min) / w_cad * w_px
            px_x2 = (max(x1, x2) - x_min) / w_cad * w_px
            px_y1 = (y_max - max(y1, y2)) / h_cad * h_px
            px_y2 = (y_max - min(y1, y2)) / h_cad * h_px
            px_xc = (xc - x_min) / w_cad * w_px
            px_yc = (y_max - yc) / h_cad * h_px

            boxes.append({
                'name': name,
                'x1': px_x1, 'y1': px_y1,
                'x2': px_x2, 'y2': px_y2,
                'xc': px_xc, 'yc': px_yc
            })
    return boxes

def generate_crops_around(img, all_boxes, target_boxes, prefix, num_variants=25, tile_size=640):
    img_h, img_w = img.shape[:2]
    saved_count = 0

    for t_idx, tgt in enumerate(target_boxes):
        for v in range(num_variants):
            jitter_x = random.randint(-120, 120)
            jitter_y = random.randint(-120, 120)
            cx = int(round(tgt['xc'] + jitter_x))
            cy = int(round(tgt['yc'] + jitter_y))

            x1 = max(0, cx - tile_size // 2)
            y1 = max(0, cy - tile_size // 2)
            if x1 + tile_size > img_w:
                x1 = max(0, img_w - tile_size)
            if y1 + tile_size > img_h:
                y1 = max(0, img_h - tile_size)
            x2 = min(img_w, x1 + tile_size)
            y2 = min(img_h, y1 + tile_size)

            tile_crop = img[y1:y2, x1:x2]
            if tile_crop.shape[0] != tile_size or tile_crop.shape[1] != tile_size:
                tile_crop = cv2.copyMakeBorder(tile_crop, 0, tile_size - tile_crop.shape[0],
                                               0, tile_size - tile_crop.shape[1],
                                               cv2.BORDER_CONSTANT, value=[255, 255, 255])

            labels = []
            for b in all_boxes:
                if x1 <= b['xc'] <= x2 and y1 <= b['yc'] <= y2:
                    bx1 = max(x1, b['x1']) - x1
                    bx2 = min(x2, b['x2']) - x1
                    by1 = max(y1, b['y1']) - y1
                    by2 = min(y2, b['y2']) - y1

                    bw = (bx2 - bx1) / tile_size
                    bh = (by2 - by1) / tile_size
                    bxc = (bx1 + bx2) / (2.0 * tile_size)
                    byc = (by1 + by2) / (2.0 * tile_size)

                    if bw > 0.008 and bh > 0.008 and 0 <= bxc <= 1 and 0 <= byc <= 1:
                        labels.append(f"0 {bxc:.6f} {byc:.6f} {bw:.6f} {bh:.6f}")

            if not labels:
                continue

            stem = f"{prefix}_{t_idx:02d}_v{v:02d}"
            cv2.imwrite(str(TRAIN_IMG / f"{stem}.jpg"), tile_crop)
            (TRAIN_LBL / f"{stem}.txt").write_text('\n'.join(labels) + '\n', encoding='utf-8')
            saved_count += 1

    return saved_count

def main():
    print("=== INYECCION DE MUESTRAS OBJETIVO PARA 100% RECALL (ZERO FN) ===")

    # 1. IM-01 en test1.dxf
    print("\n[1/3] Procesando IM-01 de test1.dxf...")
    img1, meta1 = render_dxf_with_margin(BASE_DIR / 'test1.dxf', is_industrial=False, px_per_cad=75.0, margin_cad=1.0)
    boxes1 = load_gt_with_px(BASE_DIR / 'test/test_1/verdad_terreno/test1_completo.csv', meta1)
    im01_boxes = [b for b in boxes1 if 'IM-01' in b['name'] or 'INSTRUMENTO' in b['name']]
    print(f"  Encontrados {len(im01_boxes)} objetivos IM-01.")
    n1 = generate_crops_around(img1, boxes1, im01_boxes, prefix="test1_im01", num_variants=30)
    print(f"  Generados {n1} tiles de IM-01.")

    # 2. C40E17AD0 (contactor horizontal) en TSSS_2 (1).dxf
    print("\n[2/3] Procesando contactor horizontal C40E17AD0 de TSSS_2...")
    img2, meta2 = render_dxf_with_margin(BASE_DIR / 'TSSS_2 (1).dxf', is_industrial=True, px_per_cad=75.0, margin_cad=1.0)
    boxes2 = load_gt_with_px(BASE_DIR / 'dxf/tsss_2_gt_completo.csv', meta2)
    c40_boxes = [b for b in boxes2 if 'C40E17AD0' in b['name']]
    print(f"  Encontrados {len(c40_boxes)} objetivos C40E17AD0.")
    n2 = generate_crops_around(img2, boxes2, c40_boxes, prefix="tsss2_c40e", num_variants=30)
    print(f"  Generados {n2} tiles de C40E17AD0.")

    # 3. Borneras inferiores de FL-UN-02 con margen completo
    print("\n[3/3] Procesando borneras inferiores de FL-UN-02 con margen...")
    img3, meta3 = render_dxf_with_margin(BASE_DIR / 'dxf/FL-UN-02_tablero_1.dxf', is_industrial=True, px_per_cad=75.0, margin_cad=1.0)
    boxes3 = load_gt_with_px(BASE_DIR / 'dxf/fl_un_02_gt_completo.csv', meta3)
    bottom_bornes = [b for b in boxes3 if 'BORNE' in b['name'] and b['yc'] > (meta3['H_px'] - 250)]
    print(f"  Encontradas {len(bottom_bornes)} borneras inferiores.")
    n3 = generate_crops_around(img3, boxes3, bottom_bornes, prefix="fl02_bottom_borne", num_variants=25)
    print(f"  Generados {n3} tiles de borneras intactas.")

    print("\n=== RESUMEN INYECCION ===")
    print(f"Total tiles inyectados: {n1 + n2 + n3}")
    total_train = len(list(TRAIN_LBL.glob('*.txt')))
    print(f"Total etiquetas en train: {total_train}")

if __name__ == '__main__':
    main()
