# train-maker/augment_and_inject_hard_samples.py
import os
import sys
import csv
import json
import math
import random
import shutil
from pathlib import Path
import cv2
import numpy as np
import ezdxf
import ezdxf.bbox

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
DATASET_DIR = BASE_DIR / 'dataset_unified_componente'

def main():
    random.seed(42)
    np.random.seed(42)
    
    print("=" * 70)
    print("INYECCION DE HARD NEGATIVES (BORNES, PLANILLA-UNI) Y HARD POSITIVES (SPM)")
    print("=" * 70)
    
    train_img_dir = DATASET_DIR / 'train' / 'images'
    train_lbl_dir = DATASET_DIR / 'train' / 'labels'
    val_img_dir = DATASET_DIR / 'val' / 'images'
    val_lbl_dir = DATASET_DIR / 'val' / 'labels'
    
    for d in [train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir]:
        d.mkdir(parents=True, exist_ok=True)
        
    # Cargar Render y Meta de FL-UN-02
    fl_render_path = ROOT_DIR / 'eval_multitest' / 'fl_un_02' / 'FL-UN-02_tablero_1_render.png'
    fl_meta_path = ROOT_DIR / 'eval_multitest' / 'fl_un_02' / 'FL-UN-02_tablero_1_meta.json'
    fl_gt_path = ROOT_DIR / 'dxf' / 'fl_un_02_gt.csv'
    
    if not fl_render_path.exists() or not fl_meta_path.exists():
        raise FileNotFoundError(f"No existe render o meta en {fl_render_path}")
        
    print(f"Cargando {fl_render_path}...")
    img_fl = cv2.imread(str(fl_render_path))
    H_img, W_img = img_fl.shape[:2]
    
    with open(fl_meta_path, 'r', encoding='utf-8') as f:
        meta = json.load(f)
        
    x_min_cad = meta['x_min_cad']
    y_min_cad = meta['y_min_cad']
    x_max_cad = meta['x_max_cad']
    y_max_cad = meta['y_max_cad']
    W_cad = meta['W_cad']
    H_cad = meta['H_cad']
    W_px = meta['W_px']
    H_px = meta['H_px']
    
    # Cargar Ground Truth de componentes de FL-UN-02
    gt_components = []
    with open(fl_gt_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            x1_c = float(r['x1'])
            y1_c = float(r['y1'])
            x2_c = float(r['x2'])
            y2_c = float(r['y2'])
            px1 = int(round((x1_c - x_min_cad) / W_cad * W_px))
            px2 = int(round((x2_c - x_min_cad) / W_cad * W_px))
            py1 = int(round((y_max_cad - y2_c) / H_cad * H_px))
            py2 = int(round((y_max_cad - y1_c) / H_cad * H_px))
            gt_components.append({
                'name': r['block_name'],
                'cad_bbox': [x1_c, y1_c, x2_c, y2_c],
                'px_bbox': [min(px1, px2), min(py1, py2), max(px1, px2), max(py1, py2)]
            })
    print(f"Cargados {len(gt_components)} componentes GT de FL-UN-02.")
    
    # ---------------------------------------------------------
    # 1. EXTRAER Y PREPARAR SPM CROPS AISLADOS
    # ---------------------------------------------------------
    spm_items = [g for g in gt_components if g['name'] == 'SPM']
    print(f"Encontrados {len(spm_items)} SPM en GT.")
    
    spm_isolated = []
    for idx, s in enumerate(spm_items):
        bx = s['px_bbox']
        crop = img_fl[bx[1]:bx[3], bx[0]:bx[2]].copy()
        if crop.shape[0] > 10 and crop.shape[1] > 10:
            spm_isolated.append(crop)
    print(f"Extraidos {len(spm_isolated)} recortes de SPM.")
    
    # ---------------------------------------------------------
    # 2. GENERAR POSITIVOS SPM CON ESCALA (0.6x - 1.4x) Y CONTEXTO
    # ---------------------------------------------------------
    spm_train_count = 0
    spm_val_count = 0
    
    tile_size = 640
    for s_idx, s in enumerate(spm_items):
        bx = s['px_bbox']
        c_x = (bx[0] + bx[2]) // 2
        c_y = (bx[1] + bx[3]) // 2
        
        offsets = [
            (0, 0),
            (-150, -80),
            (150, 80),
            (-80, 150),
            (120, -120),
            (-200, 100)
        ]
        
        for o_idx, (dx, dy) in enumerate(offsets):
            x1 = c_x + dx - tile_size // 2
            y1 = c_y + dy - tile_size // 2
            x2 = x1 + tile_size
            y2 = y1 + tile_size
            
            pad_left = max(0, -x1)
            pad_top = max(0, -y1)
            pad_right = max(0, x2 - W_img)
            pad_bottom = max(0, y2 - H_img)
            
            src_x1 = max(0, x1)
            src_y1 = max(0, y1)
            src_x2 = min(W_img, x2)
            src_y2 = min(H_img, y2)
            
            crop = img_fl[src_y1:src_y2, src_x1:src_x2]
            if crop.size == 0:
                continue
                
            tile = cv2.copyMakeBorder(crop, pad_top, pad_bottom, pad_left, pad_right,
                                      cv2.BORDER_CONSTANT, value=[255, 255, 255])
            if tile.shape[0] != tile_size or tile.shape[1] != tile_size:
                tile = cv2.resize(tile, (tile_size, tile_size))
                
            labels = []
            for g in gt_components:
                gb = g['px_bbox']
                tx1 = gb[0] - x1
                ty1 = gb[1] - y1
                tx2 = gb[2] - x1
                ty2 = gb[3] - y1
                
                inter_x1 = max(0, tx1)
                inter_y1 = max(0, ty1)
                inter_x2 = min(tile_size, tx2)
                inter_y2 = min(tile_size, ty2)
                
                if inter_x2 > inter_x1 and inter_y2 > inter_y1:
                    orig_area = (gb[2] - gb[0]) * (gb[3] - gb[1])
                    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                    if orig_area > 0 and (inter_area / orig_area) >= 0.45:
                        bx_w = (inter_x2 - inter_x1) / tile_size
                        bx_h = (inter_y2 - inter_y1) / tile_size
                        bx_xc = (inter_x1 + inter_x2) / 2.0 / tile_size
                        bx_yc = (inter_y1 + inter_y2) / 2.0 / tile_size
                        labels.append(f"0 {bx_xc:.6f} {bx_yc:.6f} {bx_w:.6f} {bx_h:.6f}")
                        
            is_val = (s_idx % 4 == 0 and o_idx == 0)
            target_img_dir = val_img_dir if is_val else train_img_dir
            target_lbl_dir = val_lbl_dir if is_val else train_lbl_dir
            
            fname = f"fl_context_spm_{s_idx:02d}_off_{o_idx:02d}"
            cv2.imwrite(str(target_img_dir / f"{fname}.png"), tile)
            with open(target_lbl_dir / f"{fname}.txt", 'w', encoding='utf-8') as lf:
                lf.write('\n'.join(labels) + ('\n' if labels else ''))
                
            if is_val:
                spm_val_count += 1
            else:
                spm_train_count += 1
                
    scales = [0.60, 0.75, 0.85, 1.0, 1.15, 1.30, 1.40]
    for rep in range(180):
        tile = np.full((tile_size, tile_size, 3), 255, dtype=np.uint8)
        
        n_lines = random.randint(2, 5)
        for _ in range(n_lines):
            lx = random.randint(40, tile_size - 40)
            cv2.line(tile, (lx, 0), (lx, tile_size), (0, 0, 0), random.choice([1, 2]))
            
        labels = []
        n_spm_in_tile = random.choice([1, 2, 3, 4])
        
        for _ in range(n_spm_in_tile):
            spm_patch = random.choice(spm_isolated)
            sc = random.choice(scales)
            new_w = max(15, int(round(spm_patch.shape[1] * sc)))
            new_h = max(15, int(round(spm_patch.shape[0] * sc)))
            resized_spm = cv2.resize(spm_patch, (new_w, new_h), interpolation=cv2.INTER_AREA if sc < 1.0 else cv2.INTER_LINEAR)
            
            for _ in range(20):
                px = random.randint(30, tile_size - new_w - 30)
                py = random.randint(30, tile_size - new_h - 30)
                
                collision = False
                for l in labels:
                    parts = list(map(float, l.split()[1:]))
                    ex1 = (parts[0] - parts[2]/2) * tile_size
                    ey1 = (parts[1] - parts[3]/2) * tile_size
                    ex2 = (parts[0] + parts[2]/2) * tile_size
                    ey2 = (parts[1] + parts[3]/2) * tile_size
                    if max(px, ex1) < min(px + new_w, ex2) and max(py, ey1) < min(py + new_h, ey2):
                        collision = True
                        break
                if not collision:
                    mask = (resized_spm < 235).any(axis=2)
                    tile[py:py+new_h, px:px+new_w][mask] = resized_spm[mask]
                    
                    cx_p = px + new_w // 2
                    cv2.line(tile, (cx_p, 0), (cx_p, py), (0, 0, 0), 1)
                    cv2.line(tile, (cx_p, py + new_h), (cx_p, tile_size), (0, 0, 0), 1)
                    
                    bx_xc = (px + new_w / 2.0) / tile_size
                    bx_yc = (py + new_h / 2.0) / tile_size
                    bx_w = new_w / float(tile_size)
                    bx_h = new_h / float(tile_size)
                    labels.append(f"0 {bx_xc:.6f} {bx_yc:.6f} {bx_w:.6f} {bx_h:.6f}")
                    break
                    
        if labels:
            is_val = (rep % 5 == 0)
            target_img_dir = val_img_dir if is_val else train_img_dir
            target_lbl_dir = val_lbl_dir if is_val else train_lbl_dir
            
            fname = f"synth_spm_scaled_{rep:03d}"
            cv2.imwrite(str(target_img_dir / f"{fname}.png"), tile)
            with open(target_lbl_dir / f"{fname}.txt", 'w', encoding='utf-8') as lf:
                lf.write('\n'.join(labels) + '\n')
                
            if is_val:
                spm_val_count += 1
            else:
                spm_train_count += 1
                
    print(f"Generados {spm_train_count} train y {spm_val_count} val SPM positivos.")
    
    # ---------------------------------------------------------
    # 3. EXTRAER E INYECTAR BORNES COMO HARD NEGATIVES (TXT VACIO)
    # ---------------------------------------------------------
    doc = ezdxf.readfile(str(ROOT_DIR / 'dxf' / 'FL-UN-02_tablero_1.dxf'))
    bornes = [e for e in doc.modelspace().query('INSERT') if 'BORN' in e.dxf.name.upper()]
    print(f"Encontrados {len(bornes)} bloques BORNES en DXF.")
    
    bornes_train_count = 0
    bornes_val_count = 0
    
    for x_step in range(0, W_img - 300, 180):
        x1 = x_step
        x2 = min(W_img, x1 + tile_size)
        y2 = H_img
        y1 = max(0, y2 - 250)
        
        crop = img_fl[y1:y2, x1:x2]
        if crop.size == 0:
            continue
            
        has_gt = False
        for g in gt_components:
            gb = g['px_bbox']
            if max(x1, gb[0]) < min(x2, gb[2]) and max(y1, gb[1]) < min(y2, gb[3]):
                has_gt = True
                break
        if has_gt:
            continue
            
        tile = cv2.copyMakeBorder(crop, 0, tile_size - crop.shape[0], 0, tile_size - crop.shape[1],
                                  cv2.BORDER_CONSTANT, value=[255, 255, 255])
                                  
        is_val = (bornes_train_count % 6 == 0)
        target_img_dir = val_img_dir if is_val else train_img_dir
        target_lbl_dir = val_lbl_dir if is_val else train_lbl_dir
        
        fname = f"hard_neg_borne_strip_{x_step:04d}"
        cv2.imwrite(str(target_img_dir / f"{fname}.png"), tile)
        with open(target_lbl_dir / f"{fname}.txt", 'w', encoding='utf-8') as lf:
            lf.write("")
            
        if is_val:
            bornes_val_count += 1
        else:
            bornes_train_count += 1
            
    borne_crops = []
    for b in bornes[:15]:
        bb = ezdxf.bbox.extents([b])
        px1 = int(round((bb.extmin.x - x_min_cad) / W_cad * W_px))
        px2 = int(round((bb.extmax.x - x_min_cad) / W_cad * W_px))
        py1 = int(round((y_max_cad - bb.extmax.y) / H_cad * H_px))
        py2 = int(round((y_max_cad - bb.extmin.y) / H_cad * H_px))
        if 0 <= px1 < px2 <= W_img and 0 <= py1 < py2 <= H_img:
            bc = img_fl[py1:py2, px1:px2]
            if bc.size > 0:
                borne_crops.append(bc)
                
    print(f"Extraidos {len(borne_crops)} recortes individuales de borne.")
    
    if borne_crops:
        for rep in range(120):
            tile = np.full((tile_size, tile_size, 3), 255, dtype=np.uint8)
            row_y = random.randint(150, tile_size - 150)
            n_bornes = random.randint(4, 10)
            spacing = random.randint(50, 75)
            start_x = random.randint(30, 80)
            
            for k in range(n_bornes):
                b_crop = random.choice(borne_crops)
                bw, bh = b_crop.shape[1], b_crop.shape[0]
                bx = start_x + k * spacing
                if bx + bw >= tile_size - 10:
                    break
                mask = (b_crop < 235).any(axis=2)
                tile[row_y:row_y+bh, bx:bx+bw][mask] = b_crop[mask]
                cx_b = bx + bw // 2
                cv2.line(tile, (cx_b, 0), (cx_b, row_y), (0, 0, 0), 1)
                cv2.line(tile, (cx_b, row_y + bh), (cx_b, tile_size), (0, 0, 0), 1)
                
            is_val = (rep % 5 == 0)
            target_img_dir = val_img_dir if is_val else train_img_dir
            target_lbl_dir = val_lbl_dir if is_val else train_lbl_dir
            
            fname = f"synth_hard_neg_borne_{rep:03d}"
            cv2.imwrite(str(target_img_dir / f"{fname}.png"), tile)
            with open(target_lbl_dir / f"{fname}.txt", 'w', encoding='utf-8') as lf:
                lf.write("")
                
            if is_val:
                bornes_val_count += 1
            else:
                bornes_train_count += 1
                
    print(f"Generados {bornes_train_count} train y {bornes_val_count} val BORNES negativos puros.")
    
    # ---------------------------------------------------------
    # 4. HARD NEGATIVES DE TABLA (PLANILLA-UNI) Y CONDUCTORES
    # ---------------------------------------------------------
    table_train_count = 0
    table_val_count = 0
    
    for x_step in range(100, W_img - 600, 300):
        for y_step in [100, 800, 2400]:
            x1 = x_step
            y1 = y_step
            x2 = min(W_img, x1 + tile_size)
            y2 = min(H_img, y1 + tile_size)
            
            overlap_gt = False
            for g in gt_components:
                gb = g['px_bbox']
                if max(x1, gb[0]) < min(x2, gb[2]) and max(y1, gb[1]) < min(y2, gb[3]):
                    overlap_gt = True
                    break
            if overlap_gt:
                continue
                
            crop = img_fl[y1:y2, x1:x2]
            if (crop < 240).mean() > 0.005:
                tile = cv2.copyMakeBorder(crop, 0, tile_size - crop.shape[0], 0, tile_size - crop.shape[1],
                                          cv2.BORDER_CONSTANT, value=[255, 255, 255])
                is_val = (table_train_count % 5 == 0)
                target_img_dir = val_img_dir if is_val else train_img_dir
                target_lbl_dir = val_lbl_dir if is_val else train_lbl_dir
                
                fname = f"hard_neg_background_fl_{x_step:04d}_{y_step:04d}"
                cv2.imwrite(str(target_img_dir / f"{fname}.png"), tile)
                with open(target_lbl_dir / f"{fname}.txt", 'w', encoding='utf-8') as lf:
                    lf.write("")
                    
                if is_val:
                    table_val_count += 1
                else:
                    table_train_count += 1
                    
    print(f"Generados {table_train_count} train y {table_val_count} val fondos/tablas negativos puros.")
    print("=" * 70)
    print("INYECCION COMPLETADA EXITOSAMENTE")
    print("=" * 70)

if __name__ == '__main__':
    main()
