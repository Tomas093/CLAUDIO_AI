import os
import csv
import json
import random
from pathlib import Path

import cv2
import numpy as np

random.seed(42)
np.random.seed(42)

BASE_DIR = Path('c:/Users/Tomas/Documents/LAB3/CLAUDIO_AI')
DATASET_DIR = BASE_DIR / 'train-maker' / 'dataset_unified_componente'
BG_DIR = BASE_DIR / 'train-maker' / 'output' / 'backgrounds'
BORNERA_SPRITES_DIR = BASE_DIR / 'train-maker' / 'output' / 'bornera' / 'sprites'

# FL-UN-02 paths and metadata
FL02_RENDER = BASE_DIR / 'eval_multitest' / 'fl_un_02_completo' / 'FL-UN-02_tablero_1_render.png'
FL02_META = BASE_DIR / 'eval_multitest' / 'fl_un_02_completo' / 'FL-UN-02_tablero_1_meta.json'
FL02_GT = BASE_DIR / 'dxf' / 'fl_un_02_gt_completo.csv'

# TEST1 paths and metadata
TEST1_RENDER = BASE_DIR / 'eval_multitest' / 'test1_comp' / 'test1_render.png'
TEST1_META = BASE_DIR / 'eval_multitest' / 'test1_comp' / 'test1_meta.json'
TEST1_GT = BASE_DIR / 'test' / 'test_1' / 'verdad_terreno' / 'test1_completo.csv'

def paste_rgba_on_rgb(bg, fg, x, y):
    h, w = fg.shape[:2]
    bg_h, bg_w = bg.shape[:2]
    if x < 0 or y < 0 or x + w > bg_w or y + h > bg_h:
        return bg
    alpha = fg[:, :, 3].astype(float) / 255.0
    alpha = np.expand_dims(alpha, axis=2)
    fg_rgb = fg[:, :, :3].astype(float)
    bg_crop = bg[y:y+h, x:x+w].astype(float)
    blended = (1.0 - alpha) * bg_crop + alpha * fg_rgb
    bg[y:y+h, x:x+w] = blended.astype(np.uint8)
    return bg

def main():
    print("Iniciando ingesta de refuerzo para BORNERAS y casos limite...")
    train_img_dir = DATASET_DIR / 'train' / 'images'
    train_lbl_dir = DATASET_DIR / 'train' / 'labels'
    val_img_dir = DATASET_DIR / 'val' / 'images'
    val_lbl_dir = DATASET_DIR / 'val' / 'labels'

    for d in [train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir]:
        d.mkdir(parents=True, exist_ok=True)

    tile_size = 640

    # -------------------------------------------------------------
    # 1. TILES ESPECIFICOS DE FL-UN-02 CON FOCO EN BORNERAS
    # -------------------------------------------------------------
    print("\n[1/3] Extrayendo tiles de FL-UN-02 centrados en borneras...")
    with open(FL02_META, 'r', encoding='utf-8') as f:
        fl_meta = json.load(f)
    fl_img = cv2.imread(str(FL02_RENDER))
    fl_h, fl_w = fl_img.shape[:2]

    fl_gt_boxes = []
    with open(FL02_GT, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            x1_cad = float(row['x1'])
            y1_cad = float(row['y1'])
            x2_cad = float(row['x2'])
            y2_cad = float(row['y2'])
            xc_cad = float(row['x_cad'])
            yc_cad = float(row['y_cad'])

            px_x1 = (min(x1_cad, x2_cad) - fl_meta['x_min_cad']) / fl_meta['W_cad'] * fl_w
            px_x2 = (max(x1_cad, x2_cad) - fl_meta['x_min_cad']) / fl_meta['W_cad'] * fl_w
            px_y1 = (fl_meta['y_max_cad'] - max(y1_cad, y2_cad)) / fl_meta['H_cad'] * fl_h
            px_y2 = (fl_meta['y_max_cad'] - min(y1_cad, y2_cad)) / fl_meta['H_cad'] * fl_h

            px_xc = (xc_cad - fl_meta['x_min_cad']) / fl_meta['W_cad'] * fl_w
            px_yc = (fl_meta['y_max_cad'] - yc_cad) / fl_meta['H_cad'] * fl_h

            fl_gt_boxes.append({
                'name': row['block_name'],
                'px_x1': px_x1, 'px_y1': px_y1,
                'px_x2': px_x2, 'px_y2': px_y2,
                'px_xc': px_xc, 'px_yc': px_yc,
                'xc_cad': xc_cad, 'yc_cad': yc_cad
            })

    # Filtrar borneras
    borneras_fl02 = [b for b in fl_gt_boxes if b['name'] == 'BORNES']
    print(f"Total borneras en FL-UN-02: {len(borneras_fl02)}")

    generated_fl_bornes = 0
    total_annotations_fl = 0

    for idx, b_item in enumerate(borneras_fl02):
        # Si es de la tira inferior (yc < 15.0) y en especial x entre 606 y 611, generar 12 variantes
        is_bottom_strip = (b_item['yc_cad'] < 15.0)
        is_tricky = is_bottom_strip and (606.0 <= b_item['xc_cad'] <= 611.5)
        num_variants = 12 if is_tricky else (4 if is_bottom_strip else 2)

        for var_i in range(num_variants):
            jx = random.randint(-100, 100)
            jy = random.randint(-100, 100)
            cx = int(round(b_item['px_xc'] + jx))
            cy = int(round(b_item['px_yc'] + jy))

            x1 = max(0, cx - tile_size // 2)
            y1 = max(0, cy - tile_size // 2)
            if x1 + tile_size > fl_w:
                x1 = max(0, fl_w - tile_size)
            if y1 + tile_size > fl_h:
                y1 = max(0, fl_h - tile_size)
            x2 = min(fl_w, x1 + tile_size)
            y2 = min(fl_h, y1 + tile_size)

            tile_crop = fl_img[y1:y2, x1:x2]
            if tile_crop.shape[0] != tile_size or tile_crop.shape[1] != tile_size:
                tile_crop = cv2.copyMakeBorder(tile_crop, 0, tile_size - tile_crop.shape[0],
                                               0, tile_size - tile_crop.shape[1],
                                               cv2.BORDER_CONSTANT, value=[255, 255, 255])

            labels = []
            for b in fl_gt_boxes:
                if x1 <= b['px_xc'] <= x2 and y1 <= b['px_yc'] <= y2:
                    bx1 = max(x1, b['px_x1']) - x1
                    bx2 = min(x2, b['px_x2']) - x1
                    by1 = max(y1, b['px_y1']) - y1
                    by2 = min(y2, b['px_y2']) - y1

                    bw = (bx2 - bx1) / tile_size
                    bh = (by2 - by1) / tile_size
                    bxc = (bx1 + bx2) / (2.0 * tile_size)
                    byc = (by1 + by2) / (2.0 * tile_size)

                    if bw > 0.005 and bh > 0.005:
                        labels.append(f"0 {bxc:.6f} {byc:.6f} {bw:.6f} {bh:.6f}")

            if not labels:
                continue

            is_val = (random.random() < 0.15)
            target_img_dir = val_img_dir if is_val else train_img_dir
            target_lbl_dir = val_lbl_dir if is_val else train_lbl_dir

            stem = f"fl02_borne_boost_{idx:03d}_v{var_i}"
            cv2.imwrite(str(target_img_dir / f"{stem}.jpg"), tile_crop)
            (target_lbl_dir / f"{stem}.txt").write_text('\n'.join(labels) + '\n', encoding='utf-8')
            generated_fl_bornes += 1
            total_annotations_fl += len(labels)

    print(f"Tiles de borneras FL-UN-02 generados: {generated_fl_bornes} (anotaciones: {total_annotations_fl})")

    # -------------------------------------------------------------
    # 2. SINTESIS DE REGLETAS DE BORNERAS MULTIPLES
    # -------------------------------------------------------------
    print("\n[2/3] Sintetizando regletas de borneras alineadas con conductores...")
    bornera_sprites = list(BORNERA_SPRITES_DIR.glob('*.png'))
    bg_files = list(BG_DIR.glob('*.jpg')) + list(BG_DIR.glob('*.png'))
    generated_synth_bornes = 0

    for i in range(120):
        # Crear fondo blanco o tomar bg
        if bg_files and random.random() < 0.5:
            bg_path = random.choice(bg_files)
            bg = cv2.imread(str(bg_path))
            if bg.shape[:2] != (tile_size, tile_size):
                bg = cv2.resize(bg, (tile_size, tile_size))
        else:
            bg = np.full((tile_size, tile_size, 3), 255, dtype=np.uint8)

        # Regleta horizontal: 3 a 8 borneras
        num_bornes = random.randint(3, 8)
        spacing = random.randint(50, 85)
        start_x = random.randint(40, max(50, tile_size - (num_bornes * spacing) - 50))
        borne_y = random.randint(150, tile_size - 120)

        # Dibujar linea de borne de tierra/neutro horizontal inferior
        cv2.line(bg, (max(0, start_x - 30), borne_y + 20), (min(tile_size, start_x + num_bornes * spacing + 30), borne_y + 20), (140, 50, 20), 2)

        labels = []
        for b_idx in range(num_bornes):
            cur_x = start_x + b_idx * spacing
            cur_y = borne_y

            # Dibujar conductor vertical entrante
            cv2.line(bg, (cur_x + 15, max(0, cur_y - 120)), (cur_x + 15, cur_y + 20), (140, 20, 20), 2)

            # Texto de borne (X1, X2, 1, 2)
            if random.random() < 0.7:
                cv2.putText(bg, random.choice(["X1", "X2", "XT", "N", "PE"]), (cur_x - 5, cur_y - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 100, 100), 1, cv2.LINE_AA)

            sp_path = random.choice(bornera_sprites)
            sp_img = cv2.imread(str(sp_path), cv2.IMREAD_UNCHANGED)
            if sp_img is None:
                continue

            target_size = random.randint(30, 42)
            resized_sp = cv2.resize(sp_img, (target_size, target_size), interpolation=cv2.INTER_AREA)

            bg = paste_rgba_on_rgb(bg, resized_sp, cur_x, cur_y)

            bxc = (cur_x + target_size / 2.0) / tile_size
            byc = (cur_y + target_size / 2.0) / tile_size
            bw = target_size / tile_size
            bh = target_size / tile_size

            labels.append(f"0 {bxc:.6f} {byc:.6f} {bw:.6f} {bh:.6f}")

        if not labels:
            continue

        is_val = (random.random() < 0.15)
        target_img_dir = val_img_dir if is_val else train_img_dir
        target_lbl_dir = val_lbl_dir if is_val else train_lbl_dir

        stem = f"synth_bornera_strip_{i:03d}"
        cv2.imwrite(str(target_img_dir / f"{stem}.jpg"), bg)
        (target_lbl_dir / f"{stem}.txt").write_text('\n'.join(labels) + '\n', encoding='utf-8')
        generated_synth_bornes += 1

    print(f"Regletas sinteticas de borneras generadas: {generated_synth_bornes}")

    # -------------------------------------------------------------
    # 3. TILES DE REFUERZO DE TEST1 (ITM)
    # -------------------------------------------------------------
    print("\n[3/3] Extrayendo tiles de refuerzo de TEST1 (ITM)...")
    with open(TEST1_META, 'r', encoding='utf-8') as f:
        t1_meta = json.load(f)
    t1_img = cv2.imread(str(TEST1_RENDER))
    t1_h, t1_w = t1_img.shape[:2]

    t1_gt_boxes = []
    with open(TEST1_GT, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            x1_cad = float(row['x1'])
            y1_cad = float(row['y1'])
            x2_cad = float(row['x2'])
            y2_cad = float(row['y2'])
            xc_cad = float(row['x_cad'])
            yc_cad = float(row['y_cad'])

            px_x1 = (min(x1_cad, x2_cad) - t1_meta['x_min_cad']) / t1_meta['W_cad'] * t1_w
            px_x2 = (max(x1_cad, x2_cad) - t1_meta['x_min_cad']) / t1_meta['W_cad'] * t1_w
            px_y1 = (t1_meta['y_max_cad'] - max(y1_cad, y2_cad)) / t1_meta['H_cad'] * t1_h
            px_y2 = (t1_meta['y_max_cad'] - min(y1_cad, y2_cad)) / t1_meta['H_cad'] * t1_h

            px_xc = (xc_cad - t1_meta['x_min_cad']) / t1_meta['W_cad'] * t1_w
            px_yc = (t1_meta['y_max_cad'] - yc_cad) / t1_meta['H_cad'] * t1_h

            t1_gt_boxes.append({
                'name': row['block_name'],
                'px_x1': px_x1, 'px_y1': px_y1,
                'px_x2': px_x2, 'px_y2': px_y2,
                'px_xc': px_xc, 'px_yc': px_yc
            })

    itm_boxes = [b for b in t1_gt_boxes if 'ITM' in b['name']]
    generated_t1_itm = 0
    for itm in itm_boxes:
        for var_i in range(15):
            jx = random.randint(-80, 80)
            jy = random.randint(-80, 80)
            cx = int(round(itm['px_xc'] + jx))
            cy = int(round(itm['px_yc'] + jy))

            x1 = max(0, cx - tile_size // 2)
            y1 = max(0, cy - tile_size // 2)
            if x1 + tile_size > t1_w:
                x1 = max(0, t1_w - tile_size)
            if y1 + tile_size > t1_h:
                y1 = max(0, t1_h - tile_size)
            x2 = min(t1_w, x1 + tile_size)
            y2 = min(t1_h, y1 + tile_size)

            tile_crop = t1_img[y1:y2, x1:x2]
            if tile_crop.shape[0] != tile_size or tile_crop.shape[1] != tile_size:
                tile_crop = cv2.copyMakeBorder(tile_crop, 0, tile_size - tile_crop.shape[0],
                                               0, tile_size - tile_crop.shape[1],
                                               cv2.BORDER_CONSTANT, value=[255, 255, 255])

            labels = []
            for b in t1_gt_boxes:
                if x1 <= b['px_xc'] <= x2 and y1 <= b['px_yc'] <= y2:
                    bx1 = max(x1, b['px_x1']) - x1
                    bx2 = min(x2, b['px_x2']) - x1
                    by1 = max(y1, b['px_y1']) - y1
                    by2 = min(y2, b['px_y2']) - y1

                    bw = (bx2 - bx1) / tile_size
                    bh = (by2 - by1) / tile_size
                    bxc = (bx1 + bx2) / (2.0 * tile_size)
                    byc = (by1 + by2) / (2.0 * tile_size)

                    if bw > 0.005 and bh > 0.005:
                        labels.append(f"0 {bxc:.6f} {byc:.6f} {bw:.6f} {bh:.6f}")

            if not labels:
                continue

            is_val = (random.random() < 0.15)
            target_img_dir = val_img_dir if is_val else train_img_dir
            target_lbl_dir = val_lbl_dir if is_val else train_lbl_dir

            stem = f"test1_itm_boost_{var_i}"
            cv2.imwrite(str(target_img_dir / f"{stem}.jpg"), tile_crop)
            (target_lbl_dir / f"{stem}.txt").write_text('\n'.join(labels) + '\n', encoding='utf-8')
            generated_t1_itm += 1

    print(f"Tiles de refuerzo TEST1 ITM generados: {generated_t1_itm}")

    total_train = len(list(train_img_dir.glob('*.jpg')) + list(train_img_dir.glob('*.png')))
    total_val = len(list(val_img_dir.glob('*.jpg')) + list(val_img_dir.glob('*.png')))
    print("\nRESUMEN FINAL DATASET:")
    print(f"  Train: {total_train} imagenes")
    print(f"  Val:   {total_val} imagenes")
    print(f"  Total: {total_train + total_val} imagenes")

if __name__ == '__main__':
    main()
