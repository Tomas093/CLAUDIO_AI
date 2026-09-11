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
RENDER_PATH = BASE_DIR / 'evaluation_tsss_2' / 'tsss_2_render.png'
META_PATH = BASE_DIR / 'evaluation_tsss_2' / 'tsss_2_meta.json'
GT_PATH = BASE_DIR / 'dxf' / 'tsss_2_gt_completo.csv'
DATASET_DIR = BASE_DIR / 'train-maker' / 'dataset_unified_componente'
BG_DIR = BASE_DIR / 'train-maker' / 'output' / 'backgrounds'
PAT_SPRITES_DIR = BASE_DIR / 'train-maker' / 'output' / 'puesta_a_tierra' / 'sprites'
CONTACTOR_SPRITES_DIR = BASE_DIR / 'train-maker' / 'output' / 'contactor' / 'sprites'
SECC_SPRITES_DIR = BASE_DIR / 'train-maker' / 'output' / 'seccionador_bajo_carga' / 'sprites'

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
    print("Iniciando ingesta de componentes de TSSS_2 (1).dxf y elementos faltantes...")
    with open(META_PATH, 'r', encoding='utf-8') as f:
        meta = json.load(f)

    img = cv2.imread(str(RENDER_PATH))
    if img is None:
        raise FileNotFoundError(f"No se pudo leer {RENDER_PATH}")
    img_h, img_w = img.shape[:2]

    x_min_cad = meta['x_min_cad']
    y_min_cad = meta['y_min_cad']
    x_max_cad = meta['x_max_cad']
    y_max_cad = meta['y_max_cad']
    w_cad = meta['W_cad']
    h_cad = meta['H_cad']

    # Cargar GT de componentes reales
    gt_boxes = []
    with open(GT_PATH, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            x1_cad = float(row['x1'])
            y1_cad = float(row['y1'])
            x2_cad = float(row['x2'])
            y2_cad = float(row['y2'])
            xc_cad = float(row['x_cad'])
            yc_cad = float(row['y_cad'])

            px_x1 = (min(x1_cad, x2_cad) - x_min_cad) / w_cad * img_w
            px_x2 = (max(x1_cad, x2_cad) - x_min_cad) / w_cad * img_w
            px_y1 = (y_max_cad - max(y1_cad, y2_cad)) / h_cad * img_h
            px_y2 = (y_max_cad - min(y1_cad, y2_cad)) / h_cad * img_h

            px_xc = (xc_cad - x_min_cad) / w_cad * img_w
            px_yc = (y_max_cad - yc_cad) / h_cad * img_h

            gt_boxes.append({
                'name': row['block_name'],
                'px_x1': px_x1, 'px_y1': px_y1,
                'px_x2': px_x2, 'px_y2': px_y2,
                'px_xc': px_xc, 'px_yc': px_yc
            })

    print(f"Total componentes en GT: {len(gt_boxes)}")

    train_img_dir = DATASET_DIR / 'train' / 'images'
    train_lbl_dir = DATASET_DIR / 'train' / 'labels'
    val_img_dir = DATASET_DIR / 'val' / 'images'
    val_lbl_dir = DATASET_DIR / 'val' / 'labels'

    for d in [train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir]:
        d.mkdir(parents=True, exist_ok=True)

    tile_size = 640
    generated_pos = 0
    total_annotations = 0

    # 1. TILES DIRECTOS DE TSSS_2 (Positivos)
    print("\n[1/5] Generando tiles positivos de TSSS_2...")
    for idx, comp in enumerate(gt_boxes):
        # Para A$C40E17AD0 y PAT generamos mas variantes (10 variantes)
        is_rare = comp['name'] in ['A$C40E17AD0', 'PAT']
        num_variants = 10 if is_rare else 2

        for var_i in range(num_variants):
            jitter_range = 100 if is_rare else 60
            jitter_x = random.randint(-jitter_range, jitter_range)
            jitter_y = random.randint(-jitter_range, jitter_range)

            cx = int(round(comp['px_xc'] + jitter_x))
            cy = int(round(comp['px_yc'] + jitter_y))

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
            for b in gt_boxes:
                if x1 <= b['px_xc'] <= x2 and y1 <= b['px_yc'] <= y2:
                    bx1 = max(x1, b['px_x1']) - x1
                    bx2 = min(x2, b['px_x2']) - x1
                    by1 = max(y1, b['px_y1']) - y1
                    by2 = min(y2, b['px_y2']) - y1

                    bw = (bx2 - bx1) / tile_size
                    bh = (by2 - by1) / tile_size
                    bxc = (bx1 + bx2) / (2.0 * tile_size)
                    byc = (by1 + by2) / (2.0 * tile_size)

                    if bw > 0.01 and bh > 0.01:
                        labels.append(f"0 {bxc:.6f} {byc:.6f} {bw:.6f} {bh:.6f}")

            if not labels:
                continue

            # Asignar a train / val
            is_val = (random.random() < 0.15)
            target_img_dir = val_img_dir if is_val else train_img_dir
            target_lbl_dir = val_lbl_dir if is_val else train_lbl_dir

            stem = f"tsss2_pos_{idx:03d}_{comp['name'][:10]}_v{var_i}"
            cv2.imwrite(str(target_img_dir / f"{stem}.jpg"), tile_crop)
            (target_lbl_dir / f"{stem}.txt").write_text('\n'.join(labels) + '\n', encoding='utf-8')
            generated_pos += 1
            total_annotations += len(labels)

    print(f"Tiles positivos directos generados: {generated_pos} (anotaciones: {total_annotations})")

    # Cargar imagenes de background para sintetizar
    bg_files = list(BG_DIR.glob('*.jpg')) + list(BG_DIR.glob('*.png'))
    print(f"\nFondos disponibles para composicion: {len(bg_files)}")

    # 2. SINTESIS DE PUESTA A TIERRA (PAT) CON SPRITES OFICIALES
    print("\n[2/5] Sintetizando muestras de Puesta a Tierra (PAT)...")
    pat_sprites = list(PAT_SPRITES_DIR.glob('*.png'))
    generated_pat = 0
    for i in range(70):
        sp_path = random.choice(pat_sprites)
        sp_img = cv2.imread(str(sp_path), cv2.IMREAD_UNCHANGED)
        if sp_img is None:
            continue

        # Redimensionar sprite a escala realista de diagrama unifilar (altura 50 a 90 px)
        target_h = random.randint(50, 90)
        scale = target_h / sp_img.shape[0]
        target_w = max(10, int(round(sp_img.shape[1] * scale)))
        resized_sp = cv2.resize(sp_img, (target_w, target_h), interpolation=cv2.INTER_AREA)

        # Elegir fondo o crear blanco
        if bg_files and random.random() < 0.7:
            bg_path = random.choice(bg_files)
            bg = cv2.imread(str(bg_path))
            if bg.shape[:2] != (tile_size, tile_size):
                bg = cv2.resize(bg, (tile_size, tile_size))
        else:
            bg = np.full((tile_size, tile_size, 3), 255, dtype=np.uint8)
            # Trazar lineas de conductores
            line_x = random.randint(100, 540)
            cv2.line(bg, (line_x, 0), (line_x, tile_size), (80, 50, 20), 2)

        # Posicion
        px = random.randint(50, tile_size - target_w - 50)
        py = random.randint(50, tile_size - target_h - 50)

        bg = paste_rgba_on_rgb(bg, resized_sp, px, py)

        # Bounding box
        bxc = (px + target_w / 2.0) / tile_size
        byc = (py + target_h / 2.0) / tile_size
        bw = target_w / tile_size
        bh = target_h / tile_size

        is_val = (random.random() < 0.15)
        target_img_dir = val_img_dir if is_val else train_img_dir
        target_lbl_dir = val_lbl_dir if is_val else train_lbl_dir

        stem = f"synth_pat_{i:03d}"
        cv2.imwrite(str(target_img_dir / f"{stem}.jpg"), bg)
        (target_lbl_dir / f"{stem}.txt").write_text(f"0 {bxc:.6f} {byc:.6f} {bw:.6f} {bh:.6f}\n", encoding='utf-8')
        generated_pat += 1

    print(f"Muestras de Puesta a Tierra generadas: {generated_pat}")

    # 3. SINTESIS DE CONTACTORES E INTERRUPTORES HORIZONTALES (90 y 270 grados)
    print("\n[3/5] Sintetizando contactores y seccionadores horizontales...")
    switch_sprites = list(CONTACTOR_SPRITES_DIR.glob('*.png')) + list(SECC_SPRITES_DIR.glob('*.png'))
    generated_horiz = 0

    # Ademas extraer el sprite real de A$C40E17AD0 de TSSS_2
    ac_box = [b for b in gt_boxes if b['name'] == 'A$C40E17AD0'][0]
    ac_x1, ac_y1 = int(round(ac_box['px_x1'])), int(round(ac_box['px_y1']))
    ac_x2, ac_y2 = int(round(ac_box['px_x2'])), int(round(ac_box['px_y2']))
    ac_crop = img[ac_y1:ac_y2, ac_x1:ac_x2].copy()

    for i in range(70):
        # 50% sprite real de TSSS_2, 50% sprite rotado
        if random.random() < 0.5 and ac_crop.shape[0] > 10 and ac_crop.shape[1] > 10:
            scale_f = random.uniform(0.85, 1.25)
            w_new = int(round(ac_crop.shape[1] * scale_f))
            h_new = int(round(ac_crop.shape[0] * scale_f))
            fg = cv2.resize(ac_crop, (w_new, h_new))
            target_w, target_h = w_new, h_new
            use_real_crop = True
        else:
            sp_path = random.choice(switch_sprites)
            sp_img = cv2.imread(str(sp_path), cv2.IMREAD_UNCHANGED)
            if sp_img is None:
                continue
            # Rotar 90 o 270 grados
            rot_code = cv2.ROTATE_90_CLOCKWISE if random.random() < 0.5 else cv2.ROTATE_90_COUNTERCLOCKWISE
            rotated = cv2.rotate(sp_img, rot_code)
            target_h = random.randint(60, 110)
            scale = target_h / rotated.shape[0]
            target_w = max(10, int(round(rotated.shape[1] * scale)))
            fg = cv2.resize(rotated, (target_w, target_h), interpolation=cv2.INTER_AREA)
            use_real_crop = False

        if bg_files and random.random() < 0.7:
            bg_path = random.choice(bg_files)
            bg = cv2.imread(str(bg_path))
            if bg.shape[:2] != (tile_size, tile_size):
                bg = cv2.resize(bg, (tile_size, tile_size))
        else:
            bg = np.full((tile_size, tile_size, 3), 255, dtype=np.uint8)
            # Trazar linea horizontal de barra
            bar_y = random.randint(150, 490)
            cv2.line(bg, (0, bar_y), (tile_size, bar_y), (140, 20, 20), 2)

        px = random.randint(50, tile_size - target_w - 50)
        py = random.randint(50, tile_size - target_h - 50)

        if use_real_crop:
            bg[py:py+target_h, px:px+target_w] = fg
        else:
            bg = paste_rgba_on_rgb(bg, fg, px, py)

        bxc = (px + target_w / 2.0) / tile_size
        byc = (py + target_h / 2.0) / tile_size
        bw = target_w / tile_size
        bh = target_h / tile_size

        is_val = (random.random() < 0.15)
        target_img_dir = val_img_dir if is_val else train_img_dir
        target_lbl_dir = val_lbl_dir if is_val else train_lbl_dir

        stem = f"synth_horiz_switch_{i:03d}"
        cv2.imwrite(str(target_img_dir / f"{stem}.jpg"), bg)
        (target_lbl_dir / f"{stem}.txt").write_text(f"0 {bxc:.6f} {byc:.6f} {bw:.6f} {bh:.6f}\n", encoding='utf-8')
        generated_horiz += 1

    print(f"Interruptores horizontales generados: {generated_horiz}")

    # 4. HARD NEGATIVES DE TABLA PLANILLA-UNI Y FALSOS POSITIVOS DE TEXTO
    print("\n[4/5] Generando Hard Negatives de la tabla PLANILLA-UNI y textos...")
    generated_neg = 0

    # A. Zonas de la tabla de cargas (abajo en y > 650)
    for y in range(650, img_h - 300, 150):
        for x in range(200, img_w - tile_size, 200):
            tx_center = x + tile_size // 2
            ty_center = y + tile_size // 2

            min_dist = min(np.hypot(tx_center - b['px_xc'], ty_center - b['px_yc']) for b in gt_boxes)
            if min_dist > 300:
                y1 = max(0, img_h - tile_size)
                x1 = min(max(0, x), img_w - tile_size)
                tile_crop = img[y1:y1+tile_size, x1:x1+tile_size]
                if tile_crop.shape[:2] == (tile_size, tile_size):
                    is_val = (random.random() < 0.15)
                    target_img_dir = val_img_dir if is_val else train_img_dir
                    target_lbl_dir = val_lbl_dir if is_val else train_lbl_dir

                    stem = f"neg_tsss2_table_{generated_neg:03d}"
                    cv2.imwrite(str(target_img_dir / f"{stem}.jpg"), tile_crop)
                    (target_lbl_dir / f"{stem}.txt").write_text('', encoding='utf-8')
                    generated_neg += 1
                    if generated_neg >= 40:
                        break
        if generated_neg >= 40:
            break

    # B. Hard Negatives especificos en las zonas de FP 15 y FP 16 (textos con linea)
    fp_centers = [
        (1069, 897), # FP 15: 12A AC3 S1
        (1102, 227), # FP 16: mpact INS
        (4550, 744), # FP multipolar
    ]
    for fp_idx, (fpx, fpy) in enumerate(fp_centers):
        for var_j in range(4):
            jx = random.randint(-40, 40)
            jy = random.randint(-40, 40)
            cx = fpx + jx
            cy = fpy + jy
            x1 = max(0, min(img_w - tile_size, cx - tile_size // 2))
            y1 = max(0, min(img_h - tile_size, cy - tile_size // 2))

            # Verificar que no haya ningun componente en el centro del tile
            min_dist = min(np.hypot(cx - b['px_xc'], cy - b['px_yc']) for b in gt_boxes)
            if min_dist > 250:
                tile_crop = img[y1:y1+tile_size, x1:x1+tile_size]
                if tile_crop.shape[:2] == (tile_size, tile_size):
                    is_val = (random.random() < 0.15)
                    target_img_dir = val_img_dir if is_val else train_img_dir
                    target_lbl_dir = val_lbl_dir if is_val else train_lbl_dir

                    stem = f"neg_tsss2_fptext_{fp_idx}_{var_j}"
                    cv2.imwrite(str(target_img_dir / f"{stem}.jpg"), tile_crop)
                    (target_lbl_dir / f"{stem}.txt").write_text('', encoding='utf-8')
                    generated_neg += 1

    print(f"Hard Negatives generados: {generated_neg}")

    # 5. RESUMEN FINAL
    total_train = len(list(train_img_dir.glob('*.jpg')) + list(train_img_dir.glob('*.png')))
    total_val = len(list(val_img_dir.glob('*.jpg')) + list(val_img_dir.glob('*.png')))
    print("\n[5/5] PROCESO COMPLETADO EXITOSAMENTE.")
    print("=" * 65)
    print(f"Dataset Unificado Actualizado:")
    print(f"  Train: {total_train} imagenes")
    print(f"  Val:   {total_val} imagenes")
    print(f"  Total: {total_train + total_val} imagenes")
    print("=" * 65)

if __name__ == '__main__':
    main()
