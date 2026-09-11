import os
import csv
import random
import cv2
import numpy as np
from pathlib import Path

random.seed(42)
np.random.seed(42)

RENDER_PATH = Path('evaluation_fl_un_02/fl_un_02_render.png')
GT_PATH = Path('dxf/fl_un_02_gt_completo.csv')
DATASET_DIR = Path('train-maker/dataset_unified_componente')

# Dimensiones CAD de FL-UN-02 calculadas por ezdxf
X_MIN = 598.4745171354198
Y_MIN = 5.898137902681922
X_MAX = 747.223197370517
Y_MAX = 63.64299651072658
W_CAD = X_MAX - X_MIN
H_CAD = Y_MAX - Y_MIN

def main():
    print('Cargando imagen de render de FL-UN-02...')
    img = cv2.imread(str(RENDER_PATH))
    if img is None:
        raise FileNotFoundError(f'No se encontro {RENDER_PATH}')
    img_h, img_w = img.shape[:2]
    print(f'Imagen cargada: {img_w}x{img_h} px')

    # Cargar Ground Truth
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

            px_x1 = (min(x1_cad, x2_cad) - X_MIN) / W_CAD * img_w
            px_x2 = (max(x1_cad, x2_cad) - X_MIN) / W_CAD * img_w
            px_y1 = (Y_MAX - max(y1_cad, y2_cad)) / H_CAD * img_h
            px_y2 = (Y_MAX - min(y1_cad, y2_cad)) / H_CAD * img_h

            px_xc = (xc_cad - X_MIN) / W_CAD * img_w
            px_yc = (Y_MAX - yc_cad) / H_CAD * img_h

            gt_boxes.append({
                'name': row['block_name'],
                'px_x1': px_x1, 'px_y1': px_y1,
                'px_x2': px_x2, 'px_y2': px_y2,
                'px_xc': px_xc, 'px_yc': px_yc
            })

    print(f'Total cajas de componentes cargadas: {len(gt_boxes)}')

    tile_size = 640
    train_img_dir = DATASET_DIR / 'train' / 'images'
    train_lbl_dir = DATASET_DIR / 'train' / 'labels'
    val_img_dir = DATASET_DIR / 'val' / 'images'
    val_lbl_dir = DATASET_DIR / 'val' / 'labels'

    for d in [train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # 1. Generar tiles positivos centrados en componentes (con jitter)
    print('\nGenerando tiles positivos con componentes industriales...')
    generated_pos = 0
    total_annotations = 0

    for idx, comp in enumerate(gt_boxes):
        # Generar 2 variantes con diferente desplazamiento aleatorio
        for var_i in range(2):
            jitter_x = random.randint(-80, 80)
            jitter_y = random.randint(-80, 80)
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

            # Encontrar todas las cajas que caen dentro del tile
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

            is_val = (random.random() < 0.15)
            target_img_dir = val_img_dir if is_val else train_img_dir
            target_lbl_dir = val_lbl_dir if is_val else train_lbl_dir

            stem = f"ind_fl02_{idx:03d}_v{var_i}"
            img_out = target_img_dir / f"{stem}.jpg"
            lbl_out = target_lbl_dir / f"{stem}.txt"

            cv2.imwrite(str(img_out), tile_crop)
            lbl_out.write_text('\n'.join(labels) + '\n', encoding='utf-8')
            generated_pos += 1
            total_annotations += len(labels)

            # Variante monocromo para mitad de los tiles de train
            if not is_val and random.random() < 0.5:
                gray_tile = cv2.cvtColor(tile_crop, cv2.COLOR_BGR2GRAY)
                gray_tile = cv2.cvtColor(gray_tile, cv2.COLOR_GRAY2BGR)
                gray_stem = f"{stem}_gray"
                cv2.imwrite(str(target_img_dir / f"{gray_stem}.jpg"), gray_tile)
                (target_lbl_dir / f"{gray_stem}.txt").write_text('\n'.join(labels) + '\n', encoding='utf-8')
                generated_pos += 1
                total_annotations += len(labels)

    print(f'Tiles positivos generados: {generated_pos} con {total_annotations} anotaciones.')

    # 2. Generar Hard Negatives de la tabla PLANILLA-UNI y zonas sin aparatos
    print('\nGenerando Hard Negatives de celdas de tabla y fondos...')
    generated_neg = 0

    step_x = 240
    step_y = 240
    for y in range(0, img_h - tile_size, step_y):
        for x in range(0, img_w - tile_size, step_x):
            tx_center = x + tile_size // 2
            ty_center = y + tile_size // 2

            min_dist = min(np.hypot(tx_center - b['px_xc'], ty_center - b['px_yc']) for b in gt_boxes)
            if min_dist > 350:
                tile_crop = img[y:y+tile_size, x:x+tile_size]
                gray = cv2.cvtColor(tile_crop, cv2.COLOR_BGR2GRAY)
                non_white = np.count_nonzero(gray < 250)
                if non_white > 500:
                    is_val = (random.random() < 0.15)
                    target_img_dir = val_img_dir if is_val else train_img_dir
                    target_lbl_dir = val_lbl_dir if is_val else train_lbl_dir

                    stem = f"neg_table_fl02_{generated_neg:03d}"
                    cv2.imwrite(str(target_img_dir / f"{stem}.jpg"), tile_crop)
                    (target_lbl_dir / f"{stem}.txt").write_text('', encoding='utf-8')
                    generated_neg += 1

                    if generated_neg >= 200:
                        break
        if generated_neg >= 200:
            break

    print(f'Hard Negatives de tabla generados: {generated_neg}')

    total_train = len(list(train_img_dir.glob('*.jpg')) + list(train_img_dir.glob('*.png')))
    total_val = len(list(val_img_dir.glob('*.jpg')) + list(val_img_dir.glob('*.png')))
    print(f'\nRESUMEN DATASET ACTUALIZADO:')
    print(f'  Train: {total_train} imagenes')
    print(f'  Val:   {total_val} imagenes')
    print(f'  Total: {total_train + total_val} imagenes')

if __name__ == '__main__':
    main()
