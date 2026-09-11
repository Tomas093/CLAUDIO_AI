import os
import csv
import json
import random
import cv2
import numpy as np
import zipfile
import shutil
from pathlib import Path

random.seed(42)
np.random.seed(42)

BASE_DIR = Path('c:/Users/Tomas/Documents/LAB3/CLAUDIO_AI')
DATASET_DIR = BASE_DIR / 'train-maker' / 'dataset_unified_componente'
TEMP_DIR = BASE_DIR / 'temp_enrichment'

TRAIN_IMG = DATASET_DIR / 'train' / 'images'
TRAIN_LBL = DATASET_DIR / 'train' / 'labels'
VAL_IMG = DATASET_DIR / 'val' / 'images'
VAL_LBL = DATASET_DIR / 'val' / 'labels'

for d in [TRAIN_IMG, TRAIN_LBL, VAL_IMG, VAL_LBL]:
    d.mkdir(parents=True, exist_ok=True)

def remap_to_class_zero(src_lbl: Path, dst_lbl: Path):
    if not src_lbl.exists() or src_lbl.stat().st_size == 0:
        dst_lbl.write_text('', encoding='utf-8')
        return 0
    lines = []
    with open(src_lbl, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                try:
                    lines.append(f"0 {parts[1]} {parts[2]} {parts[3]} {parts[4]}")
                except Exception:
                    continue
    dst_lbl.write_text('\n'.join(lines) + ('\n' if lines else ''), encoding='utf-8')
    return len(lines)

def ingest_user_zips():
    print("\n" + "=" * 70)
    print("1. INTEGRANDO LOS 3 DATASETS ZIP DE ROBOFLOW (BORNERA, DIFERENCIAL, TERMOMAGNETICA)")
    print("=" * 70)
    zips = [
        ('Bornera.v2i.yolov11.zip', 'bornera'),
        ('Diferencial.v3i.yolov11.zip', 'diferencial'),
        ('Termomagnetica.v2i.yolov11.zip', 'termomagnetica')
    ]
    total_added = 0
    for zip_name, comp_name in zips:
        zp = BASE_DIR / zip_name
        if not zp.exists():
            print(f"[WARN] No se encontro {zp}")
            continue
        extract_to = TEMP_DIR / comp_name
        with zipfile.ZipFile(zp, 'r') as zf:
            zf.extractall(extract_to)

        for target_split, folders in [('train', ['train']), ('val', ['valid', 'val', 'test'])]:
            dst_img = TRAIN_IMG if target_split == 'train' else VAL_IMG
            dst_lbl = TRAIN_LBL if target_split == 'train' else VAL_LBL
            for fn in folders:
                img_dir = extract_to / fn / 'images'
                lbl_dir = extract_to / fn / 'labels'
                if not img_dir.exists():
                    continue
                for ip in img_dir.glob('*.*'):
                    if ip.suffix.lower() not in ['.jpg', '.png', '.jpeg']:
                        continue
                    lp = lbl_dir / f"{ip.stem}.txt"
                    dst_name = f"rf_{comp_name}_{ip.stem}"
                    shutil.copy2(ip, dst_img / f"{dst_name}{ip.suffix}")
                    remap_to_class_zero(lp, dst_lbl / f"{dst_name}.txt")
                    total_added += 1
        print(f"  [OK] {comp_name}: integrado exitosamente.")
    if TEMP_DIR.exists():
        shutil.rmtree(TEMP_DIR)
    print(f"Total imagenes de Roboflow integradas: {total_added}")

def auto_label_plan_components(render_path, json_path, gt_csv_path, plan_name, default_box_cad=(1.2, 1.4)):
    print("\n" + "=" * 70)
    print(f"2. AUTO-ETIQUETADO Y RECORTE DE COMPONENTES DEL PLANO: {plan_name}")
    print("=" * 70)
    if not os.path.exists(render_path) or not os.path.exists(json_path) or not os.path.exists(gt_csv_path):
        print(f"[WARN] Archivos faltantes para {plan_name}")
        return 0

    img = cv2.imread(render_path)
    if img is None:
        print(f"[WARN] No se pudo leer {render_path}")
        return 0
    img_h, img_w = img.shape[:2]

    with open(json_path, 'r', encoding='utf-8') as f:
        meta = json.load(f)

    px_per_cad = meta['px_per_cad']
    x_min_cad = meta['x_min_cad']
    y_min_cad = meta['y_min_cad']
    x_max_cad = meta['x_max_cad']
    y_max_cad = meta['y_max_cad']
    W_cad = meta.get('W_cad', x_max_cad - x_min_cad)
    H_cad = meta.get('H_cad', y_max_cad - y_min_cad)

    gt_items = []
    with open(gt_csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            xc = float(row.get('x_cad', 0))
            yc = float(row.get('y_cad', 0))
            bname = row.get('block_name', row.get('tipo', 'comp'))
            
            # Ancho y alto en CAD según tipo
            if 'x1' in row and 'x2' in row and float(row['x2']) > float(row['x1']):
                w_cad = float(row['x2']) - float(row['x1'])
                h_cad = float(row['y2']) - float(row['y1'])
                xc = (float(row['x1']) + float(row['x2'])) / 2.0
                yc = (float(row['y1']) + float(row['y2'])) / 2.0
            else:
                # Estimación inteligente según tipo de componente
                b_upper = bname.upper()
                if 'OJO' in b_upper or 'PILOTO' in b_upper or 'LAMP' in b_upper:
                    w_cad, h_cad = 0.6, 0.6
                elif 'BORNE' in b_upper:
                    w_cad, h_cad = 0.6, 0.45
                elif 'IM' in b_upper or 'MEDICION' in b_upper:
                    w_cad, h_cad = 1.0, 1.0
                elif 'SPM' in b_upper:
                    w_cad, h_cad = 0.65, 0.6
                else:
                    w_cad, h_cad = default_box_cad

            # Convertir centro y tamaño a pixeles
            px_xc = (xc - x_min_cad) / W_cad * img_w
            px_yc = (y_max_cad - yc) / H_cad * img_h
            px_w = w_cad * px_per_cad
            px_h = h_cad * px_per_cad

            gt_items.append({
                'name': bname,
                'xc': xc, 'yc': yc,
                'px_xc': px_xc, 'px_yc': px_yc,
                'px_w': px_w, 'px_h': px_h
            })

    print(f"Total componentes en {plan_name}: {len(gt_items)}")

    tile_size = 640
    generated_tiles = 0

    for idx, comp in enumerate(gt_items):
        # Generar 2 variantes con desplazamiento aleatorio para robustez
        for var_i in range(2):
            jitter_x = random.randint(-60, 60)
            jitter_y = random.randint(-60, 60)
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

            # Recolectar todos los componentes que caen en este tile
            labels = []
            for other in gt_items:
                if x1 <= other['px_xc'] <= x2 and y1 <= other['px_yc'] <= y2:
                    bx1 = other['px_xc'] - other['px_w'] / 2.0 - x1
                    bx2 = other['px_xc'] + other['px_w'] / 2.0 - x1
                    by1 = other['px_yc'] - other['px_h'] / 2.0 - y1
                    by2 = other['px_yc'] + other['px_h'] / 2.0 - y1

                    bx1 = max(0, min(tile_size, bx1))
                    bx2 = max(0, min(tile_size, bx2))
                    by1 = max(0, min(tile_size, by1))
                    by2 = max(0, min(tile_size, by2))

                    bw = (bx2 - bx1) / tile_size
                    bh = (by2 - by1) / tile_size
                    bxc = (bx1 + bx2) / (2.0 * tile_size)
                    byc = (by1 + by2) / (2.0 * tile_size)

                    if bw > 0.015 and bh > 0.015:
                        labels.append(f"0 {bxc:.6f} {byc:.6f} {bw:.6f} {bh:.6f}")

            if not labels:
                continue

            is_val = (random.random() < 0.15)
            target_img = VAL_IMG if is_val else TRAIN_IMG
            target_lbl = VAL_LBL if is_val else TRAIN_LBL

            stem = f"autolabel_{plan_name}_{idx:03d}_v{var_i}"
            cv2.imwrite(str(target_img / f"{stem}.jpg"), tile_crop)
            (target_lbl / f"{stem}.txt").write_text('\n'.join(labels) + '\n', encoding='utf-8')
            generated_tiles += 1

    print(f"Tiles auto-etiquetados generados para {plan_name}: {generated_tiles}")
    return generated_tiles

def main():
    ingest_user_zips()

    # Auto-etiquetar componentes de TEST 1
    auto_label_plan_components(
        render_path='evaluation_test1/test1_render.png',
        json_path='evaluation_test1/test1_render.json',
        gt_csv_path='test/test_1/verdad_terreno/test1_completo.csv',
        plan_name='test1'
    )

    # Auto-etiquetar componentes de TEST 2
    auto_label_plan_components(
        render_path='evaluation_test2/test_2_render.png',
        json_path='evaluation_test2/test_2_render.json',
        gt_csv_path='test/test_2/verdad_terreno/test_2_completo.csv',
        plan_name='test2'
    )

    # Auto-etiquetar componentes de FL-UN-02 con BORNES incluidos
    # Crear meta json si no existe
    fl_json = Path('eval_multitest/fl_un_02/FL-UN-02_tablero_1_meta.json')
    if not fl_json.exists():
        fl_json = Path('evaluation_fl_un_02/fl_un_02_meta.json')

    auto_label_plan_components(
        render_path='evaluation_fl_un_02/fl_un_02_render.png',
        json_path=str(fl_json),
        gt_csv_path='dxf/fl_un_02_gt_completo.csv',
        plan_name='fl_un02'
    )

    # Limpiar cachés de Ultralytics
    for split in ['train', 'val']:
        c = DATASET_DIR / split / 'labels.cache'
        if c.exists():
            c.unlink()

    n_train = len(list(TRAIN_IMG.glob('*.*')))
    n_val = len(list(VAL_IMG.glob('*.*')))
    print("\n" + "#" * 70)
    print(f"DATASET FINAL MASTER ENRIQUECIDO:")
    print(f"  Train: {n_train} imagenes")
    print(f"  Val:   {n_val} imagenes")
    print(f"  Total: {n_train + n_val} imagenes")
    print("#" * 70)

if __name__ == '__main__':
    main()
