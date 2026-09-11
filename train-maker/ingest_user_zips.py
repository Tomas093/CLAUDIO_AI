import os
import zipfile
import shutil
from pathlib import Path

BASE_DIR = Path('c:/Users/Tomas/Documents/LAB3/CLAUDIO_AI')
DATASET_DIR = BASE_DIR / 'train-maker' / 'dataset_unified_componente'
TEMP_DIR = BASE_DIR / 'temp_user_zips'

ZIPS = [
    ('Bornera.v2i.yolov11.zip', 'bornera'),
    ('Diferencial.v3i.yolov11.zip', 'diferencial'),
    ('Termomagnetica.v2i.yolov11.zip', 'termomagnetica')
]

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
                    # Forzar clase 0 (componente)
                    lines.append(f"0 {parts[1]} {parts[2]} {parts[3]} {parts[4]}")
                except Exception:
                    continue
    dst_lbl.write_text('\n'.join(lines) + ('\n' if lines else ''), encoding='utf-8')
    return len(lines)

def main():
    if TEMP_DIR.exists():
        shutil.rmtree(TEMP_DIR)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)

    total_added_train = 0
    total_added_val = 0
    total_boxes = 0

    train_img_dir = DATASET_DIR / 'train' / 'images'
    train_lbl_dir = DATASET_DIR / 'train' / 'labels'
    val_img_dir = DATASET_DIR / 'val' / 'images'
    val_lbl_dir = DATASET_DIR / 'val' / 'labels'

    for d in [train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir]:
        d.mkdir(parents=True, exist_ok=True)

    for zip_name, comp_name in ZIPS:
        zip_path = BASE_DIR / zip_name
        if not zip_path.exists():
            print(f"[WARN] No se encontro {zip_path}")
            continue

        extract_to = TEMP_DIR / comp_name
        print(f"\nExtrayendo {zip_name} a {extract_to}...")
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(extract_to)

        # Buscar subcarpetas de train, valid/val, test
        splits = {
            'train': ['train'],
            'val': ['valid', 'val', 'test']
        }

        for target_split, folder_names in splits.items():
            dst_img_target = train_img_dir if target_split == 'train' else val_img_dir
            dst_lbl_target = train_lbl_dir if target_split == 'train' else val_lbl_dir

            for fn in folder_names:
                img_dir = extract_to / fn / 'images'
                lbl_dir = extract_to / fn / 'labels'
                if not img_dir.exists():
                    continue

                for img_p in img_dir.glob('*.*'):
                    if img_p.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
                        continue
                    lbl_p = lbl_dir / f"{img_p.stem}.txt"
                    
                    dst_stem = f"new_rf_{comp_name}_{img_p.stem}"
                    dst_img = dst_img_target / f"{dst_stem}{img_p.suffix}"
                    dst_lbl = dst_lbl_target / f"{dst_stem}.txt"

                    shutil.copy2(img_p, dst_img)
                    n_boxes = remap_to_class_zero(lbl_p, dst_lbl)
                    total_boxes += n_boxes
                    if target_split == 'train':
                        total_added_train += 1
                    else:
                        total_added_val += 1

        print(f"  [OK] {comp_name} procesado e integrado exitosamente.")

    # Limpiar temporales
    if TEMP_DIR.exists():
        shutil.rmtree(TEMP_DIR)

    # Eliminar caches de Ultralytics
    for split in ['train', 'val']:
        c = DATASET_DIR / split / 'labels.cache'
        if c.exists():
            c.unlink()

    print("\n" + "=" * 65)
    print("RESUMEN DE INTEGRACION DE NUEVOS DATASETS:")
    print(f"  Imagenes agregadas a train: {total_added_train}")
    print(f"  Imagenes agregadas a val:   {total_added_val}")
    print(f"  Total cajas de componentes: {total_boxes}")
    n_train = len(list(train_img_dir.glob('*.*')))
    n_val = len(list(val_img_dir.glob('*.*')))
    print(f"  Total final dataset: {n_train} train, {n_val} val ({n_train + n_val} imagenes)")
    print("=" * 65)

if __name__ == '__main__':
    main()
