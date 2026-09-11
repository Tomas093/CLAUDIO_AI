# build_unified_dataset.py
import os, sys, glob, random, shutil
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / 'dataset_unified_componente'

COMPONENTS = [
    'fotocelula',
    'grupo_electrogeno',
    'instrumento_de_medicion_multifuncion',
    'interruptor_diferencial',
    'interruptor_motorizado',
    'interruptor_temporizado',
    'interruptor_termomagnetico',
    'ojo_de_buey',
    'seccionador_bajo_carga',
    'tablero_de_transferencia_automatica',
]

def clean_and_create_dirs():
    if OUTPUT_DIR.exists():
        print(f'Limpiando directorio anterior: {OUTPUT_DIR}')
        shutil.rmtree(OUTPUT_DIR)
    for split in ['train', 'val']:
        (OUTPUT_DIR / split / 'images').mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / split / 'labels').mkdir(parents=True, exist_ok=True)

def remap_and_write_label(src_txt: Path, dst_txt: Path):
    if not src_txt.exists() or src_txt.stat().st_size == 0:
        dst_txt.write_text('', encoding='utf-8')
        return 0
    valid_lines = []
    with open(src_txt, 'r', encoding='utf-8', errors='ignore') as f_in:
        for line in f_in:
            parts = line.strip().split()
            if len(parts) >= 5:
                try:
                    x, y, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                    if 0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1:
                        valid_lines.append(f'0 {x:.6f} {y:.6f} {w:.6f} {h:.6f}')
                except ValueError:
                    continue
    with open(dst_txt, 'w', encoding='utf-8') as f_out:
        f_out.write('\n'.join(valid_lines) + ('\n' if valid_lines else ''))
    return len(valid_lines)

def process_real_datasets():
    print('\n--- Procesando datasets reales (dataset_real_*_rescaled) ---')
    counts = {'train': 0, 'val': 0, 'boxes': 0}
    for comp in COMPONENTS:
        comp_dir = BASE_DIR / f'dataset_real_{comp}_rescaled'
        if not comp_dir.exists():
            comp_dir = BASE_DIR / f'dataset_real_{comp}'
        if not comp_dir.exists():
            continue

        train_imgs = list((comp_dir / 'train' / 'images').glob('*.*'))
        for img_p in train_imgs:
            lbl_p = comp_dir / 'train' / 'labels' / f'{img_p.stem}.txt'
            dst_name = f'real_{comp}_{img_p.name}'
            dst_img = OUTPUT_DIR / 'train' / 'images' / dst_name
            dst_lbl = OUTPUT_DIR / 'train' / 'labels' / f'{Path(dst_name).stem}.txt'
            shutil.copy2(img_p, dst_img)
            n_boxes = remap_and_write_label(lbl_p, dst_lbl)
            counts['train'] += 1
            counts['boxes'] += n_boxes

        val_imgs = list((comp_dir / 'val' / 'images').glob('*.*'))
        test_imgs = list((comp_dir / 'test' / 'images').glob('*.*')) if (comp_dir / 'test' / 'images').exists() else []
        for img_p in val_imgs + test_imgs:
            lbl_p = (img_p.parent.parent / 'labels' / f'{img_p.stem}.txt')
            dst_name = f'real_{comp}_{img_p.name}'
            dst_img = OUTPUT_DIR / 'val' / 'images' / dst_name
            dst_lbl = OUTPUT_DIR / 'val' / 'labels' / f'{Path(dst_name).stem}.txt'
            shutil.copy2(img_p, dst_img)
            n_boxes = remap_and_write_label(lbl_p, dst_lbl)
            counts['val'] += 1
            counts['boxes'] += n_boxes

        print(f'  [REAL] {comp:<35}: {len(train_imgs)} train, {len(val_imgs)+len(test_imgs)} val')
    return counts

def process_synthetic_datasets(samples_train=500, samples_val=100, seed=42):
    print(f'\n--- Procesando datasets sinteticos ({samples_train} train, {samples_val} val por componente) ---')
    rng = random.Random(seed)
    counts = {'train': 0, 'val': 0, 'boxes': 0}
    for comp in COMPONENTS:
        synth_dir = BASE_DIR / f'dataset_sintetico_{comp}'
        if not synth_dir.exists():
            continue

        train_imgs = sorted(list((synth_dir / 'train' / 'images').glob('*.*')))
        selected_train = rng.sample(train_imgs, min(samples_train, len(train_imgs)))
        for img_p in selected_train:
            lbl_p = synth_dir / 'train' / 'labels' / f'{img_p.stem}.txt'
            dst_name = f'synth_{comp}_{img_p.name}'
            dst_img = OUTPUT_DIR / 'train' / 'images' / dst_name
            dst_lbl = OUTPUT_DIR / 'train' / 'labels' / f'{Path(dst_name).stem}.txt'
            shutil.copy2(img_p, dst_img)
            n_boxes = remap_and_write_label(lbl_p, dst_lbl)
            counts['train'] += 1
            counts['boxes'] += n_boxes

        val_imgs = sorted(list((synth_dir / 'val' / 'images').glob('*.*')))
        selected_val = rng.sample(val_imgs, min(samples_val, len(val_imgs)))
        for img_p in selected_val:
            lbl_p = synth_dir / 'val' / 'labels' / f'{img_p.stem}.txt'
            dst_name = f'synth_{comp}_{img_p.name}'
            dst_img = OUTPUT_DIR / 'val' / 'images' / dst_name
            dst_lbl = OUTPUT_DIR / 'val' / 'labels' / f'{Path(dst_name).stem}.txt'
            shutil.copy2(img_p, dst_img)
            n_boxes = remap_and_write_label(lbl_p, dst_lbl)
            counts['val'] += 1
            counts['boxes'] += n_boxes

        print(f'  [SYNTH] {comp:<35}: {len(selected_train)} train, {len(selected_val)} val')
    return counts

def process_background_negatives(n_train=800, n_val=150, seed=42):
    print(f'\n--- Procesando fondos negativos ({n_train} train, {n_val} val) ---')
    bg_dir = BASE_DIR / 'output' / 'backgrounds'
    if not bg_dir.exists():
        return
    all_bgs = sorted(list(bg_dir.glob('*.jpg')) + list(bg_dir.glob('*.png')))
    rng = random.Random(seed)
    rng.shuffle(all_bgs)
    selected_train = all_bgs[:n_train]
    selected_val = all_bgs[n_train:n_train + n_val]
    for img_p in selected_train:
        dst_name = f'bg_{img_p.name}'
        shutil.copy2(img_p, OUTPUT_DIR / 'train' / 'images' / dst_name)
        (OUTPUT_DIR / 'train' / 'labels' / f'{Path(dst_name).stem}.txt').write_text('', encoding='utf-8')
    for img_p in selected_val:
        dst_name = f'bg_{img_p.name}'
        shutil.copy2(img_p, OUTPUT_DIR / 'val' / 'images' / dst_name)
        (OUTPUT_DIR / 'val' / 'labels' / f'{Path(dst_name).stem}.txt').write_text('', encoding='utf-8')
    print(f'  Fondos negativos agregados: {len(selected_train)} train, {len(selected_val)} val.')

def write_data_yaml():
    yaml_path = OUTPUT_DIR / 'data.yaml'
    norm_path = str(OUTPUT_DIR.resolve()).replace('\\', '/')
    lines = [
        '# Dataset Unificado Clase Unica: componente',
        f'path: {norm_path}',
        'train: train/images',
        'val: val/images',
        '',
        'nc: 1',
        'names:',
        '  0: componente'
    ]
    yaml_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'\nYAML generado en {yaml_path}')
    return yaml_path

def main():
    clean_and_create_dirs()
    c_real = process_real_datasets()
    c_synth = process_synthetic_datasets(samples_train=500, samples_val=100)
    process_background_negatives(n_train=800, n_val=150)
    yaml_p = write_data_yaml()
    n_train = len(list((OUTPUT_DIR / 'train' / 'images').glob('*.*')))
    n_val = len(list((OUTPUT_DIR / 'val' / 'images').glob('*.*')))
    print(f'RESUMEN: {n_train} train, {n_val} val, Total={n_train+n_val}')

if __name__ == '__main__':
    main()
