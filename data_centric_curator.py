import os
import shutil
from pathlib import Path
import numpy as np

DATASET_DIR = Path('c:/Users/Tomas/Documents/LAB3/CLAUDIO_AI/train-maker/dataset_unified_componente')

def audit_and_curate():
    print("=" * 70)
    print("DATA-CENTRIC AI: CURACION Y CONTROL DE CALIDAD DEL DATASET")
    print("=" * 70)

    for split in ['train', 'val']:
        img_dir = DATASET_DIR / split / 'images'
        lbl_dir = DATASET_DIR / split / 'labels'

        all_lbls = sorted(list(lbl_dir.glob('*.txt')))
        print(f"\n[{split.upper()}] Procesando {len(all_lbls)} muestras...")

        removed_huge = 0
        removed_invalid = 0
        removed_excess_neg = 0
        cleaned_boxes = 0
        valid_positives = 0
        valid_negatives = 0

        # Identificar negativos
        neg_files = []
        pos_files = []

        for p in all_lbls:
            content = p.read_text(encoding='utf-8', errors='ignore').strip()
            if not content:
                neg_files.append(p)
            else:
                pos_files.append(p)

        print(f"  Inicial: {len(pos_files)} positivos, {len(neg_files)} negativos ({len(neg_files)/len(all_lbls)*100:.1f}%)")

        # 1. Curar Positivos: filtrar cajas gigantes (>320px), invalidas o diminutas (<6px)
        for p in pos_files:
            content = p.read_text(encoding='utf-8', errors='ignore').strip()
            lines = content.split('\n')
            new_lines = []
            has_huge = False

            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    try:
                        cls_id = int(parts[0])
                        xc, yc, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])

                        # Descartar si excede los limites [0, 1]
                        if xc < 0 or xc > 1 or yc < 0 or yc > 1 or w <= 0 or h <= 0 or w > 1 or h > 1:
                            removed_invalid += 1
                            continue

                        # Descartar si es un recorte gigante (columnas completas de Roboflow > 320 px)
                        if w > 0.50 or h > 0.50:
                            has_huge = True
                            break

                        # Descartar ruido de borde diminuto (< 6.4 px)
                        if w < 0.01 and h < 0.01:
                            continue

                        new_lines.append(f"0 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
                    except Exception:
                        removed_invalid += 1
                        continue

            stem = p.stem
            img_file = img_dir / f"{stem}.jpg"
            if not img_file.exists():
                img_file = img_dir / f"{stem}.png"

            if has_huge:
                # Eliminar muestra con cajas gigantes aberrantes
                p.unlink(missing_ok=True)
                if img_file.exists():
                    img_file.unlink(missing_ok=True)
                removed_huge += 1
            elif not new_lines:
                # Si se quedo sin cajas validas, eliminar
                p.unlink(missing_ok=True)
                if img_file.exists():
                    img_file.unlink(missing_ok=True)
                removed_invalid += 1
            else:
                p.write_text('\n'.join(new_lines) + '\n', encoding='utf-8')
                cleaned_boxes += len(new_lines)
                valid_positives += 1

        print(f"  Positivos depurados: {valid_positives} validos. Eliminados: {removed_huge} con cajas gigantes, {removed_invalid} invalidos.")

        # 2. Curar Negativos: Presupuesto optimo del 7%
        # Conservar solo Hard Negatives de alto valor (tablas, textos de cotas, conductores complejos)
        # Descartar fondos vacios repetitivos
        target_neg_count = max(10, int(round(valid_positives * 0.07)))
        print(f"  Presupuesto optimo de negativos (7% de Ultralytics): objetivo = {target_neg_count} (de {len(neg_files)} existentes)")

        # Priorizar negativos que tengan 'table', 'fptext', o 'fl02' en su nombre (Hard Negatives reales)
        hard_negs = [p for p in neg_files if any(k in p.name.lower() for k in ['table', 'fptext', 'fl02', 'tsss'])]
        other_negs = [p for p in neg_files if p not in hard_negs]

        selected_negs = set()
        for p in hard_negs:
            if len(selected_negs) < target_neg_count:
                selected_negs.add(p)

        for p in other_negs:
            if len(selected_negs) < target_neg_count:
                selected_negs.add(p)

        # Eliminar negativos excedentes
        for p in neg_files:
            stem = p.stem
            img_file = img_dir / f"{stem}.jpg"
            if not img_file.exists():
                img_file = img_dir / f"{stem}.png"

            if p in selected_negs:
                valid_negatives += 1
            else:
                p.unlink(missing_ok=True)
                if img_file.exists():
                    img_file.unlink(missing_ok=True)
                removed_excess_neg += 1

        print(f"  Negativos depurados: {valid_negatives} conservados (7%), {removed_excess_neg} excedentes eliminados.")

        total_final = valid_positives + valid_negatives
        print(f"  FINAL [{split.upper()}]: {total_final} muestras ({valid_positives} pos, {valid_negatives} neg -> {valid_negatives/total_final*100:.1f}%)")

    print("\n" + "=" * 70)
    print("CURACION COMPLETADA CON EXITO")
    print("=" * 70)

if __name__ == '__main__':
    audit_and_curate()
