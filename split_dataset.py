"""
Divide el dataset en train/val agrupando por plano (DXF de origen),
no por tile individual, para evitar data leakage.

Resultado:
    png-dataset/
        images/
            train/   <-- tiles de entrenamiento
            val/     <-- tiles de validacion
        labels/
            train/   <-- .txt correspondientes
            val/
        data.yaml    <-- actualizado con rutas correctas

Uso basico:
    python split_dataset.py

Con opciones:
    python split_dataset.py \\
        --tiles   png-dataset/tiles \\
        --labels  png-dataset/labels \\
        --out     png-dataset \\
        --val     0.2 \\
        --seed    42 \\
        --names   interruptor_termomagnetico interruptor_diferencial
"""

import re
import shutil
import random
import argparse
from pathlib import Path
from collections import defaultdict


def get_prefix(filename: str) -> str:
    """
    Extrae el prefijo del plano a partir del nombre del tile.
    Ej: '1er_Piso__TS1AE_0003.png' -> '1er_Piso__TS1AE'
    """
    # Elimina el sufijo _XXXX.png al final
    return re.sub(r'_\d{4}\.png$', '', filename)


def main():
    parser = argparse.ArgumentParser(
        description="Split train/val por plano para evitar data leakage."
    )
    parser.add_argument("--tiles",   default="png-dataset/tiles")
    parser.add_argument("--labels",  default="png-dataset/labels")
    parser.add_argument("--out",     default="png-dataset")
    parser.add_argument("--val",     type=float, default=0.2,
                        help="Fraccion de planos para validacion (default 0.2 = 20%%)")
    parser.add_argument("--seed",    type=int, default=42)
    parser.add_argument("--names",   nargs="+",
                        default=["interruptor_termomagnetico", "interruptor_diferencial"])
    parser.add_argument("--copy",    action="store_true",
                        help="Copia archivos en vez de hacer symlinks (necesario en Windows)")
    args = parser.parse_args()

    tiles_dir  = Path(args.tiles)
    labels_dir = Path(args.labels)
    out_root   = Path(args.out)

    # Directorios de salida
    img_train  = out_root / "images" / "train"
    img_val    = out_root / "images" / "val"
    lbl_train  = out_root / "labels" / "train"
    lbl_val    = out_root / "labels" / "val"
    for d in [img_train, img_val, lbl_train, lbl_val]:
        d.mkdir(parents=True, exist_ok=True)

    # Agrupamos tiles por plano
    tiles = sorted(tiles_dir.glob("*.png"))
    # Filtramos archivos tipo "copia" u otros que no sigan el patron
    tiles = [t for t in tiles if re.search(r'_\d{4}\.png$', t.name)]

    plano_tiles = defaultdict(list)
    for tile in tiles:
        prefix = get_prefix(tile.name)
        plano_tiles[prefix].append(tile)

    planos = sorted(plano_tiles.keys())
    random.seed(args.seed)
    random.shuffle(planos)

    n_val   = max(1, round(len(planos) * args.val))
    n_train = len(planos) - n_val

    val_planos   = set(planos[:n_val])
    train_planos = set(planos[n_val:])

    print(f"Planos totales : {len(planos)}")
    print(f"  Train        : {n_train}  ({', '.join(sorted(train_planos)[:5])}{'...' if n_train > 5 else ''})")
    print(f"  Val          : {n_val}   ({', '.join(sorted(val_planos))})")
    print()

    def transferir(src: Path, dst_dir: Path):
        dst = dst_dir / src.name
        if dst.exists():
            dst.unlink()
        shutil.copy2(src, dst)

    train_imgs = train_lbls = val_imgs = val_lbls = 0
    sin_label = 0

    for prefix, tile_list in plano_tiles.items():
        is_val = prefix in val_planos
        img_dst = img_val   if is_val else img_train
        lbl_dst = lbl_val   if is_val else lbl_train

        for tile_path in tile_list:
            transferir(tile_path, img_dst)
            label_path = labels_dir / tile_path.with_suffix(".txt").name
            if label_path.exists():
                transferir(label_path, lbl_dst)
            else:
                # Crea .txt vacio (tile negativo)
                (lbl_dst / tile_path.with_suffix(".txt").name).write_text("")
                sin_label += 1

            if is_val:
                val_imgs += 1
            else:
                train_imgs += 1

    print(f"Tiles copiados:")
    print(f"  Train images : {train_imgs}")
    print(f"  Val   images : {val_imgs}")
    if sin_label:
        print(f"  Sin label    : {sin_label} (creados vacios)")
    print()

    # data.yaml actualizado
    yaml_path = out_root / "data.yaml"
    out_abs   = out_root.resolve()
    yaml_content = f"""# Generado por split_dataset.py

path: {out_abs.as_posix()}
train: images/train
val:   images/val

nc: {len(args.names)}
names: {args.names}
"""
    yaml_path.write_text(yaml_content)
    print(f"data.yaml guardado en: {yaml_path.resolve()}")
    print()
    print("=" * 60)
    print("PROXIMO PASO: entrenar")
    print("=" * 60)
    print()
    print("  yolo detect train \\")
    print(f"      model=yolov8m.pt \\")
    print(f"      data={yaml_path.resolve()} \\")
    print(f"      epochs=100 \\")
    print(f"      imgsz=640 \\")
    print(f"      batch=16 \\")
    print(f"      device=0")
    print()
    print("(si no tenes GPU: device=cpu, batch=8)")
    print("=" * 60)


if __name__ == "__main__":
    main()
