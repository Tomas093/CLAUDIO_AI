"""
Genera datasets separados por clase a partir del split ya existente.

Resultado:
    png-dataset/
        dataset_termomagnetico/
            images/train/   images/val/
            labels/train/   labels/val/
            data.yaml        (nc=1)
        dataset_diferencial/
            images/train/   images/val/
            labels/train/   labels/val/
            data.yaml        (nc=1)

- Solo incluye tiles que tengan al menos una anotacion de esa clase.
- Remapea el class_id a 0 (unica clase del dataset).
- Respeta el mismo split train/val que ya existe en png-dataset/images/.

Uso:
    python split_per_class.py

Con opciones:
    python split_per_class.py \\
        --src   png-dataset \\
        --names interruptor_termomagnetico interruptor_diferencial
"""

import shutil
import argparse
from pathlib import Path


def filtrar_y_remap(label_src: Path, class_id: int) -> str | None:
    """
    Lee un .txt YOLO, filtra solo las lineas de class_id y lo remapea a 0.
    Devuelve el contenido nuevo, o None si no hay ninguna anotacion de esa clase.
    """
    text = label_src.read_text().strip()
    if not text:
        return None
    lineas_filtradas = []
    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        if int(parts[0]) == class_id:
            # Remap a clase 0
            lineas_filtradas.append("0 " + " ".join(parts[1:]))
    return "\n".join(lineas_filtradas) if lineas_filtradas else None


def procesar_split(split: str, src_root: Path, dst_root: Path, class_id: int):
    img_src = src_root / "images" / split
    lbl_src = src_root / "labels" / split
    img_dst = dst_root / "images" / split
    lbl_dst = dst_root / "labels" / split
    img_dst.mkdir(parents=True, exist_ok=True)
    lbl_dst.mkdir(parents=True, exist_ok=True)

    tiles = sorted(img_src.glob("*.png"))
    copiados = saltados = 0

    for tile_path in tiles:
        label_path = lbl_src / tile_path.with_suffix(".txt").name
        if not label_path.exists():
            saltados += 1
            continue

        nuevo_contenido = filtrar_y_remap(label_path, class_id)
        if nuevo_contenido is None:
            saltados += 1
            continue

        # Copiar imagen
        shutil.copy2(tile_path, img_dst / tile_path.name)
        # Escribir label remapeado
        (lbl_dst / label_path.name).write_text(nuevo_contenido)
        copiados += 1

    return copiados, saltados


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src",   default="png-dataset",
                        help="Raiz del dataset con el split ya hecho")
    parser.add_argument("--names", nargs="+",
                        default=["interruptor_termomagnetico", "interruptor_diferencial"])
    args = parser.parse_args()

    src_root = Path(args.src)

    for class_id, class_name in enumerate(args.names):
        print(f"\n{'='*60}")
        print(f"Clase {class_id}: {class_name}")
        print(f"{'='*60}")

        dst_root = src_root / f"dataset_{class_name}"
        dst_root.mkdir(parents=True, exist_ok=True)

        total_train, skip_train = procesar_split("train", src_root, dst_root, class_id)
        total_val,   skip_val   = procesar_split("val",   src_root, dst_root, class_id)

        print(f"  Train: {total_train} tiles con clase  ({skip_train} sin esta clase, saltados)")
        print(f"  Val  : {total_val}  tiles con clase  ({skip_val}  sin esta clase, saltados)")

        # data.yaml
        yaml_path = dst_root / "data.yaml"
        yaml_path.write_text(f"""# Dataset de una sola clase: {class_name}

path: {dst_root.resolve().as_posix()}
train: images/train
val:   images/val

nc: 1
names: ["{class_name}"]
""")
        print(f"  data.yaml -> {yaml_path.resolve()}")
        print()
        print(f"  Para entrenar:")
        print(f"    yolo detect train model=yolov8m.pt data={yaml_path.resolve()} epochs=100 imgsz=640 batch=16 device=0")

    print(f"\n{'='*60}")
    print("Listo. Tres datasets disponibles:")
    print(f"  1. {src_root}/data.yaml              (ambas clases)")
    for name in args.names:
        print(f"  2. {src_root}/dataset_{name}/data.yaml  (solo {name})")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
