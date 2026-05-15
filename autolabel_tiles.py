"""
Auto-etiqueta todos los tiles de png-dataset/tiles/ usando un modelo YOLO
existente y genera archivos .txt en formato YOLO listos para revisar y entrenar.

Resultado en png-dataset/labels/:
    <mismo nombre que el tile>.txt  con lineas:  class_id cx cy w h  (normalizados)

Tambien genera data.yaml con las clases del modelo.

Uso basico:
    python autolabel_tiles.py

Con opciones:
    python autolabel_tiles.py \\
        --modelo   medium.pt \\
        --tiles    png-dataset/tiles \\
        --out      png-dataset/labels \\
        --conf     0.25 \\
        --device   cpu

Tips:
  --conf 0.25  -> umbral bajo, captura mas (vas a borrar falsos positivos al revisar)
  --conf 0.5   -> mas limpio, puede perderse algunos simbolos
  --device cuda:0  -> mucho mas rapido si tenes GPU
"""

import os
import sys
import time
import argparse
from pathlib import Path


def generar_data_yaml(labels_dir, names, yaml_path):
    """Genera el data.yaml con las clases del modelo."""
    tiles_dir  = Path(labels_dir).parent / "tiles"
    images_dir = Path(labels_dir).parent / "images"

    lines = [
        f"# Generado automaticamente por autolabel_tiles.py",
        f"",
        f"path: {Path(labels_dir).parent.resolve()}",
        f"train: tiles   # reemplazar por images/train al hacer el split",
        f"val:   tiles   # reemplazar por images/val",
        f"",
        f"nc: {len(names)}",
        f"names: {list(names.values()) if isinstance(names, dict) else list(names)}",
    ]
    with open(yaml_path, "w") as f:
        f.write("\n".join(lines))
    print(f"[yaml] data.yaml -> {yaml_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Auto-etiqueta tiles con un modelo YOLO -> .txt YOLO format."
    )
    parser.add_argument("--modelo",   default="medium.pt",
                        help="Modelo .pt a usar (default: medium.pt)")
    parser.add_argument("--tiles",    default="png-dataset/tiles",
                        help="Carpeta con los PNG tiles (default: png-dataset/tiles)")
    parser.add_argument("--out",      default="png-dataset/labels",
                        help="Carpeta donde guardar los .txt (default: png-dataset/labels)")
    parser.add_argument("--conf",     type=float, default=0.25,
                        help="Confianza minima para incluir una deteccion (default 0.25)")
    parser.add_argument("--iou",      type=float, default=0.5,
                        help="IoU para NMS interno (default 0.5)")
    parser.add_argument("--device",   default="cpu",
                        help="cpu o cuda:0 (default: cpu)")
    parser.add_argument("--force",    action="store_true",
                        help="Re-etiqueta aunque el .txt ya exista")
    parser.add_argument("--batch",    type=int, default=32,
                        help="Tiles por batch (default 32, subi con GPU)")
    args = parser.parse_args()

    tiles_dir  = Path(args.tiles)
    labels_dir = Path(args.out)
    modelo_path = Path(args.modelo)

    # Validaciones
    if not modelo_path.exists():
        print(f"[ERROR] No se encontro el modelo: {modelo_path.resolve()}")
        sys.exit(1)
    if not tiles_dir.exists():
        print(f"[ERROR] No se encontro la carpeta de tiles: {tiles_dir.resolve()}")
        sys.exit(1)

    tiles = sorted(tiles_dir.glob("*.png"))
    if not tiles:
        print(f"[AVISO] No hay PNG en {tiles_dir.resolve()}")
        sys.exit(0)

    labels_dir.mkdir(parents=True, exist_ok=True)

    # Cargamos el modelo
    print(f"Cargando modelo {modelo_path} ...")
    from ultralytics import YOLO
    model = YOLO(str(modelo_path))
    names = model.names   # dict {0: 'clase_a', 1: 'clase_b', ...}

    print(f"Tiles     : {len(tiles)}")
    print(f"Clases    : {names}")
    print(f"Conf      : {args.conf}")
    print(f"Device    : {args.device}")
    print(f"Batch     : {args.batch}")
    print()

    # Generamos data.yaml al lado de labels/
    yaml_path = labels_dir.parent / "data.yaml"
    generar_data_yaml(labels_dir, names, yaml_path)

    # Filtramos tiles que ya tienen label (si no --force)
    pendientes = []
    saltados   = 0
    for tile in tiles:
        label_path = labels_dir / tile.with_suffix(".txt").name
        if label_path.exists() and not args.force:
            saltados += 1
        else:
            pendientes.append(tile)

    if saltados:
        print(f"[skip] {saltados} tiles ya etiquetados (usa --force para re-etiquetar)")

    if not pendientes:
        print("Nada que procesar.")
        return

    # Inferencia en batches
    t0 = time.time()
    total_dets  = 0
    total_saved = 0
    total_empty = 0

    n_batches = (len(pendientes) + args.batch - 1) // args.batch

    for b in range(n_batches):
        batch_paths = pendientes[b * args.batch : (b + 1) * args.batch]
        batch_strs  = [str(p) for p in batch_paths]

        results = model.predict(
            source=batch_strs,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            verbose=False,
            save=False,
        )

        for tile_path, result in zip(batch_paths, results):
            label_path = labels_dir / tile_path.with_suffix(".txt").name
            boxes = result.boxes

            if boxes is None or len(boxes) == 0:
                # Archivo vacio = imagen negativa (sin detecciones)
                label_path.write_text("")
                total_empty += 1
                continue

            # Convertimos a formato YOLO normalizado
            img_h, img_w = result.orig_shape
            lines = []
            xyxy  = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            clses = boxes.cls.cpu().numpy().astype(int)

            for j in range(len(boxes)):
                x1, y1, x2, y2 = xyxy[j]
                cx = (x1 + x2) / 2.0 / img_w
                cy = (y1 + y2) / 2.0 / img_h
                w  = (x2 - x1) / img_w
                h  = (y2 - y1) / img_h
                lines.append(f"{clses[j]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

            label_path.write_text("\n".join(lines))
            total_dets  += len(lines)
            total_saved += 1

        # Progreso
        done = min((b + 1) * args.batch, len(pendientes))
        pct  = done / len(pendientes) * 100
        eta  = (time.time() - t0) / done * (len(pendientes) - done) if done < len(pendientes) else 0
        print(f"[{done:>4}/{len(pendientes)}] {pct:5.1f}%  "
              f"dets acum={total_dets}  ETA={eta:.0f}s")

    elapsed = time.time() - t0
    print()
    print("=" * 60)
    print(f"Terminado en {elapsed:.1f}s")
    print(f"  Con detecciones : {total_saved}")
    print(f"  Sin detecciones : {total_empty}  (tiles negativos — utiles para entrenamiento)")
    print(f"  Detecciones tot : {total_dets}")
    print(f"  data.yaml       : {yaml_path}")
    print()
    print("Proximo paso: revisar labels en Label Studio o Roboflow,")
    print("hacer el split train/val, y correr:")
    print(f"  yolo detect train model=yolov8m.pt data={yaml_path} epochs=100 imgsz=640")
    print("=" * 60)


if __name__ == "__main__":
    main()
