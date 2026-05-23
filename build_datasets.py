"""
build_datasets.py — Genera datasets multi + singles desde synth_all.

Ejecutar UNA VEZ en Windows (cmd o PowerShell) desde CLAUDIO_AI/:
    python build_datasets.py

Crea la carpeta datasets_out/ con:
  datasets_out/multi/           → modelo 7 clases juntas
  datasets_out/single_<clase>/  → un modelo por clase (class_id=0)

Las imágenes se comparten desde synth_all/ (sin duplicar PNGs).
Los data.yaml apuntan directamente a synth_all/images/.
"""

import json, random, shutil
from pathlib import Path

random.seed(42)

HERE     = Path(__file__).parent.resolve()
SRC      = HERE / "synth_all"
DST      = HERE / "datasets_out"
N_SINGLE = 350    # imágenes de train por clase para singles
VAL_FRAC = 0.15

CLASSES = [
    "ojo_de_buey",
    "interruptor_termomagnetico",
    "interruptor_diferencial",
    "interruptor_seleccionador_manual",
    "seleccionador_bajo_carga",
    "multi_medidor",
    "interruptor_motorizado",
]

print("=" * 60)
print("build_datasets.py")
print("=" * 60)

# ── 1. Leer todos los labels ──────────────────────────────────────────────────
print("\n[1/4] Leyendo labels de synth_all ...")
img_data = {}
for split in ("train", "val"):
    img_data[split] = {}
    lbl_dir = SRC / "labels" / split
    for lbl in sorted(lbl_dir.glob("*.txt")):
        by_cls = {}
        for line in lbl.read_text().strip().splitlines():
            p = line.split()
            if not p: continue
            c = int(p[0])
            by_cls.setdefault(c, []).append("0 " + " ".join(p[1:]))
        img_data[split][lbl.stem] = by_cls
    print(f"    {split}: {len(img_data[split])} imágenes")

# ── 2. Helper ─────────────────────────────────────────────────────────────────
def write_yaml(path, train_path, val_path, nc, names):
    """data.yaml con rutas absolutas a las imágenes en synth_all/."""
    path.write_text(
        f"train: {train_path}\n"
        f"val:   {val_path}\n\n"
        f"nc: {nc}\nnames: {json.dumps(names)}\n",
        encoding="utf-8"
    )

# ── 3. Dataset MULTI ──────────────────────────────────────────────────────────
print("\n[2/4] Creando datasets_out/multi/ ...")
multi = DST / "multi"
shutil.rmtree(multi, ignore_errors=True)
for split in ("train", "val"):
    lbl_out = multi / "labels" / split
    lbl_out.mkdir(parents=True, exist_ok=True)
    for stem, by_cls in img_data[split].items():
        lines = []
        for cls_id in sorted(by_cls):
            for line in by_cls[cls_id]:
                lines.append(line.replace("0 ", f"{cls_id} ", 1))
        (lbl_out / f"{stem}.txt").write_text("\n".join(lines) + "\n")
write_yaml(
    multi / "data.yaml",
    str(SRC / "images" / "train"),
    str(SRC / "images" / "val"),
    len(CLASSES), CLASSES
)
print(f"    train: {len(img_data['train'])} / val: {len(img_data['val'])}")

# ── 4. Datasets SINGLE ────────────────────────────────────────────────────────
print("\n[3/4] Creando datasets_out/single_<clase>/ ...")
stats = {}
for cls_id, cls_name in enumerate(CLASSES):
    d = DST / f"single_{cls_name}"
    shutil.rmtree(d, ignore_errors=True)

    stems_train = [s for s, bc in img_data["train"].items() if cls_id in bc]
    random.shuffle(stems_train)
    stems_train = stems_train[:N_SINGLE]
    stems_val   = [s for s, bc in img_data["val"].items()   if cls_id in bc]
    stems_val   = stems_val[:max(1, int(len(stems_train) * VAL_FRAC))]

    for split, stems in [("train", stems_train), ("val", stems_val)]:
        lbl_out = d / "labels" / split
        lbl_out.mkdir(parents=True, exist_ok=True)
        src_map = img_data[split]
        for stem in stems:
            if stem not in src_map or cls_id not in src_map[stem]: continue
            lines = src_map[stem][cls_id]
            (lbl_out / f"{stem}.txt").write_text("\n".join(lines) + "\n")

    write_yaml(
        d / "data.yaml",
        str(SRC / "images" / "train"),
        str(SRC / "images" / "val"),
        1, [cls_name]
    )
    stats[cls_name] = {"train": len(stems_train), "val": len(stems_val)}
    print(f"    single_{cls_name:<42}  train={len(stems_train):>3}  val={len(stems_val):>3}")

# ── 5. Resumen ────────────────────────────────────────────────────────────────
print("\n[4/4] Listo.")
print(f"\ndatasets_out/")
print(f"  multi/               nc=7   train={len(img_data['train'])}  val={len(img_data['val'])}")
for n in CLASSES:
    c = stats[n]
    print(f"  single_{n:<42}  train={c['train']:>3}  val={c['val']:>3}")

print("\nPara entrenar:")
print("  # Multi-clase:")
print(f"  yolo train model=yolov8n.pt data=datasets_out/multi/data.yaml epochs=100 imgsz=640 batch=16 device=0")
print("\n  # Single-clase (ejemplo ojo_de_buey):")
print(f"  yolo train model=yolov8n.pt data=datasets_out/single_ojo_de_buey/data.yaml epochs=100 imgsz=640 batch=16 device=0")
