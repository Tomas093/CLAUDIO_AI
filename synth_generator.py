"""
synth_generator.py — Generador de imágenes sintéticas para YOLO (v2, optimizado).

Uso:
    python synth_generator.py \\
        --datasets png-dataset/dataset_ojo_de_buey png-dataset/dataset_interruptor_termomagnetico ... \\
        --out synth_dataset --n-images 2000 --max-sym-per-img 3
"""

import os, sys, json, random, argparse, re, pickle, hashlib
import numpy as np
from PIL import Image, ImageEnhance
from pathlib import Path
from collections import Counter

TILE_SIZE      = 640
MIN_SYM_PX     = 20
MAX_SYM_PX     = 220
PAD_FACTOR     = 0.20


def yolo_to_px(cx, cy, w, h, W, H):
    x1 = int((cx - w/2) * W); y1 = int((cy - h/2) * H)
    x2 = int((cx + w/2) * W); y2 = int((cy + h/2) * H)
    return x1, y1, x2, y2

def px_to_yolo(x1, y1, x2, y2, W, H):
    cx = ((x1+x2)/2)/W; cy = ((y1+y2)/2)/H
    w  = (x2-x1)/W;     h  = (y2-y1)/H
    return cx, cy, w, h

def iou(a, b):
    ix1=max(a[0],b[0]); iy1=max(a[1],b[1])
    ix2=min(a[2],b[2]); iy2=min(a[3],b[3])
    inter=max(0,ix2-ix1)*max(0,iy2-iy1)
    ua=(a[2]-a[0])*(a[3]-a[1]); ub=(b[2]-b[0])*(b[3]-b[1])
    return inter/(ua+ub-inter+1e-6)


def load_dataset(dataset_dir):
    dataset_dir = Path(dataset_dir)
    img_dir = dataset_dir/"images"/"train"
    lbl_dir = dataset_dir/"labels"/"train"
    records = []
    for img_path in sorted(img_dir.glob("*.png")):
        lbl_path = lbl_dir/(img_path.stem+".txt")
        if not lbl_path.exists(): continue
        annots = []
        for line in lbl_path.read_text().strip().splitlines():
            parts = line.strip().split()
            if len(parts)==5:
                annots.append((int(parts[0]), *map(float, parts[1:])))
        if annots:
            records.append((img_path, annots))
    return records


def extract_patches(records, class_id_map=None):
    patches = []
    for img_path, annots in records:
        img = Image.open(img_path).convert("RGBA")
        W, H = img.size
        for cls, cx, cy, w, h in annots:
            x1,y1,x2,y2 = yolo_to_px(cx,cy,w,h,W,H)
            pad_x = int((x2-x1)*PAD_FACTOR); pad_y = int((y2-y1)*PAD_FACTOR)
            px1=max(0,x1-pad_x); py1=max(0,y1-pad_y)
            px2=min(W,x2+pad_x); py2=min(H,y2+pad_y)
            if px2-px1<4 or py2-py1<4: continue
            patch = img.crop((px1,py1,px2,py2))
            new_cls = class_id_map[cls] if class_id_map else cls
            patches.append((patch, new_cls))
    return patches


def augment_patch(patch, rotation_range=18, scale_range=(0.75,1.25)):
    patch = patch.copy()
    if random.random()<0.4:
        patch = patch.transpose(Image.FLIP_LEFT_RIGHT)
    angle = random.uniform(-rotation_range, rotation_range)
    patch = patch.rotate(angle, expand=True, resample=Image.BICUBIC)
    scale = random.uniform(*scale_range)
    nw = max(MIN_SYM_PX, min(MAX_SYM_PX, int(patch.width*scale)))
    nh = max(MIN_SYM_PX, min(MAX_SYM_PX, int(patch.height*scale)))
    patch = patch.resize((nw,nh), Image.LANCZOS)
    rgb = patch.convert("RGB")
    rgb = ImageEnhance.Brightness(rgb).enhance(random.uniform(0.72,1.28))
    rgb = ImageEnhance.Contrast(rgb).enhance(random.uniform(0.80,1.20))
    alpha = patch.split()[-1]
    out = rgb.convert("RGBA"); out.putalpha(alpha)
    return out


def compose_image(background, by_class, classes, max_sym):
    """Pega símbolos balanceados por clase sobre background."""
    canvas = background.copy().convert("RGB")
    W, H = canvas.size
    placed = []; annotations = []
    n_sym = random.randint(1, max_sym)

    for _ in range(n_sym*5):
        if len(annotations)>=n_sym: break
        cls_id = random.choice(classes)
        patch_orig, _ = random.choice(by_class[cls_id])
        patch = augment_patch(patch_orig)
        pw, ph = patch.size
        if pw>W-10 or ph>H-10: continue
        for _ in range(25):
            x=random.randint(0,W-pw); y=random.randint(0,H-ph)
            box=[x,y,x+pw,y+ph]
            if any(iou(box,b)>0.15 for b in placed): continue
            canvas.paste(patch.convert("RGB"),(x,y),mask=patch.split()[-1])
            placed.append(box)
            cx,cy,bw,bh = px_to_yolo(x,y,x+pw,y+ph,W,H)
            annotations.append((cls_id,
                max(0.001,min(0.999,cx)), max(0.001,min(0.999,cy)),
                max(0.001,min(0.999,bw)), max(0.001,min(0.999,bh))))
            break

    return canvas, annotations


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets",       nargs="+", required=True)
    ap.add_argument("--out",            required=True)
    ap.add_argument("--n-images",       type=int, default=500)
    ap.add_argument("--max-sym-per-img",type=int, default=3)
    ap.add_argument("--val-split",      type=float, default=0.15)
    ap.add_argument("--seed",           type=int, default=42)
    ap.add_argument("--max-tiles-per-class", type=int, default=60,
                    help="Max tiles a abrir por clase para extraer patches (default 60, acelera carga)")
    ap.add_argument("--idx-start",      type=int, default=0,
                    help="Indice inicial para nombres de archivo (para combinar lotes)")
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed)

    all_class_names = []; dataset_infos = []
    for ds_path in args.datasets:
        data_yaml = Path(ds_path)/"data.yaml"
        names = []
        if data_yaml.exists():
            for line in data_yaml.read_text().splitlines():
                if "names" in line:
                    names = re.findall(r'"([^"]+)"', line) or re.findall(r"'([^']+)'", line)
        if not names: names=[Path(ds_path).name]
        l2g = {}
        for i,n in enumerate(names):
            if n not in all_class_names: all_class_names.append(n)
            l2g[i] = all_class_names.index(n)
        records = load_dataset(ds_path)
        # Limitar tiles para acelerar extracción de patches
        if len(records) > args.max_tiles_per_class:
            records_extract = random.sample(records, args.max_tiles_per_class)
        else:
            records_extract = records
        dataset_infos.append((ds_path, names, l2g, records_extract))
        print(f"  {Path(ds_path).name}: {len(records)} tiles  (usando {len(records_extract)})  →  {names}")

    print(f"\nClases ({len(all_class_names)}): {all_class_names}")

    # Extraer patches con caché en disco
    cache_key = hashlib.md5(str(sorted(args.datasets)).encode()).hexdigest()[:10]
    cache_path = Path(args.out) / f".patch_cache_{cache_key}.pkl"
    Path(args.out).mkdir(parents=True, exist_ok=True)

    if cache_path.exists():
        print(f"  [cache] Cargando patches desde {cache_path.name} ...")
        with open(cache_path, "rb") as f:
            by_class = pickle.load(f)
        classes = list(by_class.keys())
        total = sum(len(v) for v in by_class.values())
        for cls, patches in by_class.items():
            print(f"  [cache] clase {all_class_names[cls]}: {len(patches)} patches")
        print(f"  [cache] Total: {total} patches")
    else:
        all_patches = []
        for ds_path, names, l2g, records in dataset_infos:
            p = extract_patches(records, class_id_map=l2g)
            all_patches.extend(p)
            c = Counter(all_class_names[cls] for _,cls in p)
            print(f"  Patches {Path(ds_path).name}: {len(p)}  {dict(c)}")
        by_class = {}
        for patch, cls in all_patches:
            by_class.setdefault(cls, []).append((patch, cls))
        with open(cache_path, "wb") as f:
            pickle.dump(by_class, f)
        print(f"  [cache] Guardado en {cache_path.name}")
    classes = list(by_class.keys())

    # Fondos: solo paths, cargar on-demand
    bg_paths = []
    for img_path, _ in sum((r for _,_,_,r in dataset_infos), []):
        bg_paths.append(img_path)
    print(f"\nFondos disponibles: {len(bg_paths)}")

    # Crear carpetas
    out = Path(args.out)
    for split in ("train","val"):
        (out/"images"/split).mkdir(parents=True, exist_ok=True)
        (out/"labels"/split).mkdir(parents=True, exist_ok=True)

    n_val   = max(1, int(args.n_images*args.val_split))
    n_train = args.n_images - n_val
    print(f"Generando {n_train} train + {n_val} val ...\n")

    idx = args.idx_start
    for split, n_target in [("train",n_train),("val",n_val)]:
        generated = 0
        attempts  = 0
        while generated < n_target and attempts < n_target*4:
            attempts += 1
            bg_img = Image.open(random.choice(bg_paths)).convert("RGB")
            if bg_img.size != (TILE_SIZE, TILE_SIZE):
                bg_img = bg_img.resize((TILE_SIZE, TILE_SIZE))
            canvas, annots = compose_image(bg_img, by_class, classes, args.max_sym_per_img)
            if not annots: continue
            fname = f"synth_{idx:05d}"
            idx += 1; generated += 1
            canvas.save(out/"images"/split/f"{fname}.png")
            with open(out/"labels"/split/f"{fname}.txt","w") as f:
                for cls_id,cx,cy,bw,bh in annots:
                    f.write(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
        print(f"  {split}: {generated} imágenes")

    # data.yaml
    (out/"data.yaml").write_text(
        f"path: {out.resolve()}\ntrain: images/train\nval:   images/val\n\n"
        f"nc: {len(all_class_names)}\nnames: {json.dumps(all_class_names)}\n")

    print(f"\n✓ Listo en {out}/  ({idx - args.idx_start} imágenes total)")


if __name__=="__main__":
    main()
