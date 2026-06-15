# phase4_assembler.py — Dataset assembly: 80/10/10 split + dynamic data.yaml
# Phase 3 requirements: exact 80/10/10 random shuffle across all components,
# generate_yolo_yaml() with sequential class IDs and absolute paths,
# class remapping in .txt labels when needed.
from __future__ import annotations

import random
import shutil
from pathlib import Path

import cv2
import numpy as np

from config import (
    BACKGROUNDS_DIR,
    CLASS_NAME,
    DATASET_DIR,
    NEGATIVE_RATIO,
    SYNTHETIC_DIR,
    TRAIN_RATIO,
    load_config,
    PipelineConfig,
    ComponentConfig,
)


def create_yolo_structure(dataset_dir: Path) -> dict[str, Path]:
    """Create the YOLO directory tree with train/val/test splits."""
    dirs: dict[str, Path] = {}
    for split in ("train", "val", "test"):
        for sub in ("images", "labels"):
            d = dataset_dir / sub / split
            d.mkdir(parents=True, exist_ok=True)
            dirs[f"{split}_{sub}"] = d
    print(f"[Fase 4] Estructura YOLO creada en '{dataset_dir}'")
    return dirs


def _collect_all_component_images(cfg: PipelineConfig) -> list[tuple[Path, Path]]:
    """Walk every component's synthetic output and collect (img, lbl) pairs.

    Supports multi-variant components: each variant has its own
    ``synthetic_v{i}/`` directory, all sharing the same class_id.
    """
    pairs: list[tuple[Path, Path]] = []
    for comp in cfg.components:
        for vi in range(cfg.component_variant_count(comp)):
            syn_dir = cfg.component_synthetic_dir(comp, vi)
            imgs_dir = syn_dir / "images"
            lbls_dir = syn_dir / "labels"
            if not imgs_dir.exists():
                continue
            for img_path in sorted(imgs_dir.glob("*.jpg")):
                lbl_path = lbls_dir / (img_path.stem + ".txt")
                pairs.append((img_path, lbl_path))
    return pairs


def _remap_class_ids(label_path: Path, class_map: dict[int, int]) -> str:
    """Read a YOLO label file and remap class IDs according to *class_map*.

    Returns the (possibly modified) label content as a string.
    If the file doesn't exist, returns an empty string.
    """
    if not label_path.exists():
        return ""
    lines: list[str] = []
    for raw_line in label_path.read_text(encoding="utf-8").splitlines():
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        parts = raw_line.split()
        if len(parts) < 5:
            continue
        old_id = int(parts[0])
        new_id = class_map.get(old_id, old_id)
        parts[0] = str(new_id)
        lines.append(" ".join(parts))
    return "\n".join(lines) + ("\n" if lines else "")


def split_and_copy(*args, **kwargs): pass


def _augment_background(img: np.ndarray, rng: random.Random) -> np.ndarray:
    """Apply random augmentations to a background image to create a unique variant.

    Augmentations applied (all lightweight, preserving the 'empty plan' look):
      - Random crop (70-100% of the image) + resize back to original size
      - Random horizontal / vertical flip
      - Random brightness & contrast jitter
      - Optional random text injection to simulate CAD notes
    """
    h, w = img.shape[:2]

    # 1. Random crop (70–100 % area) then resize back
    crop_frac = rng.uniform(0.70, 1.0)
    ch, cw = int(h * crop_frac), int(w * crop_frac)
    y0 = rng.randint(0, max(h - ch, 0))
    x0 = rng.randint(0, max(w - cw, 0))
    cropped = img[y0 : y0 + ch, x0 : x0 + cw]
    if cropped.shape[0] != h or cropped.shape[1] != w:
        cropped = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)

    # 2. Random flips
    if rng.random() < 0.5:
        cropped = cv2.flip(cropped, 1)   # horizontal
    if rng.random() < 0.3:
        cropped = cv2.flip(cropped, 0)   # vertical

    # 3. Brightness / contrast jitter
    alpha = rng.uniform(0.8, 1.2)   # contrast
    beta = rng.randint(-25, 25)     # brightness
    cropped = cv2.convertScaleAbs(cropped, alpha=alpha, beta=beta)
    
    # 4. Texto aleatorio (basura) para confundir
    if rng.random() < 0.5:  # 50% de las veces agrega texto
        textos = ["TUG", "IUE", "C1", "2x16A", "RESERVA", "A", "B", "10 mm2", "C2", "2x10A", "L1", "L2", "L3", "PDT.NOR", "TABLERO"]
        n_textos = rng.randint(1, 4)
        for _ in range(n_textos):
            texto = rng.choice(textos)
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = rng.uniform(0.4, 0.8)
            thick = rng.randint(1, 2)
            px = rng.randint(10, max(11, w - 80))
            py = rng.randint(20, max(21, h - 20))
            cv2.putText(cropped, texto, (px, py), font, scale, (0,0,0), thick)

    return cropped


def inject_negatives(component_name: str, cfg: PipelineConfig, dirs: dict[str, Path], n_positives: int):
    """Injects empty backgrounds and confusing components as negative samples."""
    total_negatives = int(n_positives * cfg.g.negative_ratio)
    if total_negatives <= 0:
        return

    # 5% empty (1/3 of the 15%), 10% confusing (2/3 of the 15%)
    n_empty = int(total_negatives * 0.333)
    n_confusing = total_negatives - n_empty

    rng = random.Random(42)
    
    # 1. Pure empty backgrounds
    bg_paths = list(cfg.g.backgrounds_dir.glob("*.jpg")) + list(cfg.g.backgrounds_dir.glob("*.png"))
    if bg_paths and n_empty > 0:
        print(f"[Fase 4] Inyectando {n_empty} fondos vacíos con texto como negativos...")
        for i in range(n_empty):
            bg_path = rng.choice(bg_paths)
            img = cv2.imread(str(bg_path))
            if img is not None:
                img = _augment_background(img, rng)
                stem = f"neg_bg_{i:04d}"
                cv2.imwrite(str(dirs["train_images"] / f"{stem}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                (dirs["train_labels"] / f"{stem}.txt").write_text("", encoding="utf-8")

    # 2. Confusing components
    confusing_pairs = []
    for comp in cfg.components:
        if comp.name == component_name:
            continue
        for vi in range(cfg.component_variant_count(comp)):
            syn_dir = cfg.component_synthetic_dir(comp, vi)
            imgs_dir = syn_dir / "images"
            if imgs_dir.exists():
                confusing_pairs.extend(list(imgs_dir.glob("*.jpg")))
                
    if confusing_pairs and n_confusing > 0:
        print(f"[Fase 4] Inyectando {min(n_confusing, len(confusing_pairs))} imágenes confusas (otros símbolos) como negativos...")
        sampled = rng.sample(confusing_pairs, min(n_confusing, len(confusing_pairs)))
        for i, img_path in enumerate(sampled):
            stem = f"neg_confusing_{img_path.parent.parent.name}_{i:04d}"
            shutil.copy2(img_path, dirs["train_images"] / f"{stem}.jpg")
            (dirs["train_labels"] / f"{stem}.txt").write_text("", encoding="utf-8")

    # 3. Manual specific negatives (Hard negatives from user)
    # Checks if there is a 'negatives_<component_name>' folder in train-maker.
    from config import BASE_DIR
    manual_neg_dir = BASE_DIR / f"negatives_{component_name}"
    if manual_neg_dir.exists():
        manual_neg_paths = list(manual_neg_dir.glob("*.png")) + list(manual_neg_dir.glob("*.jpg"))
        if manual_neg_paths:
            print(f"[Fase 4] Inyectando {len(manual_neg_paths)} negativos MANUALES (hard negatives) para {component_name}...")
            # We copy them multiple times with augmentations to give them weight
            copies_per_neg = 10 
            for img_path in manual_neg_paths:
                img = cv2.imread(str(img_path))
                if img is not None:
                    for c in range(copies_per_neg):
                        aug_img = _augment_background(img, rng)
                        stem = f"neg_manual_{img_path.stem}_{c:02d}"
                        cv2.imwrite(str(dirs["train_images"] / f"{stem}.jpg"), aug_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                        (dirs["train_labels"] / f"{stem}.txt").write_text("", encoding="utf-8")


def generate_yolo_yaml(*args, **kwargs): pass


def create_data_yaml(*args, **kwargs): pass


def print_dataset_summary(dataset_dir: Path) -> None:
    print("\n" + "=" * 60)
    print("  RESUMEN DEL DATASET")
    print("=" * 60)
    total_imgs = 0
    for split in ("train", "val", "test"):
        imgs_dir = dataset_dir / "images" / split
        labels_dir = dataset_dir / "labels" / split
        if not imgs_dir.exists():
            continue
        n_imgs = len(list(imgs_dir.glob("*.*")))
        n_neg = sum(1 for f in labels_dir.glob("*.txt") if f.stat().st_size == 0)
        n_pos = n_imgs - n_neg
        total_imgs += n_imgs
        pct_neg = (n_neg / n_imgs * 100) if n_imgs else 0
        print(
            f"  {split.upper():5s} │ {n_imgs:5d} imgs │ "
            f"{n_pos:5d} pos │ {n_neg:4d} neg ({pct_neg:.0f}%)"
        )
    print(f"  {'TOTAL':5s} │ {total_imgs:5d} imgs")
    print("=" * 60 + "\n")


def assemble_dataset(*args, **kwargs): pass

def assemble_synthetic(component_name: str, cfg: PipelineConfig) -> Path:
    from config import BASE_DIR
    dataset_dir = BASE_DIR / f"dataset_sintetico_{component_name}"
    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)
    
    dirs = {}
    for sub in ("images", "labels"):
        d = dataset_dir / "train" / sub
        d.mkdir(parents=True, exist_ok=True)
        dirs[f"train_{sub}"] = d
        
    pairs = []
    for comp in cfg.components:
        if comp.name == component_name:
            for vi in range(cfg.component_variant_count(comp)):
                syn_dir = cfg.component_synthetic_dir(comp, vi)
                imgs_dir = syn_dir / "images"
                lbls_dir = syn_dir / "labels"
                if not imgs_dir.exists():
                    continue
                for img_path in sorted(imgs_dir.glob("*.jpg")):
                    lbl_path = lbls_dir / (img_path.stem + ".txt")
                    pairs.append((img_path, lbl_path))

    for img_path, lbl_path in pairs:
        shutil.copy2(img_path, dirs["train_images"] / img_path.name)
        dst_lbl = dirs["train_labels"] / (img_path.stem + ".txt")
        # Remap any class to 0
        dst_lbl.write_text(_remap_class_ids(lbl_path, {i: 0 for i in range(100)}), encoding="utf-8")

    inject_negatives(component_name, cfg, dirs, len(pairs))

    yaml_path = dataset_dir / f"sintetico_{component_name}.yaml"
    content = (
        f"# Liard — Dataset Sintetico {component_name}\n\n"
        f"path: {dataset_dir.resolve()}\n"
        f"train: train/images\n"
        f"val:   train/images\n"
        f"nc: 1\n"
        f"names:\n"
        f"  0: {component_name}\n"
    )
    yaml_path.write_text(content, encoding="utf-8")
    print(f"[Fase 4] Dataset sintético ensamblado en '{dataset_dir}'")
    return yaml_path
