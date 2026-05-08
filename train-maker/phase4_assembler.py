# phase4_assembler.py — Dataset assembly: 80/10/10 split + dynamic data.yaml
# Phase 3 requirements: exact 80/10/10 random shuffle across all components,
# generate_yolo_yaml() with sequential class IDs and absolute paths,
# class remapping in .txt labels when needed.
from __future__ import annotations

import random
import shutil
from pathlib import Path

import cv2

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


def split_and_copy(
    pairs: list[tuple[Path, Path]],
    dirs: dict[str, Path],
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    class_map: dict[int, int] | None = None,
    seed: int = 42,
) -> None:
    """80 / 10 / 10 random split of (image, label) pairs into YOLO dirs.

    *class_map* is applied during copy so labels land with the correct
    sequential class IDs.
    """
    random.seed(seed)
    shuffled = list(pairs)
    random.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    # test gets the remainder → exact 80/10/10 (±1 due to rounding)

    splits = {
        "train": shuffled[:n_train],
        "val": shuffled[n_train : n_train + n_val],
        "test": shuffled[n_train + n_val :],
    }

    for split, pair_list in splits.items():
        for img_path, lbl_path in pair_list:
            # Copy image
            shutil.copy2(img_path, dirs[f"{split}_images"] / img_path.name)
            # Copy / remap label
            dst_lbl = dirs[f"{split}_labels"] / (img_path.stem + ".txt")
            if class_map:
                dst_lbl.write_text(_remap_class_ids(lbl_path, class_map), encoding="utf-8")
            elif lbl_path.exists():
                shutil.copy2(lbl_path, dst_lbl)
            else:
                dst_lbl.touch()

        print(f"[Fase 4] {split:5s}: {len(pair_list)} muestras copiadas")


def inject_negatives(
    dirs: dict[str, Path],
    bg_dir: Path,
    negative_ratio: float = NEGATIVE_RATIO,
    seed: int = 99,
) -> None:
    random.seed(seed)
    bg_paths: list[Path] = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.tif"):
        bg_paths.extend(bg_dir.glob(ext))
    if not bg_paths:
        print("[Fase 4] ⚠️  Sin fondos para inyectar negativos.")
        return

    if negative_ratio <= 0:
        for split in ("train", "val", "test"):
            print(f"[Fase 4] {split:5s}: 0 negativos inyectados")
        return

    for split in ("train", "val", "test"):
        out_imgs = dirs[f"{split}_images"]
        out_labels = dirs[f"{split}_labels"]

        n_pos = len(list(out_imgs.glob("*.*")))
        if negative_ratio >= 1.0:
            n_neg = len(bg_paths)
        else:
            n_neg = min(int(n_pos * negative_ratio / (1.0 - negative_ratio)), len(bg_paths))

        sampled = random.sample(bg_paths, min(n_neg, len(bg_paths)))
        for i, bg_path in enumerate(sampled):
            stem = f"negative_{split}_{i:04d}"
            img = cv2.imread(str(bg_path))
            if img is None:
                continue
            cv2.imwrite(str(out_imgs / f"{stem}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            (out_labels / f"{stem}.txt").touch()

        print(f"[Fase 4] {split:5s}: {len(sampled)} negativos inyectados")


def generate_yolo_yaml(dataset_dir: Path, cfg: PipelineConfig) -> Path:
    """Write ``data.yaml`` with absolute paths and dynamic class mapping.

    Class IDs are sequential (0, 1, 2, …) matching the order in
    ``components_config.yaml``.
    """
    yaml_path = dataset_dir / "data.yaml"

    names_block = "\n".join(
        f"  {comp.class_id}: {comp.name}" for comp in cfg.components
    )

    content = (
        f"# Liard — Dataset Config (auto-generado)\n\n"
        f"path: {dataset_dir.resolve()}\n"
        f"train: images/train\n"
        f"val:   images/val\n"
        f"test:  images/test\n\n"
        f"nc: {len(cfg.components)}\n"
        f"names:\n{names_block}\n"
    )
    yaml_path.write_text(content, encoding="utf-8")
    print(f"[Fase 4] data.yaml generado en '{yaml_path}'")
    return yaml_path


def create_data_yaml(dataset_dir: Path, class_name: str = CLASS_NAME) -> Path:
    """Legacy single-class wrapper — kept for backward compat."""
    yaml_path = dataset_dir / "data.yaml"
    content = (
        f"# Liard — Dataset Config (auto-generado)\n\n"
        f"path: {dataset_dir.resolve()}\n"
        f"train: images/train\n"
        f"val:   images/val\n"
        f"test:  images/test\n\n"
        f"nc: 1\n"
        f"names:\n"
        f"  0: {class_name}\n"
    )
    yaml_path.write_text(content, encoding="utf-8")
    print(f"[Fase 4] data.yaml generado en '{yaml_path}'")
    return yaml_path


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


def assemble_dataset(
    synthetic_dir: Path = SYNTHETIC_DIR,
    bg_dir: Path = BACKGROUNDS_DIR,
    dataset_dir: Path = DATASET_DIR,
    train_ratio: float = TRAIN_RATIO,
    negative_ratio: float = NEGATIVE_RATIO,
) -> Path:
    """Legacy single-component assembler — delegates to the multi-component
    flow when called from the new pipeline, or falls back to old behavior."""

    # Clean dataset dir for idempotency
    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)

    dirs = create_yolo_structure(dataset_dir)

    # Try multi-component path
    try:
        cfg = load_config()
        pairs = _collect_all_component_images(cfg)
        if pairs:
            split_and_copy(
                pairs, dirs,
                train_ratio=cfg.g.train_ratio,
                val_ratio=cfg.g.val_ratio,
            )
            inject_negatives(dirs, bg_dir, negative_ratio)
            yaml_path = generate_yolo_yaml(dataset_dir, cfg)
            print_dataset_summary(dataset_dir)
            return yaml_path
    except Exception:
        pass

    # Fallback: single synthetic_dir (old behavior)
    imgs_src = synthetic_dir / "images"
    lbls_src = synthetic_dir / "labels"
    all_imgs = sorted(imgs_src.glob("*.jpg")) + sorted(imgs_src.glob("*.png"))
    pairs_legacy = [(img, lbls_src / (img.stem + ".txt")) for img in all_imgs]

    split_and_copy(pairs_legacy, dirs, train_ratio)
    inject_negatives(dirs, bg_dir, negative_ratio)
    yaml_path = create_data_yaml(dataset_dir)
    print_dataset_summary(dataset_dir)
    return yaml_path
