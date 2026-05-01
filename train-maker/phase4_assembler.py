import random
import shutil
from pathlib import Path

import cv2

from config import BACKGROUNDS_DIR, CLASS_NAME, DATASET_DIR, NEGATIVE_RATIO, SYNTHETIC_DIR, TRAIN_RATIO


def create_yolo_structure(dataset_dir: Path) -> dict[str, Path]:
    dirs: dict[str, Path] = {}
    for split in ("train", "val"):
        for sub in ("images", "labels"):
            d = dataset_dir / split / sub
            d.mkdir(parents=True, exist_ok=True)
            dirs[f"{split}_{sub}"] = d
    print(f"[Fase 4] Estructura YOLO creada en '{dataset_dir}'")
    return dirs


def split_and_copy(
    synthetic_dir: Path,
    dirs: dict[str, Path],
    train_ratio: float = TRAIN_RATIO,
    seed: int = 42,
) -> None:
    random.seed(seed)
    imgs_src = synthetic_dir / "images"
    lbls_src = synthetic_dir / "labels"

    all_imgs = sorted(imgs_src.glob("*.jpg")) + sorted(imgs_src.glob("*.png"))
    if not all_imgs:
        raise FileNotFoundError(f"No hay imágenes en {imgs_src}")

    random.shuffle(all_imgs)
    n_train = int(len(all_imgs) * train_ratio)
    splits = {"train": all_imgs[:n_train], "val": all_imgs[n_train:]}

    for split, img_list in splits.items():
        for img_path in img_list:
            shutil.copy2(img_path, dirs[f"{split}_images"] / img_path.name)
            lbl_src = lbls_src / (img_path.stem + ".txt")
            lbl_dst = dirs[f"{split}_labels"] / (img_path.stem + ".txt")
            if lbl_src.exists():
                shutil.copy2(lbl_src, lbl_dst)
            else:
                lbl_dst.touch()

        print(f"[Fase 4] {split:5s}: {len(img_list)} muestras positivas copiadas")


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
        for split in ("train", "val"):
            print(f"[Fase 4] {split:5s}: 0 negativos inyectados")
        return

    for split in ("train", "val"):
        out_imgs = dirs[f"{split}_images"]
        out_labels = dirs[f"{split}_labels"]

        n_pos = len(list(out_imgs.glob("*.*")))
        if negative_ratio >= 1.0:
            n_neg = len(bg_paths)
        else:
            n_neg = min(int(n_pos * negative_ratio / (1.0 - negative_ratio)), len(bg_paths))

        sampled = random.sample(bg_paths, n_neg)
        for i, bg_path in enumerate(sampled):
            stem = f"negative_{split}_{i:04d}"
            img = cv2.imread(str(bg_path))
            if img is None:
                continue
            cv2.imwrite(str(out_imgs / f"{stem}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            (out_labels / f"{stem}.txt").touch()

        print(f"[Fase 4] {split:5s}: {n_neg} negativos inyectados")


def create_data_yaml(dataset_dir: Path, class_name: str = CLASS_NAME) -> Path:
    yaml_path = dataset_dir / "data.yaml"
    content = (
        f"# Liard — Dataset Config (auto-generado)\n\n"
        f"path: {dataset_dir.resolve()}\n"
        f"train: train/images\n"
        f"val:   val/images\n\n"
        f"nc: 1\n"
        f"names:\n"
        f"  0: {class_name}\n"
    )
    yaml_path.write_text(content, encoding="utf-8")
    print(f"[Fase 4] data.yaml generado en '{yaml_path}'")
    return yaml_path


def print_dataset_summary(dataset_dir: Path) -> None:
    print("\n" + "=" * 52)
    print("  RESUMEN DEL DATASET")
    print("=" * 52)
    total_imgs = 0
    for split in ("train", "val"):
        imgs_dir = dataset_dir / split / "images"
        labels_dir = dataset_dir / split / "labels"
        n_imgs = len(list(imgs_dir.glob("*.*")))
        n_neg = sum(1 for f in labels_dir.glob("*.txt") if f.stat().st_size == 0)
        n_pos = n_imgs - n_neg
        total_imgs += n_imgs
        pct_neg = (n_neg / n_imgs * 100) if n_imgs else 0
        print(
            f"  {split.upper():5s} │ {n_imgs:4d} imgs │ "
            f"{n_pos:4d} pos │ {n_neg:3d} neg ({pct_neg:.0f}%)"
        )
    print(f"  {'TOTAL':5s} │ {total_imgs:4d} imgs")
    print("=" * 52 + "\n")


def assemble_dataset(
    synthetic_dir: Path = SYNTHETIC_DIR,
    bg_dir: Path = BACKGROUNDS_DIR,
    dataset_dir: Path = DATASET_DIR,
    train_ratio: float = TRAIN_RATIO,
    negative_ratio: float = NEGATIVE_RATIO,
) -> Path:
    dirs = create_yolo_structure(dataset_dir)
    split_and_copy(synthetic_dir, dirs, train_ratio)
    inject_negatives(dirs, bg_dir, negative_ratio)
    yaml_path = create_data_yaml(dataset_dir)
    print_dataset_summary(dataset_dir)
    return yaml_path

