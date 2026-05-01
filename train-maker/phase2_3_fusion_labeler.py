from dataclasses import dataclass
from pathlib import Path
import random

import cv2
import numpy as np

from config import (
    ALLOW_RANDOM_ROTATION,
    BACKGROUNDS_DIR,
    CLASS_ID,
    COMPONENTS_PER_IMG_MAX,
    COMPONENTS_PER_IMG_MIN,
    N_SYNTHETIC_TOTAL,
    SPRITE_SCALE_MAX,
    SPRITE_SCALE_MIN,
    SPRITES_DIR,
    SYNTHETIC_DIR,
)


@dataclass
class YoloBBox:
    class_id: int
    cx: float
    cy: float
    w: float
    h: float

    def to_line(self) -> str:
        return f"{self.class_id} {self.cx:.6f} {self.cy:.6f} {self.w:.6f} {self.h:.6f}"


def load_sprites(sprites_dir: Path) -> list[np.ndarray]:
    sprites: list[np.ndarray] = []
    for png in sorted(sprites_dir.glob("*.png")):
        img = cv2.imread(str(png), cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
        elif img.shape[2] == 3:
            alpha = np.full((*img.shape[:2], 1), 255, dtype=np.uint8)
            img = np.concatenate([img, alpha], axis=2)
        sprites.append(img)

    if not sprites:
        raise FileNotFoundError(f"No se encontraron sprites PNG en: {sprites_dir}")
    return sprites


def load_background_paths(bg_dir: Path) -> list[Path]:
    paths: list[Path] = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff"):
        paths.extend(bg_dir.glob(ext))
    if not paths:
        raise FileNotFoundError(f"No se encontraron imágenes en: {bg_dir}")
    return sorted(paths)


def rotate_sprite_bound(image: np.ndarray, angle: float) -> np.ndarray:
    """Rota la imagen asegurando que las esquinas no se recorten."""
    h, w = image.shape[:2]
    cX, cY = w // 2, h // 2

    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    return cv2.warpAffine(
        image,
        M,
        (nW, nH),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )


def scale_sprite(sprite: np.ndarray, bg_w: int) -> np.ndarray:
    scale = random.uniform(SPRITE_SCALE_MIN, SPRITE_SCALE_MAX)
    target_w = max(10, int(bg_w * scale))
    h, w = sprite.shape[:2]
    target_h = max(10, int(target_w * h / w))
    return cv2.resize(sprite, (target_w, target_h), interpolation=cv2.INTER_LINEAR)


def composite_sprite_on_bg(bg: np.ndarray, sprite: np.ndarray, x: int, y: int) -> np.ndarray:
    result = bg.copy().astype(np.float32)
    h_bg, w_bg = bg.shape[:2]
    h_s, w_s = sprite.shape[:2]

    x1_bg, y1_bg = max(0, x), max(0, y)
    x2_bg, y2_bg = min(w_bg, x + w_s), min(h_bg, y + h_s)

    if x2_bg <= x1_bg or y2_bg <= y1_bg:
        return bg

    x1_sp = x1_bg - x
    y1_sp = y1_bg - y
    x2_sp = x1_sp + (x2_bg - x1_bg)
    y2_sp = y1_sp + (y2_bg - y1_bg)

    roi = sprite[y1_sp:y2_sp, x1_sp:x2_sp]
    alpha = roi[:, :, 3:4].astype(np.float32) / 255.0
    sprite_bgr = roi[:, :, :3].astype(np.float32)

    bg_roi = result[y1_bg:y2_bg, x1_bg:x2_bg]
    result[y1_bg:y2_bg, x1_bg:x2_bg] = sprite_bgr * alpha + bg_roi * (1.0 - alpha)

    return result.astype(np.uint8)


def calculate_yolo_bbox(
    x: int,
    y: int,
    sprite_w: int,
    sprite_h: int,
    bg_w: int,
    bg_h: int,
    class_id: int = CLASS_ID,
) -> YoloBBox:
    cx_px = x + sprite_w / 2.0
    cy_px = y + sprite_h / 2.0

    cx = np.clip(cx_px / bg_w, 0.0, 1.0)
    cy = np.clip(cy_px / bg_h, 0.0, 1.0)
    w = np.clip(sprite_w / bg_w, 0.0, 1.0)
    h = np.clip(sprite_h / bg_h, 0.0, 1.0)

    return YoloBBox(class_id=class_id, cx=cx, cy=cy, w=w, h=h)


def generate_one_sample(
    sprites: list[np.ndarray],
    bg_paths: list[Path],
    out_img_path: Path,
    out_label_path: Path,
) -> None:
    bg = cv2.imread(str(random.choice(bg_paths)))
    if bg is None:
        return
    h_bg, w_bg = bg.shape[:2]

    n = random.randint(COMPONENTS_PER_IMG_MIN, COMPONENTS_PER_IMG_MAX)
    bboxes: list[YoloBBox] = []

    for _ in range(n):
        sprite_raw = random.choice(sprites)

        if ALLOW_RANDOM_ROTATION:
            angle = random.choice([0, 90, 180, 270])
            if angle != 0:
                sprite_raw = rotate_sprite_bound(sprite_raw, angle)

        sprite = scale_sprite(sprite_raw, w_bg)
        h_s, w_s = sprite.shape[:2]

        if w_s >= w_bg or h_s >= h_bg:
            continue

        x = random.randint(0, max(0, w_bg - w_s))
        y = random.randint(0, max(0, h_bg - h_s))

        bg = composite_sprite_on_bg(bg, sprite, x, y)
        bbox = calculate_yolo_bbox(x, y, w_s, h_s, w_bg, h_bg)
        bboxes.append(bbox)

    out_img_path.parent.mkdir(parents=True, exist_ok=True)
    out_label_path.parent.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(out_img_path), bg, [cv2.IMWRITE_JPEG_QUALITY, 95])
    with open(out_label_path, "w", encoding="utf-8") as f:
        for bb in bboxes:
            f.write(bb.to_line() + "\n")


def generate_synthetic_dataset(
    sprites_dir: Path = SPRITES_DIR,
    bg_dir: Path = BACKGROUNDS_DIR,
    output_dir: Path = SYNTHETIC_DIR,
    n_total: int = N_SYNTHETIC_TOTAL,
) -> tuple[list[Path], list[Path]]:
    sprites = load_sprites(sprites_dir)
    bg_paths = load_background_paths(bg_dir)
    print(f"[Fase 2/3] {len(sprites)} sprites | {len(bg_paths)} fondos | {n_total} muestras")

    img_paths: list[Path] = []
    lbl_paths: list[Path] = []
    imgs_dir = output_dir / "images"
    lbls_dir = output_dir / "labels"

    for i in range(n_total):
        stem = f"synth_{i:05d}"
        i_path = imgs_dir / f"{stem}.jpg"
        l_path = lbls_dir / f"{stem}.txt"

        generate_one_sample(sprites, bg_paths, i_path, l_path)
        img_paths.append(i_path)
        lbl_paths.append(l_path)

        if (i + 1) % 100 == 0 or (i + 1) == n_total:
            print(f"  Generadas {i + 1:>5}/{n_total} muestras...")

    print(f"[Fase 2/3] ✅ Dataset sintético guardado en '{output_dir}'\n")
    return img_paths, lbl_paths

