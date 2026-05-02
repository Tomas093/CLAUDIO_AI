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

# Carpeta donde vas a poner los PNGs de los polos (//, ///, etc.)
MODIFIERS_DIR = Path("dataset_generator/modifiers")


@dataclass
class YoloBBox:
    class_id: int
    cx: float
    cy: float
    w: float
    h: float

    def to_line(self) -> str:
        return f"{self.class_id} {self.cx:.6f} {self.cy:.6f} {self.w:.6f} {self.h:.6f}"


def load_rgba_images(folder_dir: Path) -> list[np.ndarray]:
    """Carga imágenes (bases o modificadores) asegurando formato RGBA."""
    images: list[np.ndarray] = []
    if not folder_dir.exists():
        return images

    for png in sorted(folder_dir.glob("*.png")):
        img = cv2.imread(str(png), cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
        elif img.shape[2] == 3:
            alpha = np.full((*img.shape[:2], 1), 255, dtype=np.uint8)
            img = np.concatenate([img, alpha], axis=2)
        images.append(img)
    return images


def load_background_paths(bg_dir: Path) -> list[Path]:
    paths: list[Path] = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff"):
        paths.extend(bg_dir.glob(ext))
    if not paths:
        raise FileNotFoundError(f"No se encontraron imágenes en: {bg_dir}")
    return sorted(paths)


def build_modular_sprite(base_img: np.ndarray, mod_img: np.ndarray) -> tuple:
    """
    Ensambla la base y el modificador (polos) en un solo lienzo transparente.
    Retorna el lienzo combinado y las coordenadas (x, y, w, h) donde quedó la base.
    """
    h_b, w_b = base_img.shape[:2]

    if mod_img is not None:
        h_m, w_m = mod_img.shape[:2]
    else:
        h_m, w_m = 0, 0

    canvas_w = max(w_b, w_m)
    canvas_h = h_b + h_m

    canvas = np.zeros((canvas_h, canvas_w, 4), dtype=np.uint8)

    # 1. Pegar modificador arriba (centrado)
    if mod_img is not None:
        x_m = (canvas_w - w_m) // 2
        canvas[0:h_m, x_m:x_m + w_m] = mod_img

    # 2. Pegar base abajo (centrada)
    x_b = (canvas_w - w_b) // 2
    y_b = h_m
    canvas[y_b:y_b + h_b, x_b:x_b + w_b] = base_img

    # El rastreador: coordenadas exactas de la base dentro del canvas
    base_roi = (x_b, y_b, w_b, h_b)

    return canvas, base_roi


def rotate_sprite_and_track_bbox(image: np.ndarray, angle: float, base_roi: tuple):
    """Rota el lienzo y aplica álgebra lineal para saber dónde quedó la base."""
    h, w = image.shape[:2]
    cX, cY = w // 2, h // 2

    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    rotated_img = cv2.warpAffine(
        image, M, (nW, nH),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0)
    )

    # Transformar las 4 esquinas del ROI de la base
    bx, by, bw, bh = base_roi
    corners = np.array([
        [bx, by, 1],
        [bx + bw, by, 1],
        [bx + bw, by + bh, 1],
        [bx, by + bh, 1]
    ])

    rotated_corners = M.dot(corners.T).T

    min_x = int(np.min(rotated_corners[:, 0]))
    max_x = int(np.max(rotated_corners[:, 0]))
    min_y = int(np.min(rotated_corners[:, 1]))
    max_y = int(np.max(rotated_corners[:, 1]))

    new_base_roi = (min_x, min_y, max_x - min_x, max_y - min_y)

    return rotated_img, new_base_roi


def scale_canvas_and_roi(canvas: np.ndarray, base_roi: tuple, scale: float):
    """Escala la imagen y las coordenadas del rastreador al mismo tiempo."""
    h, w = canvas.shape[:2]
    new_w, new_h = max(10, int(w * scale)), max(10, int(h * scale))
    scaled_img = cv2.resize(canvas, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    bx, by, bw, bh = base_roi
    scaled_roi = (int(bx * scale), int(by * scale), int(bw * scale), int(bh * scale))

    return scaled_img, scaled_roi


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
        x: int, y: int, sprite_w: int, sprite_h: int, bg_w: int, bg_h: int, class_id: int = CLASS_ID
) -> YoloBBox:
    cx_px = x + sprite_w / 2.0
    cy_px = y + sprite_h / 2.0

    cx = np.clip(cx_px / bg_w, 0.0, 1.0)
    cy = np.clip(cy_px / bg_h, 0.0, 1.0)
    w = np.clip(sprite_w / bg_w, 0.0, 1.0)
    h = np.clip(sprite_h / bg_h, 0.0, 1.0)

    return YoloBBox(class_id=class_id, cx=cx, cy=cy, w=w, h=h)


def generate_one_sample(
        sprites_base: list[np.ndarray],
        sprites_modifiers: list[np.ndarray],
        bg_paths: list[Path],
        out_img_path: Path,
        out_label_path: Path,
) -> None:
    bg = cv2.imread(str(random.choice(bg_paths)))
    if bg is None: return

    # DATA AUGMENTATION DEL FONDO
    if random.random() > 0.5:
        flip_code = random.choice([-1, 0, 1])
        bg = cv2.flip(bg, flip_code)

    rot_code = random.choice([None, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE])
    if rot_code is not None:
        bg = cv2.rotate(bg, rot_code)

    brillo_promedio = np.mean(bg)
    if brillo_promedio < 127:
        bg = cv2.bitwise_not(bg)

    h_bg, w_bg = bg.shape[:2]
    n = random.randint(COMPONENTS_PER_IMG_MIN, COMPONENTS_PER_IMG_MAX)
    bboxes: list[YoloBBox] = []

    for _ in range(n):
        # 1. Seleccionar base y modificador (70% de las veces usamos un modificador si existe)
        base_raw = random.choice(sprites_base)
        mod_raw = random.choice(sprites_modifiers) if sprites_modifiers and random.random() > 0.3 else None

        # --- MAGIA DE GROSOR (DILATACIÓN ALEATORIA) ---
        if mod_raw is not None:
            # Elegimos un grosor extra al azar: 1 (nada), 2 (poco) o 3 (bastante)
            grosor_extra = random.choice([1, 2, 3])
            if grosor_extra > 1:
                # Hacemos una copia para no alterar el PNG original cargado en RAM
                mod_raw = mod_raw.copy()
                # Creamos el "pincel" (kernel) matemático que va a engordar los píxeles
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (grosor_extra, grosor_extra))
                # Engordamos SÓLO el canal Alpha (la transparencia) para mantener el color original
                mod_raw[:, :, 3] = cv2.dilate(mod_raw[:, :, 3], kernel)
        # ----------------------------------------------

        # 2. Ensamblar lienzo modular
        canvas, base_roi = build_modular_sprite(base_raw, mod_raw)

        # 3. ROTACIÓN
        if ALLOW_RANDOM_ROTATION:
            angle = random.choice([0, 90, 180, 270])
            if angle != 0:
                canvas, base_roi = rotate_sprite_and_track_bbox(canvas, angle, base_roi)

        # 4. ESCALADO
        scale = random.uniform(SPRITE_SCALE_MIN, SPRITE_SCALE_MAX)
        canvas, base_roi = scale_canvas_and_roi(canvas, base_roi, scale)

        c_h, c_w = canvas.shape[:2]
        if c_w >= w_bg or c_h >= h_bg:
            continue

        # 5. Posicionar el lienzo ensamblado en el fondo
        px = random.randint(0, max(0, w_bg - c_w))
        py = random.randint(0, max(0, h_bg - c_h))

        bg = composite_sprite_on_bg(bg, canvas, px, py)

        # 6. Extraer coordenadas FINALES (solo de la base, ignorando el modificador)
        bx, by, bw, bh = base_roi
        final_x = px + bx
        final_y = py + by

        bbox = calculate_yolo_bbox(final_x, final_y, bw, bh, w_bg, h_bg)
        bboxes.append(bbox)

        # 7. Agregar texto basura cerca del componente (Ruido visual para la IA)
        if random.random() > 0.5:
            textos_cad = ["2x16A", "4x63A", "2x10A", "TUG", "RESERVA", "PDT.NOR", "TS3B-T1"]
            texto = random.choice(textos_cad)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = random.uniform(0.4, 0.6)
            thickness = random.randint(1, 2)
            # Dibuja el texto abajo del componente
            cv2.putText(bg, texto, (px, py + c_h + 20), font, font_scale, (0, 0, 0), thickness)

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
    # Cargamos tanto las bases como los modificadores
    sprites_base = load_rgba_images(sprites_dir)
    sprites_modifiers = load_rgba_images(MODIFIERS_DIR)

    bg_paths = load_background_paths(bg_dir)
    print(
        f"[Fase 2/3] {len(sprites_base)} bases | {len(sprites_modifiers)} modificadores | {len(bg_paths)} fondos | {n_total} muestras")

    img_paths: list[Path] = []
    lbl_paths: list[Path] = []
    imgs_dir = output_dir / "images"
    lbls_dir = output_dir / "labels"

    for i in range(n_total):
        stem = f"synth_{i:05d}"
        i_path = imgs_dir / f"{stem}.jpg"
        l_path = lbls_dir / f"{stem}.txt"

        generate_one_sample(sprites_base, sprites_modifiers, bg_paths, i_path, l_path)
        img_paths.append(i_path)
        lbl_paths.append(l_path)

        if (i + 1) % 100 == 0 or (i + 1) == n_total:
            print(f"  Generadas {i + 1:>5}/{n_total} muestras...")

    print(f"[Fase 2/3] ✅ Dataset sintético guardado en '{output_dir}'\n")
    return img_paths, lbl_paths