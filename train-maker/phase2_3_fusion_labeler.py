# phase2_3_fusion_labeler.py — Composición sintética + etiquetado YOLO.
#
# Cambios respecto de la versión anterior (auditoría H1, H4, H8, H12, H14):
#   * La caja se calcula del CANAL ALFA después de rotar, escalar, colocar y
#     recortar contra el borde. Antes era `shape` del lienzo, que incluye el
#     padding transparente: ~13 % de área de más en todas las cajas.
#   * La escala del sprite es fracción del LADO MENOR DEL FONDO, no del propio
#     sprite. Antes el tamaño del símbolo no tenía relación con el tamaño de
#     la imagen y quedaba 2-4× más grande que en inferencia.
#   * Se permite solape moderado (IoU) en vez de prohibir todo contacto, y los
#     fallos de colocación se cuentan y se reportan en vez de desaparecer.
#   * Una fracción de las instancias se coloca cortada por el borde, con un
#     mínimo de área visible, para que el modelo vea lo mismo que le llega
#     desde el slicing de SAHI.
#   * `negative_sprites` se filtra por componente. Esta era la causa raíz H1:
#     el propio símbolo objetivo se pegaba sin etiquetar.
from __future__ import annotations

import random
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import cv2
import numpy as np

from phase1_extractor import alpha_bbox


@dataclass
class YoloBBox:
    class_id: int
    cx: float
    cy: float
    w: float
    h: float

    def to_line(self) -> str:
        return f"{self.class_id} {self.cx:.6f} {self.cy:.6f} {self.w:.6f} {self.h:.6f}"


@dataclass
class GenStats:
    """Contadores para que nada falle en silencio (ver H14)."""

    images: int = 0
    targets_requested: int = 0
    targets_placed: int = 0
    placement_failures: int = 0
    dropped_too_small: int = 0
    dropped_too_big: int = 0
    edge_cropped: int = 0
    empty_positive: int = 0
    bg_read_errors: int = 0
    boxes: list = field(default_factory=list)
    # stem de la imagen -> stem del fondo usado. Sirve para hacer el split
    # AGRUPADO POR FONDO y que un mismo tile no caiga en train y en val.
    groups: dict = field(default_factory=dict)

    def report(self, name: str) -> str:
        import statistics
        lines = [
            f"[Fase 2/3] {name}: {self.images} imágenes, "
            f"{self.targets_placed}/{self.targets_requested} instancias colocadas",
        ]
        if self.placement_failures:
            lines.append(
                f"  · {self.placement_failures} colocaciones fallaron por falta de "
                f"lugar libre"
            )
        if self.dropped_too_big:
            lines.append(f"  · {self.dropped_too_big} descartadas por no entrar en el fondo")
        if self.dropped_too_small:
            lines.append(
                f"  · {self.dropped_too_small} descartadas por quedar demasiado "
                f"cortadas contra el borde"
            )
        if self.edge_cropped:
            lines.append(f"  · {self.edge_cropped} colocadas cortadas por el borde (a propósito)")
        if self.empty_positive:
            lines.append(
                f"  · ATENCIÓN: {self.empty_positive} imágenes marcadas como positivas "
                f"quedaron sin ninguna instancia"
            )
        if self.bg_read_errors:
            lines.append(f"  · ATENCIÓN: {self.bg_read_errors} fondos no se pudieron leer")
        if self.boxes:
            ws = [b[0] for b in self.boxes]
            hs = [b[1] for b in self.boxes]
            lines.append(
                f"  · tamaño de caja (fracción de imagen): "
                f"w med={statistics.median(ws):.3f} [{min(ws):.3f}–{max(ws):.3f}], "
                f"h med={statistics.median(hs):.3f} [{min(hs):.3f}–{max(hs):.3f}]"
            )
        return "\n".join(lines)


# ── Carga ─────────────────────────────────────────────────────────────────────

def load_rgba_images(folder_dir: Path) -> list[np.ndarray]:
    """Carga PNGs asegurando 4 canales."""
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
        raise FileNotFoundError(
            f"No hay imágenes de fondo en: {bg_dir}\n"
            f"Corré generate_backgrounds.py o poné planos DXF en input/backgrounds/."
        )
    return sorted(paths)


# ── Transformaciones ──────────────────────────────────────────────────────────

def rotate_sprite(image: np.ndarray, angle_deg: int) -> np.ndarray:
    """Rota el sprite en múltiplos de 90° sin pérdida ni relleno espurio."""
    if angle_deg % 360 == 0:
        return image
    k = (angle_deg // 90) % 4
    return np.ascontiguousarray(np.rot90(image, k))


def scale_sprite_to_fraction(canvas: np.ndarray, bg_h: int, bg_w: int,
                             fraction: float) -> np.ndarray:
    """Escala el sprite para que su lado mayor sea *fraction* del lado menor del fondo.

    Este es el arreglo de H8: antes el factor se aplicaba sobre el propio
    sprite, así que el tamaño final no tenía ninguna relación con el tamaño
    de la imagen ni con la escala de inferencia.
    """
    target_px = max(8.0, fraction * min(bg_h, bg_w))
    cur = max(canvas.shape[:2])
    if cur <= 0:
        return canvas
    scale = target_px / cur
    new_w = max(4, int(round(canvas.shape[1] * scale)))
    new_h = max(4, int(round(canvas.shape[0] * scale)))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    return cv2.resize(canvas, (new_w, new_h), interpolation=interp)


def composite_sprite_on_bg(bg: np.ndarray, sprite: np.ndarray, x: int, y: int) -> np.ndarray:
    """Alpha-blend del sprite sobre el fondo. Devuelve una imagen nueva."""
    h_bg, w_bg = bg.shape[:2]
    h_s, w_s = sprite.shape[:2]

    x1_bg, y1_bg = max(0, x), max(0, y)
    x2_bg, y2_bg = min(w_bg, x + w_s), min(h_bg, y + h_s)
    if x2_bg <= x1_bg or y2_bg <= y1_bg:
        return bg

    x1_sp, y1_sp = x1_bg - x, y1_bg - y
    x2_sp, y2_sp = x1_sp + (x2_bg - x1_bg), y1_sp + (y2_bg - y1_bg)

    result = bg.copy()
    roi = sprite[y1_sp:y2_sp, x1_sp:x2_sp]
    alpha = roi[:, :, 3:4].astype(np.float32) / 255.0
    sprite_bgr = roi[:, :, :3].astype(np.float32)
    bg_roi = result[y1_bg:y2_bg, x1_bg:x2_bg].astype(np.float32)
    result[y1_bg:y2_bg, x1_bg:x2_bg] = (
        sprite_bgr * alpha + bg_roi * (1.0 - alpha)
    ).astype(np.uint8)
    return result


def visible_bbox_px(sprite: np.ndarray, x: int, y: int, bg_h: int, bg_w: int):
    """Caja del contenido REALMENTE visible del sprite dentro de la imagen.

    Recorta primero contra los bordes y recién después mide el alfa, así que
    el resultado es exacto tanto para instancias completas como cortadas.
    Devuelve ``(x1, y1, w, h, visible_fraction)`` o ``None``.
    """
    h_s, w_s = sprite.shape[:2]

    full = alpha_bbox(sprite)
    if full is None:
        return None
    full_area = float(full[2] * full[3])
    if full_area <= 0:
        return None

    x1_bg, y1_bg = max(0, x), max(0, y)
    x2_bg, y2_bg = min(bg_w, x + w_s), min(bg_h, y + h_s)
    if x2_bg <= x1_bg or y2_bg <= y1_bg:
        return None

    sub = sprite[y1_bg - y:y2_bg - y, x1_bg - x:x2_bg - x]
    vis = alpha_bbox(sub)
    if vis is None:
        return None

    vx, vy, vw, vh = vis
    return x1_bg + vx, y1_bg + vy, vw, vh, (vw * vh) / full_area


def bbox_to_yolo(x: int, y: int, w: int, h: int, bg_w: int, bg_h: int,
                 class_id: int) -> YoloBBox | None:
    """Convierte una caja en píxeles ya recortada a formato YOLO normalizado."""
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(bg_w, x + w)
    y2 = min(bg_h, y + h)
    vw, vh = x2 - x1, y2 - y1
    if vw <= 1 or vh <= 1:
        return None
    return YoloBBox(
        class_id=class_id,
        cx=(x1 + vw / 2.0) / bg_w,
        cy=(y1 + vh / 2.0) / bg_h,
        w=vw / bg_w,
        h=vh / bg_h,
    )


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax0, ay0, aw, ah = a
    bx0, by0, bw, bh = b
    ax1, ay1 = ax0 + aw, ay0 + ah
    bx1, by1 = bx0 + bw, by0 + bh
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
    inter = iw * ih
    if inter == 0:
        return 0.0
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _max_iou(rect, existing) -> float:
    return max((_iou(rect, e) for e in existing), default=0.0)


# ── Generación de una muestra ─────────────────────────────────────────────────

def generate_one_sample(
    sprites_base: list[np.ndarray],
    negative_sprites: list[tuple[str, np.ndarray]],
    bucket: str,
    bg_paths: list[Path],
    out_img_path: Path,
    out_label_path: Path,
    rng: random.Random,
    stats: GenStats,
    class_id: int = 0,
    sprite_scale_min: float = 0.06,
    sprite_scale_max: float = 0.16,
    components_min: int = 1,
    components_max: int = 4,
    allow_rotation: bool = True,
    edge_crop_ratio: float = 0.15,
    min_visible_area: float = 0.40,
    max_placement_iou: float = 0.15,
) -> bool:
    bg_path = rng.choice(bg_paths)
    bg = cv2.imread(str(bg_path))
    if bg is None:
        stats.bg_read_errors += 1
        return False

    # Red de seguridad: si el tile viniera en negativo (líneas blancas sobre
    # negro), se invierte para que el sprite negro contraste.
    if float(np.mean(bg)) < 127:
        bg = cv2.bitwise_not(bg)

    h_bg, w_bg = bg.shape[:2]
    bboxes: list[YoloBBox] = []
    placed_rects: list[tuple[int, int, int, int]] = []

    def place(sprite_img: np.ndarray, is_target: bool) -> None:
        nonlocal bg

        canvas = sprite_img
        if allow_rotation:
            canvas = rotate_sprite(canvas, rng.choice([0, 90, 180, 270]))

        fraction = rng.uniform(sprite_scale_min, sprite_scale_max)
        canvas = scale_sprite_to_fraction(canvas, h_bg, w_bg, fraction)

        c_h, c_w = canvas.shape[:2]
        if c_w > w_bg or c_h > h_bg:
            if is_target:
                stats.dropped_too_big += 1
            return

        # Una fracción de las instancias se coloca deliberadamente cortada
        # por el borde, que es lo que produce el slicing de SAHI (H12).
        want_edge = rng.random() < edge_crop_ratio

        for _ in range(60):
            if want_edge:
                side = rng.choice(("l", "r", "t", "b"))
                off = rng.randint(int(c_w * 0.25), int(c_w * 0.75)) if side in "lr" else \
                      rng.randint(int(c_h * 0.25), int(c_h * 0.75))
                if side == "l":
                    px, py = -off, rng.randint(0, max(0, h_bg - c_h))
                elif side == "r":
                    px, py = w_bg - c_w + off, rng.randint(0, max(0, h_bg - c_h))
                elif side == "t":
                    px, py = rng.randint(0, max(0, w_bg - c_w)), -off
                else:
                    px, py = rng.randint(0, max(0, w_bg - c_w)), h_bg - c_h + off
            else:
                px = rng.randint(0, max(0, w_bg - c_w))
                py = rng.randint(0, max(0, h_bg - c_h))

            vis = visible_bbox_px(canvas, px, py, h_bg, w_bg)
            if vis is None:
                continue
            vx, vy, vw, vh, visible_fraction = vis

            if want_edge and visible_fraction < min_visible_area:
                continue
            if _max_iou((vx, vy, vw, vh), placed_rects) > max_placement_iou:
                continue

            bg = composite_sprite_on_bg(bg, canvas, px, py)
            placed_rects.append((vx, vy, vw, vh))
            if is_target:
                box = bbox_to_yolo(vx, vy, vw, vh, w_bg, h_bg, class_id)
                if box is not None:
                    bboxes.append(box)
                    stats.targets_placed += 1
                    stats.boxes.append((box.w, box.h))
                    if want_edge:
                        stats.edge_cropped += 1
                else:
                    stats.dropped_too_small += 1
            return

        if is_target:
            stats.placement_failures += 1

    if bucket in ("positive_only", "positive_with_negatives"):
        n_targets = rng.randint(components_min, components_max)
        stats.targets_requested += n_targets
        for _ in range(n_targets):
            place(rng.choice(sprites_base).copy(), is_target=True)
        if not bboxes:
            stats.empty_positive += 1

    if bucket in ("positive_with_negatives", "negative_only") and negative_sprites:
        for _ in range(rng.randint(1, 4)):
            _, neg = rng.choice(negative_sprites)
            place(neg.copy(), is_target=False)

    out_img_path.parent.mkdir(parents=True, exist_ok=True)
    out_label_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out_img_path), bg, [cv2.IMWRITE_JPEG_QUALITY, 92]):
        raise IOError(f"cv2.imwrite falló al escribir {out_img_path}")
    out_label_path.write_text(
        "".join(bb.to_line() + "\n" for bb in bboxes), encoding="utf-8"
    )
    stats.images += 1
    stats.groups[out_img_path.stem] = bg_path.stem
    return True


# ── Generación del dataset ────────────────────────────────────────────────────

def filter_negatives(negative_sprites, component_name: str,
                     exclude: list[str]) -> list[tuple[str, np.ndarray]]:
    """Saca de la bolsa de negativos el propio componente y sus parecidos.

    ESTE es el arreglo de H1. Antes la lista era global para todos los
    componentes con una única exclusión hardcodeada a
    ``seccionador_bajo_carga``, así que el símbolo objetivo se pegaba sin
    etiquetar dentro de sus propias imágenes de entrenamiento.
    """
    if not negative_sprites:
        return []
    banned = {component_name} | set(exclude)
    kept = [(n, im) for (n, im) in negative_sprites if n not in banned]
    removed = len(negative_sprites) - len(kept)
    if removed:
        print(f"[Fase 2/3] {component_name}: {removed} sprites excluidos de los "
              f"negativos ({', '.join(sorted(banned))})")
    return kept


def generate_synthetic_dataset(
    sprites_dir: Path,
    bg_dir: Path,
    output_dir: Path,
    n_total: int,
    component_name: str,
    cfg_g,
    class_id: int = 0,
    negative_sprites: list[tuple[str, np.ndarray]] | None = None,
    exclude_as_negative: list[str] | None = None,
    seed: int = 42,
) -> GenStats:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sprites_base = load_rgba_images(sprites_dir)
    if not sprites_base:
        raise FileNotFoundError(
            f"No hay sprites en {sprites_dir}. ¿Falló la fase 1 de "
            f"'{component_name}'?"
        )
    bg_paths = load_background_paths(bg_dir)

    negatives = filter_negatives(
        negative_sprites or [], component_name, exclude_as_negative or []
    )

    print(f"[Fase 2/3] {component_name}: {len(sprites_base)} sprites base | "
          f"{len(negatives)} negativos | {len(bg_paths)} fondos | {n_total} imágenes")

    rng = random.Random(f"{seed}:{component_name}")
    stats = GenStats()
    imgs_dir = output_dir / "images"
    lbls_dir = output_dir / "labels"
    prefix = f"{component_name}_"

    for i in range(n_total):
        r = rng.random()
        if r < 0.76:
            bucket = "positive_with_negatives"
        elif r < 0.85:
            bucket = "positive_only"
        else:
            bucket = "negative_only"

        stem = f"{prefix}img{i:05d}"
        generate_one_sample(
            sprites_base, negatives, bucket, bg_paths,
            imgs_dir / f"{stem}.jpg", lbls_dir / f"{stem}.txt",
            rng=rng, stats=stats, class_id=class_id,
            sprite_scale_min=cfg_g.sprite_scale_min,
            sprite_scale_max=cfg_g.sprite_scale_max,
            components_min=cfg_g.components_per_img_min,
            components_max=cfg_g.components_per_img_max,
            allow_rotation=cfg_g.allow_random_rotation,
            edge_crop_ratio=cfg_g.edge_crop_ratio,
            min_visible_area=cfg_g.min_visible_area,
            max_placement_iou=cfg_g.max_placement_iou,
        )

    print(stats.report(component_name))

    placed_ratio = (
        stats.targets_placed / stats.targets_requested
        if stats.targets_requested else 0.0
    )
    if placed_ratio < 0.75:
        raise RuntimeError(
            f"{component_name}: solo se colocó el {placed_ratio*100:.0f}% de las "
            f"instancias pedidas. El sprite es demasiado grande para el fondo o "
            f"max_placement_iou es demasiado estricto — revisá "
            f"sprite_scale_max y max_placement_iou antes de entrenar."
        )
    return stats
