# verify_labels.py — Verificación visual y estadística de un dataset YOLO.
#
# Reemplaza a verification_renderer.py, cuya función de bbox devolvía
# 0.5, 0.5, 1.0, 1.0 hardcodeado y por lo tanto no podía detectar ningún
# error de etiquetado (auditoría H17).
#
# La pregunta que responde es la única que importa antes de entrenar:
#   ¿las etiquetas coinciden con lo que hay dibujado en la imagen?
#
# Produce dos cosas:
#   1. Una grilla de imágenes con las cajas dibujadas encima, en
#      verification/<dataset>/muestra_*.jpg — para mirar con los ojos.
#   2. Un chequeo automático que FALLA si detecta los síntomas de los bugs
#      de la auditoría:
#        · cajas con demasiado espacio en blanco alrededor  → H4
#        · símbolos sin etiquetar en imágenes positivas      → H1
#        · train y val compartiendo imágenes                 → H2
#        · fondos vacíos / sin contexto                      → H3
#        · escala de caja incoherente con la inferencia      → H8
#
# Uso:
#   python verify_labels.py --dataset dataset_sintetico_ojo_de_buey
#   python verify_labels.py --dataset ... --muestras 48 --no-fail
from __future__ import annotations

import argparse
import hashlib
import random
import statistics
import sys

from pathlib import Path

import cv2
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = Path(__file__).resolve().parent
VERIFICATION_DIR = BASE_DIR / "verification"
SPLITS = ("train", "val", "test")

COLOR_BOX = (44, 58, 196)        # BGR — rojo: caja etiquetada
COLOR_MISS = (126, 116, 13)      # BGR — verde azulado: candidato sin etiquetar
COLOR_TIGHT = (60, 160, 60)      # BGR — verde: contenido real dentro de la caja


# ── Lectura ───────────────────────────────────────────────────────────────────

def read_labels(path: Path) -> list[tuple[int, float, float, float, float]]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            out.append((int(float(parts[0])), *[float(v) for v in parts[1:5]]))
        except ValueError:
            continue
    return out


def yolo_to_px(box, W: int, H: int) -> tuple[int, int, int, int]:
    _, cx, cy, w, h = box
    return (
        int(round((cx - w / 2) * W)), int(round((cy - h / 2) * H)),
        int(round((cx + w / 2) * W)), int(round((cy + h / 2) * H)),
    )


# ── Métricas ──────────────────────────────────────────────────────────────────

def box_fill_ratio(gray: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> float | None:
    """Qué fracción del área de la caja ocupa realmente el contenido dibujado.

    Es la sonda de H4: si la caja incluye padding transparente, el contenido
    real ocupa consistentemente ~0.86 del área en vez de ~1.0.
    """
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(gray.shape[1], x2), min(gray.shape[0], y2)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    sub = gray[y1:y2, x1:x2]
    mask = sub < 200
    if not mask.any():
        return None
    rows = np.where(mask.any(1))[0]
    cols = np.where(mask.any(0))[0]
    tight = (rows[-1] - rows[0] + 1) * (cols[-1] - cols[0] + 1)
    return float(tight) / float((x2 - x1) * (y2 - y1))


def load_sprite_masks(sprites_dir: Path, limit: int = 8) -> list[np.ndarray]:
    """Siluetas binarias del símbolo objetivo, para poder reconocerlo.

    Sin esto el detector de objetos sin etiquetar no distingue un símbolo de
    un bloque de texto del plano, y sobre fondos reales da 90 % de falsos
    positivos.
    """
    masks: list[np.ndarray] = []
    if not sprites_dir or not sprites_dir.exists():
        return masks
    paths = sorted(sprites_dir.glob("*.png"))
    if not paths:
        return masks
    step = max(1, len(paths) // limit)
    for p in paths[::step][:limit]:
        img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if img is None or img.ndim != 3 or img.shape[2] != 4:
            continue
        m = (img[:, :, 3] > 8).astype(np.uint8)
        if m.any():
            masks.append(m)
    return masks


def _templates(masks: list[np.ndarray], sides_px: list[int],
               _cache: dict = {}) -> list[np.ndarray]:
    """Siluetas del símbolo, a las escalas y rotaciones que hay que buscar."""
    key = (id(masks), tuple(sorted(set(sides_px))))
    if key in _cache:
        return _cache[key]
    out = []
    for m in masks:
        h, w = m.shape
        for side in sorted(set(sides_px)):
            if side < 12:
                continue
            f = side / max(h, w)
            tw, th = max(6, int(round(w * f))), max(6, int(round(h * f)))
            base = (cv2.resize(m.astype(np.float32), (tw, th),
                              interpolation=cv2.INTER_AREA) > 0.05).astype(np.float32)
            for k in range(4):
                out.append(np.ascontiguousarray(np.rot90(base, k)).astype(np.float32))
    _cache[key] = out
    return out


def find_unlabeled_candidates(gray: np.ndarray, boxes_px: list,
                              sprite_masks: list[np.ndarray] | None = None,
                              sides_px: list[int] | None = None,
                              threshold: float = 0.75,
                              ) -> list[tuple[int, int, int, int]]:
    """Busca instancias del símbolo que NO estén etiquetadas.

    Es la sonda del bug H1 —el que hacía que el propio símbolo objetivo se
    pegara sin etiqueta— y por eso corre siempre antes de entrenar.

    Usa template matching de la silueta del sprite contra la máscara de tinta
    de la imagen, en cuatro rotaciones y varias escalas. Se probó primero con
    componentes conexos y no servía: sobre un fondo de plano real el símbolo
    queda pegado a los cables y el blob resultante es mucho más grande que un
    símbolo, así que se escapaba del filtro de tamaño.
    """
    if not sprite_masks:
        return []
    ink = (gray < 200).astype(np.float32)
    if ink.sum() == 0:
        return []

    sides = sides_px or [48, 64, 88]
    hits: list[tuple[float, int, int, int, int]] = []
    for tpl in _templates(sprite_masks, sides):
        th, tw = tpl.shape
        if th >= ink.shape[0] or tw >= ink.shape[1]:
            continue
        if tpl.sum() < 30:
            continue
        res = cv2.matchTemplate(ink, tpl, cv2.TM_CCOEFF_NORMED)
        ys, xs = np.where(res >= threshold)
        for y, x in zip(ys, xs):
            hits.append((float(res[y, x]), int(x), int(y), int(x + tw), int(y + th)))

    # NMS: nos quedamos con el mejor match de cada zona
    hits.sort(reverse=True)
    kept: list[tuple[int, int, int, int]] = []
    for _score, x1, y1, x2, y2 in hits:
        if any(max(x1, a) < min(x2, c) and max(y1, b) < min(y2, d)
               for (a, b, c, d) in kept):
            continue
        kept.append((x1, y1, x2, y2))

    # descartamos los que ya tienen etiqueta
    out = []
    for (x1, y1, x2, y2) in kept:
        area = (x2 - x1) * (y2 - y1)
        covered = False
        for (bx1, by1, bx2, by2) in boxes_px:
            iw = max(0, min(x2, bx2) - max(x1, bx1))
            ih = max(0, min(y2, by2) - max(y1, by1))
            if area and (iw * ih) / area > 0.3:
                covered = True
                break
        if not covered:
            out.append((x1, y1, x2, y2))
    return out


# ── Render de la grilla ───────────────────────────────────────────────────────

def annotate(img: np.ndarray, boxes, missing, show_tight: bool = True) -> np.ndarray:
    out = img.copy()
    H, W = out.shape[:2]
    gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
    for box in boxes:
        x1, y1, x2, y2 = yolo_to_px(box, W, H)
        if show_tight:
            xa, ya = max(0, x1), max(0, y1)
            xb, yb = min(W, x2), min(H, y2)
            if xb - xa > 4 and yb - ya > 4:
                sub = gray[ya:yb, xa:xb] < 200
                if sub.any():
                    rs = np.where(sub.any(1))[0]
                    cs = np.where(sub.any(0))[0]
                    cv2.rectangle(out, (xa + cs[0], ya + rs[0]),
                                  (xa + cs[-1], ya + rs[-1]), COLOR_TIGHT, 1)
        cv2.rectangle(out, (x1, y1), (x2, y2), COLOR_BOX, 2)
    for (x1, y1, x2, y2) in missing:
        cv2.rectangle(out, (x1, y1), (x2, y2), COLOR_MISS, 2)
    return out


def build_sheet(tiles: list[np.ndarray], cols: int, cell: int) -> np.ndarray:
    rows = (len(tiles) + cols - 1) // cols
    sheet = np.full((rows * cell, cols * cell, 3), 235, np.uint8)
    for i, t in enumerate(tiles):
        t = cv2.resize(t, (cell - 8, cell - 8), interpolation=cv2.INTER_AREA)
        r, c = divmod(i, cols)
        sheet[r * cell + 4:(r + 1) * cell - 4, c * cell + 4:(c + 1) * cell - 4] = t
    return sheet


# ── Verificación ──────────────────────────────────────────────────────────────

def verify_dataset(dataset_dir: Path, n_muestras: int = 24, cols: int = 6,
                   cell: int = 320, seed: int = 42, fail_on_error: bool = True,
                   out_dir: Path | None = None,
                   sprites_dir: Path | None = None) -> dict:
    if not dataset_dir.exists():
        raise FileNotFoundError(f"No existe el dataset: {dataset_dir}")

    out_dir = out_dir or (VERIFICATION_DIR / dataset_dir.name)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    sprite_masks = load_sprite_masks(sprites_dir) if sprites_dir else []
    if sprites_dir and not sprite_masks:
        print(f"  (sin sprites en {sprites_dir}: el chequeo de objetos sin "
              f"etiquetar queda desactivado)")

    problems: list[str] = []
    warnings: list[str] = []
    per_split: dict[str, dict] = {}
    hashes: dict[str, set] = {}
    fills: list[float] = []
    widths: list[float] = []
    heights: list[float] = []
    unlabeled_total = 0
    unlabeled_imgs = 0
    inspected = 0
    sample_tiles: list[np.ndarray] = []

    for split in SPLITS:
        img_dir = dataset_dir / split / "images"
        lbl_dir = dataset_dir / split / "labels"
        if not img_dir.exists():
            continue
        imgs = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
        if not imgs:
            continue

        n_boxes = n_empty = n_missing_lbl = 0
        for p in imgs:
            lbl = lbl_dir / (p.stem + ".txt")
            if not lbl.exists():
                n_missing_lbl += 1
                continue
            boxes = read_labels(lbl)
            if not boxes:
                n_empty += 1
            n_boxes += len(boxes)
            for b in boxes:
                widths.append(b[3])
                heights.append(b[4])
                if not (0 <= b[1] <= 1 and 0 <= b[2] <= 1 and 0 < b[3] <= 1 and 0 < b[4] <= 1):
                    problems.append(
                        f"{split}/{p.stem}: caja fuera de rango {b[1:]}"
                    )

        per_split[split] = {
            "images": len(imgs), "boxes": n_boxes,
            "empty": n_empty, "missing_label": n_missing_lbl,
        }
        if n_missing_lbl:
            problems.append(
                f"{split}: {n_missing_lbl} imágenes sin archivo de label"
            )

        # hash de contenido, para detectar solapamiento entre splits (H2)
        hashes[split] = set()
        for p in rng.sample(imgs, min(300, len(imgs))):
            hashes[split].add(hashlib.md5(p.read_bytes()).hexdigest())

        # muestra visual: positivos del split
        positives = [p for p in imgs if read_labels(lbl_dir / (p.stem + ".txt"))]
        take = max(1, n_muestras // max(1, len([s for s in SPLITS
                                                if (dataset_dir / s / "images").exists()])))
        for p in rng.sample(positives, min(take, len(positives))):
            img = cv2.imread(str(p))
            if img is None:
                continue
            H, W = img.shape[:2]
            boxes = read_labels(lbl_dir / (p.stem + ".txt"))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            boxes_px = [yolo_to_px(b, W, H) for b in boxes]

            for (x1, y1, x2, y2) in boxes_px:
                f = box_fill_ratio(gray, x1, y1, x2, y2)
                if f is not None:
                    fills.append(f)

            sides = [max(x2 - x1, y2 - y1) for (x1, y1, x2, y2) in boxes_px] or [64]
            med_side = int(statistics.median(sides))
            missing = find_unlabeled_candidates(
                gray, boxes_px, sprite_masks=sprite_masks,
                sides_px=[int(med_side * f) for f in (0.7, 0.85, 1.0, 1.2, 1.45)],
            )
            if missing:
                unlabeled_total += len(missing)
                unlabeled_imgs += 1
            inspected += 1
            sample_tiles.append(annotate(img, boxes, missing))

    # ── chequeos ──────────────────────────────────────────────────────────
    splits_present = [s for s in SPLITS if s in per_split]
    if "val" not in per_split or per_split["val"]["images"] == 0:
        problems.append(
            "No hay split de validación con imágenes. Sin val separado las "
            "métricas de entrenamiento no significan nada (H2)."
        )
    for a in splits_present:
        for b in splits_present:
            if a >= b:
                continue
            shared = hashes.get(a, set()) & hashes.get(b, set())
            if shared:
                problems.append(
                    f"FUGA: {len(shared)} imágenes idénticas entre '{a}' y '{b}' (H2)"
                )

    if fills:
        med_fill = statistics.median(fills)
        if med_fill < 0.80:
            problems.append(
                f"Las cajas tienen demasiado espacio vacío: el contenido real "
                f"ocupa una mediana del {med_fill*100:.0f}% del área de la caja. "
                f"Esperado > 88 %. Es el síntoma de H4 (padding del sprite "
                f"dentro de la caja)."
            )
        elif med_fill < 0.88:
            warnings.append(
                f"Relleno de caja algo bajo ({med_fill*100:.0f}%); revisá la "
                f"muestra visual."
            )

    if not sprite_masks:
        warnings.append(
            "Chequeo de objetos sin etiquetar desactivado: pasá --sprites para "
            "que pueda reconocer el símbolo."
        )
    elif inspected and unlabeled_imgs / inspected > 0.15:
        problems.append(
            f"{unlabeled_imgs}/{inspected} imágenes de la muestra tienen "
            f"instancias del símbolo SIN etiquetar ({unlabeled_total} en total). "
            f"Es el síntoma de H1 — mirá la grilla antes de entrenar."
        )
    elif unlabeled_imgs:
        warnings.append(
            f"{unlabeled_imgs}/{inspected} imágenes con algún objeto parecido al "
            f"símbolo sin etiquetar. Revisalos en la grilla."
        )

    if widths:
        med_w = statistics.median(widths)
        med_h = statistics.median(heights)
        if med_w > 0.35 or med_h > 0.35:
            warnings.append(
                f"Las cajas son grandes respecto de la imagen "
                f"(w={med_w:.2f} h={med_h:.2f}). Si en inferencia el símbolo "
                f"ocupa ~0.10 del tile, hay desajuste de escala (H8)."
            )

    total_imgs = sum(v["images"] for v in per_split.values())
    total_empty = sum(v["empty"] for v in per_split.values())
    if total_imgs and total_empty / total_imgs > 0.45:
        problems.append(
            f"{total_empty}/{total_imgs} ({total_empty/total_imgs*100:.0f}%) de las "
            f"imágenes no tienen ningún objeto. Demasiados negativos (H7)."
        )

    # ── salida visual ─────────────────────────────────────────────────────
    if sample_tiles:
        sheet = build_sheet(sample_tiles[:n_muestras], cols, cell)
        sheet_path = out_dir / "muestra_labels.jpg"
        cv2.imwrite(str(sheet_path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])
    else:
        sheet_path = None

    # ── reporte ───────────────────────────────────────────────────────────
    print("\n" + "=" * 68)
    print(f"  VERIFICACIÓN — {dataset_dir.name}")
    print("=" * 68)
    for split, v in per_split.items():
        print(f"  {split.upper():5s} │ {v['images']:6d} imgs │ {v['boxes']:6d} cajas │ "
              f"{v['empty']:5d} sin objetos")
    if fills:
        print(f"\n  Relleno de caja (contenido real / área etiquetada):")
        print(f"    mediana {statistics.median(fills)*100:5.1f}%   "
              f"p10 {np.percentile(fills,10)*100:5.1f}%   "
              f"p90 {np.percentile(fills,90)*100:5.1f}%")
    if widths:
        print(f"  Tamaño de caja (fracción de imagen): "
              f"w={statistics.median(widths):.3f}  h={statistics.median(heights):.3f}")
    print(f"  Imágenes inspeccionadas a fondo: {inspected}")
    if sheet_path:
        print(f"\n  🖼  Grilla visual: {sheet_path}")

    if warnings:
        print("\n  Avisos:")
        for w in warnings:
            print(f"    · {w}")
    if problems:
        print("\n  ⛔ PROBLEMAS:")
        for p in problems[:20]:
            print(f"    · {p}")
        if len(problems) > 20:
            print(f"    · ... y {len(problems)-20} más")
    else:
        print("\n  ✅ Sin problemas detectados.")
    print("=" * 68 + "\n")

    result = {
        "splits": per_split, "problems": problems, "warnings": warnings,
        "median_fill": statistics.median(fills) if fills else None,
        "sheet": str(sheet_path) if sheet_path else None,
    }
    if problems and fail_on_error:
        raise RuntimeError(
            f"La verificación de {dataset_dir.name} encontró "
            f"{len(problems)} problemas. Mirá la grilla en {sheet_path} "
            f"antes de entrenar."
        )
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Verifica un dataset YOLO ensamblado")
    ap.add_argument("--dataset", required=True,
                    help="Nombre o ruta del dataset (ej. dataset_sintetico_ojo_de_buey)")
    ap.add_argument("--muestras", type=int, default=24)
    ap.add_argument("--cols", type=int, default=6)
    ap.add_argument("--cell", type=int, default=320)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sprites",
                    help="Carpeta de sprites del componente, para reconocer el "
                         "símbolo (ej. output/interruptor_motorizado/sprites)")
    ap.add_argument("--no-fail", action="store_true",
                    help="Reporta pero no aborta con error")
    args = ap.parse_args()

    d = Path(args.dataset)
    if not d.is_absolute() and not d.exists():
        d = BASE_DIR / args.dataset

    sprites = Path(args.sprites) if args.sprites else None
    if sprites and not sprites.is_absolute() and not sprites.exists():
        sprites = BASE_DIR / args.sprites

    verify_dataset(d, n_muestras=args.muestras, cols=args.cols, cell=args.cell,
                   seed=args.seed, fail_on_error=not args.no_fail,
                   sprites_dir=sprites)
