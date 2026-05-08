# validate_dataset.py — Phase 4: Pre-training validation & data rescue
# Loops through every .jpg in the assembled dataset, verifies labels exist,
# clips out-of-range bounding box coordinates instead of deleting images,
# and only raises on completely broken formats.
from __future__ import annotations

import re
from pathlib import Path

_YOLO_LINE_RE = re.compile(
    r"^\s*(\d+)"                       # class_id
    r"\s+([\d.eE+-]+)"                 # cx
    r"\s+([\d.eE+-]+)"                 # cy
    r"\s+([\d.eE+-]+)"                 # w
    r"\s+([\d.eE+-]+)"                 # h
    r"\s*$"
)


def _clip_value(v: float) -> float:
    """Clamp a normalised coordinate to [0.0, 1.0]."""
    return max(0.0, min(1.0, v))


def validate_dataset(dataset_dir: Path, max_missing_labels_pct: float = 5.0) -> dict:
    """Run sanity checks on the assembled YOLO dataset.

    For every ``.jpg`` found under ``dataset_dir/images/[train|val|test]``:

    1. **Missing label**: if there is no corresponding ``.txt``, the image
       is flagged and an empty label file is created (treated as negative).
    2. **Bounding-box clipping**: if *cx, cy, w, h* slightly exceed the
       normalised ``[0.0, 1.0]`` range (common after rotations/transforms),
       the values are clamped and the ``.txt`` is overwritten — the image
       is **not** deleted.
    3. **Broken format**: lines that cannot be parsed at all are removed
       from the label file.  If **every** line is broken the image is
       deleted and counted as ``deleted``.

    Parameters
    ----------
    dataset_dir : Path
        Root of the YOLO dataset (contains ``images/`` and ``labels/``).
    max_missing_labels_pct : float
        Maximum percentage of images that may be missing labels before the
        pipeline aborts.  If exceeded, a ``RuntimeError`` is raised — this
        catches silent bugs in Phase 2/3 that drop ``.txt`` files.
        Default: 5.0 (%).

    Returns a summary dict with counts.
    """

    stats = {
        "total_images": 0,
        "labels_ok": 0,
        "labels_missing_created": 0,
        "bboxes_clipped": 0,
        "lines_removed": 0,
        "images_deleted": 0,
    }

    images_root = dataset_dir / "images"
    labels_root = dataset_dir / "labels"

    if not images_root.exists():
        print("[Validación] ⚠️  No se encontró la carpeta images/. Nada que validar.")
        return stats

    for split in ("train", "val", "test"):
        img_dir = images_root / split
        lbl_dir = labels_root / split
        if not img_dir.exists():
            continue

        for img_path in sorted(img_dir.glob("*.jpg")):
            stats["total_images"] += 1
            lbl_path = lbl_dir / (img_path.stem + ".txt")

            # ── 1. Missing label ──────────────────────────────────────────
            if not lbl_path.exists():
                lbl_path.touch()
                stats["labels_missing_created"] += 1
                continue

            # ── 2 & 3. Parse + clip / remove ──────────────────────────────
            raw_text = lbl_path.read_text(encoding="utf-8")
            if not raw_text.strip():
                # Empty label = negative sample → perfectly valid
                stats["labels_ok"] += 1
                continue

            good_lines: list[str] = []
            file_modified = False

            for line in raw_text.splitlines():
                line = line.strip()
                if not line:
                    continue

                m = _YOLO_LINE_RE.match(line)
                if m is None:
                    # Completely broken line — remove it
                    stats["lines_removed"] += 1
                    file_modified = True
                    continue

                cls_id = int(m.group(1))
                cx = float(m.group(2))
                cy = float(m.group(3))
                w = float(m.group(4))
                h = float(m.group(5))

                # Clip if slightly out of range
                clipped = False
                for val_name, val in [("cx", cx), ("cy", cy), ("w", w), ("h", h)]:
                    if val < 0.0 or val > 1.0:
                        clipped = True
                        break

                if clipped:
                    cx = _clip_value(cx)
                    cy = _clip_value(cy)
                    w = _clip_value(w)
                    h = _clip_value(h)
                    stats["bboxes_clipped"] += 1
                    file_modified = True

                good_lines.append(
                    f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
                )

            if not good_lines:
                # Every line was broken → delete the image entirely
                img_path.unlink(missing_ok=True)
                lbl_path.unlink(missing_ok=True)
                stats["images_deleted"] += 1
                continue

            if file_modified:
                lbl_path.write_text("\n".join(good_lines) + "\n", encoding="utf-8")

            stats["labels_ok"] += 1

    # ── Summary ───────────────────────────────────────────────────────────
    total = stats["total_images"]
    missing = stats["labels_missing_created"]
    missing_pct = (missing / total * 100) if total > 0 else 0.0

    print("\n" + "=" * 60)
    print("  VALIDACIÓN DEL DATASET")
    print("=" * 60)
    print(f"  Total imágenes revisadas:    {stats['total_images']}")
    print(f"  Labels OK:                   {stats['labels_ok']}")
    print(f"  Labels faltantes (creados):  {missing}  ({missing_pct:.1f}%)")
    print(f"  Bounding boxes clippeados:   {stats['bboxes_clipped']}")
    print(f"  Líneas rotas eliminadas:     {stats['lines_removed']}")
    print(f"  Imágenes eliminadas:         {stats['images_deleted']}")
    print("=" * 60 + "\n")

    # ── Safety threshold: abort if too many labels are missing ────────────
    if total > 0 and missing_pct > max_missing_labels_pct:
        raise RuntimeError(
            f"⛔ ABORTADO: {missing_pct:.1f}% de las imágenes no tenían label "
            f"(umbral: {max_missing_labels_pct}%).\n"
            f"   Esto indica un fallo sistémico en la generación de etiquetas "
            f"(Fase 2/3).\n"
            f"   Revisá los logs de generate_synthetic_dataset() antes de "
            f"continuar."
        )

    return stats

