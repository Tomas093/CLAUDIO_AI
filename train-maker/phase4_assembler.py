# phase4_assembler.py — Ensamblado del dataset sintético.
#
# Cambios respecto de la versión anterior (auditoría H2, H7, H21):
#   * SPLIT REAL train/val/test. Antes el data.yaml escribía
#     `val: train/images` y las métricas se calculaban sobre el propio set de
#     entrenamiento (mAP50 = 0.96 en la época 1).
#   * El split es AGRUPADO POR FONDO: un mismo tile de plano nunca aparece en
#     dos splits. Sin esto, separar por imagen sigue dejando fuga: la misma
#     porción de plano se ve en train y en val.
#   * Presupuesto ÚNICO de negativos, proporcional a los positivos. Antes se
#     acumulaban por tres vías (bucket negative_only + negative_ratio + una
#     constante de 3000) y el dataset terminaba ~50 % vacío.
#   * Muestreo SIN reemplazo: `rng.choices` duplicaba imágenes.
#   * Los negativos se reparten entre los splits, no van todos a train.
#   * Se eliminaron los cuatro stubs vacíos (`split_and_copy`,
#     `generate_yolo_yaml`, `create_data_yaml`, `assemble_dataset`) que solo
#     existían para que ib_maker.py pudiera importarlos.
from __future__ import annotations

import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import cv2
import numpy as np

from config import BASE_DIR, ComponentConfig, PipelineConfig

SPLITS = ("train", "val", "test")


# ── Utilidades ────────────────────────────────────────────────────────────────

def _remap_class_ids(label_path: Path, target_id: int = 0) -> str:
    """Lee un label YOLO, fuerza la clase a *target_id* y descarta líneas rotas."""
    if not label_path.exists():
        return ""
    lines: list[str] = []
    for raw in label_path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        parts = raw.split()
        if len(parts) < 5:
            continue
        parts[0] = str(target_id)
        lines.append(" ".join(parts[:5]))
    return "\n".join(lines) + ("\n" if lines else "")


def _make_dirs(dataset_dir: Path) -> dict[str, Path]:
    dirs: dict[str, Path] = {}
    for split in SPLITS:
        for sub in ("images", "labels"):
            d = dataset_dir / split / sub
            d.mkdir(parents=True, exist_ok=True)
            dirs[f"{split}_{sub}"] = d
    return dirs


def split_by_group(
    items: list[tuple[Path, Path]],
    groups: dict[str, str],
    ratios: tuple[float, float, float],
    rng: random.Random,
) -> dict[str, list[tuple[Path, Path]]]:
    """Reparte (imagen, label) en train/val/test agrupando por fondo.

    Todas las imágenes que comparten el mismo tile de fondo caen en el mismo
    split. Es lo que evita que el modelo vea en validación un fondo que ya
    memorizó durante el entrenamiento.
    """
    by_group: dict[str, list[tuple[Path, Path]]] = defaultdict(list)
    for img, lbl in items:
        by_group[groups.get(img.stem, img.stem)].append((img, lbl))

    keys = sorted(by_group)
    rng.shuffle(keys)

    total = len(items)
    want = {
        "train": ratios[0] * total,
        "val": ratios[1] * total,
        "test": ratios[2] * total,
    }
    out: dict[str, list[tuple[Path, Path]]] = {s: [] for s in SPLITS}

    # Greedy: cada grupo va al split que esté más lejos de su cuota.
    for key in keys:
        group = by_group[key]
        deficits = {s: want[s] - len(out[s]) for s in SPLITS if want[s] > 0}
        target = max(deficits, key=deficits.get)
        out[target].extend(group)

    return out


# ── Negativos ─────────────────────────────────────────────────────────────────

def _augment_background(img: np.ndarray, rng: random.Random) -> np.ndarray:
    """Variante de un fondo vacío, para que los negativos no sean todos iguales."""
    h, w = img.shape[:2]

    crop_frac = rng.uniform(0.70, 1.0)
    ch, cw = int(h * crop_frac), int(w * crop_frac)
    y0 = rng.randint(0, max(h - ch, 0))
    x0 = rng.randint(0, max(w - cw, 0))
    out = img[y0:y0 + ch, x0:x0 + cw]
    if out.shape[0] != h or out.shape[1] != w:
        out = cv2.resize(out, (w, h), interpolation=cv2.INTER_LINEAR)

    out = cv2.convertScaleAbs(out, alpha=rng.uniform(0.85, 1.15), beta=rng.randint(-20, 20))

    if rng.random() < 0.5:
        textos = ["TUG", "IUE", "C1", "2x16A", "RESERVA", "10 mm2", "C2",
                  "2x10A", "L1", "L2", "L3", "PDT.NOR", "TABLERO"]
        for _ in range(rng.randint(1, 3)):
            cv2.putText(
                out, rng.choice(textos),
                (rng.randint(10, max(11, w - 90)), rng.randint(20, max(21, h - 20))),
                cv2.FONT_HERSHEY_SIMPLEX, rng.uniform(0.4, 0.8), (0, 0, 0),
                rng.randint(1, 2),
            )
    return out


def _collect_negative_sources(component_name: str, cfg: PipelineConfig,
                              comp: ComponentConfig) -> dict[str, list[Path]]:
    """Imágenes candidatas a negativo, por clase de origen.

    Se excluyen el propio componente y los declarados en
    ``exclude_as_negative``: meter un símbolo casi idéntico al target como
    fondo sin etiqueta es el mismo error que H1.
    """
    banned = {component_name} | set(comp.exclude_as_negative)
    sources: dict[str, list[Path]] = {}
    for other in cfg.components:
        if other.name in banned:
            continue
        d = cfg.g.output_dir / other.name / "synthetic" / "images"
        if not d.exists():
            continue
        paths = sorted(d.glob("*.jpg"))
        if paths:
            sources[other.name] = paths
    return sources


def build_negatives(component_name: str, cfg: PipelineConfig, comp: ComponentConfig,
                    n_positives: int, rng: random.Random) -> list[tuple[str, Path | np.ndarray]]:
    """Arma la lista de negativos con UN solo presupuesto.

    Devuelve ``(stem, origen)`` donde el origen es un Path a copiar o una
    imagen ya generada en memoria.
    """
    budget = int(n_positives * cfg.g.negative_ratio)
    if budget <= 0:
        return []

    n_empty = int(budget * cfg.g.negative_empty_fraction)
    n_confusing = budget - n_empty
    out: list[tuple[str, Path | np.ndarray]] = []

    # 1. Fondos vacíos con texto
    bg_paths = sorted(cfg.g.backgrounds_dir.glob("*.jpg")) + \
               sorted(cfg.g.backgrounds_dir.glob("*.png"))
    if bg_paths and n_empty > 0:
        for i in range(n_empty):
            img = cv2.imread(str(rng.choice(bg_paths)))
            if img is not None:
                out.append((f"neg_bg_{i:05d}", _augment_background(img, rng)))

    # 2. Negativos manuales curados (falsos positivos reales)
    manual_dir = BASE_DIR / f"negatives_{component_name}"
    manual = sorted(manual_dir.glob("*.png")) + sorted(manual_dir.glob("*.jpg")) \
        if manual_dir.exists() else []
    for i, p in enumerate(manual):
        out.append((f"neg_manual_{p.stem}_{i:03d}", p))
    if manual:
        print(f"[Fase 4] {len(manual)} negativos manuales desde {manual_dir.name}/")

    # 3. Componentes confusos, repartidos según hard_negatives
    sources = _collect_negative_sources(component_name, cfg, comp)
    n_confusing = max(0, n_confusing - len(manual))
    if sources and n_confusing > 0:
        weights = {k: v for k, v in comp.hard_negatives.items() if k in sources}
        if not weights:
            weights = {k: 1.0 for k in sources}
        total_w = sum(weights.values()) or 1.0

        for name, weight in sorted(weights.items()):
            paths = sources[name]
            k = int(round(n_confusing * weight / total_w))
            if k <= 0:
                continue
            # sin reemplazo: rng.choices duplicaba la misma imagen N veces
            picked = rng.sample(paths, min(k, len(paths)))
            if len(picked) < k:
                print(f"[Fase 4]   {name}: solo hay {len(picked)} imágenes "
                      f"disponibles de las {k} pedidas")
            for i, p in enumerate(picked):
                out.append((f"neg_conf_{name}_{i:05d}", p))

    missing = {n for n in comp.hard_negatives if n not in sources}
    if missing:
        print(f"[Fase 4] ATENCIÓN: hard_negatives sin datos generados: "
              f"{sorted(missing)} (¿les falta correr la fase 2/3?)")

    return out


# ── Resumen ───────────────────────────────────────────────────────────────────

def print_dataset_summary(dataset_dir: Path) -> dict:
    print("\n" + "=" * 64)
    print(f"  RESUMEN — {dataset_dir.name}")
    print("=" * 64)
    summary = {}
    total_imgs = total_boxes = 0
    for split in SPLITS:
        imgs_dir = dataset_dir / split / "images"
        lbls_dir = dataset_dir / split / "labels"
        if not imgs_dir.exists():
            continue
        n_imgs = len(list(imgs_dir.glob("*.*")))
        n_neg = n_boxes = 0
        for f in lbls_dir.glob("*.txt"):
            lines = [l for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
            if not lines:
                n_neg += 1
            n_boxes += len(lines)
        n_pos = n_imgs - n_neg
        pct = (n_neg / n_imgs * 100) if n_imgs else 0
        summary[split] = {"images": n_imgs, "pos": n_pos, "neg": n_neg, "boxes": n_boxes}
        total_imgs += n_imgs
        total_boxes += n_boxes
        print(f"  {split.upper():5s} │ {n_imgs:6d} imgs │ {n_pos:6d} pos │ "
              f"{n_neg:5d} neg ({pct:4.1f}%) │ {n_boxes:6d} cajas")
    print(f"  {'TOTAL':5s} │ {total_imgs:6d} imgs │ {'':6s}   │ {'':5s}         │ "
          f"{total_boxes:6d} cajas")
    print("=" * 64 + "\n")
    return summary


# ── Entrada principal ─────────────────────────────────────────────────────────

def assemble_synthetic(component_name: str, cfg: PipelineConfig,
                       groups: dict[str, str] | None = None) -> Path:
    """Ensambla el dataset sintético de un componente con split real."""
    comp = cfg.get(component_name)
    dataset_dir = BASE_DIR / f"dataset_sintetico_{component_name}"
    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)
    dirs = _make_dirs(dataset_dir)
    rng = random.Random(f"{cfg.g.seed}:assemble:{component_name}")

    # ── positivos ─────────────────────────────────────────────────────────
    pairs: list[tuple[Path, Path]] = []
    for vi in range(cfg.component_variant_count(comp)):
        syn = cfg.component_synthetic_dir(comp, vi)
        imgs, lbls = syn / "images", syn / "labels"
        if not imgs.exists():
            continue
        for img in sorted(imgs.glob("*.jpg")):
            pairs.append((img, lbls / (img.stem + ".txt")))

    if not pairs:
        raise FileNotFoundError(
            f"No hay imágenes sintéticas para '{component_name}'. "
            f"¿Corrió la fase 2/3?"
        )

    ratios = (cfg.g.train_ratio, cfg.g.val_ratio, cfg.g.test_ratio)
    parts = split_by_group(pairs, groups or {}, ratios, rng)

    for split, items in parts.items():
        for img, lbl in items:
            shutil.copy2(img, dirs[f"{split}_images"] / img.name)
            (dirs[f"{split}_labels"] / (img.stem + ".txt")).write_text(
                _remap_class_ids(lbl, comp.class_id), encoding="utf-8"
            )

    # ── negativos, repartidos entre splits con los mismos ratios ──────────
    negatives = build_negatives(component_name, cfg, comp, len(pairs), rng)
    rng.shuffle(negatives)
    n = len(negatives)
    bounds = (int(n * ratios[0]), int(n * (ratios[0] + ratios[1])))
    for i, (stem, src) in enumerate(negatives):
        split = "train" if i < bounds[0] else ("val" if i < bounds[1] else "test")
        dst = dirs[f"{split}_images"] / f"{stem}.jpg"
        if isinstance(src, Path):
            shutil.copy2(src, dst)
        else:
            cv2.imwrite(str(dst), src, [cv2.IMWRITE_JPEG_QUALITY, 92])
        (dirs[f"{split}_labels"] / f"{stem}.txt").write_text("", encoding="utf-8")

    print(f"[Fase 4] {len(pairs)} positivos + {n} negativos "
          f"({n / max(1, len(pairs)) * 100:.0f}% del total de positivos)")

    # ── data.yaml ─────────────────────────────────────────────────────────
    yaml_path = dataset_dir / f"sintetico_{component_name}.yaml"
    yaml_path.write_text(
        f"# Liard — dataset sintético '{component_name}'\n"
        f"# Split agrupado por fondo: un mismo tile nunca cae en dos splits.\n\n"
        f"path: {dataset_dir.resolve()}\n"
        f"train: train/images\n"
        f"val:   val/images\n"
        f"test:  test/images\n\n"
        f"nc: 1\n"
        f"names:\n"
        f"  0: {component_name}\n",
        encoding="utf-8",
    )

    summary = print_dataset_summary(dataset_dir)
    if summary.get("val", {}).get("images", 0) < 20:
        raise RuntimeError(
            f"El split de validación de '{component_name}' quedó con "
            f"{summary.get('val', {}).get('images', 0)} imágenes. Es demasiado "
            f"chico para que las métricas signifiquen algo."
        )
    print(f"[Fase 4] Dataset ensamblado en '{dataset_dir.name}'")
    return yaml_path
