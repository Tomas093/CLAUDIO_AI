# manual_ingestor.py — Ingesta de datasets reales exportados de Roboflow.
#
# Cambios respecto de la versión anterior (auditoría H2, H6, H18, H22, H23):
#   * El split se decide por la ruta RELATIVA al root del ZIP, con match
#     exacto de componente. Antes se hacía `"val" in str(ruta_absoluta)`, así
#     que cualquier carpeta ancestro con "val"/"test"/"train" en el nombre
#     desviaba el ZIP entero.
#   * Si el ZIP no trae valid/ (que es el caso de LOS DIEZ ZIPs actuales, son
#     100 % train), se hace el split acá con los ratios configurados. Antes se
#     caía silenciosamente a `val: train/images` y las métricas de fine-tuning
#     medían sobre el propio set de entrenamiento.
#   * Los negativos sintéticos inyectados tienen tope RELATIVO a la cantidad
#     de imágenes reales. Antes era una constante de 3000: con datasets reales
#     de 51 a 327 imágenes, el 90-98 % del fine-tune terminaba siendo fondo
#     sintético y el modelo colapsaba a "no detectar nada".
#   * Se verifica que el nombre de clase declarado en el data.yaml del ZIP se
#     parezca al del componente. Esto detecta ZIPs cruzados: hoy
#     instrumento_de_medicion_multifuncion.yolov11.zip es una copia byte a
#     byte del de seccionador_bajo_carga.
from __future__ import annotations

import random
import re
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from config import BASE_DIR, PipelineConfig

SPLITS = ("train", "val", "test")
_SPLIT_ALIASES = {
    "train": "train", "training": "train",
    "valid": "val", "val": "val", "validation": "val",
    "test": "test", "testing": "test",
}
_IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def _normalize(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def _detect_split(rel_parts: tuple[str, ...]) -> str | None:
    """Split según los componentes de la ruta RELATIVA al root del ZIP."""
    for part in rel_parts:
        alias = _SPLIT_ALIASES.get(part.lower())
        if alias:
            return alias
    return None


def _clean_label(src_txt: Path, target_id: int = 0) -> str:
    """Fuerza la clase y descarta líneas rotas. Devuelve el contenido."""
    if not src_txt.exists():
        return ""
    good: list[str] = []
    for line in src_txt.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            vals = [float(v) for v in parts[1:5]]
        except ValueError:
            continue
        if any(v != v for v in vals):     # NaN
            continue
        # clamp por esquinas y recálculo del centro (ver H16)
        cx, cy, w, h = vals
        x1, y1 = max(0.0, cx - w / 2), max(0.0, cy - h / 2)
        x2, y2 = min(1.0, cx + w / 2), min(1.0, cy + h / 2)
        w, h = x2 - x1, y2 - y1
        if w <= 1e-4 or h <= 1e-4:
            continue
        good.append(f"{target_id} {x1 + w/2:.6f} {y1 + h/2:.6f} {w:.6f} {h:.6f}")
    return "\n".join(good) + ("\n" if good else "")


def _check_class_name(temp_dir: Path, component_name: str) -> None:
    """Avisa si la clase del ZIP no se parece al componente.

    Detecta ZIPs cruzados o mal exportados. Hoy
    ``instrumento_de_medicion_multifuncion.yolov11.zip`` declara la clase
    ``seccionador-bajo-carga`` porque es una copia del ZIP equivocado.
    """
    yamls = list(temp_dir.rglob("data.yaml"))
    if not yamls:
        return
    txt = yamls[0].read_text(encoding="utf-8", errors="ignore")
    m = re.search(r"names:\s*\[([^\]]*)\]", txt)
    if not m:
        return
    declared = [c.strip().strip("'\"") for c in m.group(1).split(",") if c.strip()]
    if not declared:
        return
    want = _normalize(component_name)
    got = [_normalize(d) for d in declared]
    if not any(g in want or want in g for g in got):
        raise RuntimeError(
            f"⛔ El ZIP de '{component_name}' declara la clase {declared!r}.\n"
            f"   No coincide con el componente. Casi seguro es el ZIP de otra\n"
            f"   clase copiado con el nombre equivocado — verificá el archivo\n"
            f"   antes de entrenar, si no vas a fine-tunear con datos ajenos."
        )


def ingest_roboflow_zip(zip_path: str | Path, component_name: str,
                        cfg: PipelineConfig) -> Path:
    """Extrae un ZIP de Roboflow, normaliza a single-class y arma los splits.

    Devuelve la ruta absoluta al data.yaml generado.
    """
    zip_path = Path(zip_path)
    if not zip_path.exists():
        raise FileNotFoundError(f"No se encontró el ZIP de Roboflow: {zip_path}")

    dataset_dir = BASE_DIR / f"dataset_real_{component_name}"
    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)
    for split in SPLITS:
        (dataset_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (dataset_dir / split / "labels").mkdir(parents=True, exist_ok=True)

    temp_dir = Path(tempfile.mkdtemp(prefix=f"roboflow_{component_name}_"))
    rng = random.Random(f"{cfg.g.seed}:real:{component_name}")

    try:
        try:
            with zipfile.ZipFile(zip_path, "r") as z:
                z.extractall(temp_dir)
        except zipfile.BadZipFile as exc:
            raise RuntimeError(f"ZIP corrupto o inválido: {zip_path} — {exc}")

        _check_class_name(temp_dir, component_name)

        # ── recolectar (imagen, label) por split declarado en el ZIP ──────
        found: dict[str, list[tuple[Path, Path]]] = {s: [] for s in SPLITS}
        undeclared: list[tuple[Path, Path]] = []

        for img_src in sorted(temp_dir.rglob("*")):
            if not img_src.is_file() or img_src.suffix.lower() not in _IMG_EXT:
                continue
            rel = img_src.relative_to(temp_dir)
            split = _detect_split(rel.parts)
            lbl_src = img_src.parent.parent / "labels" / (img_src.stem + ".txt")
            if not lbl_src.exists():
                alt = list(temp_dir.rglob(img_src.stem + ".txt"))
                lbl_src = alt[0] if alt else lbl_src
            (found[split] if split else undeclared).append((img_src, lbl_src))

        total_declared = sum(len(v) for v in found.values())
        if not total_declared and not undeclared:
            raise RuntimeError(f"El ZIP {zip_path.name} no contiene imágenes.")

        # Los exports "dataset" de Roboflow vienen sin carpetas de split:
        # todo cae en train. En ese caso el split lo hacemos nosotros.
        need_split = (len(found["val"]) == 0)
        if need_split:
            pool = found["train"] + undeclared
            found = {s: [] for s in SPLITS}
            rng.shuffle(pool)
            n = len(pool)
            n_train = int(n * cfg.g.train_ratio)
            n_val = max(1, int(n * cfg.g.val_ratio))
            found["train"] = pool[:n_train]
            found["val"] = pool[n_train:n_train + n_val]
            found["test"] = pool[n_train + n_val:]
            print(f"[{component_name}] El ZIP no traía split de validación "
                  f"({n} imágenes, todas en train). Se hizo el split acá: "
                  f"{len(found['train'])}/{len(found['val'])}/{len(found['test'])}.")
        else:
            found["train"].extend(undeclared)

        # ── copiar ────────────────────────────────────────────────────────
        counts = {}
        for split in SPLITS:
            dst_i = dataset_dir / split / "images"
            dst_l = dataset_dir / split / "labels"
            for img_src, lbl_src in found[split]:
                shutil.copy2(img_src, dst_i / img_src.name)
                (dst_l / (img_src.stem + ".txt")).write_text(
                    _clean_label(lbl_src), encoding="utf-8"
                )
            counts[split] = len(found[split])

        if counts["val"] == 0:
            raise RuntimeError(
                f"⛔ El dataset real de '{component_name}' quedó sin imágenes de "
                f"validación. Sin val separado las métricas de fine-tuning no "
                f"significan nada (ver auditoría H2)."
            )
        if counts["val"] < 8:
            print(f"[{component_name}] ATENCIÓN: solo {counts['val']} imágenes de "
                  f"validación. Las métricas van a ser muy ruidosas.")

        # ── negativos sintéticos, con tope RELATIVO ───────────────────────
        _inject_synthetic_negatives(dataset_dir, component_name, cfg, counts, rng)

        # ── data.yaml ─────────────────────────────────────────────────────
        yaml_path = dataset_dir / f"real_{component_name}.yaml"
        yaml_path.write_text(
            f"# Liard — dataset real '{component_name}' (Roboflow)\n"
            f"# Origen: {zip_path.name}\n\n"
            f"path: {dataset_dir.resolve()}\n"
            f"train: train/images\n"
            f"val:   val/images\n"
            f"test:  test/images\n\n"
            f"nc: 1\n"
            f"names:\n"
            f"  0: {component_name}\n",
            encoding="utf-8",
        )
        print(f"[{component_name}] Dataset real: "
              f"train={counts['train']} val={counts['val']} test={counts['test']} "
              f"-> {yaml_path.name}")
        return yaml_path

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def _inject_synthetic_negatives(dataset_dir: Path, component_name: str,
                                cfg: PipelineConfig, counts: dict,
                                rng: random.Random) -> None:
    """Agrega negativos sintéticos al train real, con tope proporcional.

    El tope de 3000 fijo que había antes ahogaba datasets reales de 51-327
    imágenes: el fine-tuning pasaba a ser 90-98 % fondo sintético y el modelo
    aprendía a no detectar nada.
    """
    synth_dir = BASE_DIR / f"dataset_sintetico_{component_name}" / "train"
    src_imgs = synth_dir / "images"
    if not src_imgs.exists():
        return

    # Solo fondos vacíos. Los neg_conf_* son imágenes de otras clases y, hasta
    # que estén verificadas, no vale la pena arriesgar contaminación.
    candidates = sorted(src_imgs.glob("neg_bg_*.jpg"))
    if not candidates:
        return

    limit = int(counts["train"] * cfg.g.finetune_negative_ratio)
    if limit <= 0:
        return
    picked = rng.sample(candidates, min(limit, len(candidates)))

    dst_i = dataset_dir / "train" / "images"
    dst_l = dataset_dir / "train" / "labels"
    for p in picked:
        shutil.copy2(p, dst_i / p.name)
        (dst_l / (p.stem + ".txt")).write_text("", encoding="utf-8")

    print(f"[{component_name}] {len(picked)} negativos sintéticos inyectados "
          f"({cfg.g.finetune_negative_ratio*100:.0f}% de {counts['train']} reales)")
