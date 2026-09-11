# run_pipeline.py — Orquestador del pipeline completo.
#
# Cambios respecto de la versión anterior (auditoría H1, H20):
#   * Los sprites negativos se filtran POR COMPONENTE. Antes la lista era
#     global con una exclusión hardcodeada a 'seccionador_bajo_carga', así
#     que el símbolo objetivo se pegaba sin etiquetar en sus propias imágenes.
#     Ésta era la causa raíz del problema.
#   * Los errores por componente se registran y ese componente se saltea en
#     las fases siguientes, en vez de continuar con datos viejos o vacíos.
#   * Semilla global: antes la generación usaba `random` sin semilla y dos
#     corridas con la misma config daban datasets distintos.
#   * La verificación de labels corre siempre después de ensamblar y ABORTA
#     el entrenamiento si encuentra los síntomas de los bugs de la auditoría.
#   * El resumen final refleja todas las fases, no solo el entrenamiento.
from __future__ import annotations

import argparse
import gc
import logging
import random
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import cv2

from config import BASE_DIR, ComponentConfig, PipelineConfig, load_config
from generate_backgrounds import generate_backgrounds
from manual_ingestor import ingest_roboflow_zip
from phase1_extractor import generate_sprite_variations
from phase2_3_fusion_labeler import generate_synthetic_dataset
from phase4_assembler import assemble_synthetic
from verify_labels import verify_dataset

LOG_DIR = BASE_DIR / "logs"


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("LiardPipeline")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if logger.handlers:
        logger.handlers.clear()

    formatter = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
    LOG_DIR.mkdir(exist_ok=True)
    fh = logging.FileHandler(
        LOG_DIR / f"pipeline_{datetime.now():%Y%m%d_%H%M%S}.log", encoding="utf-8"
    )
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


log = setup_logger()


def _load_negative_sprites(cfg: PipelineConfig) -> list[tuple[str, np.ndarray]]:
    """Carga TODOS los sprites disponibles, etiquetados con su componente.

    El filtrado por componente lo hace generate_synthetic_dataset: acá solo
    se arma el catálogo completo.
    """
    negatives: list[tuple[str, np.ndarray]] = []
    for comp in cfg.components:
        for vi in range(cfg.component_variant_count(comp)):
            sdir = cfg.component_sprites_dir(comp, vi)
            if not sdir.exists():
                continue
            for p in sorted(sdir.glob("*.png")):
                img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
                if img is not None and img.ndim == 3 and img.shape[2] == 4:
                    negatives.append((comp.name, img))

    # Los modificadores (polos, anotaciones) son fragmentos, no símbolos
    # completos: se etiquetan como "modifier" y nunca se excluyen.
    if cfg.modifiers.dir.exists():
        for p in sorted(cfg.modifiers.dir.glob("*.png")):
            img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
            if img is not None and img.ndim == 3 and img.shape[2] == 4:
                negatives.append(("modifier", img))

    by_comp = len({n for n, _ in negatives})
    log.info(f"[GLOBAL] {len(negatives)} sprites negativos de {by_comp} orígenes")
    return negatives


def _phase1(cfg: PipelineConfig, comp: ComponentConfig) -> None:
    for vi, dxf_path in enumerate(comp.dxf_paths):
        log.info(f"[{comp.name}] FASE 1 — sprites ({dxf_path.name})")
        generate_sprite_variations(
            dxf_path=dxf_path,
            output_dir=cfg.component_sprites_dir(comp, vi),
            n_variations=comp.sprite_variations,
            kernel_min=comp.line_thickness_range[0],
            kernel_max=comp.line_thickness_range[1],
            component_name=comp.name,
        )


def _phase23(cfg: PipelineConfig, comp: ComponentConfig,
             negatives: list[tuple[str, np.ndarray]]) -> dict[str, str]:
    groups: dict[str, str] = {}
    for vi, dxf_path in enumerate(comp.dxf_paths):
        log.info(f"[{comp.name}] FASE 2/3 — generación sintética")
        stats = generate_synthetic_dataset(
            sprites_dir=cfg.component_sprites_dir(comp, vi),
            bg_dir=cfg.g.backgrounds_dir,
            output_dir=cfg.component_synthetic_dir(comp, vi),
            n_total=comp.images_to_generate,
            component_name=comp.name,
            cfg_g=cfg.g,
            class_id=comp.class_id,
            negative_sprites=negatives,
            exclude_as_negative=comp.exclude_as_negative,
            seed=cfg.g.seed,
        )
        groups.update(stats.groups)
    return groups


def run_pipeline(only: list[str] | None = None, skip_backgrounds: bool = False,
                 skip_training: bool = False, resume: bool = False) -> None:
    cfg = load_config()
    random.seed(cfg.g.seed)
    np.random.seed(cfg.g.seed)

    targets = [c for c in cfg.components if not c.skip_training]
    if only:
        targets = [c for c in targets if c.name in only]
        missing = set(only) - {c.name for c in targets}
        if missing:
            raise ValueError(f"Componentes desconocidos o con skip_training: {sorted(missing)}")

    log.info("=" * 64)
    log.info("  LIARD — pipeline de generación y entrenamiento")
    log.info("=" * 64)
    log.info(f"Componentes en el catálogo: {len(cfg.components)}")
    log.info(f"A entrenar: {[c.name for c in targets]}")

    for comp in cfg.components:
        for p in comp.dxf_paths:
            if not p.exists():
                raise FileNotFoundError(f"Falta el DXF: {p}")

    # ── Fase 0: fondos ────────────────────────────────────────────────────
    bg_dir = cfg.g.backgrounds_dir
    if cfg.backgrounds.enabled and not skip_backgrounds:
        log.info("[GLOBAL] FASE 0 — generando fondos desde planos reales")
        generate_backgrounds()
    if not (bg_dir.exists() and any(bg_dir.iterdir())):
        raise FileNotFoundError(
            f"No hay fondos en {bg_dir}. Activá backgrounds.enabled y poné "
            f"planos DXF en {cfg.backgrounds.dxf_sources_dir}."
        )
    n_bg = len(list(bg_dir.glob("*.jpg"))) + len(list(bg_dir.glob("*.png")))
    if n_bg < 50:
        log.warning(f"[GLOBAL] Solo {n_bg} fondos disponibles: poca variedad de contexto.")

    failed: dict[str, str] = {}
    results: list[tuple[str, str, float]] = []

    # ── Fase 1: sprites de TODOS los componentes ──────────────────────────
    log.info("\n" + "=" * 64)
    log.info("[GLOBAL] FASE 1 — sprites de todos los componentes")
    log.info("=" * 64)
    for comp in cfg.components:
        try:
            _phase1(cfg, comp)
        except Exception as exc:
            log.error(f"[{comp.name}] FALLÓ la fase 1: {exc}")
            failed[comp.name] = f"fase 1: {exc}"

    negatives = _load_negative_sprites(cfg)

    # ── Fase 2/3 + 4 + entrenamiento, por componente ──────────────────────
    log.info("\n" + "=" * 64)
    log.info("[GLOBAL] GENERACIÓN Y ENTRENAMIENTO POR COMPONENTE")
    log.info("=" * 64)

    for comp in targets:
        if comp.name in failed:
            log.warning(f"[{comp.name}] Se saltea: falló una fase anterior "
                        f"({failed[comp.name]})")
            results.append((comp.name, "SALTEADO", 0.0))
            continue

        t0 = time.time()
        status = "FALLO"
        try:
            synth_yaml = BASE_DIR / f"dataset_sintetico_{comp.name}" / f"sintetico_{comp.name}.yaml"
            if resume and synth_yaml.exists():
                log.info(f"[{comp.name}] REANUDANDO: salteando generación de datos sintéticos (ya existe)")
            else:
                groups = _phase23(cfg, comp, negatives)
                log.info(f"[{comp.name}] FASE 4 — ensamblando dataset con split real")
                synth_yaml = assemble_synthetic(comp.name, cfg, groups=groups)

            log.info(f"[{comp.name}] VERIFICACIÓN de etiquetas")
            verify_dataset(
                BASE_DIR / f"dataset_sintetico_{comp.name}",
                n_muestras=24, seed=cfg.g.seed, fail_on_error=True,
                sprites_dir=cfg.component_sprites_dir(comp, 0),
            )

            if skip_training:
                status = "DATOS OK"
                results.append((comp.name, status, time.time() - t0))
                continue

            log.info(f"[{comp.name}] FASE 1 ENTRENAMIENTO (sintético)")
            cmd = [sys.executable, "train.py", "--component", comp.name,
                   "--phase", "1", "--data-yaml", str(synth_yaml)]
            if resume:
                cmd.append("--resume")
            if subprocess.run(cmd, cwd=BASE_DIR).returncode != 0:
                raise RuntimeError("falló el entrenamiento sintético")

            best_synth = cfg.g.yolo_workspace / f"phase1_{comp.name}" / "weights" / "best.pt"

            if comp.roboflow_zip_path and comp.roboflow_zip_path.exists():
                log.info(f"[{comp.name}] INGESTA del ZIP de Roboflow")
                real_yaml = ingest_roboflow_zip(comp.roboflow_zip_path, comp.name, cfg)

                verify_dataset(
                    BASE_DIR / f"dataset_real_{comp.name}",
                    n_muestras=18, seed=cfg.g.seed, fail_on_error=False,
                    sprites_dir=cfg.component_sprites_dir(comp, 0),
                )

                log.info(f"[{comp.name}] FASE 2 ENTRENAMIENTO (fine-tune real)")
                cmd = [sys.executable, "train.py", "--component", comp.name,
                       "--phase", "2", "--data-yaml", str(real_yaml),
                       "--base-weights", str(best_synth)]
                if resume:
                    cmd.append("--resume")
                if subprocess.run(cmd, cwd=BASE_DIR).returncode != 0:
                    raise RuntimeError("falló el fine-tuning")
                status = "OK (fase 1+2)"
            else:
                log.warning(f"[{comp.name}] Sin ZIP de datos reales: solo fase 1")
                status = "OK (solo fase 1)"

        except Exception as exc:
            log.error(f"[{comp.name}] ERROR: {exc}")
            failed[comp.name] = str(exc)
        finally:
            gc.collect()
            results.append((comp.name, status, time.time() - t0))

    # ── Resumen ───────────────────────────────────────────────────────────
    log.info("\n" + "=" * 64)
    log.info("  RESUMEN FINAL")
    log.info("=" * 64)
    log.info(f"{'COMPONENTE'.ljust(38)} | {'ESTADO'.ljust(14)} | {'TIEMPO'.rjust(8)}")
    log.info("-" * 64)
    for name, status, elapsed in results:
        log.info(f"{name.ljust(38)} | {status.ljust(14)} | {f'{elapsed/60:.1f}m'.rjust(8)}")
    log.info("=" * 64)
    if failed:
        log.error(f"\n{len(failed)} componentes con error:")
        for name, why in failed.items():
            log.error(f"  · {name}: {why}")
        sys.exit(1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Pipeline Liard completo")
    ap.add_argument("--only", nargs="*", help="Solo estos componentes")
    ap.add_argument("--skip-backgrounds", action="store_true",
                    help="Reusa los fondos ya generados en output/backgrounds")
    ap.add_argument("--skip-training", action="store_true",
                    help="Genera y verifica los datos, sin entrenar")
    ap.add_argument("--resume", action="store_true",
                    help="Reanuda los runs de YOLO (ver train.py --resume)")
    args = ap.parse_args()
    run_pipeline(only=args.only, skip_backgrounds=args.skip_backgrounds,
                 skip_training=args.skip_training, resume=args.resume)
