# run_pipeline.py — Phase 5: Master Controller
# Iterates through every component defined in components_config.yaml,
# runs Phases 0→4 sequentially, then assembles the unified dataset,
# validates it, and optionally kicks off YOLO training.
from __future__ import annotations

import sys
import time
from pathlib import Path

from config import load_config, PipelineConfig, ComponentConfig
from generate_backgrounds import generate_backgrounds
from phase1_extractor import generate_sprite_variations
from phase2_3_fusion_labeler import generate_synthetic_dataset
from phase4_assembler import (
    assemble_dataset,
    create_yolo_structure,
    generate_yolo_yaml,
    inject_negatives,
    print_dataset_summary,
    split_and_copy,
    _collect_all_component_images,
)
from validate_dataset import validate_dataset


def _process_component(cfg: PipelineConfig, comp: ComponentConfig) -> None:
    """Run Phase 1 + Phase 2/3 for a single component.

    If the component has multiple DXF variants (e.g. IEC + ANSI symbols),
    each variant is processed independently — generating *images_to_generate*
    images per variant, all with the same *class_id*.
    """

    n_variants = cfg.component_variant_count(comp)

    for vi, dxf_path in enumerate(comp.dxf_paths):
        variant_label = f" (variante {vi + 1}/{n_variants}: {dxf_path.stem})" if n_variants > 1 else ""
        sprites_dir = cfg.component_sprites_dir(comp, vi)
        synthetic_dir = cfg.component_synthetic_dir(comp, vi)

        # ── Phase 1: Sprite extraction ────────────────────────────────────
        print(f"  ▶ FASE 1 — Sprites de '{comp.name}'{variant_label}")
        t0 = time.time()
        generate_sprite_variations(
            dxf_path=dxf_path,
            output_dir=sprites_dir,
            n_variations=comp.sprite_variations,
            kernel_min=comp.line_thickness_range[0],
            kernel_max=comp.line_thickness_range[1],
            dpi=cfg.g.render_dpi,
            component_name=comp.name,
        )
        print(f"    ⏱  {time.time() - t0:.1f}s\n")

        # ── Phase 2/3: Synthetic image generation ─────────────────────────
        print(f"  ▶ FASES 2/3 — Generación sintética de '{comp.name}'{variant_label}")
        t0 = time.time()
        generate_synthetic_dataset(
            sprites_dir=sprites_dir,
            bg_dir=cfg.g.backgrounds_dir,
            output_dir=synthetic_dir,
            n_total=comp.images_to_generate,
            component_name=comp.name,
            class_id=comp.class_id,
            sprite_scale_min=cfg.g.sprite_scale_min,
            sprite_scale_max=cfg.g.sprite_scale_max,
            components_min=cfg.g.components_per_img_min,
            components_max=cfg.g.components_per_img_max,
            allow_rotation=cfg.g.allow_random_rotation,
            modifiers_dir=cfg.modifiers.dir,
            modifier_probability=cfg.modifiers.probability,
            modifier_count_min=cfg.modifiers.count_min,
            modifier_count_max=cfg.modifiers.count_max,
            modifier_allow_rotation=cfg.modifiers.allow_rotation,
            modifier_thickness_min=cfg.modifiers.thickness_dilation[0],
            modifier_thickness_max=cfg.modifiers.thickness_dilation[1],
        )
        print(f"    ⏱  {time.time() - t0:.1f}s\n")


def run_pipeline(auto_train: bool = True) -> None:
    """Execute the full LIARD pipeline end-to-end.

    Parameters
    ----------
    auto_train : bool
        If *True* (default), automatically invoke YOLO training after
        validation passes.  Set to *False* for data-generation-only runs.
    """

    cfg = load_config()

    print("\n" + "═" * 60)
    print("  LIARD — Synthetic Data Generation Pipeline")
    print("═" * 60)
    print(f"  Componentes: {len(cfg.components)}")
    for c in cfg.components:
        n_v = len(c.dxf_paths)
        total_imgs = c.images_to_generate * n_v
        variant_info = f"  ({n_v} variantes × {c.images_to_generate} = {total_imgs})" if n_v > 1 else ""
        print(f"    [{c.class_id}] {c.name}  →  {total_imgs} imgs{variant_info}")
    if cfg.backgrounds.enabled:
        print(f"  Backgrounds: auto-generados desde '{cfg.backgrounds.dxf_sources_dir}'")
    print()

    t_global = time.time()

    # ── Validate component DXFs ───────────────────────────────────────────
    for comp in cfg.components:
        for dxf_path in comp.dxf_paths:
            if not dxf_path.exists():
                raise FileNotFoundError(
                    f"No se encontró el archivo DXF: {dxf_path}\n"
                    f"Componente: {comp.name}"
                )

    # ── Phase 0: Background generation from DXF floor plans ──────────────
    if cfg.backgrounds.enabled:
        print(f"\n{'─' * 60}")
        print("  ▶ FASE 0 — Generación de Backgrounds desde planos DXF")
        print(f"{'─' * 60}\n")

        t0 = time.time()
        n_tiles = generate_backgrounds()
        print(f"    ⏱  {time.time() - t0:.1f}s\n")

        if n_tiles == 0:
            print("  ⚠️  No se generaron backgrounds. Verificá que haya .dxf en:")
            print(f"      {cfg.backgrounds.dxf_sources_dir}")

    # ── Validate backgrounds exist ────────────────────────────────────────
    bg_dir = cfg.g.backgrounds_dir
    has_backgrounds = bg_dir.exists() and any(bg_dir.iterdir())
    if not has_backgrounds:
        raise FileNotFoundError(
            f"La carpeta de fondos está vacía: {bg_dir}\n"
            "Opciones:\n"
            "  1. Poné imágenes de planos reales en esa carpeta, o\n"
            "  2. Configurá backgrounds.enabled: true y poné .dxf en\n"
            f"     {cfg.backgrounds.dxf_sources_dir}"
        )

    # ── Per-component generation (Phase 1 + 2/3) ─────────────────────────
    for comp in cfg.components:
        print(f"\n{'─' * 60}")
        print(f"  COMPONENTE: {comp.name}  (class_id={comp.class_id})")
        print(f"{'─' * 60}\n")
        _process_component(cfg, comp)

    # ── Phase 4: Assemble unified dataset ─────────────────────────────────
    print(f"\n{'─' * 60}")
    print("  ▶ FASE 4 — Ensamblaje del Dataset (Split 80/10/10)")
    print(f"{'─' * 60}\n")

    t0 = time.time()

    import shutil
    if cfg.g.dataset_dir.exists():
        shutil.rmtree(cfg.g.dataset_dir)

    dirs = create_yolo_structure(cfg.g.dataset_dir)
    pairs = _collect_all_component_images(cfg)

    if not pairs:
        raise RuntimeError("No se generaron imágenes sintéticas. Revisá los DXF y fondos.")

    split_and_copy(
        pairs, dirs,
        train_ratio=cfg.g.train_ratio,
        val_ratio=cfg.g.val_ratio,
    )
    inject_negatives(dirs, cfg.g.backgrounds_dir, cfg.g.negative_ratio)
    yaml_path = generate_yolo_yaml(cfg.g.dataset_dir, cfg)
    print_dataset_summary(cfg.g.dataset_dir)
    print(f"  ⏱  {time.time() - t0:.1f}s\n")

    # ── Phase 4b: Validation (sanity checks + bbox clipping) ─────────────
    print(f"\n{'─' * 60}")
    print("  ▶ VALIDACIÓN — Sanity Checks + Clipping de Bounding Boxes")
    print(f"{'─' * 60}\n")

    t0 = time.time()
    val_stats = validate_dataset(
        cfg.g.dataset_dir,
        max_missing_labels_pct=cfg.g.max_missing_labels_pct,
    )
    print(f"  ⏱  {time.time() - t0:.1f}s\n")

    if val_stats["images_deleted"] > 0:
        print(f"  ⚠️  Se eliminaron {val_stats['images_deleted']} imágenes con labels irrecuperables.")

    # ── Summary ───────────────────────────────────────────────────────────
    elapsed = time.time() - t_global
    print("═" * 60)
    print(f"  ✅ Pipeline completado en {elapsed:.1f}s")
    print(f"  📁 Dataset en:   {cfg.g.dataset_dir.resolve()}")
    print(f"  📄 Config YOLO:  {yaml_path.resolve()}")

    # ── Phase 5: Training ─────────────────────────────────────────────────
    if auto_train:
        print()
        print(f"  🚀 Iniciando entrenamiento...")
        print(f"     Modelo: {cfg.g.yolo_model}")
        print(f"     Batch:  {cfg.g.batch_size}  |  Workers: {cfg.g.workers}")
        print("═" * 60 + "\n")

        from train import train_model
        train_model(data_yaml=yaml_path)
    else:
        print()
        print("  🚀 Para entrenar manualmente:")
        print(f"     python train.py")
        print("═" * 60 + "\n")


if __name__ == "__main__":
    # Pass --no-train to skip automatic training
    auto = "--no-train" not in sys.argv
    run_pipeline(auto_train=auto)
