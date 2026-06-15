# run_pipeline.py — Phase 5: Master Controller
# Iterates through every component defined in components_config.yaml,
# runs Phase 1 & 2/3 sequentially, assembles the synthetic dataset,
# trains Phase 1 (sintético), ingests real data (if any), and trains Phase 2.
from __future__ import annotations

import gc
import logging
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# Fix Windows encoding
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from config import load_config, PipelineConfig, ComponentConfig, BASE_DIR
from generate_backgrounds import generate_backgrounds
from phase1_extractor import generate_sprite_variations
from phase2_3_fusion_labeler import generate_synthetic_dataset
from phase4_assembler import assemble_synthetic
from manual_ingestor import ingest_roboflow_zip


def setup_logger() -> logging.Logger:
    """Configura el logger nativo de Python para archivo y consola."""
    logger = logging.getLogger("LiardPipeline")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    
    # Limpiar handlers si ya existen
    if logger.handlers:
        logger.handlers.clear()
        
    formatter = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = BASE_DIR / f"pipeline_run_{timestamp}.log"
    
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    return logger

log = setup_logger()


def _process_component(cfg: PipelineConfig, comp: ComponentConfig) -> None:
    """Run Phase 1 + Phase 2/3 for a single component."""
    n_variants = cfg.component_variant_count(comp)

    for vi, dxf_path in enumerate(comp.dxf_paths):
        variant_label = f" (variante {vi + 1}/{n_variants}: {dxf_path.stem})" if n_variants > 1 else ""
        sprites_dir = cfg.component_sprites_dir(comp, vi)
        synthetic_dir = cfg.component_synthetic_dir(comp, vi)

        # ── Phase 1: Sprite extraction ────────────────────────────────────
        log.info(f"[{comp.name}] [FASE 1] Sprites{variant_label}...")
        generate_sprite_variations(
            dxf_path=dxf_path,
            output_dir=sprites_dir,
            n_variations=comp.sprite_variations,
            kernel_min=comp.line_thickness_range[0],
            kernel_max=comp.line_thickness_range[1],
            dpi=cfg.g.render_dpi,
            component_name=comp.name,
        )

        # ── Phase 2/3: Synthetic image generation ─────────────────────────
        log.info(f"[{comp.name}] [FASES 2/3] Generación sintética{variant_label}...")
        generate_synthetic_dataset(
            sprites_dir=sprites_dir,
            bg_dir=cfg.g.backgrounds_dir,
            output_dir=synthetic_dir,
            n_total=comp.images_to_generate,
            component_name=comp.name,
            class_id=0, # Single-class
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
            require_full_visibility=cfg.g.require_full_visibility,
        )


def _cleanup_component_dirs(component_name: str, cfg: PipelineConfig) -> None:
    """Borra carpetas de datos tras un entrenamiento exitoso."""
    def _get_size(path: Path) -> float:
        if not path.exists(): return 0
        return sum(f.stat().st_size for f in path.rglob('*') if f.is_file()) / (1024 * 1024)

    dirs_to_clean = [
        BASE_DIR / f"dataset_sintetico_{component_name}",
        BASE_DIR / f"dataset_real_{component_name}"
    ]
    
    # Agregar carpetas de variantes (sprites y synthetic)
    for comp in cfg.components:
        if comp.name == component_name:
            for vi in range(cfg.component_variant_count(comp)):
                dirs_to_clean.append(cfg.component_sprites_dir(comp, vi))
                dirs_to_clean.append(cfg.component_synthetic_dir(comp, vi))

    for d in dirs_to_clean:
        if d.exists():
            mb_size = _get_size(d)
            shutil.rmtree(d, ignore_errors=True)
            log.info(f"[{component_name}] [CLEANUP] Borrada {d.name} ({mb_size:.1f} MB)")


def run_pipeline() -> None:
    cfg = load_config()

    log.info("=" * 60)
    log.info("  LIARD — Pipeline de Generación y Entrenamiento")
    log.info("=" * 60)
    log.info(f"Componentes: {len(cfg.components)}")
    
    # Validación DXFs
    for comp in cfg.components:
        for dxf_path in comp.dxf_paths:
            if not dxf_path.exists():
                raise FileNotFoundError(f"DXF faltante: {dxf_path}")

    # Phase 0: Backgrounds
    if cfg.backgrounds.enabled:
        log.info("[GLOBAL] [FASE 0] Generando backgrounds...")
        generate_backgrounds()

    bg_dir = cfg.g.backgrounds_dir
    if not (bg_dir.exists() and any(bg_dir.iterdir())):
        raise FileNotFoundError(f"Carpeta de fondos vacía: {bg_dir}")

    # Verificación preliminar
    try:
        from verification_renderer import run_verification
        run_verification(cfg)
    except Exception as exc:
        log.warning(f"[GLOBAL] [VERIFICACIÓN] Falló visualización: {exc}")

    results = []

    for comp in cfg.components:
        pass # Skip condition removed
        log.info("-" * 60)
        log.info(f"[{comp.name}] INICIANDO PROCESO")
        log.info("-" * 60)
        
        success = False
        t_start = time.time()
        
        try:
            # 1. Generación de datos
            _process_component(cfg, comp)
            
            # 2. Ensamblar YAML sintético
            if getattr(comp, 'skip_training', False):
                log.info(f"[{comp.name}] [SKIP] Skip training activado. Omitiendo fases 4 y entrenamiento.")
                continue

            log.info(f"[{comp.name}] [FASE 4] Ensamblando dataset sintético...")
            synth_yaml = assemble_synthetic(comp.name, cfg)
            
            # 3. Entrenar Fase 1 (Sintética) vía subprocess para aislar VRAM
            log.info(f"[{comp.name}] [FASE 1 ENTRENAMIENTO] Lanzando subprocess...")
            cmd_p1 = [sys.executable, "train.py", "--component", comp.name, "--phase", "1", "--data-yaml", str(synth_yaml)]
            res_p1 = subprocess.run(cmd_p1)
            if res_p1.returncode != 0:
                raise RuntimeError(f"Fallo en Fase 1 (sintética) para {comp.name}")
                
            best_synth = cfg.g.yolo_workspace / f"phase1_{comp.name}" / "weights" / "best.pt"
            
            # 4. Entrenar Fase 2 (Fine-tune real) si corresponde
            if comp.roboflow_zip_path and comp.roboflow_zip_path.exists():
                log.info(f"[{comp.name}] [INGESTA] Procesando ZIP de Roboflow...")
                real_yaml = ingest_roboflow_zip(comp.roboflow_zip_path, comp.name)
                
                log.info(f"[{comp.name}] [FASE 2 ENTRENAMIENTO] Fine-tuning vía subprocess...")
                cmd_p2 = [sys.executable, "train.py", "--component", comp.name, "--phase", "2", 
                          "--data-yaml", str(real_yaml), "--base-weights", str(best_synth)]
                res_p2 = subprocess.run(cmd_p2)
                if res_p2.returncode != 0:
                    raise RuntimeError(f"Fallo en Fase 2 (fine-tune) para {comp.name}")
            else:
                log.info(f"[{comp.name}] [FASE 2] Sin datos reales (roboflow_zip_path no definido/no encontrado), omitiendo Fase 2.")
            
            success = True
            
        except Exception as e:
            log.error(f"[{comp.name}] ERROR: {e} — Conservando artifacts para debug.")
            
        finally:
            gc.collect()
            if success:
                _cleanup_component_dirs(comp.name, cfg)
                
            t_elapsed = time.time() - t_start
            results.append((comp.name, "EXITO" if success else "FALLO", t_elapsed))

    # Resumen Final
    log.info("\n" + "=" * 60)
    log.info("  RESUMEN FINAL")
    log.info("=" * 60)
    
    # Encabezado
    log.info(f"{'COMPONENTE'.ljust(35)} | {'ESTADO'.ljust(10)} | {'TIEMPO'.rjust(8)}")
    log.info("-" * 60)
    for nombre, estado, t_elap in results:
        t_min = f"{t_elap / 60:.1f}m"
        log.info(f"{nombre.ljust(35)} | {estado.ljust(10)} | {t_min.rjust(8)}")
        
    log.info("=" * 60)


if __name__ == "__main__":
    run_pipeline()
