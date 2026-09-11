# run_component.py — Lanzador de un componente, fase por fase.
#
# Reemplaza a los cinco scripts one-shot que había antes (run_phase2.py,
# run_phase2_imm.py, run_phase2_sbc.py, run_phase4_sbc.py,
# run_retrain_sbc.py), que eran el mismo código con el nombre del componente
# escrito a mano y algún path hardcodeado (auditoría H21).
#
# Ejemplos:
#   python run_component.py ojo_de_buey --fases 1,2,3,4
#   python run_component.py ojo_de_buey --fases 4,verify     # re-ensamblar
#   python run_component.py ojo_de_buey --fases train1
#   python run_component.py ojo_de_buey --fases ingest,train2
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from config import BASE_DIR, load_config
from manual_ingestor import ingest_roboflow_zip
from phase1_extractor import generate_sprite_variations
from phase2_3_fusion_labeler import generate_synthetic_dataset
from phase4_assembler import assemble_synthetic
from verify_labels import verify_dataset

FASES = ["1", "23", "4", "verify", "train1", "ingest", "train2"]


def main() -> int:
    ap = argparse.ArgumentParser(description="Corre fases sueltas de un componente")
    ap.add_argument("component")
    ap.add_argument("--fases", default="1,23,4,verify,train1,ingest,train2",
                    help=f"Lista separada por comas. Válidas: {','.join(FASES)}")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    fases = [f.strip() for f in args.fases.split(",") if f.strip()]
    desconocidas = set(fases) - set(FASES)
    if desconocidas:
        raise SystemExit(f"Fases desconocidas: {sorted(desconocidas)}. "
                         f"Válidas: {FASES}")

    cfg = load_config()
    comp = cfg.get(args.component)
    name = comp.name
    groups: dict[str, str] = {}
    synth_yaml = BASE_DIR / f"dataset_sintetico_{name}" / f"sintetico_{name}.yaml"

    if "1" in fases:
        for vi, dxf in enumerate(comp.dxf_paths):
            generate_sprite_variations(
                dxf_path=dxf, output_dir=cfg.component_sprites_dir(comp, vi),
                n_variations=comp.sprite_variations,
                kernel_min=comp.line_thickness_range[0],
                kernel_max=comp.line_thickness_range[1],
                component_name=name,
            )

    if "23" in fases:
        import cv2
        negatives = []
        for other in cfg.components:
            for vi in range(cfg.component_variant_count(other)):
                d = cfg.component_sprites_dir(other, vi)
                if not d.exists():
                    continue
                for p in sorted(d.glob("*.png")):
                    img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
                    if img is not None and img.ndim == 3 and img.shape[2] == 4:
                        negatives.append((other.name, img))
        for vi in range(cfg.component_variant_count(comp)):
            stats = generate_synthetic_dataset(
                sprites_dir=cfg.component_sprites_dir(comp, vi),
                bg_dir=cfg.g.backgrounds_dir,
                output_dir=cfg.component_synthetic_dir(comp, vi),
                n_total=comp.images_to_generate,
                component_name=name, cfg_g=cfg.g, class_id=comp.class_id,
                negative_sprites=negatives,
                exclude_as_negative=comp.exclude_as_negative,
                seed=cfg.g.seed,
            )
            groups.update(stats.groups)

    if "4" in fases:
        synth_yaml = assemble_synthetic(name, cfg, groups=groups)

    if "verify" in fases:
        verify_dataset(BASE_DIR / f"dataset_sintetico_{name}",
                       seed=cfg.g.seed, fail_on_error=True,
                       sprites_dir=cfg.component_sprites_dir(comp, 0))

    if "train1" in fases:
        cmd = [sys.executable, "train.py", "--component", name, "--phase", "1",
               "--data-yaml", str(synth_yaml)]
        if args.resume:
            cmd.append("--resume")
        if subprocess.run(cmd, cwd=BASE_DIR).returncode != 0:
            return 1

    real_yaml = BASE_DIR / f"dataset_real_{name}" / f"real_{name}.yaml"
    if "ingest" in fases:
        if not (comp.roboflow_zip_path and comp.roboflow_zip_path.exists()):
            print(f"[{name}] Sin ZIP de datos reales; se omite la ingesta.")
        else:
            real_yaml = ingest_roboflow_zip(comp.roboflow_zip_path, name, cfg)
            verify_dataset(BASE_DIR / f"dataset_real_{name}", n_muestras=18,
                           seed=cfg.g.seed, fail_on_error=False,
                           sprites_dir=cfg.component_sprites_dir(comp, 0))

    if "train2" in fases:
        if not real_yaml.exists():
            print(f"[{name}] No existe {real_yaml}; corré la fase 'ingest' primero.")
            return 1
        base = cfg.g.yolo_workspace / f"phase1_{name}" / "weights" / "best.pt"
        cmd = [sys.executable, "train.py", "--component", name, "--phase", "2",
               "--data-yaml", str(real_yaml), "--base-weights", str(base)]
        if args.resume:
            cmd.append("--resume")
        if subprocess.run(cmd, cwd=BASE_DIR).returncode != 0:
            return 1

    print(f"\n[{name}] Fases completadas: {', '.join(fases)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
