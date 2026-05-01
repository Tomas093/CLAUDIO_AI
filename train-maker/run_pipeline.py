import time

from config import BACKGROUNDS_DIR, DATASET_DIR, DXF_FILE, N_SPRITE_VARIATIONS, N_SYNTHETIC_TOTAL, SPRITES_DIR, SYNTHETIC_DIR, TRAIN_RATIO, NEGATIVE_RATIO
from phase1_extractor import generate_sprite_variations
from phase2_3_fusion_labeler import generate_synthetic_dataset
from phase4_assembler import assemble_dataset


def run_pipeline() -> None:
    print("\n" + "═" * 60)
    print("  LIARD — Synthetic Data Generation Pipeline")
    print("═" * 60 + "\n")

    t_global = time.time()

    if not DXF_FILE.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo DXF: {DXF_FILE}\n"
            "Coloca tu archivo .dxf en la carpeta 'input/'."
        )
    if not BACKGROUNDS_DIR.exists() or not any(BACKGROUNDS_DIR.iterdir()):
        raise FileNotFoundError(
            f"La carpeta de fondos está vacía: {BACKGROUNDS_DIR}\n"
            "Agrega imágenes de planos reales (sin componentes)."
        )

    print("▶ FASE 1 — Extracción y Variación de Sprites")
    t0 = time.time()
    generate_sprite_variations(
        dxf_path=DXF_FILE,
        output_dir=SPRITES_DIR,
        n_variations=N_SPRITE_VARIATIONS,
    )
    print(f"  ⏱  {time.time() - t0:.1f}s\n")

    print("▶ FASES 2/3 — Fusión con Ruido + Auto-Etiquetado YOLO")
    t0 = time.time()
    generate_synthetic_dataset(
        sprites_dir=SPRITES_DIR,
        bg_dir=BACKGROUNDS_DIR,
        output_dir=SYNTHETIC_DIR,
        n_total=N_SYNTHETIC_TOTAL,
    )
    print(f"  ⏱  {time.time() - t0:.1f}s\n")

    print("▶ FASE 4 — Ensamblaje del Dataset (Split + Negative Mining)")
    t0 = time.time()
    yaml_path = assemble_dataset(
        synthetic_dir=SYNTHETIC_DIR,
        bg_dir=BACKGROUNDS_DIR,
        dataset_dir=DATASET_DIR,
        train_ratio=TRAIN_RATIO,
        negative_ratio=NEGATIVE_RATIO,
    )
    print(f"  ⏱  {time.time() - t0:.1f}s\n")

    elapsed = time.time() - t_global
    print("═" * 60)
    print(f"  ✅ Pipeline completado en {elapsed:.1f}s")
    print(f"  📁 Dataset en:   {DATASET_DIR.resolve()}")
    print(f"  📄 Config YOLO:  {yaml_path.resolve()}")
    print()
    print("  🚀 Para entrenar:")
    print(f"     yolo train data={yaml_path} model=yolov8n.pt epochs=100 imgsz=640")
    print("═" * 60 + "\n")


if __name__ == "__main__":
    run_pipeline()

