import sys
import subprocess
from pathlib import Path
from manual_ingestor import ingest_roboflow_zip
import config

def main():
    component_name = "seccionador_bajo_carga"
    zip_path = Path("../zips_datasets/seccionador_bajo_carga.yolov11.zip")
    
    print(f"[{component_name}] [INGESTA] Procesando ZIP de Roboflow...")
    real_yaml = ingest_roboflow_zip(zip_path, component_name)
    
    cfg = config.load_config()
    best_synth = cfg.g.yolo_workspace / f"phase1_{component_name}" / "weights" / "best.pt"
    
    if not best_synth.exists():
        print(f"ERROR: No se encontraron los pesos de la Fase 1 en {best_synth}")
        return
        
    print(f"[{component_name}] [FASE 2 ENTRENAMIENTO] Lanzando subprocess...")
    cmd_p2 = [
        sys.executable, "train.py", 
        "--component", component_name, 
        "--phase", "2", 
        "--data-yaml", str(real_yaml), 
        "--base-weights", str(best_synth)
    ]
    
    res_p2 = subprocess.run(cmd_p2)
    if res_p2.returncode != 0:
        print(f"Fallo en Fase 2 (fine-tune) para {component_name}")
    else:
        print("¡Fase 2 finalizada con exito!")

if __name__ == '__main__':
    main()
