import sys
from pathlib import Path
from manual_ingestor import ingest_roboflow_zip
from train import train_finetune
from config import load_config

if __name__ == "__main__":
    comp = "interruptor_diferencial"
    zip_path = r"c:\Users\Tomas\Documents\LAB3\CLAUDIO_AI\interruptor_diferencial.yolov11.zip"
    
    print(f"Ingesting ZIP: {zip_path}")
    yaml_path = ingest_roboflow_zip(zip_path, comp)
    
    cfg = load_config()
    base_weights = Path(cfg.g.yolo_workspace) / f"phase1_{comp}" / "weights" / "best.pt"
    
    print(f"Starting Fine-Tuning with base weights: {base_weights}")
    train_finetune(comp, yaml_path, base_weights, cfg)
    print("Fine-tuning completed successfully!")
