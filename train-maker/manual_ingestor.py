# manual_ingestor.py — Ingestión de datasets manuales de Roboflow
from __future__ import annotations

import os
import shutil
import tempfile
import zipfile
from pathlib import Path

from config import BASE_DIR


def _normalize_and_copy_label(src_txt: Path, dst_txt: Path) -> None:
    """Lee el label YOLO, fuerza la clase 0, ignora líneas rotas y lo copia."""
    if not src_txt.exists():
        dst_txt.touch()
        return

    good_lines: list[str] = []
    for line in src_txt.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
            
        parts = line.split()
        if len(parts) < 5:
            continue
            
        # Forzar single-class (0) y mantener bbox
        parts[0] = "0"
        good_lines.append(" ".join(parts))

    dst_txt.write_text("\n".join(good_lines) + "\n", encoding="utf-8")


def ingest_roboflow_zip(zip_path: str | Path, component_name: str) -> Path:
    """Extrae un ZIP de Roboflow, mapea los splits y normaliza a single-class.
    
    Retorna el path absoluto al data.yaml generado.
    """
    zip_path = Path(zip_path)
    if not zip_path.exists():
        raise FileNotFoundError(f"No se encontró el ZIP de Roboflow: {zip_path}")

    # Carpeta destino final del dataset real
    dataset_dir = BASE_DIR / f"dataset_real_{component_name}"
    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)
        
    for split in ("train", "val", "test"):
        (dataset_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (dataset_dir / split / "labels").mkdir(parents=True, exist_ok=True)

    # Extraer ZIP en carpeta temporal
    temp_dir = Path(tempfile.mkdtemp(prefix=f"roboflow_{component_name}_"))
    
    try:
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
        except zipfile.BadZipFile as e:
            raise RuntimeError(f"ZIP corrupto o inválido: {zip_path} - {e}")
            
        # Recorrer archivos extraídos
        for root, _, files in os.walk(temp_dir):
            root_path = Path(root)
            
            # Determinar split por keyword en la ruta
            path_str = str(root_path).lower()
            if "valid" in path_str or "val" in path_str:
                split = "val"
            elif "test" in path_str:
                split = "test"
            elif "train" in path_str:
                split = "train"
            else:
                split = "train"  # default
                
            dst_imgs = dataset_dir / split / "images"
            dst_lbls = dataset_dir / split / "labels"
            
            for file in files:
                if file.lower().endswith((".jpg", ".jpeg", ".png")):
                    img_src = root_path / file
                    lbl_src = root_path.parent / "labels" / (img_src.stem + ".txt")
                    
                    # Destinos
                    img_dst = dst_imgs / img_src.name
                    lbl_dst = dst_lbls / (img_src.stem + ".txt")
                    
                    # Copiar imagen
                    shutil.copy2(img_src, img_dst)
                    
                    # Normalizar y copiar label
                    _normalize_and_copy_label(lbl_src, lbl_dst)
                    
        # Generar YAML
        yaml_path = dataset_dir / f"real_{component_name}.yaml"
        has_val = any((dataset_dir / "val" / "images").iterdir())
        val_path = "val/images" if has_val else "train/images"
        content = (
            f"# Liard — Dataset Real {component_name} (Roboflow)\n\n"
            f"path: {dataset_dir.resolve()}\n"
            f"train: train/images\n"
            f"val:   {val_path}\n"
            f"test:  test/images\n\n"
            f"nc: 1\n"
            f"names:\n"
            f"  0: {component_name}\n"
        )
        yaml_path.write_text(content, encoding="utf-8")
        
        print(f"[{component_name}] Dataset real procesado: {yaml_path}")
        return yaml_path
        
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
