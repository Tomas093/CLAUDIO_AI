import os
import glob
import json
from collections import Counter
import cv2
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

from dxf_to_image import renderizar_dxf
from inference_sahi import ejecutar, dibujar

def run():
    dxf_path = "dxf/plano3.dxf"
    out_dir = "output_plano3"
    render_path = os.path.join(out_dir, "plano3_render.png")
    meta_path = os.path.join(out_dir, "plano3_render.json")
    
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "alta_confianza"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "baja_confianza"), exist_ok=True)
    
    print(f"Renderizando {dxf_path}...")
    if not os.path.exists(render_path):
        renderizar_dxf(dxf_path, render_path)
    
    models = glob.glob("best_models/*.pt")
    all_detections = []
    
    # Abrir la imagen base para recortes
    base_img = cv2.imread(render_path)
    
    for model_path in models:
        model_name = os.path.basename(model_path).replace(".pt", "")
        print(f"\n--- Corriendo modelo {model_name} ---")
        
        # Corremos con conf_min=0.7 para capturar tambien los de baja confianza (70-80%)
        conteo, detecciones = ejecutar(
            modelo_path=model_path,
            image_path=render_path,
            meta_path=meta_path,
            output_dir=os.path.join(out_dir, f"temp_{model_name}"),
            conf_min=0.70,
            iou_global=0.5,
            device="cpu"
        )
        
        # Filtramos y guardamos recortes
        img_visual_path = os.path.join(out_dir, f"plano3_render_visual_{model_name}.png")
        # Vamos a dibujar solo las detecciones sobre esta imagen
        dibujar(render_path, detecciones, img_visual_path)
        
        for i, d in enumerate(detecciones):
            d["modelo"] = model_name
            all_detections.append(d)
            
            x1, y1, x2, y2 = map(int, d["bbox_px"])
            # Agregar margen al recorte
            pad = 10
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(base_img.shape[1], x2 + pad)
            y2 = min(base_img.shape[0], y2 + pad)
            
            crop = base_img[y1:y2, x1:x2]
            
            conf = d["conf"]
            clase = d["clase"].replace("/", "_").replace(" ", "_")
            
            filename = f"{model_name}_{clase}_{conf:.3f}_{i}.png"
            
            if conf >= 0.80:
                cv2.imwrite(os.path.join(out_dir, "alta_confianza", filename), crop)
            else:
                cv2.imwrite(os.path.join(out_dir, "baja_confianza", filename), crop)
                
    # Generar reporte
    print("\n\n" + "="*50)
    print("REPORTE GLOBAL DE DETECCIONES")
    print("="*50)
    
    clase_conteo = Counter(d["clase"] for d in all_detections)
    
    for clase, cant in clase_conteo.most_common():
        confs = [d["conf"] for d in all_detections if d["clase"] == clase]
        avg_conf = sum(confs) / len(confs) if confs else 0
        print(f"Clase: {clase:<30} | Detectados: {cant:>3} | Confianza Promedio: {avg_conf:.3f}")

if __name__ == "__main__":
    run()
