import os
import json
import csv
from collections import Counter
from vector_inference import ejecutar_vectorial

def run_test():
    dxf_path = r"test_2.dxf"
    models = [
        r"train-maker\models\best_interruptor_termomagnetico.pt",
        r"train-maker\models\best_interruptor_diferencial.pt",
        r"train-maker\models\best_interruptor_motorizado.pt",
        r"train-maker\models\best_interruptor_temporizado.pt",
        r"train-maker\models\best_seccionador_bajo_carga.pt",
    ]
    
    all_detections = []
    for model in models:
        comp_name = os.path.basename(model).replace('best_', '').replace('.pt', '')
        print(f"\n--- Probando modelo {comp_name} ---")
        try:
            conteo, detecciones = ejecutar_vectorial(
                dxf_path=dxf_path,
                modelo_path=model,
                output_dir=f"./pipeline_out_{comp_name}",
                conf=0.25,
                conf_min=0.5,
                device="0"
            )
            # The detections will have class names from the model, usually 0 is the component
            for d in detecciones:
                d['clase_original'] = d['clase']
                d['clase'] = comp_name
            all_detections.extend(detecciones)
        except Exception as e:
            print(f"Error corriendo {comp_name}: {e}")
            
    print("\n\n" + "="*50)
    print("CONTEO FINAL ACUMULADO POR LA IA:")
    conteo_ia = Counter(d["clase"] for d in all_detections)
    for c, n in conteo_ia.most_common():
        print(f"  {c}: {n}")
        
    print("\n" + "="*50)
    print("CONTEO MANUAL (VERDAD TERRENO):")
    csv_path = r"test\test_2\verdad_terreno\test_2_inserts.csv"
    if os.path.exists(csv_path):
        conteo_manual = Counter()
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                tipo = row.get('tipo', row.get('TIPO', ''))
                conteo_manual[tipo] += 1
        for c, n in conteo_manual.most_common():
            print(f"  {c}: {n}")
    else:
        print(f"No se encontró el CSV en {csv_path}")

if __name__ == '__main__':
    run_test()
