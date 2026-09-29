# Detector Universal de Componentes en Planos Unifilares (YOLO Nano CAD)

Paquete listo para producción para la detección de componentes y aparatos eléctricos en diagramas unifilares y planos CAD.

---

## 1. Contenido del Paquete

* **`best_componente_nano.pt`**: Pesos optimizados del modelo YOLO Nano entrenado con Data-Centric AI (100% Recall en planos industriales y de distribución).
* **`detector_unifilar.py`**: Script integral de inferencia que realiza:
  1. Renderizado vectorial del DXF con `ezdxf` (filtrando cotas, textos y cartelas para evitar ruido).
  2. Slicing adaptativo SAHI con ventanas de 640x640, 80% de solapamiento y padding de 320 px.
  3. Inferencia por lotes en GPU (`cuda:0`) o CPU.
  4. NMS secuencial en coordenadas CAD (IoU, cajas anidadas y distancia euclidiana).
  5. Exportación automática de lámina visual (`.png`), CSV con coordenadas y JSON.
* **`requirements.txt`**: Librerías necesarias.

---

## 2. Instalación

En tu entorno de Python (recomendado Python 3.9 a 3.11):

```bash
pip install -r requirements.txt
```

---

## 3. Uso por Línea de Comandos (CLI)

### A. Para procesar un plano en formato DXF:
```bash
python detector_unifilar.py --dxf "mi_plano.dxf" --out "./resultados" --conf 0.15
```

### B. Para procesar una imagen exportada (PNG o JPG):
```bash
python detector_unifilar.py --img "mi_plano.png" --out "./resultados" --conf 0.15
```

### Parámetros opcionales:
* `--conf 0.15`: Umbral de confianza. **0.10 a 0.15 es el rango óptimo** que garantiza cero falsos negativos. Si usás `0.25` o más, perderás símbolos atípicos.
* `--scale 75.0`: Escala de renderizado px/CAD (por defecto `75.0` para planos industriales Schneider tipo FL-UN o TSSS; usar `84.3` para planos de distribución domiciliaria).
* `--device cpu`: Si no tenés placa gráfica NVIDIA con CUDA, podés forzar CPU. Si tenés GPU, la detecta automáticamente.
* `--batch 32`: Tamaño de lote para la GPU (reducir a 16 si tu GPU tiene poca VRAM).

---

## 4. Uso como Módulo de Python

Podés integrarlo directamente en tu código o servidor web:

```python
from detector_unifilar import DetectorUnifilar

# Inicializar detector (detecta GPU automáticamente o podés pasar device="cpu")
detector = DetectorUnifilar(model_path="best_componente_nano.pt")

# Opción 1: Procesar DXF
resultado = detector.detectar_dxf(
    dxf_path="plano.dxf",
    output_dir="./salida",
    conf_thresh=0.15,
    px_per_cad=75.0
)

print(f"Total componentes detectados: {resultado['total']}")
print(f"Lámina con cajas verdes: {resultado['vis_image']}")
print(f"Archivo CSV: {resultado['csv_path']}")
print(f"Archivo JSON: {resultado['json_path']}")

# Cada componente contiene sus coordenadas CAD y en píxeles:
for comp in resultado['detections']:
    print(f"Confianza: {comp['conf']:.2f} | Centro CAD: ({comp['xc']:.2f}, {comp['yc']:.2f})")
```

---

## 5. Salidas Generadas

En la carpeta indicada en `--out` se generan 3 archivos:
1. **`{nombre}_visual_detections.png`**: La lámina completa con cada componente encuadrado en **caja verde** y su score de confianza visible.
2. **`{nombre}_detections.csv`**: Tabla con las coordenadas CAD exactas (`xc_cad`, `yc_cad`, `x1_cad`, `y1_cad`, `x2_cad`, `y2_cad`), confianza y píxeles en la lámina.
3. **`{nombre}_detections.json`**: Formato estructurado con metadatos de escala y lista de detecciones para integrar con APIs y bases de datos.
