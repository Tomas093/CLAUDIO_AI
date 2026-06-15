# 🔧 LIARD — Guía de Uso del Pipeline de Entrenamiento

Pipeline completo para generar datasets sintéticos de componentes eléctricos (desde DXF), ingestar datos reales de Roboflow, y entrenar modelos YOLOv8/YOLO11 de **Clase Única en Dos Fases** (Single-Class Two-Phase) para detección automática en planos.

---

## 📁 Estructura del Proyecto

```
train-maker/
├── components_config.yaml    ← ⭐ ARCHIVO PRINCIPAL DE CONFIGURACIÓN
├── config.py                 ← Motor de configuración (no tocar)
├── generate_backgrounds.py   ← Fase 0: Genera fondos desde planos DXF
├── phase1_extractor.py       ← Fase 1: Extrae sprites desde DXF de componentes
├── phase2_3_fusion_labeler.py← Fases 2/3: Fusiona sprites + fondo + auto-etiqueta
├── phase4_assembler.py       ← Fase 4: Ensambla dataset sintético
├── validate_dataset.py       ← Validación: Clipping de bboxes + safety checks
├── manual_ingestor.py        ← Ingestión Automática de Zips de Roboflow
├── train.py                  ← Fase 5: Entrenamiento YOLO (CLI interno)
├── run_pipeline.py           ← ⭐ CONTROLADOR MAESTRO (ejecuta todo)
├── input/
│   ├── component.dxf         ← Tu DXF del componente eléctrico
│   ├── roboflow_export.zip   ← (Opcional) ZIP de Roboflow para Fine-Tuning
│   ├── modifiers/            ← PNGs de polos (//, ///) y símbolos extra
│   └── planos_completos/     ← Planos DXF completos (para generar fondos)
├── output/                   ← Sprites y carpetas temporales generadas
├── models/                   ← Modelos finales generados (best_*.pt)
└── yolov8m.pt                ← Modelo pre-entrenado base
```

> **Nota sobre el Workspace de YOLO**: Todos los pesos, tensores y métricas generadas por Ultralytics se guardan fuera de `train-maker` para no ensuciar el repositorio de código. Suelen guardarse en `../yolo_workspace`.

---

## 🚀 Inicio Rápido (4 pasos)

### Paso 1: Preparar los archivos de entrada (Sintéticos)

```bash
# 1. Poné el DXF de tu componente eléctrico en input/components/
cp mi_interruptor.dxf train-maker/input/components/interruptor.dxf

# 2. Poné tus planos completos (para generar fondos) en input/planos_completos/
cp plano_tablero_*.dxf train-maker/input/planos_completos/
```

### Paso 2 (Opcional): Preparar datos Reales (Fine-Tuning)

Si tenés datos reales etiquetados en Roboflow, descargá el export en formato "YOLOv8" (como un archivo ZIP) y dejalo en tu proyecto.

```bash
cp roboflow_export.zip train-maker/input/roboflow_interruptor.zip
```

### Paso 3: Configurar `components_config.yaml`

Editá el archivo para definir tu componente y asociarlo a sus datos:

```yaml
components:
  - name: "interruptor_termomagnetico"
    dxf_path: 
      - "input/components/interruptor.dxf"
    images_to_generate: 5000     # Cuántas imágenes sintéticas generar
    sprite_variations: 90        # Variaciones de grosor de línea
    line_thickness_range: [10, 50]
    polarity_filters: ["invert", "threshold"]
    
    # ¡Clave para la Fase 2! Ruta al ZIP descargado de Roboflow
    roboflow_zip_path: "input/roboflow_interruptor.zip"
```

### Paso 4: Ejecutar el pipeline

```bash
cd train-maker/
python run_pipeline.py
```

El pipeline procesará cada componente de forma secuencial y generará tu modelo final en `models/best_interruptor_termomagnetico.pt`.

---

## 📋 Flujo de Entrenamiento (Dos Fases / Single-Class)

El pipeline actual opera de manera **secuencial por componente** bajo un enfoque de **Clase Única (Single-Class)**. Para cada componente en la lista, se ejecuta:

```
┌─────────────────────────────────────────────────────┐
│  Fases 1 a 3 — Generación Sintética                │
│  Extrae sprites del DXF → Fusiona con planos de    │
│  fondo → Genera ~5000 imágenes sintéticas.         │
└──────────────────────┬──────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────┐
│  Fase 4 — Ensamblaje Sintético                     │
│  Empaqueta los datos en dataset_sintetico_<nombre> │
│  y fuerza la clase a '0' (Single-Class).           │
└──────────────────────┬──────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────┐
│  Entrenamiento Fase 1 (Sintético)                  │
│  Ejecuta train.py como subproceso. Entrena YOLOv8  │
│  desde cero usando SOLO imágenes sintéticas.       │
│  lr0 alto. Genera: phase1_<nombre>/weights/best.pt │
└──────────────────────┬──────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────┐
│  Ingestión Manual (Roboflow)                       │
│  (Si roboflow_zip_path está definido).             │
│  Extrae el ZIP → unifica splits → mapea la clase   │
│  a '0' → genera dataset_real_<nombre>.             │
└──────────────────────┬──────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────┐
│  Entrenamiento Fase 2 (Fine-Tuning)                │
│  Carga el best.pt de la Fase 1. Congela el backbone│
│  (freeze=10). Entrena con los datos reales usando  │
│  un lr0 muy bajo (lr0_finetune).                   │
└──────────────────────┬──────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────┐
│  Limpieza y Consolidación                          │
│  Si todo fue exitoso, borra datasets temporales,   │
│  libera VRAM y copia el peso final a la carpeta    │
│  /models para su uso en producción.                │
└─────────────────────────────────────────────────────┘
```

> **Aislamiento de Memoria (VRAM):** Los entrenamientos de PyTorch se ejecutan en subprocesos independientes (`subprocess.run`). Esto garantiza que al terminar un modelo, el SO recupere el 100% de la VRAM y RAM, impidiendo los errores de *Out Of Memory (OOM)* cuando se entrenan decenas de componentes seguidos.

---

## ⚙️ Configuración Detallada (`components_config.yaml`)

### Configuración Global (Finetuning)

```yaml
global:
  # Entrenamiento Sintético (Fase 1)
  yolo_model: "yolov8m.pt"
  epochs: 100
  lr0: 0.01          # Tasa de aprendizaje normal

  # Fine-tuning (Fase 2: datos reales sobre sintéticos)
  epochs_finetune: 100
  lr0_finetune: 0.001  # Tasa MUY BAJA para no destruir pesos previos
  lrf_finetune: 0.001

  # Output workspace para tensores (Afuera de train-maker)
  yolo_workspace: "yolo_workspace"
```

### Ingestión de ZIPs de Roboflow

El `manual_ingestor.py` asume que descargaste un dataset YOLO de Roboflow. El motor buscará carpetas que contengan `train`, `val`, `valid` o `test` en la ruta, y ubicará automáticamente las imágenes y etiquetas.
Además, reescribirá el primer número de cada archivo `.txt` a un `0` estricto. Esto permite descargar un dataset multi-clase de Roboflow y usar solo los recortes que te interesan, o asegurar la consistencia.

---

## 🎯 Resumen de Protecciones de esta Arquitectura

| Protección | Qué hace |
|---|---|
| **Single-Class** | Cada componente se entrena como la "clase 0", lo que facilita enormemente el fine-tuning y evita desbalances de clases en la matriz de confusión. |
| **Backbone Congelado** | En la Fase 2, se aplica `freeze=10` para que el modelo no desaprenda la geometría sólida que asimiló viendo miles de imágenes sintéticas. |
| **VRAM Subprocessing** | Entrenar 15 clases implicaría 30 ciclos de YOLO. El uso de `subprocess` destruye el árbol de tensores huérfanos entre cada modelo, previniendo cuelgues del sistema. |
| **Try/Except Cleanup** | Las imágenes sintéticas (pesan GBs) se borran solas al finalizar el entrenamiento con éxito. Si el entrenamiento se cae, las carpetas se mantienen intactas para poder debuggear qué archivo rompió la corrida. |
| **Logging Nativo** | El pipeline imprime y guarda en `pipeline_run_*.log` de manera limpia, sin requerir librerías gráficas extrañas. |
