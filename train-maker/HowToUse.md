# 🔧 LIARD — Guía de Uso del Pipeline de Entrenamiento

Pipeline completo para generar datasets sintéticos de componentes eléctricos (desde DXF) y entrenar un modelo YOLOv8 para detección automática en planos.

---

## 📁 Estructura del Proyecto

```
train-maker/
├── components_config.yaml    ← ⭐ ARCHIVO PRINCIPAL DE CONFIGURACIÓN
├── config.py                 ← Motor de configuración (no tocar)
├── generate_backgrounds.py   ← Fase 0: Genera fondos desde planos DXF
├── phase1_extractor.py       ← Fase 1: Extrae sprites desde DXF de componentes
├── phase2_3_fusion_labeler.py← Fases 2/3: Fusiona sprites + fondo + auto-etiqueta
├── phase4_assembler.py       ← Fase 4: Ensambla dataset (split 80/10/10)
├── validate_dataset.py       ← Validación: Clipping de bboxes + safety checks
├── train.py                  ← Fase 5: Entrenamiento YOLO
├── run_pipeline.py           ← ⭐ CONTROLADOR MAESTRO (ejecuta todo)
├── input/
│   ├── component.dxf         ← Tu DXF del componente eléctrico
│   ├── modifiers/            ← PNGs de polos (//, ///) y símbolos extra
│   ├── backgrounds/          ← Fondos (auto-generados o manuales)
│   └── planos_completos/     ← Planos DXF completos (para generar fondos)
├── output/                   ← Sprites y sintéticos (generados)
├── dataset/                  ← Dataset YOLO final (generado)
│   ├── images/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   ├── labels/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── data.yaml             ← Config YOLO (auto-generada)
└── yolov8m.pt                ← Modelo pre-entrenado
```

---

## 🚀 Inicio Rápido (3 pasos)

### Paso 1: Preparar los archivos de entrada

```bash
# 1. Poné el DXF de tu componente eléctrico en input/
cp mi_interruptor.dxf train-maker/input/component.dxf

# 2. Poné tus planos completos (para generar fondos) en input/planos_completos/
cp plano_tablero_*.dxf train-maker/input/planos_completos/

# 3. (Opcional) Poné PNGs de polos y símbolos en input/modifiers/
#    Estos se componen aleatoriamente sobre el sprite base
cp polo_doble.png polo_triple.png train-maker/input/modifiers/
```

### Paso 2: Configurar `components_config.yaml`

Editá el archivo para ajustar tu componente:

```yaml
components:
  - name: "interruptor_termomagnetico"
    dxf_path: "input/component.dxf"
    images_to_generate: 10000    # Cuántas imágenes sintéticas generar
    sprite_variations: 150       # Variaciones de grosor de línea
    line_thickness_range: [2, 150]
```

### Paso 3: Ejecutar el pipeline

```bash
cd train-maker/

# Ejecutar todo: generar datos + validar + entrenar
python3 run_pipeline.py

# O solo generar datos (sin entrenar)
python3 run_pipeline.py --no-train
```

---

## 📋 Flujo Completo del Pipeline

```
┌─────────────────────────────────────────────────────┐
│  Fase 0 — Generación de Backgrounds                │
│  Renderiza planos DXF completos → corta en tiles    │
│  → filtra tiles vacíos → guarda en backgrounds/     │
└──────────────────────┬──────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────┐
│  Fase 1 — Extracción de Sprites (por componente)    │
│  Lee DXF del componente → renderiza → varía grosor  │
│  → guarda sprites PNG con transparencia             │
└──────────────────────┬──────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────┐
│  Fases 2/3 — Fusión + Auto-Etiquetado              │
│  Sprite base + modifier aleatorio (prob=70%) →      │
│  rotación aleatoria del modifier → composición →    │
│  escala + rotación → pegar en fondo → YOLO labels   │
└──────────────────────┬──────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────┐
│  Fase 4 — Ensamblaje del Dataset                    │
│  Recolecta todos los componentes → shuffle →        │
│  split 80/10/10 → inyecta negativos → data.yaml     │
└──────────────────────┬──────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────┐
│  Validación — Sanity Checks                         │
│  Verifica .txt para cada .jpg → clippea bboxes      │
│  fuera de rango → aborta si >5% sin label           │
└──────────────────────┬──────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────┐
│  Fase 5 — Entrenamiento YOLO                        │
│  YOLOv8m + batch_size/workers controlados →         │
│  augmentation desde YAML → W&B/MLflow tracking      │
└─────────────────────────────────────────────────────┘
```

---

## ⚙️ Configuración Detallada

### Configuración Global

```yaml
global:
  # Rutas (relativas a train-maker/)
  backgrounds_dir: "input/backgrounds"
  output_dir: "output"
  dataset_dir: "dataset"

  # Split del dataset
  train_ratio: 0.80    # 80% entrenamiento
  val_ratio:   0.10    # 10% validación
  test_ratio:  0.10    # 10% test

  # Hardware (para GPUs de 8GB VRAM)
  batch_size: 16       # Tamaño de batch (NO usa auto-batch)
  workers: 4           # Workers del dataloader

  # Entrenamiento
  yolo_model: "yolov8m.pt"
  epochs: 100
  imgsz: 640
  patience: 20
  project: "Liard_Detection"
```

### Generación de Backgrounds desde DXF

```yaml
backgrounds:
  enabled: true                              # true = genera fondos automáticamente
  dxf_sources_dir: "input/planos_completos"  # Carpeta con planos DXF
  tile_size: 640                             # Tamaño del tile en píxeles
  overlap: 320                               # Solapamiento entre tiles
  render_dpi: 4098                           # DPI alto para planos completos
  min_std_dev: 10                            # Descarta tiles casi vacíos
```

> **Tip:** Si ya tenés imágenes de fondo (JPG/PNG), ponelas directamente en `input/backgrounds/` y usá `enabled: false`.

### Augmentation (Anti-Overfitting)

```yaml
augmentation:
  hsv_h: 0.015          # Shift de matiz
  hsv_s: 0.7            # Shift de saturación
  hsv_v: 0.4            # Shift de brillo
  degrees: 15.0         # Rotación aleatoria ±grados
  translate: 0.1        # Traslación aleatoria
  scale: 0.5            # Escala aleatoria
  shear: 2.0            # Cizallamiento
  perspective: 0.0      # Distorsión de perspectiva
  flipud: 0.1           # Prob. de flip vertical
  fliplr: 0.5           # Prob. de flip horizontal
  mosaic: 1.0           # Prob. de mosaico (MUY efectivo)
  mixup: 0.15           # Prob. de MixUp
  copy_paste: 0.1       # Prob. de copy-paste
  erasing: 0.4          # Prob. de borrado aleatorio
```

> **¿Por qué esto importa?** Las imágenes son sintéticas — el modelo se puede memorizar tus sprites en vez de aprender patrones reales. La augmentation agresiva previene el overfitting.

### Validación y Seguridad

```yaml
validation:
  max_missing_labels_pct: 5.0   # Aborta si >5% de labels faltan
```

> **Trampa de los empty labels:** Si un bug en la Fase 2/3 no genera el 50% de los `.txt`, la validación sin este umbral pasaría "en verde" (creando archivos vacíos), entrenando un modelo que cree que "acá no hay nada" en imágenes que SÍ tienen objetos.

### Modifiers (Polos y Símbolos)

```yaml
modifiers:
  dir: "input/modifiers"      # Carpeta con PNGs (fondo transparente)
  probability: 0.70           # 70% de prob. de agregar modifier al sprite
  count_min: 1                # Min modifiers por sprite (cuando se usa)
  count_max: 1                # Max modifiers por sprite (cuando se usa)
  allow_rotation: true        # Rota el modifier aleatoriamente 0/90/180/270°
  thickness_dilation: [1, 3]  # Grosor aleatorio de línea del modifier
```

> **¿Qué son los modifiers?** Son imágenes PNG con fondo transparente (ej: los símbolos de polos `//`, `///` que aparecen en planos eléctricos reales). Se componen **encima** del sprite base antes de pegarlo al fondo.
>
> **¿Por qué aleatoriamente?** En planos reales no todos los componentes tienen polos, y los que tienen pueden estar rotados. La aleatorización simula esta variación natural y evita que el modelo se "memorice" un patrón fijo.

---

## 🔧 Agregar un Nuevo Componente

1. Obtené el DXF del componente (ej: `contactor.dxf`)
2. Agregalo a `components_config.yaml`:

```yaml
components:
  - name: "interruptor_termomagnetico"
    dxf_path: "input/component.dxf"
    images_to_generate: 10000
    sprite_variations: 150
    line_thickness_range: [2, 150]
    polarity_filters: []

  - name: "contactor"                    # ← NUEVO
    dxf_path: "input/contactor.dxf"      # ← Ruta al DXF
    images_to_generate: 10000
    sprite_variations: 150
    line_thickness_range: [2, 150]
    polarity_filters: []
```

Los class IDs se asignan automáticamente en orden de lista (0, 1, 2…). El `data.yaml` generado tendrá:

```yaml
nc: 2
names:
  0: interruptor_termomagnetico
  1: contactor
```

---

## 🏃 Comandos Útiles

```bash
# Pipeline completo (datos + entrenamiento)
python3 run_pipeline.py

# Solo generar datos
python3 run_pipeline.py --no-train

# Solo entrenar (después de generar datos)
python3 train.py

# Solo generar backgrounds desde planos DXF
python3 generate_backgrounds.py
```

---

## 🐛 Solución de Problemas

### "La carpeta de fondos está vacía"
- Opción A: Poné imágenes JPG/PNG manualmente en `input/backgrounds/`
- Opción B: Poné archivos `.dxf` de planos completos en `input/planos_completos/` y configurá `backgrounds.enabled: true`

### "Split ratios must sum to 1.0"
- Verificá que `train_ratio + val_ratio + test_ratio = 1.0` en el YAML

### "⛔ ABORTADO: X% de las imágenes no tenían label"
- Un bug en la generación está perdiendo archivos `.txt`
- Revisá los logs de la Fase 2/3
- Si es un falso positivo, subí `validation.max_missing_labels_pct` en el YAML

### OOM (Out of Memory) en la GPU
- Bajá `batch_size` a 8 o 4
- Bajá `workers` a 2
- Usá `yolov8n.pt` en vez de `yolov8m.pt` (modelo más chico)

### Errores de Pyrefly en VS Code ("Cannot find module cv2")
- Son **falsos positivos** del linter, no del código
- El código funciona perfectamente en runtime
- Se configuró un `pyproject.toml` para suprimirlos

---

## 📊 Dependencias

```bash
pip3 install opencv-python-headless ezdxf matplotlib Pillow numpy pyyaml ultralytics torch
```

---

## 🎯 Resumen de Protecciones

| Protección | Qué hace |
|---|---|
| **Idempotencia** | Limpia directorios antes de generar → reiniciar no duplica archivos |
| **Memory Management** | Escribe cada imagen a disco inmediatamente → no acumula 10k imgs en RAM |
| **Bbox Clipping** | Coordenadas fuera de [0,1] se clampean → no borra imágenes válidas |
| **Label Safety** | Aborta si >5% de labels faltan → detecta bugs silenciosos en generación |
| **VRAM Control** | batch_size y workers fijos → no explota en GPUs de 8GB |
| **Anti-Overfitting** | Augmentation controlada desde YAML → previene memorización de sprites |
| **Modifiers Aleatorios** | Prob/rotación/grosor random → simula variación real de polos en planos |
