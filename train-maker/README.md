# train-maker — pipeline de datos sintéticos y entrenamiento YOLO

Genera datasets YOLO de símbolos eléctricos a partir de DXF y entrena un
modelo por componente.

> **Estado:** reescrito el 21/08/2026 después de la auditoría. Los bugs que
> estaban impidiendo que los modelos aprendieran están arreglados; los
> arreglos están comentados en cada archivo con la referencia (H1, H2, …).
> Los modelos entrenados con la versión anterior están en `modelos_v1/` y
> hay que reentrenarlos.

---

## Arranque rápido

```bash
cd train-maker
pip install -r requirements.txt

# 1. Generar y verificar los datos de un componente, sin entrenar
python run_pipeline.py --only ojo_de_buey --skip-training

# 2. Mirar la grilla que deja en verification/ y recién ahí entrenar
python run_pipeline.py --only ojo_de_buey

# 3. Los diez componentes con datos reales
python run_pipeline.py
```

La primera corrida genera los fondos desde los planos de `../dxf/` y tarda
unos minutos. Después se puede reusar con `--skip-backgrounds`.

---

## Cómo funciona

| Fase | Archivo | Qué hace |
|---|---|---|
| 0 | `generate_backgrounds.py` | Rinde los planos DXF completos, **borra los símbolos** y guarda tiles de 640 px como fondos. |
| 1 | `phase1_extractor.py` | Renderiza el DXF de cada componente a sprites PNG con distintos grosores de línea. |
| 2/3 | `phase2_3_fusion_labeler.py` | Pega los sprites sobre los fondos y escribe los labels YOLO. |
| 4 | `phase4_assembler.py` | Arma el dataset con split real train/val/test y agrega los negativos. |
| — | `verify_labels.py` | Verifica que las etiquetas coincidan con lo dibujado. **Aborta si no.** |
| 5 | `train.py` | Entrena (fase 1 sintética) y hace fine-tune (fase 2 con datos reales). |

`run_pipeline.py` orquesta todo. `run_component.py` corre fases sueltas de un
componente.

---

## El punto clave: los fondos

Los fondos salen de planos reales, pero **con los símbolos borrados**. Se
identifican por `INSERT`, círculos y hatches del tamaño de un símbolo, más
todo lo que quede contenido dentro de ellos; el resto del plano —cables,
textos, cotas, marcos, tablas— se conserva.

Esto importa porque:

* Antes los 500 fondos eran **la misma imagen en blanco**, así que el modelo
  entrenaba para separar "símbolo" de "nada" y nunca practicaba la
  discriminación que hace falta en un plano real.
* Si se usaran los planos tal cual, cada tile traería instancias reales del
  símbolo **sin etiquetar**, que es exactamente el bug H1 con datos reales.
* Descartar los tiles que contienen símbolos tampoco sirve: deja solamente
  los márgenes en blanco del plano.

La escala también importa. Los fondos se renderizan con
`target_symbol_px: 64`, o sea a la misma escala a la que
`dxf_to_image.py --target-px 64` rinde para inferencia, y de ahí sale la
calibración de `sprite_scale_min/max`. Si cambiás una, cambiá la otra.

---

## El test antes de entrenar

`verify_labels.py` es lo primero que hay que mirar cuando algo no cierra.

```bash
python verify_labels.py --dataset dataset_sintetico_ojo_de_buey \
                        --sprites output/ojo_de_buey/sprites
```

Deja una grilla en `verification/<dataset>/muestra_labels.jpg`:

* **rojo** — la caja etiquetada
* **verde** — el contenido real que hay dentro de esa caja
* **verde azulado** — algo que se parece al símbolo y **no** está etiquetado

Las tres preguntas al mirarla: ¿hay algún símbolo sin caja? ¿la caja abraza el
símbolo o le sobra blanco? ¿el fondo parece un plano?

Además falla automáticamente si detecta:

| Chequeo | Bug que detecta |
|---|---|
| Contenido real < 80 % del área de la caja | H4 — padding del sprite dentro de la caja |
| Instancias del símbolo sin etiquetar | H1 — el target usado como negativo |
| Imágenes idénticas entre splits | H2 — fuga de datos |
| Más del 45 % de imágenes sin objetos | H7 — exceso de negativos |
| Cajas demasiado grandes para la imagen | H8 — desajuste de escala |

El chequeo de instancias sin etiquetar es un template matching de la silueta
del sprite: es una heurística, no un detector. Da algún falso positivo sobre
texto del plano, por eso el umbral de fallo está en 12 % de las imágenes de
la muestra. **La grilla se mira igual.**

---

## Configuración

Todo vive en `components_config.yaml`. El loader **falla si hay claves
desconocidas**, así que un typo se ve enseguida en vez de ignorarse.

Parámetros que conviene entender antes de tocar:

```yaml
global:
  sprite_scale_min: 0.06   # fracción del lado MENOR del fondo, no del sprite
  sprite_scale_max: 0.16
  edge_crop_ratio: 0.15    # instancias cortadas por el borde, como en SAHI
  max_placement_iou: 0.15  # solape permitido entre instancias
  negative_ratio: 0.2      # presupuesto ÚNICO de imágenes sin objetos
  finetune_negative_ratio: 0.15   # negativos sintéticos en el fine-tune real

backgrounds:
  target_symbol_px: 64     # tiene que coincidir con el --target-px de inferencia
  min_circle_radius: 0.15  # círculo más chico que esto = punto de conexión
  max_tiles_per_plan: 1500
```

Por componente:

```yaml
- name: interruptor_motorizado
  exclude_as_negative: [circulo_M, circulo_con_palo]   # NUNCA como negativo
  hard_negatives:                                       # sí como negativo, con peso
    circulo_GF: 0.2
```

`exclude_as_negative` es la lista de símbolos tan parecidos al target que
pegarlos sin etiqueta sería mentirle al modelo. `hard_negatives` es lo
contrario: símbolos distintos que conviene mostrar como fondo para que
aprenda a diferenciarlos.

---

## Reentrenar

Los `.cache` de Ultralytics **hay que borrarlos** después de tocar cualquier
cosa de etiquetado: si el cache existe y coincide, Ultralytics no relee los
`.txt`.

```bash
del /s train-maker\*.cache        # Windows
```

`train.py` ya **no** reanuda solo. Antes bastaba con que existiera `last.pt`
para que reanudara el run anterior, y Ultralytics con `resume=True` restaura
los argumentos viejos e ignora el `data.yaml` y los hiperparámetros nuevos —
por eso "arreglé algo, reentrené y no cambió nada". Ahora hay que pedirlo
con `--resume`, y sin el flag el run anterior se archiva con timestamp.

---

## Cosas a revisar

* **`zips_datasets/instrumento_de_medicion_multifuncion.yolov11.zip` es una
  copia byte a byte del ZIP de `seccionador_bajo_carga`** (mismo MD5, declara
  la clase `seccionador-bajo-carga`). Hay que reexportarlo desde Roboflow.
  `manual_ingestor.py` ahora corta el pipeline si la clase del ZIP no se
  parece al componente.
* **Ninguno de los diez ZIPs trae split `valid/`**: son 100 % train. El
  ingestor hace el split él mismo con los ratios del YAML, pero los datasets
  son chicos (entre 51 y 327 imágenes) y el val queda con pocas imágenes.
  Vale la pena etiquetar más, sobre todo en `interruptor_motorizado` (51),
  `fotocelula` (55) y `tablero_de_transferencia_automatica` (64).
* Los negativos curados a mano valen mucho más que los sintéticos. Poné los
  falsos positivos reales en `train-maker/negatives_<componente>/`.
