# Comparativa: Modelo Multi-Clase vs. Modelos Single-Class

## Condiciones del experimento

| Parámetro | Valor |
|---|---|
| Arquitectura | YOLOv8n (nano) |
| Epochs | 100 |
| Imagen de entrada | 640 × 640 px |
| Hardware | NVIDIA RTX 3050 Laptop GPU (4 GB) |
| Optimizer | AdamW (auto) |
| Dataset | Tiles SAHI de planos eléctricos DXF |

## Resultados por clase

### mAP50

| Clase | Multi-class | Single-class | Δ |
|---|---|---|---|
| interruptor_termomagnetico | 0.978 | **0.990** | +0.012 |
| interruptor_diferencial | 0.995 | **0.995** | 0.000 |
| interruptor_seleccionador_manual | 0.982 | **0.995** | +0.013 |
| ojo_de_buey | 0.967 | **0.990** | +0.023 |
| **Promedio** | 0.981 | **0.993** | **+0.012** |

### mAP50-95 (métrica más exigente)

| Clase | Multi-class | Single-class | Δ |
|---|---|---|---|
| interruptor_termomagnetico | 0.887 | **0.927** | +0.040 |
| interruptor_diferencial | 0.983 | **0.993** | +0.010 |
| interruptor_seleccionador_manual | 0.562 | **0.578** | +0.016 |
| ojo_de_buey | 0.548 | **0.635** | +0.087 |
| **Promedio** | 0.745 | **0.783** | **+0.038** |

## Tiempos de entrenamiento

| Modelo | Tiempo |
|---|---|
| Multi-class (7 clases) | 0.131 h (~8 min) |
| Single: termomagnetico | 0.130 h (~8 min) |
| Single: diferencial | 0.119 h (~7 min) |
| Single: seleccionador_manual | 0.060 h (~4 min) |
| Single: ojo_de_buey | 0.032 h (~2 min) |
| **Total single-class (4 modelos)** | **0.341 h (~21 min)** |

## Análisis

### Precisión

Los modelos single-class superan al multi-class en todas las clases evaluadas, tanto en mAP50 como en mAP50-95. La mejora más significativa se observa en **ojo_de_buey** (+0.087 en mAP50-95), una clase con pocas instancias de validación (17), donde el modelo single-class logra mayor precisión de localización al no competir con otras clases durante el entrenamiento.

### Tiempo

El costo de entrenar 4 modelos single-class es 2.6× mayor que entrenar un único modelo multi-class (21 min vs. 8 min). Sin embargo, este costo es fijo por clase y no crece proporcionalmente al agregar nuevas clases al catálogo.

### Escalabilidad

La ventaja estructural del enfoque single-class es la **independencia entre modelos**: incorporar una nueva clase al catálogo requiere entrenar únicamente un nuevo modelo (~4-8 min) sin reentrenar ni afectar los modelos existentes. En el enfoque multi-class, cada extensión del catálogo implica reentrenar el modelo completo con todas las clases, con el riesgo asociado de interferencia entre clases y mayor costo de datos balanceados.

## Conclusión

| Criterio | Multi-class | Single-class |
|---|---|---|
| mAP50 promedio | 0.981 | **0.993** |
| mAP50-95 promedio | 0.745 | **0.783** |
| Tiempo total de entrenamiento | **~8 min** | ~21 min |
| Costo de agregar una clase nueva | Re-entrenar todo | **Entrenar 1 modelo nuevo** |
| Riesgo de interferencia entre clases | Sí | **No** |

Para un catálogo de símbolos eléctricos en crecimiento, el enfoque **single-class es superior tanto en precisión como en escalabilidad**, a costa de un mayor tiempo de entrenamiento inicial que se amortiza a medida que el catálogo crece.

Los resultados experimentales son consistentes con los hallazgos del paper, que demuestra estadísticamente que los modelos single-class superan al enfoque multi-class en precisión y F1-score. Aunque dicho trabajo opera sobre imágenes RGB en contexto agrícola, su conclusión central es transferible al dominio de esquemas eléctricos en DXF: la independencia entre modelos single-class elimina la interferencia entre clases durante el entrenamiento, lo que se traduce en mejores métricas individuales y, fundamentalmente, en una arquitectura donde incorporar una nueva clase al catálogo no requiere reentrenar ni degradar los modelos existentes.

## Entrenamiento con Datos Sintéticos y Fine-Tuning

### Motivación

El dataset real disponible era insuficiente para entrenar directamente: 48 imágenes de ojo_de_buey, 5 de interruptor_motorizado, y 6 de multi_medidor. Entrenar YOLO desde cero con tan pocos ejemplos produce modelos que no generalizan. Para resolver esto se adoptó una estrategia de dos etapas: pre-entrenamiento sintético seguido de fine-tuning sobre datos reales.

### Generación de datos sintéticos

Se implementó un pipeline de copy-paste augmentation que extrae patches de los símbolos eléctricos desde los tiles reales anotados y los pega sobre fondos de planos reales con posicionamiento aleatorio. El generador produce imágenes balanceadas entre clases mediante muestreo uniforme por clase. El resultado fue un dataset sintético de 2312 imágenes de entrenamiento y 237 de validación, con aproximadamente 350 instancias anotadas por clase.

### Resultados del modelo sintético

El modelo multi-clase entrenado únicamente sobre datos sintéticos (100 epochs) alcanzó mAP50 = 0.994 y mAP50-95 = 0.946 sobre el set de validación sintético, métricas casi perfectas. Sin embargo, al aplicarlo sobre planos DXF reales el modelo producía 0 detecciones, confirmando la existencia de un domain gap significativo entre las imágenes sintéticas y los planos reales renderizados.

### Fine-tuning sobre datos reales

Partiendo del modelo sintético como inicialización, se realizó fine-tuning sobre los 271 tiles reales anotados (90 de validación) durante 30 epochs con learning rate reducido. Este proceso adaptó las features del modelo al dominio real sin necesidad de grandes volúmenes de datos etiquetados.

### Conclusión del esquema sintético + fine-tuning

La combinación de datos sintéticos para pre-entrenamiento y datos reales para fine-tuning permitió obtener modelos funcionales con tan solo 5 ejemplos reales por clase. Este enfoque es especialmente relevante en dominios especializados como los esquemas eléctricos, donde la anotación manual es costosa.

## Referencias

- [Real-Time On-the-Go Annotation Framework Using YOLO for Automated Dataset Generation](https://arxiv.org/abs/2512.01165)
