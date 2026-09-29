# Plan F: qué cambiar antes de entrenar (16/09/2026)

Todo lo de acá salió de testear el modelo D contra los planos reales de Marcelo
(EZE4077, LU-UN-01, nyw-un-01, test5, hola(1)) y los 12 PDF del cliente.

Estado del modelo D, medido contra el GT de EZE4077 (73 componentes, cajas corregidas):

| métrica | valor |
|---|---|
| recall @ conf 0.20 | 98,6% (1 FN) |
| FP | 8 |
| IoU medio de las cajas | 0,466 |
| IoU >= 0,5 | 47% |
| área caja detectada / real | 0,74 (las cajas quedan chicas) |

---

## A. Inferencia — se prueban sin reentrenar

| # | Qué | Por qué | Estado |
|---|---|---|---|
| A1 | **Fusión de cajas (WBF) en `_suprimir`** | Hoy ordena por confianza y descarta; se queda con la caja chica. Solo escala 1.6 da IoU 0,538 y la fusión 1.0+1.6 baja a 0,466. Hay que promediar/unir en vez de descartar. | **pendiente** |
| A2 | **Revisar el par de escalas** | 1.0 aporta cajas chicas (área 0,71). Probar (1.3, 1.6, 2.0). | pendiente |
| A3 | **Portar `limpiar_mascaras` a `evaluate.py`** | Ya está en `detectar.py`. WIPEOUT, HATCH blanco y MTEXT con `bg_fill` tapan el plano entero (caso `hola.dxf`: 57% de la imagen en negro). | **pendiente** |
| A4 | Lectura directa de PDF | Evita el rodeo por DXF, que salía rotado y con el texto roto. | HECHO |
| A5 | **Auto-rotación** | `Plano 1/2/3` tienen el dibujo rotado 90° dentro de la hoja (`page.rotation` = 0). El modelo se entrenó con `degrees=0`. Probar 0° y 90°, quedarse con la mejor. | pendiente |
| A6 | **Escala cuando hay poco texto** | Con menos de ~100 textos reales el dpi automático es una lotería: ByA quedó subescalado, `Planos tableros` sobreescalado ×9. | pendiente |

## B. Dataset del plan F

| # | Qué | Evidencia | Estado |
|---|---|---|---|
| B1 | **`fila_densa`**: símbolos pegados, una caja por símbolo | 24 cajas abarcando 2-3 componentes en EZE4077 | HECHO |
| B2 | **`spm_branch`**: portafusible SPM inclinado + lámpara piloto | 14 de los 15 FN de D en FL-UN-02 y TSSS_2 | HECHO |
| B3 | **`barra_colectora`** como negativo | FP en cadena sobre la barra en los 4 IE-UNI | HECHO |
| B4 | **`caja_nombre`** como positivo | el UPS 8kVA (test5) y "Controlador para Transferencia Automática" (IE-UNI-01) | HECHO |
| B5 | **PAT fuera del dataset** | decisión de Tomas. Sacado de `GENS` y de la biblioteca (`p0281`, `p0286`) | HECHO |
| B6 | **Trafo (2 círculos solapados) vs 2 interruptores contiguos** | idea de Tomas: el modelo fusiona de a 2 porque dos círculos que se tocan se parecen al símbolo del transformador | **pendiente** |
| B7 | **Celdas de tabla con texto corto → negativo** | FP sobre las tablas `FUNCION/DESTINO` de LU-UN-01. Ojo: choca con "caja con letras = componente", se distinguen por contexto (grilla de celdas iguales vs colgado de un cable) | **pendiente** |
| B8 | **`///` + nodo → negativo, con el trazo real** | 18 de los 59 FP de test5 son ese patrón. Ya existe en `clutter` pero no se parece al real | **pendiente** |
| B9 | **Cajas que cubran el símbolo completo** | las cajas salen al 74% del tamaño real, el 35% no llega ni a la mitad | **pendiente** |

## C. Depende de vos

| # | Qué | Peso |
|---|---|---|
| C1 | Clasificar los 50 bloques anónimos de nyw-un-01 (`*U69`…`*U118`) | 54 INSERT, ~5% del plano |
| C2 | Completar el GT de LU-UN-01: de los 827 FP, 679 no caen sobre ningún bloque y **muchos son componentes reales dibujados sueltos** | sin esto no se puede medir precisión en LU/nyw |
| C3 | ¿El PAT cuenta como componente en los GT? | hoy: sí en EZE (73), no en LU/nyw |

---

## Orden sugerido

1. **A1 + A3** primero: son de inferencia, no requieren reentrenar, y A1 debería subir el IoU de 0,466 a ~0,54 solo.
2. Medir de nuevo contra EZE. Si el IoU sube, parte del problema B9 se resuelve sin tocar el dataset.
3. Recién ahí generar `ds3` y lanzar F con B1-B5 (+ B6-B8 si se implementan).
4. Comparar F vs D con `comparar.py` sobre EZE4077 **y** sobre los 4 planos de test de siempre, para confirmar que no se rompió nada de lo que ya andaba.

## Cuidado al medir F

F no va a aprender PAT, pero el GT de EZE4077 sí lo cuenta (5 componentes). Contra ese GT,
F va a mostrar 5 FN que D no tenía, y el recall va a dar peor aunque el modelo sea mejor.
Medir con las dos variantes (`_gt.csv` y `_gt_sinpat.csv`) para que la comparación sea justa.
