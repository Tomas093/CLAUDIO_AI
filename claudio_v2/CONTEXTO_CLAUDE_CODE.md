# CLAUDIO_AI · Detector YOLO "componente": contexto completo

> **⚠️ 22/09/2026: este documento es del 15-16/09 y está DESACTUALIZADO.** El estado vigente
> (mejores modelos, GT v4, decisiones de Tomas y el pedido pendiente) está en
> **`CLAUDIO_AI/OBJETIVOS.md`, sección 0**. Leer eso primero. Lo de acá sirve como contexto de
> cómo empezó el proyecto; las cifras y "el mejor modelo" que figuran abajo ya no valen.

Este documento sirve para continuar en Claude Code el trabajo empezado en Cowork (2026-09-15/16).
Usuario: Tomas. **Hablarle en español, directo y conciso.** La PC es Windows con una RTX 3060 Ti y Python 3.9 (`py -3`).
Carpeta raíz: `C:\Users\Tomas\Documents\LAB3\CLAUDIO_AI`. Todo el trabajo nuevo vive en `claudio_v2\`.

---

## 1. Objetivo

- Entrenar un YOLO con **una sola clase `componente`** que detecte **cualquier símbolo eléctrico** en planos **unifilares** (DXF de tableros de baja tensión).
- **Prioridad absoluta: recall 100%, es decir 0 falsos negativos.** Los falsos positivos se aceptan.
- **Puntaje para comparar modelos** (definido por el usuario):
  - 0.7 × recall + 0.2 × (umbral de confianza con que lo logra / 0.25) + 0.1 × (1 − FP/GT).
  - El término de recall es fuerte: cada 1% de FN resta 5% de ese término.
  - Se evalúan umbrales **0.25 / 0.20 / 0.15 / 0.10 / 0.05** mostrando recall, FN y FP de cada uno. Script: `src/comparar.py`.
- **Modelo:** nano (yolo11n). El usuario prefiere nano; se descartó el small.
- **Plazo:** jueves 17/09 a la tarde/noche como máximo.

## 2. Reglas de etiquetado (decididas con el usuario, no cambiar sin preguntar)

- **Componente (positivo):**
  - Aparatos de maniobra y protección: termomagnéticas, diferenciales (el toroide va DENTRO de la caja del diferencial), seccionadores, contactores, DPS, fusibles.
  - Cajas con letras: IM, CT, GE, TTAB, PF20, PLC, UPS, kWh, POWER METER, VAR y cajas grandes con nombre (KM, VFD, MED, DPS).
  - Cargas: motor, lámpara, toma.
  - Borneras (DIN y circulares X2).
  - PAT / tierra.
  - Transformadores y TC. Un TC suelto sobre el cable es un componente aparte.
  - Contactos, llaves y pulsadores (S-M-0-A, SPM).
  - Los símbolos de leyenda también cuentan aunque no tengan cable.
- **Negativo:**
  - Marco y título de tablero (el recuadro punteado), marcas `///` de conductores, nodos y puntos de unión.
  - Números en círculo, flechas de salida, terminaciones de cable (triángulos), tablas y rótulos, texto suelto.
  - Paneles dibujados como rectángulo grande con texto ("Panel MDP"), resistencia "TR".
- **Cajas:**
  - Solo el símbolo, sin `///` ni texto.
  - Un multipolar lleva una sola caja.
  - Partes pegadas o unidas llevan una caja (ej.: motorizado = círculo M + línea + 2 puntos); partes lejanas llevan cajas separadas.
  - Fusible y contactor, aunque estén en la misma columna, van **separados** (corrección del usuario en el Vyre).
- **Biblioteca `.elmt` (QElectroTech):** se puede usar (el usuario tiene autorización explícita), pero **no deben quedar rastros en el modelo**: DXF sin metadata (sin nombres, uuid, autor ni textos dinámicos) y el `.pt` limpio (sin `train_args`, optimizer, fecha ni versión; ver `clean()` en `train.py`).
- **No usar datasets de entrenamientos anteriores** (carpetas viejas de la raíz): estaban mal.
- **Planos externos y QElectroTech: SOLO TEST, no entrenamiento** (pedido explícito).

## 3. Planos de test

| Plano | DXF | GT | Nota |
|---|---|---|---|
| test1 | `test1.dxf` | `test/test_1/verdad_terreno/test1_completo.csv` | 98 componentes (incluye 5 Cajas en T) |
| test_2 | `test_2.dxf` | `test/test_2/verdad_terreno/test_2_completo.csv` | 117 |
| FL-UN-02 | `dxf/FL-UN-02_tablero_1.dxf` | `dxf/fl_un_02_gt_completo.csv` | 363 |
| TSSS_2 | `TSSS_2 (1).dxf` | `dxf/tsss_2_gt_completo.csv` | 206 |

- **Test extra QElectroTech:** `claudio_v2/data/test_qet.zip` se descomprime solo en `data/test_qet/`. Son 74 folios y 1560 componentes con etiquetas automáticas.
  - Se evalúa con `set EVAL_SET=qet`.
  - **Pendiente:** el usuario debe revisar la grilla de 302 tipos (`handoff`, imágenes `qet_tipos_*.jpg` en Cowork) y decir cuáles no son componente. Esa exclusión se ajusta en `NO_COMP` de `src/qet2dxf.py`.
- **Externos en revisión** (GT parcial en `handoff/gt_externos/`, cajas en px del render interno):
  - `sld_pabrik_gula_iec60617` (77, v3), `grid_utility_to_mdp_iec60617` (118, v3), `substation_110_33kv_sld` (42, v2).
  - El usuario marcó varias correcciones que ya están aplicadas. Quedaban pendientes en pabrik algunas bobinas corridas y pares de relés sin separar.
  - El CCM y el 05 unifilar se descartaron.
- **Vyre:** `UNIFILAR TABLERO GENERAL Vyre 09 09 2026.dxf` va a **entrenamiento** (86 cajas v3 ya corregidas por el usuario, en `data/real_labels.json` + `data/renders/vyre_d.png`). Plano5 no se usa.

## 4. Pipeline (`claudio_v2/src`)

- `paths.py`: BASE (raíz, env `CLAUDIO_BASE`), DATA, WORK, DS (env `CLAUDIO_DS`).
- `render.py`, `scale.py`, `postproc.py`:
  - Render DXF con ezdxf, política de color BLACK.
  - Escala automática: la mediana de altura de texto queda en 11 px (`auto_ppc`).
  - `darken`, `ink_map` y `resize_keep_strokes`.
- `elmt2dxf.py`: `.elmt` → DXF limpio (con rellenos). `build_lib_full.py` arma la biblioteca de símbolos eléctricos filtrada → `data/sym_lib_full`.
- `procsym.py`: símbolos procedurales (caja T, borneras, PAT, fusible, lámpara, caja con letras, contactor).
- `compose.py`: tiles sintéticos de 640 px.
  - Estilo unifilar: cables, texto, tablas, planillas de circuitos, recuadros punteados, `///`, nodos y fondos reales.
  - `tablero_row` (env `P_TABLERO`): borneras, selectoras y pulsadores chicos sobre la línea punteada con su texto.
  - `spm_branch` (env `P_SPM`): rama de comando con fusible SPM inclinado + lámpara piloto.
- `build_all.py`: dataset.
  - Sintéticos NP/NN, tiles reales NR (planos reales + Vyre) y capturas manuales de Roboflow (se excluyen los zips estirados v2i/v3i).
  - Env: `NP`, `NN`, `NR`, `P_TABLERO`, `P_SPM`, `CLAUDIO_DS`.
- `train.py`: entrenamiento, con "hooks" por nombre de corrida.
  - `B_s_largo` = **plan D**: arma `work/ds2` (24000/3000/1200, P_TABLERO 0.35) y entrena un nano desde `runs/A_n_largo/weights/best.pt`, con tope `D_HOURS`.
  - `F_spm` = **ajuste fino SPM**: arma `work/ds3` (6000/600/400, P_SPM 0.6) y entrena sobre ds2+ds3 (`work/ds23.yaml`) desde D, 15 épocas, lr0 0.002.
  - `E_largo` = **corrida larga**: si F no existe, primero corre F, lo evalúa y lo copia a `best_componente_v2_ft.pt`. Después entrena desde F sobre ds2+ds3, hasta 250 épocas con paciencia `PATIENCE` y sin tope de horas.
  - Siempre guarda `runs/<name>/best_limpio.pt`, sin rastros.
- `evaluate.py <pesos> <conf> <tag>`: evaluación.
  - Render del DXF con texto, **multi-escala `EVAL_SCALES` (default "1.0,1.6")**, tiles de 640 con stride 320, conf 0.05.
  - Fusión NMS en coordenadas CAD; supresión de anidadas solo con áreas dentro de 4x.
  - Asignación GT↔detección **húngara**.
  - Salida: `work/eval/<tag>/{resumen.json, *_detecciones.csv, *_visual.png}` (verde TP, azul FP, círculo rojo FN).
  - `EVAL_SET=qet` evalúa el set QElectroTech.
- `comparar.py tag1 tag2 ...`: recalcula desde los CSV de detecciones y muestra la tabla por umbral y el puntaje (env `CLAUDIO_WORK` opcional).
- `qet2dxf.py`: proyecto `.qet` → DXF por folio + cajas. Los proyectos originales están en `handoff/qet_proyectos_originales.zip`.
- `noche.py` / `run_noche.bat`: orquestador viejo (C → A → B). Ya no se usa.
- **Lanzadores:**
  - `run_D.bat`: plan D → eval `D_final` → `best_componente_v2.pt`.
  - `run_E.bat`: espera a que aparezca "plan D fin" en `log_D.txt` y lanza `train.py --name E_largo` (que incluye F) → eval `E_largo` → `best_componente_v2_largo.pt`.

**Receta de inferencia que debe usar el proyecto del usuario** (cambia respecto de la inferencia simple):

1. Render del DXF con texto y darken.
2. Escala automática (texto a 11 px).
3. Inferir **a 1.0x y 1.6x** con tiles de 640 y stride 320, conf 0.05.
4. Pasar a coordenadas CAD y fusionar con NMS (IoU 0.45 + anidadas con IoS 0.7 si las áreas están dentro de 4x).
5. Aplicar el umbral final.

## 5. Historial y resultados

- **Modelo viejo** (`detector_pack` nano), conf 0.20, 1 escala: test1 78.6%, test_2 80.3%, FL-UN-02 42.7%, TSSS_2 41.3%.
- **C_rapido_n20** (nano, 20 épocas).
  - Con 1 escala y conf 0.20: 100 / 97.4 / 72.7 / 74.8%.
  - **Con 2 escalas: 100 / 100 / 95.0 / 98.1%** (FN total 22, FP 506).
  - Las Cajas en T parecían FN por un error del evaluador (asignación greedy + GT duplicado); ya está corregido.
- **A_n_largo**: la cortó una actualización de Windows en la época 37/59 (00:43 del 16/09). Solo se usa su `best.pt` como punto de partida.
- **D (`runs/B_s_largo`, nano)**, época 34 de 86, evaluado en la nube con 2 escalas:

| conf | recall | FN | FP | puntaje |
|---|---|---|---|---|
| 0.25 | 98.9% | 9 | 499 | **0.896** |
| 0.20 | 99.0% | 8 | 565 | 0.852 |
| 0.10 | 99.5% | 4 | 743 | 0.767 |
| 0.05 | 99.6% | 3 | 923 | 0.727 |

  - Las borneras quedaron resueltas.
  - Los FN restantes son casi todos **SPM** (fusible diminuto inclinado en rama de comando, unos 12×6 px). Para eso se agregó F.
  - Queda 1 seccionador rotativo en test_2, que aparece con 0.15.
- **Estado al escribir este documento (16/09 ~10:40 hora PC):**
  - D entrenando: época 36/86, tope 6 h, termina ~13:45, después se evalúa sola.
  - `run_E.bat` ya está lanzado y esperando para correr F y después E.

## 6. Cambios importantes ya aplicados (para no repetirlos)

1. **`compose.py`: se corrigieron atajos que causaban FN.**
   - Antes se blanqueaba bajo el símbolo el 80% de las veces; ahora el 40%, y en filas de tablero la línea punteada atraviesa el símbolo.
   - Los nodos sobre cable pasaron a ser puntos llenos (el círculo hueco + `///` imitaba una bornera y estaba como negativo).
   - El umbral de "ya hay tinta" depende del tamaño.
   - Los símbolos chicos arrancan en 1.1× la altura de texto.
   - Binarización con umbral de 200 a 235.
   - La altura de texto de entrenamiento va de 8.8 a 25 px para cubrir la inferencia a 1.0–1.6x.
2. **`train.py`:** `scale` 0.2 (antes 0.35), `fliplr` 0.2 (el espejado invierte "X2", "M-0-A"), `patience` configurable.
3. **`evaluate.py`:**
   - Asignación húngara.
   - Solo se saltean tiles sin tinta.
   - Multi-escala.
   - Set QET.
4. **Investigación con subagentes, ideas pendientes:**
   - Ajuste fino final con datos reales (mosaic 0, lr0 0.002, 10–15 épocas).
   - Validar con planos reales en vez de sintéticos (la validación sintética da 0.99 y no predice el recall real).
   - Cabeza P2 solo si hace falta.
   - Minería de difíciles en planos nuevos.

## 7. Pendientes / próximos pasos

1. Cuando termine D: correr `comparar.py` sobre `D_final` contra C (tag `Cms`, 2 escalas; si no existe en la PC, re-evaluar C con `evaluate.py work\runs\C_rapido_n20\best_limpio.pt 0.20 Cms`). Mostrarle al usuario la tabla por umbral e imágenes de los FN.
2. Evaluar D, F y E también con `EVAL_SET=qet`.
3. Verificar que F arranque bien (`log_E.txt`, `work/ds3`) y comparar F contra D en SPM.
4. Revisión del usuario de la grilla de tipos QET → ajustar `NO_COMP` → regenerar `test_qet.zip`.
5. Nuevos símbolos para la biblioteca (solo si el usuario lo aprueba para entrenamiento): `qelectrotech-element-contrib`, WEG `comandos_eletricos` + `71_rgie/7110_schema`. La lista está en `handoff/contrib_utiles.txt` (664 elementos, GPLv2).
6. Dataset de Roboflow "TFG símbolos eléctricos unifilares" (CC BY 4.0): el usuario lo va a exportar en YOLOv11 a `zips_datasets`. Auditar las etiquetas antes de usarlo; se puede usar para entrenamiento o validación.
7. Bibliocad / Librería CAD: el usuario va a bajar DWG con su cuenta. Convertirlos a DXF (ODA File Converter) y etiquetarlos asistido con revisión en grilla numerada.
8. Limpieza de la carpeta raíz: el plan ya se le pasó al usuario, que decide. Usar cuarentena `_para_borrar`, no borrar directo, y no tocar `claudio_v2\work` mientras entrena.
9. Entregar un script de inferencia para el proyecto del usuario con la receta de 2 escalas (sección 4).

## 8. Forma de trabajar con el usuario

- **Antes de entrenar o cambiar etiquetas:**
  - Mostrar grillas **numeradas** (`src/numgrid.py`) y planos numerados.
  - El usuario responde con números.
  - Aplicar exactamente sus correcciones.
- **Revisión de etiquetas:** usar las dos vías, visual y con los inserts del DXF.
- **Evaluación:** siempre con los DXF con texto; nunca sacar texto para evaluar.
- **Resultados:** reportar la tabla por umbral de 0.25 a 0.05 con recall/FN/FP/puntaje, más qué bloques fallan.
- **Cambios de inferencia:** permitidos, pero **anotarlos siempre**.
- **Operaciones que tocan la PC:** no cortar entrenamientos en curso sin avisar. Explicar antes de hacer algo invasivo.
- **Windows:** la PC puede suspenderse o actualizarse y cortar el entrenamiento. Sugerir suspensión "Nunca" y pausar actualizaciones.

## 9. Material en `claudio_v2/handoff/`

- `qet_proyectos_originales.zip`: los 23 `.qet` de `qelectrotech-source-mirror/examples` (GPLv2).
- `contrib_utiles.txt`: rutas de símbolos útiles de `qelectrotech-element-contrib`.
- `gt_externos/`: GT en revisión de los 3 externos y `vyre_v3.json` (cajas en px del render interno con `auto_ppc`).
- `herramientas/`: scripts de revisión usados en la nube.
  - `sheet.py`, `region.py`, `ctx.py`: recortes con regla de coordenadas.
  - `show.py`: plano y grilla numerados.
  - `typegrid.py`: grilla por tipo QET.
  - `multiscale.py`, `score.py`: experimento de escalas.
  - Tienen rutas de la nube (`/home/claude/...`): hay que adaptarlas.

## 10. Sesion Claude Code 16/09 (tarde)

- **`claudio_v2/detectar.py` (NUEVO, pendiente 9 resuelto):** script de inferencia **autocontenido** para el
  proyecto del usuario. No importa nada de `src/`; solo pide ezdxf, matplotlib, opencv, numpy y ultralytics.
  Implementa la receta de la seccion 4 (render con texto + darken, `auto_ppc` a 11 px, escalas 1.0 y 1.6,
  tiles 640 stride 320, conf interna 0.05, fusion NMS en CAD, umbral final al final).
  - Uso: `py -3 detectar.py <plano.dxf|carpeta> --pesos best_limpio.pt --conf 0.25`
  - Flags: `--escalas`, `--ppc` (planos sin texto), `--device`, `--batch` (bajar si la GPU esta ocupada), `--sin-visual`.
  - Salidas: `<plano>_detecciones.csv` en coordenadas CAD, `<plano>_visual.png` y `resumen.json`.
  - **Verificado**: a 1 escala reproduce `src/evaluate.py` caja por caja sobre test1.
- **Hallazgo: el eval guardado `work/eval/C_rapido_n20` esta obsoleto.** Es de 1 sola escala y de una version
  previa de `evaluate.py`, anterior a la restriccion de areas 4x en la supresion de anidadas (da 125 cajas en
  test1 donde la version actual da 128). **No sirve para comparar contra D**: hay que re-evaluar C con el
  codigo actual. Se lanzo esa re-evaluacion con tag `Cms` (2 escalas).
- **Correr evals sin molestar al entrenamiento:** `CUDA_VISIBLE_DEVICES=-1` fuerza CPU
  (`CUDA_VISIBLE_DEVICES=""` no alcanza en Windows: ultralytics igual pide CUDA:0 y falla).

### Cambio de inferencia 16/09 (tarde): mascaras y fondos que tapan el plano

`detectar.py` ahora llama a `limpiar_mascaras(doc)` antes de renderizar. Con `ColorPolicy.BLACK`
hay tres cosas que se pintan de negro solido y tapan el plano entero:

1. **WIPEOUT** (mascara de AutoCAD, se dibuja del color del fondo) -> se borra del modelspace.
2. **HATCH solido blanco** (relleno de tapado) -> se borra.
3. **MTEXT con `bg_fill`** -> se le apaga el fondo y se conserva el texto (hace falta para `auto_ppc`).

Encontrado con `hola.dxf`, donde un MTEXT de notas de 17.5x15.6 con `bg_fill=3` sobre un plano de
20x20 dejaba el 57% de la imagen en negro: el modelo solo veia la columna de simbolos que quedaba
fuera del bloque. Sin el arreglo daba 18 detecciones; con el arreglo, 24 (los 14 simbolos de la
leyenda mas FP de texto). **`src/evaluate.py` NO tiene este arreglo todavia**: los 4 planos de test
no tienen wipeouts ni MTEXT con fondo, pero conviene portarlo antes de evaluar planos del cliente.
