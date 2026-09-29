# CLAUDIO_AI — objetivo, contexto y estado

> Documento para que otra IA (u otra persona) retome el proyecto sin tener que reconstruir la
> historia. Escrito el 19/09/2026. Si cambiás algo importante, actualizalo.
>
> **Si retomás a mitad de camino, empezá por la sección 0.** Tiene el estado en vivo: qué está
> corriendo, qué quedó a medias y cuál es el próximo paso concreto. Mantenela al día a medida
> que avances, no al final: es lo único que evita rehacer trabajo ya hecho.

---

## 0. Retomar acá — estado en vivo

**28/09/2026 ~15:15. CORRIENDO (vía WMI):** `cadena_orden.sh` (RF5, rfdetr 1.11, época ~40/60, termina ~21 h)
y **`cadena_ds21.sh` (relanzada 28/09 23:24 tras la revisión)**, que espera `CADENA_ORDEN_LISTA` y después arma **ds21** y entrena **N21** (yolo11n, 200 ép.,
`best_componente_v27_N21.pt`) y **RF6** (rfdetr 1.3 como RF4, 60 ép., lr_drop 48, `v28_RF6.pth`); eval con
**`GT_V6=1`** (tags `V6_N21_*`, `V6_RF6_nano_*`). Tarea programada de Windows "CLAUDIO cadena entrenamiento"
(al iniciar sesión) corre `claudio_v2/arranque_cadena.ps1`, que relanza las cadenas sin terminar tras un corte de luz
(hubo 2 el 27/09). RF5 ahora retoma desde `last.ckpt`.
- **RF4 (ds20) terminó 28/09 03:52: mejor modelo individual.** Con GT v6: 0,05 FN 1/FP 6592 · 0,08 FN 2/4174 ·
  0,13 FN 6 · 0,25 FN 10/2129. Da confianzas más bajas que RF23: usar ~0,08, no 0,13. RF4+M11 (O prob.) @0,25:
  sólo los 4 fusibles de test1. yolo11m (M11_20) no le gana a RF-DETR.
- **GT v6** (`test1_completo_v6.csv`, `GT_V6=1` en evaluate.py): v5 sin los 9 `AUDIT-rotulo_con_nombre`. Tomas:
  los rectángulos altos con el nombre del tablero en vertical NO son componentes.
- **ds21** (`src/receta_ds21.py`, grillas en `work/grillas_ds21/`, aprobadas en lo conceptual por Tomas):
  ds20 + `FONDO_BLANCO=1` (sin fondo de planos reales; lo negativo va en capa aparte y se descarta POR OBJETO
  si toca una caja positiva) + `VIS_MIN=0.90` (28/09, Tomas: "al menos el 90% del componente"; aplica a símbolos sueltos, filas, ternas, `tablero_row`, tiles reales y manuales; antes medio símbolo cortado era positivo: la mitad de los FP ≥0,5
  de RF4 eran pedazos) + `letra_recuadro` (amperímetro A$C45664E16, P 0,15, con par negativo de texto) +
  `rotulo_vertical` como NEGATIVO (P_ROTULO_NEG 0,30; `caja_nombre` no se gira) + `negativos_extra` (círculos
  numerados grandes, círculos que se cortan, flecha de alimentación). Cajas: p50 39 px, <20 px 14,2% (ds20 41/13,5%;
  plan O falló con 16,3%).
- **Auditoría de ds21 (28/09 tarde)** con `src/auditar_ds.py` (chequeos por caja + modelo R buscando componentes sin
  etiqueta) sobre una muestra, más dos subagentes (revisión de código y revisión visual de 58 tiles). Se corrigió:
  ternas que no entraban en el tile, texto de la pareja dentro de la caja vecina, filas densas pegadas/encimadas y
  sin VIS_MIN, letras fuera del tile, sprites apaisados girados (= rótulo vertical), negativos de objeto dibujados
  DESPUÉS de los positivos buscando lugar libre (antes FONDO_BLANCO borraba ~70%), símbolos cortados que ya no se
  dibujan (sin parches blancos), etiquetas de texto que no pisan cajas ni quedan debajo de otro símbolo, marco
  punteado con interior blanco; cadena_ds21.sh robusta (COCO atómico + marca COMPLETO, marca LISTA sólo si todo OK,
  eval por tanda) y arranque_cadena.ps1 no relanza si hay un entrenamiento vivo. Cajas: p50 40 px, <20 px 13,7%,
  0 pares solapados. Pseudo-etiquetas de zips (modelo K, ≥0,80): Tomas eligió filtrarlas (`src/confirmar_pseudo.py` → `data/zips_pseudo_confirmadas.json`, env `PSEUDO_CONFIRMA=1` en la receta): quedan sólo si RF4 las ve (≥0,3) y no tocan el borde de la captura (Tomas: "que se vea el 90%"); el resto pasa a zona neutra (4008 → 3271). Antes ~4% eran
  basura (círculos numerados, texto); grilla en `work/auditoria_ds21/pseudo_etiquetas_muestra.png`.
- **Revisión a mano de Tomas (28/09 noche, grillas de `claudio_v2/work/revision_tomas/`)**. Reglas: lo que no nombra NO es
  componente pero tampoco negativo (zona neutra); las cruces se dejan por las dudas. Aplicado: A/B → 228 pseudo-etiquetas
  confirmadas a mano (`data/zips_pseudo_revision_tomas.json`, las leen build_all aunque RF4 no las vea o toquen el borde);
  sintéticos con `VIS_SINT=1.0` (sólo símbolos enteros; reales/manuales VIS_MIN 0,90); 5 sprites excluidos (e00095,
  e00400, e00336, e00508, e00234); ternas sin sprites alargados (>3,5). `src/origen_muestra.py` reproduce tiles sintéticos y
  dice de qué sprite sale cada caja; build_all deja `origen_manual.json` (captura y orig/pseudo de cada etiqueta manual).
  **Pendiente de Tomas:** pseudo-etiquetas malas que RF4 también ve con 0,95+ (texto "3X LSOH", círculo "3"): ningún
  umbral las filtra; y si la flecha de alimentación (triángulo) es componente (nombró C11 y C50).
- **Revisión con subagentes (28/09 ~23 h):** las 3.271 pseudo-etiquetas no revisadas (4 agentes, 52 páginas en
  `work/revision_agentes/`) → 188 a zona neutra (185 no componente: casi todos círculos de referencia numerados, textos
  de calibre, letras de fase, ///; `data/zips_pseudo_revision_agentes.json`). Las 5.165 etiquetas originales (zips +
  planos reales; 5 agentes, 81 páginas en `work/revision_agentes_orig/`) → sólo 3 malas, en vyre (GM_U+U en una caja ×2,
  una línea) → `data/revision_originales_agentes.json`, leídas por build_all (zona neutra). Flecha de alimentación:
  Tomas confirmó que NO es componente (sigue como negativo). ds21 relanzado 28/09 23:24.
- **RF5 (rfdetr 1.11) terminó 28/09 20:23** pero su evaluación falló (rfdetr 1.11 busca pesos relativos en
  ~/.roboflow/models; arreglado en evaluate.py con abspath). Re-evaluación: `eval_rf5.sh` (tags V6_RF5_nano_*).
- Paquete para el compañero de Tomas: `claudio_v2/pipeline_claudio.zip` (pipeline + 4 modelos + 7 planos +
  `verificar.py`, que compara entorno/render/modelo/plano contra la PC de Tomas).

**Última actualización: 22/09/2026 ~18:35. CORRIENDO: `claudio_v2/cadena_XYZ.sh`** =
R2_full (yolo11n, receta de R) → X_26p2 (yolo26n-p2, batch 8, BAJADO A 100 ÉPOCAS el 23/09 en la época 65 con `src/reanudar.py`; backup del last.pt de 200 en `_para_borrar/`) → Z_26n (yolo26n,
batch 16) → Y_11s (yolo11s, batch 8), **todos sobre ds18**. Salidas: `best_componente_v19_R2 /
v20_X / v21_Z / v22_Y.pt`, evaluados con `GT_V4=1` a 0,05 (tags `V4_<plan>_005` y
`V4_<plan>_marcelo`). Progreso en `log_cadena_XYZ.txt` y `log_<plan>.txt`; la cadena es
idempotente. Un primer X_26p2 sobre ds13 se cortó a los 35 min por pedido de Tomas
(`_para_borrar/X_26p2_ds13_cortado`). La comparación justa de X/Z/Y es contra **R2**, no contra R.

**🟢 RF-DETR NANO TERMINÓ (25/09 09:31, 40 épocas, 17 h) → `best_componente_v23_RF.pth`. ES EL MEJOR:**
**100% de recall con UN solo modelo** hasta conf **0,13** (FP 3779; R+U necesitaba 0,05 y 4870 FP).
Detecta el PLC. Tabla: 0,25 FN 12 FP 2477 · 0,20 FN 7 · 0,15 FN 2 · 0,13 FN 0 FP 3779 · 0,10 FN 0 FP 4612.
FN@0,25: 3 `AUDIT-rotulo_con` (test1), 1 SPM (fl_un_02), 3 `C4C456AEF`, 3 amperímetros, 1 PULS, PLC.
Ensambles (`src/umbral_recall.py`): **RF+U (O prob.) 100% hasta 0,19 con FP 3956; @0,25 FN 2 (2
amperímetros)**. RF+R (max): 100% hasta 0,17 (FP 3938); @0,25 FN 1 (sólo el PLC). RF+S (O prob.):
100% hasta 0,17. Siguiente para el objetivo (100% @~0,25): subir la confianza de amperímetros y
`C4C456AEF` en RF (más épocas / datos de marco punteado como U).

**🔴 25/09: EL 100% DE RF ESTABA INFLADO — GT v5.** Tomas vio que en test1 faltan los fusibles: son los 5
fusibles chicos inclinados "x3" que alimentan la lámpara piloto de cada TSAsc4 (líneas sueltas, sin bloque:
no estaban en ningún GT). `test/test_1/verdad_terreno/test1_completo_v5.csv` = v4 + 5 filas
`AUDIT-fusible_x3` (caja = polilínea del cuerpo del fusible). Evaluar con **`GT_V5=1`** (cascada v5>v4>…;
total 3148). Con v5: RF @0,05 FN 2 (fusibles) · @0,25 FN 17; R @0,05 FN 6; RF+R unión @0,25 FN 6 (5 fusibles
+ PLC). Ningún modelo llega a 100% ni a 0,05. RF ve esos fusibles con conf 0,06–0,10. Es el patrón de
`compose.spm_branch` (fusible inclinado + lámpara, P_SPM 0,45): el generador existe pero no alcanza.
test_2 NO tiene fusibles (confirmado por Tomas). `src/visual_pdf.py`: PDF vectorial de una evaluación
(verde acierto / azul FP / rojo perdido); ZIP para el amigo de Tomas en `claudio_v2/detecciones_RFDETR_7planos.zip`.

**CORRIENDO desde 25/09 22:47: `claudio_v2/cadena_orden.sh`** (orden pedido por Tomas):
1. **M11_20** = yolo11m sobre ds20, 100 épocas (~13 min/época ≈ 22 h, mismo tiempo de GPU que RF4), batch 8
   (4 GB). Comparar contra RF-DETR (v23 y RF4) para decidir si seguir con RF-DETR o con un YOLO medium.
2. **RF4** retoma desde `work/runs/RF4_nano/checkpoint.pth` (murió en la época 2: al cerrarse la sesión de
   Claude se mueren los procesos de fondo). `train_rfdetr.py` ahora retoma solo si hay checkpoint.
3. **RF5** (rfdetr 1.11) al final.
**26/09: yolo11m murió en la época 7 a las 23:40 porque la app de Claude se auto-actualizó** (servicio "Claude" 2.9939.2.0,
evento 7045): `Start-Process` NO desacopla. Relanzado 26/09 09:18 vía WMI (retomó desde la época 7):
`Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='"...\sh.exe" cadena_orden.sh'; CurrentDirectory='...claudio_v2'}`
(padre = WmiPrvSE.exe). Lo de abajo sobre Start-Process queda como historia.
Log `log_cadena_orden.txt`. **(viejo, no alcanza)** Lanzar desacoplado con PowerShell
`Start-Process 'C:\Program Files\Git\usr\bin\sh.exe' -ArgumentList 'cadena_orden.sh' -WindowStyle Hidden`
(con `nohup ... &` desde la sesión muere al cerrarla) y con `export PATH=/usr/bin:...` en el script
(Start-Process no trae date/cp). `cadena_RF45.sh` y `cadena_M11.sh` quedaron en `_para_borrar/`.

**(reemplazada por cadena_orden.sh) `claudio_v2/cadena_RF45.sh`** (RF2 y RF3 CANCELADOS por Tomas: había que
corregir el fusible antes). ds20 = ds19 + `compose.fusible_dxf` (P_FUSDXF 0,15) sin los sprites
`u_fusible_1..3` (en `_para_borrar/sym_lib_tomas_fusibles_viejos`: la caja traía los cables, 84x28).
`fusible_dxf` usa la geometría exacta de `data/fusibles_tomas/{Fusible,Fusible_2,Fusible_3}.dxf` (Tomas;
Fusible_3 = mismo dibujo que test1, de otra hoja: x 2245 vs 1877-1938), cuerpo a 0,9-1,3 alturas de texto,
caja = sólo el cuerpo; lo que parece "rayado" en test1 es el cable que atraviesa el cuerpo (no hay HATCH).
Con P_FUSDXF 0,30 la mediana bajaba a 42 px y <20 px a 13,9% (cerca del plan O, §5.8): se dejó 0,15
(44 px / 12,0%, como ds9). RF4 = rfdetr 1.3 sobre ds20 (60 ép., lr_drop 48) → `best_componente_v24_RF4.pth`;
después RF5 = rfdetr 1.11 (D:) sobre ds20 → `v25_RF5.pth`. Eval con GT_V5=1, tags `V5_RF4_nano_*`, `V5_RF5_nano_*`.

**EN ESPERA: `claudio_v2/cadena_M11.sh`** (arranca cuando `log_cadena_RF45.txt` diga CADENA_RF45_LISTA).
yolo11m (~20 M parámetros) sobre ds20, hook `M11_20` en `src/train.py` (batch 8, 200 ép.), para comparar a
tamaño parecido con RF-DETR Nano (30 M; R es yolo11n, 2,6 M): sin esto no se sabe si RF gana por ser
transformer o por ser más grande. Sale `best_componente_v26_M11.pt`, tags `V5_M11_20_*` (GT_V5).

**(cancelado) `claudio_v2/cadena_RF2.sh`** = RF-DETR Nano sobre ds19 (ds18 + marco
punteado, receta de U), 60 épocas, **lr_drop=48** (rfdetr trae lr_drop=100: con 40 épocas RF_nano NUNCA
bajó el LR y seguía mejorando en la 39: mAP50-95 val 0,793 → 0,806 entre las épocas 32 y 39), paciencia 60 (que el
early stopping no corte antes de la fase de LR bajo). ~23 min/época ≈ 23 h. Sale `best_componente_v24_RF2.pth`,
tags `V4_RF2_nano_*`. Paquete de inferencia RF entregado a Tomas: `claudio_v2/detector_rfdetr.zip`
(`entrega_RF/`: detectar.py con RF-DETR y `--pdf-anotado`, conf default 0,13; `src/pdf_anotado.py`).

**EN ESPERA: `claudio_v2/cadena_RF3.sh`** (arranca sola cuando RF2 diga listo/fallo). RF3 = RF2 idéntico
pero con **rfdetr 1.11** (pedido de Tomas: probar la versión nueva, entrena ~23% más rápido). Python 3.11
**portátil en D:** (`D:\CLAUDIO_RFDETR\python` + venv `D:\CLAUDIO_RFDETRenv311`, instalado con uv; NO registrado en
Windows: `py -3` sigue siendo 3.9). torch 2.5.1+cu121, opencv-python-headless 4.10 (dos paquetes de opencv
chocan). Salida en `D:\CLAUDIO_RFDETR
uns\RF3_nano`, log `D:\CLAUDIO_RFDETR\log_RF3_nano.txt`; dataset leído de
`work/ds19_coco` (SSD). Script `src/train_rfdetr11.py` (API 1.11: `lr_scheduler_kwargs={'lr_drop':48}`;
hace falta PYTHONUTF8=1). Los pesos de 1.3.0 cargan en 1.11. `evaluate.py` ya no exige ultralytics.
Sale `best_componente_v25_RF3.pth`, tags `V4_RF3_nano_*`.

**(terminado) `claudio_v2/cadena_RF.sh`** (batch 4 × accum 4; con batch 8 desbordaba la GPU y tardaba 13 h/época; ahora ~50 min/época). Cadena XYZ TERMINADA (Y_11s 24/09 16:20). (pedido de Tomas 24/09: probar RF-DETR, el más chico). Arranca
sola cuando `log_cadena_XYZ.txt` diga `CADENA_XYZ_LISTA`. RF-DETR **Nano** con `rfdetr==1.3.0` (la
última que acepta Python 3.9) en un venv aparte: `claudio_v2/venv_rfdetr` (con `--system-site-packages`,
reusa torch 2.4.1; el entorno del sistema NO se tocó). Dataset `work/ds18_coco` (ds18 en COCO, hardlinks;
`src/yolo2coco.py`). Entrena a 640 (el Nano viene a 384), 40 épocas, early stopping 10
(`src/train_rfdetr.py`). Sale `best_componente_v23_RF.pth` limpio (sólo pesos + nombre de clase) y se
evalúa con `src/evaluate.py` (acepta `.pth` vía `RFDETRComoYOLO`), tags `V4_RF_nano_{005,marcelo}`.
Bug de rfdetr 1.3.0: con `multi_scale` en CPU falla ("Inference tensors cannot be saved for backward");
en GPU no. Ensayo de punta a punta en CPU OK. Disco C: 13 GB libres (7,6 GB son caché de pip).

**R2 terminó (23/09 03:33) → `best_componente_v19_R2.pt`.** @0,05: FN 8, FP 3319 (R: 1 / 3968).
@0,25: FN 17, FP 1803 (R: 12 / 1949). Menos FP pero PEOR recall. Pierde: 4 `PULS` + 2 amperímetros
`C45664E16` todos en UNA zona de nyw-un-01 (y≈78,2, x 55–64: fila pegada que funde), `C7AFF483F`
(test_2) y el PLC. Unión R+R2 con O probabilístico @0,25: FN 4, FP 2665 (`src/umbral_recall.py`,
analiza umbral máximo con recall 100% y reglas de fusión max / por / media).

**X_26p2 terminó (23/09 14:43, 100 épocas) → `best_componente_v20_X.pt`.** @0,05: FN 15, FP 3515;
@0,25: FN 24, FP 2188. Peor que R2 y que R. Falla en nyw-un-01 (11) y LU-UN-01 (3): amperímetros y
PULS, justo lo chico que P2 debía resolver. Sí detecta el `C7AFF483F`. Unión R2+X (O prob.)
@0,25: FN 9, FP 2704 → no sirve. Log: `claudio_v2/log_umbral_X.txt`.

**Z_26n terminó (24/09 01:28) → `best_componente_v21_Z.pt`.** @0,05: FN 8, FP 3088 (el de menos
FP); @0,25: FN 21, FP 1880. Uniones @0,25: R+R2 (O prob.) FN 4 FP 2665 (la mejor), R+Z FN 7,
R2+Z FN 8. Los FN@0,25 que se repiten en TODAS: `C7AFF483F` y `Seccionador-Rotativo` (test_2),
amperímetros `C45664E16` y el PLC. Log: `claudio_v2/log_umbral_Z.txt`.

**ds18 = ds13 corregido (22/09, aprobado por Tomas):**
- Sacados los 6 PAT/PE de `data/real_labels.json` (tsbe 16, plano3 23-24, plano 10, plano4 8,
  vyre 78). Backup: `claudio_v2/_para_borrar/real_labels_con_pat_22-09.json`. Como ya no están
  etiquetados, en los fondos sintéticos tampoco se blanquean: quedan como negativo.
- `build_all.manual_tiles`: las pseudo-etiquetas que caían >50% dentro de una zona neutra
  original del zip se descartan (472). Eran 492 cajas sobre símbolos blanqueados en ds13. Red
  final: se saca toda etiqueta de `m` con <2% de tinta (en ds18 no sacó ninguna).
- `build_all.real_job`: el plano se sortea con peso = cantidad de cajas (`REAL_UNIFORME=1` vuelve
  al sorteo viejo). Antes plano2 (4 cajas) tenía 121 tiles y vyre (86 cajas) 147.
- Auditoría: casi vacías 0,77% → 0,39%, no-ajusta 1,24% → 0,93%; en los `m`, 5,03% → 0,00%.
- Sin tocar: los tiles `m` tienen p50 de 66 px contra 46-51 de los planos reales.

### Decisión de Tomas (22/09): se usan LOS DOS modelos

**Producción = ensamble R + U** (`best_componente_v13_R.pt` + `best_componente_v16_U.pt`, ambos
en `claudio_v2/`). Se corren los dos y se unen las cajas. `src/ensamble.py` hace la MEDICIÓN;
todavía NO hay un script de inferencia de producción que corra los dos (ver "Lo que sigue").

```
union R + U @conf 0.05  ->  GT=3143  FN=0  recall=1.0000  FP=4870  prec=0.3922
union R + U @conf 0.25  ->  GT=3143  FN=6  recall=0.9981  FP=2565  prec=0.5502
```

**El 100% es a conf 0,05.** A 0,25 la unión pierde 6 (tres amperímetros, `C7AFF483F`,
`Seccionador-Rotativo`, `TRAFOIN`). Cuesta el doble de inferencia.

### Pedido de Tomas: YA LANZADO (22/09 18:35), pero sobre ds18 y con R2 primero (ver arriba)

> "quiero que entrenes con el dataset de r un yolo26n-p2 si es que existe un yolo small 11 y un
> solo 26 nano"

Tres entrenamientos, todos con **el dataset de R = `claudio_v2/work/ds13`** (ya auditado):

| corrida | modelo | qué hay |
|---|---|---|
| 1 | **yolo26n-p2** | EXISTE: `ultralytics/cfg/models/26/yolo26-p2.yaml` (ultralytics 8.4.41). No hay pesos p2 preentrenados: construir con `YOLO('yolo26n-p2.yaml').load('yolo26n.pt')` |
| 2 | **yolo11s** | `yolo11s.pt` NO está en el repo (sólo `yolo11n.pt` en la raíz); ultralytics lo baja si hay internet |
| 3 | **yolo26n** | `claudio_v2/yolo26n.pt` ya está |

A tener en cuenta:
- **YOLO26 es NMS-free** (asignación uno a uno): puede suprimir de más en filas de borneras o
  termomagnéticas idénticas pegadas. Mirarlo con `metrica_fusion.py` y las cajas dobles.
- **P2** agrega una cabeza de stride 4: apunta a lo que falla (amperímetro ~17 px, `C7AFF483F`
  angosto). Es más pesado; con la RTX 3060 Ti de 8 GB probablemente haga falta `--batch 8`.
  Igual para yolo11s.
- El proyecto dice YOLO **nano**: yolo11s es un experimento de comparación.
- Hooks nuevos en `src/train.py`: copiar el patrón de `R_full` pero con `DS = ds13` fijo (sin
  reconstruir) y `a.model` = el yaml/pt correspondiente. Evaluar SIEMPRE con `GT_V4=1`, conf 0,05,
  los 7 planos, y comparar con `rescore.py`. Cadena de cierre: copiar `cierre_U.sh`.
- 200 épocas de nano ≈ 9 h; small y p2, más.

### Tabla de todos los modelos (7 planos, 3.143 componentes, `GT_V4=1`, conf 0,05)

| modelo | archivo (`claudio_v2/`) | FN | recall | FP | prec | FN@0,25 | qué se probó |
|---|---|---|---|---|---|---|---|
| **R+U** | los dos | **0** | **1,0000** | 4870 | 0,392 | 6 | ensamble |
| **R** | `best_componente_v13_R.pt` | **1** | 0,9997 | 3968 | 0,442 | 12 | **mejor modelo solo** |
| U | `best_componente_v16_U.pt` | 2 | 0,9994 | 3867 | 0,448 | 17 | R + marco_punteado |
| R2 | `best_componente_v19_R2.pt` | 8 | 0,9975 | 3319 | — | 17 | yolo11n, receta R, ds18 |
| Z | `best_componente_v21_Z.pt` | 8 | 0,9975 | 3088 | — | 21 | yolo26n, ds18 |
| X | `best_componente_v20_X.pt` | 15 | 0,9952 | 3515 | — | 24 | yolo26n-p2, ds18, 100 ép. |
| Y | `best_componente_v22_Y.pt` | 6 | 0,9981 | 3702 | — | 19 | yolo11s, ds18 (pierde 5 amperímetros + PLC) |
| S | `best_componente_v14_S.pt` | 2 | 0,9994 | 4095 | 0,434 | 9 | R + caja_punteada + pat_negativo |
| W | `best_componente_v18_W.pt` | 3 | 0,9990 | 4956 | 0,388 | 16 | fine-tuning corto de R |
| T | `best_componente_v15_T.pt` | 5 | 0,9984 | 4169 | 0,430 | 17 | R + marco_punteado + pat_negativo |
| V | `best_componente_v17_V.pt` | 7 | 0,9978 | 4680 | 0,401 | 12 | fine-tuning largo de R |
| M | `best_componente_v8_M.pt` | 10 | 0,9968 | 3807 | 0,451 | 33 | — |
| N | `best_componente_v9_N.pt` | 10 | 0,9968 | 4083 | 0,434 | 25 | el mejor al empezar el 20/09 |
| Q | `best_componente_v12_Q.pt` | 15 | 0,9952 | 3710 | 0,457 | 43 | ternas R/S/T |

Qué pierde cada uno:
- **R**: sólo el `PLC` en marco punteado (nyw-un-01).
- **U**: `C7AFF483F` (test_2) y 1 amperímetro `C45664E16` (nyw-un-01).
- **T**: 3 `C4C456AEF` ("R" cursiva + "1300W", LU-UN-01), `C7AFF483F`, `Seccionador-Rotativo-01`.
- **V**: 3 amperímetros, 2 `C4C456AEF`, 1 `PULS`, `C7AFF483F`.
- **W**: 2 amperímetros, `C7AFF483F`.

Métricas propias en LU-UN-01 + nyw-un-01 (`src/metrica_fusion.py`):

| | amperímetros localizados | conf. mediana | PULS caja propia / tragados |
|---|---|---|---|
| N | 16/25 | 0,345 | 21 / 4 |
| Q | 23/25 | 0,146 | 6 / 18 |
| **R** | 24/25 | 0,670 | **25 / 0** |
| T | **25/25** | 0,696 | 25 / 0 |
| V | 21/25 | 0,323 | — |

Cajas que abarcan 2+ componentes (conf 0,25): N 74, Q 41, **R 39**.

### Conclusión de 5 entrenamientos: con UN modelo el piso es 1 FN

S, T, U, V y W: **cada uno que aprende el marco punteado pierde uno o dos símbolos chicos en otro
lado**. No hay una categoría que el modelo no entienda: los fallos son casos límite de tamaño y
forma — el más grande (marco 3× el típico), el más angosto (`C7AFF483F`, 0,244 × 0,880) y el más
chico (amperímetro, 0,221 × 0,244 ≈ 17 px). Es varianza entre corridas; el ensamble la compensa.

Atribución que quedó limpia:
- **`compose.marco_punteado` SÍ resuelve el PLC** (T, U, V, W lo detectan).
  `procsym.caja_punteada` NO (plan S): pasa por `sym_instance`, que lo escala al tamaño de un
  componente común, y el marco real es 3× más ancho.
- **`compose.pat_negativo` es lo que le hizo perder los `C4C456AEF` a T**: U (T sin pat) los
  conserva. Pero en S el pat subió la precisión de test1 de 0,639 a 0,706.
- **Fine-tuning desde R** (idea de Tomas): aprende el PLC en 1-1,5 h en vez de 9 h, pero sube los
  FP y, por el matching 1-a-1, le roban pareja a componentes reales. 30 épocas con mosaic (V): 7 FN.
  8 épocas, lr 0,0005, sin mosaic (W): 3 FN. Si se insiste: 3-4 épocas y lr aún menor.

### Qué es el PLC y dónde está

`nyw-un-01`, **al 83% del ancho y a media altura**, debajo de "BARRA DE ACOPLE", entre el medidor
T3 y los interruptores `I1` (400A) e `IG1` (250A). Recuadro **punteado** de 2,778 × 1,320 CAD
(x 203,45–206,228, y 61,517–62,837) con:

```
        PLC
LÓGICAS:
- TRANSF AUT.
- ACOPLE DE BARRAS
```

Es la lógica de control de la transferencia y el acople. **Imágenes en `CLAUDIO_AI/plc_nyw/`**
(`1_plc_zoom.png`, `2_plc_contexto.png`, `3_plc_ubicacion.png`, `4_plano_completo.png`).
**Ojo: esa fila la agregó la auditoría del 19/09 con seguridad "media"; no viene del CAD.** Las
reglas de etiquetado excluyen los marcos punteados VACÍOS; éste tiene nombre. Si Tomas decide que
no es componente, R sola queda en 0 FN sin entrenar nada.

### Decisiones de Tomas en esta tanda

- **"El PAT no es un componente."** No está en el GT (las marcas de auditoría eran "baja"). El
  modelo lo aprendía de las **pseudo-etiquetas** (generadas por un modelo que ya lo detectaba a
  0,94). Arreglo: `compose.pat_negativo` (P_PAT), que con P_PAT=0,30 cuesta recall.
- **"Las de reserva sí son un componente"** (termomagnéticas RES; ya en el GT).
- **Usar los dos modelos** (R+U).

### Calibración (la otra mitad del objetivo de Tomas)

R: confianza mediana de aciertos 0,942 contra 0,249 de falsos; el 95% de los aciertos está por
encima de 0,815. **Lo que falta: ~10% de los FP supera 0,905** y no se filtra por umbral. En test1,
8 de los 9 FP sobre 0,9 eran PAT.

### La medición estaba rota — todo lo medido antes del 20/09 no vale

1. Cajas infladas por ATTRIB en LU-UN-01 y nyw-un-01 (ver historial abajo).
2. 41 duplicados + 1 caja de área cero (`*U57`) introducidos por la auditoría del 19/09. Con
   matching húngaro 1-a-1, cada duplicado es un FN que ningún modelo puede evitar.
3. Dos umbrales: Marcelo a 0,20, el resto a 0,05. Escondió que el arreglo del amperímetro
   funcionaba. **Evaluar siempre al mismo umbral.**

### Versiones del GT

| variable | archivo | qué tiene |
|---|---|---|
| (ninguna) | `<n>.csv` | original, incompleto y con cajas infladas |
| `GT_V2=1` | `<n>_v2.csv` | + los faltantes de la auditoría |
| `GT_V3=1` | `<n>_v3.csv` | + cajas ceñidas sin ATTRIB |
| **`GT_V4=1`** | **`<n>_v4.csv`** | **+ sin duplicados, cajas de auditoría ajustadas. ES EL BUENO** |

Totales v4: test1 108, test_2 122, fl_un_02 372, tsss_2 206, EZE4077 68, LU-UN-01 1225,
nyw-un-01 1042 = **3.143**. Muestra al azar: 18/18 cajas correctas. Los originales están intactos.

### `src/snap_audit_gt.py` — cómo quedó y sus trampas

Genera `_v4.csv` desde `_v3`/`_v2`. Para cada fila `AUDIT-*`: busca el rectángulo dibujado que la
encierra (primero con líneas sin unir, exigiendo 70% de tinta en los bordes; después uniendo
trazos punteados, exigiendo 35%); si no, contorno en el punto; si no, mancha cercana (radio 0,8
lados, rechazando las que caen sobre TEXTO del DXF); si nada, deja la caja original. Deduplica
contra TODAS las filas (centro dentro de la otra caja, o distancia < 0,5 diagonal media) y saca las
cajas de área cero. Tarda ~15 min (renders de 77 MP).

Trampas (costaron ~12 iteraciones; están comentadas en el código):
- **CAUSA RAÍZ: el render dibuja los CONTORNOS en gris claro (167-211) y el TEXTO en negro.**
  Umbral de tinta `TINTA = 230`, nunca 128: con 128 los recuadros son invisibles y todo termina
  enganchado a las palabras.
- Iterar sólo el modelspace pierde la geometría de adentro de bloques (borró 123 componentes).
  **Si el script no encuentra algo, eso no prueba que no esté.**
- Un umbral de tinta relativo al área descarta símbolos chicos: tiene que ser absoluto.
- Un punto en el hueco entre dos recuadros queda "encerrado" por los vecinos: se filtra por
  proporción (≤ 6:1) y validación de bordes.
- Unir los trazos punteados de entrada rompe el caso normal (55 → 40 rectángulos en LU-UN-01):
  por eso son dos pasadas.
- En un marco punteado, los trazos sueltos arman un rectángulo falso INTERIOR que pasa una
  validación floja porque el texto cruza sus bordes: por eso la pasada sin unir exige 70%.
- Las ediciones repetidas dejaron **funciones duplicadas** (la segunda pisaba a la primera);
  limpiado el 22/09. Si se vuelve a editar: `grep -n "def recuadro_en" src/snap_audit_gt.py`.
- **Método: mirar sólo los fallos es un sesgo.** Verificar con muestra al azar.

### Planes y cómo se lanzan

Hooks en `src/train.py`: `R_full` (ds13), `S_full` (ds15), `T_full` (ds16), `U_full` (ds17),
`V_ft` y `W_ft` (fine-tuning desde R sobre ds17). `ds14` quedó en `_para_borrar/ds14_sin_pat`.

Variables de generación (entorno; las leen `compose.make_tile` / `procsym`): `P_PULS`
(contactor+pulsador, 0,30), `P_PAT` (negativo PAT), `P_MARCO` (marco punteado grande, 0,20),
`P_CELDAS` (0,25), `P_TERNA` (0,35), `GENS_NOMBRE_SIMPLE=1` (caja_nombre a peso simple),
`GENS_SIN_PUNTEADA=1` (saca caja_punteada de GENS).

Variables de entrenamiento nuevas en `train.py`: `LR0`, `LRF`, `WARMUP`, `MOSAIC`, `CLOSE_MOSAIC`.

Cadena de cierre: `claudio_v2/cierre_<plan>.sh` espera `work/runs/<plan>/best_limpio.pt`, copia a
`best_componente_vNN_X.pt`, evalúa los 7 planos con `GT_V4=1` a conf 0,05 y corre rescore +
métricas. Resultado en `log_cierre_<plan>.txt`.

### Herramientas nuevas

| archivo | para qué |
|---|---|
| `src/rescore.py` | re-puntúa evaluaciones ya corridas contra otro GT, SIN GPU: `GT_V4=1 py -3 src/rescore.py V3 V4` |
| `src/ensamble.py` | recall de la unión de dos modelos: `GT_V4=1 py -3 src/ensamble.py V4_v13_R V4_v16_U` |
| `src/snap_audit_gt.py` | GT v4 |
| `src/auditar_ds.py` | auditoría de dataset (§8). Sano: 0 anidadas, 0 minúsculas, p50 ~46-50 px |
| `src/metrica_fusion.py` | amperímetros localizados y PULS con caja propia / tragados |
| `src/metrica_confianza.py` | separación de confianza TP vs FP |
| `src/metrica_tablas.py` | detecciones sobre celdas de planilla (Q 25, N 17: 0,9%, marginal) |
| `compose.puls_contactor`, `compose.marco_punteado`, `compose.pat_negativo` | generadores nuevos |

### Trampas operativas de este entorno (Windows + Git Bash)

- **Nunca `... | head` sobre un entrenamiento**: el SIGPIPE lo mata. Lanzar con
  `nohup py -3 -u src/train.py --name X > log_X.txt 2>&1 &`.
- `cmd //c run_X.bat` desde Git Bash no encuentra el .bat: lanzar `train.py` directo.
- Las cadenas `sh` en segundo plano a veces mueren en horas largas: hacerlas idempotentes.
- Hay un "reaper" que mata procesos de fondo cuando la RAM baja (~3 GB libres con un
  entrenamiento corriendo). No lanzar renders pesados en paralelo.
- La evaluación de LU-UN-01/nyw-un-01 puede morir en silencio por memoria: **si `rescore` muestra
  un GT distinto de 3143, faltan planos**.
- Los logs de ultralytics usan `\r`: leerlos con `tr '\r' '\n' < log`.
- Los heredocs con comillas raras se rompen en Git Bash: escribir el script a un archivo.

### Lo que sigue

1. **El pedido pendiente: yolo26n-p2, yolo11s y yolo26n sobre ds13** (arriba).
2. **Script de inferencia de producción con los dos modelos** (R+U): hoy sólo existe la
   medición. Correr `detectar.py` con cada peso y fusionar (NMS + supresión de anidadas, como
   `evaluate.fuse`). Actualizar el ZIP del compañero de Tomas **con el arreglo del giro** (§5.7.1,
   vale 340 componentes).
3. Reintroducir `pat_negativo` con P_PAT 0,10-0,15 y medir que no cueste recall.
4. Regenerar las pseudo-etiquetas con R (son la fuente del PAT).
5. Auditar el GT de los folios QET (1.560 cajas) con el chequeo de cajas infladas.
6. Re-detectar los 4 planos que estaban mal rotados.

### Historial del 20/09 (ya resuelto; se deja como referencia)

### 🔴 EL GT DE LU-UN-01 Y NYW-UN-01 TIENE LAS CAJAS INFLADAS (20/09) — lo más grave

Salió de una pregunta de Tomas: "¿cómo sabés que tiene que ser tomada completa?". No lo sabía;
al ir a buscar evidencia apareció esto, que invalida buena parte de lo medido.

**Las cajas del GT de esos dos planos incluyen el texto de atributo del bloque** (`12A`, `AC3`,
`S1`). Es el problema que el proyecto ya tenía identificado — el GT correcto excluye
`ATTRIB/ATTDEF/TEXT/MTEXT` — pero en estos dos planos no se aplicó.

| Bloque | n | GT | bbox real del DXF | área |
|---|---|---|---|---|
| `S-M-0-A` | 18 | 0,524 × 0,594 | 0,238 × 0,408 | **3,21×** |
| `CONTACTOR` | 70 | 0,930 × 1,258 | 0,569 × 0,875 | **2,35×** |
| `SECC-BC-FUS` | 47 | 0,788 × 0,980 | 0,382 × 0,875 | **2,31×** |
| `SECCIONADOR` | 28 | 0,673 × 0,875 | 0,339 × 0,875 | **1,98×** |
| `TM-DIN` | 490 | 0,759 × 0,913 | 0,518 × 0,913 | **1,46×** |
| `MEDIDOR` | 117 | 0,315 × 0,527 | 0,315 × 0,527 | 1,00× |
| `A$C45664E16` | 16 | 0,221 × 0,244 | 0,221 × 0,244 | 1,00× |

Los que tienen texto de atributo están inflados; los que no, exactos. **`TM-DIN` e `INT-DIF`
son 686 de las 1143 cajas de LU-UN-01**, así que no es un caso de borde.

**Y el centro queda corrido**, que es lo que rompe el matching (asigna por distancia de
centros, no por IoU):

| Bloque | desvío del centro | ancho del símbolo |
|---|---|---|
| `CONTACTOR` | **0,263** | 0,569 |
| `SECC-BC-FUS` | **0,210** | 0,382 |
| `SECCIONADOR` | 0,167 | 0,339 |
| `TM-DIN` | 0,120 | 0,518 |

Medio ancho de desvío. **Un modelo que ciñe bien la caja al símbolo queda lejos del centro del
GT y se cuenta como fallo.** Ésa es la causa de los "6 contactores perdidos por Q": Q ciñe, N
pone cajas más grandes que alcanzan el centro corrido. **Q es penalizado por encuadrar mejor.**

**YA CORREGIDO (20/09).** `src/cenir_gt.py --aplicar` recalcula la caja de cada fila que
corresponde a un INSERT, tomando el bbox del bloque sin `ATTRIB/ATTDEF/TEXT/MTEXT`. No regenera
el GT desde cero: corrige las cajas y deja intactas las demás filas (geometría suelta y las 221
que agregó la auditoría).

| | filas | cajas ceñidas | desvío del centro (med / max) |
|---|---|---|---|
| LU-UN-01 | 1240 | **776** (63%) | 0,120 / 0,467 |
| nyw-un-01 | 1054 | **616** (58%) | 0,120 / 0,385 |

Salida: `<nombre>_gt_v3.csv`, que lleva **las dos correcciones** (completo + ceñido).

**Las tres versiones del GT y cómo usarlas** (`evaluate.py` elige en cascada):

| variable | usa | qué tiene |
|---|---|---|
| (ninguna) | `<n>.csv` | el original, incompleto y con cajas infladas |
| `GT_V2=1` | `<n>_v2.csv` | + los 221 componentes que faltaban |
| `GT_V3=1` | `<n>_v3.csv`, si no `_v2` | + las cajas ceñidas — **es el bueno** |

Con `GT_V3=1` los planos que no tenían el problema caen solos al `_v2`.

**Respuesta a la pregunta de Tomas:** `CONTACTOR` y `PULS` son dos bloques INSERT independientes
en el DXF (70 y 16 instancias; el bloque `CONTACTOR` contiene 3 ATTDEF y 6 LWPOLYLINE, y NO
contiene al PULS). Son **dos componentes distintos**, no uno. El anidamiento no es real: aparece
sólo porque la caja del contactor se infla con su texto hasta tragarse el pulsador.

### Consecuencia: las etiquetas anidadas del GT

**El hallazgo más importante sobre la medición.** Tomas miró un contactor y dijo que le parecía
que el componente se debía tomar entero (el rectángulo con diagonal + el triángulo). Tenía
razón, y el GT es el que se contradice.

En `LU-UN-01` y `nyw-un-01` hay **25 pares de cajas anidadas en el GT**, todas del mismo tipo:

| | cajas | anidadas | qué dentro de qué |
|---|---|---|---|
| LU-UN-01 | 1143 | 16 (1,4%) | `PULS` dentro de `CONTACTOR` |
| nyw-un-01 | 969 | 9 (0,9%) | `PULS` dentro de `CONTACTOR` |
| fl_un_02 | 363 | 0 | — |
| test_2 | 117 | 0 | — |

La caja de `CONTACTOR` **ya contiene** al triángulo (y además el texto `12A AC3 S1`), y encima
el triángulo lleva su propia etiqueta `PULS`. O sea, el GT pide DOS cajas superpuestas ahí.

**Consecuencia medida:**

| | pierde de los 16 contactores con PULS anidado | detecciones que pone dentro |
|---|---|---|
| N | **0** | mediana 2,0 |
| Q | **6** | mediana 1,5 |

**Q está siendo castigado por hacer lo que le pedimos.** Todo el trabajo del proyecto apunta a
"una caja por componente, sin anidadas" (§5.4, la red de seguridad de `compose.py`); Q lo
aprendió mejor que N, pone una sola caja, y el matching 1-a-1 deja una de las dos referencias
sin pareja. N pone dos cajas superpuestas — lo que NO queremos — y por eso "acierta".

**Esto invalida parte de la comparación N vs Q**: de los 10 `CONTACTOR` que Q pierde, al menos 6
son artefacto del GT, no un empeoramiento real.

**(RESUELTO 20/09: con las cajas ceñidas del GT v3 el anidamiento desapareció; ya no hace falta decidirlo.)** ¿el pulsador dentro del contactor es un componente aparte o
parte del contactor? Si es parte, hay que sacar los 25 `PULS` anidados del GT y volver a medir.

### Además: `caja_nombre` se comió las tablas

Primero se sospechó que `terna_rst` partía el contactor tripolar en polos. **Se midió y es
falso**: contactores donde Q saca 2+ cajas y N saca 1 o 0 hay **cero**, y la distribución de
detecciones por contactor es casi idéntica entre los dos.

Se buscaron entonces los contactores concretos (mismo matching húngaro que `evaluate.py`) y
aparecieron **los 6 en la misma fila**, y=57.435, espaciados regular. Al renderizar esa fila
con las detecciones de los dos modelos encima se ve la causa:

**Q marca como componente casi todas las celdas de la tabla de circuitos** (`C7`, `C21`, `C23`,
`C25`, `C27`, `C29`, `C31`, `C33`, `C35`, `C37`, `C39`, `C53`); N marca bastantes menos. Son
falsos positivos, y como el matching es 1-a-1 global, compiten por las asignaciones y dejan
contactores sin pareja.

**La causa es el refuerzo de `caja_nombre`** (§6.2.bis): se le enseñó que "recuadro con texto
adentro = un componente" con peso doble y 20 rótulos, y el modelo generalizó de más hasta las
celdas de la planilla de circuitos, que NO son componentes. El negativo que debía contrarrestar
esto (`tabla_celdas`, P_CELDAS=0.18) no alcanzó frente al refuerzo.

**Para el plan R:** subir `P_CELDAS` y/o bajar el peso de `caja_nombre`, y medir la tasa de
detecciones que caen sobre celdas de tabla como métrica propia. El cambio de rótulos arregló
un problema (los recuadros de equipo de los PDF de Marcelo) y creó otro.


---

## 1. Qué se quiere lograr

Detectar **componentes eléctricos en planos unifilares** (DXF y PDF) con un detector YOLO nano
de **una sola clase**: `componente`.

Esto es la **etapa 1** de un sistema de dos etapas:

| Etapa | Qué hace | Dónde vive |
|---|---|---|
| **1 — detección** | encontrar *dónde* hay un componente y devolver su caja | este repo |
| **2 — clasificación** | decir *qué* componente es cada caja | **otro repo** |

Por eso importan dos cosas distintas y a veces en tensión:

1. **Recall 100%** — no perder ningún componente. Es la prioridad declarada por Tomas.
   Los falsos positivos **se aceptan**: la etapa 2 puede descartarlos.
2. **Cajas limpias** — que una caja encierre **un** componente. Las cajas que abarcan dos
   componentes son el problema que más le complica la etapa 2, porque no hay forma de
   clasificar una caja que contiene dos cosas.

### El dueño del proyecto

Tomas. Habla español, quiere respuestas **directas y concisas**. Es el experto del dominio:
cuando dice que un símbolo es o no es un componente, **tiene razón** y hay que corregir la
heurística, no discutirle (ver §7, "errores que ya cometí").

---

## 2. Reglas del proyecto (no negociables)

Están también en `CLAUDE.md`, pero se repiten acá porque son fáciles de pisar:

- **Hablar en español**, directo y conciso.
- **Prioridad: recall 100%.** Los FP se aceptan.
- **Evaluar siempre con los DXF con texto**, usando `claudio_v2/src/evaluate.py`.
- **Los planos externos y los folios de QElectroTech son SOLO para test.**
- **Nada de borrado definitivo**: lo que se descarta se mueve a `_para_borrar/`.
- **No cortar entrenamientos en curso sin avisar.**
- **Mostrar grillas numeradas** antes de cambiar etiquetas o entrenar con datos nuevos.
  Tomas revisa símbolo por símbolo y decide; esto es lo que evitó varios errores.

---

## 3. Estado al 19/09/2026 (HISTÓRICO — el estado vigente está en §0)

### El mejor modelo: `claudio_v2/best_componente_v9_N.pt` (plan N)

Medido sobre 7 planos con verdad de terreno, **2.964 componentes**:

| Plano | GT | FN | Recall |
|---|---|---|---|
| test1 | 98 | 0 | 100% |
| test_2 | 117 | 2 | 98,3% |
| fl_un_02 | 363 | 0 | 100% |
| tsss_2 | 206 | 0 | 100% |
| EZE4077 | 68 | 0 | 100% |
| LU-UN-01 | 1143 | 2 | 99,8% |
| nyw-un-01 | 969 | 6 | 99,4% |
| **Total** | **2964** | **10** | **99,66%** |

- **IoU global: 0,843**
- **Cajas que abarcan 2+ componentes a conf 0,25: 1** (en TSSS_2: 0)

### Evolución de los modelos

| Modelo | Archivo | FN totales | IoU | Qué aportó |
|---|---|---|---|---|
| G | `best_componente_v3.pt` | 40 | 0,736 | base |
| J | `best_componente_v6_J.pt` | 36 | 0,714 | fusible y ojo separados |
| K | `best_componente_v6_K.pt` | 40 | 0,810 | sprite del diferencial corregido |
| L | `best_componente_v7_L.pt` | 36 | **0,861** | sin etiquetas anidadas, cajas ceñidas |
| M | `best_componente_v8_M.pt` | 29 | 0,832 | pseudo-etiquetas (con K: **mal**) |
| **N** | **`best_componente_v9_N.pt`** | **10** | 0,843 | pseudo-etiquetas con L |
| O | (entrenando) | ? | ? | cobertura de símbolos chicos |

---

## 4. Cómo funciona el pipeline

### Inferencia (`claudio_v2/detectar.py`, autocontenido)

```bash
py -3 detectar.py plano.dxf --pesos best_componente_v9_N.pt --conf 0.20
py -3 detectar.py carpeta_con_planos --pesos best_componente_v9_N.pt --conf 0.20
py -3 detectar.py plano.pdf --pesos best_componente_v9_N.pt --conf 0.20
```

1. **Render** del DXF con su texto (`ColorPolicy.BLACK` sobre blanco) + realce de trazos finos.
2. **Limpieza de máscaras**: se sacan WIPEOUT, HATCH sólidos blancos y el `bg_fill` de los
   MTEXT. Con la política BLACK esos elementos se pintan negros y pueden tapar el plano entero
   (visto en un plano real: 57% de la imagen en negro).
3. **Escala automática**: la mediana de altura de texto se lleva a **11 px**. Es lo que permite
   usar el mismo modelo en planos de escalas distintas. Con menos de 5 textos hay que pasar
   `--ppc` (DXF) o `--dpi` (PDF).
4. **Auto-enderezado (PDF)**: si más del 70% del texto queda vertical, el plano está de costado;
   se rota antes de inferir y las cajas se rotan de vuelta. Sin esto, `Plano 3` daba 93
   detecciones y con esto da 188.
5. **Inferencia por tiles** de 640 con paso 320, a dos escalas (1.0 y 1.6), umbral interno 0.05.
6. **Fusión**: NMS + supresión de anidadas, promediando las cajas del mismo objeto (box voting)
   en vez de quedarse con la de mayor confianza (que suele ser la más chica).
7. Recién al final se aplica el `--conf` pedido.

### Entrenamiento

```bash
cd claudio_v2
run_N.bat          # cada plan tiene su .bat; el hook vive en src/train.py
```

`src/train.py` tiene un hook por plan (`L_full`, `M_full`, `N_full`, `O_full`…) que construye
el dataset si no existe y lanza el entrenamiento. Cada plan usa su propio `work/dsN`.

### Generación del dataset (`src/build_all.py` + `src/compose.py`)

Tres fuentes:

| Fuente | % de las cajas | Qué es |
|---|---|---|
| Sintético (`compose.py`) | ~90% | tiles 640×640 armados con sprites de biblioteca |
| Zips manuales (Roboflow) | ~7% | capturas de planos reales anotadas a mano |
| Planos reales (`real_labels.json`) | ~3% | recortes de 6 planos con GT |

---

## 5. Los hallazgos que más movieron la aguja

Esto es lo más valioso del documento: **qué estaba mal y cómo se encontró**.

### 5.1. Etiquetas que enseñaban a ignorar componentes (el más grande)

Los zips manuales estaban anotados **para un solo tipo de componente**: en una captura que
muestra medio tablero se etiquetó sólo el interruptor motorizado (o la fotocélula) y todo el
resto quedó sin etiqueta. **El 32% de los componentes de esos tiles no tenía etiqueta**, y cada
uno es un negativo falso que le enseña al modelo que ese componente NO es un componente.

**Solución**: `src/pseudo_zips.py` completa las etiquetas con el modelo. Lo firme (conf ≥ 0,80)
entra como etiqueta, lo dudoso (0,12–0,80) se blanquea como zona neutra.

**Detalle crítico**: hay que generar las pseudo-etiquetas con el modelo que **mejor encuadra**,
no con el que tengas a mano. Con K (IoU 0,810) el modelo heredó su mal encuadre y empeoró
(M: IoU 0,832, 5 cajas dobles). Con L (IoU 0,861) salió bien (N: 10 FN, 1 caja doble).

### 5.2. Sprites que no eran componentes

La biblioteca tenía la palabra "CT" renderizada como imagen, marcos vacíos, y 65 sprites sin
un solo píxel de tinta. Cada uno enseñaba una caja alrededor de texto o de aire.

**Solución**: `EXCLUIR_SIM` en `compose.py` (lista revisada por Tomas) + descarte de sprites
vacíos. Biblioteca: 964 → 833 sprites.

### 5.3. Cajas más grandes que el símbolo

Los generadores que dibujan a mano (`spm_branch`, `fusible_solo`, `tablero_row`) armaban la
caja con márgenes fijos de 3-4 px sobre la geometría. En un símbolo de 20 px eso es un tercio
de aire. El 43-44% de sus cajas tenía más de 18% de holgura.

**Solución**: `cenir()` ajusta la caja a la tinta **después** de dibujar. Dataset: 8,9% → 1,2%.

**Ojo**: para símbolos que un cable atraviesa de lado a lado, `cenir()` no puede achicar en ese
eje. Ahí hay que usar el contorno exacto del símbolo (ver `spm_branch`).

### 5.4. Etiquetas anidadas

`fila_densa` se salteaba el chequeo de solapamiento por completo y dibujaba encima de lo que ya
habían puesto otros generadores. 58 pares anidados cada 1500 tiles.

**Solución**: respeta las cajas previas (pero sigue salteando el chequeo **entre sus propios
símbolos**, que es su razón de ser) + red de seguridad al final de `make_tile`. Ahora: 0.

### 5.5. Símbolos chicos subrepresentados

`sym_instance` escalaba los símbolos a **mínimo 1,8 veces** la altura del texto. El amperímetro
de LU-UN-01 y nyw-un-01 mide **1,22 veces** — o sea caía en un rango que el dataset **nunca
generaba**. El modelo lo encontraba (IoU 0,82-0,93) pero con confianza 0,06-0,17.

**Solución** (plan O, en curso): tramo chico explícito 1,10–1,95× con probabilidad 0,18.
Cajas de menos de 16 px: 5,2% → 7,3%.

**Cuidado**: al hacer esto no hay que bajar el piso del tramo **grande** — eso corre la mediana
de todo el dataset (pasó: de 50 px a 32 px, cuando los planos reales están en 51).

### 5.6. GT con cajas mal puestas

Varias veces un "falso negativo" resultó ser un error del GT:

- El **punto de inserción de un bloque DXF puede estar lejos del dibujo**. Hizo ver 14 FN
  fantasma en los SPM.
- Los **ATTDEF/ATTRIB inflan el bbox** de un bloque. La caja correcta es la del **dibujo**,
  excluyendo `ATTRIB/ATTDEF/TEXT/MTEXT`.
- El GT de `test_2` tenía **las 117 filas sin caja**, sólo el punto. Se regeneró el 19/09.

**Antes de perseguir un FN, verificá que el GT esté bien.**

---

## 5.7. La auditoría con un modelo de visión (19/09) — el hallazgo más grande

Sobre los PDF de Marcelo no hay verdad de terreno, así que no se sabía ni cuántas cajas estaban
mal ni cuántos componentes se perdían. Se armó material en `revision_ia/` y lo auditó un modelo
de visión externo (Antigravity). El pipeline está en `src/preparar_revision.py` (genera) y
`src/leer_revision.py` (lee). La clave del diseño: **no se le pide que detecte desde cero**, que
es donde alucina, sino dos preguntas cerradas — cuántos componentes hay dentro de una caja ya
dibujada, y qué quedó sin marcar en una celda donde lo detectado ya está pintado de azul.

**Los faltantes se verificaron a mano antes de creerles**: se dibujaron sobre las celdas y se
miraron. Son reales (termomagnéticas `2x16A` sin caja, con vecinas idénticas sí marcadas).

### 5.7.1. El auto-enderezado rotaba para el lado equivocado

Al mirar esas celdas se vio que **el texto se leía al revés**: el plano estaba invertido.
`detectar.py` detectaba bien que el plano estaba de costado pero **rotaba siempre en sentido
horario**, sin mirar hacia dónde apunta el texto. Para un `dir` de `(0,1)` el giro horario deja
el contenido a 180°. El modelo nunca vio símbolos invertidos.

| | marcadas | faltan | recall |
|---|---|---|---|
| Planos mal rotados (Plano 1/2/3, Planos tableros) | 551 | 340 | **61,8%** |
| Planos bien orientados | 1.887 | 196 | **90,6%** |
| Total | 2.438 | 536 | 82,0% |

Cuatro planos de doce concentraban el 63% de todo lo perdido. **Arreglado**: ahora el sentido se
elige por el signo de `dy` y `px2pdf` deshace los dos casos (`giro` 90 y 270). Verificado en los
12 planos. **Este arreglo hay que pasarlo al repo de inferencia del compañero de Tomas.**

Cuidado con una tentación: al principio se subió también el mínimo de texto de 5 a 15 líneas,
con el argumento de que en un unifilar las etiquetas de circuito van verticales. Es cierto en
general, pero **rompía Plano 1/2/3**, que tienen 5 líneas extraíbles y están de costado de
verdad. Se revirtió: el bug era sólo el sentido.

### 5.7.2. Las ternas R/S/T (lo que queda del lado del modelo)

De los 196 faltantes en planos bien orientados, **el 73% son fusibles (92) y lámparas (51)**. Al
mismo tiempo, las cajas que abarcaban varios componentes eran justo *tres fusibles + tres
lámparas R/S/T dentro de una sola caja*. Es el mismo problema por los dos lados: en la terna el
detector saca **una** caja para las tres columnas, y eso cuenta como una acertada y dos perdidas.

Es exactamente lo que Tomas pidió resolver el primer día. De ahí sale `terna_rst` en
`compose.py` (`P_TERNA`, plan Q): 2-4 columnas del mismo símbolo colgando de una barra, cada una
con su cable, la letra de fase debajo y muy seguido un segundo piso (fusible arriba, lámpara
abajo). **Cada símbolo lleva su caja**: una terna de dos pisos son seis cajas, no una.

No se agregó rotación de 180° al generador a propósito: la causa de los símbolos invertidos era
el bug de `detectar.py`, ya arreglado, y meter símbolos de cabeza en el dataset sólo agregaría
falsos positivos.

### 5.7.3. Un cuarto de las cajas corta el símbolo

La tarea A también midió el encuadre: **27,7% `corta`** (deja parte del símbolo afuera), 12,7%
`sobra_cable`, 4,2% `sobra_texto`. No se estaba midiendo con nada y le complica la etapa 2 tanto
como las cajas dobles. Sin atacar todavía.

---

## 5.8. El plan O falló: los símbolos chicos NO eran el problema (20/09)

Hipótesis de O: el amperímetro `A$C45664E16` se perdía (7 de los 10 FN de N) porque el dataset
tenía sólo 13% de cajas de menos de 20 px contra 27-32% en los planos reales. El remedio fue un
tramo chico explícito en `sym_instance` (1,10-1,95 veces la altura de texto, 18% de las veces).

**Resultado, a la misma confianza y el mismo GT:**

| Plano | conf | N | O |
|---|---|---|---|
| test1 | 0,05 | 0 | 0 |
| test_2 | 0,05 | 2 | 1 |
| fl_un_02 | 0,05 | 0 | 0 |
| tsss_2 | 0,05 | 0 | 0 |
| EZE4077 | 0,20 | 0 | 0 |
| LU-UN-01 | 0,20 | 7 | 16 |
| nyw-un-01 | 0,20 | 10 | 22 |
| **Total FN** | | **19** | **39** |

O **duplicó** los falsos negativos. Y el detalle es peor que el total:

| Bloque | N | O |
|---|---|---|
| `A$C45664E16` (el amperímetro, el objetivo de O) | 15 | **21** |
| `PULS` | 1 | **7** |
| `A$C4C456AEF` | 0 | **5** |
| `TRAFOIN` | 0 | **3** |
| `CONTACTOR` | 0 | **1** |

No sólo no arregló el amperímetro: lo empeoró, y rompió tres bloques que N detectaba perfecto.
**Llenar el dataset de símbolos diminutos degrada los medianos sin ganar nada en los chicos.**

La causa del amperímetro sigue sin explicar. Lo que sí quedó descartado es que sea falta de
ejemplos de ese tamaño.

### Cómo se revirtió (sirve si hay que reconstruir otra base)

`compose.py` no está en git, así que no había versión anterior a la que volver. Se reconstruyó
**midiendo `work/ds9`**, que es la base de N y quedó en disco. Comparando sólo los tiles
sintéticos (`p*.txt`, que son los que dependen del generador; los `m*` manuales son idénticos
entre datasets y tapan la diferencia si se mezclan):

| | p50 | <20 px | <16 px |
|---|---|---|---|
| ds9 (base de N) | 45,0 px | 12,8% | 5,3% |
| ds10 (base de O) | 37,0 px | 16,3% | 6,9% |

Con un barrido sobre `sym_instance` se encontró que `lo, hi = 1.80, 12.0` sin tramo chico
reproduce ds9 (44 px, 11,4%). **El margen proporcional del `overlap` se dejó**: revertirlo
también bajaba la mediana a 39 px, o sea que ése no era el problema.

---

## 5.9. El amperimetro era un cuadradito con una 'A' en cursiva (20/09)

`A$C45664E16` es el falso negativo mas grande que queda: **15 de los 19 de N**. Durante dos
planes se lo trato como "simbolo chico subrepresentado" y el plan O se quemo nueve horas de GPU
en esa hipotesis (§5.8).

**Nunca se lo habia renderizado para mirarlo.** Al hacerlo resulto ser un cuadradito con la
letra **`A` en cursiva**, pegado a un cable horizontal, al lado de un triangulo con un punto.
Mide 0,221 x 0,244 unidades CAD, o sea **17,9 px** a la escala del detector — que no es chico:
el dataset ya tenia 11-13% de cajas de ese tamano. Por eso O no podia funcionar.

Lo que si estaba mal representado era el caso en si. En `procsym.caja_letras`:

| | Antes | Ahora |
|---|---|---|
| Cajas con **una sola letra** | 23,5% de `caja_letras` | **46,8%** |
| En cursiva (`FONT_ITALIC`) | nunca | 35% |
| Con el cable pegado al costado | nunca | 45% |

El del cable es probablemente el que mas pesa: en el plano el cuadrito **siempre** aparece con
el cable pegado, y el generador lo dibujaba suelto y aislado.

**Leccion, que es la misma de §5.8 y ya van dos:** antes de teorizar sobre por que se pierde un
componente, hay que **renderizarlo y mirarlo**. Las dos veces la hipotesis a ciegas fue falsa y
las dos veces la respuesta estaba a la vista apenas se abrio la imagen.

---

## 6. Problemas abiertos

### 6.1. Cajas que abarcan varias borneras (lo que más molesta a la etapa 2)

En los planos de Marcelo se ven recuadros que se tragan media fila de borneras. Es el problema
que Tomas quiere resolver. `fila_densa` existe justamente para esto pero no alcanza.

**19/09, dato nuevo: en el dataset no hay NI UNA bornera real.** Las 13 fuentes de
`zips_merged.json` se parten en dos grupos y `build_all.py:157` descarta uno entero:

```python
if 'v2i' in key or 'v3i' in key: continue   # exportes Roboflow 'Resize 640x640 (Stretch)'
```

Eso saca 218 capturas de 1.535: `Diferencial.v3i` (86), `Termomagnetica.v2i` (81) y
`Bornera.v2i` (51). Para las dos primeras no importa — los zips nuevos traen 1.499 y 1.694
cajas de lo mismo, diez veces más y mejor etiquetadas. Pero **`Bornera.v2i` era la única
fuente de borneras**, así que las borneras salen solamente del generador procedural
(`bornera_din`, `bornera_circ`, `bornera_strip`, `fila_densa`). El modelo nunca vio una real.

Ojo antes de correr a recuperarlas: **no alcanzarían**. Son recortes apretados de UNA bornera
sola (la caja ocupa el 82% de la imagen), no filas, así que no enseñan a separar una fila en
sus partes, que es justo lo que falla. Y vienen con el criterio de etiquetado viejo: la caja se
traga el símbolo, el cable entero y el texto ("25A 30mA" queda adentro). Si se recuperan hay
que volver a ceñirlas y además deshacer el stretch, que es lo que motivó la exclusión.

Camino más prometedor: reforzar `fila_densa` (que ya etiqueta bornera por bornera) y medir,
en vez de meter 51 capturas deformadas con etiquetas de otro criterio.

### 6.2. Falsos negativos vistos por Tomas en los PDFs (19/09, sin resolver)

- **Descargadores triangulares**: 4 iguales en fila, detecta 2.
- **"Controlador para Transferencia Automática"**: caja rectangular con texto en 3 líneas, no
  la detecta. Existe un generador para esto (`caja_nombre` en `procsym.py`) — revisar por qué
  no alcanza.

### 6.2.bis. Rótulos de varias líneas: la causa de las cajas mal puestas (19/09, arreglo en el plan P)

Tomas marcó que "hay muchas bounding boxes mal". Sobre las 2.106 detecciones de N en los 12
PDFs de Marcelo, con criterios auto-consistentes (sin GT): **36 englobadoras, 13 solapadas,
71 gigantes**. Mirando los 12 peores casos uno por uno, casi todas caen sobre **lo mismo**:
recuadros con texto de 2-3 líneas — `Iso-Gard IG6`, `Fuente 24 VCC 2 A`, `UPS 6 kVA 15 min`,
`UPS 2.5 kVA 30 min`, `VigilOhm IM400` — donde el modelo pone **3 o 4 cajas superpuestas y
desalineadas** en vez de una sola. No es que no los vea: no sabe dónde termina el recuadro.

Causa en el dataset: `caja_nombre` existía pero con **8 rótulos fijos, ninguno parecido a los
de estos planos**, y pesaba 1 de 9 generadores (~11% de los símbolos procedurales). Y faltaba
el negativo complementario: **texto de varias líneas SIN recuadro no lleva ninguna caja**. Sin
ese par el modelo sólo ve "varias líneas juntas" y no tiene con qué decidir.

Arreglo (ya puesto, entra en el ds11 del plan P):

- `procsym.caja_nombre`: 20 rótulos, incluidos los reales de Marcelo.
- `GENS`: `caja_nombre` y `caja_letras` repetidos → 18,5% y 17,9% de los procedurales.
- `compose.clutter`: negativo nuevo de **notas de 2-4 líneas alineadas sin recuadro**.

### 6.3. SPM y lámpara de señalización mal encuadrados

IoU 0,33 y 0,61 contra 0,97 del resto. Son **14 componentes de 2.964** y se detectan todos: el
problema es sólo el recuadro. Se decidió **no perseguirlo** porque forzar el sintético para
corregirlos arriesga el encuadre de los otros 2.950.

Causa medida: el 64,6% de las pseudo-etiquetas son cajas altas y angostas (los interruptores de
los zips), y ese sesgo de forma se traslada al SPM, que es un rectángulo ancho (3,1:1).

### 6.4. test_2 probablemente esté contaminado

Los nombres de los zips de entrenamiento coinciden uno a uno con los componentes anotados a
mano en el GT de test_2 (`instrumento_de_medicion`=IM, `Ojo_de_buey`, `fotocelula`,
`tablero_de_transferencia`=TTAB, `grupo_electrogeno`=GE), y en las capturas se leen los mismos
textos. **No está confirmado** (el template matching no fue concluyente).

Afecta a todos los modelos por igual, así que la comparación entre ellos sigue siendo válida,
pero **el 100% de test_2 vale menos que el de fl_un_02 o tsss_2**.

### 6.5. Otros

- **572 cajas casi vacías** (0,45%) en el dataset: zonas neutras que borran parte de un símbolo
  etiquetado. Efecto: más FP, no menos recall. Ya hay un arreglo puesto para el próximo dataset.
- **PDFs con el texto vectorizado**: `ByA-IE-EU-Transferencia Unidades-Model.pdf` tiene 0 textos
  y hay que pasarle `--dpi` a mano (se probó 160).

---

## 7. Errores que ya cometí — no los repitas

1. **Filtrar símbolos por forma.** Armé un filtro que descartaba "cables, bloques macizos y
   marcos vacíos". Tomas revisó lo descartado y **casi todo eran componentes de verdad**: las
   barras de arcos son borneras, los rectángulos negros son aparatos, el símbolo dentro de su
   recuadro también lo es. En un unifilar un componente puede ser largo y fino, o una mancha
   negra. **Preguntale a Tomas con una grilla numerada en vez de inventar heurísticas.**

2. **Medir contra la fuente equivocada.** Medí los negativos falsos de los zips contra los
   `labels/*.txt`, pero el pipeline usa `zips_merged.json`, que ya tenía casi el doble de
   etiquetas. Dije "6,6× sin etiquetar" cuando era **32%**.

3. **Pseudo-etiquetar con un modelo que encuadra mal** (ver §5.1).

4. **Tocar un parámetro y arrastrar otro.** Al crear el tramo de símbolos chicos también bajé el
   piso del tramo grande, y corrí la mediana de todo el dataset.

5. **Diagnosticar sin mirar.** Dije que el SPM eran "dos rayitas" cuando era un fusible
   rectangular. Tomas tenía razón. **Hacé zoom antes de opinar.**

6. **Auditar el dataset SIEMPRE antes de entrenar 7 horas.** El script de auditoría
   (§8) cazó tres defectos que yo mismo había introducido, cada uno a los 5 minutos de
   lanzar en vez de a las 7 horas.

---

## 8. Herramientas útiles

### Auditoría del dataset (correr SIEMPRE antes de entrenar)

Chequea 7 defectos sobre cada tile y sus etiquetas: cajas vacías, macizas, que no ajustan a la
tinta, anidadas, gigantes, minúsculas, y tiles positivos sin etiquetas. Valores sanos del ds9:

```
1 caja casi vacia                     556  (0.44%)
2 caja macizo negro                   373  (0.29%)
3 caja que no ajusta a la tinta      1156  (0.91%)
4 caja anidada                          3  (0.00%)
6 caja minuscula                       94  (0.07%)
7 positivo sin etiquetas              249  (0.20%)
```

### Evaluación

```bash
py -3 src/evaluate.py best_componente_v9_N.pt 0.05 N_005          # los 4 planos base
set EVAL_SET=qet && set QET_DIR=test_marcelo
py -3 src/evaluate.py best_componente_v9_N.pt 0.20 N_marcelo      # los 3 de Marcelo
```

El emparejamiento es **húngaro 1-a-1** (`scipy.optimize.linear_sum_assignment`), así que dos
detecciones sobre el mismo componente cuentan una como FP.

---

## 9. Entorno

- Windows 10, RTX 3060 Ti (8 GB)
- ultralytics 8.4.41, PyTorch 2.4.1+cu121, Python 3.9
- Un entrenamiento de 200 épocas tarda **~7 horas**
- `ezdxf` + `matplotlib` para render DXF, `PyMuPDF` (fitz) para PDF
- **Cuidado con el disco C:** ya se llenó una vez con un entrenamiento corriendo

---

## 10. Qué haría yo ahora

**El orden vigente está en §0 → "Lo que sigue".** Resumen del porqué, al 22/09:

1. **El pedido pendiente de Tomas** (yolo26n-p2, yolo11s, yolo26n sobre ds13) va primero: es lo
   que pidió y responde si otra arquitectura baja el piso de 1 FN de un solo modelo. P2 es el
   candidato con más fundamento: los fallos que quedan son símbolos chicos o angostos.
2. **El script de inferencia de producción R+U**: Tomas decidió usar los dos, y hoy sólo existe
   la medición del ensamble. Sin eso la decisión no se puede usar.
3. **El ZIP para el compañero de Tomas con el arreglo del giro** (§5.7.1, 340 componentes).
4. **Precisión / calibración**: `pat_negativo` con menos peso y regenerar pseudo-etiquetas con R.
5. El encuadre (§5.7.3, 27,7% de cajas que `corta`) y las cajas que abarcan varias borneras (§6.1)
   siguen sin atacar; Tomas aclaró el 19/09 que lo de borneras no es prioridad.

**Antes de cualquier entrenamiento con datos nuevos**: `src/auditar_ds.py` y mostrarle a Tomas una
grilla numerada de lo que vaya a cambiar. **Antes de creerle a una comparación**: `GT_V4=1`, mismo
umbral en los 7 planos, y verificar que el GT total sea 3143.
