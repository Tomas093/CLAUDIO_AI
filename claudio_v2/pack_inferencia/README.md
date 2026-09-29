# Detector de componentes en planos unifilares

Detecta componentes eléctricos (una sola clase: `componente`) en planos DXF y en PDF vectoriales.

## Qué hay adentro

| Archivo | Qué es |
|---|---|
| `detectar.py` | el script, autocontenido |
| `best_componente_v6_J.pt` | modelo **J** — el mejor, usar este |
| `best_componente_v3.pt` | modelo **G** — el anterior |
| `README.md` | esto |

## Instalación

```
pip install ultralytics ezdxf opencv-python matplotlib numpy
pip install pymupdf        # solo si vas a leer PDF
```

## Uso

```
py -3 detectar.py plano.dxf --pesos best_componente_v3.pt --conf 0.15
py -3 detectar.py carpeta_con_planos --pesos best_componente_v3.pt --conf 0.15
py -3 detectar.py plano.pdf --pesos best_componente_v3.pt --conf 0.15
```

Opciones:

| Flag | Para qué |
|---|---|
| `--conf 0.15` | umbral de confianza. Más bajo = más detecciones y más falsos positivos |
| `--salida carpeta` | dónde escribir (por defecto `work/detecciones`) |
| `--device 0` | GPU. Sin esto usa lo que encuentre; `--device cpu` fuerza CPU |
| `--escalas 1.0,1.6` | escalas de inferencia |
| `--ppc N` | fuerza la escala en DXF sin texto suficiente |
| `--dpi N` | fuerza el dpi en PDF con el texto vectorizado |
| `--sin-visual` | no genera el PNG con las cajas |

## Qué devuelve

Por cada plano, en la carpeta de salida:

- `<plano>_detecciones.csv` — `x1,y1,x2,y2,conf`. En DXF las coordenadas son **unidades CAD**; en PDF son **puntos de la página**.
- `<plano>_visual.png` — el render con las cajas dibujadas.
- `resumen.json` — escala usada, cantidad de detecciones y parámetros.

## Cómo funciona

1. **Render** del DXF con su texto (política de color BLACK sobre fondo blanco) y realce de trazos finos.
2. **Escala automática**: la mediana de altura de texto se lleva a 11 px. Es lo que hace que el mismo modelo funcione en planos de escalas distintas. Si el plano no tiene al menos 5 textos, hay que pasar `--ppc` (DXF) o `--dpi` (PDF).
3. **Inferencia por tiles** de 640 con paso 320, a dos escalas (1.0 y 1.6), con umbral interno bajo (0.05).
4. **Fusión de cajas**: NMS más supresión de anidadas, promediando las cajas del mismo objeto en vez de descartarlas.
5. Recién al final se aplica el `--conf` que pediste.

### Dos cosas que resuelve y conviene conocer

**Máscaras que tapan el plano.** Antes de renderizar se sacan los WIPEOUT, los HATCH sólidos blancos y se apaga el `bg_fill` de los MTEXT. Con la política de color BLACK esos elementos se pintan negros y pueden tapar el dibujo entero (visto en un plano real: el 57% de la imagen en negro y el modelo veía solo un borde).

**PDF sin pasar por DXF.** El PDF se renderiza directo a imagen con PyMuPDF, que respeta la rotación de página, las fuentes y los anchos de línea. Convertir PDF→DXF y de ahí a imagen arrastra errores de rotación y de texto.

## Qué modelo usar

**`best_componente_v6_J.pt` (J)** es el que hay que usar. Medido sobre 7 planos con verdad de terreno:

| | J | G (anterior) |
|---|---|---|
| planos de test (784 componentes) | **99,9%** · 1 FN | 98,6% · 11 FN |
| fl_un_02 | **100%** · 0 FN | 97,5% · 9 FN |
| tsss_2 | **100%** · 0 FN | 99,5% · 1 FN |
| EZE4077 | **100%** · 0 FN | 100% · 0 FN |

J toma el fusible y el ojo de buey de la rama de comando como **dos componentes separados**, que es el criterio correcto. G detectaba uno solo de los dos.

El precio son más falsos positivos (530 contra 358 en los planos de test a conf 0.25). Si molestan, subí el `--conf`: J mantiene el 100% de fl_un_02 hasta 0.06 y el de test1 hasta 0.60.

**`best_componente_v3.pt` (G)** queda por si hace falta comparar.

## Límites conocidos

- **Escala**: si el plano tiene menos de ~100 textos reales, la escala automática es poco confiable. Pasá `--ppc` o `--dpi` a mano.
- **Planos rotados**: si el dibujo está rotado 90° dentro de la hoja, el recall baja. El modelo se entrenó sin rotaciones.
- **Símbolos pegados**: cuando varios símbolos se tocan, a veces una caja abarca dos o tres.
- **Amperímetros y pulsadores**: son los que peor detecta hoy.
