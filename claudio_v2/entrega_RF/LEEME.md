# Detector de componentes (RF-DETR Nano)

Detecta todos los símbolos eléctricos de un plano unifilar (DXF o PDF) y devuelve una caja por componente.

## Instalar (una vez)

Python 3.9 o más nuevo.

```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

La primera línea es para GPU NVIDIA. Sin GPU: `pip install torch torchvision` (anda, pero más lento).

## Usar

```
python detectar.py <plano.dxf | plano.pdf | carpeta> --pesos modelo_rfdetr.pth --pdf-anotado
```

Deja en `work/detecciones/` (se cambia con `--salida`):

| archivo | qué es |
|---|---|
| `<plano>_detecciones.csv` | una fila por componente: `x1,y1,x2,y2,conf` (DXF: unidades CAD; PDF: puntos de la página) |
| `<plano>_visual.png` | imagen rápida con las cajas |
| `<plano>_detecciones.pdf` | sólo PDF con `--pdf-anotado`: el plano original en blanco y negro, vectorial (zoom sin perder calidad), cajas rojas (conf ≥ 0,25) y naranjas (dudosas) |

## Opciones útiles

- `--conf 0.13`: umbral. **0,13 es el default y detectó el 100% en los planos de prueba.** Subirlo baja
  los falsos positivos pero puede perder componentes.
- `--device cpu`: forzar CPU. `--batch 8`: bajarlo si la GPU se queda sin memoria.
- `--dpi 200`: sólo PDF con el texto convertido a curvas (el script avisa cuando hace falta).

Los mensajes `Using a different number of positional encodings…` al arrancar son normales.
