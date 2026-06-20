"""
slice_and_filter.py
--------------------
Recorta una imagen (o un DXF que primero renderiza) en tiles de 640x640
y elimina automáticamente los tiles en blanco o casi en blanco.

Uso:
    # Desde un DXF:
    python slice_and_filter.py plano.dxf --output-dir tiles/

    # Desde una imagen ya existente:
    python slice_and_filter.py plano.png --output-dir tiles/

Opciones:
    --output-dir    Carpeta donde se guardan los tiles (default: tiles_out/)
    --tile-size     Tamaño del tile en píxeles (default: 640)
    --whiteness     Porcentaje mínimo de píxeles NO blancos para conservar un
                    tile. Un tile se descarta si tiene más del (100-whiteness)%
                    de píxeles "blancos". Default: 1  (descartar si >99% blanco)
    --white-thresh  Valor de umbral por canal: píxeles con TODOS los canales
                    >= white_thresh se consideran "blancos". Default: 245
    --no-render     No renderiza el DXF; asume que la entrada ya es una imagen.
    --target-px     Píxeles objetivo por símbolo CAD al renderizar (default: 64)
    --max-dim-px    Límite máximo de dimensión al renderizar (default: 16000)
"""

import argparse
import os
import sys

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def is_blank(tile: np.ndarray, white_thresh: int = 245, whiteness: float = 1.0) -> bool:
    """
    Devuelve True si el tile está "en blanco" (se debe descartar).

    Un píxel se considera blanco si TODOS sus canales >= white_thresh.
    El tile se descarta si el porcentaje de píxeles no-blancos es menor
    que `whiteness` (%).
    """
    # Para imágenes en escala de grises convertidas a BGR, los tres canales son iguales
    if tile.ndim == 2:
        mask_white = tile >= white_thresh
    else:
        # píxel es blanco si todos los canales superan el umbral
        mask_white = np.all(tile >= white_thresh, axis=2)

    total = mask_white.size
    white_count = int(np.sum(mask_white))
    non_white_pct = (total - white_count) / total * 100.0

    return non_white_pct < whiteness


def slice_image(image: np.ndarray, tile_size: int, output_dir: str,
                base_name: str, white_thresh: int, whiteness: float) -> tuple[int, int]:
    """
    Recorta `image` en tiles de `tile_size x tile_size`.
    Guarda los tiles que NO son en blanco en `output_dir`.

    Devuelve (total_tiles, saved_tiles).
    """
    os.makedirs(output_dir, exist_ok=True)

    h, w = image.shape[:2]
    max_y = h // tile_size
    max_x = w // tile_size

    total = 0
    saved = 0

    for i in range(max_y):
        y1 = i * tile_size
        y2 = y1 + tile_size
        for j in range(max_x):
            x1 = j * tile_size
            x2 = x1 + tile_size

            tile = image[y1:y2, x1:x2]
            total += 1

            if is_blank(tile, white_thresh=white_thresh, whiteness=whiteness):
                continue  # descartamos el tile en blanco

            tile_name = f"{base_name}_y{i:04d}_x{j:04d}.png"
            out_path = os.path.join(output_dir, tile_name)
            cv2.imwrite(out_path, tile)
            saved += 1

    return total, saved


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Recorta imagen/DXF en tiles 640x640 y elimina los blancos."
    )
    parser.add_argument("input", help="Ruta al archivo DXF o imagen PNG/JPG.")
    parser.add_argument("--output-dir", default="tiles_out",
                        help="Carpeta de salida para los tiles (default: tiles_out/).")
    parser.add_argument("--tile-size", type=int, default=640,
                        help="Tamaño del tile en píxeles (default: 640).")
    parser.add_argument("--whiteness", type=float, default=1.0,
                        help="Porcentaje mínimo de píxeles no-blancos para conservar "
                             "un tile (default: 1.0 %%). Tiles con menos se descartan.")
    parser.add_argument("--white-thresh", type=int, default=245,
                        help="Umbral de valor por canal para considerar un píxel 'blanco' "
                             "(default: 245, rango 0-255).")
    parser.add_argument("--no-render", action="store_true",
                        help="No renderiza el DXF; trata la entrada como imagen directamente.")
    parser.add_argument("--target-px", type=int, default=64,
                        help="Píxeles objetivo por símbolo CAD al renderizar DXF (default: 64).")
    parser.add_argument("--max-dim-px", type=int, default=16000,
                        help="Límite máximo de píxeles por lado al renderizar DXF (default: 16000).")

    args = parser.parse_args()

    input_path = args.input
    ext = os.path.splitext(input_path)[1].lower()

    # ----- Determinar ruta de imagen -----
    if ext == ".dxf" and not args.no_render:
        # Renderizar DXF a PNG temporal
        print(f"[info] Renderizando DXF: {input_path}")
        png_path = os.path.splitext(input_path)[0] + "_render.png"

        # Importamos aquí para no depender de ezdxf si se usa --no-render
        try:
            from dxf_to_image import renderizar_dxf
        except ImportError:
            print("[error] No se encontró dxf_to_image.py. "
                  "Ejecuta el script desde el directorio raíz del proyecto.")
            sys.exit(1)

        renderizar_dxf(
            dxf_path=input_path,
            output_path=png_path,
            target_px=args.target_px,
            max_dim_px=args.max_dim_px,
        )
        image_path = png_path
    else:
        image_path = input_path

    # ----- Leer imagen -----
    print(f"[info] Cargando imagen: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        print(f"[error] No se pudo leer la imagen: {image_path}")
        sys.exit(1)

    h, w = image.shape[:2]
    print(f"[info] Dimensiones: {w}x{h} px")
    print(f"[info] Tile size: {args.tile_size}x{args.tile_size} px")
    print(f"[info] Grid: {w // args.tile_size} cols x {h // args.tile_size} filas "
          f"= {(w // args.tile_size) * (h // args.tile_size)} tiles posibles")

    # ----- Recortar y filtrar -----
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    total, saved = slice_image(
        image=image,
        tile_size=args.tile_size,
        output_dir=args.output_dir,
        base_name=base_name,
        white_thresh=args.white_thresh,
        whiteness=args.whiteness,
    )

    discarded = total - saved
    print(f"\n[resultado] Tiles totales  : {total}")
    print(f"[resultado] Tiles guardados: {saved}")
    print(f"[resultado] Tiles descartados (blancos): {discarded}")
    print(f"[resultado] Directorio de salida: {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    main()
