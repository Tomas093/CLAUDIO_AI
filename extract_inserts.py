"""
extract_inserts.py
-------------------
Detecta los INSERTs (referencias a bloques/símbolos) de un plano DXF,
renderiza el plano completo y recorta un tile de tile_size x tile_size
centrado en cada INSERT.

En vez de un sliding window ciego, usa directamente la geometría del DXF
para saber exactamente dónde está cada símbolo.

Uso:
    python extract_inserts.py plano.dxf [opciones]

Opciones:
    --output-dir      Carpeta de salida (default: inserts_out/<nombre_dxf>/)
    --tile-size       Tamaño del recorte en píxeles (default: 640)
    --target-px       Píxeles objetivo por símbolo CAD al renderizar (default: 64)
    --max-dim-px      Límite máximo por lado al renderizar (default: 50000)
    --filter-blocks   Filtrar sólo bloques cuyos nombres contienen este texto
    --export-csv      Exportar tabla de INSERTs (bloque, pos CAD, pos px, tile)
    --no-render       Usar imagen PNG ya existente (mismo nombre que el DXF)
    --depth           Profundidad máxima de búsqueda de INSERTs (default: 1)
"""

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict

import cv2
import numpy as np
import ezdxf
from ezdxf.bbox import extents as dxf_extents


# ---------------------------------------------------------------------------
# Utilidades de transformación CAD → Píxel
# ---------------------------------------------------------------------------

def build_cad_to_px(meta: dict):
    """
    Devuelve una función (x_cad, y_cad) -> (px_col, px_row) usando los
    metadatos generados por dxf_to_image.renderizar_dxf().

    En matplotlib con ylim(y_min, y_max), el eje Y está invertido respecto
    a píxeles de imagen (px_row=0 es arriba, CAD y_max es arriba).
    """
    px_per_cad = meta["px_per_cad"]
    x_min_cad  = meta["x_min_cad"]
    y_min_cad  = meta["y_min_cad"]
    img_h      = meta["image_height_px"]

    def cad_to_px(x_cad, y_cad):
        col = (x_cad - x_min_cad) * px_per_cad
        # Eje Y de matplotlib va de abajo hacia arriba; en imagen va de arriba hacia abajo
        row = img_h - (y_cad - y_min_cad) * px_per_cad
        return int(round(col)), int(round(row))

    return cad_to_px


# ---------------------------------------------------------------------------
# Recolección de INSERTs desde el DXF
# ---------------------------------------------------------------------------

def collect_inserts(doc, msp, depth_max: int = 1, filter_text: str = None):
    """
    Recorre los INSERTs del modelspace hasta depth_max niveles de anidamiento.
    Resuelve las transformaciones (traslación + escala + rotación) para obtener
    la posición absoluta en coordenadas CAD del modelspace.

    Devuelve lista de dicts:
      { block_name, x_cad, y_cad, layer, depth, xscale, yscale, rotation }
    """
    results = []

    def _recurse(block, depth, tx, ty, sx, sy, angle_deg):
        """
        tx, ty : offset acumulado en coords del padre
        sx, sy : escala acumulada
        angle_deg : rotación acumulada (grados)
        """
        if depth > depth_max:
            return
        for entity in block:
            if entity.dxftype() != "INSERT":
                continue
            name = entity.dxf.name

            # Posición local del INSERT en el sistema del bloque padre
            lx = entity.dxf.insert.x
            ly = entity.dxf.insert.y

            # Rotación local
            local_rot = getattr(entity.dxf, "rotation", 0.0)
            local_sx  = getattr(entity.dxf, "xscale",   1.0)
            local_sy  = getattr(entity.dxf, "yscale",   1.0)

            # Aplicar transformación padre: rotar y escalar la posición local
            angle_rad = math.radians(angle_deg)
            rx = lx * sx * math.cos(angle_rad) - ly * sy * math.sin(angle_rad)
            ry = lx * sx * math.sin(angle_rad) + ly * sy * math.cos(angle_rad)

            abs_x = tx + rx
            abs_y = ty + ry

            # Escala y rotación acumuladas para niveles más profundos
            new_sx    = sx * local_sx
            new_sy    = sy * local_sy
            new_angle = angle_deg + local_rot

            if filter_text is None or filter_text.upper() in name.upper():
                results.append({
                    "block_name": name,
                    "x_cad": abs_x,
                    "y_cad": abs_y,
                    "layer": entity.dxf.layer,
                    "depth": depth,
                    "xscale": new_sx,
                    "yscale": new_sy,
                    "rotation": new_angle,
                })

            # Recursivo
            if name in doc.blocks and depth < depth_max:
                _recurse(doc.blocks[name], depth + 1, abs_x, abs_y, new_sx, new_sy, new_angle)

    _recurse(msp, depth=1, tx=0.0, ty=0.0, sx=1.0, sy=1.0, angle_deg=0.0)
    return results


# ---------------------------------------------------------------------------
# Recorte de tiles
# ---------------------------------------------------------------------------

def crop_tile(image: np.ndarray, cx: int, cy: int, tile_size: int) -> np.ndarray | None:
    """
    Recorta un tile centrado en (cx, cy). Devuelve None si el centro está
    completamente fuera de la imagen.
    """
    h, w = image.shape[:2]
    half = tile_size // 2

    x1 = cx - half
    y1 = cy - half
    x2 = x1 + tile_size
    y2 = y1 + tile_size

    # Coordenadas dentro de la imagen (clipeadas)
    ix1 = max(x1, 0)
    iy1 = max(y1, 0)
    ix2 = min(x2, w)
    iy2 = min(y2, h)

    if ix1 >= ix2 or iy1 >= iy2:
        return None  # fuera de imagen

    # Crear canvas blanco y pegar la región recortada
    canvas = np.full((tile_size, tile_size, 3), 255, dtype=np.uint8)
    src = image[iy1:iy2, ix1:ix2]

    # Posición en el canvas donde pegar
    dst_x1 = ix1 - x1
    dst_y1 = iy1 - y1
    canvas[dst_y1:dst_y1 + src.shape[0], dst_x1:dst_x1 + src.shape[1]] = src

    return canvas


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Extrae INSERTs del DXF y recorta tiles 640x640 en cada símbolo."
    )
    parser.add_argument("dxf", help="Ruta al archivo DXF.")
    parser.add_argument("--output-dir", default=None,
                        help="Carpeta de salida (default: inserts_out/<nombre_dxf>/).")
    parser.add_argument("--tile-size", type=int, default=640,
                        help="Tamaño del tile en píxeles (default: 640).")
    parser.add_argument("--target-px", type=int, default=64,
                        help="Píxeles objetivo por símbolo CAD (default: 64).")
    parser.add_argument("--max-dim-px", type=int, default=50000,
                        help="Límite máximo de dimensión al renderizar (default: 50000).")
    parser.add_argument("--filter-blocks", default=None,
                        help="Solo procesar bloques cuyo nombre contiene este texto.")
    parser.add_argument("--export-csv", action="store_true",
                        help="Exportar tabla CSV de todos los INSERTs detectados.")
    parser.add_argument("--no-render", action="store_true",
                        help="Usar imagen PNG existente en vez de renderizar el DXF.")
    parser.add_argument("--depth", type=int, default=1,
                        help="Profundidad máxima de búsqueda de INSERTs (default: 1).")
    args = parser.parse_args()

    dxf_path = args.dxf
    base_name = os.path.splitext(os.path.basename(dxf_path))[0]

    if args.output_dir is None:
        args.output_dir = os.path.join("inserts_out", base_name)

    os.makedirs(args.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Leer DXF y detectar INSERTs
    # ------------------------------------------------------------------
    print(f"[dxf] Leyendo: {dxf_path}")
    doc = ezdxf.readfile(dxf_path)
    msp = doc.modelspace()

    print(f"[dxf] Buscando INSERTs (depth_max={args.depth}) ...")
    inserts = collect_inserts(doc, msp, depth_max=args.depth, filter_text=args.filter_blocks)
    print(f"[dxf] INSERTs encontrados: {len(inserts)}")

    if not inserts:
        print("[warn] No se encontraron INSERTs con los filtros dados. Saliendo.")
        sys.exit(0)

    # Resumen por bloque
    from collections import Counter
    by_block = Counter(i["block_name"] for i in inserts)
    print("[dxf] Distribución por bloque:")
    for name, count in by_block.most_common(20):
        print(f"        {count:5d}  {name}")

    # ------------------------------------------------------------------
    # 2. Renderizar DXF (o usar imagen existente)
    # ------------------------------------------------------------------
    png_path = os.path.splitext(dxf_path)[0] + "_render.png"
    meta_path = os.path.splitext(dxf_path)[0] + "_render.json"

    if args.no_render and os.path.exists(png_path) and os.path.exists(meta_path):
        print(f"[render] Usando imagen existente: {png_path}")
        with open(meta_path) as f:
            meta = json.load(f)
    else:
        print(f"[render] Renderizando DXF → {png_path}")
        try:
            from dxf_to_image import renderizar_dxf
        except ImportError:
            print("[error] No se encontró dxf_to_image.py. "
                  "Ejecuta el script desde el directorio raíz del proyecto.")
            sys.exit(1)
        meta = renderizar_dxf(
            dxf_path=dxf_path,
            output_path=png_path,
            target_px=args.target_px,
            max_dim_px=args.max_dim_px,
        )

    print(f"[render] Imagen: {meta['image_width_px']}x{meta['image_height_px']} px")
    print(f"[render] Escala: {meta['px_per_cad']:.4f} px/CAD")

    # ------------------------------------------------------------------
    # 3. Leer imagen renderizada
    # ------------------------------------------------------------------
    image = cv2.imread(png_path)
    if image is None:
        print(f"[error] No se pudo leer: {png_path}")
        sys.exit(1)

    cad_to_px = build_cad_to_px(meta)

    # ------------------------------------------------------------------
    # 4. Recortar tile por cada INSERT
    # ------------------------------------------------------------------
    print(f"\n[tiles] Recortando {len(inserts)} tiles de {args.tile_size}x{args.tile_size} px ...")

    saved = 0
    skipped_oob = 0
    csv_rows = []

    # Agrupar por nombre de bloque para numerar
    block_counters = defaultdict(int)

    for ins in inserts:
        bname = ins["block_name"]
        cx, cy = cad_to_px(ins["x_cad"], ins["y_cad"])

        tile = crop_tile(image, cx, cy, args.tile_size)
        if tile is None:
            skipped_oob += 1
            continue

        block_counters[bname] += 1
        idx = block_counters[bname]

        # Nombre del archivo: <bloque>_<índice>.png
        safe_name = bname.replace(" ", "_").replace("/", "-").replace("\\", "-")
        fname = f"{safe_name}_{idx:04d}.png"
        out_path = os.path.join(args.output_dir, fname)
        cv2.imwrite(out_path, tile)
        saved += 1

        csv_rows.append({
            "block_name": bname,
            "x_cad": f"{ins['x_cad']:.4f}",
            "y_cad": f"{ins['y_cad']:.4f}",
            "px_col": cx,
            "px_row": cy,
            "layer": ins["layer"],
            "rotation": f"{ins['rotation']:.2f}",
            "tile_file": fname,
        })

    print(f"\n[resultado] INSERTs detectados : {len(inserts)}")
    print(f"[resultado] Tiles guardados     : {saved}")
    print(f"[resultado] Fuera de imagen     : {skipped_oob}")
    print(f"[resultado] Directorio          : {os.path.abspath(args.output_dir)}")

    # ------------------------------------------------------------------
    # 5. Exportar CSV (opcional)
    # ------------------------------------------------------------------
    if args.export_csv and csv_rows:
        csv_path = os.path.join(args.output_dir, f"{base_name}_inserts.csv")
        fieldnames = ["block_name", "x_cad", "y_cad", "px_col", "px_row",
                      "layer", "rotation", "tile_file"]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"[csv] Tabla exportada → {csv_path}")


if __name__ == "__main__":
    main()
