"""
Renderiza en batch todos los .dxf de dxf-dataset/ y aplica el tileo de SAHI,
igual que hace el pipeline pero SIN inferencia.

Resultado en png-dataset/:
  renders/   -> un PNG completo por DXF  (+ .json de metadatos)
  tiles/     -> tiles individuales, prefijados con el nombre del DXF
                  ej: 1er_Piso__TS1AE__slice_0000.png
  (opcional) grids/ -> overview del grid de slicing por DXF

Uso basico:
    python dxf_batch_render.py

Con opciones:
    python dxf_batch_render.py \\
        --dataset  dxf-dataset \\
        --out      png-dataset \\
        --target-px 64 \\
        --bw       color \\
        --max-dim-px 16000 \\
        --slice    640 \\
        --overlap  0.2 \\
        --zoom     1.0 \\
        --capas    ELECTRICA TOMAS ILUMINACION \\
        --save-grids \\
        --force

Flags:
  --force       Re-renderiza y re-tilea aunque los archivos ya existan.
  --save-grids  Guarda un PNG de overview del grid de slicing por DXF.
"""

import os
import sys
import time
import argparse
import traceback
from pathlib import Path


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def _es_blanco(img_bgr, blank_thresh):
    """
    Devuelve True si el tile es mayormente fondo blanco.
    blank_thresh: fraccion minima de pixeles NO-blancos para considerar el
                  tile con contenido. Ej: 0.01 = descarta tiles con menos
                  del 1% de pixeles con trazos.
    Se convierte a gris y se umbraliza en 250 (casi blanco = fondo).
    """
    import cv2
    import numpy as np
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    n_oscuros = np.count_nonzero(gray < 250)
    fraccion  = n_oscuros / gray.size
    return fraccion < blank_thresh


def _procesar_uno(args_tuple):
    """
    1) Renderiza el DXF a PNG completo.
    2) Tiledea con SAHI y guarda los tiles (filtrando blancos si blank_thresh > 0).
    Devuelve (dxf_path, ok, msg, n_tiles).
    """
    (dxf_path, render_path, tiles_dir, grid_path,
     capas_incluir, target_px, modo_color, max_dim_px,
     slice_size, overlap, zoom, min_tiles_por_eje,
     blank_thresh, force) = args_tuple

    render_path = Path(render_path)
    tiles_dir   = Path(tiles_dir)

    # --- Render ---
    if render_path.exists() and not force:
        print(f"  [render] saltado (ya existe): {render_path.name}")
    else:
        render_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            from dxf_to_image import renderizar_dxf
            renderizar_dxf(
                str(dxf_path),
                str(render_path),
                capas_incluir=capas_incluir if capas_incluir else None,
                target_px=target_px,
                modo_color=modo_color,
                max_dim_px=max_dim_px,
            )
        except Exception as e:
            tb = traceback.format_exc()
            return (str(dxf_path), False, f"ERROR en render: {e}\n{tb}", 0)

    # --- Tileo SAHI ---
    # Verificamos si los tiles ya existen
    tiles_dir.mkdir(parents=True, exist_ok=True)
    prefix = _make_prefix(dxf_path)
    existing = list(tiles_dir.glob(f"{prefix}_*.png"))
    if existing and not force:
        n = len(existing)
        return (str(dxf_path), True, f"saltado ({n} tiles ya existen)", n)

    try:
        import cv2
        from sahi.slicing import slice_image
        from inference_sahi import calcular_slice_size

        # Recalculamos slice_size si se paso zoom o min_tiles
        effective_slice = slice_size
        if zoom != 1.0 or min_tiles_por_eje is not None:
            effective_slice = calcular_slice_size(
                str(render_path),
                base_slice=slice_size,
                zoom=zoom,
                min_tiles_por_eje=min_tiles_por_eje,
            )

        result = slice_image(
            image=str(render_path),
            slice_height=effective_slice,
            slice_width=effective_slice,
            overlap_height_ratio=overlap,
            overlap_width_ratio=overlap,
            min_area_ratio=0.1,
        )

        n_total  = len(result)
        n_tiles  = 0
        n_blancos = 0
        for i, sl in enumerate(result):
            tile_img = sl["image"]           # numpy array BGR
            if blank_thresh > 0 and _es_blanco(tile_img, blank_thresh):
                n_blancos += 1
                continue
            out_name = f"{prefix}_{i:04d}.png"
            cv2.imwrite(str(tiles_dir / out_name), tile_img)
            n_tiles += 1

        # Grid overview opcional
        if grid_path:
            grid_path = Path(grid_path)
            grid_path.parent.mkdir(parents=True, exist_ok=True)
            img_grid = cv2.imread(str(render_path))
            if img_grid is not None:
                for sl in result:
                    x1, y1 = sl["starting_pixel"]
                    h, w = sl["image"].shape[:2]
                    x2, y2 = x1 + w, y1 + h
                    cv2.rectangle(img_grid, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.imwrite(str(grid_path), img_grid)

        msg = f"ok — {n_tiles} tiles guardados"
        if n_blancos:
            msg += f"  ({n_blancos} blancos descartados de {n_total})"
        return (str(dxf_path), True, msg, n_tiles)

    except Exception as e:
        tb = traceback.format_exc()
        return (str(dxf_path), False, f"ERROR en tileo: {e}\n{tb}", 0)


def _make_prefix(dxf_path):
    """
    Construye un prefijo unico para los tiles a partir de la ruta.
    Ej: 'dxf-dataset/1er Piso/TS1AE.dxf' -> '1er_Piso__TS1AE'
    """
    p = Path(dxf_path)
    parts = list(p.parts)
    # Tomamos subcarpeta + stem, limpiando caracteres problematicos
    def clean(s): return s.replace(" ", "_").replace("/", "_").replace("\\", "_")
    if len(parts) >= 2:
        return f"{clean(parts[-2])}__{clean(p.stem)}"
    return clean(p.stem)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Batch DXF -> PNG render + tileo SAHI (sin inferencia)."
    )
    parser.add_argument("--dataset",       default="dxf-dataset")
    parser.add_argument("--out",           default="png-dataset")
    parser.add_argument("--target-px",     type=int,   default=64)
    parser.add_argument("--bw",            default="grayscale",
                        choices=["color", "grayscale", "mono", "binary"])
    parser.add_argument("--max-dim-px",    type=int,   default=16000)
    parser.add_argument("--capas",         nargs="*",  default=None)
    parser.add_argument("--slice",         type=int,   default=640,
                        help="Tamano base de cada tile en px (default 640)")
    parser.add_argument("--overlap",       type=float, default=0.2,
                        help="Overlap entre tiles (default 0.2)")
    parser.add_argument("--zoom",          type=float, default=1.0,
                        help="zoom=2 -> tiles mas chicos (mas detalle por tile)")
    parser.add_argument("--min-tiles",     type=int,   default=None,
                        help="Tiles minimos por eje (anula --zoom)")
    parser.add_argument("--blank-thresh",  type=float, default=0.01,
                        help="Fraccion minima de pixeles con trazos para "
                             "guardar un tile. 0.01 = descarta tiles con "
                             "menos del 1%% de contenido (default). "
                             "0 = guarda todo sin filtrar.")
    parser.add_argument("--save-grids",    action="store_true",
                        help="Guarda PNG con el grid de slicing por DXF")
    parser.add_argument("--force",         action="store_true",
                        help="Re-renderiza y re-tiledea aunque ya existan")
    parser.add_argument("--workers",       type=int,   default=1,
                        help="Procesos paralelos (default 1, mas estable en Windows)")
    args = parser.parse_args()

    dataset_root = Path(args.dataset)
    out_root     = Path(args.out)
    renders_dir  = out_root / "renders"
    tiles_dir    = out_root / "tiles"
    grids_dir    = out_root / "grids" if args.save_grids else None

    if not dataset_root.exists():
        print(f"[ERROR] No se encontro: {dataset_root.resolve()}")
        sys.exit(1)

    dxf_files = sorted(dataset_root.rglob("*.dxf"))
    if not dxf_files:
        print(f"[AVISO] No hay .dxf en {dataset_root.resolve()}")
        sys.exit(0)

    print(f"Dataset   : {dataset_root.resolve()}")
    print(f"Renders   : {renders_dir.resolve()}")
    print(f"Tiles     : {tiles_dir.resolve()}")
    if grids_dir:
        print(f"Grids     : {grids_dir.resolve()}")
    print(f"DXFs      : {len(dxf_files)}")
    print(f"Workers   : {args.workers}")
    print(f"Force     : {args.force}")
    print(f"target-px={args.target_px}  bw={args.bw}  max-dim-px={args.max_dim_px}")
    print(f"slice={args.slice}  overlap={args.overlap}  zoom={args.zoom}  blank-thresh={args.blank_thresh}")
    if args.min_tiles:
        print(f"min-tiles : {args.min_tiles}")
    if args.capas:
        print(f"Capas     : {args.capas}")
    print()

    # Construimos jobs
    jobs = []
    for dxf_path in dxf_files:
        rel       = dxf_path.relative_to(dataset_root)
        render_p  = renders_dir / rel.with_suffix(".png")
        prefix    = _make_prefix(str(dxf_path))
        grid_p    = str(grids_dir / f"{prefix}_grid.png") if grids_dir else None
        jobs.append((
            str(dxf_path),
            str(render_p),
            str(tiles_dir),
            grid_p,
            args.capas,
            args.target_px,
            args.bw,
            args.max_dim_px,
            args.slice,
            args.overlap,
            args.zoom,
            args.min_tiles,
            args.blank_thresh,
            args.force,
        ))

    t0 = time.time()
    ok_count = skip_count = err_count = total_tiles = 0
    errors = []

    def handle(i, dxf_str, ok, msg, n_tiles):
        nonlocal ok_count, skip_count, err_count, total_tiles
        status = "✓" if ok else "✗"
        short  = _make_prefix(dxf_str)
        first_line = msg.splitlines()[0]
        print(f"[{i:>3}/{len(jobs)}] {status} {short:45s}  {first_line}")
        if ok:
            if "saltado" in msg:
                skip_count += 1
            else:
                ok_count += 1
            total_tiles += n_tiles
        else:
            err_count += 1
            errors.append((dxf_str, msg))

    if args.workers == 1:
        for i, job in enumerate(jobs, 1):
            dxf_str, ok, msg, n_tiles = _procesar_uno(job)
            handle(i, dxf_str, ok, msg, n_tiles)
    else:
        from multiprocessing import Pool
        with Pool(processes=args.workers) as pool:
            for i, res in enumerate(pool.imap_unordered(_procesar_uno, jobs), 1):
                dxf_str, ok, msg, n_tiles = res
                handle(i, dxf_str, ok, msg, n_tiles)

    elapsed = time.time() - t0
    print()
    print("=" * 60)
    print(f"Terminado en {elapsed:.1f}s")
    print(f"  Procesados   : {ok_count}")
    print(f"  Saltados     : {skip_count}")
    print(f"  Errores      : {err_count}")
    print(f"  Tiles totales: {total_tiles}")
    if errors:
        print()
        print("Errores detalle:")
        for dxf_str, msg in errors:
            print(f"  {_make_prefix(dxf_str)}")
            for line in msg.splitlines()[:5]:
                print(f"    {line}")
    print("=" * 60)


if __name__ == "__main__":
    main()
