# generate_backgrounds.py — Background tile generation from full DXF floor plans
# Renders each .dxf in the sources directory at high DPI, slices into
# overlapping tiles, filters out blank tiles, and saves to backgrounds_dir.
# Integrated into the pipeline via run_pipeline.py.
from __future__ import annotations

import io
from pathlib import Path

import cv2
import ezdxf
import matplotlib
import numpy as np
from ezdxf.addons.drawing import RenderContext, Frontend
from ezdxf.addons.drawing.matplotlib import MatplotlibBackend

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import load_config, BackgroundsConfig


def dxf_to_image(dxf_path: Path, dpi: int = 4098) -> np.ndarray | None:
    """Render a full DXF floor plan to a BGR image in memory."""
    try:
        doc = ezdxf.readfile(str(dxf_path))
        msp = doc.modelspace()

        fig = plt.figure(figsize=(20, 20))
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_axis_off()

        ctx = RenderContext(doc)
        backend = MatplotlibBackend(ax)
        Frontend(ctx, backend).draw_layout(msp, finalize=True)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi)
        plt.close(fig)
        buf.seek(0)

        img = cv2.imdecode(np.frombuffer(buf.read(), np.uint8), cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"  ⚠️  Error procesando {dxf_path.name}: {e}")
        return None


def slice_and_save(
    img: np.ndarray,
    output_dir: Path,
    filename_base: str,
    tile_size: int = 640,
    overlap: int = 320,
    min_std_dev: float = 10.0,
) -> int:
    """Slice *img* into overlapping tiles and save non-blank ones as JPG.

    Returns the number of tiles saved.
    """
    h, w = img.shape[:2]
    step = tile_size - overlap
    if step <= 0:
        print("  ⚠️  overlap >= tile_size — no se pueden generar tiles.")
        return 0

    count = 0
    for y in range(0, max(1, h - tile_size), step):
        for x in range(0, max(1, w - tile_size), step):
            tile = img[y : y + tile_size, x : x + tile_size]

            # Skip blank / nearly uniform tiles
            if np.std(tile) < min_std_dev:
                continue

            out_path = output_dir / f"bg_{filename_base}_{count:04d}.jpg"
            cv2.imwrite(str(out_path), tile, [cv2.IMWRITE_JPEG_QUALITY, 95])
            count += 1

    return count


def generate_backgrounds(
    dxf_sources_dir: Path | None = None,
    output_dir: Path | None = None,
    tile_size: int | None = None,
    overlap: int | None = None,
    render_dpi: int | None = None,
    min_std_dev: float | None = None,
) -> int:
    """Generate background tiles from all DXF files in *dxf_sources_dir*.

    Parameters default to values from ``components_config.yaml`` when not
    provided explicitly.

    Returns the total number of tiles generated.
    """

    cfg = load_config()
    bg_cfg = cfg.backgrounds

    dxf_sources_dir = dxf_sources_dir or bg_cfg.dxf_sources_dir
    output_dir = output_dir or cfg.g.backgrounds_dir
    tile_size = tile_size if tile_size is not None else bg_cfg.tile_size
    overlap = overlap if overlap is not None else bg_cfg.overlap
    render_dpi = render_dpi if render_dpi is not None else bg_cfg.render_dpi
    min_std_dev = min_std_dev if min_std_dev is not None else bg_cfg.min_std_dev

    # Validate source dir
    if not dxf_sources_dir.exists():
        dxf_sources_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"[Backgrounds] Creada carpeta '{dxf_sources_dir}'.\n"
            f"  Poné tus planos DXF completos ahí y volvé a ejecutar."
        )
        return 0

    planos = sorted(dxf_sources_dir.glob("*.dxf"))
    if not planos:
        print(f"[Backgrounds] ⚠️  No se encontraron archivos .dxf en '{dxf_sources_dir}'")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Backgrounds] {len(planos)} planos DXF encontrados en '{dxf_sources_dir}'")
    print(f"[Backgrounds] Tile: {tile_size}px  |  Overlap: {overlap}px  |  DPI: {render_dpi}")

    total_tiles = 0
    for plano in planos:
        print(f"  Renderizando {plano.name}...")
        img = dxf_to_image(plano, dpi=render_dpi)
        if img is None:
            continue

        n = slice_and_save(
            img, output_dir, plano.stem,
            tile_size=tile_size,
            overlap=overlap,
            min_std_dev=min_std_dev,
        )
        print(f"    → {n} fondos generados de este plano.")
        total_tiles += n

        # Free memory after each plan (can be several GB at high DPI)
        del img

    print(f"[Backgrounds] ✅ {total_tiles} fondos guardados en '{output_dir}'\n")
    return total_tiles


if __name__ == "__main__":
    generate_backgrounds()