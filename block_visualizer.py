"""
block_visualizer.py — Crops thumbnail of each unique INSERT block from a rendered DXF image.

Usage:
    python block_visualizer.py <dxf_path> <render_json> <render_png> [--out-dir thumbnails]

The script:
  1. Reads the DXF and collects all INSERT entities by block name.
  2. Uses the render JSON (px_per_cad, x_min_cad, y_min_cad, image_height_px) to convert
     CAD coordinates -> pixel coordinates.
  3. Crops a NxN thumbnail around each block's first instance and saves it.

This avoids the blank-image problem of rendering isolated blocks in a temporary document.
"""

import os, sys, json, argparse
import ezdxf
from PIL import Image

CROP_PX = 160   # thumbnail side in pixels (centered on block insert point)


def cad_to_px(x_cad, y_cad, meta):
    px = (x_cad - meta["x_min_cad"]) * meta["px_per_cad"]
    py = meta["image_height_px"] - (y_cad - meta["y_min_cad"]) * meta["px_per_cad"]
    return int(round(px)), int(round(py))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dxf")
    ap.add_argument("render_json")
    ap.add_argument("render_png")
    ap.add_argument("--out-dir", default="thumbnails")
    ap.add_argument("--crop", type=int, default=CROP_PX)
    args = ap.parse_args()

    with open(args.render_json) as f:
        meta = json.load(f)

    img = Image.open(args.render_png)
    W, H = img.size
    print(f"Imagen: {W}x{H} px")

    doc = ezdxf.readfile(args.dxf)
    msp = doc.modelspace()

    # Collect first occurrence of each block name
    blocks = {}
    for ent in msp.query("INSERT"):
        name = ent.dxf.name
        if name.startswith("*"):
            continue
        if name not in blocks:
            ip = ent.dxf.insert
            blocks[name] = (float(ip.x), float(ip.y))

    print(f"\nBloques únicos (no anónimos): {len(blocks)}\n")

    os.makedirs(args.out_dir, exist_ok=True)
    half = args.crop // 2

    # Summary table
    print(f"{'Block name':<60} {'CAD x':>10} {'CAD y':>10} {'px_x':>6} {'px_y':>6}  Status")
    print("-"*110)

    for name, (cx, cy) in sorted(blocks.items()):
        px, py = cad_to_px(cx, cy, meta)
        x0 = max(0, px - half); y0 = max(0, py - half)
        x1 = min(W, px + half); y1 = min(H, py + half)

        in_bounds = (0 <= px <= W) and (0 <= py <= H)
        status = "OK" if in_bounds else "OUT_OF_FRAME"

        print(f"{name:<60} {cx:>10.2f} {cy:>10.2f} {px:>6} {py:>6}  {status}")

        # Save thumbnail regardless (might be partially clipped)
        crop = img.crop((x0, y0, x1, y1))
        safe_name = name.replace("/","_").replace("\\","_").replace(" ","_")[:80]
        thumb_path = os.path.join(args.out_dir, f"{safe_name}.png")
        crop.save(thumb_path)

    print(f"\nThumbnails guardados en: {args.out_dir}/")


if __name__ == "__main__":
    main()
