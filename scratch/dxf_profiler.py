"""
DXF Symbol Profiler — Extracts real symbol size metrics from jijiji.dxf.
Analyzes INSERT blocks, circles, and other entities to determine the actual
bounding box sizes and aspect ratios of electrical components in the plan.
Also computes what those sizes become in pixels after the autoescalado pipeline.
"""

import sys
import os
import json
import numpy as np
from collections import Counter
from pathlib import Path

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ezdxf
import ezdxf.bbox
from scale_analyzer import calcular_factor_escala, LAYERS_EXCLUIR

DXF_PATH = str(Path(__file__).resolve().parent.parent / "dxf" / "jijiji.dxf")
MAX_DIM_PX = 24000  # from dxf_to_image.py default

def profile_dxf(dxf_path):
    print(f"=== DXF Symbol Profiler ===")
    print(f"File: {dxf_path}")
    print(f"Size: {os.path.getsize(dxf_path) / 1e6:.1f} MB")
    
    doc = ezdxf.readfile(dxf_path)
    msp = doc.modelspace()
    
    # 1. Global bounding box
    bbox = ezdxf.bbox.extents(msp)
    if not bbox.has_data:
        print("ERROR: Empty modelspace")
        return
    
    x_min, y_min = bbox.extmin.x, bbox.extmin.y
    x_max, y_max = bbox.extmax.x, bbox.extmax.y
    ancho_cad = x_max - x_min
    alto_cad = y_max - y_min
    
    print(f"\n--- Global Bounding Box ---")
    print(f"  CAD extents: ({x_min:.2f}, {y_min:.2f}) to ({x_max:.2f}, {y_max:.2f})")
    print(f"  CAD size: {ancho_cad:.2f} x {alto_cad:.2f} units")
    
    # 2. Autoescalado analysis
    px_per_cad, ref = calcular_factor_escala(dxf_path, target_px=64)
    print(f"\n--- Autoescalado (target=64px) ---")
    print(f"  px_per_cad: {px_per_cad:.4f}")
    print(f"  Reference: {ref[0]} = {ref[1]:.4f} CAD units")
    
    ancho_px_raw = int(round(ancho_cad * px_per_cad))
    alto_px_raw = int(round(alto_cad * px_per_cad))
    print(f"  Raw image size: {ancho_px_raw} x {alto_px_raw} px")
    
    # Check cap
    capped = False
    effective_px_per_cad = px_per_cad
    if max(ancho_px_raw, alto_px_raw) > MAX_DIM_PX:
        factor = MAX_DIM_PX / max(ancho_px_raw, alto_px_raw)
        effective_px_per_cad = px_per_cad * factor
        ancho_px = int(round(ancho_cad * effective_px_per_cad))
        alto_px = int(round(alto_cad * effective_px_per_cad))
        capped = True
        print(f"  ⚠️  CAPPED to {MAX_DIM_PX}px max dimension!")
        print(f"  Effective px_per_cad: {effective_px_per_cad:.4f}")
        print(f"  Capped image size: {ancho_px} x {alto_px} px")
        print(f"  Reduction factor: {factor:.4f} ({factor*100:.1f}%)")
    else:
        ancho_px = ancho_px_raw
        alto_px = alto_px_raw
        print(f"  Image fits within cap: {ancho_px} x {alto_px} px")
    
    # 3. INSERT block analysis
    print(f"\n--- INSERT Block Analysis ---")
    insert_data = []
    block_name_counter = Counter()
    
    for ent in msp.query("INSERT"):
        layer = ""
        try:
            layer = ent.dxf.layer.upper()
        except:
            pass
        if layer in LAYERS_EXCLUIR:
            continue
        
        block_name = ""
        try:
            block_name = ent.dxf.name
        except:
            pass
        
        try:
            bb = ezdxf.bbox.extents([ent])
            if not bb.has_data:
                continue
            w_cad = bb.size.x
            h_cad = bb.size.y
            if w_cad <= 0 or h_cad <= 0:
                continue
            
            w_px = w_cad * effective_px_per_cad
            h_px = h_cad * effective_px_per_cad
            aspect = max(w_cad, h_cad) / min(w_cad, h_cad) if min(w_cad, h_cad) > 0 else float('inf')
            diag_cad = max(w_cad, h_cad)
            diag_px = diag_cad * effective_px_per_cad
            area_px = w_px * h_px
            
            insert_data.append({
                'block': block_name,
                'layer': layer,
                'w_cad': w_cad,
                'h_cad': h_cad,
                'w_px': w_px,
                'h_px': h_px,
                'diag_px': diag_px,
                'aspect': aspect,
                'area_px': area_px,
            })
            block_name_counter[block_name] += 1
        except Exception as e:
            continue
    
    if insert_data:
        w_px_arr = np.array([d['w_px'] for d in insert_data])
        h_px_arr = np.array([d['h_px'] for d in insert_data])
        diag_arr = np.array([d['diag_px'] for d in insert_data])
        aspect_arr = np.array([d['aspect'] for d in insert_data])
        area_arr = np.array([d['area_px'] for d in insert_data])
        
        # IQR filtering on diagonal
        p25, p75 = np.percentile(diag_arr, [25, 75])
        iqr = p75 - p25
        mask = (diag_arr >= p25 - 1.5 * iqr) & (diag_arr <= p75 + 1.5 * iqr)
        
        print(f"  Total INSERTs: {len(insert_data)}")
        print(f"  After IQR filter: {mask.sum()}")
        print(f"  Unique block names: {len(block_name_counter)}")
        
        # Show top blocks
        print(f"\n  Top 20 most frequent blocks:")
        for name, count in block_name_counter.most_common(20):
            print(f"    {name:40s} × {count}")
        
        # Stats on ALL inserts
        print(f"\n  ALL INSERTs (before IQR filter):")
        print(f"    Width  (px): min={w_px_arr.min():.1f}  p25={np.percentile(w_px_arr, 25):.1f}  "
              f"median={np.median(w_px_arr):.1f}  p75={np.percentile(w_px_arr, 75):.1f}  max={w_px_arr.max():.1f}")
        print(f"    Height (px): min={h_px_arr.min():.1f}  p25={np.percentile(h_px_arr, 25):.1f}  "
              f"median={np.median(h_px_arr):.1f}  p75={np.percentile(h_px_arr, 75):.1f}  max={h_px_arr.max():.1f}")
        print(f"    Diag   (px): min={diag_arr.min():.1f}  p25={p25:.1f}  "
              f"median={np.median(diag_arr):.1f}  p75={p75:.1f}  max={diag_arr.max():.1f}")
        print(f"    Aspect:      min={aspect_arr.min():.2f}  median={np.median(aspect_arr):.2f}  "
              f"mean={aspect_arr.mean():.2f}  max={aspect_arr.max():.2f}")
        print(f"    Area   (px²): median={np.median(area_arr):.1f}  mean={area_arr.mean():.1f}")
        
        # Stats on FILTERED inserts (electrical symbols)
        if mask.sum() > 0:
            fw = w_px_arr[mask]
            fh = h_px_arr[mask]
            fd = diag_arr[mask]
            fa = aspect_arr[mask]
            
            print(f"\n  FILTERED INSERTs (IQR, likely electrical symbols):")
            print(f"    Width  (px): min={fw.min():.1f}  median={np.median(fw):.1f}  mean={fw.mean():.1f}  max={fw.max():.1f}")
            print(f"    Height (px): min={fh.min():.1f}  median={np.median(fh):.1f}  mean={fh.mean():.1f}  max={fh.max():.1f}")
            print(f"    Diag   (px): min={fd.min():.1f}  median={np.median(fd):.1f}  mean={fd.mean():.1f}  max={fd.max():.1f}")
            print(f"    Aspect:      min={fa.min():.2f}  median={np.median(fa):.2f}  mean={fa.mean():.2f}  max={fa.max():.2f}")
            
            # Size distribution buckets  
            print(f"\n  Size distribution (diag, filtered, in pixels):")
            buckets = [(0, 8), (8, 16), (16, 32), (32, 64), (64, 128), (128, 256), (256, 512), (512, float('inf'))]
            for lo, hi in buckets:
                count = ((fd >= lo) & (fd < hi)).sum()
                pct = count / len(fd) * 100
                bar = '█' * int(pct / 2)
                label = f"{lo}-{hi}px" if hi != float('inf') else f">{lo}px"
                print(f"      {label:>10s}: {count:5d} ({pct:5.1f}%) {bar}")
    
    # 4. CIRCLE analysis
    print(f"\n--- CIRCLE Analysis ---")
    circle_radii_cad = []
    for ent in msp.query("CIRCLE"):
        layer = ""
        try:
            layer = ent.dxf.layer.upper()
        except:
            pass
        if layer in LAYERS_EXCLUIR:
            continue
        try:
            r = float(ent.dxf.radius)
            if r > 0:
                circle_radii_cad.append(r)
        except:
            continue
    
    if circle_radii_cad:
        radii_px = np.array(circle_radii_cad) * effective_px_per_cad
        diameters_px = radii_px * 2
        print(f"  Total circles: {len(circle_radii_cad)}")
        print(f"  Diameter (px): min={diameters_px.min():.1f}  median={np.median(diameters_px):.1f}  "
              f"mean={diameters_px.mean():.1f}  max={diameters_px.max():.1f}")
    
    # 5. TEXT analysis
    print(f"\n--- TEXT/MTEXT Analysis ---")
    text_heights_cad = []
    for ent in msp.query("TEXT MTEXT"):
        layer = ""
        try:
            layer = ent.dxf.layer.upper()
        except:
            pass
        if layer in LAYERS_EXCLUIR:
            continue
        try:
            h = float(getattr(ent.dxf, 'height', 0) or 0)
            if h > 0:
                text_heights_cad.append(h)
        except:
            continue
    
    if text_heights_cad:
        heights_px = np.array(text_heights_cad) * effective_px_per_cad
        print(f"  Total texts: {len(text_heights_cad)}")
        print(f"  Height (px): min={heights_px.min():.1f}  median={np.median(heights_px):.1f}  "
              f"mean={heights_px.mean():.1f}  max={heights_px.max():.1f}")
    
    # 6. Tile analysis for SAHI at 640x640
    print(f"\n--- SAHI Tiling Analysis (slice=640, overlap=0.2) ---")
    slice_size = 640
    overlap = 0.2
    step = int(slice_size * (1 - overlap))
    n_tiles_x = max(1, (ancho_px - slice_size) // step + 1) + 1
    n_tiles_y = max(1, (alto_px - slice_size) // step + 1) + 1
    total_tiles = n_tiles_x * n_tiles_y
    print(f"  Image size: {ancho_px} x {alto_px}")
    print(f"  Tiles per axis: {n_tiles_x} x {n_tiles_y} = {total_tiles} total")
    print(f"  Step size: {step}px")
    
    # What fraction of a 640x640 tile does a typical symbol occupy?
    if insert_data and mask.sum() > 0:
        median_w = np.median(fw)
        median_h = np.median(fh)
        pct_tile_w = median_w / slice_size * 100
        pct_tile_h = median_h / slice_size * 100
        pct_tile_area = (median_w * median_h) / (slice_size * slice_size) * 100
        print(f"\n  Median symbol vs 640x640 tile:")
        print(f"    Symbol: {median_w:.1f} x {median_h:.1f} px")
        print(f"    Width %:  {pct_tile_w:.1f}%  of tile")
        print(f"    Height %: {pct_tile_h:.1f}%  of tile")
        print(f"    Area %:   {pct_tile_area:.2f}% of tile")
        
        # Effective size after YOLO resize to 640x640 (already 640, so no extra resize)
        # But during TRAINING at imgsz=640, the synthetic images ARE the tiles
        print(f"\n  Symbol size relative to YOLO imgsz=640:")
        print(f"    At inference (SAHI tile = 640): symbol is {median_w:.1f} x {median_h:.1f} px → ✓ native size")
        print(f"    At training (imgsz=640): depends on synthetic image sprite_scale")
    
    # 7. Component DXF size profiling
    print(f"\n--- Component Sprite Source DXFs ---")
    comp_dir = Path(__file__).resolve().parent.parent / "train-maker" / "input" / "components"
    if comp_dir.exists():
        for dxf_file in sorted(comp_dir.glob("*.dxf")):
            try:
                comp_doc = ezdxf.readfile(str(dxf_file))
                comp_msp = comp_doc.modelspace()
                comp_bb = ezdxf.bbox.extents(comp_msp)
                if comp_bb.has_data:
                    cw = comp_bb.size.x
                    ch = comp_bb.size.y
                    ar = max(cw, ch) / min(cw, ch) if min(cw, ch) > 0 else float('inf')
                    print(f"  {dxf_file.stem:45s}  CAD: {cw:.2f} x {ch:.2f}  AR={ar:.2f}")
            except Exception as e:
                print(f"  {dxf_file.stem}: ERROR - {e}")
    
    # 8. Summary: Training sprite scale analysis
    print(f"\n--- Training Sprite Scale Analysis ---")
    sprite_scale_min = 0.04  # from components_config.yaml
    sprite_scale_max = 0.08
    bg_tile_size = 640  # background tiles are 640x640
    
    if comp_dir.exists():
        print(f"  Config: sprite_scale_min={sprite_scale_min}, sprite_scale_max={sprite_scale_max}")
        print(f"  Background tile size: {bg_tile_size}x{bg_tile_size}")
        print()
        for dxf_file in sorted(comp_dir.glob("*.dxf")):
            try:
                comp_doc = ezdxf.readfile(str(dxf_file))
                comp_msp = comp_doc.modelspace()
                comp_bb = ezdxf.bbox.extents(comp_msp)
                if comp_bb.has_data:
                    # Sprite is rendered at 200 DPI, then scaled by sprite_scale
                    # The sprite PNG size depends on the render DPI and CAD bbox
                    # From phase1_extractor, the sprite is cropped to its content
                    # Typical sprite size from comments in config: 434x998px (interruptor_diferencial)
                    # So on a 640x640 bg with scale 0.04-0.08:
                    # min: 434*0.04 = 17px wide, 998*0.04 = 40px tall
                    # max: 434*0.08 = 35px wide, 998*0.08 = 80px tall
                    pass
            except:
                pass
    
    # Use the documented sprite size from config comment
    sprite_w_base = 434  # documented in config for interruptor_diferencial
    sprite_h_base = 998
    print(f"  Example sprite (interruptor_diferencial): {sprite_w_base} x {sprite_h_base} px (base)")
    print(f"  After scale {sprite_scale_min}: {sprite_w_base*sprite_scale_min:.0f} x {sprite_h_base*sprite_scale_min:.0f} px")
    print(f"  After scale {sprite_scale_max}: {sprite_w_base*sprite_scale_max:.0f} x {sprite_h_base*sprite_scale_max:.0f} px")
    
    if insert_data and mask.sum() > 0:
        print(f"\n  COMPARISON:")
        print(f"    Real symbols in plan (median): {median_w:.1f} x {median_h:.1f} px")
        print(f"    Synthetic training (min scale): {sprite_w_base*sprite_scale_min:.0f} x {sprite_h_base*sprite_scale_min:.0f} px")
        print(f"    Synthetic training (max scale): {sprite_w_base*sprite_scale_max:.0f} x {sprite_h_base*sprite_scale_max:.0f} px")
    
    print(f"\n=== Profiling Complete ===")


if __name__ == "__main__":
    profile_dxf(DXF_PATH)
