# phase1_extractor.py — Sprite extraction from DXF with per-component naming
# Phase 2 requirements: idempotent (wipes target dir before start),
# component-prefixed filenames, incremental disk writes (no mass RAM).
from __future__ import annotations

import io
import shutil
from pathlib import Path

import cv2
import ezdxf
import matplotlib
import numpy as np
from PIL import Image
from ezdxf.addons.drawing import Frontend, RenderContext
from ezdxf.addons.drawing.matplotlib import MatplotlibBackend

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import (
    BINARIZE_THRESHOLD,
    DILATION_KERNEL_MAX,
    DILATION_KERNEL_MIN,
    DXF_FILE,
    N_SPRITE_VARIATIONS,
    RENDER_DPI,
    SPRITES_DIR,
)


def _forzar_color_negro(doc):
    """Setea color 7 (negro) a TODAS las capas y entidades."""
    for layer in doc.layers:
        try:
            layer.color = 7
        except Exception:
            pass
    for entity in doc.modelspace():
        try:
            if hasattr(entity.dxf, "color"):
                entity.dxf.color = 256
        except Exception:
            pass

def _force_black_recursive(doc):
    """Force ALL entities to true black (RGB 0,0,0), including those inside blocks.
    
    DXF color 7 is 'adaptive' — it shows as white on white backgrounds in
    ezdxf's matplotlib renderer.  We bypass this by setting true_color (RGB
    override) to pure black on every entity in modelspace AND inside every
    block definition.  We also set all layers to color 250 (dark gray close
    to black) as a fallback for BYLAYER entities that somehow miss the
    true_color override.
    """
    # Set all layers to a very dark color (250 = dark gray in ACI)
    for layer in doc.layers:
        try:
            layer.color = 250
        except Exception:
            pass

    def _force_entity_black(entity):
        try:
            # true_color is an RGB override that ezdxf always respects
            entity.dxf.true_color = 0x000000  # Pure black RGB
        except Exception:
            pass

    # Force entities in modelspace
    for entity in doc.modelspace():
        _force_entity_black(entity)

    # Force entities inside ALL block definitions
    for block in doc.blocks:
        for entity in block:
            _force_entity_black(entity)


def render_dxf_to_rgba(dxf_path: Path, dpi: int = RENDER_DPI) -> np.ndarray:
    """Render a DXF file to an RGBA numpy array with transparent background.
    
    Renders the DXF directly using ezdxf + matplotlib with:
    - target_px=1000 so the symbol is large enough for sprite extraction
    - true_color override on ALL entities (including block internals) to
      guarantee black lines regardless of background color
    - White background → threshold to extract alpha mask
    """
    doc = ezdxf.readfile(str(dxf_path))
    msp = doc.modelspace()

    # Force all entities to pure black via true_color RGB override
    _force_black_recursive(doc)

    # Calculate bounding box and image size
    from ezdxf import bbox as ezdxf_bbox
    bb = ezdxf_bbox.extents(msp)
    if not bb.has_data:
        raise RuntimeError(f"ModelSpace vacio o sin bbox en {dxf_path}")

    x_min, y_min = bb.extmin.x, bb.extmin.y
    x_max, y_max = bb.extmax.x, bb.extmax.y
    ancho_cad = x_max - x_min
    alto_cad = y_max - y_min

    # Scale: largest side of the symbol → ~1000 px
    target_px = 1000
    max_cad = max(ancho_cad, alto_cad)
    if max_cad <= 0:
        raise RuntimeError(f"BBox degenerado en {dxf_path}")
    px_per_cad = target_px / max_cad

    ancho_px = int(round(ancho_cad * px_per_cad))
    alto_px = int(round(alto_cad * px_per_cad))
    pad_px = 32
    pad_cad = pad_px / px_per_cad

    fig_w = (ancho_px + 2 * pad_px) / 100.0
    fig_h = (alto_px + 2 * pad_px) / 100.0

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=100)
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(x_min - pad_cad, x_max + pad_cad)
    ax.set_ylim(y_min - pad_cad, y_max + pad_cad)
    ax.axis("off")

    ctx = RenderContext(doc)
    from ezdxf.addons.drawing.config import Configuration
    config = Configuration.defaults()
    backend = MatplotlibBackend(ax)
    Frontend(ctx, backend, config=config).draw_layout(msp, finalize=False)

    # Render to in-memory buffer (avoid temp file)
    buf = io.BytesIO()
    plt.savefig(buf, dpi=100, format="png", facecolor="white", edgecolor="none")
    plt.close(fig)
    buf.seek(0)

    # Decode from buffer
    img_pil = Image.open(buf)
    img_bgr = cv2.cvtColor(np.array(img_pil.convert("RGB")), cv2.COLOR_RGB2BGR)
    buf.close()

    # Convert to RGBA with transparent background
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, alpha = cv2.threshold(img_gray, 250, 255, cv2.THRESH_BINARY_INV)

    # Sanity check
    nonzero = np.count_nonzero(alpha)
    total = alpha.shape[0] * alpha.shape[1]
    print(f"[Fase 1] Render {dxf_path.name}: {img_bgr.shape[1]}x{img_bgr.shape[0]} px, "
          f"alpha nonzero: {nonzero}/{total} ({nonzero/total*100:.1f}%)")
    if nonzero == 0:
        print(f"[Fase 1] WARNING: No se detecto contenido visible en {dxf_path.name}!")

    rgba_img = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGBA)
    rgba_img[:, :, 0:3] = 0  # Black color
    rgba_img[:, :, 3] = alpha

    return rgba_img


def crop_to_content(rgba: np.ndarray, padding: int = 75) -> np.ndarray:
    """Crop an RGBA image to its non-transparent bounding box + padding."""
    alpha = rgba[:, :, 3]
    rows = np.any(alpha > 0, axis=1)
    cols = np.any(alpha > 0, axis=0)

    if not rows.any():
        return rgba

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    rmin = max(0, rmin - padding)
    rmax = min(rgba.shape[0] - 1, rmax + padding)
    cmin = max(0, cmin - padding)
    cmax = min(rgba.shape[1] - 1, cmax + padding)

    return rgba[rmin : rmax + 1, cmin : cmax + 1]


def apply_dilation(rgba: np.ndarray, kernel_size: int) -> np.ndarray:
    """Dilate the alpha channel to simulate thicker drawing lines."""
    if kernel_size <= 1:
        return rgba.copy()

    if kernel_size % 2 == 0:
        kernel_size += 1

    result = rgba.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    result[:, :, 3] = cv2.dilate(rgba[:, :, 3], kernel)
    return result


def generate_sprite_variations(
    dxf_path: Path = DXF_FILE,
    output_dir: Path = SPRITES_DIR,
    n_variations: int = N_SPRITE_VARIATIONS,
    kernel_min: int = DILATION_KERNEL_MIN,
    kernel_max: int = DILATION_KERNEL_MAX,
    dpi: int = RENDER_DPI,
    component_name: str = "",
) -> list[Path]:
    """Generate *n_variations* sprite PNGs with incremental line thickness.

    **Idempotency**: wipes *output_dir* before generating to avoid duplicates
    if the process was previously interrupted.

    **Memory**: each sprite is written to disk immediately after creation —
    no list of images is held in RAM.

    **Naming**: files are prefixed with *component_name* when provided,
    e.g. ``interruptor_termomagnetico_sprite_001_k5.png``.
    """

    # ── Idempotency: clean slate ──────────────────────────────────────────
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prefix = f"{component_name}_" if component_name else ""

    print(f"[Fase 1] Renderizando DXF base: {dxf_path.name}...")
    base_rgba = render_dxf_to_rgba(dxf_path, dpi=dpi)
    base_rgba = crop_to_content(base_rgba)
    
    # Redimensionar si excede el tamaño máximo para asegurar cabida en tiles de 640x640
    # O si es muy pequeña, ampliarla para que la dilatación no destruya los detalles.
    h_base, w_base = base_rgba.shape[:2]
    max_dim = max(h_base, w_base)
    
    # Target dimension around 1500 to allow smooth dilation
    scale_f = 1500.0 / max_dim
    new_w = int(w_base * scale_f)
    new_h = int(h_base * scale_f)
    
    # We use INTER_NEAREST for upscaling to avoid jagged edges which become huge when dilated
    interpolation = cv2.INTER_AREA if scale_f < 1.0 else cv2.INTER_NEAREST
    base_rgba = cv2.resize(base_rgba, (new_w, new_h), interpolation=interpolation)
    print(f"[Fase 1] Sprite redimensionado de {w_base}x{h_base} px a {new_w}x{new_h} px para dilatación óptima")
    
    # Re-threshold the alpha mask after resizing to keep it binary
    _, alpha = cv2.threshold(base_rgba[:, :, 3], 127, 255, cv2.THRESH_BINARY)
    base_rgba[:, :, 3] = alpha
        
    print(f"[Fase 1] Tamaño del sprite base: {base_rgba.shape[1]}x{base_rgba.shape[0]} px")

    # Dynamically adjust kernel_max based on the image size so it doesn't destroy details
    # We cap it at 3% of the minimum dimension, or the original kernel_max, whichever is smaller.
    min_dim = min(new_w if 'new_w' in locals() else w_base, new_h if 'new_h' in locals() else h_base)
    dynamic_kernel_max = min(kernel_max, max(3, int(min_dim * 0.03)))

    kernels = [
        int(kernel_min + (dynamic_kernel_max - kernel_min) * i / max(n_variations - 1, 1))
        for i in range(n_variations)
    ]

    generated: list[Path] = []
    for i, k in enumerate(kernels):
        sprite = apply_dilation(base_rgba, k)
        out_path = output_dir / f"{prefix}sprite_{i:04d}_k{k:02d}.png"
        # Incremental write — sprite is freed at end of loop iteration
        Image.fromarray(sprite).save(out_path, format="PNG")
        generated.append(out_path)

        if (i + 1) % 50 == 0 or (i + 1) == n_variations:
            print(f"  [{i + 1:3d}/{n_variations}] {out_path.name}  (kernel={k})")

    print(f"[Fase 1] ✅ {len(generated)} sprites guardados en '{output_dir}'\n")
    return generated
