import io
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


def render_dxf_to_rgba(dxf_path: Path, dpi: int = RENDER_DPI) -> np.ndarray:
    doc = ezdxf.readfile(str(dxf_path))
    msp = doc.modelspace()

    fig = plt.figure(figsize=(10, 10), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    ctx = RenderContext(doc)
    backend = MatplotlibBackend(ax)
    Frontend(ctx, backend).draw_layout(msp, finalize=True)

    buf = io.BytesIO()
    fig.savefig(
        buf,
        format="png",
        dpi=dpi,
        facecolor="white",
        bbox_inches="tight",
        pad_inches=0.1,
    )
    plt.close(fig)
    buf.seek(0)

    rgb = np.array(Image.open(buf).convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    _, alpha_mask = cv2.threshold(gray, BINARIZE_THRESHOLD, 255, cv2.THRESH_BINARY_INV)

    rgba = np.zeros((rgb.shape[0], rgb.shape[1], 4), dtype=np.uint8)
    rgba[:, :, 3] = alpha_mask
    return rgba


def crop_to_content(rgba: np.ndarray, padding: int = 8) -> np.ndarray:
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
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Fase 1] Renderizando DXF base: {dxf_path.name}...")
    base_rgba = render_dxf_to_rgba(dxf_path, dpi=dpi)
    base_rgba = crop_to_content(base_rgba)
    print(f"[Fase 1] Tamaño del sprite base: {base_rgba.shape[1]}x{base_rgba.shape[0]} px")

    kernels = [
        int(kernel_min + (kernel_max - kernel_min) * i / max(n_variations - 1, 1))
        for i in range(n_variations)
    ]

    generated: list[Path] = []
    for i, k in enumerate(kernels):
        sprite = apply_dilation(base_rgba, k)
        out_path = output_dir / f"sprite_{i:03d}_k{k:02d}.png"
        Image.fromarray(sprite).save(out_path, format="PNG")
        generated.append(out_path)
        print(f"  [{i + 1:3d}/{n_variations}] {out_path.name}  (kernel={k})")

    print(f"[Fase 1] ✅ {len(generated)} sprites guardados en '{output_dir}'\n")
    return generated

