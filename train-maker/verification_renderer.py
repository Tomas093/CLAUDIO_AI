# verification_renderer.py — Pre-Phase-1 Visual Label Verification
#
# Generates two PNG images per DXF for visual sanity-checking of the
# synthetic labelling pipeline:
#
#   1. <component>_render.png   — clean rendering of the DXF component
#   2. <component>_bbox.png     — same rendering with the YOLO bounding box
#                                 drawn on top (red rectangle, normalised
#                                 coordinates converted back to pixel space)
#
# Output directory: train-maker/verification/<component_name>/
#
# Usage (standalone):
#   cd train-maker/
#   python verification_renderer.py                        # all components
#   python verification_renderer.py --component interruptor_termomagnetico
#
# Usage (from pipeline):
#   from verification_renderer import run_verification
#   run_verification(cfg)          # runs for all components before Phase 1
from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

# Force UTF-8 output encoding for Windows terminals to support emojis/box borders
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import cv2
import matplotlib
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── ezdxf imports ─────────────────────────────────────────────────────────────
import ezdxf
from ezdxf import bbox as dxf_bbox
from ezdxf.addons.drawing import Frontend, RenderContext
from ezdxf.addons.drawing.config import Configuration
from ezdxf.addons.drawing.matplotlib import MatplotlibBackend

# Root of train-maker/ (this file lives there)
_HERE = Path(__file__).resolve().parent
_VERIFICATION_DIR = _HERE / "verification"

_DEFAULT_DPI = 200


# ── Core rendering ────────────────────────────────────────────────────────────

def _render_dxf_to_rgb(
    dxf_path: Path,
    dpi: int = _DEFAULT_DPI,
    bg_color: str = "white",
) -> np.ndarray:
    """Render a DXF file to an RGB numpy array.

    Parameters
    ----------
    bg_color : str
        Matplotlib colour string for the background (e.g. ``"white"`` or
        ``"#1e1e1e"``).  Pass the result of ``_pick_background_color()``
        so the component is always visible against the background.

    Uses ezdxf.addons.drawing + MatplotlibBackend — the same rendering path
    as phase1_extractor.py so the verification image matches what the pipeline
    actually processes.
    """
    doc = ezdxf.readfile(str(dxf_path))
    msp = doc.modelspace()

    fig = plt.figure(figsize=(8, 8), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.set_facecolor(bg_color)
    fig.patch.set_facecolor(bg_color)

    ctx = RenderContext(doc)
    backend = MatplotlibBackend(ax)
    # lineweight_scaling > 1 makes strokes visibly thicker in the verification
    # image (does NOT affect the actual pipeline sprites).
    cfg = Configuration.defaults()
    cfg = cfg.with_changes(lineweight_scaling=5)
    Frontend(ctx, backend, config=cfg).draw_layout(msp, finalize=True)

    buf = io.BytesIO()
    fig.savefig(
        buf,
        format="png",
        dpi=dpi,
        facecolor=bg_color,
        bbox_inches="tight",
        pad_inches=0.05,
    )
    plt.close(fig)
    buf.seek(0)

    rgb = np.array(Image.open(buf).convert("RGB"))
    return rgb


def _pick_background_color(dxf_path: Path, dpi: int = _DEFAULT_DPI) -> str:
    """Choose a background colour that contrasts with the DXF component.

    Strategy
    --------
    1. Render the DXF on a **mid-gray** (#808080) background — neutral enough
       that neither white nor black lines disappear into it.
    2. Separate pixels that are close to mid-gray (background) from those
       that are not (component strokes).
    3. Compute the mean luminance of the stroke pixels:
       - If strokes are **bright** (luminance > 128) → the component is light
         coloured → use a **dark** background (``#1e1e1e``).
       - If strokes are **dark** (luminance ≤ 128) → the component is dark
         coloured → use a **white** background.

    Falls back to ``"white"`` if no stroke pixels are detected.
    """
    PROBE_BG = "#808080"           # neutral mid-gray probe render
    PROBE_RGB = (128, 128, 128)    # same colour as numpy array
    TOLERANCE = 30                 # pixels within ±30 of probe are "background"

    rgb = _render_dxf_to_rgb(dxf_path, dpi=min(dpi, 100), bg_color=PROBE_BG)

    # Mask pixels that differ significantly from the probe background
    diff = np.abs(rgb.astype(np.int32) - np.array(PROBE_RGB, dtype=np.int32))
    is_stroke = np.max(diff, axis=2) > TOLERANCE   # shape (H, W)

    if not np.any(is_stroke):
        # Fallback: nothing detected, default to white bg
        return "white"

    stroke_pixels = rgb[is_stroke]                 # shape (N, 3)
    mean_luminance = float(np.mean(
        0.299 * stroke_pixels[:, 0] +
        0.587 * stroke_pixels[:, 1] +
        0.114 * stroke_pixels[:, 2]
    ))

    if mean_luminance > 128:
        # Light/white strokes → dark background
        return "#1e1e1e"
    else:
        # Dark/black strokes → white background
        return "white"


def _compute_modelspace_bbox_yolo(dxf_path: Path) -> tuple[float, float, float, float] | None:
    """Compute the normalised YOLO bbox (cx, cy, w, h) from the DXF modelspace extents.

    Returns None if the extents cannot be determined (empty or degenerate file).
    """
    doc = ezdxf.readfile(str(dxf_path))
    msp = doc.modelspace()

    try:
        box = dxf_bbox.extents(msp)
    except Exception:
        return None

    if box is None or box.extmin is None or box.extmax is None:
        return None

    x0, y0, _ = box.extmin
    x1, y1, _ = box.extmax

    dxf_w = x1 - x0
    dxf_h = y1 - y0

    if dxf_w <= 0 or dxf_h <= 0:
        return None

    # In DXF space the bbox IS the entire modelspace content, so the
    # normalised YOLO bbox is always the whole image: cx=0.5, cy=0.5,
    # w=1.0, h=1.0 when tightly cropped to the content.
    # This function exists for transparency and future extension where
    # only a sub-region of the modelspace is the component of interest.
    cx, cy, w, h = 0.5, 0.5, 1.0, 1.0
    return cx, cy, w, h


def _draw_bbox_on_image(
    rgb: np.ndarray,
    cx: float,
    cy: float,
    w: float,
    h: float,
    *,
    label: str = "componente (class 0)",
    color: tuple[int, int, int] = (220, 38, 38),   # Tailwind red-600
    thickness: int = 3,
) -> np.ndarray:
    """Draw a YOLO bounding box rectangle on an RGB image and return the result.

    Parameters
    ----------
    rgb : np.ndarray
        Source image (H × W × 3, uint8).
    cx, cy, w, h : float
        Normalised YOLO coordinates (0–1).
    label : str
        Text label rendered in the top-left corner of the box.
    color : RGB tuple
        Rectangle + label colour.
    thickness : int
        Rectangle line thickness in pixels.
    """
    result = rgb.copy()
    img_h, img_w = result.shape[:2]

    # Convert normalised → pixel coordinates
    px_cx = int(cx * img_w)
    px_cy = int(cy * img_h)
    px_w  = int(w  * img_w)
    px_h  = int(h  * img_h)

    x1 = max(0, px_cx - px_w  // 2)
    y1 = max(0, px_cy - px_h  // 2)
    x2 = min(img_w - 1, px_cx + px_w  // 2)
    y2 = min(img_h - 1, px_cy + px_h  // 2)

    # OpenCV works in BGR
    bgr_color = (color[2], color[1], color[0])

    cv2.rectangle(result, (x1, y1), (x2, y2), bgr_color, thickness)

    # Label background pill
    font          = cv2.FONT_HERSHEY_SIMPLEX
    font_scale    = max(0.5, img_w / 1200)
    text_thickness = max(1, thickness - 1)
    (tw, th), baseline = cv2.getTextSize(label, font, font_scale, text_thickness)

    lbl_x1 = x1
    lbl_y1 = max(0, y1 - th - baseline - 6)
    lbl_x2 = x1 + tw + 8
    lbl_y2 = y1

    cv2.rectangle(result, (lbl_x1, lbl_y1), (lbl_x2, lbl_y2), bgr_color, -1)
    cv2.putText(
        result,
        label,
        (lbl_x1 + 4, lbl_y2 - baseline - 2),
        font,
        font_scale,
        (255, 255, 255),
        text_thickness,
        cv2.LINE_AA,
    )

    # Corner crosshair at centre point (helpful for debugging)
    cross_len = max(8, thickness * 3)
    cv2.line(result, (px_cx - cross_len, px_cy), (px_cx + cross_len, px_cy), bgr_color, max(1, thickness - 1))
    cv2.line(result, (px_cx, px_cy - cross_len), (px_cx, px_cy + cross_len), bgr_color, max(1, thickness - 1))

    return result


# ── Public API ─────────────────────────────────────────────────────────────────

def render_verification_images(
    dxf_path: Path,
    component_name: str,
    output_dir: Path = _VERIFICATION_DIR,
    dpi: int = _DEFAULT_DPI,
    class_id: int = 0,
) -> tuple[Path, Path]:
    """Render and save two verification PNGs for *dxf_path*.

    Generates:
      - ``<output_dir>/<component_name>/<stem>_render.png``  — clean render
      - ``<output_dir>/<component_name>/<stem>_bbox.png``    — render + bbox

    Parameters
    ----------
    dxf_path : Path
        Source DXF file.
    component_name : str
        Component name (used as subfolder inside *output_dir*).
    output_dir : Path
        Root verification directory.  Defaults to ``train-maker/verification/``.
    dpi : int
        Rendering DPI (matches the pipeline default of 200).
    class_id : int
        Class ID written on the bbox label (always 0 in per-component mode).

    Returns
    -------
    (render_path, bbox_path) : tuple[Path, Path]
        Paths to the two generated PNG files.
    """
    comp_dir = output_dir / component_name
    comp_dir.mkdir(parents=True, exist_ok=True)

    stem = dxf_path.stem
    render_path = comp_dir / f"{stem}_render.png"
    bbox_path   = comp_dir / f"{stem}_bbox.png"

    print(f"  [Verificación] Renderizando '{dxf_path.name}'...")

    # Auto-detect background colour so the component is always visible
    bg_color = _pick_background_color(dxf_path, dpi=dpi)
    bg_label = "oscuro" if bg_color != "white" else "blanco"
    print(f"  [Verificación] 🎨 Fondo auto-detectado: {bg_label} ({bg_color})")

    rgb = _render_dxf_to_rgb(dxf_path, dpi=dpi, bg_color=bg_color)

    # 1. Clean render
    # OpenCV saves as BGR so convert first
    cv2.imwrite(str(render_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    print(f"  [Verificación] ✅ Render limpio  → {render_path.relative_to(_HERE)}")

    # 2. Bbox render
    yolo_box = _compute_modelspace_bbox_yolo(dxf_path)
    if yolo_box is None:
        print(
            f"  [Verificación] ⚠️  No se pudo calcular el bbox de '{dxf_path.name}'. "
            "El archivo puede estar vacío o corrupto."
        )
        # Save a copy of the render so bbox_path always exists
        cv2.imwrite(str(bbox_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    else:
        cx, cy, w, h = yolo_box
        label = f"class {class_id}  cx={cx:.3f} cy={cy:.3f} w={w:.3f} h={h:.3f}"
        annotated = _draw_bbox_on_image(
            cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),   # work in BGR from here
            cx, cy, w, h,
            label=label,
        )
        cv2.imwrite(str(bbox_path), annotated)
        print(f"  [Verificación] ✅ Imagen + bbox → {bbox_path.relative_to(_HERE)}")

    return render_path, bbox_path


def run_verification(cfg: "PipelineConfig") -> None:  # type: ignore[name-defined]
    """Run verification for all components defined in *cfg*.

    Called from ``run_pipeline.py`` before Phase 1.  Failures are caught and
    reported as warnings — they never abort the main pipeline.
    """
    print(f"\n{'-' * 60}")
    print("  ▶ VERIFICACIÓN VISUAL — Renders de componentes + bounding boxes")
    print(f"{'-' * 60}\n")
    print(f"  Directorio de salida: verification/\n")

    ok = 0
    failed = 0

    for comp in cfg.components:
        for vi, dxf_path in enumerate(comp.dxf_paths):
            n_variants = len(comp.dxf_paths)
            variant_tag = f" (v{vi + 1}/{n_variants})" if n_variants > 1 else ""
            print(f"  📐 {comp.name}{variant_tag}  ←  {dxf_path.name}")
            try:
                render_verification_images(
                    dxf_path=dxf_path,
                    component_name=comp.name,
                    output_dir=_VERIFICATION_DIR,
                    dpi=cfg.g.render_dpi,
                    class_id=comp.class_id,
                )
                ok += 1
            except Exception as exc:  # noqa: BLE001
                print(f"  [Verificación] ⚠️  ERROR procesando '{dxf_path.name}': {exc}")
                failed += 1

    print()
    summary = f"  ✅ {ok} DXF(s) verificados"
    if failed:
        summary += f"  ⚠️  {failed} con errores (ver logs arriba)"
    print(summary)
    print(f"  📁 Imágenes guardadas en: {_VERIFICATION_DIR}\n")


# ── CLI entry point ────────────────────────────────────────────────────────────

def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="Genera imágenes de verificación visual (render + bbox) para cada DXF."
    )
    parser.add_argument(
        "--component",
        metavar="NAME",
        default=None,
        help="Nombre del componente a verificar (por defecto: todos).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=_DEFAULT_DPI,
        help=f"DPI de renderizado (defecto: {_DEFAULT_DPI}).",
    )
    args = parser.parse_args()

    # Load config from the same directory as this script
    sys.path.insert(0, str(_HERE))
    from config import load_config  # noqa: PLC0415

    cfg = load_config()

    components = cfg.components
    if args.component:
        components = [c for c in components if c.name == args.component]
        if not components:
            print(f"[Error] Componente '{args.component}' no encontrado en components_config.yaml")
            sys.exit(1)

    for comp in components:
        for vi, dxf_path in enumerate(comp.dxf_paths):
            n_variants = len(comp.dxf_paths)
            variant_tag = f" (v{vi + 1}/{n_variants})" if n_variants > 1 else ""
            print(f"\n📐 {comp.name}{variant_tag}  ←  {dxf_path.name}")
            try:
                render_verification_images(
                    dxf_path=dxf_path,
                    component_name=comp.name,
                    output_dir=_VERIFICATION_DIR,
                    dpi=args.dpi,
                    class_id=comp.class_id,
                )
            except Exception as exc:  # noqa: BLE001
                print(f"  ⚠️  ERROR: {exc}")

    print(f"\n✅ Verificación completa. Imágenes en: {_VERIFICATION_DIR}")


if __name__ == "__main__":
    _cli()
