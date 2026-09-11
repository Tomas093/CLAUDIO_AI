# phase1_extractor.py — Extracción de sprites desde DXF.
#
# Cambios respecto de la versión anterior (auditoría H13, H15, H4):
#   * Canal alfa CONTINUO en vez de umbralizado. Antes el sprite salía en
#     negro puro con bordes escalonados, mientras que el render de inferencia
#     sale antialiaseado: un gap de textura que el modelo aprende enseguida.
#   * Interpolación lineal al reescalar y sin re-binarizar después.
#   * Los kernels de dilatación se deduplican y se reporta cuántos grosores
#     efectivos quedaron. Antes `sprite_variations: 20` podía producir un
#     único sprite repetido 20 veces.
#   * El padding pasó a 0: el margen transparente se calcula a partir del
#     kernel de dilatación de cada variante, que es lo único que lo necesita.
from __future__ import annotations

import io
import shutil
from pathlib import Path

import cv2
import ezdxf
import matplotlib
import numpy as np
from PIL import Image
from ezdxf import bbox as ezdxf_bbox
from ezdxf.addons.drawing import Frontend, RenderContext
from ezdxf.addons.drawing.config import Configuration
from ezdxf.addons.drawing.matplotlib import MatplotlibBackend

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Lado mayor del sprite base, en píxeles, antes de dilatar.
BASE_SPRITE_PX = 1200


def _force_black_recursive(doc) -> None:
    """Fuerza todas las entidades a negro puro, también dentro de los bloques.

    El color 7 del DXF es "adaptativo" y se renderiza blanco sobre fondo
    blanco; true_color es un override RGB que ezdxf siempre respeta.
    """
    for layer in doc.layers:
        try:
            layer.color = 250
        except Exception:
            pass

    def _black(entity):
        try:
            entity.dxf.true_color = 0x000000
        except Exception:
            pass

    for entity in doc.modelspace():
        _black(entity)
    for block in doc.blocks:
        for entity in block:
            _black(entity)


def render_dxf_to_rgba(dxf_path: Path, target_px: int = BASE_SPRITE_PX) -> np.ndarray:
    """Renderiza un DXF a RGBA con fondo transparente y alfa continuo.

    El alfa se deriva del gris del render (``255 - gris``), así que conserva
    el antialiasing de matplotlib en vez de convertirlo en un borde duro.
    """
    doc = ezdxf.readfile(str(dxf_path))
    msp = doc.modelspace()
    _force_black_recursive(doc)

    bb = ezdxf_bbox.extents(msp)
    if not bb.has_data:
        raise RuntimeError(f"ModelSpace vacío o sin bbox en {dxf_path}")

    x_min, y_min = bb.extmin.x, bb.extmin.y
    x_max, y_max = bb.extmax.x, bb.extmax.y
    ancho_cad, alto_cad = x_max - x_min, y_max - y_min
    max_cad = max(ancho_cad, alto_cad)
    if max_cad <= 0:
        raise RuntimeError(f"BBox degenerado en {dxf_path}")

    px_per_cad = target_px / max_cad
    ancho_px = max(1, int(round(ancho_cad * px_per_cad)))
    alto_px = max(1, int(round(alto_cad * px_per_cad)))

    pad_px = 8
    pad_cad = pad_px / px_per_cad
    dpi = 100
    fig = plt.figure(
        figsize=((ancho_px + 2 * pad_px) / dpi, (alto_px + 2 * pad_px) / dpi), dpi=dpi
    )
    try:
        fig.patch.set_facecolor("white")
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(x_min - pad_cad, x_max + pad_cad)
        ax.set_ylim(y_min - pad_cad, y_max + pad_cad)
        ax.set_facecolor("white")
        ax.axis("off")

        ctx = RenderContext(doc)
        backend = MatplotlibBackend(ax)
        Frontend(ctx, backend, config=Configuration.defaults()).draw_layout(
            msp, finalize=False
        )

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, facecolor="white", edgecolor="none")
        buf.seek(0)
        img_bgr = cv2.imdecode(np.frombuffer(buf.read(), np.uint8), cv2.IMREAD_COLOR)
        buf.close()
    finally:
        plt.close(fig)

    if img_bgr is None:
        raise RuntimeError(f"El render de {dxf_path.name} no produjo imagen")

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Alfa continuo: conserva el antialiasing en vez de destruirlo con un
    # threshold. Un píxel blanco (255) queda transparente, uno negro opaco.
    alpha = (255 - gray).astype(np.uint8)

    rgba = np.zeros((*gray.shape, 4), dtype=np.uint8)
    rgba[:, :, 0:3] = 0          # trazo negro
    rgba[:, :, 3] = alpha

    ink = float(np.count_nonzero(alpha > 8)) / alpha.size * 100.0
    print(f"[Fase 1] {dxf_path.name}: {img_bgr.shape[1]}x{img_bgr.shape[0]}px, "
          f"tinta {ink:.1f}%")
    if ink < 0.05:
        raise RuntimeError(
            f"{dxf_path.name} renderizó prácticamente vacío ({ink:.3f}% de tinta). "
            f"Revisá el DXF antes de generar sprites."
        )
    return rgba


def crop_to_content(rgba: np.ndarray, padding: int = 0) -> np.ndarray:
    """Recorta al bounding box del contenido no transparente."""
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
    return rgba[rmin:rmax + 1, cmin:cmax + 1]


def alpha_bbox(rgba: np.ndarray, threshold: int = 8) -> tuple[int, int, int, int] | None:
    """Bounding box (x, y, w, h) del contenido visible según el canal alfa.

    Es la única fuente de verdad para la caja YOLO: usar ``shape`` incluye el
    padding transparente y las bandas que agrega la rotación (ver H4).
    """
    alpha = rgba[:, :, 3]
    rows = np.any(alpha > threshold, axis=1)
    cols = np.any(alpha > threshold, axis=0)
    if not rows.any() or not cols.any():
        return None
    r0, r1 = np.where(rows)[0][[0, -1]]
    c0, c1 = np.where(cols)[0][[0, -1]]
    return int(c0), int(r0), int(c1 - c0 + 1), int(r1 - r0 + 1)


def apply_dilation(rgba: np.ndarray, kernel_size: int) -> np.ndarray:
    """Engrosa el trazo dilatando el canal alfa.

    Se agrega un borde del tamaño del kernel para que la dilatación no quede
    recortada contra el límite de la imagen.
    """
    if kernel_size <= 1:
        return rgba.copy()
    if kernel_size % 2 == 0:
        kernel_size += 1

    pad = kernel_size // 2 + 1
    padded = cv2.copyMakeBorder(
        rgba, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(0, 0, 0, 0)
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    padded[:, :, 3] = cv2.dilate(padded[:, :, 3], kernel)
    return padded


def _effective_kernels(kernel_min: int, kernel_max: int, n_variations: int,
                       min_dim: int) -> list[int]:
    """Kernels de dilatación realmente distintos, dentro de un techo sensato.

    El techo evita que un kernel enorme convierta el símbolo en una mancha;
    la deduplicación evita generar el mismo sprite N veces (ver H15).
    """
    dynamic_max = min(kernel_max, max(kernel_min + 1, int(min_dim * 0.03)))
    if n_variations <= 1:
        return [kernel_min]
    span = dynamic_max - kernel_min
    raw = [
        int(round(kernel_min + span * i / (n_variations - 1)))
        for i in range(n_variations)
    ]
    return sorted(set(raw))


def generate_sprite_variations(
    dxf_path: Path,
    output_dir: Path,
    n_variations: int,
    kernel_min: int,
    kernel_max: int,
    component_name: str = "",
    target_px: int = BASE_SPRITE_PX,
) -> list[Path]:
    """Genera los PNG de sprite con distintos grosores de línea.

    Idempotente: borra *output_dir* antes de escribir.
    """
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prefix = f"{component_name}_" if component_name else ""

    base_rgba = render_dxf_to_rgba(dxf_path, target_px=target_px)
    base_rgba = crop_to_content(base_rgba, padding=0)

    h_base, w_base = base_rgba.shape[:2]
    scale_f = target_px / max(h_base, w_base)
    new_w = max(1, int(round(w_base * scale_f)))
    new_h = max(1, int(round(h_base * scale_f)))
    interpolation = cv2.INTER_AREA if scale_f < 1.0 else cv2.INTER_LINEAR
    base_rgba = cv2.resize(base_rgba, (new_w, new_h), interpolation=interpolation)
    # Nada de re-binarizar acá: eso volvía a destruir el antialiasing.

    kernels = _effective_kernels(kernel_min, kernel_max, n_variations, min(new_w, new_h))
    if len(kernels) < n_variations:
        print(f"[Fase 1] {component_name}: {len(kernels)} grosores efectivos de "
              f"{n_variations} pedidos (kernels {kernels[0]}–{kernels[-1]}); "
              f"el resto salían duplicados.")

    generated: list[Path] = []
    for i, k in enumerate(kernels):
        sprite = apply_dilation(base_rgba, k)
        sprite = crop_to_content(sprite, padding=0)
        out_path = output_dir / f"{prefix}sprite_{i:04d}_k{k:02d}.png"
        Image.fromarray(sprite).save(out_path, format="PNG")
        generated.append(out_path)

    print(f"[Fase 1] {len(generated)} sprites en '{output_dir.name}' "
          f"({base_rgba.shape[1]}x{base_rgba.shape[0]}px base)\n")
    return generated
