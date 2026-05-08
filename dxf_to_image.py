"""
Renderiza un DXF a UNA imagen PNG con escala consistente entre planos.
Genera tambien un .json con los metadatos necesarios para mapear
detecciones (pixeles) -> coordenadas CAD.

A diferencia del enfoque de tiles en CAD: aca una sola imagen, y el
slicing lo hace SAHI directamente sobre la imagen final.
"""

import os
import json

import ezdxf
from ezdxf.addons.drawing import RenderContext, Frontend
from ezdxf.addons.drawing.matplotlib import MatplotlibBackend
from ezdxf.addons.drawing.config import Configuration
import matplotlib.pyplot as plt
from PIL import Image

from scale_analyzer import calcular_factor_escala

# Limite de pixeles por lado para no quedarte sin RAM (32k cabe en 16GB)
MAX_DIM_PX = 16000
PAD_PX = 32  # padding negro/blanco para que simbolos en el borde no queden cortados


def _aplicar_filtro_capas(doc, capas_incluir):
    """Apaga las capas que no estan en capas_incluir."""
    capas_set = {c.upper() for c in capas_incluir}
    for layer in doc.layers:
        if layer.dxf.name.upper() not in capas_set:
            layer.off()


def _forzar_color_negro(doc):
    """
    Setea color 7 (negro/blanco segun fondo) a TODAS las capas y entidades.
    Es mas robusto que depender de Configuration.color_policy entre versiones.
    """
    # Color 7 en AutoCAD = blanco sobre fondo oscuro, negro sobre fondo claro
    for layer in doc.layers:
        try:
            layer.color = 7
        except Exception:
            pass
    for entity in doc.modelspace():
        try:
            if hasattr(entity.dxf, "color"):
                entity.dxf.color = 256  # 256 = "BYLAYER", asi hereda el negro de la capa
        except Exception:
            pass


def _post_procesar_bw(image_path, modo):
    """
    modo: "color" (no toca), "grayscale" (gris en 3 canales), "binary" (1-bit B&N).
    Mantiene 3 canales RGB para que YOLO no se queje del input.
    """
    if modo == "color":
        return
    img = Image.open(image_path)
    if modo == "grayscale":
        img = img.convert("L").convert("RGB")
    elif modo == "binary":
        img = img.convert("L").point(lambda p: 255 if p > 200 else 0).convert("RGB")
    else:
        raise ValueError(f"modo_color desconocido: {modo}")
    img.save(image_path)


def renderizar_dxf(dxf_path, output_path, capas_incluir=None, target_px=64,
                    modo_color="color"):
    """
    Renderiza dxf_path -> output_path.
    capas_incluir: si se pasa, solo esas capas se dibujan.
    modo_color: "color" | "grayscale" | "mono" | "binary".
        - "mono"      : fuerza todas las entidades a negro DURANTE el render.
        - "grayscale" : render normal y luego post-procesa a gris (3 canales).
        - "binary"    : render normal y luego umbraliza a B&N puro.
        - "color"     : default, sin cambios.
    Devuelve dict con metadatos (tambien guardado como JSON al lado).
    """
    px_per_cad, ref = calcular_factor_escala(dxf_path, target_px=target_px)
    print(f"[scale] {px_per_cad:.4f} px/CAD  (ref={ref[0]}={ref[1]:.4f})")

    doc = ezdxf.readfile(dxf_path)
    msp = doc.modelspace()

    if capas_incluir:
        _aplicar_filtro_capas(doc, capas_incluir)

    if modo_color == "mono":
        _forzar_color_negro(doc)

    bbox = ezdxf.bbox.extents(msp)
    if not bbox.has_data:
        raise RuntimeError("ModelSpace vacio o sin bbox calculable.")

    x_min, y_min = bbox.extmin.x, bbox.extmin.y
    x_max, y_max = bbox.extmax.x, bbox.extmax.y
    ancho_cad = x_max - x_min
    alto_cad = y_max - y_min

    ancho_px = int(round(ancho_cad * px_per_cad))
    alto_px = int(round(alto_cad * px_per_cad))

    # Cap por seguridad de memoria
    if max(ancho_px, alto_px) > MAX_DIM_PX:
        factor = MAX_DIM_PX / max(ancho_px, alto_px)
        px_per_cad *= factor
        ancho_px = int(round(ancho_cad * px_per_cad))
        alto_px = int(round(alto_cad * px_per_cad))
        print(f"[scale] limitado a {ancho_px}x{alto_px} (cap={MAX_DIM_PX})")

    pad_cad = PAD_PX / px_per_cad

    # matplotlib usa pulgadas; mantenemos dpi=100 para que figsize*100 = px
    fig_w = (ancho_px + 2 * PAD_PX) / 100.0
    fig_h = (alto_px + 2 * PAD_PX) / 100.0

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=100)
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(x_min - pad_cad, x_max + pad_cad)
    ax.set_ylim(y_min - pad_cad, y_max + pad_cad)
    ax.axis("off")

    ctx = RenderContext(doc)
    config = Configuration.defaults()
    backend = MatplotlibBackend(ax)
    Frontend(ctx, backend, config=config).draw_layout(msp, finalize=False)

    plt.savefig(output_path, dpi=100, facecolor="white", edgecolor="none",
                pil_kwargs={"compress_level": 3})
    plt.close(fig)

    # Post-procesado a B&N si corresponde (grayscale / binary)
    _post_procesar_bw(output_path, modo_color if modo_color in ("grayscale", "binary") else "color")
    print(f"[color] modo={modo_color}")

    metadata = {
        "dxf_path": os.path.abspath(dxf_path),
        "image_path": os.path.abspath(output_path),
        "px_per_cad": px_per_cad,
        # Limites del canvas (incluyen padding) - se usan para mapear px -> CAD
        "x_min_cad": x_min - pad_cad,
        "y_min_cad": y_min - pad_cad,
        "x_max_cad": x_max + pad_cad,
        "y_max_cad": y_max + pad_cad,
        "image_width_px": ancho_px + 2 * PAD_PX,
        "image_height_px": alto_px + 2 * PAD_PX,
        "scale_reference": {"tipo": ref[0], "valor_cad": ref[1]},
        "capas_incluir": capas_incluir,
        "modo_color": modo_color,
    }
    meta_path = os.path.splitext(output_path)[0] + ".json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"[render] imagen -> {output_path}")
    print(f"[render] meta   -> {meta_path}")
    return metadata


if __name__ == "__main__":
    import sys
    dxf = sys.argv[1] if len(sys.argv) > 1 else "plano.dxf"
    out = sys.argv[2] if len(sys.argv) > 2 else "plano_render.png"
    renderizar_dxf(dxf, out)
