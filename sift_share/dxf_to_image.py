"""
Renderiza un DXF a UNA imagen PNG con escala consistente entre planos.
Genera tambien un .json con los metadatos necesarios para mapear
detecciones (pixeles) -> coordenadas CAD.

Es independiente del metodo de deteccion posterior (sirve para YOLO,
SIFT, template matching, etc.).

Requiere: pip install ezdxf matplotlib pillow
"""

import os
import json

import ezdxf
from ezdxf.addons.drawing import RenderContext, Frontend
from ezdxf.addons.drawing.matplotlib import MatplotlibBackend
from ezdxf.addons.drawing.config import Configuration
import matplotlib.pyplot as plt
from PIL import Image

# Deshabilitamos el check anti "decompression bomb" de PIL: las imagenes
# son generadas por nosotros, no es contenido sospechoso. Sin esto, PIL
# rechaza imagenes > ~179 MP.
Image.MAX_IMAGE_PIXELS = None

from scale_analyzer import calcular_factor_escala

# Limite de pixeles por lado por defecto (subible con argumento)
MAX_DIM_PX_DEFAULT = 16000
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
    """
    for layer in doc.layers:
        try:
            layer.color = 7
        except Exception:
            pass
    for entity in doc.modelspace():
        try:
            if hasattr(entity.dxf, "color"):
                entity.dxf.color = 256  # 256 = "BYLAYER", hereda de la capa
        except Exception:
            pass


def _post_procesar_bw(image_path, modo):
    """
    modo: "color" (no toca), "grayscale" (gris en 3 canales), "binary" (1-bit B&N).
    Mantiene 3 canales RGB (PIL lo guarda asi por compatibilidad).
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
                    modo_color="color", max_dim_px=MAX_DIM_PX_DEFAULT):
    """
    Renderiza dxf_path -> output_path con escala consistente.

    Parametros:
      dxf_path: archivo DXF de entrada.
      output_path: PNG de salida.
      capas_incluir: lista de nombres de capa a dibujar; None = todas.
      target_px: tamano objetivo en px de un simbolo tipico (64 para YOLO,
                 100-200 para SIFT).
      modo_color: "color" | "grayscale" | "mono" | "binary".
          - "mono"      : fuerza todas las entidades a negro DURANTE el render.
          - "grayscale" : render normal y luego post-procesa a gris (3 canales).
          - "binary"    : render normal y luego umbraliza a B&N puro.
          - "color"     : default, sin cambios.
      max_dim_px: techo del lado mayor de la imagen (RAM safety).

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
    if max(ancho_px, alto_px) > max_dim_px:
        factor = max_dim_px / max(ancho_px, alto_px)
        px_per_cad *= factor
        ancho_px = int(round(ancho_cad * px_per_cad))
        alto_px = int(round(alto_cad * px_per_cad))
        print(f"[scale] limitado a {ancho_px}x{alto_px} (cap={max_dim_px})")
        print(f"[scale] aviso: simbolos quedaron mas chicos que {target_px}px target.")

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
    _post_procesar_bw(output_path,
                      modo_color if modo_color in ("grayscale", "binary") else "color")
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
    renderizar_dxf(dxf, out, modo_color="grayscale", target_px=150)
