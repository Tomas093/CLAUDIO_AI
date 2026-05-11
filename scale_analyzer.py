"""
Analiza un DXF y devuelve el factor de escala (px/CAD) optimo para que un
simbolo electrico tipico mida ~64 px en la imagen renderizada.

Estrategia (en orden de prioridad):
  1. Bloques INSERT: la mediana de su bbox es el tamano caracteristico de simbolos.
  2. CIRCLE pequenos: los simbolos electricos casi siempre tienen un circulo.
  3. TEXT/MTEXT: usamos la moda de las alturas (excluyendo cajetin).
  4. Fallback: bbox global del plano.
"""

import ezdxf
import numpy as np
from collections import Counter

# Capas que NO queremos como referencia (suelen tener texto/figuras gigantes)
LAYERS_EXCLUIR = {
    "CAJETIN", "TITLE", "TITULO", "BORDE", "FRAME",
    "DEFPOINTS", "VIEWPORT",
}

# Tamanos objetivo en pixeles (ideales para YOLOv8 con input 640)
TAMANO_SIMBOLO_OBJETIVO_PX = 64
ALTURA_TEXTO_OBJETIVO_PX = 16


def _safe_layer(entity):
    try:
        return entity.dxf.layer.upper()
    except Exception:
        return ""


def analizar_textos(msp):
    """Devuelve la altura representativa (mediana del cluster mas numeroso)."""
    alturas = []
    for ent in msp.query("TEXT MTEXT"):
        if _safe_layer(ent) in LAYERS_EXCLUIR:
            continue
        try:
            h = float(getattr(ent.dxf, "height", 0) or 0)
        except Exception:
            h = 0
        if h > 0:
            alturas.append(h)
    if not alturas:
        return None

    arr = np.array(alturas)
    # Cluster por bins logaritmicos: agrupa "tipos" de texto naturalmente
    bins = np.round(np.log10(arr) * 10) / 10
    moda = Counter(bins).most_common(1)[0][0]
    cluster = arr[bins == moda]
    return float(np.median(cluster))


def analizar_circulos(msp):
    """Mediana del radio de circulos pequenos (filtra outliers con IQR)."""
    radios = []
    for ent in msp.query("CIRCLE"):
        if _safe_layer(ent) in LAYERS_EXCLUIR:
            continue
        try:
            r = float(ent.dxf.radius)
        except Exception:
            continue
        if r > 0:
            radios.append(r)
    if not radios:
        return None
    arr = np.array(radios)
    p25, p75 = np.percentile(arr, [25, 75])
    iqr = p75 - p25
    mask = (arr >= p25 - 1.5 * iqr) & (arr <= p75 + 1.5 * iqr)
    base = arr[mask] if mask.any() else arr
    return float(np.median(base))


def analizar_inserts(msp):
    """Mediana del lado mayor del bbox de bloques INSERT."""
    diagonales = []
    for ent in msp.query("INSERT"):
        if _safe_layer(ent) in LAYERS_EXCLUIR:
            continue
        try:
            bb = ezdxf.bbox.extents([ent])
            if not bb.has_data:
                continue
            lado = max(bb.size.x, bb.size.y)
            if lado > 0:
                diagonales.append(lado)
        except Exception:
            continue
    if not diagonales:
        return None
    arr = np.array(diagonales)
    # Filtramos outliers (bloques de cajetin, leyendas, etc.)
    p25, p75 = np.percentile(arr, [25, 75])
    iqr = p75 - p25
    mask = (arr >= p25 - 1.5 * iqr) & (arr <= p75 + 1.5 * iqr)
    base = arr[mask] if mask.any() else arr
    return float(np.median(base))


def calcular_factor_escala(dxf_path, target_px=TAMANO_SIMBOLO_OBJETIVO_PX):
    """
    Devuelve (px_per_cad, ref) donde ref es la entidad usada como referencia.
    """
    doc = ezdxf.readfile(dxf_path)
    msp = doc.modelspace()

    tam_insert = analizar_inserts(msp)
    tam_circle = analizar_circulos(msp)
    tam_text = analizar_textos(msp)

    if tam_insert:
        px_per_cad = target_px / tam_insert
        ref = ("INSERT_diag", tam_insert)
    elif tam_circle:
        # El radio del circulo del simbolo es ~50% del lado del simbolo
        px_per_cad = (target_px * 0.5) / tam_circle
        ref = ("CIRCLE_radius", tam_circle)
    elif tam_text:
        px_per_cad = ALTURA_TEXTO_OBJETIVO_PX / tam_text
        ref = ("TEXT_height", tam_text)
    else:
        bbox = ezdxf.bbox.extents(msp)
        diag = max(bbox.size.x, bbox.size.y)
        px_per_cad = 8000.0 / diag if diag > 0 else 1.0
        ref = ("BBOX_fallback", diag)

    return px_per_cad, ref


if __name__ == "__main__":
    import sys
    dxf = sys.argv[1] if len(sys.argv) > 1 else "plano.dxf"
    px_per_cad, ref = calcular_factor_escala(dxf)
    print(f"Archivo: {dxf}")
    print(f"Factor de escala: {px_per_cad:.4f} px/CAD")
    print(f"Referencia usada: {ref[0]} = {ref[1]:.4f} unidades CAD")
