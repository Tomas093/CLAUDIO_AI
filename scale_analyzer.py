"""
Analiza un DXF y devuelve el factor de escala (px/CAD) optimo para que un
simbolo electrico tipico mida ~64 px en la imagen renderizada.

Estrategia:
  1. Calcula estimaciones independientes desde INSERTs, CIRCLEs y TEXTs.
  2. Usa clustering logaritmico (robusto a distribuciones multi-modales)
     para encontrar el tamano representativo de cada tipo de entidad.
  3. Prioriza INSERT (representa bloques de simbolos), si no hay, usa CIRCLE, luego TEXT.
"""

import ezdxf
import ezdxf.bbox
import numpy as np
from collections import Counter

# Capas que NO queremos como referencia (suelen tener texto/figuras gigantes)
LAYERS_EXCLUIR = {
    "CAJETIN", "TITLE", "TITULO", "BORDE", "FRAME",
    "DEFPOINTS", "VIEWPORT",
}

# Tamano objetivo en pixeles (ideal para YOLO con input 640 y deteccion de trazos finos)
TAMANO_SIMBOLO_OBJETIVO_PX = 100


def _safe_layer(entity):
    try:
        return entity.dxf.layer.upper()
    except Exception:
        return ""


def _cluster_log_bin(arr):
    if len(arr) == 0:
        return None
    bins = np.round(np.log10(arr) * 10) / 10
    moda = Counter(bins).most_common(1)[0][0]
    cluster = arr[bins == moda]
    return float(np.median(cluster))


def analizar_textos(msp):
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
    return _cluster_log_bin(np.array(alturas))


def analizar_circulos(msp):
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
    return _cluster_log_bin(np.array(radios))


def analizar_inserts(msp):
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
    return _cluster_log_bin(np.array(diagonales))


def calcular_factor_escala(dxf_path, target_px=TAMANO_SIMBOLO_OBJETIVO_PX):
    doc = ezdxf.readfile(dxf_path)
    msp = doc.modelspace()

    tam_insert = analizar_inserts(msp)
    tam_circle = analizar_circulos(msp)
    tam_text = analizar_textos(msp)

    if tam_insert:
        px_per_cad = target_px / tam_insert
        print(f"[scale] Usando INSERT (diag={tam_insert:.4f}) -> {px_per_cad:.2f} px/CAD")
        return px_per_cad, ("INSERT_diag", tam_insert)
    
    if tam_circle:
        diametro = 2 * tam_circle
        px_per_cad = target_px / diametro
        print(f"[scale] Usando CIRCLE (diam={diametro:.4f}) -> {px_per_cad:.2f} px/CAD")
        return px_per_cad, ("CIRCLE_diam", diametro)

    if tam_text:
        ref_cad = tam_text * 4.0
        px_per_cad = target_px / ref_cad
        print(f"[scale] Usando TEXT fallback -> {px_per_cad:.2f} px/CAD")
        return px_per_cad, ("TEXT_height", tam_text)

    bbox = ezdxf.bbox.extents(msp)
    diag = max(bbox.size.x, bbox.size.y) if bbox.has_data else 100.0
    px_per_cad = 8000.0 / diag if diag > 0 else 1.0
    print(f"[scale] WARN: sin senales validas, fallback bbox={diag:.2f} -> {px_per_cad:.2f} px/CAD")
    return px_per_cad, ("BBOX", diag)


if __name__ == "__main__":
    import sys
    dxf = sys.argv[1] if len(sys.argv) > 1 else "plano.dxf"
    px_per_cad, ref = calcular_factor_escala(dxf)
    print(f"Archivo: {dxf}")
    print(f"Factor de escala: {px_per_cad:.4f} px/CAD")
