"""
Analiza un DXF y devuelve el factor de escala (px/CAD) optimo para que un
simbolo electrico tipico mida ~64 px en la imagen renderizada.

Estrategia:
  1. Calcula estimaciones independientes desde INSERTs, CIRCLEs y TEXTs.
  2. Usa clustering logaritmico (robusto a distribuciones multi-modales)
     para encontrar el tamano representativo de cada tipo de entidad.
  3. Toma la mediana de las estimaciones primarias (INSERT, CIRCLE).
  4. TEXT se usa como fallback si no hay senales geometricas directas.
"""

import ezdxf
import numpy as np
from collections import Counter

# Capas que NO queremos como referencia (suelen tener texto/figuras gigantes)
LAYERS_EXCLUIR = {
    "CAJETIN", "TITLE", "TITULO", "BORDE", "FRAME",
    "DEFPOINTS", "VIEWPORT",
}

# Tamano objetivo en pixeles (ideal para YOLO con input 640)
TAMANO_SIMBOLO_OBJETIVO_PX = 90


def _safe_layer(entity):
    try:
        return entity.dxf.layer.upper()
    except Exception:
        return ""


def _cluster_log_bin(arr):
    """Agrupa valores por bins logaritmicos y devuelve la mediana del cluster
    mas numeroso.  Robusto a distribuciones multi-modales (ej: puntitos de
    conexion + circulos de referencia conviviendo en el mismo DXF)."""
    if len(arr) == 0:
        return None
    bins = np.round(np.log10(arr) * 10) / 10
    moda = Counter(bins).most_common(1)[0][0]
    cluster = arr[bins == moda]
    return float(np.median(cluster))


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
    return _cluster_log_bin(np.array(alturas))


def analizar_circulos(msp):
    """Mediana del radio del cluster de circulos mas representativo."""
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
    """Mediana del lado mayor del bbox del cluster de INSERTs mas representativo."""
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
    """
    Devuelve (px_per_cad, ref).

    Calcula estimaciones de escala desde multiples senales (INSERT, CIRCLE)
    y toma la mediana.  TEXT se usa como fallback si no hay senales
    geometricas directas.
    """
    doc = ezdxf.readfile(dxf_path)
    msp = doc.modelspace()

    tam_insert = analizar_inserts(msp)
    tam_circle = analizar_circulos(msp)
    tam_text = analizar_textos(msp)

    # --- Senales primarias (miden entidades del tamano de un simbolo) ---
    estimaciones = []
    detalles = []

    if tam_insert:
        e = target_px / tam_insert
        estimaciones.append(e)
        detalles.append(("INSERT_diag", tam_insert, e))

    if tam_circle:
        # Diametro del circulo representativo ≈ tamano caracteristico del simbolo
        diametro = 2 * tam_circle
        e = target_px / diametro
        estimaciones.append(e)
        detalles.append(("CIRCLE_diam", diametro, e))

    # Log de todas las senales encontradas
    for tipo, val, est in detalles:
        print(f"[scale]   {tipo}={val:.4f} -> {est:.2f} px/CAD")
    if tam_text:
        print(f"[scale]   TEXT_height={tam_text:.4f}")

    # --- Consenso ---
    if estimaciones:
        px_per_cad = float(np.median(estimaciones))
        if len(estimaciones) > 1:
            ref = ("CONSENSO", len(estimaciones))
            print(f"[scale] consenso de {len(estimaciones)} senales -> {px_per_cad:.2f} px/CAD")
        else:
            ref = (detalles[0][0], detalles[0][1])

        # Sanity check: si hay texto, verificar que quede legible
        # Texto de referencia en planos electricos suele quedar entre 8-40px
        if tam_text:
            text_px = tam_text * px_per_cad
            if text_px < 6 or text_px > 60:
                # La estimacion geometrica da un resultado absurdo: el texto
                # quedaria microscopico o gigante.  Corregimos usando texto.
                symbol_est = tam_text * 3
                px_per_cad_corr = target_px / symbol_est
                print(f"[scale] WARN: texto a {text_px:.1f}px (fuera de rango 6-60px)")
                print(f"[scale] Corrigiendo con TEXT: simbolo~{symbol_est:.4f} -> {px_per_cad_corr:.2f} px/CAD")
                px_per_cad = px_per_cad_corr
                ref = ("TEXT_correccion", tam_text)

    elif tam_text:
        # Fallback: simbolos ~ 3x altura de texto (relacion conservadora)
        symbol_est = tam_text * 3
        px_per_cad = target_px / symbol_est
        ref = ("TEXT_fallback", tam_text)
        print(f"[scale]   TEXT_fallback: simbolo~{symbol_est:.4f} -> {px_per_cad:.2f} px/CAD")
    else:
        # Ultimo recurso: bbox global
        bbox = ezdxf.bbox.extents(msp)
        diag = max(bbox.size.x, bbox.size.y)
        px_per_cad = 8000.0 / diag if diag > 0 else 1.0
        ref = ("BBOX_fallback", diag)
        print(f"[scale] WARN: sin senales validas, fallback bbox={diag:.2f}")

    return px_per_cad, ref


if __name__ == "__main__":
    import sys
    dxf = sys.argv[1] if len(sys.argv) > 1 else "plano.dxf"
    px_per_cad, ref = calcular_factor_escala(dxf)
    print(f"Archivo: {dxf}")
    print(f"Factor de escala: {px_per_cad:.4f} px/CAD")
    print(f"Referencia usada: {ref[0]} = {ref[1]:.4f}" if isinstance(ref[1], float) else f"Referencia usada: {ref}")

