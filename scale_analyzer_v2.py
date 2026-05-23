"""
scale_analyzer_v2.py — Analizador de escala mejorado.

Mejoras sobre v1:
  - No usa ezdxf.bbox.extents() (falla en algunas versiones)
  - Filtra bloques anonimos (*U*, etc.) y bloques gigantes
  - Devuelve region de contenido (extent de INSERTs significativos)
  - Acepta ruta (str) o doc ya cargado para evitar doble lectura
"""

import ezdxf
import numpy as np
from collections import Counter

LAYERS_EXCLUIR = {"CAJETIN","TITLE","TITULO","BORDE","FRAME","DEFPOINTS","VIEWPORT"}
TAMANO_SIMBOLO_OBJETIVO_PX = 64
ALTURA_TEXTO_OBJETIVO_PX   = 16


def _safe_layer(entity):
    try: return entity.dxf.layer.upper()
    except: return ""


def _block_bbox(blk, sx=1.0, sy=1.0):
    xs, ys = [], []
    for e in blk:
        try:
            t = e.dxftype()
            if t == "LINE":
                xs += [e.dxf.start.x, e.dxf.end.x]
                ys += [e.dxf.start.y, e.dxf.end.y]
            elif t in ("CIRCLE","ARC"):
                r = float(e.dxf.radius)
                cx, cy = e.dxf.center.x, e.dxf.center.y
                xs += [cx-r, cx+r]; ys += [cy-r, cy+r]
            elif t == "LWPOLYLINE":
                pts = list(e.get_points())
                if pts:
                    xs += [p[0] for p in pts]
                    ys += [p[1] for p in pts]
            elif t == "POLYLINE":
                for v in e.vertices:
                    try: xs.append(v.dxf.location.x); ys.append(v.dxf.location.y)
                    except: pass
            elif hasattr(e.dxf, "insert"):
                xs.append(e.dxf.insert.x); ys.append(e.dxf.insert.y)
        except: pass
    if not xs or not ys: return 0.0, 0.0
    return (max(xs)-min(xs))*abs(sx), (max(ys)-min(ys))*abs(sy)


def analizar_inserts_v2(msp, doc):
    lados, posiciones = [], []
    for ent in msp.query("INSERT"):
        if _safe_layer(ent) in LAYERS_EXCLUIR: continue
        nombre = ent.dxf.name
        if nombre.startswith("*"): continue
        try:
            sx = float(getattr(ent.dxf,"xscale",1.0) or 1.0)
            sy = float(getattr(ent.dxf,"yscale",1.0) or 1.0)
            blk = doc.blocks.get(nombre)
            if blk is None: continue
            ancho, alto = _block_bbox(blk, sx, sy)
            lado = max(ancho, alto)
            if lado > 0:
                lados.append(lado)
                ip = ent.dxf.insert
                posiciones.append((ip.x, ip.y))
        except: continue

    if not lados: return None, None

    arr = np.array(lados)
    pts = np.array(posiciones)
    mediana = float(np.median(arr))
    mask = arr <= mediana * 5
    if mask.sum() == 0: mask = np.ones(len(arr), dtype=bool)
    arr_f = arr[mask]; pts_f = pts[mask]

    region = {
        "x_min": float(pts_f[:,0].min()), "y_min": float(pts_f[:,1].min()),
        "x_max": float(pts_f[:,0].max()), "y_max": float(pts_f[:,1].max()),
        "n_simbolos": int(mask.sum()),
    }
    return float(np.median(arr_f)), region


def analizar_circulos(msp):
    radios = []
    for ent in msp.query("CIRCLE"):
        if _safe_layer(ent) in LAYERS_EXCLUIR: continue
        try:
            r = float(ent.dxf.radius)
            if r > 0: radios.append(r)
        except: continue
    if not radios: return None
    arr = np.array(radios)
    p25, p75 = np.percentile(arr,[25,75]); iqr = p75-p25
    mask = (arr >= p25-1.5*iqr) & (arr <= p75+1.5*iqr)
    return float(np.median(arr[mask] if mask.any() else arr))


def analizar_textos(msp):
    alturas = []
    for ent in msp.query("TEXT MTEXT"):
        if _safe_layer(ent) in LAYERS_EXCLUIR: continue
        try:
            h = float(getattr(ent.dxf,"height",0) or 0)
            if h > 0: alturas.append(h)
        except: continue
    if not alturas: return None
    arr = np.array(alturas)
    bins = np.round(np.log10(arr)*10)/10
    moda = Counter(bins).most_common(1)[0][0]
    return float(np.median(arr[bins==moda]))


def calcular_factor_escala_v2(dxf_path_o_doc, target_px=TAMANO_SIMBOLO_OBJETIVO_PX):
    """
    Devuelve (px_per_cad, ref, region_contenido).
    dxf_path_o_doc: str (ruta) o doc ezdxf ya cargado (evita doble lectura).
    """
    if isinstance(dxf_path_o_doc, str):
        doc = ezdxf.readfile(dxf_path_o_doc)
    else:
        doc = dxf_path_o_doc
    msp = doc.modelspace()

    tam_insert, region = analizar_inserts_v2(msp, doc)

    if tam_insert and tam_insert > 0:
        px_per_cad = target_px / tam_insert
        ref = ("INSERT_lado", tam_insert)
    else:
        region = None
        tam_circle = analizar_circulos(msp)
        if tam_circle:
            px_per_cad = (target_px * 0.5) / tam_circle
            ref = ("CIRCLE_radius", tam_circle)
        else:
            tam_text = analizar_textos(msp)
            if tam_text:
                px_per_cad = ALTURA_TEXTO_OBJETIVO_PX / tam_text
                ref = ("TEXT_height", tam_text)
            else:
                px_per_cad = None
                ref = ("BBOX_fallback", None)

    return px_per_cad, ref, region


if __name__ == "__main__":
    import sys
    dxf = sys.argv[1] if len(sys.argv) > 1 else "plano.dxf"
    px_per_cad, ref, region = calcular_factor_escala_v2(dxf)
    print(f"Archivo       : {dxf}")
    if px_per_cad:
        print(f"Factor escala : {px_per_cad:.4f} px/CAD")
    else:
        print("Factor escala : (fallback bbox)")
    if ref[1]:
        print(f"Referencia    : {ref[0]} = {ref[1]:.4f}")
    else:
        print(f"Referencia    : {ref[0]}")
    if region:
        ancho = region['x_max'] - region['x_min']
        alto  = region['y_max'] - region['y_min']
        print(f"Region content: {ancho:.3f} x {alto:.3f} CAD  ({region['n_simbolos']} simbolos)")
        if px_per_cad:
            print(f"Imagen ideal  : {ancho*px_per_cad:.0f} x {alto*px_per_cad:.0f} px")
