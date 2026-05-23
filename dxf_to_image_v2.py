"""
dxf_to_image_v2.py — Renderizador DXF mejorado para planos con coordenadas
dispersas (e.g., contenido real en 60x30 CAD dentro de un espacio de 3700x1400).

Mejoras sobre v1:
  - Usa scale_analyzer_v2 (sin ezdxf.bbox, filtra bloques anonimos y gigantes)
  - Renderiza SOLO la region de contenido (extent de INSERTs significativos)
  - Filtro espacial previo al render: elimina entidades fuera de la region
    -> critico para DXFs grandes (>10 MB) con contenido concentrado
  - Sube max_dim_px automaticamente si simbolos quedarian < MIN_SYM_PX
  - Drop-in replacement de dxf_to_image.renderizar_dxf()
"""

import os, json
import ezdxf
from ezdxf.addons.drawing import RenderContext, Frontend
from ezdxf.addons.drawing.matplotlib import MatplotlibBackend
from ezdxf.addons.drawing.config import Configuration
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from scale_analyzer_v2 import calcular_factor_escala_v2

PAD_PX              = 48
MIN_SYM_PX          = 32
MAX_DIM_PX_DEFAULT  = 16000
FILTRO_BUFFER_FACTOR = 8.0


# ── Filtro espacial ──────────────────────────────────────────────────────────

def _pos(ent):
    try:
        t = ent.dxftype()
        if t == "INSERT":
            ip = ent.dxf.insert; return (ip.x, ip.y)
        if t == "LINE":
            s,e = ent.dxf.start, ent.dxf.end; return ((s.x+e.x)/2,(s.y+e.y)/2)
        if t in ("CIRCLE","ARC"):
            c = ent.dxf.center; return (c.x, c.y)
        if t in ("TEXT","MTEXT"):
            ip = ent.dxf.insert; return (ip.x, ip.y)
        if t == "LWPOLYLINE":
            pts = list(ent.get_points())
            if pts: return (sum(p[0] for p in pts)/len(pts), sum(p[1] for p in pts)/len(pts))
        if t == "POLYLINE":
            vs = list(ent.vertices)
            if vs: return (sum(v.dxf.location.x for v in vs)/len(vs),
                           sum(v.dxf.location.y for v in vs)/len(vs))
        if hasattr(ent.dxf,"insert"):
            ip = ent.dxf.insert; return (ip.x, ip.y)
    except Exception:
        pass
    return None


def _filtrar(msp, x0, y0, x1, y1, buf):
    kill = []
    for ent in msp:
        p = _pos(ent)
        if p is None: continue
        if not (x0-buf <= p[0] <= x1+buf and y0-buf <= p[1] <= y1+buf):
            kill.append(ent)
    for ent in kill:
        try: msp.delete_entity(ent)
        except: pass
    return len(kill)


# ── Helpers de render ────────────────────────────────────────────────────────

def _apagar_capas(doc, capas_incluir):
    s = {c.upper() for c in capas_incluir}
    for l in doc.layers:
        if l.dxf.name.upper() not in s: l.off()


def _negro(doc):
    for l in doc.layers:
        try: l.color = 7
        except: pass
    for e in doc.modelspace():
        try:
            if hasattr(e.dxf,"color"): e.dxf.color = 256
        except: pass


def _negro_textos_mem(doc):
    """Fuerza negro (true color RGB) solo en TEXT/MTEXT/ATTRIBs, sin tocar simbolos."""
    for e in doc.modelspace():
        t = e.dxftype()
        if t in ("TEXT", "MTEXT", "ATTRIB", "ATTDEF"):
            try: e.rgb = (0, 0, 0)
            except: pass
        if t == "INSERT":
            try:
                for att in e.attribs:
                    att.rgb = (0, 0, 0)
            except: pass


def _blanco_textos_mem(doc):
    """Fuerza blanco (true color RGB) en TEXT/MTEXT/ATTRIBs → invisibles sobre fondo blanco."""
    for e in doc.modelspace():
        t = e.dxftype()
        if t in ("TEXT", "MTEXT", "ATTRIB", "ATTDEF"):
            try: e.rgb = (255, 255, 255)
            except: pass
        if t == "INSERT":
            try:
                for att in e.attribs:
                    att.rgb = (255, 255, 255)
            except: pass


def _bw(path, modo):
    if modo == "color": return
    img = Image.open(path)
    if modo == "grayscale": img = img.convert("L").convert("RGB")
    elif modo == "binary":  img = img.convert("L").point(lambda p: 255 if p>200 else 0).convert("RGB")
    img.save(path)


def _limites(msp, px_cad, region, max_dim, tam_sym):
    aviso = None
    if region:
        pad = (tam_sym*1.5) if tam_sym else 0.1
        x0,y0 = region["x_min"]-pad, region["y_min"]-pad
        x1,y1 = region["x_max"]+pad, region["y_max"]+pad
        origen = "region_contenido"
    else:
        # Intenta ezdxf.bbox.extents (como v1) para capturar toda la geometria,
        # incluyendo cables/lineas que salen de la region de INSERTs.
        try:
            import ezdxf.bbox as _bbox
            bb = _bbox.extents(msp)
            if bb.has_data:
                x0, y0 = bb.extmin.x, bb.extmin.y
                x1, y1 = bb.extmax.x, bb.extmax.y
                origen = "bbox_ezdxf"
            else:
                raise RuntimeError("bbox sin datos")
        except Exception as _e:
            print(f"[v2/bbox  ] ezdxf.bbox falló ({_e}), usando midpoints")
            coords = [_pos(e) for e in msp]
            coords = [c for c in coords if c]
            if not coords: raise RuntimeError("No se pudo calcular bbox del DXF.")
            x0 = min(c[0] for c in coords); x1 = max(c[0] for c in coords)
            y0 = min(c[1] for c in coords); y1 = max(c[1] for c in coords)
            origen = "bbox_midpoints"

    aw, ah = x1-x0, y1-y0
    if px_cad is None:
        px_cad = max_dim/max(aw,ah) if max(aw,ah)>0 else 1.0

    dim = max(aw*px_cad, ah*px_cad)
    if dim > max_dim:
        pf = px_cad * max_dim/dim
        if tam_sym and tam_sym*pf < MIN_SYM_PX:
            pf = MIN_SYM_PX/tam_sym
            aviso = (f"max_dim_px={max_dim} dejaba simbolos a {tam_sym*(px_cad*max_dim/dim):.1f}px. "
                     f"Ajustado para simbolos de {MIN_SYM_PX}px.")
    else:
        pf = px_cad

    print(f"[v2/limites] origen={origen}  {aw:.3f}x{ah:.3f} CAD  px/cad={pf:.4f}")
    if tam_sym: print(f"[v2/limites] simbolo esperado: {tam_sym*pf:.1f} px")
    if aviso:   print(f"[v2/aviso ] {aviso}")
    return x0, y0, x1, y1, pf, aviso


# ── API publica ──────────────────────────────────────────────────────────────

def renderizar_dxf(dxf_path, output_path,
                   capas_incluir=None, target_px=64,
                   modo_color="color", max_dim_px=MAX_DIM_PX_DEFAULT,
                   texto_negro=False):
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # Leer DXF una sola vez y reutilizar en scale_analyzer (evita doble lectura)
    import time as _t; _t0 = _t.time()
    doc = ezdxf.readfile(dxf_path)
    print(f"[v2/read  ] {dxf_path}  ({_t.time()-_t0:.1f}s)")
    msp = doc.modelspace()

    px_raw, ref, region = calcular_factor_escala_v2(doc, target_px=target_px)
    n_sym = region["n_simbolos"] if region else 0
    print(f"[v2/scale ] {px_raw:.4f} px/CAD  ref={ref}  simbolos={n_sym}")

    if capas_incluir: _apagar_capas(doc, capas_incluir)
    if modo_color == "mono": _negro(doc)
    if texto_negro: _negro_textos_mem(doc)
    else: _blanco_textos_mem(doc)  # siempre fuerza texto blanco en render de deteccion

    tam_sym = ref[1] if (ref and ref[1]) else None

    # ── Auto-detección: crop a región de contenido vs. extent completo ────────
    # Si el contenido (INSERTs) ocupa la mayor parte del DXF → render completo
    # (DXFs tipo TGBT donde no hay coordenadas dispersas).
    # Si el contenido es pequeño vs. el DXF total → crop (DXFs tipo plano5).
    usar_region = True
    if region:
        all_pos = [_pos(e) for e in msp]
        all_pos = [p for p in all_pos if p]
        if all_pos:
            fw = max(max(p[0] for p in all_pos) - min(p[0] for p in all_pos), 0.001)
            fh = max(max(p[1] for p in all_pos) - min(p[1] for p in all_pos), 0.001)
            cw = region["x_max"] - region["x_min"]
            ch = region["y_max"] - region["y_min"]
            ratio = max(cw / fw, ch / fh)
            if ratio > 0.4:          # contenido ocupa >40% del DXF en al menos una dimension
                usar_region = False  # → render completo, sin filtro espacial
                print(f"[v2/auto  ] ratio={ratio:.2f} → extent completo (DXF compacto, sin filtro)")
            else:
                print(f"[v2/auto  ] ratio={ratio:.2f} → crop a región de contenido")

    x0, y0, x1, y1, px, aviso = _limites(msp, px_raw,
                                          region if usar_region else None,
                                          max_dim_px, tam_sym)

    # Filtro espacial (solo si se usa región de contenido)
    n_antes = sum(1 for _ in msp)
    if usar_region:
        buf = (tam_sym*FILTRO_BUFFER_FACTOR) if tam_sym else max(x1-x0, y1-y0)*0.1
        n_elim = _filtrar(msp, x0, y0, x1, y1, buf)
        n_despues = sum(1 for _ in msp)
        if n_elim > 0:
            print(f"[v2/filtro] {n_elim} entidades eliminadas  ({n_despues}/{n_antes} conservadas)")
    else:
        print(f"[v2/filtro] omitido (modo extent completo)")

    aw, ah = x1-x0, y1-y0
    wpx = int(round(aw*px)); hpx = int(round(ah*px))
    pad_cad = PAD_PX/px
    print(f"[v2/render] imagen final: {wpx+2*PAD_PX} x {hpx+2*PAD_PX} px")

    fig = plt.figure(figsize=((wpx+2*PAD_PX)/100.0, (hpx+2*PAD_PX)/100.0), dpi=100)
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0,0,1,1])
    ax.set_xlim(x0-pad_cad, x1+pad_cad)
    ax.set_ylim(y0-pad_cad, y1+pad_cad)
    ax.axis("off")

    ctx = RenderContext(doc)
    Frontend(ctx, MatplotlibBackend(ax), config=Configuration.defaults()).draw_layout(msp, finalize=False)

    plt.savefig(output_path, dpi=100, facecolor="white", edgecolor="none",
                pil_kwargs={"compress_level": 3})
    plt.close(fig)
    _bw(output_path, modo_color if modo_color in ("grayscale","binary") else "color")
    print(f"[v2/color ] modo={modo_color}")

    meta = {
        "dxf_path": os.path.abspath(dxf_path),
        "image_path": os.path.abspath(output_path),
        "px_per_cad": px,
        "x_min_cad": x0-pad_cad, "y_min_cad": y0-pad_cad,
        "x_max_cad": x1+pad_cad, "y_max_cad": y1+pad_cad,
        "image_width_px": wpx+2*PAD_PX, "image_height_px": hpx+2*PAD_PX,
        "scale_reference": {"tipo": ref[0], "valor_cad": ref[1]},
        "capas_incluir": capas_incluir, "modo_color": modo_color,
        "render_v2": True, "region_contenido": region, "aviso": aviso,
    }
    meta_path = os.path.splitext(output_path)[0] + ".json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[v2/render] imagen -> {output_path}")
    print(f"[v2/render] meta   -> {meta_path}")
    return meta


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="dxf_to_image_v2")
    p.add_argument("dxf")
    p.add_argument("out", nargs="?", default=None)
    p.add_argument("--target-px",  type=int, default=64)
    p.add_argument("--max-dim-px", type=int, default=MAX_DIM_PX_DEFAULT)
    p.add_argument("--bw", default="color", choices=["color","grayscale","mono","binary"])
    p.add_argument("--capas", nargs="*", default=None)
    a = p.parse_args()
    salida = a.out or os.path.splitext(a.dxf)[0]+"_v2.png"
    renderizar_dxf(a.dxf, salida, capas_incluir=a.capas,
                   target_px=a.target_px, modo_color=a.bw, max_dim_px=a.max_dim_px)
