"""
spec_extractor.py — Extrae los specs de texto asociados a cada componente detectado.

Logica:
  Para cada componente detectado, toma la banda vertical definida por su
  bounding box (y_min_cad .. y_max_cad) y recolecta TODOS los textos del DXF
  que caigan dentro de esa banda y esten a la derecha del borde derecho del
  componente (x_max_cad). Sin limite horizontal — lo que haya en esa banda
  son los specs.
"""

import ezdxf
import json


# ── Extraccion de textos del DXF ─────────────────────────────────────────────

def extraer_textos_dxf(dxf_path):
    """
    Extrae todos los textos del DXF con su posicion CAD.
    Retorna lista de dicts: {"x", "y", "texto"}
    Incluye TEXT, MTEXT y ATTRIBs dentro de INSERTs.
    """
    doc = ezdxf.readfile(dxf_path)
    msp = doc.modelspace()
    textos = []

    for e in msp:
        t = e.dxftype()
        try:
            if t == "TEXT":
                pos = e.dxf.insert
                txt = e.dxf.text.strip()
                if txt:
                    textos.append({"x": pos.x, "y": pos.y, "texto": txt})

            elif t == "MTEXT":
                pos = e.dxf.insert
                txt = e.plain_mtext().strip()
                if txt:
                    textos.append({"x": pos.x, "y": pos.y, "texto": txt})

            elif t == "INSERT":
                for att in e.attribs:
                    try:
                        pos = att.dxf.insert
                        txt = att.dxf.text.strip()
                        if txt:
                            textos.append({"x": pos.x, "y": pos.y, "texto": txt})
                    except Exception:
                        pass
        except Exception:
            pass

    return textos


# ── Conversion bbox_px -> CAD ─────────────────────────────────────────────────

def _bbox_cad(det, meta):
    """
    Convierte bbox_px de la deteccion a coordenadas CAD.
    Retorna (x_min, y_min, x_max, y_max) en CAD.
    """
    px_per_cad = meta["px_per_cad"]
    x0_canvas  = meta["x_min_cad"]
    y0_canvas  = meta["y_min_cad"]
    h_img      = meta["image_height_px"]

    x1_px, y1_px, x2_px, y2_px = det["bbox_px"]

    x_min_cad = x0_canvas + x1_px / px_per_cad
    x_max_cad = x0_canvas + x2_px / px_per_cad
    # Imagen: Y hacia abajo. CAD: Y hacia arriba → invertir
    y_max_cad = y0_canvas + (h_img - y1_px) / px_per_cad
    y_min_cad = y0_canvas + (h_img - y2_px) / px_per_cad

    return x_min_cad, y_min_cad, x_max_cad, y_max_cad


# ── Asignacion de specs ───────────────────────────────────────────────────────

def asignar_specs(detecciones, textos_dxf, meta):
    """
    Para cada deteccion:
      - Arranca desde el centro horizontal del bbox hacia la derecha
      - Banda vertical: y_min a y_max del bbox
      - Limite derecho: x_min del proximo componente detectado en la misma fila
      - Corta si hay un gap mayor que la altura del componente
    """
    bboxes = [_bbox_cad(det, meta) for det in detecciones]

    resultado = []

    for i, (det, (x_min, y_min, x_max, y_max)) in enumerate(zip(detecciones, bboxes)):
        cy      = (y_min + y_max) / 2.0
        alto    = max(y_max - y_min, 0.001)
        x_start = (x_min + x_max) / 2.0

        # Limite derecho = x_min de la proxima deteccion a la derecha en la misma fila
        x_limite = float("inf")
        for j, (bx_min, by_min, bx_max, by_max) in enumerate(bboxes):
            if j == i:
                continue
            by_center  = (by_min + by_max) / 2.0
            alto_medio = max((alto + by_max - by_min) / 2.0, 0.001)
            if abs(by_center - cy) < alto_medio and bx_min > x_max:
                x_limite = min(x_limite, bx_min)

        # Recolectar textos en la banda
        candidatos = []
        for txt in textos_dxf:
            if x_start <= txt["x"] <= x_limite and y_min <= txt["y"] <= y_max:
                candidatos.append({"texto": txt["texto"], "x": txt["x"]})

        candidatos.sort(key=lambda c: c["x"])

        # Cortar en el primer gap mayor que la altura del componente
        specs = []
        for k, c in enumerate(candidatos):
            if k == 0:
                specs.append(c["texto"])
            else:
                if c["x"] - candidatos[k-1]["x"] > alto:
                    break
                specs.append(c["texto"])

        det_enriquecido = dict(det)
        det_enriquecido["specs"] = specs
        resultado.append(det_enriquecido)

    return resultado


# ── API principal ─────────────────────────────────────────────────────────────

def extraer_specs_pipeline(dxf_path, detecciones_json_path, meta_json_path,
                            output_json_path=None):
    """
    Dado el DXF y el JSON de detecciones del pipeline,
    enriquece cada deteccion con sus specs de texto.
    """
    with open(detecciones_json_path) as f:
        detecciones = json.load(f)
    with open(meta_json_path) as f:
        meta = json.load(f)

    print(f"[specs] Extrayendo textos de {dxf_path} ...")
    textos = extraer_textos_dxf(dxf_path)
    print(f"[specs] {len(textos)} textos encontrados en el DXF")

    resultado = asignar_specs(detecciones, textos, meta)

    print("\n" + "-" * 60)
    print("COMPONENTES CON SPECS:")
    print("-" * 60)
    for d in sorted(resultado, key=lambda x: (x["clase"], -x["conf"])):
        specs_str = " | ".join(d["specs"]) if d["specs"] else "(sin specs)"
        print(f"  {d['clase']:<32} conf={d['conf']:.2f}  specs: {specs_str}")

    if output_json_path:
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(resultado, f, indent=2, ensure_ascii=False)
        print(f"\n[specs] guardado -> {output_json_path}")

    return resultado


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Extrae specs de texto para cada componente detectado")
    p.add_argument("--dxf",         required=True, help="Archivo DXF original")
    p.add_argument("--detecciones", required=True, help="JSON de detecciones del pipeline")
    p.add_argument("--meta",        required=True, help="JSON de metadatos del render")
    p.add_argument("--out",         default=None,  help="JSON de salida enriquecido")
    a = p.parse_args()

    extraer_specs_pipeline(
        dxf_path=a.dxf,
        detecciones_json_path=a.detecciones,
        meta_json_path=a.meta,
        output_json_path=a.out,
    )
