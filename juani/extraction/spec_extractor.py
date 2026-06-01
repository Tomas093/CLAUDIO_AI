"""
extraction/spec_extractor.py — Extracción de specs de texto para cada detección.

Responsabilidad: dada una detección (bbox en px) y los textos del DXF,
encontrar los textos que corresponden a los specs del componente detectado.

Mejoras sobre spec_extractor.py original:
  - Usa CoordinateMapper en vez de _bbox_cad() propia (fuente única de verdad).
  - La clase SpecExtractor recibe el doc ya cargado si está disponible,
    evitando la doble lectura del DXF (ver método extract vs extract_from_doc).
  - Logging estructurado en vez de prints.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import ezdxf

from detection.coordinates import CanvasMeta, CoordinateMapper

logger = logging.getLogger(__name__)

Detection = Dict[str, Any]


class SpecExtractor:
    """
    Enriquece cada detección con los textos del DXF que le corresponden.

    Algoritmo:
      Para cada componente detectado:
        1. Convertir bbox_px → CAD.
        2. x_start = centro horizontal del bbox.
        3. x_limit = x_min del próximo componente a la derecha en la misma fila.
        4. Recolectar textos en la banda vertical [y_min, y_max] desde x_start.
        5. Ordenar por X y cortar en el primer gap > altura del componente.
    """

    def extract(
        self,
        dxf_path: str | Path,
        detections: List[Detection],
        canvas: CanvasMeta,
    ) -> List[Detection]:
        """
        Carga el DXF y enriquece las detecciones con specs.
        Usar extract_from_doc si el doc ya está en memoria para evitar re-lectura.
        """
        logger.info("SpecExtractor: abriendo DXF para extracción de textos")
        doc = ezdxf.readfile(str(dxf_path))
        return self.extract_from_doc(doc, detections, canvas)

    def extract_from_doc(
        self,
        doc,
        detections: List[Detection],
        canvas: CanvasMeta,
    ) -> List[Detection]:
        """
        Enriquece las detecciones usando un doc ya cargado en memoria.
        Evita la doble lectura del DXF cuando el pipeline ya lo tiene abierto.
        """
        texts  = self._collect_texts(doc.modelspace())
        logger.info("SpecExtractor: %d textos encontrados en el DXF", len(texts))

        mapper = CoordinateMapper(canvas)
        result = []
        for det in detections:
            specs = self._assign_specs(det, detections, texts, mapper)
            result.append({**det, "specs": specs})

        # Log resumen
        logger.info("-" * 60)
        logger.info("COMPONENTES CON SPECS:")
        logger.info("-" * 60)
        for d in sorted(result, key=lambda x: (x["clase"], -x["conf"])):
            specs_str = " | ".join(d["specs"]) if d["specs"] else "(sin specs)"
            logger.info("  %-32s conf=%.2f  specs: %s", d["clase"], d["conf"], specs_str)

        return result

    # ── Lógica interna ────────────────────────────────────────────────────────

    @staticmethod
    def _collect_texts(msp) -> List[Dict]:
        """Extrae todos los textos del modelspace con su posición CAD."""
        texts = []
        for e in msp:
            t = e.dxftype()
            try:
                if t == "TEXT":
                    pos = e.dxf.insert
                    txt = e.dxf.text.strip()
                    if txt:
                        texts.append({"x": pos.x, "y": pos.y, "texto": txt})
                elif t == "MTEXT":
                    pos = e.dxf.insert
                    txt = e.plain_mtext().strip()
                    if txt:
                        texts.append({"x": pos.x, "y": pos.y, "texto": txt})
                elif t == "INSERT":
                    for att in e.attribs:
                        try:
                            pos = att.dxf.insert
                            txt = att.dxf.text.strip()
                            if txt:
                                texts.append({"x": pos.x, "y": pos.y, "texto": txt})
                        except Exception:
                            pass
            except Exception:
                pass
        return texts

    @staticmethod
    def _assign_specs(
        det: Detection,
        all_dets: List[Detection],
        texts: List[Dict],
        mapper: CoordinateMapper,
    ) -> List[str]:
        """
        Asigna specs a una detección usando coordenadas CAD.

        x_start: centro horizontal del bbox (ignoramos la parte izquierda del símbolo).
        x_limit: x_min del próximo componente detectado a la derecha en la misma fila.
        gap_cut: para si el gap entre textos consecutivos supera la altura del componente.
        """
        x1, y1, x2, y2 = det["bbox_px"]
        x_min_cad, y_min_cad, x_max_cad, y_max_cad = mapper.bbox_px_to_cad(x1, y1, x2, y2)

        cy   = (y_min_cad + y_max_cad) / 2.0
        alto = max(y_max_cad - y_min_cad, 0.001)
        x_start = (x_min_cad + x_max_cad) / 2.0

        # x_limit: borde izquierdo del próximo componente a la derecha en la misma fila
        x_limit = float("inf")
        for other in all_dets:
            if other is det:
                continue
            ox1, oy1, ox2, oy2 = other["bbox_px"]
            ox_min, oy_min, ox_max, oy_max = mapper.bbox_px_to_cad(ox1, oy1, ox2, oy2)
            o_cy        = (oy_min + oy_max) / 2.0
            o_alto      = max(oy_max - oy_min, 0.001)
            alto_medio  = max((alto + o_alto) / 2.0, 0.001)
            if abs(o_cy - cy) < alto_medio and ox_min > x_max_cad:
                x_limit = min(x_limit, ox_min)

        # Recolectar candidatos en la banda vertical del componente
        candidatos = [
            {"texto": t["texto"], "x": t["x"]}
            for t in texts
            if x_start <= t["x"] <= x_limit and y_min_cad <= t["y"] <= y_max_cad
        ]
        candidatos.sort(key=lambda c: c["x"])

        # Gap-cutting: cortar si el espacio entre textos supera la altura del componente
        specs: List[str] = []
        for k, c in enumerate(candidatos):
            if k == 0:
                specs.append(c["texto"])
            else:
                if c["x"] - candidatos[k - 1]["x"] > alto:
                    break
                specs.append(c["texto"])

        return specs
