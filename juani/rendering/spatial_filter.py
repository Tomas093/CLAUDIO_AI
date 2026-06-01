"""
rendering/spatial_filter.py — Filtro espacial de entidades DXF.

Responsabilidad única: dada una región CAD, eliminar del modelspace todas
las entidades que caen fuera de ella (con buffer configurable).

Extraído de dxf_to_image_v2._filtrar() para que el renderer no mezcle
geometría de filtrado con lógica de renderizado.
"""
from __future__ import annotations

import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

Region = Tuple[float, float, float, float]   # (x0, y0, x1, y1) en CAD


class SpatialFilter:
    """
    Elimina entidades del modelspace que están fuera de una región + buffer.
    Mejora el rendimiento en DXFs grandes donde el contenido relevante ocupa
    una fracción pequeña del espacio CAD total.
    """

    def __init__(self, buffer_factor: float = 8.0):
        """
        buffer_factor: el buffer se calcula como `tam_simbolo × buffer_factor`.
        Un valor mayor → menos entidades eliminadas, render más completo.
        """
        self._buffer_factor = buffer_factor

    def apply(self, msp, region: Region, symbol_size: Optional[float]) -> int:
        """
        Elimina entidades fuera de la región y retorna el número de eliminadas.

        region:      (x0, y0, x1, y1) límites de la región de contenido.
        symbol_size: tamaño de referencia del símbolo en CAD (para calcular buffer).
                     Si es None usa 10% de la diagonal de la región como buffer.
        """
        x0, y0, x1, y1 = region
        buf = (symbol_size * self._buffer_factor) if symbol_size else max(x1 - x0, y1 - y0) * 0.1

        to_delete = []
        for ent in msp:
            pos = self._get_position(ent)
            if pos is None:
                continue
            px, py = pos
            if not (x0 - buf <= px <= x1 + buf and y0 - buf <= py <= y1 + buf):
                to_delete.append(ent)

        for ent in to_delete:
            try:
                msp.delete_entity(ent)
            except Exception:
                pass

        logger.info(
            "SpatialFilter: %d entidades eliminadas (región %.2f×%.2f + buffer %.2f)",
            len(to_delete), x1 - x0, y1 - y0, buf,
        )
        return len(to_delete)

    @staticmethod
    def should_apply(msp, region: dict) -> bool:
        """
        Heurística: aplica el filtro sólo si el contenido (INSERTs) ocupa
        menos del 40% del DXF en al menos una dimensión.
        DXFs "compactos" (tipo TGBT) no necesitan filtro.
        """
        all_pos = []
        for e in msp:
            p = SpatialFilter._get_position(e)
            if p:
                all_pos.append(p)
        if not all_pos:
            return False

        fw = max(max(p[0] for p in all_pos) - min(p[0] for p in all_pos), 0.001)
        fh = max(max(p[1] for p in all_pos) - min(p[1] for p in all_pos), 0.001)
        cw = region["x_max"] - region["x_min"]
        ch = region["y_max"] - region["y_min"]
        ratio = max(cw / fw, ch / fh)

        if ratio > 0.4:
            logger.info(
                "SpatialFilter: omitido (ratio=%.2f → DXF compacto, render completo)", ratio
            )
            return False

        logger.info("SpatialFilter: ratio=%.2f → crop a región de contenido", ratio)
        return True

    @staticmethod
    def _get_position(ent):
        """Extrae una posición representativa de la entidad (centro aproximado)."""
        try:
            t = ent.dxftype()
            if t == "INSERT":
                ip = ent.dxf.insert
                return ip.x, ip.y
            if t == "LINE":
                s, e = ent.dxf.start, ent.dxf.end
                return (s.x + e.x) / 2, (s.y + e.y) / 2
            if t in ("CIRCLE", "ARC"):
                c = ent.dxf.center
                return c.x, c.y
            if t in ("TEXT", "MTEXT"):
                ip = ent.dxf.insert
                return ip.x, ip.y
            if t == "LWPOLYLINE":
                pts = list(ent.get_points())
                if pts:
                    return (
                        sum(p[0] for p in pts) / len(pts),
                        sum(p[1] for p in pts) / len(pts),
                    )
            if t == "POLYLINE":
                vs = list(ent.vertices)
                if vs:
                    return (
                        sum(v.dxf.location.x for v in vs) / len(vs),
                        sum(v.dxf.location.y for v in vs) / len(vs),
                    )
            if hasattr(ent.dxf, "insert"):
                ip = ent.dxf.insert
                return ip.x, ip.y
        except Exception:
            pass
        return None
