"""
rendering/scale/strategies.py — Estrategias de estimación de escala DXF.

Anti-pattern eliminado: la cadena if/elif en calcular_factor_escala_v2() era
un Strategy implícito. Al hacerlo explícito con Protocol:
  - Cada estrategia es testeable de forma aislada.
  - Añadir una nueva estrategia = nueva clase, sin modificar el analyzer.
  - El orden de prioridad es configurable en ScaleAnalyzer.default().

Estrategias disponibles (en orden de prioridad por defecto):
  1. InsertScaleStrategy  — usa la mediana del lado mayor de los INSERTs
  2. CircleScaleStrategy  — usa radios de círculos (filtro IQR)
  3. TextScaleStrategy    — usa la moda de alturas de texto
  4. BboxFallbackStrategy — siempre retorna algo (último recurso)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import FrozenSet, Optional, Protocol, Tuple, runtime_checkable

import numpy as np
from collections import Counter


@dataclass
class ScaleEstimate:
    """Resultado de una estrategia de escala."""
    px_per_cad:  float
    description: str
    region:      Optional[dict] = None  # extent de contenido (sólo InsertStrategy)


@runtime_checkable
class ScaleStrategy(Protocol):
    """Interfaz de estrategia de escala. Retorna None si no aplica."""
    def estimate(self, msp, doc) -> Optional[ScaleEstimate]: ...


# ── Helpers compartidos ───────────────────────────────────────────────────────

def _safe_layer(entity) -> str:
    try:
        return entity.dxf.layer.upper()
    except Exception:
        return ""


def _block_bbox(blk, sx: float = 1.0, sy: float = 1.0) -> Tuple[float, float]:
    """Calcula el ancho y alto de un bloque DXF aplicando escala."""
    xs, ys = [], []
    for e in blk:
        try:
            t = e.dxftype()
            if t == "LINE":
                xs += [e.dxf.start.x, e.dxf.end.x]
                ys += [e.dxf.start.y, e.dxf.end.y]
            elif t in ("CIRCLE", "ARC"):
                r = float(e.dxf.radius)
                cx, cy = e.dxf.center.x, e.dxf.center.y
                xs += [cx - r, cx + r]
                ys += [cy - r, cy + r]
            elif t == "LWPOLYLINE":
                pts = list(e.get_points())
                if pts:
                    xs += [p[0] for p in pts]
                    ys += [p[1] for p in pts]
            elif t == "POLYLINE":
                for v in e.vertices:
                    try:
                        xs.append(v.dxf.location.x)
                        ys.append(v.dxf.location.y)
                    except Exception:
                        pass
            elif hasattr(e.dxf, "insert"):
                xs.append(e.dxf.insert.x)
                ys.append(e.dxf.insert.y)
        except Exception:
            pass
    if not xs or not ys:
        return 0.0, 0.0
    return (max(xs) - min(xs)) * abs(sx), (max(ys) - min(ys)) * abs(sy)


# ── Implementaciones ──────────────────────────────────────────────────────────

class InsertScaleStrategy:
    """
    Estrategia principal: mediana del lado mayor de los bloques INSERT.
    También calcula la región de contenido (extent de INSERTs significativos)
    que el renderer usa para hacer crop en planos con coordenadas dispersas.
    """
    def __init__(
        self,
        target_px:       int,
        excluded_layers: FrozenSet[str],
        outlier_factor:  float,
    ):
        self._target_px      = target_px
        self._excluded       = excluded_layers
        self._outlier_factor = outlier_factor

    def estimate(self, msp, doc) -> Optional[ScaleEstimate]:
        sizes, positions = [], []

        for ent in msp.query("INSERT"):
            if _safe_layer(ent) in self._excluded:
                continue
            name = ent.dxf.name
            if name.startswith("*"):           # bloques anónimos de SAHI/autocad
                continue
            try:
                sx   = float(getattr(ent.dxf, "xscale", 1.0) or 1.0)
                sy   = float(getattr(ent.dxf, "yscale", 1.0) or 1.0)
                blk  = doc.blocks.get(name)
                if blk is None:
                    continue
                w, h = _block_bbox(blk, sx, sy)
                lado = max(w, h)
                if lado > 0:
                    sizes.append(lado)
                    ip = ent.dxf.insert
                    positions.append((ip.x, ip.y))
            except Exception:
                continue

        if not sizes:
            return None

        arr     = np.array(sizes)
        pts     = np.array(positions)
        median  = float(np.median(arr))
        mask    = arr <= median * self._outlier_factor
        if not mask.any():
            mask = np.ones(len(arr), dtype=bool)

        arr_f   = arr[mask]
        pts_f   = pts[mask]
        tam     = float(np.median(arr_f))

        region = {
            "x_min":      float(pts_f[:, 0].min()),
            "y_min":      float(pts_f[:, 1].min()),
            "x_max":      float(pts_f[:, 0].max()),
            "y_max":      float(pts_f[:, 1].max()),
            "n_simbolos": int(mask.sum()),
        }

        return ScaleEstimate(
            px_per_cad  = self._target_px / tam,
            description = f"INSERT_lado={tam:.4f}",
            region      = region,
        )


class CircleScaleStrategy:
    """
    Estrategia de respaldo: mediana de radios de círculos (filtro IQR).
    El radio del círculo de un símbolo eléctrico es ~50% del lado del símbolo.
    """
    def __init__(self, target_px: int, excluded_layers: FrozenSet[str]):
        self._target_px = target_px
        self._excluded  = excluded_layers

    def estimate(self, msp, doc) -> Optional[ScaleEstimate]:
        radios = []
        for ent in msp.query("CIRCLE"):
            if _safe_layer(ent) in self._excluded:
                continue
            try:
                r = float(ent.dxf.radius)
                if r > 0:
                    radios.append(r)
            except Exception:
                continue

        if not radios:
            return None

        arr        = np.array(radios)
        p25, p75   = np.percentile(arr, [25, 75])
        iqr        = p75 - p25
        mask       = (arr >= p25 - 1.5 * iqr) & (arr <= p75 + 1.5 * iqr)
        base       = arr[mask] if mask.any() else arr
        radio      = float(np.median(base))

        return ScaleEstimate(
            px_per_cad  = (self._target_px * 0.5) / radio,
            description = f"CIRCLE_radius={radio:.4f}",
        )


class TextScaleStrategy:
    """
    Estrategia de respaldo: moda de alturas de texto.
    Agrupa por bins logarítmicos y toma la mediana del grupo más frecuente.
    """
    def __init__(self, target_px: int, excluded_layers: FrozenSet[str]):
        self._target_px = target_px
        self._excluded  = excluded_layers

    def estimate(self, msp, doc) -> Optional[ScaleEstimate]:
        alturas = []
        for ent in msp.query("TEXT MTEXT"):
            if _safe_layer(ent) in self._excluded:
                continue
            try:
                h = float(getattr(ent.dxf, "height", 0) or 0)
                if h > 0:
                    alturas.append(h)
            except Exception:
                continue

        if not alturas:
            return None

        arr  = np.array(alturas)
        bins = np.round(np.log10(arr) * 10) / 10
        moda = Counter(bins).most_common(1)[0][0]
        tam  = float(np.median(arr[bins == moda]))

        return ScaleEstimate(
            px_per_cad  = self._target_px / tam,
            description = f"TEXT_height={tam:.4f}",
        )


class BboxFallbackStrategy:
    """
    Estrategia de último recurso: escala para que el plano entre en 8000px.
    Siempre retorna un resultado — nunca retorna None.
    """
    def estimate(self, msp, doc) -> Optional[ScaleEstimate]:
        import ezdxf.bbox as _bbox
        try:
            bb = _bbox.extents(msp)
            if bb.has_data:
                diag = max(bb.size.x, bb.size.y)
                if diag > 0:
                    return ScaleEstimate(
                        px_per_cad  = 8000.0 / diag,
                        description = f"BBOX_fallback={diag:.4f}",
                    )
        except Exception:
            pass

        # Fallback total: coordenadas de entidades
        coords = []
        for e in msp:
            try:
                if hasattr(e.dxf, "insert"):
                    ip = e.dxf.insert
                    coords.append((ip.x, ip.y))
            except Exception:
                pass

        if coords:
            xs = [c[0] for c in coords]
            ys = [c[1] for c in coords]
            diag = max(max(xs) - min(xs), max(ys) - min(ys), 1.0)
            return ScaleEstimate(
                px_per_cad  = 8000.0 / diag,
                description = f"COORDS_fallback={diag:.4f}",
            )

        return ScaleEstimate(px_per_cad=1.0, description="UNIT_fallback")
