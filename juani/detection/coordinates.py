"""
detection/coordinates.py — Fuente única de verdad para conversiones px↔CAD.

Anti-pattern eliminado: la misma fórmula de conversión px→CAD existía en:
  - inference_sahi.py::correr_sahi_batched()
  - inference_sahi.py::mapear_a_cad()
  - spec_extractor.py::_bbox_cad()

Con esta clase centralizada, cualquier bug o cambio en la fórmula se corrige
en un solo lugar. Además, CanvasMeta es inmutable (frozen dataclass), evitando
mutaciones accidentales durante el pipeline.

Convención del sistema de coordenadas:
  - Píxeles: origen en esquina superior-izquierda, Y crece hacia abajo.
  - CAD:     origen en esquina inferior-izquierda, Y crece hacia arriba.
  → La inversión del eje Y es el único punto crítico de la conversión.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class CanvasMeta:
    """
    Metadatos del canvas de renderizado. Immutable — se construye una vez
    y se pasa a todos los módulos que necesitan convertir coordenadas.
    """
    px_per_cad:        float
    x_min_cad:         float
    y_min_cad:         float
    x_max_cad:         float
    y_max_cad:         float
    image_width_px:    int
    image_height_px:   int

    @classmethod
    def from_dict(cls, d: dict) -> "CanvasMeta":
        """Construye desde el dict de metadata guardado por DxfRenderer."""
        return cls(
            px_per_cad     = d["px_per_cad"],
            x_min_cad      = d["x_min_cad"],
            y_min_cad      = d["y_min_cad"],
            x_max_cad      = d["x_max_cad"],
            y_max_cad      = d["y_max_cad"],
            image_width_px = d["image_width_px"],
            image_height_px= d["image_height_px"],
        )

    def to_dict(self) -> dict:
        return {
            "px_per_cad":      self.px_per_cad,
            "x_min_cad":       self.x_min_cad,
            "y_min_cad":       self.y_min_cad,
            "x_max_cad":       self.x_max_cad,
            "y_max_cad":       self.y_max_cad,
            "image_width_px":  self.image_width_px,
            "image_height_px": self.image_height_px,
        }


class CoordinateMapper:
    """
    Convierte entre el espacio de píxeles de la imagen renderizada
    y el espacio CAD del archivo DXF original.

    Todas las conversiones están en un solo lugar para evitar inconsistencias.
    """

    def __init__(self, meta: CanvasMeta):
        self._meta = meta

    # ── px → CAD ─────────────────────────────────────────────────────────────

    def center_px_to_cad(self, cx_px: float, cy_px: float) -> Tuple[float, float]:
        """Convierte el centro de un bbox (en píxeles) a coordenadas CAD."""
        m = self._meta
        x_cad = m.x_min_cad + cx_px / m.px_per_cad
        # Y en imagen crece hacia abajo; Y en CAD crece hacia arriba → invertir
        y_cad = m.y_max_cad - cy_px / m.px_per_cad
        return x_cad, y_cad

    def bbox_px_to_cad(
        self, x1: float, y1: float, x2: float, y2: float
    ) -> Tuple[float, float, float, float]:
        """
        Convierte un bounding box de píxeles a coordenadas CAD.

        Retorna (x_min_cad, y_min_cad, x_max_cad, y_max_cad).

        Nota sobre la inversión de Y:
          y1_px es el borde superior de la caja (valor pequeño en px).
          En CAD, el borde superior corresponde a y_max_cad (valor grande).
          Por eso y1_px → y_max_cad y y2_px → y_min_cad.
        """
        m = self._meta
        x_min_cad = m.x_min_cad + x1 / m.px_per_cad
        x_max_cad = m.x_min_cad + x2 / m.px_per_cad
        y_max_cad = m.y_min_cad + (m.image_height_px - y1) / m.px_per_cad
        y_min_cad = m.y_min_cad + (m.image_height_px - y2) / m.px_per_cad
        return x_min_cad, y_min_cad, x_max_cad, y_max_cad

    # ── CAD → px ─────────────────────────────────────────────────────────────

    def cad_to_px(self, x_cad: float, y_cad: float) -> Tuple[float, float]:
        """Convierte coordenadas CAD a píxeles (útil para debug/visualización)."""
        m = self._meta
        px = (x_cad - m.x_min_cad) * m.px_per_cad
        py = (m.y_max_cad - y_cad) * m.px_per_cad
        return px, py
