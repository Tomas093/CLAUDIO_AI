"""
rendering/color_policy.py — Políticas de color para el renderizado DXF.

Anti-pattern eliminado: el parámetro booleano `texto_negro` en renderizar_dxf()
mezclaba decisión de qué color con la lógica de cómo aplicarlo.

Con Protocol + implementaciones:
  - Añadir un nuevo modo (ej: texto por capa) = nueva clase, sin tocar el renderer.
  - El renderer recibe cualquier objeto que implemente ColorPolicy.
  - Testeable de forma aislada.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable
import logging

logger = logging.getLogger(__name__)


@runtime_checkable
class ColorPolicy(Protocol):
    """Interfaz de política de color. Modifica el doc DXF en memoria."""
    def apply(self, doc) -> None: ...
    def description(self) -> str: ...


class NoOpColorPolicy:
    """Color original del DXF — sin modificaciones."""
    def apply(self, doc) -> None:
        pass

    def description(self) -> str:
        return "color_original"


class TextInvisiblePolicy:
    """
    Fuerza texto/atributos a blanco (invisible sobre fondo blanco).
    Uso: render de detección — el modelo ve sólo la geometría de símbolos,
    sin texto que confunda las predicciones.
    """
    def apply(self, doc) -> None:
        count = 0
        for e in doc.modelspace():
            t = e.dxftype()
            if t in ("TEXT", "MTEXT", "ATTRIB", "ATTDEF"):
                try:
                    e.rgb = (255, 255, 255)
                    count += 1
                except Exception:
                    pass
            elif t == "INSERT":
                try:
                    for att in e.attribs:
                        att.rgb = (255, 255, 255)
                        count += 1
                except Exception:
                    pass
        logger.debug("TextInvisiblePolicy: %d entidades de texto forzadas a blanco", count)

    def description(self) -> str:
        return "texto_blanco"


class TextBlackPolicy:
    """
    Fuerza texto/atributos a negro RGB verdadero.
    Uso: render de visualización humana — los specs y etiquetas son legibles.

    Nota: usamos `e.rgb = (0,0,0)` y NO `e.dxf.color = 7`.
    ACI color 7 es context-dependent (blanco sobre fondo oscuro, negro sobre claro),
    mientras que el true color RGB es siempre negro.
    """
    def apply(self, doc) -> None:
        count = 0
        for e in doc.modelspace():
            t = e.dxftype()
            if t in ("TEXT", "MTEXT", "ATTRIB", "ATTDEF"):
                try:
                    e.rgb = (0, 0, 0)
                    count += 1
                except Exception:
                    pass
            elif t == "INSERT":
                try:
                    for att in e.attribs:
                        att.rgb = (0, 0, 0)
                        count += 1
                except Exception:
                    pass
        logger.debug("TextBlackPolicy: %d entidades de texto forzadas a negro", count)

    def description(self) -> str:
        return "texto_negro"


class MonoPolicy:
    """
    Fuerza TODAS las entidades a negro (no sólo texto).
    Uso: entrenamiento de modelos en escala de grises donde el color no aporta.

    Advertencia: cambia colores de símbolos también — puede afectar la calidad
    de detección si el modelo fue entrenado en imágenes a color.
    """
    def apply(self, doc) -> None:
        for layer in doc.layers:
            try:
                layer.color = 7
            except Exception:
                pass
        for e in doc.modelspace():
            try:
                if hasattr(e.dxf, "color"):
                    e.dxf.color = 256  # 256 = BYLAYER → hereda color 7 de la capa
            except Exception:
                pass
        logger.debug("MonoPolicy: todas las entidades forzadas a negro")

    def description(self) -> str:
        return "mono"


def policy_from_mode(color_mode: str, for_detection: bool) -> ColorPolicy:
    """
    Factory que construye la política correcta según el modo y el propósito.

    color_mode: "color" | "grayscale" | "binary" | "mono"
    for_detection: True → render para inferencia (texto invisible)
                   False → render para visualización humana (texto negro)
    """
    if color_mode == "mono":
        return MonoPolicy()
    if for_detection:
        return TextInvisiblePolicy()
    return TextBlackPolicy()
