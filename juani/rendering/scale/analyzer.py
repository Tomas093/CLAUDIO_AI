"""
rendering/scale/analyzer.py — Analizador de escala compuesto.

ScaleAnalyzer prueba las estrategias en orden y retorna el primer resultado
exitoso. El orden y las estrategias son configurables por construcción.

Reemplaza: scale_analyzer.py y scale_analyzer_v2.py
"""
from __future__ import annotations

import logging
from typing import List

from .strategies import (
    ScaleEstimate,
    ScaleStrategy,
    InsertScaleStrategy,
    CircleScaleStrategy,
    TextScaleStrategy,
    BboxFallbackStrategy,
)
from config import (
    SCALE_TARGET_SYMBOL_PX,
    SCALE_TARGET_TEXT_PX,
    SCALE_OUTLIER_FACTOR,
    SCALE_EXCLUDED_LAYERS,
)

logger = logging.getLogger(__name__)


class ScaleAnalyzer:
    """
    Analiza un DXF y estima el factor de escala óptimo (px/CAD).

    Usa una cadena de estrategias en orden de prioridad. La primera que
    retorna un resultado gana. Esto reemplaza la cadena if/elif implícita
    de las versiones anteriores, y es extensible sin modificar esta clase.
    """

    def __init__(self, strategies: List[ScaleStrategy]):
        if not strategies:
            raise ValueError("ScaleAnalyzer requiere al menos una estrategia")
        self._strategies = strategies

    @classmethod
    def default(cls) -> "ScaleAnalyzer":
        """
        Construye el analizador con la cadena de estrategias por defecto:
        INSERT → CIRCLE → TEXT → BBOX_fallback
        """
        return cls([
            InsertScaleStrategy(
                target_px       = SCALE_TARGET_SYMBOL_PX,
                excluded_layers = SCALE_EXCLUDED_LAYERS,
                outlier_factor  = SCALE_OUTLIER_FACTOR,
            ),
            CircleScaleStrategy(
                target_px       = SCALE_TARGET_SYMBOL_PX,
                excluded_layers = SCALE_EXCLUDED_LAYERS,
            ),
            TextScaleStrategy(
                target_px       = SCALE_TARGET_TEXT_PX,
                excluded_layers = SCALE_EXCLUDED_LAYERS,
            ),
            BboxFallbackStrategy(),
        ])

    def analyze(self, msp, doc) -> ScaleEstimate:
        """
        Retorna el primer ScaleEstimate exitoso.
        Nunca lanza excepción — BboxFallbackStrategy garantiza siempre un resultado.
        """
        for strategy in self._strategies:
            name = type(strategy).__name__
            try:
                result = strategy.estimate(msp, doc)
                if result is not None:
                    logger.info(
                        "ScaleAnalyzer: %s → %.4f px/CAD  (%s)",
                        name, result.px_per_cad, result.description,
                    )
                    if result.region:
                        n = result.region["n_simbolos"]
                        w = result.region["x_max"] - result.region["x_min"]
                        h = result.region["y_max"] - result.region["y_min"]
                        logger.info(
                            "ScaleAnalyzer: región de contenido %.2f×%.2f CAD (%d símbolos)",
                            w, h, n,
                        )
                    return result
            except Exception as exc:
                logger.warning("ScaleAnalyzer: %s falló con %s: %s", name, type(exc).__name__, exc)

        # No debería llegar aquí nunca (BboxFallbackStrategy siempre retorna algo)
        raise RuntimeError("Ninguna estrategia de escala produjo un resultado")
