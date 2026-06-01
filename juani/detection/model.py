"""
detection/model.py — Wrapper de modelo de detección (SAHI + YOLO).

Responsabilidad: cargar el modelo y exponer una interfaz limpia.
El pipeline no depende de SAHI directamente, sólo de DetectionModel.
Esto facilita reemplazar SAHI por otro backend de inferencia en el futuro.
"""
from __future__ import annotations

import logging
from typing import List

from config import PipelineConfig

logger = logging.getLogger(__name__)


class DetectionModel:
    """
    Wrapper sobre AutoDetectionModel de SAHI.

    Anti-pattern eliminado: el modelo SAHI se cargaba en medio de funciones
    que también hacían inferencia y post-proceso. Ahora la carga está separada.
    """

    def __init__(self, sahi_model, conf_threshold: float):
        self._model  = sahi_model
        self._conf   = conf_threshold

    @classmethod
    def load(cls, model_path: str, cfg: PipelineConfig) -> "DetectionModel":
        """Carga el modelo desde disco. Lanza excepción si el archivo no existe."""
        from sahi import AutoDetectionModel

        logger.info("Cargando modelo: %s  (device=%s)", model_path, cfg.device)
        sahi_model = AutoDetectionModel.from_pretrained(
            model_type        = cfg.model_type,
            model_path        = model_path,
            confidence_threshold = cfg.conf_raw,
            device            = cfg.device,
        )
        logger.info("Modelo cargado: %s", model_path)
        return cls(sahi_model, cfg.conf_raw)

    @property
    def inner(self):
        """Acceso al modelo SAHI subyacente (para ImageSlicer)."""
        return self._model

    @property
    def conf_threshold(self) -> float:
        return self._conf
