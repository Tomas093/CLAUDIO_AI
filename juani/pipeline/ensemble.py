"""
pipeline/ensemble.py — Pipeline ensemble: agrega detecciones de múltiples modelos.

Anti-pattern eliminado: correr_pipeline_ensemble() duplicaba el 80% de
correr_pipeline(). La única diferencia era el paso de inferencia.

Con Template Method (sobreescribir sólo _run_inference()):
  - EnsemblePipeline reutiliza render, post-proceso, viz, specs y escritura.
  - Añadir un tercer modelo al ensemble = 0 cambios de código.
  - Consistencia garantizada: ensemble y single usan exactamente los mismos filtros.
"""
from __future__ import annotations

import logging
from typing import Dict, List

from detection.model import DetectionModel
from pipeline.base import Pipeline

logger = logging.getLogger(__name__)

Detection = Dict


class EnsemblePipeline(Pipeline):
    """
    Extiende Pipeline corriendo múltiples modelos y acumulando sus detecciones
    antes del post-proceso. Todo lo demás (render, filtros, specs, I/O) es idéntico.

    Uso:
        cfg = PipelineConfig(
            dxf_path    = "plano.dxf",
            model_paths = ["termomagnetico.pt", "diferencial.pt"],
        )
        conteo, dets = EnsemblePipeline(cfg).run()
    """

    def _run_inference(self, image_path: str, meta: dict) -> List[Detection]:
        if len(self.cfg.model_paths) < 2:
            logger.warning(
                "EnsemblePipeline con un solo modelo — considera usar Pipeline."
            )

        all_dets = []
        for i, model_path in enumerate(self.cfg.model_paths, 1):
            logger.info(
                "Ensemble %d/%d: %s", i, len(self.cfg.model_paths), model_path
            )
            model = DetectionModel.load(model_path, self.cfg)
            dets  = self._slicer.run_batched(model, image_path, meta)
            logger.info("  → %d detecciones crudas", len(dets))
            all_dets.extend(dets)

        logger.info("Ensemble: total crudas acumuladas: %d", len(all_dets))
        return all_dets
