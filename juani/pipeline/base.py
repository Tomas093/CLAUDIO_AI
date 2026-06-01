"""
pipeline/base.py — Orquestador principal del pipeline de detección.

Anti-pattern eliminado: correr_pipeline() y correr_pipeline_ensemble()
duplicaban ~80% del código en pipeline_v2.py.

Con Pipeline + EnsemblePipeline:
  - Pipeline maneja el flujo completo.
  - EnsemblePipeline sólo sobreescribe _run_inference() (Template Method).
  - Añadir una etapa nueva (ej: OCR post-detección) = 3 líneas en run().
  - Reemplazar cualquier componente = cambiar 1 argumento en __init__().

Etapas del pipeline:
  1. Render de detección   → imagen con texto invisible (modelo no ve texto)
  2. Inferencia            → detecciones crudas
  3. Post-proceso          → cadena de filtros (borde, NMS, anidadas, conf)
  4. Render de visualización → imagen con texto negro
  5. Visualización         → PNG con bboxes dibujados
  6. Extracción de specs   → textos DXF asignados a cada detección
  7. Escritura de resultados → JSONs en output_dir
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Tuple

from config import PipelineConfig
from rendering.renderer import DxfRenderer
from rendering.color_policy import policy_from_mode
from detection.model import DetectionModel
from detection.slicer import ImageSlicer
from detection.postprocess import PostProcessorChain
from detection.coordinates import CanvasMeta
from extraction.spec_extractor import SpecExtractor
from outputs.result_writer import ResultWriter
from outputs.visualizer import Visualizer

logger = logging.getLogger(__name__)

Detection = Dict


class Pipeline:
    """
    Orquestador del pipeline de detección de componentes eléctricos en DXF.

    Recibe sus dependencias por constructor (DIP) — esto permite reemplazar
    cualquier componente sin tocar la lógica de orquestación.
    """

    def __init__(self, cfg: PipelineConfig):
        self.cfg       = cfg
        self._renderer = DxfRenderer(cfg)
        self._slicer   = ImageSlicer(cfg)
        self._postproc = PostProcessorChain.default(cfg)
        self._extractor= SpecExtractor()
        self._writer   = ResultWriter(cfg.output_dir)
        self._viz      = Visualizer()

    def run(self) -> Tuple[dict, List[Detection]]:
        """
        Ejecuta el pipeline completo.
        Retorna (conteo_por_clase, detecciones_con_specs).
        """
        out = Path(self.cfg.output_dir)
        out.mkdir(parents=True, exist_ok=True)

        logger.info("=" * 60)
        logger.info("Pipeline: %s", self.cfg.dxf_path)
        logger.info("=" * 60)

        # ── Etapa 1: render limpio para detección ─────────────────────────────
        logger.info("[1/7] Render de detección")
        detect_policy = policy_from_mode(self.cfg.color_mode, for_detection=True)
        detect_img, meta = self._renderer.render(
            self.cfg.dxf_path,
            out / "plano_render.png",
            color_policy=detect_policy,
        )
        canvas = CanvasMeta.from_dict(meta)

        # ── Etapa 2: inferencia ───────────────────────────────────────────────
        logger.info("[2/7] Inferencia")
        raw_dets = self._run_inference(str(detect_img), meta)
        logger.info("Detecciones crudas: %d", len(raw_dets))

        # ── Etapa 3: post-proceso ─────────────────────────────────────────────
        logger.info("[3/7] Post-proceso")
        dets = self._postproc.run(raw_dets, meta=meta)

        # ── Etapa 4: render de visualización ──────────────────────────────────
        logger.info("[4/7] Render de visualización")
        viz_policy = policy_from_mode(self.cfg.color_mode, for_detection=False)
        viz_img, _ = self._renderer.render(
            self.cfg.dxf_path,
            out / "plano_render_viz.png",
            color_policy=viz_policy,
        )

        # ── Etapa 5: dibujar bboxes ───────────────────────────────────────────
        logger.info("[5/7] Visualización")
        self._viz.draw(viz_img, dets, out / "deteccion_visual.png")

        # ── Etapa 6: extracción de specs ──────────────────────────────────────
        logger.info("[6/7] Extracción de specs")
        dets_with_specs = self._extractor.extract(
            self.cfg.dxf_path, dets, canvas
        )

        # ── Etapa 7: escribir resultados ──────────────────────────────────────
        logger.info("[7/7] Escritura de resultados")
        self._writer.write(dets, dets_with_specs, meta)

        from collections import Counter
        conteo = Counter(d["clase"] for d in dets)
        logger.info("Pipeline completado: %d componentes detectados", len(dets))
        return dict(conteo), dets_with_specs

    def _run_inference(self, image_path: str, meta: dict) -> List[Detection]:
        """
        Hook para subclases. Por defecto corre un solo modelo.
        EnsemblePipeline sobreescribe esto para agregar múltiples modelos.
        """
        if not self.cfg.model_paths:
            raise ValueError("PipelineConfig.model_paths está vacío")
        model = DetectionModel.load(self.cfg.model_paths[0], self.cfg)
        return self._slicer.run_batched(model, image_path, meta)
