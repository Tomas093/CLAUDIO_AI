"""
detection/slicer.py — Slicing de imagen + inferencia en batch.

Extrae la lógica de correr_sahi_batched() de inference_sahi.py.
También maneja el cálculo de slice_size adaptativo y el guardado de slices.

Usa CoordinateMapper para convertir coords de tiles a espacio global,
eliminando la conversión inline que existía en correr_sahi_batched.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

from config import PipelineConfig, DETECT_SLICE_MIN, DETECT_SLICE_MAX
from detection.coordinates import CanvasMeta, CoordinateMapper

logger = logging.getLogger(__name__)

Detection = Dict


class ImageSlicer:
    """
    Divide la imagen en tiles y corre inferencia en batch sobre cada uno.

    Ventaja sobre get_sliced_prediction de SAHI: procesa tiles en grupos
    (batch_size) aprovechando la vectorización de YOLO/PyTorch → speedup 3-8x.
    """

    def __init__(self, cfg: PipelineConfig):
        self._cfg = cfg

    def run_batched(
        self,
        model,
        image_path: str | Path,
        meta: dict,
    ) -> List[Detection]:
        """
        Corre slicing + inferencia + mapeo de coordenadas.

        Retorna lista de Detection dicts con coords globales en px y CAD.
        """
        from sahi.slicing import slice_image

        cfg     = self._cfg
        canvas  = CanvasMeta.from_dict(meta)
        mapper  = CoordinateMapper(canvas)

        # ── Calcular slice_size efectivo ──────────────────────────────────────
        eff_slice = self._compute_slice_size(str(image_path))

        # ── Slicing ───────────────────────────────────────────────────────────
        logger.info("Slicing: slice=%d  overlap=%.2f", eff_slice, cfg.overlap)
        slice_result = slice_image(
            image               = str(image_path),
            slice_height        = eff_slice,
            slice_width         = eff_slice,
            overlap_height_ratio= cfg.overlap,
            overlap_width_ratio = cfg.overlap,
            min_area_ratio      = 0.1,
        )
        n_total   = len(slice_result)
        n_batches = (n_total + cfg.batch_size - 1) // cfg.batch_size
        logger.info("Tiles: %d  |  batches: %d  (batch_size=%d)",
                    n_total, n_batches, cfg.batch_size)

        # ── Guardado de slices (opcional) ─────────────────────────────────────
        if cfg.save_slices:
            self._save_slices_debug(image_path, slice_result)

        # ── Batch inference ───────────────────────────────────────────────────
        yolo_model      = model.inner.model
        conf_threshold  = model.conf_threshold
        detections      = []
        t_start         = time.time()

        for batch_idx in range(n_batches):
            t_batch_start = time.time()
            start = batch_idx * cfg.batch_size
            end   = min(start + cfg.batch_size, n_total)

            batch_images  = []
            batch_offsets = []
            for i in range(start, end):
                sl = slice_result[i]
                batch_images.append(sl["image"])
                batch_offsets.append(sl["starting_pixel"])

            results = yolo_model(batch_images, verbose=False, conf=conf_threshold)

            n_batch_dets = 0
            for i, result in enumerate(results):
                x_off, y_off = batch_offsets[i]
                boxes = result.boxes
                if boxes is None or len(boxes) == 0:
                    continue

                xyxy  = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                clses = boxes.cls.cpu().numpy().astype(int)
                names = result.names

                for j in range(len(boxes)):
                    x1, y1, x2, y2 = xyxy[j]
                    gx1 = float(x1 + x_off)
                    gy1 = float(y1 + y_off)
                    gx2 = float(x2 + x_off)
                    gy2 = float(y2 + y_off)
                    cx_px = (gx1 + gx2) / 2.0
                    cy_px = (gy1 + gy2) / 2.0

                    # Conversión usando CoordinateMapper (fuente única de verdad)
                    x_cad, y_cad = mapper.center_px_to_cad(cx_px, cy_px)

                    detections.append({
                        "clase":    names[int(clses[j])],
                        "conf":     float(confs[j]),
                        "bbox_px":  [gx1, gy1, gx2, gy2],
                        "centro_px":[cx_px, cy_px],
                        "x_cad":    x_cad,
                        "y_cad":    y_cad,
                    })
                    n_batch_dets += 1

            t_batch = time.time() - t_batch_start
            t_total = time.time() - t_start
            pct     = end / n_total * 100
            eta     = (t_total / end) * (n_total - end) if end < n_total else 0
            logger.debug(
                "Batch %d/%d  tiles %d-%d  (%.1f%%)  +%d det (acum: %d)  "
                "t=%.2fs  ETA=%.0fs",
                batch_idx + 1, n_batches, start + 1, end,
                pct, n_batch_dets, len(detections), t_batch, eta,
            )

        logger.info(
            "Inferencia completa: %d detecciones crudas en %.1fs",
            len(detections), time.time() - t_start,
        )
        return detections

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _compute_slice_size(self, image_path: str) -> int:
        """Ajusta el slice_size según zoom y min_tiles_per_axis."""
        cfg = self._cfg
        if cfg.zoom == 1.0 and cfg.min_tiles_per_axis is None:
            return cfg.slice_size

        img = cv2.imread(image_path)
        if img is None:
            return cfg.slice_size
        H, W = img.shape[:2]

        if cfg.min_tiles_per_axis is not None and cfg.min_tiles_per_axis > 0:
            size  = min(W, H) // cfg.min_tiles_per_axis
            razon = f"min_tiles={cfg.min_tiles_per_axis}"
        else:
            size  = int(round(cfg.slice_size / cfg.zoom))
            razon = f"zoom={cfg.zoom}"

        size = max(DETECT_SLICE_MIN, min(size, DETECT_SLICE_MAX))
        logger.info("Slice size adaptativo: %d  (%s)", size, razon)
        return size

    @staticmethod
    def _save_slices_debug(image_path, slice_result) -> None:
        """Guarda los tiles como PNGs numerados y un grid overview."""
        import os
        out_dir = Path(str(image_path)).parent / "slices"
        out_dir.mkdir(parents=True, exist_ok=True)

        for idx, sl in enumerate(slice_result):
            tile_path = out_dir / f"slice_{idx:04d}.png"
            cv2.imwrite(str(tile_path), sl["image"])

        img_grid = cv2.imread(str(image_path))
        if img_grid is not None:
            for sl in slice_result:
                x1, y1 = sl["starting_pixel"]
                h, w   = sl["image"].shape[:2]
                cv2.rectangle(img_grid, (x1, y1), (x1 + w, y1 + h), (255, 0, 0), 2)
            cv2.imwrite(str(out_dir / "_grid_overview.png"), img_grid)

        logger.info("Slices guardados: %d tiles en %s", len(slice_result), out_dir)
