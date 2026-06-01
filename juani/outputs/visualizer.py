"""
io/visualizer.py — Dibujo de bounding boxes sobre imagen renderizada.

Responsabilidad única: dada una imagen y una lista de detecciones, producir
una imagen con las cajas y etiquetas dibujadas.

Separado del pipeline para que se pueda reusar de forma independiente
(e.g., en scripts de debug o notebooks).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

import cv2

logger = logging.getLogger(__name__)

Detection = Dict[str, Any]


class Visualizer:
    """Dibuja bounding boxes y etiquetas sobre una imagen."""

    def __init__(self, color=(0, 200, 0), thickness: int = 2, font_scale: float = 0.45):
        self._color      = color
        self._thickness  = thickness
        self._font_scale = font_scale

    def draw(
        self,
        image_path:  str | Path,
        detections:  List[Detection],
        output_path: str | Path,
    ) -> Path:
        """
        Lee image_path, dibuja las detecciones y guarda en output_path.
        Retorna la ruta del archivo generado.
        """
        output_path = Path(output_path)
        img = cv2.imread(str(image_path))
        if img is None:
            logger.warning("Visualizer: no se pudo leer la imagen: %s", image_path)
            return output_path

        for d in detections:
            x1, y1, x2, y2 = (int(v) for v in d["bbox_px"])
            cv2.rectangle(img, (x1, y1), (x2, y2), self._color, self._thickness)
            label = f"{d['clase']} {d['conf']:.2f}"
            cv2.putText(
                img, label, (x1, max(y1 - 5, 12)),
                cv2.FONT_HERSHEY_SIMPLEX, self._font_scale,
                self._color, 1,
            )

        cv2.imwrite(str(output_path), img)
        logger.info("Visual → %s  (%d detecciones)", output_path, len(detections))
        return output_path
