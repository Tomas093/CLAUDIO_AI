"""
io/result_writer.py — Escritura de resultados del pipeline.

Responsabilidad única: persistir en disco los resultados del pipeline.
Separar I/O de la lógica de orquestación (Pipeline) sigue SRP y facilita
el testing: se puede mockear ResultWriter para tests sin tocar disco.
"""
from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

Detection = Dict[str, Any]


class ResultWriter:
    """Guarda detecciones (con y sin specs) y metadata en el directorio de salida."""

    def __init__(self, output_dir: str | Path):
        self._out = Path(output_dir)

    def write(
        self,
        detections:       List[Detection],
        detections_specs: List[Detection],
        meta:             dict,
    ) -> None:
        """
        Guarda:
          - detecciones.json            → lista de detecciones finales
          - detecciones_con_specs.json  → ídem con campo "specs"
        Y loggea el resumen final (reemplaza los print() de pipeline_v2.py).
        """
        self._out.mkdir(parents=True, exist_ok=True)

        det_path   = self._out / "detecciones.json"
        specs_path = self._out / "detecciones_con_specs.json"

        with open(det_path, "w", encoding="utf-8") as f:
            json.dump(detections, f, indent=2, ensure_ascii=False)

        with open(specs_path, "w", encoding="utf-8") as f:
            json.dump(detections_specs, f, indent=2, ensure_ascii=False)

        logger.info("Resultados → %s", det_path)
        logger.info("Con specs  → %s", specs_path)

        self._log_summary(detections)

    def write_detections_only(self, detections: List[Detection]) -> Path:
        """Guarda sólo las detecciones (sin specs). Retorna la ruta."""
        path = self._out / "detecciones.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(detections, f, indent=2, ensure_ascii=False)
        logger.info("Detecciones → %s", path)
        return path

    @staticmethod
    def _log_summary(detections: List[Detection]) -> None:
        """Loggea tabla de detecciones individuales y conteo por clase."""
        sorted_dets = sorted(detections, key=lambda d: (d["clase"], -d["conf"]))

        logger.info("-" * 60)
        logger.info("DETECCIONES INDIVIDUALES:")
        logger.info("-" * 60)
        for i, d in enumerate(sorted_dets, 1):
            cx, cy = d["centro_px"]
            logger.info(
                "  %3d. %-28s conf=%.3f  px=(%4d,%4d)  cad=(%.2f,%.2f)",
                i, d["clase"], d["conf"], int(cx), int(cy), d["x_cad"], d["y_cad"],
            )

        conteo = Counter(d["clase"] for d in detections)
        logger.info("-" * 60)
        logger.info("CONTEO DE COMPONENTES:")
        logger.info("-" * 60)
        for clase, n in conteo.most_common():
            confs = [d["conf"] for d in detections if d["clase"] == clase]
            logger.info(
                "  %-28s %3d   conf media=%.3f  min=%.3f  max=%.3f",
                clase.upper(), n, sum(confs) / len(confs), min(confs), max(confs),
            )
