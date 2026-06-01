"""
detection/postprocess.py — Cadena de filtros post-detección.

Anti-pattern eliminado: los 4 filtros estaban pegados con prints y lógica
propia dentro de ejecutar() en inference_sahi.py.

Con Chain of Responsibility:
  - Cada filtro es testeable de forma aislada.
  - El orden es configurable sin tocar los filtros individuales.
  - Añadir un nuevo filtro = nueva clase, sin modificar el pipeline.
  - El logging está centralizado en cada filtro.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Protocol, runtime_checkable

import numpy as np

from config import (
    PipelineConfig,
    DETECT_BORDER_MARGIN_PX,
    DETECT_BORDER_CONF_SAFE,
    DETECT_IOU_NMS,
    DETECT_IOS_NESTED,
    DETECT_CONF_MIN,
)

logger = logging.getLogger(__name__)

Detection = Dict   # {clase, conf, bbox_px, centro_px, x_cad, y_cad}


@runtime_checkable
class DetectionFilter(Protocol):
    """Interfaz de filtro de detecciones."""
    def filter(self, detections: List[Detection], **ctx) -> List[Detection]: ...
    def name(self) -> str: ...


# ── Filtros individuales ──────────────────────────────────────────────────────

class BorderFilter:
    """
    Descarta detecciones cuyo bbox toca el borde de la imagen.
    Las cajas pegadas al borde suelen ser artefactos del clipping de SAHI.
    Se perdonan las de confianza muy alta (≥ conf_safe).
    """
    def __init__(self, margin_px: int = DETECT_BORDER_MARGIN_PX,
                 conf_safe: float = DETECT_BORDER_CONF_SAFE):
        self._margin    = margin_px
        self._conf_safe = conf_safe

    def name(self) -> str:
        return "BorderFilter"

    def filter(self, detections: List[Detection], *, meta: dict, **_) -> List[Detection]:
        W = meta["image_width_px"]
        H = meta["image_height_px"]
        m = self._margin
        kept, dropped = [], []

        for d in detections:
            x1, y1, x2, y2 = d["bbox_px"]
            touches = x1 <= m or y1 <= m or x2 >= W - m or y2 >= H - m
            if touches and d["conf"] < self._conf_safe:
                dropped.append(d)
            else:
                kept.append(d)

        if dropped:
            logger.info(
                "BorderFilter: %d descartadas (pegadas al borde)  [%d → %d]",
                len(dropped), len(detections), len(kept),
            )
            for d in dropped[:5]:   # log sólo las primeras 5 para no saturar
                x1, y1, x2, y2 = d["bbox_px"]
                logger.debug(
                    "  - %s conf=%.2f bbox=(%.0f,%.0f,%.0f,%.0f)",
                    d["clase"], d["conf"], x1, y1, x2, y2,
                )
        return kept


class AgnosticNMSFilter:
    """
    NMS que ignora la clase. Si dos detecciones cualesquiera se solapan
    > iou_thresh, conserva la de mayor confianza.
    Resuelve confusiones interruptor/tomacorriente en el mismo lugar.
    """
    def __init__(self, iou_thresh: float = DETECT_IOU_NMS):
        self._iou_thresh = iou_thresh

    def name(self) -> str:
        return "AgnosticNMSFilter"

    def filter(self, detections: List[Detection], **_) -> List[Detection]:
        if not detections:
            return []

        boxes = np.array([d["bbox_px"] for d in detections])
        confs = np.array([d["conf"]    for d in detections])
        idxs  = confs.argsort()[::-1]
        keep  = []

        while len(idxs) > 0:
            i = idxs[0]
            keep.append(int(i))
            if len(idxs) == 1:
                break
            rest = idxs[1:]
            xx1  = np.maximum(boxes[i, 0], boxes[rest, 0])
            yy1  = np.maximum(boxes[i, 1], boxes[rest, 1])
            xx2  = np.minimum(boxes[i, 2], boxes[rest, 2])
            yy2  = np.minimum(boxes[i, 3], boxes[rest, 3])
            w    = np.clip(xx2 - xx1, 0, None)
            h    = np.clip(yy2 - yy1, 0, None)
            inter = w * h
            ai   = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
            ar   = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
            iou  = inter / (ai + ar - inter + 1e-9)
            idxs = rest[iou < self._iou_thresh]

        result = [detections[i] for i in keep]
        removed = len(detections) - len(result)
        if removed:
            logger.info("AgnosticNMSFilter: %d suprimidas  [%d → %d]",
                        removed, len(detections), len(result))
        return result


class NestedBoxFilter:
    """
    Suprime detecciones cuya caja está significativamente contenida dentro
    de otra de mayor confianza. Usa IOS (Intersection Over Smaller area),
    más sensible que IoU para cajas de tamaños muy distintos.
    """
    def __init__(self, ios_thresh: float = DETECT_IOS_NESTED,
                 class_agnostic: bool = False):
        self._ios_thresh     = ios_thresh
        self._class_agnostic = class_agnostic

    def name(self) -> str:
        return "NestedBoxFilter"

    def filter(self, detections: List[Detection], **_) -> List[Detection]:
        if not detections:
            return []

        if self._class_agnostic:
            groups = [list(range(len(detections)))]
        else:
            by_class: Dict[str, List[int]] = {}
            for i, d in enumerate(detections):
                by_class.setdefault(d["clase"], []).append(i)
            groups = list(by_class.values())

        to_remove = set()
        log_lines = []

        for group in groups:
            ordered = sorted(group, key=lambda i: -detections[i]["conf"])
            for idx_a, i in enumerate(ordered):
                if i in to_remove:
                    continue
                bi = detections[i]["bbox_px"]
                ai = max((bi[2] - bi[0]) * (bi[3] - bi[1]), 1e-9)
                for j in ordered[idx_a + 1:]:
                    if j in to_remove:
                        continue
                    bj = detections[j]["bbox_px"]
                    aj = max((bj[2] - bj[0]) * (bj[3] - bj[1]), 1e-9)
                    xx1 = max(bi[0], bj[0]); yy1 = max(bi[1], bj[1])
                    xx2 = min(bi[2], bj[2]); yy2 = min(bi[3], bj[3])
                    inter = max(0, xx2 - xx1) * max(0, yy2 - yy1)
                    if inter <= 0:
                        continue
                    ios = inter / min(ai, aj)
                    if ios >= self._ios_thresh:
                        to_remove.add(j)
                        log_lines.append(
                            f"  - {detections[j]['clase']:<25} conf={detections[j]['conf']:.3f}"
                            f"  contenida en otra conf={detections[i]['conf']:.3f}"
                            f"  (IOS={ios:.2f})"
                        )

        if log_lines:
            logger.info(
                "NestedBoxFilter: %d suprimidas  [%d → %d]",
                len(to_remove), len(detections), len(detections) - len(to_remove),
            )
            for line in log_lines[:10]:
                logger.debug(line)

        return [d for i, d in enumerate(detections) if i not in to_remove]


class ConfidenceFilter:
    """Filtro final: descarta detecciones con conf < conf_min."""
    def __init__(self, conf_min: float = DETECT_CONF_MIN):
        self._conf_min = conf_min

    def name(self) -> str:
        return "ConfidenceFilter"

    def filter(self, detections: List[Detection], **_) -> List[Detection]:
        if self._conf_min <= 0:
            return detections
        result = [d for d in detections if d["conf"] >= self._conf_min]
        removed = len(detections) - len(result)
        if removed:
            logger.info(
                "ConfidenceFilter(min=%.2f): %d descartadas  [%d → %d]",
                self._conf_min, removed, len(detections), len(result),
            )
        return result


# ── Cadena de filtros ─────────────────────────────────────────────────────────

class PostProcessorChain:
    """
    Aplica una lista de filtros en secuencia.

    Permite reordenar, desactivar o añadir filtros sin tocar el pipeline
    ni ninguno de los filtros individuales.

    Ejemplo:
        chain = PostProcessorChain.default(cfg)
        detections = chain.run(raw_detections, meta=meta_dict)
    """

    def __init__(self, filters: List[DetectionFilter]):
        self._filters = filters

    @classmethod
    def default(cls, cfg: PipelineConfig) -> "PostProcessorChain":
        """Cadena estándar: borde → NMS → anidadas → confianza."""
        return cls([
            BorderFilter(cfg.border_margin_px, cfg.border_conf_safe),
            AgnosticNMSFilter(cfg.iou_nms),
            NestedBoxFilter(cfg.ios_nested),
            ConfidenceFilter(cfg.conf_min),
        ])

    def run(self, detections: List[Detection], **ctx) -> List[Detection]:
        """Aplica todos los filtros en orden. ctx se pasa a cada filtro."""
        logger.info("PostProcessorChain: %d detecciones crudas", len(detections))
        for f in self._filters:
            detections = f.filter(detections, **ctx)
        logger.info("PostProcessorChain: %d detecciones finales", len(detections))
        return detections
