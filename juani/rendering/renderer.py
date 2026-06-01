"""
rendering/renderer.py — Renderizador DXF a imagen PNG.

Reemplaza: dxf_to_image.py y dxf_to_image_v2.py (unifica ambas versiones).

Responsabilidad: dado un DXF y una política de color, producir una imagen PNG
con metadatos de mapeo px↔CAD. No decide qué política aplicar — eso es del caller.

Anti-patterns eliminados:
  - Versionado en nombre de archivo (_v2).
  - Mezcla de 6 responsabilidades en una función monolítica.
  - Parámetro booleano `texto_negro` (reemplazado por ColorPolicy).
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Optional, Tuple

import ezdxf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from config import (
    PipelineConfig,
    RENDER_PAD_PX,
    RENDER_MIN_SYM_PX,
    RENDER_MAX_DIM_PX,
    RENDER_FILTER_BUFFER_FACTOR,
)
from rendering.color_policy import ColorPolicy, NoOpColorPolicy
from rendering.scale.analyzer import ScaleAnalyzer
from rendering.spatial_filter import SpatialFilter

Image.MAX_IMAGE_PIXELS = None   # planos grandes: desactiva anti-bomb de PIL

logger = logging.getLogger(__name__)

RenderResult = Tuple[Path, dict]   # (ruta_imagen, metadata)


class DxfRenderer:
    """
    Convierte un archivo DXF en una imagen PNG a escala consistente.

    Produce también un .json con los metadatos necesarios para mapear
    detecciones (píxeles) → coordenadas CAD.

    Uso:
        renderer = DxfRenderer(cfg)
        img_path, meta = renderer.render(dxf_path, output_path, policy)
    """

    def __init__(self, cfg: PipelineConfig):
        self._cfg      = cfg
        self._analyzer = ScaleAnalyzer.default()
        self._filter   = SpatialFilter(buffer_factor=RENDER_FILTER_BUFFER_FACTOR)

    def render(
        self,
        dxf_path:     str,
        output_path:  str | Path,
        color_policy: Optional[ColorPolicy] = None,
    ) -> RenderResult:
        """
        Renderiza dxf_path → output_path.

        color_policy: instancia de ColorPolicy a aplicar antes del render.
                      None → NoOpColorPolicy (color original).
        Retorna (path_imagen, dict_metadata).
        """
        if color_policy is None:
            color_policy = NoOpColorPolicy()

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # ── 1. Leer DXF ───────────────────────────────────────────────────────
        t0  = time.time()
        doc = ezdxf.readfile(str(dxf_path))
        msp = doc.modelspace()
        logger.info("DXF leído: %s (%.1fs)", dxf_path, time.time() - t0)

        # ── 2. Política de capas ──────────────────────────────────────────────
        if self._cfg.layers_include:
            self._apply_layer_filter(doc, self._cfg.layers_include)

        # ── 3. Política de color ──────────────────────────────────────────────
        color_policy.apply(doc)
        logger.debug("ColorPolicy aplicada: %s", color_policy.description())

        # ── 4. Análisis de escala ─────────────────────────────────────────────
        estimate   = self._analyzer.analyze(msp, doc)
        px_per_cad = estimate.px_per_cad
        region     = estimate.region
        sym_size   = estimate.region and (
            (estimate.region["x_max"] - estimate.region["x_min"] +
             estimate.region["y_max"] - estimate.region["y_min"]) / (2 * region["n_simbolos"])
            if region and region["n_simbolos"] > 0 else None
        )
        # tamaño de símbolo en CAD (para buffer y logs)
        sym_cad = self._estimate_symbol_cad_size(estimate)

        # ── 5. Filtro espacial (sólo si el contenido es compacto en el DXF) ──
        use_region = False
        if region and SpatialFilter.should_apply(msp, region):
            use_region = True
            self._filter.apply(
                msp,
                (region["x_min"], region["y_min"], region["x_max"], region["y_max"]),
                symbol_size=sym_cad,
            )

        # ── 6. Calcular límites del canvas ────────────────────────────────────
        x0, y0, x1, y1, px_final, aviso = self._compute_limits(
            msp, px_per_cad, region if use_region else None,
            self._cfg.max_dim_px, sym_cad,
        )

        # ── 7. Render matplotlib ──────────────────────────────────────────────
        from ezdxf.addons.drawing import RenderContext, Frontend
        from ezdxf.addons.drawing.matplotlib import MatplotlibBackend
        from ezdxf.addons.drawing.config import Configuration

        aw, ah      = x1 - x0, y1 - y0
        wpx         = int(round(aw * px_final))
        hpx         = int(round(ah * px_final))
        pad_cad     = RENDER_PAD_PX / px_final
        total_w     = wpx + 2 * RENDER_PAD_PX
        total_h     = hpx + 2 * RENDER_PAD_PX

        logger.info("Render: %d×%d px  (%.2f×%.2f CAD @ %.4f px/CAD)",
                    total_w, total_h, aw, ah, px_final)

        fig = plt.figure(
            figsize=(total_w / 100.0, total_h / 100.0), dpi=100
        )
        fig.patch.set_facecolor("white")
        ax  = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(x0 - pad_cad, x1 + pad_cad)
        ax.set_ylim(y0 - pad_cad, y1 + pad_cad)
        ax.axis("off")

        ctx = RenderContext(doc)
        Frontend(ctx, MatplotlibBackend(ax), config=Configuration.defaults()).draw_layout(
            msp, finalize=False
        )
        plt.savefig(
            str(output_path), dpi=100, facecolor="white", edgecolor="none",
            pil_kwargs={"compress_level": 3},
        )
        plt.close(fig)

        # ── 8. Post-proceso de imagen (grayscale / binary) ────────────────────
        self._postprocess_image(output_path, self._cfg.color_mode)

        # ── 9. Metadata ───────────────────────────────────────────────────────
        meta = {
            "dxf_path":       os.path.abspath(str(dxf_path)),
            "image_path":     os.path.abspath(str(output_path)),
            "px_per_cad":     px_final,
            "x_min_cad":      x0 - pad_cad,
            "y_min_cad":      y0 - pad_cad,
            "x_max_cad":      x1 + pad_cad,
            "y_max_cad":      y1 + pad_cad,
            "image_width_px": total_w,
            "image_height_px":total_h,
            "scale_reference":estimate.description,
            "color_policy":   color_policy.description(),
            "color_mode":     self._cfg.color_mode,
            "layers_include": self._cfg.layers_include,
            "region_contenido": region,
            "aviso":          aviso,
            "render_version": "refactored",
        }
        meta_path = output_path.with_suffix(".json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        logger.info("Imagen → %s", output_path)
        logger.info("Meta   → %s", meta_path)
        return output_path, meta

    # ── Helpers privados ──────────────────────────────────────────────────────

    @staticmethod
    def _apply_layer_filter(doc, layers_include):
        keep = {c.upper() for c in layers_include}
        for layer in doc.layers:
            if layer.dxf.name.upper() not in keep:
                layer.off()

    @staticmethod
    def _estimate_symbol_cad_size(estimate) -> Optional[float]:
        """Extrae el tamaño del símbolo en CAD desde la descripción de la estrategia."""
        desc = estimate.description
        if desc.startswith("INSERT_lado="):
            try:
                return float(desc.split("=")[1])
            except Exception:
                pass
        return None

    @staticmethod
    def _compute_limits(msp, px_per_cad, region, max_dim_px, sym_cad):
        """
        Calcula x0, y0, x1, y1 del canvas (en CAD) y el px_per_cad final.
        Ajusta la escala si los símbolos quedarían muy pequeños.
        """
        aviso = None

        if region:
            pad   = (sym_cad * 1.5) if sym_cad else 0.1
            x0, y0 = region["x_min"] - pad, region["y_min"] - pad
            x1, y1 = region["x_max"] + pad, region["y_max"] + pad
        else:
            try:
                import ezdxf.bbox as _bbox
                bb = _bbox.extents(msp)
                if bb.has_data:
                    x0, y0 = bb.extmin.x, bb.extmin.y
                    x1, y1 = bb.extmax.x, bb.extmax.y
                else:
                    raise RuntimeError("bbox sin datos")
            except Exception:
                coords = [SpatialFilter._get_position(e) for e in msp]
                coords = [c for c in coords if c]
                if not coords:
                    raise RuntimeError("No se pudo calcular bbox del DXF.")
                x0 = min(c[0] for c in coords); x1 = max(c[0] for c in coords)
                y0 = min(c[1] for c in coords); y1 = max(c[1] for c in coords)

        aw, ah = x1 - x0, y1 - y0
        if px_per_cad is None:
            px_per_cad = max_dim_px / max(aw, ah) if max(aw, ah) > 0 else 1.0

        dim = max(aw * px_per_cad, ah * px_per_cad)
        if dim > max_dim_px:
            pf = px_per_cad * max_dim_px / dim
            if sym_cad and sym_cad * pf < RENDER_MIN_SYM_PX:
                pf = RENDER_MIN_SYM_PX / sym_cad
                aviso = (
                    f"max_dim_px={max_dim_px} dejaba símbolos a "
                    f"{sym_cad * (px_per_cad * max_dim_px / dim):.1f}px. "
                    f"Ajustado para símbolos de {RENDER_MIN_SYM_PX}px."
                )
        else:
            pf = px_per_cad

        if aviso:
            logger.warning("Render: %s", aviso)

        return x0, y0, x1, y1, pf, aviso

    @staticmethod
    def _postprocess_image(path: Path, mode: str) -> None:
        """Convierte la imagen a escala de grises o binario si corresponde."""
        if mode == "color":
            return
        img = Image.open(path)
        if mode == "grayscale":
            img = img.convert("L").convert("RGB")
        elif mode == "binary":
            img = img.convert("L").point(lambda p: 255 if p > 200 else 0).convert("RGB")
        img.save(str(path))
