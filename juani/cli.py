"""
cli.py — Punto de entrada único para el pipeline CLAUDIO_AI.

Anti-pattern eliminado: 5 archivos diferentes con if __name__ == "__main__"
y argparse propios, sin relación entre ellos.

Con subcomandos (claudio run / claudio render / claudio specs):
  - Un solo ejecutable, múltiples modos.
  - Logging centralizado con nivel configurable.
  - Construcción de PipelineConfig en un solo lugar.

Uso:
  python cli.py run --dxf plano.dxf --modelo best.pt
  python cli.py run --dxf plano.dxf --modelos m1.pt m2.pt
  python cli.py render --dxf plano.dxf --out render.png
  python cli.py specs --dxf plano.dxf --detecciones det.json --meta meta.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level   = logging.DEBUG if verbose else logging.INFO,
        format  = "%(asctime)s [%(name)-30s] %(levelname)-8s %(message)s",
        datefmt = "%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


# ── Subcomando: run ───────────────────────────────────────────────────────────

def cmd_run(args) -> None:
    """Ejecuta el pipeline completo de detección sobre un DXF."""
    from config import PipelineConfig
    from pipeline.base import Pipeline
    from pipeline.ensemble import EnsemblePipeline

    if args.modelo and args.modelos:
        print("ERROR: usa --modelo O --modelos, no ambos.", file=sys.stderr)
        sys.exit(1)
    if not args.modelo and not args.modelos:
        print("ERROR: debes indicar --modelo o --modelos.", file=sys.stderr)
        sys.exit(1)

    model_paths = [args.modelo] if args.modelo else args.modelos

    cfg = PipelineConfig(
        dxf_path           = args.dxf,
        output_dir         = args.out,
        model_paths        = model_paths,
        target_px          = args.target_px,
        max_dim_px         = args.max_dim_px,
        layers_include     = args.capas,
        color_mode         = args.bw,
        conf_raw           = args.conf,
        conf_min           = args.conf_min,
        iou_nms            = args.iou,
        slice_size         = args.slice,
        overlap            = args.overlap,
        zoom               = args.zoom,
        min_tiles_per_axis = args.min_tiles,
        batch_size         = args.batch_size,
        save_slices        = args.save_slices,
        device             = "cuda" if args.cuda else args.device,
        model_type         = args.model_type,
    )

    pipeline = EnsemblePipeline(cfg) if len(model_paths) > 1 else Pipeline(cfg)
    conteo, dets = pipeline.run()

    print(f"\nDetecciones finales: {len(dets)}")
    for clase, n in sorted(conteo.items(), key=lambda x: -x[1]):
        print(f"  {clase:<30} {n}")


# ── Subcomando: render ────────────────────────────────────────────────────────

def cmd_render(args) -> None:
    """Solo renderiza un DXF a imagen PNG."""
    from config import PipelineConfig
    from rendering.renderer import DxfRenderer
    from rendering.color_policy import policy_from_mode

    cfg = PipelineConfig(
        dxf_path      = args.dxf,
        output_dir    = str(Path(args.out).parent),
        target_px     = args.target_px,
        max_dim_px    = args.max_dim_px,
        layers_include= args.capas,
        color_mode    = args.bw,
    )
    renderer = DxfRenderer(cfg)
    policy   = policy_from_mode(args.bw, for_detection=not args.texto_negro)
    img_path, meta = renderer.render(args.dxf, args.out, color_policy=policy)
    print(f"Imagen → {img_path}")
    print(f"Meta   → {img_path.with_suffix('.json')}")
    if args.print_meta:
        print(json.dumps(meta, indent=2))


# ── Subcomando: specs ─────────────────────────────────────────────────────────

def cmd_specs(args) -> None:
    """Extrae specs de texto para detecciones existentes (sin re-correr el modelo)."""
    from detection.coordinates import CanvasMeta
    from extraction.spec_extractor import SpecExtractor

    with open(args.detecciones) as f:
        dets = json.load(f)
    with open(args.meta) as f:
        meta = json.load(f)

    canvas    = CanvasMeta.from_dict(meta)
    extractor = SpecExtractor()
    result    = extractor.extract(args.dxf, dets, canvas)

    out_path = args.out or args.detecciones.replace(".json", "_con_specs.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"Guardado → {out_path}")


# ── Parsers ───────────────────────────────────────────────────────────────────

def _add_common_render_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--dxf",          required=True, help="Archivo DXF")
    p.add_argument("--target-px",    type=int,   default=64,      metavar="N")
    p.add_argument("--max-dim-px",   type=int,   default=16_000,  metavar="N")
    p.add_argument("--bw",           default="color",
                   choices=["color", "grayscale", "binary", "mono"])
    p.add_argument("--capas",        nargs="*",  default=None)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog        = "claudio",
        description = "Pipeline de detección de componentes eléctricos en DXF.",
    )
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Logging en nivel DEBUG")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # ── run ───────────────────────────────────────────────────────────────────
    p_run = sub.add_parser("run", help="Detectar componentes en un DXF")
    _add_common_render_args(p_run)
    p_run.add_argument("--modelo",      default=None,    help="Un modelo .pt")
    p_run.add_argument("--modelos",     nargs="+",  default=None, help="Ensemble de modelos .pt")
    p_run.add_argument("--out",         default="./pipeline_out")
    p_run.add_argument("--conf",        type=float, default=0.25)
    p_run.add_argument("--conf-min",    type=float, default=0.50)
    p_run.add_argument("--iou",         type=float, default=0.50)
    p_run.add_argument("--slice",       type=int,   default=640)
    p_run.add_argument("--overlap",     type=float, default=0.20)
    p_run.add_argument("--zoom",        type=float, default=1.0)
    p_run.add_argument("--min-tiles",   type=int,   default=None)
    p_run.add_argument("--batch-size",  type=int,   default=16)
    p_run.add_argument("--save-slices", action="store_true")
    p_run.add_argument("--device",      default="cpu",
                       help="Dispositivo de inferencia (default: cpu)")
    p_run.add_argument("--cuda",        action="store_true",
                       help="Shorthand para --device cuda")
    p_run.add_argument("--model-type",  default="yolov8")
    p_run.set_defaults(func=cmd_run)

    # ── render ────────────────────────────────────────────────────────────────
    p_render = sub.add_parser("render", help="Solo renderizar un DXF a PNG")
    _add_common_render_args(p_render)
    p_render.add_argument("--out",          default=None,  help="Ruta de salida .png")
    p_render.add_argument("--texto-negro",  action="store_true",
                          help="Texto negro en vez de invisible")
    p_render.add_argument("--print-meta",   action="store_true",
                          help="Imprimir metadata JSON al stdout")
    p_render.set_defaults(func=cmd_render)

    # ── specs ─────────────────────────────────────────────────────────────────
    p_specs = sub.add_parser("specs", help="Extraer specs de texto para detecciones existentes")
    p_specs.add_argument("--dxf",          required=True)
    p_specs.add_argument("--detecciones",  required=True, help="JSON de detecciones")
    p_specs.add_argument("--meta",         required=True, help="JSON de metadata del render")
    p_specs.add_argument("--out",          default=None,  help="JSON de salida (opcional)")
    p_specs.set_defaults(func=cmd_specs)

    return parser


def main() -> None:
    parser = build_parser()
    args   = parser.parse_args()
    _setup_logging(args.verbose)

    # Normalizar guiones a guión_bajo en los atributos de argparse
    # (argparse convierte --conf-min a conf_min automáticamente)
    args.func(args)


if __name__ == "__main__":
    main()
