"""
Pipeline end-to-end: DXF -> imagen renderizada -> inferencia con SAHI -> conteo.

Uso minimo:
    python pipeline.py --dxf plano.dxf --modelo best.pt

Con filtrado de capas (recomendado para subir precision):
    python pipeline.py --dxf plano.dxf --modelo best.pt \
        --capas ELECTRICA TOMAS ILUMINACION ANOTACIONES
"""

import os
import argparse

from vector_inference import ejecutar_vectorial


def correr_pipeline(dxf_path, modelo_path,
                    output_dir="./pipeline_out",
                    capas_incluir=None,
                    target_px=90,
                    modo_color="color",
                    max_dim_px=16000,
                    conf=0.25, conf_min=0.5, iou_global=0.5,
                    slice_size=640, overlap=0.2,
                    zoom=1.0, min_tiles_por_eje=None,
                    save_slices=False, batch_size=16,
                    device="cpu", model_type="yolov8"):
    
    # max_dim_px, zoom, min_tiles_por_eje are ignored in the new vector pipeline
    # as scaling and tiling is handled infinitely and natively based on target_px.
    
    return ejecutar_vectorial(
        dxf_path=dxf_path,
        modelo_path=modelo_path,
        output_dir=output_dir,
        capas_incluir=capas_incluir,
        target_px=target_px,
        modo_color=modo_color,
        conf=conf,
        conf_min=conf_min,
        iou_global=iou_global,
        slice_size=slice_size,
        overlap=overlap,
        save_slices=save_slices,
        batch_size=batch_size,
        device=device
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dxf", required=True)
    parser.add_argument("--modelo", default="best.pt")
    parser.add_argument("--out", default="./pipeline_out")
    parser.add_argument("--capas", nargs="*", default=None,
                        help="Solo renderiza estas capas. Vacio = todas.")
    parser.add_argument("--target-px", type=int, default=64,
                        help="Tamano objetivo en px de un simbolo tipico")
    parser.add_argument("--bw", default="color",
                        choices=["color", "grayscale", "mono", "binary"],
                        help="Modo de color: color (default), grayscale, "
                             "mono (todas entidades en negro), binary (1-bit B&N)")
    parser.add_argument("--max-dim-px", type=int, default=16000,
                        help="Limite del lado mayor de la imagen renderizada. "
                             "Subir para planos grandes (ej. 24000); "
                             "bajar si se queda sin RAM.")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Umbral SAHI interno (bajo permite mejor "
                             "merging entre tiles)")
    parser.add_argument("--conf-min", type=float, default=0.5,
                        help="Confianza minima del filtro post-proceso "
                             "(este es el que vas a tocar normalmente)")
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--slice", type=int, default=640,
                        help="Tamano base de slice en px")
    parser.add_argument("--zoom", type=float, default=1.0,
                        help="Multiplicador de zoom de los slices. "
                             "zoom=2 -> slices a la mitad (mas tiles, "
                             "mas detalle por componente)")
    parser.add_argument("--min-tiles", type=int, default=None,
                        help="Tiles minimos por eje. Anula --zoom y "
                             "calcula el slice segun el tamano de imagen")
    parser.add_argument("--overlap", type=float, default=0.2)
    parser.add_argument("--save-slices", action="store_true",
                        help="Guarda los slices de SAHI en out/slices/ "
                             "para inspeccion visual")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Tiles procesados por batch (default 16). "
                             "Con GPU buena, subi a 32-64. En CPU con poca "
                             "RAM, baja a 4-8.")
    parser.add_argument("--device", default="cpu",
                        help="cpu o cuda:0")
    parser.add_argument("--model-type", default="yolov8")
    args = parser.parse_args()

    correr_pipeline(
        dxf_path=args.dxf,
        modelo_path=args.modelo,
        output_dir=args.out,
        capas_incluir=args.capas,
        target_px=args.target_px,
        modo_color=args.bw,
        max_dim_px=args.max_dim_px,
        conf=args.conf,
        conf_min=args.conf_min,
        iou_global=args.iou,
        slice_size=args.slice,
        overlap=args.overlap,
        zoom=args.zoom,
        min_tiles_por_eje=args.min_tiles,
        save_slices=args.save_slices,
        batch_size=args.batch_size,
        device=args.device,
        model_type=args.model_type,
    )
