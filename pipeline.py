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

from dxf_to_image import renderizar_dxf
from inference_sahi import ejecutar


def correr_pipeline(dxf_path, modelo_path,
                    output_dir="./pipeline_out",
                    capas_incluir=None,
                    target_px=64,
                    modo_color="color",
                    conf=0.25, conf_min=0.5, iou_global=0.5,
                    slice_size=640, overlap=0.2,
                    save_slices=False,
                    device="cpu", model_type="yolov8"):
    os.makedirs(output_dir, exist_ok=True)

    img_path = os.path.join(output_dir, "plano_render.png")
    meta = renderizar_dxf(
        dxf_path, img_path,
        capas_incluir=capas_incluir,
        target_px=target_px,
        modo_color=modo_color,
    )
    meta_path = os.path.splitext(img_path)[0] + ".json"

    return ejecutar(
        modelo_path, img_path, meta_path,
        output_dir=output_dir,
        conf=conf, conf_min=conf_min, iou_global=iou_global,
        slice_size=slice_size, overlap=overlap,
        save_slices=save_slices,
        device=device, model_type=model_type,
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
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Umbral SAHI interno (bajo permite mejor "
                             "merging entre tiles)")
    parser.add_argument("--conf-min", type=float, default=0.5,
                        help="Confianza minima del filtro post-proceso "
                             "(este es el que vas a tocar normalmente)")
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--slice", type=int, default=640)
    parser.add_argument("--overlap", type=float, default=0.2)
    parser.add_argument("--save-slices", action="store_true",
                        help="Guarda los slices de SAHI en out/slices/ "
                             "para inspeccion visual")
    parser.add_argument("--device", default="cpu",
                        help="cpu o cuda:0")
    parser.add_argument("--model-type", default="yolov8")
    args = parser.parse_args()

    correr_pipeline(
        args.dxf, args.modelo, args.out, args.capas, args.target_px,
        args.bw,
        args.conf, args.conf_min, args.iou,
        args.slice, args.overlap, args.save_slices,
        args.device, args.model_type,
    )
