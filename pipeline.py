"""
Pipeline end-to-end: DXF -> imagen renderizada -> inferencia con SAHI -> conteo.

Modo single model (multi-clase):
    python pipeline.py --dxf plano.dxf --modelo best.pt

Modo ensemble (varios modelos single-class encadenados):
    python pipeline.py --dxf plano.dxf \\
        --modelos runs/single/termomagnetico/weights/best.pt \\
                  runs/single/diferencial/weights/best.pt \\
                  runs/single/seleccionador_manual/weights/best.pt

Con filtrado de capas (recomendado para subir precision):
    python pipeline.py --dxf plano.dxf --modelo best.pt \\
        --capas ELECTRICA TOMAS ILUMINACION ANOTACIONES
"""

import os
import json
import argparse
from collections import Counter

from dxf_to_image import renderizar_dxf
from inference_sahi import (
    ejecutar,
    cargar_modelo,
    correr_sahi_batched,
    descartar_pegadas_al_borde,
    nms_agnostico_clase,
    eliminar_anidadas,
    filtrar_por_confianza,
    calcular_slice_size,
    guardar_slices,
    dibujar,
)


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
    """Modo original: un solo modelo (multi-clase o single-class)."""
    os.makedirs(output_dir, exist_ok=True)

    img_path = os.path.join(output_dir, "plano_render.png")
    renderizar_dxf(
        dxf_path, img_path,
        capas_incluir=capas_incluir,
        target_px=target_px,
        modo_color=modo_color,
        max_dim_px=max_dim_px,
    )
    meta_path = os.path.splitext(img_path)[0] + ".json"

    return ejecutar(
        modelo_path, img_path, meta_path,
        output_dir=output_dir,
        conf=conf, conf_min=conf_min, iou_global=iou_global,
        slice_size=slice_size, overlap=overlap,
        zoom=zoom, min_tiles_por_eje=min_tiles_por_eje,
        save_slices=save_slices, batch_size=batch_size,
        device=device, model_type=model_type,
    )


def correr_pipeline_ensemble(dxf_path, modelos_paths,
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
    """
    Modo ensemble: renderiza el DXF UNA SOLA VEZ y luego encadena
    N modelos single-class sobre la misma imagen. Las detecciones de
    todos los modelos se juntan y se les aplica NMS global al final.
    """
    os.makedirs(output_dir, exist_ok=True)

    # --- Render (una sola vez) ---
    img_path = os.path.join(output_dir, "plano_render.png")
    renderizar_dxf(
        dxf_path, img_path,
        capas_incluir=capas_incluir,
        target_px=target_px,
        modo_color=modo_color,
        max_dim_px=max_dim_px,
    )
    meta_path = os.path.splitext(img_path)[0] + ".json"

    with open(meta_path) as f:
        meta = json.load(f)

    # --- Slice size efectivo ---
    eff_slice = slice_size
    if zoom != 1.0 or min_tiles_por_eje is not None:
        eff_slice = calcular_slice_size(
            img_path, base_slice=slice_size, zoom=zoom,
            min_tiles_por_eje=min_tiles_por_eje,
        )

    if save_slices:
        guardar_slices(img_path, os.path.join(output_dir, "slices"),
                       slice_size=eff_slice, overlap=overlap)

    # --- Inferencia con cada modelo ---
    todas_detecciones = []
    for i, modelo_path in enumerate(modelos_paths, 1):
        print(f"\n{'='*60}")
        print(f"[ensemble] Modelo {i}/{len(modelos_paths)}: {modelo_path}")
        print(f"{'='*60}")
        modelo = cargar_modelo(modelo_path, conf=conf, device=device,
                               model_type=model_type)
        dets = correr_sahi_batched(
            modelo, img_path, meta,
            slice_size=eff_slice, overlap=overlap,
            batch_size=batch_size,
        )
        print(f"[ensemble] modelo {i} -> {len(dets)} detecciones crudas")
        todas_detecciones.extend(dets)

    print(f"\n[ensemble] total detecciones crudas de todos los modelos: "
          f"{len(todas_detecciones)}")

    # --- Filtros globales ---
    todas_detecciones = descartar_pegadas_al_borde(todas_detecciones, meta)
    print(f"[borde   ] tras filtro de borde     : {len(todas_detecciones)}")

    todas_detecciones = nms_agnostico_clase(todas_detecciones,
                                             iou_thresh=iou_global)
    print(f"[nms     ] tras NMS agnostico        : {len(todas_detecciones)}")

    todas_detecciones = eliminar_anidadas(todas_detecciones, ios_thresh=0.7)
    print(f"[anidadas] tras filtro de contenidas : {len(todas_detecciones)}")

    todas_detecciones = filtrar_por_confianza(todas_detecciones, conf_min)
    print(f"[conf    ] tras conf_min={conf_min}          : "
          f"{len(todas_detecciones)}")

    # --- Guardar resultados ---
    json_path = os.path.join(output_dir, "detecciones.json")
    with open(json_path, "w") as f:
        json.dump(todas_detecciones, f, indent=2)

    vis_path = os.path.join(output_dir, "deteccion_visual.png")
    dibujar(img_path, todas_detecciones, vis_path)

    # --- Resumen ---
    print("-" * 60)
    print("DETECCIONES INDIVIDUALES:")
    print("-" * 60)
    for i, d in enumerate(sorted(todas_detecciones,
                                  key=lambda x: (x["clase"], -x["conf"])), 1):
        cx, cy = d["centro_px"]
        print(f"  {i:3d}. {d['clase']:<28} conf={d['conf']:.3f}  "
              f"px=({int(cx):4d},{int(cy):4d})  "
              f"cad=({d['x_cad']:.2f},{d['y_cad']:.2f})")

    conteo = Counter(d["clase"] for d in todas_detecciones)
    print("-" * 60)
    print("CONTEO DE COMPONENTES:")
    print("-" * 60)
    for clase, n in conteo.most_common():
        confs = [d["conf"] for d in todas_detecciones if d["clase"] == clase]
        media = sum(confs) / len(confs)
        print(f"  {clase.upper():<28} {n:>3}   "
              f"conf media={media:.3f}  "
              f"min={min(confs):.3f}  max={max(confs):.3f}")

    print(f"\n[ensemble] imagen  -> {vis_path}")
    print(f"[ensemble] json    -> {json_path}")
    return conteo, todas_detecciones


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pipeline DXF -> PNG -> YOLO. "
                    "Usa --modelo para un modelo solo, "
                    "o --modelos para encadenar varios single-class."
    )
    parser.add_argument("--dxf", required=True)
    parser.add_argument("--modelo", default=None,
                        help="Ruta a un unico modelo .pt (multi-clase o single-class)")
    parser.add_argument("--modelos", nargs="+", default=None,
                        help="Rutas a varios modelos .pt single-class para ensemble. "
                             "Ej: --modelos m1.pt m2.pt m3.pt")
    parser.add_argument("--out", default="./pipeline_out")
    parser.add_argument("--capas", nargs="*", default=None,
                        help="Solo renderiza estas capas. Vacio = todas.")
    parser.add_argument("--target-px", type=int, default=64,
                        help="Tamano objetivo en px de un simbolo tipico")
    parser.add_argument("--bw", default="color",
                        choices=["color", "grayscale", "mono", "binary"])
    parser.add_argument("--max-dim-px", type=int, default=16000)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--conf-min", type=float, default=0.5)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--slice", type=int, default=640)
    parser.add_argument("--zoom", type=float, default=1.0)
    parser.add_argument("--min-tiles", type=int, default=None)
    parser.add_argument("--overlap", type=float, default=0.2)
    parser.add_argument("--save-slices", action="store_true")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cpu", help="cpu o cuda:0")
    parser.add_argument("--model-type", default="yolov8")
    args = parser.parse_args()

    # Validacion: necesita --modelo o --modelos, no ambos ni ninguno
    if args.modelo and args.modelos:
        parser.error("Usa --modelo O --modelos, no ambos.")
    if not args.modelo and not args.modelos:
        parser.error("Debes indicar --modelo (un modelo) o --modelos (varios).")

    kwargs = dict(
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

    if args.modelo:
        correr_pipeline(args.dxf, args.modelo, **kwargs)
    else:
        correr_pipeline_ensemble(args.dxf, args.modelos, **kwargs)
