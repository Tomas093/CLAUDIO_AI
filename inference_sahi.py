"""
Inferencia con SAHI sobre la imagen renderizada por dxf_to_image.py.

Pasos:
  1. Cargar imagen + metadata JSON.
  2. Correr get_sliced_prediction (SAHI hace tiling 640x640 con overlap).
  3. SAHI ya aplica NMS por clase con metrica IOS (mejor que IoU para simbolos cortados).
  4. Aplicamos un NMS adicional AGNOSTICO de clase para resolver
     confusiones tipo "interruptor vs tomacorriente" en el mismo lugar.
  5. Mapeamos pixeles -> coordenadas CAD usando la metadata.
  6. Guardamos JSON con detecciones y PNG con visualizacion.

Requiere: pip install sahi ultralytics opencv-python
"""

import os
import json
import time
import argparse
from collections import Counter

import numpy as np
import cv2

from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from sahi.slicing import slice_image


def cargar_modelo(ruta_modelo, conf=0.25, device="cpu", model_type="yolov8"):
    return AutoDetectionModel.from_pretrained(
        model_type=model_type,
        model_path=ruta_modelo,
        confidence_threshold=conf,
        device=device,
    )


def calcular_slice_size(image_path, base_slice=640, zoom=1.0,
                        min_tiles_por_eje=None,
                        slice_min=160, slice_max=1280):
    """
    Calcula el slice_size efectivo segun el tamano de la imagen.

    base_slice: tamano nominal por defecto (640 para YOLOv8).
    zoom: multiplicador. zoom=2 -> slice mas chico (mas zoom por tile).
    min_tiles_por_eje: si se pasa, anula zoom y elige el slice que da al
                       menos esa cantidad de tiles por eje (sirve para
                       imagenes muy chicas o muy grandes, automatico).
    slice_min, slice_max: limites de seguridad.
    """
    img = cv2.imread(image_path)
    if img is None:
        return base_slice
    H, W = img.shape[:2]

    if min_tiles_por_eje is not None and min_tiles_por_eje > 0:
        slice_size = min(W, H) // min_tiles_por_eje
        razon = f"min_tiles={min_tiles_por_eje}"
    else:
        slice_size = int(round(base_slice / zoom))
        razon = f"zoom={zoom}"

    slice_size = max(slice_min, min(slice_size, slice_max))
    print(f"[slice] imagen {W}x{H}  ->  slice_size={slice_size}  ({razon})")
    return slice_size


def guardar_slices(image_path, output_dir, slice_size=640, overlap=0.2):
    """
    Genera los mismos slices que SAHI usa internamente y los guarda como PNG
    numerados, mas un grid index PNG para entender como se mapean al original.
    """
    os.makedirs(output_dir, exist_ok=True)
    result = slice_image(
        image=image_path,
        output_file_name="slice",
        output_dir=output_dir,
        slice_height=slice_size,
        slice_width=slice_size,
        overlap_height_ratio=overlap,
        overlap_width_ratio=overlap,
        min_area_ratio=0.1,
        out_ext=".png",
    )
    print(f"[slices] {len(result)} tiles guardados en {output_dir}")

    # Visualizacion: dibujar los bordes de cada slice sobre la imagen completa
    img_grid = cv2.imread(image_path)
    if img_grid is not None:
        for sl in result:
            x1, y1 = sl["starting_pixel"]
            h, w = sl["image"].shape[:2] if hasattr(sl["image"], "shape") else (slice_size, slice_size)
            x2, y2 = x1 + w, y1 + h
            cv2.rectangle(img_grid, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(img_grid, str(sl.get("index", "")),
                        (x1 + 5, y1 + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        cv2.imwrite(os.path.join(output_dir, "_grid_overview.png"), img_grid)
        print(f"[slices] grid overview -> {output_dir}/_grid_overview.png")
    return result


def eliminar_anidadas(detecciones, ios_thresh=0.7, agnostico_clase=False):
    """
    Elimina detecciones cuya caja esta significativamente contenida dentro
    de otra de mayor confianza. Usa IOS (intersection over smaller area),
    que detecta cajas anidadas aunque sus tamanos sean muy diferentes
    (donde IoU clasico fallaria).

    ios_thresh: umbral de "porcentaje de la chica que cae dentro de la grande"
                a partir del cual se suprime la chica. 0.7 = bastante agresivo,
                0.85 = mas conservador.
    agnostico_clase: si True, suprime aunque sean clases distintas. Default False.
    """
    if not detecciones:
        return []

    if agnostico_clase:
        grupos = [list(range(len(detecciones)))]
    else:
        por_clase = {}
        for i, d in enumerate(detecciones):
            por_clase.setdefault(d["clase"], []).append(i)
        grupos = list(por_clase.values())

    eliminar = set()
    descartados_log = []
    for grupo in grupos:
        # Procesamos por confianza descendente: las mas seguras "ganan"
        grupo_ord = sorted(grupo, key=lambda i: -detecciones[i]["conf"])
        for idx_a, i in enumerate(grupo_ord):
            if i in eliminar:
                continue
            bi = detecciones[i]["bbox_px"]
            ai = max((bi[2] - bi[0]) * (bi[3] - bi[1]), 1e-9)
            for j in grupo_ord[idx_a + 1:]:
                if j in eliminar:
                    continue
                bj = detecciones[j]["bbox_px"]
                aj = max((bj[2] - bj[0]) * (bj[3] - bj[1]), 1e-9)
                # interseccion
                xx1 = max(bi[0], bj[0]); yy1 = max(bi[1], bj[1])
                xx2 = min(bi[2], bj[2]); yy2 = min(bi[3], bj[3])
                inter = max(0, xx2 - xx1) * max(0, yy2 - yy1)
                if inter <= 0:
                    continue
                ios = inter / min(ai, aj)
                if ios >= ios_thresh:
                    eliminar.add(j)
                    descartados_log.append(
                        (detecciones[j], detecciones[i], ios)
                    )

    if descartados_log:
        print(f"[anidadas] {len(descartados_log)} detecciones suprimidas por estar contenidas:")
        for fuera, dentro, ios in descartados_log[:10]:
            print(f"   - {fuera['clase']:<25} conf={fuera['conf']:.3f}  "
                  f"contenida en otra de conf={dentro['conf']:.3f}  "
                  f"(IOS={ios:.2f})")

    return [d for i, d in enumerate(detecciones) if i not in eliminar]


def filtrar_por_confianza(detecciones, conf_min):
    """Filtro post-proceso: descarta detecciones con conf < conf_min."""
    if conf_min <= 0:
        return detecciones
    n_antes = len(detecciones)
    filtradas = [d for d in detecciones if d["conf"] >= conf_min]
    n_descartadas = n_antes - len(filtradas)
    if n_descartadas > 0:
        print(f"[conf-min] {n_descartadas} descartadas por conf < {conf_min}")
    return filtradas


def correr_sahi_batched(modelo_sahi, image_path, meta,
                         slice_size=640, overlap=0.2,
                         batch_size=16, verbose=True):
    """
    Slicing manual + batch inference + mapeo a CAD en una sola pasada.

    A diferencia de get_sliced_prediction (que procesa los tiles 1x1),
    aca agrupamos los tiles en lotes y los pasamos juntos al modelo YOLO,
    aprovechando la vectorizacion del framework. Speedup tipico 3x-8x.

    Tambien reportamos progreso en tiempo real (batch por batch) con ETA.

    Devuelve lista de dicts con: clase, conf, bbox_px, centro_px, x_cad, y_cad
    (mismo formato que mapear_a_cad).
    """
    slice_result = slice_image(
        image=image_path,
        slice_height=slice_size,
        slice_width=slice_size,
        overlap_height_ratio=overlap,
        overlap_width_ratio=overlap,
        min_area_ratio=0.1,
    )

    n_total = len(slice_result)
    n_batches = (n_total + batch_size - 1) // batch_size

    yolo_model = modelo_sahi.model
    conf_threshold = modelo_sahi.confidence_threshold

    px_per_cad = meta["px_per_cad"]
    x_min_cad = meta["x_min_cad"]
    y_max_cad = meta["y_max_cad"]

    print(f"[batch] {n_total} tiles totales | batch_size={batch_size} | "
          f"{n_batches} batches")

    detecciones = []
    t_start = time.time()

    for batch_idx in range(n_batches):
        t_batch_start = time.time()
        start = batch_idx * batch_size
        end = min(start + batch_size, n_total)

        batch_images = []
        batch_offsets = []
        for i in range(start, end):
            sl = slice_result[i]
            batch_images.append(sl["image"])
            batch_offsets.append(sl["starting_pixel"])

        # Batch inference (ultralytics acepta listas nativamente)
        results = yolo_model(batch_images, verbose=False, conf=conf_threshold)

        n_batch_dets = 0
        for i, result in enumerate(results):
            x_off, y_off = batch_offsets[i]
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue
            xyxy = boxes.xyxy.cpu().numpy()
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

                x_cad = x_min_cad + cx_px / px_per_cad
                y_cad = y_max_cad - cy_px / px_per_cad

                detecciones.append({
                    "clase": names[int(clses[j])],
                    "conf": float(confs[j]),
                    "bbox_px": [gx1, gy1, gx2, gy2],
                    "centro_px": [cx_px, cy_px],
                    "x_cad": x_cad,
                    "y_cad": y_cad,
                })
                n_batch_dets += 1

        t_batch = time.time() - t_batch_start
        t_total = time.time() - t_start
        pct = (end / n_total) * 100
        eta = (t_total / end) * (n_total - end) if end < n_total else 0

        if verbose:
            print(f"[batch] {batch_idx+1:3d}/{n_batches}  "
                  f"tiles {start+1:4d}-{end:<4d}/{n_total}  "
                  f"({pct:5.1f}%)  "
                  f"+{n_batch_dets:2d} det (acum: {len(detecciones)})  "
                  f"t={t_batch:.2f}s  ETA={eta:.0f}s")

    print(f"[batch] inferencia completa: {len(detecciones)} detecciones "
          f"crudas en {time.time()-t_start:.1f}s")
    return detecciones


def correr_sahi(modelo, image_path, slice_size=640, overlap=0.2,
                postprocess_threshold=0.5):
    """
    Tiling + inferencia + NMS por clase.
    IMPORTANTE: usamos NMS (suprime duplicados) en lugar de GREEDYNMM (fusiona).
    GREEDYNMM con metrica IOS produce cajas alargadas cuando hay detecciones
    espurias en tiles adyacentes a lo largo de un conductor vertical.
    """
    return get_sliced_prediction(
        image_path,
        modelo,
        slice_height=slice_size,
        slice_width=slice_size,
        overlap_height_ratio=overlap,
        overlap_width_ratio=overlap,
        postprocess_type="NMS",          # antes: GREEDYNMM
        postprocess_match_metric="IOU",  # antes: IOS
        postprocess_match_threshold=postprocess_threshold,
        postprocess_class_agnostic=False,
        verbose=1,
    )


def descartar_pegadas_al_borde(detecciones, meta, margen_px=2, conf_safe=0.9):
    """
    Descarta detecciones cuyo bbox toca el borde de la imagen.
    Una caja pegada al borde casi siempre es un artefacto del clipping
    que SAHI hace cuando el modelo predice fuera de la imagen.
    Se perdonan las de confianza muy alta (conf >= conf_safe).
    """
    if not detecciones:
        return []
    W = meta["image_width_px"]
    H = meta["image_height_px"]
    sobrevivientes = []
    descartados = []
    for d in detecciones:
        x1, y1, x2, y2 = d["bbox_px"]
        toca_borde = (
            x1 <= margen_px or y1 <= margen_px
            or x2 >= W - margen_px or y2 >= H - margen_px
        )
        if toca_borde and d["conf"] < conf_safe:
            descartados.append(d)
        else:
            sobrevivientes.append(d)
    if descartados:
        print(f"[borde] descartadas {len(descartados)} detecciones pegadas al borde:")
        for d in descartados:
            x1, y1, x2, y2 = d["bbox_px"]
            print(f"   - {d['clase']:<25} bbox=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}) "
                  f"conf={d['conf']:.2f}")
    return sobrevivientes


def nms_agnostico_clase(detecciones, iou_thresh=0.5):
    """
    NMS que ignora la clase. Si dos detecciones cualesquiera se solapan
    > iou_thresh, nos quedamos con la de mayor confianza. Resuelve confusiones
    interruptor/tomacorriente y otras parecidas.
    """
    if not detecciones:
        return []
    boxes = np.array([d["bbox_px"] for d in detecciones])
    confs = np.array([d["conf"] for d in detecciones])
    idxs = confs.argsort()[::-1]
    keep = []
    while len(idxs) > 0:
        i = idxs[0]
        keep.append(int(i))
        if len(idxs) == 1:
            break
        rest = idxs[1:]
        xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        w = np.clip(xx2 - xx1, 0, None)
        h = np.clip(yy2 - yy1, 0, None)
        inter = w * h
        ai = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        ar = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
        iou = inter / (ai + ar - inter + 1e-9)
        idxs = rest[iou < iou_thresh]
    return [detecciones[i] for i in keep]


def mapear_a_cad(resultado_sahi, meta):
    """Convierte cada object_prediction de SAHI a un dict con coords px y CAD."""
    px_per_cad = meta["px_per_cad"]
    x_min_cad = meta["x_min_cad"]
    y_max_cad = meta["y_max_cad"]  # Y en CAD aumenta hacia arriba

    out = []
    for pred in resultado_sahi.object_prediction_list:
        bb = pred.bbox
        x1, y1, x2, y2 = bb.minx, bb.miny, bb.maxx, bb.maxy
        cx_px = (x1 + x2) / 2.0
        cy_px = (y1 + y2) / 2.0

        x_cad = x_min_cad + cx_px / px_per_cad
        # Y de imagen aumenta hacia abajo, Y de CAD hacia arriba => invertimos
        y_cad = y_max_cad - cy_px / px_per_cad

        out.append({
            "clase": pred.category.name,
            "conf": float(pred.score.value),
            "bbox_px": [float(x1), float(y1), float(x2), float(y2)],
            "centro_px": [cx_px, cy_px],
            "x_cad": x_cad,
            "y_cad": y_cad,
        })
    return out


def dibujar(image_path, detecciones, output_path):
    img = cv2.imread(image_path)
    if img is None:
        return
    for d in detecciones:
        x1, y1, x2, y2 = (int(v) for v in d["bbox_px"])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 0), 2)
        label = f"{d['clase']} {d['conf']:.2f}"
        cv2.putText(img, label, (x1, max(y1 - 5, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 0), 1)
    cv2.imwrite(output_path, img)


def ejecutar(modelo_path, image_path, meta_path,
             output_dir="./resultados",
             conf=0.25, conf_min=0.5, iou_global=0.5,
             slice_size=640, overlap=0.2,
             zoom=1.0, min_tiles_por_eje=None,
             save_slices=False, batch_size=16,
             device="cpu", model_type="yolov8"):
    os.makedirs(output_dir, exist_ok=True)
    with open(meta_path) as f:
        meta = json.load(f)

    # Si zoom != 1 o min_tiles_por_eje fue seteado, recalculamos slice
    if zoom != 1.0 or min_tiles_por_eje is not None:
        slice_size = calcular_slice_size(
            image_path, base_slice=slice_size, zoom=zoom,
            min_tiles_por_eje=min_tiles_por_eje,
        )

    if save_slices:
        slices_dir = os.path.join(output_dir, "slices")
        guardar_slices(image_path, slices_dir,
                       slice_size=slice_size, overlap=overlap)

    modelo = cargar_modelo(modelo_path, conf=conf, device=device, model_type=model_type)

    # Inferencia con batching + progreso en tiempo real
    detecciones = correr_sahi_batched(
        modelo, image_path, meta,
        slice_size=slice_size, overlap=overlap,
        batch_size=batch_size,
    )
    print(f"[batch] detecciones crudas (pre-filtros): {len(detecciones)}")

    detecciones = descartar_pegadas_al_borde(detecciones, meta)
    print(f"[borde] detecciones tras filtro de borde: {len(detecciones)}")

    detecciones = nms_agnostico_clase(detecciones, iou_thresh=iou_global)
    print(f"[nms ] detecciones tras NMS agnostico: {len(detecciones)}")

    detecciones = eliminar_anidadas(detecciones, ios_thresh=0.7)
    print(f"[anidadas] detecciones tras filtro de cajas contenidas: {len(detecciones)}")

    detecciones = filtrar_por_confianza(detecciones, conf_min)
    print(f"[conf] detecciones tras conf_min={conf_min}: {len(detecciones)}")

    json_path = os.path.join(output_dir, "detecciones.json")
    with open(json_path, "w") as f:
        json.dump(detecciones, f, indent=2)

    vis_path = os.path.join(output_dir, "deteccion_visual.png")
    dibujar(image_path, detecciones, vis_path)

    # Detalle por deteccion (ordenado por clase y luego por confianza desc)
    print("-" * 60)
    print("DETECCIONES INDIVIDUALES:")
    print("-" * 60)
    detecciones_ordenadas = sorted(
        detecciones, key=lambda d: (d["clase"], -d["conf"])
    )
    for i, d in enumerate(detecciones_ordenadas, 1):
        cx, cy = d["centro_px"]
        print(f"  {i:3d}. {d['clase']:<28} conf={d['conf']:.3f}  "
              f"px=({int(cx):4d},{int(cy):4d})  "
              f"cad=({d['x_cad']:.2f},{d['y_cad']:.2f})")

    conteo = Counter(d["clase"] for d in detecciones)
    print("-" * 60)
    print("CONTEO DE COMPONENTES:")
    print("-" * 60)
    for clase, n in conteo.most_common():
        # Confianza media y rango por clase
        confs = [d["conf"] for d in detecciones if d["clase"] == clase]
        media = sum(confs) / len(confs)
        print(f"  {clase.upper():<28} {n:>3}   "
              f"conf media={media:.3f}  "
              f"min={min(confs):.3f}  max={max(confs):.3f}")

    return conteo, detecciones


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--modelo", default="best.pt")
    parser.add_argument("--imagen", default="plano_render.png")
    parser.add_argument("--meta", default="plano_render.json")
    parser.add_argument("--out", default="./resultados")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Umbral de confianza interno de SAHI (bajo "
                             "permite mejor merging; usar --conf-min para "
                             "filtro final)")
    parser.add_argument("--conf-min", type=float, default=0.5,
                        help="Confianza minima del filtro post-proceso")
    parser.add_argument("--iou", type=float, default=0.5,
                        help="IoU para el NMS agnostico de clase")
    parser.add_argument("--slice", type=int, default=640)
    parser.add_argument("--zoom", type=float, default=1.0,
                        help="Multiplicador. zoom=2 -> slice a la mitad")
    parser.add_argument("--min-tiles", type=int, default=None,
                        help="Tiles minimos por eje (anula --zoom)")
    parser.add_argument("--overlap", type=float, default=0.2)
    parser.add_argument("--save-slices", action="store_true",
                        help="Guarda los tiles que SAHI usa, en out/slices/")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Tiles procesados por batch (subi a 32 con GPU "
                             "buena, baja a 4 en CPU si te quedas sin RAM)")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--model-type", default="yolov8")
    args = parser.parse_args()
    ejecutar(args.modelo, args.imagen, args.meta, args.out,
             args.conf, args.conf_min, args.iou,
             args.slice, args.overlap,
             args.zoom, args.min_tiles, args.save_slices,
             args.batch_size,
             args.device, args.model_type)
