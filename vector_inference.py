import os
import time
import json
import shutil
from collections import Counter

import numpy as np
import ezdxf
from ezdxf.addons.drawing import RenderContext, Frontend
from ezdxf.addons.drawing.matplotlib import MatplotlibBackend
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ultralytics import YOLO

from scale_analyzer import calcular_factor_escala
from dxf_to_image import aplicar_filtro_capas, forzar_color_negro


def eliminar_anidadas_cad(detecciones, ios_thresh=0.7, agnostico_clase=False):
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
    for grupo in grupos:
        grupo_ord = sorted(grupo, key=lambda i: -detecciones[i]["conf"])
        for idx_a, i in enumerate(grupo_ord):
            if i in eliminar:
                continue
            bi = detecciones[i]["bbox_cad"]
            ai = max((bi[2] - bi[0]) * (bi[3] - bi[1]), 1e-9)
            for j in grupo_ord[idx_a + 1:]:
                if j in eliminar:
                    continue
                bj = detecciones[j]["bbox_cad"]
                aj = max((bj[2] - bj[0]) * (bj[3] - bj[1]), 1e-9)
                xx1 = max(bi[0], bj[0]); yy1 = max(bi[1], bj[1])
                xx2 = min(bi[2], bj[2]); yy2 = min(bi[3], bj[3])
                inter = max(0, xx2 - xx1) * max(0, yy2 - yy1)
                if inter <= 0:
                    continue
                ios = inter / min(ai, aj)
                if ios >= ios_thresh:
                    eliminar.add(j)

    return [d for i, d in enumerate(detecciones) if i not in eliminar]


def nms_agnostico_clase_cad(detecciones, iou_thresh=0.5):
    if not detecciones:
        return []
    boxes = np.array([d["bbox_cad"] for d in detecciones])
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


def ejecutar_vectorial(dxf_path, modelo_path, output_dir="./pipeline_out",
                       capas_incluir=None, target_px=64, modo_color="color",
                       conf=0.25, conf_min=0.5, iou_global=0.5,
                       slice_size=640, overlap=0.2, save_slices=False, batch_size=16,
                       device="cpu"):
    
    os.makedirs(output_dir, exist_ok=True)
    temp_dir = os.path.join(output_dir, "temp_tiles")
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)

    # 1. Calculo de escala
    px_per_cad, ref = calcular_factor_escala(dxf_path, target_px=target_px)
    print(f"[vector-scale] {px_per_cad:.4f} px/CAD (ref={ref[1]:.4f} CAD -> {target_px}px)")

    tile_size_cad = slice_size / px_per_cad
    overlap_cad = tile_size_cad * overlap
    paso_cad = tile_size_cad - overlap_cad
    
    print(f"[vector-scale] Tile Size CAD: {tile_size_cad:.2f}, Overlap CAD: {overlap_cad:.2f}, Paso: {paso_cad:.2f}")

    # 2. Leer DXF
    print(f"[vector-read] Leyendo DXF {dxf_path}...")
    doc = ezdxf.readfile(dxf_path)
    msp = doc.modelspace()

    if capas_incluir:
        aplicar_filtro_capas(doc, capas_incluir)
    if modo_color == "mono":
        forzar_color_negro(doc)

    bbox = ezdxf.bbox.extents(msp)
    if not bbox.has_data:
        raise RuntimeError("ModelSpace vacio o sin bbox calculable.")

    x_min, y_min = bbox.extmin.x, bbox.extmin.y
    x_max, y_max = bbox.extmax.x, bbox.extmax.y
    print(f"[vector-bounds] X({x_min:.2f} a {x_max:.2f}), Y({y_min:.2f} a {y_max:.2f})")

    ctx = RenderContext(doc)
    
    # 3. Preparar Modelo YOLO
    print(f"[vector-model] Cargando YOLO {modelo_path} en {device}...")
    model = YOLO(modelo_path)
    model.to(device)

    # Variables de Inferencia
    detecciones = []
    batch_images = []
    batch_offsets = []
    contador_tiles = 0
    t_start = time.time()

    def procesar_batch():
        nonlocal detecciones, batch_images, batch_offsets
        if not batch_images:
            return
        
        results = model(batch_images, verbose=False, conf=conf)
        
        for i, result in enumerate(results):
            x_cad_start, y_cad_start = batch_offsets[i]
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue
                
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            clses = boxes.cls.cpu().numpy().astype(int)
            names = result.names
            
            for j in range(len(boxes)):
                # Coordenadas en px dentro del tile de 640x640
                px_x1, px_y1, px_x2, px_y2 = xyxy[j]
                
                # Mapeo a CAD. Y en imagen baja, Y en CAD sube.
                cad_x1 = x_cad_start + (px_x1 / slice_size) * tile_size_cad
                cad_x2 = x_cad_start + (px_x2 / slice_size) * tile_size_cad
                
                cad_y1 = (y_cad_start + tile_size_cad) - (px_y1 / slice_size) * tile_size_cad
                cad_y2 = (y_cad_start + tile_size_cad) - (px_y2 / slice_size) * tile_size_cad
                
                # Normalizar min/max por si la inversion de Y los cruza
                final_y1 = min(cad_y1, cad_y2)
                final_y2 = max(cad_y1, cad_y2)
                
                cx_cad = (cad_x1 + cad_x2) / 2.0
                cy_cad = (final_y1 + final_y2) / 2.0
                
                detecciones.append({
                    "clase": names[int(clses[j])],
                    "conf": float(confs[j]),
                    "bbox_cad": [cad_x1, final_y1, cad_x2, final_y2],
                    "centro_cad": [cx_cad, cy_cad],
                    "x_cad": cx_cad,
                    "y_cad": cy_cad,
                })
        
        batch_images.clear()
        batch_offsets.clear()

    # 4. Generacion e Inferencia
    x_actual = x_min
    while x_actual < x_max:
        y_actual = y_min
        while y_actual < y_max:
            x_fin = x_actual + tile_size_cad
            y_fin = y_actual + tile_size_cad
            
            # Render tile
            fig = plt.figure(figsize=(slice_size/100.0, slice_size/100.0), dpi=100)
            fig.patch.set_facecolor('white')
            ax = fig.add_axes([0, 0, 1, 1])
            ax.set_xlim(x_actual, x_fin)
            ax.set_ylim(y_actual, y_fin)
            ax.axis('off')
            
            out = MatplotlibBackend(ax)
            Frontend(ctx, out).draw_layout(msp, finalize=False)
            
            tile_name = f"tile_X{x_actual:.2f}_Y{y_actual:.2f}.png"
            tile_path = os.path.join(temp_dir, tile_name)
            
            plt.savefig(tile_path, dpi=100, facecolor=fig.get_facecolor(), edgecolor='none')
            plt.close(fig)
            
            # Si se desea aplicar modo_color como grayscale o binary, se haria aqui
            # (omitido por simpleza y velocidad, YOLO maneja bien el color)
            
            batch_images.append(tile_path)
            batch_offsets.append((x_actual, y_actual))
            contador_tiles += 1
            
            if len(batch_images) >= batch_size:
                procesar_batch()
                
            y_actual += paso_cad
        x_actual += paso_cad
        
    # Procesar remanente
    if batch_images:
        procesar_batch()

    print(f"[vector-batch] Inferencia completa: {len(detecciones)} detecciones crudas en {time.time()-t_start:.1f}s")
    print(f"[vector-batch] Se generaron y procesaron {contador_tiles} tiles.")

    # 5. Post-Procesamiento (NMS Global)
    detecciones = nms_agnostico_clase_cad(detecciones, iou_thresh=iou_global)
    print(f"[nms ] detecciones tras NMS agnostico: {len(detecciones)}")

    detecciones = eliminar_anidadas_cad(detecciones, ios_thresh=0.7)
    print(f"[anidadas] detecciones tras filtro de cajas contenidas: {len(detecciones)}")

    detecciones = [d for d in detecciones if d["conf"] >= conf_min]
    print(f"[conf] detecciones tras conf_min={conf_min}: {len(detecciones)}")

    # Guardar resultados
    json_path = os.path.join(output_dir, "detecciones.json")
    with open(json_path, "w") as f:
        # Convert types to python native types
        def convert_numpy(obj):
            if isinstance(obj, np.generic): return obj.item()
            raise TypeError
        json.dump(detecciones, f, indent=2, default=convert_numpy)

    # Limpieza
    if save_slices:
        slices_dir = os.path.join(output_dir, "slices")
        if os.path.exists(slices_dir):
            shutil.rmtree(slices_dir)
        os.rename(temp_dir, slices_dir)
        print(f"[limpieza] Tiles guardados en {slices_dir}")
    else:
        shutil.rmtree(temp_dir)
        print(f"[limpieza] Archivos temporales eliminados.")

    # Detalle final
    conteo = Counter(d["clase"] for d in detecciones)
    print("-" * 60)
    print("CONTEO DE COMPONENTES:")
    print("-" * 60)
    for clase, n in conteo.most_common():
        confs = [d["conf"] for d in detecciones if d["clase"] == clase]
        media = sum(confs) / len(confs)
        print(f"  {clase.upper():<28} {n:>3}   conf media={media:.3f}  min={min(confs):.3f}  max={max(confs):.3f}")

    return conteo, detecciones
