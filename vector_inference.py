import os
import sys
import math
import time
import json
import shutil
from collections import Counter

import torch
import numpy as np
import ezdxf
import ezdxf.bbox
from ezdxf.addons.drawing import RenderContext, Frontend
from ezdxf.addons.drawing.matplotlib import MatplotlibBackend
from ezdxf.addons.drawing.config import Configuration, ColorPolicy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2

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


def nms_distancia_cad(detecciones, d_min=0.50, agnostico_clase=True):
    """
    NMS inteligente por distancia centroidal euclidiana en CAD (d_min ~ 0.50 CAD).
    Suprime duplicados en interruptores tripolares/tetrapolares contiguos o producidos por solapamiento de tiles.
    """
    if not detecciones:
        return []
    
    confs = np.array([d["conf"] for d in detecciones])
    centroids = np.array([d.get("centro_cad", [d.get("xc", 0.0), d.get("yc", 0.0)]) for d in detecciones])
    idxs = confs.argsort()[::-1]
    
    keep = []
    while len(idxs) > 0:
        i = idxs[0]
        keep.append(int(i))
        if len(idxs) == 1:
            break
        rest = idxs[1:]
        dists = np.hypot(centroids[rest, 0] - centroids[i, 0], centroids[rest, 1] - centroids[i, 1])
        idxs = rest[dists >= d_min]
        
    return [detecciones[i] for i in keep]


def fusionar_multipolares_cad(detecciones, d_max=0.50):
    """
    Fusiona cajas contiguas en interruptores multipolares (tripolares/tetrapolares)
    o disyuntores contiguos cuya separacion centroidal sea <= d_max CAD.
    """
    if not detecciones:
        return []
        
    used = set()
    fused = []
    sorted_dets = sorted(detecciones, key=lambda d: -d["conf"])
    
    for i, d in enumerate(sorted_dets):
        if i in used:
            continue
        used.add(i)
        
        group = [d]
        cx_i = d.get("xc", d.get("centro_cad", [0, 0])[0])
        cy_i = d.get("yc", d.get("centro_cad", [0, 0])[1])
        b_i = list(d["bbox_cad"])
        
        for j, d2 in enumerate(sorted_dets):
            if j in used:
                continue
            cx_j = d2.get("xc", d2.get("centro_cad", [0, 0])[0])
            cy_j = d2.get("yc", d2.get("centro_cad", [0, 0])[1])
            dist = math.hypot(cx_i - cx_j, cy_i - cy_j)
            
            if dist <= d_max:
                used.add(j)
                group.append(d2)
                b2 = d2["bbox_cad"]
                b_i[0] = min(b_i[0], b2[0])
                b_i[1] = min(b_i[1], b2[1])
                b_i[2] = max(b_i[2], b2[2])
                b_i[3] = max(b_i[3], b2[3])
                
        best_d = group[0].copy()
        best_d["bbox_cad"] = b_i
        best_d["xc"] = (b_i[0] + b_i[2]) / 2.0
        best_d["yc"] = (b_i[1] + b_i[3]) / 2.0
        if "centro_cad" in best_d:
            best_d["centro_cad"] = [best_d["xc"], best_d["yc"]]
        best_d["fused_count"] = len(group)
        fused.append(best_d)
        
    return fused


def snap_a_bloques_cad(detecciones, dxf_path, max_dist=1.0, allowed_blocks=None, excluded_blocks=None):
    """
    Snap geometrico vectorial a nivel CAD:
    Asocia cada deteccion con el INSERT o bloque de CAD mas cercano para entregar
    bboxes y coordenadas exactas a nivel de dibujo vectorial.
    """
    if not detecciones or not os.path.exists(dxf_path):
        return detecciones
        
    doc = ezdxf.readfile(dxf_path)
    msp = doc.modelspace()
    
    cad_blocks = []
    for e in msp.query("INSERT"):
        bname = e.dxf.name
        if excluded_blocks and any(ex.upper() in bname.upper() for ex in excluded_blocks):
            continue
        if allowed_blocks and not any(al.upper() in bname.upper() for al in allowed_blocks):
            continue
        try:
            bb = ezdxf.bbox.extents([e])
            cad_blocks.append({
                'name': bname,
                'xc': float(e.dxf.insert.x),
                'yc': float(e.dxf.insert.y),
                'bbox': [bb.extmin.x, bb.extmin.y, bb.extmax.x, bb.extmax.y]
            })
        except Exception:
            cad_blocks.append({
                'name': bname,
                'xc': float(e.dxf.insert.x),
                'yc': float(e.dxf.insert.y),
                'bbox': [float(e.dxf.insert.x)-0.25, float(e.dxf.insert.y)-0.25,
                         float(e.dxf.insert.x)+0.25, float(e.dxf.insert.y)+0.25]
            })
            
    if not cad_blocks:
        return detecciones
        
    snapped = []
    for d in detecciones:
        cx = d.get("xc", d.get("centro_cad", [0, 0])[0])
        cy = d.get("yc", d.get("centro_cad", [0, 0])[1])
        
        best_blk = None
        min_d = float('inf')
        for blk in cad_blocks:
            dist = math.hypot(blk['xc'] - cx, blk['yc'] - cy)
            if dist < min_d:
                min_d = dist
                best_blk = blk
                
        d_copy = d.copy()
        if best_blk and min_d <= max_dist:
            d_copy['cad_snapped'] = True
            d_copy['cad_block_name'] = best_blk['name']
            d_copy['cad_snap_dist'] = min_d
            d_copy['bbox_cad_snapped'] = best_blk['bbox']
            d_copy['xc_snapped'] = best_blk['xc']
            d_copy['yc_snapped'] = best_blk['yc']
        else:
            d_copy['cad_snapped'] = False
        snapped.append(d_copy)
        
    return snapped


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

    # Filtrar entidades geometricas (excluir textos/cotas que colisionan con simbolos)
    geom_entities = [e for e in msp if e.dxftype() not in ('TEXT', 'MTEXT', 'HATCH', 'DIMENSION', 'LEADER')]
    print(f"[vector-filter] Entidades geometricas: {len(geom_entities)}/{len(list(msp))} (textos excluidos del render)")

    cfg = Configuration(
        color_policy=ColorPolicy.COLOR if modo_color != "mono" else ColorPolicy.CUSTOM,
        custom_fg_color='#000000',
        custom_bg_color='#ffffff'
    )

    # 4. Generacion e Inferencia
    x_actual = x_min
    while x_actual < x_max:
        y_actual = y_min
        while y_actual < y_max:
            x_fin = x_actual + tile_size_cad
            y_fin = y_actual + tile_size_cad
            
            # Render tile preciso usando draw_entities y fijando xlim/ylim al final
            fig = plt.figure(figsize=(slice_size/100.0, slice_size/100.0), dpi=100)
            fig.patch.set_facecolor('white')
            ax = fig.add_axes([0, 0, 1, 1])
            ax.axis('off')
            
            out = MatplotlibBackend(ax)
            Frontend(ctx, out, config=cfg).draw_entities(geom_entities)
            ax.set_xlim(x_actual, x_fin)
            ax.set_ylim(y_actual, y_fin)
            ax.set_aspect('equal')
            
            fig.canvas.draw()
            buf = np.array(fig.canvas.buffer_rgba())[:, :, :3]
            plt.close(fig)
            
            tile_img = cv2.cvtColor(buf, cv2.COLOR_RGB2BGR)
            
            # Saltar tiles totalmente en blanco
            if np.all(tile_img == 255):
                y_actual += paso_cad
                continue
            
            tile_name = f"tile_X{x_actual:.2f}_Y{y_actual:.2f}.png"
            tile_path = os.path.join(temp_dir, tile_name)
            cv2.imwrite(tile_path, tile_img)
            
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

    detecciones = nms_distancia_cad(detecciones, d_min=0.45)
    print(f"[dist-nms] detecciones tras NMS por distancia CAD: {len(detecciones)}")

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


def ejecutar_vectorial_multi(dxf_path, modelos_dict=None, thresholds=None,
                             output_dir="./pipeline_out", capas_incluir=None,
                             target_px=100, modo_color="color",
                             slice_size=640, overlap=0.35,
                             device="cuda:0" if torch.cuda.is_available() else "cpu"):
    """
    Ejecuta inferencia multi-modelo en una sola pasada de rendering vectorial sobre el plano CAD.
    Aprovecha la escala calibrada, colores de capas CAD y filtrado de anotaciones.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Thresholds calibrados por defecto
    default_thresholds = {
        'interruptor_termomagnetico': 0.25,
        'interruptor_diferencial': 0.35,
        'seccionador_bajo_carga': 0.70,
        'interruptor_motorizado': 0.80,
        'interruptor_temporizado': 0.50,
        'fotocelula': 0.88,
        'grupo_electrogeno': 0.80,
        'instrumento_de_medicion_multifuncion': 0.80,
        'tablero_de_transferencia_automatica': 0.80,
        'ojo_de_buey': 0.80
    }
    if thresholds:
        default_thresholds.update(thresholds)
    thresholds = default_thresholds

    # 2. Cargar modelos YOLO
    if modelos_dict is None:
        modelos_dict = {}
        for comp in thresholds.keys():
            p = f'yolo_workspace/phase2_{comp}/weights/best.pt'
            if not os.path.exists(p):
                p = f'train-maker/models/best_{comp}.pt'
            if os.path.exists(p):
                modelos_dict[comp] = p

    loaded_models = {}
    print(f"[vector-multi] Cargando {len(modelos_dict)} modelos en {device}...")
    for comp, m_spec in modelos_dict.items():
        if isinstance(m_spec, str):
            loaded_models[comp] = YOLO(m_spec).to(device)
        else:
            loaded_models[comp] = m_spec.to(device)
        print(f"  ✓ {comp} (umbral={thresholds.get(comp, 0.5):.2f})")

    # 3. Escala y Bounds CAD
    px_per_cad, ref = calcular_factor_escala(dxf_path, target_px=target_px)
    tile_cad = slice_size / px_per_cad
    step_cad = tile_cad * (1.0 - overlap)
    print(f"[vector-multi] Escala: {px_per_cad:.2f} px/CAD | Tile: {tile_cad:.2f} CAD | Paso: {step_cad:.2f} CAD")

    doc = ezdxf.readfile(dxf_path)
    msp = doc.modelspace()
    if capas_incluir:
        aplicar_filtro_capas(doc, capas_incluir)
    
    ctx = RenderContext(doc)
    geom_entities = [e for e in msp if e.dxftype() not in ('TEXT', 'MTEXT', 'HATCH', 'DIMENSION', 'LEADER')]
    print(f"[vector-multi] Entidades geometricas: {len(geom_entities)}/{len(list(msp))} (anotaciones excluidas)")

    bbox = ezdxf.bbox.extents(geom_entities)
    x_min, y_min = bbox.extmin.x, bbox.extmin.y
    x_max, y_max = bbox.extmax.x, bbox.extmax.y

    cfg = Configuration(
        color_policy=ColorPolicy.COLOR if modo_color != "mono" else ColorPolicy.CUSTOM,
        custom_fg_color='#000000',
        custom_bg_color='#ffffff'
    )

    # 4. Inferencia multi-modelo por tile
    all_detections = []
    dpi = 100
    tile_count = 0
    t0 = time.time()

    x = x_min
    while x < x_max:
        y = y_max
        while y > y_min:
            x1 = x + tile_cad
            y1 = y - tile_cad

            fig = plt.figure(figsize=(slice_size/dpi, slice_size/dpi), dpi=dpi)
            ax = fig.add_axes([0, 0, 1, 1]); ax.axis('off')
            fe = Frontend(ctx, MatplotlibBackend(ax), config=cfg)
            fe.draw_entities(geom_entities)
            ax.set_xlim(x, x1); ax.set_ylim(y1, y); ax.set_aspect('equal')
            fig.canvas.draw()
            buf = np.array(fig.canvas.buffer_rgba())[:, :, :3]
            plt.close(fig)

            tile = cv2.cvtColor(buf, cv2.COLOR_RGB2BGR)
            if np.all(tile == 255):
                y -= step_cad
                continue
            tile_count += 1

            for comp, model in loaded_models.items():
                th = thresholds.get(comp, 0.50)
                res = model(tile, verbose=False, conf=th, device=device)[0]
                if res.boxes:
                    for b, c in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy()):
                        cad_x1 = x + (b[0] / slice_size) * tile_cad
                        cad_x2 = x + (b[2] / slice_size) * tile_cad
                        cad_y1 = y - (b[3] / slice_size) * tile_cad
                        cad_y2 = y - (b[1] / slice_size) * tile_cad
                        all_detections.append({
                            'clase': comp,
                            'conf': float(c),
                            'bbox_cad': [cad_x1, min(cad_y1, cad_y2), cad_x2, max(cad_y1, cad_y2)],
                            'centro_cad': [(cad_x1 + cad_x2) / 2.0, (cad_y1 + cad_y2) / 2.0]
                        })
            y -= step_cad
        x += step_cad

    # 5. Post-proceso NMS por clase y eliminacion de anidadas
    final_dets = []
    for comp in loaded_models.keys():
        dets_c = [d for d in all_detections if d['clase'] == comp]
        dets_c = nms_agnostico_clase_cad(dets_c, iou_thresh=0.45)
        dets_c = eliminar_anidadas_cad(dets_c, ios_thresh=0.70)
        final_dets.extend(dets_c)

    # Guardar resultados
    json_path = os.path.join(output_dir, "detecciones_multi.json")
    with open(json_path, "w") as f:
        def convert_numpy(obj):
            if isinstance(obj, np.generic): return obj.item()
            raise TypeError
        json.dump(final_dets, f, indent=2, default=convert_numpy)

    conteo = Counter(d['clase'] for d in final_dets)
    print(f"\n[vector-multi] Procesados {tile_count} tiles no vacios en {time.time()-t0:.1f}s")
    print("=" * 60)
    print("CONTEO MULTI-MODELO:")
    print("=" * 60)
    for comp, n in conteo.most_common():
        confs = [d['conf'] for d in final_dets if d['clase'] == comp]
        media = sum(confs) / len(confs) if confs else 0
        print(f"  {comp:<35} {n:>3}  conf media={media:.3f}")
    print("=" * 60)

    return conteo, final_dets

