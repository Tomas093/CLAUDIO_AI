"""
========================================================================================
DETECTOR UNIVERSAL DE COMPONENTES ELECTRICOS EN PLANOS UNIFILARES (YOLO NANO CAD)
========================================================================================
Este script realiza la detección completa de componentes eléctricos (disyuntores,
térmicas, diferenciales, seccionadores, contactores, descargadores, transformadores,
luces piloto, instrumentos y borneras) sobre planos en formato DXF o imágenes (PNG/JPG).

Características:
1. Renderizado vectorial limpio con ezdxf (filtrado de cotas, textos y tramas).
2. Slicing adaptativo SAHI (ventanas 640x640 con 80% de solapamiento y padding de borde).
3. Inferencia por lotes en GPU o CPU.
4. Post-procesamiento NMS en coordenadas CAD (IoU, cajas anidadas y distancia euclidiana).
5. Exportación de lámina visual con bounding boxes en verde, CSV con coordenadas y JSON.
========================================================================================
"""

import os
import sys
import csv
import json
import time
import math
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

# ezdxf para procesamiento de planos CAD
import ezdxf
import ezdxf.bbox
from ezdxf.addons.drawing import RenderContext, Frontend
from ezdxf.addons.drawing.matplotlib import MatplotlibBackend
from ezdxf.addons.drawing.config import Configuration, ColorPolicy, TextPolicy, HatchPolicy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Configuración de codificación de salida
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')


# ======================================================================================
# 1. RENDERIZADO VECTORIAL DEL DXF A IMAGEN LIMPIA
# ======================================================================================
def render_dxf(dxf_path, px_per_cad=75.0, is_industrial=True):
    """
    Lee un archivo DXF, filtra las capas de texto, cotas y cartelas, y lo dibuja
    como una imagen de alta resolución a la escala px/CAD especificada.
    """
    dxf_path = Path(dxf_path)
    if not dxf_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo DXF: {dxf_path}")

    doc = ezdxf.readfile(str(dxf_path))
    msp = doc.modelspace()

    # Auto-detect DXFs designed for dark backgrounds:
    # If layers use non-white colors (ACI != 7 and != 0), those colored entities
    # (red=1, yellow=2, green=3, cyan=4, blue=5, magenta=6, etc.) will be
    # nearly invisible on a white background. Force ALL layers to color 7
    # so that COLOR_SWAP_BW renders everything as black on white.
    colored_layers = sum(1 for layer in doc.layers
                         if hasattr(layer.dxf, 'color')
                         and layer.dxf.color not in (0, 7, 256))
    if colored_layers > 0:
        for layer in doc.layers:
            layer.dxf.color = 7

    ctx = RenderContext(doc)

    # Excluir entidades que introducen ruido sobre los símbolos eléctricos
    excluded_types = ('TEXT', 'MTEXT', 'DIMENSION', 'LEADER')
    excluded_layers = ('IE-UN-TEXTOS', 'FORMATO', 'CARATULA', 'DEFPOINTS')

    if is_industrial:
        geom_entities = [
            e for e in msp
            if e.dxftype() not in excluded_types
            and e.dxf.layer.upper() not in excluded_layers
        ]
    else:
        geom_entities = [e for e in msp if e.dxftype() not in excluded_types]

    if not geom_entities:
        geom_entities = [e for e in msp if e.dxftype() not in ('TEXT', 'MTEXT')]

    bbox = ezdxf.bbox.extents(geom_entities)
    x_min, y_min = bbox.extmin.x, bbox.extmin.y
    x_max, y_max = bbox.extmax.x, bbox.extmax.y

    W_cad = max(1e-3, x_max - x_min)
    H_cad = max(1e-3, y_max - y_min)

    W_px = int(round(W_cad * px_per_cad))
    H_px = int(round(H_cad * px_per_cad))

    dpi = 100
    fig = plt.figure(figsize=(W_px / dpi, H_px / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')

    cfg = Configuration(
        color_policy=ColorPolicy.COLOR_SWAP_BW,
        hatch_policy=HatchPolicy.NORMAL,
        custom_bg_color='#ffffff'
    )
    backend = MatplotlibBackend(ax)
    frontend = Frontend(ctx, backend, config=cfg)
    frontend.draw_entities(geom_entities)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect('equal')

    fig.canvas.draw()
    buf = np.array(fig.canvas.buffer_rgba())[:, :, :3]
    plt.close(fig)

    img_bgr = cv2.cvtColor(buf, cv2.COLOR_RGB2BGR)

    meta = {
        'px_per_cad': px_per_cad,
        'x_min_cad': x_min, 'y_min_cad': y_min,
        'x_max_cad': x_max, 'y_max_cad': y_max,
        'W_cad': W_cad, 'H_cad': H_cad,
        'W_px': img_bgr.shape[1], 'H_px': img_bgr.shape[0]
    }
    return img_bgr, meta


# ======================================================================================
# 2. SLICING ADAPTATIVO (VENTANA DESLIZANTE CON SOLAPAMIENTO)
# ======================================================================================
def generate_slices(H, W, slice_size=640, overlap=0.80):
    """
    Genera las coordenadas de corte con solapamiento alto (80%) para que ningún
    símbolo quede cortado en los límites de una baldosa.
    """
    step = int(round(slice_size * (1.0 - overlap)))
    slices = []
    y = 0
    while y < H:
        x = 0
        y2 = min(y + slice_size, H)
        y1 = max(0, y2 - slice_size)
        while x < W:
            x2 = min(x + slice_size, W)
            x1 = max(0, x2 - slice_size)
            slices.append((x1, y1, x2, y2))
            if x2 >= W:
                break
            x += step
        if y2 >= H:
            break
        y += step
    return slices


# ======================================================================================
# 3. POST-PROCESAMIENTO NMS EN COORDENADAS CAD
# ======================================================================================
def nms_iou_cad(detections, iou_thresh=0.45):
    """Elimina duplicados de un mismo símbolo detectado en múltiples tiles adyacentes."""
    if not detections:
        return []
    boxes = np.array([d["bbox_cad"] for d in detections])
    confs = np.array([d["conf"] for d in detections])
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
    return [detections[i] for i in keep]


def eliminar_anidadas(detections, ios_thresh=0.60):
    """Elimina cajas más pequeñas o fantasmas que quedan dentro de otra caja mayor."""
    if not detections:
        return []
    ord_idx = sorted(range(len(detections)), key=lambda i: -detections[i]["conf"])
    eliminar = set()
    for idx_a, i in enumerate(ord_idx):
        if i in eliminar:
            continue
        bi = detections[i]["bbox_cad"]
        ai = max((bi[2] - bi[0]) * (bi[3] - bi[1]), 1e-9)
        for j in ord_idx[idx_a + 1:]:
            if j in eliminar:
                continue
            bj = detections[j]["bbox_cad"]
            aj = max((bj[2] - bj[0]) * (bj[3] - bj[1]), 1e-9)
            xx1 = max(bi[0], bj[0])
            yy1 = max(bi[1], bj[1])
            xx2 = min(bi[2], bj[2])
            yy2 = min(bi[3], bj[3])
            inter = max(0, xx2 - xx1) * max(0, yy2 - yy1)
            if inter <= 0:
                continue
            ios = inter / min(ai, aj)
            if ios >= ios_thresh:
                eliminar.add(j)
    return [d for i, d in enumerate(detections) if i not in eliminar]


def nms_distancia_cad(detections, d_min=0.35):
    """
    NMS inteligente por distancia euclidiana entre centros en CAD (d_min ~ 0.35 unidades).
    Evita que disyuntores bipolares o tetrapolares contiguos generen duplicados cruzados.
    """
    if not detections:
        return []
    confs = np.array([d["conf"] for d in detections])
    centroids = np.array([
        [
            d.get("xc", d.get("x_cad", (d["bbox_cad"][0] + d["bbox_cad"][2]) / 2.0)),
            d.get("yc", d.get("y_cad", (d["bbox_cad"][1] + d["bbox_cad"][3]) / 2.0))
        ]
        for d in detections
    ])
    idxs = confs.argsort()[::-1]
    keep = []
    while len(idxs) > 0:
        i = idxs[0]
        keep.append(int(i))
        if len(idxs) == 1:
            break
        rest = idxs[1:]
        dists = np.linalg.norm(centroids[rest] - centroids[i], axis=1)
        idxs = rest[dists >= d_min]
    return [detections[i] for i in keep]


# ======================================================================================
# 4. CLASE PRINCIPAL: DETECTOR UNIFILAR
# ======================================================================================
class DetectorUnifilar:
    def __init__(self, model_path="best_componente_nano.pt", device=None):
        if device is None:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        if not os.path.exists(model_path):
            # Buscar en directorio local o train-maker
            alt_path = os.path.join(os.path.dirname(__file__), model_path)
            if os.path.exists(alt_path):
                model_path = alt_path
            else:
                raise FileNotFoundError(f"No se encontraron los pesos del modelo en: {model_path}")

        print(f"[Detector] Cargando modelo: {model_path} en {self.device}")
        self.model = YOLO(model_path)

    def detectar_dxf(self, dxf_path, output_dir="./resultados", conf_thresh=0.15,
                     px_per_cad=75.0, is_industrial=True, batch_size=32, d_min=None):
        """
        Ejecuta el pipeline completo desde archivo DXF: renderizado, slicing,
        inferencia, NMS y exportación de lámina visual, CSV y JSON.
        """
        os.makedirs(output_dir, exist_ok=True)
        dxf_stem = Path(dxf_path).stem

        print(f"\n[1/4] Renderizando {dxf_path} a {px_per_cad} px/CAD...")
        img_bgr, meta = render_dxf(dxf_path, px_per_cad=px_per_cad, is_industrial=is_industrial)
        H_px, W_px = img_bgr.shape[:2]
        print(f"      Dimensiones de renderizado: {W_px} x {H_px} px")

        # Inferencia
        return self._procesar_imagen(
            img_bgr=img_bgr,
            meta=meta,
            stem=dxf_stem,
            output_dir=output_dir,
            conf_thresh=conf_thresh,
            batch_size=batch_size,
            d_min=d_min
        )

    def detectar_imagen(self, img_path, output_dir="./resultados", conf_thresh=0.15, batch_size=32, d_min=None):
        """
        Ejecuta el detector directamente sobre una imagen (PNG / JPG / BMP) exportada de CAD.
        """
        os.makedirs(output_dir, exist_ok=True)
        img_stem = Path(img_path).stem
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            raise FileNotFoundError(f"No se pudo cargar la imagen: {img_path}")

        H_px, W_px = img_bgr.shape[:2]
        meta = {
            'px_per_cad': 1.0,
            'x_min_cad': 0.0, 'y_min_cad': 0.0,
            'x_max_cad': float(W_px), 'y_max_cad': float(H_px),
            'W_cad': float(W_px), 'H_cad': float(H_px),
            'W_px': W_px, 'H_px': H_px
        }

        return self._procesar_imagen(
            img_bgr=img_bgr,
            meta=meta,
            stem=img_stem,
            output_dir=output_dir,
            conf_thresh=conf_thresh,
            batch_size=batch_size,
            d_min=d_min
        )

    def _procesar_imagen(self, img_bgr, meta, stem, output_dir, conf_thresh, batch_size, d_min=None):
        H_px, W_px = img_bgr.shape[:2]
        W_cad, H_cad = meta['W_cad'], meta['H_cad']
        x_min, y_max = meta['x_min_cad'], meta['y_max_cad']

        # Padding de 320 px para que los símbolos perimétricos no se corten
        pad = 320
        slice_size = 640
        padded_img = cv2.copyMakeBorder(
            img_bgr, pad, pad, pad, pad,
            cv2.BORDER_CONSTANT, value=[255, 255, 255]
        )
        pad_h, pad_w = padded_img.shape[:2]

        print("[2/4] Generando cortes con 80% de solapamiento (slice 640x640)...")
        slices = generate_slices(pad_h, pad_w, slice_size=slice_size, overlap=0.80)
        print(f"      Total baldosas a procesar: {len(slices)}")

        # Inferencia por lotes a baja confianza (conf=0.01) para no perder bordes
        print(f"[3/4] Ejecutando inferencia en {self.device}...")
        raw_dets = []
        t0 = time.time()
        for i in range(0, len(slices), batch_size):
            b_slices = slices[i:i + batch_size]
            b_imgs = [padded_img[y1:y2, x1:x2] for (x1, y1, x2, y2) in b_slices]
            results = self.model(b_imgs, verbose=False, conf=0.01, device=self.device)

            for s_idx, res in enumerate(results):
                x1_tile, y1_tile, _, _ = b_slices[s_idx]
                if res.boxes:
                    xyxy = res.boxes.xyxy.cpu().numpy()
                    confs = res.boxes.conf.cpu().numpy()
                    for b, c in zip(xyxy, confs):
                        px_x1 = max(0, min(W_px, (x1_tile + b[0]) - pad))
                        px_y1 = max(0, min(H_px, (y1_tile + b[1]) - pad))
                        px_x2 = max(0, min(W_px, (x1_tile + b[2]) - pad))
                        px_y2 = max(0, min(H_px, (y1_tile + b[3]) - pad))

                        if px_x2 <= px_x1 or px_y2 <= px_y1:
                            continue

                        # Conversión exacta a coordenadas CAD (inversión de eje Y)
                        cad_x1 = x_min + (px_x1 / W_px) * W_cad
                        cad_x2 = x_min + (px_x2 / W_px) * W_cad
                        cad_y1 = y_max - (px_y2 / H_px) * H_cad
                        cad_y2 = y_max - (px_y1 / H_px) * H_cad

                        raw_dets.append({
                            'conf': float(c),
                            'bbox_cad': [min(cad_x1, cad_x2), min(cad_y1, cad_y2), max(cad_x1, cad_x2), max(cad_y1, cad_y2)],
                            'xc': (cad_x1 + cad_x2) / 2.0,
                            'yc': (cad_y1 + cad_y2) / 2.0,
                            'bbox_px': [int(px_x1), int(px_y1), int(px_x2), int(px_y2)]
                        })

        t_inf = time.time() - t0
        print(f"      Inferencia terminada en {t_inf:.1f}s ({len(slices)/(t_inf+1e-6):.1f} baldosas/s). Detecciones crudas: {len(raw_dets)}")

        # Post-procesamiento
        print(f"[4/4] Filtrando a umbral conf >= {conf_thresh} y aplicando NMS CAD...")
        dets = [d for d in raw_dets if d['conf'] >= conf_thresh]
        dets = nms_iou_cad(dets, iou_thresh=0.45)
        dets = eliminar_anidadas(dets, ios_thresh=0.60)
        # d_min en CAD (default 0.20 CAD conforme a Req R3): si es imagen px, usar ~34 px
        if d_min is None:
            calc_d_min = 0.20 if meta['px_per_cad'] > 1.0 else 34.0
        else:
            calc_d_min = d_min
        dets = nms_distancia_cad(dets, d_min=calc_d_min)
        dets.sort(key=lambda d: (round(d['yc'], 1), d['xc']))

        print(f"\n>>> EXITO: {len(dets)} componentes eléctricos detectados.")

        # Exportar lámina visual con cajas verdes
        vis_canvas = img_bgr.copy()
        for d in dets:
            x1, y1, x2, y2 = d['bbox_px']
            cv2.rectangle(vis_canvas, (x1, y1), (x2, y2), (0, 200, 0), 2)
            cv2.putText(
                vis_canvas, f"{d['conf']:.2f}", (x1, max(12, y1 - 3)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 200, 0), 1, cv2.LINE_AA
            )

        vis_file = os.path.join(output_dir, f"{stem}_visual_detections.png")
        cv2.imwrite(vis_file, vis_canvas)

        # Exportar CSV
        csv_file = os.path.join(output_dir, f"{stem}_detections.csv")
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['id', 'xc_cad', 'yc_cad', 'x1_cad', 'y1_cad', 'x2_cad', 'y2_cad', 'conf', 'px_x1', 'px_y1', 'px_x2', 'px_y2'])
            for idx, d in enumerate(dets, 1):
                b = d['bbox_cad']
                p = d['bbox_px']
                writer.writerow([idx, f"{d['xc']:.4f}", f"{d['yc']:.4f}", f"{b[0]:.4f}", f"{b[1]:.4f}", f"{b[2]:.4f}", f"{b[3]:.4f}", f"{d['conf']:.4f}", p[0], p[1], p[2], p[3]])

        # Exportar JSON
        clean_dets = []
        for d in dets:
            clean_dets.append({
                'conf': float(d['conf']),
                'bbox_cad': [float(v) for v in d['bbox_cad']],
                'xc': float(d['xc']),
                'yc': float(d['yc']),
                'bbox_px': [int(v) for v in d['bbox_px']]
            })
        clean_meta = {k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v) for k, v in meta.items()}

        json_file = os.path.join(output_dir, f"{stem}_detections.json")
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump({
                'metadata': clean_meta,
                'total_detected': len(clean_dets),
                'confidence_threshold': float(conf_thresh),
                'components': clean_dets
            }, f, indent=2)

        print(f"      [Lámina]: {vis_file}")
        print(f"      [CSV]:    {csv_file}")
        print(f"      [JSON]:   {json_file}")

        return {
            'total': len(dets),
            'detections': dets,
            'vis_image': vis_file,
            'csv_path': csv_file,
            'json_path': json_file
        }


# ======================================================================================
# 5. CLI INTERFAZ POR LINEA DE COMANDOS
# ======================================================================================
def main():
    parser = argparse.ArgumentParser(description="Detector Universal de Componentes en Planos Eléctricos")
    parser.add_argument("--dxf", type=str, help="Ruta al plano en formato .dxf")
    parser.add_argument("--img", type=str, help="Ruta a una imagen del plano (.png, .jpg)")
    parser.add_argument("--model", type=str, default="best_componente_nano.pt", help="Ruta al archivo de pesos .pt")
    parser.add_argument("--out", type=str, default="./resultados", help="Carpeta de salida")
    parser.add_argument("--conf", type=float, default=0.15, help="Umbral de confianza (default: 0.15 para 100% recall)")
    parser.add_argument("--scale", type=float, default=75.0, help="Escala px/CAD (default: 75.0)")
    parser.add_argument("--device", type=str, default=None, help="Dispositivo: 'cuda:0' o 'cpu'")
    parser.add_argument("--batch", type=int, default=32, help="Tamaño de lote para GPU")
    parser.add_argument("--d_min", type=float, default=0.20, help="Distancia mínima NMS en coordenadas CAD (default: 0.20)")

    args = parser.parse_args()

    if not args.dxf and not args.img:
        parser.error("Debes especificar un archivo de entrada con --dxf o con --img")

    detector = DetectorUnifilar(model_path=args.model, device=args.device)

    if args.dxf:
        detector.detectar_dxf(
            dxf_path=args.dxf,
            output_dir=args.out,
            conf_thresh=args.conf,
            px_per_cad=args.scale,
            batch_size=args.batch,
            d_min=args.d_min
        )
    elif args.img:
        detector.detectar_imagen(
            img_path=args.img,
            output_dir=args.out,
            conf_thresh=args.conf,
            batch_size=args.batch,
            d_min=args.d_min
        )


if __name__ == '__main__':
    main()
