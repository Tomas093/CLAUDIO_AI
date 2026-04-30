import os
import re
import math
import cv2  # Importante para guardar las imágenes de debug
from ultralytics import YOLO


def filtrar_duplicados(detecciones_globales, umbral_distancia):
    """
    Elimina detecciones solapadas calculando la distancia euclidiana en unidades CAD.
    """
    detecciones_filtradas = []

    # Ordenar por confianza (nos quedamos con la mejor predicción)
    detecciones_globales.sort(key=lambda d: d['conf'], reverse=True)

    for det in detecciones_globales:
        es_duplicado = False
        for aceptada in detecciones_filtradas:
            if det['clase'] == aceptada['clase']:
                dist_x = det['x'] - aceptada['x']
                dist_y = det['y'] - aceptada['y']
                distancia = math.hypot(dist_x, dist_y)

                if distancia < umbral_distancia:
                    es_duplicado = True
                    break

        if not es_duplicado:
            detecciones_filtradas.append(det)

    return detecciones_filtradas


def ejecutar_conteo_liard(ruta_modelo, carpeta_tiles, tile_size_cad, umbral_distancia, umbral_confianza):
    print("Cargando modelo YOLO...")
    model = YOLO(ruta_modelo)

    detecciones_totales = []
    archivos_procesados = 0

    # Crear carpeta de debug para guardar solo lo que detecta
    carpeta_debug = "./debug_vision"
    os.makedirs(carpeta_debug, exist_ok=True)

    print("Iniciando inferencia...")
    for archivo in os.listdir(carpeta_tiles):
        if not archivo.endswith(".png"):
            continue

        # 1. Extraer coordenadas globales de origen del nombre del archivo
        match = re.search(r"tile_[xX]([0-9.-]+)_[yY]([0-9.-]+)\.png", archivo)
        if not match:
            print(f"Ignorando {archivo}: No tiene el formato de coordenadas esperado.")
            continue

        tile_x_origen = float(match.group(1))
        tile_y_origen = float(match.group(2))

        # 2. Correr inferencia con YOLO
        ruta_img = os.path.join(carpeta_tiles, archivo)
        # verbose=False para que no inunde la consola
        # 2. Correr inferencia con YOLO
        # Le pasamos el umbral directamente a YOLO para que ni siquiera dibuje lo que no llega
        resultados = model(ruta_img, conf=umbral_confianza, verbose=False)

        for result in resultados:
            # ---> LÓGICA DE DEBUG VISUAL <---
            # Guardamos la imagen pintada solo si encontró algo
            if len(result.boxes) > 0:
                img_pintada = result.plot()
                ruta_debug = os.path.join(carpeta_debug, f"detectado_{archivo}")
                cv2.imwrite(ruta_debug, img_pintada)
            # --------------------------------

            # Obtener tamaño de la imagen en píxeles (ej. 1800x1800)
            img_height, img_width = result.orig_shape

            # Calcular factor de escala (Unidades CAD por Píxel)
            escala_x = tile_size_cad / img_width
            escala_y = tile_size_cad / img_height

            # 3. Procesar cada objeto detectado
            boxes = result.boxes
            for i in range(len(boxes)):
                confianza = float(boxes.conf[i].cpu().numpy())

                # ---> FILTRO DE CONFIANZA <---
                # Si la IA no está segura, descartamos el objeto y pasamos al siguiente
                if confianza < umbral_confianza:
                    continue
                # ------------------------------

                # YOLO devuelve centro_x, centro_y, ancho, alto en píxeles
                x_centro_px, y_centro_px, w, h = boxes.xywh[i].cpu().numpy()
                clase_id = int(boxes.cls[i].cpu().numpy())
                nombre_clase = model.names[clase_id]

                # 4. CONVERSIÓN CRÍTICA: Píxeles a Unidades CAD
                local_x_cad = x_centro_px * escala_x
                # OJO ACÁ: Las imágenes en computación tienen el Y=0 arriba.
                # En AutoCAD, el Y=0 está abajo. Hay que invertir el eje Y.
                local_y_cad = (img_height - y_centro_px) * escala_y

                # 5. Calcular coordenada global final
                global_x = tile_x_origen + local_x_cad
                global_y = tile_y_origen + local_y_cad

                detecciones_totales.append({
                    'clase': nombre_clase,
                    'x': global_x,
                    'y': global_y,
                    'conf': confianza
                })

        archivos_procesados += 1

    print(f"Inferencia terminada. Se procesaron {archivos_procesados} imágenes.")
    print(
        f"Detecciones válidas (> {umbral_confianza * 100}% seguras, con posibles duplicados): {len(detecciones_totales)}")

    # 6. Filtrar duplicados en las zonas de solapamiento
    detecciones_finales = filtrar_duplicados(detecciones_totales, umbral_distancia)

    print("-" * 30)
    print("RESUMEN DE MATERIALES (LIARD):")
    print("-" * 30)

    # Contar por clase
    conteo_final = {}
    for det in detecciones_finales:
        clase = det['clase']
        conteo_final[clase] = conteo_final.get(clase, 0) + 1

    for clase, cantidad in conteo_final.items():
        print(f"-> {clase.upper()}: {cantidad} unidades")

    return conteo_final, detecciones_finales


# --- EJECUCIÓN ---
if __name__ == "__main__":
    conteo, detalles = ejecutar_conteo_liard(
        ruta_modelo="best.pt",
        carpeta_tiles="./tiles_output_manual",
        tile_size_cad=5,  # Ajustá esto según cómo sacaste los tiles
        umbral_distancia=1.25,  # Distancia mínima entre dos componentes reales
        umbral_confianza=0.80  # Solo cuenta objetos si la IA tiene un 50% o más de seguridad
    )