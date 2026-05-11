"""
Ejemplo de template matching con SIFT en planos electricos DXF.

Uso:
    python sift_example.py --dxf plano.dxf --template interruptor.png

El flujo es:
  1. Calcular factor de escala del DXF + renderizar a PNG en grayscale.
  2. Cargar template (debe ser un recorte tomado a la MISMA escala).
  3. Detectar y describir keypoints con SIFT.
  4. Matchear template vs plano con FLANN + filtro de Lowe.
  5. Convertir matches a coordenadas CAD usando la metadata.

Requiere:
    pip install ezdxf matplotlib pillow numpy opencv-contrib-python

Nota: opencv-contrib-python (no opencv-python pelado) porque SIFT
estaba bajo patente y vive en contrib historicamente. En versiones
recientes (>=4.4) ya viene en core, pero contrib lo asegura.
"""

import os
import argparse
import cv2
import numpy as np

from dxf_to_image import renderizar_dxf


def render_plano(dxf_path, output_dir, target_px=150):
    """Convierte el DXF a una imagen grayscale lista para SIFT."""
    os.makedirs(output_dir, exist_ok=True)
    img_path = os.path.join(output_dir, "plano_render.png")
    metadata = renderizar_dxf(
        dxf_path=dxf_path,
        output_path=img_path,
        modo_color="grayscale",  # SIFT trabaja en grayscale
        target_px=target_px,     # 150 px tipico para SIFT
    )
    return img_path, metadata


def matchear_sift(plano_path, template_path,
                   ratio_lowe=0.7, min_matches=4, n_features=10000):
    """
    Encuentra el template dentro del plano usando SIFT + FLANN.
    Devuelve lista de matches buenos como dicts con keypoint del plano.
    """
    plano = cv2.imread(plano_path, cv2.IMREAD_GRAYSCALE)
    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    if plano is None:
        raise FileNotFoundError(plano_path)
    if template is None:
        raise FileNotFoundError(template_path)

    # n_features pone tope a la cantidad de keypoints (control de memoria)
    sift = cv2.SIFT_create(nfeatures=n_features)

    print("[sift] extrayendo keypoints del template...")
    kp_t, desc_t = sift.detectAndCompute(template, None)
    print(f"[sift] template: {len(kp_t)} keypoints")

    print("[sift] extrayendo keypoints del plano...")
    kp_p, desc_p = sift.detectAndCompute(plano, None)
    print(f"[sift] plano: {len(kp_p)} keypoints")

    if desc_t is None or desc_p is None:
        print("[sift] sin descriptores, no se puede matchear.")
        return [], kp_t, kp_p

    # FLANN matcher (rapido para muchos descriptores)
    flann = cv2.FlannBasedMatcher(
        dict(algorithm=1, trees=5),  # KDTree
        dict(checks=50),
    )
    raw_matches = flann.knnMatch(desc_t, desc_p, k=2)

    # Filtro de Lowe: la mejor distancia debe ser
    # significativamente menor que la segunda mejor
    buenos = []
    for par in raw_matches:
        if len(par) < 2:
            continue
        m, n = par
        if m.distance < ratio_lowe * n.distance:
            buenos.append(m)

    print(f"[sift] matches buenos: {len(buenos)} (de {len(raw_matches)} totales)")

    if len(buenos) < min_matches:
        print(f"[sift] AVISO: pocos matches ({len(buenos)} < {min_matches}). "
              "El template podria no estar en el plano, o estar a escala distinta.")

    return buenos, kp_t, kp_p


def localizar_template(matches, kp_t, kp_p, template_shape):
    """
    Estima la posicion del template en el plano usando homografia con RANSAC.
    Devuelve las 4 esquinas del template proyectadas al plano, o None si falla.
    """
    if len(matches) < 4:
        return None

    src_pts = np.float32([kp_t[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_p[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if H is None:
        return None

    h, w = template_shape[:2]
    esquinas_template = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
    esquinas_plano = cv2.perspectiveTransform(esquinas_template, H)
    return esquinas_plano.reshape(-1, 2), H, mask


def px_a_cad(px_x, px_y, metadata):
    """Convierte un punto pixel (en la imagen renderizada) a coords CAD."""
    px_per_cad = metadata["px_per_cad"]
    x_min_cad = metadata["x_min_cad"]
    y_max_cad = metadata["y_max_cad"]
    cad_x = x_min_cad + px_x / px_per_cad
    # Y de imagen va para abajo, Y de CAD va para arriba
    cad_y = y_max_cad - px_y / px_per_cad
    return cad_x, cad_y


def dibujar_resultado(plano_path, esquinas_plano, output_path):
    """Dibuja un poligono sobre el plano marcando donde se encontro el template."""
    img = cv2.imread(plano_path)
    if img is None or esquinas_plano is None:
        return
    pts = esquinas_plano.astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(img, [pts], isClosed=True, color=(0, 200, 0), thickness=3)
    cv2.imwrite(output_path, img)
    print(f"[viz] resultado guardado en {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dxf", required=True, help="Plano DXF de entrada")
    parser.add_argument("--template", required=True, help="PNG del simbolo a buscar")
    parser.add_argument("--out", default="./sift_out", help="Carpeta de salida")
    parser.add_argument("--target-px", type=int, default=150,
                        help="Tamano en px de un simbolo tipico en el render")
    parser.add_argument("--ratio", type=float, default=0.7,
                        help="Umbral del filtro de Lowe (mas chico = mas estricto)")
    parser.add_argument("--n-features", type=int, default=10000,
                        help="Tope de keypoints SIFT (controla memoria/velocidad)")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # 1. Renderizar el DXF a imagen
    plano_path, metadata = render_plano(args.dxf, args.out, target_px=args.target_px)

    # 2. SIFT + matching
    matches, kp_t, kp_p = matchear_sift(
        plano_path, args.template,
        ratio_lowe=args.ratio,
        n_features=args.n_features,
    )
    if not matches:
        return

    # 3. Localizar el template con homografia
    template = cv2.imread(args.template, cv2.IMREAD_GRAYSCALE)
    resultado = localizar_template(matches, kp_t, kp_p, template.shape)

    if resultado is None:
        print("[match] no se pudo estimar homografia.")
        return

    esquinas_plano, H, mask = resultado
    n_inliers = int(mask.sum())
    print(f"[match] homografia OK con {n_inliers}/{len(matches)} inliers")

    # 4. Centro del match en pixeles y CAD
    cx_px = float(esquinas_plano[:, 0].mean())
    cy_px = float(esquinas_plano[:, 1].mean())
    cx_cad, cy_cad = px_a_cad(cx_px, cy_px, metadata)
    print(f"[match] centro en pixeles: ({cx_px:.0f}, {cy_px:.0f})")
    print(f"[match] centro en CAD:     ({cx_cad:.2f}, {cy_cad:.2f})")

    # 5. Dibujar resultado
    out_viz = os.path.join(args.out, "match_visual.png")
    dibujar_resultado(plano_path, esquinas_plano, out_viz)


if __name__ == "__main__":
    main()
