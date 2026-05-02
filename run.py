from ultralytics import YOLO
import cv2
import numpy as np


def redimensionar_proporcional(img, max_ancho=1280, max_alto=720):
    """
    Achica la imagen si es muy grande, manteniendo la relación de aspecto (no la estira).
    """
    alto, ancho = img.shape[:2]

    # Solo achicamos si la imagen es más grande que nuestro límite
    if ancho > max_ancho or alto > max_alto:
        # Calculamos el factor de escala necesario
        escala = min(max_ancho / ancho, max_alto / alto)
        nuevo_ancho = int(ancho * escala)
        nuevo_alto = int(alto * escala)
        # Redimensionamos
        img = cv2.resize(img, (nuevo_ancho, nuevo_alto), interpolation=cv2.INTER_AREA)

    return img


def probar_plano_real():
    # 1. Cargamos el modelo
    ruta_modelo = r"C:\Users\Tomas\Documents\LAB3\CLAUDIO_AI\best.pt"
    model = YOLO(ruta_modelo)

    ruta_imagen_real = "img.png"  # Reemplazá con tu imagen

    img = cv2.imread(ruta_imagen_real)
    if img is None:
        print(f"❌ Error: No se encontró la imagen {ruta_imagen_real}")
        return

    # --- PROCESAMIENTO A BLANCO Y NEGRO ---
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    brillo_promedio = np.mean(img_gray)
    if brillo_promedio < 127:
        print("Fondo oscuro detectado. Invirtiendo colores...")
        img_gray = cv2.bitwise_not(img_gray)

    img_final = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    # -----------------------------------------

    print("Analizando plano...")
    resultados = model.predict(source=img_final, conf=0.8, save=True)

    # Mostrar el resultado sin deformar
    for r in resultados:
        im_array = r.plot()  # YOLO dibuja las cajas

        # Achicamos la imagen manteniendo proporciones (ideal para tu monitor)
        im_array_vista = redimensionar_proporcional(im_array, max_ancho=1280, max_alto=720)

        # Volvemos a usar la ventana normal (WINDOW_AUTOSIZE), ya que la imagen ahora tiene el tamaño correcto
        cv2.imshow("Prueba Liard", im_array_vista)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    probar_plano_real()