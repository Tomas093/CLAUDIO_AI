import os
import shutil

# Tus rutas
src_1 = r"C:\Users\Tomas\Documents\LAB3\CLAUDIO_AI\train-maker\datasets\dataset_1"
destino = r"C:\Users\Tomas\Documents\LAB3\CLAUDIO_AI\train-maker\datasets\dataset_unified"

carpetas = [
    r"train\images", r"train\labels",
    r"val\images", r"val\labels"
]

for subcarpeta in carpetas:
    ruta_origen = os.path.join(src_1, subcarpeta)
    ruta_dest = os.path.join(destino, subcarpeta)

    # Asegurarnos de que la carpeta de destino exista
    os.makedirs(ruta_dest, exist_ok=True)

    if os.path.exists(ruta_origen):
        for archivo in os.listdir(ruta_origen):
            # Acá está la clave: le metemos un prefijo al archivo nuevo para no pisar el viejo
            nuevo_nombre = f"simbolo1_{archivo}"

            path_archivo_origen = os.path.join(ruta_origen, archivo)
            path_archivo_destino = os.path.join(ruta_dest, nuevo_nombre)

            # Copiamos el archivo con su nuevo nombre
            shutil.copy2(path_archivo_origen, path_archivo_destino)

        print(f"Archivos de {subcarpeta} agregados con prefijo 'simbolo1_'")
    else:
        print(f"Falta la carpeta origen: {ruta_origen}")

print("¡Listo! Todo agregado sin sobreescribir ni un solo archivo.")