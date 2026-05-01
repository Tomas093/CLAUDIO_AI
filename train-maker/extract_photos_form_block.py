import os
import ezdxf
import re
from ezdxf.addons.drawing import RenderContext, Frontend
from ezdxf.addons.drawing.matplotlib import MatplotlibBackend
import matplotlib.pyplot as plt


def extraer_fotos_de_bloques(dxf_path, output_folder, nombres_bloques_objetivo=None, margen=5.0, dpi=300):
    """
    Busca bloques específicos en el DXF y genera una imagen centrada de cada uno.

    nombres_bloques_objetivo: Lista de strings con los nombres de los bloques a buscar.
                              Si es None, procesa todos los bloques del plano.
    margen: Cuántas unidades CAD sumar alrededor del centro del bloque para el encuadre.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    try:
        print(f"Leyendo archivo: {dxf_path}...")
        doc = ezdxf.readfile(dxf_path)
        msp = doc.modelspace()
    except Exception as e:
        print(f"Error al leer el archivo: {e}")
        return

    # Buscar todas las inserciones de bloques en el ModelSpace
    inserciones = msp.query('INSERT')

    if not inserciones:
        print("No se encontraron bloques (entidades INSERT) en este plano.")
        return

    ctx = RenderContext(doc)
    contador = 0

    print(f"Se encontraron {len(inserciones)} inserciones en total. Procesando...")

    for insercion in inserciones:
        nombre_bloque = insercion.dxf.name

        # Si le pasamos una lista de nombres, filtramos. Si no, agarramos todos.
        # Filtrar también los bloques cuyo nombre empieza con 'U' seguido de un número (p.ej. U10, U4)
        if (nombres_bloques_objetivo and nombre_bloque not in nombres_bloques_objetivo) or nombre_bloque.startswith(r'^U\d'):
            continue

        # Obtener las coordenadas del punto de inserción (el "centro" del bloque)
        punto_x = insercion.dxf.insert.x
        punto_y = insercion.dxf.insert.y

        # Definir la "ventana" de la foto (Bounding Box local)
        # Centramos el bloque en la imagen restando y sumando el margen
        x_min = punto_x - margen
        x_max = punto_x + margen
        y_min = punto_y - margen
        y_max = punto_y + margen

        # Configurar la foto
        fig = plt.figure(figsize=(4, 4))
        fig.patch.set_facecolor('white')
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.axis('off')

        # Renderizar
        out = MatplotlibBackend(ax)
        Frontend(ctx, out).draw_layout(msp, finalize=False)

        # Nombre de archivo inteligente para Liard:
        # Guardamos el nombre real del bloque y sus coordenadas exactas.
        nombre_archivo = f"{nombre_bloque}_X{punto_x:.2f}_Y{punto_y:.2f}.png"

        # Limpiar caracteres raros del nombre del bloque (por si las dudas)
        nombre_archivo = "".join([c for c in nombre_archivo if c.isalnum() or c in " ._-"])

        ruta_completa = os.path.join(output_folder, nombre_archivo)

        plt.savefig(ruta_completa, dpi=dpi, facecolor=fig.get_facecolor(), edgecolor='none')
        plt.close(fig)

        print(f"Foto extraída: {nombre_archivo}")
        contador += 1

    print(f"Proceso finalizado. Se extrajeron {contador} imágenes de bloques.")


# --- EJECUCIÓN ---
extraer_fotos_de_bloques(
    dxf_path="Tablerotsbe.dxf",
    output_folder="./bloques_extraidos",
    margen=2,  # El "zoom" de la cámara. Ajustá según el tamaño del bloque.
    dpi=300
)