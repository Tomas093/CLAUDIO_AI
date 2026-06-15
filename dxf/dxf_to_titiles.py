import os
import ezdxf
from ezdxf.addons.drawing import RenderContext, Frontend
from ezdxf.addons.drawing.matplotlib import MatplotlibBackend
import matplotlib.pyplot as plt


def generar_tiles_manuales(dxf_path, output_folder, tile_size, overlap, dpi=300, max_tiles=5):
    """
    Divide un plano DXF en recortes solapados usando valores de escala manuales.
    Incluye el fix para ezdxf en el cálculo del Bounding Box.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    try:
        print(f"Leyendo archivo: {dxf_path}...")
        doc = ezdxf.readfile(dxf_path)
        msp = doc.modelspace()
    except Exception as e:
        print(f"Error crítico al leer el archivo: {e}")
        return

    # 1. Obtener los límites del dibujo con la sintaxis corregida
    bbox = ezdxf.bbox.extents(msp)
    if not bbox.has_data:
        print("El ModelSpace está vacío o no tiene límites calculables.")
        return

    # ACÁ ESTÁ EL FIX APLICADO: extmin y extmax
    x_min, y_min = bbox.extmin.x, bbox.extmin.y
    x_max, y_max = bbox.extmax.x, bbox.extmax.y

    print(f"Límites detectados: X({x_min:.2f} a {x_max:.2f}), Y({y_min:.2f} a {y_max:.2f})")

    # 2. Calcular el avance real
    paso = tile_size - overlap
    if paso <= 0:
        print("Error: El solapamiento no puede ser mayor o igual al tamaño del tile.")
        return

    ctx = RenderContext(doc)
    
    # FORZAMOS INICIO DONDE HAY SÍMBOLOS PARA DEMOSTRACIÓN (evita espacios en blanco)
    x_min = 3345.0
    y_min = 1605.0
    x_actual = x_min
    contador = 0

    print("Iniciando renderizado de tiles...")

    # 3. Bucle de generación
    while x_actual < x_max:
        y_actual = y_min
        while y_actual < y_max:
            x_fin = x_actual + tile_size
            y_fin = y_actual + tile_size

            # Crear la figura sin márgenes
            fig = plt.figure(figsize=(6, 6))
            fig.patch.set_facecolor('white')
            ax = fig.add_axes([0, 0, 1, 1])
            ax.set_xlim(x_actual, x_fin)
            ax.set_ylim(y_actual, y_fin)
            ax.axis('off')

            out = MatplotlibBackend(ax)
            Frontend(ctx, out).draw_layout(msp, finalize=False)

            nombre_archivo = f"tile_X{x_actual:.2f}_Y{y_actual:.2f}.png"
            ruta_completa = os.path.join(output_folder, nombre_archivo)

            plt.savefig(ruta_completa, dpi=dpi, facecolor=fig.get_facecolor(), edgecolor='none')
            plt.close(fig)

            print(f"Guardado: {nombre_archivo}")
            contador += 1
            
            if contador >= max_tiles:
                break

            y_actual += paso
        if contador >= max_tiles:
            break
        x_actual += paso

    print(f"Listo. Se generaron {contador} imágenes.")


# --- EJECUCIÓN ---
if __name__ == "__main__":
    generar_tiles_manuales(
        dxf_path="jijiji.dxf",
        output_folder="../pipeline_out/tiles_output_highres",
        tile_size=5.0,
        overlap=1.0,
        dpi=300,
        max_tiles=5
    )