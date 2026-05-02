import os
import io
import ezdxf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from ezdxf.addons.drawing import RenderContext, Frontend
from ezdxf.addons.drawing.matplotlib import MatplotlibBackend

# === CONFIGURACIÓN ===
INPUT_DXF_DIR = Path("planos_completos")  # Poné acá tus planos reales
OUTPUT_BG_DIR = Path("input/backgrounds")  # Donde se guardarán los fondos
TILE_SIZE = 640  # Tamaño de cada recorte (en píxeles)
OVERLAP = 320  # Solapamiento para no perder info
DPI = 4098  # Resolución del renderizado


def dxf_to_image(dxf_path):
    """Convierte un DXF completo a una imagen en memoria."""
    try:
        doc = ezdxf.readfile(str(dxf_path))
        msp = doc.modelspace()

        # Configuramos el renderizado
        fig = plt.figure(figsize=(20, 20))
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_axis_off()

        ctx = RenderContext(doc)
        backend = MatplotlibBackend(ax)
        Frontend(ctx, backend).draw_layout(msp, finalize=True)

        # Guardar en buffer para OpenCV
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=DPI)
        plt.close(fig)
        buf.seek(0)

        img = cv2.imdecode(np.frombuffer(buf.read(), np.uint8), 1)
        return img
    except Exception as e:
        print(f"Error procesando {dxf_path}: {e}")
        return None


def slice_and_save(img, filename_base):
    """Corta la imagen en tiles y los guarda."""
    h, w = img.shape[:2]
    count = 0

    for y in range(0, h - TILE_SIZE, TILE_SIZE - OVERLAP):
        for x in range(0, w - TILE_SIZE, TILE_SIZE - OVERLAP):
            tile = img[y:y + TILE_SIZE, x:x + TILE_SIZE]

            # FILTRO: No guardar si la imagen es casi toda blanca o negra pura
            # Esto evita guardar pedazos de papel vacíos
            std_dev = np.std(tile)
            if std_dev < 10:  # Si hay poca variación, es un fondo liso
                continue

            out_path = OUTPUT_BG_DIR / f"bg_{filename_base}_{count}.jpg"
            cv2.imwrite(str(out_path), tile)
            count += 1
    return count


def run():
    if not INPUT_DXF_DIR.exists():
        INPUT_DXF_DIR.mkdir()
        print(f"Creada carpeta '{INPUT_DXF_DIR}'. Poné tus archivos DXF ahí.")
        return

    OUTPUT_BG_DIR.mkdir(parents=True, exist_ok=True)

    planos = list(INPUT_DXF_DIR.glob("*.dxf"))
    print(f"Encontrados {len(planos)} planos para procesar.")

    total_tiles = 0
    for plano in planos:
        print(f"Renderizando {plano.name}...")
        img = dxf_to_image(plano)
        if img is not None:
            n = slice_and_save(img, plano.stem)
            print(f"  -> Generados {n} fondos de este plano.")
            total_tiles += n

    print(f"\n✅ Proceso terminado. {total_tiles} fondos guardados en '{OUTPUT_BG_DIR}'.")


if __name__ == "__main__":
    run()