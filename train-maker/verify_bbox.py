# verify_bbox.py — Visual debugging tool for ezdxf bounding box calculation
# Loads a DXF file, computes the bounding box of its contents using
# ezdxf.bbox.extents, draws a red rectangle along those extents, and
# saves the result as [original_filename]_with_bbox.dxf.
from __future__ import annotations

import sys
from pathlib import Path

import ezdxf
import ezdxf.bbox


def _add_bbox_rectangle(msp, doc, x_min, y_min, x_max, y_max, color: int = 1):
    """Draw a rectangle on the modelspace using the best method for the DXF version.

    Parameters
    ----------
    msp : ezdxf Modelspace
    doc : ezdxf Drawing
    x_min, y_min, x_max, y_max : float
        Corners of the bounding box.
    color : int
        DXF color index (1 = red).
    """
    points = [
        (x_min, y_min),
        (x_max, y_min),
        (x_max, y_max),
        (x_min, y_max),
    ]

    dxf_version = doc.dxfversion

    # R2000 (AC1015) and later support LWPOLYLINE — lightweight and preferred.
    if dxf_versions_supports_lwpolyline(dxf_version):
        msp.add_lwpolyline(
            points,
            close=True,
            dxfattribs={"color": color},
        )
    else:
        # For R12 (AC1009) / R13 (AC1012) / R14 (AC1014) fall back to POLYLINE2D.
        try:
            msp.add_polyline2d(
                points + [points[0]],  # close manually
                dxfattribs={"color": color},
            )
        except AttributeError:
            # Ultimate fallback — generic add_polyline (very old ezdxf versions)
            msp.add_polyline3d(
                [(p[0], p[1], 0) for p in points] + [(points[0][0], points[0][1], 0)],
                dxfattribs={"color": color},
            )

    print(f"  ✅ Rectángulo dibujado: ({x_min:.4f}, {y_min:.4f}) → ({x_max:.4f}, {y_max:.4f})")


def dxf_versions_supports_lwpolyline(version: str) -> bool:
    """Return True if the DXF version string supports LWPOLYLINE (>= R2000)."""
    # ezdxf version strings: 'AC1009' (R12), 'AC1015' (R2000), etc.
    try:
        return version >= "AC1015"
    except TypeError:
        return True  # default to modern path


def verify_bbox(dxf_path: str | Path) -> Path:
    """Load a DXF, compute extents, draw a red bbox rectangle, and save.

    Returns the path to the saved file.
    """
    dxf_path = Path(dxf_path)
    if not dxf_path.exists():
        print(f"❌ Archivo no encontrado: {dxf_path}")
        sys.exit(1)

    print(f"Cargando: {dxf_path}")
    doc = ezdxf.readfile(str(dxf_path))
    msp = doc.modelspace()

    # Collect all entities for bounding box calculation
    entities = list(msp)
    if not entities:
        print("⚠ No se encontraron entidades en el ModelSpace.")
        sys.exit(1)

    print(f"  Entidades encontradas: {len(entities)}")

    bbox = ezdxf.bbox.extents(entities)

    if not bbox.has_data:
        print("❌ No se pudo calcular el bounding box (sin datos geométricos).")
        sys.exit(1)

    x_min, y_min = bbox.extmin.x, bbox.extmin.y
    x_max, y_max = bbox.extmax.x, bbox.extmax.y

    print(f"  Bounding Box calculado:")
    print(f"    Min: ({x_min:.4f}, {y_min:.4f})")
    print(f"    Max: ({x_max:.4f}, {y_max:.4f})")
    print(f"    Tamaño: {x_max - x_min:.4f} x {y_max - y_min:.4f}")

    # Draw the bounding box as a red rectangle
    _add_bbox_rectangle(msp, doc, x_min, y_min, x_max, y_max, color=1)

    # Save the modified DXF
    output_path = dxf_path.parent / f"{dxf_path.stem}_with_bbox.dxf"
    doc.saveas(str(output_path))
    print(f"\n✅ Archivo guardado: {output_path}")

    return output_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python verify_bbox.py <archivo.dxf>")
        print("  Genera un archivo [nombre]_with_bbox.dxf con un rectángulo rojo")
        print("  mostrando las extensiones calculadas por ezdxf.bbox.extents.")
        sys.exit(0)

    verify_bbox(sys.argv[1])
