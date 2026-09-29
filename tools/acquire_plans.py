"""
tools/acquire_plans.py
Autonomous acquisition and validation script for electrical CAD single-line diagrams.
Downloads open-source CAD plans from public repositories and validates their DXF structure,
entity counts, and bounding boxes using ezdxf.
"""

import os
import sys
import argparse
import urllib.request
from pathlib import Path
import ezdxf
import ezdxf.bbox

# Curated open-source electrical CAD single-line diagrams
CURATED_PLANS = [
    {
        "filename": "05_diagrama_unifilar.dxf",
        "alias": "ext_unifilar_residencial_qd.dxf",
        "url": "https://raw.githubusercontent.com/r-menegueli/portfolio-autocad-engenharia-eletrica/HEAD/02-projeto-eletrico-residencial/fonte/05_diagrama_unifilar.dxf",
        "description": "Residential distribution board QG1 (IEC / AEA 90364-7-770 compliant)"
    },
    {
        "filename": "01_diagrama_unifilar.dxf",
        "alias": "ext_unifilar_motor_inversor.dxf",
        "url": "https://raw.githubusercontent.com/r-menegueli/portfolio-autocad-engenharia-eletrica/HEAD/01-painel-motor-inversor/fonte/01_diagrama_unifilar.dxf",
        "description": "Motor drive and frequency inverter single-line diagram"
    },
    {
        "filename": "01_diagrama_unifilar_ccm.dxf",
        "alias": "ext_unifilar_ccm_industrial.dxf",
        "url": "https://raw.githubusercontent.com/r-menegueli/portfolio-autocad-engenharia-eletrica/HEAD/06-centro-controle-motores/fonte/01_diagrama_unifilar_ccm.dxf",
        "description": "Industrial Motor Control Center (CCM) single-line diagram"
    },
    {
        "filename": "03_diagrama_unifilar_ca.dxf",
        "alias": "ext_unifilar_fotovoltaico_ca.dxf",
        "url": "https://raw.githubusercontent.com/r-menegueli/portfolio-autocad-engenharia-eletrica/HEAD/08-sistema-fotovoltaico/fonte/03_diagrama_unifilar_ca.dxf",
        "description": "Photovoltaic grid-tied distributed generation AC single-line diagram"
    }
]


def download_file(url: str, dest_path: Path, timeout: int = 30) -> Path:
    """
    Downloads a file from a URL to dest_path with standard user-agent.
    """
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "CLAUDIO-AI-Acquisition/1.0 (Windows NT 10.0; Win64; x64)"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        content = response.read()
        with open(dest_path, "wb") as f:
            f.write(content)
    return dest_path


def validate_dxf(file_path: Path) -> dict:
    """
    Validates a DXF file with ezdxf, verifying header, modelspace entities,
    and bounding box calculation.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    file_size = file_path.stat().st_size
    if file_size == 0:
        raise ValueError(f"Empty DXF file: {file_path}")

    doc = ezdxf.readfile(str(file_path))
    dxf_version = doc.dxfversion
    msp = doc.modelspace()

    entity_counts = {}
    for entity in msp:
        dxftype = entity.dxftype()
        entity_counts[dxftype] = entity_counts.get(dxftype, 0) + 1

    total_entities = sum(entity_counts.values())

    bbox_info = None
    try:
        bbox = ezdxf.bbox.extents(msp)
        if bbox.has_data:
            bbox_info = {
                "extmin": (bbox.extmin.x, bbox.extmin.y, bbox.extmin.z),
                "extmax": (bbox.extmax.x, bbox.extmax.y, bbox.extmax.z),
                "width": bbox.extmax.x - bbox.extmin.x,
                "height": bbox.extmax.y - bbox.extmin.y
            }
    except Exception as e:
        bbox_info = {"error": str(e)}

    return {
        "valid": True,
        "file_size": file_size,
        "dxf_version": dxf_version,
        "total_msp_entities": total_entities,
        "entity_counts": entity_counts,
        "bbox": bbox_info,
        "blocks_count": len(doc.blocks),
        "layers_count": len(doc.layers)
    }


def acquire_plans(output_dir: Path) -> list:
    """
    Acquires all curated unifilar plans, saves them to output_dir,
    and runs full structural validation.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []

    print(f"=== CLAUDIO_AI: Autonomous CAD Plan Acquisition ===")
    print(f"Target directory: {output_dir.resolve()}\n")

    for plan in CURATED_PLANS:
        fname = plan["filename"]
        alias = plan.get("alias")
        url = plan["url"]
        desc = plan["description"]

        dest_file = output_dir / fname
        print(f"[*] Downloading {fname}...")
        print(f"    Source: {url}")
        print(f"    Description: {desc}")

        try:
            download_file(url, dest_file)
            size_kb = dest_file.stat().st_size / 1024.0
            print(f"    Downloaded successfully ({size_kb:.2f} KB)")

            # Also create alias copy if specified
            if alias:
                alias_file = output_dir / alias
                with open(dest_file, "rb") as src, open(alias_file, "wb") as dst:
                    dst.write(src.read())

            # Validate
            val = validate_dxf(dest_file)
            print(f"    [VALID] DXF Version: {val['dxf_version']}, Entities: {val['total_msp_entities']}, Blocks: {val['blocks_count']}")
            if val.get("bbox") and "width" in val["bbox"]:
                bb = val["bbox"]
                print(f"    Bounding Box: {bb['width']:.2f} x {bb['height']:.2f} CAD units")

            results.append({
                "plan": plan,
                "path": str(dest_file),
                "validation": val,
                "status": "SUCCESS"
            })
        except Exception as e:
            print(f"    [ERROR] Failed to acquire {fname}: {e}")
            results.append({
                "plan": plan,
                "path": str(dest_file),
                "error": str(e),
                "status": "FAILED"
            })
        print()

    successful = [r for r in results if r["status"] == "SUCCESS"]
    print(f"Summary: {len(successful)}/{len(CURATED_PLANS)} plans successfully acquired and validated.\n")
    return results


def main():
    parser = argparse.ArgumentParser(description="Acquire and validate open-source electrical CAD single-line plans.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="dxf/externos",
        help="Destination directory for acquired CAD plans (default: dxf/externos)"
    )
    parser.add_argument(
        "--url",
        type=str,
        default=None,
        help="Optional custom URL to download a specific DXF plan"
    )
    parser.add_argument(
        "--filename",
        type=str,
        default=None,
        help="Optional filename for custom URL download"
    )

    args = parser.parse_args()
    out_dir = Path(args.output_dir).resolve()

    if args.url:
        fname = args.filename or Path(args.url.split("?")[0]).name or "custom_plan.dxf"
        dest = out_dir / fname
        print(f"Downloading custom plan from {args.url} -> {dest}...")
        download_file(args.url, dest)
        val = validate_dxf(dest)
        print(f"Validation result: {val}")
        return

    results = acquire_plans(out_dir)
    failures = [r for r in results if r["status"] != "SUCCESS"]
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
